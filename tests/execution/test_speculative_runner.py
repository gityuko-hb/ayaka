"""Speculative decoding end to end on the tiny model: the ids, the KV and the draft accounting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.speculative import SpeculativeConfig
from ayaka.core.params import SamplingParams
from ayaka.core.request import TokenizedRequest
from ayaka.core.types import FinishReason, ForwardMode, RequestId
from ayaka.speculative.ngram import LookupProposer
from ayaka.speculative.policy import CostAwareController, FixedController

torch: Any = pytest.importorskip("torch")

from adversarial_proposers import (  # noqa: E402
    AlwaysRightProposer,
    AlwaysWrongProposer,
    EmptyProposer,
    OutOfVocabProposer,
    OverLongProposer,
    RaisingProposer,
    RandomProposer,
)
from ayaka.execution.runner import ModelRunner  # noqa: E402
from ayaka.kvcache.manager import KVCacheManager  # noqa: E402
from ayaka.kvcache.prefix.hash_cache import HashPrefixCache  # noqa: E402
from ayaka.kvcache.prefix.hashing import ChainRoot  # noqa: E402
from ayaka.kvcache.storage.pool import KVPool  # noqa: E402
from ayaka.kvcache.storage.spec import KVGroup  # noqa: E402
from ayaka.scheduler.scheduler import Scheduler  # noqa: E402
from tiny_model import build_tiny_stack, greedy_reference  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.types import TokenId
    from ayaka.speculative.proposer import Proposer
    from tiny_model import TinyStack

pytestmark = pytest.mark.unit


def _prompt(seed: int, length: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return [int(t) for t in torch.randint(32, 127, (length,), generator=gen)]


def _req(
    name: str,
    prompt: Sequence[int],
    *,
    max_tokens: int = 12,
    temperature: float = 0.0,
    seed: int = 7,
    stop: frozenset[int] = frozenset(),
    presence_penalty: float = 0.0,
    min_tokens: int = 0,
    prediction: Sequence[int] = (),
) -> TokenizedRequest:
    return TokenizedRequest(
        request_id=RequestId(name),
        prompt_token_ids=tuple(prompt),
        sampling=SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            stop_token_ids=stop,
            presence_penalty=presence_penalty,
            min_tokens=min_tokens,
        ),
        prediction_token_ids=tuple(prediction),
    )


def _run_stack(
    stack: TinyStack,
    requests: Sequence[TokenizedRequest],
    *,
    has_constraint: set[RequestId] | None = None,
) -> tuple[dict[str, list[TokenId]], list[int], list[tuple[str, int]]]:
    for r in requests:
        stack.scheduler.add_request(r)
        stack.runner.admit_speculative(
            r.request_id,
            r.prompt_token_ids,
            sampling=r.sampling,
            has_constraint=bool(has_constraint and r.request_id in has_constraint),
            prediction=r.prediction_token_ids,
        )
    emitted: dict[str, list[TokenId]] = {}
    verify_rows: list[int] = []
    disabled_events: list[tuple[str, int]] = []
    for _ in range(500):
        lookahead = stack.runner.plan_lookahead()
        step = stack.scheduler.schedule(lookahead)
        for ev in stack.scheduler.take_events():
            emitted.setdefault(ev.request_id, []).extend(ev.token_ids)
            if ev.finished:
                stack.runner.release_speculative(ev.request_id)
        if step is None:
            return emitted, verify_rows, disabled_events
        if step.mode is ForwardMode.VERIFY:
            verify_rows.extend(step.rows_per_request)
        out = stack.runner.execute(step)
        disabled_events.extend(out.spec_disabled)
        stack.scheduler.update(out)
        for ev in stack.scheduler.take_events():
            emitted.setdefault(ev.request_id, []).extend(ev.token_ids)
            if ev.finished:
                stack.runner.release_speculative(ev.request_id)
            elif ev.token_ids:
                # As the engine does: only committed tokens advance a grammar and the proposer.
                stack.runner.constraints.accept(ev.request_id, ev.token_ids)
                stack.runner.commit_speculative(ev.request_id, ev.token_ids)
        stack.kv.check_integrity()
    raise AssertionError("loop did not finish")


def _build_spec_stack(
    proposer: Proposer | None,
    *,
    max_k: int = 8,
    fixed_k: int | None = None,
    max_running: int = 4,
    block_size: int = 4,
    prefix_cache: bool = False,
) -> TinyStack:
    base = build_tiny_stack(max_running=max_running, block_size=block_size, max_tokens_per_step=64)
    groups = base.model.kv_specs(block_size=block_size, layout=base.model.ops.kvcache.layout())
    prefix = (
        HashPrefixCache(
            ChainRoot(model_id="tiny", dtype="fp32", codec="raw", block_size=block_size)
        )
        if prefix_cache
        else None
    )
    kv = KVCacheManager(
        groups,
        [64],
        max_requests=max_running,
        max_blocks_per_request=32,
        prefix=prefix,
    )
    group = groups[0]
    assert isinstance(group, KVGroup)
    pool = KVPool(group, 64, device="cpu")
    scheduler = Scheduler(
        base.scheduler._config,
        kv,
        max_model_len=128,
        max_chunk_tokens=64,
    )
    spec_cfg = SpeculativeConfig(
        enabled=proposer is not None,
        max_draft_tokens=max(1, max_k),
        controller="fixed" if fixed_k is not None else "cost",
        fixed_k=fixed_k if fixed_k is not None else 2,
    )
    ctrl = FixedController(fixed_k) if fixed_k is not None else CostAwareController(max_k=max_k)
    runner = ModelRunner(
        base.model,
        [pool],
        kv.table,
        max_rows=ModelRunner.required_rows(
            token_budget=64, max_running=max_running, lookahead=max_k
        ),
        device="cpu",
        speculative=spec_cfg,
        proposer=proposer,
        controller=ctrl,
    )
    from tiny_model import TinyStack

    return TinyStack(
        model=base.model,
        config=base.config,
        state=base.state,
        kv=kv,
        pool=pool,
        scheduler=scheduler,
        runner=runner,
    )


@pytest.mark.parametrize("k", [1, 2, 3, 4, 8])
def test_greedy_exactness_for_fixed_k_and_repetitive_lookup(k: int) -> None:
    plain = build_tiny_stack(max_tokens_per_step=64)
    prompt = _prompt(11, 10) * 2  # repetitive prompt so lookup triggers
    expected = greedy_reference(plain.state, plain.config, prompt, 14)

    stack = _build_spec_stack(LookupProposer(min_match=3), max_k=k, fixed_k=k)
    emitted, _, _ = _run_stack(stack, [_req("a", prompt, max_tokens=14)])
    assert emitted["a"] == expected


@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_sampled_exactness_matches_plain_decoding(temperature: float) -> None:
    prompt = _prompt(21, 12) * 2
    plain = _build_spec_stack(None, max_k=0, fixed_k=0)
    plain_out, _, _ = _run_stack(
        plain, [_req("s", prompt, max_tokens=16, temperature=temperature, seed=123)]
    )

    spec = _build_spec_stack(RandomProposer(plain.config.vocab_size, seed=5), max_k=4, fixed_k=4)
    spec_out, verify_rows, _ = _run_stack(
        spec, [_req("s", prompt, max_tokens=16, temperature=temperature, seed=123)]
    )
    assert spec_out["s"] == plain_out["s"]
    assert any(r > 1 for r in verify_rows)


def test_ragged_batch_and_all_adversarial_proposers_preserve_exact_tokens() -> None:
    p_a = _prompt(31, 12)
    p_b = _prompt(32, 15)
    plain = _build_spec_stack(None, max_k=0, fixed_k=0, max_running=4)
    plain_out, _, _ = _run_stack(
        plain,
        [
            _req("a", p_a, max_tokens=12, temperature=0.0),
            _req("b", p_b, max_tokens=12, temperature=0.8, seed=99),
        ],
    )
    oracle = {RequestId("a"): plain_out["a"], RequestId("b"): plain_out["b"]}
    vocab = plain.config.vocab_size

    proposers: list[Proposer] = [
        RandomProposer(vocab, seed=17),
        AlwaysWrongProposer(bad_token=0),
        AlwaysRightProposer(oracle),
        OutOfVocabProposer(vocab),
        OverLongProposer(),
        RaisingProposer(),
        EmptyProposer(),
    ]
    for prop in proposers:
        spec = _build_spec_stack(prop, max_k=4, fixed_k=4, max_running=4)
        spec_out, verify_rows, _ = _run_stack(
            spec,
            [
                _req("a", p_a, max_tokens=12, temperature=0.0),
                _req("b", p_b, max_tokens=12, temperature=0.8, seed=99),
            ],
        )
        if isinstance(prop, AlwaysRightProposer):
            # The proposer must really draft, or this case would only test an empty window.
            assert max(verify_rows, default=0) > 1, "always-right proposer never drafted"
        assert spec_out["a"] == plain_out["a"], f"mismatch on proposer {prop.name}"
        assert spec_out["b"] == plain_out["b"], f"mismatch on proposer {prop.name}"


def test_penalty_requests_disable_speculation_with_reason_and_keep_ragged_batch() -> None:
    p_a = _prompt(41, 10) * 2
    p_b = _prompt(42, 10) * 2
    plain = _build_spec_stack(None, max_k=0, fixed_k=0, max_running=4)
    requests = [
        _req("free", p_a, max_tokens=10),
        _req("pen", p_b, max_tokens=10, presence_penalty=0.5),
    ]
    plain_out, _, _ = _run_stack(plain, requests)

    spec = _build_spec_stack(
        RandomProposer(plain.config.vocab_size, seed=3), max_k=3, fixed_k=3, max_running=4
    )
    spec_out, verify_rows, disabled = _run_stack(spec, requests)
    assert spec_out == plain_out
    assert 1 in verify_rows
    assert any(r > 1 for r in verify_rows)
    reasons = {r for r, c in disabled if c > 0}
    assert "penalty" in reasons


_GRAMMAR_PATTERN = "[abc]{18}"


def _admit_grammar(stack: TinyStack, request_id: str) -> Any:
    """Compile ``_GRAMMAR_PATTERN`` for the byte tokenizer and admit it for ``request_id``."""
    from ayaka.core.constraints import ConstraintKind, StructuredParams
    from ayaka.sampling.structured.compiler import GrammarCompiler
    from tokenizer_fakes import EOS, FakeTokenizer

    tok = FakeTokenizer(stream=False)
    compiler = GrammarCompiler(tok, vocab_size=stack.config.vocab_size, eos_token_ids=(EOS,))
    params = StructuredParams(kind=ConstraintKind.REGEX, pattern=_GRAMMAR_PATTERN)
    matcher = compiler.matcher(compiler.compile(params), params)
    stack.runner.constraints.admit(RequestId(request_id), matcher)
    return compiler


@pytest.mark.parametrize("drafter", ["lookup", "always_wrong"])
def test_grammar_requests_speculate_and_emit_what_plain_constrained_decoding_emits(
    drafter: str,
) -> None:
    pytest.importorskip("llguidance")
    prompt = [ord(ch) for ch in "abcabcabcabcabcabcabc"]
    request = [_req("gram", prompt, max_tokens=12)]
    plain = _build_spec_stack(None, max_k=0, fixed_k=0)
    plain_compiler = _admit_grammar(plain, "gram")
    try:
        plain_out, _, _ = _run_stack(plain, request, has_constraint={RequestId("gram")})
    finally:
        plain_compiler.close()

    proposer: Proposer = (
        LookupProposer(min_match=3) if drafter == "lookup" else AlwaysWrongProposer(bad_token=1)
    )
    spec = _build_spec_stack(proposer, max_k=4, fixed_k=4)
    spec_compiler = _admit_grammar(spec, "gram")
    try:
        spec_out, verify_rows, disabled = _run_stack(
            spec, request, has_constraint={RequestId("gram")}
        )
    finally:
        spec_compiler.close()

    assert spec_out == plain_out
    assert all(chr(token) in "abc" for token in spec_out["gram"])
    reasons = {r for r, c in disabled if c > 0}
    if drafter == "lookup":
        # The lookup copies the prompt's "abc" pattern, which the grammar allows: it drafts.
        assert any(r > 1 for r in verify_rows)
    else:
        # Every draft is a token the grammar refuses: it is cut before verification.
        assert "grammar" in reasons
        assert all(r == 1 for r in verify_rows)


def test_a_grammar_cuts_a_proposal_before_its_first_refused_token() -> None:
    pytest.importorskip("llguidance")
    from ayaka.execution.speculation import SpeculationState
    from ayaka.speculative.proposer import Proposal

    stack = _build_spec_stack(None, max_k=0, fixed_k=0)
    compiler = _admit_grammar(stack, "gram")
    try:
        state = SpeculationState(stack.runner.constraints)
        a, b, z = ord("a"), ord("b"), ord("z")
        cut = state._grammar_prefix(  # pyright: ignore[reportPrivateUsage]
            RequestId("gram"), Proposal(tokens=(a, b, a, z, a), probs=(0.9,) * 5)
        )
        assert cut.tokens == (a, b, a)
        assert cut.probs == (0.9,) * 3
        whole = Proposal(tokens=(a, b, b))
        assert state._grammar_prefix(RequestId("gram"), whole) is whole  # pyright: ignore[reportPrivateUsage]
        # The committed grammar state is unchanged: "z" is still refused, "a" still allowed.
        assert stack.runner.constraints.validate_tokens(RequestId("gram"), (a,))
        assert not stack.runner.constraints.validate_tokens(RequestId("gram"), (z,))
    finally:
        compiler.close()


def test_a_verify_window_counts_as_masked_when_any_request_has_a_grammar() -> None:
    pytest.importorskip("llguidance")
    from types import SimpleNamespace

    from ayaka.execution.speculation import SpeculationState

    stack = _build_spec_stack(None, max_k=0, fixed_k=0)
    state = SpeculationState(stack.runner.constraints)
    step: Any = SimpleNamespace(request_ids=(RequestId("free"), RequestId("gram")))
    drafts = ((ord("a"),), (ord("a"), ord("b")))
    # No request has a grammar: a captured verify graph may replay the window.
    assert not state.window_masked(step, drafts)
    compiler = _admit_grammar(stack, "gram")
    try:
        # One request's rows are masked: the window must run eagerly, where rows get masks.
        assert state.window_masked(step, drafts)
    finally:
        compiler.close()


def test_kv_and_prefix_cache_contain_only_committed_tokens_after_rejections() -> None:
    prompt = _prompt(51, 8)
    plain = _build_spec_stack(None, max_k=0, fixed_k=0, block_size=4, prefix_cache=True)
    plain_out, _, _ = _run_stack(plain, [_req("x", prompt, max_tokens=12)])

    # AlwaysWrongProposer forces rejections on every round while writing draft KV past committed.
    spec = _build_spec_stack(
        AlwaysWrongProposer(bad_token=1), max_k=3, fixed_k=3, block_size=4, prefix_cache=True
    )
    spec_out, _, _ = _run_stack(spec, [_req("x", prompt, max_tokens=12)])
    assert spec_out["x"] == plain_out["x"]

    # Every cached digest in prefix cache must correspond to committed tokens only.
    # Run a second request sharing prompt + first 4 generated tokens: must hit prefix cache cleanly.
    follow_prompt = list(prompt) + list(plain_out["x"][:4])
    follow_req = _req("y", follow_prompt, max_tokens=4)
    spec.scheduler.add_request(follow_req)
    step = spec.scheduler.schedule()
    assert step is not None
    req_obj = spec.scheduler.get(RequestId("y"))
    assert req_obj is not None
    assert req_obj.cached_tokens >= 8  # prefix cache attached full blocks cleanly


def test_stop_token_and_cancel_inside_verify_window() -> None:
    prompt = _prompt(61, 10)
    plain = build_tiny_stack(max_tokens_per_step=64)
    expected = greedy_reference(plain.state, plain.config, prompt, 10)
    # Pick a stop token that isn't already expected[0] (prefill's first emitted token).
    stop = next(t for t in expected[1:] if t != expected[0])
    stop_pos = expected.index(stop)

    oracle = {RequestId("s"): expected}
    spec = _build_spec_stack(AlwaysRightProposer(oracle), max_k=6, fixed_k=6)
    out, verify_rows, _ = _run_stack(
        spec, [_req("s", prompt, max_tokens=10, stop=frozenset({stop}))]
    )
    assert max(verify_rows, default=0) > 1, "always-right proposer never drafted"
    assert out["s"] == expected[: stop_pos + 1]

    # Cancellation while a step is in flight
    spec2 = _build_spec_stack(AlwaysRightProposer(oracle), max_k=4, fixed_k=4)
    req = _req("c", prompt, max_tokens=10)
    spec2.scheduler.add_request(req)
    spec2.runner.admit_speculative(req.request_id, req.prompt_token_ids, sampling=req.sampling)
    while (step := spec2.scheduler.schedule(spec2.runner.plan_lookahead())) is not None:
        if step.mode.is_decode_like:
            assert spec2.scheduler.abort(req.request_id, FinishReason.CANCELLED)
            spec2.scheduler.update(spec2.runner.execute(step))
            events = spec2.scheduler.take_events()
            assert any(ev.finish_reason is FinishReason.CANCELLED for ev in events)
            spec2.runner.release_speculative(req.request_id)
            break
        spec2.scheduler.update(spec2.runner.execute(step))
        for ev in spec2.scheduler.take_events():
            if ev.token_ids:
                spec2.runner.commit_speculative(ev.request_id, ev.token_ids)
    spec2.kv.check_integrity()
