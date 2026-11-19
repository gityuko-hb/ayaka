"""Benchmark prompt template rendering and tokenization on long coding-agent conversations.

Uses the recorded 23-tool Claude Code fixture (`004-anthropic.json`) expanded cleanly to target
prompt sizes (default 13,600 and 50,000 tokens), timing template serialization + rendering and
native encoding separately, and comparing against measured cold TTFT when provided.

    uv run python benchmarks/tokenizers/encode_long.py --model unsloth/Llama-3.2-1B-Instruct \
        --offline --out out.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import timeit
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "_support"))

from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.entrypoints.anthropic.protocol import parse_messages_request
from ayaka.entrypoints.conversation import Conversation, ToolCall, Turn
from ayaka.entrypoints.http import JsonObject
from ayaka.tokenizers.contracts import EncodeOptions
from ayaka.tokenizers.factory import DefaultTokenizerFactory
from ayaka.tokenizers.service import TokenizerService

if TYPE_CHECKING:
    from ayaka.tokenizers.contracts import LoadedTokenizer
    from ayaka.tokenizers.ports import TokenizerLike

SCHEMA: Final = "ayaka.encode-long/1"
_FIXTURE: Final = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "entrypoints"
    / "fixtures"
    / "claude-code-2.1.327"
    / "004-anthropic.json"
)
_TURN_BODY: Final = (
    "Inspecting `python/ayaka/entrypoints/serving.py` and verifying how requests enter "
    "tokenization, structured output constraints, streaming encoders, and shutdown handling "
    "under concurrent load across multiple clients. Iteration details and diagnostics: "
    "read=420ms, encode=18ms, cache_prefix=12800, kv_blocks=800."
)


def _load_base_conversation() -> Conversation:
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    parsed = parse_messages_request(JsonObject(payload["body"]))
    return parsed.conversation


def _make_turn_pair(index: int) -> tuple[Turn, Turn, Turn]:
    call_id = f"call_{index}"
    return (
        Turn(
            role="assistant",
            text=f"Step {index}: checking repository state before running tests. {_TURN_BODY}",
            tool_calls=(
                ToolCall(
                    id=call_id,
                    name="Bash",
                    arguments={"command": f"uv run pytest -q tests/unit/test_{index}.py"},
                ),
            ),
        ),
        Turn(
            role="tool",
            text=f"Exit code 0 for step {index}. {_TURN_BODY}",
            tool_call_id=call_id,
            name="Bash",
        ),
        Turn(role="user", text=f"Continue with verification step {index + 1}. {_TURN_BODY}"),
    )


def _token_count(adapter: TokenizerLike, conversation: Conversation) -> int:
    tmpl = conversation.template_input()
    rendered = adapter.apply_chat_template(
        json.loads(tmpl.messages_json),
        tools=json.loads(tmpl.tools_json),
        add_generation_prompt=True,
    )
    return len(adapter.encode(rendered, add_special_tokens=False))


def build_conversation_for_tokens(
    adapter: TokenizerLike, target_tokens: int
) -> tuple[Conversation, int]:
    """Expand the recorded multi-tool fixture with synthetic turns to reach ``target_tokens``."""
    base = _load_base_conversation()
    current_turns = list(base.turns)
    probe = dataclasses.replace(base, turns=tuple(current_turns))
    if _token_count(adapter, probe) >= target_tokens:
        return probe, _token_count(adapter, probe)

    turn_batch: list[Turn] = []
    idx = 0
    # Measure growth per 3-turn cycle cleanly first, then refine step by step.
    sample_turns = [*current_turns, *_make_turn_pair(0)]
    base_len = _token_count(adapter, probe)
    cycle_len = max(
        1,
        _token_count(adapter, dataclasses.replace(base, turns=tuple(sample_turns))) - base_len,
    )
    approx_cycles = max(0, (target_tokens - base_len) // cycle_len - 1)
    for idx in range(approx_cycles):
        turn_batch.extend(_make_turn_pair(idx))
    current_turns.extend(turn_batch)

    next_idx = approx_cycles
    while True:
        candidate = dataclasses.replace(base, turns=tuple(current_turns))
        count = _token_count(adapter, candidate)
        if count >= target_tokens:
            return candidate, count
        current_turns.extend(_make_turn_pair(next_idx))
        next_idx += 1


def measure_prompt(
    service: TokenizerService,
    adapter: TokenizerLike,
    conversation: Conversation,
    *,
    token_count: int,
    rounds: int,
    cold_ttft_ms: float | None,
) -> dict[str, Any]:
    """Time template serialization, Jinja render and native encode separately over ``rounds``."""
    render_ms_rounds: list[float] = []
    encode_ms_rounds: list[float] = []
    serialize_ms_rounds: list[float] = []
    service_uncached_ms_rounds: list[float] = []

    # Warm up template compilation and caches cleanly.
    warmup_tmpl = conversation.template_input()
    adapter.apply_chat_template(
        json.loads(warmup_tmpl.messages_json),
        tools=json.loads(warmup_tmpl.tools_json),
        add_generation_prompt=True,
    )

    for r in range(rounds):
        t0 = timeit.default_timer()
        tmpl = conversation.template_input()
        t1 = timeit.default_timer()
        messages_obj = json.loads(tmpl.messages_json)
        tools_obj = json.loads(tmpl.tools_json)
        rendered = adapter.apply_chat_template(
            messages_obj, tools=tools_obj, add_generation_prompt=True
        )
        t2 = timeit.default_timer()
        ids = adapter.encode(rendered, add_special_tokens=False)
        t3 = timeit.default_timer()
        if len(ids) != token_count:
            raise RuntimeError(f"expected {token_count} encoded tokens, got {len(ids)}")

        # End-to-end uncached TokenizerService path (unique namespace bypasses session hit).
        t4 = timeit.default_timer()
        result = service.encode(
            tmpl,
            EncodeOptions(
                max_output_tokens=1,
                truncate=False,
                cache_namespace=f"bench-{token_count}-{r}",
            ),
        )
        t5 = timeit.default_timer()
        if len(result.token_ids) < token_count:
            raise RuntimeError("TokenizerService returned fewer tokens than expected")

        serialize_ms_rounds.append((t1 - t0) * 1e3)
        render_ms_rounds.append((t2 - t1) * 1e3)
        encode_ms_rounds.append((t3 - t2) * 1e3)
        service_uncached_ms_rounds.append((t5 - t4) * 1e3)

    med_serialize = statistics.median(serialize_ms_rounds)
    med_render = statistics.median(render_ms_rounds)
    med_encode = statistics.median(encode_ms_rounds)
    med_render_encode = statistics.median(
        [r + e for r, e in zip(render_ms_rounds, encode_ms_rounds, strict=True)]
    )
    med_service = statistics.median(service_uncached_ms_rounds)
    row: dict[str, Any] = {
        "tokens": token_count,
        "rounds": {
            "serialize_ms": serialize_ms_rounds,
            "render_ms": render_ms_rounds,
            "encode_ms": encode_ms_rounds,
            "service_uncached_ms": service_uncached_ms_rounds,
        },
        "median": {
            "serialize_ms": med_serialize,
            "render_ms": med_render,
            "encode_ms": med_encode,
            "render_plus_encode_ms": med_render_encode,
            "service_uncached_ms": med_service,
        },
    }
    if cold_ttft_ms is not None and cold_ttft_ms > 0:
        row["cold_ttft_ms"] = cold_ttft_ms
        row["render_encode_share_of_ttft"] = med_render_encode / cold_ttft_ms
        row["service_share_of_ttft"] = med_service / cold_ttft_ms
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render & encode latency for long agent prompts.")
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--fake", action="store_true", help="use byte-level fake tokenizer")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--targets", type=int, nargs="+", default=[13600, 50000])
    parser.add_argument(
        "--prefill-ms-per-token",
        type=float,
        default=None,
        help="measured prefill ms/token used to compute TTFT share for each target length",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    from _machine import machine

    max_len = max(max(args.targets) + 1024, 65536)
    loaded: LoadedTokenizer
    if args.fake:
        from tokenizer_fakes import FakeTokenizer, loaded_for

        loaded = loaded_for(FakeTokenizer())
        model_label = "fake-byte"
    else:
        loaded = DefaultTokenizerFactory.from_config(
            TokenizerConfig(args.model, offline=args.offline)
        )
        model_label = args.model

    if loaded.tokenizer is None:
        raise RuntimeError("tokenizer adapter is disabled")
    adapter = loaded.tokenizer
    results: list[dict[str, Any]] = []
    service = TokenizerService(
        TokenizerConfig(model_label, mode="hf"),
        max_model_len=max_len,
        model_vocab_size=max(adapter.vocab_size, adapter.max_token_id + 1, 1024),
        loaded=loaded,
    )
    try:
        for target in args.targets:
            conv, actual_tokens = build_conversation_for_tokens(adapter, target)
            ttft_ms = (
                actual_tokens * args.prefill_ms_per_token
                if args.prefill_ms_per_token is not None
                else None
            )
            row = measure_prompt(
                service,
                adapter,
                conv,
                token_count=actual_tokens,
                rounds=args.rounds,
                cold_ttft_ms=ttft_ms,
            )
            results.append(row)
            med = row["median"]
            share_note = (
                f", share={row['service_share_of_ttft']:.2%} of {ttft_ms:.1f}ms"
                if "service_share_of_ttft" in row and ttft_ms is not None
                else ""
            )
            print(
                f"tokens={actual_tokens}: serialize={med['serialize_ms']:.2f}ms "
                f"render={med['render_ms']:.2f}ms encode={med['encode_ms']:.2f}ms "
                f"service={med['service_uncached_ms']:.2f}ms{share_note}"
            )
    finally:
        service.close()

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "machine": machine(),
        "settings": {
            "model": model_label,
            "rounds": args.rounds,
            "targets": args.targets,
            "prefill_ms_per_token": args.prefill_ms_per_token,
        },
        "prompts": results,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
