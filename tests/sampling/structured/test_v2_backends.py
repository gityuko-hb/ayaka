"""Tests for Structured Output version 2 backends, rollback, and guided speculation."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from ayaka.core.constraints import (
    ConstraintKind,
    StructuralTagItem,
    StructuralTagSpec,
    StructuredParams,
    ToolCallSyntax,
    ToolSchema,
)
from ayaka.core.params import SamplingParams
from ayaka.core.types import RequestId
from ayaka.sampling.structured.compiler import GrammarCompiler
from tokenizer_fakes import EOS, SPECIALS, FakeTokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.sampling.structured.backend import StructuredOutputBackendName

pytestmark = pytest.mark.unit


class _MiniTokenizer(FakeTokenizer):
    """Byte-level test tokenizer with single-token tool call markers."""

    def __init__(self) -> None:
        super().__init__(stream=False)
        self._extras: dict[str, int] = {"<tool_call>": 260, "</tool_call>": 261}
        self._table: list[bytes] = [
            *(bytes([i]) for i in range(256)),
            *(s.encode("utf-8") for s in SPECIALS),
            b"<tool_call>",
            b"</tool_call>",
        ]

    @property
    def vocab_size(self) -> int:
        return len(self._table)

    @property
    def all_special_ids(self) -> frozenset[int]:
        return frozenset({*SPECIALS.values(), 260, 261})

    @property
    def all_special_tokens(self) -> tuple[str, ...]:
        return (*SPECIALS, "<tool_call>", "</tool_call>")

    def grammar_token_bytes(self) -> tuple[bytes, ...]:
        return tuple(self._table)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        if text in self._extras:
            return [self._extras[text]]
        ids: list[int] = []
        pos = 0
        while pos < len(text):
            matched = next((m for m in self._extras if text.startswith(m, pos)), None)
            if matched is not None:
                ids.append(self._extras[matched])
                pos += len(matched)
                continue
            chunk_end = pos + 1
            while chunk_end < len(text) and not any(
                text.startswith(m, chunk_end) for m in self._extras
            ):
                chunk_end += 1
            ids.extend(super().encode(text[pos:chunk_end], add_special_tokens=add_special_tokens))
            pos = chunk_end
        return ids

    def decode(
        self,
        ids: Sequence[int] | int,
        *,
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = True,
    ) -> str:
        rev = {260: "<tool_call>", 261: "</tool_call>"}
        seq = [ids] if isinstance(ids, int) else ids
        out: list[str] = []
        for tid in seq:
            if tid in rev:
                if not skip_special_tokens:
                    out.append(rev[tid])
            else:
                out.append(
                    super().decode(
                        [tid],
                        skip_special_tokens=skip_special_tokens,
                        spaces_between_special_tokens=spaces_between_special_tokens,
                    )
                )
        return "".join(out)


def _all_kinds_cases() -> list[tuple[StructuredParams, list[str]]]:
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    return [
        (StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema=schema), ['{"ok":true}']),
        (StructuredParams(kind=ConstraintKind.JSON_OBJECT), ['{"a":"b"}']),
        (StructuredParams(kind=ConstraintKind.REGEX, pattern="[ab]+[12]+"), ["aba12"]),
        (StructuredParams(kind=ConstraintKind.CHOICE, choices=("aba", "c12")), ["aba"]),
        (
            StructuredParams(kind=ConstraintKind.GRAMMAR, grammar='start: "a" ("1" | "2") "b"\n'),
            ["a2b"],
        ),
        (
            StructuredParams(
                kind=ConstraintKind.STRUCTURAL_TAG,
                structural_tag=StructuralTagSpec(
                    tags=(
                        StructuralTagItem(
                            begin="<tool_call><function=f>",
                            grammar='start: "<parameter=x>" ("1"|"2"|"3")+ "</parameter>"',
                            end="</function></tool_call>",
                        ),
                    ),
                    triggers=("<tool_call>",),
                    at_least_one=True,
                ),
            ),
            ["<tool_call><function=f><parameter=x>12</parameter></function></tool_call>"],
        ),
        (
            StructuredParams(
                kind=ConstraintKind.TOOL_CALL,
                tools=(ToolSchema("ok", schema),),
                syntax=ToolCallSyntax(
                    open="<tool_call>", close="</tool_call>", open_token=260, close_token=261
                ),
                tool_choice="required",
                strict=True,
            ),
            ['<tool_call>{"name":"ok","arguments":{"ok":true}}</tool_call>'],
        ),
    ]


@pytest.mark.parametrize("backend_name", ["llguidance", "xgrammar"])
def test_all_seven_constraint_kinds_and_rollback(backend_name: str) -> None:
    pytest.importorskip(backend_name)
    tok = _MiniTokenizer()
    compiler = GrammarCompiler(
        tok,
        vocab_size=tok.vocab_size,
        eos_token_ids=(EOS,),
        backend=cast("StructuredOutputBackendName", backend_name),
    )
    try:
        for params, samples in _all_kinds_cases():
            cid = compiler.compile(params)
            matcher = compiler.matcher(cid)
            token_seq = [tid for s in samples for tid in tok.encode(s, add_special_tokens=False)]
            assert matcher.validate_tokens(token_seq)
            mask_before = matcher.mask()
            window = matcher.window_masks(token_seq[:-1])
            assert len(window) == len(token_seq)
            if mask_before is not None and window[0] is not None:
                assert np.array_equal(window[0], mask_before)
            matcher.accept(token_seq)
            if len(token_seq) > 1:
                matcher.rollback(1)
                mask_after_rb = matcher.mask()
                if window[-1] is not None and mask_after_rb is not None:
                    assert np.array_equal(mask_after_rb, window[-1])
    finally:
        compiler.close()


def test_guided_speculative_verification_matches_step_by_step_decoding() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("llguidance")
    import torch

    from ayaka.sampling.logits import LogitRow, build_controls
    from ayaka.sampling.structured.runtime import ConstraintRuntime

    tok = _MiniTokenizer()
    compiler = GrammarCompiler(tok, vocab_size=tok.vocab_size, eos_token_ids=(EOS,))
    params = StructuredParams(kind=ConstraintKind.REGEX, pattern="aba12")
    cid = compiler.compile(params)
    target_ids = tok.encode("aba12", add_special_tokens=False)

    # Non-speculative single-step masks
    seq_matcher = compiler.matcher(cid)
    seq_tokens: list[int] = []
    for expected in target_ids:
        runtime = ConstraintRuntime()
        runtime.admit(RequestId("r1"), seq_matcher)
        controls = build_controls(
            [LogitRow(RequestId("r1"), SamplingParams(temperature=0.0), len(seq_tokens))],
            vocab_size=tok.vocab_size,
            eos_token_ids=(EOS,),
            penalties=None,
            device="cpu",
            constraints=runtime,
        )
        assert controls is not None
        assert controls.bitmask is not None
        logits = torch.full((tok.vocab_size,), -100.0)
        # Prefer an illegal ASCII character loudly unless bitmask masks it out.
        logits[ord("{")] = 50.0
        logits[expected] = 10.0
        words = controls.bitmask[0].tolist()
        allowed = torch.tensor(
            [(words[i // 32] >> (i % 32)) & 1 for i in range(tok.vocab_size)],
            dtype=torch.bool,
        )
        drawn = int(logits.masked_fill(~allowed, float("-inf")).argmax().item())
        seq_tokens.append(drawn)
        seq_matcher.accept([drawn])

    # Speculative single verification window covering all draft positions in one shot
    spec_matcher = compiler.matcher(cid)
    drafts = target_ids[:-1]
    spec_runtime = ConstraintRuntime()
    spec_runtime.admit(RequestId("r1"), spec_matcher)
    rows = [
        LogitRow(RequestId("r1"), SamplingParams(temperature=0.0), j)
        for j in range(len(drafts) + 1)
    ]
    spec_controls = build_controls(
        rows,
        vocab_size=tok.vocab_size,
        eos_token_ids=(EOS,),
        penalties=None,
        device="cpu",
        constraints=spec_runtime,
        draft_tokens={RequestId("r1"): drafts},
    )
    assert spec_controls is not None
    assert spec_controls.bitmask is not None
    assert spec_controls.bitmask.shape[0] == len(target_ids)
    spec_tokens: list[int] = []
    for row_idx, expected in enumerate(target_ids):
        logits = torch.full((tok.vocab_size,), -100.0)
        logits[ord("{")] = 50.0
        logits[expected] = 10.0
        words = spec_controls.bitmask[row_idx].tolist()
        allowed = torch.tensor(
            [(words[i // 32] >> (i % 32)) & 1 for i in range(tok.vocab_size)],
            dtype=torch.bool,
        )
        spec_tokens.append(int(logits.masked_fill(~allowed, float("-inf")).argmax().item()))

    assert spec_tokens == seq_tokens == target_ids
    compiler.close()


def test_import_ayaka_does_not_load_structured_or_torch() -> None:
    code = (
        "import sys, ayaka\n"
        "bad = [m for m in ('torch', 'llguidance', 'xgrammar') if m in sys.modules]\n"
        "assert not bad, f'unexpected eager imports: {bad}'\n"
    )
    subprocess.check_call([sys.executable, "-c", code])  # noqa: S603  # fixed python probe
