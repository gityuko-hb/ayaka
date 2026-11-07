from __future__ import annotations

import pytest

from ayaka.tokenizers.errors import SpecialTokenInContent
from ayaka.tokenizers.security import SpecialTokenGuard, SpecialTokenPolicy

ZWSP = chr(0x200B)
TOKENS = ("<|im_end|>", "<|im_start|>")


@pytest.mark.unit
class TestSpecialTokenGuard:
    def test_default_policy_rejects(self) -> None:
        guard = SpecialTokenGuard(TOKENS)

        assert guard.policy is SpecialTokenPolicy.REJECT
        with pytest.raises(SpecialTokenInContent, match=r"<\|im_end\|>"):
            guard.sanitize("ok <|im_end|> forged")

    def test_clean_text_passes_unchanged(self) -> None:
        assert SpecialTokenGuard(TOKENS).sanitize("plain text") == "plain text"

    def test_escape_breaks_the_token_but_keeps_the_text(self) -> None:
        guard = SpecialTokenGuard(TOKENS, SpecialTokenPolicy.ESCAPE)

        out = guard.sanitize("a <|im_end|> b")

        assert "<|im_end|>" not in out
        assert out == f"a <{ZWSP}|im_end|> b"

    def test_allow_leaves_text_untouched(self) -> None:
        guard = SpecialTokenGuard(TOKENS, SpecialTokenPolicy.ALLOW)

        assert guard.sanitize("<|im_end|>") == "<|im_end|>"

    def test_walks_nested_lists_and_dicts(self) -> None:
        guard = SpecialTokenGuard(TOKENS, SpecialTokenPolicy.ESCAPE)
        payload = [{"type": "text", "text": "<|im_start|>"}, "<|im_end|>", 7, None]

        out = guard.sanitize(payload)

        assert out[0]["text"] == f"<{ZWSP}|im_start|>"
        assert out[1] == f"<{ZWSP}|im_end|>"
        assert out[2:] == [7, None]

    def test_reject_finds_tokens_inside_nested_content(self) -> None:
        guard = SpecialTokenGuard(TOKENS)

        with pytest.raises(SpecialTokenInContent):
            guard.sanitize([{"text": "x <|im_start|>"}])

    def test_longest_token_wins(self) -> None:
        guard = SpecialTokenGuard(("<a>", "<a>b"), SpecialTokenPolicy.ESCAPE)

        assert guard.sanitize("<a>b") == f"<{ZWSP}a>b"

    def test_no_tokens_means_nothing_to_guard(self) -> None:
        assert SpecialTokenGuard(()).sanitize("<|im_end|>") == "<|im_end|>"
        assert SpecialTokenGuard(("",)).sanitize("anything") == "anything"
