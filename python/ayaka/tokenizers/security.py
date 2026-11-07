"""Keeping user text from forging chat-template control tokens.

A chat template wraps messages in control tokens such as ``<|im_start|>``. If a user
message contains that literal string and the rendered prompt is tokenized with
special-token parsing on, the text becomes a *real* control token and the user can
fake turns or close their own message early. :class:`SpecialTokenGuard` neutralizes
or rejects such text **before** templating::

    message content --> SpecialTokenGuard.sanitize --> chat template --> encode
                          reject  -> SpecialTokenInContent
                          escape  -> a zero-width space (U+200B) is inserted inside the token
                          allow   -> unchanged

The service applies the guard for every adapter, driven by
``TokenizerConfig.special_token_policy`` (default ``reject``), so the protection does
not depend on which tokenizer is loaded. It covers every string in the messages and
tool definitions of ``ChatInput`` and ``TemplateInput``, including roles. It does not
cover:

* Plain completion text (``TextInput``) and already-rendered prompts
  (``RenderedInput``): pre-templated prompts legitimately contain control tokens.
* The ``chat_template`` override, which is template code.
* Added tokens that the tokenizer does not flag as special (for example a model's
  ``<think>`` or ``<tool_response>`` markers). The guard uses the same definition of
  "special" as ``skip_special_tokens``.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ayaka.tokenizers.errors import SpecialTokenInContent

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["SpecialTokenGuard", "SpecialTokenPolicy"]

_ZERO_WIDTH_SPACE = "\u200b"


class SpecialTokenPolicy(StrEnum):
    """What to do with literal special tokens found in user-provided chat content."""

    REJECT = "reject"
    """Raise :class:`SpecialTokenInContent` (default; a literal control token is an injection)."""

    ESCAPE = "escape"
    """Insert a zero-width space after the first character so it no longer matches."""

    ALLOW = "allow"
    """Leave the text untouched; templates are then open to token hijacking."""


class SpecialTokenGuard:
    """Applies a :class:`SpecialTokenPolicy` to nested chat content.

    Strings are checked directly; lists and dicts (multimodal payloads) are walked
    recursively; every other value is returned as is.

    Attributes:
        policy: The policy this guard enforces.
    """

    __slots__ = ("_pattern", "policy")

    def __init__(
        self,
        special_tokens: Iterable[str],
        policy: SpecialTokenPolicy = SpecialTokenPolicy.REJECT,
    ) -> None:
        """Build a guard for the given literal special-token strings.

        Longer tokens are matched first so a token is never split by a shorter
        prefix of itself. Empty strings are ignored.
        """
        self.policy = policy
        literals = sorted({token for token in special_tokens if token}, key=len, reverse=True)
        self._pattern = re.compile("|".join(map(re.escape, literals))) if literals else None

    def sanitize(self, content: Any) -> Any:
        """Return ``content`` made safe under the policy.

        Raises:
            SpecialTokenInContent: Under ``REJECT``, if any string contains a
                special token.
        """
        if self._pattern is None or self.policy is SpecialTokenPolicy.ALLOW:
            return content
        if isinstance(content, str):
            return self._sanitize_text(content, self._pattern)
        if isinstance(content, list):
            return [self.sanitize(item) for item in content]  # pyright: ignore[reportUnknownVariableType]
        if isinstance(content, dict):
            return {key: self.sanitize(value) for key, value in content.items()}  # pyright: ignore[reportUnknownVariableType]
        return content

    def _sanitize_text(self, text: str, pattern: re.Pattern[str]) -> str:
        if self.policy is SpecialTokenPolicy.REJECT:
            match = pattern.search(text)
            if match is not None:
                raise SpecialTokenInContent(
                    f"content contains literal special token {match.group(0)!r} "
                    f"at index {match.start()}; use the 'escape' policy to neutralize it "
                    "or 'allow' to accept the risk of template injection"
                )
            return text
        return pattern.sub(
            lambda m: m.group(0)[:1] + _ZERO_WIDTH_SPACE + m.group(0)[1:],
            text,
        )
