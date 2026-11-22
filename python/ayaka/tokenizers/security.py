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
tool definitions of ``ChatInput``, ``TemplateInput`` and ``TemplateData``, including roles.
For ``TemplateData`` the one walk that checks the strings also turns the values into what a
JSON round trip would give back (:func:`plain_json`), so the template sees the same thing either
way. It does not cover:

* Plain completion text (``TextInput``) and already-rendered prompts
  (``RenderedInput``): pre-templated prompts legitimately contain control tokens.
* The ``chat_template`` override, which is template code.
* Added tokens that the tokenizer does not flag as special (for example a model's
  ``<think>`` or ``<tool_response>`` markers). The guard uses the same definition of
  "special" as ``skip_special_tokens``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

from ayaka.tokenizers.errors import SpecialTokenInContent

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = ["SpecialTokenGuard", "SpecialTokenPolicy", "plain_json"]

_ZERO_WIDTH_SPACE = "\u200b"


def _json_key(key: object) -> str:
    """A mapping key as ``json.dumps`` writes it."""
    if type(key) is str:
        return key
    if isinstance(key, str):
        return str.__str__(key)
    if key is None or isinstance(key, bool):
        return {None: "null", True: "true", False: "false"}[key]
    if isinstance(key, int):
        return int.__repr__(key)
    if isinstance(key, float):
        return {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}.get(
            float.__repr__(key), float.__repr__(key)
        )
    raise TypeError(f"keys must be str, int, float, bool or None, not {type(key).__name__}")


def plain_json(value: object, text: Callable[[str], str] | None = None) -> Any:
    """``value`` as ``json.loads(json.dumps(value))`` gives it back, in one walk.

    Mappings become dicts with string keys, lists and tuples become lists, numbers and strings
    keep their value as their exact built-in type. Each string goes through ``text`` when given.

    Raises:
        TypeError: For a value JSON cannot hold, as ``json.dumps`` would.
    """
    if isinstance(value, str):
        plain = value if type(value) is str else str.__str__(value)
        return plain if text is None else text(plain)
    if value is None or type(value) in (bool, int, float):
        return value
    if isinstance(value, Mapping):
        items = cast("Mapping[object, object]", value).items()
        return {_json_key(key): plain_json(item, text) for key, item in items}
    if isinstance(value, list | tuple):
        return [plain_json(item, text) for item in cast("list[object]", value)]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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

    def sanitize_json(self, value: object) -> Any:
        """``value`` made safe under the policy and turned into plain JSON values, in one walk.

        The result equals :meth:`sanitize` of ``json.loads(json.dumps(value))``.

        Raises:
            SpecialTokenInContent: Under ``REJECT``, if any string contains a special token.
            TypeError: For a value JSON cannot hold.
        """
        pattern = self._pattern
        if pattern is None or self.policy is SpecialTokenPolicy.ALLOW:
            return plain_json(value)
        return plain_json(value, lambda text: self._sanitize_text(text, pattern))

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
