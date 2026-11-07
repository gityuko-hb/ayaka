"""Errors that callers of the tokenizer layer are expected to catch by type.

Everything else raised here is a plain ``ValueError``/``TypeError``/``RuntimeError``
that signals a bug in the caller. These two are part of the serving contract: the
front end maps ``TokenizerOverloaded`` to "try again later" and
``SpecialTokenInContent`` to "bad request".
"""

from __future__ import annotations

__all__ = ["SpecialTokenInContent", "TokenizerOverloaded"]


class TokenizerOverloaded(RuntimeError):  # noqa: N818  # serving contract name
    """Queued plus running input exceeds a configured budget; retry later."""


class SpecialTokenInContent(ValueError):  # noqa: N818  # serving contract name
    """User or tool content contains a literal special token and the policy is ``reject``."""
