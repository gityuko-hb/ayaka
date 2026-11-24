"""The proposer contract: what every draft source implements.

A proposer holds no reference to the engine or scheduler. It tracks per-request committed token
history (or hidden states / caches in model-backed subclasses) and returns draft proposals when
asked. If a proposer raises during planning, the runner disables it and continues without
speculation so a request never fails.

This module imports no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Protocol

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from ayaka.configs.model import ModelConfig
    from ayaka.core.types import RequestId, TokenId

__all__ = ["EMPTY_PROPOSAL", "Proposal", "Proposer"]


@dataclass(frozen=True, slots=True)
class Proposal:
    """Draft tokens proposed for one request's next verify step.

    Attributes:
        tokens: Draft token ids ``d_1 .. d_k``; may be empty.
        confidence: Integer confidence score (e.g., n-gram match length for lookup).
        source: Short label identifying the draft origin (e.g., ``"lookup"``, ``"prediction"``,
            ``"mtp"``, ``"dflash"``, ``"token_recycling"``).
        probs: Optional per-token draft probabilities in ``[0, 1]`` for confidence-based window
            trimming (used by block drafters).
    """

    tokens: tuple[TokenId, ...]
    confidence: int = 0
    source: str = ""
    probs: tuple[float, ...] = ()


EMPTY_PROPOSAL: Final = Proposal(tokens=())


class Proposer(Protocol):
    """Protocol implemented by model-free and model-backed draft proposers."""

    name: ClassVar[str]

    def supports(self, model_config: ModelConfig) -> str | None:
        """Return ``None`` if usable with ``model_config``, otherwise a human-readable reason."""
        ...

    def on_admit(
        self,
        request_id: RequestId,
        prompt: Sequence[TokenId],
        *,
        prediction: Sequence[TokenId] = (),
        banned_tokens: Collection[TokenId] = (),
        allowed_tokens: Collection[TokenId] | None = None,
    ) -> None:
        """Initialize per-request state from the committed prompt (and optional prediction)."""
        ...

    def propose(self, request_id: RequestId, max_k: int) -> Proposal:
        """Return up to ``max_k`` draft tokens for ``request_id``."""
        ...

    def on_commit(self, request_id: RequestId, tokens: Sequence[TokenId]) -> None:
        """Append verified committed tokens to ``request_id``'s history."""
        ...

    def release(self, request_id: RequestId) -> None:
        """Free all per-request state when ``request_id`` finishes."""
        ...
