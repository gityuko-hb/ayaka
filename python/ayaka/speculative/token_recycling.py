"""Token recycling drafter (experimental model-free baseline for #127).

Maintains a bounded per-request transition table ``adj[token] -> recent next tokens`` updated
from committed tokens and proposes a greedy linear chain of transitions starting from the last
committed token. Kept only if the benchmark in #127 shows it wins over plain decoding and lookup.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from ayaka.speculative.proposer import EMPTY_PROPOSAL, Proposal

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from ayaka.configs.model import ModelConfig
    from ayaka.core.types import RequestId, TokenId

__all__ = ["TokenRecyclingProposer"]


@dataclass(slots=True)
class _RecyclingState:
    adj: dict[int, deque[int]] = field(default_factory=dict[int, deque[int]])
    last_token: int | None = None
    banned: frozenset[TokenId] = field(default_factory=frozenset[int])
    allowed: frozenset[TokenId] | None = None

    def observe(self, tokens: Sequence[TokenId], *, width: int) -> None:
        for tok in tokens:
            if self.last_token is not None:
                bucket = self.adj.get(self.last_token)
                if bucket is None:
                    bucket = deque[int](maxlen=width)
                    self.adj[self.last_token] = bucket
                if tok in bucket:
                    bucket.remove(tok)
                bucket.append(tok)
            self.last_token = tok


class TokenRecyclingProposer:
    """Experimental model-free transition-table drafter."""

    name: ClassVar[str] = "token_recycling"

    def __init__(self, *, width: int = 8) -> None:
        self._width = width
        self._requests: dict[RequestId, _RecyclingState] = {}

    def supports(self, model_config: ModelConfig) -> str | None:
        del model_config
        return None

    def on_admit(
        self,
        request_id: RequestId,
        prompt: Sequence[TokenId],
        *,
        prediction: Sequence[TokenId] = (),
        banned_tokens: Collection[TokenId] = (),
        allowed_tokens: Collection[TokenId] | None = None,
    ) -> None:
        del prediction
        state = _RecyclingState(
            banned=frozenset(banned_tokens),
            allowed=frozenset(allowed_tokens) if allowed_tokens is not None else None,
        )
        state.observe(prompt, width=self._width)
        self._requests[request_id] = state

    def propose(self, request_id: RequestId, max_k: int) -> Proposal:
        state = self._requests.get(request_id)
        if state is None or state.last_token is None or max_k <= 0:
            return EMPTY_PROPOSAL
        drafts: list[TokenId] = []
        cur = state.last_token
        seen_edges: set[tuple[int, int]] = set()
        for _ in range(max_k):
            bucket = state.adj.get(cur)
            if not bucket:
                break
            nxt = bucket[-1]
            if (cur, nxt) in seen_edges:
                break
            if nxt in state.banned or (state.allowed is not None and nxt not in state.allowed):
                break
            seen_edges.add((cur, nxt))
            drafts.append(nxt)
            cur = nxt
        if not drafts:
            return EMPTY_PROPOSAL
        return Proposal(tokens=tuple(drafts), confidence=len(drafts), source=self.name)

    def on_commit(self, request_id: RequestId, tokens: Sequence[TokenId]) -> None:
        state = self._requests.get(request_id)
        if state is None or not tokens:
            return
        state.observe(tokens, width=self._width)

    def release(self, request_id: RequestId) -> None:
        self._requests.pop(request_id, None)
