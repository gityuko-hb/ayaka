"""Adversarial draft proposers for testing lossless speculative invariants."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, ClassVar

from ayaka.speculative.proposer import EMPTY_PROPOSAL, Proposal

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from ayaka.configs.model import ModelConfig
    from ayaka.core.types import RequestId, TokenId

__all__ = [
    "AlwaysRightProposer",
    "AlwaysWrongProposer",
    "EmptyProposer",
    "OutOfVocabProposer",
    "OverLongProposer",
    "RaisingProposer",
    "RandomProposer",
]


class _BaseTestProposer:
    name: ClassVar[str] = "adversarial"

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
        del request_id, prompt, prediction, banned_tokens, allowed_tokens

    def propose(self, request_id: RequestId, max_k: int) -> Proposal:
        del request_id, max_k
        return EMPTY_PROPOSAL

    def on_commit(self, request_id: RequestId, tokens: Sequence[TokenId]) -> None:
        del request_id, tokens

    def release(self, request_id: RequestId) -> None:
        del request_id


class RandomProposer(_BaseTestProposer):
    name: ClassVar[str] = "random"

    def __init__(self, vocab_size: int, *, seed: int = 42) -> None:
        self._vocab = vocab_size
        self._rng = random.Random(seed)  # noqa: S311

    def propose(self, request_id: RequestId, max_k: int) -> Proposal:
        del request_id
        if max_k <= 0:
            return EMPTY_PROPOSAL
        k = self._rng.randint(0, max_k)
        return Proposal(
            tokens=tuple(self._rng.randrange(self._vocab) for _ in range(k)),
            confidence=k,
            source=self.name,
        )


class AlwaysWrongProposer(_BaseTestProposer):
    name: ClassVar[str] = "always_wrong"

    def __init__(self, bad_token: TokenId = 0) -> None:
        self._bad = bad_token

    def propose(self, request_id: RequestId, max_k: int) -> Proposal:
        del request_id
        return Proposal(tokens=(self._bad,) * max_k, confidence=max_k, source=self.name)


class AlwaysRightProposer(_BaseTestProposer):
    """Echoes an oracle token sequence passed at construction per request."""

    name: ClassVar[str] = "always_right"

    def __init__(self, oracle: Mapping[RequestId, Sequence[TokenId]]) -> None:
        self._oracle = oracle
        self._pos: dict[RequestId, int] = {}

    def on_admit(
        self,
        request_id: RequestId,
        prompt: Sequence[TokenId],
        *,
        prediction: Sequence[TokenId] = (),
        banned_tokens: Collection[TokenId] = (),
        allowed_tokens: Collection[TokenId] | None = None,
    ) -> None:
        del prompt, prediction, banned_tokens, allowed_tokens
        self._pos[request_id] = 0

    def propose(self, request_id: RequestId, max_k: int) -> Proposal:
        seq = self._oracle.get(request_id, ())
        pos = self._pos.get(request_id, 0)
        # First emitted token is pending (index 0); drafts predict indices pos+1 .. pos+max_k.
        tokens = tuple(seq[pos + 1 : pos + 1 + max_k])
        return Proposal(tokens=tokens, confidence=len(tokens), source=self.name)

    def on_commit(self, request_id: RequestId, tokens: Sequence[TokenId]) -> None:
        self._pos[request_id] = self._pos.get(request_id, 0) + len(tokens)

    def release(self, request_id: RequestId) -> None:
        self._pos.pop(request_id, None)


class OutOfVocabProposer(_BaseTestProposer):
    name: ClassVar[str] = "oov"

    def __init__(self, vocab_size: int) -> None:
        self._oov = vocab_size + 999

    def propose(self, request_id: RequestId, max_k: int) -> Proposal:
        del request_id
        if max_k <= 0:
            return EMPTY_PROPOSAL
        return Proposal(tokens=(self._oov, -5)[:max_k], confidence=3, source=self.name)


class OverLongProposer(_BaseTestProposer):
    name: ClassVar[str] = "over_long"

    def propose(self, request_id: RequestId, max_k: int) -> Proposal:
        del request_id
        return Proposal(tokens=(1,) * (max_k + 10), confidence=max_k + 10, source=self.name)


class RaisingProposer(_BaseTestProposer):
    name: ClassVar[str] = "raising"

    def propose(self, request_id: RequestId, max_k: int) -> Proposal:
        del request_id, max_k
        raise RuntimeError("boom from adversarial proposer")


class EmptyProposer(_BaseTestProposer):
    name: ClassVar[str] = "empty"

    def propose(self, request_id: RequestId, max_k: int) -> Proposal:
        del request_id, max_k
        return EMPTY_PROPOSAL
