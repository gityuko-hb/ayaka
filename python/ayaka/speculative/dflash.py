"""DFlash diffusion block drafter proposer (L5, protocol-only over draft model)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol

import torch

from ayaka.speculative.ngram import LookupProposer
from ayaka.speculative.proposer import EMPTY_PROPOSAL, Proposal

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from ayaka.configs.model import ModelConfig
    from ayaka.core.types import RequestId, TokenId

__all__ = ["DFlashDraftProtocol", "DFlashEmbedProtocol", "DFlashProposer"]


class DFlashDraftProtocol(Protocol):
    """Structural contract for a DFlash block draft model."""

    def compute_context_kv(
        self, target_features: torch.Tensor, positions: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]: ...

    def forward_block(
        self,
        noise_embeds: torch.Tensor,
        positions: torch.Tensor,
        ctx_ks: Sequence[torch.Tensor],
        ctx_vs: Sequence[torch.Tensor],
    ) -> torch.Tensor: ...


class DFlashEmbedProtocol(Protocol):
    """Structural contract for shared target token embeddings."""

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor: ...


@dataclass(slots=True)
class _DFlashRequestState:
    ctx_len: int = 0
    last_token: TokenId | None = None
    disabled: bool = False
    k_ctx: list[torch.Tensor] = field(default_factory=lambda: [])
    v_ctx: list[torch.Tensor] = field(default_factory=lambda: [])


class DFlashProposer:
    """Lossless DFlash block speculative proposer with cached per-request context KV."""

    name: ClassVar[str] = "dflash"

    def __init__(
        self,
        draft_model: DFlashDraftProtocol,
        embed_tokens: DFlashEmbedProtocol,
        compute_logits: Callable[[torch.Tensor], torch.Tensor],
        *,
        block_size: int,
        mask_token_id: int,
        target_layer_ids: Sequence[int],
        target_hidden_size: int,
        target_num_layers: int,
        device: torch.device | str,
        confidence_threshold: float = 0.0,
        context_cap: int = 0,
        lookup_override_match: int = 0,
    ) -> None:
        self._draft = draft_model
        self._embed = embed_tokens
        self._compute_logits = compute_logits
        self._block_size = block_size
        self._mask_token_id = mask_token_id
        self._target_layer_ids = tuple(target_layer_ids)
        self._target_hidden_size = target_hidden_size
        self._target_num_layers = target_num_layers
        self._device = torch.device(device)
        self._confidence_threshold = confidence_threshold
        self._context_cap = context_cap
        self._lookup_override = lookup_override_match
        self._lookup = (
            LookupProposer(min_match=max(1, lookup_override_match))
            if lookup_override_match > 0
            else None
        )
        self._states: dict[RequestId, _DFlashRequestState] = {}

    @property
    def target_layer_ids(self) -> tuple[int, ...]:
        """Target decoder layer indices required by this DFlash checkpoint."""
        return self._target_layer_ids

    def supports(self, model_config: ModelConfig) -> str | None:
        """Verify target hidden width and layer depth match the loaded DFlash checkpoint."""
        if model_config.hidden_size != self._target_hidden_size:
            return (
                f"DFlash expects target hidden_size={self._target_hidden_size}, "
                f"got {model_config.hidden_size}"
            )
        if max(self._target_layer_ids, default=-1) >= model_config.num_layers:
            return (
                f"DFlash target_layer_ids={self._target_layer_ids} exceeds target "
                f"num_layers={model_config.num_layers}"
            )
        return None

    def on_admit(
        self,
        request_id: RequestId,
        prompt: Sequence[TokenId],
        *,
        prediction: Sequence[TokenId] = (),
        allowed_tokens: Collection[TokenId] | None = None,
    ) -> None:
        self._states[request_id] = _DFlashRequestState(disabled=allowed_tokens is not None)
        if self._lookup is not None:
            self._lookup.on_admit(
                request_id,
                prompt,
                prediction=prediction,
                allowed_tokens=allowed_tokens,
            )

    def observe_features(
        self,
        request_ids: Sequence[RequestId],
        start_positions: Sequence[int],
        rows_per_request: Sequence[int],
        emitted_tokens: Sequence[tuple[TokenId, ...]],
        features: torch.Tensor,
    ) -> None:
        """Append newly committed target layer features to each request's DFlash context KV."""
        offset = 0
        for idx, request_id in enumerate(request_ids):
            total_rows = rows_per_request[idx]
            row_start = offset
            offset += total_rows
            state = self._states.get(request_id)
            if state is None or state.disabled or total_rows <= 0:
                continue
            emitted = emitted_tokens[idx] if idx < len(emitted_tokens) else ()
            start_pos = start_positions[idx]
            # For prefill chunks (which emit either 0 on intermediate or 1 token on final chunk),
            # every chunk row belongs to committed prompt positions. For verify/decode, only the
            # first len(emitted) rows were accepted/committed!
            if not emitted:
                kept = total_rows
            elif start_pos == state.ctx_len and total_rows > len(emitted) and len(emitted) == 1:
                # Could be either a multi-token prefill final chunk OR a verify step rejecting all
                # drafts (kept=1). Distinguish cleanly by checking state.last_token: before prefill
                # ends, state.last_token is None; once prefill has emitted, state.last_token is set!
                kept = total_rows if state.last_token is None else 1
            else:
                kept = min(len(emitted), total_rows)
            if kept <= 0:
                continue

            new_feats = features[row_start : row_start + kept]
            positions = torch.arange(
                start_pos, start_pos + kept, dtype=torch.int64, device=self._device
            )
            ks, vs = self._draft.compute_context_kv(new_feats, positions)
            if not state.k_ctx:
                state.k_ctx = ks
                state.v_ctx = vs
            else:
                cap = self._context_cap
                for layer_idx in range(len(ks)):
                    k_cat = torch.cat([state.k_ctx[layer_idx], ks[layer_idx]], dim=1)
                    v_cat = torch.cat([state.v_ctx[layer_idx], vs[layer_idx]], dim=1)
                    if cap > 0 and k_cat.shape[1] > cap:
                        k_cat = k_cat[:, -cap:].contiguous()
                        v_cat = v_cat[:, -cap:].contiguous()
                    state.k_ctx[layer_idx] = k_cat
                    state.v_ctx[layer_idx] = v_cat
            state.ctx_len = start_pos + kept
            if emitted:
                state.last_token = emitted[-1]

    def propose(self, request_id: RequestId, max_k: int) -> Proposal:
        if max_k <= 0:
            return EMPTY_PROPOSAL
        state = self._states.get(request_id)
        if state is None or state.disabled or state.last_token is None or not state.k_ctx:
            return EMPTY_PROPOSAL
        if self._lookup is not None:
            look = self._lookup.propose(request_id, max_k)
            if look.confidence >= self._lookup_override and look.tokens:
                return look

        block_tokens = [state.last_token, *([self._mask_token_id] * (self._block_size - 1))]
        blk_tensor = torch.tensor(block_tokens, dtype=torch.int64, device=self._device)
        positions = torch.arange(
            state.ctx_len,
            state.ctx_len + self._block_size,
            dtype=torch.int64,
            device=self._device,
        )
        with torch.inference_mode():
            hid = self._draft.forward_block(
                self._embed(blk_tensor), positions, state.k_ctx, state.v_ctx
            )
            logits = self._compute_logits(hid[1:])
            limit = min(max_k, int(logits.shape[0]))
            logits = logits[:limit]
            top_ids = torch.argmax(logits, dim=-1).cpu()
            draft_ids = tuple(int(top_ids[i].item()) for i in range(limit))

            if self._confidence_threshold > 0.0:
                probs_t = torch.softmax(logits.float(), dim=-1).max(dim=-1).values.cpu()
                probs = tuple(float(probs_t[i].item()) for i in range(limit))
                kept = _cut_by_confidence(probs, self._confidence_threshold)
                draft_ids = draft_ids[:kept]
                probs = probs[:kept]
                confidence = round(sum(probs))
            else:
                probs = ()
                confidence = len(draft_ids)

        if not draft_ids:
            return EMPTY_PROPOSAL
        return Proposal(
            tokens=draft_ids,
            confidence=confidence,
            source=self.name,
            probs=probs,
        )

    def on_commit(self, request_id: RequestId, tokens: Sequence[TokenId]) -> None:
        if self._lookup is not None:
            self._lookup.on_commit(request_id, tokens)

    def release(self, request_id: RequestId) -> None:
        self._states.pop(request_id, None)
        if self._lookup is not None:
            self._lookup.release(request_id)


def _cut_by_confidence(probs: Sequence[float], threshold: float) -> int:
    """Keep leading draft tokens whose cumulative max-prob product stays >= ``threshold``."""
    cum = 1.0
    kept = 0
    for idx, prob in enumerate(probs):
        cum *= prob
        if cum < threshold and idx > 0:
            break
        kept = idx + 1
    return kept
