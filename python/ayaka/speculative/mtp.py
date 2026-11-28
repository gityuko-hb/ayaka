"""Multi-token-prediction (MTP) drafter (e.g., XiaomiMiMo/MiMo-7B-RL layer 36)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

import torch

from ayaka.core.types import ForwardMode, RequestId, TokenId
from ayaka.layers.context import ForwardContext
from ayaka.speculative.ngram import LookupProposer
from ayaka.speculative.proposer import EMPTY_PROPOSAL, Proposal
from ayaka.utils.gc_control import paused_gc

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from ayaka.configs.model import ModelConfig
    from ayaka.ops.attention import AttentionMetadata, AttentionOps
    from ayaka.ops.kvcache import KVBuffers

__all__ = ["MTPEmbedProtocol", "MTPLayerProtocol", "MTPProposer"]


class MTPLayerProtocol(Protocol):
    """Callable contract of a single MTP decoder layer (e.g. ``MiMoMTPLayer``)."""

    def __call__(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        context: ForwardContext,
    ) -> torch.Tensor: ...


class MTPEmbedProtocol(Protocol):
    """Callable contract of the shared target token embedding."""

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor: ...


class _Depth2Capable(Protocol):
    token_layernorm: Callable[[torch.Tensor], torch.Tensor]
    hidden_layernorm: Callable[[torch.Tensor], torch.Tensor]
    input_proj: Callable[[torch.Tensor], torch.Tensor]
    post_attention_layernorm: Callable[[torch.Tensor], torch.Tensor]
    mlp: Callable[[torch.Tensor], torch.Tensor]
    final_layernorm: Callable[[torch.Tensor], torch.Tensor]


@dataclass(slots=True)
class _MTPRequestState:
    prompt: list[TokenId]
    draft_tokens: tuple[TokenId, ...] = ()
    last_hidden: torch.Tensor | None = None
    last_pos: int = 0
    disabled: bool = False


class MTPProposer:
    """Lossless 1-step (and experimental 2-step) multi-token-prediction drafter."""

    name: ClassVar[str] = "mtp"

    def __init__(
        self,
        layer: MTPLayerProtocol,
        embed_tokens: MTPEmbedProtocol,
        compute_logits: Callable[[torch.Tensor], torch.Tensor],
        *,
        attention: AttentionOps,
        kv_buffers: Mapping[int, KVBuffers],
        mtp_layer_id: int,
        block_size: int,
        vocab_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        depth: int = 1,
        vocab_limit: int = 0,
        lookup_override_match: int = 0,
        lm_head_weight: torch.Tensor | None = None,
    ) -> None:
        self._layer = layer
        self._embed = embed_tokens
        self._compute_logits = compute_logits
        self._attention = attention
        self._kv_buffers: dict[int, KVBuffers] = {mtp_layer_id: kv_buffers[mtp_layer_id]}
        self._layer_id = mtp_layer_id
        self._block_size = block_size
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._dtype = dtype
        self._device = torch.device(device)
        self._depth = max(1, depth)
        self._vocab_limit = vocab_limit
        self._lm_head_weight = (
            lm_head_weight[:vocab_limit]
            if (lm_head_weight is not None and 0 < vocab_limit < vocab_size)
            else None
        )
        self._lookup_override = lookup_override_match
        self._lookup = (
            LookupProposer(min_match=max(1, lookup_override_match))
            if lookup_override_match > 0
            else None
        )
        self._states: dict[RequestId, _MTPRequestState] = {}
        # Optional CUDA graphs for committed verify/decode updates (by padded row bucket).
        self._graph_buckets: tuple[int, ...] = ()
        self._graph_meta: dict[int, AttentionMetadata] = {}
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._graph_hidden: torch.Tensor | None = None
        self._graph_tokens: torch.Tensor | None = None
        self._graph_argmax: torch.Tensor | None = None
        self._graph_max_requests = 0
        self._graph_max_blocks = 0
        self.graph_replays = 0

    def supports(self, model_config: ModelConfig) -> str | None:
        """MTP needs at least one checkpoint multi-token-prediction layer."""
        if model_config.num_nextn_predict_layers < 1:
            return f"{model_config.architecture} checkpoint has num_nextn_predict_layers=0"
        return None

    def capture_graphs(
        self,
        *,
        max_requests: int,
        max_blocks: int,
        buckets: Sequence[int] = (2, 4, 8, 16),
        pool: Any = None,
    ) -> int:
        """Capture MTP layer + draft head CUDA graphs for fast replay after verify steps."""
        if self._device.type != "cuda" or not buckets:
            return 0
        usable = tuple(sorted({b for b in buckets if b >= 2}))
        if not usable:
            return 0
        max_b = usable[-1]
        torch.cuda.synchronize(self._device)
        before = torch.cuda.memory_reserved(self._device)
        self._graph_max_requests = max_requests
        self._graph_max_blocks = max_blocks
        self._graph_hidden = torch.zeros(
            (max_b, self._hidden_size), dtype=self._dtype, device=self._device
        )
        self._graph_tokens = torch.zeros(max_b, dtype=torch.int64, device=self._device)
        self._graph_argmax = torch.zeros(max_b, dtype=torch.int64, device=self._device)
        graph_pool = pool or torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream(self._device)
        zeros_rows = torch.zeros(max_requests, dtype=torch.int32)
        zeros_len = torch.zeros(max_requests, dtype=torch.int32)
        zeros_tbl = torch.zeros((max_requests, max_blocks), dtype=torch.int32)

        kept: list[int] = []
        for bucket in reversed(usable):
            try:
                meta = self._attention.verify_graph_metadata(
                    bucket, max_requests, max_blocks, self._block_size, self._device
                )
                self._graph_meta[bucket] = meta
                self._graphs[bucket] = self._capture_bucket(
                    bucket, meta, graph_pool, side, zeros_rows, zeros_len, zeros_tbl
                )
                kept.append(bucket)
            except torch.cuda.OutOfMemoryError:
                self._graph_meta.pop(bucket, None)
                self._graphs.pop(bucket, None)
                torch.cuda.empty_cache()
        self._graph_buckets = tuple(sorted(kept))
        torch.cuda.synchronize(self._device)
        after = torch.cuda.memory_reserved(self._device)
        return max(after - before, 0)

    def _capture_bucket(
        self,
        bucket: int,
        meta: AttentionMetadata,
        graph_pool: Any,
        side: torch.cuda.Stream,
        zeros_rows: torch.Tensor,
        zeros_len: torch.Tensor,
        zeros_tbl: torch.Tensor,
    ) -> torch.cuda.CUDAGraph:
        if self._graph_hidden is None or self._graph_tokens is None or self._graph_argmax is None:
            raise RuntimeError("MTP graph buffers must be initialized before capture")
        rows = zeros_rows.clone()
        rows[0] = bucket
        self._attention.update_verify_graph_metadata(meta, rows, zeros_len, zeros_tbl)
        hid = self._graph_hidden[:bucket]
        tok = self._graph_tokens[:bucket]
        out_arg = self._graph_argmax[:bucket]
        ctx = ForwardContext(attention=meta, kv_buffers=self._kv_buffers)

        def body() -> None:
            mtp_h = self._layer(self._embed(tok), hid, ctx)
            logits = self._draft_logits(mtp_h)
            torch.argmax(logits, dim=-1, out=out_arg)

        side.wait_stream(torch.cuda.current_stream(self._device))
        with torch.inference_mode(), torch.cuda.stream(side):
            body()
            body()
        torch.cuda.current_stream(self._device).wait_stream(side)
        torch.cuda.synchronize(self._device)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), paused_gc(), torch.cuda.graph(graph, pool=graph_pool):
            body()
        return graph

    def on_admit(
        self,
        request_id: RequestId,
        prompt: Sequence[TokenId],
        *,
        prediction: Sequence[TokenId] = (),
        allowed_tokens: Collection[TokenId] | None = None,
    ) -> None:
        self._states[request_id] = _MTPRequestState(
            prompt=list(prompt),
            disabled=allowed_tokens is not None,
        )
        if self._lookup is not None:
            self._lookup.on_admit(
                request_id,
                prompt,
                prediction=prediction,
                allowed_tokens=allowed_tokens,
            )

    def propose(self, request_id: RequestId, max_k: int) -> Proposal:
        if max_k <= 0:
            return EMPTY_PROPOSAL
        state = self._states.get(request_id)
        if state is None or state.disabled:
            return EMPTY_PROPOSAL
        if self._lookup is not None:
            look = self._lookup.propose(request_id, max_k)
            if look.confidence >= self._lookup_override and look.tokens:
                return look
        if not state.draft_tokens:
            return EMPTY_PROPOSAL
        toks = state.draft_tokens[:max_k]
        return Proposal(tokens=toks, confidence=len(toks), source=self.name)

    def on_commit(self, request_id: RequestId, tokens: Sequence[TokenId]) -> None:
        if self._lookup is not None:
            self._lookup.on_commit(request_id, tokens)

    def release(self, request_id: RequestId) -> None:
        self._states.pop(request_id, None)
        if self._lookup is not None:
            self._lookup.release(request_id)

    def observe_prefill(
        self,
        request_ids: Sequence[RequestId],
        start_positions: Sequence[int],
        chunk_lengths: Sequence[int],
        emitted_tokens: Sequence[tuple[TokenId, ...]],
        hidden: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> None:
        """Advance layer-36 KV for prefill chunk rows and sample draft(s) when chunk ends prompt."""
        rows_hidden: list[torch.Tensor] = []
        rows_toks: list[TokenId] = []
        active_indices: list[int] = []
        q_lens: list[int] = []
        kv_lens: list[int] = []
        draft_targets: list[tuple[RequestId, int]] = []

        offset = 0
        row_cursor = 0
        for idx, request_id in enumerate(request_ids):
            length = chunk_lengths[idx]
            row_start = offset
            offset += length
            state = self._states.get(request_id)
            if state is None or state.disabled or length <= 0:
                continue
            start_pos = start_positions[idx]
            emitted = emitted_tokens[idx] if idx < len(emitted_tokens) else ()
            # Pair position p hidden state with token p + 1 (prompt[p+1] or first emitted token).
            paired: list[TokenId] = []
            for r in range(length):
                next_idx = start_pos + r + 1
                if next_idx < len(state.prompt):
                    paired.append(state.prompt[next_idx])
                elif emitted:
                    paired.append(emitted[0])
                else:
                    break
            used = len(paired)
            if used == 0:
                continue
            rows_hidden.append(hidden[row_start : row_start + used])
            rows_toks.extend(paired)
            active_indices.append(idx)
            q_lens.append(used)
            kv_lens.append(start_pos + used)
            row_cursor += used
            if emitted:
                draft_targets.append((request_id, row_cursor - 1))

        if not rows_hidden:
            return
        all_hid = torch.cat(rows_hidden, dim=0)
        all_tok = torch.tensor(rows_toks, dtype=torch.int64, device=self._device)
        bt = block_tables[active_indices]
        meta = self._attention.plan(
            ForwardMode.PREFILL,
            torch.tensor(q_lens, dtype=torch.int32, device=self._device),
            torch.tensor(kv_lens, dtype=torch.int32, device=self._device),
            bt,
            self._block_size,
        )
        mtp_out = self._layer(self._embed(all_tok), all_hid, ForwardContext(meta, self._kv_buffers))
        if not draft_targets:
            return
        pick = torch.tensor(
            [pos for _, pos in draft_targets], dtype=torch.int64, device=self._device
        )
        chosen_h = mtp_out.index_select(0, pick)
        first = self._argmax(chosen_h)
        for j, (request_id, _) in enumerate(draft_targets):
            state = self._states.get(request_id)
            if state is not None:
                tok1 = first[j]
                if self._depth > 1:
                    tok2 = self._step_depth2(chosen_h[j : j + 1], tok1, request_id, bt, j)
                    state.draft_tokens = (tok1, tok2)
                else:
                    state.draft_tokens = (tok1,)

    def observe_decode_like(
        self,
        request_ids: Sequence[RequestId],
        start_positions: Sequence[int],
        rows_per_request: Sequence[int],
        emitted_tokens: Sequence[tuple[TokenId, ...]],
        hidden: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> None:
        """Advance layer-36 KV over committed tokens and predict next draft token(s)."""
        rows_hidden: list[torch.Tensor] = []
        rows_toks: list[TokenId] = []
        active_indices: list[int] = []
        q_lens: list[int] = []
        kv_lens: list[int] = []
        last_indices: list[tuple[RequestId, int]] = []

        offset = 0
        total_kept = 0
        for idx, request_id in enumerate(request_ids):
            total_rows = rows_per_request[idx]
            row_start = offset
            offset += total_rows
            state = self._states.get(request_id)
            emitted = emitted_tokens[idx] if idx < len(emitted_tokens) else ()
            if state is None or state.disabled or not emitted:
                continue
            kept = min(len(emitted), total_rows)
            if kept <= 0:
                continue
            rows_hidden.append(hidden[row_start : row_start + kept])
            rows_toks.extend(emitted[:kept])
            active_indices.append(idx)
            q_lens.append(kept)
            kv_lens.append(start_positions[idx] + kept)
            total_kept += kept
            last_indices.append((request_id, total_kept - 1))

        if not rows_hidden:
            return
        all_hid = torch.cat(rows_hidden, dim=0)
        all_tok = torch.tensor(rows_toks, dtype=torch.int64, device=self._device)
        bt = block_tables[active_indices]

        bucket = self._graph_bucket_for(total_kept, len(active_indices))
        if (
            bucket is not None
            and self._depth == 1
            and self._graph_hidden is not None
            and self._graph_tokens is not None
            and self._graph_argmax is not None
        ):
            self._graph_hidden[:total_kept].copy_(all_hid)
            self._graph_tokens[:total_kept].copy_(all_tok)
            host_rows = torch.zeros(self._graph_max_requests, dtype=torch.int32)
            host_lens = torch.zeros(self._graph_max_requests, dtype=torch.int32)
            host_tbl = torch.zeros(
                (self._graph_max_requests, self._graph_max_blocks), dtype=torch.int32
            )
            host_rows[: len(q_lens)] = torch.tensor(q_lens, dtype=torch.int32)
            if bucket > total_kept and len(q_lens) < self._graph_max_requests:
                host_rows[len(q_lens)] = bucket - total_kept
            host_lens[: len(kv_lens)] = torch.tensor(kv_lens, dtype=torch.int32)
            cols = min(int(bt.shape[1]), self._graph_max_blocks)
            host_tbl[: len(active_indices), :cols].copy_(bt[:, :cols].cpu())
            self._attention.update_verify_graph_metadata(
                self._graph_meta[bucket], host_rows, host_lens, host_tbl
            )
            self._graphs[bucket].replay()
            self.graph_replays += 1
            cpu_preds = self._graph_argmax[:total_kept].cpu()
            all_preds = [int(cpu_preds[i].item()) for i in range(total_kept)]
            for request_id, pick_idx in last_indices:
                state = self._states.get(request_id)
                if state is not None:
                    state.draft_tokens = (all_preds[pick_idx],)
            return

        mode = ForwardMode.DECODE if all(q == 1 for q in q_lens) else ForwardMode.MIXED
        meta = self._attention.plan(
            mode,
            torch.tensor(q_lens, dtype=torch.int32, device=self._device),
            torch.tensor(kv_lens, dtype=torch.int32, device=self._device),
            bt,
            self._block_size,
        )
        mtp_out = self._layer(self._embed(all_tok), all_hid, ForwardContext(meta, self._kv_buffers))
        pick = torch.tensor(
            [pos for _, pos in last_indices], dtype=torch.int64, device=self._device
        )
        chosen_h = mtp_out.index_select(0, pick)
        first = self._argmax(chosen_h)
        for j, (request_id, _) in enumerate(last_indices):
            state = self._states.get(request_id)
            if state is not None:
                tok1 = first[j]
                if self._depth > 1:
                    tok2 = self._step_depth2(chosen_h[j : j + 1], tok1, request_id, bt, j)
                    state.draft_tokens = (tok1, tok2)
                else:
                    state.draft_tokens = (tok1,)

    def _graph_bucket_for(self, total_rows: int, active_reqs: int) -> int | None:
        for bucket in self._graph_buckets:
            if bucket >= total_rows and (
                active_reqs < self._graph_max_requests or bucket == total_rows
            ):
                return bucket
        return None

    def _draft_logits(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        if self._lm_head_weight is not None:
            return torch.nn.functional.linear(hidden_rows, self._lm_head_weight)
        return self._compute_logits(hidden_rows)

    def _argmax(self, hidden_rows: torch.Tensor) -> list[TokenId]:
        logits = self._draft_logits(hidden_rows)
        cpu_ids = torch.argmax(logits, dim=-1).cpu()
        return [int(cpu_ids[i].item()) for i in range(int(cpu_ids.shape[0]))]

    def _step_depth2(
        self,
        h_prev: torch.Tensor,
        tok1: TokenId,
        request_id: RequestId,
        block_tables: torch.Tensor,
        req_index: int,
    ) -> TokenId:
        """Experimental depth-2 rollout without mutating committed KV cache."""
        del request_id, block_tables, req_index
        tok_t = torch.tensor([tok1], dtype=torch.int64, device=self._device)
        emb = self._embed(tok_t)
        layer = self._layer
        if (
            hasattr(layer, "token_layernorm")
            and hasattr(layer, "hidden_layernorm")
            and hasattr(layer, "input_proj")
            and hasattr(layer, "post_attention_layernorm")
            and hasattr(layer, "mlp")
            and hasattr(layer, "final_layernorm")
        ):
            mtp = cast("_Depth2Capable", layer)
            tok_n = mtp.token_layernorm(emb)
            hid_n = mtp.hidden_layernorm(h_prev)
            x = mtp.input_proj(torch.cat([hid_n, tok_n], dim=-1))
            residual = x
            h = mtp.post_attention_layernorm(residual)
            out = mtp.final_layernorm(mtp.mlp(h) + residual)
            return self._argmax(out)[0]
        return tok1
