"""FeatureSink: capturing intermediate post-layer residual sums during Llama/Qwen/MiMo forward."""

from __future__ import annotations

from typing import Any

import pytest

from ayaka.configs.device import DeviceConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.core.types import ForwardMode
from ayaka.layers.context import FeatureSink, ForwardContext
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.kvcache.storage.pool import KVPool  # noqa: E402
from ayaka.model_loader.loader import load_model  # noqa: E402
from ayaka.ops.backend import load_backend  # noqa: E402
from llama_support import llama_config, llama_state, write_llama  # noqa: E402

pytestmark = pytest.mark.unit


def _make_context(
    model: Any, seq_len: int, sink: FeatureSink | None = None
) -> tuple[Any, ForwardContext]:
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    [group] = model.kv_specs(block_size=4, layout=ops.kvcache.layout())
    pool = KVPool(group, num_blocks=8, device="cpu")
    kv_buffers = {lid: pool.buffers(lid) for lid in group.layer_ids}
    block_table = torch.zeros((1, 4), dtype=torch.int32)
    blocks_used = (seq_len + 3) // 4
    block_table[0, :blocks_used] = torch.arange(1, 1 + blocks_used, dtype=torch.int32)
    meta = ops.attention.plan(
        ForwardMode.PREFILL,
        torch.tensor([seq_len], dtype=torch.int32),
        torch.tensor([seq_len], dtype=torch.int32),
        block_table,
        4,
    )
    return pool, ForwardContext(attention=meta, kv_buffers=kv_buffers, feature_sink=sink)


def test_feature_sink_captures_exact_post_layer_residuals(tmp_path: Any) -> None:
    config = llama_config(num_layers=4)
    state = llama_state(config, seed=11)
    directory = write_llama(tmp_path / "llama", state, config)
    model: Any = load_model(ModelSourceConfig(str(directory)), DeviceConfig(dtype=DType.FP32))

    tokens = torch.tensor([1, 5, 9, 13, 17], dtype=torch.long)
    _, ctx_plain = _make_context(model, len(tokens), None)
    with torch.inference_mode():
        plain = model(tokens, ctx_plain)

    sink = FeatureSink(layer_ids=frozenset({0, 2, 3}))
    _, ctx_sink = _make_context(model, len(tokens), sink)
    with torch.inference_mode():
        with_sink = model(tokens, ctx_sink)

    assert torch.allclose(plain, with_sink, atol=1e-6)
    assert set(sink.captured) == {0, 2, 3}

    # Verify captured values equal manual post-layer hidden states (x + residual).
    _, ctx_ref = _make_context(model, len(tokens), None)
    with torch.inference_mode():
        x = model.embed_tokens(tokens)
        residual = None
        for idx, layer in enumerate(model.layers):
            x, residual = layer(x, residual, ctx_ref)
            if idx in {0, 2, 3}:
                assert residual is not None
                assert torch.allclose(sink.get(idx), x + residual, atol=1e-5)

    # Pre-allocated buffers write in place.
    buffers = {idx: torch.empty(8, config.hidden_size, dtype=torch.float32) for idx in (1, 3)}
    buf_sink = FeatureSink(layer_ids=frozenset({1, 3}), buffers=buffers)
    _, ctx_buf = _make_context(model, len(tokens), buf_sink)
    with torch.inference_mode():
        model(tokens, ctx_buf)
    assert torch.allclose(buf_sink.get(3), sink.get(3), atol=1e-5)
    assert buf_sink.get(3).data_ptr() == buffers[3].data_ptr()
    buf_sink.clear()
    assert buf_sink.captured == {}
