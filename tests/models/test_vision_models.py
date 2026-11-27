"""Unit and Hugging Face parity tests for vision towers, M-RoPE, PLE masking, and lazy loading."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

torch: Any = pytest.importorskip("torch")

from ayaka.configs.device import DeviceConfig  # noqa: E402
from ayaka.configs.model import ModelConfig  # noqa: E402
from ayaka.configs.model_source import ModelSourceConfig, RequestFormat  # noqa: E402
from ayaka.configs.scheduler import SchedulerConfig  # noqa: E402
from ayaka.core.multimodal import ImageInput, ImageSpan  # noqa: E402
from ayaka.core.params import SamplingParams  # noqa: E402
from ayaka.core.request import TokenizedRequest  # noqa: E402
from ayaka.core.types import ForwardMode, KVLayoutKind, RequestId  # noqa: E402
from ayaka.execution.runner import ModelRunner  # noqa: E402
from ayaka.kvcache.manager import KVCacheManager  # noqa: E402
from ayaka.kvcache.prefix.hash_cache import HashPrefixCache  # noqa: E402
from ayaka.kvcache.prefix.hashing import ChainRoot  # noqa: E402
from ayaka.kvcache.state_slots import StateCheckpointStore, StateSlotPool  # noqa: E402
from ayaka.kvcache.storage.pool import KVPool, RecurrentStatePool  # noqa: E402
from ayaka.kvcache.storage.spec import KVGroup  # noqa: E402
from ayaka.model_loader.loader import load_model  # noqa: E402
from ayaka.models._vision_gemma4 import Gemma4VisionTower  # noqa: E402
from ayaka.models._vision_qwen3_5 import Qwen3_5VisionTower  # noqa: E402
from ayaka.models.gemma4 import Gemma4ForConditionalGeneration  # noqa: E402
from ayaka.models.qwen3_5 import (  # noqa: E402
    Qwen3_5ForConditionalGeneration,
    _qwen3_5_delta,  # pyright: ignore[reportPrivateUsage]
    _qwen3_5_positions,  # pyright: ignore[reportPrivateUsage]
)
from ayaka.multimodal.png import make_ide_screenshot_png  # noqa: E402
from ayaka.multimodal.preprocess import preprocess_gemma4, preprocess_qwen3_5  # noqa: E402
from ayaka.multimodal.tokens import expand_image_placeholders  # noqa: E402
from ayaka.ops.backend import load_backend  # noqa: E402
from ayaka.scheduler.scheduler import Scheduler  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402
from gpt2_support import tensor_data  # noqa: E402
from safetensors_files import write_safetensors  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


def _tiny_qwen3_5_vision_cfg() -> dict[str, Any]:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings": False,
        "image_token_id": 60,
        "vision_config": {
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 16,
            "out_hidden_size": 32,
            "hidden_act": "gelu_pytorch_tanh",
        },
        "text_config": {
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "attn_output_gate": True,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 256,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_interleaved": True,
                "mrope_section": [3, 3, 2],
            },
        },
    }


def _tiny_gemma4_vision_cfg(*, use_bidir: bool = True) -> dict[str, Any]:
    return {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "tie_word_embeddings": False,
        "image_token_id": 10,
        "boi_token_id": 11,
        "eoi_token_id": 12,
        "vision_soft_tokens_per_image": 4,
        "vision_config": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "default_output_length": 4,
            "position_embedding_size": 32,
            "rms_norm_eps": 1e-6,
            "standardize": False,
            "use_clipped_linears": True,
            "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
        },
        "text_config": {
            "vocab_size": 64,
            "vocab_size_per_layer_input": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_kv_heads": 1,
            "head_dim": 16,
            "global_head_dim": 16,
            "hidden_size_per_layer_input": 8,
            "num_kv_shared_layers": 0,
            "hidden_act": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 128,
            "pad_token_id": 0,
            "use_bidirectional_attention": "vision" if use_bidir else None,
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                "full_attention": {
                    "rope_theta": 100000.0,
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.5,
                },
            },
        },
    }


def _export_qwen3_5_checkpoint(directory: Path, raw_cfg: dict[str, Any]) -> None:
    torch.manual_seed(11)
    cfg = ModelConfig.from_hf_config(raw_cfg)
    ops = load_backend(("torch",), device="cpu", dtype=DType.FP32, head_dim=cfg.head_dim)
    llm = Qwen3_5ForConditionalGeneration(cfg, ops=ops, dtype=DType.FP32, device="cpu")
    vision = Qwen3_5VisionTower(cfg, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for mod in (llm, vision):
            for name, param in mod.named_parameters():
                if "a_log" in name:
                    param.uniform_(-2.0, 0.5)
                elif "norm" in name or "dt_bias" in name:
                    param.normal_(0.0, 0.05)
                else:
                    param.normal_(0.0, 0.1)

    tensors: dict[str, Any] = {}
    for name, param in llm.named_parameters():
        tensor = param.detach().clone()
        if name == "lm_head.weight":
            tensors["lm_head.weight"] = tensor
        elif ".mlp.gate_up_proj." in name:
            half = tensor.shape[0] // 2
            tensors[f"model.language_model.{name.replace('gate_up_proj', 'gate_proj')}"] = tensor[
                :half
            ].clone()
            tensors[f"model.language_model.{name.replace('gate_up_proj', 'up_proj')}"] = tensor[
                half:
            ].clone()
        elif ".attn." in name:
            mapped = (
                name.replace(".attn.conv1d_weight", ".linear_attn.conv1d.weight")
                .replace(".attn.a_log", ".linear_attn.A_log")
                .replace(".attn.dt_bias", ".linear_attn.dt_bias")
                .replace(".attn.norm_weight", ".linear_attn.norm.weight")
            )
            for lin_key in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                mapped = mapped.replace(f".attn.{lin_key}.", f".linear_attn.{lin_key}.")
            mapped = mapped.replace(".attn.", ".self_attn.")
            tensors[f"model.language_model.{mapped}"] = tensor
        else:
            tensors[f"model.language_model.{name}"] = tensor
    for name, param in vision.named_parameters():
        tensors[f"model.visual.{name}"] = param.detach().clone()

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(raw_cfg), encoding="utf-8")
    write_safetensors(
        directory / "model.safetensors",
        {name: tensor_data(tensor, tensor.dtype) for name, tensor in tensors.items()},
    )


@pytest.mark.unit
def test_qwen3_5_mrope_positions_and_delta_match_hf_rules() -> None:
    spans = (ImageSpan(image_index=0, token_offset=3, token_count=6, grid_thw=(1, 4, 6)),)
    pos = _qwen3_5_positions(12, spans, 2, torch.device("cpu"))
    delta = _qwen3_5_delta(12, spans, 2)

    assert tuple(pos.shape) == (3, 12)
    assert pos[:, :3].tolist() == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    # 2x3 merged image grid starts at position 3, advances by max(2, 3)=3 to position 6
    assert pos[0, 3:9].tolist() == [3, 3, 3, 3, 3, 3]
    assert pos[1, 3:9].tolist() == [3, 3, 3, 4, 4, 4]
    assert pos[2, 3:9].tolist() == [3, 4, 5, 3, 4, 5]
    assert pos[:, 9:].tolist() == [[6, 7, 8], [6, 7, 8], [6, 7, 8]]
    assert delta == int(pos.max().item()) + 1 - 12 == -3


@pytest.mark.unit
def test_qwen3_5_lazy_load_text_stays_unloaded_then_image_runs_and_caches(
    tmp_path: Path,
) -> None:
    raw_cfg = _tiny_qwen3_5_vision_cfg()
    _export_qwen3_5_checkpoint(tmp_path, raw_cfg)

    source = ModelSourceConfig(model=str(tmp_path), format=RequestFormat.SAFETENSORS, offline=True)
    lazy_dev = DeviceConfig(device="cpu", dtype=DType.FP32, backends=("torch",), vision_load="lazy")
    ops = load_backend(("torch",), device="cpu", dtype=DType.FP32, head_dim=16)
    lazy_model = load_model(source, lazy_dev, ops=ops)
    assert lazy_model.has_vision_tower()
    assert not lazy_model.vision_loaded()
    assert lazy_model.vision_weight_bytes() > 0

    eager_dev = DeviceConfig(
        device="cpu", dtype=DType.FP32, backends=("torch",), vision_load="eager"
    )
    eager_model = load_model(source, eager_dev, ops=ops)
    assert eager_model.vision_loaded()

    kv_groups = lazy_model.kv_specs(block_size=16, layout=KVLayoutKind.NHD)
    kv_group = kv_groups[0]
    assert isinstance(kv_group, KVGroup)
    state_group = lazy_model.state_spec()
    assert state_group is not None
    pool = KVPool(kv_group, 32, device="cpu")
    state_slots = StateSlotPool(5)
    state_pool = RecurrentStatePool(state_group, 5, device="cpu")
    checkpoints = StateCheckpointStore(state_group.spec.bytes_per_slot() * 8, stride_blocks=1)
    prefix = HashPrefixCache(
        ChainRoot(model_id="tiny-q35", dtype="fp32", codec="raw", block_size=16)
    )
    kv_mgr = KVCacheManager(
        kv_groups,
        [32],
        max_requests=4,
        max_blocks_per_request=32,
        prefix=prefix,
        state_slots=state_slots,
        state_checkpoints=checkpoints,
    )
    runner = ModelRunner(
        lazy_model,
        [pool],
        kv_mgr.table,
        max_rows=128,
        device="cpu",
        state_pool=state_pool,
        state_checkpoints=checkpoints,
    )
    scheduler = Scheduler(
        SchedulerConfig(max_running=4, max_tokens_per_step=64),
        kv_mgr,
        max_model_len=128,
        max_chunk_tokens=64,
    )

    # Text-only request leaves vision tower unloaded
    text_req = TokenizedRequest(
        request_id=RequestId("txt-1"),
        prompt_token_ids=(1, 2, 3, 4),
        sampling=SamplingParams(temperature=0.0, max_tokens=2),
    )
    scheduler.add_request(text_req)
    while scheduler.has_unfinished:
        step = scheduler.schedule(runner.plan_lookahead())
        if step is None:
            break
        scheduler.update(runner.execute(step))
    assert not lazy_model.vision_loaded()

    # Multimodal request triggers lazy load, populates vision cache, and completes prefill+decode
    shot = ImageInput(data=make_ide_screenshot_png(64, 64), media_type="image/png")
    spec = lazy_model.config.vision_spec
    assert spec is not None
    expanded, spans = expand_image_placeholders([1, spec.image_token_id, 2], [shot], spec)
    mm_req_1 = TokenizedRequest(
        request_id=RequestId("mm-1"),
        prompt_token_ids=expanded,
        sampling=SamplingParams(temperature=0.0, max_tokens=2),
        images=(shot,),
        image_spans=spans,
    )
    scheduler.add_request(mm_req_1)
    while scheduler.has_unfinished:
        step = scheduler.schedule(runner.plan_lookahead())
        if step is None:
            break
        scheduler.update(runner.execute(step))
    assert lazy_model.vision_loaded()
    assert len(runner._vision_cache) == 1  # pyright: ignore[reportPrivateUsage]

    # Different image payload gets distinct cache salt (no cross-image hit) and adds entry
    shot2 = ImageInput(data=make_ide_screenshot_png(96, 64), media_type="image/png")
    expanded2, spans2 = expand_image_placeholders([1, spec.image_token_id, 2], [shot2], spec)
    mm_req_2 = TokenizedRequest(
        request_id=RequestId("mm-2"),
        prompt_token_ids=expanded2,
        sampling=SamplingParams(temperature=0.0, max_tokens=1),
        images=(shot2,),
        image_spans=spans2,
    )
    scheduler.add_request(mm_req_2)
    while scheduler.has_unfinished:
        step = scheduler.schedule(runner.plan_lookahead())
        if step is None:
            break
        scheduler.update(runner.execute(step))
    assert len(runner._vision_cache) == 2  # pyright: ignore[reportPrivateUsage]


@pytest.mark.unit
def test_gemma4_ple_masks_image_token_and_bidirectional_sliding_attends() -> None:
    cfg = ModelConfig.from_hf_config(_tiny_gemma4_vision_cfg(use_bidir=True))
    ops = load_backend(("torch",), device="cpu", dtype=DType.FP32, head_dim=cfg.head_dim)
    model = Gemma4ForConditionalGeneration(cfg, ops=ops, dtype=DType.FP32, device="cpu")
    torch.manual_seed(5)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.1)

    tokens = torch.tensor([1, 10, 10, 10, 2], dtype=torch.long)
    # Changing image_token_id row in embed_tokens & embed_tokens_per_layer must have zero effect
    embeds_a = model.embed_input_ids(tokens)
    ple_a = model._per_layer_inputs(tokens, embeds_a)  # pyright: ignore[reportPrivateUsage]
    with torch.no_grad():
        model.embed_tokens.weight[10].add_(99.0)
        assert model.embed_tokens_per_layer is not None
        model.embed_tokens_per_layer.weight[10].add_(99.0)
    embeds_b = model.embed_input_ids(tokens)
    ple_b = model._per_layer_inputs(tokens, embeds_b)  # pyright: ignore[reportPrivateUsage]
    torch.testing.assert_close(embeds_a, embeds_b)
    assert ple_a is not None
    assert ple_b is not None
    torch.testing.assert_close(ple_a, ple_b)

    # Bidirectional sliding attention allows token 1 to depend on future token 3 in span
    assert model.uses_bidirectional_vision()
    groups = model.kv_specs(block_size=16, layout=KVLayoutKind.NHD)
    pools = [KVPool(g, 8, device="cpu") for g in groups if isinstance(g, KVGroup)]
    kv_mgr = KVCacheManager(groups, [8] * len(groups), max_requests=2, max_blocks_per_request=8)
    kv_mgr.allocate(RequestId("g1"), 5)
    kv_mgr.table.take_dirty()
    table_dev = torch.tensor(kv_mgr.table.rows[0, :1], dtype=torch.int32)
    meta = ops.attention.plan(
        ForwardMode.PREFILL,
        torch.tensor([5], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int32),
        table_dev,
        16,
    )
    kv_bufs = {lid: p.buffers(lid) for p in pools for lid in p.group.layer_ids}
    from ayaka.layers.context import ForwardContext

    embeds_mod = embeds_a.clone()
    embeds_mod[3] += 3.0
    ctx1 = ForwardContext(
        attention=meta,
        kv_buffers=kv_bufs,
        layer_attention={0: meta, 1: meta},
        input_embeds=embeds_a,
        bidirectional_spans=((1, 4),),
    )
    ctx2 = ForwardContext(
        attention=meta,
        kv_buffers=kv_bufs,
        layer_attention={0: meta, 1: meta},
        input_embeds=embeds_mod,
        bidirectional_spans=((1, 4),),
    )
    with torch.no_grad():
        out1 = model(tokens, ctx1)
        out2 = model(tokens, ctx2)
    # Token at position 1 attended to future image token at position 3 because of bidirectional span
    assert float((out1[1] - out2[1]).abs().max()) > 1e-3


@pytest.mark.integration
def test_qwen3_5_vision_tower_parity_with_transformers() -> None:
    transformers = pytest.importorskip("transformers")
    cfg = ModelConfig.from_hf_config(_tiny_qwen3_5_vision_cfg())
    vc = cfg.vision_config
    assert vc is not None
    hf_cfg = transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5VisionConfig(
        depth=vc["depth"],
        hidden_size=vc["hidden_size"],
        intermediate_size=vc["intermediate_size"],
        num_heads=vc["num_heads"],
        in_channels=vc["in_channels"],
        patch_size=vc["patch_size"],
        temporal_patch_size=vc["temporal_patch_size"],
        spatial_merge_size=vc["spatial_merge_size"],
        num_position_embeddings=vc["num_position_embeddings"],
        out_hidden_size=vc["out_hidden_size"],
        hidden_act=vc["hidden_act"],
    )
    hf_cfg._attn_implementation = "eager"  # match PyTorch reference SDPA
    torch.manual_seed(17)
    hf_model = transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5VisionModel(hf_cfg).eval()
    ours = Qwen3_5VisionTower(cfg, dtype=torch.float32, device="cpu").eval()
    ours.load_state_dict(hf_model.state_dict(), strict=True)

    shot = ImageInput(data=make_ide_screenshot_png(96, 64), media_type="image/png")
    spec = cfg.vision_spec
    assert spec is not None
    pixels, grid_thw = preprocess_qwen3_5([shot], spec)
    with torch.no_grad():
        hf_out = hf_model(pixels, grid_thw).pooler_output
        our_out = ours(pixels, grid_thw)
    torch.testing.assert_close(our_out, hf_out, atol=1e-4, rtol=1e-4)


@pytest.mark.integration
def test_gemma4_vision_tower_parity_with_transformers() -> None:
    transformers = pytest.importorskip("transformers")
    raw = _tiny_gemma4_vision_cfg(use_bidir=False)
    cfg = ModelConfig.from_hf_config(raw)
    vc = raw["vision_config"]
    tc = raw["text_config"]
    hf_vc = transformers.models.gemma4.configuration_gemma4.Gemma4VisionConfig(
        hidden_size=vc["hidden_size"],
        intermediate_size=vc["intermediate_size"],
        num_hidden_layers=vc["num_hidden_layers"],
        num_attention_heads=vc["num_attention_heads"],
        num_key_value_heads=vc["num_key_value_heads"],
        head_dim=vc["head_dim"],
        patch_size=vc["patch_size"],
        pooling_kernel_size=vc["pooling_kernel_size"],
        default_output_length=vc["default_output_length"],
        position_embedding_size=vc["position_embedding_size"],
        rms_norm_eps=vc["rms_norm_eps"],
        standardize=vc["standardize"],
        use_clipped_linears=vc["use_clipped_linears"],
        rope_parameters=vc["rope_parameters"],
    )
    hf_vc._attn_implementation = "eager"
    hf_tc = transformers.models.gemma4.configuration_gemma4.Gemma4TextConfig(
        vocab_size=tc["vocab_size"],
        hidden_size=tc["hidden_size"],
    )
    torch.manual_seed(19)
    hf_tower = transformers.models.gemma4.modeling_gemma4.Gemma4VisionModel(hf_vc).eval()
    hf_embed = transformers.models.gemma4.modeling_gemma4.Gemma4MultimodalEmbedder(
        hf_vc, hf_tc
    ).eval()
    with torch.no_grad():
        for name, buf in hf_tower.named_buffers():
            if name.endswith(("_min",)):
                buf.fill_(-10.0)
            elif name.endswith(("_max",)):
                buf.fill_(10.0)
    ours = Gemma4VisionTower(cfg, dtype=torch.float32, device="cpu").eval()
    state = {f"vision_tower.{k}": v for k, v in hf_tower.state_dict().items()} | {
        f"embed_vision.{k}": v for k, v in hf_embed.state_dict().items()
    }
    ours.load_state_dict(state, strict=True)

    shot = ImageInput(data=make_ide_screenshot_png(96, 64), media_type="image/png")
    spec = cfg.vision_spec
    assert spec is not None
    pixels, pos_ids, _ = preprocess_gemma4([shot], spec)
    with torch.no_grad():
        hf_soft = hf_tower(pixels, pos_ids).last_hidden_state
        hf_proj = hf_embed(hf_soft)
        our_proj = ours(pixels, pos_ids)
    torch.testing.assert_close(our_proj, hf_proj, atol=1e-4, rtol=1e-4)
