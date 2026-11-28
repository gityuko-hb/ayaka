"""Evaluate Qwen3.5 and Gemma 4 multimodal vision parity and eager vs lazy loading on A100.

Measures:
1. Exact 16-token greedy token-ID parity against Hugging Face ``AutoModelForImageTextToText``
   on a deterministic 1024x768 IDE screenshot PNG for ``Qwen/Qwen3.5-9B`` and
   ``google/gemma-4-E2B-it`` (bfloat16, FlashInfer + CUDA graphs active).
2. ``--vision-load eager`` vs ``--vision-load lazy`` startup VRAM, post-text-request vision load
   state, first-image latency + post-image VRAM, and warm second-image cache-hit latency.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from _machine import machine, write
from huggingface_hub import snapshot_download

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model import ModelConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.multimodal import ImageInput
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId
from ayaka.engine.llm import LLM
from ayaka.multimodal.png import make_ide_screenshot_png
from ayaka.multimodal.preprocess import preprocess_gemma4, preprocess_qwen3_5
from ayaka.multimodal.tokens import expand_image_placeholders
from ayaka.tokenizers.contracts import ChatMessage
from ayaka.utils.dtypes import DType

PROMPT_TEXT = (
    "Describe the main layout regions and status indicators shown in this IDE screenshot "
    "in one concise sentence."
)


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _nvml_used_bytes() -> int | None:
    from ayaka.device.info import probe_devices

    for dev in probe_devices():
        if dev.kind == "cuda" and dev.index == 0:
            return dev.total_bytes - dev.free_bytes
    return None


def _torch_allocated_bytes() -> int:
    torch.cuda.synchronize()
    return int(torch.cuda.memory_allocated(0))


def _build_mm_prompt(model_id: str, image: ImageInput) -> tuple[ModelConfig, tuple[int, ...], Any]:
    from transformers import AutoTokenizer

    directory = Path(snapshot_download(model_id, local_files_only=True))
    raw_cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model_cfg = ModelConfig.from_hf_config(raw_cfg)
    spec = model_cfg.vision_spec
    if spec is None:
        raise RuntimeError(f"{model_id} has no vision_spec")

    tokenizer: Any = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT_TEXT}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    raw_ids = [int(t) for t in tokenizer.encode(rendered, add_special_tokens=False)]
    expanded, spans = expand_image_placeholders(raw_ids, [image], spec)
    return model_cfg, expanded, (spec, spans)


def _hf_multimodal_greedy(
    model_id: str,
    model_cfg: ModelConfig,
    prompt_ids: tuple[int, ...],
    image: ImageInput,
    gen_tokens: int,
) -> list[int]:
    from transformers import AutoModelForImageTextToText

    spec = model_cfg.vision_spec
    if spec is None:
        raise RuntimeError(f"{model_id} missing vision_spec")
    hf_raw: Any = AutoModelForImageTextToText.from_pretrained(
        model_id, local_files_only=True, dtype=torch.bfloat16
    )
    hf_model: Any = hf_raw.to("cuda")
    hf_model.eval()
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.int64, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    mm_types = (input_ids == spec.image_token_id).to(torch.int32)
    extra: dict[str, Any] = {"mm_token_type_ids": mm_types}
    if spec.family == "qwen3_5":
        pixels, grid_thw = preprocess_qwen3_5([image], spec, device="cuda")
        extra["pixel_values"] = pixels.to(torch.bfloat16)
        extra["image_grid_thw"] = grid_thw
    else:
        pixels, pos_ids, _ = preprocess_gemma4([image], spec, device="cuda")
        extra["pixel_values"] = pixels.to(torch.bfloat16)
        extra["image_position_ids"] = pos_ids

    with torch.inference_mode():
        seq = hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_tokens,
            do_sample=False,
            **extra,
        )
    tokens = [int(t) for t in seq[0, len(prompt_ids) :].tolist()]
    del hf_model
    _clear_cuda()
    return tokens


def _engine_config(model_id: str, vision_load: Literal["eager", "lazy"]) -> EngineConfig:
    return EngineConfig(
        source=ModelSourceConfig(model_id, offline=True),
        device=DeviceConfig(
            device="cuda",
            dtype=DType.BF16,
            backends=("flashinfer", "torch"),
            cuda_graph=True,
            vision_load=vision_load,
        ),
        cache=CacheConfig(
            block_size=16,
            kv_reserve_tokens=2048,
            prefix_cache=True,
            state_checkpoint_max_bytes=2 * 2**30,
        ),
        scheduler=SchedulerConfig(max_running=4, max_tokens_per_step=1024),
    )


def _run_mode(
    model_id: str,
    vision_load: Literal["eager", "lazy"],
    image: ImageInput,
    prompt_ids: tuple[int, ...],
    spans: Any,
    gen_tokens: int,
    hf_tokens: list[int],
) -> dict[str, Any]:
    sampling = SamplingParams(temperature=0.0, max_tokens=gen_tokens, ignore_eos=True)
    with LLM.from_config(_engine_config(model_id, vision_load)) as llm:
        model = llm.engine.runner.model
        providers = dict(model.ops.providers)
        if "flashinfer" not in providers.values():
            raise RuntimeError(f"expected flashinfer backend, got {providers}")
        startup = llm.engine.startup
        startup_weights = startup.profile.weights_bytes if startup is not None else 0
        alloc_startup = _torch_allocated_bytes()
        nvml_startup = _nvml_used_bytes()
        loaded_startup = model.vision_loaded()

        # Run one text-only request to confirm lazy stays unloaded on text traffic
        txt_out = llm.generate("Hello!", SamplingParams(temperature=0.0, max_tokens=2))
        if isinstance(txt_out, list):
            txt_out = txt_out[0]
        if txt_out.error:
            raise RuntimeError(f"text request failed: {txt_out.error}")
        loaded_after_text = model.vision_loaded()
        alloc_before_img = _torch_allocated_bytes()
        nvml_before_img = _nvml_used_bytes()

        messages = [
            ChatMessage(
                role="user",
                content=PROMPT_TEXT,
                images=(image,),
            )
        ]
        # First image request (unique salt prevents accidental text-prefix interference)
        t0 = time.perf_counter()
        out1 = llm.chat(messages, sampling)
        first_ms = (time.perf_counter() - t0) * 1000.0
        if out1.error:
            raise RuntimeError(f"first image request failed: {out1.error}")
        tokens1 = [int(t) for t in out1.outputs[0].token_ids]

        loaded_after_img = model.vision_loaded()
        alloc_after_img = _torch_allocated_bytes()
        nvml_after_img = _nvml_used_bytes()

        # Second image request with same image (tests vision feature + KV prefix reuse)
        t1 = time.perf_counter()
        out2 = llm.chat(messages, sampling)
        second_ms = (time.perf_counter() - t1) * 1000.0
        if out2.error:
            raise RuntimeError(f"second image request failed: {out2.error}")
        tokens2 = [int(t) for t in out2.outputs[0].token_ids]

        # Explicit raw pre-expanded image request without prefix cache to double check pure parity
        req = GenerateRequest(
            request_id=RequestId("mm-cold-parity"),
            prompt=prompt_ids,
            sampling=sampling,
            prefix_cache=False,
            images=(image,),
            image_spans=spans,
        )
        cold_out = llm.generate(req)
        if isinstance(cold_out, list):
            cold_out = cold_out[0]
        if cold_out.error:
            raise RuntimeError(f"cold parity failed: {cold_out.error}")
        cold_tokens = [int(t) for t in cold_out.outputs[0].token_ids]

    _clear_cuda()
    return {
        "vision_load": vision_load,
        "startup_weights_bytes": startup_weights,
        "torch_allocated_startup_bytes": alloc_startup,
        "torch_allocated_before_image_bytes": alloc_before_img,
        "torch_allocated_after_image_bytes": alloc_after_img,
        "torch_image_delta_bytes": alloc_after_img - alloc_before_img,
        "nvml_used_startup_bytes": nvml_startup,
        "nvml_used_before_image_bytes": nvml_before_img,
        "nvml_used_after_image_bytes": nvml_after_img,
        "vision_loaded_at_startup": loaded_startup,
        "vision_loaded_after_text": loaded_after_text,
        "vision_loaded_after_image": loaded_after_img,
        "first_image_ms": first_ms,
        "warm_second_image_ms": second_ms,
        "first_image_prompt_tokens": out1.usage.prompt_tokens,
        "first_image_tokens": tokens1,
        "warm_second_image_tokens": tokens2,
        "cold_noprefix_tokens": cold_tokens,
        "hf_tokens": hf_tokens,
        "exact_greedy_match_vs_hf": (
            tokens1 == hf_tokens and tokens2 == hf_tokens and cold_tokens == hf_tokens
        ),
        "vision_weights_bytes": model.vision_weight_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision parity & eager/lazy benchmark on A100.")
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument(
        "--out",
        default="benchmarks/results/2026-11-27-nvidia-a100-sxm4-80gb/vision-eager-vs-lazy.json",
    )
    args = parser.parse_args()

    png_bytes = make_ide_screenshot_png(1024, 768)
    shot = ImageInput(data=png_bytes, media_type="image/png")

    models = ["Qwen/Qwen3.5-9B", "google/gemma-4-E2B-it"]
    rows: list[dict[str, Any]] = []
    for model_id in models:
        model_cfg, prompt_ids, (_, spans) = _build_mm_prompt(model_id, shot)
        hf_tokens = _hf_multimodal_greedy(model_id, model_cfg, prompt_ids, shot, args.gen_tokens)
        eager = _run_mode(model_id, "eager", shot, prompt_ids, spans, args.gen_tokens, hf_tokens)
        lazy = _run_mode(model_id, "lazy", shot, prompt_ids, spans, args.gen_tokens, hf_tokens)
        saved_mib = (
            eager["torch_allocated_startup_bytes"] - lazy["torch_allocated_startup_bytes"]
        ) / 2**20
        print(
            f"[vision] {model_id}: eager_match={eager['exact_greedy_match_vs_hf']} "
            f"lazy_match={lazy['exact_greedy_match_vs_hf']} "
            f"lazy_vram_saved={saved_mib:.1f} MiB "
            f"eager_1st_ms={eager['first_image_ms']:.1f} lazy_1st_ms={lazy['first_image_ms']:.1f}",
            flush=True,
        )
        rows.append(
            {
                "model": model_id,
                "image_size": [1024, 768],
                "prompt_len": len(prompt_ids),
                "eager": eager,
                "lazy": lazy,
                "startup_vram_saved_bytes": (
                    eager["torch_allocated_startup_bytes"] - lazy["torch_allocated_startup_bytes"]
                ),
            }
        )

    payload = {
        "benchmark": "vision_parity_and_eager_vs_lazy",
        "machine": machine(),
        "results": rows,
    }
    write(payload, args.out)


if __name__ == "__main__":
    main()
