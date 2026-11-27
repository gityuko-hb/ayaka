"""Live GPU E2E tests for multimodal HTTP serving, vision parity, and lazy VRAM savings."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")
aiohttp: Any = pytest.importorskip("aiohttp")

from ayaka.configs.cache import CacheConfig  # noqa: E402
from ayaka.configs.device import DeviceConfig  # noqa: E402
from ayaka.configs.engine import EngineConfig  # noqa: E402
from ayaka.configs.model_source import ModelSourceConfig  # noqa: E402
from ayaka.configs.scheduler import SchedulerConfig  # noqa: E402
from ayaka.configs.server import ServerConfig  # noqa: E402
from ayaka.entrypoints.serve import serve  # noqa: E402
from ayaka.multimodal.png import make_ide_screenshot_png  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402

pytestmark = pytest.mark.gpu


def _require_gpu_and_model(repo_id: str) -> None:
    if not torch.cuda.is_available():
        if os.environ.get("AYAKA_REQUIRE_GPU") == "1":
            pytest.fail("AYAKA_REQUIRE_GPU=1 but CUDA is unavailable")
        pytest.skip("CUDA unavailable")
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id, local_files_only=True)
    except Exception:
        if os.environ.get("AYAKA_REQUIRE_GPU") == "1":
            pytest.fail(f"missing cached model {repo_id}")
        pytest.skip(f"cached model {repo_id} not available locally")


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3.5-9B", "google/gemma-4-E2B-it"])
def test_live_serve_multimodal_1024x768_openai_and_anthropic(model_id: str) -> None:
    _require_gpu_and_model(model_id)
    b64_png = base64.b64encode(make_ide_screenshot_png(1024, 768)).decode("ascii")
    schema = {
        "type": "object",
        "properties": {
            "panels": {"type": "integer", "minimum": 1},
            "note": {"type": "string", "maxLength": 64},
        },
        "required": ["panels", "note"],
        "additionalProperties": False,
    }

    async def exercise() -> tuple[int, dict[str, Any], int, dict[str, Any], int, int]:
        stop = asyncio.Event()
        ready = asyncio.Event()
        bound: dict[str, str] = {}

        def on_ready(host: str, port: int) -> None:
            bound["url"] = f"http://{host}:{port}"
            ready.set()

        server_cfg = ServerConfig(host="127.0.0.1", port=0, shutdown_timeout=2.0)
        engine_cfg = EngineConfig(
            source=ModelSourceConfig(model_id, offline=True),
            device=DeviceConfig(
                device="cuda",
                dtype=DType.BF16,
                backends=("flashinfer", "torch"),
                cuda_graph=True,
            ),
            cache=CacheConfig(block_size=16, kv_reserve_tokens=2048, prefix_cache=True),
            scheduler=SchedulerConfig(max_running=4, max_tokens_per_step=1024),
        )
        task = asyncio.create_task(
            serve(
                server_cfg,
                engine_cfg,
                model_name=model_id,
                say=lambda _line: None,
                stop=stop,
                handle_signals=False,
                on_ready=on_ready,
            )
        )
        try:
            await ready.wait()
            url = bound["url"]
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    f"{url}/v1/chat/completions",
                    json={
                        "model": model_id,
                        "temperature": 0.0,
                        "max_tokens": 160,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {"name": "ui", "strict": True, "schema": schema},
                        },
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{b64_png}"},
                                    },
                                    {"type": "text", "text": "Respond in JSON."},
                                ],
                            }
                        ],
                    },
                ) as r1:
                    s1, d1 = r1.status, await r1.json()
                async with client.post(
                    f"{url}/v1/messages",
                    json={
                        "model": model_id,
                        "max_tokens": 24,
                        "temperature": 0.0,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": b64_png,
                                        },
                                    },
                                    {"type": "text", "text": "Summarize this screenshot."},
                                ],
                            }
                        ],
                    },
                ) as r2:
                    s2, d2 = r2.status, await r2.json()
                async with client.post(
                    f"{url}/v1/chat/completions",
                    json={
                        "model": model_id,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": "https://example.com/a.png"},
                                    }
                                ],
                            }
                        ],
                    },
                ) as r3:
                    s3 = r3.status
                async with client.post(
                    f"{url}/v1/messages",
                    json={
                        "model": model_id,
                        "max_tokens": 8,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "url",
                                            "url": "https://example.com/a.png",
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                ) as r4:
                    s4 = r4.status
            return s1, d1, s2, d2, s3, s4
        finally:
            stop.set()
            await task

    s1, d1, s2, d2, s3, s4 = asyncio.run(exercise())
    assert s1 == 200
    parsed = json.loads(d1["choices"][0]["message"]["content"])
    assert isinstance(parsed["panels"], int)
    assert isinstance(parsed["note"], str)
    assert d1["usage"]["prompt_tokens"] >= 280

    assert s2 == 200
    assert d2["content"][0]["text"]
    ant_usage = d2["usage"]
    ant_total = (
        ant_usage["input_tokens"]
        + ant_usage.get("cache_creation_input_tokens", 0)
        + ant_usage.get("cache_read_input_tokens", 0)
    )
    assert ant_total >= 280
    assert ant_usage.get("cache_read_input_tokens", 0) > 0

    assert s3 == 400
    assert s4 == 400


def test_a100_vision_parity_and_lazy_savings() -> None:
    _require_gpu_and_model("Qwen/Qwen3.5-9B")
    _require_gpu_and_model("google/gemma-4-E2B-it")
    eval_path = Path(__file__).resolve().parents[2] / "benchmarks" / "e2e" / "vision_eval.py"
    spec = importlib.util.spec_from_file_location("vision_eval", eval_path)
    assert spec is not None
    assert spec.loader is not None
    vision_eval: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vision_eval)

    shot = vision_eval.ImageInput(data=make_ide_screenshot_png(1024, 768), media_type="image/png")
    for model_id in ("Qwen/Qwen3.5-9B", "google/gemma-4-E2B-it"):
        cfg, prompt_ids, (_, spans) = vision_eval._build_mm_prompt(model_id, shot)  # pyright: ignore[reportPrivateUsage]
        hf_tokens = vision_eval._hf_multimodal_greedy(model_id, cfg, prompt_ids, shot, 16)  # pyright: ignore[reportPrivateUsage]
        eager = vision_eval._run_mode(model_id, "eager", shot, prompt_ids, spans, 16, hf_tokens)  # pyright: ignore[reportPrivateUsage]
        lazy = vision_eval._run_mode(model_id, "lazy", shot, prompt_ids, spans, 16, hf_tokens)  # pyright: ignore[reportPrivateUsage]
        assert eager["exact_greedy_match_vs_hf"] is True
        assert lazy["exact_greedy_match_vs_hf"] is True
        assert eager["vision_loaded_at_startup"] is True
        assert lazy["vision_loaded_at_startup"] is False
        assert lazy["vision_loaded_after_text"] is False
        assert lazy["vision_loaded_after_image"] is True
        assert (
            eager["torch_allocated_startup_bytes"] - lazy["torch_allocated_startup_bytes"]
        ) > 200 * 2**20
