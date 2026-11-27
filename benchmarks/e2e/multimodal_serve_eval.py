"""Evaluate live multimodal HTTP serving with a 1024x768 PNG screenshot on A100."""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aiohttp
import torch
from _machine import machine, write

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.configs.server import ServerConfig
from ayaka.entrypoints.serve import serve
from ayaka.multimodal.png import make_ide_screenshot_png
from ayaka.utils.dtypes import DType

SCHEMA = {
    "type": "object",
    "properties": {
        "panel_count": {"type": "integer", "minimum": 1, "maximum": 10},
        "summary": {"type": "string", "minLength": 4, "maxLength": 64},
    },
    "required": ["panel_count", "summary"],
    "additionalProperties": False,
}


async def _exercise_model(model_id: str, b64_png: str) -> dict[str, Any]:
    stop = asyncio.Event()
    ready = asyncio.Event()
    bound: dict[str, Any] = {}

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

    server_task = asyncio.create_task(
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
        base = bound["url"]
        async with aiohttp.ClientSession() as client:
            # 1. OpenAI Chat Completions with 1024x768 base64 PNG + JSON schema
            oai_body = {
                "model": model_id,
                "temperature": 0.0,
                "max_tokens": 160,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "ui_summary", "strict": True, "schema": SCHEMA},
                },
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_png}"},
                            },
                            {"type": "text", "text": "Count main panels and summarize in JSON."},
                        ],
                    }
                ],
            }
            t0 = time.perf_counter()
            async with client.post(f"{base}/v1/chat/completions", json=oai_body) as resp:
                oai_status = resp.status
                oai_json = await resp.json()
            oai_ms = (time.perf_counter() - t0) * 1000.0
            content = oai_json["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            # 2. Anthropic Messages with 1024x768 base64 PNG
            ant_body = {
                "model": model_id,
                "max_tokens": 32,
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
                            {"type": "text", "text": "Describe the top toolbar and sidebar."},
                        ],
                    }
                ],
            }
            t1 = time.perf_counter()
            async with client.post(f"{base}/v1/messages", json=ant_body) as resp:
                ant_status = resp.status
                ant_json = await resp.json()
            ant_ms = (time.perf_counter() - t1) * 1000.0

            # 3. Remote URL refusal on both OpenAI and Anthropic
            bad_oai = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/shot.png"},
                            }
                        ],
                    }
                ],
            }
            async with client.post(f"{base}/v1/chat/completions", json=bad_oai) as resp:
                oai_url_refused_status = resp.status

            bad_ant = {
                "model": model_id,
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "url", "url": "https://example.com/shot.png"},
                            }
                        ],
                    }
                ],
            }
            async with client.post(f"{base}/v1/messages", json=bad_ant) as resp:
                ant_url_refused_status = resp.status
    finally:
        stop.set()
        await server_task

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    ant_usage = ant_json["usage"]
    ant_total_prompt = (
        ant_usage["input_tokens"]
        + ant_usage.get("cache_creation_input_tokens", 0)
        + ant_usage.get("cache_read_input_tokens", 0)
    )
    return {
        "model": model_id,
        "openai_chat": {
            "status": oai_status,
            "prompt_tokens": oai_json["usage"]["prompt_tokens"],
            "completion_tokens": oai_json["usage"]["completion_tokens"],
            "parsed_json": parsed,
            "latency_ms": oai_ms,
        },
        "anthropic_messages": {
            "status": ant_status,
            "input_tokens": ant_usage["input_tokens"],
            "cache_read_input_tokens": ant_usage.get("cache_read_input_tokens", 0),
            "cache_creation_input_tokens": ant_usage.get("cache_creation_input_tokens", 0),
            "total_prompt_tokens": ant_total_prompt,
            "output_tokens": ant_usage["output_tokens"],
            "text": ant_json["content"][0]["text"],
            "latency_ms": ant_ms,
        },
        "remote_url_refused": {
            "openai_status": oai_url_refused_status,
            "anthropic_status": ant_url_refused_status,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Live 1024x768 screenshot HTTP serving E2E check.")
    parser.add_argument(
        "--out",
        default="benchmarks/results/2026-11-27-nvidia-a100-sxm4-80gb/multimodal-serving-e2e.json",
    )
    args = parser.parse_args()

    b64_png = base64.b64encode(make_ide_screenshot_png(1024, 768)).decode("ascii")
    results: list[dict[str, Any]] = []
    for model_id in ("Qwen/Qwen3.5-9B", "google/gemma-4-E2B-it"):
        row = asyncio.run(_exercise_model(model_id, b64_png))
        print(
            f"[serve-e2e] {model_id}: oai={row['openai_chat']['status']} "
            f"(tok={row['openai_chat']['prompt_tokens']}) "
            f"ant={row['anthropic_messages']['status']} "
            f"(tok={row['anthropic_messages']['total_prompt_tokens']}, "
            f"cached={row['anthropic_messages']['cache_read_input_tokens']}) "
            f"url_refused=({row['remote_url_refused']['openai_status']},"
            f"{row['remote_url_refused']['anthropic_status']})",
            flush=True,
        )
        results.append(row)

    write(
        {"benchmark": "multimodal_serving_e2e", "machine": machine(), "results": results},
        args.out,
    )


if __name__ == "__main__":
    main()
