"""Long-context and multimodal image probes used by :mod:`ayaka.entrypoints.bench`."""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING, Protocol

from ayaka.entrypoints.bench_prompts import build_image_prompt, build_prompt, prompt_digest

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ayaka.core.multimodal import ImageInput
    from ayaka.core.outputs import RequestOutput

__all__ = ["BenchProbeHost", "measure_extended_probes"]


class BenchProbeHost(Protocol):
    """Minimal surface :mod:`ayaka.entrypoints.bench` passes to extended probes."""

    seed: int
    repeats: int
    quick: bool
    has_vision: bool

    def encode(self, text: str) -> list[int]: ...
    def prompt_read(self, prompt: tuple[int, ...]) -> list[float]: ...
    def generation(self, prompt: tuple[int, ...]) -> list[float]: ...
    def chat_turn(self, prompt_text: str, image: ImageInput) -> RequestOutput: ...


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def measure_extended_probes(
    host: BenchProbeHost, usable: int, say: Callable[[str], None]
) -> dict[str, dict[str, object]]:
    """Measure optional long-context and 1024x768 image probes when the model fits them."""
    extra: dict[str, dict[str, object]] = {}
    if usable >= 4096:
        long_len = min(32768, usable)
        say(f"long context: {long_len} tokens")
        long_prompt = build_prompt("long", long_len, host.encode)
        long_sec = host.prompt_read(long_prompt.token_ids)
        long_rates = [long_len / value for value in long_sec]
        long_gen = host.generation(long_prompt.token_ids)
        extra["long_context"] = {
            "group": "long",
            "length": long_len,
            "digest": prompt_digest(long_prompt),
            "seconds": long_sec,
            "tokens_per_s": long_rates,
            "median_tokens_per_s": _median(long_rates),
            "output_tokens": 256,
            "generation_tokens_per_s": long_gen,
            "generation_median_tokens_per_s": _median(long_gen),
        }
    if host.has_vision:
        say("image probe: 1024x768")
        img_prompt, shot = build_image_prompt(1024, 768)
        cold_img: list[float] = []
        warm_img: list[float] = []
        prompt_tokens = 0
        for repetition in range(host.repeats + 1):
            text = f"Vision {host.seed}-{repetition}: {img_prompt}"
            run_times: list[float] = []
            for _ in range(2):
                started = time.perf_counter()
                output = host.chat_turn(text, shot)
                if output.error is not None:
                    raise RuntimeError(f"image benchmark request failed: {output.error.message}")
                run_times.append(time.perf_counter() - started)
                prompt_tokens = output.usage.prompt_tokens
            if repetition:
                cold_img.append(run_times[0])
                warm_img.append(run_times[1])
        extra["image"] = {
            "image_size": [1024, 768],
            "prompt_tokens": prompt_tokens,
            "cold_s": cold_img,
            "warm_s": warm_img,
            "cold_median_s": _median(cold_img),
            "warm_median_s": _median(warm_img),
        }
    return extra
