"""A captured decode step reads the device exactly once: when it reads its sampled tokens.

PyTorch's sync debug mode raises on every operation that makes the host wait for the GPU
(``.item()``, ``.tolist()``, a blocking copy to the host, a ``bool`` of a device tensor). The test
writes a decode step into the graph's inputs, plans it and queues its replay under that mode, for
both backends; any wait in there fails the test. The one declared wait, the read of the sampled
tokens, is made after the guarded block, and must give the tokens eager decoding gives.

Eager steps (prefill, and decode without graphs) read the device in ``attention.plan`` and in the
reference sampler's checks; they are not what this test guards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.params import SamplingParams
from ayaka.core.request import TokenizedRequest
from ayaka.core.types import ForwardMode, RequestId
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.utils.torch_utils import no_device_sync  # noqa: E402
from tiny_model import TinyStack, build_tiny_stack, tiny_gpt2_config  # noqa: E402

if TYPE_CHECKING:
    from ayaka.scheduler.step import StepInput

pytestmark = pytest.mark.gpu

CASES = [
    pytest.param(("torch",), DType.FP16, id="torch-fp16"),
    pytest.param(("flashinfer", "torch"), DType.FP16, id="flashinfer-fp16"),
]


def _stack(backends: tuple[str, ...], dtype: DType, *, graphs: bool) -> TinyStack:
    if "flashinfer" in backends:
        pytest.importorskip("flashinfer", reason="the flashinfer extra is not installed")
    config = tiny_gpt2_config(hidden_size=128, num_attention_heads=2, num_kv_heads=2, head_dim=64)
    stack = build_tiny_stack(
        max_tokens_per_step=64,
        device="cuda",
        dtype=dtype,
        backends=backends,
        cuda_graph=graphs,
        config=config,
        block_size=16,
    )
    if graphs:
        stack.runner.capture_graphs((1, 2, 4))
    return stack


def _decode_step(stack: TinyStack) -> StepInput:
    while (step := stack.scheduler.schedule()) is not None:
        if step.mode is ForwardMode.DECODE:
            return step
        stack.scheduler.update(stack.runner.execute(step))
    raise AssertionError("no decode step")


def _add(stack: TinyStack, temperature: float) -> None:
    for index, length in enumerate((5, 11, 3)):
        stack.scheduler.add_request(
            TokenizedRequest(
                request_id=RequestId(f"r{index}"),
                prompt_token_ids=tuple(range(40, 40 + length)),
                sampling=SamplingParams(temperature=temperature, max_tokens=8, seed=index),
            )
        )


@pytest.mark.parametrize(("backends", "dtype"), CASES)
@pytest.mark.parametrize("temperature", [0.0, 0.9], ids=["greedy", "sampled"])
def test_a_replayed_decode_step_waits_only_for_its_tokens(
    backends: tuple[str, ...], dtype: DType, temperature: float
) -> None:
    stack = _stack(backends, dtype, graphs=True)
    _add(stack, temperature)
    step = _decode_step(stack)
    graph = stack.runner.graph
    assert graph is not None
    torch.cuda.synchronize()

    with torch.inference_mode(), no_device_sync(strict=True):
        staged = graph.launch(step)
    tokens = stack.runner.builder.read_sampled(staged.count)  # the declared wait

    eager = _stack(backends, dtype, graphs=False)
    _add(eager, temperature)
    output = eager.runner.execute(_decode_step(eager))
    assert tokens == [drawn[0] for drawn in output.tokens]


def test_the_guard_catches_a_hidden_synchronisation() -> None:
    """The guard is real: a ``.item()`` inside it fails."""
    with pytest.raises(RuntimeError, match="synchroniz"), no_device_sync(strict=True):
        torch.ones(1, device="cuda").sum().item()
