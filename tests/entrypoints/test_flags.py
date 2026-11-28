"""The shared engine flags: what each spelling of ``--cuda-graph`` hands to the config."""

from __future__ import annotations

import pytest

from ayaka.configs.device import DeviceConfig
from ayaka.entrypoints._cli_parsers import build_parser

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("command", ["run", "serve", "plan", "bench"])
def test_every_engine_command_takes_both_spellings(command: str) -> None:
    parser = build_parser()
    for argv, flag in (((), None), (("--cuda-graph",), True), (("--no-cuda-graph",), False)):
        assert parser.parse_args([command, "m", *argv]).cuda_graph is flag


def _graph_flag(*argv: str) -> bool | None:
    return build_parser().parse_args(["run", "m", *argv]).cuda_graph


@pytest.mark.parametrize(
    ("argv", "flag", "on_gpu"),
    [((), None, True), (("--cuda-graph",), True, True), (("--no-cuda-graph",), False, False)],
    ids=["default", "on", "off"],
)
def test_the_graph_flag_is_on_by_default_on_a_gpu_and_can_be_turned_off(
    argv: tuple[str, ...], flag: bool | None, on_gpu: bool
) -> None:
    parsed = _graph_flag(*argv)

    assert parsed is flag
    assert DeviceConfig(device="cuda", cuda_graph=parsed).uses_cuda_graph is on_gpu


def test_the_default_leaves_the_cpu_eager() -> None:
    assert not DeviceConfig(device="cpu", cuda_graph=_graph_flag()).uses_cuda_graph
