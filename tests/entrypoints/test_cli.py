"""The command line, through ``main(argv)`` on a tiny checkpoint written by the test.

The byte-level fake tokenizer stands in for the real adapters (``patch_tokenizer_loading``), so a
whole run, from the flags to the printed text, happens on a CPU with no extra installed. The oracle
is ``greedy_reference``: GPT-2's formulas with no cache, decoded by the fake tokenizer.
"""

from __future__ import annotations

import io
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import CapabilityError, ErrorInfo
from ayaka.core.outputs import CompletionOutput, RequestOutput
from ayaka.core.types import FinishReason, RequestId
from ayaka.entrypoints.cli import main
from ayaka.scheduler.step import StepOutput
from ayaka.tokenizers.factory import DefaultTokenizerFactory
from tokenizer_fakes import BOS, FakeTokenizer

torch: Any = pytest.importorskip("torch")

from ayaka.engine.llm import LLM  # noqa: E402
from ayaka.execution.runner import ModelRunner  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    MAX_POSITIONS,
    TinyCheckpoint,
    greedy_reference,
    patch_tokenizer_loading,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

_FAKE = FakeTokenizer()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("cli-gpt2"))


@pytest.fixture(autouse=True)
def tiny_tokenizer_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_tokenizer_loading(monkeypatch)


def _expected(checkpoint: TinyCheckpoint, prompt: str, count: int) -> str:
    ids = [BOS, *prompt.encode()]
    tokens = greedy_reference(checkpoint.state, checkpoint.config, ids, count)
    return _FAKE.decode(tokens, skip_special_tokens=True)


def _run(checkpoint: TinyCheckpoint, *flags: str) -> list[str]:
    return ["run", str(checkpoint.directory), "--offline", *flags]


def _fails(capsys: pytest.CaptureFixture[str], argv: list[str], code: int) -> str:
    """Run the command, check its exit code and that no traceback was printed; return stderr."""
    assert main(argv) == code
    err = capsys.readouterr().err
    assert "Traceback" not in err
    return err


class TestRun:
    def test_it_prints_the_generated_text(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(_run(checkpoint, "-p", "hello", "--max-tokens", "10", "-q"))

        captured = capsys.readouterr()
        assert code == 0
        assert captured.out == _expected(checkpoint, "hello", 10) + "\n"
        assert captured.err == ""

    def test_the_start_up_line_names_the_values_in_use_and_goes_to_stderr(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(_run(checkpoint, "-p", "hi", "--max-tokens", "2", "--dtype", "fp32"))

        err = capsys.readouterr().err
        assert err.startswith("ayaka: running ")
        assert "on cpu in fp32" in err
        assert "ops backends: torch" in err

    def test_the_prompt_may_come_from_standard_input(
        self,
        checkpoint: TinyCheckpoint,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO("piped prompt\n"))

        code = main(_run(checkpoint, "--max-tokens", "6", "-q"))

        assert code == 0
        assert capsys.readouterr().out == _expected(checkpoint, "piped prompt", 6) + "\n"

    def test_a_stop_text_cuts_the_output(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        text = _expected(checkpoint, "stop", 16)
        stop = text[5:9]

        main(_run(checkpoint, "-p", "stop", "--max-tokens", "16", "--stop", stop, "-q"))

        assert capsys.readouterr().out == text[: text.index(stop)] + "\n"

    def test_chat_mode_renders_the_template(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        template = b"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        tokens = greedy_reference(checkpoint.state, checkpoint.config, list(template), 5)

        code = main(_run(checkpoint, "-p", "hi", "--chat", "--max-tokens", "5", "-q"))

        assert code == 0
        assert capsys.readouterr().out == _FAKE.decode(tokens, skip_special_tokens=True) + "\n"

    def test_a_seed_makes_sampling_repeatable(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = ("-p", "sample", "--temperature", "1.0", "--max-tokens", "12", "-q")

        main(_run(checkpoint, *base, "--seed", "5"))
        first = capsys.readouterr().out
        main(_run(checkpoint, *base, "--seed", "5"))
        second = capsys.readouterr().out
        main(_run(checkpoint, *base, "--seed", "6"))
        other = capsys.readouterr().out

        assert first == second
        assert first != other

    def test_a_text_that_already_ends_in_a_newline_is_not_given_a_second(
        self,
        checkpoint: TinyCheckpoint,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The tiny model never writes a newline, so end the stream in one by hand.
        done = CompletionOutput(index=0, text="done\n", finish_reason=FinishReason.STOP)
        stream = [RequestOutput(request_id=RequestId("x"), outputs=(done,), finished=True)]
        monkeypatch.setattr(LLM, "stream", lambda self, prompt, sampling=None: iter(stream))

        assert main(_run(checkpoint, "-p", "x", "-q")) == 0

        assert capsys.readouterr().out == "done\n"


class TestErrors:
    """A user error is one line on stderr and a code; never a traceback."""

    def test_a_model_that_is_not_there_exits_3(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = ["run", str(tmp_path / "nowhere"), "-p", "x", "--offline", "-q"]

        err = _fails(capsys, argv, 3)

        assert err.startswith("ayaka: error: ")
        assert "nowhere" in err

    def test_a_tokenizer_that_cannot_load_exits_3(
        self,
        checkpoint: TinyCheckpoint,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def broken(cls: Any, config: Any) -> None:
            raise OSError("no tokenizer.json here")

        monkeypatch.setattr(DefaultTokenizerFactory, "from_config", classmethod(broken))

        err = _fails(capsys, _run(checkpoint, "-p", "x", "-q"), 3)

        assert "no tokenizer.json here" in err

    def test_a_missing_package_exits_4_and_says_how_to_install_it(
        self,
        checkpoint: TinyCheckpoint,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def missing(cls: Any, config: Any) -> None:
            raise CapabilityError(
                "tokenizer.hf", detail="cannot import transformers", remedy="uv sync --extra hf"
            )

        monkeypatch.setattr(DefaultTokenizerFactory, "from_config", classmethod(missing))

        err = _fails(capsys, _run(checkpoint, "-p", "x", "-q"), 4)

        assert "cannot import transformers" in err
        assert "fix: uv sync --extra hf" in err

    def test_a_cache_that_cannot_fit_exits_5(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = _run(checkpoint, "-p", "x", "--memory-ratio", "0.000000001", "-q")

        err = _fails(capsys, argv, 5)

        assert "short" in err

    def test_a_prompt_that_does_not_fit_the_context_exits_2(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = _run(checkpoint, "-p", "x" * MAX_POSITIONS, "--max-tokens", "4", "-q")

        err = _fails(capsys, argv, 2)

        assert "ayaka: error:" in err

    def test_a_device_that_is_not_cpu_or_cuda_exits_2(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        err = _fails(capsys, _run(checkpoint, "-p", "x", "--device", "tpu", "-q"), 2)

        assert "INVALID_DEVICE" in err

    def test_an_unregistered_backend_exits_2(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = _run(checkpoint, "-p", "x", "--backends", "nope,torch", "-q")

        err = _fails(capsys, argv, 2)

        assert "not registered: ['nope']" in err

    def test_cuda_graphs_on_the_cpu_exit_2(
        self, checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
    ) -> None:
        err = _fails(capsys, _run(checkpoint, "-p", "x", "--cuda-graph", "-q"), 2)

        assert "GRAPH_NEEDS_CUDA" in err

    def test_no_prompt_on_a_terminal_exits_2(
        self,
        checkpoint: TinyCheckpoint,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class Terminal(io.StringIO):
            def isatty(self) -> bool:
                return True

        monkeypatch.setattr(sys, "stdin", Terminal())

        err = _fails(capsys, _run(checkpoint, "-q"), 2)

        assert "no prompt" in err

    def test_an_empty_piped_prompt_exits_2(
        self,
        checkpoint: TinyCheckpoint,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO("  \n"))

        _fails(capsys, _run(checkpoint, "-q"), 2)

    def test_a_generation_that_fails_in_the_engine_exits_6(
        self,
        checkpoint: TinyCheckpoint,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def failing(self: ModelRunner, step: Any, drafts: Any = None) -> StepOutput:
            return StepOutput(
                step_id=step.step_id,
                request_ids=step.request_ids,
                error=ErrorInfo("backend_error", "kernel said no"),
            )

        monkeypatch.setattr(ModelRunner, "execute", failing)

        err = _fails(capsys, _run(checkpoint, "-p", "x", "--max-tokens", "4", "-q"), 6)

        assert "kernel said no" in err

    def test_an_unexpected_exception_exits_1_and_hides_the_traceback_unless_asked(
        self,
        checkpoint: TinyCheckpoint,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("a bug")

        monkeypatch.setattr(LLM, "from_config", classmethod(boom))

        err = _fails(capsys, _run(checkpoint, "-p", "x", "-q"), 1)
        assert "unexpected RuntimeError: a bug" in err
        assert "-v" in err
        with pytest.raises(RuntimeError, match="a bug"):
            main(_run(checkpoint, "-p", "x", "-q", "-v"))

    def test_an_interrupt_exits_130(
        self,
        checkpoint: TinyCheckpoint,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def interrupted(*args: object, **kwargs: object) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(LLM, "from_config", classmethod(interrupted))

        _fails(capsys, _run(checkpoint, "-p", "x", "-q"), 130)

    @pytest.mark.parametrize(
        "flags",
        [["--dtype", "fp64"], ["--max-tokens", "many"], ["--temperature"]],
        ids=["dtype", "number", "missing-value"],
    )
    def test_a_bad_flag_is_an_argparse_usage_error(
        self, checkpoint: TinyCheckpoint, flags: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as stop:
            main(_run(checkpoint, "-p", "x", *flags))

        assert stop.value.code == 2
        assert "usage:" in capsys.readouterr().err

    def test_no_command_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main([]) == 2
        assert "no command" in capsys.readouterr().err

    def test_version_and_help_exit_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        for flag in ("--version", "--help"):
            with pytest.raises(SystemExit) as stop:
                main([flag])
            assert stop.value.code == 0
        assert "ayaka" in capsys.readouterr().out


def _python(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-I", *args], capture_output=True, text=True, check=False
    )


class TestAsAProgram:
    def test_help_does_not_load_torch(self) -> None:
        code = (
            "import sys\n"
            "from ayaka.entrypoints.cli import main\n"
            "for argv in (['--help'], ['run', '--help'], ['run', 'm', '--dtype', 'nope']):\n"
            "    try:\n"
            "        main(argv)\n"
            "    except SystemExit:\n"
            "        pass\n"
            "assert 'torch' not in sys.modules, 'torch was imported'\n"
        )

        result = _python("-c", code)

        assert result.returncode == 0, result.stderr

    def test_python_dash_m_ayaka_answers_help_and_version(self) -> None:
        helped = _python("-m", "ayaka", "run", "--help")
        versioned = _python("-m", "ayaka", "--version")

        assert helped.returncode == 0
        assert "--max-tokens" in helped.stdout
        assert "--offline" in helped.stdout
        assert versioned.returncode == 0
        assert versioned.stdout.startswith("ayaka ")

    def test_python_dash_m_ayaka_reports_a_missing_model_without_a_traceback(
        self, tmp_path: Path
    ) -> None:
        result = _python("-m", "ayaka", "run", str(tmp_path / "nowhere"), "-p", "x", "--offline")

        assert result.returncode == 3
        assert result.stdout == ""
        assert result.stderr.splitlines()[-1].startswith("ayaka: error: ")
        assert "Traceback" not in result.stderr

    def test_a_real_checkpoint_without_a_tokenizer_package_fails_cleanly(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        # No tokenizer files are written and the fake is not patched into a child process, so
        # the run stops at the tokenizer: 4 when the adapter package is missing, 3 when it is
        # there and finds no files.
        argv = ["-m", "ayaka", "run", str(checkpoint.directory), "-p", "x", "--offline", "-q"]

        result = _python(*argv)

        assert result.returncode in (3, 4)
        assert "Traceback" not in result.stderr
        assert result.stderr.startswith("ayaka: error: ")
