"""The session recorder: a finished request becomes one versioned line, read back the same."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId
from ayaka.engine.recorder import FORMAT, SessionRecorder, read_session

torch: Any = pytest.importorskip("torch")

from ayaka.engine.llm import LLM  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    TinyCheckpoint,
    prompt_ids,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("recorder"))


def _run(checkpoint: TinyCheckpoint, directory: Path, *, stream: bool) -> tuple[Any, Path]:
    recorder = SessionRecorder(directory)
    with LLM.from_config(
        tiny_engine_config(checkpoint.directory), tokenizer=tiny_tokenizer(), recorder=recorder
    ) as llm:
        prompt = tuple(prompt_ids("the recorder keeps token ids"))
        request = GenerateRequest(
            request_id=RequestId("rec-1"),
            prompt=prompt,
            # top_k keeps the draws among the ids the byte tokenizer can decode.
            sampling=SamplingParams(
                temperature=0.8, top_k=4, seed=1, max_tokens=6, ignore_eos=True, n=2
            ),
        )
        outputs = list(llm.stream(request)) if stream else [llm.generate(request)]
    return outputs, recorder.path


@pytest.mark.parametrize("stream", [False, True])
def test_a_request_round_trips_with_its_tokens_and_cache_stats(
    checkpoint: TinyCheckpoint, tmp_path: Path, stream: bool
) -> None:
    outputs, path = _run(checkpoint, tmp_path, stream=stream)

    [record] = read_session(path)
    final = outputs[-1]
    produced: dict[int, list[int]] = {}
    for out in outputs:
        for sequence in out.outputs:
            produced.setdefault(int(sequence.index), []).extend(sequence.token_ids)
    assert record.request_id == "rec-1"
    assert record.prompt_token_ids == tuple(prompt_ids("the recorder keeps token ids"))
    assert {_index(o): o["token_ids"] for o in record.outputs} == produced
    assert sorted(produced) == [0, 1]
    assert all(len(tokens) == 6 for tokens in produced.values())
    assert record.sampling["max_tokens"] == 6
    assert record.sampling["n"] == 2
    assert record.completion_tokens == final.usage.completion_tokens
    assert record.prompt_tokens == final.usage.prompt_tokens
    assert record.first_token is not None
    assert record.arrival <= record.first_token <= record.finished


def _index(output: Any) -> int:
    return int(output["index"])


def test_the_file_is_versioned_and_owner_only(checkpoint: TinyCheckpoint, tmp_path: Path) -> None:
    _, path = _run(checkpoint, tmp_path, stream=False)

    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert header["format"] == FORMAT
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


def test_another_format_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.jsonl"
    path.write_text('{"format": "something/9"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"not ayaka\.session/1"):
        read_session(path)
