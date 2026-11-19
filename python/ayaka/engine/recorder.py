"""The session recorder: what each request was and what it produced, as token ids.

``ayaka serve --record DIR`` and ``ayaka bench --record DIR`` give a :class:`SessionRecorder` to
the engine when they build it. Every request that finishes becomes one JSON line in
``DIR/session-<UTC time>.jsonl``: its prompt token ids, the sampling parameters, the output token
ids of each sequence, the finish reasons, the times (arrival, first token, end) and what the
prefix cache gave it.

The records hold the session's content in token form, so the recorder is off by default, writes
only to a directory the user names, makes its file owner-only on POSIX and logs counts only. The
format is versioned (:data:`FORMAT`); :func:`read_session` reads a file back and refuses another
format. Replaying a recording is the memory milestone's work; this milestone writes and reads.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from ayaka.core.outputs import RequestOutput
    from ayaka.core.params import SamplingParams
    from ayaka.core.request import TokenizedRequest

__all__ = ["FORMAT", "RecordedRequest", "SessionRecorder", "read_session"]

#: The format of a session file; a reader refuses any other.
FORMAT: Final = "ayaka.session/1"


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    """One finished request, as a session file holds it.

    Attributes:
        request_id: The engine's id.
        prompt_token_ids: The prompt the model read.
        sampling: The sampling parameters, as JSON values.
        outputs: Each sequence's ``index``, ``token_ids`` and ``finish_reason``.
        arrival: When the request was admitted, in seconds of the engine clock.
        first_token: When its first token was produced, or ``None``.
        finished: When it ended.
        prompt_tokens: The prompt's length as the usage reports it.
        cached_tokens: How many prompt tokens the prefix cache gave.
        completion_tokens: Tokens produced, over every sequence.
        error: The error's code when the request failed.
    """

    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling: Mapping[str, object]
    outputs: tuple[Mapping[str, object], ...]
    arrival: float
    first_token: float | None
    finished: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    error: str | None = None


def _jsonable(value: object) -> object:
    """A sampling value as JSON: sets sorted, tuples and mappings as lists and objects."""
    if isinstance(value, Enum):
        # A plain value as itself; a structured one (DType's tuple) by its label or name.
        plain = value.value
        if isinstance(plain, (str, int, float)):
            return plain
        return str(getattr(value, "label", value.name))
    if isinstance(value, (frozenset, set)):
        return sorted(cast("Iterable[int]", value))
    if isinstance(value, dict):
        items = cast("dict[object, object]", value).items()
        return {str(key): _jsonable(item) for key, item in items}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in cast("Iterable[object]", value)]
    return value


def sampling_json(sampling: SamplingParams) -> dict[str, object]:
    """Every field of the sampling parameters as JSON values."""
    return {item.name: _jsonable(getattr(sampling, item.name)) for item in fields(sampling)}


@dataclass(slots=True)
class _Open:
    prompt: tuple[int, ...]
    sampling: dict[str, object]
    arrival: float
    first_token: float | None = None
    tokens: dict[int, list[int]] = field(default_factory=dict[int, list[int]])


class SessionRecorder:
    """Writes one JSON line per finished request; the engine calls it, nothing else does.

    Args:
        directory: Where the session file goes; created if missing.
        clock: The engine's clock, for the times.
        name: The file's name; ``session-<UTC time>.jsonl`` by default.
    """

    def __init__(
        self,
        directory: Path,
        *,
        clock: Callable[[], float] = time.perf_counter,
        name: str | None = None,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = directory / (name or f"session-{stamp}.jsonl")
        self._clock = clock
        self._open: dict[str, _Open] = {}
        self._lock = threading.Lock()
        self.count = 0
        self.recording = True
        """Whether requests admitted now are recorded; the server pauses it for the warm-up."""
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        descriptor = os.open(self.path, flags, 0o600)  # owner-only from the first byte on POSIX
        self._stream = os.fdopen(descriptor, "a", encoding="utf-8")
        self._write({"format": FORMAT, "created": datetime.now(UTC).isoformat()})

    def admitted(self, request: TokenizedRequest) -> None:
        """A request entered the engine."""
        if not self.recording:
            return
        with self._lock:
            self._open[str(request.request_id)] = _Open(
                prompt=tuple(request.prompt_token_ids),
                sampling=sampling_json(request.sampling),
                arrival=self._clock(),
            )

    def observed(self, output: RequestOutput) -> None:
        """An output left the engine: collect its tokens, write the record when it finished."""
        key = str(output.request_id)
        with self._lock:
            state = self._open.get(key)
            if state is None:
                return
            for sequence in output.outputs:
                if sequence.token_ids and state.first_token is None:
                    state.first_token = self._clock()
                state.tokens.setdefault(int(sequence.index), []).extend(sequence.token_ids)
            if not output.finished:
                return
            del self._open[key]
            reasons = {int(s.index): s.finish_reason for s in output.outputs}
            record = RecordedRequest(
                request_id=key,
                prompt_token_ids=state.prompt,
                sampling=state.sampling,
                outputs=tuple(
                    {
                        "index": index,
                        "token_ids": tokens,
                        "finish_reason": _jsonable(reasons.get(index)),
                    }
                    for index, tokens in sorted(state.tokens.items())
                ),
                arrival=state.arrival,
                first_token=state.first_token,
                finished=self._clock(),
                prompt_tokens=output.usage.prompt_tokens,
                cached_tokens=output.usage.cached_tokens,
                completion_tokens=output.usage.completion_tokens,
                error=output.error.code if output.error is not None else None,
            )
            self._write({"request": _record_json(record)})
            self.count += 1

    def close(self) -> None:
        """Flush and close the file; later records are dropped."""
        with self._lock:
            if not self._stream.closed:
                self._stream.close()

    def _write(self, line: Mapping[str, object]) -> None:
        if self._stream.closed:
            return
        self._stream.write(json.dumps(line, separators=(",", ":")) + "\n")
        self._stream.flush()


def _record_json(record: RecordedRequest) -> dict[str, object]:
    return {item.name: _jsonable(getattr(record, item.name)) for item in fields(record)}


def read_session(path: Path) -> list[RecordedRequest]:
    """The records of a session file.

    Raises:
        ValueError: If the file is not a session file of :data:`FORMAT`.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")
    header = cast("dict[str, object]", json.loads(lines[0]))
    if header.get("format") != FORMAT:
        raise ValueError(f"{path} is {header.get('format')!r}, not {FORMAT}")
    records: list[RecordedRequest] = []
    for line in lines[1:]:
        data = cast("dict[str, object]", json.loads(line)["request"])
        records.append(
            RecordedRequest(
                request_id=str(data["request_id"]),
                prompt_token_ids=tuple(cast("list[int]", data["prompt_token_ids"])),
                sampling=cast("dict[str, object]", data["sampling"]),
                outputs=tuple(cast("list[dict[str, object]]", data["outputs"])),
                arrival=float(cast("float", data["arrival"])),
                first_token=cast("float | None", data["first_token"]),
                finished=float(cast("float", data["finished"])),
                prompt_tokens=int(cast("int", data["prompt_tokens"])),
                cached_tokens=int(cast("int", data["cached_tokens"])),
                completion_tokens=int(cast("int", data["completion_tokens"])),
                error=cast("str | None", data.get("error")),
            )
        )
    return records
