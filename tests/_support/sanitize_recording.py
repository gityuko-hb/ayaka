"""Turn a recorded request into a fixture that can be committed.

``ayaka connect <tool> --observe --record DIR`` keeps each request whole: the tool's system
prompt, the project files it carries and the session ids of the conversation. That is private,
so raw recordings stay outside the repository (for example ``~/scratch/rec-<tool>``) and only
sanitized fixtures are committed. A fixture keeps the request's *shape* -- the protocol path, the
feature headers, the field names a tool sends, the tool schemas -- and replaces every free text
and identifier that could name a person, a machine or a project with a placeholder.

Run it on a raw recording and write the fixture next to the others::

    uv run python tests/_support/sanitize_recording.py ~/scratch/rec-codex/*.json \\
        --to tests/entrypoints/fixtures/codex-0.163.0

``PRIVATE_MARKERS`` lists what must never survive; ``assert_clean`` checks a fixture against
them, and the replay test for the fixtures runs it on every file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

__all__ = [
    "PRIVATE_MARKERS",
    "REDACTED_ARGUMENTS",
    "REDACTED_ID",
    "REDACTED_TEXT",
    "assert_clean",
    "sanitize",
]

#: Header names a fixture keeps: the ones that say how the tool talks, none that say who it is.
_KEEP_HEADERS: Final = frozenset(
    {
        "accept",
        "anthropic-beta",
        "anthropic-dangerous-direct-browser-access",
        "anthropic-version",
        "content-type",
        "originator",
        "user-agent",
        "x-codex-beta-features",
    }
)
_KEEP_HEADER_PREFIXES: Final = ("x-stainless-",)

#: Text and identifiers a person or a project could be recognised in never survive sanitizing.
PRIVATE_MARKERS: Final = (
    r"[A-Za-z]:\\",  # a Windows path (in the file it is escaped as JSON writes it)
    r"[A-Za-z]:\\\\",
    r"/home/",
    r"/Users/",
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",  # a UUID
    "device_id",
    "account_uuid",
)

#: What replaces redacted free text, so a fixture reads as a fixture and not as an accident.
REDACTED_TEXT: Final = "<redacted: private prompt content>"
REDACTED_TOOL_TEXT: Final = "<redacted: tool description>"
REDACTED_ARGUMENTS: Final = json.dumps({"redacted": True})
REDACTED_ID: Final = "fixture-id"


def _kept_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """The headers a fixture keeps, with the original spelling of the names."""
    kept: dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered in _KEEP_HEADERS or lowered.startswith(_KEEP_HEADER_PREFIXES):
            kept[name] = value
    return kept


def _objects(value: object) -> list[Mapping[str, object]]:
    """The mappings of a list field, or an empty list when it is not one."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _tool(tool: Mapping[str, object]) -> dict[str, object]:
    """One tool definition: names, types and schemas kept, every prose replaced.

    The schema is the tool's API shape -- what the server has to accept -- and stays as it was,
    except for the descriptions inside it, which are product text. A namespace tool holds nested
    definitions; an OpenAI tool nests its fields under ``function``.
    """
    clean: dict[str, object] = {}
    for key, value in tool.items():
        if key == "description" and isinstance(value, str):
            clean[key] = REDACTED_TOOL_TEXT
        elif key == "tools":
            clean[key] = [_tool(nested) for nested in _objects(value)]
        else:
            clean[key] = _walk(value)
    return clean


def _walk(value: object) -> object:
    """``value`` with every nested ``description`` string replaced."""
    if isinstance(value, Mapping):
        return {
            key: REDACTED_TOOL_TEXT
            if key == "description" and isinstance(item, str)
            else _walk(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_walk(item) for item in value]
    return value


def _blocks(content: object) -> list[object]:
    """The content as a list of blocks; a plain string is one block."""
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return content
    return []


def _arguments(value: object) -> object:
    """A tool call's arguments with their values dropped, still in the form the server checks.

    Anthropic sends an object; OpenAI sends the object as JSON text, which the server parses, so
    JSON text stays JSON text (``{"redacted": true}``). Anything else is replaced as free text.
    """
    if isinstance(value, Mapping):
        return {"redacted": True}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return REDACTED_TEXT
        return REDACTED_ARGUMENTS if isinstance(parsed, dict) else REDACTED_TEXT
    return value


def _part(part: object) -> object:
    """One content part or text: its ``type`` kept, its free text and ids replaced."""
    if not isinstance(part, Mapping):
        return REDACTED_TEXT if isinstance(part, str) else part
    clean = dict(part)
    if isinstance(clean.get("text"), str):
        clean["text"] = REDACTED_TEXT
    if "arguments" in clean:
        clean["arguments"] = _arguments(clean["arguments"])
    for key in ("input", "content"):
        value = clean.get(key)
        if isinstance(value, str):
            clean[key] = REDACTED_TEXT
        elif isinstance(value, Mapping):  # a tool call's arguments are an object of values
            clean[key] = {"redacted": True}
    for key in ("id", "tool_use_id", "tool_call_id", "call_id"):
        if isinstance(clean.get(key), str):
            clean[key] = REDACTED_ID
    return clean


def _text_or_parts(value: object) -> object:
    """A free-text field: a string replaced, a list of parts replaced part by part."""
    if isinstance(value, str):
        return REDACTED_TEXT
    if isinstance(value, list):
        return [_part(part) for part in cast("list[object]", value)]
    return value


def _chat_tool_call(call: Mapping[str, object]) -> dict[str, object]:
    """One Chat Completions ``tool_calls`` entry: its id and arguments replaced, the name kept."""
    clean = cast("dict[str, object]", _part(call))
    function = clean.get("function")
    if isinstance(function, Mapping):
        clean["function"] = _part(cast("Mapping[str, object]", function))
    return clean


def _item(item: Mapping[str, object]) -> dict[str, object]:
    """One message or input item: role and type kept, text and ids replaced.

    The content keeps its own shape: a string stays a string, a block list stays a list. Tool
    round trips carry private text outside ``content`` too: an assistant message's
    ``tool_calls`` (Chat Completions), and the ``arguments`` of a ``function_call``, the
    ``output`` of a ``function_call_output`` and the ``summary`` and ``encrypted_content`` of a
    ``reasoning`` item (Responses).
    """
    clean = dict(item)
    for key in ("id", "tool_use_id", "tool_call_id", "call_id"):
        if isinstance(clean.get(key), str):
            clean[key] = REDACTED_ID
    content = clean.get("content")
    if isinstance(content, str):
        clean["content"] = REDACTED_TEXT
    elif isinstance(content, list):
        clean["content"] = [_part(part) for part in content]
    if "arguments" in clean:
        clean["arguments"] = _arguments(clean["arguments"])
    for key in ("output", "summary"):
        if key in clean:
            clean[key] = _text_or_parts(clean[key])
    if isinstance(clean.get("encrypted_content"), str):
        clean["encrypted_content"] = REDACTED_TEXT
    if isinstance(clean.get("tool_calls"), list):
        clean["tool_calls"] = [_chat_tool_call(call) for call in _objects(clean["tool_calls"])]
    return clean


def sanitize(recording: Mapping[str, object]) -> dict[str, object]:
    """A recorded request with everything private replaced; the shape stays."""
    raw_headers = recording.get("headers")
    raw_body = recording.get("body")
    body = dict(cast("Mapping[str, object]", raw_body if isinstance(raw_body, Mapping) else {}))
    clean: dict[str, object] = {
        "path": recording.get("path"),
        "headers": _kept_headers(
            cast("Mapping[str, str]", raw_headers if isinstance(raw_headers, Mapping) else {})
        ),
        "body": body,
    }

    # Fields that say which session this was, wherever the tool put them.
    for field in ("metadata", "client_metadata"):
        sessions = body.get(field)
        if isinstance(sessions, Mapping):
            body[field] = dict.fromkeys(sessions, REDACTED_ID)
    if isinstance(body.get("prompt_cache_key"), str):
        body["prompt_cache_key"] = REDACTED_ID

    if isinstance(body.get("system"), (str, list)):
        body["system"] = [_part(part) for part in _blocks(body.get("system"))]
    if isinstance(body.get("instructions"), str) and body["instructions"]:
        body["instructions"] = REDACTED_TEXT

    if isinstance(body.get("messages"), list):
        body["messages"] = [_item(item) for item in _objects(body.get("messages"))]
    if isinstance(body.get("input"), list):
        body["input"] = [_item(item) for item in _objects(body.get("input"))]
    if isinstance(body.get("tools"), list):
        body["tools"] = [_tool(tool) for tool in _objects(body.get("tools"))]

    return clean


# -- what must never survive ------------------------------------------------------------------


def assert_clean(fixture: Mapping[str, object]) -> list[str]:
    """The markers found in a fixture, empty when it may be committed."""
    text = json.dumps(fixture, ensure_ascii=False)
    return [marker for marker in PRIVATE_MARKERS if re.search(marker, text)]


def main(argv: Sequence[str] | None = None) -> int:
    """Sanitize recordings into ``--to``, refusing to write anything a marker survives."""
    parser = argparse.ArgumentParser(
        description="Turn recorded requests (ayaka connect --record) into committed fixtures: "
        "shape kept, private content replaced."
    )
    parser.add_argument("recordings", nargs="+", type=Path, help="recorded JSON files")
    parser.add_argument("--to", type=Path, required=True, metavar="DIR", help="the fixtures dir")
    args = parser.parse_args(argv)

    args.to.mkdir(parents=True, exist_ok=True)
    for source in args.recordings:
        clean = sanitize(json.loads(source.read_text(encoding="utf-8")))
        dirty = assert_clean(clean)
        if dirty:
            print(f"{source}: private content survived: {dirty}", file=sys.stderr)
            return 1
        target = args.to / source.name
        target.write_text(json.dumps(clean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{source} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
