"""The ``prediction`` field of an OpenAI chat request (predicted outputs)."""

from __future__ import annotations

from typing import cast

from ayaka.core.errors import InvalidRequestError
from ayaka.entrypoints.wire.reader import JsonObject

__all__ = ["parse_prediction"]


def parse_prediction(value: JsonObject | None) -> str | None:
    """The text of a chat request's ``prediction`` field, or ``None`` when absent or empty.

    OpenAI's predicted outputs: ``{"type": "content", "content": "..."}``, the content a string
    or an array of text parts. Ayaka reads it only as a draft source for speculative decoding.

    Raises:
        InvalidRequestError: For another type, or content that is not text.
    """
    if value is None:
        return None
    kind = value.string("type")
    if kind != "content":
        raise InvalidRequestError(
            "prediction.type: must be 'content'", context={"param": "prediction"}
        )
    content = value.raw("content")
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts: list[str] = []
        for idx, item in enumerate(cast("list[object]", content)):
            part = JsonObject.of(item, f"prediction.content[{idx}]")
            if part.string("type") != "text":
                raise InvalidRequestError(
                    "prediction.content: array items must have type 'text'",
                    context={"param": "prediction"},
                )
            parts.append(part.text("text"))
        joined = "".join(parts)
        return joined or None
    raise InvalidRequestError(
        "prediction.content: must be a string or an array of text parts",
        context={"param": "prediction"},
    )
