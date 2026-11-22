"""Every recorded conversation renders the same from its values as from its JSON.

The server renders :meth:`Conversation.template_data` in its own process; before, it rendered the
JSON of :meth:`Conversation.template_input`. Both are checked against each other on every recorded
request, with the fake tokenizer and with a real Hugging Face tokenizer whose template writes the
tools and the call arguments with ``tojson``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ayaka.entrypoints.anthropic.protocol import parse_messages_request
from ayaka.entrypoints.openai.protocol import parse_chat_request
from ayaka.entrypoints.wire.reader import JsonObject
from ayaka.tokenizers.service import TokenizerService
from real_tokenizers import build_hf_dir, hf_config
from tokenizer_fakes import make_service

if TYPE_CHECKING:
    from ayaka.entrypoints.conversation import Conversation

FIXTURES = Path(__file__).parent / "fixtures"

TOOL_TEMPLATE = (
    "{% if tools %}<|im_start|>system\n{{ tools | tojson }}<|im_end|>\n{% endif %}"
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}"
    "{% for call in m.get('tool_calls') or [] %}"
    "{{ call['function']['name'] }}{{ call['function']['arguments'] | tojson }}"
    "{% endfor %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _conversations() -> list[tuple[str, Conversation]]:
    found: list[tuple[str, Conversation]] = []
    for path in sorted(FIXTURES.glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or "body" not in record or "path" not in record:
            continue
        name = f"{path.parent.name}/{path.name}"
        body = JsonObject(record["body"])
        route = str(record["path"]).split("?", 1)[0]
        if route == "/v1/messages":
            found.append((name, parse_messages_request(body).conversation))
        elif route == "/v1/chat/completions":
            found.append((name, parse_chat_request(body).conversation))
        elif route == "/v1/responses":
            pytest.importorskip("aiohttp")
            from ayaka.entrypoints.openai.responses import parse_responses_request

            found.append((name, parse_responses_request(body).conversation))
    return found


@pytest.mark.unit
def test_every_protocol_has_a_recording() -> None:
    names = [name for name, _ in _conversations()]

    assert any("anthropic" in name for name in names)
    assert any("openai-chat" in name for name in names)


@pytest.mark.unit
def test_every_recording_renders_the_same_with_the_fake_tokenizer() -> None:
    with make_service() as service:
        for name, conversation in _conversations():
            from_json = service.render_chat(conversation.template_input())
            from_values = service.render_chat(conversation.template_data())
            assert from_values == from_json, name


@pytest.mark.integration
def test_every_recording_renders_the_same_with_a_real_template(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    directory = build_hf_dir(tmp_path / "tok", chat_template=TOOL_TEMPLATE)
    service = TokenizerService(
        hf_config(directory, special_token_policy="escape"),
        max_model_len=1 << 20,
        model_vocab_size=512,
    )
    try:
        conversations = _conversations()
        assert any(conversation.tools for _, conversation in conversations)
        for name, conversation in conversations:
            from_json = service.render_chat(conversation.template_input())
            from_values = service.render_chat(conversation.template_data())
            assert from_values == from_json, name
    finally:
        service.close()
