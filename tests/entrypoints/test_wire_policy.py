"""The wire policy table: every field the recorded clients send has a rule, and the rules hold.

The recorded requests are the bodies Claude Code, Codex and OpenCode really sent
(``tests/entrypoints/fixtures/<tool>-<version>/``); a field the table does not know would be logged
and ignored, so a new client field shows up here first.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from ayaka.core.errors import InvalidRequestError, UnsupportedFeatureError
from ayaka.entrypoints.wire.policy import ENDPOINTS, EndpointPolicy, FieldRule, Rule, apply_policy
from ayaka.entrypoints.wire.reader import JsonObject

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).parent / "fixtures"
_RECORDED = sorted(
    path
    for path in _FIXTURES.glob("*-*/*.json")
    if "path" in json.loads(path.read_text(encoding="utf-8"))
)


def test_there_are_recorded_requests_for_three_protocols() -> None:
    paths = {json.loads(path.read_text(encoding="utf-8"))["path"] for path in _RECORDED}
    assert {path.partition("?")[0] for path in paths} == {
        "/v1/messages",
        "/v1/responses",
        "/v1/chat/completions",
    }


@pytest.mark.parametrize("recording", _RECORDED, ids=lambda path: f"{path.parent.name}/{path.name}")
def test_every_field_of_a_recorded_request_has_a_rule(recording: Path) -> None:
    data = json.loads(recording.read_text(encoding="utf-8"))
    policy = ENDPOINTS[data["path"].partition("?")[0]]

    missing = sorted(set(data["body"]) - policy.names)

    assert missing == [], f"{recording.name}: no rule for {missing} in {policy.path}"


_REFUSALS = [
    (policy.path, rule)
    for policy in ENDPOINTS.values()
    for rule in policy.rules
    if rule.rule in (Rule.REFUSE, Rule.INVALID)
]


@pytest.mark.parametrize(
    ("path", "rule"), _REFUSALS, ids=[f"{path}:{rule.name}" for path, rule in _REFUSALS]
)
def test_a_refused_field_is_refused_with_its_reason(path: str, rule: FieldRule) -> None:
    expected = UnsupportedFeatureError if rule.rule is Rule.REFUSE else InvalidRequestError

    with pytest.raises(expected) as caught:
        apply_policy(JsonObject({rule.name: "x"}), path)

    error = caught.value
    assert type(error) is expected
    assert error.message == f"{rule.name}: {rule.reason}"
    assert error.context["param"] == rule.name


def test_a_null_field_is_absent() -> None:
    apply_policy(JsonObject({"audio": None, "messages": []}), "/v1/chat/completions")


def test_hints_and_unknown_fields_are_logged_and_pass(caplog: pytest.LogCaptureFixture) -> None:
    body = JsonObject({"reasoning_effort": "high", "brand_new_field": 1})

    with caplog.at_level(logging.DEBUG, logger="ayaka.serve"):
        apply_policy(body, "/v1/chat/completions")

    messages = [record.getMessage() for record in caplog.records]
    assert "chat completions: ignoring unknown fields ['brand_new_field']" in messages


def test_the_first_refused_field_in_table_order_is_named() -> None:
    with pytest.raises(UnsupportedFeatureError, match=r"^functions:"):
        apply_policy(JsonObject({"audio": {}, "functions": []}), "/v1/chat/completions")


def test_count_tokens_reads_the_prompt_fields_of_messages_and_ignores_the_rest() -> None:
    messages, count = ENDPOINTS["/v1/messages"], ENDPOINTS["/v1/messages/count_tokens"]
    assert count.names == messages.names
    read = {rule.name for rule in count.rules if rule.rule is Rule.READ}
    assert read == {"model", "messages", "system", "tools", "tool_choice", "n"}


def test_every_generation_route_has_a_policy() -> None:
    assert set(ENDPOINTS) == {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/messages",
        "/v1/messages/count_tokens",
    }


def test_a_field_with_two_rules_is_refused() -> None:
    twice = (FieldRule("a", Rule.READ, "x"), FieldRule("a", Rule.IGNORE, "y"))
    with pytest.raises(ValueError, match="two rules"):
        EndpointPolicy("/x", "x", twice)


@pytest.mark.parametrize(
    "module",
    [
        "ayaka.entrypoints.wire",
        "ayaka.entrypoints.wire.policy",
        "ayaka.entrypoints.openai.protocol",
        "ayaka.entrypoints.anthropic.protocol",
    ],
)
def test_reading_a_request_does_not_load_aiohttp(module: str) -> None:
    code = f"import sys, {module}; print('aiohttp' in sys.modules)"
    result = subprocess.run(  # noqa: S603  # our own interpreter and a fixed program
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
