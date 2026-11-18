"""The prompt normalizer: the client's churn rules, their boundaries and their measured gain.

The rules are data (:mod:`ayaka.entrypoints.normalize`), justified by the recorded sessions of
#50: only the metadata a client reports about itself is rewritten, never text the model will
act on. These tests check the rules against their fixtures, the boundaries (which fields are
touched, idempotence, the off switch, an unknown client), the token positions at the replacement,
the handlers' wiring, and -- on the real engine -- that a second session with a different
billing header reclaims the cached prefix only when normalization is on.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.server import ServerConfig
from ayaka.entrypoints.conversation import Conversation, ConversationBuilder, ToolSpec, Turn
from ayaka.entrypoints.normalize import RULES, Normalizer, NormalizeRule, identify_client

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures" / "normalize"

_BILLING_V1 = "x-anthropic-billing-header: cc_version=2.1.327.a8f; cc_entrypoint=sdk-cli;"
_BILLING_V2 = "x-anthropic-billing-header: cc_version=2.2.0.1b7; cc_entrypoint=terminal;"
_AGENT = "You are a Claude agent, built on Anthropic's Claude Agent SDK."
_REPLACEMENT = "[ayaka:normalized:claude-code.billing-header:v1]"

_SESSION_UUID = "2d6f5abd-5926-5364-9e45-caff4605ae79"

_CLAUDE_HEADERS = {"user-agent": "claude-cli/2.1.327 (external, sdk-cli)"}


def _claude_session(billing: str) -> Conversation:
    """A Claude Code-shaped conversation: the billing line first in the system prompt."""
    builder = ConversationBuilder()
    builder.system(billing)
    builder.system(_AGENT)
    builder.system("the recorded system prompt with its harness rules")
    builder.user("Say hello.")
    return builder.build()


@pytest.fixture(scope="module")
def long_checkpoint() -> Any:
    """A long fixture GPT-2 for the handler tests; only tokenized, the engine is scripted."""
    pytest.importorskip("torch")
    from serving_support import write_long_gpt2

    return write_long_gpt2(Path(tempfile.mkdtemp()), positions=4096)


def _greedy() -> Any:
    from ayaka.core.params import SamplingParams

    return SamplingParams(temperature=0.0, max_tokens=4)


def _find(haystack: tuple[int, ...], needle: tuple[int, ...]) -> int:
    """The first index of ``needle`` in ``haystack``, or -1."""
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


# -- who a request comes from -----------------------------------------------------------------


class TestClientIdentification:
    @pytest.mark.parametrize(
        ("headers", "client"),
        [
            ({"user-agent": "claude-cli/2.1.327 (external, sdk-cli)"}, "claude-code"),
            ({"User-Agent": "codex_exec/0.163.0 (Windows 10.0.26200; x86_64)"}, "codex"),
            ({"user-agent": "other/1.0", "originator": "codex_exec"}, "codex"),
            ({"user-agent": "opencode/latest/2.3.2/cli"}, "opencode"),
            ({"user-agent": "other/1.0", "x-opencode-client": "cli"}, "opencode"),
            ({"user-agent": "openai-python/2.1.0"}, None),
            ({"user-agent": "a-lookalike/9 (claude-cli skin)"}, None),
        ],
        ids=[
            "claude",
            "codex",
            "codex-originator",
            "opencode",
            "opencode-header",
            "unknown",
            "not-a-prefix",
        ],
    )
    def test_the_recorded_user_agents_name_the_client(
        self, headers: dict[str, str], client: str | None
    ) -> None:
        assert identify_client(headers) == client

    def test_the_recordings_identify_their_clients(self) -> None:
        """The header shapes come from the recordings; the fixtures keep them."""
        for name, client in (
            ("claude-code-2.1.327/001-anthropic.json", "claude-code"),
            ("codex-0.163.0/001-openai-responses.json", "codex"),
            ("opencode-2.3.2/002-openai-chat.json", "opencode"),
        ):
            fixture = FIXTURES.parent / name
            headers = json.loads(fixture.read_text(encoding="utf-8"))["headers"]
            assert identify_client(headers) == client, name


# -- the rules --------------------------------------------------------------------------------


class TestRules:
    def test_the_billing_rule_matches_its_fixture(self) -> None:
        before = json.loads((FIXTURES / "claude-code-billing-before.json").read_text("utf-8"))
        after = json.loads((FIXTURES / "claude-code-billing-after.json").read_text("utf-8"))

        normalized, applied = Normalizer(rules=RULES).normalize(
            Conversation(
                system=before["system"],
                turns=tuple(Turn(**turn) for turn in before["turns"]),
                tools=(),
            ),
            identify_client(before["headers"]),
        )

        assert normalized.system == after["system"]
        assert [(rule.rule_id, rule.count) for rule in applied] == [
            ("claude-code.billing-header", 1)
        ]

    def test_only_the_system_field_changes_and_user_content_is_byte_identical(self) -> None:
        conversation = _claude_session(_BILLING_V1)

        normalized, _ = Normalizer(rules=RULES).normalize(conversation, "claude-code")

        assert normalized.system is not None
        assert _BILLING_V1 not in normalized.system
        assert normalized.system.startswith("x-anthropic-billing-header: ")
        assert normalized.system.count(_REPLACEMENT) == 1
        assert normalized.turns == conversation.turns  # the user turn keeps its bytes
        assert normalized.tools == conversation.tools

    def test_applying_twice_changes_nothing_more(self) -> None:
        conversation = _claude_session(_BILLING_V1)
        once, _ = Normalizer(rules=RULES).normalize(conversation, "claude-code")

        twice, applied_again = Normalizer(rules=RULES).normalize(once, "claude-code")

        assert twice == once
        assert applied_again == ()

    def test_sessions_of_two_builds_normalize_to_the_same_prompt(self) -> None:
        """The churn is gone: two sessions of different builds normalize to the same system."""
        first, _ = Normalizer(rules=RULES).normalize(_claude_session(_BILLING_V1), "claude-code")
        second, _ = Normalizer(rules=RULES).normalize(_claude_session(_BILLING_V2), "claude-code")

        assert first.system == second.system
        assert first.system != _claude_session(_BILLING_V1).system

    def test_an_unknown_client_gets_the_original(self) -> None:
        conversation = _claude_session(_BILLING_V1)

        normalized, applied = Normalizer(rules=RULES).normalize(conversation, "some-other-agent")

        assert normalized == conversation
        assert applied == ()

    def test_the_off_switch_returns_the_original_and_logs_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conversation = _claude_session(_BILLING_V1)

        with caplog.at_level(logging.DEBUG):
            normalized, applied = Normalizer(rules=RULES, enabled=False).normalize(
                conversation, "claude-code"
            )

        assert normalized == conversation
        assert applied == ()
        assert "normalize" not in caplog.text

    def test_a_rule_that_matches_nothing_is_a_no_op(self) -> None:
        rule = NormalizeRule(
            rule_id="claude-code.absent",
            client="claude-code",
            field="system",
            pattern=re.compile(r"nothing here"),
            replacement="[ayaka:normalized:absent:v1]",
        )
        conversation = _claude_session(_BILLING_V1)

        normalized, applied = Normalizer(rules=(rule,)).normalize(conversation, "claude-code")

        assert normalized == conversation
        assert applied == ()

    def test_a_rule_that_throws_never_rejects_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rule = NormalizeRule(
            rule_id="claude-code.boom",
            client="claude-code",
            field="system",
            pattern=re.compile(r".*"),
            replacement="[ayaka:normalized:boom:v1]",
        )
        monkeypatch.setattr(
            "ayaka.entrypoints.normalize._substitute",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("bad rule")),
        )
        conversation = _claude_session(_BILLING_V1)

        normalized, applied = Normalizer(rules=(rule,)).normalize(conversation, "claude-code")

        assert normalized == conversation
        assert applied == ()

    def test_a_developer_rule_reaches_the_system_turns_only(self) -> None:
        """A developer message became a system turn; user turns are not touched."""
        rule = NormalizeRule(
            rule_id="codex.session-id",
            client="codex",
            field="developer",
            pattern=re.compile(_SESSION_UUID),
            replacement="[ayaka:normalized:codex.session-id:v1]",
        )
        builder = ConversationBuilder()
        builder.system("the codex system prompt")
        builder.user("first message")
        builder.system(f"skills live in scratchpad\\{_SESSION_UUID}\\skills")
        builder.user("second message")
        conversation = builder.build()

        normalized, applied = Normalizer(rules=(rule,)).normalize(conversation, "codex")

        assert [(rule.rule_id, rule.count) for rule in applied] == [("codex.session-id", 1)]
        assert normalized.turns[0].text == "first message"
        assert normalized.turns[1].role == "system"
        assert _SESSION_UUID not in normalized.turns[1].text
        assert "scratchpad\\" in normalized.turns[1].text
        assert normalized.turns[2].text == "second message"

    def test_a_tool_description_rule_reaches_the_descriptions_only(self) -> None:
        rule = NormalizeRule(
            rule_id="opencode.tool-version",
            client="opencode",
            field="tool_description",
            pattern=re.compile(r"since opencode/2\.3\.2"),
            replacement="[ayaka:normalized:opencode.tool-version:v1]",
        )
        builder = ConversationBuilder()
        builder.system("the opencode system prompt")
        builder.user("prompt")
        builder.tool(ToolSpec("read", "Reads a file, since opencode/2.3.2", {"type": "object"}))
        conversation = builder.build()

        normalized, applied = Normalizer(rules=(rule,)).normalize(conversation, "opencode")

        assert [(rule.rule_id, rule.count) for rule in applied] == [("opencode.tool-version", 1)]
        assert normalized.tools[0].description is not None
        assert "opencode/2.3.2" not in normalized.tools[0].description

    def test_the_replacements_are_counted_per_rule(self) -> None:
        class Counter:
            def __init__(self) -> None:
                self.calls: list[tuple[str, ...]] = []

            def labels(self, *labels: str) -> Any:
                self.calls.append(labels)
                return self

            def inc(self) -> None:
                pass

        counter = Counter()
        normalizer = Normalizer(rules=RULES, metrics=counter)  # type: ignore[arg-type]

        normalizer.normalize(_claude_session(_BILLING_V1), "claude-code", request_id="req-1")
        normalizer.normalize(_claude_session(_BILLING_V2), "claude-code", request_id="req-2")
        normalizer.normalize(_claude_session(_BILLING_V1), "opencode", request_id="req-3")

        assert counter.calls == [
            ("claude-code.billing-header", "claude-code"),
            ("claude-code.billing-header", "claude-code"),
        ]

    def test_the_log_line_names_the_rule_and_never_the_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conversation = _claude_session(_BILLING_V1)

        with caplog.at_level(logging.INFO):
            Normalizer(rules=RULES).normalize(conversation, "claude-code", request_id="req-9")

        assert "claude-code.billing-header" in caplog.text
        assert "req-9" in caplog.text
        assert "cc_version" not in caplog.text  # no churn value, and no prompt text


# -- the token boundary -----------------------------------------------------------------------


class TestTokenBoundary:
    def test_the_changed_tokens_begin_at_the_replacement(self) -> None:
        """The ids of before and after agree up to the region the rule replaced."""
        pytest.importorskip("torch")
        from tokenizer_fakes import FakeTokenizer

        tokenizer = FakeTokenizer()
        text = f"{_BILLING_V1}\n{_AGENT}"
        after, _ = RULES[0].pattern.subn(RULES[0].replacement, text)

        ids_before = tokenizer.encode(text, add_special_tokens=False)
        ids_after = tokenizer.encode(after, add_special_tokens=False)
        offset = text.index("cc_version=")

        shared = 0
        for a, b in zip(ids_before, ids_after, strict=False):
            if a != b:
                break
            shared += 1
        assert shared == offset  # a byte tokenizer: the ids agree up to the replaced region


# -- the handlers apply it --------------------------------------------------------------------


def _messages_body(billing: str) -> dict[str, object]:
    return {
        "model": "tiny",
        "max_tokens": 8,
        "stream": False,
        "system": [{"type": "text", "text": billing}, {"type": "text", "text": _AGENT}],
        "messages": [{"role": "user", "content": "Say hello."}],
    }


class TestHandlers:
    def test_a_claude_request_is_normalized_before_tokenization(
        self, long_checkpoint: Path
    ) -> None:
        pytest.importorskip("aiohttp")
        from serving_support import ScriptedLLM, script, serve
        from tokenizer_fakes import ToolTokenizer, loaded_for

        tokenizer = ToolTokenizer()

        async def run() -> list[tuple[int, ...]]:
            async with serve(
                long_checkpoint,
                config=ServerConfig(),
                tokenizer=loaded_for(tokenizer, stream=False),
                num_blocks=1100,  # one request may hold the whole prompt (4096 positions)
                scripts=[script("Hello!"), script("Hello!")],
            ) as (client, _, llm):
                assert isinstance(llm, ScriptedLLM)
                for billing in (_BILLING_V1, _BILLING_V2):
                    response = await client.post(
                        "/v1/messages", json=_messages_body(billing), headers=_CLAUDE_HEADERS
                    )
                    assert response.status == 200
                prompts = []
                for request in llm.requests:
                    assert isinstance(request.prompt, tuple)
                    prompts.append(request.prompt)
                return prompts

        seen = asyncio.run(run())

        placeholder = tuple(tokenizer.encode(_REPLACEMENT, add_special_tokens=False))
        assert len(seen) == 2
        for prompt in seen:
            position = _find(prompt, placeholder)
            assert position != -1
            assert position < 128  # the replacement sits where the billing line was

    def test_the_off_config_sends_the_prompt_unchanged(self, long_checkpoint: Path) -> None:
        pytest.importorskip("aiohttp")
        pytest.importorskip("llguidance")
        from ayaka.entrypoints.parsers.output import OutputFormat
        from serving_support import ScriptedLLM, script, serve
        from tokenizer_fakes import ToolTokenizer, loaded_for

        tokenizer = ToolTokenizer()

        async def run() -> list[tuple[int, ...]]:
            async with serve(
                long_checkpoint,
                config=ServerConfig(normalize_prompts=False),
                tokenizer=loaded_for(tokenizer, stream=False),
                output_format=OutputFormat.resolve(tool_calls="hermes", tokenizer=tokenizer),
                num_blocks=1100,
                scripts=[script("Hello!"), script("Hello!")],
            ) as (client, _, llm):
                assert isinstance(llm, ScriptedLLM)
                for billing in (_BILLING_V1, _BILLING_V2):
                    response = await client.post(
                        "/v1/messages", json=_messages_body(billing), headers=_CLAUDE_HEADERS
                    )
                    assert response.status == 200
                prompts = []
                for request in llm.requests:
                    assert isinstance(request.prompt, tuple)
                    prompts.append(request.prompt)
                return prompts

        seen = asyncio.run(run())

        assert len(seen) == 2
        assert seen[0][:64] != seen[1][:64]  # the two billing lines reach the prompt ids


# -- the measured gain ------------------------------------------------------------------------


class TestPrefixCacheGain:
    """Two sessions that differ only in the billing line, on the real engine.

    The threshold is written before the run: with normalization, the second session must read
    at least half of its prompt from the cache; without it, less than half. The normalizer runs
    on the conversation, the result is rendered as the template would render it, so this
    measures normalizer plus prefix cache, not the handlers.
    """

    _FILLER = "the recorded system prompt with its harness rules. " * 24

    @pytest.fixture(scope="class")
    @classmethod
    def engine(cls, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
        pytest.importorskip("torch")
        from ayaka.engine.llm import LLM
        from serving_support import write_long_gpt2
        from tiny_model import tiny_engine_config, tiny_tokenizer

        checkpoint = write_long_gpt2(tmp_path_factory.mktemp("normalize-prefix"), positions=4096)
        with LLM.from_config(
            tiny_engine_config(checkpoint, num_blocks=1100), tokenizer=tiny_tokenizer()
        ) as llm:
            yield llm

    @staticmethod
    def _prompt(conversation: Conversation) -> tuple[int, ...]:
        from tiny_model import BOS, prompt_ids

        del BOS
        assert conversation.system is not None
        rendered = (
            f"<|im_start|>system\n{conversation.system}<|im_end|>\n"
            f"<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n"
        )
        return tuple(prompt_ids(rendered))

    @pytest.mark.parametrize("normalize", [True, False], ids=["normalized", "unchanged"])
    def test_a_second_session_of_another_build(self, engine: Any, normalize: bool) -> None:
        first, _ = Normalizer(rules=RULES, enabled=normalize).normalize(
            _claude_session(f"{_BILLING_V1}\n{_AGENT}\n{self._FILLER}"), "claude-code"
        )
        second, _ = Normalizer(rules=RULES, enabled=normalize).normalize(
            _claude_session(f"{_BILLING_V2}\n{_AGENT}\n{self._FILLER}"), "claude-code"
        )
        first_prompt = self._prompt(first)
        second_prompt = self._prompt(second)

        engine.generate(first_prompt, _greedy())
        result = engine.generate(second_prompt, _greedy())

        assert result.usage is not None
        cached = result.usage.cached_tokens
        share = cached / len(second_prompt)
        if normalize:
            assert first_prompt == second_prompt
            assert share >= 0.5, f"cached {cached} of {len(second_prompt)} ({share:.0%})"
        else:
            assert first_prompt != second_prompt
            assert share < 0.5, f"cached {cached} of {len(second_prompt)} ({share:.0%})"
