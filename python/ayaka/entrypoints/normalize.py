"""Normalising the churn coding clients inject into their prompts, before tokenization.

A prefix cache only hits when a request starts with token ids the cache already holds. Coding
clients put request-varying values near the top of their fixed prompt -- a billing header with
the client's build, a timestamp, a session id -- so two sessions of the same tool, or of two
versions of it, miss everything after the first differing token.

A :class:`NormalizeRule` is data: a field of the conversation, a pattern and a stable
replacement. A :class:`Normalizer` applies the rules of one recognised client to the
conversation before it is tokenized:

* only the field a rule names is touched -- user, assistant and tool-result content never is;
* a replacement is deterministic and stable across requests, never an empty string, so token
  boundaries stay where they were and no content hides;
* a rule is idempotent: applying it twice reads the same as once;
* a rule that matches nothing, or that throws, is a no-op -- normalization never rejects a
  request;
* a client is recognised from the headers it actually sends (see :func:`identify_client`,
  checked against the recordings of #50); an unknown client gets no rules.

The rules here come from the recorded sessions: the values a rule rewrites are metadata a tool
reports about itself, not text a tool's model will act on -- a path the model is meant to open
with a tool must not be rewritten, and no rule here touches one.

The server builds one :class:`Normalizer` at start-up (``ServerConfig.normalize_prompts``,
``--no-normalize`` turns every rule off) and counts the replacements per rule in
``ayaka_normalized_replacements``; the log says rule id, count and client, never content.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ayaka.entrypoints.conversation import Conversation, ToolSpec, Turn

__all__ = ["RULES", "AppliedRule", "NormalizeRule", "Normalizer", "identify_client"]

type RuleField = Literal["system", "developer", "tool_description"]

_LOG = logging.getLogger(__name__)


class _Counter(Protocol):
    """The metrics counter a server wires in: labelled, and its series increments."""

    def labels(self, *labels: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class NormalizeRule:
    """One churn pattern of one client, and what replaces it.

    Attributes:
        rule_id: The rule's name in logs, metrics and reviews, versioned.
        client: The client the rule applies to, as :func:`identify_client` spells it.
        field: Which part of the conversation the pattern may touch: the system prompt
            (``system``, the prefix the conversation carries), the later system turns a
            developer message became (``developer``), or the tools' descriptions.
        pattern: What churns. Reviewed by hand: it runs on the client's fixed prompt, and its
            cost is bounded by that text, not by user content.
        replacement: The stable text that takes the churn's place; contains the rule's id, so a
            review sees where it came from and applying it twice changes nothing.
        version: The rule's own version, bumped when its pattern or replacement changes.
    """

    rule_id: str
    client: str
    field: RuleField
    pattern: re.Pattern[str]
    replacement: str
    version: int = 1


@dataclass(frozen=True, slots=True)
class AppliedRule:
    """What one request's normalization did with one rule."""

    rule_id: str
    count: int


# -- the rules the recorded sessions justify ---------------------------------------------------

#: Claude Code opens its system prompt with a billing header whose values change with every
#: session, not only with the client's version: three sessions of 2.1.327 sent
#: ``cc_version=2.1.327.a8f``, ``.e78`` and ``.e0f`` (``cc_entrypoint=sdk-cli;`` after it). They
#: are metadata the model never acts on, and everything after them was the same, so without the
#: rule a new session reads only the block before the header from the cache (measured in
#: docs/EXPERIMENTS.md: 0.2% without, 84.5% with it).
_CLAUDE_BILLING: Final[NormalizeRule] = NormalizeRule(
    rule_id="claude-code.billing-header",
    client="claude-code",
    field="system",
    # the field name stays; only the build and entrypoint values go
    pattern=re.compile(
        r"(?<=x-anthropic-billing-header: )cc_version=[^;\n]*; cc_entrypoint=[^;\n]*;"
    ),
    replacement="[ayaka:normalized:claude-code.billing-header:v1]",
)

#: The Codex CLI lists its skills with the paths of its per-session home, which carry the
#: session's id. Those paths are what the model is told to open with its tools, so rewriting
#: them would break the calls: no rule, recorded as a deliberate omission (see the module
#: docstring). OpenCode's recorded sessions showed no churn in a normalizable field.

RULES: Final[tuple[NormalizeRule, ...]] = (_CLAUDE_BILLING,)

#: How a client is recognised: the prefixes its ``user-agent`` starts with, checked against the
#: recordings of #50. The headers a client sends besides it decide when the agent is not the
#: first product of its vendor.
_CLIENTS: Final[tuple[tuple[str, tuple[str, ...], Mapping[str, str]], ...]] = (
    ("claude-code", ("claude-cli/",), {}),
    ("codex", ("codex_exec/", "codex-cli/"), {"originator": "codex_exec"}),
    ("opencode", ("opencode/",), {"x-opencode-client": "cli"}),
)


def identify_client(headers: Mapping[str, str]) -> str | None:
    """The client a request comes from, or ``None`` when no record matches it.

    The lookup is case-insensitive, like the HTTP headers that carry them. A client's own
    ``user-agent`` decides; the recorded extra headers identify a client only when its agent
    does not say so.
    """
    lowered = {key.lower(): value for key, value in headers.items()}
    agent = lowered.get("user-agent", "")
    for client, prefixes, extra in _CLIENTS:
        if any(agent.startswith(prefix) for prefix in prefixes):
            return client
        if extra and all(lowered.get(key) == value for key, value in extra.items()):
            return client
    return None


@dataclass(slots=True)
class Normalizer:
    """Applies the rules of one client to a conversation, before tokenization.

    Attributes:
        rules: The rule table, :data:`RULES` by default.
        enabled: ``ServerConfig.normalize_prompts``; ``False`` disables every rule at once.
        metrics: The counter that counts the replacements per rule and client, when the server
            has metrics.
    """

    rules: tuple[NormalizeRule, ...] = RULES
    enabled: bool = True
    metrics: _Counter | None = None

    def normalize(
        self, conversation: Conversation, client: str | None, *, request_id: str = ""
    ) -> tuple[Conversation, tuple[AppliedRule, ...]]:
        """The conversation with ``client``'s rules applied, and what was replaced.

        An unknown client, a disabled normalizer or a rule that matches nothing returns the
        conversation unchanged. A rule that throws is skipped: normalization never rejects a
        request.
        """
        if not self.enabled or client is None:
            return conversation, ()
        applied: list[AppliedRule] = []
        system = conversation.system
        turns = conversation.turns
        tools = conversation.tools
        for rule in self.rules:
            if rule.client != client:
                continue
            try:
                count = 0
                if rule.field == "system":
                    if system is None:
                        continue
                    system, count = _substitute(rule, system)
                elif rule.field == "developer":
                    replaced_turns: list[Turn] = []
                    for turn in turns:
                        if turn.role == "system":
                            text, n = _substitute(rule, turn.text)
                            count += n
                            turn = replace(turn, text=text)
                        replaced_turns.append(turn)
                    turns = tuple(replaced_turns)
                else:
                    replaced_tools: list[ToolSpec] = []
                    for tool in tools:
                        if tool.description is not None:
                            text, n = _substitute(rule, tool.description)
                            count += n
                            tool = replace(tool, description=text)
                        replaced_tools.append(tool)
                    tools = tuple(replaced_tools)
            except Exception:  # a rule must never break a request; skip it and say why
                _LOG.debug(
                    "normalize %s: the rule failed and was skipped", rule.rule_id, exc_info=True
                )
                continue
            if not count:
                continue
            applied.append(AppliedRule(rule_id=rule.rule_id, count=count))
            self._count(rule, client)
            _LOG.info(
                "normalize %s: %d replacement(s) for %s (request %s)",
                rule.rule_id,
                count,
                client,
                request_id or "-",
            )
        if not applied:
            return conversation, ()
        return replace(conversation, system=system, turns=turns, tools=tools), tuple(applied)

    def _count(self, rule: NormalizeRule, client: str) -> None:
        if self.metrics is not None:
            self.metrics.labels(rule.rule_id, client).inc()


def _substitute(rule: NormalizeRule, text: str) -> tuple[str, int]:
    """``text`` with the rule's pattern replaced, and how many replacements that made."""
    replaced, count = rule.pattern.subn(rule.replacement, text)
    return replaced, count
