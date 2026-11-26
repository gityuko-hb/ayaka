"""What the server does with each top-level field of each request it accepts: one table.

Every field a client may send to a generation endpoint has one :class:`FieldRule` here, with a
reason a person can read. The rule says what happens when the field is present and not null:

* ``read``: the server uses it. A value it cannot serve is refused by the protocol's reader, for
  the reason the table gives (``modalities`` other than text, ``n`` other than 1).
* ``hint``: accepted; it does not change the answer. A debug log line says so.
* ``ignore``: accepted and unused.
* ``refuse``: 400 ``unsupported_feature``: the field would change the answer and the server
  cannot do it.
* ``invalid``: 400 ``invalid_request``: the request asks for something this server never does.

:func:`apply_policy` runs those decisions before a reader looks at the values; a field with no
rule is logged at debug and ignored, so a new client field is visible without breaking anything.
:func:`wire_reference` renders the table for ``docs/AI_REFERENCE.md``
(``tools/generate_wire_reference.py`` writes it; ``tests/test_docs.py`` checks it). Only top-level
fields are listed: the readers check what is inside them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from ayaka.core.errors import InvalidRequestError, UnsupportedFeatureError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ayaka.entrypoints.wire.reader import JsonObject

__all__ = [
    "ENDPOINTS",
    "EndpointPolicy",
    "FieldRule",
    "Rule",
    "apply_policy",
    "wire_reference",
]

_LOG = logging.getLogger("ayaka.serve")


class Rule(StrEnum):
    """What the server does with a field that is present and not null."""

    READ = "read"
    HINT = "hint"
    IGNORE = "ignore"
    REFUSE = "refuse"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class FieldRule:
    """One top-level field of one endpoint.

    Attributes:
        name: The field.
        rule: What the server does with it.
        reason: Why, in words a client author can act on; for a refusal it is the error text.
    """

    name: str
    rule: Rule
    reason: str


@dataclass(frozen=True, slots=True)
class EndpointPolicy:
    """The rules of one endpoint, in the order they are checked.

    Attributes:
        path: The endpoint, as routed.
        label: How log lines name it.
        rules: One rule per field.
    """

    path: str
    label: str
    rules: tuple[FieldRule, ...]

    def __post_init__(self) -> None:
        names = [rule.name for rule in self.rules]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.path}: a field has two rules")

    def rule(self, name: str) -> FieldRule | None:
        """The rule of ``name``, or ``None`` for a field the table does not know."""
        return next((rule for rule in self.rules if rule.name == name), None)

    @property
    def names(self) -> frozenset[str]:
        """Every field with a rule."""
        return frozenset(rule.name for rule in self.rules)


def _rules(rule: Rule, reason: str, *names: str) -> tuple[FieldRule, ...]:
    return tuple(FieldRule(name, rule, reason) for name in names)


_UNAVAILABLE: Final = "not available on this server"

#: The OpenAI sampling fields Chat Completions and Completions share.
_OPENAI_SAMPLING: Final = (
    *_rules(Rule.READ, "the model the request is for; another name is 404", "model"),
    *_rules(Rule.READ, "answer with server-sent events", "stream"),
    *_rules(Rule.READ, "include_usage adds a usage chunk at the end", "stream_options"),
    *_rules(Rule.READ, "capped by the room the prompt leaves", "max_tokens"),
    *_rules(
        Rule.READ,
        "sampling",
        "n",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "seed",
        "stop",
        "logit_bias",
        "logprobs",
        "min_tokens",
        "ignore_eos",
    ),
    *_rules(
        Rule.IGNORE,
        "accounting and storage fields; nothing is stored",
        "user",
        "metadata",
        "store",
        "service_tier",
    ),
)

_CHAT: Final = EndpointPolicy(
    "/v1/chat/completions",
    "chat completions",
    (
        *_rules(Rule.READ, "the conversation", "messages"),
        *_OPENAI_SAMPLING,
        *_rules(Rule.READ, "the output limit; wins over max_tokens", "max_completion_tokens"),
        *_rules(Rule.READ, "alternatives per token, with logprobs", "top_logprobs"),
        *_rules(
            Rule.READ,
            "function tools; a call needs --tool-call-parser",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
        ),
        *_rules(Rule.READ, "json_object or json_schema, always enforced", "response_format"),
        *_rules(Rule.READ, "only text output is available", "modalities"),
        *_rules(Rule.HINT, "the model reasons as trained", "reasoning_effort"),
        *_rules(
            Rule.REFUSE,
            "the deprecated function fields are not served; use tools",
            "functions",
            "function_call",
        ),
        *_rules(Rule.REFUSE, _UNAVAILABLE, "audio", "prediction", "web_search_options"),
    ),
)

_COMPLETIONS: Final = EndpointPolicy(
    "/v1/completions",
    "completions",
    (
        *_rules(Rule.READ, "a string or an array of token ids; one prompt only", "prompt"),
        *_OPENAI_SAMPLING,
        *_rules(Rule.READ, "true is refused: returning the prompt is not available", "echo"),
        *_rules(Rule.READ, "above 1 is refused: one candidate per request", "best_of"),
        *_rules(Rule.REFUSE, "fill-in-the-middle is not available", "suffix"),
    ),
)

_RESPONSES: Final = EndpointPolicy(
    "/v1/responses",
    "responses",
    (
        *_rules(Rule.READ, "the model the request is for; another name is 404", "model"),
        *_rules(Rule.READ, "the conversation and the system prompt", "input", "instructions"),
        *_rules(
            Rule.READ,
            "function tools; a call needs --tool-call-parser",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
        ),
        *_rules(Rule.READ, "capped by the room the prompt leaves", "max_output_tokens"),
        *_rules(Rule.READ, "sampling", "temperature", "top_p"),
        *_rules(Rule.READ, "answer with server-sent events", "stream"),
        *_rules(Rule.READ, "text.format: json_object or json_schema, enforced", "text"),
        *_rules(
            Rule.READ,
            "disabled (default) or auto to drop oldest turns to fit context",
            "truncation",
        ),
        *_rules(Rule.READ, "one answer per request; 1 only", "n"),
        *_rules(Rule.READ, "true is refused: this server stores no responses", "store"),
        *_rules(Rule.READ, "echoed in the response object", "metadata"),
        *_rules(Rule.HINT, "reasoning.effort: the model reasons as trained", "reasoning"),
        *_rules(
            Rule.IGNORE,
            "accounting, caching and output-selection fields with no effect here",
            "stream_options",
            "include",
            "user",
            "prompt_cache_key",
            "safety_identifier",
            "service_tier",
            "client_metadata",
        ),
        *_rules(
            Rule.INVALID,
            "this server keeps no conversation state; send the whole conversation in input",
            "previous_response_id",
            "conversation",
        ),
        *_rules(Rule.REFUSE, _UNAVAILABLE, "background", "prompt", "max_tool_calls"),
        *_rules(Rule.REFUSE, "logprobs are not available yet", "top_logprobs"),
    ),
)

#: The Messages fields that make the prompt, which count_tokens reads too.
_MESSAGES_PROMPT: Final = (
    *_rules(Rule.READ, "the model the request is for; another name is 404", "model"),
    *_rules(Rule.READ, "the conversation and the system prompt", "messages", "system"),
    *_rules(Rule.READ, "client tools; a call needs --tool-call-parser", "tools", "tool_choice"),
    *_rules(Rule.READ, "one answer per request; 1 only", "n"),
    *_rules(Rule.REFUSE, _UNAVAILABLE, "container", "mcp_servers"),
)

#: The Messages fields that shape the answer, not the prompt.
_MESSAGES_ANSWER: Final = (
    "max_tokens",
    "stop_sequences",
    "temperature",
    "top_p",
    "top_k",
    "stream",
    "thinking",
    "output_format",
    "output_config",
)

_MESSAGES_IGNORED: Final = (
    *_rules(
        Rule.IGNORE,
        "accounting fields; the prefix cache works without cache_control",
        "metadata",
        "service_tier",
        "cache_control",
    ),
    *_rules(
        Rule.IGNORE,
        "server-side context editing is not available; the prompt is used as sent",
        "context_management",
    ),
)

_MESSAGES: Final = EndpointPolicy(
    "/v1/messages",
    "messages",
    (
        *_MESSAGES_PROMPT,
        *_rules(Rule.READ, "required; capped by the room the prompt leaves", "max_tokens"),
        *_rules(Rule.READ, "sampling", "stop_sequences", "temperature", "top_p", "top_k"),
        *_rules(Rule.READ, "answer with server-sent events", "stream"),
        *_rules(
            Rule.READ,
            "checked; the budget is a hint, the model reasons as trained",
            "thinking",
        ),
        *_rules(
            Rule.READ,
            "json_schema output, enforced",
            "output_format",
            "output_config",
        ),
        *_MESSAGES_IGNORED,
    ),
)

_COUNT_TOKENS: Final = EndpointPolicy(
    "/v1/messages/count_tokens",
    "count_tokens",
    (
        *_MESSAGES_PROMPT,
        *_rules(Rule.IGNORE, "counting reads only the prompt", *_MESSAGES_ANSWER),
        *_MESSAGES_IGNORED,
    ),
)

#: The policy of every endpoint that reads a request body, by path.
ENDPOINTS: Final[Mapping[str, EndpointPolicy]] = MappingProxyType(
    {policy.path: policy for policy in (_CHAT, _COMPLETIONS, _RESPONSES, _MESSAGES, _COUNT_TOKENS)}
)


def apply_policy(body: JsonObject, path: str) -> None:
    """Refuse, log and pass the top-level fields of ``body`` by the table of ``path``.

    Fields are checked in table order, so the first refused field is the one named.

    Raises:
        UnsupportedFeatureError: For a ``refuse`` field that is present and not null.
        InvalidRequestError: For an ``invalid`` field that is present and not null.
    """
    policy = ENDPOINTS[path]
    for rule in policy.rules:
        if not body.has(rule.name):
            continue
        match rule.rule:
            case Rule.REFUSE:
                raise UnsupportedFeatureError(
                    f"{rule.name}: {rule.reason}",
                    context={"features": (rule.name,), "param": rule.name},
                )
            case Rule.INVALID:
                raise InvalidRequestError(
                    f"{rule.name}: {rule.reason}", context={"param": rule.name}
                )
            case Rule.HINT:
                _LOG.debug("%s: %s is a hint; %s", policy.label, rule.name, rule.reason)
            case Rule.READ | Rule.IGNORE:
                pass
    unknown = body.unknown(policy.names)
    if unknown:
        _LOG.debug("%s: ignoring unknown fields %s", policy.label, unknown)


def wire_reference() -> str:
    """The table as the Markdown section of ``docs/AI_REFERENCE.md``, one table per endpoint."""
    lines: list[str] = []
    for policy in ENDPOINTS.values():
        lines += [f"`POST {policy.path}`", "", "| Field | Rule | Why |", "| --- | --- | --- |"]
        lines += [
            f"| `{rule.name}` | {rule.rule} | {rule.reason} |" for rule in _sorted(policy.rules)
        ]
        lines.append("")
    return "\n".join(lines)


def _sorted(rules: Iterable[FieldRule]) -> list[FieldRule]:
    """Rules grouped by kind (read, hint, ignore, refuse, invalid), in table order inside each."""
    order = list(Rule)
    return sorted(rules, key=lambda rule: order.index(rule.rule))
