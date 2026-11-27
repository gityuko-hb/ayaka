"""Unit tests for Jinja thinking-policy detection across real model template patterns."""

from __future__ import annotations

import pytest

from ayaka.entrypoints.render.thinking import (
    AlwaysThinking,
    NativeToggle,
    ReasoningEffortPolicy,
    TemplateToggle,
    UnknownThinkingPolicy,
    detect_thinking_policy,
)

pytestmark = pytest.mark.unit

_QWEN3_TEMPLATE = """
{%- for message in messages %}
    {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}
"""

_QWEN3_OPT_IN_TEMPLATE = """
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking %}
        {{- '<think>\\n' }}
    {%- endif %}
{%- endif %}
"""

_DEEPSEEK_R1_TEMPLATE = """
{%- for message in messages %}{{ message['role'] }}: {{ message['content'] }}{% endfor %}
{%- if add_generation_prompt %}{{'<\uff5cAssistant\uff5c><think>\\n'}}{% endif %}
"""

_LLAMA3_TEMPLATE = """
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}
    {{- message['content'] + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
{%- endif %}
"""

_MIMO_TEMPLATE = """
{%- for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is not defined or enable_thinking -%}
        {{- '<think>\\n' }}
    {%- endif -%}
{%- endif %}
"""

_GLM_TEMPLATE = """
{%- for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}
{%- if add_generation_prompt %}
    {%- if thinking | default(true) %}<think>{% endif %}
{%- endif %}
"""

_MINIMAX_TEMPLATE = """
{%- if reasoning_effort is defined %}
    {%- if reasoning_effort == 'none' %}{% elif reasoning_effort == 'high' %}{% endif %}
{%- endif %}
"""


class TestDetectThinkingPolicy:
    def test_qwen3_default_enabled_template_toggle(self) -> None:
        policy = detect_thinking_policy(_QWEN3_TEMPLATE)
        assert policy == TemplateToggle(key="enable_thinking", default_enabled=True)

    def test_qwen3_opt_in_template_toggle(self) -> None:
        policy = detect_thinking_policy(_QWEN3_OPT_IN_TEMPLATE)
        assert policy == TemplateToggle(key="enable_thinking", default_enabled=False)

    def test_deepseek_r1_always_thinking(self) -> None:
        assert detect_thinking_policy(_DEEPSEEK_R1_TEMPLATE) == AlwaysThinking()

    def test_llama3_has_unknown_thinking_policy(self) -> None:
        assert detect_thinking_policy(_LLAMA3_TEMPLATE) == UnknownThinkingPolicy()

    def test_mimo_template_toggle(self) -> None:
        assert detect_thinking_policy(_MIMO_TEMPLATE) == TemplateToggle(
            key="enable_thinking", default_enabled=True
        )

    def test_glm_thinking_toggle(self) -> None:
        assert detect_thinking_policy(_GLM_TEMPLATE) == TemplateToggle(
            key="thinking", default_enabled=True
        )

    def test_minimax_reasoning_effort_policy(self) -> None:
        policy = detect_thinking_policy(_MINIMAX_TEMPLATE)
        assert isinstance(policy, ReasoningEffortPolicy)
        assert "none" in policy.levels
        assert "high" in policy.levels

    def test_comments_raw_blocks_and_string_literals_are_ignored(self) -> None:
        template = """
        {# enable_thinking is false; reasoning_effort = "high"; <think> #}
        {% raw %}
            {% if enable_thinking %} <think> {% endif %}
            {% if reasoning_effort == "high" %} {% endif %}
        {% endraw %}
        {% set note = "enable_thinking thinking reasoning_effort <think>" %}
        {% for m in messages %}{{ "enable_thinking" }}{{ m.content }}{% endfor %}
        {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}
        """
        assert detect_thinking_policy(template) == UnknownThinkingPolicy()

    def test_glm_model_hint_without_var_yields_native_toggle(self) -> None:
        template = "{% for m in messages %}{{ m.content }}{% endfor %}{# thinking #}"
        assert detect_thinking_policy(template, model_hint="GLM-4.5") == UnknownThinkingPolicy()

        native_template = (
            "{% for m in messages %}{% set _ = 'thinking' %}{{ m.content }}{% endfor %}"
        )
        assert detect_thinking_policy(native_template, model_hint="glm-4.5") == NativeToggle(
            default_enabled=True, named_tool_disables=True
        )

    def test_empty_or_missing_template_returns_unknown(self) -> None:
        assert detect_thinking_policy(None) == UnknownThinkingPolicy()
        assert detect_thinking_policy("") == UnknownThinkingPolicy()
