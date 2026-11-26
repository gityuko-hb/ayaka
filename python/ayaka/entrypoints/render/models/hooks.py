"""Model-specific render hooks for prompt layout and reasoning-block boundary detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["ModelRenderHook", "select_render_hook"]


@dataclass(frozen=True, slots=True)
class ModelRenderHook:
    """Model-family adjustments around Jinja chat template rendering.

    Attributes:
        family: Short family identifier (``qwen3``, ``mimo``, ``deepseek``, ``glm``,
            ``minimax``, or ``default``).
        reasoning_open: Opening reasoning marker emitted by this family.
        reasoning_close: Closing reasoning marker emitted by this family.
        include_reasoning_in_history: Whether prior assistant turns retain ``reasoning_content``.
    """

    family: str
    reasoning_open: str = "<think>"
    reasoning_close: str = "</think>"
    include_reasoning_in_history: bool = False

    def prepare_messages(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        thinking_enabled: bool | None,
    ) -> tuple[Mapping[str, object], ...]:
        """Adjust historical messages for the model family before rendering."""
        if self.include_reasoning_in_history and thinking_enabled is not False:
            return tuple(messages)
        cleaned: list[Mapping[str, object]] = []
        for message in messages:
            if "reasoning_content" in message and not (
                self.include_reasoning_in_history and thinking_enabled is not False
            ):
                updated = {k: v for k, v in message.items() if k != "reasoning_content"}
                cleaned.append(updated)
            else:
                cleaned.append(message)
        return tuple(cleaned)

    def detect_starts_inside(self, prompt_text: str, *, thinking_enabled: bool | None) -> bool:
        """Whether ``prompt_text`` leaves an unclosed reasoning tag open at the tail.

        Distinguishes DeepSeek-R1's trailing unclosed ``<think>\\n`` (returns ``True``) from
        Qwen3's disabled-thinking suffix ``<think>\\n\\n</think>\\n\\n`` (returns ``False``).
        """
        if thinking_enabled is False:
            return False
        last_open = prompt_text.rfind(self.reasoning_open)
        if last_open < 0:
            return False
        last_close = prompt_text.rfind(self.reasoning_close)
        return last_open > last_close


_HOOKS: dict[str, ModelRenderHook] = {
    "qwen3": ModelRenderHook("qwen3", "<think>", "</think>", include_reasoning_in_history=True),
    "mimo": ModelRenderHook("mimo", "<think>", "</think>", include_reasoning_in_history=False),
    "deepseek": ModelRenderHook(
        "deepseek", "<think>", "</think>", include_reasoning_in_history=True
    ),
    "glm": ModelRenderHook("glm", "<think>", "</think>", include_reasoning_in_history=True),
    "minimax": ModelRenderHook("minimax", "<think>", "</think>", include_reasoning_in_history=True),
    "default": ModelRenderHook(
        "default", "<think>", "</think>", include_reasoning_in_history=False
    ),
}


def select_render_hook(model_name: str | None, chat_template: str | None = None) -> ModelRenderHook:
    """Pick the :class:`ModelRenderHook` matching ``model_name`` or ``chat_template``."""
    hint = (model_name or "").lower()
    template = (chat_template or "").lower()
    if "qwen3" in hint or ("qwen" in hint and "enable_thinking" in template):
        return _HOOKS["qwen3"]
    if "mimo" in hint:
        return _HOOKS["mimo"]
    if "deepseek" in hint or "<\uff5cassistant\uff5c>" in template or "<|assistant|>" in hint:
        return _HOOKS["deepseek"]
    if "glm" in hint:
        return _HOOKS["glm"]
    if "minimax" in hint or "minimax:tool_call" in template:
        return _HOOKS["minimax"]
    return _HOOKS["default"]
