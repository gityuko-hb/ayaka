"""Model-family render hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.entrypoints.render.models.hooks import ModelRenderHook, select_render_hook

__all__ = ["ModelRenderHook", "select_render_hook"]

_EXPORTS = {
    "hooks": ("ModelRenderHook", "select_render_hook"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
