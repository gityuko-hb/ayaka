"""From a model name to a model that is ready to run.

``resolve_source`` turns a local directory or a Hugging Face repo id into a directory of
files; ``load_model`` builds the architecture its ``config.json`` names and streams the
safetensors weights into it; ``load_weights`` fills a model that already exists.

Public names are resolved from their submodule on first access. The loader imports torch,
so importing this package does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.model_loader.loader import describe_model, load_model, load_weights
    from ayaka.model_loader.source import resolve_source

__all__ = ["describe_model", "load_model", "load_weights", "resolve_source"]

_EXPORTS = {
    "loader": ("describe_model", "load_model", "load_weights"),
    "source": ("resolve_source",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
