"""From a model name to a directory of files: a local checkpoint or a Hugging Face snapshot.

``ModelSourceConfig.model`` is either a directory on disk or a Hub repo id
(``owner/name``). A directory that exists always wins, so a local folder named like a
repo is never replaced by a download. A repo id is fetched with ``huggingface_hub``
(the ``hf`` extra, imported only here and only for a repo id) into the Hub cache, at the
configured revision; ``offline`` means the cache alone, with no request made, even to
resolve a revision.

Only the files the loader reads are fetched: ``config.json``, the safetensors shards and
their index. No code from the repository is run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from ayaka.core.errors import ModelLoadError
from ayaka.utils.hf_utils import hub_offline
from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from ayaka.configs.model_source import ModelSourceConfig

__all__ = ["resolve_source"]

#: What ``snapshot_download`` fetches: the config and the weights, nothing else.
_PATTERNS: Final = ["config.json", "*.safetensors", "model.safetensors.index.json"]


def resolve_source(source: ModelSourceConfig) -> Path:
    """Return the directory that holds the checkpoint named by ``source``.

    Args:
        source: A local directory or a Hub repo id, with the revision, cache directory and
            offline switch to use for the Hub.

    Returns:
        A directory: ``source.model`` itself when it is one, else the Hub snapshot of the
        repo at ``source.revision`` (the default revision when it is empty).

    Raises:
        ModelLoadError: If ``source.model`` is not a directory and does not look like a
            repo id, or the Hub does not have the repo or revision, cannot be reached, or,
            offline, the cache does not hold it.
        CapabilityError: If a repo id is given and ``huggingface_hub`` is not installed.
    """
    local = Path(source.model).expanduser()
    if local.is_dir():
        return local
    if not source.looks_like_hub_id():
        raise ModelLoadError(
            f"{source.model!r} is not a directory, and it is not a Hugging Face repo id "
            "of the form 'owner/name'",
            context={"model": source.model},
        )

    hub: Any = require_module(
        "huggingface_hub", capability="model_loader.hub", remedy="uv sync --extra hf"
    )
    try:
        with hub_offline(source.offline):
            snapshot = hub.snapshot_download(
                source.model,
                revision=source.revision or None,
                cache_dir=source.cache_dir or None,
                local_files_only=source.offline,
                allow_patterns=_PATTERNS,
            )
    except (OSError, ValueError) as exc:
        # Every error of ``huggingface_hub`` (repo or revision not found, gated, no network,
        # not in the cache while offline, malformed id) is one of these.
        where = "the local Hub cache (offline)" if source.offline else "the Hugging Face Hub"
        raise ModelLoadError(
            f"cannot get {source.model!r} from {where}: {exc}",
            context={
                "model": source.model,
                "revision": source.revision,
                "offline": source.offline,
            },
        ) from exc
    return Path(snapshot)
