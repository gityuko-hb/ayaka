"""Helpers for the optional files that sit next to a HuggingFace checkpoint.

A repository or directory may carry files beyond what ``AutoTokenizer`` or
``AutoModel`` read. Callers (the tokenizer adapter, and later the model loader) find
them here, treating "not present" as normal and anything else (network failure,
corrupt JSON) as something to report, not to hide::

    source (directory or hub id)
        +-- tokenizer.json          added tokens missing from tokenizer_config.json
        +-- chat_template.json      template stored outside tokenizer_config.json
        '-- generation_config.json  extra stop token ids

Lookup is local-first: a directory is read in place; a hub id goes through
``huggingface_hub`` and honours ``revision``, ``cache_dir`` and ``offline``.
``huggingface_hub`` is imported lazily, so importing this module needs no extras.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["optional_file", "parse_eos_token_ids", "read_json"]

_LOGGER = logging.getLogger(__name__)


def optional_file(
    source: str,
    filename: str,
    *,
    revision: str | None,
    cache_dir: str | None,
    offline: bool,
) -> Path | None:
    """Locate ``filename`` for ``source``, or return ``None`` if it does not exist.

    Args:
        source: A local directory or a hub repo id.
        filename: File name inside the directory or repository.
        revision: Hub revision; ignored for a local directory.
        cache_dir: Hub download directory; ignored for a local directory.
        offline: Never touch the network; use only what is already cached.

    Returns:
        The path of the file, or ``None`` when it is absent (or, offline, uncached).

    Raises:
        Exception: Network and authentication failures from ``huggingface_hub``
            propagate, because the caller cannot tell them from "absent".
    """
    local = Path(source)
    if local.is_dir():
        candidate = local / filename
        return candidate if candidate.is_file() else None
    hub = require_module(
        "huggingface_hub", capability="tokenizer.hf", remedy="pip install huggingface_hub"
    )
    errors = require_module("huggingface_hub.errors", capability="tokenizer.hf")
    try:
        found = hub.hf_hub_download(
            source, filename, revision=revision, cache_dir=cache_dir, local_files_only=offline
        )
    except (errors.EntryNotFoundError, errors.LocalEntryNotFoundError):
        return None
    return Path(cast("str", found))


def read_json(path: Path | None) -> Any:
    """Parse a JSON file, returning ``None`` (with a warning) if it is missing or invalid."""
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _LOGGER.warning("ignoring unreadable tokenizer sidecar file %s", path, exc_info=True)
        return None


def parse_eos_token_ids(generation_config: Mapping[str, Any] | None) -> frozenset[int]:
    """Stop-token ids declared by ``generation_config.json``.

    ``eos_token_id`` may be one integer or a list; booleans and non-integers are
    ignored. Returns an empty set when nothing usable is declared.
    """
    if not generation_config:
        return frozenset()
    raw: object = generation_config.get("eos_token_id")
    candidates = cast("list[object]", raw) if isinstance(raw, list) else [raw]
    return frozenset(c for c in candidates if type(c) is int)
