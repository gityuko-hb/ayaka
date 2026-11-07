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

"Offline" has to be enforced with :func:`hub_offline`, not only with the
``local_files_only`` argument: ``hf_hub_download`` builds its request headers before it
looks at that argument, and building them can fetch an agent registry from the Hub.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ayaka.utils.import_utils import require_module

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

__all__ = ["hub_offline", "optional_file", "parse_eos_token_ids", "read_json"]

_LOGGER = logging.getLogger(__name__)


class _OfflineGate:
    """Counts overlapping :func:`hub_offline` blocks so the flag is restored exactly once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0
        self._previous = False

    def enter(self, constants: Any) -> None:
        with self._lock:
            if self._depth == 0:
                self._previous = bool(constants.HF_HUB_OFFLINE)
                constants.HF_HUB_OFFLINE = True
            self._depth += 1

    def exit(self, constants: Any) -> None:
        with self._lock:
            self._depth -= 1
            if self._depth == 0:
                constants.HF_HUB_OFFLINE = self._previous


_GATE = _OfflineGate()


@contextmanager
def hub_offline(enabled: bool = True) -> Generator[None]:
    """Make ``huggingface_hub`` (and ``transformers`` through it) offline inside the block.

    Passing ``local_files_only=True`` is not enough: before ``hf_hub_download`` checks it,
    it builds request headers, and that can fetch an agent registry from the Hub (at most
    once a day, cached on disk). On a machine without that cache, such as CI or a fresh
    container, an "offline" load therefore tried to resolve ``huggingface.co``. The Hub's
    own offline switch, ``constants.HF_HUB_OFFLINE``, is the one that stops every request,
    so this flips it for the duration of the block and puts it back afterwards.

    The flag is process-wide: while a block runs, other threads in this process also see
    the Hub as offline. Blocks may nest and overlap; the flag is restored when the last
    one ends. Setting the environment variable ``HF_HUB_OFFLINE=1`` before the process
    starts is the stronger, global alternative.

    Args:
        enabled: With ``False`` the block runs unchanged and ``huggingface_hub`` is not
            imported, so callers can pass their ``offline`` setting straight through.

    Raises:
        CapabilityError: If ``enabled`` and ``huggingface_hub`` is not installed.
    """
    if not enabled:
        yield
        return
    constants: Any = require_module(
        "huggingface_hub.constants", capability="tokenizer.hf", remedy="pip install huggingface_hub"
    )
    _GATE.enter(constants)
    try:
        yield
    finally:
        _GATE.exit(constants)


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
        offline: Never touch the network; use only what is already cached. Enforced with
            :func:`hub_offline`, so it holds even when the Hub's own cache is cold.

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
        with hub_offline(offline):
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
