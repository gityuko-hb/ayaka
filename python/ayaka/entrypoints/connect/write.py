"""``ayaka connect --write``: a diff, a backup, a confirmation, then an atomic replace."""

from __future__ import annotations

import difflib
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["WriteResult", "make_private", "write_config"]


@dataclass(frozen=True, slots=True)
class WriteResult:
    """What ``--write`` did."""

    path: Path
    changed: bool
    backup: Path | None
    written: bool


def write_config(
    path: Path,
    content: str,
    *,
    confirm: Callable[[str], bool],
    create_dirs: bool = False,
    show: Callable[[str], None] = print,
    redactor: Callable[[str], str] = lambda text: text,
) -> WriteResult:
    """Show a diff, back the old file up, ask, then write atomically.

    Args:
        path: The file.
        content: Its new content.
        confirm: Asked with the question; the file is written only on ``True``.
        create_dirs: Whether a missing parent directory may be created.
        show: Where the diff goes.
        redactor: Applied to the diff before it is shown, so a key never appears.

    Raises:
        FileNotFoundError: If the parent directory is missing and ``create_dirs`` is false.
    """
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if old == content:
        show(f"{path} already has this configuration")
        return WriteResult(path, changed=False, backup=None, written=False)
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        content.splitlines(keepends=True),
        fromfile=str(path),
        tofile=f"{path} (new)",
    )
    show(redactor("".join(diff)))
    if not path.parent.exists() and not create_dirs:
        raise FileNotFoundError(f"{path.parent} does not exist; pass --create-dirs to create it")
    if not confirm(f"write {path}?"):
        show("nothing written")
        return WriteResult(path, changed=True, backup=None, written=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if path.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = path.with_name(f"{path.name}.{stamp}.bak")
        serial = 1
        while backup.exists():  # two writes in one second keep both backups
            backup = path.with_name(f"{path.name}.{stamp}-{serial}.bak")
            serial += 1
        shutil.copy2(path, backup)
        make_private(backup)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        make_private(Path(temporary))
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    show(f"wrote {path}" + (f" (backup: {backup})" if backup else ""))
    return WriteResult(path, changed=True, backup=backup, written=True)


def make_private(path: Path) -> None:
    """Owner-only permissions on POSIX; Windows keeps the user profile's ACL."""
    if os.name == "posix":
        path.chmod(0o600)
