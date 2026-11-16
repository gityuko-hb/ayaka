"""``python -m ayaka``: the same command line as the ``ayaka`` script."""

from __future__ import annotations

from ayaka.entrypoints.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
