"""What every ``ayaka`` command shares with its user: exit codes and the standard-error line.

Exit codes are the contract of the command line::

    0    the text was generated, the server stopped cleanly, or the plan fits
    1    an unexpected error: a bug; ``-v`` prints its traceback
    2    the arguments or the request are wrong (a flag, a prompt too long, an unsupported option)
    3    the model or the tokenizer cannot be loaded
    4    something the run needs is not installed or not available (torch, a tokenizer package,
         a GPU)
    5    the memory plan does not fit this machine (``plan`` prints three ways out)
    6    the engine failed while generating
    130  interrupted
"""

from __future__ import annotations

import sys
from typing import Final

__all__ = [
    "EXIT_BUG",
    "EXIT_ENGINE",
    "EXIT_INTERRUPTED",
    "EXIT_LOAD",
    "EXIT_MEMORY",
    "EXIT_MISSING",
    "EXIT_OK",
    "EXIT_USAGE",
    "PROG",
    "say",
]

EXIT_OK: Final = 0
EXIT_BUG: Final = 1
EXIT_USAGE: Final = 2
EXIT_LOAD: Final = 3
EXIT_MISSING: Final = 4
EXIT_MEMORY: Final = 5
EXIT_ENGINE: Final = 6
EXIT_INTERRUPTED: Final = 130

#: The program name every line to standard error starts with.
PROG: Final = "ayaka"


def say(message: str) -> None:
    """One line to standard error, after the program name: progress, warnings, errors."""
    print(f"{PROG}: {message}", file=sys.stderr)
