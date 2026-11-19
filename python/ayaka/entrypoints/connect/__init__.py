"""``ayaka connect``: point a coding tool at a running ``ayaka serve`` and check that it works.

Four jobs, one record per tool:

* print the configuration the tool needs (the default);
* ``--write``: show a diff of the tool's configuration file, back the file up, ask, then write it
  through a temporary file and an atomic replace (:mod:`~ayaka.entrypoints.connect.write`);
* ``--check``: seven steps against the running server, in the tool's own protocol, each with a
  fix when it fails (:mod:`~ayaka.entrypoints.connect.check`);
* ``--observe``: a small proxy between the tool and the server that counts the tokens of each
  model request by part (system prompt, tool definitions, conversation) and stores only the counts
  of the largest one, the tool's floor (a tool's first request is not always its agent request:
  OpenCode first asks for a session title); with ``--record DIR`` it also keeps each request
  (credentials removed) to build replay fixtures (:mod:`~ayaka.entrypoints.connect.observe`).

A tool is a :class:`~ayaka.entrypoints.connect.tools.ToolRecord`: data, not code paths. The
records follow the tools' documentation and an installed version (see each record); a client
changing its format is a one-record change. The API key is never printed, never written to a
report and never kept in a recording. :mod:`~ayaka.entrypoints.connect.command` runs the command.

Public names are resolved from the module that defines them on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.entrypoints.connect.check import CheckReport, run_check
    from ayaka.entrypoints.connect.command import connect_command
    from ayaka.entrypoints.connect.observe import ObservedPrompt, serve_proxy
    from ayaka.entrypoints.connect.tools import TOOLS, ServerInfo, ToolRecord, tool_record
    from ayaka.entrypoints.connect.write import write_config

__all__ = [
    "TOOLS",
    "CheckReport",
    "ObservedPrompt",
    "ServerInfo",
    "ToolRecord",
    "connect_command",
    "run_check",
    "serve_proxy",
    "tool_record",
    "write_config",
]

_EXPORTS = {
    "check": ("CheckReport", "run_check"),
    "command": ("connect_command",),
    "observe": ("ObservedPrompt", "serve_proxy"),
    "tools": ("TOOLS", "ServerInfo", "ToolRecord", "tool_record"),
    "write": ("write_config",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
