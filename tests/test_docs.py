"""The public documentation: the generated table, its source, links, and the commands it names."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from ayaka.entrypoints.cli import _parser  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted([ROOT / "README.md", *(ROOT / "docs").glob("*.md")])
PUBLIC = ("INSTALL.md", "TROUBLESHOOTING.md", "CAPABILITIES.md", "SAFETY.md", "AI_SETUP.md")


def _generator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "generate_capabilities", ROOT / "tools" / "generate_capabilities.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_capabilities"] = module
    spec.loader.exec_module(module)
    return module


generator = _generator()


class TestCapabilities:
    def test_the_page_is_what_the_generator_writes(self) -> None:
        page = (ROOT / "docs" / "CAPABILITIES.md").read_text(encoding="utf-8")

        assert page == generator.render(generator.load()), (
            "docs/CAPABILITIES.md was edited by hand or its source changed: "
            "run uv run python tools/generate_capabilities.py"
        )

    def test_every_number_names_its_result_and_version(self) -> None:
        for row in generator.load():
            line = next(
                x
                for x in generator.render([row]).splitlines()
                if x.startswith(f"| {row['model']} | {row['precision']} |")
            )
            if row["status"] == "runs":
                assert row["result"] in line
                assert "ayaka " in line
            else:
                assert not re.search(r"\d tok/s", line)

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            ({"status": "runs"}, "needs the result file"),
            ({"status": "untested", "result": "x"}, "untested row has no result"),
            ({"status": "does-not-run", "result": "README.md"}, "needs its reason"),
            ({"status": "supported"}, "status must be one of"),
            ({"status": "untested", "speed": 9}, "unknown keys"),
            ({"status": "runs", "result": "no/such/file.json"}, "does not exist"),
        ],
    )
    def test_the_source_refuses_rows_that_break_its_rules(
        self, tmp_path: Path, row: dict[str, Any], message: str
    ) -> None:
        fields = {"model": "m", "precision": "fp16", "machine": "x", **row}
        body = "\n".join(
            f"{key} = {value!r}" if isinstance(value, str) else f"{key} = {value}"
            for key, value in fields.items()
        ).replace("'", '"')
        source = tmp_path / "capabilities.toml"
        source.write_text(f"version = 1\n\n[[row]]\n{body}\n", encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            generator.load(source)


def _links(page: Path) -> list[str]:
    text = page.read_text(encoding="utf-8")
    return [target for target in re.findall(r"\]\(([^)\s]+)\)", text) if "://" not in target]


class TestPages:
    @pytest.mark.parametrize("name", PUBLIC)
    def test_the_public_pages_exist(self, name: str) -> None:
        assert (ROOT / "docs" / name).is_file()

    @pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
    def test_every_relative_link_resolves(self, page: Path) -> None:
        for target in _links(page):
            path = target.split("#", 1)[0]
            if path:
                assert (page.parent / path).exists(), f"{page.name} links to missing {target}"


def _options(parser: argparse.ArgumentParser) -> dict[str, set[str]]:
    commands: dict[str, set[str]] = {}
    for action in parser._actions:  # pyright: ignore[reportPrivateUsage]
        if isinstance(action, argparse._SubParsersAction):  # pyright: ignore[reportPrivateUsage]
            for name, sub in action.choices.items():  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                flags = {
                    option
                    for item in sub._actions  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]
                    for option in item.option_strings  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                }
                commands[str(name)] = {str(flag) for flag in flags}  # pyright: ignore[reportUnknownArgumentType]
    return commands


def _ayaka_commands(page: Path) -> list[tuple[str, list[str]]]:
    """Every ``ayaka COMMAND --flag ...`` in a page's code: the command and its flags."""
    text = page.read_text(encoding="utf-8").replace("\\\n", " ")
    found: list[tuple[str, list[str]]] = []
    for code in re.findall(r"```[a-z]*\n(.*?)```|`([^`\n]+)`", text, flags=re.DOTALL):
        for line in "".join(code).splitlines():
            for match in re.finditer(r"\bayaka ([a-z]+)((?: +[^ |;&]+)*)", line):
                flags = re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", match.group(2))
                found.append((match.group(1), flags))
    return found


class TestCommandsExist:
    @pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
    def test_every_command_and_flag_a_page_names_exists(self, page: Path) -> None:
        options = _options(_parser())
        for command, flags in _ayaka_commands(page):
            if command in ("info", "config") and not flags:
                continue
            assert command in options, f"{page.name} names ayaka {command}, which does not exist"
            missing = [flag for flag in flags if flag not in options[command]]
            assert not missing, f"{page.name}: ayaka {command} has no {missing}"
