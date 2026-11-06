from importlib.metadata import version

import pytest

import ayaka


@pytest.mark.unit
def test_version_matches_the_installed_metadata() -> None:
    assert ayaka.__version__ == version("ayaka")
