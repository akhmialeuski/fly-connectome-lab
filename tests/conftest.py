"""Isolate every test's generated data from the checkout."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def flystate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Assign a fresh experiment home to each test.

    :param tmp_path: Test-owned temporary directory.
    :type tmp_path: Path
    :param monkeypatch: Environment restoration helper.
    :type monkeypatch: pytest.MonkeyPatch
    """
    monkeypatch.setenv(name='FLYSTATE_HOME', value=str(tmp_path / 'home'))
