"""Isolate every test's generated data from the checkout."""

from pathlib import Path

import pytest

from tests.synthetic import make_synthetic_brain, make_synthetic_celeba


@pytest.fixture(autouse=True)
def flystate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Assign a fresh experiment home to each test.

    :param tmp_path: Test-owned temporary directory.
    :type tmp_path: Path
    :param monkeypatch: Environment restoration helper.
    :type monkeypatch: pytest.MonkeyPatch
    """
    monkeypatch.setenv(name='FLYSTATE_HOME', value=str(tmp_path / 'home'))


@pytest.fixture
def synthetic_brain_dir(tmp_path: Path) -> Path:
    """Create an offline brain compatible with the unmodified flybrain loader.

    :param tmp_path: Test-owned temporary directory.
    :type tmp_path: Path
    :returns: Synthetic brain directory.
    :rtype: Path
    """
    path = tmp_path / 'brain'
    make_synthetic_brain(path=path)
    return path


@pytest.fixture
def synthetic_celeba_dir(tmp_path: Path) -> Path:
    """Create synthetic faces in the official CelebA annotation layout.

    :param tmp_path: Test-owned temporary directory.
    :type tmp_path: Path
    :returns: Parent containing the generated celeba directory.
    :rtype: Path
    """
    root = tmp_path / 'dataset'
    make_synthetic_celeba(root=root)
    return root
