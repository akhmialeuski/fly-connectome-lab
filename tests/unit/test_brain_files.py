"""Verify synthetic connectomes and offline acquisition command behavior."""

import json
import os
import shutil
from pathlib import Path
from unittest.mock import Mock

import flybrain
import numba
import numpy as np
import pytest
from flybrain import data
from scipy import sparse
from typer.testing import CliRunner

from flystate.brain import files
from flystate.cli.main import app
from flystate.hashing import sha256_file
from flystate.settings import get_paths
from tests.synthetic import make_synthetic_brain


class TestBrainFiles:
    """Check published-file integrity contracts without network access."""

    def test_metadata_and_simulator(self, synthetic_brain_dir: Path) -> None:
        """Load the fixture through unmodified flybrain and advance one CPU step.

        :param synthetic_brain_dir: Generated test connectome.
        :type synthetic_brain_dir: Path
        """
        stats = files.brain_stats(brain_dir=synthetic_brain_dir)
        assert stats == {
            'neurons': 2000,
            'connections': 60000,
            'photoreceptors': 50,
            'superclass_counts': {
                'descending_neuron': 100,
                'visual_projection': 600,
                'ol_sensory': 50,
                'cb_intrinsic': 1250,
            },
            'groups': ['escape_L', 'escape_R'],
        }
        weights = sparse.load_npz(file=synthetic_brain_dir / 'weights.npz')
        assert weights.dtype == np.float32
        assert np.max(np.asarray(abs(weights).sum(axis=1))) <= 1.000001
        previous = numba.get_num_threads()
        try:
            numba.set_num_threads(n=1)
            brain = flybrain.FlyBrain(data=synthetic_brain_dir, device='cpu', sensory_input=False)
            brain.step()
            assert brain.steps == 1
            assert brain.v.shape == (2000, 1)
        finally:
            numba.set_num_threads(n=previous)

    def test_hashes(self, synthetic_brain_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Distinguish missing, unchecked, mismatched and verified files.

        :param synthetic_brain_dir: Generated test connectome.
        :type synthetic_brain_dir: Path
        :param monkeypatch: Scoped published-digest replacement.
        :type monkeypatch: pytest.MonkeyPatch
        """
        result = files.verify_brain_files(brain_dir=synthetic_brain_dir)
        assert all(item['exists'] and item['sha256_ok'] is False for item in result.values())
        assert all(item['size_bytes'] > 0 for item in result.values())
        unchecked = files.verify_brain_files(brain_dir=synthetic_brain_dir, check_hash=False)
        assert all(
            item['sha256'] is None and item['sha256_ok'] is None for item in unchecked.values()
        )
        monkeypatch.setattr(
            target=data,
            name='FILES',
            value={name: sha256_file(path=synthetic_brain_dir / name) for name in data.FILES},
        )
        assert all(
            item['sha256_ok']
            for item in files.verify_brain_files(brain_dir=synthetic_brain_dir).values()
        )
        missing = files.verify_brain_files(brain_dir=synthetic_brain_dir / 'absent')
        assert all(not item['exists'] and item['size_bytes'] == 0 for item in missing.values())

    def test_shape_mismatch(self, synthetic_brain_dir: Path) -> None:
        """Reject inconsistent weight and metadata dimensions.

        :param synthetic_brain_dir: Generated test connectome.
        :type synthetic_brain_dir: Path
        """
        sparse.save_npz(
            file=synthetic_brain_dir / 'weights.npz', matrix=sparse.eye(3, format='csr')
        )
        with pytest.raises(expected_exception=ValueError, match='dimensions disagree'):
            files.brain_stats(brain_dir=synthetic_brain_dir)

    @pytest.mark.parametrize('arguments', [{'n_descending': 1}, {'n': 100}, {'out_degree': 2001}])
    def test_invalid_fixture(self, tmp_path: Path, arguments: dict[str, int]) -> None:
        """Reject impossible fixture sizes before writing files.

        :param tmp_path: Fixture output directory.
        :type tmp_path: Path
        :param arguments: Invalid generator settings.
        :type arguments: dict[str, int]
        """
        with pytest.raises(expected_exception=ValueError):
            make_synthetic_brain(path=tmp_path, **arguments)

    def test_cli_download(self, synthetic_brain_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify successful downloads and explicit checksum/count failures offline.

        :param synthetic_brain_dir: Source fixture copied by the fake downloader.
        :type synthetic_brain_dir: Path
        :param monkeypatch: Scoped downloader and published-data settings.
        :type monkeypatch: pytest.MonkeyPatch
        """
        paths = get_paths()
        shutil.copytree(src=synthetic_brain_dir, dst=paths.brain)
        downloader = Mock(return_value=paths.brain)
        monkeypatch.setattr(target=flybrain, name='download', value=downloader)
        runner = CliRunner()
        command = ['brain', 'download', '--json']
        mismatch = runner.invoke(app=app, args=command)
        assert mismatch.exit_code == 1
        assert json.loads(s=mismatch.stdout)['valid'] is False
        monkeypatch.setattr(
            target=data,
            name='FILES',
            value={name: sha256_file(path=paths.brain / name) for name in data.FILES},
        )
        wrong_counts = runner.invoke(app=app, args=command)
        assert wrong_counts.exit_code == 1
        monkeypatch.setattr(target=files, name='EXPECTED_NEURONS', value=2000)
        monkeypatch.setattr(target=files, name='EXPECTED_CONNECTIONS', value=60000)
        success = runner.invoke(app=app, args=[*command, '--force'])
        assert success.exit_code == 0, success.exception
        assert json.loads(s=success.stdout)['valid'] is True
        downloader.assert_called_with(data=paths.brain, force=True)
        info = runner.invoke(app=app, args=['brain', 'info', '--json', '--no-hash'])
        assert info.exit_code == 0
        assert json.loads(s=info.stdout)['stats']['neurons'] == 2000

    def test_cli_missing_and_corrupt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return one JSON error or stderr detail for failed file operations.

        :param monkeypatch: Scoped downloader failure.
        :type monkeypatch: pytest.MonkeyPatch
        """
        runner = CliRunner()
        missing = runner.invoke(app=app, args=['brain', 'info', '--json'])
        assert missing.exit_code == 1
        assert json.loads(s=missing.stdout)['stats'] == {}
        paths = get_paths()
        paths.brain.mkdir(parents=True)
        for name in data.FILES:
            (paths.brain / name).write_bytes(data=b'corrupt')
        corrupt = runner.invoke(app=app, args=['brain', 'info', '--json'])
        assert corrupt.exit_code == 1
        assert 'error' in json.loads(s=corrupt.stdout)
        monkeypatch.setattr(
            target=flybrain, name='download', value=Mock(side_effect=RuntimeError('offline'))
        )
        for flags in ([], ['--json']):
            failure = runner.invoke(app=app, args=['brain', 'download', *flags])
            assert failure.exit_code == 1
            assert 'offline' in failure.stderr
        for command in ('info', 'download'):
            assert runner.invoke(app=app, args=['brain', command, '--help']).exit_code == 0

    @pytest.mark.connectome
    @pytest.mark.skipif(os.environ.get('FLYSTATE_REAL_BRAIN') != '1', reason='Real data is opt-in.')
    def test_real_metadata(self) -> None:
        """Check published MaleCNS dimensions from the explicitly supplied data home."""
        root = os.environ.get('FLYSTATE_REAL_HOME')
        assert root, 'Set FLYSTATE_REAL_HOME to the actual data home.'
        stats = files.brain_stats(brain_dir=get_paths(home=Path(root)).brain)
        assert stats['neurons'] == 166700
        assert stats['connections'] == 25582938
        counts = stats['superclass_counts']
        assert isinstance(counts, dict)
        assert counts['visual_projection'] == 9201
        assert counts['descending_neuron'] == 1314
