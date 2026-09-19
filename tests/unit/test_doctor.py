"""Check diagnostic ordering, failure isolation, thresholds, and CLI output."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from flybrain import data
from typer.testing import CliRunner

from flystate import doctor
from flystate.cli.main import app
from flystate.hashing import sha256_file
from flystate.settings import get_paths

CHECK_NAMES: tuple[str, ...] = (
    'python',
    'imports',
    'home_writable',
    'wsl_paths',
    'disk_free',
    'memory',
    'cpu',
    'numba_threads',
    'brain_files',
    'datasets',
    'gpu',
)


class TestDoctor:
    """Cover independent diagnostics on isolated, offline test installations."""

    def test_order_and_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Report eleven checks even without registered data.

        :param monkeypatch: Stable resource readings independent of CI capacity.
        :type monkeypatch: pytest.MonkeyPatch
        """
        monkeypatch.setattr(
            target=doctor.shutil,
            name='disk_usage',
            value=Mock(return_value=SimpleNamespace(free=20 * doctor.GIGABYTE)),
        )
        result = doctor.run_checks(paths=get_paths(), check_hash=False)
        assert tuple(check.name for check in result) == CHECK_NAMES
        assert not any(check.status == 'error' for check in result)
        assert result[8].status == 'warn'
        assert result[9].status == 'warn'
        assert 'not used by flystate' in result[10].detail
        assert doctor._parallel_probe.py_func() == 1
        runner = CliRunner()
        for flags in (['--json'], []):
            response = runner.invoke(app=app, args=['doctor', '--no-hash', *flags])
            assert response.exit_code == 0, response.exception
            if flags:
                assert (
                    tuple(item['name'] for item in json.loads(s=response.stdout)['checks'])
                    == CHECK_NAMES
                )
            else:
                assert 'brain_files: warn:' in response.stdout
        assert runner.invoke(app=app, args=['doctor', '--help']).exit_code == 0

    @pytest.mark.parametrize('free,status', [(4, 'error'), (5, 'warn'), (14, 'warn'), (15, 'ok')])
    def test_disk_thresholds(self, monkeypatch: pytest.MonkeyPatch, free: int, status: str) -> None:
        """Respect inclusive disk thresholds in decimal gigabytes.

        :param monkeypatch: Isolated disk reading.
        :type monkeypatch: pytest.MonkeyPatch
        :param free: Available decimal gigabytes.
        :type free: int
        :param status: Expected severity.
        :type status: str
        """
        monkeypatch.setattr(
            target=doctor.shutil,
            name='disk_usage',
            value=Mock(return_value=SimpleNamespace(free=free * doctor.GIGABYTE)),
        )
        checks = doctor.run_checks(paths=get_paths(), check_hash=False)
        assert checks[4].status == status

    def test_warnings_and_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Continue after import, filesystem, and registry errors.

        :param monkeypatch: Isolated dependency and resource failure simulation.
        :type monkeypatch: pytest.MonkeyPatch
        """
        monkeypatch.setattr(
            target=doctor, name='REQUIRED_MODULES', value=('flystate_missing_package',)
        )
        monkeypatch.setattr(
            target=doctor.psutil,
            name='virtual_memory',
            value=Mock(return_value=SimpleNamespace(total=7 * doctor.GIGABYTE)),
        )
        paths = get_paths()
        paths.datasets.mkdir(parents=True)
        (paths.datasets / 'registry.json').write_text(data='[]', encoding='utf-8')
        monkeypatch.setattr(
            target=doctor, name='ensure_dirs', value=Mock(side_effect=OSError('read only'))
        )
        checks = doctor.run_checks(paths=paths, check_hash=False)
        assert checks[1].status == 'error'
        assert checks[2].status == 'error'
        assert checks[5].status == 'warn'
        assert checks[9].status == 'error'
        assert checks[10].status == 'ok'
        result = CliRunner().invoke(app=app, args=['doctor', '--json'])
        assert result.exit_code == 1
        assert len(json.loads(s=result.stdout)['checks']) == 11

    def test_wsl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Warn about Windows mounts without writing to the mocked location.

        :param monkeypatch: Mock platform and filesystem probes.
        :type monkeypatch: pytest.MonkeyPatch
        """
        monkeypatch.setattr(
            target=doctor.platform, name='release', value=Mock(return_value='microsoft-WSL2')
        )
        monkeypatch.setattr(
            target=doctor, name='_home', value=Mock(side_effect=OSError('unavailable'))
        )
        checks = doctor.run_checks(paths=get_paths(home=Path('/mnt/c/flystate-doctor-test')))
        assert checks[3].status == 'warn'
        assert 'Linux file system' in checks[3].detail
        monkeypatch.setattr(target=doctor.sys, name='version_info', value=(3, 11))
        assert doctor.run_checks(paths=get_paths())[0].status == 'error'

    def test_brain_and_registry(
        self, synthetic_brain_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Show corrupt brain files, verified files, and dataset registration status.

        :param synthetic_brain_dir: Generated connectome fixture.
        :type synthetic_brain_dir: Path
        :param monkeypatch: Scoped published hashes.
        :type monkeypatch: pytest.MonkeyPatch
        """
        paths = get_paths()
        shutil.copytree(src=synthetic_brain_dir, dst=paths.brain)
        paths.datasets.mkdir()
        (paths.datasets / 'registry.json').write_text(data='{"celeba":{"validated":true}}')
        assert doctor.run_checks(paths=paths)[8].status == 'error'
        monkeypatch.setattr(
            target=data,
            name='FILES',
            value={name: sha256_file(path=paths.brain / name) for name in data.FILES},
        )
        checks = doctor.run_checks(paths=paths)
        assert checks[8].status == 'ok'
        assert checks[9].status == 'ok'
        assert 'celeba' in checks[9].detail
        unchecked = doctor.run_checks(paths=paths, check_hash=False)
        assert 'hashes skipped' in unchecked[8].detail
