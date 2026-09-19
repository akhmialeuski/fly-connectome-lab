"""Verify matched-noise calibration, latency boundaries, and operational reports."""

import json
from pathlib import Path
from unittest.mock import patch

import numba
import numpy as np
import pytest
from typer.testing import CliRunner

from flystate.brain.calibrate import calibrate, recommend, response_latencies
from flystate.cli.main import app
from flystate.datasets import registry
from flystate.datasets.celeba import CelebAAdapter
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.settings import get_paths


@pytest.fixture
def calibration_config(synthetic_celeba_dir: Path) -> ExperimentConfig:
    """Register synthetic images with a feasible small encoder and fixed thread count.

    :param synthetic_celeba_dir: Generated CelebA annotations and images.
    :type synthetic_celeba_dir: Path
    :returns: Validated four-identity experiment.
    :rtype: ExperimentConfig
    """
    paths = get_paths()
    registry.register(paths=paths, name='celeba', root=synthetic_celeba_dir)
    registry.set_validation(
        paths=paths,
        name='celeba',
        full=True,
        report=CelebAAdapter(root=synthetic_celeba_dir, expected=None).validate(full=True),
    )
    return ExperimentConfig.model_validate(
        obj={
            'name': 'calibration',
            'dataset': {'subset': {'n_identities': 4}},
            'encoder': {'neurons_per_channel': 64, 'position': {'neurons_per_axis': 4}},
            'brain': {'threads': 1, 'batch_size': 4, 'warmup_steps': 4},
        }
    )


class TestLatency:
    """Check response thresholds and avoid invented fallback recommendations."""

    def test_crossings(self) -> None:
        """Require one-based crossings and a detectable response for recommendations."""
        assert response_latencies(differences=np.asarray(a=[0, 1e-6, 0.2, 0.6, 1])) == (3, 4)
        assert response_latencies(differences=np.zeros(shape=4)) == (None, None)
        assert response_latencies(differences=np.asarray(a=[1e-9])) == (None, 1)
        rows: list[dict[str, float | int | None]] = [
            {'amplitude': 0.1, 'latency_first': 1, 'latency_half': 4, 'rate_readout_hz': 1},
            {'amplitude': 0.05, 'latency_first': None, 'latency_half': 1, 'rate_readout_hz': 0},
        ]
        result, warnings = recommend(amplitudes=rows)
        assert result == {'amplitude': 0.1, 'steps_per_observation': 5} and not warnings
        rows[0]['latency_half'] = 11
        result, warnings = recommend(amplitudes=rows)
        assert result == {'amplitude': None, 'steps_per_observation': None}
        assert warnings == ['latency_exceeds_10']
        rows[0]['rate_readout_hz'] = 6
        assert recommend(amplitudes=rows)[1] == ['no_admissible_amplitude']
        assert recommend(amplitudes=[])[0]['amplitude'] is None

    @pytest.mark.parametrize('curve', [[], [[0]], [-1], [float('nan')]])
    def test_invalid_curve(self, curve: list) -> None:
        """Reject malformed response measurements.

        :param curve: Invalid voltage curve.
        :type curve: list
        """
        with pytest.raises(expected_exception=ValueError):
            response_latencies(differences=np.asarray(a=curve))


class TestCalibration:
    """Exercise real synthetic simulation and ensure noise pairing survives chunking."""

    def test_cli(
        self, calibration_config: ExperimentConfig, synthetic_brain_dir: Path, tmp_path: Path
    ) -> None:
        """Run two amplitudes, verify persisted measurements, and exercise human-readable output.

        :param calibration_config: Registered synthetic experiment.
        :type calibration_config: ExperimentConfig
        :param synthetic_brain_dir: Small compatible connectome.
        :type synthetic_brain_dir: Path
        :param tmp_path: Isolated YAML directory.
        :type tmp_path: Path
        """
        path = tmp_path / 'calibration.yaml'
        path.write_text(data=effective_yaml(cfg=calibration_config))
        args = [
            'brain',
            'calibrate',
            str(path),
            '--brain-dir',
            str(synthetic_brain_dir),
            '--probes',
            '3',
            '--steps',
            '6',
            '--spontaneous-steps',
            '6',
            '--amplitudes',
            '.05,.2',
        ]
        runner = CliRunner()
        original_threads = numba.get_num_threads()
        result = runner.invoke(app=app, args=[*args, '--json'])
        assert result.exit_code == 0, result.output
        report = json.loads(s=result.stdout)
        assert numba.get_num_threads() == original_threads
        assert json.loads(s=Path(report['output']).read_text()) == report
        assert report['threads'] == 1 and len(report['probe_ids']) == 3
        assert len(report['amplitudes']) == 2
        assert all(len(row['d']) == 6 for row in report['amplitudes'])
        assert all(row['rate_input_hz'] >= 0 for row in report['amplitudes'])
        assert len(report['brain_files_sha256']['weights.npz']) == 64
        with patch(target='flystate.cli.brain.calibrate', return_value=report):
            human = runner.invoke(app=app, args=args)
        assert human.exit_code == 0 and 'half-peak step' in human.stdout
        for value in ('invalid', 'nan', '0', '1.1'):
            invalid = runner.invoke(app=app, args=[*args, '--amplitudes', value, '--json'])
            assert invalid.exit_code == 2 and 'error' in json.loads(s=invalid.stdout)
        invalid = runner.invoke(
            app=app, args=['brain', 'calibrate', str(tmp_path / 'absent'), '--json']
        )
        assert invalid.exit_code == 2
        with patch(target='flystate.cli.brain.calibrate', side_effect=ValueError('no probes')):
            failed = runner.invoke(app=app, args=[*args, '--json'])
        assert failed.exit_code == 1 and json.loads(s=failed.stdout)['error'] == 'no probes'

    def test_zero_drive_and_chunking(
        self, calibration_config: ExperimentConfig, synthetic_brain_dir: Path
    ) -> None:
        """Compare zero-drive pairs, then verify invariance across paired batch sizes.

        :param calibration_config: Registered synthetic experiment.
        :type calibration_config: ExperimentConfig
        :param synthetic_brain_dir: Generated connectome.
        :type synthetic_brain_dir: Path
        """
        with patch.object(
            target=SparseProjectionEncoder, attribute='encode', autospec=True
        ) as encode:
            encode.side_effect = zero_encode
            zero = calibrate(
                cfg=calibration_config,
                paths=get_paths(),
                brain_dir=synthetic_brain_dir,
                amplitudes=[0.1],
                probes=3,
                steps=6,
                spontaneous_steps=6,
            )
        assert zero['amplitudes'][0]['d'] == [0] * 6
        assert 'no_response:0.1' in zero['warnings']
        assert zero['recommendation']['amplitude'] is None
        first = calibrate(
            cfg=calibration_config,
            paths=get_paths(),
            brain_dir=synthetic_brain_dir,
            amplitudes=[0.2],
            probes=3,
            steps=6,
            spontaneous_steps=6,
        )
        cfg = calibration_config.model_copy(
            update={'brain': calibration_config.brain.model_copy(update={'batch_size': 1})}
        )
        second = calibrate(
            cfg=cfg,
            paths=get_paths(),
            brain_dir=synthetic_brain_dir,
            amplitudes=[0.2],
            probes=3,
            steps=6,
            spontaneous_steps=6,
        )
        assert first['amplitudes'] == second['amplitudes']
        assert first['spontaneous'] == second['spontaneous']

    def test_warnings_and_cleanup(
        self, calibration_config: ExperimentConfig, synthetic_brain_dir: Path
    ) -> None:
        """Force firing warnings, then fail a simulation and verify thread restoration.

        :param calibration_config: Registered synthetic experiment.
        :type calibration_config: ExperimentConfig
        :param synthetic_brain_dir: Generated connectome.
        :type synthetic_brain_dir: Path
        """
        with patch(
            target='flystate.brain.calibrate._spontaneous',
            return_value={'rate_all_hz': 6, 'rate_readout_hz': 0},
        ):
            report = calibrate(
                cfg=calibration_config,
                paths=get_paths(),
                brain_dir=synthetic_brain_dir,
                amplitudes=[0.1],
                probes=1,
                steps=1,
                spontaneous_steps=1,
            )
        assert 'runaway' in report['warnings'] and 'silent_readout' in report['warnings']
        original = numba.get_num_threads()
        with (
            patch(
                target='flystate.brain.calibrate._spontaneous', side_effect=RuntimeError('failed')
            ),
            pytest.raises(expected_exception=RuntimeError, match='failed'),
        ):
            calibrate(
                cfg=calibration_config,
                paths=get_paths(),
                brain_dir=synthetic_brain_dir,
                probes=1,
                steps=1,
                spontaneous_steps=1,
            )
        assert numba.get_num_threads() == original
        with pytest.raises(expected_exception=ValueError, match='only'):
            calibrate(
                cfg=calibration_config, paths=get_paths(), brain_dir=synthetic_brain_dir, probes=100
            )

    @pytest.mark.parametrize(
        'amplitudes,probes', [([], 1), ([0], 1), ([2], 1), ([float('nan')], 1), ([0.1], 0)]
    )
    def test_invalid_settings(self, amplitudes: list[float], probes: int) -> None:
        """Reject invalid settings before reading datasets or constructing a brain.

        :param amplitudes: Candidate current scales.
        :type amplitudes: list[float]
        :param probes: Probe count.
        :type probes: int
        """
        with pytest.raises(expected_exception=ValueError):
            calibrate(
                cfg=ExperimentConfig(name='invalid'),
                paths=get_paths(),
                brain_dir=get_paths().brain,
                amplitudes=amplitudes,
                probes=probes,
            )


def zero_encode(
    self: SparseProjectionEncoder, observations: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Replace input drive while preserving every runtime noise operation.

    :param self: Encoder whose neuron count determines the current matrix.
    :type self: SparseProjectionEncoder
    :param observations: Batched RGB observations.
    :type observations: np.ndarray
    :param positions: Batched spatial coordinates, intentionally unused.
    :type positions: np.ndarray
    :returns: Zero float32 currents with the normal encoder shape.
    :rtype: np.ndarray
    """
    return np.zeros(shape=(len(self.input_idx), len(observations)), dtype=np.float32)
