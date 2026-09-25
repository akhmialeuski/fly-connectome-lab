"""Regression coverage for immutable configuration and safe YAML overrides."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.experiments.config import (
    ConfigError,
    ExperimentConfig,
    apply_overrides,
    config_hash,
    effective_yaml,
    load_config,
    section_hash,
)
from flystate.hashing import sha256_obj

CONFIG_DIRECTORY: Path = Path(__file__).resolve().parents[2] / 'configs'


class TestConfiguration:
    """Validate committed experiments and scientific feasibility constraints."""

    def test_examples(self) -> None:
        """Load all complete examples and check the isolated control differences."""
        configs = {path.stem: load_config(path=path) for path in CONFIG_DIRECTORY.glob('*.yaml')}
        assert set(configs) == {
            'celeba-smoke',
            'celeba-persistent',
            'celeba-reset',
            'celeba-confirm',
        }
        for name, cfg in configs.items():
            supplied = yaml.safe_load(stream=(CONFIG_DIRECTORY / f'{name}.yaml').read_text())
            assert supplied == cfg.model_dump(mode='json')
        persistent = configs['celeba-persistent'].model_dump(mode='json')
        reset = configs['celeba-reset'].model_dump(mode='json')
        assert persistent.pop('name') == 'celeba-persistent'
        assert reset.pop('name') == 'celeba-reset'
        assert persistent.pop('memory') == {'mode': 'persistent'}
        assert reset.pop('memory') == {'mode': 'reset'}
        assert persistent == reset
        smoke = configs['celeba-smoke'].model_dump(mode='json')
        confirm = configs['celeba-confirm'].model_dump(mode='json')
        assert (smoke.pop('name'), confirm.pop('name')) == ('celeba-smoke', 'celeba-confirm')
        assert smoke['dataset']['subset'].pop('selection_seed') == 0
        assert confirm['dataset']['subset'].pop('selection_seed') == 2
        assert smoke == confirm

    @pytest.mark.parametrize(
        'override',
        [
            'extra=1',
            'brain.extra=1',
            'dataset.split.fractions=[0.6,0.2,0.3]',
            'dataset.split.fractions=[0,0.5,0.5]',
            'episodes.window=129',
            'episodes.trajectory.strategy=raster',
            'seed=-1',
            'seed=true',
            'brain.dt_s=0',
            'brain.noise.rate_hz=51',
            'encoder.amplitude=.nan',
            'readout.trace_tau_s=.inf',
            'readout.features=[]',
            'readout.features=[voltage,voltage]',
            'readout.c_grid=[]',
            'readout.c_grid=[0]',
            'readout.cv_folds=15',
            'brain.sensory_input=true',
            'dataset.split.fractions=[0.98,0.01,0.01]',
            'name=Bad_Name',
            'episodes.window=1',
            'dataset.preprocess.interocular=200',
            'dataset.preprocess.eye_center=[64,128]',
            'brain.threads=0',
        ],
    )
    def test_invalid_values(self, tmp_path: Path, override: str) -> None:
        """Reject impossible, nonfinite, unknown, and ill-typed values.

        :param tmp_path: Isolated YAML directory.
        :type tmp_path: Path
        :param override: One invalid field assignment.
        :type override: str
        """
        source = tmp_path / 'experiment.yaml'
        source.write_text(data='name: example\n', encoding='utf-8')
        with pytest.raises(expected_exception=ConfigError):
            load_config(path=source, overrides=[override])

    def test_immutable(self) -> None:
        """Freeze nested models and sequence contents, including default values."""
        cfg = ExperimentConfig(name='example')
        with pytest.raises(expected_exception=ValidationError):
            cfg.brain.batch_size = 2
        assert isinstance(cfg.readout.features, tuple)
        assert isinstance(cfg.readout.c_grid, tuple)
        assert isinstance(cfg.dataset.split.fractions, tuple)
        assert isinstance(cfg.dataset.preprocess.eye_center, tuple)

    def test_hashes(self, tmp_path: Path) -> None:
        """Prove key-order independence, round trips, and section isolation.

        :param tmp_path: Temporary YAML directory.
        :type tmp_path: Path
        """
        cfg = ExperimentConfig(name='example')
        data = cfg.model_dump(mode='json')
        source = tmp_path / 'experiment.yaml'
        source.write_text(data=yaml.safe_dump(data=dict(reversed(list(data.items())))))
        restored = load_config(path=source)
        assert config_hash(cfg=restored) == config_hash(cfg=cfg)
        assert yaml.safe_load(stream=effective_yaml(cfg=cfg)) == data
        changed = load_config(path=source, overrides=['brain.steps_per_observation=3'])
        assert config_hash(cfg=changed) != config_hash(cfg=cfg)
        assert section_hash(cfg=changed, section='encoder') == section_hash(
            cfg=cfg, section='encoder'
        )
        assert section_hash(cfg=cfg, section='brain') == sha256_obj(obj=data['brain'])
        with pytest.raises(expected_exception=ConfigError, match='Unknown configuration section'):
            section_hash(cfg=cfg, section='unknown')

    @pytest.mark.parametrize('strategy,steps', [('random', 1), ('raster', 4), ('permuted', 16)])
    def test_valid_trajectories(self, strategy: str, steps: int) -> None:
        """Accept supported random and square-grid observation counts.

        :param strategy: Trajectory strategy.
        :type strategy: str
        :param steps: Observation count.
        :type steps: int
        """
        cfg = ExperimentConfig.model_validate(
            obj={
                'name': 'example',
                'episodes': {'steps': steps, 'trajectory': {'strategy': strategy}},
            }
        )
        assert cfg.episodes.steps == steps


class TestOverrides:
    """Exercise partial documents, explicit nulls, syntax, and input preservation."""

    def test_defaults_and_input_preservation(self) -> None:
        """Allow omitted schema fields without mutating source dictionaries."""
        data: dict[str, object] = {'name': 'example', 'brain': {'batch_size': 2}}
        changed = apply_overrides(
            data=data,
            overrides=[
                'brain.steps_per_observation=3',
                'brain.threads=2',
                'brain.threads=null',
                'brain.noise.enabled=false',
                'readout.c_grid=[0.5,2.0]',
                'name=renamed',
            ],
        )
        cfg = ExperimentConfig.model_validate(obj=changed)
        assert cfg.name == 'renamed'
        assert cfg.brain.batch_size == 2
        assert cfg.brain.steps_per_observation == 3
        assert cfg.brain.threads is None
        assert cfg.brain.noise.enabled is False
        assert cfg.readout.c_grid == (0.5, 2.0)
        assert data == {'name': 'example', 'brain': {'batch_size': 2}}
        supplied_name = apply_overrides(data={}, overrides=['name=provided'])
        assert ExperimentConfig.model_validate(obj=supplied_name).name == 'provided'

    @pytest.mark.parametrize(
        'override',
        [
            'brain',
            '=3',
            'brain.threads=',
            'brain.threads=[',
            'brain..threads=2',
            'brain.threads.value=3',
            'readout.features.0=voltage',
            'unknown=1',
        ],
    )
    def test_bad_syntax(self, override: str) -> None:
        """Reject malformed and unknown dotted assignments.

        :param override: Invalid assignment.
        :type override: str
        """
        with pytest.raises(expected_exception=ConfigError):
            apply_overrides(data={'name': 'example'}, overrides=[override])

    def test_invalid_parent(self) -> None:
        """Never silently replace an explicitly invalid parent mapping."""
        with pytest.raises(expected_exception=ConfigError, match='parent is not a mapping'):
            apply_overrides(data={'brain': None}, overrides=['brain.threads=2'])

    @pytest.mark.parametrize(
        'document',
        [
            '',
            '[]',
            'name: [',
            '{}',
            'name: example\nextra: 1',
            'name: example\nbrain:\n  extra: 1',
            'name: example\nbrain: null',
        ],
    )
    def test_bad_documents(self, tmp_path: Path, document: str) -> None:
        """Reject missing fields, wrong roots, malformed YAML, and unknown keys.

        :param tmp_path: Temporary source directory.
        :type tmp_path: Path
        :param document: Invalid YAML source.
        :type document: str
        """
        source = tmp_path / 'invalid.yaml'
        source.write_text(data=document, encoding='utf-8')
        with pytest.raises(expected_exception=ConfigError):
            load_config(path=source)

    def test_unreadable_source(self, tmp_path: Path) -> None:
        """Wrap filesystem and encoding failures in the public exception.

        :param tmp_path: Temporary source directory.
        :type tmp_path: Path
        """
        source = tmp_path / 'missing.yaml'
        with pytest.raises(expected_exception=ConfigError):
            load_config(path=source)
        source.write_bytes(data=b'\xff')
        with pytest.raises(expected_exception=ConfigError):
            load_config(path=source)

    def test_cli(self, tmp_path: Path) -> None:
        """Exercise help, valid overrides, human output, and both failure modes.

        :param tmp_path: Temporary invalid document directory.
        :type tmp_path: Path
        """
        runner = CliRunner()
        command = ['experiment', 'validate']
        assert runner.invoke(app=app, args=[*command, '--help']).exit_code == 0
        source = str(CONFIG_DIRECTORY / 'celeba-smoke.yaml')
        result = runner.invoke(
            app=app, args=[*command, source, '--set', 'brain.batch_size=1', '--json']
        )
        assert result.exit_code == 0, result.exception
        payload = json.loads(s=result.stdout)
        assert len(payload['config_hash']) == 64
        assert yaml.safe_load(stream=payload['effective_yaml'])['brain']['batch_size'] == 1
        assert runner.invoke(app=app, args=[*command, source]).exit_code == 0
        for output in ([], ['--json']):
            result = runner.invoke(app=app, args=[*command, str(tmp_path / 'missing'), *output])
            assert result.exit_code == 2
            assert 'invalid_config' in result.stderr
            if output:
                assert 'error' in json.loads(s=result.stdout)
            else:
                assert result.stdout == ''
