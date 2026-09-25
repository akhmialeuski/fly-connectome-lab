"""The graded backend runs through the standard trace, training and evaluation pipeline (T36)."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from flystate.brain.rate import RateReservoir
from flystate.brain.runtime import EpisodeBrain
from flystate.datasets.preprocess import prepare_dataset
from flystate.diagnostics.rate_access import simulate_states
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.episodes.episode import EpisodeBuilder
from flystate.evaluation.evaluate import evaluate
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.readouts.training import train
from flystate.settings import get_paths
from flystate.traces.builder import build_trace, open_trace

RATE_BRAIN: dict[str, object] = {
    'backend': 'rate',
    'rate': {'gain': 1.0, 'leak': 0.25, 'driven_leak': 1.0, 'input_scale': 20.0},
    'warmup_steps': 0,
    'steps_per_observation': 2,
    'noise': {'enabled': False},
    'threads': 1,
    'batch_size': 7,
}
VOLTAGE: tuple[str, ...] = ('voltage',)
BRAIN: str = 'brain'
RATE: str = 'rate'
READOUT: str = 'readout'
FEATURES: str = 'features'
JSON: str = 'json'
PERSISTENT: str = 'persistent'
RESET: str = 'reset'
EXACTLY_WHEN: str = 'exactly when'
NOISE_FREE: str = 'noise-free'


def _rate_config(base: ExperimentConfig, mode: str) -> ExperimentConfig:
    """Turn the tiny spiking experiment into a graded one with the requested memory mode.

    :param base: Offline synthetic experiment.
    :type base: ExperimentConfig
    :param mode: Memory mode: persistent, reset or reset_concat.
    :type mode: str
    :returns: Validated graded configuration.
    :rtype: ExperimentConfig
    """
    data = base.model_dump(mode=JSON)
    data[BRAIN] = {**data[BRAIN], **RATE_BRAIN}
    data[READOUT][FEATURES] = list(VOLTAGE)
    data['memory'].update(mode=mode)
    data['name'] = f'tiny-rate-{mode.replace("_", "-")}'
    return ExperimentConfig.model_validate(obj=data)


def test_rate_backend_validation_and_stable_spiking_dump(tiny_experiment: ExperimentConfig) -> None:
    """Require a complete noise-free rate section, and keep spiking dumps free of it.

    :param tiny_experiment: Offline synthetic experiment.
    :type tiny_experiment: ExperimentConfig
    """
    assert RATE not in tiny_experiment.model_dump(mode=JSON)[BRAIN]
    assert RATE in _rate_config(tiny_experiment, PERSISTENT).model_dump(mode=JSON)[BRAIN]
    data = _rate_config(tiny_experiment, PERSISTENT).model_dump(mode=JSON)
    for key, value, message in (
        (RATE, None, EXACTLY_WHEN),
        ('noise', {'enabled': True}, NOISE_FREE),
        ('warmup_steps', 5, NOISE_FREE),
    ):
        broken = {**data, BRAIN: {**data[BRAIN], key: value}}
        with pytest.raises(expected_exception=ValidationError, match=message):
            ExperimentConfig.model_validate(obj=broken)
    spikes = {**data, READOUT: {**data[READOUT], FEATURES: ['spike_trace', *VOLTAGE]}}
    with pytest.raises(expected_exception=ValidationError, match='only its graded state'):
        ExperimentConfig.model_validate(obj=spikes)
    spiking_with_rate = tiny_experiment.model_dump(mode=JSON)
    spiking_with_rate[BRAIN][RATE] = RATE_BRAIN[RATE]
    with pytest.raises(expected_exception=ValidationError, match=EXACTLY_WHEN):
        ExperimentConfig.model_validate(obj=spiking_with_rate)


def test_spiking_runtime_refuses_rate_config(tiny_experiment: ExperimentConfig) -> None:
    """Stop any spiking-only code path from silently simulating a graded configuration.

    :param tiny_experiment: Offline synthetic experiment with installed brain files.
    :type tiny_experiment: ExperimentConfig
    """
    cfg = _rate_config(tiny_experiment, PERSISTENT)
    with pytest.raises(expected_exception=ValueError, match='only the flybrain spiking'):
        EpisodeBrain(
            brain_dir=get_paths().brain,
            brain_cfg=cfg.brain,
            readout_cfg=cfg.readout,
            batch_size=1,
            threads=1,
        )


@pytest.mark.parametrize(argnames='mode', argvalues=[PERSISTENT, RESET])
def test_trace_matches_direct_simulation(tiny_experiment: ExperimentConfig, mode: str) -> None:
    """Store exactly the float16-rounded states of the T33 simulation for every observation.

    :param tiny_experiment: Offline synthetic experiment with installed brain files.
    :type tiny_experiment: ExperimentConfig
    :param mode: Memory mode of the trace.
    :type mode: str
    """
    paths = get_paths()
    cfg = _rate_config(tiny_experiment, mode)
    build_trace(cfg=cfg, paths=paths)
    store = open_trace(cfg=cfg, paths=paths)
    prepared = prepare_dataset(cfg=cfg, paths=paths)
    builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
    assert cfg.brain.rate is not None
    reservoir = RateReservoir(
        brain_dir=paths.brain,
        gain=cfg.brain.rate.gain,
        leak=cfg.brain.rate.leak,
        batch_size=len(prepared.samples),
    )
    encoder = SparseProjectionEncoder(
        cfg=cfg.encoder,
        window=cfg.episodes.window,
        candidate_neurons=reservoir.cells(types=[cfg.encoder.target_population]),
    )
    reservoir.leak[encoder.input_idx] = np.float32(1.0)
    episodes = [
        builder.build(sample=sample, image=image)
        for sample, image in zip(prepared.samples, prepared.images, strict=True)
    ]
    inputs = np.stack(
        [encoder.encode(observations=e.observations, positions=e.positions).T for e in episodes]
    )
    readout = reservoir.cells(types=[cfg.readout.population])
    expected = simulate_states(
        reservoir=reservoir,
        input_idx=encoder.input_idx,
        inputs=(inputs * np.float32(cfg.brain.rate.input_scale)).astype(np.float32),
        populations={READOUT: readout},
        steps_per_window=cfg.brain.steps_per_observation,
        reset_each_window=mode == RESET,
    )['state_readout']
    stored = np.asarray(store.array(name=FEATURES)[:], dtype=np.float32)
    assert np.array_equal(stored, expected.astype(np.float16).astype(np.float32))
    assert np.abs(stored).max() > 0


def test_rate_runs_train_and_evaluate(tiny_experiment: ExperimentConfig) -> None:
    """Produce standard run directories for all three memory modes from graded traces.

    :param tiny_experiment: Offline synthetic experiment with installed brain files.
    :type tiny_experiment: ExperimentConfig
    """
    paths = get_paths()
    for mode in (PERSISTENT, RESET, 'reset_concat'):
        cfg = _rate_config(tiny_experiment, mode)
        build_trace(cfg=cfg, paths=paths)
        result = train(cfg=cfg, paths=paths, original_yaml=effective_yaml(cfg=cfg))
        run_dir = Path(result['run_dir'])
        evaluation = evaluate(run_dir=run_dir, paths=paths)
        assert (run_dir / 'model' / 'weights.npz').exists()
        accuracies = list(evaluation['accuracy_by_t'].values())
        assert len(accuracies) == cfg.episodes.steps
        assert all(0 <= value <= 1 for value in accuracies)
