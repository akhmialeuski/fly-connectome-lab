"""Test numerical equivalence and episode identity independently of batching."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import cast

import numba
import numpy as np
import pytest
from flybrain import FlyBrain

from flystate.brain.runtime import EpisodeBrain
from flystate.experiments.config import BrainConfig, NoiseConfig, ReadoutConfig


@pytest.fixture(autouse=True)
def restore_threads() -> Iterator[None]:
    """Restore the process Numba thread setting after every runtime test.

    :returns: One test scope with thread cleanup.
    :rtype: Iterator[None]
    """
    original = numba.get_num_threads()
    yield
    numba.set_num_threads(n=original)


def make_runtime(
    path: Path, batch_size: int = 1, noise: bool = True, warmup: int = 25
) -> EpisodeBrain:
    """Construct a single-threaded fixture runtime with explicit noise behavior.

    :param path: Synthetic brain directory.
    :type path: Path
    :param batch_size: Independent fly count.
    :type batch_size: int
    :param noise: Enable seeded per-fly noise.
    :type noise: bool
    :param warmup: Rest-state warmup steps.
    :type warmup: int
    :returns: Isolated fixture runtime.
    :rtype: EpisodeBrain
    """
    return EpisodeBrain(
        brain_dir=path,
        brain_cfg=BrainConfig(noise=NoiseConfig(enabled=noise), warmup_steps=warmup),
        readout_cfg=ReadoutConfig(),
        batch_size=batch_size,
        threads=1,
    )


class TestEpisodeBrain:
    """Verify deterministic dynamics and scientifically meaningful controls."""

    def test_batch_and_order_independence(self, synthetic_brain_dir: Path) -> None:
        """Track one episode alone, in a full batch, and with padding.

        :param synthetic_brain_dir: Offline signed connectome.
        :type synthetic_brain_dir: Path
        """
        single = make_runtime(path=synthetic_brain_dir)
        batched = make_runtime(path=synthetic_brain_dir, batch_size=3)
        padded = make_runtime(path=synthetic_brain_dir, batch_size=3)
        rest = single.compute_rest_state(seed=7)
        batch_rest = batched.compute_rest_state(seed=7)
        np.testing.assert_array_equal(actual=rest.v, desired=batch_rest.v)
        np.testing.assert_array_equal(actual=rest.fired, desired=batch_rest.fired)
        single.begin(rest=rest, sample_ids=['s1'], seed=11)
        batched.begin(rest=rest, sample_ids=['a', 'b', 's1'], seed=11)
        padded.begin(rest=rest, sample_ids=['s1'], seed=11)
        indices = single.cells(superclasses=['visual_projection'])[:20]
        currents = np.linspace(start=0.01, stop=0.5, num=20, dtype=np.float32)[:, None]
        for _ in range(3):
            first = single.run(input_idx=indices, currents=currents, n_steps=5)
            many = batched.run(
                input_idx=indices, currents=np.repeat(a=currents, repeats=3, axis=1), n_steps=5
            )
            pad = padded.run(
                input_idx=indices, currents=np.repeat(a=currents, repeats=3, axis=1), n_steps=5
            )
            np.testing.assert_array_equal(
                actual=single.features(kinds=['spike_trace', 'voltage'])[0],
                desired=batched.features(kinds=['spike_trace', 'voltage'])[2],
            )
            np.testing.assert_array_equal(
                actual=single.features(kinds=['voltage'])[0],
                desired=padded.features(kinds=['voltage'])[0],
            )
            assert first.spikes_total[0] == many.spikes_total[2] == pad.spikes_total[0]
            assert first.active_neurons[0] == many.active_neurons[2]
            assert first.readout_spikes[0] == many.readout_spikes[2]
            assert first.input_spikes[0] == many.input_spikes[2]
        assert batched.features(kinds=['spike_trace', 'voltage']).shape == (3, 200)

    def test_upstream_equivalence(self, synthetic_brain_dir: Path) -> None:
        """Compare every voltage and spike for fifty noiseless upstream steps.

        :param synthetic_brain_dir: Offline signed connectome.
        :type synthetic_brain_dir: Path
        """
        runtime = make_runtime(path=synthetic_brain_dir, noise=False, warmup=0)
        rest = runtime.compute_rest_state(seed=7)
        runtime.begin(rest=rest, sample_ids=['s1'], seed=7)
        upstream = FlyBrain(data=synthetic_brain_dir, seed=0, device='cpu', sensory_input=False)
        upstream.noise_amp = 0.0
        indices = np.arange(20, dtype=np.int64)
        currents = np.full(shape=(20, 1), fill_value=0.3, dtype=np.float32)
        expected_trace = np.zeros(shape=100, dtype=np.float32)
        decay = np.float32(np.exp(-0.02 / 0.1))
        for _ in range(50):
            report = runtime.run(input_idx=indices, currents=currents, n_steps=1)
            fired = upstream.step(inject=[(indices, 0.3)])
            assert isinstance(fired, np.ndarray)
            np.testing.assert_array_equal(actual=runtime._fb.fired, desired=fired)
            np.testing.assert_array_equal(actual=runtime._fb.v, desired=upstream.v)
            expected_trace *= decay
            expected_trace[fired[fired < 100]] += 1.0
            np.testing.assert_array_equal(
                actual=runtime.features(kinds=['spike_trace'])[0], desired=expected_trace
            )
            assert report.spikes_total[0] == len(fired)
            assert report.active_neurons[0] == len(fired)
            assert report.readout_spikes[0] == np.count_nonzero(fired < 100)
            assert report.input_spikes[0] == np.count_nonzero(fired < 20)

    def test_rest_and_continuing_noise(self, synthetic_brain_dir: Path) -> None:
        """Restore dynamic state without rewinding stochastic streams or aliasing arrays.

        :param synthetic_brain_dir: Offline signed connectome.
        :type synthetic_brain_dir: Path
        """
        runtime = make_runtime(path=synthetic_brain_dir, batch_size=2)
        rest = runtime.compute_rest_state(seed=7)
        repeated = runtime.compute_rest_state(seed=7)
        np.testing.assert_array_equal(actual=rest.v, desired=repeated.v)
        np.testing.assert_array_equal(actual=rest.fired, desired=repeated.fired)
        assert np.mean(a=rest.v) > 0.1
        assert not rest.v.flags.writeable
        assert not rest.fired.flags.writeable
        runtime.begin(rest=rest, sample_ids=['same', 'same'], seed=3)
        runtime.run(
            input_idx=np.arange(20, dtype=np.int64),
            currents=np.ones(shape=(20, 2), dtype=np.float32),
            n_steps=5,
        )
        state = runtime._generators[0].bit_generator.state
        runtime.restore_rest(rest=rest)
        assert runtime._generators[0].bit_generator.state == state
        np.testing.assert_array_equal(
            actual=runtime._fb.v, desired=np.repeat(a=rest.v[:, None], repeats=2, axis=1)
        )
        assert not runtime.features(kinds=['spike_trace']).any()
        assert not runtime.last_noise_kicks.any()
        runtime.run(input_idx=np.empty(shape=0, dtype=np.int64), currents=None, n_steps=2)
        assert runtime._generators[0].bit_generator.state != state
        np.testing.assert_array_equal(
            actual=runtime.features(kinds=['voltage'])[0],
            desired=runtime.features(kinds=['voltage'])[1],
        )
        saved = runtime.features(kinds=['voltage'])
        saved.fill(-10)
        assert np.all(runtime.features(kinds=['voltage']) != -10)

    def test_noise_statistics(self, synthetic_brain_dir: Path) -> None:
        """Check the binomial mean within five standard errors over 500 steps.

        :param synthetic_brain_dir: Offline signed connectome.
        :type synthetic_brain_dir: Path
        """
        runtime = make_runtime(path=synthetic_brain_dir, batch_size=3)
        runtime.begin(rest=runtime.compute_rest_state(seed=0), sample_ids=['a', 'b', 'c'], seed=0)
        kicks: list[np.ndarray] = []
        for _ in range(500):
            runtime.run(input_idx=np.empty(shape=0, dtype=np.int64), currents=None, n_steps=1)
            kicks.append(runtime.last_noise_kicks.copy())
        expected = runtime.n * 1.2 * 0.02
        standard_error = np.sqrt(expected * (1 - 1.2 * 0.02) / (500 * 3))
        assert abs(np.mean(a=kicks) - expected) < 5 * standard_error

    def test_reject_invalid_state_and_input(self, synthetic_brain_dir: Path) -> None:
        """Validate arrays before any simulation state is mutated.

        :param synthetic_brain_dir: Offline signed connectome.
        :type synthetic_brain_dir: Path
        """
        runtime = make_runtime(path=synthetic_brain_dir)
        empty = np.empty(shape=0, dtype=np.int64)
        with pytest.raises(expected_exception=RuntimeError, match='begin'):
            runtime.run(input_idx=empty, currents=None, n_steps=1)
        rest = runtime.compute_rest_state(seed=0)
        with pytest.raises(expected_exception=ValueError, match='batch_size'):
            runtime.begin(rest=rest, sample_ids=['a', 'b'], seed=0)
        invalid_voltages: tuple[np.ndarray, ...] = (
            np.zeros(shape=2, dtype=np.float32),
            rest.v.astype(np.float64),
            np.full_like(a=rest.v, fill_value=np.nan),
        )
        for invalid in invalid_voltages:
            with pytest.raises(expected_exception=ValueError, match='Rest voltages'):
                runtime.restore_rest(rest=replace(rest, v=invalid))
        for invalid_spikes in (
            np.array([[-1]], dtype=np.int64),
            np.array([0.0]),
            np.array([-1], dtype=np.int64),
            np.array([runtime.n], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
        ):
            with pytest.raises(expected_exception=ValueError, match='Rest spikes'):
                runtime.restore_rest(rest=replace(rest, fired=invalid_spikes))
        for invalid_idx in (
            np.array([[0]], dtype=np.int64),
            np.array([0.0]),
            np.array([-1], dtype=np.int64),
            np.array([runtime.n], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
        ):
            with pytest.raises(expected_exception=ValueError, match='Input indices'):
                runtime.run(input_idx=invalid_idx, currents=None, n_steps=1)
        idx = np.arange(2, dtype=np.int64)
        invalid_currents: tuple[np.ndarray, ...] = (
            np.zeros(shape=(2, 2), dtype=np.float32),
            np.zeros(shape=(2, 1), dtype=np.float64),
            np.full(shape=(2, 1), fill_value=np.nan, dtype=np.float32),
        )
        for invalid_current in invalid_currents:
            with pytest.raises(expected_exception=ValueError, match='Currents'):
                runtime.run(input_idx=idx, currents=cast(np.ndarray, invalid_current), n_steps=1)
        with pytest.raises(expected_exception=ValueError, match='n_steps'):
            runtime.run(input_idx=empty, currents=None, n_steps=-1)
        for kinds in ([], ['unknown'], ['voltage', 'voltage']):
            with pytest.raises(expected_exception=ValueError, match='Features'):
                runtime.features(kinds=kinds)

    def test_constructor_errors(self, synthetic_brain_dir: Path) -> None:
        """Fail explicitly rather than downloading missing brain files implicitly.

        :param synthetic_brain_dir: Offline signed connectome.
        :type synthetic_brain_dir: Path
        """
        with pytest.raises(expected_exception=ValueError, match='batch_size'):
            make_runtime(path=synthetic_brain_dir, batch_size=0)
        with pytest.raises(expected_exception=FileNotFoundError, match='explicitly'):
            make_runtime(path=synthetic_brain_dir / 'missing')
        with pytest.raises(expected_exception=ValueError, match='Empty readout'):
            EpisodeBrain(
                brain_dir=synthetic_brain_dir,
                brain_cfg=BrainConfig(),
                readout_cfg=ReadoutConfig(population='absent'),
                batch_size=1,
                threads=1,
            )
