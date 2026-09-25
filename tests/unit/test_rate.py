"""Graded-dynamics reservoir equivalence, determinism and reset behaviour."""

from pathlib import Path

import numba
import numpy as np
from scipy import sparse

from flystate.brain.rate import RateReservoir
from flystate.diagnostics.rate_access import simulate_states, window_recall

GAIN: float = 1.2
LEAK: float = 0.5
BATCH: int = 3
DRIVEN: int = 40
POPULATION: str = 'all'
STATE_KEY: str = 'state_all'


def _inputs(windows: int) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed driven indices and per-window inputs.

    :param windows: Number of windows T.
    :type windows: int
    :returns: Driven indices (I,) int64 and inputs (B,T,I) float32.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    rng = np.random.default_rng(seed=3)
    driven = np.arange(100, 100 + DRIVEN, dtype=np.int64)
    inputs = rng.uniform(low=-1, high=1, size=(BATCH, windows, DRIVEN)).astype(np.float32)
    return driven, inputs


def test_step_matches_reference_equation(synthetic_brain_dir: Path) -> None:
    """Reproduce ``(1-leak) x + leak tanh(gain W x + u)`` with an independent SciPy product.

    :param synthetic_brain_dir: Offline flybrain-format connectome.
    :type synthetic_brain_dir: Path
    """
    reservoir = RateReservoir(brain_dir=synthetic_brain_dir, gain=GAIN, leak=LEAK, batch_size=BATCH)
    meta = np.load(file=synthetic_brain_dir / 'brain.npz')
    weights = sparse.load_npz(file=synthetic_brain_dir / 'weights.npz').tocsr()
    sensory = np.char.find(meta['superclass'].astype(str), 'sensory') >= 0
    effective = (sparse.diags((~sensory).astype(np.float32)) @ weights).astype(np.float64)
    driven, inputs = _inputs(windows=2)
    expected = np.zeros(shape=(reservoir.n, BATCH))
    for window in range(2):
        reservoir.step(input_idx=driven, inputs=np.ascontiguousarray(inputs[:, window].T))
        drive = GAIN * (effective @ expected)
        drive[driven] += inputs[:, window].T
        expected = (1 - LEAK) * expected + LEAK * np.tanh(drive)
    assert np.allclose(reservoir.state, expected, atol=1e-5)
    assert np.abs(reservoir.state).max() > 0


def test_rest_is_zero_and_threads_do_not_change_states(synthetic_brain_dir: Path) -> None:
    """Keep the zero state without input, and give identical states for one and two threads.

    :param synthetic_brain_dir: Offline flybrain-format connectome.
    :type synthetic_brain_dir: Path
    """
    reservoir = RateReservoir(brain_dir=synthetic_brain_dir, gain=GAIN, leak=LEAK, batch_size=BATCH)
    for _ in range(5):
        reservoir.step(input_idx=np.empty(shape=0, dtype=np.int64), inputs=None)
    assert not reservoir.state.any()
    driven, inputs = _inputs(windows=4)
    populations = {POPULATION: np.arange(reservoir.n, dtype=np.int64)}
    runs = []
    for threads in (1, 2):
        numba.set_num_threads(threads)
        runs.append(
            simulate_states(
                reservoir=reservoir,
                input_idx=driven,
                inputs=inputs,
                populations=populations,
                steps_per_window=3,
                reset_each_window=False,
            )[STATE_KEY]
        )
    assert np.array_equal(runs[0], runs[1])


def test_reset_makes_each_window_independent_of_history(synthetic_brain_dir: Path) -> None:
    """Make the last-window state of the reset control depend only on the last window.

    :param synthetic_brain_dir: Offline flybrain-format connectome.
    :type synthetic_brain_dir: Path
    """
    reservoir = RateReservoir(brain_dir=synthetic_brain_dir, gain=GAIN, leak=LEAK, batch_size=BATCH)
    driven, inputs = _inputs(windows=3)
    changed = inputs.copy()
    changed[:, 0] *= -1
    populations = {POPULATION: np.arange(reservoir.n, dtype=np.int64)}
    outcomes = {
        (reset, index): simulate_states(
            reservoir=reservoir,
            input_idx=driven,
            inputs=values,
            populations=populations,
            steps_per_window=4,
            reset_each_window=reset,
        )[STATE_KEY][:, -1]
        for reset in (False, True)
        for index, values in enumerate((inputs, changed))
    }
    assert np.array_equal(outcomes[(True, 0)], outcomes[(True, 1)])
    assert not np.array_equal(outcomes[(False, 0)], outcomes[(False, 1)])


def test_leak_override_applies_only_to_listed_neurons(synthetic_brain_dir: Path) -> None:
    """Replace the driven neurons' state every step while the rest integrate slowly.

    :param synthetic_brain_dir: Offline flybrain-format connectome.
    :type synthetic_brain_dir: Path
    """
    driven, inputs = _inputs(windows=1)
    slow = 0.1
    reservoir = RateReservoir(
        brain_dir=synthetic_brain_dir,
        gain=GAIN,
        leak=slow,
        batch_size=BATCH,
        leak_overrides={1.0: driven},
    )
    assert np.all(reservoir.leak[driven] == 1.0)
    others = np.setdiff1d(np.arange(reservoir.n), driven)
    assert np.all(reservoir.leak[others] == np.float32(slow))
    drive = np.ascontiguousarray(inputs[:, 0].T)
    reservoir.step(input_idx=driven, inputs=drive)
    assert np.allclose(reservoir.state[driven], np.tanh(drive), atol=1e-6)
    assert np.all(reservoir.state[others] == 0)


def test_window_recall_finds_only_the_encoded_window() -> None:
    """Recall the window that the final state encodes and nothing from independent windows."""
    rng = np.random.default_rng(seed=5)
    inputs = rng.normal(size=(200, 4, 30)).astype(np.float32)
    mixing = rng.normal(size=(30, 50)).astype(np.float32)
    final_state = (inputs[:, -1] @ mixing).astype(np.float32)
    recall = window_recall(final_state=final_state, window_inputs=inputs, seed=0)
    assert recall[-1] > 0.95
    assert max(recall[:-1]) < 0.1
