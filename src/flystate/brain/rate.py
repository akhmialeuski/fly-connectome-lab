"""Deterministic graded (rate) dynamics on flybrain's effective MaleCNS connectome.

The spiking runtime in :mod:`flystate.brain.runtime` follows flybrain's leaky integrate-and-fire
model. This module keeps the same effective graph, including flybrain's removal of synapses onto
sensory neurons, but replaces spikes with a leaky echo-state update, the neuron model used by
connectome reservoirs such as conn2res, wetware and fly-self-driving:

``x[t+1] = (1 - leak) * x[t] + leak * tanh(gain * W @ x[t] + u[t])``

There is no noise and no tonic drive, so ``x = 0`` is the rest state and every result is a
deterministic function of the input.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numba
import numpy as np
from flybrain import FlyBrain
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigs
from threadpoolctl import threadpool_limits

from flystate.brain.runtime import RestState, RunSummary
from flystate.experiments.config import BrainConfig, ReadoutConfig
from flystate.hashing import stable_int

SHUFFLE_NAMESPACE: str = 'degree-preserving-shuffle'
RANDOM_TARGET_NAMESPACE: str = 'random-target-shuffle'
# ARPACK with a single requested eigenvalue can settle on a close neighbour of the largest one
# (0.2361 instead of 0.2365 on the synthetic graph); six give the dense answer to 1e-15.
EIGS_COUNT: int = 6
# ARPACK needs k < n - 1; smaller components are solved densely.
MIN_EIGS_SIZE: int = EIGS_COUNT + 2
EIGS_MAX_ITERATIONS: int = 10000
VOLTAGE: str = 'voltage'
# The same numba.prange object under a name mypy treats as a callable returning a range.
PARALLEL_RANGE: Any = numba.prange


@numba.njit(parallel=True, cache=True)
def _propagate(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int32],
    weights: NDArray[np.float32],
    state: NDArray[np.float32],
    out: NDArray[np.float32],
) -> None:  # pragma: no cover - compiled; covered through RateReservoir.step
    """Write ``W @ state`` row by row; each row is summed by one thread in edge order.

    :param indptr: CSR row pointers of the postsynaptic-row matrix, shape (N+1,).
    :type indptr: NDArray[np.int64]
    :param indices: CSR presynaptic column indices, shape (E,).
    :type indices: NDArray[np.int32]
    :param weights: CSR signed dimensionless weights, shape (E,).
    :type weights: NDArray[np.float32]
    :param state: Presynaptic states, shape (N,B), float32.
    :type state: NDArray[np.float32]
    :param out: Destination for postsynaptic input, shape (N,B), float32.
    :type out: NDArray[np.float32]
    """
    for row in PARALLEL_RANGE(len(indptr) - 1):
        accumulator = np.zeros(state.shape[1], dtype=np.float32)
        for edge in range(indptr[row], indptr[row + 1]):
            accumulator += weights[edge] * state[indices[edge]]
        out[row] = accumulator


def _retarget(
    matrix: sparse.csr_matrix | sparse.csr_array,
    seed: int,
    namespace: str,
    targets: Callable[[np.random.Generator, NDArray[np.int32]], NDArray[np.int32]],
) -> sparse.csr_matrix:
    """Move every edge to a drawn postsynaptic target, then restore each row's absolute sum.

    :param matrix: Effective weights with postsynaptic rows, shape (N,N), float32.
    :type matrix: sparse.csr_matrix | sparse.csr_array
    :param seed: Nonnegative null-model seed.
    :type seed: int
    :param namespace: Name that separates the random streams of different null models.
    :type namespace: str
    :param targets: Draws the new target row of every edge from the canonical rows, shape (E,).
    :type targets: Callable[[np.random.Generator, NDArray[np.int32]], NDArray[np.int32]]
    :returns: Retargeted, renormalized float32 CSR matrix, shape (N,N).
    :rtype: sparse.csr_matrix
    """
    # Canonical order on a copy: the draw must not depend on the caller's storage order.
    canonical = sparse.csr_matrix(matrix, copy=True)
    canonical.sum_duplicates()
    canonical.sort_indices()
    coo = canonical.tocoo()
    generator = np.random.default_rng(
        seed=np.random.SeedSequence(entropy=[seed, stable_int(key=namespace)])
    )
    shuffled = sparse.csr_matrix(
        (coo.data.astype(np.float64), (targets(generator, coo.row), coo.col)), shape=matrix.shape
    )
    shuffled.sum_duplicates()
    before = np.asarray(abs(canonical).sum(axis=1), dtype=np.float64).ravel()
    after = np.asarray(abs(shuffled).sum(axis=1), dtype=np.float64).ravel()
    scale = np.divide(before, after, out=np.zeros_like(before), where=after > 0)
    shuffled = sparse.diags(scale) @ shuffled
    return sparse.csr_matrix(shuffled, dtype=np.float32)


def degree_preserving_shuffle(
    matrix: sparse.csr_matrix | sparse.csr_array, seed: int
) -> sparse.csr_matrix:
    """Permute the postsynaptic targets of all edges, then restore flybrain's row normalization.

    Every edge keeps its presynaptic neuron and weight, so each neuron keeps its out-degree and the
    sign of its outgoing weights. Targets are permuted over all edges, so each neuron also keeps its
    in-degree, and rows that were empty (sensory neurons) stay empty. Parallel edges created by the
    permutation are summed. Each nonempty row is rescaled so that its absolute weights sum to the
    same value as before, which is 1 for every neuron with inputs.

    :param matrix: Effective weights with postsynaptic rows, shape (N,N), float32.
    :type matrix: sparse.csr_matrix | sparse.csr_array
    :param seed: Nonnegative shuffle seed.
    :type seed: int
    :returns: Shuffled, renormalized float32 CSR matrix, shape (N,N).
    :rtype: sparse.csr_matrix
    """
    return _retarget(
        matrix=matrix,
        seed=seed,
        namespace=SHUFFLE_NAMESPACE,
        targets=lambda generator, rows: generator.permutation(rows),
    )


def random_target_shuffle(
    matrix: sparse.csr_matrix | sparse.csr_array, seed: int
) -> sparse.csr_matrix:
    """Send every edge to a uniformly drawn neuron that has inputs, then restore row normalization.

    Each neuron keeps its out-degree, the sign of its outgoing weights and their magnitudes, as in
    :func:`degree_preserving_shuffle`, but in-degrees become binomial instead of the fly's.
    Targets are drawn with replacement from the rows that had inputs, so sensory rows stay empty.
    A row that had inputs but receives no edge stays empty; for the MaleCNS graph, with a mean
    in-degree near 150, that has negligible probability.

    :param matrix: Effective weights with postsynaptic rows, shape (N,N), float32.
    :type matrix: sparse.csr_matrix | sparse.csr_array
    :param seed: Nonnegative null-model seed.
    :type seed: int
    :returns: Retargeted, renormalized float32 CSR matrix, shape (N,N).
    :rtype: sparse.csr_matrix
    """
    receiving = np.flatnonzero(np.diff(sparse.csr_matrix(matrix).indptr) > 0).astype(np.int32)
    return _retarget(
        matrix=matrix,
        seed=seed,
        namespace=RANDOM_TARGET_NAMESPACE,
        targets=lambda generator, rows: generator.choice(a=receiving, size=len(rows)),
    )


def feedforward_only(
    matrix: sparse.csr_matrix | sparse.csr_array, sources: NDArray[np.int64]
) -> sparse.csr_matrix:
    """Keep only the synapses whose presynaptic neuron is listed, with unchanged weights.

    With the driven neurons as sources, every other neuron receives exactly the drive it receives
    in the full graph from those neurons, but nothing from any non-driven neuron. Rows are not
    renormalized, so no synapse is strengthened.

    :param matrix: Effective weights with postsynaptic rows, shape (N,N), float32.
    :type matrix: sparse.csr_matrix | sparse.csr_array
    :param sources: Presynaptic neuron indices to keep, shape (S,), int64.
    :type sources: NDArray[np.int64]
    :returns: Pruned float32 CSR matrix, shape (N,N).
    :rtype: sparse.csr_matrix
    """
    pruned = sparse.csr_matrix(matrix, copy=True, dtype=np.float32)
    pruned.data[~np.isin(element=pruned.indices, test_elements=sources)] = 0
    pruned.eliminate_zeros()
    return pruned


def giant_component_radius(matrix: sparse.csr_matrix | sparse.csr_array) -> float:
    """Return the spectral radius of the largest strongly connected component.

    The eigenvalues of a directed graph are the union of those of its strongly connected
    components, so this is the radius of the recurrent core that nearly all neurons belong to. It
    ignores small isolated loops, such as the MaleCNS pair of ENS neurons that are each other's
    only input and alone give eigenvalues of modulus 1.

    :param matrix: Signed weights with postsynaptic rows, shape (N,N).
    :type matrix: sparse.csr_matrix | sparse.csr_array
    :returns: Largest eigenvalue modulus of the giant component, dimensionless, identical on
        every call for the same matrix.
    :rtype: float
    """
    graph = sparse.csr_matrix(matrix)
    _, labels = connected_components(csgraph=graph, directed=True, connection='strong')
    members = np.flatnonzero(labels == np.bincount(labels).argmax())
    core = graph[members][:, members].astype(np.float64)
    if members.size < MIN_EIGS_SIZE:
        return float(np.abs(np.linalg.eigvals(core.toarray())).max())
    # ARPACK starts from a random vector unless given one; a fixed start and one BLAS thread make
    # the radius, and therefore every gain derived from it, bitwise reproducible.
    start = np.full(shape=members.size, fill_value=1 / np.sqrt(members.size))
    with threadpool_limits(limits=1, user_api='blas'):
        values = eigs(
            A=core,
            k=EIGS_COUNT,
            which='LM',
            v0=start,
            return_eigenvectors=False,
            maxiter=EIGS_MAX_ITERATIONS,
        )
    return float(np.abs(values).max())


class RateReservoir:
    """Batched leaky-tanh dynamics on the unchanged effective flybrain weight matrix."""

    def __init__(
        self,
        brain_dir: Path,
        gain: float,
        leak: float,
        batch_size: int,
        leak_overrides: dict[float, NDArray[np.int64]] | None = None,
        shuffle_seed: int | None = None,
        transform: Callable[[sparse.csr_matrix], sparse.csr_matrix] | None = None,
    ) -> None:
        """Load the effective graph exactly as flybrain builds it for ``sensory_input=False``.

        :param brain_dir: Directory with flybrain ``brain.npz`` and ``weights.npz``.
        :type brain_dir: Path
        :param gain: Positive multiplier of the row-normalized weights, dimensionless.
        :type gain: float
        :param leak: Update fraction in (0, 1] for every neuron; 1 replaces the state each step.
        :type leak: float
        :param batch_size: Positive number of independent episodes advanced together.
        :type batch_size: int
        :param leak_overrides: Optional leak values for listed neuron indices, each shape (K,).
        :type leak_overrides: Optional[dict[float, NDArray[np.int64]]]
        :param shuffle_seed: When given, replace the graph by a degree-preserving shuffle (see
            :func:`degree_preserving_shuffle`); the unchanged graph otherwise.
        :type shuffle_seed: Optional[int]
        :param transform: Optional replacement of the (possibly shuffled) graph, for null models
            such as :func:`random_target_shuffle` or :func:`feedforward_only`.
        :type transform: Optional[Callable[[sparse.csr_matrix], sparse.csr_matrix]]
        :raises ValueError: If a parameter is outside its valid range.
        """
        overrides = leak_overrides or {}
        if not (
            gain > 0 and batch_size >= 1 and all(0 < value <= 1 for value in (leak, *overrides))
        ):
            raise ValueError('Gain must be positive, leaks in (0, 1], and batch_size positive.')
        spiking = FlyBrain(data=brain_dir, seed=0, device='cpu', batch=1, sensory_input=False)
        matrix = sparse.csc_matrix(
            (spiking.weights, spiking.indices, spiking.indptr), shape=(spiking.n, spiking.n)
        ).tocsr()
        if shuffle_seed is not None:
            matrix = degree_preserving_shuffle(matrix=matrix, seed=shuffle_seed)
        if transform is not None:
            matrix = sparse.csr_matrix(transform(sparse.csr_matrix(matrix)), dtype=np.float32)
        matrix.sort_indices()
        self._flybrain: FlyBrain = spiking
        self.n: int = spiking.n
        self.edges: int = int(matrix.nnz)
        self.gain: np.float32 = np.float32(gain)
        self.leak: NDArray[np.float32] = np.full(
            shape=(spiking.n, 1), fill_value=leak, dtype=np.float32
        )
        for value, indices in overrides.items():
            self.leak[indices] = value
        self._indptr: NDArray[np.int64] = matrix.indptr.astype(np.int64)
        self._indices: NDArray[np.int32] = matrix.indices.astype(np.int32)
        self._weights: NDArray[np.float32] = matrix.data.astype(np.float32)
        self.state: NDArray[np.float32] = np.zeros(shape=(self.n, batch_size), dtype=np.float32)
        self._drive: NDArray[np.float32] = np.empty_like(self.state)

    def cells(self, types: Sequence[str]) -> NDArray[np.int64]:
        """Select sorted neuron indices by superclass or cell type, as flybrain does.

        :param types: Superclass or cell-type names.
        :type types: Sequence[str]
        :returns: Sorted int64 neuron indices, shape (K,).
        :rtype: NDArray[np.int64]
        """
        return np.asarray(self._flybrain.cells(types=list(types)), dtype=np.int64)

    def matrix(self) -> sparse.csr_matrix:
        """Return the simulated weights without the gain, sharing the stored arrays.

        :returns: Signed float32 CSR matrix with postsynaptic rows, shape (N,N).
        :rtype: sparse.csr_matrix
        """
        return sparse.csr_matrix(
            (self._weights, self._indices, self._indptr), shape=(self.n, self.n), copy=False
        )

    def reset(self) -> None:
        """Return every episode to the zero rest state."""
        self.state.fill(0)

    def step(self, input_idx: NDArray[np.int64], inputs: NDArray[np.float32] | None) -> None:
        """Advance every episode by one synaptic update.

        :param input_idx: Distinct driven neuron indices, shape (I,), int64.
        :type input_idx: NDArray[np.int64]
        :param inputs: Additive input to the driven neurons, shape (I,B), float32, or none.
        :type inputs: Optional[NDArray[np.float32]]
        :raises ValueError: If the input shape does not match the driven neurons and batch.
        """
        if inputs is not None and inputs.shape != (len(input_idx), self.state.shape[1]):
            raise ValueError('Inputs must have shape (len(input_idx), batch_size).')
        _propagate(self._indptr, self._indices, self._weights, self.state, self._drive)
        self._drive *= self.gain
        if inputs is not None:
            self._drive[input_idx] += inputs
        np.tanh(self._drive, out=self._drive)
        self.state *= np.float32(1) - self.leak
        self._drive *= self.leak
        self.state += self._drive


class RateEpisodeBrain:
    """Run the graded model behind the trace builder's ``EpisodeBrain`` interface.

    The rest state is zero and the model is deterministic, so episode seeds and warmup have no
    effect. The only recorded feature block is the graded state, stored under ``voltage``. No
    spikes exist, so every spike count in a run summary is zero.
    """

    def __init__(
        self,
        brain_dir: Path,
        brain_cfg: BrainConfig,
        readout_cfg: ReadoutConfig,
        batch_size: int,
        threads: int,
    ) -> None:
        """Load the effective graph and select the readout population.

        :param brain_dir: Directory with flybrain ``brain.npz`` and ``weights.npz``.
        :type brain_dir: Path
        :param brain_cfg: Configuration with ``backend`` rate and its ``rate`` parameters.
        :type brain_cfg: BrainConfig
        :param readout_cfg: Readout population; only the ``voltage`` block is recorded.
        :type readout_cfg: ReadoutConfig
        :param batch_size: Positive number of episodes advanced together.
        :type batch_size: int
        :param threads: Numba threads for the propagation kernel.
        :type threads: int
        :raises ValueError: If the configuration is not a rate one or the readout is empty.
        """
        if brain_cfg.backend != 'rate' or brain_cfg.rate is None:
            raise ValueError('RateEpisodeBrain requires brain.backend rate with brain.rate set.')
        numba.set_num_threads(threads)
        self._rate = brain_cfg.rate
        self._reservoir = RateReservoir(
            brain_dir=brain_dir,
            gain=brain_cfg.rate.gain,
            leak=brain_cfg.rate.leak,
            batch_size=batch_size,
        )
        self._driven: NDArray[np.int64] | None = None
        self.n: int = self._reservoir.n
        self.batch_size: int = batch_size
        self.threads: int = threads
        self.readout_idx: NDArray[np.int64] = self.cells(superclasses=[readout_cfg.population])
        if not self.readout_idx.size:
            raise ValueError(f'Empty readout population: {readout_cfg.population}.')

    def cells(self, superclasses: Sequence[str]) -> NDArray[np.int64]:
        """Select sorted neuron indices by superclass or cell type.

        :param superclasses: Population names passed to flybrain.
        :type superclasses: Sequence[str]
        :returns: Sorted int64 neuron indices, shape (K,).
        :rtype: NDArray[np.int64]
        """
        return self._reservoir.cells(types=superclasses)

    def compute_rest_state(self, seed: int) -> RestState:
        """Return the zero rest state; the graded model needs no warmup.

        :param seed: Experiment seed, recorded only.
        :type seed: int
        :returns: Zero float32 voltages (N,) and no spikes.
        :rtype: RestState
        """
        voltages = np.zeros(shape=self.n, dtype=np.float32)
        fired = np.empty(shape=0, dtype=np.int64)
        voltages.flags.writeable = False
        fired.flags.writeable = False
        return RestState(v=voltages, fired=fired, warmup_steps=0, seed=seed)

    def begin(self, rest: RestState, sample_ids: Sequence[str], seed: int) -> None:
        """Start new episodes from the rest state; the deterministic model ignores the seed.

        :param rest: Rest state from ``compute_rest_state``.
        :type rest: RestState
        :param sample_ids: At most B episode identifiers.
        :type sample_ids: Sequence[str]
        :param seed: Experiment seed, unused by the noise-free model.
        :type seed: int
        :raises ValueError: If the episode count exceeds the batch capacity.
        """
        if len(sample_ids) > self.batch_size:
            raise ValueError('Episode count exceeds batch_size.')
        self.restore_rest(rest=rest)

    def restore_rest(self, rest: RestState) -> None:
        """Set every episode's state to the rest voltages.

        :param rest: Rest state with float32 voltages, shape (N,).
        :type rest: RestState
        :raises ValueError: If the rest voltages do not match the network.
        """
        if rest.v.shape != (self.n,) or not np.isfinite(rest.v).all():
            raise ValueError('Rest voltages must be finite with shape (n,).')
        self._reservoir.state[:] = rest.v[:, None]

    def run(
        self, input_idx: NDArray[np.int64], currents: NDArray[np.float32] | None, n_steps: int
    ) -> RunSummary:
        """Advance every episode by ``n_steps`` updates with the scaled encoder current.

        :param input_idx: Encoder-driven neuron indices, shape (I,), int64.
        :type input_idx: NDArray[np.int64]
        :param currents: Encoder current, shape (I,B), float32, or no drive.
        :type currents: Optional[NDArray[np.float32]]
        :param n_steps: Nonnegative number of updates.
        :type n_steps: int
        :returns: All-zero spike counts, because the graded model emits no spikes.
        :rtype: RunSummary
        :raises ValueError: If the step count is negative or the driven neurons change.
        """
        if n_steps < 0:
            raise ValueError('n_steps must be nonnegative.')
        if self._driven is None:
            self._driven = input_idx.copy()
            if self._rate.driven_leak is not None:
                self._reservoir.leak[input_idx] = np.float32(self._rate.driven_leak)
        elif not np.array_equal(self._driven, input_idx):
            raise ValueError('The driven neurons must stay the same for the whole trace.')
        inputs = None if currents is None else (currents * np.float32(self._rate.input_scale))
        for _ in range(n_steps):
            self._reservoir.step(input_idx=input_idx, inputs=inputs)
        zeros = np.zeros(shape=self.batch_size, dtype=np.int64)
        return RunSummary(
            spikes_total=zeros, active_neurons=zeros, readout_spikes=zeros, input_spikes=zeros
        )

    def features(self, kinds: Sequence[str]) -> NDArray[np.float32]:
        """Copy the readout population's graded state, recorded as the ``voltage`` block.

        :param kinds: Exactly ``['voltage']``.
        :type kinds: Sequence[str]
        :returns: Float32 matrix of shape (B,R).
        :rtype: NDArray[np.float32]
        :raises ValueError: If any other feature block is requested.
        """
        if list(kinds) != [VOLTAGE]:
            raise ValueError('The rate backend records only the voltage block.')
        return np.ascontiguousarray(self._reservoir.state[self.readout_idx].T)
