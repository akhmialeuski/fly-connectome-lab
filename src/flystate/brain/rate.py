"""Deterministic graded (rate) dynamics on flybrain's effective MaleCNS connectome.

The spiking runtime in :mod:`flystate.brain.runtime` follows flybrain's leaky integrate-and-fire
model. This module keeps the same effective graph, including flybrain's removal of synapses onto
sensory neurons, but replaces spikes with a leaky echo-state update, the neuron model used by
connectome reservoirs such as conn2res, wetware and fly-self-driving:

``x[t+1] = (1 - leak) * x[t] + leak * tanh(gain * W @ x[t] + u[t])``

There is no noise and no tonic drive, so ``x = 0`` is the rest state and every result is a
deterministic function of the input.
"""

from pathlib import Path

import numba
import numpy as np
from flybrain import FlyBrain
from numpy.typing import NDArray
from scipy import sparse

from flystate.hashing import stable_int

SHUFFLE_NAMESPACE: str = 'degree-preserving-shuffle'


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
    for row in numba.prange(len(indptr) - 1):
        accumulator = np.zeros(state.shape[1], dtype=np.float32)
        for edge in range(indptr[row], indptr[row + 1]):
            accumulator += weights[edge] * state[indices[edge]]
        out[row] = accumulator


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
    # Canonical order on a copy: the permutation must not depend on the caller's storage order.
    canonical = sparse.csr_matrix(matrix, copy=True)
    canonical.sum_duplicates()
    canonical.sort_indices()
    coo = canonical.tocoo()
    generator = np.random.default_rng(
        seed=np.random.SeedSequence(entropy=[seed, stable_int(key=SHUFFLE_NAMESPACE)])
    )
    targets = generator.permutation(coo.row)
    shuffled = sparse.csr_matrix(
        (coo.data.astype(np.float64), (targets, coo.col)), shape=matrix.shape
    )
    shuffled.sum_duplicates()
    before = np.asarray(abs(canonical).sum(axis=1), dtype=np.float64).ravel()
    after = np.asarray(abs(shuffled).sum(axis=1), dtype=np.float64).ravel()
    scale = np.divide(before, after, out=np.zeros_like(before), where=after > 0)
    shuffled = sparse.diags(scale) @ shuffled
    return sparse.csr_matrix(shuffled, dtype=np.float32)


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
        matrix.sort_indices()
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
