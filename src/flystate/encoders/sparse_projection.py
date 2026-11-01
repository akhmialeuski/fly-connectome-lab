"""Sparse signed RGB projections and Gaussian spatial input channels."""

import hashlib

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from flystate.experiments.config import EncoderConfig
from flystate.hashing import stable_int

RGB_CHANNELS: int = 3
PIXEL_MAX: float = 255.0


def _generator(seed: int, namespace: str) -> np.random.Generator:
    """Construct the specified independent PCG64 stream.

    :param seed: Nonnegative experiment encoder seed.
    :type seed: int
    :param namespace: Stable neuron or channel projection identifier.
    :type namespace: str
    :returns: Explicit PCG64 generator initialized from a child SeedSequence.
    :rtype: np.random.Generator
    """
    return np.random.Generator(
        bit_generator=np.random.PCG64(
            seed=np.random.SeedSequence(entropy=[seed, stable_int(key=namespace)])
        )
    )


def _projection(cfg: EncoderConfig, window: int, channel: int) -> csr_matrix:
    """Construct one fixed sparse matrix with distinct selected columns per row.

    :param cfg: Validated projection sparsity and seed settings.
    :type cfg: EncoderConfig
    :param window: Window side length in pixels.
    :type window: int
    :param channel: RGB channel index in [0,2].
    :type channel: int
    :returns: Read-only float32 CSR matrix with shape (P,W*W).
    :rtype: csr_matrix
    """
    generator = _generator(seed=cfg.seed, namespace=f'encoder-projection-{channel}')
    count = cfg.nonzeros_per_row
    indices = np.empty(shape=(cfg.neurons_per_channel, count), dtype=np.int32)
    data = np.empty(shape=indices.shape, dtype=np.float32)
    for row in range(cfg.neurons_per_channel):
        indices[row] = generator.choice(a=window * window, size=count, replace=False)
        data[row] = generator.choice(a=[-1, 1], size=count) / np.sqrt(count)
    indptr = np.arange(cfg.neurons_per_channel + 1, dtype=np.int32) * count
    matrix = csr_matrix(
        arg1=(data.ravel(), indices.ravel(), indptr),
        shape=(cfg.neurons_per_channel, window * window),
        dtype=np.float32,
    )
    for array in (matrix.indices, matrix.indptr, matrix.data):
        array.flags.writeable = False
    return matrix


class SparseProjectionEncoder:
    """Project RGB windows and centers onto disjoint, reproducibly selected neurons."""

    def __init__(self, cfg: EncoderConfig, window: int, candidate_neurons: NDArray) -> None:
        """Freeze the neuron selection and independent sparse channel projections.

        :param cfg: Validated encoder settings.
        :type cfg: EncoderConfig
        :param window: Positive window side length in pixels.
        :type window: int
        :param candidate_neurons: Distinct nonnegative integer neuron indices, shape (K,).
        :type candidate_neurons: NDArray
        :raises ValueError: If geometry, candidate indices, or population capacity are invalid.
        """
        if window < 1 or cfg.nonzeros_per_row > window * window:
            raise ValueError(
                'Window must be positive and contain enough distinct projection pixels.'
            )
        if (
            candidate_neurons.ndim != 1
            or candidate_neurons.dtype.kind not in 'iu'
            or np.any(candidate_neurons < 0)
            or np.any(candidate_neurons > np.iinfo(np.int64).max)
            or len(np.unique(ar=candidate_neurons)) != len(candidate_neurons)
        ):
            raise ValueError(
                'Candidate neurons must be distinct nonnegative int64-compatible indices.'
            )
        self.cfg: EncoderConfig = cfg
        self.window: int = window
        self.position_count: int = cfg.position.neurons_per_axis if cfg.position.enabled else 0
        count = RGB_CHANNELS * cfg.neurons_per_channel + 2 * self.position_count
        if count > len(candidate_neurons):
            raise ValueError(
                f'Encoder needs {count} neurons but only {len(candidate_neurons)} are available.'
            )
        generator = _generator(seed=cfg.seed, namespace='encoder-neurons')
        self.input_idx: NDArray[np.int64] = generator.permutation(
            x=np.sort(a=candidate_neurons).astype(np.int64)
        )[:count]
        self.input_idx.flags.writeable = False
        self.projections: tuple[csr_matrix, ...] = tuple(
            _projection(cfg=cfg, window=window, channel=channel) for channel in range(RGB_CHANNELS)
        )

    def encode(self, observations: NDArray, positions: NDArray) -> NDArray[np.float32]:
        """Return float32 voltage kicks in RGB, horizontal, vertical neuron order.

        :param observations: Nonempty RGB uint8 windows, shape (B,W,W,3).
        :type observations: NDArray
        :param positions: Finite normalized float32 centers in [0,1], shape (B,2).
        :type positions: NDArray
        :returns: Currents bounded by encoder amplitude, shape (M,B), float32.
        :rtype: NDArray[np.float32]
        :raises ValueError: If shapes, dtypes, or coordinate ranges violate the encoder contract.
        """
        if (
            observations.ndim != 4
            or len(observations) < 1
            or observations.shape[1:] != (self.window, self.window, RGB_CHANNELS)
            or observations.dtype != np.uint8
        ):
            raise ValueError(
                'Observations must be nonempty RGB uint8 windows with shape (B,W,W,3).'
            )
        if (
            positions.shape != (len(observations), 2)
            or positions.dtype != np.float32
            or not np.isfinite(positions).all()
            or np.any(positions < 0)
            or np.any(positions > 1)
        ):
            raise ValueError('Positions must be finite float32 centers in [0,1] with shape (B,2).')
        pixels = 2 * observations.astype(np.float32) / PIXEL_MAX - 1
        groups = [
            self.cfg.amplitude
            * np.clip(
                a=matrix @ pixels[..., channel].reshape(len(observations), -1).T, a_min=-1, a_max=1
            )
            for channel, matrix in enumerate(self.projections)
        ]
        if self.position_count:
            centers = (np.arange(self.position_count, dtype=np.float32) + 0.5) / self.position_count
            sigma = 1 / self.position_count
            for axis in range(2):
                distances = positions[:, axis][None, :] - centers[:, None]
                groups.append(
                    self.cfg.amplitude * np.exp(-distances * distances / (2 * sigma * sigma))
                )
        return np.vstack(tup=groups).astype(np.float32)

    def spec(self) -> dict[str, object]:
        """Describe configuration and exact neuron/projection byte identities.

        :returns: JSON-compatible immutable-input specification and SHA-256 digests.
        :rtype: dict[str, object]
        """
        projection_digest = hashlib.sha256()
        for matrix in self.projections:
            for array in (matrix.indices, matrix.indptr, matrix.data):
                projection_digest.update(array.tobytes())
        return {
            'kind': self.cfg.kind,
            'window': self.window,
            'neurons_per_channel': self.cfg.neurons_per_channel,
            'nonzeros_per_row': self.cfg.nonzeros_per_row,
            'position': self.cfg.position.model_dump(mode='json'),
            'amplitude': self.cfg.amplitude,
            'seed': self.cfg.seed,
            'input_idx_sha256': hashlib.sha256(string=self.input_idx.tobytes()).hexdigest(),
            'projection_sha256': projection_digest.hexdigest(),
        }
