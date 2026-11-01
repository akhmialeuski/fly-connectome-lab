"""Verify exact seeded projections, channel order, current bounds, and input guards."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from flystate.brain.runtime import EpisodeBrain
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.experiments.config import BrainConfig, EncoderConfig, PositionConfig, ReadoutConfig
from flystate.hashing import stable_int


class TestSparseProjectionEncoder:
    """Exercise synthetic connectome populations and explicit mathematical reference values."""

    def test_population_determinism(self, synthetic_brain_dir: Path) -> None:
        """Load a population, construct encoders, and verify deterministic selections and currents.

        :param synthetic_brain_dir: Small generated flybrain file pair.
        :type synthetic_brain_dir: Path
        """
        brain = EpisodeBrain(
            brain_dir=synthetic_brain_dir,
            brain_cfg=BrainConfig(),
            readout_cfg=ReadoutConfig(),
            batch_size=1,
            threads=1,
        )
        candidates = brain.cells(superclasses=['visual_projection'])
        cfg = EncoderConfig(neurons_per_channel=64, position=PositionConfig(neurons_per_axis=8))
        first = SparseProjectionEncoder(cfg=cfg, window=32, candidate_neurons=candidates)
        second = SparseProjectionEncoder(cfg=cfg, window=32, candidate_neurons=candidates[::-1])
        assert first.spec() == second.spec()
        assert len(np.unique(ar=first.input_idx)) == 208
        assert np.isin(element=first.input_idx, test_elements=candidates).all()
        expected_rng = np.random.Generator(
            bit_generator=np.random.PCG64(
                seed=np.random.SeedSequence(entropy=[cfg.seed, stable_int(key='encoder-neurons')])
            )
        )
        np.testing.assert_array_equal(
            actual=first.input_idx, desired=expected_rng.permutation(x=np.sort(a=candidates))[:208]
        )
        for matrix in first.projections:
            assert matrix.dtype == np.float32
            np.testing.assert_array_equal(
                actual=np.diff(a=matrix.indptr), desired=np.full(shape=64, fill_value=8)
            )
            np.testing.assert_allclose(actual=np.abs(matrix.data), desired=1 / np.sqrt(8))
            for row in range(64):
                assert (
                    len(np.unique(ar=matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]]))
                    == 8
                )
        observations = np.random.default_rng(seed=2).integers(
            low=0, high=256, size=(3, 32, 32, 3), dtype=np.uint8
        )
        positions = np.asarray(a=[[0.5, 0.5], [0.1, 0.9], [0, 1]], dtype=np.float32)
        actual = first.encode(observations=observations, positions=positions)
        assert actual.shape == (208, 3) and actual.dtype == np.float32
        assert np.max(a=np.abs(actual)) <= cfg.amplitude
        np.testing.assert_array_equal(
            actual=actual, desired=second.encode(observations=observations, positions=positions)
        )
        assert np.argmax(a=actual[192:200, 0]) in (3, 4)
        assert not first.input_idx.flags.writeable
        third = SparseProjectionEncoder(
            cfg=cfg.model_copy(update={'seed': 1}), window=32, candidate_neurons=candidates
        )
        assert not np.array_equal(a1=first.input_idx, a2=third.input_idx)
        assert (
            first.spec()['input_idx_sha256']
            == hashlib.sha256(string=first.input_idx.tobytes()).hexdigest()
        )
        projection_bytes = b''.join(
            array.tobytes()
            for matrix in first.projections
            for array in (matrix.indices, matrix.indptr, matrix.data)
        )
        assert (
            first.spec()['projection_sha256'] == hashlib.sha256(string=projection_bytes).hexdigest()
        )

    def test_exact_projection_and_positions(self) -> None:
        """Recreate random draws, then compare RGB and spatial currents to their equations."""
        cfg = EncoderConfig(
            neurons_per_channel=4,
            nonzeros_per_row=3,
            position=PositionConfig(neurons_per_axis=4),
            amplitude=0.2,
        )
        encoder = SparseProjectionEncoder(cfg=cfg, window=4, candidate_neurons=np.arange(40))
        generator = np.random.Generator(
            bit_generator=np.random.PCG64(
                seed=np.random.SeedSequence(entropy=[0, stable_int(key='encoder-projection-0')])
            )
        )
        for row in range(4):
            columns = generator.choice(a=16, size=3, replace=False)
            values = generator.choice(a=[-1, 1], size=3) / np.sqrt(3)
            np.testing.assert_array_equal(
                actual=encoder.projections[0].indices[row * 3 : (row + 1) * 3], desired=columns
            )
            np.testing.assert_allclose(
                actual=encoder.projections[0].data[row * 3 : (row + 1) * 3], desired=values
            )
        observations = np.arange(96, dtype=np.uint8).reshape(2, 4, 4, 3)
        positions = np.asarray(a=[[0.2, 0.8], [0.5, 0.5]], dtype=np.float32)
        actual = encoder.encode(observations=observations, positions=positions)
        for channel, matrix in enumerate(encoder.projections):
            pixels = 2 * observations[..., channel].astype(np.float32).reshape(2, 16) / 255 - 1
            expected = 0.2 * np.clip(a=matrix.toarray() @ pixels.T, a_min=-1, a_max=1)
            np.testing.assert_allclose(
                actual=actual[channel * 4 : (channel + 1) * 4], desired=expected, atol=1e-7
            )
        for axis in range(2):
            centers = (np.arange(4) + 0.5) / 4
            expected = 0.2 * np.exp(
                -((positions[:, axis][None, :] - centers[:, None]) ** 2) / (2 * 0.25**2)
            )
            np.testing.assert_allclose(
                actual=actual[12 + axis * 4 : 16 + axis * 4], desired=expected, atol=1e-7
            )

    def test_disabled_position_and_opposites(self) -> None:
        """Verify opposite black/white currents with one nonzero and no position channels."""
        cfg = EncoderConfig(
            neurons_per_channel=2, nonzeros_per_row=1, position=PositionConfig(enabled=False)
        )
        encoder = SparseProjectionEncoder(cfg=cfg, window=2, candidate_neurons=np.arange(6))
        observations = np.stack(
            arrays=[
                np.zeros(shape=(2, 2, 3), dtype=np.uint8),
                np.full(shape=(2, 2, 3), fill_value=255, dtype=np.uint8),
            ]
        )
        actual = encoder.encode(
            observations=observations, positions=np.zeros(shape=(2, 2), dtype=np.float32)
        )
        assert actual.shape == (6, 2)
        np.testing.assert_array_equal(actual=actual[:, 0], desired=-actual[:, 1])

    @pytest.mark.parametrize(
        'candidates',
        [
            np.arange(5),
            np.asarray([1, 1, 2, 3, 4, 5]),
            np.arange(6, dtype=float),
            np.asarray([-1, 0, 1, 2, 3, 4]),
            np.ones(shape=(2, 3), dtype=int),
            np.asarray([2**64 - 1], dtype=np.uint64),
        ],
    )
    def test_invalid_candidates(self, candidates: NDArray) -> None:
        """Reject insufficient, duplicate, negative, noninteger, and nonvector neuron populations.

        :param candidates: Invalid candidate neuron array.
        :type candidates: NDArray
        """
        with pytest.raises(expected_exception=ValueError):
            SparseProjectionEncoder(
                cfg=EncoderConfig(
                    neurons_per_channel=2,
                    nonzeros_per_row=1,
                    position=PositionConfig(enabled=False),
                ),
                window=2,
                candidate_neurons=candidates,
            )

    @pytest.mark.parametrize('window', [0, 2])
    def test_invalid_geometry(self, window: int) -> None:
        """Reject empty windows or projection rows requesting too many distinct pixels.

        :param window: Invalid side length for eight nonzeros.
        :type window: int
        """
        with pytest.raises(expected_exception=ValueError, match='Window'):
            SparseProjectionEncoder(
                cfg=EncoderConfig(), window=window, candidate_neurons=np.arange(4000)
            )

    @pytest.mark.parametrize(
        'case',
        ['shape', 'empty', 'dtype', 'position_shape', 'position_dtype', 'nan', 'low', 'high'],
    )
    def test_invalid_encoding(self, case: str) -> None:
        """Reject malformed windows and centers before sparse multiplication.

        :param case: Invalid observation or position property.
        :type case: str
        """
        encoder = SparseProjectionEncoder(
            cfg=EncoderConfig(neurons_per_channel=2), window=4, candidate_neurons=np.arange(40)
        )
        observations: NDArray = np.zeros(shape=(1, 4, 4, 3), dtype=np.uint8)
        positions: NDArray = np.zeros(shape=(1, 2), dtype=np.float32)
        if case == 'shape':
            observations = observations[0]
        elif case == 'empty':
            observations = observations[:0]
        elif case == 'dtype':
            observations = observations.astype(np.float32)
        elif case == 'position_shape':
            positions = positions[0]
        elif case == 'position_dtype':
            positions = positions.astype(np.float64)
        else:
            positions[0, 0] = {'nan': np.nan, 'low': -1, 'high': 2}[case]
        with pytest.raises(expected_exception=ValueError):
            encoder.encode(observations=observations, positions=positions)
