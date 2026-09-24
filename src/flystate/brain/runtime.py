"""CPU connectome episodes with independent noise streams and explicit rest states."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np
from flybrain import FlyBrain
from numpy.typing import NDArray

from flystate.brain.files import verify_brain_files
from flystate.experiments.config import BrainConfig, ReadoutConfig
from flystate.hashing import stable_int


@dataclass(frozen=True)
class RestState:
    """One warmed fly: float32 voltages (N,) and last int64 spike indices (K,)."""

    v: NDArray[np.float32]
    fired: NDArray[np.int64]
    warmup_steps: int
    seed: int


@dataclass(frozen=True)
class RunSummary:
    """Per-fly int64 counts (B,) over exactly one observation interval."""

    spikes_total: NDArray[np.int64]
    active_neurons: NDArray[np.int64]
    readout_spikes: NDArray[np.int64]
    input_spikes: NDArray[np.int64]


@dataclass(frozen=True)
class TemporalResponse:
    """Checkpoint states for one batch; voltages are float32 and counts are int32/int64."""

    checkpoints: tuple[int, ...]
    voltages: dict[str, NDArray[np.float32]]
    spike_counts: dict[str, NDArray[np.int32]]
    readout_traces: NDArray[np.float32]
    noise_kicks: NDArray[np.int64]
    total_spikes: NDArray[np.int64]
    active_neurons: NDArray[np.int64]


class EpisodeBrain:
    """Use flybrain propagation with deterministic, episode-specific CPU noise."""

    def __init__(
        self,
        brain_dir: Path,
        brain_cfg: BrainConfig,
        readout_cfg: ReadoutConfig,
        batch_size: int,
        threads: int,
    ) -> None:
        """Construct one explicitly configured batch without implicit downloads.

        :param brain_dir: Existing brain metadata and weight files.
        :type brain_dir: Path
        :param brain_cfg: Timing, warmup and noise parameters.
        :type brain_cfg: BrainConfig
        :param readout_cfg: Readout population and trace decay parameters.
        :type readout_cfg: ReadoutConfig
        :param batch_size: Positive number of independent flies.
        :type batch_size: int
        :param threads: Fixed Numba thread count for reproducible accumulation.
        :type threads: int
        :raises ValueError: If the batch, threads, or readout population are invalid.
        :raises FileNotFoundError: If either brain file is absent.
        """
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')
        verification = verify_brain_files(brain_dir=brain_dir, check_hash=False)
        if not all(item['exists'] for item in verification.values()):
            raise FileNotFoundError(
                'Brain files are missing; run flystate brain download explicitly.'
            )
        numba.set_num_threads(n=threads)
        self._fb = FlyBrain(
            data=brain_dir,
            seed=0,
            device='cpu',
            batch=batch_size,
            dt=brain_cfg.dt_s,
            sensory_input=brain_cfg.sensory_input,
            refractory=0.0,
        )
        self.n: int = self._fb.n
        self.batch_size: int = batch_size
        self.threads: int = threads
        self._cfg: BrainConfig = brain_cfg
        self.readout_idx: NDArray[np.int64] = self.cells(superclasses=[readout_cfg.population])
        if not self.readout_idx.size:
            raise ValueError(f'Empty readout population: {readout_cfg.population}.')
        self._trace_decay = np.float32(np.exp(-brain_cfg.dt_s / readout_cfg.trace_tau_s))
        self._trace: NDArray[np.float32] = np.zeros(
            shape=(len(self.readout_idx), batch_size), dtype=np.float32
        )
        self._readout_lookup: NDArray[np.int64] = np.full(
            shape=self.n, fill_value=-1, dtype=np.int64
        )
        self._readout_lookup[self.readout_idx] = np.arange(len(self.readout_idx), dtype=np.int64)
        self._generators: list[np.random.Generator] = []
        self.last_noise_kicks: NDArray[np.int64] = np.zeros(shape=batch_size, dtype=np.int64)

    def cells(self, superclasses: Sequence[str]) -> NDArray[np.int64]:
        """Select sorted neuron indices by upstream superclass or cell type.

        :param superclasses: Population names passed to flybrain.cells.
        :type superclasses: Sequence[str]
        :returns: Sorted int64 neuron indices, shape (K,).
        :rtype: NDArray[np.int64]
        """
        return np.asarray(a=self._fb.cells(types=list(superclasses)), dtype=np.int64)

    def compute_rest_state(self, seed: int) -> RestState:
        """Warm all flies with identical independent generators and retain fly zero.

        :param seed: Nonnegative experiment seed.
        :type seed: int
        :returns: Read-only float32 voltages and int64 spikes of one warmed fly.
        :rtype: RestState
        """
        self._fb.v.fill(0)
        self._fb.fired = np.empty(shape=0, dtype=np.int64)
        self._fb.steps = 0
        self._trace.fill(0)
        self._generators = [
            np.random.default_rng(
                seed=np.random.SeedSequence(entropy=[seed, stable_int(key='warmup')])
            )
            for _ in range(self.batch_size)
        ]
        self.run(
            input_idx=np.empty(shape=0, dtype=np.int64),
            currents=None,
            n_steps=self._cfg.warmup_steps,
        )
        voltages = self._fb.v[:, 0].copy()
        fired = (self._fb.fired[self._fb.fired % self.batch_size == 0] // self.batch_size).copy()
        voltages.flags.writeable = False
        fired.flags.writeable = False
        return RestState(v=voltages, fired=fired, warmup_steps=self._cfg.warmup_steps, seed=seed)

    def begin(self, rest: RestState, sample_ids: Sequence[str], seed: int) -> None:
        """Restore a common rest state and seed every fly by its episode identity.

        :param rest: Compatible warmed single-fly state.
        :type rest: RestState
        :param sample_ids: At most B episode identifiers; remaining flies are padding.
        :type sample_ids: Sequence[str]
        :param seed: Nonnegative experiment seed.
        :type seed: int
        :raises ValueError: If the episode count exceeds batch capacity.
        """
        if len(sample_ids) > self.batch_size:
            raise ValueError('Episode count exceeds batch_size.')
        self.restore_rest(rest=rest)
        identifiers = [
            *sample_ids,
            *(f'__pad__{i}' for i in range(len(sample_ids), self.batch_size)),
        ]
        self._generators = [
            np.random.default_rng(
                seed=np.random.SeedSequence(entropy=[seed, stable_int(key=identifier)])
            )
            for identifier in identifiers
        ]

    def restore_rest(self, rest: RestState) -> None:
        """Restore voltages, prior spikes and zero traces without rewinding noise.

        :param rest: Single-fly float32 voltages (N,) and int64 spikes (K,).
        :type rest: RestState
        :raises ValueError: If the state arrays are incompatible or nonfinite.
        """
        if rest.v.shape != (self.n,) or rest.v.dtype != np.float32 or not np.isfinite(rest.v).all():
            raise ValueError('Rest voltages must be finite float32 with shape (n,).')
        if (
            rest.fired.ndim != 1
            or rest.fired.dtype != np.int64
            or (
                np.any(rest.fired < 0)
                or np.any(rest.fired >= self.n)
                or np.unique(ar=rest.fired).size != rest.fired.size
            )
        ):
            raise ValueError('Rest spikes must be distinct int64 neuron indices in range.')
        self._fb.v[:] = rest.v[:, None]
        self._fb.fired = (
            rest.fired[:, None] * self.batch_size + np.arange(self.batch_size)[None, :]
        ).ravel()
        self._fb.steps = rest.warmup_steps
        self._trace.fill(0)
        self.last_noise_kicks.fill(0)

    def _step(self, input_idx: NDArray[np.int64], currents: NDArray[np.float32] | None) -> None:
        """Apply one upstream-ordered LIF step and update per-fly readout traces.

        :param input_idx: Distinct input neuron indices, shape (I,).
        :type input_idx: NDArray[np.int64]
        :param currents: Float32 voltage kicks (I,B), or no drive.
        :type currents: Optional[NDArray[np.float32]]
        """
        current = self._fb.synaptic_input(fired=self._fb.fired) * self._fb.gain
        voltage = self._fb.v
        voltage *= self._fb.decay
        voltage += current + self._fb.tonic
        self.last_noise_kicks.fill(0)
        if self._cfg.noise.enabled:
            probability = self._cfg.noise.rate_hz * self._cfg.dt_s
            for column, generator in enumerate(self._generators):
                count = generator.binomial(n=self.n, p=probability)
                indices = generator.choice(a=self.n, size=count, replace=False)
                voltage[indices, column] += np.float32(self._cfg.noise.amplitude)
                self.last_noise_kicks[column] = count
        if currents is not None:
            voltage[input_idx, :] += currents
        fired = np.flatnonzero(a=voltage >= 1.0)
        voltage.ravel()[fired] = 0.0
        self._fb.fired = fired
        self._fb.steps += 1
        self._trace *= self._trace_decay
        rows, columns = np.divmod(fired, self.batch_size)
        readout_rows = self._readout_lookup[rows]
        selected = readout_rows >= 0
        self._trace[readout_rows[selected], columns[selected]] += 1.0

    def run(
        self, input_idx: NDArray[np.int64], currents: NDArray[np.float32] | None, n_steps: int
    ) -> RunSummary:
        """Advance one observation with constant input and collect per-fly counts.

        :param input_idx: Distinct int64 input indices, shape (I,).
        :type input_idx: NDArray[np.int64]
        :param currents: Finite float32 voltage kicks, shape (I,B), or no drive.
        :type currents: Optional[NDArray[np.float32]]
        :param n_steps: Nonnegative number of simulation steps.
        :type n_steps: int
        :returns: Spike and active-neuron counts over these steps only.
        :rtype: RunSummary
        :raises ValueError: If steps, input indices, or current arrays are invalid.
        :raises RuntimeError: If no rest computation or episode initialization has occurred.
        """
        if not self._generators:
            raise RuntimeError('Call compute_rest_state or begin before running the brain.')
        if n_steps < 0:
            raise ValueError('n_steps must be nonnegative.')
        if (
            input_idx.ndim != 1
            or input_idx.dtype != np.int64
            or (
                np.any(input_idx < 0)
                or np.any(input_idx >= self.n)
                or np.unique(ar=input_idx).size != input_idx.size
            )
        ):
            raise ValueError('Input indices must be distinct int64 neuron indices in range.')
        if currents is not None and (
            currents.shape != (len(input_idx), self.batch_size)
            or currents.dtype != np.float32
            or not np.isfinite(currents).all()
        ):
            raise ValueError(
                'Currents must be finite float32 with shape (len(input_idx), batch_size).'
            )
        numba.set_num_threads(n=self.threads)
        total = np.zeros(shape=self.batch_size, dtype=np.int64)
        readout = np.zeros_like(a=total)
        input_spikes = np.zeros_like(a=total)
        active = np.zeros(shape=(self.n, self.batch_size), dtype=np.bool_)
        input_mask = np.zeros(shape=self.n, dtype=np.bool_)
        input_mask[input_idx] = True
        for _ in range(n_steps):
            self._step(input_idx=input_idx, currents=currents)
            fired = self._fb.fired
            active.ravel()[fired] = True
            rows, columns = np.divmod(fired, self.batch_size)
            total += np.bincount(columns, minlength=self.batch_size)
            readout += np.bincount(
                columns[self._readout_lookup[rows] >= 0], minlength=self.batch_size
            )
            input_spikes += np.bincount(columns[input_mask[rows]], minlength=self.batch_size)
        return RunSummary(
            spikes_total=total,
            active_neurons=active.sum(axis=0, dtype=np.int64),
            readout_spikes=readout,
            input_spikes=input_spikes,
        )

    def features(self, kinds: Sequence[str]) -> NDArray[np.float32]:
        """Copy readout feature blocks in the requested order without batch averaging.

        :param kinds: Nonempty unique names from spike_trace and voltage.
        :type kinds: Sequence[str]
        :returns: Float32 matrix of shape (B,R*len(kinds)).
        :rtype: NDArray[np.float32]
        :raises ValueError: If requested feature blocks are empty, repeated, or unknown.
        """
        if not kinds or len(set(kinds)) != len(kinds) or set(kinds) - {'spike_trace', 'voltage'}:
            raise ValueError('Features must be unique names from spike_trace and voltage.')
        blocks = [
            self._trace.T if kind == 'spike_trace' else self._fb.v[self.readout_idx, :].T
            for kind in kinds
        ]
        return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)

    def run_recorded(
        self,
        input_idx: NDArray[np.int64],
        currents: NDArray[np.float32] | None,
        stimulus_steps: int,
        recovery_steps: int,
        populations: dict[str, NDArray[np.int64]],
        checkpoints: tuple[int, ...],
    ) -> TemporalResponse:
        """Record selected neuron states during fixed input and input-free recovery.

        :param input_idx: Distinct int64 encoder neuron indices, shape (I,).
        :type input_idx: NDArray[np.int64]
        :param currents: Constant finite float32 stimulus kicks, shape (I,B), or blank.
        :type currents: Optional[NDArray[np.float32]]
        :param stimulus_steps: Positive number of 20 ms steps with the specified stimulus.
        :type stimulus_steps: int
        :param recovery_steps: Nonnegative number of input-free steps.
        :type recovery_steps: int
        :param populations: Nonempty named sorted unique int64 neuron indices, each shape (K,).
        :type populations: dict[str, NDArray[np.int64]]
        :param checkpoints: Strictly increasing steps including zero, at most total steps.
        :type checkpoints: tuple[int, ...]
        :returns: Voltage and cumulative spike arrays (C,B,K), traces (C,B,R), and
            per-step noise/spike/active counts (S,B).
        :rtype: TemporalResponse
        :raises ValueError: If a checkpoint, population, or timing parameter is invalid.
        :raises RuntimeError: If the episode has not been initialized.
        """
        total_steps = stimulus_steps + recovery_steps
        if stimulus_steps < 1 or recovery_steps < 0:
            raise ValueError('Stimulus steps must be positive and recovery steps nonnegative.')
        if (
            not checkpoints
            or checkpoints[0] != 0
            or checkpoints[-1] > total_steps
            or (tuple(sorted(set(checkpoints))) != checkpoints)
        ):
            raise ValueError('Checkpoints must be distinct, increasing, and include zero.')
        if not populations:
            raise ValueError('At least one population must be selected.')
        lookups: dict[str, NDArray[np.int64]] = {}
        for name, indices in populations.items():
            if (
                not name
                or indices.ndim != 1
                or indices.dtype != np.int64
                or not indices.size
                or np.any(indices < 0)
                or np.any(indices >= self.n)
                or np.any(np.diff(indices) <= 0)
            ):
                raise ValueError('Population indices must be nonempty sorted unique int64.')
            lookup = np.full(shape=self.n, fill_value=-1, dtype=np.int64)
            lookup[indices] = np.arange(len(indices), dtype=np.int64)
            lookups[name] = lookup
        self.run(input_idx=input_idx, currents=currents, n_steps=0)
        observed = set(checkpoints)
        voltage_snapshots = {
            name: np.empty(
                shape=(len(checkpoints), self.batch_size, len(indices)), dtype=np.float32
            )
            for name, indices in populations.items()
        }
        spike_snapshots = {
            name: np.empty(shape=(len(checkpoints), self.batch_size, len(indices)), dtype=np.int32)
            for name, indices in populations.items()
        }
        cumulative = {
            name: np.zeros(shape=(self.batch_size, len(indices)), dtype=np.int32)
            for name, indices in populations.items()
        }
        traces = np.empty(
            shape=(len(checkpoints), self.batch_size, len(self.readout_idx)), dtype=np.float32
        )
        noise = np.zeros(shape=(total_steps, self.batch_size), dtype=np.int64)
        spikes = np.zeros_like(a=noise)
        active_counts = np.zeros_like(a=noise)
        active = np.zeros(shape=(self.n, self.batch_size), dtype=np.bool_)
        active_total = np.zeros(shape=self.batch_size, dtype=np.int64)

        def capture(index: int) -> None:
            """Copy a checkpoint before later simulation steps can mutate it.

            :param index: Position in the checkpoint list.
            :type index: int
            """
            for name, indices in populations.items():
                voltage_snapshots[name][index] = self._fb.v[indices, :].T
                spike_snapshots[name][index] = cumulative[name]
            traces[index] = self._trace.T

        capture(index=0)
        slot = 1
        for step in range(1, total_steps + 1):
            self._step(
                input_idx=input_idx,
                currents=currents if step <= stimulus_steps else None,
            )
            noise[step - 1] = self.last_noise_kicks
            fired = self._fb.fired
            newly_active = fired[~active.ravel()[fired]]
            active.ravel()[fired] = True
            rows, columns = np.divmod(fired, self.batch_size)
            spikes[step - 1] = np.bincount(columns, minlength=self.batch_size)
            active_total += np.bincount(newly_active % self.batch_size, minlength=self.batch_size)
            active_counts[step - 1] = active_total
            for name, lookup in lookups.items():
                local = lookup[rows]
                selected = local >= 0
                np.add.at(cumulative[name], (columns[selected], local[selected]), 1)
            if step in observed:
                capture(index=slot)
                slot += 1
        return TemporalResponse(
            checkpoints=checkpoints,
            voltages=voltage_snapshots,
            spike_counts=spike_snapshots,
            readout_traces=traces,
            noise_kicks=noise,
            total_spikes=spikes,
            active_neurons=active_counts,
        )
