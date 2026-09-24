"""Paired, source-verified analysis of the frozen temporal-response pilot."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flystate.diagnostics.artifacts import attempt, verify_attempt_inventory
from flystate.diagnostics.temporal import CASES, TemporalCase
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.hashing import sha256_file, sha256_obj
from flystate.settings import Paths, output_path
from flystate.storage.json import write_json
from flystate.storage.parquet import write_table

NEAR_THRESHOLD_VOLTAGE: float = 0.9
HIGH_RATE_FRACTION: float = 0.5
DRIVEN_CASES: tuple[str, ...] = ('C1', 'C2', 'C3', 'C4', 'C5')
BLANK_FOR: dict[str, str] = {
    'C1': 'C0R',
    'C2': 'C0R',
    'C3': 'C0R',
    'C4': 'C0R',
    'C5': 'C6R',
}


@dataclass(frozen=True)
class CaseEvidence:
    """One verified immutable attempt with numeric arrays and ordered sample windows."""

    case: TemporalCase
    checkpoints: tuple[int, ...]
    rows: list[dict[str, Any]]
    arrays: dict[str, NDArray[Any]]
    report: dict[str, Any]
    source_sha256: str
    input_hashes: dict[str, str]


def _load_case(paths: Paths, source: Path, case_id: str) -> CaseEvidence:
    """Check inventory, declared condition, shapes, dtypes, and numerical finiteness.

    :param paths: Working data home.
    :type paths: Paths
    :param source: Parent directory containing C0-C6 and C0R/C6R attempts.
    :type source: Path
    :param case_id: Expected frozen condition identifier.
    :type case_id: str
    :returns: Verified numeric response arrays and ordered sample-window rows.
    :rtype: CaseEvidence
    :raises ValueError: If provenance, schema, or numeric evidence is incompatible.
    """
    directory = source / case_id
    manifest = verify_attempt_inventory(directory=directory, paths=paths)
    case = CASES[case_id]
    report = json.loads(s=(directory / 'report.json').read_text(encoding='utf-8'))
    rows = json.loads(s=(directory / 'sample-windows.json').read_text(encoding='utf-8'))
    checkpoints = tuple(report['checkpoints'])
    if (
        manifest['status'] != 'completed'
        or manifest['parameters']['case'] != case_id
        or manifest['parameters']['kind'] != 'temporal_response'
        or manifest['parameters']['current_scale'] != case.current_scale
        or manifest['parameters']['noise_enabled'] != case.noise_enabled
        or report['case'] != case_id
        or report['stimulus_steps'] != case.stimulus_steps
        or report['current_scale'] != case.current_scale
        or report['noise_enabled'] != case.noise_enabled
        or report['response_sha256'] != sha256_file(path=directory / 'responses.npz')
        or len(rows) != report['sample_windows']
        or checkpoints[0] != 0
        or checkpoints[-1] != case.stimulus_steps + case.recovery_steps
        or tuple(sorted(set(checkpoints))) != checkpoints
    ):
        raise ValueError(f'Attempt evidence differs from the declared case: {case_id}.')
    with np.load(file=directory / 'responses.npz', allow_pickle=False) as source_arrays:
        arrays = {name: source_arrays[name].copy() for name in source_arrays.files}
    names = {
        *(f'voltage_{name}' for name in report['populations']),
        *(f'spikes_{name}' for name in report['populations']),
        'readout_trace',
        'noise_kicks',
        'total_spikes',
        'active_neurons',
    }
    if set(arrays) != names:
        raise ValueError(f'Incomplete or unexpected numeric response arrays: {case_id}.')
    n = len(rows)
    total_steps = case.stimulus_steps + case.recovery_steps
    for name, width in report['populations'].items():
        voltage = arrays[f'voltage_{name}']
        spikes = arrays[f'spikes_{name}']
        expected_shape = (n, len(checkpoints), width)
        if (
            voltage.shape != expected_shape
            or voltage.dtype != np.float32
            or not np.isfinite(voltage).all()
            or spikes.shape != expected_shape
            or spikes.dtype != np.int32
            or np.any(spikes < 0)
            or np.any(np.diff(spikes, axis=1) < 0)
            or np.any(spikes[:, 0] != 0)
        ):
            raise ValueError(f'Invalid selected-neuron trajectory for {name} in {case_id}.')
    traces = arrays['readout_trace']
    if (
        traces.shape != (n, len(checkpoints), report['populations']['descending_neuron'])
        or traces.dtype != np.float32
        or not np.isfinite(traces).all()
    ):
        raise ValueError(f'Invalid readout trace in {case_id}.')
    for name in ('noise_kicks', 'total_spikes', 'active_neurons'):
        values = arrays[name]
        if values.shape != (n, total_steps) or values.dtype != np.int64 or np.any(values < 0):
            raise ValueError(f'Invalid per-step {name} in {case_id}.')
    if np.any(np.diff(arrays['active_neurons'], axis=1) < 0):
        raise ValueError(f'Active-neuron counts decreased within {case_id}.')
    return CaseEvidence(
        case=case,
        checkpoints=checkpoints,
        rows=rows,
        arrays=arrays,
        report=report,
        source_sha256=sha256_file(path=directory / 'checksums.sha256'),
        input_hashes={
            key: manifest['parameters'][key]
            for key in ('cohort_sha256', 'masks_sha256', 'membership_sha256')
        },
    )


def _verify_controls(records: dict[str, CaseEvidence], source: Path) -> dict[str, Any]:
    """Require exact corrected-blank replay and matched stochastic streams.

    :param records: All nine completed cases keyed by frozen identifier.
    :type records: dict[str, CaseEvidence]
    :param source: Parent attempt directory for effective config comparison.
    :type source: Path
    :returns: Verified control counts and source identity.
    :rtype: dict[str, Any]
    :raises ValueError: If input membership, blank replay, or noise matching fails.
    """
    anchor = records['C0']
    reference_config = json.loads(s=(source / 'C0' / 'config.json').read_text(encoding='utf-8'))
    for name, record in records.items():
        config = json.loads(s=(source / name / 'config.json').read_text(encoding='utf-8'))
        if (
            record.rows != anchor.rows
            or config != reference_config
            or record.report['encoder'] != anchor.report['encoder']
            or record.report['populations'] != anchor.report['populations']
            or record.input_hashes != anchor.input_hashes
        ):
            raise ValueError(f'Case {name} changed the frozen cohort, encoder, or masks.')
    for original, corrected in (('C0', 'C0R'), ('C6', 'C6R')):
        old, new = records[original], records[corrected]
        slots = [new.checkpoints.index(step) for step in old.checkpoints]
        for name, values in old.arrays.items():
            expected = (
                new.arrays[name][:, slots]
                if name.startswith(('voltage_', 'spikes_')) or name == 'readout_trace'
                else new.arrays[name]
            )
            if not np.array_equal(values, expected):
                raise ValueError(f'Corrected blank differs from {original}: {name}.')
    noisy_blank = records['C0R'].arrays['noise_kicks']
    for case_id in ('C1', 'C2', 'C3', 'C4'):
        noise = records[case_id].arrays['noise_kicks']
        if not np.array_equal(noise, noisy_blank[:, : noise.shape[1]]):
            raise ValueError(f'Episode noise differs from paired blank: {case_id}.')
    if np.any(records['C5'].arrays['noise_kicks']) or np.any(records['C6R'].arrays['noise_kicks']):
        raise ValueError('Noise-off conditions contain episode-noise kicks.')
    return {
        'cases_verified': len(records),
        'sample_windows': len(anchor.rows),
        'corrected_blank_overlap_exact': True,
        'paired_noise_exact': True,
        'noise_off_zero': True,
        'source_config_sha256': sha256_obj(obj=reference_config),
    }


def _measurement_rows(records: dict[str, CaseEvidence]) -> list[dict[str, Any]]:
    """Calculate declared time-matched response metrics for every training window.

    :param records: Verified stimulus and corrected-blank arrays (N,C,K).
    :type records: dict[str, CaseEvidence]
    :returns: One descriptive row per case, window, checkpoint, and population.
    :rtype: list[dict[str, Any]]
    """
    measurements: list[dict[str, Any]] = []
    for case_id in DRIVEN_CASES:
        driven = records[case_id]
        blank = records[BLANK_FOR[case_id]]
        for name in driven.report['populations']:
            voltage = driven.arrays[f'voltage_{name}'].astype(np.float64)
            baseline = blank.arrays[f'voltage_{name}'].astype(np.float64)
            spike_counts = driven.arrays[f'spikes_{name}']
            endpoint = driven.checkpoints.index(driven.case.stimulus_steps)
            high_rate_count = int(np.ceil(HIGH_RATE_FRACTION * driven.case.stimulus_steps))
            for slot, step in enumerate(driven.checkpoints):
                blank_slot = blank.checkpoints.index(step)
                difference = voltage[:, slot] - baseline[:, blank_slot]
                for index, item in enumerate(driven.rows):
                    values = voltage[index, slot]
                    delta = difference[index]
                    stimulus_spikes = spike_counts[index, endpoint]
                    measurements.append(
                        {
                            'case': case_id,
                            'blank_case': blank.case.name,
                            'sample_id': item['sample_id'],
                            'label': item['label'],
                            'window': item['window'],
                            'population': name,
                            'global_step': step,
                            'seconds_since_onset': step * driven.report['dt_s'],
                            'steps_since_offset': max(0, step - driven.case.stimulus_steps),
                            'phase': 'stimulus'
                            if step <= driven.case.stimulus_steps
                            else 'recovery',
                            'mean_absolute_delta_voltage': float(np.mean(a=np.abs(delta))),
                            'rms_delta_voltage': float(np.sqrt(np.mean(a=delta * delta))),
                            'mean_delta_voltage': float(np.mean(a=delta)),
                            'mean_voltage': float(np.mean(a=values)),
                            'mean_blank_voltage': float(np.mean(a=baseline[index, blank_slot])),
                            'near_threshold_fraction': float(
                                np.mean(a=(values >= NEAR_THRESHOLD_VOLTAGE) & (values < 1.0))
                            ),
                            'silent_stimulus_fraction': float(np.mean(a=stimulus_spikes == 0)),
                            'high_rate_stimulus_fraction': float(
                                np.mean(a=stimulus_spikes >= high_rate_count)
                            ),
                            'mean_stimulus_spikes_per_neuron': float(np.mean(a=stimulus_spikes)),
                        }
                    )
    return measurements


def _summaries(
    records: dict[str, CaseEvidence], measurements: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Summarize endpoint response without selecting a population by performance.

    :param records: Verified case arrays and sample-window membership.
    :type records: dict[str, CaseEvidence]
    :param measurements: Paired per-unit checkpoint rows.
    :type measurements: list[dict[str, Any]]
    :returns: Descriptive summaries for all fixed cases and populations.
    :rtype: dict[str, dict[str, Any]]
    """
    summaries: dict[str, dict[str, Any]] = {}
    for case_id in DRIVEN_CASES:
        record = records[case_id]
        blank = records[BLANK_FOR[case_id]]
        endpoint = record.case.stimulus_steps
        by_population: dict[str, Any] = {}
        for name in record.report['populations']:
            rows = [
                row
                for row in measurements
                if row['case'] == case_id
                and row['population'] == name
                and row['global_step'] == endpoint
            ]
            rms = np.asarray(a=[row['rms_delta_voltage'] for row in rows], dtype=np.float64)
            response = record.arrays[f'voltage_{name}'][
                :, record.checkpoints.index(endpoint)
            ].astype(np.float64) - blank.arrays[f'voltage_{name}'][
                :, blank.checkpoints.index(endpoint)
            ].astype(np.float64)
            window_variance = {
                str(window): float(np.var(a=response[indices], axis=0, dtype=np.float64).mean())
                for window in sorted({row['window'] for row in record.rows})
                if (indices := [i for i, row in enumerate(record.rows) if row['window'] == window])
            }
            by_population[name] = {
                'median_endpoint_rms_delta_voltage': float(np.median(a=rms)),
                'min_endpoint_rms_delta_voltage': float(np.min(a=rms)),
                'max_endpoint_rms_delta_voltage': float(np.max(a=rms)),
                'mean_silent_stimulus_fraction': float(
                    np.mean(a=[row['silent_stimulus_fraction'] for row in rows])
                ),
                'mean_high_rate_stimulus_fraction': float(
                    np.mean(a=[row['high_rate_stimulus_fraction'] for row in rows])
                ),
                'mean_near_threshold_fraction': float(
                    np.mean(a=[row['near_threshold_fraction'] for row in rows])
                ),
                'across_image_voltage_variance_by_window': window_variance,
            }
        summaries[case_id] = {
            'stimulus_steps': record.case.stimulus_steps,
            'current_scale': record.case.current_scale,
            'noise_enabled': record.case.noise_enabled,
            'populations': by_population,
            'mean_total_spikes_per_step': np.mean(
                a=record.arrays['total_spikes'], axis=0, dtype=np.float64
            ).tolist(),
            'mean_active_neurons_per_step': np.mean(
                a=record.arrays['active_neurons'], axis=0, dtype=np.float64
            ).tolist(),
        }
    return summaries


def analyze_temporal_pilot(
    cfg: ExperimentConfig, paths: Paths, source: Path, output: Path
) -> dict[str, Any]:
    """Analyze all immutable T27 attempts after validating time-matched controls.

    :param cfg: Frozen source experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data and run directories.
    :type paths: Paths
    :param source: Parent of completed case attempts within FLYSTATE_HOME.
    :type source: Path
    :param output: Fresh immutable analysis attempt under FLYSTATE_HOME.
    :type output: Path
    :returns: Control gate, full descriptive summaries, and analysis artifact location.
    :rtype: dict[str, Any]
    :raises ValueError: If any case, source, or paired control is invalid.
    """
    source_dir = output_path(path=source, paths=paths)
    parameters = {
        'kind': 'temporal_response_analysis',
        'hypothesis': 'Measure stimulus delivery and blank recovery in fixed fly populations.',
        'source': str(source_dir.relative_to(paths.home)),
        'near_threshold_voltage': NEAR_THRESHOLD_VOLTAGE,
        'high_rate_fraction': HIGH_RATE_FRACTION,
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        records = {
            case_id: _load_case(paths=paths, source=source_dir, case_id=case_id)
            for case_id in CASES
        }
        controls = _verify_controls(records=records, source=source_dir)
        if config_hash(cfg=cfg) != controls['source_config_sha256']:
            raise ValueError('Analysis configuration differs from source case configuration.')
        measurements = _measurement_rows(records=records)
        summaries = _summaries(records=records, measurements=measurements)
        original = summaries['C1']['populations']
        responsive = [
            name
            for name, values in original.items()
            if values['median_endpoint_rms_delta_voltage'] > 0
        ]
        decision = 'expand_training_only' if responsive else 'inspect_encoder_and_recorder'
        write_table(path=directory / 'per-unit.parquet', rows=measurements)
        report = {
            'schema_version': 1,
            'controls': controls,
            'cases': summaries,
            'responsive_populations': responsive,
            'decision': decision,
            'near_threshold_voltage': NEAR_THRESHOLD_VOLTAGE,
            'high_rate_fraction': HIGH_RATE_FRACTION,
            'rows': len(measurements),
            'source_inventories_sha256': {
                name: record.source_sha256 for name, record in records.items()
            },
            'conclusion': (
                f'All {len(records)} attempt inventories and paired controls passed. '
                f'C1 had a finite nonzero median evoked voltage response in '
                f'{len(responsive)} of {len(original)} observed populations. '
                'This pilot does not measure identity decoding or sequential memory.'
            ),
        }
        write_json(path=directory / 'report.json', value=report)
    return {'status': 'completed', 'output': str(directory), **report}
