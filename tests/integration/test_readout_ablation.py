"""Offline regression checks for the frozen-trace readout ablation analysis."""

import shutil

import pytest

from flystate.diagnostics.noise import generate_noise_trace, load_noise_trace
from flystate.diagnostics.probes import run_probe
from flystate.diagnostics.readout_ablation import (
    EXPECTED_CASES,
    analyze_readout_ablation,
    replay_exported_predictions,
)
from flystate.experiments.config import ExperimentConfig
from flystate.hashing import sha256_file, sha256_obj
from flystate.settings import get_paths
from flystate.storage.parquet import read_table, write_table
from flystate.traces.builder import build_trace, resolve_trace


class TestReadoutAblation:
    """Verify the full offline screen and independently detect coefficient replay errors."""

    def test_frozen_screen_and_probability_tamper(self, tiny_experiment: ExperimentConfig) -> None:
        """Build synthetic native traces, fit all cases, then reject changed probabilities.

        :param tiny_experiment: Generated images, synthetic connectome and isolated data home.
        :type tiny_experiment: ExperimentConfig
        """
        base = tiny_experiment
        cfg = base.model_copy(
            update={
                'dataset': base.dataset.model_copy(
                    update={
                        'subset': base.dataset.subset.model_copy(update={'images_per_identity': 20})
                    }
                ),
                'episodes': base.episodes.model_copy(update={'steps': 16}),
                'readout': base.readout.model_copy(
                    update={
                        'pca_components': 60,
                        'c_grid': (0.01, 0.1, 1.0, 10.0),
                        'cv_folds': 5,
                    }
                ),
            }
        )
        paths = get_paths()
        build_trace(cfg=cfg, paths=paths)
        prepared, _, _ = resolve_trace(cfg=cfg, paths=paths)
        groups = {
            split: sorted(sample.sample_id for sample in prepared.samples if sample.split == split)
            for split in ('train', 'val')
        }
        membership_sha256 = sha256_obj(obj=groups)
        source = paths.runs / 'source-seed0'
        robustness = paths.runs / 'source-seed1'
        for output, seed in ((source, 0), (robustness, 1)):
            generate_noise_trace(
                cfg=cfg,
                paths=paths,
                output=output,
                membership=groups,
                membership_sha256=membership_sha256,
                episode_seed=seed,
                noise_enabled=True,
            )
        prefix = paths.runs / 'readout-screen'
        reference = paths.runs / 'reference-B0'
        cases = [
            {'id': case_id, 'features': features, 'history': history, 'components': components}
            for case_id, features, history, components in EXPECTED_CASES
        ]
        for case_id, features, history, components in EXPECTED_CASES:
            run_probe(
                cfg=cfg,
                paths=paths,
                output=reference if case_id == 'B0' else prefix / case_id,
                representation='neural',
                history=history,
                features=features,
                components=components,
                label_mode='true',
                max_iterations=50000,
                train_per_class=14,
                subset_seed=0,
                trace_source=source,
                trace_precision='float32',
            )
        schedule = {
            'membership_sha256': membership_sha256,
            'source_trace': str(source),
            'source_trace_inventory_sha256': sha256_file(path=source / 'checksums.sha256'),
            'robustness_trace': str(robustness),
            'robustness_trace_inventory_sha256': sha256_file(path=robustness / 'checksums.sha256'),
            'reference_probe': str(reference),
            'reference_probe_inventory_sha256': sha256_file(path=reference / 'checksums.sha256'),
            'working_prefix': str(prefix),
            'trace_precision': 'float32',
            'train_per_class': 14,
            'subset_seed': 0,
            'bootstrap_seed': 0,
            'bootstrap_resamples': 2000,
            'advancement_minimum_percentage_points': 10,
            'cases': cases,
        }
        result = analyze_readout_ablation(
            cfg=cfg, paths=paths, output=paths.runs / 'analysis', schedule=schedule
        )
        assert set(result['fits']) == {case['id'] for case in cases}
        assert all(fit['status'] == 'completed' for fit in result['fits'].values())
        assert set(result['paired_validation']) == {case['id'] for case in cases[1:]}
        assert all(
            fit['numeric_replay']['prediction_rows'] == 68 for fit in result['fits'].values()
        )
        assert result['gate'] in {'advance_to_seed1', 'do_not_advance'}
        assert result['paired_validation']['B1']['clusters'] == 4
        data = load_noise_trace(
            cfg=cfg,
            paths=paths,
            source=source,
            precision='float32',
            history='last',
            features='voltage',
        )
        tampered = paths.home / 'tampered-probe-copy'
        shutil.copytree(src=prefix / 'B1', dst=tampered)
        predictions = tampered / 'validation-predictions.parquet'
        rows = read_table(path=predictions)
        rows[0]['probabilities'][0] += 0.01
        predictions.unlink()
        write_table(path=predictions, rows=rows)
        with pytest.raises(expected_exception=ValueError, match='do not replay'):
            replay_exported_predictions(
                data=data,
                directory=tampered,
                expected_sha256=result['fits']['B1']['model']['weights_sha256'],
            )
