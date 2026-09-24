"""Single frozen confirmation of the graded MaleCNS memory model on untouched identities (T35)."""

from pathlib import Path
from time import perf_counter
from typing import Any

import numba
import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

from flystate.brain.benchmark import resolve_threads
from flystate.brain.rate import RateReservoir
from flystate.datasets.preprocess import prepare_dataset
from flystate.diagnostics.artifacts import attempt, verify_attempt_inventory
from flystate.diagnostics.drive_sweep import (
    PARAMETERS,
    POPULATION_SIZES,
    REPORT_FILE,
    RESPONSES_FILE,
    population_indices,
)
from flystate.diagnostics.input_access import _features, _read_json
from flystate.diagnostics.rate_access import BRAIN_METADATA, ENCODED_CURRENT, simulate_states
from flystate.diagnostics.temporal import _write_arrays
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.evaluation.stats import bootstrap_mean_interval, mcnemar, wilson_interval
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.hashing import sha256_file
from flystate.readouts.fitting import fit_classifier
from flystate.settings import Paths, output_path
from flystate.storage.json import write_json
from flystate.storage.parquet import write_table

FINAL_PREFIX: str = 'final_'
TRAIN_SPLIT: str = 'train'
BOOTSTRAP_SAMPLES: int = 10000
BOOTSTRAP_NAMESPACE: str = 't35-identity-cluster-bootstrap'
MAX_ITERATIONS: int = 50000
TOLERANCE: float = 1e-6
CHANCE_ALPHA: float = 0.001
INPUT_REFERENCE: str = 'input'
ISSUE: int = 67
KIND: str = 'kind'
ISSUE_KEY: str = 'issue'
SAMPLE_IDS: str = 'sample_ids'
SCORES: str = 'scores'
COMPARISONS: str = 'comparisons'
HELD_OUT_CORRECT: str = 'held_out_correct'


def record_confirmation(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    gain: float,
    leak: float,
    driven_leak: float | None,
    input_scale: float,
    steps_per_window: int,
    reset_each_window: bool,
    shuffle_seed: int | None,
) -> dict[str, Any]:
    """Record the final graded state of every population for every photograph of a cohort.

    :param cfg: Confirmation experiment configuration (its dataset defines the cohort).
    :type cfg: ExperimentConfig
    :param paths: Working data home.
    :type paths: Paths
    :param output: New immutable attempt directory.
    :type output: Path
    :param gain: Multiplier of the row-normalized effective weights.
    :type gain: float
    :param leak: Update fraction in (0, 1] of every neuron.
    :type leak: float
    :param driven_leak: Separate update fraction of the driven neurons, or the common leak.
    :type driven_leak: Optional[float]
    :param input_scale: Multiplier of the encoded current, dimensionless.
    :type input_scale: float
    :param steps_per_window: Synaptic updates per window.
    :type steps_per_window: int
    :param reset_each_window: Whether the state is zeroed before every window.
    :type reset_each_window: bool
    :param shuffle_seed: Degree-preserving shuffle seed, or the unchanged fly graph.
    :type shuffle_seed: Optional[int]
    :returns: Recording summary.
    :rtype: dict[str, Any]
    :raises ValueError: If a recorded state is not finite.
    """
    threads = resolve_threads(paths=paths, configured=cfg.brain.threads)
    parameters = {
        KIND: 'confirmation_record',
        ISSUE_KEY: ISSUE,
        'config_sha256': config_hash(cfg=cfg),
        'gain': gain,
        'leak': leak,
        'driven_leak': leak if driven_leak is None else driven_leak,
        'input_scale': input_scale,
        'steps_per_window': steps_per_window,
        'reset_each_window': reset_each_window,
        'graph': 'fly' if shuffle_seed is None else 'degree_preserving_shuffle',
        'shuffle_seed': shuffle_seed,
        'numba_threads': threads,
        'trainable_fly_parameters': [],
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        prepared = prepare_dataset(cfg=cfg, paths=paths)
        rows = list(range(len(prepared.samples)))
        features, provenance = _features(cfg=cfg, paths=paths, prepared=prepared, rows=rows)
        currents = features[ENCODED_CURRENT].reshape(len(rows), cfg.episodes.steps, -1)
        with np.load(file=paths.brain / BRAIN_METADATA, allow_pickle=False) as meta:
            candidates = np.flatnonzero(meta['superclass'] == cfg.encoder.target_population)
        encoder = SparseProjectionEncoder(
            cfg=cfg.encoder, window=cfg.episodes.window, candidate_neurons=candidates
        )
        numba.set_num_threads(threads)
        reservoir = RateReservoir(
            brain_dir=paths.brain,
            gain=gain,
            leak=leak,
            batch_size=len(rows),
            leak_overrides=None if driven_leak is None else {driven_leak: encoder.input_idx},
            shuffle_seed=shuffle_seed,
        )
        populations = {
            name: indices
            for name, indices in population_indices(
                brain_file=paths.brain / BRAIN_METADATA, input_idx=encoder.input_idx
            ).items()
            if indices.size
        }
        started = perf_counter()
        with threadpool_limits(limits=1, user_api='blas'):
            states = simulate_states(
                reservoir=reservoir,
                input_idx=encoder.input_idx,
                inputs=(currents * np.float32(input_scale)).astype(np.float32),
                populations=populations,
                steps_per_window=steps_per_window,
                reset_each_window=reset_each_window,
            )
        elapsed = perf_counter() - started
        finals = {
            f'{FINAL_PREFIX}{key.split("_", 1)[1]}': value[:, -1] for key, value in states.items()
        }
        if not all(np.isfinite(value).all() for value in finals.values()):
            raise ValueError('Recorded final states contain nonfinite values.')
        _write_arrays(path=directory / RESPONSES_FILE, arrays=finals)
        summary = {
            PARAMETERS: parameters,
            'simulation_seconds': elapsed,
            'episodes': len(rows),
            'edges': reservoir.edges,
            'dataset_fingerprint': prepared.fingerprint,
            'encoder_input_sha256': provenance['feature_sha256'][ENCODED_CURRENT],
            SAMPLE_IDS: [sample.sample_id for sample in prepared.samples],
            POPULATION_SIZES: {name: len(indices) for name, indices in populations.items()},
            'responses_sha256': sha256_file(path=directory / RESPONSES_FILE),
        }
        write_json(path=directory / REPORT_FILE, value=summary)
    return summary


def _score(
    x: NDArray[np.float32],
    labels: NDArray[np.int64],
    train: NDArray[np.bool_],
    cfg: ExperimentConfig,
) -> tuple[NDArray[np.bool_], NDArray[np.int64], dict[str, Any]]:
    """Fit the project's standard readout on training photographs and score the held-out ones.

    :param x: Features of every photograph, shape (N,F), float32.
    :type x: NDArray[np.float32]
    :param labels: Identity labels, shape (N,), int64.
    :type labels: NDArray[np.int64]
    :param train: Training-photograph mask, shape (N,), bool.
    :type train: NDArray[np.bool_]
    :param cfg: Configuration providing PCA size, C grid, folds and seed.
    :type cfg: ExperimentConfig
    :returns: Held-out correctness (M,), predictions (M,), and a metrics summary.
    :rtype: tuple[NDArray[np.bool_], NDArray[np.int64], dict[str, Any]]
    """
    model, cv_scores = fit_classifier(
        x_train=x[train],
        y_train=labels[train],
        pca_components=cfg.readout.pca_components,
        c_grid=cfg.readout.c_grid,
        cv_folds=cfg.readout.cv_folds,
        seed=cfg.seed,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
    )
    predicted = model.predict(X=x[~train].astype(np.float64)).astype(np.int64)
    correct = predicted == labels[~train]
    successes = int(correct.sum())
    chance = 1 / len(np.unique(labels))
    low, high = wilson_interval(successes=successes, n=len(correct))
    summary = {
        HELD_OUT_CORRECT: successes,
        'held_out': len(correct),
        'accuracy': successes / len(correct),
        'wilson_95': [low, high],
        'binomial_p_above_chance': float(
            binomtest(k=successes, n=len(correct), p=chance, alternative='greater').pvalue
        ),
        'chance': chance,
        'C': float(model.named_steps['classifier'].C),
        'cv_accuracy_by_C': cv_scores,
    }
    return correct, predicted, summary


def evaluate_confirmation(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    recordings: dict[str, Path],
    populations: list[str],
    comparisons: list[tuple[str, str]],
) -> dict[str, Any]:
    """Train on each identity's training photographs and score every untouched held-out photograph.

    Cases are ``<recording>/<population>`` final states plus input references
    ``input/encoded_current_all``, ``input/encoded_current_last`` and ``input/pixels_all``.
    Comparisons are paired on the held-out photographs: identity-cluster bootstrap of the
    accuracy difference (10,000 draws over identities) and McNemar's test.

    :param cfg: Confirmation experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home.
    :type paths: Paths
    :param output: New immutable attempt directory.
    :type output: Path
    :param recordings: Recording name to completed ``confirm-record`` attempt.
    :type recordings: dict[str, Path]
    :param populations: Populations whose final state is scored.
    :type populations: list[str]
    :param comparisons: Pairs of case names (a, b); the difference is a minus b.
    :type comparisons: list[tuple[str, str]]
    :returns: Per-case held-out scores and paired comparisons.
    :rtype: dict[str, Any]
    :raises ValueError: If a recording is incomplete or belongs to another cohort.
    """
    parameters = {
        KIND: 'confirmation_evaluate',
        ISSUE_KEY: ISSUE,
        'recordings': {name: str(path) for name, path in recordings.items()},
        'populations': populations,
        COMPARISONS: [list(pair) for pair in comparisons],
        'readout': 'flystate.readouts.fitting.fit_classifier (CV accuracy selects C, then refit)',
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        prepared = prepare_dataset(cfg=cfg, paths=paths)
        ids = [sample.sample_id for sample in prepared.samples]
        labels = np.asarray([sample.label for sample in prepared.samples], dtype=np.int64)
        train = np.array(
            [sample.split == TRAIN_SPLIT for sample in prepared.samples], dtype=np.bool_
        )
        rows = list(range(len(ids)))
        references, _ = _features(cfg=cfg, paths=paths, prepared=prepared, rows=rows)
        width = references[ENCODED_CURRENT].shape[1] // cfg.episodes.steps
        cases: dict[str, NDArray[np.float32]] = {
            f'{INPUT_REFERENCE}/encoded_current_all': references[ENCODED_CURRENT],
            f'{INPUT_REFERENCE}/encoded_current_last': references[ENCODED_CURRENT][:, -width:],
            f'{INPUT_REFERENCE}/pixels_all': references['pixels'],
        }
        for name, recording in recordings.items():
            if verify_attempt_inventory(directory=recording, paths=paths)['status'] != 'completed':
                raise ValueError(f'The recording is not completed: {recording}.')
            source = output_path(path=recording, paths=paths)
            report = _read_json(path=source / REPORT_FILE)
            if report[SAMPLE_IDS] != ids:
                raise ValueError(f'The recording belongs to another cohort: {recording}.')
            with np.load(file=source / RESPONSES_FILE, allow_pickle=False) as data:
                for population in populations:
                    cases[f'{name}/{population}'] = data[f'{FINAL_PREFIX}{population}']
        held_out_ids = [sample_id for sample_id, keep in zip(ids, ~train, strict=True) if keep]
        held_out_labels = labels[~train]
        correctness: dict[str, NDArray[np.bool_]] = {}
        scores: dict[str, dict[str, Any]] = {}
        prediction_rows: list[dict[str, Any]] = []
        for case, x in cases.items():
            correct, predicted, summary = _score(x=x, labels=labels, train=train, cfg=cfg)
            correctness[case] = correct
            scores[case] = summary
            prediction_rows.extend(
                {'case': case, 'sample_id': sample_id, 'label': int(label), 'predicted': int(value)}
                for sample_id, label, value in zip(
                    held_out_ids, held_out_labels, predicted, strict=True
                )
            )
        paired: list[dict[str, Any]] = []
        identities = np.unique(held_out_labels)
        for first, second in comparisons:
            difference = correctness[first].astype(np.float64) - correctness[second].astype(
                np.float64
            )
            per_identity = np.asarray(
                [difference[held_out_labels == identity].mean() for identity in identities]
            )
            low, high = bootstrap_mean_interval(
                values=per_identity,
                samples=BOOTSTRAP_SAMPLES,
                seed=cfg.seed,
                namespace=BOOTSTRAP_NAMESPACE,
            )
            paired.append(
                {
                    'a': first,
                    'b': second,
                    'a_minus_b_pp': float(difference.mean() * 100),
                    'identity_cluster_95_pp': [low * 100, high * 100],
                    'mcnemar': mcnemar(
                        n01=int(np.sum(~correctness[first] & correctness[second])),
                        n10=int(np.sum(correctness[first] & ~correctness[second])),
                    ),
                }
            )
        write_table(path=directory / 'predictions.parquet', rows=prediction_rows)
        result = {
            PARAMETERS: parameters,
            'train_photographs': int(train.sum()),
            'held_out_photographs': int((~train).sum()),
            'identities': len(identities),
            SCORES: scores,
            COMPARISONS: paired,
            'chance_alpha': CHANCE_ALPHA,
        }
        write_json(path=directory / REPORT_FILE, value=result)
    return {
        'output': str(directory),
        SCORES: {case: value[HELD_OUT_CORRECT] for case, value in scores.items()},
    }
