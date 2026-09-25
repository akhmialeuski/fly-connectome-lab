"""Wiring specificity of the graded MaleCNS memory model at matched operating points (T37).

T35 compared MaleCNS with one degree-preserving shuffle at the same gain, but the shuffle's giant
strongly connected component has a spectral radius of 0.257 against 0.737 for MaleCNS. Following
conn2res (Suárez et al., Nat. Commun. 2024), every graph here is run at ``gain = alpha / radius``,
so graphs are compared at the same operating point, and nulls form ensembles over seeds.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numba
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from flystate.brain.benchmark import resolve_threads
from flystate.brain.rate import (
    RateReservoir,
    feedforward_only,
    giant_component_radius,
    random_target_shuffle,
)
from flystate.diagnostics.artifacts import attempt, verify_attempt_inventory
from flystate.diagnostics.confirmation import (
    ISSUE_KEY,
    KIND,
    SCORES,
    cohort_size,
    driven_neurons,
    record_cohort,
)
from flystate.diagnostics.drive_sweep import PARAMETERS, REPORT_FILE
from flystate.diagnostics.input_access import _read_json
from flystate.evaluation.stats import mcnemar, wilson_interval
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.hashing import stable_int
from flystate.settings import Paths, output_path
from flystate.storage.json import write_json
from flystate.storage.parquet import read_table

FLY: str = 'fly'
DEGREE: str = 'degree'
RANDOM_TARGET: str = 'random_target'
FEEDFORWARD: str = 'feedforward'
FAMILIES: tuple[str, ...] = (FLY, DEGREE, RANDOM_TARGET, FEEDFORWARD)
ADVANTAGE: str = 'advantage'
DISADVANTAGE: str = 'disadvantage'
EQUIVALENT: str = 'no practically relevant difference'
INCONCLUSIVE: str = 'inconclusive'
COMPLETED: str = 'completed'
ACCURACY: str = 'accuracy'
CV_ACCURACY: str = 'cv_accuracy'
CONTRASTS: str = 'contrasts'
ENSEMBLE: str = 'ensemble'
NAME: str = 'name'
PHOTOGRAPHS: str = 'photographs'
PREDICTED: str = 'predicted'
ISSUE: int = 74
RADIUS_CACHE: str = 'spectral-radius'
# Changes whenever giant_component_radius changes its solver settings.
RADIUS_METHOD: str = 'arpack-k6-ones-v1'
RADIUS_KEY: str = 'giant_component_radius'
KEEP: tuple[str, ...] = ('central_brain', 'descending')
SEEDED: tuple[str, ...] = (DEGREE, RANDOM_TARGET)
BOOTSTRAP_SAMPLES: int = 10000
BOOTSTRAP_NAMESPACE: str = 't37-two-level-bootstrap'
CONFIDENCE_TAILS: tuple[float, float] = (2.5, 97.5)
PERCENT: float = 100.0
EQUIVALENCE_MARGIN_PP: float = 5.0
CV_SCORES: str = 'cv_accuracy_by_C'
PREDICTIONS_FILE: str = 'predictions.parquet'
CASE: str = 'case'
SAMPLE_ID: str = 'sample_id'
LABEL: str = 'label'


@dataclass(frozen=True)
class Contrast:
    """Paired difference between one case and the mean of an ensemble of cases.

    :param name: Reader-facing contrast name.
    :param first: Case ``<recording>/<population>`` under test.
    :param ensemble: Cases averaged into the reference, one per null seed.
    """

    name: str
    first: str
    ensemble: tuple[str, ...]


def build_reservoir(
    paths: Paths,
    family: str,
    seed: int | None,
    input_idx: NDArray[np.int64],
    leak: float,
    driven_leak: float,
    batch_size: int,
) -> RateReservoir:
    """Build the graded reservoir on MaleCNS or one of its null graphs, at gain 1.

    :param paths: Working data home with the flybrain files.
    :type paths: Paths
    :param family: ``fly``, ``degree`` (degree-preserving shuffle), ``random_target`` or
        ``feedforward`` (only synapses leaving the driven neurons).
    :type family: str
    :param seed: Null-model seed for the seeded families, none otherwise.
    :type seed: Optional[int]
    :param input_idx: Driven neuron indices, shape (I,), int64.
    :type input_idx: NDArray[np.int64]
    :param leak: Update fraction of every non-driven neuron.
    :type leak: float
    :param driven_leak: Update fraction of the driven neurons.
    :type driven_leak: float
    :param batch_size: Photographs simulated together.
    :type batch_size: int
    :returns: Reservoir with gain 1; the caller sets the operating gain.
    :rtype: RateReservoir
    :raises ValueError: If a seed is missing for a seeded family or given for another one.
    """
    if family not in FAMILIES or (seed is None) == (family in SEEDED):
        raise ValueError(f'Unknown family, or a seed given outside the families {SEEDED}.')
    transform = None
    if family == RANDOM_TARGET and seed is not None:
        transform = partial(random_target_shuffle, seed=seed)
    elif family == FEEDFORWARD:
        transform = partial(feedforward_only, sources=input_idx)
    return RateReservoir(
        brain_dir=paths.brain,
        gain=1.0,
        leak=leak,
        batch_size=batch_size,
        leak_overrides={driven_leak: input_idx},
        shuffle_seed=seed if family == DEGREE else None,
        transform=transform,
    )


def _require_completed(directory: Path, paths: Paths) -> None:
    """Verify an attempt's inventory and refuse anything but a completed attempt.

    :param directory: Attempt directory within the data home.
    :type directory: Path
    :param paths: Working data home.
    :type paths: Paths
    :raises ValueError: If the inventory is altered or the attempt did not complete.
    """
    if verify_attempt_inventory(directory=directory, paths=paths)['status'] != COMPLETED:
        raise ValueError(f'The attempt is not completed; move it aside: {directory}.')


def _radius(matrix: sparse.csr_matrix, digest: str, paths: Paths) -> tuple[float, str]:
    """Return the giant-component spectral radius, computed once per exact graph.

    The radius is a deterministic function of the stored CSR arrays, whose SHA-256 is ``digest``.
    It is cached as JSON, which round-trips a float exactly, so every recording of the same graph
    derives exactly the same gain without repeating the eigenvalue solve.

    :param matrix: Simulated weights with postsynaptic rows, shape (N,N).
    :type matrix: sparse.csr_matrix
    :param digest: SHA-256 of the matrix's indptr, indices and data bytes.
    :type digest: str
    :param paths: Working data home whose cache directory stores the radius.
    :type paths: Paths
    :returns: Radius, dimensionless, and whether it was ``computed`` or read from ``cache``.
    :rtype: tuple[float, str]
    """
    source = paths.cache / RADIUS_CACHE / f'{RADIUS_METHOD}-{digest}.json'
    if source.exists():
        return float(_read_json(path=source)[RADIUS_KEY]), 'cache'
    radius = giant_component_radius(matrix=matrix)
    source.parent.mkdir(parents=True, exist_ok=True)
    write_json(path=source, value={RADIUS_KEY: radius, 'graph_sha256': digest})
    return radius, 'computed'


def _cohort_invariant(cfg: ExperimentConfig) -> dict[str, Any]:
    """Return the configuration without its name and identity-selection seed.

    :param cfg: Cohort configuration.
    :type cfg: ExperimentConfig
    :returns: JSON-compatible configuration that must be equal across cohorts.
    :rtype: dict[str, Any]
    """
    data = cfg.model_dump(mode='json')
    data.pop(NAME)
    data['dataset']['subset'].pop('selection_seed')
    return data


def record_wiring(
    cohorts: dict[str, ExperimentConfig],
    paths: Paths,
    output: Path,
    family: str,
    seed: int | None,
    alpha: float | None,
    gain: float | None,
    leak: float,
    driven_leak: float,
    input_scale: float,
    steps_per_window: int,
    reset_each_window: bool,
    threads: int | None,
) -> dict[str, Any]:
    """Build one graph once and record the final states of every cohort on it.

    Each cohort becomes its own immutable attempt ``output/<cohort>``. Completed attempts are
    verified and skipped, so an interrupted run resumes without rebuilding finished cohorts.

    :param cohorts: Cohort name to configuration; configurations may differ only in name and
        identity-selection seed.
    :type cohorts: dict[str, ExperimentConfig]
    :param paths: Working data home.
    :type paths: Paths
    :param output: Directory that receives one attempt per cohort.
    :type output: Path
    :param family: Graph family, see :func:`build_reservoir`.
    :type family: str
    :param seed: Null-model seed for the seeded families.
    :type seed: Optional[int]
    :param alpha: Operating point; the gain becomes ``alpha`` over the graph's giant-component
        spectral radius. Exactly one of ``alpha`` and ``gain`` is given.
    :type alpha: Optional[float]
    :param gain: Explicit gain, dimensionless.
    :type gain: Optional[float]
    :param leak: Update fraction of the non-driven neurons.
    :type leak: float
    :param driven_leak: Update fraction of the driven neurons.
    :type driven_leak: float
    :param input_scale: Multiplier of the encoded current.
    :type input_scale: float
    :param steps_per_window: Synaptic updates per window.
    :type steps_per_window: int
    :param reset_each_window: Whether the state is zeroed before every window.
    :type reset_each_window: bool
    :param threads: Numba threads, or the configuration's setting.
    :type threads: Optional[int]
    :returns: Graph description and the attempt status of every cohort.
    :rtype: dict[str, Any]
    :raises ValueError: If the operating point, the cohorts or a partial attempt is invalid.
    """
    if (alpha is None) == (gain is None) or (family == FEEDFORWARD and alpha is not None):
        raise ValueError('Give exactly one of alpha and gain; the feedforward graph needs a gain.')
    first = next(iter(cohorts.values()))
    if any(_cohort_invariant(cfg=cfg) != _cohort_invariant(cfg=first) for cfg in cohorts.values()):
        raise ValueError('Cohort configurations may differ only in name and selection seed.')
    resolved = resolve_threads(paths=paths, configured=threads or first.brain.threads)
    numba.set_num_threads(resolved)
    input_idx = driven_neurons(cfg=first, paths=paths)
    reservoir = build_reservoir(
        paths=paths,
        family=family,
        seed=seed,
        input_idx=input_idx,
        leak=leak,
        driven_leak=driven_leak,
        batch_size=cohort_size(cfg=first),
    )
    matrix = reservoir.matrix()
    digest = hashlib.sha256()
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.ascontiguousarray(array).tobytes())
    radius, radius_source = _radius(matrix=matrix, digest=digest.hexdigest(), paths=paths)
    effective_gain = alpha / radius if alpha is not None else gain
    assert effective_gain is not None
    reservoir.gain = np.float32(effective_gain)
    graph = {
        'family': family,
        'seed': seed,
        'edges': reservoir.edges,
        RADIUS_KEY: radius,
        'radius_source': radius_source,
        'alpha': alpha,
        'gain': effective_gain,
        'graph_sha256': digest.hexdigest(),
    }
    status: dict[str, str] = {}
    for name, cfg in cohorts.items():
        target = output / name
        if output_path(path=target, paths=paths).exists():
            _require_completed(directory=target, paths=paths)
            status[name] = 'existing'
            continue
        parameters = {
            KIND: 'wiring_record',
            ISSUE_KEY: ISSUE,
            'cohort': name,
            'config_sha256': config_hash(cfg=cfg),
            **graph,
            'leak': leak,
            'driven_leak': driven_leak,
            'input_scale': input_scale,
            'steps_per_window': steps_per_window,
            'reset_each_window': reset_each_window,
            'numba_threads': resolved,
            'populations': list(KEEP),
            'trainable_fly_parameters': [],
        }
        with attempt(paths=paths, cfg=cfg, output=target, parameters=parameters) as directory:
            record_cohort(
                cfg=cfg,
                paths=paths,
                directory=directory,
                reservoir=reservoir,
                input_idx=input_idx,
                input_scale=input_scale,
                steps_per_window=steps_per_window,
                reset_each_window=reset_each_window,
                parameters=parameters,
                keep=KEEP,
            )
        status[name] = 'recorded'
    return {'graph': graph, 'cohorts': status}


def select_operating_points(
    cfg: ExperimentConfig,
    paths: Paths,
    evaluation: Path,
    output: Path,
    candidates: dict[str, dict[float, str]],
) -> dict[str, Any]:
    """Freeze, per graph family, the alpha with the best training-only CV accuracy.

    The score of a case is its best mean 5-fold CV accuracy over the C grid on the training
    photographs, as computed by the standard readout. Held-out accuracy is reported but never
    used. Ties go to the smaller alpha.

    :param cfg: Development cohort configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home.
    :type paths: Paths
    :param evaluation: Completed development evaluation attempt.
    :type evaluation: Path
    :param output: New immutable attempt directory.
    :type output: Path
    :param candidates: Family name to ``{alpha: case}``.
    :type candidates: dict[str, dict[float, str]]
    :returns: Selected alpha and all scores per family.
    :rtype: dict[str, Any]
    :raises ValueError: If the evaluation is not completed.
    """
    parameters = {
        KIND: 'wiring_select',
        ISSUE_KEY: ISSUE,
        'evaluation': str(evaluation),
        'candidates': {
            family: {str(a): c for a, c in cases.items()} for family, cases in candidates.items()
        },
        'rule': 'max over C of mean CV accuracy on training photographs; ties to smaller alpha',
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        _require_completed(directory=evaluation, paths=paths)
        scores = _read_json(path=output_path(path=evaluation, paths=paths) / REPORT_FILE)[SCORES]
        families: dict[str, Any] = {}
        for family, cases in candidates.items():
            table = {
                alpha: {
                    CV_ACCURACY: max(scores[case][CV_SCORES].values()),
                    'held_out_accuracy': scores[case][ACCURACY],
                    'C': scores[case]['C'],
                }
                for alpha, case in sorted(cases.items())
            }
            best = max(table, key=lambda alpha: (table[alpha][CV_ACCURACY], -alpha))
            families[family] = {
                'selected_alpha': best,
                'scores': {str(a): v for a, v in table.items()},
            }
        result = {PARAMETERS: parameters, 'families': families}
        write_json(path=directory / REPORT_FILE, value=result)
    return result


def _correctness(
    evaluations: dict[str, Path], paths: Paths, cases: Sequence[str]
) -> tuple[dict[str, NDArray[np.bool_]], NDArray[np.int64], NDArray[np.int64]]:
    """Load held-out correctness of every case, photographs stacked over cohorts.

    Identities are numbered globally, so an identity of one cohort never merges with another.

    :param evaluations: Cohort name to completed evaluation attempt.
    :type evaluations: dict[str, Path]
    :param paths: Working data home.
    :type paths: Paths
    :param cases: Case names to load.
    :type cases: Sequence[str]
    :returns: Correctness per case, shape (M,); global identity of every photograph, shape (M,);
        cohort index of every identity, shape (identities,).
    :rtype: tuple[dict[str, NDArray[np.bool_]], NDArray[np.int64], NDArray[np.int64]]
    :raises ValueError: If an evaluation is incomplete or cases disagree on their photographs.
    """
    correct: dict[str, list[NDArray[np.bool_]]] = {case: [] for case in cases}
    identity_of: list[NDArray[np.int64]] = []
    cohort_of: list[NDArray[np.int64]] = []
    offset = 0
    for index, evaluation in enumerate(evaluations.values()):
        _require_completed(directory=evaluation, paths=paths)
        rows = read_table(path=output_path(path=evaluation, paths=paths) / PREDICTIONS_FILE)
        reference: list[tuple[str, int]] | None = None
        for case in cases:
            selected = sorted(
                (row[SAMPLE_ID], row[LABEL], row[PREDICTED]) for row in rows if row[CASE] == case
            )
            photos = [(sample, label) for sample, label, _ in selected]
            if not selected or (reference is not None and photos != reference):
                raise ValueError(f'Case {case} is missing or covers other photographs.')
            reference = photos
            correct[case].append(
                np.asarray([label == guess for _, label, guess in selected], dtype=np.bool_)
            )
        assert reference is not None
        _, local = np.unique([label for _, label in reference], return_inverse=True)
        identity_of.append(local.astype(np.int64) + offset)
        cohort_of.append(np.full(shape=local.max() + 1, fill_value=index, dtype=np.int64))
        offset += int(local.max()) + 1
    return (
        {case: np.concatenate(values) for case, values in correct.items()},
        np.concatenate(identity_of),
        np.concatenate(cohort_of),
    )


def classify(low: float, high: float, margin: float = EQUIVALENCE_MARGIN_PP) -> str:
    """Read a 95% interval of a difference in percentage points against the preregistered margin.

    :param low: Lower interval bound, percentage points.
    :type low: float
    :param high: Upper interval bound, percentage points.
    :type high: float
    :param margin: Smallest difference of interest, percentage points.
    :type margin: float
    :returns: The frozen three-way reading, or inconclusive.
    :rtype: str
    """
    if low > 0:
        return ADVANTAGE
    if high < 0:
        return DISADVANTAGE
    if -margin < low and high < margin:
        return EQUIVALENT
    return INCONCLUSIVE


def analyze_wiring(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    evaluations: dict[str, Path],
    contrasts: list[Contrast],
) -> dict[str, Any]:
    """Pool every cohort and estimate each contrast with a two-level bootstrap.

    The point estimate is the pooled accuracy of ``first`` minus the mean pooled accuracy of the
    ensemble, and a single-case ensemble also gets McNemar's test. Each of 10,000 draws
    resamples identities with replacement within every cohort and, independently, the
    ensemble's seeds with replacement, so the interval covers photograph sampling and null-graph
    realization.

    :param cfg: First confirmation cohort configuration, for the attempt record.
    :type cfg: ExperimentConfig
    :param paths: Working data home.
    :type paths: Paths
    :param output: New immutable attempt directory.
    :type output: Path
    :param evaluations: Cohort name to completed evaluation attempt.
    :type evaluations: dict[str, Path]
    :param contrasts: Preregistered contrasts.
    :type contrasts: list[Contrast]
    :returns: Pooled accuracies, contrasts with intervals and decisions, and per-cohort values.
    :rtype: dict[str, Any]
    """
    parameters = {
        KIND: 'wiring_analyze',
        ISSUE_KEY: ISSUE,
        'evaluations': {name: str(path) for name, path in evaluations.items()},
        CONTRASTS: [vars(contrast) | {ENSEMBLE: list(contrast.ensemble)} for contrast in contrasts],
        'bootstrap_samples': BOOTSTRAP_SAMPLES,
        'equivalence_margin_pp': EQUIVALENCE_MARGIN_PP,
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        cases = sorted({case for c in contrasts for case in (c.first, *c.ensemble)})
        correct, identity_of, cohort_of = _correctness(
            evaluations=evaluations, paths=paths, cases=cases
        )
        photos = len(identity_of)
        counts = {
            case: np.bincount(identity_of, weights=values, minlength=len(cohort_of))
            for case, values in correct.items()
        }
        generator = np.random.default_rng(
            seed=np.random.SeedSequence(entropy=[cfg.seed, stable_int(key=BOOTSTRAP_NAMESPACE)])
        )
        members = [np.flatnonzero(cohort_of == index) for index in range(len(evaluations))]
        draws = np.stack(
            [
                np.concatenate([generator.choice(a=group, size=len(group)) for group in members])
                for _ in range(BOOTSTRAP_SAMPLES)
            ]
        )
        accuracy = {
            case: {
                'correct': int(values.sum()),
                PHOTOGRAPHS: photos,
                'accuracy_pct': float(values.sum() / photos * PERCENT),
                'wilson_95_pct': [
                    bound * PERCENT
                    for bound in wilson_interval(successes=int(values.sum()), n=photos)
                ],
            }
            for case, values in counts.items()
        }
        results = []
        for contrast in contrasts:
            first = counts[contrast.first]
            ensemble = np.stack([counts[case] for case in contrast.ensemble])
            seeds = generator.integers(
                low=0, high=len(ensemble), size=(BOOTSTRAP_SAMPLES, len(ensemble))
            )
            reference = ensemble[seeds[:, :, None], draws[:, None, :]].sum(axis=2).mean(axis=1)
            differences = (first[draws].sum(axis=1) - reference) / photos * PERCENT
            low, high = (float(value) for value in np.percentile(differences, CONFIDENCE_TAILS))
            per_cohort = {
                name: float(
                    (
                        first[cohort_of == index].sum()
                        - ensemble[:, cohort_of == index].sum(axis=1).mean()
                    )
                    / np.isin(
                        element=identity_of, test_elements=np.flatnonzero(cohort_of == index)
                    ).sum()
                    * PERCENT
                )
                for index, name in enumerate(evaluations)
            }
            entry: dict[str, Any] = {
                NAME: contrast.name,
                'first': contrast.first,
                ENSEMBLE: list(contrast.ensemble),
                'difference_pp': float(
                    (first.sum() - ensemble.sum(axis=1).mean()) / photos * PERCENT
                ),
                'interval_95_pp': [low, high],
                'decision': classify(low=low, high=high),
                'per_cohort_pp': per_cohort,
            }
            if len(contrast.ensemble) == 1:
                other = correct[contrast.ensemble[0]]
                entry['mcnemar'] = mcnemar(
                    n01=int(np.sum(~correct[contrast.first] & other)),
                    n10=int(np.sum(correct[contrast.first] & ~other)),
                )
            results.append(entry)
        result = {
            PARAMETERS: parameters,
            'identities': len(cohort_of),
            PHOTOGRAPHS: photos,
            ACCURACY: accuracy,
            CONTRASTS: results,
        }
        write_json(path=directory / REPORT_FILE, value=result)
    return result
