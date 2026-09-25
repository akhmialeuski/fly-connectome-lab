"""Independently verify the T37 archive from its predictions, and optionally from recorded states.

Without ``--home`` the script needs only this study directory. It checks every archived readout
digest, recomputes every held-out score from ``predictions.parquet``, repeats the Phase A
selection rule, and recomputes every pooled accuracy, contrast, per-cohort value, McNemar count
and two-level bootstrap interval of the Phase B analysis with its own NumPy code.

With ``--home`` it also applies every archived affine readout to the recorded final states in the
working data home and requires the stored prediction for every held-out photograph. Run it this
way before the states are deleted.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import binomtest

STUDY: Path = Path(__file__).resolve().parents[1]
SNAPSHOT: Path = STUDY / 'snapshot'
RUNS: str = 'runs/diagnostics/2026-09-25-wiring'
REPORT: str = 'report.json'
PREDICTIONS: str = 'predictions.parquet'
WEIGHTS: str = 'weights.npz'
METADATA: str = 'model.json'
RESPONSES: str = 'responses.npz'
INVENTORY: str = 'checksums.sha256'
FINAL_PREFIX: str = 'final_'
BOOTSTRAP_SAMPLES: int = 10000
BOOTSTRAP_NAMESPACE: str = 't37-two-level-bootstrap'
TAILS: tuple[float, float] = (2.5, 97.5)
MARGIN_PP: float = 5.0
PERCENT: float = 100.0
TOLERANCE: float = 1e-9
BLOCK: int = 1 << 20
CASE: str = 'case'
PREDICTED: str = 'predicted'
LABEL: str = 'label'
SAMPLE_ID: str = 'sample_id'
CONTRASTS: str = 'contrasts'
ENSEMBLE: str = 'ensemble'
EVALUATE: str = 'evaluate'
EVALUATIONS: str = 'evaluations'
FIRST: str = 'first'
MCNEMAR: str = 'mcnemar'
NAME: str = 'name'
PARAMETERS: str = 'parameters'
PHASE_A: str = 'phase-a'
PHASE_B: str = 'phase-b'
REPLAYED: str = 'replayed_predictions'
SCORES: str = 'scores'
ENCODING: str = 'utf-8'


def _json(path: Path) -> dict[str, Any]:
    """Read one JSON object.

    :param path: JSON file.
    :type path: Path
    :returns: Parsed object.
    :rtype: dict[str, Any]
    """
    return json.loads(s=path.read_text(encoding=ENCODING))


def _sha256(path: Path) -> str:
    """Hash a file in blocks.

    :param path: File to hash.
    :type path: Path
    :returns: Lowercase SHA-256 hex digest.
    :rtype: str
    """
    digest = hashlib.sha256()
    with path.open(mode='rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            digest.update(block)
    return digest.hexdigest()


def _stable_int(key: str) -> int:
    """Return the first 8 bytes of SHA-256 as a big-endian integer, as ``flystate.hashing`` does.

    :param key: Namespace string.
    :type key: str
    :returns: Unsigned 64-bit integer.
    :rtype: int
    """
    return int.from_bytes(hashlib.sha256(key.encode(ENCODING)).digest()[:8], byteorder='big')


def verify_inventory(directory: Path) -> int:
    """Check an archived attempt against its own ``checksums.sha256``.

    Every listed file must match its digest, except the recorded state arrays, which stay
    outside Git. No unlisted file may exist.

    :param directory: Archived attempt directory.
    :type directory: Path
    :returns: Number of verified files.
    :rtype: int
    :raises ValueError: If a file is changed, missing, or unlisted.
    """
    listed = {}
    for line in (directory / INVENTORY).read_text(encoding=ENCODING).splitlines():
        digest, name = line.split('  ', 1)
        listed[name] = digest
    for name, digest in listed.items():
        member = directory / name
        if name == RESPONSES and not member.exists():
            continue
        if not member.is_file() or _sha256(path=member) != digest:
            raise ValueError(f'Archived file changed or missing: {member}')
    present = {str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file()}
    if present - set(listed) - {INVENTORY}:
        raise ValueError(f'Unlisted files in {directory}')
    return len(listed)


def _rows(evaluation: Path) -> dict[str, list[tuple[str, int, int]]]:
    """Group sorted (sample, label, prediction) rows by case.

    :param evaluation: Archived evaluation attempt.
    :type evaluation: Path
    :returns: Rows of every case sorted by sample identifier.
    :rtype: dict[str, list[tuple[str, int, int]]]
    """
    table = pq.read_table(source=evaluation / PREDICTIONS).to_pylist()
    grouped: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for row in table:
        grouped[row[CASE]].append((row[SAMPLE_ID], int(row[LABEL]), int(row[PREDICTED])))
    return {case: sorted(values) for case, values in grouped.items()}


def verify_evaluation(evaluation: Path, home: Path | None) -> int:
    """Check readout digests and held-out scores, and optionally replay every prediction.

    :param evaluation: Archived evaluation attempt.
    :type evaluation: Path
    :param home: Working data home with the recorded states, or none.
    :type home: Optional[Path]
    :returns: Number of replayed predictions (0 without ``home``).
    :rtype: int
    :raises ValueError: On any mismatch.
    """
    report = _json(path=evaluation / REPORT)
    rows = _rows(evaluation=evaluation)
    replayed = 0
    if set(rows) != set(report[SCORES]):
        raise ValueError(f'Cases differ between report and predictions: {evaluation}')
    for case, score in report[SCORES].items():
        directory = evaluation / 'models' / case
        metadata = _json(path=directory / METADATA)
        if (
            metadata != score['model']
            or _sha256(path=directory / WEIGHTS) != metadata['weights_sha256']
        ):
            raise ValueError(f'Readout metadata or digest differs: {evaluation.name}/{case}')
        correct = sum(label == guess for _, label, guess in rows[case])
        chance = score['chance']
        p_value = binomtest(k=correct, n=len(rows[case]), p=chance, alternative='greater').pvalue
        if correct != score['held_out_correct'] or not np.isclose(
            p_value, score['binomial_p_above_chance'], rtol=TOLERANCE
        ):
            raise ValueError(f'Held-out score differs: {evaluation.name}/{case}')
        if home is None:
            continue
        recording_name, population = case.split('/', 1)
        recording = home / report[PARAMETERS]['recordings'][recording_name]
        record = _json(path=recording / REPORT)
        if _sha256(path=recording / RESPONSES) != record['responses_sha256']:
            raise ValueError(f'Recorded states changed: {recording}')
        order = {sample: index for index, sample in enumerate(record['sample_ids'])}
        with np.load(file=recording / RESPONSES, allow_pickle=False) as states:
            x = states[f'{FINAL_PREFIX}{population}'][[order[s] for s, _, _ in rows[case]]]
        with np.load(file=directory / WEIGHTS, allow_pickle=False) as model:
            scores = x.astype(np.float64) @ model['weights'].T + model['bias']
            predicted = model['classes'][scores.argmax(axis=1)]
        if not np.array_equal(predicted, [guess for _, _, guess in rows[case]]):
            raise ValueError(f'Affine readout replay differs: {evaluation.name}/{case}')
        replayed += len(predicted)
    return replayed


def verify_selection(select: Path, evaluation: Path) -> None:
    """Repeat the Phase A rule: best-C CV accuracy, ties to the smaller alpha.

    :param select: Archived selection attempt.
    :type select: Path
    :param evaluation: Archived development evaluation.
    :type evaluation: Path
    :raises ValueError: If any selected alpha differs.
    """
    report = _json(path=select / REPORT)
    scores = _json(path=evaluation / REPORT)[SCORES]
    for family, cases in report[PARAMETERS]['candidates'].items():
        table = {
            float(alpha): max(scores[case]['cv_accuracy_by_C'].values())
            for alpha, case in cases.items()
        }
        best = max(table, key=lambda alpha: (table[alpha], -alpha))
        if best != report['families'][family]['selected_alpha']:
            raise ValueError(f'Selection differs for {family}.')


def _classify(low: float, high: float) -> str:
    """Apply the frozen interval reading.

    :param low: Lower bound, percentage points.
    :type low: float
    :param high: Upper bound, percentage points.
    :type high: float
    :returns: Reading.
    :rtype: str
    """
    if low > 0:
        return 'advantage'
    if high < 0:
        return 'disadvantage'
    if low > -MARGIN_PP and high < MARGIN_PP:
        return 'no practically relevant difference'
    return 'inconclusive'


def verify_analysis(analysis: Path, evaluations: dict[str, Path], seed: int) -> None:
    """Recompute every pooled number of the Phase B analysis from the predictions.

    :param analysis: Archived analysis attempt.
    :type analysis: Path
    :param evaluations: Cohort name to archived evaluation, in the analysis order.
    :type evaluations: dict[str, Path]
    :param seed: Configuration seed that drove the bootstrap.
    :type seed: int
    :raises ValueError: On any mismatch.
    """
    report = _json(path=analysis / REPORT)
    cases = sorted({c for item in report[CONTRASTS] for c in (item[FIRST], *item[ENSEMBLE])})
    correct: dict[str, list[np.ndarray]] = {case: [] for case in cases}
    identity: list[np.ndarray] = []
    cohort_of: list[np.ndarray] = []
    offset = 0
    for index, evaluation in enumerate(evaluations.values()):
        rows = _rows(evaluation=evaluation)
        for case in cases:
            correct[case].append(np.array([label == guess for _, label, guess in rows[case]]))
        _, local = np.unique([label for _, label, _ in rows[cases[0]]], return_inverse=True)
        identity.append(local + offset)
        cohort_of.append(np.full(local.max() + 1, index))
        offset += local.max() + 1
    flat = {case: np.concatenate(values) for case, values in correct.items()}
    identity_of, cohorts = np.concatenate(identity), np.concatenate(cohort_of)
    photos = len(identity_of)
    counts = {
        case: np.bincount(identity_of, weights=v, minlength=len(cohorts))
        for case, v in flat.items()
    }
    for case, values in counts.items():
        if int(values.sum()) != report['accuracy'][case]['correct']:
            raise ValueError(f'Pooled accuracy differs: {case}')
    generator = np.random.default_rng(
        np.random.SeedSequence([seed, _stable_int(BOOTSTRAP_NAMESPACE)])
    )
    groups = [np.flatnonzero(cohorts == index) for index in range(len(evaluations))]
    draws = np.stack(
        [
            np.concatenate([generator.choice(g, size=len(g)) for g in groups])
            for _ in range(BOOTSTRAP_SAMPLES)
        ]
    )
    for item in report[CONTRASTS]:
        first = counts[item[FIRST]]
        ensemble = np.stack([counts[case] for case in item[ENSEMBLE]])
        seeds = generator.integers(0, len(ensemble), size=(BOOTSTRAP_SAMPLES, len(ensemble)))
        reference = ensemble[seeds[:, :, None], draws[:, None, :]].sum(axis=2).mean(axis=1)
        low, high = np.percentile((first[draws].sum(axis=1) - reference) / photos * PERCENT, TAILS)
        point = (first.sum() - ensemble.sum(axis=1).mean()) / photos * PERCENT
        if not np.allclose(
            [point, low, high], [item['difference_pp'], *item['interval_95_pp']], atol=TOLERANCE
        ):
            raise ValueError(f'Contrast differs: {item["name"]}')
        if _classify(low=float(low), high=float(high)) != item['decision']:
            raise ValueError(f'Decision differs: {item["name"]}')
        if len(item[ENSEMBLE]) == 1:
            a, b = flat[item[FIRST]], flat[item[ENSEMBLE][0]]
            expected = (int(np.sum(~a & b)), int(np.sum(a & ~b)))
            if (item[MCNEMAR]['n01'], item[MCNEMAR]['n10']) != expected:
                raise ValueError(f'McNemar counts differ: {item["name"]}')


def main() -> int:
    """Run every available check and print one JSON summary.

    :returns: Process exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--home', type=Path, default=None, help='FLYSTATE_HOME with recorded states.'
    )
    arguments = parser.parse_args()
    home = None if arguments.home is None else arguments.home.resolve()
    summary: dict[str, Any] = {EVALUATIONS: 0, REPLAYED: 0}
    attempts = sorted(inventory.parent for inventory in SNAPSHOT.rglob(INVENTORY))
    summary['verified_files'] = sum(verify_inventory(directory=attempt) for attempt in attempts)
    summary['attempts'] = len(attempts)
    for phase in (PHASE_A, PHASE_B):
        root = SNAPSHOT / phase / EVALUATE
        for evaluation in sorted(root.glob('s*')) if root.exists() else []:
            summary[REPLAYED] += verify_evaluation(evaluation=evaluation, home=home)
            summary[EVALUATIONS] += 1
    select = SNAPSHOT / PHASE_A / 'select'
    if select.exists():
        verify_selection(select=select, evaluation=SNAPSHOT / PHASE_A / EVALUATE / 's0')
        summary['selection'] = 'verified'
    analysis = SNAPSHOT / PHASE_B / 'analyze'
    if analysis.exists():
        report = _json(path=analysis / REPORT)
        evaluations = {
            name: SNAPSHOT / PHASE_B / EVALUATE / Path(path).name
            for name, path in report[PARAMETERS][EVALUATIONS].items()
        }
        seed = _json(path=analysis / 'config.json')['seed']
        verify_analysis(analysis=analysis, evaluations=evaluations, seed=seed)
        summary[CONTRASTS] = len(report[CONTRASTS])
    summary['home'] = None if home is None else str(home)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
