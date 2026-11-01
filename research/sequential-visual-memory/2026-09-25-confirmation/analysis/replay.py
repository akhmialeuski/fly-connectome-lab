"""Independently recompute T35 held-out scores and paired tests from the saved predictions."""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import binomtest

SNAPSHOT = Path(__file__).resolve().parents[1] / 'snapshot' / 'evaluate'


def main() -> None:
    """Compare recomputed accuracies, binomial p-values and McNemar counts with the report."""
    report = json.loads((SNAPSHOT / 'report.json').read_text(encoding='utf-8'))
    table = pq.read_table(SNAPSHOT / 'predictions.parquet').to_pylist()
    correct: dict[str, dict[str, bool]] = {}
    for row in table:
        correct.setdefault(row['case'], {})[row['sample_id']] = row['label'] == row['predicted']
    mismatches = 0
    for case, stored in report['scores'].items():
        values = np.array(list(correct[case].values()))
        successes = int(values.sum())
        p_value = binomtest(successes, len(values), 0.05, alternative='greater').pvalue
        if successes != stored['held_out_correct'] or not np.isclose(
            p_value, stored['binomial_p_above_chance'], rtol=1e-9
        ):
            mismatches += 1
    for comparison in report['comparisons']:
        first, second = correct[comparison['a']], correct[comparison['b']]
        n01 = sum(not first[s] and second[s] for s in first)
        n10 = sum(first[s] and not second[s] for s in first)
        gain = 100 * (sum(first.values()) - sum(second.values())) / len(first)
        stored = comparison['mcnemar']
        if (n01, n10) != (stored['n01'], stored['n10']) or not np.isclose(
            gain, comparison['a_minus_b_pp']
        ):
            mismatches += 1
    result = {
        'cases': len(report['scores']),
        'comparisons': len(report['comparisons']),
        'prediction_rows': len(table),
        'mismatches': mismatches,
    }
    print(json.dumps(result))
    sys.exit(1 if mismatches else 0)


if __name__ == '__main__':
    main()
