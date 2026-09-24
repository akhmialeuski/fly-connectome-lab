"""Markdown tables of the T33 decode and memory attempts, as posted in #65."""

import json
import math
import os
from pathlib import Path

BASE = Path(os.environ['FLYSTATE_HOME']) / 'runs/diagnostics/2026-09-25-rate-access'
SETTINGS = ['g0.5-l1.0', 'g1.0-l1.0', 'g1.5-l1.0', 'g0.5-l0.25', 'g1.0-l0.25', 'g1.5-l0.25']
LABELS = {
    'driven': 'Driven neurons (control)',
    'visual_projection_other': 'Other visual projection',
    'central_brain': 'Central brain',
    'kenyon': 'Kenyon cells',
    'descending': 'Descending neurons',
    'optic_lobe_sample': 'Optic lobe sample',
    'vnc': 'VNC intrinsic',
}


def cell(row: dict | None) -> str:
    """Format one decode metric as correct-of-200 with log loss, bold if delivery is restored.

    :param row: Decode metric row, or none when absent.
    :type row: Optional[dict]
    :returns: Markdown table cell text.
    :rtype: str
    """
    if row is None:
        return '-'
    if row['status'] != 'completed':
        return 'fail'
    bold = '**' if row['top1_correct'] >= 46 and row['log_loss'] < math.log(20) else ''
    return f'{bold}{row["top1_correct"]}{bold} ({row["log_loss"]:.2f})'


def main() -> None:
    """Print the all-window, last-window and memory-test tables."""
    metrics = {}
    for setting in SETTINGS:
        report = json.loads((BASE / 'decode' / setting / 'report.json').read_text())
        for row in report['metrics']:
            metrics[(row['condition'], row['population'], row['representation'])] = row
    header = '| Population | ' + ' | '.join(SETTINGS) + ' |'
    rule = '| --- | ' + ' | '.join('---:' for _ in SETTINGS) + ' |'
    print(header, rule, sep='\n')
    for population, label in LABELS.items():
        cells = [
            cell(metrics.get((f'{s}-persistent', population, 'all_windows_state')))
            for s in SETTINGS
        ]
        print(f'| {label} | ' + ' | '.join(cells) + ' |')
    print(header, rule, sep='\n')
    for population, label in LABELS.items():
        cells = [
            f'{metrics[(f"{s}-persistent", population, "last_window_state")]["top1_correct"]} / '
            f'{metrics[(f"{s}-reset", population, "last_window_state")]["top1_correct"]}'
            for s in SETTINGS
        ]
        print(f'| {label} | ' + ' | '.join(cells) + ' |')
    memory = json.loads((BASE / 'memory' / 'report.json').read_text())
    for row in memory['comparisons']:
        low, high = row['interval_bonferroni_pp']
        print(
            LABELS[row['population']],
            row['setting'],
            row['reset_correct'],
            row['persistent_correct'],
            f'{row["persistent_minus_reset_pp"]:+.1f}',
            [round(value, 1) for value in row['interval_95_pp']],
            [round(low, 1), round(high, 1)],
            row['memory_supported'],
        )


if __name__ == '__main__':
    main()
