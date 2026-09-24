"""Summarize T32 decode attempts into Markdown tables and a Mermaid chart."""

import json
import math
import os
from pathlib import Path

BASE = Path(os.environ['FLYSTATE_HOME']) / 'runs/diagnostics/2026-09-25-drive-sweep'
CONDITIONS = ['s0-on', 's1-on', 's2-on', 's4-on', 's8-on', 's16-on', 's1-off', 's4-off', 's16-off']
POPULATIONS = [
    'driven',
    'visual_projection_other',
    'central_brain',
    'kenyon',
    'descending',
    'optic_lobe_sample',
    'vnc',
]
LABELS = {
    'driven': 'Driven neurons (control)',
    'visual_projection_other': 'Other visual projection',
    'central_brain': 'Central brain',
    'kenyon': 'Kenyon cells',
    'descending': 'Descending neurons',
    'optic_lobe_sample': 'Optic lobe sample',
    'vnc': 'VNC intrinsic',
}
REPS = {
    'all_windows_counts': 'all-window spike counts',
    'last_window_counts': 'last-window spike counts',
    'final_voltage': 'final voltage',
}
table = {}
for c in CONDITIONS:
    f = BASE / 'decode' / c / 'report.json'
    if not f.exists():
        continue
    for m in json.loads(f.read_text())['metrics']:
        table[(m['condition'], m['population'], m['representation'])] = m
done = [c for c in CONDITIONS if (c, 'driven', 'all_windows_counts') in table]
for rep, title in REPS.items():
    print(f'\n#### {title.capitalize()}, OOF top-1 correct of 200 (log loss)\n')
    print('| Population | ' + ' | '.join(done) + ' |')
    print('| --- | ' + ' | '.join('---:' for _ in done) + ' |')
    for p in POPULATIONS:
        cells = []
        for c in done:
            m = table.get((c, p, rep))
            if m is None or m['status'] != 'completed':
                cells.append('fail' if m else '-')
                continue
            mark = '**' if m['top1_correct'] >= 46 and m['log_loss'] < math.log(20) else ''
            cells.append(f'{mark}{m["top1_correct"]}{mark} ({m["log_loss"]:.2f})')
        print(f'| {LABELS[p]} | ' + ' | '.join(cells) + ' |')
on = [c for c in done if c.endswith('-on')]
print('\n```mermaid\nxychart-beta\n  title "All-window spike counts, noise on: OOF top-1 of 200"')
print('  x-axis "Encoder amplitude multiplier" [' + ', '.join(f'"{c[1:-3]}"' for c in on) + ']')
print('  y-axis "Correct of 200" 0 --> 100')
for p in ['driven', 'central_brain', 'descending', 'visual_projection_other']:
    print(
        '  line ['
        + ', '.join(str(table[(c, p, 'all_windows_counts')]['top1_correct']) for c in on)
        + ']'
    )
print('```')
