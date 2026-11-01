"""Blank (no image) firing rate per superclass/KC over 50 steps after warmup, noise on and off."""

from pathlib import Path

import numpy as np

from flystate.brain.runtime import EpisodeBrain
from flystate.experiments.config import load_config
from flystate.settings import get_paths

paths = get_paths()
meta = np.load(paths.brain / 'brain.npz')
sc = meta['superclass'].astype(str)
ct = meta['cell_type'].astype(str)
groups = sc.copy()
groups[np.char.startswith(ct, 'KC')] = 'KC'
groups[np.char.startswith(ct, 'MBON')] = 'MBON'
names, inv = np.unique(groups, return_inverse=True)
sizes = np.bincount(inv)
for noise in ('true', 'false'):
    cfg = load_config(Path('configs/celeba-smoke.yaml'), overrides=[f'brain.noise.enabled={noise}'])
    b = EpisodeBrain(
        brain_dir=paths.brain, brain_cfg=cfg.brain, readout_cfg=cfg.readout, batch_size=1, threads=8
    )
    rest = b.compute_rest_state(seed=0)
    b.begin(rest=rest, sample_ids=['blank'], seed=0)
    counts = np.zeros(len(names))
    ever = np.zeros(b.n, bool)
    for _ in range(50):
        b._step(input_idx=np.empty(0, np.int64), currents=None)
        counts += np.bincount(inv[b._fb.fired], minlength=len(names))
        ever[b._fb.fired] = True
    rate = counts / sizes / (50 * 0.02)
    act = np.bincount(inv[ever], minlength=len(names)) / sizes
    print(f'noise={noise}')
    for n, r, a, s in sorted(zip(names, rate, act, sizes, strict=True), key=lambda t: -t[1]):
        if s >= 50:
            print(f'  {n:22s} n={s:6d} rate={r:6.2f} Hz  active_frac={a:.3f}')
