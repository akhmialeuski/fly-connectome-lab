"""Exploratory probe: evoked spikes per superclass vs amplitude and noise (paired noise streams)."""

from pathlib import Path

import numpy as np

from flystate.brain.runtime import EpisodeBrain
from flystate.datasets.preprocess import prepare_dataset
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.episodes.episode import EpisodeBuilder
from flystate.experiments.config import load_config
from flystate.settings import get_paths

cfg0 = load_config(Path('configs/celeba-smoke.yaml'))
paths = get_paths()
prep = prepare_dataset(cfg=cfg0, paths=paths)
meta = np.load(paths.brain / 'brain.npz')
sc = meta['superclass'].astype(str)
names, inv = np.unique(sc, return_inverse=True)
print({n: int((inv == i).sum()) for i, n in enumerate(names)})
row = next(i for i, s in enumerate(prep.samples) if s.split == 'train')
for noise in (True, False):
    cfg = load_config(
        Path('configs/celeba-smoke.yaml'), overrides=[f'brain.noise.enabled={str(noise).lower()}']
    )
    brain = EpisodeBrain(
        brain_dir=paths.brain, brain_cfg=cfg.brain, readout_cfg=cfg.readout, batch_size=1, threads=4
    )
    enc = SparseProjectionEncoder(
        cfg=cfg.encoder, window=32, candidate_neurons=brain.cells(['visual_projection'])
    )
    rest = brain.compute_rest_state(seed=0)
    ep = EpisodeBuilder(episodes=cfg.episodes, image_size=128).build(
        sample=prep.samples[row], image=prep.images[row]
    )
    for scale in (0.0, 1.0, 4.0, 10.0):
        brain.begin(rest=rest, sample_ids=[ep.sample_id], seed=0)
        counts = np.zeros(len(names))
        for step in range(4):
            cur = (
                enc.encode(
                    observations=ep.observations[step][None], positions=ep.positions[step][None]
                )
                * scale
            )
            for _ in range(10):
                brain._step(input_idx=enc.input_idx, currents=cur.astype(np.float32))
                counts += np.bincount(inv[brain._fb.fired], minlength=len(names))
        if scale == 0.0:
            base = counts.copy()
        diff = counts - base
        print(
            f'noise={noise} scale={scale}: evoked per superclass (4 windows):',
            {n: int(d) for n, d in zip(names, diff, strict=True) if abs(d) > 0 or scale == 0},
            'base DN',
            int(base[list(names).index('descending_neuron')]),
        )
