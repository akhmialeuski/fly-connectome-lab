"""Exploratory probe: does the encoded image actually make input neurons spike?"""

from pathlib import Path

import numpy as np

from flystate.brain.runtime import EpisodeBrain
from flystate.datasets.preprocess import prepare_dataset
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.episodes.episode import EpisodeBuilder
from flystate.experiments.config import load_config
from flystate.settings import get_paths

cfg = load_config(Path('configs/celeba-smoke.yaml'))
paths = get_paths()
prep = prepare_dataset(cfg=cfg, paths=paths)
brain = EpisodeBrain(
    brain_dir=paths.brain, brain_cfg=cfg.brain, readout_cfg=cfg.readout, batch_size=1, threads=4
)
enc = SparseProjectionEncoder(
    cfg=cfg.encoder,
    window=cfg.episodes.window,
    candidate_neurons=brain.cells(superclasses=[cfg.encoder.target_population]),
)
builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
rest = brain.compute_rest_state(seed=cfg.seed)
print('n', brain.n, 'input', len(enc.input_idx), 'readout', len(brain.readout_idx))
print('rest v stats', np.percentile(rest.v, [1, 10, 50, 90, 99]), 'rest fired', len(rest.fired))
train_rows = [i for i, s in enumerate(prep.samples) if s.split == 'train'][:6]
vpn = brain.cells(['visual_projection'])
allcur = []
for row in train_rows:
    ep = builder.build(sample=prep.samples[row], image=prep.images[row])
    for cond in ('stim', 'blank'):
        brain.begin(rest=rest, sample_ids=[ep.sample_id], seed=cfg.seed)
        tot = []
        for step in range(cfg.episodes.steps):
            cur = enc.encode(
                observations=ep.observations[step][None], positions=ep.positions[step][None]
            )
            if cond == 'stim':
                allcur.append(cur[:, 0])
            s = brain.run(
                input_idx=enc.input_idx,
                currents=cur if cond == 'stim' else np.zeros_like(cur),
                n_steps=10,
            )
            tot.append((int(s.input_spikes[0]), int(s.readout_spikes[0]), int(s.spikes_total[0])))
        tot = np.array(tot)
        print(row, cond, 'input/readout/total spikes per window mean', tot.mean(0).round(1))
cur = np.concatenate(allcur)
print('current pct', np.percentile(cur, [0, 1, 5, 25, 50, 75, 95, 99, 100]))
print('frac current > 0.0413 (suprathreshold alone):', (cur > 0.0413).mean())
