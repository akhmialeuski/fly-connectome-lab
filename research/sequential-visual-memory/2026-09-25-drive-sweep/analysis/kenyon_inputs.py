"""Composition of Kenyon-cell inputs in flybrain's row-normalized weights (reported in #64)."""

import os
from pathlib import Path

import numpy as np
from scipy import sparse

BRAIN = Path(os.environ['FLYSTATE_HOME']) / 'data/brain'
meta = np.load(BRAIN / 'brain.npz')
cell_type = meta['cell_type'].astype(str)
weights = sparse.load_npz(BRAIN / 'weights.npz').tocsr()
kenyon = np.char.startswith(cell_type, 'KC')
apl = np.isin(cell_type, ['APL'])
rows = weights[kenyon]
print('Kenyon cells', int(kenyon.sum()))
print('mean input weight from other Kenyon cells', float(rows[:, kenyon].sum(axis=1).mean()))
print('mean input weight from APL', float(rows[:, apl].sum(axis=1).mean()))
print('input if every Kenyon cell fires, gain 3', 3 * float(rows[:, kenyon].sum(axis=1).mean()))
