"""Leading eigenvalues of flybrain's effective (sensory-masked) weight matrix."""

import os
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

meta = np.load(Path(os.environ['FLYSTATE_HOME']) / 'data/brain/brain.npz')
W = sparse.load_npz(Path(os.environ['FLYSTATE_HOME']) / 'data/brain/weights.npz').tocsr()
sensory = np.char.find(meta['superclass'].astype(str), 'sensory') >= 0
W = (sparse.diags((~sensory).astype(np.float32)) @ W).tocsr().astype(np.float64)
W.eliminate_zeros()
print('effective nnz', W.nnz)
t = time.time()
vals = eigs(W, k=6, which='LM', return_eigenvectors=False, tol=1e-6, maxiter=20000)
print('top |lambda|', sorted(np.abs(vals))[::-1], 'values', vals, round(time.time() - t), 's')
