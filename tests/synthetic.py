"""Deterministic offline fixtures matching the released flybrain file formats."""

from pathlib import Path

import numpy as np
from scipy import sparse


def make_synthetic_brain(
    path: Path,
    n: int = 2000,
    n_descending: int = 100,
    n_visual_projection: int = 600,
    n_sensory: int = 50,
    out_degree: int = 30,
    seed: int = 0,
) -> None:
    """Write a small signed row-normalized connectome and typed NPZ metadata.

    :param path: Destination directory.
    :type path: Path
    :param n: Total neuron count.
    :type n: int
    :param n_descending: Descending population size, at least two.
    :type n_descending: int
    :param n_visual_projection: Visual input population size.
    :type n_visual_projection: int
    :param n_sensory: Photoreceptor population size.
    :type n_sensory: int
    :param out_degree: Distinct postsynaptic targets per presynaptic neuron.
    :type out_degree: int
    :param seed: Explicit generator seed.
    :type seed: int
    :raises ValueError: If requested populations or connectivity do not fit.
    """
    if n_descending < 2 or min(n_visual_projection, n_sensory) < 1:
        raise ValueError('Synthetic populations require two descending and positive input counts.')
    if n_descending + n_visual_projection + n_sensory > n or not 1 <= out_degree <= n:
        raise ValueError('Synthetic populations and out-degree must fit the neuron count.')
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=seed)
    posts = np.concatenate([rng.choice(a=n, size=out_degree, replace=False) for _ in range(n)])
    pres = np.repeat(a=np.arange(n), repeats=out_degree)
    values = rng.normal(size=n * out_degree).astype(np.float32)
    weights = sparse.csr_matrix((values, (posts, pres)), shape=(n, n), dtype=np.float32)
    row_sums = np.asarray(abs(weights).sum(axis=1)).ravel()
    weights.data /= np.repeat(a=np.maximum(row_sums, 1.0), repeats=np.diff(a=weights.indptr))
    sparse.save_npz(file=path / 'weights.npz', matrix=weights, compressed=False)
    classes = np.full(shape=n, fill_value='cb_intrinsic', dtype='<U20')
    classes[:n_descending] = 'descending_neuron'
    visual_end = n_descending + n_visual_projection
    classes[n_descending:visual_end] = 'visual_projection'
    classes[visual_end : visual_end + n_sensory] = 'ol_sensory'
    np.savez(
        file=path / 'brain.npz',
        ids=np.arange(n, dtype=np.int64) + 10**9,
        visual=np.arange(visual_end, visual_end + n_sensory, dtype=np.int64),
        azimuth=np.linspace(start=-1, stop=1, num=n_sensory, dtype=np.float32),
        cell_type=classes,
        side=np.where(np.arange(n) % 2 == 0, 'L', 'R'),
        positions=np.zeros(shape=(n, 3), dtype=np.float32),
        superclass=classes,
        group_escape_L=np.array(object=[0], dtype=np.int64),
        group_escape_R=np.array(object=[1], dtype=np.int64),
    )
