"""Deterministic offline fixtures matching the released flybrain file formats."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import sparse

from flystate.datasets.celeba import LANDMARK_HEADER


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


def make_synthetic_celeba(
    root: Path, identities: int = 8, images_per_identity: int = 25, seed: int = 0
) -> None:
    """Generate local JPEGs and all official annotation formats without real face data.

    :param root: Parent of the torchvision-style celeba directory.
    :type root: Path
    :param identities: Positive number of synthetic identities.
    :type identities: int
    :param images_per_identity: Positive image count per identity.
    :type images_per_identity: int
    :param seed: Deterministic image and geometry seed.
    :type seed: int
    :raises ValueError: If either dataset dimension is nonpositive.
    """
    if min(identities, images_per_identity) < 1:
        raise ValueError('Synthetic identity and image counts must be positive.')
    directory = root / 'celeba'
    images = directory / 'img_align_celeba'
    images.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed=seed)
    template = np.array(
        object=[[69, 109], [106, 113], [87, 132], [73, 152], [108, 154]], dtype=np.float64
    )
    identity_lines: list[str] = []
    landmark_lines = [str(identities * images_per_identity), ' '.join(LANDMARK_HEADER)]
    partition_lines: list[str] = []
    for index in range(identities * images_per_identity):
        identity = index // images_per_identity + 1
        filename = f'{index + 1:06d}.jpg'
        angle = np.deg2rad(generator.uniform(low=-10, high=10))
        rotation = np.array(
            object=[[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        scale = generator.uniform(low=0.9, high=1.1)
        shift = generator.uniform(low=-5, high=5, size=2)
        landmarks = (
            (template - template.mean(axis=0)) @ rotation.T * scale + template.mean(axis=0) + shift
        )
        background = generator.integers(low=90, high=230, size=(218, 178, 3), dtype=np.uint8)
        image = Image.fromarray(obj=background)
        draw = ImageDraw.Draw(im=image)
        for x, y in landmarks[:2]:
            draw.ellipse(xy=(x - 5, y - 3, x + 5, y + 3), fill=(20, 20, 20))
        image.save(fp=images / filename, format='JPEG', quality=85)
        identity_lines.append(f'{filename} {identity}')
        landmark_lines.append(
            f'{filename} ' + ' '.join(f'{value:.6f}' for value in landmarks.ravel())
        )
        partition_lines.append(f'{filename} {(0, 0, 0, 1, 2)[index % 5]}')
    for name, lines in (
        ('identity_CelebA.txt', identity_lines),
        ('list_landmarks_align_celeba.txt', landmark_lines),
        ('list_eval_partition.txt', partition_lines),
    ):
        (directory / name).write_text(data='\n'.join(lines) + '\n', encoding='utf-8')
