"""Seeded moving-window geometry independent of images and model state."""

import math

import numpy as np
from numpy.typing import NDArray

from flystate.experiments.config import TrajectoryConfig
from flystate.hashing import stable_int

MAX_WINDOW_DRAWS: int = 1000


class TrajectoryError(Exception):
    """Requested window geometry cannot produce a valid observation sequence."""


def make_boxes(
    sample_id: str, image_size: int, window: int, steps: int, trajectory: TrajectoryConfig
) -> NDArray[np.int32]:
    """Generate deterministic half-open pixel boxes for one episode.

    :param sample_id: Stable episode identifier used to derive the random stream.
    :type sample_id: str
    :param image_size: Positive aligned image side S in pixels.
    :type image_size: int
    :param window: Window side W, 1 <= W <= S.
    :type window: int
    :param steps: Positive observation count T.
    :type steps: int
    :param trajectory: Strategy, maximum shared-area fraction, and seed.
    :type trajectory: TrajectoryConfig
    :returns: Int32 boxes (T,4), each x0,y0,x1,y1 with exclusive upper bounds.
    :rtype: NDArray[np.int32]
    :raises TrajectoryError: If dimensions, grid size, or random placements are infeasible.
    """
    if not 1 <= window <= image_size or steps < 1:
        raise TrajectoryError('Require positive steps and 1 <= window <= image_size.')
    generator = np.random.default_rng(
        seed=np.random.SeedSequence(entropy=[trajectory.seed, stable_int(key=sample_id)])
    )
    if trajectory.strategy in {'raster', 'permuted'}:
        grid = math.isqrt(steps)
        if grid < 2 or grid * grid != steps:
            raise TrajectoryError('Grid trajectories require a square step count of at least four.')
        offsets = [round(index * (image_size - window) / (grid - 1)) for index in range(grid)]
        boxes = np.asarray(
            a=[(x, y, x + window, y + window) for y in offsets for x in offsets], dtype=np.int32
        )
        return boxes[generator.permutation(x=steps)] if trajectory.strategy == 'permuted' else boxes
    accepted: list[tuple[int, int, int, int]] = []
    maximum = trajectory.max_overlap * window * window
    for _ in range(steps):
        for _attempt in range(MAX_WINDOW_DRAWS):
            x, y = generator.integers(low=0, high=image_size - window + 1, size=2)
            candidate = (int(x), int(y), int(x) + window, int(y) + window)
            if all(
                max(0, min(candidate[2], box[2]) - max(candidate[0], box[0]))
                * max(0, min(candidate[3], box[3]) - max(candidate[1], box[1]))
                <= maximum
                for box in accepted
            ):
                accepted.append(candidate)
                break
        else:
            raise TrajectoryError(
                f'Cannot place observation {len(accepted) + 1} for {sample_id!r} '
                f'after {MAX_WINDOW_DRAWS} draws.'
            )
    return np.asarray(a=accepted, dtype=np.int32)
