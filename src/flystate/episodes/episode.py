"""Content-preserving image observations with stable geometry and provenance."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flystate.datasets.subset import Sample
from flystate.episodes.trajectory import TrajectoryError, make_boxes
from flystate.experiments.config import EpisodesConfig
from flystate.hashing import sha256_obj


@dataclass(frozen=True)
class Episode:
    """One target and T RGB windows with int32 boxes and normalized float32 centers."""

    sample_id: str
    split: str
    target: int
    observations: NDArray[np.uint8]
    boxes: NDArray[np.int32]
    positions: NDArray[np.float32]


class EpisodeBuilder:
    """Create identical observations for every model from a config and sample identity."""

    def __init__(self, episodes: EpisodesConfig, image_size: int) -> None:
        """Bind window settings and aligned image dimensions.

        :param episodes: Validated episode and trajectory configuration.
        :type episodes: EpisodesConfig
        :param image_size: Positive aligned image side in pixels.
        :type image_size: int
        :raises TrajectoryError: If the window cannot fit inside the image.
        """
        if not 1 <= episodes.window <= image_size:
            raise TrajectoryError('Episode window must fit inside the aligned image.')
        self.episodes: EpisodesConfig = episodes
        self.image_size: int = image_size

    def boxes_for(self, sample_id: str) -> NDArray[np.int32]:
        """Generate window geometry without reading pixels.

        :param sample_id: Stable sample identifier.
        :type sample_id: str
        :returns: Int32 half-open pixel boxes with shape (T,4).
        :rtype: NDArray[np.int32]
        """
        return make_boxes(
            sample_id=sample_id,
            image_size=self.image_size,
            window=self.episodes.window,
            steps=self.episodes.steps,
            trajectory=self.episodes.trajectory,
        )

    def build(self, sample: Sample, image: NDArray[np.uint8]) -> Episode:
        """Copy selected windows and normalize their centers by image side length.

        :param sample: Selected image identity, label, and split.
        :type sample: Sample
        :param image: Aligned RGB uint8 pixels with shape (S,S,3).
        :type image: NDArray[np.uint8]
        :returns: Episode with observations (T,W,W,3), boxes (T,4), and positions (T,2).
        :rtype: Episode
        :raises ValueError: If the image dimensions or dtype are incompatible.
        """
        if image.shape != (self.image_size, self.image_size, 3) or image.dtype != np.uint8:
            raise ValueError('Episode image must be RGB uint8 with the configured square size.')
        boxes = self.boxes_for(sample_id=sample.sample_id)
        observations = np.stack(arrays=[image[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes])
        positions = (boxes[:, :2].astype(np.float32) + self.episodes.window / 2) / self.image_size
        return Episode(
            sample_id=sample.sample_id,
            split=sample.split,
            target=sample.label,
            observations=observations,
            boxes=boxes,
            positions=positions.astype(np.float32),
        )

    def trajectory_hash(self, sample_ids: Sequence[str]) -> str:
        """Hash boxes in caller-supplied episode order for comparison provenance.

        :param sample_ids: Ordered stable sample identifiers.
        :type sample_ids: Sequence[str]
        :returns: Canonical SHA-256 digest of identifiers and generated boxes.
        :rtype: str
        """
        return sha256_obj(
            obj=[
                [sample_id, self.boxes_for(sample_id=sample_id).tolist()]
                for sample_id in sample_ids
            ]
        )
