"""Training-only landmark templates and reflection-free similarity alignment."""

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from flystate.experiments.config import PreprocessConfig


def similarity_transform(src: NDArray, dst: NDArray) -> NDArray[np.float64]:
    """Fit the least-squares uniform scale, rotation, and translation without reflection.

    :param src: Finite source pixel coordinates of shape (5,2).
    :type src: NDArray
    :param dst: Finite target pixel coordinates of shape (5,2).
    :type dst: NDArray
    :returns: Float64 matrix A (2,3), mapping source columns by A[:,:2]@src.T+A[:,2:].
    :rtype: NDArray[np.float64]
    :raises ValueError: If landmarks are malformed, nonfinite, or geometrically degenerate.
    """
    if (
        src.shape != (5, 2)
        or dst.shape != (5, 2)
        or not (np.isfinite(src).all() and np.isfinite(dst).all())
    ):
        raise ValueError('Similarity landmarks must be finite arrays with shape (5,2).')
    source = np.asarray(a=src, dtype=np.float64)
    target = np.asarray(a=dst, dtype=np.float64)
    src_mean, dst_mean = source.mean(axis=0), target.mean(axis=0)
    source = source - src_mean
    target = target - dst_mean
    if np.linalg.matrix_rank(source) < 2 or np.linalg.matrix_rank(target) < 2:
        raise ValueError('Similarity landmarks must span two spatial dimensions.')
    covariance = target.T @ source / len(source)
    left, singular, right = np.linalg.svd(a=covariance)
    signs = np.array(object=[1.0, np.sign(np.linalg.det(a=left @ right))])
    rotation = (left * signs) @ right
    scale = np.sum(a=singular * signs) / np.mean(a=np.sum(a=source**2, axis=1))
    if scale <= np.finfo(np.float64).eps:
        raise ValueError('Similarity transform has a nonpositive or degenerate scale.')
    linear = scale * rotation
    return np.column_stack(tup=(linear, dst_mean - linear @ src_mean))


def canonical_template(train_landmarks: NDArray, pre: PreprocessConfig) -> NDArray[np.float64]:
    """Normalize mean training landmarks by interocular distance and eye midpoint.

    :param train_landmarks: Finite training-only pixel coordinates, shape (N,5,2), N>0.
    :type train_landmarks: NDArray
    :param pre: Output eye midpoint and interocular distance in pixels.
    :type pre: PreprocessConfig
    :returns: Float64 canonical target landmarks of shape (5,2).
    :rtype: NDArray[np.float64]
    :raises ValueError: If input is empty, invalid, or has coincident mean eyes.
    """
    if (
        train_landmarks.ndim != 3
        or train_landmarks.shape[1:] != (5, 2)
        or (len(train_landmarks) == 0 or not np.isfinite(train_landmarks).all())
    ):
        raise ValueError('Training landmarks must be finite with nonempty shape (N,5,2).')
    mean = np.mean(a=train_landmarks, axis=0, dtype=np.float64)
    eye_center = (mean[0] + mean[1]) / 2
    distance = np.linalg.norm(x=mean[1] - mean[0])
    if distance <= np.finfo(np.float64).eps:
        raise ValueError('Mean training eye landmarks must be distinct.')
    return (mean - eye_center) * (pre.interocular / distance) + np.asarray(a=pre.eye_center)


def align_face(
    image: NDArray[np.uint8], landmarks: NDArray, template: NDArray, size: int
) -> NDArray[np.uint8]:
    """Warp RGB pixels with the inverse fitted similarity map and bilinear sampling.

    :param image: Source RGB uint8 pixels, shape (H,W,3).
    :type image: NDArray[np.uint8]
    :param landmarks: Source pixel landmarks, shape (5,2).
    :type landmarks: NDArray
    :param template: Target pixel landmarks, shape (5,2).
    :type template: NDArray
    :param size: Positive output image side in pixels.
    :type size: int
    :returns: Aligned uint8 RGB pixels, shape (size,size,3), with black outside the source.
    :rtype: NDArray[np.uint8]
    :raises ValueError: If image shape, dtype, size, or landmarks are invalid.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or size < 1:
        raise ValueError('Alignment requires RGB uint8 pixels and a positive output size.')
    transform = similarity_transform(src=landmarks, dst=template)
    inverse = np.linalg.inv(a=np.vstack(tup=(transform, [0, 0, 1])))[:2]
    warped = Image.fromarray(obj=image).transform(
        size=(size, size),
        method=Image.Transform.AFFINE,
        data=tuple(inverse.ravel()),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )
    return np.array(object=warped, dtype=np.uint8)


def map_landmarks(landmarks: NDArray, transform: NDArray) -> NDArray[np.float64]:
    """Apply a fitted affine matrix to pixel landmark rows.

    :param landmarks: Finite pixel coordinates, shape (5,2).
    :type landmarks: NDArray
    :param transform: Finite affine matrix, shape (2,3).
    :type transform: NDArray
    :returns: Float64 transformed coordinates, shape (5,2).
    :rtype: NDArray[np.float64]
    :raises ValueError: If matrix or point dimensions and values are invalid.
    """
    if (
        landmarks.shape != (5, 2)
        or transform.shape != (2, 3)
        or not (np.isfinite(landmarks).all() and np.isfinite(transform).all())
    ):
        raise ValueError('Expected finite landmarks (5,2) and affine transform (2,3).')
    return np.asarray(a=landmarks @ transform[:, :2].T + transform[:, 2], dtype=np.float64)
