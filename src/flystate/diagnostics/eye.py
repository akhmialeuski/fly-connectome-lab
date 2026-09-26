"""Face identity that survives Drosophila compound-eye sampling (T39).

The eye is modelled as a lattice of ommatidia with a fixed interommatidial angle, each averaging
the image under a Gaussian acceptance function. The face is assumed to span a given visual angle.
Samples are written like a recording, one array per case, so the T35 evaluation scores them with
the standard readout. A frozen rule then decides whether faces remain a target.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flystate.datasets.preprocess import prepare_dataset
from flystate.diagnostics.artifacts import attempt, verify_attempt_inventory
from flystate.diagnostics.confirmation import FINAL_PREFIX, ISSUE_KEY, KIND, SAMPLE_IDS, SCORES
from flystate.diagnostics.drive_sweep import PARAMETERS, REPORT_FILE, RESPONSES_FILE
from flystate.diagnostics.input_access import _read_json
from flystate.diagnostics.temporal import _write_arrays
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.hashing import sha256_file
from flystate.settings import Paths, output_path
from flystate.storage.json import write_json

ISSUE: int = 78
# Interommatidial angle of the Drosophila eye, degrees (about 4.5 degrees in the literature cited
# by the T39 protocol).
SPACING_DEG: float = 4.5
# Acceptance angle as a full width at half maximum, degrees: primary value and sensitivity bounds
# of the 7.7 to 9.5 degree range.
ACCEPTANCE_DEG: float = 8.6
ACCEPTANCE_RANGE_DEG: tuple[float, float] = (7.7, 9.5)
WIDTHS_DEG: tuple[int, ...] = (30, 60, 90, 120)
DECISION_WIDTH_DEG: int = 90
VIABLE_FRACTION: float = 0.5
LIMITED_FRACTION: float = 0.25
# ITU-R BT.601 luma weights, a broadband stand-in for the R1 to R6 photoreceptors.
LUMA: tuple[float, float, float] = (0.299, 0.587, 0.114)
FWHM_TO_SIGMA: float = 2.0 * np.sqrt(2.0 * np.log(2.0))
TRUNCATION_SIGMAS: float = 3.0
PIXEL_REFERENCE: str = 'pixels_full'
PRIMARY: str = 'hex_luma'
ACCURACY: str = 'accuracy'
CHANCE: str = 'chance'
COMPLETED: str = 'completed'
VIABLE: str = 'viable'
LIMITED: str = 'limited'
NOT_VIABLE: str = 'not viable'
HEX: str = 'hex'
SQUARE: str = 'square'


def hex_lattice(width_deg: float, spacing_deg: float) -> NDArray[np.float64]:
    """Return hexagonal lattice points covering a square field centred on 0.

    :param width_deg: Side of the square field, degrees.
    :type width_deg: float
    :param spacing_deg: Distance between neighbouring points, degrees.
    :type spacing_deg: float
    :returns: Point coordinates (x, y), shape (K,2), degrees.
    :rtype: NDArray[np.float64]
    """
    half = width_deg / 2
    row_step = spacing_deg * np.sqrt(3.0) / 2
    points = []
    for row in range(-int(half // row_step), int(half // row_step) + 1):
        offset = spacing_deg / 2 if row % 2 else 0.0
        y = row * row_step
        for column in range(-int(half // spacing_deg) - 1, int(half // spacing_deg) + 2):
            x = column * spacing_deg + offset
            if abs(x) <= half and abs(y) <= half:
                points.append((x, y))
    return np.asarray(points, dtype=np.float64)


def square_lattice(count: int, width_deg: float) -> NDArray[np.float64]:
    """Return a centred square lattice with about ``count`` points over the same field.

    :param count: Target number of points.
    :type count: int
    :param width_deg: Side of the square field, degrees.
    :type width_deg: float
    :returns: Point coordinates (x, y), shape (n*n,2), degrees.
    :rtype: NDArray[np.float64]
    """
    side = max(1, round(np.sqrt(count)))
    step = width_deg / side
    axis = (np.arange(side) + 0.5) * step - width_deg / 2
    x, y = np.meshgrid(axis, axis)
    return np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float64)


def sampling_matrix(
    points_deg: NDArray[np.float64], size: int, width_deg: float, fwhm_deg: float | None
) -> NDArray[np.float32]:
    """Build the weights that turn a square image into ommatidial samples.

    Each row averages the image under a Gaussian of the given full width at half maximum,
    truncated at three standard deviations and normalized to sum 1. Without a width, each row
    takes the single nearest pixel.

    :param points_deg: Ommatidial directions (x, y), shape (K,2), degrees.
    :type points_deg: NDArray[np.float64]
    :param size: Image side, pixels.
    :type size: int
    :param width_deg: Visual angle spanned by the image side, degrees.
    :type width_deg: float
    :param fwhm_deg: Acceptance angle, degrees, or none for point sampling.
    :type fwhm_deg: Optional[float]
    :returns: Weight matrix, shape (K, size*size), float32.
    :rtype: NDArray[np.float32]
    """
    centres = (np.arange(size) + 0.5) * width_deg / size - width_deg / 2
    px, py = np.meshgrid(centres, centres)
    grid = np.stack([px.ravel(), py.ravel()], axis=1)
    weights = np.zeros(shape=(len(points_deg), size * size), dtype=np.float64)
    for row, point in enumerate(points_deg):
        distance2 = np.sum((grid - point) ** 2, axis=1)
        if fwhm_deg is None:
            weights[row, int(np.argmin(distance2))] = 1.0
            continue
        sigma = fwhm_deg / FWHM_TO_SIGMA
        kernel = np.exp(-distance2 / (2 * sigma**2))
        kernel[distance2 > (TRUNCATION_SIGMAS * sigma) ** 2] = 0.0
        weights[row] = kernel / kernel.sum()
    return weights.astype(np.float32)


@dataclass(frozen=True)
class EyeCase:
    """One preregistered sampling of the face by the eye.

    :param width: Visual angle spanned by the face, degrees.
    :param lattice: ``hex`` for the ommatidial lattice or ``square`` for the equal-count control.
    :param fwhm: Acceptance angle, degrees, or none for point sampling.
    :param rgb: Whether the three colour channels are sampled instead of luminance.
    """

    width: int
    lattice: str
    fwhm: float | None
    rgb: bool


def eye_cases() -> dict[str, EyeCase]:
    """Return every preregistered case by name.

    :returns: Case name to specification.
    :rtype: dict[str, EyeCase]
    """
    cases: dict[str, EyeCase] = {}
    for width in WIDTHS_DEG:
        tag = f'w{width}'
        cases[f'{PRIMARY}_{tag}'] = EyeCase(width, HEX, ACCEPTANCE_DEG, rgb=False)
        for bound in ACCEPTANCE_RANGE_DEG:
            name = f'{PRIMARY}_{tag}_a{str(bound).replace(".", "p")}'
            cases[name] = EyeCase(width, HEX, bound, rgb=False)
        cases[f'hex_rgb_{tag}'] = EyeCase(width, HEX, ACCEPTANCE_DEG, rgb=True)
        cases[f'square_luma_{tag}'] = EyeCase(width, SQUARE, ACCEPTANCE_DEG, rgb=False)
        cases[f'hex_point_{tag}'] = EyeCase(width, HEX, None, rgb=False)
    return cases


def record_eye(cfg: ExperimentConfig, paths: Paths, output: Path) -> dict[str, Any]:
    """Write the eye samples of every photograph for every case, plus the pixel reference.

    :param cfg: Cohort configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home.
    :type paths: Paths
    :param output: New immutable attempt directory.
    :type output: Path
    :returns: Recording summary with the number of samples per case.
    :rtype: dict[str, Any]
    """
    cases = eye_cases()
    parameters = {
        KIND: 'eye_record',
        ISSUE_KEY: ISSUE,
        'config_sha256': config_hash(cfg=cfg),
        'spacing_deg': SPACING_DEG,
        'luma_weights': list(LUMA),
        'cases': {name: asdict(case) for name, case in cases.items()},
        'trainable_fly_parameters': [],
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        prepared = prepare_dataset(cfg=cfg, paths=paths)
        images = np.asarray(prepared.images, dtype=np.float32) / np.float32(255.0)
        count, size = images.shape[0], images.shape[1]
        flat = images.reshape(count, size * size, 3)
        luma = flat @ np.asarray(LUMA, dtype=np.float32)
        arrays: dict[str, NDArray[np.float32]] = {
            f'{FINAL_PREFIX}{PIXEL_REFERENCE}': images.reshape(count, -1)
        }
        samples: dict[str, int] = {}
        lattices: dict[str, list[list[float]]] = {}
        for name, case in cases.items():
            points = hex_lattice(width_deg=case.width, spacing_deg=SPACING_DEG)
            if case.lattice == SQUARE:
                points = square_lattice(count=len(points), width_deg=case.width)
            weights = sampling_matrix(
                points_deg=points, size=size, width_deg=case.width, fwhm_deg=case.fwhm
            )
            if case.rgb:
                values = np.einsum('kp,npc->nkc', weights, flat).reshape(count, -1)
            else:
                values = luma @ weights.T
            arrays[f'{FINAL_PREFIX}{name}'] = values.astype(np.float32)
            samples[name] = len(points)
            lattices[name] = points.round(6).tolist()
        _write_arrays(path=directory / RESPONSES_FILE, arrays=arrays)
        write_json(path=directory / 'lattices.json', value=lattices)
        summary = {
            PARAMETERS: parameters,
            'episodes': count,
            'image_size': size,
            'dataset_fingerprint': prepared.fingerprint,
            SAMPLE_IDS: [sample.sample_id for sample in prepared.samples],
            'ommatidia_per_case': samples,
            'responses_sha256': sha256_file(path=directory / RESPONSES_FILE),
        }
        write_json(path=directory / REPORT_FILE, value=summary)
    return summary


def retained_fractions(scores: dict[str, Any], recording_name: str) -> dict[str, float]:
    """Return each eye case's share of the pixel reference's above-chance accuracy.

    :param scores: Evaluation scores by case ``<recording>/<case>``.
    :type scores: dict[str, Any]
    :param recording_name: Name the evaluation gave the eye recording.
    :type recording_name: str
    :returns: Case name to (A_eye - chance) / (A_pixels - chance).
    :rtype: dict[str, float]
    :raises ValueError: If the pixel reference is not above chance, leaving the share undefined.
    """
    reference = scores[f'{recording_name}/{PIXEL_REFERENCE}']
    chance = reference[CHANCE]
    if reference[ACCURACY] <= chance:
        raise ValueError('The pixel reference is not above chance, so no share can be defined.')
    return {
        name: (scores[f'{recording_name}/{name}'][ACCURACY] - chance)
        / (reference[ACCURACY] - chance)
        for name in eye_cases()
    }


def classify_retention(fraction: float) -> str:
    """Apply the frozen T39 thresholds to the retained fraction at the decision width.

    :param fraction: Retained fraction of above-chance accuracy.
    :type fraction: float
    :returns: ``viable``, ``limited`` or ``not viable``.
    :rtype: str
    """
    if fraction >= VIABLE_FRACTION:
        return VIABLE
    return LIMITED if fraction >= LIMITED_FRACTION else NOT_VIABLE


def decide(
    cfg: ExperimentConfig, paths: Paths, output: Path, evaluation: Path, recording_name: str
) -> dict[str, Any]:
    """Apply the frozen T39 rule to one evaluated cohort.

    :param cfg: Cohort configuration, for the attempt record.
    :type cfg: ExperimentConfig
    :param paths: Working data home.
    :type paths: Paths
    :param output: New immutable attempt directory.
    :type output: Path
    :param evaluation: Completed evaluation of the eye recording.
    :type evaluation: Path
    :param recording_name: Name the evaluation gave the eye recording.
    :type recording_name: str
    :returns: Retained fraction per width and case, and the decision at the frozen width.
    :rtype: dict[str, Any]
    :raises ValueError: If the evaluation is not completed.
    """
    parameters = {
        KIND: 'eye_decision',
        ISSUE_KEY: ISSUE,
        'evaluation': str(evaluation),
        'decision_width_deg': DECISION_WIDTH_DEG,
        'thresholds': {VIABLE: VIABLE_FRACTION, LIMITED: LIMITED_FRACTION},
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        if verify_attempt_inventory(directory=evaluation, paths=paths)['status'] != COMPLETED:
            raise ValueError(f'The evaluation is not completed: {evaluation}.')
        scores = _read_json(path=output_path(path=evaluation, paths=paths) / REPORT_FILE)[SCORES]
        reference = scores[f'{recording_name}/{PIXEL_REFERENCE}']
        chance = reference[CHANCE]
        retained = retained_fractions(scores=scores, recording_name=recording_name)
        primary = retained[f'{PRIMARY}_w{DECISION_WIDTH_DEG}']
        decision = classify_retention(fraction=primary)
        result = {
            PARAMETERS: parameters,
            'pixel_reference_accuracy': reference[ACCURACY],
            CHANCE: chance,
            'retained_fraction': retained,
            'primary_retained_fraction': primary,
            'decision': decision,
        }
        write_json(path=directory / REPORT_FILE, value=result)
    return result
