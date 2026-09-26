"""Compound-eye lattices and acceptance-angle sampling (T39)."""

import numpy as np
import pytest

from flystate.diagnostics.eye import (
    ACCEPTANCE_DEG,
    LIMITED,
    NOT_VIABLE,
    PIXEL_REFERENCE,
    SPACING_DEG,
    VIABLE,
    classify_retention,
    eye_cases,
    hex_lattice,
    retained_fractions,
    sampling_matrix,
    square_lattice,
)

SIZE: int = 32
WIDTH: float = 60.0


def test_hex_lattice_has_the_interommatidial_spacing() -> None:
    """Place every point inside the field with nearest neighbours one spacing apart."""
    points = hex_lattice(width_deg=WIDTH, spacing_deg=SPACING_DEG)
    assert np.all(np.abs(points) <= WIDTH / 2 + 1e-9)
    gaps = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(gaps, np.inf)
    assert np.allclose(gaps.min(axis=1), SPACING_DEG)
    expected = (WIDTH / SPACING_DEG) ** 2 * 2 / np.sqrt(3)
    assert 0.85 * expected < len(points) < 1.25 * expected


def test_square_lattice_matches_the_requested_count() -> None:
    """Give an equal-count control lattice within one row of the target."""
    points = square_lattice(count=200, width_deg=WIDTH)
    assert abs(len(points) - 200) <= 2 * np.sqrt(200)
    assert np.all(np.abs(points) < WIDTH / 2)


@pytest.mark.parametrize(argnames='fwhm', argvalues=[ACCEPTANCE_DEG, None])
def test_sampling_preserves_constant_images_and_locates_a_point(fwhm: float | None) -> None:
    """Keep a uniform image uniform, and centre a single bright pixel's response on its point.

    :param fwhm: Acceptance angle, degrees, or none for point sampling.
    :type fwhm: Optional[float]
    """
    points = hex_lattice(width_deg=WIDTH, spacing_deg=SPACING_DEG)
    weights = sampling_matrix(points_deg=points, size=SIZE, width_deg=WIDTH, fwhm_deg=fwhm)
    assert weights.shape == (len(points), SIZE * SIZE)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)
    constant = np.full(shape=SIZE * SIZE, fill_value=0.7, dtype=np.float32)
    assert np.allclose(weights @ constant, 0.7, atol=1e-5)
    target = int(np.argmin(np.linalg.norm(points, axis=1)))
    image = np.zeros(shape=(SIZE, SIZE), dtype=np.float32)
    centre = SIZE // 2
    image[centre - 1 : centre + 1, centre - 1 : centre + 1] = 1.0
    response = weights @ image.ravel()
    assert int(np.argmax(response)) == target


def test_wider_acceptance_blurs_more() -> None:
    """Spread one bright pixel over more ommatidia when the acceptance angle grows."""
    points = hex_lattice(width_deg=WIDTH, spacing_deg=SPACING_DEG)
    image = np.zeros(shape=SIZE * SIZE, dtype=np.float32)
    image[(SIZE // 2) * SIZE + SIZE // 2] = 1.0
    spread = []
    for fwhm in (4.0, 12.0):
        response = sampling_matrix(points, SIZE, WIDTH, fwhm) @ image
        spread.append(int(np.sum(response > 0.01 * response.max())))
    assert spread[1] > spread[0]


def test_cases_cover_every_width_and_control() -> None:
    """Preregister six cases per face width, with distinct names."""
    cases = eye_cases()
    assert len(cases) == 24
    assert {case.width for case in cases.values()} == {30, 60, 90, 120}
    assert all(name.isidentifier() for name in cases)


def test_retained_fraction_and_frozen_thresholds() -> None:
    """Compute the share of above-chance accuracy and read it with the frozen thresholds."""
    scores = {f'eye/{name}': {'accuracy': 0.21, 'chance': 0.01} for name in eye_cases()}
    scores[f'eye/{PIXEL_REFERENCE}'] = {'accuracy': 0.41, 'chance': 0.01}
    retained = retained_fractions(scores=scores, recording_name='eye')
    assert retained['hex_luma_w90'] == pytest.approx(0.5)
    assert [classify_retention(fraction=f) for f in (0.5, 0.49, 0.25, 0.24)] == [
        VIABLE,
        LIMITED,
        LIMITED,
        NOT_VIABLE,
    ]
    scores[f'eye/{PIXEL_REFERENCE}'] = {'accuracy': 0.01, 'chance': 0.01}
    with pytest.raises(expected_exception=ValueError, match='not above chance'):
        retained_fractions(scores=scores, recording_name='eye')
