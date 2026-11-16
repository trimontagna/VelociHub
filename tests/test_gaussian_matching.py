"""Gaussian kernel matching (preprocessing.gaussian_matching): the image becomes its correlation map with a Gaussian
particle image."""

import numpy as np
import pytest

from velocihub.preprocessing import ContrastSettings, PODBackgroundRemoval, gaussian_matching, gaussian_template


def particle_image(shape, centres, amplitudes, sigma, background=0.0):
    rows, cols = np.mgrid[: shape[0], : shape[1]].astype(float)
    image = np.full(shape, background)
    for (r, c), a in zip(centres, amplitudes):
        image += a * np.exp(-((rows - r) ** 2 + (cols - c) ** 2) / (2 * sigma**2))
    return image


def test_template_zero_mean_and_size():
    for sigma, side in ((0.5, 3), (1.0, 5), (1.3, 7), (2.0, 9)):
        T = gaussian_template(sigma)
        assert T.shape == (side, side)
        assert abs(T.sum()) < 1e-12


def test_particles_match_whatever_their_brightness():
    """A particle of the template's size gives rho = 1 at its centre, faint or bright; the flat background 0."""
    sigma = 1.0
    A = particle_image((64, 64), [(20, 20), (40, 44)], [12.0, 200.0], sigma, background=10.0)
    R = gaussian_matching(A.astype(np.float32), sigma, min_contrast=0.5)
    assert R.dtype == np.float32
    assert R[20, 20] == pytest.approx(1.0, abs=1e-3)
    assert R[40, 44] == pytest.approx(1.0, abs=1e-3)
    assert R[5:10, 50:60].max() < 1e-6  # flat background (round-off of the zero-mean template)
    assert R.min() >= 0 and R.max() <= 1


def test_integer_image_fills_its_range():
    A = particle_image((48, 48), [(24, 24)], [100.0], 1.0).round().astype(np.uint8)
    R = gaussian_matching(A, 1.0, min_contrast=2)
    assert R.dtype == np.uint8
    assert R[24, 24] >= 250
    R16 = gaussian_matching(A.astype(np.uint16) * 256, 1.0, min_contrast=2 * 256)
    assert R16.dtype == np.uint16 and R16[24, 24] >= 64000


def test_minimum_contrast_dims_faint_matches():
    """The floor of the local standard deviation lowers the faint particles, not the bright ones."""
    A = particle_image((48, 64), [(24, 16), (24, 48)], [3.0, 150.0], 1.0)
    low, high = gaussian_matching(A, 1.0, 0.1), gaussian_matching(A, 1.0, 10.0)
    assert low[24, 16] > 0.99 and high[24, 16] < 0.2
    assert high[24, 48] > 0.99


def test_wrong_size_matches_less():
    A = particle_image((48, 48), [(24, 24)], [100.0], 2.5)
    assert gaussian_matching(A, 1.0, 0.5)[24, 24] < gaussian_matching(A, 2.5, 0.5)[24, 24]


def test_masks_are_ignored():
    """Masked pixels give 0 and the edges of the masks don't match (they are filled with the level around)."""
    A = particle_image((64, 64), [(32, 16)], [80.0], 1.0, background=20.0)
    A[:, 40:] = 250.0  # bright masked region (e.g. a reflection)
    valid = np.ones(A.shape, bool)
    valid[:, 40:] = False
    R = gaussian_matching(A, 1.0, 1.0, valid)
    assert R[:, 40:].max() == 0
    assert R[:, 34:40].max() < 0.05  # along the mask
    assert R[32, 16] == pytest.approx(1.0, abs=1e-3)


def test_settings_and_recipe(tmp_path):
    c = ContrastSettings(filters=("capping", "gaussian"), gm_sigma=0.8, gm_contrast=3)
    assert "Gaussian matching (σ 0.8 px, min. contrast 3)" in c.describe()
    with pytest.raises(ValueError):
        ContrastSettings(filters=("gaussian",), gm_sigma=0)
    with pytest.raises(ValueError):
        ContrastSettings(filters=("gaussian",), gm_contrast=-1)
    session = PODBackgroundRemoval()
    session.set_contrast(["gaussian"], stage="post", gm_sigma=1.0, gm_contrast=2.0)
    A = particle_image((40, 40), [(20, 20)], [60.0], 1.0).round().astype(np.uint8)
    assert np.array_equal(session.enhance_contrast(A), gaussian_matching(A, 1.0, 2.0))
