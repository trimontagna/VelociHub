"""The gradients of the post-processing (PostProcessing.gradient_fields): the smoothing gradient_sigma (0.5 cell by
default), and gradient_smooth_after (the differences first, one-sided next to the holes, then smoothed), on the
laminar boundary layer of the benchmark (its wall masked)."""

import numpy as np
import pytest

from velocihub.postprocessing import GRADIENT_SIGMA, PostProcessing, PostSettings, gradient_nan


def test_gradient_nan_exact_on_quadratics():
    """Central differences inside, one-sided of second order next to the holes and at the edges: exact for a
    quadratic; first order with one point on a side only; NaN in the holes and between two holes."""
    y = np.arange(10.0) * 2
    F = np.tile((3 * y**2 - y + 1)[:, None], (1, 3))
    F[4, 0] = np.nan  # A hole: its neighbours one-sided
    F[6, 1] = F[8, 1] = np.nan  # The point between two holes, and the last one after a hole: no difference
    G = gradient_nan(F, y, 0)
    exact = 6 * y - 1
    ok = np.isfinite(G)
    assert np.allclose(G[ok], np.broadcast_to(exact[:, None], F.shape)[ok])
    assert np.isnan(G[[4, 6, 7, 8, 9], [0, 1, 1, 1, 1]]).all() and ok.sum() == F.size - 5
    # Along the other axis, a descending coordinate (rows from the top)
    assert np.allclose(gradient_nan(F[:, 2][None, ::-1], y[::-1], 1), exact[None, ::-1])


def test_settings():
    s = PostSettings()
    assert s.gradient_sigma == GRADIENT_SIGMA == 0.5 and not s.gradient_smooth_after
    assert not any("gradients" in step for step in s.steps())
    assert "gradients smoothed over σ = 1.5 cells after the differences" in PostSettings(
        gradient_sigma=1.5, gradient_smooth_after=True).steps()
    with pytest.raises(ValueError):
        PostSettings(gradient_sigma=-1)


@pytest.fixture(scope="module")
def blasius(tmp_path_factory):
    from velocihub.benchmark import FLOWS
    from velocihub.benchmark.measurement import CAMERA, write_exact
    from velocihub.fields import load_vectors

    flow = FLOWS["blasius"]()
    path = write_exact(str(tmp_path_factory.mktemp("blasius")), flow, 16, [0.5, 1.5, 2.5])
    return flow, load_vectors(path, camera=CAMERA)


def test_smooth_after_differencing(blasius):
    """Both orders give the same gradients away from the wall; next to it, the default gives none, smoothed after
    the differences gives the gradient of the wall within a few %. The smoothing setting used, and in the titles."""
    from velocihub.benchmark.measurement import MM_PER_PX, PER_S

    flow, fields = blasius
    default = PostProcessing(fields, PostSettings())
    after = PostProcessing(fields, PostSettings(gradient_smooth_after=True))
    d0, d1 = default.gradient_fields()["du"] / PER_S, after.gradient_fields()["du"] / PER_S
    Y = fields.y / MM_PER_PX
    wall = Y == Y[fields.status == 0].min()  # The grid points of the flow next to the wall
    away = (Y > Y[wall][0] + 4 * 16) & (fields.status == 0)
    assert np.allclose(d0[away], d1[away], atol=1e-9)
    assert np.isnan(d0[wall]).all() and np.isfinite(d1[wall]).all()
    _, exact, _, _ = flow.gradients(fields.x[wall] / MM_PER_PX, Y[wall])
    assert np.median(d1[wall] / exact) == pytest.approx(1, abs=0.06)
    # The smoothing: none, the field's own differences; more, a flatter gradient at its peak
    none = PostProcessing(fields, PostSettings(gradient_sigma=0)).gradient_fields()["du"]
    wide = PostProcessing(fields, PostSettings(gradient_sigma=2)).gradient_fields()["du"]
    assert np.nanmax(np.abs(wide)) < np.nanmax(np.abs(d0 * PER_S)) < np.nanmax(np.abs(none))
    assert "σ = 0.5 cells after the differences" in after.smoothing_text()
    assert PostProcessing(fields, PostSettings(gradient_sigma=0)).smoothing_text() == ""
