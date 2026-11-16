"""The fit of a map by SPICY (piv/spicy_worker.py), without engine: on the grid vectors (PIV) and on scattered particles
(PTV), a smooth known field recovered on the grid; the particles in masked cells left out."""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("spicy_vki")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "velocihub", "piv"))
import spicy_worker as sw  # noqa: E402

WIDTH, HEIGHT, SPACING = 320, 192, 16
SETTINGS = {"basis": "gauss", "n_k": [4], "method": "clustering", "r_min": 0.01, "r_max": 0.3, "eps_l": 0.7,
            "alpha_div": 0, "n_hb": 0, "k_cond": 1e12}


def field(x, y):
    """A smooth displacement field (px): a Gaussian jet along x plus a slow vertical wave."""
    return 4 * np.exp(-((y - HEIGHT / 2) ** 2) / (2 * 25.0**2)), 0.5 * np.sin(2 * np.pi * x / WIDTH)


def grid():
    xs, ys = np.arange(SPACING, WIDTH - SPACING + 0.5, SPACING), np.arange(SPACING, HEIGHT - SPACING + 0.5, SPACING)
    return np.meshgrid(xs.astype(float), ys.astype(float))


def test_grid_vectors():
    xpx, ypx = grid()
    u, v = field(xpx, ypx)
    info = np.ones(xpx.shape, np.int16)
    info[3, 4] = 0  # an invalid vector: filled
    u[3, 4] = v[3, 4] = np.nan
    out = sw.fit(u.astype(np.float32), v.astype(np.float32), info, xpx, ypx, np.ones(xpx.shape, bool), WIDTH, SETTINGS)
    fu, fv, finfo, residual, n_rbf, n_points = out
    tu, tv = field(xpx, ypx)
    assert n_points == xpx.size - 1 and finfo.all()
    # Close everywhere, closest inside (the corners of the grid have fewer data around them)
    for err in (np.abs(fu - tu), np.abs(fv - tv)):
        assert np.median(err) < 0.05 and err.max() < 0.25
    assert residual < 0.1


def test_particles():
    rng = np.random.default_rng(1)
    xpx, ypx = grid()
    x, y = rng.uniform(0, WIDTH, 3000), rng.uniform(0, HEIGHT, 3000)
    pu, pv = field(x, y)
    pu, pv = pu + rng.normal(0, 0.05, x.size), pv + rng.normal(0, 0.05, x.size)
    nan = np.full(xpx.shape, np.nan, np.float32)
    out = sw.fit(nan, nan, np.zeros(xpx.shape, np.int16), xpx, ypx, np.ones(xpx.shape, bool), WIDTH, SETTINGS,
                 (x, y, pu, pv))
    fu, fv, finfo, residual, n_rbf, n_points = out
    tu, tv = field(xpx, ypx)
    assert n_points == 3000 and finfo.all()
    for err in (np.abs(fu - tu), np.abs(fv - tv)):
        assert np.median(err) < 0.05 and err.max() < 0.25
    assert residual == pytest.approx(0.05 * np.sqrt(2), abs=0.02)  # the noise of the particles


def test_links_masked_cells_left_out(tmp_path):
    xpx, ypx = grid()
    status = np.zeros(xpx.shape, int)
    status[:, :3] = 2  # masked on the left
    k = np.array([0, 0, 0, 1], np.int32)
    x = np.array([20.0, 200.0, -30.0, 100.0], np.float32)  # masked cell, flow, outside the grid; pair 1
    y = np.array([50.0, 50.0, 50.0, 60.0], np.float32)
    np.savez(tmp_path / "links.npz", k=k, x=x, y=y, u=np.ones(4, np.float32), v=np.zeros(4, np.float32))
    links = sw.Links(str(tmp_path), xpx, ypx, status)
    assert links.of(0)[0].tolist() == [200.0] and links.of(1)[0].tolist() == [100.0] and links.of(2)[0].size == 0


def test_links_missing(tmp_path):
    xpx, ypx = grid()
    with pytest.raises(RuntimeError, match="run the PTV again"):
        sw.Links(str(tmp_path), xpx, ypx, np.zeros(xpx.shape, int))
