"""The fit of a map by SPICY (piv/spicy_worker.py), without engine: on the grid vectors (PIV) and on scattered particles
(PTV), a smooth known field recovered on the grid; the particles in masked cells left out; the wrong links left out
by the outlier test."""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("spicy_vki")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "velocihub", "piv"))
import spicy_worker as sw  # noqa: E402

WIDTH, HEIGHT, SPACING = 320, 192, 16
SETTINGS = {"basis": "gauss", "n_k": [4], "method": "clustering", "r_min": 0.01, "r_max": 0.3, "eps_l": 0.7,
            "alpha_div": 0, "n_hb": 0, "k_cond": 1e12, "outlier_test": 0}


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
    fu, fv, finfo, residual, n_rbf, n_points, rejected = out
    tu, tv = field(xpx, ypx)
    assert n_points == xpx.size - 1 and finfo.all() and rejected == 0
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
    fu, fv, finfo, residual, n_rbf, n_points, rejected = out
    tu, tv = field(xpx, ypx)
    assert n_points == 3000 and finfo.all() and rejected == 0
    for err in (np.abs(fu - tu), np.abs(fv - tv)):
        assert np.median(err) < 0.05 and err.max() < 0.25
    assert residual == pytest.approx(0.05 * np.sqrt(2), abs=0.02)  # the noise of the particles


def test_link_outliers():
    """The normalized median test of the links: the wrong ones (random displacements) found, almost no good one,
    even across the steep gradient of the jet; a link with too few neighbours kept."""
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0, WIDTH, 2000), rng.uniform(0, HEIGHT, 2000)
    pu, pv = field(x, y)
    pu, pv = pu + rng.normal(0, 0.1, x.size), pv + rng.normal(0, 0.1, x.size)
    wrong = rng.random(x.size) < 0.03
    pu[wrong], pv[wrong] = rng.uniform(-8, 8, wrong.sum()), rng.uniform(-8, 8, wrong.sum())
    bad = wrong & (np.hypot(pu - field(x, y)[0], pv - field(x, y)[1]) > 1)
    rejected = sw.link_outliers(x, y, pu, pv, 3.0, 10, 3 * SPACING)
    assert rejected[bad].mean() > 0.95 and rejected[~wrong].mean() < 0.005
    # Isolated: fewer than 4 neighbours within the radius, kept whatever its displacement
    lone = sw.link_outliers(np.array([0.0, 1, 2, 500]), np.zeros(4), np.array([0.0, 0, 0, 9]), np.zeros(4), 3.0, 10, 48)
    assert not lone.any()


def test_particles_with_wrong_links():
    """SPICY on links with 3 % of them wrong: as accurate as without, with the outlier test; much worse without."""
    rng = np.random.default_rng(3)
    xpx, ypx = grid()
    x, y = rng.uniform(0, WIDTH, 3000), rng.uniform(0, HEIGHT, 3000)
    pu, pv = field(x, y)
    pu, pv = pu + rng.normal(0, 0.05, x.size), pv + rng.normal(0, 0.05, x.size)
    wrong = rng.random(x.size) < 0.03
    pu[wrong], pv[wrong] = rng.uniform(-8, 8, wrong.sum()), rng.uniform(-8, 8, wrong.sum())
    nan = np.full(xpx.shape, np.nan, np.float32)
    tu, tv = field(xpx, ypx)
    errors = {}
    for test in (0, 3.0):
        out = sw.fit(nan, nan, np.zeros(xpx.shape, np.int16), xpx, ypx, np.ones(xpx.shape, bool), WIDTH,
                     {**SETTINGS, "outlier_test": test, "outlier_neighbours": 10}, (x, y, pu, pv))
        errors[test] = np.sqrt(np.mean((out[0] - tu) ** 2 + (out[1] - tv) ** 2))
        if test:
            assert out[6] >= 0.9 * wrong.sum() and out[5] == x.size - out[6]
    assert errors[3.0] < 0.06 and errors[0] > 3 * errors[3.0]


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


def test_points_per_rbf_auto(monkeypatch):
    """The points per RBF default to "auto": 3 for the vectors of a PIV, 4 for the particles of a PTV (an explicit
    value is kept for both); the worker gets the numbers."""
    from velocihub.piv import regression as rg

    checked = rg.check_regression({"method": "spicy"})
    assert checked["settings"]["n_k"] == "auto"
    assert rg.regression_text(checked)["settings"]["n_k"] == "auto"
    assert rg.check_regression({"method": "spicy", "settings": {"n_k": "AUTO"}})["settings"]["n_k"] == "auto"
    assert rg.points_per_rbf(checked["settings"], particles=False) == (3,)
    assert rg.points_per_rbf(checked["settings"], particles=True) == (4,)
    explicit = rg.check_regression({"method": "spicy", "settings": {"n_k": "6 12"}})
    assert rg.points_per_rbf(explicit["settings"], False) == rg.points_per_rbf(explicit["settings"], True) == (6, 12)
    with pytest.raises(ValueError):
        rg.check_regression({"method": "spicy", "settings": {"n_k": "0"}})
    jobs = []
    monkeypatch.setattr(rg, "spicy_available", lambda: (True, ""))
    monkeypatch.setattr(rg, "WorkerProcess", lambda exe, script, job, work: jobs.append(job))
    for particles in (False, True):
        rg.start_regression("work", checked, [], 320, "key", particles=particles)
    assert [job["settings"]["n_k"] for job in jobs] == [(3,), (4,)]
    assert jobs[0]["key"] != jobs[1]["key"]
