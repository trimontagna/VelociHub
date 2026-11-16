"""The PIV and PTV engines on the synthetic images (the displacement of the jet known): the results of PaIRS-UniNa and
TracTrac as the post-processing reads them, and their accuracy. Skipped when the engine isn't installed."""

import json
import os

import numpy as np
import pytest

import synthetic as s
import velocihub as vh
from velocihub import piv


def run(engine, settings, copy):
    folder = copy("images")
    images = sorted(os.path.join(folder, "PreProcessing", n) for n in os.listdir(os.path.join(folder, "PreProcessing"))
                    if n.endswith(".tif"))
    available, reason = piv.get_engine(engine).available()
    if not available:
        pytest.skip(reason)
    output = os.path.join(folder, piv.get_engine(engine).results_folder)
    vectors = piv.run_piv(images, None, engine, settings, output=output, info={"measurement_folder": folder},
                          report=lambda message: None)
    return vh.PostProcessing(vh.load_vectors(vectors, pixels=True), vh.PostSettings())


def check_jet(post, window=16):
    """The jet measured: its centre displacement along x that of the images averaged over the final window (or the
    cell) of window px, within 0.25 px; its profile across y falling off on both sides of the centre (the edges,
    where the first passes of a PIV still count, not compared to the pixel)."""
    with post.all_maps():
        mean_u = post.statistics()["mean_U"]
    y_px = s.HEIGHT - post.fields.y[:, 0]  # Rows from the top of the images (sensor space: y from the bottom)
    profile = np.nanmedian(mean_u, axis=1)
    centre = int(np.argmin(np.abs(y_px - s.HEIGHT / 2)))
    offsets = np.linspace(-window / 2, window / 2, 33)
    assert profile[centre] == pytest.approx(s.jet(s.HEIGHT / 2 + offsets).mean(), abs=0.25)
    assert np.all(np.diff(profile[:centre + 1]) > 0) and np.all(np.diff(profile[centre:]) < 0)
    assert profile[0] < profile[centre] / 2 and profile[-1] < profile[centre] / 2


def test_pairs(copy):
    post = run("pairs", {"windows": (32, 16), "overlap": 50}, copy)
    check_jet(post)
    assert post.fields.peak_ratio is not None


def test_tractrac(copy):
    pytest.importorskip("imutils")
    post = run("tractrac", {"spacing": 16, "vid_loop": 0}, copy)
    assert post.ptv() and post.fields.tracks
    check_jet(post)


def test_tractrac_spicy_fits_the_particles(copy):
    """TracTrac then SPICY: each pair is fitted on the links of its particles (without masks: exactly those binned in
    the cells of the grid, more than the cells, less those the outlier test leaves out: a few), evaluated on the grid;
    the binned cells are kept as the raw fields. The jet is still measured."""
    pytest.importorskip("imutils")
    pytest.importorskip("spicy_vki")
    folder = copy("images")
    images = sorted(os.path.join(folder, "PreProcessing", n) for n in os.listdir(os.path.join(folder, "PreProcessing"))
                    if n.endswith(".tif"))
    engine = piv.get_engine("tractrac")
    run = piv.PIVRun(images, None, "tractrac", {"spacing": 16, "vid_loop": 0},
                     os.path.join(folder, engine.results_folder), info={"measurement_folder": folder},
                     regression={"method": "spicy"})
    path = run.run(report=lambda message: None)
    with np.load(path) as z:
        settings = json.loads(str(z["settings"]))
        n_points, residual = z["regression_n_points"], z["regression_residual_px"]
        rejected = z["regression_n_rejected"]
        binned = z["ptv_particles"].reshape(len(n_points), -1).sum(1)
        cells = z["x_px"].size
        assert "raw_u_px" in z.files and "raw_info" in z.files
    assert settings["regression_data"] == "particles"
    assert np.array_equal(n_points + rejected, binned) and np.all(n_points > cells) and rejected.sum() < 0.15 * binned.sum()
    # RMS at the particles (px): the jet, and the links TracTrac gets wrong that the outlier test keeps
    assert np.all(np.isfinite(residual)) and np.median(residual) < 1
    post = vh.PostProcessing(vh.load_vectors(path, pixels=True), vh.PostSettings())
    check_jet(post)
