"""The PIV and PTV engines on the synthetic images (the displacement of the jet known): the results of PaIRS-UniNa and
TracTrac as the post-processing reads them, and their accuracy. Skipped when the engine isn't installed."""

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
    y_px = s.HEIGHT / 2 - post.fields.y[:, 0]  # Rows from the top of the images
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
