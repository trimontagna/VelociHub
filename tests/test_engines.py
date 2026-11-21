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


def test_grid_in_the_image(measurements):
    """The grid of the vector fields read in pixels placed in the image as the PIV saved it (x_px, y_px from the
    top-left corner): each displacement at its own position (the tile of the pair and its vectors)."""
    from velocihub.fields import load_vectors
    from velocihub.piv.displacements import grid_pixels

    path = os.path.join(measurements["piv"], "PIV", "vectors.npz")
    raw = np.load(path, allow_pickle=True)
    fields = load_vectors(path, pixels=True)
    x, y = grid_pixels(fields)
    saved = {(float(a), float(b)): float(u) for a, b, u in zip(raw["x_px"].ravel(), raw["y_px"].ravel(),
                                                                raw["u_px"][0].ravel())}
    shown = {(float(a), float(b)): float(u) for a, b, u in zip(x.ravel(), y.ravel(), fields.u_px[0].ravel())}
    assert shown.keys() == saved.keys()
    assert all(np.isclose(shown[k], saved[k], equal_nan=True) for k in saved)


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


def test_spicy_every_nth_map(copy):
    """SPICY fitting every 2nd map of a PTV (setting map_step): the results hold the maps 0, 2, 4... (their pairs and
    their binned cells), each fitted on the links of its own pair; the tracks keep every frame, the map k being the
    pair of the frames 2k and 2k + 1; the post-processing and its PTV figures read them."""
    pytest.importorskip("imutils")
    pytest.importorskip("spicy_vki")
    from velocihub.piv.displacements import pair_particles, read_tracks

    folder = copy("images")
    images = sorted(os.path.join(folder, "PreProcessing", n) for n in os.listdir(os.path.join(folder, "PreProcessing"))
                    if n.endswith(".tif"))
    engine = piv.get_engine("tractrac")
    run = piv.PIVRun(images, None, "tractrac", {"spacing": 16, "vid_loop": 0},
                     os.path.join(folder, engine.results_folder), info={"measurement_folder": folder},
                     regression={"method": "spicy", "settings": {"map_step": 2}})
    pairs = run.pairs
    path = run.run(report=lambda message: None)
    with np.load(path) as z:
        settings = json.loads(str(z["settings"]))
        n_points, rejected = z["regression_n_points"], z["regression_n_rejected"]
        binned = z["ptv_particles"].reshape(len(n_points), -1).sum(1)
        n_maps = z["u_px"].shape[0]
    assert n_maps == len(range(0, len(pairs), 2)) == len(settings["pairs"])
    assert settings["pairs"] == [list(p) for p in pairs[::2]] and settings["regression_map_step"] == 2
    assert settings["computed_pairs"] == len(pairs)
    # Each map fitted on the links of its own pair: those binned in its cells
    assert np.array_equal(n_points + rejected, binned)
    fields = vh.load_vectors(path, pixels=True)
    assert fields.map_step == 2 and fields.n == n_maps
    # The raw fields kept are the maps 0, 2, 4... of the same PTV without SPICY; the tracks have their frames
    alone = piv.PIVRun(images, None, "tractrac", {"spacing": 16, "vid_loop": 0}, os.path.join(folder, "PTV_alone"),
                       info={"measurement_folder": folder}).run(report=lambda message: None)
    with np.load(alone) as a, np.load(path) as z:
        assert np.allclose(z["raw_u_px"], a["u_px"][::2], equal_nan=True)
        assert np.array_equal(z["ptv_particles"], a["ptv_particles"][::2])
    T = read_tracks(fields.tracks)
    assert all(pair_particles(T, 2 * k) is not None for k in range(n_maps))
    post = vh.PostProcessing(fields, vh.PostSettings())
    check_jet(post)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    post.draw("ptv_tracks", Figure(), index=n_maps - 1)
    assert post.acceleration() is not None
