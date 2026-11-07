"""Synthetic measurements for the tests: small particle images of a jet (its displacement known), their POD and
pre-processing made by VelociHub itself, and PIV or PTV results written as the engines write them (PIVRun.vectors, the
arrays of tracks.npz), in a few seconds, without PaIRS-UniNa nor TracTrac."""

import json
import os
import shutil

import numpy as np
from PIL import Image

from velocihub import piv
from velocihub.preprocessing import PREPROCESSING_FOLDER, PODBackgroundRemoval

WIDTH, HEIGHT, FRAMES = 160, 96, 16
PARTICLES = 350
# The jet: displacement along x of the frames (px per frame), Gaussian across y, centred in the images
JET_PEAK, JET_WIDTH = 4.0, 14.0
# The grid of the vector fields (px): every SPACING px from SPACING to the size minus SPACING
SPACING = 16
MODES, MODES_REMOVED = 3, 2
CAMERA = piv.Camera(fov_width_mm=80.0, x_offset_mm=0.0, frame_rate_hz=100.0)


def jet(y):
    """Displacement along x (px per frame) at the rows y (px from the top)."""
    return JET_PEAK * np.exp(-((np.asarray(y, float) - HEIGHT / 2) ** 2) / (2 * JET_WIDTH ** 2))


def particle_tracks(seed=0):
    """The particles of the frames: positions (frame, particle) x, y (px from the top-left corner), advected by the
    jet (wrapped along x) with a small transverse jitter."""
    rng = np.random.default_rng(seed)
    x = np.empty((FRAMES, PARTICLES))
    y = np.empty((FRAMES, PARTICLES))
    x[0], y[0] = rng.uniform(0, WIDTH, PARTICLES), rng.uniform(2, HEIGHT - 2, PARTICLES)
    for k in range(1, FRAMES):
        x[k] = (x[k - 1] + jet(y[k - 1])) % WIDTH
        y[k] = np.clip(y[k - 1] + rng.normal(0, 0.05, PARTICLES), 2, HEIGHT - 2)
    return x, y


def render(xs, ys, k, rng):
    """The image of the frame k: Gaussian particles over a background that drifts slowly (what the POD removes)."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(float)
    image = 20 + 10 * np.sin(2 * np.pi * (xx / WIDTH + k / FRAMES)) + 8 * yy / HEIGHT
    for x, y in zip(xs, ys):
        x0, y0 = int(round(x)), int(round(y))
        a, b = max(x0 - 3, 0), min(x0 + 4, WIDTH)
        c, d = max(y0 - 3, 0), min(y0 + 4, HEIGHT)
        image[c:d, a:b] += 180 * np.exp(-((xx[c:d, a:b] - x) ** 2 + (yy[c:d, a:b] - y) ** 2) / 2.0)
    image += rng.normal(0, 2, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def make_images(folder, seed=0):
    """The raw images of a measurement in folder (img_000.tif ...). Returns their paths."""
    os.makedirs(folder, exist_ok=True)
    x, y = particle_tracks(seed)
    rng = np.random.default_rng(seed + 1)
    paths = []
    for k in range(FRAMES):
        path = os.path.join(folder, f"img_{k:03d}.tif")
        Image.fromarray(render(x[k], y[k], k, rng)).save(path)
        paths.append(path)
    return paths


def preprocess(folder):
    """The POD of the images (MODES modes, saved in PreProcessing/pod.npz) and the images exported without the
    first MODES_REMOVED modes (PreProcessing, with preprocessing.json). Returns (paths of the images exported, the
    recipe, its steps)."""
    out = os.path.join(folder, PREPROCESSING_FOLDER)
    with PODBackgroundRemoval(folder, pattern="*.tif", verbose=False) as session:
        session.evaluate_modes(MODES)
        session.save_pod(os.path.join(out, "pod.npz"))
        paths = session.export_images(out, MODES_REMOVED)
        recipe = session.recipe_record(MODES_REMOVED)
        steps = " → ".join(session.recipe_steps(MODES_REMOVED, detailed=True))
    return paths, recipe, steps


def grid():
    xs = np.arange(SPACING, WIDTH - SPACING + 0.5, SPACING)
    ys = np.arange(SPACING, HEIGHT - SPACING + 0.5, SPACING)
    return np.meshgrid(xs, ys)


def write_results(folder, images, engine, info, seed=0, count=None, tracks=None):
    """A results file of a PIV or a PTV (engine: "pairs" or "tractrac") of images, as PIVRun.save writes it: the
    displacements of the jet on the grid (noise, a few invalid vectors), the peak ratio (PIV) or the particles of each
    cell (PTV: count) and the particles tracked (tracks: the arrays of tracks.npz). Returns the file vectors.npz."""
    output = os.path.join(folder, piv.get_engine(engine).results_folder)  # As the interface: in the measurement
    run = piv.PIVRun(images, None, engine, {}, output=output, info=info)
    xpx, ypx = grid()
    rng = np.random.default_rng(seed)
    n = len(run.pairs)
    u = np.repeat(jet(ypx)[None], n, 0) + rng.normal(0, 0.05, (n,) + xpx.shape)
    v = rng.normal(0, 0.05, (n,) + xpx.shape)
    info_arr = np.where(rng.random((n,) + xpx.shape) < 0.03, 0, 1).astype(np.int16)
    sn = None if engine == "tractrac" else 2.5 + rng.random((n,) + xpx.shape)
    arrays, settings = run.vectors(xpx, ypx, u.astype(np.float32), v.astype(np.float32), sn, info_arr, run.pairs,
                                   count=count if count is None else count[:n], tracks=tracks is not None)
    os.makedirs(output, exist_ok=True)
    np.savez_compressed(os.path.join(output, piv.VECTORS_FILE), **arrays, settings=json.dumps(settings))
    with open(os.path.join(output, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=1)
    if tracks is not None:
        np.savez(os.path.join(output, "tracks.npz"), **tracks)
    return os.path.join(output, piv.VECTORS_FILE)


def tracks_of(seed=0):
    """The particles tracked by a PTV of the frames (the arrays of tracks.npz): each particle at each frame but the
    first and the last ones, with its velocity (px per frame) and acceleration."""
    x, y = particle_tracks(seed)
    rows = {q: [] for q in ("frame", "id", "x", "y", "u", "v", "ax", "ay", "err")}
    for k in range(1, FRAMES - 1):
        u = (x[k + 1] - x[k - 1]) / 2
        u = np.where(np.abs(u) > WIDTH / 2, jet(y[k]), u)  # Not across the wrap
        rows["frame"].append(np.full(PARTICLES, k, np.int32))
        rows["id"].append(np.arange(PARTICLES, dtype=np.int64))
        rows["x"].append(x[k].astype(np.float32))
        rows["y"].append(y[k].astype(np.float32))
        rows["u"].append(u.astype(np.float32))
        rows["v"].append(((y[k + 1] - y[k - 1]) / 2).astype(np.float32))
        rows["ax"].append(np.zeros(PARTICLES, np.float32))
        rows["ay"].append(np.zeros(PARTICLES, np.float32))
        rows["err"].append(np.zeros(PARTICLES, np.float32))
    out = {q: np.concatenate(v) for q, v in rows.items()}
    out.update(detected=np.full(FRAMES, PARTICLES, np.int32), tracked=np.full(FRAMES, PARTICLES - 20, np.int32),
               sequence=np.arange(FRAMES, dtype=np.int32))
    return out


def make_measurement(folder, piv_results=True, ptv_results=False, seed=0):
    """A measurement folder: raw images, their POD and pre-processing, and PIV (subfolder PIV) and/or PTV (subfolder
    PTV, with its particles tracked) results of the images pre-processed. Returns the folder."""
    make_images(folder, seed)
    images, recipe, steps = preprocess(folder)
    info = {"measurement_folder": folder, "preprocessing": steps, "preprocessing_recipe": recipe}
    if piv_results:
        write_results(folder, images, "pairs", info, seed)
    if ptv_results:
        n = FRAMES - 1
        count = np.random.default_rng(seed).integers(3, 9, (n,) + grid()[0].shape).astype(np.int16)
        write_results(folder, images, "tractrac", info, seed, count=count, tracks=tracks_of(seed))
    return folder


def copy_measurement(source, target):
    """A copy of a measurement (the images and the results hard-linked when possible: fast, read-only use)."""
    def link(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    shutil.copytree(source, target, copy_function=lambda s, d: shutil.copy2(s, d) if s.endswith(".json") else link(s, d))
    return target
