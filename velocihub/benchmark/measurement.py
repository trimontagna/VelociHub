"""The measurements of the benchmark: folders laid out as the ones of VelociHub (images, PIV/PTV results in their
subfolders), so that each step runs on them as on real data.

The camera of every case (CAMERA): 0.2 mm per pixel (a 102.4 mm wide field of view) at 1000 frames per second, the
pairs consecutive frames: 1 px per pair is 0.2 m/s. The exact fields of a case are written as a PIV writes its
results (PIVRun.vectors: vectors.npz in pixels and with the velocities, settings.json), in the subfolder EXACT_FOLDER
of the measurement, so that the post-processing reads them as any results."""

import json
import os

import numpy as np
from PIL import Image

from ..piv import VECTORS_FILE, Camera, PIVRun
from .flows import HEIGHT, WIDTH

# The camera of the cases: mm per pixel, frames per second
MM_PER_PX, FRAME_RATE = 0.2, 1000.0
CAMERA = Camera(fov_width_mm=WIDTH * MM_PER_PX, frame_rate_hz=FRAME_RATE)
# m/s per px/pair, 1/s per 1/pair
MS_PER_PX = MM_PER_PX * 1e-3 * FRAME_RATE
PER_S = FRAME_RATE
# The subfolder of the exact fields of a measurement
EXACT_FOLDER = "Exact"
# The description of the case of a measurement (its flow, the camera)
CASE_FILE = "benchmark_case.json"


def grid(spacing):
    """Grid points (px from the top-left corner of the images, as the engines give them): x, y."""
    xs = np.arange(spacing, WIDTH - spacing + 0.5, spacing, dtype=np.float64)
    ys = np.arange(spacing, HEIGHT - spacing + 0.5, spacing, dtype=np.float64)
    return xs, ys


def exact_image(flow, x, y, t=0.0):
    """The exact displacement at image positions x, y (y down) at time t (frames): u, v in image coordinates (v
    positive down)."""
    U, V = flow.velocity(x, HEIGHT - np.asarray(y, np.float64), t)
    return U, -V


def solid_pixels(flow):
    """The pixels of the images inside the body of the flow (the mask)."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    return flow.solid(xx + 0.5, HEIGHT - (yy + 0.5))


def blank_images(folder, n=2):
    """n black images (frame_0000.tif...) in folder, when a measurement has no images: the steps that need their
    size (the PIV results) can run. Returns their paths."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    for k in range(n):
        path = os.path.join(folder, f"frame_{k:04d}.tif")
        if not os.path.isfile(path):
            Image.fromarray(np.zeros((HEIGHT, WIDTH), np.uint8)).save(path)
        paths.append(path)
    return paths


def piv_run(folder, images, engine, output, n_pairs, flow, regression=None, settings=None):
    """The PIVRun of a measurement as the interface makes it: the camera of the cases, the pairs (k, k + 1) of the
    maps (n_pairs, whatever the images), the body of the flow masked."""
    return PIVRun(images, CAMERA, engine, settings or {}, output=output, pairs=[(k, k + 1) for k in range(n_pairs)],
                  masked=solid_pixels(flow), info={"measurement_folder": folder}, regression=regression)


def write_exact(folder, flow, spacing, times, images=None, noise=0.0, seed=0, name=EXACT_FOLDER, description=None):
    """The exact fields of a flow at the grid points, map k at the time times[k] (frames; the middle of its pair),
    written in folder/name as the results of a PIV (vectors.npz, settings.json), with a Gaussian noise (px) on each
    component if noise. images: those of the measurement (two blank ones by default). Returns vectors.npz."""
    images = images or blank_images(folder)
    xs, ys = grid(spacing)
    xpx, ypx = np.meshgrid(xs, ys)
    n = len(times)
    u, v = np.empty((n,) + xpx.shape), np.empty((n,) + xpx.shape)
    for k, t in enumerate(times):
        u[k], v[k] = exact_image(flow, xpx, ypx, t)
    if noise:
        rng = np.random.default_rng(seed)
        u, v = u + rng.normal(0, noise, u.shape), v + rng.normal(0, noise, v.shape)
    run = piv_run(folder, images, "pairs", os.path.join(folder, name), n, flow)
    solid = flow.solid(xpx, HEIGHT - ypx)
    info = np.broadcast_to(np.where(solid, 0, 1).astype(np.int16), u.shape)
    status = np.where(solid, 2, 0).astype(np.int16)
    u, v = (np.where(solid, np.nan, q).astype(np.float32) for q in (u, v))
    arrays, settings = run.vectors(xpx, ypx, u, v, None, info, run.pairs, status=status)
    settings.update(engine="Exact", engine_settings={}, engine_description=description or (
        f"the exact field of the benchmark case {flow.name}" + (f", with a noise of {noise:g} px" if noise else "")),
        benchmark_case=flow.key, benchmark_noise_px=noise, benchmark_times=[float(t) for t in times])
    out = run.output
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, VECTORS_FILE), **arrays, settings=json.dumps(settings))
    with open(os.path.join(out, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=1, ensure_ascii=False)
    return os.path.join(out, VECTORS_FILE)


def write_case_file(folder, flow, extra=None):
    """The description of the case of a measurement (CASE_FILE): the flow and its parameters, the camera."""
    data = {"format": "VelociHub benchmark case", "case": flow.key, "name": flow.name,
            "description": flow.description, "parameters": {k: str(v) for k, v in flow.parameters().items()},
            "camera": {"fov_width_mm": CAMERA.fov_width_mm, "frame_rate_hz": CAMERA.frame_rate_hz,
                       "mm_per_px": MM_PER_PX},
            "image_px": [WIDTH, HEIGHT], **(extra or {})}
    with open(os.path.join(folder, CASE_FILE), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
