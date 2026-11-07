"""Alignment of the fields of view of a stitching (the computations of the window Align of the interface,
gui/alignment.py): the backgrounds common to the fields of view (Source: mean raw image, mean velocity |u|, calibration
image) and the shift of a field of view that best correlates its overlap with another (phase correlation:
suggest_offset).

    from velocihub.archiving import alignment

    ref = alignment.background("D:/Jet/FOV_1/Run_30", fov_width_mm=196.1)         # mean raw image
    mov = alignment.background("D:/Jet/FOV_2/Run_30", fov_width_mm=196.5)
    offset, quality = alignment.suggest_offset(ref, (0, 0), mov, (120, 0), radius_mm=10)
"""

import os
import warnings

import numpy as np
from PIL import Image
from scipy import ndimage

from ..fields import VECTORS_FILE, load_vectors
from ..piv import Camera, PIV_FOLDER
from ..workspace import image_names


BACKGROUNDS = ("Mean raw image", "Mean velocity |u|", "Calibration image")


# The raw images of the mean (evenly spaced), their width at most (block means), the pixels of the composite at most
MEAN_IMAGES = 40


MAX_WIDTH = 1000


CALIBRATION = "Calibration"


class Source:
    """A background of a field of view: arr (rows downwards, normalised 0 to 1, NaN where there is none), its pixel
    (mm), and its left and top edges (mm) without the offsets of the field of view."""

    def __init__(self, arr, pixel, left, top):
        finite = arr[np.isfinite(arr)]
        low, high = np.percentile(finite, (1, 99)) if finite.size else (0.0, 1.0)
        self.arr = np.clip((arr - low) / ((high - low) or 1.0), 0, 1).astype(np.float32)
        self.pixel, self.left, self.top = float(pixel), float(left), float(top)


def read_image(path, reduce):
    with Image.open(path) as im:
        im = im.convert("F") if im.mode not in ("F", "I;16", "I") else im
        arr = np.asarray(im, np.float32)
    if reduce > 1:
        h, w = arr.shape[0] // reduce * reduce, arr.shape[1] // reduce * reduce
        arr = arr[:h, :w].reshape(h // reduce, reduce, w // reduce, reduce).mean(axis=(1, 3))
    return arr


def image_source(paths, fov_width_mm, median=True):
    """The background of images (the median of them; their width: that of the field of view)."""
    with Image.open(paths[0]) as im:
        width = im.size[0]
    reduce = max(1, int(np.ceil(width / MAX_WIDTH)))
    stack = [read_image(p, reduce) for p in paths]
    arr = np.median(np.stack(stack), axis=0) if median and len(stack) > 2 else np.mean(stack, axis=0)
    pixel = fov_width_mm / width * reduce
    return Source(arr, pixel, 0.0, arr.shape[0] * pixel / 2)


def mean_image(project, camera):
    names = image_names(project.folder)
    if not names:
        return None
    picked = [names[int(i)] for i in np.linspace(0, len(names) - 1, min(MEAN_IMAGES, len(names)))]
    return image_source([os.path.join(project.folder, n) for n in picked], camera["fov_width_mm"])


def calibration_image(project, camera):
    """The first image of a folder Calibration in the project or above it (up to its root)."""
    here, root = os.path.normpath(project.folder), os.path.normpath(project.root or project.folder)
    while True:
        folder = os.path.join(here, CALIBRATION)
        names = image_names(folder) if os.path.isdir(folder) else []
        if names:
            return image_source([os.path.join(folder, names[0])], camera["fov_width_mm"], median=False)
        if os.path.normcase(here) == os.path.normcase(root) or os.path.dirname(here) == here:
            return None
        here = os.path.dirname(here)


def mean_velocity(project, camera):
    maps = os.path.join(project.folder, PIV_FOLDER, VECTORS_FILE)
    if not os.path.isfile(maps):
        maps = os.path.join(project.folder, VECTORS_FILE)
    if not os.path.isfile(maps):
        return None
    fields = load_vectors(maps, camera=Camera(fov_width_mm=float(camera["fov_width_mm"]), x_offset_mm=0.0,
                                              y_offset_mm=0.0, frame_rate_hz=float(camera["frame_rate_hz"])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nanmean(np.hypot(fields.U, fields.V), axis=0)
    x, y = fields.x[0], fields.y[:, 0]
    pixel = float(np.median(np.diff(x)))
    # Rows downwards (y upwards in the vector fields)
    arr = speed[::-1] if y[-1] > y[0] else speed
    return Source(arr, pixel, float(x.min()) - pixel / 2, float(y.max()) + pixel / 2)


LOADERS = {BACKGROUNDS[0]: mean_image, BACKGROUNDS[1]: mean_velocity, BACKGROUNDS[2]: calibration_image}


def phase_shift(a, b, radius):
    """The shift (rows, columns; subpixel) of b that best matches a (a(x) ≈ b(x - shift)), within radius pixels:
    phase correlation of the two (NaN: their mean; the large scales left out, a Hann window). Returns (shift,
    quality: the peak over the next highest one)."""
    def prepared(arr):
        arr = np.where(np.isfinite(arr), arr, np.nanmean(arr))
        arr = arr - ndimage.gaussian_filter(arr, 8)
        return arr * np.outer(np.hanning(arr.shape[0]), np.hanning(arr.shape[1]))
    fa, fb = np.fft.fft2(prepared(a)), np.fft.fft2(prepared(b))
    cross = fa * np.conj(fb)
    r = np.fft.ifft2(cross / (np.abs(cross) + 1e-12)).real
    r = np.fft.fftshift(r)
    centre = np.array(r.shape) // 2
    rows, cols = np.indices(r.shape)
    r[np.hypot(rows - centre[0], cols - centre[1]) > radius] = -np.inf
    peak = np.unravel_index(np.argmax(r), r.shape)
    shift = []
    for axis in (0, 1):
        k = peak[axis]
        before = r[(k - 1, peak[1]) if axis == 0 else (peak[0], k - 1)] if k > 0 else -np.inf
        after = r[(k + 1, peak[1]) if axis == 0 else (peak[0], k + 1)] if k < r.shape[axis] - 1 else -np.inf
        top = r[peak]
        sub = 0.0
        if np.isfinite(before) and np.isfinite(after) and before - 2 * top + after < 0:
            sub = 0.5 * (before - after) / (before - 2 * top + after)
        shift.append(k - centre[axis] + sub)
    masked = r.copy()
    masked[max(peak[0] - 2, 0):peak[0] + 3, max(peak[1] - 2, 0):peak[1] + 3] = -np.inf
    second = np.max(masked[np.isfinite(masked)]) if np.isfinite(masked).any() else 0.0
    return np.array(shift), float(r[peak] / second) if second > 0 else float("inf")


def background(folder, fov_width_mm, kind="Mean raw image", frame_rate_hz=150.0, root=None):
    """The background of a project folder for the alignment (a Source; None if it has none): kind among BACKGROUNDS,
    "Mean raw image" (the median of MEAN_IMAGES of its images), "Mean velocity |u|" (of its PIV results; needs the
    frame rate) or "Calibration image" (the first image of a folder Calibration in the folder or above it, up to
    root)."""
    from types import SimpleNamespace

    project = SimpleNamespace(folder=folder, root=root)
    return LOADERS[kind](project, {"fov_width_mm": float(fov_width_mm), "frame_rate_hz": float(frame_rate_hz)})


def resample(source, offset, onto, onto_offset):
    """The background source drawn at its offset (x, y in mm) on the pixels of the background onto at its offset (NaN
    outside source)."""
    rows, cols = np.indices(onto.arr.shape, dtype=np.float64)
    x = onto.left + onto_offset[0] + (cols + 0.5) * onto.pixel
    y = onto.top + onto_offset[1] - (rows + 0.5) * onto.pixel
    c = (x - source.left - offset[0]) / source.pixel - 0.5
    r = (source.top + offset[1] - y) / source.pixel - 0.5
    return ndimage.map_coordinates(source.arr, [r, c], order=1, cval=np.nan)


def suggest_offset(reference, reference_offset, moving, offset, radius_mm=10.0):
    """The offset (x, y in mm) of the field of view of the background moving whose overlap best matches the background
    reference at its offset, searched within radius_mm of offset (phase correlation, then a finer pass), as Suggest of
    the window Align. Returns (offset, quality: the correlation peak over the next one: above 1.5 a clear peak, below
    1.15 likely wrong). ValueError if the two overlap too little."""
    offset = np.array(offset, np.float64)
    quality = None
    for radius in (radius_mm / moving.pixel, 3.0):
        o = resample(reference, reference_offset, moving, offset)
        rows, cols = np.nonzero(np.isfinite(o))
        if rows.size < 400:
            raise ValueError("Too little overlap between the two fields of view: move it closer first")
        pad = int(np.ceil(radius))
        r0, r1 = max(rows.min() - pad, 0), rows.max() + pad + 1
        c0, c1 = max(cols.min() - pad, 0), cols.max() + pad + 1
        shift, q = phase_shift(o[r0:r1, c0:c1], moving.arr[r0:r1, c0:c1], radius)
        quality = q if quality is None else quality
        # o(x) ≈ moving(x - shift): moved by shift (rows downwards: y upwards)
        offset += np.array([shift[1] * moving.pixel, -shift[0] * moving.pixel])
    return (float(offset[0]), float(offset[1])), quality
