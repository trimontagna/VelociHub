"""Pre-processing of the PIV images (POD-based background removal), without interface.

The images are read in one channel (imread: greyscale.read_grey, the brightness V = max(R, G, B) of a colour image,
as every step of VelociHub reads them), then processed with a recipe of three optional steps, applied in this order:
    1. masks: polygons whose pixels are set to 0;
    2. contrast enhancement: intensity capping, local min-max normalization, CLAHE and edge enhancement, in the
       order chosen, applied before the POD (the modes are those of the enhanced images) or after the background
       removal;
    3. background removal: subtraction of the first POD modes of the training images.
The images exported are described in the file preprocessing.json of their folder (recipe, source and names), so
that the PIV uses them if they were processed with the recipe set, instead of processing them again. They are
exported to the subfolder PreProcessing of the image folder, unless another folder is set (path_export), with the
spectrum and the modes evaluated (pod.npz, restored by load_pod). They are named [project]_PreProcessing_Image_
[number].[format]: the format is export_format (EXPORT_FORMATS: that of each source image by default, or PNG, BMP,
TIFF), the project is the name of the folder of the images unless another one is set
(project), and the number is the one of the image in the set.

Example:
    import velocihub as vh

    with vh.PODBackgroundRemoval("images", pattern="*.tif") as session:
        session.set_contrast(["capping", "minmax"], stage="post", cap_n=10, mm_window=15, mm_contrast=4)
        session.evaluate_modes(20)
        session.export_spectrum("results/spectrum.png")
        session.export_modes("results/modes", n_modes=10)
        session.remove_background("results/images", n_modes_removed=10)
"""

import glob
import hashlib
import itertools
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
import psutil
from matplotlib.figure import Figure
from PIL import Image
from scipy import ndimage

from .greyscale import read_grey, to_grey
from .project import absolute, comparable_recipe, default_project_name, export_name, relative

BGREMKEY = "backgroundremovaltool000000001"
# Task in the names of the files exported (export_name)
TASK = "PreProcessing"
# Images selected in a folder when no pattern is given
IMAGE_EXTENSIONS = (".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".gif")
# Filters of the contrast enhancement, their parameters, and stages where it is applied
FILTERS = ("capping", "minmax", "clahe", "edge")
FILTER_PARAMETERS = {"capping": ("cap_n",), "minmax": ("mm_window", "mm_contrast"), "clahe": ("clip", "tiles"),
                     "edge": ("edge_sigma", "edge_factor")}
STAGES = ("pre", "post")
# Subfolder of the image folder where the images are pre-processed for the PIV, and file describing the images
# exported in a folder
PREPROCESSING_FOLDER = "PreProcessing"
MANIFEST_FILE = "preprocessing.json"
# Spectrum and modes evaluated, saved in the export folder (save_pod) to restore them (load_pod)
POD_FILE = "pod.npz"
# Number of pixel values converted to double precision at once to evaluate the correlation matrix
CHUNK = 2**22
# Threads reading, processing and writing the images (the files and most filters release the GIL)
WORKERS = min(8, os.cpu_count() or 1)


################################################################################
# Images and MATLAB functions


def imread(filename):
    """The image of a file in one channel, as every step reads it (greyscale.read_grey: the brightness V = max(R, G,
    B) of a colour image, whatever its colour space), with its original bit depth (8 or 16 bit, 32 bit float)."""
    return read_grey(filename)


# Formats of the images exported: "same" keeps the one of each source image (a JPEG is then compressed again, with
# losses); the others are lossless. Extension and depths each one holds (BMP: 8 bit; PNG: 8 and 16 bit; TIFF: all)
EXPORT_FORMATS = {"same": None, "png": ".png", "bmp": ".bmp", "tiff": ".tif"}
EXPORT_DEPTHS = {"png": (np.uint8, np.uint16), "bmp": (np.uint8,)}
DEFAULT_EXPORT_FORMAT = "same"


def export_format_of(value):
    """The key of EXPORT_FORMATS of a format written in any way ("TIFF", ".tif", "png", "same"...): ValueError if it
    is none of them."""
    key = {"tif": "tiff", "": "same", "none": "same"}.get(str(value or "").strip().lower().lstrip("."),
                                                       str(value or "").strip().lower().lstrip("."))
    if key not in EXPORT_FORMATS:
        raise ValueError(f"Unknown image format {value!r}: one of {', '.join(EXPORT_FORMATS)}")
    return key


def export_extension(fmt, name):
    """Extension of an image exported in the format fmt (EXPORT_FORMATS) from the source image name."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unknown image format {fmt!r}: one of {', '.join(EXPORT_FORMATS)}")
    return EXPORT_FORMATS[fmt] or os.path.splitext(name)[1]


def check_export_depth(fmt, dtype, name=""):
    """ValueError if the format fmt (EXPORT_FORMATS) can't hold images of type dtype (those of the source name with
    "same")."""
    if fmt == "same":
        fmt = {".png": "png", ".bmp": "bmp"}.get(os.path.splitext(name)[1].lower())
    if fmt in EXPORT_DEPTHS and np.dtype(dtype) not in [np.dtype(d) for d in EXPORT_DEPTHS[fmt]]:
        depth = {np.dtype(np.uint16): "16 bit", np.dtype(np.float32): "32 bit float"}.get(np.dtype(dtype),
                                                                                           str(np.dtype(dtype)))
        raise ValueError(f"{fmt.upper()} can't hold {depth} images: choose TIFF"
                         + (" or PNG" if np.dtype(dtype) == np.uint16 else ""))


def imwrite(A, filename):
    """MATLAB imwrite(A, filename, 'Compression', 'none')."""
    options = {"compression": "raw"} if filename.lower().endswith((".tif", ".tiff")) else {}
    Image.fromarray(A).save(filename, **options)


def cast_like(x, A):
    """MATLAB bf = zeros(size(A), 'like', A); bf(:) = x(:);
    Integers are rounded half away from zero and saturated, as MATLAB does."""
    if np.issubdtype(A.dtype, np.integer):
        info = np.iinfo(A.dtype)
        x = np.clip(np.sign(x) * np.floor(np.abs(x) + 0.5), info.min, info.max)
    return x.astype(A.dtype).reshape(A.shape)


# The contrast enhancement filters are mask-aware: valid is the boolean image of the pixels not masked (None: all
# of them). The masked pixels don't bias the statistics, the local extremes or the histograms of the others, so that
# no band appears along the masks. Their own value is meaningless: the masks are applied again afterwards.


def has_mask(valid):
    return valid is not None and not valid.all()


def expand_mask(valid, A):
    """Mask of the pixels of a color image (the same for all the channels)."""
    return valid.reshape(valid.shape + (1,) * (A.ndim - valid.ndim))


def capping_level(A, n, valid=None):
    """Level of the intensity capping: median + n * std of the valid pixels, and the fraction of them above it (None
    without valid pixel)."""
    values = A[np.broadcast_to(expand_mask(valid, A), A.shape)] if has_mask(valid) else A
    if values.size == 0:
        return None
    cap = np.median(values) + n * np.std(values)
    return cap, (values > cap).mean()


def intensity_capping(A, n, valid=None, level=None):
    """Intensity capping (Shavit, Lowe & Steinbuck, Exp. Fluids 2007): the pixels brighter than
    median + n * std of the image are set to that value, so bright particles don't dominate the correlation.
    The median and the standard deviation are those of the valid pixels (level: capping_level, if evaluated)."""
    level = level or capping_level(A, n, valid)
    return A if level is None else cast_like(np.minimum(A, level[0]), A)


def local_mean(x, window):
    """Mean of the finite values of x over window x window pixels (NaN where there is none)."""
    finite = np.isfinite(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ndimage.uniform_filter(np.where(finite, x, 0).astype(np.float32), window)
                / ndimage.uniform_filter(finite.astype(np.float32), window))


def minmax_normalization(A, window, min_contrast, valid=None):
    """Local min-max normalization: (A - min) / max(max - min, min_contrast), where min and max are the
    local extremes over window x window pixels, smoothed over the same window. The minimum contrast (in grey
    levels) prevents amplifying the noise where there are no particles. The result fills the range of the class.
    The local extremes are those of the valid pixels."""
    a = A.astype(np.float32)
    if not has_mask(valid):
        low = ndimage.uniform_filter(ndimage.minimum_filter(a, window), window)
        high = ndimage.uniform_filter(ndimage.maximum_filter(a, window), window)
    else:
        # Extremes of the valid pixels of the window, smoothed over the windows which have some
        valid = expand_mask(valid, a)
        low = local_mean(ndimage.minimum_filter(np.where(valid, a, np.inf), window), window)
        high = local_mean(ndimage.maximum_filter(np.where(valid, a, -np.inf), window), window)
    with np.errstate(invalid="ignore"):
        normalized = np.clip((a - low) / np.maximum(high - low, min_contrast), 0, 1)
    if has_mask(valid):
        normalized = np.where(valid, normalized, 0)
    return cast_like(normalized * (np.iinfo(A.dtype).max if np.issubdtype(A.dtype, np.integer) else 1), A)


def fill_masked(A, valid, sigma):
    """Masked pixels replaced by the local level of the valid pixels around them (normalized Gaussian convolution
    of standard deviation sigma, and the median of the valid pixels far from them)."""
    a = A.astype(np.float32)
    w = valid.astype(np.float32)
    weight = cv2.GaussianBlur(w, (0, 0), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        level = cv2.GaussianBlur(a * w, (0, 0), sigma) / weight
    level = np.where(weight > 1e-3, level, np.median(A[valid]))
    return np.where(valid, A, cast_like(level, A))


def clahe(A, clip, tiles, valid=None):
    """Contrast Limited Adaptive Histogram Equalization of an 8 or 16 bit grayscale image. OpenCV can't exclude
    pixels from the histograms of the tiles: the masked pixels are filled with the local level of the valid
    pixels, which is close to excluding them (rather than adding a peak at 0, or a bright reflection)."""
    if A.ndim != 2 or A.dtype not in (np.uint8, np.uint16):
        raise ValueError("CLAHE requires 8 or 16 bit grayscale images")
    if has_mask(valid) and valid.any():
        A = fill_masked(A, valid, max(A.shape) / tiles / 2)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(A)


def edge_enhancement(A, sigma, factor, valid=None):
    """Edge enhancement (unsharp masking): A + factor * (A - low-pass of A), the low-pass being a Gaussian filter
    of standard deviation sigma (px). It sharpens the particles and lowers the slow variations of the background
    relative to them. The low-pass is the one of the valid pixels (normalized convolution), so that the masks
    don't make a bright band along their edges."""
    a = A.astype(np.float32)
    if has_mask(valid):
        w = expand_mask(valid, a).astype(np.float32) * np.ones_like(a)
        with np.errstate(invalid="ignore", divide="ignore"):
            low = cv2.GaussianBlur(a * w, (0, 0), sigma) / cv2.GaussianBlur(w, (0, 0), sigma)
        low = np.where(np.isfinite(low), low, a)
    else:
        low = cv2.GaussianBlur(a, (0, 0), sigma)
    return cast_like(a + factor * (a - low), A)


def ordered_map(function, items, workers=WORKERS):
    """function(item) of each item, computed by threads a few items ahead, yielded in the order of the items. The
    items not started are dropped when the loop stops."""
    items = iter(items)
    with ThreadPoolExecutor(workers) as executor:
        pending = deque(executor.submit(function, item) for item in itertools.islice(items, 2 * workers))
        try:
            while pending:
                result = pending.popleft().result()
                pending.extend(executor.submit(function, item) for item in itertools.islice(items, 1))
                yield result
        finally:
            for future in pending:
                future.cancel()


def save_npz(filename, **variables):
    """MATLAB save(filename, variables..., 'bgremkey'). The key identifies the files created by this software."""
    np.savez(filename, bgremkey=BGREMKEY, **variables)


def load_npz(filename, *names):
    """MATLAB load(filename, names...)."""
    with np.load(filename) as data:
        return {name: data[name] for name in names}


def list_files(folder, pattern):
    """MATLAB dir(fullfile(folder, pattern)): sorted names of the matching files."""
    return sorted(f for f in glob.glob(pattern, root_dir=folder) if os.path.isfile(os.path.join(folder, f)))


def list_images(folder):
    """Sorted names of the image files of a folder."""
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(folder, f)))


# Bit depth of the images, from their PIL mode
BIT_DEPTHS = {"1": "1 bit", "L": "8 bit", "P": "8 bit (palette)", "I;16": "16 bit", "I;16B": "16 bit",
              "I;16L": "16 bit", "I": "32 bit", "F": "32 bit float", "RGB": "3 x 8 bit (RGB)",
              "RGBA": "4 x 8 bit (RGBA)"}


def image_properties(filename):
    """Width, height (px) and bit depth of an image file."""
    with Image.open(filename) as im:
        return im.size[0], im.size[1], BIT_DEPTHS.get(im.mode, im.mode)


def describe_images(paths):
    """Number, names, size and bit depth of images, as shown in the interface."""
    if not paths:
        return "No images"
    names = [os.path.basename(p) for p in paths]
    width, height, depth = image_properties(paths[0])
    return (f"{len(paths)} images ({names[0]}" + (f" ... {names[-1]}" if len(names) > 1 else "")
            + f"), {width} x {height} px, {depth}")


def folder_images(folder):
    """Images of a folder and the description of their pre-processing (None if there is none): the images exported
    with a recipe (described by preprocessing.json), or else all the images of the folder."""
    manifest = read_manifest(folder)
    if manifest is not None:
        paths = [os.path.join(folder, name) for name in manifest.get("files", [])]
        if paths and all(os.path.isfile(p) for p in paths):
            return paths, manifest
    return [os.path.join(folder, name) for name in list_images(folder)], None


def rasterize_masks(polygons, shape):
    """Boolean image of the pixels of the polygons [[x, y], ...] (pixel centers at integer coordinates from 1;
    the points may be outside the image)."""
    # Polygons rasterized one by one, so that overlapping masks don't cancel out; 4 bits of subpixel precision
    masked = np.zeros(shape[:2], np.uint8)
    for points in polygons:
        cv2.fillPoly(masked, [np.round((np.asarray(points, float) - 1) * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
    return masked.astype(bool)


def read_manifest(folder):
    """Description of the images exported in a folder (written by export_images), None if there is none."""
    try:
        with open(os.path.join(folder, MANIFEST_FILE), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def same_path(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def available_memory():
    """MATLAB memory().MaxPossibleArrayBytes, limited to the physical memory available."""
    return psutil.virtual_memory().available


def mround(x):
    """MATLAB round of a positive number (Python rounds the halves to the even number)."""
    return int(np.floor(x + 0.5))


def str2double(s):
    """MATLAB str2double: NaN if the text is not a number (or None)."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def num2str(x):
    """MATLAB num2str of a scalar."""
    return str(int(x)) if float(x).is_integer() else f"{x:.5g}"


def as2d(A):
    """Image as drawn (H x W): one channel, the intensity every step reads (greyscale.to_grey: the brightness of a
    colour image). Not MATLAB's [H, W] = size(A), which put the channels side by side."""
    return to_grey(A)


def gram(A, B, progress=None):
    """A * B' in double precision. Few columns (pixels) are converted at a time to limit the memory used."""
    C = np.zeros((A.shape[0], B.shape[0]))
    step = max(1, CHUNK // max(A.shape[0], B.shape[0]))
    for j in range(0, A.shape[1], step):
        a = A[:, j : j + step].astype(np.float64)
        b = a if B is A else B[:, j : j + step].astype(np.float64)
        C += a @ b.T
        if progress is not None:
            progress(min(j + step, A.shape[1]) / A.shape[1])
    return C


################################################################################
# Polygons


def closed(P):
    """Points of a polygon followed by the first one, to draw its outline."""
    return np.vstack([P, P[:1]])


def polygon_centroid(P):
    """Centroid of the area of a polygon (the mean of its points if the area is 0)."""
    x, y = P[:, 0], P[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2
    if abs(area) < 1e-9:
        return P.mean(axis=0)
    return np.array([((x + np.roll(x, -1)) * cross).sum(), ((y + np.roll(y, -1)) * cross).sum()]) / (6 * area)


def segment_distance(p, a, b):
    """Distance of the point p to the segment [a, b]."""
    ab = b - a
    t = np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
    return np.hypot(*(a + t * ab - p))


################################################################################
# Settings of the recipe


@dataclass
class ContrastSettings:
    """Contrast enhancement: the filters, applied in the order given, and their parameters.

    filters: names among "capping" (intensity capping), "minmax" (local min-max normalization), "clahe" and "edge"
    (edge enhancement: image + factor * (image - Gaussian low-pass of the image)).
    stage: "pre" (before the POD: the modes are evaluated on the enhanced images) or "post" (after the
    background removal)."""

    filters: tuple = ()
    stage: str = "post"
    # Intensity capping: cap at median + cap_n * std of the image
    cap_n: float = 2.0
    # Local min-max normalization: window (px) and minimum contrast (grey levels)
    mm_window: int = 15
    mm_contrast: float = 10.0
    # CLAHE: clip limit and number of tiles per side
    clip: float = 10.0
    tiles: int = 8
    # Edge enhancement: standard deviation (px) of the Gaussian low-pass, and factor of the high-pass added
    edge_sigma: float = 2.0
    edge_factor: float = 1.0

    def __post_init__(self):
        self.filters = (self.filters,) if isinstance(self.filters, str) else tuple(self.filters)
        unknown = [f for f in self.filters if f not in FILTERS]
        if unknown or len(set(self.filters)) != len(self.filters):
            raise ValueError(f"Invalid filters {self.filters}: each one of {FILTERS} at most once")
        if self.stage not in STAGES:
            raise ValueError(f'Invalid stage "{self.stage}": "pre" (before POD) or "post" (after background removal)')
        self.cap_n, self.mm_contrast, self.clip = float(self.cap_n), float(self.mm_contrast), float(self.clip)
        self.edge_sigma, self.edge_factor = float(self.edge_sigma), float(self.edge_factor)
        if not (self.cap_n >= 0 and self.mm_window >= 2 and self.mm_contrast > 0 and self.clip > 0
                and self.tiles >= 1 and self.edge_sigma > 0 and self.edge_factor >= 0):
            raise ValueError("Invalid parameters: cap_n must be positive or zero, mm_window at least 2 px, "
                             "mm_contrast and clip positive, tiles at least 1, edge_sigma positive and edge_factor "
                             "positive or zero")
        self.mm_window, self.tiles = int(self.mm_window), int(self.tiles)

    def describe(self):
        """Filters and their parameters, in the order they are applied."""
        return " + ".join({"capping": f"intensity capping (n = {num2str(self.cap_n)})",
                           "minmax": f"min-max normalization ({self.mm_window} px, "
                                     f"min. contrast {num2str(self.mm_contrast)})",
                           "clahe": f"CLAHE (clip {num2str(self.clip)}, {self.tiles}x{self.tiles} tiles)",
                           "edge": f"edge enhancement (σ {num2str(self.edge_sigma)} px, factor "
                                   f"{num2str(self.edge_factor)})"}[f]
                          for f in self.filters)


# Recipe the interface starts with, and the one of the batch processing (batch.py): intensity capping (n = 10) after
# the background removal, and the number of modes evaluated and removed
DEFAULT_CONTRAST = ContrastSettings(filters=("capping",), stage="post", cap_n=10)
DEFAULT_MODES = 6


@dataclass(eq=False)
class Mask:
    """Polygon whose pixels are set to 0. points: [[x, y], ...] in pixels, x along the columns and y along the
    rows, the pixel centers having integer coordinates from 1 (as in MATLAB)."""

    points: np.ndarray
    name: str = "Mask"
    enabled: bool = True

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 2)
        if len(self.points) < 3:
            raise ValueError("A mask needs at least 3 points")


################################################################################
# Figures


def draw_spectrum(fig, eigVal, mean_PSI, PSI=None, mode=None):
    """Plot in a figure the spectrum, the temporal averages of the modes and the temporal coefficient of a mode
    (None: no mode). The modes n are sorted by decreasing energy, the images k are the snapshots in time."""
    k = np.arange(1, len(eigVal) + 1)
    fig.clf()

    # Eigenvalues of the correlation matrix M M' (squares of the singular values of M)
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(k, eigVal, "o-", linewidth=2)
    ax.set_xlabel("Mode number n")
    ax.set_ylabel(r"Energy $\lambda_n = \sigma_n^2$")
    if mode is not None:
        ax.plot(mode, eigVal[mode - 1], "or", linewidth=2, markerfacecolor="r", label=f"mode {mode}")
        ax.legend(fontsize="small")
    ax.set_title("Video spectrum", fontsize="medium")

    ax = fig.add_subplot(1, 3, 2)
    ax.plot(k, mean_PSI, "o-", linewidth=2)
    ax.set_xlabel("Mode number n")
    ax.set_ylabel(r"Temporal mean $\langle\psi_n,1\rangle$")
    ax.set_title("Mode's Temporal Averages", fontsize="medium")
    if mode is not None:
        ax.plot(mode, mean_PSI[mode - 1], "or", linewidth=2, markerfacecolor="r", label=f"mode {mode}")
        ax.legend(fontsize="small")

    ax = fig.add_subplot(1, 3, 3)
    if mode is not None:
        draw_temporal_coefficient(ax, PSI[:, mode - 1], mode)
    else:
        ax.text(0.5, 0.5, "Evaluate the modes to see\ntheir temporal coefficients", ha="center", va="center",
                color="0.45", transform=ax.transAxes)
        ax.set_title("Temporal coefficient", fontsize="medium")
        ax.set_xlabel("Image number k (time)")
        ax.set_ylabel(r"$\psi_n(k)$ (unit norm)")
    return fig


def draw_temporal_coefficient(ax, psi, mode):
    """Plot the temporal coefficient of a mode."""
    ax.plot(np.arange(1, len(psi) + 1), psi, "o-", linewidth=2)
    ax.set_title(f"Temporal coefficient of mode {mode}", fontsize="medium")
    ax.set_xlabel("Image number k (time)")
    ax.set_ylabel(rf"$\psi_{{{mode}}}(k)$ (unit norm)")


def title_lines(text, width=110):
    """A title in lines of at most width characters (its own lines kept, the longer ones broken at the spaces). The
    titles aren't wrapped by Matplotlib (wrap=True): its wrapping depends on where the title is during the layout, which
    can make the constrained layout reserve a tall band for it and shrink the axes (adding a colour bar, for example)."""
    import textwrap
    return "\n".join(part for line in str(text).split("\n")
                     for part in (textwrap.wrap(line, width, break_long_words=False) or [""]))


def draw_image(fig, img, cmap="gray", clim=None, title="", clabel=None):
    """MATLAB imagesc(img); axis ij equal tight; colorbar: the pixel centers have integer coordinates from 1.
    Returns the axes."""
    img = as2d(img)
    H, W = img.shape
    ax = fig.add_subplot()
    im = ax.imshow(img, cmap=cmap, extent=(0.5, W + 0.5, H + 0.5, 0.5),
                   vmin=None if clim is None else clim[0], vmax=None if clim is None else clim[1])
    ax.set_title(title_lines(title), fontsize="medium")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    if clabel is not None:
        fig.colorbar(im, ax=ax, label=clabel)
    return ax


MODE_LABEL = r"Spatial mode $\phi_n$ (unit norm, a.u.)"


def mode_limits(PHI):
    """Color limits of a spatial mode: symmetric, at the 99.9th percentile of its magnitude, so that a few extreme
    pixels don't hide the structure of the mode."""
    v = float(np.percentile(np.abs(PHI), 99.9)) or float(np.abs(PHI).max()) or 1.0
    return -v, v


def mode_title(mode, n_modes, eigVal=None):
    """Title of a mode: its number, its share of the energy of the images (eigenvalue over their sum) and that of the
    modes up to it (the energy the background removal of that many modes takes away)."""
    title = f"Mode {mode} of {n_modes}"
    eigVal = None if eigVal is None else np.asarray(eigVal, np.float64)
    if eigVal is not None and eigVal.size >= mode and eigVal.sum() > 0:
        total = eigVal.sum()
        title += (f"\nEnergy: {100 * eigVal[mode - 1] / total:.3g} % "
                  f"(modes 1-{mode}: {100 * eigVal[:mode].sum() / total:.4g} %)")
    return title


################################################################################
# Progress of the processing


class Stopped(Exception):
    """The user stopped a process (raised by its progress, e.g. the Stop button of the interface): the results of
    what was done before stay consistent (the modes saved, no description of an export interrupted)."""


class ConsoleProgress:
    """Progress of a process printed in the console, at most every 5 s (the interface shows a progress bar)."""

    def __init__(self, title, verbose=True):
        self.title = title
        self.verbose = verbose
        self.last = time.monotonic()

    def __call__(self, x, message=None):
        if self.verbose and time.monotonic() - self.last >= 5:
            self.last = time.monotonic()
            print(f"  {self.title} {x:4.0%}  {message or ''}", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass


################################################################################
# Background removal


class PODBackgroundRemoval:
    """POD-based background removal of a set of PIV images (Mendez et al., Exp. Therm. Fluid Sci. 2017).

    The training images are the images of a folder (set_images). Their POD modes are evaluated (evaluate_spectrum,
    evaluate_modes) and stored in a temporary folder; the correlation of the images is evaluated in memory, or by
    blocs saved in temporary files if they don't fit. The images are processed with the recipe (masks, contrast
    enhancement and background removal) by remove_background and export_images, and for the PIV by piv_images
    (in the subfolder PreProcessing, unless they were already exported with the recipe). The figures of the
    spectrum and of the modes are exported by export_spectrum and export_modes.

    Without temp_folder, the temporary files are stored in a new temporary folder, deleted by close() (the session
    can be used in a "with" statement); temp_folder="images": in the folder of the images, as the original tool.
    The attributes keep the names of the fields of the MATLAB original (background_removal.m)."""

    def __init__(self, folder=None, pattern=None, files=None, first=1, last=None, temp_folder=None,
                 memory_fraction=0.8, verbose=True):
        self.verbose = verbose
        self.initialize_program()
        self.mem_safe = memory_fraction
        # Temporary folder created by the session, deleted by close()
        self.own_temp = None
        if temp_folder is None:
            self.own_temp = self.pathtemp = tempfile.mkdtemp(prefix="bgrem_")
            self.pathtemp_user = True
        elif temp_folder != "images":
            self.set_temp_folder(temp_folder)
        self.reset_session()
        if folder is not None or files is not None:
            self.set_images(folder, pattern, files, first, last)

    def initialize_program(self):
        """Settings kept from a session to the next."""
        # Name of the project, at the start of the names of the files exported: None for the name of the folder
        # of the images
        self.project = None
        self.pathtemp = self.path_in = os.getcwd()
        # The temporary folder is the image folder unless the user selects one
        self.pathtemp_user = False
        self.path_out = self.path_in
        # Folder of the images exported for the PIV and of the POD saved: None for the subfolder PreProcessing of
        # the image folder
        self.path_export = None
        # Format of the images exported (EXPORT_FORMATS: "same" as the source images, "png", "bmp" or "tiff")
        self.export_format = DEFAULT_EXPORT_FORMAT
        self.file_in = []
        self.Use_tmp_file = False
        self.mem_safe = 0.8
        # Steps of the recipe: the masks enabled and the filters of the contrast enhancement (the background removal
        # is a number of modes); masks_enabled and contrast_enabled switch them off without changing them
        self.masks_enabled = True
        self.contrast_enabled = True
        self.contrast = ContrastSettings()
        # Intensity cap of the last image enhanced: (cap, fraction of the pixels capped)
        self.last_cap = None
        # Masks, and the pixels they mask
        self.masks = []
        self.mask_cache = None
        # Built in
        self.bytesPerElem = 4  # 4 for single, 8 for double

    def reset_session(self):
        """New session: no images, spectrum nor modes."""
        self.NImg = 0
        self.NMod_saved = 0
        # Epsilon for the automatic threshold
        self.eps_auto_psi = 1e-4
        self.eps_auto_sigma = 1e-4
        # Checks
        self.Check_imsel = False  # Images selected
        self.Check_imload = False  # Images loaded into memory
        self.Check_eigproc = False  # Eigenvalues evaluated
        self.Check_modproc = False  # Modes evaluated
        # Images of the previous session
        self.M_bloc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        """Delete the temporary files, and the temporary folder created by the session."""
        self.clean_temporary_files()
        if self.own_temp is not None:
            shutil.rmtree(self.own_temp, ignore_errors=True)
            self.own_temp = None

    ############################################################################
    # Hooks of the interface: messages, progress and questions

    def update_status(self, msg):
        """Message on the processing (printed if verbose, shown in the status log of the interface)."""
        if self.verbose and isinstance(msg, str) and msg.strip():
            # The console may not encode the arrows of the recipe
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            if "→" in msg and "→".encode(encoding, errors="replace") != "→".encode("utf-8"):
                msg = msg.replace("→", "->")
            print(msg, flush=True)

    def progress(self, title):
        """Progress of a long process: context manager called with the fraction done and a message."""
        return ConsoleProgress(title, self.verbose)

    def confirm_temp_files(self):
        """The images don't fit in memory: continue with temporary files?"""
        self.update_status("The memory available is not enough to process the images in memory: "
                           f"temporary files are stored in {self.pathtemp}")
        return True

    ############################################################################
    # Images

    @staticmethod
    def find_images(folder=None, pattern=None, files=None, first=1, last=None):
        """Folder and sorted names of images: the files of a folder matching a pattern (all the images without
        pattern), from the first to the last (1-based, included), or a list of files (absolute, or relative to
        the folder)."""
        if files is not None:
            paths = [f if folder is None else os.path.join(folder, f) for f in files]
            if not paths:
                return folder, []
            folder = os.path.dirname(os.path.abspath(paths[0]))
            names = sorted(os.path.basename(p) for p in paths)
        else:
            if folder is None:
                raise ValueError("A folder or a list of files is needed")
            folder = os.path.abspath(folder)
            names = list_files(folder, pattern) if pattern else list_images(folder)
        return os.path.normpath(folder), names[max(first, 1) - 1 : last]

    def set_images(self, folder=None, pattern=None, files=None, first=1, last=None):
        """Training images: the files of a folder matching a pattern (all the images without pattern), from the
        first to the last (1-based, included), or a list of files. Starts a new session. Returns their number."""
        folder, names = self.find_images(folder, pattern, files, first, last)
        if len(names) < 2:
            raise ValueError(f"At least two images are needed: {len(names)} found in {folder}")
        self.reset_session()
        # The export folder chosen for other images isn't the one of these
        if not same_path(folder, self.path_in):
            self.path_export = None
        self.file_in = names
        self.path_in = folder
        if not self.pathtemp_user:
            self.pathtemp = folder
        self.NImg = len(names)
        self.Check_imsel = True
        self.update_status(f"{self.NImg} images selected in {self.path_in}")
        return self.NImg

    @property
    def images(self):
        """Paths of the training images."""
        return [os.path.join(self.path_in, f) for f in self.file_in]

    def set_temp_folder(self, folder):
        """Folder of the temporary files. The images loaded and the modes are discarded."""
        os.makedirs(folder, exist_ok=True)
        self.pathtemp = os.path.normpath(folder)
        self.pathtemp_user = True
        self.Check_imload = self.Check_eigproc = self.Check_modproc = False
        self.NMod_saved = 0

    def load_images_memory(self):
        """Load the training images (with the masks and the contrast enhancement before POD) into memory, or into
        temporary files if they don't fit. Returns False if the user stops it."""
        if not self.Check_imsel:
            raise RuntimeError("No images selected: call set_images first")

        # Process the images
        NImg = self.NImg
        sector = 1
        # Free the memory used by the images loaded before
        self.M_bloc = None

        def read(name):
            try:
                return self.read_image(os.path.join(self.path_in, name))
            except Exception as me:
                raise OSError(f'"{me}"\nThe error occurred while trying to open the file:\n{name}') from me

        # Load images into memory and save them into temporary files. Each
        # temporary file is a "bloc" that contains a matrix M.
        with self.progress("Load images in memory") as wh:
            for im_i, (name, A) in enumerate(zip(self.file_in, ordered_map(read, self.file_in)), 1):
                wh(im_i / NImg, f"Loading image {im_i} of {NImg}...")

                # Initialize variables
                if im_i == 1:
                    Dim = A.shape

                    # Evaluate the memory needed
                    MaxImg = self.max_images(available_memory(), A.size)
                    if NImg <= MaxImg:
                        N_bloc = 0
                        self.Use_tmp_file = False
                    else:
                        N_bloc = -(-NImg // MaxImg)
                        self.Use_tmp_file = True
                        if not self.confirm_temp_files():
                            return False

                    M_bloc = np.zeros((min(MaxImg, NImg), A.size), np.float32)
                elif A.shape != Dim:
                    raise ValueError(f"The image {name} doesn't have the size of the first image.")

                M_bloc[im_i - 1 - (sector - 1) * MaxImg] = A.ravel()

                if im_i == sector * MaxImg or im_i == NImg:
                    if self.Use_tmp_file:
                        wh(im_i / NImg, f"Saving temporary file {sector} of {N_bloc}...")
                        save_npz(self.temp_file(sector), M_bloc=M_bloc)
                        sector += 1
                        M_bloc = None
                        if im_i < NImg:
                            M_bloc = np.zeros((min(MaxImg, NImg - im_i), A.size), np.float32)
                    else:
                        self.M_bloc = M_bloc

        # Save gui data
        self.Check_imload = True
        self.N_bloc = N_bloc
        self.Dim = Dim
        self.Check_eigproc = False
        self.MaxImg = MaxImg

        # Update status
        self.update_status(" ")
        self.update_status(f"{self.N_bloc} temporary file created")
        self.update_status(f"{self.NImg} images loaded into memory")
        return True

    def max_images(self, MemAv, n_pixels):
        """Number of images that can be processed in memory, without temporary files."""
        # Memory available / bytes per element / image size / 2
        # Divided by 2 is because two dataset must be kept in memory
        return max(1, int(MemAv / self.bytesPerElem / n_pixels / 2 * self.mem_safe))

    ############################################################################
    # POD

    def evaluate_spectrum(self):
        """Evaluate the correlation matrix of the training images (loaded first if needed) and its eigenvalues and
        eigenvectors (temporal coefficients of the modes). Returns the eigenvalues, None if the user stops it."""
        if not self.Check_imload and not self.load_images_memory():
            return None

        NImg = self.NImg
        N_bloc = self.N_bloc

        with self.progress("Evaluating eigenvalues...") as wh:
            # Create the matrix C. It is accumulated in double precision because it
            # contains very high values (exactly, for 8 and 16 bit images).
            if self.Use_tmp_file:
                C = np.zeros((NImg, NImg))
                # Progress: number of blocs couples correlated
                n_pairs = N_bloc * (N_bloc + 1) // 2
                pair = 0
                for bloc_i in range(1, N_bloc + 1):
                    wh(pair / n_pairs, f"Loading bloc {bloc_i} of {N_bloc}...")
                    M_i = self.load_bloc(bloc_i)

                    # Covariance on the diagonal elements can be directly calculated
                    # using M_block
                    index = self.bloc_index(bloc_i)
                    C[index, index] = gram(M_i, M_i)
                    pair += 1

                    for bloc_j in range(bloc_i + 1, N_bloc + 1):
                        wh(pair / n_pairs, f"Loading bloc {bloc_j} of {N_bloc}...")
                        M_j = self.load_bloc(bloc_j)
                        ind2 = self.bloc_index(bloc_j)
                        C[index, ind2] = gram(M_i, M_j, lambda x: wh((pair + x) / n_pairs,
                                                                     f"Correlating blocs {bloc_i} and {bloc_j}..."))
                        C[ind2, index] = C[index, ind2].T
                        pair += 1
                        # The variable must be deleted before loading the next bloc,
                        # otherwise both blocs are in memory at the same time
                        del M_j
                    del M_i
            else:
                C = gram(self.M_bloc, self.M_bloc, lambda x: wh(x, "Evaluating the correlation matrix..."))

            # Update status
            self.update_status(" ")
            self.update_status("Images spectrum correctly evaluated")

            wh(1, "Evaluating eigenvalues...")
            PSI, eigVal, _ = np.linalg.svd(C)
            mean_PSI = PSI.mean(axis=0)

        # Save gui data
        self.eigVal = eigVal
        self.mean_PSI = mean_PSI
        self.PSI = PSI
        self.Check_eigproc = True
        return eigVal

    def automatic_N_modes(self):
        """Automatic number of modes for the tolerances eps_auto_sigma and eps_auto_psi (spectrum evaluated): the
        first mode whose temporal mean |mean psi_i| (the temporal coefficients are normalized: no unit) and energy
        difference with the next one, |lambda_i - lambda_i+1| / sum lambda (a fraction of the energy of the images,
        as the spectrum shows it), are below them. 0 if there is none. (The original compared the difference over
        the median eigenvalue with eps_auto_sigma times the first eigenvalue: a number with the unit of an
        eigenvalue, so large that the criterion was always met and only the temporal mean decided.)"""
        eigVal = np.asarray(self.eigVal, np.float64)
        total = eigVal.sum()
        if not total > 0:
            return 0
        for i in range(1, self.NImg):
            Mean_PSI = abs(self.mean_PSI[i - 1])
            Sig_Diff = abs(eigVal[i - 1] - eigVal[i]) / total
            if Mean_PSI < self.eps_auto_psi and Sig_Diff < self.eps_auto_sigma:
                return i
        return 0

    def evaluate_modes(self, n_modes, on_mode=None):
        """Evaluate and store the first n_modes modes (the spectrum is evaluated first if needed); the modes
        already evaluated are kept. on_mode(n) is called when the mode n is stored. Returns the number of modes
        evaluated, None if the user stops it."""
        if not self.Check_eigproc and self.evaluate_spectrum() is None:
            return None
        if not n_modes >= 1:
            raise ValueError("The number of modes must be higher than 0")
        NMod = int(min(n_modes, self.NImg))
        if NMod > self.NMod_saved and not self.Check_imload:
            # Spectrum restored by load_pod: the images are loaded again to evaluate the new modes
            if not self.load_images_memory():
                return None
            self.Check_eigproc = True
        NImg = self.NImg
        N_bloc = self.N_bloc

        # Check if there are other modes already saved
        first_mod = self.NMod_saved + 1
        N_new = NMod - first_mod + 1

        with self.progress("Evaluating modes...") as wh:
            # Evaluate the new modes all together, so that the images are read only twice
            if N_new > 0:
                PSI = self.PSI[:, first_mod - 1 : NMod].astype(np.float32)

                # Evaluate PHI
                if self.Use_tmp_file:
                    PHI_bloc = np.zeros((N_new, int(np.prod(self.Dim))), np.float32)
                    for bloc_i in range(1, N_bloc + 1):
                        wh(bloc_i / (2 * N_bloc),
                           f"Evaluating modes {first_mod} to {NMod} (bloc {bloc_i} of {N_bloc})...")
                        PHI_bloc += PSI[self.bloc_index(bloc_i)].T @ self.load_bloc(bloc_i)
                else:
                    wh(0, f"Evaluating modes {first_mod} to {NMod}...")
                    PHI_bloc = PSI.T @ self.M_bloc
                PHI_bloc /= np.linalg.norm(PHI_bloc, axis=1, keepdims=True)

                # Evaluate T
                if self.Use_tmp_file:
                    TCoeff_bloc = np.zeros((NImg, N_new), np.float32)
                    for bloc_i in range(1, N_bloc + 1):
                        wh((N_bloc + bloc_i) / (2 * N_bloc),
                           f"Evaluating temporal coefficients (bloc {bloc_i} of {N_bloc})...")
                        TCoeff_bloc[self.bloc_index(bloc_i)] = self.load_bloc(bloc_i) @ PHI_bloc.T
                else:
                    TCoeff_bloc = self.M_bloc @ PHI_bloc.T

                for k, mod_i in enumerate(range(first_mod, NMod + 1)):
                    wh((k + 1) / N_new, f"Saving mode {mod_i} of {NMod}...")
                    save_npz(self.mode_file(mod_i), PHI_bloc=PHI_bloc[k], TCoeff_bloc=TCoeff_bloc[:, k])
                    # Counted once saved: a process stopped keeps the modes saved before
                    self.NMod_saved, self.Check_modproc = mod_i, True
                    if on_mode is not None:
                        on_mode(mod_i)
        self.Check_modproc = self.Check_modproc or self.NMod_saved > 0

        self.update_status(" ")
        self.update_status(f"{max(N_new, 0)} modes were evaluated and stored into files")
        return self.NMod_saved

    def pod_key(self, folder=False):
        """Identifies the training images as the POD sees them: their names, the masks enabled and the contrast
        enhancement before POD (not their folder, which may be moved with the project; folder: the key of the
        previous versions, with the absolute folder)."""
        record = self.recipe_record(0)
        if record["contrast"] is not None and record["contrast"]["stage"] != "pre":
            record["contrast"] = None
        record.update(files=self.file_in)
        if folder:
            record.update(folder=os.path.normcase(os.path.abspath(self.path_in)))
        return hashlib.sha1(json.dumps(record, sort_keys=True).encode()).hexdigest()

    def save_pod(self, filename=None):
        """Save the spectrum and the modes evaluated (pod.npz of the export folder by default), so that load_pod
        restores them without evaluating them again. Returns the file name, None if the spectrum isn't
        evaluated."""
        if not self.Check_eigproc:
            return None
        filename = filename or os.path.join(self.export_folder(), POD_FILE)
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        modes = self.load_modes(self.NMod_saved)
        size = int(np.prod(self.Dim))
        PHI = np.array([m["PHI_bloc"].ravel() for m in modes], np.float32).reshape(-1, size)
        TCoeff = np.array([m["TCoeff_bloc"] for m in modes], np.float32).reshape(-1, self.NImg)
        # Written next to the file and renamed, so that an interrupted save doesn't leave a broken file
        temporary = filename + ".part.npz"
        save_npz(temporary, key=self.pod_key(), eigVal=self.eigVal, mean_PSI=self.mean_PSI, PSI=self.PSI,
                 Dim=np.array(self.Dim), PHI=PHI, TCoeff=TCoeff)
        os.replace(temporary, filename)
        return filename

    def load_pod(self, filename=None):
        """Restore the spectrum and the modes saved by save_pod, if they are those of the training images with the
        masks and the contrast enhancement before POD set. Returns the number of modes restored, None if there is
        no such file."""
        filename = filename or os.path.join(self.export_folder(), POD_FILE)
        if not self.Check_imsel or not os.path.isfile(filename):
            return None
        try:
            with np.load(filename) as z:
                if str(z["key"]) not in (self.pod_key(), self.pod_key(folder=True)):
                    return None
                eigVal, mean_PSI, PSI, Dim, PHI, TCoeff = (z[k] for k in ("eigVal", "mean_PSI", "PSI", "Dim", "PHI",
                                                                          "TCoeff"))
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return None
        self.M_bloc = None
        self.Check_imload = False
        self.eigVal, self.mean_PSI, self.PSI = eigVal, mean_PSI, PSI
        self.Dim = tuple(int(d) for d in Dim)
        self.Check_eigproc = True
        for n in range(len(PHI)):
            save_npz(self.mode_file(n + 1), PHI_bloc=PHI[n], TCoeff_bloc=TCoeff[n])
        self.NMod_saved = len(PHI)
        self.Check_modproc = self.NMod_saved > 0
        self.update_status(f"Spectrum and {self.NMod_saved} modes restored from {filename}")
        return self.NMod_saved

    def discard_pod(self):
        """The training images changed: the spectrum and the modes must be evaluated again."""
        self.Check_imload = self.Check_eigproc = self.Check_modproc = False
        self.NMod_saved = 0
        self.M_bloc = None
        self.update_status("The training images changed: the spectrum and the modes must be evaluated again")

    def mode_map(self, n):
        """Spatial mode n (unit norm) as an image."""
        self.check_mode(n)
        return load_npz(self.mode_file(n), "PHI_bloc")["PHI_bloc"].reshape(self.Dim)

    def temporal_coefficient(self, n):
        """Temporal coefficient of the mode n (unit norm), for each training image: it only needs the spectrum,
        evaluated if needed."""
        if not self.Check_eigproc and self.evaluate_spectrum() is None:
            return None
        if not 1 <= n <= self.NImg:
            raise ValueError(f"Mode {n} out of the {self.NImg} modes of the spectrum")
        return self.PSI[:, n - 1]

    def check_mode(self, n):
        if not 1 <= n <= self.NMod_saved:
            raise ValueError(f"Mode {n} not evaluated: {self.NMod_saved} modes available (see evaluate_modes)")

    ############################################################################
    # Figures

    def plot_spectrum(self, mode=None, fig=None):
        """Figure of the spectrum, of the temporal averages of the modes and of the temporal coefficient of a mode
        (None: no mode). The spectrum is evaluated if needed."""
        if not self.Check_eigproc and self.evaluate_spectrum() is None:
            return None
        if mode is not None and not 1 <= mode <= self.NImg:
            raise ValueError(f"Mode {mode} out of the {self.NImg} modes of the spectrum")
        fig = fig or Figure(figsize=(15, 4.5), layout="constrained")
        return draw_spectrum(fig, self.eigVal, self.mean_PSI, self.PSI, mode)

    def export_spectrum(self, filename, mode=None, dpi=150):
        """Save the figure of the spectrum (plot_spectrum). Returns the file name."""
        folder = os.path.dirname(os.path.abspath(filename))
        os.makedirs(folder, exist_ok=True)
        self.plot_spectrum(mode).savefig(filename, dpi=dpi)
        self.update_status(f"Spectrum saved in {filename}")
        return filename

    def plot_mode(self, n, cmap="hot", fig=None):
        """Figure of the spatial mode n."""
        fig = fig or Figure(figsize=(8, 5.5), layout="compressed")
        PHI = self.mode_map(n)
        draw_image(fig, PHI, cmap, mode_limits(PHI), mode_title(n, self.NMod_saved, self.eigVal), MODE_LABEL)
        return fig

    def plot_temporal_coefficient(self, n, fig=None):
        """Figure of the temporal coefficient of the mode n."""
        fig = fig or Figure(figsize=(8, 4), layout="constrained")
        draw_temporal_coefficient(fig.add_subplot(), self.temporal_coefficient(n), n)
        return fig

    def export_modes(self, folder, n_modes=None, fmt="png", dpi=150, cmap="hot", save_data=False):
        """Save, for the modes 1 to n_modes (all the modes evaluated by default; they are evaluated if needed),
        the figure of the spatial mode (mode_001_map.png...) and the graph of its temporal coefficient
        (mode_001_temporal_coefficient.png...); save_data: also the mode as an array (mode_001.npy). Returns the
        files written."""
        n_modes = n_modes or self.NMod_saved
        if n_modes > self.NMod_saved and self.evaluate_modes(n_modes) is None:
            return []
        os.makedirs(folder, exist_ok=True)
        written = []
        with self.progress("Export modes") as wh:
            for n in range(1, n_modes + 1):
                wh(n / n_modes, f"Saving mode {n} of {n_modes}...")
                name = os.path.join(folder, f"mode_{n:03d}")
                self.plot_mode(n, cmap).savefig(f"{name}_map.{fmt}", dpi=dpi)
                self.plot_temporal_coefficient(n).savefig(f"{name}_temporal_coefficient.{fmt}", dpi=dpi)
                written += [f"{name}_map.{fmt}", f"{name}_temporal_coefficient.{fmt}"]
                if save_data:
                    np.save(f"{name}.npy", self.mode_map(n))
                    written.append(f"{name}.npy")
        self.update_status(f"{n_modes} modes saved in {folder}")
        return written

    ############################################################################
    # Recipe: masks, contrast enhancement and background removal

    def training_recipe(self):
        """What changes the training images: the masks and the contrast enhancement before POD."""
        masks = tuple((m.points.tobytes(), m.points.shape) for m in self.masks if m.enabled and self.masks_enabled)
        return masks, self.contrast if self.contrast_on() and self.contrast.stage == "pre" else None

    def change_recipe(self, change):
        """Apply a change of the recipe. If it changes the training images, the spectrum and the modes are
        discarded."""
        before = self.training_recipe()
        result = change()
        self.mask_cache = None
        if self.training_recipe() != before and (self.Check_imload or self.Check_eigproc):
            self.discard_pod()
        return result

    def set_contrast(self, filters=None, stage=None, enabled=True, **parameters):
        """Contrast enhancement: filters among "capping" (intensity capping), "minmax" (local min-max
        normalization), "clahe" and "edge" (edge enhancement), applied in the order given (none: no contrast
        enhancement), before the POD (stage="pre") or after the background removal (stage="post"). parameters:
        cap_n (cap at median + cap_n * std), mm_window (px), mm_contrast (grey levels), clip and tiles (CLAHE),
        edge_sigma (px) and edge_factor (edge enhancement). The filters, the stage and the parameters not given are
        kept. enabled=False: no contrast enhancement, the settings being kept.
        Returns the settings."""
        settings = {**vars(self.contrast), **parameters}
        if filters is not None:
            settings["filters"] = filters
        if stage is not None:
            settings["stage"] = stage
        new = ContrastSettings(**settings)

        def change():
            self.contrast, self.contrast_enabled = new, enabled
        self.change_recipe(change)
        self.update_status(f"Contrast enhancement {'before POD' if new.stage == 'pre' else 'after background removal'}"
                           f": {new.describe() if enabled and new.filters else 'none'}")
        return new

    def contrast_on(self):
        return self.contrast_enabled and bool(self.contrast.filters)

    def contrast_description(self):
        """Filters of the contrast enhancement and their parameters, as shown in the titles."""
        return self.contrast.describe()

    def enhance_contrast(self, A, valid=None):
        """Apply the contrast enhancement filters to an image, in their order. valid: the pixels not masked, which
        the filters only use (None: all). The intensity cap is kept in self.last_cap."""
        c = self.contrast
        for name in c.filters:
            if name == "capping":
                level = capping_level(A, c.cap_n, valid)
                self.last_cap = level or self.last_cap
                A = intensity_capping(A, c.cap_n, valid, level)
            elif name == "minmax":
                A = minmax_normalization(A, c.mm_window, c.mm_contrast, valid)
            elif name == "edge":
                A = edge_enhancement(A, c.edge_sigma, c.edge_factor, valid)
            else:
                A = clahe(A, c.clip, c.tiles, valid)
        return A

    def add_mask(self, points, name=None, enabled=True):
        """Add a polygonal mask: points [[x, y], ...] in pixels (pixel centers at integer coordinates from 1).
        Returns the mask."""
        if name is None:
            names = {mask.name for mask in self.masks}
            n = 1
            while f"Mask {n}" in names:
                n += 1
            name = f"Mask {n}"
        mask = Mask(points, name, enabled)
        self.change_recipe(lambda: self.masks.append(mask))
        self.update_status(f"{mask.name} added ({len(mask.points)} points)")
        return mask

    def save_masks(self, filename):
        """Save the masks in a JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"masks": [{"name": m.name, "enabled": m.enabled, "points": m.points.tolist()}
                                 for m in self.masks]}, f, indent=1)
        self.update_status(f"{len(self.masks)} masks saved in {filename}")

    def load_masks(self, filename, replace=True):
        """Load masks from a JSON file (saved by save_masks), replacing the masks or added to them."""
        with open(filename, encoding="utf-8") as f:
            masks = [Mask(m["points"], m.get("name", "Mask"), m.get("enabled", True)) for m in json.load(f)["masks"]]
        self.change_recipe(lambda: self.masks.__setitem__(slice(None), (masks if replace else self.masks + masks)))
        self.update_status(f"{len(masks)} masks loaded from {filename}")
        return masks

    def masks_active(self):
        return self.masks_enabled and any(mask.enabled for mask in self.masks)

    def masked_pixels(self, shape):
        """Boolean image of the masked pixels (None if the masks aren't applied)."""
        if not self.masks_active():
            return None
        shape = shape[:2]
        if self.mask_cache is None or self.mask_cache.shape != shape:
            self.mask_cache = rasterize_masks([mask.points for mask in self.masks if mask.enabled], shape)
        return self.mask_cache

    def valid_pixels(self, shape, masks=True):
        """Boolean image of the pixels not masked, used by the contrast enhancement (None: all of them)."""
        masked = self.masked_pixels(shape) if masks else None
        return None if masked is None else ~masked

    def apply_masks(self, A):
        """Set the masked pixels of an image to 0, if the masks are in the recipe."""
        masked = self.masked_pixels(A.shape)
        if masked is None:
            return A
        A = A.copy()
        A[masked] = 0
        return A

    def read_image(self, filename, masks=True):
        """Read an image, with the masks and the contrast enhancement if it is applied before the POD. The masks
        are applied first, so that the masked pixels don't take part in the POD, and the contrast enhancement
        ignores them."""
        A = imread(filename)
        if masks:
            A = self.apply_masks(A)
        if self.contrast_on() and self.contrast.stage == "pre":
            A = self.enhance_contrast(A, self.valid_pixels(A.shape, masks))
            if masks:
                A = self.apply_masks(A)
        return A

    def load_modes(self, NMod):
        return [load_npz(self.mode_file(mod_i), "PHI_bloc", "TCoeff_bloc") for mod_i in range(1, NMod + 1)]

    def process_image(self, filename, index=0, modes=(), masks=True):
        """The recipe applied to an image: masks, contrast enhancement before POD, removal of the modes (with the
        temporal coefficients of the training image index) and contrast enhancement after it. masks=False: the
        masks aren't applied (while they are edited). Returns the image and the background removed (None
        without background removal)."""
        A = self.read_image(filename, masks)
        img, background = A, None
        if modes:
            if A.size != modes[0]["PHI_bloc"].size or not np.issubdtype(A.dtype, np.number):
                raise ValueError("Target image not compatible.")
            tmp_im = A.astype(np.float32).ravel()
            for data in modes:
                tmp_im -= data["TCoeff_bloc"][index] * data["PHI_bloc"]
            # Create a variable with the same bitdepth as the original image
            img = cast_like(tmp_im.reshape(A.shape), A)
            background = cast_like(A.astype(np.float64) - img, A)
        if self.contrast_on() and self.contrast.stage == "post":
            img = self.enhance_contrast(img, self.valid_pixels(img.shape, masks))
        # The values of the masked pixels after the contrast enhancement are meaningless
        return (self.apply_masks(img) if masks else img), background

    def recipe_steps(self, n_modes_removed=0, detailed=False, editing=False):
        """Steps of the recipe switched on, in the order they are applied."""
        steps = []
        if editing:
            steps.append("masks not applied while editing")
        elif self.masks_active():
            steps.append(f"masks ({sum(m.enabled for m in self.masks)})")
        contrast = "contrast enhancement" + (f": {self.contrast_description()}" if detailed else "")
        if self.contrast_on() and self.contrast.stage == "pre":
            steps.append(contrast + (" (before POD)" if detailed else ""))
        if n_modes_removed:
            steps.append(f"background removal ({n_modes_removed} modes)")
        if self.contrast_on() and self.contrast.stage == "post":
            steps.append(contrast)
        return steps

    def project_name(self):
        """Name of the project: project, or the name of the folder of the training images."""
        return self.project or default_project_name(self.path_in)

    def export_images(self, output_folder, n_modes_removed=0, folder=None, files=None, save_background=False):
        """Save the images processed with the recipe: masks and contrast enhancement if they are enabled, and
        subtraction of the first n_modes_removed modes (0: no background removal). The images are the training
        images, or the files of a folder (at most as many as the training images to remove their background:
        the background of the image i is reconstructed with the temporal coefficients of the training image i).
        They are named [project]_PreProcessing_Image_[number].[format], the number being the one of the image in
        the set and the format export_format (that of the source image with "same"). save_background: also save the
        backgrounds removed ([project]_PreProcessing_Background_[number]). Returns the files written."""
        if folder is None and files is None:
            folder, files = self.path_in, self.file_in
        elif files is None:
            folder, files = self.find_images(folder)
        if self.recipe_empty(n_modes_removed):
            raise ValueError("The recipe is empty: enable the masks or the contrast enhancement, or remove modes")
        fmt = self.export_format or DEFAULT_EXPORT_FORMAT
        if files:
            check_export_depth(fmt, imread(os.path.join(folder, files[0])).dtype, files[0])
        NImg = len(files)
        if n_modes_removed:
            if n_modes_removed > self.NMod_saved:
                raise ValueError(f"{n_modes_removed} modes to remove, {self.NMod_saved} modes evaluated")
            # The background of the image i is reconstructed with the temporal
            # coefficients of the training image i
            if NImg > self.NImg:
                raise ValueError(f"The target contains {NImg} images, more than the {self.NImg} training images "
                                 "used to evaluate the modes.")
        project = self.project_name()
        self.path_out = os.path.normpath(output_folder)
        # Modes to remove, loaded once for all the images
        modes = self.load_modes(n_modes_removed)

        written = []
        with self.progress("Export images") as wh:
            os.makedirs(self.path_out, exist_ok=True)
            # The description of the images exported before is removed first: if the export is interrupted, the
            # images of the folder aren't taken for images processed with a recipe
            manifest = os.path.join(self.path_out, MANIFEST_FILE)
            if os.path.isfile(manifest):
                os.remove(manifest)
            def export(item):
                im_i, name = item
                img, background = self.process_image(os.path.join(folder, name), im_i, modes)
                extension = export_extension(fmt, name)
                # Save single background if requested
                if save_background and background is not None:
                    imwrite(background, os.path.join(self.path_out, export_name(project, TASK, "Background", im_i + 1,
                                                                                extension, NImg)))
                path = os.path.join(self.path_out, export_name(project, TASK, "Image", im_i + 1, extension, NImg))
                imwrite(img, path)
                return path

            for im_i, path in enumerate(ordered_map(export, enumerate(files))):
                wh((im_i + 1) / NImg, f"Saving image {im_i + 1} of {NImg}...")
                written.append(path)
            steps = " → ".join(self.recipe_steps(n_modes_removed, detailed=True))
            with open(manifest, "w", encoding="utf-8") as f:
                # The folders relative to the one of the project (measurement_folder: relative to this one)
                json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": steps,
                           "recipe": self.recipe_record(n_modes_removed),
                           "measurement_folder": relative(self.path_in, self.path_out),
                           "source_folder": relative(folder, self.path_in), "source_files": list(files),
                           "format": fmt, "files": [os.path.basename(p) for p in written]}, f, indent=1,
                          ensure_ascii=False)

        self.update_status(" ")
        self.update_status(f"{NImg} images exported in {self.path_out} ({os.path.basename(written[0])} ...). "
                           "Recipe: " + steps)
        return written

    def recipe_record(self, n_modes_removed=0):
        """The recipe as it is saved with the images exported (MANIFEST_FILE): the points of the masks enabled,
        the contrast enhancement and the background removal, with the training images whose modes are removed.
        The same record gives the same images (the training images: those of the project, whatever its folder)."""
        contrast = None
        if self.contrast_on():
            c = self.contrast
            contrast = {"filters": list(c.filters), "stage": c.stage,
                        **{p: getattr(c, p) for f in c.filters for p in FILTER_PARAMETERS[f]}}
        background_removal = None
        if n_modes_removed:
            background_removal = {"modes_removed": n_modes_removed,
                                  "training_images": hashlib.sha1("\n".join(self.file_in).encode()).hexdigest()}
        record = {"masks": [m.points.tolist() for m in self.masks if m.enabled] if self.masks_active() else [],
                  "contrast": contrast, "background_removal": background_removal}
        # As read from the file: lists and floats
        return json.loads(json.dumps(record))

    def export_folder(self):
        """Folder of the images exported for the PIV: path_export, or the subfolder PreProcessing of the image
        folder."""
        return self.path_export or os.path.join(self.path_in, PREPROCESSING_FOLDER)

    def recipe_empty(self, n_modes_removed=0):
        return not (self.masks_active() or self.contrast_on() or n_modes_removed)

    def preprocessed_images(self, n_modes_removed=0, folders=None):
        """Paths of the training images already exported with the recipe (masks and contrast enhancement if they
        are enabled, and removal of n_modes_removed modes) in the format export_format: those of the first folder whose description matches,
        among the folders given, or in the export folder (export_folder). None if there is none."""
        if folders is None:
            folders = [self.export_folder()]
        record = self.recipe_record(n_modes_removed)
        for folder in folders:
            m = read_manifest(folder)
            if m is None:
                continue
            # The source folder relative to the folder of the project (absolute in the previous versions)
            project = absolute(m.get("measurement_folder"), folder) or folder
            source = absolute(m.get("source_folder"), project) or ""
            if (comparable_recipe(m.get("recipe")) != record or m.get("source_files") != self.file_in
                    or not same_path(source, self.path_in)):
                continue
            # In the format chosen (the previous versions kept the one of the source images)
            if (m.get("format") or DEFAULT_EXPORT_FORMAT) != (self.export_format or DEFAULT_EXPORT_FORMAT):
                continue
            paths = [os.path.join(folder, name) for name in m.get("files", [])]
            if len(paths) == self.NImg and all(os.path.isfile(p) for p in paths):
                return paths
        return None

    def piv_images(self, n_modes_removed=0):
        """Images of the PIV: the training images processed with the recipe (removal of n_modes_removed modes,
        evaluated if needed). The images already exported with the recipe are used (preprocessed_images);
        otherwise they are exported in the export folder (export_folder: the subfolder PreProcessing of the image
        folder by default).
        Without recipe: the original images. Returns their paths."""
        if self.recipe_empty(n_modes_removed):
            return self.images
        paths = self.preprocessed_images(n_modes_removed)
        if paths is not None:
            self.update_status(f"Images pre-processed with the recipe found in {os.path.dirname(paths[0])}")
            return paths
        if n_modes_removed > self.NMod_saved and self.evaluate_modes(n_modes_removed) is None:
            return None
        return self.export_images(self.export_folder(), n_modes_removed)

    def remove_background(self, output_folder, n_modes_removed, n_modes_evaluated=None, folder=None, pattern=None,
                          files=None, save_background=False):
        """Remove the background of the training images, or of other images of the same size (folder and pattern,
        or files), with the recipe: masks and contrast enhancement if they are enabled, and subtraction of the
        first n_modes_removed POD modes. The spectrum and n_modes_evaluated modes (at least n_modes_removed) are
        evaluated if needed. Returns the files written."""
        n_modes = max(n_modes_evaluated or 0, n_modes_removed)
        if n_modes > self.NMod_saved and self.evaluate_modes(n_modes) is None:
            return []
        if folder is not None or files is not None:
            folder, files = self.find_images(folder, pattern, files)
        return self.export_images(output_folder, n_modes_removed, folder, files, save_background)

    ############################################################################
    # Temporary files

    def temp_file(self, bloc_i):
        return os.path.join(self.pathtemp, f"Temp_{bloc_i}.npz")

    def mode_file(self, mod_i):
        return os.path.join(self.pathtemp, f"Modes_{mod_i}.npz")

    def bloc_index(self, bloc_i):
        """Indices of the images saved in a bloc."""
        return slice((bloc_i - 1) * self.MaxImg, min(bloc_i * self.MaxImg, self.NImg))

    def load_bloc(self, bloc_i):
        return load_npz(self.temp_file(bloc_i), "M_bloc")["M_bloc"]

    def delete_files(self, pattern):
        """Delete the files of the temporary folder matching the pattern. Returns the number of files deleted."""
        deleted = 0
        if not os.path.isdir(self.pathtemp):
            return 0
        for name in list_files(self.pathtemp, pattern):
            tmp = os.path.join(self.pathtemp, name)
            # Check that the file removed were created by this software
            try:
                with np.load(tmp) as data:
                    created_here = "bgremkey" in data.files and str(data["bgremkey"]) == BGREMKEY
                if created_here:
                    os.remove(tmp)
                    deleted += 1
            except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                pass
        return deleted

    def clean_temporary_files(self):
        """Delete the temporary files created by the software. Returns the numbers of bloc and mode files
        deleted."""
        return self.delete_files("Temp_*.npz"), self.delete_files("Modes_*.npz")

    def clear_cache(self):
        """Free the memory and the temporary folder: the images loaded (in memory or in temporary files) and the
        modes stored in temporary files are deleted; the spectrum is kept. The modes are evaluated again when
        they are needed (or restored by load_pod). Returns the numbers of bloc and mode files deleted."""
        deleted = self.clean_temporary_files()
        self.M_bloc = None
        self.mask_cache = None
        self.Check_imload = self.Check_modproc = False
        self.NMod_saved = 0
        return deleted
