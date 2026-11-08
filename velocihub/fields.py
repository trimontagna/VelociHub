"""Vector fields of the PIV and the base of their analyses (postprocessing.py); the page PIV reads the vector fields
too.

The vector fields are those saved by the PIV (vectors.npz, in pixels: converted into mm and m/s with the camera
settings) or a folder of DynamicStudio CSV exports (load_vectors, load_csv_folder). The analyses (FieldAnalysis)
share the statistical outlier filtering, the statistics, the scale space and the reference quantities (u_ref,
t_flow, L_ref) and the figures (maps, LaTeX labels, export).

Scale space: the vector fields in pixels (load_vectors without camera) are shown in the sensor space: positions in
the image (px, from its top-left corner), displacements (px) instead of velocities, times as pairs (Pair k/N, N
pairs), no reference quantity. With the camera settings, in the physical space: positions in mm (or over L_ref),
velocities in m/s (or over u_ref), times in s (or in flow times t_flow).
"""

import glob
import json
import os
import re
import shutil
import warnings
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields as dataclass_fields

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.quiver import Quiver
from matplotlib.text import Text

from .project import absolute, export_name

# Folder of the results of the PIV (subfolder of the image folder) and the file of its vector fields; the folder of
# the results of the PTV (TracTrac), and the file of its particles tracked (beside vectors.npz)
PIV_FOLDER = "PIV"
VECTORS_FILE = "vectors.npz"
PTV_FOLDER = "PTV"
TRACKS_FILE = "tracks.npz"
# Size of the figures exported (inches): the maps, and the others (FieldAnalysis.FIGSIZE); their resolution (dots
# per inch)
MAP_SIZE = (10, 5.6)
EXPORT_DPI = 130
# Videos of the instantaneous figures: frames per second by default, resolution (dots per inch: 1000 x 560 pixels
# for the maps) and format: "gif" (by default; 256 colours per frame) or "mp4" (H.264, with ffmpeg)
VIDEO_FPS = 10
VIDEO_DPI = 100
VIDEO_FORMATS = ("gif", "mp4")
VIDEO_FORMAT = "gif"
# Frames measured before a video is written for the scales of all its frames (Scales: evenly spaced, the first and the
# last ones among them)
VIDEO_SCALE_FRAMES = 25
# The labels of the figures are in LaTeX (Matplotlib mathtext, no LaTeX installation needed), its symbols in
# Computer Modern
MATH_FONT = {"mathtext.fontset": "cm"}
# Divider of the comparisons with a wiper (Wiper)
DIVIDER_COLOR = "#ffcc00"
LREF, UREF, TFLOW = r"L_\mathrm{ref}", r"u_\mathrm{ref}", r"t_\mathrm{flow}"
POSITION_SYMBOLS = {"x": "x", "x global": r"x_\mathrm{global}", "y": "y", "y global": r"y_\mathrm{global}"}

# The figures in the sensor space: the velocities are the displacements of the pairs (LaTeX and words replaced in
# all their texts, in this order)
SENSOR_TEXTS = [(r"\mathbf{u}", "d"), (r"\overline{u'v'}", r"\overline{d_x'd_y'}"), (r"{u'}^2", r"{d_x'}^2"),
                (r"{v'}^2", r"{d_y'}^2"), (r"\overline{u}", r"\overline{d_x}"), (r"\overline{v}", r"\overline{d_y}"),
                (r"\sigma_u", r"\sigma_{d_x}"), (r"\sigma_v", r"\sigma_{d_y}"), (r"\gamma_u", r"\gamma_{d_x}"),
                (r"\gamma_v", r"\gamma_{d_y}"), (r"\beta_u", r"\beta_{d_x}"), (r"\beta_v", r"\beta_{d_y}"),
                ("$u$", "$d_x$"), ("$v$", "$d_y$"), (r"u_\mathrm{ref}", r"d_\mathrm{ref}"), ("(1/s)", "(1/pair)"),
                ("Instantaneous velocity", "Instantaneous displacement"), ("Mean velocity", "Mean displacement"),
                ("reference velocity", "reference displacement"), ("Velocity along", "Displacement along")]

# DynamicStudio CSV exports: header lines before the names of the columns, and the columns read
CSV_SKIPROWS = 8
# The time of the first image of a map (ms) at the end of the name of a DynamicStudio CSV export (csv_times)
CSV_TIME = re.compile(r"-(\d+(?:\.\d*)?)$")
CSV_COLUMNS = {"i": "X[-]", "j": "Y[-]", "x": "X (m)[m]", "y": "Y (m)[m]", "U": "U[m/s]", "V": "V[m/s]",
               "status": "Status[-]", "peak_ratio": "Peak height ratio[-]"}


@contextmanager
def quiet():
    """Statistics without warning for the grid points without values (always NaN, e.g. masked)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        yield


def num2str(x):
    return str(int(x)) if float(x).is_integer() else f"{x:.4g}"


def significant(x, digits=3, trim=False):
    """A number with at most digits significant digits, without exponent: 12345 -> 12300, 8.066 -> 8.07,
    0.02214 -> 0.0221 (digits = 3); trim: without the zeros ending the decimals (0.500 -> 0.5, 1.00 -> 1)."""
    x = float(x)
    if not np.isfinite(x) or x == 0:
        return f"{x:g}"
    r = float(f"{x:.{digits}g}")
    decimals = digits - 1 - int(np.floor(np.log10(abs(r))))
    text = f"{r:.{max(decimals, 0)}f}"
    return text.rstrip("0").rstrip(".") if trim and "." in text else text


def sig2(x):
    """A non-dimensional number (time in flow times, position over L_ref) with 2 significant digits, without
    exponent: 123 -> 120, 8.06 -> 8.1, 0.0221 -> 0.022."""
    return significant(x, 2)


################################################################################
# Vector fields


@dataclass(eq=False)
class VectorFields:
    """Vector fields on a regular grid: arrays (map, row, column), the rows along y (upwards) and the columns
    along x (to the right).

    x, y: positions (mm), from the origin of the axes of the camera (piv.Camera.x_shift, y_shift: the bottom-left
    corner of the images by default) plus the offsets of the field of view, y upwards; U, V: velocities (m/s), NaN
    where the vector isn't valid; u_px, v_px: displacements (px), None if unknown; valid: valid vectors; status: grid
    points without flow (0: flow, 1: no particles, 2: masked); t: time of each map (s); peak_ratio: correlation peak
    ratio (None if unknown); dt: time between the images of a pair (s); scale: mm per pixel horizontally (the
    horizontal pixel pitch); scale_y: vertically (None: scale, square pixels); name: name of the
    measurement; folder: folder of the measurement (of the images; the post-processing is exported in its
    subfolder PostProcessing); source: file or folder read; settings: those of the PIV; piv_valid: the valid
    vectors of the PIV, which the quality of the PIV uses (valid unless the fields are a regression of them);
    particles: the particles of each vector of a PTV (None otherwise); tracks: the file of its particles tracked
    ("" if none)."""

    x: np.ndarray
    y: np.ndarray
    U: np.ndarray
    V: np.ndarray
    valid: np.ndarray
    status: np.ndarray
    t: np.ndarray
    u_px: np.ndarray = None
    v_px: np.ndarray = None
    peak_ratio: np.ndarray = None
    dt: float = None
    scale: float = None
    name: str = ""
    folder: str = ""
    source: str = ""
    settings: dict = field(default_factory=dict)
    piv_valid: np.ndarray = None
    particles: np.ndarray = None
    tracks: str = ""
    scale_y: float = None

    def __post_init__(self):
        # Rows from the bottom to the top, columns from the left to the right
        rows = slice(None, None, -1) if self.y[0, 0] > self.y[-1, 0] else slice(None)
        cols = slice(None, None, -1) if self.x[0, 0] > self.x[0, -1] else slice(None)
        self.x, self.y, self.status = self.x[rows, cols], self.y[rows, cols], self.status[rows, cols]
        for name in ("U", "V", "valid", "u_px", "v_px", "peak_ratio", "piv_valid", "particles"):
            if getattr(self, name) is not None:
                setattr(self, name, getattr(self, name)[:, rows, cols])
        if self.piv_valid is None:
            self.piv_valid = self.valid
        if self.scale_y is None:
            self.scale_y = self.scale

    n = property(lambda self: self.U.shape[0])
    shape = property(lambda self: self.U.shape[1:])

    def describe(self):
        text = f"{self.n} maps of {self.shape[1]} x {self.shape[0]} vectors"
        if self.dt:
            text += f", dt = {self.dt * 1000:.4g} ms"
        if self.scale and self.scale_y and abs(self.scale_y - self.scale) > 1e-12 * self.scale:
            text += f", {self.scale:.5f} x {self.scale_y:.5f} mm/px"
        elif self.scale:
            text += f", {self.scale:.5f} mm/px"
        return text


def pixel_positions(z, settings):
    """Positions of the vectors of a file vectors.npz in the images (px from their top-left corner): saved by the
    PIV, or from the positions in mm of the files of the previous versions (y from the bottom of the images) and
    their scale. None if unknown."""
    if "x_px" in z.files:
        return z["x_px"], z["y_px"]
    scale, size = settings.get("scale_mm_per_px"), settings.get("image_px")
    if "x_mm" in z.files and scale and size:
        return z["x_mm"] / scale, size[1] - z["y_mm"] / scale
    return None


def load_vectors(path, dt=None, scale=None, camera=None, pixels=False):
    """Vector fields of a file vectors.npz saved by the PIV (or of a folder containing one), or of a folder of
    DynamicStudio CSV exports: for these, dt (time between the images of a pair, s) and scale (mm/px) give the
    displacements in pixels (peak locking).

    The PIV saves the vector fields in pixels: camera (piv.Camera) converts them into mm and m/s (the pixel pitches
    are the width and the height of its field of view divided by those of the images; the origin of the axes is its
    x_shift, y_shift). Without camera, they are in pixels (x and y in px from the bottom-left corner of the images,
    y upwards, U and V in px per pair, settings["units"] = "px"), unless the file has the velocities (PIV with a
    camera, or the files of the previous versions, whose y = 0 is at the middle of the height of the images).
    pixels: in pixels whatever the file (the sensor space), when the positions of the vectors in the images are
    known (pixel_positions)."""
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, VECTORS_FILE)):
            path = os.path.join(path, VECTORS_FILE)
        else:
            return load_csv_folder(path, dt, scale)
    with np.load(path) as z:
        settings = json.loads(str(z["settings"])) if "settings" in z.files else {}
        status = z["status"]
        if status.ndim == 3:  # PIV_PAIRS_batch.py: the status of each map
            status = status.max(axis=0)
        valid = (z["info"] == 1) & (status == 0)
        # Fields regressed after the PIV (SPICY): the valid vectors of the PIV kept for its quality
        piv_valid = (z["raw_info"] == 1) & (status == 0) if "raw_info" in z.files else valid
        u_px, v_px = z["u_px"], z["v_px"]
        positions = pixel_positions(z, settings)
        if positions is not None and (pixels or "x_px" in z.files and (camera is not None or "U" not in z.files)):
            x_px, y_px = positions
            width, height = settings["image_px"]
            first = np.array([a for a, _ in settings.get("pairs") or []] or range(len(u_px)), np.float64)
            if camera is not None and not pixels:
                x, y = camera.positions(x_px, y_px, width, height)
                U, V = camera.velocities(u_px, v_px, width, height)
                t = first / camera.frame_rate_hz
                settings = {**settings, "camera": asdict(camera), "scale_mm_per_px": camera.scale(width),
                            "scale_y_mm_per_px": camera.scale_y(width, height), "dt_s": camera.dt}
            else:
                # The sensor space: px from the bottom-left corner of the images, y upwards
                x, y, U, V, t = x_px, height - y_px, u_px, v_px, first
                settings = {**settings, "units": "px", "scale_mm_per_px": 1.0, "dt_s": 1.0}
        else:
            x = z["x_global_mm"] if "x_global_mm" in z.files else z["x_mm"]
            y, U, V, t = z["y_mm"], z["U"], z["V"], z["t_s"]
            # The previous versions had y = 0 at the bottom of the images
            if "y_px" not in z.files and settings.get("image_px") and settings.get("scale_mm_per_px"):
                y = y - settings["image_px"][1] / 2 * settings["scale_mm_per_px"]
        nan = np.float32(np.nan)
        # No peak ratio for a PTV, but the particles of each vector
        fields = VectorFields(x=x, y=y, U=np.where(valid, U, nan), V=np.where(valid, V, nan),
                              valid=valid, status=status, t=t,
                              u_px=np.where(valid, u_px, nan), v_px=np.where(valid, v_px, nan),
                              peak_ratio=np.where(piv_valid, z["peak_ratio"], nan) if "peak_ratio" in z.files else None,
                              dt=settings.get("dt_s"), scale=settings.get("scale_mm_per_px"),
                              scale_y=settings.get("scale_y_mm_per_px"),
                              source=os.path.abspath(path), settings=settings, piv_valid=piv_valid,
                              particles=z["ptv_particles"] if "ptv_particles" in z.files else None)
    results = os.path.dirname(os.path.abspath(path))
    fields.folder = measurement_folder(path, settings)
    fields.name = os.path.basename(fields.folder.rstrip("/\\"))
    tracks = os.path.join(results, settings["tracks_file"]) if settings.get("tracks_file") else ""
    fields.tracks = tracks if tracks and os.path.isfile(tracks) else ""
    return fields


def measurement_folder(path, settings):
    """Folder of the measurement of a file of vector fields (vectors.npz) with its settings: the one of the images,
    whose subfolder PIV (PTV) contains the results (saved relative to the folder of the results; absolute in the
    previous versions)."""
    results = os.path.dirname(os.path.abspath(path))
    return absolute(settings.get("measurement_folder"), results) or (
        os.path.dirname(results) if os.path.basename(results) in (PIV_FOLDER, PTV_FOLDER) else results)


def csv_times(files):
    """Times of the maps (s) given by the names of DynamicStudio CSV exports, the time of their first image in ms
    after the last dash of the name (e.g. Run.000001-6.6672.csv: 6.6672 ms); None unless every name has one and they
    increase."""
    times = []
    for name in files:
        m = CSV_TIME.search(os.path.splitext(os.path.basename(name))[0])
        if m is None:
            return None
        times.append(float(m[1]) / 1000.0)
    times = np.asarray(times)
    return times if times.size and np.all(np.diff(times) > 0) else None


def load_csv_folder(folder, dt=None, scale=None, progress=None, frame_rate_hz=None):
    """Vector fields of a folder of DynamicStudio CSV exports (one file per map, sorted by name; 8 lines before
    the names of the columns): X[-], Y[-] (grid indices), X (m)[m], Y (m)[m], U[m/s], V[m/s], Status[-] (0 =
    valid) and Peak height ratio[-]. dt (time between the two images of a pair, s) and scale (mm/px) give the
    displacements in pixels. The positions are those of the export (mm, from the origin of its calibration). The
    time of each map: the one of its name (csv_times), else its number over frame_rate_hz (the frame rate of the
    maps), else its number times dt (time-resolved: each map from the image of the previous one, as a time-resolved
    PIV is computed; the time between double-frame maps can only come from the names or the frame rate)."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise ValueError(f"No vectors.npz nor CSV files in {folder}")
    with open(files[0], encoding="utf-8", errors="replace") as f:
        header = [f.readline() for _ in range(CSV_SKIPROWS + 1)][-1]
    names = [h.strip().strip('"') for h in header.strip().split(",")]
    missing = [c for k, c in CSV_COLUMNS.items() if c not in names and k != "peak_ratio"]
    if missing:
        raise ValueError(f"Columns not found in {files[0]}: {', '.join(missing)}")
    keys = [k for k, c in CSV_COLUMNS.items() if c in names]
    use = [names.index(CSV_COLUMNS[k]) for k in keys]
    maps = []
    for n, name in enumerate(files):
        maps.append(np.loadtxt(name, delimiter=",", skiprows=CSV_SKIPROWS + 1, usecols=use, ndmin=2))
        if progress is not None:
            progress((n + 1) / len(files), f"Reading {n + 1} of {len(files)} files...")
    maps = np.stack(maps)  # (map, vector, column), the vectors in the same order in every file
    column = {k: c for c, k in enumerate(keys)}
    i, j = maps[0, :, column["i"]].astype(int), maps[0, :, column["j"]].astype(int)
    G = np.full((len(files), j.max() + 1, i.max() + 1, len(keys)), np.nan)
    G[:, j, i] = maps
    x, y = ((G[0, ..., column[k]] * 1000).astype(np.float32) for k in ("x", "y"))
    stack = {k: G[..., column[k]].astype(np.float32) for k in keys if k not in ("i", "j", "x", "y")}
    valid = stack["status"] == 0
    status = np.where(valid.mean(0) < 0.1, 1, 0).astype(np.int16)
    valid &= status == 0
    nan = np.float32(np.nan)
    U, V = np.where(valid, stack["U"], nan), np.where(valid, stack["V"], nan)
    u_px = v_px = None
    if dt and scale:
        px_per_ms = dt / (scale / 1000)
        u_px, v_px = U * px_per_ms, V * px_per_ms
    peak = np.where(valid, stack["peak_ratio"], nan) if "peak_ratio" in stack else None
    t = csv_times(files)
    if t is None:
        t = np.arange(len(files)) / frame_rate_hz if frame_rate_hz else np.arange(len(files)) * (dt or 1.0)
    return VectorFields(x=x, y=y, U=U, V=V, valid=valid, status=status, t=t,
                        u_px=u_px, v_px=v_px, peak_ratio=peak, dt=dt, scale=scale,
                        name=os.path.basename(os.path.abspath(folder)), folder=os.path.abspath(folder),
                        source=os.path.abspath(folder),
                        settings={"source": "DynamicStudio CSV export", "files": len(files)})


def count_maps(path):
    """The number of maps of a file vectors.npz (from the header of its array, without reading it)."""
    with zipfile.ZipFile(path) as z, z.open("u_px.npy") as f:
        version = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        return int(read(f)[0][0])


def ptv_results(folder):
    """The vector fields of a folder (its vectors.npz) are those of a PTV (the particles of their cells known,
    whatever the name of the folder); False without vectors.npz."""
    path = os.path.join(folder, VECTORS_FILE) if folder else ""
    try:
        with zipfile.ZipFile(path) as z:
            return "ptv_particles.npy" in z.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def nanmedian(A, axis):
    """np.nanmedian(A, axis), vectorised (np.nanmedian goes point by point): one sort along the axis (NaN last),
    the middle values of the ones that aren't NaN; NaN where there are none."""
    S = np.sort(A, axis=axis)
    n = np.expand_dims((~np.isnan(A)).sum(axis), axis)
    low = np.take_along_axis(S, np.maximum((n - 1) // 2, 0), axis)
    high = np.take_along_axis(S, np.maximum(n // 2, 0), axis)
    with np.errstate(invalid="ignore"):
        return np.squeeze(np.where(n > 0, (low + high) / 2, np.nan), axis).astype(S.dtype, copy=False)


def outliers(A, k):
    """Outliers of maps A (map, row, column): the values further than k standard deviations from the mean of the
    other values of their grid point (leave-one-out: the mean and the standard deviation, N - 2 degrees of freedom,
    of the N - 1 others), NaN ignored; none at a grid point with fewer than 4 values."""
    with quiet():
        finite = np.isfinite(A)
        n = finite.sum(0)
        mean = np.nanmean(A, 0, dtype=np.float64)
        d = np.where(finite, A - mean, 0.0)  # About the mean: no cancellation
        s2 = (d * d).sum(0)
        m = n - 1  # The others
        mean_others = -d / np.maximum(m, 1)  # Their mean, about the mean of all (their sum is -d)
        var_others = (s2 - d * d - m * mean_others ** 2) / np.maximum(m - 1, 1)
        far = np.abs(d - mean_others) > k * np.sqrt(np.maximum(var_others, 0.0))
    return finite & far & (m >= 3)


def nanstd1(A):
    """Sample standard deviation (N - 1) over the maps (axis 0), NaN ignored; NaN with fewer than 2 values."""
    with quiet():
        return np.nanstd(A, 0, dtype=np.float64, ddof=1)


def nancov1(A, B, mean_A, mean_B):
    """Sample covariance (N - 1) of A and B over the maps (axis 0) about their means, NaN ignored; NaN with fewer
    than 2 values."""
    with quiet():
        prod = (A - mean_A) * (B - mean_B)
        n = np.isfinite(prod).sum(0)
        return np.where(n >= 2, np.nansum(prod, 0, dtype=np.float64) / np.maximum(n - 1, 1), np.nan)


def map_indices(indices, n):
    """Map indices among n maps, the negative ones from the end (-1: the last map), as Python indexes; those out of
    the maps left out."""
    return [int(k) + n if k < 0 else int(k) for k in indices if -n <= k < n]


def ffmpeg_available():
    """ffmpeg found for the MP4 videos: on the PATH, else the one of the package imageio-ffmpeg (pip install
    imageio-ffmpeg), given to Matplotlib."""
    from matplotlib import animation

    if not animation.writers.is_available("ffmpeg") and not shutil.which("ffmpeg"):
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            return False
    return animation.writers.is_available("ffmpeg")


class GifWriter:
    """Animated GIF written by Pillow, each frame reduced to its 256 colours when it is grabbed (a quarter of the
    memory of the frames in RGBA until the file is written); the frame time in hundredths of a second (24 frames/s:
    0.04 s, 25 frames/s), as GIF stores it."""

    def __init__(self, fps):
        self.duration = 10 * max(2, round(100 / fps))  # ms; the players show at most 50 frames/s

    def setup(self, fig, path, dpi):
        self.fig, self.path, self.dpi, self.frames = fig, path, dpi, []

    def grab_frame(self):
        from io import BytesIO
        from PIL import Image

        buf = BytesIO()
        self.fig.savefig(buf, format="rgba", dpi=self.dpi)
        size = (int(self.fig.get_figwidth() * self.dpi), int(self.fig.get_figheight() * self.dpi))
        frame = Image.frombuffer("RGBA", size, buf.getbuffer(), "raw", "RGBA", 0, 1).convert("RGB")
        self.frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE))

    def finish(self):
        self.frames[0].save(self.path, save_all=True, append_images=self.frames[1:], duration=self.duration, loop=0)
        self.frames = []


class Wiper:
    """Comparison of two layers of artists of an axes, either side of a vertical divider (as the comparison of the
    images before and after the pre-processing in the interface): the artists of the left layer are clipped left
    of the divider, those of the right layer right of it, and labels at the top of the divider name them. The divider
    is at a fraction of the width of the axes, so that it follows the zoom and the pan."""

    def __init__(self, ax, left, right, labels=("before", "after")):
        self.ax, self.layers, self.labels = ax, (list(left), list(right)), labels
        self.divider = []

    def place(self, fraction):
        """Divider at this fraction of the width of the axes (0: the right layer only, 1: the left one only)."""
        self.clear_divider()
        ax, f = self.ax, min(max(float(fraction), 0.0), 1.0)
        for artists, (a, b) in zip(self.layers, ((0.0, f), (f, 1.0))):
            # A polygon (Matplotlib turns a Rectangle into a clip box, which would replace the clipping by the axes)
            clip = MplPolygon([[a, 0], [b, 0], [b, 1], [a, 1]], transform=ax.transAxes)
            for artist in artists:
                artist.set_clip_path(clip)
                artist.set_visible(b > a)
        if 0 < f < 1:
            self.divider.append(ax.add_artist(Line2D([f, f], [0, 1], transform=ax.transAxes,
                                                     color=DIVIDER_COLOR, linewidth=1.5)))
        box = dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none")
        for text, ha in ((f"◀ {self.labels[0]} ", "right"), (f" {self.labels[1]} ▶", "left")):
            # The labels may be LaTeX symbols: in the math font of the figures (Computer Modern)
            self.divider.append(ax.text(f, 0.98, text, transform=ax.transAxes, ha=ha, va="top", color="white",
                                        fontsize="small", bbox=box, clip_on=True, math_fontfamily="cm"))

    def clear_divider(self):
        for artist in self.divider:
            artist.remove()
        self.divider = []


class Scales:
    """The same scales in all the frames of a video: the colour ranges of the maps and the limits of the axes (not
    those of the colour bars) of the frames measured (measure(fig), once drawn), each the union over them, and the
    scale of the arrows, the largest of them (the shortest arrows), applied to each frame drawn (apply(fig)). Frames
    whose maps, axes and arrows aren't those of the first frame measured keep theirs."""

    def __init__(self):
        self.layout, self.clims, self.lims, self.arrows = None, [], [], []

    @staticmethod
    def parts(fig):
        maps = FieldAnalysis.colour_maps(fig)
        bars = {a.colorbar.ax for a in maps}
        axes = [ax for ax in fig.axes if ax not in bars]
        return maps, axes, [c for ax in axes for c in ax.collections if isinstance(c, Quiver)]

    @staticmethod
    def arrow_scale(q):
        """The scale of the arrows of a quiver: the one set, else the one Matplotlib gives them (arrows in units of
        the width of the axes: 1.8 x their mean length x max(10, sqrt(N)) arrow widths); None if unknown."""
        if q.scale is not None:
            return float(q.scale)
        if q.units != "width" or q.scale_units is not None:
            return None
        # As Matplotlib does (Quiver._make_verts)
        a = np.abs(np.ma.asarray(q.U) + 1j * np.ma.asarray(q.V))
        mean = a[~q.Umask].mean() if q.Umask is not np.ma.nomask else a.mean()
        mean = float(np.ma.filled(mean, np.nan))
        return 1.8 * mean * max(10, np.sqrt(q.N)) if np.isfinite(mean) and mean > 0 else None

    @staticmethod
    def union(range_, values):
        values = [float(v) for v in values if v is not None and np.isfinite(v)]
        if not values:
            return range_
        low, high = min(values), max(values)
        return (low, high) if range_ is None else (min(range_[0], low), max(range_[1], high))

    def measure(self, fig):
        maps, axes, arrows = self.parts(fig)
        layout = (len(maps), len(axes), len(arrows))
        # A frame without axes (not available: a message) isn't measured
        if not axes:
            return
        if self.layout is None:
            self.layout, self.clims, self.lims = layout, [None] * len(maps), [[None, None] for _ in axes]
            self.arrows = [None] * len(arrows)
        elif layout != self.layout:
            return
        for i, q in enumerate(arrows):
            scale = self.arrow_scale(q)
            if scale is not None:
                self.arrows[i] = max(scale, self.arrows[i] or 0)
        for i, a in enumerate(maps):
            # A uniform field: its value (its colour bar widens its range around it)
            v = np.ma.filled(np.ma.asarray(a.get_array(), np.float64), np.nan)
            v = v[np.isfinite(v)]
            uniform = v.size and v.min() == v.max()
            self.clims[i] = self.union(self.clims[i], (v.min(), v.max()) if uniform else a.get_clim())
        for lims, ax in zip(self.lims, axes):
            lims[0], lims[1] = self.union(lims[0], ax.get_xlim()), self.union(lims[1], ax.get_ylim())

    def apply(self, fig):
        maps, axes, arrows = self.parts(fig)
        if (len(maps), len(axes), len(arrows)) != self.layout:
            return
        for q, scale in zip(arrows, self.arrows):
            if scale is not None:
                q.scale = scale
        for a, clim in zip(maps, self.clims):
            if clim is not None:
                a.set_clim(*clim)
        # The direction of the axes kept (the images: y downwards)
        for ax, (x, y) in zip(axes, self.lims):
            if x is not None:
                ax.set_xlim(*(x[::-1] if ax.xaxis_inverted() else x))
            if y is not None:
                ax.set_ylim(*(y[::-1] if ax.yaxis_inverted() else y))


def scale_frames(indices, n=VIDEO_SCALE_FRAMES):
    """The maps of a video (indices) measured for its scales (Scales): n evenly spaced, the first and the last."""
    indices = list(indices)
    if len(indices) <= n:
        return indices
    return [indices[i] for i in np.unique(np.linspace(0, len(indices) - 1, n).round().astype(int))]


class Video:
    """Video written frame by frame: an animated GIF (fmt "gif"; 256 colours per frame), or MP4 (fmt "mp4": H.264
    with ffmpeg, its index at the start of the file; a GIF if ffmpeg isn't found), the extension of path changed.
    add(draw, k) draws a frame with draw(fig, k) in its figure (figsize, dpi) and writes it; close() ends the file,
    which is then a complete video of the frames written, also when it is closed before the last frame planned
    (stopped). Without any frame, no file. measure(draw, k), before the first frame, draws the frame of the map k to
    give all the frames the same scales (Scales: those of the frames measured)."""

    def __init__(self, path, fps=VIDEO_FPS, figsize=MAP_SIZE, dpi=VIDEO_DPI, fmt=VIDEO_FORMAT):
        if fmt not in VIDEO_FORMATS:
            raise ValueError(f"Video format: one of {', '.join(VIDEO_FORMATS)}")
        if fmt == "mp4" and ffmpeg_available():
            from matplotlib import animation

            self.path = os.path.splitext(path)[0] + ".mp4"
            # yuv420p: read by every player (even width and height); faststart: the index first
            self.writer = animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=[
                "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart"])
        else:
            self.path = os.path.splitext(path)[0] + ".gif"
            self.writer = GifWriter(fps)
        self.fig, self.dpi, self.frames = Figure(figsize=figsize, layout="constrained"), dpi, 0
        self.scales = None

    def measure(self, draw, k):
        draw(self.fig, k)
        self.scales = self.scales or Scales()
        self.scales.measure(self.fig)

    def add(self, draw, k):
        if self.frames == 0:
            # The writer started with the first frame: no file without frames
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self.writer.setup(self.fig, self.path, self.dpi)
        draw(self.fig, k)
        if self.scales is not None:
            self.scales.apply(self.fig)
        self.writer.grab_frame()
        self.frames += 1

    def close(self):
        """End the file (the frames written). Returns it, None without frames."""
        if self.frames == 0:
            return None
        self.frames = 0
        self.writer.finish()
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


################################################################################
# Settings and analyses


@dataclass
class FieldSettings:
    """Settings common to the analyses of vector fields.

    outlier_k: vectors further than outlier_k standard deviations from the mean at their grid point are removed
        (0: none), the mean and the standard deviation being those of the other vectors of the grid point in the
        maps of the statistics (all the maps, or the window of the rolling statistics: FieldAnalysis.clean);
    use_u_ref, u_ref: the velocities are shown over the reference velocity u_ref (m/s; None: the highest mean
        speed, the norm of the mean velocity). The turbulence intensity and the default flow time are always
        relative to it (the highest mean speed without use_u_ref);
    use_t_flow, t_flow: the times are shown in flow times t_flow (s; None: L_FoV / u_ref, the time the flow takes
        to cross the field of view);
    l_ref: reference length (mm): the positions are shown as x / L_ref and y / L_ref (None: in mm);
    nu: kinematic viscosity of the fluid (m²/s): the dissipation of the k-ε scales (air 1.5e-5, water 1.0e-6);
    rolling, rolling_maps: rolling statistics, those of map k over the rolling_maps maps up to it (all the maps
        before it while there are fewer): every figure then has one per map (FieldAnalysis.window).
    The reference quantities are those of the physical space: in the sensor space (vector fields in pixels), pixels
    and pairs."""

    outlier_k: float = 4.0
    use_u_ref: bool = False
    u_ref: float = None
    use_t_flow: bool = True
    t_flow: float = None
    l_ref: float = None
    nu: float = 1.5e-5
    rolling: bool = False
    rolling_maps: int = 100

    def __post_init__(self):
        self.use_u_ref, self.use_t_flow, self.rolling = bool(self.use_u_ref), bool(self.use_t_flow), bool(self.rolling)
        self.integers(self, ("rolling_maps",))
        if self.rolling_maps < 1:
            raise ValueError("Invalid settings: rolling_maps (maps of the rolling statistics) >= 1")
        self.u_ref = float(self.u_ref) if self.u_ref else None
        self.t_flow = float(self.t_flow) if self.t_flow else None
        self.l_ref = float(self.l_ref) if self.l_ref else None
        self.nu = float(self.nu) if self.nu is not None else 1.5e-5
        for name, label in (("u_ref", "reference velocity"), ("t_flow", "flow time"), ("l_ref", "reference length"),
                            ("nu", "kinematic viscosity")):
            if getattr(self, name) is not None and not getattr(self, name) > 0:
                raise ValueError(f"The {label} must be positive")
        self.outlier_k = float(self.outlier_k)
        if not self.outlier_k >= 0:
            raise ValueError("Invalid settings: outlier_k >= 0")

    @staticmethod
    def integers(settings, names):
        """Settings that must be integers, converted (ValueError if they aren't)."""
        for name in names:
            value = getattr(settings, name)
            if value is None or float(value) != int(value):
                raise ValueError(f"{name} must be an integer")
            setattr(settings, name, int(value))

    def steps(self):
        """Steps of the recipe, in the order they are applied."""
        steps = []
        if self.outlier_k:
            steps.append(f"outliers > {num2str(self.outlier_k)} σ removed")
        if self.rolling:
            steps.append(f"rolling statistics over {self.rolling_maps} maps")
        return steps


class FieldAnalysis:
    """Analyses of vector fields with settings (SETTINGS), evaluated when they are needed and kept until the
    settings they depend on change, and their figures (FIGURES: key and name; FIGURE_GROUPS: groups of the menus;
    INSTANTANEOUS: those of one map; FIGURE_FILES: element of the names of the files exported; FIGSIZE: sizes of
    the figures exported that aren't maps). The subclasses draw their figures with methods draw_[key](fig,
    prefix), or draw_map(fig, key, k, prefix, when, lim) for the instantaneous maps (colour limits: field_limits;
    USES_U_REF: those relative to the reference velocity, whatever use_u_ref; SCALED: those whose velocities are
    shown over it with use_u_ref, the only others with it in their titles).

    The statistics are those of the maps of the window (window): all the maps, or with rolling statistics those up
    to the map drawn (map_index, set by draw). The analyses of the window are cached by windowed, the outlier
    filtering too (against the statistics of the window); the reference quantities (u_ref, t_flow) are those of all
    the maps."""

    SETTINGS = FieldSettings
    TASK = ""
    FIGURE_TITLES = {}
    # Symbols (LaTeX, without $) of the quantities of the maps: they name the maps compared (plot_comparison)
    FIGURE_SYMBOLS = {}
    FIGURE_GROUPS = []
    INSTANTANEOUS = set()
    FIGURE_FILES = {}
    FIGSIZE = {}
    USES_U_REF = set()
    SCALED = set()
    # Settings that don't change the instantaneous maps: their colour ranges kept when they change
    DISPLAY_SETTINGS = ()
    # Axes where map() draws, instead of new axes (plot_comparison: the second map over the first one)
    overlay = None

    def __init__(self, fields, settings=None):
        self.fields = fields
        self.settings = settings or self.SETTINGS()
        self.cache = {}
        self.map_index = 0  # Map drawn: the end of the window of the rolling statistics
        self.whole = 0  # Inside all_maps: the statistics of all the maps

    def cached(self, name, key, compute):
        """Result of an analysis, evaluated again only when its key (the settings it depends on) changes."""
        if name not in self.cache or self.cache[name][0] != key:
            self.cache[name] = (key, compute())
        return self.cache[name][1]

    def window(self):
        """Maps of the statistics (first, last + 1): all the maps, or with rolling statistics the rolling_maps maps
        up to the map drawn (all those before it while there are fewer)."""
        n = self.fields.n
        if not self.rolling():
            return 0, n
        k = int(np.clip(self.map_index, 0, n - 1))
        return max(0, k - self.settings.rolling_maps + 1), k + 1

    def rolling(self):
        """The statistics are rolling (not inside all_maps)."""
        return self.settings.rolling and not self.whole

    @contextmanager
    def all_maps(self):
        """The statistics of all the maps, whatever the rolling statistics (the reference quantities, the key
        numbers)."""
        self.whole += 1
        try:
            yield
        finally:
            self.whole -= 1

    def windowed(self, name, key, compute):
        """Result of an analysis of the maps of the window (cached: those of all the maps, and the last window)."""
        window = self.window()
        if window != (0, self.fields.n):
            name, key = name + " (rolling)", (key, window)
        return self.cached(name, key, compute)

    def instantaneous(self, key):
        """The figure key has one per map: an instantaneous figure, or any figure with rolling statistics."""
        return key in self.INSTANTANEOUS or (self.settings.rolling and key in self.FIGURE_TITLES)

    ############################################################################
    # Analyses

    def clean(self):
        """U and V of the maps of the window (window: all the maps, or those of the rolling statistics) without the
        outliers, and the number of vectors removed. A vector is an outlier when one of its components is further
        than outlier_k standard deviations from the mean of the others of its grid point in the window: the mean and
        the standard deviation (N - 2 degrees of freedom) of the N - 1 other vectors, so that the vector doesn't
        inflate the spread it is compared with (with all N, no vector could be further than sqrt(N - 1) standard
        deviations: none removed from fewer than 18 maps at 4 standard deviations). A grid point with fewer than 3
        other vectors keeps its vectors."""
        k = self.settings.outlier_k

        def compute():
            a, b = self.window()
            U, V = self.fields.U[a:b], self.fields.V[a:b]
            if not k:
                return U, V, 0
            out = outliers(U, k) | outliers(V, k)
            U, V = U.copy(), V.copy()
            U[out] = V[out] = np.nan
            return U, V, int(out.sum())
        return self.windowed("clean", k, compute)

    def statistics(self):
        """Mean, fluctuations (rms: sample standard deviations, Reynolds shear stress <u'v'>: sample covariance),
        turbulence intensity
        sqrt((u_rms² + v_rms²) / 2) / U_ref, valid vectors and median peak ratio of the maps of the window, at the
        reliable grid points (in the flow)."""
        s = self.settings

        def compute():
            f = self.fields
            a, b = self.window()
            U, V, n_out = self.clean()
            with quiet():
                reliable = f.status == 0
                mean_U, mean_V = np.nanmean(U, 0, dtype=np.float64), np.nanmean(V, 0, dtype=np.float64)
                # Sample standard deviations and covariance (N - 1): unbiased with few maps (rolling statistics)
                u_rms, v_rms = nanstd1(U), nanstd1(V)
                uv = nancov1(U, V, mean_U, mean_V)
                peak = (nanmedian(f.peak_ratio[a:b], 0) if f.peak_ratio is not None
                        else np.full(f.shape, np.nan))
            for F in (mean_U, mean_V, u_rms, v_rms, uv, peak):
                F[~reliable] = np.nan
            # The reference velocity entered (physical space), else the highest mean speed of all the maps (the norm
            # of the mean velocity: whatever the direction of the flow)
            if (a, b) != (0, f.n):
                with self.all_maps():
                    u_ref = self.statistics()["u_ref"]
            else:
                speed = np.hypot(mean_U, mean_V)
                u_ref = ((s.u_ref if s.use_u_ref and not self.sensor() else None)
                         or (float(np.nanmax(speed)) if np.isfinite(speed).any() and np.nanmax(speed) > 0 else 1.0))
            ti = 100 * np.sqrt((u_rms ** 2 + v_rms ** 2) / 2) / abs(u_ref)
            # The valid vectors of a PIV (not computed for a PTV: the valid cells of its binning aren't a reliable
            # quality)
            valid = (np.where(f.status == 0, 100 * f.piv_valid[a:b].mean(0), np.nan) if f.particles is None
                     else None)
            return dict(mean_U=mean_U, mean_V=mean_V, u_rms=u_rms, v_rms=v_rms, uv=uv, ti=ti, u_ref=u_ref,
                        valid=valid, peak_ratio=peak, reliable=reliable, outliers=n_out)
        return self.windowed("statistics", (s.outlier_k, s.use_u_ref, s.u_ref), compute)

    ############################################################################
    # Figures

    @classmethod
    def figures(cls, groups=None):
        """Figures of groups of quantities (FIGURE_GROUPS by default): (key, title)."""
        return [(key, cls.FIGURE_TITLES[key]) for _, keys in (groups or cls.FIGURE_GROUPS) for key in keys]

    def plot(self, key, fig=None, index=0, name=None, clim=None, cmap=None, contour=None):
        """Draw a figure (FIGURE_TITLES) in a Matplotlib figure (a new one if None): the instantaneous figures are those
        of the map index (and the rolling statistics those of the window ending at it). name: of the measurement, in
        the titles; clim: (vmin, vmax) of the colour bar of the
        maps (None, or a None value: the automatic one); cmap: colour map of the maps (None: the one of the figure);
        contour: value of an isocontour of the map (contour_maps; None: none). Returns the figure."""
        fig = fig or Figure(figsize=self.FIGSIZE.get(key, MAP_SIZE), layout="constrained")
        fig.clf()
        try:
            self.draw(key, fig, index, name, clim, cmap)
            if self.sensor():
                self.sensor_texts(fig)
            if contour is not None:
                self.contour_maps(fig, contour, symbol=self.symbol(key) if key in self.FIGURE_SYMBOLS else None)
        except Exception as e:  # The figure says why it can't be drawn, the others are still drawn
            fig.clf()
            fig.text(0.5, 0.5, f"{self.FIGURE_TITLES.get(key, key)}\n\nnot available: {e}", ha="center", va="center",
                     color="0.45", wrap=True)
        return fig

    def draw(self, key, fig, index=0, name=None, clim=None, cmap=None):
        """Draw a figure in a Matplotlib figure, without clearing it (plot), its colour range and colour map applied
        to the maps it draws only. Returns these maps (the colour-mapped fields with a colour bar)."""
        self.map_index = int(np.clip(index, 0, self.fields.n - 1))
        before = self.colour_maps(fig)
        prefix = f"{name} · " if name else ""
        draw = getattr(self, f"draw_{key}", None)
        with matplotlib.rc_context(MATH_FONT):
            if draw is not None:
                draw(fig, prefix)
            else:
                self.draw_field(fig, key, index, prefix)
        maps = [a for a in self.colour_maps(fig) if a not in before]
        if clim is not None:
            self.apply_clim(maps, clim)
        if cmap is not None:
            self.apply_cmap(maps, cmap)
        return maps

    def plot_comparison(self, key, other, fig=None, index=0, name=None, clims=(None, None), cmaps=(None, None),
                        contour=None):
        """Two maps in the same axes, compared with a wiper: the figure key (drawn by plot) and the map of the
        figure other drawn over it, each with its colour range and colour map (clims, cmaps), their colour bars
        right of the map (the first one's first, each titled with the symbol of its quantity: symbol), both titles
        above. Returns (fig, layers, reason): layers is (axes, artists of the first map, artists of the second), to
        clip on either side of the wiper; None, with the reason, if one of the figures isn't a single map (the
        figure key is then drawn alone). contour: value of an isocontour of the first map (on its side of the
        wiper)."""
        fig = self.plot(key, fig, index, name, clims[0], cmaps[0], contour)
        maps = self.colour_maps(fig)
        # A map: its axes and the one of its colour bar
        if len(maps) != 1 or len(fig.axes) != 2 or fig.texts:
            return fig, None, f"{self.FIGURE_TITLES.get(key, key)} is not a map"
        ax = maps[0].axes
        first, title = self.data_artists(ax), ax.get_title()
        # The legend of the first map kept when the second one adds its own
        legend = ax.get_legend()
        if legend is not None:
            ax.legend_ = None
            ax.add_artist(legend)
        n_axes = len(fig.axes)
        self.overlay = ax
        try:
            second = self.draw(other, fig, index, name, clims[1], cmaps[1])
            reason = None
            if len(second) != 1 or len(fig.axes) != n_axes + 1 or fig.texts:
                reason = f"{self.FIGURE_TITLES.get(other, other)} is not available as a map"
        except Exception as e:
            reason = f"{self.FIGURE_TITLES.get(other, other)} not available: {e}"
        finally:
            self.overlay = None
        if reason is not None:
            return self.plot(key, fig, index, name, clims[0], cmaps[0], contour), None, reason
        second_title = ax.get_title()
        # The first colour bar added again, so that it is next to the map, before the second one (the constrained
        # layout keeps the colour bars of a map of fixed aspect next to it only on one side)
        bar = maps[0].colorbar
        label = bar.ax.get_ylabel()
        bar.remove()
        fig.colorbar(maps[0], ax=ax, shrink=0.85, label=label)
        symbols = [f"${self.FIGURE_SYMBOLS[k]}$" for k in (key, other)]
        for m, symbol in zip(maps + second, symbols):
            m.colorbar.ax.set_title(symbol, fontsize="small", fontweight="bold")
        # The second lines of the titles (timing, reference quantities) once if they are the same
        first_lines, second_lines = title.split("\n", 1), second_title.split("\n", 1)
        if len(first_lines) == len(second_lines) == 2 and first_lines[1] == second_lines[1]:
            combined = f"{symbols[0]}: {first_lines[0]}\n{symbols[1]}: {second_lines[0]}\n{first_lines[1]}"
        else:
            combined = f"{symbols[0]}: {title}\n{symbols[1]}: {second_title}"
        ax.set_title(combined, fontweight="bold", fontsize="medium")
        if self.sensor():
            self.sensor_texts(fig)
        return fig, (ax, first, [a for a in self.data_artists(ax) if a not in first]), None

    @classmethod
    def contour_maps(cls, fig, level, color="white", symbol=None):
        """Isocontour of the value level on the first map of a figure (its field as drawn), outlined in black so
        that it shows on any colour map, and its entry in the legend of the map: "Isocontour [symbol] = [level]"
        (symbol: of the quantity of the map, LaTeX with $; level with 3 significant digits), added to the legend
        the map has, if any. Removing a contour set drawn (remove) removes its entry too, the legend restored as it
        was. Returns the contour sets drawn (none if the map doesn't reach level)."""
        from matplotlib import patheffects

        for a in cls.colour_maps(fig):
            field = getattr(a, "field", None)
            if field is None:
                continue
            X, Y, F = field
            if not (np.isfinite(F).any() and np.nanmin(F) <= level <= np.nanmax(F)):
                return []
            ax = a.axes
            outline = [patheffects.withStroke(linewidth=2.8, foreground="black")]
            with quiet():
                cs = ax.contour(X, Y, F, levels=[level], colors=color, linewidths=1.3)
            cs.set_path_effects(outline)
            cls.contour_legend(ax, cs, f"Isocontour {symbol + ' ' if symbol else ''}= "
                                       f"{significant(level, trim=True).replace('-', chr(0x2212))}",
                               Line2D([], [], color=color, linewidth=1.3, path_effects=outline))
            return [cs]
        return []

    @staticmethod
    def contour_legend(ax, cs, label, handle):
        """The entry (handle, label) of an isocontour (cs) added to the legend of its axes (a new one if it has
        none, at the place, size and frame of the legend replaced otherwise), and removed with it (cs.remove)."""
        previous = ax.get_legend()
        handles, labels, options = [], [], {"loc": "best", "fontsize": "x-small"}
        if previous is not None:
            for h, t in zip(previous.legend_handles, previous.texts):
                if h is not None:
                    handles.append(h)
                    labels.append(t.get_text())
            options = {"loc": previous._loc, "fontsize": previous.prop.get_size_in_points(),
                       "frameon": previous.get_frame_on()}
        legend = ax.legend(handles + [handle], labels + [label], **options)
        remove = cs.remove

        def remove_with_entry():
            remove()
            if ax.get_legend() is legend:
                legend.remove()
                ax.legend_ = previous
        cs.remove = remove_with_entry

    def symbol(self, key):
        """Symbol of the quantity of a map (LaTeX, with $), as in its figure: relabelled in the sensor space."""
        symbol = f"${self.FIGURE_SYMBOLS[key]}$"
        return self.relabel(symbol) if self.sensor() else symbol

    @staticmethod
    def data_artists(ax):
        """The artists an axes draws in its data area (fields, vectors, lines, markers, legends)."""
        return [*ax.collections, *ax.images, *ax.lines, *ax.patches, *ax.texts, *ax.artists]

    def sensor(self):
        """The vector fields are in pixels: the figures are in the sensor space (pixels and pairs)."""
        return self.fields.settings.get("units") == "px"

    @staticmethod
    def relabel(content):
        """A text in the sensor space: displacements instead of velocities (SENSOR_TEXTS)."""
        for old, new in SENSOR_TEXTS:
            content = content.replace(old, new)
        return content

    @classmethod
    def sensor_texts(cls, fig):
        """The texts of a figure in the sensor space (relabel)."""
        for text in fig.findobj(Text):
            content = cls.relabel(text.get_text())
            if content != text.get_text():
                text.set_text(content)

    def length(self):
        """Reference length of the figures (mm): the positions are divided by it (1: in mm, or in the sensor
        space)."""
        return 1.0 if self.sensor() else self.settings.l_ref or 1.0

    def unit(self):
        """Unit of the positions: mm, or px in the sensor space."""
        return "px" if self.sensor() else "mm"

    def image_height(self):
        """Height of the images (px) in the sensor space, None if unknown (or in the physical space)."""
        return (self.fields.settings.get("image_px") or [None, None])[1] if self.sensor() else None

    def plot_x(self, x):
        """Abscissa of positions x of the vector fields in the figures: over L_ref (physical space)."""
        return np.asarray(x) / self.length()

    def plot_y(self, y):
        """Ordinate of positions y of the vector fields in the figures: over L_ref (physical space), or in the image
        (px from its top, downwards: sensor space, where y is from its bottom)."""
        height = self.image_height()
        return height - np.asarray(y) if height else np.asarray(y) / self.length()

    def position_label(self, name):
        """Label (LaTeX) of an axis of positions: x (mm), or x / L_ref with the reference length (its value given
        in the title: refs); in the image in the sensor space: x (px)."""
        if self.sensor():
            return f"${name[0]}$ (px)"
        L = self.settings.l_ref
        symbol = POSITION_SYMBOLS.get(name, name)
        if not L:
            return f"${symbol}$ ({self.unit()})"
        return f"${symbol} / {LREF}$"

    def position(self, name, value_mm, decimals=1):
        """A position in a text (LaTeX): x = 50.0 mm, or x / L_ref = 1.25 with the reference length; in the image in
        the sensor space: x = 512 px."""
        if self.sensor():
            return f"${name} = {float(self.plot_y(value_mm) if name == 'y' else value_mm):.0f}$ px"
        L = self.settings.l_ref
        if L:
            return f"${name} / {LREF} = {sig2(value_mm / L)}$"
        return f"${name} = {value_mm:.{decimals}f}$ mm"

    def fov_length(self):
        """Length of the field of view along x, crossed by the flow (mm; px for vector fields in pixels): the width
        of the field of view of the camera, else the width of the images, else the one of the grid of vectors."""
        f = self.fields
        s = f.settings
        width_px = (s.get("image_px") or [None])[0]
        if s.get("units") == "px" and width_px:
            return float(width_px)
        fov = (s.get("camera") or {}).get("fov_width_mm")
        if fov:
            return float(fov)
        if width_px and f.scale:
            return float(width_px * f.scale)
        spacing = float(np.nanmedian(np.abs(np.diff(f.x[0])))) if f.x.shape[1] > 1 else 0.0
        return float(np.nanmax(f.x) - np.nanmin(f.x) + spacing)

    def default_flow_time(self):
        """Flow time L_FoV / u_ref (s; frames for vector fields in pixels): the time the flow takes to cross the field
        of view at the reference velocity. In the sensor space, u_ref is in px per pair, a pair spanning frame_step
        frames: the flow time in frames is L_FoV / u_ref pairs times frame_step."""
        if self.sensor():
            return self.fov_length() / abs(self.statistics()["u_ref"]) * self.frame_step()
        return self.fov_length() * 1e-3 / abs(self.statistics()["u_ref"])

    def frame_step(self):
        """Frames between the two images of a pair (1 if unknown)."""
        pairs = self.fields.settings.get("pairs") or []
        step = pairs[0][1] - pairs[0][0] if pairs and len(pairs[0]) == 2 else None
        return int(step) if step and step > 0 else int(self.fields.settings.get("frame_step") or 1)

    def flow_time(self):
        """Flow time t_flow (s; frames in the sensor space): the one entered, else L_FoV / u_ref."""
        s = self.settings
        return s.t_flow if s.use_t_flow and s.t_flow and not self.sensor() else self.default_flow_time()

    def map_interval(self):
        """Time between two successive maps (s; frames for vector fields in pixels)."""
        f = self.fields
        d = np.diff(np.asarray(f.t, np.float64))
        d = d[d > 0]
        return float(np.median(d)) if d.size else float(f.dt or 1.0)

    def flow_times(self, n_maps):
        """Duration of n_maps successive maps, in flow times."""
        return np.asarray(n_maps, np.float64) * self.map_interval() / self.flow_time()

    def uses_flow_time(self):
        """The times are shown in flow times (physical space)."""
        return self.settings.use_t_flow and not self.sensor()

    def scaling_velocity(self):
        """The reference velocity the velocities are shown over (physical space, use_u_ref), None if they aren't."""
        return abs(self.statistics()["u_ref"]) if self.settings.use_u_ref and not self.sensor() else None

    def velocity(self, F, symbol, power=1):
        """A field of velocities (power 1) or of their products (power 2) as shown, and its label (LaTeX): over
        u_ref^power with the reference velocity, else in m/s (px in the sensor space)."""
        u = self.scaling_velocity()
        if u:
            return F / u ** power, rf"${symbol}\ /\ {UREF}" + ("^2$" if power == 2 else "$")
        unit = ("px" if power == 1 else "px²") if self.sensor() else ("m/s" if power == 1 else "m²/s²")
        return F, f"${symbol}$ ({unit})"

    def u_ref_text(self, u_ref=None):
        """The reference velocity (that of the statistics, else u_ref), in the titles of the figures that use it
        (LaTeX)."""
        return f"${UREF}$ = {u_ref or abs(self.statistics()['u_ref']):.3g} {'px' if self.sensor() else 'm/s'}"

    def t_flow_text(self):
        """The flow time, in the titles of the figures that use it (LaTeX)."""
        return f"${TFLOW}$ = {self.flow_time() * 1000:.3g} ms"

    def refs(self, *parts, positions=True, u_ref=False, scaled=True, newline=True):
        """End of a title: its parts in parentheses (time...), with the reference velocity when the figure uses it
        (u_ref: True, or its value when it isn't that of the statistics; use_u_ref for the figures whose velocities
        are scaled with it: scaled) and the reference length when it shows positions divided by it (physical space);
        empty without any. On a second line of the title (newline; False: after a space, e.g. in a legend), so that
        the long titles fit."""
        parts = [p for p in parts if p]
        value = None if u_ref is True else u_ref
        if not self.sensor():
            if u_ref or scaled and self.settings.use_u_ref:
                parts.append(self.u_ref_text(value))
            if positions and self.settings.l_ref:
                parts.append(f"${LREF}$ = {num2str(self.settings.l_ref)} mm")
        elif u_ref:
            parts.append(self.u_ref_text(value))
        return ("\n" if newline else " ") + f"({', '.join(parts)})" if parts else ""

    def map_time(self, k, u_ref=False, scaled=True):
        """End of the title of an instantaneous map: the pair among the pairs (sensor space), or its time, in flow
        times (with the flow time) or in s, with the reference quantities used (refs)."""
        f = self.fields
        if self.sensor():
            return self.refs(f"Pair {k + 1}/{f.n}", u_ref=u_ref, scaled=scaled)
        if self.uses_flow_time():
            return self.refs(f"$t$ = {sig2(f.t[k] / self.flow_time())} ${TFLOW}$", self.t_flow_text(), u_ref=u_ref,
                             scaled=scaled)
        return self.refs(f"$t$ = {f.t[k]:.4g} s", u_ref=u_ref, scaled=scaled)

    def window_text(self):
        """The rolling statistics in the titles (LaTeX): the pair drawn and the pairs averaged up to it (sensor space),
        or the time of the map drawn and the averaging time (those of the maps averaged: fewer than rolling_maps at
        the start), in flow times (with the flow time) or in s; "" without rolling statistics."""
        if not self.rolling():
            return ""
        a, b = self.window()
        f, m = self.fields, b - a
        if self.sensor():
            return f"Pair {b}/{f.n}, {m} pair{'s' if m > 1 else ''} averaged"
        if self.uses_flow_time():
            return (rf"$t$ = {sig2(f.t[b - 1] / self.flow_time())} ${TFLOW}$, "
                    rf"$\Delta t$ = {sig2(self.flow_times(m))} ${TFLOW}$, " + self.t_flow_text())
        return rf"$t$ = {f.t[b - 1]:.4g} s, $\Delta t$ = {m * self.map_interval():.4g} s"

    def average_time(self, u_ref=False, scaled=True):
        """End of the title of a map of the statistics: the rolling statistics (window_text), or all the maps: their
        number (sensor space), or their duration (Δt), in flow times (with the flow time) or in s; with the reference
        quantities used (refs)."""
        f = self.fields
        if self.rolling():
            return self.refs(self.window_text(), u_ref=u_ref, scaled=scaled)
        if self.sensor():
            return self.refs(f"{f.n} pairs", u_ref=u_ref, scaled=scaled)
        if self.uses_flow_time():
            return self.refs(rf"$\Delta t$ = {sig2(self.flow_times(f.n))} ${TFLOW}$", self.t_flow_text(),
                             u_ref=u_ref, scaled=scaled)
        return self.refs(rf"$\Delta t$ = {f.n * self.map_interval():.4g} s", u_ref=u_ref, scaled=scaled)

    def scaled(self, key):
        """The figure key shows velocities over the reference velocity with use_u_ref (SCALED)."""
        return key in self.SCALED

    @staticmethod
    def colour_maps(fig):
        """The colour-mapped fields of a figure that have a colour bar (the maps)."""
        return [a for ax in fig.axes for a in ax.collections + ax.images if getattr(a, "colorbar", None) is not None]

    @classmethod
    def apply_clim(cls, maps, clim):
        """Colour range (vmin, vmax; None: the automatic one) of maps (those of a figure, if a figure), and of
        their colour bars."""
        for a in cls.colour_maps(maps) if isinstance(maps, Figure) else maps:
            low, high = a.get_clim()
            a.set_clim(low if clim[0] is None else clim[0], high if clim[1] is None else clim[1])

    @classmethod
    def apply_cmap(cls, maps, cmap):
        """Colour map of maps (those of a figure, if a figure), and of their colour bars."""
        for a in cls.colour_maps(maps) if isinstance(maps, Figure) else maps:
            a.set_cmap(cmap)

    def message(self, fig, title, text):
        fig.text(0.5, 0.5, f"{title}\n\n{text}", ha="center", va="center", color="0.45", wrap=True)

    def map(self, fig, F, title, label, cmap="viridis", vmin=None, vmax=None, quiver=None, step=4):
        """Map of a field on the grid, with a colorbar and optionally the vectors (one of step in each
        direction). Drawn in the axes overlay if it is set (plot_comparison: over the map there, which keeps its
        limits and labels). Returns the axes."""
        f = self.fields
        X, Y, image = self.map_positions()
        if self.overlay is not None:
            ax = self.overlay
        else:
            ax = fig.add_subplot()
            ax.set_facecolor("#d5d9e0")
        im = ax.pcolormesh(X, Y, F, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
        im.field = (X, Y, np.asarray(F, np.float64))  # As drawn: its isocontours (contour_maps)
        if quiver is not None:
            sl = (slice(None, None, step), slice(None, None, step))
            ax.quiver(X[sl], Y[sl], quiver[0][sl], quiver[1][sl], color="white", width=0.0022)
        if self.overlay is None:
            ax.set_aspect("equal")
            ax.set_xlim(np.nanmin(X), np.nanmax(X))
            # The images: y downwards from their top
            ax.set_ylim(*((np.nanmax(Y), np.nanmin(Y)) if image else (np.nanmin(Y), np.nanmax(Y))))
            camera = f.settings.get("camera", {})
            ax.set_xlabel(self.position_label("x global" if camera.get("x_offset_mm") else "x"))
            ax.set_ylabel(self.position_label("y global" if camera.get("y_offset_mm") else "y"))
        ax.set_title(title, fontweight="bold", fontsize="medium")
        fig.colorbar(im, ax=ax, shrink=0.85, label=label)
        return ax

    def map_positions(self):
        """Positions of the maps (plot_x, plot_y), and True for the axes of an image (y downwards)."""
        return self.plot_x(self.fields.x), self.plot_y(self.fields.y), bool(self.image_height())

    @staticmethod
    def limits(F, symmetric=False, center=0.0, low=None, q=99):
        """Color limits at the q-th percentile (symmetric around center), so that a few extreme values don't
        hide the field."""
        v = F[np.isfinite(F)]
        if v.size == 0:
            return None, None
        if symmetric:
            d = float(np.percentile(np.abs(v - center), q)) or 1.0
            return center - d, center + d
        return (float(np.percentile(v, 100 - q)) if low is None else low), float(np.percentile(v, q))

    def instantaneous_limits(self, key):
        """Color limits of an instantaneous field, the same for all the maps (field_limits, from 50 of them)."""
        f = self.fields
        sub = slice(None, None, max(1, f.n // 50))
        # The fields scaled with the reference quantities depend on the settings (their values: the settings object
        # may be changed in place); the same limits with rolling statistics (those of all the maps)

        def compute():
            with self.all_maps():
                return self.field_limits(key, sub)
        settings = tuple(getattr(self.settings, f.name) for f in dataclass_fields(self.settings)
                         if f.name not in self.DISPLAY_SETTINGS)
        return self.cached(f"limits_{key}", settings, compute)

    def field_limits(self, key, sub):
        """Color limits of an instantaneous field from the maps sub."""
        raise NotImplementedError

    def missing(self, key):
        """Why an instantaneous field can't be drawn (None if it can)."""
        return None

    def exported(self, key):
        """The figure key is exported (export_figures): the data it needs are there."""
        return True

    def draw_field(self, fig, key, index, prefix):
        """Instantaneous field of the map index (draw_map), with its time (map_time)."""
        f = self.fields
        k = int(np.clip(index, 0, f.n - 1))
        when = self.map_time(k, key in self.USES_U_REF, self.scaled(key))
        missing = self.missing(key)
        if missing:
            return self.message(fig, self.FIGURE_TITLES[key], missing)
        self.draw_map(fig, key, k, prefix, when, self.instantaneous_limits(key))

    def draw_map(self, fig, key, k, prefix, when, lim):
        raise NotImplementedError

    def stat_map(self, fig, prefix, F, title, label, cmap="viridis", symmetric=False, center=0.0, low=None,
                 quiver=None, u_ref=False, key=None):
        """Map of a statistics of all the maps (u_ref: relative to the reference velocity, whatever use_u_ref; key:
        the figure, with the reference velocity in its title only if it is scaled with it)."""
        vmin, vmax = self.limits(F, symmetric, center, low)
        return self.map(fig, F, prefix + title + self.average_time(u_ref, self.scaled(key)), label, cmap, vmin, vmax,
                        quiver)

    ############################################################################
    # Export

    def plot_export(self, key, fig, index=0, name=None, clims=None, cmaps=None, contours=None, compares=None):
        """A figure as exported (plot): alone, or its map compared with the map of another figure with a wiper
        (compares: {key: (other, fraction of the wiper)}), each map with its colour range and colour map (clims,
        cmaps: {key: ...}), the isocontour of the first one (contours). Returns the figure."""
        clims, cmaps, contours = clims or {}, cmaps or {}, contours or {}
        other, fraction = (compares or {}).get(key) or (None, None)
        if other is None:
            return self.plot(key, fig, index, name, clims.get(key), cmaps.get(key), contours.get(key))
        fig, layers, _ = self.plot_comparison(key, other, fig, index, name, (clims.get(key), clims.get(other)),
                                              (cmaps.get(key), cmaps.get(other)), contours.get(key))
        if layers is not None:
            Wiper(*layers, labels=(self.symbol(key), self.symbol(other))).place(fraction)
        return fig

    def export_element(self, key, compares=None):
        """Element of the names of the files of a figure: its own, joined with the one of the figure it is
        compared with ("-")."""
        other = ((compares or {}).get(key) or (None,))[0]
        return "-".join(self.FIGURE_FILES[k] for k in (key, other) if k is not None)

    def export_figures(self, folder, groups=None, index=0, dpi=EXPORT_DPI, name=None, task=None, progress=None,
                       clims=None, cmaps=None, contours=None, keys=None, maps=None, compares=None,
                       image_options=None):
        """Save the figures of groups of quantities (FIGURE_GROUPS by default; keys: only these figures) as PNG in a
        folder, named [name]_[task]_[element](_[map]) (task: TASK by default): the instantaneous ones (all of them
        with rolling statistics) of the map index (numbered with it), or of the maps of maps (indices), each figure
        in its own subfolder
        [name]_[task]_[element]. progress(fraction, message) is called for each image; clims, cmaps, contours:
        colour ranges ({key: (vmin, vmax)}), colour maps ({key: name}) and isocontours ({key: value}) of the maps
        of some figures; compares: the figures compared with another map ({key: (other, fraction of the wiper)},
        plot_export); image_options: the maps of the instantaneous figures of some figures instead of maps ({key:
        indices}). Returns the files."""
        os.makedirs(folder, exist_ok=True)
        name = name or self.fields.name
        task = task or self.TASK
        written = []
        figures = [(key, title) for key, title in self.figures(groups)
                   if (keys is None or key in keys) and self.exported(key)]
        # Each image: (figure, title, map, folder)
        images = []
        for key, title in figures:
            chosen = (image_options or {}).get(key, maps)
            if not self.instantaneous(key):
                images.append((key, title, None, folder))
            elif chosen is None:
                images.append((key, title, index, folder))
            else:
                # The maps of each instantaneous figure: in its own folder
                subfolder = os.path.join(folder, export_name(name, task, self.export_element(key, compares)))
                images += [(key, title, k, subfolder) for k in chosen]
        for n, (key, title, k, where) in enumerate(images):
            if progress is not None:
                progress(n / len(images), f"Image {n + 1} of {len(images)}: {title}"
                         + ("" if k is None else f" (map {k + 1})") + "...")
            fig = self.plot_export(key, Figure(figsize=self.FIGSIZE.get(key, MAP_SIZE), layout="constrained"),
                                   index if k is None else k, name, clims, cmaps, contours, compares)
            os.makedirs(where, exist_ok=True)
            path = os.path.join(where, export_name(name, task, self.export_element(key, compares),
                                                   None if k is None else k + 1, "png", self.fields.n))
            fig.savefig(path, dpi=dpi)
            written.append(path)
        return written

    def video_path(self, folder, keys, name=None, task=None):
        """File of the video of a figure (or of figures compared: keys), in a folder: [name]_[task]_[elements].mp4
        (the elements of the figures joined by "-")."""
        # The element of each figure as its images name it (export_element)
        element = "-".join(self.export_element(k) for k in keys if k is not None)
        return os.path.join(folder, export_name(name or self.fields.name, task or self.TASK, element, extension="mp4"))

    def write_video(self, path, draw, indices, fps=VIDEO_FPS, figsize=MAP_SIZE, dpi=VIDEO_DPI, progress=None,
                    fmt=VIDEO_FORMAT):
        """Video (fmt: "gif" or "mp4", see Video) of the frames drawn by draw(fig, k) for the maps k of indices.
        The frames have the same scales (Video.measure: those of scale_frames). progress(fraction, message) is called
        for each frame. Returns the file written."""
        indices = list(indices)
        if not indices:
            raise ValueError("no map to put in the video")
        measured = scale_frames(indices)
        total = len(measured) + len(indices)
        with Video(path, fps, figsize, dpi, fmt) as video:
            for n, k in enumerate(measured):
                if progress is not None:
                    progress(n / total, f"Scales: frame {n + 1} of {len(measured)} (map {k + 1})...")
                video.measure(draw, k)
            for n, k in enumerate(indices):
                if progress is not None:
                    progress((len(measured) + n) / total, f"Frame {n + 1} of {len(indices)} (map {k + 1})...")
                video.add(draw, k)
        return video.path

    def export_video(self, key, folder, step=1, fps=VIDEO_FPS, name=None, clim=None, cmap=None, progress=None,
                     maps=None, contour=None, fmt=VIDEO_FORMAT, compare=None, clims=None, cmaps=None):
        """Video (fmt: "gif" or "mp4") of a figure of each map (instantaneous) over the maps, every step maps
        (maps: these maps, indices), in a folder (video_path, write_video): clim, cmap, contour: its colour range,
        colour map and isocontour; compare: (other figure, fraction of the wiper) to compare its map with the map
        of another figure (clims, cmaps: those of both, {key: ...}). Returns the file written."""
        if not self.instantaneous(key):
            raise ValueError(f"{self.FIGURE_TITLES.get(key, key)} is not an instantaneous figure (nor rolling)")
        name = name or self.fields.name
        clims, cmaps = {**(clims or {}), key: clim}, {**(cmaps or {}), key: cmap}
        compares = {key: compare} if compare and compare[0] is not None else None
        return self.write_video(self.video_path(folder, [key, (compare or (None,))[0]], name),
                                lambda fig, k: self.plot_export(key, fig, k, name, clims, cmaps, {key: contour},
                                                                compares),
                                range(0, self.fields.n, max(1, int(step))) if maps is None else maps, fps,
                                progress=progress, fmt=fmt)
