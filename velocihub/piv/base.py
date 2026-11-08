"""PIV of the pre-processed images: the image pairs, the PIV software and the vector fields, in pixels.

A PIV software is an engine (PIVEngine) that computes the displacement of the image pairs in pixels. The engines are
registered in ENGINES: PaIRS-UniNa (pairs_engine.py) is the first one, TracTrac (ptv_engine.py: particle tracking,
binned on the grid) the second; another software is added with a subclass of PIVEngine. The camera settings (Camera: width of the field of view, frame rate...) convert the pixels into mm and
m/s: the post-processing does it (postprocessing.load_vectors), the pixel pitches being the width (height) of the
field of view divided by the width (height) of the images.

Results, in the folder of the PIV (subfolder PIV of the image folder by default, PTV for TracTrac):
    vectors.npz      all the vector fields: x_px, y_px (positions, px from the top-left corner of the image),
                     u_px, v_px (displacements, px, v positive upwards), peak_ratio (none for TracTrac, which has
                     ptv_particles: the particles of each cell), info (engine: 1 = valid),
                     status (per grid point: 0 = flow, 1 = no particles, 2 = masked) and settings (JSON text, with
                     the pairs and the size of the images), read by the post-processing; with a camera (Camera),
                     also x_mm, y_mm, x_global_mm (from the origin of the axes of the camera, Camera.x_shift and
                     y_shift, y upwards), U, V (m/s) and t_s (time of the first image of each pair); with a regression
                     (regression.py), u_px, v_px and info are those of the regression, and raw_u_px, raw_v_px,
                     raw_info those of the PIV, with regression_residual_px and regression_n_rbf per map
    settings.json    every setting used, and the run time (run_time_s, run_time, seconds_per_pair, resumed_pairs)
    tracks.npz       TracTrac: the particles tracked (ptv_engine.py)
The project is the name of the folder of the images unless another one is given.
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields

import numpy as np
from PIL import Image

from ..fields import PIV_FOLDER, TRACKS_FILE, VECTORS_FILE
from ..project import clean_name, comparable_recipe, default_project_name, format_duration, relative

# Frame rate of the camera when none is given (Hz): the times are then in frames
DEFAULT_FRAME_RATE = 1.0
# Pairs saved together while they are computed: the computation stopped continues from the last chunk saved
CHUNK = 100
# Vectors of a grid point valid in less than this fraction of the pairs: no particles there
MIN_VALID_FRACTION = 0.1
# Task in the names of the files exported (export_name)
TASK = "PIV"


def num2str(x):
    return str(int(x)) if float(x).is_integer() else f"{x:.6g}"


################################################################################
# Camera and pairs


@dataclass
class Camera:
    """Camera and acquisition settings.

    fov_width_mm: width of the field of view, the one of the images: the horizontal pixel pitch (mm/px) is
    fov_width_mm divided by the width of the images in pixels;
    x_offset_mm: horizontal offset, the position of the origin of x in the global axes (x_global = x + x_offset_mm,
    the stitching of fields of view);
    frame_rate_hz: frame rate of the camera (DEFAULT_FRAME_RATE: 1 Hz, the times in frames);
    frame_step: the images k and k + frame_step form a pair, separated by dt = frame_step / frame_rate_hz;
    pair_step: the first images of the pairs are 1, 1 + pair_step, 1 + 2 pair_step... (1: every image);
    y_offset_mm: vertical offset, the position of the origin of y in the global axes (y_global = y + y_offset_mm);
    fov_height_mm: height of the field of view: the vertical pixel pitch is fov_height_mm divided by the height of
    the images in pixels (None: the horizontal pitch, square pixels);
    x_shift, y_shift: the origin of the axes, from the bottom-left corner of the images, in fractions of the width
    and of the height of the field of view: x = 0 at x_shift times the width from the left edge, y = 0 at y_shift
    times the height from the bottom edge, y upwards (0, 0: the bottom-left corner; y_shift = 0.5: the middle of the
    height, e.g. a jet centred in the images; 0.5, 0.5: the centre)."""

    fov_width_mm: float
    x_offset_mm: float = 0.0
    frame_rate_hz: float = DEFAULT_FRAME_RATE
    frame_step: int = 1
    pair_step: int = 1
    y_offset_mm: float = 0.0
    fov_height_mm: float = None
    x_shift: float = 0.0
    y_shift: float = 0.0

    def __post_init__(self):
        self.fov_width_mm, self.x_offset_mm = float(self.fov_width_mm), float(self.x_offset_mm)
        self.frame_rate_hz, self.y_offset_mm = float(self.frame_rate_hz), float(self.y_offset_mm)
        self.fov_height_mm = float(self.fov_height_mm) if self.fov_height_mm not in (None, "") else None
        self.x_shift, self.y_shift = float(self.x_shift or 0), float(self.y_shift or 0)
        if not (self.fov_width_mm > 0 and self.frame_rate_hz > 0 and np.isfinite(self.x_offset_mm)
                and np.isfinite(self.y_offset_mm)):
            raise ValueError("The width of the field of view and the frame rate must be positive")
        if self.fov_height_mm is not None and not self.fov_height_mm > 0:
            raise ValueError("The height of the field of view must be positive (empty: square pixels)")
        if not (np.isfinite(self.x_shift) and np.isfinite(self.y_shift)):
            raise ValueError("The shifts of the axes must be numbers (fractions of the field of view)")
        if float(self.frame_step) != int(self.frame_step) or float(self.pair_step) != int(self.pair_step):
            raise ValueError("The frame step and the pair step must be integers")
        self.frame_step, self.pair_step = int(self.frame_step), int(self.pair_step)
        if self.frame_step < 1 or self.pair_step < 1:
            raise ValueError("The frame step and the pair step must be at least 1")

    @classmethod
    def from_record(cls, record, **changes):
        """The camera of a record (asdict of a Camera, as saved with the vector fields), the keys it doesn't know
        ignored. A record of the previous versions, without y_shift, had y = 0 at the middle of the height of the
        images: y_shift 0.5."""
        names = {f.name for f in dataclass_fields(cls)}
        values = {k: v for k, v in dict(record).items() if k in names}
        values.setdefault("y_shift", 0.5)
        values.update(changes)
        return cls(**values)

    @property
    def dt(self):
        """Time between the two images of a pair (s)."""
        return self.frame_step / self.frame_rate_hz

    def scale(self, width_px):
        """Horizontal pixel pitch in the flow (mm/px): the width of the field of view divided by the width of the
        images in pixels."""
        return self.fov_width_mm / width_px

    def scale_y(self, width_px, height_px):
        """Vertical pixel pitch in the flow (mm/px): the height of the field of view divided by the height of the
        images in pixels, or the horizontal pitch without height (square pixels)."""
        return self.fov_height_mm / height_px if self.fov_height_mm else self.scale(width_px)

    def pitches(self, width_px, height_px):
        """Horizontal and vertical pixel pitches (mm/px) of images of this size."""
        return self.scale(width_px), self.scale_y(width_px, height_px)

    def positions(self, x_px, y_px, width_px, height_px, offsets=True):
        """Positions (mm) of points of images of this size given in pixels from their top-left corner: x to the right
        and y upwards from the origin (x_shift, y_shift), plus the offsets of the field of view (offsets)."""
        sx, sy = self.pitches(width_px, height_px)
        x = (np.asarray(x_px, np.float64) - self.x_shift * width_px) * sx
        y = ((1.0 - self.y_shift) * height_px - np.asarray(y_px, np.float64)) * sy
        return (x + self.x_offset_mm, y + self.y_offset_mm) if offsets else (x, y)

    def velocities(self, u_px, v_px, width_px, height_px):
        """Velocities (m/s) of displacements (px per pair, v upwards) in images of this size."""
        sx, sy = self.pitches(width_px, height_px)
        return u_px * (sx / 1000.0 / self.dt), v_px * (sy / 1000.0 / self.dt)

    def pairs(self, n_images, n_pairs=None):
        """Pairs of images (0-based indices): (k, k + frame_step) for k = 0, pair_step, 2 pair_step... The first
        n_pairs (None: all)."""
        pairs = [(k, k + self.frame_step) for k in range(0, n_images - self.frame_step, self.pair_step)]
        return pairs if n_pairs is None else pairs[: int(n_pairs)]

    def describe(self, width_px=None, height_px=None):
        text = f"FOV {num2str(self.fov_width_mm)}"
        text += f" x {num2str(self.fov_height_mm)} mm" if self.fov_height_mm else " mm wide"
        if width_px:
            sx = self.scale(width_px)
            if self.fov_height_mm and height_px:
                text += f" ({sx:.5f} x {self.scale_y(width_px, height_px):.5f} mm/px)"
            else:
                text += f" ({sx:.5f} mm/px)"
        if self.x_shift or self.y_shift:
            text += f", origin at ({num2str(self.x_shift)}, {num2str(self.y_shift)}) of the field of view"
        return (text + f", horiz. offset {num2str(self.x_offset_mm)} mm"
                + (f", vert. offset {num2str(self.y_offset_mm)} mm" if self.y_offset_mm else "")
                + f", {num2str(self.frame_rate_hz)} Hz, "
                f"pairs (k, k + {self.frame_step}) every {self.pair_step} image(s), dt = {self.dt * 1000:.4g} ms")


################################################################################
# Engines


@dataclass(frozen=True)
class Parameter:
    """A setting of a PIV engine, shown in the interface: its key, label, default value, kind ("float", "int",
    "ints": integers separated by spaces, "bool", or "choice": one of choices) and help."""

    key: str
    label: str
    default: object
    kind: str = "float"
    help: str = ""
    choices: tuple = ()

    def parse(self, value):
        """The value of a text (or of a value): ValueError if it is invalid."""
        if self.kind == "choice":
            value = str(value).strip().lower()
            if value not in self.choices:
                raise ValueError(f"{self.label}: one of {', '.join(self.choices)}")
            return value
        if self.kind == "bool":
            return value.strip().lower() in ("1", "true", "yes", "on") if isinstance(value, str) else bool(value)
        if self.kind == "ints":
            items = value.replace(",", " ").split() if isinstance(value, str) else list(value)
            values = tuple(int(float(v)) for v in items)
            if not values or any(float(v) != int(float(v)) for v in items):
                raise ValueError(f"{self.label}: integers separated by spaces")
            return values
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{self.label}: a number")
        if self.kind == "int":
            if number != int(number):
                raise ValueError(f"{self.label}: an integer")
            return int(number)
        return number

    def format(self, value):
        """Text of a value, as entered in the interface."""
        if self.kind == "ints":
            return " ".join(str(v) for v in value)
        if self.kind == "bool":
            return bool(value)
        if self.kind == "choice":
            return value
        # Large numbers (a condition number) in powers of ten
        return f"{value:g}" if self.kind == "float" and abs(value) >= 1e6 else num2str(value)


class PIVProcess:
    """A PIV computation running. poll() returns the messages sent since the last call: dictionaries with a
    "type": "log" (text), "progress" (phase, done, total, rate in s per pair, and for each pair its valid
    vectors in % and median peak ratio; for the pairs of a chunk saved before, resumed, first and the lists valid
    and peak_ratio of its pairs), "field" (optional: pair, the index of a pair just computed, and file, an
    npz file with its xpx, ypx, u, v, sn and info, shown while the PIV runs), "done" (optional: seconds, the
    computation of the pairs, resumed, the pairs computed before, and resumed_seconds, their seconds; see
    PIVRun.timing), "stopped" or "error" (text). stop() asks it to stop after the pair being computed: the chunks
    of pairs saved are kept, and the computation continues from them when it is started again with the same images
    and settings."""

    def poll(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def kill(self):
        """Stop at once (the chunks of pairs saved are kept)."""
        self.stop()

    def wait(self, timeout=None):
        """Wait for the end of the process, after its last message."""

    @property
    def running(self):
        raise NotImplementedError


class WorkerProcess(PIVProcess):
    """PIV computed by a Python script in another process, which may be run by another Python interpreter: the
    script reads the job (JSON file of the work folder) and writes its messages as JSON lines on its output. It
    stops when the file STOP is created in the work folder."""

    def __init__(self, python, script, job, work):
        os.makedirs(work, exist_ok=True)
        self.work = work
        self.stop_file = os.path.join(work, "STOP")
        if os.path.exists(self.stop_file):
            os.remove(self.stop_file)
        job_file = os.path.join(work, "job.json")
        with open(job_file, "w") as f:
            json.dump(job, f)
        # Without console window when started from the interface
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen([python, "-u", "-B", script, job_file], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                                     errors="replace", creationflags=flags)
        self.queue = queue.Queue()
        self.stderr = []
        self.ended = False
        self.readers = [threading.Thread(target=self.read, args=(stream, err), daemon=True)
                        for stream, err in ((self.proc.stdout, False), (self.proc.stderr, True))]
        for reader in self.readers:
            reader.start()

    def read(self, stream, err):
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if err:
                self.stderr.append(line)
                continue
            try:
                message = json.loads(line)
            except ValueError:
                message = {"type": "log", "text": line}
            self.queue.put(message)

    def poll(self):
        messages = []
        finished = self.proc.poll() is not None and not any(r.is_alive() for r in self.readers)
        while True:
            try:
                messages.append(self.queue.get_nowait())
            except queue.Empty:
                break
        if any(m.get("type") in ("done", "stopped", "error") for m in messages):
            self.ended = True
        elif finished and not self.ended:
            # The process ended without saying it (crash)
            self.ended = True
            details = " | ".join(self.stderr[-6:]) or "no message"
            messages.append({"type": "error", "text": f"the PIV/PTV process ended unexpectedly (exit code "
                                                      f"{self.proc.returncode}): {details}"})
        return messages

    def stop(self):
        if self.proc.poll() is None:
            open(self.stop_file, "w").close()

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()

    def wait(self, timeout=30):
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    @property
    def running(self):
        return not self.ended


class PIVEngine:
    """A PIV software: computes the displacement fields of image pairs, in pixels.

    A subclass defines the name, the parameters (the settings shown in the interface, see Parameter), available
    (whether the software can run on this computer) and start, which processes the pairs without blocking and
    returns a PIVProcess. The process writes in the work folder raw.npz: xpx, ypx (positions of the vectors, in
    pixels from the top-left corner of the image), u, v (displacements in pixels, v downwards), sn (correlation
    peak ratio; optional) and info (1 for the valid vectors), arrays (pairs, rows, columns), and failed (pairs not
    computed); optionally count (the particles of each vector) and tracks.npz. Register it with @register to make it
    available in the interface. results_folder: the default folder of its results in the folder of the measurement;
    sequence: it needs the pairs of a sequence (consecutive frames), not any pairs."""

    key = ""
    name = ""
    # The name of its pipeline (velocihub/pipelines.py): its results in the menus, its post-processing exported to
    # PostProcessing/[label]
    label = ""
    description = ""
    parameters = ()
    results_folder = PIV_FOLDER
    sequence = False
    method = "PIV"  # Shown with the name in the menus: PIV or PTV

    def defaults(self):
        return {p.key: p.default for p in self.parameters}

    def check(self, settings):
        """Settings completed with the defaults and validated: ValueError if they are invalid."""
        unknown = set(settings) - {p.key for p in self.parameters}
        if unknown:
            raise ValueError(f"Unknown settings of {self.name}: {', '.join(sorted(unknown))}")
        return {p.key: p.parse(settings[p.key]) if p.key in settings else p.default for p in self.parameters}

    def available(self):
        """(True, description) if the software can run, (False, reason) otherwise."""
        return True, self.name

    def describe(self, settings):
        """The settings, as shown in the interface and saved."""
        return ", ".join(f"{p.label.lower()} {p.format(settings[p.key])}" for p in self.parameters)

    def start(self, images, pairs, settings, work, key=None):
        """Start computing the pairs (images[a], images[b]) with the settings checked, in the work folder.
        key identifies the computation: the chunks of pairs saved in the work folder with another key are
        discarded. Returns a PIVProcess."""
        raise NotImplementedError


ENGINES = {}
DEFAULT_ENGINE = "pairs"


def register(cls):
    """Class decorator: makes a PIV engine available (in the interface and get_engine)."""
    cls.label = cls.label or cls.name
    ENGINES[cls.key] = cls
    return cls


def get_engine(engine=None, **options):
    """A PIV engine: an instance of PIVEngine, or its key or name (case-insensitive; None: PaIRS-UniNa) and the
    options of its class."""
    if isinstance(engine, PIVEngine):
        return engine
    name = (engine or DEFAULT_ENGINE).lower()
    for key, cls in ENGINES.items():
        if name in (key.lower(), cls.name.lower()):
            return cls(**options)
    raise ValueError(f'Unknown PIV/PTV engine "{engine}": {", ".join(c.name for c in ENGINES.values())}')


################################################################################
# PIV of a set of images


def image_key(paths):
    """Identity of the images (name, size and date): the progress of a computation is kept only for the same
    images, not for images processed again with another recipe."""
    key = []
    for p in paths:
        st = os.stat(p)
        key.append([os.path.basename(p), st.st_size, int(st.st_mtime)])
    return key


class PIVRun:
    """PIV of a set of images: the pairs (each image and the next one, those of the camera settings, or those given),
    computed by an engine and saved in the output folder, in pixels (vectors.npz, settings.json). Started again with
    the same images and settings, the computation continues from the chunks of pairs saved.

    images: paths of the images, in time order; camera: Camera (None: the results in pixels only; with a camera,
    vectors.npz also has the velocities); engine: PIVEngine, or its key or name; settings:
    of the engine (the defaults for the others); output: folder of the results (subfolder PIV of the folder of
    the images by default, results_folder of the engine); n_pairs: number of pairs computed (None: all); masked: boolean image of the masked
    pixels (the vectors whose window center is masked get the status 2); info: dictionary saved with the
    settings (the pre-processing of the images, for instance); project: name of the project (None: the name of
    the measurement folder, the one of the images); regression: the regression of the vector fields after the PIV
    ({"method": "spicy", "settings": {...}}, regression.check_regression; None: none)."""

    def __init__(self, images, camera=None, engine=None, settings=None, output=None, n_pairs=None, masked=None,
                 info=None, pairs=None, project=None, regression=None):
        from .regression import check_regression

        self.images = [os.path.abspath(p) for p in images]
        if len(self.images) < 2:
            raise ValueError("At least two images are needed")
        self.camera = camera
        self.engine = get_engine(engine)
        self.settings = self.engine.check(settings or {})
        self.regression = check_regression(regression)
        self.output = os.path.abspath(output or os.path.join(os.path.dirname(self.images[0]),
                                                             self.engine.results_folder))
        steps = camera or Camera(1.0)
        self.pairs = [tuple(p) for p in pairs] if pairs is not None else steps.pairs(len(self.images), n_pairs)
        if not self.pairs:
            raise ValueError(f"No image pair with {len(self.images)} images and pairs (k, k + {steps.frame_step})")
        with Image.open(self.images[0]) as im:
            self.width, self.height = im.size
        self.masked = masked
        self.info = info or {}
        # The measurement folder: the one of the images, or the one whose subfolder PIV holds the results
        measurement = self.info.get("measurement_folder") or (
            os.path.dirname(self.output) if os.path.basename(self.output) in (PIV_FOLDER, self.engine.results_folder)
            else os.path.dirname(self.images[0]))
        self.measurement = measurement
        self.project = project or default_project_name(measurement)
        self.work = os.path.join(self.output, "_progress")
        self.process = None
        self.t0 = None

    def start(self):
        """Start the computation, without waiting: returns the PIVProcess."""
        ok, reason = self.engine.available()
        if not ok:
            raise RuntimeError(reason)
        self.t0 = time.time()
        key = {"images": image_key(self.images), "pairs": self.pairs, "settings": self.settings,
               "engine": self.engine.key}
        self.process = self.engine.start(self.images, self.pairs, self.settings, self.work, key)
        if self.regression is not None:
            from .regression import ChainProcess, spicy_available, start_regression

            ok, reason = spicy_available()
            if not ok:
                self.process.kill()
                raise RuntimeError(reason)
            self.process = ChainProcess(self.process, lambda done: start_regression(
                self.work, self.regression, self.raw_status().tolist(), self.width, key))
        return self.process

    def raw_status(self):
        """Status of the grid points of the displacements computed (raw.npz of the work folder; see vectors)."""
        with np.load(os.path.join(self.work, "raw.npz")) as raw:
            return self.status(raw["xpx"], raw["ypx"], raw["info"] == 1)

    def status(self, xpx, ypx, good):
        """Status of the grid points: 1, no particles (valid in less than 10 % of the pairs), 2, masked (window
        center masked), else 0 (flow)."""
        status = np.where(good.mean(0) < MIN_VALID_FRACTION, 1, 0).astype(np.int16)
        if self.masked is not None:
            rows = np.clip(np.round(ypx - 0.5).astype(int), 0, self.masked.shape[0] - 1)
            cols = np.clip(np.round(xpx - 0.5).astype(int), 0, self.masked.shape[1] - 1)
            status[self.masked[rows, cols]] = 2
        return status

    def run(self, report=None, interval=0.2):
        """Compute and save the vector fields. report(message) receives the messages of the process (printed
        by default, the progress at most every 10 s). Returns the file of the vector fields, None if stopped."""
        report = report or ConsoleReport()
        process = self.start()
        try:
            while True:
                for message in process.poll():
                    report(message)
                    if message["type"] == "done":
                        return self.save(done=message)
                    if message["type"] == "stopped":
                        return None
                    if message["type"] == "error":
                        raise RuntimeError(message["text"])
                time.sleep(interval)
        except KeyboardInterrupt:
            process.stop()
            raise

    def save(self, done=None):
        """Save the displacements computed (raw.npz of the work folder) in the output folder (with the velocities if
        there is a camera) and delete the work folder. done is the message "done" of the process, with the time of
        the computation (see timing). Returns the file of the vector fields."""
        with np.load(os.path.join(self.work, "raw.npz")) as raw:
            xpx, ypx, failed = raw["xpx"], raw["ypx"], raw["failed"].tolist()
            u, v, info = (raw[k] for k in ("u", "v", "info"))
            # The peak ratio of a correlation, the particles of each vector of a tracking
            sn, count = (raw[k] if k in raw.files else None for k in ("sn", "count"))
        tracks = os.path.join(self.work, TRACKS_FILE)
        tracks = tracks if os.path.isfile(tracks) else None
        if self.regression is None:
            arrays, settings = self.vectors(xpx, ypx, u, v, sn, info, self.pairs, failed, count=count,
                                            tracks=tracks is not None)
        else:
            from .regression import describe_regression, regression_record

            # The fields of the regression; those of the PIV kept (raw_), its status (flow, no particles, masked)
            with np.load(os.path.join(self.work, "regressed.npz")) as reg:
                ru, rv, rinfo, residual, n_rbf, rfailed = (reg[k] for k in ("u", "v", "info", "residual", "n_rbf",
                                                                             "failed"))
            arrays, settings = self.vectors(xpx, ypx, ru, rv, sn, rinfo, self.pairs, failed,
                                            status=self.status(xpx, ypx, info == 1), count=count,
                                            tracks=tracks is not None)
            arrays.update(raw_u_px=u, raw_v_px=-v, raw_info=info, regression_residual_px=residual,
                          regression_n_rbf=n_rbf)
            finite = residual[np.isfinite(residual)]
            settings.update(regression=regression_record(self.regression),
                            regression_description=describe_regression(self.regression),
                            regression_failed_maps=rfailed.tolist(),
                            regression_median_residual_px=float(np.median(finite)) if finite.size else None,
                            regression_seconds=(done or {}).get("regression_seconds"))
        os.makedirs(self.output, exist_ok=True)
        # The files of a previous PIV of the project are replaced (with the CSV files of the vector maps and of the
        # mean field that earlier versions wrote: they would no longer match the vector fields)
        start = f"{clean_name(self.project)}_{TASK}_"
        for name in os.listdir(self.output):
            if ((name.startswith(start) and name.endswith(".csv"))
                    or name in (VECTORS_FILE, "settings.json", TRACKS_FILE)):
                os.remove(os.path.join(self.output, name))
        settings.update(self.timing(done))
        path = os.path.join(self.output, VECTORS_FILE)
        np.savez_compressed(path, **arrays, settings=json.dumps(settings))
        if tracks is not None:
            shutil.move(tracks, os.path.join(self.output, TRACKS_FILE))
        with open(os.path.join(self.output, "settings.json"), "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=1, ensure_ascii=False)
        # The work folder is deleted once the process has ended (it may still have its files open)
        if self.process is not None:
            self.process.wait()
        shutil.rmtree(self.work, ignore_errors=True)
        return path

    def timing(self, done=None):
        """Run time of the PIV, as saved with its settings: run_time_s, from start() to the results saved (with the
        seconds of the pairs computed before, if the PIV was stopped and continued), run_time (text), resumed_pairs
        (pairs computed before), seconds_per_pair (time of the computation of the pairs, without the start of the
        engine and the saving of the files) and minutes (run_time_s in minutes, as saved before). done: the message
        "done" of the process, with seconds (the computation of the pairs), resumed (pairs) and resumed_seconds
        (None: unknown, then the run time is the one of this run only)."""
        done = done or {}
        resumed = int(done.get("resumed") or 0)
        before = done.get("resumed_seconds") if resumed else 0.0
        elapsed, compute = time.time() - self.t0 if self.t0 else None, done.get("seconds")
        if before is None:
            # Time of the pairs computed before unknown: the rate is the one of the pairs computed now
            counted = len(self.pairs) - resumed
        else:
            counted = len(self.pairs)
            elapsed = None if elapsed is None else elapsed + before
            compute = None if compute is None else compute + before
        return dict(run_time_s=None if elapsed is None else round(elapsed, 1), run_time=format_duration(elapsed),
                    resumed_pairs=resumed,
                    seconds_per_pair=round(compute / counted, 3) if compute is not None and counted > 0 else None,
                    minutes=None if elapsed is None else round(elapsed / 60, 1))

    def preview(self, message):
        """Vector field of the message "field" of the process (the last pair computed), saved as a file of vector
        fields of this pair in the work folder. Returns the file."""
        k = message["pair"]
        with np.load(message["file"]) as z:
            xpx, ypx = z["xpx"], z["ypx"]
            u, v, info = (z[q][None] for q in ("u", "v", "info"))
            sn = z["sn"][None] if "sn" in z.files else None
        arrays, settings = self.vectors(xpx, ypx, u, v, sn, info, [self.pairs[k]], where=self.work)
        path = os.path.join(self.work, "preview.npz")
        np.savez(path, **arrays, settings=json.dumps(settings))
        return path

    def vectors(self, xpx, ypx, u, v, sn, info, pairs, failed=(), where=None, status=None, count=None, tracks=False):
        """Arrays of a file of vector fields (in pixels, and the velocities if there is a camera) and its settings,
        from the displacements of the pairs computed. The folders in the settings are relative to the one of the
        measurement (of the project), which is relative to the folder of the file (where: the output folder by
        default). status: of the grid points (None: from info, see status); sn: the peak ratio (None: unknown);
        count: the particles of each vector (PTV); tracks: the particles tracked are saved beside (TRACKS_FILE)."""
        if status is None:
            status = self.status(xpx, ypx, info == 1)
        settings = dict(engine=self.engine.name, engine_settings=self.settings,
                        engine_description=self.engine.describe(self.settings), image_px=[self.width, self.height],
                        image_folder=os.path.dirname(self.images[0]), first_image=os.path.basename(self.images[0]),
                        last_image=os.path.basename(self.images[-1]), n_images=len(self.images),
                        n_pairs=len(pairs), pairs=[list(p) for p in pairs], failed_pairs=list(failed),
                        spacing_px=[float(np.median(np.diff(xpx[0]))), float(np.median(np.abs(np.diff(ypx[:, 0]))))],
                        project=self.project, date=time.strftime("%Y-%m-%d %H:%M"), **self.info)
        settings.update(measurement_folder=relative(self.measurement, where or self.output),
                        image_folder=relative(os.path.dirname(self.images[0]), self.measurement))
        if settings.get("preprocessing_recipe"):
            settings["preprocessing_recipe"] = comparable_recipe(settings["preprocessing_recipe"])
        if tracks:
            settings["tracks_file"] = TRACKS_FILE
        # Image y points down, v_px points up
        arrays = dict(x_px=xpx, y_px=ypx, u_px=u, v_px=-v, info=info, status=status)
        if sn is not None:
            arrays["peak_ratio"] = sn
        if count is not None:
            arrays["ptv_particles"] = count
        camera = self.camera
        if camera is not None:
            # From the origin of the axes (Camera.x_shift, y_shift), y upwards
            x_mm, y_mm = camera.positions(xpx, ypx, self.width, self.height, offsets=False)
            U, V = camera.velocities(u, -v, self.width, self.height)
            arrays.update(x_mm=x_mm, y_mm=y_mm + camera.y_offset_mm, x_global_mm=x_mm + camera.x_offset_mm, U=U, V=V,
                          t_s=np.array([a for a, _ in pairs]) / camera.frame_rate_hz)
            settings.update(camera=asdict(camera), scale_mm_per_px=camera.scale(self.width),
                            scale_y_mm_per_px=camera.scale_y(self.width, self.height), dt_s=camera.dt)
        return arrays, settings


def console(text):
    """Print a message, without the characters the console can't encode."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    print(text.replace("→", "->").encode(encoding, errors="replace").decode(encoding), flush=True)


class ConsoleReport:
    """Messages of a PIV process printed in the console, the progress at most every 10 s."""

    def __init__(self, every=10):
        self.every = every
        self.last = 0.0

    def __call__(self, message):
        kind = message["type"]
        if kind == "progress":
            if time.monotonic() - self.last < self.every and message["done"] < message["total"]:
                return
            self.last = time.monotonic()
            rate = message.get("rate")
            left = "" if not rate else f", {(message['total'] - message['done']) * rate / 60:.1f} min left"
            console(f"  {message['phase']} {message['done']}/{message['total']}{left}")
        elif kind in ("log", "error"):
            console(("ERROR " if kind == "error" else "  ") + message["text"])
        elif kind == "stopped":
            console("  stopped: started again, the computation continues from the last chunk saved")


def run_piv(images, camera=None, engine=None, settings=None, output=None, n_pairs=None, masked=None, info=None,
            report=None, project=None):
    """PIV of a set of images (see PIVRun), waiting for the end. Returns the file of the vector fields."""
    return PIVRun(images, camera, engine, settings, output, n_pairs, masked, info, project=project).run(report)
