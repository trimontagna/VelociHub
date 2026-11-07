"""Alignment of the fields of view of a stitching by hand (window Batch of the post-processing: right-click on a node
of a stitching, stitching.py): their Horiz. and Vert. offsets set on a background common to them, then written in
the node settings of their projects (Apply; nothing is changed before).

The fields of view are drawn at their offsets, at the scale of their camera settings (the width of the field of view
over that of the images; y upwards from the middle of the height of the images, as the vector fields), over one of
the backgrounds:
- Mean raw image: the median of some raw images of the project. The particles move and fade out; the scene that
  doesn't move (nozzle, walls, edges, reflections) stays: the same objects in all the fields of view, even measured
  in separate runs;
- Mean velocity |u|: that of the PIV results (the jet: its core and shear layers; less exact, the runs differ);
- Calibration image: the first image of a folder Calibration above the project (if there is one).
Views: Blend (the field of view moved over the others, its opacity), Difference (the overlaps: dark where they
match, bright where they don't) and Flicker (the field of view moved and the others in turn). A field of view is
moved by dragging it, by the arrows (0.1 mm; with Shift 1 mm) or by its offsets typed in; the reference stays.
The wheel zooms around the mouse, the right button pans, a double-click (or Fit) shows all: the composite is drawn
for the area shown, in the detail of the backgrounds when zoomed in.
Suggest: the shift of the field of view selected over the others that best correlates their overlap (phase
correlation of the background, its large scales left out), within a search range around where it is: only shown,
to be checked, as any position, before Apply."""

import os
import tkinter as tk
import warnings
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import ndimage

from ..fields import VECTORS_FILE, load_vectors
from ..piv import Camera, PIV_FOLDER
from ..workspace import image_names

BACKGROUNDS = ("Mean raw image", "Mean velocity |u|", "Calibration image")
VIEWS = ("Blend", "Difference", "Flicker")
# The raw images of the mean (evenly spaced), their width at most (block means), the pixels of the composite at most
MEAN_IMAGES = 40
MAX_WIDTH = 1000
MAX_PIXELS = 1.5e6
CALIBRATION = "Calibration"
COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf")


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


class AlignWindow(tk.Toplevel):
    """The alignment of the fields of view of a stitching: nodes (their labels, the reference first), chains
    [(label, projects: one per node)] of the projects shown (choice: the one shown first), targets: the projects of
    each node given its offsets (Apply), dialog: the window Batch (settings_of, apply_to)."""

    def __init__(self, dialog, nodes, chains, targets, choice=0):
        super().__init__(dialog)
        self.dialog, self.nodes, self.chains, self.targets = dialog, nodes, chains, targets
        self.title("Align the stitching: " + " → ".join(nodes))
        self.geometry(f"{min(1500, self.winfo_screenwidth() - 60)}x{min(860, self.winfo_screenheight() - 100)}")
        self.transient(dialog)
        self.cache = {}  # (background, folder): Source or None
        self.sources, self.offsets, self.start = [], [], []
        self.active, self.drag, self.flick, self.flicker_job, self.redraw_job = 1, None, False, None, None
        self.pan = None  # The right button pressed: (x, y in pixels, the limits then)
        self.grid_ = None
        body = ttk.Frame(self, padding=8)
        body.pack(fill=tk.BOTH, expand=True)
        side = ttk.Frame(body, width=330)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        plot = ttk.Frame(body)
        plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(9, 6), layout="constrained")
        self.ax = self.fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.fig, plot)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot, pack_toolbar=False)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.press)
        self.canvas.mpl_connect("motion_notify_event", self.motion)
        self.canvas.mpl_connect("button_release_event", self.release)
        self.canvas.mpl_connect("scroll_event", self.scroll)
        # The controls
        self.chain_var = tk.StringVar(self, chains[choice][0])
        if len(chains) > 1:
            ttk.Label(side, text="Projects shown").pack(anchor=tk.W)
            box = ttk.Combobox(side, textvariable=self.chain_var, values=[label for label, _ in chains],
                               state="readonly")
            box.pack(fill=tk.X)
            box.bind("<<ComboboxSelected>>", lambda event: self.load(keep_offsets=True))
        ttk.Label(side, text="Background").pack(anchor=tk.W, pady=(8, 0))
        self.background = tk.StringVar(self, BACKGROUNDS[0])
        box = ttk.Combobox(side, textvariable=self.background, values=BACKGROUNDS, state="readonly")
        box.pack(fill=tk.X)
        box.bind("<<ComboboxSelected>>", lambda event: self.load(keep_offsets=True))
        ttk.Label(side, text="View").pack(anchor=tk.W, pady=(8, 0))
        self.view = tk.StringVar(self, VIEWS[0])
        row = ttk.Frame(side)
        row.pack(fill=tk.X)
        for view in VIEWS:
            ttk.Radiobutton(row, text=view, value=view, variable=self.view, command=self.view_changed).pack(
                side=tk.LEFT, padx=(0, 8))
        ttk.Label(side, text="Opacity of the field of view selected (Blend)").pack(anchor=tk.W, pady=(6, 0))
        self.opacity = tk.DoubleVar(self, 0.5)
        ttk.Scale(side, from_=0.0, to=1.0, variable=self.opacity, command=lambda value: self.schedule()).pack(
            fill=tk.X)
        table = ttk.LabelFrame(side, text="Offsets (mm)", padding=6)
        table.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(table, text="Field of view").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(table, text="Horiz.").grid(row=0, column=1)
        ttk.Label(table, text="Vert.").grid(row=0, column=2)
        self.active_var = tk.IntVar(self, 1 if len(nodes) > 1 else 0)
        self.entries = []
        for i, name in enumerate(nodes):
            text = f"{name} (reference)" if i == 0 else name
            if i == 0:
                tk.Label(table, text="■", foreground=COLOURS[0]).grid(row=i + 1, column=0, sticky=tk.W)
                ttk.Label(table, text=text).grid(row=i + 1, column=0, sticky=tk.W, padx=(16, 0))
            else:
                cell = ttk.Frame(table)
                cell.grid(row=i + 1, column=0, sticky=tk.W)
                tk.Label(cell, text="■", foreground=COLOURS[i % len(COLOURS)]).pack(side=tk.LEFT)
                ttk.Radiobutton(cell, text=text, value=i, variable=self.active_var,
                                command=self.active_changed).pack(side=tk.LEFT)
            pair = []
            for column in (1, 2):
                var = tk.StringVar(self)
                entry = ttk.Entry(table, textvariable=var, width=8)
                entry.grid(row=i + 1, column=column, padx=2, pady=1)
                entry.bind("<Return>", lambda event, i=i: self.typed(i))
                entry.bind("<FocusOut>", lambda event, i=i: self.typed(i))
                if i == 0:
                    entry.state(["disabled"])
                pair.append(var)
            self.entries.append(pair)
        suggest = ttk.LabelFrame(side, text="Suggest (to check before Apply)", padding=6)
        suggest.pack(fill=tk.X, pady=(10, 0))
        row = ttk.Frame(suggest)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Search ± (mm)").pack(side=tk.LEFT)
        self.radius = tk.StringVar(self, "5")
        ttk.Entry(row, textvariable=self.radius, width=6).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(row, text="Suggest", command=self.suggest).pack(side=tk.LEFT)
        self.suggest_label = ttk.Label(suggest, foreground="#555555", wraplength=300, justify=tk.LEFT)
        self.suggest_label.pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(side, text="Drag a field of view, or the arrows: 0.1 mm, with Shift 1 mm. The reference stays. "
                             "Wheel: zoom; right button: pan; double-click: all. Difference: dark where the overlaps "
                             "match.",
                  foreground="#555555", wraplength=310, justify=tk.LEFT).pack(anchor=tk.W, pady=(10, 0))
        self.status = ttk.Label(side, foreground="#a33", wraplength=310, justify=tk.LEFT)
        self.status.pack(anchor=tk.W, pady=(6, 0))
        self.apply_note = ttk.Label(side, foreground="#555555", wraplength=310, justify=tk.LEFT)
        self.apply_note.pack(anchor=tk.W, pady=(6, 0))
        buttons = ttk.Frame(side)
        buttons.pack(side=tk.BOTTOM, anchor=tk.E, pady=(10, 0))
        ttk.Button(buttons, text="Apply", command=self.ok).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Cancel", command=self.close).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Fit", command=lambda: self.fit(True)).pack(side=tk.LEFT, padx=(6, 0))
        for sequence, dx, dy in (("<Left>", -1, 0), ("<Right>", 1, 0), ("<Up>", 0, 1), ("<Down>", 0, -1)):
            self.bind(sequence, lambda event, dx=dx, dy=dy: self.nudge(dx, dy, event))
        self.protocol("WM_DELETE_WINDOW", self.close)
        n = sum(len(t) for t in targets)
        self.apply_note["text"] = (f"Apply: the offsets in the node settings of the {n} projects of these nodes "
                                   "(each node: all its projects).")
        self.load()

    # ------------------------------------------------------------------ data

    def projects(self):
        return next(projects for label, projects in self.chains if label == self.chain_var.get())

    def camera(self, project):
        return self.dialog.settings_of(project).get("camera") or {}

    def load(self, keep_offsets=False):
        """The backgrounds of the fields of view (the projects shown), the offsets of their settings (kept when
        the background or the projects shown change)."""
        projects = self.projects()
        cameras = [self.camera(p) for p in projects]
        missing = [p.path for p, c in zip(projects, cameras) if not c.get("fov_width_mm")]
        if missing:
            messagebox.showwarning("Align the stitching", "Set the field of view width of the projects first "
                                   "(node settings): " + ", ".join(missing), parent=self)
            self.after_idle(self.destroy)
            return
        if not keep_offsets or not self.offsets:
            self.offsets = [[float(c.get("x_offset_mm") or 0), float(c.get("y_offset_mm") or 0)] for c in cameras]
            self.start = [list(o) for o in self.offsets]
        background = self.background.get()
        self.config(cursor="watch")
        self.update_idletasks()
        self.sources = []
        try:
            for p, camera in zip(projects, cameras):
                k = (background, p.folder, camera.get("fov_width_mm"))
                if k not in self.cache:
                    self.status["text"] = f"Reading {p.path}..."
                    self.update_idletasks()
                    try:
                        self.cache[k] = LOADERS[background](p, camera)
                    except Exception as e:  # The others shown
                        self.cache[k] = None
                        self.status["text"] = f"{p.path}: {e}"
                self.sources.append(self.cache[k])
        finally:
            self.config(cursor="")
        absent = [p.path for p, s in zip(projects, self.sources) if s is None]
        self.status["text"] = (f"No {background.lower()} for " + ", ".join(absent)) if absent else ""
        self.show_offsets()
        self.fit(True)

    # ------------------------------------------------------------------ geometry

    def extent(self, i):
        """Left, right, bottom, top (mm) of the field of view i at its offsets."""
        s = self.sources[i]
        left, top = s.left + self.offsets[i][0], s.top + self.offsets[i][1]
        h, w = s.arr.shape
        return left, left + w * s.pixel, top - h * s.pixel, top

    def fit(self, view=False):
        """The axes on the fields of view with a margin (the composite drawn for the area shown: redraw)."""
        shown = [i for i, s in enumerate(self.sources) if s is not None]
        if not shown:
            return
        boxes = np.array([self.extent(i) for i in shown])
        left, right, bottom, top = boxes[:, 0].min(), boxes[:, 1].max(), boxes[:, 2].min(), boxes[:, 3].max()
        margin = 0.15 * max(right - left, top - bottom)
        left, right, bottom, top = left - margin, right + margin, bottom - margin, top + margin
        self.ax.clear()
        self.image = self.ax.imshow(np.zeros((1, 1, 4)), extent=(left, right, bottom, top), interpolation="nearest")
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)
        self.ax.set_autoscale_on(False)
        # The area shown changed (wheel, pan, toolbar): drawn again for it
        self.ax.callbacks.connect("xlim_changed", lambda ax: self.schedule())
        self.ax.callbacks.connect("ylim_changed", lambda ax: self.schedule())
        self.ax.set_facecolor("#20242a")
        self.ax.set_xlabel("x (mm)")
        self.ax.set_ylabel("y (mm)")
        self.ax.set_aspect("equal")
        self.rects, self.labels = [], []
        for i, name in enumerate(self.nodes):
            colour = COLOURS[i % len(COLOURS)]
            rect = Rectangle((0, 0), 1, 1, fill=False, edgecolor=colour, linewidth=1)
            self.ax.add_patch(rect)
            self.rects.append(rect)
            # Inside the field of view, a few points from the top left of its part shown (redraw)
            self.labels.append(self.ax.annotate(name, (0, 0), xytext=(4, -4), textcoords="offset points", ha="left",
                                                va="top", color=colour, fontsize="small", clip_on=True,
                                                bbox=dict(boxstyle="round,pad=0.2", facecolor="#20242a",
                                                          edgecolor="none", alpha=0.7)))
        if view:
            self.toolbar.update()  # Home: this view
        self.redraw()

    def sample(self, i):
        """The background of the field of view i on the grid of the composite (NaN outside)."""
        left, top, pixel, nx, ny = self.grid_
        out = np.full((ny, nx), np.nan, np.float32)
        s = self.sources[i]
        if s is None:
            return out
        x = left + (np.arange(nx) + 0.5) * pixel
        y = top - (np.arange(ny) + 0.5) * pixel
        e_left, _, _, e_top = self.extent(i)
        cols = np.floor((x - e_left) / s.pixel).astype(int)
        rows = np.floor((e_top - y) / s.pixel).astype(int)
        vc = (cols >= 0) & (cols < s.arr.shape[1])
        vr = (rows >= 0) & (rows < s.arr.shape[0])
        if vc.any() and vr.any():
            out[np.ix_(vr, vc)] = s.arr[np.ix_(rows[vr], cols[vc])]
        return out

    def others(self, layers):
        """The mean of the fields of view but the one selected (NaN where none)."""
        rest = [layer for i, layer in enumerate(layers) if i != self.active]
        if not rest:
            return np.full_like(layers[0], np.nan)
        stack = np.stack(rest)
        count = np.isfinite(stack).sum(axis=0)
        total = np.nansum(stack, axis=0)
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)

    def others_on(self, i):
        """The mean of the other fields of view on the pixels of the field of view i (bilinear; NaN where none)."""
        s = self.sources[i]
        left, _, _, top = self.extent(i)
        h, w = s.arr.shape
        x = left + (np.arange(w) + 0.5) * s.pixel
        y = top - (np.arange(h) + 0.5) * s.pixel
        total, count = np.zeros((h, w)), np.zeros((h, w))
        for j, other in enumerate(self.sources):
            if j == i or other is None:
                continue
            o_left, _, _, o_top = self.extent(j)
            cols = (x - o_left) / other.pixel - 0.5
            rows = (o_top - y) / other.pixel - 0.5
            vc = (cols >= 0) & (cols <= other.arr.shape[1] - 1)
            vr = (rows >= 0) & (rows <= other.arr.shape[0] - 1)
            if not vc.any() or not vr.any():
                continue
            rr, cc = np.meshgrid(rows[vr], cols[vc], indexing="ij")
            total[np.ix_(vr, vc)] += ndimage.map_coordinates(other.arr, [rr, cc], order=1)
            count[np.ix_(vr, vc)] += 1
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)

    def redraw(self):
        self.redraw_job = None
        shown = [s for s in self.sources if s is not None]
        if not shown or not hasattr(self, "image"):
            return
        # The grid of the composite: the area shown, at the pixel of the backgrounds, at most that of the screen
        left, right = sorted(self.ax.get_xlim())
        bottom, top = sorted(self.ax.get_ylim())
        screen = (right - left) / max(self.ax.bbox.width, 1.0)
        pixel = max(min(s.pixel for s in shown), screen, np.sqrt((right - left) * (top - bottom) / MAX_PIXELS))
        nx, ny = max(int(np.ceil((right - left) / pixel)), 1), max(int(np.ceil((top - bottom) / pixel)), 1)
        self.grid_ = (left, top, pixel, nx, ny)
        self.image.set_extent((left, left + nx * pixel, top - ny * pixel, top))
        layers = [self.sample(i) for i in range(len(self.sources))]
        a, o = layers[self.active], self.others(layers)
        has_a, has_o = np.isfinite(a), np.isfinite(o)
        both = has_a & has_o
        a0, o0 = np.nan_to_num(a), np.nan_to_num(o)
        view = self.view.get()
        # The image: grey levels (one channel), or colours where the views need them; 8 bits (drawn as they are)
        rgba = np.zeros((ny, nx, 4), np.uint8)
        if view == "Blend":
            w = self.opacity.get()
            grey = np.where(both, w * a0 + (1 - w) * o0, np.where(has_a, a0, o0))
        elif view == "Flicker":
            grey = np.where(has_o, o0, a0) if self.flick else np.where(has_a, a0, o0)
        else:
            grey = 0.55 * np.where(has_a, a0, o0)
        rgba[..., :3] = (np.clip(grey, 0, 1) * 255).astype(np.uint8)[..., None]
        if view == "Difference":
            d = (np.clip(np.abs(a0 - o0)[both], 0, 1) * 255).astype(np.uint8)
            rgba[both] = np.stack([d, (0.35 * d).astype(np.uint8), d, np.full_like(d, 255)], axis=1)
        rgba[..., 3] = np.where(has_a | has_o, 255, 0)
        self.image.set_data(rgba)
        # The name of each field of view: at the top left of its part shown (hidden if none of it is)
        view_left, view_right = sorted(self.ax.get_xlim())
        view_bottom, view_top = sorted(self.ax.get_ylim())
        corners = []  # Those of the names shown: one under the other when at the same place
        for i, (rect, label) in enumerate(zip(self.rects, self.labels)):
            if self.sources[i] is None:
                rect.set_visible(False)
                label.set_visible(False)
                continue
            left, right, bottom, top = self.extent(i)
            rect.set_bounds(left, bottom, right - left, top - bottom)
            rect.set_linewidth(2.2 if i == self.active else 1)
            shown_left, shown_top = max(left, view_left), min(top, view_top)
            label.xy = (shown_left, shown_top)
            label.set_visible(shown_left < min(right, view_right) and max(bottom, view_bottom) < shown_top)
            if label.get_visible():
                near = 0.02 * (view_right - view_left)
                k = sum(abs(x - shown_left) < near and abs(y - shown_top) < near for x, y in corners)
                label.xyann = (4, -4 - 13 * k)
                corners.append((shown_left, shown_top))
        self.canvas.draw_idle()

    def schedule(self):
        if self.redraw_job is None:
            self.redraw_job = self.after(15, self.redraw)

    # ------------------------------------------------------------------ moving

    def show_offsets(self):
        for i, (x, y) in enumerate(self.offsets):
            self.entries[i][0].set(f"{x:.2f}")
            self.entries[i][1].set(f"{y:.2f}")

    def typed(self, i):
        try:
            x, y = (float(var.get().replace(",", ".")) for var in self.entries[i])
        except ValueError:
            self.bell()
            self.show_offsets()
            return
        if [x, y] != self.offsets[i]:
            self.offsets[i] = [x, y]
            self.schedule()

    def active_changed(self):
        self.active = self.active_var.get()
        self.schedule()

    def view_changed(self):
        if self.flicker_job is not None:
            self.after_cancel(self.flicker_job)
            self.flicker_job = None
        if self.view.get() == "Flicker":
            self.flicker()
        self.schedule()

    def flicker(self):
        self.flick = not self.flick
        self.redraw()
        self.flicker_job = self.after(450, self.flicker)

    def nudge(self, dx, dy, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry)) or self.active == 0:
            return
        step = 1.0 if event.state & 0x1 else 0.1
        self.offsets[self.active][0] += dx * step
        self.offsets[self.active][1] += dy * step
        self.show_offsets()
        self.schedule()

    def scroll(self, event):
        """The wheel: zoom in or out around the mouse."""
        if event.inaxes is not self.ax or event.xdata is None:
            return
        factor = 1 / 1.25 if event.button == "up" else 1.25
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        x, y = event.xdata, event.ydata
        self.ax.set_xlim(x - (x - x0) * factor, x + (x1 - x) * factor)
        self.ax.set_ylim(y - (y - y0) * factor, y + (y1 - y) * factor)

    def release(self, event):
        self.drag = self.pan = None

    def press(self, event):
        if event.inaxes is not self.ax or self.toolbar.mode != "":
            return
        if event.button in (2, 3):  # Pan
            self.pan = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())
            return
        if event.button == 1 and event.dblclick:
            self.fit(True)
            return
        if event.button != 1 or event.xdata is None:
            return
        # The field of view under the mouse (the one selected first), not the reference
        order = [self.active] + [i for i in range(len(self.sources) - 1, 0, -1) if i != self.active]
        for i in order:
            if i == 0 or self.sources[i] is None:
                continue
            left, right, bottom, top = self.extent(i)
            if left <= event.xdata <= right and bottom <= event.ydata <= top:
                self.active = i
                self.active_var.set(i)
                self.drag = (event.xdata, event.ydata, list(self.offsets[i]))
                self.schedule()
                return

    def motion(self, event):
        if self.pan is not None:
            x, y, (x0, x1), (y0, y1) = self.pan
            box = self.ax.bbox
            dx, dy = (event.x - x) * (x1 - x0) / box.width, (event.y - y) * (y1 - y0) / box.height
            self.ax.set_xlim(x0 - dx, x1 - dx)
            self.ax.set_ylim(y0 - dy, y1 - dy)
            return
        if self.drag is None or event.xdata is None:
            return
        x0, y0, start = self.drag
        self.offsets[self.active] = [start[0] + event.xdata - x0, start[1] + event.ydata - y0]
        self.show_offsets()
        self.schedule()

    def suggest(self):
        """The shift of the field of view selected over the others that best correlates their overlap: shown (the
        field of view moved there), not applied."""
        if self.active == 0 or self.grid_ is None:
            return
        try:
            radius_mm = float(self.radius.get().replace(",", "."))
        except ValueError:
            self.bell()
            return
        source = self.sources[self.active]
        if source is None:
            return
        pixel, moved, quality = source.pixel, np.zeros(2), None
        # On the pixels of the field of view selected; a second pass for the rest (the resampling)
        for radius in (radius_mm / pixel, 3.0):
            o = self.others_on(self.active)
            rows, cols = np.nonzero(np.isfinite(o))
            if rows.size < 400:
                self.suggest_label["text"] = "Too little overlap with the other fields of view: move it closer first."
                return
            pad = int(np.ceil(radius))
            r0, r1 = max(rows.min() - pad, 0), rows.max() + pad + 1
            c0, c1 = max(cols.min() - pad, 0), cols.max() + pad + 1
            shift, q = phase_shift(o[r0:r1, c0:c1], source.arr[r0:r1, c0:c1], radius)
            quality = q if quality is None else quality
            # o(x) ≈ a(x - shift): the field of view selected moved by shift (rows downwards: y upwards)
            step = np.array([shift[1] * pixel, -shift[0] * pixel])
            self.offsets[self.active][0] += step[0]
            self.offsets[self.active][1] += step[1]
            moved += step
        dx, dy = moved
        self.show_offsets()
        self.schedule()
        sure = "clear peak" if quality > 1.5 else "weak peak: check it carefully" if quality > 1.15 else \
            "no clear peak: likely wrong"
        self.suggest_label["text"] = (f"Moved by {dx:+.2f} mm, {dy:+.2f} mm ({sure}, ratio {quality:.2f}). Check it "
                                      "in Difference or Flicker; Cancel or move it back if it doesn't match.")

    # ------------------------------------------------------------------ the end

    def ok(self):
        for i, (x, y) in enumerate(self.offsets):
            x, y = round(x, 2), round(y, 2)

            def change(settings, x=x, y=y):
                camera = dict(settings.get("camera") or {})
                camera.update(x_offset_mm=x, y_offset_mm=y)
                settings["camera"] = camera
            self.dialog.apply_to(self.targets[i], change)
        self.dialog.refresh()
        self.close()

    def close(self):
        if self.flicker_job is not None:
            self.after_cancel(self.flicker_job)
        self.destroy()
