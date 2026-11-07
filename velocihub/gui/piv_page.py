"""Page of the PIV: the images pre-processed with the recipe of the page of the pre-processing, and the PIV software
with its settings. The tiles show the pair of images as the PIV sees it (with the interrogation windows and the
vectors computed), and the quality of the pairs computed. The results are in pixels: their figures (in the sensor
or the physical space) are those of the page of the post-processing.

The images are by default the ones exported with the recipe in the export folder of the pre-processing
(PreProcessing by default); if there are none, they are pre-processed (the POD modes evaluated if needed) and
exported there before the PIV. The images of another folder can be chosen: they are used as they are. The results
are saved in the subfolder PIV of the image folder, or in the folder chosen. The PIV runs in another process: the
interface stays usable, and the pairs computed are kept (by chunks of 100) when it is stopped.

The windows drawn on the image are a diagnostic, independent of the settings of the software: three nested windows
(64, 32 and 16 px by default; the mouse wheel doubles or halves them, the smallest 16 px), moved with a click or a
drag, where the particles are counted (piv/particles.py) to check that there are enough of them (about 10 per
window for the PIV). The page needs the page of the pre-processing; it shows its results in the page of the
post-processing if there is one."""

import bisect
import dataclasses
import json
import os
import shutil
import tempfile
import time
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib
import numpy as np
from matplotlib.patches import Rectangle

from .. import piv
from ..batch import valid_percent
from ..fields import MATH_FONT, load_vectors, nanmedian
from ..piv.particles import DEFAULT_DIAMETER, DEFAULT_SNR, MIN_PARTICLES, count_particles
from ..preprocessing import describe_images, folder_images, imread, imwrite, num2str, rasterize_masks, same_path
from ..project import format_duration, piv_results, run_time
from .batch_dialog import BatchStep, Field, PanelForm, PanelPage, run_batch
from .widgets import FigurePanel, FigureSelector, FolderEntry, HelpDot, Page, SidePanel
from ..prefs import getpref, setpref
from ..settings import (piv_from_project, piv_to_project, validate_piv)
from ..piv.displacements import (displacement_component, displacement_limits, grid_pixels, pair_particles,
                                 read_tracks)

# Seconds per pair to estimate the time of a PIV before the first one is measured
SECONDS_PER_PAIR = 0.6
# TracTrac: seconds per frame and pass, and the frames before and after a pair tracked to test it (the motion model
# needs them)
PTV_SECONDS_PER_FRAME = 0.3
PTV_TEST_FRAMES = 10
# While the PIV runs, the pair just computed is shown, unless the user browsed the pairs less than these seconds ago
BROWSE_PAUSE = 5.0
# Fraction of the time spent drawing the pairs computed while the PIV runs, at most: the window stays responsive
LIVE_LOAD = 0.3
# Color of the interrogation windows and of the vectors drawn over the images
OVERLAY_COLOR = "#ffcc00"
# Windows of the preview of the particles: the largest by default, the smallest size (the windows halved down to it
# collapse on one another there) and the number of windows
PREVIEW_SIZE = 64
PREVIEW_MIN = 16
PREVIEW_WINDOWS = 3
# Views of the tile of the pair (the menu of its bar): the first image of the pair, or the displacements of the pair:
# on the grid for a PIV, at the particles tracked for a PTV (on its grid while it runs: the tracks are saved at the end)
PAIR_VIEWS = {"image": "Image of the pair", "displacement": "Displacement |d|", "displacement_x": "Displacement d_x",
              "displacement_y": "Displacement d_y"}
# Symbol (LaTeX) and colour map of each map of the displacements (those of the page of the post-processing)
DISPLACEMENTS = {"displacement": ("|d|", "viridis"), "displacement_x": ("d_x", "viridis"),
                 "displacement_y": ("d_y", "RdBu_r")}
# Background of the maps on the grid (where there are no vectors) and of the particles tracked (mid grey: the dark
# ends of the colour maps and the white of RdBu both show on it)
GRID_BACKGROUND = "#d5d9e0"
PARTICLES_BACKGROUND = "#7a808c"
# Arrows across the width of the map |d| of a PIV, at most; maps of the results giving the colour limits of the
# displacements (the same for all the pairs)
ARROWS = 32
LIMIT_MAPS = 50


# Detection of the particles of the interrogation windows: signal-to-noise ratio and diameter
PARTICLE_SETTINGS = [
    ("particle_snr", "Particle S/N threshold", num2str(DEFAULT_SNR),
     ("Minimum height of a particle above the background, in units of the noise, in the image filtered with a "
      "particle image. 3 to 5 is usual: lower counts noise as particles, higher misses the faint ones.")),
    ("particle_diameter", "Particle diameter (px)", num2str(DEFAULT_DIAMETER),
     ("Diameter of the particle images, 2 to 4 px usual. It sets the filter (a Gaussian of standard deviation "
      "diameter / 4) and the neighbourhood of a maximum.")),
]
PARTICLES_HELP = ("Particles found in three nested windows of the pair shown, a diagnostic independent of the "
                  f"settings of the software: {PREVIEW_SIZE}, {PREVIEW_SIZE // 2} and {PREVIEW_SIZE // 4} px by "
                  "default. The mouse wheel over the image or the preview doubles or halves them (down to "
                  f"{PREVIEW_MIN} px, where they collapse on one another); click or drag on the image to move them. "
                  "The largest window is magnified with the smaller ones inside, and the particles are circled and "
                  "counted in each (the counts, the background and the noise are written on it). The PIV needs at "
                  f"least about {MIN_PARTICLES} particles per window.\n\nA particle is a local maximum of the "
                  "filtered image that stands at "
                  "least S/N times the noise above the background (the median of the window; the noise from its "
                  "median absolute deviation).")
# Labels of the axes of the images and of the maps of the page (LaTeX)
AXIS_LABELS = ("$x$ (px)", "$y$ (px)")
IMAGES_HELP = ("The PIV/PTV uses the pre-processed images by default, exporting them first if needed, or the original "
               "images when the recipe is empty. ... picks another folder whose images are used as they are (with "
               "the masks of its preprocessing.json, if any); ↺ goes back to the pre-processed images.")
OUTPUT_HELP = ("The vector fields (vectors.npz) and the settings (settings.json) are saved in the subfolder PIV "
               "of the image folder (PTV for TracTrac, with its tracks: tracks.npz); ... picks another folder, ↺ goes "
               "back to the default one.\n\nSingular runs the PIV/PTV of this "
               "project; after Stop, running again continues from the pairs computed. Batch runs the "
               "pre-processing and the PIV/PTV of several projects of the workspace, each with its own settings.")


REGRESSION_HELP = ("Optional fit of each vector field after the PIV/PTV with SPICY (constrained radial basis "
                   "functions, von Karman Institute). It fills the gaps of the invalid vectors and smooths the noise: "
                   "the post-processing then uses the regressed fields, while the quality of the PIV/PTV (valid "
                   "vectors, peak ratio or particles per cell) stays that of the PIV/PTV, whose fields vectors.npz "
                   "also keeps. It runs after the PIV/PTV, a few seconds per map, and continues from the maps fitted "
                   "when stopped.")


def single_map(fields, k):
    """The map k of vector fields, alone."""
    one = {name: getattr(fields, name)[k:k + 1] for name in ("U", "V", "valid", "u_px", "v_px", "peak_ratio",
                                                             "piv_valid", "particles", "t")
           if getattr(fields, name) is not None}
    return dataclasses.replace(fields, **one)


def engine_label(cls):
    """Name of an engine in the menus, with its method: PaIRS-UniNa (PIV), TracTrac (PTV)."""
    return f"{cls.name} ({cls.method})"


def describe_piv(settings):
    """The settings of the PIV of a batch, in a line."""
    try:
        engine = piv.get_engine(settings.get("engine"), **(settings.get("options") or {}))
        text = f"{engine.name}: {engine.describe(engine.check(settings.get('settings') or {}))}"
        regression = piv.check_regression(settings.get("regression"))
        if regression is not None:
            text += f"; then {piv.describe_regression(regression)}"
    except (ValueError, TypeError) as e:
        text, engine = f"PIV/PTV settings not valid: {e}", None
    folder = settings.get("output_folder") or (engine.results_folder if engine is not None else piv.PIV_FOLDER)
    return text + f"; results in {folder} (with the pre-processing settings)"


def piv_fields(settings):
    """The fields of the batch settings of the PIV (window Batch): its software, the settings of that engine, its
    results folder; the settings of the other engines of each project kept (hidden, not compared)."""
    engine = piv.get_engine(settings.get("engine"), **(settings.get("options") or {}))
    engines = tuple((key, engine_label(cls)) for key, cls in piv.ENGINES.items())
    fields = [Field(("engine",), "PIV/PTV software", "choice", engines, default=piv.DEFAULT_ENGINE,
                    help=("The PIV/PTV software of the projects. The settings below follow it; each project keeps its "
                          "settings of the other software."))]
    fields += [Field(("settings", p.key), p.label, p.kind if p.kind in ("bool", "choice") else "text",
                     tuple((c, c) for c in p.choices), help=p.help,
                     default=p.format(p.default) if p.kind != "bool" else bool(p.default))
               for p in engine.parameters]
    fields.append(Field(("regression", "method"), "Regression", "choice", piv.REGRESSIONS, help=REGRESSION_HELP,
                        default="none"))
    fields += [Field(("regression", "settings", p.key), f"SPICY: {p.label}", "choice" if p.kind == "choice" else "text",
                     tuple((c, c) for c in p.choices), help=p.help, default=p.format(p.default))
               for p in piv.SPICY_PARAMETERS]
    return fields + [Field(("output_folder",), "Export Path (Relative)", "folder", default=engine.results_folder,
                           help=(f"Folder of the PIV/PTV results, relative to each project "
                                 f"(↺: {engine.results_folder}). "
                                 "Set per project, not on folders."), project_only=True),
                     Field(("engine_settings",), "Settings of the other software", "hidden", project_only=True)]


class PIVPanel(SidePanel):
    """Side panel of the PIV: the images, the particles of the interrogation windows, the PIV software and its
    settings, the results and the buttons that run the PIV and test it on the pair shown. batch: the settings of the
    window Batch (BatchPIVForm): the software, its settings and the regression, without the images, the particles nor
    the run."""

    def __init__(self, master, page, batch=False):
        super().__init__(master)
        self.page, self.batch = page, batch
        saved = getpref("piv_settings", {}) if not batch else {}
        wrap = self.wrap

        if not batch:  # The run, the images and the particles: those of the page only
            # Results folder and run, at the bottom
            box = self.section("Run step", self.footer, OUTPUT_HELP)
            self.output = FolderEntry(box, page.pre.image_folder, self.default_output, "Folder of the PIV/PTV results...",
                                      on_change=page.output_changed)
            self.output.pack(fill=tk.X)
            self.results_label = ttk.Label(box, wraplength=wrap, justify=tk.LEFT)
            self.results_label.pack(anchor=tk.W, pady=(4, 0))
            self.progress_label = ttk.Label(box, wraplength=wrap, justify=tk.LEFT)
            self.progress_label.pack(anchor=tk.W, pady=(8, 0))
            self.progress = ttk.Progressbar(box, maximum=1.0)
            self.progress.pack(fill=tk.X, pady=(4, 0))
            row = ttk.Frame(box)
            row.pack(anchor=tk.E, pady=(6, 0))
            self.pair_button = ttk.Button(row, text="Run on this pair   (F6)", command=page.run_pair)
            self.pair_button.pack(side=tk.LEFT, padx=(0, 6))
            ttk.Button(row, text="Batch...", command=page.batch).pack(side=tk.LEFT, padx=(0, 6))
            self.run_button = ttk.Button(row, text="Singular   (F5)", style="Export.TButton", command=page.run_or_stop)
            self.run_button.pack(side=tk.LEFT)

            # Images, at the top
            box = self.section("Images", self.header)
            row = self.row(box, pady=(0, 2))
            ttk.Label(row, text="Images folder").pack(side=tk.LEFT)
            HelpDot(row, IMAGES_HELP).pack(side=tk.LEFT, padx=(4, 0))
            self.images = FolderEntry(box, page.pre.image_folder, page.default_images_folder, "Images of the PIV/PTV...",
                                      on_change=page.images_folder_changed)
            self.images.pack(fill=tk.X)
            self.images_label = ttk.Label(box, wraplength=wrap, justify=tk.LEFT)
            self.images_label.pack(anchor=tk.W, pady=(4, 0))

            # Particles of the interrogation windows, at their position on the image shown
            box = self.section("Particles in the windows", help=PARTICLES_HELP)
            grid = self.grid(box)
            self.particle_vars = {}
            for key, label, default, help in PARTICLE_SETTINGS:
                self.particle_vars[key] = tk.StringVar(self, saved.get(key, default))
                self.setting(grid, label, self.particle_vars[key], help, command=page.particles_changed)

        # PIV software and its settings
        box = self.section("PIV/PTV software")
        row = self.row(box, pady=(0, 4))
        self.engine_names = [cls.name for cls in piv.ENGINES.values()]
        names = self.engine_names
        # The name of the engine (as saved), and its label in the menu (with its method)
        self.engine_name = tk.StringVar(self, saved.get("engine") if saved.get("engine") in names else names[0])
        labels = {cls.name: engine_label(cls) for cls in piv.ENGINES.values()}
        self.engine_label_var = tk.StringVar(self, labels[self.engine_name.get()])
        self.engine_name.trace_add("write", lambda *args: self.engine_label_var.set(labels[self.engine_name.get()]))
        self.engine_box = ttk.Combobox(row, textvariable=self.engine_label_var, values=list(labels.values()),
                                       state="readonly", width=20)
        self.engine_box.pack(side=tk.LEFT)
        self.engine_box.bind("<<ComboboxSelected>>", lambda event: self.engine_chosen(labels))
        self.python_button = ttk.Button(row, text="Python...", command=page.choose_python)
        self.python_button.pack(side=tk.LEFT, padx=(6, 0))
        HelpDot(row, ("PaIRS-UniNa is a compiled library for Python 3.10 to 3.13. It runs in its own process, "
                      "with the Python of an environment where it is installed (found among the conda "
                      "environments). Python... picks another python.exe.")).pack(
            side=tk.LEFT, padx=(4, 0))
        self.engine_label = ttk.Label(box, wraplength=wrap, justify=tk.LEFT)
        self.engine_label.pack(anchor=tk.W)
        self.engine_grid = self.grid(box)
        self.engine_vars = {}
        self.engine_widgets = []

        self.build_engine(saved.get("engine_settings", {}))
        self.regression_box(saved.get("regression") or {})

    def regression_box(self, saved):
        """Box of the regression after the PIV software: none or SPICY, with its settings (greyed without it)."""
        box = self.section("Regression", help=REGRESSION_HELP)
        row = self.row(box, pady=(0, 4))
        method = saved.get("method")
        self.regression = tk.StringVar(self, method if method in dict(piv.REGRESSIONS) else "none")
        self.regression_buttons = []
        for key, name in piv.REGRESSIONS:
            button = ttk.Radiobutton(row, text=name, value=key, variable=self.regression,
                                     command=self.regression_changed)
            button.pack(side=tk.LEFT, padx=(0, 12))
            self.regression_buttons.append(button)
        # Why SPICY can't run, if it can't (shown only then)
        self.regression_label = ttk.Label(box, wraplength=self.wrap, justify=tk.LEFT, foreground="#b3261e")
        grid = self.grid(box)
        self.regression_grid = grid
        self.regression_vars, self.regression_widgets = {}, []
        for p in piv.SPICY_PARAMETERS:
            var = self.regression_vars[p.key] = tk.StringVar(self)
            if p.kind == "choice":
                r = grid.grid_size()[1]
                ttk.Label(grid, text=p.label).grid(row=r, column=0, sticky=tk.W, padx=(0, 4), pady=1)
                widget = ttk.Combobox(grid, textvariable=var, values=p.choices, state="readonly", width=10)
                widget.grid(row=r, column=1, sticky=tk.E, pady=1)
                widget.bind("<<ComboboxSelected>>", lambda event: self.page.settings_changed())
                HelpDot(grid, p.help).grid(row=r, column=2, padx=(4, 0))
            else:
                widget = self.setting(grid, p.label, var, p.help, command=self.page.settings_changed)
            self.regression_widgets.append(widget)
        self.show_regression(saved)

    def show_regression(self, saved):
        """Show regression settings ({"method", "settings"}; the defaults for those missing or invalid)."""
        method = saved.get("method")
        self.regression.set(method if method in dict(piv.REGRESSIONS) else "none")
        values = saved.get("settings") or {}
        for p in piv.SPICY_PARAMETERS:
            try:
                value = p.format(p.parse(values[p.key]) if p.key in values else p.default)
            except ValueError:
                value = p.format(p.default)
            self.regression_vars[p.key].set(value)
        self.update_regression()

    def regression_changed(self):
        self.update_regression()
        self.page.settings_changed()

    def update_regression(self, running=False):
        """The settings of SPICY greyed without the regression (or while the PIV runs); whether it can run."""
        on = self.regression.get() != "none"
        for widget in self.regression_widgets:
            readonly = isinstance(widget, ttk.Combobox)
            widget.configure(state=("readonly" if readonly else tk.NORMAL) if on and not running else tk.DISABLED)
        for button in self.regression_buttons:
            button.configure(state=tk.DISABLED if running else tk.NORMAL)
        ok, text = piv.spicy_available() if on else (True, "")
        self.regression_label.configure(text=text)
        if ok:
            self.regression_label.pack_forget()
        elif not self.regression_label.winfo_manager():
            self.regression_label.pack(anchor=tk.W, before=self.regression_grid)

    def regression_settings(self):
        """Regression entered (as text)."""
        return {"method": self.regression.get(), "settings": {k: var.get() for k, var in self.regression_vars.items()}}

    def read_regression(self):
        """Regression checked (piv.check_regression; None: none): ValueError with a message if it is invalid."""
        try:
            return piv.check_regression(self.regression_settings())
        except ValueError as e:
            raise ValueError(f"Regression: {e}") from None

    def engine(self):
        """Engine selected, with its options (the Python interpreter of PaIRS-UniNa chosen by the user)."""
        cls = next(c for c in piv.ENGINES.values() if c.name == self.engine_name.get())
        options = getpref("piv_engine_options", {}).get(cls.key, {})
        return cls(**options)

    def default_output(self):
        """Default folder of the results: the folder of the results of the engine (PIV, PTV) in the image folder."""
        root = self.page.pre.image_folder()
        engine = self.engine() if hasattr(self, "engine_name") else None
        return os.path.join(root, engine.results_folder if engine is not None else piv.PIV_FOLDER) if root else None

    def engine_chosen(self, labels):
        """An engine chosen in the menu (by its label)."""
        self.engine_name.set(next(name for name, label in labels.items() if label == self.engine_label_var.get()))
        self.engine_selected()

    def engine_selected(self):
        """Another engine: its settings, and its default results folder (unless another folder was chosen)."""
        # The folder of the previous engine, when its settings were built (the menu has already changed the
        # engine when this is called)
        if self.batch:  # Its settings: those of the projects (their defaults otherwise), told to the form
            self.build_engine({})
            self.page.settings_changed()
            return
        before = self.engine_output
        self.build_engine()
        self.output.refresh()
        if not same_path(before or "", self.output.get() or ""):
            self.page.output_changed()
        else:
            self.page.update_particles()  # The rule of thumb of the PIV shown or not

    def build_engine(self, values=None):
        """Settings of the engine selected, with the values given ({engine key: {setting: value}}), those saved
        for the next session, or the defaults."""
        for widget in self.engine_grid.winfo_children():
            widget.destroy()
        engine = self.engine()
        self.engine_output = self.output.get() if not self.batch else None
        if values is None:
            values = getpref("piv_settings", {}).get("engine_settings", {})
        saved = values.get(engine.key, {})
        self.engine_vars = {}
        self.engine_widgets = []
        for p in engine.parameters:
            try:
                value = p.format(p.parse(saved[p.key]) if p.key in saved else p.default)
            except ValueError:
                value = p.format(p.default)
            var = tk.BooleanVar(self, value) if p.kind == "bool" else tk.StringVar(self, value)
            self.engine_vars[p.key] = var
            if p.kind == "choice":
                r = self.engine_grid.grid_size()[1]
                ttk.Label(self.engine_grid, text=p.label).grid(row=r, column=0, sticky=tk.W, padx=(0, 4), pady=1)
                widget = ttk.Combobox(self.engine_grid, textvariable=var, values=p.choices, state="readonly", width=10)
                widget.grid(row=r, column=1, sticky=tk.E, pady=1)
                widget.bind("<<ComboboxSelected>>", lambda event: self.page.settings_changed())
                HelpDot(self.engine_grid, p.help).grid(row=r, column=2, padx=(4, 0))
            else:
                widget = self.setting(self.engine_grid, p.label, var, p.help, command=self.page.settings_changed)
            self.engine_widgets.append(widget)
        self.python_button.configure(state=tk.NORMAL if hasattr(engine, "python") else tk.DISABLED)
        self.update_engine_label()

    def update_engine_label(self):
        ok, text = self.engine().available()
        self.engine_label.configure(text=text, foreground="#1b7a3e" if ok else "#b3261e")

    # ------------------------------------------------------------------ settings

    def read_particles(self):
        """Signal-to-noise ratio and diameter of the particles entered: ValueError with a message if invalid."""
        try:
            snr, diameter = (float(self.particle_vars[key].get()) for key in ("particle_snr", "particle_diameter"))
        except ValueError:
            raise ValueError("Particles: enter numbers for the S/N threshold and the diameter") from None
        if not (snr > 0 and diameter > 0):
            raise ValueError("Particles: the S/N threshold and the diameter must be positive")
        return snr, diameter

    def read_engine(self):
        """Engine and its settings checked: ValueError with a message if they are invalid."""
        engine = self.engine()
        settings = {key: var.get() for key, var in self.engine_vars.items()}
        try:
            return engine, engine.check(settings)
        except ValueError as e:
            raise ValueError(f"{engine.name} settings: {e}") from None

    @staticmethod
    def pairs(n_images):
        """Pairs of the images: each image and the next one."""
        return piv.Camera(1.0).pairs(n_images)

    def settings(self):
        """Settings entered (as text): particles, engine and its settings."""
        engine_settings = getpref("piv_settings", {}).get("engine_settings", {})
        engine_settings[self.engine().key] = {key: var.get() for key, var in self.engine_vars.items()}
        return {**{key: var.get() for key, var in self.particle_vars.items()},
                "engine": self.engine_name.get(),
                "engine_settings": engine_settings,
                "regression": self.regression_settings()}

    def show(self, settings):
        """Show settings (those of settings())."""
        for key, var in self.particle_vars.items():
            if key in settings:
                var.set(settings[key])
        if settings.get("engine") in self.engine_names:
            self.engine_name.set(settings["engine"])
        self.build_engine(settings.get("engine_settings") or {})
        self.output.refresh()
        self.show_regression(settings.get("regression") or {})

    def save(self):
        """The settings entered are kept for the next session (the settings of the new projects)."""
        setpref("piv_settings", self.settings())

    def set_running(self, running, testing=False):
        """Settings disabled while the PIV runs; the button Run becomes Stop."""
        state = tk.DISABLED if running or testing else tk.NORMAL
        for widget in self.engine_widgets + [self.python_button]:
            widget.configure(state="readonly" if isinstance(widget, ttk.Combobox) and state == tk.NORMAL else state)
        self.engine_box.configure(state=tk.DISABLED if running or testing else "readonly")
        self.update_regression(running=running or testing)
        self.output.configure_state(not (running or testing))
        self.images.configure_state(not (running or testing))
        self.pair_button.configure(state=state)
        self.run_button.configure(text="Stop   (F5)" if running else "Singular   (F5)",
                                  state=tk.DISABLED if testing else tk.NORMAL)


def panel_labels():
    """The labels of the settings of the side panel of the PIV in the window Batch: {label: [paths]} (the settings of
    all the software, those of SPICY, the regression)."""
    labels = {"Regression": [("regression", "method")]}
    for cls in piv.ENGINES.values():
        for p in cls.parameters:
            labels.setdefault(p.label, []).append(("settings", p.key))
    for p in piv.SPICY_PARAMETERS:
        labels.setdefault(p.label, []).append(("regression", "settings", p.key))
    return labels


class PIVBatchPage(PanelPage):
    """What the side panel of the PIV tells its page, in the window Batch: each change of the settings."""

    def settings_changed(self):
        self.commit()


class BatchPIVForm(PanelForm):
    """The settings of the PIV/PTV of the node selected in the window Batch (PanelForm): those of the side panel of the
    page (PIVPanel in batch mode: the software, its settings, the regression), the results folder above it. Another
    software: told alone (its settings: those the projects had, else its defaults)."""

    PATHS, LABELS, PRIMARY = ("engine", "settings", "regression"), panel_labels(), (("engine",),)

    def make_panel(self):
        return PIVPanel(self, PIVBatchPage(self), batch=True)

    def load_panel(self, settings, unavailable=None, limits=None):
        panel = self.panel
        try:
            engine = piv.get_engine(settings.get("engine"))
        except ValueError:
            engine = piv.get_engine(None)
        panel.engine_name.set(engine.name)
        panel.build_engine({engine.key: dict(settings.get("settings") or {})})
        panel.show_regression(settings.get("regression") or {})

    def restore_states(self, unavailable):
        panel = self.panel
        panel.engine_box.configure(state="readonly")
        panel.python_button.configure(state=tk.DISABLED)  # The interpreter: chosen in the page
        panel.update_regression()

    def record(self):
        """The settings entered in the panel, as batch settings (as text, checked by the window: validate_piv)."""
        panel = self.panel
        return {"engine": panel.engine().key,
                "settings": {key: var.get() for key, var in panel.engine_vars.items()},
                "regression": panel.regression_settings()}

    def normalize(self, field, value):
        """The values compared as their software reads them (16 and "16" the same)."""
        path = field.path
        if path == ("engine",):
            try:
                return piv.get_engine(value).key
            except ValueError:
                return value
        parameters = {}
        if path[0] == "settings":
            try:
                parameters = {p.key: p for p in piv.get_engine(self.shown.get("engine")).parameters}
            except ValueError:
                parameters = {}
        elif path[:2] == ("regression", "settings"):
            parameters = {p.key: p for p in piv.SPICY_PARAMETERS}
        p = parameters.get(path[-1])
        if p is None or value is None:
            return value
        try:
            return p.parse(value)
        except (TypeError, ValueError):
            return value


class PIVPage(Page):
    """Page of the PIV: tiles of the pair shown (with its vectors) and of the quality of the pairs, and the side
    panel. The arrow keys (over any tile) and the slider browse the pairs: the tiles follow the pair. The pair
    shown can be computed alone (run_pair), to check the settings."""

    key = "piv"
    title = "PIV/PTV"
    project_key = "piv"

    def __init__(self, master, app):
        super().__init__(master, app)
        # Images, recipe and masks of the pre-processing
        self.pre = app.page("pre")
        self.index = 0  # Pair shown (0-based)
        self.run = None  # PIV running
        self.test = None  # PIV of the pair shown running: (PIVRun, temporary folder, pair)
        self.test_result = None  # (pair, vector fields) of the last pair computed alone
        self.test_particles = None  # (pair, its particles tracked: pair_particles) of the last pair tracked alone
        self.live = None  # (pair, vector fields) of the last pair computed by the PIV running
        self.live_message = None  # Its message "field" not shown yet
        self.browsed = 0.0  # Time the user last browsed the pairs
        self.live_drawn = (0.0, 0.0)  # Time the last pair computed was drawn, and the time it took
        self.chosen = (None, [], None)  # Images of the folder chosen: (folder, paths, preprocessing.json)
        self.results = None  # Vector fields of the results folder (vectors.npz), loaded when shown
        self.results_checked = False
        # Quality of the pairs computed: valid vectors (%) and median peak ratio (PIV), or valid cells (%) and
        # particles tracked (PTV: monitor_ptv); the curves drawn and the lengths of the tracks of the results
        self.monitor = None
        self.monitor_ptv = False
        self.monitor_drawn = 0.0
        self.monitor_line = None
        self.monitor_curves = None
        self.track_lengths = (None, None)  # (tracks.npz and its time, lengths of its tracks in frames)
        self.source_cache = None  # Images of the PIV (source()) and the settings they depend on
        self.overlay = []
        self.last_rate = None
        self.timer = None  # Next poll of the PIV process
        self.size_cache = (None, None)
        # Interrogation windows: their centre (x, y in pixels from the top-left corner of the image; None: the
        # centre of the image), moved with the mouse, and the image shown, where their particles are counted
        self.ia_center = None
        self.preview_size = PREVIEW_SIZE  # Largest window of the preview (px)
        self.dragging = False
        self.pair_image = None
        # View of the tile of the pair (PAIR_VIEWS), the one drawn and the size of its image; the particles tracked
        # of the results (tracks.npz and its time, read_tracks) and the colour limits of their displacements
        self.view = "image"
        self.drawn = (None, None)
        self.tracks_cache = (None, None)
        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        left = ttk.PanedWindow(body, orient=tk.VERTICAL)
        body.add(left, weight=1)
        self.panel = PIVPanel(body, self)
        body.add(self.panel, weight=0)
        # The image of the pair, and the preview of its windows next to it
        self.top = ttk.PanedWindow(left, orient=tk.HORIZONTAL)
        left.add(self.top, weight=4)
        self.figure_pair = FigurePanel(self.top, [("Colormap", self.toggle_colormap)], "compressed", figsize=(6, 4),
                                       clabel="Intensity (grey levels)", item="Pair", on_slide=self.slide_to,
                                       on_activate=lambda: self.set_active_tile(self.figure_pair),
                                       axis_labels=AXIS_LABELS)
        self.top.add(self.figure_pair, weight=3)
        # Menu of the views of the pair, first in the bar (as the figures of the tiles of the post-processing)
        bar = self.figure_pair.bar
        self.view_selector = FigureSelector(bar, [(None, tuple(PAIR_VIEWS))], PAIR_VIEWS, self.view_selected,
                                            width=18)
        self.view_selector.pack(side=tk.LEFT, padx=(2, 6), before=bar.winfo_children()[0])
        self.view_selector.show(self.view)
        self.figure_windows = FigurePanel(self.top, (), None, figsize=(4, 4))
        self.top.add(self.figure_windows, weight=2)
        self.figure_monitor = FigurePanel(left, (), "constrained", figsize=(8, 1.8),
                                          on_activate=lambda: self.set_active_tile(self.figure_monitor))
        left.add(self.figure_monitor, weight=1)
        # A click on the graph of the quality of the pairs shows the pair
        self.figure_monitor.canvas.mpl_connect("button_press_event", self.monitor_clicked)
        # A click or a drag on the image moves the interrogation windows
        canvas = self.figure_pair.canvas
        canvas.mpl_connect("button_press_event", self.windows_pressed)
        canvas.mpl_connect("motion_notify_event", self.windows_dragged)
        canvas.mpl_connect("button_release_event", self.windows_released)
        # The mouse wheel over the image or the preview doubles or halves the windows
        for figure in (self.figure_pair, self.figure_windows):
            figure.canvas.mpl_connect("scroll_event", self.preview_scrolled)
        self.tiles = (self.figure_pair, self.figure_monitor)
        self.set_active_tile(self.figure_pair)
        self.reset_tiles()

    # ------------------------------------------------------------------ page

    def menus(self):
        return [("Settings", [("PaIRS Python interpreter...", None, self.choose_python)]),
                ("PIV/PTV", [("Run the PIV/PTV · Stop", "F5", self.run_or_stop),
                         ("Run on the pair shown", "F6", self.run_pair), None,
                         ("Clear the cache", None, self.clear_cache)])]

    def clear_cache(self):
        """File > Clear cache: the pairs computed by a PIV stopped (_progress of the results folder, after a
        confirmation), and the images, results and sizes kept in memory (read again)."""
        app = self.app
        if self.busy():
            app.msgbox("The PIV/PTV is running: stop it first.", "Clear the cache", "warn")
            return
        progress = os.path.join(self.output_folder() or "", "_progress")
        text = "PIV/PTV cache cleared"
        if self.output_folder() and os.path.isdir(progress):
            chunks = [f for f in os.listdir(progress) if f.startswith("chunk_")]
            if chunks and app.questdlg(f"Delete the {len(chunks) * piv.CHUNK} pairs (at most) computed by the PIV/PTV "
                                       f"stopped? Run then starts again from the first pair.\n\n{progress}",
                                       "Clear the cache", ["Delete", "Keep them"], "Keep them") != "Delete":
                chunks = None
            if chunks is not None:
                shutil.rmtree(progress, ignore_errors=True)
                text += f": {progress} deleted"
        self.source_cache, self.size_cache = None, (None, None)
        self.chosen = (None, [], None)
        self.results, self.results_checked, self.monitor = None, False, None
        self.test_result = self.test_particles = None
        self.tracks_cache = (None, None)
        if app.current_page() == self.key:
            self.shown()
        else:
            self.update_results_label()
        app.update_status(text)

    def folders(self):
        return [("PIV/PTV results", self.output_folder)]

    def batch_settings(self):
        """The settings of the PIV of the interface for a batch: the engine, its options, its settings (as entered),
        the regression and the results folder (relative to the image folder)."""
        engine = self.panel.engine()
        root = self.pre.image_folder()
        output = self.output_folder()
        return {"engine": engine.key, "options": getpref("piv_engine_options", {}).get(engine.key, {}),
                "settings": {key: var.get() for key, var in self.panel.engine_vars.items()},
                "settings_engine": engine.key, "regression": self.panel.regression_settings(),
                "output_folder": os.path.relpath(output, root) if root and output else None}

    def batch_step(self):
        return BatchStep("piv", "PIV/PTV", lambda project: project.images, self.batch_settings, describe_piv,
                         lambda project, settings, settings_of: {"preprocessing": settings_of("preprocessing",
                                                                                             project),
                                                                 "piv": settings},
                         fields=piv_fields, validate=validate_piv, to_project=("piv", piv_to_project),
                         from_project=piv_from_project, form=BatchPIVForm)

    def batch(self):
        """Run step > Batch...: the pre-processing and the PIV of projects of the workspace."""
        run_batch(self.app, self.batch_step())

    def uses_folder(self, folder):
        """The PIV running reads the images of this folder."""
        return self.run is not None and same_path(os.path.dirname(self.run.images[0]), folder)

    def project_state(self, rel):
        images = self.panel.images
        return {"settings": self.panel.settings(),
                "images_folder": None if images.is_default() else rel(images.get()),
                "output_folder": rel(self.output_folder()), "results": self.results_summary(rel),
                "window_center": list(self.ia_center) if self.ia_center is not None else None,
                "window_size": self.preview_size, "pair_view": self.view}

    def restore(self, data, path):
        view = data.get("pair_view")
        self.view = view if view in PAIR_VIEWS else "image"
        self.view_selector.show(self.view)
        if data.get("settings"):
            self.panel.show(data["settings"])
        self.panel.images.set(path(data.get("images_folder")))
        self.panel.output.set(path(data.get("output_folder")))
        center = data.get("window_center")
        self.ia_center = (tuple(float(v) for v in center) if isinstance(center, list) and len(center) == 2
                          else None)
        size = data.get("window_size")
        self.preview_size = (int(size) if isinstance(size, int) and size >= PREVIEW_MIN
                             and (size & (size - 1)) == 0 else PREVIEW_SIZE)
        self.results = None
        self.results_checked = False

    def results_summary(self, rel):
        """The results of the results folder, as kept in the project."""
        path = self.results_file()
        if path is None:
            return None
        f = self.results
        if f is not None and same_path(f.source, path):
            return piv_results(rel(path), {**f.settings, "n_pairs": f.n}, valid_percent(f))
        try:
            with open(os.path.join(os.path.dirname(path), "settings.json"), encoding="utf-8") as file:
                settings = json.load(file)
        except (OSError, ValueError):
            settings = {}
        return piv_results(rel(path), settings)

    # ------------------------------------------------------------------ images and pairs

    def image_size(self):
        """Size (width, height) of the images, None without images."""
        paths = self.images()
        if not paths:
            return None
        if self.size_cache[0] != paths[0]:
            from PIL import Image
            with Image.open(paths[0]) as im:
                self.size_cache = (paths[0], im.size)
        return self.size_cache[1]

    def pair_list(self):
        """Pairs of images browsed: those of the PIV with the steps entered."""
        return self.panel.pairs(self.n_images())

    def default_images_folder(self):
        """Folder of the images of the pre-processing: processed with its recipe (export folder), or the original
        ones if the recipe is empty."""
        pre = self.pre
        if not pre.Check_imsel:
            return None
        return pre.path_in if pre.recipe_empty(pre.n_modes_removed()) else pre.export_folder()

    def images_chosen(self):
        """The images of another folder are chosen (used as they are)."""
        return self.pre.Check_imsel and not self.panel.images.is_default()

    def chosen_images(self):
        """Images of the folder chosen and the description of their pre-processing (None if there is none)."""
        folder = self.panel.images.get()
        if self.chosen[0] != folder:
            paths, manifest = [], None
            if same_path(folder, self.pre.path_in):
                paths = self.pre.images
            elif os.path.isdir(folder):
                paths, manifest = folder_images(folder)
            self.chosen = (folder, paths, manifest)
        return self.chosen[1], self.chosen[2]

    def images(self):
        """Paths of the images of the PIV (those to be exported: the original ones)."""
        if not self.pre.Check_imsel:
            return []
        kind, where = self.source()
        return where if kind != "export" else self.pre.images

    def n_images(self):
        return len(self.images())

    def masked(self):
        """Masked pixels of the images of the PIV: those of the masks of their recipe."""
        size = self.image_size()
        if size is None:
            return None
        if self.images_chosen():
            manifest = self.chosen_images()[1]
            polygons = ((manifest or {}).get("recipe") or {}).get("masks") or []
            return rasterize_masks(polygons, size[::-1]) if polygons else None
        return self.pre.masked_pixels(size[::-1])

    def images_folder_changed(self):
        """Images of another folder chosen (or back to those of the pre-processing)."""
        self.chosen, self.source_cache = (None, [], None), None
        self.test_result = None
        if self.app.current_page() == self.key:
            self.shown()
        self.app.save_project()

    def output_folder(self):
        """Folder of the results: the subfolder PIV of the image folder, or the folder chosen."""
        return self.panel.output.get()

    def output_changed(self):
        """Another results folder: its results are shown."""
        self.results = None
        self.results_checked = False
        self.monitor = None
        if self.app.current_page() == self.key:
            self.shown()
        else:
            self.update_results_label()
        self.app.save_project()

    def images_changed(self):
        """New training images (or none): the pairs, the results of their folder, the progress of the last PIV and the
        tiles are reset."""
        self.index = 0
        self.ia_center = None
        self.results = None
        self.results_checked = False
        self.test_result = None
        self.monitor = None
        self.chosen, self.source_cache = (None, [], None), None
        self.panel.images.refresh()
        self.panel.output.refresh()
        # The end of the PIV of the previous images ("Done: ..."), unless a PIV still runs: its progress is shown
        if not self.busy():
            self.panel.progress_label["text"] = ""
            self.panel.progress["value"] = 0
        if self.app.current_page() == self.key:
            self.shown()
        else:
            self.update_results_label()
            self.reset_tiles()

    def reset_tiles(self):
        self.pair_image = None
        self.update_particles()
        self.figure_pair.message("Image pairs\n\nSelect the training images in the page Pre-processing")
        self.figure_pair.set_slider(1, 0)
        self.figure_monitor.message("Quality of the pairs computed: valid vectors and correlation peak ratio")
        self.overlay = []
        self.drawn = (None, None)
        self.monitor_line = None

    def set_active_tile(self, tile):
        """The tiles all show the pair: the one under the mouse is active, the others follow it."""
        for t in self.tiles:
            t.set_active(True if t is tile else "linked")

    def modes_missing(self):
        """The background removal of the recipe needs modes which aren't evaluated yet."""
        return self.pre.n_modes_removed() > self.pre.NMod_saved

    def source(self):
        """Images of the PIV and their description: ("chosen", paths) in the folder chosen, ("original", paths),
        ("preprocessed", paths) or ("export", folder) if they must be pre-processed and exported first."""
        pre = self.pre
        key = (self.panel.images.get(), json.dumps(pre.recipe_record(pre.n_modes_removed())), pre.export_folder(),
               len(pre.file_in))
        if self.source_cache is None or self.source_cache[0] != key:
            self.source_cache = (key, self.find_source())
        return self.source_cache[1]

    def find_source(self):
        pre = self.pre
        if self.images_chosen():
            return "chosen", self.chosen_images()[0]
        n_modes = pre.n_modes_removed()
        if pre.recipe_empty(n_modes):
            return "original", pre.images
        paths = pre.preprocessed_images(n_modes)
        if paths is not None:
            return "preprocessed", paths
        return "export", pre.export_folder()

    def update_images_label(self):
        """Images of the PIV: number, size, bit depth and pre-processing."""
        pre = self.pre
        self.panel.images.refresh()
        if not pre.Check_imsel:
            self.panel.images_label["text"] = "No images: select the training images in the page Pre-processing."
            return
        kind, where = self.source()
        if kind == "chosen":
            if not where:
                text = "No images in this folder"
            else:
                manifest = self.chosen_images()[1]
                text = describe_images(where) + "\nUsed as they are" + (
                    " (pre-processed, described by preprocessing.json)" if manifest else "")
        elif kind == "original":
            text = describe_images(where) + "\nNo Pre-processing"
        elif kind == "preprocessed":
            text = describe_images(where) + "\nPre-processing done"
        else:
            text = (describe_images(pre.images) + "\nPre-Process will be run before PIV/PTV")
        self.panel.images_label["text"] = text

    def update_results_label(self):
        if not self.pre.Check_imsel:
            self.panel.results_label["text"] = ""
            return
        if self.results is not None:
            f = self.results
            seconds = run_time(f.settings)
            residual = f.settings.get("regression_median_residual_px")
            text = (f"{f.n} pairs ({f.settings.get('date', '')}), "
                    f"{valid_percent(f):.1f} % valid vectors"
                    + (f", run time {format_duration(seconds)}" if seconds is not None else "")
                    + (f"; PTV with {f.settings.get('engine')}, {np.mean(f.particles[:, f.status == 0]):.2f} particles "
                       "per cell" if f.particles is not None and (f.status == 0).any() else "")
                    + (f"; regressed with SPICY (median residual {residual:.2f} px)" if f.settings.get("regression")
                       and residual is not None else "; regressed with SPICY" if f.settings.get("regression") else ""))
        elif self.results_file() is not None:
            text = f"{piv.VECTORS_FILE} saved before (shown when the page is open)"
        else:
            text = "No results yet"
        self.panel.results_label["text"] = text

    def results_file(self):
        folder = self.output_folder()
        path = os.path.join(folder, piv.VECTORS_FILE) if folder else None
        return path if path and os.path.isfile(path) else None

    def shown(self):
        """The page is shown: updated with the recipe of the pre-processing, the results of the folder, and the pair
        of the image looked at in the other pages."""
        self.chosen, self.source_cache = (None, [], None), None
        self.panel.output.refresh()
        self.update_images_label()
        self.panel.update_engine_label()
        if not self.results_checked and self.run is None:
            self.results_checked = True
            path = self.results_file()
            if path is not None:
                try:
                    self.results = load_vectors(path, pixels=True)
                except (OSError, ValueError, KeyError) as e:
                    self.app.update_status(f"PIV/PTV results not read ({path}): {e}")
        self.update_results_label()
        if not self.n_images():
            self.reset_tiles()
            return
        pairs = self.pair_list()
        if pairs and not (0 <= self.index < len(pairs) and pairs[self.index][0] == self.app.image_index):
            self.index = max(bisect.bisect_right([a for a, _ in pairs], self.app.image_index) - 1, 0)
        self.show_pair()
        self.show_monitor()

    def slide_to(self, i):
        # The slider moved by show_pair calls this again: nothing to do then
        if i - 1 != self.index:
            self.browsed = time.monotonic()
            self.show_pair(i - 1)

    def arrow(self, step):
        if self.n_images():
            self.browsed = time.monotonic()
            self.show_pair(self.index + step)

    def monitor_clicked(self, event):
        if (event.inaxes is not None and event.xdata is not None and self.figure_monitor.toolbar.mode == ""
                and self.n_images()):
            self.browsed = time.monotonic()
            self.show_pair(int(round(event.xdata)) - 1)

    def input_image(self, k):
        """Image k as the PIV sees it: the original, the image pre-processed and exported, or the image processed
        with the recipe (as it would be exported; without the background removal if its modes aren't
        evaluated yet)."""
        pre = self.pre
        kind, where = self.source()
        if kind != "export":
            return imread(where[k])
        n_modes = pre.n_modes_removed()
        return pre.process_image(pre.images[k], k, pre.load_modes(0 if self.modes_missing() else n_modes))[0]

    def show_pair(self, index=None):
        """Show the first image of the pair, with the interrogation windows and its vectors, or the map of its
        displacements (the view chosen in the menu of the tile) if they are computed."""
        pairs = self.pair_list()
        if not pairs:
            self.reset_tiles()
            return
        self.index = int(np.clip(self.index if index is None else index, 0, len(pairs) - 1))
        a, b = pairs[self.index]
        self.app.image_index = a
        img = self.input_image(a)
        source = self.source()[0]
        kind = {"chosen": "folder chosen", "original": "original image", "preprocessed": "pre-processed",
                "export": ("pre-processed, without the background removal until its modes are evaluated"
                           if self.modes_missing() else "pre-processed")}[source]
        title = (f"Pair {self.index + 1} of {len(pairs)}: images {a + 1} and {b + 1} ({kind})\n"
                 f"{os.path.basename(self.images()[a])}")
        panel = self.figure_pair
        shape = img.shape[:2]
        data = self.pair_displacements() if self.view != "image" else None
        if data is not None:
            self.draw_displacements(data, shape, f"Pair {self.index + 1} of {len(pairs)}: images {a + 1} and {b + 1}")
        else:
            if self.view != "image":
                title += ("\nNo displacements for this pair yet: Run the PIV/PTV (F5) or Run on the pair shown (F6)"
                          if self.results is None or self.results.u_px is not None
                          else "\nThe displacements in pixels of these results are unknown")
            if self.drawn != ("image", shape) or panel.im is None:
                limits = self.view_limits(shape)
                with matplotlib.rc_context(MATH_FONT):
                    panel.imagesc(img, "gray", (0, max(1, float(np.max(img)))), title, colorbar=False)
                self.overlay = []
                self.restore_limits(limits)
                self.drawn = ("image", shape)
            else:
                panel.set_image(img, (0, max(1, float(np.max(img)))), title)
        panel.set_slider(self.index + 1, len(pairs))
        self.pair_image = img
        self.draw_overlay(shape)
        self.update_particles()
        self.show_monitor(current_only=True)

    def view_selected(self, key):
        """A view of the pair chosen in the menu of its tile: the image or a map of the displacements."""
        self.view = key
        self.view_selector.show(key)
        if self.n_images():
            self.show_pair()
        self.app.save_project()

    def view_limits(self, shape):
        """Zoom of the tile of the pair (its axes limits), kept when it is drawn again for an image of the same size
        (the views share the coordinates of the image); None otherwise."""
        ax = self.figure_pair.ax
        if ax is None or self.drawn[1] != shape:
            return None
        return ax.get_xlim(), ax.get_ylim()

    def restore_limits(self, limits):
        if limits is not None and self.figure_pair.ax is not None:
            self.figure_pair.ax.set_xlim(limits[0])
            self.figure_pair.ax.set_ylim(limits[1])

    def pair_tracks(self, path):
        """Particles tracked of the results (read_tracks), read once; None if they can't be read."""
        try:
            stamp = (path, os.path.getmtime(path))
            if self.tracks_cache[0] != stamp:
                self.tracks_cache = (None, None)
                self.tracks_cache = (stamp, read_tracks(path))
        except (OSError, ValueError, KeyError) as e:
            self.app.update_status(f"Particles tracked not read ({path}): {e}")
            return None
        return self.tracks_cache[1]

    def pair_displacements(self):
        """Displacements of the pair shown, None if they aren't computed: a dictionary with kind "grid" (a PIV, or a
        PTV whose particles tracked are unknown: X, Y the grid in the coordinates of the image, F the fields, k its
        map) or "particles" (a PTV: those of pair_particles), what they are (what) and the fields (fields, k: the
        colour limits are those of their maps)."""
        fields, k, source = self.field_shown()
        if fields is None or fields.u_px is None:
            return None
        what = {"live": "computing", "test": "pair computed alone", "results": "results"}[source]
        data = dict(fields=fields, k=k, what=what)
        if fields.particles is not None:
            particles, note = None, "the particles tracked are known when the PTV ends"
            if source == "test":
                if self.test_particles is not None and self.test_particles[0] == self.index:
                    particles = self.test_particles[1]
                note = "no particle tracked"
            elif source == "results":
                T = self.pair_tracks(fields.tracks) if fields.tracks else None
                particles = pair_particles(T, k) if T is not None else None
                note = (f"no particle tracked at frames {k + 1} and {k + 2}" if T is not None
                        else "tracks.npz not found")
            if particles is not None:
                return {**data, **particles, "kind": "particles"}
            data["note"] = f"PTV binned on its grid: {note}"
        X, Y = (c + 0.5 for c in self.grid_pixels(fields))
        return {**data, "kind": "grid", "X": X, "Y": Y, "dx": fields.u_px[k], "dy": fields.v_px[k]}

    def colour_limits(self, data):
        """Colour limits of the map shown: those of the maps of the results (LIMIT_MAPS of them, the same for all the
        pairs), or of the pair (computing, computed alone)."""
        key, f = self.view, data["fields"]
        if data["what"] != "results":
            return displacement_limits(key, displacement_component(key, data["dx"], data["dy"]))
        cache = f.__dict__.setdefault("pair_view_limits", {})
        if key not in cache:
            sub = slice(None, None, max(1, f.n // LIMIT_MAPS))
            cache[key] = displacement_limits(key, displacement_component(key, f.u_px[sub], f.v_px[sub]))
        return cache[key]

    def draw_displacements(self, data, shape, heading):
        """Map of the displacements of the pair (the view chosen) in the coordinates of the image, the zoom kept: on
        the grid (PIV; the vectors drawn on |d|) or at the particles tracked (PTV)."""
        key = self.view
        symbol, cmap = DISPLACEMENTS[key]
        panel = self.figure_pair
        fig = panel.fig
        limits = self.view_limits(shape)
        vmin, vmax = self.colour_limits(data)
        F = displacement_component(key, data["dx"], data["dy"])
        f = data["fields"]
        method = "PTV" if f.particles is not None else "PIV"
        with matplotlib.rc_context(MATH_FONT):
            fig.clf()
            self.overlay = []
            ax = fig.add_subplot()
            if data["kind"] == "grid":
                ax.set_facecolor(GRID_BACKGROUND)
                drawn = ax.pcolormesh(data["X"], data["Y"], F, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
                if key == "displacement":
                    step = max(1, -(-F.shape[1] // ARROWS))
                    s = (slice(None, None, step), slice(None, None, step))
                    # y upwards: the arrows ("uv" angles) point the same way on the image (y downwards)
                    ax.quiver(data["X"][s], data["Y"][s], data["dx"][s], data["dy"][s], color="white",
                              width=0.0022)
                count = f"{int(np.isfinite(F).sum())} valid vectors" if method == "PIV" else (
                    f"{int(np.isfinite(F).sum())} valid cells")
                detail = data.get("note") or ""
            else:
                ax.set_facecolor(PARTICLES_BACKGROUND)
                drawn = ax.scatter(data["x"] + 0.5, data["y"] + 0.5, c=F, s=3, cmap=cmap, vmin=vmin, vmax=vmax,
                                   linewidths=0)
                count = f"{F.size} particles tracked"
                detail = ("" if data["linked"] else
                          "particles of one frame of the pair, with their velocity (px per frame)")
            ax.set_aspect("equal")
            ax.set_xlim(0.5, shape[1] + 0.5)
            ax.set_ylim(shape[0] + 0.5, 0.5)
            ax.set_xlabel(AXIS_LABELS[0])
            ax.set_ylabel(AXIS_LABELS[1])
            title = f"{heading}, displacement ${symbol}$ ({method} {data['what']}: {count})"
            ax.set_title(title + (f"\n{detail}" if detail else ""), fontsize="medium", wrap=True)
            fig.colorbar(drawn, ax=ax, label=f"${symbol}$ (px)")
        panel.ax, panel.im, panel.cbar, panel.before, panel.divider = ax, None, None, None, []
        self.restore_limits(limits)
        self.drawn = (key, shape)
        panel.toolbar.update()
        panel.canvas.draw_idle()

    def toggle_colormap(self):
        panel = self.figure_pair
        if panel.im is not None:
            gray = panel.im.get_cmap().name == "gray"
            panel.set_colormap("hot" if gray else "gray", colorbar=gray)

    def field_shown(self):
        """Vector fields and map of the pair shown: the pair just computed by the PIV running, the test of the
        pair, or the results of the PIV."""
        if self.live is not None and self.live[0] == self.index:
            return self.live[1], 0, "live"
        if self.test_result is not None and self.test_result[0] == self.index:
            return self.test_result[1], 0, "test"
        if self.results is not None and self.index < self.results.n:
            return self.results, self.index, "results"
        return None, None, None

    def window_sizes(self):
        """Sizes of the windows of the preview (px), from the largest: halved from the size chosen, down to the
        smallest (where they collapse on one another)."""
        return sorted({max(self.preview_size >> n, PREVIEW_MIN) for n in range(PREVIEW_WINDOWS)}, reverse=True)

    def preview_largest(self, shape):
        """Largest window of the preview for an image of this shape: the smallest size doubled while it fits."""
        size = PREVIEW_MIN
        while size * 2 <= min(shape[:2]):
            size *= 2
        return size

    def preview_scrolled(self, event):
        """The mouse wheel over the image or the preview: the windows doubled (up) or halved (down)."""
        if self.pair_image is None or (event.canvas is self.figure_pair.canvas
                                       and event.inaxes is not self.figure_pair.ax):
            return
        size = self.preview_size * 2 if event.step > 0 else self.preview_size // 2
        size = int(np.clip(size, PREVIEW_MIN, self.preview_largest(self.pair_image.shape)))
        if size != self.preview_size:
            self.preview_size = size
            shape = self.pair_image.shape[:2]
            if self.ia_center is not None:
                self.ia_center = self.window_center(shape)
            self.draw_overlay(shape)
            self.update_particles()
            self.app.save_project()

    def window_center(self, shape):
        """Centre of the interrogation windows (x, y in pixels from the top-left corner of the image): the one
        chosen, or the centre of the image, on a pixel corner and with the largest window inside the image."""
        H, W = shape
        big = max(self.window_sizes() or [0])
        x, y = self.ia_center if self.ia_center is not None else (W / 2, H / 2)
        return (float(np.clip(round(x), min(big / 2, W / 2), max(W - big / 2, W / 2))),
                float(np.clip(round(y), min(big / 2, H / 2), max(H - big / 2, H / 2))))

    def windows_pressed(self, event):
        """A click on the image (without zoom nor pan) moves the interrogation windows there."""
        panel = self.figure_pair
        if (event.button == 1 and self.pair_image is not None and panel.ax is not None and event.inaxes is panel.ax
                and panel.toolbar.mode == "" and event.xdata is not None):
            self.dragging = True
            self.move_windows(event.xdata, event.ydata)

    def windows_dragged(self, event):
        if self.dragging and event.inaxes is self.figure_pair.ax and event.xdata is not None:
            self.move_windows(event.xdata, event.ydata)

    def windows_released(self, event):
        if self.dragging:
            self.dragging = False
            self.app.save_project()

    def move_windows(self, x, y):
        """Interrogation windows centred on the point (x, y) of the image (its axes: the pixel centres at the
        integer coordinates from 1; those of the maps of the displacements too), and their particles counted
        there."""
        shape = self.pair_image.shape[:2]
        self.ia_center = (x - 0.5, y - 0.5)
        self.ia_center = self.window_center(shape)
        self.draw_overlay(shape)
        self.update_particles()

    def particles_changed(self):
        self.panel.save()
        self.update_particles()
        self.app.save_project()

    def update_particles(self):
        """Particles of the interrogation windows on the image shown, at their position: the largest window
        magnified with the particles found, and the number of particles of each window (enough: about 10)."""
        panel = self.panel
        fig = self.figure_windows.fig
        fig.clf()
        sizes = self.window_sizes()
        message = None
        if self.pair_image is None:
            message = "No image"
        else:
            try:
                snr, diameter = panel.read_particles()
            except ValueError as e:
                message = str(e)
        if message is not None:
            fig.text(0.5, 0.5, message, ha="center", va="center", color="0.45", wrap=True)
            self.figure_windows.canvas.draw_idle()
            return
        image = self.pair_image
        center = self.window_center(image.shape[:2])
        found = count_particles(image, center, sizes, snr, diameter)
        self.draw_particles(image, found, center)

    def draw_particles(self, image, found, center):
        """The largest window magnified, the smaller ones inside it, and the particles found, with the number of
        particles of each window at the top and the centre, background and noise at the bottom."""
        fig = self.figure_windows.fig
        x0, y0, big = found["box"]
        window = np.asarray(image[y0 : y0 + big, x0 : x0 + big], float)
        if window.ndim == 3:
            window = window.mean(axis=2)
        low, high = np.percentile(window, [1, 99.8]) if window.size else (0, 1)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(window, cmap="gray", vmin=low, vmax=max(high, low + 1), interpolation="nearest",
                  extent=(x0, x0 + big, y0 + big, y0))
        for w in found["windows"][1:]:
            ax.add_patch(Rectangle((w["x0"], w["y0"]), w["size"], w["size"], fill=False, edgecolor=OVERLAY_COLOR,
                                   linewidth=1))
        ax.plot(found["x"], found["y"], "o", markerfacecolor="none", markeredgecolor="#00e5ff", markersize=5,
                markeredgewidth=1.8)
        ax.set_xlim(x0, x0 + big)
        ax.set_ylim(y0 + big, y0)
        ax.set_axis_off()
        box = dict(boxstyle="round,pad=0.35", facecolor="black", edgecolor="none", alpha=0.6)
        # The rule of thumb of the PIV (about 10 particles per window) only for a PIV software
        rule = self.panel.engine().method == "PIV"
        for n, w in enumerate(found["windows"]):
            enough = w["count"] >= MIN_PARTICLES or not rule
            ax.annotate(f"{w['size']} px window: {w['count']} particle{'' if w['count'] == 1 else 's'}"
                        + (("  ✓" if enough else f"  ✗ fewer than {MIN_PARTICLES}") if rule else ""), (0.02, 0.98),
                        xycoords="axes fraction", xytext=(0, -17 * n), textcoords="offset points", ha="left",
                        va="top", fontsize=8, color="white" if enough else "#ff6b6b", bbox=box)
        ax.text(0.02, 0.02, f"Centre : ({center[0]:.0f}, {center[1]:.0f}) px\n"
                f"Background : {found['background']:.3g} Grey Lvl.\nNoise : {found['noise']:.3g} Grey Lvl.",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="white", bbox=box,
                linespacing=1.3)
        self.figure_windows.canvas.draw_idle()

    def draw_overlay(self, shape):
        """Windows of the preview at their position (the centre of the image unless they are moved), and on the image
        the vectors of the pair (magnified) if they are computed (the map |d| has its own)."""
        panel = self.figure_pair
        for artist in self.overlay:
            try:
                artist.remove()
            except (NotImplementedError, ValueError):  # Its figure was drawn again (cleared) meanwhile
                pass
        self.overlay = []
        ax = panel.ax
        if ax is None:
            return
        x, y = self.window_center(shape)
        cx, cy = x + 0.5, y + 0.5
        windows = self.window_sizes()
        for n, size in enumerate(windows):
            alpha = 0.45 + 0.55 * (n + 1) / len(windows)
            self.overlay.append(ax.add_patch(Rectangle((cx - size / 2, cy - size / 2), size, size, fill=False,
                                                       edgecolor=OVERLAY_COLOR, linewidth=1.2, alpha=alpha)))
            self.overlay.append(ax.text(cx + size / 2, cy - size / 2, f" {size}", color=OVERLAY_COLOR, fontsize=7,
                                        va="bottom", ha="left", alpha=alpha, clip_on=True))
        fields, k, _ = self.field_shown()
        if fields is not None and fields.scale and self.drawn[0] == "image":
            s = (slice(None, None, 3), slice(None, None, 3))
            xi, yi = (c[s] + 0.5 for c in self.grid_pixels(fields))
            du, dv = fields.u_px[k][s], -fields.v_px[k][s]
            self.overlay.append(ax.quiver(xi, yi, du, dv, color=OVERLAY_COLOR, angles="xy", scale_units="xy",
                                          scale=1 / 3, width=0.0025))
            self.overlay.append(ax.text(0.01, 0.01, "vectors × 3", transform=ax.transAxes, color=OVERLAY_COLOR,
                                        fontsize=8, va="bottom"))
        panel.canvas.draw_idle()

    @staticmethod
    def grid_pixels(fields):
        """Positions of the vectors in the image (px from its top-left corner): displacements.grid_pixels."""
        return grid_pixels(fields)

    def show_monitor(self, current_only=False, upto=None):
        """Quality of the pairs computed, the pair shown marked: valid vectors and median correlation peak ratio
        (PIV), or valid cells and particles tracked, with the lengths of the tracks of the results (PTV). upto: only
        the pairs up to this one (the pair just computed, shown while the PIV runs: the mark is on the last
        point)."""
        panel = self.figure_monitor
        data = self.monitor_data()
        if data is None:
            panel.message("Quality of the pairs computed: valid vectors and correlation peak ratio (PIV), valid "
                          "cells, particles tracked and lengths of the tracks (PTV)")
            self.monitor_curves = None
            return
        if upto is not None:
            data = data.copy()
            data[:, upto + 1:] = np.nan
        line = self.monitor_line
        if current_only and line is not None and line.axes is not None:
            line.set_xdata([self.index + 1])
            panel.canvas.draw_idle()
            return
        ptv = self.monitor_ptv
        n = data.shape[1]
        done = np.isfinite(data[0]).sum()
        valid = np.nanmean(data[0]) if done else float("nan")
        if ptv:
            tracked = np.nanmedian(data[1]) if np.isfinite(data[1]).any() else float("nan")
            title = (f"Quality of the pairs: {done} of {n} computed, {valid:.1f} % valid cells on average, "
                     f"{tracked:.0f} particles tracked per pair (median)")
        else:
            title = f"Quality of the pairs: {done} of {n} computed, {valid:.1f} % valid vectors on average"
        lengths = self.results_track_lengths() if ptv else None
        layout = (ptv, n, lengths is not None and self.track_lengths[0])
        fig = panel.fig
        curves = self.monitor_curves
        if curves is not None and curves[0] == layout and curves[1][0].axes in fig.axes:
            # The same graph: the curves are updated (faster than drawing the figure again)
            for curve, values in zip(curves[1], data):
                curve.set_ydata(values)
                curve.axes.relim()
                curve.axes.autoscale_view(scalex=False)
            line.set_xdata([self.index + 1])
            line.axes.set_title(title, fontsize="medium")
        else:
            with matplotlib.rc_context(MATH_FONT):
                fig.clf()
                if ptv:
                    ax, hist = fig.subplots(1, 2, width_ratios=(3, 1))
                else:
                    ax, hist = fig.add_subplot(), None
                k = np.arange(1, n + 1)
                first, = ax.plot(k, data[0], color="#2a78d6", lw=1)
                ax.set_ylabel("Valid cells (%)" if ptv else "Valid vectors (%)", color="#2a78d6")
                ax.set_xlabel("Pair")
                ax.set_xlim(0.5, n + 0.5)
                twin = ax.twinx()
                second, = twin.plot(k, data[1], color="#eb6834", lw=1)
                twin.set_ylabel("Particles tracked" if ptv else "Median peak ratio $S/N$", color="#eb6834")
                self.monitor_line = ax.axvline(self.index + 1, color="0.4", lw=1, ls="--")
                ax.set_title(title, fontsize="medium")
                if hist is not None:
                    self.draw_track_lengths(hist, lengths)
                self.monitor_curves = (layout, (first, second))
        panel.canvas.draw_idle()
        self.monitor_drawn = time.monotonic()

    def draw_track_lengths(self, ax, lengths):
        """Histogram of the lengths of the tracks of the results (frames where each particle is tracked), the median
        marked; a message if they are unknown."""
        if lengths is None or not lengths.size:
            text = ("Lengths of the tracks:\nwith the results of the PTV" if self.results is None
                    or self.results.particles is None else "Lengths of the tracks:\nunknown (no tracks.npz)")
            ax.text(0.5, 0.5, text, ha="center", va="center", color="0.45", transform=ax.transAxes)
            ax.set_axis_off()
            return
        top = int(max(np.percentile(lengths, 99.5), 3))
        ax.hist(lengths, bins=np.arange(0.5, top + 1.5), color="#2a78d6", edgecolor="white", linewidth=0.3)
        ax.set_yscale("log")
        median = float(np.median(lengths))
        ax.axvline(median, color="#eb6834", lw=1.2)
        ax.set_xlim(0.5, top + 0.5)
        ax.set_xlabel("Track length (frames)")
        ax.set_ylabel("Tracks")
        ax.set_title(f"{lengths.size} tracks, median {median:.0f} frames", fontsize="medium")

    def results_track_lengths(self):
        """Lengths of the tracks of the results of a PTV (frames where each particle is tracked: the identities of
        its tracks.npz, read once), None if unknown."""
        f = self.results
        path = f.tracks if f is not None and f.particles is not None else ""
        if not path or not os.path.isfile(path):
            return None
        stamp = (path, os.path.getmtime(path))
        if self.track_lengths[0] != stamp:
            try:
                with np.load(path) as z:
                    ids = z["id"]
            except (OSError, ValueError, KeyError):
                return None
            if ids.size and ids.max() - ids.min() < 4 * ids.size:
                counts = np.bincount(ids - ids.min())
                lengths = counts[counts > 0]
            else:
                lengths = np.unique(ids, return_counts=True)[1]
            self.track_lengths = (stamp, lengths)
        return self.track_lengths[1]

    def monitor_data(self):
        """Quality of the pairs computed (2 x pairs): valid vectors (%) and median peak ratio (PIV), or valid cells
        (%) and particles tracked (PTV: monitor_ptv), of the PIV/PTV running or of its results. None if there are
        none."""
        if self.monitor is None and self.results is not None:
            f = self.results
            flow = f.status == 0
            self.monitor_ptv = f.particles is not None
            with np.errstate(all="ignore"):
                second = (f.particles.reshape(f.n, -1).sum(1).astype(float) if self.monitor_ptv
                          else nanmedian(f.peak_ratio[:, flow], 1) if f.peak_ratio is not None
                          else np.full(f.n, np.nan))
                self.monitor = np.vstack([100 * f.piv_valid[:, flow].mean(1), second])
        return self.monitor

    def pair_quality(self, fields, k=0):
        """Valid vectors (%) of the map k of vector fields at the grid points with flow (those of the results if they
        have the same grid: a pair computed alone can't tell where the particles are missing; or else all but the
        masked ones), and its median peak ratio (PIV) or its particles tracked (PTV)."""
        r = self.results
        flow = r.status == 0 if r is not None and r.shape == fields.shape else fields.status != 2
        if not flow.any():
            return np.nan, np.nan
        valid = 100 * float(fields.piv_valid[k][flow].mean())
        if fields.particles is not None:
            return valid, float(fields.particles[k].sum())
        peaks = np.asarray(fields.peak_ratio[k][flow] if fields.peak_ratio is not None else [], float)
        peaks = peaks[np.isfinite(peaks)]
        return valid, float(np.median(peaks)) if peaks.size else np.nan

    def monitor_pair(self, index, fields):
        """A pair computed again alone: its quality replaces the one of the graph of the quality of the pairs (the
        others are kept, unless they are those of the other method: PIV or PTV)."""
        n = len(self.pair_list())
        ptv = fields.particles is not None
        data = self.monitor_data()
        if data is None or data.shape[1] != n or self.monitor_ptv != ptv:
            data = np.full((2, n), np.nan)
        self.monitor_ptv = ptv
        if index < n:
            data[:, index] = self.pair_quality(fields)
        self.monitor = data
        self.show_monitor()

    def settings_changed(self):
        """A setting was entered: the pairs and the overlay follow it."""
        if self.n_images() and self.app.current_page() == self.key:
            self.show_pair()
        self.app.save_project()

    # ------------------------------------------------------------------ PaIRS interpreter

    def choose_python(self):
        """Choose the Python interpreter running PaIRS-UniNa (python.exe of an environment where it is
        installed)."""
        app = self.app
        filename = filedialog.askopenfilename(parent=app.parent(), title="Python interpreter with PaIRS-UniNa...",
                                              filetypes=[("Python", "python*.exe python python3*"), ("All files", "*.*")])
        if not filename:
            return
        if not piv.has_pairs(filename) and app.questdlg(
                f"PaIRS-UniNa was not found in the environment of {filename}. Use this interpreter anyway?",
                "PaIRS-UniNa", ["Yes", "No"], "No") != "Yes":
            return
        options = getpref("piv_engine_options", {})
        options["pairs"] = {"python": os.path.normpath(filename)}
        setpref("piv_engine_options", options)
        self.panel.update_engine_label()
        app.update_status(f"PaIRS-UniNa run by {filename}")

    def check_engine(self, engine):
        """The engine can run, or the user is told why (and may choose the interpreter of PaIRS-UniNa)."""
        ok, reason = engine.available()
        if not ok:
            if hasattr(engine, "python"):
                if self.app.questdlg(reason, engine.name, ["Choose Python...", "Cancel"], "Choose Python...") \
                        == "Choose Python...":
                    self.choose_python()
            else:
                self.app.msgbox(reason, engine.name, "warn")
        return ok

    def read_regression(self):
        """Regression entered (None: none), False (the user is told why) if it is invalid or SPICY can't run."""
        try:
            regression = self.panel.read_regression()
        except ValueError as e:
            self.app.msgbox(str(e), "PIV/PTV settings", "warn")
            return False
        ok, reason = piv.spicy_available() if regression is not None else (True, "")
        if not ok:
            self.app.msgbox(reason, "SPICY", "warn")
            return False
        return regression

    def read_settings(self):
        """Engine and settings entered, None (the user is told why) if invalid."""
        try:
            return self.panel.read_engine()
        except ValueError as e:
            self.app.msgbox(str(e), "PIV/PTV settings", "warn")
            return None

    # ------------------------------------------------------------------ PIV

    def busy(self):
        return self.run is not None or self.test is not None

    def run_or_stop(self):
        if self.run is not None:
            self.stop()
        else:
            self.start_run()

    def start_run(self):
        """Run the PIV of the images pre-processed with the recipe (exported first if needed), in the background."""
        app, pre = self.app, self.pre
        if self.test is not None or pre.no_images_selected():
            return
        values = self.read_settings()
        if values is None:
            return
        engine, settings = values
        if not self.check_engine(engine):
            return
        regression = self.read_regression()
        if regression is False:
            return
        pairs = self.pair_list()
        if not pairs:
            app.msgbox(f"No pair of images with {self.n_images()} images.", "PIV/PTV", "warn")
            return
        self.update_images_label()
        kind, where = self.source()
        n_modes = pre.n_modes_removed()
        steps = " → ".join(pre.recipe_steps(n_modes, detailed=True))
        manifest = self.chosen_images()[1] if kind == "chosen" else None
        if kind == "chosen":
            images_text = f"those of {self.panel.images.get()}, used as they are"
        elif kind == "original":
            images_text = "the original images (the recipe of the page Pre-processing is empty)"
        elif kind == "preprocessed":
            images_text = f"pre-processed with the recipe of the page Pre-processing, in {os.path.dirname(where[0])}"
        else:
            images_text = f"pre-processed with the recipe of the page Pre-processing and exported to {where} first"
        output = self.output_folder()
        rate = (PTV_SECONDS_PER_FRAME * (1 + settings.get("vid_loop", 0)) if engine.sequence
                else self.last_rate or SECONDS_PER_PAIR)
        message = (f"Run the PIV/PTV of {len(pairs)} pairs (about {max(1, round(len(pairs) * rate / 60))} min)?\n\n"
                   f"Images: {images_text}\n\nPIV/PTV: {engine.name}, {engine.describe(settings)}\n\nResults (in "
                   f"pixels): {output}"
                   + (f"\n\nThen {piv.describe_regression(regression)}, a few seconds per map"
                      if regression is not None else "")
                   + ("\n\nThe pairs computed before are kept if the images and the settings are the same."
                      if os.path.isdir(os.path.join(output, "_progress")) else ""))
        if app.questdlg(message, "Run the PIV/PTV", ["Run", "Cancel"], "Run") != "Run":
            return
        self.panel.save()
        if kind == "chosen":
            images = where
            info = {"measurement_folder": pre.path_in,
                    "preprocessing": (manifest.get("steps") or "none") if manifest else "unknown (images chosen)",
                    "preprocessing_recipe": manifest.get("recipe") if manifest else None}
        else:
            try:
                images = pre.piv_images(n_modes)
            except (OSError, ValueError) as e:
                app.status_error(f"The images could not be pre-processed: {e}")
                return
            if images is None:
                return
            self.source_cache = None
            info = {"measurement_folder": pre.path_in, "preprocessing": steps or "none (original images)",
                    "preprocessing_recipe": pre.recipe_record(n_modes)}
        self.update_images_label()
        try:
            run = piv.PIVRun(images, None, engine, settings, output, masked=self.masked(), info=info,
                             project=app.project_name(), regression=regression)
            run.start()
        except (OSError, ValueError, RuntimeError) as e:
            app.status_error(f"The PIV/PTV could not start: {e}")
            return
        self.run = run
        self.results = None
        self.test_result = self.live = self.live_message = None
        self.monitor = np.full((2, len(run.pairs)), np.nan)
        self.monitor_ptv = engine.method == "PTV"
        self.panel.set_running(True)
        self.panel.progress["value"] = 0
        self.panel.progress_label["text"] = f"Starting {engine.name}..."
        self.update_results_label()
        self.show_pair()
        self.show_monitor()
        app.update_status(f"PIV/PTV of {len(run.pairs)} pairs started ({engine.name}): results in {output}")
        app.save_project()
        self.schedule(self.poll)

    def stop(self):
        if self.run is not None:
            self.run.process.stop()
            self.panel.run_button.configure(state=tk.DISABLED)
            self.panel.progress_label["text"] = "Stopping after the pair being computed..."

    def poll(self):
        """Messages of the PIV process: progress, end or error."""
        run = self.run
        if run is None:
            return
        try:
            for message in run.process.poll():
                if self.handle(run, message):
                    return
        except Exception as e:  # The results could not be saved: the PIV ends with the error
            self.finish()
            self.app.status_error(f"PIV/PTV: {type(e).__name__}: {e}")
            return
        if self.live_message is not None:
            self.show_live(run, self.live_message)
            self.live_message = None
        self.schedule(self.poll)

    def following(self):
        """The PIV runs and the page shows the pairs it computes (the user isn't browsing other pairs)."""
        return (self.run is not None and self.app.current_page() == self.key
                and time.monotonic() - self.browsed >= BROWSE_PAUSE)

    def show_live(self, run, message):
        """The pair just computed, with its image and its vectors, and the quality of the pairs up to it, unless the
        user is browsing other pairs."""
        k = message["pair"]
        drawn, cost = self.live_drawn
        if (not self.following() or k >= len(self.pair_list())
                or time.monotonic() - drawn < cost / LIVE_LOAD):
            return
        start = time.monotonic()
        try:
            fields = load_vectors(run.preview(message), pixels=True)
        except (OSError, ValueError, KeyError):  # Replaced by the next pair meanwhile: shown with that one
            return
        first = self.live is None
        self.live = (k, fields)
        self.show_pair(k)
        self.show_monitor(upto=k)
        # Drawn now, to know how long it takes (the first time, longer: the figures are made)
        self.update_idletasks()
        self.live_drawn = (time.monotonic(), 0.0 if first else time.monotonic() - start)

    def schedule(self, poll):
        """Poll the process of the PIV (or of the test) again in 300 ms."""
        self.timer = self.after(300, poll)

    def handle(self, run, message):
        """Handle a message of the PIV process. Returns True when the PIV has ended."""
        app, panel, kind = self.app, self.panel, message["type"]
        if kind == "progress":
            done, total = message["done"], message["total"]
            if message.get("phase") == "background":
                panel.progress_label["text"] = f"Mean image: {done} of {total} images"
                return False
            if message.get("phase") == "training":
                # TracTrac: its motion model trained over passes back and forth before the last one
                panel.progress["value"] = done / max(total, 1)
                rate = message.get("rate")
                text = f"TracTrac: training the motion model, pass {message.get('pass')} of {message.get('passes')}"
                if rate:
                    left = (total - done) * rate
                    ready = time.strftime('%H:%M', time.localtime(time.time() + left))
                    text += f" · ready at {ready} ({left / 60:.0f} min)"
                panel.progress_label["text"] = text
                return False
            if message.get("phase") == "spicy":
                panel.progress["value"] = done / max(total, 1)
                rate = message.get("rate")
                if rate:
                    left = (total - done) * rate
                    ready = time.strftime('%H:%M', time.localtime(time.time() + left))
                    panel.progress_label["text"] = (f"SPICY regression: map {done} of {total} · {rate:.1f} s per map "
                                                    f"· ready at {ready} ({left / 60:.0f} min)")
                else:
                    panel.progress_label["text"] = f"SPICY regression: maps fitted before: {done} of {total}"
                return False
            panel.progress["value"] = done / max(total, 1)
            rate = message.get("rate")
            if rate:
                self.last_rate = rate
                left = (total - done) * rate
                panel.progress_label["text"] = (f"Pair {done} of {total} · {rate:.2f} s per pair · ready at "
                                                f"{time.strftime('%H:%M', time.localtime(time.time() + left))} "
                                                f"({left / 60:.0f} min)")
            else:
                panel.progress_label["text"] = f"Pairs computed before: {done} of {total}"
            k = message.get("pair")
            # With the valid vectors: the median peak ratio (PIV) or the particles tracked (PTV)
            second = "particles" if self.monitor_ptv else "peak_ratio"
            if k is not None and message.get("valid") is not None:
                self.monitor[:, k] = message["valid"], message.get(second) or np.nan
            if message.get("resumed") and isinstance(message.get("valid"), list):
                # The quality of the pairs computed before the PIV was stopped
                first, valids = message.get("first", 0), message["valid"]
                seconds = list(message.get(second) or [])
                seconds += [None] * (len(valids) - len(seconds))
                for i, (valid, value) in enumerate(zip(valids, seconds)):
                    if first + i < self.monitor.shape[1]:
                        self.monitor[:, first + i] = (np.nan if valid is None else valid,
                                                      np.nan if value is None else value)
            # The quality is drawn with the pair shown while the PIV is followed (every 5 s if no pair is shown),
            # every 2 s otherwise
            if time.monotonic() - self.monitor_drawn > (5 if self.following() else 2):
                self.show_monitor()
        elif kind == "field":
            # Only the last one of the messages received is shown
            self.live_message = message
        elif kind == "log":
            app.update_status(f"PIV/PTV: {message['text']}")
        elif kind == "done":
            panel.progress_label["text"] = "Saving the vector fields..."
            panel.update_idletasks()
            path = run.save(message)
            self.finish()
            f = load_vectors(path, pixels=True)
            valid = valid_percent(f)
            seconds = run_time(f.settings)
            took = f" in {format_duration(seconds)}" if seconds is not None else ""
            resumed = f.settings.get("resumed_pairs")
            if took and resumed:
                took += f" (with {resumed} pairs computed before the PIV/PTV was stopped)"
            panel.progress["value"] = 1
            panel.progress_label["text"] = f"Done: {f.n} pairs, {valid:.1f} % valid vectors{took}"
            # Shown with the images, unless other images (or another results folder) were chosen meanwhile
            current = bool(self.pre.Check_imsel and same_path(run.info["measurement_folder"], self.pre.path_in)
                           and same_path(os.path.dirname(path), self.output_folder()))
            if current:
                self.results, self.results_checked, self.monitor = f, True, None
                self.update_results_label()
                self.show_pair()
                self.show_monitor()
            post = app.page("post")
            if post is not None:
                post.piv_done(path, f)
            failed = message.get("failed", 0)
            text = (f"PIV/PTV done: {f.n} pairs, {valid:.1f} % valid vectors{took}"
                    + (f", {failed} pair(s) failed" if failed else "") + ". Results: ")
            app.update_status(text + path)
            if current:
                app.log_step("piv", text + app.project_path(path))
            app.msgbox(f"The PIV/PTV is done: {f.n} pairs, {valid:.1f} % valid vectors{took}.\n\nThe vector fields are "
                       f"saved in {os.path.dirname(path)}"
                       + (" and shown in the page Post-processing." if post else "."), "PIV/PTV")
            return True
        elif kind == "stopped":
            self.finish()
            panel.progress_label["text"] = ("Stopped: Run continues from the last chunk of 100 pairs saved "
                                            "(same images and settings)")
            self.update_results_label()
            app.update_status("PIV/PTV stopped: the chunks of 100 pairs computed are kept")
            return True
        elif kind == "error":
            self.finish()
            panel.progress_label["text"] = "Error: see the status"
            app.status_error(f"PIV/PTV: {message['text']}")
            return True
        return False

    def finish(self):
        self.run = None
        self.live = self.live_message = None
        self.panel.set_running(False)

    def on_exit(self):
        """The window is closed: the PIV running is stopped if the user confirms (its progress is kept). Returns
        False if the user keeps it running."""
        if self.run is not None:
            if self.app.questdlg("The PIV/PTV is running. Stop it and quit? The chunks of 100 pairs computed are kept: "
                                 "Run continues from them.", "Quit", ["Stop and quit", "Cancel"],
                                 "Cancel") != "Stop and quit":
                return False
            process = self.run.process
            process.stop()
            process.wait(10)
            process.kill()
        if self.test is not None:
            self.test[0].process.kill()
            self.test[0].process.wait(10)
            shutil.rmtree(self.test[1], ignore_errors=True)
        if self.timer is not None:
            self.after_cancel(self.timer)
        return True

    # ------------------------------------------------------------------ PIV of the pair shown

    def run_pair(self):
        """Compute the pair shown alone with the settings entered, to check them (in a temporary folder)."""
        app, pre = self.app, self.pre
        if self.busy() or pre.no_images_selected():
            return
        values = self.read_settings()
        if values is None:
            return
        engine, settings = values
        if not self.check_engine(engine):
            return
        regression = self.read_regression()
        if regression is False:
            return
        self.update_images_label()
        pairs = self.pair_list()
        if not pairs:
            return
        n_modes = pre.n_modes_removed()
        kind, where = self.source()
        if kind == "export" and self.modes_missing():
            if app.questdlg(f"The background removal of the recipe needs {n_modes} POD modes: evaluate them now?",
                            "PIV/PTV of the pair", ["Evaluate the modes", "Cancel"],
                            "Evaluate the modes") != "Evaluate the modes" or pre.evaluate_modes(n_modes) is None:
                return
            self.source_cache = None
        a, b = pairs[self.index]
        frames = [a, b]
        if engine.sequence:
            # A tracking: the frames of the sequence before and after the pair too, for its motion model (the pair
            # kept: test_map)
            try:
                sequence = piv.sequence_of(pairs)
            except ValueError as e:
                app.msgbox(str(e), "PIV/PTV settings", "warn")
                return
            frames = sequence[max(0, self.index - PTV_TEST_FRAMES) : self.index + PTV_TEST_FRAMES + 2]
        folder = tempfile.mkdtemp(prefix="velocihub_pair_")
        if kind != "export":
            images = [where[k] for k in frames]
        else:
            # The images processed with the recipe, as they would be exported
            images = [os.path.join(folder, f"image_{k + 1}.tif") for k in frames]
            for k, path in zip(frames, images):
                imwrite(self.input_image(k), path)
        self.panel.save()
        try:
            run = piv.PIVRun(images, None, engine, settings, os.path.join(folder, "PIV"), masked=self.masked(),
                             pairs=[(i, i + 1) for i in range(len(frames) - 1)], regression=regression)
            run.test_map = frames.index(a)
            run.start()
        except (OSError, ValueError, RuntimeError) as e:
            shutil.rmtree(folder, ignore_errors=True)
            app.status_error(f"The PIV/PTV of the pair could not start: {e}")
            return
        self.test = (run, folder, self.index)
        self.panel.set_running(False, testing=True)
        self.panel.progress_label["text"] = f"PIV/PTV of pair {self.index + 1} (images {a + 1}, {b + 1})..."
        self.schedule(self.poll_test)

    def poll_test(self):
        if self.test is None:
            return
        run, folder, index = self.test
        for message in run.process.poll():
            kind = message["type"]
            if kind in ("done", "error", "stopped"):
                self.test = None
                self.panel.set_running(False)
                try:
                    if kind == "done":
                        fields = single_map(load_vectors(run.save(), pixels=True), run.test_map)
                        self.test_result = (index, fields)
                        # The particles tracked over the pair (PTV), read before the temporary folder is deleted
                        particles = None
                        if fields.tracks:
                            try:
                                particles = pair_particles(read_tracks(fields.tracks), run.test_map)
                            except (OSError, ValueError, KeyError):
                                pass
                        self.test_particles = (index, particles)
                        valid, peak = self.pair_quality(fields)
                        text = f"Pair {index + 1}: {valid:.1f} % valid vectors, " + (
                            f"median peak ratio {peak:.2f}" if fields.particles is None else
                            f"{np.mean(fields.particles[0][fields.status != 2]):.2f} particles per cell")
                        self.panel.progress_label["text"] = text
                        self.app.update_status(text)
                        # Its quality replaces the one of the PIV in the graph
                        self.monitor_pair(index, fields)
                        if self.index == index:
                            self.show_pair()
                    elif kind == "error":
                        self.panel.progress_label["text"] = "PIV/PTV of the pair failed: see the status"
                        self.app.status_error(f"PIV/PTV of the pair: {message['text']}")
                finally:
                    shutil.rmtree(folder, ignore_errors=True)
                return
        self.schedule(self.poll_test)
