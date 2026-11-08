"""Interface of the post-processing of the archive (velocihub/archiving/post_archive.py): the options of the header of the script
(the archive, the fields of view, the recipe, the reference velocities, the exports and the stitching), the setups
to process, and the run, in a separate process (stopped at any time), its messages shown below.

    velocihub-post-archive-gui

The options are kept between sessions (the preferences of VelociHub); "Defaults of the script" goes back to those of the
header of velocihub/archiving/post_archive.py.
"""
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import zipfile
from dataclasses import fields as dataclass_fields
from tkinter import messagebox, ttk
from types import SimpleNamespace

import numpy as np

import velocihub as vh
from velocihub.archiving import post_archive as r
from velocihub.fields import num2str
from velocihub.gui.postprocessing_page import (CORE_MODE_HELP, L_REF_HELP, REFERENCE_SETTINGS, ROLLING_HELP, SETTINGS as
                                               PAGE_SETTINGS, ExportDialog)
from velocihub.settings import export_choice, export_maps
from velocihub.gui.widgets import HelpDot, PathBox
from velocihub.prefs import getpref, setpref
from velocihub.jet import CORE_MODES
from velocihub.postprocessing import FIGURE_TITLES, INSTANTANEOUS, MAP_FIGURES

# The folder of the package velocihub (where python -m finds it when it isn't installed)
PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The options in the preferences of VelociHub
PREF = "archive_gui"
# The figures that can be stitched
STATISTICS_MAPS = [key for key in MAP_FIGURES if key not in INSTANTANEOUS]
INSTANTANEOUS_MAPS = [key for key in MAP_FIGURES if key in INSTANTANEOUS]
JET_PLOTS = ["core_profiles", "core_centreline"]
# A warning of Python in the messages of the run ("file.py:12: RuntimeWarning: ..."), followed by its source line
WARNING = re.compile(r"^\S.*:\d+: \w*Warning: ")
# The recipe: labels and help of the settings (those of the post-processing page where it has them)
LABELS = {"use_u_ref": "Velocities over u_ref", "u_ref": "Ref. velocity u_ref (m/s)",
          "use_t_flow": "Times in flow times", "t_flow": "Flow time t_flow (s)", "l_ref": "Ref. length L_ref (mm)",
          "core_u_ref": "Jet theory u_ref (m/s)", "rolling": "Rolling statistics",
          "rolling_maps": "Rolling interval (maps)", "core_mode": "Potential core detection"}
HELP = {key: text for _, key, _, text in PAGE_SETTINGS}
HELP.update({key: text for _, key, _, text in REFERENCE_SETTINGS})
HELP.update(core_mode=CORE_MODE_HELP, use_u_ref=HELP["u_ref"], use_t_flow=HELP["t_flow"], l_ref=L_REF_HELP,
            rolling=ROLLING_HELP,
            rolling_maps="Maps of the rolling statistics: those of each map are those of this many maps up to it "
                         "(all those before it while there are fewer).",
            u_ref=HELP["u_ref"] + "\n\nIn the archive, the reference velocity of each setup (Reference velocities "
                                  "below) replaces it.")
# Help of the options of the script
OPTION_HELP = {
    "archive": ("The archive folder: a folder per field of view (FOV_1, FOV_2...), each with a folder per "
                "measurement setup holding its PIV results (PIV/vectors.npz)."),
    "export_root": ("Empty: the exports of each measurement go to its subfolder PostProcessing (as the interface "
                    "exports them); else to [folder]/[FOV]/[setup]."),
    "stitched": "Folder of the stitched figures, a subfolder per setup. Empty: Stitched in the archive.",
    "setups": ("The setups processed, in all their fields of view; the stitching needs a setup measured in "
               "several fields of view."),
    "fovs": ("Width of each field of view, horizontal offset of its left edge along the jet and vertical offset "
             "of the middle of its height (mm). The stitching keeps the common height; a field of view without "
             "width is skipped."),
    "frame_rate_hz": "Frame rate of the camera (Hz), which sets the time between the images of a pair.",
    "reference_fov": ("The reference velocity of each setup is the maximum of the mean u of its measurement in "
                      "this field of view (the jet exit)."),
    "match_velocity": ("Matches the reference velocities of the fields of view of a setup in their overlaps, "
                       "since their runs had different flow rates. The factor between their mean u is fitted on "
                       "the cells of the jet in the overlap (at least the jet fraction of its highest mean u), "
                       "the edge columns left out; the highest maximum, brought to the level of the reference "
                       "field of view, is its reference velocity. Off: that of the reference field of view for all."),
    "index": ("Map of the instantaneous figures when no range of maps is exported: 0 the first, negative from the "
              "end (-1 the last)."),
    "export": ("The figures exported for each measurement, the maps of the instantaneous ones (the map above, or "
               "a range with a folder per figure) and their videos (over the range, else all the maps). With "
               "rolling statistics (Recipe), every figure has one per map and is exported like the instantaneous "
               "ones."),
    "stitched_lists": ("The figures stitched for each setup: the fields of view side by side, each with its own x "
                       "axis and references, with one colour scale. The instantaneous maps are those of the map "
                       "above or of each map of the range of the exports; their videos use that range (else all "
                       "the maps) with the same scales in all the frames, about 0.7 s per frame. With rolling "
                       "statistics (Recipe), the maps of the statistics are per map too, and the jet theory plots "
                       "use the map above."),
}


def script_defaults():
    """The options of the header of velocihub/archiving/post_archive.py."""
    choice = export_choice(dict(figures=r.EXPORT_FIGURES, videos=list(r.EXPORT_VIDEOS), fps=r.EXPORT_FPS,
                                format=r.EXPORT_FORMAT))
    if isinstance(r.EXPORT_MAPS, range) and r.EXPORT_MAPS.start >= 0:
        choice.update(range=True, first=r.EXPORT_MAPS.start + 1, last=r.EXPORT_MAPS.stop, step=r.EXPORT_MAPS.step)
    return dict(
        archive=r.ARCHIVE, export_root="", stitched="", setups=None,
        fov_width=dict(r.FOV_WIDTH_MM), fov_offset=dict(r.FOV_X_OFFSET_MM),
        fov_offset_y=dict(r.FOV_Y_OFFSET_MM), frame_rate_hz=r.FRAME_RATE_HZ,
        settings={f.name: getattr(r.SETTINGS, f.name) for f in dataclass_fields(vh.PostSettings)},
        reference_fov=r.REFERENCE_FOV, match_velocity=r.MATCH_VELOCITY, match_jet=r.MATCH_JET,
        match_edge=r.MATCH_EDGE, index=r.INDEX, export=choice,
        stitched_maps=list(r.STITCHED),
        stitched_instantaneous=list(r.STITCHED_INSTANTANEOUS), stitched_videos=list(r.STITCHED_VIDEOS),
        stitched_jet=list(r.STITCHED_JET))


def saved_options():
    """The options of the last session over the defaults of the script."""
    options = script_defaults()
    saved = getpref(PREF)
    if isinstance(saved, dict):
        for key, value in saved.items():
            if key in options and (key != "settings" or isinstance(value, dict)):
                options[key] = ({**options["settings"], **value} if key == "settings" else value)
    options["export"] = export_choice(options["export"])
    return options


def scan(archive):
    """The measurements of an archive: {fov: [setups with PIV results]} (its subfolders that have some)."""
    found = {}
    try:
        names = sorted(os.listdir(archive))
    except OSError:
        return found
    for fov in names:
        root = os.path.join(archive, fov)
        if not os.path.isdir(root):
            continue
        setups = sorted(name for name in os.listdir(root)
                        if os.path.isfile(os.path.join(root, name, vh.PIV_FOLDER, "vectors.npz")))
        if setups:
            found[fov] = setups
    return found


def count_maps(archive, found):
    """The number of maps of the first measurement of the archive (from the header of its vectors), None if unknown."""
    for fov, setups in found.items():
        for setup in setups:
            try:
                path = os.path.join(archive, fov, setup, vh.PIV_FOLDER, "vectors.npz")
                with zipfile.ZipFile(path) as z, z.open("u_px.npy") as f:
                    version = np.lib.format.read_magic(f)
                    read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                            else np.lib.format.read_array_header_2_0)
                    return int(read(f)[0][0])
            except (OSError, KeyError, ValueError, zipfile.BadZipFile):
                continue
    return None


def describe_export(choice, n, index, rolling=False):
    """The exports of each measurement, in a line: the figures, the maps of the instantaneous ones (all of them with
    rolling statistics), the videos."""
    figures = len(FIGURE_TITLES) if choice["figures"] is None else len(choice["figures"])
    maps = export_maps(choice, n) if choice["range"] and n else None
    text = (f"{figures} of {len(FIGURE_TITLES)} figures; "
            + ("all of them (rolling statistics): " if rolling else "instantaneous: ")) + (
        f"maps {maps.start + 1} to {maps[-1] + 1} every {maps.step} ({len(maps)} images each)" if maps
        else f"the map {index}")
    if choice["videos"]:
        text += (f"; {len(choice['videos'])} {choice['format'].upper()} video(s) over "
                 + ("the same maps" if maps else "all the maps")
                 + f" at {num2str(choice['fps'])} frames/s")
    return text


def apply(options):
    """The options set in post_archive (its globals, read when it runs)."""
    r.ARCHIVE = options["archive"]
    widths = {fov: float(w) for fov, w in options["fov_width"].items() if w not in (None, "")}
    r.FOV_WIDTH_MM = widths
    r.FOV_X_OFFSET_MM = {fov: float(options["fov_offset"].get(fov) or 0) for fov in widths}
    r.FOV_Y_OFFSET_MM = {fov: float((options.get("fov_offset_y") or {}).get(fov) or 0) for fov in widths}
    r.FRAME_RATE_HZ = float(options["frame_rate_hz"])
    r.SETTINGS = vh.PostSettings(**options["settings"])
    r.REFERENCE_FOV = options["reference_fov"]
    r.MATCH_VELOCITY = bool(options["match_velocity"])
    r.MATCH_JET = float(options["match_jet"])
    r.MATCH_EDGE = int(options["match_edge"])
    r.INDEX = int(options["index"])
    choice = options["export"]
    r.EXPORT_FIGURES, r.EXPORT_VIDEOS, r.EXPORT_FPS = choice["figures"], choice["videos"], float(choice["fps"])
    r.EXPORT_FORMAT = choice["format"]
    n = options.get("n_maps")
    r.EXPORT_MAPS = list(export_maps(choice, n)) if choice["range"] and n else None
    r.STITCHED_FOLDER = options["stitched"] or os.path.join(options["archive"], "Stitched")
    r.STITCHED = list(options["stitched_maps"])
    r.STITCHED_INSTANTANEOUS = list(options["stitched_instantaneous"])
    r.STITCHED_VIDEOS = list(options["stitched_videos"])
    r.STITCHED_JET = list(options["stitched_jet"])


def run(path):
    """The run of the options saved in a file (the process started by the interface)."""
    with open(path, encoding="utf-8") as f:
        options = json.load(f)
    apply(options)
    ok = r.main(r.ARCHIVE, set(options["setups"]) if options["setups"] else None, options["export_root"] or None,
                r.STITCHED_FOLDER)
    return 0 if ok else 1


def number(text, name, kind=float, empty=None):
    """A number entered (empty: empty), ValueError naming the option."""
    text = text.strip()
    if not text:
        if empty is not None:
            return empty
        raise ValueError(f"{name}: a number is needed")
    try:
        return kind(float(text)) if kind is int and float(text).is_integer() else kind(text)
    except ValueError:
        raise ValueError(f"{name}: {'an integer' if kind is int else 'a number'} is needed, not {text!r}") from None


class Checklist(ttk.LabelFrame):
    """Check boxes of figures (their titles), in columns, with All / None."""

    def __init__(self, master, title, keys, chosen, columns=2):
        super().__init__(master, text=title, padding=6)
        self.vars = {key: tk.BooleanVar(self, key in chosen) for key in keys}
        for i, key in enumerate(keys):
            ttk.Checkbutton(self, text=FIGURE_TITLES[key], variable=self.vars[key]).grid(
                row=i // columns, column=i % columns, sticky=tk.W, padx=(0, 12))
        row = ttk.Frame(self)
        row.grid(row=(len(keys) - 1) // columns + 1, column=0, columnspan=columns, sticky=tk.W, pady=(4, 0))
        ttk.Button(row, text="All", width=6, command=lambda: self.set_all(True)).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row, text="None", width=6, command=lambda: self.set_all(False)).pack(side=tk.LEFT)

    def set_all(self, value):
        for var in self.vars.values():
            var.set(value)

    def get(self):
        return [key for key, var in self.vars.items() if var.get()]

    def set(self, chosen):
        for key, var in self.vars.items():
            var.set(key in chosen)


class ArchiveGUI(ttk.Frame):
    """The window: the options in tabs, the run below (Run, Stop, progress, messages)."""

    def __init__(self, root):
        super().__init__(root, padding=8)
        self.root = root
        root.title("VelociHub · Post-processing of the archive")
        self.pack(fill=tk.BOTH, expand=True)
        self.process, self.lines = None, queue.Queue()
        self.stopping, self.skip_source, self.config_path = False, False, None
        self.n_maps, self.found = None, {}
        options = saved_options()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=False)
        self.build_archive_tab()
        self.build_fov_tab()
        self.build_recipe_tab()
        self.build_export_tab()
        self.build_stitch_tab()
        self.build_run_bar()
        self.show(options)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_id = self.after(100, self.poll)

    # ------------------------------------------------------------------ building
    def tab(self, title):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=title)
        return frame

    @staticmethod
    def row(grid, r, label, widget, help_text=None, column=0):
        ttk.Label(grid, text=label).grid(row=r, column=column, sticky=tk.W, pady=2, padx=(0, 8))
        widget.grid(row=r, column=column + 1, sticky=tk.EW, pady=2)
        if help_text:
            HelpDot(grid, help_text).grid(row=r, column=column + 2, padx=(6, 12))

    def build_archive_tab(self):
        tab = self.tab("Archive")
        self.archive, self.export_root, self.stitched = tk.StringVar(self), tk.StringVar(self), tk.StringVar(self)
        grid = ttk.Frame(tab)
        grid.pack(fill=tk.X)
        grid.columnconfigure(1, weight=1)
        self.paths = {}
        for r_, (key, var, label, reset) in enumerate((("archive", self.archive, "Archive folder", False),
                                                       ("export_root", self.export_root, "Exports root (optional)",
                                                        True),
                                                       ("stitched", self.stitched, "Stitched figures (optional)",
                                                        True))):
            box = PathBox(grid, label, lambda folder, var=var, key=key: self.path_chosen(key, var, folder),
                          reset=reset, width=70)
            self.paths[key] = box
            self.row(grid, r_, label, box, OPTION_HELP[key])
        box = ttk.LabelFrame(tab, text="Setups to process", padding=6)
        box.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        top = ttk.Frame(box)
        top.pack(fill=tk.X)
        ttk.Button(top, text="All", width=6, command=lambda: self.setup_list.select_set(0, tk.END)).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Button(top, text="None", width=6, command=lambda: self.setup_list.select_clear(0, tk.END)).pack(
            side=tk.LEFT)
        ttk.Button(top, text="Several fields of view", command=self.select_stitched).pack(side=tk.LEFT, padx=(4, 0))
        HelpDot(top, OPTION_HELP["setups"]).pack(side=tk.LEFT, padx=6)
        self.setup_count = ttk.Label(top, foreground="#555555")
        self.setup_count.pack(side=tk.RIGHT)
        frame = ttk.Frame(box)
        frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.setup_list = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=9, exportselection=False,
                                     activestyle="none")
        bar = ttk.Scrollbar(frame, command=self.setup_list.yview)
        self.setup_list.configure(yscrollcommand=bar.set)
        self.setup_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar.pack(side=tk.LEFT, fill=tk.Y)
        self.setup_list.bind("<<ListboxSelect>>", lambda event: self.update_count())
        self.setup_names = []

    def build_fov_tab(self):
        tab = self.tab("Fields of view")
        self.fov_box = ttk.LabelFrame(tab, text="Fields of view", padding=6)
        self.fov_box.pack(fill=tk.X)
        self.fov_vars = {}
        grid = ttk.Frame(tab)
        grid.pack(fill=tk.X, pady=(8, 0))
        self.frame_rate = tk.StringVar(self)
        self.row(grid, 0, "Frame rate (Hz)", ttk.Entry(grid, textvariable=self.frame_rate, width=14),
                 OPTION_HELP["frame_rate_hz"])

    def build_fov_rows(self, widths, offsets, vertical=None):
        """The rows of the fields of view: those of the archive and those of the options."""
        vertical = vertical or {}
        for child in self.fov_box.winfo_children():
            child.destroy()
        fovs = sorted(set(self.found) | {fov for fov, w in widths.items() if w not in (None, "")})
        for c, text in enumerate(("Field of view", "Width (mm)", "Horiz. offset (mm)", "Vert. offset (mm)",
                                  "Setups")):
            ttk.Label(self.fov_box, text=text, foreground="#555555").grid(row=0, column=c, sticky=tk.W, padx=(0, 12))
        HelpDot(self.fov_box, OPTION_HELP["fovs"]).grid(row=0, column=5)
        self.fov_vars = {}
        for r_, fov in enumerate(fovs, 1):
            width = tk.StringVar(self, "" if widths.get(fov) in (None, "") else num2str(widths[fov]))
            offset = tk.StringVar(self, "" if offsets.get(fov) in (None, "") else num2str(offsets[fov]))
            offset_y = tk.StringVar(self, "" if vertical.get(fov) in (None, "") else num2str(vertical[fov]))
            self.fov_vars[fov] = (width, offset, offset_y)
            ttk.Label(self.fov_box, text=fov).grid(row=r_, column=0, sticky=tk.W)
            for c, var in enumerate((width, offset, offset_y), 1):
                ttk.Entry(self.fov_box, textvariable=var, width=12).grid(row=r_, column=c, sticky=tk.W, pady=1)
            ttk.Label(self.fov_box, text=str(len(self.found.get(fov, []))) if fov in self.found else "not in the "
                      "archive", foreground="#555555").grid(row=r_, column=4, sticky=tk.W)
        self.reference_box.configure(values=fovs)

    def build_recipe_tab(self):
        tab = self.tab("Recipe")
        box = ttk.LabelFrame(tab, text="Post-processing recipe (common to all the measurements)", padding=6)
        box.pack(fill=tk.X)
        self.settings_vars = {}
        names = [f for f in dataclass_fields(vh.PostSettings)]
        labels = {key: label for _, key, label, _ in PAGE_SETTINGS}
        half = (len(names) + 1) // 2
        for i, f in enumerate(names):
            r_, c = i % half, 4 * (i // half)
            label = LABELS.get(f.name) or labels.get(f.name) or f.name
            if f.type is bool or f.type == "bool":
                var = tk.BooleanVar(self)
                widget = ttk.Checkbutton(box, variable=var)
            elif f.name == "core_mode":
                var = tk.StringVar(self)
                widget = ttk.Combobox(box, textvariable=var, values=CORE_MODES, width=10, state="readonly")
            else:
                var = tk.StringVar(self)
                widget = ttk.Entry(box, textvariable=var, width=12)
            self.settings_vars[f.name] = var
            self.row(box, r_, label, widget, HELP.get(f.name), column=c)
        # The exports of each map: all the figures with rolling statistics
        self.settings_vars["rolling"].trace_add("write", lambda *args: self.update_export_label())
        box = ttk.LabelFrame(tab, text="Reference velocities", padding=6)
        box.pack(fill=tk.X, pady=(8, 0))
        self.reference_fov, self.match_velocity = tk.StringVar(self), tk.BooleanVar(self)
        self.match_jet, self.match_edge = tk.StringVar(self), tk.StringVar(self)
        self.reference_box = ttk.Combobox(box, textvariable=self.reference_fov, width=12, state="readonly")
        self.row(box, 0, "Reference field of view", self.reference_box, OPTION_HELP["reference_fov"])
        ttk.Checkbutton(box, text="Match the fields of view in their overlaps",
                        variable=self.match_velocity).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)
        HelpDot(box, OPTION_HELP["match_velocity"]).grid(row=1, column=2, padx=(6, 12))
        self.row(box, 2, "Jet cells: at least (fraction of the max)", ttk.Entry(box, textvariable=self.match_jet,
                                                                                 width=12))
        self.row(box, 3, "Edge columns left out", ttk.Entry(box, textvariable=self.match_edge, width=12))

    def build_export_tab(self):
        tab = self.tab("Exports")
        grid = ttk.Frame(tab)
        grid.pack(fill=tk.X)
        self.index = tk.StringVar(self)
        self.row(grid, 0, "Map of the instantaneous figures", ttk.Entry(grid, textvariable=self.index, width=12),
                 OPTION_HELP["index"])
        for var in (self.index,):
            var.trace_add("write", lambda *args: self.update_export_label())
        row = ttk.Frame(tab)
        row.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(row, text="Figures to export...", command=self.choose_export).pack(side=tk.LEFT)
        HelpDot(row, OPTION_HELP["export"]).pack(side=tk.LEFT, padx=6)
        self.export_label = ttk.Label(tab, foreground="#555555", wraplength=760, justify=tk.LEFT)
        self.export_label.pack(anchor=tk.W, pady=(6, 0))
        self.export = None

    def build_stitch_tab(self):
        tab = self.tab("Stitching")
        grid = ttk.Frame(tab)
        grid.pack(fill=tk.X)
        ttk.Label(grid, text="The figures stitched (their size from the maps)").grid(row=0, column=0, sticky=tk.W)
        HelpDot(grid, OPTION_HELP["stitched_lists"]).grid(row=0, column=1, padx=6)
        lists = ttk.Frame(tab)
        lists.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.stitch_maps = Checklist(lists, "Maps of the statistics", STATISTICS_MAPS, [], columns=3)
        self.stitch_maps.grid(row=0, column=0, columnspan=3, sticky="nsew", pady=(0, 6))
        self.stitch_instantaneous = Checklist(lists, "Instantaneous maps", INSTANTANEOUS_MAPS, [], columns=1)
        self.stitch_instantaneous.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.stitch_videos = Checklist(lists, "Videos (slow; statistics: rolling only)", MAP_FIGURES, [],
                                       columns=1)
        self.stitch_videos.grid(row=1, column=1, sticky="nsew", padx=(0, 6))
        self.stitch_jet = Checklist(lists, "Jet theory plots", JET_PLOTS, [], columns=1)
        self.stitch_jet.grid(row=1, column=2, sticky="nsew")

    def build_run_bar(self):
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, pady=(8, 4))
        self.run_button = ttk.Button(bar, text="Run the export", command=self.start)
        self.run_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(bar, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(bar, text="Defaults of the script", command=lambda: self.show(script_defaults())).pack(
            side=tk.LEFT, padx=(18, 0))
        ttk.Button(bar, text="Open the stitched figures", command=self.open_stitched).pack(side=tk.RIGHT)
        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill=tk.X)
        self.status = ttk.Label(self, foreground="#555555")
        self.status.pack(anchor=tk.W, pady=(2, 4))
        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True)
        self.log = tk.Text(frame, height=14, wrap=tk.NONE, font=("Consolas", 9), state=tk.DISABLED)
        bar_y = ttk.Scrollbar(frame, command=self.log.yview)
        bar_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.log.xview)
        self.log.configure(yscrollcommand=bar_y.set, xscrollcommand=bar_x.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        bar_y.grid(row=0, column=1, sticky="ns")
        bar_x.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    # ------------------------------------------------------------------ options
    def show(self, options):
        """Show options (saved_options, script_defaults)."""
        self.archive.set(options["archive"])
        self.export_root.set(options["export_root"] or "")
        self.stitched.set(options["stitched"] or "")
        self.frame_rate.set(num2str(options["frame_rate_hz"]))
        for key, var in self.settings_vars.items():
            value = options["settings"].get(key)
            var.set(bool(value) if isinstance(var, tk.BooleanVar) else "" if value is None
                    else value if isinstance(value, str) else num2str(value))
        self.reference_fov.set(options["reference_fov"])
        self.match_velocity.set(bool(options["match_velocity"]))
        self.match_jet.set(num2str(options["match_jet"]))
        self.match_edge.set(str(options["match_edge"]))
        self.index.set(str(options["index"]))
        self.export = dict(options["export"])
        self.stitch_maps.set(options["stitched_maps"])
        self.stitch_instantaneous.set(options["stitched_instantaneous"])
        self.stitch_videos.set(options["stitched_videos"])
        self.stitch_jet.set(options["stitched_jet"])
        self.load_archive(options["fov_width"], options["fov_offset"], options["setups"],
                          options.get("fov_offset_y"))

    def path_chosen(self, key, var, folder):
        var.set(folder or "")
        self.paths[key].show(folder or "")
        if key == "archive":
            widths = {fov: w.get() for fov, (w, _, _) in self.fov_vars.items()}
            offsets = {fov: o.get() for fov, (_, o, _) in self.fov_vars.items()}
            vertical = {fov: v.get() for fov, (_, _, v) in self.fov_vars.items()}
            self.load_archive(widths, offsets, None, vertical)

    def load_archive(self, widths, offsets, setups, vertical=None):
        """The measurements of the archive: the setups listed (setups selected; None: all), the fields of view."""
        for key, var in (("archive", self.archive), ("export_root", self.export_root), ("stitched", self.stitched)):
            self.paths[key].show(var.get())
        self.found = scan(self.archive.get())
        self.n_maps = count_maps(self.archive.get(), self.found)
        fovs_of = {}
        for fov, names in self.found.items():
            for name in names:
                fovs_of.setdefault(name, []).append(fov)
        self.setup_names = sorted(fovs_of)
        self.fovs_of = fovs_of
        self.setup_list.delete(0, tk.END)
        for name in self.setup_names:
            self.setup_list.insert(tk.END, f"{name}   ({', '.join(fovs_of[name])})")
        for i, name in enumerate(self.setup_names):
            if setups is None or name in setups:
                self.setup_list.select_set(i)
        self.build_fov_rows(widths, offsets, vertical)
        self.update_count()
        self.update_export_label()
        if not self.found:
            self.set_status(f"No measurement with PIV results in {self.archive.get()}")
        else:
            self.set_status(f"{sum(len(v) for v in self.found.values())} measurements in {len(self.found)} fields "
                            f"of view, {len(self.setup_names)} setups"
                            + (f", {self.n_maps} maps each" if self.n_maps else ""))

    def select_stitched(self):
        self.setup_list.select_clear(0, tk.END)
        for i, name in enumerate(self.setup_names):
            if len(self.fovs_of[name]) > 1:
                self.setup_list.select_set(i)
        self.update_count()

    def update_count(self):
        chosen = self.chosen_setups()
        self.setup_count["text"] = (f"{len(chosen)} of {len(self.setup_names)} setups, "
                                    f"{sum(len(self.fovs_of[s]) for s in chosen)} measurements")

    def chosen_setups(self):
        return [self.setup_names[i] for i in self.setup_list.curselection()]

    def choose_export(self):
        n = self.n_maps or max(int(self.export.get("last") or 1), 1)
        # The figures of each map: the instantaneous ones, or all of them with rolling statistics (Recipe)
        rolling = self.settings_vars["rolling"].get()
        page = SimpleNamespace(post=SimpleNamespace(
            fields=SimpleNamespace(n=n),
            instantaneous=lambda key: key in INSTANTANEOUS or (rolling and key in FIGURE_TITLES)))
        result = ExportDialog(self.root, page, self.export).run()
        if result is not None:
            self.export = result
            self.update_export_label()

    def update_export_label(self):
        if self.export is None:
            return
        self.export_label["text"] = describe_export(self.export, self.n_maps, self.index.get().strip() or "?",
                                                    self.settings_vars["rolling"].get())

    def options(self):
        """The options entered: ValueError with a message if one is invalid."""
        settings = {}
        for f in dataclass_fields(vh.PostSettings):
            var = self.settings_vars[f.name]
            if isinstance(var, tk.BooleanVar) or f.name == "core_mode":
                settings[f.name] = var.get()
            else:
                text = var.get().strip()
                settings[f.name] = None if not text else number(text, LABELS.get(f.name, f.name),
                                                                int if f.type is int else float)
        vh.PostSettings(**settings)  # ValueError if invalid
        widths, offsets, vertical = {}, {}, {}
        for fov, (width, offset, offset_y) in self.fov_vars.items():
            if width.get().strip():
                widths[fov] = number(width.get(), f"Width of {fov}")
                offsets[fov] = number(offset.get(), f"Horiz. offset of {fov}", empty=0.0)
                vertical[fov] = number(offset_y.get(), f"Vert. offset of {fov}", empty=0.0)
        if not widths:
            raise ValueError("Fields of view: a width is needed for at least one of them")
        if not self.archive.get() or not os.path.isdir(self.archive.get()):
            raise ValueError("The archive folder doesn't exist")
        setups = self.chosen_setups()
        if not setups:
            raise ValueError("Choose at least one setup to process")
        return dict(
            archive=self.archive.get(), export_root=self.export_root.get(), stitched=self.stitched.get(),
            setups=None if len(setups) == len(self.setup_names) else setups,
            fov_width=widths, fov_offset=offsets, fov_offset_y=vertical,
            frame_rate_hz=number(self.frame_rate.get(), "Frame rate"), settings=settings,
            reference_fov=self.reference_fov.get(), match_velocity=self.match_velocity.get(),
            match_jet=number(self.match_jet.get(), "Jet cells"), match_edge=number(self.match_edge.get(),
                                                                                    "Edge columns", int),
            index=number(self.index.get(), "Map of the instantaneous figures", int),
            export=self.export, n_maps=self.n_maps,
            stitched_maps=self.stitch_maps.get(), stitched_instantaneous=self.stitch_instantaneous.get(),
            stitched_videos=self.stitch_videos.get(), stitched_jet=self.stitch_jet.get())

    def save(self):
        """Keep the options entered (if valid) for the next session."""
        try:
            options = self.options()
        except ValueError:
            return None
        setpref(PREF, {key: value for key, value in options.items() if key != "n_maps"})
        return options

    # ------------------------------------------------------------------ run
    def start(self):
        try:
            options = self.options()
        except ValueError as e:
            messagebox.showerror("Post-processing of the archive", str(e), parent=self.root)
            return
        self.save()
        where = options["export_root"] or "the PostProcessing subfolder of each measurement"
        stitched = options["stitched"] or os.path.join(options["archive"], "Stitched")
        if not messagebox.askokcancel(
                "Post-processing of the archive",
                f"Process {len(self.chosen_setups())} setups?\n\nExports: {where}\nStitched figures: {stitched}\n\n"
                "Files of the same names there are replaced.", parent=self.root):
            return
        handle, path = tempfile.mkstemp(prefix="velocihub_archive_", suffix=".json")
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            json.dump(options, f)
        self.clear_log()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.process = subprocess.Popen([sys.executable, "-u", "-m", "velocihub.archiving.post_archive_gui", "--run",
                                         path], cwd=PACKAGE_PARENT, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                        env=env, creationflags=flags)
        self.config_path = path
        threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()
        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.progress.configure(value=0, maximum=1)
        self.set_status("Running...")

    def read_output(self, process):
        for line in process.stdout:
            self.lines.put(line)
        process.wait()
        self.lines.put(None)

    def poll(self):
        """The messages of the run shown, its progress from the measurements started ("[k/n] folder")."""
        try:
            while True:
                line = self.lines.get_nowait()
                if line is None:
                    self.finished()
                    continue
                # The warnings of Python (numpy: empty slices...) and their source line left out
                if self.skip_source and line.startswith(" "):
                    self.skip_source = False
                    continue
                self.skip_source = bool(WARNING.match(line))
                if self.skip_source:
                    continue
                self.append_log(line)
                match = re.match(r"\[(\d+)/(\d+)\]\s+(.*)", line.strip())
                if match:
                    k, n = int(match.group(1)), int(match.group(2))
                    self.progress.configure(maximum=n, value=k - 1)
                    self.set_status(f"Measurement {k} of {n}: {os.path.basename(match.group(3))}")
        except queue.Empty:
            pass
        self.poll_id = self.after(100, self.poll)

    def finished(self):
        code = self.process.returncode if self.process is not None else None
        stopped = getattr(self, "stopping", False)
        self.process, self.stopping = None, False
        try:
            os.remove(self.config_path)
        except OSError:
            pass
        self.run_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        if stopped:
            self.set_status("Stopped")
        elif code == 0:
            self.progress.configure(value=self.progress["maximum"])
            self.set_status("Done")
        else:
            self.set_status(f"Finished with errors (exit code {code}): see the messages")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.stopping = True
            self.process.terminate()
            self.append_log("\nStopped by the user\n")

    def append_log(self, text):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def clear_log(self):
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    def set_status(self, text):
        self.status["text"] = text

    def open_stitched(self):
        folder = self.stitched.get() or os.path.join(self.archive.get(), "Stitched")
        if os.path.isdir(folder):
            os.startfile(folder) if sys.platform == "win32" else subprocess.Popen(["xdg-open", folder])
        else:
            messagebox.showinfo("Stitched figures", f"The folder doesn't exist yet:\n{folder}", parent=self.root)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Post-processing of the archive", "A run is in progress: stop it and quit?",
                                       parent=self.root):
                return
            self.stop()
        self.save()
        self.after_cancel(self.poll_id)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Interface of the post-processing of the archive")
    parser.add_argument("--run", help=argparse.SUPPRESS)  # The run of the options of a file (by the interface)
    args = parser.parse_args()
    if args.run:
        sys.exit(run(args.run))
    root = tk.Tk()
    ArchiveGUI(root)
    root.minsize(860, 640)
    root.mainloop()


if __name__ == "__main__":
    main()
