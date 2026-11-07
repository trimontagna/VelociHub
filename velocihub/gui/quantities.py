"""The quantities of the post-processing in their groups, with their images and their video exported (columns Images
and Video, clicked to change them), the parameters of the quantity selected as the tiles have them (colour range,
colour map, isocontour, second quantity compared with a wiper and the position of the wiper) and its own video and
instantaneous maps (overriding the global settings), and the global settings of the export: the videos and the
instantaneous maps. The parameters changed from the default are marked ≠, their number in the column Parameters.
The quantities of the PTV are greyed out (not chosen) when the vector fields aren't those of a PTV, those of the images
(raw, pre-processed) and of their POD when they aren't found (the POD mode chosen not evaluated).
The panel (QuantitiesEditor) edits an export specification (export_spec): in the window Quantities of the page (Run
step > Quantities..., QuantitiesDialog), and in the window Batch for the node selected."""

import copy
import tkinter as tk
from tkinter import messagebox, ttk

from ..fields import num2str
from ..imaging import STATIC
from ..lagrangian import GROUP as PTV_GROUP
from ..postprocessing import FIGURE_GROUPS, FIGURE_TITLES, INSTANTANEOUS, MAP_FIGURES
from .postprocessing_page import FORMAT_LABELS, images_summary, video_summary
from ..settings import COMPARE_FRACTION, DEFAULT_EXPORT, image_maps, image_settings, read_step_fps, video_settings
from .widgets import Dialog, RangeSlider
from ..settings import COLORMAPS
from ..settings import export_spec

AUTO = "(the figure's)"
NONE = "None"
# The cells of the columns Images and Video: exported, not exported, not possible (a setting rules it out); some of
# the quantities of a group exported
CHECKED, UNCHECKED, UNAVAILABLE, PARTIAL = "☑", "☐", "⬚", "▣"
# The quantities of the PTV (exported for the results of a PTV only) and their group
PTV_KEYS, PTV_GROUP_ROW = set(PTV_GROUP[1]), "group|" + PTV_GROUP[0]


def quantity_changes(spec, key):
    """The parameters of a quantity changed from the default export: images (not exported), video, colour range,
    colour map, isocontour, second quantity, wiper position, its own video settings and instantaneous maps."""
    choice, params = spec["choice"], spec["params"]
    out = set()
    if not (choice["figures"] is None or key in choice["figures"]):
        out.add("images")
    if key in choice["videos"]:
        out.add("video")
    out |= {name for name in ("clims", "cmaps", "contours", "compares") if key in params[name]}
    if abs(params["wipers"].get(key, COMPARE_FRACTION) - COMPARE_FRACTION) > 1e-9:
        out.add("wipers")
    if key in spec["videos"]:
        out.add("own_video")
    if key in spec.get("images", {}):
        out.add("own_images")
    return out


def global_changes(spec):
    """The global settings of the export changed from the default: videos (every N maps, frames per second, the
    first and the last maps, the format), instantaneous maps (the maps chosen, every N maps)."""
    choice = spec["choice"]
    out = {"video_" + name for name in ("step", "fps", "format") if choice[name] != DEFAULT_EXPORT[name]}
    if (int(choice["first"] or 1), choice["last"]) != (int(DEFAULT_EXPORT["first"] or 1), DEFAULT_EXPORT["last"]):
        out.add("video_maps")
    if choice.get("map"):
        out.add("images_maps")
    if (choice.get("map_step") or 1) != 1:
        out.add("images_step")
    return out


class VideoBox:
    """The settings of videos (a box): every N maps, frames per second, the first and the last maps (a two-sided
    slider, described below it), the format; override: with a toggle overriding the global settings."""

    def __init__(self, editor, master, title, override=False, marks="", width=300):
        self.frame = e = ttk.LabelFrame(master, text=title, padding=6)
        self.override = tk.BooleanVar(editor) if override else None
        callback = editor.quantity_changed if override else editor.global_changed
        r = 0
        if override:
            self.toggle = ttk.Checkbutton(e, text="Override global settings", variable=self.override,
                                          command=callback)
            self.toggle.grid(row=0, column=0, columnspan=3, sticky=tk.W)
            editor.mark("own_video", self.toggle)
            r = 1
        self.step, self.fps, self.format = tk.StringVar(editor), tk.StringVar(editor), tk.StringVar(editor)
        self.widgets = []

        def label(text, name):
            widget = ttk.Label(e, text=text)
            return editor.mark(marks + name, widget) if marks else widget
        for i, (text, name, var) in enumerate((("Every N maps", "step", self.step),
                                               ("Frames per second", "fps", self.fps))):
            label(text, name).grid(row=r + i, column=0, sticky=tk.W, pady=1)
            entry = ttk.Entry(e, textvariable=var, width=8)
            entry.grid(row=r + i, column=1, sticky=tk.W, pady=1)
            self.widgets.append(entry)
            var.trace_add("write", lambda *args: callback())
        label("First and last maps", "maps").grid(row=r + 2, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))
        self.slider = RangeSlider(e, 1, editor.maps(), command=lambda first, last: callback(), width=width)
        self.slider.grid(row=r + 3, column=0, columnspan=3, sticky="ew")
        self.info = ttk.Label(e, foreground="#555555", wraplength=width + 20, justify=tk.LEFT)
        self.info.grid(row=r + 4, column=0, columnspan=3, sticky=tk.W)
        label("Format", "format").grid(row=r + 5, column=0, sticky=tk.W, pady=1)
        self.box = ttk.Combobox(e, values=list(FORMAT_LABELS.values()), state="readonly", width=18)
        self.box.grid(row=r + 5, column=1, columnspan=2, sticky=tk.W, pady=1)
        self.box.bind("<<ComboboxSelected>>", lambda event: (self.format.set(
            next(k for k, v in FORMAT_LABELS.items() if v == self.box.get())), callback()))
        self.widgets.append(self.box)

    def set(self, editor, settings):
        """Show video settings (video_settings)."""
        self.step.set(str(settings["step"]))
        self.fps.set(num2str(settings["fps"]))
        editor.place(self.slider, int(settings["first"] or 1), settings["last"])
        self.format.set(settings["format"])
        self.box.set(FORMAT_LABELS[settings["format"]])

    def read(self, editor):
        """The settings shown (video_settings): ValueError if every N maps or frames per second are invalid."""
        step, fps = read_step_fps(self.step.get(), self.fps.get())
        first, last = editor.read_maps(self.slider)
        return video_settings({"step": step, "fps": fps, "format": self.format.get(), "first": first, "last": last})

    def enable(self, on):
        for w in self.widgets:
            w.state(["!disabled"] if on else ["disabled"])
        self.slider.set_enabled(on)

    def describe(self, editor, compute):
        """The maps of the slider below it, as in the window Video of the page (the vector fields loaded when
        compute); its maps only otherwise."""
        first, last = self.slider.get()
        if not compute:
            self.info["text"] = f"Maps {first} to {last}" + (f" of {editor.maps()}" if editor.known() else "")
            return
        post = editor.vector_fields()
        try:
            step, fps = read_step_fps(self.step.get(), self.fps.get())
        except ValueError as e:
            self.info["text"] = str(e)
            return
        self.info["text"] = video_summary(post, first, last, step, fps)


class ImagesBox:
    """The instantaneous maps exported (a box): the first and the last maps (a two-sided slider, described below
    it: their times computed when it is moved), every N maps; override: with a toggle overriding the global
    settings."""

    def __init__(self, editor, master, title, override=False, marks="", width=300):
        self.frame = e = ttk.LabelFrame(master, text=title, padding=6)
        self.override = tk.BooleanVar(editor) if override else None
        callback = editor.quantity_changed if override else editor.global_changed
        r = 0
        if override:
            self.toggle = ttk.Checkbutton(e, text="Override global settings", variable=self.override,
                                          command=callback)
            self.toggle.grid(row=0, column=0, columnspan=3, sticky=tk.W)
            editor.mark("own_images", self.toggle)
            r = 1

        def label(text, name):
            widget = ttk.Label(e, text=text)
            return editor.mark(marks + name, widget) if marks else widget
        label("First and last maps", "maps").grid(row=r, column=0, columnspan=3, sticky=tk.W)
        self.slider = RangeSlider(e, 1, editor.maps(), command=lambda first, last: callback(moved=True), width=width)
        self.slider.grid(row=r + 1, column=0, columnspan=3, sticky="ew")
        self.info = ttk.Label(e, foreground="#555555", wraplength=width + 20, justify=tk.LEFT)
        self.info.grid(row=r + 2, column=0, columnspan=3, sticky=tk.W)
        self.step = tk.StringVar(editor)
        label("Every N maps", "step").grid(row=r + 3, column=0, sticky=tk.W, pady=1)
        entry = ttk.Entry(e, textvariable=self.step, width=8)
        entry.grid(row=r + 3, column=1, sticky=tk.W, pady=1)
        self.step.trace_add("write", lambda *args: callback())
        self.widgets = [entry]

    def set(self, editor, first, last, step):
        """Show maps (first None: the last map, the slider at it if the maps are known)."""
        if not first:
            n = editor.maps() if editor.known() else 1
            editor.place(self.slider, n, n)
        else:
            editor.place(self.slider, int(first), last)
        self.step.set(str(step or 1))

    def read(self):
        """The maps shown: first, last, every N maps (ValueError if invalid)."""
        first, last = self.slider.get()
        step = int(self.step.get())
        if step < 1:
            raise ValueError
        return first, last, step

    def enable(self, on):
        for w in self.widgets:
            w.state(["!disabled"] if on else ["disabled"])
        self.slider.set_enabled(on)

    def describe(self, editor, first, last, step, compute):
        """The maps below the slider (first None: the last map); their times when compute."""
        if not first:
            self.info["text"] = "Default: the last map (move the handles to choose maps)"
            return
        last = min(last or editor.maps(), editor.maps()) if editor.known() else (last or first)
        self.info["text"] = images_summary(editor.vector_fields() if compute else editor.post, first, last, step)


class QuantitiesEditor(ttk.Frame):
    """The quantities of an export specification (edited on a copy: spec) and their parameters, and the global
    settings of the export. rolling: every figure has one per map (videos of all); on_change(spec): called with the
    specification at each valid change; height: rows of the list; post: the post-processing of the vector fields (the
    maps: their number, their times; set_post), None if unknown; loader: a function giving it when it is needed (a
    summary of a video of the quantity selected with its video on, the times of the instantaneous maps moved), with
    n_maps the number of maps meanwhile (set_loader); ptv: the vector fields are those of a PTV (set_ptv; None:
    unknown), the quantities of the PTV greyed out if not; missing: why the quantities of the images and of their POD
    can't be exported ({key: reason}, set_imaging), those greyed out."""

    def __init__(self, master, spec, rolling=False, on_change=None, height=20, post=None):
        super().__init__(master)
        self.spec, self.rolling, self.on_change, self.post = export_spec(copy.deepcopy(spec)), rolling, on_change, post
        self.loader, self.n_maps, self.ptv, self.missing = None, None, None, {}
        # The pictures of the POD modes exported: one per mode of the range, or their sum, a figure of each map
        self.pod_images, self.pod_per_map = 1, False
        self.key, self.loading, self.enabled = None, False, True
        self.marks = {}  # The labels of the parameters marked ≠ when they are changed: {name: (widget, text)}
        ttk.Label(self, text=f"Images, Video: click to export them ({CHECKED}: exported, {UNCHECKED}: not exported, "
                             f"{UNAVAILABLE}: rolling average dependent), on a group to export all of its quantities "
                             f"or none ({PARTIAL}: some); greyed: not available (the PTV quantities: for the "
                             "results of a PTV only; the images and the POD: when they are found); ≠: "
                             "changed from the default.", foreground="#555555", wraplength=900,
                  justify=tk.LEFT).pack(anchor=tk.W)
        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        # Left: the quantities, the global settings below them; right: the quantity selected
        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rows = ttk.Frame(left)
        rows.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(rows, columns=("images", "video", "params"), height=height, selectmode="browse")
        for column, text, width in (("#0", "Quantity", 200), ("images", "Images", 50), ("video", "Video", 44),
                                    ("params", "Parameters", 80)):
            self.tree.heading(column, text=text, anchor=tk.W if column in ("#0", "params") else tk.CENTER)
            self.tree.column(column, width=width, anchor=tk.W if column in ("#0", "params") else tk.CENTER,
                             stretch=column == "params")
        bar = ttk.Scrollbar(rows, command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar.pack(side=tk.LEFT, fill=tk.Y)
        self.groups = {}  # The quantities of each group: {its row: keys}
        for group, keys in FIGURE_GROUPS:
            gid = "group|" + (group or "Other")
            self.groups[gid] = list(keys)
            self.tree.insert("", tk.END, iid=gid, text=group or "Other", open=True)
            for key in keys:
                self.tree.insert(gid, tk.END, iid=key, text=FIGURE_TITLES[key])
        self.tree.tag_configure("unavailable", foreground="#9a9a9a")
        self.tree.bind("<<TreeviewSelect>>", lambda event: self.select())
        self.tree.bind("<Button-1>", self.click, add="+")
        boxes = ttk.LabelFrame(left, text="Global", padding=6)
        boxes.pack(fill=tk.X, pady=(6, 0))
        self.global_video = VideoBox(self, boxes, "Videos", marks="video_", width=220)
        self.global_video.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.global_images = ImagesBox(self, boxes, "Instantaneous maps", marks="images_", width=220)
        self.global_images.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        row = ttk.Frame(left)
        row.pack(fill=tk.X, pady=(6, 0))
        # All the quantities exported, or none
        self.check_buttons = [ttk.Button(row, text=text, command=lambda on=on: self.check_all(on))
                              for text, on in (("Check all", True), ("Uncheck all", False))]
        for n, button in enumerate(self.check_buttons):
            button.pack(side=tk.LEFT, padx=(0 if n == 0 else 6, 0))
        self.reset_button = ttk.Button(row, text="Reset all quantities to the default", command=self.reset)
        self.reset_button.pack(side=tk.LEFT, padx=(12, 0))
        self.summary = ttk.Label(row, foreground="#555555")
        self.summary.pack(side=tk.LEFT, padx=(10, 0))
        side = ttk.Frame(body)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        self.params_frame = ttk.LabelFrame(side, text="Parameters of the quantity", padding=6)
        self.params_frame.pack(fill=tk.X)
        self.build_params()
        # Its video and its instantaneous maps (overriding the global settings), with its parameters
        self.video_box = VideoBox(self, self.params_frame, "Video", override=True, width=240)
        self.video_box.frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.images_box = ImagesBox(self, self.params_frame, "Instantaneous maps", override=True, width=240)
        self.images_box.frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.reset_quantity_button = ttk.Button(side, text="Reset this quantity to defaults",
                                                command=self.reset_quantity)
        self.reset_quantity_button.pack(anchor=tk.W, pady=(6, 0))
        self.note = ttk.Label(side, foreground="#a33", wraplength=330, justify=tk.LEFT)
        self.note.pack(anchor=tk.W, pady=(4, 0))
        self.show(self.spec)
        if post is not None:
            self.set_post(post)  # The quantities it can't export greyed out
        first = FIGURE_GROUPS[0][1][0]
        self.tree.selection_set(first)
        self.tree.focus(first)

    # ------------------------------------------------------------------ widgets

    def build_params(self):
        e = self.params_frame
        self.title_label = ttk.Label(e, font=("TkDefaultFont", 9, "bold"), wraplength=320)
        self.title_label.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
        self.vmin, self.vmax, self.contour = tk.StringVar(self), tk.StringVar(self), tk.StringVar(self)
        self.cmap, self.compare = tk.StringVar(self), tk.StringVar(self)
        self.mark("clims", ttk.Label(e, text="Colour range (min, max)")).grid(row=1, column=0, sticky=tk.W)
        row = ttk.Frame(e)
        row.grid(row=1, column=1, columnspan=2, sticky=tk.W)
        self.map_widgets = []
        for var in (self.vmin, self.vmax):
            entry = ttk.Entry(row, textvariable=var, width=7)
            entry.pack(side=tk.LEFT, padx=(0, 4))
            self.map_widgets.append(entry)
        self.mark("cmaps", ttk.Label(e, text="Colour map")).grid(row=2, column=0, sticky=tk.W, pady=(4, 0))
        box = ttk.Combobox(e, textvariable=self.cmap, values=[AUTO] + list(COLORMAPS), state="readonly", width=13)
        box.grid(row=2, column=1, columnspan=2, sticky=tk.W, pady=(4, 0))
        self.map_widgets.append(box)
        self.mark("contours", ttk.Label(e, text="Isocontour (empty: none)")).grid(row=3, column=0, sticky=tk.W,
                                                                                pady=(4, 0))
        entry = ttk.Entry(e, textvariable=self.contour, width=10)
        entry.grid(row=3, column=1, columnspan=2, sticky=tk.W, pady=(4, 0))
        self.map_widgets.append(entry)
        self.mark("compares", ttk.Label(e, text="Compare with")).grid(row=4, column=0, sticky=tk.W, pady=(4, 0))
        self.compare_box = ttk.Combobox(e, textvariable=self.compare, state="readonly", width=16,
                                        values=[NONE] + [FIGURE_TITLES[k] for k in MAP_FIGURES])
        self.compare_box.grid(row=4, column=1, columnspan=2, sticky=tk.W, pady=(4, 0))
        self.map_widgets.append(self.compare_box)
        self.mark("wipers", ttk.Label(e, text="Wiper position")).grid(row=5, column=0, sticky=tk.W, pady=(4, 0))
        self.wiper = tk.DoubleVar(self, COMPARE_FRACTION)
        self.wiper_scale = ttk.Scale(e, from_=0, to=1, variable=self.wiper, orient=tk.HORIZONTAL, length=120,
                                     command=lambda value: self.quantity_changed())
        self.wiper_scale.grid(row=5, column=1, sticky=tk.W, pady=(4, 0))
        self.wiper_label = ttk.Label(e, width=5)
        self.wiper_label.grid(row=5, column=2, sticky=tk.W, pady=(4, 0))
        self.map_widgets.append(self.wiper_scale)
        for var in (self.vmin, self.vmax, self.contour, self.cmap, self.compare):
            var.trace_add("write", lambda *args: self.quantity_changed())

    def mark(self, name, widget):
        """A label of a parameter, marked ≠ when it is changed from the default (update_marks). Returns it."""
        self.marks[name] = (widget, widget.cget("text"))
        return widget

    def update_marks(self):
        """The parameters of the quantity selected and the global settings changed from the default: ≠."""
        changed = (quantity_changes(self.spec, self.key) if self.key is not None else set())
        changed |= global_changes(self.spec)
        for name, (widget, text) in self.marks.items():
            widget.configure(text=text + ("  ≠" if name in changed else ""))

    # ------------------------------------------------------------------ the maps of the vector fields

    def maps(self):
        """The number of maps of the sliders: those of the vector fields (unknown: 1)."""
        return self.post.fields.n if self.post is not None else self.n_maps or 1

    def known(self):
        return self.post is not None or self.n_maps is not None

    def vector_fields(self):
        """The post-processing of the vector fields (loaded now if needed: loader), None if unknown."""
        if self.post is None and self.loader is not None:
            self.post, self.loader = self.loader(), None
        return self.post

    def set_post(self, post):
        """The post-processing of the vector fields of the maps of the sliders (None: unknown), the quantities of the
        PTV greyed out if they aren't those of a PTV."""
        self.post, self.loader = post, None
        self.n_maps = post.fields.n if post is not None else None
        if post is not None:
            self.ptv, self.missing = post.ptv(), self.reasons(post.imaging_unavailable())
            self.pod_images, self.pod_per_map = max(len(post.pod_modes_exported()), 1), post.pod_combined()
            self.refresh_tags()
            self.refresh_rows(["pod_mode"])  # Its video possible or not
        self.set_range()

    def set_ptv(self, ptv):
        """The vector fields are those of a PTV (None: unknown): the quantities of the PTV greyed out if not."""
        self.ptv = ptv
        self.refresh_tags()

    def set_imaging(self, missing):
        """Why the quantities of the images and of their POD can't be exported ({key: reason, None if they can};
        None: unknown): those greyed out."""
        self.missing = self.reasons(missing)
        self.refresh_tags()

    @staticmethod
    def reasons(missing):
        return {key: reason for key, reason in (missing or {}).items() if reason}

    def unavailable(self, key):
        """Why a quantity can't be exported (None if it can): one of the PTV for vector fields that aren't those of
        a PTV, one of the images or of their POD not found; a group when none of its quantities can."""
        if self.ptv is False and (key in PTV_KEYS or key == PTV_GROUP_ROW):
            return "The quantities of the PTV: for the results of a PTV only"
        if key in self.groups:
            reasons = [self.unavailable(k) for k in self.groups[key]]
            return None if not all(reasons) else "; ".join(dict.fromkeys(reasons))
        return self.missing.get(key)

    def refresh_tags(self):
        """The quantities (and groups) that can't be exported greyed out; the count of the images exported."""
        for iid in list(FIGURE_TITLES) + list(self.groups):
            if self.tree.exists(iid):
                self.tree.item(iid, tags=("unavailable",) if self.unavailable(iid) else ())
        self.summary["text"] = self.count()

    def set_loader(self, loader, n_maps):
        """The post-processing given by loader() when a summary needs it (the maps: n_maps, None if unknown)."""
        self.post, self.loader, self.n_maps = None, loader, n_maps
        self.set_range()

    def set_range(self):
        for slider in (self.global_video.slider, self.video_box.slider, self.global_images.slider,
                       self.images_box.slider):
            slider.set_range(1, self.maps() if self.known() else max(self.maps(), slider.values[1]))

    def place(self, slider, first, last):
        """Maps on a slider (last None: the last one), its range widened to them if the maps are unknown."""
        if not self.known():
            slider.set_range(1, max(slider.hi, first, last or 1))
        slider.set(first, last or slider.hi)

    def read_maps(self, slider):
        """The first and the last maps of a slider (last None: the last one)."""
        first, last = slider.get()
        return first, None if last == slider.hi and (self.known() or last == 1) else last

    def per_map(self, key):
        """The quantity has one figure per map (its video, its instantaneous maps): instantaneous, or any with
        rolling statistics (not those of the POD, the same for all the maps, but the sum of the POD modes: the
        background of each image)."""
        if key == "pod_mode" and self.pod_per_map:
            return True
        return key in INSTANTANEOUS or (self.rolling and key not in STATIC)

    # ------------------------------------------------------------------ showing

    def show(self, spec, rolling=None, enabled=True):
        """Another specification (a copy is edited); rolling: that of its recipe (None: unchanged); enabled: it can
        be changed."""
        self.spec = export_spec(copy.deepcopy(spec))
        if rolling is not None:
            self.rolling = rolling
        self.enabled = enabled
        self.loading = True
        try:
            choice = self.spec["choice"]
            self.global_video.set(self, video_settings(choice))
            self.global_images.set(self, choice.get("map"), choice.get("map_last"), choice.get("map_step"))
            for box in (self.global_video, self.global_images):
                box.enable(enabled)
            for button in [self.reset_button, self.reset_quantity_button] + self.check_buttons:
                button.state(["!disabled"] if enabled else ["disabled"])
        finally:
            self.loading = False
        self.refresh_rows()
        self.note["text"] = ""
        if self.key is not None:
            self.select()
        else:
            self.describe_global()
            self.update_marks()

    def select(self):
        """A quantity selected: its parameters, its video and its instantaneous maps."""
        key = self.tree.focus()
        if key not in FIGURE_TITLES:
            return
        self.key, self.loading = key, True
        try:
            spec, params = self.spec, self.spec["params"]
            choice = spec["choice"]
            self.title_label["text"] = FIGURE_TITLES[key]
            clim = params["clims"].get(key)
            self.vmin.set("" if clim is None or clim[0] is None else num2str(clim[0]))
            self.vmax.set("" if clim is None or clim[1] is None else num2str(clim[1]))
            self.cmap.set(params["cmaps"].get(key, AUTO))
            self.contour.set("" if key not in params["contours"] else num2str(params["contours"][key]))
            compare = params["compares"].get(key)
            self.compare.set(FIGURE_TITLES[compare] if compare else NONE)
            self.wiper.set(params["wipers"].get(key, COMPARE_FRACTION))
            own = spec["videos"].get(key)
            self.video_box.override.set(own is not None)
            self.video_box.set(self, own or video_settings(choice))
            own = spec["images"].get(key)
            self.images_box.override.set(own is not None)
            if own is not None:
                self.images_box.set(self, own["first"], own["last"], own["step"])
            else:
                self.images_box.set(self, choice.get("map"), choice.get("map_last"), choice.get("map_step"))
            self.update_states()
            self.update_marks()
        finally:
            self.loading = False

    def video_on(self):
        """The quantity selected has its video exported (the summaries of the videos computed only then)."""
        return self.key is not None and self.key in self.spec["choice"]["videos"]

    def describe_global(self, moved=False):
        choice = self.spec["choice"]
        self.global_video.describe(self, self.video_on())
        self.global_images.describe(self, choice.get("map"), choice.get("map_last"), choice.get("map_step") or 1,
                                    moved)

    def update_states(self, moved=False):
        """The widgets of the quantity selected enabled or not, the summaries below the sliders."""
        key, enabled = self.key, self.enabled

        def state(on):
            return ["!disabled"] if on and enabled else ["disabled"]
        for w in self.map_widgets:
            w.state(state(key in MAP_FIGURES))
        self.wiper_scale.state(state(key in MAP_FIGURES and self.compare.get() != NONE))
        self.wiper_label["text"] = f"{self.wiper.get():.2f}"
        per_map = self.per_map(key)
        for box in (self.video_box, self.images_box):
            box.toggle.state(state(per_map))
            box.enable(per_map and box.override.get() and enabled)
        self.video_box.describe(self, self.video_on() and self.video_box.override.get())
        own = self.spec["images"].get(key)
        if own is not None:
            self.images_box.describe(self, own["first"], own["last"], own["step"], moved)
        else:
            choice = self.spec["choice"]
            self.images_box.describe(self, choice.get("map"), choice.get("map_last"), choice.get("map_step") or 1,
                                     False)
        self.describe_global()

    def refresh_rows(self, keys=None):
        """The rows of the quantities: their images and video (possible or not), the number of their parameters
        changed from the default; the count of the images and videos exported."""
        spec = self.spec
        choice = spec["choice"]
        for key in keys or FIGURE_TITLES:
            images = choice["figures"] is None or key in choice["figures"]
            video = key in choice["videos"]
            n = len(quantity_changes(spec, key))
            video_cell = (CHECKED if video else UNCHECKED) if self.per_map(key) else UNAVAILABLE
            self.tree.item(key, values=(CHECKED if images else UNCHECKED, video_cell, f"{n} different" if n else ""))
        # The groups: all of their quantities exported, none or some (their videos: of those that can have one)
        figures = set(FIGURE_TITLES if choice["figures"] is None else choice["figures"])

        def cell(keys, chosen):
            # Those that can be exported (all of them if none can)
            keys = [k for k in keys if not self.unavailable(k)] or keys
            n = sum(k in chosen for k in keys)
            return UNAVAILABLE if not keys else CHECKED if n == len(keys) else UNCHECKED if not n else PARTIAL
        for gid, keys in self.groups.items():
            self.tree.item(gid, values=(cell(keys, figures),
                                        cell([k for k in keys if self.per_map(k)], set(choice["videos"])), ""))
        self.summary["text"] = self.count()

    def count(self):
        """The images exported (each instantaneous map of the instantaneous figures: the global ones, or their own)
        and the videos."""
        choice, n = self.spec["choice"], self.maps()
        figures = list(FIGURE_TITLES) if choice["figures"] is None else choice["figures"]
        figures = [k for k in figures if not self.unavailable(k)]  # Not exported
        total = 0
        for key in figures:
            own = self.spec["images"].get(key)
            if key == "pod_mode" and not self.pod_per_map:
                total += self.pod_images
            elif not self.per_map(key):
                total += 1
            elif own is not None:
                total += len(image_maps(own["first"], own["last"], own["step"], n))
            elif choice.get("map"):
                total += len(image_maps(choice["map"], choice.get("map_last") or choice["map"],
                                        choice.get("map_step") or 1, n))
            else:
                total += 1
        videos = [k for k in choice["videos"] if self.per_map(k) and not self.unavailable(k)]
        return f"{total} images ({len(figures)} figures), {len(videos)} videos" + ("" if self.known() else
                                                                                   " (maps unknown)")

    # ------------------------------------------------------------------ changes

    def click(self, event):
        """A click in the column Images or Video: the images or the video of the quantity exported or not (the
        video only if possible); on a group, those of all of its quantities (group_click)."""
        column, key = self.tree.identify_column(event.x), self.tree.identify_row(event.y)
        reason = self.unavailable(key) if key else None
        if reason and column in ("#1", "#2"):
            self.bell()
            self.note["text"] = reason
            return "break"
        if key in self.groups and column in ("#1", "#2") and self.enabled:
            return self.group_click(key, column)
        if key not in FIGURE_TITLES or column not in ("#1", "#2") or not self.enabled:
            return
        choice = self.spec["choice"]
        if column == "#1":
            figures = list(FIGURE_TITLES) if choice["figures"] is None else list(choice["figures"])
            figures = [k for k in figures if k != key] if key in figures else figures + [key]
            choice["figures"] = (None if set(figures) == set(FIGURE_TITLES)
                                 else [k for k in FIGURE_TITLES if k in figures])
        else:
            if not self.per_map(key):
                self.bell()
                return
            videos = [k for k in choice["videos"] if k != key] + ([] if key in choice["videos"] else [key])
            choice["videos"] = [k for k in FIGURE_TITLES if k in videos]
        self.refresh_rows([key])
        if key == self.key:
            self.update_states()
            self.update_marks()
        self.report([])

    def check_all(self, on):
        """All the quantities exported (their images; the videos as they are), or none (nor their videos)."""
        if not self.enabled:
            return
        choice = self.spec["choice"]
        choice["figures"] = None if on else []
        if not on:
            choice["videos"] = []
        self.refresh_rows()
        if self.key is not None:
            self.update_states()
            self.update_marks()
        self.report([])

    def group_click(self, gid, column):
        """A click on a group in the column Images or Video: the images or the videos of all of its quantities (those
        that can have one) exported if they aren't all, none otherwise."""
        choice = self.spec["choice"]
        # Those that can be exported
        keys = [k for k in self.groups[gid] if not self.unavailable(k)]
        if column == "#1":
            chosen = set(FIGURE_TITLES if choice["figures"] is None else choice["figures"])
        else:
            keys, chosen = [k for k in keys if self.per_map(k)], set(choice["videos"])
            if not keys:
                self.bell()
                return "break"
        chosen = chosen - set(keys) if chosen >= set(keys) else chosen | set(keys)
        if column == "#1":
            choice["figures"] = None if chosen == set(FIGURE_TITLES) else [k for k in FIGURE_TITLES if k in chosen]
        else:
            choice["videos"] = [k for k in FIGURE_TITLES if k in chosen]
        self.refresh_rows(keys)
        if self.key in keys:
            self.update_states()
            self.update_marks()
        self.report([])
        return "break"

    def report(self, problems):
        """The problems of the values entered said; the specification given to on_change if there are none."""
        self.note["text"] = "Invalid: " + "; ".join(problems) if problems else ""
        if not problems and self.on_change is not None:
            self.on_change(copy.deepcopy(self.spec))

    def quantity_changed(self, moved=False):
        """A parameter of the quantity selected changed (its video, its instantaneous maps): kept in the
        specification."""
        if self.loading or self.key is None or not self.enabled:
            return
        key, spec, params = self.key, self.spec, self.spec["params"]
        problems = []
        if key in MAP_FIGURES:
            try:
                low = float(self.vmin.get()) if self.vmin.get().strip() else None
                high = float(self.vmax.get()) if self.vmax.get().strip() else None
                if low is None and high is None:
                    params["clims"].pop(key, None)
                else:
                    params["clims"][key] = [low, high]
            except ValueError:
                problems.append("colour range: numbers or nothing")
            if self.cmap.get() in COLORMAPS:
                params["cmaps"][key] = self.cmap.get()
            else:
                params["cmaps"].pop(key, None)
            try:
                if self.contour.get().strip():
                    params["contours"][key] = float(self.contour.get())
                else:
                    params["contours"].pop(key, None)
            except ValueError:
                problems.append("isocontour: a number or nothing")
            compare = next((k for k in MAP_FIGURES if FIGURE_TITLES[k] == self.compare.get()), None)
            if compare is None:
                params["compares"].pop(key, None)
            else:
                params["compares"][key] = compare
            # The wiper at its default position: none of its own
            wiper = round(float(self.wiper.get()), 3)
            if abs(wiper - COMPARE_FRACTION) > 1e-9:
                params["wipers"][key] = wiper
            else:
                params["wipers"].pop(key, None)
        per_map = self.per_map(key)
        if self.video_box.override.get() and per_map:
            try:
                spec["videos"][key] = self.video_box.read(self)
            except ValueError:
                problems.append("video: every N maps an integer >= 1, frames per second > 0")
        else:
            spec["videos"].pop(key, None)
        if self.images_box.override.get() and per_map:
            try:
                first, last, step = self.images_box.read()
                spec["images"][key] = image_settings({"first": first, "step": step,
                                                      "last": None if last == self.images_box.slider.hi
                                                      and self.known() else last})
            except ValueError:
                problems.append("instantaneous maps: every N maps an integer >= 1")
        else:
            spec["images"].pop(key, None)
        self.update_states(moved)
        self.refresh_rows([key])
        self.update_marks()
        self.report(problems)

    def global_changed(self, moved=False):
        """A global setting of the export changed (videos, instantaneous maps): kept in the specification."""
        if self.loading or not self.enabled:
            return
        problems = []
        choice = self.spec["choice"]
        try:
            video = self.global_video.read(self)
            choice.update(first=video["first"], last=video["last"], step=video["step"], fps=video["fps"],
                          format=video["format"])
        except ValueError:
            problems.append("videos: every N maps an integer >= 1, frames per second > 0")
        try:
            first, last, step = self.global_images.read()
            # The maps chosen once the slider is moved (before: the last map)
            if moved or choice.get("map"):
                choice.update(map=first, map_last=last, map_step=step, range=False)
            else:
                choice["map_step"] = step
        except ValueError:
            problems.append("instantaneous maps: every N maps an integer >= 1")
        self.global_images.describe(self, choice.get("map"), choice.get("map_last"), choice.get("map_step") or 1,
                                    moved)
        self.global_video.describe(self, self.video_on())
        self.refresh_rows()
        if self.key is not None and not moved:
            self.select()
        self.update_marks()
        self.report(problems)

    def reset_quantity(self):
        """Reset this quantity to defaults: its images exported, no video, no parameter, video settings nor
        instantaneous maps of its own."""
        key, spec = self.key, self.spec
        if key is None or not self.enabled:
            return
        choice = spec["choice"]
        if choice["figures"] is not None and key not in choice["figures"]:
            figures = choice["figures"] + [key]
            choice["figures"] = (None if set(figures) == set(FIGURE_TITLES)
                                 else [k for k in FIGURE_TITLES if k in figures])
        choice["videos"] = [k for k in choice["videos"] if k != key]
        for values in spec["params"].values():
            values.pop(key, None)
        spec["videos"].pop(key, None)
        spec["images"].pop(key, None)
        self.refresh_rows([key])
        self.select()
        self.report([])

    def reset(self):
        """Reset all quantities to the default (asked first): the default export (all the images, no video, no
        parameter, video settings nor instantaneous maps of their own, the default global settings)."""
        if not self.enabled:
            return
        if not messagebox.askyesno("Quantities", "Reset all the quantities to the default export? Their parameters, "
                                                 "their videos and the global settings are lost.",
                                   parent=self.winfo_toplevel()):
            return
        self.show(export_spec(None), None, self.enabled)
        self.report([])


class QuantitiesDialog(Dialog):
    """The window Quantities: the quantities of an export specification and their parameters (QuantitiesEditor). The
    result: the specification, None if cancelled."""

    def __init__(self, parent, spec, rolling=False, title="Quantities", post=None):
        super().__init__(parent, title)
        self.resizable(True, True)
        self.editor = QuantitiesEditor(self.frame, spec, rolling, post=post)
        self.editor.set_post(post)
        self.editor.show(self.editor.spec, rolling)
        self.editor.pack(fill=tk.BOTH, expand=True)
        buttons = ttk.Frame(self.frame)
        buttons.pack(anchor=tk.E, pady=(10, 0))
        ttk.Button(buttons, text="OK", command=self.ok).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side=tk.LEFT)

    def ok(self):
        if self.editor.note["text"]:
            self.bell()
            return
        self.result = self.editor.spec
        self.destroy()
