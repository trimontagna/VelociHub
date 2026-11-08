"""Page of the pre-processing: the POD-based background removal of the PIV images (port of the GUI of
background_removal.m), with its side panel (masks, contrast enhancement, POD modes, background removal and the
export folder) and the editor of the masks drawn on the image.

The background removal is an implementation of the algorithm described in the following research paper:
"POD-based background removal for Particle Image Velocimetry"
M.A. Mendez, M. Raiola, A. Masullo, S. Discetti, A. Ianiro, R. Theunissen, J.-M. Buchlin
Experimental Thermal and Fluid Science, Volume 80, January 2017, Pages 181-192.
For information, feedback and bug report, please visit the website:
http://seis.bris.ac.uk/~aexrt/PIVPODPreprocessing
This software is distributed under the CC BY-NC-SA 4.0 license by Creative Common."""

import json
import os
import tkinter as tk
from dataclasses import fields as dataclass_fields
from tkinter import filedialog, ttk
from types import SimpleNamespace

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Polygon as MplPolygon

from ..batch import masks_from_file
from ..preprocessing import (DEFAULT_CONTRAST, DEFAULT_EXPORT_FORMAT, DEFAULT_MODES, FILTERS, MODE_LABEL, PREPROCESSING_FOLDER,
                             ContrastSettings, Mask, PODBackgroundRemoval, available_memory, closed, describe_images,
                             draw_spectrum, imread, list_files, load_npz, mode_limits, mode_title, mround, num2str,
                             polygon_centroid, read_manifest, same_path, segment_distance, str2double)
from ..project import comparable_recipe, relative
from .batch_dialog import BatchStep, Field, PanelForm, PanelPage, run_batch
from .widgets import FigurePanel, FolderEntry, HelpDot, Page, PathBox, SidePanel, alive, bring_to_front, magic
from ..prefs import getpref, setpref
from ..settings import (preprocessing_from_project, preprocessing_to_project, validate_preprocessing)

IMAGE_FILES = [("Image files", "*.tif *.tiff *.bmp *.png *.jpg *.gif *.raw"), ("All files", "*.*")]
MASK_FILES = [("Masks", "*.json"), ("All files", "*.*")]
# Color of the image shown, marked on the temporal coefficient of the mode shown
IMAGE_MARK_COLOR = "#eb6834"


def describe_preprocessing(section):
    """The settings of the pre-processing of a batch, in a line."""
    masks = [m for m in section.get("masks") or [] if m.get("enabled", True)]
    if section.get("masks_file"):
        try:
            masks = [m for m in masks_from_file(section["masks_file"]) if m["enabled"]]
        except ValueError:
            masks = []
    known = {f.name for f in dataclass_fields(ContrastSettings)}
    contrast = ContrastSettings(**{k: v for k, v in (section.get("contrast") or {}).items() if k in known})
    bgrem = section.get("background_removal") or {}
    source = f" (of {os.path.basename(section['masks_file'])})" if section.get("masks_file") else ""
    return "; ".join([f"{len(masks)} mask(s) enabled{source}" if masks else f"no mask{source}",
                      f"contrast: {contrast.describe()}" if contrast.filters else "no contrast enhancement",
                      f"{bgrem.get('modes_removed')} POD modes removed" if bgrem.get("enabled")
                      else "no background removal",
                      f"exported to {section.get('export_folder') or 'PreProcessing'}"])


def preprocessing_fields(section):
    """The fields of the batch settings of the pre-processing (window Batch): the file of masks (a setting of the
    nodes, inherited as the others), the masks of the project (without file), the recipe, the export folder."""
    fields = [Field(("masks_file",), "Masks file (masks.json)", "file", MASK_FILES,
                    help=("Uses the masks of this file (saved by Export in the box Masks) instead of the masks of "
                          "each project. Set on a folder, it applies to all the projects below it. Empty: each "
                          "project keeps its own masks."))]
    if section.get("masks") and not section.get("masks_file"):
        fields.append(Field(("masks",), "Masks enabled", "masks"))
    fields += [Field(("background_removal", "enabled"), "Background removal", "bool", default=False),
               Field(("background_removal", "modes_removed"), "POD modes removed", "int", default=DEFAULT_MODES),
               Field(("contrast", "filters"), "Contrast enhancement: filters, in this order", "multi",
                     tuple((name, label) for name, label, _ in ControlPanel.FILTER_WIDGETS)),
               Field(("contrast", "stage"), "Contrast enhancement applied", "choice",
                     tuple(ControlPanel.STAGES.items()), default="post")]
    defaults = ContrastSettings()
    for name, label, params in ControlPanel.FILTER_WIDGETS:
        for param, text in params:
            kind = "int" if isinstance(getattr(defaults, param), int) else "float"
            fields.append(Field(("contrast", param), f"{label}: {text}", kind, help=ControlPanel.FILTER_HELP[name],
                                default=getattr(defaults, param)))
    fields.append(Field(("export_folder",), "Export Path (Relative)", "folder", default=PREPROCESSING_FOLDER,
                        help=("Folder of the images exported, relative to each project (↺: PreProcessing). Set "
                              "per project, not on folders."), project_only=True))
    fields.append(Field(("export_format",), "Format of the images exported", "choice",
                        tuple(EXPORT_FORMAT_LABELS.items()), help=EXPORT_FORMAT_HELP, default=DEFAULT_EXPORT_FORMAT))
    return fields


# Formats of the images exported (preprocessing.EXPORT_FORMATS), as the interface names them
EXPORT_FORMAT_LABELS = {"same": "Same as the source", "png": "PNG", "bmp": "BMP", "tiff": "TIFF"}
EXPORT_FORMAT_HELP = ("Format of the images exported. Same as the source: the format of each source image (a JPEG is "
                      "compressed again, with losses). PNG (8 or 16 bit), BMP (8 bit only) and TIFF (any depth, "
                      "uncompressed) are lossless. The images are in one channel (grey levels): a colour image is "
                      "read as its brightness, max(R, G, B).")


BATCH_MASKS_HINT = ("The masks of the project (drawn in the page Pre-processing): click ☑/☐ to enable or disable a "
                    "mask. With a file of masks (above), those of the file.")


class ControlPanel(SidePanel):
    """Side panel of the pre-processing page, in the order of the processing: masks, contrast enhancement, POD
    modes and background removal. The on/off buttons of the masks, of the contrast enhancement and of the background removal
    set the recipe, applied to the image shown and to the images exported with the button "Export". batch: the
    settings of the window Batch (BatchPreForm): the masks (enabled or not), the contrast enhancement and the
    background removal, without the images, the run step, the drawing of the masks nor the POD modes."""

    STAGES = {"pre": "before POD", "post": "after background removal"}
    # Filter, label and parameters, in the order they are applied by default
    FILTER_WIDGETS = [("capping", "Intensity capping", [("cap_n", "n (cap at median + n std)")]),
               ("minmax", "Local min-max normalization", [("mm_window", "window (px)"),
                                                           ("mm_contrast", "minimum contrast (grey levels)")]),
               ("clahe", "CLAHE", [("clip", "clip limit"), ("tiles", "tiles per side")]),
               ("edge", "Edge enhancement", [("edge_sigma", "low-pass σ (px)"), ("edge_factor", "factor")])]
    FILTER_HELP = {
        "capping": "The pixels brighter than median + n std of the image are set to that value, so that a few "
                   "bright particles don't dominate the correlation (Shavit et al. 2007).",
        "minmax": "(image − local min) / max(local max − local min, minimum contrast), the extremes over the window "
                  "smoothed over the window: evens out the brightness of the particles across the image.",
        "clahe": "Contrast Limited Adaptive Histogram Equalization over tiles of the image (8 or 16 bit images).",
        "edge": "Unsharp masking: image + factor × (image − low-pass), the low-pass being a Gaussian of standard "
                "deviation σ. Sharpens the particles (σ of about their size, 1 to 3 px) and lowers the slow "
                "variations of the background. Factor 0: no effect; 1 to 3: usual."}
    EXPORT_HELP = ("Singular exports the images of this project, processed with the recipe, to the subfolder "
                   "PreProcessing of the image folder (or the folder chosen) as "
                   "[project]_PreProcessing_Image_[number], with preprocessing.json (the recipe) and the POD "
                   "spectrum and modes (pod.npz). The PIV/PTV uses these images, exporting them first if needed. ... "
                   "picks another folder, ↺ goes back to PreProcessing. Stop ends the export; a partial export "
                   "isn't used.\n\nBatch does the same on several projects of the workspace, each with its own "
                   "settings.")
    HINTS = {None: "New: draw a polygon on the image. The pixels of the masks enabled (☑) are set to 0: click ☑/☐ "
                   "to enable or disable a mask, double click its name to rename it.",
             "draw": "Left click: add a point. Double click, Enter or click on the first point: close the polygon. "
                     "Right click: remove the last point or cancel. Esc: cancel.",
             "edit": "Drag the points to move them, the centroid ✚ to translate the mask and the round handle to "
                     "rotate and scale it (Shift: scale only, Ctrl: rotate only). Right click: add or remove a "
                     "point. Enter or Esc: done."}

    def __init__(self, master, page, batch=False):
        super().__init__(master)
        self.page, self.batch = page, batch
        # Prevents applying the contrast settings again while a question is asked about them
        self.busy = False
        wrap = self.wrap

        if not batch:
            # Tiles of the side panel: the images at the top, the export at the bottom, the settings in between
            footer = self.section("Run step", self.footer, self.EXPORT_HELP)
            self.export_folder = FolderEntry(footer, lambda: page.path_in if page.Check_imsel else None,
                                             PREPROCESSING_FOLDER, "Export folder of the pre-processed images...",
                                             on_change=page.export_folder_changed)
            self.export_folder.pack(fill=tk.X)
            row = ttk.Frame(footer)
            row.pack(fill=tk.X, pady=(6, 0))
            ttk.Label(row, text="Image format").pack(side=tk.LEFT)
            HelpDot(row, EXPORT_FORMAT_HELP).pack(side=tk.LEFT, padx=(4, 6))
            self.export_format = tk.StringVar(value=EXPORT_FORMAT_LABELS[DEFAULT_EXPORT_FORMAT])
            box = ttk.Combobox(row, textvariable=self.export_format, values=list(EXPORT_FORMAT_LABELS.values()),
                               state="readonly", width=18)
            box.pack(side=tk.LEFT)
            box.bind("<<ComboboxSelected>>", lambda _event: page.export_format_changed())
            row = ttk.Frame(footer)
            row.pack(anchor=tk.E, pady=(6, 0))
            ttk.Button(row, text="Batch...", command=page.batch).pack(side=tk.LEFT, padx=(0, 6))
            self.export_button = ttk.Button(row, text="Singular   (Ctrl+B)", style="Export.TButton",
                                            command=lambda: page.export("training"))
            self.export_button.pack(side=tk.LEFT)

            # Images
            box = self.section("Images", self.header)
            row = self.row(box, pady=(0, 2))
            ttk.Label(row, text="Images folder").pack(side=tk.LEFT)
            HelpDot(row, "Folder of the training images, all of them opened. Choose it with ... or type it in.").pack(side=tk.LEFT, padx=(4, 0))
            self.images = PathBox(box, "Folder of the training images...", page.open_folder, reset=False)
            self.images.pack(fill=tk.X)
            self.images_label = ttk.Label(box, wraplength=wrap, justify=tk.LEFT)
            self.images_label.pack(anchor=tk.W, pady=(4, 0))

        # Masks
        box = self.section("Masks")
        self.mask_tree = ttk.Treeview(box, columns=("on", "name", "points"), show="headings", height=4,
                                      selectmode="browse")
        for column, text, width, anchor in (("on", "On", 40, tk.CENTER), ("name", "Mask", 170, tk.W),
                                            ("points", "Points", 60, tk.CENTER)):
            self.mask_tree.heading(column, text=text)
            self.mask_tree.column(column, width=width, anchor=anchor, stretch=column == "name")
        self.mask_tree.pack(fill=tk.X)
        self.mask_tree.bind("<Button-1>", self.mask_clicked)
        self.mask_tree.bind("<Double-1>", self.mask_double_clicked)
        self.mask_tree.bind("<<TreeviewSelect>>", lambda event: self.update_mask_buttons())
        if not batch:
            row = self.row(box, pady=(6, 0))
            self.new_mask_button = ttk.Button(row, text="New", command=page.new_mask)
            self.edit_mask_button = ttk.Button(row, text="Edit", command=page.edit_mask)
            self.delete_mask_button = ttk.Button(row, text="Delete", command=page.delete_mask)
            for button in (self.new_mask_button, self.edit_mask_button, self.delete_mask_button):
                button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
            row = self.row(box, pady=(4, 0))
            self.import_mask_button = ttk.Button(row, text="Import...", command=page.load_masks_dialog)
            self.export_mask_button = ttk.Button(row, text="Export...", command=page.save_masks_dialog)
            for button in (self.import_mask_button, self.export_mask_button):
                button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
            HelpDot(row, ("Import replaces the masks with those of a .json file (saved by Export or by a script). "
                          "Export saves the masks to a file, to use them on other images. Points may be outside the "
                          "image: the masks then reach its edges.")).pack(side=tk.LEFT)
        self.mask_hint = ttk.Label(box, foreground="#555555", wraplength=wrap, justify=tk.LEFT)
        self.mask_hint.pack(anchor=tk.W, pady=(4, 0))

        # Contrast enhancement
        box = self.section("Contrast enhancement")
        ttk.Label(box, text="Filters, applied in this order (none selected: no contrast enhancement):",
                  wraplength=wrap, justify=tk.LEFT).pack(anchor=tk.W)
        grid = ttk.Frame(box)
        grid.pack(fill=tk.X)
        grid.columnconfigure(0, weight=1)
        self.checks = {}
        self.entries = {}
        r = 0
        for name, label, params in self.FILTER_WIDGETS:
            self.checks[name] = tk.BooleanVar(self)
            ttk.Checkbutton(grid, text=label, variable=self.checks[name], command=self.contrast_changed).grid(
                row=r, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
            HelpDot(grid, self.FILTER_HELP[name]).grid(row=r, column=2, padx=(4, 0), pady=(6, 0))
            r += 1
            for param, text in params:
                ttk.Label(grid, text=text).grid(row=r, column=0, sticky=tk.W, padx=(24, 4), pady=1)
                entry = ttk.Entry(grid, width=7)
                entry.grid(row=r, column=1, sticky=tk.E, pady=1)
                entry.bind("<Return>", self.contrast_changed if batch else self.focus_image)
                entry.bind("<FocusOut>", self.contrast_changed)
                self.entries[param] = entry
                r += 1
        self.cap_label = ttk.Label(box, foreground="#555555", wraplength=wrap, justify=tk.LEFT)
        self.cap_label.pack(anchor=tk.W, pady=(4, 0))
        row = self.row(box, pady=(6, 0))
        ttk.Label(row, text="Applied").pack(side=tk.LEFT)
        self.stage = tk.StringVar(self)
        for stage, label in self.STAGES.items():
            ttk.Radiobutton(row, text=label, value=stage, variable=self.stage,
                            command=self.contrast_changed).pack(side=tk.LEFT, padx=(8, 0))

        if not batch:
            # POD modes
            box = self.section("POD modes")
            self.spectrum_button = ttk.Button(box, text="Evaluate video spectrum   (Ctrl+R)",
                                              command=page.evaluate_spectrum)
            self.spectrum_button.pack(fill=tk.X)
            row = self.row(box, pady=(8, 2))
            ttk.Label(row, text="Automatic threshold   ε σ").pack(side=tk.LEFT)
            self.eps_sigma = tk.StringVar(self)
            self.eps_psi = tk.StringVar(self)
            for var, label in ((self.eps_sigma, None), (self.eps_psi, "ε ψ")):
                if label:
                    ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(6, 0))
                entry = ttk.Entry(row, textvariable=var, width=7)
                entry.pack(side=tk.LEFT, padx=(4, 0))
                entry.bind("<Return>", lambda event: page.set_automatic_modes())
            row = self.row(box)
            self.auto_button = ttk.Button(row, text="Evaluate threshold   (Ctrl+T)", command=page.set_automatic_modes)
            self.auto_button.pack(side=tk.LEFT)
            self.auto_label = ttk.Label(row)
            self.auto_label.pack(side=tk.LEFT, padx=(8, 0))
            row = self.row(box, pady=(8, 0))
            ttk.Label(row, text="Modes to evaluate").pack(side=tk.LEFT)
            self.n_modes = tk.StringVar(self)
            self.n_modes_entry = ttk.Entry(row, textvariable=self.n_modes, width=6)
            self.n_modes_entry.pack(side=tk.LEFT, padx=(6, 6))
            self.n_modes_entry.bind("<Return>", lambda event: page.evaluate_modes())
            self.modes_button = ttk.Button(row, text="Evaluate modes   (Ctrl+M)", command=page.evaluate_modes)
            self.modes_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Background removal
        box = self.section("Background removal")
        self.bgrem_toggle = self.toggle(box, "Remove the background   (Ctrl+P)", page.use_bgrem, page.toggle_bgrem)
        row = self.row(box)
        ttk.Label(row, text="Modes removed").pack(side=tk.LEFT)
        self.modes_removed = tk.StringVar(self, "0")
        self.modes_spinbox = ttk.Spinbox(row, textvariable=self.modes_removed, from_=0, to=0, width=6,
                                         command=page.set_modes_removed)
        self.modes_spinbox.pack(side=tk.LEFT, padx=6)
        self.modes_spinbox.bind("<Return>", (lambda event: page.set_modes_removed()) if batch else self.focus_image)
        self.modes_spinbox.bind("<FocusOut>", lambda event: page.set_modes_removed())
        self.modes_label = ttk.Label(row)
        self.modes_label.pack(side=tk.LEFT)

        self.show_contrast_settings(page.contrast)

    def focus_image(self, event=None):
        """Give the focus to the image (the entry left applies its value), so that the arrows browse the images."""
        self.page.figure_main.canvas.get_tk_widget().focus_set()

    def update_state(self):
        """Enable the buttons that can be used at this stage of the analysis."""
        page = self.page
        self.update_images()
        for widget in (self.spectrum_button, self.auto_button, self.modes_button, self.bgrem_toggle,
                       self.modes_spinbox):
            widget.configure(state=tk.NORMAL if page.Check_imsel else tk.DISABLED)
        # The background removal can be in the recipe before the modes are evaluated: they are evaluated by the
        # export or the PIV
        self.modes_spinbox.configure(to=max(page.NImg, 1))
        self.modes_label["text"] = (f"{page.NMod_saved} evaluated" if page.Check_modproc
                                    else "(evaluated when needed)")
        self.update_mask_buttons()
        self.update_recipe()

    def update_images(self):
        """Folder of the images, their number, size and bit depth."""
        page = self.page
        self.images.show(page.path_in if page.Check_imsel else "")
        self.images_label["text"] = (describe_images(page.images) if page.Check_imsel
                                     else "No images: choose their folder")

    def update_recipe(self):
        """The button "Export" is enabled when the recipe has something to export."""
        page = self.page
        steps = page.recipe_steps(page.n_modes_removed())
        state = tk.NORMAL if steps and page.Check_imsel else tk.DISABLED
        self.export_button.configure(state=state)
        self.update_export()

    def update_export(self):
        """Export folder."""
        self.export_folder.refresh()

    # ------------------------------------------------------------------ masks

    def update_masks(self):
        """List of the masks."""
        selected = self.selected_mask()
        self.mask_tree.delete(*self.mask_tree.get_children())
        for i, mask in enumerate(self.page.masks):
            self.mask_tree.insert("", tk.END, iid=str(i),
                                  values=("☑" if mask.enabled else "☐", mask.name, len(mask.points)))
        if selected is not None and selected < len(self.page.masks):
            self.select_mask(selected)
        self.update_mask_buttons()

    def selected_mask(self):
        selection = self.mask_tree.selection()
        return int(selection[0]) if selection else None

    def select_mask(self, i):
        self.mask_tree.selection_set(str(i))
        self.mask_tree.see(str(i))

    def update_mask_buttons(self):
        if self.batch:  # No buttons: the masks are drawn in the page
            self.mask_hint["text"] = BATCH_MASKS_HINT
            return
        mode = self.page.mask_editor.mode
        selected = self.selected_mask() is not None
        states = {self.new_mask_button: self.page.Check_imsel and mode is None,
                  self.edit_mask_button: mode == "edit" or (mode is None and selected and self.page.Check_imsel),
                  self.delete_mask_button: mode is None and selected}
        for button, enabled in states.items():
            button.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        self.edit_mask_button["text"] = "Done" if mode == "edit" else "Edit"
        self.mask_hint["text"] = self.HINTS[mode]

    def mask_clicked(self, event):
        """A click on ☑/☐ enables or disables the mask."""
        row = self.mask_tree.identify_row(event.y)
        if row and self.mask_tree.identify_column(event.x) == "#1":
            self.page.toggle_mask(int(row))

    def mask_double_clicked(self, event):
        row = self.mask_tree.identify_row(event.y)
        if row and self.mask_tree.identify_column(event.x) == "#2":
            self.page.rename_mask(int(row))

    # ------------------------------------------------------------------ contrast enhancement

    def show_contrast_settings(self, settings):
        self.stage.set(settings.stage)
        for name, var in self.checks.items():
            var.set(name in settings.filters)
        for param, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, num2str(getattr(settings, param)))

    def read_contrast(self):
        """Contrast settings entered, None if they are invalid. The filters are applied in the order of the panel,
        unless the same filters are selected: their order, which a script may have changed, is kept."""
        v = {param: str2double(entry.get()) for param, entry in self.entries.items()}
        checked = {name for name, var in self.checks.items() if var.get()}
        current = self.page.contrast.filters
        filters = current if checked == set(current) else [f for f in FILTERS if f in checked]
        try:
            return ContrastSettings(filters, self.stage.get(), **v)
        except ValueError:
            return None

    def contrast_changed(self, event=None):
        """Apply the contrast settings entered. Invalid or refused settings are replaced by the current ones."""
        if self.busy:
            return
        self.busy = True
        try:
            new = self.read_contrast()
            if new is None:
                self.bell()
            if new is None or not self.page.apply_contrast(new):
                self.show_contrast_settings(self.page.contrast)
        finally:
            self.busy = False


class PrePage(PanelPage):
    """What the side panel of the pre-processing tells its page, in the window Batch: the recipe of the node (its
    masks, contrast enhancement and background removal), each change to the form."""

    def __init__(self, form, master):
        super().__init__(form)
        self.contrast, self.masks, self.use_bgrem = ContrastSettings(), [], tk.BooleanVar(master, False)
        self.mask_editor = SimpleNamespace(mode=None)

    def apply_contrast(self, new):
        self.contrast = new
        return self.commit()

    def toggle_bgrem(self):
        self.commit()

    def set_modes_removed(self):
        self.commit()

    def toggle_mask(self, i):
        if not self.form.masks_file:
            self.masks[i].enabled = not self.masks[i].enabled
            self.form.panel.update_masks()
            self.commit()


class BatchPreForm(PanelForm):
    """The settings of the pre-processing of the node selected in the window Batch (PanelForm): those of the side panel
    of the page (ControlPanel in batch mode: masks, contrast enhancement, background removal), the file of masks and
    the export folder above it."""

    PATHS = ("masks", "contrast", "background_removal")
    # Each filter compared on its own (on in some projects, off in others)
    CHOICES = {label: (("contrast", "filters"), name) for name, label, _ in ControlPanel.FILTER_WIDGETS}
    LABELS = {**{text: [("contrast", param)] for _, _, params in ControlPanel.FILTER_WIDGETS for param, text in params},
              "Applied": [("contrast", "stage")], "Remove the background   (Ctrl+P)": [("background_removal", "enabled")],
              "Modes removed": [("background_removal", "modes_removed")]}

    def make_panel(self):
        self.masks_file = None
        page = PrePage(self, self)
        return ControlPanel(self, page, batch=True)

    def load_panel(self, settings, unavailable=None, limits=None):
        panel, page = self.panel, self.panel.page
        known = {f.name for f in dataclass_fields(ContrastSettings)}
        contrast = {k: v for k, v in (settings.get("contrast") or {}).items() if k in known}
        try:
            page.contrast = ContrastSettings(**{**contrast, "filters": tuple(contrast.get("filters") or ())})
        except (TypeError, ValueError):
            page.contrast = ContrastSettings()
        panel.show_contrast_settings(page.contrast)
        self.masks_file = settings.get("masks_file") or None
        try:
            masks = masks_from_file(self.masks_file) if self.masks_file else settings.get("masks") or []
        except ValueError:
            masks = []
        page.masks = [SimpleNamespace(name=m.get("name", "Mask"), enabled=bool(m.get("enabled", True)),
                                      points=list(m.get("points") or [])) for m in masks]
        panel.update_masks()
        bgrem = settings.get("background_removal") or {}
        page.use_bgrem.set(bool(bgrem.get("enabled")))
        panel.bgrem_toggle.configure(variable=page.use_bgrem)
        panel.modes_spinbox.configure(to=10000)
        panel.modes_removed.set(str(bgrem.get("modes_removed", DEFAULT_MODES)))

    def restore_states(self, unavailable):
        # The masks of a file: shown, not changed here
        self.panel.mask_tree.state(["disabled"] if self.masks_file else ["!disabled"])

    def record(self):
        """The recipe entered in the panel, as batch settings: ValueError if invalid."""
        panel, page = self.panel, self.panel.page
        contrast = panel.read_contrast()
        if contrast is None:
            raise ValueError("Contrast enhancement: invalid settings (numbers expected)")
        try:
            modes = int(float(panel.modes_removed.get()))
        except ValueError:
            raise ValueError("POD modes removed: an integer") from None
        out = {"contrast": {"filters": list(contrast.filters), "stage": contrast.stage,
                            **{p: getattr(contrast, p) for _, _, params in ControlPanel.FILTER_WIDGETS
                               for p, _ in params}},
               "background_removal": {"enabled": bool(page.use_bgrem.get()), "modes_removed": modes}}
        if not self.masks_file:
            out["masks"] = [{"name": m.name, "enabled": m.enabled, "points": m.points} for m in page.masks]
        return out

    @staticmethod
    def normalize(field, value):
        if field.path == ("masks",):
            return [(m.get("name"), bool(m.get("enabled", True)), json.dumps(m.get("points"))) for m in value or []]
        if field.path == ("contrast", "filters"):
            return list(value or [])
        if field.path == ("background_removal", "modes_removed"):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return value
        return value


class MaskEditor:
    """Drawing and editing of the polygonal masks on the image panel. A new mask is drawn point by point. A mask
    is edited by dragging its points, its centroid (translation) and a handle (rotation and scaling); points are
    added and removed with the right click. The polygon drawn or edited is redrawn by blitting, the image isn't.
    While a mask is drawn or edited, the view shows a margin around the image (grey), where the points can be
    placed so that the mask surely reaches the edges of the image; they may also be placed beyond the axes."""

    # Distance in screen pixels to pick a point
    PICK = 9
    # Margin shown around the image while a mask is drawn or edited (fraction of its largest side)
    MARGIN = 0.08
    COLOR = "#00e5ff"
    EDIT_COLOR = "#ffcc00"

    def __init__(self, page, panel):
        self.page = page
        self.panel = panel
        self.mode = None  # None, "draw" or "edit"
        self.points = []  # Points of the polygon being drawn
        self.cursor = None  # Position of the mouse while drawing
        self.mask = None  # Mask edited
        self.angle = 0.0  # Direction of the handle from the centroid
        self.drag = None  # What is dragged, with the state at the start of the drag
        self.static = []  # Outlines of the masks
        self.dynamic = {}  # Polygon drawn or edited
        self.background = None
        self.widened = None  # Limits of the view widened around the image
        for event, callback in (("button_press_event", self.on_press), ("motion_notify_event", self.on_motion),
                                ("button_release_event", self.on_release), ("key_press_event", self.on_key),
                                ("draw_event", self.on_draw)):
            panel.canvas.mpl_connect(event, callback)

    # ------------------------------------------------------------------ modes

    def start_draw(self):
        self.mode, self.mask, self.points, self.cursor = "draw", None, [], None
        self.widen()
        self.refresh()

    def start_edit(self, mask):
        self.mode, self.mask, self.angle = "edit", mask, 0.0
        self.widen()
        self.refresh()

    def stop(self):
        self.mode, self.mask, self.points, self.drag = None, None, [], None
        self.narrow()
        self.refresh()

    def view(self):
        ax = self.panel.ax
        return tuple(ax.get_xlim()) + tuple(ax.get_ylim())

    def widen(self):
        """Show a margin around the image, if the whole image is shown."""
        panel = self.panel
        if panel.ax is None or panel.im is None or self.widened is not None:
            return
        x0, x1, y1, y0 = panel.im.get_extent()
        if not np.allclose(self.view(), (x0, x1, y1, y0)):
            return
        m = self.MARGIN * max(x1 - x0, y1 - y0)
        panel.ax.set_xlim(x0 - m, x1 + m)
        panel.ax.set_ylim(y1 + m, y0 - m)
        self.widened = self.view()
        panel.canvas.draw_idle()

    def narrow(self):
        """Back to the whole image, unless the view was zoomed or panned."""
        panel = self.panel
        widened, self.widened = self.widened, None
        if widened is None or panel.ax is None or panel.im is None or not np.allclose(self.view(), widened):
            return
        x0, x1, y1, y0 = panel.im.get_extent()
        panel.ax.set_xlim(x0, x1)
        panel.ax.set_ylim(y1, y0)
        panel.canvas.draw_idle()

    # ------------------------------------------------------------------ drawing

    def refresh(self):
        """Draw the outlines of the masks and the polygon drawn or edited over the image shown."""
        ax = self.panel.ax
        for artist in self.static + list(self.dynamic.values()):
            if artist.axes is ax and ax is not None:
                artist.remove()
        self.static, self.dynamic = [], {}
        if ax is None or not self.page.Check_imsel:
            return
        for mask in self.page.masks:
            if mask is self.mask:
                continue
            style = (dict(color=self.COLOR, linewidth=1.2) if mask.enabled
                     else dict(color="0.6", linewidth=1, linestyle="--"))
            P = closed(mask.points)
            self.static += ax.plot(P[:, 0], P[:, 1], **style)
            c = polygon_centroid(mask.points)
            self.static.append(ax.text(c[0], c[1], mask.name, color=style["color"], fontsize=8, ha="center",
                                       va="center", clip_on=True))
        color = self.EDIT_COLOR
        if self.mode == "draw":
            self.dynamic["line"], = ax.plot([], [], "o-", color=color, markersize=4, linewidth=1.2, animated=True)
            self.dynamic["rubber"], = ax.plot([], [], "--", color=color, linewidth=1, animated=True)
        elif self.mode == "edit":
            self.dynamic["fill"] = ax.add_patch(MplPolygon(self.mask.points, closed=True, facecolor=color,
                                                           alpha=0.25, edgecolor="none", animated=True))
            self.dynamic["line"], = ax.plot([], [], "o-", color=color, markersize=5, markerfacecolor="white",
                                            linewidth=1.5, animated=True)
            self.dynamic["arm"], = ax.plot([], [], ":", color=color, linewidth=1.2, animated=True)
            self.dynamic["centroid"], = ax.plot([], [], "P", color=color, markersize=13, markeredgecolor="black",
                                                animated=True)
            self.dynamic["handle"], = ax.plot([], [], "o", color=color, markersize=10, markeredgecolor="black",
                                              animated=True)
        self.update_dynamic(blit=False)
        self.panel.canvas.draw_idle()

    def update_dynamic(self, blit=True):
        """Update the polygon drawn or edited, and redraw it alone over the background saved."""
        d = self.dynamic
        if self.mode == "draw" and d:
            P = np.array(self.points).reshape(-1, 2)
            d["line"].set_data(P[:, 0], P[:, 1])
            rubber = [P[-1], self.cursor] if len(P) and self.cursor is not None else np.empty((0, 2))
            d["rubber"].set_data(*np.transpose(rubber))
        elif self.mode == "edit" and d:
            P = self.mask.points
            c, h = polygon_centroid(P), self.handle_position()
            d["fill"].set_xy(P)
            d["line"].set_data(*closed(P).T)
            d["arm"].set_data([c[0], h[0]], [c[1], h[1]])
            d["centroid"].set_data([c[0]], [c[1]])
            d["handle"].set_data([h[0]], [h[1]])
        canvas = self.panel.canvas
        if blit and self.background is not None:
            canvas.restore_region(self.background)
            for artist in d.values():
                self.panel.ax.draw_artist(artist)
            canvas.blit(self.panel.fig.bbox)
        elif blit:
            canvas.draw_idle()

    def on_draw(self, event):
        """Save the figure without the polygon drawn or edited, and draw it."""
        self.background = self.panel.canvas.copy_from_bbox(self.panel.fig.bbox)
        for artist in self.dynamic.values():
            if artist.axes is self.panel.ax:
                self.panel.ax.draw_artist(artist)

    def handle_position(self):
        """Handle of the rotation and of the scaling, out of the polygon in the direction self.angle."""
        P = self.mask.points
        c = polygon_centroid(P)
        r = max(1.25 * np.hypot(*(P - c).T).max(), 5)
        return c + r * np.array([np.cos(self.angle), np.sin(self.angle)])

    # ------------------------------------------------------------------ mouse and keys

    def active(self, event):
        """The event is for the editor: drawing or editing on the image or around it (also beyond the axes, but
        not on the colorbar), the zoom and pan tools are off."""
        return (self.mode is not None and self.panel.ax is not None and event.x is not None
                and event.inaxes in (self.panel.ax, None) and self.panel.toolbar.mode == "")

    def position(self, event):
        """Position of the mouse in the pixels of the image, also beyond the axes."""
        if event.inaxes is self.panel.ax and event.xdata is not None:
            return np.array([event.xdata, event.ydata])
        return self.panel.ax.transData.inverted().transform((event.x, event.y))

    def near(self, p, event):
        return np.hypot(*(self.panel.ax.transData.transform(p) - (event.x, event.y))) <= self.PICK

    def nearest_point(self, event):
        """Index of the point of the mask edited under the mouse, None if there is none."""
        dist = np.hypot(*(self.panel.ax.transData.transform(self.mask.points) - (event.x, event.y)).T)
        i = int(np.argmin(dist))
        return i if dist[i] <= self.PICK else None

    def on_press(self, event):
        if not self.active(event):
            return
        self.panel.canvas.get_tk_widget().focus_set()
        xy = self.position(event)
        if self.mode == "draw":
            if event.button == MouseButton.LEFT:
                if event.dblclick or (len(self.points) >= 3 and self.near(self.points[0], event)):
                    self.close_polygon()
                else:
                    self.points.append(xy)
                    self.update_dynamic()
            elif event.button == MouseButton.RIGHT:
                self.popup(event, [("Remove the last point", self.remove_last, bool(self.points)),
                                   ("Close the polygon", self.close_polygon, len(self.points) >= 3),
                                   ("Cancel", self.page.cancel_mask, True)])
            return

        P = self.mask.points
        if event.button == MouseButton.LEFT:
            if self.near(self.handle_position(), event):
                kind = "handle"
            elif self.near(polygon_centroid(P), event):
                kind = "centroid"
            else:
                i = self.nearest_point(event)
                kind = None if i is None else i
            if kind is not None:
                self.drag = SimpleNamespace(kind=kind, start=xy, points=P.copy(), center=polygon_centroid(P),
                                            angle=self.angle)
        elif event.button == MouseButton.RIGHT:
            i = self.nearest_point(event)
            self.popup(event, [("Add a point here", lambda: self.add_point(xy), True),
                               ("Remove this point", lambda: self.remove_point(i), i is not None and len(P) > 3),
                               ("Done editing", self.page.finish_mask, True)])

    def on_motion(self, event):
        if self.mode == "draw":
            self.cursor = self.position(event) if self.active(event) else None
            self.update_dynamic()
        elif self.mode == "edit" and self.drag is not None and event.x is not None:
            d = self.drag
            xy = self.position(event)
            if d.kind == "centroid":
                self.mask.points = d.points + (xy - d.start)
            elif d.kind == "handle":
                modifiers = getattr(event, "modifiers", None) or ()
                v0, v = d.start - d.center, xy - d.center
                theta = 0.0 if "shift" in modifiers else np.arctan2(v[1], v[0]) - np.arctan2(v0[1], v0[0])
                scale = 1.0 if "ctrl" in modifiers else max(np.hypot(*v) / max(np.hypot(*v0), 1e-9), 1e-3)
                R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                self.mask.points = d.center + scale * (d.points - d.center) @ R.T
                self.angle = d.angle + theta
            else:
                P = d.points.copy()
                P[d.kind] = xy
                self.mask.points = P
            self.update_dynamic()

    def on_release(self, event):
        if self.drag is not None:
            self.drag = None
            self.page.mask_edited()

    def on_key(self, event):
        if self.mode == "draw":
            if event.key == "enter":
                self.close_polygon()
            elif event.key == "escape":
                self.page.cancel_mask()
            elif event.key == "backspace":
                self.remove_last()
        elif self.mode == "edit" and event.key in ("enter", "escape"):
            self.page.finish_mask()

    def popup(self, event, items):
        """Right click menu: items (label, command, enabled)."""
        menu = tk.Menu(self.panel.canvas.get_tk_widget(), tearoff=0)
        for label, command, enabled in items:
            menu.add_command(label=label, command=command, state=tk.NORMAL if enabled else tk.DISABLED)
        if event.guiEvent is not None:
            menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)

    # ------------------------------------------------------------------ polygon changes

    def remove_last(self):
        if self.points:
            self.points.pop()
            self.update_dynamic()

    def close_polygon(self):
        if len(self.points) < 3:
            self.panel.bell()
            return
        points = np.array(self.points, dtype=float)
        self.points = []
        self.page.add_mask(points)

    def add_point(self, xy):
        """Insert a point in the closest side of the polygon."""
        P = self.mask.points
        sides = [segment_distance(xy, P[i], P[(i + 1) % len(P)]) for i in range(len(P))]
        self.mask.points = np.insert(P, int(np.argmin(sides)) + 1, xy, axis=0)
        self.page.mask_edited()

    def remove_point(self, i):
        if i is not None and len(self.mask.points) > 3:
            self.mask.points = np.delete(self.mask.points, i, axis=0)
            self.page.mask_edited()


class PreProcessingPage(Page, PODBackgroundRemoval):
    """Page of the pre-processing: the POD-based background removal of the PIV images, a port of the GUI of
    background_removal.m, where the MATLAB figures, menus and dialogs are Tkinter widgets with embedded Matplotlib
    figures tiled in the page (the image and the modes on top, the spectrum below, the side panel on the right).
    Its nested functions are the methods of this class, the processing is the one of PODBackgroundRemoval, which the
    page extends, and the fields of the MATLAB structure "hand" are its attributes.

    It gives the images to the window (image_folder, image_files): the other pages use them, processed with the
    recipe set here (masks, contrast enhancement and background removal)."""

    key = "pre"
    title = "Pre-processing"
    project_key = "preprocessing"

    def __init__(self, master, app):
        Page.__init__(self, master, app)
        # Initialize global variables: the temporary files are stored in the folder of the images, as in the
        # original tool, unless the user selects another folder
        PODBackgroundRemoval.__init__(self, temp_folder="images", verbose=False)
        self.pathtemp = self.path_in = getpref("last_folder", os.getcwd())

        # Steps of the recipe applied to the image shown and to the images exported: the masks enabled, the filters
        # of the contrast enhancement selected, and the background removal if its button is on (its modes are
        # evaluated when they are needed: by the export or the PIV)
        self.masks_enabled = self.contrast_enabled = True
        self.contrast = DEFAULT_CONTRAST
        self.use_bgrem = tk.BooleanVar(self, True)
        # File of the spectrum and of the modes saved, and image shown marked on the temporal coefficient
        self.pod_file = None
        self.image_marker = None

        # Everything is tiled: the image and the modes on top, the spectrum below, and the side panel on the right
        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        left = ttk.PanedWindow(body, orient=tk.VERTICAL)
        body.add(left, weight=1)
        self.panel = ControlPanel(body, self)
        body.add(self.panel, weight=0)
        images = ttk.PanedWindow(left, orient=tk.HORIZONTAL)
        left.add(images, weight=4)
        # Comparison before / after the processing: fraction of the width of the image shown before, and the
        # last image read before the processing
        self.compare_fraction = 1.0
        self.before_cache = (None, None)
        self.figure_main = FigurePanel(images, self.image_tools("main"), "compressed", figsize=(6, 4),
                                       clabel="Intensity (grey levels)", item="Image",
                                       on_slide=lambda i: self.callback_arrows(0, "main", i),
                                       on_activate=lambda: self.set_active_panel("main"),
                                       on_compare=self.set_compare)
        images.add(self.figure_main, weight=1)
        self.figure_modes = FigurePanel(images, self.image_tools("modes"), "compressed", figsize=(3.5, 4),
                                        clabel=MODE_LABEL, item="Mode",
                                        on_slide=lambda i: self.callback_arrows(0, "modes", i),
                                        on_activate=lambda: self.set_active_panel("modes"))
        images.add(self.figure_modes, weight=1)
        self.mask_editor = MaskEditor(self, self.figure_main)
        self.figure_spectrum = FigurePanel(left, figsize=(5, 2), on_activate=lambda: self.set_active_panel("spectrum"))
        left.add(self.figure_spectrum, weight=1)
        self.set_active_panel("main")

    def menus(self):
        return [
            ("Settings", [
                ("Set temporary folder...", None, self.change_temp_fold),
                ("Max memory usage...", None, self.set_memory),
                ("Memory info", None, self.memory_info),
                ("Temporary files", None, self.temp_files_info)]),
            ("Pre-processing", [
                ("Load images in memory", "Ctrl+L", self.load_images_memory),
                ("Evaluate video spectrum", "Ctrl+R", self.evaluate_spectrum),
                ("Evaluate automatic threshold", "Ctrl+T", self.set_automatic_modes),
                ("Evaluate modes", "Ctrl+M", self.evaluate_modes),
                None,
                ("Background removal on/off", "Ctrl+P", lambda: self.toggle_bgrem(flip=True)),
                None,
                ("Export", [
                    ("Training images", "Ctrl+B", lambda: self.export("training")),
                    ("New images", [
                        ("Pick multiple files...", None, lambda: self.export("new_target_direct")),
                        ("Pick matching pattern...", None, lambda: self.export("new_target_pattern"))])]),
                None,
                ("Clear the cache", None, self.clear_cache)])]

    def folders(self):
        return [("Temporary folder", lambda: self.pathtemp), ("Image folder", lambda: self.path_in),
                ("Pre-processing folder", self.export_folder)]

    def start(self):
        self.initialize_session()
        self.update_status(f"Temporary folder: {self.pathtemp}")
        self.update_status(f"Image folder: {self.path_in}")
        self.update_status('Please choose the folder of the images to analyse (side panel, "Images")')

    # Images given to the other pages

    def image_folder(self):
        """Folder of the images, None if none are selected."""
        return self.path_in if self.Check_imsel else None

    def image_files(self):
        return list(self.file_in) if self.Check_imsel else []

    # Hooks of PODBackgroundRemoval: messages and progress in the window

    def update_status(self, msg):
        self.app.update_status(msg)

    def progress(self, title):
        """The progress of the pre-processing, with its button Stop (Stopped: the process ends there)."""
        return self.app.progress(title, stoppable=True)

    # Batch

    def batch_settings(self):
        """The settings of the pre-processing of the interface, as the batch applies them (batch.apply_preprocessing):
        those of the section of the project (the export folder relative to the image folder)."""
        root = self.path_in if self.Check_imsel else None
        state = self.project_state(lambda path: relative(path, root) if root else None)
        return {key: state[key] for key in ("masks", "contrast", "background_removal", "export_folder",
                                            "export_format")}

    def batch_step(self):
        return BatchStep("preprocessing", "Pre-processing", lambda project: project.images, self.batch_settings,
                         describe_preprocessing, lambda project, settings, resolve: {"preprocessing": settings},
                         fields=preprocessing_fields, validate=validate_preprocessing,
                         to_project=("preprocessing", preprocessing_to_project),
                         from_project=preprocessing_from_project, form=BatchPreForm)

    def batch(self):
        """Run step > Batch...: the pre-processing of projects of the workspace."""
        run_batch(self.app, self.batch_step())

    ############################################################################
    def open_images(self, folder=None, pattern=None, files=None, first=1, last=None, restore=True):
        """Training images selected (set_images) and shown, with their project: restore=True: the project of their
        folder, if there is one; a project (dictionary): this one; False: none (a new project)."""
        self.initialize_session()
        try:
            self.set_images(folder, pattern, files, first, last)
        except ValueError as me:
            self.app.status_error(str(me))
            return
        self.panel.update_state()

        # Show the first image
        self.callback_arrows(0, "main")

        # Update status
        self.update_status(-1)
        self.update_status(f"Temporary folder: {self.pathtemp}")
        self.update_status(f"Image folder: {self.path_in}")

        # Set preference on last folder used
        setpref("last_folder", self.path_in)

        # Update status
        self.update_status(f"{self.NImg} images selected")
        self.update_status("The slider or the arrow keys browse the images (mouse over the image)")
        self.update_status("Please inspect your images or evaluate the video spectrum from the side panel")
        self.app.images_opened(restore)

    def initialize_session(self):
        """This function initializes the global variable to a new session."""
        # The changes of the project of the previous images are saved first
        self.app.flush_project()
        self.pod_file = None
        self.reset_session()
        self.showing_im_i = 1
        self.showing_mod_i = 1
        self.im_caxis = SimpleNamespace(active=False, val=None)
        self.phi_caxis = SimpleNamespace(active=False, val=None)

        # Side panel and figures of the new session
        self.panel.eps_sigma.set(num2str(self.eps_auto_sigma))
        self.panel.eps_psi.set(num2str(self.eps_auto_psi))
        self.panel.n_modes.set(str(DEFAULT_MODES))
        self.panel.modes_removed.set(str(DEFAULT_MODES))
        self.panel.cap_label["text"] = ""
        self.mask_editor.stop()
        self.panel.update_masks()
        self.reset_pod_figures()
        # The other pages without the images of the previous session
        self.app.images_changed()

        # Random image
        self.show_initial_image()

    def shown(self):
        """The page is shown: the image looked at in the other pages, the modes evaluated and the images exported
        by the PIV."""
        if self.Check_imsel and self.app.image_index + 1 != self.showing_im_i:
            self.callback_arrows(0, "main", self.app.image_index + 1)
        self.panel.update_state()

    def arrow(self, step):
        self.callback_arrows(step, self.active_panel)

    def export_format_changed(self):
        """Another format of the images exported: the images exported in it, and the pages using them, follow."""
        labels = {label: key for key, label in EXPORT_FORMAT_LABELS.items()}
        self.export_format = labels.get(self.panel.export_format.get(), DEFAULT_EXPORT_FORMAT)
        self.panel.update_export()
        self.app.save_project()

    def export_folder_changed(self):
        """Another export folder: the images exported in it, and the pages using them, follow."""
        self.path_export = self.panel.export_folder.get()
        self.panel.update_export()
        self.app.save_project()

    def on_exit(self):
        # Delete temporary files
        self.delete_temporary_files(True)
        return True

    def temp_files_info(self):
        """Settings > Temporary files: why the background removal may write temporary files."""
        self.app.msgbox("The background removal tool requires the entire set of images to be loaded into memory. "
                        "If the available memory is insufficient for the operation, images will be processed "
                        "through some temporary files that will be stored on the hard drive. The process of "
                        "saving/loading temporary files can be time consuming but it allows the user to process "
                        "any set of images independently from the memory available.", self.app.title)

    ############################################################################
    # Project

    def project_state(self, rel):
        c = self.contrast
        contrast = {f.name: getattr(c, f.name) for f in dataclass_fields(ContrastSettings)}
        contrast["filters"] = list(c.filters)
        manifest = read_manifest(self.export_folder())
        export = None
        if manifest is not None and manifest.get("source_files") == self.file_in:
            export = {"date": manifest.get("date"), "steps": manifest.get("steps"),
                      "n_images": len(manifest.get("files", [])), "recipe": comparable_recipe(manifest.get("recipe"))}
        pod = None
        if self.Check_eigproc and self.pod_file and os.path.isfile(self.pod_file):
            pod = {"file": rel(self.pod_file), "modes_evaluated": self.NMod_saved}
        panel = self.panel
        return {"masks": [{"name": m.name, "enabled": m.enabled, "points": m.points.tolist()} for m in self.masks],
                "contrast": contrast,
                "background_removal": {"enabled": bool(self.use_bgrem.get()),
                                       "modes_removed": panel.modes_removed.get(),
                                       "modes_to_evaluate": panel.n_modes.get()},
                "automatic_threshold": {"eps_sigma": panel.eps_sigma.get(), "eps_psi": panel.eps_psi.get()},
                "export_folder": rel(self.export_folder()),
                "export_format": self.export_format,
                "export": export,
                "pod": pod}

    def restore(self, data, path):
        panel = self.panel
        # The image looked at, before anything is shown
        self.showing_im_i = min(self.app.image_index + 1, self.NImg)
        try:
            self.masks[:] = [Mask(m["points"], m.get("name", "Mask"), m.get("enabled", True))
                             for m in data.get("masks", [])]
        except (KeyError, TypeError, ValueError) as e:
            self.update_status(f"Project: masks not restored ({e})")
        self.masks_changed()
        contrast = dict(data.get("contrast") or {})
        known = {f.name for f in dataclass_fields(ContrastSettings)}
        try:
            self.contrast = ContrastSettings(**{k: v for k, v in contrast.items() if k in known})
        except (TypeError, ValueError) as e:
            self.update_status(f"Project: contrast enhancement not restored ({e})")
        panel.show_contrast_settings(self.contrast)
        bgrem = data.get("background_removal") or {}
        self.use_bgrem.set(bool(bgrem.get("enabled", self.use_bgrem.get())))
        for var, key in ((panel.modes_removed, "modes_removed"), (panel.n_modes, "modes_to_evaluate")):
            if bgrem.get(key) is not None:
                var.set(str(bgrem[key]))
        eps = data.get("automatic_threshold") or {}
        for var, key in ((panel.eps_sigma, "eps_sigma"), (panel.eps_psi, "eps_psi")):
            if eps.get(key) is not None and not np.isnan(str2double(eps[key])):
                var.set(str(eps[key]))
                setattr(self, "eps_auto_" + key[4:], str2double(eps[key]))
        panel.export_folder.set(path(data.get("export_folder")))
        self.path_export = panel.export_folder.get()
        fmt = data.get("export_format")
        self.export_format = fmt if fmt in EXPORT_FORMAT_LABELS else DEFAULT_EXPORT_FORMAT
        panel.export_format.set(EXPORT_FORMAT_LABELS[self.export_format])
        pod = data.get("pod")
        if pod:
            filename = path(pod.get("file"))
            if self.load_pod(filename) is not None:
                self.pod_file = filename
                self.pod_changed(save=False)
            else:
                self.update_status(f"The spectrum and the modes saved ({filename}) are not those of the images with "
                                   "these masks and contrast enhancement: evaluate them again")
        self.panel.update_state()
        self.show_main()

    def open_folder(self, folder):
        """Folder entered or chosen in the side panel: all its images are opened."""
        if not folder:
            return
        if not os.path.isdir(folder):
            self.app.status_error(f"The folder doesn't exist: {folder}")
            self.panel.update_images()
            return
        self.open_images(folder)
        self.panel.update_images()

    def load_images_memory(self):
        """Load the images into memory or generate temporary files if they don't fit."""

        # Check that the user selected the images
        if self.no_images_selected():
            return False
        try:
            loaded = super().load_images_memory()
        except (OSError, ValueError) as me:
            self.app.msgbox(f"An error occurred during the analysis of the images.\n{me}", "Error", "error")
            return False
        if loaded:
            self.update_status("Please evaluate the images spectrum from the side panel")
        return loaded

    def confirm_temp_files(self):
        """Handle the memory info message."""
        if not getpref("mem_warn_msg", True):
            return True
        Message = ("The memory available in this system is not enough to process the "
                   "selected set of images. Some temporary files will be stored in the "
                   "temporary folder on you hard drive to allow the calculation and this "
                   "process may slow down the analysis. Do you wish to continue anyway?")
        button = self.app.questdlg(Message, "Memory info", ["Yes", "No", "Yes, don't show again"], "Yes")
        if button is None or button == "No":
            return False
        if button == "Yes, don't show again":
            setpref("mem_warn_msg", False)
        return True

    def evaluate_spectrum(self):
        """Evaluate the spectrum of the images."""

        # Check that the user selected the images
        if self.no_images_selected():
            return None
        elif not self.Check_imload:
            # If images haven't been loaded in memory, load them before
            # evaluating the modes.
            self.load_images_memory()
            # If the user didn't load the images into memory
            if not self.Check_imload:
                return None

        eigVal = super().evaluate_spectrum()
        self.update_status("Image modes can now be evaluated from the side panel")

        # Plot spectrum
        self.show_spectrum(self.Check_modproc)

        # Automatic number of modes, which replaces the number of modes to evaluate only if there is none
        N_auto = self.automatic_N_modes()
        self.panel.auto_label["text"] = f"{N_auto} modes" if N_auto > 0 else "no threshold found"
        if N_auto > 0 and not self.panel.n_modes.get().strip():
            self.panel.n_modes.set(str(N_auto))
        self.store_pod()
        return eigVal

    def evaluate_modes(self, n_modes=None, on_mode=None):
        """Evaluate the eigenmodes of the images: the number of modes entered in the side panel, or n_modes (the
        modes removed by the export or the PIV). They are shown, and saved with the project."""
        if n_modes is None:
            # Check that the images are selected, loaded into memory and their spectrum evaluated
            if not self.spectrum_ready():
                return None

            # Number of modes entered in the side panel
            NMod = str2double(self.panel.n_modes.get())
            if not NMod >= 1:
                self.app.msgbox('Please enter in "Modes to evaluate" a number of modes higher than 0.',
                            "Evaluate eigenmodes", "warn")
                return None
            NMod = int(min(NMod, self.NImg))
            self.panel.n_modes.set(str(NMod))
        else:
            NMod = n_modes

        def show(mod_i):
            # Show evaluated mod
            self.showing_mod_i = mod_i
            self.callback_arrows(0, "modes")
            if on_mode is not None:
                on_mode(mod_i)

        if super().evaluate_modes(NMod, show) is None:
            return None
        self.update_status("The slider or the arrow keys browse the modes (mouse over the modes)")
        self.update_status('Add the background removal to the recipe and export the images from the side panel')

        # Modes removed: all the modes evaluated, unless the user chose fewer
        if n_modes is None and not 0 < str2double(self.panel.modes_removed.get()) <= self.NMod_saved:
            self.panel.modes_removed.set(str(NMod))
        self.showing_mod_i = NMod
        self.pod_changed()
        self.app.log_step("modes", f"Spectrum and {self.NMod_saved} POD modes evaluated")
        return self.NMod_saved

    def pod_changed(self, save=True):
        """Spectrum and modes evaluated (by the side panel, the export or the PIV) or restored: they are shown, and
        saved with the project."""
        if self.Check_modproc:
            self.showing_mod_i = int(np.clip(self.showing_mod_i, 1, self.NMod_saved))
            self.callback_arrows(0, "modes")
        elif self.Check_eigproc:
            self.show_spectrum(False)
        if self.Check_eigproc:
            N_auto = self.automatic_N_modes()
            self.panel.auto_label["text"] = f"{N_auto} modes" if N_auto > 0 else "no threshold found"
        self.panel.update_state()
        if self.Check_imsel:
            self.show_main()
        if save:
            self.store_pod()

    def store_pod(self):
        """Save the spectrum and the modes evaluated (pod.npz of the export folder), restored with the project."""
        try:
            self.pod_file = self.save_pod()
        except OSError as e:
            self.update_status(f"The spectrum and the modes could not be saved: {e}")
        self.app.save_project()

    def set_automatic_modes(self):
        """Set the automatic number of modes."""

        # Check that the images are selected, loaded into memory and their spectrum evaluated
        if not self.spectrum_ready():
            return

        # Epsilon entered in the side panel
        eps_sigma = str2double(self.panel.eps_sigma.get())
        eps_psi = str2double(self.panel.eps_psi.get())
        if np.isnan(eps_sigma) or np.isnan(eps_psi):
            self.app.msgbox("Please enter numbers for the tolerances ε σ and ε ψ.", "Automatic number of modes", "warn")
            return
        self.eps_auto_sigma = eps_sigma
        self.eps_auto_psi = eps_psi
        N_auto = self.automatic_N_modes()

        # The result is shown in the side panel and proposed as the number of modes to evaluate
        self.update_status(" ")
        if N_auto > 0:
            self.panel.auto_label["text"] = f"{N_auto} modes"
            self.panel.n_modes.set(str(N_auto))
            Message = f"The automatic number of modes based on the user input epsilon is {N_auto}"
        else:
            self.panel.auto_label["text"] = "no threshold found"
            Message = ("The user input epsilon did not produce any automatic number of modes. "
                       "Please select a different value of epsilon")
        self.update_status(Message)
        self.app.save_project()

    def export(self, target):
        """Export the images processed with the recipe set in the side panel (masks, contrast enhancement and
        background removal), exactly as the image shown."""

        # Nothing to export (the button is disabled, but not the shortcut)
        if self.app.root.grab_current() is not None:
            return
        NMod = self.n_modes_removed()
        if not (self.masks_active() or self.contrast_on() or NMod):
            self.app.window.bell()
            return

        # Check the target
        selection = self.select_target(target)
        if selection is None:
            return
        target_path, target_list = selection
        NImg = len(target_list)

        # The background of the image i is reconstructed with the temporal
        # coefficients of the training image i
        if NMod and NImg > self.NImg:
            self.app.status_error(f"The target contains {NImg} images, more than the {self.NImg} training images "
                              "used to evaluate the modes.")
            return

        # Output folder: the export folder of the side panel for the training images (PreProcessing by default),
        # the subfolder PreProcessing of the folder of other images
        bf = (self.export_folder() if same_path(target_path, self.path_in)
              else os.path.join(target_path, PREPROCESSING_FOLDER))
        if self.app.folder_in_use(bf):
            self.app.msgbox("The PIV/PTV is running on the images of this folder: export the images in another "
                            "folder, or stop the PIV/PTV first.", "Export", "warn")
            return

        # The modes removed are evaluated first if needed
        if NMod > self.NMod_saved and self.evaluate_modes(NMod) is None:
            return

        # Ask the user to export background images
        save_background = False
        if NMod:
            Message = ("Selected images will be exported without background. "
                       "Do you also want to export background images in different files?")
            save_background = self.app.questdlg(Message, "Save background", ["Yes", "No"], "No") == "Yes"

        try:
            self.export_images(bf, NMod, target_path, target_list, save_background)
        except ValueError:
            self.app.msgbox("Target image not compatible.", "Error", "error")

    def export_images(self, output_folder, n_modes_removed=0, folder=None, files=None, save_background=False):
        """Export of the images (button Export, or the PIV): kept in the project and shown in the side panel."""
        written = super().export_images(output_folder, n_modes_removed, folder, files, save_background)
        steps = " → ".join(self.recipe_steps(n_modes_removed, detailed=True))
        self.app.log_step("export", f"{len(written)} images exported to {self.app.project_path(self.path_out)} "
                                    f"({steps})")
        self.panel.update_export()
        return written

    def project_name(self):
        """Name of the project, at the start of the names of the images exported: the one of the window."""
        return self.app.project_name()

    def export_folder(self):
        """Folder of the images exported for the PIV: the one of the side panel (PreProcessing by default)."""
        return self.panel.export_folder.get() or super().export_folder()

    def select_target(self, target):
        """Folder and names of the images to process: the training images ("training") or a new target picked by
        the user ("new_target_direct" or "new_target_pattern"). None if cancelled."""
        if target == "training":
            if self.no_images_selected():
                return None
            return self.path_in, self.file_in

        if target == "new_target_direct":
            # Select a new set of images to process
            files = filedialog.askopenfilenames(parent=self.app.parent(), title="Select images...",
                                                initialdir=self.path_in, filetypes=IMAGE_FILES)
            if not files:
                return None
            return os.path.normpath(os.path.dirname(files[0])), sorted(os.path.basename(f) for f in files)

        # Open the pattern select dialog
        selection = self.pattern_selection()
        if selection is not None and not selection[1]:
            self.app.status_error("The user pattern did not select any image")
            return None
        return selection

    def show_spectrum(self, show_current):
        """Plot the spectrum, the temporal averages of the modes and the temporal coefficient of the mode shown,
        with the position of the current mode if show_current, and the image shown on its temporal coefficient."""
        fig = self.figure_spectrum.fig
        draw_spectrum(fig, self.eigVal, self.mean_PSI, self.PSI, self.showing_mod_i if show_current else None)
        self.image_marker = None
        if show_current and self.Check_imsel:
            ax = fig.axes[2]
            self.image_marker = (ax.axvline(1, color=IMAGE_MARK_COLOR, linewidth=1, linestyle="--"),
                                 ax.plot([], [], "o", markersize=9, markerfacecolor="none",
                                         markeredgecolor=IMAGE_MARK_COLOR, markeredgewidth=2, label="image shown")[0])
            ax.legend(fontsize="small", loc="upper right")
            self.mark_image(draw=False)
        self.figure_spectrum.canvas.draw_idle()

    def mark_image(self, draw=True):
        """Image shown marked on the temporal coefficient of the mode shown."""
        if self.image_marker is None or self.image_marker[1].axes is None:
            return
        k = min(self.showing_im_i, self.PSI.shape[0])
        line, point = self.image_marker
        line.set_xdata([k, k])
        point.set_data([k], [self.PSI[k - 1, self.showing_mod_i - 1]])
        if draw:
            self.figure_spectrum.canvas.draw_idle()

    def image_tools(self, fig_name):
        """Buttons of the toolbar of the panels showing the images and the modes."""
        return [("Colormap", lambda: self.change_colormap(fig_name)),
                ("Color limits...", lambda: self.adjust_image(fig_name))]

    def set_active_panel(self, fig_name):
        """Panel ("main", "modes" or "spectrum") whose series the arrow keys browse: the images, or the modes (also
        in the spectrum). The spectrum follows the image (marked on the temporal coefficient) and the mode shown."""
        self.active_panel = fig_name
        linked = {"main": ("spectrum",), "modes": ("spectrum",), "spectrum": ("modes",)}[fig_name]
        for name, panel in (("main", self.figure_main), ("modes", self.figure_modes),
                            ("spectrum", self.figure_spectrum)):
            panel.set_active(True if name == fig_name else "linked" if name in linked else False)

    def callback_arrows(self, step, fig_name, index=None):
        """Show the image or the mode step after the current one, or the one of the index chosen with the
        slider."""

        # Check if the arrow was pressed in the figure of modes (or of the spectrum) or main
        if fig_name == "spectrum":
            fig_name = "modes"
        if fig_name == "modes" and self.Check_modproc:
            if index is not None:
                if index == self.showing_mod_i:
                    return
                step = index - self.showing_mod_i
            self.showing_mod_i = min(max(self.showing_mod_i + step, 1), self.NMod_saved)

            data = load_npz(self.mode_file(self.showing_mod_i), "PHI_bloc")
            PHI_bloc = data["PHI_bloc"].reshape(self.Dim)

            # Plot the spectrum
            self.show_spectrum(True)

            # Show the mode, keeping the zoom and the colormap
            clim = self.phi_caxis.val if self.phi_caxis.active else mode_limits(PHI_bloc)
            title = mode_title(self.showing_mod_i, self.NMod_saved, self.eigVal)
            if self.figure_modes.im is None:
                self.phi_caxis.val = self.figure_modes.imagesc(PHI_bloc, "hot", clim, title, colorbar=True)
            else:
                self.phi_caxis.val = self.figure_modes.set_image(PHI_bloc, clim, title)
            self.figure_modes.set_slider(self.showing_mod_i, self.NMod_saved)

        elif fig_name == "main" and self.Check_imsel:
            if index is not None:
                if index == self.showing_im_i:
                    return
                step = index - self.showing_im_i
            self.showing_im_i = min(max(self.showing_im_i + step, 1), self.NImg)

            # Show the image
            self.show_main()

    def show_main(self):
        """Show the current image in the main panel processed with the recipe set in the side panel, as it is
        exported. The zoom and the colormap are kept, so that the images, and the steps of the recipe, can be
        compared. While a mask is drawn or edited, the masks aren't applied, to see what they cover."""
        self.panel.update_recipe()
        if not self.Check_imsel:
            return
        name = self.file_in[self.showing_im_i - 1]
        self.app.image_index = self.showing_im_i - 1
        self.mark_image()
        # Without the modes evaluated, the background isn't removed yet
        NMod = self.n_modes_removed()
        removed = NMod if NMod <= self.NMod_saved else 0
        editing = self.mask_editor.mode is not None
        self.last_cap = None
        img, _ = self.process_image(os.path.join(self.path_in, name), self.showing_im_i - 1,
                                    self.load_modes(removed), masks=not editing)
        clim = self.im_caxis.val if self.im_caxis.active else (0, np.max(img)) if removed else None

        # Title: the image, then the steps of the recipe applied to it
        steps = self.recipe_steps(removed, detailed=True, editing=editing)
        title = f"Image {self.showing_im_i} of {self.NImg}: {name}\n" + (" → ".join(steps) or "original image")
        if NMod > removed:
            title += f" (background removal of {NMod} modes once they are evaluated)"
        self.im_caxis.val = self.figure_main.set_image(img, clim, title)
        self.figure_main.set_slider(self.showing_im_i, self.NImg)
        self.figure_main.compare_slider.state(["!disabled"])
        self.show_compare()
        self.mask_editor.refresh()

        # Intensity cap of the image shown
        if self.last_cap is not None:
            cap, fraction = self.last_cap
            self.panel.cap_label["text"] = (f"Cap on this image: {cap:.1f} grey levels "
                                            f"({fraction * 100:.2f} % of the pixels capped)")
        else:
            self.panel.cap_label["text"] = ""

    def set_compare(self, fraction):
        """Slider of the comparison before / after moved: the image before is shown left of the fraction of the
        width."""
        self.compare_fraction = fraction
        if self.Check_imsel and hasattr(self, "figure_main"):
            self.show_compare()

    def show_compare(self):
        """Comparison of the image shown with the image before the processing (as read from its file)."""
        before = None
        if self.compare_fraction < 1:
            filename = os.path.join(self.path_in, self.file_in[self.showing_im_i - 1])
            if self.before_cache[0] != filename:
                self.before_cache = (filename, imread(filename))
            before = self.before_cache[1]
        self.figure_main.compare(before, self.compare_fraction, self.im_caxis.val if self.im_caxis.active else None)

    def n_modes_removed(self):
        """Number of modes removed by the recipe (0 if the background removal is off). They may not be evaluated
        yet: the export and the PIV evaluate them."""
        if not (self.use_bgrem.get() and self.Check_imsel):
            return 0
        NMod = str2double(self.panel.modes_removed.get())
        return int(min(NMod, self.NImg)) if NMod >= 1 else 0

    def set_modes_removed(self):
        """Number of modes removed changed in the side panel."""
        NMod = str2double(self.panel.modes_removed.get())
        NMod = int(np.clip(NMod, 0, max(self.NImg, 1))) if not np.isnan(NMod) else (self.NMod_saved or DEFAULT_MODES)
        if self.panel.modes_removed.get() != str(NMod):
            self.panel.modes_removed.set(str(NMod))
        self.show_main()
        self.app.save_project()

    def toggle_bgrem(self, flip=False):
        """Background removal in the recipe on or off (flip: switch the button of the side panel). The modes are
        evaluated when they are needed (export, PIV), or with the side panel."""
        if flip:
            self.use_bgrem.set(not self.use_bgrem.get())
        self.show_main()
        NMod = self.n_modes_removed()
        if NMod > self.NMod_saved:
            self.update_status(f"The background removal needs {NMod} modes: they are evaluated by the export or the "
                               "PIV/PTV, or now with Ctrl+M")
        self.app.save_project()

    def change_temp_fold(self):
        """Change the temporary folder where files are stored. If images are
        already loaded into memory, ask the user if they want to proceed."""

        # Check if there are images already loaded into memory
        if self.Check_imload:
            Message = ("A set of images has already been loaded into memory. "
                       "By changing the temporary folder a new session will be started.\n"
                       "Do you wish to continue?")
            button = self.app.questdlg(Message, "Change temporary folder", ["Yes", "No"], "Yes")
        else:
            button = "Yes"

        if button == "Yes":
            # Set new temporary folder
            bf = filedialog.askdirectory(parent=self.app.parent(), title="Select temporary folder...",
                                         initialdir=self.pathtemp)
            if not bf:
                return
            self.initialize_session()
            self.pathtemp = os.path.normpath(bf)
            # Keep it when new images are selected
            self.pathtemp_user = True

            # Update status
            self.update_status(" ")
            self.update_status(f"Temporary folder changed to: {self.pathtemp}")

    def memory_info(self):
        if self.Check_imsel:
            n_pixels = imread(os.path.join(self.path_in, self.file_in[0])).size
            NImg = self.NImg
        else:
            n_pixels = 1000 * 1000
            NImg = 100

        # Evaluate the memory needed
        MemAv = available_memory()
        MaxImg = self.max_images(MemAv, n_pixels)
        N_bloc = -(-NImg // MaxImg) if NImg > MaxImg else 0
        # Dialog
        Message = (f"Memory available: {MemAv / 1e9:.2f} GB\n"
                   f"Memory usable: {MemAv / 1e9 * self.mem_safe:.2f} GB\n"
                   f"Max #N images (without temp file): {MaxImg}\n"
                   f"Temporary files needed: {N_bloc}")
        if not self.Check_imsel:
            Message = f"No images selected!\nValues are shown for 100 test images of 1 Mpx\n{Message}"
        self.app.msgbox(Message, "Memory info")

    def set_memory(self):
        """Change the maximum memory usable for the software."""

        # Evaluate the memory needed
        MemAv = available_memory()

        Message = ("In order to prevent system crashes, only part of total memory available in the system is "
                   "used by the software. The usable memory can be changed from this dialog. Using the entire "
                   "memory could cause crashes and instabilities!\n\n"
                   f"Total memory available: {MemAv / 1e9:.2f}GB\nUsable memory: {MemAv / 1e9 * self.mem_safe:.2f} GB\n"
                   "Fraction of the total memory to use (between 0.1 and 1):")
        answ = self.app.inputdlg([Message], "Change memory usage", [num2str(self.mem_safe)])
        if answ is None or np.isnan(str2double(answ[0])):
            return

        old = self.mem_safe
        self.mem_safe = max(0.1, min(1, str2double(answ[0])))

        self.update_status(" ")
        self.update_status(f"Fraction of usable memory changed from {old:.2f} to {self.mem_safe:.2f}")

    def apply_contrast(self, new):
        """New contrast enhancement settings (intensity capping, local min-max normalization, CLAHE), entered in
        the side panel and applied before the POD or after the background removal (no filter selected: no
        contrast enhancement). Returns False if the user keeps the current settings."""
        if new == self.contrast:
            return True

        # A new pre-processing changes the training images: they must be loaded again
        def pre_pod(settings):
            return settings if settings.stage == "pre" and settings.filters else None

        if pre_pod(self.contrast) != pre_pod(new) and not self.confirm_training_change():
            return False
        self.contrast = new

        self.update_status(" ")
        self.update_status(f"Contrast enhancement {ControlPanel.STAGES[new.stage]}: "
                           + (self.contrast_description() or "none"))
        self.show_main()
        self.app.save_project()
        return True

    def confirm_training_change(self):
        """Ask the user before a change of the training images (masks or contrast enhancement before POD), which
        discards the spectrum and the modes evaluated. Returns False if the user keeps them."""
        if not (self.Check_imload or self.Check_eigproc):
            return True
        Message = ("This changes the training images: the spectrum and the modes evaluated will be discarded "
                   "and must be evaluated again.\nDo you wish to continue?")
        if self.app.questdlg(Message, "Training images changed", ["Yes", "No"], "No") != "Yes":
            return False
        self.discard_pod()
        return True

    def discard_pod(self):
        """The training images changed: the spectrum and the modes must be evaluated again."""
        super().discard_pod()
        self.reset_pod_figures()

    def reset_pod_figures(self):
        """Empty the panels of the modes and of the spectrum, which must be evaluated."""
        self.figure_modes.message("Spatial modes\n\nEvaluate them from the side panel (Ctrl+M)")
        self.figure_modes.set_slider(1, 0)
        self.figure_spectrum.message("Video spectrum and temporal coefficients\n\n"
                                     "Evaluate them from the side panel (Ctrl+R)")
        self.panel.auto_label["text"] = ""
        self.panel.update_state()

    def new_mask(self):
        """Draw a new mask on the image."""
        if self.no_images_selected() or self.mask_editor.mode is not None:
            return
        self.mask_editor.start_draw()
        self.panel.update_mask_buttons()
        self.show_main()

    def add_mask(self, points, name=None, enabled=True):
        """Polygon drawn: new mask, edited to adjust it."""
        if enabled and not self.confirm_training_change():
            self.cancel_mask()
            return None
        mask = super().add_mask(points, name, enabled)
        self.masks_changed()
        self.panel.select_mask(len(self.masks) - 1)
        self.mask_editor.start_edit(mask)
        self.panel.update_mask_buttons()
        self.show_main()
        return mask

    def load_masks_dialog(self):
        """Load masks saved in a file (by the interface or by a script), replacing the current ones."""
        filename = filedialog.askopenfilename(parent=self.app.parent(), title="Import masks...", initialdir=self.path_in,
                                              filetypes=MASK_FILES)
        if not filename or self.mask_editor.mode is not None:
            return
        if not self.confirm_training_change():
            return
        try:
            self.load_masks(filename)
        except (OSError, ValueError, KeyError) as me:
            self.app.status_error(f"The masks could not be loaded from {filename}: {me}")
            return
        self.masks_changed()
        self.show_main()

    def save_masks_dialog(self):
        if not self.masks:
            self.app.msgbox("There is no mask to export.", "Export masks", "warn")
            return
        filename = filedialog.asksaveasfilename(parent=self.app.parent(), title="Export masks...", initialdir=self.path_in,
                                                initialfile="masks.json", filetypes=MASK_FILES,
                                                defaultextension=".json")
        if filename:
            self.save_masks(filename)

    def edit_mask(self):
        """Button "Edit" of the masks: edit the mask selected, or end the edition ("Done")."""
        if self.mask_editor.mode == "edit":
            self.finish_mask()
            return
        i = self.panel.selected_mask()
        if i is None or self.mask_editor.mode is not None:
            return
        if self.masks[i].enabled and not self.confirm_training_change():
            return
        self.mask_editor.start_edit(self.masks[i])
        self.panel.update_mask_buttons()
        self.show_main()

    def finish_mask(self):
        mask = self.mask_editor.mask
        self.mask_editor.stop()
        self.masks_changed(f"{mask.name} edited ({len(mask.points)} points)" if mask else None)
        self.show_main()

    def cancel_mask(self):
        self.mask_editor.stop()
        self.panel.update_mask_buttons()
        self.show_main()

    def mask_edited(self):
        """Mask moved, or point added or removed, while editing it."""
        self.mask_cache = None
        self.panel.update_masks()
        self.mask_editor.refresh()

    def delete_mask(self):
        i = self.panel.selected_mask()
        if i is None or self.mask_editor.mode is not None:
            return
        mask = self.masks[i]
        if mask.enabled and not self.confirm_training_change():
            return
        del self.masks[i]
        self.masks_changed(f"{mask.name} deleted")
        self.show_main()

    def toggle_mask(self, i):
        """Enable or disable a mask."""
        if self.mask_editor.mode is not None:
            return
        mask = self.masks[i]
        if not self.confirm_training_change():
            return
        mask.enabled = not mask.enabled
        self.masks_changed(f"{mask.name} {'enabled' if mask.enabled else 'disabled'}")
        self.show_main()

    def rename_mask(self, i):
        answ = self.app.inputdlg(["Name of the mask"], "Rename mask", [self.masks[i].name])
        if answ is None or not answ[0].strip():
            return
        self.masks[i].name = answ[0].strip()
        self.masks_changed()
        self.mask_editor.refresh()

    def masks_changed(self, message=None):
        self.mask_cache = None
        self.panel.update_masks()
        if message:
            self.update_status(message)
        self.app.save_project()

    def change_colormap(self, fig_name):
        win = self.figure_main if fig_name == "main" else self.figure_modes
        if not alive(win) or win.im is None:
            return
        if win.im.get_cmap().name == "gray":
            win.set_colormap("hot", colorbar=True)
        else:
            win.set_colormap("gray", colorbar=False)

    def adjust_image(self, fig_name):
        # If no image is selected, simply return
        if not self.Check_imsel:
            return
        # Check in which window the button was pressed
        caxis = self.phi_caxis if fig_name == "modes" else self.im_caxis
        if caxis.val is None:
            return

        Message = "Insert grayscale display limit. Press cancel for automatic\nMinimum:"
        answ = self.app.inputdlg([Message, "Maximum:"], "Adjust image display", [num2str(v) for v in caxis.val])
        if answ is None:
            caxis.active = False
        else:
            cax = [str2double(a) for a in answ]
            if np.isnan(cax).any():
                return
            cax[1] = max(cax[1], cax[0] + 1e-6)
            caxis.val = cax
            caxis.active = True

        # Update the figure
        if fig_name == "main":
            self.show_main()
        else:
            self.callback_arrows(0, fig_name)

    def pattern_selection(self):
        """Open a dialog to select files with a pattern and folder.
        Returns the folder and the files in the range selected, None if cancelled."""

        # First, select a folder
        sel_folder = filedialog.askdirectory(parent=self.app.parent(), title="Select images folder...",
                                             initialdir=self.path_in)
        # Detect no folder selected
        if not sel_folder:
            return None
        sel_folder = os.path.normpath(sel_folder)

        # Detect a default filename for the images
        flist = [f for ext in ("tif", "tiff", "bmp", "png", "jpg") for f in list_files(sel_folder, "*." + ext)]
        defname = flist[mround(len(flist) / 2) - 1] if flist else "*.tif"

        # Then, insert a pattern
        answ = self.app.inputdlg(["Enter the file extension or wildcard pattern to select files\n\n"
                              "Ex: airfoil_*a.tif will select:\n"
                              "airfoil_1a.tif, airfoil_0001a.tif, airfoil_0002a.tif, etc."],
                             "Select images", [defname])
        # Check empty pattern
        if answ is None:
            return None

        # Check that the files exist
        filelist = list_files(sel_folder, answ[0])
        if len(filelist) < 2:
            self.app.msgbox("No files found with the selected pattern.", "No images selected", "warn")
            self.update_status(" ")
            self.update_status(f'No images found with the pattern "{answ[0]}" in the folder "{sel_folder}"')
            return None

        prompt = [f"{len(filelist)} images were found with the user pattern\n"
                  "Please input the range of images to select.\nFrom:", "To:"]
        answ = self.app.inputdlg(prompt, "Select images", ["1", str(len(filelist))])
        if answ is None:
            return None
        first, last = str2double(answ[0]), str2double(answ[1])
        if not (1 <= first and last <= len(filelist)):
            self.app.status_error(f"The range of images must be between 1 and {len(filelist)}")
            return None
        return sel_folder, filelist[int(first) - 1 : int(last)]

    def clear_cache(self):
        """File > Clear cache: the images loaded and the modes in the temporary folder are deleted.
        The spectrum is kept, and the modes are restored from pod.npz if they were saved (evaluated again
        otherwise when they are needed)."""
        deleted_tmp, deleted_mod = super().clear_cache()
        self.before_cache = (None, None)
        text = (f"Pre-processing cache cleared: images unloaded, {deleted_tmp} image files and {deleted_mod} mode "
                f"files deleted from {self.pathtemp}")
        if self.Check_eigproc and self.pod_file and self.load_pod(self.pod_file):
            text += f"; {self.NMod_saved} modes restored from {self.pod_file}"
        self.pod_changed(save=False)
        self.update_status(text)

    def delete_temporary_files(self, exiting):
        """Delete all the temporary files saved by the software and start a
        new session. If exiting is false, ask a confirmation to the user,
        otherwise proceed without confirmation."""

        if exiting:
            button = "Yes"
        else:
            Message = ("This action will delete all the temporary files created and "
                       "will start a new session. Do you wish to continue?")
            button = self.app.questdlg(Message, "Start new session", ["Yes", "No"], "Yes")

        if button == "Yes":
            # Delete temporary files
            deleted_tmp, deleted_mod = self.clean_temporary_files()

            if not exiting:
                # Clear status window
                self.update_status(0)
                self.update_status(f"{deleted_tmp} temporary files deleted")
                self.update_status(f"{deleted_mod} stored modes deleted")
                self.update_status(f"Temporary folder: {self.pathtemp}")
                self.update_status(f"Image folder: {self.path_in}")
                self.update_status('Please choose the folder of the images to analyse (side panel, "Images")')

                # Initialize a new session, without the masks
                self.initialize_session()
                self.masks.clear()
                self.masks_changed()

    def show_initial_image(self):
        bring_to_front(self.app.window)
        self.figure_main.imagesc(magic(32), "gray", title='Choose the folder of the training images')
        self.figure_main.ax.set_axis_off()
        self.figure_main.set_slider(1, 0)
        self.figure_main.compare_slider.state(["disabled"])

    def no_images_selected(self):
        """Warn the user if no images are selected (returns True in that case)."""
        if not self.Check_imsel:
            self.app.msgbox('No images selected. Please choose the folder of the images (side panel, "Images").', "No images selected", "warn")
        return not self.Check_imsel

    def spectrum_ready(self):
        """Load the images and evaluate their spectrum if needed. False if it isn't possible."""
        if self.no_images_selected():
            return False
        if not self.Check_eigproc:
            self.evaluate_spectrum()
        return self.Check_eigproc
