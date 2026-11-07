"""Batch runs of a step on projects of the workspace (Run step > Batch..., and the button Archive projects...).

The window Batch (BatchDialog) shows the tree of the workspace: the projects checked are run (those the step can't
run are greyed out), each with its settings: those of its project file (the project open, or one without them:
those of the interface). The settings are changed as differences: with a project selected, for this project only;
with a folder selected, that setting for all the projects below it (each keeps its other settings). The form shows
the settings of the node selected (of its first project for a folder; in bold with a button ↺: those that differ
between its projects, ↺ giving them all the value shown), the panels (the quantities of the post-processing) too.
Reset gives the projects of the node the settings of the project open. The column Differences compares each project
with the project open. The options of the run (archive file, stitch matching) are in their own box.

The settings changed are saved soon in the project files (the section of the step: to_project), so that a project
opened shows them; the project open takes them in its page when the window closes or runs. The archive keeps its
settings (the categories) by project in the workspace.

The run (BatchRunWindow) is a process of its own (batch_run.py): its log, its progress and the button Stop (the
step running ends there; the PIV keeps the pairs computed). Meanwhile no other project can be opened; at its end the
workspace is refreshed and the project open, if it was run, opened again."""

import copy
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

from ..project import PROJECT_FILE, absolute, read_project, write_project
from ..workspace import detect as workspace_detect, image_names, key
from .widgets import Dialog, HelpDot, PathBox, RangeSlider

GREY, LOST = "#9a9a9a", "#c62828"


def cap(text):
    """A text as shown in the boxes, the columns and the buttons: its first letter in capital."""
    return text[:1].upper() + text[1:] if text else text
HELP = ("Check the projects to run; checking a folder checks everything below it. Select a project to edit its "
        "settings, or a folder to set one setting for all the projects below it. Bold settings differ between "
        "those projects; ↺ gives them all the value shown. Reset copies the settings of the open project to the "
        "node. Greyed projects can't be run by this step; red ones weren't found.")


@dataclass
class Field:
    """A setting of the form of the window Batch: its path in the settings (keys), label, kind ("bool", "int",
    "float", "optional": a number or empty (None), "text", "folder": a folder inside the folder or the project
    selected, kept relative to it (empty: the default one), "file": a file (choices: its file types; empty: none),
    "choice" and "multi": among choices ((value, label)),
    "masks": the masks enabled, "range": the first and the last integers of a range (a two-sided slider; end: the path
    of the last one, None meaning the first one), "hidden": kept but not shown), help, default value (shown when the
    settings don't have it, or for a folder None)
    and project_only: a setting of each project (its export path): not shown for a folder, nor compared (differences,
    bold), nor changed by Reset."""

    path: tuple
    label: str
    kind: str = "text"
    choices: tuple = ()
    help: str = None
    default: object = None
    project_only: bool = False
    end: tuple = None


def get_path(settings, path, default=None):
    value = settings
    for k in path:
        if not isinstance(value, dict) or k not in value:
            return default
        value = value[k]
    return value


def set_path(settings, path, value):
    target = settings
    for k in path[:-1]:
        if not isinstance(target.get(k), dict):
            target[k] = {}
        target = target[k]
    target[path[-1]] = value


class NodeFolderBox(PathBox):
    """A folder of the settings of a node, shown relative to the folder of the node (base), chosen from it with
    "..."; "↺": the default one."""

    def __init__(self, master, title, chosen, base):
        super().__init__(master, title, chosen, width=16)
        self.base = base

    def path_shown(self):
        """The folder revealed in the explorer: the one of the box in the folder of the node."""
        return os.path.join(self.base, self.text) if self.base else self.text

    def browse(self):
        start = os.path.join(self.base, self.text) if self.base else None
        while start and not os.path.isdir(start) and os.path.dirname(start) != start:
            start = os.path.dirname(start)
        folder = filedialog.askdirectory(parent=self.winfo_toplevel(), title=self.title,
                                         initialdir=start or self.base or None)
        if folder:
            self.chosen(os.path.normpath(folder))


class NodeFileBox(PathBox):
    """A file of the settings of a node (the masks of the pre-processing), chosen with "..." (filetypes), entered, or
    emptied (none: chosen(None))."""

    def __init__(self, master, title, chosen, filetypes):
        super().__init__(master, title, chosen, reset=False, width=16)
        self.filetypes = filetypes

    def browse(self):
        start = os.path.dirname(self.text) if self.text else None
        path = filedialog.askopenfilename(parent=self.winfo_toplevel(), title=self.title,
                                          initialdir=start if start and os.path.isdir(start) else None,
                                          filetypes=list(self.filetypes) or None)
        if path:
            self.chosen(os.path.normpath(path))


class SettingsForm(ttk.Frame):
    """The fields of settings (Field), scrolled; changed(field, value) is called when a field is changed (the value
    converted: ValueError if invalid); unify(field): the button of a setting not in common (the value shown for all
    the projects of the node); choice(field, value, on): a choice of a field "multi" checked or not, for each project
    (their other choices kept; its button ↺: the state shown)."""

    def __init__(self, master, changed, unify=None, choice=None):
        super().__init__(master)
        self.changed_callback, self.unify_callback, self.choice_callback = changed, unify, choice
        ttk.Style(self).configure("Bold.TCheckbutton", font=("TkDefaultFont", 9, "bold"))
        self.canvas = tk.Canvas(self, highlightthickness=0, height=260,
                                background=ttk.Style(self).lookup("TFrame", "background") or None)
        bar = ttk.Scrollbar(self, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar.pack(side=tk.LEFT, fill=tk.Y)
        self.inner = ttk.Frame(self.canvas)
        self.window = self.canvas.create_window(0, 0, window=self.inner, anchor=tk.NW)
        self.inner.bind("<Configure>", lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.window, width=event.width))
        for widget in (self.canvas, self.inner):
            widget.bind("<Enter>", lambda event: self.bind_all("<MouseWheel>", self.wheel))
            widget.bind("<Leave>", lambda event: self.unbind_all("<MouseWheel>"))
        self.widgets = []
        self.pending = []  # The entries: their value applied if it changed (flush)

    def flush(self):
        """The entries changed and not applied yet (no Enter, nor focus left) applied."""
        for entered in self.pending:
            entered()

    def wheel(self, event):
        self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def show(self, fields, settings, enabled=True, base=None, differs=(), differ_choices=None, unavailable=None,
             limits=None):
        """The fields with the values of the settings (enabled: they can be changed; base: the folder of the node,
        for the folders; differs: the paths of the values that differ between the projects of a folder, their labels
        in bold with a button ↺: the value shown for all of them; differ_choices: {path of a field "multi": its
        choices that differ}, each in bold with its button; unavailable: {path: reason} of the fields that the
        projects of the node can't use, greyed out with the reason in their help; limits: {path: (lo, hi)} of the
        fields "range")."""
        self.base, self.limits = base, limits or {}
        from .widgets import HelpDot

        for child in self.inner.winfo_children():
            child.destroy()
        self.widgets, self.pending = [], []
        state = ["!disabled"] if enabled else ["disabled"]
        fields = [f for f in fields if f.kind != "hidden"]  # Settings kept but not shown
        for r, field in enumerate(fields):
            value = get_path(settings, field.path, field.default)
            cell = ttk.Frame(self.inner)
            differ = any(d[:len(p)] == p for d in differs for p in (field.path, field.end) if p)
            font = ("TkDefaultFont", 9, "bold") if differ else None
            reason = (unavailable or {}).get(field.path)
            field_state = ["disabled"] if reason else state
            if field.kind in ("multi", "masks"):
                ttk.Label(self.inner, text=field.label, font=font).grid(row=2 * r, column=0, columnspan=2,
                                                                        sticky=tk.W, pady=(6, 0))
                cell.grid(row=2 * r + 1, column=0, columnspan=2, sticky=tk.W, padx=(12, 0))
                self.add_choices(cell, field, value, field_state, (differ_choices or {}).get(field.path, set()))
            else:
                label = ttk.Label(self.inner, text=field.label, wraplength=200, font=font)
                if reason:
                    label.configure(foreground="#9a9a9a")
                label.grid(row=2 * r, column=0, sticky=tk.W, pady=1)
                cell.grid(row=2 * r, column=1, sticky=tk.W, pady=1)
                if field.kind == "range":
                    self.add_range(cell, field, value, get_path(settings, field.end), field_state)
                else:
                    self.add_entry(cell, field, value, field_state)
            if field.help or reason:
                text = "\n\n".join(t for t in (f"Not available: {reason}" if reason else None, field.help) if t)
                HelpDot(self.inner, text).grid(row=2 * r, column=2, padx=(4, 0))
            # A field "multi": the buttons of its choices (add_choices)
            if differ and self.unify_callback is not None and not (field.kind == "multi" and self.choice_callback):
                button = ttk.Button(self.inner, text="↺", width=3,
                                    command=lambda field=field: self.unify_callback(field))
                button.state(state)
                button.grid(row=2 * r, column=3, padx=(4, 0))
                self.widgets.append(button)
        self.inner.columnconfigure(0, weight=1)

    def add_range(self, cell, field, first, last, state):
        """A field "range": a two-sided slider from lo to hi (limits; else up to 10 or its last value) and the range
        shown next to it, changed when a handle is released (the value: (first, last))."""
        first = int(first or field.default or 1)
        last = max(int(last or first), first)
        lo, hi = self.limits.get(field.path) or (1, max(10, last))
        label = ttk.Label(cell, width=8)

        def show(a, b):
            label["text"] = f"{a} to {b}" if b > a else f"{a}"
        slider = RangeSlider(cell, lo, max(hi, last, lo + 1), command=show, width=140,
                             on_release=lambda a, b, handle: self.changed_callback(field, (int(a), int(b))))
        slider.set(first, last)
        slider.set_enabled(state == ["!disabled"])
        slider.pack(side=tk.LEFT)
        label.pack(side=tk.LEFT, padx=(4, 0))

    def add_entry(self, cell, field, value, state):
        if field.kind == "file":
            box = NodeFileBox(cell, field.label, lambda path: self.changed_callback(field, path), field.choices)
            box.show(value or "", enabled=state == ["!disabled"])
            box.pack(side=tk.LEFT)
            self.widgets.append(box.entry)
            self.pending.append(box.entered)
            return
        if field.kind == "folder":
            box = NodeFolderBox(cell, field.label, lambda folder: self.changed(field, folder), self.base)
            box.show(value or field.default or "", enabled=state == ["!disabled"])
            box.pack(side=tk.LEFT)
            self.widgets.append(box.entry)
            self.pending.append(box.entered)
            return
        if field.kind == "bool":
            var = tk.BooleanVar(cell, bool(value))
            widget = ttk.Checkbutton(cell, variable=var, command=lambda: self.changed(field, var.get()))
        elif field.kind == "choice":
            labels = {v: cap(label) for v, label in field.choices}
            var = tk.StringVar(cell, labels.get(value, str(value)))
            widget = ttk.Combobox(cell, textvariable=var, values=list(labels.values()), state="readonly", width=22)
            widget.bind("<<ComboboxSelected>>", lambda event: self.changed(
                field, next(v for v, label in labels.items() if label == var.get())))
        else:
            shown = "" if value is None else (f"{value:g}" if isinstance(value, float) else str(value))
            var = tk.StringVar(cell, shown)
            widget = ttk.Entry(cell, textvariable=var, width=14)
            last = [shown]

            def entered(event=None):
                if var.get() != last[0]:
                    last[0] = var.get()
                    self.changed(field, var.get())
            widget.bind("<Return>", entered)
            widget.bind("<FocusOut>", entered)
            self.pending.append(entered)
        widget.pack(side=tk.LEFT)
        widget.state(state)
        if field.kind == "choice" and state == ["!disabled"]:
            widget.state(["readonly"])
        self.widgets.append(widget)

    def add_choices(self, cell, field, value, state, differ=()):
        if field.kind == "masks":
            masks = list(value or [])
            for i, mask in enumerate(masks):
                var = tk.BooleanVar(cell, mask.get("enabled", True))

                def toggle(i=i, var=var):
                    new = [dict(m) for m in masks]
                    new[i]["enabled"] = var.get()
                    self.changed(field, new)
                widget = ttk.Checkbutton(cell, text=f"{mask.get('name', 'Mask')} ({len(mask.get('points', []))} "
                                                    "points)", variable=var, command=toggle)
                widget.pack(anchor=tk.W)
                widget.state(state)
                self.widgets.append(widget)
            return
        chosen = list(value or [])
        variables = {}
        for v, label in field.choices:
            variables[v] = tk.BooleanVar(cell, v in chosen)
            row = ttk.Frame(cell)
            row.pack(anchor=tk.W, fill=tk.X)
            if self.choice_callback is not None:
                # This choice for each project (their other choices kept)
                command = lambda v=v: self.choice_callback(field, v, variables[v].get())
            else:
                command = lambda: self.changed(field, [k for k, _ in field.choices if variables[k].get()])
            widget = ttk.Checkbutton(row, text=cap(label), variable=variables[v], command=command,
                                     style="Bold.TCheckbutton" if v in differ else "TCheckbutton")
            widget.pack(side=tk.LEFT)
            widget.state(state)
            self.widgets.append(widget)
            if v in differ and self.choice_callback is not None:
                button = ttk.Button(row, text="↺", width=3,
                                    command=lambda v=v: self.choice_callback(field, v, variables[v].get()))
                button.pack(side=tk.LEFT, padx=(6, 0))
                button.state(state)
                self.widgets.append(button)

    def changed(self, field, raw):
        if field.kind == "folder":
            folder = self.relative_folder(field, raw)
            # The default folder: none of its own
            return self.changed_callback(field, None if folder == field.default else folder)
        try:
            value = self.convert(field, raw)
        except ValueError:
            return self.changed_callback(field, ValueError(f"{field.label}: "
                                                           + ("an integer" if field.kind == "int" else "a number")
                                                           + (" or nothing" if field.kind == "optional" else "")))
        self.changed_callback(field, value)

    def relative_folder(self, field, folder):
        """A folder chosen or entered, relative to the folder of the node (None: the default one); ValueError if it
        is outside it."""
        if not folder:
            return None
        if not os.path.isabs(folder):
            folder = os.path.normpath(folder)
        elif self.base:
            try:
                folder = os.path.relpath(folder, self.base)
            except ValueError:  # Another drive
                folder = os.pardir
        if folder.startswith(os.pardir) or os.path.isabs(folder):
            return ValueError(f"{field.label}: a folder inside the folder or the project (relative to each project)")
        return None if folder == os.curdir else folder.replace(os.sep, "/")

    @staticmethod
    def convert(field, raw):
        if field.kind == "int":
            return int(float(raw))
        if field.kind == "float":
            return float(raw)
        if field.kind == "optional":
            return float(raw) if str(raw).strip() else None
        if field.kind == "text":
            return str(raw).strip() or None
        return raw


BOLD = ("TkDefaultFont", 9, "bold")


class PanelPage:
    """What the side panel of a page tells its page, in the window Batch: its changes, to the form (PanelForm.commit);
    the other calls of the panel (those of the page only) do nothing."""

    def __init__(self, form):
        self.form = form

    def commit(self, *args):
        self.form.commit()
        return True

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class PanelForm(ttk.Frame):
    """The settings of the node selected in the window Batch, in the side panel of the page of the step (its batch
    mode: the boxes of its settings, not those of the page only), and the others in a SettingsForm above it (those of
    the window: the export folder...). A change in the panel is told for each setting changed (changed(field,
    value)); the labels of the settings that differ between the projects of a folder are in bold, each with its
    button ↺ giving them all the value shown (unify; the choices of a field "multi" one by one: choice). The interface
    of SettingsForm (show, flush, canvas).

    A subclass gives PATHS (the first keys of the settings in the panel), LABELS ({label of the panel: [paths]}),
    CHOICES ({label of the panel: (path of a field "multi", its choice)}: in bold when that choice differs),
    make_panel() (the panel), load_panel(settings, unavailable, limits) (the settings of the node in it),
    restore_states() (its own greyed widgets once enabled), record() (its settings, ValueError if invalid) and
    normalize(field, value) (a value compared: those entered and saved compare)."""

    # PRIMARY: the paths whose change alone is told when it changes with others (the software: the settings of the
    # previous one left out)
    PATHS, LABELS, PRIMARY, CHOICES = (), {}, (), {}

    def __init__(self, master, changed, unify=None, choice=None):
        super().__init__(master)
        self.changed_callback, self.unify_callback, self.choice_callback = changed, unify, choice
        self.extra = SettingsForm(self, changed, unify, choice)
        self.extra.canvas.configure(height=58)
        self.extra.pack(side=tk.TOP, fill=tk.X)
        self.fields, self.shown, self.loading, self.enabled, self.differs = {}, {}, False, True, set()
        self.differ_choices, self.unify_buttons = {}, []
        self.panel = self.make_panel()
        self.panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = self.panel.canvas
        self.labels = self.find_labels(self.panel.inner)

    def find_labels(self, widget):
        """The labels of the settings of the panel: {path, or (path, choice) for a choice of a field "multi": [label
        or check button]}."""
        out = {}
        for child in widget.winfo_children():
            kind = child.winfo_class()
            text = str(child.cget("text")) if kind in ("TLabel", "TCheckbutton", "Checkbutton") else None
            for path in self.LABELS.get(text, ()):
                out.setdefault(path, []).append(child)
            if text in self.CHOICES:
                out.setdefault(self.CHOICES[text], []).append(child)
            for path, found in self.find_labels(child).items():
                out.setdefault(path, []).extend(found)
        return out

    def in_panel(self, field):
        return field.path[0] in self.PATHS

    def show(self, fields, settings, enabled=True, base=None, differs=(), differ_choices=None, unavailable=None,
             limits=None):
        self.fields = {f.path: f for f in fields if self.in_panel(f)}
        self.extra.show([f for f in fields if not self.in_panel(f)], settings, enabled, base, differs, differ_choices,
                        unavailable, limits)
        self.shown, self.enabled = copy.deepcopy(settings), enabled
        self.differs = {f.path for f in self.fields.values()
                        if any(d[:len(p)] == p for d in differs for p in (f.path, f.end) if p)}
        self.differ_choices = {path: set(choices) for path, choices in (differ_choices or {}).items()
                               if path in self.fields}
        self.load(settings, unavailable or {}, limits or {})

    def load(self, settings, unavailable=None, limits=None):
        """The settings of the node in the panel (and the labels of those that differ in bold)."""
        self.loading = True
        try:
            self.load_panel(settings, unavailable, limits)
            self.set_enabled(self.panel.inner, self.enabled)
            if self.enabled:
                self.restore_states(unavailable)
            self.labels = self.find_labels(self.panel.inner)  # Those of the widgets built again (another software)
            for button in self.unify_buttons:
                button.destroy()
            self.unify_buttons = []
            for label, widgets in self.labels.items():
                bold, command = self.differing(label)
                for n, widget in enumerate(widgets):
                    if widget.winfo_class() in ("TLabel", "Checkbutton"):
                        widget.configure(font=BOLD if bold else "TkDefaultFont")
                    else:
                        widget.configure(style="Bold.TCheckbutton" if bold else "TCheckbutton")
                    if bold and n == 0 and command is not None and self.enabled:
                        self.place_unify(widget, command)
        finally:
            self.loading = False

    def differing(self, label):
        """A setting of the panel (path, or (path, choice) of a field "multi"): whether it differs between the
        projects of the node, and its button ↺ (the value shown for all of them; None: none)."""
        if len(label) == 2 and isinstance(label[0], tuple):  # A choice of a field "multi"
            path, choice = label
            field = self.fields.get(path)
            if field is None or choice not in self.differ_choices.get(path, ()):
                return False, None
            on = choice in (get_path(self.shown, path, field.default) or [])
            return True, (None if self.choice_callback is None
                          else lambda: self.choice_callback(field, choice, on))
        field = self.fields.get(label)
        if field is None or label not in self.differs or field.kind == "multi":
            return False, None
        return True, (None if self.unify_callback is None else lambda: self.unify_callback(field))

    def place_unify(self, widget, command):
        """The button ↺ of a setting, next to its label: in the column after its help (a grid), after it (a row), or
        at its right (alone in its box)."""
        parent = widget.nametowidget(widget.winfo_parent())
        button = ttk.Button(parent, text="↺", width=3, command=command)
        manager = widget.winfo_manager()
        if manager == "grid":
            button.grid(row=int(widget.grid_info()["row"]), column=3, padx=(4, 0))
        elif manager == "pack" and widget.pack_info().get("side") == "left":
            button.pack(side=tk.LEFT, padx=(4, 0), after=widget)
        else:
            button.place(in_=widget, relx=1.0, x=6, rely=0.5, anchor=tk.W)
        self.unify_buttons.append(button)

    def set_enabled(self, widget, on):
        """The widgets of the panel enabled or not (its own states restored afterwards: restore_states)."""
        for child in widget.winfo_children():
            if isinstance(child, RangeSlider):
                child.set_enabled(on)
            elif child.winfo_class().startswith("T") and hasattr(child, "state"):
                child.state(["!disabled"] if on else ["disabled"])
            elif child.winfo_class() in ("Checkbutton", "Button", "Spinbox"):
                child.configure(state=tk.NORMAL if on else tk.DISABLED)
            self.set_enabled(child, on)

    def values(self, settings, field):
        value = get_path(settings, field.path, field.default)
        if field.kind == "range":
            return value, get_path(settings, field.end) or value
        return self.normalize(field, value)

    def commit(self):
        """The panel changed: each setting changed told (changed(field, value)); invalid values said, the values of
        the node shown again."""
        if self.loading or not self.enabled or not self.fields:
            return
        try:
            new = self.record()
        except ValueError as e:
            self.changed_callback(next(iter(self.fields.values())), ValueError(str(e)))
            self.load(self.shown)
            return
        changes = [(field, get_path(new, field.path, field.default) if field.kind != "range"
                    else self.values(new, field)) for field in self.fields.values()
                   if self.values(new, field) != self.values(self.shown, field)]
        primary = [(field, value) for field, value in changes if field.path in self.PRIMARY]
        for field, value in primary or changes:
            self.changed_callback(field, value)
        # The settings of the node as kept by the window (those refused not kept), what the next changes compare with
        self.shown = copy.deepcopy(getattr(self.winfo_toplevel(), "shown", self.shown))

    def flush(self):
        self.extra.flush()
        self.commit()

    def changed(self, field, raw):
        """A setting of the form above the panel entered (as SettingsForm.changed)."""
        return self.extra.changed(field, raw)

    # The rows and the widgets of the form above the panel (those of a SettingsForm)
    inner = property(lambda self: self.extra.inner)
    widgets = property(lambda self: self.extra.widgets)

    def unify_all(self):
        """The settings of the panel that differ between the projects of the node: the values shown for all."""
        for path in sorted(self.differs):
            if self.fields[path].kind != "multi":
                self.unify_callback(self.fields[path])
        for path, choices in self.differ_choices.items():
            for choice in sorted(choices):
                self.choice_callback(self.fields[path], choice,
                                     choice in (get_path(self.shown, path, self.fields[path].default) or []))

    # The subclasses
    def make_panel(self):
        raise NotImplementedError

    def load_panel(self, settings, unavailable, limits):
        raise NotImplementedError

    def restore_states(self, unavailable):
        pass

    def record(self):
        raise NotImplementedError

    @staticmethod
    def normalize(field, value):
        return value


class BatchStep:
    """What a batch runs. key: the settings of the step; job: the step of batch_run.py (key by default);
    usable(project): it can run the project; interface(): the settings of the interface (those of the project open);
    describe(settings): them in a line; job_settings(project, settings, settings_of): the settings of the job of a
    project (settings_of(key, project): the settings of another step for it); options: the common options
    (build(parent, saved), read(): ValueError if invalid, job(projects): the options of the job, below the node
    settings); fields(settings): the fields of the form of the settings (Field); validate(settings): settings checked (ValueError with a message), or normalized (returns them);
    panels: [(title, key, factory(master, on_change))] editors of a part of the settings (settings[key]), shown in a
    column: their show(value, settings, enabled) and on_change(value, replace=False) at each change (the quantities of
    the post-processing; replace: the whole value for all the projects of the node, not only what changed; their
    context(project, settings, items), if they have one: the first project of the node and its settings, and all of
    them, [(project, settings)]); unavailable(items): {path: reason} of the fields that the projects of a node (items)
    can't use, greyed out (None: none); limits(items): {path: (lo, hi)} of their fields "range" (None: none); form: the
    form of the settings of a node (SettingsForm by default: its interface, and its arguments); confirm(dialog,
    entries, options): asked before the run (the results it would overwrite), the options it adds to the job (None:
    the run cancelled). The
    settings of each project: in its project file (to_project: (section, merge(section,
    settings): the section with the settings); from_project(section): the settings of a section), else (the archive)
    in the workspace (its batch settings, by project)."""

    def __init__(self, key, title, usable, interface, describe, job_settings, job=None, options=None, fields=None,
                 validate=None, panels=(), to_project=None, from_project=None, unavailable=None, limits=None,
                 form=None, confirm=None):
        self.key, self.title, self.usable, self.interface, self.describe = key, title, usable, interface, describe
        self.job_settings, self.job, self.options = job_settings, job or key, options
        self.fields, self.validate, self.panels = fields, validate, panels
        self.to_project, self.from_project = to_project, from_project
        self.unavailable, self.limits, self.form, self.confirm = unavailable, limits, form, confirm


# A value missing (flatten, changes)
MISSING = object()


def flatten(settings, prefix=()):
    """The values of settings by path (the dictionaries opened, the other values as they are, the numbers as
    floats, the numbers in texts too, so that 6, 6.0 and "6" are the same)."""
    out = {}
    # A dictionary: its values (an empty one: none, so that adding to it or emptying it changes its values only)
    if isinstance(settings, dict):
        for k, v in settings.items():
            out.update(flatten(v, prefix + (k,)))
        return out
    value = settings
    if isinstance(value, bool) or value is None:
        pass
    elif isinstance(value, (int, float)):
        value = float(value)
    elif isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    elif isinstance(value, tuple):
        value = list(value)
    out[prefix] = value
    return out


def changes(old, new):
    """What changed from old to new (settings): [(path, value)] the values set or changed, [path] those removed."""
    before, after = flatten(old), flatten(new)
    changed = [(path, get_path(new, path)) for path, v in after.items() if before.get(path, MISSING) != v]
    return changed, [path for path in before if path not in after and path]


def del_path(settings, path):
    """A value removed from settings, and the dictionaries it leaves empty (an entry removed, not emptied)."""
    target = get_path(settings, path[:-1]) if len(path) > 1 else settings
    if isinstance(target, dict):
        target.pop(path[-1], None)
        if not target and len(path) > 1:
            del_path(settings, path[:-1])


class BatchDialog(Dialog):
    """Choice of the projects of a batch and of their settings (step: BatchStep). The result: the job, None if
    cancelled."""

    def __init__(self, parent, app, step):
        super().__init__(parent, f"Batch: {step.title}")
        self.resizable(True, True)
        self.unbind("<Return>")  # Enter doesn't run the batch
        self.app, self.step, self.ws = app, step, app.workspace.ws
        self.items = {}
        self.checked = {key(f) for f in self.ws.checked.get(step.job, [])}
        # The settings of the projects (by folder), those changed to be saved (soon: sync_job, or when the window
        # closes), the settings shown for the node selected, the project open changed (its page at the end)
        self.cache, self.dirty, self.sync_job, self.shown, self.current_changed = {}, set(), None, {}, False
        # Cached: the settings compared of each project (flat) and of the reference, the paths of each project only,
        # the settings of the interface (the window is modal: it doesn't change)
        self.flat, self.reference_flat, self.own_paths, self.interface_settings = {}, None, None, None
        frame = self.frame
        top = ttk.Frame(frame)
        top.pack(fill=tk.X)
        ttk.Label(top, text=f"{step.title}: the projects checked are run, each with its settings.",
                  font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
        HelpDot(top, HELP).pack(side=tk.LEFT, padx=6)
        body = ttk.Frame(frame)  # Packed after the buttons below it, so that they stay visible
        # A column of the options of the step (options.column, column_width: the stitching of the post-processing;
        # options.prepare(dialog): before the tree is refreshed, options.cell(dialog, iid): the text of a node;
        # options.ctrl_click(dialog, iid): Ctrl+click on a node, done or not), the options attached to the window
        self.extra = getattr(step.options, "column", None)
        if hasattr(step.options, "attach"):
            step.options.attach(self)
        self.tree = ttk.Treeview(body, columns=("run", "extra", "settings") if self.extra else ("run", "settings"),
                                 height=24, selectmode="browse")
        self.tree.heading("#0", text="Project", anchor=tk.W)
        self.tree.heading("run", text="Run")
        if self.extra:
            self.tree.heading("extra", text=self.extra)
            self.tree.column("extra", width=getattr(step.options, "column_width", 72))
        self.tree.heading("settings", text="Differences vs. Open Project", anchor=tk.W)
        self.tree.column("#0", width=150 if self.extra else 170)
        self.tree.column("run", width=44, anchor=tk.CENTER, stretch=False)
        self.tree.column("settings", width=170)
        # The columns: the tree, the options (if any), the node selected with its settings, the panels
        columns = ttk.PanedWindow(body, orient=tk.HORIZONTAL)
        columns.pack(fill=tk.BOTH, expand=True)
        # The options of the run: below the tree (options.below_tree), else below the node settings
        below_tree = step.options is not None and getattr(step.options, "below_tree", False)
        tree_column = ttk.PanedWindow(columns, orient=tk.VERTICAL)
        columns.add(tree_column, weight=2)
        cell = ttk.Frame(tree_column)
        tree_column.add(cell, weight=3)
        self.tree.pack(in_=cell, side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar = ttk.Scrollbar(cell, command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        bar.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.lift(cell)
        self.tree.tag_configure("grey", foreground=GREY)
        self.tree.tag_configure("lost", foreground=LOST)
        self.tree.bind("<Button-1>", self.click, add="+")
        self.tree.bind("<Control-Button-1>", self.ctrl_click)
        # The menu of a node: options.context_items(dialog, iid): [(label, command or None: greyed)]
        if hasattr(step.options, "context_items"):
            self.tree.bind("<Button-3>", self.context_menu)
        # The text of the column of the options in full while the mouse is over it (it may be cut)
        self.cell_tip, self.cell_tip_at = None, None
        if self.extra:
            self.tree.bind("<Motion>", self.show_cell_tip, add="+")
            self.tree.bind("<Leave>", lambda event: self.hide_cell_tip(), add="+")
        self.tree.bind("<space>", lambda event: self.toggle(self.tree.focus()))
        self.tree.bind("<<TreeviewSelect>>", lambda event: self.show_node())
        # The node settings, the common options of the run below them (a split moved to share the height)
        node_column = ttk.PanedWindow(columns, orient=tk.VERTICAL)
        columns.add(node_column, weight=1)
        tab = ttk.LabelFrame(node_column, text="Node settings", padding=8)
        node_column.add(tab, weight=3)
        self.build_node_pane(tab)
        if step.options is not None:
            # Its title: options.title (the stitching: "Stitch matching")
            split = tree_column if below_tree else node_column
            options_tab = ttk.LabelFrame(split, text=getattr(step.options, "title", "Options (common to the run)"),
                                         padding=8)
            split.add(options_tab, weight=1)
            step.options.build(options_tab, self.ws.options.get(step.job, {}))
            # The split at first shown: the options in full (the tree or the node settings scrolled above them)
            def fit(event):
                split.unbind("<Configure>")
                split.sashpos(0, max(event.height - options_tab.winfo_reqheight() - 6, event.height // 3))
            split.bind("<Configure>", fit)
        self.panels = []
        for title, panel_key, factory in step.panels:
            frame_ = ttk.LabelFrame(columns, text=title, padding=6)
            columns.add(frame_, weight=2)
            widget = factory(frame_, lambda value, replace=False, panel_key=panel_key: self.panel_changed(
                panel_key, value, replace))
            widget.pack(fill=tk.BOTH, expand=True)
            self.panels.append((panel_key, widget))
        width = 560 + 420 + 780 * bool(step.panels)
        self.geometry(f"{min(width, self.winfo_screenwidth() - 40)}x{min(880, self.winfo_screenheight() - 80)}")
        buttons = ttk.Frame(frame)
        buttons.pack(side=tk.BOTTOM, anchor=tk.E, pady=(10, 0))
        row = ttk.Frame(frame)
        row.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        body.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        ttk.Button(row, text="Check all", command=lambda: self.check_all(True)).pack(side=tk.LEFT)
        ttk.Button(row, text="Uncheck all", command=lambda: self.check_all(False)).pack(side=tk.LEFT, padx=(6, 0))
        self.reset_all_button = ttk.Button(row, text="Reset all to the open project", command=self.reset_all)
        self.reset_all_button.pack(side=tk.LEFT, padx=(18, 0))
        self.summary = ttk.Label(row, foreground="#555555")
        self.summary.pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(buttons, text="Run", command=self.ok).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Close", command=self.close).pack(side=tk.LEFT)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda event: self.close())  # The settings saved, as with Close
        self.fill()

    # ------------------------------------------------------------------ tree

    @staticmethod
    def node_id(root, path=""):
        return f"{key(root)}|{path}"

    def fill(self):
        """The tree of the workspace: roots, folders, projects."""
        for r in self.ws.roots:
            iid = self.node_id(r["folder"])
            self.tree.insert("", tk.END, iid=iid, text=os.path.basename(r["folder"]) or r["folder"], open=True)
            self.items[iid] = ("root", r["folder"], "")
        for p in self.ws.projects:
            parent = self.node_id(p.root)
            if not self.tree.exists(parent):
                continue
            parts = p.path.split("/") if p.path else []
            for i in range(len(parts) - 1):
                path = "/".join(parts[:i + 1])
                iid = self.node_id(p.root, path)
                if not self.tree.exists(iid):
                    self.tree.insert(parent, tk.END, iid=iid, text=parts[i], open=True)
                    self.items[iid] = ("node", p.root, path)
                parent = iid
            iid = "project|" + key(p.folder)
            self.tree.insert(parent, tk.END, iid=iid, text=p.label())
            self.items[iid] = ("project", p.root, p.path, p)
        self.refresh()
        first = next(iter(self.items), None)
        if first:
            self.tree.selection_set(first)
            self.tree.focus(first)

    def projects_below(self, iid):
        item = self.items.get(iid)
        if item is None:
            return []
        if item[0] == "project":
            return [item[3]]
        out = []
        for child in self.tree.get_children(iid):
            out += self.projects_below(child)
        return out

    def runnable(self, project):
        return not project.lost and self.step.usable(project)

    def targets(self, iid):
        """The projects of a node whose settings are shown and changed: those below it that the step can run."""
        return [p for p in self.projects_below(iid) if self.runnable(p)]

    def node_label(self, root, path):
        return path.rsplit("/", 1)[-1] if path else os.path.basename(root) or root

    def project_paths(self):
        """The paths of the settings of each project only (Field.project_only: the export paths)."""
        if self.step.fields is None:
            return []
        if self.own_paths is None:
            try:
                self.own_paths = [f.path for f in self.step.fields(self.reference()) if f.project_only]
            except (KeyError, TypeError, ValueError):
                return []
        return self.own_paths

    def compared(self, settings):
        """The values of settings compared between the projects (flatten), those of each project only left out."""
        own = self.project_paths()
        return {path: v for path, v in flatten(settings).items() if not any(path[:len(p)] == p for p in own)}

    def compared_of(self, project):
        """compared of the settings of a project (cached until they change: store)."""
        k = key(project.folder)
        if k not in self.flat:
            self.flat[k] = self.compared(self.settings_of(project))
        return self.flat[k]

    def compared_reference(self):
        if self.reference_flat is None:
            self.reference_flat = self.compared(self.reference())
        return self.reference_flat

    def store(self, new):
        """Settings of projects changed (new: by folder): kept, their caches dropped, saved soon."""
        self.cache.update(new)
        for k in new:
            self.flat.pop(k, None)
        self.reference_flat = None
        self.changed(new)

    def differences(self, iid):
        """The column of a node: how its settings differ from those of the project open (the settings of each
        project only left out: its export path)."""
        projects = self.targets(iid)
        if not projects:
            return ""
        reference = self.compared_reference()
        item = self.items[iid]
        if item[0] == "project":
            if self.is_open(item[3]):
                return "Open project"
            settings = self.compared_of(item[3])
            n = sum(settings.get(path, MISSING) != v for path, v in reference.items())
            n += sum(path not in reference for path in settings)
            return "None" if not n else f"{n} setting{'s differ' if n > 1 else ' differs'}"
        differ = [p for p in projects if not self.is_open(p) and self.compared_of(p) != reference]
        return f"{len(differ)} of {len(projects)} differ" if differ else "None"

    def refresh(self, form=True):
        """The columns Run and Differences (and that of the options), the colours, the node selected (its form too)
        and the summary."""
        if self.extra and hasattr(self.step.options, "prepare"):
            self.step.options.prepare(self)
        for iid, item in self.items.items():
            below = self.projects_below(iid)
            runnable = [p for p in below if self.runnable(p)]
            checked = [p for p in runnable if key(p.folder) in self.checked]
            run = "☑" if runnable and len(checked) == len(runnable) else "◩" if checked else "☐" if runnable else ""
            tags = ["lost"] if below and all(p.lost for p in below) else [] if runnable else ["grey"]
            values = (run, self.step.options.cell(self, iid)) if self.extra else (run,)
            self.tree.item(iid, values=values + (self.differences(iid),), tags=tags)
        # Greyed when all the projects have the settings of the project open
        self.reset_all_button.state(["disabled"] if self.same_as_open([p for p in self.ws.projects if self.runnable(p)])
                                    else ["!disabled"])
        n = len(self.chosen())
        greyed = sum(not self.runnable(p) for p in self.ws.projects)
        self.summary["text"] = f"{n} project(s) to run" + (f" ({greyed} can't be run)" if greyed else "")
        self.show_node(form)

    def chosen(self):
        return [p for p in self.ws.projects if self.runnable(p) and key(p.folder) in self.checked]

    def click(self, event):
        column, iid = self.tree.identify_column(event.x), self.tree.identify_row(event.y)
        if iid and column == "#1":
            self.toggle(iid)

    def ctrl_click(self, event):
        """Ctrl+click on a node: to the options (the stitching), the node selected kept."""
        iid = self.tree.identify_row(event.y)
        if iid and hasattr(self.step.options, "ctrl_click"):
            if not self.step.options.ctrl_click(self, iid):
                self.bell()
            self.refresh(form=False)
        return "break"

    def context_menu(self, event):
        iid = self.tree.identify_row(event.y)
        items = self.step.options.context_items(self, iid) if iid else []
        if not items:
            return
        menu = tk.Menu(self, tearoff=0)
        for label, command in items:
            menu.add_command(label=label, command=command, state=tk.NORMAL if command else tk.DISABLED)
        menu.tk_popup(event.x_root, event.y_root)

    def show_cell_tip(self, event):
        iid = self.tree.identify_row(event.y)
        text = self.tree.set(iid, "extra") if iid and self.tree.identify_column(event.x) == "#2" else ""
        if (iid, text) == self.cell_tip_at:
            return
        self.hide_cell_tip()
        if not text:
            return
        self.cell_tip_at = (iid, text)
        self.cell_tip = tk.Toplevel(self)
        self.cell_tip.wm_overrideredirect(True)
        self.cell_tip.attributes("-topmost", True)
        tk.Label(self.cell_tip, text=text, background="#ffffe8", foreground="#222222", relief=tk.SOLID, borderwidth=1,
                 padx=6, pady=3).pack()
        self.cell_tip.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 16}")

    def hide_cell_tip(self):
        if self.cell_tip is not None:
            self.cell_tip.destroy()
        self.cell_tip, self.cell_tip_at = None, None

    def toggle(self, iid):
        """Check or uncheck a project, or all those below a folder."""
        runnable = [p for p in self.projects_below(iid) if self.runnable(p)]
        on = not all(key(p.folder) in self.checked for p in runnable)
        for p in runnable:
            (self.checked.add if on else self.checked.discard)(key(p.folder))
        self.refresh(form=False)

    def check_all(self, on):
        self.checked = {key(p.folder) for p in self.ws.projects if self.runnable(p)} if on else set()
        self.refresh(form=False)

    # ------------------------------------------------------------------ the settings of the projects

    def is_open(self, project):
        current = self.app.project_folder()
        return bool(current) and key(project.folder) == key(current)

    def open_project(self):
        return next((p for p in self.ws.projects if self.is_open(p)), None)

    def load(self, step, project):
        """The settings of a step for a project: those of its project file (the project open: those of the
        interface), else (the archive) those kept in the workspace; without any, those of the interface."""
        if step.to_project is not None:
            if self.is_open(project):
                return step.interface()
            try:
                section = (read_project(project.folder) or {}).get(step.to_project[0])
            except (OSError, ValueError):
                section = None
            if section:
                try:
                    return step.from_project(section)
                except (KeyError, TypeError, ValueError):
                    pass
            return step.interface()
        entry = self.ws.node_settings(step.key, project.root, project.path)
        return copy.deepcopy(entry["settings"]) if entry and entry.get("settings") else step.interface()

    def settings_of(self, project):
        """The settings of the step for a project (as changed here), completed (validate), so that they compare with
        those of the panels."""
        k = key(project.folder)
        if k not in self.cache:
            settings = self.load(self.step, project)
            if self.step.validate is not None:
                try:
                    settings = self.step.validate(settings) or settings
                except (KeyError, TypeError, ValueError):
                    pass
            self.cache[k] = settings
        return self.cache[k]

    def reference(self):
        """The settings of the project open (as changed here), else of the interface."""
        project = self.open_project()
        if project is not None:
            return self.settings_of(project)
        if self.interface_settings is None:
            self.interface_settings = self.step.interface()
        return self.interface_settings

    def settings_of_step(self, step_key, project):
        """The settings of a step for a project (settings_of of job_settings): of this step, or of another one."""
        if step_key == self.step.key:
            return copy.deepcopy(self.settings_of(project))
        return self.load(self.app.batch_step(step_key), project)

    # ------------------------------------------------------------------ the node selected

    def build_node_pane(self, tab):
        self.node_title = ttk.Label(tab, font=("TkDefaultFont", 9, "bold"), wraplength=360)
        self.node_title.pack(anchor=tk.W)
        self.node_text = ttk.Label(tab, wraplength=360, justify=tk.LEFT)
        self.node_text.pack(anchor=tk.W, pady=(4, 0))
        self.reset_button = ttk.Button(tab, command=self.reset_node)
        self.reset_button.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))
        # The settings of the reference of the stitching of each project (options with stitchings)
        self.stitch_button = None
        if hasattr(self.step.options, "reference_of"):
            self.stitch_button = ttk.Button(tab, text="Reset to the stitch reference project",
                                            command=lambda: self.copy_stitch_reference())
            self.stitch_button.pack(anchor=tk.W, fill=tk.X, pady=(4, 0))
        self.node_note = ttk.Label(tab, foreground="#a33", wraplength=360)
        self.node_note.pack(anchor=tk.W, pady=(6, 0))
        self.form = None
        if self.step.fields is not None:
            ttk.Label(tab, text="Settings",
                      wraplength=360, font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 2))
            self.form = (self.step.form or SettingsForm)(tab, self.field_changed, self.unify_field, self.choice_changed)
            self.form.pack(fill=tk.BOTH, expand=True)

    def node(self):
        item = self.items.get(self.tree.focus())
        return (item[0], item[1], item[2]) if item else (None, None, None)

    def node_folder(self, kind, root, path):
        """The folder of a node: that of the project, or of the folder of the tree."""
        if kind == "project":
            project = next((p for p in self.ws.projects if key(p.root) == key(root) and p.path == path), None)
            if project is not None:
                return project.folder
        return os.path.join(root, *path.split("/")) if path else root

    def show_node(self, form=True):
        iid = self.tree.focus()
        kind, root, path = self.node()
        if kind is None:
            return
        projects = self.targets(iid)
        name = self.node_label(root, path)
        self.node_title["text"] = (f"Project {path or name}" if kind == "project" else f"Folder {path or root}")
        opened = self.open_project()
        self.reset_button.configure(text=f"Reset to the settings of the open project ({opened.label()})"
                                    if opened else "Reset to the settings of the interface")
        # Greyed without projects, for the project open (its settings: those of the interface) and when they all have
        # its settings already; the stitch reference likewise
        self.reset_button.state(["!disabled"] if projects and not self.same_as_open(projects) else ["disabled"])
        if self.stitch_button is not None:
            self.stitch_button.state(["!disabled"] if self.stitch_references(projects, differing=True)
                                     else ["disabled"])
        first = self.settings_of(projects[0]) if projects else {}
        others = [self.compared_of(p) for p in projects[1:]]
        flat = self.compared_of(projects[0]) if projects else {}
        differs = {path for f in others for path in set(flat) | set(f) if flat.get(path) != f.get(path)}
        try:
            text = cap(self.step.describe(first)) if projects else "No project of this step below it"
        except (KeyError, TypeError, ValueError) as e:
            text = f"(Settings not readable: {e})"
        if kind != "project" and len(projects) > 1:
            n = len(differs)
            text += (f"\n(Shown: those of {projects[0].label()}; "
                     + (f"{n} setting{'s differ' if n > 1 else ' differs'} between the {len(projects)} projects)"
                        if differs else f"the same for its {len(projects)} projects)"))
        self.node_text["text"] = text
        if not form:
            return
        # A field changed in the form of the previous node is applied to it first
        if self.form is not None:
            self.form.flush()
        self.form_node = iid
        self.node_note["text"] = ""
        self.shown = copy.deepcopy(first)
        try:
            if self.form is not None:
                # A folder: the settings of its projects but those of each project only (their export paths)
                fields = [f for f in self.step.fields(first) if kind == "project" or not f.project_only]
                # The choices of the fields "multi" that differ between the projects
                differ_choices = {}
                for f in fields:
                    if f.kind == "multi" and len(projects) > 1:
                        chosen = [set(get_path(self.settings_of(p), f.path, f.default) or []) for p in projects]
                        differ_choices[f.path] = set.union(*chosen) - set.intersection(*chosen)
                items = [(p, self.settings_of(p)) for p in projects]
                unavailable = self.step.unavailable(items) if self.step.unavailable is not None and projects else None
                limits = self.step.limits(items) if self.step.limits is not None and projects else None
                self.form.show(fields if projects else [], first, enabled=bool(projects),
                               base=self.node_folder(kind, root, path), differs=differs,
                               differ_choices=differ_choices, unavailable=unavailable, limits=limits)
            self.show_panels(projects[0] if projects else None, first, bool(projects))
        except (KeyError, TypeError, ValueError) as e:
            self.node_note["text"] = f"The settings can't be shown: {e}"

    def show_panels(self, project, settings, enabled=True):
        """The panels with the settings of the node (their context: the first project and its settings)."""
        items = [(p, self.settings_of(p)) for p in self.targets(getattr(self, "form_node", None))] if project else []
        for panel_key, widget in self.panels:
            if hasattr(widget, "context"):
                widget.context(project, settings, items)
            widget.show(settings.get(panel_key), settings, enabled=enabled)

    def apply(self, change):
        """A change (change(settings): the settings of a project changed in place) to the projects of the node of
        the form: checked (validate) for all of them before it is kept; saved soon."""
        return self.apply_to(self.targets(getattr(self, "form_node", None)), change)

    def apply_to(self, projects, change):
        """A change to projects (apply; the alignment of a stitching: their offsets)."""
        new = {}
        try:
            for p in projects:
                settings = copy.deepcopy(self.settings_of(p))
                change(settings)
                if self.step.validate is not None:
                    settings = self.step.validate(settings) or settings
                new[key(p.folder)] = settings
        except (TypeError, ValueError) as e:
            self.node_note["text"] = f"Not valid: {e}"
            self.bell()
            return False
        self.store(new)
        return True

    def field_changed(self, field, value):
        """A field of the form changed: for the project selected, or for all the projects of the folder (that
        setting only, their other settings kept)."""
        if isinstance(value, ValueError):
            self.node_note["text"] = str(value)
            self.bell()
            return
        before = self.field_paths(self.shown)

        def change(settings):
            if field.kind == "range":  # Its first and its last values
                set_path(settings, field.path, value[0])
                set_path(settings, field.end, value[1])
            else:
                set_path(settings, field.path, value)
        if self.apply(change):
            change(self.shown)
            self.node_note["text"] = ""
            self.refresh(form=False)
            if self.field_paths(self.shown) != before:
                # Other fields (another PIV software): the form of the node again
                self.show_node_kept()
                return
            # The panels follow the settings (the videos of every quantity with the rolling statistics, the maps
            # with the camera)
            projects = self.targets(self.form_node)
            self.show_panels(projects[0] if projects else None, self.shown, bool(projects))

    def field_paths(self, settings):
        try:
            return [f.path for f in self.step.fields(settings)] if self.step.fields is not None else []
        except (KeyError, TypeError, ValueError):
            return []

    def unify_field(self, field):
        """The button of a setting not in common: the value shown for all the projects of the node."""
        value = get_path(self.shown, field.path, field.default)
        if field.kind == "range":
            value = (value, get_path(self.shown, field.end) or value)
        self.field_changed(field, value)
        self.show_node_kept()

    def choice_changed(self, field, choice, on):
        """A choice of a field "multi" checked or not (or its button ↺: the state shown): for each project of the
        node, their other choices kept."""
        def change(settings):
            chosen = set(get_path(settings, field.path, field.default) or [])
            chosen = chosen | {choice} if on else chosen - {choice}
            set_path(settings, field.path, [k for k, _ in field.choices if k in chosen])
        if self.apply(change):
            self.node_note["text"] = ""
            self.refresh(form=False)
            self.show_node_kept()

    def show_node_kept(self):
        """The node shown again (its bold settings), the form scrolled where it was."""
        top = self.form.canvas.yview()[0] if self.form is not None else 0
        self.show_node()
        if self.form is not None:
            self.form.canvas.update_idletasks()
            self.form.canvas.yview_moveto(top)

    def panel_changed(self, panel_key, value, replace=False):
        """A panel (the quantities) changed: what changed in it, for the project selected or all the projects of
        the folder (their other settings kept); replace: its whole value for all of them (the same as the project
        open), the panels shown again."""
        if replace:
            def replace_part(settings):
                settings[panel_key] = copy.deepcopy(value)
            if self.apply(replace_part):
                self.shown[panel_key] = copy.deepcopy(value)
                self.refresh(form=False)
                projects = self.targets(self.form_node)
                self.show_panels(projects[0] if projects else None, self.shown, bool(projects))
            return
        changed, removed = changes(self.shown.get(panel_key) or {}, value or {})

        def change(settings):
            part = settings.get(panel_key)
            part = copy.deepcopy(part) if isinstance(part, dict) else {}
            for path, v in changed:
                set_path(part, path, copy.deepcopy(v))
            for path in removed:
                del_path(part, path)
            settings[panel_key] = part
        if (changed or removed) and self.apply(change):
            self.shown[panel_key] = copy.deepcopy(value)
            self.refresh(form=False)

    def changed(self, new):
        """Settings of projects changed (new: by folder): saved soon."""
        self.dirty |= set(new)
        if self.sync_job is None:
            self.sync_job = self.after(600, self.sync)

    def reset_to(self, projects):
        """Projects with the settings of the project open (of the interface), each keeping its settings of each
        project only (its export path)."""
        reference, own = self.reference(), self.project_paths()
        new = {}
        for p in projects:
            settings = copy.deepcopy(reference)
            for path in own:
                set_path(settings, path, copy.deepcopy(get_path(self.settings_of(p), path)))
            new[key(p.folder)] = settings
        self.store(new)
        self.refresh()

    def same_as_open(self, projects):
        """All the projects (but the project open) have the settings of the project open (those of each project only
        left out): resetting them to it changes nothing."""
        reference = self.compared_reference()
        return all(self.is_open(p) or self.compared_of(p) == reference for p in projects)

    def stitch_references(self, projects, differing=False, part=None):
        """The projects stitched to a reference project (not the references): {key: (project, reference)}; none
        without stitchings. differing: only those whose settings (part: settings[part] only) differ from those of
        their reference, but those of each project and of its field of view (own_paths: the camera)."""
        options = self.step.options
        if not hasattr(options, "reference_of"):
            return {}
        own = self.project_paths() + [tuple(p) for p in getattr(options, "own_paths", ())]

        def compared(project):
            settings = self.settings_of(project)
            if part is not None:
                return settings.get(part)
            return {k: v for k, v in self.compared(settings).items() if not any(k[:len(o)] == o for o in own)}
        out = {}
        for p in projects:
            reference = options.reference_of(p)
            if reference is not None and key(reference.folder) != key(p.folder):
                if not differing or compared(p) != compared(reference):
                    out[key(p.folder)] = (p, reference)
        return out

    def copy_stitch_reference(self, part=None):
        """The projects of the node selected stitched to a reference: the settings of the reference of their
        stitching, each keeping its settings of each project only (its export path) and those of its field of view
        (options.own_paths: its camera); part: only settings[part] (the quantities exported). Returns the number of
        projects changed."""
        pairs = self.stitch_references(self.targets(getattr(self, "form_node", None) or self.tree.focus()),
                                       differing=True, part=part)
        if not pairs:
            self.bell()
            self.node_note["text"] = ("No project of this node is stitched to a reference project with other "
                                      + ("quantities exported" if part == "export" else "settings"))
            return 0
        own = self.project_paths() + [tuple(p) for p in getattr(self.step.options, "own_paths", ())]
        new = {}
        try:
            for k, (p, reference) in pairs.items():
                source, mine = self.settings_of(reference), self.settings_of(p)
                if part is None:
                    settings = copy.deepcopy(source)
                    for path in own:
                        value = get_path(mine, path, MISSING)
                        if value is MISSING:
                            del_path(settings, path)
                        else:
                            set_path(settings, path, copy.deepcopy(value))
                else:
                    settings = copy.deepcopy(mine)
                    settings[part] = copy.deepcopy(source.get(part))
                if self.step.validate is not None:
                    settings = self.step.validate(settings) or settings
                new[k] = settings
        except (TypeError, ValueError) as e:
            self.node_note["text"] = f"Not valid: {e}"
            self.bell()
            return 0
        self.store(new)
        self.refresh()
        self.show_node_kept()
        what = "quantities exported" if part == "export" else "settings"
        self.node_note["text"] = (f"{len(new)} project{'s' if len(new) > 1 else ''}: the {what} of the reference of "
                                  "their stitching" + ("" if part else " (their cameras kept)"))
        return len(new)

    def reset_node(self):
        """The projects of the node selected with the settings of the project open (of the interface)."""
        self.reset_to([p for p in self.targets(self.tree.focus()) if not self.is_open(p)])

    def reset_all(self):
        """All the projects with the settings of the project open (of the interface)."""
        if not messagebox.askyesno("Batch", "Give all the projects the settings of the project open (those of the "
                                            "interface)?", parent=self):
            return
        self.reset_to([p for p in self.ws.projects if self.runnable(p) and not self.is_open(p)])

    # ------------------------------------------------------------------ saving

    def sync(self):
        """The settings of the projects changed saved: in their project files (the section of the step), else in
        the workspace; the project open: in its page later (sync_current)."""
        self.sync_job = None
        projects = [p for p in self.ws.projects if key(p.folder) in self.dirty]
        self.dirty.clear()
        for p in projects:
            settings = copy.deepcopy(self.cache[key(p.folder)])
            if self.step.to_project is None:
                self.ws.set_node_settings(self.step.key, p.root, p.path, settings)
                continue
            if self.is_open(p):
                self.current_changed = True
                continue
            section, merge = self.step.to_project
            try:
                data = read_project(p.folder) or {}
                data = {k: v for k, v in data.items() if k not in ("format", "version", "saved")}
                if not data and p.images:
                    data["images"] = {"folder": ".", "files": image_names(p.folder)}
                data[section] = merge(dict(data.get(section) or {}), settings)
                write_project(os.path.join(p.folder, PROJECT_FILE), data)
                p.project_file = True
            except (OSError, ValueError, TypeError) as e:
                self.app.update_status(f"The settings could not be saved in the project of {p.folder}: {e}")
        if projects and self.step.to_project is None:
            self.app.workspace.save()

    def sync_current(self):
        """The settings of the project open, changed here, in its page (and so in its project file)."""
        self.sync()
        if not self.current_changed:
            return
        self.current_changed = False
        section, merge = self.step.to_project
        folder = self.app.project_folder()
        project = self.open_project()
        page = next((pg for pg in self.app.pages.values() if pg.project_key == section), None)
        if project is None or page is None:
            return
        self.app.flush_project()
        data = (read_project(folder) or {}).get(section) or {}
        page.restore(merge(dict(data), copy.deepcopy(self.cache[key(folder)])), lambda path: absolute(path, folder))
        if self.app.current_page() == page.key:
            page.shown()
        self.app.save_project()
        self.app.update_status(f"The settings of {project.label()} changed in the window Batch are in the page "
                               f"{page.title}")

    # ------------------------------------------------------------------ run

    def save_state(self, options=None):
        self.ws.checked[self.step.job] = [p.folder for p in self.ws.projects if key(p.folder) in self.checked]
        saved = dict(self.ws.options.get(self.step.job, {}))
        if options is not None:
            saved.update(options)
        self.ws.options[self.step.job] = saved
        self.app.workspace.save()

    def close(self):
        options = None
        if self.step.options is not None:
            try:
                options = self.step.options.read()
            except ValueError:
                options = None
        if self.form is not None:
            self.form.flush()
        self.save_state(options)
        self.sync_current()
        self.destroy()

    def ok(self):
        if self.form is not None:
            self.form.flush()
        projects = self.chosen()
        if not projects:
            self.summary["text"] = "Check at least one project"
            self.bell()
            return
        options = {}
        try:
            if self.step.options is not None:
                options = self.step.options.read()
            entries = []
            for p in projects:
                settings = copy.deepcopy(self.settings_of(p))
                entries.append({"folder": p.folder, "root": p.root, "label": p.label(),
                                "settings": self.step.job_settings(p, settings, self.settings_of_step)})
            job_options = self.step.options.job(projects) if self.step.options is not None else {}
        except ValueError as e:
            self.summary["text"] = str(e)
            self.bell()
            return
        if self.step.confirm is not None:
            more = self.step.confirm(self, entries, job_options)
            if more is None:
                self.summary["text"] = "Not run: the results of the previous exports kept"
                return
            job_options.update(more)
        self.save_state(options)
        self.sync_current()
        self.result = {"step": self.step.job, "projects": entries, "options": job_options}
        self.destroy()


class BatchRunWindow(tk.Toplevel):
    """A batch run in its own process (batch_run.py): its log, the progress of the projects and of the step running,
    the preview of the files it writes (figures, images, frames of the videos), Stop (Ctrl+Break: the step running
    ends there) and Close at the end. Meanwhile, the panel Projects marks the project processed, each project is
    refreshed at its result (the project open: opened again, its results shown) and the results go to the status.
    done(results) is called at its end with [(label, status, message)]."""

    def __init__(self, app, job, title, done=None):
        super().__init__(app.window)
        self.app, self.job, self.done_callback = app, job, done
        self.title(f"Batch: {title}")
        self.geometry("1280x560")
        self.folders, self.preview_image = [], None
        self.results, self.current, self.lines = [], None, queue.Queue()
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        self.project_label = ttk.Label(frame, font=("TkDefaultFont", 9, "bold"))
        self.project_label.pack(anchor=tk.W)
        self.projects_bar = ttk.Progressbar(frame, maximum=1.0)
        self.projects_bar.pack(fill=tk.X, pady=(4, 0))
        self.step_label = ttk.Label(frame, foreground="#555555")
        self.step_label.pack(anchor=tk.W, pady=(6, 0))
        self.step_bar = ttk.Progressbar(frame, maximum=1.0)
        self.step_bar.pack(fill=tk.X, pady=(2, 0))
        box = ttk.Frame(frame)
        box.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        # The latest file written (a figure, an image, a frame of a video), right of the log
        side = ttk.LabelFrame(box, text="Latest output", padding=4)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        self.preview = ttk.Label(side, text="(none yet)", anchor=tk.CENTER, width=60)
        self.preview.pack(fill=tk.BOTH, expand=True)
        self.preview_name = ttk.Label(side, foreground="#555555", wraplength=420)
        self.preview_name.pack(anchor=tk.W)
        self.log = tk.Text(box, height=16, wrap=tk.NONE, font=("Consolas", 9), state=tk.DISABLED)
        bar = ttk.Scrollbar(box, command=self.log.yview)
        self.log.configure(yscrollcommand=bar.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar.pack(side=tk.LEFT, fill=tk.Y)
        buttons = ttk.Frame(frame)
        buttons.pack(anchor=tk.E, pady=(8, 0))
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 6))
        self.close_button = ttk.Button(buttons, text="Close", command=self.close, state=tk.DISABLED)
        self.close_button.pack(side=tk.LEFT)
        self.protocol("WM_DELETE_WINDOW", self.close)
        handle, self.job_file = tempfile.mkstemp(prefix="velocihub_batch_", suffix=".json")
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            json.dump(job, f)
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        package = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.process = subprocess.Popen([sys.executable, "-u", "-m", "velocihub.batch_run", self.job_file], cwd=package,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                        errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                                        creationflags=flags)
        threading.Thread(target=self.read, daemon=True).start()
        app.batch_window = self
        self.write(f"{len(job['projects'])} project(s): {title}\n")
        self.after(100, self.poll)

    def read(self):
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def write(self, text):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def poll(self):
        try:
            while True:
                line = self.lines.get_nowait()
                if line is None:
                    return self.finish()
                self.handle(line)
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def handle(self, line):
        if not line.startswith("@@"):
            self.write(line + "\n")
            return
        kind, _, rest = line[2:].partition(" ")
        if kind == "project":
            i, n, label = rest.split(" ", 2)
            self.current = label
            projects, units = self.job["projects"], self.job.get("options", {}).get("units")
            self.folders = (units[int(i) - 1]["folders"] if units and len(units) == int(n)
                            else [projects[int(i) - 1]["folder"]] if len(projects) == int(n) else [])
            self.app.workspace.set_running(self.folders[0] if self.folders else None)
            self.project_label["text"] = f"Project {i} of {n}: {label}"
            self.projects_bar["value"] = (int(i) - 1) / max(int(n), 1)
            self.step_bar["value"] = 0
        elif kind == "progress":
            fraction, _, message = rest.partition(" ")
            self.step_bar["value"] = float(fraction)
            self.step_label["text"] = message
        elif kind == "result":
            status, _, message = rest.partition(" ")
            self.results.append((self.current, status, message))
            self.write(f"  → {cap(status)}: {message}\n")
            self.app.update_status(f"Batch: {self.current}: {cap(status)}: {message}")
            for folder in self.folders:
                self.project_done(folder)
        elif kind == "preview":
            self.show_preview(rest)

    def project_done(self, folder):
        """A project processed: its marks in the panel Projects (images, PIV results); the project open opened again
        (its results shown)."""
        workspace = self.app.workspace
        project = workspace.ws.project(folder)
        if project is not None:
            info = workspace_detect(folder)
            if info is not None:
                workspace.ws.update(project, info)
            workspace.rebuild()
        current = self.app.project_folder()
        if current and key(current) == key(folder) and self.job["step"] in ("preprocessing", "piv", "postprocessing"):
            self.app.reload_project()
            self.reloaded = True

    def show_preview(self, path):
        """The image of a file just written, in the preview (16-bit images scaled to their range)."""
        try:
            from PIL import Image, ImageTk
            import numpy as np

            with Image.open(path) as im:
                if im.mode in ("I;16", "I;16B", "I", "F"):
                    a = np.asarray(im, np.float64)
                    low, high = np.percentile(a, (1, 99.5))
                    im = Image.fromarray(np.uint8(np.clip((a - low) / max(high - low, 1e-9), 0, 1) * 255))
                im = im.convert("RGB")
                im.thumbnail((420, 300))
                self.preview_image = ImageTk.PhotoImage(im)
        except (OSError, ValueError, SyntaxError):
            return
        self.preview.configure(image=self.preview_image, text="")
        self.preview_name["text"] = os.path.basename(path)

    def stop(self):
        """Stop: Ctrl+Break to the process (the step running ends there), killed if it hasn't ended 30 s later."""
        if self.process.poll() is not None:
            return
        self.stop_button.configure(text="Stopping...", state=tk.DISABLED)
        self.write("Stopping...\n")
        try:
            if sys.platform == "win32":
                os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
            else:
                self.process.send_signal(signal.SIGINT)
        except OSError:
            pass
        self.after(30000, self.kill)

    def kill(self):
        if self.process.poll() is None:
            self.write("Not stopped after 30 s: the process is killed\n")
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)], capture_output=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                self.process.kill()

    def finish(self):
        self.process.wait()
        try:
            os.remove(self.job_file)
        except OSError:
            pass
        self.app.batch_window = None
        self.app.workspace.set_running(None)
        done = sum(s == "done" for _, s, _ in self.results)
        text = (f"Batch ended: {done} done, {sum(s == 'skipped' for _, s, _ in self.results)} skipped, "
                f"{sum(s in ('failed', 'stopped') for _, s, _ in self.results)} failed or stopped")
        self.projects_bar["value"] = 1
        self.project_label["text"] = text
        self.write(text + "\n")
        self.stop_button.configure(state=tk.DISABLED)
        self.close_button.configure(state=tk.NORMAL)
        self.app.update_status(text)
        if self.done_callback is not None:
            self.done_callback(self.results)

    def close(self):
        if self.process.poll() is None:
            from tkinter import messagebox
            if messagebox.askyesno("Batch", "The batch is running: stop it?", parent=self):
                self.stop()
            return
        self.destroy()


def run_batch(app, step):
    """The window Batch of a step, then its run (the projects run refreshed, the project open opened again if it
    was run)."""
    reason = app.busy()
    if reason:
        app.msgbox(reason, "Batch", "warn")
        return None
    if not app.workspace.ws.projects:
        app.msgbox("The workspace has no project: add a root folder in the panel Projects (+ Root folder).", "Batch",
                   "warn")
        return None
    app.flush_project()
    job = BatchDialog(app.parent(), app, step).run()
    if job is None:
        return None
    folders = {key(p["folder"]) for p in job["projects"]}

    def done(results):
        app.workspace.refresh()
        current = app.project_folder()
        # The project open: opened again (if it wasn't at its result, e.g. the stitching)
        if (current and key(current) in folders and not getattr(window, "reloaded", False)
                and step.job in ("preprocessing", "piv", "postprocessing")):
            app.reload_project()
    window = BatchRunWindow(app, job, step.title, done)
    return window
