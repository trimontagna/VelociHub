"""The tool Archive projects (button of the panel Projects), on projects of the workspace with the batch window
(batch_dialog.py):

- Archive projects (velocihub/archiving/archive.py): the files of the projects checked into a ZIP or TAR archive. Options: the
  archive and its format; the categories of files of each project in the node settings (a folder: all the projects
  below it), the files of the roots outside the projects too (archived for the roots of the projects with them).

The stitching of fields of view: in the window Batch of the post-processing (stitching.py)."""

import importlib
import os
import time
import tkinter as tk
from tkinter import ttk

from .batch_dialog import BatchStep, Field, run_batch
from .widgets import FileBox

def script(name):
    """A module of the add-on Archiving (velocihub.archiving: archive, post_archive)."""
    return importlib.import_module(f"velocihub.archiving.{name}")


class ArchiveOptions:
    """The options of an archive: the file and its format (the categories: the node settings)."""

    def __init__(self, ap, ws):
        self.ap, self.ws = ap, ws

    def default_output(self):
        base = os.path.dirname(self.ws.file) if self.ws.file else os.getcwd()
        return os.path.join(base, f"VelociHub_archive_{time.strftime('%Y-%m-%d')}{self.ap.EXTENSIONS[self.format.get()]}")

    def build(self, parent, saved):
        ap = self.ap
        self.format = tk.StringVar(parent, saved.get("format") if saved.get("format") in ap.FORMAT_LABELS else "zip")
        self.output = tk.StringVar(parent, saved.get("output") or self.default_output())
        ttk.Label(parent, text="Archive (↺: in the folder of the workspace)").pack(anchor=tk.W)
        self.output_box = FileBox(parent, "Archive", self.output_chosen,
                                  [("Archive", "*.zip *.tar"), ("All files", "*.*")],
                                  ap.EXTENSIONS[self.format.get()])
        self.output_box.pack(fill=tk.X)
        self.output_box.show(self.output.get())
        ttk.Label(parent, text="Format").pack(anchor=tk.W, pady=(6, 0))
        box = ttk.Combobox(parent, values=list(ap.FORMAT_LABELS.values()), state="readonly", width=40)
        box.set(ap.FORMAT_LABELS[self.format.get()])
        box.bind("<<ComboboxSelected>>", lambda event: self.format.set(
            next(k for k, v in ap.FORMAT_LABELS.items() if v == box.get())))
        box.pack(anchor=tk.W)

    def output_chosen(self, path):
        self.output.set(path or self.default_output())
        self.output_box.show(self.output.get())

    def read(self):
        self.output_box.entered()  # The file entered without Return
        if not self.output.get().strip():
            raise ValueError("Options: choose the archive file")
        return {"output": self.output.get().strip(), "format": self.format.get()}

    def job(self, projects):
        output = self.output.get().strip()
        extension = self.ap.EXTENSIONS[self.format.get()]
        if not output.lower().endswith(extension):
            output = os.path.splitext(output)[0] + extension
        return {"output": output, "format": self.format.get(), "projects": [p.folder for p in self.ws.projects]}


def archive_projects(app):
    """The button Archive projects... of the panel Projects."""
    ap = script("archive")
    options = ArchiveOptions(ap, app.workspace.ws)
    labels = {k: label for k, label, _ in ap.CATEGORIES}

    def describe(settings):
        chosen = settings.get("categories") or []
        return (f"{len(chosen)} {'category' if len(chosen) == 1 else 'categories'}: "
                + ", ".join(labels[c] for c in chosen if c in labels))

    def fields(settings):
        return [Field(("categories",), "Files archived", "multi",
                      tuple((k, label) for k, label, _ in ap.CATEGORIES),
                      help=("Files in a root folder but outside any project, such as calibration images. Archived "
                            "once per root."))]
    # The settings of a project without its own: the categories archived by default
    step = BatchStep("archive", "Archive projects", lambda project: True,
                     lambda: {"categories": list(ap.DEFAULT_CATEGORIES)}, describe,
                     lambda project, settings, resolve: {"archive": settings}, options=options, fields=fields)
    run_batch(app, step)
