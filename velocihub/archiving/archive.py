"""Archive of VelociHub projects: the projects found from the roots of folders.txt (the folders that hold a project file
VelociHub_project.json, or those of the previous names of the tool, at any depth), with what was chosen of
each project in a small interface: the original
images, the pre-processed images, the PIV maps (vectors.npz), the post-processing figures and, by
default, the project file, the masks and the settings (JSON files) and the POD spectrum and modes (pod.npz).

    velocihub-archive [folders.txt] [--output archive.zip] [--format zip|zip-compressed|tar]
    velocihub-archive folders.txt --list
    velocihub-archive folders.txt --output X:/Archives/FOV.zip --no-gui [--include ...] [--exclude ...]

folders.txt lists the roots, one per line (relative to the folder of the file, or absolute; empty lines and lines
starting with # are skipped), as for velocihub-batch. The files of each project are found from its project
file (the folders of its images and of the results of the three steps), so that results exported elsewhere than in
the default folders are archived too. The files of a root outside its projects (Calibration, folders not processed)
form one more row, archived like a project.

The archive keeps the folder structure from the lowest common folder of the folders archived (its parent if it is one
of them, so that a single project keeps the name of its folder); a file outside of it is saved under
_external/[drive]/[path]. A summary of what it holds (VelociHub_archive.json) is added at its top. It is written as
[archive].part and renamed when complete, so that an archive stopped or failed isn't mistaken for a complete one.

This module uses the standard library only (Tkinter for its interface).
"""
import argparse
import io
import json
import os
import queue
import shutil
import stat
import sys
import tarfile
import threading
import time
import zipfile
from dataclasses import dataclass, field

# Names of VelociHub (project.py, preprocessing.py, fields.py, postprocessing.py, batch.py), copied so that the archive
# runs without its packages (numpy, matplotlib...); the project files and formats of VelociHub, then of its previous names (the
# versions before the rename)
PROJECT_FILES = ("VelociHub_project.json", "PPP_project.json", "POD_PAIRS_project.json")
PROJECT_FORMATS = ("VelociHub project", "PPP project", "POD_PAIRS project")
BACKUP_SUFFIX = ".bak"
PREPROCESSING_FOLDER = "PreProcessing"
MANIFEST_FILE = "preprocessing.json"
POD_FILE = "pod.npz"
PIV_FOLDER = "PIV"
VECTORS_FILE = "vectors.npz"
PIV_SETTINGS_FILE = "settings.json"
POSTPROCESSING_FOLDER = "PostProcessing"
POSTPROCESSING_FILES = ("postprocessing.json", "metrics.json")
IMAGE_EXTENSIONS = (".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".gif")

ARCHIVE_SUMMARY = "VelociHub_archive.json"
ARCHIVE_FORMAT = "VelociHub archive"
EXTERNAL_FOLDER = "_external"
# Bytes copied at once into the archive
CHUNK = 4 * 2**20

# What can be archived of a project: key, label, archived by default
CATEGORIES = [
    ("originals", "Original images", True),
    ("preprocessed", "Pre-processed images", True),
    ("vectors", "PIV maps: vector fields (vectors.npz)", True),
    ("csv", "PIV maps: CSV files", True),
    ("postprocessing", "Post-processing: figures, report and statistics", True),
    ("pod", "POD spectrum and modes (pod.npz)", True),
    ("project", "Project, masks and settings (JSON files)", True),
    ("other", "Other files of the project folders", False),
    ("outside", "Files of the roots outside the projects (Calibration...)", True),
]
LABELS = {key: label for key, label, _ in CATEGORIES}
DEFAULT_CATEGORIES = [key for key, _, default in CATEGORIES if default]

# Formats of the archive: key, label, extension
FORMATS = [("zip", "ZIP, not compressed (fastest)", ".zip"),
           ("zip-compressed", "ZIP, compressed (smaller, much slower)", ".zip"),
           ("tar", "TAR, not compressed", ".tar")]
FORMAT_LABELS = {key: label for key, label, _ in FORMATS}
EXTENSIONS = {key: extension for key, _, extension in FORMATS}


################################################################################
# Projects and their files


def format_size(n):
    """Size as shown to the user: "850 bytes", "12.3 GB"."""
    if n < 1024:
        return f"{n} bytes"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024
        if n < 1000 or unit == "TB":
            return f"{n:.3g} {unit}" if n < 100 else f"{n:.0f} {unit}"


def format_duration(seconds):
    """Duration as shown to the user: "45 s", "7 min 02 s" or "1 h 05 min" (as in VelociHub.project)."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def path_key(path):
    """Path compared with others (case ignored on Windows)."""
    return os.path.normcase(os.path.abspath(path))


def absolute(path, root):
    """Path read from a project (relative to its folder, root), None if there is none."""
    if not path:
        return None
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(root, path))


def read_json(path):
    """Dictionary of a JSON file, None if it doesn't exist or can't be read."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def files_of(folder):
    """Names of the files of a folder (not of its subfolders), sorted; none if it doesn't exist."""
    try:
        with os.scandir(folder) as entries:
            return sorted(entry.name for entry in entries if entry.is_file())
    except OSError:
        return []


def is_image(name):
    return name.lower().endswith(IMAGE_EXTENSIONS)


def read_roots(filename):
    """Roots listed in a folders file: one per line, relative to the folder of the file or absolute (empty lines and
    lines starting with # skipped)."""
    base = os.path.dirname(os.path.abspath(filename))
    roots = []
    with open(filename, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip().strip('"')
            if line and not line.startswith("#"):
                roots.append(os.path.normpath(os.path.join(base, line)))
    return roots


@dataclass
class Unit:
    """What can be archived of a folder: a project (kind "project"), or the files of a root outside its projects (kind
    "outside"). files: {category: {path: size}}; warnings: what is missing."""

    folder: str
    root: str
    kind: str = "project"
    name: str = ""
    files: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    claimed: set = field(default_factory=set, repr=False)

    def add(self, category, path):
        """Add a file to a category, unless it is in one already. Returns whether it exists."""
        key = path_key(path)
        if key in self.claimed:
            return True
        try:
            info = os.stat(path)
        except OSError:
            return False
        if not stat.S_ISREG(info.st_mode):
            return False
        self.claimed.add(key)
        self.files.setdefault(category, {})[os.path.abspath(path)] = info.st_size
        return True

    def add_listed(self, category, folder, names, what):
        """Add the files of a list (names in a folder), with a warning if some are missing."""
        missing = sum(not self.add(category, os.path.join(folder, name)) for name in names)
        if missing:
            self.warnings.append(f"{missing} of {len(names)} {what} missing")

    def count(self, categories):
        return sum(len(self.files.get(c, ())) for c in categories)

    def size(self, categories):
        return sum(sum(self.files.get(c, {}).values()) for c in categories)


def scan_project(folder, root):
    """Files of a project, by category, found from its project file: the folders of its images and of the results of
    the three steps (default folders if the project doesn't say), then the other files of its folder (those of the
    projects in its subfolders left out)."""
    unit = Unit(os.path.abspath(folder), root)
    project_file = next(path for path in (os.path.join(folder, name) for name in PROJECT_FILES) if os.path.isfile(path))
    name = os.path.basename(project_file)
    data = read_json(project_file)
    if data is None:
        unit.warnings.append(f"{name} can't be read")
        data = {}
    elif data.get("format") not in PROJECT_FORMATS:
        unit.warnings.append(f"{name} isn't a {PROJECT_FORMATS[0]} file")
    unit.name = data.get("name") or os.path.basename(os.path.normpath(folder))
    images = data.get("images") or {}
    pre = data.get("preprocessing") or {}
    piv = data.get("piv") or {}
    post = data.get("postprocessing") or {}

    # Project, masks and settings first: the project, its copy kept by the batch, the masks exported next to it and
    # the records of the three steps
    for name in PROJECT_FILES:
        unit.add("project", os.path.join(folder, name))
        unit.add("project", os.path.join(folder, name + BACKUP_SUFFIX))
    for name in files_of(folder):
        if name.lower().endswith(".json"):
            unit.add("project", os.path.join(folder, name))
    export = absolute(pre.get("export_folder") or PREPROCESSING_FOLDER, folder)
    unit.add("project", os.path.join(export, MANIFEST_FILE))
    output = absolute(piv.get("output_folder") or PIV_FOLDER, folder)
    unit.add("project", os.path.join(output, PIV_SETTINGS_FILE))
    post_folder = absolute(post.get("export_folder") or POSTPROCESSING_FOLDER, folder)
    for name in POSTPROCESSING_FILES:
        unit.add("project", os.path.join(post_folder, name))
    unit.add("pod", absolute((pre.get("pod") or {}).get("file"), folder) or os.path.join(export, POD_FILE))

    # Original images: those of the project, or else all the images of its folder
    image_folder = absolute(images.get("folder") or ".", folder)
    names = images.get("files") or [name for name in files_of(image_folder) if is_image(name)]
    unit.add_listed("originals", image_folder, names, "original images")

    # Pre-processed images: those of the last export (preprocessing.json), or else all the images of its folder
    manifest = read_json(os.path.join(export, MANIFEST_FILE)) or {}
    names = manifest.get("files") or [name for name in files_of(export) if is_image(name)]
    unit.add_listed("preprocessed", export, names, "pre-processed images")

    # PIV: the vector fields and the CSV files (vector maps, mean field), and those of another folder of vector maps
    # opened in the post-processing (DynamicStudio exports)
    results = absolute((piv.get("results") or {}).get("file"), folder) or os.path.join(output, VECTORS_FILE)
    if not unit.add("vectors", results) and piv.get("results"):
        unit.warnings.append(f"PIV results missing ({os.path.basename(results)})")
    maps_folders = [output] + ([absolute(post["maps_folder"], folder)] if post.get("maps_folder") else [])
    for maps in maps_folders:
        for name in files_of(maps):
            if name.lower().endswith(".csv"):
                unit.add("csv", os.path.join(maps, name))
            elif name.lower().endswith(".npz"):
                unit.add("vectors", os.path.join(maps, name))

    # Post-processing: the figures, the report and the statistics
    for dirpath, dirnames, filenames in os.walk(post_folder):
        dirnames.sort()
        for name in sorted(filenames):
            unit.add("postprocessing", os.path.join(dirpath, name))

    # Other files of the project folder (DynamicStudio exports, the chunks of a PIV not finished...)
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if not is_project(os.path.join(dirpath, d)))
        for name in sorted(filenames):
            unit.add("other", os.path.join(dirpath, name))
    return unit


def is_project(folder):
    return any(os.path.isfile(os.path.join(folder, name)) for name in PROJECT_FILES)


def find_projects(root):
    """Folders of the projects under a root (the root itself included), sorted."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if any(name in filenames for name in PROJECT_FILES):
            found.append(os.path.abspath(dirpath))
    return found


def scan(roots, status=None):
    """Units of the roots: their projects, then, for each root, its files outside its projects (if any). A project
    under two roots is found once. status(message): called for each folder scanned. Returns the units and the
    warnings about the roots."""
    units, warnings, seen = [], [], set()
    for root in roots:
        if not os.path.isdir(root):
            warnings.append(f"The folder doesn't exist: {root}")
            continue
        if status is not None:
            status(f"Looking for projects in {root}...")
        for folder in find_projects(root):
            if path_key(folder) not in seen:
                seen.add(path_key(folder))
                if status is not None:
                    status(f"Scanning {folder}...")
                units.append(scan_project(folder, root))
    claimed = set().union(*(unit.claimed for unit in units))
    for root in roots:
        if not os.path.isdir(root) or path_key(root) in seen:
            continue
        unit = Unit(os.path.abspath(root), root, "outside", "Files outside the projects", claimed=claimed)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if path_key(os.path.join(dirpath, d)) not in seen)
            for name in sorted(filenames):
                unit.add("outside", os.path.join(dirpath, name))
        if unit.files:
            units.append(unit)
    for unit in units:
        unit.claimed = set()
    return units, warnings


################################################################################
# Archive


class Cancelled(Exception):
    """The archive was stopped."""


class Progress:
    """Progress of an archive, shared with the thread that shows it: bytes archived of the total, file being archived
    and the request to stop (cancel)."""

    def __init__(self, total):
        self.total = total
        self.done = 0
        self.name = ""
        self.start = time.time()
        self.cancel = threading.Event()

    def describe(self):
        """Progress as shown to the user: bytes archived, speed and time left."""
        elapsed = time.time() - self.start
        text = f"{format_size(self.done)} of {format_size(self.total)}"
        if elapsed > 2 and self.done:
            speed = self.done / elapsed
            text += f", {format_size(speed)}/s, {format_duration((self.total - self.done) / speed)} left"
        return text


def base_folder(units):
    """Folder from which the archive keeps the folder structure: the lowest common folder of the units, its parent if
    it is one of them. None if they are on different drives."""
    folders = [os.path.abspath(unit.folder) for unit in units]
    if not folders:
        return None
    try:
        common = os.path.commonpath(folders)
    except ValueError:
        return None
    if any(path_key(f) == path_key(common) for f in folders):
        common = os.path.dirname(common)
    return common


def archive_name(path, base):
    """Name of a file in the archive: its path from the base folder, or _external/[drive]/[path] if it is outside
    ([drive]/[path] if there is no base folder)."""
    path = os.path.abspath(path)
    if base is not None:
        try:
            relative = os.path.relpath(path, base)
            if relative != os.pardir and not relative.startswith(os.pardir + os.sep):
                return relative.replace("\\", "/")
        except ValueError:  # Another drive
            pass
    drive, rest = os.path.splitdrive(path)
    parts = ([EXTERNAL_FOLDER] if base is not None else []) + [drive.replace(":", "").replace("\\", "/").strip("/"),
                                                                 rest.replace("\\", "/").strip("/")]
    return "/".join(part for part in parts if part)


def unit_categories(categories, unit):
    """The categories archived of a unit: those of all (a list), or categories(unit) (a function: by unit)."""
    return categories(unit) if callable(categories) else categories


def archive_entries(units, categories):
    """Files of the categories of the units to archive (categories: a list, or a function of the unit), the base
    folder (base_folder) and the list of (path, name in the archive, size), each file once."""
    units = [unit for unit in units if unit.count(unit_categories(categories, unit))]
    base = base_folder(units)
    entries, names = [], set()
    for unit in units:
        for category in unit_categories(categories, unit):
            for path, size in unit.files.get(category, {}).items():
                name = archive_name(path, base)
                if name.lower() not in names:
                    names.add(name.lower())
                    entries.append((path, name, size))
    return base, entries


def archive_summary(units, categories, base, fmt, errors=()):
    """Summary of an archive (VelociHub_archive.json): its source, the categories archived and what was archived of
    each folder (number of files and size per category, what was missing), and the files that couldn't be read."""
    folders = []
    for unit in units:
        chosen = unit_categories(categories, unit)
        if unit.count(chosen):
            folders.append({"folder": archive_name(unit.folder, base) or ".", "source": unit.folder,
                            "kind": unit.kind, "name": unit.name,
                            "files": {LABELS[c]: {"n_files": len(unit.files[c]), "bytes": sum(unit.files[c].values())}
                                      for c in chosen if unit.files.get(c)},
                            "warnings": unit.warnings})
    every = categories if not callable(categories) else sorted(
        {c for unit in units for c in categories(unit)}, key=[k for k, _, _ in CATEGORIES].index)
    return {"format": ARCHIVE_FORMAT, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "base_folder": base,
            "archive_format": FORMAT_LABELS[fmt], "categories": [LABELS[c] for c in every],
            "n_files": sum(f["n_files"] for folder in folders for f in folder["files"].values()),
            "bytes": sum(f["bytes"] for folder in folders for f in folder["files"].values()),
            "folders": folders, "errors": list(errors)}


class Reader:
    """File read into the archive, counting the bytes read; Cancelled raised when the archive is stopped."""

    def __init__(self, f, progress):
        self.f = f
        self.progress = progress

    def read(self, n=-1):
        if self.progress.cancel.is_set():
            raise Cancelled()
        data = self.f.read(n)
        self.progress.done += len(data)
        return data


class ArchiveWriter:
    """ZIP or TAR archive (a format of FORMATS) written file by file."""

    def __init__(self, filename, fmt):
        self.tar = self.zip = None
        if fmt == "tar":
            self.tar = tarfile.open(filename, "w", format=tarfile.PAX_FORMAT)
            self.tar.copybufsize = CHUNK
        else:
            self.compression = zipfile.ZIP_DEFLATED if fmt == "zip-compressed" else zipfile.ZIP_STORED
            self.zip = zipfile.ZipFile(filename, "w", self.compression, allowZip64=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        (self.tar or self.zip).close()

    def add(self, path, name, f, progress):
        """Add a file (path, open as f) under a name."""
        reader = Reader(f, progress)
        if self.tar is not None:
            self.tar.addfile(self.tar.gettarinfo(path, name), reader)
        else:
            info = zipfile.ZipInfo.from_file(path, name)
            info.compress_type = self.compression
            with self.zip.open(info, "w") as dst:
                shutil.copyfileobj(reader, dst, CHUNK)

    def add_bytes(self, name, data):
        if self.tar is not None:
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), time.time()
            self.tar.addfile(info, io.BytesIO(data))
        else:
            self.zip.writestr(zipfile.ZipInfo(name, time.localtime()[:6]), data, self.compression)


def write_archive(output, entries, fmt, summary, progress, report=print):
    """Write the files (entries of archive_entries) into an archive of a format of FORMATS, with its summary
    (summary(errors): its dictionary, with the files that couldn't be read), written last. The archive is written as
    output.part and renamed when complete; it is removed if stopped (progress.cancel set: Cancelled raised) or
    failed. report(message): the files that can't be read. Returns their errors."""
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    part = output + ".part"
    errors = []
    try:
        with ArchiveWriter(part, fmt) as writer:
            for path, name, size in entries:
                if progress.cancel.is_set():
                    raise Cancelled()
                progress.name = name
                try:
                    f = open(path, "rb")
                except OSError as e:
                    errors.append(f"{path}: {e.strerror or e}")
                    report(f"Skipped (can't be read): {errors[-1]}")
                    progress.done += size
                    continue
                with f:
                    writer.add(path, name, f, progress)
            writer.add_bytes(ARCHIVE_SUMMARY, json.dumps(summary(errors), indent=1, ensure_ascii=False).encode("utf-8"))
        os.replace(part, output)
    except BaseException:
        try:
            os.remove(part)
        except OSError:
            pass
        raise
    return errors


def free_space(output):
    """Free space (bytes) where the archive is written, None if unknown."""
    folder = os.path.dirname(os.path.abspath(output))
    while folder and not os.path.isdir(folder):
        parent = os.path.dirname(folder)
        if parent == folder:
            return None
        folder = parent
    try:
        return shutil.disk_usage(folder).free
    except OSError:
        return None


def space_problem(output, total, fmt):
    """Message if the archive may not fit where it is written (None if it fits)."""
    free = free_space(output)
    if free is None or free > total * 1.01 + 2**20:
        return None
    return (f"The archive needs {format_size(total)}" + (" at most" if fmt == "zip-compressed" else "")
            + f", and only {format_size(free)} are free there.")


################################################################################
# Interface


class ArchiveWindow:
    """The interface: the roots of a folders file scanned, the projects to archive and what to archive of them chosen,
    and the archive written in a thread, with its progress."""

    CHECKED, UNCHECKED, PARTIAL = "\u2611", "\u2610", "\u25a3"

    def __init__(self, master, folders_file=None, output=None, fmt="zip", categories=None):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.master = master
        master.title("VelociHub archive")
        master.minsize(900, 640)
        self.units = []
        self.unit_of = {}      # tree item of a unit -> unit
        self.checked = {}      # tree item of a unit -> archived or not
        self.labels = {}       # tree item -> its text, without the check mark
        self.messages = queue.Queue()
        self.worker = None
        self.progress = None
        self.closing = False
        self.base = None

        categories = DEFAULT_CATEGORIES if categories is None else categories
        self.folders_file = tk.StringVar(value=folders_file or "")
        self.output = tk.StringVar(value=output or "")
        self.format = tk.StringVar(value=FORMAT_LABELS[fmt])
        self.categories = {key: tk.BooleanVar(value=key in categories) for key, _, _ in CATEGORIES}
        self.status = tk.StringVar(value="Choose the folders file and scan it.")
        self.info = tk.StringVar()
        pad = {"padx": 8, "pady": 4}

        top = ttk.Frame(master)
        top.pack(fill=tk.X, **pad)
        ttk.Label(top, text="Folders file").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.folders_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.scan_button = ttk.Button(top, text="Scan", command=self.scan)
        self.scan_button.pack(side=tk.RIGHT)
        self.browse_folders_button = ttk.Button(top, text="Browse...", command=self.browse_folders)
        self.browse_folders_button.pack(side=tk.RIGHT, padx=4)

        # Projects, by root
        box = ttk.LabelFrame(master, text="Projects (click a row to archive it or not)")
        box.pack(fill=tk.BOTH, expand=True, **pad)
        columns = {"#0": ("Folder", 250), "originals": ("Original images", 95), "preprocessed": ("Pre-processed", 95),
                   "piv": ("PIV maps", 80), "postprocessing": ("Post-processing", 95), "size": ("Archived", 80),
                   "notes": ("Notes", 200)}
        frame = ttk.Frame(box)
        frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.tree = ttk.Treeview(frame, columns=list(columns)[1:], height=12, selectmode="browse")
        for column, (text, width) in columns.items():
            left = column in ("#0", "notes")
            self.tree.heading(column, text=text, anchor=tk.W if left else tk.E)
            self.tree.column(column, width=width, minwidth=50, stretch=left, anchor=tk.W if left else tk.E)
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<Button-1>", self.click)
        self.tree.bind("<space>", lambda event: self.toggle(self.tree.focus()))
        row = ttk.Frame(box)
        row.pack(fill=tk.X, padx=4, pady=(0, 4))
        self.all_button = ttk.Button(row, text="All", command=lambda: self.check_all(True))
        self.all_button.pack(side=tk.LEFT)
        self.none_button = ttk.Button(row, text="None", command=lambda: self.check_all(False))
        self.none_button.pack(side=tk.LEFT, padx=4)

        # What to archive
        what = ttk.LabelFrame(master, text="What to archive")
        what.pack(fill=tk.X, **pad)
        self.category_buttons = {}
        for i, (key, label, _) in enumerate(CATEGORIES):
            button = ttk.Checkbutton(what, text=label, variable=self.categories[key], command=self.update_sizes)
            button.grid(row=i // 2, column=i % 2, sticky=tk.W, padx=8, pady=1)
            self.category_buttons[key] = button
        what.columnconfigure((0, 1), weight=1)

        # Archive
        out = ttk.LabelFrame(master, text="Archive")
        out.pack(fill=tk.X, **pad)
        ttk.Label(out, text="File").grid(row=0, column=0, sticky=tk.W, padx=(8, 4), pady=2)
        ttk.Entry(out, textvariable=self.output).grid(row=0, column=1, sticky=tk.EW, pady=2)
        self.browse_output_button = ttk.Button(out, text="Browse...", command=self.browse_output)
        self.browse_output_button.grid(row=0, column=2, padx=8, pady=2)
        ttk.Label(out, text="Format").grid(row=1, column=0, sticky=tk.W, padx=(8, 4), pady=2)
        self.format_box = ttk.Combobox(out, textvariable=self.format, values=[label for _, label, _ in FORMATS],
                                       state="readonly", width=40)
        self.format_box.grid(row=1, column=1, sticky=tk.W, pady=2)
        self.format_box.bind("<<ComboboxSelected>>", self.format_changed)
        ttk.Label(out, textvariable=self.info, foreground="#5f6b7d").grid(row=2, column=0, columnspan=3,
                                                                          sticky=tk.W, padx=8, pady=(2, 4))
        out.columnconfigure(1, weight=1)

        # Progress and messages
        self.bar = ttk.Progressbar(master, maximum=1000)
        self.bar.pack(fill=tk.X, **pad)
        ttk.Label(master, textvariable=self.status).pack(fill=tk.X, padx=8)
        frame = ttk.Frame(master)
        frame.pack(fill=tk.BOTH, padx=8, pady=4)
        self.log = tk.Text(frame, height=5, wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT,
                           background=master.cget("background"))
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        bottom = ttk.Frame(master)
        bottom.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(bottom, text="Close", command=self.close).pack(side=tk.RIGHT)
        self.stop_button = ttk.Button(bottom, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=4)
        self.start_button = ttk.Button(bottom, text="Create archive", command=self.start, state=tk.DISABLED)
        self.start_button.pack(side=tk.RIGHT)

        master.protocol("WM_DELETE_WINDOW", self.close)
        master.after(100, self.poll)
        if folders_file and os.path.isfile(folders_file):
            master.after(50, self.scan)

    # Messages and state

    def write_log(self, text):
        self.log.configure(state=self.tk.NORMAL)
        self.log.insert(self.tk.END, text + "\n")
        self.log.see(self.tk.END)
        self.log.configure(state=self.tk.DISABLED)

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def set_busy(self, busy, archiving=False):
        state = self.tk.DISABLED if busy else self.tk.NORMAL
        for widget in (self.scan_button, self.browse_folders_button, self.browse_output_button, self.all_button,
                       self.none_button, *self.category_buttons.values()):
            widget.configure(state=state)
        self.format_box.configure(state=self.tk.DISABLED if busy else "readonly")
        self.start_button.configure(state=self.tk.DISABLED if busy or not self.units else self.tk.NORMAL)
        self.stop_button.configure(state=self.tk.NORMAL if archiving else self.tk.DISABLED)

    def run(self, target, *args):
        """Run a task in a thread: its messages (kind, value) are put in the queue, read by poll."""
        def work():
            try:
                target(*args)
            except Cancelled:
                self.messages.put(("stopped", None))
            except Exception as e:  # Shown in the interface
                self.messages.put(("failed", e))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def poll(self):
        while True:
            try:
                kind, value = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status.set(value)
            elif kind == "log":
                self.write_log(value)
            elif kind == "scanned":
                self.show_units(*value)
            elif kind == "archived":
                self.archived(*value)
            elif kind == "stopped":
                self.set_busy(False)
                self.bar.configure(value=0)
                self.status.set("Archive stopped: the incomplete file was removed.")
                self.write_log(self.status.get())
            elif kind == "failed":
                self.set_busy(False)
                self.status.set(f"Failed: {value}")
                self.write_log(self.status.get())
        if self.progress is not None and self.busy():
            self.bar.configure(value=1000 * self.progress.done / max(self.progress.total, 1))
            self.status.set(f"{self.progress.describe()} - {self.progress.name}")
        if self.closing and not self.busy():
            self.master.destroy()
            return
        self.master.after(200, self.poll)

    # Projects

    def browse_folders(self):
        from tkinter import filedialog
        current = self.folders_file.get()
        filename = filedialog.askopenfilename(parent=self.master, title="Folders file...",
                                              initialdir=os.path.dirname(current) if current else os.getcwd(),
                                              filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filename:
            self.folders_file.set(os.path.normpath(filename))
            self.scan()

    def scan(self):
        if self.busy():
            return
        filename = self.folders_file.get().strip()
        try:
            roots = read_roots(filename)
        except OSError as e:
            self.status.set(f"The folders file can't be read: {e.strerror or e}")
            return
        if not roots:
            self.status.set("The folders file lists no folder.")
            return
        self.tree.delete(*self.tree.get_children())
        self.units, self.unit_of, self.checked, self.labels = [], {}, {}, {}
        self.write_log(f"Scanning {len(roots)} folder(s) of {filename}...")
        self.set_busy(True)

        def task():
            units, warnings = scan(roots, lambda message: self.messages.put(("status", message)))
            self.messages.put(("scanned", (roots, units, warnings)))
        self.run(task)

    def show_units(self, roots, units, warnings):
        self.set_busy(False)
        self.units = units
        for warning in warnings:
            self.write_log(warning)
        base = os.path.dirname(os.path.abspath(self.folders_file.get().strip()))
        for root in roots:
            children = [unit for unit in units if unit.root == root]
            if not children:
                continue
            item = self.tree.insert("", self.tk.END, open=True)
            # The root as in the folders file: relative to it if it is inside its folder
            label = archive_name(root, base)
            self.labels[item] = label if not label.startswith(EXTERNAL_FOLDER + "/") else root
            for unit in children:
                if unit.kind == "outside":
                    label = f"({unit.name})"
                else:
                    label = os.path.relpath(unit.folder, root)
                    label = os.path.basename(unit.folder) if label == os.curdir else label
                child = self.tree.insert(item, self.tk.END)
                self.labels[child], self.unit_of[child], self.checked[child] = label, unit, True
                for warning in unit.warnings:
                    self.write_log(f"{unit.folder}: {warning}")
        n_projects = sum(unit.kind == "project" for unit in units)
        self.status.set(f"{n_projects} project(s) found." if n_projects else "No project found.")
        self.write_log(self.status.get())
        self.set_busy(False)
        self.update_sizes()

    def click(self, event):
        if self.busy():
            return "break"
        if self.tree.identify_region(event.x, event.y) not in ("tree", "cell"):
            return None
        if "indicator" in self.tree.identify_element(event.x, event.y):  # Opens or closes a root
            return None
        item = self.tree.identify_row(event.y)
        if item:
            self.toggle(item)
        return None

    def toggle(self, item):
        if not item or self.busy():
            return
        children = [item] if item in self.unit_of else self.tree.get_children(item)
        value = not all(self.checked[child] for child in children)
        for child in children:
            self.checked[child] = value
        self.update_sizes()

    def check_all(self, value):
        for item in self.checked:
            self.checked[item] = value
        self.update_sizes()

    def selected_categories(self):
        return [key for key, _, _ in CATEGORIES if self.categories[key].get()]

    def selected_units(self):
        return [self.unit_of[item] for item in self.unit_of if self.checked[item]]

    def update_sizes(self):
        """Check marks, numbers of files and sizes of the projects and of the categories, and the archive."""
        categories = self.selected_categories()
        for item in self.tree.get_children():
            children = self.tree.get_children(item)
            states = [self.checked[child] for child in children]
            mark = self.CHECKED if all(states) else self.UNCHECKED if not any(states) else self.PARTIAL
            size = sum(self.unit_of[c].size(categories) for c in children if self.checked[c])
            self.tree.item(item, text=f"{mark} {self.labels[item]}", values=("", "", "", "", format_size(size), ""))
            for child in children:
                unit = self.unit_of[child]
                n = {key: len(unit.files.get(key, ())) for key, _, _ in CATEGORIES}
                if unit.kind == "outside":
                    tops = sorted({os.path.relpath(p, unit.folder).split(os.sep)[0] for p in unit.files["outside"]})
                    notes = f"{n['outside']} files: " + ", ".join(tops[:6]) + (", ..." if len(tops) > 6 else "")
                else:
                    notes = "; ".join(unit.warnings)
                self.tree.item(child, text=f"{self.CHECKED if self.checked[child] else self.UNCHECKED} "
                                           f"{self.labels[child]}",
                               values=(n["originals"] or "-", n["preprocessed"] or "-",
                                       n["vectors"] + n["csv"] or "-", n["postprocessing"] or "-",
                                       format_size(unit.size(categories)) if self.checked[child] else "", notes))
        units = self.selected_units()
        for key, label, _ in CATEGORIES:
            count = sum(len(unit.files.get(key, ())) for unit in units)
            size = sum(sum(unit.files.get(key, {}).values()) for unit in units)
            self.category_buttons[key].configure(text=f"{label}: {count} files, {format_size(size)}" if count
                                                 else f"{label}: none")
        self.base, entries = archive_entries(units, categories)
        if not self.output.get().strip() and self.base:
            self.output.set(os.path.join(self.base, f"VelociHub_archive_{time.strftime('%Y-%m-%d')}"
                                                    + EXTENSIONS[self.format_key()]))
        self.update_info(entries)
        if not self.busy():
            self.start_button.configure(state=self.tk.NORMAL if entries else self.tk.DISABLED)

    def update_info(self, entries=None):
        if entries is None:
            self.base, entries = archive_entries(self.selected_units(), self.selected_categories())
        if not entries:
            self.info.set("Nothing to archive.")
            return
        total = sum(entry[2] for entry in entries)
        free = free_space(self.output.get()) if self.output.get().strip() else None
        self.info.set(f"{len(entries)} files, {format_size(total)}"
                      + (" before compression" if self.format_key() == "zip-compressed" else "")
                      + (f" (free space: {format_size(free)})" if free is not None else "")
                      + f"  -  folders kept from {self.base or 'the drives'}"
                      + f", e.g. {entries[0][1]}")

    # Archive

    def format_key(self):
        return next(key for key, label, _ in FORMATS if label == self.format.get())

    def format_changed(self, event=None):
        output = self.output.get().strip()
        root, extension = os.path.splitext(output)
        if output and extension.lower() in EXTENSIONS.values():
            self.output.set(root + EXTENSIONS[self.format_key()])
        self.update_info()

    def browse_output(self):
        from tkinter import filedialog
        extension = EXTENSIONS[self.format_key()]
        current = self.output.get().strip()
        filename = filedialog.asksaveasfilename(
            parent=self.master, title="Archive...", initialdir=os.path.dirname(current) if current else self.base,
            initialfile=os.path.basename(current) if current else None, defaultextension=extension,
            filetypes=[(f"{extension[1:].upper()} archives", "*" + extension), ("All files", "*.*")])
        if filename:
            self.output.set(os.path.normpath(filename))
            self.update_info()

    def start(self):
        from tkinter import messagebox
        if self.busy():
            return
        units, categories, fmt = self.selected_units(), self.selected_categories(), self.format_key()
        base, entries = archive_entries(units, categories)
        if not entries:
            messagebox.showinfo("VelociHub archive", "Nothing to archive.", parent=self.master)
            return
        output = self.output.get().strip()
        if not output:
            messagebox.showerror("VelociHub archive", "Choose the file of the archive.", parent=self.master)
            return
        output = os.path.abspath(output)
        if not os.path.splitext(output)[1]:
            output += EXTENSIONS[fmt]
            self.output.set(output)
        if any(path_key(path) == path_key(output) for path, _, _ in entries):
            messagebox.showerror("VelociHub archive", "The archive would replace one of the files archived.",
                                 parent=self.master)
            return
        if os.path.exists(output) and not messagebox.askyesno(
                "VelociHub archive", f"{output} exists already. Replace it?", parent=self.master):
            return
        total = sum(entry[2] for entry in entries)
        problem = space_problem(output, total, fmt)
        if problem and not messagebox.askyesno("VelociHub archive", problem + " Create the archive anyway?",
                                               icon=messagebox.WARNING, parent=self.master):
            return
        self.progress = Progress(total)
        self.write_log(f"Archive of {len(entries)} files ({format_size(total)}) from {base or 'the drives'} "
                       f"into {output}...")
        self.set_busy(True, archiving=True)

        def task():
            errors = write_archive(output, entries, fmt,
                                   lambda errors: archive_summary(units, categories, base, fmt, errors),
                                   self.progress, lambda message: self.messages.put(("log", message)))
            self.messages.put(("archived", (output, errors)))
        self.run(task)

    def archived(self, output, errors):
        self.set_busy(False)
        self.bar.configure(value=1000)
        elapsed = format_duration(time.time() - self.progress.start)
        self.status.set(f"Archive created in {elapsed}: {output} ({format_size(os.path.getsize(output))})"
                        + (f", {len(errors)} file(s) couldn't be read" if errors else ""))
        self.write_log(self.status.get())

    def stop(self):
        if self.busy() and self.progress is not None:
            self.progress.cancel.set()
            self.status.set("Stopping...")

    def close(self):
        from tkinter import messagebox
        if self.busy():
            if self.progress is None or not messagebox.askyesno(
                    "VelociHub archive", "Stop the archive and close?", parent=self.master):
                return
            self.stop()
            self.closing = True  # Closed by poll when the archive is stopped (its file removed)
            return
        self.master.destroy()


def launch_gui(folders_file=None, output=None, fmt="zip", categories=None):
    import tkinter as tk
    root = tk.Tk()
    ArchiveWindow(root, folders_file, output, fmt, categories)
    root.mainloop()


################################################################################
# Command line


def print_units(units, categories, base, entries):
    for unit in units:
        if unit.count(categories):
            print(f"{unit.folder}: {unit.count(categories)} files, {format_size(unit.size(categories))}"
                  + (f" ({'; '.join(unit.warnings)})" if unit.warnings else ""))
            for key in categories:
                if unit.files.get(key):
                    print(f"  {LABELS[key]}: {len(unit.files[key])} files, "
                          f"{format_size(sum(unit.files[key].values()))}")
    total = sum(entry[2] for entry in entries)
    print(f"{len(entries)} files, {format_size(total)}, folders kept from {base or 'the drives'}")


def main():
    default_folders = os.path.join(os.getcwd(), "folders.txt")
    keys = [key for key, _, _ in CATEGORIES]
    parser = argparse.ArgumentParser(
        prog="velocihub-archive",
        description="Archive of VelociHub projects (the interface opens unless --list or --no-gui)",
        epilog="Categories: " + ", ".join(f"{key} ({label}{', default' if default else ''})"
                                          for key, label, default in CATEGORIES))
    parser.add_argument("folders", nargs="?", default=default_folders if os.path.isfile(default_folders) else None,
                        help="text file of the roots, one per line (default: folders.txt of the current folder)")
    parser.add_argument("--output", help="file of the archive")
    parser.add_argument("--format", choices=[key for key, _, _ in FORMATS], default="zip",
                        help="zip (not compressed, default), zip-compressed or tar")
    parser.add_argument("--include", nargs="+", choices=keys, default=[], metavar="CATEGORY",
                        help="categories archived besides the default ones")
    parser.add_argument("--exclude", nargs="+", choices=keys, default=[], metavar="CATEGORY",
                        help="default categories not archived")
    parser.add_argument("--list", action="store_true", help="only list what would be archived")
    parser.add_argument("--no-gui", action="store_true", help="create the archive without the interface")
    args = parser.parse_args()
    categories = [key for key in keys if (key in DEFAULT_CATEGORIES or key in args.include)
                  and key not in args.exclude]

    if not (args.list or args.no_gui):
        launch_gui(args.folders, args.output, args.format, categories)
        return 0
    if not args.folders:
        parser.error("the folders file is needed")
    units, warnings = scan(read_roots(args.folders))
    for warning in warnings:
        print(warning)
    base, entries = archive_entries(units, categories)
    print_units(units, categories, base, entries)
    if args.list:
        return 0
    if not entries:
        print("Nothing to archive")
        return 1
    if not args.output:
        parser.error("--output is needed with --no-gui")
    output = os.path.abspath(args.output)
    if not os.path.splitext(output)[1]:
        output += EXTENSIONS[args.format]
    if os.path.exists(output):
        print(f"{output} exists already: remove it or choose another file")
        return 1
    progress = Progress(sum(entry[2] for entry in entries))
    problem = space_problem(output, progress.total, args.format)
    if problem:
        print(problem)
        if args.format != "zip-compressed":
            return 1
    print(f"Archive into {output}...")
    result = {}

    def task():
        try:
            result["errors"] = write_archive(output, entries, args.format,
                                             lambda errors: archive_summary(units, categories, base, args.format,
                                                                            errors), progress)
        except Cancelled:
            result["stopped"] = True
        except Exception as e:  # Reported below
            result["failed"] = e
    worker = threading.Thread(target=task, daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            worker.join(10)
            if worker.is_alive():
                print(f"  {progress.describe()}")
    except KeyboardInterrupt:
        progress.cancel.set()
        worker.join()
    if "stopped" in result:
        print("Stopped: the incomplete archive was removed")
        return 1
    if "failed" in result:
        print(f"Failed: {result['failed']}")
        return 1
    print(f"Archive created in {format_duration(time.time() - progress.start)}: {output} "
          f"({format_size(os.path.getsize(output))})")
    for error in result["errors"]:
        print(f"  Not archived: {error}")
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
