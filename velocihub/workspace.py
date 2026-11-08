"""Workspace of the interface: the root folders scanned for projects and the projects found in them, kept in a
workspace file (VelociHub_workspace.json by default, wherever the user saves it), so that one window works on many
projects: the tree of the folders from each root to its projects, as the panel Projects shows it.

A project is a folder with a project file (VelociHub_project.json), with images (two or more: pre-processing and PIV),
or with the results of a velocimetry pipeline (pipelines.py: PIV/vectors.npz of PaIRS, PTV/vectors.npz of TracTrac...,
or vectors.npz in the folder itself; a project may have several, shown apart in the panel Projects). The subfolders of
the results (PreProcessing, those of the pipelines, PostProcessing, PostProcessing_PTV, the export folders of its
project file) and the figures stitched (Stitched) aren't scanned.

Roots: folders scanned at each refresh (scan=True), or the folder of a project added alone (scan=False: only that
project, nothing else detected there). The projects removed by the user (excluded) aren't added again by the scans.
When a project is no longer in its folder, the refresh looks for it in the folders found under the roots that no
project of the workspace has: the same path from its root, else the same folder name and the same contents
(signature: its images, or the size of its vectors.npz); found, it follows; otherwise it is lost (shown in red), until
the user relinks it (a folder, a project file, or a folder of the tree with all the projects below it).

The paths are saved absolute and relative to the folder of the workspace file, so that a tree moved with its
workspace file is found again.

Batch settings: the settings of the steps whose settings aren't in the project files (the archive), by project
(the window Batch keeps the others in the project files)."""

import json
import os
import time
from dataclasses import asdict, dataclass

from .fields import PIV_FOLDER, PTV_FOLDER, VECTORS_FILE
from .pipelines import results_folder_names, results_folders
from .postprocessing import POSTPROCESSING_FOLDER, POSTPROCESSING_PTV_FOLDER
from .preprocessing import IMAGE_EXTENSIONS, PREPROCESSING_FOLDER
from .project import PROJECT_FILES, read_project

WORKSPACE_FILE = "VelociHub_workspace.json"
WORKSPACE_FORMAT = "VelociHub workspace"
# The format of the workspace files of the previous name of the tool: still read
LEGACY_WORKSPACE_FORMATS = ("PPP workspace",)
WORKSPACE_VERSION = 1
# Subfolders not scanned: the results of the steps and the figures stitched (velocihub/archiving/post_archive.py)
SKIPPED_FOLDERS = {PREPROCESSING_FOLDER, PIV_FOLDER, PTV_FOLDER, POSTPROCESSING_FOLDER, POSTPROCESSING_PTV_FOLDER,
                   "Stitched"} | results_folder_names()
# Images exported by the post-processing (figures): not images of a measurement
FIGURE_NAMES = ("_PostProcessing_", "_Stitched_")


def key(path):
    """A path as compared (case and separators of the system)."""
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def inside(path, folder):
    """The path is the folder or inside it."""
    path, folder = key(path), key(folder)
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


@dataclass
class Project:
    """A project of the workspace: its folder, its root and its path from it (the folders of the tree, joined by
    "/"; "" for a root that is a project), the name given in the workspace (None: its own), what it has (images,
    results of the velocimetry, project file), its maps folder (its first results: absolute), the folders of its
    results by pipeline (pipelines: {label: folder relative to the project}, in the order of the pipelines; None: not
    detected yet, a workspace of the previous versions), the signature of its contents and whether it was lost (not
    found at the last refresh)."""

    folder: str
    root: str
    path: str
    name: str = None
    images: bool = False
    results: bool = False
    project_file: bool = False
    maps: str = None
    signature: str = None
    lost: bool = False
    pipelines: dict = None

    def label(self):
        return self.name or (self.path.rsplit("/", 1)[-1] if self.path else os.path.basename(self.folder))

    def results_folders(self):
        """The folders of its results by pipeline ({"PaIRS": folder, "TracTrac": folder}: those it has)."""
        if self.pipelines is None:
            return results_folders(self.folder) if self.results and os.path.isdir(self.folder) else {}
        return {label: os.path.normpath(os.path.join(self.folder, rel)) for label, rel in self.pipelines.items()}


def image_names(folder, names=None):
    """The images of a folder (not the figures exported)."""
    names = os.listdir(folder) if names is None else names
    return sorted(n for n in names if n.lower().endswith(IMAGE_EXTENSIONS) and not any(f in n for f in FIGURE_NAMES)
                  and os.path.isfile(os.path.join(folder, n)))


def detect(folder, names=None):
    """What a folder holds as a project: None if it isn't one, else (images, results, project file, maps folder,
    exported folders, signature, pipelines): images (two or more), results of the velocimetry, project file, its maps
    folder (its first results), the export folders of its project file (not scanned) and the folders of its results
    by pipeline ({label: folder relative to it}; pipelines.results_folders)."""
    try:
        names = os.listdir(folder) if names is None else names
    except OSError:
        return None
    images = image_names(folder, names)
    found = ({} if os.path.basename(folder) in results_folder_names() and VECTORS_FILE in names
             else results_folders(folder))
    results = bool(found)
    maps = next(iter(found.values()), None)
    has_file = any(n in names for n in PROJECT_FILES)
    if len(images) < 2 and not results and not has_file:
        return None
    exported = []
    if has_file:
        try:
            data = read_project(folder) or {}
        except (OSError, ValueError):
            data = {}
        for section, name in (("preprocessing", "export_folder"), ("piv", "output_folder"),
                              ("postprocessing", "export_folder")):
            path = (data.get(section) or {}).get(name)
            if isinstance(path, str) and path:
                exported.append(os.path.normpath(os.path.join(folder, path)))
    vectors = os.path.join(maps or os.path.join(folder, PIV_FOLDER), VECTORS_FILE)
    signature = (f"images:{len(images)}:{images[0]}:{images[-1]}" if len(images) >= 2
                 else f"vectors:{os.path.getsize(vectors)}" if results else None)
    pipelines = {label: os.path.relpath(path, folder) for label, path in found.items()}
    return len(images) >= 2, results, has_file, maps, exported, signature, pipelines


def scan(root):
    """The projects under a root folder: {key(folder): (folder, path from the root, detect(folder))}."""
    found = {}
    for folder, subfolders, names in os.walk(root):
        subfolders.sort()
        info = detect(folder, names)
        skipped = set(SKIPPED_FOLDERS)
        if info is not None:
            rel = os.path.relpath(folder, root)
            found[key(folder)] = (folder, "" if rel == "." else rel.replace(os.sep, "/"), info)
            skipped |= {key(p) for p in info[4]}
        subfolders[:] = [s for s in subfolders if not s.startswith(".") and s not in skipped
                         and key(os.path.join(folder, s)) not in skipped]
    return found


class Workspace:
    """The roots and the projects of a workspace file (file: None while it isn't saved)."""

    def __init__(self, file=None):
        self.file = file
        self.roots = []  # [{"folder", "scan", "lost"}]
        self.projects = []
        self.excluded = []  # Folders of the projects removed by the user
        self.batch = []  # Batch settings by project: [{"step", "root", "path", "settings"}]
        self.checked = {}  # The projects checked for the batch of each step: {step: [folders]}
        self.options = {}  # The options of the batch runs of each step (archive, stitching): {step: {...}}

    # ------------------------------------------------------------------ file

    def to_dict(self):
        base = os.path.dirname(os.path.abspath(self.file)) if self.file else None

        def rel(path):
            if base is None or path is None:
                return None
            try:
                return os.path.relpath(path, base).replace(os.sep, "/")
            except ValueError:  # Another drive
                return None
        return {"format": WORKSPACE_FORMAT, "version": WORKSPACE_VERSION, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                "roots": [{"folder": r["folder"], "relative": rel(r["folder"]), "scan": r["scan"]} for r in self.roots],
                "projects": [{**{k: v for k, v in asdict(p).items() if k != "lost"}, "relative": rel(p.folder),
                              "root_relative": rel(p.root)} for p in self.projects],
                "excluded": [{"folder": e, "relative": rel(e)} for e in self.excluded],
                "batch": [{**b, "root_relative": rel(b["root"])} for b in self.batch],
                "checked": {step: [{"folder": f, "relative": rel(f)} for f in folders]
                            for step, folders in self.checked.items()},
                "options": self.options}

    def save(self, file=None):
        """Save the workspace (in file: from then on its file); written next to it and renamed."""
        self.file = file or self.file
        data = self.to_dict()
        temporary = self.file + ".part"
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        os.replace(temporary, self.file)
        return self.file

    @classmethod
    def load(cls, file):
        """A workspace file: ValueError if it isn't one. Each path: the absolute one if it exists, else the one
        relative to the workspace file (the tree moved with it)."""
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("format") not in (WORKSPACE_FORMAT, *LEGACY_WORKSPACE_FORMATS):
            raise ValueError(f"{file} is not a {WORKSPACE_FORMAT} file")
        ws, base = cls(os.path.abspath(file)), os.path.dirname(os.path.abspath(file))

        def found(path, relative):
            if path and os.path.exists(path):
                return os.path.normpath(path)
            moved = os.path.normpath(os.path.join(base, relative)) if relative else None
            return moved if moved and os.path.exists(moved) else os.path.normpath(path) if path else moved
        for r in data.get("roots") or []:
            ws.roots.append({"folder": found(r.get("folder"), r.get("relative")), "scan": bool(r.get("scan", True)),
                             "lost": False})
        fields = set(Project.__dataclass_fields__) - {"lost"}
        for p in data.get("projects") or []:
            values = {k: v for k, v in p.items() if k in fields}
            values["folder"] = found(p.get("folder"), p.get("relative"))
            values["root"] = found(p.get("root"), p.get("root_relative"))
            ws.projects.append(Project(**values))
        ws.excluded = [found(e.get("folder"), e.get("relative")) for e in data.get("excluded") or []
                       if isinstance(e, dict) and e.get("folder")]
        ws.batch = [{"step": b["step"], "root": found(b.get("root"), b.get("root_relative")), "path": b.get("path", ""),
                     "settings": b.get("settings")}
                    for b in data.get("batch") or [] if isinstance(b, dict) and b.get("step") and b.get("root")
                    and isinstance(b.get("settings"), dict)]
        ws.checked = {step: [found(f.get("folder"), f.get("relative")) for f in folders if isinstance(f, dict)]
                      for step, folders in (data.get("checked") or {}).items() if isinstance(folders, list)}
        ws.options = data.get("options") if isinstance(data.get("options"), dict) else {}
        return ws

    # ------------------------------------------------------------------ roots and projects

    def root(self, folder):
        return next((r for r in self.roots if key(r["folder"]) == key(folder)), None)

    def project(self, folder):
        return next((p for p in self.projects if key(p.folder) == key(folder)), None)

    def add_root(self, folder):
        """A root folder scanned (its projects detected at once). Returns the projects added."""
        folder = os.path.normpath(os.path.abspath(folder))
        existing = self.root(folder)
        if existing is None:
            self.roots.append({"folder": folder, "scan": True, "lost": False})
        else:
            existing["scan"] = True
        self.excluded = [e for e in self.excluded if not inside(e, folder)]
        return self.refresh()

    def add_project(self, folder):
        """A project added alone (its folder, or the folder of a project file): its parent is its root, not
        scanned. Returns the project (None if the folder isn't one)."""
        folder = os.path.normpath(os.path.abspath(folder))
        info = detect(folder)
        if info is None:
            return None
        self.excluded = [e for e in self.excluded if key(e) != key(folder)]
        project = self.project(folder)
        if project is None:
            parent = os.path.dirname(folder)
            if self.root(parent) is None and not any(inside(folder, r["folder"]) for r in self.roots):
                self.roots.append({"folder": parent, "scan": False, "lost": False})
            root = next(r["folder"] for r in self.roots if inside(folder, r["folder"]))
            rel = os.path.relpath(folder, root)
            project = Project(folder, root, "" if rel == "." else rel.replace(os.sep, "/"))
            self.projects.append(project)
        self.update(project, info)
        return project

    @staticmethod
    def update(project, info):
        project.images, project.results, project.project_file, maps, _, project.signature, project.pipelines = info
        project.maps = maps
        project.lost = False

    def remove(self, projects):
        """Projects removed from the workspace (their files stay): not added again by the scans. The roots left
        without project and not scanned go too."""
        gone = {key(p.folder) for p in projects}
        self.projects = [p for p in self.projects if key(p.folder) not in gone]
        self.excluded += [p.folder for p in projects if p.folder not in self.excluded]
        self.roots = [r for r in self.roots if r["scan"] or any(key(p.root) == key(r["folder"])
                                                                for p in self.projects)]

    def remove_root(self, folder):
        """A root and its projects removed from the workspace."""
        self.roots = [r for r in self.roots if key(r["folder"]) != key(folder)]
        self.projects = [p for p in self.projects if key(p.root) != key(folder)]
        self.batch = [b for b in self.batch if key(b["root"]) != key(folder)]
        self.excluded = [e for e in self.excluded if not inside(e, folder)]

    def refresh(self):
        """Scan the roots again: the projects found added (not those removed by the user), those of the workspace
        updated, followed where they moved (see the module), or lost. Returns the projects added."""
        found = {}
        for r in self.roots:
            r["lost"] = not os.path.isdir(r["folder"])
            if r["scan"] and not r["lost"]:
                found.update({k: (r["folder"],) + v for k, v in scan(r["folder"]).items()})
        known = {key(p.folder) for p in self.projects}
        excluded = {key(e) for e in self.excluded}
        free = {k: v for k, v in found.items() if k not in known and k not in excluded}
        for p in self.projects:
            here = found.get(key(p.folder))
            info = here[3] if here else (detect(p.folder) if os.path.isdir(p.folder) else None)
            if info is not None:
                self.update(p, info)
                continue
            # Moved: the same path from its root, else the same name and contents
            same_path = [k for k, (root, _, path, _) in free.items() if key(root) == key(p.root) and path == p.path]
            same_contents = [k for k, (_, folder, _, info) in free.items()
                             if os.path.basename(folder) == os.path.basename(p.folder) and p.signature
                             and info[5] == p.signature]
            match = (same_path or same_contents or [None])[0]
            if match is None:
                p.lost = True
                continue
            root, folder, path, info = free.pop(match)
            p.folder, p.root, p.path = folder, root, path
            self.update(p, info)
        added = []
        for root, folder, path, info in free.values():
            project = Project(folder, root, path)
            self.update(project, info)
            self.projects.append(project)
            added.append(project)
        self.projects.sort(key=lambda p: ([key(r["folder"]) for r in self.roots].index(key(p.root))
                                          if self.root(p.root) else len(self.roots), p.path.lower()))
        return added

    # ------------------------------------------------------------------ relinking

    def move_paths(self, old, new):
        """The paths kept in the folder old (projects removed by the user, projects checked for the batches, roots
        of the stitchings) are in the folder new."""
        def moved(path):
            return os.path.normpath(os.path.join(new, os.path.relpath(path, old))) if inside(path, old) else path
        self.excluded = [moved(e) for e in self.excluded]
        self.checked = {step: [moved(f) for f in folders] for step, folders in self.checked.items()}
        for options in self.options.values():
            if not isinstance(options, dict):
                continue
            for chain in options.get("stitchings") or []:
                for node in chain:
                    if isinstance(node, dict) and node.get("root"):
                        node["root"] = moved(node["root"])

    def relink_root(self, root, folder):
        """A root found in another folder: its projects follow (the same paths from it). Returns the projects
        added by its scan."""
        folder = os.path.normpath(os.path.abspath(folder))
        self.move_paths(root, folder)
        for r in self.roots:
            if key(r["folder"]) == key(root):
                r["folder"] = folder
        for b in self.batch:
            if key(b["root"]) == key(root):
                b["root"] = folder
        for p in self.projects:
            if key(p.root) == key(root):
                p.root = folder
                p.folder = os.path.normpath(os.path.join(folder, *p.path.split("/"))) if p.path else folder
        return self.refresh()

    def relink_node(self, root, path, folder):
        """A folder of the tree (path from its root) found in another folder: the projects below it follow (their
        paths from it), their paths in the tree kept."""
        folder = os.path.normpath(os.path.abspath(folder))
        self.move_paths(os.path.join(root, *path.split("/")), folder)
        for p in self.projects:
            if key(p.root) == key(root) and (p.path == path or p.path.startswith(path + "/")):
                rest = p.path[len(path):].lstrip("/")
                p.folder = os.path.normpath(os.path.join(folder, *rest.split("/"))) if rest else folder
        return self.refresh()

    def relink_project(self, project, path):
        """A project found in another folder (or project file): None if it holds no project, else the project."""
        folder = os.path.dirname(path) if os.path.isfile(path) else path
        folder = os.path.normpath(os.path.abspath(folder))
        info = detect(folder)
        if info is None:
            return None
        project.folder = folder
        self.update(project, info)
        return project

    # ------------------------------------------------------------------ batch settings

    def node_settings(self, step, root, path):
        """The batch settings of a project (root, its path from it), None if none."""
        return next((b for b in self.batch if b["step"] == step and key(b["root"]) == key(root)
                     and b["path"] == path), None)

    def set_node_settings(self, step, root, path, settings):
        """The batch settings of a project (root, its path from it)."""
        entry = self.node_settings(step, root, path)
        if entry is None:
            self.batch.append({"step": step, "root": root, "path": path, "settings": settings})
        else:
            entry["settings"] = settings
