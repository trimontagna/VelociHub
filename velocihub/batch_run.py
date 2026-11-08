"""Batch runs of the interface, in their own process so that the window stays usable:

    python -m velocihub.batch_run job.json

The job (written by gui/batch_dialog.py): {"step": "preprocessing", "piv", "postprocessing" or "archive",
"projects": [{"folder", "label", "settings": {...}}], "options": {...}}. The settings of each project are those of
its node of the workspace (or of the interface), as the step needs them:

- preprocessing: {"preprocessing": section of the project} (batch.preprocess_folder);
- piv: {"preprocessing": section, "piv": {"engine", "options", "settings", "output_folder"}} (batch.process_folder);
- postprocessing: {"postprocessing": {"settings": PostSettings as a dictionary, "space", "camera":
  settings.CAMERA_KEYS (fov_width_mm, fov_height_mm, offsets, shifts, frame_rate_hz) or None, "export_folder", "export": export
  specification}}; with stitchings, options: "globals" (of velocihub/archiving/post_archive.py) and "units" (gui/stitching.py:
  the projects outside the stitchings one by one, then each chain of projects stitched: run_stitching);
- archive: {"archive": {"categories": [...]}} (velocihub/archiving/archive.py; "outside": the files of the root of the project
  outside the projects), options: output, format and projects (all those of the workspace: not outside).

The window reads the lines "@@project i n label", "@@progress fraction message", "@@result status message" and
"@@preview path" (an image just written, shown by the window: a figure, a pre-processed image, the frame of a video,
at most every PREVIEW_EVERY s; previews_on); the other lines are its log. Ctrl+Break (the button Stop) stops the run: the step running ends there (KeyboardInterrupt;
the PIV keeps its chunks of pairs computed, the archive its file is removed) and the next projects aren't run."""

import json
import os
import signal
import sys
import tempfile
import threading
import time
import traceback

from . import batch, piv
from .fields import count_maps, load_vectors
from .pipelines import chosen_results, pipeline_of, results_folders
from .postprocessing import PostProcessing, PostSettings, export_folder_name, previous_results, remove_results
from .project import format_duration, read_project
from .settings import camera_of, camera_record

def emit(*parts):
    print("@@" + " ".join(str(p) for p in parts), flush=True)


# The previews of the files written: at most every PREVIEW_EVERY s (2 s for the frames of the videos, drawn for it)
PREVIEW_EVERY = 1.0
VIDEO_PREVIEW_EVERY = 2.0
PREVIEW_FILE = os.path.join(tempfile.gettempdir(), f"velocihub_batch_preview_{os.getpid()}.png")


def previews_on():
    """The previews of the window: the figures saved (PNG), the images exported by the pre-processing, the frames
    of the videos (a small copy of the frame of a GIF; an MP4 frame drawn again at a low resolution)."""
    import matplotlib.figure

    from . import fields, preprocessing

    last = {"file": 0.0, "video": 0.0}

    def due(kind, every):
        if time.monotonic() - last[kind] < every:
            return False
        last[kind] = time.monotonic()
        return True

    savefig = matplotlib.figure.Figure.savefig

    def saved(self, fname, *args, **kwargs):
        savefig(self, fname, *args, **kwargs)
        if isinstance(fname, str) and fname.lower().endswith(".png") and fname != PREVIEW_FILE and due("file", PREVIEW_EVERY):
            emit("preview", fname)
    matplotlib.figure.Figure.savefig = saved

    imwrite = preprocessing.imwrite

    def written(A, filename):
        imwrite(A, filename)
        if due("file", PREVIEW_EVERY):
            emit("preview", filename)
    preprocessing.imwrite = written

    add = fields.Video.add

    def frame(self, draw, k):
        add(self, draw, k)
        if not due("video", VIDEO_PREVIEW_EVERY):
            return
        frames = getattr(self.writer, "frames", None)
        if frames:
            image = frames[-1].convert("RGB")
            image.thumbnail((640, 400))
            image.save(PREVIEW_FILE)
        else:
            savefig(self.fig, PREVIEW_FILE, dpi=40)
        emit("preview", PREVIEW_FILE)
    fields.Video.add = frame


class Progress:
    """Progress of a step printed for the window (at most every 0.25 s)."""

    def __init__(self, title):
        self.title, self.last = title, 0.0

    def __call__(self, x, message=None):
        if time.monotonic() - self.last >= 0.25 or x >= 1:
            self.last = time.monotonic()
            emit("progress", f"{float(x):.4f}", f"{self.title}: {message or ''}")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass


def inside(folder, path, default):
    """A folder of the settings in a project: path if it is inside it (relative), else default."""
    ok = isinstance(path, str) and path and not os.path.isabs(path) and not path.startswith("..")
    return os.path.normpath(os.path.join(folder, path if ok else default))


def post_export_folder(folder, s, maps):
    """The export folder of the post-processing of a project (s: its settings; maps: the folder of its vector
    fields): the one of its settings (with several results sets: the subfolder of their pipeline in it), else the
    subfolder of their pipeline in PostProcessing."""
    out = inside(folder, export_folder_name(s.get("export_folder"), maps), export_folder_name(None, maps))
    label = pipeline_of(maps)
    if len(s.get("maps_folders") or {}) > 1 and label and os.path.basename(os.path.normpath(out)) != label:
        out = os.path.join(out, label)
    return out


def post_datasets(folder, s, pipelines=None):
    """The vector fields of the results sets of a project post-processed (s: its settings; pipelines: only those):
    {label: folder} (maps_folders, else the maps folder of post_inputs)."""
    datasets = s.get("maps_folders")
    if not datasets:
        maps = post_inputs(folder, s)[0]
        datasets = {pipeline_of(maps) or "": maps}
    return {label: maps for label, maps in datasets.items() if pipelines is None or label in pipelines}


def project_name(folder):
    """The name of a project: the one of its project file, else its folder."""
    try:
        name = (read_project(folder) or {}).get("name")
    except (OSError, ValueError):
        name = None
    return name or os.path.basename(os.path.normpath(folder))


def post_inputs(folder, s):
    """The vector fields, the camera and the recipe of the post-processing of a project (s: its settings): those of
    its maps folder, else of the pipeline chosen (results; the default one: those of the first pipeline found if it
    has none)."""
    maps = (s.get("maps_folder") or next(iter((s.get("maps_folders") or {}).values()), None)
            or chosen_results(results_folders(folder), s.get("results")) or os.path.join(folder, piv.PIV_FOLDER))
    names = set(PostSettings.__dataclass_fields__)
    settings = PostSettings(**{k: v for k, v in (s.get("settings") or {}).items() if k in names})
    camera = s.get("camera") if s.get("space", "physical") == "physical" else None
    camera = camera_of(camera) if camera else None
    return maps, camera, settings


# ------------------------------------------------------------------ steps of a project

def run_preprocessing(project, job, progress=Progress):
    result = batch.preprocess_folder(project["folder"], project["settings"]["preprocessing"], progress=progress)
    return result.status, result.message


def run_piv(project, job, progress=Progress):
    s = project["settings"]
    p = s["piv"]
    engine = piv.get_engine(p.get("engine"), **(p.get("options") or {}))
    result = batch.process_folder(project["folder"], engine=engine, settings=p.get("settings") or {},
                                  preprocessing=s["preprocessing"], progress=progress,
                                  output_folder=p.get("output_folder"), force=bool(job["options"].get("force")),
                                  report=print, regression=p.get("regression"))
    return result.status, result.describe()


def run_postprocessing(project, job, progress=Progress, pipelines=None):
    """The post-processing of a project: the same export for each of its results sets (pipelines: only those; the
    others are those of its stitchings)."""
    from .settings import export_choice, export_index, export_options

    folder = project["folder"]
    s = project["settings"]["postprocessing"]
    datasets = post_datasets(folder, s, pipelines)
    if not datasets:
        return "failed", "no results of the results sets chosen"
    t0, messages = time.time(), []
    for label, maps in datasets.items():
        _, camera, settings = post_inputs(folder, {**s, "maps_folder": maps})
        if s.get("space", "physical") == "physical" and camera is None:
            return "failed", "physical space without camera settings (field of view width, frame rate)"
        fields = load_vectors(os.path.join(maps, piv.VECTORS_FILE), camera=camera, pixels=camera is None)
        post = PostProcessing(fields, settings)
        n = fields.n
        # The map of the instantaneous figures: the one chosen in the quantities, else the last one
        index = export_index(export_choice((s.get("export") or {}).get("choice")))
        export_folder = post_export_folder(folder, s, maps)
        if job["options"].get("overwrite"):  # The results of a previous export replaced (asked in the window Batch)
            remove_results(previous_results(export_folder, project_name(folder)))
        files = post.export(export_folder, index if -n <= index < n else 0, name=project_name(folder),
                            progress=progress(f"Export {label}".strip()) if progress else None,
                            **export_options(s.get("export") or {}, n))
        messages.append(f"{label + ': ' if label else ''}{len(files)} files")
    return "done", f"{', '.join(messages)} exported in {format_duration(time.time() - t0)}"


# ------------------------------------------------------------------ steps of the whole job

def run_archive(job):
    from .archiving import archive as ap

    options = job["options"]
    units, by_folder = [], {}
    for project in job["projects"]:
        unit = ap.scan_project(project["folder"], project.get("root") or os.path.dirname(project["folder"])) \
            if any(os.path.isfile(os.path.join(project["folder"], n)) for n in ap.PROJECT_FILES) \
            else bare_unit(ap, project)
        units.append(unit)
        by_folder[ap.path_key(unit.folder)] = project["settings"]["archive"]["categories"]
    # The files of the roots outside the projects: not those of the other projects of the workspace either
    projects = [ap.path_key(f) for f in options.get("projects") or []]
    # The roots whose files outside the projects are archived: those of the projects with the category "outside"
    roots_outside = sorted({project.get("root") or os.path.dirname(project["folder"]) for project in job["projects"]
                            if "outside" in project["settings"]["archive"]["categories"]})
    for root in roots_outside:
        found, _ = ap.scan([root])
        for unit in found:
            if unit.kind == "outside":
                for files in unit.files.values():
                    for path in [p for p in files if any(ap.path_key(p).startswith(f + os.sep) for f in projects)]:
                        del files[path]
                units.append(unit)
                by_folder[ap.path_key(unit.folder)] = ["outside"]

    def categories(unit):
        return by_folder.get(ap.path_key(unit.folder), [])
    base, entries = ap.archive_entries(units, categories)
    # Not the archive itself (an archive written before in a root)
    output = ap.path_key(options["output"])
    entries = [e for e in entries if ap.path_key(e[0]) not in (output, output + ".part")]
    progress = ap.Progress(sum(size for _, _, size in entries))
    stop = threading.Event()

    def show():
        while not stop.wait(0.5):
            emit("progress", f"{progress.done / max(progress.total, 1):.4f}", f"Archive: {progress.describe()}")
    threading.Thread(target=show, daemon=True).start()
    print(f"{len(entries)} files ({ap.format_size(progress.total)}) from {len(units)} folders into {options['output']}")
    try:
        errors = ap.write_archive(options["output"], entries, options["format"],
                                  lambda errors: ap.archive_summary(units, categories, base, options["format"], errors),
                                  progress)
    except (KeyboardInterrupt, ap.Cancelled):
        emit("result", "stopped", "archive stopped (its file removed)")
        return
    finally:
        stop.set()
    emit("result", "done", f"{len(entries)} files archived in {options['output']}"
         + (f", {len(errors)} unreadable" if errors else ""))


def bare_unit(ap, project):
    """What can be archived of a project without project file (PIV results only): the files of its folder and of
    its subfolder PIV."""
    unit = ap.Unit(os.path.abspath(project["folder"]), project.get("root") or os.path.dirname(project["folder"]))
    unit.name = os.path.basename(unit.folder)
    for folder in (unit.folder, os.path.join(unit.folder, piv.PIV_FOLDER)):
        for name in ap.files_of(folder):
            category = ("vectors" if name == piv.VECTORS_FILE else "csv" if name.lower().endswith(".csv")
                        else "project" if name.lower().endswith(".json")
                        else "originals" if ap.is_image(name) else "other")
            unit.add(category, os.path.join(folder, name))
    return unit


def run_stitching(job, stitching):
    """A chain of projects post-processed and stitched (archiving/post_archive.main; stitching: {"setup", "members":
    [{"folder", "root", "fov"}], the reference first, "pipeline": the results set stitched, None: the maps folder of
    each}): the quantities of the reference exported for all of them, the stitched figures in the folder Stitched of
    its export folder."""
    from .settings import export_choice, export_index, export_options

    from .archiving import post_archive as r

    for name, value in (job["options"].get("globals") or {}).items():
        if not name.startswith(("MATCH_OFFSETS", "MATCH_VERTICAL", "MATCH_SEARCH", "MATCH_MIN", "CALIBRATION")):
            setattr(r, name, value)  # Not those of the offsets measured before (saved settings)
    members = stitching["members"]
    r.REFERENCE_FOV = members[0]["fov"]
    by_folder = {os.path.normcase(os.path.normpath(p["folder"])): p for p in job["projects"]}
    projects = [by_folder[os.path.normcase(os.path.normpath(m["folder"]))] for m in members]
    # The quantities exported (export specification): those of the reference, for all the projects of the chain
    spec = projects[0]["settings"]["postprocessing"].get("export") or {}
    items = []
    for member, project in zip(members, projects):
        folder = os.path.normpath(project["folder"])
        s = project["settings"]["postprocessing"]
        label = stitching.get("pipeline")
        if label:
            if label not in (s.get("maps_folders") or {}):
                return "failed", f"{member['fov']}: no {label} results"
            s = {**s, "maps_folder": s["maps_folders"][label]}
        maps, camera, settings = post_inputs(folder, s)
        if camera is None:
            return "failed", f"{member['fov']}: no camera settings (the stitching needs the physical space)"
        n = count_maps(os.path.join(maps, piv.VECTORS_FILE))
        export = export_options(spec, n)
        export.update(export_folder=post_export_folder(folder, s, maps),
                      maps_folder=maps, name=project_name(folder))
        items.append({"folder": folder, "fov": member["fov"], "setup": stitching["setup"], "settings": settings,
                      "export": export, "index": export_index(export_choice(spec.get("choice"))),
                      "camera": camera_record(camera)})
    stitched = os.path.join(items[0]["export"]["export_folder"], "Stitched")
    items[0]["stitched_folder"] = stitched
    if job["options"].get("overwrite"):  # The results of a previous export replaced (asked in the window Batch)
        for item in items:
            remove_results(previous_results(item["export"]["export_folder"], item["export"]["name"]))
        remove_results(previous_results(stitched, stitching["setup"], "Stitched", records=False))
    ok = r.main(items=items, stitched_folder=stitched)
    return ("done" if ok else "failed"), f"{len(items)} projects post-processed, stitched in {stitched}"


PROJECT_STEPS = {"preprocessing": run_preprocessing, "piv": run_piv, "postprocessing": run_postprocessing}
JOB_STEPS = {"archive": run_archive}


def stop_on_break():
    """Ctrl+Break (Windows: the button Stop of the window) raises KeyboardInterrupt, as Ctrl+C."""
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, interrupt)


def main(path):
    with open(path, encoding="utf-8") as f:
        job = json.load(f)
    job.setdefault("options", {})
    stop_on_break()
    previews_on()
    step = job["step"]
    if step in JOB_STEPS:
        emit("project", 1, 1, step)
        try:
            JOB_STEPS[step](job)
        except KeyboardInterrupt:
            emit("result", "stopped", "stopped")
        except Exception as e:
            traceback.print_exc()
            emit("result", "failed", f"{type(e).__name__}: {e}")
        return 0
    # The units of the run: the projects one by one, else (options units: the post-processing with stitchings) the
    # projects outside the stitchings, then each stitching
    projects = {os.path.normcase(os.path.normpath(p["folder"])): p for p in job["projects"]}
    units = job["options"].get("units") or [{"label": p.get("label") or p["folder"], "folders": [p["folder"]]}
                                            for p in job["projects"]]
    for i, unit in enumerate(units):
        emit("project", i + 1, len(units), unit["label"])
        print(f"[{i + 1}/{len(units)}] {unit['label'] if 'stitching' in unit else unit['folders'][0]}", flush=True)
        try:
            if "stitching" in unit:
                status, message = run_stitching(job, unit["stitching"])
            else:
                project = projects[os.path.normcase(os.path.normpath(unit["folders"][0]))]
                if step == "postprocessing" and unit.get("pipelines") is not None:
                    # The results sets of a project that aren't stitched (those stitched: in their stitchings)
                    status, message = run_postprocessing(project, job, pipelines=unit["pipelines"])
                else:
                    status, message = PROJECT_STEPS[step](project, job)
        except KeyboardInterrupt:
            emit("result", "stopped", "stopped by the user: the next projects aren't run")
            break
        except Exception as e:  # The next projects are run
            traceback.print_exc()
            status, message = "failed", f"{type(e).__name__}: {e}"
        emit("result", status, message)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
