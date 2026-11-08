"""Project of the interface: the file VelociHub_project.json in the folder of the images (the project files of
the previous names of the tool, LEGACY_PROJECT_FILES: still read, then saved as VelociHub_project.json), which keeps what was done
on them and with which settings, so that it is restored when the images are opened again (or the project file is
opened):

- name: the name of the project, at the start of the names of the files exported (None: the name of the folder
  of the images);
- images: their names (folder: the one of the project file);
- preprocessing: masks, contrast enhancement, background removal, export folder, the spectrum and the modes saved
  (pod.npz) and the last export (from preprocessing.json);
- piv: camera, engine and its settings, results folder and the results (vectors.npz, with the run time);
- postprocessing: recipe, figures of the tiles, export folder and the last export (postprocessing.json);
- history: the steps done, with their date.

All the paths are saved relative to the image folder (the folder of the project file; ../ for the folders
outside it), so that the folder can be moved with its project. The files written next to the results
(preprocessing.json, settings.json and vectors.npz of the PIV, postprocessing.json) keep their paths relative to
the folder of the project too, which they give as measurement_folder, relative to their own folder.

The files exported by the steps are named [project]_[task]_[element]_[number].[format] (export_name): the name of
the project (the folder of the images by default), the step (PreProcessing, PIV or PostProcessing), what the file
holds (Image, MeanVelocity, ValidVectors...) and the number of the image or of the pair it belongs to, if any."""

import json
import os
import re
import time

PROJECT_FILE = "VelociHub_project.json"
PROJECT_FORMAT = "VelociHub project"
# The project files and their formats of the previous names of the tool (Pre_PIV_Post, POD_PAIRS): still read
LEGACY_PROJECT_FILES = ("PPP_project.json", "POD_PAIRS_project.json")
LEGACY_PROJECT_FORMATS = ("PPP project", "POD_PAIRS project")
# All the project files read, the current one first
PROJECT_FILES = (PROJECT_FILE, *LEGACY_PROJECT_FILES)
PROJECT_VERSION = 1


def default_project_name(folder):
    """Name of the project of the images of a folder: the name of the folder."""
    return os.path.basename(os.path.normpath(folder)) if folder else ""


def clean_name(name):
    """Name usable in file names: the characters that Windows forbids replaced by "-"."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", str(name or "")).strip(" .")


def export_name(project, task, element, number=None, extension="", total=None):
    """Name of a file exported: [project]_[task]_[element]_[number].[extension]. The number (of the image or of
    the pair, from 1) is left out if None; it has as many digits as total (zeros in front), so that the files
    sort in order."""
    parts = [clean_name(project) or "Project", task, element]
    if number is not None:
        parts.append(str(number).zfill(len(str(total or number))))
    return "_".join(parts) + ("." + extension.lstrip(".") if extension else "")


def format_duration(seconds):
    """Duration as shown to the user: "45 s", "7 min 02 s" or "1 h 05 min". None: ""."""
    if seconds is None:
        return ""
    seconds = int(round(float(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def run_time(settings):
    """Run time of a PIV (s) from its settings (settings.json): run_time_s, or the minutes of the results saved
    before it was kept. None if unknown."""
    if settings.get("run_time_s") is not None:
        return float(settings["run_time_s"])
    if settings.get("minutes") is not None:
        return 60.0 * float(settings["minutes"])
    return None


def valid_text(valid_percent, before=", "):
    """The valid vectors (%) of a PIV in a sentence (after before); nothing for a PTV (valid_percent None)."""
    return "" if valid_percent is None else f"{before}{valid_percent:.1f} % valid vectors"


def piv_results(file, settings, valid_percent=None):
    """The results of a PIV as kept in the project (section piv, results): the file of the vector fields (as
    saved in the project), its date, number of pairs, valid vectors (%; a PIV only) and run time, from its settings."""
    seconds = run_time(settings)
    results = {"file": file, "date": settings.get("date"), "n_pairs": settings.get("n_pairs")}
    if valid_percent is not None:
        results["valid_percent"] = round(float(valid_percent), 1)
    if seconds is not None:
        results.update(run_time_s=round(seconds, 1), run_time=format_duration(seconds))
        if settings.get("seconds_per_pair") is not None:
            results["seconds_per_pair"] = settings["seconds_per_pair"]
        if settings.get("resumed_pairs"):
            results["resumed_pairs"] = settings["resumed_pairs"]
    return results


def relative(path, root):
    """Path saved in the project: relative to its folder (root), with ../ outside it; absolute only on another
    drive, or without a folder (no images open: no project)."""
    if not path:
        return None
    path = os.path.abspath(path)
    if not root:
        return path
    try:
        return os.path.relpath(path, os.path.abspath(root)).replace("\\", "/")
    except ValueError:  # Another drive
        return path


def comparable_recipe(record):
    """A recipe record (recipe_record) without the training folder that the previous versions kept in it (an
    absolute path): the records of the same recipe are equal wherever the folder of the project is."""
    if not record or not record.get("background_removal"):
        return record
    removal = {k: v for k, v in record["background_removal"].items() if k != "training_folder"}
    return {**record, "background_removal": removal}


def absolute(path, root):
    """Path read from the project (relative to its folder, root)."""
    if not path:
        return None
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(root, path))


def read_project(path):
    """Project of a folder (or of a project file): its dictionary, None if there is none. ValueError if the file
    isn't a project."""
    if os.path.isdir(path):
        # The project file, else the one of the previous versions (saved as the project file from then on)
        found = [os.path.join(path, n) for n in PROJECT_FILES
                 if os.path.isfile(os.path.join(path, n))]
        path = found[0] if found else os.path.join(path, PROJECT_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("format") not in (PROJECT_FORMAT, *LEGACY_PROJECT_FORMATS):
        raise ValueError(f"{path} is not a {PROJECT_FORMAT} file")
    return data


def write_project(path, data):
    """Save a project (written next to the file and renamed, so that an interrupted save keeps the previous one)."""
    data = {"format": PROJECT_FORMAT, "version": PROJECT_VERSION, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            **data}
    temporary = path + ".part"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(temporary, path)
    return path


def summary(data):
    """What was done in a project, as lines of text."""
    lines = [f"Project name: {data['name']}"] if data.get("name") else []
    images = data.get("images", {})
    files = images.get("files", [])
    if files:
        lines.append(f"Images: {len(files)} ({files[0]} ... {files[-1]})")
    pre = data.get("preprocessing", {})
    pod = pre.get("pod")
    if pod:
        lines.append(f"POD: spectrum and {pod.get('modes_evaluated', 0)} modes evaluated ({pod.get('file')})")
    export = pre.get("export")
    if export:
        lines.append(f"Pre-processing: {export.get('n_images')} images exported on {export.get('date')} to "
                     f"{pre.get('export_folder')} ({export.get('steps') or 'no step'})")
    results = data.get("piv", {}).get("results")
    if results:
        lines.append(f"PIV/PTV: {results.get('n_pairs')} pairs on {results.get('date')}"
                     + valid_text(results.get("valid_percent"))
                     + (f", run time {results['run_time']}" if results.get("run_time") else "")
                     + f" ({results.get('file')})")
    post = data.get("postprocessing", {}).get("export")
    if post:
        lines.append(f"Post-processing: figures exported on {post.get('date')} to "
                     f"{data['postprocessing'].get('export_folder')} ({post.get('recipe')})")
    return lines
