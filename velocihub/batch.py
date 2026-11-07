"""Batch processing of measurement folders: for each folder of training images, the pre-processing (POD background
removal) and the PIV with the default settings of the interface, saved as the interface saves them: the images
pre-processed and the POD in the subfolder PreProcessing, the vector fields in the subfolder PIV, and the project
(VelociHub_project.json), so that the folder opened in the interface shows them.

    import velocihub as vh

    for result in vh.batch.process_folders(["X:/FOV_3"]):  # Each measurement folder of FOV_3
        print(result.describe())

The default recipe is the one the interface starts with (DEFAULT_CONTRAST: intensity capping, n = 10, after the
background removal of DEFAULT_MODES = 6 POD modes; no masks) and the default settings of the PIV engine
(PaIRS-UniNa: windows 128 64 32 px, 50 % overlap...). A folder whose PIV results were computed with the same
images, recipe and settings is skipped (force=True computes it again); the images pre-processed and the POD saved
are reused as in the interface, and a PIV stopped continues from its last chunk of pairs saved.

From the command line: velocihub-batch folder [folder ...]

The post-processing of the PIV results of measurement folders (postprocess_folders), with the camera settings of
each (its field of view) and a recipe, exports their figures as the interface does:

    camera = vh.Camera(fov_width_mm=196.1, x_offset_mm=0, frame_rate_hz=150)
    for result in vh.batch.postprocess_folders(folders, camera, vh.PostSettings(outlier_k=4)):
        print(result.describe())
"""

import json
import os
import shutil
import time
from dataclasses import dataclass, fields as dataclass_fields

from PIL import Image

from . import piv
from .fields import EXPORT_DPI, load_vectors
from .postprocessing import PostProcessing, PostSettings, default_export_folder
from .piv.particles import DEFAULT_DIAMETER, DEFAULT_SNR
from .preprocessing import (DEFAULT_CONTRAST, DEFAULT_MODES, POD_FILE, PREPROCESSING_FOLDER, ContrastSettings, Mask,
                            PODBackgroundRemoval, list_files, list_images, num2str, read_manifest, str2double)
from .project import (PROJECT_FILE, clean_name, comparable_recipe, format_duration, piv_results, read_project,
                      relative, run_time, write_project)

# Copy of the project of a folder kept when the batch replaces settings that differ (masks, recipe, PIV settings)
BACKUP_SUFFIX = ".bak"
# The file of masks at the root of a folder used if there are several (the name the interface proposes)
MASKS_FILE = "masks.json"


def folder_has_images(folder, pattern=None):
    """At least two images (matching the pattern) in the folder."""
    try:
        return len(list_files(folder, pattern) if pattern else list_images(folder)) >= 2
    except OSError:
        return False


def measurement_folders(root, pattern=None):
    """Folders of training images of a root folder: the root itself if it holds images (at least two, matching
    the pattern), or else its subfolders that hold images (one level down, sorted by name)."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise ValueError(f"The folder doesn't exist: {root}")
    if folder_has_images(root, pattern):
        return [root]
    subfolders = sorted(os.path.join(root, name) for name in os.listdir(root)
                        if os.path.isdir(os.path.join(root, name)))
    return [folder for folder in subfolders if folder_has_images(folder, pattern)]


def find_masks(root):
    """The file of masks at the root of a folder: its JSON file with a list "masks" (saved by the interface, Export
    masks..., or by save_masks), MASKS_FILE if there are several; None if there is none. ValueError if there are
    several and none of them is MASKS_FILE."""
    root = os.path.abspath(root)
    found = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not name.lower().endswith(".json") or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                content = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(content, dict) and isinstance(content.get("masks"), list):
            found.append(path)
    if len(found) > 1:
        found = [path for path in found if os.path.basename(path).lower() == MASKS_FILE]
        if not found:
            raise ValueError(f"several files of masks at the root of {root}: name the one to use {MASKS_FILE}")
    return found[0] if found else None


def masks_from_file(path):
    """The masks of a file of masks (saved by the interface, Export masks..., or by save_masks): [{"name", "enabled",
    "points"}]. ValueError if it isn't one."""
    try:
        with open(path, encoding="utf-8") as f:
            content = json.load(f)
    except OSError as e:
        raise ValueError(f"the file of masks can't be read: {e}") from None
    except ValueError:
        raise ValueError(f"{os.path.basename(path)} is not a JSON file") from None
    if not isinstance(content, dict) or not isinstance(content.get("masks"), list):
        raise ValueError(f"{os.path.basename(path)} is not a file of masks (no list \"masks\")")
    return [{"name": m.get("name", "Mask"), "enabled": bool(m.get("enabled", True)), "points": m["points"]}
            for m in content["masks"]]


def measurement_masks(folder, root_masks=None):
    """The file of masks of a measurement folder: its own (find_masks, more specific to it), else root_masks (the
    one of its root)."""
    return find_masks(folder) or root_masks


@dataclass
class BatchResult:
    """Result of the processing of a folder: status "done", "skipped" (results up to date), "stopped" (the PIV
    stopped: started again, it continues from the chunks of pairs saved) or "failed" (message: why); the vector
    fields (vectors.npz), their pairs, % of valid vectors, the run time of the PIV and the time of the
    pre-processing (s)."""

    folder: str
    status: str
    vectors: str = None
    n_pairs: int = None
    valid_percent: float = None
    piv_seconds: float = None
    preprocessing_seconds: float = None
    message: str = ""

    def describe(self):
        if self.status in ("done", "skipped") and self.n_pairs is None:  # Pre-processing only
            return f"{self.status}: {self.message}"
        if self.status in ("done", "skipped"):
            text = (f"{self.status}: {self.n_pairs} pairs, {self.valid_percent:.1f} % valid vectors"
                    + (f", PIV/PTV {format_duration(self.piv_seconds)}" if self.piv_seconds is not None else ""))
            if self.preprocessing_seconds is not None:
                text += f", pre-processing {format_duration(self.preprocessing_seconds)}"
            return text
        return f"{self.status}: {self.message}"


def engine_settings_text(engine, settings):
    """Settings of the engine as the interface saves them in the project (texts)."""
    return {p.key: p.format(settings[p.key]) for p in engine.parameters}


def up_to_date(output, engine, settings, recipe, n_pairs, regression=None):
    """The PIV results of the output folder were computed with this engine, its settings, the images of this recipe,
    as many pairs and this regression (checked: piv.check_regression)."""
    try:
        with open(os.path.join(output, "settings.json"), encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return False
    return (os.path.isfile(os.path.join(output, piv.VECTORS_FILE)) and saved.get("engine") == engine.name
            and saved.get("engine_settings") == json.loads(json.dumps(settings))
            and comparable_recipe(saved.get("preprocessing_recipe")) == comparable_recipe(recipe)
            and saved.get("n_pairs") == n_pairs and saved.get("regression") == piv.regression_record(regression))


def settings_signature(data):
    """What the batch replaces in a project: the masks, the recipe, the settings of the PIV engine and the
    regression."""
    pre, piv_settings = data.get("preprocessing") or {}, (data.get("piv") or {}).get("settings") or {}
    bgrem = pre.get("background_removal") or {}
    # No regression, whether saved as none or not saved (projects before the regression)
    regression = piv_settings.get("regression") or {}
    return json.dumps([pre.get("masks") or [], pre.get("contrast"), bgrem.get("enabled"), bgrem.get("modes_removed"),
                       piv_settings.get("engine"), piv_settings.get("engine_settings"),
                       regression if regression.get("method", "none") != "none" else None], sort_keys=True)


def valid_percent(fields):
    """Valid vectors (%) of the vector fields, in the flow (as the interface shows it)."""
    return 100 * float(fields.piv_valid[:, fields.status == 0].mean())


def save_project(session, n_modes, engine, settings, fields, pod_file, history, report=print, regression=None):
    """Save the project of the folder of the training images as the interface does, with the recipe, the PIV
    settings and the results (fields: the vector fields saved, VectorFields; pod_file: the POD saved, None if
    there is none; regression: the one after the PIV, checked). The project of the same images is updated (its
    name, post-processing and history are kept; a copy is kept if its masks, recipe or PIV settings were others),
    that of other images replaced. Returns the file of the project."""
    folder = session.path_in
    path = os.path.join(folder, PROJECT_FILE)
    try:
        previous = read_project(folder) or {}
    except (OSError, ValueError):
        previous = {}
    same = previous.get("images", {}).get("files") == session.file_in
    state = {k: v for k, v in previous.items() if k not in ("format", "version", "saved")} if same else {}

    def rel(p):
        return relative(p, folder)

    contrast = {f.name: getattr(session.contrast, f.name) for f in dataclass_fields(ContrastSettings)}
    contrast["filters"] = list(session.contrast.filters)
    manifest = read_manifest(session.export_folder())
    export = None
    if manifest is not None and manifest.get("source_files") == session.file_in:
        export = {"date": manifest.get("date"), "steps": manifest.get("steps"),
                  "n_images": len(manifest.get("files", [])), "recipe": comparable_recipe(manifest.get("recipe"))}
    pre = state.get("preprocessing") or {}
    state["preprocessing"] = {
        "masks": [{"name": m.name, "enabled": m.enabled, "points": m.points.tolist()} for m in session.masks],
        "contrast": contrast,
        "background_removal": {"enabled": bool(n_modes), "modes_removed": str(n_modes),
                               "modes_to_evaluate": str(max(n_modes, session.NMod_saved))},
        "automatic_threshold": pre.get("automatic_threshold") or {"eps_sigma": num2str(session.eps_auto_sigma),
                                                                  "eps_psi": num2str(session.eps_auto_psi)},
        "export_folder": rel(session.export_folder()),
        "export": export,
        "pod": {"file": rel(pod_file), "modes_evaluated": session.NMod_saved} if pod_file else pre.get("pod")}
    old_piv = state.get("piv") or {}
    particles = old_piv.get("settings") or {}
    state["piv"] = {
        "settings": {"particle_snr": particles.get("particle_snr", num2str(DEFAULT_SNR)),
                     "particle_diameter": particles.get("particle_diameter", num2str(DEFAULT_DIAMETER)),
                     "engine": engine.name,
                     "engine_settings": {**(particles.get("engine_settings") or {}),
                                         engine.key: engine_settings_text(engine, settings)},
                     "regression": piv.regression_text(regression)},
        "images_folder": None,
        "output_folder": rel(os.path.dirname(fields.source)),
        "results": piv_results(rel(fields.source), {**fields.settings, "n_pairs": fields.n}, valid_percent(fields)),
        "window_center": old_piv.get("window_center")}
    state["images"] = {"folder": ".", "files": session.file_in}
    state.setdefault("name", None)
    state.setdefault("image_shown", 1)
    state["history"] = (list(state.get("history") or []) + history)[-500:]
    if previous and settings_signature(previous) != settings_signature(state):
        shutil.copyfile(path, path + BACKUP_SUFFIX)
        report(f"  The project had other settings (masks, recipe or PIV/PTV): copy kept in {path + BACKUP_SUFFIX}")
    return write_project(path, state)


def process_folder(folder, pattern=None, engine=None, settings=None, n_modes=DEFAULT_MODES,
                   contrast=DEFAULT_CONTRAST, masks=None, temp_folder="images", force=False, report=None,
                   verbose=True, preprocessing=None, progress=None, output_folder=None, regression=None):
    """Pre-process the training images of a folder (all its images, or those matching the pattern) and compute
    their PIV, with the default recipe and PIV settings unless others are given: n_modes POD modes removed (0: no
    background removal), contrast (ContrastSettings, None: no contrast enhancement), masks (file saved by the
    interface or save_masks, None: no masks), engine and settings (of the PIV engine: its defaults for the
    others). The temporary files of the POD are in the folder of the images (temp_folder="images", as in the
    interface), in temp_folder, or in a new temporary folder (None). force: compute the PIV even if its results
    are up to date. report(message): the messages of the PIV (printed by default). preprocessing: the settings of
    the pre-processing of a project section instead of n_modes, contrast and masks (apply_preprocessing); progress:
    the progress of the session (preprocess_folder); output_folder: the folder of the PIV results relative to the
    folder of the images (PIV by default; a path outside it: PIV); regression: of the vector fields after the PIV
    ({"method": "spicy", "settings": {...}}; None: none). Returns a BatchResult."""
    say = print if verbose else (lambda *args: None)
    engine = piv.get_engine(engine)
    settings = engine.check(settings or {})
    regression = piv.check_regression(regression)
    if regression is not None and not piv.spicy_available()[0]:
        raise RuntimeError(piv.spicy_available()[1])
    ok, reason = engine.available()
    if not ok:
        raise RuntimeError(reason)
    with PODBackgroundRemoval(folder, pattern, temp_folder=temp_folder, verbose=verbose) as session:
        if progress is not None:
            session.progress = progress
        try:
            previous = read_project(session.path_in) or {}
        except (OSError, ValueError):
            previous = {}
        # The name of the project of the same images, at the start of the files exported
        if previous.get("images", {}).get("files") == session.file_in:
            session.project = clean_name(previous.get("name")) or None
        if preprocessing is not None:
            n_modes = apply_preprocessing(session, preprocessing)
        else:
            session.contrast, session.contrast_enabled = contrast or ContrastSettings(), contrast is not None
            if masks is not None:
                session.load_masks(masks)
            n_modes = int(min(max(n_modes, 0), session.NImg))
        recipe = session.recipe_record(n_modes)
        inside = (isinstance(output_folder, str) and output_folder and not os.path.isabs(output_folder)
                  and not output_folder.startswith(".."))
        output = os.path.normpath(os.path.join(session.path_in, output_folder if inside else engine.results_folder))
        n_pairs = len(piv.Camera(1.0).pairs(session.NImg))
        if not force and up_to_date(output, engine, settings, recipe, n_pairs, regression):
            fields = load_vectors(os.path.join(output, piv.VECTORS_FILE), pixels=True)
            return BatchResult(session.path_in, "skipped", fields.source, fields.n, valid_percent(fields),
                               run_time(fields.settings), message="results up to date (force to compute them again)")

        # Pre-processing: the images exported with the recipe are reused, or else the POD saved (pod.npz), or else
        # the modes are evaluated
        history = []
        if previous.get("images", {}).get("files") != session.file_in:
            history.append(entry("images", f"{session.NImg} images: {session.file_in[0]} ... "
                                           f"{session.file_in[-1]}"))
        t0 = time.time()
        pod_file = os.path.join(session.export_folder(), POD_FILE)
        reused = session.preprocessed_images(n_modes) is not None
        if not reused and n_modes and (session.load_pod(pod_file) or 0) < n_modes:
            if session.evaluate_modes(n_modes) is None:
                raise RuntimeError("The POD modes could not be evaluated")
            session.save_pod(pod_file)
            history.append(entry("modes", f"Batch: spectrum and {session.NMod_saved} POD modes evaluated"))
        images = session.piv_images(n_modes)
        if images is None:
            raise RuntimeError("The images could not be pre-processed")
        steps = " → ".join(session.recipe_steps(n_modes, detailed=True))
        if not reused and not session.recipe_empty(n_modes):
            history.append(entry("export", f"Batch: {len(images)} images exported to "
                                           f"{relative(session.export_folder(), session.path_in)} "
                                           f"({steps})"))
        pre_seconds = time.time() - t0
        say(f"  Pre-processing: {steps or 'none (original images)'} ({format_duration(pre_seconds)})")

        # PIV, the vectors whose window center is masked marked as such (as in the interface)
        masked = None
        if session.masks_active():
            with Image.open(images[0]) as im:
                masked = session.masked_pixels(im.size[::-1])
        info = {"measurement_folder": session.path_in, "preprocessing": steps or "none (original images)",
                "preprocessing_recipe": recipe}
        run = piv.PIVRun(images, None, engine, settings, output, masked=masked, info=info,
                         project=session.project_name(), regression=regression)
        say(f"  PIV/PTV of {len(run.pairs)} pairs ({engine.name}, {engine.describe(settings)}) in {output}"
            + (f", then {piv.describe_regression(regression)}" if regression else ""))
        vectors = run.run(report)
        if vectors is None:
            return BatchResult(session.path_in, "stopped", preprocessing_seconds=pre_seconds,
                               message="PIV/PTV stopped: run again to continue from the last chunk of pairs saved")
        fields = load_vectors(vectors, pixels=True)
        valid, seconds = valid_percent(fields), run_time(fields.settings)
        failed = len(fields.settings.get("failed_pairs") or [])
        history.append(entry("piv", f"Batch: PIV/PTV done: {fields.n} pairs, {valid:.1f} % valid vectors in "
                                    f"{format_duration(seconds)}" + (f", {failed} pair(s) failed" if failed else "")
                                    + f". Results: {relative(vectors, session.path_in)}"))
        # The POD evaluated or restored here, or else the one the project had
        pod = pod_file if session.Check_eigproc and os.path.isfile(pod_file) else None
        save_project(session, n_modes, engine, settings, fields, pod, history, say, regression)
        return BatchResult(session.path_in, "done", vectors, fields.n, valid, seconds, pre_seconds)


def apply_preprocessing(session, section):
    """The settings of the pre-processing of a project section (as the interface saves them: masks, contrast,
    background_removal, export_folder; masks_file: the masks of this file instead) applied to a session. The export folder: the one of the section if it is
    inside the folder of the images (relative, e.g. the default PreProcessing), else the default one. Returns the
    number of modes removed (0: no background removal)."""
    masks = masks_from_file(section["masks_file"]) if section.get("masks_file") else section.get("masks") or []
    session.masks[:] = [Mask(m["points"], m.get("name", "Mask"), m.get("enabled", True)) for m in masks]
    known = {f.name for f in dataclass_fields(ContrastSettings)}
    session.contrast = ContrastSettings(**{k: v for k, v in (section.get("contrast") or {}).items() if k in known})
    # In the interface, the contrast enhancement is off when no filter is selected
    session.contrast_enabled = bool(session.contrast.filters)
    folder = section.get("export_folder")
    inside = isinstance(folder, str) and folder and not os.path.isabs(folder) and not folder.startswith("..")
    session.path_export = os.path.normpath(os.path.join(session.path_in, folder if inside else PREPROCESSING_FOLDER))
    bgrem = section.get("background_removal") or {}
    n_modes = str2double(bgrem.get("modes_removed")) if bgrem.get("enabled") else 0
    return int(min(n_modes, session.NImg)) if n_modes >= 1 else 0


def preprocess_folder(folder, section, progress=None, temp_folder="images", verbose=True):
    """Pre-process the images of a folder with the settings of a project section (apply_preprocessing), as the
    button Export of the interface: the POD modes evaluated (or those saved in pod.npz), the images exported with
    their description, and the project of the folder updated (its section preprocessing, its history), so that the
    interface shows them. Skipped if the images exported are up to date or the recipe is empty. progress(title):
    the progress of the session (context manager called with the fraction done and a message; it may raise
    preprocessing.Stopped). Returns a BatchResult."""
    with PODBackgroundRemoval(folder, temp_folder=temp_folder, verbose=verbose) as session:
        if progress is not None:
            session.progress = progress
        try:
            previous = read_project(session.path_in) or {}
        except (OSError, ValueError):
            previous = {}
        same = previous.get("images", {}).get("files") == session.file_in
        if same:
            session.project = clean_name(previous.get("name")) or None
        n_modes = apply_preprocessing(session, section)
        steps = " → ".join(session.recipe_steps(n_modes, detailed=True))
        if session.recipe_empty(n_modes):
            return BatchResult(session.path_in, "skipped", message="nothing to export: the recipe is empty")
        if session.preprocessed_images(n_modes) is not None:
            return BatchResult(session.path_in, "skipped", message=f"images exported up to date ({steps})")
        t0 = time.time()
        history = [] if same else [entry("images", f"{session.NImg} images: {session.file_in[0]} ... "
                                                   f"{session.file_in[-1]}")]
        pod_file = os.path.join(session.export_folder(), POD_FILE)
        if n_modes and (session.load_pod(pod_file) or 0) < n_modes:
            if session.evaluate_modes(n_modes) is None:
                raise RuntimeError("The POD modes could not be evaluated")
            session.save_pod(pod_file)
            history.append(entry("modes", f"Batch: spectrum and {session.NMod_saved} POD modes evaluated"))
        written = session.export_images(session.export_folder(), n_modes)
        seconds = time.time() - t0
        export_rel = relative(session.export_folder(), session.path_in)
        history.append(entry("export", f"Batch: {len(written)} images exported to {export_rel} ({steps})"))
        # The project: its section preprocessing as the interface saves it, the other sections kept (the same
        # images)
        data = {k: v for k, v in previous.items() if k not in ("format", "version", "saved")} if same else {}
        data.setdefault("images", {"folder": ".", "files": list(session.file_in)})
        own = data.get("preprocessing") or {}
        contrast = {**vars(session.contrast), "filters": list(session.contrast.filters)}
        pod = ({"file": relative(pod_file, session.path_in), "modes_evaluated": session.NMod_saved}
               if session.Check_eigproc and os.path.isfile(pod_file) else own.get("pod"))
        data["preprocessing"] = {
            **own, "masks": [{"name": m.name, "enabled": m.enabled, "points": m.points.tolist()}
                             for m in session.masks],
            "contrast": contrast,
            "background_removal": {**(own.get("background_removal") or {}), **(section.get("background_removal") or {})},
            "export_folder": export_rel,
            "export": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": steps, "n_images": len(written),
                       "recipe": comparable_recipe(session.recipe_record(n_modes))},
            "pod": pod}
        data["history"] = (data.get("history") or []) + history
        write_project(os.path.join(session.path_in, PROJECT_FILE), data)
        return BatchResult(session.path_in, "done", preprocessing_seconds=seconds,
                           message=f"{len(written)} images exported to {export_rel} ({steps}) in "
                                   f"{format_duration(seconds)}")


def entry(step, text):
    """A step of the history of the project."""
    return {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step, "text": text}


def process_folders(roots, pattern=None, report=None, verbose=True, root_masks=False, **options):
    """Process the measurement folders of the root folders (measurement_folders: a root with images, or its
    subfolders with images), one after the other: yields the BatchResult of each. A folder that fails doesn't
    stop the others (status "failed"); Ctrl+C stops the PIV running (its chunks of pairs are kept) and the batch.
    root_masks: the masks of each folder are those of its own file of masks, else of the file of masks at the root
    (measurement_masks, find_masks; none: no masks), instead of options["masks"]. options: those of
    process_folder."""
    say = print if verbose else (lambda *args: None)
    masks = options.pop("masks", None)
    folders = {}  # Each folder: its masks
    for root in roots:
        try:
            found = measurement_folders(root, pattern)
            if root_masks:
                masks = find_masks(root)
                say(f"{os.path.abspath(root)}: " + (f"masks of {masks}" if masks else "no file of masks at the root"))
        except ValueError as e:
            yield BatchResult(os.path.abspath(root), "failed", message=str(e))
            continue
        if not found:
            yield BatchResult(os.path.abspath(root), "failed", message="no images in this folder nor in its "
                                                                        "subfolders")
        for folder in found:
            folders.setdefault(folder, masks)
    for n, (folder, masks) in enumerate(folders.items()):
        say(f"[{n + 1}/{len(folders)}] {folder}")
        try:
            if root_masks:
                own = find_masks(folder)
                if own and own != masks:
                    say(f"  masks of {own}")
                masks = own or masks
            result = process_folder(folder, pattern, masks=masks, report=report, verbose=verbose, **options)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # The other folders are processed
            result = BatchResult(folder, "failed", message=f"{type(e).__name__}: {e}")
        say(f"  {result.describe()}")
        yield result


################################################################################
# Post-processing


@dataclass
class PostResult:
    """Result of the post-processing of a measurement folder: status "done" or "failed" (message: why); its
    post-processing (PostProcessing: its analyses, for further use), the files exported and the time (s)."""

    folder: str
    status: str
    post: object = None
    files: list = None
    seconds: float = None
    message: str = ""

    def describe(self):
        if self.status == "done":
            return f"done: {len(self.files)} files exported in {format_duration(self.seconds)}"
        return f"{self.status}: {self.message}"


def postprocess_folder(folder, camera, settings=None, export_folder=None, index=0, dpi=EXPORT_DPI, name=None, figures=None,
                       maps=None, videos=(), fps=None, video_format=None, maps_folder=None, pixels=False, **export):
    """Post-processing of the PIV results of a measurement folder (vectors.npz of its subfolder PIV) with the camera
    settings (piv.Camera: width, horizontal and vertical offsets of the field of view, frame rate) and the recipe (PostSettings; the
    defaults of the interface if None): the figures of the recipe exported (the instantaneous ones of the map
    index), with the key numbers (metrics.json) and the recipe (postprocessing.json), to its subfolder
    PostProcessing (PostProcessing_PTV for the results of a PTV; or export_folder), named after the project (name; the name of the folder by default), as the
    interface exports them; figures, maps, videos, fps, video_format: the figures exported, the maps of the
    instantaneous ones and their videos ("gif" by default, or "mp4"; PostProcessing.export; index and maps: negative
    from the end, -1 the last map). maps_folder: the folder of vectors.npz (the subfolder PIV by default);
    pixels: in the sensor space (no camera); export: the other options of PostProcessing.export (clims, cmaps,
    contours, compares, video_maps, video_options). Returns a PostResult."""
    t0 = time.time()
    vectors = os.path.join(maps_folder or os.path.join(folder, piv.PIV_FOLDER), piv.VECTORS_FILE)
    if not os.path.isfile(vectors):
        return PostResult(folder, "failed", message=f"no PIV/PTV results ({vectors})")
    post = PostProcessing(load_vectors(vectors, camera=None if pixels else camera, pixels=pixels),
                          settings or PostSettings())
    export_folder = export_folder or os.path.join(folder, default_export_folder(os.path.dirname(vectors)))
    files = post.export(export_folder, index, dpi,
                        name or os.path.basename(os.path.normpath(folder)), figures=figures,
                        maps=maps, videos=videos, **({} if fps is None else {"fps": fps}),
                        **({} if video_format is None else {"video_format": video_format}), **export)
    return PostResult(folder, "done", post, files, time.time() - t0)


def postprocess_folders(folders, camera, settings=None, verbose=True, **options):
    """Post-processing of measurement folders, one after the other (postprocess_folder): camera is the Camera of
    all of them, or a function camera(folder) giving the one of each (that of its field of view); settings the
    PostSettings of all of them, or a function settings(folder) (called when the folder is processed, after the
    PostResult of the previous ones: they may set its reference quantities). Yields the
    PostResult of each (its PostProcessing is only kept there: drop it once used). A folder that fails doesn't stop
    the others (status "failed"). options: those of postprocess_folder (export_folder: a function of the folder
    too)."""
    say = print if verbose else (lambda *args: None)
    folders = list(folders)
    for n, folder in enumerate(folders):
        say(f"[{n + 1}/{len(folders)}] {folder}")
        kwargs = {key: value(folder) if key == "export_folder" and callable(value) else value
                  for key, value in options.items()}
        try:
            result = postprocess_folder(folder, camera(folder) if callable(camera) else camera,
                                        settings(folder) if callable(settings) else settings, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # The other folders are post-processed
            result = PostResult(folder, "failed", message=f"{type(e).__name__}: {e}")
        say(f"  {result.describe()}")
        yield result
