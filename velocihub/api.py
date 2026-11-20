"""High-level API of VelociHub: a measurement folder as the interface sees it (Project), the projects of a workspace or
of root folders (projects), the batch run of a step on many of them (run), and the stitching of fields of view
(stitch). It does what the pages and the window Batch do, with the same settings, through the same functions
(settings.py, batch.py, batch_run.py): the results and the project file (VelociHub_project.json) are those the
interface writes, so a project prepared or run here opens in the interface as it was left.

    import velocihub as vh

    p = vh.Project("D:/Jet/FOV_1/Run_30")
    p.set_preprocessing(masks="D:/Jet/masks.json", filters=["capping"], cap_n=10, modes_removed=6)
    p.set_piv(engine="PaIRS-UniNa", windows=(128, 64, 32), overlap=50)
    p.set_postprocessing(camera=vh.Camera(196.1, frame_rate_hz=150, y_shift=0.5), outlier_k=4, use_u_ref=True, l_ref=50,
                         figures=["mean_velocity", "u_rms"], videos=["velocity"], fps=24)
    p.save()                                       # the settings in the project file (as the window Batch)
    p.run()                                        # pre-processing, PIV, post-processing

    for project in vh.projects("D:/Jet"):          # every project found under the root folder
        project.set_piv(overlap=75)
        project.save()
    vh.run("piv", vh.projects("D:/Jet"))           # the PIV of all of them, each with its own settings

The settings of each step are the "batch settings" of settings.py, kept in Project.preprocessing, Project.piv and
Project.postprocessing (dictionaries: they can also be edited directly before save or run).
"""

import copy
import os
from dataclasses import asdict

from . import batch, batch_run, piv
from .fields import load_vectors
from .pipelines import DEFAULT_PIPELINE, chosen_results, results_folders, selected_results
from .postprocessing import PostProcessing, PostSettings
from .preprocessing import (DEFAULT_CONTRAST, DEFAULT_EXPORT_FORMAT, DEFAULT_MODES, PREPROCESSING_FOLDER, ContrastSettings,
                            DEFAULT_BACKGROUND_OFFSET, PODBackgroundRemoval, background_offset_of, export_format_of,
                            same_path)
from .project import PROJECT_FILE, read_project, write_project
from .settings import (camera_of, camera_record, export_spec, piv_from_project, piv_to_project, post_from_project, post_to_project,
                       preprocessing_from_project, preprocessing_to_project, settings_from_record, settings_record,
                       validate_piv, validate_post, validate_preprocessing)
from .workspace import Workspace, image_names, scan

STEPS = ("preprocessing", "piv", "postprocessing")


def default_preprocessing():
    """The pre-processing of a project without project file: that of the interface when it starts (intensity capping
    n = 10 after the background removal of 6 POD modes, no masks)."""
    contrast = asdict(DEFAULT_CONTRAST)
    contrast["filters"] = list(DEFAULT_CONTRAST.filters)
    return {"masks": [], "contrast": contrast,
            "background_removal": {"enabled": True, "modes_removed": str(DEFAULT_MODES),
                                   "offset": DEFAULT_BACKGROUND_OFFSET},
            "export_folder": PREPROCESSING_FOLDER, "export_format": DEFAULT_EXPORT_FORMAT}


def mask_record(mask, i=0):
    """A mask as the project file keeps it ({"name", "enabled", "points"}) from a preprocessing.Mask, a dictionary of
    the project file or a list of points [[x, y], ...]."""
    if isinstance(mask, dict):
        name, enabled, points = mask.get("name"), mask.get("enabled", True), mask["points"]
    elif isinstance(mask, (list, tuple)):
        name, enabled, points = None, True, mask
    else:
        name, enabled, points = mask.name, mask.enabled, mask.points
    return {"name": name or f"Mask {i + 1}", "enabled": bool(enabled),
            "points": [[float(x), float(y)] for x, y in points]}


class Project:
    """A measurement folder (images and/or PIV/PTV results) and its project file: the settings of the three steps
    (preprocessing, piv, postprocessing: batch settings, see settings.py), their changes (set_*), their saving in the
    project file (save) and their runs (preprocess, run_piv, postprocess, run), as the interface does them.

    folder: the measurement folder (the one of the images; or of the results for a project without images)."""

    def __init__(self, folder):
        self.folder = os.path.abspath(folder)
        if not os.path.isdir(self.folder):
            raise ValueError(f"The folder doesn't exist: {self.folder}")
        self.reload()

    def __repr__(self):
        return f"Project({self.folder!r})"

    # ------------------------------------------------------------------ what it is

    @property
    def name(self):
        """The name of the project (the start of the names of the files exported): the one of its project file,
        else the name of its folder."""
        return batch_run.project_name(self.folder)

    @property
    def images(self):
        """The paths of its images (not the figures exported)."""
        return [os.path.join(self.folder, n) for n in image_names(self.folder)]

    @property
    def project_file(self):
        return os.path.join(self.folder, PROJECT_FILE)

    def results_folders(self):
        """The folders of its results by pipeline ({"PaIRS": folder, "TracTrac": folder}: those it has)."""
        return results_folders(self.folder)

    # ------------------------------------------------------------------ settings

    def reload(self):
        """The settings of its project file (those of the interface when it starts for the steps it has none)."""
        data = read_project(self.folder) or {}
        self.data = data
        self.preprocessing = (preprocessing_from_project(data["preprocessing"]) if data.get("preprocessing")
                              else default_preprocessing())
        self.piv = validate_piv(piv_from_project(data.get("piv") or {}))
        self.postprocessing = validate_post(post_from_project(data.get("postprocessing") or {}))
        return self

    def set_preprocessing(self, masks=None, masks_file=None, contrast=None, filters=None, stage=None,
                          background_removal=None, modes_removed=None, background_offset=None, export_folder=None,
                          export_format=None, **parameters):
        """Change the pre-processing (the side panel of the page Pre-processing; None: unchanged).

        masks: a list of polygons [[x, y], ...] or of preprocessing.Mask, or the path of a file of masks (its masks
        copied); masks_file: a file of masks used instead at each run (as the window Batch: "" for none);
        contrast: a ContrastSettings (or its dictionary); filters, stage and the parameters of the filters (cap_n,
        mm_window, mm_contrast, clip, tiles, edge_sigma, edge_factor, gm_sigma, gm_contrast): those changed in it; background_removal:
        on/off; modes_removed: the POD modes removed (turns the background removal on, 0 off); background_offset:
        the offset added after the removal, "auto" or grey levels (preprocessing.DEFAULT_BACKGROUND_OFFSET); export_folder: the
        folder of the images exported, relative to the project (PreProcessing by default); export_format: their
        format, "same" (as the source images, the default), "png", "bmp" or "tiff"."""
        s = self.preprocessing
        if masks is not None:
            if isinstance(masks, (str, os.PathLike)):
                masks = batch.masks_from_file(os.fspath(masks))
            s["masks"] = [mask_record(m, i) for i, m in enumerate(masks)]
        if masks_file is not None:
            s["masks_file"] = os.fspath(masks_file) or None
        current = dict(s.get("contrast") or {})
        if contrast is not None:
            current = asdict(contrast) if isinstance(contrast, ContrastSettings) else dict(contrast)
        if filters is not None:
            current["filters"] = [filters] if isinstance(filters, str) else list(filters)
        if stage is not None:
            current["stage"] = stage
        current.update(parameters)
        current["filters"] = list(current.get("filters") or [])
        ContrastSettings(**{**current, "filters": tuple(current["filters"])})  # Checked
        s["contrast"] = current
        bgrem = dict(s.get("background_removal") or {})
        if modes_removed is not None:
            bgrem.update(enabled=int(modes_removed) > 0, modes_removed=str(int(modes_removed) or DEFAULT_MODES))
        if background_removal is not None:
            bgrem["enabled"] = bool(background_removal)
        if background_offset is not None:
            bgrem["offset"] = background_offset_of(background_offset)
        s["background_removal"] = bgrem
        if export_folder is not None:
            s["export_folder"] = export_folder
        if export_format is not None:
            s["export_format"] = export_format_of(export_format)
        self.preprocessing = validate_preprocessing(s)
        return self

    def set_piv(self, engine=None, settings=None, regression=None, output_folder=None, **engine_settings):
        """Change the PIV/PTV (the side panel of the page Velocimetry; None: unchanged).

        engine: "PaIRS-UniNa" or "TracTrac" (name or key); settings: the settings of the engine (a dictionary,
        merged with those it has), or as keywords (windows=(128, 64, 32), overlap=50, particle_size=3...; see
        docs/processing.md); regression: None (unchanged), "none", "spicy", or {"method": "spicy", "settings": {...}};
        output_folder: the folder of the results, relative to the project (PIV, or PTV for TracTrac, by default). As in
        the page, another engine takes its own default folder unless another folder than the default of the previous
        one was chosen."""
        s = dict(self.piv)
        previous = s.get("engine")
        if engine is not None:
            s["engine"] = piv.get_engine(engine).key
            folder = s.get("output_folder")
            if (output_folder is None and previous and previous != s["engine"]
                    and (not folder or same_path(folder, piv.get_engine(previous).results_folder))):
                s["output_folder"] = piv.get_engine(s["engine"]).results_folder
        s = validate_piv(s)  # The settings of the engine chosen
        engine = piv.get_engine(s["engine"], **(s.get("options") or {}))
        merged = engine.check({**(s.get("settings") or {}), **(settings or {}), **engine_settings})
        # As the project file and the form of the page keep them (texts, all of them): the same settings compare equal
        s["settings"] = batch.engine_settings_text(engine, merged)
        if regression is not None:
            s["regression"] = {"method": regression, "settings": {}} if isinstance(regression, str) else regression
        if output_folder is not None:
            s["output_folder"] = output_folder
        self.piv = validate_piv(s)
        return self

    def set_postprocessing(self, results=None, space=None, camera=None, settings=None, export_folder=None,
                           figures=None, videos=None, fps=None, video_format=None, video_maps=None, maps=None,
                           clims=None, cmaps=None, contours=None, compares=None, **recipe):
        """Change the post-processing (the side panel of the page Post-processing and its window Quantities; None:
        unchanged).

        results: the results sets exported (pipelines.py: "PaIRS", "TracTrac", or a list of them; the same export for
        each); space: "physical" or "sensor"; camera: a piv.Camera
        (or a dictionary of its fields: fov_width_mm, fov_height_mm, x_offset_mm, y_offset_mm, x_shift, y_shift,
        frame_rate_hz; space physical); settings: a
        PostSettings, or the fields of the recipe as keywords (outlier_k=4, use_u_ref=True, l_ref=50...; see
        docs/postprocessing.md); export_folder: relative to the project (PostProcessing by default).
        The export (window Quantities): figures (keys; None in the call: unchanged, [] none... use "all" for all);
        videos (keys of instantaneous figures), fps, video_format ("gif" or "mp4"), video_maps (first, last, step:
        1-based, last None for the last map); maps: the maps of the instantaneous figures (a map number, 1-based, or
        (first, last, step)); clims {key: (vmin, vmax)}, cmaps {key: name}, contours {key: level}, compares {key:
        other key or (other key, fraction of the wiper)}."""
        s = copy.deepcopy(self.postprocessing)
        if results is not None:
            s["results"] = results
        if space is not None:
            s["space"] = space
        if camera is not None:
            s["camera"] = camera_record(camera_of(asdict(camera) if isinstance(camera, piv.Camera)
                                                  else dict(camera)))
            s["space"] = space or "physical"
        record = dict(s.get("settings") or settings_record(PostSettings()))
        if settings is not None:
            record = settings_record(settings) if isinstance(settings, PostSettings) else dict(settings)
        record.update(recipe)
        s["settings"] = settings_record(settings_from_record(record))  # Checked
        if export_folder is not None:
            s["export_folder"] = export_folder
        spec = export_spec(s.get("export"))
        choice, params = spec["choice"], spec["params"]
        if figures is not None:
            choice["figures"] = None if figures == "all" else list(figures)
        if videos is not None:
            choice["videos"] = list(videos)
        if fps is not None:
            choice["fps"] = float(fps)
        if video_format is not None:
            choice["format"] = video_format
        if video_maps is not None:
            choice["first"], choice["last"], choice["step"] = (list(video_maps) + [None, 1])[:3]
        if maps is not None:
            first, last, step = ((maps, maps, 1) if isinstance(maps, int) else tuple((list(maps) + [None, 1])[:3]))
            choice.update(map=first, map_last=last, map_step=step or 1)
        for name, values in (("clims", clims), ("cmaps", cmaps), ("contours", contours)):
            if values is not None:
                params[name] = {k: list(v) if name == "clims" else v for k, v in values.items()}
        if compares is not None:
            for key, other in compares.items():
                other, fraction = (other, None) if isinstance(other, str) else other
                params["compares"][key] = other
                if fraction is not None:
                    params["wipers"][key] = float(fraction)
        s["export"] = export_spec(spec)
        self.postprocessing = validate_post(s)
        return self

    def save(self):
        """Write the settings of the three steps in the project file (created if there is none), as the window Batch
        saves them: the rest of the file (results, history, tiles) kept. Returns its path."""
        data = read_project(self.folder) or {}
        data = {k: v for k, v in data.items() if k not in ("format", "version", "saved")}
        if not data.get("images") and self.images:
            data["images"] = {"folder": ".", "files": image_names(self.folder)}
        data.setdefault("name", None)
        for section, merge, settings in (("preprocessing", preprocessing_to_project, self.preprocessing),
                                         ("piv", piv_to_project, self.piv),
                                         ("postprocessing", post_to_project, self.postprocessing)):
            data[section] = merge(dict(data.get(section) or {}), copy.deepcopy(settings))
        path = write_project(self.project_file, data)
        self.data = read_project(self.folder)
        return path

    def set_name(self, name):
        """The name of the project (File > Project name...: the start of the names of the files exported; None: the
        name of its folder), saved in the project file."""
        self.save()
        data = {k: v for k, v in (read_project(self.folder) or {}).items() if k not in ("format", "version", "saved")}
        data["name"] = name or None
        write_project(self.project_file, data)
        return self.reload()

    # ------------------------------------------------------------------ runs (as the window Batch)

    def _job(self, **options):
        settings = {"preprocessing": copy.deepcopy(self.preprocessing), "piv": copy.deepcopy(self.piv),
                    "postprocessing": copy.deepcopy(self.postprocessing)}
        folders = selected_results(self.results_folders(), settings["postprocessing"].get("results"))
        settings["postprocessing"].update(maps_folders=folders, maps_folder=next(iter(folders.values()), None))
        return {"folder": self.folder, "settings": settings}, {"options": options}

    def preprocess(self):
        """Run step > Singular of the page Pre-processing: the images exported with the recipe (the POD modes evaluated
        if needed; skipped if up to date), the project file updated. Returns a batch.BatchResult."""
        project, _ = self._job()
        result = batch.preprocess_folder(self.folder, project["settings"]["preprocessing"])
        if result.status == "done":  # The project file written by the run (skipped: unchanged, the settings kept)
            self.reload()
        return result

    def run_piv(self, force=False, report=None):
        """Run step > Singular of the page Velocimetry: the images pre-processed first if needed, then the PIV/PTV (and the
        regression), continued from the pairs saved; skipped if the results are up to date (force: computed again).
        report(message): its messages (printed by default). Returns a batch.BatchResult."""
        p = self.piv
        engine = piv.get_engine(p.get("engine"), **(p.get("options") or {}))
        result = batch.process_folder(self.folder, engine=engine, settings=p.get("settings") or {},
                                      preprocessing=self.preprocessing, output_folder=p.get("output_folder"),
                                      force=force, report=report, regression=p.get("regression"))
        self.reload()
        return result

    def postprocess(self, overwrite=False):
        """Run step > Singular of the page Post-processing: the quantities chosen exported (figures, videos,
        metrics.json, postprocessing.json) to its export folder. overwrite: the files of a previous export removed
        first. Returns (status, message)."""
        project, job = self._job(overwrite=overwrite)
        if not project["settings"]["postprocessing"]["maps_folder"]:
            return "failed", f"no PIV/PTV results in {self.folder}"
        return batch_run.run_postprocessing(project, job, progress=None)

    def run(self, steps=STEPS, force=False):
        """The steps given, in order (pre-processing, PIV/PTV, post-processing). Returns {step: result}."""
        out = {}
        for step in steps:
            out[step] = {"preprocessing": self.preprocess, "piv": lambda: self.run_piv(force=force),
                         "postprocessing": self.postprocess}[step]()
        return out

    # ------------------------------------------------------------------ objects of each step

    def session(self):
        """A PODBackgroundRemoval of its images with its pre-processing (masks, contrast enhancement; the modes saved
        restored): use it in a "with" statement. The modes removed: n_modes_removed(session)."""
        session = PODBackgroundRemoval(self.folder, temp_folder="images", verbose=False)
        n_modes = batch.apply_preprocessing(session, self.preprocessing)
        try:
            session.load_pod()
        except (OSError, ValueError, KeyError):
            pass
        session.modes_removed = n_modes
        return session

    def vector_fields(self, results=None, pixels=None):
        """Its vector fields (results: a pipeline, "PaIRS", "TracTrac"...; by default the first results set of the
        post-processing it has), in the space of the post-processing (pixels: True for the sensor space, False with the
        camera)."""
        s = self.postprocessing
        chosen = results or s.get("results")
        folder = chosen_results(self.results_folders(), chosen)
        if folder is None:
            raise ValueError(f"No {chosen or DEFAULT_PIPELINE} results in {self.folder}")
        camera = s.get("camera")
        if pixels is None:
            pixels = s.get("space") == "sensor" or not camera
        camera = None if pixels else piv.Camera(**camera)
        return load_vectors(os.path.join(folder, piv.VECTORS_FILE), camera=camera, pixels=pixels)

    def post_processing(self, results=None):
        """A PostProcessing of its vector fields with its recipe (plot, metrics, export...)."""
        return PostProcessing(self.vector_fields(results), settings_from_record(self.postprocessing["settings"]))


def projects(source):
    """The projects of a source, as Project objects: a workspace file (VelociHub_workspace.json) or Workspace, a root
    folder (the projects found under it, at any depth, as the panel Projects finds them), or a list of them."""
    if isinstance(source, Workspace):
        return [Project(p.folder) for p in source.projects if not p.lost]
    if isinstance(source, (list, tuple)):
        return [p for item in source for p in projects(item)]
    source = os.fspath(source)
    if os.path.isfile(source):
        return projects(Workspace.load(source))
    return [Project(folder) for folder, _, _ in scan(source).values()]


def run(step, items, force=False, overwrite=False, verbose=True):
    """A step run on projects (Project objects, or what projects() takes), each with its own settings, as the window
    Batch runs them: "preprocessing", "piv", "postprocessing". A project that fails doesn't stop the others; Ctrl+C
    stops the run. Returns [(project, result)]."""
    if step not in STEPS:
        raise ValueError(f"Unknown step {step!r}: {', '.join(STEPS)}")
    items = items if isinstance(items, (list, tuple)) and all(isinstance(p, Project) for p in items) \
        else projects(items)
    out = []
    for i, project in enumerate(items, 1):
        if verbose:
            print(f"[{i}/{len(items)}] {step}: {project.folder}")
        try:
            result = (project.preprocess() if step == "preprocessing" else project.run_piv(force=force)
                      if step == "piv" else project.postprocess(overwrite=overwrite))
        except KeyboardInterrupt:
            print("Stopped")
            break
        except Exception as e:  # The others go on, as in the window Batch
            result = ("failed", str(e))
        out.append((project, result))
        if verbose:
            print("   ", result.describe() if hasattr(result, "describe") else " ".join(map(str, result)))
    return out


def stitch(chain, setup=None, overwrite=False, **post_archive_options):
    """The stitching of fields of view (the stitchings of the window Batch of the post-processing): chain, the
    projects of one setup in its fields of view, the reference first (Project objects or folders), each with its
    post-processing settings (physical space: a camera with its offsets). Each is post-processed with the quantities
    of the reference, then the fields of view are stitched in the folder Stitched of the reference's export folder.
    setup: the name of the setup (the folder name of the reference by default); post_archive_options: settings of
    archiving/post_archive.py (MATCH_VELOCITY=True, MATCH_JET=0.5...). The figures, maps and videos stitched are those
    of the export of the reference (its set_postprocessing: figures, maps, videos). Returns (status, message)."""
    chain = [p if isinstance(p, Project) else Project(p) for p in chain]
    if len(chain) < 2:
        raise ValueError("A stitching needs at least two projects (fields of view)")
    job_projects, members = [], []
    for i, p in enumerate(chain):
        project, _ = p._job()
        job_projects.append(project)
        members.append({"folder": p.folder, "root": os.path.dirname(p.folder), "fov": f"FOV_{i + 1}"})
    job = {"projects": job_projects, "options": {"overwrite": overwrite, "globals": post_archive_options}}
    stitching = {"setup": setup or os.path.basename(chain[0].folder), "members": members}
    return batch_run.run_stitching(job, stitching)
