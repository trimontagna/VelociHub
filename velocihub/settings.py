"""Settings of each step as the window Batch and the API handle them ("batch settings"), and their conversion to and
from the sections of a project file (VelociHub_project.json):

- pre-processing: {"masks", "masks_file", "contrast", "background_removal": {"enabled", "modes_removed"},
  "export_folder"} (preprocessing_from_project, preprocessing_to_project, validate_preprocessing);
- PIV/PTV: {"engine", "options", "settings", "engine_settings", "regression", "output_folder"} (piv_from_project,
  piv_to_project, validate_piv);
- post-processing: {"results": "PIV" or "PTV", "settings": PostSettings as a dictionary, "space": "physical" or
  "sensor", "camera", "export_folder", "export": export specification} (post_from_project, post_to_project,
  validate_post), the export specification (export_spec: the figures, maps and videos chosen in the window Quantities,
  and the parameters of the figures) and the options of PostProcessing.export it gives (export_options).

No interface here: the pages, the window Batch, the batch runs (batch_run.py) and the API (api.py) share these
functions.
"""

import copy
import os
from dataclasses import asdict, fields as dataclass_fields

from . import piv
from .batch import masks_from_file
from .fields import VIDEO_FORMAT, VIDEO_FORMATS, VIDEO_FPS, num2str
from .postprocessing import FIGURE_TITLES, MAP_FIGURES, PostSettings
from .preprocessing import ContrastSettings, str2double
from .prefs import getpref


# Colour maps proposed for the maps of the tiles ("Default": the one chosen for the figure)
COLORMAPS = ("viridis", "magma", "inferno", "plasma", "cividis", "RdBu_r", "coolwarm", "seismic", "turbo", "jet",
             "gray")


def preprocessing_to_project(section, settings):
    """The section preprocessing of a project file with the batch settings of the pre-processing (the rest kept:
    the spectrum and the modes saved, the last export, the automatic threshold)."""
    out = {**section, **{k: settings[k] for k in ("masks", "contrast", "export_folder", "masks_file")
                         if k in settings}}
    # The masks of the file: those of the project (the interface shows them)
    if settings.get("masks_file"):
        try:
            out["masks"] = masks_from_file(settings["masks_file"])
        except ValueError:
            pass
    out["background_removal"] = {**(section.get("background_removal") or {}),
                                 **(settings.get("background_removal") or {})}
    return out


def preprocessing_from_project(section):
    """The batch settings of the pre-processing of the section preprocessing of a project file."""
    return {k: copy.deepcopy(section[k]) for k in ("masks", "contrast", "background_removal", "export_folder",
                                                    "masks_file") if k in section}


def validate_preprocessing(section):
    """Batch settings of the pre-processing checked: ValueError with a message (the file of masks: one that can be
    read; none: masks_file None)."""
    section = {**section, "masks_file": section.get("masks_file") or None}
    if section["masks_file"]:
        masks_from_file(section["masks_file"])
    known = {f.name for f in dataclass_fields(ContrastSettings)}
    contrast = {k: v for k, v in (section.get("contrast") or {}).items() if k in known}
    ContrastSettings(**{**contrast, "filters": tuple(contrast.get("filters") or ())})
    bgrem = section.get("background_removal") or {}
    if bgrem.get("enabled") and not str2double(bgrem.get("modes_removed")) >= 1:
        raise ValueError("POD modes removed: at least 1 with the background removal")
    return section


def piv_to_project(section, settings):
    """The section piv of a project file with the batch settings of the PIV: the engine and its settings (those
    of the other engines and the particles kept), the results folder (the results kept)."""
    engine = piv.get_engine(settings.get("engine"), **(settings.get("options") or {}))
    saved = dict(section.get("settings") or {})
    engines = {**(saved.get("engine_settings") or {}), **(settings.get("engine_settings") or {}),
               engine.key: dict(settings.get("settings") or {})}
    out = {**section, "settings": {**saved, "engine": engine.name, "engine_settings": engines}}
    if "regression" in settings:
        out["settings"]["regression"] = settings["regression"]
    if "output_folder" in settings:
        out["output_folder"] = settings["output_folder"]
    return out


def piv_from_project(section):
    """The batch settings of the PIV of the section piv of a project file: its engine and its settings, its results
    folder (ValueError without engine)."""
    saved = section.get("settings") or {}
    engine = piv.get_engine(saved.get("engine"))
    engines = {k: dict(v) for k, v in (saved.get("engine_settings") or {}).items() if isinstance(v, dict)}
    return {"engine": engine.key, "options": getpref("piv_engine_options", {}).get(engine.key, {}),
            "settings": engines.pop(engine.key, {}), "settings_engine": engine.key, "engine_settings": engines,
            "regression": saved.get("regression") or {"method": "none", "settings": {}},
            "output_folder": section.get("output_folder")}


def validate_piv(settings):
    """Batch settings of the PIV checked by its engine: ValueError with a message. The software changed (engine:
    not that of the settings, settings_engine): the settings of the previous one kept (engine_settings), those of
    the new one: those it had, else its defaults, with its options."""
    key = piv.get_engine(settings.get("engine")).key
    owner = settings.get("settings_engine") or key
    if owner != key:
        kept = dict(settings.get("engine_settings") or {})
        kept[owner] = dict(settings.get("settings") or {})
        settings = {**settings, "engine": key, "settings": kept.pop(key, {}), "engine_settings": kept,
                    "options": getpref("piv_engine_options", {}).get(key, {})}
    settings = {**settings, "engine": key, "settings_engine": key}
    engine = piv.get_engine(key, **(settings.get("options") or {}))
    engine.check({k: v for k, v in (settings.get("settings") or {}).items()})
    piv.check_regression(settings.get("regression"))
    # The settings of SPICY completed with their defaults (as the page shows them), also without regression: the
    # projects compare whatever their files keep
    regression = dict(settings.get("regression") or {})
    given = dict(regression.get("settings") or {})
    regression["settings"] = {p.key: given[p.key] if p.key in given else p.format(p.default)
                              for p in piv.SPICY_PARAMETERS}
    regression.setdefault("method", "none")
    return {**settings, "regression": regression}


# Position of the wiper when a second quantity is chosen (fraction of the width of the map)
COMPARE_FRACTION = 0.5


# The export choice by default: all the figures, no video. first, last, step, fps, format: the maps and settings of
# the videos; map, map_last, map_step: the maps of the instantaneous figures (1 to N; None: the last map); range: the
# maps of the videos for the instantaneous figures too (ExportDialog of the archive script)
DEFAULT_EXPORT = {"figures": None, "videos": [], "range": False, "first": 1, "last": None, "step": 1,
                  "fps": VIDEO_FPS, "format": VIDEO_FORMAT, "map": None, "map_last": None, "map_step": 1}


def export_choice(saved):
    """The export choice of saved settings (preferences, project), completed with the defaults (the figures that no
    longer exist left out)."""
    choice = dict(DEFAULT_EXPORT)
    if isinstance(saved, dict):
        choice.update({key: value for key, value in saved.items() if key in DEFAULT_EXPORT})
    if choice["figures"] is not None:
        choice["figures"] = [key for key in choice["figures"] if key in FIGURE_TITLES]
    # The videos of the figures of each map (any figure with rolling statistics)
    choice["videos"] = [key for key in choice["videos"] or [] if key in FIGURE_TITLES]
    if choice["format"] not in VIDEO_FORMATS:
        choice["format"] = VIDEO_FORMAT
    return choice


def video_settings(saved):
    """The settings of a video (VideoDialog) saved: every N maps, frames per second, format, first and last maps (1
    to N; last None: up to the last one), the defaults for those missing or invalid."""
    saved = saved if isinstance(saved, dict) else {}
    out = {"step": 1, "fps": VIDEO_FPS, "format": VIDEO_FORMAT, "first": 1, "last": None}
    try:
        out["step"], out["fps"] = read_step_fps(str(saved.get("step", 1)), str(saved.get("fps", VIDEO_FPS)))
    except ValueError:
        pass
    if saved.get("format") in VIDEO_FORMATS:
        out["format"] = saved["format"]
    for key in ("first", "last"):
        if isinstance(saved.get(key), int) and not isinstance(saved[key], bool) and saved[key] >= 1:
            out[key] = saved[key]
    return out


def figure_params_of(saved):
    """The parameters of the figures by quantity saved (project, batch settings), the invalid ones left out:
    {"clims": {key: (vmin, vmax)}, "cmaps": {key: name}, "contours": {key: value}, "compares": {key: second map},
    "wipers": {key: fraction}}."""
    saved = saved if isinstance(saved, dict) else {}

    def of(name, check, convert=lambda v: v):
        values = saved.get(name) if isinstance(saved.get(name), dict) else {}
        return {k: convert(v) for k, v in values.items() if k in FIGURE_TITLES and check(v)}
    return {"clims": of("clims", lambda v: isinstance(v, (list, tuple)) and len(v) == 2, tuple),
            "cmaps": of("cmaps", lambda v: v in COLORMAPS),
            "contours": of("contours", lambda v: isinstance(v, (int, float)), float),
            "compares": of("compares", lambda v: v in MAP_FIGURES),
            "wipers": of("wipers", lambda v: isinstance(v, (int, float)) and 0 <= v <= 1, float)}


def figure_params_from_tiles(data):
    """The parameters of the figures of a project of the versions that kept them by tile: those of the first tile
    showing each quantity."""
    out = {"clims": {}, "cmaps": {}, "contours": {}, "compares": {}, "wipers": {}}
    views = data.get("views") or []
    for name, saved in (("clims", "colour_ranges"), ("cmaps", "colour_maps"), ("contours", "contours")):
        for values in data.get(saved) or []:
            for key, value in (values or {}).items():
                out[name].setdefault(key, value)
    for view, view2 in zip(views, data.get("compare_views") or []):
        if view2 is not None:
            out["compares"].setdefault(view, view2)
    return figure_params_of(out)


def image_settings(saved):
    """The maps of the instantaneous figures of a quantity saved (its own: first, last, step; 1 to N), completed."""
    saved = saved if isinstance(saved, dict) else {}
    first = saved.get("first") if isinstance(saved.get("first"), int) and saved["first"] >= 1 else 1
    last = saved.get("last") if isinstance(saved.get("last"), int) and saved["last"] >= first else None
    step = saved.get("step") if isinstance(saved.get("step"), int) and saved["step"] >= 1 else 1
    return {"first": first, "last": last, "step": step}


def image_settings_of(saved):
    """The maps of the instantaneous figures of the quantities with their own ({figure: settings})."""
    return {key: image_settings(value) for key, value in (saved if isinstance(saved, dict) else {}).items()
            if key in FIGURE_TITLES}


def image_maps(first, last, step, n):
    """The maps (indices) of the instantaneous figures from the map first to the map last (None: the last one)
    every step maps, among n maps."""
    last = min(int(last or n), n)
    return range(min(max(int(first), 1), last) - 1, last, max(int(step), 1))


def video_settings_of(saved):
    """The settings of the videos of the main quantities saved ({figure: settings}, the figures that no longer exist
    left out)."""
    return {key: video_settings(value) for key, value in (saved if isinstance(saved, dict) else {}).items()
            if key in FIGURE_TITLES}


def read_step_fps(step, fps):
    """Every N maps and frames per second entered (texts): ValueError with a message if they are invalid."""
    try:
        step, fps = int(step), float(fps)
    except ValueError:
        step = fps = 0
    if step < 1 or not fps > 0:
        raise ValueError("Every N maps: an integer >= 1; frames per second: a positive number")
    return step, fps


def export_maps(choice, n):
    """The maps (indices) of an export choice for n maps: from the first to the last one, every step."""
    last = min(int(choice["last"] or n), n)
    return range(min(max(int(choice["first"] or 1), 1), last) - 1, last, max(int(choice["step"]), 1))


def own_video_maps(settings, n):
    """The maps of a video with its own settings (video_settings) among n maps: indices."""
    last = min(settings["last"] or n, n)
    return range(min(settings["first"], last) - 1, last, settings["step"])


def export_index(choice):
    """The map of the instantaneous figures exported (0-based; -1: the last one): the one chosen in the quantities
    (choice["map"]), else the last one."""
    return int(choice["map"]) - 1 if choice.get("map") else -1


def export_options(spec, n):
    """The options of PostProcessing.export of an export specification (quantities.export_spec: choice, params,
    videos) for n maps: the figures, the maps of the instantaneous ones, the videos (those of the quantities with
    their own settings unless the same for all), the parameters of the figures by quantity and the comparisons."""
    choice = export_choice(spec.get("choice"))
    params = figure_params_of(spec.get("params"))
    maps = export_maps(choice, n)
    own = {key: dict(maps=list(own_video_maps(value, n)), fps=value["fps"], format=value["format"])
           for key, value in video_settings_of(spec.get("videos")).items() if key in choice["videos"]}
    # The instantaneous figures: the maps chosen (None: the last map), those of the quantities with their own
    images = (list(maps) if choice["range"] else list(image_maps(choice["map"], choice["map_last"] or choice["map"],
                                                                 choice["map_step"], n)) if choice["map"] else None)
    own_images = {key: list(image_maps(value["first"], value["last"], value["step"], n))
                  for key, value in image_settings_of(spec.get("images")).items()}
    return dict(figures=choice["figures"], maps=images, videos=choice["videos"], image_options=own_images,
                fps=choice["fps"], video_maps=list(maps), video_format=choice["format"], video_options=own,
                clims=params["clims"], cmaps=params["cmaps"], contours=params["contours"],
                compares={k: (v, params["wipers"].get(k, COMPARE_FRACTION)) for k, v in params["compares"].items()})


def results_method(section):
    """The results post-processed of the section postprocessing of a project file: PTV if its maps folder is its
    subfolder PTV, else PIV (the default: those of the PTV for a project without PIV results)."""
    maps = section.get("maps_folder")
    return "PTV" if isinstance(maps, str) and os.path.normpath(maps) == piv.PTV_FOLDER else "PIV"


def post_to_project(section, settings):
    """The section postprocessing of a project file with the batch settings of the post-processing: the recipe,
    the results post-processed (the maps folder: the subfolder PTV, or the default one for PIV), the scale space,
    the camera (as the side panel keeps it), the export folder and the quantities exported (the rest kept: the
    tiles, another maps folder, the last export)."""
    out = dict(section)
    if settings.get("results") == "PTV":
        out["maps_folder"] = piv.PTV_FOLDER
    elif settings.get("results") == "PIV" and results_method(out) == "PTV":
        out["maps_folder"] = None
    if settings.get("settings") is not None:
        out["settings"] = settings["settings"]
    if settings.get("space"):
        out["space"] = settings["space"]
    camera = settings.get("camera")
    out["camera"] = ({"fov_width": num2str(camera["fov_width_mm"]), "x_offset": num2str(camera["x_offset_mm"]),
                      "y_offset": num2str(camera.get("y_offset_mm") or 0),
                      "frame_rate": num2str(camera["frame_rate_hz"])} if camera
                     else {**(out.get("camera") or {}), "fov_width": ""})
    if "export_folder" in settings:
        out["export_folder"] = settings["export_folder"]
    spec = settings.get("export")
    if isinstance(spec, dict):
        out.update(export_choice=spec.get("choice"), figure_params=spec.get("params"),
                   video_settings=spec.get("videos"), image_settings=spec.get("images"))
    return out


def post_from_project(section):
    """The batch settings of the post-processing of the section postprocessing of a project file: the results
    post-processed, the recipe, the scale space, the camera, the export folder and the quantities exported."""
    camera = section.get("camera") or {}
    try:
        camera = {"fov_width_mm": float(camera["fov_width"]), "x_offset_mm": float(camera.get("x_offset") or 0),
                  "y_offset_mm": float(camera.get("y_offset") or 0),
                  "frame_rate_hz": float(camera.get("frame_rate") or piv.PRESET_FRAME_RATE)}
    except (KeyError, TypeError, ValueError):
        camera = None
    params = section.get("figure_params") if "figure_params" in section else figure_params_from_tiles(section)
    return {"results": results_method(section), "settings": section.get("settings") or settings_record(PostSettings()),
            "space": section.get("space") or "physical", "camera": camera,
            "export_folder": section.get("export_folder"),
            "export": {"choice": section.get("export_choice"), "params": params,
                       "videos": section.get("video_settings"), "images": section.get("image_settings")}}


def validate_post(settings):
    """Batch settings of the post-processing checked (the recipe: PostSettings): ValueError with a message. The
    camera without the width of the field of view: none; the export specification completed."""
    settings_from_record(settings.get("settings") or {})
    settings["export"] = export_spec(settings.get("export"))
    camera = settings.get("camera")
    if isinstance(camera, dict):
        if camera.get("fov_width_mm") is None:
            settings["camera"] = None
        else:
            camera.setdefault("x_offset_mm", 0.0)
            camera.setdefault("y_offset_mm", 0.0)
            camera.setdefault("frame_rate_hz", piv.PRESET_FRAME_RATE)
            piv.Camera(float(camera["fov_width_mm"]), float(camera["x_offset_mm"]), float(camera["frame_rate_hz"]),
                       y_offset_mm=float(camera["y_offset_mm"]))
    return settings


def settings_record(settings):
    """Settings of the post-processing, as saved (JSON)."""
    return asdict(settings)


def settings_from_record(record):
    """Recipe of saved settings (settings_record), the settings that no longer exist ignored: TypeError or
    ValueError if it is invalid."""
    names = {f.name for f in dataclass_fields(PostSettings)}
    return PostSettings(**{key: value for key, value in record.items() if key in names})


def export_spec(saved):
    """An export specification (saved: in the project or the batch settings), completed: {"choice": export choice,
    "params": parameters of the figures by quantity, "videos": settings of the videos of the quantities with their
    own, "images": instantaneous maps of the quantities with their own}."""
    saved = saved if isinstance(saved, dict) else {}
    return {"choice": export_choice(saved.get("choice")),
            "params": {k: (dict(v) if k != "clims" else {q: list(c) for q, c in v.items()})
                       for k, v in figure_params_of(saved.get("params")).items()},
            "videos": video_settings_of(saved.get("videos")),
            "images": image_settings_of(saved.get("images"))}


def describe_spec(spec):
    """An export specification in a line."""
    choice = spec["choice"]
    figures = len(FIGURE_TITLES) if choice["figures"] is None else len(choice["figures"])
    params = spec["params"]
    tuned = {k for name in ("clims", "cmaps", "contours", "compares") for k in params[name]}
    return (f"{figures} figures, {len(choice['videos'])} videos"
            + (f", {len(tuned)} quantities with their own parameters" if tuned else "")
            + (f", {len(spec['videos'])} with their own video settings" if spec["videos"] else ""))
