"""Page of the post-processing: the vector fields of the PIV (or of a folder of DynamicStudio CSV exports), in the
sensor space (pixels and pairs) or the physical space (with the camera settings), the recipe of the
post-processing (statistical outlier filtering, reference quantities, peak locking, the add-ons: jet theory; all
the analyses are made) and its figures in tiles (menus: a submenu per group of quantities, the images and the POD in
Pre-Processing, the add-ons in Add-ons). The figures of
the recipe are exported to the subfolder PostProcessing of the folder of the measurement (or to the folder chosen),
with the key numbers and the recipe. The vector fields are those of a folder: the results of the page of the PIV by
default (if there is one), or another folder (vectors.npz, or DynamicStudio CSV exports)."""

import bisect
import glob
import json
import os
import re
import tkinter as tk
from dataclasses import asdict, fields as dataclass_fields, replace
from tkinter import filedialog, ttk

from PIL import Image

from .. import imaging, jet, lagrangian, piv
from ..fields import (VIDEO_FORMATS, Video, Wiper, count_maps, load_csv_folder, load_vectors, num2str, ptv_results,
                      scale_frames, sig2)
from ..postprocessing import (FIGURE_GROUPS, FIGURE_SYMBOLS, FIGURE_TITLES, INSTANTANEOUS, MAP_FIGURES, MENU_PARENTS,
                              PIV_GROUP, POSTPROCESSING_FOLDER, PostProcessing, PostSettings, count_results,
                              default_export_folder, previous_results, remove_results)
from ..pipelines import chosen_results, pipelines, results_folders, selected_results
from ..preprocessing import same_path
from .batch_dialog import BatchStep, Field, PanelForm, run_batch
from .cursor import MapCursor
from .widgets import (Dialog, FigurePanel, FigureSelector, FolderEntry, HelpDot, Page, RangeSlider, SidePanel,
                      ask_colour_range, colour_tools, hover_tip)
from ..prefs import getpref, setpref
from ..settings import (CAMERA_KEYS, COMPARE_FRACTION, camera_of, camera_record, export_choice, export_index, export_maps, export_options,
                        figure_params_from_tiles, figure_params_of, image_settings_of, post_from_project,
                        post_to_project, read_step_fps, settings_from_record, settings_record, validate_post,
                        video_settings, video_settings_of)

# Figures shown in the tiles at the start (the instantaneous ones are browsed with the slider and the arrows)
DEFAULT_VIEWS = ("velocity", "mean_velocity", "turbulence_intensity", "mean_vorticity")
# Second quantity of a tile, compared with the first one with a wiper: the maps, or none (the default)
NO_COMPARISON = "none"
COMPARE_GROUPS = [(None, (NO_COMPARISON,))] + [(group, tuple(k for k in keys if k in MAP_FIGURES))
                                               for group, keys in FIGURE_GROUPS]
COMPARE_TITLES = {**FIGURE_TITLES, NO_COMPARISON: "None"}
# The symbols of the quantities (LaTeX) in plain text, for the widgets (Tk can't show LaTeX): replaced in this order,
# then \overline{...} as "Mean ...", \widetilde{...} as "Med. ..." (Tk draws a combining bar after its letter, not
# over it), the standard deviations, skewness and kurtosis in words (σ_u: RMS u, γ_u: Skew. u, β_u: Kurt. u)
PLAIN_SYMBOLS = [("$", ""), (r"\mathbf{u}", "u"), (r"\mathbf{a}", "a"), (r"\mathrm{valid}", "valid"), (r"\omega", "ω"), (r"\sigma", "σ"),
                 (r"\gamma", "γ"), (r"\beta", "β"), (r"\tau", "τ"), (r"\%", "%"), ("_{d_x}", "_dx"), ("_{d_y}", "_dy"),
                 (r"\partial ", "∂"), (r"\partial", "∂"), (r"\phi", "φ"), (r"\mathrm{PIV}", "PIV")]


def short_title(key, length=14):
    """The text of a quantity on the narrow menu buttons of the tiles: its symbol in plain text (that of the labels of
    its figure: ū, σ_u, Tu_%...), else its name shortened (the name in full in their tip and their menu)."""
    if key in WIDGET_SYMBOLS:
        return WIDGET_SYMBOLS[key]
    if key in WIDGET_NAMES:
        return WIDGET_NAMES[key]
    if key in FIGURE_SYMBOLS:
        return plain_symbol(f"${FIGURE_SYMBOLS[key]}$")
    title = COMPARE_TITLES.get(key, key)
    # Abbreviated: a dot at the end
    return title if len(title) <= length else title[:length - 1].rstrip() + "."


def plain_symbol(symbol):
    """A symbol of a quantity (LaTeX, with $) in plain text, its words capitalised: Mean u, RMS u, Tu_%, ∂(Mean u)/∂y,
    Med. S/N..."""
    for old, new in PLAIN_SYMBOLS:
        symbol = symbol.replace(old, new)
    for command, word in ((r"\overline", "Mean"), (r"\widetilde", "Med.")):
        # In parentheses after ∂ (a derivative of the mean)
        symbol = re.sub(r"(∂?)" + re.escape(command) + r"\{([^}]*)\}",
                        lambda m: f"∂({word} {m[2]})" if m[1] else f"{word} {m[2]}", symbol)
    for letter, word in (("σ", "RMS"), ("γ", "Skew."), ("β", "Kurt.")):
        symbol = re.sub(letter + r"_\{?(\w)\}?", rf"{word} \1", symbol)
    # Without the marks of the indices and of the percentages (Tu_% → Tu, d_x → dx)
    for mark in ("{", "}", "%", "_", "^"):
        symbol = symbol.replace(mark, "")
    return symbol


# The quantities whose symbol on the menu buttons of the tiles isn't that of their figures: the turbulence intensity
# of the fluctuation of each map (a prime), the integral scales (Lt: time; Lx u: of u along x...), the image of the PIV
WIDGET_SYMBOLS = {"turbulence_intensity_map": "Tu'", "integral_time": "Lt", "integral_length_x": "Lx u",
                  "integral_length_y": "Ly u", "integral_length_vx": "Lx v", "integral_length_vy": "Ly v",
                  "ke_time": "Lt k-ε", "ke_length": "L k-ε", "piv_image": "I PIV", "valid": "Valid"}
# The figures without a symbol on the menu buttons of the tiles: their names abbreviated (a dot at the end of each
# abbreviation)
WIDGET_NAMES = {"peak_locking": "Peak locking", "peak_ratio_histogram": "S/N histogram", "convergence": "Convergence",
                "ptv_tracks": "PTV tracks", "ptv_yield": "Part. tracked", "ptv_track_length": "Track lengths",
                "ptv_vs_piv": "PTV vs PIV", "pod_spectrum": "POD spectrum", "core_profiles": "Core profiles",
                "core_centreline": "Core centrel."}


# Camera settings, which convert the vector fields of the PIV (in pixels) into mm and m/s: key (that of the project
# file: settings.CAMERA_KEYS), label, default, help
CAMERA_SETTINGS = [
    ("fov_width", "Field of view width (mm)", "",
     "Width of the field of view: the horizontal pixel pitch (mm/px) is this width divided by the width of the images "
     "in pixels.\n\n" + jet.fov_help()),
    ("fov_height", "Field of view height (mm)", "",
     "Height of the field of view: the vertical pixel pitch (mm/px) is this height divided by the height of the "
     "images in pixels. Empty: the horizontal pitch (square pixels)."),
    ("x_shift", "Origin of x (fraction of the width)", "0",
     "Where x = 0 is, from the left edge of the images, as a fraction of the width of the field of view: 0 the left "
     "edge, 0.5 the middle, 1 the right edge (any number: outside the images too). x upwards to the right."),
    ("y_shift", "Origin of y (fraction of the height)", "0",
     "Where y = 0 is, from the bottom edge of the images, as a fraction of the height of the field of view: 0 the "
     "bottom edge, 0.5 the middle (the jet axis of the jet measurements), 1 the top edge. y upwards. Projects of "
     "the previous versions: 0.5."),
    ("x_offset", "Horiz. offset of the field of view (mm)", "0",
     "Position of the origin of x in the global axes, those of the stitching of fields of view: x global = x + "
     "offset. The jet measurements (12 cm between the fields of view):\n" + jet.fov_offsets()),
    ("y_offset", "Vert. offset of the field of view (mm)", "0",
     ("Position of the origin of y in the global axes: y global = y + offset. The stitching keeps only the height "
      "common to all the fields of view.")),
    ("frame_rate", "Frame rate (Hz)", num2str(piv.DEFAULT_FRAME_RATE),
     "Frame rate of the camera; the two images of a pair are consecutive, so dt = 1 / frame rate. "
     f"{num2str(piv.DEFAULT_FRAME_RATE)} Hz by default (times in frames); {num2str(jet.PRESET_FRAME_RATE)} Hz for "
     "the jet measurements (the names of the DynamicStudio exports give the time of each image, e.g. 000001-6.6672 "
     "= 6.6672 ms = 1 / 150 s)."),
]
# The camera settings of the side panel and the keys of piv.Camera (the batch settings)
CAMERA_FIELDS = {key: path for path, key in CAMERA_KEYS.items()}
# Scale space of the figures
SPACES = (("sensor", "Sensor (px, pairs)"), ("physical", "Physical (mm, s)"))
SPACE_HELP = ("Sensor shows the vector fields of the PIV/PTV as they are: positions and displacements in pixels, the "
              "maps numbered by pair, no reference quantities.\n\nPhysical converts them with the camera settings: "
              "positions in mm (or over L_ref), velocities in m/s (or over u_ref), times in s (or in flow times).")
# The results post-processed by the window Batch
RESULTS_HELP = ("The results sets of each project exported: those of the velocimetry pipelines checked (PaIRS: "
                "subfolder PIV, TracTrac: subfolder PTV...) that it has, each with the same settings and quantities, "
                "to the subfolder of its pipeline in PostProcessing (PostProcessing/PaIRS, PostProcessing/TracTrac). "
                "A stitching is made for a results set when all the projects of the stitching have it and export it.")
# Settings of the recipe: section, key, label and help (the reference quantities: REFERENCE_SETTINGS)
SETTINGS = [
    ("Statistical outlier filtering", "outlier_k", "Drop cell if beyond (σ)",
     ("Vectors further than this many standard deviations from the mean at their grid point are removed (0: none; "
      "4 in PostProcessing_General_PIV.py).")),
    ("Data Reconstruction", "hold_maps", "Invalid cells decay (# of maps)",
     "Instantaneous maps: an invalid cell (in the flow) shows the most recent valid value of the cell in the previous "
     "maps, at most this number of maps back; left blank if it has none so recent. Empty or 0: none. The average "
     "delay of the cells filled is given below. The statistics, and the quality maps (peak ratio), are not "
     "changed."),
    ("Peak locking", "peak_locking_bins", "Bins per pixel (odd)",
     ("Number of bins per pixel of the histograms of |d_x| and |d_y|, odd so that each integer is in the middle "
      "of a bin. The quality 1 - C compares the histogram with its moving average over 1 px: too few bins miss "
      "the peaks at the integers, too many make the bins noisy. 21 in PostProcessing_General_PIV.py.")),
    ("Jet theory (add-on)", "core_level", "Core: mean u ≥ (% of u_ref)",
     ("Velocity level detection: the potential core is where the mean u is at least this percentage of the u_ref "
      "of the jet theory (below). Its boundary is that isocontour, closed by the edge of the field where the core "
      "goes beyond it; its centreline is fitted to the middle of its edges, and the core ends at the downstream tip "
      "of the contour. Also used by the skewness detection when no profile meets its criteria.")),
    ("Jet theory (add-on)", "skew_zero", "Skewness ≈ 0 below (% of max)",
     ("Skewness detection: a profile counts as symmetric when its |skewness| is at most this percentage of the "
      "highest |skewness| of the profiles.")),
    ("Jet theory (add-on)", "skew_change", "Skewness change below (% of prev.)",
     ("Skewness detection: the skewness must also differ from that of the previous profile by at most this "
      "percentage of the previous one's |skewness|. The closer the profiles (more of them), the less they differ.")),
    ("Jet theory (add-on)", "skew_profiles", "Profiles for detection",
     ("Skewness detection: number of profiles analysed, spread from the first profile to the end of the field "
      "(at most one per column of vectors). More of them locate the end of the core more finely.")),
    ("Jet theory (add-on)", "n_stations", "Profiles of mean u/u_ref (stations)",
     ("Number of x positions where the profiles of R = mean u / u_ref are shown, with the edges of the core (and "
      "their skewness with the skewness detection).")),
    ("Jet theory (add-on)", "first_offset", "First profile offset (columns)",
     ("Columns of vectors between the exit profile and the first profile of the profiles figure, to stay clear of "
      "the nozzle; the others are spread evenly up to the last one. Empty or 0: the first one at the exit "
      "profile. The velocity level detection doesn't change; the skewness detection starts at the first profile.")),
    ("Jet theory (add-on)", "core_u_ref", "Ref. velocity (m/s)",
     ("u_ref of the jet theory figures (mean u / u_ref) when the reference velocity of the Reference quantities "
      "is off; empty: the maximum of the mean u. Greyed when that one is on, since the jet theory then uses it.")),
]
# The settings of the jet theory (hidden when the add-on is off)
JET_KEYS = {key for section, key, _, _ in SETTINGS if section == "Jet theory (add-on)"}
JET_THEORY_HELP = ("The jet theory is an add-on, for a jet coming from the left of the field of view (its mean u "
                   "along +x): the potential core (where the mean u stays near the velocity of the jet), its "
                   "centreline, its end and the profiles of mean u / u_ref along it (figures Jet theory), with its "
                   "numbers in metrics.json. Off: those figures aren't drawn nor exported. The rest of the "
                   "post-processing makes no assumption on the flow.")
# The boxes inside the box Miscellaneous, before the POD
MISC_SECTIONS = ("Peak locking",)
# Sections in the box Add-ons, below the box Miscellaneous
ADDON_SECTIONS = ("Jet theory (add-on)",)
# The mode of the POD of the images shown (box POD of the box Miscellaneous)
POD_MODE_HELP = ("Modes of the POD of the images (page Pre-processing, POD modes) exported with the figure POD mode "
                 "(Modal decomposition (POD) in the menus of the tiles): the range between the two handles of the "
                 "slider, from 1 to the number of modes evaluated (both handles on the same mode: that mode alone). "
                 "Export: each mode its own PNG ([project]_PostProcessing_PODMode[n]), or their sum: for the image "
                 "of each map, the modes weighted by its temporal coefficients, the background that the "
                 "pre-processing subtracts from it with these modes (modes 1 to N for the background removal of N "
                 "modes). The sum is a figure of each map, as the raw image (exported for the maps chosen, videos, "
                 "compared with the raw image with the wiper: [project]_PostProcessing_PODBackground[first]-[last]). "
                 "The tiles show the mode of the handle moved last (the sum for the map browsed when summed), and "
                 "the spectrum marks the range; their share of the energy below. The POD is that of the images of the "
                 "measurement (its masks and contrast enhancement before POD applied), not of their fluctuations: the "
                 "first mode is close to their mean image. The figures of the images (Images in the menus) show the "
                 "raw image of each map and the image the PIV/PTV computed it from (pre-processed).")
# Detections of the potential core (PostSettings.core_mode): value, label
CORE_MODE_LABELS = (("level", "Velocity level"), ("skewness", "Skewness"))
CORE_MODE_HELP = ("Velocity level: the core is where the mean u is at least the percentage below of u_ref.\n\n"
                  "Skewness: the skewness of each profile is taken about its maximum, the jet centreline. It is large "
                  "while the profile is flat-topped (the potential core) and nears 0 once the profile becomes "
                  "bell-shaped. The core ends at the first profile where the skewness is close to 0 and changes "
                  "little from the previous profile; the maximum of that profile is the cutoff velocity, whose "
                  "isocontour outlines the core. The profiles analysed are more than those shown (Profiles for "
                  "detection).")
# Formats of the videos, as chosen in the dialogs
FORMAT_LABELS = {"gif": "GIF (256 colours, plays everywhere)", "mp4": "MP4 (H.264, full colours, needs ffmpeg)"}


def format_box(master, var):
    """List of the formats of the videos (FORMAT_LABELS), its value in var (the key of the format)."""
    box = ttk.Combobox(master, values=list(FORMAT_LABELS.values()), state="readonly", width=36)
    box.set(FORMAT_LABELS[var.get()])
    box.bind("<<ComboboxSelected>>", lambda event: var.set(next(k for k, v in FORMAT_LABELS.items()
                                                                  if v == box.get())))
    return box


def describe_export(choice, n, own=()):
    """An export choice, in a line: the figures, the maps of the instantaneous ones, the videos (own: the quantities
    with video settings of their own)."""
    figures = len(FIGURE_TITLES) if choice["figures"] is None else len(choice["figures"])
    maps = export_maps(choice, n)
    where = (f"maps {maps.start + 1} to {maps[-1] + 1} every {maps.step} ({len(maps)} images each)" if maps
             else "no map")
    images = ("the last map" if not choice.get("map") else f"map {choice['map']}"
              if (choice.get("map_last") or choice["map"]) == choice["map"]
              else f"maps {choice['map']} to {choice['map_last']} every {choice.get('map_step') or 1}")
    text = f"{figures} of {len(FIGURE_TITLES)} figures; instantaneous: " + (where if choice["range"] else images)
    if choice["videos"]:
        own = [key for key in choice["videos"] if key in own]
        text += (f"; {len(choice['videos'])} video(s)" + (f", {len(own)} with their own settings" if own else "")
                 + (": others " if own else ": ") + f"{choice['format'].upper()}, "
                 + f"{where if not choice['range'] else 'the same maps'}, {num2str(choice['fps'])} frames/s")
    return text


def images_summary(post, first, last, step):
    """The instantaneous maps exported from the map first to the map last every step maps, as the slider of the
    instantaneous maps says them: their number, and their times (in s, and in flow times with the flow time; the pairs
    in the sensor space). post: the post-processing (None: the maps only)."""
    n = post.fields.n if post is not None else None
    last = min(last, n) if n else last
    count = len(range(first - 1, last, max(int(step), 1)))
    text = (f"Map {first}" if first == last else f"Maps {first} to {last} every {step}: {count} images")
    text += f" (of {n})" if n else ""
    if post is None:
        return text
    if post.sensor():
        return text + (f"\nPair {first}" if first == last else f"\nPairs {first} to {last}")
    t0, t1 = float(post.fields.t[first - 1]), float(post.fields.t[last - 1])
    text += f"\nt = {t0:.4g} s" if first == last else f"\nt = {t0:.4g} to {t1:.4g} s"
    if post.uses_flow_time():
        flow = post.flow_time()
        text += (f" = {sig2(t0 / flow)}" + ("" if first == last else f" to {sig2(t1 / flow)}")
                 + f" t_flow (t_flow = {flow * 1000:.3g} ms)")
    return text


def video_summary(post, first, last, step, fps):
    """The maps of a video from the map first to the map last every step maps, at fps frames per second, as the
    window Video says them: their number, the duration of the video and the time of the flow captured (in s, and in
    flow times with the flow time; pairs in the sensor space). post: the post-processing of the vector fields (None:
    the maps only)."""
    frames = len(range(first - 1, last, step))
    text = f"Maps {first} to {last}" + (f" of {post.fields.n}" if post is not None else "")
    text += f": {frames} frames, {frames / fps:.1f} s at {num2str(fps)} frames/s"
    if post is None:
        return text
    if post.sensor():
        return text + f"\nFlow captured: {last - first + 1} pairs (sensor space: no time)"
    t = post.fields.t
    seconds = float(t[last - 1] - t[first - 1]) + post.map_interval()
    text += f"\nFlow time captured: {seconds:.4g} s"
    if post.uses_flow_time():
        text += f" = {sig2(seconds / post.flow_time())} t_flow (t_flow = {post.flow_time() * 1000:.3g} ms)"
    return text


def describe_post(settings):
    """The settings of the post-processing of a batch, in a line."""
    from ..settings import describe_spec, export_spec

    try:
        steps = " → ".join(settings_from_record(settings.get("settings") or {}).steps())
    except (TypeError, ValueError) as e:
        steps = f"recipe not valid ({e})"
    camera = settings.get("camera")
    try:
        described = camera_of(camera).describe() if camera else None
    except (TypeError, ValueError) as e:
        described = f"camera not valid ({e})"
    space = ("sensor space" if settings.get("space") == "sensor" else
             f"physical space, {described}" if camera else "physical space without camera settings")
    return f"{steps or 'default recipe'}; {space}; {describe_spec(export_spec(settings.get('export')))}"


# Labels of the recipe in the window Batch (those of the side panel otherwise)
BATCH_LABELS = {"use_u_ref": "Velocities over u_ref", "u_ref": "Ref. velocity u_ref (m/s; empty: max of mean u)",
                "use_t_flow": "Times in flow times", "t_flow": "Flow time t_flow (s; empty: L_FoV / u_ref)",
                "l_ref": "Ref. length L_ref (mm; empty: positions in mm)", "nu": "Kinematic viscosity ν (m²/s)",
                "rolling": "Rolling statistics",
                "rolling_maps": "Rolling interval (maps)", "core_u_ref": "Jet theory u_ref (m/s; empty: max of mean u)",
                "jet_theory": "Jet theory add-on (potential core)",
                "pod_mode": "POD modes exported (first to last)",
                "pod_combine": "POD modes summed (the background of each image)"}


def batch_maps_folders(project, settings):
    """The vector fields of a project post-processed by the batch: those of the results sets checked (results) that it
    has, {label: folder}."""
    return selected_results(project.results_folders(), settings.get("results"))


def batch_maps_folder(project, settings):
    """The vector fields shown for a project in the window Batch (the maps of its videos...): those of its first
    results set exported, else its maps folder."""
    return next(iter(batch_maps_folders(project, settings).values()), None) or project.maps


def post_fields(settings):
    """The fields of the batch settings of the post-processing (window Batch): the results sets exported (the
    pipelines: PaIRS, TracTrac...), the scale space, the camera, the recipe and the export folder (the quantities: Quantities...)."""
    labels = {key: label for _, key, label, _ in SETTINGS}
    helps = {key: text for _, key, _, text in SETTINGS}
    helps.update({key: text for _, key, _, text in REFERENCE_SETTINGS})
    helps.update(use_u_ref=helps.get("u_ref"), use_t_flow=helps.get("t_flow"), l_ref=L_REF_HELP, nu=NU_HELP,
                 rolling=ROLLING_HELP, pod_mode=POD_MODE_HELP)
    camera = {key: help_ for key, _, _, help_ in CAMERA_SETTINGS}
    fields = [Field(("results",), "Results sets exported", "multi", tuple((label, label) for label in pipelines()),
                    default=list(pipelines()), help=RESULTS_HELP),
              Field(("space",), "Scale space", "choice", tuple((k, label) for k, label in SPACES), default="physical",
                    help=SPACE_HELP),
              Field(("camera", "fov_width_mm"), "Field of view width (mm; empty: sensor space only)", "optional",
                    help=camera["fov_width"]),
              Field(("camera", "fov_height_mm"), "Field of view height (mm; empty: square pixels)", "optional",
                    help=camera["fov_height"]),
              Field(("camera", "x_shift"), "Origin of x (fraction of the width)", "float", default=0.0,
                    help=camera["x_shift"]),
              Field(("camera", "y_shift"), "Origin of y (fraction of the height)", "float", default=0.0,
                    help=camera["y_shift"]),
              Field(("camera", "x_offset_mm"), "Horiz. offset of the field of view (mm)", "float", default=0.0,
                    help=camera["x_offset"]),
              Field(("camera", "y_offset_mm"), "Vert. offset of the field of view (mm)", "float", default=0.0,
                    help=camera["y_offset"]),
              Field(("camera", "frame_rate_hz"), "Frame rate (Hz)", "float", default=piv.DEFAULT_FRAME_RATE,
                    help=camera["frame_rate"])]
    defaults = PostSettings()
    for f in dataclass_fields(PostSettings):
        default = getattr(defaults, f.name)
        if f.name == "pod_mode":
            # The range of the modes: the first and the last ones (a two-sided slider)
            fields.append(Field(("settings", "pod_mode"), BATCH_LABELS["pod_mode"], "range", help=POD_MODE_HELP,
                                default=1, end=("settings", "pod_mode_last")))
            continue
        if f.name == "pod_mode_last":
            continue
        if f.name == "core_mode":
            fields.append(Field(("settings", f.name), "Potential core detection", "choice", CORE_MODE_LABELS,
                                help=CORE_MODE_HELP, default=default))
            continue
        kind = ("bool" if isinstance(default, bool) else "int" if isinstance(default, int)
                else "optional" if default is None else "float")
        fields.append(Field(("settings", f.name), BATCH_LABELS.get(f.name) or labels.get(f.name) or f.name, kind,
                            help=helps.get(f.name), default=default))
    return fields + [Field(("export_folder",), "Export Path (Relative)", "folder", default=POSTPROCESSING_FOLDER,
                           help=("Folder of the figures exported, relative to each project (↺: the subfolder of "
                                 "the pipeline of the results in PostProcessing: PostProcessing/PaIRS, "
                                 "PostProcessing/TracTrac). Set per project, not on folders."), project_only=True)]


# Where the statistical outlier filtering is used (below its settings)
FILTERING_NOTE = ("Used when statistics are performed to filter strong, sudden variations. "
                  "Especially improves the quality of higher order moments")


# Reference quantities of the physical space: toggle, key, label and help (the reference length: no toggle)
REFERENCE_SETTINGS = [
    ("use_u_ref", "u_ref", "Ref. velocity (m/s)",
     ("Shows the velocities over u_ref (the mean and instantaneous velocities, their standard deviations, the "
      "Reynolds stress, and the vorticity with L_ref); empty: the maximum of the mean u. The turbulence intensity "
      "and the default flow time always use u_ref (the maximum of the mean u when off), and so does the potential "
      "core when on. Each figure gives the value used.")),
    ("use_t_flow", "t_flow", "Flow time (ms)",
     ("Shows the times in flow times t_flow (the times of the maps, the averaging time of the statistics and of "
      "the convergence), else in s. Empty: t_flow = L_FoV / u_ref, the time the flow takes to cross the field of "
      "view.")),
]
ROLLING_HELP = ("Computes the statistics of each map over the interval of maps up to it (all the maps before it "
                "while there are fewer). Every figure then has one per map, browsed and exported (images and "
                "videos) like the instantaneous ones. The reference velocity, the flow time and the outlier "
                "filtering still use all the maps; the titles give the maps used.")
# Maximum of the slider of the interval of the rolling statistics before vector fields are read (then: their maps)
ROLLING_MAX = 1000
NU_HELP = ("Kinematic viscosity of the fluid, for the dissipation ε of the k-ε time and length scales (Turbulence). "
           "Air: 1.5e-5 m²/s, water: 1.0e-6 m²/s (at 20 °C). Air by default.")
L_REF_HELP = ("The positions are shown as x / L_ref and y / L_ref, with its value in the titles. The nozzle "
              "diameter: 50 mm for the jet measurements. Empty: positions in mm.")


def format_setting(key, value):
    """A setting as entered (the flow time in ms)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return num2str(round(value * 1000, 6) if key == "t_flow" else value)


class PostPanel(SidePanel):
    """Side panel of the post-processing: the vector fields, the scale space, the camera settings and the recipe
    (statistical outlier filtering, reference quantities, peak locking, jet theory add-on, miscellaneous). The button
    that exports the figures is at the bottom. batch: the settings of the window Batch (BatchPostForm), without the
    vector fields nor the export, nor the preferences; page: what is told of the changes (apply_settings,
    camera_changed, space_changed, describe_rolling, pod_handle)."""

    def __init__(self, master, page, batch=False):
        super().__init__(master)
        self.page, self.batch = page, batch
        wrap = self.wrap
        if not batch:
            self.export_and_maps(page)

        # Camera settings: the vector fields of the PIV are in pixels
        box = self.section("Camera and acquisition")
        grid = self.grid(box)
        saved = getpref("post_camera", {}) if not batch else {}
        self.camera = {}
        for key, label, default, help in CAMERA_SETTINGS:
            self.camera[key] = tk.StringVar(self, saved.get(key, default))
            self.setting(grid, label, self.camera[key], help, command=page.camera_changed)
        self.camera_label = ttk.Label(box, foreground="#555555", wraplength=wrap, justify=tk.LEFT)
        self.camera_label.pack(anchor=tk.W, pady=(4, 0))

        settings = self.saved()
        self.vars = {}
        grids = {}
        for section, key, label, help in SETTINGS:
            if section not in grids:
                # The scale space before the filtering, the reference quantities after the data reconstruction
                if section == "Statistical outlier filtering":
                    self.space_box()
                elif section == "Peak locking":
                    self.reference_box(settings)
                elif section == "Data Reconstruction":
                    self.rolling_box(settings)
                if section in MISC_SECTIONS:
                    # Inside the box Miscellaneous (with the POD below: misc_box)
                    if not hasattr(self, "misc"):
                        self.misc = self.section("Miscellaneous")
                    box = self.section(section, self.misc)
                    box.pack_configure(pady=(0, 6))
                    grids[section] = self.grid(box)
                elif section in ADDON_SECTIONS:
                    # Inside the box Add-ons, below the box Miscellaneous (its POD first)
                    if not hasattr(self, "addons"):
                        self.misc_box(settings)
                        self.addons = self.section("Add-ons")
                    box = self.section(section, self.addons)
                    box.pack_configure(pady=(0, 6))
                    grids[section] = self.grid(box)
                else:
                    grids[section] = self.grid(self.section(section))
                if section == "Statistical outlier filtering":
                    # Where the two settings are used, below them
                    self.filtering_label = ttk.Label(grids[section].master, text=FILTERING_NOTE, foreground="#555555",
                                                     wraplength=wrap, justify=tk.LEFT)
                    self.filtering_label.pack(anchor=tk.W, pady=(4, 0))
                if section == "Data Reconstruction":
                    # The average delay of the invalid cells filled, below the setting (when there is one)
                    self.hold_label = ttk.Label(grids[section].master, foreground="#555555", wraplength=wrap,
                                                justify=tk.LEFT)
                if section == "Jet theory (add-on)":
                    # Its toggle, its settings below (hidden when it is off)
                    self.jet_row(grids[section], settings)
                    grids[section] = self.jet_settings = self.grid(grids[section].master)
                    self.core_mode_row(grids[section], settings)
            self.vars[key] = tk.StringVar(self, format_setting(key, getattr(settings, key)))
            entry = self.setting(grids[section], label, self.vars[key], help, command=page.apply_settings)
            if key == "core_u_ref":
                self.core_u_ref_entry = entry
            if key in ("core_level", "skew_zero", "skew_change", "skew_profiles"):
                self.core_entries[key] = entry
            if key in JET_KEYS:
                self.jet_entries[key] = entry
        self.update_core_u_ref()
        self.update_core_mode()
        if not hasattr(self, "pod_slider"):
            self.misc_box(settings)

    def export_and_maps(self, page):
        """The export at the bottom (folder, quantities, buttons) and the vector fields at the top (the maps
        folder): those of the page only (not in the window Batch)."""
        wrap = self.wrap
        footer = self.section("Export", self.footer,
                              ("Exports the quantities chosen (Quantities...; all the figures by default) as PNG "
                               "to the subfolder of the pipeline of the results in PostProcessing "
                               "(PostProcessing/PaIRS, PostProcessing/TracTrac: the pipelines side by side), named "
                               "[project]_PostProcessing_[figure] (with the map number for the instantaneous "
                               "figures), with metrics.json and postprocessing.json. Instantaneous figures of "
                               "several maps get a folder per figure, and videos (GIF or MP4) if chosen. ... "
                               "picks another folder, ↺ goes back to the default one."))
        self.export_folder = FolderEntry(footer, page.measurement_folder, page.default_export_folder,
                                         "Export folder of the figures...", on_change=page.export_folder_changed)
        self.export_folder.pack(fill=tk.X)
        # The figures to export: chosen in a dialog, summed up next to its button
        row = ttk.Frame(footer)
        row.pack(fill=tk.X, pady=(6, 0))
        self.quantities_button = ttk.Button(row, text="Quantities...", command=page.choose_quantities)
        self.quantities_button.pack(side=tk.LEFT, anchor=tk.N)
        self.export_label = ttk.Label(row, foreground="#555555", wraplength=self.wrap - 130, justify=tk.LEFT)
        self.export_label.pack(side=tk.LEFT, padx=(8, 0))
        row = ttk.Frame(footer)
        row.pack(anchor=tk.E, pady=(6, 0))
        self.batch_button = ttk.Button(row, text="Batch...", command=page.batch)
        self.batch_button.pack(side=tk.LEFT, padx=(0, 6))
        self.export_button = ttk.Button(row, text="Singular   (Ctrl+E)", style="Export.TButton", command=page.export)
        self.export_button.pack(side=tk.LEFT)

        box = self.section("Vector fields", self.header)
        row = self.row(box, pady=(0, 2))
        ttk.Label(row, text="Maps folder").pack(side=tk.LEFT)
        HelpDot(row, ("The PIV/PTV results of this project by default (vectors.npz). ... picks another folder with a "
                      "vectors.npz or DynamicStudio CSV exports (all its CSV files are read); ↺ goes back to the "
                      "PIV/PTV results.")).pack(
            side=tk.LEFT, padx=(4, 0))
        self.maps = FolderEntry(box, page.project_folder, page.default_maps_folder, "Folder of the vector fields...",
                                on_change=page.maps_folder_changed, needs_root=False)
        self.maps.pack(fill=tk.X)
        self.maps_label = ttk.Label(box, wraplength=wrap, justify=tk.LEFT)
        self.maps_label.pack(anchor=tk.W, pady=(4, 0))

    def misc_box(self, settings):
        """Box POD of the box Miscellaneous (below the peak locking): the modes of the POD
        of the images exported (a two-sided slider, from 1 to the modes evaluated: the first and the last ones), each
        exported alone or summed (the background of the image of each map), the POD and the images found below."""
        if not hasattr(self, "misc"):
            self.misc = self.section("Miscellaneous")
        box = self.section("POD", self.misc, help=POD_MODE_HELP)
        row = self.row(box)
        ttk.Label(row, text="POD modes").pack(side=tk.LEFT)
        first, last = settings.pod_mode, settings.pod_mode_last or settings.pod_mode
        self.vars["pod_mode"], self.vars["pod_mode_last"] = tk.StringVar(self, str(first)), tk.StringVar(self, str(last))
        self.pod_range_label = ttk.Label(row, width=8, anchor=tk.E)
        self.pod_range_label.pack(side=tk.RIGHT)
        # Applied when a handle is released (not at every mode while it is dragged)
        self.pod_slider = RangeSlider(row, 1, max(2, last), command=self.pod_slid, on_release=self.pod_released,
                                      width=150)
        self.pod_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.pod_slider.set(first, last)
        row = self.row(box)
        ttk.Label(row, text="Export").pack(side=tk.LEFT)
        self.pod_combine = tk.BooleanVar(self, settings.pod_combine)
        self.pod_buttons = [ttk.Radiobutton(row, text=text, value=value, variable=self.pod_combine,
                                            command=self.page.apply_settings)
                            for value, text in ((False, "Each mode"), (True, "Sum (background)"))]
        for button in self.pod_buttons:
            button.pack(side=tk.LEFT, padx=(8, 0))
        self.pod_label = ttk.Label(box, foreground="#555555", wraplength=self.wrap, justify=tk.LEFT)
        self.pod_label.pack(anchor=tk.W, pady=(4, 0))

    def pod_slid(self, first, last):
        """The slider of the POD modes moved: the range shown (applied when released)."""
        self.vars["pod_mode"].set(str(first))
        self.vars["pod_mode_last"].set(str(last))
        self.pod_range_label["text"] = f"{first} to {last}" if last > first else f"{first}"

    def pod_released(self, first, last, handle):
        """A handle of the POD modes released: applied, the tiles showing its mode."""
        self.page.pod_handle = handle
        self.page.apply_settings()

    def show_pod(self, text, modes):
        """The POD of the images (text) and its modes evaluated: the range of the slider (None: unknown, kept; the
        slider greyed out without modes to choose from)."""
        self.pod_label["text"] = text
        if modes is not None:
            self.pod_slider.set_range(1, max(modes, int(self.vars["pod_mode_last"].get() or 1), 2))
        self.pod_slider.set_enabled(bool(modes and modes >= 2))
        # The sum of a single mode: its share of the background
        for button in self.pod_buttons:
            button.state(["!disabled"] if modes else ["disabled"])

    def jet_row(self, grid, settings):
        """The toggle of the jet theory add-on: its figures, its numbers and its settings below."""
        self.jet_theory = tk.BooleanVar(self, settings.jet_theory)
        self.jet_entries, self.core_buttons = {}, []
        ttk.Checkbutton(grid, text="Jet theory add-on", variable=self.jet_theory,
                        command=self.core_mode_changed).grid(row=grid.grid_size()[1], column=0, columnspan=2,
                                                             sticky=tk.W, pady=(0, 2))
        HelpDot(grid, JET_THEORY_HELP).grid(row=grid.grid_size()[1] - 1, column=2, padx=(4, 0))

    def core_mode_row(self, grid, settings):
        """Row of the detection of the potential core: velocity level or skewness."""
        self.core_mode = tk.StringVar(self, settings.core_mode)
        self.core_entries = {}
        row = ttk.Frame(grid)
        row.grid(row=grid.grid_size()[1], column=0, columnspan=2, sticky=tk.W, pady=(0, 2))
        ttk.Label(row, text="Detection").pack(side=tk.LEFT, padx=(0, 8))
        for value, text in CORE_MODE_LABELS:
            button = ttk.Radiobutton(row, text=text, value=value, variable=self.core_mode,
                                     command=self.core_mode_changed)
            button.pack(side=tk.LEFT, padx=(0, 8))
            self.core_buttons.append(button)
        HelpDot(grid, CORE_MODE_HELP).grid(row=grid.grid_size()[1] - 1, column=2, padx=(4, 0))

    def core_mode_changed(self):
        self.update_core_mode()
        self.page.apply_settings()

    def update_core_mode(self):
        """The settings of the jet theory hidden when it is off, those of the detection not chosen greyed out."""
        on = self.jet_theory.get()
        if on and not self.jet_settings.winfo_manager():
            self.jet_settings.pack(fill=tk.X)
        elif not on:
            self.jet_settings.pack_forget()
        skewness = self.core_mode.get() == "skewness"
        for key, entry in self.jet_entries.items():
            used = key not in self.core_entries or (key != "core_level") == skewness
            entry.configure(state=tk.NORMAL if on and used else tk.DISABLED)
        for button in self.core_buttons:
            button.state(["!disabled"] if on else ["disabled"])
        if on and hasattr(self, "core_u_ref_entry"):
            self.update_core_u_ref()

    def rolling_box(self, settings):
        """Box of the rolling statistics: its toggle (off by default), the interval (maps: slider and entry) and its
        duration below."""
        box = self.section("Rolling statistics", help=ROLLING_HELP)
        self.rolling = tk.BooleanVar(self, settings.rolling)
        ttk.Checkbutton(box, text="Rolling statistics", variable=self.rolling,
                        command=self.page.apply_settings).pack(anchor=tk.W)
        row = self.row(box)
        ttk.Label(row, text="Interval (maps)").pack(side=tk.LEFT)
        self.vars["rolling_maps"] = tk.StringVar(self, str(settings.rolling_maps))
        entry = ttk.Entry(row, textvariable=self.vars["rolling_maps"], width=6)
        entry.pack(side=tk.RIGHT)
        entry.bind("<Return>", lambda event: self.rolling_entered())
        entry.bind("<FocusOut>", lambda event: self.rolling_entered())
        self.rolling_scale = ttk.Scale(row, orient=tk.HORIZONTAL, from_=1, to=max(ROLLING_MAX, settings.rolling_maps))
        self.rolling_scale.set(settings.rolling_maps)
        self.rolling_scale.configure(command=self.rolling_slid)
        self.rolling_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        # Applied when the slider is released (not at every map while it is dragged)
        for sequence in ("<ButtonRelease-1>", "<KeyRelease>"):
            self.rolling_scale.bind(sequence, lambda event: self.page.apply_settings())
        self.rolling_label = ttk.Label(box, foreground="#555555", wraplength=self.wrap, justify=tk.LEFT)
        self.rolling_label.pack(anchor=tk.W, pady=(4, 0))

    def rolling_slid(self, value):
        """The slider of the interval moved: its number of maps and its duration shown (applied when released)."""
        self.vars["rolling_maps"].set(str(int(round(float(value)))))
        self.rolling_label["text"] = self.page.describe_rolling(int(round(float(value))))

    def rolling_entered(self):
        """The interval entered: the slider follows it (applied)."""
        try:
            maps = int(float(self.vars["rolling_maps"].get()))
        except ValueError:
            maps = None
        if maps is not None and maps >= 1:
            self.rolling_scale.configure(to=max(float(self.rolling_scale["to"]), maps))
            self.rolling_scale.set(maps)
        self.page.apply_settings()

    def rolling_range(self, n):
        """The slider of the interval up to the n maps of the vector fields (or up to the interval set)."""
        self.rolling_scale.configure(to=max(n, 2, int(float(self.vars["rolling_maps"].get() or 1))))

    def update_core_u_ref(self):
        """The reference velocity of the jet theory: its own entry when the one of the reference quantities is off,
        else greyed out showing that one (greyed out too in the sensor space, where it isn't used)."""
        shared = self.uses["use_u_ref"].get()
        off = not self.jet_theory.get()
        self.core_u_ref_entry.configure(textvariable=self.vars["u_ref" if shared else "core_u_ref"],
                                        state=tk.DISABLED if off or shared or self.space.get() == "sensor"
                                        else tk.NORMAL)

    def space_box(self):
        """Box of the scale space: the vector fields of the PIV in pixels, or converted with the camera settings."""
        box = self.section("Scale space", help=SPACE_HELP)
        space = getpref("post_space", "physical")
        self.space = tk.StringVar(self, space if space in dict(SPACES) else "physical")
        row = self.row(box)
        for value, text in SPACES:
            ttk.Radiobutton(row, text=text, value=value, variable=self.space, command=self.page.space_changed).pack(
                side=tk.LEFT, padx=(0, 12))

    def show_hold(self, text):
        """The average delay of the invalid cells filled by previous map data, below its setting; hidden without text."""
        self.hold_label["text"] = text
        if text and not self.hold_label.winfo_manager():
            self.hold_label.pack(anchor=tk.W, pady=(4, 0))
        elif not text:
            self.hold_label.pack_forget()

    def reference_box(self, settings):
        """Box of the reference quantities: the reference velocity and the flow time, each with its toggle (its
        entry shown when it is on), and the reference length at the bottom; the values used below (all greyed out
        in the sensor space, where they aren't used)."""
        box = self.section("Reference quantities")
        grid = self.grid(box)
        self.uses, self.reference_entries, self.reference_widgets = {}, {}, []
        for use, key, label, help in REFERENCE_SETTINGS:
            r = grid.grid_size()[1]
            self.uses[use] = tk.BooleanVar(self, getattr(settings, use))
            check = ttk.Checkbutton(grid, text=label, variable=self.uses[use], command=self.reference_toggled)
            check.grid(row=r, column=0, sticky=tk.W, pady=1)
            self.reference_widgets.append(check)
            self.vars[key] = tk.StringVar(self, format_setting(key, getattr(settings, key)))
            entry = self.reference_entries[use] = ttk.Entry(grid, textvariable=self.vars[key], width=9)
            entry.grid(row=r, column=1, sticky=tk.E, pady=1)
            self.reference_widgets.append(entry)
            entry.bind("<Return>", lambda event: self.page.apply_settings())
            entry.bind("<FocusOut>", lambda event: self.page.apply_settings())
            HelpDot(grid, help).grid(row=r, column=2, padx=(4, 0))
        self.vars["l_ref"] = tk.StringVar(self, format_setting("l_ref", settings.l_ref))
        self.reference_widgets.append(self.setting(grid, "Ref. Length (mm)", self.vars["l_ref"], L_REF_HELP,
                                                   command=self.page.apply_settings))
        self.vars["nu"] = tk.StringVar(self, format_setting("nu", settings.nu))
        self.reference_widgets.append(self.setting(grid, "Kin. viscosity ν (m²/s)", self.vars["nu"], NU_HELP,
                                                   command=self.page.apply_settings))
        self.reference_label = ttk.Label(box, foreground="#555555", wraplength=self.wrap, justify=tk.LEFT)
        self.reference_label.pack(anchor=tk.W, pady=(4, 0))
        self.show_reference_entries()
        self.update_reference_state()

    def update_reference_state(self):
        """The reference quantities greyed out in the sensor space (they don't change its figures)."""
        sensor = self.space.get() == "sensor"
        for widget in self.reference_widgets:
            if isinstance(widget, ttk.Checkbutton):
                widget.state(["disabled"] if sensor else ["!disabled"])
            else:
                widget.configure(state=tk.DISABLED if sensor else tk.NORMAL)
        if hasattr(self, "core_u_ref_entry"):
            self.update_core_u_ref()

    def show_reference_entries(self):
        """The entry of a reference quantity, only when its toggle is on (and the reference velocity of the jet
        theory)."""
        for use, entry in self.reference_entries.items():
            if self.uses[use].get():
                entry.grid()
            else:
                entry.grid_remove()
        if hasattr(self, "core_u_ref_entry"):
            self.update_core_u_ref()

    def reference_toggled(self):
        self.show_reference_entries()
        self.page.apply_settings()

    def read_camera(self):
        """Camera settings entered, None if the width of the field of view isn't: ValueError with a message if
        they are invalid."""
        c = self.camera
        if not c["fov_width"].get().strip():
            return None
        try:
            return camera_of({path: float(c[key].get()) for key, path in CAMERA_FIELDS.items()
                              if c[key].get().strip()})
        except ValueError as e:
            raise ValueError(f"Camera settings: {e}") from None

    def camera_settings(self):
        return {key: var.get() for key, var in self.camera.items()}

    def update_camera_label(self, width, n_images=None, height=None):
        """Pixel pitches, time between the images of a pair and final time of the image set (width, height: of the
        images in pixels, n_images: their number; None if unknown)."""
        try:
            camera = self.read_camera()
        except ValueError as e:
            self.camera_label["text"] = str(e)
            return
        if camera is None:
            self.camera_label["text"] = ("Enter the width of the field of view: the vector fields of the PIV/PTV are in "
                                         "pixels, converted into mm and m/s with these settings.")
            return
        parts = []
        if width and camera.fov_height_mm and height:
            parts.append(f"Pixel pitches {camera.scale(width):.5f} x {camera.scale_y(width, height):.5f} mm/px "
                         f"(images {width} x {height} px)")
        elif width:
            parts.append(f"Pixel pitch {camera.scale(width):.5f} mm/px (images {width}px wide"
                         + (", square pixels)" if not camera.fov_height_mm else ")"))
        parts.append(f"Δt = {camera.dt * 1000:.4g} ms between the images of a pair")
        if n_images:
            parts.append(f"Final time of the image set: {(n_images - 1) / camera.frame_rate_hz:.4g} s "
                         f"({n_images} images)")
        self.camera_label["text"] = "\n".join(parts)

    @staticmethod
    def saved():
        """Recipe of the last session (the defaults if there is none or it is invalid)."""
        try:
            return settings_from_record(getpref("post_settings", {}))
        except (TypeError, ValueError):
            return PostSettings()

    def read(self):
        """Recipe entered: ValueError with a message if it is invalid."""
        values = {use: var.get() for use, var in self.uses.items()}
        for key, var in self.vars.items():
            text = var.get().strip()
            if key in ("u_ref", "t_flow", "l_ref", "core_u_ref", "hold_maps", "first_offset", "pod_mode_last"):
                values[key] = float(text) if text else None
            else:
                values[key] = float(text)
        # The flow time entered in ms
        if values["t_flow"] is not None:
            values["t_flow"] /= 1000
        return PostSettings(rolling=self.rolling.get(), jet_theory=self.jet_theory.get(),
                            core_mode=self.core_mode.get(), pod_combine=self.pod_combine.get(), **values)

    def show(self, settings):
        for key, var in self.vars.items():
            var.set(format_setting(key, getattr(settings, key)))
        self.rolling.set(settings.rolling)
        self.rolling_scale.set(settings.rolling_maps)
        self.pod_slider.set(settings.pod_mode, settings.pod_mode_last or settings.pod_mode)
        self.pod_combine.set(settings.pod_combine)
        self.jet_theory.set(settings.jet_theory)
        self.core_mode.set(settings.core_mode)
        self.update_core_mode()
        for use, var in self.uses.items():
            var.set(getattr(settings, use))
        self.show_reference_entries()

    @staticmethod
    def save(settings):
        setpref("post_settings", settings_record(settings))


# The settings of the side panel in the window Batch: the paths of the batch settings it shows, and the label of each
# setting in the panel (bold when the projects of a node differ)
PANEL_PATHS = ("space", "camera", "settings")
PANEL_LABELS = {**{label: [("settings", key)] for _, key, label, _ in SETTINGS},
                **{label: [("settings", use), ("settings", key)] for use, key, label, _ in REFERENCE_SETTINGS},
                **{label: [("camera", CAMERA_FIELDS[key])] for key, label, _, _ in CAMERA_SETTINGS},
                "Ref. Length (mm)": [("settings", "l_ref")], "Kin. viscosity ν (m²/s)": [("settings", "nu")],
                "Rolling statistics": [("settings", "rolling")],
                "Interval (maps)": [("settings", "rolling_maps")], "Detection": [("settings", "core_mode")],
                "Jet theory add-on": [("settings", "jet_theory")],
                "Scale space": [("space",)], "POD modes": [("settings", "pod_mode"), ("settings", "pod_mode_last")],
                "Export": [("settings", "pod_combine")]}
class BatchPage:
    """What the side panel tells the page, in the window Batch: its changes, to the form (BatchPostForm.commit)."""

    def __init__(self, form):
        self.form, self.pod_handle = form, None

    def apply_settings(self):
        self.form.commit()

    camera_changed = space_changed = apply_settings

    @staticmethod
    def describe_rolling(maps=None):
        return ""


class BatchPostForm(PanelForm):
    """The settings of the post-processing of the node selected in the window Batch (PanelForm): those of the side
    panel of the page (PostPanel in batch mode: scale space, camera, recipe, POD modes), the results post-processed and
    the export folder above it."""

    PATHS, LABELS = PANEL_PATHS, PANEL_LABELS

    def make_panel(self):
        return PostPanel(self, BatchPage(self), batch=True)

    def load_panel(self, settings, unavailable=None, limits=None):
        panel = self.panel
        try:
            recipe = settings_from_record(settings.get("settings") or {})
        except (TypeError, ValueError):
            recipe = PostSettings()
        panel.show(recipe)
        panel.space.set(settings.get("space") or "physical")
        camera = settings.get("camera") or {}
        defaults = {key: default for key, _, default, _ in CAMERA_SETTINGS}
        for key, path in CAMERA_FIELDS.items():
            value = camera.get(path)
            panel.camera[key].set(defaults[key] if value is None and path != "fov_width_mm" and key != "fov_height"
                                  else "" if value is None else num2str(value))
        panel.update_camera_label(None)
        if unavailable is not None:
            reason = unavailable.get(("settings", "pod_mode"))
            modes = (limits or {}).get(("settings", "pod_mode"), (1, None))[1]
            panel.show_pod(f"Not available: {reason}" if reason else
                           f"Up to {modes} modes evaluated by the projects of the node" if modes else
                           "The modes evaluated are unknown", None if reason else modes)
        self.unavailable = unavailable if unavailable is not None else getattr(self, "unavailable", {})

    def restore_states(self, unavailable):
        panel = self.panel
        panel.update_reference_state()
        panel.show_reference_entries()
        panel.update_core_mode()
        if (self.unavailable or {}).get(("settings", "pod_mode")):
            panel.pod_slider.set_enabled(False)
            for button in panel.pod_buttons:
                button.state(["disabled"])

    def record(self):
        """The settings entered in the panel, as batch settings (space, camera, recipe): ValueError if invalid."""
        panel = self.panel
        recipe = panel.read()
        camera = panel.read_camera()
        self.panel.page.pod_handle = None  # The mode shown: the page's only
        return {"space": panel.space.get(), "settings": settings_record(recipe),
                "camera": None if camera is None else camera_record(camera)}


OVERWRITE = "Overwrite all"


def overwrite_question(app, conflicts, what="the export"):
    """Ask once whether the results already there are overwritten (conflicts: [(folder, paths of
    previous_results)]): True to overwrite all of them (removed before the export, so that each folder holds the new
    results only), False to keep them (nothing exported)."""
    conflicts = [(folder, paths) for folder, paths in conflicts if paths]
    if not conflicts:
        return True
    n = sum(count_results(paths) for _, paths in conflicts)
    folders = [folder for folder, _ in conflicts]
    where = (folders[0] if len(folders) == 1
             else f"{len(folders)} folders ({folders[0]}, ...)")
    example = os.path.basename(conflicts[0][1][0])
    message = (f"{n} files of previous results are in {where} (e.g. {example}).\n\nOverwrite all of them? They are "
               f"removed before {what}, so that the folder holds the new results only (no figure, map or video "
               "left from the previous export). Cancel: nothing is exported, the previous results are kept.")
    return app.questdlg(message, "Overwrite the results", [OVERWRITE, "Cancel"], OVERWRITE) == OVERWRITE


def describe_fields(f):
    """Number of the vector fields and where they come from, as shown in the side panel."""
    ny, nx = f.shape
    lines = [f"{f.n} maps of {nx} x {ny} vectors"]
    settings = f.settings
    if settings.get("engine"):
        lines.append(f"PIV/PTV: {settings['engine']}, {settings.get('date', '')}")
    if settings.get("preprocessing"):
        lines.append(f"Images: {settings['preprocessing']}")
    return "\n".join(lines)


class PostProcessingPage(Page):
    """Page of the post-processing: 4 tiles, each showing a figure of the recipe chosen in its list, and the side
    panel. The slider and the arrow keys (over any tile) browse the maps: the tiles of the instantaneous figures
    follow them. A tile may compare its map with a second map chosen in its second list: drawn over the first one,
    the first left of a wiper and the second right of it, the wiper moved with the slider below the tile (shown only
    with a second quantity; none at the start). The two quantities are named by their symbols."""

    key = "post"
    needs_images = False
    title = "Post-processing"
    project_key = "postprocessing"

    def __init__(self, master, app):
        super().__init__(master, app)
        self.post = None  # PostProcessing of the vector fields loaded
        self.index = 0  # Map of the instantaneous figures
        self.loaded = None  # Folder of the vector fields read (None: to be read when the page is shown)
        self.loaded_camera = None  # The camera settings they were read with
        self.calibration = {}  # Time between the images (s) and scale (mm/px) of the folders of CSV exports
        self.export_choice = export_choice(getpref("post_export"))  # The figures to export (quantities)
        # The settings of the videos of the tiles, by main quantity (VideoDialog): the same for all its tiles
        self.video_settings = video_settings_of(getpref("post_videos"))
        # The instantaneous maps exported of the quantities with their own (Quantities...)
        self.image_settings = image_settings_of(getpref("post_images"))
        self.placing_contour = False  # The slider of an isocontour set by the page (not moved)
        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        grid = ttk.PanedWindow(body, orient=tk.VERTICAL)
        body.add(grid, weight=1)
        self.panel = PostPanel(body, self)
        body.add(self.panel, weight=0)
        views = getpref("post_views", list(DEFAULT_VIEWS))
        if len(views) != 4 or any(v not in FIGURE_TITLES for v in views):
            views = list(DEFAULT_VIEWS)
        # The parameters of the figures by quantity, shared by the tiles showing it and used by the export: colour
        # range {key: (vmin, vmax)}, colour map {key: name}, isocontour {key: value}, second quantity compared with
        # a wiper {key: map} and the position of the wiper {key: fraction}
        self.clims, self.cmaps, self.contours, self.compares, self.wipers = {}, {}, {}, {}, {}
        self.tiles = []
        for r in range(2):
            row = ttk.PanedWindow(grid, orient=tk.HORIZONTAL)
            grid.add(row, weight=1)
            for c in range(2):
                n = 2 * r + c
                tile = FigurePanel(row, (), "constrained", figsize=(5, 3.4), item="Map", on_slide=self.slide_to,
                                   on_activate=lambda n=n: self.set_active_tile(self.tiles[n]),
                                   on_compare=lambda fraction, n=n: self.wipe(self.tiles[n], fraction),
                                   compare_labels=("", ""), compare_value=COMPARE_FRACTION)
                tile.view = views[n]
                # Second quantity (None: no comparison) and position of the wiper (those of its quantity), the
                # wiper drawn
                tile.view2, tile.fraction, tile.wiper = None, COMPARE_FRACTION, None
                # The parameters of the figures by quantity (those of the page)
                tile.clims, tile.cmaps, tile.contours = self.clims, self.cmaps, self.contours
                # Menu of the figures: a submenu per group of quantities
                # (narrow: the symbol of the quantity, its name in the menu and in the tip of the button)
                tile.selector = FigureSelector(tile.bar, FIGURE_GROUPS, FIGURE_TITLES,
                                               lambda key, tile=tile: self.view_selected(tile, key), width=14,
                                               parents=MENU_PARENTS, short=short_title)
                tile.selector.pack(side=tk.LEFT, padx=(2, 4), before=tile.toolbar)
                # Menu of the second quantity: the maps
                tile.selector2 = FigureSelector(tile.bar, COMPARE_GROUPS, COMPARE_TITLES,
                                                lambda key, tile=tile: self.view2_selected(tile, key), width=10,
                                                parents=MENU_PARENTS,
                                                short=lambda key: "Compare" if key == NO_COMPARISON else short_title(key))
                tile.selector2.pack(side=tk.LEFT, padx=(0, 4), before=tile.toolbar)
                # Colour map of the figure (the one of the figure by default), next to its colour range
                tile.cmap_box = colour_tools(tile.bar, tile.toolbar,
                                             lambda cmap, tile=tile: self.cmap_selected(tile, cmap),
                                             lambda tile=tile: self.ask_clim(tile))
                # Isocontour of the map shown: its value for each figure ({key: value}), toggled in the bar, chosen
                # with the slider below the figure (in the colour range of the map)
                tile.contour_sets = []
                # The x-y cursor of its maps (a left click freezes it), the value on the side of the wiper
                tile.cursor = MapCursor(tile.canvas, tile.toolbar,
                                        side=lambda tile=tile: tile.fraction if tile.view2 is not None else None)
                tile.contour_var = tk.BooleanVar(self, False)
                tile.contour_check = ttk.Checkbutton(tile.bar, text="Contour", variable=tile.contour_var,
                                                     command=lambda tile=tile: self.contour_toggled(tile))
                tile.contour_check.pack(side=tk.LEFT, padx=(0, 4), before=tile.toolbar)
                hover_tip(tile.contour_check, "Isocontour of the map, its value chosen with the slider below it")
                tile.contour_bar = ttk.Frame(tile, padding=(8, 2, 8, 0))
                ttk.Label(tile.contour_bar, text="Isocontour").pack(side=tk.LEFT)
                tile.contour_value = ttk.Label(tile.contour_bar, width=17, anchor=tk.E)
                tile.contour_value.pack(side=tk.RIGHT, padx=(8, 0))
                tile.contour_slider = ttk.Scale(tile.contour_bar, orient=tk.HORIZONTAL, from_=0, to=1,
                                                command=lambda value, tile=tile: self.contour_slid(tile, float(value)))
                tile.contour_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
                # Video of what the tile shows over the maps: only for the instantaneous figures (show_selectors)
                tile.video_button = tk.Button(tile.bar, text="Video",
                                              command=lambda tile=tile: self.export_video(tile))
                hover_tip(tile.video_button, "Video of what the tile shows over the maps...")
                tile.show_compare(False)
                row.add(tile, weight=1)
                self.tiles.append(tile)
        self.active_tile = self.tiles[0]
        self.clear()

    # ------------------------------------------------------------------ page

    def menus(self):
        return [("Post-processing", [("Open vector fields...", None, self.open_dialog),
                                     ("Export the figures", "Ctrl+E", self.export), None,
                                     ("Clear the cache", None, self.clear_cache)])]

    def clear_cache(self):
        """File > Clear cache: the vector fields are read again from the maps folder, and the
        analyses evaluated again."""
        self.loaded = self.post = None
        if self.app.current_page() == self.key:
            self.shown()
        else:
            self.clear()
            self.update_panel()
        self.app.update_status("Post-processing cache cleared: the vector fields are read again and the analyses "
                               "evaluated again")

    def folders(self):
        return [("Post-processing", self.export_folder)]

    def start(self):
        self.update_panel()

    def project_state(self, rel):
        export = None
        folder = self.export_folder()
        default_export = self.panel.export_folder.is_default()
        try:
            with open(os.path.join(folder, "postprocessing.json"), encoding="utf-8") as f:
                record = json.load(f)
            export = {"date": record.get("date"), "recipe": record.get("recipe")}
        except (OSError, ValueError, TypeError):
            pass
        settings = self.post.settings if self.post is not None else self.panel.saved()
        maps = self.panel.maps
        return {"settings": settings_record(settings), "space": self.panel.space.get(),
                "camera": self.panel.camera_settings(),
                "views": [t.view for t in self.tiles],
                "figure_params": self.figure_params(),
                "maps_folder": None if maps.is_default() else rel(maps.get()),
                "calibration": self.calibration.get(os.path.normcase(maps.get() or "")),
                "export_folder": None if default_export else rel(folder), "export": export, "export_choice": dict(self.export_choice),
                "video_settings": {key: dict(value) for key, value in self.video_settings.items()},
                "image_settings": {key: dict(value) for key, value in self.image_settings.items()}}

    def restore(self, data, path):
        try:
            settings = settings_from_record(data["settings"])
        except (KeyError, TypeError, ValueError):
            settings = None
        if settings is not None:
            self.panel.show(settings)
            if self.post is not None:
                self.post.settings = settings
        if data.get("space") in dict(SPACES):
            self.panel.space.set(data["space"])
            self.panel.update_reference_state()
        camera = data.get("camera")
        if isinstance(camera, dict):
            if camera.get("fov_width") and "y_shift" not in camera:
                camera = {**camera, "y_shift": "0.5"}  # Saved by a previous version: y = 0 at the middle
            defaults = {key: default for key, _, default, _ in CAMERA_SETTINGS}
            for key, var in self.panel.camera.items():
                var.set(camera[key] if key in camera else defaults[key])
        views = data.get("views") or []
        if len(views) == len(self.tiles) and all(v in FIGURE_TITLES for v in views):
            for tile, view in zip(self.tiles, views):
                tile.view = view
        # The parameters of the figures by quantity (by tile in the projects of the previous versions)
        self.set_figure_params(figure_params_of(data["figure_params"]) if "figure_params" in data
                               else figure_params_from_tiles(data))
        folder = path(data.get("maps_folder"))
        if folder and data.get("calibration"):
            self.calibration[os.path.normcase(folder)] = tuple(data["calibration"])
        self.panel.maps.set(folder)
        self.loaded = None
        # PostProcessing, the default of the previous versions: that of the PIV (the default one)
        export = data.get("export_folder")
        legacy = isinstance(export, str) and os.path.normpath(export) == POSTPROCESSING_FOLDER
        self.panel.export_folder.set(None if legacy else path(export))
        if "export_choice" in data:
            self.export_choice = export_choice(data["export_choice"])
        if "video_settings" in data:
            self.video_settings = video_settings_of(data["video_settings"])
        if "image_settings" in data:
            self.image_settings = image_settings_of(data["image_settings"])
        self.update_panel()

    def measurement_folder(self):
        """Folder of the measurement: the one of the vector fields shown, or of the project (images, or PIV results
        only: its folder, whatever the measurement folder its vector fields name)."""
        if self.post is not None and self.app.results_folder is None:
            return self.post.fields.folder
        return self.app.project_folder()

    def project_folder(self):
        return self.app.project_folder()

    def image_folder(self):
        """Folder of the images of the first page (None without images)."""
        source = self.app.source
        return source.image_folder() if source is not self else None

    def export_folder_changed(self):
        self.update_panel()
        self.app.save_project()

    # ------------------------------------------------------------------ vector fields

    def default_maps_folder(self):
        """The results of the project post-processed by default, whatever the engine chosen in the page Velocimetry: those
        of the default pipeline, else of the first pipeline it has (pipelines.chosen_results), else those of the
        results folder of the page Velocimetry (a folder of its own); without any, that folder (or the subfolder PIV of the
        project)."""
        piv_page = self.app.page("piv")
        output = piv_page.output_folder() if piv_page is not None else None
        root = self.app.project_folder()
        found = chosen_results(results_folders(root)) if root else None
        if found:
            return found
        if output and os.path.isfile(os.path.join(output, piv.VECTORS_FILE)):
            return output
        return output or (os.path.join(root, piv.PIV_FOLDER) if root else None)

    def any_results(self):
        """A project of the workspace has PIV/PTV results (or the project open, without workspace)."""
        ws = getattr(getattr(self.app, "workspace", None), "ws", None)
        if ws is not None and any(p.results for p in ws.projects):
            return True
        folder = self.maps_folder()
        return bool(folder) and os.path.isfile(os.path.join(folder, piv.VECTORS_FILE))

    def maps_folder(self):
        return self.panel.maps.get()

    def default_export_folder(self):
        """Default export folder: the subfolder of the pipeline of the vector fields in the subfolder PostProcessing of
        the folder of the measurement (PostProcessing/PaIRS, PostProcessing/TracTrac)."""
        root = self.measurement_folder()
        panel = getattr(self, "panel", None)
        maps = panel.maps.get() if panel is not None and hasattr(panel, "maps") else None
        return os.path.join(root, default_export_folder(maps)) if root else None

    def maps_folder_changed(self):
        """Another folder of vector fields: read when the page is shown (its default export folder shown; the
        results shown marked in the panel Projects)."""
        self.loaded = None
        self.panel.export_folder.refresh()
        if self.app.current_page() == self.key:
            self.load_maps()
        self.app.save_project()
        self.app.workspace.update_states()

    def load_maps(self):
        """Read and show the vector fields of the maps folder: vectors.npz, or DynamicStudio CSV exports (whose
        time between the images and scale are asked once)."""
        app = self.app
        folder = self.loaded = self.maps_folder()
        # The camera settings they are read with (camera_changed)
        self.loaded_camera = self.panel.camera_settings()
        self.post = None
        message = None
        csv = sorted(glob.glob(os.path.join(folder, "*.csv"))) if folder and os.path.isdir(folder) else []
        if not folder or not os.path.isdir(folder):
            message = f"The folder doesn't exist (yet): {folder}" if folder else None
        elif not os.path.isfile(os.path.join(folder, piv.VECTORS_FILE)) and not csv:
            message = f"No vector fields in this folder ({piv.VECTORS_FILE} or DynamicStudio CSV exports)"
        else:
            calibration = camera = None
            physical = self.panel.space.get() == "physical"
            if os.path.isfile(os.path.join(folder, piv.VECTORS_FILE)):
                # Without the camera settings in the sensor space: in pixels
                try:
                    camera = self.panel.read_camera() if physical else None
                except ValueError as e:
                    message = str(e)
            else:
                calibration = self.calibration.get(os.path.normcase(folder)) or self.ask_calibration()
                if calibration is None:
                    message = "DynamicStudio CSV exports: not read (time between the images and scale not given)"
                else:
                    self.calibration[os.path.normcase(folder)] = calibration
            if message is None:
                try:
                    with app.progress("Reading the vector fields") as wh:
                        wh(0, folder)
                        fields = (load_csv_folder(folder, *calibration, progress=wh,
                                                  frame_rate_hz=self.csv_frame_rate()) if calibration is not None
                                  else load_vectors(os.path.join(folder, piv.VECTORS_FILE), camera=camera,
                                                    pixels=not physical))
                except (OSError, ValueError, KeyError) as e:
                    message = f"The vector fields could not be read: {e}"
                else:
                    width, height = fields.settings.get("image_px") or self.image_size()
                    self.panel.update_camera_label(width, self.image_count(fields), height)
                    self.panel.rolling_range(fields.n)
                    if fields.settings.get("units") == "px" and physical:
                        message = (f"The vector fields of the PIV/PTV are in pixels ({fields.n} maps): enter the camera "
                                   "settings (side panel) to get them in mm and m/s, or choose the sensor space.")
                    else:
                        self.show_fields(fields)
                        return
        self.clear()
        self.update_panel(message)

    def show_fields(self, fields):
        """Show vector fields, with the recipe of the side panel."""
        try:
            settings = self.panel.read()
        except ValueError:
            settings = PostSettings()
        self.post = PostProcessing(fields, settings)
        self.index = self.map_of_image()
        self.update_panel()
        self.redraw()
        self.app.update_status(f"Vector fields: {fields.describe()} ({fields.source})")
        self.app.save_project()

    def piv_done(self, path, fields):
        """The PIV saved vector fields (in pixels): read with the camera settings if they are in the maps folder."""
        if same_path(os.path.dirname(path), self.maps_folder() or ""):
            self.loaded = None
            if self.app.current_page() == self.key:
                self.load_maps()

    def open_dialog(self):
        """Open a file vectors.npz, or a file of a folder of DynamicStudio CSV exports: its folder is the maps
        folder."""
        filename = filedialog.askopenfilename(
            parent=self.app.parent(), title="Open vector fields...",
            initialdir=self.maps_folder() or self.measurement_folder() or os.getcwd(),
            filetypes=[("Vector fields (vectors.npz, DynamicStudio CSV)", "*.npz *.csv"), ("All files", "*.*")])
        if not filename:
            return
        self.panel.maps.set(os.path.dirname(os.path.abspath(filename)))
        self.app.show_page(self.key)
        if self.loaded is None:
            self.load_maps()

    def image_size(self):
        """Width and height of the images of the first page (px), (None, None) without images."""
        paths = getattr(self.app.source, "images", None) if self.app.source is not self else None
        if not paths:
            return None, None
        try:
            with Image.open(paths[0]) as im:
                return im.size
        except OSError:
            return None, None

    def image_width(self):
        """Width of the images of the first page (px), None without images."""
        return self.image_size()[0]

    def image_count(self, fields=None):
        """Number of images of the set: those of the first page, or of the pairs of the vector fields (None if
        unknown)."""
        paths = getattr(self.app.source, "images", None) if self.app.source is not self else None
        if paths:
            return len(paths)
        fields = fields if fields is not None else (self.post.fields if self.post is not None else None)
        pairs = fields.settings.get("pairs") if fields is not None else None
        return max(b for _, b in pairs) + 1 if pairs else None

    def space_changed(self):
        """Another scale space: the vector fields are read again (in pixels, or with the camera settings)."""
        setpref("post_space", self.panel.space.get())
        self.panel.update_reference_state()
        self.loaded = None
        if self.app.current_page() == self.key:
            self.shown()
        self.app.save_project()

    def camera_changed(self):
        """Camera settings entered: the vector fields are read again with them. Nothing to do when they are those the
        vector fields were read with (the entry left without a change: reading them shows a progress window, which
        takes the focus from the entry, and its <FocusOut> would read them again, endlessly)."""
        if self.loaded is not None and self.panel.camera_settings() == self.loaded_camera:
            return
        setpref("post_camera", self.panel.camera_settings())
        self.loaded = None
        if self.app.current_page() == self.key:
            self.shown()
        else:
            self.panel.update_camera_label(*self.image_size()[:1], self.image_count(), self.image_size()[1])
        self.app.save_project()

    def ask_calibration(self):
        """Time between the images (s) and scale (mm/px) of DynamicStudio CSV exports, for the displacements in
        pixels (peak locking): those of the camera settings are proposed. None if cancelled."""
        app = self.app
        dt, scale = "", ""
        try:
            camera = self.panel.read_camera()
            if camera is not None:
                dt = num2str(round(camera.dt * 1000, 6))
                width = self.image_width()
                scale = num2str(round(camera.scale(width), 7)) if width else ""
        except ValueError:
            pass
        answ = app.inputdlg(["DynamicStudio CSV export: all the CSV files of the folder are read.\n\nTime between "
                             "the two images of a pair (ms), for the displacements in pixels (peak locking). "
                             "Empty: unknown.", "Scale (mm/px):"], "Vector fields", [dt, scale])
        if answ is None:
            return None
        try:
            return (float(answ[0]) / 1000 if answ[0].strip() else None,
                    float(answ[1]) if answ[1].strip() else None)
        except ValueError:
            app.msgbox("Enter numbers (or nothing) for the time between the images and the scale.", "Vector fields",
                       "warn")
            return None

    def csv_frame_rate(self):
        """The frame rate of the camera settings (the time between the maps of DynamicStudio CSV exports whose names
        don't give it), None if it isn't entered."""
        try:
            camera = self.panel.read_camera()
        except ValueError:
            return None
        return camera.frame_rate_hz if camera is not None else None

    def images_changed(self):
        """New training images: the vector fields of their maps folder are read when the page is shown."""
        self.panel.maps.refresh()
        if not same_path(self.loaded or "", self.maps_folder() or ""):
            self.loaded = None
            self.post = None
            self.clear()
        self.update_panel()

    def shown(self):
        """The page is shown: the vector fields of the maps folder are read if they aren't; the map of the image
        looked at in the other pages is shown."""
        self.panel.maps.refresh()
        self.panel.update_camera_label(*self.image_size()[:1], self.image_count(), self.image_size()[1])
        if self.loaded is None or not same_path(self.loaded, self.maps_folder() or ""):
            self.load_maps()
        elif self.post is not None and self.map_of_image() != self.index:
            self.index = self.map_of_image()
            self.redraw(instantaneous_only=True)
        self.update_panel()

    def map_images(self):
        """First image of the pair of each map, if the vector fields are the PIV results of the images (None
        otherwise)."""
        if self.post is None:
            return None
        pairs = self.post.fields.settings.get("pairs")
        pre = self.app.source
        if not pairs or len(pairs) != self.post.fields.n or not same_path(self.post.fields.folder,
                                                                           pre.image_folder() or ""):
            return None
        return [a for a, _ in pairs]

    def map_of_image(self):
        """Map of the pair starting with the image looked at in the other pages (the current one if unknown)."""
        firsts = self.map_images()
        if firsts is None:
            return min(self.index, self.post.fields.n - 1) if self.post is not None else 0
        return max(bisect.bisect_right(firsts, self.app.image_index) - 1, 0)

    def show_map(self, index):
        """Show the map index in the instantaneous figures; the other pages follow its first image."""
        if self.post is None or index == self.index:
            return
        self.index = index
        firsts = self.map_images()
        if firsts is not None:
            self.app.image_index = firsts[index]
        self.redraw(instantaneous_only=True)

    def export_folder(self):
        """Folder of the export: the subfolder PostProcessing of the folder of the measurement, or the folder
        chosen."""
        return self.panel.export_folder.get()

    # ------------------------------------------------------------------ recipe

    def apply_settings(self):
        """Recipe entered in the side panel: the figures are drawn again. Invalid settings are replaced by the
        current ones."""
        current = self.post.settings if self.post is not None else self.panel.saved()
        try:
            settings = self.panel.read()
        except ValueError as e:
            self.bell()
            self.app.update_status(f"Post-processing: {e}")
            self.panel.show(current)
            return
        if settings == current and self.post is not None:
            # A handle of the POD modes released without changing them: its mode shown
            handle, self.pod_handle = getattr(self, "pod_handle", None), None
            if handle is not None and self.post.pod_shown_mode() != self.post.pod_range()[handle]:
                self.post.pod_shown = self.post.pod_range()[handle]
                tiles = [t for t in self.tiles if {t.view, t.view2} & imaging.POD_FIGURES]
                if tiles:
                    self.redraw(tiles)
            return
        self.panel.save(settings)
        if self.post is not None:
            # Other POD modes only: the tiles of the POD drawn again, showing the mode of the handle moved
            only_pod = replace(settings, pod_mode=current.pod_mode, pod_mode_last=current.pod_mode_last,
                               pod_combine=current.pod_combine) == current
            self.post.settings = settings
            handle, self.pod_handle = getattr(self, "pod_handle", None), None
            if handle is not None:
                self.post.pod_shown = self.post.pod_range()[handle]
            tiles = [t for t in self.tiles if {t.view, t.view2} & imaging.POD_FIGURES] if only_pod else None
            if tiles is None or tiles:
                self.redraw(tiles)
        else:
            # The menus of the figures follow the add-ons turned on or off
            for tile in self.tiles:
                self.show_selectors(tile)
        self.update_panel()
        self.app.save_project()

    def update_panel(self, message=None):
        """Vector fields (or the message saying why there are none), recipe and button of the export."""
        panel = self.panel
        panel.maps.refresh()
        panel.export_folder.refresh()
        panel.export_label["text"] = describe_export(self.export_choice, self.post.fields.n if self.post else 1,
                                                     self.video_settings)
        panel.reference_label["text"] = self.describe_references()
        panel.show_hold(self.describe_hold())
        panel.rolling_label["text"] = self.describe_rolling()
        panel.show_pod(*self.describe_pod())
        # The export (its folder, quantities and button) greyed without vector fields; the batch without results in
        # any project
        state = tk.NORMAL if self.post is not None else tk.DISABLED
        panel.export_button.configure(state=state)
        panel.quantities_button.configure(state=state)
        panel.export_folder.configure_state(self.post is not None)
        panel.batch_button.configure(state=tk.NORMAL if self.any_results() else tk.DISABLED)
        if self.post is None:
            panel.maps_label["text"] = message or (
                "No vector fields: run the PIV/PTV (page Velocimetry), or choose a folder of vector fields."
                if self.loaded is None else panel.maps_label["text"])
            return
        panel.maps_label["text"] = describe_fields(self.post.fields)

    def describe_hold(self):
        """Invalid cells decay, as shown below its setting: the average delay of those filled."""
        post = self.post
        hold = None if post is None else post.hold_delay()
        if hold is None:
            return ""
        if not hold["invalid"]:
            return "No invalid cell in the flow"
        if hold["mean_delay"] is None:
            return "No invalid cell has a valid value so recent"
        return (f"Average delay of invalid cells {hold['mean_delay']:.1f} maps "
                f"({100 * hold['filled']:.0f} % invalid cells filled)")

    def describe_rolling(self, maps=None):
        """The interval of the rolling statistics (maps: the one set by default), as shown below it: its maps and
        their duration (in s and in flow times in the physical space)."""
        post = self.post
        if maps is None:
            maps = post.settings.rolling_maps if post is not None else None
            if not self.panel.rolling.get():
                return ("Off: the statistics of all the maps" + (f" ({post.fields.n})" if post is not None else ""))
        if post is None or maps is None:
            return ""
        n = post.fields.n
        text = f"{maps} of {n} " + ("pairs" if post.sensor() else "maps")
        if not post.sensor():
            with post.all_maps():
                text += f": {maps * post.map_interval():.4g} s"
                if post.uses_flow_time():
                    text += f" = {sig2(post.flow_times(maps))} t_flow"
        return text + (" (all of them)" if maps >= n else "")

    def describe_pod(self):
        """The POD of the images and the images found, as shown in the box Miscellaneous, and the number of modes
        evaluated (None: no POD)."""
        if self.post is None:
            return "The POD and the images of the vector fields: once they are read.", None
        pod = self.post.pod()
        text = self.post.pod_text()
        reasons = {self.post.image_missing(k) for k in ("raw_image", "piv_image")} - {None}
        text += "".join(f"\nImages: {reason}" for reason in sorted(reasons))
        return text, None if isinstance(pod, str) else pod["modes"]

    def describe_references(self):
        """The reference quantities used, as shown below them."""
        post = self.post
        if post is None:
            return ""
        if post.sensor():
            return "Sensor space: pixels and pairs, no reference quantity."
        s, u = post.settings, abs(post.statistics()["u_ref"])
        lines = [f"u_ref = {u:.3g} m/s" + ("" if s.use_u_ref and s.u_ref else " (maximum of the mean u)")]
        lines.append(f"t_flow = {post.flow_time() * 1000:.4g} ms" + (
            "" if s.use_t_flow and s.t_flow else f" (L_FoV / u_ref = {post.fov_length():.4g} mm / {u:.3g} m/s)"))
        return "; ".join(lines)

    def project_name(self):
        """Name at the start of the files exported: the project of the window for its measurement, the name of the
        measurement for the vector fields of another one."""
        folder = self.image_folder()
        fields = self.post.fields
        if folder and fields.folder and same_path(fields.folder, folder):
            return self.app.project_name()
        return fields.name or self.app.project_name()

    def ask_clim(self, tile):
        """Button "Colour range..." of a tile: the minimum and the maximum of the colour bar of its figure (empty:
        automatic), kept for this figure in this tile (and its export)."""
        app = self.app
        maps = PostProcessing.colour_maps(tile.fig) if self.post is not None else []
        if not maps:
            app.msgbox("This figure has no colour bar.", "Colour range", "info")
            return
        clim = ask_colour_range(app, FIGURE_TITLES[tile.view], maps[0].get_clim(), tile.clims.get(tile.view))
        if clim is None:
            return
        if clim == (None, None):
            self.clims.pop(tile.view, None)
        else:
            self.clims[tile.view] = clim
        self.redraw(self.tiles_showing(tile.view))
        app.save_project()

    def cmap_selected(self, tile, cmap):
        """A colour map chosen in the list of a tile (None: the one of the figure), kept for its figure (all the tiles
        showing it, and its export)."""
        if cmap is None:
            self.cmaps.pop(tile.view, None)
        else:
            self.cmaps[tile.view] = cmap
        self.redraw(self.tiles_showing(tile.view))
        tile.canvas.get_tk_widget().focus_set()
        self.app.save_project()


    def export(self):
        """Export the figures (and the key numbers and the recipe) to the subfolder PostProcessing of the image
        folder."""
        app = self.app
        if self.post is None:
            app.window.bell()
            return
        folder = self.export_folder()
        # The results of a previous export in the folder: overwritten (all of them), or nothing exported
        previous = previous_results(folder, self.project_name())
        if not overwrite_question(app, [(folder, previous)]):
            app.update_status(f"Not exported: the previous results in {folder} kept")
            return
        # The figures, the maps and the videos of the quantities (Quantities...), the parameters of the tiles
        options = export_options(self.export_spec(), self.post.fields.n)
        try:
            remove_results(previous)
            with app.progress("Export the figures") as wh:
                files = self.post.export(folder, export_index(self.export_choice),
                                         name=self.project_name(), progress=wh, **options)
        except OSError as e:
            app.status_error(f"The figures could not be exported in {folder}: {e}")
            return
        details = (f" (figures, key numbers, recipe: {os.path.basename(files[0])} ...). Recipe: "
                   + " → ".join(self.post.settings.steps()))
        app.update_status(f"{len(files)} files exported in {folder}" + details)
        app.log_step("postprocessing", f"{len(files)} files exported in {app.project_path(folder)}" + details)

    def export_video(self, tile):
        """Dialog of the video of what a tile shows over the maps (VideoDialog), shown without waiting. Returns it
        (None if the tile doesn't follow the maps)."""
        if self.post is None or not self.follows_maps(tile):
            self.app.window.bell()
            return None
        dialog = VideoDialog(self.app.parent(), self, tile)
        dialog.show()
        return dialog

    def video_frames(self, tile):
        """What a tile shows, as it is now, for its video: draw(fig, k) draws it for the map k (its instantaneous
        figure, or its comparison with the wiper where it is, with their colour ranges, colour maps and isocontour);
        and the file of the video in the export folder."""
        post, name, keys, fraction = self.post, self.project_name(), (tile.view, tile.view2), tile.fraction
        clims, cmaps = tuple(tile.clims.get(k) for k in keys), tuple(tile.cmaps.get(k) for k in keys)
        labels = (self.symbol(tile.view), self.symbol(tile.view2)) if tile.view2 is not None else None
        contour = tile.contours.get(tile.view)

        def draw(fig, k):
            if keys[1] is None:
                post.plot(keys[0], fig, k, name, clims[0], cmaps[0], contour)
            else:
                _, layers, _ = post.plot_comparison(keys[0], keys[1], fig, k, name, clims, cmaps, contour)
                if layers is not None:
                    Wiper(*layers, labels=labels).place(fraction)
        return draw, post.video_path(self.export_folder(), keys, name)

    def figure_params(self):
        """The parameters of the figures by quantity, as saved (figure_params_of)."""
        return {"clims": {k: list(v) for k, v in self.clims.items()}, "cmaps": dict(self.cmaps),
                "contours": dict(self.contours), "compares": dict(self.compares), "wipers": dict(self.wipers)}

    def set_figure_params(self, params):
        """The parameters of the figures by quantity (figure_params_of) in the tiles (the dictionaries they share
        changed in place)."""
        for name in ("clims", "cmaps", "contours", "compares", "wipers"):
            getattr(self, name).clear()
            getattr(self, name).update(params[name])
        for tile in self.tiles:
            self.follow_quantity(tile)
            self.show_selectors(tile)

    def follow_quantity(self, tile):
        """A tile takes the second quantity and the position of the wiper of its quantity."""
        tile.view2 = self.compares.get(tile.view) if tile.view in MAP_FIGURES else None
        tile.fraction = self.wipers.get(tile.view, COMPARE_FRACTION)
        tile.set_compare_value(tile.fraction)

    def tiles_showing(self, key):
        """The tiles that show a quantity (first or second one)."""
        return [t for t in self.tiles if key in (t.view, t.view2)]

    def export_spec(self):
        """What the export does (quantities.export_spec): the figures and videos chosen, the parameters of the
        figures by quantity and the settings of the videos of the quantities."""
        return {"choice": dict(self.export_choice), "params": self.figure_params(),
                "videos": {k: dict(v) for k, v in self.video_settings.items()},
                "images": {k: dict(v) for k, v in self.image_settings.items()}}

    def set_export_spec(self, spec):
        """An export specification edited (Quantities...) in the page: kept in the preferences and the project."""
        self.export_choice = export_choice(spec["choice"])
        self.video_settings = video_settings_of(spec["videos"])
        self.image_settings = image_settings_of(spec.get("images"))
        self.set_figure_params(figure_params_of(spec["params"]))
        setpref("post_export", self.export_choice)
        setpref("post_videos", self.video_settings)
        setpref("post_images", self.image_settings)
        self.redraw()
        self.update_panel()
        self.app.save_project()

    def choose_quantities(self):
        """Run step > Quantities...: the quantities exported and their parameters (QuantitiesDialog)."""
        from .quantities import QuantitiesDialog
        from ..settings import export_spec

        spec = QuantitiesDialog(self.app.parent(), export_spec(self.export_spec()), rolling=self.panel.rolling.get(),
                                post=self.post, jet=self.jet_theory_on()).run()
        if spec is not None:
            self.set_export_spec(spec)

    def batch_settings(self):
        """The settings of the post-processing of the interface for a batch: the recipe, the scale space, the camera,
        the export folder (relative to the project) and the export specification."""
        try:
            settings = self.panel.read()
        except ValueError:
            settings = self.post.settings if self.post is not None else PostSettings()
        try:
            camera = self.panel.read_camera()
        except ValueError:
            camera = None
        root = self.app.project_folder()
        folder = self.export_folder()
        return {"settings": settings_record(settings), "space": self.panel.space.get(),
                "camera": None if camera is None else camera_record(camera),
                "export_folder": os.path.relpath(folder, root) if root and folder else None,
                "export": self.export_spec()}

    def batch_step(self, title="Post-processing", job=None, options=None):
        from .quantities import QuantitiesEditor
        from ..settings import export_spec

        page = self
        # The PTV and the figures of the images and of the POD of the projects, from the settings of their vector
        # fields (not read), for the life of the window: {(vectors.npz, its date, POD mode): (PTV, {key: reason},
        # reason without POD, modes of the POD evaluated)}
        states = {}

        def dataset_state(maps, settings):
            """(PTV, why each figure of the images and of the POD can't be exported ({key: reason, None if it
            can}), why there is no POD (None: there is one), the modes of the POD evaluated (None: none)) of the
            vector fields of a folder with the batch settings of their project; None if they are unknown."""
            from ..batch_run import post_inputs

            try:
                _, _, recipe = post_inputs(os.path.dirname(maps), {**settings, "maps_folder": maps})
                vectors = os.path.join(maps, piv.VECTORS_FILE)
                signature = (os.path.normcase(os.path.abspath(vectors)), os.path.getmtime(vectors), recipe.pod_mode)
            except (OSError, ValueError, KeyError, TypeError):
                return None
            if signature not in states:
                # The POD mode 1: whether there is a POD at all (any mode evaluated)
                states[signature] = (ptv_results(maps), imaging.availability(vectors, recipe.pod_mode),
                                     imaging.availability(vectors, 1)["pod_mode"], imaging.pod_modes(vectors))
            return states[signature]

        def project_state(project, settings):
            """The states of the results sets exported of a project with its batch settings (dataset_state), [] if
            unknown."""
            if project is None or not project.results:
                return []
            return [s for s in (dataset_state(maps, settings)
                                for maps in batch_maps_folders(project, settings).values()) if s is not None]

        def node_state(items):
            """The projects of a node ([(project, settings)]), over their results sets exported: (one is a PIV, one
            is a PTV (None: unknown), {key: reason} of the figures of the images and of the POD that none of them can
            export, why none has a POD (None: one has), the most modes of their POD evaluated (None: none))."""
            known = [s for p, settings in items for s in project_state(p, settings)]
            if not known:
                return None, None, {}, None, None

            def none_of(reasons):
                if not all(reasons):
                    return None
                return reasons[0] if len(reasons) == 1 else f"none of the {len(reasons)} results: {reasons[0]}"
            missing = {key: none_of([s[1][key] for s in known]) for key in imaging.FIGURE_FILES}
            return (any(not s[0] for s in known), any(s[0] for s in known), {k: r for k, r in missing.items() if r},
                    none_of([s[2] for s in known]), max((s[3] for s in known if s[3]), default=None))

        def unavailable(items):
            """The POD modes greyed out when none of the projects of the node has a POD."""
            reason = node_state(items)[3]
            return {("settings", "pod_mode"): reason, ("settings", "pod_combine"): reason} if reason else None

        def limits(items):
            """The range of the POD modes: up to the most modes evaluated by the projects of the node."""
            modes = node_state(items)[4]
            return {("settings", "pod_mode"): (1, modes)} if modes else None

        class Quantities(QuantitiesEditor):
            """The quantities of the node selected (its export specification), in a narrow column of the window Batch
            (the parameters and the global settings below the quantities, in tabs): the videos of every figure with
            rolling statistics, the maps of the videos and of the export described with the vector fields of its
            project (context); compared with the quantities of the project open (the reference of all the nodes)."""

            def __init__(self, master, on_change):
                super().__init__(master, export_spec(None), on_change=on_change, height=12, stacked=True)
                self.loaded = None  # (key of the vector fields and settings, PostProcessing)
                self.node = None  # (PIV, PTV, {key: reason}, jet theory) of all the projects of the node selected

            def context(self, project, settings, items=None):
                """The post-processing of a project with its settings, for the maps of the videos (their number;
                the flow time captured: the vector fields read when a quantity with its video needs it, those of
                the previous one kept if the same); the quantities greyed out that none of the projects of the node
                (items: [(project, settings)]; the project alone by default) can export: those of the PTV, of the
                images and of the POD, those of the jet theory when the add-on is off for all of them. The
                reference: the quantities of the project open (the default export for the project open itself)."""
                items = items or [(project, settings)]
                dialog = self.winfo_toplevel()
                mine = [p for p, _ in items if p is not None]
                if hasattr(dialog, "reference") and mine and not all(dialog.is_open(p) for p in mine):
                    self.set_reference(dialog.reference().get("export"))
                else:
                    self.set_reference(None)
                piv_, ptv, missing, _, _ = node_state(items)
                jet = any(((s or {}).get("settings") or {}).get("jet_theory") for _, s in items)
                self.node = (piv_, ptv, missing, jet)
                self.set_methods(piv_, ptv)
                self.set_imaging(missing)
                self.set_jet(jet)
                if project is None or not project.results:
                    return self.set_post(None)
                from ..batch_run import post_inputs

                try:
                    maps, camera, recipe = post_inputs(project.folder,
                                                       {**settings, "maps_folder": batch_maps_folder(project, settings)})
                    vectors = os.path.join(maps, piv.VECTORS_FILE)
                    n = count_maps(vectors)
                except (OSError, ValueError, KeyError, TypeError):
                    return self.set_post(None)
                signature = json.dumps([maps, None if camera is None else asdict(camera), settings.get("space"),
                                        asdict(recipe)], default=str)
                if self.loaded and self.loaded[0] == signature:
                    return self.set_post(self.loaded[1])

                def load():
                    self.winfo_toplevel().configure(cursor="watch")
                    self.update_idletasks()
                    try:
                        post = PostProcessing(load_vectors(vectors, camera=camera, pixels=camera is None), recipe)
                    except (OSError, ValueError, KeyError):
                        return None
                    finally:
                        self.winfo_toplevel().configure(cursor="")
                    self.loaded = (signature, post)
                    return post
                self.set_loader(load, n)

            def set_post(self, post):
                """The vector fields of the first project (the maps of the sliders), the quantities greyed out
                those of all the projects of the node."""
                super().set_post(post)
                if self.node is not None:
                    self.piv, self.ptv, self.jet = self.node[0], self.node[1], self.node[3]
                    self.missing = self.reasons(self.node[2])
                    self.refresh_tags()

            def show(self, value, settings=None, enabled=True):
                rolling = bool(((settings or {}).get("settings") or {}).get("rolling")) if settings else None
                # The POD modes: a picture each, or their sum, a figure of each map
                recipe = (settings or {}).get("settings") or {}
                first = int(recipe.get("pod_mode") or 1)
                last = max(int(recipe.get("pod_mode_last") or first), first)
                self.pod_images, self.pod_per_map = last - first + 1, bool(recipe.get("pod_combine"))
                super().show(export_spec(value), rolling, enabled)

        def confirm(dialog, entries, options):
            """The results of previous exports of the projects run (and of their stitchings): overwritten (all of
            them, asked once), else not run (None)."""
            from ..batch_run import post_datasets, post_export_folder, project_name

            conflicts, references = [], {}
            for entry in entries:
                s = entry["settings"]["postprocessing"]
                try:
                    datasets = post_datasets(entry["folder"], s)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                for label, maps in datasets.items():
                    folder = post_export_folder(entry["folder"], s, maps)
                    references[(os.path.normcase(os.path.normpath(entry["folder"])), label)] = folder
                    conflicts.append((folder, previous_results(folder, project_name(entry["folder"]))))
            for unit in options.get("units") or []:
                stitching = unit.get("stitching")
                if stitching:
                    folder = references.get((os.path.normcase(os.path.normpath(stitching["members"][0]["folder"])),
                                             stitching.get("pipeline") or ""))
                    if folder:
                        stitched = os.path.join(folder, "Stitched")
                        conflicts.append((stitched, previous_results(stitched, stitching["setup"], "Stitched",
                                                                     records=False)))
            if not any(paths for _, paths in conflicts):
                return {}
            return {"overwrite": True} if overwrite_question(page.app, conflicts, "the run") else None

        def job_settings(project, settings, resolve):
            if settings.get("space") == "physical" and not settings.get("camera"):
                raise ValueError(f"{project.label()}: the physical space needs the camera settings (field of view "
                                 "width, frame rate) of the side panel")
            folders = batch_maps_folders(project, settings)
            if not folders:
                raise ValueError(f"{project.label()}: none of the results sets checked "
                                 f"({', '.join(settings.get('results') or []) or 'none'})")
            return {"postprocessing": {**settings, "maps_folders": folders,
                                       "maps_folder": next(iter(folders.values()))}}
        return BatchStep("postprocessing", title, lambda project: project.results, self.batch_settings,
                         describe_post, job_settings, job=job, options=options, fields=post_fields,
                         validate=validate_post, to_project=("postprocessing", post_to_project),
                         from_project=post_from_project, unavailable=unavailable, limits=limits, form=BatchPostForm,
                         confirm=confirm,
                         panels=(("Quantities Export for this node", "export", Quantities),))

    def batch(self):
        """Run step > Batch...: the post-processing of projects of the workspace, with the stitchings of fields of
        view (stitching.py)."""
        from .archive_tools import script
        from .stitching import StitchOptions

        options = StitchOptions(script("post_archive"), self.app.workspace.ws)
        run_batch(self.app, self.batch_step(options=options))

    def video_settings_of(self, key):
        """The settings of the video of a main quantity: its own (saved when one is written), else every N maps,
        frames per second and format of the last video, over all the maps."""
        if key in self.video_settings:
            return dict(self.video_settings[key])
        return dict(video_settings(getpref("post_video")), first=1, last=None)

    def keep_video_settings(self, key, settings):
        """The settings of the video of a main quantity kept (preferences and project), and every N maps, frames per
        second and format as those of the next new quantity."""
        self.video_settings[key] = video_settings(settings)
        setpref("post_videos", self.video_settings)
        setpref("post_video", {k: settings[k] for k in ("step", "fps", "format")})
        self.panel.export_label["text"] = describe_export(self.export_choice, self.post.fields.n, self.video_settings)
        self.app.save_project()

    def video_written(self, tile, path, frames, planned, first, last, step, fps):
        """A video written (frames of the planned ones: fewer if it was stopped), in the status and the history."""
        what = FIGURE_TITLES[tile.view] + (f" vs {FIGURE_TITLES[tile.view2]}" if tile.view2 is not None else "")
        stopped = f", stopped after {frames} of {planned} frames" if frames < planned else ""
        text = (f"Video of {what}: maps {first} to {last} every {step} ({frames} frames{stopped}) at "
                f"{num2str(fps)} frames/s in ")
        self.app.update_status(text + path)
        self.app.log_step("postprocessing", text + self.app.project_path(path))

    # ------------------------------------------------------------------ tiles

    def clear(self):
        for tile in self.tiles:
            self.show_selectors(tile)
            tile.wiper = None
            tile.message(f"{FIGURE_TITLES[tile.view]}\n\nRun the PIV/PTV (page Velocimetry) or open vector fields")
            tile.set_slider(1, 0)
            tile.contour_sets = []
            self.show_contour(tile)

    def symbol(self, key):
        """Symbol of the quantity of a map (LaTeX, with $; relabelled in the sensor space when maps are loaded)."""
        return self.post.symbol(key) if self.post is not None else f"${FIGURE_SYMBOLS[key]}$"

    def jet_theory_on(self):
        """The jet theory add-on is on: in the recipe of the vector fields, else in the side panel."""
        if self.post is not None:
            return self.post.settings.jet_theory
        jet = getattr(getattr(self, "panel", None), "jet_theory", None)
        return bool(jet.get()) if jet is not None else self.panel.saved().jet_theory

    def show_selectors(self, tile):
        """The quantities of a tile in its menus, the maps it can be compared with in the second one, the wiper
        shown when there is a second quantity (its slider named by the symbols of the two quantities), and the
        button of its video when it follows the maps."""
        compared = tile.view2 is not None and tile.view in MAP_FIGURES
        # The quantities of the PTV greyed out for vector fields that aren't those of a PTV, those of the quality of a
        # PIV for those of a PTV, those of the images and of their POD when they aren't found, those of the jet theory
        # when the add-on is off
        disabled = set() if self.jet_theory_on() else set(jet.JET_GROUP[1])
        if self.post is not None:
            disabled |= set(PIV_GROUP[1]) if self.post.ptv() else set(lagrangian.GROUP[1])
            # Each image on its own (the raw images may be missing, those pre-processed kept)
            disabled |= {k for k in imaging.IMAGE_FIGURES if self.post.image_missing(k)}
            if isinstance(self.post.pod(), str):
                disabled |= imaging.POD_FIGURES
        tile.selector.show(tile.view, disabled=disabled)
        tile.selector2.prefix = "vs " if compared else ""
        maps = None if self.post is None else [NO_COMPARISON] + [k for k in MAP_FIGURES if self.post.map_available(k)]
        tile.selector2.show(tile.view2 or NO_COMPARISON, maps)
        tile.show_compare(tile.view2 is not None)
        video = self.post is not None and self.follows_maps(tile)
        if video and not tile.video_button.winfo_manager():
            tile.video_button.pack(side=tk.LEFT, padx=(0, 6), before=tile.toolbar)
        elif not video:
            tile.video_button.pack_forget()
        if compared:
            tile.set_compare_labels(*(plain_symbol(self.symbol(k)) for k in (tile.view, tile.view2)))

    def view_selected(self, tile, key):
        """A figure chosen in the menu of a tile: with the second quantity and the wiper of this figure."""
        tile.view = key
        self.follow_quantity(tile)
        setpref("post_views", [t.view for t in self.tiles])
        if self.post is None:
            self.show_selectors(tile)
            tile.message(f"{FIGURE_TITLES[key]}\n\nRun the PIV/PTV (page Velocimetry) or open vector fields")
        self.redraw([tile])
        tile.canvas.get_tk_widget().focus_set()
        self.app.save_project()

    def view2_selected(self, tile, key):
        """A second quantity chosen in the second menu of a tile (NO_COMPARISON: none): compared with the first one
        with a wiper, in all the tiles showing the first one."""
        view2 = None if key == NO_COMPARISON else key
        if view2 is None:
            self.compares.pop(tile.view, None)
        else:
            self.compares[tile.view] = view2
        tiles = [t for t in self.tiles if t.view == tile.view]
        for t in tiles:
            t.view2 = view2
            self.show_selectors(t)
        self.redraw(tiles)
        tile.canvas.get_tk_widget().focus_set()
        self.app.save_project()

    def wipe(self, tile, fraction):
        """The wiper of a tile moved: the first map left of it, the second right of it (in all the tiles showing its
        quantity)."""
        self.wipers[tile.view] = fraction
        for t in self.tiles:
            if t.view == tile.view:
                t.fraction = fraction
                if t is not tile:
                    t.set_compare_value(fraction)
                if t.wiper is not None:
                    t.wiper.place(fraction)
                    t.canvas.draw_idle()

    def follows_maps(self, tile):
        """The tile shows a figure of each map (first or second quantity: an instantaneous figure, or any figure
        with rolling statistics): it follows the map browsed."""
        def per_map(key):
            return key is not None and (self.post.instantaneous(key) if self.post is not None
                                        else key in INSTANTANEOUS)
        return per_map(tile.view) or per_map(tile.view2)

    def redraw(self, tiles=None, instantaneous_only=False):
        """Draw the figures of the tiles (the instantaneous ones of the map self.index)."""
        if self.post is None:
            return
        app = self.app
        app.window.configure(cursor="watch")
        app.window.update_idletasks()
        try:
            for tile in tiles or self.tiles:
                if instantaneous_only and not self.follows_maps(tile):
                    continue
                self.show_selectors(tile)
                tile.wiper = None
                if tile.view2 is None:
                    self.post.plot(tile.view, tile.fig, self.index, clim=tile.clims.get(tile.view),
                                   cmap=tile.cmaps.get(tile.view))
                else:
                    # The second map over the first, each with the colour range and the colour map chosen for its
                    # figure in the tile
                    _, layers, reason = self.post.plot_comparison(
                        tile.view, tile.view2, tile.fig, self.index,
                        clims=(tile.clims.get(tile.view), tile.clims.get(tile.view2)),
                        cmaps=(tile.cmaps.get(tile.view), tile.cmaps.get(tile.view2)))
                    tile.compare_note(f"No comparison: {reason}" if reason else None)
                    if layers is not None:
                        tile.wiper = Wiper(*layers, labels=(self.symbol(tile.view), self.symbol(tile.view2)))
                        tile.wiper.place(tile.fraction)
                tile.cmap_box.show(tile.cmaps.get(tile.view), bool(PostProcessing.colour_maps(tile.fig)))
                tile.contour_sets = []  # Cleared with the figure
                self.show_contour(tile)
                tile.ax = tile.im = tile.cbar = None
                if self.follows_maps(tile):
                    tile.set_slider(self.index + 1, self.post.fields.n)
                else:
                    tile.set_slider(1, 0)
                tile.canvas.draw_idle()
            self.set_active_tile(self.active_tile)
        finally:
            app.window.configure(cursor="")

    @staticmethod
    def main_map(tile):
        """The map of the first quantity of a tile (its field as drawn), None if the figure has none."""
        return next((a for a in PostProcessing.colour_maps(tile.fig) if getattr(a, "field", None) is not None), None)

    def contour_toggled(self, tile):
        """The isocontour of a tile toggled: at the middle of the colour range of its map at first."""
        if tile.contour_var.get():
            m = self.main_map(tile)
            if m is None:
                tile.contour_var.set(False)
                return
            low, high = m.get_clim()
            self.contours[tile.view] = (low + high) / 2
        else:
            self.contours.pop(tile.view, None)
        for t in self.tiles:
            if t.view == tile.view:
                self.show_contour(t)
        self.app.save_project()

    def contour_slid(self, tile, fraction):
        """The slider of the isocontour of a tile moved: its value, in the colour range of the map."""
        m = self.main_map(tile)
        if self.placing_contour or m is None or tile.view not in tile.contours:
            return
        low, high = m.get_clim()
        self.contours[tile.view] = low + fraction * (high - low)
        for t in self.tiles:
            if t.view == tile.view:
                if t is not tile:
                    self.show_contour(t)
                else:
                    self.draw_contour(t)

    def show_contour(self, tile):
        """The toggle of the isocontour of a tile (enabled for the figures with a map), and its slider below the
        figure when it is on, at its value; the isocontour drawn."""
        m = self.main_map(tile) if self.post is not None else None
        tile.contour_check.state(["!disabled"] if m is not None else ["disabled"])
        on = m is not None and tile.view in tile.contours
        tile.contour_var.set(on)
        if on:
            if not tile.contour_bar.winfo_manager():
                tile.contour_bar.pack(side=tk.BOTTOM, fill=tk.X, before=tile.canvas.get_tk_widget())
            low, high = m.get_clim()
            self.placing_contour = True
            try:
                tile.contour_slider.set(min(max((tile.contours[tile.view] - low) / (high - low), 0.0), 1.0)
                                        if high > low else 0.5)
            finally:
                self.placing_contour = False
        else:
            tile.contour_bar.pack_forget()
        self.draw_contour(tile)

    def draw_contour(self, tile):
        """The isocontour of a tile drawn again at its value (none if off), on the side of its map of the wiper."""
        for cs in tile.contour_sets:
            if tile.wiper is not None and cs in tile.wiper.layers[0]:
                tile.wiper.layers[0].remove(cs)
            try:
                cs.remove()
            except (ValueError, AttributeError):
                pass
        tile.contour_sets = []
        level = tile.contours.get(tile.view)
        if level is not None and self.post is not None:
            tile.contour_sets = self.post.contour_maps(
                tile.fig, level, symbol=self.symbol(tile.view) if tile.view in FIGURE_SYMBOLS else None)
            if tile.wiper is not None:
                tile.wiper.layers[0].extend(tile.contour_sets)
                tile.wiper.place(tile.fraction)
        tile.contour_value["text"] = "" if level is None else f"{level:.4g}"
        tile.canvas.draw_idle()

    def set_active_tile(self, tile):
        """Tile under the mouse: the arrow keys browse the maps, which the tiles of the instantaneous figures
        follow."""
        self.active_tile = tile
        for t in self.tiles:
            t.set_active(True if t is tile else "linked" if self.follows_maps(t) else False)

    def slide_to(self, i):
        self.show_map(i - 1)

    def arrow(self, step):
        if self.post is not None:
            self.show_map(min(max(self.index + step, 0), self.post.fields.n - 1))


class VideoDialog(Dialog):
    """Export of the video of what a tile shows over the maps (PostProcessingPage.video_frames): every N maps from
    the first to the last one chosen with the two-sided slider, at a frame rate. Run starts it and Cancel closes the
    dialog; while the video is written, frame by frame between the events of the window (which stays responsive),
    Run is greyed out and Stop ends the file at the frame written: a complete, shorter video. The dialog stays open
    at the end, to export another one. Some frames are drawn first for the scales of all of them (Video.measure).
    The settings are those of the main quantity of the tile (PostProcessingPage.video_settings_of), kept when a
    video is run: the same in any tile that shows it."""

    def __init__(self, parent, page, tile):
        super().__init__(parent, "Export a video")
        self.page, self.tile, self.n = page, tile, page.post.fields.n
        self.video, self.stopping, self.closing = None, False, False
        saved = page.video_settings_of(tile.view)
        frame = self.frame
        what = FIGURE_TITLES[tile.view] + (f" vs {FIGURE_TITLES[tile.view2]}" if tile.view2 is not None else "")
        ttk.Label(frame, text=f"Video of {what}, as the tile shows it", wraplength=self.wrap,
                  justify=tk.LEFT).pack(anchor=tk.W)
        grid = ttk.Frame(frame)
        grid.pack(fill=tk.X, pady=(8, 0))
        self.step = tk.StringVar(self, str(saved["step"]))
        self.fps = tk.StringVar(self, num2str(saved["fps"]))
        self.format = tk.StringVar(self, saved["format"])
        self.entries = []
        for r, (label, var) in enumerate((("Every N maps (1: all the maps)", self.step),
                                          ("Frames per second", self.fps))):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky=tk.W, pady=1)
            entry = ttk.Entry(grid, textvariable=var, width=9)
            entry.grid(row=r, column=1, sticky=tk.E, pady=1)
            self.entries.append(entry)
            var.trace_add("write", lambda *args: self.update_summary())
        ttk.Label(grid, text="Format").grid(row=2, column=0, sticky=tk.W, pady=1)
        box = format_box(grid, self.format)
        box.grid(row=2, column=1, sticky=tk.E, pady=1)
        self.entries.append(box)
        grid.columnconfigure(0, weight=1)
        ttk.Label(frame, text="First and last maps").pack(anchor=tk.W, pady=(8, 0))
        self.range = RangeSlider(frame, 1, self.n, command=lambda first, last: self.update_summary())
        self.range.pack(fill=tk.X)
        self.summary = ttk.Label(frame, foreground="#555555", wraplength=self.wrap, justify=tk.LEFT)
        self.summary.pack(anchor=tk.W, pady=(2, 0))
        self.range.set(saved["first"], saved["last"] or self.n)
        self.progress = ttk.Progressbar(frame, maximum=1.0)
        self.progress.pack(fill=tk.X, pady=(10, 0))
        self.status = ttk.Label(frame, wraplength=self.wrap, justify=tk.LEFT)
        self.status.pack(anchor=tk.W, pady=(2, 0))
        buttons = ttk.Frame(frame)
        buttons.pack(anchor=tk.E, pady=(12, 0))
        self.run_button = ttk.Button(buttons, text="Run", command=self.ok)
        self.run_button.pack(side=tk.LEFT, padx=(0, 6))
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self.close)
        self.cancel_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop)
        self.bind("<Escape>", lambda event: self.close())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.initial_focus = self.run_button
        self.update_summary()

    def settings(self):
        """Every N maps, frames per second and the maps of the video (indices): ValueError with a message."""
        step, fps = read_step_fps(self.step.get(), self.fps.get())
        first, last = self.range.get()
        return step, fps, first, last, range(first - 1, last, step)

    def update_summary(self):
        try:
            step, fps, first, last, indices = self.settings()
        except ValueError as e:
            self.summary["text"] = str(e)
            return
        self.summary["text"] = video_summary(self.page.post, first, last, step, fps)

    def ok(self):
        """Run: write the video."""
        if self.video is not None:
            return
        try:
            step, fps, first, last, indices = self.settings()
        except ValueError as e:
            self.status["text"] = str(e)
            self.bell()
            return
        # Up to the last map: up to the last one of any measurement
        self.page.keep_video_settings(self.tile.view, {"step": step, "fps": fps, "format": self.format.get(),
                                                       "first": first, "last": None if last == self.n else last})
        self.draw, path = self.page.video_frames(self.tile)
        # The video of a previous export: overwritten, or not written
        existing = [p for p in (os.path.splitext(path)[0] + "." + fmt for fmt in VIDEO_FORMATS) if os.path.isfile(p)]
        if existing and not overwrite_question(self.page.app, [(os.path.dirname(path), existing)], "the video"):
            self.status["text"] = "Not written: the previous video kept"
            return
        for p in existing:
            os.remove(p)
        self.video = Video(path, fps, fmt=self.format.get())
        self.run_settings = (first, last, step, fps)
        self.indices, self.done, self.stopping = list(indices), 0, False
        self.measured = scale_frames(self.indices)
        self.run_button.state(["disabled"])
        for entry in self.entries:
            entry.state(["disabled"])
        self.range.set_enabled(False)
        self.cancel_button.pack_forget()
        self.stop_button.state(["!disabled"])
        self.stop_button.pack(side=tk.LEFT)
        self.progress["value"] = 0
        self.status["text"] = f"Writing {os.path.basename(self.video.path)}..."
        self.after(1, self.next_frame)

    def next_frame(self):
        if self.video is None:
            return
        if self.stopping or self.done == len(self.indices):
            return self.finish()
        # The scales of the frames first, then the frames
        if self.measured:
            k = self.measured.pop(0)
            try:
                self.video.measure(self.draw, k)
            except Exception as e:
                return self.finish(f"The video could not be written: {e}")
            self.status["text"] = f"Scales of the frames: map {k + 1} ({len(self.measured)} more)"
            return self.after(1, self.next_frame)
        k = self.indices[self.done]
        try:
            self.video.add(self.draw, k)
        except Exception as e:  # The frames written are kept: the video ends there
            return self.finish(f"The video could not be written: {e}")
        self.done += 1
        self.progress["value"] = self.done / len(self.indices)
        self.status["text"] = f"Frame {self.done} of {len(self.indices)} (map {k + 1})"
        self.after(1, self.next_frame)

    def stop(self):
        """Stop: the video ends at the frame written."""
        self.stopping = True
        self.stop_button.state(["disabled"])
        self.status["text"] = "Stopping: the video ends at the frame written..."

    def finish(self, error=None):
        """End the file (the frames written: a complete video, shorter if stopped) and get ready for another one."""
        video, self.video = self.video, None
        frames = video.frames
        try:
            path = video.close()
        except (OSError, RuntimeError, ValueError) as e:
            path, error = None, error or f"The video could not be finished: {e}"
        first, last, step, fps = self.run_settings
        if path is not None:
            self.page.video_written(self.tile, path, frames, len(self.indices), first, last, step, fps)
            self.status["text"] = (f"Saved: {path} ({frames} frames" + (f" of {len(self.indices)}, stopped)"
                                                                         if frames < len(self.indices) else ")"))
        else:
            self.status["text"] = "No frame written: no video"
        if error:
            self.status["text"] = error + (f". {self.status['text']}" if path else "")
            self.page.app.update_status(error)
        self.stop_button.pack_forget()
        self.cancel_button.configure(text="Close")
        self.cancel_button.pack(side=tk.LEFT)
        self.run_button.state(["!disabled"])
        for entry in self.entries:
            entry.state(["!disabled"])
        self.range.set_enabled(True)
        if self.closing:
            self.destroy()

    def close(self):
        """Cancel / Close, Escape or the window closed: while the video is written, it is stopped (ended at the
        frame written) and the dialog closed then."""
        if self.video is not None:
            self.closing = True
            self.stop()
        else:
            self.destroy()


class ExportDialog(Dialog):
    """Choice of the figures exported (Export box): each figure (its menu group), the video of the instantaneous
    ones; the instantaneous figures of the map shown, or of the maps from the first to the last one every N maps
    (two-sided slider; each figure in its own folder), the same maps in their videos, at a frame rate. These video
    settings are those of the quantities without their own (page.video_settings, saved from their Video button).
    The result is the choice (DEFAULT_EXPORT), None if cancelled."""

    COLUMNS = 3

    def __init__(self, parent, page, choice):
        super().__init__(parent, "Figures to export")
        self.n = page.post.fields.n if page.post is not None else max(int(choice["last"] or 1), 1)
        # The quantities with video settings of their own (none without them: the archive interface)
        self.own = getattr(page, "video_settings", None)
        frame = self.frame
        groups = ttk.Frame(frame)
        groups.pack(fill=tk.X)
        chosen = set(FIGURE_TITLES) if choice["figures"] is None else set(choice["figures"])
        self.images, self.videos = {}, {}
        # The groups of the menus of the tiles (those of a submenu named after it: Add-ons › Jet theory), the jet
        # theory greyed out with its add-on off
        jet_off = page.post is not None and not page.post.settings.jet_theory
        for i, (group, keys) in enumerate(FIGURE_GROUPS):
            text = f"{MENU_PARENTS[group]} › {group}" if group in MENU_PARENTS else group or "Other"
            box = ttk.LabelFrame(groups, text=text, padding=6)
            box.grid(row=i // self.COLUMNS, column=i % self.COLUMNS, sticky="nsew", padx=3, pady=3)
            for r, key in enumerate(keys):
                self.images[key] = tk.BooleanVar(self, key in chosen)
                button = ttk.Checkbutton(box, text=FIGURE_TITLES[key], variable=self.images[key],
                                         command=self.update_summary)
                button.grid(row=r, column=0, sticky=tk.W)
                if jet_off and key in jet.JET_GROUP[1]:
                    button.state(["disabled"])
                if page.post.instantaneous(key) if page.post is not None else key in INSTANTANEOUS:
                    self.videos[key] = tk.BooleanVar(self, key in choice["videos"])
                    own = " (own settings)" if self.own and key in self.own else ""
                    ttk.Checkbutton(box, text="Video" + own, variable=self.videos[key],
                                    command=self.update_summary).grid(row=r, column=1, sticky=tk.W, padx=(8, 0))
        row = ttk.Frame(frame)
        row.pack(anchor=tk.W, pady=(4, 0))
        ttk.Button(row, text="All figures", command=lambda: self.check_all(True)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="No figure", command=lambda: self.check_all(False)).pack(side=tk.LEFT)
        # The maps of the instantaneous figures and of the videos
        box = ttk.LabelFrame(frame, text="Instantaneous figures and videos", padding=6)
        box.pack(fill=tk.X, pady=(8, 0))
        self.range = tk.BooleanVar(self, bool(choice["range"]))
        for value, text in ((False, "Images of the map shown"),
                            (True, "Images of the maps below (each figure in its own folder)")):
            ttk.Radiobutton(box, text=text, value=value, variable=self.range,
                            command=self.update_summary).pack(anchor=tk.W)
        ttk.Label(box, text="First and last maps (images and videos)").pack(anchor=tk.W, pady=(6, 0))
        self.slider = RangeSlider(box, 1, self.n, command=lambda first, last: self.update_summary())
        self.slider.pack(fill=tk.X)
        grid = ttk.Frame(box)
        grid.pack(fill=tk.X, pady=(4, 0))
        self.step = tk.StringVar(self, str(choice["step"]))
        self.fps = tk.StringVar(self, num2str(choice["fps"]))
        self.format = tk.StringVar(self, choice["format"])
        for r, (label, var) in enumerate((("Every N maps (1: all the maps)", self.step),
                                          ("Frames per second (videos)", self.fps))):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky=tk.W, pady=1)
            ttk.Entry(grid, textvariable=var, width=9).grid(row=r, column=1, sticky=tk.E, pady=1)
            var.trace_add("write", lambda *args: self.update_summary())
        ttk.Label(grid, text="Format (videos)").grid(row=2, column=0, sticky=tk.W, pady=1)
        format_box(grid, self.format).grid(row=2, column=1, sticky=tk.E, pady=1)
        self.format.trace_add("write", lambda *args: self.update_summary())
        grid.columnconfigure(0, weight=1)
        if self.own is not None:
            ttk.Label(box, text="The videos of the quantities with settings of their own (saved from the Video "
                                "button of a tile) keep them: the maps, frame rate and format above are those of "
                                "the others.", foreground="#555555", wraplength=self.wrap * 2,
                      justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))
        self.summary = ttk.Label(frame, foreground="#555555", wraplength=self.wrap * 2, justify=tk.LEFT)
        self.summary.pack(anchor=tk.W, pady=(8, 0))
        buttons = ttk.Frame(frame)
        buttons.pack(anchor=tk.E, pady=(10, 0))
        ttk.Button(buttons, text="OK", command=self.ok).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side=tk.LEFT)
        self.slider.set(int(choice["first"] or 1), int(choice["last"] or self.n))  # Updates the summary

    def check_all(self, value):
        for var in self.images.values():
            var.set(value)
        self.update_summary()

    def choice(self):
        """The choice entered (DEFAULT_EXPORT): ValueError with a message."""
        step, fps = read_step_fps(self.step.get(), self.fps.get())
        first, last = self.slider.get()
        figures = [key for key, var in self.images.items() if var.get()]
        return dict(figures=None if len(figures) == len(self.images) else figures,
                    videos=[key for key, var in self.videos.items() if var.get()], range=self.range.get(),
                    first=first, last=None if last == self.n else last, step=step, fps=fps,
                    format=self.format.get())

    def update_summary(self):
        try:
            self.summary["text"] = describe_export(self.choice(), self.n, self.own or ())
        except ValueError as e:
            self.summary["text"] = str(e)

    def ok(self):
        try:
            self.result = self.choice()
        except ValueError as e:
            self.summary["text"] = str(e)
            self.bell()
            return
        self.destroy()
