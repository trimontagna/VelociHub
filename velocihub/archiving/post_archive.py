"""Post-processing of all the PIV results of the archive with the batch API of VelociHub, and the stitching of the fields
of view of each setup.

The offsets of the fields of view are those given (FOV_X_OFFSET_MM, FOV_Y_OFFSET_MM; the camera settings of each
project in the interface), from a calibration made separately: they aren't measured here.
1. Each measurement folder of each field of view (ARCHIVE/FOV_n/[setup], with PIV/vectors.npz) is post-processed
   (vh.batch.postprocess_folders) with the camera of its field of view (FOV_WIDTH_MM, its horizontal offset,
   FRAME_RATE_HZ) and the recipe common to all (SETTINGS): its figures, metrics.json and postprocessing.json exported
   to its subfolder PostProcessing, as the interface exports them.
2. The setups measured in several fields of view (the same folder name) are stitched, once all their fields of
   view are post-processed: each map is drawn again with the fields of view side by side, each with its own x axis
   (global x, with its horizontal offset), on the height common to all of them (global y, with their vertical
   offsets FOV_Y_OFFSET_MM), one colour scale (and one arrow scale), and its reference quantities (time, t_flow,
   u_ref, L_ref) in its title, in STITCHED_FOLDER/[setup]:
   - the maps of the statistics (STITCHED), with their lines (potential core: its boundary, centerline and
     profile locations): [setup]_Stitched_[figure].png;
   - the jet theory plots (STITCHED_JET): the velocity profiles of the potential core of each field of view one under
     the other, and the velocity on the core centerline of all of them along x;
   - the instantaneous maps (STITCHED_INSTANTANEOUS) of the map INDEX: [setup]_Stitched_[figure]_[map].png, or of
     each map of EXPORT_MAPS, each figure in its own folder [setup]_Stitched_[figure];
   - the videos of the instantaneous maps of STITCHED_VIDEOS, over the maps of EXPORT_MAPS (all of them if None) at
     EXPORT_FPS frames per second, as EXPORT_FORMAT (GIF or MP4): [setup]_Stitched_[figure].gif (.mp4), all their
     frames with the same scales (colours, axes, arrows).
   The fields of view were not measured at the same time: their maps of the same number are side by side, each
   with its own time. The measurements given (main(items), the window of VelociHub): the figures stitched, their maps and
   the videos are instead those of the exports of the measurement of each setup in REFERENCE_FOV (stitch_setup).

Rolling statistics (SETTINGS.rolling): the statistics of each map are those of the SETTINGS.rolling_maps maps up to it
(all those before it while there are fewer), so every figure has one per map, like the instantaneous ones: exported
for the map INDEX or each map of EXPORT_MAPS, with videos (EXPORT_VIDEOS, STITCHED_VIDEOS: any map), the stitched maps
of the statistics too, and the jet theory plots those of the map INDEX; the titles give the time of the map and the
averaging time. The reference velocities (and their matching), the outlier filtering and metrics.json stay those of
all the maps.

    velocihub-post-archive [archive folder] [--setups name ...] [--export-root folder] [--stitched folder]

The interface of VelociHub (the window Batch of the post-processing, its stitchings) runs it on each chain of projects
stitched (main with items: each its field of view, its camera, recipe and exports; the stitched figures in the
folder Stitched of the PostProcessing folder of the reference).

--setups: only these setups; --export-root: the exports in [folder]/[FOV]/[setup] instead of the PostProcessing
subfolder of each measurement; --stitched: the stitched figures there instead of STITCHED_FOLDER.

The maps relative to the reference velocity (turbulence intensity, potential core...) are comparable between the
fields of view only with consistent reference velocities (the velocities then shown over them: use_u_ref). With
MATCH_VELOCITY, those of the fields of view of a setup make its mean velocity u / u_ref the same in their overlaps:
the fields of view were measured in separate runs whose flow rates differ, so each has its own u_ref (matched_
references); otherwise (or for a setup in one field of view), the maximum of the mean velocity of its measurement in
REFERENCE_FOV (FOV_1, the jet exit), used for all its fields of view. A setup without REFERENCE_FOV keeps SETTINGS
(u_ref; None: the maximum of each field of view).
"""
import argparse
import dataclasses
import os
import re
import time

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.quiver import Quiver

import velocihub as vh
from velocihub.postprocessing import FIGURE_FILES, INSTANTANEOUS, MAP_FIGURES
from velocihub.fields import EXPORT_DPI, TFLOW, UREF, Video, map_indices, scale_frames, sig2
from velocihub.project import export_name, format_duration

# The archive: a folder per field of view (FOV_1, FOV_2...), a folder per measurement setup in each
ARCHIVE = os.getcwd()
# The fields of view: width (mm), horizontal offset of their left edge (mm, along the jet) and vertical offset of
# the middle of their height (mm; none: 0)
FOV_WIDTH_MM = {"FOV_1": 196.1, "FOV_2": 196.5, "FOV_3": 196.3}
FOV_X_OFFSET_MM = {"FOV_1": 0.0, "FOV_2": 120.0, "FOV_3": 240.0}
FOV_Y_OFFSET_MM = {}
# The order of the fields of view of the measurements given (main(items)): theirs, that of the stitched figures when
# their horizontal offsets are the same
FOV_ORDER = {}
# Common to all the measurements
FRAME_RATE_HZ = 150.0
SETTINGS = vh.PostSettings(
    outlier_k=4.0,          # Statistical outlier filtering: drop a cell beyond this many standard deviations (0: none)
    hold_maps=5,            # Data reconstruction: invalid cells of the instantaneous maps held (# of maps; 0: none)
    use_u_ref=True,        # Reference velocity: the velocities shown over u_ref (False: in m/s)
    u_ref=None,             # ... u_ref (m/s; None: the maximum of the mean u of each field of view)
    use_t_flow=True,        # Flow time: the times in flow times t_flow (False: in s)
    t_flow=None,            # ... t_flow (s; None: L_FoV / u_ref)
    l_ref=50,             # Reference length (mm; None: the positions in mm), e.g. 50 (nozzle diameter)
    peak_locking_bins=21,   # Peak locking: bins per pixel of its histograms (odd)
    core_mode="level",      # Potential core: detection, "level" (core_level) or "skewness" of the profiles
    core_level=80.0,        # ... mean u >= this % of u_ref
    skew_zero=20.0,         # ... skewness: |skewness| at most this % of the highest one
    skew_change=10.0,       # ... and its change from the previous profile at most this % of the previous one
    skew_profiles=40,       # ... profiles analysed
    n_stations=6,           # ... profiles of mean u / u_ref
    first_offset=5,         # ... columns between the exit and the first profile
    core_u_ref=None,        # ... reference velocity of the jet theory without use_u_ref (m/s; None: max of mean u)
    rolling=False,          # Rolling statistics: those of each map over the rolling_maps maps up to it (False: all)
    rolling_maps=100,       # ... maps of the rolling statistics
)
# The reference velocity of each setup: the maximum of the mean u of its measurement in this field of view (None:
# SETTINGS for all)
REFERENCE_FOV = "FOV_1"
# The reference velocities of the fields of view of a setup matched in their overlaps (matched_references): the factor
# between their mean u fitted on the cells of the jet there (both at least MATCH_JET of the highest mean u of the
# overlap), MATCH_EDGE columns at the edges of each field of view left out; the highest maximum of the mean u, all
# brought to the level of REFERENCE_FOV, the reference velocity there. False: the one of REFERENCE_FOV for all
MATCH_VELOCITY = True
MATCH_JET = 0.5
MATCH_EDGE = 2
INDEX = -1                  # Map of the instantaneous figures exported (0: the first; negative: from the end, -1 the last)
# The figures exported (keys of vh.postprocessing.FIGURE_TITLES; None: all of them)
EXPORT_FIGURES = None
# The instantaneous figures (all of them with rolling statistics) of these maps (indices, e.g. range(0, 100, 5),
# negative from the end, e.g. [0, -1]; None: the map INDEX only), each figure in its own folder
# [setup]_PostProcessing_[figure]
EXPORT_MAPS = None
# The videos of these instantaneous figures (any figure with rolling statistics; e.g. ["velocity"]), over the maps of
# EXPORT_MAPS (None: all the maps), at EXPORT_FPS frames per second, as EXPORT_FORMAT: "gif" (256 colours) or "mp4"
# (H.264, with ffmpeg)
EXPORT_VIDEOS = []
EXPORT_FPS = 24
EXPORT_FORMAT = "gif"
# The maps of the stitched videos (indices; None: those of EXPORT_MAPS, else all the maps)
STITCHED_VIDEO_MAPS = None
# The stitched figures: in this folder, one subfolder per setup
STITCHED_FOLDER = os.path.join(ARCHIVE, "Stitched")
# The figures stitched (sized from the maps: stitched_size): the maps of the statistics (but the instantaneous ones)
STITCHED = [key for key in MAP_FIGURES if key not in INSTANTANEOUS]
# ... the instantaneous maps (of the map INDEX, or of each map of EXPORT_MAPS; with rolling statistics, the maps of
# STITCHED too)
STITCHED_INSTANTANEOUS = [key for key in MAP_FIGURES if key in INSTANTANEOUS]
# ... and the videos of these instantaneous maps (any map with rolling statistics; e.g. ["velocity"]), over the maps
# of EXPORT_MAPS (None: all the maps) at EXPORT_FPS frames per second
STITCHED_VIDEOS = ["velocity"]
# ... and the jet theory plots (the potential core map is one of the maps of STITCHED)
STITCHED_JET = ["core_profiles", "core_centreline"]


def measurements(archive):
    """Measurement folders of the archive with PIV results, by field of view: {fov: {setup: folder}} (the fields of
    view of FOV_WIDTH_MM only)."""
    found = {}
    for fov in FOV_WIDTH_MM:
        root = os.path.join(archive, fov)
        if not os.path.isdir(root):
            continue
        found[fov] = {name: os.path.join(root, name) for name in sorted(os.listdir(root))
                      if os.path.isfile(os.path.join(root, name, vh.PIV_FOLDER, "vectors.npz"))}
    return found


def camera(fov, offsets=FOV_X_OFFSET_MM):
    return vh.Camera(fov_width_mm=FOV_WIDTH_MM[fov], x_offset_mm=offsets[fov], frame_rate_hz=FRAME_RATE_HZ,
                      y_offset_mm=FOV_Y_OFFSET_MM.get(fov, 0.0))


def arrows_of(q):
    """The arrows of a quiver: X, Y, U, V (U, V NaN where they are masked: Matplotlib keeps 1 there)."""
    mask = np.broadcast_to(np.asarray(q.Umask, bool), np.shape(q.U))
    U, V = (np.where(mask, np.nan, np.asarray(A, np.float64)) for A in (q.U, q.V))
    return np.asarray(q.X), np.asarray(q.Y), U, V


def maps_of(post, keys, index=0, clims=None, cmaps=None):
    """The maps of figures as the post-processing draws them (the instantaneous ones: of the map index; clims, cmaps:
    the colour ranges and colour maps forced for some of them, {key: (vmin, vmax)}, {key: name}, as exported): {key:
    positions X, Y, field F, colour range, colour map, labels, title, vectors (X, Y, U, V of the arrows; None
    without), lines (x, y, style, label) and extent (left, right, bottom, top of an image: the images and the POD
    modes; None for the maps of the grid)} (the figures not drawn as a single map left out)."""
    maps = {}
    for key in keys:
        fig = post.plot(key, index=index, clim=(clims or {}).get(key), cmap=(cmaps or {}).get(key))
        drawn = post.colour_maps(fig)
        if len(drawn) != 1 or getattr(drawn[0], "field", None) is None:
            continue
        m, ax = drawn[0], drawn[0].axes
        X, Y, F = m.field
        q = next((c for c in ax.collections if isinstance(c, Quiver)), None)
        maps[key] = dict(X=X, Y=Y, F=np.ma.filled(np.ma.masked_invalid(F), np.nan), clim=m.get_clim(),
                         cmap=m.get_cmap().name, label=m.colorbar.ax.get_ylabel(), title=ax.get_title(),
                         xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(), inverted=ax.yaxis_inverted(),
                         quiver=None if q is None else arrows_of(q),
                         extent=tuple(m.get_extent()) if isinstance(m, AxesImage) else None,
                         lines=[(l.get_xdata(), l.get_ydata(), dict(
                             color=l.get_color(), linestyle=l.get_linestyle(), linewidth=l.get_linewidth(),
                             marker=l.get_marker(), markersize=l.get_markersize(),
                             markerfacecolor=l.get_markerfacecolor(), markeredgecolor=l.get_markeredgecolor(),
                             zorder=l.get_zorder()), l.get_label())
                                for l in ax.lines if l.get_transform() == ax.transData])
    return maps


def split_title(title):
    """A title of the post-processing and its reference quantities (the parentheses at its end: time, t_flow, u_ref,
    L_ref...): (title, references), references empty without them."""
    if title.endswith(")"):
        depth = 0
        for i in range(len(title) - 1, -1, -1):
            depth += {")": 1, "(": -1}.get(title[i], 0)
            if depth == 0:
                return title[:i].rstrip(), title[i + 1:-1]
    return title, ""


def map_label(post, k):
    """The map k in the title of a stitched instantaneous map: its pair number in the sensor space only, else its
    time (in flow times with them, or in s)."""
    f = post.fields
    if post.sensor():
        return f" · pair {k + 1}"
    if post.uses_flow_time():
        return rf" · $t$ = {sig2(f.t[k] / post.flow_time())} ${TFLOW}$"
    return rf" · $t$ = {f.t[k]:.4g} s"


def instantaneous(posts, key, k, clims=None, cmaps=None):
    """The parts of a stitched instantaneous map: [(fov, map k of each field of view)] (maps_of, with the colour
    ranges and colour maps forced: clims, cmaps)."""
    return [(fov, maps[key]) for fov, post in posts.items() if k < post.fields.n
            for maps in [maps_of(post, [key], k, clims, cmaps)] if key in maps]


def common_height(parts):
    """The y range common to the maps of the fields of view (parts of draw_stitched: their vertical offsets), else
    (none in common) that of all of them: (low, high)."""
    low, high = max(np.nanmin(m["Y"]) for _, m in parts), min(np.nanmax(m["Y"]) for _, m in parts)
    if not low < high:
        low, high = min(np.nanmin(m["Y"]) for _, m in parts), max(np.nanmax(m["Y"]) for _, m in parts)
    return low, high


def stitched_size(parts):
    """Size of the figure of a stitched map (inches) from its maps (parts of draw_stitched): the maps side by side at
    their aspect (the widths of the fields of view over their common height), about 5.5 inches of width per field of
    view (10 to 22 in all), and the room of the titles, labels and colour bar; the height 3 to 12 (a narrower figure
    for maps taller than that)."""
    widths = sum(np.nanmax(m["X"]) - np.nanmin(m["X"]) for _, m in parts)
    low, high = common_height(parts)
    aspect = (high - low) / widths if widths > 0 else 0.5
    side, top = 1.6, 1.6  # The colour bar and y label; the titles and x label
    width = min(max(5.5 * len(parts), 10), 22)
    height = (width - side) * aspect + top
    if height > 12:
        width, height = side + (12 - top) / aspect, 12
    return round(width, 2), round(max(height, 3), 2)


def wrap_references(text, width):
    """The reference quantities of a title (a, b, ...) on lines of about width inches (the title of an axes, small
    font), split at their commas."""
    def length(s):
        # Inches of the text shown: the math markup left out, a symbol (\Delta...) as one character
        shown = re.sub(r"[${}_^]", "", re.sub(r"\\[A-Za-z]+", "x", s.replace(r"\mathrm", "")))
        return 0.064 * len(shown)
    lines, line = [], ""
    for item in text.split(", "):
        joined = f"{line}, {item}" if line else item
        if line and length(joined) > width:
            lines.append(line + ",")
            line = item
        else:
            line = joined
    return "\n".join(lines + [line])


def draw_stitched(fig, setup, parts, what=""):
    """Draw a map of several fields of view in a figure (parts: [(fov, map)] of maps_of): each in full, side by side
    with its own x axis (global x; widths as their x ranges; in the order of parts), one colour scale (the range of all) and one scale of the arrows, each titled with
    its field of view and the reference quantities of its map (time, t_flow, u_ref, L_ref); the title gives the setup,
    the quantity, and what (the map...)."""
    fig.clf()
    vmin = min(m["clim"][0] for _, m in parts)
    vmax = max(m["clim"][1] for _, m in parts)
    first = parts[0][1]
    # The arrows (instantaneous velocity...): the same scale in all, the fastest ones as long as their spacing
    arrows = [m["quiver"] for _, m in parts if m["quiver"] is not None]
    speeds = [np.nanpercentile(np.hypot(U, V), 99) for _, _, U, V in arrows if np.isfinite(U).any()]
    spacing = min((np.median(np.diff(np.unique(X))) for X, _, _, _ in arrows), default=1.0)
    scale = (max(speeds) / spacing) if speeds and max(speeds) > 0 else None
    # The widths of the axes: the x ranges of the fields of view (the maps at the same scale)
    ratios = [max(np.nanmax(m["X"]) - np.nanmin(m["X"]), 1e-9) for _, m in parts]
    axes = fig.subplots(1, len(parts), sharey=True, squeeze=False, gridspec_kw={"width_ratios": ratios})[0]
    # One norm: the colour range of all the fields of view changed with that of the colour bar (scales of the videos)
    norm = Normalize(vmin, vmax)
    for n, (ax, (fov, m)) in enumerate(zip(axes, parts)):
        ax.set_facecolor("#d5d9e0")
        if m["extent"] is not None:
            im = ax.imshow(m["F"], cmap=first["cmap"], norm=norm, extent=m["extent"], interpolation="antialiased")
        else:
            im = ax.pcolormesh(m["X"], m["Y"], m["F"], cmap=first["cmap"], norm=norm, shading="nearest")
        if m["quiver"] is not None:
            ax.quiver(*m["quiver"], color="white", width=0.0022, scale_units="xy", scale=scale)
        for x, y, style, label in m["lines"]:
            ax.plot(x, y, label=label, **style)
        if any(not label.startswith("_") for *_, label in m["lines"]):
            ax.legend(loc="upper right", fontsize="xx-small")
        ax.set_aspect("equal")
        ax.set_xlim(np.nanmin(m["X"]), np.nanmax(m["X"]))
        ax.set_xlabel(m["xlabel"])
        # The reference quantities on lines as wide as the map (the room of the colour bar and labels left out)
        references = split_title(m["title"])[1]
        # The width of the figure (that of a subfigure: of its figure, the subfigures one under the other)
        width = (fig.get_figure(root=True).get_size_inches()[0] - 1.6) * ratios[n] / sum(ratios)
        ax.set_title(fov.replace("_", " ") + (f"\n({wrap_references(references, width)})" if references else ""),
                     fontsize="small")
    # The height common to all the fields of view (their vertical offsets), else (none in common) all of them
    axes[0].set_ylim(*common_height(parts))
    if first["inverted"]:
        axes[0].invert_yaxis()
    # Global y if a field of view has a vertical offset
    axes[0].set_ylabel(next((m["ylabel"] for _, m in parts if "global" in m["ylabel"]), first["ylabel"]))
    fig.suptitle(f"{setup} · {split_title(first['title'])[0]}{what}", fontweight="bold", fontsize="medium")
    fig.colorbar(im, ax=list(axes), shrink=0.85, label=first["label"])


def stitch(setup, parts, path, what=""):
    """A stitched map (draw_stitched) saved as a PNG. Returns the file written."""
    fig = Figure(figsize=stitched_size(parts), layout="constrained")
    draw_stitched(fig, setup, parts, what)
    return save(fig, path)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=EXPORT_DPI)
    return path


def stitch_pod_modes(setup, posts, folder, clims=None, cmaps=None):
    """The POD modes of the fields of view stitched (posts: {fov: PostProcessing}, in their order): each mode of the
    range of the first one (pod_range: those evaluated) stitched alone ([setup]_Stitched_PODMode[n].png); none when
    they are summed (pod_combine: the background of the image of each map, stitched as the instantaneous maps). Each
    field of view has its own POD: the mode n of one isn't that of another (nor its sign). Returns the files
    written."""
    first = next(iter(posts.values()))
    if first.pod_combined():
        return []
    modes = first.pod_modes_exported()
    rows = []
    for mode in modes:
        parts = []
        for fov, post in posts.items():
            with post.single_mode(mode):
                maps = maps_of(post, ["pod_mode"], clims=clims, cmaps=cmaps)
            if "pod_mode" in maps:
                parts.append((fov, maps["pod_mode"]))
        if len(parts) > 1:
            rows.append((mode, parts))
    written = []
    for mode, parts in rows:
        with first.single_mode(mode):
            element = first.export_element("pod_mode")
        written.append(stitch(setup, parts, os.path.join(folder, export_name(setup, "Stitched", element,
                                                                             extension="png"))))
    return written


def stitch_core_profiles(setup, posts, path):
    """The velocity profiles of the potential core of the fields of view (posts: {fov: PostProcessing}, from
    upstream), one row each, titled with its field of view and its reference quantities. Returns the file written."""
    width, height = vh.postprocessing.FIGSIZE["core_profiles"]
    fig = Figure(figsize=(width, height * 0.72 * len(posts)), layout="constrained")
    for sub, (fov, post) in zip(np.atleast_1d(fig.subfigures(len(posts), 1)), posts.items()):
        post.draw_core_profiles(sub, f"{fov.replace('_', ' ')} · ")
    fig.suptitle(f"{setup} · Potential core velocity profiles", fontweight="bold", fontsize="large")
    return save(fig, path)


def stitch_core_centreline(setup, posts, path):
    """The velocity on the potential core centerline of the fields of view (posts: {fov: PostProcessing}, from
    upstream) along x, each in its colour with its end of core; the reference quantities in the title (those of a
    field of view in its label if they differ). Returns the file written, None without any core."""
    cores = [(fov, post, post.potential_core()) for fov, post in posts.items()]
    cores = [(fov, post, pc) for fov, post, pc in cores if pc["error"] is None]
    if not cores:
        return None
    first = cores[0][1].refs(cores[0][1].window_text(), u_ref=cores[0][2]["u_ref"], newline=False)
    fig = Figure(figsize=vh.postprocessing.FIGSIZE["core_centreline"], layout="constrained")
    ax = fig.add_subplot()
    for n, (fov, post, pc) in enumerate(cores):
        name, colour, i0 = fov.replace("_", " "), f"C{n}", pc["stations"][0]
        refs = post.refs(post.window_text(), u_ref=pc["u_ref"], newline=False)
        ending = ("; the core doesn't end in the field" if pc["x_end_pc"] is None
                  else "; the core ended before the field" if pc["ended_before"] else "")
        ax.plot(post.plot_x(pc["x_mm"])[i0:], pc["R_pc"][i0:], "o-", ms=3, color=colour,
                label=f"{name}: core centerline" + ("" if refs == first else refs) + ending)
        if pc["x_end_pc"] is not None and not pc["ended_before"]:
            ax.axvline(post.plot_x(pc["x_end_pc"]), color=colour, ls="--",
                       label=f"{name}: end of core, {post.position('x', pc['x_end_pc'], 0)}")
    ax.axhline(1, color="k", lw=0.8)
    # The end condition: each field of view its own when they differ (the skewness detection)
    if len({pc["level"] for _, _, pc in cores}) == 1:
        ax.axhline(cores[0][2]["level"], color="r", ls=":", label="End Condition")
    else:
        for n, (fov, _, pc) in enumerate(cores):
            ax.axhline(pc["level"], color=f"C{n}", ls=":", label=f"{fov.replace('_', ' ')}: end condition")
    ax.set_xlabel(cores[0][1].position_label("x global"))
    ax.set_ylabel(rf"$\overline{{u}}\ /\ {UREF}$")
    ax.set_title(f"{setup} · Potential core velocity on centerline" + ("\n" + first.strip() if first else ""),
                 fontweight="bold", fontsize="medium")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    return save(fig, path)


def stitch_setup(setup, posts, folder, export=None, index=None):
    """The stitched figures of a setup from the post-processing of its fields of view (posts: {fov:
    PostProcessing}), in a folder: the maps of the statistics (STITCHED), the instantaneous ones
    (STITCHED_INSTANTANEOUS) of the map INDEX or of each map of EXPORT_MAPS (each figure in its own folder), and
    the videos of STITCHED_VIDEOS, and the jet theory plots (STITCHED_JET). The maps are numbered up to the fewest
    maps of the fields of view. With rolling statistics, the maps of STITCHED are those of each map, as the
    instantaneous ones, and the jet theory plots those of the map INDEX. export, index: the exports of the setup
    instead (options of vh.batch.postprocess_folder, those of its measurement in REFERENCE_FOV: its figures, the
    maps of the instantaneous ones and the videos, each with its own maps, frame rate and format), and its map of the
    instantaneous figures. Returns the files written."""
    # Along x (their horizontal offsets), else (the same offset) in their order (FOV_ORDER)
    offsets = {fov: FOV_X_OFFSET_MM[fov] for fov in posts}
    along = len(set(offsets.values())) > 1
    posts = dict(sorted(posts.items(), key=lambda item: offsets[item[0]] if along else FOV_ORDER.get(item[0], 0)))
    n = min(post.fields.n for post in posts.values())
    first = next(iter(posts.values()))
    if export is None:
        figures, videos, index, maps_of_images = STITCHED + STITCHED_INSTANTANEOUS + STITCHED_JET, STITCHED_VIDEOS, \
            INDEX, EXPORT_MAPS
        video_maps = (STITCHED_VIDEO_MAPS if STITCHED_VIDEO_MAPS is not None
                      else range(n) if EXPORT_MAPS is None else EXPORT_MAPS)
        fps, video_format, image_options, video_options = EXPORT_FPS, EXPORT_FORMAT, {}, {}
    else:
        figures = list(FIGURE_FILES) if export.get("figures") is None else export["figures"]
        videos, maps_of_images, video_maps = export.get("videos") or [], export.get("maps"), export["video_maps"]
        fps, video_format = export.get("fps", EXPORT_FPS), export.get("video_format", EXPORT_FORMAT)
        image_options, video_options = export.get("image_options") or {}, export.get("video_options") or {}
        index = INDEX if index is None else index
    # The colour ranges and colour maps forced for the quantities (those of their exports; the stitched map in the
    # range forced, the same in all its fields of view)
    clims, cmaps = (export or {}).get("clims") or {}, (export or {}).get("cmaps") or {}
    # The maps of each map (instantaneous, or any with rolling statistics) and those of all the maps
    per_map = [key for key in figures if key in MAP_FIGURES and first.instantaneous(key)]
    # The POD modes apart (each mode of the range, or all combined: stitch_pod_modes)
    whole = [key for key in figures if key in MAP_FIGURES and not first.instantaneous(key) and key != "pod_mode"]

    def name(key, k=None, extension="png"):
        # The element of the figure (that of the POD mode with its number)
        return export_name(setup, "Stitched", first.export_element(key), None if k is None else k + 1, extension, n)
    written = []
    statistics = {fov: maps_of(post, whole, clims=clims, cmaps=cmaps) for fov, post in posts.items()}
    for key in whole:
        parts = [(fov, maps[key]) for fov, maps in statistics.items() if key in maps]
        if len(parts) > 1:
            written.append(stitch(setup, parts, os.path.join(folder, name(key))))
    if "pod_mode" in figures:
        written += stitch_pod_modes(setup, posts, folder, clims, cmaps)
    # The jet theory plots: of the map index (rolling statistics: the window ending there)
    for post in posts.values():
        post.map_index = map_indices([index], n)[0]
    for key, draw in (("core_profiles", stitch_core_profiles), ("core_centreline", stitch_core_centreline)):
        if key in figures:
            path = draw(setup, posts, os.path.join(folder, name(key)))
            written += [path] if path else []
    for key in per_map:
        # The maps of the figure: its own ones, else those of all (None: the map index, in the folder)
        own = image_options.get(key, maps_of_images)
        where = folder if own is None else os.path.join(folder, name(key, extension=""))
        for k in map_indices([index] if own is None else own, n):
            parts = instantaneous(posts, key, k, clims, cmaps)
            if len(parts) > 1:
                written.append(stitch(setup, parts, os.path.join(where, name(key, k)), map_label(first, k)))
    for key in videos:
        # The maps, frame rate and format of the video: its own ones, else those of all
        own = video_options.get(key) or {}
        frames = map_indices(own.get("maps", video_maps), n)
        fps, video_format = own.get("fps", fps), own.get("format", video_format)
        if not first.instantaneous(key) or not frames or len(instantaneous(posts, key, frames[0], clims, cmaps)) < 2:
            continue
        t0 = time.time()

        def draw(fig, k):
            draw_stitched(fig, setup, instantaneous(posts, key, k, clims, cmaps), map_label(first, k))
        # The size of the frames: that of the first one (the same fields of view in all)
        size = stitched_size(instantaneous(posts, key, frames[0], clims, cmaps))
        with Video(os.path.join(folder, name(key, extension=video_format)), fps, size, fmt=video_format) as video:
            # The same scales in all the frames (those of some of them)
            for k in scale_frames(frames):
                video.measure(draw, k)
            for k in frames:
                video.add(draw, k)
        written.append(video.path)
        print(f"  video {os.path.basename(video.path)}: {len(frames)} frames in {format_duration(time.time() - t0)}")
    return written


def overlap_factor(a, b):
    """Factor of the mean u of a field of view b over that of a in their overlap (b ≈ factor x a), fitted (least
    squares) on the cells of the jet there (both at least MATCH_JET of the highest mean u of the overlap), b
    interpolated on the grid of a, MATCH_EDGE columns at the edges of each left out. a, b: (x, y, mean u) [row,
    column], x global. Returns (factor, cells), (None, 0) without overlap."""
    from scipy.interpolate import RegularGridInterpolator

    (xa, ya, Ua), (xb, yb, Ub) = a, b
    xa, xb = xa[MATCH_EDGE:-MATCH_EDGE], xb[MATCH_EDGE:-MATCH_EDGE]
    Ua, Ub = Ua[:, MATCH_EDGE:-MATCH_EDGE], Ub[:, MATCH_EDGE:-MATCH_EDGE]
    columns = (xa >= xb.min()) & (xa <= xb.max())
    if columns.sum() < 2:
        return None, 0
    # The grid of b ascending for the interpolation
    rows, cols = np.argsort(yb), np.argsort(xb)
    interpolate = RegularGridInterpolator((yb[rows], xb[cols]), Ub[np.ix_(rows, cols)], bounds_error=False,
                                          fill_value=np.nan)
    X, Y = np.meshgrid(xa[columns], ya)
    A, B = Ua[:, columns], interpolate(np.stack([Y, X], axis=-1))
    with np.errstate(invalid="ignore"):
        top = np.nanmax(np.where(np.isfinite(B), A, np.nan))
        jet = np.isfinite(A) & np.isfinite(B) & (A >= MATCH_JET * top) & (B >= MATCH_JET * top)
    if jet.sum() < 10:
        return None, int(jet.sum())
    return float((A[jet] * B[jet]).sum() / (A[jet] ** 2).sum()), int(jet.sum())


def matched_references(folders, offsets, setups, base=lambda folder: SETTINGS,
                       maps=lambda folder: os.path.join(folder, vh.PIV_FOLDER), camera_of=None):
    """The reference velocities of the fields of view of each of the setups (folders: (fov, setup, folder), those of
    REFERENCE_FOV first) that make their mean u / u_ref the same in their overlaps: the factor of the mean u of each
    field of view over the previous one (along x) in their overlap (overlap_factor), chained from the first; all the
    mean u brought to its level (over their factors), the highest of their maxima is the reference velocity of the
    first, that of each other one its factor times it. maps(folder): the folder of the vector fields of a
    measurement, camera_of(folder): its camera (else that of its field of view). Returns {setup: {fov: u_ref
    (m/s)}}."""
    out = {}
    for setup in sorted(setups):
        means = {}
        for fov, _, folder in (item for item in folders if item[1] == setup):
            cam = camera_of(folder) if camera_of else camera(fov, offsets)
            post = vh.PostProcessing(vh.load_vectors(maps(folder), camera=cam), base(folder))
            with post.all_maps():
                mean_U = np.array(post.statistics()["mean_U"], np.float64)
            means[fov] = (post.fields.x[0].astype(np.float64), post.fields.y[:, 0].astype(np.float64), mean_U)
            del post
        order = sorted(means, key=lambda fov: means[fov][0].min())
        factors, notes = {order[0]: 1.0}, []
        for previous, fov in zip(order, order[1:]):
            factor, cells = overlap_factor(means[previous], means[fov])
            if factor is None:
                factor = 1.0
                notes.append(f"{fov}/{previous}: no overlap of the jet ({cells} cells), 1")
            else:
                notes.append(f"{fov}/{previous} = {factor:.3f} ({cells} cells)")
            factors[fov] = factors[previous] * factor
        # The highest maximum of the mean u at the level of the first field of view
        top = {fov: float(np.nanmax(means[fov][2])) / factors[fov] for fov in order}
        highest = max(top, key=top.get)
        out[setup] = {fov: top[highest] * factors[fov] for fov in order}
        print(f"  reference velocities of {setup}: " + ", ".join(f"{fov} {u:.4g} m/s" for fov, u in out[setup].items())
              + f" (overlap factors {'; '.join(notes)}; the highest maximum of the mean u in {highest})")
    return out


def use_items(items):
    """The measurements given (items: {"folder", "fov", "setup", "camera": {"fov_width_mm", "x_offset_mm",
    "frame_rate_hz"}, "settings": PostSettings, "export": options of vh.batch.postprocess_folder}): the widths, the
    horizontal and the vertical offsets of their fields of view (those of the first measurement of each). Returns
    {fov: {setup: folder}}."""
    global FOV_WIDTH_MM, FOV_X_OFFSET_MM, FOV_Y_OFFSET_MM, FOV_ORDER
    found, widths, offsets, vertical = {}, {}, {}, {}
    for item in items:
        fov = item["fov"]
        found.setdefault(fov, {})[item["setup"]] = item["folder"]
        widths.setdefault(fov, float(item["camera"]["fov_width_mm"]))
        offsets.setdefault(fov, float(item["camera"].get("x_offset_mm") or 0))
        vertical.setdefault(fov, float(item["camera"].get("y_offset_mm") or 0))
    FOV_WIDTH_MM, FOV_X_OFFSET_MM, FOV_Y_OFFSET_MM = widths, offsets, vertical
    FOV_ORDER = {fov: n for n, fov in enumerate(found)}
    return found


def main(archive=ARCHIVE, setups=None, export_root=None, stitched_folder=STITCHED_FOLDER, items=None):
    """Post-process and stitch the measurements of an archive (see the module), or those given (items: use_items;
    each with its camera, recipe and exports, the stitching of the setups in several fields of view as for an
    archive). Returns whether all of them were post-processed."""
    t0 = time.time()
    by_folder = {os.path.normpath(item["folder"]): item for item in items or []}
    found = measurements(archive) if items is None else use_items(items)
    folders = [(fov, setup, folder) for fov, by_setup in found.items() for setup, folder in by_setup.items()
               if setups is None or setup in setups]
    # Setup by setup (stitched once all its fields of view are post-processed), its measurement in the reference
    # field of view first: the reference velocity of the others
    folders.sort(key=lambda item: (item[1], item[0] != REFERENCE_FOV, item[0]))
    fovs_of = {}
    for fov, setup, _ in folders:
        fovs_of.setdefault(setup, []).append(fov)
    common = {setup for setup, fovs in fovs_of.items() if len(fovs) > 1}
    print(f"{len(folders)} measurements in {len(found)} fields of view; {len(common)} setups in several: "
          + ", ".join(sorted(common)))
    fov_of = {folder: fov for fov, _, folder in folders}
    # The setup of each measurement (that of its item, not the name of its folder, for those given)
    setup_of = {folder: setup for _, setup, folder in folders}
    # The horizontal offsets of the fields of view: those given
    offsets = {fov: FOV_X_OFFSET_MM[fov] for fov in found}
    export = (None if export_root is None
              else lambda folder: os.path.join(export_root, fov_of[folder], os.path.basename(folder)))
    # The reference velocity of each setup (m/s), from its measurement in the reference field of view
    references = {}
    # The reference velocities of the fields of view of the setups in several, matched in their overlaps
    matched = {}
    def base(folder):
        """The recipe of a measurement before its reference velocity: SETTINGS, or its own (items)."""
        item = by_folder.get(os.path.normpath(folder))
        return item["settings"] if item is not None else SETTINGS

    def maps_of_folder(folder):
        """The folder of the vector fields of a measurement (its own for those given: maps_folder)."""
        item = by_folder.get(os.path.normpath(folder))
        return (item or {}).get("export", {}).get("maps_folder") or os.path.join(folder, vh.PIV_FOLDER)

    def camera_of(folder):
        item = by_folder.get(os.path.normpath(folder))
        cam = camera(fov_of[folder], offsets)
        return cam if item is None else vh.Camera(fov_width_mm=FOV_WIDTH_MM[fov_of[folder]],
                                                    x_offset_mm=offsets[fov_of[folder]],
                                                    y_offset_mm=FOV_Y_OFFSET_MM.get(fov_of[folder], 0.0),
                                                    frame_rate_hz=float(item["camera"]["frame_rate_hz"]))
    if MATCH_VELOCITY:
        # The setups whose fields of view have the same horizontal offset: their overlaps unknown, not matched
        placed = {setup for setup in common if len({offsets[fov] for fov in fovs_of[setup]}) > 1}
        for setup in sorted(common - placed):
            print(f"{setup}: the same horiz. offset for all its fields of view (overlaps unknown): reference "
                  f"velocities not matched, that of {REFERENCE_FOV} for all")
        if placed:
            print("The reference velocities matched in the overlaps of the fields of view:")
        matched = matched_references(folders, offsets, placed, base, maps_of_folder, camera_of)

    def settings(folder):
        """The recipe of a measurement: SETTINGS, with the reference velocity of its field of view matched in the
        overlaps (matched), else of its setup: in the reference field of view, the maximum of its mean u (u_ref
        None); in the others, that one (if the setup has a measurement there)."""
        setup = setup_of[folder]
        own = base(folder)
        if setup in matched:
            return dataclasses.replace(own, use_u_ref=True, u_ref=matched[setup][fov_of[folder]])
        if fov_of[folder] == REFERENCE_FOV:
            return dataclasses.replace(own, use_u_ref=True, u_ref=None)
        u_ref = references.get(setup)
        return own if u_ref is None else dataclasses.replace(own, use_u_ref=True, u_ref=u_ref)

    # The post-processing of the fields of view of the setup being processed, kept until it is stitched
    pending, failed, written = {}, [], [0]

    def reference_item(setup):
        """The measurement given of a setup whose exports are stitched: the one in REFERENCE_FOV, else the first."""
        mine = [item for item in items or [] if item["setup"] == setup]
        return next((item for item in mine if item["fov"] == REFERENCE_FOV), mine[0] if mine else None)

    def stitch_pending():
        for setup, posts in pending.items():
            if len(posts) > 1:
                item = reference_item(setup)
                # The stitched figures: in the folder of the setup, or that of its reference measurement given
                folder = (item or {}).get("stitched_folder") or os.path.join(stitched_folder, setup)
                files = (stitch_setup(setup, posts, folder) if item is None else
                         stitch_setup(setup, posts, folder, item["export"], item.get("index")))
                written[0] += len(files)
                print(f"stitched {setup} ({' + '.join(posts)}): {len(files)} files in {folder}")
        pending.clear()  # Their vector fields released

    def results():
        """The post-processing of the measurements: the exports of the header, or those of each (items)."""
        if items is None:
            yield from vh.batch.postprocess_folders([folder for _, _, folder in folders], camera_of, settings,
                                                     export_folder=export, index=INDEX,
                                                     figures=EXPORT_FIGURES, maps=EXPORT_MAPS, videos=EXPORT_VIDEOS,
                                                     fps=EXPORT_FPS, video_format=EXPORT_FORMAT)
            return
        for _, _, folder in folders:
            item = by_folder[os.path.normpath(folder)]
            yield from vh.batch.postprocess_folders([folder], camera_of, settings, index=item.get("index", INDEX),
                                                     **item["export"])

    for result in results():
        setup = setup_of[result.folder]
        if setup not in pending:
            stitch_pending()
        if result.status != "done":
            failed.append(result)
        elif fov_of[result.folder] == REFERENCE_FOV and setup not in matched:
            with result.post.all_maps():
                references[setup] = float(np.nanmax(result.post.statistics()["mean_U"]))
            print(f"  reference velocity of {setup}: {references[setup]:.4g} m/s (maximum of the mean u in "
                  f"{REFERENCE_FOV}), for all its fields of view")
        if result.status == "done" and setup in common:
            pending.setdefault(setup, {})[fov_of[result.folder]] = result.post
        else:
            pending.setdefault(setup, {})
        result.post = None
    stitch_pending()
    print(f"\n{len(folders) - len(failed)} measurements post-processed, {written[0]} stitched files, in "
          f"{format_duration(time.time() - t0)}")
    for result in failed:
        print(f"  failed: {result.folder}: {result.message}")
    return not failed


def cli(argv=None):
    """The command line (velocihub-post-archive)."""
    parser = argparse.ArgumentParser(prog="velocihub-post-archive", description="Post-processing of the archive (batch API of VelociHub) and stitching "
                                                 "of the fields of view of each setup")
    parser.add_argument("archive", nargs="?", default=ARCHIVE, help="the archive folder (FOV_1, FOV_2...)")
    parser.add_argument("--setups", nargs="+", help="only these setups (names of the measurement folders)")
    parser.add_argument("--export-root", help="the exports in [folder]/[FOV]/[setup] instead of each PostProcessing")
    parser.add_argument("--stitched", help="the stitched figures in this folder")
    args = parser.parse_args(argv)
    ok = main(args.archive, set(args.setups) if args.setups else None, args.export_root,
              args.stitched or (STITCHED_FOLDER if args.archive == ARCHIVE
                                else os.path.join(args.archive, "Stitched")))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    cli()
