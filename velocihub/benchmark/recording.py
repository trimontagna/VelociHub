"""The pre-processing recipes compared on a real recording: the same runs as the hard images of the benchmark
(image_level.RECIPES, the PIV with PaIRS-UniNa and the PTV with TracTrac), on the images of a measurement folder.

A recording has no exact field, so the measures are those that need none:
- PIV: the valid vectors (those the PIV kept), the median correlation peak ratio, the spatial noise of the fields (the
  median distance of each vector from the median of its 8 neighbours, px: the random error, plus the second
  derivative of the flow, the same for every recipe; the vectors with their 8 neighbours), the cells with a value;
- PTV: the particles linked per pair (median), the suspect links (those that fail the normalized median test against
  their neighbouring links, %: an estimate of the wrong links), the lengths of the tracks, the particles detected per
  frame and the share of them tracked, the cells of the binned field with a value.

The images are linked (or copied) into a work folder, with the masks and the PIV settings of the project of the
recording (VelociHub_project.json, or masks.json), which is only read. The pre-processed images are exported as PNG.

    python -m velocihub.benchmark --recording D:/Jet/Run_30 [--frames 100] [--recipes none pod1 pod1_edge]

or from Python: compare_recording(folder, work) then write_recording_report(result, out).
"""

import glob
import json
import os
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..fields import load_vectors
from ..piv.displacements import pair_particles, read_tracks
from .image_level import ENGINES, HARD_RECIPES, PIV_SETTINGS, RECIPES, Run, link_images, run_recipes

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg")
# The link test of the PTV (as SPICY's outlier test of the links: spicy_worker.link_outliers)
LINK_TEST, LINK_NEIGHBOURS, LINK_RADIUS = 3.0, 10, 48.0


@dataclass
class RecordingResult:
    """The comparison on a recording: its folder, images, masks, settings, the metrics ({"rec_<recipe>": {method:
    {metric}}}), the runs ({run folder: record}) and the time."""

    folder: str
    images: list
    masks: list
    piv_settings: dict
    metrics: dict = field(default_factory=dict)
    runs: dict = field(default_factory=dict)
    seconds: float = 0.0
    date: str = ""


def recording_images(folder, frames=None):
    """The images of a recording folder (sorted), the first frames of them."""
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMAGE_EXTENSIONS))
    return paths[:frames] if frames else paths


def recording_setup(folder):
    """The masks (polygons) and the PIV settings (windows, overlap) of the project of a recording, or its masks.json;
    the defaults of the benchmark without them."""
    from ..project import read_project

    masks, piv_settings = [], dict(PIV_SETTINGS)
    data = read_project(folder) or {}
    pre = data.get("preprocessing") or {}
    if pre.get("masks"):
        masks = [m["points"] for m in pre["masks"] if m.get("enabled", True)]
    elif os.path.isfile(os.path.join(folder, "masks.json")):
        with open(os.path.join(folder, "masks.json"), encoding="utf-8") as f:
            masks = [m["points"] for m in json.load(f).get("masks", []) if m.get("enabled", True)]
    pairs = ((data.get("piv") or {}).get("settings") or {}).get("engine_settings", {}).get("pairs") or {}
    if pairs.get("windows"):
        piv_settings["windows"] = tuple(int(w) for w in str(pairs["windows"]).split())
    if pairs.get("overlap"):
        piv_settings["overlap"] = float(pairs["overlap"])
    return masks, piv_settings


def neighbour_residual(U, V, flowing):
    """The distance of each vector from the median of its 8 neighbours (px; NaN without all of them: at the edges
    and next to the masks, where a one-sided median is biased by the gradient)."""
    U = np.where(flowing, U, np.nan)
    V = np.where(flowing, V, np.nan)
    pad = [(0, 0)] * (U.ndim - 2) + [(1, 1), (1, 1)]
    PU, PV = np.pad(U, pad, constant_values=np.nan), np.pad(V, pad, constant_values=np.nan)
    h, w = U.shape[-2:]
    shifts = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]
    NU = np.stack([PU[..., 1 + i:h + 1 + i, 1 + j:w + 1 + j] for i, j in shifts])
    NV = np.stack([PV[..., 1 + i:h + 1 + i, 1 + j:w + 1 + j] for i, j in shifts])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN neighbourhoods
        mu, mv = np.nanmedian(NU, 0), np.nanmedian(NV, 0)
        enough = np.isfinite(NU).all(0)
    return np.where(enough, np.hypot(U - mu, V - mv), np.nan)


def recording_metrics(path):
    """The measures of a PIV or a PTV of a recording that need no exact field (see the module)."""
    from ..piv.spicy_worker import link_outliers

    f = load_vectors(path, pixels=True)
    flowing = f.status == 0
    U, V = f.u_px.astype(np.float64), f.v_px.astype(np.float64)
    m = {"coverage": float(100 * np.mean(np.isfinite(U)[:, flowing]))}
    r = neighbour_residual(U, V, flowing)[:, flowing]
    r = r[np.isfinite(r)]
    m["spatial_noise"] = float(np.median(r)) if r.size else np.nan
    d = np.hypot(U, V)[:, flowing]
    d = d[np.isfinite(d)]
    m["displacement"] = float(np.median(d)) if d.size else np.nan
    if not f.tracks:
        m["valid"] = float(100 * np.mean(np.asarray(f.valid)[:, flowing]))
        sn = np.asarray(f.peak_ratio, np.float64)[:, flowing] if getattr(f, "peak_ratio", None) is not None else None
        if sn is not None:
            sn = sn[np.isfinite(sn)]
            m["peak_ratio"] = float(np.median(sn)) if sn.size else np.nan
        return m
    T = read_tracks(f.tracks)
    counts, suspect, total = [], 0, 0
    for k in range(f.n):
        pp = pair_particles(T, k)
        if pp is None or not pp["linked"]:
            counts.append(0)
            continue
        counts.append(pp["x"].size)
        bad = link_outliers(pp["x"], pp["y"], pp["dx"], pp["dy"], LINK_TEST, LINK_NEIGHBOURS, LINK_RADIUS)
        suspect += int(bad.sum())
        total += bad.size
    m["links_per_pair"] = float(np.median(counts)) if counts else 0.0
    m["suspect_links"] = float(100 * suspect / total) if total else np.nan
    _, lengths = np.unique(T["id"], return_counts=True)
    m["track_length_median"] = float(np.median(lengths)) if lengths.size else 0.0
    m["track_length_mean"] = float(np.mean(lengths)) if lengths.size else 0.0
    with np.load(f.tracks) as z:
        detected, tracked = np.asarray(z["detected"], np.float64), np.asarray(z["tracked"], np.float64)
    m["detected_per_frame"] = float(np.median(detected)) if detected.size else 0.0
    m["yield"] = float(100 * tracked.sum() / max(detected.sum(), 1))
    return m


def compare_recording(folder, work=None, recipes=HARD_RECIPES, frames=None, report=None, progress=None, keep=False):
    """The recipes compared on the images of a recording folder (the first frames of them): RecordingResult.
    work: the folder of the runs (a temporary one by default, removed at the end unless keep). report(text): the
    progress; progress(done, total): the runs done."""
    from datetime import datetime

    report = report or (lambda text: None)
    progress = progress or (lambda done, total: None)
    t0 = time.time()
    images = recording_images(folder, frames)
    if len(images) < 2:
        raise ValueError(f"No recording in {folder}: fewer than 2 images")
    masks, piv_settings = recording_setup(folder)
    result = RecordingResult(os.path.abspath(folder), images, masks, piv_settings,
                             date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    own = work is None
    work = work or tempfile.mkdtemp(prefix="velocihub_recording_")
    try:
        target = os.path.join(work, os.path.basename(os.path.normpath(folder)))
        report(f"Recording: {len(images)} images linked into {target}")
        link_images(images, target)
        runs = [Run(r, method) for r in recipes for method in ENGINES]
        done = [0]

        def tick():
            done[0] += 1
            progress(done[0], len(runs) + 1)
        progress(0, len(runs) + 1)
        records = run_recipes(target, os.path.basename(target), masks, runs, report, tick,
                              label=f"Recording ({len(images)} images)", piv_settings=piv_settings,
                              export_format="png")
        for run, rec in records.items():
            result.runs[run.folder] = {**rec, "recipe": run.recipe, "method": run.method}
            if rec["file"] is not None:
                report(f"Metrics: {run.folder}")
                result.metrics.setdefault(f"rec_{run.recipe}", {})[run.method] = recording_metrics(rec["file"])
        progress(len(runs) + 1, len(runs) + 1)
        result.seconds = time.time() - t0
        return result
    finally:
        if own and not keep:
            shutil.rmtree(work, ignore_errors=True)


################################################################################
# The report

RECORDING_COLUMNS = {"piv": (("valid", False), ("peak_ratio", False), ("spatial_noise", True), ("coverage", False),
                             ("displacement", None)),
                     "ptv": (("links_per_pair", False), ("suspect_links", True), ("track_length_median", False),
                             ("track_length_mean", False), ("yield", False), ("detected_per_frame", False),
                             ("coverage", False), ("displacement", None))}
RECORDING_NOTES = {"piv": "The vectors the PIV kept (the others replaced by its median test), the median correlation "
                          "peak ratio (S/N), the spatial noise (median distance of a vector from the median of its 8 "
                          "neighbours: the random error, plus the curvature of the flow, the same for all) and the "
                          "cells with a value; the median displacement for reference (not ranked).",
                   "ptv": "The particles linked per pair (median), the suspect links (failing the normalized median "
                          "test against their 10 nearest links: an estimate of the wrong links), the lengths of the "
                          "tracks, the particles tracked of those detected, the particles detected per frame and the "
                          "cells of the binned field with a value; the median displacement of the binned field, for "
                          "reference (not ranked: a lower one than the PIV's means fast particles not tracked)."}


def recording_figure(metrics, folder, name):
    """The recipes on the recording: four panels (PIV valid vectors and spatial noise, PTV links per pair and
    suspect links), the best of each darker."""
    from .report import GRID, INK_2, SERIES, _figure, _save, _style, fmt, recipes_of

    recipes = recipes_of(metrics, "rec")
    if not recipes:
        return None
    fig = _figure(8.6, 0.55 * len(recipes) + 1.6)
    axes = fig.subplots(2, 2, sharey=True)
    y = np.arange(len(recipes))[::-1]
    panels = (("piv", "valid", False, "PIV: valid vectors (%)", SERIES["spicy_piv"]),
              ("piv", "spatial_noise", True, "PIV: spatial noise (px)", SERIES["spicy_piv"]),
              ("ptv", "links_per_pair", False, "PTV: particles linked per pair", SERIES["binned"]),
              ("ptv", "suspect_links", True, "PTV: suspect links (%)", SERIES["binned"]))
    for ax, (method, key, low, title, colour) in zip(axes.ravel(), panels):
        _style(ax)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        vals = np.array([metrics[f"rec_{r}"].get(method, {}).get(key, np.nan) for r in recipes], float)
        finite = vals[np.isfinite(vals)]
        target = (finite.min() if low else finite.max()) if finite.size else None
        ax.barh(y, vals, 0.62, color=[colour if v == target else colour + "80" for v in vals])
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v, yi, " " + fmt(key, v), va="center", ha="left", fontsize=7, color=INK_2)
        ax.set_xlabel(title, fontsize=8)
        ax.margins(x=0.3)
    for ax in axes[:, 0]:
        ax.set_yticks(y, [RECIPES[r].name for r in recipes], fontsize=8)
    return _save(fig, folder, name)


def write_recording_report(result, folder):
    """The comparison on a recording as a page (index.html, figures/, results.json) in folder. Returns index.html."""
    from .report import _clean, _runs_json, badge, esc, figures_html, fmt, page, recipe_table

    os.makedirs(os.path.join(folder, "figures"), exist_ok=True)
    name = os.path.basename(result.folder)
    rows = []
    for method, key, low in (("piv", "valid", False), ("piv", "spatial_noise", True), ("ptv", "links_per_pair", False),
                             ("ptv", "suspect_links", True), ("ptv", "track_length_median", False)):
        vals = [(r, v[method][key]) for r, v in ((s[4:], v) for s, v in result.metrics.items())
                if method in v and v[method].get(key) is not None and np.isfinite(v[method][key])]
        if not vals:
            continue
        vals.sort(key=lambda rv: rv[1] if low else -rv[1])
        none = result.metrics.get("rec_none", {}).get(method, {}).get(key)
        rows.append(f"<tr><td>{esc(ENGINES[method])}: {esc(RECORDING_COLUMNS_LABELS[key])}</td>"
                    f"<td>{esc(RECIPES[vals[0][0]].name)}</td><td class=num>{fmt(key, vals[0][1])}</td>"
                    f"<td class=num>{fmt(key, none)}</td><td>{esc(RECIPES[vals[-1][0]].name)}</td></tr>")
    errors = "".join(f'<p>{badge("fail")} {esc(f)}: {esc(rec.get("status"))} {esc(rec.get("message") or "")}</p>'
                     for f, rec in result.runs.items() if rec.get("file") is None)
    settings = (f"windows {' '.join(str(w) for w in result.piv_settings['windows'])} px, "
                f"{result.piv_settings['overlap']:g} % overlap")
    parts = [f"<h1>The pre-processing compared on {esc(name)}</h1>",
             f'<p class="lede">The recipes of the benchmark run on a real recording: {len(result.images)} images of '
             f"{esc(result.folder)}, pre-processed with each recipe, then processed by the PIV (PaIRS-UniNa, "
             f"{esc(settings)}) and the PTV (TracTrac, its defaults). Without an exact field, the measures are those "
             "of the quality of the results.</p>",
             f'<p class="meta">{esc(result.date)} · {len(result.masks)} mask(s) · {result.seconds / 60:.0f} min</p>',
             errors, "<h2>The best recipe for each measure</h2>",
             '<div class="scroll"><table><thead><tr><th>Measure</th><th>Best</th><th class=num>Its value</th>'
             "<th class=num>Without pre-processing</th><th>Worst</th></tr></thead><tbody>" + "".join(rows)
             + "</tbody></table></div>"]
    file = recording_figure(result.metrics, os.path.join(folder, "figures"), "recording_recipes.png")
    if file:
        parts.append(figures_html([("The recipes on the recording", "Valid vectors and spatial noise of the PIV, "
                                    "particles linked per pair and suspect links of the PTV; the best of each "
                                    "darker.", file)]))
    for method, title in (("piv", "PIV (PaIRS-UniNa)"), ("ptv", "PTV (TracTrac)")):
        parts.append(f"<h2>{esc(title)}</h2><p class=meta>{esc(RECORDING_NOTES[method])} Hover a recipe for what "
                     "it does; the best of each column in bold.</p>"
                     + recipe_table(result.metrics, "rec", method, RECORDING_COLUMNS[method]))
    times = "".join(f"<tr><td>{esc(f)}</td><td>{esc(rec.get('status'))}</td>"
                    f"<td class=num>{(rec.get('seconds') or 0) / 60:.1f}</td></tr>" for f, rec in result.runs.items())
    parts.append("<h2>Runs</h2><div class=scroll><table><thead><tr><th>Run</th><th>Status</th>"
                 f"<th class=num>Minutes</th></tr></thead><tbody>{times}</tbody></table></div>")
    index = os.path.join(folder, "index.html")
    with open(index, "w", encoding="utf-8") as f:
        f.write(page(f"{name}: the pre-processing compared", "".join(parts), '<a href="index.html">Summary</a>'))
    with open(os.path.join(folder, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"format": "VelociHub recording comparison", "folder": result.folder, "images": len(result.images),
                   "masks": result.masks, "piv_settings": _clean(result.piv_settings), "date": result.date,
                   "seconds": result.seconds, "metrics": _clean(result.metrics),
                   "runs": _runs_json({"recording": result.runs})["recording"]}, f, indent=1, ensure_ascii=False)
    return index


RECORDING_COLUMNS_LABELS = {"valid": "valid vectors (%)", "spatial_noise": "spatial noise (px)",
                            "links_per_pair": "particles linked per pair", "suspect_links": "suspect links (%)",
                            "track_length_median": "median track length (frames)"}


def default_folder(folder):
    """The folder of the report of a recording: reports_folder()/recording_<name>_<date>_<time>."""
    from datetime import datetime

    from .report import reports_folder
    return os.path.join(reports_folder(), f"recording_{os.path.basename(os.path.normpath(folder))}_"
                                          f"{datetime.now():%Y%m%d_%H%M%S}")
