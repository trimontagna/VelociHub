"""The benchmark on particle images: the pre-processing and the processing of VelociHub (level "images"), then the
post-processing of their results (level "chain"), against the exact fields of the flows.

The images of each case (images.render: a time-resolved sequence, particles advected by the flow) are made into a
measurement folder and run as a user runs them, through the API (api.Project): the pre-processing set with a recipe
(RECIPES: masks, contrast filters, POD background removal), then the PIV (PaIRS-UniNa) or the PTV (TracTrac), with or
without SPICY. Each run writes its own folders (PreProcessing_<recipe>, PIV_<recipe>, PTV_<recipe>...).

Level "images" (pre-processing -> processing):
- the test images (CLEAN) of every case, and those of the background removal test (the cylinder with reflections),
  with a few recipes: that the chain runs and gives what is expected;
- the same flows rendered harder (HARD: dim particles of varied brightness and size, noise, an uneven light sheet,
  reflections), for the comparison of all the recipes: which pre-processing helps the correlation (RMS error, valid
  vectors, peak ratio, outliers) and the tracking (particles tracked, wrong links, error of the links, track
  lengths) the most.

Level "chain" (pre-processing -> processing -> post): the test images with the background removal (1 mode),
processed by the PIV and the PTV, with and without SPICY, and post-processed (PostProcessing, the default recipe):
the statistics against the exact ones (post_level.post_metrics). The measurement of the background removal test runs
with the background removal on and off (its single case through the whole chain).

The images are cached (IMAGE_CACHE: rendered once, ~25 s for the turbulence): the runs are deterministic.

    from velocihub.benchmark import run_benchmark
    result = run_benchmark(["cylinder"], levels=("images", "chain"))
    result.cases[0].metrics["images"]["hard_pod1_clahe"]["piv"]["rms"]
"""

import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from ..fields import load_vectors
from ..piv.displacements import pair_particles, read_tracks
from ..postprocessing import PostProcessing, PostSettings
from .flows import FLOWS, HEIGHT
from .images import BACKGROUND_CASE, CLEAN, HARD, frames_of, mask_polygons, render
from .measurement import CAMERA
from .post_level import NU, post_metrics
from .regression_level import PARALLEL, field_metrics, rms

# Where the images are kept between the runs (rendered again when their parameters change)
IMAGE_CACHE = os.path.join(os.path.expanduser("~"), "VelociHub", "benchmarks", "images")
IMAGE_VERSION = 1
# The PIV settings (64 then 32 px windows, 50 % overlap: the 16 px grid of the cases); the PTV with its defaults
PIV_SETTINGS = {"windows": (64, 32), "overlap": 50}
ENGINES = {"piv": "PaIRS-UniNa", "ptv": "TracTrac"}
# A displacement error beyond this is an outlier of the PIV (px), a link beyond this a wrong link of the PTV (px)
OUTLIER_PX, WRONG_LINK_PX = 0.5, 1.0


@dataclass(frozen=True)
class Recipe:
    """A pre-processing: the POD modes removed (0: no background removal) and the contrast filters (applied after
    the background removal, with their default settings)."""

    key: str
    name: str
    description: str
    modes: int = 0
    filters: tuple = ()


RECIPES = {r.key: r for r in (
    Recipe("none", "None", "No pre-processing (the mask of the body only)."),
    Recipe("pod1", "POD, 1 mode", "The background removal: the first POD mode (about the mean image) removed.",
           modes=1),
    Recipe("default", "Default (POD 6 modes + capping)",
           "The default of the interface: 6 POD modes removed, then intensity capping (n = 10).",
           modes=6, filters=("capping",)),
    Recipe("pod1_capping", "POD 1 + capping", "1 POD mode removed, then intensity capping (n = 10).",
           modes=1, filters=("capping",)),
    Recipe("pod1_minmax", "POD 1 + min-max", "1 POD mode removed, then the local min-max normalization.",
           modes=1, filters=("minmax",)),
    Recipe("pod1_clahe", "POD 1 + CLAHE", "1 POD mode removed, then CLAHE (contrast-limited adaptive histogram "
                                          "equalization).", modes=1, filters=("clahe",)),
    Recipe("pod1_edge", "POD 1 + edges", "1 POD mode removed, then the edge enhancement.", modes=1,
           filters=("edge",)),
    Recipe("pod1_gaussian", "POD 1 + Gaussian matching", "1 POD mode removed, then the Gaussian kernel matching "
                                                         "(correlation with a Gaussian particle).", modes=1,
           filters=("gaussian",)),
)}
# The recipes of the test images (the chain checked), of the hard images (all of them compared)
CLEAN_RECIPES = ("none", "pod1", "default")
HARD_RECIPES = tuple(RECIPES)
# The flows rendered hard, and their frames (fewer for the turbulence: the comparison needs no long record)
HARD_FLOWS = ("cylinder", "jet", "turbulence")
HARD_FRAMES = {True: 41, False: 61}
# The recipe of the chain (1 mode: the default 6 modes remove the particles of single images of these short
# sequences, a known issue of the level images), and those of the background removal on and off
CHAIN_RECIPE = "pod1"
BACKGROUND_ON_OFF = {"on": "pod1", "off": "none"}

# The scenarios of the levels (report), the methods
IMAGE_SCENARIOS = {f"{style.key}_{r}": f"{style.name}: {RECIPES[r].name}"
                   for style, recipes in ((CLEAN, CLEAN_RECIPES), (HARD, HARD_RECIPES)) for r in recipes}
CHAIN_SCENARIOS = {"clean": "Test images, background removal (1 mode)",
                   "on": "Background removal on (1 mode)", "off": "Background removal off"}
IMAGE_METHODS = (("piv", "PIV (PaIRS-UniNa)"), ("ptv", "PTV (TracTrac, binned)"))
CHAIN_METHODS = IMAGE_METHODS + (("piv_spicy", "PIV + SPICY"), ("ptv_spicy", "PTV + SPICY"))
# The metrics of the chain kept from the post-processing (post_level.post_metrics) and those of the fields
CHAIN_KEYS = ("positions_error", "units_error", "mean_vs_samples", "mean_error", "rms", "rms_u_ratio", "rms_v_ratio",
              "tke_ratio", "uv_error", "skewness", "kurtosis", "vort_mean_error", "vort_mean_error_interior",
              "gradient_error", "lx_u_ratio", "ly_u_ratio", "lx_v_ratio", "ly_v_ratio", "t_ratio",
              "dissipation_grid_ratio", "dissipation_ratio", "outliers_removed")


@dataclass(frozen=True)
class Measurement:
    """Images of a flow to run: name (of its folder), the flow, the style of its images, their seed and frames,
    with the reflections or not."""

    name: str
    flow_key: str
    style: object
    seed: int
    frames: int
    reflections: bool = False

    def flow(self):
        return FLOWS[self.flow_key]()


@dataclass(frozen=True)
class Run:
    """A run of a measurement: its recipe, the method (piv, ptv) and SPICY after it or not."""

    recipe: str
    method: str
    spicy: bool = False

    @property
    def folder(self):
        return f"{self.method.upper()}_{self.recipe}" + ("_spicy" if self.spicy else "")


def clean_measurement(key):
    """The test images of a case (those of the test workspace: the same seed)."""
    flow = FLOWS[key]()
    return Measurement(key, key, CLEAN, list(FLOWS).index(key), frames_of(flow))


def background_measurement():
    """The images of the background removal test (the cylinder with reflections)."""
    return Measurement(BACKGROUND_CASE, "cylinder", CLEAN, 99, frames_of(FLOWS["cylinder"]()), reflections=True)


def hard_measurement(key):
    flow = FLOWS[key]()
    return Measurement(key, key, HARD, 100 + list(FLOWS).index(key), HARD_FRAMES[flow.steady])


def cached_images(m, cache=None):
    """The images of a measurement, rendered in the cache unless they are there with the same parameters."""
    folder = os.path.join(cache or IMAGE_CACHE, m.style.key, m.name)
    stamp = {"version": IMAGE_VERSION, "flow": m.flow_key, "seed": m.seed, "frames": m.frames,
             "reflections": m.reflections, "style": asdict(m.style)}
    stamp = json.loads(json.dumps(stamp))  # Tuples as lists, as read back
    paths = [os.path.join(folder, f"img_{k:04d}.tif") for k in range(m.frames)]
    file = os.path.join(folder, "render.json")
    try:
        with open(file, encoding="utf-8") as f:
            if json.load(f) == stamp and all(os.path.isfile(p) for p in paths):
                return paths
    except (OSError, ValueError):
        pass
    shutil.rmtree(folder, ignore_errors=True)
    paths, _ = render(m.flow(), folder, frames=m.frames, seed=m.seed, reflections=m.reflections, style=m.style)
    with open(file, "w", encoding="utf-8") as f:
        json.dump(stamp, f, indent=1)
    return paths


def link_images(paths, folder):
    """The images in a measurement folder (hard links when possible: fast, read-only use)."""
    os.makedirs(folder, exist_ok=True)
    for p in paths:
        target = os.path.join(folder, os.path.basename(p))
        if not os.path.isfile(target):
            try:
                os.link(p, target)
            except OSError:
                shutil.copy2(p, target)


def run_measurement(m, runs, folder, report, tick=None):
    """The runs of a measurement in its folder, one after the other, through the API (Project): {run: {"file": its
    vectors.npz or None, "status", "message", "seconds"}}. tick(): after each run."""
    return run_recipes(folder, f"{m.style.key}_{m.name}", mask_polygons(m.flow()), runs, report, tick,
                       label=f"Images: {m.style.name.lower()} of {m.name}")


def run_recipes(folder, name, polygons, runs, report, tick=None, label="Images", piv_settings=None,
                ptv_settings=None, export_format=None):
    """The runs of the images of a folder (its project: name; the masks: polygons), one after the other, through
    the API (Project) as Run step > Singular of the page Velocimetry: {run: {"file": its vectors.npz or None,
    "status", "message", "seconds"}}. piv_settings, ptv_settings: of the engines (PIV_SETTINGS and the defaults of
    TracTrac by default); export_format: of the pre-processed images (those of the images by default). tick(): after
    each run."""
    from .. import batch, piv
    from ..api import Project

    project = Project(folder)
    project.set_name(name)
    out = {}
    for run in runs:
        recipe = RECIPES[run.recipe]
        report(f"{label}, {recipe.name}, {ENGINES[run.method]}" + (" + SPICY" if run.spicy else ""))
        t0 = time.time()
        try:
            project.set_preprocessing(masks=polygons, filters=list(recipe.filters), stage="post",
                                      modes_removed=recipe.modes, export_folder=f"PreProcessing_{recipe.key}",
                                      export_format=export_format)
            extra = ((piv_settings or PIV_SETTINGS) if run.method == "piv" else (ptv_settings or {}))
            project.set_piv(engine=ENGINES[run.method], regression="spicy" if run.spicy else "none",
                            output_folder=run.folder, **extra)
            project.save()
            # As Project.run_piv (Run step > Singular of the page Velocimetry), quiet
            settings = project.piv
            engine = piv.get_engine(settings["engine"], **(settings.get("options") or {}))
            result = batch.process_folder(folder, engine=engine, settings=settings.get("settings") or {},
                                          preprocessing=project.preprocessing,
                                          output_folder=settings.get("output_folder"),
                                          regression=settings.get("regression"), report=lambda message: None,
                                          verbose=False)
            file = os.path.join(folder, run.folder, "vectors.npz")
            out[run] = {"file": file if os.path.isfile(file) else None, "status": result.status,
                        "message": getattr(result, "message", ""), "seconds": time.time() - t0}
        except Exception as e:  # Reported with the run, the others go on
            out[run] = {"file": None, "status": "failed", "message": f"{type(e).__name__}: {e}",
                        "seconds": time.time() - t0}
        if tick is not None:
            tick()
    return out


################################################################################
# Metrics


def exact_fields(flow, X, Y, n):
    """The exact fields of the pairs (k, k + 1) at the grid points (px, Y up): at the middle of each pair."""
    times = [k + 0.5 for k in range(n)]
    ex = [flow.velocity(X, Y, t) for t in times]
    return ({"U": np.array([e[0] for e in ex]), "V": np.array([e[1] for e in ex]),
             "W": np.array([flow.vorticity(X, Y, t) for t in times])}, times)


def image_metrics(flow, path):
    """The results of a PIV or a PTV of the images of a flow against its exact field: the field metrics
    (regression_level.field_metrics) and the outliers (% beyond OUTLIER_PX); for a PIV, its valid vectors (%) and
    median peak ratio; for a PTV, the particles linked per pair, the wrong links (% beyond WRONG_LINK_PX), the RMS
    error of the right links, the lengths of the tracks (frames), the particles detected per frame and the yield."""
    f = load_vectors(path, pixels=True)
    X, Y, status = f.x.astype(np.float64), f.y.astype(np.float64), f.status
    U, V = f.u_px.astype(np.float64), f.v_px.astype(np.float64)
    exact, _ = exact_fields(flow, X, Y, f.n)
    m = field_metrics(flow, X, Y, status, U, V, exact)
    flowing = status == 0
    err = np.hypot(U - exact["U"], V - exact["V"])[:, flowing]
    with np.errstate(invalid="ignore"):
        m["outliers"] = float(100 * np.mean(err > OUTLIER_PX))
    if getattr(f, "valid", None) is not None and not f.tracks:
        m["valid"] = float(100 * np.mean(np.asarray(f.valid)[:, flowing]))
    if getattr(f, "peak_ratio", None) is not None:
        sn = np.asarray(f.peak_ratio, np.float64)[:, flowing]
        sn = sn[np.isfinite(sn)]
        m["peak_ratio"] = float(np.median(sn)) if sn.size else np.nan
    if f.tracks and os.path.isfile(f.tracks):
        T = read_tracks(f.tracks)
        counts, errors = [], []
        for k in range(f.n):
            pp = pair_particles(T, k)
            if pp is None or not pp["linked"]:
                counts.append(0)
                continue
            eu, ev = flow.velocity(pp["x"], HEIGHT - pp["y"], k + 0.5)
            e = np.hypot(pp["dx"] - eu, pp["dy"] - ev)
            counts.append(e.size)
            errors.append(e)
        e = np.concatenate(errors) if errors else np.zeros(0)
        m["links_per_pair"] = float(np.median(counts)) if counts else 0.0
        m["wrong_links"] = float(100 * np.mean(e > WRONG_LINK_PX)) if e.size else np.nan
        m["link_rms"] = rms(e[e <= WRONG_LINK_PX])
        _, lengths = np.unique(T["id"], return_counts=True)
        m["track_length_median"] = float(np.median(lengths)) if lengths.size else 0.0
        m["track_length_mean"] = float(np.mean(lengths)) if lengths.size else 0.0
        with np.load(f.tracks) as z:
            detected, tracked = np.asarray(z["detected"], np.float64), np.asarray(z["tracked"], np.float64)
        m["detected_per_frame"] = float(np.median(detected)) if detected.size else 0.0
        m["yield"] = float(100 * tracked.sum() / max(detected.sum(), 1))
    return m


def chain_metrics(flow, case, path):
    """The results of a run post-processed (load_vectors with the camera, PostProcessing with its default recipe)
    against the exact statistics (post_level.post_metrics, no noise added: the errors of the measurement count as
    fluctuations), and the RMS error of its fields (field_rms, px)."""
    fields = load_vectors(path, camera=CAMERA)
    post = PostProcessing(fields, PostSettings(nu=NU))
    times = [k + 0.5 for k in range(fields.n)]
    full = post_metrics(flow, case, fields, post, 0.0, times)
    m = {k: full[k] for k in CHAIN_KEYS if k in full}
    m["field_rms"] = image_metrics(flow, path)["rms"]
    return m


################################################################################
# The levels


def plan(cases, levels, background):
    """The measurements and their runs for the levels ("images", "chain") of the cases (keys), with the background
    removal test (its images) or not: [(Measurement, [Run])]."""
    out = []
    images, chain = "images" in levels, "chain" in levels
    for key in cases:
        runs = []
        if images:
            runs += [Run(r, method) for r in CLEAN_RECIPES for method in ENGINES]
        if chain:
            runs += [Run(CHAIN_RECIPE, method, spicy) for spicy in (False, True) for method in ENGINES]
        out.append((clean_measurement(key), list(dict.fromkeys(runs))))
    if background:
        recipes = CLEAN_RECIPES if images else tuple(BACKGROUND_ON_OFF.values())
        out.append((background_measurement(), [Run(r, method) for r in recipes for method in ENGINES]))
    if images:
        out += [(hard_measurement(key), [Run(r, method) for r in HARD_RECIPES for method in ENGINES])
                for key in cases if key in HARD_FLOWS]
    return out


def run_images(cases, levels, work, report, parallel=PARALLEL, background=True, cache=None, progress=None):
    """The image levels ("images", "chain" in levels) of the cases (CaseResult: their metrics filled,
    metrics["images"] {"<style>_<recipe>": {method: {metric}}} and metrics["chain"] {"clean": {method: {metric}}},
    and runs: {measurement: {run folder: record}}), and of the background removal test (background: its metrics
    {"images": ..., "chain": {"on"/"off": ...}} and runs returned). progress(level, done, total): the runs done (as
    the level images, or chain when it runs alone)."""
    lock = threading.Lock()
    progress = progress or (lambda level, done, total: None)
    level = "images" if "images" in levels else "chain"
    count = [0]

    def say(text):
        with lock:
            report(text)

    def tick():
        with lock:
            count[0] += 1
            progress(level, count[0], total)

    jobs = plan([r.case.key for r in cases], levels, background)
    folders = {}
    for m, _ in jobs:
        say(f"Images: {m.style.name.lower()} of {m.name} ({m.frames} frames)")
        folder = os.path.join(work, "images", m.style.key, m.name)
        link_images(cached_images(m, cache), folder)
        folders[m] = folder
    total = sum(len(r) for _, r in jobs) + 1  # The metrics at the end: the last step
    progress(level, 0, total)
    say(f"Images: {total - 1} runs of the PIV/PTV, {parallel} measurements at once")
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = {m: pool.submit(run_measurement, m, runs, folders[m], say, tick) for m, runs in jobs}
        done = {m: f.result() for m, f in futures.items()}

    from .cases import CASES

    def records(m):
        return {run.folder: {**rec, "recipe": run.recipe, "method": run.method, "spicy": run.spicy}
                for run, rec in done[m].items()}

    for r in cases:
        flow = r.flow
        for m in [m for m, _ in jobs if m.name == r.case.key]:
            r.runs[f"{m.style.key}"] = records(m)
            for run, rec in done[m].items():
                if rec["file"] is None:
                    continue
                if "images" in levels and not run.spicy and (m.style is HARD or run.recipe in CLEAN_RECIPES):
                    say(f"Metrics: {m.name}, {m.style.key} images, {run.folder}")
                    r.metrics.setdefault("images", {}).setdefault(f"{m.style.key}_{run.recipe}", {})[run.method] = (
                        image_metrics(flow, rec["file"]))
                if "chain" in levels and m.style is CLEAN and run.recipe == CHAIN_RECIPE:
                    name = run.method + ("_spicy" if run.spicy else "")
                    say(f"Post-processing: {m.name}, {name}")
                    r.metrics.setdefault("chain", {}).setdefault("clean", {})[name] = chain_metrics(
                        flow, r.case, rec["file"])
    bg = None
    if background:
        m = next(m for m, _ in jobs if m.name == BACKGROUND_CASE)
        flow = m.flow()
        bg = {"metrics": {}, "runs": records(m)}
        for run, rec in done[m].items():
            if rec["file"] is None:
                continue
            if "images" in levels:
                bg["metrics"].setdefault("images", {}).setdefault(f"clean_{run.recipe}", {})[run.method] = (
                    image_metrics(flow, rec["file"]))
            if "chain" in levels:
                for scenario, recipe in BACKGROUND_ON_OFF.items():
                    if run.recipe == recipe:
                        say(f"Post-processing: background removal {scenario}, {run.method}")
                        bg["metrics"].setdefault("chain", {}).setdefault(scenario, {})[run.method] = chain_metrics(
                            flow, CASES["cylinder"], rec["file"])
    return bg


def ranking(metrics, style="hard"):
    """The recipes of the images of a style ranked for each metric: {method: {metric: [recipe keys, best first]}}
    (lower is better for the errors, outliers and wrong links; higher for the others)."""
    lower = {"rms", "rms_interior", "outliers", "wrong_links", "link_rms", "vort_rms", "p99"}
    out = {}
    for method, _ in IMAGE_METHODS:
        rows = {s[len(style) + 1:]: v[method] for s, v in metrics.items()
                if s.startswith(style + "_") and method in v}
        keys = {k for v in rows.values() for k in v}
        out[method] = {}
        for k in keys:
            vals = [(r, v.get(k)) for r, v in rows.items() if v.get(k) is not None and np.isfinite(v.get(k))]
            vals.sort(key=lambda rv: rv[1] if k in lower else -rv[1])
            out[method][k] = [r for r, _ in vals]
    return out
