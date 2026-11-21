"""The benchmark at the level of the processing (without images): each case (cases.py) sampled as a PTV and a PIV
would give it, run through the code of VelociHub that follows the PIV software, saved as its results and
post-processed, and compared with the exact field.

For each pair of a scenario, particles are seeded at random in the flow (not in its body); each link is at the middle
of its two positions with the exact displacement there plus the noise of the scenario (and its wrong links). They are
binned on the grid by the binning of the PTV (ptv_worker.bin_links, with its median test) and written as the PTV writes
them (raw.npz, links.npz) in the work folder of a PIVRun of the measurement (two blank images, the camera of the
cases, the body masked); the PIV vectors are the exact field at the grid points plus the noise. The regressions are
run by start_regression, in their own processes as after a PIV/PTV (spicy_worker.py), several at once, and the
results saved by PIVRun.save (vectors.npz: PTV with SPICY, PTV_binned without, PIV with SPICY). The fields are read
back by load_vectors (in pixels) and compared with the exact field in the frame of the post-processing (Y upwards),
the vorticity computed as the post-processing computes it (PostProcessing.vorticity_of); then post-processed
(load_vectors with the camera, PostProcessing): the mean field of the post-processing against the fields read and
the exact one.

    from velocihub.benchmark import run_benchmark
    result = run_benchmark(["cylinder"])          # all the cases and levels by default
    result.cases[0].metrics["processing"]["nominal"]["spicy_ptv"]["rms"]
    result.failures()                              # the checks failed (the known issues apart)
"""

import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from ..fields import load_vectors
from ..piv import ptv_worker
from ..piv.regression import start_regression
from ..postprocessing import PostProcessing, PostSettings
from .cases import METHODS
from .flows import HEIGHT, WIDTH
from .measurement import MS_PER_PX, PER_S, CAMERA, blank_images, exact_image, grid, piv_run
from .uncertainty import predict, uncertainty_metrics

# Regressions run at once (each in its own process)
PARALLEL = max(1, min(6, (os.cpu_count() or 2) // 2))
# The pairs of an unsteady flow: frames apart (independent snapshots of the turbulence)
SNAPSHOT_FRAMES = 200


@dataclass
class CheckResult:
    """A check (cases.Check) evaluated: the value of its metric, its limit (resolved), passed or not."""

    check: object
    value: float
    limit: object
    passed: bool

    @property
    def status(self):
        """pass, fail, known (a known issue failing, as expected), fixed (a known issue passing: no longer one) or
        missing (the metric could not be computed)."""
        if self.value is None or not np.isfinite(self.value):
            return "missing"
        if self.check.known_issue:
            return "fixed" if self.passed else "known"
        return "pass" if self.passed else "fail"


@dataclass
class ScenarioData:
    """The data and the results of a scenario of a case at the processing level: the fields of each method (U, V:
    pairs x rows x columns, in the frame of the post-processing, NaN where there is none), what SPICY reported, the
    files of the results, the links of each bin (particles: of the PTV methods) and the predicted uncertainty
    (uncertainty.py: {method: (errors, uncertainties)} of the components, for the report)."""

    fields: dict = field(default_factory=dict)
    spicy: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    particles: dict = field(default_factory=dict)
    uncertainty: dict = field(default_factory=dict)


@dataclass
class CaseResult:
    """The results of a case: X, Y, status of its grid (px, rows upwards), the exact fields of its pairs (processing
    level), the data of each scenario (processing level), the post-processings of the post level ({scenario:
    (fields, post)}), the metrics ({level: {scenario: {method: {metric: value}}}}) and the checks."""

    case: object
    flow: object
    X: np.ndarray = None
    Y: np.ndarray = None
    status: np.ndarray = None
    exact: dict = None
    data: dict = field(default_factory=dict)
    post: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)
    # The runs of the PIV/PTV on its images (image_level.py): {style: {run folder: record}}
    runs: dict = field(default_factory=dict)

    def failures(self):
        return [c for c in self.checks if c.status in ("fail", "fixed", "missing")]


@dataclass
class BackgroundResult:
    """The background removal test (images.run_background): its data (None if its pre-processing level didn't
    run), metrics ({scenario: {"pod": {metric}}}) and checks; the metrics of its images through the PIV/PTV and the
    post-processing (levels: {level: {scenario: {method: {metric}}}}, image_level.py) and their runs."""

    data: dict
    metrics: dict
    checks: list
    levels: dict = field(default_factory=dict)
    runs: dict = field(default_factory=dict)

    def failures(self):
        return [c for c in self.checks if c.status in ("fail", "fixed", "missing")]


@dataclass
class BenchmarkResult:
    """The cases (CaseResult) and the other tests (extra: {"background": BackgroundResult})."""

    cases: list
    environment: dict
    seconds: float
    levels: tuple = ("preprocessing", "post", "processing", "images", "chain")
    extra: dict = field(default_factory=dict)

    def all_checks(self):
        out = [(r.case.key, c) for r in self.cases for c in r.checks]
        for key, part in self.extra.items():
            out += [(key, c) for c in part.checks]
        return out

    def failures(self):
        return [(key, c) for key, c in self.all_checks() if c.status in ("fail", "fixed", "missing")]

    def counts(self):
        counts = {"pass": 0, "fail": 0, "known": 0, "fixed": 0, "missing": 0}
        for _, c in self.all_checks():
            counts[c.status] += 1
        return counts


def environment():
    """What the results depend on: versions, platform, date."""
    from .. import credits

    try:
        from importlib.metadata import version
        spicy = version("spicy_vki")
    except Exception:
        spicy = None
    commit = None
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import subprocess
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        out = subprocess.run(["git", "-C", root, "log", "-1", "--format=%h %s"], capture_output=True, text=True,
                             timeout=5, creationflags=flags)
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"], capture_output=True, text=True,
                               timeout=5, creationflags=flags).stdout.strip()
        if out.returncode == 0:
            commit = out.stdout.strip() + (" (with uncommitted changes)" if dirty else "")
    except Exception:
        pass
    return {"velocihub": credits.__version__, "commit": commit, "spicy_vki": spicy, "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


################################################################################
# Data of the cases


def pair_times(case, flow):
    """The time of each pair (frames, the middle of the pair): 0 for a steady flow, snapshots SNAPSHOT_FRAMES apart
    otherwise."""
    return [0.0 if flow.steady else k * SNAPSHOT_FRAMES + 0.5 for k in range(case.pairs)]


def sample_pair(case, flow, scenario, rng, t):
    """Positions of the particles of a pair at its two frames (a, b: N x 2, image coordinates): links at the middle
    of the two positions, the exact displacement there plus the noise of the scenario, some links wrong."""
    n = int(round(case.seeding * WIDTH * HEIGHT))
    m = np.column_stack([rng.uniform(0, WIDTH, n), rng.uniform(0, HEIGHT, n)])
    m = m[~flow.solid(m[:, 0], HEIGHT - m[:, 1])]
    u, v = exact_image(flow, m[:, 0], m[:, 1], t)
    d = np.column_stack([u, v]) + rng.normal(0, scenario.noise_px, (len(m), 2))
    if scenario.outliers:
        wrong = rng.random(len(m)) < scenario.outliers
        d[wrong] = rng.uniform(-scenario.outlier_px, scenario.outlier_px, (int(wrong.sum()), 2))
    return m - d / 2, m + d / 2


def prepare(case, flow, scenario, folder, regression, scenario_index):
    """The data of a scenario of a case written as the engines write them in the work folders of the PIVRuns of a
    measurement (folder): PTV (with SPICY), PTV_binned (saved at once), PIV (with SPICY). Returns the jobs of the
    regressions ({"name", "run", "particles", "status"}) and the files already saved ({method: vectors.npz})."""
    images = blank_images(folder)
    xs, ys = grid(case.spacing)
    xpx, ypx = np.meshgrid(xs, ys)
    times = pair_times(case, flow)
    rng = np.random.default_rng([case.seed, scenario_index])
    binning = {"x": xs.tolist(), "y": ys.tolist(), "spacing": float(case.spacing)}
    U, V, INFO, COUNT, links = [], [], [], [], []
    for k, t in enumerate(times):
        a, b = sample_pair(case, flow, scenario, rng, t)
        u, v, info, count = ptv_worker.bin_links(a, b, binning, case.binning)
        U.append(u), V.append(v), INFO.append(info), COUNT.append(count)
        links.append(ptv_worker.pair_links(k, a, b))
    raw = dict(xpx=xpx.astype(np.float32), ypx=ypx.astype(np.float32), u=np.array(U), v=np.array(V),
               info=np.array(INFO), count=np.array(COUNT), failed=np.array([], int))
    ptv_settings = {"spacing": case.spacing, **case.binning}
    jobs, files = [], {}
    # The PTV without regression: saved at once
    binned = piv_run(folder, images, "tractrac", os.path.join(folder, "PTV_binned"), case.pairs, flow,
                     settings=ptv_settings)
    os.makedirs(binned.work, exist_ok=True)
    np.savez(os.path.join(binned.work, "raw.npz"), **raw)
    files["binned"] = binned.save()
    # The PTV with SPICY on its links
    ptv = piv_run(folder, images, "tractrac", os.path.join(folder, "PTV"), case.pairs, flow, regression=regression,
                  settings=ptv_settings)
    os.makedirs(ptv.work, exist_ok=True)
    np.savez(os.path.join(ptv.work, "raw.npz"), **raw)
    np.savez(os.path.join(ptv.work, "links.npz"), **{q: np.concatenate([lk[q] for lk in links])
                                                     for q in ptv_worker.LINKS})
    jobs.append({"name": "spicy_ptv", "run": ptv, "particles": True})
    if scenario.piv:
        solid = flow.solid(xpx, HEIGHT - ypx)
        noise = rng.normal(0, scenario.noise_px, (2, case.pairs) + xpx.shape)
        pu, pv = np.empty(noise.shape[1:]), np.empty(noise.shape[1:])
        for k, t in enumerate(times):
            u, v = exact_image(flow, xpx, ypx, t)
            pu[k], pv[k] = u + noise[0, k], v + noise[1, k]
        pu, pv = (np.where(solid, np.nan, q).astype(np.float32) for q in (pu, pv))
        info = np.broadcast_to(np.where(solid, 0, 1).astype(np.int16), pu.shape).copy()
        piv = piv_run(folder, images, "pairs", os.path.join(folder, "PIV"), case.pairs, flow, regression=regression)
        os.makedirs(piv.work, exist_ok=True)
        np.savez(os.path.join(piv.work, "raw.npz"), xpx=raw["xpx"], ypx=raw["ypx"], u=pu, v=pv, info=info,
                 failed=np.array([], int))
        jobs.append({"name": "spicy_piv", "run": piv, "particles": False})
    for job in jobs:
        job["status"] = job["run"].raw_status()
    return jobs, files


################################################################################
# Regressions


def run_regressions(jobs, report=None, parallel=PARALLEL):
    """The regressions of the jobs run by start_regression, parallel at once, then saved by their PIVRun. Each job
    gets "file" (vectors.npz), "seconds" or "error"."""
    pending, running = list(jobs), []
    total, finished = len(jobs), 0
    while pending or running:
        while pending and len(running) < parallel:
            job = pending.pop(0)
            run = job["run"]
            try:
                job["process"] = start_regression(run.work, run.regression, job["status"].tolist(), run.width,
                                                  job["label"], particles=job["particles"])
                running.append(job)
            except Exception as e:
                job["error"] = str(e)
                finished += 1
        for job in list(running):
            for message in job["process"].poll():
                if message["type"] == "done":
                    job["done"] = message
                elif message["type"] in ("error", "stopped"):
                    job["error"] = message.get("text", message["type"])
            if job["process"].running:
                continue
            job["process"].wait()
            running.remove(job)
            finished += 1
            if "error" not in job:
                try:
                    done = job.get("done") or {}
                    job["seconds"] = done.get("seconds")
                    job["file"] = job["run"].save(done={"seconds": done.get("seconds"),
                                                        "regression_seconds": done.get("seconds")})
                except Exception as e:
                    job["error"] = f"the results could not be saved: {type(e).__name__}: {e}"
            if report:
                report(f"SPICY {finished}/{total}: {job['label']}" + (f" (error: {job['error']})"
                                                                    if "error" in job else ""))
        time.sleep(0.05)
    return jobs


################################################################################
# Metrics


def vorticity(X, Y, U, V):
    """Vorticity of fields on the grid as the post-processing computes it (PostProcessing.vorticity_of, fields in
    pixels)."""
    host = SimpleNamespace(fields=SimpleNamespace(settings={"units": "px"}, x=X, y=Y))
    return PostProcessing.vorticity_of(host, U, V)


def divergence(X, Y, U, V):
    with np.errstate(invalid="ignore"):
        return (np.gradient(np.asarray(U, np.float64), X[0], axis=-1)
                + np.gradient(np.asarray(V, np.float64), Y[:, 0], axis=-2))


def rms(a):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a * a))) if a.size else np.nan


def regions(status):
    """The grid points of the flow (status 0) split into: edge (the outer ring of the grid), body (next to a masked
    point, 8 neighbours) and interior (the others)."""
    flow = status == 0
    edge = np.zeros(status.shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    masked = np.pad(status == 2, 1)
    h, w = status.shape
    near = np.zeros(status.shape, bool)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            near |= masked[1 + i:h + 1 + i, 1 + j:w + 1 + j]
    body = flow & near
    return {"interior": flow & ~edge & ~body, "edge": flow & edge & ~body, "body": body}


def field_metrics(flow, X, Y, status, U, V, exact):
    """The metrics of fields U, V (pairs x rows x columns, NaN where invalid) against the exact field: RMS error of
    the displacement vector (all the flow, interior, edge, body), bias, 99th percentile, coverage, errors of the
    vorticity and the divergence (by differences on the grid, as the post-processing)."""
    flowing = status == 0
    U = np.where(flowing, U, np.nan)
    V = np.where(flowing, V, np.nan)
    eu, ev = U - exact["U"], V - exact["V"]
    mag = np.sqrt(eu**2 + ev**2)
    m = {"rms": rms(mag)}
    for name, where in regions(status).items():
        m[f"rms_{name}"] = rms(mag[:, where]) if where.any() else None
    with np.errstate(invalid="ignore"):
        m["bias_u"], m["bias_v"] = float(np.nanmean(eu)), float(np.nanmean(ev))
        m["p99"] = float(np.nanpercentile(mag, 99)) if np.isfinite(mag).any() else np.nan
    m["coverage"] = float(100 * np.isfinite(mag[:, flowing]).mean())
    W = vorticity(X, Y, U, V)
    m["vort_rms"] = rms(W - exact["W"])
    m["div_rms"] = rms(divergence(X, Y, U, V))
    return m


def column_profiles(Y, U, status, y_wall):
    """The profiles of each column of a field over a wall (Blasius): (eta, U) from the wall (U = 0 there) to the top
    of the grid, the masked points left out; None for a column with a gap."""
    out = []
    for j in range(U.shape[1]):
        keep = status[:, j] == 0
        y, u = Y[keep, j], U[keep, j]
        order = np.argsort(y)
        y, u = np.concatenate([[y_wall], y[order]]), np.concatenate([[0.0], u[order]])
        out.append((y - y_wall, u) if np.isfinite(u).all() else None)
    return out


def specific_metrics(flow, X, Y, status, U, V, exact):
    """The metrics of a flow in particular, of the fields of each pair averaged over the pairs."""
    flowing = status == 0
    per_pair = []
    for k in range(U.shape[0]):
        u, v = np.where(flowing, U[k], np.nan), np.where(flowing, V[k], np.nan)
        m = {}
        if flow.key == "cylinder":
            speed, true = np.sqrt(u**2 + v**2), np.hypot(exact["U"][k], exact["V"][k])
            ok = np.isfinite(speed)
            m["peak_speed_ratio"] = float(speed[ok].max() / true[ok].max()) if ok.any() else np.nan
        elif flow.key == "jet":
            widths, ratios = [], []
            dudy = np.gradient(u, Y[:, 0], axis=0)
            for j in range(u.shape[1]):
                col, y = u[:, j], Y[:, j]
                if not np.isfinite(col).all():
                    continue
                order = np.argsort(y)
                y, col = y[order], col[order]
                half = flow.u_ref / 2
                above = col >= half
                if not above.any():
                    continue
                i0, i1 = np.argmax(above), len(above) - 1 - np.argmax(above[::-1])
                if i0 == 0 or i1 == len(col) - 1:
                    continue
                low = y[i0 - 1] + (half - col[i0 - 1]) / (col[i0] - col[i0 - 1]) * (y[i0] - y[i0 - 1])
                high = y[i1] + (half - col[i1]) / (col[i1 + 1] - col[i1]) * (y[i1 + 1] - y[i1])
                widths.append((high - low) / 2 - flow.h)
                ratios.append(np.nanmax(np.abs(dudy[:, j])) / (flow.u_ref / (2 * flow.delta(X[0, j]))))
            m["half_width_error"] = rms(widths)
            m["shear_ratio"] = float(np.mean(ratios)) if ratios else np.nan
        elif flow.key == "modes":
            ok = np.isfinite(u) & np.isfinite(v)
            basis = [np.concatenate([np.ones(ok.sum()), np.zeros(ok.sum())]),
                     np.concatenate([np.zeros(ok.sum()), np.ones(ok.sum())])]
            for i in range(len(flow.WAVELENGTHS)):
                mu, mv = flow.mode(i, X[ok], Y[ok])
                basis.append(np.concatenate([mu, mv]))
            coef = np.linalg.lstsq(np.column_stack(basis), np.concatenate([u[ok], v[ok]]), rcond=None)[0]
            for i, wl in enumerate(flow.WAVELENGTHS):
                m[f"transfer_{wl:g}"] = float(coef[2 + i] / flow.amplitude)
        elif flow.key == "blasius":
            errs_d, errs_t, ratios = [], [], []
            for j, prof in enumerate(column_profiles(Y, u, status, flow.y_wall)):
                if prof is None:
                    continue
                eta, col = prof
                x = X[0, j]
                # Integrated up to 1.5 delta99 (the deficit is 1e-5 there): beyond, only the noise of the free stream
                top = eta <= 1.5 * flow.delta99(x)
                top[np.argmin(top)] = True  # The first point beyond (the integral reaches the limit)
                eta, col = eta[top], col[top]
                deficit = 1 - col / flow.u_ref
                d_star = np.trapezoid(deficit, eta)
                theta = np.trapezoid(col / flow.u_ref * deficit, eta)
                errs_d.append(100 * (d_star / flow.displacement_thickness(x) - 1))
                errs_t.append(100 * (theta / flow.momentum_thickness(x) - 1))
                g = np.gradient(col, eta)[1]
                _, true, _, _ = flow.gradients(x, flow.y_wall + eta[1])
                ratios.append(g / float(true))
            m["displacement_thickness_error"] = rms(errs_d)
            m["momentum_thickness_error"] = rms(errs_t)
            m["wall_gradient_ratio"] = float(np.mean(ratios)) if ratios else np.nan
        per_pair.append(m)
    keys = per_pair[0].keys() if per_pair else ()
    return {q: float(np.nanmean([p[q] for p in per_pair])) for q in keys}




def post_chain_metrics(path, U, V, exact_mean_U, exact_mean_V, exact_mean_W):
    """The results file of a method post-processed (load_vectors with the camera of the cases, PostProcessing with
    its default recipe): its mean field against the fields read in pixels (U, V: the chain, to the rounding) and
    against the exact mean field; the vorticity of its mean against the exact one."""
    fields = load_vectors(path, camera=CAMERA)
    post = PostProcessing(fields, PostSettings())
    st = post.statistics()
    flowing = st["reliable"]
    mean_u, mean_v = st["mean_U"] / MS_PER_PX, st["mean_V"] / MS_PER_PX
    with np.errstate(invalid="ignore"):
        ru, rv = np.nanmean(U, 0), np.nanmean(V, 0)
    diff = np.maximum(np.abs(mean_u - ru), np.abs(mean_v - rv))[flowing]
    diff = diff[np.isfinite(diff)]
    W = post.vorticity_statistics()["mean"] / PER_S
    return {"post_vs_fields": float(diff.max()) if diff.size else np.nan,
            "post_mean_error": rms(np.hypot(mean_u - exact_mean_U, mean_v - exact_mean_V)[flowing]),
            "post_vort_error": rms((W - exact_mean_W)[flowing])}


################################################################################
# The benchmark


def evaluate_check(check, metrics):
    """A check against the metrics of a level of the case ({scenario: {method: {metric: value}}})."""

    def lookup(scenario, method, metric):
        return metrics.get(scenario, {}).get(method, {}).get(metric)

    value = lookup(check.scenario, check.method, check.metric)
    limit = check.limit
    if isinstance(limit, str):
        scenario, ref = (limit.split(":", 1) if ":" in limit else (check.scenario, limit))
        method, metric = ref.split(".", 1)
        limit = lookup(scenario, method, metric)
        limit = None if limit is None else limit * check.factor
    elif check.op != "within":
        limit = limit * check.factor
    if value is None or limit is None or not np.isfinite(value):
        return CheckResult(check, value, limit, False)
    if check.op == "within":
        passed = limit[0] <= value <= limit[1]
    else:
        passed = {"<=": value <= limit, ">=": value >= limit, "<": value < limit, ">": value > limit}[check.op]
    return CheckResult(check, float(value), limit, bool(passed))


def run_processing(cases, scenarios, work, report, parallel=PARALLEL):
    """The processing level of the cases in their scenarios: their CaseResult filled (X, Y, status, exact, data,
    metrics["processing"])."""
    from ..piv.regression import check_regression

    prepared, jobs = [], []
    for result in cases:
        case, flow = result.case, result.flow
        regression = {"method": "spicy", "settings": case.regression}
        check_regression(regression)
        for index, scenario in enumerate(scenarios):
            report(f"Data: {flow.name}, {scenario.name.lower()}")
            folder = os.path.join(work, case.key, "processing", scenario.key)
            case_jobs, files = prepare(case, flow, scenario, folder, regression, index)
            for job in case_jobs:
                job["label"] = f"{case.key}-{scenario.key}-{job['name']}"
            prepared.append((result, scenario, files))
            jobs += case_jobs
    report(f"Fitting with SPICY: {len(jobs)} regressions, {parallel} at once")
    run_regressions(jobs, report, parallel)
    by_label = {j["label"]: j for j in jobs}
    for result, scenario, files in prepared:
        case, flow = result.case, result.flow
        d = ScenarioData(files=dict(files))
        for name in ("spicy_ptv", "spicy_piv"):
            job = by_label.get(f"{case.key}-{scenario.key}-{name}")
            if job is None:
                continue
            if "error" in job:
                d.errors.append(f"{name}: {job['error']}")
            else:
                d.files[name] = job["file"]
        metrics = result.metrics.setdefault("processing", {}).setdefault(scenario.key, {})
        times = pair_times(case, flow)
        for method, _ in METHODS:
            if method != "exact" and method not in d.files:
                continue
            if method == "exact":
                continue
            path = d.files[method]
            f = load_vectors(path, pixels=True)
            if result.X is None:
                # The grid of the results (px, rows upwards) and the exact fields of the pairs
                result.X, result.Y, result.status = f.x.astype(np.float64), f.y.astype(np.float64), f.status
                ex = [flow.velocity(result.X, result.Y, t) for t in times]
                result.exact = {"U": np.array([e[0] for e in ex]), "V": np.array([e[1] for e in ex]),
                                "W": np.array([flow.vorticity(result.X, result.Y, t) for t in times])}
            U, V = f.u_px.astype(np.float64), f.v_px.astype(np.float64)
            d.fields[method] = (U, V)
            if f.particles is not None:
                d.particles[method] = np.asarray(f.particles)
            with np.load(path) as z:
                if "regression_residual_px" in z.files:
                    d.spicy[method] = {"residual": z["regression_residual_px"], "n_rbf": z["regression_n_rbf"],
                                       "n_points": z["regression_n_points"] if "regression_n_points" in z.files
                                       else np.zeros(1),
                                       "n_rejected": z["regression_n_rejected"] if "regression_n_rejected" in z.files
                                       else np.zeros(1)}
            job = by_label.get(f"{case.key}-{scenario.key}-{method}")
            if method in d.spicy and job is not None:
                d.spicy[method]["seconds"] = job.get("seconds")
        if result.X is None:
            result.data[scenario.key] = d
            continue
        n = case.pairs
        d.fields["exact"] = (result.exact["U"].copy(), result.exact["V"].copy())
        X, Y, status = result.X, result.Y, result.status
        mean_U, mean_V = result.exact["U"].mean(0), result.exact["V"].mean(0)
        mean_W = result.exact["W"].mean(0)
        for method, _ in METHODS:
            if method not in d.fields:
                continue
            U, V = d.fields[method]
            m = field_metrics(flow, X, Y, status, U, V, result.exact)
            m.update(specific_metrics(flow, X, Y, status, U, V, result.exact))
            if method in d.spicy:
                s = d.spicy[method]
                res = np.asarray(s["residual"], np.float64)
                m["residual"] = float(np.nanmedian(res)) if np.isfinite(res).any() else np.nan
                m["residual_ratio"] = m["residual"] / (scenario.noise_px * np.sqrt(2))
                m["n_rbf"] = float(np.mean(s["n_rbf"]))
                m["n_points"] = float(np.mean(s["n_points"]))
                if method == "spicy_ptv":
                    total = float(np.sum(s["n_points"]) + np.sum(s["n_rejected"]))
                    m["links_rejected"] = 100 * float(np.sum(s["n_rejected"])) / total if total else np.nan
                m["seconds_per_map"] = s["seconds"] / n if s.get("seconds") else None
            if method == "binned" and method in d.particles:
                # The uncertainty predicted from the flow and the links of each bin, against the actual error
                pred = predict(flow, X, Y, status, times, d.particles[method], scenario.noise_px, case.spacing)
                um, arrays = uncertainty_metrics(np.where(status == 0, U, np.nan), np.where(status == 0, V, np.nan),
                                                 result.exact, pred)
                m.update(um)
                if arrays is not None:
                    d.uncertainty[method] = {"errors": arrays[0], "total": arrays[1], "pred": pred}
            if method in d.files:
                m.update(post_chain_metrics(d.files[method], U, V, mean_U, mean_V, mean_W))
            metrics[method] = m
        result.data[scenario.key] = d


def run_benchmark(cases=None, scenarios=None, work=None, report=None, parallel=PARALLEL, keep=False,
                  levels=("preprocessing", "post", "processing", "images", "chain"), progress=None):
    """Run the cases (keys of CASES, all by default) at the levels (cases.LEVELS: the pre-processing level is the
    background removal test, not a case; the levels images and chain run the PIV/PTV on particle images,
    image_level.py, with the images of the background removal test when all the cases run) in their scenarios (keys
    of SCENARIOS for the processing level, all by default): report(text) receives the progress. work: the folder of
    the data (a temporary one by default, removed at the end unless keep). progress(level, done, total): the
    progress of each level (0 then 1 of 1 at its start and end; the runs of the PIV/PTV of the image levels counted).
    Returns a BenchmarkResult."""
    import shutil
    import tempfile

    from .cases import BACKGROUND_CHECKS, BACKGROUND_MODES, CASES, SCENARIOS
    from .images import run_background
    from .post_level import run_post

    t0 = time.time()
    report = report or (lambda text: None)
    progress = progress or (lambda level, done, total: None)
    results = [CaseResult(CASES[k], CASES[k].flow()) for k in (cases if cases is not None else list(CASES))]
    own = work is None
    work = work or tempfile.mkdtemp(prefix="velocihub_benchmark_")
    extra = {}
    # The background removal test: its pre-processing level, and its images through the PIV/PTV and the post
    bg_data, bg_levels, bg_runs = None, {}, {}
    try:
        if "preprocessing" in levels:
            progress("preprocessing", 0, 1)
            bg_data = run_background(os.path.join(work, "_background"), BACKGROUND_MODES, report, keep=True)
            bg_levels["preprocessing"] = {f"modes_{n}": {"pod": m} for n, m in bg_data["modes"].items()}
            progress("preprocessing", 1, 1)
        if "post" in levels:
            progress("post", 0, 1)
            metrics, kept = run_post([r.case for r in results], work=os.path.join(work, "_post"), report=report,
                                     keep=True)
            for r in results:
                r.metrics["post"] = metrics[r.case.key]
                r.post = kept[r.case.key]
            progress("post", 1, 1)
        if "processing" in levels:
            progress("processing", 0, 1)
            run_processing(results, [SCENARIOS[k] for k in (scenarios or list(SCENARIOS))], work, report, parallel)
            progress("processing", 1, 1)
        if {"images", "chain"} & set(levels):
            from .image_level import run_images
            bg = run_images(results, levels, work, report, parallel, background=cases is None, progress=progress)
            if bg is not None:
                bg_levels.update(bg["metrics"])
                bg_runs = bg["runs"]
            for level in ("images", "chain"):
                if level in levels:
                    progress(level, 1, 1)
        if bg_levels:
            checks = [evaluate_check(c, bg_levels[c.level]) for c in BACKGROUND_CHECKS
                      if c.level in bg_levels and c.scenario in bg_levels[c.level]]
            extra["background"] = BackgroundResult(bg_data, bg_levels.get("preprocessing", {}), checks, bg_levels,
                                                   bg_runs)
            failed = [c for c in checks if c.status in ("fail", "fixed", "missing")]
            report(f"Background removal: {len(checks) - len(failed)}/{len(checks)} checks as expected")
        for r in results:
            r.checks = [evaluate_check(c, r.metrics.get(c.level, {})) for c in r.case.checks
                        if c.level in r.metrics and c.scenario in r.metrics[c.level]]
            failed = [c for c in r.checks if c.status in ("fail", "fixed", "missing")]
            report(f"{r.flow.name}: {len(r.checks) - len(failed)}/{len(r.checks)} checks as expected")
        return BenchmarkResult(results, environment(), time.time() - t0, tuple(levels), extra)
    finally:
        if own and not keep:
            shutil.rmtree(work, ignore_errors=True)
