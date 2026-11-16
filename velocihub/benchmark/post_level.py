"""The benchmark at the level of the post-processing: the exact fields of each case, with or without noise, written as
the results of a PIV (measurement.write_exact) and post-processed by VelociHub (load_vectors with the camera of the
cases, PostProcessing with its default recipe), without PIV/PTV: the statistics and the derivatives against

- the samples themselves (the mean, the standard deviations and the covariance computed with numpy from the same
  displacements: the verification of the computation, its units and orientation, to the rounding);
- the analytic values (the mean field, the vorticity and the gradient of the flow; for the noise, its standard
  deviation, no Reynolds stress, a skewness of 0 and a kurtosis of 3, the vorticity of differences of white noise;
  for the turbulence, its variances, integral length and time scales and dissipation from its modes).

The steady cases have POST_MAPS maps (the same field plus the noise); the turbulence is time-resolved,
TURBULENCE_MAPS maps of consecutive pairs (the map k at the time k + 1/2 frames)."""

import os
import shutil
import tempfile

import numpy as np
from scipy import ndimage

from ..fields import load_vectors
from dataclasses import replace

from ..postprocessing import PostProcessing, PostSettings, gradient_nan, smooth_nan
from .flows import HEIGHT
from .measurement import CAMERA, MM_PER_PX, MS_PER_PX, PER_S, write_exact

# Maps of the steady cases, of the turbulence (time-resolved)
POST_MAPS, TURBULENCE_MAPS = 200, 2000
# Kinematic viscosity of the post-processing (m²/s, water) and in px² per pair
NU = 1e-6
NU_PX = NU / (MM_PER_PX * 1e-3) ** 2 / CAMERA.frame_rate_hz


def times_of(flow):
    n = POST_MAPS if flow.steady else TURBULENCE_MAPS
    return [k + 0.5 for k in range(n)]


def _median(F, where=None):
    F = np.asarray(F, np.float64)
    if where is not None:
        F = F[where]
    F = F[np.isfinite(F)]
    return float(np.median(F)) if F.size else np.nan


def _rms(F, where=None):
    F = np.asarray(F, np.float64)
    if where is not None:
        F = F[where]
    F = F[np.isfinite(F)]
    return float(np.sqrt(np.mean(F * F))) if F.size else np.nan


def _maxabs(F, where=None):
    F = np.asarray(F, np.float64)
    if where is not None:
        F = F[where]
    F = F[np.isfinite(F)]
    return float(np.abs(F).max()) if F.size else np.nan


def interior(status):
    """The grid points of the flow away from the edges of the grid and from the masks (central differences)."""
    flow = status == 0
    out = flow.copy()
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = False
    masked = np.pad(status != 0, 1)
    h, w = status.shape
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            out &= ~masked[1 + i:h + 1 + i, 1 + j:w + 1 + j]
    return out


# The smoothing of the mean gradient studied (cells), the previous default of VelociHub (before 1.2), the noise of the
# mean field added to see where the smoothing pays (px), the distance from the edges and masks beyond which the
# smoothing is unaffected by them (cells: 3 sigma of the previous default)
SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
PREVIOUS_SIGMA = 1.5
MEAN_NOISES = (0.0, 0.1, 0.2)
FAR_CELLS = 4.5


def gradient_study(flow, case, fields, post):
    """The mean gradient dU/dY of the post-processing split into its parts (steady flows): the differences on the
    grid (the exact field on the grid, central differences: the discretisation), the smoothing (gradient_sigma of
    the settings of post, holes ignored), and the edges and masks (the smoothing and the one-sided differences there).
    The gradient of VelociHub (gradient_fields) recomputed from its mean field, to the rounding (verify), in both
    orders: smoothed then differenced (smoothed), and differenced then smoothed (after: gradient_smooth_after); then
    with each smoothing of SIGMAS, of the exact grid field (both orders) and of the mean field with a noise
    (MEAN_NOISES). All in 1/pair; the errors RMS, in % of the largest exact gradient, over interior (the grid points
    away from the edges and masks by one cell) and far (by FAR_CELLS cells)."""
    if not flow.steady:
        return None
    st = post.statistics()
    flowing = st["reliable"]
    X, Y = fields.x / MM_PER_PX, fields.y / MM_PER_PX
    y = Y[:, 0].astype(np.float64)
    _, G, _, _ = flow.gradients(X, Y)
    G = np.where(flowing, G, np.nan)
    scale = float(np.nanmax(np.abs(G))) if flowing.any() else 0.0
    if not scale > 0:
        return None
    inner = interior(fields.status) & flowing
    dist = ndimage.distance_transform_edt(np.pad(fields.status == 0, 1, constant_values=False))[1:-1, 1:-1]
    far = inner & (dist > FAR_CELLS)
    mean_u = np.where(flowing, st["mean_U"] / MS_PER_PX, np.nan)
    EU, _ = flow.velocity(X, Y)
    EU = np.where(flowing, EU, np.nan)

    def grad(F, sigma):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.gradient(smooth_nan(F, sigma), y, axis=0)

    def grad_after(F, sigma):
        return smooth_nan(gradient_nan(F, y, 0), sigma)

    def err(F, where):
        return 100 * _rms(F - G, where) / scale

    sigma = post.settings.gradient_sigma
    on = post if post.settings.gradient_smooth_after else None
    off = post if on is None else None
    off = off or PostProcessing(fields, replace(post.settings, gradient_smooth_after=False))
    on = on or PostProcessing(fields, replace(post.settings, gradient_smooth_after=True))
    mine, after = grad(mean_u, sigma), grad_after(mean_u, sigma)
    flowing_any = flowing | np.isfinite(after)
    out = dict(X=X, Y=Y, G=G, scale=scale, inner=inner, far=far, sigma=sigma, grid=grad(EU, 0.0), smoothed=mine,
               after=after, previous=grad(mean_u, PREVIOUS_SIGMA),
               verify=_maxabs(mine - off.gradient_fields()["du"] / PER_S, flowing_any),
               verify_after=_maxabs(after - on.gradient_fields()["du"] / PER_S, flowing_any), sweep={})
    rng = np.random.default_rng(case.seed + 7)
    for noise in MEAN_NOISES:
        F = EU + rng.normal(0, noise, EU.shape) if noise else EU
        out["sweep"][noise] = [(err(grad(F, s), inner), err(grad(F, s), far)) for s in SIGMAS]
    out["sweep_after"] = [err(grad_after(EU, s), inner) for s in SIGMAS]
    return out


def wall_gradient(flow, F, X, Y, status):
    """Blasius: dU/dY at the grid points of the flow next to the wall (the first above it in each column), over the
    exact one there: their mean; NaN if none has a value."""
    ratios = []
    for j in range(F.shape[1]):
        rows = np.flatnonzero(status[:, j] == 0)
        if not rows.size:
            continue
        i = rows[np.argmin(Y[rows, j])]
        _, true, _, _ = flow.gradients(X[i, j], Y[i, j])
        if np.isfinite(F[i, j]):
            ratios.append(F[i, j] / float(true))
    return float(np.mean(ratios)) if ratios else np.nan


def gradient_transfer(flow, G, X, Y, where):
    """The amplitude of the dU/dY of each mode (modes flow) kept in a gradient field G: least-squares projection on
    the exact dU/dY of the modes over where."""
    if flow.key != "modes" or not where.any():
        return {}
    basis = []
    for i, wl in enumerate(flow.WAVELENGTHS):
        k = 2 * np.pi / wl
        px, py = flow.PHASES[i]
        basis.append((-k * np.sin(k * X + px) * np.sin(k * Y + py))[where])
    coef = np.linalg.lstsq(np.column_stack(basis), G[where], rcond=None)[0]
    return {wl: float(c / flow.amplitude) for wl, c in zip(flow.WAVELENGTHS, coef)}


def gradient_transfer_theory(wavelength, spacing, sigma):
    """The amplitude of the dU/dY of a mode (wavelength along X and Y, px) kept by the central differences on a grid
    (sin(kh) / kh) and the Gaussian smoothing of sigma cells in X and Y (the transfer of the sampled kernel of
    scipy, truncated at 4 sigma)."""
    kh = 2 * np.pi / wavelength * spacing
    out = np.sin(kh) / kh
    if sigma:
        r = int(4 * sigma + 0.5)
        j = np.arange(-r, r + 1)
        w = np.exp(-j**2 / (2 * sigma**2))
        out *= (np.sum(w * np.cos(kh * j)) / w.sum()) ** 2
    return float(out)


def peak_shear(flow, G, X, Y, status):
    """Jet: the largest |dU/dY| of each shear layer of each column of a gradient field / the exact one (U / 2
    delta), with delta / grid spacing: arrays (delta, ratio)."""
    deltas, ratios = [], []
    for j in range(G.shape[1]):
        col, ok = np.abs(G[:, j]), status[:, j] == 0
        x = X[0, j]
        for upper in (True, False):
            side = ok & ((Y[:, j] > flow.yc) if upper else (Y[:, j] < flow.yc)) & np.isfinite(col)
            if side.any():
                deltas.append(float(flow.delta(x)))
                ratios.append(float(col[side].max() / (flow.u_ref / (2 * flow.delta(x)))))
    return np.array(deltas), np.array(ratios)


def post_metrics(flow, case, fields, post, noise, times):
    """The metrics of a post-processing (post) of exact fields with a noise (px) against the samples and the
    analytic values. All in px and pairs (the physical values divided by the camera's)."""
    st = post.statistics()
    flowing = st["reliable"]
    inner = interior(fields.status) & flowing
    X, Y = fields.x / MM_PER_PX, fields.y / MM_PER_PX  # px from the bottom-left corner
    m = {}
    # The positions: the grid of the case, x right and y up from the bottom-left corner
    xs = np.arange(case.spacing, X.shape[1] * case.spacing + 0.5, case.spacing)
    ys = HEIGHT - np.arange(case.spacing, X.shape[0] * case.spacing + 0.5, case.spacing)[::-1]
    m["positions_error"] = float(max(np.abs(X - xs[None]).max(), np.abs(Y - ys[:, None]).max()))
    # The units: the velocities of the file are its displacements times the camera's
    m["units_error"] = max(_maxabs(fields.U / MS_PER_PX - fields.u_px), _maxabs(fields.V / MS_PER_PX - fields.v_px))
    # The samples: the displacements the statistics are computed from (the outliers removed by the recipe), as numpy
    # computes their statistics
    cu, cv, _ = post.clean()
    u, v = cu.astype(np.float64) / MS_PER_PX, cv.astype(np.float64) / MS_PER_PX
    with np.errstate(invalid="ignore"):
        su, sv = np.nanmean(u, 0), np.nanmean(v, 0)
        ru, rv = np.nanstd(u, 0, ddof=1), np.nanstd(v, 0, ddof=1)
        cuv = np.nansum((u - su) * (v - sv), 0) / (np.isfinite(u).sum(0) - 1)
    mean_u, mean_v = st["mean_U"] / MS_PER_PX, st["mean_V"] / MS_PER_PX
    u_rms, v_rms, uv = st["u_rms"] / MS_PER_PX, st["v_rms"] / MS_PER_PX, st["uv"] / MS_PER_PX**2
    m["mean_vs_samples"] = max(_maxabs(mean_u - su, flowing), _maxabs(mean_v - sv, flowing))
    m["rms_vs_samples"] = max(_maxabs(u_rms - ru, flowing), _maxabs(v_rms - rv, flowing))
    m["uv_vs_samples"] = _maxabs(uv - cuv, flowing)
    m["outliers_removed"] = float(st["outliers"])
    # The analytic values: the time mean of the flow, its fluctuations
    if flow.steady:
        EU, EV = flow.velocity(X, Y)
        var_u = var_v = noise**2
        cov = 0.0
        W = flow.vorticity(X, Y)
        _, dUdY, _, _ = flow.gradients(X, Y)
    else:
        EU, EV = np.full(X.shape, flow.u0), np.zeros(X.shape)
        uu, vv, cov = flow.variances()
        var_u, var_v = uu + noise**2, vv + noise**2
        W, dUdY = np.zeros(X.shape), np.zeros(X.shape)
    m["mean_error"] = _rms(np.hypot(mean_u - EU, mean_v - EV), flowing)
    k = (u_rms**2 + v_rms**2) / 2
    m["rms"] = _median(np.sqrt(k), flowing)
    expected_k = (var_u + var_v) / 2
    if expected_k > 0:
        m["rms_u_ratio"] = _median(u_rms, flowing) / np.sqrt(var_u)
        m["rms_v_ratio"] = _median(v_rms, flowing) / np.sqrt(var_v)
        m["tke_ratio"] = _median(k, flowing) / expected_k
        m["uv_error"] = abs(_median(uv, flowing) - cov) / expected_k
        u_ref = st["u_ref"] / MS_PER_PX
        m["ti_ratio"] = _median(st["ti"], flowing) / (100 * np.sqrt(expected_k) / u_ref)
        ho = post.higher_order()
        m["skewness"] = _median((ho["skewness_u"] + ho["skewness_v"]) / 2, flowing)
        m["kurtosis"] = _median((ho["kurtosis_u"] + ho["kurtosis_v"]) / 2, flowing)
    vs = post.vorticity_statistics()
    m["vort_mean_error"] = _rms(vs["mean"] / PER_S - W, flowing)
    m["vort_mean_error_interior"] = _rms(vs["mean"] / PER_S - W, inner)
    if flow.steady and noise:
        # Central differences of white noise: var(dv/dx) = var(du/dy) = noise² / (2 h²)
        m["vort_rms_ratio"] = _median(vs["rms"] / PER_S, inner) / (noise / case.spacing)
    gf = post.gradient_fields()
    scale = np.nanmax(np.abs(dUdY[flowing])) if flowing.any() else 0
    if scale > 0:
        m["gradient_error"] = 100 * _rms(gf["du"] / PER_S - dUdY, inner) / scale
    g = gradient_study(flow, case, fields, post) if not noise else None
    if g is not None:
        # The parts of the error: the grid alone (no smoothing), and away from the edges and masks
        m["gradient_vs_code"] = g["verify"]
        m["gradient_vs_code_after"] = g["verify_after"]
        m["gradient_error_after"] = 100 * _rms(g["after"] - g["G"], g["inner"]) / g["scale"]
        m["gradient_error_previous"] = 100 * _rms(g["previous"] - g["G"], g["inner"]) / g["scale"]
        m["gradient_error_far"] = 100 * _rms(g["smoothed"] - g["G"], g["far"]) / g["scale"]
        m["gradient_error_grid"] = 100 * _rms(g["grid"] - g["G"], g["inner"]) / g["scale"]
        m["gradient_error_grid_far"] = 100 * _rms(g["grid"] - g["G"], g["far"]) / g["scale"]
        for wl, a in gradient_transfer(flow, g["smoothed"], g["X"], g["Y"], g["far"]).items():
            m[f"gradient_transfer_{wl:g}"] = a
        for wl, a in gradient_transfer(flow, g["grid"], g["X"], g["Y"], g["far"]).items():
            m[f"gradient_transfer_grid_{wl:g}"] = a
        if flow.key == "blasius":
            m["wall_gradient_ratio"] = wall_gradient(flow, g["smoothed"], g["X"], g["Y"], fields.status)
            m["wall_gradient_ratio_after"] = wall_gradient(flow, g["after"], g["X"], g["Y"], fields.status)
        if flow.key == "modes":
            m["gradient_transfer_theory_error"] = max(
                max(abs(m[f"gradient_transfer_{wl:g}"] - gradient_transfer_theory(wl, case.spacing, g["sigma"])),
                    abs(m[f"gradient_transfer_grid_{wl:g}"] - gradient_transfer_theory(wl, case.spacing, 0)))
                for wl in flow.WAVELENGTHS)
        if flow.key == "jet":
            m["peak_shear_ratio"] = float(np.mean(peak_shear(flow, g["smoothed"], g["X"], g["Y"], fields.status)[1]))
            m["peak_shear_ratio_grid"] = float(np.mean(peak_shear(flow, g["grid"], g["X"], g["Y"],
                                                                  fields.status)[1]))
    if not flow.steady:
        metrics = post.metrics()
        for key, axis, comp in (("lx_u", "x", "u"), ("ly_u", "y", "u"), ("lx_v", "x", "v"), ("ly_v", "y", "v")):
            name = f"median_integral_length_{'' if comp == 'u' else 'v'}{axis}_mm"
            value = metrics.get(name)
            m[f"{key}_ratio"] = (value / MM_PER_PX / flow.integral_length(axis, comp)) if value else np.nan
        T = metrics.get("median_integral_time_s")
        m["t_ratio"] = (T * CAMERA.frame_rate_hz / flow.integral_time()) if T else np.nan
        eps = metrics.get("median_dissipation_m2_s3")
        grads = flow.dissipation(NU_PX, case.spacing) * (MM_PER_PX * 1e-3) ** 2 * PER_S**3
        true = flow.dissipation(NU_PX) * (MM_PER_PX * 1e-3) ** 2 * PER_S**3
        if eps and not noise:
            m["dissipation_grid_ratio"] = eps / grads
            m["dissipation_ratio"] = eps / true
    return m


def run_post(cases, work=None, report=None, keep=False):
    """The post-processing level of cases (Case objects): {case key: {scenario: {"post": metrics}}}, and the fields
    of each post-processing ({case key: {scenario: (fields, post)}}) for the report."""
    from .cases import POST_SCENARIOS

    report = report or (lambda text: None)
    own = work is None
    work = work or tempfile.mkdtemp(prefix="velocihub_post_")
    metrics, kept = {}, {}
    try:
        for case in cases:
            flow = case.flow()
            times = times_of(flow)
            metrics[case.key], kept[case.key] = {}, {}
            for scenario in POST_SCENARIOS.values():
                report(f"Post-processing: {flow.name}, {scenario.name.lower()} ({len(times)} maps)")
                folder = os.path.join(work, case.key, "post", scenario.key)
                path = write_exact(folder, flow, case.spacing, times, noise=scenario.noise_px,
                                   seed=case.seed + 1)
                fields = load_vectors(path, camera=CAMERA)
                post = PostProcessing(fields, PostSettings(nu=NU))
                metrics[case.key][scenario.key] = {"post": post_metrics(flow, case, fields, post,
                                                                        scenario.noise_px, times)}
                kept[case.key][scenario.key] = (fields, post)
        return metrics, kept
    finally:
        if own and not keep:
            shutil.rmtree(work, ignore_errors=True)
