"""The report of a benchmark (regression_level.run_benchmark): a folder of HTML pages, a summary page (index.html), a
page per case (a section per level: post-processing, processing → post) and one for the background removal test, with
the tables of the checks and of the metrics and the maps of the fields and of their differences from the exact ones;
results.json holds the numbers (to compare a run with a previous one).

    from velocihub.benchmark import report
    index = report.write_report(result, "benchmark_report", compare="previous_report")
"""

import html
import json
import os
from datetime import datetime

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from .cases import BACKGROUND_SCENARIOS, LEVELS, METHODS, POST_SCENARIOS, SCENARIOS
from .image_level import CHAIN_METHODS, CHAIN_SCENARIOS, HARD_FLOWS, IMAGE_METHODS, IMAGE_SCENARIOS, RECIPES
from .measurement import MM_PER_PX, MS_PER_PX, PER_S

# Colours (the light chart surface, the ink, the categorical series, the status)
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {"spicy_ptv": "#2a78d6", "binned": "#eb6834", "spicy_piv": "#1baf7a", "exact": INK}
ORDINAL = ("#9ec5f4", "#5598e7", "#256abf", "#104281")  # The modes removed (0 to 3): light to dark
NO_DATA = "#c3c2b7"  # Grid points without a value: masked, no particles, invalid
SEQUENTIAL = LinearSegmentedColormap.from_list("vh_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("vh_diverging", ["#104281", "#3987e5", "#9ec5f4", "#f0efec",
                                                               "#f2a39f", "#e34948", "#8f1d1c"])
METHOD_NAMES = dict(METHODS)
SHORT_NAMES = {"exact": "Exact", "spicy_ptv": "SPICY (PTV)", "binned": "Binned (PTV)", "spicy_piv": "SPICY (PIV)",
               "post": "VelociHub", "pod": "POD", **dict(CHAIN_METHODS)}
STATUS = {"pass": ("✓", "Pass"), "fail": ("✗", "Fail"), "known": ("●", "Known issue"),
          "fixed": ("★", "Known issue now passing"), "missing": ("?", "Not computed")}
BACKGROUND_PAGE = "background.html"

# The metrics: label (with unit), factor of the values shown, decimals (None: 3 significant digits)
METRICS = {
    # Processing level
    "rms": ("RMS error of the displacement |Δd| (px)", 1, None),
    "rms_interior": ("— interior (px)", 1, None),
    "rms_edge": ("— outer ring of the grid (px)", 1, None),
    "rms_body": ("— next to the body (px)", 1, None),
    "p99": ("99th percentile of |Δd| (px)", 1, None),
    "bias_u": ("Bias of U (px)", 1, None),
    "bias_v": ("Bias of V (px)", 1, None),
    "coverage": ("Grid points of the flow with a value (%)", 1, 1),
    "vort_rms": ("RMS error of the vorticity ω (10⁻³/pair)", 1e3, None),
    "div_rms": ("RMS divergence (10⁻³/pair; exact: 0)", 1e3, None),
    "residual": ("Residual of the fit at the data (px)", 1, None),
    "residual_ratio": ("Residual / noise of the data", 1, 2),
    "n_points": ("Data fitted per map", 1, 0),
    "links_rejected": ("Links rejected by the outlier test (%)", 1, 2),
    "n_rbf": ("RBFs per map", 1, 0),
    "seconds_per_map": ("Fit time per map (s)", 1, 1),
    "peak_speed_ratio": ("Highest speed / exact", 1, 3),
    "half_width_error": ("RMS error of the half-width (px)", 1, None),
    "shear_ratio": ("Peak shear dU/dY / exact", 1, 3),
    "displacement_thickness_error": ("RMS error of δ* (%)", 1, 2),
    "momentum_thickness_error": ("RMS error of θ (%)", 1, 2),
    "wall_gradient_ratio": ("dU/dY next to the wall / exact", 1, 3),
    "wall_gradient_ratio_after": ("— smoothed after the differences", 1, 3),
    "unc_rms": ("Predicted uncertainty of |Δd|: RMS (px)", 1, None),
    "unc_ratio": ("Predicted uncertainty / actual RMS error", 1, 3),
    "unc_z": ("(Error − predicted bias) / predicted random error: std", 1, 3),
    "unc_cover_1": ("Errors within 1 predicted uncertainty (%)", 1, 1),
    "unc_cover_2": ("Errors within 2 predicted uncertainties (%)", 1, 1),
    "post_vs_fields": ("Post-processed mean − mean of the fields read (px)", 1, None),
    "post_mean_error": ("Post-processed mean: RMS error vs the exact mean (px)", 1, None),
    "post_vort_error": ("Vorticity of the post-processed mean: RMS error (10⁻³/pair)", 1e3, None),
    # Images (pre-processing -> processing) and the chain
    "outliers": ("Outliers: |Δd| beyond 0.5 px (%)", 1, 2),
    "valid": ("Valid vectors of the PIV (%)", 1, 1),
    "peak_ratio": ("Correlation peak ratio, median", 1, 2),
    "links_per_pair": ("Particles linked per pair, median", 1, 0),
    "wrong_links": ("Wrong links: |Δd| beyond 1 px (%)", 1, 2),
    "link_rms": ("RMS error of the right links (px)", 1, None),
    "track_length_median": ("Track length, median (frames)", 1, 0),
    "track_length_mean": ("Track length, mean (frames)", 1, 1),
    "detected_per_frame": ("Particles detected per frame, median", 1, 0),
    "yield": ("Particles tracked / detected (%)", 1, 1),
    "field_rms": ("RMS error of the fields |Δd| (px)", 1, None),
    "spatial_noise": ("Spatial noise: median distance of a vector from the median of its neighbours (px)", 1, None),
    "suspect_links": ("Suspect links: failing the median test against their neighbours (%)", 1, 2),
    "displacement": ("Displacement |d|, median (px)", 1, None),
    # Post-processing level
    "positions_error": ("Positions: largest error (px)", 1, None),
    "units_error": ("Velocities / camera − displacements: largest (px)", 1, None),
    "mean_vs_samples": ("Mean − mean of the samples: largest (px)", 1, None),
    "rms_vs_samples": ("RMS − standard deviation of the samples: largest (px)", 1, None),
    "uv_vs_samples": ("<u'v'> − covariance of the samples: largest (px²)", 1, None),
    "outliers_removed": ("Vectors removed by the outlier filter (4 σ)", 1, 0),
    "mean_error": ("Mean: RMS error vs the exact mean (px)", 1, None),
    "rms": ("RMS of the fluctuations, median (px)", 1, None),
    "rms_u_ratio": ("RMS of u / expected", 1, 3),
    "rms_v_ratio": ("RMS of v / expected", 1, 3),
    "tke_ratio": ("k / expected", 1, 3),
    "ti_ratio": ("Turbulence intensity / expected", 1, 3),
    "uv_error": ("|<u'v'> − expected| / k", 1, None),
    "skewness": ("Skewness, median (expected 0)", 1, 3),
    "kurtosis": ("Kurtosis, median (expected 3)", 1, 3),
    "vort_mean_error": ("Vorticity of the mean: RMS error (10⁻³/pair)", 1e3, None),
    "vort_mean_error_interior": ("— away from the edges and masks (10⁻³/pair)", 1e3, None),
    "vort_rms_ratio": ("RMS of ω / (noise / grid spacing)", 1, 3),
    "gradient_error": ("Mean dU/dY: RMS error (% of its largest value)", 1, 1),
    "gradient_error_far": ("— away from the edges and masks (> 3σ of the smoothing)", 1, 1),
    "gradient_error_grid": ("— the grid alone: exact field, central differences, no smoothing", 1, 1),
    "gradient_error_grid_far": ("— the grid alone, away from the edges and masks", 1, 1),
    "gradient_vs_code": ("Mean dU/dY recomputed − VelociHub's: largest (1/pair)", 1, None),
    "gradient_vs_code_after": ("— smoothed after the differences (1/pair)", 1, None),
    "gradient_error_after": ("Mean dU/dY smoothed after the differences: RMS error (%)", 1, 1),
    "gradient_error_previous": ("Mean dU/dY with σ = 1.5 (before VelociHub 1.2): RMS error (%)", 1, 1),
    "gradient_transfer_theory_error": ("Mean dU/dY kept per wavelength − theory: largest", 1, 4),
    "peak_shear_ratio": ("Peak |dU/dY| of the shear layers / exact", 1, 3),
    "peak_shear_ratio_grid": ("— the grid alone (no smoothing)", 1, 3),
    "lx_u_ratio": ("Integral length L_x^u / modes", 1, 3),
    "ly_u_ratio": ("Integral length L_y^u / modes", 1, 3),
    "lx_v_ratio": ("Integral length L_x^v / modes", 1, 3),
    "ly_v_ratio": ("Integral length L_y^v / modes", 1, 3),
    "t_ratio": ("Integral time T_u / (L_x^u / U0)", 1, 3),
    "dissipation_grid_ratio": ("Dissipation ε / ε with the grid's differences", 1, 3),
    "dissipation_ratio": ("Dissipation ε / exact ε", 1, 3),
    # Background removal
    "particles_kept": ("Particles kept (% of their brightness)", 1, 1),
    "particle_kept_median": ("Each particle: brightness kept, median (%)", 1, 1),
    "particle_kept_p05": ("Each particle: brightness kept, 5th percentile (%)", 1, 1),
    "particles_dimmed": ("Particles below half their brightness (%)", 1, 2),
    "dimmed_worst_image": ("— in the image with the most of them (%)", 1, 1),
    "particle_kept_over_features": ("Particles over a reflection: brightness kept, median (%)", 1, 1),
    "particles_measured": ("Isolated particles measured (all the images)", 1, 0),
    "clipped_pixels": ("Pixels at 0 after the removal (negative values clipped, %)", 1, 1),
    "offset": ("Offset added after the removal (grey levels; automatic)", 1, 0),
    "particle_kept_without_offset": ("— without the offset, clipped at 0 (median, %)", 1, 1),
    "residual_rms": ("Background left: RMS of the mean image less the particles, about its level (grey levels)", 1, 2),
    "lines_kept": ("Lines and circles left: largest (% of their brightness)", 1, 2),
    "lines_min_kept": ("Lines and circles: smallest measured (%)", 1, 1),
    "glare_kept": ("Glare spot left (%)", 1, 2),
    "hough_found": ("Lines and circles the Hough transform finds", 1, 0),
    "hough_total": ("Lines and circles in the images", 1, 0),
}
# The metrics that differ between the levels (the same key): label by level
LEVEL_LABELS = {("post", "rms"): "RMS of the fluctuations, median (px)",
                ("processing", "rms"): "RMS error of the displacement |Δd| (px)",
                ("images", "rms"): "RMS error of the displacement |Δd| (px)",
                ("chain", "rms"): "RMS of the fluctuations, median (px)"}


def metric_label(key, level=None):
    if (level, key) in LEVEL_LABELS:
        return LEVEL_LABELS[(level, key)]
    if key.startswith("transfer_"):
        return f"Amplitude kept at λ = {key[9:]} px"
    if key.startswith("gradient_transfer_grid_"):
        return f"— the grid alone, λ = {key[23:]} px"
    if key.startswith("gradient_transfer_"):
        return f"Mean dU/dY kept at λ = {key[18:]} px (away from the edges)"
    if key.startswith("left_"):
        return f"{key[5:].capitalize()} left: peak (grey levels)"
    if key.startswith("kept_"):
        return f"{key[5:].capitalize()} left (% of its brightness)"
    if key.startswith("hough_") and key not in METRICS:
        return f"{key[6:].capitalize()} found by the Hough transform (1: yes)"
    return METRICS.get(key, (key, 1, None))[0]


def fmt(key, value):
    """A metric value as shown (its factor applied)."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    default = (3 if key.startswith(("transfer_", "gradient_transfer_")) else 2 if key.startswith(("kept_", "left_"))
               else 0 if key.startswith("hough_") else None)
    _, factor, decimals = METRICS.get(key, (key, 1, default))
    v = float(value) * factor
    if decimals is not None:
        return f"{v:,.{decimals}f}"
    if abs(v) < 1e-9:
        return "0"
    return f"{v:.3g}" if 1e-3 <= abs(v) < 1e4 else f"{v:.2e}"


def scenario_name(level, key):
    table = {"post": POST_SCENARIOS, "processing": SCENARIOS, "preprocessing": BACKGROUND_SCENARIOS,
             "images": IMAGE_SCENARIOS, "chain": CHAIN_SCENARIOS}.get(level, {})
    if key not in table:
        return key
    return table[key] if isinstance(table[key], str) else table[key].name


def fmt_limit(check, limit):
    if limit is None:
        return "—"
    if check.op == "within":
        return f"{fmt(check.metric, limit[0])} to {fmt(check.metric, limit[1])}"
    ref = ""
    if isinstance(check.limit, str):
        scenario, name = check.limit.split(":", 1) if ":" in check.limit else ("", check.limit)
        method, metric = name.split(".", 1)
        ref = f" ({'×' + format(check.factor, 'g') + ' ' if check.factor != 1 else ''}{SHORT_NAMES[method]}"
        if metric != check.metric:
            ref += f" {metric_label(metric, check.level).split(' (')[0].lower()}"
        ref += f"{', ' + scenario_name(check.level, scenario).lower() if scenario else ''})"
    return f"{check.op} {fmt(check.metric, limit)}{ref}"


def check_id(case, check):
    return f"{case}|{check.level}|{check.scenario}|{check.method}|{check.metric}|{check.op}|{check.label}"


################################################################################
# Figures


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(NO_DATA)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _figure(w, h):
    return Figure(figsize=(w, h), facecolor=SURFACE, layout="constrained")


def _save(fig, folder, name):
    fig.savefig(os.path.join(folder, name), dpi=110, facecolor=SURFACE)
    return name


def _limits(arrays, signed):
    arrays = [np.asarray(a, np.float64) for a in arrays if a is not None]
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays]) if arrays else np.zeros(0)
    if vals.size == 0:
        return (-1, 1) if signed else (0, 1)
    if signed:
        m = float(np.percentile(np.abs(vals), 99.5)) or 1e-9
        return -m, m
    lo, hi = float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5))
    if hi - lo < 1e-9 * max(1.0, abs(hi)):
        lo, hi = lo - max(1e-3, 0.05 * abs(lo)), hi + max(1e-3, 0.05 * abs(hi))
    return lo, hi


def _signed(*arrays):
    vals = np.concatenate([np.asarray(a, np.float64)[np.isfinite(a)].ravel() for a in arrays if a is not None]
                          or [np.zeros(0)])
    return bool(vals.size and vals.min() < 0 < vals.max())


def _map(ax, X, Y, a, lims, signed):
    cmap = (DIVERGING if signed else SEQUENTIAL).with_extremes(bad=NO_DATA)
    _style(ax)
    ax.set_aspect("equal")
    if a is None:
        ax.text(0.5, 0.5, "not computed", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return None
    return ax.pcolormesh(X, Y, np.ma.masked_invalid(a), cmap=cmap, vmin=lims[0], vmax=lims[1], shading="nearest",
                         rasterized=True)


def _colorbar(fig, mesh, axes, label):
    if mesh is None:
        return
    cb = fig.colorbar(mesh, ax=axes, shrink=0.9, pad=0.01, aspect=14)
    cb.set_label(label, color=INK_2, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=7)
    cb.outline.set_edgecolor(NO_DATA)


def map_grid(X, Y, rows, columns, title_of, folder, name):
    """A grid of maps: rows (label, function(column) -> 2-D array or None, signed), columns; each row with its
    colour bar (the same range for all its maps)."""
    fig = _figure(2.75 * len(columns) + 0.9, 1.95 * len(rows) + 0.35)
    axes = fig.subplots(len(rows), len(columns), squeeze=False, sharex=True, sharey=True)
    for i, (label, get, signed) in enumerate(rows):
        arrays = {c: get(c) for c in columns}
        lims = _limits(arrays.values(), signed)
        mesh = None
        for j, c in enumerate(columns):
            ax = axes[i, j]
            mesh = _map(ax, X, Y, arrays[c], lims, signed) or mesh
            if i == 0:
                ax.set_title(title_of(c), color=INK, fontsize=9)
            if j == 0:
                ax.set_ylabel("Y (px)", fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel("X (px)", fontsize=8)
        _colorbar(fig, mesh, axes[i, :].tolist(), label)
    return _save(fig, folder, name)


def compare_grid(X, Y, rows, folder, name, titles=("Expected", "VelociHub", "Difference")):
    """Rows of (label, expected, measured): the two maps with the same colour range, and their difference with its
    own (diverging) colour bar."""
    fig = _figure(10.2, 1.95 * len(rows) + 0.35)
    axes = fig.subplots(len(rows), 3, squeeze=False, sharex=True, sharey=True)
    for i, (label, expected, measured) in enumerate(rows):
        signed = _signed(expected, measured)
        lims = _limits([expected, measured], signed)
        m1 = _map(axes[i, 0], X, Y, expected, lims, signed)
        m2 = _map(axes[i, 1], X, Y, measured, lims, signed)
        _colorbar(fig, m2 or m1, axes[i, :2].tolist(), label)
        diff = None if measured is None or expected is None else measured - expected
        md = _map(axes[i, 2], X, Y, diff, _limits([diff], True), True)
        _colorbar(fig, md, axes[i, 2], "difference")
        axes[i, 0].set_ylabel("Y (px)", fontsize=8)
        for j in range(3):
            if i == 0:
                axes[i, j].set_title(titles[j], color=INK, fontsize=9)
            if i == len(rows) - 1:
                axes[i, j].set_xlabel("X (px)", fontsize=8)
    return _save(fig, folder, name)


def processing_figures(result, folder):
    """The figures of the processing level of a case: [(title, caption, file)]."""
    from .regression_level import vorticity

    key = result.case.key
    out = []
    flowing = result.status == 0

    def pick(scenario, method, q, pair=0):
        d = result.data[scenario]
        if method not in d.fields:
            return None
        U, V = d.fields[method]
        U, V = np.where(flowing, U[pair], np.nan), np.where(flowing, V[pair], np.nan)
        return {"U": U, "V": V}.get(q) if q != "W" else vorticity(result.X, result.Y, U, V)

    exact = {q: np.where(flowing, result.exact[q][0], np.nan) for q in ("U", "V", "W")}
    for scenario, d in result.data.items():
        if not d.fields:
            continue
        name = scenario_name("processing", scenario).lower()
        methods = [m for m, _ in METHODS if m in d.fields]
        rows = [("U (px/pair)", lambda m: pick(scenario, m, "U"), _signed(result.exact["U"][0])),
                ("V (px/pair)", lambda m: pick(scenario, m, "V"), True),
                ("ω (1/pair)", lambda m: pick(scenario, m, "W"), True)]
        out.append((f"Fields — {name}",
                    "The exact field sampled at the grid points, and the fields of each method read from its results "
                    "(the first pair). Grey: no value (masked, no particle in the bin, or invalid). The vorticity is "
                    "computed by differences on the grid, as in the post-processing.",
                    map_grid(result.X, result.Y, rows, methods, lambda m: SHORT_NAMES[m], folder,
                             f"{key}_{scenario}_fields.png")))
        others = [m for m in methods if m != "exact"]
        diff = [("U − U exact (px)", lambda m: _sub(pick(scenario, m, "U"), exact["U"]), True),
                ("V − V exact (px)", lambda m: _sub(pick(scenario, m, "V"), exact["V"]), True),
                ("ω − ω exact (1/pair)", lambda m: _sub(pick(scenario, m, "W"), exact["W"]), True)]
        out.append((f"Differences from the exact field — {name}",
                    "Each method minus the exact field (the first pair), with the same colour range for a row. For "
                    "ω, the difference includes the error of the differences on the grid (see the exact column of "
                    "the metrics).",
                    map_grid(result.X, result.Y, diff, others, lambda m: SHORT_NAMES[m], folder,
                             f"{key}_{scenario}_differences.png")))
    special = SPECIAL.get(key)
    if special is not None and "nominal" in result.data and result.data["nominal"].fields:
        out.append(special(result, folder))
    unc = uncertainty_figure(result, folder)
    if unc is not None:
        out.append(unc)
    return out


def post_figures(result, folder):
    """The figures of the post-processing level of a case: the noise scenario, the post-processing of VelociHub
    against the expected fields."""
    from .post_level import NU_PX  # noqa: F401 (the same post-processing)

    out = []
    flow = result.flow
    for scenario in ("noise",):
        if scenario not in result.post:
            continue
        fields, post = result.post[scenario]
        X, Y = fields.x / MM_PER_PX, fields.y / MM_PER_PX
        st = post.statistics()
        flowing = st["reliable"]
        noise = POST_SCENARIOS[scenario].noise_px

        def flow_only(a):
            return np.where(flowing, a, np.nan)
        if flow.steady:
            EU, _ = flow.velocity(X, Y)
            W = flow.vorticity(X, Y)
            sigma = np.full(X.shape, noise)
        else:
            EU, W = np.full(X.shape, flow.u0), np.zeros(X.shape)
            sigma = np.full(X.shape, np.sqrt(flow.variances()[0] + noise**2))
        rows = [("mean U (px/pair)", flow_only(EU), flow_only(st["mean_U"] / MS_PER_PX)),
                ("RMS of u (px/pair)", flow_only(sigma), flow_only(st["u_rms"] / MS_PER_PX)),
                ("mean ω (1/pair)", flow_only(W), flow_only(post.vorticity_statistics()["mean"] / PER_S))]
        if not flow.steady:
            rows.append(("L_x^u (px)", flow_only(np.full(X.shape, flow.integral_length("x", "u"))),
                         flow_only(post.integral_length("x", "u") / MM_PER_PX)))
        out.append((f"Post-processing — {scenario_name('post', scenario).lower()}",
                    f"The statistics VelociHub computes from {fields.n} maps (load_vectors with the camera, "
                    "PostProcessing with its default recipe, converted back to px and pairs) against the expected "
                    "ones: the exact mean, the noise" + (" and the turbulence of the modes" if not flow.steady else "")
                    + ", the analytic vorticity.",
                    compare_grid(X, Y, rows, folder, f"{flow.key}_post_{scenario}.png")))
    if not flow.steady and "exact" in result.post:
        out.append(turbulence_correlations(result, folder))
    return out + gradient_figures(result, folder)


# Three series of a chart (the categorical order of SERIES)
THREE = ("#2a78d6", "#eb6834", "#1baf7a")


def gradient_figures(result, folder):
    """The mean gradient dU/dY of the post-processing split into its parts (post_level.gradient_study): the maps of
    the exact gradient and of the errors of the grid alone and of VelociHub; its error against the smoothing, with a
    noise of the mean field; for the modes, the amplitude kept per wavelength against the theory; for the jet, the
    peak shear against the thickness of the shear layers."""
    from .post_level import MEAN_NOISES, SIGMAS, gradient_study, gradient_transfer_theory, peak_shear

    if "exact" not in result.post:
        return []
    flow, case = result.flow, result.case
    fields, post = result.post["exact"]
    g = gradient_study(flow, case, fields, post)
    if g is None:
        return []
    key, h = flow.key, case.spacing
    X, Y, G = g["X"], g["Y"], g["G"]
    GRADIENT_SIGMA = g["sigma"]
    out = []
    fig = _figure(12.4, 2.35)
    axes = fig.subplots(1, 4, sharex=True, sharey=True)
    signed = _signed(G)
    m0 = _map(axes[0], X, Y, G, _limits([G], signed), signed)
    _colorbar(fig, m0, axes[0], "dU/dY (1/pair)")
    errors = (g["grid"] - G, g["smoothed"] - G, g["after"] - G)
    lims = _limits(errors, True)
    m1 = None
    for ax, e in zip(axes[1:], errors):
        m1 = _map(ax, X, Y, e, lims, True) or m1
    _colorbar(fig, m1, axes[1:].tolist(), "error (1/pair)")
    titles = ("Exact dU/dY", "Grid alone − exact", f"σ = {GRADIENT_SIGMA:g} − exact",
              f"σ = {GRADIENT_SIGMA:g}, smoothed after − exact")
    for ax, title in zip(axes, titles):
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xlabel("X (px)", fontsize=8)
    axes[0].set_ylabel("Y (px)", fontsize=8)
    out.append(("Mean gradient: the grid and the smoothing",
                "The exact dU/dY at the grid points; the error of the grid alone (the exact field on the grid, "
                "central differences: the discretisation, one-sided at the edges and masks); the error of VelociHub "
                f"(its mean field smoothed over σ = {GRADIENT_SIGMA:g} cells ignoring the holes, then the same "
                "differences), and with the differences first, smoothed after (Smooth after differencing). The "
                "exact fields scenario.", _save(fig, folder, f"{key}_gradient_maps.png")))
    fig = _figure(6.4, 3.4)
    ax = _profile_axes(fig, 1)[0]
    for i, noise in enumerate(MEAN_NOISES):
        sweep = g["sweep"][noise]
        label = "exact mean field" if not noise else f"mean field + {noise:g} px noise"
        ax.plot(SIGMAS, [a for a, _ in sweep], "-o", color=THREE[i], linewidth=2, markersize=5,
                markeredgecolor=SURFACE, label=label)
        if not noise:
            ax.plot(SIGMAS, [b for _, b in sweep], "--", color=THREE[i], linewidth=1.5,
                    label="exact, away from the edges and masks")
            ax.plot(SIGMAS, g["sweep_after"], "-s", color=INK_2, linewidth=1.5, markersize=4,
                    markeredgecolor=SURFACE, label="exact, smoothed after the differences")
    ax.set_ylim(bottom=0)
    ax.axvline(GRADIENT_SIGMA, color=INK_2, linewidth=1, linestyle=":")
    ax.text(GRADIENT_SIGMA, ax.get_ylim()[1], " default", color=INK_2, fontsize=7, va="top", ha="left")
    ax.set_xlabel(f"σ of the smoothing (cells of {h} px)", fontsize=8)
    ax.set_ylabel("RMS error of dU/dY (% of its largest)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="best")
    out.append(("Mean gradient against the smoothing",
                "The RMS error of dU/dY (grid points one cell away from the edges and masks; dashed: more than 3σ "
                "away) for each smoothing: at σ = 0 the grid alone. A white noise added to the mean field (0.1 and "
                "0.2 px: the noise of a single map) shows where the smoothing pays.",
                _save(fig, folder, f"{key}_gradient_sweep.png")))
    if key == "modes":
        fig = _figure(6.4, 3.4)
        ax = _profile_axes(fig, 1)[0]
        wl = np.geomspace(2.2 * h, 25 * h, 200)
        m = result.metrics["post"]["exact"]["post"]
        points = np.array(flow.WAVELENGTHS)
        series = ((0.0, "grid alone", "gradient_transfer_grid_"),
                  (GRADIENT_SIGMA, f"VelociHub (σ = {GRADIENT_SIGMA:g})", "gradient_transfer_"))
        for i, (sigma, name, prefix) in enumerate(series):
            ax.plot(wl / h, [gradient_transfer_theory(w, h, sigma) for w in wl], "-", color=THREE[i], linewidth=2,
                    label=f"{name}: theory")
            ax.plot(points / h, [m.get(f"{prefix}{w:g}") for w in points], "o", color=THREE[i], markersize=6,
                    markeredgecolor=SURFACE, label=f"{name}: measured")
        ax.axhline(1, color=INK, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xticks(points / h, [f"{w / h:g}" for w in points])
        ax.minorticks_off()
        ax.set_xlabel("Wavelength (grid spacings)", fontsize=8)
        ax.set_ylabel("Amplitude of dU/dY kept", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        out.append(("Mean gradient: transfer function",
                    "The amplitude of dU/dY of each mode kept (projection on the modes, away from the edges), and the "
                    "theory: sin(kh)/kh for the central differences, times the transfer of the Gaussian smoothing in "
                    "X and in Y.", _save(fig, folder, "modes_gradient_transfer.png")))
    if key == "jet":
        fig = _figure(6.4, 3.4)
        ax = _profile_axes(fig, 1)[0]
        series = ((g["grid"], "grid alone"), (g["smoothed"], f"VelociHub (σ = {GRADIENT_SIGMA:g} cells)"))
        for i, (F, name) in enumerate(series):
            d, r = peak_shear(flow, F, X, Y, fields.status)
            ax.plot(d / h, r, "o", color=THREE[i], markersize=5, markeredgecolor=SURFACE, label=name)
        ax.axhline(1, color=INK, linestyle="--", linewidth=1, label="Exact")
        ax.set_xlabel("Shear layer thickness δ / grid spacing", fontsize=8)
        ax.set_ylabel("Peak |dU/dY| / exact (U / 2δ)", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        out.append(("Mean gradient: peak shear",
                    "The largest |dU/dY| across each shear layer of each column, over the exact one, against the "
                    "thickness of the layer, which grows downstream.", _save(fig, folder, "jet_peak_shear.png")))
    return out


def uncertainty_figure(result, folder):
    """The uncertainty of the binned PTV predicted from the flow (uncertainty.py) against its actual error: the RMS
    error in classes of predicted uncertainty, and the maps of both (first pair)."""
    d = result.data.get("nominal")
    if d is None or "binned" not in d.uncertainty:
        return None
    u = d.uncertainty["binned"]
    e, total, pred = u["errors"], u["total"], u["pred"]
    fig = _figure(10.2, 2.7)
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1.15, 1.15))
    ax = fig.add_subplot(grid[0])
    _style(ax)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    edges = np.unique(np.quantile(total, np.linspace(0, 1, 13)))
    which = np.clip(np.digitize(total, edges[1:-1]), 0, len(edges) - 2)
    classes = [i for i in range(len(edges) - 1) if (which == i).any()]
    pu = [np.sqrt(np.mean(total[which == i] ** 2)) for i in classes]
    pe = [np.sqrt(np.mean(e[which == i] ** 2)) for i in classes]
    top = 1.1 * max(max(pu), max(pe))
    ax.plot([0, top], [0, top], "--", color=INK, linewidth=1)
    ax.plot(pu, pe, "o", color=SERIES["binned"], markersize=6, markeredgecolor=SURFACE)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Predicted uncertainty (px)", fontsize=8)
    ax.set_ylabel("Actual RMS error (px)", fontsize=8)
    ax.set_title("Each component, 12 classes", color=INK, fontsize=9)
    flowing = result.status == 0
    unc = np.sqrt(pred["sigma_u"][0] ** 2 + pred["bias_u"][0] ** 2 + pred["sigma_v"][0] ** 2 + pred["bias_v"][0] ** 2)
    U, V = d.fields["binned"]
    err = np.where(flowing, np.hypot(U[0] - result.exact["U"][0], V[0] - result.exact["V"][0]), np.nan)
    unc = np.where(np.isfinite(err), unc, np.nan)
    lims = (0, _limits([unc, err], False)[1])
    a1, a2 = fig.add_subplot(grid[1]), fig.add_subplot(grid[2])
    _map(a1, result.X, result.Y, unc, lims, False)
    mesh = _map(a2, result.X, result.Y, err, lims, False)
    a1.set_title("Predicted uncertainty of |Δd|", color=INK, fontsize=9)
    a2.set_title("Actual |Δd| (first pair)", color=INK, fontsize=9)
    for a in (a1, a2):
        a.set_xlabel("X (px)", fontsize=8)
    _colorbar(fig, mesh, [a1, a2], "px")
    m = result.metrics["processing"]["nominal"]["binned"]
    return ("Uncertainty predicted from the flow — binned PTV",
            "Each bin: random error √((σ₀² + h²/12 |∇u|²) / N) (the noise of the N links and the spread of the "
            "displacement over the bin) and bias h²/24 ∇²u (the curvature averaged over the bin), from the exact "
            "gradients and curvature and the links of each bin. Predicted / actual RMS: "
            f"{fmt('unc_ratio', m.get('unc_ratio'))}; within 1 and 2 uncertainties: "
            f"{fmt('unc_cover_1', m.get('unc_cover_1'))} % and {fmt('unc_cover_2', m.get('unc_cover_2'))} % (68 and "
            "95 for a Gaussian error). On the left, classes on the dashed line: the prediction holds at every level "
            "of uncertainty.", _save(fig, folder, f"{result.case.key}_uncertainty.png"))


def turbulence_correlations(result, folder):
    """The autocorrelations of the turbulence: those of its modes, and the measured ones (VelociHub's estimators),
    with their integrals to the first zero crossing."""
    from ..turbulence import space_autocorrelation, time_autocorrelation

    flow = result.flow
    fields, post = result.post["exact"]
    U, V, _ = post.clean()
    fig = _figure(10.2, 3.2)
    axes = fig.subplots(1, 3, squeeze=False)[0]
    spacing = result.case.spacing
    for ax, (title, axis, comp) in zip(axes[:2], (("Along x", "x", "u"), ("Along y", "y", "u"))):
        _style(ax)
        ax.grid(True, color=GRID, linewidth=0.6)
        r = np.linspace(0, 6 * flow.peak_wavelength / 2, 400)
        for c, color, name in (("u", SERIES["spicy_ptv"], "u'"), ("v", SERIES["binned"], "v'")):
            ax.plot(r, flow.correlation(r, axis, c), "--", color=color, linewidth=1.2)
            rho = space_autocorrelation(U if c == "u" else V, 2 if axis == "x" else 1)
            mean = np.nanmean(rho.reshape(rho.shape[0], -1)[:, :], 1)
            ax.plot(np.arange(len(mean)) * spacing, mean, "o-", color=color, linewidth=2, markersize=4,
                    markeredgecolor=SURFACE, label=f"{name}: measured (dashed: modes)")
        ax.axhline(0, color=NO_DATA, linewidth=0.8)
        ax.set_xlim(0, r[-1])
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel(f"separation along {axis} (px)", fontsize=8)
    axes[0].set_ylabel("autocorrelation", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    ax = axes[2]
    _style(ax)
    ax.grid(True, color=GRID, linewidth=0.6)
    tau = np.linspace(0, 6 * flow.integral_time(), 300)
    ax.plot(tau, flow.correlation(flow.u0 * tau, "x", "u"), "--", color=SERIES["spicy_ptv"], linewidth=1.2)
    rho = time_autocorrelation(U, max_lag=int(tau[-1]) + 1)
    mean = np.nanmean(rho.reshape(rho.shape[0], -1), 1)
    ax.plot(np.arange(len(mean)), mean, "o-", color=SERIES["spicy_ptv"], linewidth=2, markersize=4,
            markeredgecolor=SURFACE, label="u': measured (dashed: modes)")
    ax.axhline(0, color=NO_DATA, linewidth=0.8)
    ax.set_title("In time (frozen turbulence)", fontsize=9, color=INK)
    ax.set_xlabel("lag (frames)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    return ("Autocorrelations of the turbulence",
            "The spatial and temporal autocorrelations of the exact fields as VelociHub estimates them "
            "(turbulence.py, averaged over the grid) and those of the modes (dashed); the integral scales integrate "
            "them to their first zero crossing.",
            _save(fig, folder, "turbulence_correlations.png"))


def _sub(a, b):
    return None if a is None else a - b


def _profile_axes(fig, n):
    axes = fig.subplots(1, n, squeeze=False, sharey=True)[0]
    for ax in axes:
        _style(ax)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    return axes


def _plot_methods(ax, x_of, y_of, result, scenario="nominal"):
    d = result.data[scenario]
    for m in ("spicy_ptv", "binned", "spicy_piv"):
        if m not in d.fields:
            continue
        x, y = x_of(m), y_of(m)
        style = dict(color=SERIES[m], label=SHORT_NAMES[m])
        if m == "binned":
            ax.plot(x, y, "o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8, **style)
        else:
            ax.plot(x, y, "-", linewidth=2, **style)


def _columns(result, n=3):
    w = result.X.shape[1]
    return [int(round(i)) for i in np.linspace(1, w - 2, n)]


def special_cylinder(result, folder):
    flow = result.flow
    j = int(np.argmin(np.abs(result.X[0] - flow.centre[0])))
    x0 = result.X[0, j]
    fig = _figure(6.4, 3.4)
    ax = _profile_axes(fig, 1)[0]
    y = np.linspace(result.Y.min(), result.Y.max(), 600)
    U, _ = flow.velocity(np.full_like(y, x0), y)
    U[flow.solid(np.full_like(y, x0), y)] = np.nan
    ax.plot(U, y, "--", color=INK, linewidth=1.2, label="Exact")
    d = result.data["nominal"]
    _plot_methods(ax, lambda m: np.where(result.status[:, j] == 0, d.fields[m][0][0, :, j], np.nan),
                  lambda m: result.Y[:, j], result)
    ax.axhspan(flow.centre[1] - flow.radius, flow.centre[1] + flow.radius, color=GRID, zorder=0)
    ax.set_xlabel("U (px/pair)", fontsize=8)
    ax.set_ylabel("Y (px)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    return ("Profile through the cylinder",
            f"U along the vertical line X = {x0:g} px through the centre of the cylinder (shaded), noise scenario, "
            "first pair: the speed doubles at its shoulders.",
            _save(fig, folder, "cylinder_profile.png"))


def special_jet(result, folder):
    flow = result.flow
    cols = _columns(result)
    fig = _figure(8.6, 3.4)
    axes = _profile_axes(fig, len(cols))
    d = result.data["nominal"]
    y = np.linspace(result.Y.min(), result.Y.max(), 600)
    for ax, j in zip(axes, cols):
        x0 = result.X[0, j]
        U, _ = flow.velocity(np.full_like(y, x0), y)
        ax.plot(U, y, "--", color=INK, linewidth=1.2, label="Exact")
        _plot_methods(ax, lambda m: d.fields[m][0][0, :, j], lambda m: result.Y[:, j], result)
        ax.set_title(f"X = {x0:g} px (δ = {flow.delta(x0):.0f} px)", fontsize=9, color=INK)
        ax.set_xlabel("U (px/pair)", fontsize=8)
    axes[0].set_ylabel("Y (px)", fontsize=8)
    axes[-1].legend(fontsize=7, frameon=False, loc="center right")
    return ("Profiles across the jet",
            "U across the jet at three stations, noise scenario, first pair: the shear layers thicken downstream.",
            _save(fig, folder, "jet_profiles.png"))


def special_modes(result, folder):
    flow = result.flow
    fig = _figure(6.4, 3.4)
    ax = _profile_axes(fig, 1)[0]
    wl = np.array(flow.WAVELENGTHS)
    spacing = result.case.spacing
    m_all = result.metrics["processing"]["nominal"]
    for m in ("spicy_ptv", "binned", "spicy_piv"):
        if m not in m_all:
            continue
        t = [m_all[m].get(f"transfer_{w:g}") for w in wl]
        ax.plot(wl / spacing, t, "-o", color=SERIES[m], linewidth=2, markersize=6, markeredgecolor=SURFACE,
                label=SHORT_NAMES[m])
    ax.axhline(1, color=INK, linestyle="--", linewidth=1.2, label="Exact")
    ax.set_xscale("log")
    ax.set_xticks(wl / spacing, [f"{w / spacing:g}" for w in wl])
    ax.minorticks_off()
    ax.set_xlabel("Wavelength (grid spacings)", fontsize=8)
    ax.set_ylabel("Amplitude kept", fontsize=8)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    return ("Spatial resolution",
            "The amplitude of each wavelength a method keeps (least-squares projection of its fields on the modes, "
            "noise scenario, mean of the pairs): its transfer function.",
            _save(fig, folder, "modes_transfer.png"))


def special_blasius(result, folder):
    flow = result.flow
    cols = _columns(result)
    fig = _figure(8.6, 3.4)
    axes = _profile_axes(fig, len(cols))
    d = result.data["nominal"]
    for ax, j in zip(axes, cols):
        x0 = result.X[0, j]
        g = flow.scale(x0)
        eta = np.linspace(0, 10, 400)
        U, _ = flow.velocity(np.full_like(eta, x0), flow.y_wall + eta * g)
        ax.plot(U / flow.u_ref, eta, "--", color=INK, linewidth=1.2, label="Exact")
        keep = result.status[:, j] == 0
        _plot_methods(ax, lambda m: d.fields[m][0][0, keep, j] / flow.u_ref,
                      lambda m: (result.Y[keep, j] - flow.y_wall) / g, result)
        ax.set_ylim(0, 10)
        ax.set_title(f"X = {x0:g} px (δ99 = {flow.delta99(x0):.0f} px)", fontsize=9, color=INK)
        ax.set_xlabel("U / U_e", fontsize=8)
    axes[0].set_ylabel("η = (Y − Y_wall) / √(νx/U)", fontsize=8)
    axes[-1].legend(fontsize=7, frameon=False, loc="lower right")
    return ("Boundary layer profiles",
            "U/U_e against the similarity variable at three stations, noise scenario, first pair (the exact profile "
            "is the same at every station).",
            _save(fig, folder, "blasius_profiles.png"))


SPECIAL = {"cylinder": special_cylinder, "jet": special_jet, "modes": special_modes, "blasius": special_blasius}


def summary_chart(result, folder):
    """RMS error of each method in each case (processing level, noise scenario): grouped bars."""
    cases = [r for r in result.cases if "nominal" in r.metrics.get("processing", {})]
    if not cases:
        return None
    methods = ("spicy_ptv", "binned", "spicy_piv")
    fig = _figure(8.6, 3.0)
    ax = fig.subplots()
    _style(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    width = 0.26
    x = np.arange(len(cases))
    for i, m in enumerate(methods):
        vals = [r.metrics["processing"]["nominal"].get(m, {}).get("rms", np.nan) for r in cases]
        ax.bar(x + (i - 1) * (width + 0.02), vals, width, color=SERIES[m], label=SHORT_NAMES[m])
    ax.axhline(0.1 * np.sqrt(2), color=INK_2, linestyle="--", linewidth=1)
    ax.text(-0.45, 0.1 * np.sqrt(2), "noise of the data", color=INK_2, fontsize=7, va="bottom", ha="left")
    ax.set_xticks(x, [r.flow.name.replace(" (Blasius)", "").replace(" around a cylinder", "") for r in cases],
                  fontsize=8)
    ax.set_ylabel("RMS error |Δd| (px)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    return _save(fig, folder, "summary_rms.png")


# The columns of the comparison of the recipes: (metric, lower is better), and their short headings
RECIPE_COLUMNS = {"piv": (("rms", True), ("rms_interior", True), ("outliers", True), ("valid", False),
                          ("peak_ratio", False)),
                  "ptv": (("links_per_pair", False), ("wrong_links", True), ("link_rms", True),
                          ("track_length_median", False), ("yield", False), ("rms", True), ("coverage", False))}
RECIPE_HEADS = {"rms": "RMS error (px)", "rms_interior": "— interior (px)", "outliers": "Outliers (%)",
                "valid": "Valid (%)", "peak_ratio": "Peak ratio", "links_per_pair": "Links per pair",
                "wrong_links": "Wrong links (%)", "link_rms": "Link error (px)",
                "track_length_median": "Track length (frames)", "yield": "Tracked (%)", "coverage": "Cells (%)",
                "spatial_noise": "Spatial noise (px)", "suspect_links": "Suspect links (%)",
                "track_length_mean": "Mean track (frames)", "detected_per_frame": "Detected per frame",
                "displacement": "Median |d| (px)"}
RECIPE_NOTES = {"piv": "RMS error of the PIV vectors against the exact field, all the flow and away from the edges "
                       "and masks; outliers: vectors more than 0.5 px off; valid: the vectors the PIV kept (the "
                       "others replaced); the median correlation peak ratio (S/N).",
                "ptv": "The particles linked in each pair (median), the links more than 1 px off (wrong), the RMS "
                       "error of the others, the median length of the tracks, the particles tracked of those "
                       "detected, then the field binned on the grid: its RMS error and the cells with a value."}


def recipes_of(metrics, style):
    """The recipes run on the images of a style, in the order of RECIPES."""
    return [k for k in RECIPES if f"{style}_{k}" in metrics]


def recipe_table(metrics, style, method, columns=None):
    """The recipes of the images of a style (rows) for a method: the metrics of columns ((metric, lower is better),
    RECIPE_COLUMNS of the method by default), the best of each column in bold."""
    recipes = [k for k in recipes_of(metrics, style) if method in metrics[f"{style}_{k}"]]
    if not recipes:
        return "<p class=meta>No result.</p>"
    columns = [(k, low) for k, low in (columns or RECIPE_COLUMNS[method])
               if any(metrics[f"{style}_{r}"][method].get(k) is not None for r in recipes)]
    best = {}
    for k, low in columns:
        if low is None:  # Shown, not ranked
            continue
        vals = [metrics[f"{style}_{r}"][method].get(k) for r in recipes]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if len(vals) > 1:
            best[k] = min(vals) if low else max(vals)
    rows = []
    for r in recipes:
        m = metrics[f"{style}_{r}"][method]
        cells = []
        for k, _ in columns:
            text = fmt(k, m.get(k))
            cells.append(f'<td class="num">{"<strong>" + text + "</strong>" if k in best and m.get(k) == best[k] else text}'
                         "</td>")
        rows.append(f'<tr><td title="{esc(RECIPES[r].description)}">{esc(RECIPES[r].name)}</td>{"".join(cells)}</tr>')
    head = "".join(f'<th class="num" title="{esc(metric_label(k, "images"))}">{esc(RECIPE_HEADS[k])}</th>'
                   for k, _ in columns)
    return (f'<div class="scroll"><table><thead><tr><th>Pre-processing</th>{head}</tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def recipe_figure(metrics, style, folder, name):
    """The recipes of the images of a style compared: the RMS error of the PIV, the particles linked per pair of the
    PTV and its wrong links (horizontal bars, the best of each darker)."""
    recipes = recipes_of(metrics, style)
    if not recipes:
        return None
    fig = _figure(8.6, 0.32 * len(recipes) + 1.1)
    axes = fig.subplots(1, 3, sharey=True)
    y = np.arange(len(recipes))[::-1]
    for ax, (method, key, low, title, colour) in zip(axes, (
            ("piv", "rms", True, "PIV: RMS error |Δd| (px)", SERIES["spicy_piv"]),
            ("ptv", "links_per_pair", False, "PTV: particles linked per pair", SERIES["binned"]),
            ("ptv", "wrong_links", True, "PTV: wrong links (%)", SERIES["binned"]))):
        _style(ax)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        vals = np.array([metrics[f"{style}_{r}"].get(method, {}).get(key, np.nan) for r in recipes], float)
        finite = vals[np.isfinite(vals)]
        target = (finite.min() if low else finite.max()) if finite.size else None
        ax.barh(y, vals, 0.62, color=[colour if v == target else colour + "80" for v in vals])
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v, yi, " " + fmt(key, v), va="center", ha="left", fontsize=7, color=INK_2)
        ax.set_xlabel(title, fontsize=8)
        ax.margins(x=0.3)
    axes[0].set_yticks(y, [RECIPES[r].name for r in recipes], fontsize=8)
    return _save(fig, folder, name)


def recipes_summary_figure(result, folder):
    """The recipes on the hard images of each flow, relative to no pre-processing: the RMS error of the PIV and the
    wrong links of the PTV (grouped bars, a colour per flow; log scale: below 1, better than without)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    cases = [r for r in result.cases if recipes_of(r.metrics.get("images", {}), "hard")]
    if not cases:
        return None
    recipes = [k for k in RECIPES if k != "none"]
    fig = _figure(8.6, 4.8)
    axes = fig.subplots(2, 1, sharex=True)
    width = 0.8 / len(cases)
    x = np.arange(len(recipes))
    for ax, (method, key, label) in zip(axes, (("piv", "rms", "PIV: RMS error\n÷ without"),
                                               ("ptv", "wrong_links", "PTV: wrong links\n÷ without"))):
        _style(ax)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for i, r in enumerate(cases):
            m = r.metrics["images"]
            base = m.get("hard_none", {}).get(method, {}).get(key)
            vals = [m.get(f"hard_{k}", {}).get(method, {}).get(key, np.nan) / base if base else np.nan
                    for k in recipes]
            ax.bar(x + (i - (len(cases) - 1) / 2) * width, vals, width * 0.92, color=THREE[i % len(THREE)],
                   label=r.flow.name)
        ax.axhline(1, color=INK_2, linewidth=1, linestyle="--")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(label, fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, ncol=len(cases), loc="upper left", bbox_to_anchor=(0, 1.18))
    axes[1].set_xticks(x, [RECIPES[k].name.replace(" + ", "\n+ ") for k in recipes], fontsize=7)
    return _save(fig, folder, "recipes_hard.png")


def background_figures(bg, folder):
    """The mean images (raw, and with 1 and 2 modes removed) and what each removal leaves of each feature."""
    data = bg.data
    fig = _figure(10.2, 2.6)
    axes = fig.subplots(1, 3)
    for ax, (title, img) in zip(axes, (("Raw (mean of the images)", data["raw_mean"]),
                                       ("1 mode removed", data["means"].get(1)),
                                       ("2 modes removed", data["means"].get(2)))):
        ax.set_axis_off()
        if img is None:
            continue
        hi = np.percentile(img, 99.8) or 1
        ax.imshow(img, cmap="gray", vmin=0, vmax=hi)
        ax.set_title(title, fontsize=9, color=INK)
    first = _save(fig, folder, "background_means.png")
    fig = _figure(8.6, 3.2)
    ax = fig.subplots()
    _style(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    feats = list(data["features"])
    modes = sorted(data["modes"])
    width = 0.8 / len(modes)
    x = np.arange(len(feats) + 1)
    for i, n in enumerate(modes):
        m = data["modes"][n]
        vals = [m.get(f"kept_{k}", np.nan) for k in feats] + [m["particles_kept"]]
        ax.bar(x + (i - (len(modes) - 1) / 2) * width, vals, width * 0.92, color=ORDINAL[i % len(ORDINAL)],
               label=scenario_name("preprocessing", f"modes_{n}"))
    ax.axhline(0, color=NO_DATA, linewidth=0.8)
    ax.set_xticks(x, [k.capitalize() for k in feats] + ["Particles"], fontsize=8)
    ax.set_ylabel("Kept (% of its brightness)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=len(modes), loc="upper right")
    second = _save(fig, folder, "background_kept.png")
    # The particles one by one: the distribution of their brightness kept, where the dimmed ones are
    ratios = data.get("particle_ratios", {})
    third = fourth = None
    if ratios:
        fig = _figure(10.2, 3.0)
        grid = fig.add_gridspec(1, 2, width_ratios=(1, 1.35))
        ax = fig.add_subplot(grid[0])
        _style(ax)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        bins = np.linspace(0, 130, 66)
        for i, n in enumerate(sorted(ratios)):
            ax.hist(np.clip(ratios[n][0], 0, 130), bins, histtype="step", linewidth=2,
                    color=ORDINAL[i % len(ORDINAL)], density=True, label=scenario_name("preprocessing", f"modes_{n}"))
        ax.axvline(50, color=INK_2, linewidth=1, linestyle=":")
        ax.text(49, 0.5 * ax.get_ylim()[1], "half", color=INK_2, fontsize=7, va="center", ha="right",
                rotation=90)
        ax.set_xlabel("Brightness kept (% of its raw peak)", fontsize=8)
        ax.set_ylabel("Particles (density)", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax = fig.add_subplot(grid[1])
        ax.set_axis_off()
        n = max(ratios)
        r, _, where = ratios[n]
        ax.imshow(data["raw_mean"], cmap="gray", vmin=0, vmax=np.percentile(data["raw_mean"], 99.8) or 1)
        dim = r < 50
        ax.plot(where[dim, 0] - 0.5, where[dim, 1] - 0.5, "o", markersize=2.5, color=SERIES["binned"],
                markeredgewidth=0)
        ax.set_title(f"Particles dimmed below half, {scenario_name('preprocessing', f'modes_{n}').lower()} "
                     f"({int(dim.sum())} of {len(r)}, all the images)", fontsize=9, color=INK)
        third = _save(fig, folder, "background_particles.png")
    residuals = data.get("residuals", {})
    shown = [n for n in (1, 2) if n in residuals]
    if shown:
        fig = _figure(10.2, 2.6)
        axes = fig.subplots(1, len(shown), squeeze=False)[0]
        lims = _limits([residuals[n] for n in shown], True)
        mesh = None
        for ax, n in zip(axes, shown):
            ax.set_axis_off()
            mesh = ax.imshow(np.ma.masked_invalid(residuals[n]), cmap=DIVERGING.with_extremes(bad=NO_DATA),
                             vmin=lims[0], vmax=lims[1])
            ax.set_title(scenario_name("preprocessing", f"modes_{n}"), fontsize=9, color=INK)
        _colorbar(fig, mesh, axes.tolist(), "grey levels")
        fourth = _save(fig, folder, "background_residual.png")
    out = [("Mean images", "The mean of the raw images (the stationary features: the rim of the cylinder, a streak, "
                            "a wall line, a ring and a glare spot), and of the images processed by the POD background "
                            "removal (the cylinder masked).", first),
            ("What the background removal keeps",
             "Each feature measured in the processed images by a matched filter of its known shape, relative to the "
             "particles kept (the particles: their brightness relative to the raw images).", second)]
    if third:
        out.append(("The particles one by one",
                    "The isolated particles of all the images (no other within 5 px, out of the mask), each measured "
                    "in the processed image by the least-squares amplitude of its Gaussian spot over a 7 × 7 px "
                    "window, over its raw peak. On the right, the dimmed ones on the mean raw image: spread over "
                    "the field, they belong to a few images. With 41 images, the modes beyond the first carry the "
                    "particles of single images (their temporal coefficients peak there): removing them removes "
                    "those particles.", third))
    if fourth:
        out.append(("The background left",
                    "The mean of the processed images less the particles (the mean of their images, times the "
                    "fraction kept) and the offset, about its median: what the removal leaves of the background and "
                    "of the reflections, in grey levels (camera noise: 2). Grey: the mask.", fourth))
    return out


################################################################################
# Pages

CSS = """
:root { color-scheme: light; --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --line: #e1e0d9; --axis: #c3c2b7; --accent: #2a78d6; --good: #006300; --good-bg: #e3f3e3; --bad: #b02a2a;
  --bad-bg: #fbe5e4; --warn: #7a5200; --warn-bg: #fdf0d2; --fixed: #8a3a12; --fixed-bg: #fbe7dc; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --page: #0d0d0d;
  --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781; --line: #2c2c2a; --axis: #383835;
  --accent: #3987e5; --good: #0ca30c; --good-bg: #13261a; --bad: #e66767; --bad-bg: #2e1615; --warn: #fab219;
  --warn-bg: #2d2410; --fixed: #ec835a; --fixed-bg: #2e1c13; } }
:root[data-theme="dark"] { color-scheme: dark; --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7;
  --muted: #898781; --line: #2c2c2a; --axis: #383835; --accent: #3987e5; --good: #0ca30c; --good-bg: #13261a;
  --bad: #e66767; --bad-bg: #2e1615; --warn: #fab219; --warn-bg: #2d2410; --fixed: #ec835a; --fixed-bg: #2e1c13; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 14px/1.5 system-ui, -apple-system, "Segoe UI",
  sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
nav { font-size: 13px; color: var(--ink-2); margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 4px 14px; }
nav a, a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
h1 { font-size: 24px; margin: 4px 0 4px; } h2 { font-size: 19px; margin: 36px 0 8px; padding-top: 12px;
  border-top: 1px solid var(--line); } h3 { font-size: 15px; margin: 22px 0 6px; }
p { margin: 6px 0; max-width: 80ch; } .lede { color: var(--ink-2); }
.meta { color: var(--ink-2); font-size: 13px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 16px 0; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; }
.tile .n { font-size: 26px; font-weight: 600; } .tile .l { color: var(--ink-2); font-size: 13px; }
.scroll { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; background: var(--surface); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
.badge { display: inline-block; white-space: nowrap; border-radius: 999px; padding: 1px 9px; font-size: 12px;
  font-weight: 600; }
.badge.pass { color: var(--good); background: var(--good-bg); } .badge.fail { color: var(--bad); background: var(--bad-bg); }
.badge.known { color: var(--warn); background: var(--warn-bg); }
.badge.fixed, .badge.missing { color: var(--fixed); background: var(--fixed-bg); }
.note { color: var(--ink-2); font-size: 12px; }
.delta-worse { color: var(--bad); } .delta-better { color: var(--good); }
figure { margin: 16px 0 24px; background: var(--surface); border: 1px solid var(--line); border-radius: 8px;
  padding: 10px; }
figure img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
figcaption { color: var(--ink-2); font-size: 13px; margin-top: 6px; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 16px; margin: 8px 0; font-size: 13px; }
dt { color: var(--ink-2); } dd { margin: 0; font-variant-numeric: tabular-nums; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
"""


def esc(text):
    return html.escape(str(text))


def badge(status):
    icon, label = STATUS[status]
    return f'<span class="badge {status}">{icon} {esc(label)}</span>'


def page(title, body, nav):
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>"
            f"<nav>{nav}</nav>{body}</main></body></html>\n")


def navigation(result, current=None):
    links = ['<a href="index.html">Summary</a>' if current else "<strong>Summary</strong>"]
    if "background" in result.extra:
        links.append("<strong>Background removal</strong>" if current == "background"
                     else f'<a href="{BACKGROUND_PAGE}">Background removal</a>')
    for r in result.cases:
        name = esc(r.flow.name)
        links.append(f"<strong>{name}</strong>" if r.case.key == current else f'<a href="{r.case.key}.html">{name}</a>')
    return " · ".join(links)


def counts_of(checks):
    c = {k: 0 for k in STATUS}
    for ch in checks:
        c[ch.status] += 1
    return c


def counts_text(checks):
    c = counts_of(checks)
    parts = [f"{c['pass']} passed", f"{c['fail']} failed", f"{c['known']} known issues"]
    if c["fixed"]:
        parts.append(f"{c['fixed']} known issues now passing")
    if c["missing"]:
        parts.append(f"{c['missing']} not computed")
    return ", ".join(parts) + "."


def delta_html(key, value, previous, lower_is_better=True):
    if previous is None or value is None or not np.isfinite(value) or not np.isfinite(previous):
        return "—"
    d = value - previous
    if abs(d) <= 1e-12 + 1e-6 * abs(previous):
        return "="
    worse = (d > 0) == lower_is_better
    return f'<span class="delta-{"worse" if worse else "better"}">{"+" if d > 0 else "−"}{fmt(key, abs(d))}</span>'


def checks_table(case_key, checks, previous):
    rows = []
    has_prev = previous is not None
    for c in checks:
        ch = c.check
        prev = previous.get(check_id(case_key, ch)) if has_prev else None
        note = f'<div class="note">{esc(ch.known_issue)}</div>' if ch.known_issue else ""
        cells = [badge(c.status), f"{esc(ch.label)}{note}", esc(scenario_name(ch.level, ch.scenario)),
                 fmt(ch.metric, c.value), esc(fmt_limit(ch, c.limit))]
        if has_prev:
            cells.append(fmt(ch.metric, prev["value"]) + f' <span class="note">({esc(STATUS[prev["status"]][1].lower())})'
                         "</span>" if prev else "—")
        rows.append("<tr>" + "".join(f'<td{" class=num" if i == 3 else ""}>{cell}</td>'
                                     for i, cell in enumerate(cells)) + "</tr>")
    head = ["Status", "Check", "Scenario", "Value", "Tolerance"] + (["Previous run"] if has_prev else [])
    return ('<div class="scroll"><table><thead><tr>' + "".join(
        f'<th{" class=num" if h == "Value" else ""}>{h}</th>' for h in head) + "</tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>")


def table_of(columns, rows_of, level):
    """A table of metrics: columns [(title, {metric: value})], the rows all their metrics, in the order of
    METRICS."""
    keys = []
    for _, values in columns:
        for k in values:
            if k not in keys:
                keys.append(k)
    order = list(METRICS)
    keys.sort(key=lambda k: (order.index(k) if k in order else len(order), k))
    body = "".join(f"<tr><td>{esc(metric_label(k, level))}</td>"
                   + "".join(f'<td class="num">{fmt(k, values.get(k))}</td>' for _, values in columns) + "</tr>"
                   for k in keys)
    head = "".join(f'<th class="num">{esc(t)}</th>' for t, _ in columns)
    return (f'<div class="scroll"><table><thead><tr><th>Metric</th>{head}</tr></thead><tbody>{body}'
            "</tbody></table></div>")


def figures_html(figures):
    return "".join(f'<figure><img src="figures/{esc(file)}" alt="{esc(title)}" loading="lazy">'
                   f"<figcaption><strong>{esc(title)}.</strong> {esc(caption)}</figcaption></figure>"
                   for title, caption, file in figures)


def case_page(result, r, folder, previous):
    from ..piv.regression import check_regression, describe_regression
    from .post_level import POST_MAPS, TURBULENCE_MAPS

    case, flow = r.case, r.flow
    parts = [f"<h1>{esc(flow.name)}</h1>", f'<p class="lede">{esc(flow.description)}</p>']
    params = dict(flow.parameters())
    params["Images (px)"] = "512 × 320, 0.2 mm/px, 1000 frames/s (1 px/pair = 0.2 m/s)"
    params["Grid spacing / bin size (px)"] = case.spacing
    if r.status is not None:
        params["Grid points (flow / masked)"] = f"{int((r.status == 0).sum())} / {int((r.status == 2).sum())}"
    params["Post-processing level"] = f"{POST_MAPS if flow.steady else TURBULENCE_MAPS} maps per scenario"
    params["Processing level"] = (f"{case.pairs} pairs per scenario, {case.seeding:g} particles per pixel; binning "
                                  + ", ".join(f"{k.replace('_', ' ')} {v:g}" for k, v in case.binning.items()))
    params["SPICY"] = describe_regression(check_regression({"method": "spicy", "settings": case.regression}))
    parts.append("<h3>Flow and sampling</h3><dl>" + "".join(
        f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in params.items()) + "</dl>")
    parts.append(f"<p class=meta>All checks: {counts_text(r.checks)}</p>")
    figs = os.path.join(folder, "figures")
    for level in ("post", "processing", "images", "chain"):
        if level not in r.metrics:
            continue
        name, description = LEVELS[level]
        checks = [c for c in r.checks if c.check.level == level]
        parts.append(f"<h2>{esc(name)}</h2><p class=lede>{esc(description)}</p>")
        if level == "processing":
            for s, d in r.data.items():
                for e in d.errors:
                    parts.append(f'<p>{badge("fail")} {esc(scenario_name(level, s))}: {esc(e)}</p>')
        parts.append(f"<h3>Checks</h3><p class=meta>{counts_text(checks)}</p>")
        parts.append(checks_table(case.key, checks, previous))
        parts.append("<h3>Metrics</h3>")
        if level == "images":
            parts.append(images_section(r.metrics[level], r.runs, figs, case.key))
        elif level == "chain":
            parts.append(chain_section(r.metrics[level], r.runs))
        elif level == "post":
            parts.append("<p class=meta>In px and pairs (VelociHub's physical values divided by the camera's). "
                         "Medians over the grid points of the flow; the expected values: the exact mean, the noise, "
                         "the statistics of the modes of the turbulence.</p>")
            parts.append(table_of([(scenario_name(level, s), v["post"]) for s, v in r.metrics[level].items()],
                                  None, level))
            parts.append(figures_html(post_figures(r, figs)))
        else:
            parts.append("<p class=meta>Over all the pairs and the grid points of the flow (those with a value "
                         "for the binning). The exact column is the floor set by the grid: its derivatives are "
                         "computed by differences on the grid, as in the post-processing.</p>")
            for s, ms in r.metrics[level].items():
                parts.append(f"<h3>{esc(scenario_name(level, s))}</h3>"
                             + table_of([(SHORT_NAMES[m], ms[m]) for m, _ in METHODS if m in ms], None, level))
            parts.append(figures_html(processing_figures(r, figs)))
    return page(f"{flow.name} — VelociHub benchmark", "".join(parts), navigation(result, case.key))


def run_errors(runs, chain=False):
    """The runs of the PIV/PTV on images that failed (records of image_level.run_measurement), as lines."""
    out = []
    for style, records in runs.items():
        for folder, rec in records.items():
            if rec.get("file") is None and bool(rec.get("spicy")) <= chain:
                out.append(f'<p>{badge("fail")} {esc(style)} images, {esc(folder)}: {esc(rec.get("status"))} '
                           f'{esc(rec.get("message") or "")}</p>')
    return "".join(out)


def run_times(runs):
    """The time of the runs (s), by style of images."""
    return {style: sum(rec.get("seconds") or 0 for rec in records.values()) for style, records in runs.items()}


def images_section(metrics, runs, figs, key):
    """The level images of a case (or of the background test): the recipes on its test images, then compared on
    its hard images."""
    from .images import CLEAN, HARD

    parts = [run_errors({s: rr for s, rr in runs.items()})]
    for style in (CLEAN, HARD):
        if not recipes_of(metrics, style.key):
            continue
        title = ("Test images" if style is CLEAN else "Hard images: the pre-processing compared")
        parts.append(f"<h3>{esc(title)}</h3><p class=meta>{esc(style.description)} Pairs: consecutive frames. "
                     "Hover a recipe for what it does; the best value of each column in bold.</p>")
        for method, name in IMAGE_METHODS:
            parts.append(f"<p><strong>{esc(name)}.</strong> <span class=note>{esc(RECIPE_NOTES[method])}</span></p>"
                         + recipe_table(metrics, style.key, method))
        if style is HARD:
            file = recipe_figure(metrics, style.key, figs, f"{key}_recipes_{style.key}.png")
            if file:
                parts.append(figures_html([("The pre-processing on the hard images", "RMS error of the PIV, "
                                            "particles linked per pair and wrong links of the PTV for each recipe; "
                                            "the best of each darker.",
                                            file)]))
    return "".join(parts)


def chain_section(metrics, runs):
    """The level chain of a case (or of the background test): each scenario, its methods post-processed."""
    parts = [run_errors(runs, chain=True),
             "<p class=meta>The results of the PIV/PTV of the images post-processed with the default recipe, in px "
             "and pairs. No noise is added: the errors of the measurement are the fluctuations (RMS), so the "
             "ratios of the turbulence include them. The field error is that of the instantaneous fields.</p>"]
    for s, ms in metrics.items():
        parts.append(f"<h3>{esc(scenario_name('chain', s))}</h3>"
                     + table_of([(SHORT_NAMES[m], ms[m]) for m, _ in CHAIN_METHODS if m in ms], None, "chain"))
    return "".join(parts)


BACKGROUND_METHOD = """<h3>How what is left is measured</h3>
<ul>
<li><strong>Each reflection, by a matched filter of its shape.</strong> Its exact image is known (rendered). Over its
support (where it is brighter than 5 % of its peak, out of the mask), the least-squares factor of that shape in the
processed image, both with their mean removed: 100 % is the reflection untouched, 0 % removed, below 0 over-subtracted.
It is divided by the laser energy of each image and by the fraction of the particles kept (the removal may scale the
intensities), and averaged over the images. Times the peak of the reflection, the peak left in grey levels.</li>
<li><strong>The lines and circles, by a Hough transform</strong> of the mean processed image (OpenCV: Gaussian blur, Canny
edges then probabilistic Hough lines, at least 80 px long; Hough gradient circles within 0.8–1.25 of the radius). A
line is found if a segment lies within 12 px and 6° of it, a circle if one is within 4 px of its centre and 5 px of its
radius. This is how a detector would see what is left, independently of the known amplitude.</li>
<li><strong>The particles, as a whole</strong>: the same matched filter with the image of all the particles of each
frame. <strong>One by one</strong>: each isolated particle (no other within 5 px) measured by the least-squares
amplitude of its Gaussian spot plus a constant over a 7 × 7 px window, over its raw peak; over a reflection or not.</li>
<li><strong>The background left</strong>: the mean processed image less the particles (times the fraction kept), in
grey levels, and the pixels at 0: the processed image is cast back to the depth of the raw one (8 bits), so its
negative values are clipped, which also cuts the tails of the particles.</li>
</ul>"""


def feature_table(bg):
    """What each removal leaves of each reflection: % of its brightness, its peak left (grey levels), found by the
    Hough transform."""
    data = bg.data
    modes = sorted(data["modes"])
    rows = []
    for key, (kind, geometry) in data["features"].items():
        cells = [esc(key.capitalize()), esc(kind)]
        for n in modes:
            m = data["modes"][n]
            found = m.get(f"hough_{key}")
            hough = "" if found is None else (" · found" if found else " · not found")
            cells.append(f"{fmt('kept_' + key, m.get('kept_' + key))} % · {fmt('left_' + key, m.get('left_' + key))}"
                         f"{hough}")
        rows.append("<tr>" + "".join(f"<td{' class=num' if i > 1 else ''}>{c}</td>" for i, c in enumerate(cells))
                    + "</tr>")
    particles = ["Particles", "spots"] + [
        f"{fmt('particle_kept_median', data['modes'][n].get('particle_kept_median'))} % · "
        f"{fmt('particles_dimmed', data['modes'][n].get('particles_dimmed'))} % dimmed" for n in modes]
    rows.append("<tr>" + "".join(f"<td{' class=num' if i > 1 else ''}>{c}</td>" for i, c in enumerate(particles))
                + "</tr>")
    head = "".join(f'<th class="num">{esc(scenario_name("preprocessing", f"modes_{n}"))}</th>' for n in modes)
    return ("<h3>What is left of each feature</h3><p class=meta>% of its brightness · its peak left in grey levels "
            "(camera noise: 2) · found or not by the Hough transform. Particles: median of each one's brightness kept "
            "· below half.</p>"
            f'<div class="scroll"><table><thead><tr><th>Feature</th><th>Shape</th>{head}</tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def background_page(result, folder, previous):
    bg = result.extra["background"]
    name, description = LEVELS["preprocessing"]
    parts = [f"<h1>Background removal</h1><p class=lede>{esc(description)} The cylinder flow with stationary "
             "reflections (the measurement background_removal of the test workspace), its POD with the mask of the "
             "cylinder, and the images processed with 0 to 3 modes removed; then the same images through the "
             "PIV/PTV and the post-processing, with the background removal on and off.</p>",
             f"<h3>Checks</h3><p class=meta>{counts_text(bg.checks)}</p>", checks_table("background", bg.checks, previous)]
    if bg.data is not None:
        parts += [BACKGROUND_METHOD, feature_table(bg), "<h3>Metrics</h3>",
                  table_of([(scenario_name("preprocessing", s), v["pod"]) for s, v in bg.metrics.items()], None,
                           "preprocessing"),
                  figures_html(background_figures(bg, os.path.join(folder, "figures")))]
    for level in ("images", "chain"):
        if bg.levels.get(level):
            parts.append(f"<h2>{esc(LEVELS[level][0])}</h2>")
            parts.append(images_section(bg.levels[level], {"clean": bg.runs}, os.path.join(folder, "figures"),
                                        "background") if level == "images" else
                         chain_section(bg.levels[level], {"clean": bg.runs}))
    return page("Background removal — VelociHub benchmark", "".join(parts), navigation(result, "background"))


def _issue_rows(result, statuses):
    rows = []
    entries = [(r.flow.name, f"{r.case.key}.html", r.checks) for r in result.cases]
    if "background" in result.extra:
        entries.insert(0, ("Background removal", BACKGROUND_PAGE, result.extra["background"].checks))
    for name, link, checks in entries:
        for ch in checks:
            if ch.status in statuses:
                rows.append(f'<tr><td>{badge(ch.status)}</td><td><a href="{link}">{esc(name)}</a></td>'
                            f"<td>{esc(LEVELS[ch.check.level][0])}</td><td>{esc(ch.check.label)}"
                            f'{"<div class=note>" + esc(ch.check.known_issue) + "</div>" if ch.check.known_issue else ""}'
                            f'</td><td class="num">{fmt(ch.check.metric, ch.value)}</td>'
                            f"<td>{esc(fmt_limit(ch.check, ch.limit))}</td></tr>")
    return rows


def _level_badge(checks):
    if not checks:
        return '<span class="note">not run</span>'
    c = counts_of(checks)
    state = "pass" if c["pass"] + c["known"] == len(checks) else "fail"
    return (f'{badge(state)} <span class="note">{c["pass"]}/{len(checks)}'
            f'{", " + str(c["known"]) + " known" if c["known"] else ""}</span>')


def summary_page(result, chart, previous, compare_label, recipes_chart=None):
    from .image_level import ranking

    c = result.counts()
    env = result.environment
    parts = ["<h1>VelociHub benchmark</h1>",
             '<p class="lede">Analytic flows, whose fields are known exactly, run through VelociHub step by step: '
             "the post-processing alone (the exact fields as results), the processing then the post-processing "
             "(particle links and PIV vectors sampled from the exact fields, without images), the background "
             "removal of particle images with reflections, and particle images through the pre-processing, the "
             "PIV/PTV and the post-processing (the pre-processing methods compared on harder images).</p>",
             f'<p class="meta">{esc(env["date"])} · VelociHub {esc(env["velocihub"])}'
             f'{" · " + esc(env["commit"]) if env.get("commit") else ""} · spicy_vki {esc(env.get("spicy_vki"))} · '
             f'Python {esc(env["python"])} · {result.seconds:.0f} s'
             f'{" · compared with " + esc(compare_label) if compare_label else ""}</p>']
    tiles = [("pass", "checks passed"), ("fail", "checks failed"), ("known", "known issues"),
             ("fixed", "known issues now passing")]
    parts.append('<div class="tiles">' + "".join(
        f'<div class="tile"><div class="n">{c[k]}</div><div class="l">{badge(k)} {esc(label)}</div></div>'
        for k, label in tiles if k != "fixed" or c[k]) + "</div>")
    failures = result.failures()
    verdict = ("All the checks are as expected." if not failures else
               f"{len(failures)} checks are not as expected: see below.")
    parts.append(f"<p><strong>{esc(verdict)}</strong> A failed check is a regression, or an improvement that should "
                 "tighten its tolerance; a known issue is a limitation documented by a check expected to fail.</p>")
    # The levels of the cases
    levels = [lv for lv in ("post", "processing", "images", "chain") if lv in result.levels]
    head = ["Case"] + [LEVELS[lv][0] for lv in levels] + ["Post: mean error (noise)", "Post: vorticity floor",
                                                         "SPICY (PTV)", "Binned (PTV)", "SPICY (PIV)"]
    if previous:
        head.append("Δ SPICY (PTV) vs previous")
    rows = []
    if "background" in result.extra:
        bg = result.extra["background"]
        rows.append(f'<tr><td><a href="{BACKGROUND_PAGE}">Background removal</a> '
                    f'<span class="note">(pre-processing)</span></td><td colspan="{len(head) - 1}">'
                    f"{_level_badge(bg.checks)}</td></tr>")
    for r in result.cases:
        cells = [f'<a href="{r.case.key}.html">{esc(r.flow.name)}</a>']
        for lv in levels:
            cells.append(_level_badge([c for c in r.checks if c.check.level == lv]))
        post = r.metrics.get("post", {})
        cells.append(fmt("mean_error", post.get("noise", {}).get("post", {}).get("mean_error")))
        cells.append(fmt("vort_mean_error", post.get("exact", {}).get("post", {}).get("vort_mean_error")))
        m = r.metrics.get("processing", {}).get("nominal", {})
        for meth in ("spicy_ptv", "binned", "spicy_piv"):
            cells.append(fmt("rms", m.get(meth, {}).get("rms")))
        if previous:
            prev = previous.get("cases", {}).get(r.case.key, {}).get("metrics", {})
            prev = prev.get("processing", prev).get("nominal", {})
            cells.append(delta_html("rms", m.get("spicy_ptv", {}).get("rms"), prev.get("spicy_ptv", {}).get("rms")))
        rows.append("<tr>" + "".join(f'<td{" class=num" if i > len(levels) else ""}>{x}</td>'
                                     for i, x in enumerate(cells)) + "</tr>")
    parts.append("<h2>Cases</h2><p class=meta>Checks by level. Post-processing: the RMS error of the mean of 200 "
                 "noisy maps (px), and the error of the vorticity of the exact mean (10⁻³/pair: the differences on "
                 "the grid). Processing: the RMS error of the displacement (px), noise scenario (0.1 px on each "
                 "component of the data, 0.14 px on the vector).</p>")
    parts.append('<div class="scroll"><table><thead><tr>' + "".join(
        f'<th{" class=num" if i > len(levels) else ""}>{esc(h)}</th>' for i, h in enumerate(head))
        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
    if chart:
        parts.append(f'<figure><img src="figures/{esc(chart)}" alt="RMS error per case and method">'
                     "<figcaption><strong>Processing level: RMS error per case and method</strong>, noise scenario. "
                     "The dashed line is the noise of the data each method is given.</figcaption></figure>")
    hard = [r for r in result.cases if recipes_of(r.metrics.get("images", {}), "hard")]
    if hard:
        parts.append("<h2>Which pre-processing helps the PIV/PTV</h2><p class=meta>The hard images of each flow (dim "
                     "particles of varied brightness and size, camera noise, an uneven light sheet, reflections) "
                     "with each recipe: the best for each measure, and its value against no pre-processing. See the "
                     "page of each flow for all of them.</p>")
        rows = []
        for r in hard:
            m = r.metrics["images"]
            best = ranking(m)
            cells = [f'<a href="{r.case.key}.html">{esc(r.flow.name)}</a>']
            for method, key in (("piv", "rms"), ("piv", "outliers"), ("ptv", "links_per_pair"),
                                ("ptv", "wrong_links"), ("ptv", "track_length_median")):
                order = best.get(method, {}).get(key, [])
                if not order:
                    cells.append("—")
                    continue
                top = order[0]
                cells.append(f"{esc(RECIPES[top].name)}<br><span class=note>{fmt(key, m[f'hard_{top}'][method][key])}"
                             f" vs {fmt(key, m.get('hard_none', {}).get(method, {}).get(key))} without</span>")
            rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        head = ["Flow", "PIV: lowest RMS error (px)", "PIV: fewest outliers (%)", "PTV: most links per pair",
                "PTV: fewest wrong links (%)", "PTV: longest tracks (frames)"]
        parts.append('<div class="scroll"><table><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in head)
                     + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
        if recipes_chart:
            parts.append(f'<figure><img src="figures/{esc(recipes_chart)}" alt="The pre-processing compared">'
                         "<figcaption><strong>The pre-processing on the hard images</strong>, relative to none "
                         "(dashed line): below it, a lower error of the PIV, fewer wrong links of the PTV."
                         "</figcaption></figure>")
    for title, statuses in (("Checks not as expected", ("fail", "fixed", "missing")), ("Known issues", ("known",))):
        rows = _issue_rows(result, statuses)
        parts.append(f"<h2>{title}</h2>")
        parts.append('<div class="scroll"><table><thead><tr><th>Status</th><th>Case</th><th>Level</th><th>Check</th>'
                     '<th class="num">Value</th><th>Tolerance</th></tr></thead><tbody>' + "".join(rows)
                     + "</tbody></table></div>" if rows else "<p class=meta>None.</p>")
    parts.append("<h2>Levels and methods</h2><dl>" + "".join(
        f"<dt>{esc(name)}</dt><dd>{esc(desc)}</dd>" for name, desc in LEVELS.values()) + "".join(
        f"<dt>{esc(SHORT_NAMES[m])}</dt><dd>{esc(desc)}</dd>" for m, desc in (
            ("exact", "The exact field sampled at the grid points: the floor of the derivatives on this grid."),
            ("spicy_ptv", "SPICY fitted on the particle links of each pair (the PTV path), evaluated on the grid."),
            ("binned", "The particle links averaged in the cells of the grid, with the median test (the PTV without "
                       "regression)."),
            ("spicy_piv", "SPICY fitted on vectors at the grid points (the PIV path)."))) + "</dl>")
    parts.append("<h2>Environment</h2><dl>" + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in env.items())
                 + "</dl>")
    return page("VelociHub benchmark", "".join(parts), navigation(result))


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def _checks_json(key, checks):
    return [{"id": check_id(key, c.check), "level": c.check.level, "label": c.check.label,
             "scenario": c.check.scenario, "method": c.check.method, "metric": c.check.metric, "op": c.check.op,
             "value": _clean(c.value), "limit": _clean(c.limit), "status": c.status,
             "known_issue": c.check.known_issue} for c in checks]


def _runs_json(runs):
    return {style: {folder: {k: _clean(rec.get(k)) for k in ("status", "message", "seconds")}
                    for folder, rec in records.items()} for style, records in runs.items()}


def to_json(result):
    cases = {}
    for r in result.cases:
        cases[r.case.key] = {
            "name": r.flow.name, "parameters": {k: str(v) for k, v in r.flow.parameters().items()},
            "seeding": r.case.seeding, "pairs": r.case.pairs, "spacing": r.case.spacing,
            "regression": r.case.regression, "binning": r.case.binning, "metrics": _clean(r.metrics),
            "checks": _checks_json(r.case.key, r.checks),
            "errors": {s: d.errors for s, d in r.data.items() if d.errors},
            "runs": _runs_json(r.runs)}
    out = {"format": "VelociHub benchmark", "levels": list(result.levels), "environment": result.environment,
           "seconds": result.seconds, "counts": result.counts(), "cases": cases}
    if "background" in result.extra:
        bg = result.extra["background"]
        out["background"] = {"metrics": _clean(bg.metrics), "checks": _checks_json("background", bg.checks),
                             "levels": _clean({k: v for k, v in bg.levels.items() if k != "preprocessing"}),
                             "runs": _runs_json({"clean": bg.runs})}
    return out


def load_previous(compare):
    """results.json of a previous report (its folder or the file); None if it can't be read."""
    if not compare:
        return None
    path = os.path.join(compare, "results.json") if os.path.isdir(compare) else compare
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_report(result, folder, compare=None):
    """Write the report of result in folder (created): index.html, a page per case, background.html, figures/,
    results.json. compare: a previous report (folder or results.json) whose values are shown beside. Returns
    index.html."""
    os.makedirs(os.path.join(folder, "figures"), exist_ok=True)
    previous = load_previous(compare)
    prev_checks = None
    label = None
    if previous:
        prev_checks = {c["id"]: c for case in previous.get("cases", {}).values() for c in case["checks"]}
        prev_checks.update({c["id"]: c for c in previous.get("background", {}).get("checks", [])})
        env = previous.get("environment", {})
        label = f"the run of {env.get('date', '?')}" + (f" ({env['commit']})" if env.get("commit") else "")
    for r in result.cases:
        with open(os.path.join(folder, f"{r.case.key}.html"), "w", encoding="utf-8") as f:
            f.write(case_page(result, r, folder, prev_checks))
    if "background" in result.extra:
        with open(os.path.join(folder, BACKGROUND_PAGE), "w", encoding="utf-8") as f:
            f.write(background_page(result, folder, prev_checks))
    chart = summary_chart(result, os.path.join(folder, "figures"))
    recipes_chart = recipes_summary_figure(result, os.path.join(folder, "figures"))
    index = os.path.join(folder, "index.html")
    with open(index, "w", encoding="utf-8") as f:
        f.write(summary_page(result, chart, previous, label, recipes_chart))
    with open(os.path.join(folder, "results.json"), "w", encoding="utf-8") as f:
        json.dump(to_json(result), f, indent=1, ensure_ascii=False)
    return index


def reports_folder():
    """The folder of the reports: benchmark_reports in the folder of the repository of VelociHub (left out of git by
    its .gitignore), or ~/VelociHub/benchmarks when VelociHub is installed without its repository."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.isfile(os.path.join(repo, "pyproject.toml")) and os.path.isdir(os.path.join(repo, "velocihub")):
        return os.path.join(repo, "benchmark_reports")
    return os.path.join(os.path.expanduser("~"), "VelociHub", "benchmarks")


def default_folder():
    """A new folder for a report, named by the date and time of the run: reports_folder()/report_<date>_<time>."""
    return os.path.join(reports_folder(), f"report_{datetime.now():%Y%m%d_%H%M%S}")
