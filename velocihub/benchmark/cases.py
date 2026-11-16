"""The cases of the benchmark at the regression level: an analytic flow (flows.py) sampled as a PTV and a PIV would
give it, without images, and the checks of the results against the exact field.

A case is run in scenarios (SCENARIOS): the measurement noise of every case, and the same with wrong links among the
particles (what the tracking of real images gives). Each scenario compares the methods (METHODS) on the grid:

- exact: the exact field sampled at the grid points. Its errors are 0; those of its derivatives (by finite differences
  on the grid, as the post-processing computes them) are the floor that no method can beat on this grid.
- spicy_ptv: SPICY fitted on the links of the particles (the PTV path of VelociHub), evaluated on the grid.
- binned: the links of the particles binned on the grid (the PTV without regression).
- spicy_piv: SPICY fitted on vectors at the grid points (the PIV path of VelociHub).

A check compares a metric of a method (a number, or a metric of another method) and has a tolerance. The tolerances are
those of VelociHub as it is (with a margin): a check that fails after a change is a regression, or an improvement
that should tighten its tolerance. A check expected to fail (known_issue) documents a limitation: it is reported, and
it fails the tests only when it starts passing (the limitation solved: the check should then become a normal one).
"""

from dataclasses import dataclass, field

from .flows import FLOWS

# Methods compared, in the order of the report: key, name
METHODS = (("exact", "Exact on the grid"), ("spicy_ptv", "SPICY on particles (PTV)"),
           ("binned", "Binned particles (PTV)"), ("spicy_piv", "SPICY on vectors (PIV)"))


@dataclass(frozen=True)
class Scenario:
    """The corruption of the data of a case: noise of every displacement (px, standard deviation of each
    component), fraction of wrong links among the particles (displacement random within +- outlier_px), PIV vectors
    computed or not."""

    key: str
    name: str
    description: str
    noise_px: float = 0.1
    outliers: float = 0.0
    outlier_px: float = 8.0
    piv: bool = True


SCENARIOS = {s.key: s for s in (
    Scenario("nominal", "Noise", "Every displacement (particles and PIV vectors) with a Gaussian noise of 0.1 px on "
                                 "each component."),
    Scenario("outliers", "Noise and wrong links", "As the noise scenario, with 2 % of the particle links wrong (a "
                                                  "random displacement within ±8 px), as the tracking of real images "
                                                  "gives some. PTV only.", outliers=0.02, piv=False),
)}

# The scenarios of the post-processing level: the exact fields, and with a noise
POST_SCENARIOS = {s.key: s for s in (
    Scenario("exact", "Exact fields", "The exact fields of the flow, without noise.", noise_px=0.0),
    Scenario("noise", "Noise", "The exact fields with a Gaussian noise of 0.1 px on each component of every vector "
                               "(independent between the maps and the grid points)."),
)}

# The levels of the benchmark, from the last step of the processing back: key, name, description
LEVELS = {
    "preprocessing": ("Pre-processing", "Particle images with stationary reflections of known shape: how much of them "
                                        "the POD background removal leaves (matched filter, Hough transform), and of "
                                        "the particles."),
    "post": ("Post-processing", "The exact fields written as PIV results and post-processed, without PIV/PTV: the "
                                "statistics and derivatives of VelociHub against the samples and the analytic values."),
    "processing": ("Processing → post", "Particle links and PIV vectors sampled from the exact field (without "
                                        "images), binned and regressed by VelociHub, saved as its results "
                                        "(vectors.npz) and post-processed: the fields against the exact field."),
}


@dataclass(frozen=True)
class Check:
    """metric of method (in scenario) op limit: limit a number or "method.metric" (the same metric of another
    method, or another one), times factor. known_issue: the reason it is expected to fail. level: of the benchmark
    (LEVELS)."""

    scenario: str
    method: str
    metric: str
    op: str
    limit: object
    label: str
    factor: float = 1.0
    known_issue: str = ""
    level: str = "processing"

    def __post_init__(self):
        assert self.op in ("<=", ">=", "<", ">", "within"), self.op


def below(scenario, method, metric, limit, label, **kw):
    return Check(scenario, method, metric, "<=", limit, label, **kw)


def above(scenario, method, metric, limit, label, **kw):
    return Check(scenario, method, metric, ">=", limit, label, **kw)


def within(scenario, method, metric, low, high, label, **kw):
    return Check(scenario, method, metric, "within", (low, high), label, **kw)



# The known issues of VelociHub as it is (the reasons of the checks expected to fail)
FEW_PER_RBF = ("With the default 4 points per RBF, each RBF is fitted on about 4 particles: their noise is little "
               "averaged, so the fit is noisier than the bins of the grid.")
EDGES = ("The RBFs at the edges of the field of view (and of the masks) are constrained by data on one side only: "
         "the fit is worse there.")
PIV_RESOLUTION = ("With 4 vectors per RBF the RBFs are about 2 grid spacings apart, and wide (0.7 at the nearest one): "
                  "structures shorter than about 5 grid spacings are smoothed out of the PIV vectors.")
UNRESOLVED = ("The shortest wavelength (3 grid spacings) is not resolved by the RBFs (about 21 px apart): the "
              "residual is that mode.")

# Verification: equal to the rounding of the files (float32)
ROUNDING = 1e-4


def common_checks(rms, rms_piv, vort, known=None):
    """The checks of every case at the processing level: the accuracy of SPICY on particles (rms, vort) and on
    vectors (rms_piv) as it is, the behaviour expected of the methods, and the chain to the post-processing. known:
    {check name: reason} of the behaviours that fail in this case (known issues): "better_than_binning", "residual",
    "edges", "piv_better_than_noise"."""
    known = known or {}
    checks = [
        below("nominal", "spicy_ptv", "rms", rms, "SPICY (PTV): RMS error of the displacement (px)"),
        below("nominal", "spicy_ptv", "vort_rms", vort, "SPICY (PTV): RMS error of the vorticity (10⁻³/pair)"),
        below("nominal", "spicy_piv", "rms", rms_piv, "SPICY (PIV): RMS error of the displacement (px)"),
        below("nominal", "spicy_ptv", "rms", "binned.rms", "SPICY on particles more accurate than the binning",
              known_issue=known.get("better_than_binning", "")),
        within("nominal", "spicy_ptv", "residual_ratio", 0.8, 1.2,
               "SPICY (PTV): residual at the data ≈ noise of the data (neither under- nor overfitted)",
               known_issue=known.get("residual", "")),
        below("nominal", "spicy_ptv", "rms_edge", "spicy_ptv.rms_interior",
              "SPICY (PTV): at the edges of the grid, about as accurate as inside", factor=1.5,
              known_issue=known.get("edges", "")),
        below("nominal", "spicy_piv", "rms", 0.1 * 2 ** 0.5,
              "SPICY (PIV): more accurate than the PIV vectors it fits (their noise, px)",
              known_issue=known.get("piv_better_than_noise", "")),
        below("outliers", "binned", "rms", "nominal:binned.rms",
              "Binning with wrong links: about as accurate as without (its median test)", factor=1.25),
        below("outliers", "spicy_ptv", "rms", "nominal:spicy_ptv.rms",
              "SPICY (PTV) with wrong links: about as accurate as without (the outlier test of the links)",
              factor=1.15),
        below("nominal", "spicy_ptv", "links_rejected", 1.0,
              "SPICY (PTV): good links rejected by the outlier test (%; no wrong link in this scenario)"),
        within("outliers", "spicy_ptv", "links_rejected", 1.7, 2.6,
               "SPICY (PTV): links rejected with 2 % of them wrong (%)"),
    ]
    checks += [
        within("nominal", "binned", "unc_ratio", 0.85, 1.15,
               "Binned (PTV): uncertainty predicted from the flow / actual RMS error (uncertainty.py)"),
        within("nominal", "binned", "unc_z", 0.9, 1.1,
               "Binned (PTV): (error − predicted bias) / predicted random error, standard deviation 1"),
        within("nominal", "binned", "unc_cover_1", 62, 74,
               "Binned (PTV): errors within one predicted uncertainty (%; 68 for a Gaussian error)"),
        within("nominal", "binned", "unc_cover_2", 92, 98,
               "Binned (PTV): errors within two predicted uncertainties (%; 95 for a Gaussian error)"),
    ]
    for method, name in (("spicy_ptv", "SPICY (PTV)"), ("binned", "binned (PTV)"), ("spicy_piv", "SPICY (PIV)")):
        checks.append(below("nominal", method, "post_vs_fields", ROUNDING,
                            f"Results {name} saved, read and post-processed: mean = mean of the fields read (px)"))
    return checks


def post_checks(vort=None, gradient=None, steady=True):
    """The checks of the post-processing level: the verification against the samples (the rounding), the analytic
    values of the noise (its statistics, the vorticity of differences of white noise), the vorticity of the mean
    (vort: its RMS error, 1/pair, the floor of the grid) and the mean gradient dU/dY (gradient: its RMS error, % of its
    largest value: the smoothing of the post-processing)."""
    def post(scenario, op, metric, limit, label, **kw):
        if op == "within":
            return within(scenario, "post", metric, limit[0], limit[1], label, level="post", **kw)
        return (below if op == "<=" else above)(scenario, "post", metric, limit, label, level="post", **kw)

    checks = [
        post("exact", "<=", "positions_error", 1e-6, "Positions of the grid: x right, y up from the bottom-left "
                                                     "corner (px)"),
        post("exact", "<=", "units_error", ROUNDING, "Velocities = displacements × pixel pitch × frame rate (px)"),
    ]
    for scenario in ("exact", "noise"):
        checks += [
            post(scenario, "<=", "mean_vs_samples", ROUNDING, "Mean = mean of the samples (px)"),
            post(scenario, "<=", "rms_vs_samples", ROUNDING, "RMS = sample standard deviation of the samples (px)"),
            post(scenario, "<=", "uv_vs_samples", ROUNDING, "Reynolds stress = sample covariance of the samples "
                                                            "(px²)"),
        ]
    if vort is not None:
        checks.append(post("exact", "<=", "vort_mean_error", vort,
                           "Vorticity of the mean: RMS error (10⁻³/pair; the differences on the grid)"))
    if gradient is not None:
        checks += [
            post("exact", "<=", "gradient_error", gradient,
                 "Mean gradient dU/dY: RMS error (% of its largest value; smoothed over the grid)"),
            post("exact", "<=", "gradient_vs_code", ROUNDING,
                 "Mean gradient recomputed from the mean field (smoothing of gradient_sigma cells, central "
                 "differences) = that of VelociHub (1/pair): the study of its parts is that of the code"),
            post("exact", "<=", "gradient_vs_code_after", ROUNDING,
                 "The same, smoothed after the differences (gradient_smooth_after) = that of VelociHub (1/pair)"),
        ]
    if steady:
        checks += [
            post("exact", "<=", "mean_error", ROUNDING, "Mean = the exact field (px)"),
            post("exact", "<=", "rms", ROUNDING, "No fluctuation in identical maps (px)"),
            post("noise", "<=", "mean_error", 0.015, "With noise: mean = the exact field (px; noise / √maps)"),
            post("noise", "within", "rms_u_ratio", (0.97, 1.03), "With noise: RMS of u / noise"),
            post("noise", "within", "rms_v_ratio", (0.97, 1.03), "With noise: RMS of v / noise"),
            post("noise", "within", "tke_ratio", (0.95, 1.05), "With noise: k / noise²"),
            post("noise", "within", "ti_ratio", (0.97, 1.03), "With noise: turbulence intensity / its value"),
            post("noise", "<=", "uv_error", 0.02, "With noise: no Reynolds stress (|<u'v'>| / noise²)"),
            post("noise", "within", "skewness", (-0.05, 0.05), "With noise: skewness 0"),
            post("noise", "within", "kurtosis", (2.85, 3.1), "With noise: kurtosis 3 (a little less: the outliers "
                                                             "beyond 4 σ removed)"),
            post("noise", "within", "vort_rms_ratio", (0.95, 1.05),
                 "With noise: RMS of the vorticity / noise / grid spacing (central differences of white noise)"),
        ]
    return checks


def turbulence_post_checks():
    """The statistics of the synthetic turbulence against those of its modes."""
    def post(scenario, op, metric, limit, label):
        if op == "within":
            return within(scenario, "post", metric, limit[0], limit[1], label, level="post")
        return (below if op == "<=" else above)(scenario, "post", metric, limit, label, level="post")

    checks = []
    for scenario, what in (("exact", ""), ("noise", "With noise: ")):
        checks += [
            post(scenario, "<=", "mean_error", 0.03, f"{what}Mean = U0 (px; the record of 2000 maps)"),
            post(scenario, "within", "rms_u_ratio", (0.95, 1.05), f"{what}RMS of u / that of the modes"),
            post(scenario, "within", "rms_v_ratio", (0.95, 1.05), f"{what}RMS of v / that of the modes"),
            post(scenario, "within", "tke_ratio", (0.95, 1.05), f"{what}k / that of the modes"),
            post(scenario, "<=", "uv_error", 0.02, f"{what}Reynolds stress: that of the modes (error / k)"),
            post(scenario, "within", "skewness", (-0.05, 0.05), f"{what}Skewness 0"),
            post(scenario, "within", "kurtosis", (2.7, 3.1), f"{what}Kurtosis about 3 (a sum of many modes)"),
            post(scenario, "within", "lx_u_ratio", (0.85, 1.1), f"{what}Integral length L_x^u / that of the modes"),
            post(scenario, "within", "ly_u_ratio", (0.85, 1.1), f"{what}Integral length L_y^u / that of the modes"),
            post(scenario, "within", "lx_v_ratio", (0.85, 1.1), f"{what}Integral length L_x^v / that of the modes"),
            post(scenario, "within", "ly_v_ratio", (0.85, 1.1), f"{what}Integral length L_y^v / that of the modes"),
            post(scenario, "within", "t_ratio", (0.85, 1.05),
                 f"{what}Integral time T_u / L_x^u / U0 (frozen turbulence)"),
        ]
    checks += [
        post("exact", "within", "dissipation_grid_ratio", (0.95, 1.08),
             "Dissipation ε / its value with the differences of the grid (the modes' gradients through sin(kh)/h)"),
        post("exact", "within", "dissipation_ratio", (0.7, 0.9),
             "Dissipation ε / exact ε: below 1, the gradients the 16 px grid misses"),
    ]
    return checks


@dataclass
class Case:
    """A flow of the benchmark (flows.py: its key) with its sampling and its checks. seeding: particles per pixel of
    each pair; pairs: pairs (independent samples) per scenario; spacing: of the grid (px), and the size of the bins;
    regression: the SPICY settings changed from the defaults of VelociHub; binning: the settings of the PTV."""

    key: str
    checks: list
    seeding: float = 0.01
    pairs: int = 3
    spacing: int = 16
    regression: dict = field(default_factory=dict)
    binning: dict = field(default_factory=lambda: {"min_particles": 1, "median_test": 2.0})
    seed: int = 0

    def flow(self):
        return FLOWS[self.key]()


# The tolerances: the values of VelociHub 1.2 (SPICY defaults, spicy_vki 1.1.2) with a margin of about 30 %. The
# vorticity in 1/pair (shown in 10⁻³/pair).
def _cases():
    return [
        Case("uniform", common_checks(rms=0.16, rms_piv=0.125, vort=0.0095, known={
            "better_than_binning": FEW_PER_RBF, "edges": EDGES}) + [
            within("nominal", "spicy_ptv", "bias_u", -0.015, 0.015, "SPICY (PTV): no bias of U (px)"),
            within("nominal", "spicy_ptv", "bias_v", -0.015, 0.015, "SPICY (PTV): no bias of V (px)"),
            below("nominal", "spicy_ptv", "rms_edge", 0.3, "SPICY (PTV): RMS error at the edges of the grid (px)"),
        ] + post_checks(vort=ROUNDING), seed=10),
        Case("cylinder", common_checks(rms=0.18, rms_piv=0.18, vort=0.0095, known={
            "better_than_binning": FEW_PER_RBF, "edges": EDGES}) + [
            below("nominal", "spicy_ptv", "rms_body", 0.35, "SPICY (PTV): RMS error next to the cylinder (px)"),
            within("nominal", "spicy_ptv", "peak_speed_ratio", 0.95, 1.06,
                   "SPICY (PTV): highest speed / exact (the shoulders of the cylinder)"),
        ] + post_checks(vort=0.003, gradient=0.9), seed=20),
        Case("jet", common_checks(rms=0.15, rms_piv=0.21, vort=0.03, known={
            "piv_better_than_noise": PIV_RESOLUTION}) + [
            below("nominal", "spicy_ptv", "half_width_error", 1.0, "SPICY (PTV): RMS error of the jet half-width (px)"),
            above("nominal", "spicy_ptv", "shear_ratio", "exact.shear_ratio",
                  "SPICY (PTV): peak shear at least that of the exact field on the grid", factor=0.95),
        ] + post_checks(vort=0.03, gradient=14), seed=30),
        Case("modes", common_checks(rms=0.9, rms_piv=1.15, vort=0.15, known={
            "better_than_binning": UNRESOLVED, "residual": UNRESOLVED, "piv_better_than_noise": PIV_RESOLUTION}) + [
            above("nominal", "spicy_ptv", "transfer_320", 0.97, "SPICY (PTV): amplitude kept at λ = 320 px"),
            above("nominal", "spicy_ptv", "transfer_160", 0.95, "SPICY (PTV): amplitude kept at λ = 160 px"),
            above("nominal", "spicy_ptv", "transfer_80", 0.9, "SPICY (PTV): amplitude kept at λ = 80 px"),
            above("nominal", "spicy_ptv", "transfer_48", 0.2, "SPICY (PTV): amplitude kept at λ = 48 px"),
            above("nominal", "spicy_piv", "transfer_80", 0.9, "SPICY (PIV): amplitude kept at λ = 80 px",
                  known_issue=PIV_RESOLUTION),
        ] + post_checks(vort=0.1, gradient=30) + [
            below("exact", "post", "gradient_transfer_theory_error", 0.005,
                  "Mean dU/dY: amplitude kept at each wavelength = theory, sin(kh)/kh × the Gaussian smoothing "
                  "(largest difference)", level="post"),
        ], seed=40),
        Case("blasius", common_checks(rms=0.15, rms_piv=0.125, vort=0.008, known={"edges": EDGES}) + [
            below("nominal", "spicy_ptv", "displacement_thickness_error", 6.5,
                  "SPICY (PTV): RMS error of the displacement thickness δ* (%)"),
            below("nominal", "spicy_ptv", "momentum_thickness_error", 12.0,
                  "SPICY (PTV): RMS error of the momentum thickness θ (%)"),
            within("nominal", "spicy_ptv", "wall_gradient_ratio", 0.9, 1.1,
                   "SPICY (PTV): dU/dY next to the wall / exact"),
        ] + post_checks(vort=0.001, gradient=2.6) + [
            below("exact", "post", "gradient_error_after", 1.4,
                  "Mean dU/dY smoothed after the differences: RMS error (%; closer at the wall)", level="post"),
            within("exact", "post", "wall_gradient_ratio_after", 0.95, 1.1,
                   "Mean dU/dY smoothed after the differences, next to the wall / exact (none without the toggle)",
                   level="post"),
        ], seed=50),
        Case("turbulence", common_checks(rms=0.16, rms_piv=0.16, vort=0.014, known={"edges": EDGES})
             + post_checks(vort=0.001, steady=False)
             + turbulence_post_checks(), seed=60),
    ]


CASES = {case.key: case for case in _cases()}

# The background removal test (images.run_background): the modes removed, its scenarios
BACKGROUND_MODES = (0, 1, 2, 3)
BACKGROUND_SCENARIOS = {f"modes_{n}": Scenario(f"modes_{n}", "No mode removed" if n == 0 else
                                               f"{n} mode{'s' if n > 1 else ''} removed",
                                               f"The POD background removal with {n} modes removed.", noise_px=0.0)
                        for n in BACKGROUND_MODES}


def _background_checks():
    def pod(scenario, op, metric, limit, label):
        if op == "within":
            return within(scenario, "pod", metric, limit[0], limit[1], label, level="preprocessing")
        return (below if op == "<=" else above)(scenario, "pod", metric, limit, label, level="preprocessing")

    checks = [
        pod("modes_0", ">=", "lines_min_kept", 90, "Without removal: the lines and circles fully measured (% of their "
                                                    "brightness; the matched filter)"),
        pod("modes_0", ">=", "hough_found", 4, "Without removal: the Hough transform finds the 4 lines and circles"),
    ]
    checks += [
        pod("modes_1", "<=", "dimmed_worst_image", 5,
            "1 mode removed: in every image, particles dimmed below half their brightness (%)"),
        pod("modes_1", "<=", "clipped_pixels", 3,
            "1 mode removed: pixels clipped at 0 (%; the automatic offset after the removal)"),
        pod("modes_1", ">=", "particle_kept_median", 95,
            "1 mode removed: each particle keeps (median, % of its brightness; 89 without the offset)"),
    ]
    for n in (1, 2):
        sc = f"modes_{n}"
        what = f"{n} mode{'s' if n > 1 else ''} removed"
        checks += [
            pod(sc, "<=", "lines_kept", 1, f"{what}: lines and circles left (% of their brightness)"),
            pod(sc, "<=", "glare_kept", 1, f"{what}: glare spot left (% of its brightness)"),
            pod(sc, "<=", "hough_found", 0, f"{what}: the Hough transform finds none of the lines and circles"),
            pod(sc, ">=", "particles_kept", 90, f"{what}: particles kept (% of their brightness)"),
            pod(sc, ">=", "particle_kept_p05", 60, f"{what}: the 5 % dimmest particles keep at least (% of their "
                                                   "brightness, each one measured)"),
            pod(sc, "<=", "particles_dimmed", 3, f"{what}: particles dimmed below half their brightness (%)"),
        ]
    return checks


BACKGROUND_CHECKS = _background_checks()
