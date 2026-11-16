"""Post-processing of the PIV vector fields: statistics, vorticity, turbulence, statistical moments, gradients,
displacements, quality of the PIV (valid vectors, correlation peak ratio), peak locking and convergence of the
statistics, with their figures (from PostProcessing_General_PIV.py and the reports of PIV_PAIRS_batch.py); those of
a PTV (lagrangian.py), the images with their POD (imaging.py), and the jet potential core of the jet theory add-on
(jet/theory.py, with PostSettings.jet_theory). The
vector fields and the base of the analyses (scale space, reference quantities) are those of fields.py.

The vector fields are those saved by the PIV (vectors.npz, converted with the camera settings) or a folder of
DynamicStudio CSV exports: without the camera settings, the vector fields in pixels give the figures in the sensor
space (pixels and pairs). The recipe of the post-processing (PostSettings) gives the statistical outlier filtering,
the reference quantities, the peak locking and the jet theory; all the analyses are made. Their figures are
drawn in Matplotlib figures (PostProcessing.plot) and exported with the key numbers and the statistics
(PostProcessing.export: subfolder PostProcessing of the image folder in the interface).

    import velocihub as vh

    post = vh.PostProcessing(vh.load_vectors("images/PIV/vectors.npz", camera=camera), vh.PostSettings(outlier_k=4))
    print(post.metrics())
    post.export("images/PostProcessing")
"""

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from matplotlib.ticker import FuncFormatter, NullFormatter

from . import imaging, lagrangian, turbulence
from .fields import (EXPORT_DPI, LREF, TFLOW, UREF, VIDEO_FORMAT, VIDEO_FPS, FieldAnalysis, FieldSettings, map_indices,
                     nanstd1, num2str, quiet)
from .imaging import ImageAnalysis
from .jet.theory import (JET_FIGSIZE, JET_FIGURES, JET_FILES, JET_GROUP, JET_MAPS, JET_SYMBOLS, JetAnalysis,
                         JetSettings)
from .lagrangian import LagrangianAnalysis
from .pipelines import pipeline_of
from .project import export_name, relative

POSTPROCESSING_FOLDER = "PostProcessing"
# The export of each pipeline in its subfolder (PostProcessing/PaIRS, PostProcessing/TracTrac: pipelines.py); that of
# the results of a PTV of the previous versions (a default export folder still, now in PostProcessing/TracTrac)
POSTPROCESSING_PTV_FOLDER = "PostProcessing_PTV"
# The integral time scale: at least this many maps (its autocorrelation over the maps)
MIN_TIME_MAPS = 20
# Task in the names of the files exported (export_name)
TASK = "PostProcessing"
# Figures: key and name (in the menus)
FIGURES = [("velocity", "Instantaneous Velocity |u|"),
           ("u", "Instantaneous Velocity u"),
           ("v", "Instantaneous Velocity v"),
           ("mean_magnitude", "Mean Velocity |u|"),
           ("mean_velocity", "Mean Velocity u"),
           ("mean_v", "Mean of Velocity v"),
           ("vorticity", "Instantaneous Vorticity ω"),
           ("mean_vorticity", "Mean Vorticity ω"),
           ("mean_vorticity_abs", "Mean Vorticity ω (absolute value)"),
           ("vorticity_rms", "RMS of Vorticity ω"),
           ("u_rms", "RMS of Velocity u"),
           ("v_rms", "RMS of Velocity v"),
           ("turbulence_intensity", "Turbulence intensity Tu (Velocity RMS based)"),
           ("turbulence_intensity_map", "Turbulence intensity Tu (Velocity Fluctuation based)"),
           ("reynolds_stress", "Reynolds shear stress τ"),
           ("reynolds_stress_abs", "Reynolds shear stress |τ|"),
           ("tke", "Turbulent kinetic energy k"),
           ("integral_time", "Integral time scale T_u"),
           ("integral_length_x", "Integral length scale L_x (u along x)"),
           ("integral_length_y", "Integral length scale L_y (u along y)"),
           ("integral_length_vx", "Integral length scale L_x (v along x)"),
           ("integral_length_vy", "Integral length scale L_y (v along y)"),
           ("ke_time", "k-ε time scale k/ε"),
           ("ke_length", "k-ε length scale k^1.5/ε"),
           ("skewness_u", "Skewness of u"),
           ("kurtosis_u", "Kurtosis of u"),
           ("skewness_v", "Skewness of v"),
           ("kurtosis_v", "Kurtosis of v"),
           ("velocity_gradient", "Mean velocity gradient ∂u/∂y"),
           ("velocity_gradient_abs", "Mean velocity gradient |∂u/∂y|"),
           ("axial_gradient", "Mean velocity gradient ∂u/∂x"),
           ("axial_gradient_abs", "Mean velocity gradient |∂u/∂x|"),
           ("ti_gradient", "Turbulence intensity gradient ∂Tu/∂y"),
           ("ti_gradient_abs", "Turbulence intensity gradient |∂Tu/∂y|"),
           ("reynolds_gradient", "Reynolds shear stress gradient ∂τ/∂y"),
           ("reynolds_gradient_abs", "Reynolds shear stress gradient |∂τ/∂y|"),
           ("vorticity_gradient", "Mean vorticity gradient ∂ω/∂y"),
           ("vorticity_gradient_abs", "Mean vorticity gradient |∂ω/∂y|"),
           ("displacement", "Instantaneous displacement |d|"),
           ("displacement_x", "Instantaneous displacement x"),
           ("displacement_y", "Instantaneous displacement y"),
           ("mean_displacement", "Mean displacement |d|"),
           ("mean_displacement_x", "Mean displacement x"),
           ("mean_displacement_y", "Mean displacement y"),
           ("peak_locking", "Peak locking"),
           ("valid", "Valid vectors"),
           ("peak_ratio_map", "Instantaneous peak ratio"),
           ("peak_ratio", "Median peak ratio"),
           ("peak_ratio_histogram", "Peak ratio histogram"),
           ("convergence", "Convergence of the statistics")] + lagrangian.FIGURES + imaging.FIGURES + JET_FIGURES
FIGURE_TITLES = dict(FIGURES)
# The quality of a PIV (correlation): drawn for the results of a PIV only, not of a PTV (the valid cells of its binning
# aren't a reliable quality)
PIV_GROUP = ("PIV", ("valid", "peak_ratio_map", "peak_ratio", "peak_ratio_histogram", "peak_locking"))
PIV_FIGURES = set(PIV_GROUP[1])
NOT_PIV = "Only for the results of a PIV (correlation), not of a PTV"
# Groups of quantities of the menus of the figures (each figure in one group)
FIGURE_GROUPS = [("Velocity", ("velocity", "u", "v", "mean_magnitude", "mean_velocity", "mean_v", "u_rms", "v_rms",
                              "skewness_u", "kurtosis_u", "skewness_v", "kurtosis_v")),
                 ("Vorticity", ("vorticity", "mean_vorticity", "mean_vorticity_abs", "vorticity_rms")),
                 ("Turbulence", ("turbulence_intensity", "turbulence_intensity_map", "reynolds_stress",
                                 "reynolds_stress_abs", "tke", "integral_time", "integral_length_x",
                                 "integral_length_y", "integral_length_vx", "integral_length_vy", "ke_time",
                                 "ke_length")),
                 ("Gradients", ("velocity_gradient", "velocity_gradient_abs", "axial_gradient", "axial_gradient_abs",
                                "ti_gradient", "ti_gradient_abs",
                                "reynolds_gradient", "reynolds_gradient_abs", "vorticity_gradient",
                                "vorticity_gradient_abs")),
                 ("Displacement", ("displacement", "displacement_x", "displacement_y", "mean_displacement",
                                  "mean_displacement_x", "mean_displacement_y")),
                 PIV_GROUP,
                 lagrangian.GROUP,
                 *imaging.GROUPS,
                 JET_GROUP,
                 # At the root of the menu
                 (None, ("convergence",))]
assert sorted(k for _, keys in FIGURE_GROUPS for k in keys) == sorted(FIGURE_TITLES)
# Groups gathered in a submenu of the menus of the figures: {group: submenu}
MENU_PARENTS = {imaging.GROUPS[0][0]: "Pre-Processing", imaging.GROUPS[1][0]: "Pre-Processing",
                JET_GROUP[0]: "Add-ons"}
# Figures drawn as a single map (FieldAnalysis.map), which can be compared with a wiper (plot_comparison); the
# others are profiles, curves and histograms
MAP_FIGURES = tuple(key for key, _ in FIGURES if key not in ("peak_locking", "peak_ratio_histogram", "convergence",
                                                                 "pod_spectrum")
                    and (key not in lagrangian.FIGURE_FILES or key in lagrangian.MAPS)
                    and (key not in JET_FILES or key in JET_MAPS))
# Symbols (LaTeX) of the quantities of the maps, in their titles and labels, and naming the maps compared
FIGURE_SYMBOLS = {"velocity": r"|\mathbf{u}|", "u": "u", "v": "v", "mean_magnitude": r"\overline{|\mathbf{u}|}",
                  "mean_velocity": r"\overline{u}", "mean_v": r"\overline{v}", "vorticity": r"\omega",
                  "mean_vorticity": r"\overline{\omega}", "mean_vorticity_abs": r"|\overline{\omega}|",
                  "vorticity_rms": r"\sigma_\omega", "u_rms": r"\sigma_u", "v_rms": r"\sigma_v",
                  "turbulence_intensity": r"Tu_{\%}", "turbulence_intensity_map": r"Tu_{\%}", "reynolds_stress": r"\tau",
                  "reynolds_stress_abs": r"|\tau|", "tke": "k", "integral_time": "T_u", "integral_length_x": "L_x^u",
                  "integral_length_y": "L_y^u", "integral_length_vx": "L_x^v", "integral_length_vy": "L_y^v",
                  "ke_time": r"T_{k\varepsilon}", "ke_length": r"L_{k\varepsilon}", "skewness_u": r"\gamma_u", "kurtosis_u": r"\beta_u",
                  "skewness_v": r"\gamma_v", "kurtosis_v": r"\beta_v", "displacement": "|d|",
                  "displacement_x": "d_x", "displacement_y": "d_y", "mean_displacement": r"\overline{|d|}",
                  "mean_displacement_x": r"\overline{d_x}", "mean_displacement_y": r"\overline{d_y}",
                  "valid": r"\mathrm{valid}_{\%}", "peak_ratio_map": "S/N", "peak_ratio": r"\widetilde{S/N}",
                  "velocity_gradient": r"\partial\overline{u}/\partial y",
                  "velocity_gradient_abs": r"|\partial\overline{u}/\partial y|",
                  "axial_gradient": r"\partial\overline{u}/\partial x",
                  "axial_gradient_abs": r"|\partial\overline{u}/\partial x|",
                  "ti_gradient": r"\partial Tu_{\%}/\partial y", "ti_gradient_abs": r"|\partial Tu_{\%}/\partial y|",
                  "reynolds_gradient": r"\partial\tau/\partial y", "reynolds_gradient_abs": r"|\partial\tau/\partial y|",
                  "vorticity_gradient": r"\partial\overline{\omega}/\partial y",
                  "vorticity_gradient_abs": r"|\partial\overline{\omega}/\partial y|", **lagrangian.MAPS,
                  **imaging.MAPS, **JET_SYMBOLS}
assert set(MAP_FIGURES) == set(FIGURE_SYMBOLS)
# Figures of one map (the others are those of all the maps): the instantaneous fields, the turbulence intensity
# of a map (its fluctuation from the mean) and its vorticity
INSTANTANEOUS = {"velocity", "u", "v", "vorticity", "turbulence_intensity_map", "displacement", "displacement_x",
                 "displacement_y", "peak_ratio_map"} | lagrangian.INSTANTANEOUS | imaging.INSTANTANEOUS
# Element of the names of the figures exported ([project]_PostProcessing_[element](_[map]).png)
FIGURE_FILES = {"velocity": "InstantaneousVelocity", "u": "InstantaneousU", "v": "InstantaneousV",
                "mean_magnitude": "MeanVelocity", "mean_velocity": "MeanU", "mean_v": "MeanV",
                "vorticity": "InstantaneousVorticity", "mean_vorticity": "MeanVorticity",
                "mean_vorticity_abs": "MeanVorticityAbs", "vorticity_rms": "VorticityRms", "u_rms": "URms", "v_rms": "VRms",
                "turbulence_intensity": "TurbulenceIntensity",
                "turbulence_intensity_map": "InstantaneousTurbulenceIntensity", "reynolds_stress": "ReynoldsStress",
                "reynolds_stress_abs": "ReynoldsStressAbs", "tke": "TurbulentKineticEnergy",
                "integral_time": "IntegralTimeScale", "integral_length_x": "IntegralLengthScaleX",
                "integral_length_y": "IntegralLengthScaleY", "integral_length_vx": "IntegralLengthScaleVX",
                "integral_length_vy": "IntegralLengthScaleVY", "ke_time": "KEpsilonTimeScale",
                "ke_length": "KEpsilonLengthScale",
                "skewness_u": "SkewnessU", "kurtosis_u": "KurtosisU", "skewness_v": "SkewnessV",
                "kurtosis_v": "KurtosisV", "displacement": "InstantaneousDisplacement",
                "displacement_x": "InstantaneousDisplacementX", "displacement_y": "InstantaneousDisplacementY",
                "mean_displacement": "MeanDisplacement", "mean_displacement_x": "MeanDisplacementX",
                "mean_displacement_y": "MeanDisplacementY", "peak_locking": "PeakLocking", "valid": "ValidVectors",
                "peak_ratio_map": "InstantaneousPeakRatio", "peak_ratio": "PeakRatio",
                "peak_ratio_histogram": "PeakRatioHistogram", "convergence": "Convergence",
                "velocity_gradient": "MeanVelocityGradient", "velocity_gradient_abs": "MeanVelocityGradientAbs",
                "axial_gradient": "MeanVelocityAxialGradient", "axial_gradient_abs": "MeanVelocityAxialGradientAbs",
                "ti_gradient": "TurbulenceIntensityGradient", "ti_gradient_abs": "TurbulenceIntensityGradientAbs",
                "reynolds_gradient": "ReynoldsStressGradient", "reynolds_gradient_abs": "ReynoldsStressGradientAbs",
                "vorticity_gradient": "MeanVorticityGradient", "vorticity_gradient_abs": "MeanVorticityGradientAbs",
                **lagrangian.FIGURE_FILES, **imaging.FIGURE_FILES, **JET_FILES}
# Smoothing of the maps of the gradients by default (cells, Gaussian, the holes ignored: PostSettings.gradient_sigma).
# 1.5 before VelociHub 1.2: the benchmark (docs/benchmark.md) showed it never the best on its flows, even with the
# noise of a single map in the mean field
GRADIENT_SIGMA = 0.5
# Size of the figures exported that aren't maps (inches)
FIGSIZE = {"convergence": (10, 5), "peak_locking": (10, 7), "peak_ratio_histogram": (8, 4.5), **lagrangian.FIGSIZE,
           **imaging.FIGSIZE, **JET_FIGSIZE}
# Bins per pixel of the histograms of the peak locking (odd: each integer in the middle of a bin)
PEAK_LOCKING_BINS = 21
# Maps summed at once by the convergence (memory: about 15 arrays of this many maps)
CONVERGENCE_CHUNK = 50


def default_export_folder(maps):
    """Default export folder (relative to the measurement) of the vector fields of a folder: the subfolder of their
    pipeline in PostProcessing (PostProcessing/PaIRS, PostProcessing/TracTrac: the exports of the pipelines of a
    measurement side by side), PostProcessing itself for vector fields of no pipeline (DynamicStudio CSV exports)."""
    label = pipeline_of(maps)
    return os.path.join(POSTPROCESSING_FOLDER, label) if label else POSTPROCESSING_FOLDER


def export_folder_name(saved, maps):
    """Export folder (relative to the measurement) of the vector fields of a folder: the one saved, else the default
    one; the default ones of the previous versions (PostProcessing, PostProcessing_PTV) give the default one."""
    if not saved or os.path.normpath(saved) in (POSTPROCESSING_FOLDER, POSTPROCESSING_PTV_FOLDER):
        return default_export_folder(maps)
    return saved


def previous_results(folder, name, task=TASK, records=True):
    """The results of a previous export of a project in a folder, that an export would overwrite or leave beside
    its own (figures, maps or videos it no longer exports): the files and the folders of figures named
    [name]_[task]_..., and metrics.json and postprocessing.json (records). [] if there are none (or no folder)."""
    if not folder or not os.path.isdir(folder):
        return []
    prefix = export_name(name, task, "")
    return [os.path.join(folder, entry) for entry in sorted(os.listdir(folder))
            if entry.startswith(prefix) or records and entry in ("metrics.json", "postprocessing.json")
            and os.path.isfile(os.path.join(folder, entry))]


def count_results(paths):
    """The files of results (previous_results): those of their folders counted."""
    return sum(sum(os.path.isfile(os.path.join(p, e)) for e in os.listdir(p)) if os.path.isdir(p) else 1
               for p in paths if os.path.exists(p))


def remove_results(paths):
    """Remove the results of a previous export (previous_results): its files, and in its folders of figures the
    files named as them (any other file kept), each folder removed once empty. Returns the number of files
    removed."""
    removed = 0
    for path in paths:
        if os.path.isdir(path):
            base = os.path.basename(path)
            for entry in os.listdir(path):
                file = os.path.join(path, entry)
                if os.path.isfile(file) and entry.startswith(base):
                    os.remove(file)
                    removed += 1
            if not os.listdir(path):
                os.rmdir(path)
        elif os.path.isfile(path):
            os.remove(path)
            removed += 1
    return removed


@dataclass
class PostSettings(JetSettings, FieldSettings):
    """Recipe of the post-processing: the statistical outlier filtering and the reference quantities
    (FieldSettings); the data reconstruction: hold_maps, invalid cells of the instantaneous maps filled with the most
    recent valid value of at most this number of previous maps (sample and hold; 0: none); the peak locking:
    peak_locking_bins, bins per pixel of its histograms (odd: each integer in the middle of a bin); the jet theory
    add-on (jet.theory.JetSettings: jet_theory, core_mode, core_level, skew_zero, skew_change, skew_profiles,
    n_stations, first_offset, core_u_ref); pod_mode, pod_mode_last: the modes of the POD of the images exported, the
    first and the last ones (from 1; None: the first one alone; one PNG each), the interface showing one of them;
    pod_combine: those modes summed instead, the background of the image of each map (each mode weighted by its
    temporal coefficient, as the pre-processing subtracts it; imaging.py); the gradients (gradient_fields):
    gradient_sigma, the smoothing of the mean fields (cells of the grid, Gaussian, the holes ignored; 0: none), and
    gradient_smooth_after, the differences taken first (one-sided next to the masks and the edges) and smoothed
    after, instead of the smoothed fields differenced: the same away from the masks and the edges, closer to the
    gradient at a wall, and a value at the grid points next to it."""

    hold_maps: int = 0
    peak_locking_bins: int = PEAK_LOCKING_BINS
    pod_mode: int = 1
    pod_mode_last: int = None
    pod_combine: bool = False
    gradient_sigma: float = GRADIENT_SIGMA
    gradient_smooth_after: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.check_jet()
        self.hold_maps = self.hold_maps or 0
        self.integers(self, ("hold_maps", "peak_locking_bins", "pod_mode"))
        if self.pod_mode < 1:
            raise ValueError("Invalid post-processing settings: pod_mode (mode of the POD, from 1) >= 1")
        self.pod_combine = bool(self.pod_combine)
        self.gradient_sigma = float(self.gradient_sigma if self.gradient_sigma is not None else 0)
        if self.gradient_sigma < 0:
            raise ValueError("Invalid post-processing settings: gradient_sigma (smoothing of the gradients, cells) "
                             ">= 0")
        self.gradient_smooth_after = bool(self.gradient_smooth_after)
        if self.pod_mode_last is not None:
            if float(self.pod_mode_last) != int(self.pod_mode_last):
                raise ValueError("pod_mode_last must be an integer")
            self.pod_mode_last = int(self.pod_mode_last)
            if self.pod_mode_last < self.pod_mode:
                raise ValueError("Invalid post-processing settings: pod_mode_last (last mode of the POD exported) "
                                 ">= pod_mode")
        if self.hold_maps < 0:
            raise ValueError("Invalid post-processing settings: hold_maps (Sample&Hold) must be >= 0")
        if not (self.peak_locking_bins >= 3 and self.peak_locking_bins % 2 == 1):
            raise ValueError("Invalid post-processing settings: peak_locking_bins must be an odd integer >= 3")

    def steps(self):
        steps = super().steps()
        if self.hold_maps:
            steps.append(f"invalid cells of the instantaneous maps held (Sample&Hold, at most {self.hold_maps} "
                         "maps back)")
        if self.gradient_sigma != GRADIENT_SIGMA or self.gradient_smooth_after:
            steps.append(f"gradients smoothed over σ = {num2str(self.gradient_sigma)} cells"
                         + (" after the differences" if self.gradient_smooth_after else ""))
        return steps + self.jet_steps()


################################################################################
# Analyses


def gradient_nan(F, coords, axis):
    """dF/dcoords along axis of a field with holes (NaN), on a uniform grid: central differences where both
    neighbours have a value, else one-sided (second order if two points on that side, else first order); NaN in
    the holes and where neither neighbour has a value."""
    F = np.moveaxis(np.asarray(F, np.float64), axis, 0)
    h = float(coords[1] - coords[0])
    n = F.shape[0]
    pad = np.full((2,) + F.shape[1:], np.nan)
    G = np.concatenate([pad, F, pad])
    fm2, fm1, fp1, fp2 = G[0:n], G[1:n + 1], G[3:n + 3], G[4:n + 4]
    with quiet():
        options = ((fp1 - fm1) / (2 * h), (-3 * F + 4 * fp1 - fp2) / (2 * h), (3 * F - 4 * fm1 + fm2) / (2 * h),
                   (fp1 - F) / h, (F - fm1) / h)
    out = np.full(F.shape, np.nan)
    for option in reversed(options):
        out = np.where(np.isfinite(option), option, out)
    out[~np.isfinite(F)] = np.nan
    return np.moveaxis(out, 0, axis)


def smooth_nan(F, sigma):
    """Gaussian smoothing over sigma cells ignoring the holes (normalized convolution); the holes kept."""
    if not sigma:
        return np.array(F, np.float64)
    ok = np.isfinite(F)
    with quiet():
        out = (ndimage.gaussian_filter(np.where(ok, F, 0.0), sigma)
               / ndimage.gaussian_filter(ok.astype(np.float64), sigma))
    out[~ok] = np.nan
    return out


def peak_locking_check(d, bins_per_px=PEAK_LOCKING_BINS):
    """Peak locking of displacements d (px, their absolute values) (PostProcessing_General_PIV.py): histogram with
    bins_per_px bins per pixel, each integer in the middle of a bin, divided by its moving average over 1 px (the
    shape of the flow disappears) and folded onto one pixel. Flat at 1: no peak locking; above 1 at the integers and
    below at +-0.5 px: peak locking. Degree of peak locking C = 1 - min / max of the folded ratio (0: none, 1: all the
    vectors on integers). The histogram returned (centres, counts, smooth) starts at 0: the bars below, empty, are
    left out. None if there are too few displacements."""
    d = d[np.isfinite(d)]
    if d.size < 100:
        return None
    lo, hi = np.floor(np.percentile(d, 0.5)), np.ceil(np.percentile(d, 99.5))
    n_pixels = int(hi - lo) + 1
    edges = np.linspace(lo - 0.5, hi + 0.5, n_pixels * bins_per_px + 1)
    counts, edges = np.histogram(d, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    smooth = np.convolve(counts, np.ones(bins_per_px) / bins_per_px, mode="same")
    # Not the ends (incomplete moving average) nor the almost empty bars
    use = smooth > 0.01 * smooth.max()
    use[:bins_per_px] = False
    use[-bins_per_px:] = False
    ratio = np.full(len(counts), np.nan)
    ratio[use] = counts[use] / smooth[use]
    position = np.arange(len(counts)) % bins_per_px
    with quiet():
        folded = np.array([np.nanmean(ratio[(position == k) & use]) if np.any((position == k) & use) else np.nan
                           for k in range(bins_per_px)])
        folded = folded / np.nanmean(folded)
    C = float(1 - np.nanmin(folded) / np.nanmax(folded)) if np.isfinite(folded).any() else float("nan")
    # The absolute values: no bin below 0 (the first pixel isn't in the folded ratio anyway), and the smooth copy
    # only where its moving average is complete (not over the empty bars below 0, nor beyond the last one)
    shown = edges[:-1] >= 0
    smooth = smooth.astype(float)
    smooth[:bins_per_px] = smooth[-bins_per_px:] = np.nan
    return dict(centres=centres[shown], counts=counts[shown], smooth=smooth[shown], C=C, n=int(d.size))


def sci2(x, pos=None):
    """A number in scientific notation with 2 significant digits (LaTeX): 1.2 x 10^4 (a tick formatter)."""
    if not x:
        return "0"
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


class PostProcessing(JetAnalysis, ImageAnalysis, LagrangianAnalysis, FieldAnalysis):
    """Post-processing of vector fields with a recipe (PostSettings): the analyses are evaluated when they are
    needed and kept until the settings they depend on change; those of a PTV with LagrangianAnalysis, the images
    and their POD with ImageAnalysis, the jet theory (add-on, jet_theory on) with JetAnalysis."""

    SETTINGS = PostSettings
    TASK = TASK
    FIGURE_TITLES = FIGURE_TITLES
    FIGURE_SYMBOLS = FIGURE_SYMBOLS
    FIGURE_GROUPS = FIGURE_GROUPS
    INSTANTANEOUS = INSTANTANEOUS
    FIGURE_FILES = FIGURE_FILES
    FIGSIZE = FIGSIZE
    USES_U_REF = {"turbulence_intensity_map"}
    # Times L_ref over u_ref (u_ref² for the stress), in 1/s (m/s²) without the reference length
    PER_LENGTH = {"vorticity", "mean_vorticity", "mean_vorticity_abs", "vorticity_rms", "velocity_gradient",
                  "velocity_gradient_abs", "axial_gradient", "axial_gradient_abs", "reynolds_gradient",
                  "reynolds_gradient_abs", "vorticity_gradient", "vorticity_gradient_abs"}
    # Times L_ref over u_ref² (acceleration_factor)
    ACCELERATIONS = {"mean_acceleration", "mean_acceleration_x", "mean_acceleration_y"}
    SCALED = {"velocity", "u", "v", "mean_magnitude", "mean_velocity", "mean_v", "u_rms", "v_rms", "reynolds_stress",
              "reynolds_stress_abs", "tke", "ptv_tracks"} | PER_LENGTH | ACCELERATIONS
    # The modes of the POD don't change the instantaneous maps
    DISPLAY_SETTINGS = ("pod_mode", "pod_mode_last", "pod_combine")

    def scaled(self, key):
        return super().scaled(key) and (bool(self.settings.l_ref) or key not in self.PER_LENGTH | self.ACCELERATIONS)

    def piv_missing(self, key):
        """Why a figure of the quality of a PIV can't be drawn: the vector fields are those of a PTV (None if it
        can)."""
        return NOT_PIV if key in PIV_FIGURES and self.ptv() else None

    def draw(self, key, fig, index=0, name=None, clim=None, cmap=None):
        """The figures of the quality of a PIV say why they aren't drawn for a PTV, those of the jet theory with the
        add-on off (nothing computed; FieldAnalysis.draw)."""
        reason = self.piv_missing(key) or self.jet_off(key)
        if reason:
            self.map_index = int(np.clip(index, 0, self.fields.n - 1))
            self.message(fig, FIGURE_TITLES[key], reason)
            return None
        return super().draw(key, fig, index, name, clim, cmap)

    def exported(self, key):
        """The figures of the PTV are exported for a PTV only (with its tracks, and the PIV to compare with), those
        of the quality of a PIV for a PIV only, those of the images and of their POD when they are found, those of the
        jet theory with it on."""
        if key in PIV_FIGURES:
            return self.piv_missing(key) is None
        if key in imaging.FIGURE_FILES:
            return self.imaging_missing(key) is None
        if key in JET_FILES:
            return self.jet_off(key) is None
        return key not in lagrangian.FIGURE_FILES or self.ptv_missing(key) is None

    def instantaneous(self, key):
        """The figures of the POD are the same for all the maps (also with rolling statistics), but the sum of the
        modes: the background of the image of each map."""
        if key == "pod_mode" and self.pod_combined():
            return True
        return key not in imaging.STATIC and super().instantaneous(key)

    def export_element(self, key, compares=None):
        """The file of the POD mode named with its number (PODMode3)."""
        other = ((compares or {}).get(key) or (None,))[0]
        return "-".join(self.element(k) if k in imaging.FIGURE_FILES else self.FIGURE_FILES[k]
                        for k in (key, other) if k is not None)

    def hold(self, F, k):
        """Map k of the maps F (n, rows, columns) with its invalid cells (NaN) of the flow filled by sample and hold:
        the value of the most recent of the hold_maps previous maps where the cell is valid; left invalid without
        one (and outside the flow: masked, no particles)."""
        n = self.settings.hold_maps
        F_k = F[k]
        if not n:
            return F_k
        F_k = np.array(F_k, copy=True)
        missing = ~np.isfinite(F_k) & (self.fields.status == 0)
        for j in range(k - 1, max(k - n, 0) - 1, -1):
            if not missing.any():
                break
            take = missing & np.isfinite(F[j])
            F_k[take] = F[j][take]
            missing &= ~take
        return F_k

    def hold_delay(self):
        """Sample and hold of the invalid cells of the flow over all the maps (hold): the mean delay of those filled
        (maps since their value), the fraction of them filled and their number; None without sample and hold."""
        s = self.settings
        if not s.hold_maps:
            return None

        def compute():
            f = self.fields
            flow = f.status == 0
            last = np.full(f.shape, -np.inf)  # Last map where each cell is valid
            total, filled, invalid = 0.0, 0, 0
            for k in range(f.n):
                valid = np.isfinite(f.U[k])
                delay = k - last[~valid & flow]
                held = delay <= s.hold_maps
                total, filled, invalid = total + delay[held].sum(), filled + int(held.sum()), invalid + delay.size
                last[valid] = k
            return dict(mean_delay=total / filled if filled else None, filled=filled / invalid if invalid else None,
                        invalid=invalid)
        return self.cached("hold_delay", s.hold_maps, compute)

    def turbulence_intensity_map(self, index):
        """Turbulence intensity of the map index: 100 sqrt((u'² + v'²) / 2) / U_ref, its fluctuation u' = U - mean
        U, v' = V - mean V (the outliers removed; with rolling statistics, the mean of the window ending at the map)
        at the reliable grid points, its invalid cells held (sample and hold, within the window). Without sample and
        hold, the sum of its squares over the N maps of the statistics is (N - 1) times the square of the turbulence
        intensity of the statistics (sample variances), cell by cell; the cells held add values that the statistics
        don't have."""
        k = int(np.clip(index, 0, self.fields.n - 1))
        drawn, self.map_index = self.map_index, k
        try:
            st = self.statistics()
            a = self.window()[0]
            U, V, _ = self.clean()
        finally:
            self.map_index = drawn
        U_k, V_k = self.hold(U, k - a), self.hold(V, k - a)
        return 100 * np.sqrt(((U_k - st["mean_U"]) ** 2 + (V_k - st["mean_V"]) ** 2) / 2) / abs(st["u_ref"])

    def vorticity_of(self, U, V):
        """Vorticity omega = dV/dx - dU/dy of velocity fields (..., row, column), by central differences on the grid
        (one-sided at its edges; 1/s, 1/frame for vector fields in pixels)."""
        f = self.fields
        mm = 1.0 if f.settings.get("units") == "px" else 1e-3
        x, y = f.x[0].astype(np.float64) * mm, f.y[:, 0].astype(np.float64) * mm
        with quiet():
            return (np.gradient(np.asarray(V, np.float64), x, axis=-1)
                    - np.gradient(np.asarray(U, np.float64), y, axis=-2))

    def vorticity_factor(self, symbol=r"\omega"):
        """Factor and label (LaTeX, of the quantity symbol) of the vorticity in the figures: omega L_ref / u_ref with
        the reference length and velocity (physical space), else in 1/s (1/pair in the sensor space)."""
        L, u = self.settings.l_ref, self.scaling_velocity()
        if L and u:
            return L * 1e-3 / u, rf"${symbol}\,{LREF}\ /\ {UREF}$"
        return 1.0, rf"${symbol}$ (1/s)"

    def vorticity_statistics(self):
        """Vorticity of the mean field and rms of the vorticity of the maps (the outliers removed), at the reliable
        grid points (1/s)."""
        s = self.settings

        def compute():
            st = self.statistics()
            U, V, _ = self.clean()
            mean = self.vorticity_of(st["mean_U"], st["mean_V"])
            rms = nanstd1(self.vorticity_of(U, V))
            for F in (mean, rms):
                F[~st["reliable"]] = np.nan
            return dict(mean=mean, rms=rms)
        return self.windowed("vorticity", (s.outlier_k), compute)

    def higher_order(self):
        """Skewness m3 / m2^1.5 and kurtosis m4 / m2² (3 for a Gaussian) of u and v at the reliable grid
        points."""
        s = self.settings

        def compute():
            U, V, _ = self.clean()
            st = self.statistics()
            out = {}
            with quiet():
                # The means of the statistics (NaN outside the reliable grid points, left out anyway); products, as
                # ** is much slower
                for name, A, mean in (("u", U, st["mean_U"]), ("v", V, st["mean_V"])):
                    d = A - mean
                    d2 = d * d
                    m2 = np.nanmean(d2, 0)
                    out[f"skewness_{name}"] = np.where(st["reliable"], np.nanmean(d2 * d, 0) / m2 ** 1.5, np.nan)
                    out[f"kurtosis_{name}"] = np.where(st["reliable"], np.nanmean(d2 * d2, 0) / m2 ** 2, np.nan)
            return out
        return self.windowed("higher_order", (s.outlier_k), compute)

    def gradient_fields(self):
        """The mean velocity u, the turbulence intensity and the Reynolds shear stress tau = -<u'v'> smoothed over
        gradient_sigma cells (holes ignored), and their transverse gradients d/dy [row, column]: du (1/s; 1/pair in the
        sensor space), dti (%/mm; %/px) and dtau (m/s²; px/pair² in the sensor space); and the axial gradient of the
        mean velocity d/dx: dudx (1/s; 1/pair); the mean vorticity smoothed the same way, w (1/s; 1/pair), and its
        transverse gradient dw (1/(m s); 1/(px pair)). The gradients: of the smoothed fields (central differences,
        none next to a hole), or with gradient_smooth_after the differences of the fields (gradient_nan: one-sided
        next to the holes and the edges) smoothed."""
        s = self.settings
        sigma, after = s.gradient_sigma, s.gradient_smooth_after

        def compute():
            st, f = self.statistics(), self.fields
            mm = 1.0 if f.settings.get("units") == "px" else 1e-3
            x, y = f.x[0].astype(np.float64), f.y[:, 0].astype(np.float64)

            def derivative(F, smoothed, coords, axis):
                if after:
                    return smooth_nan(gradient_nan(F, coords, axis), sigma)
                with quiet():
                    return np.gradient(smoothed, coords, axis=axis)

            out = {}
            fields = {"u": st["mean_U"], "ti": st["ti"], "tau": -st["uv"], "w": self.vorticity_statistics()["mean"]}
            for name, F in fields.items():
                out[name] = smooth_nan(F, sigma)
            for name, scale in (("u", mm), ("ti", 1.0), ("tau", mm), ("w", mm)):
                out["d" + name] = derivative(fields[name], out[name], y * scale, 0)
            out["dudx"] = derivative(fields["u"], out["u"], x * mm, 1)
            return out
        return self.windowed("gradient_fields", (s.outlier_k, s.use_u_ref, s.u_ref, sigma, after), compute)

    def smoothed_mean_u(self, sigma):
        """The mean velocity u smoothed over sigma cells (holes ignored)."""
        s = self.settings
        return self.windowed("smoothed_mean_u", (s.outlier_k, sigma),
                             lambda: smooth_nan(self.statistics()["mean_U"], sigma))

    def convergence(self):
        """Residuals between the statistics of U with N and N - 1 maps (N = 2, 3...: PostProcessing_General_PIV.py
        with a step of 1 map), RMS over the field, plotted against the averaging time in flow times: of the mean
        and standard deviation fields in % of U_ref (absolute) and of the previous field (relative), of the
        skewness and kurtosis fields without unit (absolute) and in % (relative). converged: first N from which
        the residual of the mean stays below 1 % of U_ref (None if never). The running sums of the powers of U are
        evaluated by chunks of maps (CONVERGENCE_CHUNK), so that every N costs little."""
        s = self.settings

        def compute():
            U = self.clean()[0]
            n, shape = len(U), self.fields.shape
            u_ref = abs(self.statistics()["u_ref"])
            names = ("mean", "std", "skew", "kurt")
            res = {f"{name}_{kind}": [] for kind in ("abs", "rel") for name in names}

            def rms(A):
                return np.sqrt(np.nanmean(A * A, axis=(1, 2)))

            with quiet():
                # Moments about the mean field: the same central moments, without the cancellation of a large mean
                centre = np.nan_to_num(np.nanmean(U, 0, dtype=np.float64))
                carry, previous = np.zeros((5,) + shape), None
                for first in range(0, n, CONVERGENCE_CHUNK):
                    d = U[first:first + CONVERGENCE_CHUNK].astype(np.float64) - centre
                    finite = np.isfinite(d)
                    d = np.where(finite, d, 0.0)
                    d2 = d * d
                    # Sums of the maps 1 to N, for every N of the chunk (products: ** is much slower)
                    sums = np.cumsum(np.stack([finite, d, d2, d2 * d, d2 * d2]), axis=1) + carry[:, None]
                    carry = sums[:, -1]
                    count = sums[0]
                    e1, e2, e3, e4 = (sums[k] / count for k in range(1, 5))
                    e1sq = e1 * e1
                    m2 = e2 - e1sq
                    m3 = e3 - e1 * (3 * e2 - 2 * e1sq)
                    m4 = e4 - e1 * (4 * e3 - e1 * (6 * e2 - 3 * e1sq))
                    # A single map has no spread (exactly, whatever the rounding); the skewness needs 3 maps and the
                    # kurtosis 4
                    var = np.where(count >= 2, np.maximum(m2, 0), 0.0)
                    std = np.sqrt(var)
                    stats = [e1 + centre, std, np.where(count >= 3, m3 / (var * std), np.nan),
                             np.where(count >= 4, m4 / (var * var), np.nan)]
                    if previous is not None:
                        stats = [np.concatenate([p[None], F]) for p, F in zip(previous, stats)]
                    previous = [F[-1] for F in stats]
                    if len(stats[0]) < 2:
                        continue
                    for name, F in zip(names, stats):
                        diff, ref = rms(F[1:] - F[:-1]), rms(F[:-1])
                        # % of the previous field, NaN if it is 0 (the standard deviation of a single map)
                        res[f"{name}_rel"].append(np.where(ref > 0, 100 * diff / ref, np.nan))
                        res[f"{name}_abs"].append(100 * diff / u_ref if name in ("mean", "std") else diff)
            res = {k: np.concatenate(v) if v else np.zeros(0) for k, v in res.items()}
            N = np.arange(2, n + 1)
            # The first N after the last residual not below 1 % (NaN included)
            above = np.flatnonzero(~(res["mean_abs"] < 1.0))
            first = above[-1] + 1 if above.size else 0
            converged = int(N[first]) if first < len(N) else None
            return dict(N=N, converged=converged, u_ref=u_ref, **res)
        return self.windowed("convergence", (s.outlier_k, s.use_u_ref, s.u_ref), compute)

    def mean_displacements(self):
        """Mean of the displacements d_x, d_y (px) and of their norm |d| over the pairs (the outliers removed), at
        the reliable grid points; None if the displacements in pixels are unknown."""
        s = self.settings

        def compute():
            f = self.fields
            if f.u_px is None:
                return None
            a, b = self.window()
            kept = np.isfinite(self.clean()[0])
            du, dv = np.where(kept, f.u_px[a:b], np.nan), np.where(kept, f.v_px[a:b], np.nan)
            with quiet():
                out = {"x": np.nanmean(du, 0, dtype=np.float64), "y": np.nanmean(dv, 0, dtype=np.float64),
                       "norm": np.nanmean(np.hypot(du, dv), 0, dtype=np.float64)}
            for F in out.values():
                F[~self.statistics()["reliable"]] = np.nan
            return out
        return self.windowed("mean_displacements", (s.outlier_k), compute)

    def peak_locking(self):
        """Peak locking of the displacements d_x and d_y (px) of all the valid vectors (the outliers removed), from
        their absolute values (peak_locking_check). None if the displacements in pixels are unknown."""
        s = self.settings

        def compute():
            f = self.fields
            if f.u_px is None:
                return None
            a, b = self.window()
            keep = np.isfinite(self.clean()[0]) & np.isfinite(f.u_px[a:b])
            return {c: peak_locking_check(np.abs(d[a:b][keep]), s.peak_locking_bins)
                    for c, d in (("u", f.u_px), ("v", f.v_px))}
        return self.windowed("peak_locking", (s.outlier_k, s.peak_locking_bins), compute)

    def metrics(self):
        """Key numbers of the measurement and of the analyses (of all the maps, whatever the rolling statistics)."""
        with self.all_maps():
            return self.all_metrics()

    def all_metrics(self):
        f = self.fields
        st = self.statistics()
        flow = f.status == 0
        m = dict(name=f.name, maps=f.n, grid=[f.shape[1], f.shape[0]], dt_s=f.dt, scale_mm_per_px=f.scale,
                 outliers_removed=st["outliers"], u_ref_ms=st["u_ref"], fov_length_mm=self.fov_length(),
                 flow_time_s=self.flow_time(), duration_flow_times=float(self.flow_times(f.n)),
                 max_mean_U_ms=float(np.nanmax(st["mean_U"])) if np.isfinite(st["mean_U"]).any() else None,
                 median_peak_ratio=(float(np.nanmedian(st["peak_ratio"])) if np.isfinite(st["peak_ratio"]).any()
                                    else None))
        if not self.ptv():
            # The valid vectors of a PIV (not the valid cells of a PTV: not a reliable quality)
            m["valid_percent"] = float(100 * f.piv_valid[:, flow].mean()) if flow.any() else 0.0
        if not self.sensor():
            # The integral scales: their medians over the grid points where they are defined
            def median(F):
                return float(np.nanmedian(F)) if F is not None and np.isfinite(F).any() else None
            T_ke, L_ke = self.ke_scales()
            m.update(median_integral_time_s=median(self.integral_time()[0]),
                     median_integral_length_x_mm=median(self.integral_length("x")),
                     median_integral_length_y_mm=median(self.integral_length("y")),
                     median_integral_length_vx_mm=median(self.integral_length("x", "v")),
                     median_integral_length_vy_mm=median(self.integral_length("y", "v")),
                     median_dissipation_m2_s3=median(self.dissipation()),
                     median_ke_time_s=median(T_ke), median_ke_length_mm=median(None if L_ke is None else L_ke * 1000))
        m.update(self.ptv_metrics())
        m["maps_to_converge_1pct"] = self.convergence()["converged"]
        pl = self.peak_locking()
        for c in ("u", "v"):
            r = None if pl is None else pl[c]
            m[f"peak_locking_C_{c}"] = None if r is None else r["C"]
        m.update(self.jet_metrics())
        return m

    ############################################################################
    # Figures

    def field_limits(self, key, sub):
        f = self.fields
        if key in ("velocity", "ptv_tracks"):  # The tracks of the PTV with the colours of the velocity
            return 0, self.limits(self.velocity(np.hypot(f.U[sub], f.V[sub]), "")[0])[1]
        if key == "u":
            return self.limits(self.velocity(f.U[sub], "")[0])
        if key == "v":
            return self.limits(self.velocity(f.V[sub], "")[0], symmetric=True)
        if key == "displacement_x":
            return self.limits(f.u_px[sub])
        if key == "displacement_y":
            return self.limits(f.v_px[sub], symmetric=True)
        if key == "displacement":
            return 0, self.limits(np.hypot(f.u_px[sub], f.v_px[sub]))[1]
        if key == "peak_ratio_map":
            return 1, self.limits(f.peak_ratio[sub])[1]
        if key == "turbulence_intensity_map":
            return 0, self.limits(np.stack([self.turbulence_intensity_map(k) for k in range(f.n)[sub]]))[1]
        if key == "vorticity":
            return self.limits(self.vorticity_of(f.U[sub], f.V[sub]) * self.vorticity_factor()[0], symmetric=True)
        raise KeyError(key)

    def map_available(self, key):
        """The figure key is a map that the vector fields and the settings can draw (the maps a map can be compared
        with)."""
        f = self.fields
        if key not in MAP_FIGURES or ("displacement" in key and f.u_px is None
                                      or key.startswith("peak_ratio") and f.peak_ratio is None) or self.piv_missing(key):
            return False
        if key in lagrangian.MAPS:
            return self.ptv_missing(key) is None
        if key in imaging.MAPS:
            return self.imaging_missing(key) is None
        if key in JET_MAPS:
            return self.jet_off(key) is None and self.potential_core()["error"] is None
        return True

    def missing(self, key):
        f = self.fields
        if self.jet_off(key):
            return self.jet_off(key)
        if self.piv_missing(key):
            return self.piv_missing(key)
        if key.startswith("displacement") and f.u_px is None:
            return "The displacements in pixels are unknown"
        if key == "peak_ratio_map" and f.peak_ratio is None:
            return "The correlation peak ratio is unknown"
        return None

    def draw_map(self, fig, key, k, prefix, when, lim):
        f = self.fields
        # The invalid cells held (sample and hold), but in the quality maps
        U, V = self.hold(f.U, k), self.hold(f.V, k)
        if key == "velocity":
            F, label = self.velocity(np.hypot(U, V), r"|\mathbf{u}|")
            self.map(fig, F, rf"{prefix}Instantaneous velocity $|\mathbf{{u}}|${when}", label, "viridis", *lim,
                     quiver=(U, V))
        elif key == "u":
            F, label = self.velocity(U, "u")
            self.map(fig, F, f"{prefix}Instantaneous $u${when}", label, "viridis", *lim)
        elif key == "v":
            F, label = self.velocity(V, "v")
            self.map(fig, F, f"{prefix}Instantaneous $v${when}", label, "RdBu_r", *lim)
        elif key == "turbulence_intensity_map":
            self.map(fig, self.turbulence_intensity_map(k), rf"{prefix}Turbulence intensity $Tu_{{\%}}${when}",
                     r"$Tu_{\%} = 100\,\sqrt{({u'}^2 + {v'}^2) / 2}\ /\ u_\mathrm{ref}$", "magma", *lim)
        elif key == "vorticity":
            factor, label = self.vorticity_factor()
            self.map(fig, self.vorticity_of(U, V) * factor, rf"{prefix}Instantaneous vorticity $\omega${when}",
                     label, "RdBu_r", *lim)
        elif key == "displacement":
            du, dv = self.hold(f.u_px, k), self.hold(f.v_px, k)
            self.map(fig, np.hypot(du, dv), rf"{prefix}Instantaneous displacement $|d|${when}",
                     r"$|d|$ (px)", "viridis", *lim, quiver=(du, dv))
        elif key == "displacement_x":
            self.map(fig, self.hold(f.u_px, k), f"{prefix}Instantaneous displacement $d_x${when}", "$d_x$ (px)",
                     "viridis", *lim)
        elif key == "displacement_y":
            self.map(fig, self.hold(f.v_px, k), f"{prefix}Instantaneous displacement $d_y${when}", "$d_y$ (px)",
                     "RdBu_r", *lim)
        elif key == "peak_ratio_map":
            self.map(fig, f.peak_ratio[k], f"{prefix}Peak ratio $S/N${when}", "Peak ratio $S/N$",
                     "viridis", *lim)

    def mean_magnitude(self):
        """Mean of the norm of the velocity over the maps (the outliers removed), at the reliable grid points: not
        the norm of the mean velocity, which the fluctuations don't add to."""
        s = self.settings

        def compute():
            U, V, _ = self.clean()
            with quiet():
                F = np.nanmean(np.hypot(U, V), 0, dtype=np.float64)
            F[~self.statistics()["reliable"]] = np.nan
            return F
        return self.windowed("mean_magnitude", (s.outlier_k), compute)

    def draw_mean_magnitude(self, fig, prefix):
        # With the direction of the mean velocity
        st = self.statistics()
        F, label = self.velocity(self.mean_magnitude(), r"\overline{|\mathbf{u}|}")
        self.stat_map(fig, prefix, F, r"Mean velocity $\overline{|\mathbf{u}|}$", label, low=0,
                      quiver=(st["mean_U"], st["mean_V"]), key="mean_magnitude")

    def draw_mean_velocity(self, fig, prefix):
        st = self.statistics()
        F, label = self.velocity(st["mean_U"], r"\overline{u}")
        self.stat_map(fig, prefix, F, r"Mean velocity $\overline{u}$", label, low=0,
                      quiver=(st["mean_U"], st["mean_V"]), key="mean_velocity")

    def draw_mean_v(self, fig, prefix):
        F, label = self.velocity(self.statistics()["mean_V"], r"\overline{v}")
        self.stat_map(fig, prefix, F, r"Mean velocity $\overline{v}$", label, "RdBu_r", True, key="mean_v")

    def draw_u_rms(self, fig, prefix):
        F, label = self.velocity(self.statistics()["u_rms"], r"\sigma_u")
        self.stat_map(fig, prefix, F, r"Standard deviation $\sigma_u$ (axial)", label, "magma", low=0,
                      key="u_rms")

    def draw_v_rms(self, fig, prefix):
        F, label = self.velocity(self.statistics()["v_rms"], r"\sigma_v")
        self.stat_map(fig, prefix, F, r"Standard deviation $\sigma_v$ (transverse)", label, "magma",
                      low=0, key="v_rms")

    def draw_turbulence_intensity(self, fig, prefix):
        st = self.statistics()
        self.stat_map(fig, prefix, st["ti"], r"Turbulence intensity $Tu_{\%}$",
                      r"$Tu_{\%} = 100\,\sqrt{(\sigma_u^2 + \sigma_v^2) / 2}\ /\ u_\mathrm{ref}$", "magma", low=0,
                      u_ref=True)

    def smoothing_text(self):
        """The smoothing of the gradient maps, in their titles."""
        s = self.settings
        if not s.gradient_sigma:
            return ""
        return f", σ = {num2str(s.gradient_sigma)} cells" + (" after the differences" if s.gradient_smooth_after
                                                              else "")

    def draw_velocity_gradient(self, fig, prefix, absolute=False, axis="y"):
        """Map of the gradient of the mean velocity along y (transverse) or x (axial), or of its absolute value."""
        symbol = rf"\partial\overline{{u}}/\partial {axis}"
        symbol = f"|{symbol}|" if absolute else symbol
        factor, label = self.vorticity_factor(symbol)
        F = self.gradient_fields()["du" if axis == "y" else "dudx"] * factor
        key = ("velocity_gradient" if axis == "y" else "axial_gradient") + ("_abs" if absolute else "")
        self.stat_map(fig, prefix, np.abs(F) if absolute else F,
                      rf"Mean velocity gradient ${symbol}${self.smoothing_text()}", label,
                      "magma" if absolute else "RdBu_r", not absolute, low=0 if absolute else None, key=key)

    def draw_velocity_gradient_abs(self, fig, prefix):
        self.draw_velocity_gradient(fig, prefix, absolute=True)

    def draw_axial_gradient(self, fig, prefix):
        self.draw_velocity_gradient(fig, prefix, axis="x")

    def draw_axial_gradient_abs(self, fig, prefix):
        self.draw_velocity_gradient(fig, prefix, absolute=True, axis="x")

    def draw_ti_gradient(self, fig, prefix, absolute=False):
        symbol = r"|\partial Tu_{\%}/\partial y|" if absolute else r"\partial Tu_{\%}/\partial y"
        F = self.gradient_fields()["dti"]
        L = self.settings.l_ref
        if L and not self.sensor():
            F, label = F * L, rf"${symbol}\ {LREF}$ (%)"
        else:
            F, label = F, f"${symbol}$ (%/{'px' if self.sensor() else 'mm'})"
        self.stat_map(fig, prefix, np.abs(F) if absolute else F,
                      rf"Turbulence intensity gradient ${symbol}${self.smoothing_text()}", label,
                      "magma" if absolute else "RdBu_r", not absolute, low=0 if absolute else None, u_ref=True)

    def draw_ti_gradient_abs(self, fig, prefix):
        self.draw_ti_gradient(fig, prefix, absolute=True)

    def draw_reynolds_gradient(self, fig, prefix, absolute=False):
        symbol = r"|\partial\tau/\partial y|" if absolute else r"\partial\tau/\partial y"
        F = self.gradient_fields()["dtau"]
        L, u = self.settings.l_ref, self.scaling_velocity()
        if L and u:
            F, label = F * L * 1e-3 / u ** 2, rf"${symbol}\,{LREF}\ /\ {UREF}^2$"
        else:
            label = f"${symbol}$ ({'px/pair²' if self.sensor() else 'm/s²'})"
        self.stat_map(fig, prefix, np.abs(F) if absolute else F,
                      rf"Reynolds shear stress gradient ${symbol}${self.smoothing_text()}", label,
                      "magma" if absolute else "RdBu_r", not absolute, low=0 if absolute else None,
                      key="reynolds_gradient_abs" if absolute else "reynolds_gradient")

    def draw_reynolds_gradient_abs(self, fig, prefix):
        self.draw_reynolds_gradient(fig, prefix, absolute=True)

    def draw_vorticity_gradient(self, fig, prefix, absolute=False):
        symbol = r"\partial\overline{\omega}/\partial y"
        symbol = f"|{symbol}|" if absolute else symbol
        F = self.gradient_fields()["dw"]
        L, u = self.settings.l_ref, self.scaling_velocity()
        if L and u:
            F, label = F * (L * 1e-3) ** 2 / u, rf"${symbol}\,{LREF}^2\ /\ {UREF}$"
        else:
            label = f"${symbol}$ ({'1/(px pair)' if self.sensor() else '1/(m s)'})"
        key = "vorticity_gradient_abs" if absolute else "vorticity_gradient"
        self.stat_map(fig, prefix, np.abs(F) if absolute else F,
                      rf"Mean vorticity gradient ${symbol}${self.smoothing_text()}", label,
                      "magma" if absolute else "RdBu_r", not absolute, low=0 if absolute else None, key=key)

    def draw_vorticity_gradient_abs(self, fig, prefix):
        self.draw_vorticity_gradient(fig, prefix, absolute=True)

    def draw_reynolds_stress(self, fig, prefix):
        st = self.statistics()
        F, label = self.velocity(-st["uv"], r"\tau", power=2)
        self.stat_map(fig, prefix, F, r"Reynolds shear stress $\tau = -\overline{u'v'}$", label, "RdBu_r", True,
                      key="reynolds_stress")

    def draw_reynolds_stress_abs(self, fig, prefix):
        st = self.statistics()
        F, label = self.velocity(np.abs(st["uv"]), r"|\tau|", power=2)
        self.stat_map(fig, prefix, F, r"Reynolds shear stress $|\tau| = |\overline{u'v'}|$", label, "magma", low=0,
                      key="reynolds_stress_abs")

    def draw_tke(self, fig, prefix):
        st = self.statistics()
        F, label = self.velocity((st["u_rms"] ** 2 + st["v_rms"] ** 2) / 2, "k", power=2)
        self.stat_map(fig, prefix, F, r"Turbulent kinetic energy $k = (\sigma_u^2 + \sigma_v^2)\,/\,2$ (2 components)",
                      label, "magma", low=0, key="tke")

    def integral_time(self):
        """The integral time scale of u at each grid point (s; frames in the sensor space; turbulence.integral_time:
        the temporal autocorrelation of u' over the successive maps of the window, integrated to its first zero
        crossing) and why it can't be computed (None: it can): too few maps, or maps not evenly spaced in time."""
        def compute():
            f = self.fields
            a, b = self.window()
            if b - a < MIN_TIME_MAPS:
                return None, f"At least {MIN_TIME_MAPS} maps are needed (the autocorrelation over the maps)"
            d = np.diff(np.asarray(f.t[a:b], np.float64))
            if not d.size or d.min() <= 0 or d.max() - d.min() > 0.01 * np.median(d):
                return None, "The maps aren't evenly spaced in time (the autocorrelation needs a time series)"
            U = np.where(f.status == 0, self.clean()[0], np.nan)
            return turbulence.integral_time(U, float(np.median(d))), None
        return self.windowed("integral_time", self.settings.outlier_k, compute)

    def integral_length(self, axis="x", component="u"):
        """The integral length scale of u (or v) along x or y at each grid point (mm; px in the sensor space;
        turbulence.integral_length: the spatial autocorrelation of the fluctuation, over the maps of the window,
        integrated to its first zero crossing towards both directions)."""
        def compute():
            f = self.fields
            U = np.where(f.status == 0, self.clean()[0 if component == "u" else 1], np.nan)
            positions = f.x[0] if axis == "x" else f.y[:, 0]
            steps = np.abs(np.diff(np.asarray(positions, np.float64)))
            spacing = float(np.median(steps)) if steps.size else 1.0
            return turbulence.integral_length(U, spacing, 2 if axis == "x" else 1)
        return self.windowed(f"integral_length_{component}{axis}", self.settings.outlier_k, compute)

    def dissipation(self):
        """The dissipation rate ε of the turbulent kinetic energy at each grid point (m²/s³; None in the sensor space),
        estimated from the in-plane gradients of the fluctuations (turbulence.planar_dissipation: the planar direct
        estimate, local isotropy for the gradients not measured), with the kinematic viscosity nu."""
        s = self.settings

        def compute():
            if self.sensor():
                return None
            f = self.fields
            U, V, _ = self.clean()
            reliable = f.status == 0
            eps = turbulence.planar_dissipation(np.where(reliable, U, np.nan), np.where(reliable, V, np.nan),
                                                np.asarray(f.x[0], np.float64) * 1e-3,
                                                np.asarray(f.y[:, 0], np.float64) * 1e-3, s.nu)
            eps[~reliable] = np.nan
            return eps
        return self.windowed("dissipation", (s.outlier_k, s.nu), compute)

    def ke_scales(self):
        """The time (s) and length (m) scales of the k-ε model at each grid point: k / ε and k^(3/2) / ε (k: the
        turbulent kinetic energy of the two components measured), None in the sensor space."""
        eps = self.dissipation()
        if eps is None:
            return None, None
        st = self.statistics()
        k = (st["u_rms"] ** 2 + st["v_rms"] ** 2) / 2
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = np.where(eps > 0, eps, np.nan)
            return k / eps, k ** 1.5 / eps

    def time_map(self, T, symbol):
        """A field of times (s; frames in the sensor space) as shown, and its label: in ms, or in flow times."""
        if self.sensor():
            return T, f"${symbol}$ (frames)"
        if self.uses_flow_time():
            return T / self.flow_time(), rf"${symbol}\ /\ {TFLOW}$"
        return T * 1000, f"${symbol}$ (ms)"

    def length_map(self, L, symbol):
        """A field of lengths (mm; px in the sensor space) as shown, and its label: in mm, or over L_ref."""
        if self.settings.l_ref and not self.sensor():
            return L / self.length(), rf"${symbol}\ /\ {LREF}$"
        return L, f"${symbol}$ ({self.unit()})"

    def draw_ke_time(self, fig, prefix):
        T = self.ke_scales()[0]
        if T is None:
            return self.message(fig, FIGURE_TITLES["ke_time"], "Only in the physical space (velocities in m/s)")
        F, label = self.time_map(T, r"T_{k\varepsilon}")
        self.stat_map(fig, prefix, F, rf"k-ε time scale $T_{{k\varepsilon}} = k\,/\,\varepsilon$ "
                      rf"($\nu$ = {self.settings.nu:.3g} m²/s)", label, "viridis", low=0, key="ke_time")

    def draw_ke_length(self, fig, prefix):
        L = self.ke_scales()[1]
        if L is None:
            return self.message(fig, FIGURE_TITLES["ke_length"], "Only in the physical space (velocities in m/s)")
        F, label = self.length_map(L * 1000, r"L_{k\varepsilon}")
        self.stat_map(fig, prefix, F, rf"k-ε length scale $L_{{k\varepsilon}} = k^{{3/2}}\,/\,\varepsilon$ "
                      rf"($\nu$ = {self.settings.nu:.3g} m²/s)", label, "viridis", low=0, key="ke_length")

    def draw_integral_time(self, fig, prefix):
        T, error = self.integral_time()
        if error:
            return self.message(fig, FIGURE_TITLES["integral_time"], error)
        F, label = self.time_map(T, "T_u")
        self.stat_map(fig, prefix, F, r"Integral time scale $T_u = \int_0^{\tau_0} \rho_{uu}(\tau)\,d\tau$", label,
                      "viridis", low=0, key="integral_time")

    def draw_integral_length(self, fig, prefix, axis="x", component="u"):
        symbol = f"L_{axis}^{component}"
        F, label = self.length_map(self.integral_length(axis, component), symbol)
        key = f"integral_length_{'' if component == 'u' else component}{axis}"
        self.stat_map(fig, prefix, F, rf"Integral length scale ${symbol} = \int_0^{{r_0}} "
                      rf"\rho_{{{component}{component}}}(r)\,dr$ (along {axis})", label, "viridis", low=0, key=key)

    def draw_integral_length_x(self, fig, prefix):
        self.draw_integral_length(fig, prefix, "x")

    def draw_integral_length_y(self, fig, prefix):
        self.draw_integral_length(fig, prefix, "y")

    def draw_integral_length_vx(self, fig, prefix):
        self.draw_integral_length(fig, prefix, "x", "v")

    def draw_integral_length_vy(self, fig, prefix):
        self.draw_integral_length(fig, prefix, "y", "v")

    def draw_mean_vorticity(self, fig, prefix):
        factor, label = self.vorticity_factor(r"\overline{\omega}")
        self.stat_map(fig, prefix, self.vorticity_statistics()["mean"] * factor,
                      r"Mean Vorticity field $\overline{\omega}$", label, "RdBu_r", True, key="mean_vorticity")

    def draw_mean_vorticity_abs(self, fig, prefix):
        factor, label = self.vorticity_factor(r"|\overline{\omega}|")
        self.stat_map(fig, prefix, np.abs(self.vorticity_statistics()["mean"] * factor),
                      r"Absolute Mean Vorticity field $|\overline{\omega}|$", label, "magma", low=0,
                      key="mean_vorticity_abs")

    def draw_vorticity_rms(self, fig, prefix):
        factor, label = self.vorticity_factor(r"\sigma_\omega")
        self.stat_map(fig, prefix, self.vorticity_statistics()["rms"] * factor,
                      r"Vorticity fluctuations (RMS) $\sigma_\omega$", label, "magma", low=0, key="vorticity_rms")

    def draw_mean_displacement(self, fig, prefix):
        # The mean of the norm, with the arrows of the mean displacement
        m = self.mean_displacements()
        if m is None:
            return self.message(fig, FIGURE_TITLES["mean_displacement"], "the displacements in pixels are unknown")
        self.stat_map(fig, prefix, m["norm"], r"Mean displacement $\overline{|d|}$",
                      r"$\overline{|d|}$ (px)", low=0, quiver=(m["x"], m["y"]))

    def draw_mean_displacement_x(self, fig, prefix):
        m = self.mean_displacements()
        if m is None:
            return self.message(fig, FIGURE_TITLES["mean_displacement_x"], "the displacements in pixels are unknown")
        self.stat_map(fig, prefix, m["x"], r"Mean displacement $\overline{d_x}$", r"$\overline{d_x}$ (px)")

    def draw_mean_displacement_y(self, fig, prefix):
        m = self.mean_displacements()
        if m is None:
            return self.message(fig, FIGURE_TITLES["mean_displacement_y"], "the displacements in pixels are unknown")
        self.stat_map(fig, prefix, m["y"], r"Mean displacement $\overline{d_y}$", r"$\overline{d_y}$ (px)",
                      "RdBu_r", True)

    def draw_valid(self, fig, prefix):
        st = self.statistics()
        self.stat_map(fig, prefix, st["valid"], r"Valid vectors $\mathrm{valid}_{\%}$",
                      r"$\mathrm{valid}_{\%}$ (% of the maps)", "viridis",
                      low=min(80.0, float(np.nanmin(st["valid"])) if np.isfinite(st["valid"]).any() else 0.0))

    def draw_peak_ratio(self, fig, prefix):
        if self.fields.peak_ratio is None:
            return self.message(fig, FIGURE_TITLES["peak_ratio"], "The correlation peak ratio is unknown")
        self.stat_map(fig, prefix, self.statistics()["peak_ratio"], r"Median peak ratio $\widetilde{S/N}$",
                      r"Median peak ratio $\widetilde{S/N}$", "viridis", low=1)

    def draw_peak_ratio_histogram(self, fig, prefix):
        f = self.fields
        if f.peak_ratio is None:
            return self.message(fig, FIGURE_TITLES["peak_ratio_histogram"], "The correlation peak ratio is unknown")
        a, b = self.window()
        v = f.peak_ratio[a:b][np.isfinite(f.peak_ratio[a:b])]
        ax = fig.add_subplot()
        hi = float(np.percentile(v, 99.5)) if v.size else 1
        ax.hist(v, bins=80, range=(0, hi), color="#2a78d6", edgecolor="white", linewidth=0.3)
        median = float(np.median(v)) if v.size else float("nan")
        ax.axvline(median, color="#eb6834", lw=1.5, label=f"Median: $S/N$ = {median:.2f}")
        ax.set_xlabel("Peak ratio $S/N$")
        ax.set_ylabel("Number of valid vectors")
        ax.legend(frameon=False)
        ax.set_title(f"{prefix}Peak ratio of the {v.size} valid vectors"
                     + (self.refs(self.window_text(), positions=False, scaled=False) if self.rolling() else ""),
                     fontweight="bold", fontsize="medium")

    def draw_peak_locking(self, fig, prefix):
        pl = self.peak_locking()
        if pl is None:
            return self.message(fig, FIGURE_TITLES["peak_locking"], "the displacements in pixels are unknown "
                                                                     "(time between the images and scale)")
        axs = fig.subplots(2, 1)
        for a, (key, comp) in zip(axs, (("u", "d_x"), ("v", "d_y"))):
            r = pl[key]
            if r is None:
                a.text(0.5, 0.5, "too few vectors", ha="center", transform=a.transAxes)
                continue
            # Histogram of the absolute values, its smooth copy and the integers (the peaks of a peak locking)
            a.plot(r["centres"], r["counts"], "-", lw=0.8, label="Histogram")
            a.plot(r["centres"], r["smooth"], "-", color="orange", lw=1.2, label="Smoothed (1 px)")
            for k in range(int(np.ceil(r["centres"][0])), int(np.floor(r["centres"][-1])) + 1):
                a.axvline(k, color="grey", lw=0.4)
            a.set_xlim(0, r["centres"][-1])
            a.set_xlabel(f"Absolute displacement $|{comp}|$ (px)")
            a.set_ylabel("Number of vectors")
            a.yaxis.set_major_formatter(FuncFormatter(sci2))
            a.yaxis.set_minor_formatter(NullFormatter())
            a.set_title(f"$|{comp}|$: {sci2(r['n'])} vectors, Quality = {(1-r['C']):.2f}", fontsize="medium")
            a.legend(fontsize="x-small")
        fig.suptitle(f"{prefix}Peak locking\n({self.n_maps_text()})",
                     fontweight="bold", fontsize="medium")

    def n_maps_text(self):
        """The maps of a figure of the statistics: N pairs (sensor space) or N maps, or those of the rolling
        statistics."""
        return self.window_text() or f"{self.fields.n} " + ("pairs" if self.sensor() else "maps")

    def draw_convergence(self, fig, prefix):
        c = self.convergence()
        if len(c["N"]) == 0:
            return self.message(fig, FIGURE_TITLES["convergence"], "a single map: at least 2 are needed")
        # Against the number of pairs (sensor space), or the averaging time: in flow times or in s (the maps of the
        # rolling statistics with them)
        n = len(c["N"]) + 1
        if self.sensor():
            T, xlabel, xmax = c["N"], "number of pairs $N$", n
        elif self.uses_flow_time():
            T, xlabel = self.flow_times(c["N"]), rf"averaging time $\Delta t / {TFLOW}$"
            xmax = float(self.flow_times(n))
        else:
            T, xlabel = c["N"] * self.map_interval(), r"averaging time $\Delta t$ (s)"
            xmax = n * self.map_interval()
        ax = fig.add_subplot()
        for key, label, color in (("mean_rel", r"$\overline{u}$", "tab:blue"), ("std_rel", r"$\sigma_u$", "tab:orange"),
                                  ("skew_rel", r"$\gamma_u$", "tab:green"), ("kurt_rel", r"$\beta_u$", "tab:red")):
            ax.plot(T, c[key], "-", lw=1.2, color=color, label=label)
        ax.set_ylabel(r"$\mathrm{RMS}\,|\phi_N - \phi_{N-1}|\ /\ \mathrm{RMS}\,|\phi_{N-1}|$ (%)")
        ax.set_title(f"{prefix}Convergence of the statistics $\\phi$"
                     + self.refs(self.window_text() or (self.t_flow_text() if self.uses_flow_time() else ""),
                                 positions=False, scaled=False),
                     fontweight="bold", fontsize="medium")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, xmax * 1.02)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize="small")

    def draw_higher(self, fig, prefix, key):
        F = self.higher_order()[key]
        kind, comp = key.split("_")
        if kind == "skewness":
            self.stat_map(fig, prefix, F, rf"Skewness $\gamma_{comp}$", rf"$\gamma_{comp}$ (0 for a Gaussian)",
                          "RdBu_r", True)
        else:
            self.stat_map(fig, prefix, F, rf"Kurtosis $\beta_{comp}$", rf"$\beta_{comp}$ (3 for a Gaussian)",
                          "RdBu_r", True, center=3.0)

    def draw_skewness_u(self, fig, prefix):
        self.draw_higher(fig, prefix, "skewness_u")

    def draw_kurtosis_u(self, fig, prefix):
        self.draw_higher(fig, prefix, "kurtosis_u")

    def draw_skewness_v(self, fig, prefix):
        self.draw_higher(fig, prefix, "skewness_v")

    def draw_kurtosis_v(self, fig, prefix):
        self.draw_higher(fig, prefix, "kurtosis_v")

    ############################################################################
    # Export

    def export(self, folder, index=0, dpi=EXPORT_DPI, name=None, progress=None, clims=None, cmaps=None, contours=None,
               figures=None, maps=None, videos=(), fps=VIDEO_FPS, video_maps=None,
               video_format=VIDEO_FORMAT, video_options=None, compares=None, image_options=None):
        """Save in a folder the figures of the post-processing as PNG (figures: only these ones), the key numbers
        (metrics.json) and the recipe (postprocessing.json): the instantaneous figures of the map index, or of each
        map of maps (indices, negative from the end: -1 the last map; each figure in its own subfolder
        [name]_PostProcessing_[element]), and the videos (video_format: "gif" or "mp4") of the instantaneous figures
        of videos over the maps of maps (all of them by default; video_maps: these maps instead) at fps frames per
        second. The files are named [name]_PostProcessing_[element](_[map]): name is the one of the project (None:
        the name of the measurement), the element says what the figure shows (FIGURE_FILES). progress(fraction,
        message) is called for each image and each frame; clims, cmaps, contours: colour ranges, colour maps and
        isocontours of the maps of some figures; video_options: the maps, fps and format of the videos of some
        figures instead ({key: {"maps": indices, "fps": ..., "format": ...}}); compares: the figures whose map is
        compared with the map of another figure, images and videos ({key: (other, fraction of the wiper)});
        image_options: the maps of the instantaneous figures of some figures instead of maps ({key: indices}).
        Returns the files written."""
        os.makedirs(folder, exist_ok=True)
        name = name or self.fields.name
        videos = [key for key in videos if self.instantaneous(key) and self.exported(key)]
        # The maps from the end for the negative indices (-1: the last one)
        n = self.fields.n
        index = map_indices([index], n)[0] if -n <= index < n else 0
        maps = None if maps is None else map_indices(maps, n)
        image_options = {key: map_indices(own, n) for key, own in (image_options or {}).items()}
        frames = map_indices(video_maps, n) if video_maps is not None else list(range(n) if maps is None else maps)
        # The maps, frame rate and format of each video: its own (video_options), else the common ones
        options = {key: dict(maps=frames, fps=fps, format=video_format) for key in videos}
        for key, own in (video_options or {}).items():
            if key in options:
                options[key].update(own)
                options[key]["maps"] = map_indices(options[key]["maps"], n)
        # The share of the progress of the images, and of each video (by their numbers of images and frames)
        # The modes of the POD: one PNG each (those of the range evaluated); summed, a figure of each map as the
        # instantaneous ones
        pod_modes = (self.pod_modes_exported() if (figures is None or "pod_mode" in figures)
                     and self.exported("pod_mode") else [])
        summed = bool(pod_modes) and self.pod_combined()
        pod_pictures = [] if summed else pod_modes
        keys = [key for key, _ in self.figures() if (figures is None or key in figures)
                and (key != "pod_mode" or summed)]
        n_images = sum(1 if not self.instantaneous(key) or image_options.get(key, maps) is None
                       else len(image_options.get(key, maps))
                       for key in keys if self.exported(key)) + len(pod_pictures)
        total = max(n_images + sum(len(o["maps"]) for o in options.values()), 1)

        def part(start, size):
            if progress is None:
                return None
            return lambda fraction, message: progress((start + fraction * size) / total, message)
        written = self.export_figures(folder, index=index, dpi=dpi, name=name,
                                      progress=part(0, n_images - len(pod_pictures)), clims=clims, cmaps=cmaps,
                                      contours=contours, keys=keys, maps=maps, compares=compares,
                                      image_options=image_options)
        shown = getattr(self, "pod_shown", None)
        try:
            for n, mode in enumerate(pod_pictures):
                self.pod_shown = mode  # Its file named with its number (export_element)
                written += self.export_figures(folder, index=index, dpi=dpi, name=name,
                                               progress=part(n_images - len(pod_pictures) + n, 1), clims=clims,
                                               cmaps=cmaps, contours=contours, keys=["pod_mode"], compares=compares)
        finally:
            self.pod_shown = shown
        start = n_images
        for key in videos:
            o = options[key]
            written.append(self.export_video(key, folder, fps=o["fps"], name=name, clim=(clims or {}).get(key),
                                             cmap=(cmaps or {}).get(key), contour=(contours or {}).get(key),
                                             maps=o["maps"], progress=part(start, len(o["maps"])), fmt=o["format"],
                                             compare=(compares or {}).get(key), clims=clims, cmaps=cmaps))
            start += len(o["maps"])
        with open(os.path.join(folder, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(self.metrics(), f, indent=1, ensure_ascii=False)
        with open(os.path.join(folder, "postprocessing.json"), "w", encoding="utf-8") as f:
            # The vector fields relative to the folder of the measurement (of the project), which is relative to
            # this one
            measurement = getattr(self.fields, "folder", None) or folder
            json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "measurement_folder": relative(measurement, folder),
                       "source": relative(self.fields.source, measurement) if self.fields.source else None,
                       "instantaneous_map": int(index) + 1, "recipe": " → ".join(self.settings.steps()),
                       "figures": None if figures is None else list(figures), "pod_modes": pod_modes or None,
                       "pod_summed": summed,
                       "maps": None if maps is None else [k + 1 for k in maps], "videos": videos,
                       "video_maps": [k + 1 for k in frames] if videos else None, "fps": fps,
                       "video_format": video_format if videos else None,
                       "compares": {k: list(v) for k, v in (compares or {}).items()} or None,
                       # Those of the videos with their own maps, frame rate or format
                       "image_options": {key: [k + 1 for k in own] for key, own in image_options.items()} or None,
                       "video_options": {key: dict(maps=[k + 1 for k in o["maps"]], fps=o["fps"], format=o["format"])
                                         for key, o in options.items()
                                         if (o["maps"], o["fps"], o["format"]) != (frames, fps, video_format)} or None,
                       "settings": asdict(self.settings)}, f, indent=1, ensure_ascii=False)
        written += [os.path.join(folder, n) for n in ("metrics.json", "postprocessing.json")]
        if progress is not None:
            progress(1.0, "Figures exported")
        return written
