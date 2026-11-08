"""PaIRS-UniNa (University of Naples Federico II): the PIV engine of VelociHub, from PIV_PAIRS_batch.py.

PaIRS-UniNa is a compiled library for Python 3.10 to 3.13 (pip install PaIRS-UniNa). It runs in another process
(pairs_worker.py), with a Python interpreter where it is installed, which may differ from the one of the
interface: the interpreter given, or else this one if PaIRS-UniNa is installed here, or else the first conda
environment where it is installed.

Process of each pair (by default the PaIRS "standard" process): multigrid windows (128 -> 64 -> 32 px by default)
with image deformation, a number of final iterations, and the PaIRS median test. The settings of PaIRS that change
the results are parameters of the engine (PAIRS_CODES: their values as PaIRS numbers them); their help follows the
PaIRS interface (its tooltips) and its configuration files.
"""

import glob
import importlib.util
import os
import sys

from .base import CHUNK, Parameter, PIVEngine, WorkerProcess, register

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairs_worker.py")

# The choices of the parameters, as PaIRS numbers them (Process_Tab.py of PaIRS-UniNa)
PAIRS_CODES = {
    # Image interpolation of the image deformation (IntIniz, IntFin): B-spline of order n = 50 + n, sinc of half-width
    # n = 40 + n, shift theorem of kernel width n = 20 + n
    "image_interpolation": {"bspline3": 53, "bspline5": 55, "bspline7": 57, "bicubic": 7, "biquadratic": 2,
                            "bilinear": 5, "simplex": 1, "linrev": 10, "sinc3": 43, "shift5": 25, "moving": 3,
                            "none": 0},
    # Interpolation of the displacement field, the dense predictor (IntVel)
    "velocity_interpolation": {"bspline2": 52, "bspline3": 53, "bilinear": 1, "linrev": 5, "simplex": 2,
                               "shift": 3},
    # Sub-pixel interpolation of the correlation peak (IntCorr; PaIRS adds 3)
    "peak_interpolation": {"gauss": 0, "gauss_rev": 1, "simplex": 2},
    # Median test (FlagValid)
    "median_test": {"universal": 2, "classic": 1, "none": 0},
    # Correction of the outliers (FlagCorrezioneVel)
    "outlier_replacement": {"weighted": 1, "average": 0, "iterative": 2},
    # Weighting windows (FlagWindowing: of the correlation map; FlagCalcVel: of the displacement field), Astarita 2007
    "correlation_window": {"tophat": 0, "tophat50": 3, "nogueira": 1, "blackman": 5, "bharris": 7, "hann": 23,
                           "triangular": 22},
    "predictor_window": {"tophat": 0, "nogueira": 1, "blackman": 5, "bharris": 7, "hann": 23, "triangular": 22},
    # Largest displacement allowed (MaxDisp < 0: the window size over that number)
    "max_displacement": {"1/2": -2, "1/3": -3, "1/4": -4, "1/5": -5},
}


def has_pairs(python):
    """PaIRS-UniNa is installed for this Python interpreter (looked for in its site-packages, without running
    it)."""
    root = os.path.dirname(os.path.abspath(python))
    folders = [os.path.join(root, "Lib", "site-packages", "PaIRS_UniNa")]
    folders += glob.glob(os.path.join(os.path.dirname(root), "lib", "python3*", "site-packages", "PaIRS_UniNa"))
    return any(os.path.isdir(f) for f in folders)


def conda_environments():
    """Folders of the conda environments of this computer (and of the conda installations)."""
    roots = set()
    prefix = os.path.abspath(sys.prefix)
    if os.path.basename(os.path.dirname(prefix)).lower() == "envs":
        roots.add(os.path.dirname(os.path.dirname(prefix)))
    else:
        roots.add(prefix)
    if os.environ.get("CONDA_EXE"):
        roots.add(os.path.dirname(os.path.dirname(os.environ["CONDA_EXE"])))
    home = os.path.expanduser("~")
    roots.update(os.path.join(home, name) for name in ("anaconda3", "miniconda3", "miniforge3", "mambaforge"))
    folders = []
    for root in sorted(roots):
        envs = os.path.join(root, "envs")
        folders.append(root)
        if os.path.isdir(envs):
            folders += sorted(os.path.join(envs, e) for e in os.listdir(envs))
    folders += sorted(glob.glob(os.path.join(home, ".conda", "envs", "*")))
    return folders


def find_pairs_python():
    """Python interpreter where PaIRS-UniNa is installed: this one, or else the one of the first conda environment
    where it is installed. None if there is none."""
    if importlib.util.find_spec("PaIRS_UniNa") is not None:
        return sys.executable
    for env in conda_environments():
        python = os.path.join(env, "python.exe") if sys.platform == "win32" else os.path.join(env, "bin", "python")
        if os.path.isfile(python) and has_pairs(python):
            return python
    return None


@register
class PaIRSEngine(PIVEngine):
    """PaIRS-UniNa, run by the Python interpreter python (None: found automatically, see find_pairs_python)."""

    key = "pairs"
    name = "PaIRS-UniNa"
    label = "PaIRS"
    description = "PaIRS-UniNa: multigrid, image deformation, median test (the PaIRS standard process by default)"
    parameters = (
        Parameter("windows", "Window sizes (px)", (128, 64, 32), "ints",
                  "Interrogation windows of the multigrid passes, from the first one to the final one. The final "
                  "window sets the resolution: 32 px with 50 % overlap gives a vector every 16 px."),
        Parameter("overlap", "Overlap (%)", 50.0, "float",
                  "Overlap of neighbouring windows: the vector spacing is window x (1 - overlap)."),
        Parameter("final_iterations", "Final iterations", 2, "int",
                  "Iterations of the final pass, with the image deformed by the displacement of the previous one."),
        Parameter("min_mean", "Min. window mean (grey levels)", 0.01, "float",
                  "Windows darker than this are not correlated. The PaIRS default (2) rejects almost every window "
                  "of dark images: 0.01 only rejects windows without any particle (the masked areas)."),
        Parameter("min_std", "Min. window std (grey levels)", 0.05, "float",
                  "Windows with less contrast than this are not correlated (PaIRS default: 3)."),
        Parameter("subtract_mean", "Subtract the mean image", False, "bool",
                  "Subtract from each image the mean of about 200 images of the set (negative values set to 0), "
                  "as PIV_PAIRS_batch.py. Not needed after the POD background removal of the pre-processing."),
        # The settings of the PaIRS process (the defaults: its standard process)
        Parameter("border", "First vector one spacing from the edge", True, "bool",
                  "PaIRS: \"If activated the first vector is placed at a distance equal to the spacing from the image "
                  "border\" (else the first window touches the border)."),
        Parameter("image_interpolation", "Image interpolation", "bspline3", "choice",
                  "PaIRS: type of interpolation for the image deformation method, in the multigrid passes and the "
                  "final iterations. bspline3/5/7: B-spline of order 3, 5, 7 (standard: 3; advanced: 7; the higher, "
                  "the less bias and the more cost); bicubic, biquadratic, bilinear: repeated polynomial; simplex; "
                  "linrev: linear revitalized; sinc3: sinc (Whittaker-Shannon) of kernel half-width 3; shift5: shift "
                  "theorem of kernel width 5; moving: moving window (symmetric), the window shifted, not deformed; "
                  "none: no image deformation (Astarita & Cardone 2005).",
                  tuple(PAIRS_CODES["image_interpolation"])),
        Parameter("velocity_interpolation", "Predictor interpolation", "bspline2", "choice",
                  "PaIRS: type of interpolation for the computation of the dense predictor, the displacement of "
                  "every pixel that deforms the images from the vectors of the previous pass. bspline2/3: B-spline "
                  "of order 2 (standard), 3; bilinear; linrev: linear revitalized; simplex; shift: shift theorem.",
                  tuple(PAIRS_CODES["velocity_interpolation"])),
        Parameter("peak_interpolation", "Peak interpolation", "gauss", "choice",
                  "PaIRS: type of interpolation for the local estimation of the correlation peak, the sub-pixel "
                  "displacement: gauss: Gaussian 3-point fit (classic, standard); gauss_rev: Gaussian (reviewed); "
                  "simplex. Its bias towards integer displacements is the peak locking (figure Peak locking of the "
                  "post-processing).", tuple(PAIRS_CODES["peak_interpolation"])),
        Parameter("median_test", "Median test", "universal", "choice",
                  "PaIRS: type of median test of the vectors (validation): universal: normalized median test "
                  "(Westerweel & Scarano 2005, standard); classic: median test; none: no median test. The vectors "
                  "rejected are replaced (Outlier replacement) and marked invalid.",
                  tuple(PAIRS_CODES["median_test"])),
        Parameter("median_kernel", "Median test semi-kernel", 1, "int",
                  "PaIRS: semi-size of the kernel used for the test: the vectors within this many spacings, "
                  "(2 n + 1)² - 1 neighbours (1: the 8 neighbours, standard)."),
        Parameter("median_threshold", "Median test threshold", 2.0, "float",
                  "PaIRS: alpha threshold for the median test, on the residual normalized by the median residual of "
                  "the neighbours (advised value 2, range 1 to 3): lower rejects more vectors."),
        Parameter("median_epsilon", "Median test epsilon (px)", 0.5, "float",
                  "PaIRS: epsilon threshold for the median test, the minimum value allowed for the normalization "
                  "(the allowed error in pixels: the advised value of PaIRS is 0.1 to 0.5; 0: no limit). It keeps "
                  "the test from rejecting vectors of uniform flows, where the residuals are all near 0."),
        Parameter("sn_threshold", "S/N test threshold (0: off)", 0.0, "float",
                  "PaIRS: signal-to-noise test, threshold on S/N defined as the ratio of the first and second largest "
                  "peaks in the correlation map (advised value 1.5). 0: no test (standard)."),
        Parameter("cc_threshold", "Correlation peak test (0: off)", 0.0, "float",
                  "PaIRS: correlation peak test, minimum value of the correlation peak allowed (normalized, advised "
                  "value 0.25). 0: no test (standard)."),
        Parameter("outlier_replacement", "Outlier replacement", "weighted", "choice",
                  "PaIRS: type of correction of the outliers, from the valid neighbours: weighted: distance-weighted "
                  "average (standard); average; iterative: iterative average.",
                  tuple(PAIRS_CODES["outlier_replacement"])),
        Parameter("second_peak", "Correct with the second peak", True, "bool",
                  "PaIRS: correct an outlier with the displacement related to the second largest peak in the "
                  "correlation map when it passes the test (standard), before the replacement by the neighbours."),
        Parameter("correlation_window", "Correlation weighting window", "tophat", "choice",
                  "PaIRS: type of weighting window of the interrogation windows for the correlation map (Astarita "
                  "2007): tophat: full, no windowing (standard); tophat50: top-hat at 50 %; nogueira (Nogueira et al. "
                  "1999); blackman; bharris: Blackman-Harris (3 terms, 67 dB); hann; triangular. A window "
                  "concentrated at the centre raises the spatial resolution and the noise.",
                  tuple(PAIRS_CODES["correlation_window"])),
        Parameter("predictor_window", "Displacement weighting window", "tophat", "choice",
                  "PaIRS: type of weighting window for the displacement field, the filter of the predictor between "
                  "the iterations (Astarita 2007): tophat (standard), nogueira, blackman (advanced process), bharris, "
                  "hann, triangular. Stabilizes the iterative image deformation (it would otherwise amplify the "
                  "small wavelengths).", tuple(PAIRS_CODES["predictor_window"])),
        Parameter("predictor_halfwidth", "Displacement window half-width", 0, "int",
                  "PaIRS: semi-size of the weighting window for the displacement field (vectors). If 0 the size is "
                  "equal to that of the interrogation window (standard)."),
        Parameter("max_displacement", "Max. displacement (of the window)", "1/4", "choice",
                  "PaIRS: maximum displacement allowed as a fraction of the interrogation window size, in each "
                  "pass (standard: 1/4, the one-quarter rule). The peaks beyond it are ignored.",
                  tuple(PAIRS_CODES["max_displacement"])),
    )

    def __init__(self, python=None):
        self.python = python

    def interpreter(self):
        """Python interpreter running PaIRS-UniNa: the one given, or found (None if none)."""
        return self.python or find_pairs_python()

    def available(self):
        python = self.interpreter()
        if python is None:
            return False, ("PaIRS-UniNa not found. It needs Python 3.10 to 3.13: install it in a conda environment "
                           "(conda create -n PaIRS python=3.13, then pip install PaIRS-UniNa in it) and choose "
                           "the Python interpreter of that environment.")
        if not os.path.isfile(python):
            return False, f"Python interpreter not found: {python}"
        return True, f"{self.name} with {python}"

    def check(self, settings):
        settings = super().check(settings)
        if any(w < 4 for w in settings["windows"]) or list(settings["windows"]) != sorted(settings["windows"],
                                                                                        reverse=True):
            raise ValueError("Window sizes: at least 4 px, from the largest to the smallest")
        if not 0 <= settings["overlap"] < 100:
            raise ValueError("Overlap: from 0 to 100 % (excluded)")
        if settings["final_iterations"] < 1:
            raise ValueError("Final iterations: at least 1")
        if settings["min_mean"] < 0 or settings["min_std"] < 0:
            raise ValueError("Minimum window mean and std: positive or zero")
        if not 1 <= settings["median_kernel"] <= 5:
            raise ValueError("Median test semi-kernel: from 1 to 5")
        if not settings["median_threshold"] > 0 or settings["median_epsilon"] < 0:
            raise ValueError("Median test: threshold positive, epsilon positive or zero")
        if settings["sn_threshold"] < 0 or not 0 <= settings["cc_threshold"] < 1:
            raise ValueError("S/N test threshold: positive or zero (0: off); correlation peak test: from 0 to 1")
        if settings["predictor_halfwidth"] < 0:
            raise ValueError("Displacement window half-width: positive or zero (0: the window size)")
        return settings

    def start(self, images, pairs, settings, work, key=None):
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(reason)
        codes = {key: PAIRS_CODES[key][settings[key]] for key in PAIRS_CODES if key in settings}
        job = {"images": list(images), "pairs": [list(p) for p in pairs], "settings": settings, "codes": codes,
               "chunk": CHUNK,
               "key": key if key is not None else {"pairs": [list(p) for p in pairs], "settings": settings}}
        return WorkerProcess(self.interpreter(), WORKER, job, work)
