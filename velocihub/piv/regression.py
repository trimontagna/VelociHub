"""Regression of the vector fields of the PIV/PTV with SPICY (spicy_vki: constrained radial basis functions,
Sperotto, Pieraccini & Mendez, von Karman Institute), after the PIV software. SPICY fits scattered data: after a PTV,
each pair is fitted on the links of its particles themselves (never on the binned cells); after a PIV, on its valid
vectors (the only data it gives). The fit is evaluated on the grid of the PIV (of the bins of the PTV) where there is
flow (the vectors without particles or masked are kept as the PIV gave them, the binned cells of a PTV). The gaps are
filled and the noise smoothed; vectors.npz keeps the fields of the PIV (raw_u_px, raw_v_px, raw_info: the binned
cells of a PTV) and the residual of each fit, with the number of data fitted.

The regression runs in its own process after the one of the PIV (ChainProcess: spicy_worker.py), by chunks of maps
saved in the work folder, so that a run stopped continues from the last chunk saved.

Coordinates are divided by the width of the images: the radii of the RBFs are fractions of it."""

import hashlib
import importlib.util
import json
import os
import sys

from .base import AUTO, Parameter, PIVProcess, WorkerProcess

# Regressions after the PIV software: key, name
REGRESSIONS = (("none", "None"), ("spicy", "SPICY"))
# Maps per chunk saved by the worker
SPICY_CHUNK = 20
# Version of the chunks the worker saves: those of another version are discarded (part of the key of a regression)
SPICY_FORMAT = 3
# The points per RBF of "auto": for the vectors of a PIV and for the particles of a PTV. More points per RBF make
# fewer, wider RBFs, so a smoother fit. The vectors of a PIV are already averaged over their windows: fewer points
# follow the field more closely (3 does better than 4 on most flows of the benchmark, and keeps 82 % of a wave 5 grid
# spacings long against 29 %). The links of a PTV each carry the noise of a single particle: fewer points average less
# of it (3 is worse than 4 on most flows). See docs/processing.md.
N_K_AUTO = {"vectors": (3,), "particles": (4,)}
SPICY_PARAMETERS = (
    Parameter("basis", "RBF basis", "gauss", "choice",
              "Gaussian RBFs, or C4 RBFs (compact support: faster, a little less smooth).", ("gauss", "c4")),
    Parameter("n_k", "Points per RBF (per level)", AUTO, "ints",
              ("Number of data (vectors, or particles after a PTV) each RBF should cover; one number per level of RBFs "
               "(e.g. 4 10: a fine and a coarse level). Fewer points: more RBFs, a closer fit and a longer "
               "regression. auto: 3 for the vectors of a PIV (already averaged over their windows), 4 for the "
               "particles of a PTV (each with the noise of its link).")),
    Parameter("method", "Collocation", "clustering", "choice",
              ("Where the RBFs are placed: k-means clustering of the vectors, Halton points (semirandom) or a "
               "regular grid."), ("clustering", "semirandom", "regular")),
    Parameter("r_min", "RBF radius min (image width)", 0.01,
              help="Smallest radius of the RBFs, where an RBF is 0.5, as a fraction of the image width."),
    Parameter("r_max", "RBF radius max (image width)", 0.3,
              help="Largest radius of the RBFs, as a fraction of the image width."),
    Parameter("eps_l", "RBF value at the nearest RBF", 0.7,
              help="Value of an RBF at the nearest other RBF (0 to 1), which sets its radius between the two above."),
    Parameter("alpha_div", "Divergence penalty (0: none)", 0.0,
              help=("Penalty on the 2D divergence over the whole field. Leave at 0 for an axisymmetric jet in a "
                    "plane through its axis: there, du/dx + dv/dy = -v/r, not 0.")),
    Parameter("n_hb", "Harmonic basis functions", 0, "int",
              help="Number of sine and cosine functions added to capture global trends (0: none)."),
    Parameter("k_cond", "Condition number", 1e12,
              help="Regularization: the matrix of the fit is regularized to this condition number."),
    Parameter("outlier_test", "Particle outlier test (0: none)", 3.0,
              help=("PTV: before the fit, each link is compared with the median of its nearest links (normalized "
                    "median test of scattered data, 2 passes); it is left out if its difference exceeds this many "
                    "times the median difference of the neighbours (plus 0.1 px). The PIV vectors are validated by "
                    "the PIV.")),
    Parameter("outlier_neighbours", "Neighbours of the outlier test", 10, "int",
              help="PTV: the nearest links each link is compared with (within 3 grid spacings; with fewer than 4, "
                   "the link is kept)."),
    Parameter("map_step", "Fit every n-th map", 1, "int",
              help=("SPICY fits the maps 1, 1 + n, 1 + 2n... only, and the results hold those maps (with their "
                    "pairs); 1: every map. For the statistics of a long time-resolved record: maps closer than about "
                    "half the integral time repeat each other, so fitting every n-th map changes the statistics "
                    "little and divides the time of SPICY by n. The PIV/PTV still computes every pair.")),
)


def check_regression(regression):
    """The regression of PIV settings ({"method": key, "settings": {parameter: value}}) completed with the defaults
    and validated: None if there is none, else {"method": "spicy", "settings": {...}}. ValueError with a message."""
    regression = regression or {}
    method = (regression.get("method") or "none").lower()
    if method not in dict(REGRESSIONS):
        raise ValueError(f'Unknown regression "{method}": {", ".join(k for k, _ in REGRESSIONS)}')
    if method == "none":
        return None
    given = regression.get("settings") or {}
    s = {p.key: p.parse(given[p.key]) if p.key in given else p.default for p in SPICY_PARAMETERS}
    if not 0 < s["r_min"] < s["r_max"]:
        raise ValueError("SPICY: 0 < RBF radius min < RBF radius max")
    if not 0 < s["eps_l"] < 1:
        raise ValueError("SPICY: RBF value at the nearest RBF between 0 and 1")
    if s["n_k"] != AUTO and any(n < 1 for n in s["n_k"]):
        raise ValueError("SPICY: points per RBF >= 1")
    if s["alpha_div"] < 0 or s["n_hb"] < 0:
        raise ValueError("SPICY: divergence penalty and harmonic basis functions >= 0")
    if not s["k_cond"] > 1:
        raise ValueError("SPICY: condition number > 1")
    if s["outlier_test"] < 0:
        raise ValueError("SPICY: particle outlier test >= 0 (0: none)")
    if s["outlier_neighbours"] < 4:
        raise ValueError("SPICY: neighbours of the outlier test >= 4")
    if s["map_step"] < 1:
        raise ValueError("SPICY: fit every n-th map, n >= 1")
    return {"method": method, "settings": s}


def points_per_rbf(settings, particles):
    """The points per RBF of SPICY settings (checked) for the particles of a PTV or the vectors of a PIV: n_k, or
    N_K_AUTO if it is "auto"."""
    n_k = settings["n_k"]
    return N_K_AUTO["particles" if particles else "vectors"] if n_k == AUTO else tuple(n_k)


def describe_regression(regression):
    """The regression checked (check_regression), in a line."""
    if regression is None:
        return "none"
    s = regression["settings"]
    return "SPICY: " + ", ".join(f"{p.label if p.label[:3].isupper() else p.label.lower()} {p.format(s[p.key])}"
                                 for p in SPICY_PARAMETERS)


def regression_text(regression):
    """The regression checked, as the interface saves it in a project (texts)."""
    if regression is None:
        return {"method": "none", "settings": {}}
    return {"method": regression["method"],
            "settings": {p.key: p.format(regression["settings"][p.key]) for p in SPICY_PARAMETERS}}


def regression_record(regression):
    """The regression checked, as saved with the results (JSON: tuples as lists)."""
    return None if regression is None else json.loads(json.dumps(regression))


def spicy_available():
    """(True, description) if SPICY can run (spicy_vki installed with the Python of the interface), else (False,
    reason)."""
    if importlib.util.find_spec("spicy_vki") is None:
        return False, "SPICY is not installed: pip install spicy_vki"
    return True, "SPICY (spicy_vki)"


def start_regression(work, regression, status, width, key, particles=False):
    """Start the regression of the displacement fields of the PIV (raw.npz of the work folder) in its own process:
    status, the grid points with flow (0); width, of the images in pixels; key, of the PIV and the regression (the
    chunks of another key are discarded); particles: the PIV is a PTV, whose links of particles (links.npz of the
    work folder) are fitted. Returns the WorkerProcess, which writes regressed.npz."""
    ok, reason = spicy_available()
    if not ok:
        raise RuntimeError(reason)
    # The worker gets the points per RBF themselves ("auto": those of the data, points_per_rbf)
    settings = {**regression["settings"], "n_k": points_per_rbf(regression["settings"], particles)}
    job = {"work": work, "settings": settings, "status": status, "width": width,
           "chunk": SPICY_CHUNK, "particles": bool(particles),
           "key": hashlib.sha1(json.dumps([key, regression_record(regression), bool(particles), SPICY_FORMAT],
                                          sort_keys=True, default=str).encode()).hexdigest()[:16]}
    return WorkerProcess(sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "spicy_worker.py"),
                         job, work)


class ChainProcess(PIVProcess):
    """The PIV process, then the one started by then(done) when it is done (the regression). Its own message done is
    the one of the PIV, with regression_seconds (and regression_resumed: the maps fitted before)."""

    def __init__(self, first, then):
        self.current, self.then, self.piv_done = first, then, None

    def poll(self):
        messages = []
        for message in self.current.poll():
            if message["type"] == "done" and self.then is not None:
                self.piv_done, then, self.then = message, self.then, None
                try:
                    self.current = then(message)
                except Exception as e:
                    messages.append({"type": "error", "text": f"the regression could not start: {e}"})
                    self.current = _Ended()
                    break
                continue
            if message["type"] == "done" and self.piv_done is not None:
                message = {**self.piv_done, "regression_seconds": message.get("seconds"),
                           "regression_resumed": message.get("resumed")}
            messages.append(message)
        return messages

    def stop(self):
        self.current.stop()

    def kill(self):
        self.current.kill()

    def wait(self, timeout=None):
        self.current.wait(timeout) if timeout is not None else self.current.wait()

    @property
    def running(self):
        return self.then is not None or self.current.running


class _Ended(PIVProcess):
    """A process already ended (the regression that could not start)."""

    def poll(self):
        return []

    def stop(self):
        pass

    @property
    def running(self):
        return False
