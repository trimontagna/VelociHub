"""PaIRS-UniNa (University of Naples Federico II): the PIV engine of VelociHub, from PIV_PAIRS_batch.py.

PaIRS-UniNa is a compiled library for Python 3.10 to 3.13 (pip install PaIRS-UniNa). It runs in another process
(pairs_worker.py), with a Python interpreter where it is installed, which may differ from the one of the
interface: the interpreter given, or else this one if PaIRS-UniNa is installed here, or else the first conda
environment where it is installed.

Process of each pair (as the PaIRS "standard" process): multigrid windows (128 -> 64 -> 32 px by default) with
image deformation, a number of final iterations, and the PaIRS median test.
"""

import glob
import importlib.util
import os
import sys

from .base import CHUNK, Parameter, PIVEngine, WorkerProcess, register

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairs_worker.py")


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
    description = "PaIRS-UniNa standard process: multigrid, image deformation, PaIRS median test"
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
        return settings

    def start(self, images, pairs, settings, work, key=None):
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(reason)
        job = {"images": list(images), "pairs": [list(p) for p in pairs], "settings": settings, "chunk": CHUNK,
               "key": key if key is not None else {"pairs": [list(p) for p in pairs], "settings": settings}}
        return WorkerProcess(self.interpreter(), WORKER, job, work)
