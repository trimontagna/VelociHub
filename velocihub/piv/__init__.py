"""PIV of the pre-processed images, with a PIV software chosen among the engines registered (PaIRS-UniNa, or the
PTV of TracTrac: ptv_engine.py), and an optional regression of the vector fields after it (SPICY: regression.py).

    from velocihub import piv

    vectors = piv.run_piv(images, None, "PaIRS-UniNa", {"windows": (128, 64, 32), "overlap": 50})  # in pixels
    found = piv.count_particles(image, center=(960, 600), sizes=(128, 64, 32))  # particles of the windows
"""

from . import pairs_engine, ptv_engine
from .base import (CHUNK, DEFAULT_ENGINE, ENGINES, FOV_PRESETS, PIV_FOLDER, PRESET_FRAME_RATE, VECTORS_FILE, Camera,
                   ConsoleReport, Parameter, PIVEngine, PIVProcess, PIVRun, WorkerProcess, fov_help, get_engine,
                   register, run_piv)
from .pairs_engine import PaIRSEngine, find_pairs_python, has_pairs
from .ptv_engine import PTV_FOLDER, TracTracEngine, sequence_of
from .particles import MIN_PARTICLES, count_particles
from .regression import (REGRESSIONS, SPICY_PARAMETERS, check_regression, describe_regression, regression_record,
                         regression_text, spicy_available)

__all__ = ["CHUNK", "DEFAULT_ENGINE", "ENGINES", "FOV_PRESETS", "MIN_PARTICLES", "PIV_FOLDER", "PRESET_FRAME_RATE",
           "VECTORS_FILE", "Camera", "ConsoleReport", "PaIRSEngine", "Parameter", "PIVEngine", "PIVProcess", "PIVRun",
           "WorkerProcess", "count_particles", "find_pairs_python", "fov_help", "get_engine", "has_pairs",
           "pairs_engine", "register", "run_piv", "REGRESSIONS", "SPICY_PARAMETERS", "check_regression",
           "describe_regression", "regression_record", "regression_text", "spicy_available", "ptv_engine",
           "PTV_FOLDER", "TracTracEngine", "sequence_of"]
