"""TracTrac (Heyman 2019, MIT licence: velocihub/piv/tractrac): particle tracking velocimetry (PTV) of the image sequence,
the second engine of VelociHub after PaIRS-UniNa.

TracTrac detects the particles of each frame and links them over three frames, a motion model (trained over passes
back and forth over the sequence) predicting their displacement; it runs in its own process (ptv_worker.py) with the
Python interpreter of the interface. The particles tracked give the links of the pairs of consecutive frames, binned
on the grid of the PIV (cells of the grid spacing centred on its points, those of PaIRS with 32 px windows and 50 %
overlap by default): the vector fields, as those of the other engines, with the number of particles of each cell
(raw.npz: count) and the tracks (tracks.npz). It needs the pairs of a sequence: (k, k + 1) of every image, or of every
frame_step images.
"""

import importlib.util
import os
import sys

import numpy as np
from PIL import Image

from ..fields import PTV_FOLDER
from .base import Parameter, PIVEngine, WorkerProcess, register

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptv_worker.py")
# Modules tractrac.py imports
MODULES = ("cv2", "scipy", "matplotlib", "imutils", "parse", "h5py")
PACKAGES = {"cv2": "opencv-python"}


def sequence_of(pairs):
    """Frames tracked (indices of the images, in time order) for pairs of consecutive frames of a sequence:
    ValueError if the pairs aren't."""
    pairs = [tuple(int(k) for k in p) for p in pairs]
    if not pairs or any(b <= a for a, b in pairs) or any(p[1] != q[0] for p, q in zip(pairs, pairs[1:])):
        raise ValueError("TracTrac tracks a sequence of images: the pairs must be its consecutive frames (k, k + 1) "
                         "of every image, or of every frame_step images with a pair step equal to the frame step")
    return [pairs[0][0]] + [b for _, b in pairs]


def grid_of(width, height, spacing):
    """Points of the grid (px from the top-left corner of the images): every spacing px from spacing to the size of the
    images minus spacing, as the grid of PaIRS with windows of 2 spacing and 50 % overlap."""
    return dict(x=np.arange(spacing, width - spacing + 0.5, spacing).tolist(),
                y=np.arange(spacing, height - spacing + 0.5, spacing).tolist(), spacing=spacing)


@register
class TracTracEngine(PIVEngine):
    """TracTrac, run by the Python interpreter of the interface."""

    key = "tractrac"
    name = "TracTrac"
    description = "TracTrac PTV: particles tracked over the sequence, their displacements binned on the grid"
    results_folder = PTV_FOLDER
    sequence = True
    method = "PTV"
    parameters = (
        Parameter("spacing", "Grid spacing / bin size (px)", 16, "int",
                  "The displacements of the particles of each pair are averaged in cells of this size centred on the "
                  "points of the grid: 16 px gives the grid of PaIRS with 32 px windows and 50 % overlap, so that the "
                  "PTV and the PIV compare point by point."),
        Parameter("min_particles", "Min. particles per cell", 1, "int",
                  "A cell with fewer particles in a pair is invalid in that pair."),
        Parameter("median_test", "Median test threshold (0: none)", 2.0, "float",
                  "Normalized median test of each cell against its 8 neighbours (Westerweel & Scarano 2005, noise "
                  "0.1 px): a cell above this threshold is invalid, as is a cell with fewer than 2 valid neighbours."),
        Parameter("particle_size", "Particle size (px)", 3.0, "float",
                  "Typical size of the particle images: the size of the detection kernel (TracTrac peak_conv_size; "
                  "its default is 2 px). On the jet images, 3 px detects fewer particles than 2 px (6,000 instead of "
                  "10,700 per frame) but fewer peaks of noise: the links agree better with the PIV and with their "
                  "neighbours."),
        Parameter("kernel", "Detection kernel", "dog", "choice",
                  "Filter of the images before the detection of the particles: difference of Gaussians (dog), "
                  "Laplacian of Gaussian (log), none, or the corners of Shi and Tomasi (shi-tomasi).",
                  ("dog", "log", "none", "shi-tomasi")),
        Parameter("auto_threshold", "Automatic detection threshold", True, "bool",
                  "Threshold of the detection: the mean of the filtered second image plus half its standard "
                  "deviation (not with shi-tomasi, which uses the manual threshold)."),
        Parameter("threshold", "Manual detection threshold", 0.02, "float",
                  "Threshold of the filtered images (grey levels divided by 256 or 65536) without the automatic one."),
        Parameter("peak_neigh", "Min. distance between particles (px)", 1, "int",
                  "Particles closer than this are one particle (TracTrac peak_neigh)."),
        Parameter("subpixel", "Subpixel position", "gaussian", "choice",
                  "Fit of the position of each particle: Gaussian or quadratic.", ("gaussian", "quadratic")),
        Parameter("dark", "Dark particles", False, "bool", "Track dark particles on a bright background."),
        Parameter("motion", "Motion model", True, "bool",
                  "Predict the displacement of each particle from those of its neighbours in the frames before, "
                  "which links the particles that move further than their spacing."),
        Parameter("motion_av", "Motion model neighbours", 10, "int",
                  "Particles over which the motion model is averaged."),
        Parameter("motion_it", "Motion model iterations", 0, "int",
                  "Iterations of the motion model at each frame (0: one prediction)."),
        Parameter("vid_loop", "Training passes (even)", 2, "int",
                  "Passes back and forth over the sequence to train the motion model before the last one (forward), "
                  "whose tracks are kept: even, 0 for none. Each pass takes as long as the tracking."),
        Parameter("filter", "Outlier filter", True, "bool",
                  "Reject the particles whose displacement departs from the motion model."),
        Parameter("filter_std", "Outlier threshold (std)", 1.5, "float",
                  "Rejected beyond this many standard deviations of the error of the motion model."),
        Parameter("filter_time", "Outlier threshold memory", 1.0, "float",
                  "Weight of the previous frames in the threshold of the outliers (TracTrac filter_time)."),
        Parameter("background", "Subtract a moving background", False, "bool",
                  "TracTrac's own background (the mean of the first 100 frames, updated at each frame). Not needed "
                  "on images pre-processed by VelociHub."),
        Parameter("background_speed", "Background adaptation speed", 0.01, "float",
                  "Speed of the update of the background (0 to 1)."),
        Parameter("median_filter", "Median filter", False, "bool", "Median filter of the images before detection."),
        Parameter("median_size", "Median filter size (px)", 3, "int", "Size of the median filter: 3 or 5."),
    )

    def available(self):
        missing = [m for m in MODULES if importlib.util.find_spec(m) is None]
        if missing:
            return False, (f"TracTrac needs {', '.join(missing)} in the Python of the interface ({sys.executable}): "
                           f"pip install {' '.join(PACKAGES.get(m, m) for m in missing)}")
        return True, f"{self.name} (velocihub/piv/tractrac) with {sys.executable}"

    def check(self, settings):
        settings = super().check(settings)
        if settings["vid_loop"] < 0 or settings["vid_loop"] % 2:
            raise ValueError("Training passes: even (0, 2, 4...), so that the last pass is forward")
        if settings["spacing"] < 4:
            raise ValueError("Grid spacing: at least 4 px")
        if settings["min_particles"] < 1:
            raise ValueError("Min. particles per cell: at least 1")
        if settings["median_test"] < 0:
            raise ValueError("Median test threshold: positive or 0")
        if settings["particle_size"] <= 0 or settings["peak_neigh"] < 1:
            raise ValueError("Particle size: positive; min. distance between particles: at least 1 px")
        if settings["motion_av"] < 1 or settings["motion_it"] < 0:
            raise ValueError("Motion model: at least 1 neighbour, 0 or more iterations")
        if settings["filter_time"] < 0 or not 0 < settings["background_speed"] <= 1:
            raise ValueError("Outlier threshold memory: positive or 0; background adaptation speed: 0 to 1")
        if settings["median_size"] not in (3, 5):
            raise ValueError("Median filter size: 3 or 5")
        if settings["kernel"] == "shi-tomasi" and not 0 < settings["threshold"] < 1:
            raise ValueError("The corners of Shi and Tomasi need a manual threshold between 0 and 1")
        return settings

    def describe(self, settings):
        return "PTV: " + super().describe(settings)

    def start(self, images, pairs, settings, work, key=None):
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(reason)
        sequence = sequence_of(pairs)
        with Image.open(images[sequence[0]]) as im:
            width, height = im.size
        job = {"images": list(images), "sequence": sequence, "settings": settings,
               "grid": grid_of(width, height, settings["spacing"]),
               "key": key if key is not None else {"sequence": sequence, "settings": settings}}
        return WorkerProcess(sys.executable, WORKER, job, work)
