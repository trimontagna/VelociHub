"""VelociHub: pre-processing of PIV images (POD-based background removal, masks, contrast enhancement), PIV with
PaIRS-UniNa or PTV with TracTrac, post-processing of the vector fields, batch processing and archiving. Usable from a
script (this API) or with its interface (launch_gui, the command velocihub).

Author: Tristan Montagnier (tristanmontagnier@gmail.com), made with Claude (Anthropic), with input from Filippo
Vaselli, Reda Lamrani and M.A. Mendez, as part of a PIV Laboratory at the von Karman Institute for Fluid Dynamics.
Licence: CC BY-NC-SA 4.0.
The open-source software it uses, their authors and licences: credits.py (Help > Credits in the interface,
CREDITS.md in the repository).

The POD-based background removal is an implementation of the algorithm described in the following research paper:
"POD-based background removal for Particle Image Velocimetry"
M.A. Mendez, M. Raiola, A. Masullo, S. Discetti, A. Ianiro, R. Theunissen, J.-M. Buchlin
Experimental Thermal and Fluid Science, Volume 80, January 2017, Pages 181-192.
It is distributed under the CC BY-NC-SA 4.0 license by Creative Commons. The PIV is computed by PaIRS-UniNa
(University of Naples Federico II), a compiled library for Python 3.10 to 3.13 run in its own process.

    import velocihub as vh

    # 1. Pre-processing (Python port of background_removal.m)
    with vh.PODBackgroundRemoval("images", pattern="*.tif") as session:
        session.load_masks("masks.json")
        session.set_contrast(["capping"], stage="post", cap_n=10)
        session.evaluate_modes(6)
        session.export_spectrum("results/spectrum.png")
        images = session.piv_images(n_modes_removed=6)      # exported to images/PreProcessing
        masked = session.masked_pixels((1200, 1920))

    # 2. PIV (results in pixels in images/PIV)
    vectors = vh.run_piv(images, None, "PaIRS-UniNa", {"windows": (128, 64, 32), "overlap": 50},
                         output="images/PIV", masked=masked)

    # 3. Post-processing with the camera settings (figures and key numbers in images/PostProcessing)
    camera = vh.Camera(fov_width_mm=196.1, x_offset_mm=0, frame_rate_hz=150, y_shift=0.5)
    fields = vh.load_vectors(vectors, camera=camera)
    post = vh.PostProcessing(fields, vh.PostSettings(outlier_k=4))
    post.export("images/PostProcessing")

    # With the interface
    vh.launch_gui("images", pattern="*.tif")

    # Batch: pre-processing and PIV of measurement folders with the default settings of the interface
    for result in vh.batch.process_folders(["X:/FOV_3"]):
        print(result.folder, result.describe())

Modules: preprocessing (POD background removal, masks and contrast enhancement), piv (camera settings and PIV
engines: PaIRS-UniNa), postprocessing (statistics, vorticity, turbulence, potential core, displacements, quality of
the PIV, peak locking and convergence, and their figures, in the sensor or the physical space), fields (the vector
fields and the base of their analyses),
project (the file VelociHub_project.json of the interface, in the image folder, which keeps the settings and the
results of the three steps), batch (pre-processing and PIV of measurement folders, one after the other) and gui
(the Tkinter interface).
"""

import sys


def _console_safe():
    """The messages of VelociHub use characters (→, σ, ω...) that some consoles can't encode (cp1252 on Windows, output
redirected to a file): printed as ? instead of stopping the processing."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


_console_safe()

from . import api, batch, jet, pipelines, piv, settings  # noqa: E402 (after the console made safe)
from .api import Project, projects, run, stitch
from .credits import AUTHOR as __author__, __version__
from .piv import PIV_FOLDER, Camera, PIVEngine, PIVRun, get_engine, register, run_piv
from .fields import VectorFields, load_vectors
from .postprocessing import FIGURES, POSTPROCESSING_FOLDER, PostProcessing, PostSettings
from .preprocessing import (FILTERS, POD_FILE, PREPROCESSING_FOLDER, STAGES, ContrastSettings, Mask,
                            PODBackgroundRemoval, clahe, draw_spectrum, edge_enhancement, imread, imwrite,
                            intensity_capping, minmax_normalization)
from .project import PROJECT_FILE, read_project

__all__ = ["__author__", "__version__", "Project", "api", "batch", "pipelines", "projects", "run", "settings", "stitch", "Camera", "ContrastSettings", "FIGURES", "FILTERS", "Mask", "PIVEngine",
           "PIVRun", "PIV_FOLDER", "PODBackgroundRemoval", "POD_FILE", "POSTPROCESSING_FOLDER", "PREPROCESSING_FOLDER",
           "PROJECT_FILE", "PostProcessing", "PostSettings", "STAGES", "VectorFields", "clahe", "draw_spectrum",
           "edge_enhancement", "get_engine", "imread", "imwrite", "intensity_capping", "launch_gui", "load_vectors",
           "jet", "minmax_normalization", "piv", "read_project", "register", "run_piv"]


def launch_gui(folder=None, pattern=None, masks=None, contrast=None):
    """Run the interface, optionally with the training images of a folder (all the images, or those matching the
    pattern), the masks of a file saved by PODBackgroundRemoval.save_masks or by the interface, and contrast
    enhancement settings (a dictionary of the arguments of PODBackgroundRemoval.set_contrast). Returns when the
    window is closed. Tkinter is imported only here, so the processing runs without it."""
    from .gui import main
    main(folder, pattern, masks, contrast)
