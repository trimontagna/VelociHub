"""Credits of VelociHub: its author, its context and the open-source software it uses, with their authors, licences
and links. Shown by Help > Credits in the interface (gui/credits_window.py) and written to CREDITS.md of the repository
(markdown()), so that both say the same.

Each entry of SOFTWARE: name, what VelociHub uses it for, authors, licence, links [(label, url)] and the reference
to cite (None: none).
"""

from dataclasses import dataclass

__version__ = "1.2.0"

AUTHOR = "Tristan Montagnier"
AUTHOR_EMAIL = "tristanmontagnier@gmail.com"
LICENSE = "CC BY-NC-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
ABOUT = ("VelociHub was developed by Tristan Montagnier, made with Claude (Anthropic), with input from Filippo "
         "Vaselli, Reda Lamrani and M.A. Mendez, as part of a PIV Laboratory at the von Karman Institute for Fluid "
         "Dynamics. It uses the open-source software listed below.")
ABOUT_LINKS = [("Claude (Anthropic)", "https://www.anthropic.com/claude"),
               ("von Karman Institute for Fluid Dynamics", "https://www.vki.ac.be")]
LICENSE_TEXT = ("VelociHub is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 "
                "International licence (CC BY-NC-SA 4.0), the licence of the POD-based background removal it ports: "
                "you may share and adapt it with attribution, not for commercial purposes, and under the same "
                "licence. The software listed below keeps its own licence.")

POD_BIBTEX = """@article{Mendez2017,
  title   = {POD-based background removal for particle image velocimetry},
  author  = {Mendez, M.A. and Raiola, M. and Masullo, A. and Discetti, S. and Ianiro, A. and Theunissen, R. and
             Buchlin, J.-M.},
  journal = {Experimental Thermal and Fluid Science},
  volume  = {80},
  pages   = {181--192},
  year    = {2017},
  doi     = {10.1016/j.expthermflusci.2016.08.021}
}"""


@dataclass(frozen=True)
class Software:
    name: str
    use: str
    authors: str
    license: str
    links: tuple
    reference: str = None


# Methods and programs that VelociHub integrates (ported, run or vendored)
METHODS = (
    Software("POD-based background removal (background_removal.m)",
             "The pre-processing: velocihub/preprocessing.py and its page are a Python port of the MATLAB tool.",
             "M.A. Mendez, M. Raiola, A. Masullo, S. Discetti, A. Ianiro, R. Theunissen, J.-M. Buchlin (von Karman "
             "Institute, Universidad Carlos III de Madrid, University of Bristol)",
             "CC BY-NC-SA 4.0",
             (("Paper", "https://doi.org/10.1016/j.expthermflusci.2016.08.021"),
              ("Original tool", "http://seis.bris.ac.uk/~aexrt/PIVPODPreprocessing"),
              ("Bug report (original tool)", "http://seis.bris.ac.uk/~aexrt/PIVPODPreprocessing/bugreport.html")),
             "M.A. Mendez et al., POD-based background removal for particle image velocimetry, Experimental Thermal "
             "and Fluid Science 80 (2017) 181-192."),
    Software("PaIRS-UniNa",
             "The PIV engine (cross-correlation, multigrid with image deformation), run in its own Python 3.10-3.13 "
             "process (velocihub/piv/pairs_engine.py).",
             "Gerardo Paolillo, Tommaso Astarita (University of Naples Federico II)",
             "LGPL-3.0-or-later",
             (("Website", "https://pairs.unina.it/"), ("PyPI", "https://pypi.org/project/PaIRS-UniNa/"))),
    Software("TracTrac",
             "The PTV engine (particle detection, motion model and tracking): tractrac.py is vendored in "
             "velocihub/piv/tractrac with three lines changed for numpy 2.",
             "Joris Heyman",
             "MIT",
             (("Repository", "https://github.com/jorishey1234/tractrac"),),
             "J. Heyman, TracTrac: a fast multi-object tracking algorithm for motion estimation, Computers & "
             "Geosciences 128 (2019) 11-18."),
    Software("SPICY (spicy_vki)",
             "The optional regression of the vector fields with constrained radial basis functions "
             "(velocihub/piv/regression.py).",
             "P. Sperotto, M. Ratz, M.A. Mendez (von Karman Institute)",
             "BSD-3-Clause",
             (("Repository", "https://github.com/mendezVKI/SPICY_VKI"), ("PyPI", "https://pypi.org/project/spicy-vki/"))),
)

# Python libraries VelociHub imports (and those of SPICY and of the videos)
LIBRARIES = (
    Software("NumPy", "Arrays and all the numerical work.", "Travis E. Oliphant et al., the NumPy developers",
             "BSD-3-Clause", (("Repository", "https://github.com/numpy/numpy"), ("Website", "https://numpy.org"))),
    Software("SciPy", "Image filters, interpolation and statistics.", "The SciPy developers", "BSD-3-Clause",
             (("Repository", "https://github.com/scipy/scipy"), ("Website", "https://scipy.org"))),
    Software("Matplotlib", "Every figure, the tiles of the interface and the videos.",
             "John D. Hunter, Michael Droettboom et al., the Matplotlib development team",
             "Matplotlib License (PSF-based)",
             (("Repository", "https://github.com/matplotlib/matplotlib"), ("Website", "https://matplotlib.org"))),
    Software("contourpy", "Isocontours (potential core of the jet).", "Ian Thomas", "BSD-3-Clause",
             (("Repository", "https://github.com/contourpy/contourpy"),)),
    Software("Pillow", "Reading and writing the images.", "Jeffrey A. Clark and contributors (fork of PIL by "
             "Fredrik Lundh)", "MIT-CMU", (("Repository", "https://github.com/python-pillow/Pillow"),)),
    Software("OpenCV (opencv-python)", "CLAHE, Gaussian filters, and the image processing of TracTrac.",
             "The OpenCV team", "Apache-2.0", (("Repository", "https://github.com/opencv/opencv-python"),)),
    Software("psutil", "Available memory for the POD.", "Giampaolo Rodola", "BSD-3-Clause",
             (("Repository", "https://github.com/giampaolo/psutil"),)),
    Software("imutils", "Needed by TracTrac.", "Adrian Rosebrock", "MIT",
             (("Repository", "https://github.com/jrosebr1/imutils"),)),
    Software("parse", "Needed by TracTrac.", "Richard Jones", "MIT",
             (("Repository", "https://github.com/r1chardj0n3s/parse"),)),
    Software("h5py", "Needed by TracTrac.", "Andrew Collette and contributors", "BSD-3-Clause",
             (("Repository", "https://github.com/h5py/h5py"),)),
    Software("scikit-learn", "Needed by SPICY (clustering of the RBF collocation points).",
             "The scikit-learn developers", "BSD-3-Clause",
             (("Repository", "https://github.com/scikit-learn/scikit-learn"),)),
    Software("Shapely", "Needed by SPICY.", "Sean Gillies and contributors", "BSD-3-Clause",
             (("Repository", "https://github.com/shapely/shapely"),)),
    Software("imageio-ffmpeg / FFmpeg", "MP4 videos (optional; GIF otherwise).",
             "The imageio contributors; the FFmpeg developers", "BSD-2-Clause; FFmpeg: LGPL-2.1-or-later",
             (("Repository", "https://github.com/imageio/imageio-ffmpeg"), ("FFmpeg", "https://ffmpeg.org"))),
    Software("Python and Tkinter (Tcl/Tk)", "The language and the interface.",
             "The Python Software Foundation; the Tcl/Tk developers", "PSF License; Tcl/Tk License (BSD-style)",
             (("Python", "https://www.python.org"), ("Tcl/Tk", "https://www.tcl-lang.org"))),
)

SECTIONS = (("Methods and programs integrated", METHODS), ("Python libraries", LIBRARIES))


def markdown():
    """The credits as Markdown (CREDITS.md of the repository)."""
    lines = ["# Credits", "", ABOUT.replace("Claude (Anthropic)", "[Claude](https://www.anthropic.com/claude) "
                                            "(Anthropic)"), "",
             f"Author: {AUTHOR} ({AUTHOR_EMAIL})", "", f"Licence: [{LICENSE}]({LICENSE_URL}). {LICENSE_TEXT}", ""]
    for title, entries in SECTIONS:
        lines += [f"## {title}", "", "| Software | Used for | Authors | Licence | Links |", "|---|---|---|---|---|"]
        for s in entries:
            links = ", ".join(f"[{label}]({url})" for label, url in s.links)
            lines.append(f"| {s.name} | {s.use} | {s.authors} | {s.license} | {links} |")
        lines.append("")
        cited = [s for s in entries if s.reference]
        if cited:
            lines += ["References to cite:", ""] + [f"- {s.name}: {s.reference}" for s in cited] + [""]
    lines += ["BibTeX of the POD-based background removal:", "", "```bibtex", POD_BIBTEX, "```", ""]
    return "\n".join(lines)
