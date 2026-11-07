# Credits

VelociHub was developed by Tristan Montagnier, made with [Claude](https://www.anthropic.com/claude) (Anthropic), with input from Filippo Vaselli, Reda Lamrani and M.A. Mendez, as part of a PIV Laboratory at the von Karman Institute for Fluid Dynamics. It uses the open-source software listed below.

Author: Tristan Montagnier (tristanmontagnier@gmail.com)

Licence: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). VelociHub is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International licence (CC BY-NC-SA 4.0), the licence of the POD-based background removal it ports: you may share and adapt it with attribution, not for commercial purposes, and under the same licence. The software listed below keeps its own licence.

## Methods and programs integrated

| Software | Used for | Authors | Licence | Links |
|---|---|---|---|---|
| POD-based background removal (background_removal.m) | The pre-processing: velocihub/preprocessing.py and its page are a Python port of the MATLAB tool. | M.A. Mendez, M. Raiola, A. Masullo, S. Discetti, A. Ianiro, R. Theunissen, J.-M. Buchlin (von Karman Institute, Universidad Carlos III de Madrid, University of Bristol) | CC BY-NC-SA 4.0 | [Paper](https://doi.org/10.1016/j.expthermflusci.2016.08.021), [Original tool](http://seis.bris.ac.uk/~aexrt/PIVPODPreprocessing), [Bug report (original tool)](http://seis.bris.ac.uk/~aexrt/PIVPODPreprocessing/bugreport.html) |
| PaIRS-UniNa | The PIV engine (cross-correlation, multigrid with image deformation), run in its own Python 3.10-3.13 process (velocihub/piv/pairs_engine.py). | Gerardo Paolillo, Tommaso Astarita (University of Naples Federico II) | LGPL-3.0-or-later | [Website](https://pairs.unina.it/), [PyPI](https://pypi.org/project/PaIRS-UniNa/) |
| TracTrac | The PTV engine (particle detection, motion model and tracking): tractrac.py is vendored in velocihub/piv/tractrac with three lines changed for numpy 2. | Joris Heyman | MIT | [Repository](https://github.com/jorishey1234/tractrac) |
| SPICY (spicy_vki) | The optional regression of the vector fields with constrained radial basis functions (velocihub/piv/regression.py). | P. Sperotto, M. Ratz, M.A. Mendez (von Karman Institute) | BSD-3-Clause | [Repository](https://github.com/mendezVKI/SPICY_VKI), [PyPI](https://pypi.org/project/spicy-vki/) |

References to cite:

- POD-based background removal (background_removal.m): M.A. Mendez et al., POD-based background removal for particle image velocimetry, Experimental Thermal and Fluid Science 80 (2017) 181-192.
- TracTrac: J. Heyman, TracTrac: a fast multi-object tracking algorithm for motion estimation, Computers & Geosciences 128 (2019) 11-18.

## Python libraries

| Software | Used for | Authors | Licence | Links |
|---|---|---|---|---|
| NumPy | Arrays and all the numerical work. | Travis E. Oliphant et al., the NumPy developers | BSD-3-Clause | [Repository](https://github.com/numpy/numpy), [Website](https://numpy.org) |
| SciPy | Image filters, interpolation and statistics. | The SciPy developers | BSD-3-Clause | [Repository](https://github.com/scipy/scipy), [Website](https://scipy.org) |
| Matplotlib | Every figure, the tiles of the interface and the videos. | John D. Hunter, Michael Droettboom et al., the Matplotlib development team | Matplotlib License (PSF-based) | [Repository](https://github.com/matplotlib/matplotlib), [Website](https://matplotlib.org) |
| contourpy | Isocontours (potential core of the jet). | Ian Thomas | BSD-3-Clause | [Repository](https://github.com/contourpy/contourpy) |
| Pillow | Reading and writing the images. | Jeffrey A. Clark and contributors (fork of PIL by Fredrik Lundh) | MIT-CMU | [Repository](https://github.com/python-pillow/Pillow) |
| OpenCV (opencv-python) | CLAHE, Gaussian filters, and the image processing of TracTrac. | The OpenCV team | Apache-2.0 | [Repository](https://github.com/opencv/opencv-python) |
| psutil | Available memory for the POD. | Giampaolo Rodola | BSD-3-Clause | [Repository](https://github.com/giampaolo/psutil) |
| imutils | Needed by TracTrac. | Adrian Rosebrock | MIT | [Repository](https://github.com/jrosebr1/imutils) |
| parse | Needed by TracTrac. | Richard Jones | MIT | [Repository](https://github.com/r1chardj0n3s/parse) |
| h5py | Needed by TracTrac. | Andrew Collette and contributors | BSD-3-Clause | [Repository](https://github.com/h5py/h5py) |
| scikit-learn | Needed by SPICY (clustering of the RBF collocation points). | The scikit-learn developers | BSD-3-Clause | [Repository](https://github.com/scikit-learn/scikit-learn) |
| Shapely | Needed by SPICY. | Sean Gillies and contributors | BSD-3-Clause | [Repository](https://github.com/shapely/shapely) |
| imageio-ffmpeg / FFmpeg | MP4 videos (optional; GIF otherwise). | The imageio contributors; the FFmpeg developers | BSD-2-Clause; FFmpeg: LGPL-2.1-or-later | [Repository](https://github.com/imageio/imageio-ffmpeg), [FFmpeg](https://ffmpeg.org) |
| Python and Tkinter (Tcl/Tk) | The language and the interface. | The Python Software Foundation; the Tcl/Tk developers | PSF License; Tcl/Tk License (BSD-style) | [Python](https://www.python.org), [Tcl/Tk](https://www.tcl-lang.org) |

BibTeX of the POD-based background removal:

```bibtex
@article{Mendez2017,
  title   = {POD-based background removal for particle image velocimetry},
  author  = {Mendez, M.A. and Raiola, M. and Masullo, A. and Discetti, S. and Ianiro, A. and Theunissen, R. and
             Buchlin, J.-M.},
  journal = {Experimental Thermal and Fluid Science},
  volume  = {80},
  pages   = {181--192},
  year    = {2017},
  doi     = {10.1016/j.expthermflusci.2016.08.021}
}
```
