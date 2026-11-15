# VelociHub

VelociHub covers the entire particle velocimetry pipeline, from pre-processing, to processing, to post-processing with various fields and quantities able to be observed, in one Graphical Interface or via Python.

Be aware that this tool is vibe coded, but validated with established PIV solutions (DynamicStudio from Dantec Dynamics). That said, bugs and incoherences may still occur, despite heavy unit and manual testing. Do not hesitate to report any issues you may find or contribute ideas to improve the software.

The documentation is also work-in-progress, with the first main objective to go over the tooltips of the software to streamline the help queues given to the user. The documentation of the repository is also work-in-progress. Keep in mind that this tool was built during a lab which lasted two weeks, past that time frame, it's a hobby project essentially.

| Part | What it does |
|---|---|
| **1. Pre-processing** | Masks, contrast enhancement and POD-based background removal of the images (Mendez et al. 2017). |
| **2. Processing (PIV / PTV)** | Cross-correlation PIV with PaIRS-UniNa, or particle tracking (PTV) with TracTrac; optional regression of the fields with SPICY. |
| **3. Post-processing** | Statistics, vorticity, turbulence, moments, gradients, displacements, quality (valid vectors, peak ratio, peak locking, convergence), PTV tracks and accelerations, images and POD modes. Figures, comparisons, videos and exports. |
| **Batch processing** | Any step on many projects of a workspace, each with its own settings (window Batch, command line or API). |
| **Jet theory (add-on)** | The potential core of a jet (velocity level or peak skewness), its profiles and centreline, and the presets of the jet measurements (`velocihub/jet`, `PostSettings(jet_theory=True)`). |
| **Archiving (add-on)** | Projects packed into ZIP/TAR archives, and the post-processing + stitching of several fields of view of an archive. |

Every setting and result is kept in a project file (`VelociHub_project.json`) next to the images, so a folder opened
again comes back as you left it. Everything also has a Python API that does exactly what the interface does, with the
same parameters:

```python
import velocihub as vh

p = vh.Project("D:/Jet/Run_30")
p.set_piv(engine="PaIRS-UniNa", windows=(128, 64, 32), overlap=50)
p.set_postprocessing(camera=vh.Camera(196.1, frame_rate_hz=150, y_shift=0.5), outlier_k=4,
                     figures=["mean_velocity", "u_rms"])
p.save()          # opens in the interface with these settings
p.run()           # pre-processing, PIV, post-processing
```

## Quick start

```bash
# 1. VelociHub (Python 3.12 or later)
git clone https://github.com/RexAttaque/VelociHub.git
cd VelociHub
pip install -e ".[all]"            # or: conda env create -f environment.yml

# 2. The PIV engine PaIRS-UniNa, in its own environment (it supports Python 3.10-3.13 only)
conda env create -f environment-pairs.yml

# 3. Run
velocihub                          # or: python run_VelociHub.py
```

Then choose the folder of your images in **Pre-processing > Images folder** and follow the pages from left to right.
The full tutorial is in [docs/installation.md](docs/installation.md).

## Documentation

| Document | Contents |
|---|---|
| [Installation and tutorial](docs/installation.md) | Install (pip, conda), the PaIRS environment, dependencies, first project step by step |
| [Pre-processing](docs/preprocessing.md) | Masks, contrast filters, POD background removal: features and every parameter |
| [Processing (PIV/PTV)](docs/processing.md) | PaIRS-UniNa, TracTrac, SPICY regression, particle diagnostic: features and every parameter |
| [Post-processing](docs/postprocessing.md) | Camera, scale space, reference quantities, recipe, figure catalogue, exports and videos |
| [Batch processing](docs/batch-processing.md) | Workspace, window Batch, `velocihub-batch`, batch API |
| [Archiving (add-on)](docs/archiving.md) | Archives of projects, post-processing and stitching of fields of view, alignment |
| [Python API](docs/api.md) | `vh.Project` (a measurement as the interface sees it: settings, save, run), batch and stitching helpers, and the objects of each step: every action of the interface from Python, with the same parameters |
| [Packaging](docs/packaging.md) | Building and installing VelociHub with pip and conda |
| [Credits](CREDITS.md) | Authors and licences of all the software VelociHub uses |

## Commands

| Command | Script in the repository | Does |
|---|---|---|
| `velocihub [folder [pattern]] [--masks file]` | `run_VelociHub.py` | The interface |
| `velocihub-batch roots... [--pattern] [--masks] [--force] [--list]` | `run_VelociHub_batch.py` | Pre-processing + PIV of measurement folders |
| `velocihub-archive [folders.txt] [--output] [--format] [--no-gui]` | `run_VelociHub_archive.py` | Archive of projects |
| `velocihub-post-archive [archive] [--setups] [--export-root] [--stitched]` | `run_VelociHub_post_archive.py` | Post-processing + stitching of an archive |
| `velocihub-post-archive-gui` | `run_VelociHub_post_archive_gui.py` | Its interface |

`python -m velocihub` also opens the interface.

## Dependencies

| Package | Needed for |
|---|---|
| numpy, scipy, matplotlib, contourpy, pillow, opencv-python, psutil | VelociHub (installed with it) |
| Tkinter | The interface (included with Python on Windows and macOS; `python3-tk` on Linux) |
| PaIRS-UniNa (Python 3.10-3.13, separate environment) | The PIV |
| imutils, parse, h5py (`pip install velocihub[ptv]`) | The PTV with TracTrac (vendored) |
| spicy_vki (`velocihub[regression]`) | The regression of the vector fields |
| imageio-ffmpeg (`velocihub[video]`) | MP4 videos (GIF without it) |

Tested on Windows 11 with Python 3.12-3.14, numpy 2.5, matplotlib 3.11, PaIRS-UniNa 0.2.12 (Python 3.13).

## Tests

`pip install velocihub[test]`, then `python -m pytest` in the repository (about a minute). The tests make their own
synthetic images of a jet (no data needed): the library, the Python API and batch, the interface (skipped without a
display) and the PIV/PTV engines installed (PaIRS-UniNa, TracTrac; skipped otherwise).

## Authors and licence

VelociHub was developed by **Tristan Montagnier** (tristanmontagnier@gmail.com), made with
[Claude](https://www.anthropic.com/claude) (Anthropic), with input from Filippo Vaselli, Reda Lamrani and M.A. Mendez, as part of a PIV
Laboratory at the [von Karman Institute for Fluid Dynamics](https://www.vki.ac.be). It uses open-source code from the
POD-based background removal of Mendez et al. (2017), PaIRS-UniNa (Paolillo & Astarita, University of Naples Federico
II), TracTrac (Heyman 2019), SPICY (Sperotto, Ratz & Mendez, VKI) and the scientific Python stack: see
[CREDITS.md](CREDITS.md), also shown in the interface under **Help > Credits**.

VelociHub is distributed under the [CC BY-NC-SA 4.0](LICENSE) licence, the licence of the POD-based background removal
it ports: share and adapt it with attribution, not for commercial purposes, under the same licence. The vendored
TracTrac keeps its MIT licence ([NOTICE.md](NOTICE.md)).

If you use VelociHub in a publication, please cite the methods it relies on (references in [CREDITS.md](CREDITS.md)).
