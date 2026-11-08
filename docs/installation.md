# Installation and tutorial

## 1. What you need

- **Python 3.12 or later** for VelociHub (tested with 3.12, 3.13 and 3.14).
- **A second Python, 3.10 to 3.13, for PaIRS-UniNa**, the PIV engine. PaIRS-UniNa is a compiled library that only
  exists for those versions, so it runs in its own process with its own interpreter. VelociHub finds it among your
  conda environments automatically (or choose its `python.exe` in **Settings > PaIRS Python interpreter...**). If you
  only use the PTV (TracTrac), you don't need it.
- **Tkinter** for the interface. It comes with Python on Windows and macOS and with conda; on Debian/Ubuntu install
  `python3-tk`.

### Dependencies

| Package | Version tested | Required for | Installed by |
|---|---|---|---|
| numpy | 2.5 | everything | `pip install velocihub` |
| scipy | 1.18 | filters, statistics | `pip install velocihub` |
| matplotlib | 3.11 | figures, interface tiles, videos | `pip install velocihub` |
| contourpy | 1.4 | isocontours (potential core) | `pip install velocihub` |
| pillow | 12 | image files | `pip install velocihub` |
| opencv-python | 5.0 | CLAHE, filters, TracTrac | `pip install velocihub` |
| psutil | 7.2 | memory available for the POD | `pip install velocihub` |
| imutils, parse, h5py | 0.5 / 1.22 / 3.15 | PTV (TracTrac) | `pip install "velocihub[ptv]"` |
| spicy_vki | 1.1 | regression of the vector fields | `pip install "velocihub[regression]"` |
| imageio-ffmpeg | 0.4+ | MP4 videos (GIF otherwise) | `pip install "velocihub[video]"` |
| PaIRS-UniNa | 0.2.12 | PIV (separate Python 3.10-3.13) | `environment-pairs.yml` |

`velocihub[all]` installs the PTV, regression and video extras together.

## 2. Install

### Option A: pip (from the repository)

```bash
git clone https://github.com/RexAttaque/VelociHub.git
cd VelociHub
python -m pip install -e ".[all]"     # -e: editable, the code of the clone is used directly
```

Without cloning: `python -m pip install "velocihub[all] @ git+https://github.com/RexAttaque/VelociHub.git"`
(a private repository needs your GitHub credentials).

### Option B: conda

```bash
git clone https://github.com/RexAttaque/VelociHub.git
cd VelociHub
conda env create -f environment.yml   # environment "velocihub": dependencies from conda-forge + VelociHub (editable)
conda activate velocihub
```

To build a real conda package instead, see [packaging.md](packaging.md).

### The PaIRS-UniNa environment

```bash
conda env create -f environment-pairs.yml     # environment "PaIRS": Python 3.13 + PaIRS-UniNa
```

Without conda: install Python 3.13 and run `py -3.13 -m pip install PaIRS-UniNa`, then select that `python.exe` in
**Settings > PaIRS Python interpreter...** of the PIV/PTV page.

### Without installing

From a clone, with the dependencies installed: `python run_VelociHub.py` (the scripts at the root of the repository
use the package next to them).

### Check

```bash
velocihub-batch --help
python -c "import velocihub; print(velocihub.__version__)"
```

## 3. Run the interface

```bash
velocihub                       # or python run_VelociHub.py, or python -m velocihub
velocihub D:/Jet/FOV_1/Run_30   # opens the images of this folder
velocihub D:/Jet/Run_30 "*.tif" --masks D:/Jet/masks.json
```

The window has a panel **Projects** on the left (the workspace), one page per step (**Pre-processing**,
**PIV/PTV**, **Post-processing**) and a status log at the bottom. Each page has its settings on the right and figure
tiles in the middle; the arrow keys browse the images or maps of the tile under the mouse, and the tiles that show the
same series follow each other.

## 4. Tutorial: a first measurement

The example is a folder of time-resolved PIV images of a jet, `D:/Jet/Run_30`, 500 images of 1920 x 1200 px taken at
150 Hz with a field of view 196.1 mm wide.

### Step 1 — Pre-processing

1. **Images folder**: click `...` and choose `D:/Jet/Run_30`. The images are listed with their size and bit depth.
2. **Masks** (optional): **New**, then click the corners of the region to blank (the nozzle, a reflection); double-click
   to close. **Edit** moves the points, the centroid (translate) or the handle (rotate/scale). **Export...** saves
   them to reuse on other runs.
3. **POD modes**: **Evaluate video spectrum** (Ctrl+R) computes the POD of the images; the spectrum and the modes
   appear in the tiles. **Evaluate threshold** (Ctrl+T) proposes a number of modes; **Evaluate modes** (Ctrl+M)
   computes them.
4. **Background removal**: switch it on and set **Modes removed** (6 by default). The image tile has a slider to
   compare before/after.
5. **Contrast enhancement** (optional): tick filters, e.g. **Intensity capping** with n = 10, applied **after
   background removal**.
6. **Run step > Singular** (Ctrl+B) exports the images to `Run_30/PreProcessing` as
   `Run_30_PreProcessing_Image_001.tif`... with `preprocessing.json` (the recipe) and `pod.npz` (the modes).

You can skip step 6: the PIV exports the images itself when it needs them.

### Step 2 — PIV or PTV

1. Open the **PIV/PTV** page. **Images** shows which images the PIV will use (pre-processed by default).
2. **PIV/PTV software**: **PaIRS-UniNa (PIV)** with windows `128 64 32` px and 50 % overlap (the defaults), or
   **TracTrac (PTV)**.
3. Check the seeding: the three yellow windows on the image are counted in **Particles in the windows** (about
   10 particles per window are needed). Drag them on the image; the mouse wheel resizes them.
4. **Run on the pair shown** (F6) computes only the pair shown, to check the settings. The image tile menu
   switches between the image and the displacement maps (|d|, d_x, d_y): a grid for the PIV, the particles for the PTV.
5. **Run step > Singular** (F5) computes all the pairs into `Run_30/PIV/vectors.npz` (`Run_30/PTV` for TracTrac).
   The quality of each pair is plotted below the image. **Stop** keeps the pairs computed; running again continues.

### Step 3 — Post-processing

1. Open the **Post-processing** page: it loads the results of the PIV page.
2. **Camera and acquisition**: field of view width `196.1` mm, origin of y `0.5` (y = 0 on the jet axis), frame rate `150` Hz (offsets 0 for a single field of
   view). **Scale space**: **Physical (mm, s)**.
3. Set the recipe: **Statistical outlier filtering** 4 σ, **Reference quantities** (u_ref, t_flow, L_ref = 50 mm for
   a 50 mm nozzle), **Jet theory (add-on)** for the potential core of a jet...
4. Each of the four tiles has a figure menu (mean velocity, RMS, vorticity, turbulence, potential core, quality...),
   a comparison menu (wiper between two maps), a colour map, a colour range and an optional isocontour.
5. **Export > Quantities...** chooses the figures (and videos); **Singular** (Ctrl+E) writes them to
   `Run_30/PostProcessing` with `metrics.json` (key numbers) and `postprocessing.json` (the recipe).

### Step 4 — More runs

Add the root folder of all your measurements in the panel **Projects** (right-click > **Add a root folder (detect its projects)...**): every
folder with images or results becomes a project of the workspace. **Batch...** in each page runs that step on the
projects you tick, each with its own settings ([batch-processing.md](batch-processing.md)). For several fields of
view, see [archiving.md](archiving.md).

## 5. Where the files go

```
Run_30/
  *.tif                         your images (never modified)
  VelociHub_project.json        settings and results of the three pages (restored when reopened)
  PreProcessing/                pre-processed images, preprocessing.json, pod.npz
  PIV/                          vectors.npz (all the vector fields, in pixels), settings.json (PaIRS)
  PTV/                          the same for TracTrac, with tracks.npz
  PostProcessing/
    PaIRS/                      PNG figures, videos, metrics.json, postprocessing.json of the PaIRS results
    TracTrac/                   the same for the TracTrac results
```

Exported files are named `[project]_[task]_[element]_[number].[format]`; the project is the folder name unless you set
one in **File > Project name...**. Projects of the previous names of the tool (`PPP_project.json`,
`POD_PAIRS_project.json`) are still read, then saved as `VelociHub_project.json`; the preferences are in
`~/.velocihub.json`.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| "PaIRS-UniNa not found" | Create the PaIRS environment, or choose its `python.exe` in Settings > PaIRS Python interpreter... |
| "TracTrac needs imutils, parse..." | `pip install "velocihub[ptv]"` in the environment of VelociHub |
| "SPICY is not installed" | `pip install "velocihub[regression]"` |
| Videos are GIF, not MP4 | `pip install "velocihub[video]"` (or put ffmpeg on the PATH) |
| Not enough memory for the POD | Images are processed through temporary files automatically (Settings > Temporary files, Max memory usage...) |
| Colour images | Read in one channel by every step: their brightness max(R, G, B) (HSV value) |
