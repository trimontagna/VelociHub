# Part 1 — Pre-processing

The pre-processing prepares the images for the PIV/PTV with a **recipe** of three optional steps, always applied in
this order:

1. **Masks** — polygons whose pixels are set to 0 (nozzle, walls, reflections, regions without flow).
2. **Contrast enhancement** — up to four filters, in the order chosen, applied either **before the POD** (the modes
   are those of the enhanced images) or **after the background removal**.
3. **Background removal** — subtraction of the first POD modes of the training images: the part of each image that
   doesn't move with the flow (reflections, illumination, fixed particles).

The POD-based background removal is the method of M.A. Mendez, M. Raiola, A. Masullo, S. Discetti, A. Ianiro,
R. Theunissen and J.-M. Buchlin, *POD-based background removal for particle image velocimetry*, Experimental Thermal
and Fluid Science 80 (2017) 181-192. VelociHub's pre-processing is a Python port of their MATLAB tool
(`background_removal.m`, CC BY-NC-SA 4.0).

Module: `velocihub/preprocessing.py` (class `PODBackgroundRemoval`), page: `velocihub/gui/preprocessing_page.py`.

## Features

- **Training images**: every image of a folder (or those matching a pattern), at least two, any size and bit depth
  (8/16-bit grey, 32-bit float, any colour space). Every step of VelociHub (pre-processing, PIV, PTV, particle
  diagnostic, figures) reads them the same way (`velocihub/greyscale.py`): in one channel, a colour image as its
  brightness, the value V = max(R, G, B) of the HSV colour space (palette images by their colours, alpha left out,
  CMYK/YCbCr/LAB converted to RGB first), the bit depth kept. The pre-processed images are in grey levels.
- **POD of the images**: spectrum (energy of each mode), spatial modes and temporal coefficients, shown in tiles and
  browsed with the arrow keys. The correlation matrix is built in memory, or by blocks in temporary files when the
  images don't fit (no limit on the number of images).
- **Automatic number of modes** from two tolerances (ε σ, ε ψ).
- **Mask editor** on the image: draw, edit (move points, translate, rotate, scale), enable/disable, rename, import and
  export (JSON). Points may lie outside the image (the mask then reaches the edges).
- **Live preview**: the image tile shows the recipe applied, with a before/after slider.
- **Export** of the pre-processed images (same format as the originals, or PNG, BMP, TIFF) with `preprocessing.json` (recipe, source
  images, names) and `pod.npz` (spectrum and modes, restored when the project is reopened). The PIV reuses the images
  exported with the same recipe instead of processing them again.
- **New images**: the recipe applied to other images of the same size (Pre-processing > Export > New images), their
  background reconstructed with the temporal coefficients of the training image of the same index.

## Parameters

### Images

| Interface | API | Meaning |
|---|---|---|
| Images folder | `PODBackgroundRemoval(folder, pattern=None, files=None, first=1, last=None)` | Training images: all the images of the folder, or those matching `pattern` (e.g. `"*.tif"`), from `first` to `last` (1-based), or an explicit list `files`. |

### Masks

| Interface | API | Meaning |
|---|---|---|
| New / Edit / Delete | `session.add_mask(points, name=None, enabled=True)`; `session.masks` | Polygon `[[x, y], ...]` in pixels (pixel centres at integer coordinates from 1, as in MATLAB). |
| ☑ / ☐ | `Mask.enabled` | Only the enabled masks are applied. |
| Import... / Export... | `session.load_masks(file, replace=True)`, `session.save_masks(file)` | JSON file of masks, shared between the interface, scripts and batch runs. |

### Contrast enhancement

No filter ticked means no contrast enhancement. The filters run in the order of the list below (the order in
`filters=` in the API).

| Filter (interface) | API name | Parameters (interface → API, default) | What it does |
|---|---|---|---|
| Intensity capping | `"capping"` | n (cap at median + n std) → `cap_n` = 2.0 | Pixels brighter than median + n·std of the image are set to that value, so a few bright particles don't dominate the correlation (Shavit et al. 2007). |
| Local min-max normalization | `"minmax"` | window (px) → `mm_window` = 15; minimum contrast (grey levels) → `mm_contrast` = 10 | (image − local min) / max(local max − local min, minimum contrast), the extremes smoothed over the window: evens out the brightness of the particles across the image. |
| CLAHE | `"clahe"` | clip limit → `clip` = 10; tiles per side → `tiles` = 8 | Contrast Limited Adaptive Histogram Equalization over tiles (8 or 16-bit images). |
| Edge enhancement | `"edge"` | low-pass σ (px) → `edge_sigma` = 2.0; factor → `edge_factor` = 1.0 | Unsharp masking: image + factor × (image − Gaussian low-pass). Sharpens the particles (σ about their size, 1 to 3 px) and lowers slow background variations. Factor 0: no effect; 1-3 usual. |
| Applied: before POD / after background removal | `stage="pre"` / `"post"` | — | Before the POD, the modes are those of the enhanced images; after, the filters act on the images with their background removed. |

The masked pixels are excluded from every filter's statistics (no bright band along the masks).
The interface starts with **intensity capping, n = 10, after the background removal** (`DEFAULT_CONTRAST`).

### POD modes

| Interface | API | Default | Meaning |
|---|---|---|---|
| Evaluate video spectrum (Ctrl+R) | `session.evaluate_spectrum()` | — | Correlation matrix of the images and its eigenvalues (the energy of each mode). |
| Automatic threshold ε σ, ε ψ; Evaluate threshold (Ctrl+T) | `session.eps_auto_sigma`, `session.eps_auto_psi`, `session.automatic_N_modes()` | 1e-4, 1e-4 | Number of modes: the first mode i whose temporal mean \|mean ψ_i\| is below ε ψ and whose energy difference with the next one, \|λ_i − λ_i+1\| / Σλ (a fraction of the energy of the images), is below ε σ. |
| Modes to evaluate; Evaluate modes (Ctrl+M) | `session.evaluate_modes(n_modes)` | 6 | Spatial modes and temporal coefficients computed and stored. |
| Settings > Set temporary folder... | `PODBackgroundRemoval(temp_folder=...)` | images folder (interface); a new temporary folder (API) | Where the blocks of the correlation are written when the images don't fit in memory. |
| Settings > Max memory usage... | `PODBackgroundRemoval(memory_fraction=0.8)` | 0.8 | Fraction of the available memory the images may use before temporary files are used. |

### Background removal

| Interface | API | Default | Meaning |
|---|---|---|---|
| Remove the background (Ctrl+P) | `n_modes_removed > 0` | on | Subtracts the first modes from each image. |
| Modes removed | `n_modes_removed` | 6 | The first mode is close to the mean image; a few more modes remove the slowly varying background. Too many modes start to remove particles. |

### Export (Run step)

| Interface | API | Meaning |
|---|---|---|
| Export folder (`PreProcessing` by default, `...` to change, ↺ to reset) | `session.path_export`; `session.export_folder()` | Folder of the pre-processed images. |
| Image format: Same as the source / PNG / BMP / TIFF | `session.export_format = "same"` / `"png"` / `"bmp"` / `"tiff"` (`p.set_preprocessing(export_format=...)`) | Format of the images exported. Same as the source: a JPEG is compressed again, with losses. PNG holds 8 and 16 bit, BMP 8 bit only, TIFF any depth (uncompressed); a format that can't hold the depth of the images is refused. |
| Singular (Ctrl+B) | `session.export_images(folder, n_modes_removed)` or `session.piv_images(n_modes_removed)` | Writes `[project]_PreProcessing_Image_[n]` (in the format chosen), `preprocessing.json` and `pod.npz`. `piv_images` reuses an export made with the same recipe in the same format. |
| Export > New images | `session.remove_background(folder, n_modes_removed, folder=..., pattern=...)` | The recipe on other images of the same size. |
| Batch... | `velocihub.batch.preprocess_folder` | See [batch-processing.md](batch-processing.md). |

## Outputs

| File | Content |
|---|---|
| `PreProcessing/[project]_PreProcessing_Image_[n].[ext]` | Pre-processed images (grey levels, the input bit depth, the format chosen) |
| `PreProcessing/preprocessing.json` | Recipe, steps, source folder and files, format, names of the images exported |
| `PreProcessing/pod.npz` | Eigenvalues, spatial modes, temporal coefficients (restored on reopening, used by the post-processing's POD figures) |
| `VelociHub_project.json`, section `preprocessing` | The settings of the page (masks, contrast, background removal, thresholds, export folder) |

## Tips

- Evaluate a few more modes than you remove: the tiles let you check what each mode contains before choosing.
- Remove the background before capping the intensity (the default): the capping then acts on the particles only.
- The same masks file can be used for all the runs of a field of view (`velocihub-batch --masks`).
