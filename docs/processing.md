# Part 2 — Processing (PIV / PTV)

The processing computes the displacement of the particles between the two images of each pair, **in pixels**, with a
PIV/PTV *engine*:

- **PaIRS-UniNa (PIV)** — cross-correlation, multigrid passes with image deformation (G. Paolillo, T. Astarita,
  University of Naples Federico II). Runs in its own Python 3.10-3.13 process.
- **TracTrac (PTV)** — particle detection, motion model and tracking over the image sequence (J. Heyman 2019);
  the particles linked in each pair are also binned on a grid, so the PTV gives vector fields like the PIV, plus the
  tracks of the particles. Vendored in `velocihub/piv/tractrac`.

An optional **regression** (SPICY, constrained radial basis functions, VKI) can follow either engine, to fill the
gaps of invalid vectors and smooth the noise.

Modules: `velocihub/piv/` (`base.py`: `PIVRun`, `Camera`, engines; `pairs_engine.py`; `ptv_engine.py`;
`regression.py`; `particles.py`), page: `velocihub/gui/piv_page.py`. Other engines can be added: subclass
`PIVEngine`, decorate it with `@register` (see `base.py`).

## Features

- **Images used**: those exported by the pre-processing with the current recipe (exported first if needed, the POD
  modes evaluated if missing), the original images when the recipe is empty, or any other folder (used as it is, with
  the masks of its `preprocessing.json`).
- **Pairs**: each image with the next one (k, k+1), all the pairs.
- **Background run**: the PIV runs in another process; the interface stays usable. The pairs are saved by chunks of
  100: **Stop** keeps them and the next run continues from there (also after a crash).
- **Live view**: the pair just computed is shown while the PIV runs (unless you are browsing other pairs).
- **Run on the pair shown** (F6): computes only the pair on screen (with 10 frames around it for TracTrac's motion
  model), to tune the settings in seconds.
- **Image tile views** (menu in the tile header): *Image of the pair* (with the interrogation windows and the
  vectors × 3), *Displacement |d|*, *d_x*, *d_y*. For the PIV, the map on its grid (arrows on |d|); for the PTV, the
  tracked particles themselves, coloured by their displacement between the two frames (the grid cells while the PTV
  runs, before its tracks are saved). Colour limits are those of all the pairs (99th percentile).
- **Particles in the windows**: a seeding diagnostic independent of the engine. Three nested windows (64, 32, 16 px by
  default) on the image, moved by click/drag, resized with the mouse wheel; the largest is magnified with the
  particles circled and counted in each window (about **10 particles per window** are needed).
- **Quality of the pairs**: valid vectors (%) and median correlation peak ratio per pair (PIV), or valid cells (%),
  particles tracked per pair and the histogram of track lengths (PTV); click a point to show that pair.
- **Masks**: vectors whose window centre is masked get status 2 (masked) and are excluded everywhere.

## Parameters

### Images and run (side panel)

| Interface | API | Meaning |
|---|---|---|
| Images folder (pre-processed by default, `...` another folder, ↺ back) | `run_piv(images, ...)` — list of paths in time order | The images the PIV reads. |
| PIV/PTV software | `engine="PaIRS-UniNa"` or `"TracTrac"` | The engine. |
| Run step > results folder (`PIV`, or `PTV` for TracTrac) | `output=` | Folder of `vectors.npz`, `settings.json` (and `tracks.npz`). |
| Singular (F5) / Stop | `run_piv(...)`, `PIVRun(...).run()` | All the pairs. |
| Run on the pair shown (F6) | `PIVRun(images, pairs=[(k, k + 1)], ...)` | One pair. |
| Settings > PaIRS Python interpreter... | `get_engine("PaIRS-UniNa", python="C:/.../envs/PaIRS/python.exe")` | Interpreter with PaIRS-UniNa (found automatically among the conda environments otherwise). |

### PaIRS-UniNa (PIV)

| Interface | API key | Default | Meaning |
|---|---|---|---|
| Window sizes (px) | `windows` | `(128, 64, 32)` | Interrogation windows of the multigrid passes, first to final (decreasing, at least 4 px). The final window sets the resolution. |
| Overlap (%) | `overlap` | 50 | Overlap of neighbouring windows: vector spacing = final window × (1 − overlap); 32 px at 50 % gives a vector every 16 px. |
| Final iterations | `final_iterations` | 2 | Iterations of the final pass, the image deformed by the previous displacement. |
| Min. window mean (grey levels) | `min_mean` | 0.01 | Darker windows are not correlated. PaIRS's own default (2) rejects almost every window of dark images; 0.01 only rejects windows without particles (masked areas). |
| Min. window std (grey levels) | `min_std` | 0.05 | Windows with less contrast are not correlated (PaIRS default 3). |
| Subtract the mean image | `subtract_mean` | False | Subtracts the mean of about 200 images from each image. Not needed after the POD background removal. |

### TracTrac (PTV)

| Interface | API key | Default | Meaning |
|---|---|---|---|
| Grid spacing / bin size (px) | `spacing` | 16 | The displacements of the particles of each pair are averaged in cells of this size centred on the grid points (16 px matches PaIRS with 32 px windows at 50 %, so PIV and PTV compare point by point). |
| Min. particles per cell | `min_particles` | 1 | A cell with fewer particles in a pair is invalid in that pair. |
| Median test threshold (0: none) | `median_test` | 2.0 | Normalized median test of each cell against its 8 neighbours (Westerweel & Scarano 2005, noise 0.1 px). |
| Particle size (px) | `particle_size` | 3.0 | Size of the detection kernel (TracTrac `peak_conv_size`). 3 px detects fewer but cleaner particles than 2 px. |
| Detection kernel | `kernel` | `"dog"` | Filter before detection: difference of Gaussians `dog`, Laplacian of Gaussian `log`, `none`, or Shi-Tomasi corners `shi-tomasi`. |
| Automatic detection threshold | `auto_threshold` | True | Threshold = mean + ½ std of the filtered image (not with shi-tomasi). |
| Manual detection threshold | `threshold` | 0.02 | Threshold of the filtered images (grey levels / 256 or 65536) without the automatic one. |
| Min. distance between particles (px) | `peak_neigh` | 1 | Closer particles are one particle. |
| Subpixel position | `subpixel` | `"gaussian"` | Fit of the particle position: `gaussian` or `quadratic`. |
| Dark particles | `dark` | False | Dark particles on a bright background. |
| Motion model | `motion` | True | Predicts each particle's displacement from its neighbours in the previous frames (links particles that move further than their spacing). |
| Motion model neighbours | `motion_av` | 10 | Particles averaged by the motion model. |
| Motion model iterations | `motion_it` | 0 | Iterations of the motion model per frame (0: one prediction). |
| Training passes (even) | `vid_loop` | 2 | Passes back and forth over the sequence to train the motion model before the final forward pass (whose tracks are kept). Each pass takes as long as the tracking. |
| Outlier filter | `filter` | True | Rejects particles whose displacement departs from the motion model. |
| Outlier threshold (std) | `filter_std` | 1.5 | Rejection beyond this many standard deviations of the motion-model error. |
| Outlier threshold memory | `filter_time` | 1.0 | Weight of previous frames in the outlier threshold. |
| Subtract a moving background | `background` | False | TracTrac's own background (not needed on pre-processed images). |
| Background adaptation speed | `background_speed` | 0.01 | Update speed of that background (0-1). |
| Median filter / size (px) | `median_filter` / `median_size` | False / 3 | Median filter of the images before detection (3 or 5 px). |

### Regression (SPICY, optional)

| Interface | API (`regression={"method": "spicy", "settings": {...}}`) | Default | Meaning |
|---|---|---|---|
| Regression: None / SPICY | `method` | `"none"` | Fit each displacement field after the PIV/PTV. |
| RBF basis | `basis` | `"gauss"` | Gaussian RBFs or `c4` (compact support: faster, slightly less smooth). |
| Points per RBF (per level) | `n_k` | `(4,)` | Vectors each RBF covers, one number per level (e.g. `(4, 10)`: fine + coarse). Fewer: closer fit, slower. |
| Collocation | `method` (of the settings) | `"clustering"` | Where RBFs are placed: k-means `clustering`, Halton `semirandom`, or `regular` grid. |
| RBF radius min / max (image width) | `r_min` / `r_max` | 0.01 / 0.3 | Smallest/largest RBF radius as a fraction of the image width. |
| RBF value at the nearest RBF | `eps_l` | 0.7 | Value of an RBF at its nearest neighbour (0-1), which sets its radius. |
| Divergence penalty (0: none) | `alpha_div` | 0 | Penalty on the 2D divergence. Keep 0 for an axisymmetric jet in a plane through its axis (there du/dx + dv/dy = −v/r). |
| Harmonic basis functions | `n_hb` | 0 | Sines and cosines added for global trends. |
| Condition number | `k_cond` | 1e12 | Regularization of the fit matrix. |

`vectors.npz` then holds the regressed fields, plus the raw ones (`raw_u_px`, `raw_v_px`, `raw_info`) and the
residual of each fit. The quality figures keep using the raw PIV/PTV validity.

### Particles in the windows (diagnostic only)

| Interface | API (`count_particles(image, center, sizes, snr, diameter)`) | Default | Meaning |
|---|---|---|---|
| Particle S/N threshold | `snr` | 4.0 | Minimum height of a particle above the background, in units of the noise (3-5 usual). |
| Particle diameter (px) | `diameter` | 3.0 | Particle image size (2-4 px usual): the Gaussian matched filter has σ = diameter / 4. |
| Windows (mouse wheel, drag) | `sizes`, `center` | 64, 32, 16 px at the image centre | Nested windows where particles are counted. |

## Outputs

| File | Content |
|---|---|
| `PIV/vectors.npz` (`PTV/` for TracTrac) | `x_px`, `y_px` (positions, px from the top-left corner), `u_px`, `v_px` (displacements in px, **v positive upwards**), `peak_ratio` (PIV) or `ptv_particles` (PTV), `info` (1 = valid), `status` (0 flow, 1 no particles, 2 masked), `settings` (JSON: pairs, image size, engine settings, pre-processing) |
| `PIV/settings.json` | Every setting, the run time (`run_time_s`, `seconds_per_pair`, resumed pairs) |
| `PTV/tracks.npz` | TracTrac's last pass: `frame`, `id`, `x`, `y` (px), `u`, `v` (px/frame), `ax`, `ay` (px/frame²) |
| `VelociHub_project.json`, section `piv` | Engine, settings, results summary, window position and size, tile view |

The results stay in pixels: the camera (field of view, frame rate) is applied by the post-processing, so the same
results can be shown in the sensor or the physical space.
