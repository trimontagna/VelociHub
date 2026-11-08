# Part 3 — Post-processing

The post-processing turns the vector fields of the PIV/PTV (in pixels) into statistics, analyses and figures. All the
analyses are computed; the **recipe** (`PostSettings`) only sets how: the outlier filtering, the data reconstruction,
the reference quantities, the rolling statistics, the peak locking, the jet theory add-on and the POD modes shown.

Modules: `velocihub/fields.py` (`load_vectors`, `VectorFields`, `FieldSettings`, `FieldAnalysis`),
`velocihub/postprocessing.py` (`PostProcessing`, `PostSettings`, the figures), `velocihub/lagrangian.py` (PTV),
`velocihub/imaging.py` (images and POD); page: `velocihub/gui/postprocessing_page.py`.

## Features

- **Inputs**: the results of the Velocimetry page (`PIV/vectors.npz`, or `PTV/`), any other `vectors.npz`, or a folder of
  DynamicStudio CSV exports.
- **Two scale spaces**: *Sensor* — the fields as computed: positions in image pixels, displacements in px, maps
  numbered by pair; *Physical* — converted with the camera settings: positions in mm (or over L_ref), velocities in
  m/s (or over u_ref), times in s (or in flow times t_flow). Coordinates: x to the right and y upwards from the origin
  of the camera (the bottom-left corner of the images by default; *Origin of x / y* moves it by fractions of the field
  of view), plus the field-of-view offsets. The sensor space has its origin at the bottom-left corner of the images.
- **Four figure tiles**, each with a compact header: a figure menu (the button shows the symbol of the quantity, as on
  the figure labels in plain text: Mean u, RMS u, Skew. u, Med. S/N, τ, Tu, Lx u...; its name in full in the menu and
  in the tip), a *Compare* menu (a second map on
  the same axes with a draggable wiper; `vs` and its symbol), a colour map list, *Range* (the colour range), *Contour*
  (an isocontour: checkbox + level slider), a map slider for instantaneous figures and a *Video* button. Tiles showing
  maps follow each other when browsing. The figures of an add-on turned off aren't drawn (their message instead).
- **Cursor** on the maps of the tiles: two lines through the point under the mouse, its position and value in a box
  at the top left of the map, named as its axes and its colour bar name them (x = 48 mm, |u| = 0.101 m/s; 3
  significant digits), the value marked by a dashed line on the colour bar (with a wiper: the map on the
  side of the point). A left click freezes it, another one releases it; off while the zoom or the pan is on.
- **Statistical outlier filtering** (against the statistics of the maps averaged: all of them, or the window of the
  rolling statistics), **sample-and-hold reconstruction** of invalid cells, **rolling statistics**.
- **Jet theory add-on** (`velocihub/jet/`, off by default): potential core detection (velocity level or peak
  skewness), with profiles and centreline plots. The rest of the post-processing makes no assumption on the flow.
- **Quality of a PIV** (menu PIV, for the results of a PIV only, greyed out for those of a PTV): valid vectors,
  correlation peak ratio maps and histograms, peak locking. The convergence of the statistics, for any results. A PTV
  has no valid-vector percentage: the valid cells of its binning on the grid aren't a reliable quality (its quality is
  in the menu PTV: particles per cell, particles tracked, track lengths).
- **PTV**: tracks, particles per cell and per frame, track lengths, material acceleration, PTV vs PIV profiles.
- **Images and POD**: raw and pre-processed images at the positions of the maps (comparable with a wiper), POD modes
  and spectrum of the pre-processing.
- **Exports**: PNG figures (LaTeX labels with Computer Modern math), videos (GIF or MP4) of instantaneous figures,
  `metrics.json` (key numbers), `postprocessing.json` (the recipe); a **Quantities...** window chooses what to export.

## Figure catalogue

Keys are the names used by the API (`PostProcessing.plot(key)`, `export(figures=[...])`). ★ = instantaneous (one per
map: browsed, exported for chosen maps, available as video). With rolling statistics, every figure has one per map.
In the menus of the tiles, the groups Images and POD are in the submenu **Pre-Processing**, the jet theory in the
submenu **Add-ons** (greyed out while the add-on is off).

| Group | Figures (key — name) |
|---|---|
| Velocity | `velocity` ★ \|u\|; `u` ★; `v` ★; `mean_magnitude` mean \|u\|; `mean_velocity` mean u; `mean_v` mean v; `u_rms`; `v_rms`; `skewness_u`; `kurtosis_u`; `skewness_v`; `kurtosis_v` |
| Vorticity | `vorticity` ★ ω; `mean_vorticity`; `mean_vorticity_abs`; `vorticity_rms` |
| Turbulence | `turbulence_intensity` Tu (RMS based); `turbulence_intensity_map` ★ Tu (fluctuation based); `reynolds_stress` τ; `reynolds_stress_abs` \|τ\|; `tke` k; `integral_time` T_u; `integral_length_x` L_x^u; `integral_length_y` L_y^u; `integral_length_vx` L_x^v; `integral_length_vy` L_y^v; `ke_time` k/ε; `ke_length` k^1.5/ε (below) |
| Gradients | `velocity_gradient` ∂ū/∂y (and `_abs`); `axial_gradient` ∂ū/∂x (and `_abs`); `ti_gradient` ∂Tu/∂y (and `_abs`); `reynolds_gradient` ∂τ/∂y (and `_abs`); `vorticity_gradient` ∂ω̄/∂y (and `_abs`) |
| Jet theory (add-on: `jet_theory=True`) | `core_map` u/u_ref with the core boundary and centreline; `core_profiles` axial velocity profiles; `core_centreline` velocity along the centreline |
| Displacement | `displacement` ★ \|d\|; `displacement_x` ★; `displacement_y` ★; `mean_displacement`; `mean_displacement_x`; `mean_displacement_y` |
| PIV (results of a PIV only) | `valid` valid vectors (%); `peak_ratio_map` ★; `peak_ratio` median peak ratio; `peak_ratio_histogram`; `peak_locking` |
| PTV | `ptv_tracks` ★; `ptv_particles` particles per cell; `ptv_yield` particles tracked per frame; `ptv_track_length`; `mean_acceleration` (and `_x`, `_y`); `ptv_vs_piv` |
| Images | `raw_image` ★; `piv_image` ★ (pre-processed) |
| Modal decomposition (POD) | `pod_mode` φ_n; `pod_spectrum` spectrum and temporal coefficient |
| (root) | `convergence` |

Every figure states in its title the reference quantities it uses.

**Turbulent kinetic energy and integral scales** (`velocihub/turbulence.py`):

- `tke`: k = (σ_u² + σ_v²) / 2, from the two components measured (the out-of-plane one isn't: with w'² ≈ (u'² + v'²) / 2,
  the three-component estimate would be 1.5 times larger). Over u_ref² with the reference velocity.
- `integral_time`: T_u = ∫₀^τ₀ ρ_uu(τ) dτ at each grid point, ρ_uu the temporal autocorrelation of u' over the
  successive maps (the gaps left out), integrated to its first zero crossing τ₀. In ms, or in flow times. It needs a
  time series: at least 20 maps, evenly spaced in time, sampled faster than the flow decorrelates.
- `integral_length_x`, `integral_length_y`: L^u = ∫₀^r₀ ρ_uu(r) dr, ρ_uu the spatial autocorrelation coefficient of u'
  along x (longitudinal) or y (lateral), over the maps, integrated to its first zero crossing towards both directions
  (their mean). In mm, or over L_ref. Time-resolved data isn't needed. `integral_length_vx`, `integral_length_vy`: the
  same for v' (lateral along x, longitudinal along y).
- `ke_time`, `ke_length`: the scales of the k-ε model, T = k/ε and L = k^{3/2}/ε, ε the dissipation rate estimated
  from the in-plane gradients of the fluctuations (the planar direct estimate, local isotropy for the gradients not
  measured): ε = ν[2<(∂u'/∂x)²> + 2<(∂v'/∂y)²> + 3<(∂u'/∂y)²> + 3<(∂v'/∂x)²> + 2<∂u'/∂y ∂v'/∂x>], with the kinematic
  viscosity ν of the reference quantities. Physical space only. ε is underestimated (the scales overestimated) where
  the vector spacing is coarser than a few Kolmogorov scales: the gradients of the smallest eddies aren't resolved.
- A scale is left blank where its autocorrelation doesn't cross zero (a scale longer than the record, or than the
  field of view in that direction) or crosses it within 2 lags (not resolved by the sampling or the vector spacing).
  The medians are in `metrics.json` (physical space).

## Parameters

### Vector fields

| Interface | API | Meaning |
|---|---|---|
| Vector fields folder (the PIV/PTV results by default) | `load_vectors(path, camera=None, pixels=False)` | `vectors.npz`, its folder, or a folder of DynamicStudio CSV files (`load_csv_folder(folder, dt, scale, frame_rate_hz=None)`: the time of each map from the file names, e.g. `Run.000001-6.6672.csv` = 6.6672 ms, else map number / frame rate, else map number × dt for time-resolved PIV; positions as exported). |

### Camera and acquisition

| Interface | API (`Camera(...)`) | Default | Meaning |
|---|---|---|---|
| Field of view width (mm) | `fov_width_mm` | — (required for the physical space) | Width of the field of view; horizontal pixel pitch (mm/px) = width / image width in pixels. |
| Field of view height (mm) | `fov_height_mm` | empty (square pixels) | Vertical pixel pitch = height / image height in pixels; empty: the horizontal pitch. |
| Origin of x (fraction of the width) | `x_shift` | 0 | x = 0 at this fraction of the width from the left edge (0.5: the middle). |
| Origin of y (fraction of the height) | `y_shift` | 0 | y = 0 at this fraction of the height from the bottom edge (0.5: the middle, the jet axis of the jet measurements). Projects of the previous versions: 0.5. |
| Horiz. offset of the field of view (mm) | `x_offset_mm` | 0 | Position of the origin of x in the global axes: x global = x + offset. |
| Vert. offset of the field of view (mm) | `y_offset_mm` | 0 | Position of the origin of y in the global axes: y global = y + offset. |
| Frame rate (Hz) | `frame_rate_hz` | 1 (times in frames) | dt = 1 / frame rate between the two images of a pair. |

### Scale space

| Interface | API | Meaning |
|---|---|---|
| Sensor (px, pairs) | `load_vectors(path)` (no camera) | Pixels, displacements, pair numbers; no reference quantities. |
| Physical (mm, s) | `load_vectors(path, camera=Camera(...))` | mm, m/s, s; reference quantities available. |

### Recipe (`PostSettings`)

| Box / interface label | API field | Default | Meaning |
|---|---|---|---|
| Statistical outlier filtering — Drop cell if beyond (σ) | `outlier_k` | 4.0 | Vectors further than k standard deviations from the mean of the other vectors of their grid point are removed (0: none): those of the maps averaged, the window of the rolling statistics with them. The vector is left out of the mean and the standard deviation it is compared with (else none could be removed from fewer than 18 maps at 4 σ). |
| Data Reconstruction — Invalid cells decay (# of maps) | `hold_maps` | 0 | Instantaneous maps: an invalid cell in the flow shows its most recent valid value from at most this many maps back (sample and hold). Statistics unchanged. |
| Reference quantities — Ref. velocity (m/s) toggle + value | `use_u_ref`, `u_ref` | off, None (= the highest mean speed, the norm of the mean velocity: any flow direction) | Velocities shown over u_ref. Turbulence intensity and default flow time always use u_ref. |
| Reference quantities — Flow time (ms) toggle + value | `use_t_flow`, `t_flow` (s in the API) | on, None (= L_FoV / u_ref) | Times shown in flow times. |
| Reference quantities — Ref. length L_ref (mm) | `l_ref` | None | Positions shown as x / L_ref, y / L_ref (e.g. the nozzle diameter). |
| Reference quantities — Kin. viscosity ν (m²/s) | `nu` | 1.5e-5 (air) | Kinematic viscosity of the fluid, for the dissipation of the k-ε scales (water: 1.0e-6 m²/s). |
| Rolling statistics — toggle + interval | `rolling`, `rolling_maps` | off, 100 | Statistics of each map over the maps up to it: every figure gets one per map (browsable, exportable, videos). |
| Peak locking — Bins per pixel (odd) | `peak_locking_bins` | 21 | Bins per pixel of the \|d_x\|, \|d_y\| histograms; quality = 1 − C compares the histogram with its 1 px moving average. |
| Add-ons — Jet theory add-on (toggle) | `jet_theory` | off | The potential core, its figures and its numbers in metrics.json (`velocihub/jet/theory.py`). In the box Add-ons below Miscellaneous; its settings below are shown only when it is on. |
| — detection | `core_mode` | `"level"` | `level`: the core is where mean u ≥ core_level % of u_ref; `skewness`: the core ends where the peak skewness of the profiles (their third moment about their maximum, not about their centroid) nears 0. |
| — Core: mean u ≥ (% of u_ref) | `core_level` | 95 | Level of the isocontour (velocity level detection). |
| — Skewness ≈ 0 below (% of max) | `skew_zero` | 20 | A profile is symmetric when \|skewness\| ≤ this % of the highest. |
| — Skewness change below (% of prev.) | `skew_change` | 10 | ...and its skewness changes by at most this % from the previous profile. |
| — Profiles for detection | `skew_profiles` | 40 | Profiles analysed by the skewness detection. |
| — Profiles of mean u/u_ref (stations) | `n_stations` | 6 | Profiles shown in `core_profiles`. |
| — First profile offset (columns) | `first_offset` | 0 | Columns between the exit and the first profile (clear of the nozzle). |
| — Ref. velocity (m/s) | `core_u_ref` | None (= the reference velocity) | u_ref of the jet theory when the Reference u_ref is off. |
| POD — mode range slider, Sum | `pod_mode`, `pod_mode_last`, `pod_combine` | 1, None, False | POD modes exported by `pod_mode` (one PNG each), or their sum weighted by each map's temporal coefficients (the background removed from that image). |

### Tile options (saved per figure in the project, used by the exports)

| Interface | API (`export(...)` arguments) | Meaning |
|---|---|---|
| Colour range... | `clims={key: (vmin, vmax)}` | Fixed colour range of a figure. |
| Colour map list | `cmaps={key: "viridis"}` | Colour map of a figure ("Default" keeps the figure's own). |
| Isocontour + level | `contours={key: level}` | An isocontour drawn on the map. |
| Compare: ... + wiper | `compares={key: (other_key, fraction)}` | Second map on the other side of a wiper. |

### Export

| Interface | API | Meaning |
|---|---|---|
| Export folder (`PostProcessing/[pipeline]`: `PostProcessing/PaIRS`, `PostProcessing/TracTrac`; `...`, ↺) | `post.export(folder, ...)` | Where the files go: each pipeline of a measurement in its own subfolder (the default folders of the previous versions, `PostProcessing` and `PostProcessing_PTV`, give it). |
| Quantities... — figures ticked | `figures=[keys]` (None: all) | Figures exported. |
| Quantities... — map shown / range of maps, first, last, every N | `index=`, `maps=[indices]` (negative from the end) | Instantaneous figures of one map, or of several maps (one subfolder per figure). |
| Quantities... — Video, fps, format | `videos=[keys]`, `fps=`, `video_maps=`, `video_format="gif"/"mp4"` | Videos of instantaneous figures. |
| Per-quantity overrides | `video_options={key: {...}}`, `image_options={key: [maps]}` | Maps/fps/format of one figure's video or images. |
| Singular (Ctrl+E) | `post.export(...)` | Writes the files. |
| Video... (tile) | `post.export_video(key, folder, step=1, fps=10, maps=None, fmt="gif")` | Video of exactly what a tile shows. |
| Batch... | `velocihub.batch.postprocess_folder(s)` | See [batch-processing.md](batch-processing.md). |

### Statistics

The standard deviations (u_rms, v_rms, vorticity rms) and the Reynolds shear stress are sample statistics (N − 1).
The turbulence intensity is the 2D form Tu = 100 √((σu² + σv²) / 2) / u_ref. In the sensor space the flow time is in
frames: L_FoV / u_ref pairs, each pair spanning frame_step frames.

## Outputs

| File | Content |
|---|---|
| `[project]_PostProcessing_[Figure].png` | Statistics figures (e.g. `Run_30_PostProcessing_MeanU.png`) |
| `[project]_PostProcessing_[Figure]_[map].png` or `[project]_PostProcessing_[Figure]/` | Instantaneous figures of one or several maps |
| `[project]_PostProcessing_[Figure].gif` / `.mp4` | Videos |
| `metrics.json` | Key numbers: valid vectors (PIV), peak ratio, reference quantities, potential core, peak locking, convergence, integral scales (medians), PTV metrics |
| `postprocessing.json` | The recipe and the camera used |
