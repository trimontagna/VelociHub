# Part 3 — Post-processing

The post-processing turns the vector fields of the PIV/PTV (in pixels) into statistics, analyses and figures. All the
analyses are computed; the **recipe** (`PostSettings`) only sets how: the outlier filtering, the data reconstruction,
the reference quantities, the rolling statistics, the peak locking, the jet potential core and the POD modes shown.

Modules: `velocihub/fields.py` (`load_vectors`, `VectorFields`, `FieldSettings`, `FieldAnalysis`),
`velocihub/postprocessing.py` (`PostProcessing`, `PostSettings`, the figures), `velocihub/lagrangian.py` (PTV),
`velocihub/imaging.py` (images and POD); page: `velocihub/gui/postprocessing_page.py`.

## Features

- **Inputs**: the results of the PIV/PTV page (`PIV/vectors.npz`, or `PTV/`), any other `vectors.npz`, or a folder of
  DynamicStudio CSV exports.
- **Two scale spaces**: *Sensor* — the fields as computed: positions in image pixels, displacements in px, maps
  numbered by pair; *Physical* — converted with the camera settings: positions in mm (or over L_ref), velocities in
  m/s (or over u_ref), times in s (or in flow times t_flow). Coordinates: x along the jet from the left edge (+ the
  field-of-view offset), y = 0 at the middle of the image height, upwards.
- **Four figure tiles**, each with: a figure menu (groups below), a *Compare* menu (a second map on the same axes with
  a draggable wiper), a colour map list, a colour range dialog, an isocontour (checkbox + level slider), a map slider
  for instantaneous figures and a *Video...* button. Tiles showing maps follow each other when browsing.
- **Statistical outlier filtering**, **sample-and-hold reconstruction** of invalid cells, **rolling statistics**.
- **Jet potential core** detection (velocity level or skewness), with profiles and centreline plots.
- **Quality**: valid vectors, correlation peak ratio maps and histograms, peak locking, convergence of the statistics.
- **PTV**: tracks, particles per cell and per frame, track lengths, material acceleration, PTV vs PIV profiles.
- **Images and POD**: raw and pre-processed images at the positions of the maps (comparable with a wiper), POD modes
  and spectrum of the pre-processing.
- **Exports**: PNG figures (LaTeX labels with Computer Modern math), videos (GIF or MP4) of instantaneous figures,
  `metrics.json` (key numbers), `postprocessing.json` (the recipe); a **Quantities...** window chooses what to export.

## Figure catalogue

Keys are the names used by the API (`PostProcessing.plot(key)`, `export(figures=[...])`). ★ = instantaneous (one per
map: browsed, exported for chosen maps, available as video). With rolling statistics, every figure has one per map.

| Group | Figures (key — name) |
|---|---|
| Velocity | `velocity` ★ \|u\|; `u` ★; `v` ★; `mean_magnitude` mean \|u\|; `mean_velocity` mean u; `mean_v` mean v; `u_rms`; `v_rms` |
| Vorticity | `vorticity` ★ ω; `mean_vorticity`; `mean_vorticity_abs`; `vorticity_rms` |
| Turbulence | `turbulence_intensity` Tu (RMS based); `turbulence_intensity_map` ★ Tu (fluctuation based); `reynolds_stress` τ; `reynolds_stress_abs` \|τ\| |
| Statistical moments | `skewness_u`; `kurtosis_u`; `skewness_v`; `kurtosis_v` |
| Gradients | `velocity_gradient` ∂ū/∂y (and `_abs`); `axial_gradient` ∂ū/∂x (and `_abs`); `ti_gradient` ∂Tu/∂y (and `_abs`); `reynolds_gradient` ∂τ/∂y (and `_abs`); `vorticity_gradient` ∂ω̄/∂y (and `_abs`) |
| Jet theory | `core_map` u/u_ref with the core boundary and centreline; `core_profiles` axial velocity profiles; `core_centreline` velocity along the centreline |
| Displacement | `displacement` ★ \|d\|; `displacement_x` ★; `displacement_y` ★; `mean_displacement`; `mean_displacement_x`; `mean_displacement_y` |
| Quality | `valid` valid vectors (%); `peak_ratio_map` ★; `peak_ratio` median peak ratio; `peak_ratio_histogram` |
| PTV | `ptv_tracks` ★; `ptv_particles` particles per cell; `ptv_yield` particles tracked per frame; `ptv_track_length`; `mean_acceleration` (and `_x`, `_y`); `ptv_vs_piv` |
| Images | `raw_image` ★; `piv_image` ★ (pre-processed) |
| Modal decomposition (POD) | `pod_mode` φ_n; `pod_spectrum` spectrum and temporal coefficient |
| (root) | `peak_locking`; `convergence` |

Every figure states in its title the reference quantities it uses.

## Parameters

### Vector fields

| Interface | API | Meaning |
|---|---|---|
| Vector fields folder (the PIV/PTV results by default) | `load_vectors(path, camera=None, pixels=False)` | `vectors.npz`, its folder, or a folder of DynamicStudio CSV files. |

### Camera and acquisition

| Interface | API (`Camera(...)`) | Default | Meaning |
|---|---|---|---|
| Field of view width (mm) | `fov_width_mm` | — (required for the physical space) | Width of the field of view; scale (mm/px) = width / image width in pixels. |
| Horiz. offset of the field of view (mm) | `x_offset_mm` | 0 | Position of the left edge along the jet: x global = x + offset. |
| Vert. offset of the field of view (mm) | `y_offset_mm` | 0 | Position of the middle of the image height: y global = y + offset. |
| Frame rate (Hz) | `frame_rate_hz` | 150 | dt = 1 / frame rate between the two images of a pair. |

### Scale space

| Interface | API | Meaning |
|---|---|---|
| Sensor (px, pairs) | `load_vectors(path)` (no camera) | Pixels, displacements, pair numbers; no reference quantities. |
| Physical (mm, s) | `load_vectors(path, camera=Camera(...))` | mm, m/s, s; reference quantities available. |

### Recipe (`PostSettings`)

| Box / interface label | API field | Default | Meaning |
|---|---|---|---|
| Statistical outlier filtering — Drop cell if beyond (σ) | `outlier_k` | 4.0 | Vectors further than k standard deviations from the mean of their grid point are removed (0: none). |
| Data Reconstruction — Invalid cells decay (# of maps) | `hold_maps` | 0 | Instantaneous maps: an invalid cell in the flow shows its most recent valid value from at most this many maps back (sample and hold). Statistics unchanged. |
| Reference quantities — Ref. velocity (m/s) toggle + value | `use_u_ref`, `u_ref` | off, None (= max of mean u) | Velocities shown over u_ref. Turbulence intensity and default flow time always use u_ref. |
| Reference quantities — Flow time (ms) toggle + value | `use_t_flow`, `t_flow` (s in the API) | on, None (= L_FoV / u_ref) | Times shown in flow times. |
| Reference quantities — Ref. length L_ref (mm) | `l_ref` | None | Positions shown as x / L_ref, y / L_ref (e.g. the nozzle diameter). |
| Rolling statistics — toggle + interval | `rolling`, `rolling_maps` | off, 100 | Statistics of each map over the maps up to it: every figure gets one per map (browsable, exportable, videos). |
| Peak locking — Bins per pixel (odd) | `peak_locking_bins` | 21 | Bins per pixel of the \|d_x\|, \|d_y\| histograms; quality = 1 − C compares the histogram with its 1 px moving average. |
| Jet potential core — detection | `core_mode` | `"level"` | `level`: the core is where mean u ≥ core_level % of u_ref; `skewness`: the core ends where the skewness of the profiles nears 0. |
| — Core: mean u ≥ (% of u_ref) | `core_level` | 95 | Level of the isocontour (velocity level detection). |
| — Skewness ≈ 0 below (% of max) | `skew_zero` | 20 | A profile is symmetric when \|skewness\| ≤ this % of the highest. |
| — Skewness change below (% of prev.) | `skew_change` | 10 | ...and its skewness changes by at most this % from the previous profile. |
| — Profiles for detection | `skew_profiles` | 40 | Profiles analysed by the skewness detection. |
| — Profiles of mean u/u_ref (stations) | `n_stations` | 6 | Profiles shown in `core_profiles`. |
| — First profile offset (columns) | `first_offset` | 0 | Columns between the exit and the first profile (clear of the nozzle). |
| — Ref. velocity (m/s) | `core_u_ref` | None (= max of mean u) | u_ref of the jet theory when the Reference u_ref is off. |
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
| Export folder (`PostProcessing`, `PostProcessing_PTV` for a PTV; `...`, ↺) | `post.export(folder, ...)` | Where the files go. |
| Quantities... — figures ticked | `figures=[keys]` (None: all) | Figures exported. |
| Quantities... — map shown / range of maps, first, last, every N | `index=`, `maps=[indices]` (negative from the end) | Instantaneous figures of one map, or of several maps (one subfolder per figure). |
| Quantities... — Video, fps, format | `videos=[keys]`, `fps=`, `video_maps=`, `video_format="gif"/"mp4"` | Videos of instantaneous figures. |
| Per-quantity overrides | `video_options={key: {...}}`, `image_options={key: [maps]}` | Maps/fps/format of one figure's video or images. |
| Singular (Ctrl+E) | `post.export(...)` | Writes the files. |
| Video... (tile) | `post.export_video(key, folder, step=1, fps=10, maps=None, fmt="gif")` | Video of exactly what a tile shows. |
| Batch... | `velocihub.batch.postprocess_folder(s)` | See [batch-processing.md](batch-processing.md). |

## Outputs

| File | Content |
|---|---|
| `[project]_PostProcessing_[Figure].png` | Statistics figures (e.g. `Run_30_PostProcessing_MeanU.png`) |
| `[project]_PostProcessing_[Figure]_[map].png` or `[project]_PostProcessing_[Figure]/` | Instantaneous figures of one or several maps |
| `[project]_PostProcessing_[Figure].gif` / `.mp4` | Videos |
| `metrics.json` | Key numbers: valid vectors, peak ratio, reference quantities, potential core, peak locking, convergence, PTV metrics |
| `postprocessing.json` | The recipe and the camera used |
