# Python API

Everything the interface does can be done from Python with the same parameters: the interface itself calls these
functions. This page goes through each part in the order of the interface, and gives for each action the call that
does it.

```python
import velocihub as vh       # the main names: PODBackgroundRemoval, run_piv, Camera, load_vectors, PostProcessing...
```

The processing runs without Tkinter: the interface (`velocihub.gui`) is only imported by `vh.launch_gui()`.

| Module | Contents |
|---|---|
| `velocihub.preprocessing` | `PODBackgroundRemoval`, `ContrastSettings`, `Mask`, filters (`intensity_capping`, `minmax_normalization`, `clahe`, `edge_enhancement`), `imread`, `imwrite` |
| `velocihub.piv` | `run_piv`, `PIVRun`, `Camera`, `get_engine`, `ENGINES`, `PIVEngine`, `register`, `count_particles`, `check_regression` |
| `velocihub.fields` | `load_vectors`, `VectorFields`, `FieldSettings` |
| `velocihub.postprocessing` | `PostProcessing`, `PostSettings`, `FIGURES`, `FIGURE_GROUPS`, `INSTANTANEOUS` |
| `velocihub.batch` | Batch functions ([batch-processing.md](batch-processing.md)) |
| `velocihub.project`, `velocihub.workspace` | Project and workspace files |
| `velocihub.archiving` | `archive`, `post_archive` ([archiving.md](archiving.md)) |

---

## 1. Pre-processing — `PODBackgroundRemoval`

### Session and images

| Interface | API |
|---|---|
| Images folder | `session = vh.PODBackgroundRemoval(folder, pattern="*.tif")` or `session.set_images(folder, pattern, files=None, first=1, last=None)` |
| Settings > Set temporary folder... | `PODBackgroundRemoval(..., temp_folder="D:/tmp")` (`"images"`: the image folder, as the interface; `None`: a new temporary folder deleted by `close()`) |
| Settings > Max memory usage... | `PODBackgroundRemoval(..., memory_fraction=0.8)` |
| Load images in memory (Ctrl+L) | `session.load_images_memory()` (done automatically when needed) |

Use the session as a context manager so its temporary files are deleted:

```python
with vh.PODBackgroundRemoval("D:/Jet/Run_30", pattern="*.tif") as session:
    print(session.NImg, "images in", session.path_in)
```

### Masks

| Interface | API |
|---|---|
| New (draw a polygon) | `session.add_mask([[x1, y1], [x2, y2], ...], name="Nozzle")` (pixels, centres at integer coordinates from 1) |
| ☑/☐, rename, Delete | `session.masks[i].enabled = False`, `session.masks[i].name = "..."`, `del session.masks[i]` |
| Import... / Export... | `session.load_masks("masks.json", replace=True)`, `session.save_masks("masks.json")` |
| (masked pixels) | `session.masked_pixels((height, width))` — boolean image, used as `masked=` by the PIV |

### Contrast enhancement

| Interface | API |
|---|---|
| Tick filters, set their parameters, Applied before POD / after background removal | `session.set_contrast(["capping", "minmax"], stage="post", cap_n=10, mm_window=15, mm_contrast=10)` |
| No filter ticked | `session.set_contrast(enabled=False)` or `set_contrast([])` |

Filter names and parameters: `capping` (`cap_n`), `minmax` (`mm_window`, `mm_contrast`), `clahe` (`clip`, `tiles`),
`edge` (`edge_sigma`, `edge_factor`); `stage="pre"` or `"post"` — see [preprocessing.md](preprocessing.md). The
filters also exist as functions (`vh.intensity_capping(A, n)`, `vh.minmax_normalization(A, window, min_contrast)`,
`vh.clahe(A, clip, tiles)`, `vh.edge_enhancement(A, sigma, factor)`).

### POD modes

| Interface | API |
|---|---|
| Evaluate video spectrum (Ctrl+R) | `session.evaluate_spectrum()` |
| ε σ, ε ψ; Evaluate threshold (Ctrl+T) | `session.eps_auto_sigma = 1e-4; session.eps_auto_psi = 1e-4; n = session.automatic_N_modes()` |
| Modes to evaluate; Evaluate modes (Ctrl+M) | `session.evaluate_modes(6)` |
| Spectrum and mode tiles | `session.plot_spectrum(mode=1)`, `session.plot_mode(n)`, `session.plot_temporal_coefficient(n)` (Matplotlib figures); `session.export_spectrum("spectrum.png")`, `session.export_modes("modes", n_modes=10, save_data=False)` |
| (modes kept in `pod.npz`) | `session.save_pod()`, `session.load_pod()` |

### Background removal and export

| Interface | API |
|---|---|
| Remove the background, Modes removed | the argument `n_modes_removed` of the calls below (0: none) |
| Run step > Singular (Ctrl+B) to the export folder | `paths = session.export_images(session.export_folder(), n_modes_removed=6)` |
| (the PIV's "use the export, or export first") | `paths = session.piv_images(n_modes_removed=6)` |
| Export > New images | `session.remove_background("out", n_modes_removed=6, folder="D:/Jet/Run_31", pattern="*.tif")` |
| (image tile: the recipe on one image) | `image, background = session.process_image(path, index, session.load_modes(6))` |

```python
with vh.PODBackgroundRemoval("D:/Jet/Run_30", pattern="*.tif") as session:
    session.load_masks("D:/Jet/masks.json")
    session.set_contrast(["capping"], stage="post", cap_n=10)       # the interface default
    session.evaluate_modes(8)
    session.export_spectrum("D:/Jet/Run_30/spectrum.png")
    images = session.piv_images(n_modes_removed=6)                 # -> D:/Jet/Run_30/PreProcessing
    masked = session.masked_pixels(vh.imread(images[0]).shape[:2])
```

---

## 2. Processing — PIV/PTV

### Engines and their settings

| Interface | API |
|---|---|
| PIV/PTV software list | `vh.piv.ENGINES` (`{"pairs": PaIRS-UniNa, "tractrac": TracTrac}`); `engine = vh.get_engine("PaIRS-UniNa")` (key or name) |
| The settings fields and their defaults | `[(p.key, p.label, p.default) for p in engine.parameters]`; `engine.defaults()` |
| Validation (red fields) | `engine.check({"windows": (128, 64, 32), "overlap": 50})` — completes with the defaults, `ValueError` if invalid |
| Settings > PaIRS Python interpreter... | `vh.get_engine("PaIRS-UniNa", python="C:/Users/me/anaconda3/envs/PaIRS/python.exe")` |
| Is it installed? | `engine.available()` → `(ok, message)` |

Settings keys are listed in [processing.md](processing.md) (`windows`, `overlap`, `final_iterations`, `min_mean`,
`min_std`, `subtract_mean` for PaIRS; `spacing`, `particle_size`, `kernel`, `vid_loop`... for TracTrac).

### Running

| Interface | API |
|---|---|
| Run step > Singular (F5) | `vectors = vh.run_piv(images, None, "PaIRS-UniNa", settings, output="D:/Jet/Run_30/PIV", masked=masked)` |
| ... with SPICY regression | `vh.PIVRun(images, None, engine, settings, output, masked=masked, regression={"method": "spicy", "settings": {"n_k": (4,)}}).run()` |
| Run on the pair shown (F6) | `vh.PIVRun(images, engine=engine, settings=settings, output=tmp, pairs=[(k, k + 1)]).run()` |
| Stop / continue | Ctrl+C in a script; running again with the same images and settings continues from the chunks saved |
| Run in the background | `run = vh.PIVRun(...); process = run.start()` then `process.poll()` yields messages; `run.save()` writes the results |

`run_piv` / `PIVRun.run` return the path of `vectors.npz` (None if stopped). `camera=None` keeps the results in pixels,
as the interface does; the camera is applied in the post-processing.

```python
vectors = vh.run_piv(images, engine="TracTrac", settings={"particle_size": 3, "vid_loop": 2},
                     output="D:/Jet/Run_30/PTV")
```

### Particles in the windows

| Interface | API |
|---|---|
| S/N threshold, diameter, windows | `found = vh.piv.count_particles(image, center=(960, 600), sizes=(64, 32, 16), snr=4, diameter=3)` → `found["windows"]` (each `size`, `count`), `found["background"]`, `found["noise"]` |

### Reading the results

```python
f = vh.load_vectors("D:/Jet/Run_30/PIV/vectors.npz")            # sensor space: px, pairs
print(f.n, f.shape, f.u_px.shape, f.valid.mean())                # maps, grid, displacements, valid fraction
print(f.settings["engine"], f.settings["pairs"][:3])
```

`VectorFields` holds `x`, `y`, `U`, `V`, `u_px`, `v_px`, `valid`, `status`, `peak_ratio` (PIV), `particles` (PTV),
`tracks` (path of `tracks.npz`), `t`, `settings`.

---

## 3. Post-processing — `PostProcessing`

### Vector fields, camera, scale space

| Interface | API |
|---|---|
| Vector fields folder | `vh.load_vectors(path)` (`vectors.npz`, its folder, or a folder of DynamicStudio CSV files) |
| Camera and acquisition | `camera = vh.Camera(fov_width_mm=196.1, x_offset_mm=0, y_offset_mm=0, frame_rate_hz=150)` |
| Scale space: Sensor | `fields = vh.load_vectors(path)` |
| Scale space: Physical | `fields = vh.load_vectors(path, camera=camera)` |

### Recipe

Every box of the side panel is a field of `PostSettings` (table in [postprocessing.md](postprocessing.md)):

```python
settings = vh.PostSettings(
    outlier_k=4,                       # Statistical outlier filtering
    hold_maps=5,                       # Data Reconstruction (sample and hold)
    use_u_ref=True, u_ref=None,        # Reference quantities: u_ref (None: max of mean u)
    use_t_flow=True, t_flow=None,      # ... flow time in s (the interface shows ms)
    l_ref=50,                          # ... reference length (mm)
    rolling=False, rolling_maps=100,   # Rolling statistics
    peak_locking_bins=21,              # Peak locking
    core_mode="level", core_level=95,  # Jet potential core
    n_stations=6, first_offset=0, core_u_ref=None,
    pod_mode=1, pod_mode_last=None, pod_combine=False)   # POD modes shown/exported
post = vh.PostProcessing(fields, settings)
```

### Figures, tiles and numbers

| Interface | API |
|---|---|
| Figure menu of a tile | `fig = post.plot("mean_velocity")` (Matplotlib `Figure`); keys: `vh.FIGURES`, groups: `vh.postprocessing.FIGURE_GROUPS` |
| Map slider (instantaneous figures) | `post.plot("velocity", index=k)` (negative from the end) |
| Colour range... / colour map / isocontour | `post.plot(key, clim=(0, 1.2), cmap="magma", contour=0.5)` |
| Compare: ... + wiper | `post.export(folder, figures=[key], compares={key: ("u_rms", 0.5)})` |
| Video... | `post.export_video("velocity", folder, step=1, fps=24, maps=range(0, 200), fmt="mp4")` |
| Key numbers (`metrics.json`) | `post.metrics()` |
| Is a figure available? | `post.map_available(key)`, `post.missing(key)` |

### Export

| Interface | API |
|---|---|
| Quantities... + Singular (Ctrl+E) | `post.export(folder, figures=None, index=0, maps=None, videos=(), fps=10, video_maps=None, video_format="gif", clims=None, cmaps=None, contours=None, compares=None, video_options=None, image_options=None, dpi=130, name=None)` |

```python
camera = vh.Camera(fov_width_mm=196.1, frame_rate_hz=150)
post = vh.PostProcessing(vh.load_vectors("D:/Jet/Run_30/PIV", camera=camera),
                         vh.PostSettings(outlier_k=4, use_u_ref=True, l_ref=50))
files = post.export("D:/Jet/Run_30/PostProcessing",
                    figures=["mean_velocity", "u_rms", "core_map", "velocity", "convergence"],
                    maps=[0, -1],                     # instantaneous figures of the first and last maps
                    videos=["velocity"], fps=24, video_format="mp4",
                    cmaps={"mean_velocity": "magma"}, contours={"mean_velocity": 0.95})
print(post.metrics())                                   # the key numbers, as in metrics.json
```

---

## 4. Batch processing

| Interface | API |
|---|---|
| Pre-processing > Batch... | `vh.batch.preprocess_folder(folder, section)` per project |
| PIV/PTV > Batch... | `vh.batch.process_folder(folder, engine=..., settings=..., n_modes=..., contrast=..., masks=..., regression=...)`, or `process_folders(roots, ...)` |
| Post-processing > Batch... | `vh.batch.postprocess_folder(folder, camera, settings, figures=..., maps=..., videos=...)`, `postprocess_folders(...)` |
| The window Batch itself | `python -m velocihub.batch_run job.json` (job format in `velocihub/batch_run.py`) |

Details and examples: [batch-processing.md](batch-processing.md).

---

## 5. Archiving

| Interface | API |
|---|---|
| Archive projects... | the command `velocihub-archive folders.txt --output out.zip --no-gui`, or the functions of `velocihub.archiving.archive` (example below) |
| Post-processing > Batch... > stitchings | `from velocihub.archiving import post_archive as pa; pa.main(archive, setups=None, export_root=None, stitched_folder=os.path.join(archive, "Stitched"), items=None)` |

```python
from velocihub.archiving import archive as ap

units, warnings = ap.scan(ap.read_roots("folders.txt"))       # the projects (and files outside them) of the roots
categories = list(ap.DEFAULT_CATEGORIES)                      # or e.g. ["vectors", "postprocessing", "project"]
base, entries = ap.archive_entries(units, categories)         # (path, name in the archive, size) of each file
errors = ap.write_archive("X:/Archives/Jet.zip", entries, "zip",
                          lambda errors: ap.archive_summary(units, categories, base, "zip", errors),
                          ap.Progress(sum(size for _, _, size in entries)))   # total bytes, for its progress
```

Details: [archiving.md](archiving.md).

---

## 6. Project files

The interface saves everything in `VelociHub_project.json` (sections `preprocessing`, `piv`, `postprocessing`, with
the history); the batch functions (`velocihub.batch`) write the same file. The lower-level objects of sections 1-3
(`PODBackgroundRemoval`, `run_piv`, `PostProcessing`) only write their results (images, `vectors.npz`, figures),
not the project file: use the batch functions when a script's results should open in the interface with their
settings.

```python
project = vh.read_project("D:/Jet/Run_30")          # dict, or None without project file
print(project["preprocessing"]["background_removal"], project["piv"]["settings"])
```

A project section `preprocessing` is what `batch.preprocess_folder` and `batch.process_folder(preprocessing=...)`
take, so a recipe tuned in the interface can be replayed by a script:

```python
section = vh.read_project("D:/Jet/Run_30")["preprocessing"]
for folder in vh.batch.measurement_folders("D:/Jet/FOV_1"):
    vh.batch.process_folder(folder, preprocessing=section)
```

## 7. Extending: a new PIV engine

```python
from velocihub.piv import PIVEngine, register
from velocihub.piv.base import Parameter

@register
class MyPIV(PIVEngine):
    key, name, method = "mypiv", "MyPIV", "PIV"
    parameters = (Parameter("window", "Window (px)", 32, "int", "Interrogation window."),)

    def available(self):
        return True, "MyPIV ready"

    def start(self, images, pairs, settings, work, key=None):
        # Start the computation of the pairs (a WorkerProcess, see PaIRSEngine in pairs_engine.py for the protocol:
        # the worker writes raw.npz in the work folder and reports its progress as JSON lines)
        ...
```

Registered engines appear in the PIV/PTV page and the batch automatically.
