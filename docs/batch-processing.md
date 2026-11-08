# Batch processing

Every step can run on many projects at once, each project keeping its own settings. There are three ways to do it:
the **window Batch** of the interface, the command **`velocihub-batch`**, and the **batch API**
(`velocihub.batch`). All three save the results exactly as the interface does (same folders, same
`VelociHub_project.json`), so a project processed in batch opens in the interface with everything restored.

## 1. The workspace (panel Projects)

The panel on the left of the window is the **workspace**: root folders scanned for projects, saved in a workspace
file (`VelociHub_workspace.json` in the first root added, or wherever you save it; the last one reopens at start).

- A **project** is a folder with a project file, with images (two or more: pre-processing and PIV), or with the
  results of a velocimetry pipeline (`PIV/vectors.npz` of PaIRS, `PTV/vectors.npz` of TracTrac, or `vectors.npz` in
  the folder). The column **Velocimetry** lists the pipelines whose results each project has (`PaIRS, TracTrac`); a
  project with several shows a row for each. A pipeline is a registered engine (`velocihub/pipelines.py`): a new
  engine adds its own.
- Right-click: **Add a root folder (detect its projects)...**, add a single project (folder or project file), open, rename, remove from the workspace (files stay), relink,
  refresh. Projects that the page shown can't use are greyed out; lost ones (moved) are red until found again or
  relinked. Paths are saved absolute and relative to the workspace file, so a tree moved with its workspace file is
  found again.
- The results subfolders (`PreProcessing`, those of the pipelines (`PIV`, `PTV`), `PostProcessing*`, `Stitched`)
  are not scanned.

## 2. The window Batch

**Run step > Batch...** in each page (and **Archive projects...** in the panel Projects) opens the window Batch for
that step:

- **Tree of the workspace** with checkboxes (**Check all** / **Uncheck all** right below it): the ticked projects are
  run; those the step can't run are greyed out. The **open project is the reference of every node**: a project whose
  settings differ from its own has its name in **bold** (and the folders containing one), with a **↺** in its row that
  gives it the settings of the open project (also in its right-click menu; a folder asks first).
- **Settings** of the node selected (the box is named after it). Each project starts with the settings of its own
  project file (or those of the interface if it has none). Changes are *differences*: with a project selected they
  apply to it only; with a folder selected, that setting is applied to every project below it (each keeps its other
  settings). Settings that differ between the projects of a folder are shown in bold with a ↺ button to give them all
  the value shown.
- Settings changed are saved into the project files (the section of the step), so a project opened later shows them.
- **Run** starts a separate process (the window stays usable) with a log, a progress bar and previews of the files
  written. **Stop** ends the step running (the PIV keeps the pairs computed) and skips the remaining projects.

What each step's form contains:

| Step | Settings per project | Options of the run |
|---|---|---|
| Pre-processing | Masks (enabled), contrast enhancement, background removal (enabled, modes removed) | — |
| Velocimetry | Engine and its settings, regression, results folder; with the pre-processing settings (the images are pre-processed first if needed) | — |
| Post-processing | **Results sets exported** (the pipelines checked, PaIRS, TracTrac...: each project exports those it has, each with the same settings and quantities, to `PostProcessing/[pipeline]`), scale space, camera, recipe, export folder, quantities to export (as the Quantities window: the groups of the menus of the tiles; those that the projects can't export greyed out and folded; a quantity whose parameters differ from the open project's in bold with its ↺; exporting a quantity or not isn't a parameter change) | Stitchings (below) |
| Archive projects | Categories of files archived per project | Archive file, format |

**Stitching** (post-processing batch): select nodes in order (the first is the reference), Ctrl+click to add or
remove. Names are matched: stitching `FOV1`, `FOV2`, `FOV3` that each contain `110Hz/A` stitches `FOV1/110Hz/A` with
`FOV2/110Hz/A` and `FOV3/110Hz/A`. The column **Stitch** names each chain of projects by a letter and each project by
its rank: `A.1` (the reference), `A.2`, ..., `B.1`... (`✗`: not stitched, the reason in its tip; the nodes chosen:
`N1`, `N2`...). A chain is stitched only if all its projects are exported, once per results set that all of them have
and export (`PostProcessing/[pipeline]/Stitched` of the reference). Right-click a stitched project: **Take settings of
A.1** (its stitch reference: settings and quantities, its own camera kept), or **Align Group A** (the fields of view
aligned by hand, [archiving.md](archiving.md)).

## 3. Command line: `velocihub-batch`

Pre-processing and PIV of measurement folders with the **default settings of the interface** (intensity capping
n = 10 after the removal of 6 POD modes, no masks; PaIRS-UniNa 128-64-32 px, 50 % overlap):

```bash
velocihub-batch D:/Jet/FOV_3                          # each subfolder of FOV_3 with images is a measurement
velocihub-batch D:/Jet/FOV_1 D:/Jet/FOV_2 --pattern "*.tif"
velocihub-batch @folders.txt --masks                  # roots read from a file, one per line
velocihub-batch D:/Jet/FOV_3 --list                   # only list what would be processed
velocihub-batch D:/Jet/FOV_3 --force                  # recompute even if up to date
```

| Option | Meaning |
|---|---|
| `roots` | Folders of images, or folders whose subfolders hold images (one level down). |
| `@file` | Read the roots from a text file. |
| `--pattern` | Image files to use, e.g. `"*.tif"` (default: all images). |
| `--masks` | Each measurement uses the masks file in its own folder, else the masks file at its root (`masks.json` if there are several). |
| `--force` | Compute the PIV even if its results are up to date (same images, recipe and settings are skipped otherwise). |
| `--list` | List the measurement folders and masks found, without processing. |

Ctrl+C stops the PIV after its current pair; running again continues from the last chunk saved. The script
`run_VelociHub_batch.py` does the same from a clone.

## 4. Batch API

The simplest way is the high-level API ([api.md](api.md), section 0), which runs each project with the settings of
its project file, as the window Batch does:

```python
import velocihub as vh

items = vh.projects("D:/Jet")             # or a workspace file
for p in items:
    p.set_piv(overlap=75)
    p.save()
vh.run("piv", items)
vh.run("postprocessing", items)
```

The lower-level functions of `velocihub.batch` take the settings as arguments instead:

```python
import velocihub as vh

# Pre-processing + PIV of every measurement of FOV_3 (process_folder options are accepted, see below)
for result in vh.batch.process_folders(["D:/Jet/FOV_3"], pattern="*.tif", root_masks=True):
    print(result.folder, result.status, result.describe())

# One folder with chosen settings
result = vh.batch.process_folder(
    "D:/Jet/FOV_3/Run_30",
    engine="PaIRS-UniNa", settings={"windows": (96, 48, 24), "overlap": 75},
    n_modes=8,                                                     # POD modes removed (0: none)
    contrast=vh.ContrastSettings(filters=("capping",), stage="post", cap_n=10),
    masks="D:/Jet/FOV_3/masks.json",
    regression={"method": "spicy", "settings": {"n_k": (4,)}},     # optional
    force=False)

# Pre-processing only, with a project section (as the interface saves it)
section = {"masks": [], "contrast": {"filters": ["capping"], "stage": "post", "cap_n": 10},
           "background_removal": {"enabled": True, "modes_removed": 6}}
vh.batch.preprocess_folder("D:/Jet/FOV_3/Run_30", section)

# Post-processing of several folders with a camera and a recipe, exported as the interface does
camera = vh.Camera(fov_width_mm=196.1, x_offset_mm=0, frame_rate_hz=150, y_shift=0.5)
settings = vh.PostSettings(outlier_k=4, use_u_ref=True, l_ref=50)
folders = vh.batch.measurement_folders("D:/Jet/FOV_3")
for result in vh.batch.postprocess_folders(folders, camera, settings, figures=["mean_velocity", "u_rms"],
                                           maps=[0, -1], videos=["velocity"], fps=24, video_format="mp4"):
    print(result.describe())
```

| Function | Does |
|---|---|
| `measurement_folders(root, pattern=None)` | The measurement folders of a root (itself if it has images, else its subfolders with images). |
| `find_masks(root)`, `measurement_masks(folder, root_masks)` | The masks file of a root / of a measurement. |
| `process_folder(folder, pattern, engine, settings, n_modes, contrast, masks, temp_folder, force, ..., output_folder, regression)` | Pre-processing + PIV of one folder; returns a `BatchResult` (`status`: done, skipped, failed, stopped; `describe()`). |
| `process_folders(roots, pattern, report, verbose, root_masks, **options)` | Same for every measurement of the roots; yields a `BatchResult` per folder. |
| `preprocess_folder(folder, section, progress, temp_folder, verbose)` | The pre-processing alone, from a project section. |
| `postprocess_folder(folder, camera, settings, export_folder, index, dpi, name, figures, maps, videos, fps, video_format, maps_folder, pixels, **export)` | Post-processing export of one folder (`camera=None` or `pixels=True`: sensor space). |
| `postprocess_folders(folders, camera, settings, verbose, **options)` | Same for several folders. |

The window Batch runs `python -m velocihub.batch_run job.json`, a JSON job (step, projects with their settings,
options) documented in `velocihub/batch_run.py`; you can write such jobs yourself to replay a batch.
