# Archiving (add-on)

Archiving is not a step of the processing: it is a set of tools to **keep** and **combine** finished measurements.
It lives in `velocihub/archiving/` and has two parts:

1. **Archive projects** (`archive.py`) — pack the files of projects into a ZIP or TAR archive, with a summary.
2. **Post-processing and stitching of an archive** (`post_archive.py`, interface `post_archive_gui.py`) —
   post-process every measurement of several fields of view with one recipe, then stitch the fields of view of each
   setup side by side. The interface reaches the same code through the stitchings of the post-processing
   window Batch, with the manual **alignment** of the fields of view (`gui/alignment.py`).

## 1. Archive projects

### Features

- Projects found from root folders (every folder holding a `VelociHub_project.json`, or a project file of the
  previous names, at any depth), plus the files of the roots outside the projects (calibration, folders not processed)
  as one more row.
- The files of each project are found from its **project file** (the folders of its images and of the results of
  the three steps), so results exported outside the default folders are archived too.
- Categories of files chosen per project; the folder structure kept from the lowest common folder (a file outside it
  goes to `_external/[drive]/[path]`); a summary `VelociHub_archive.json` at the top of the archive.
- Written as `[archive].part` and renamed when complete: an interrupted archive is never mistaken for a complete one.
  Free space is checked first.
- Uses the Python standard library only.

### Where

- **Interface**: panel Projects > **Archive projects...** — the window Batch with the categories per project/folder
  and the archive file and format as options.
- **Command**: `velocihub-archive` (script `run_VelociHub_archive.py`), with its own small interface unless
  `--list` or `--no-gui`.

### Parameters

| Option | Meaning |
|---|---|
| `folders.txt` | Text file of the root folders, one per line (relative to the file or absolute; empty lines and `#` comments skipped). Default: `folders.txt` of the current folder. |
| `--output archive.zip` | Archive file (interface default: `VelociHub_archive_[date].zip` next to the workspace file). |
| `--format` | `zip` (not compressed, fastest — default), `zip-compressed` (smaller, much slower), `tar`. |
| `--include CATEGORY...` / `--exclude CATEGORY...` | Categories added to / removed from the defaults. |
| `--list` | Only list what would be archived (files and sizes per category and project). |
| `--no-gui` | Create the archive without the interface. |

| Category | Default | Files |
|---|---|---|
| `originals` | yes | Original images |
| `preprocessed` | yes | Pre-processed images |
| `vectors` | yes | PIV maps: vector fields (`vectors.npz`) |
| `csv` | yes | PIV maps: CSV files (older versions) |
| `postprocessing` | yes | Post-processing: figures, report and statistics |
| `pod` | yes | POD spectrum and modes (`pod.npz`) |
| `project` | yes | Project, masks and settings (JSON files) |
| `other` | no | Other files of the project folders |
| `outside` | yes | Files of the roots outside the projects (Calibration...) |

```bash
velocihub-archive folders.txt --list
velocihub-archive folders.txt --output X:/Archives/Jet_2026.zip --no-gui --exclude originals
```

## 2. Post-processing and stitching of an archive

### Features

An **archive** here is a folder with one subfolder per field of view (`FOV_1`, `FOV_2`, ...), each holding one
subfolder per measurement setup with its PIV results (`[setup]/PIV/vectors.npz`). Then:

1. Every measurement is post-processed (`batch.postprocess_folders`) with the camera of its field of view and a
   common recipe; figures, `metrics.json` and `postprocessing.json` go to its `PostProcessing` folder (or under
   `--export-root`).
2. The setups measured in several fields of view (same folder name) are **stitched** into `Stitched/[setup]`:
   - statistics maps with the fields of view side by side, each with its own global x axis, the common height
     (global y), one colour scale and one arrow scale, and each panel's reference quantities in its title;
   - jet theory plots: potential-core profiles of each field of view, and the centreline velocity of all of them on
     one global x axis;
   - instantaneous maps of a chosen map (or several), and **videos** of instantaneous maps with fixed scales;
   - POD modes.
3. **Reference velocities**: either the maximum mean velocity of the setup in a reference field of view (the jet exit)
   for all, or **velocity matching** — each field of view gets its own u_ref so that the mean u/u_ref agrees in the
   overlaps (the fields of view are separate runs whose flow rates differ).
4. **Rolling statistics** are supported: every stitched figure then exists per map, with videos.

### Where

- **Command**: `velocihub-post-archive [archive] [--setups name...] [--export-root folder] [--stitched folder]`
  (script `run_VelociHub_post_archive.py`). The options are the module-level settings of
  `velocihub/archiving/post_archive.py` (below), which you can set from Python before calling `main()`.
- **Interface for the archive**: `velocihub-post-archive-gui` (script `run_VelociHub_post_archive_gui.py`): tabs
  **Archive** (folders, setups), **Fields of view**, **Recipe** (every `PostSettings` field and the velocity
  matching), **Exports**, **Stitching**; runs in a separate process that can be stopped; options remembered.
- **Main interface**: post-processing page > **Batch...** > stitchings of the tree (see
  [batch-processing.md](batch-processing.md)); each chain of projects is post-processed and stitched into the folder
  `Stitched` of the reference's `PostProcessing` folder.

### Parameters (`velocihub.archiving.post_archive`)

| Setting | Default | Meaning |
|---|---|---|
| `ARCHIVE` | current folder | The archive folder. |
| `FOV_WIDTH_MM` | `{"FOV_1": 196.1, "FOV_2": 196.5, "FOV_3": 196.3}` | Width of each field of view (mm). **Set these for your setup.** |
| `FOV_X_OFFSET_MM` / `FOV_Y_OFFSET_MM` | 0, 120, 240 / `{}` | Horizontal offset of each field of view's left edge, vertical offset of its middle (mm). |
| `FRAME_RATE_HZ` | 150 | Frame rate of the camera. |
| `SETTINGS` | a `PostSettings` | The common recipe (outlier_k, hold_maps, reference quantities, core detection, rolling...). |
| `REFERENCE_FOV` | `"FOV_1"` | Field of view giving the reference velocity of each setup. |
| `MATCH_VELOCITY`, `MATCH_JET`, `MATCH_EDGE` | True, 0.5, 2 | Velocity matching in the overlaps: on/off, jet cells used (≥ this fraction of the overlap maximum), edge columns left out. |
| `INDEX` | −1 | Map of the instantaneous figures exported (negative from the end). |
| `EXPORT_FIGURES` | None (all) | Figures exported per measurement. |
| `EXPORT_MAPS` | None (the map `INDEX`) | Maps of the instantaneous figures (e.g. `range(0, 100, 5)`). |
| `EXPORT_VIDEOS`, `EXPORT_FPS`, `EXPORT_FORMAT` | `[]`, 24, `"gif"` | Videos per measurement. |
| `STITCHED_FOLDER` | `ARCHIVE/Stitched` | Where the stitched figures go. |
| `STITCHED` | all statistics maps | Maps stitched. |
| `STITCHED_INSTANTANEOUS`, `STITCHED_VIDEOS`, `STITCHED_VIDEO_MAPS` | all instantaneous maps, `["velocity"]`, None | Stitched instantaneous maps and videos. |
| `STITCHED_JET` | `["core_profiles", "core_centreline"]` | Jet theory plots stitched. |

```python
from velocihub.archiving import post_archive as pa
import velocihub as vh

pa.FOV_WIDTH_MM = {"FOV_1": 200.0, "FOV_2": 200.0}
pa.FOV_X_OFFSET_MM = {"FOV_1": 0.0, "FOV_2": 150.0}
pa.SETTINGS = vh.PostSettings(outlier_k=4, use_u_ref=True, l_ref=50)
pa.main("D:/Jet_archive", setups={"Run_30"}, stitched_folder="D:/Jet_archive/Stitched")
```

## 3. Alignment of the fields of view

Right-click a node of a stitching in the post-processing window Batch > **Align the stitching (...)...**: the fields of view are drawn at their
offsets and scales over a common background — **Mean raw image** (the median of raw images: the fixed scene, the
same in every field of view), **Mean velocity |u|**, or a **Calibration image** (first image of a `Calibration`
folder above the project). Views: Blend, Difference (dark where the overlap matches), Flicker. Drag a field of view,
use the arrows (0.1 mm, Shift 1 mm) or type its offsets; **Suggest** proposes the shift that best correlates the
overlap (phase correlation, to be checked). **Apply** writes the offsets into the camera settings of the projects.
