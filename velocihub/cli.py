"""Command-line entry points of VelociHub (installed as commands by pip: pyproject.toml):

- velocihub [folder [pattern]] [--masks masks.json]: the interface (gui);
- velocihub-batch: the pre-processing and the PIV of measurement folders (batch, below);
- velocihub-archive, velocihub-post-archive, velocihub-post-archive-gui: the add-on Archiving
  (velocihub/archiving).

Batch processing with VelociHub: the pre-processing (POD background removal) and the PIV (PaIRS-UniNa) of
measurement folders, one after the other, with the default settings of the interface.

    velocihub-batch root [root ...] [--pattern "*.tif"] [--masks] [--force] [--list]
    velocihub-batch @folders.txt [--masks]

Each root is a folder of training images, or a folder whose subfolders hold the images (one level down: X:/FOV_3
processes each measurement of FOV_3). @folders.txt reads the roots from a text file, one per line. --masks: the
masks of each measurement are those of its own file of masks (in its folder, more specific to it), else of the file
of masks at its root (the JSON file saved by the interface, Export masks..., masks.json if there are several; none:
no masks). The results are
saved as the interface saves them, in each measurement folder: PreProcessing (images pre-processed, pod.npz), PIV
(vectors.npz, settings.json with the run time) and VelociHub_project.json, so that the folder opened
in the interface shows them. A folder whose PIV results are up to date is skipped (--force: computed again).
Ctrl+C stops the PIV running after its current pair: run again, it continues from its last chunk of pairs saved.

The PIV needs PaIRS-UniNa (Python 3.10 to 3.13), found automatically among the conda environments.
"""
import argparse
import sys

import velocihub as vh
from velocihub.project import format_duration


def safe_output():
    """The characters the console can't encode (→, σ... in a cp1252 console, or output redirected to a file) printed
    as ? instead of stopping the run."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def gui(argv=None):
    """The interface, optionally with the images of a folder (those matching the pattern) and a file of masks."""
    safe_output()
    parser = argparse.ArgumentParser(prog="velocihub", description="VelociHub: pre-processing, PIV/PTV and "
                                     "post-processing of PIV images, with its interface")
    parser.add_argument("folder", nargs="?", help="folder of the training images")
    parser.add_argument("pattern", nargs="?", help='pattern of the images in the folder, e.g. "*.tif"')
    parser.add_argument("--masks", help="masks saved by the interface or by save_masks (JSON file)")
    args = parser.parse_args(argv)
    vh.launch_gui(args.folder, args.pattern, args.masks)


def batch(argv=None):
    """Pre-processing and PIV of measurement folders with the default settings of the interface (velocihub-batch)."""
    safe_output()
    parser = argparse.ArgumentParser(prog="velocihub-batch", description="VelociHub batch: pre-processing and PIV of measurement folders "
                                                 "with the default settings of the interface",
                                     fromfile_prefix_chars="@")
    parser.add_argument("roots", nargs="+", help="folders of training images, or folders whose subfolders hold them")
    parser.add_argument("--pattern", help='pattern of the images in the folders, e.g. "*.tif" (default: all images)')
    parser.add_argument("--masks", action="store_true", help="the masks of the file of masks in each measurement "
                                                             "folder, else at its root (masks.json if there are "
                                                             "several)")
    parser.add_argument("--force", action="store_true", help="compute the PIV even if its results are up to date")
    parser.add_argument("--list", action="store_true", help="only list the measurement folders found")
    args = parser.parse_args(argv)

    if args.list:
        for root in args.roots:
            try:
                folders = vh.batch.measurement_folders(root, args.pattern)
                masks = vh.batch.find_masks(root) if args.masks else None
            except ValueError as e:
                print(f"{root}: {e}")
                continue
            print(f"{root}: {len(folders)} measurement folder(s)"
                  + (f", masks of {masks}" if masks else ", no file of masks at the root" if args.masks else ""))
            for folder in folders:
                try:
                    own = vh.batch.measurement_masks(folder, masks) if args.masks else masks
                except ValueError as e:
                    own = e
                print(f"  {folder}" + (f": {own}" if isinstance(own, ValueError)
                                       else f": masks of {own}" if own != masks else ""))
        sys.exit(0)

    results = []
    try:
        for result in vh.batch.process_folders(args.roots, args.pattern, root_masks=args.masks, force=args.force):
            results.append(result)
    except KeyboardInterrupt:
        print("\nStopped: run again to continue (the chunks of pairs computed are kept)")
    print("\nSummary:")
    for result in results:
        print(f"  {result.folder}\n    {result.describe()}")
    done = [r for r in results if r.status == "done"]
    total = sum((r.piv_seconds or 0) + (r.preprocessing_seconds or 0) for r in done)
    print(f"{len(done)} done, {sum(r.status == 'skipped' for r in results)} skipped, "
          f"{sum(r.status in ('failed', 'stopped') for r in results)} failed or stopped"
          + (f" ({format_duration(total)} of processing)" if total else ""))
    sys.exit(0 if results and all(r.status in ("done", "skipped") for r in results) else 1)


if __name__ == "__main__":
    batch()
