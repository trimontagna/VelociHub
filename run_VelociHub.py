"""Run VelociHub with its interface: pre-processing, PIV/PTV and post-processing, one page each.

    python run_VelociHub.py [folder [pattern]] [--masks masks.json]

Same as the command velocihub (after pip install) or python -m velocihub. Without arguments, the images are chosen in
the page Pre-processing; with a folder, its images (or those matching the pattern, e.g. "Export*.bmp") are opened, and
with --masks the masks of a file. Settings and results are kept in VelociHub_project.json in the image folder.
"""
from velocihub.cli import gui

if __name__ == "__main__":
    gui()
