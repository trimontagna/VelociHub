"""Post-process the PIV results of an archive of fields of view and stitch them (add-on Archiving).

    python run_VelociHub_post_archive.py [archive folder] [--setups name ...] [--export-root folder] [--stitched folder]

Same as the command velocihub-post-archive. The options are the globals at the top of
velocihub/archiving/post_archive.py (or set them in the interface: run_VelociHub_post_archive_gui.py).
"""
from velocihub.archiving.post_archive import cli

if __name__ == "__main__":
    cli()
