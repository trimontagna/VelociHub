"""Archive VelociHub projects into a ZIP or TAR file (add-on Archiving).

    python run_VelociHub_archive.py [folders.txt] [--output archive.zip] [--format zip|zip-compressed|tar] [--no-gui]

Same as the command velocihub-archive (see velocihub/archiving/archive.py and docs/archiving.md).
"""
from velocihub.archiving.archive import main

if __name__ == "__main__":
    main()
