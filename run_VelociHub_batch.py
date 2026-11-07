"""Batch pre-processing and PIV of measurement folders with the default settings of the interface.

    python run_VelociHub_batch.py root [root ...] [--pattern "*.tif"] [--masks] [--force] [--list]
    python run_VelociHub_batch.py @folders.txt [--masks]

Same as the command velocihub-batch (see velocihub/cli.py and docs/batch-processing.md).
"""
from velocihub.cli import batch

if __name__ == "__main__":
    batch()
