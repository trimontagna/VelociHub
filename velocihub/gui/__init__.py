"""The interface of VelociHub (Tkinter): one window (app.py) with a page for each step: the pre-processing, the
PIV and the post-processing. Run it with main(), or velocihub.launch_gui().

Each page is a module: to keep only some steps, remove the others from PAGES in app.py (and delete their modules):
the PIV needs the pre-processing, the post-processing uses the PIV if it is there."""

from .app import PAGES, MainWindow, main
from .preprocessing_page import DEFAULT_CONTRAST, DEFAULT_MODES, PreProcessingPage

__all__ = ["DEFAULT_CONTRAST", "DEFAULT_MODES", "MainWindow", "PAGES", "PreProcessingPage", "main"]
