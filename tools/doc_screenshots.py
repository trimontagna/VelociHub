"""The pictures of the documentation (docs/images): the pages and windows of the interface on the synthetic turbulence
of the benchmark (velocihub/benchmark: a frozen random field convected by a uniform stream, 512 x 320 px, 201 frames).

    python tools/doc_screenshots.py [--example FOLDER] [--prepare-only] [--force] [--only NAME,NAME...]

1. The example project (FOLDER, ~/VelociHub/examples by default): the images of the turbulence rendered as in the test
   workspace, then pre-processed, processed by PIV (PaIRS-UniNa) and PTV (TracTrac) and post-processed with the API,
   as the window Batch would (skipped when the results are there; --force: made again).
2. The interface opened on it, with scratch preferences (yours untouched), each page and window set up and captured
   (Windows: the screen area of the window, so keep it uncovered while the script runs).

Run it again after a change of the interface: the pictures are overwritten.
"""

import argparse
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(REPO, "docs", "images")
DEFAULT_EXAMPLE = os.path.join(os.path.expanduser("~"), "VelociHub", "examples")
PROJECT = "Turbulence"
CASE = "turbulence"
# The camera of the case (benchmark_case.json): 0.2 mm/px, 1 kHz
FOV_WIDTH_MM = 102.4
FRAME_RATE_HZ = 1000.0
PIV_WINDOWS = (64, 32)


def prepare(example, force=False):
    """The example project in example/Turbulence, processed (PIV and PTV) and post-processed. Returns its folder."""
    sys.path.insert(0, REPO)
    import velocihub as vh
    from velocihub.benchmark import FLOWS
    from velocihub.benchmark.images import write_measurement

    folder = os.path.join(example, PROJECT)
    if force and os.path.isdir(folder):
        shutil.rmtree(folder)
    if not os.path.isdir(folder):
        print("Images of the synthetic turbulence...")
        write_measurement(FLOWS[CASE](), folder, seed=list(FLOWS).index(CASE))
        # A measurement as any other: no exact fields, no description of a benchmark case
        shutil.rmtree(os.path.join(folder, "Exact"), ignore_errors=True)
        for name in ("benchmark_case.json", "masks.json"):
            if os.path.isfile(os.path.join(folder, name)):
                os.remove(os.path.join(folder, name))
    p = vh.Project(folder)
    p.set_name(PROJECT)
    p.set_preprocessing(modes_removed=1)
    p.set_postprocessing(camera={"fov_width_mm": FOV_WIDTH_MM, "frame_rate_hz": FRAME_RATE_HZ},
                         figures=["mean_velocity", "u_rms", "velocity"], maps=1)
    p.save()
    print("Pre-processing:", p.preprocess().status)
    p.set_piv(engine="TracTrac")
    p.save()
    print("PTV (TracTrac):", p.run_piv().status)
    p.set_piv(engine="PaIRS-UniNa", windows=PIV_WINDOWS, overlap=50)
    p.save()
    print("PIV (PaIRS-UniNa):", p.run_piv().status)
    print("Post-processing:", p.postprocess(overwrite=True))
    return folder


################################################################################
# Capture (in a process of its own: scratch preferences, set before velocihub is imported)

# Size of the main window (px at 96 dpi; scaled with the display)
WINDOW = (1600, 960)
# Width of the panel Projects (px at 96 dpi)
WORKSPACE_WIDTH = 330
# Pause before a capture (s): the window drawn by the system
SETTLE = 0.6


class Shooter:
    """The captures of the windows of the interface into docs/images."""

    def __init__(self, root, only=None):
        self.root, self.only = root, only
        self.taken = []

    def wanted(self, name):
        return self.only is None or name in self.only

    def settle(self, seconds=SETTLE):
        import time
        end = time.time() + seconds
        while time.time() < end:
            self.root.update()
            time.sleep(0.05)

    def frame_box(self, window):
        """The box of a top-level window on the screen with its frame and title bar (Windows), its client area
        otherwise."""
        window.update_idletasks()
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            hwnd = int(window.wm_frame(), 16)
            rect = wintypes.RECT()
            # DWMWA_EXTENDED_FRAME_BOUNDS: the visible frame, without the invisible resize borders
            if ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)) == 0:
                return rect.left, rect.top, rect.right, rect.bottom
        return self.widget_box(window)

    @staticmethod
    def widget_box(widget):
        x, y = widget.winfo_rootx(), widget.winfo_rooty()
        return x, y, x + widget.winfo_width(), y + widget.winfo_height()

    def grab(self, name, window=None, widget=None, settle=SETTLE):
        """Capture a top-level window (with its frame) or a widget (its area) into docs/images/name.png."""
        from PIL import ImageGrab
        if not self.wanted(name):
            return
        top = window if window is not None else widget.winfo_toplevel()
        top.deiconify()
        top.lift()
        top.attributes("-topmost", True)
        self.settle(settle)
        box = self.frame_box(window) if widget is None else self.widget_box(widget)
        image = ImageGrab.grab(bbox=box)
        top.attributes("-topmost", False)
        path = os.path.join(IMAGES, name + ".png")
        image.save(path, optimize=True)
        self.taken.append(path)
        print(f"{name}.png  {image.size[0]} x {image.size[1]}")


def capture(folder, only=None):
    """The pictures of the interface opened on the example project (folder), with scratch preferences (set before
    velocihub is imported: its preferences file is read at import). The project file of the example is put back as it
    was at the end (the interface saves its state in it)."""
    if "velocihub" in sys.modules:
        raise RuntimeError("capture() needs a process where velocihub isn't imported yet (scratch preferences)")
    home = tempfile.mkdtemp(prefix="velocihub_docs_home_")
    os.environ["USERPROFILE"] = os.environ["HOME"] = home
    project_file = os.path.join(folder, "VelociHub_project.json")
    with open(project_file, "rb") as f:
        saved_project = f.read()
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.OleDLL("shcore").SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    sys.path.insert(0, REPO)
    import tkinter as tk
    from tkinter import messagebox

    from velocihub import prefs
    from velocihub.gui import app as app_module, widgets
    from velocihub.workspace import WORKSPACE_FILE, Workspace

    os.makedirs(IMAGES, exist_ok=True)
    # No message box stops the script: they are printed
    for name in ("showinfo", "showwarning", "showerror"):
        setattr(messagebox, name, lambda title, message, **k: print(f"[{title}] {message}"))
    messagebox.askyesno = messagebox.askokcancel = lambda *a, **k: True
    widgets.QuestionDialog.run = lambda dialog: None
    prefs.setpref("workspace_file", None)

    root = tk.Tk()
    app = app_module.MainWindow(root)
    win = app.window
    scale = root.winfo_fpixels("1i") / 96
    shoot = Shooter(root, only)
    shoot.settle(1.0)

    # The workspace: the folder of the example, its project opened
    example = os.path.dirname(folder)
    ws = Workspace(os.path.join(example, WORKSPACE_FILE))
    ws.add_root(example)
    ws.save()
    app.workspace.load(ws.file)
    shoot.settle(0.5)
    project = next(p for p in app.workspace.ws.projects if os.path.normcase(p.folder) == os.path.normcase(folder))
    app.workspace.open(project)
    shoot.settle(2.0)
    # The size of the window (maximized when it is shown)
    win.state("normal")
    win.geometry(f"{int(WINDOW[0] * scale)}x{int(WINDOW[1] * scale)}+10+10")
    shoot.settle(1.0)
    app.workspace.master.sashpos(0, int(WORKSPACE_WIDTH * scale))
    shoot.settle(0.5)

    try:
        for step in STEPS:
            step(app, shoot)
    finally:
        root.destroy()
        with open(project_file, "wb") as f:
            f.write(saved_project)
        shutil.rmtree(home, ignore_errors=True)
    print(f"{len(shoot.taken)} pictures in {IMAGES}")


# The image (pre-processing) and the pair (velocimetry) shown, 0-based
IMAGE = 60
# The figures of the four tiles of the post-processing: PIV, then PTV
POST_VIEWS = ("velocity", "mean_velocity", "u_rms", "vorticity")
# The isocontour shown (fraction of the colour range of its map)
CONTOUR = 0.7
PTV_VIEWS = ("ptv_tracks", "ptv_particles", "ptv_track_length", "mean_acceleration")


def step_preprocessing(app, shoot):
    """The page Pre-processing: the image with the comparison before / after the recipe, a mode, the spectrum."""
    app.show_page("pre")
    page = app.page("pre")
    shoot.settle(2.0)
    images = page.figure_main.master  # The image and the modes side by side
    images.sashpos(0, int(images.winfo_width() * 0.6))
    page.callback_arrows(0, "main", IMAGE)
    page.figure_main.set_compare_value(0.5)
    page.set_compare(0.5)
    shoot.settle(2.0)
    shoot.grab("preprocessing", app.window)
    shoot.grab("preprocessing_panel", widget=page.panel)


def step_workspace(app, shoot):
    """The panel Projects (the workspace)."""
    shoot.grab("workspace", widget=app.workspace)


def piv_engine(app, name):
    page = app.page("piv")
    page.panel.engine_name.set(name)
    page.panel.engine_selected()


def step_velocimetry(app, shoot):
    """The page Velocimetry: the PIV (PaIRS-UniNa) and the PTV (TracTrac) of the pair shown, the quality of the
    pairs."""
    app.show_page("piv")
    page = app.page("piv")
    page.show_pair(IMAGE)
    shoot.settle(2.0)
    shoot.grab("velocimetry_piv", app.window)
    shoot.grab("velocimetry_panel", widget=page.panel)
    page.view_selector.show("displacement")
    page.view_selected("displacement")
    shoot.settle(1.5)
    shoot.grab("velocimetry_piv_displacement", widget=page.figure_pair)
    piv_engine(app, "TracTrac")
    shoot.settle(1.5)
    page.show_pair(IMAGE)
    shoot.settle(2.0)
    shoot.grab("velocimetry_ptv", app.window)
    page.view_selector.show("image")
    page.view_selected("image")


def post_views(page, views):
    for tile, key in zip(page.tiles, views):
        page.view_selected(tile, key)


def step_postprocessing(app, shoot):
    """The page Post-processing: four tiles on the PIV, a comparison with a wiper and an isocontour, the PTV, and its
    windows (Quantities, video)."""
    from velocihub.gui.quantities import QuantitiesDialog
    from velocihub.settings import export_spec

    app.show_page("post")
    page = app.page("post")
    project = app.workspace.ws.projects[0]
    app.show_results(project, "PaIRS")
    shoot.settle(3.0)
    post_views(page, POST_VIEWS)
    page.slide_to(IMAGE)
    shoot.settle(3.0)
    shoot.grab("postprocessing", app.window)
    shoot.grab("postprocessing_panel", widget=page.panel)
    # A tile comparing two maps with its wiper, another one with an isocontour
    tile = page.tiles[0]
    page.view2_selected(tile, "vorticity")
    tile.set_compare_value(0.45)
    page.wipe(tile, 0.45)
    contour = page.tiles[2]  # RMS u, bottom left: its isocontour at CONTOUR of its colour range
    contour.contour_var.set(True)
    page.contour_toggled(contour)
    contour.contour_slider.set(CONTOUR)
    page.contour_slid(contour, CONTOUR)
    # Each one enlarged (the dividers of the grid of tiles moved), then put back
    row, grid, row2 = tile.master, tile.master.master, contour.master
    before = row.sashpos(0), grid.sashpos(0), row2.sashpos(0)
    grid.sashpos(0, int(grid.winfo_height() * 0.8))
    row.sashpos(0, int(row.winfo_width() * 0.68))
    shoot.settle(2.0)
    shoot.grab("postprocessing_compare", widget=tile)
    grid.sashpos(0, int(grid.winfo_height() * 0.2))
    row2.sashpos(0, int(row2.winfo_width() * 0.68))
    shoot.settle(2.0)
    shoot.grab("postprocessing_contour", widget=contour)
    grid.sashpos(0, before[1])
    row.sashpos(0, before[0])
    row2.sashpos(0, before[2])
    page.view2_selected(tile, "none")
    contour.contour_var.set(False)
    page.contour_toggled(contour)
    # The window Quantities and the video of a tile
    if shoot.wanted("quantities"):
        dialog = QuantitiesDialog(app.parent(), export_spec(page.export_spec()), rolling=page.panel.rolling.get(),
                                  post=page.post, jet=page.jet_theory_on())
        dialog.show()
        shoot.grab("quantities", dialog, settle=1.5)
        dialog.grab_release()
        dialog.destroy()
    if shoot.wanted("video"):
        dialog = page.export_video(page.tiles[0])
        shoot.grab("video", dialog, settle=1.0)
        dialog.grab_release()
        dialog.destroy()
    # The PTV
    app.show_results(project, "TracTrac")
    shoot.settle(3.0)
    post_views(page, PTV_VIEWS)
    shoot.settle(3.0)
    shoot.grab("postprocessing_ptv", app.window)
    app.show_results(project, "PaIRS")
    post_views(page, POST_VIEWS)
    shoot.settle(2.0)


def step_batch(app, shoot):
    """The window Batch of the post-processing."""
    from velocihub.gui.batch_dialog import BatchDialog

    if not shoot.wanted("batch"):
        return
    app.show_page("post")
    dialog = BatchDialog(app.parent(), app, app.batch_step("postprocessing"))
    dialog.show()
    shoot.settle(1.0)
    top = dialog.tree.get_children()[0]
    project = dialog.tree.get_children(top)[0]
    dialog.toggle(project)  # Ticked: run
    dialog.tree.selection_set(project)
    dialog.tree.focus(project)
    dialog.refresh()  # Its settings in the form
    shoot.grab("batch", dialog, settle=2.0)
    dialog.grab_release()
    dialog.destroy()


STEPS = [step_preprocessing, step_workspace, step_velocimetry, step_postprocessing, step_batch]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--example", default=DEFAULT_EXAMPLE, help="folder of the example project")
    parser.add_argument("--prepare-only", action="store_true", help="only make the example project")
    parser.add_argument("--force", action="store_true", help="make the example project again")
    parser.add_argument("--only", help="the pictures taken, by name (comma-separated; all by default)")
    parser.add_argument("--capture", metavar="FOLDER", help=argparse.SUPPRESS)  # The capture process
    args = parser.parse_args()
    only = set(args.only.split(",")) if args.only else None
    if args.capture:
        capture(args.capture, only)
        return
    folder = prepare(os.path.abspath(args.example), args.force)
    if args.prepare_only:
        return
    # The capture in a new process: scratch preferences (velocihub, imported by prepare, has read yours)
    import subprocess
    command = [sys.executable, os.path.abspath(__file__), "--capture", folder] + (["--only", args.only] if only else [])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
