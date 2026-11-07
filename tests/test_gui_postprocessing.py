"""The page of the post-processing (gui/postprocessing_page.py) in the interface: its side panel, the results of the
project (whatever the page PIV/PTV), the POD modes (slider, tiles, sum), the quantities greyed out that can't be drawn,
the export overwriting the previous one or not, the PIV and PTV results of a project (panel Projects) and a new
workspace unloading them."""

import os

import pytest

pytest.importorskip("tkinter")

from velocihub import imaging, lagrangian, piv  # noqa: E402
from velocihub.workspace import WORKSPACE_FILE, Workspace  # noqa: E402


def open_project(gui, root, path):
    """The projects under root in a new workspace, the project path opened and the page shown with the camera."""
    ws = Workspace(os.path.join(root, WORKSPACE_FILE))
    ws.add_root(root)
    ws.save()
    app = gui.app
    app.workspace.load(ws.file)
    gui.root.update()
    projects = {p.path: p for p in app.workspace.ws.projects}
    app.workspace.open(projects[path])
    gui.root.update()
    app.show_page("post")
    page = app.page("post")
    page.panel.space.set("physical")
    page.panel.camera["fov_width"].set("80")
    page.panel.camera["frame_rate"].set("100")
    page.camera_changed()
    gui.root.update()
    return page, projects


def descendants(widget):
    out = [widget]
    for child in widget.winfo_children():
        out += descendants(child)
    return out


def title(box):
    widget = box.cget("labelwidget")
    return str(box.nametowidget(widget).winfo_children()[0].cget("text")) if widget else str(box.cget("text"))


def menu_state(tile, group):
    menu = tile.selector.menu
    return next(menu.entrycget(i, "state") for i in range(menu.index("end") + 1) if menu.entrycget(i, "label") == group)


def release(gui, slider, first, last, handle):
    """The handles of a two-sided slider at first and last, the handle released."""
    slider.set(first, last)
    slider.active = handle
    slider.release()
    gui.root.update()


def test_side_panel(gui):
    """Peak locking, Jet potential core and POD inside the box Miscellaneous, the POD modes and the text in POD."""
    panel = gui.app.page("post").panel
    top = [title(w) for w in panel.inner.winfo_children() if w.winfo_class() == "TLabelframe"]
    assert top[-1] == "Miscellaneous" and "Peak locking" not in top and "Jet potential core" not in top
    inside = [w for w in panel.misc.winfo_children() if w.winfo_class() == "TLabelframe"]
    assert [title(w) for w in inside] == ["Peak locking", "Jet potential core", "POD"]
    pod = descendants(inside[-1])
    assert panel.pod_slider in pod and panel.pod_label in pod and all(b in pod for b in panel.pod_buttons)


def test_results_whatever_the_piv_page(gui, copy, tmp_path):
    """A project with PIV results only, the PTV chosen in the page PIV/PTV: its PIV results post-processed."""
    copy("piv", "campaign", "m")
    piv_page = gui.app.page("piv")
    page, projects = open_project(gui, str(tmp_path / "campaign"), "m")
    piv_page.panel.engine_name.set("TracTrac")
    piv_page.panel.engine_selected()
    gui.root.update()
    assert piv_page.output_folder().endswith(piv.PTV_FOLDER)
    page.loaded = None
    page.shown()
    gui.root.update()
    assert os.path.basename(page.maps_folder()) == piv.PIV_FOLDER and page.post is not None
    assert page.panel.export_button.instate(["!disabled"]) and page.panel.batch_button.instate(["!disabled"])
    piv_page.panel.engine_name.set("PaIRS-UniNa")
    piv_page.panel.engine_selected()


def test_pod_modes(gui, copy, tmp_path):
    """The two-sided slider of the POD modes: up to the modes evaluated, the tile showing the mode of the handle
    released (only the tiles of the POD drawn again); summed: the background of the image of the map browsed."""
    copy("piv", "campaign", "m")
    page, _ = open_project(gui, str(tmp_path / "campaign"), "m")
    tiles = page.tiles
    for tile, view in zip(tiles, ("raw_image", "pod_mode", "pod_spectrum", "velocity")):
        page.view_selected(tile, view)
    gui.root.update()
    slider = page.panel.pod_slider
    assert slider.enabled and slider.hi == 3
    drawn, original = [], page.post.plot
    page.post.plot = lambda key, *a, **k: (drawn.append(key), original(key, *a, **k))[1]
    release(gui, slider, 2, 3, 1)
    assert page.post.pod_range() == (2, 3) and sorted(drawn) == ["pod_mode", "pod_spectrum"]
    assert r"\phi_{3}" in tiles[1].fig.axes[0].get_title()
    release(gui, slider, 2, 3, 0)
    assert r"\phi_{2}" in tiles[1].fig.axes[0].get_title()
    page.post.plot = original
    page.panel.pod_combine.set(True)
    page.apply_settings()
    gui.root.update()
    assert page.follows_maps(tiles[1]) and "sum of modes 2 to 3" in tiles[1].fig.axes[0].get_title()
    page.show_map(4)
    gui.root.update()
    assert "img_004.tif" in tiles[1].fig.axes[0].get_title()
    page.panel.pod_combine.set(False)
    page.apply_settings()


def test_unavailable_greyed(gui, copy, tmp_path):
    """Without raw images nor POD: those of the menus greyed out (the reason in the box), the PTV too (PIV results);
    the image of the PIV available."""
    folder = copy("piv", "campaign", "m")
    os.remove(os.path.join(folder, "PreProcessing", "pod.npz"))
    for name in os.listdir(folder):
        if name.endswith(".tif"):
            os.remove(os.path.join(folder, name))
    page, _ = open_project(gui, str(tmp_path / "campaign"), "m")
    tile = page.tiles[0]
    page.show_selectors(tile)
    assert menu_state(tile, imaging.GROUPS[1][0]) == "disabled" and menu_state(tile, lagrangian.GROUP[0]) == "disabled"
    assert menu_state(tile, imaging.GROUPS[0][0]) == "normal"  # The image of the PIV still there
    assert "No POD" in page.panel.pod_label["text"] and not page.panel.pod_slider.enabled


def test_export_overwrite(gui, copy, tmp_path):
    """Results already in the export folder: asked once; Cancel exports nothing, Overwrite all replaces them (the
    other files kept)."""
    copy("piv", "campaign", "m")
    page, _ = open_project(gui, str(tmp_path / "campaign"), "m")
    page.export_choice = dict(page.export_choice, figures=["mean_velocity", "u_rms"], videos=[])
    del gui.asked[:]
    page.export()
    folder = page.export_folder()
    assert not gui.asked and any(n.endswith("URms.png") for n in os.listdir(folder))
    open(os.path.join(folder, "notes.txt"), "w").close()
    page.export_choice = dict(page.export_choice, figures=["mean_velocity"])
    before = sorted(os.listdir(folder))
    gui.answers[:] = ["Cancel"]
    page.export()
    assert len(gui.asked) == 1 and sorted(os.listdir(folder)) == before
    gui.answers[:] = ["Overwrite all"]
    page.export()
    names = os.listdir(folder)
    assert "notes.txt" in names and not any(n.endswith("URms.png") for n in names)


def test_piv_and_ptv_results(gui, copy, tmp_path):
    """A project with PIV and PTV results: a row for each (PIV+PTV); each opened in the post-processing, exported to
    its own folder, the quantities of the PTV greyed out for the PIV."""
    copy("both", "campaign", "both")
    page, projects = open_project(gui, str(tmp_path / "campaign"), "both")
    panel = gui.app.workspace
    rows = sorted(item[2] for item in panel.items.values() if item[0] == "results")
    assert rows == ["PIV", "PTV"]
    panel.open(projects["both"], "PTV")
    gui.root.update()
    page.shown()
    gui.root.update()
    assert page.post is not None and page.post.ptv()
    assert os.path.basename(page.export_folder()) == "PostProcessing_PTV"
    tile = page.tiles[0]
    page.show_selectors(tile)
    assert menu_state(tile, lagrangian.GROUP[0]) == "normal"
    panel.open(projects["both"], "PIV")
    gui.root.update()
    page.shown()
    gui.root.update()
    page.show_selectors(tile)
    assert not page.post.ptv() and os.path.basename(page.export_folder()) == "PostProcessing"
    assert menu_state(tile, lagrangian.GROUP[0]) == "disabled"


def test_new_workspace(gui, copy, tmp_path):
    """A new workspace unloads the project open (images, results, vector fields); refused while the PIV runs."""
    copy("piv", "campaign", "m")
    page, _ = open_project(gui, str(tmp_path / "campaign"), "m")
    app, panel = gui.app, gui.app.workspace
    assert page.post is not None and app.project_folder()
    app.page("piv").run = object()
    panel.new()
    assert gui.boxes and "running" in gui.boxes[-1] and page.post is not None
    app.page("piv").run = None
    panel.new()
    gui.root.update()
    assert not panel.ws.projects and app.project_folder() is None and not app.source.Check_imsel
    assert page.post is None and page.maps_folder() is None
    assert page.panel.export_button.instate(["disabled"]) and page.panel.batch_button.instate(["disabled"])
