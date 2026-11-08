"""The page of the post-processing (gui/postprocessing_page.py) in the interface: its side panel, the results of the
project (whatever the page PIV/PTV), the POD modes (slider, tiles, sum), the quantities greyed out that can't be drawn,
the export overwriting the previous one or not, the PIV and PTV results of a project (panel Projects) and a new
workspace unloading them."""

import os
from dataclasses import replace

import pytest

pytest.importorskip("tkinter")

from velocihub import imaging, lagrangian, piv  # noqa: E402
from velocihub.jet import JET_GROUP  # noqa: E402
from velocihub.postprocessing import FIGURE_TITLES, PIV_GROUP  # noqa: E402
from velocihub.workspace import WORKSPACE_FILE, Workspace, key  # noqa: E402


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


def menu_entries(menu, path=()):
    """The entries of a menu and of its submenus: {label: (state, path of the submenus above it)}."""
    out = {}
    for i in range(menu.index("end") + 1):
        if menu.type(i) == "separator":
            continue
        label = menu.entrycget(i, "label")
        out[label] = (menu.entrycget(i, "state"), path)
        if menu.type(i) == "cascade":
            out.update(menu_entries(menu.nametowidget(menu.entrycget(i, "menu")), path + (label,)))
    return out


def menu_state(tile, group):
    """The state of a group of the menu of the figures of a tile, wherever it is (in a submenu too)."""
    return menu_entries(tile.selector.menu)[group][0]


def release(gui, slider, first, last, handle):
    """The handles of a two-sided slider at first and last, the handle released."""
    slider.set(first, last)
    slider.active = handle
    slider.release()
    gui.root.update()


def test_side_panel(gui):
    """Peak locking and POD inside the box Miscellaneous, the POD modes and the text in POD; the jet theory in the box
    Add-ons below it, its settings hidden when it is off."""
    panel = gui.app.page("post").panel
    top = [title(w) for w in panel.inner.winfo_children() if w.winfo_class() == "TLabelframe"]
    assert top[-2:] == ["Miscellaneous", "Add-ons"] and "Peak locking" not in top and "Jet theory (add-on)" not in top
    inside = [w for w in panel.misc.winfo_children() if w.winfo_class() == "TLabelframe"]
    assert [title(w) for w in inside] == ["Peak locking", "POD"]
    pod = descendants(inside[-1])
    assert panel.pod_slider in pod and panel.pod_label in pod and all(b in pod for b in panel.pod_buttons)
    assert [title(w) for w in panel.addons.winfo_children() if w.winfo_class() == "TLabelframe"] == [
        "Jet theory (add-on)"]
    jet = panel.jet_theory.get()
    for on in (False, True):
        panel.jet_theory.set(on)
        panel.update_core_mode()
        assert bool(panel.jet_settings.winfo_manager()) == on
        assert all(e in descendants(panel.jet_settings) for e in panel.jet_entries.values())
    panel.jet_theory.set(jet)
    panel.update_core_mode()


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


def test_tile_labels():
    """The labels of the menu buttons of the tiles: the symbols in plain text without %, _ nor ^, words capitalised,
    abbreviations with a dot (Mean, Med., RMS, Skew., Kurt.), the integral scales by name, at most 14 characters."""
    from velocihub.gui.postprocessing_page import short_title
    from velocihub.postprocessing import FIGURE_TITLES
    expected = {"mean_vorticity": "Mean ω", "peak_ratio": "Med. S/N", "u_rms": "RMS u", "skewness_u": "Skew. u",
                "kurtosis_v": "Kurt. v", "reynolds_stress": "τ", "reynolds_gradient_abs": "|∂τ/∂y|",
                "turbulence_intensity": "Tu", "turbulence_intensity_map": "Tu'", "displacement_x": "dx",
                "integral_time": "Lt", "integral_length_vy": "Ly v", "ke_length": "L k-ε",
                "velocity_gradient": "∂(Mean u)/∂y", "convergence": "Convergence", "ptv_yield": "Part. tracked"}
    assert {k: short_title(k) for k in expected} == expected
    labels = [short_title(k) for k in FIGURE_TITLES]
    assert all(len(label) <= 14 and not set(label) & set(r"\%_^{}" + "̄") for label in labels), labels


def test_map_cursor(gui, copy, tmp_path):
    """The x-y cursor of a map: the value of the grid point under the mouse in the box (named as the axes and the
    colour bar name them) and on the colour
    bar (dashed line); a left click freezes it (the mouse then moves nothing), another one releases it; none outside
    the field."""
    import types
    import numpy as np
    from velocihub.gui.cursor import nearest
    copy("piv", "campaign", "m")
    page, _ = open_project(gui, str(tmp_path / "campaign"), "m")
    tile = page.tiles[0]
    tile.canvas.draw()
    gui.root.update()
    c = tile.cursor
    assert c.maps and c.ax is c.maps[0].axes
    X, Y, F = c.maps[0].field
    xs, ys = (X[0], Y[:, 0]) if np.ndim(X) == 2 else (np.asarray(X), np.asarray(Y))  # A grid, or an image
    j, i = np.argwhere(np.isfinite(F))[len(np.argwhere(np.isfinite(F))) // 2]

    def event(x, y, button=None):
        return types.SimpleNamespace(button=button, inaxes=c.ax, xdata=x, ydata=y)
    c.moved(event(xs[i], ys[j]))
    x, y, m, value = c.values()
    # Named as the axes and the colour bar name them, 3 significant digits
    lines = c.box.get_text().splitlines()
    assert value == F[j, i] and lines[2].startswith(c.maps[0].colorbar.ax.get_ylabel().split("$ ")[0])
    assert f" = {value:.3g}" in lines[2] and lines[0].startswith(c.ax.get_xlabel().split("$ ")[0])
    assert c.marks[id(m)].get_ydata()[0] == value
    c.pressed(event(xs[i], ys[j], 1))
    assert c.frozen
    c.moved(event(xs[0], ys[0]))
    assert c.point == (xs[i], ys[j])
    c.pressed(event(xs[0], ys[0], 1))
    assert not c.frozen
    assert nearest((X, Y, F), float(np.nanmax(xs)) * 10 + 100, ys[j]) is None


def test_camera_entered_once(gui, copy, tmp_path):
    """The camera settings entered (the height of the field of view: non-square pixels) read the vector fields once;
    the entry left again without a change (its <FocusOut>, as the progress window takes the focus) reads nothing."""
    copy("piv", "campaign", "m")
    page, _ = open_project(gui, str(tmp_path / "campaign"), "m")
    reads, original = [], page.load_maps
    page.load_maps = lambda: (reads.append(1), original())[1]
    page.panel.camera["fov_height"].set("40")
    page.camera_changed()
    gui.root.update()
    page.camera_changed()
    page.camera_changed()
    gui.root.update()
    assert len(reads) == 1 and page.post is not None
    width, height = page.post.fields.settings["image_px"]
    assert abs(page.post.fields.scale_y - 40 / height) < 1e-9 and abs(page.post.fields.scale - 80 / width) < 1e-9
    page.panel.camera["fov_height"].set("")
    page.camera_changed()
    assert len(reads) == 2 and page.post.fields.scale_y == page.post.fields.scale
    page.load_maps = original


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
    # The images and the POD in Pre-Processing, the quality of the PIV in PIV (PIV results), the moments in Velocity,
    # the jet theory in Add-ons, greyed out while the add-on is off
    entries = menu_entries(tile.selector.menu)
    assert entries[imaging.GROUPS[0][0]][1] == entries[imaging.GROUPS[1][0]][1] == ("Pre-Processing",)
    assert entries["Pre-Processing"][0] == "normal" and menu_state(tile, PIV_GROUP[0]) == "normal"
    assert all(entries[FIGURE_TITLES[k]][1] == (PIV_GROUP[0],) for k in PIV_GROUP[1])
    assert entries[FIGURE_TITLES["skewness_u"]][1] == entries[FIGURE_TITLES["kurtosis_v"]][1] == ("Velocity",)
    assert entries[JET_GROUP[0]][1] == ("Add-ons",)
    page.post.settings = replace(page.post.settings, jet_theory=False)
    page.show_selectors(tile)
    assert menu_state(tile, "Add-ons") == "disabled"
    page.post.settings = replace(page.post.settings, jet_theory=True)
    page.show_selectors(tile)
    assert menu_state(tile, "Add-ons") == menu_state(tile, JET_GROUP[0]) == "normal"
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
    """A project with the results of two pipelines: a row for each (column Velocimetry: PaIRS, TracTrac); each opened
    in the post-processing, exported to the subfolder of its pipeline, the quantities of the PTV greyed out for the
    PIV."""
    copy("both", "campaign", "both")
    page, projects = open_project(gui, str(tmp_path / "campaign"), "both")
    panel = gui.app.workspace
    rows = sorted(item[2] for item in panel.items.values() if item[0] == "results")
    assert rows == ["PaIRS", "TracTrac"]
    assert panel.tree.heading("results")["text"] == "Velocimetry"
    assert panel.tree.set("project|" + key(projects["both"].folder), "results") == "PaIRS, TracTrac"
    panel.open(projects["both"], "TracTrac")
    gui.root.update()
    page.shown()
    gui.root.update()
    assert page.post is not None and page.post.ptv()
    assert page.export_folder().endswith(os.path.join("PostProcessing", "TracTrac"))
    tile = page.tiles[0]
    page.show_selectors(tile)
    assert menu_state(tile, lagrangian.GROUP[0]) == "normal"
    panel.open(projects["both"], "PaIRS")
    gui.root.update()
    page.shown()
    gui.root.update()
    page.show_selectors(tile)
    assert not page.post.ptv() and page.export_folder().endswith(os.path.join("PostProcessing", "PaIRS"))
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
