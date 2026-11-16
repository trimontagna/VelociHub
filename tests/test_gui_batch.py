"""The windows Batch (gui/batch_dialog.py) of the three steps: the settings of a node in the side panel of the page
(the settings that differ in bold, each with its ↺; a filter of the pre-processing on its own; another PIV software),
the projects whose settings differ from those of the open project in bold with their ↺, the stitchings (A.1, A.2;
the settings of the stitch reference from the menu of a project), the quantities exported of the post-processing
(Check all, compared with the open project, greyed per node), the jobs and the results of a previous export
overwritten."""

import os

import pytest

pytest.importorskip("tkinter")

from velocihub import imaging, piv  # noqa: E402
from velocihub.gui import batch_dialog  # noqa: E402
from velocihub.gui.piv_page import BatchPIVForm  # noqa: E402
from velocihub.gui.postprocessing_page import BatchPostForm  # noqa: E402
from velocihub.gui.preprocessing_page import BatchPreForm  # noqa: E402
from velocihub.gui.quantities import export_spec  # noqa: E402
from velocihub.workspace import WORKSPACE_FILE, Workspace, key  # noqa: E402


@pytest.fixture
def campaign(gui, copy, tmp_path):
    """Three projects a, b, c (PIV results) in a workspace, a open with its camera; batch(page) opens the window
    Batch of a page (its dialog returned, not run)."""
    for name in "abc":
        copy("piv", "campaign", name)
    root = os.path.join(str(tmp_path), "campaign")
    ws = Workspace(os.path.join(root, WORKSPACE_FILE))
    ws.add_root(root)
    ws.save()
    app = gui.app
    app.workspace.load(ws.file)
    gui.root.update()
    projects = {p.path: p for p in app.workspace.ws.projects}
    app.workspace.open(projects["a"])
    gui.root.update()
    post = app.page("post")
    post.panel.space.set("physical")
    post.panel.camera["fov_width"].set("80")
    post.panel.camera["frame_rate"].set("100")
    post.camera_changed()
    dialogs = []
    batch_dialog.BatchDialog.run = lambda self: dialogs.append(self) or None

    def batch(page):
        app.page(page).batch()
        return dialogs[-1]
    state = type("Campaign", (), {})()
    state.root, state.projects, state.batch, state.gui = root, projects, batch, gui
    state.node = lambda name: "project|" + key(os.path.join(root, name))
    state.folder = batch_dialog.BatchDialog.node_id(root)
    yield state
    for d in dialogs:
        if d.winfo_exists():
            d.destroy()


def select(d, iid):
    d.tree.selection_set(iid)
    d.tree.focus(iid)
    d.show_node()
    d.update()


def row_button(form, label):
    """The button ↺ next to the label of a setting (path, or (path, choice)), None if there is none."""
    widget = form.labels[label][0]
    for button in form.unify_buttons:
        if button.winfo_parent() != widget.winfo_parent():
            continue
        if widget.winfo_manager() == "grid" and button.winfo_manager() == "grid":
            if button.grid_info()["row"] == widget.grid_info()["row"]:
                return button
        elif button.winfo_manager() == "place" and button.place_info().get("in") == str(widget):
            return button
        elif widget.winfo_manager() == "pack" and button.winfo_manager() == "pack":
            return button
    return None


def differs(d, iid):
    """A node of the tree in bold with its ↺: settings that differ from those of the open project."""
    return "differs" in d.tree.item(iid, "tags") and d.tree.set(iid, "reset") == "↺"


def bold(form, label):
    widget = form.labels[label][0]
    return ("bold" in str(widget.cget("font")) if widget.winfo_class() in ("TLabel", "Checkbutton")
            else str(widget.cget("style")) == "Bold.TCheckbutton")


def test_postprocessing(campaign):
    """The side panel of the page as the settings of a node: a change for that project only, an invalid one refused;
    at the folder, the setting that differs in bold with its ↺; the reset greyed when the settings are those of the
    open project; the POD modes up to those evaluated."""
    d = campaign.batch("post")
    form, settings_of, b = d.form, d.settings_of, campaign.projects["b"]
    assert isinstance(form, BatchPostForm)
    select(d, campaign.node("b"))
    assert not differs(d, campaign.node("b")) and not differs(d, campaign.folder)
    assert "settings" not in d.tree["columns"] and d.node_tab.cget("text") == "Settings: b"
    assert (form.panel.pod_slider.lo, form.panel.pod_slider.hi) == (1, 3)
    form.panel.vars["outlier_k"].set("3")
    form.commit()
    assert settings_of(b)["settings"]["outlier_k"] == 3.0
    assert settings_of(campaign.projects["c"])["settings"].get("outlier_k", 4.0) == 4.0
    assert differs(d, campaign.node("b")) and differs(d, campaign.folder) and not differs(d, campaign.node("c"))
    # The name in bold only (Unicode bold letters: the font of a row is that of all its columns), the columns plain
    assert d.tree.item(campaign.node("b"), "text") == batch_dialog.bold_text("b") != "b"
    assert d.tree.item(campaign.node("c"), "text") == "c" and d.names[campaign.node("b")] == "b"
    form.panel.vars["outlier_k"].set("abc")
    form.commit()
    assert settings_of(b)["settings"]["outlier_k"] == 3.0 and form.panel.vars["outlier_k"].get() == "3" and d.node_note["text"]
    select(d, campaign.folder)
    assert bold(form, ("settings", "outlier_k")) and not bold(form, ("settings", "skew_zero"))
    assert row_button(form, ("settings", "skew_zero")) is None
    row_button(form, ("settings", "outlier_k")).invoke()
    d.update()
    assert len({settings_of(p)["settings"].get("outlier_k", 4.0) for p in campaign.projects.values()}) == 1
    assert not differs(d, campaign.folder)


def test_stitch_reference_and_quantities(campaign):
    """The stitching a → b: A.1 and A.2 in its column; from the menu of b, all the settings of A.1 (its quantities
    exported too, its camera kept); the quantities box: Check all, Uncheck all, compared with the quantities of the
    open project (bold, ↺), no reset button."""
    d = campaign.batch("post")
    form, settings_of, a, b = d.form, d.settings_of, campaign.projects["a"], campaign.projects["b"]
    q = d.panels[0][1]
    assert not hasattr(q, "reset_button") and not hasattr(q, "same_button")
    d.step.options.chains = [[(a.root, "a"), (a.root, "b")]]
    d.check_all(True)
    d.refresh()
    assert d.tree.set(campaign.node("a"), "extra") == "A.1", d.step.options.tip(d, campaign.node("a")) and d.tree.set(campaign.node("b"), "extra") == "A.2"
    assert "A.2, stitched to A.1" in d.step.options.tip(d, campaign.node("b"))

    def take():
        items = d.step.options.context_items(d, campaign.node("b"))
        assert [label for label, _ in items] == ["Take settings of A.1", "Align Group A"]
        return items[0][1]
    select(d, campaign.node("b"))
    assert take() is None  # Already the same
    form.panel.vars["skew_zero"].set("33")
    form.commit()
    form.panel.camera["x_offset"].set("60")
    form.commit()
    q.check_buttons[1].invoke()
    d.update()
    assert export_spec(settings_of(b)["export"])["choice"]["figures"] == []
    take()()
    d.update()
    assert settings_of(b)["settings"] == settings_of(a)["settings"]
    assert settings_of(b)["camera"]["x_offset_mm"] == 60.0
    assert export_spec(settings_of(b)["export"]) == export_spec(settings_of(a)["export"])
    # Stitched for each results set that both have and export (PaIRS), not without one
    units = d.step.options.job(d.chosen())["units"]
    assert [u["stitching"]["pipeline"] for u in units if "stitching" in u] == ["PaIRS"]
    assert d.apply_to([b], lambda settings: settings.update(results=["TracTrac"]))
    d.refresh()
    assert d.tree.set(campaign.node("b"), "extra") == "A.2 ✗" and "results set" in d.step.options.tip(d, campaign.node("b"))
    assert d.apply_to([b], lambda settings: settings.update(results=["PaIRS", "TracTrac"]))
    d.refresh()
    # A parameter of a quantity of c that differs from the open project: bold with its ↺, which gives it back
    select(d, campaign.node("c"))
    q.tree.selection_set("velocity")
    q.tree.focus("velocity")
    q.select()
    q.contour.set("2")
    d.update()
    assert export_spec(settings_of(campaign.projects["c"])["export"])["params"]["contours"]["velocity"] == 2.0
    assert "differs" in q.tree.item("velocity", "tags") and differs(d, campaign.node("c"))
    q.reset_quantity("velocity")
    d.update()
    assert "velocity" not in export_spec(settings_of(campaign.projects["c"])["export"])["params"]["contours"]
    assert not differs(d, campaign.node("c"))


def test_preprocessing(campaign):
    """The pre-processing of a node in the side panel of its page: a filter on its own (only it in bold at the
    folder, its ↺), its settings, the background removal; the reset to the open project."""
    d = campaign.batch("pre")
    form, settings_of, b = d.form, d.settings_of, campaign.projects["b"]
    assert isinstance(form, BatchPreForm) and not hasattr(form.panel, "new_mask_button")
    select(d, campaign.node("b"))
    assert not differs(d, campaign.node("b"))
    panel = form.panel
    before = "minmax" in (settings_of(b)["contrast"].get("filters") or [])
    panel.checks["minmax"].set(not before)
    panel.contrast_changed()
    assert ("minmax" in settings_of(b)["contrast"]["filters"]) != before
    panel.entries["cap_n"].delete(0, "end")
    panel.entries["cap_n"].insert(0, "7")
    panel.contrast_changed()
    panel.modes_removed.set("5")
    panel.page.set_modes_removed()
    panel.background_offset.set("12")
    panel.page.set_background_offset()
    assert settings_of(b)["contrast"]["cap_n"] == 7.0
    assert int(settings_of(b)["background_removal"]["modes_removed"]) == 5
    assert settings_of(b)["background_removal"]["offset"] == 12
    assert differs(d, campaign.node("b"))
    select(d, campaign.folder)
    filters = ("contrast", "filters")
    assert bold(form, (filters, "minmax")) and not bold(form, (filters, "capping"))
    assert row_button(form, (filters, "capping")) is None and row_button(form, ("contrast", "cap_n"))
    row_button(form, (filters, "minmax")).invoke()
    d.update()
    assert len({"minmax" in (settings_of(p)["contrast"].get("filters") or []) for p in campaign.projects.values()}) == 1
    assert settings_of(b)["contrast"]["cap_n"] == 7.0  # The other setting kept
    select(d, campaign.node("b"))
    d.reset_node(campaign.node("b"))  # Its ↺
    d.update()
    assert not differs(d, campaign.node("b"))


def test_piv(campaign):
    """The PIV/PTV of a node in the side panel of its page: a setting, another software (the settings of the first
    kept), the regression; the reset greyed for a project with the settings of the open one."""
    d = campaign.batch("piv")
    form, settings_of, b = d.form, d.settings_of, campaign.projects["b"]
    assert isinstance(form, BatchPIVForm)
    select(d, campaign.node("b"))
    assert not differs(d, campaign.node("b"))
    engine = settings_of(b)["engine"]
    p = next(p for p in piv.get_engine(engine).parameters if p.kind in ("int", "float"))
    form.panel.engine_vars[p.key].set("3" if form.panel.engine_vars[p.key].get() != "3" else "4")
    form.commit()
    assert str(settings_of(b)["settings"][p.key]) == form.panel.engine_vars[p.key].get()
    other = next(cls for cls in piv.ENGINES.values() if cls.key != engine)
    form.panel.engine_name.set(other.name)
    form.panel.engine_selected()
    d.update()
    assert settings_of(b)["engine"] == other.key and engine in settings_of(b)["engine_settings"]
    form.panel.regression.set("spicy")
    form.panel.regression_changed()
    assert settings_of(b)["regression"]["method"] == "spicy"
    d.reset_node(campaign.node("b"))
    d.update()
    assert settings_of(b)["engine"] == engine and not differs(d, campaign.node("b"))


def test_quantities_greyed_per_node(campaign):
    """The quantities of the images and of the POD greyed only when none of the projects of the node can export them
    (the reason in the help), the POD modes of the form too."""
    folder = campaign.projects["c"].folder
    os.remove(os.path.join(folder, "PreProcessing", "pod.npz"))
    d = campaign.batch("post")
    q = d.panels[0][1]

    def greyed():
        return {k for k in imaging.FIGURE_FILES if "unavailable" in q.tree.item(k, "tags")}
    select(d, campaign.node("c"))
    assert greyed() == imaging.POD_FIGURES and not d.form.panel.pod_slider.enabled
    select(d, campaign.folder)
    assert not greyed() and d.form.panel.pod_slider.enabled


def test_jobs_and_overwrite(campaign):
    """The jobs of the three steps; the results of a previous export: asked once for all the projects, Cancel (not
    run) or Overwrite all (removed by the run)."""
    gui = campaign.gui
    for page, step in (("pre", "preprocessing"), ("piv", "piv")):
        d = campaign.batch(page)
        d.check_all(True)
        d.ok()
        assert d.result and len(d.result["projects"]) == 3 and step in d.result["projects"][0]["settings"]
    folder = os.path.join(campaign.projects["b"].folder, "PostProcessing", "PaIRS")
    os.makedirs(folder)
    open(os.path.join(folder, "b_PostProcessing_Old.png"), "w").close()
    d = campaign.batch("post")
    d.check_all(True)
    del gui.asked[:]
    gui.answers[:] = ["Cancel"]
    d.ok()
    assert len(gui.asked) == 1 and d.result is None
    gui.answers[:] = ["Overwrite all"]
    d.ok()
    assert d.result["options"]["overwrite"] is True
    from velocihub import batch_run

    entry = next(e for e in d.result["projects"] if os.path.basename(e["folder"]) == "b")
    post = entry["settings"]["postprocessing"]
    spec = export_spec(post.get("export"))
    spec["choice"].update(figures=["mean_velocity"], videos=[])  # One figure: the overwrite tested, not the export
    post["export"] = spec
    status, _ = batch_run.run_postprocessing(entry, d.result)
    assert status == "done" and "b_PostProcessing_Old.png" not in os.listdir(folder)
