"""The windows Batch (gui/batch_dialog.py) of the three steps: the settings of a node in the side panel of the page
(the settings that differ in bold, each with its ↺; a filter of the pre-processing on its own; another PIV software),
the buttons resetting to the open project or to the stitch reference (greyed when they would change nothing), the
quantities exported of the post-processing (Check all, the open project, greyed per node), the jobs and the results
of a previous export overwritten."""

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
    assert d.reset_button.instate(["disabled"]) and d.reset_all_button.instate(["disabled"])
    assert (form.panel.pod_slider.lo, form.panel.pod_slider.hi) == (1, 3)
    form.panel.vars["outlier_k"].set("3")
    form.commit()
    assert settings_of(b)["settings"]["outlier_k"] == 3.0
    assert settings_of(campaign.projects["c"])["settings"].get("outlier_k", 4.0) == 4.0
    assert d.reset_button.instate(["!disabled"]) and d.reset_all_button.instate(["!disabled"])
    form.panel.vars["outlier_k"].set("abc")
    form.commit()
    assert settings_of(b)["settings"]["outlier_k"] == 3.0 and form.panel.vars["outlier_k"].get() == "3" and d.node_note["text"]
    select(d, campaign.folder)
    assert bold(form, ("settings", "outlier_k")) and not bold(form, ("settings", "skew_zero"))
    assert row_button(form, ("settings", "skew_zero")) is None
    row_button(form, ("settings", "outlier_k")).invoke()
    d.update()
    assert len({settings_of(p)["settings"].get("outlier_k", 4.0) for p in campaign.projects.values()}) == 1
    assert d.reset_all_button.instate(["disabled"])


def test_stitch_reference_and_quantities(campaign):
    """The stitch reference (a → b): all its settings for b (its camera kept), or its quantities only; the quantities
    box: Check all, Uncheck all, the open project (greyed when the same), no reset to the default."""
    d = campaign.batch("post")
    form, settings_of, a, b = d.form, d.settings_of, campaign.projects["a"], campaign.projects["b"]
    q = d.panels[0][1]
    assert not q.reset_button.winfo_manager()
    d.step.options.chains = [[(a.root, "a"), (a.root, "b")]]
    d.refresh()
    select(d, campaign.node("b"))
    assert d.stitch_button.instate(["disabled"]) and q.same_button.instate(["disabled"])  # Already the same
    form.panel.vars["skew_zero"].set("33")
    form.commit()
    form.panel.camera["x_offset"].set("60")
    form.commit()
    assert d.stitch_button.instate(["!disabled"])
    d.stitch_button.invoke()
    d.update()
    assert settings_of(b)["settings"] == settings_of(a)["settings"]
    assert settings_of(b)["camera"]["x_offset_mm"] == 60.0
    q.check_buttons[1].invoke()
    d.update()
    assert export_spec(settings_of(b)["export"])["choice"]["figures"] == []
    assert q.same_button.instate(["!disabled"]) and q.stitch_button.instate(["!disabled"])
    q.stitch_button.invoke()
    d.update()
    assert export_spec(settings_of(b)["export"]) == export_spec(settings_of(a)["export"])
    select(d, campaign.node("c"))
    assert d.stitch_button.instate(["disabled"]) and q.stitch_button.instate(["disabled"])
    q.check_buttons[1].invoke()
    d.update()
    q.same_button.invoke()
    d.update()
    assert q.same_button.instate(["disabled"])
    assert export_spec(settings_of(campaign.projects["c"])["export"]) == export_spec(settings_of(a)["export"])


def test_preprocessing(campaign):
    """The pre-processing of a node in the side panel of its page: a filter on its own (only it in bold at the
    folder, its ↺), its settings, the background removal; the reset to the open project."""
    d = campaign.batch("pre")
    form, settings_of, b = d.form, d.settings_of, campaign.projects["b"]
    assert isinstance(form, BatchPreForm) and not hasattr(form.panel, "new_mask_button")
    select(d, campaign.node("b"))
    assert d.reset_button.instate(["disabled"])
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
    assert settings_of(b)["contrast"]["cap_n"] == 7.0
    assert int(settings_of(b)["background_removal"]["modes_removed"]) == 5
    assert d.reset_button.instate(["!disabled"])
    select(d, campaign.folder)
    filters = ("contrast", "filters")
    assert bold(form, (filters, "minmax")) and not bold(form, (filters, "capping"))
    assert row_button(form, (filters, "capping")) is None and row_button(form, ("contrast", "cap_n"))
    row_button(form, (filters, "minmax")).invoke()
    d.update()
    assert len({"minmax" in (settings_of(p)["contrast"].get("filters") or []) for p in campaign.projects.values()}) == 1
    assert settings_of(b)["contrast"]["cap_n"] == 7.0  # The other setting kept
    select(d, campaign.node("b"))
    d.reset_button.invoke()
    d.update()
    assert d.reset_button.instate(["disabled"])


def test_piv(campaign):
    """The PIV/PTV of a node in the side panel of its page: a setting, another software (the settings of the first
    kept), the regression; the reset greyed for a project with the settings of the open one."""
    d = campaign.batch("piv")
    form, settings_of, b = d.form, d.settings_of, campaign.projects["b"]
    assert isinstance(form, BatchPIVForm)
    select(d, campaign.node("b"))
    assert d.reset_button.instate(["disabled"])
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
    d.reset_button.invoke()
    d.update()
    assert settings_of(b)["engine"] == engine and d.reset_button.instate(["disabled"])


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
    folder = os.path.join(campaign.projects["b"].folder, "PostProcessing")
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
