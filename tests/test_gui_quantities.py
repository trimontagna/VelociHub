"""The quantities exported (gui/quantities.py, the window Quantities and the box of the window Batch): the check marks of
the quantities and of their groups, Check all and Uncheck all, and the quantities greyed out that can't be exported
(those of the PTV, of the images and of the POD), their clicks refused."""

import os
import types

import pytest

import synthetic as s
from conftest import tk_root

tk = pytest.importorskip("tkinter")

import velocihub as vh  # noqa: E402
from velocihub.gui.quantities import (CHECKED, PARTIAL, PTV_KEYS, UNAVAILABLE, UNCHECKED,  # noqa: E402
                                      QuantitiesEditor, export_spec)
from velocihub.imaging import FIGURE_FILES as IMAGING  # noqa: E402
from velocihub.postprocessing import FIGURE_GROUPS, FIGURE_TITLES, INSTANTANEOUS, PostProcessing  # noqa: E402


@pytest.fixture
def editor():
    root = tk_root()
    specs = []
    editor = QuantitiesEditor(root, export_spec(None), on_change=specs.append)
    editor.pack()
    root.update()
    editor.specs = specs
    yield editor
    root.destroy()


def cells(editor, iid):
    return tuple(editor.tree.item(iid, "values"))[:2]


def click(editor, iid, column):
    """A click on the cell of a row (as the mouse does)."""
    editor.tree.see(iid)
    editor.update()
    x, y, w, h = editor.tree.bbox(iid, column)
    return editor.click(types.SimpleNamespace(x=x + w // 2, y=y + h // 2))


def test_group_check_marks(editor):
    """A group: all of its quantities exported or none (images, videos), its mark all, none or some."""
    group, keys = next((g, k) for g, k in FIGURE_GROUPS if any(q in INSTANTANEOUS for q in k)
                       and not all(q in INSTANTANEOUS for q in k))
    gid, videos = "group|" + (group or "Other"), [k for k in keys if k in INSTANTANEOUS]
    assert cells(editor, gid) == (CHECKED, UNCHECKED)
    click(editor, gid, "#1")
    assert set(editor.specs[-1]["choice"]["figures"]) == set(FIGURE_TITLES) - set(keys)
    assert all(cells(editor, k)[0] == UNCHECKED for k in keys)
    click(editor, gid, "#1")
    assert editor.specs[-1]["choice"]["figures"] is None
    click(editor, keys[0], "#1")
    assert cells(editor, gid)[0] == PARTIAL
    click(editor, gid, "#2")
    assert set(editor.specs[-1]["choice"]["videos"]) == set(videos)
    assert all(cells(editor, k)[1] == (CHECKED if k in videos else UNAVAILABLE) for k in keys)
    editor.show(export_spec(None), rolling=True)  # Every quantity with its video
    click(editor, gid, "#2")
    assert set(editor.specs[-1]["choice"]["videos"]) == set(keys)
    editor.show(editor.specs[-1], enabled=False)
    n = len(editor.specs)
    click(editor, gid, "#1")
    assert len(editor.specs) == n


def test_check_all(editor):
    editor.check_buttons[1].invoke()
    assert editor.specs[-1]["choice"]["figures"] == [] and editor.specs[-1]["choice"]["videos"] == []
    assert cells(editor, "velocity")[0] == UNCHECKED
    editor.check_buttons[0].invoke()
    assert editor.specs[-1]["choice"]["figures"] is None and cells(editor, "velocity")[0] == CHECKED
    editor.show(export_spec(None), enabled=False)
    assert all(b.instate(["disabled"]) for b in editor.check_buttons)


def test_unavailable_greyed(editor, copy):
    """For PIV results without POD nor raw images: the PTV, the raw image and the POD greyed, their clicks refused
    (the reason said), not counted; the image of the PIV available."""
    folder = copy("piv")
    os.remove(os.path.join(folder, "PreProcessing", "pod.npz"))
    for name in os.listdir(folder):
        if name.endswith(".tif"):
            os.remove(os.path.join(folder, name))
    post = PostProcessing(vh.load_vectors(os.path.join(folder, "PIV"), camera=s.CAMERA), vh.PostSettings())
    editor.set_post(post)
    greyed = {k for k in FIGURE_TITLES if "unavailable" in editor.tree.item(k, "tags")}
    assert greyed == set(PTV_KEYS) | {"raw_image", "pod_mode", "pod_spectrum"}
    assert "piv_image" in IMAGING and "piv_image" not in greyed
    n = len(editor.specs)
    click(editor, "raw_image", "#1")
    assert len(editor.specs) == n and "raw images" in editor.note["text"]
    click(editor, "ptv_tracks", "#1")
    assert len(editor.specs) == n and "PTV" in editor.note["text"]
    editor.set_post(None)
