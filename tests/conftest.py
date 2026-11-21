"""Shared fixtures of the tests: a scratch home (the preferences of VelociHub, read at import, never the user's), the
synthetic measurements (synthetic.py: built once per session) and their copies, and the interface (one per module)."""

import os
import sys
import tempfile

HOME = tempfile.mkdtemp(prefix="velocihub_tests_home_")
os.environ["USERPROFILE"] = os.environ["HOME"] = HOME  # Before velocihub: its preferences file is set at import
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

import synthetic  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--benchmark-report", default=None, metavar="FOLDER",
                     help="write the report of the benchmark (test_benchmark.py) in this folder (by default in "
                          "benchmark_reports of the repository, a folder per run)")
    parser.addoption("--benchmark-images", action="store_true",
                     help="also run the levels of the benchmark on particle images (pre-processing -> processing, "
                          "the full chain: PaIRS-UniNa and TracTrac, about 20 minutes)")


@pytest.fixture(scope="session")
def measurements(tmp_path_factory):
    """The synthetic measurements, built once: "piv" (PIV results), "both" (PIV and PTV results), "ptv" (PTV
    results only), "images" (images and pre-processing, no results)."""
    root = tmp_path_factory.mktemp("measurements")
    out = {}
    for name, piv_results, ptv_results in (("piv", True, False), ("both", True, True), ("ptv", False, True),
                                           ("images", False, False)):
        out[name] = synthetic.make_measurement(str(root / name), piv_results, ptv_results)
    return out


@pytest.fixture
def copy(measurements, tmp_path):
    """copy(name, *path): a fresh copy of a synthetic measurement under tmp_path (path: its place there)."""
    def make(name, *path):
        return synthetic.copy_measurement(measurements[name], os.path.join(str(tmp_path), *(path or (name,))))
    return make


def tk_root():
    """A Tk root, tried three times (the first one of a process sometimes fails to read a file of Tk on Windows);
    the test skipped without a display."""
    tk = pytest.importorskip("tkinter")
    for attempt in range(3):
        try:
            return tk.Tk()
        except tk.TclError as e:
            if "display" in str(e).lower() or attempt == 2:
                pytest.skip(f"no display: {e}")


@pytest.fixture(scope="module")
def gui():
    """The interface (Tk root and main window) of a module, the message boxes and questions answered by the test
    (answers: the replies of the next questions; boxes and asked: what was shown). Skipped without a display."""
    tk = pytest.importorskip("tkinter")
    from tkinter import messagebox

    from velocihub import prefs
    from velocihub.gui import app as app_module, widgets

    root = tk_root()
    if os.path.isfile(prefs.PREFS_FILE):
        os.remove(prefs.PREFS_FILE)
    state = type("GUI", (), {})()
    state.answers, state.asked, state.boxes, state.errors = [], [], [], []
    saved = (messagebox.showinfo, messagebox.showwarning, messagebox.showerror, messagebox.askyesno,
             messagebox.askokcancel, widgets.QuestionDialog.run)
    messagebox.showinfo = messagebox.showwarning = messagebox.showerror = (
        lambda title, message, **k: state.boxes.append(message))
    messagebox.askyesno = messagebox.askokcancel = lambda *a, **k: True

    def answer(dialog):
        state.asked.append(dialog)
        return state.answers.pop(0) if state.answers else None
    widgets.QuestionDialog.run = answer
    prefs.setpref("workspace_file", None)
    state.root = root
    state.app = app_module.MainWindow(root)
    root.report_callback_exception = lambda e, v, t: state.errors.append(v)
    root.update()
    yield state
    root.destroy()
    (messagebox.showinfo, messagebox.showwarning, messagebox.showerror, messagebox.askyesno,
     messagebox.askokcancel, widgets.QuestionDialog.run) = saved
    assert not state.errors, state.errors
