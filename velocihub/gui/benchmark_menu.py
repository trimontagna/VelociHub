"""The menu Tests of the main window: the benchmark of VelociHub on analytic flows (velocihub/benchmark), run in its
own process (python -m velocihub.benchmark) while the interface stays usable, its progress in the status log, its
report opened in the browser at the end (compared with the previous one; the reports in ~/VelociHub/benchmarks); and
the test workspace: the particle images of the cases as measurements (with their exact fields in their subfolder
Exact, post-processable at once), written once and opened as the workspace of the interface (test mode), then back to
the workspace of the user."""

import json
import os
import queue
import subprocess
import sys
import threading
import webbrowser

from ..prefs import getpref, setpref
from .widgets import winopen


class BenchmarkMenu:
    """The menu Tests and the process it runs (one at a time)."""

    def __init__(self, app):
        self.app = app
        self.proc = None
        self.messages = queue.Queue()
        self.on_done = None

    def items(self):
        from ..benchmark import CASES, FLOWS

        cases = [(f"{FLOWS[key].name}", None, lambda key=key: self.run([key], levels=["post", "processing"]))
                 for key in CASES]
        return [("Benchmark", [
                    ("Run everything", None, lambda: self.run(None)), None,
                    ("Pre-processing: background removal", None, lambda: self.run([], levels=["preprocessing"])),
                    ("Post-processing: all the cases", None, lambda: self.run(None, levels=["post"])),
                    ("Processing → post: all the cases", None, lambda: self.run(None, levels=["processing"])),
                    None, *cases]),
                ("Test workspace", [
                    ("Open the test workspace", None, self.open_test_workspace),
                    ("Write the test images again", None, lambda: self.write_test_workspace(open_it=True)),
                    ("Back to my workspace", None, self.leave_test_workspace)]),
                None,
                ("Open the last report", None, self.open_last),
                ("Open the reports folder", None, self.open_folder)]

    @staticmethod
    def reports_folder():
        return os.path.join(os.path.expanduser("~"), "VelociHub", "benchmarks")

    # ------------------------------------------------------------------ the process

    def busy(self):
        if self.proc is not None and self.proc.poll() is None:
            self.app.msgbox("A benchmark or the test images are being written: wait for its end.", "Tests", "warn")
            return True
        return False

    def start(self, args, text, on_done):
        """python -m velocihub.benchmark args --json in its own process; on_done(message "done") at its end."""
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = {**os.environ, "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")}
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            self.proc = subprocess.Popen([sys.executable, "-u", "-m", "velocihub.benchmark", *args, "--json"],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                         text=True, encoding="utf-8", errors="replace", creationflags=flags, env=env)
        except OSError as e:
            self.app.status_error(f"The benchmark could not start: {e}")
            return
        self.stderr, self.on_done = [], on_done
        self.messages = queue.Queue()
        threading.Thread(target=self.read, args=(self.proc.stdout, False), daemon=True).start()
        threading.Thread(target=self.read, args=(self.proc.stderr, True), daemon=True).start()
        self.app.update_status(text)
        self.app.window.after(300, self.poll)

    def run(self, cases, levels=None):
        """Run the benchmark (cases: keys, None: all; levels: those of cases.LEVELS, None: all)."""
        from ..benchmark.report import default_folder
        from ..piv.regression import spicy_available

        if self.busy():
            return
        ok, reason = spicy_available()
        if not ok and (levels is None or "processing" in levels):
            self.app.status_error(f"The benchmark needs SPICY: {reason}")
            return
        args = [*(cases or []), "--no-open", "--out", default_folder()]
        if levels:
            args += ["--levels", *levels]
        last = getpref("benchmark_last_report")
        if last and os.path.isfile(last):
            args += ["--compare", os.path.dirname(last)]

        def done(message):
            setpref("benchmark_last_report", message["report"])
            webbrowser.open(message["report"])
        what = ("everything" if cases is None and not levels else
                ", ".join(levels or []) + (" of " + ", ".join(cases) if cases else ""))
        self.start(args, f"Benchmark started ({what}): the report opens in the browser when it is done", done)

    def read(self, stream, err):
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if err:
                self.stderr.append(line)
                continue
            try:
                self.messages.put(json.loads(line))
            except ValueError:
                self.messages.put({"type": "progress", "text": line})

    def poll(self):
        ended = False
        while True:
            try:
                m = self.messages.get_nowait()
            except queue.Empty:
                break
            if m["type"] == "progress":
                self.app.update_status(f"Tests: {m['text']}")
            elif m["type"] == "done":
                ended = True
                self.app.update_status(f"Tests: {m['text']}")
                if self.on_done:
                    self.on_done(m)
            elif m["type"] == "error":
                ended = True
                self.app.status_error(f"Tests: {m['text']}")
        if not ended and self.proc.poll() is not None and self.messages.empty():
            # Ended without its last message: crashed
            if self.proc.returncode not in (0, 1):
                details = " | ".join(self.stderr[-4:]) or "no message"
                self.app.status_error(f"The tests ended unexpectedly (exit code {self.proc.returncode}): {details}")
            return
        if not ended:
            self.app.window.after(300, self.poll)

    # ------------------------------------------------------------------ the test workspace

    @staticmethod
    def test_workspace_file():
        from ..benchmark.images import TEST_WORKSPACE
        from ..workspace import WORKSPACE_FILE
        return os.path.join(TEST_WORKSPACE, WORKSPACE_FILE)

    def write_test_workspace(self, open_it=False):
        """Write the test measurements (particle images of the cases) in their own process; open_it: then open them
        as the workspace."""
        if self.busy():
            return
        self.start(["--test-workspace"], "Writing the test images (a few minutes)...",
                   lambda message: self.switch_to(message["workspace"]) if open_it else None)

    def open_test_workspace(self):
        """Test mode: the test workspace as the workspace of the interface (written first if it isn't yet)."""
        file = self.test_workspace_file()
        if os.path.isfile(file):
            self.switch_to(file)
        else:
            self.write_test_workspace(open_it=True)

    def switch_to(self, file):
        current = getpref("workspace_file")
        if current and os.path.normcase(os.path.abspath(current)) != os.path.normcase(os.path.abspath(file)):
            setpref("workspace_before_tests", current)
        self.app.workspace.load(file)
        self.app.update_status("Test workspace: each case is a measurement with its images; its exact fields are in "
                               "its folder Exact (Post-processing: vector fields folder). Tests > Test workspace > "
                               "Back to my workspace to leave it")

    def leave_test_workspace(self):
        previous = getpref("workspace_before_tests")
        if previous and os.path.isfile(previous):
            self.app.workspace.load(previous)
        else:
            self.app.workspace.new()

    # ------------------------------------------------------------------ the reports

    def open_last(self):
        last = getpref("benchmark_last_report")
        if not last or not os.path.isfile(last):
            self.app.msgbox("No report yet: run the benchmark (Tests > Benchmark).", "Benchmark", "info")
            return
        webbrowser.open(last)

    def open_folder(self):
        folder = self.reports_folder()
        if not os.path.isdir(folder):
            self.app.msgbox(f"No report yet: {folder} doesn't exist.", "Benchmark", "info")
            return
        winopen(folder)

    def stop(self):
        """At the exit of the interface: the process running ended (its regressions finish their maps alone)."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
