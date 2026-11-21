"""The menu Tests of the main window: the benchmark of VelociHub on analytic flows (velocihub/benchmark), run in its
own process (python -m velocihub.benchmark) while the interface stays usable, its progress in a window of its own
(ProgressWindow: each level, the current step, the time, the messages; Stop) and in the status log, its report opened
in the browser at the end (compared with the previous one; the reports in benchmark_reports of the repository, one
folder per run named by its date and time: report.reports_folder); and
the test workspace: the particle images of the cases as measurements (with their exact fields in their subfolder
Exact, post-processable at once), written once and opened as the workspace of the interface (test mode), then back to
the workspace of the user."""

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import ttk

from ..prefs import getpref, setpref
from .widgets import winopen

# Lines of the log of the progress window
LOG_LINES = 400


class ProgressWindow(tk.Toplevel):
    """The progress of a run of the benchmark (or of the writing of the test images), not modal: a row per level
    (its bar, waiting / running / done), the current step, the time since the start, the log of the messages, and
    the buttons Stop (the process ended), Open the report (at the end) and Close (the window hidden, the run goes
    on: Tests > Show the progress)."""

    def __init__(self, menu, title, levels=()):
        super().__init__(menu.app.window)
        self.menu = menu
        self.title(title)
        self.transient(menu.app.window)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text=title, font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)
        self.levels_frame = ttk.Frame(frame)
        self.levels_frame.pack(fill=tk.X, pady=(8, 4))
        self.rows = {}
        self.set_levels(levels)
        self.current = ttk.Label(frame, text="Starting...", wraplength=560, justify=tk.LEFT)
        self.current.pack(anchor=tk.W, pady=(6, 0))
        self.clock = ttk.Label(frame, foreground="#555555")
        self.clock.pack(anchor=tk.W)
        box = ttk.Frame(frame)
        box.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log = tk.Listbox(box, height=12, width=90, activestyle="none", background="white")
        bar = ttk.Scrollbar(box, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=bar.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        buttons = ttk.Frame(frame)
        buttons.pack(anchor=tk.E, pady=(10, 0))
        self.stop_button = ttk.Button(buttons, text="Stop", command=menu.stop_run)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 6))
        self.report_button = ttk.Button(buttons, text="Open the report", state=tk.DISABLED, command=self.open_report)
        self.report_button.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Close", command=self.withdraw).pack(side=tk.LEFT)
        self.report = None
        self.t0 = time.time()
        self.running = True
        self.tick()

    def set_levels(self, levels):
        """A row per level (cases.LEVELS keys), all waiting."""
        from ..benchmark.cases import LEVELS

        for child in self.levels_frame.winfo_children():
            child.destroy()
        self.rows = {}
        for r, level in enumerate(levels):
            ttk.Label(self.levels_frame, text=LEVELS.get(level, (level,))[0]).grid(row=r, column=0, sticky=tk.W,
                                                                                   padx=(0, 10), pady=1)
            bar = ttk.Progressbar(self.levels_frame, maximum=1.0, length=280)
            bar.grid(row=r, column=1, pady=1)
            state = ttk.Label(self.levels_frame, text="waiting", foreground="#555555", width=16)
            state.grid(row=r, column=2, sticky=tk.W, padx=(10, 0), pady=1)
            self.rows[level] = (bar, state)

    def level(self, level, done, total):
        """A level started (done 0), progressed (done of total) or ended (done = total)."""
        if level not in self.rows:
            return
        bar, state = self.rows[level]
        bar["value"] = done / total if total else 0
        state["text"] = ("done" if done >= total else "running" if total == 1 else f"{done} of {total - 1} runs"
                         if done < total - 1 else "the metrics")
        state["foreground"] = "#1a7f37" if done >= total else "#000000"
        if level == "images" and done < total and "chain" in self.rows:  # The runs of the chain are among them
            chain = self.rows["chain"][1]
            if chain["text"] == "waiting":
                chain["text"] = "runs counted above"

    def message(self, text):
        self.current["text"] = text
        self.log.insert(tk.END, f"{time.time() - self.t0:6.0f} s  {text}")
        if self.log.size() > LOG_LINES:
            self.log.delete(0, self.log.size() - LOG_LINES - 1)
        self.log.see(tk.END)

    def finished(self, text, report=None, error=False):
        """The end of the run: its last message, its report."""
        self.running = False
        self.message(text)
        self.current["foreground"] = "#b42318" if error else "#1a7f37"
        self.stop_button["state"] = tk.DISABLED
        self.report = report
        if report:
            self.report_button["state"] = tk.NORMAL
        self.tick()

    def open_report(self):
        if self.report:
            webbrowser.open(self.report)

    def tick(self):
        seconds = int(time.time() - self.t0)
        self.clock["text"] = (f"{'Running for' if self.running else 'Ran for'} {seconds // 60} min "
                              f"{seconds % 60:02d} s")
        if self.running and self.winfo_exists():
            self.after(1000, self.tick)

    def show(self):
        self.deiconify()
        self.lift()


class BenchmarkMenu:
    """The menu Tests and the process it runs (one at a time)."""

    def __init__(self, app):
        self.app = app
        self.proc = None
        self.messages = queue.Queue()
        self.on_done = None
        self.window = None  # ProgressWindow of the last run

    def items(self):
        from ..benchmark import CASES, FLOWS

        cases = [(f"{FLOWS[key].name}", None, lambda key=key: self.run([key], levels=["post", "processing"]))
                 for key in CASES]
        return [("Benchmark", [
                    ("Run everything", None, lambda: self.run(None)), None,
                    ("Pre-processing: background removal", None, lambda: self.run([], levels=["preprocessing"])),
                    ("Post-processing: all the cases", None, lambda: self.run(None, levels=["post"])),
                    ("Processing → post: all the cases", None, lambda: self.run(None, levels=["processing"])),
                    ("Pre-processing → processing: images", None, lambda: self.run(None, levels=["images"])),
                    ("Full chain: images → post", None, lambda: self.run(None, levels=["chain"])),
                    None, *cases]),
                ("Show the progress", None, self.show_progress),
                ("Test workspace", [
                    ("Open the test workspace", None, self.open_test_workspace),
                    ("Write the test images again", None, lambda: self.write_test_workspace(open_it=True)),
                    ("Back to my workspace", None, self.leave_test_workspace)]),
                None,
                ("Open the last report", None, self.open_last),
                ("Open the reports folder", None, self.open_folder)]

    @staticmethod
    def reports_folder():
        from ..benchmark.report import reports_folder
        return reports_folder()

    # ------------------------------------------------------------------ the process

    def busy(self):
        if self.proc is not None and self.proc.poll() is None:
            self.app.msgbox("A benchmark or the test images are being written: wait for its end.", "Tests", "warn")
            return True
        return False

    def start(self, args, text, on_done, title="Benchmark", levels=()):
        """python -m velocihub.benchmark args --json in its own process; on_done(message "done") at its end. Its
        progress in a ProgressWindow (title; the rows of the levels)."""
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
        if self.window is not None and self.window.winfo_exists():
            self.window.destroy()
        self.window = ProgressWindow(self, title, levels)
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
        from ..benchmark.cases import LEVELS

        what = ("everything" if cases is None and not levels else
                ", ".join(LEVELS[lv][0] for lv in levels or []) + (" of " + ", ".join(cases) if cases else ""))
        self.start(args, f"Benchmark started ({what}): the report opens in the browser when it is done", done,
                   title=f"Benchmark: {what}", levels=levels or list(LEVELS))

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
            window = self.window if self.window is not None and self.window.winfo_exists() else None
            if m["type"] == "progress":
                self.app.update_status(f"Tests: {m['text']}")
                if window:
                    window.message(m["text"])
            elif m["type"] == "levels":
                if window:
                    window.set_levels(m["levels"])
            elif m["type"] == "level":
                if window:
                    window.level(m["level"], m["done"], m["total"])
            elif m["type"] == "done":
                ended = True
                self.app.update_status(f"Tests: {m['text']}")
                if window:
                    window.finished(m["text"], m.get("report"))
                if self.on_done:
                    self.on_done(m)
            elif m["type"] == "error":
                ended = True
                self.app.status_error(f"Tests: {m['text']}")
                if window:
                    window.finished(m["text"], error=True)
        if not ended and self.proc.poll() is not None and self.messages.empty():
            # Ended without its last message: crashed or stopped
            window = self.window if self.window is not None and self.window.winfo_exists() else None
            if getattr(self, "stopping", False):
                self.stopping = False
                self.app.update_status("Tests: stopped")
                if window:
                    window.finished("Stopped.", error=True)
            elif self.proc.returncode not in (0, 1):
                details = " | ".join(self.stderr[-4:]) or "no message"
                self.app.status_error(f"The tests ended unexpectedly (exit code {self.proc.returncode}): {details}")
                if window:
                    window.finished(f"Ended unexpectedly (exit code {self.proc.returncode}): {details}", error=True)
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
                   lambda message: self.switch_to(message["workspace"]) if open_it else None,
                   title="Writing the test images")

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

    def show_progress(self):
        """Tests > Show the progress: the window of the last run."""
        if self.window is not None and self.window.winfo_exists():
            self.window.show()
        else:
            self.app.msgbox("No benchmark run yet.", "Benchmark", "info")

    def kill(self):
        """The process running ended, with the processes it started (the PIV/PTV, the regressions)."""
        if self.proc is None or self.proc.poll() is not None:
            return
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            self.proc.kill()

    def stop_run(self):
        """Stop in the progress window."""
        if self.proc is not None and self.proc.poll() is None:
            self.stopping = True
            self.kill()

    def stop(self):
        """At the exit of the interface: the process running ended."""
        self.kill()
