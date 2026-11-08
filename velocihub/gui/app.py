"""VelociHub: the main window of the interface, with a page for each step of the processing:

1. Pre-processing: the POD-based background removal of the PIV images, with masks and contrast enhancement
   (preprocessing_page.py);
2. PIV of the images pre-processed with the recipe, with PaIRS-UniNa (piv_page.py);
3. Post-processing of the vector fields: statistics, convergence, peak locking and potential core
   (postprocessing_page.py).

The window holds what the pages share: the menus (built from those of the pages), the keyboard shortcuts, the
arrow keys (sent to the page shown), the status log, the progress of the processing, the dialogs, the image looked
at (followed by the pages), the panel Projects (the workspace: workspace_panel.py), and the project of the images
(VelociHub_project.json in their folder, project.py), saved as the settings and the results of the pages change and
restored when the images are opened again. A project with PIV results but no images (results_folder) is opened in
the pages that don't need images (the post-processing): its project file keeps the sections of the others.

The pages are in PAGES: a page can be removed from it (and its module deleted) with the pages that need it. The
first page gives the images (PreProcessingPage); the PIV needs it, the post-processing uses the PIV if it is there.

Run it with:
    velocihub  (or python run_VelociHub.py, python -m velocihub)
"""

import ctypes
import os
import sys
import tkinter as tk
import traceback
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from ..preprocessing import Stopped, same_path
from ..project import (PROJECT_FILE, PROJECT_FILES, absolute, clean_name, default_project_name, read_project,
                       relative, summary, write_project)
from .piv_page import PIVPage
from .postprocessing_page import PostProcessingPage
from .preprocessing_page import PreProcessingPage
from .widgets import InputDialog, QuestionDialog, Waitbar, add_menu, place, winopen
from .workspace_panel import WorkspacePanel

# Pages of the window, in the order of the processing (the first one gives the images)
PAGES = (PreProcessingPage, PIVPage, PostProcessingPage)
# Position of the main window in normalized screen units [left bottom width height]
MAIN_POS = [0, 0.05, 1, 0.95]
PROJECT_FILE_TYPES = [("VelociHub project", PROJECT_FILES), ("JSON", "*.json"), ("All files", "*.*")]
# Menus shared by the pages, first and last; the menus of the pages are between them
FIRST_MENUS = ("File", "Settings")
# The menus of the pages not shown (their items never used from the menu bar: their keyboard shortcuts kept)
HIDDEN_MENUS = ("Pre-processing", "PIV/PTV", "Post-processing")
LAST_MENUS = ("Help",)


class MainWindow:
    """Main window: a tab for each page, the status log and the progress at the bottom, and the project of the
    images. The first page gives the images (image_folder(), image_files(), open_images())."""

    def __init__(self, root, pages=PAGES):
        self.root = root
        root.withdraw()
        root.report_callback_exception = self.report_exception
        # Project of the images: its name (None: the name of the folder of the images), steps done, pending save,
        # and restoring (nothing is saved meanwhile)
        self.name = None
        self.history = []
        self.project_job = None
        self.restoring = False
        # Folder of the project opened without images (PIV results only: post-processing), None otherwise
        self.results_folder = None
        # The window of the batch running (batch_dialog.BatchRunWindow), None if none
        self.batch_window = None
        # Image looked at (0-based), followed by the pages: the image of the pre-processing, the pair of the PIV
        # starting with it and the map of this pair in the post-processing
        self.image_index = 0

        self.window = win = tk.Toplevel(root)
        place(win, MAIN_POS)
        if sys.platform == "win32":
            win.state("zoomed")
        win.protocol("WM_DELETE_WINDOW", self.exit_program)
        # The arrow keys browse the series of the panel where the mouse is or was last, in the page shown; the
        # panels showing the same series follow
        for key, step in (("Left", -1), ("Up", -1), ("Right", 1), ("Down", 1)):
            win.bind(f"<{key}>", lambda event, step=step: self.arrow_key(event, step))
        self.progress_area = ttk.Frame(win)
        self.progress_area.pack(side=tk.BOTTOM, fill=tk.X)
        outer = ttk.PanedWindow(win, orient=tk.VERTICAL)
        outer.pack(fill=tk.BOTH, expand=True)
        # The panel Projects left of the pages
        body = ttk.PanedWindow(outer, orient=tk.HORIZONTAL)
        outer.add(body, weight=1)
        self.workspace = WorkspacePanel(body, self)
        body.add(self.workspace, weight=0)
        self.notebook = ttk.Notebook(body)
        body.add(self.notebook, weight=1)

        # Status log of all the pages, the most recent message first: one line (the mouse wheel scrolls the previous
        # ones, and the sash above it shows more lines)
        status = ttk.Frame(outer)
        self.status_listbox = tk.Listbox(status, height=1, background="white", activestyle="none")
        self.status_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Pages
        self.pages = {}
        for n, cls in enumerate(pages):
            page = cls(self.notebook, self)
            self.pages[page.key] = page
            self.notebook.add(page, text=f"  {n + 1} · {page.title}  ")
        self.source = next(iter(self.pages.values()))
        outer.add(status, weight=0)
        self.title = "VelociHub: " + ", ".join(page.title for page in self.pages.values())
        win.title(self.title)
        self.notebook.bind("<<NotebookTabChanged>>", lambda event: self.page_changed())
        self.build_menus()

        self.update_status(f"--- {datetime.now():%d-%b-%Y %H:%M:%S} ---")
        for page in self.pages.values():
            page.start()
        self.workspace.start()

    ############################################################################
    # Pages, menus and keys

    def page(self, key):
        """Page of the window (None if it isn't in the window)."""
        return self.pages.get(key)

    def current_page(self):
        """Key of the page shown."""
        selected = self.notebook.select()
        return next((key for key, page in self.pages.items() if str(page) == selected), None)

    def show_page(self, key):
        if key in self.pages:
            self.notebook.select(self.pages[key])

    def page_changed(self):
        """A page is shown: it is updated with the images and the results of the other pages, and shows the image
        looked at in them."""
        page = self.pages.get(self.current_page())
        if page is not None:
            page.shown()
        self.workspace.update_states()
        self.save_project()

    def build_menus(self):
        """Menus of the pages, those shared by several pages put together, and their keyboard shortcuts."""
        menus = {label: [] for label in FIRST_MENUS}
        for page in self.pages.values():
            for label, items in page.menus():
                if menus.get(label) and items:
                    menus[label].append(None)
                menus.setdefault(label, []).extend(items)
        explorer = [("Working folder", None, lambda: self.open_explorer(os.getcwd))]
        explorer += [(label, None, lambda folder=folder: self.open_explorer(folder))
                     for page in self.pages.values() for label, folder in page.folders()]
        menus["File"] += [None] + self.workspace.menu_items() + [None, ("Open project...", None,
                                                                         self.open_project_dialog),
                          ("Project name...", None, self.ask_project_name), None,
                          ("Clear cache", None, self.clear_cache), None,
                          ("Open in explorer", explorer), None, ("Exit", "Ctrl+Q", self.exit_program)]
        hidden = [menus.pop(label) for label in HIDDEN_MENUS if label in menus]
        menus["Help"] = [("Credits", None, self.show_credits)]
        order = list(FIRST_MENUS) + [m for m in menus if m not in FIRST_MENUS + LAST_MENUS] + list(LAST_MENUS)
        menubar = tk.Menu(self.window, tearoff=0)
        add_menu(menubar, [(label, menus[label]) for label in order if menus[label]])
        self.window.config(menu=menubar)
        self.menus = {label: menubar.nametowidget(menubar.entrycget(label, "menu")) for label in order
                      if menus[label]}
        self.bind_shortcuts(list(menus.values()) + hidden)

    def clear_cache(self):
        """File > Clear cache: the caches of all the pages (pre-processing: images loaded and modes in the temporary
        folder; PIV: images and results in memory, pairs of a PIV stopped after a confirmation; post-processing:
        vector fields and analyses)."""
        cleared = []
        for key in ("pre", "piv", "post"):
            page = self.pages.get(key)
            if page is not None and hasattr(page, "clear_cache"):
                try:
                    page.clear_cache()
                    cleared.append(page.title)
                except Exception as e:  # The other caches are cleared
                    self.status_error(f"{key}: the cache could not be cleared: {e}")
        self.update_status(f"Caches cleared: {', '.join(cleared)}")

    def bind_shortcuts(self, menus):
        """Keyboard shortcuts of the menu items: their accelerators (Ctrl+X or F5)."""
        def items(entries):
            for item in entries:
                if item is None:
                    continue
                if isinstance(item[1], list):
                    yield from items(item[1])
                elif item[1]:
                    yield item

        for _, accelerator, command in (item for entries in menus for item in items(entries)):
            if accelerator.startswith("Ctrl+"):
                key = accelerator[5:].lower()
                sequences = [f"<Control-{key}>", f"<Control-{key.upper()}>"]
            else:
                sequences = [f"<{accelerator}>"]
            for sequence in sequences:
                self.window.bind(sequence, lambda event, command=command: self.shortcut(command))

    def shortcut(self, command):
        # Nothing happens while a process is running
        if self.root.grab_current() is None:
            command()

    def arrow_key(self, event, step):
        """Arrow keys: previous (left, up) or next (right, down) item of the panel under the mouse."""
        # The arrows move the cursor in the entries, the selection in the lists and the sliders (which then
        # show their item), and nothing happens while a process is running
        if (isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Listbox, ttk.Treeview, ttk.Scale, ttk.Combobox))
                or self.root.grab_current() is not None):
            return
        page = self.pages.get(self.current_page())
        if page is not None:
            page.arrow(step)

    ############################################################################
    # Images, shared by the pages

    def images_changed(self):
        """Other images (or none): a new project, and the other pages without the images of the previous ones."""
        self.name = None
        self.history = []
        self.results_folder = None
        self.image_index = 0
        for page in self.pages.values():
            if page is not self.source:
                page.images_changed()

    def images_opened(self, restore=True):
        """Images opened by the first page: the other pages are updated, and their project restored (restore=True:
        the project of their folder if there is one; a dictionary: this project; False: a new project)."""
        self.images_changed()
        if isinstance(restore, dict):
            self.restore_project(restore)
        elif restore:
            self.check_project()
        files = self.source.image_files()
        if not self.history:
            self.history.append({"date": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "step": "images",
                                 "text": f"{len(files)} images: {files[0]} ... {files[-1]}"})
        page = self.pages.get(self.current_page())
        if page is not None and page is not self.source:
            page.shown()
        self.show_project_name()
        self.save_project()
        self.workspace.update_states()

    def open_results(self, folder, maps=None):
        """Open a project with PIV results but no images (its folder; maps: the folder of its vector fields, None:
        its subfolder PIV) in the pages that don't need images, with its project if it has one."""
        self.source.initialize_session()  # No images, the other pages without those of the previous project
        self.results_folder = os.path.normpath(folder)
        try:
            data = read_project(folder)
        except (OSError, ValueError) as e:
            self.update_status(f"The project of the folder could not be read: {e}")
            data = None
        if data is not None:
            self.restore_project(data)
        post = self.page("post")
        if post is not None and maps is not None:
            post.panel.maps.set(maps)
        if not self.history:
            self.history.append({"date": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "step": "results",
                                 "text": f"PIV/PTV results without images: {maps or os.path.join(folder, 'PIV')}"})
        page = self.pages.get(self.current_page())
        if page is not None:
            page.shown()
        self.show_project_name()
        self.save_project()
        self.workspace.update_states()

    def close_project(self):
        """No project open any more (a new workspace): its images, its PIV/PTV results and the vector fields of the
        post-processing unloaded from the pages (its changes saved first), the pages as at the start."""
        self.source.initialize_session()  # The images; the other pages without those of the project (images_changed)
        self.results_folder = None
        post = self.page("post")
        if post is not None:
            post.loaded = post.post = None
            post.panel.maps.refresh()
            post.clear()
            post.update_panel()
        page = self.pages.get(self.current_page())
        if page is not None and page is not self.source:
            page.shown()
        self.show_project_name()

    def show_results(self, project, method):
        """The results of a pipeline (PaIRS, TracTrac...) of the project open in the post-processing: its maps folder (kept in
        the project)."""
        post, folder = self.page("post"), project.results_folders().get(method)
        if post is None or folder is None:
            return
        post.panel.maps.set(folder)
        self.update_status(f"{project.label()}: its {method} results in the post-processing ({folder})")

    def busy(self):
        """Why another project can't be opened now (a process running: the PIV, or a process with a progress bar such
        as the pre-processing), None if it can."""
        if getattr(self.page("piv"), "run", None) is not None:
            return "The PIV/PTV is running: stop it before opening another project."
        if self.root.grab_current() is not None:
            return "A process is running (its progress bar at the bottom): wait for its end or stop it first."
        if self.batch_window is not None:
            return "A batch is running (its window): wait for its end or stop it first."
        return None

    def batch_step(self, key):
        """The batch step of a page (key: preprocessing, piv, postprocessing)."""
        for page in self.pages.values():
            step = page.batch_step() if hasattr(page, "batch_step") else None
            if step is not None and step.key == key:
                return step
        raise ValueError(f"No page for the batch settings {key}")

    def open_workspace_project(self, project, method=None):
        """Open a project of the workspace: its images (with its project), else its results; method: with its
        results of this pipeline (PaIRS, TracTrac...) in the post-processing (show_results)."""
        reason = self.busy()
        if reason:
            self.msgbox(reason, "Projects", "warn")
            return
        folders = project.results_folders()
        if project.images:
            file = next((os.path.join(project.folder, n) for n in PROJECT_FILES
                         if os.path.isfile(os.path.join(project.folder, n))), None)
            if file is None or not self.open_project_file(file, quiet=True):
                self.source.open_images(project.folder, restore=True)
            if method in folders:
                self.show_results(project, method)
        else:
            self.open_results(project.folder, folders.get(method, project.maps))
        # The name given in the workspace, if the project has none
        if project.name and self.name is None and self.project_folder():
            self.name = project.name
            self.show_project_name()
            self.save_project()

    def reload_project(self):
        """Open the project open again from its file (changed by a batch run)."""
        folder = self.project_folder()
        if folder is None:
            return
        self.flush_project()
        if self.results_folder is not None:
            post = self.page("post")
            maps = None if post is None or post.panel.maps.is_default() else post.maps_folder()
            self.open_results(folder, maps)
        elif not self.open_project_file(self.project_file(), quiet=True):
            self.source.open_images(folder, restore=True)

    def folder_in_use(self, folder):
        """A page reads the files of this folder (they mustn't be overwritten)."""
        return any(page.uses_folder(folder) for page in self.pages.values())

    ############################################################################
    # Project: VelociHub_project.json in the folder of the images

    def project_folder(self):
        """Folder of the project open: the one of the images, or of the PIV results without images (None)."""
        return self.source.image_folder() or self.results_folder

    def project_file(self):
        folder = self.project_folder()
        return os.path.join(folder, PROJECT_FILE) if folder else None

    def project_name(self):
        """Name of the project, at the start of the names of the files exported: the one chosen, or the name of its
        folder."""
        return self.name or default_project_name(self.project_folder())

    def ask_project_name(self):
        """Choose the name of the project (empty: the name of the folder of the images)."""
        if self.project_folder() is None:
            self.msgbox('No images selected. Please choose the folder of the images (side panel, "Images").',
                        "Project", "warn")
            return
        answer = self.inputdlg(["Name of the project, at the start of the names of the files exported (empty: the "
                                "name of the folder of the images)"], "Project name", [self.name or ""])
        if answer is None:
            return
        self.name = clean_name(answer[0]) or None
        self.show_project_name()
        self.update_status(f"Project name: {self.project_name()} (files exported: {self.project_name()}_[task]_"
                           "[element]_[number])")
        self.save_project()
        project = self.workspace.ws.project(self.project_folder())
        if project is not None:
            project.name = self.name
            self.workspace.changed()

    def show_project_name(self):
        """The name of the project in the title of the window."""
        name = self.project_name()
        self.window.title(f"{name} · {self.title}" if name else self.title)

    def project_path(self, path):
        """A path as the project keeps it, in its history too: relative to its folder."""
        return relative(path, self.project_folder())

    def log_step(self, step, text):
        """A step done, kept in the history of the project (its paths: project_path)."""
        self.history.append({"date": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "step": step, "text": text})
        self.save_project()

    def save_project(self):
        """Save the project shortly (the changes made together are saved once)."""
        if self.restoring or self.project_file() is None or self.project_job is not None:
            return
        self.project_job = self.root.after(300, self.write_project)

    def flush_project(self):
        """Save now the changes waiting to be saved."""
        if self.project_job is not None:
            self.write_project()

    def write_project(self):
        """Save the project now."""
        if self.project_job is not None:
            self.root.after_cancel(self.project_job)
            self.project_job = None
        filename = self.project_file()
        if self.restoring or filename is None:
            return
        try:
            state = self.project_state()
            # The sections of the pages that aren't in the window are kept (project of the same images)
            try:
                # The project of the folder (also the file of the previous versions)
                previous = read_project(os.path.dirname(filename)) or {}
            except (OSError, ValueError):
                previous = {}
            if self.source.image_folder() is None:
                # PIV results only: the sections of the pages that don't need images, the others kept
                kept = {key: value for key, value in previous.items() if key not in ("format", "version", "saved")}
                own = ["name", "history"] + [p.project_key for p in self.pages.values() if not p.needs_images]
                state = {**kept, **{key: state[key] for key in own}}
                state.setdefault("images", {"folder": ".", "files": []})
            elif previous.get("images", {}).get("files") == state["images"]["files"]:
                state.update({key: value for key, value in previous.items()
                              if key not in state and key not in ("format", "version", "saved")})
            write_project(filename, state)
        except (OSError, TypeError, ValueError) as e:
            self.update_status(f"The project could not be saved in {filename}: {e}")

    def project_state(self):
        """Images, settings and results of the pages, and the steps done, as saved in the project."""
        root = self.project_folder()
        state = {"name": self.name, "images": {"folder": ".", "files": self.source.image_files()},
                 "image_shown": self.image_index + 1}
        for page in self.pages.values():
            state[page.project_key] = page.project_state(lambda path: relative(path, root))
        state["history"] = self.history[-500:]
        return state

    def check_project(self):
        """Restore the project of the folder of the images opened: at once if it is the project of these images,
        if the user wants it otherwise (its settings only)."""
        try:
            data = read_project(self.source.image_folder())
        except (OSError, ValueError) as e:
            self.update_status(f"The project of the folder could not be read: {e}")
            return
        if data is None:
            return
        files = data.get("images", {}).get("files", [])
        if files != self.source.image_files():
            Message = (f"The folder has the project of other images ({len(files)} images"
                       + (f": {files[0]} ... {files[-1]}" if files else "") + "). Restore its settings? The project "
                       "is then replaced by the one of these images.")
            if self.questdlg(Message, "Project", ["Restore the settings", "New project"],
                             "Restore the settings") != "Restore the settings":
                return
            data = {**data, "history": [], "image_shown": 1}
        self.restore_project(data)

    def restore_project(self, data):
        """Restore the settings and the results of the pages from a project of the images opened."""
        root = self.project_folder()
        shown = data.get("image_shown")
        n_images = len(self.source.image_files())
        self.image_index = shown - 1 if isinstance(shown, int) and 1 <= shown <= n_images else 0
        self.restoring = True
        try:
            for page in self.pages.values():
                # Without images: the pages that don't need them only
                if page.needs_images and self.source.image_folder() is None:
                    continue
                page.restore(data.get(page.project_key) or {}, lambda path: absolute(path, root))
            self.history = list(data.get("history") or [])
            self.name = clean_name(data.get("name")) or None
        finally:
            self.restoring = False
        self.show_project_name()
        self.update_status(f"Project restored from {self.project_file()}")
        for line in reversed(summary(data)):
            self.update_status("   " + line)

    def open_project_dialog(self):
        """Open a project file: its images, with their settings and results."""
        filename = filedialog.askopenfilename(parent=self.parent(), title="Open project...",
                                              initialdir=self.source.image_folder() or os.getcwd(),
                                              filetypes=PROJECT_FILE_TYPES)
        if filename:
            self.open_project_file(filename)

    def open_project_file(self, filename, quiet=False):
        """Open the images of a project file, with their settings and results. Returns whether it was opened
        (quiet: no message box if it wasn't)."""
        report = self.update_status if quiet else self.status_error
        try:
            data = read_project(filename)
        except (OSError, ValueError) as e:
            report(f"The project could not be read: {e}")
            return False
        folder = os.path.dirname(os.path.abspath(filename))
        files = [f for f in data.get("images", {}).get("files", []) if os.path.isfile(os.path.join(folder, f))]
        if len(files) < 2:
            report(f"The images of the project were not found in {folder}")
            return False
        self.source.open_images(folder, files=files, restore=data)
        # A project opened with another name is saved as the project of the folder
        if not same_path(filename, self.project_file() or ""):
            self.update_status(f"The project is saved in {self.project_file()}")
        return True

    ############################################################################
    # Status, progress and dialogs

    def update_status(self, msg):
        """Append the message contained in the variable msg to the status log at the bottom of the main window,
        the most recent first. msg = -1 or 0: the status is cleared. The blank separators of the original
        status window aren't shown, to keep the recent messages visible in the few lines of the log."""
        if isinstance(msg, str):
            msg = msg.replace("\n", " ")
            if msg.strip():
                self.status_listbox.insert(0, msg)
        elif msg in (0, -1):
            self.status_listbox.delete(0, tk.END)

    def status_error(self, msg):
        """Show an error message and write it in the status window."""
        self.msgbox(msg, "Error", "error")
        self.update_status(msg)

    def progress(self, title, stoppable=False):
        """Progress of a process, at the bottom of the window (MATLAB waitbar; stoppable: with a button Stop, which
        ends the process with Stopped)."""
        return Waitbar(self.progress_area, title, stoppable)

    def parent(self):
        """Window the user is working in, parent of the dialogs."""
        try:
            widget = self.root.focus_get()
        except (KeyError, tk.TclError):
            widget = None
        if widget is None or not widget.winfo_viewable():
            return self.window
        return widget.winfo_toplevel()

    def inputdlg(self, prompts, title, defaults):
        return InputDialog(self.parent(), prompts, title, defaults).run()

    def questdlg(self, message, title, buttons, default):
        return QuestionDialog(self.parent(), message, title, buttons, default).run()

    def msgbox(self, message, title, icon="info"):
        show = {"info": messagebox.showinfo, "warn": messagebox.showwarning, "error": messagebox.showerror}[icon]
        show(title, message, parent=self.parent())

    def report_exception(self, exc, val, tb):
        """Show the errors in a message box (MATLAB prints them in the command window); a process stopped by the
        user (Stopped) in the status only."""
        if issubclass(exc, Stopped):
            self.update_status("Stopped by the user")
            return
        traceback.print_exception(exc, val, tb)
        messagebox.showerror("Error", f"{exc.__name__}: {val}", parent=self.parent())

    def open_explorer(self, folder):
        """Open a folder (folder(): its path) in the file explorer."""
        path = folder()
        if not path or not os.path.isdir(path):
            self.msgbox(f"The folder doesn't exist yet: {path or 'no images selected'}", "Open in explorer", "warn")
            return
        winopen(path)

    def show_credits(self):
        """Help > Credits: the authors, the licence and the open-source software used (credits_window.py)."""
        from .credits_window import show_credits
        show_credits(self.window)

    ############################################################################
    def exit_program(self):
        # Wait for the end of the process when a waitbar or a dialog is open
        if self.root.grab_current() is not None:
            return
        # Each page may keep the window open (the PIV running, if the user doesn't want to stop it)
        for page in reversed(list(self.pages.values())):
            if not page.on_exit():
                return
        self.write_project()
        # Close all the windows
        self.root.destroy()


def main(folder=None, pattern=None, masks=None, contrast=None, pages=PAGES):
    """Run the interface. Optionally, with the training images of a folder (all the images, or those matching
    the pattern), the masks of a file saved by save_masks, and contrast enhancement settings (a dictionary of
    the arguments of PODBackgroundRemoval.set_contrast), which replace those of the project of the images.
    Returns when the window is closed."""
    if sys.platform == "win32":
        # Sharp text and figures on scaled displays
        try:
            ctypes.OleDLL("shcore").SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    app = MainWindow(root, pages)
    pre = app.source
    if masks is not None:
        pre.load_masks(masks)
        pre.masks_changed()
    if contrast is not None:
        pre.set_contrast(**contrast)
        # In the interface, the contrast enhancement is off when no filter is selected
        if not pre.contrast_enabled:
            pre.set_contrast(filters=())
        pre.panel.show_contrast_settings(pre.contrast)
    if folder is not None:
        pre.open_images(folder, pattern, restore=masks is None and contrast is None)
    root.mainloop()
