"""Panel Projects, left of the pages: the workspace (workspace.py), its root folders and the projects found in them,
in the tree of their folders, folded and unfolded at each node. A click on a project opens it, as opening its images
did (or its vector fields, for a project with PIV/PTV results only: post-processing). A project with the results of a
PIV and of a PTV has a row for each below it: a click opens the project with those results in the post-processing
(its maps folder). The projects that the page shown
can't use are greyed out (without images: pre-processing and PIV; without PIV results: post-processing, where those
with images still open), the lost ones in red. Right click: add a root folder or a project, open, rename, remove from
the workspace (the files stay), relink (a folder, or a project file for a project), refresh.

The workspace is saved in its file at each change (the first root folder added without workspace file: in that
folder, VelociHub_workspace.json); the last one is opened again at the start."""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..project import PROJECT_FILE, PROJECT_FILES, clean_name, read_project, write_project
from ..workspace import WORKSPACE_FILE, Workspace, key
from .widgets import HelpDot, winopen
from ..prefs import getpref, setpref

WORKSPACE_FILES = [("VelociHub workspace", "*.json"), ("All files", "*.*")]
PROJECT_FILES = [("VelociHub project", PROJECT_FILES), ("JSON", "*.json")]
LOST, GREY = "#c62828", "#9a9a9a"
HELP = ("The root folders of the workspace are scanned for projects (folders with a project file, images or PIV/PTV "
        "results), shown in the tree of their folders. Click a project to open it. Greyed projects can't be used "
        "by the page shown (no images for the pre-processing and the PIV/PTV, no PIV/PTV results for the "
        "post-processing, where a project with images still opens); red ones weren't found at the last refresh "
        "(right click: relink). Img: images; PIV/PTV: the results it has (PIV, PTV or both). A project with both has "
        "a row for each below it (PIV results, PTV results): click one to post-process those results (the maps "
        "folder of the post-processing; in bold: those shown).\n\nRight click for the other actions. The workspace is saved in its file (File menu: new, open, save as). Archive projects... puts the files of projects in a ZIP or TAR "
        "file.")


class WorkspacePanel(ttk.Frame):
    """The tree of the projects of the workspace (ws) of the window (app)."""

    def __init__(self, master, app):
        super().__init__(master, padding=(4, 4, 0, 4))
        self.app = app
        self.ws = Workspace()
        self.items = {}  # Tree item: ("root", folder), ("node", root, path), ("project", Project) or ("results",
        # Project, method): the results of a method of a project with those of a PIV and of a PTV
        self.folded = set()  # Nodes folded (root|path)
        header = ttk.Frame(self)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Projects", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        HelpDot(header, HELP).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(header, text="⟳", width=3, command=self.refresh).pack(side=tk.RIGHT)
        ttk.Button(header, text="+ Root folder", command=self.add_root).pack(side=tk.RIGHT, padx=(0, 4))
        self.file_label = ttk.Label(self, foreground="#555555")
        self.file_label.pack(anchor=tk.W, pady=(2, 2))
        # The tools on the projects of the workspace, below them (packed first: always visible)
        tools = ttk.LabelFrame(self, text="Tools", padding=6)
        tools.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))
        ttk.Button(tools, text="Archive projects...", command=self.archive).pack(side=tk.LEFT)
        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(frame, columns=("images", "results"), selectmode="browse")
        self.tree.heading("#0", text="Project", anchor=tk.W)
        for column, text, width in (("images", "Img", 34), ("results", "PIV/PTV", 62)):
            self.tree.heading(column, text=text)
            self.tree.column(column, width=width, minwidth=30, stretch=False, anchor=tk.CENTER)
        self.tree.column("#0", width=230, minwidth=120)
        bar = ttk.Scrollbar(frame, command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.tag_configure("lost", foreground=LOST)
        self.tree.tag_configure("grey", foreground=GREY)
        self.tree.tag_configure("current", font=("TkDefaultFont", 9, "bold"))
        self.tree.tag_configure("running", foreground="#1565c0")
        self.running = None  # The folder of the project processed by a batch (set_running)
        self.tree.bind("<ButtonRelease-1>", self.click)
        self.tree.bind("<Return>", lambda event: self.open(self.selected_project()))
        self.tree.bind("<Button-3>", self.context_menu)
        self.show_file()

    def archive(self):
        from .archive_tools import archive_projects

        archive_projects(self.app)

    def start(self):
        """The workspace of the last session, refreshed."""
        file = getpref("workspace_file")
        if file and os.path.isfile(file):
            self.load(file)

    # ------------------------------------------------------------------ workspace file

    def menu_items(self):
        """Items of the File menu."""
        return [("New workspace", None, self.new), ("Open workspace...", None, self.open_dialog),
                ("Save workspace as...", None, self.save_as), ("Add a root folder...", None, self.add_root),
                ("Refresh the projects", None, self.refresh)]

    def load(self, file):
        try:
            ws = Workspace.load(file)
        except (OSError, ValueError) as e:
            self.app.status_error(f"The workspace could not be read: {e}")
            return
        self.ws = ws
        setpref("workspace_file", ws.file)
        self.refresh()
        self.app.update_status(f"Workspace {ws.file}: {len(ws.projects)} projects in {len(ws.roots)} root folders")

    def new(self):
        """A new workspace, empty: the project open and its results unloaded (not while a process runs)."""
        reason = self.app.busy()
        if reason:
            self.app.msgbox(reason, "New workspace", "warn")
            return
        self.ws = Workspace()
        setpref("workspace_file", None)
        self.app.close_project()  # In the new workspace (no project with results: the batch greyed)
        self.rebuild()
        self.app.update_status("New workspace: add a root folder (+ Root folder) to find its projects")

    def open_dialog(self):
        file = filedialog.askopenfilename(parent=self.app.parent(), title="Open workspace...",
                                          initialdir=os.path.dirname(self.ws.file or "") or os.getcwd(),
                                          filetypes=WORKSPACE_FILES)
        if file:
            self.load(file)

    def save_as(self):
        file = filedialog.asksaveasfilename(parent=self.app.parent(), title="Save workspace as...",
                                            initialdir=os.path.dirname(self.ws.file or "") or os.getcwd(),
                                            initialfile=WORKSPACE_FILE, defaultextension=".json",
                                            filetypes=WORKSPACE_FILES)
        if file:
            self.save(file)
            setpref("workspace_file", self.ws.file)

    def save(self, file=None):
        """Save the workspace in its file (none yet: not saved)."""
        if not (file or self.ws.file):
            return
        try:
            self.ws.save(file)
            setpref("workspace_file", self.ws.file)  # Opened again at the start
        except OSError as e:
            self.app.update_status(f"The workspace could not be saved in {file or self.ws.file}: {e}")
        self.show_file()

    def show_file(self):
        self.file_label["text"] = (f"Workspace: {os.path.basename(self.ws.file)}" if self.ws.file
                                   else "Workspace not saved yet")

    # ------------------------------------------------------------------ tree

    @staticmethod
    def node_id(root, path=""):
        return f"{key(root)}|{path}"

    def rebuild(self):
        """The tree of the workspace (the nodes folded kept)."""
        selected = self.tree.selection()
        self.folded = {iid for iid, item in self.items.items()
                       if item[0] != "results" and self.tree.exists(iid) and not self.tree.item(iid, "open")}
        self.tree.delete(*self.tree.get_children())
        self.items = {}
        for r in self.ws.roots:
            iid = self.node_id(r["folder"])
            # Its name first (the panel is narrow), then where it is
            name, parent = os.path.basename(r["folder"]) or r["folder"], os.path.dirname(r["folder"])
            text = f"{name}  ({'projects added, ' if not r['scan'] else ''}{parent})"
            self.tree.insert("", tk.END, iid=iid, text=text, open=iid not in self.folded)
            self.items[iid] = ("root", r["folder"])
        for p in self.ws.projects:
            parent = self.node_id(p.root)
            if not self.tree.exists(parent):
                continue
            parts = p.path.split("/") if p.path else []
            for i in range(len(parts) - 1):
                path = "/".join(parts[:i + 1])
                iid = self.node_id(p.root, path)
                if not self.tree.exists(iid):
                    self.tree.insert(parent, tk.END, iid=iid, text=parts[i], open=iid not in self.folded)
                    self.items[iid] = ("node", p.root, path)
                parent = iid
            iid = "project|" + key(p.folder)
            processed = self.running is not None and key(p.folder) == key(self.running)
            methods = list(p.results_folders())
            self.tree.insert(parent, tk.END, iid=iid, text=("▶ " if processed else "") + p.label(),
                             values=("✓" if p.images else "", "+".join(methods)), open=iid not in self.folded)
            self.items[iid] = ("project", p)
            # Both results: a row for each
            if len(methods) > 1:
                for method in methods:
                    child = f"results|{method}|{key(p.folder)}"
                    self.tree.insert(iid, tk.END, iid=child, text=f"{method} results", values=("", "✓"))
                    self.items[child] = ("results", p, method)
        if selected and self.tree.exists(selected[0]):
            self.tree.selection_set(selected[0])
        self.update_states()

    def projects_below(self, iid):
        """The projects of a node (all those below it)."""
        kind = self.items.get(iid, (None,))
        if kind[0] in ("project", "results"):
            return [kind[1]]
        out = []
        for child in self.tree.get_children(iid):
            out += self.projects_below(child)
        return out

    def usable(self, project):
        """The page shown can use the project (images; with the post-processing, PIV results)."""
        page = self.app.pages.get(self.app.current_page())
        return project.images if getattr(page, "needs_images", True) else project.results

    def openable(self, project):
        """The project can be opened from the page shown: its images, or its PIV results in the post-processing."""
        page = self.app.pages.get(self.app.current_page())
        return not project.lost and (project.images or (not getattr(page, "needs_images", True) and project.results))

    def shown_method(self, project):
        """The method of the results of the project open shown by the post-processing (its maps folder): PIV, PTV,
        or None (another folder)."""
        post = self.app.page("post")
        maps = post.maps_folder() if post is not None else None
        return next((m for m, folder in project.results_folders().items() if maps and key(folder) == key(maps)), None)

    def set_running(self, folder):
        """The project processed by a batch (None: none): in blue, ▶ before its name, shown in the tree."""
        self.running = folder
        for iid, item in self.items.items():
            if item[0] == "project":
                p = item[1]
                processed = folder is not None and key(p.folder) == key(folder)
                self.tree.item(iid, text=("▶ " if processed else "") + p.label())
                if processed:
                    self.tree.see(iid)
        self.update_states()

    def update_states(self):
        """The colours of the tree: greyed out for the page shown, lost in red, the project open in bold, the
        project processed by a batch in blue."""
        current = self.app.project_folder()
        for iid, item in self.items.items():
            if item[0] in ("project", "results"):
                p = item[1]
                tags = ["lost" if p.lost else "grey" if not self.usable(p) else ""]
                if current and key(p.folder) == key(current) and (item[0] == "project"
                                                                  or self.shown_method(p) == item[2]):
                    tags.append("current")
                if self.running and key(p.folder) == key(self.running):
                    tags.append("running")
            else:
                below = self.projects_below(iid)
                lost = (item[0] == "root" and (self.ws.root(item[1]) or {}).get("lost")) or (below and all(
                    p.lost for p in below))
                tags = ["lost" if lost else ""]
            self.tree.item(iid, tags=[t for t in tags if t])

    # ------------------------------------------------------------------ actions

    def selected_project(self):
        item = self.items.get(self.tree.focus())
        return item[1] if item and item[0] in ("project", "results") else None

    def click(self, event):
        """A project clicked (not its folding indicator): opened."""
        iid = self.tree.identify_row(event.y)
        if not iid or "indicator" in self.tree.identify_element(event.x, event.y):
            return
        item = self.items.get(iid)
        if item and item[0] == "project":
            self.open(item[1])
        elif item and item[0] == "results":
            self.open(item[1], item[2])

    def open(self, project, method=None):
        """Open a project in the pages (its images, or its PIV/PTV results in the post-processing); method: with its
        results of this method (PIV, PTV) in the post-processing."""
        if project is None:
            return
        current = self.app.project_folder()
        if current and key(project.folder) == key(current):
            if method is not None and not project.lost:
                self.app.show_results(project, method)
                self.update_states()
            return
        if project.lost:
            self.app.update_status(f"{project.label()}: not found in {project.folder} (right click: relink)")
            return
        if not self.openable(project):
            self.app.update_status(f"{project.label()}: no images for this page (PIV/PTV results only: open it from the "
                                   "post-processing page)")
            return
        self.app.open_workspace_project(project, method)
        self.update_states()

    def context_menu(self, event):
        iid = self.tree.identify_row(event.y)
        if iid:
            self.tree.selection_set(iid)
            self.tree.focus(iid)
        item = self.items.get(iid)
        menu = tk.Menu(self, tearoff=0)
        if item is not None and item[0] in ("project", "results"):
            p, method = item[1], item[2] if item[0] == "results" else None
            if method is not None:
                iid = "project|" + key(p.folder)  # Removed from the workspace: the project
            menu.add_command(label=f"Open with its {method} results" if method else "Open",
                             command=lambda: self.open(p, method),
                             state=tk.NORMAL if self.openable(p) else tk.DISABLED)
            menu.add_command(label="Rename...", command=lambda: self.rename(p))
            menu.add_command(label="Relink to a folder...", command=lambda: self.relink(item, "folder"))
            menu.add_command(label="Relink to a project file...", command=lambda: self.relink(item, "file"))
            menu.add_command(label="Open in explorer", command=lambda: self.explore(p.folder))
            menu.add_separator()
            menu.add_command(label="Remove from the workspace", command=lambda: self.remove(iid))
        elif item is not None:
            folder = item[1] if item[0] == "root" else os.path.join(item[1], *item[2].split("/"))
            menu.add_command(label="Relink to a folder...", command=lambda: self.relink(item, "folder"))
            menu.add_command(label="Unfold all", command=lambda: self.fold(iid, True))
            menu.add_command(label="Fold all", command=lambda: self.fold(iid, False))
            menu.add_command(label="Open in explorer", command=lambda: self.explore(folder))
            menu.add_separator()
            menu.add_command(label="Remove from the workspace", command=lambda: self.remove(iid))
        if item is not None:
            menu.add_separator()
        menu.add_command(label="Add a root folder (detect its projects)...", command=self.add_root)
        menu.add_command(label="Add a project (folder)...", command=lambda: self.add_project("folder"))
        menu.add_command(label="Add a project (project file)...", command=lambda: self.add_project("file"))
        menu.add_command(label="Refresh", command=self.refresh)
        menu.tk_popup(event.x_root, event.y_root)

    def fold(self, iid, open_):
        for child in (iid,) + self.descendants(iid):
            if self.items.get(child, ("project",))[0] not in ("project", "results"):
                self.tree.item(child, open=open_)

    def descendants(self, iid):
        out = ()
        for child in self.tree.get_children(iid):
            out += (child,) + self.descendants(child)
        return out

    def changed(self, message=None):
        """The workspace changed: saved, the tree shown again."""
        self.save()
        self.rebuild()
        if message:
            self.app.update_status(message)

    def refresh(self):
        """The roots scanned again (projects added, followed or lost)."""
        added = self.ws.refresh()
        lost = sum(p.lost for p in self.ws.projects)
        self.changed(f"Projects refreshed: {len(self.ws.projects)} projects" + (f", {len(added)} new" if added else "")
                     + (f", {lost} not found (in red: right click to relink)" if lost else ""))

    def add_root(self):
        folder = filedialog.askdirectory(parent=self.app.parent(), title="Root folder: its projects are detected",
                                         initialdir=os.path.dirname(self.ws.file or "") or os.getcwd())
        if not folder:
            return
        if self.ws.file is None:
            self.ws.file = os.path.join(folder, WORKSPACE_FILE)
            setpref("workspace_file", self.ws.file)
        added = self.ws.add_root(folder)
        self.changed(f"{len(added)} projects found in {folder} (workspace: {self.ws.file})")

    def add_project(self, how):
        parent = self.app.parent()
        if how == "file":
            path = filedialog.askopenfilename(parent=parent, title="Project file", filetypes=PROJECT_FILES)
            path = os.path.dirname(path) if path else None
        else:
            path = filedialog.askdirectory(parent=parent, title="Folder of the project")
        if not path:
            return
        if self.ws.file is None:
            self.ws.file = os.path.join(os.path.dirname(os.path.abspath(path)), WORKSPACE_FILE)
            setpref("workspace_file", self.ws.file)
        project = self.ws.add_project(path)
        if project is None:
            self.app.msgbox(f"No project in {path}: no project file, images or PIV/PTV results.", "Projects", "warn")
            return
        self.changed(f"Project added: {project.folder}")

    def rename(self, project):
        """The name of a project (empty: its folder): in its project file, at the start of the files exported."""
        answer = self.app.inputdlg(["Name of the project, at the start of the names of the files exported (empty: "
                                    "the name of its folder)"], "Rename the project", [project.label()])
        if answer is None:
            return
        name = clean_name(answer[0]) or None
        project.name = name
        current = self.app.project_folder()
        if current and key(project.folder) == key(current):
            self.app.name = name
            self.app.show_project_name()
            self.app.save_project()
        elif project.project_file:
            try:
                data = read_project(project.folder) or {}
                data = {k: v for k, v in data.items() if k not in ("format", "version", "saved")}
                write_project(os.path.join(project.folder, PROJECT_FILE), {**data, "name": name})
            except (OSError, ValueError) as e:
                self.app.update_status(f"The name could not be saved in the project file of {project.folder}: {e}")
        self.changed(f"Project {project.folder} renamed: {project.label()}")

    def remove(self, iid):
        item = self.items.get(iid)
        if item is None:
            return
        projects = self.projects_below(iid)
        what = (f"the root folder {item[1]} and its {len(projects)} projects" if item[0] == "root"
                else f"{len(projects)} project(s)")
        if not messagebox.askokcancel("Projects", f"Remove {what} from the workspace? Their files stay on disk.",
                                      parent=self.app.parent()):
            return
        if item[0] == "root":
            self.ws.remove_root(item[1])
        else:
            self.ws.remove(projects)
        self.changed(f"Removed from the workspace: {what}")

    def relink(self, item, how):
        """A root, a folder of the tree or a project found elsewhere (a folder; a project file for a project)."""
        parent = self.app.parent()
        if how == "file":
            path = filedialog.askopenfilename(parent=parent, title="Project file of the project",
                                              filetypes=PROJECT_FILES)
        else:
            path = filedialog.askdirectory(parent=parent, title="Folder where it is now")
        if not path:
            return
        if item[0] == "root":
            self.ws.relink_root(item[1], path)
        elif item[0] == "node":
            self.ws.relink_node(item[1], item[2], path)
        elif self.ws.relink_project(item[1], path) is None:
            self.app.msgbox(f"No project in {path}: no project file, images or PIV/PTV results.", "Relink", "warn")
            return
        lost = sum(p.lost for p in self.ws.projects)
        self.changed(f"Relinked to {path}" + (f"; {lost} projects still not found" if lost else ""))

    def explore(self, folder):
        if os.path.isdir(folder):
            winopen(folder)
        else:
            self.app.update_status(f"The folder doesn't exist: {folder}")
