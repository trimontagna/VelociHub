"""The stitching in the window Batch of the post-processing (batch_dialog.py): the stitchings defined in the tree of
the projects, the options of velocihub/archiving/post_archive.py below the node settings.

Node stitching is done in order, the first one is the reference, then Ctrl+click on others to add or remove. 
Name pattern matching is performed (For example, if stitching is selected between FOV1, FOV2 and FOV3 nodes,
and each FOV has two nodes, 110Hz and 130Hz, then FOV1/110Hz/A will be stitched in order with FOV2/110Hz/A then FOV3/110Hz/A). 
A chain of projects is only stitched if all of the ones in the chain is exported."""

import os
import re
import tkinter as tk
from tkinter import ttk

from ..workspace import key


def same(a, b):
    """Two nodes (root, path) the same."""
    return key(a[0]) == key(b[0]) and a[1] == b[1]


def related(a, b):
    """Two nodes (root, path) one in the other (or the same)."""
    if key(a[0]) != key(b[0]):
        return False
    pa, pb = a[1], b[1]
    return not pa or not pb or pa == pb or pb.startswith(pa + "/") or pa.startswith(pb + "/")


def physical(settings):
    """Settings of the post-processing of a project in the physical space, with its camera (the stitching)."""
    camera = settings.get("camera") or {}
    return settings.get("space", "physical") == "physical" and bool(camera.get("fov_width_mm"))


def fov_labels(chain):
    """The names of the fields of view of a chain of projects (in the stitched figures): the parts of their paths
    that differ (FOV1, FOV2...), else their paths (with their root if they are the same)."""
    parts = [p.path.split("/") for p in chain]
    if len({len(x) for x in parts}) == 1:
        labels = ["/".join(part for i, part in enumerate(x) if any(other[i] != part for other in parts)) for x in parts]
        if all(labels) and len(set(labels)) == len(labels):
            return labels
    labels = [p.path or p.label() for p in chain]
    if len(set(labels)) < len(labels):
        labels = [f"{os.path.basename(p.root)}/{label}" for p, label in zip(chain, labels)]
    return labels


def setup_names(chains):
    """The names of the stitchings of chains of projects (their folders and files of stitched figures): the parts
    of the paths common to their projects (110Hz_A), else the name of the reference; unique."""
    names = []
    for chain in chains:
        parts = [p.path.split("/") for p in chain]
        common = [part for i, part in enumerate(parts[0])
                  if all(len(other) == len(parts[0]) and other[i] == part for other in parts)]
        name = re.sub(r'[\\/:*?"<>|]+', "_", "_".join(common) or chain[0].label()).strip() or "Stitching"
        unique, n = name, 2
        while unique in names:
            unique, n = f"{name} ({n})", n + 1
        names.append(unique)
    return names


class StitchOptions:
    """The stitchings of the window Batch of the post-processing (the column Stitching of the tree, Ctrl+click) and
    their options (the globals of velocihub/archiving/post_archive.py, r)."""

    # The options of the velocities matched (the offsets: the camera settings of each project)
    FIELDS = [("MATCH_JET", "Jet cells: at least (fraction of the max)", float),
              ("MATCH_EDGE", "Edge columns left out", int)]
    # The title of the box of the options, the column of the tree: the stitching of each node
    title, column, column_width = "Stitching", "Stitching", 210
    # The settings of each field of view, kept when a project takes those of the reference of its stitching: its camera
    # (its width, offsets and frame rate: the positions of its maps)
    own_paths = (("camera",),)
    below_tree = True  # The box of the options: below the tree of the projects

    def __init__(self, r, ws):
        self.r, self.ws, self.dialog = r, ws, None
        self.chains, self.texts = [], {}  # The stitchings: [[(root, path) of the nodes]]; the column by node

    def attach(self, dialog):
        self.dialog = dialog

    def build(self, parent, saved):
        self.chains = [[(node["root"], node.get("path") or "") for node in chain
                        if isinstance(node, dict) and node.get("root")]
                       for chain in saved.get("stitchings") or [] if isinstance(chain, list)]
        self.chains = [chain for chain in self.chains if len(chain) > 1]
        values = saved.get("globals") or {}
        self.flags, self.vars = {}, {}
        # The options greyed while no stitching is run (enabled)
        self.velocities = velocities = ttk.LabelFrame(parent, text="Velocity matching (Hack)", padding=6)
        velocities.pack(fill=tk.X)
        self.add_flags(velocities, values, (("MATCH_VELOCITY", "Matched in the overlaps"),))
        self.add_fields(velocities, values, self.FIELDS)
        ttk.Label(parent, text="Stitching: select a reference node, then Ctrl+click the others in order to add or remove. "
                               "Name patten matching is performed within the node tree structure to figure out stiching automatically between common nodes. "
                               "Figures for each project are still exported. Stitched results are in the reference's folder. "
                               "Right click any stitched node to open the alignment window.",
                  foreground="#555555", wraplength=340, justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))

    def add_flags(self, parent, values, flags):
        for name, text in flags:
            self.flags[name] = tk.BooleanVar(parent, values.get(name, getattr(self.r, name)))
            ttk.Checkbutton(parent, text=text, variable=self.flags[name]).pack(anchor=tk.W, pady=(4, 0))

    def add_fields(self, parent, values, fields):
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.X, pady=(4, 0))
        for i, (name, text, _) in enumerate(fields):
            self.vars[name] = tk.StringVar(parent, str(values.get(name, getattr(self.r, name))))
            ttk.Label(grid, text=text).grid(row=i, column=0, sticky=tk.W)
            ttk.Entry(grid, textvariable=self.vars[name], width=12).grid(row=i, column=1, sticky=tk.W, pady=1,
                                                                         padx=(6, 0))

    # ------------------------------------------------------------------ stitchings

    def node_of(self, iid):
        item = self.dialog.items.get(iid)
        return None if item is None else (item[1], item[2])

    def iid_of(self, node):
        """The item of a node in the tree (None if it isn't there)."""
        project = next((p for p in self.ws.projects if same((p.root, p.path), node)), None)
        iid = "project|" + key(project.folder) if project else self.dialog.node_id(*node)
        return iid if iid in self.dialog.items else None

    def below(self, iid):
        """The projects of a node by their path below it ({"": project} for a project node)."""
        path = self.node_of(iid)[1]
        out = {}
        for p in self.dialog.projects_below(iid):
            out["" if p.path == path else p.path[len(path) + 1:] if path else p.path] = p
        return out

    def resolve(self):
        """The chains of projects of the stitchings: [(items of the nodes, projects (None: not found), reason not
        stitched or None)]."""
        out, used = [], set()
        dialog = self.dialog
        for chain in self.chains:
            iids = [iid for iid in (self.iid_of(node) for node in chain) if iid]
            if len(iids) < 2:
                continue
            maps = [self.below(iid) for iid in iids]
            labels = [dialog.tree.item(iid, "text") for iid in iids]
            for path in maps[0]:
                projects = [found.get(path) for found in maps]
                missing = [label for label, p in zip(labels, projects) if p is None]
                reason = None
                if missing:
                    reason = f"{', '.join(missing)}: no {path or 'project node'}"
                else:
                    for p in projects:
                        if not dialog.runnable(p):
                            reason = f"{p.path} can't be run"
                        elif key(p.folder) not in dialog.checked:
                            reason = f"{p.path} isn't checked"
                        elif key(p.folder) in used:
                            reason = f"{p.path} is in another stitching"
                        elif not physical(dialog.settings_of(p)):
                            reason = f"{p.path}: no camera settings (the stitching needs the physical space)"
                        if reason:
                            break
                if reason is None:
                    used.update(key(p.folder) for p in projects)
                out.append((iids, projects, reason))
        return out

    def prepare(self, dialog):
        """The column Stitching of the nodes (before the tree is refreshed): the nodes chosen in their order, each
        reference project with the projects it is stitched to, the others with their reference."""
        self.texts = {}
        resolved = self.resolve()
        for chain in self.chains:
            iids = [iid for iid in (self.iid_of(node) for node in chain) if iid]
            labels = [dialog.tree.item(iid, "text") for iid in iids]
            for n, iid in enumerate(iids if len(iids) > 1 else []):
                if dialog.items[iid][0] != "project":
                    self.texts[iid] = ("Reference → " + " → ".join(labels[1:]) if n == 0
                                       else f"Stitched to {labels[0]} ({n + 1} of {len(iids)})")
        for _, projects, reason in resolved:
            reference = projects[0]
            for n, p in enumerate(projects):
                if p is None:
                    continue
                iid = "project|" + key(p.folder)
                if reason is not None:
                    self.texts[iid] = f"Not stitched: {reason}"
                elif n == 0:
                    self.texts[iid] = "This project → " + " → ".join(other.path for other in projects[1:])
                else:
                    self.texts[iid] = f"Stitched to {reference.path}"

        self.enable(any(reason is None for _, _, reason in resolved))

    def enable(self, on):
        """The options usable (a stitching run), else greyed."""
        def walk(widget):
            for child in widget.winfo_children():
                try:
                    child.state(["!disabled"] if on else ["disabled"])
                except (AttributeError, tk.TclError):
                    pass
                walk(child)
        self.velocities.state(["!disabled"] if on else ["disabled"])
        walk(self.velocities)

    def cell(self, dialog, iid):
        return self.texts.get(iid, "")

    def group_of(self, iid):
        """The stitching of a node (one of its nodes, or a project of it): (labels of its nodes, chains of projects
        found in all its nodes [(label, projects)], projects of each node, index of the chain of the project
        clicked), None if it isn't in one."""
        for chain in self.chains:
            iids = [i for i in (self.iid_of(node) for node in chain) if i]
            if len(iids) < 2:
                continue
            maps = [self.below(i) for i in iids]
            chains = [(path or projects[0].path, projects) for path in maps[0]
                      for projects in [[found.get(path) for found in maps]] if None not in projects]
            hit = next((n for n, (_, projects) in enumerate(chains)
                        if any("project|" + key(p.folder) == iid for p in projects)), None)
            if iid in iids or hit is not None:
                labels = [self.dialog.tree.item(i, "text") for i in iids]
                targets = [self.dialog.targets(i) for i in iids]
                return labels, chains, targets, hit or 0
        return None

    def reference_of(self, project):
        """The reference project of the stitching of a project (the first one of its chain of projects), None if it
        isn't in one."""
        iid = "project|" + key(project.folder)
        group = self.group_of(iid)
        if group is None:
            return None
        for _, projects in group[1]:
            if any(key(p.folder) == key(project.folder) for p in projects):
                return projects[0]
        return None

    def context_items(self, dialog, iid):
        """The menu of a node of a stitching: Align the stitching... (alignment.py)."""
        group = self.group_of(iid)
        if group is None:
            return []
        labels, chains, targets, choice = group
        label = "Align the stitching (" + " → ".join(labels) + ")..."
        if not chains:
            return [(label + ": no project in all its nodes", None)]

        def align():
            from .alignment import AlignWindow

            AlignWindow(dialog, labels, chains, targets, choice)
        return [(label, align)]

    def ctrl_click(self, dialog, iid):
        """Ctrl+click on a node: out of its stitching if it is in one, else added to the stitching of the node
        selected (a new one: the node selected its reference). Returns whether it was done."""
        node = self.node_of(iid)
        for chain in self.chains:
            hit = next((n for n in chain if same(n, node)), None)
            if hit is not None:
                chain.remove(hit)
                self.chains = [c for c in self.chains if len(c) > 1]
                return True
        anchor = dialog.tree.focus()
        if not anchor or anchor == iid or anchor not in dialog.items:
            return False
        first = self.node_of(anchor)
        chain = next((c for c in self.chains if any(same(n, first) for n in c)), None)
        if any(related(n, node) for n in chain or [first]):
            return False  # A node in another one of the stitching
        if chain is None:
            chain = [first]
            self.chains.append(chain)
        chain.append(node)
        return True

    # ------------------------------------------------------------------ the run

    def read(self):
        values = {name: flag.get() for name, flag in self.flags.items()}
        try:
            for name, text, kind in self.FIELDS:
                values[name] = kind(self.vars[name].get().strip())
        except ValueError:
            raise ValueError("Options: numbers expected in the stitching options") from None
        return {"globals": values,
                "stitchings": [[{"root": root, "path": path} for root, path in chain] for chain in self.chains]}

    def job(self, projects):
        """The options of the run: with stitchings, its units (batch_run.py): the projects checked outside them, one
        by one, then each chain of projects stitched ({"setup", "members": [{"folder", "root", "fov"}], the
        reference first})."""
        options = self.read()
        del options["stitchings"]
        chains = [chain for _, chain, reason in self.resolve() if reason is None]
        if chains:
            stitched = {key(p.folder) for chain in chains for p in chain}
            units = [{"label": p.label(), "folders": [p.folder]} for p in projects if key(p.folder) not in stitched]
            for name, chain in zip(setup_names(chains), chains):
                units.append({"label": f"Stitching {name}: " + " → ".join(p.path for p in chain),
                              "folders": [p.folder for p in chain],
                              "stitching": {"setup": name, "members": [{"folder": p.folder, "root": p.root,
                                                                        "fov": fov}
                                                                       for p, fov in zip(chain, fov_labels(chain))]}})
            options["units"] = units
        return options
