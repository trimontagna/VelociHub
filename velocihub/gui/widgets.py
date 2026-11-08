"""Widgets of the interface: Matplotlib figure panels, waitbar, dialogs, side panels and help dots, and the
preferences kept from a session to the next."""

import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.transforms import blended_transform_factory

from ..preprocessing import Stopped, as2d, same_path, title_lines
from ..settings import (COLORMAPS)


################################################################################
# MATLAB functions


def winopen(path):
    """MATLAB winopen(path) on a folder."""
    if sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


def reveal(path):
    """Show a path in the file explorer: a folder opened, a file selected in its folder; a path that doesn't exist
    (yet): the nearest folder above it that exists. Returns the folder or the file shown (None: none)."""
    path = os.path.normpath(path) if path else None
    while path and not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    if not path or not os.path.exists(path):
        return None
    if os.path.isfile(path):
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", path])
        else:
            winopen(os.path.dirname(path))
    else:
        winopen(path)
    return path


def folder_icon(widget):
    """The icon of the buttons revealing a path in the explorer: a folder (16 x 16, drawn once per window)."""
    root = widget._root()
    icon = getattr(root, "_velocihub_folder_icon", None)
    if icon is None:
        icon = tk.PhotoImage(master=root, width=16, height=16)
        dark, light, front = "#9a7424", "#f2c45a", "#e3ab35"
        icon.put(dark, to=(1, 2, 7, 4))  # The tab
        icon.put(dark, to=(1, 4, 15, 14))  # The outline
        icon.put(front, to=(2, 5, 14, 13))
        icon.put(light, to=(2, 7, 14, 13))
        root._velocihub_folder_icon = icon
    return icon


def magic(n):
    """MATLAB magic(n) for n multiple of 4."""
    I, J = np.mgrid[1 : n + 1, 1 : n + 1]
    M = np.arange(1, n * n + 1).reshape(n, n)
    K = (I % 4) // 2 == (J % 4) // 2
    M[K] = n * n + 1 - M[K]
    return M


################################################################################
# Windows and dialogs


def place(win, pos):
    """Set the position of a window from a MATLAB normalized 'outerposition' [left bottom width height]."""
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    left, bottom, width, height = pos
    # The size set by Tk excludes the borders and the title bar of the window
    win.geometry(f"{int(width * sw) - 16}x{int(height * sh) - 40}+{int(left * sw)}+{int((1 - bottom - height) * sh)}")


def alive(win):
    """MATLAB isfield(hand, 'figure_x') && ishandle(hand.figure_x)."""
    return win is not None and bool(win.winfo_exists())


def bring_to_front(win):
    """MATLAB figure(h)."""
    win.deiconify()
    win.lift()


def add_menu(menu, items):
    """Add to a menu the items (label, accelerator, callback), (label, [submenu items]) or None (separator)."""
    for item in items:
        if item is None:
            menu.add_separator()
        elif isinstance(item[1], list):
            submenu = tk.Menu(menu, tearoff=0)
            add_menu(submenu, item[1])
            menu.add_cascade(label=item[0], menu=submenu)
        else:
            label, accelerator, command = item
            menu.add_command(label=label, accelerator=accelerator or "", command=command)


class Toolbar(NavigationToolbar2Tk):
    """Matplotlib zoom, pan and save tools (the subplot tool doesn't work with the constrained layout)."""

    toolitems = [item for item in NavigationToolbar2Tk.toolitems if item[0] != "Subplots"]


DEFAULT_COLORMAP = "Default"


def ask_colour_range(app, name, current, custom=None):
    """Colour range of the colour bar of a figure (name: in the question), asked to the user from the current one
    (vmin, vmax) or the one chosen before (custom): (vmin, vmax), a None value for the automatic one; None if
    cancelled or invalid (the user is told)."""
    answer = app.inputdlg([f"{name}: colour bar from (empty: automatic)", "to"], "Colour range",
                          ["" if v is None else f"{v:.6g}" for v in custom] if custom is not None
                          else [f"{v:.4g}" for v in current])
    if answer is None:
        return None
    try:
        clim = tuple(float(v) if v.strip() else None for v in answer)
    except ValueError:
        app.msgbox("Enter numbers (or nothing for the automatic range).", "Colour range", "warn")
        return None
    if None not in clim and clim[0] >= clim[1]:
        app.msgbox("The minimum must be below the maximum.", "Colour range", "warn")
        return None
    return clim


def colour_tools(bar, before, on_cmap, on_clim):
    """Colour tools of a tile, next to one another before the Matplotlib tools: its colour map (ColormapBox) and
    the button of its colour range (short: the bar of a tile is narrow). Returns the list of the colour maps."""
    box = ColormapBox(bar, on_cmap, width=7)
    box.pack(side=tk.LEFT, padx=(0, 2), before=before)
    button = tk.Button(bar, text="Range", command=on_clim)
    button.pack(side=tk.LEFT, padx=(0, 4), before=before)
    hover_tip(button, "Colour range of the colour bar (min, max)")
    hover_tip(box, "Colour map of the figure (Default: the one of the figure)")
    return box


def hover_tip(widget, text):
    """The text (or text(): its text when shown) in a tip while the mouse is over a widget."""
    state = {"tip": None}

    def show(event):
        message = text() if callable(text) else text
        if state["tip"] is not None or not message:
            return
        tip = state["tip"] = tk.Toplevel(widget)
        tip.wm_overrideredirect(True)
        tip.attributes("-topmost", True)
        tk.Label(tip, text=message, justify=tk.LEFT, background="#ffffe8", foreground="#222222", relief=tk.SOLID,
                 borderwidth=1, padx=6, pady=3).pack()
        tip.wm_geometry(f"+{widget.winfo_rootx()}+{widget.winfo_rooty() + widget.winfo_height() + 2}")

    def hide(event=None):
        if state["tip"] is not None:
            state["tip"].destroy()
            state["tip"] = None
    widget.bind("<Enter>", show, add="+")
    widget.bind("<Leave>", hide, add="+")
    widget.bind("<ButtonPress>", hide, add="+")


class ColormapBox(ttk.Combobox):
    """List of the colour maps of the figure of a tile: Default (the one chosen for the figure) or another one.
    command(cmap) is called with the colour map chosen (None: the default)."""

    def __init__(self, master, command, width=9):
        super().__init__(master, values=(DEFAULT_COLORMAP,) + COLORMAPS, state="readonly", width=width)
        self.set(DEFAULT_COLORMAP)
        self.bind("<<ComboboxSelected>>",
                  lambda event: command(None if self.get() == DEFAULT_COLORMAP else self.get()))

    def show(self, cmap, enabled=True):
        """The colour map of the figure shown (None: the default); disabled for the figures without a colour map."""
        self.set(cmap or DEFAULT_COLORMAP)
        self.configure(state="readonly" if enabled else tk.DISABLED)


class FigureSelector(ttk.Menubutton):
    """Menu button of the figures of a tile: a submenu per group of figures (groups: [(group, keys)]; group None:
    the figures at the root of the menu; titles: their names; parents: {group: submenu} the groups gathered in a
    submenu, at the place of the first of them), the figure shown on the button (after prefix; short(key): its text
    on the button, its name in full in a tip). command(key) is called with the figure chosen."""

    def __init__(self, master, groups, titles, command, width=36, prefix="", parents=None, short=None):
        super().__init__(master, width=width)
        self.groups, self.titles, self.command, self.prefix = groups, titles, command, prefix
        self.parents, self.short, self.current = dict(parents or {}), short, None
        if short is not None:
            hover_tip(self, lambda: self.prefix + self.titles[self.current] if self.current is not None else "")
        self.menu = tk.Menu(self, tearoff=False)
        self["menu"] = self.menu
        self.var = tk.StringVar(self)
        self.keys = self.disabled = None

    def show(self, current, keys=None, disabled=()):
        """The figure shown (current), among the figures available (keys: all those of the groups by default), those
        disabled greyed out (a group greyed out when all of its figures are)."""
        keys = list(keys) if keys is not None else [k for _, members in self.groups for k in members]
        disabled = set(disabled)
        if keys != self.keys or disabled != self.disabled:
            self.keys, self.disabled = keys, disabled
            self.menu.delete(0, tk.END)
            for child in self.menu.winfo_children():
                child.destroy()
            # The submenus gathering groups ({label: menu}) and the figures in them ({label: keys})
            parents, parent_keys = {}, {}
            for group, members in self.groups:
                available = [key for key in members if key in keys]
                if not available:
                    continue
                parent = self.menu
                label = self.parents.get(group)
                if label is not None:
                    if label not in parents:
                        parents[label] = tk.Menu(self.menu, tearoff=False)
                        parent_keys[label] = set()
                        self.menu.add_cascade(label=label, menu=parents[label])
                    parent = parents[label]
                    parent_keys[label] |= set(available)
                sub = self.menu if group is None else tk.Menu(parent, tearoff=False)
                for key in available:
                    sub.add_radiobutton(label=self.titles[key], value=key, variable=self.var,
                                        command=lambda key=key: self.command(key),
                                        state=tk.DISABLED if key in disabled else tk.NORMAL)
                if group is not None:
                    parent.add_cascade(label=group, menu=sub,
                                       state=tk.DISABLED if disabled >= set(available) else tk.NORMAL)
            for label, members in parent_keys.items():
                self.menu.entryconfigure(label, state=tk.DISABLED if disabled >= members else tk.NORMAL)
        self.var.set(current)
        self.current = current
        self.configure(text=self.prefix + (self.short(current) if self.short is not None else self.titles[current]))


class FigurePanel(ttk.Frame):
    """Matplotlib figure with its toolbar, tiled in the main window (a MATLAB figure in the original).
    The panels of a series (images, modes) have a slider at the bottom, which calls on_slide(i) with the index
    chosen, and become active (on_activate) when the mouse enters them or clicks in them: the arrow keys browse
    the series of the active panel, marked by a colored strip at the top (lighter on the panels linked to it, which
    follow it). The panel of the images has a second
    slider to compare them before and after the processing (on_compare(fraction)): the image before is shown left
    of a vertical divider, at the fraction of the width chosen, and the image after right of it. The labels of the
    ends of this slider are compare_labels, its position at the start compare_value; show_compare shows or hides
    it."""

    ACTIVE_COLOR = "#3b8ed8"
    LINKED_COLOR = "#a9cdef"
    # Around the image: where the masks can be drawn beyond its edges
    OUTSIDE_COLOR = "#d9d9d9"
    DIVIDER_COLOR = "#ffcc00"
    # Delay (ms) before showing the item chosen with the slider, so that only the last one is shown while dragging
    SLIDE_DELAY = 40

    def __init__(self, master, tools=(), layout="constrained", figsize=(5, 4), clabel="", on_slide=None,
                 on_activate=None, item="Image", on_compare=None, axis_labels=("x (px)", "y (px)"),
                 compare_labels=("Before", "After"), compare_value=1):
        super().__init__(master)
        # Labels of the colorbar and of the axes of the images
        self.clabel = clabel
        self.axis_labels = axis_labels
        self.fig = Figure(figsize=figsize, layout=layout)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.strip = tk.Frame(self, height=3)
        self.strip.pack(side=tk.TOP, fill=tk.X)
        self.inactive_color = self.strip["background"]
        # Toolbar: buttons of the image panels followed by the Matplotlib tools
        bar = self.bar = tk.Frame(self)
        for text, command in tools:
            tk.Button(bar, text=text, command=command).pack(side=tk.LEFT, padx=1)
        self.toolbar = Toolbar(self.canvas, bar, pack_toolbar=False)
        self.toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        bar.pack(side=tk.TOP, fill=tk.X)

        # Slider of the series
        self.on_slide = on_slide
        self.item = item
        self.pending = None
        if on_slide is not None:
            bottom = ttk.Frame(self, padding=(8, 2, 8, 4))
            bottom.pack(side=tk.BOTTOM, fill=tk.X)
            self.slider_label = ttk.Label(bottom, width=17, anchor=tk.E)
            self.slider_label.pack(side=tk.RIGHT, padx=(8, 0))
            self.slider = ttk.Scale(bottom, orient=tk.HORIZONTAL, from_=1, to=1, command=self.slid)
            self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.set_slider(1, 0)

        # Slider of the comparison before / after the processing: position of the divider
        self.compare_bar = None
        if on_compare is not None:
            compare = self.compare_bar = ttk.Frame(self, padding=(8, 2, 8, 0))
            compare.pack(side=tk.BOTTOM, fill=tk.X)
            # Labels of the ends of the slider (set_compare_labels)
            self.compare_ends = (ttk.Label(compare, text=compare_labels[0]),
                                 ttk.Label(compare, text=compare_labels[1], width=17))
            self.compare_ends[0].pack(side=tk.LEFT)
            self.compare_ends[1].pack(side=tk.RIGHT, padx=(8, 0))
            self.compare_slider = ttk.Scale(compare, orient=tk.HORIZONTAL, from_=0, to=1, value=compare_value,
                                            command=lambda value: on_compare(float(value)))
            self.compare_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
            # Why there is no comparison, in place of the slider (compare_note)
            self.compare_label = ttk.Label(compare, foreground="#555555")
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # The figure takes the focus when the mouse enters it, so that the arrow keys browse its series (and not
        # the text of an entry of the side panel, which applies its value)
        self.canvas.get_tk_widget().bind("<Enter>", lambda event: self.take_focus(), add="+")
        if on_activate is not None:
            for widget in (self.canvas.get_tk_widget(), *([self.slider] if on_slide else [])):
                widget.bind("<Enter>", lambda event: on_activate(), add="+")
                widget.bind("<Button-1>", lambda event: on_activate(), add="+")
        self.ax = self.im = self.cbar = None
        # Image before the processing, divider of the comparison and its position (None: no comparison)
        self.before = None
        self.divider = []
        self.fraction = None

    def show_compare(self, shown):
        """Show or hide the slider of the comparison (in its place, above the slider of the series)."""
        if shown and not self.compare_bar.winfo_manager():
            self.compare_bar.pack(side=tk.BOTTOM, fill=tk.X, before=self.canvas.get_tk_widget())
        elif not shown:
            self.compare_bar.pack_forget()

    def set_compare_labels(self, left, right):
        """Labels of the ends of the slider of the comparison."""
        for label, text in zip(self.compare_ends, (left, right)):
            label.configure(text=text)

    def set_compare_value(self, value):
        """Position of the slider of the comparison, set without calling on_compare."""
        command = self.compare_slider.cget("command")
        self.compare_slider.configure(command="")
        self.compare_slider.set(value)
        self.compare_slider.configure(command=command)

    def compare_note(self, text):
        """Text in place of the slider of the comparison (why there is none); None: the slider."""
        if text:
            self.compare_slider.pack_forget()
            self.compare_label.configure(text=text)
            self.compare_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        else:
            self.compare_label.pack_forget()
            self.compare_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

    def set_slider(self, i, n):
        """Position of the slider: item i of n (n = 0: nothing to browse)."""
        # Set without calling on_slide, and enabled meanwhile: a disabled scale ignores the value set
        self.slider.configure(to=max(n, 1), command="")
        self.slider.state(["!disabled"])
        self.slider.set(i)
        self.slider.configure(command=self.slid)
        if n <= 1:
            self.slider.state(["disabled"])
        self.slider_label["text"] = f"{self.item} {i} of {n}" if n else ""

    def slid(self, value):
        """The slider moved: the item is shown when the slider stops."""
        i = int(round(float(value)))
        self.slider_label["text"] = f"{self.item} {i} of {int(self.slider['to'])}"
        if self.pending is not None:
            self.after_cancel(self.pending)
        self.pending = self.after(self.SLIDE_DELAY, lambda: self.slide_to(i))

    def slide_to(self, i):
        self.pending = None
        self.on_slide(i)

    def take_focus(self):
        """Focus on the figure, unless a dialog or a list is open."""
        widget = self.canvas.get_tk_widget()
        try:
            focus = self.focus_get()
        except (KeyError, tk.TclError):  # The focus is in a popdown list
            return
        if focus is None or focus.winfo_toplevel() is widget.winfo_toplevel():
            widget.focus_set()

    def set_active(self, active):
        """True: the arrow keys browse the series of this panel; "linked": it follows the active one."""
        self.strip["background"] = (self.ACTIVE_COLOR if active is True else self.LINKED_COLOR if active == "linked"
                                    else self.inactive_color)

    def message(self, text):
        """Empty figure with a message, until its content is evaluated."""
        self.fig.clf()
        self.ax = self.im = self.cbar = self.before = None
        self.divider = []
        self.fig.text(0.5, 0.5, text, ha="center", va="center", color="0.45", wrap=True)
        self.canvas.draw_idle()

    def imagesc(self, img, cmap, clim=None, title="", colorbar=False):
        """MATLAB cla; imagesc(img); caxis(clim); colormap(cmap); axis ij equal tight; title(title)
        Without clim, the color limits are the extremes of the image. Returns the color limits."""
        img = as2d(img)
        if clim is None:
            clim = (np.nanmin(img), np.nanmax(img))
        self.fig.clf()
        self.before = None
        self.divider = []
        self.ax = self.fig.add_subplot()
        self.ax.set_facecolor(self.OUTSIDE_COLOR)
        # The divider of the comparison stays at its fraction of the view when it is zoomed or panned
        self.ax.callbacks.connect("xlim_changed", self.place_divider)
        H, W = img.shape
        # Pixel centers have integer coordinates starting from 1, as in MATLAB
        self.im = self.ax.imshow(img, cmap=cmap, vmin=clim[0], vmax=clim[1], extent=(0.5, W + 0.5, H + 0.5, 0.5))
        # The masks drawn over the image don't change its limits
        self.ax.set_autoscale_on(False)
        # Broken into lines here: Matplotlib's wrapping shrinks the axes when a colour bar is added (title_lines)
        self.ax.set_title(title_lines(title), fontsize="medium")
        self.ax.set_xlabel(self.axis_labels[0])
        self.ax.set_ylabel(self.axis_labels[1])
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, label=self.clabel) if colorbar else None
        self.toolbar.update()
        self.canvas.draw_idle()
        return clim

    def set_image(self, img, clim=None, title=""):
        """Replace the image shown, keeping the colormap and the zoom (imagesc if the size changes).
        Returns the color limits."""
        img = as2d(img)
        if self.im is None or self.im.get_array().shape != img.shape:
            return self.imagesc(img, "gray", clim, title)
        if clim is None:
            clim = (np.nanmin(img), np.nanmax(img))
        self.im.set_data(img)
        self.im.set_clim(*clim)
        self.ax.set_title(title_lines(title), fontsize="medium")
        self.canvas.draw_idle()
        return clim

    def set_colormap(self, cmap, colorbar):
        """MATLAB colormap(cmap) followed by colorbar or colorbar hide."""
        self.im.set_cmap(cmap)
        if self.before is not None:
            self.before.set_cmap(cmap)
        if colorbar and self.cbar is None:
            self.cbar = self.fig.colorbar(self.im, ax=self.ax, label=self.clabel)
        elif not colorbar and self.cbar is not None:
            self.cbar.remove()
            self.cbar = None
        self.canvas.draw_idle()

    def compare(self, before, fraction, clim=None):
        """Comparison before / after: the image before (None: not shown) left of a vertical divider at the
        fraction of the width of the view (it follows the zoom), the image after (self.im) right of it. Each one
        has its own color limits, unless clim is given. The zoom is kept."""
        if self.im is None:
            return
        if before is None or fraction >= 1:
            self.fraction = None
            if self.before is not None:
                self.before.set_visible(False)
            self.place_divider()
            self.canvas.draw_idle()
            return

        before = as2d(before)
        if clim is None:
            clim = (np.nanmin(before), np.nanmax(before))
        if self.before is None or self.before.get_array().shape != before.shape:
            if self.before is not None:
                self.before.remove()
            # Under the image after, with the same colormap
            self.before = self.ax.imshow(before, cmap=self.im.get_cmap(), vmin=clim[0], vmax=clim[1],
                                         extent=self.im.get_extent(), zorder=self.im.get_zorder() - 0.5)
        else:
            self.before.set_data(before)
            self.before.set_clim(*clim)
        self.before.set_visible(True)
        self.fraction = fraction
        self.place_divider()
        self.canvas.draw_idle()

    def place_divider(self, ax=None):
        """Divider of the comparison at its fraction of the width of the view: the image after is clipped right
        of it. Called again when the view changes (zoom, pan)."""
        for artist in self.divider:
            artist.remove()
        self.divider = []
        if self.im is None:
            return
        if self.fraction is None:
            # Clipped by the axes only, as an image added to the axes
            self.im.set_clip_path(self.ax.patch)
            self.im.set_visible(True)
            return
        H, W = self.im.get_array().shape
        x0, x1 = sorted(self.ax.get_xlim())
        x = x0 + self.fraction * (x1 - x0)
        # The image after is clipped by the axes and right of the divider (by a polygon: Matplotlib turns a
        # Rectangle into a clip box, which would replace the clipping by the axes)
        self.im.set_visible(self.fraction > 0)
        right = MplPolygon([[x, 0.5], [W + 0.5, 0.5], [W + 0.5, H + 0.5], [x, H + 0.5]], transform=self.ax.transData)
        self.im.set_clip_path(right)
        self.im.set_clip_box(self.ax.bbox)
        if self.fraction > 0:
            self.divider.append(self.ax.axvline(x, color=self.DIVIDER_COLOR, linewidth=1.5))
        # Labels at the top of the divider
        at_top = blended_transform_factory(self.ax.transData, self.ax.transAxes)
        box = dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none")
        for text, ha in (("◀ before ", "right"), (" after ▶", "left")):
            self.divider.append(self.ax.text(x, 0.98, text, transform=at_top, ha=ha, va="top", color="white",
                                             fontsize="small", bbox=box, clip_on=True))


class ScrolledFrame(ttk.Frame):
    """A frame scrolled vertically (its widgets in inner): a scroll bar at its right, the mouse wheel over it; as wide
    as its widgets at least (height: its height at first)."""

    def __init__(self, master, height=200):
        super().__init__(master)
        background = ttk.Style(self).lookup("TFrame", "background") or None
        self.canvas = canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, height=height,
                                         background=background)
        bar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar.pack(side=tk.LEFT, fill=tk.Y)
        self.inner = ttk.Frame(canvas)
        window = canvas.create_window(0, 0, window=self.inner, anchor=tk.NW)

        def fit(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"), width=self.inner.winfo_reqwidth())
        self.inner.bind("<Configure>", fit)
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(
            window, width=max(event.width, self.inner.winfo_reqwidth())))
        for widget in (canvas, self.inner):
            widget.bind("<Enter>", lambda event: self.bind_all("<MouseWheel>", self.wheel), add="+")
            widget.bind("<Leave>", lambda event: self.unbind_all("<MouseWheel>"), add="+")

    def wheel(self, event):
        if self.canvas.yview() != (0.0, 1.0):
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")


class PathBox(ttk.Frame):
    """Folder in an entry, chosen with "...", or entered (Return or the focus leaving the entry); "↺" (if reset)
    goes back to the default one; the folder icon reveals it in the file explorer (path_shown). chosen(folder) is
    called with the folder (None for the default one)."""

    def __init__(self, master, title, chosen, reset=True, width=28):
        super().__init__(master)
        self.title, self.chosen = title, chosen
        self.text = ""  # Folder shown
        self.var = tk.StringVar(self)
        self.entry = ttk.Entry(self, textvariable=self.var, width=width)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", lambda event: self.entered())
        self.entry.bind("<FocusOut>", lambda event: self.entered())
        self.buttons = [ttk.Button(self, text="...", width=3, command=self.browse)]
        if reset:
            self.buttons.append(ttk.Button(self, text="↺", width=3, command=lambda: self.chosen(None)))
        for button in self.buttons:
            button.pack(side=tk.LEFT, padx=(4, 0))
        # Revealed in the explorer, even when the box is greyed out
        self.reveal_button = ttk.Button(self, image=folder_icon(self), command=self.reveal, padding=(3, 1))
        self.reveal_button.pack(side=tk.LEFT, padx=(4, 0))

    def show(self, folder, enabled=True):
        self.text = folder or ""
        self.var.set(self.text)
        # The end of the path is shown: the name of the folder
        self.entry.xview_moveto(1)
        self.configure_state(enabled)

    def configure_state(self, enabled):
        for widget in [self.entry] + self.buttons:
            widget.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def entered(self):
        text = self.var.get().strip()
        if text != self.text:
            self.chosen(os.path.normpath(text) if text else None)

    def path_shown(self):
        """The path revealed in the explorer: the one of the box."""
        return self.text

    def reveal(self):
        """Reveal the path of the box in the file explorer (the nearest folder that exists)."""
        if reveal(self.path_shown()) is None:
            self.bell()

    def browse(self):
        start = self.text
        while start and not os.path.isdir(start) and os.path.dirname(start) != start:
            start = os.path.dirname(start)
        folder = filedialog.askdirectory(parent=self.winfo_toplevel(), title=self.title, initialdir=start or None)
        if folder:
            self.chosen(os.path.normpath(folder))


class FileBox(PathBox):
    """A file to write in an entry, chosen with "..." (save as: filetypes, the extension added), or entered; "↺"
    goes back to the default one (chosen(None))."""

    def __init__(self, master, title, chosen, filetypes=(("All files", "*.*"),), extension="", width=28):
        super().__init__(master, title, chosen, width=width)
        self.filetypes, self.extension = list(filetypes), extension

    def browse(self):
        path = filedialog.asksaveasfilename(parent=self.winfo_toplevel(), title=self.title,
                                            initialdir=os.path.dirname(self.text) or None,
                                            initialfile=os.path.basename(self.text), filetypes=self.filetypes,
                                            defaultextension=self.extension)
        if path:
            self.chosen(os.path.normpath(path))


class FolderEntry(PathBox):
    """Folder used by a step: by default, the designated subfolder of a root folder (the folder of the images), or
    the folder given by default() if it is a function; or the folder entered or chosen with "...", kept for this
    root folder only. "↺" goes back to the default folder. on_change() is called when the folder changes;
    refresh() shows it (when the root folder or the default one change). needs_root=False: a folder can be chosen
    without root folder."""

    def __init__(self, master, root, default, title, on_change=None, needs_root=True, width=28):
        super().__init__(master, title, self.set, width=width)
        self.root, self.default_folder, self.on_change, self.needs_root = root, default, on_change, needs_root
        self.custom = None  # (root, folder) chosen by the user

    def default(self):
        if callable(self.default_folder):
            return self.default_folder()
        root = self.root()
        return os.path.join(root, self.default_folder) if root else None

    def get(self):
        """Folder used: the one chosen for the current root folder, or the default one."""
        if self.custom is not None and same_path(self.custom[0] or "", self.root() or ""):
            return self.custom[1]
        return self.default()

    def is_default(self):
        return self.get() == self.default()

    def set(self, folder):
        """Folder chosen (None or the default one: the default one)."""
        before = self.get()
        root = self.root()
        if folder and (root or not self.needs_root) and not same_path(folder, self.default() or ""):
            self.custom = (root, os.path.normpath(os.path.abspath(folder)))
        else:
            self.custom = None
        self.refresh()
        if self.on_change is not None and not same_path(before or "", self.get() or ""):
            self.on_change()

    def refresh(self):
        """Show the folder (the root folder or the default one changed)."""
        self.show(self.get(), bool(self.root()) or not self.needs_root)

    def configure_state(self, enabled):
        super().configure_state(enabled and (bool(self.root()) or not self.needs_root))


class Waitbar(ttk.Frame):
    """MATLAB waitbar, shown at the bottom of the main window. It takes the grab, so that the GUI can't be used
    while the images are processed. stoppable: with a button Stop, after which the next call raises Stopped (the
    process ends there)."""

    def __init__(self, master, title, stoppable=False):
        super().__init__(master, padding=(8, 4))
        self.stopped = False
        ttk.Label(self, text=title).pack(side=tk.LEFT)
        self.bar = ttk.Progressbar(self, length=360, maximum=1.0)
        self.bar.pack(side=tk.LEFT, padx=10)
        if stoppable:
            self.stop_button = ttk.Button(self, text="Stop", command=self.stop)
            self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        self.label = ttk.Label(self)
        self.label.pack(side=tk.LEFT)
        self.pack(fill=tk.X)
        self.update()
        self.take_grab()

    def take_grab(self):
        try:
            self.grab_set()
        except tk.TclError:  # The main window isn't visible
            pass

    def stop(self):
        self.stopped = True
        self.stop_button.configure(text="Stopping...", state=tk.DISABLED)

    def __call__(self, x, message=None):
        """MATLAB waitbar(x, wh, message). Raises Stopped once Stop was pressed."""
        if self.stopped:
            raise Stopped()
        self.bar["value"] = x
        if message is not None:
            self.label["text"] = message
        # A dialog may have taken the grab in the meantime
        if self.grab_status() is None:
            self.take_grab()
        self.update()
        if self.stopped:
            raise Stopped()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.grab_release()
        self.destroy()


class Dialog(tk.Toplevel):
    """Base of the modal dialogs replacing MATLAB's inputdlg and questdlg."""

    def __init__(self, parent, title):
        super().__init__(parent)
        self.withdraw()
        self.title(title)
        self.resizable(False, False)
        if parent.winfo_viewable():
            self.transient(parent)
        self.result = None
        self.initial_focus = self
        self.wrap = int(self.winfo_fpixels("4.5i"))
        self.frame = ttk.Frame(self, padding=12)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.bind("<Escape>", lambda event: self.destroy())
        self.bind("<Return>", lambda event: event.widget.invoke() if isinstance(event.widget, ttk.Button) else self.ok())

    def ok(self):
        pass

    def show(self):
        """Show the dialog over its parent, modal (the other windows don't take the events), without waiting."""
        parent = self.master
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_reqwidth()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_reqheight()) // 3
        self.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        self.deiconify()
        self.initial_focus.focus_set()
        previous_grab = self.grab_current()
        self.wait_visibility()
        self.grab_set()
        return previous_grab

    def run(self):
        """Show the dialog over its parent, wait until it is closed and return its result."""
        previous_grab = self.show()
        self.wait_window()
        if previous_grab is not None and previous_grab.winfo_exists():
            previous_grab.grab_set()
        return self.result


class RangeSlider(tk.Canvas):
    """Two-sided slider: the first and the last integers of a range from lo to hi, each with its handle (the one
    nearest to the click is dragged), the first never after the last. command(first, last) is called when they
    change, on_release(first, last, handle) when a handle dragged is released (handle: 0 the first, 1 the last);
    set_enabled greys it out."""

    PAD = 9
    RADIUS = 6
    COLOR, TRACK, DISABLED = "#3b8ed8", "#c8c8c8", "#a0a0a0"

    def __init__(self, master, lo, hi, command=None, width=340, height=24, on_release=None):
        background = ttk.Style(master).lookup("TFrame", "background") or None
        super().__init__(master, width=width, height=height, highlightthickness=0, background=background)
        self.lo, self.hi, self.command, self.on_release = int(lo), max(int(hi), int(lo)), command, on_release
        self.values, self.active, self.enabled = [self.lo, self.hi], None, True
        self.bind("<Configure>", lambda event: self.draw())
        self.bind("<Button-1>", self.press)
        self.bind("<B1-Motion>", self.drag)
        self.bind("<ButtonRelease-1>", self.release)
        self.draw()

    def release(self, event=None):
        handle, self.active = self.active, None
        if handle is not None and self.on_release is not None:
            self.on_release(*self.values, handle)

    def x_of(self, value):
        width = max(self.winfo_width(), int(self["width"]))
        return self.PAD + (value - self.lo) / max(self.hi - self.lo, 1) * (width - 2 * self.PAD)

    def value_at(self, x):
        width = max(self.winfo_width(), int(self["width"]))
        value = self.lo + (x - self.PAD) / max(width - 2 * self.PAD, 1) * (self.hi - self.lo)
        return int(min(max(round(value), self.lo), self.hi))

    def draw(self):
        self.delete("all")
        y = int(self["height"]) // 2
        color = self.COLOR if self.enabled else self.DISABLED
        self.create_line(self.x_of(self.lo), y, self.x_of(self.hi), y, fill=self.TRACK, width=4, capstyle=tk.ROUND)
        a, b = (self.x_of(v) for v in self.values)
        self.create_line(a, y, b, y, fill=color, width=4)
        for x in (a, b):
            self.create_oval(x - self.RADIUS, y - self.RADIUS, x + self.RADIUS, y + self.RADIUS, fill="white",
                             outline=color, width=2)

    def press(self, event):
        if not self.enabled:
            return
        a, b = (self.x_of(v) for v in self.values)
        # The nearest handle; on the same point, the one that can move towards the click
        self.active = 0 if abs(event.x - a) < abs(event.x - b) or (a == b and event.x < a) else 1
        self.drag(event)

    def drag(self, event):
        if not self.enabled or self.active is None:
            return
        value = self.value_at(event.x)
        value = min(value, self.values[1]) if self.active == 0 else max(value, self.values[0])
        if value != self.values[self.active]:
            self.values[self.active] = value
            self.draw()
            if self.command is not None:
                self.command(*self.values)

    def get(self):
        return tuple(self.values)

    def set(self, first, last):
        first = int(min(max(first, self.lo), self.hi))
        self.values = [first, int(min(max(last, first), self.hi))]
        self.draw()
        if self.command is not None:
            self.command(*self.values)

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self.draw()

    def set_range(self, lo, hi):
        """Another range (the values kept in it), without calling command."""
        self.lo, self.hi = int(lo), max(int(hi), int(lo))
        first = int(min(max(self.values[0], self.lo), self.hi))
        self.values = [first, int(min(max(self.values[1], first), self.hi))]
        self.draw()


class InputDialog(Dialog):
    """MATLAB inputdlg: one entry per prompt. The result is the list of answers, None if cancelled."""

    def __init__(self, parent, prompts, title, defaults):
        super().__init__(parent, title)
        self.entries = []
        for prompt, default in zip(prompts, defaults):
            ttk.Label(self.frame, text=prompt, wraplength=self.wrap, justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 2))
            entry = ttk.Entry(self.frame, width=50)
            entry.insert(0, default)
            entry.pack(fill=tk.X)
            self.entries.append(entry)
        buttons = ttk.Frame(self.frame)
        buttons.pack(anchor=tk.E, pady=(12, 0))
        ttk.Button(buttons, text="OK", command=self.ok).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side=tk.LEFT)
        self.entries[0].select_range(0, tk.END)
        self.initial_focus = self.entries[0]

    def ok(self):
        self.result = [entry.get() for entry in self.entries]
        self.destroy()


class QuestionDialog(Dialog):
    """MATLAB questdlg. The result is the label of the button pressed, None if the dialog is closed."""

    def __init__(self, parent, message, title, buttons, default):
        super().__init__(parent, title)
        ttk.Label(self.frame, text=message, wraplength=self.wrap, justify=tk.LEFT).pack(anchor=tk.W)
        row = ttk.Frame(self.frame)
        row.pack(pady=(12, 0))
        for text in buttons:
            button = ttk.Button(row, text=text, command=lambda text=text: self.choose(text))
            button.pack(side=tk.LEFT, padx=3)
            if text == default:
                self.initial_focus = button

    def choose(self, text):
        self.result = text
        self.destroy()


class SidePanel(ttk.Frame):
    """Side panel of a page, in three tiles: the inputs at the top (header), the settings of the process in the
    middle, which scroll if the screen is small, and the folder of the results with the buttons that run the
    process at the bottom (footer)."""

    def __init__(self, master):
        super().__init__(master)
        self.wrap = int(self.winfo_fpixels("2.9i"))
        ttk.Style(self).configure("Export.TButton", font=("Segoe UI", 10, "bold"), padding=(18, 6))
        # The header and the footer take the height of their content, the settings the rest
        self.header = ttk.Frame(self, padding=(8, 6, 8, 4))
        self.header.pack(side=tk.TOP, fill=tk.X)
        self.footer = ttk.Frame(self, padding=(8, 4, 8, 6))
        self.footer.pack(side=tk.BOTTOM, fill=tk.X)
        middle = ttk.Frame(self)
        middle.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(middle, highlightthickness=0, borderwidth=0)
        scroll = ttk.Scrollbar(middle, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner = ttk.Frame(self.canvas, padding=(8, 4))
        self.canvas.create_window((0, 0), window=self.inner, anchor=tk.NW)
        self.inner.bind("<Configure>", lambda event: self.canvas.configure(
            scrollregion=self.canvas.bbox("all"), width=self.inner.winfo_reqwidth()))
        self.bind_all("<MouseWheel>", self.scroll, add=True)

    def section(self, title, area=None, help=None):
        """Section of the settings (area: the header or the footer instead), with a help dot after its title."""
        box = ttk.LabelFrame(area or self.inner, text=title, padding=8)
        if help:
            label = ttk.Frame(box)
            ttk.Label(label, text=title).pack(side=tk.LEFT)
            HelpDot(label, help).pack(side=tk.LEFT, padx=(4, 0))
            box.configure(labelwidget=label)
        box.pack(fill=tk.X, pady=(0, 8) if area is None else 0)
        return box

    @staticmethod
    def row(box, pady=3):
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=pady)
        return row

    @staticmethod
    def toggle(box, text, variable, command):
        """On/off button of a step of the recipe: sunken and colored when on."""
        button = tk.Checkbutton(box, text=text, variable=variable, command=command, indicatoron=False,
                                selectcolor="#bcdcf5", padx=6, pady=4)
        button.pack(fill=tk.X, pady=(0, 6))
        return button

    @staticmethod
    def grid(box):
        """Grid of settings: label, value and help dot on each row."""
        grid = ttk.Frame(box)
        grid.pack(fill=tk.X)
        grid.columnconfigure(0, weight=1)
        return grid

    def setting(self, grid, label, variable, help=None, width=9, command=None):
        """Row of a grid of settings: a label, an entry (a check button if the variable is a BooleanVar) and a
        help dot. command is called when the value is entered (Return, or the focus leaves the entry). Returns the
        entry or the check button."""
        r = grid.grid_size()[1]
        if isinstance(variable, tk.BooleanVar):
            widget = ttk.Checkbutton(grid, text=label, variable=variable, command=command)
            widget.grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=1)
        else:
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky=tk.W, padx=(0, 4), pady=1)
            widget = ttk.Entry(grid, textvariable=variable, width=width)
            widget.grid(row=r, column=1, sticky=tk.E, pady=1)
            if command is not None:
                widget.bind("<Return>", lambda event: command())
                widget.bind("<FocusOut>", lambda event: command())
        if help:
            HelpDot(grid, help).grid(row=r, column=2, padx=(4, 0))
        return widget

    def scroll(self, event):
        """Mouse wheel over the panel."""
        widget = self.winfo_containing(event.x_root, event.y_root)
        if (widget is not None and str(widget).startswith(str(self.canvas))
                and not isinstance(widget, ttk.Treeview)):
            self.canvas.yview_scroll(-event.delta // 120, "units")


class HelpDot(tk.Canvas):
    """Small round "?" next to a setting: its help is shown while the mouse is over it, and stays after a click
    (until the next click)."""

    SIZE = 15
    COLOR = "#3b8ed8"

    def __init__(self, master, text):
        background = ttk.Style(master).lookup("TFrame", "background") or "SystemButtonFace"
        super().__init__(master, width=self.SIZE, height=self.SIZE, highlightthickness=0, borderwidth=0,
                         background=background, cursor="question_arrow")
        s = self.SIZE
        self.create_oval(1, 1, s - 1, s - 1, fill=self.COLOR, outline=self.COLOR)
        self.create_text(s / 2, s / 2, text="?", fill="white", font=("Segoe UI", 8, "bold"))
        self.text = text
        self.tip = None
        self.pinned = False
        self.bind("<Enter>", lambda event: self.show())
        self.bind("<Leave>", lambda event: None if self.pinned else self.hide())
        self.bind("<Button-1>", self.click)

    def show(self):
        if self.tip is not None:
            return
        self.tip = tk.Toplevel(self)
        self.tip.wm_overrideredirect(True)
        self.tip.attributes("-topmost", True)
        tk.Label(self.tip, text=self.text, justify=tk.LEFT, background="#ffffe8", foreground="#222222",
                 relief=tk.SOLID, borderwidth=1, wraplength=int(self.winfo_fpixels("3.6i")), padx=8,
                 pady=6).pack()
        self.tip.update_idletasks()
        x, y = self.winfo_rootx() + self.SIZE + 4, self.winfo_rooty() + self.SIZE
        width = self.tip.winfo_reqwidth()
        if x + width > self.winfo_screenwidth():
            x = max(0, self.winfo_rootx() - width - 4)
        self.tip.wm_geometry(f"+{x}+{y}")

    def hide(self):
        self.pinned = False
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None

    def click(self, event):
        if self.pinned:
            self.hide()
        else:
            self.show()
            self.pinned = True


class Page(ttk.Frame):
    """Page of the main window, for a step of the processing. The window builds its menus from menus(), binds the
    keyboard shortcuts given as their accelerators, sends it the arrow keys (arrow) when it is shown, and saves its
    settings and results in the project of the images (section project_key). Other pages are reached with
    app.page(key), which is None if that page isn't in the window: each page can be removed from the window
    (PAGES in app.py) with the pages that need it."""

    key = ""  # Name used by the other pages (app.page(key))
    title = ""  # Title of its tab
    project_key = ""  # Section of the project
    # The page works on the images (a project with PIV results only isn't opened in it, nor restored)
    needs_images = True

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app

    def menus(self):
        """Menus of the page: [(menu label, [items])], an item being (label, accelerator, command), (label,
        [items]) or None (separator). The items of a menu shared by several pages ("File", "Settings", "Help") are
        put together."""
        return []

    def folders(self):
        """Folders of the page opened by File > Open in explorer: [(label, function returning the folder)]."""
        return []

    def start(self):
        """The window is built (with the menus)."""

    def shown(self):
        """The page is shown."""

    def arrow(self, step):
        """Arrow key pressed on the page: previous (-1) or next (1) item of the panel under the mouse."""

    def images_changed(self):
        """Other images were opened (or none)."""

    def uses_folder(self, folder):
        """A process of the page reads the files of this folder: they mustn't be overwritten."""
        return False

    def project_state(self, rel):
        """Settings and results saved in the project (rel(path): the path saved)."""
        return {}

    def restore(self, data, path):
        """Restore the settings and the results of the project (path(saved): the path restored)."""

    def on_exit(self):
        """The window is closed: False keeps it open."""
        return True
