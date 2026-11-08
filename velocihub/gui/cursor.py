"""The x-y cursor of the maps of the tiles of the post-processing: two lines through the point under the mouse, the
value of the map there marked by a dashed line on its colour bar, and its position and value in a box at the top left
of the map, named as the axes and the colour bar of the map name them (x = 48.0 mm, |u| = 0.101 m/s; 3 significant
digits). A left click freezes the cursor at the point (another one releases it); ignored while the zoom or the pan of
the toolbar is on. Drawn by blitting over the figure as last drawn (the figure itself isn't drawn again).

The maps are the colour-mapped artists of a figure with the field they show (their attribute field: (X, Y, F), the
positions as plotted, 2D grids or the 1D positions of the columns and of the rows of an image). With two maps compared
by a wiper, the value is that of the map on the side of the point (side(): the fraction of the width of the axes where
the wiper is; None without wiper)."""

import re

import numpy as np

# The cursor: its lines and the box of its values
LINE = dict(color="black", linewidth=0.8, alpha=0.75)
BOX = dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor="black", linewidth=0.8)


def nearest(field, x, y):
    """The value of a field (X, Y, F) at the grid point nearest to (x, y), None outside the field (beyond half a
    spacing of its edges) or where it has no value."""
    X, Y, F = field
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    xs, ys = (X[0], Y[:, 0]) if X.ndim == 2 else (X, Y)
    if not xs.size or not ys.size:
        return None
    i, j = int(np.nanargmin(np.abs(xs - x))), int(np.nanargmin(np.abs(ys - y)))

    def inside(values, k, v):
        step = np.nanmedian(np.abs(np.diff(values))) if values.size > 1 else np.inf
        return abs(values[k] - v) <= step / 2 + 1e-12
    if not (inside(xs, i, x) and inside(ys, j, y)):
        return None
    value = F[j, i]
    return float(value) if np.isfinite(value) else None


def number(value):
    """A number in the box: 3 significant digits."""
    return f"{value:.3g}"


def labelled(label, value, fallback):
    """A value named as an axis or a colour bar names its quantity: "$x$ (mm)" gives "$x$ = 48.0 mm", "$x / L$" gives
    "$x / L$ = 0.96", a formula "$Tu = 100 ...$" its symbol "$Tu$ = 12.3"; fallback without label; "—" for no
    value."""
    text = "—" if value is None else number(value)
    label = (label or "").strip()
    if not label:
        return f"{fallback} = {text}"
    found = re.fullmatch(r"(\$[^$]*\$)\s*(?:\((.*)\))?", label)
    if found is None:
        return f"{label} = {text}"
    symbol, unit = found.group(1), found.group(2)
    if " = " in symbol:  # A formula: its symbol only
        symbol = symbol.split(" = ", 1)[0] + "$"
    return f"{symbol} = {text}" + (f" {unit}" if unit and value is not None else "")


class MapCursor:
    """The cursor of the maps of a figure canvas (FigureCanvasTkAgg, with its toolbar). side(): the fraction of the
    width of the axes where the wiper of two maps compared is (None: no wiper)."""

    def __init__(self, canvas, toolbar, side=lambda: None):
        self.canvas, self.toolbar, self.side = canvas, toolbar, side
        self.maps, self.ax, self.artists, self.marks = [], None, [], {}
        self.background, self.point, self.frozen = None, None, False
        canvas.mpl_connect("draw_event", self.drawn)
        canvas.mpl_connect("motion_notify_event", self.moved)
        canvas.mpl_connect("button_press_event", self.pressed)
        canvas.mpl_connect("figure_leave_event", self.left)

    # ------------------------------------------------------------------ the maps of the figure

    def find_maps(self, fig):
        """The maps of a figure with their field and colour bar, those of a single axes (the first map's)."""
        maps = [a for ax in fig.axes for a in ax.collections + ax.images
                if getattr(a, "field", None) is not None and getattr(a, "colorbar", None) is not None]
        return [a for a in maps if a.axes is maps[0].axes] if maps else []

    def build(self, maps):
        """The artists of the cursor for these maps (animated: drawn by blitting only)."""
        for artist in self.artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
        self.maps, self.artists, self.marks = maps, [], {}
        self.ax = maps[0].axes if maps else None
        if self.ax is None:
            self.point, self.frozen = None, False
            return
        ax = self.ax
        self.vline = ax.axvline(np.nan, animated=True, **LINE)
        self.hline = ax.axhline(np.nan, animated=True, **LINE)
        self.box = ax.text(0.015, 0.985, "", transform=ax.transAxes, ha="left", va="top", fontsize="small",
                           bbox=BOX, animated=True, zorder=10, math_fontfamily="cm", linespacing=1.4)
        self.artists = [self.vline, self.hline, self.box]
        for m in maps:
            mark = m.colorbar.ax.axhline(np.nan, color="black", linestyle="--", linewidth=1.4, animated=True)
            self.marks[id(m)] = mark
            self.artists.append(mark)

    # ------------------------------------------------------------------ events

    def drawn(self, event):
        """The figure drawn: the cursor follows its maps (built again for other maps), the background kept for the
        blitting, the cursor drawn over it."""
        maps = self.find_maps(self.canvas.figure)
        if [id(m) for m in maps] != [id(m) for m in self.maps]:
            self.point = None if not maps else self.point
            self.frozen = self.frozen and bool(maps)
            self.build(maps)
        self.background = self.canvas.copy_from_bbox(self.canvas.figure.bbox) if self.maps else None
        self.show()

    def active(self):
        """The cursor follows the mouse: maps, and neither the zoom nor the pan of the toolbar."""
        return bool(self.maps) and not getattr(self.toolbar, "mode", "")

    def moved(self, event):
        if not self.active() or self.frozen:
            return
        self.point = (event.xdata, event.ydata) if event.inaxes is self.ax and event.xdata is not None else None
        self.show()

    def pressed(self, event):
        """A left click on the map: the cursor frozen there, or released (following the mouse again)."""
        if event.button != 1 or not self.active() or event.inaxes is not self.ax or event.xdata is None:
            return
        self.frozen = not self.frozen
        self.point = (event.xdata, event.ydata)
        self.show()

    def left(self, event):
        if not self.frozen and self.point is not None:
            self.point = None
            self.show()

    # ------------------------------------------------------------------ drawing

    def map_at(self, x):
        """The map whose value is shown at the abscissa x: the first one, or with a wiper the one on its side."""
        fraction = self.side() if len(self.maps) > 1 else None
        if fraction is None:
            return self.maps[0]
        x0, x1 = self.ax.get_xlim()
        at = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
        return self.maps[0] if at < fraction else self.maps[1]

    def values(self):
        """The cursor at the point: (x, y, the map shown, its value or None); None without a point."""
        if self.point is None or not self.maps:
            return None
        x, y = self.point
        m = self.map_at(x)
        return x, y, m, nearest(m.field, x, y)

    def show(self):
        """The cursor drawn over the background (blitting): its lines, the mark on the colour bar of the map whose
        value is shown, the box of its values; nothing without a point."""
        if self.background is None or not self.maps:
            return
        canvas = self.canvas
        canvas.restore_region(self.background)
        found = self.values()
        if found is not None:
            x, y, m, value = found
            self.vline.set_xdata([x, x])
            self.hline.set_ydata([y, y])
            # Named as the axes and the colour bar of the map name them
            self.box.set_text("\n".join((labelled(self.ax.get_xlabel(), x, "x"), labelled(self.ax.get_ylabel(), y, "y"),
                                         labelled(m.colorbar.ax.get_ylabel(), value, "val."))))
            for artist in (self.vline, self.hline, self.box):
                self.ax.draw_artist(artist)
            for mid, mark in self.marks.items():
                if mid == id(m) and value is not None:
                    mark.set_ydata([value, value])
                    mark.axes.draw_artist(mark)
        canvas.blit(canvas.figure.bbox)
