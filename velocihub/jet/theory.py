"""Jet theory: the potential core of a jet coming from the left of the field of view, an add-on of the
post-processing (PostSettings.jet_theory).

The core is the region where the mean axial velocity u (along +x) is near the reference velocity of the jet
(core_u_ref): either the isocontour at core_level % of it ("level"), or the one at the centreline velocity where the
peak skewness of the profiles ends ("skewness": peak_skewness, skewness_end). Its centreline, its end, the profiles
of u / u_ref at stations along it and the velocity on the centreline are drawn by the figures of JET_FIGURES (group
"Jet theory" of the menus), and its numbers added to metrics.json. The jet measurements (fields of view, frame rate,
axes centred on the jet) are in presets.py.

    post = vh.PostProcessing(fields, vh.PostSettings(jet_theory=True, core_mode="skewness"))
    pc = post.potential_core()
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..fields import UREF, quiet

# The figures of the jet theory: key, title; their group in the menus, symbols, files and sizes
JET_FIGURES = [("core_map", "Potential core: u / u_ref"),
               ("core_profiles", "Potential core: Axial Vel. Profiles"),
               ("core_centreline", "Potential core: Long. Vel. Profile")]
JET_GROUP = ("Jet theory", tuple(key for key, _ in JET_FIGURES))
JET_MAPS = ("core_map",)
JET_SYMBOLS = {"core_map": "R"}
JET_FILES = {"core_map": "PotentialCoreMap", "core_profiles": "PotentialCoreProfiles",
             "core_centreline": "PotentialCoreCentreline"}
JET_FIGSIZE = {"core_profiles": (15, 5), "core_centreline": (9, 4.5)}
# Detections of the potential core (JetSettings.core_mode): by a velocity level, by the skewness of the profiles
CORE_MODES = ("level", "skewness")
# The skewness detection: the profiles from the first column where the jet (smoothed mean u) reaches this fraction
# of u_ref
JET_START = 0.5


@dataclass
class JetSettings:
    """Settings of the jet theory, part of the recipe of the post-processing (PostSettings). jet_theory: the add-on
    is on (its figures drawn and exported, its numbers in metrics.json); core_mode, the detection of the potential
    core (CORE_MODES): "level", the isocontour of the mean velocity at core_level % of the reference velocity of the
    jet theory; "skewness", the isocontour at the centreline velocity where the peak skewness of skew_profiles
    profiles ends (skew_zero: % of the highest one, skew_change: % of the previous one; potential_core); n_stations,
    profiles of u / u_ref shown, the first one first_offset columns downstream of the exit (clear of the nozzle; the
    others spread up to the last one); core_u_ref, reference velocity of the jet theory without use_u_ref (m/s,
    physical space; None: the maximum of the mean u), else the one of the reference quantities."""

    jet_theory: bool = False
    core_mode: str = "level"
    core_level: float = 95.0
    skew_zero: float = 20.0
    skew_change: float = 10.0
    skew_profiles: int = 40
    n_stations: int = 6
    first_offset: int = 0
    core_u_ref: float = None

    def check_jet(self):
        """The settings of the jet theory checked and converted (called by PostSettings.__post_init__)."""
        self.jet_theory = bool(self.jet_theory)
        self.core_u_ref = float(self.core_u_ref) if self.core_u_ref else None
        if self.core_u_ref is not None and not self.core_u_ref > 0:
            raise ValueError("The reference velocity of the jet theory must be positive")
        self.core_level = float(self.core_level)
        self.first_offset = self.first_offset or 0
        for name in ("n_stations", "first_offset", "skew_profiles"):
            value = getattr(self, name)
            if value is None or float(value) != int(value):
                raise ValueError(f"{name} must be an integer")
            setattr(self, name, int(value))
        if not 0 < self.core_level <= 100:
            raise ValueError("Invalid post-processing settings: core_level (% of u_ref) in ]0, 100]")
        if self.core_mode not in CORE_MODES:
            raise ValueError(f"Invalid post-processing settings: core_mode in {', '.join(CORE_MODES)}")
        self.skew_zero, self.skew_change = float(self.skew_zero), float(self.skew_change)
        if not 0 < self.skew_zero <= 100:
            raise ValueError("Invalid post-processing settings: skew_zero (% of the highest skewness) in ]0, 100]")
        if not self.skew_change > 0:
            raise ValueError("Invalid post-processing settings: skew_change (% of the previous skewness) must be > 0")
        if self.skew_profiles < 2:
            raise ValueError("Invalid post-processing settings: skew_profiles >= 2")
        if self.first_offset < 0:
            raise ValueError("Invalid post-processing settings: first_offset (columns) must be >= 0")
        if self.n_stations < 1:
            raise ValueError("Invalid post-processing settings: n_stations >= 1")

    def jet_steps(self):
        """The jet theory in the steps of the recipe."""
        if not self.jet_theory:
            return []
        return [f"jet theory: potential core by {'skewness' if self.core_mode == 'skewness' else 'velocity level'}"]


################################################################################
# Potential core


def core_region(R, level):
    """Cells of the potential core in a map of R [row, column]: the largest connected region where R is at least
    level, its holes (invalid cells, dips below the level) filled. None if no cell reaches the level."""
    with quiet():
        inside = np.isfinite(R) & (R >= level)
    labels, n = ndimage.label(inside)
    if n == 0:
        return None
    return ndimage.binary_fill_holes(labels == 1 + int(np.argmax(np.bincount(labels.ravel())[1:])))


def peak_skewness(r, s, y):
    """Peak skewness of a velocity profile r(y): its third moment about its maximum (the jet centreline, located on
    the smoothed profile s, sub-cell: the parabola through the highest cell and its neighbours), not about its
    centroid as the usual skewness, over the widest band symmetric about the maximum within the valid cells, r
    clipped at 0 and its holes interpolated: gamma_p = sum r (y - y_p)^3 / sum r / m2^1.5, m2 = sum r (y - y_p)^2 /
    sum r. It is large while the profile is flat-topped (its maximum anywhere on the plateau: the potential core)
    and nears 0 once it is peaked at its centre (bell-shaped), where the usual skewness of a symmetric profile is 0
    in both cases: it is the criterion of the end of the core, not comparable with a skewness of the literature.
    Returns (gamma_p, y and value of the maximum); NaN if s has fewer than 3 valid cells."""
    rows = np.flatnonzero(np.isfinite(s))
    if rows.size < 3:
        return np.nan, np.nan, np.nan
    j = rows[np.argmax(s[rows])]
    y_peak, peak = float(y[j]), float(s[j])
    if 0 < j < len(s) - 1 and np.isfinite(s[j - 1]) and np.isfinite(s[j + 1]):
        curvature = s[j - 1] - 2 * s[j] + s[j + 1]
        if curvature < 0:
            p = 0.5 * (s[j - 1] - s[j + 1]) / curvature
            y_peak, peak = float(y[j] + p * (y[j + 1] - y[j])), float(s[j] - 0.25 * (s[j - 1] - s[j + 1]) * p)
    half = min(abs(y_peak - y[rows[0]]), abs(y[rows[-1]] - y_peak))
    d = y - y_peak
    band = (np.abs(d) <= half * (1 + 1e-9)) & (np.arange(len(y)) >= rows[0]) & (np.arange(len(y)) <= rows[-1])
    ok = np.isfinite(r)
    if band.sum() < 3 or not ok[band].any():
        return np.nan, y_peak, peak
    order = np.argsort(y[ok])
    w = np.clip(np.interp(y[band], y[ok][order], r[ok][order]), 0, None)
    if not w.sum() > 0:
        return np.nan, y_peak, peak
    m2 = (w * d[band] ** 2).sum() / w.sum()
    return float((w * d[band] ** 3).sum() / w.sum() / m2 ** 1.5) if m2 > 0 else np.nan, y_peak, peak


def skewness_end(R, S, y, columns, zero, change):
    """End of the potential core by the skewness of the profiles of R [x index, y index] at the columns
    (peak_skewness, the maxima located on the smoothed S): the first column after the first one where |skewness| is
    at most zero times the highest |skewness| of the columns, and its change from the previous column at most change
    times the |skewness| of the previous one. Returns the skewness and the maxima (y, value) of the columns, the index
    of that column (None: none) and the cutoff level, the maximum there."""
    out = np.array([peak_skewness(R[i], S[i], y) for i in columns], np.float64).reshape(-1, 3)
    g = out[:, 0]
    with quiet():
        top = np.nanmax(np.abs(g)) if np.isfinite(g).any() else np.nan
    end = None
    if np.isfinite(top) and top > 0:
        end = next((k for k in range(1, len(g))
                    if abs(g[k]) <= zero * top and abs(g[k] - g[k - 1]) <= change * abs(g[k - 1])), None)
    return dict(skewness=g, peaks=out[:, 1:], top=top, end=end, level=None if end is None else float(out[end, 2]))


def region_centreline(edges, x):
    """Straight line fitted to the middle of the edges of a region in each column (region_edges): (slope,
    intercept) of y(x), and the columns of the region."""
    edge_a, edge_b = edges
    columns = np.flatnonzero(np.isfinite(edge_a))
    middle = (edge_a[columns] + edge_b[columns]) / 2
    slope, intercept = np.polyfit(x[columns], middle, 1) if len(columns) > 1 else (0.0, float(middle[0]))
    return slope, intercept, columns


def region_contour(R, region, level, x, y):
    """Closed isocontour R = level around a region of a map [row, column] on the grid x (columns), y (rows): between
    the cells of the region and those below the level next to it (interpolated), and along the field edge where the
    region reaches it. Returns (x, y), the first point repeated at the end."""
    import contourpy

    with quiet():
        Z = np.where(region, np.where(np.isfinite(R) & (R >= level), R, level + 1e-9),
                     np.where(np.isfinite(R) & (R < level), R, level - 1.0))
    # A margin below the level around the map: the contour closed by the field edge
    Z = np.pad(Z, 1, constant_values=level - 1.0)
    dx, dy = (x[1] - x[0]) if len(x) > 1 else 1.0, (y[1] - y[0]) if len(y) > 1 else 1.0
    xp = np.concatenate([[x[0] - dx], x, [x[-1] + dx]]).astype(np.float64)
    yp = np.concatenate([[y[0] - dy], y, [y[-1] + dy]]).astype(np.float64)
    lines = contourpy.contour_generator(xp, yp, Z).lines(level)
    line = max(lines, key=len)
    cx = np.clip(line[:, 0], min(x[0], x[-1]), max(x[0], x[-1]))
    cy = np.clip(line[:, 1], min(y[0], y[-1]), max(y[0], y[-1]))
    if cx[0] != cx[-1] or cy[0] != cy[-1]:
        cx, cy = np.append(cx, cx[0]), np.append(cy, cy[0])
    return cx, cy


def region_edges(R, region, level, y):
    """Edges of a region of a map R [row, column] in each column (y, NaN where it has none): its first and last
    rows, moved to where R crosses the level towards the next row outside."""
    ny, nx = R.shape
    edges = np.full((2, nx), np.nan)
    for i in range(nx):
        rows = np.flatnonzero(region[:, i])
        if rows.size == 0:
            continue
        for k, (j, out) in enumerate(((rows[0], rows[0] - 1), (rows[-1], rows[-1] + 1))):
            edges[k, i] = y[j]
            if 0 <= out < ny and np.isfinite(R[out, i]) and np.isfinite(R[j, i]) and R[j, i] > R[out, i]:
                t = min(max((R[j, i] - level) / (R[j, i] - R[out, i]), 0.0), 1.0)
                edges[k, i] = y[j] + t * (y[out] - y[j])
    return edges


# Smoothing of the mean u read by the skewness detection (cells): the smoothing of the gradients before VelociHub 1.2
JET_SIGMA = 1.5


class JetAnalysis:
    """The jet theory of a post-processing (a mixin of postprocessing.PostProcessing, which gives the statistics, the
    smoothed mean velocity and the drawing of the maps): the potential core and its figures, when the settings have
    jet_theory on. The skewness detection reads the mean u smoothed over JET_SIGMA cells, whatever the smoothing of
    the gradients (PostSettings.gradient_sigma)."""

    def jet_off(self, key):
        """Why the figure key of the jet theory can't be drawn: the add-on off (None: it can)."""
        if key in JET_FILES and not self.settings.jet_theory:
            return "The jet theory is off (Recipe: Jet theory add-on)"
        return None

    def core_u_ref(self):
        """Reference velocity of the jet theory: the one of the reference quantities with use_u_ref, else the one
        entered for the jet (physical space), else the maximum of the mean u."""
        s = self.settings
        if s.core_u_ref and not s.use_u_ref and not self.sensor():
            return s.core_u_ref
        return abs(self.statistics()["u_ref"])

    def potential_core(self):
        """Potential core of the jet, with the velocities relative to the reference velocity of the jet theory:
        R(x, y) = U(x, y) / u_ref (core_u_ref), on the whole map. The core is the largest region where R is at least
        core_level % (core_region), its contour closed (by the field edge where the core goes beyond the field;
        region_contour); the core starts at its first column i0 (the nozzle exit). The core centreline is the
        straight line fitted to the middle of the edges of the core in each column (region_edges,
        region_centreline), R along it from i0, and the length of the core along it to its end (the downstream tip
        of its contour; None if the core goes beyond the field). The profiles are taken at n_stations columns, the
        first one first_offset columns after i0, the last one near the end of the field. ended_before: R reaches the
        level nowhere (the core ended before the field; the jet centreline instead, fitted to the centroids of the
        profiles of the mean velocity, from the first column with velocities); "error" if there is no mean velocity.
        The skewness detection (core_mode) sets the level instead: skew_profiles profiles (skew_columns) spread like
        the stations, from the first column where the jet reaches JET_START of u_ref; the level is the maximum of the
        profile (the centreline velocity) at the column where their skewness ends (skewness_end: skew_detect,
        end_column); core_level if it ends at none (fallback). skewness, peaks: those of the stations, shown."""
        s = self.settings

        def compute():
            MU = self.statistics()["mean_U"].T  # [x index, y index]
            x_mm, y_mm = self.fields.x[0, :], self.fields.y[:, 0]
            nx, ny = MU.shape
            if not np.isfinite(MU).any():
                return dict(error="no mean velocity in the field", x_mm=x_mm, y_mm=y_mm)
            u_ref, level = self.core_u_ref(), s.core_level / 100
            with quiet():
                R = MU / u_ref
            core = dict(error=None, R=R.T, u_ref=u_ref, x_mm=x_mm, y_mm=y_mm, contour=None, angle=0.0,
                        x_end_pc=None, length=None, ended_before=False, mode=s.core_mode, skewness=None, peaks=None,
                        skew_columns=None, skew_detect=None, skewness_top=None, end_column=None, fallback=False)
            stations = None
            if s.core_mode == "skewness":
                S = self.smoothed_mean_u(JET_SIGMA).T / u_ref
                with quiet():
                    jet = np.flatnonzero(np.nanmax(np.where(np.isfinite(S), S, -np.inf), axis=1) >= JET_START)
                if jet.size:
                    y = y_mm.astype(np.float64)
                    stations = self.station_columns(int(jet[0]), nx)
                    columns = np.unique(self.station_columns(int(jet[0]), nx, s.skew_profiles))
                    found = skewness_end(R, S, y, columns, s.skew_zero / 100, s.skew_change / 100)
                    shown = np.array([peak_skewness(R[i], S[i], y) for i in stations], np.float64).reshape(-1, 3)
                    core.update(skewness=shown[:, 0], peaks=shown[:, 1:], skew_columns=columns,
                                skew_detect=found["skewness"], skewness_top=found["top"],
                                end_column=None if found["end"] is None else int(columns[found["end"]]))
                    if found["end"] is not None:
                        level = found["level"]
                core["fallback"] = core["end_column"] is None
            core["level"] = level
            Z, zero = R.T, level
            region = core_region(Z, zero)
            if region is None:  # No velocity at the level: no core
                i0 = int(np.flatnonzero(np.isfinite(MU).any(axis=1))[0])
                # The jet centreline instead: the straight line fitted to the centroids of the profiles of the mean
                # velocity (weighted by u², the still air hardly counts), from the first column with velocities
                w = np.where(np.isfinite(MU), np.clip(MU, 0, None), 0.0) ** 2
                columns = np.flatnonzero(w.sum(axis=1) > 0)
                columns = columns[columns >= i0]
                centroids = (w[columns] * y_mm).sum(axis=1) / w[columns].sum(axis=1)
                slope, intercept = (np.polyfit(x_mm[columns], centroids, 1) if len(columns) > 1
                                    else (0.0, float(centroids[0])))
                y_pc = intercept + slope * x_mm
                edge_a = edge_b = np.full(nx, np.nan)
                core.update(ended_before=True, x_end_pc=float(x_mm[i0]), length=0.0,
                            angle=float(np.degrees(np.arctan(slope))))
            else:
                contour = region_contour(Z, region, zero, x_mm, y_mm)
                edge_a, edge_b = region_edges(Z, region, zero, y_mm)
                # The core centreline: the straight line fitted to the middle of its edges
                slope, intercept, columns = region_centreline((edge_a, edge_b), x_mm)
                i0 = int(columns[0])
                y_pc = intercept + slope * x_mm
                angle = float(np.degrees(np.arctan(slope)))
                # The end of the core: the downstream tip of its contour, none if it reaches the end of the field
                downstream = int(np.argmax(np.abs(x_mm - x_mm[0])))
                x_end = (None if region[:, downstream].any()
                         else float(contour[0][np.argmax(np.abs(contour[0] - x_mm[0]))]))
                length = None if x_end is None else abs(x_end - float(x_mm[i0])) / np.cos(np.radians(angle))
                core.update(contour=contour, angle=angle, x_end_pc=x_end, length=length)
            # R along the core centreline, from the start of the core
            order = np.argsort(y_mm)
            R_pc = np.full(nx, np.nan)
            for i in range(i0, nx):
                R_pc[i] = np.interp(y_pc[i], y_mm[order], R[i, order])
            if stations is None:
                stations = self.station_columns(i0, nx)
            core.update(i0=i0, x0=float(x_mm[i0]), stations=stations, edge_a=edge_a, edge_b=edge_b, y_pc=y_pc,
                        R_pc=R_pc)
            return core
        return self.windowed("potential_core", (s.outlier_k, s.use_u_ref, s.u_ref, s.core_u_ref, s.core_mode,
                                                s.core_level, s.skew_zero, s.skew_change, s.skew_profiles,
                                                s.n_stations, s.first_offset), compute)

    def station_columns(self, i0, nx, n=None):
        """Columns of n profiles of the potential core (None: n_stations): the first one first_offset columns
        downstream of the exit i0 (clear of the nozzle), the last one near the end of the field."""
        return np.linspace(min(i0 + self.settings.first_offset, nx - 3), nx - 3,
                           n or self.settings.n_stations).astype(int)

    def jet_metrics(self):
        """The numbers of the potential core in metrics.json (none when the jet theory is off)."""
        if not self.settings.jet_theory:
            return {}
        m = {}
        pc = self.potential_core()
        if pc["error"] is None:
            m.update(core_u_ref_ms=pc["u_ref"], core_centreline_y_mm=float(pc["y_pc"][pc["i0"]]),
                     core_centreline_angle_deg=pc["angle"], core_end_x_mm=pc["x_end_pc"], core_length_mm=pc["length"],
                     core_level_percent=100 * pc["level"], core_detection=pc["mode"])
            if pc["skew_detect"] is not None:
                i = pc["end_column"]
                m.update(core_skewness=[None if not np.isfinite(g) else float(g) for g in pc["skew_detect"]],
                         core_skewness_x_mm=[float(pc["x_mm"][j]) for j in pc["skew_columns"]],
                         core_skewness_end_x_mm=None if i is None else float(pc["x_mm"][i]))
            if pc["fallback"]:
                m["core_skewness_end"] = "no profile meets the skewness criteria: core_level used"
        else:
            m["potential_core"] = pc["error"]
        return m

    def core_or_message(self, fig, key):
        pc = self.potential_core()
        if pc["error"] is not None:
            self.message(fig, self.FIGURE_TITLES[key], pc["error"])
            return None
        return pc

    def draw_core_map(self, fig, prefix):
        pc = self.core_or_message(fig, "core_map")
        if pc is None:
            return
        X, Y = self.plot_x, self.plot_y
        # The whole map, the core drawn over it
        ax = self.map(fig, pc["R"], f"{prefix}Potential core velocity" + self.average_time(u_ref=pc["u_ref"]),
                      rf"$\overline{{u}}\ /\ {UREF}$", "RdBu_r", 0.5, 1)
        if pc["contour"] is not None:
            xs, ys = pc["contour"]
            by = (" by skewness" if pc["mode"] == "skewness" and not pc["fallback"] else
                  " (skewness: no end)" if pc["mode"] == "skewness" else "")
            ax.plot(X(xs), Y(ys), "k-", lw=1.8,
                    label=rf"Potential core{by} ($\overline{{u}}$ ≥ {pc['level'] * 100:.3g} % of ${UREF}$)")
        # The centreline along which the profiles are taken, to the end of the field (past the end of the core): the
        # core centreline from the start of the core, else (no core) the jet centreline (centroids of the profiles)
        # from the first column with velocities
        columns = np.arange(len(pc["x_mm"]))
        if pc["contour"] is not None:
            edges = np.flatnonzero(np.isfinite(pc["edge_a"]))
            inside = columns >= (edges[0] if edges.size else pc["i0"])
            label = f"Potential core centerline ({pc['angle']:.1f}°)"
        else:
            inside = columns >= pc["i0"]
            label = f"Jet centerline ({pc['angle']:.1f}°)"
            ax.text(0.99, 0.03, "The core ended before the field", transform=ax.transAxes, ha="right",
                    va="bottom", color="0.2", fontsize="small")
        ax.plot(X(pc["x_mm"][inside]), Y(pc["y_pc"][inside]), "--", color="orange", lw=1.5, label=label)
        # Where the velocity profiles are taken (core_profiles): on the centreline
        stations = pc["stations"]
        ax.plot(X(pc["x_mm"][stations]), Y(pc["y_pc"][stations]), "o", color="lime", mec="k", mew=1, ms=6, zorder=5,
                label="Profile locations")
        ax.legend(loc="upper right", fontsize="x-small")

    def draw_core_profiles(self, fig, prefix):
        pc = self.core_or_message(fig, "core_profiles")
        if pc is None:
            return
        Y = self.plot_y
        axs = np.atleast_1d(fig.subplots(1, len(pc["stations"]), sharey=True))
        for k, (a, i) in enumerate(zip(axs, pc["stations"])):
            a.plot(pc["R"][:, i], Y(pc["y_mm"]), "o-", ms=2, color="tab:blue")
            a.axvline(1, color="k", lw=0.8)
            a.axvline(pc["level"], color="r", ls=":", lw=1)
            # The edges of the core at this station (none past its end)
            for edge in (pc["edge_a"][i], pc["edge_b"][i]):
                if np.isfinite(edge):
                    a.axhline(Y(edge), color="k", ls=":", lw=1.2)
            title = self.position("x", pc["x_mm"][i], 0)
            if pc["skewness"] is not None:
                # The maximum of the profile (the jet centreline) about which the skewness is taken
                y_peak, peak = pc["peaks"][k]
                if np.isfinite(y_peak):
                    a.plot(peak, Y(y_peak), "D", ms=5, color="orange", mec="k", mew=0.6, zorder=5)
                g = pc["skewness"][k]
                title += "\n" + (rf"$\gamma_p$ = {g:+.2f}" if np.isfinite(g) else r"$\gamma_p$: none")
            a.set_title(title, fontsize="small")
            a.set_xlim(0, 1.2)
            a.set_xlabel(rf"$\overline{{u}}\ /\ {UREF}$")
            a.grid(True, alpha=0.3)
        axs[0].set_ylabel(self.position_label("y"))
        if self.image_height():  # The image: y downwards
            axs[0].invert_yaxis()
        # The lines named next to them: the criterion along its line at the top of the last profile, the edges of
        # the core on the first one
        a = axs[-1]
        a.annotate("End Condition", (pc["level"], 0.02), xycoords=a.get_xaxis_transform(),
                   xytext=(-2, 0), textcoords="offset points", rotation=90, ha="right", va="bottom", color="r",
                   fontsize="x-small")
        a, i = axs[0], pc["stations"][0]
        edges = [e for e in (pc["edge_a"][i], pc["edge_b"][i]) if np.isfinite(e)]
        if edges:
            a.annotate("Core Edges", (0.05, Y(max(edges))), xycoords=a.get_yaxis_transform(), xytext=(0, -4),
                       textcoords="offset points", ha="left", va="top", color="k", fontsize="x-small")
        if pc["skewness"] is not None:
            fig.supxlabel(self.skewness_text(pc), fontsize="x-small", color="0.3")
        fig.suptitle(f"{prefix}Potential core velocity profiles" + self.refs(self.window_text(), u_ref=pc["u_ref"]),
                     fontweight="bold",
                     fontsize="medium")

    def draw_core_centreline(self, fig, prefix):
        pc = self.core_or_message(fig, "core_centreline")
        if pc is None:
            return
        # From the first profile on (first_offset columns downstream of the exit: clear of the nozzle)
        i0 = pc["stations"][0]
        x = self.plot_x(pc["x_mm"])
        ax = fig.add_subplot()
        ax.plot(x[i0:], pc["R_pc"][i0:], "o-", ms=3, color="orange", label="Core centerline")
        ax.axhline(1, color="k", lw=0.8)
        ax.axhline(pc["level"], color="r", ls=":", label="End Condition")
        if pc["x_end_pc"] is not None and not pc["ended_before"]:
            ax.axvline(self.plot_x(pc["x_end_pc"]), color="g", ls="--",
                       label=f"End of core: {self.position('x', pc['x_end_pc'], 0)}")
        ax.set_xlabel(self.position_label("x"))
        ax.set_ylabel(rf"$\overline{{u}}\ /\ {UREF}$")
        ax.set_title(f"{prefix}Potential core velocity on centerline"
                     + self.refs(self.window_text(), u_ref=pc["u_ref"]),
                     fontweight="bold",
                     fontsize="medium")
        ending = ("The core doesn't end in the field" if pc["x_end_pc"] is None
                  else "The core ended before the field" if pc["ended_before"] else None)
        if ending:
            ax.text(0.99, 0.9, ending, transform=ax.transAxes, ha="right", va="top", color="0.35",
                    fontsize="small")
        handles, labels = ax.get_legend_handles_labels()
        if pc["skewness"] is not None:
            # The skewness of the profiles of the detection (right axis): near 0 within skew_zero % of the highest
            twin = ax.twinx()
            columns = pc["skew_columns"]
            twin.plot(x[columns], pc["skew_detect"], "s--", ms=3, color="tab:purple", lw=1, label=r"Peak skewness $\gamma_p$")
            band = self.settings.skew_zero / 100 * (pc["skewness_top"] or 0)
            twin.axhspan(-band, band, color="tab:purple", alpha=0.08, lw=0)
            twin.axhline(0, color="tab:purple", lw=0.6, alpha=0.5)
            if pc["end_column"] is not None:
                k = int(np.flatnonzero(columns == pc["end_column"])[0])
                twin.plot(x[pc["end_column"]], pc["skew_detect"][k], "o", ms=9, mfc="none", mec="g", mew=1.5)
            twin.set_ylabel(r"Peak skewness $\gamma_p$ of the profiles", color="tab:purple")
            twin.tick_params(axis="y", colors="tab:purple")
            more_handles, more_labels = twin.get_legend_handles_labels()
            handles, labels = handles + more_handles, labels + more_labels
            fig.supxlabel(self.skewness_text(pc), fontsize="x-small", color="0.3")
        ax.legend(handles, labels, fontsize="small")
        ax.grid(True, alpha=0.3)

    def skewness_text(self, pc):
        """The end of the potential core by the skewness of its profiles, as written under its figures."""
        s = self.settings
        criteria = (f"|γp| ≤ {s.skew_zero:g} % of the highest |γp| and change ≤ {s.skew_change:g} % of the "
                    f"previous |γp| ({len(pc['skew_columns'] if pc['skew_columns'] is not None else ())} profiles)")
        if pc["fallback"]:
            return f"Peak skewness: no profile meets {criteria}; the core at {s.core_level:g} % of ${UREF}$ instead"
        return (f"Peak skewness: {criteria} at {self.position('x', pc['x_mm'][pc['end_column']], 0)}; cutoff "
                f"velocity there: {100 * pc['level']:.3g} % of ${UREF}$")
