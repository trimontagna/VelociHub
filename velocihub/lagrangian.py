"""Post-processing of the PTV (TracTrac: piv/ptv_engine.py): the particles tracked (tracks.npz, beside vectors.npz),
the particles of each cell of the vector fields, the material acceleration of the particles binned on the grid, and
the comparison of the PTV with the PIV (PaIRS) of the same measurement. LagrangianAnalysis is a part of
PostProcessing (postprocessing.py), whose figures it adds (FIGURES, GROUP...).

The tracks are those of the last pass of TracTrac: for each particle tracked at a frame, its position (px from the
top-left corner of the images), its velocity (px per frame: the mean of its displacements from the frame before and
to the frame after) and its acceleration (px per frame²: their difference), with its identity. The frames are those
of the sequence of the pairs: the map k is the pair of the frames k and k + 1.
"""

import os

import numpy as np
from matplotlib.collections import LineCollection

from .fields import LREF, PIV_FOLDER, TFLOW, UREF, VECTORS_FILE, load_vectors, num2str, quiet, sig2

# Figures: key and name (in the menus), their group, the maps among them (with their symbols, LaTeX), those of one map,
# the elements of the names of their files and the sizes of those that aren't maps
FIGURES = [("ptv_tracks", "PTV tracks"),
           ("ptv_particles", "Particles per cell"),
           ("ptv_yield", "Particles tracked per frame"),
           ("ptv_track_length", "Track lengths"),
           ("mean_acceleration", "Mean acceleration |a|"),
           ("mean_acceleration_x", "Mean acceleration a_x"),
           ("mean_acceleration_y", "Mean acceleration a_y"),
           ("ptv_vs_piv", "PTV vs PIV profiles")]
GROUP = ("PTV", tuple(key for key, _ in FIGURES))
MAPS = {"ptv_particles": r"\overline{N_p}", "mean_acceleration": r"|\overline{\mathbf{a}}|",
        "mean_acceleration_x": r"\overline{a_x}", "mean_acceleration_y": r"\overline{a_y}"}
INSTANTANEOUS = {"ptv_tracks"}
FIGURE_FILES = {"ptv_tracks": "PTVTracks", "ptv_particles": "PTVParticles", "ptv_yield": "PTVYield",
                "ptv_track_length": "PTVTrackLengths", "mean_acceleration": "MeanAcceleration",
                "mean_acceleration_x": "MeanAccelerationX", "mean_acceleration_y": "MeanAccelerationY",
                "ptv_vs_piv": "PTVvsPIV"}
FIGSIZE = {"ptv_yield": (10, 4.5), "ptv_track_length": (8, 4.5), "ptv_vs_piv": (15, 7)}
# Figures of the tracks (and of the acceleration, from them), and those of the particles of the cells
TRACK_FIGURES = {"ptv_tracks", "ptv_yield", "ptv_track_length", "mean_acceleration", "mean_acceleration_x",
                 "mean_acceleration_y"}
# Frames before and after the pair drawn whose tracks are drawn
TRACK_FRAMES = 10
# Particles a cell needs for its mean acceleration (after the outliers removed)
MIN_ACCELERATION_SAMPLES = 10
NOT_PTV = "Only for the results of a PTV (TracTrac)"
NO_TRACKS = "The particles tracked are unknown: they are those of a PTV (TracTrac), in tracks.npz beside vectors.npz"


class LagrangianAnalysis:
    """Analyses and figures of the PTV (a mixin of PostProcessing: fields, settings, cached, window...)."""

    def ptv(self):
        """The vector fields are those of a PTV (the particles of their cells are known)."""
        return self.fields.particles is not None

    def ptv_missing(self, key):
        """Why a figure of the PTV can't be drawn (None if it can)."""
        if key in TRACK_FIGURES and not self.fields.tracks:
            return NO_TRACKS if self.ptv() else NOT_PTV
        if key in ("ptv_particles", "ptv_vs_piv") and not self.ptv():
            return NOT_PTV
        if key == "ptv_vs_piv" and self.piv_comparison() is None:
            return (f"No PIV of this measurement to compare with: {os.path.join(PIV_FOLDER, VECTORS_FILE)} in "
                    f"{self.fields.folder}")
        return None

    def tracks(self):
        """The particles tracked (arrays of tracks.npz), None if unknown."""
        path = self.fields.tracks
        if not path:
            return None

        def compute():
            with np.load(path) as z:
                return {k: z[k] for k in z.files if k != "tractrac"}
        return self.cached("ptv_tracks", path, compute)

    def track_camera(self):
        """The camera of the vector fields (piv.Camera), which converts the tracks as it converted them; None for
        vector fields in pixels."""
        f = self.fields
        camera = f.settings.get("camera")
        if f.settings.get("units") == "px" or not camera or not f.dt:
            return None
        from .piv import Camera

        return Camera.from_record(camera)

    def track_units(self):
        """Pixel pitches (mm per px, horizontal and vertical) and time between frames (s) of the tracks, as those of
        the vector fields; None for vector fields in pixels."""
        f = self.fields
        if self.track_camera() is None or not f.scale:
            return None
        return f.scale, f.scale_y or f.scale, f.dt

    def track_positions(self, x_px, y_px):
        """Positions of particles in the images (px from their top-left corner) as those of the vector fields (mm,
        from the origin of the axes of the camera, y upwards; px from the bottom-left corner of the images in the
        sensor space)."""
        width, height = self.fields.settings["image_px"]
        camera = self.track_camera()
        if camera is None:
            return np.asarray(x_px, np.float64), height - np.asarray(y_px, np.float64)
        return camera.positions(x_px, y_px, width, height)

    def track_velocity(self, u_px, v_px, power=1):
        """Velocities (power 1, px per frame) or accelerations (power 2, px per frame²) of particles, as those of the
        vector fields (m/s, m/s²; px per pair in pixels), y upwards."""
        units = self.track_units()
        if units is None:
            return u_px * 1.0, -v_px * 1.0
        sx, sy, dt = units
        return u_px * (sx / 1000.0 / dt ** power), -v_px * (sy / 1000.0 / dt ** power)

    def cells(self, x, y):
        """Cell of the grid of the vector fields (row, column; -1 outside) of positions x, y (those of the fields)."""
        f = self.fields
        x0, y0 = f.x[0, 0], f.y[0, 0]
        dx, dy = f.x[0, 1] - x0, f.y[1, 0] - y0
        col, row = np.round((x - x0) / dx).astype(int), np.round((y - y0) / dy).astype(int)
        inside = (col >= 0) & (col < f.shape[1]) & (row >= 0) & (row < f.shape[0])
        return np.where(inside, row, -1), np.where(inside, col, -1)

    def acceleration(self):
        """Mean material acceleration of the particles of each cell over the frames of the window (m/s², y upwards;
        px per pair² in pixels): the particles further than outlier_k standard deviations from the mean of their
        cell removed (as the vectors of the statistics), the cells with fewer than MIN_ACCELERATION_SAMPLES
        particles and those outside the flow NaN. Returns a dictionary: ax, ay, n (particles of each cell), se_x, se_y
        (standard errors of the means: std / sqrt(n)); None without tracks."""
        T = self.tracks()
        if T is None:
            return None
        k_out = self.settings.outlier_k

        def compute():
            f = self.fields
            a, b = self.window()
            # The frames of the maps a to b - 1 (the map k: the frames k and k + 1); the rows are in time order
            lo, hi = np.searchsorted(T["frame"], [a, b + 1])
            x, y = self.track_positions(T["x"][lo:hi], T["y"][lo:hi])
            ax, ay = self.track_velocity(T["ax"][lo:hi].astype(np.float64), T["ay"][lo:hi].astype(np.float64), 2)
            row, col = self.cells(x, y)
            ok = (row >= 0) & np.isfinite(ax) & np.isfinite(ay)
            cell = row[ok] * f.shape[1] + col[ok]
            ax, ay = ax[ok], ay[ok]
            size = f.shape[0] * f.shape[1]

            def stats(keep):
                n = np.bincount(cell[keep], minlength=size).astype(np.float64)
                with quiet():
                    mean = [np.bincount(cell[keep], q[keep], minlength=size) / n for q in (ax, ay)]
                    std = [np.sqrt(np.maximum(np.bincount(cell[keep], q[keep] ** 2, minlength=size) / n - m ** 2, 0))
                           for q, m in zip((ax, ay), mean)]
                return n, mean, std
            keep = np.ones(cell.size, bool)
            n, mean, std = stats(keep)
            if k_out:
                with quiet():
                    keep = ((np.abs(ax - mean[0][cell]) <= k_out * std[0][cell])
                            & (np.abs(ay - mean[1][cell]) <= k_out * std[1][cell]))
                n, mean, std = stats(keep)
            enough = (n >= MIN_ACCELERATION_SAMPLES).reshape(f.shape) & (f.status == 0)
            out = [np.where(enough, m.reshape(f.shape), np.nan) for m in mean]
            with quiet():
                se = [np.where(enough, (s / np.sqrt(n)).reshape(f.shape), np.nan) for s in std]
            return dict(ax=out[0], ay=out[1], n=n.reshape(f.shape), se_x=se[0], se_y=se[1])
        return self.windowed("acceleration", (self.fields.tracks, k_out), compute)

    def acceleration_factor(self, symbol):
        """Factor and label (LaTeX, of the quantity symbol) of the accelerations in the figures: a L_ref / u_ref² with
        the reference length and velocity (physical space), else in m/s² (px per pair² in the sensor space)."""
        L, u = self.settings.l_ref, self.scaling_velocity()
        if L and u:
            return L * 1e-3 / u ** 2, rf"${symbol}\,{LREF}\ /\ {UREF}^2$"
        return 1.0, rf"${symbol}$ ({'px/pair²' if self.sensor() else 'm/s²'})"

    def piv_comparison(self):
        """The PIV of the same measurement (subfolder PIV of its folder), as a post-processing with the same recipe,
        None if there is none (or if the vector fields are themselves those of that folder)."""
        f = self.fields
        path = os.path.join(f.folder, PIV_FOLDER, VECTORS_FILE)
        if not self.ptv() or not os.path.isfile(path) or os.path.normcase(os.path.abspath(path)) == os.path.normcase(
                f.source):
            return None

        def compute():
            camera = f.settings.get("camera")
            if f.settings.get("units") == "px" or not camera:
                other = load_vectors(path, pixels=True)
            else:
                from .piv import Camera

                other = load_vectors(path, camera=Camera.from_record(camera))
            return type(self)(other, self.settings)
        post = self.cached("ptv_piv", (path, os.path.getmtime(path)), compute)
        post.settings = self.settings
        return post

    def ptv_metrics(self):
        """Key numbers of the PTV: particles detected and tracked per frame (medians), the yield, the lengths of
        the tracks (frames) and the particles per cell; {} without PTV."""
        if not self.ptv():
            return {}
        f = self.fields
        flow = f.status == 0
        m = dict(ptv_particles_per_cell=float(f.particles[:, flow].mean()) if flow.any() else None)
        T = self.tracks()
        if T is not None:
            frames = T["detected"] >= 0
            detected, tracked = T["detected"][frames], T["tracked"][frames]
            _, lengths = np.unique(T["id"], return_counts=True)
            m.update(ptv_detected_per_frame=float(np.median(detected)), ptv_tracked_per_frame=float(np.median(tracked)),
                     ptv_yield_percent=float(100 * tracked.sum() / max(detected.sum(), 1)), ptv_tracks=int(lengths.size),
                     ptv_track_length_median=float(np.median(lengths)) if lengths.size else None,
                     ptv_track_length_mean=float(np.mean(lengths)) if lengths.size else None)
        return m

    ############################################################################
    # Figures

    def ptv_message(self, fig, key):
        """The message of a figure of the PTV that can't be drawn: True if there is one."""
        missing = self.ptv_missing(key)
        if missing:
            self.message(fig, self.FIGURE_TITLES[key], missing)
        return bool(missing)

    def draw_ptv_tracks(self, fig, prefix):
        if self.ptv_message(fig, "ptv_tracks"):
            return
        T, f = self.tracks(), self.fields
        k = int(np.clip(self.map_index, 0, f.n - 1))
        lo, hi = np.searchsorted(T["frame"], [k - TRACK_FRAMES, k + 1 + TRACK_FRAMES + 1])
        ids, frames = T["id"][lo:hi], T["frame"][lo:hi]
        order = np.lexsort((frames, ids))
        ids, frames = ids[order], frames[order]
        x, y = self.track_positions(T["x"][lo:hi][order], T["y"][lo:hi][order])
        U, V = self.track_velocity(T["u"][lo:hi][order].astype(np.float64), T["v"][lo:hi][order].astype(np.float64))
        speed, label = self.velocity(np.hypot(U, V), r"|\mathbf{u}_p|")
        X, Y = self.plot_x(x), self.plot_y(y)
        # Segments between the successive frames of each particle
        join = (ids[1:] == ids[:-1]) & (frames[1:] == frames[:-1] + 1)
        segments = np.stack([np.stack([X[:-1], Y[:-1]], 1), np.stack([X[1:], Y[1:]], 1)], 1)[join]
        colours = ((speed[:-1] + speed[1:]) / 2)[join]
        ax = fig.add_subplot()
        ax.set_facecolor("#1d2330")
        # The same colours for all the maps (those of the instantaneous velocity)
        vmin, vmax = self.instantaneous_limits("ptv_tracks")
        lines = LineCollection(segments, cmap="viridis", linewidths=0.6)
        lines.set_array(colours)
        lines.set_clim(vmin, vmax)
        ax.add_collection(lines)
        # The particles of the pair drawn
        now = (frames == k) | (frames == k + 1)
        ax.plot(X[now], Y[now], ".", ms=0.8, color="white", alpha=0.5)
        MX, MY, image = self.map_positions()
        ax.set_aspect("equal")
        ax.set_xlim(np.nanmin(MX), np.nanmax(MX))
        ax.set_ylim(*((np.nanmax(MY), np.nanmin(MY)) if image else (np.nanmin(MY), np.nanmax(MY))))
        camera = f.settings.get("camera", {})
        ax.set_xlabel(self.position_label("x global" if camera.get("x_offset_mm") else "x"))
        ax.set_ylabel(self.position_label("y global" if camera.get("y_offset_mm") else "y"))
        n_tracks = int(np.unique(ids[:-1][join]).size)
        # The duration of the tracks drawn: from their first frame to their last (frames in the sensor space)
        first, last = max(k - TRACK_FRAMES, 0), min(k + 1 + TRACK_FRAMES, f.n)
        if self.sensor():
            span = f"frames {first + 1} to {last + 1}"
        elif self.uses_flow_time():
            span = rf"$\Delta t$ = {sig2(self.flow_times(last - first))} ${TFLOW}$"
        else:
            span = rf"$\Delta t$ = {(last - first) * self.map_interval():.4g} s"
        ax.set_title(f"{prefix}PTV tracks of {n_tracks} particles over {span}" + self.map_time(k, scaled=True),
                     fontweight="bold", fontsize="medium")
        fig.colorbar(lines, ax=ax, shrink=0.85, label=label)

    def draw_ptv_particles(self, fig, prefix):
        if self.ptv_message(fig, "ptv_particles"):
            return
        f = self.fields
        a, b = self.window()
        F = f.particles[a:b].mean(0, dtype=np.float64)
        F[f.status != 0] = np.nan
        self.stat_map(fig, prefix, F, r"Particles per cell $\overline{N_p}$",
                      rf"$\overline{{N_p}}$ (particles per pair, cells of {num2str(self.cell_size())} px)", "viridis",
                      low=0)

    def cell_size(self):
        """Size of the cells of the PTV (px)."""
        return (self.fields.settings.get("engine_settings") or {}).get("spacing") or (
            self.fields.settings.get("spacing_px") or [0])[0]

    def draw_ptv_yield(self, fig, prefix):
        if self.ptv_message(fig, "ptv_yield"):
            return
        T, f = self.tracks(), self.fields
        frames = np.flatnonzero(T["detected"] >= 0)
        detected, tracked = T["detected"][frames], T["tracked"][frames]
        # The time of the frame k: that of the map k (its first frame)
        t = np.interp(frames, np.arange(f.n), np.asarray(f.t, np.float64))
        if self.sensor():
            t, xlabel = frames + 1.0, "Frame"
        elif self.uses_flow_time():
            t, xlabel = t / self.flow_time(), rf"$t\ /\ {TFLOW}$"
        else:
            xlabel = "$t$ (s)"
        ax = fig.add_subplot()
        ax.plot(t, detected, color="#2a78d6", lw=1, label=f"Detected (median {np.median(detected):.0f})")
        ax.plot(t, tracked, color="#eb6834", lw=1, label=f"Tracked (median {np.median(tracked):.0f})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Particles per frame")
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        right = ax.twinx()
        with quiet():
            right.plot(t, 100 * tracked / np.maximum(detected, 1), color="0.4", lw=0.8, ls=":", label="Yield")
        right.set_ylabel("Yield (% tracked)")
        right.set_ylim(0, 100)
        lines = ax.get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0]
        ax.legend(lines, [h.get_label() for h in lines], frameon=False, loc="lower right", fontsize="small")
        total = 100 * tracked.sum() / max(detected.sum(), 1)
        ax.set_title(f"{prefix}Particles detected and tracked: {total:.1f} % tracked"
                     + self.refs(self.t_flow_text() if self.uses_flow_time() else "", positions=False, scaled=False),
                     fontweight="bold", fontsize="medium")

    def draw_ptv_track_length(self, fig, prefix):
        if self.ptv_message(fig, "ptv_track_length"):
            return
        T = self.tracks()
        _, lengths = np.unique(T["id"], return_counts=True)
        ax = fig.add_subplot()
        top = int(max(np.percentile(lengths, 99.5), 2)) if lengths.size else 2
        ax.hist(lengths, bins=np.arange(0.5, top + 1.5), color="#2a78d6", edgecolor="white", linewidth=0.3)
        ax.set_yscale("log")
        median = float(np.median(lengths)) if lengths.size else float("nan")
        ax.axvline(median, color="#eb6834", lw=1.5, label=f"Median: {median:.0f} frames")
        ax.set_xlabel("Track length (frames where the particle is tracked)")
        ax.set_ylabel("Number of tracks")
        ax.legend(frameon=False)
        ax.set_title(f"{prefix}Lengths of the {lengths.size} tracks: mean {lengths.mean():.1f} frames"
                     if lengths.size else f"{prefix}No track", fontweight="bold", fontsize="medium")

    def draw_acceleration(self, fig, prefix, component=None):
        key = "mean_acceleration" + (f"_{component}" if component else "")
        if self.ptv_message(fig, key):
            return
        acc = self.acceleration()
        ax_, ay_ = acc["ax"], acc["ay"]
        symbol = MAPS[key]
        factor, label = self.acceleration_factor(symbol)
        title = (rf"Mean acceleration ${symbol}$ of the particles (cells with at least {MIN_ACCELERATION_SAMPLES} "
                 "particles)")
        if component is None:
            self.stat_map(fig, prefix, np.hypot(ax_, ay_) * factor, title, label, "magma", low=0, quiver=(ax_, ay_),
                          key=key)
        else:
            self.stat_map(fig, prefix, (ax_ if component == "x" else ay_) * factor, title, label, "RdBu_r", True,
                          key=key)
        # The noise of the mean: where the mean is below a few standard errors, it is noise
        se = np.hypot(acc["se_x"], acc["se_y"]) if component is None else acc[f"se_{component}"]
        with quiet():
            median = float(np.nanmedian(se * factor))
        unit = "" if factor != 1.0 else (" px/pair²" if self.sensor() else " m/s²")
        fig.supxlabel(f"Standard error of the mean (σ / √N): median {median:.3g}{unit} — a mean below 2 to 3 times "
                      "it is noise", fontsize="x-small", color="0.3")

    def draw_mean_acceleration(self, fig, prefix):
        self.draw_acceleration(fig, prefix)

    def draw_mean_acceleration_x(self, fig, prefix):
        self.draw_acceleration(fig, prefix, "x")

    def draw_mean_acceleration_y(self, fig, prefix):
        self.draw_acceleration(fig, prefix, "y")

    def draw_ptv_vs_piv(self, fig, prefix):
        if self.ptv_message(fig, "ptv_vs_piv"):
            return
        other = self.piv_comparison()
        f, g = self.fields, other.fields
        with self.all_maps(), other.all_maps():
            st, so = self.statistics(), other.statistics()
        n = max(self.settings.n_stations, 1)
        columns = np.linspace(2, f.shape[1] - 3, n).astype(int)
        axs = fig.subplots(2, n, sharey=True, squeeze=False)
        names = (f"PTV ({f.settings.get('engine', 'TracTrac')})", f"PIV ({g.settings.get('engine', 'PIV')})")
        Y = self.plot_y
        for j, i in enumerate(columns):
            x = f.x[0, i]
            i2 = int(np.argmin(np.abs(g.x[0] - x)))
            for r, (key, symbol) in enumerate((("mean_U", r"\overline{u}"), ("u_rms", r"\sigma_u"))):
                a = axs[r, j]
                for s, fields, col, style in ((st, f, i, "-"), (so, g, i2, "--")):
                    F, label = self.velocity(s[key][:, col], symbol)
                    a.plot(F, Y(fields.y[:, col]), style, lw=1.2, color="#2a78d6" if style == "-" else "#eb6834")
                a.set_xlabel(label, fontsize="small")
                a.grid(True, alpha=0.3)
                if r == 0:
                    a.set_title(self.position("x", x, 0), fontsize="small")
        for a in axs[:, 0]:
            a.set_ylabel(self.position_label("y"))
        if self.image_height():
            axs[0, 0].invert_yaxis()
        axs[0, -1].legend(axs[0, -1].get_lines()[:2], names, frameon=False, fontsize="x-small")
        # Over the points of the flow of both (same grid)
        if f.shape == g.shape:
            both = st["reliable"] & so["reliable"] & np.isfinite(st["mean_U"]) & np.isfinite(so["mean_U"])
            u = abs(st["u_ref"])
            d_mean = 100 * np.median(np.abs(st["mean_U"][both] - so["mean_U"][both])) / u if both.any() else np.nan
            ratio = np.median(st["u_rms"][both] / so["u_rms"][both]) if both.any() else np.nan
            fig.supxlabel(f"Over the {int(both.sum())} grid points of both: median |PTV − PIV| of the mean u "
                          f"{d_mean:.1f} % of the reference velocity; median ratio PTV / PIV of σ_u {ratio:.2f}",
                          fontsize="x-small", color="0.3")
        fig.suptitle(f"{prefix}PTV and PIV of the measurement: mean velocity and fluctuations"
                     + self.average_time(scaled=True), fontweight="bold", fontsize="medium")
