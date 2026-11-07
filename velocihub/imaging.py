"""Post-processing of the images of the PIV/PTV: the raw images (those of the measurement, before any
pre-processing), the images of the PIV (those it computed the vector fields from: pre-processed, or the raw ones),
and the modal decomposition of the images (the POD of the page Pre-processing, pod.npz): its spatial modes and its
spectrum. ImageAnalysis is a part of PostProcessing (postprocessing.py), whose figures it adds (FIGURES, GROUPS...).

The image of the map k is the first image of its pair (the pairs of the settings of the vector fields index the
images of the PIV, in time order). The images are drawn at the positions of the vector fields (mm, or over L_ref;
pixels in the sensor space), so that they can be compared with the maps (wiper) and stitched as them. The images of
the PIV are found from the settings of the vector fields (image_folder, first_image, last_image, n_images), the raw
images from the description of the images pre-processed (preprocessing.json: source_folder, source_files).

The POD is that of the training images of the measurement (its project: preprocessing.pod.file, else pod.npz in the
subfolder PreProcessing): it is shown only when it has as many images as the PIV and their size. Its modes are those
evaluated: the modes exported are those of a range (PostSettings.pod_mode to pod_mode_last, from 1; one PNG each), the
mode shown one of them (pod_shown, the first by default); or summed (pod_combine): for the image of each map, the sum
of the modes weighted by its temporal coefficients, the background the pre-processing subtracts from it with these
modes (one per map, as the raw image). The modes are those of the correlation matrix of the images
(not of their fluctuations): the first one is close to the mean image.
"""

import json
import os
import zipfile
from contextlib import contextmanager

import numpy as np

from .fields import TFLOW, sig2
from .preprocessing import POD_FILE, PREPROCESSING_FOLDER, folder_images, imread, mode_limits
from .project import absolute, read_project

# Figures: key and name (in the menus), their groups, the maps among them (with their symbols, LaTeX), those of one map,
# those that are the same for all the maps (also with rolling statistics), the elements of the names of their files and
# the sizes of those that aren't maps
FIGURES = [("raw_image", "Raw image"),
           ("piv_image", "Image of the PIV (pre-processed)"),
           ("pod_mode", "POD mode φ_n"),
           ("pod_spectrum", "POD spectrum and temporal coefficient")]
GROUPS = [("Images", ("raw_image", "piv_image")),
          ("Modal decomposition (POD)", ("pod_mode", "pod_spectrum"))]
MAPS = {"raw_image": "I", "piv_image": r"I_\mathrm{PIV}", "pod_mode": r"\phi_n"}
INSTANTANEOUS = {"raw_image", "piv_image"}
STATIC = {"pod_mode", "pod_spectrum"}
FIGURE_FILES = {"raw_image": "RawImage", "piv_image": "PIVImage", "pod_mode": "PODMode",
                "pod_spectrum": "PODSpectrum"}
FIGSIZE = {"pod_spectrum": (14, 4.5)}
IMAGE_FIGURES, POD_FIGURES = set(GROUPS[0][1]), set(GROUPS[1][1])
# Images read for the colour range of the images (the same for all the maps), evenly spaced, and the pixels kept of
# each (one of IMAGE_STEP in each direction)
IMAGE_SCALE_FRAMES = 8
IMAGE_STEP = 4
# Percentiles of the colour range of the images: a few hot pixels don't hide the particles
IMAGE_PERCENTILES = (0.5, 99.5)
# Images kept in memory (the last ones read)
IMAGE_CACHE = 6


def npz_row(path, name, row):
    """Row of a 2D array of an .npz file (name: the array), read alone when the file isn't compressed (the modes of
    pod.npz: a mode without the others)."""
    with zipfile.ZipFile(path) as z:
        member = name + ".npy"
        if z.getinfo(member).compress_type == zipfile.ZIP_STORED:
            with z.open(member) as f:
                version = np.lib.format.read_magic(f)
                read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                        else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = read(f)
                if len(shape) == 2 and not fortran and not dtype.hasobject:
                    size = shape[1] * dtype.itemsize
                    f.seek(f.tell() + row * size)
                    return np.frombuffer(f.read(size), dtype).copy()
    with np.load(path) as z:
        return np.array(z[name][row])


class ImageAnalysis:
    """Figures of the images of the PIV/PTV and of their POD (a mixin of PostProcessing: fields, settings, cached,
    map_index...)."""

    ############################################################################
    # Images

    def image_files(self):
        """The images of the PIV in time order (those its pairs index) and the raw image of each: (piv, raw), lists
        of paths (raw: the reason why they are unknown instead, str); or the reason why the images of the PIV are
        unknown (str)."""
        f = self.fields

        def compute():
            s = f.settings
            if not s.get("image_px") or not s.get("pairs"):
                return "The images of the PIV/PTV are unknown (vector fields without their settings)"
            folder = absolute(s.get("image_folder"), f.folder)
            if not folder or not os.path.isdir(folder):
                return f"The images of the PIV/PTV are not found: {folder or 'no image folder in the settings'}"
            paths, manifest = folder_images(folder)
            names = [os.path.basename(p) for p in paths]
            first, last = s.get("first_image"), s.get("last_image")
            if first in names and last in names:
                paths = paths[names.index(first):names.index(last) + 1]
            n = s.get("n_images")
            if n and len(paths) != n:
                return f"The images of the PIV/PTV are not those of {folder}: {len(paths)} images there, {n} for the PIV"
            raw = paths
            if manifest is not None:
                # The images pre-processed: their sources, the raw images
                project = absolute(manifest.get("measurement_folder"), folder) or folder
                source = absolute(manifest.get("source_folder"), project)
                files, exported = manifest.get("source_files") or [], manifest.get("files") or []
                index = {name: i for i, name in enumerate(exported)}
                if not source or len(files) != len(exported) or any(os.path.basename(p) not in index for p in paths):
                    raw = f"The raw images of the images pre-processed of {folder} are unknown (preprocessing.json)"
                else:
                    raw = [os.path.join(source, files[index[os.path.basename(p)]]) for p in paths]
            return paths, raw
        return self.cached("image_files", (f.source, f.settings.get("image_folder")), compute)

    def image_missing(self, key):
        """Why the images of a figure can't be drawn (None if they can): the images of the PIV, or the raw ones
        (raw_image), unknown or not found (the first and the last ones looked for: the raw images may have been
        removed, those pre-processed kept)."""
        files = self.image_files()
        if isinstance(files, str):
            return files
        paths = files[1] if key == "raw_image" else files[0]
        if isinstance(paths, str):
            return paths
        if max(a for a, _ in self.fields.settings["pairs"]) >= len(paths):
            return "The pairs of the vector fields are not those of the images found"
        what = "raw images" if key == "raw_image" else "images of the PIV/PTV"
        for path in (paths[0], paths[-1]):
            if not os.path.isfile(path):
                return f"The {what} are not found: {path}"
        return None

    def image_path(self, key, k):
        """File of the image of the map k: the first image of its pair (raw_image: the raw one)."""
        piv_images, raw = self.image_files()
        first = self.fields.settings["pairs"][k][0]
        return (raw if key == "raw_image" else piv_images)[first]

    def read_image(self, path):
        """An image as drawn (2D, its intensities; the mean of the channels of a colour image), the last ones kept."""
        cache = self.__dict__.setdefault("image_cache", {})
        if path not in cache:
            A = imread(path)
            if A.ndim == 3:
                A = A[..., :3].mean(-1)
            if len(cache) >= IMAGE_CACHE:
                cache.pop(next(iter(cache)))
            cache[path] = A
        return cache[path]

    def image_size(self):
        """Width and height of the images of the PIV (px)."""
        return tuple(self.fields.settings["image_px"])

    def image_positions(self, width, height):
        """Positions of the pixels of an image as those of the maps (plot_x, plot_y of the vector fields): the
        extent of the image (left, right, bottom, top: its edges) and the centres of its columns and rows."""
        x, y = self.track_positions(np.array([0.0, width]), np.array([0.0, height]))
        X, Y = self.plot_x(x), self.plot_y(y)
        cx, _ = self.track_positions(np.arange(width) + 0.5, np.zeros(width))
        _, cy = self.track_positions(np.zeros(height), np.arange(height) + 0.5)
        return (float(X[0]), float(X[1]), float(Y[1]), float(Y[0])), self.plot_x(cx), self.plot_y(cy)

    def image_map(self, fig, A, title, label, cmap="gray", vmin=None, vmax=None):
        """Map of an image (as map: in the axes overlay if it is set) at the positions of the vector fields, with a
        colour bar. Returns the axes."""
        f = self.fields
        extent, cx, cy = self.image_positions(A.shape[1], A.shape[0])
        if self.overlay is not None:
            ax = self.overlay
        else:
            ax = fig.add_subplot()
            ax.set_facecolor("#d5d9e0")
        im = ax.imshow(A, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation="antialiased")
        im.field = (cx, cy, np.asarray(A, np.float64))  # As drawn: its isocontours (contour_maps), the stitching
        if self.overlay is None:
            ax.set_aspect("equal")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            camera = f.settings.get("camera", {})
            ax.set_xlabel(self.position_label("x global" if camera.get("x_offset_mm") else "x"))
            ax.set_ylabel(self.position_label("y global" if camera.get("y_offset_mm") else "y"))
        ax.set_title(title, fontweight="bold", fontsize="medium")
        fig.colorbar(im, ax=ax, shrink=0.85, label=label)
        return ax

    def image_limits(self, key):
        """Colour range of the images of a figure, the same for all the maps: the percentiles IMAGE_PERCENTILES of
        IMAGE_SCALE_FRAMES images evenly spaced."""
        def compute():
            n = self.fields.n
            maps = np.unique(np.linspace(0, n - 1, min(IMAGE_SCALE_FRAMES, n)).round().astype(int))
            values = np.concatenate([self.read_image(self.image_path(key, k))[::IMAGE_STEP, ::IMAGE_STEP].ravel()
                                     for k in maps]).astype(np.float64)
            values = values[np.isfinite(values)]
            if not values.size:
                return None, None
            low, high = (float(v) for v in np.percentile(values, IMAGE_PERCENTILES))
            return low, high if high > low else low + 1
        return self.cached(f"limits_{key}", self.image_files(), compute)

    def draw_image(self, fig, prefix, key):
        missing = self.image_missing(key)
        if missing:
            return self.message(fig, self.FIGURE_TITLES[key], missing)
        k = int(np.clip(self.map_index, 0, self.fields.n - 1))
        path = self.image_path(key, k)
        if not os.path.isfile(path):
            return self.message(fig, self.FIGURE_TITLES[key], f"The image is not found: {path}")
        A = self.read_image(path)
        if (A.shape[1], A.shape[0]) != self.image_size():
            return self.message(fig, self.FIGURE_TITLES[key], f"The image {os.path.basename(path)} is "
                                f"{A.shape[1]} x {A.shape[0]} px, those of the PIV/PTV "
                                f"{' x '.join(str(v) for v in self.image_size())} px")
        if key == "raw_image":
            what = "Raw image"
        else:
            steps = self.fields.settings.get("preprocessing")
            what = "Image of the PIV" + (f" ({steps})" if steps and "none" not in steps else
                                         " (not pre-processed)" if steps else "")
        # The file with the time of the map (each field of view its own in a stitched map)
        when = self.map_time(k, scaled=False).replace("\n(", f"\n({os.path.basename(path)}, ", 1)
        self.image_map(fig, A, f"{prefix}{what}, first image of the pair" + when, "Intensity (counts)", "gray",
                       *self.image_limits(key))

    def draw_raw_image(self, fig, prefix):
        self.draw_image(fig, prefix, "raw_image")

    def draw_piv_image(self, fig, prefix):
        self.draw_image(fig, prefix, "piv_image")

    ############################################################################
    # POD of the images

    def pod_file(self):
        """The POD of the training images of the measurement: the one of its project, else pod.npz in its subfolder
        PreProcessing; None if there is none."""
        folder = self.fields.folder
        try:
            data = read_project(folder) if folder and os.path.isdir(folder) else None
        except (OSError, ValueError):
            data = None
        saved = ((data or {}).get("preprocessing") or {}).get("pod") or {}
        for path in (absolute(saved.get("file"), folder), os.path.join(folder, PREPROCESSING_FOLDER, POD_FILE)):
            if path and os.path.isfile(path):
                return path
        return None

    def pod(self):
        """The POD of the images (pod_file): its file, spectrum (eigVal: the energy of each mode), shape of the
        images (Dim: rows, columns), the number of modes evaluated (modes) and their temporal coefficients (TCoeff:
        the projection of each training image on each mode); or the reason why there is none (str)."""
        path = self.pod_file()
        if path is None:
            return ("No POD of the images: evaluate the modes in the page Pre-processing (POD modes; saved as "
                    f"{os.path.join(PREPROCESSING_FOLDER, POD_FILE)})")

        def compute():
            try:
                with np.load(path) as z:
                    eigVal, Dim, TCoeff = (np.array(z[k]) for k in ("eigVal", "Dim", "TCoeff"))
                    with zipfile.ZipFile(path) as zf, zf.open("PHI.npy") as f:
                        version = np.lib.format.read_magic(f)
                        read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                else np.lib.format.read_array_header_2_0)
                        modes = int(read(f)[0][0])
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as e:
                return f"The POD could not be read ({path}): {e}"
            Dim = tuple(int(d) for d in Dim)
            width, height = self.image_size() if self.fields.settings.get("image_px") else (None, None)
            n = self.fields.settings.get("n_images")
            if width is not None and Dim != (height, width):
                return (f"The POD ({path}) is that of images of {Dim[1]} x {Dim[0]} px, not of the images of the PIV/PTV "
                        f"({width} x {height} px)")
            if n and len(eigVal) != n:
                return f"The POD ({path}) is that of {len(eigVal)} images, not of the {n} images of the PIV/PTV"
            return dict(file=path, eigVal=np.asarray(eigVal, np.float64), Dim=Dim, modes=modes,
                        TCoeff=np.asarray(TCoeff, np.float64).reshape(modes, -1))
        return self.cached("pod", (path, os.path.getmtime(path), self.fields.source), compute)

    def pod_range(self):
        """The modes of the POD exported: the first and the last ones (PostSettings.pod_mode, pod_mode_last; the
        first alone if pod_mode_last is None)."""
        s = self.settings
        last = getattr(s, "pod_mode_last", None)
        return s.pod_mode, max(s.pod_mode, last) if last is not None else s.pod_mode

    def pod_shown_mode(self):
        """The mode drawn: pod_shown (set by the interface: the handle of the range moved last, or by the export of
        each mode) if it is in the range, else the first one."""
        first, last = self.pod_range()
        shown = getattr(self, "pod_shown", None)
        return shown if shown is not None and first <= shown <= last else first

    def pod_modes_exported(self):
        """The modes of the range that are evaluated (those exported), [] without POD."""
        pod = self.pod()
        if isinstance(pod, str):
            return []
        first, last = self.pod_range()
        return list(range(first, min(last, pod["modes"]) + 1))

    def pod_combined(self):
        """The modes of the range summed (PostSettings.pod_combine, modes evaluated in the range; not inside
        single_mode): the background of the image of each map (pod_background), a figure of each map."""
        return (bool(getattr(self.settings, "pod_combine", False)) and not getattr(self, "pod_single", False)
                and bool(self.pod_modes_exported()))

    @contextmanager
    def single_mode(self, n):
        """The mode n drawn alone, whatever the combination (the stitching: each mode stitched)."""
        shown, single = getattr(self, "pod_shown", None), getattr(self, "pod_single", False)
        self.pod_shown, self.pod_single = n, True
        try:
            yield
        finally:
            self.pod_shown, self.pod_single = shown, single

    def pod_missing(self, key, n=None):
        """Why the figures of the POD can't be drawn (None if they can): no POD, or the mode n (the mode shown by
        default) not evaluated."""
        pod = self.pod()
        if isinstance(pod, str):
            return pod
        n = self.pod_shown_mode() if n is None else n
        if not 1 <= n <= pod["modes"]:
            return f"Mode {n} not evaluated: {pod['modes']} modes in {pod['file']} (POD modes: Miscellaneous)"
        return None

    def pod_energy(self, n):
        """Share of the energy of the mode n in the spectrum (%)."""
        eigVal = self.pod()["eigVal"]
        return 100 * eigVal[n - 1] / eigVal.sum()

    def pod_text(self):
        """The POD of the images, the modes exported (their share of the energy) and the mode shown, as written in
        the side panel."""
        pod = self.pod()
        if isinstance(pod, str):
            return pod
        text = f"POD of {len(pod['eigVal'])} images, {pod['modes']} modes evaluated"
        first, last = self.pod_range()
        modes = self.pod_modes_exported()
        if not modes:
            return text + f": mode {first} not evaluated"
        energy = sum(self.pod_energy(n) for n in modes)
        what = f"mode {first}" if len(modes) == 1 else f"modes {modes[0]} to {modes[-1]}"
        text += (f"\nExported: {what}" + (" summed: the background of the image of each map" if self.pod_combined()
                                          else "") + f", {energy:.3g} % of the energy")
        if last > modes[-1]:
            text += f" (modes {modes[-1] + 1} to {last} not evaluated)"
        n = self.pod_shown_mode()
        if len(modes) > 1 and n in modes and not self.pod_combined():
            text += f"\nShown: mode {n}, {self.pod_energy(n):.3g} %"
        return text

    def pod_mode_map(self, n):
        """Spatial mode n (unit norm) as an image (cached: the last one)."""
        pod = self.pod()
        return self.cached("pod_mode", (pod["file"], n), lambda: npz_row(pod["file"], "PHI", n - 1).reshape(
            pod["Dim"]).astype(np.float64))

    def draw_pod_mode(self, fig, prefix):
        missing = self.pod_missing("pod_mode")
        if missing:
            return self.message(fig, self.FIGURE_TITLES["pod_mode"], missing)
        if self.pod_combined():
            return self.draw_pod_background(fig, prefix)
        pod, n = self.pod(), self.pod_shown_mode()
        PHI = self.pod_mode_map(n)
        self.image_map(fig, PHI, f"{prefix}POD mode $\\phi_{{{n}}}$ of the images: {self.pod_energy(n):.3g} % of "
                                 "the energy" + self.refs(f"mode {n} of {pod['modes']}",
                                                          *self.images_span(len(pod['eigVal'])), scaled=False),
                       rf"$\phi_{{{n}}}$ (unit norm, a.u.)", "RdBu_r", *mode_limits(PHI))

    def pod_stack(self):
        """The spatial modes of the range evaluated, as images: an array (mode, row, column) (cached: the last
        range)."""
        pod, modes = self.pod(), self.pod_modes_exported()

        def compute():
            return np.stack([npz_row(pod["file"], "PHI", n - 1).reshape(pod["Dim"]) for n in modes])
        return self.cached("pod_stack", (pod["file"], tuple(modes)), compute)

    def pod_background(self, k):
        """The sum of the modes of the range for the image of the map k (the first image of its pair): each mode
        weighted by the temporal coefficient of that image, sum of a_n(i) phi_n: the background the pre-processing
        subtracts from the image i with these modes (intensity counts; the image as the POD saw it: masks and
        contrast enhancement before POD applied)."""
        pod, modes = self.pod(), self.pod_modes_exported()
        i = self.fields.settings["pairs"][k][0]
        a = pod["TCoeff"][np.array(modes) - 1, i]
        return np.tensordot(a, self.pod_stack().astype(np.float64), axes=1)

    def pod_background_limits(self):
        """Colour range of the sums of the modes, the same for all the maps: the percentiles IMAGE_PERCENTILES of
        those of IMAGE_SCALE_FRAMES maps evenly spaced."""
        def compute():
            n = self.fields.n
            maps = np.unique(np.linspace(0, n - 1, min(IMAGE_SCALE_FRAMES, n)).round().astype(int))
            values = np.concatenate([self.pod_background(k)[::IMAGE_STEP, ::IMAGE_STEP].ravel() for k in maps])
            values = values[np.isfinite(values)]
            if not values.size:
                return None, None
            low, high = (float(v) for v in np.percentile(values, IMAGE_PERCENTILES))
            return low, high if high > low else low + 1
        return self.cached("limits_pod_background", (self.pod()["file"], tuple(self.pod_modes_exported())), compute)

    def draw_pod_background(self, fig, prefix):
        """The sum of the modes of the range for the image of the map drawn (pod_background), at the positions of
        the vector fields, as an image."""
        modes = self.pod_modes_exported()
        k = int(np.clip(self.map_index, 0, self.fields.n - 1))
        what = f"mode {modes[0]}" if len(modes) == 1 else f"modes {modes[0]} to {modes[-1]}"
        energy = sum(self.pod_energy(n) for n in modes)
        # The image with the time of the map (each field of view its own in a stitched map)
        files = self.image_files()
        name = (os.path.basename(files[1][self.fields.settings["pairs"][k][0]])
                if not isinstance(files, str) and not isinstance(files[1], str) else None)
        when = self.map_time(k, scaled=False)
        if name:
            when = when.replace("\n(", f"\n({name}, ", 1)
        self.image_map(fig, self.pod_background(k), f"{prefix}POD background of the image: sum of {what} "
                                                    f"({energy:.3g} % of the energy)" + when,
                       "Intensity (counts)", "gray", *self.pod_background_limits())

    def images_span(self, n_images):
        """The span of the images of the POD (all of them), in the titles of its figures: their number, and their
        duration Delta t (as that of the statistics: the number of images over the frame rate), in flow times (with the
        flow time) or in s; their number alone in the sensor space, or the frame rate unknown."""
        rate = (self.fields.settings.get("camera") or {}).get("frame_rate_hz")
        parts = [f"{n_images} images"]
        if self.sensor() or not rate:
            return parts
        duration = n_images / rate
        if self.uses_flow_time():
            return parts + [rf"$\Delta t$ = {sig2(duration / self.flow_time())} ${TFLOW}$", self.t_flow_text()]
        return parts + [rf"$\Delta t$ = {duration:.4g} s"]

    def image_times(self, n_images):
        """Times of the images of the PIV (those of its maps: the first image of their pair) and their label: image
        number (sensor space, or the frame rate unknown), in s, or in flow times."""
        f = self.fields
        rate = (f.settings.get("camera") or {}).get("frame_rate_hz")
        k = np.arange(n_images, dtype=np.float64)
        if self.sensor() or not rate:
            return k + 1, "Image number"
        t = k / rate
        if self.uses_flow_time():
            return t / self.flow_time(), rf"$t\ /\ {TFLOW}$"
        return t, "$t$ (s)"

    def draw_pod_spectrum(self, fig, prefix):
        missing = self.pod_missing("pod_spectrum")
        if missing:
            return self.message(fig, self.FIGURE_TITLES["pod_spectrum"], missing)
        pod, n = self.pod(), self.pod_shown_mode()
        exported = self.pod_modes_exported()
        eigVal = pod["eigVal"]
        energy = 100 * eigVal / eigVal.sum()
        modes = np.arange(1, len(eigVal) + 1)
        left, right = fig.subplots(1, 2, width_ratios=(1, 1.4))
        left.semilogy(modes, energy, "o-", ms=2.5, lw=1, color="#2a78d6", label="Energy of the mode")
        left.semilogy(modes[:pod["modes"]], energy[:pod["modes"]], "o", ms=4, mfc="none", mec="#2a78d6",
                      label=f"Modes evaluated ({pod['modes']})")
        if len(exported) > 1:
            left.semilogy(exported, energy[np.array(exported) - 1], "o", ms=5, color="#f2a679",
                          label=f"Modes exported ({exported[0]} to {exported[-1]}: "
                                f"{energy[np.array(exported) - 1].sum():.3g} %)")
        left.semilogy(n, energy[n - 1], "o", ms=7, color="#eb6834", label=f"Mode {n}: {energy[n - 1]:.3g} %")
        left.set_xlabel("Mode number $n$")
        left.set_ylabel(r"Energy $\lambda_n\ /\ \Sigma\lambda$ (%)")
        left.grid(True, which="both", alpha=0.3)
        twin = left.twinx()
        twin.plot(modes, np.cumsum(energy), color="0.4", lw=0.8, ls=":", label="Cumulated")
        twin.set_ylabel("Cumulated energy (%)")
        twin.set_ylim(0, 100)
        handles = left.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
        left.legend(handles, [h.get_label() for h in handles], frameon=False, fontsize="x-small", loc="center right")
        left.set_title("Spectrum of the images", fontsize="medium")
        t, xlabel = self.image_times(pod["TCoeff"].shape[1])
        right.plot(t, pod["TCoeff"][n - 1], "-", lw=0.8, color="#eb6834")
        right.set_xlabel(xlabel)
        right.set_ylabel(rf"$\langle I, \phi_{{{n}}}\rangle$ (counts)")
        right.grid(True, alpha=0.3)
        right.set_title(f"Temporal coefficient of mode {n}: the projection of each image on it", fontsize="medium")
        fig.suptitle(f"{prefix}POD of the {len(eigVal)} images ({pod['modes']} modes evaluated)"
                     + self.refs(*self.images_span(len(eigVal))[1:], positions=False, scaled=False),
                     fontweight="bold", fontsize="medium")

    ############################################################################
    # Both

    def imaging_unavailable(self):
        """Why each figure of the images and of their POD can't be drawn: {key: reason, None if it can}."""
        return {key: self.imaging_missing(key) for key in FIGURE_FILES}

    def imaging_missing(self, key):
        """Why a figure of the images or of their POD can't be drawn (None if it can)."""
        if key in IMAGE_FIGURES:
            return self.image_missing(key)
        if key in POD_FIGURES:
            return self.pod_missing(key)
        return None

    def element(self, key):
        """Element of the names of the files of a figure of the images: that of the POD mode with the number of the
        mode drawn."""
        if key != "pod_mode":
            return FIGURE_FILES[key]
        if self.pod_combined():
            modes = self.pod_modes_exported()
            return f"PODBackground{modes[0]}" + (f"-{modes[-1]}" if len(modes) > 1 else "")
        return f"{FIGURE_FILES[key]}{self.pod_shown_mode()}"


class ImageProbe(ImageAnalysis):
    """The figures of the images and of the POD of a file of vector fields, from its settings only (the vector fields
    not read): availability."""

    def __init__(self, path, pod_mode=1):
        from types import SimpleNamespace

        from .fields import measurement_folder

        with np.load(path) as z:
            settings = json.loads(str(z["settings"])) if "settings" in z.files else {}
        self.fields = SimpleNamespace(settings=settings, folder=measurement_folder(path, settings),
                                      source=os.path.abspath(path), n=len(settings.get("pairs") or ()))
        self.settings = SimpleNamespace(pod_mode=pod_mode)
        self.cache = {}

    def cached(self, name, key, compute):
        if name not in self.cache or self.cache[name][0] != key:
            self.cache[name] = (key, compute())
        return self.cache[name][1]


def pod_modes(path):
    """The number of modes of the POD of the images of a file of vector fields (vectors.npz), from its settings
    only; None without POD (or if the file can't be read)."""
    try:
        pod = ImageProbe(path).pod()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    return None if isinstance(pod, str) else pod["modes"]


def availability(path, pod_mode=1):
    """Why each figure of the images and of the POD of a file of vector fields (vectors.npz) can't be drawn, from its
    settings only: {key: reason, None if it can} (all of them unavailable if the file can't be read)."""
    try:
        return ImageProbe(path, pod_mode).imaging_unavailable()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
        return {key: f"The vector fields could not be read: {e}" for key in FIGURE_FILES}
