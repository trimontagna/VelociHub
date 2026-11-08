"""Displacements of a pair of images, as the image tile of the page Velocimetry shows them (its views |d|, d_x, d_y):
on the grid of the vector fields for a PIV, at the particles tracked for a PTV (tracks.npz: pair_particles), with the
colour limits of the maps (displacement_limits). Displacements in pixels, y upwards; positions in pixels from the
top-left corner of the images.

    from velocihub.piv import displacements

    d = displacements.pair_displacements(fields, k)     # fields: load_vectors(..., pixels=True); k: the pair
    d["kind"], d["x"], d["y"], d["dx"], d["dy"]         # "grid" (2D arrays) or "particles" (1D arrays)
"""

import numpy as np


# Arrays of tracks.npz read to draw the particles tracked
TRACK_ARRAYS = ("frame", "id", "x", "y", "u", "v")


def read_tracks(path):
    """The particles tracked by a PTV (TRACK_ARRAYS of tracks.npz), in the order of their frames."""
    with np.load(path) as z:
        T = {q: z[q] for q in TRACK_ARRAYS}
    if T["frame"].size and (np.diff(T["frame"]) < 0).any():
        order = np.argsort(T["frame"], kind="stable")
        T = {q: a[order] for q, a in T.items()}
    return T


def pair_particles(T, k):
    """Displacements of the particles tracked over the pair k (the frames k and k + 1 of the tracks T, read_tracks):
    the particles of both frames, at the middle of their displacement (px from the top-left corner of the images) and
    their displacement (px, y upwards). The tracks have no row at the first and last frames of the sequence: for the
    pairs there, the particles of their other frame with their velocity (px per frame: the mean of their
    displacements from the frame before and to the frame after; linked False). None without particles."""
    frame = T["frame"]
    lo, mid, hi = np.searchsorted(frame, [k, k + 1, k + 2])
    if lo < mid and mid < hi:
        a, b = slice(lo, mid), slice(mid, hi)
        _, i, j = np.intersect1d(T["id"][a], T["id"][b], return_indices=True)
        x0, y0, x1, y1 = T["x"][a][i], T["y"][a][i], T["x"][b][j], T["y"][b][j]
        x, y, dx, dy, linked = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y0 - y1, True
    elif lo < hi:
        rows = slice(lo, hi)
        x, y, dx, dy, linked = T["x"][rows], T["y"][rows], T["u"][rows], -T["v"][rows], False
    else:
        return None
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & np.isfinite(dy)
    if not ok.any():
        return None
    return dict(x=np.asarray(x[ok], np.float64), y=np.asarray(y[ok], np.float64),
                dx=np.asarray(dx[ok], np.float64), dy=np.asarray(dy[ok], np.float64), linked=linked)


def displacement_component(key, dx, dy):
    """The quantity of the map key of displacements dx, dy: |d|, d_x or d_y."""
    return {"displacement": lambda: np.hypot(dx, dy), "displacement_x": lambda: dx,
            "displacement_y": lambda: dy}[key]()


def displacement_limits(key, F):
    """Colour limits of the displacements F of the map key (99th percentile, as the maps of the post-processing):
    from 0 for |d|, symmetric around 0 for d_y."""
    v = np.asarray(F, np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return 0.0, 1.0
    if key == "displacement":
        return 0.0, float(np.percentile(v, 99)) or 1.0
    if key == "displacement_y":
        d = float(np.percentile(np.abs(v), 99)) or 1.0
        return -d, d
    low, high = float(np.percentile(v, 1)), float(np.percentile(v, 99))
    return (low, high) if high > low else (low - 0.5, high + 0.5)


def grid_pixels(fields):
    """Positions of the vectors in the image (px from its top-left corner, y downwards): the vector fields read in
    pixels (load_vectors(..., pixels=True)) have y upwards from the middle of the height of the image."""
    return fields.x, fields.settings["image_px"][1] / 2 - fields.y


def pair_displacements(fields, k, tracks=None):
    """Displacements of the map k of vector fields read in pixels (load_vectors(path, pixels=True)), as the image
    tile of the page Velocimetry draws them: for a PTV whose particles tracked are known (tracks: read_tracks of
    fields.tracks, read if None), the particles of the pair ("kind": "particles", 1D arrays: pair_particles);
    otherwise the grid ("kind": "grid", 2D arrays). Returns {"kind", "x", "y" (px from the top-left corner), "dx",
    "dy" (px, y upwards)}, or None if the displacements in pixels are unknown."""
    if fields.u_px is None:
        return None
    if fields.particles is not None:
        if tracks is None and fields.tracks:
            tracks = read_tracks(fields.tracks)
        particles = pair_particles(tracks, k) if tracks is not None else None
        if particles is not None:
            return {**particles, "kind": "particles"}
    x, y = grid_pixels(fields)
    return {"kind": "grid", "x": x, "y": y, "dx": fields.u_px[k], "dy": fields.v_px[k]}
