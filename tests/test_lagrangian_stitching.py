"""The PTV in the post-processing (lagrangian.py) and the stitching of fields of view (archiving/post_archive.py)."""

import dataclasses
import json
import os

import pytest
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver

import synthetic as s
import velocihub as vh
from velocihub.archiving import post_archive as arch
from velocihub.postprocessing import PostProcessing, PostSettings


def post_of(folder, method="PIV", sensor=False, camera=s.CAMERA, **settings):
    path = os.path.join(folder, method, "vectors.npz")
    fields = vh.load_vectors(path, pixels=True) if sensor else vh.load_vectors(path, camera=camera)
    return PostProcessing(fields, PostSettings(**settings))


def test_ptv_figures(measurements):
    """The figures of the PTV for PTV results only; the title of the tracks: their duration (Δt in flow times or
    s, frames in pixels), not the pair."""
    ptv, piv = post_of(measurements["both"], "PTV"), post_of(measurements["both"], "PIV")
    assert ptv.ptv() and not piv.ptv()
    assert ptv.ptv_missing("ptv_tracks") is None and piv.ptv_missing("ptv_tracks")
    assert ptv.ptv_missing("ptv_vs_piv") is None  # The PIV of the same measurement to compare with
    title = ptv.plot("ptv_tracks", index=5).axes[0].get_title()
    assert r"$\Delta t$ =" in title and "t_\\mathrm{flow}" in title and "dots" not in title
    seconds = post_of(measurements["both"], "PTV", use_t_flow=False).plot("ptv_tracks", index=5).axes[0].get_title()
    assert r"$\Delta t$ =" in seconds and " s" in seconds
    frames = post_of(measurements["both"], "PTV", sensor=True).plot("ptv_tracks", index=5).axes[0].get_title()
    assert "frames" in frames


@pytest.fixture
def fields_of_view(measurements):
    """The same measurement in two fields of view (offsets 0 and 60 mm), with a range of POD modes."""
    posts = {}
    for fov, offset in (("FOV_1", 0.0), ("FOV_2", 60.0)):
        camera = dataclasses.replace(s.CAMERA, x_offset_mm=offset)
        posts[fov] = post_of(measurements["piv"], camera=camera, pod_mode=1, pod_mode_last=2)
    return posts


def stitched_maps(fig, n):
    """The maps of a stitched figure (its first n axes: the fields of view)."""
    return [a for ax in fig.axes[:n] for a in ax.collections + ax.images
            if a.get_array() is not None and not isinstance(a, Quiver)]


def test_stitched_maps(fields_of_view, tmp_path):
    """Each field of view at its offset, in the colour range and colour map forced (or the range of all), the images
    as images; the POD modes one by one, or summed per map; the time of a map in its title."""
    clims, cmaps = {"velocity": (0.01, 0.05)}, {"velocity": "magma"}
    parts = arch.instantaneous(fields_of_view, "velocity", 2, clims, cmaps)
    fig = Figure(figsize=arch.stitched_size(parts), layout="constrained")
    arch.draw_stitched(fig, "setup", parts)
    maps = stitched_maps(fig, 2)
    assert [tuple(m.get_clim()) for m in maps] == [(0.01, 0.05)] * 2 and {m.get_cmap().name for m in maps} == {"magma"}
    # The second field of view from its offset (its first column of vectors: SPACING px in)
    assert fig.axes[1].get_xlim()[0] == pytest.approx(60.0 + s.SPACING * s.CAMERA.fov_width_mm / s.WIDTH)
    parts = arch.instantaneous(fields_of_view, "raw_image", 2)
    fig = Figure(figsize=arch.stitched_size(parts), layout="constrained")
    arch.draw_stitched(fig, "setup", parts)
    assert all(len(ax.images) == 1 for ax in fig.axes[:2])
    first = next(iter(fields_of_view.values()))
    assert arch.map_label(first, 3).startswith(" · $t$ =")
    export = {"figures": ["pod_mode", "velocity"], "videos": [], "video_maps": [], "maps": [0, 2]}
    names = sorted(os.path.basename(p) for p in arch.stitch_setup("setup", fields_of_view, str(tmp_path), export, 0))
    assert names == ["setup_Stitched_InstantaneousVelocity_01.png", "setup_Stitched_InstantaneousVelocity_03.png",
                     "setup_Stitched_PODMode1.png", "setup_Stitched_PODMode2.png"]
    for post in fields_of_view.values():
        post.settings.pod_combine = True
    names = sorted(os.path.basename(p) for p in arch.stitch_setup("setup", fields_of_view, str(tmp_path / "sum"),
                                                                  {**export, "figures": ["pod_mode"]}, 0))
    assert names == ["setup_Stitched_PODBackground1-2_01.png", "setup_Stitched_PODBackground1-2_03.png"]


def test_metrics_exported(copy):
    """The key numbers of a PTV with those of its tracks."""
    folder = copy("ptv")
    post = post_of(folder, "PTV")
    post.export(os.path.join(folder, "out"), figures=["ptv_yield"])
    with open(os.path.join(folder, "out", "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["maps"] == s.FRAMES - 1 and metrics["ptv_tracks"] == s.PARTICLES
