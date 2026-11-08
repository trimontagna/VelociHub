"""The post-processing of vector fields (postprocessing.py, fields.py): the conversion with the camera, every figure
drawn, the times of the titles, the export (its files, the results of a previous export) and its folders."""

import json
import os

import numpy as np
import pytest
from matplotlib.text import Text

import synthetic as s
import velocihub as vh
from velocihub.postprocessing import (FIGURE_TITLES, POSTPROCESSING_FOLDER, POSTPROCESSING_PTV_FOLDER, PostProcessing,
                                      PostSettings, count_results, default_export_folder, export_folder_name,
                                      previous_results, remove_results)


def post_of(folder, method="PIV", sensor=False, **settings):
    path = os.path.join(folder, method, "vectors.npz")
    fields = vh.load_vectors(path, pixels=True) if sensor else vh.load_vectors(path, camera=s.CAMERA)
    return PostProcessing(fields, PostSettings(**settings))


def texts(fig):
    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


def test_vectors_converted_with_the_camera(measurements):
    """mm and m/s with the camera (the jet: JET_PEAK px per frame), pixels without."""
    post, sensor = post_of(measurements["piv"]), post_of(measurements["piv"], sensor=True)
    scale = s.CAMERA.fov_width_mm / s.WIDTH
    assert post.fields.scale == pytest.approx(scale)
    peak = s.JET_PEAK * scale / 1000 * s.CAMERA.frame_rate_hz
    with post.all_maps():
        assert np.nanmax(post.statistics()["mean_U"]) == pytest.approx(peak, rel=0.05)
    assert np.nanmax(sensor.statistics()["mean_U"]) == pytest.approx(s.JET_PEAK, rel=0.05)
    assert sensor.sensor() and not post.sensor()
    np.testing.assert_allclose(post.fields.x[0], sensor.fields.x[0] * scale)


@pytest.mark.parametrize("sensor,rolling", [(False, False), (True, False), (False, True)])
def test_every_figure_drawn(measurements, sensor, rolling):
    """Each figure drawn without error (the PTV ones say why they can't be: PIV results)."""
    post = post_of(measurements["piv"], sensor=sensor, rolling=rolling, rolling_maps=5, l_ref=10)
    for key in FIGURE_TITLES:
        fig = post.plot(key, index=7)
        assert not any("not available:" in t for t in texts(fig)), (key, texts(fig))


def test_times_of_the_titles(measurements):
    """Δt for the figures of a span (statistics, POD), t for those of one time (instantaneous maps)."""
    post = post_of(measurements["piv"])
    mean = post.plot("mean_velocity").axes[0].get_title()
    assert r"$\Delta t$ =" in mean and "$t$ =" not in mean
    assert "$t$ =" in post.plot("velocity", index=3).axes[0].get_title()
    assert r"\Delta t" in post.plot("convergence").axes[0].get_xlabel()
    assert r"$\Delta t$ =" in post.plot("pod_mode").axes[0].get_title()
    rolling = post_of(measurements["piv"], rolling=True, rolling_maps=5)
    title = rolling.plot("mean_velocity", index=8).axes[0].get_title()
    assert "$t$ =" in title and r"$\Delta t$ =" in title  # The map of the window, and its span
    flow = post_of(measurements["piv"], use_t_flow=False).plot("mean_velocity").axes[0].get_title()
    assert r"$\Delta t$ =" in flow and " s" in flow


def test_export_and_previous_results(copy):
    """The files of an export (instantaneous figures per map in their folder), then found and removed as previous
    results (the other files kept)."""
    folder = copy("piv")
    post = post_of(folder)
    out = os.path.join(folder, POSTPROCESSING_FOLDER)
    written = post.export(out, figures=["velocity", "mean_velocity"], maps=[0, 4], name="proj")
    names = sorted(os.path.relpath(p, out) for p in written)
    assert "proj_PostProcessing_MeanU.png" in names
    assert os.path.join("proj_PostProcessing_InstantaneousVelocity",
                        "proj_PostProcessing_InstantaneousVelocity_05.png") in names
    assert {"metrics.json", "postprocessing.json"} <= set(names)
    open(os.path.join(out, "notes.txt"), "w").close()
    previous = previous_results(out, "proj")
    assert count_results(previous) == len(written)
    assert remove_results(previous) == len(written)
    assert os.listdir(out) == ["notes.txt"]
    assert previous_results(os.path.join(folder, "none"), "proj") == []


def test_pod_modes_exported(copy):
    """The POD modes of a range: a PNG each; summed: the background of the image of each map (a figure per map)."""
    folder = copy("piv")
    post = post_of(folder, pod_mode=1, pod_mode_last=3)
    out = os.path.join(folder, "range")
    names = sorted(os.path.basename(p) for p in post.export(out, figures=["pod_mode"], name="p") if p.endswith(".png"))
    assert names == [f"p_PostProcessing_PODMode{n}.png" for n in (1, 2, 3)]
    post.settings.pod_combine = True
    assert post.instantaneous("pod_mode")
    out = os.path.join(folder, "summed")
    written = post.export(out, figures=["pod_mode"], maps=[0, 2], name="p")
    assert sorted(os.path.basename(p) for p in written if p.endswith(".png")) == [
        "p_PostProcessing_PODBackground1-3_01.png", "p_PostProcessing_PODBackground1-3_03.png"]
    with open(os.path.join(out, "postprocessing.json"), encoding="utf-8") as f:
        record = json.load(f)
    assert record["pod_summed"] is True and record["pod_modes"] == [1, 2, 3]


def test_export_folders(measurements):
    """The subfolder of the pipeline in PostProcessing (PostProcessing/PaIRS, PostProcessing/TracTrac; the default
    folders of the previous versions, PostProcessing and PostProcessing_PTV, give it), a folder of its own kept;
    PostProcessing for vector fields of no pipeline."""
    piv_maps, ptv_maps = (os.path.join(measurements["both"], m) for m in ("PIV", "PTV"))
    assert default_export_folder(piv_maps) == os.path.join(POSTPROCESSING_FOLDER, "PaIRS")
    assert default_export_folder(ptv_maps) == os.path.join(POSTPROCESSING_FOLDER, "TracTrac")
    assert export_folder_name("PostProcessing", ptv_maps) == os.path.join(POSTPROCESSING_FOLDER, "TracTrac")
    assert export_folder_name(POSTPROCESSING_PTV_FOLDER, ptv_maps) == os.path.join(POSTPROCESSING_FOLDER, "TracTrac")
    assert export_folder_name(None, piv_maps) == os.path.join(POSTPROCESSING_FOLDER, "PaIRS")
    assert export_folder_name("Mine", ptv_maps) == "Mine"
    assert default_export_folder(measurements["images"]) == POSTPROCESSING_FOLDER
