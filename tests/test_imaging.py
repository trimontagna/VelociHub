"""The images and their POD in the post-processing (imaging.py): the images found (raw, pre-processed), their place,
the POD read, and its sum: the background the pre-processing subtracted."""

import os

import numpy as np

import synthetic as s
import velocihub as vh
from velocihub.imaging import availability, npz_row
from velocihub.postprocessing import PostProcessing, PostSettings
from velocihub.preprocessing import cast_like, imread


def post_of(folder, sensor=False, **settings):
    path = os.path.join(folder, "PIV", "vectors.npz")
    fields = vh.load_vectors(path, pixels=True) if sensor else vh.load_vectors(path, camera=s.CAMERA)
    return PostProcessing(fields, PostSettings(**settings))


def test_images_found(measurements):
    """The images of the PIV (pre-processed) and the raw image of each, the first image of the pair of a map."""
    post = post_of(measurements["piv"])
    piv_images, raw = post.image_files()
    assert all(os.path.dirname(p).endswith("PreProcessing") for p in piv_images)
    assert os.path.basename(post.image_path("raw_image", 3)) == "img_003.tif"
    assert post.image_missing("raw_image") is None and post.image_missing("piv_image") is None
    fig = post.plot("raw_image", index=3)
    np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), imread(raw[3]))


def test_image_positions(measurements):
    """The images at the positions of the maps: in mm with the camera, in pixels (y downwards) without."""
    extent, _, _ = post_of(measurements["piv"]).image_positions(s.WIDTH, s.HEIGHT)
    half = s.HEIGHT / 2 * s.CAMERA.fov_width_mm / s.WIDTH
    np.testing.assert_allclose(extent, (0, s.CAMERA.fov_width_mm, -half, half))
    sensor, _, _ = post_of(measurements["piv"], sensor=True).image_positions(s.WIDTH, s.HEIGHT)
    assert sensor == (0.0, float(s.WIDTH), float(s.HEIGHT), 0.0)


def test_raw_images_removed(copy):
    """The raw images removed, those pre-processed kept: only the raw image can't be shown (nor exported)."""
    folder = copy("piv")
    for name in os.listdir(folder):
        if name.endswith(".tif"):
            os.remove(os.path.join(folder, name))
    post = post_of(folder)
    assert "raw images" in post.image_missing("raw_image") and post.image_missing("piv_image") is None
    assert not post.exported("raw_image") and post.exported("piv_image")
    probe = availability(os.path.join(folder, "PIV", "vectors.npz"))
    assert probe["raw_image"] and probe["piv_image"] is None and probe["pod_mode"] is None


def test_pod_sum_is_the_background_removed(measurements):
    """The raw image minus the sum of the modes removed (each weighted by the temporal coefficient of the image) is
    the image the pre-processing exported, to the count."""
    post = post_of(measurements["piv"], pod_mode=1, pod_mode_last=s.MODES_REMOVED, pod_combine=True)
    piv_images, raw = post.image_files()
    for k in (0, 6, post.fields.n - 1):
        i = post.fields.settings["pairs"][k][0]
        A = imread(raw[i])
        expected = cast_like(A.astype(np.float64) - post.pod_background(k), A)
        np.testing.assert_array_equal(expected, imread(piv_images[i]))


def test_pod_read(measurements):
    """A mode read alone from pod.npz, its share of the energy in the title; a mode not evaluated: said, not
    exported."""
    post = post_of(measurements["piv"], pod_mode=2)
    path = os.path.join(measurements["piv"], "PreProcessing", "pod.npz")
    with np.load(path) as z:
        np.testing.assert_array_equal(npz_row(path, "PHI", 1), z["PHI"][1])
        energy = 100 * z["eigVal"][1] / z["eigVal"].sum()
    assert f"{energy:.3g} %" in post.plot("pod_mode").axes[0].get_title()
    post.settings.pod_mode = s.MODES + 1
    assert "not evaluated" in post.pod_missing("pod_mode") and not post.exported("pod_mode")


def test_no_pod(copy):
    folder = copy("piv")
    os.remove(os.path.join(folder, "PreProcessing", "pod.npz"))
    post = post_of(folder)
    assert post.pod_text().startswith("No POD") and not post.exported("pod_spectrum")
    assert availability(os.path.join(folder, "PIV", "vectors.npz"))["pod_mode"]
