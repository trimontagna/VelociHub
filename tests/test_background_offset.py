"""The offset after the POD background removal (preprocessing: background_offset): the pixels darker than the
background of the modes kept instead of clipped at 0, automatic by default; the masked pixels at the offset; recorded
with the recipe (the records of the previous versions: no offset). On the cylinder images of the benchmark with
reflections (their particles known)."""

import os

import numpy as np
import pytest

from velocihub.preprocessing import DEFAULT_BACKGROUND_OFFSET, PODBackgroundRemoval, background_offset_of
from velocihub.project import comparable_recipe


def test_offset_values():
    assert DEFAULT_BACKGROUND_OFFSET == "auto"
    assert [background_offset_of(v) for v in (None, "", " Auto ", "12", 3.5, 0)] == ["auto", "auto", "auto", 12, 3.5, 0]
    for bad in ("-1", "much", -2):
        with pytest.raises(ValueError):
            background_offset_of(bad)


@pytest.fixture(scope="module")
def measurement(tmp_path_factory):
    from velocihub.benchmark import FLOWS
    from velocihub.benchmark.images import BACKGROUND_CASE, write_measurement

    folder = os.path.join(str(tmp_path_factory.mktemp("offset")), BACKGROUND_CASE)
    _, parts = write_measurement(FLOWS["cylinder"](), folder, reflections=True, seed=99)
    return folder, parts


def processed(session, n, offset):
    session.background_offset = offset
    modes = session.load_modes(n)
    return np.array([session.process_image(os.path.join(session.path_in, name), i, modes)[0]
                     for i, name in enumerate(session.file_in)])


def test_offset_keeps_the_particles(measurement):
    """Automatic: a few % of the pixels at 0 instead of most of them, the particles keep their brightness (their
    images: the processed images less the offset, against the particles rendered), the masked pixels at the offset;
    the background (the modes) the same; 0: clipped at 0 as before."""
    from velocihub.benchmark.images import _projection

    folder, parts = measurement
    with PODBackgroundRemoval(folder, pattern="img_*.tif", verbose=False) as s:
        s.load_masks(os.path.join(folder, "masks.json"))
        s.evaluate_modes(2)
        auto = processed(s, 1, "auto")
        value = s.offset_used
        clipped = processed(s, 1, 0)
        mask = s.masked_pixels(auto[0].shape)
        _, background_clipped = s.process_image(os.path.join(s.path_in, s.file_in[3]), 3, s.load_modes(1))
        assert "offset 0" in " ".join(s.recipe_steps(1))
        s.background_offset = "auto"
        _, background_auto = s.process_image(os.path.join(s.path_in, s.file_in[3]), 3, s.load_modes(1))
        record = s.recipe_record(1)
    assert auto.dtype == np.uint8 and 10 <= value <= 40
    assert np.mean(auto[:, ~mask] == 0) < 0.03 and np.mean(clipped[:, ~mask] == 0) > 0.5
    assert (auto[:, mask] == value).all() and (clipped[:, mask] == 0).all()
    assert np.array_equal(background_auto, background_clipped)

    def kept(images):
        return np.mean([_projection(img.astype(float), p, ~mask) / e
                        for img, p, e in zip(images, parts["particles"], parts["energy"])])
    assert kept(auto) > 0.95 > 0.9 > kept(clipped)
    # Recorded with the recipe; the previous versions (no offset) as 0
    assert record["background_removal"]["offset"] == "auto"
    old = {**record, "background_removal": {k: v for k, v in record["background_removal"].items() if k != "offset"}}
    assert comparable_recipe(old)["background_removal"]["offset"] == 0 != comparable_recipe(record)
