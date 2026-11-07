"""The Python API (api.py: Project, projects, run) and the batch functions it shares with the interface (batch.py,
batch_run.py): the settings saved in the project file, the pre-processing up to date, the post-processing exported
(overwriting the previous export or not), the results chosen (PIV, PTV), and the archive of projects."""

import os
import zipfile

import synthetic as s
import velocihub as vh
from velocihub import batch
from velocihub.archiving import archive as ap
from velocihub.project import read_project


def test_settings_saved_and_read(copy):
    """The settings of the three steps kept in the project file, read again by another Project."""
    folder = copy("piv")
    p = vh.Project(folder)
    p.set_preprocessing(filters=["capping"], cap_n=7, modes_removed=2)
    p.set_postprocessing(camera=s.CAMERA, outlier_k=3, figures=["mean_velocity"], pod_mode=2)
    p.save()
    again = vh.Project(folder)
    assert again.preprocessing["contrast"]["cap_n"] == 7
    assert again.postprocessing["settings"]["outlier_k"] == 3 and again.postprocessing["settings"]["pod_mode"] == 2
    assert read_project(folder)["postprocessing"]["export_choice"]["figures"] == ["mean_velocity"]
    assert [q.folder for q in vh.projects(os.path.dirname(folder))] == [folder]


def test_postprocessing_exported_and_overwritten(copy):
    """The quantities chosen exported; again: the previous export kept beside it, or replaced (overwrite)."""
    folder = copy("piv")
    p = vh.Project(folder)
    p.set_postprocessing(camera=s.CAMERA, figures=["mean_velocity", "u_rms"])
    status, _ = p.postprocess()
    out = os.path.join(folder, "PostProcessing")
    assert status == "done" and any(n.endswith("URms.png") for n in os.listdir(out))
    p.set_postprocessing(figures=["mean_velocity"])
    p.postprocess()
    assert any(n.endswith("URms.png") for n in os.listdir(out))  # Not overwritten: the previous figure left
    p.postprocess(overwrite=True)
    assert not any(n.endswith("URms.png") for n in os.listdir(out))
    assert any(n.endswith("MeanU.png") for n in os.listdir(out))


def test_results_chosen(copy):
    """A project with PIV and PTV results: those chosen post-processed, in their own export folder."""
    folder = copy("both")
    p = vh.Project(folder)
    p.set_postprocessing(results="PTV", camera=s.CAMERA, figures=["ptv_yield"])
    assert p.postprocess()[0] == "done"
    assert os.path.isfile(os.path.join(folder, "PostProcessing_PTV", "metrics.json"))
    assert not os.path.isdir(os.path.join(folder, "PostProcessing"))


def test_preprocessing_up_to_date(copy):
    """The images pre-processed with the same recipe reused: skipped."""
    folder = copy("images")
    p = vh.Project(folder)
    p.set_preprocessing(modes_removed=s.MODES_REMOVED, filters=[])
    first = p.preprocess()
    assert first.status in ("done", "skipped"), first.describe()
    second = p.preprocess()
    assert second.status == "skipped", second.describe()


def test_archive(copy, tmp_path):
    """The archive of a project: the categories chosen (its project file, the vector fields), not the others."""
    folder = copy("piv", "root", "m")
    vh.Project(folder).save()
    unit = ap.scan_project(folder, os.path.dirname(folder))
    base, entries = ap.archive_entries([unit], lambda u: ["project", "vectors"])
    output = os.path.join(str(tmp_path), "archive.zip")
    errors = ap.write_archive(output, entries, "zip", lambda errors: "summary", ap.Progress(sum(e[2] for e in entries)))
    assert not errors
    with zipfile.ZipFile(output) as z:
        names = z.namelist()
    assert any(n.endswith("vectors.npz") for n in names) and any(n.endswith("VelociHub_project.json") for n in names)
    assert not any(n.endswith(".tif") for n in names)
    assert batch.valid_percent(vh.load_vectors(os.path.join(folder, "PIV"), pixels=True)) > 90
