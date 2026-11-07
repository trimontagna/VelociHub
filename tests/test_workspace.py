"""The workspace (workspace.py): projects detected under root folders (images, PIV and PTV results; not the folders of
results), its file, the projects moved (followed, lost, relinked) and removed."""

import json
import os
import shutil

from velocihub.project import PROJECT_FILE, PROJECT_FORMAT
from velocihub.workspace import WORKSPACE_FILE, Workspace


def touch(root, *parts, data=b"x"):
    path = os.path.join(root, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def tree(T):
    for i in range(3):
        touch(T, "root", "FOV_1", "setupA", f"im_{i}.tif")
        touch(T, "root", "FOV_1", "setupB", f"im_{i}.tif")
        touch(T, "root", "FOV_1", "setupA", "PreProcessing", f"pre_{i}.tif")  # Results: not a project
        touch(T, "root", "FOV_1", "setupA", "PostProcessing", "fig", f"a_PostProcessing_x_{i}.png")
        touch(T, "root", "Stitched", "setupA", f"setupA_Stitched_x_{i}.png")
    touch(T, "root", "FOV_1", "Calibration", "cal.tif")  # One image: not a project
    touch(T, "root", "FOV_2", "setupA", "PIV", "vectors.npz", data=b"12345")  # Results only
    for i in range(2):
        touch(T, "single", "lone", f"im_{i}.png")


def paths(ws):
    """The paths of the projects of the root folder."""
    root = ws.roots[0]["folder"]
    return sorted(p.path for p in ws.projects if p.root == root)


def test_projects_followed(tmp_path):
    T = str(tmp_path)
    tree(T)
    ws = Workspace(os.path.join(T, WORKSPACE_FILE))
    ws.add_root(os.path.join(T, "root"))
    lone = ws.add_project(os.path.join(T, "single", "lone"))
    assert paths(ws) == ["FOV_1/setupA", "FOV_1/setupB", "FOV_2/setupA"]
    results = next(p for p in ws.projects if p.path == "FOV_2/setupA")
    assert results.results and not results.images
    ws.save()
    assert [p.folder for p in Workspace.load(ws.file).projects] == [p.folder for p in ws.projects]
    # Removed: not added again by a refresh
    ws.remove([p for p in ws.projects if p.path == "FOV_1/setupB"])
    ws.refresh()
    assert paths(ws) == ["FOV_1/setupA", "FOV_2/setupA"]
    # Moved inside the root: followed
    os.makedirs(os.path.join(T, "root", "FOV_2b"))
    shutil.move(os.path.join(T, "root", "FOV_2", "setupA"), os.path.join(T, "root", "FOV_2b", "setupA"))
    ws.refresh()
    assert "FOV_2b/setupA" in paths(ws) and not any(p.lost for p in ws.projects)
    # The root moved: lost, then relinked
    shutil.move(os.path.join(T, "root"), os.path.join(T, "root_moved"))
    ws.refresh()
    assert ws.roots[0]["lost"] and all(p.lost for p in ws.projects if p is not lone)
    ws.relink_root(ws.roots[0]["folder"], os.path.join(T, "root_moved"))
    assert not any(p.lost for p in ws.projects)
    # A node moved out of the root: relinked
    shutil.move(os.path.join(T, "root_moved", "FOV_1"), os.path.join(T, "elsewhere"))
    ws.refresh()
    assert next(p for p in ws.projects if p.path == "FOV_1/setupA").lost
    ws.relink_node(ws.roots[0]["folder"], "FOV_1", os.path.join(T, "elsewhere"))
    setup = next(p for p in ws.projects if p.path == "FOV_1/setupA")
    assert not setup.lost and os.path.dirname(setup.folder) == os.path.join(T, "elsewhere")
    # A single project relinked with its project file
    shutil.move(os.path.join(T, "single", "lone"), os.path.join(T, "lone2"))
    ws.refresh()
    assert lone.lost
    with open(os.path.join(T, "lone2", PROJECT_FILE), "w") as f:
        json.dump({"format": PROJECT_FORMAT}, f)
    ws.relink_project(lone, os.path.join(T, "lone2", PROJECT_FILE))
    assert not lone.lost and lone.folder == os.path.join(T, "lone2")
    # The tree moved with the workspace file: found by their relative paths
    ws.save()
    shutil.move(T, T + "_whole")
    moved = Workspace.load(os.path.join(T + "_whole", WORKSPACE_FILE))
    moved.refresh()
    assert moved.projects and not any(p.lost for p in moved.projects)
    shutil.rmtree(T + "_whole")


def test_piv_and_ptv_results(copy, tmp_path):
    """A project with PIV and PTV results: both detected (their folders); PTV only: that folder."""
    copy("both", "campaign", "both")
    copy("ptv", "campaign", "ptv_only")
    ws = Workspace(os.path.join(str(tmp_path), WORKSPACE_FILE))
    ws.add_root(os.path.join(str(tmp_path), "campaign"))
    found = {p.path: p for p in ws.projects}
    both, ptv = found["both"], found["ptv_only"]
    assert set(both.results_folders()) == {"PIV", "PTV"} and both.ptv and both.images
    assert list(ptv.results_folders()) == ["PTV"] and ptv.maps.endswith("PTV")
