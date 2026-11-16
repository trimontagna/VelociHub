"""The benchmark (velocihub/benchmark): every check of every case and level against the analytic flows, the flows
themselves, the test measurements (particle images) and the menu Tests.

A check that fails is a regression of VelociHub (or an improvement: then tighten its tolerance in cases.py). A known
issue is expected to fail (xfail, with its reason); it fails the test when it starts passing: the issue is solved,
make it a normal check. The benchmark runs once for the module (about a minute); its report is written with

    pytest tests/test_benchmark.py --benchmark-report <folder>
"""

import json
import os
import time

import numpy as np
import pytest

pytest.importorskip("spicy_vki")

from velocihub.benchmark import CASES, FLOWS  # noqa: E402
from velocihub.benchmark.cases import BACKGROUND_CHECKS  # noqa: E402
from velocihub.benchmark.flows import HEIGHT, WIDTH  # noqa: E402

CHECKS = ([(key, i, check) for key, case in CASES.items() for i, check in enumerate(case.checks)]
          + [("background", i, check) for i, check in enumerate(BACKGROUND_CHECKS)])


@pytest.fixture(scope="module")
def benchmark(request):
    from velocihub.benchmark import report, run_benchmark

    result = run_benchmark()
    folder = request.config.getoption("--benchmark-report")
    if folder:
        print(f"\nBenchmark report: {report.write_report(result, folder)}")
    out = {r.case.key: r for r in result.cases}
    out["background"] = result.extra["background"]
    return out


@pytest.mark.parametrize("key", list(FLOWS))
def test_flows_divergence_free(key):
    """The analytic flows are divergence-free (their formulas checked), and 0 inside their body."""
    flow = FLOWS[key]()
    X, Y = np.meshgrid(np.linspace(1, WIDTH - 1, 60), np.linspace(1, HEIGHT - 1, 40))
    solid = flow.solid(X, Y)
    # Away from the surface of the body (the differences would straddle it)
    near = flow.solid(X + 1, Y) | flow.solid(X - 1, Y) | flow.solid(X, Y + 1) | flow.solid(X, Y - 1)
    assert np.abs(flow.divergence(X, Y, 3.0)[~near]).max() < 1e-6
    U, V = flow.velocity(X, Y, 3.0)
    assert not U[solid].any() and not V[solid].any()


def test_blasius_thicknesses():
    """The Blasius profile integrates to its displacement and momentum thicknesses, its wall shear is f''(0)."""
    flow = FLOWS["blasius"]()
    y = flow.y_wall + np.linspace(0, 15 * flow.scale(200), 200001)
    U, _ = flow.velocity(np.full_like(y, 200), y)
    u = U / flow.u_ref
    assert np.trapezoid(1 - u, y) == pytest.approx(flow.displacement_thickness(200), rel=1e-6)
    assert np.trapezoid(u * (1 - u), y) == pytest.approx(flow.momentum_thickness(200), rel=1e-5)
    _, dudy, _, _ = flow.gradients(200, flow.y_wall + 1e-2)
    assert dudy == pytest.approx(flow.wall_shear(200), rel=1e-3)


def test_cylinder_shoulders():
    """Potential flow: twice the free stream at the shoulders, stagnation at the front and back."""
    flow = FLOWS["cylinder"]()
    cx, cy = flow.centre
    U, V = flow.velocity(np.array([cx, cx - flow.radius, cx + flow.radius]), np.array([cy + flow.radius, cy, cy]))
    assert U[0] == pytest.approx(2 * flow.u_ref) and np.allclose(U[1:], 0, atol=1e-9) and np.allclose(V, 0, atol=1e-9)


def test_turbulence_statistics():
    """The synthetic turbulence: its statistics over a long line are those of its modes (variances, the
    autocorrelation along x), and it is frozen: the field at t is the one at 0 shifted by U0 t."""
    flow = FLOWS["turbulence"]()
    x = np.linspace(0, 100000, 200001)
    u, v = flow.fluctuation(x, np.full_like(x, 100.0))
    uu, vv, _ = flow.variances()
    assert u.var() == pytest.approx(uu, rel=0.05) and v.var() == pytest.approx(vv, rel=0.05)
    lag = 40  # 20 px
    assert np.mean(u[:-lag] * u[lag:]) / u.var() == pytest.approx(flow.correlation([20.0], "x", "u")[0], abs=0.03)
    U0, _ = flow.velocity(np.array([50.0]), np.array([80.0]), 0.0)
    U1, _ = flow.velocity(np.array([50.0 + 4 * flow.u0]), np.array([80.0]), 4.0)
    assert U1 == pytest.approx(U0)
    assert flow.dissipation(1.0, 16) < flow.dissipation(1.0)


def test_test_workspace(tmp_path):
    """The test measurements: the images of the cases, their mask, their exact fields as PIV results (read by the
    post-processing as the exact field), the case file, and a workspace of these measurements."""
    from velocihub.benchmark.images import BACKGROUND_CASE, FRAMES, test_workspace
    from velocihub.benchmark.measurement import CAMERA, MS_PER_PX
    from velocihub.fields import load_vectors
    from velocihub.workspace import Workspace

    file = test_workspace(str(tmp_path), ["cylinder"])
    ws = Workspace.load(file)
    # The measurements (images), and their exact fields: projects with results only, opened in the post-processing
    projects = {p.path: p for p in ws.projects}
    assert set(projects) == {"cylinder", "cylinder/Exact", BACKGROUND_CASE, f"{BACKGROUND_CASE}/Exact"}
    assert projects["cylinder"].images and projects["cylinder/Exact"].results
    folder = tmp_path / "cylinder"
    images = sorted(folder.glob("img_*.tif"))
    assert len(images) == FRAMES
    with open(folder / "masks.json") as f:
        assert len(json.load(f)["masks"][0]["points"]) > 10
    with open(folder / "benchmark_case.json") as f:
        assert json.load(f)["case"] == "cylinder"
    fields = load_vectors(str(folder / "Exact"), camera=CAMERA)
    assert fields.n == FRAMES - 1 and (fields.status == 2).any()
    flow = FLOWS["cylinder"]()
    U, _ = flow.velocity(fields.x / 0.2, fields.y / 0.2)
    flowing = fields.status == 0
    assert np.allclose(fields.U[0][flowing] / MS_PER_PX, U[flowing], atol=1e-4)
    # The particles: bright spots on a dark background, none in the cylinder
    from PIL import Image
    img = np.asarray(Image.open(images[5]), float)
    assert img.max() > 120 and np.median(img) < 40
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH] + 0.5
    inside = np.hypot(xx - flow.centre[0], yy - (HEIGHT - flow.centre[1])) < flow.radius - 3
    assert img[inside].max() < 60


@pytest.mark.parametrize("key, index, check", CHECKS,
                         ids=[f"{k}-{c.level}-{c.scenario}-{c.method}-{c.metric}-{i}" for k, i, c in CHECKS])
def test_check(benchmark, key, index, check):
    result = benchmark[key]
    for scenario, data in getattr(result, "data", {}).items():
        if hasattr(data, "errors"):
            assert not data.errors, f"{scenario}: {data.errors}"
    r = result.checks[index] if key == "background" else next(c for c in result.checks if c.check is check)
    assert r.check is check
    message = f"{check.label}: {r.value} (tolerance {check.op} {r.limit})"
    if check.known_issue:
        if r.status == "known":
            pytest.xfail(f"known issue: {check.known_issue} ({message})")
        pytest.fail(f"known issue now passing, make it a normal check in cases.py: {message}")
    assert r.status == "pass", message


def test_menu_runs_a_case(gui, monkeypatch):
    """Tests > Benchmark > a case: the benchmark runs in its own process (python -m velocihub.benchmark), its
    progress in the status log, and its report is opened at the end (and remembered: Open the last report)."""
    import webbrowser

    from velocihub import prefs

    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    menu = gui.app.menus["Tests"]
    labels = [menu.entrycget(i, "label") for i in range(menu.index("end") + 1) if menu.type(i) != "separator"]
    assert labels[0] == "Benchmark" and "Test workspace" in labels and "Open the last report" in labels
    gui.app.benchmark.run(["uniform"], levels=["post", "processing"])
    deadline = time.time() + 240
    while not opened and time.time() < deadline:
        gui.root.update()
        time.sleep(0.1)
    assert opened and os.path.isfile(opened[0]), gui.app.status_listbox.get(0, 5)
    assert prefs.getpref("benchmark_last_report") == opened[0]
    status = gui.app.status_listbox.get(0, "end")
    assert any("checks passed" in line for line in status) and any("SPICY 1/" in line for line in status)
    assert not gui.boxes
