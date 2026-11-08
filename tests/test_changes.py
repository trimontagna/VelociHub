"""The single reader of the images (greyscale.py) in every step, the format of the images exported, the camera (pixel
pitches, origin of the axes), the outlier filter of the rolling statistics, the times (DynamicStudio CSV, flow time
of the sensor space), the automatic number of POD modes, the jet theory add-on and the settings of PaIRS."""

import dataclasses
import importlib.util
import json
import os

import numpy as np
import pytest
from PIL import Image

import synthetic as s
import velocihub as vh
from velocihub import greyscale, piv, settings
from velocihub.fields import VectorFields, csv_times, load_csv_folder, outliers
from velocihub.piv.particles import count_particles
from velocihub.postprocessing import PostProcessing, PostSettings
from velocihub.preprocessing import PODBackgroundRemoval, check_export_depth, export_format_of, imread, mode_title

WORKERS = os.path.join(os.path.dirname(os.path.abspath(vh.__file__)), "piv")


def worker(name):
    """A worker script of the PIV/PTV (loaded as its interpreter runs it: from its file)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(WORKERS, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


################################################################################
# Images: one channel, the same in every step


@pytest.fixture
def colour_images(tmp_path):
    """The same grey 100 (red 200 for the colour one) in RGB, RGBA, palette and LA files, and 16 bit."""
    out = {}
    rgb = np.zeros((8, 10, 3), np.uint8)
    rgb[..., 0] = 200
    out["red"] = tmp_path / "red.png"
    Image.fromarray(rgb).save(out["red"])
    grey = np.full((8, 10, 4), 100, np.uint8)
    grey[..., 3] = 255
    out["rgba"] = tmp_path / "rgba.png"
    Image.fromarray(grey).save(out["rgba"])
    out["palette"] = tmp_path / "palette.png"
    Image.fromarray(grey[..., :3]).convert("P", palette=Image.Palette.ADAPTIVE).save(out["palette"])
    out["la"] = tmp_path / "la.png"
    Image.fromarray(grey[..., 0], "L").convert("LA").save(out["la"])
    deep = np.full((8, 10), 40000, np.uint16)
    out["deep"] = tmp_path / "deep.png"
    Image.fromarray(deep).save(out["deep"])
    return out


def test_one_channel_everywhere(colour_images):
    """The pre-processing, the PIV and PTV workers, the particle counter and the figures read the same image: the
    brightness max(R, G, B), alpha and palette indices left out, the depth kept."""
    pairs, ptv = worker("pairs_worker"), worker("ptv_worker")
    for name, expected in (("red", 200), ("rgba", 100), ("palette", 100), ("la", 100), ("deep", 40000)):
        path = str(colour_images[name])
        A = imread(path)
        assert A.ndim == 2 and np.all(A == expected), name
        assert np.all(pairs.read(path) == expected), name
        assert np.all(ptv.read(path) == expected) and ptv.read(path).dtype in (np.uint8, np.uint16), name
    assert imread(str(colour_images["deep"])).dtype == np.uint16
    # The particle counter takes a colour array as the readers do
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[20, 20, 1] = 250
    assert count_particles(rgb, (20, 20), (32,))["windows"][0]["count"] == 1


def test_to_grey():
    """HSV value of the colour channels, whatever their order; floats and 32 bit integers."""
    A = np.zeros((2, 2, 3), np.uint8)
    A[0, 0] = (10, 200, 30)
    assert greyscale.to_grey(A)[0, 0] == 200 and greyscale.to_grey(A[..., ::-1])[0, 0] == 200
    assert greyscale.to_grey(np.full((2, 2), 70000, np.int32)).dtype == np.float32
    assert greyscale.to_grey(np.full((2, 2), 1000, np.int32)).dtype == np.uint16
    assert greyscale.to_grey(np.ones((2, 2), bool)).max() == 255


################################################################################
# Format of the images exported


def test_export_formats(measurements, tmp_path):
    """PNG, BMP, TIFF or the format of the source; BMP refuses 16 bit; the images exported in another format aren't
    taken for those of the recipe."""
    folder = measurements["images"]
    assert export_format_of(".TIF") == "tiff" and export_format_of("") == "same"
    with pytest.raises(ValueError):
        export_format_of("jpeg2000")
    with pytest.raises(ValueError):
        check_export_depth("bmp", np.uint16)
    check_export_depth("png", np.uint16)
    with pytest.raises(ValueError):
        check_export_depth("same", np.float32, "a.png")
    with PODBackgroundRemoval(folder, pattern="*.tif", verbose=False) as session:
        session.set_contrast(["capping"], stage="pre", cap_n=10)
        out = str(tmp_path / "out")
        session.path_export = out
        session.export_format = "png"
        written = session.export_images(out)
        assert all(p.endswith(".png") for p in written)
        assert json.load(open(os.path.join(out, "preprocessing.json"), encoding="utf-8"))["format"] == "png"
        assert session.preprocessed_images() == written
        session.export_format = "bmp"
        assert session.preprocessed_images() is None  # Exported as PNG
        written = session.export_images(out)
        assert all(p.endswith(".bmp") for p in written)
        np.testing.assert_array_equal(imread(written[0]), imread(written[0].replace(".bmp", ".png")))


################################################################################
# Camera: pixel pitches and origin of the axes


def test_camera_geometry():
    """FOV height: the vertical pitch (else square pixels); the origin at fractions of the field of view."""
    camera = piv.Camera(100.0, fov_height_mm=60.0, x_shift=0.5, y_shift=0.25, x_offset_mm=10.0)
    assert camera.pitches(200, 100) == (0.5, 0.6)
    x, y = camera.positions(np.array([0.0, 200.0]), np.array([100.0, 0.0]), 200, 100)
    np.testing.assert_allclose(x, [10 - 50, 10 + 50])
    np.testing.assert_allclose(y, [-0.25 * 60, 0.75 * 60])
    assert piv.Camera(100.0).scale_y(200, 100) == 0.5
    assert piv.Camera(100.0).frame_rate_hz == 1.0
    # A camera saved by the previous versions: y = 0 at the middle of the height
    assert piv.Camera.from_record({"fov_width_mm": 100.0, "frame_rate_hz": 150.0}).y_shift == 0.5
    assert settings.camera_from_project({"fov_width": "100", "frame_rate": "150"})["y_shift"] == 0.5
    saved = settings.camera_to_project(settings.camera_record(camera))
    assert settings.camera_from_project(saved) == settings.camera_record(camera)


def test_vectors_with_the_camera_geometry(measurements):
    """The vector fields from the bottom-left corner by default (y upwards), the vertical velocities with the
    vertical pitch."""
    path = os.path.join(measurements["piv"], "PIV", "vectors.npz")
    square = vh.load_vectors(path, camera=dataclasses.replace(s.CAMERA, y_shift=0.0))
    tall = vh.load_vectors(path, camera=dataclasses.replace(s.CAMERA, y_shift=0.0, fov_height_mm=2 * 80.0 * s.HEIGHT
                                                              / s.WIDTH))
    pitch = s.CAMERA.fov_width_mm / s.WIDTH
    np.testing.assert_allclose(square.y[:, 0], (s.HEIGHT - s.grid()[1][:, 0][::-1]) * pitch, rtol=1e-6)
    np.testing.assert_allclose(tall.y, 2 * square.y, rtol=1e-6)
    np.testing.assert_allclose(tall.V, 2 * square.V, rtol=1e-5)
    np.testing.assert_allclose(tall.U, square.U, rtol=1e-6)
    assert tall.scale_y == pytest.approx(2 * pitch)
    sensor = vh.load_vectors(path, pixels=True)
    np.testing.assert_allclose(sensor.y[:, 0], s.HEIGHT - s.grid()[1][:, 0][::-1])


################################################################################
# Statistics


def fields_of(U, V=None, t=None):
    n, ny, nx = U.shape
    x, y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    V = np.zeros_like(U) if V is None else V
    return VectorFields(x=x, y=y, U=U, V=V, valid=np.isfinite(U), status=np.zeros((ny, nx), np.int16),
                        t=np.arange(n, dtype=float) if t is None else t, dt=1.0, scale=1.0,
                        settings={"units": "mm"})


def test_outlier_filter():
    """An outlier among 10 maps is removed (it couldn't be at 4 sigma with the vector in its own statistics); with
    rolling statistics, the vectors are compared with the maps of the window."""
    rng = np.random.default_rng(0)
    U = (1 + 0.01 * rng.standard_normal((10, 2, 2))).astype(np.float32)
    U[4, 0, 0] = 5.0
    assert outliers(U, 4).sum() == 1 and outliers(U, 4)[4, 0, 0]
    # A slow drift: its last maps are outliers of all the maps, not of their window
    drift = (np.linspace(0, 10, 200)[:, None, None] + 0.01 * rng.standard_normal((200, 1, 1))).astype(np.float32)
    drift[150] += 2.0
    post = PostProcessing(fields_of(drift), PostSettings(outlier_k=4, rolling=True, rolling_maps=20))
    post.map_index = 160
    U_window, _, removed = post.clean()
    assert len(U_window) == 20 and removed == 1 and np.isnan(U_window[150 - 141, 0, 0])
    # Against all the maps, the drift hides it
    assert PostProcessing(fields_of(drift), PostSettings(outlier_k=4)).clean()[2] == 0


def test_reference_velocity_any_direction():
    """The reference velocity: the highest mean speed, also of a flow towards -x."""
    U = -np.ones((5, 2, 3), np.float32)
    post = PostProcessing(fields_of(U), PostSettings(outlier_k=0))
    assert post.statistics()["u_ref"] == pytest.approx(1.0)
    assert np.all(post.statistics()["ti"][np.isfinite(post.statistics()["ti"])] == 0)


def test_sample_statistics():
    """Standard deviations with N - 1."""
    U = np.array([1, 2, 3, 4], np.float32)[:, None, None] * np.ones((1, 1, 1), np.float32)
    post = PostProcessing(fields_of(U), PostSettings(outlier_k=0))
    assert post.statistics()["u_rms"][0, 0] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))


################################################################################
# Times


def test_csv_times(tmp_path):
    """The time of each map from the names of the exports, else from the frame rate, else dt."""
    header = "\n" * 8 + "X[-],Y[-],X (m)[m],Y (m)[m],U[m/s],V[m/s],Status[-]\n"
    for k, name in enumerate(("Run.000001-6.6672.csv", "Run.000002-13.3344.csv", "Run.000003-20.0016.csv")):
        with open(tmp_path / name, "w") as f:
            f.write(header + "0,0,0.001,0.002,1,0,0\n1,0,0.002,0.002,1,0,0\n")
    assert csv_times(sorted(os.listdir(tmp_path))) == pytest.approx([6.6672e-3, 13.3344e-3, 20.0016e-3])
    fields = load_csv_folder(str(tmp_path), dt=1e-5, scale=0.1)
    assert np.diff(fields.t) == pytest.approx([6.6672e-3] * 2)
    assert fields.y[0, 0] == pytest.approx(2.0)  # mm, as exported
    for k, name in enumerate(sorted(os.listdir(tmp_path))):
        os.replace(tmp_path / name, tmp_path / f"map{k}.csv")
    assert np.diff(load_csv_folder(str(tmp_path), dt=1e-5, frame_rate_hz=150).t) == pytest.approx([1 / 150] * 2)


def test_sensor_flow_time_counts_frames(measurements):
    """In the sensor space the flow time is in frames: L_FoV / u_ref pairs of frame_step frames."""
    path = os.path.join(measurements["piv"], "PIV", "vectors.npz")
    post = PostProcessing(vh.load_vectors(path, pixels=True), PostSettings())
    one = post.default_flow_time()
    post.fields.settings["pairs"] = [[a, a + 3] for a, _ in post.fields.settings["pairs"]]
    post.cache.clear()
    assert post.default_flow_time() == pytest.approx(3 * one)


################################################################################
# POD


def test_automatic_modes_dimensionless(measurements):
    """The energy difference of the automatic number of modes is a fraction of the energy: a tolerance of 1 is met
    by the first mode whose temporal mean is below its tolerance, none by a tolerance of 0."""
    with PODBackgroundRemoval(measurements["images"], pattern="*.tif", verbose=False) as session:
        session.evaluate_spectrum()
        session.eps_auto_psi = 1.0
        session.eps_auto_sigma = 1.0
        assert session.automatic_N_modes() == 1
        session.eps_auto_sigma = 0.0
        assert session.automatic_N_modes() == 0
        title = mode_title(1, 3, session.eigVal)
        share = 100 * session.eigVal[0] / session.eigVal.sum()
        assert f"{share:.3g} %" in title


################################################################################
# Jet theory add-on


def test_jet_theory_add_on(measurements, tmp_path):
    """Off by default: its figures neither drawn nor exported, no numbers; on: the potential core."""
    path = os.path.join(measurements["piv"], "PIV", "vectors.npz")
    off = PostProcessing(vh.load_vectors(path, camera=s.CAMERA), PostSettings())
    assert not off.exported("core_map") and not off.map_available("core_map") and off.missing("core_profiles")
    assert "core_u_ref_ms" not in off.metrics()
    # Not drawn (a figure shown before the add-on was turned off): its message, the core not computed
    from matplotlib.figure import Figure
    off.potential_core = None  # Would fail if called
    for key in ("core_map", "core_profiles", "core_centreline"):
        fig = off.plot(key, Figure())
        assert not fig.axes and "jet theory is off" in fig.texts[0].get_text()
    del off.potential_core
    on = PostProcessing(vh.load_vectors(path, camera=s.CAMERA), PostSettings(jet_theory=True))
    assert on.exported("core_profiles") and on.potential_core()["error"] is None
    assert "core_u_ref_ms" in on.metrics()
    assert any("jet theory" in step for step in on.settings.steps())


def test_piv_quality_only_for_a_piv(measurements):
    """The quality of a PIV (valid vectors, peak ratio, peak locking) for the results of a PIV only: for a PTV neither
    drawn nor exported, and its valid cells not computed (statistics, metrics, batch, project)."""
    from matplotlib.figure import Figure
    from velocihub.batch import valid_percent
    from velocihub.postprocessing import PIV_GROUP
    from velocihub.project import valid_text
    f_piv = vh.load_vectors(os.path.join(measurements["both"], "PIV", "vectors.npz"), camera=s.CAMERA)
    f_ptv = vh.load_vectors(os.path.join(measurements["both"], "PTV", "vectors.npz"), camera=s.CAMERA)
    pv, pt = PostProcessing(f_piv, PostSettings()), PostProcessing(f_ptv, PostSettings())
    for key in PIV_GROUP[1]:
        assert pv.exported(key) and pv.missing(key) is None
        assert not pt.exported(key) and pt.missing(key) and not pt.map_available(key)
        assert "PIV" in pt.plot(key, Figure()).texts[0].get_text()
    assert pt.statistics()["valid"] is None and "valid_percent" not in pt.metrics()
    assert pv.statistics()["valid"].shape == f_piv.shape and pv.metrics()["valid_percent"] > 90
    assert valid_percent(f_ptv) is None and valid_text(None) == ""
    assert valid_text(valid_percent(f_piv)).endswith("% valid vectors")


################################################################################
# PaIRS


def test_pairs_parameters(tmp_path):
    """The settings of the PaIRS process: their defaults (its standard process), checked, sent to the worker as PaIRS
    numbers them."""
    engine = piv.get_engine("pairs")
    checked = engine.check({})
    assert checked["image_interpolation"] == "bspline3" and checked["median_test"] == "universal"
    assert all(p.help for p in engine.parameters)
    with pytest.raises(ValueError):
        engine.check({"median_test": "nogueira"})
    with pytest.raises(ValueError):
        engine.check({"median_kernel": 0})
    codes = {key: piv.pairs_engine.PAIRS_CODES[key][checked[key]] for key in piv.pairs_engine.PAIRS_CODES}
    assert codes == worker("pairs_worker").STANDARD  # The defaults: the standard process of PaIRS


################################################################################
# Turbulence: kinetic energy and integral scales


def ar1(n, a, shape, rng, axis=0):
    """An AR(1) process along an axis: ρ(k) = a^k, its integral scale (trapezoids) (1 + a) / (2 (1 - a)) steps."""
    noise = rng.normal(size=(n,) + shape)
    out = np.empty_like(noise)
    out[0] = noise[0] / np.sqrt(1 - a * a)
    for k in range(1, n):
        out[k] = a * out[k - 1] + noise[k]
    return np.moveaxis(out, 0, axis)


def test_integral_scales():
    """The integral of the autocorrelation to its first zero: a triangle (exact), an AR(1) process in time and in
    space (its known scale); undefined without zero crossing or with a scale under MIN_LAGS lags."""
    from velocihub import turbulence
    tri = np.clip(1 - np.arange(10) / 4.0, -0.5, None)[:, None]  # Zero at lag 4: integral 2 steps
    assert abs(turbulence.integral_to_zero(tri, 0.5)[0] - 1.0) < 1e-12
    assert np.isnan(turbulence.integral_to_zero(np.exp(-np.arange(10) / 3.0)[:, None], 1.0)[0])  # No zero crossing
    assert np.isnan(turbulence.integral_to_zero(np.array([1.0, -0.1, 0.0])[:, None], 1.0)[0])  # Not resolved
    rng = np.random.default_rng(1)
    a, exact = 0.9, (1 + 0.9) / (2 * (1 - 0.9))
    U = ar1(20000, a, (3, 3), rng) + 3.0
    U[rng.random(U.shape) < 0.05] = np.nan  # Gaps: no bias
    T = turbulence.integral_time(U, 0.01) / 0.01
    # Unbiased, its scatter about ±10 % with 20000 maps (a few times the scale over the record)
    assert abs(T.mean() / exact - 1) < 0.06 and np.allclose(T, exact, rtol=0.25), T
    V = ar1(160, a, (500, 3), rng, axis=2)  # 500 maps, 3 rows, 160 columns: correlated along x
    L = turbulence.integral_length(V, 2.0, 2)
    assert abs(np.nanmedian(L[:, 30:-30]) / 2.0 / exact - 1) < 0.2, np.nanmedian(L) / 2.0
    # Uncorrelated along y: undefined, or the noise of the autocorrelation (a scale of about a cell at most)
    Ly = turbulence.integral_length(V, 2.0, 1)
    assert np.isnan(Ly).mean() > 0.5 and not (Ly > 1.5 * 2.0).any()


def test_turbulence_figures(measurements):
    """Turbulent kinetic energy (σ_u² + σ_v²) / 2; the integral time scale needs a time series of enough maps (its
    reason said otherwise); the integral length scales in the unit of the positions."""
    from matplotlib.figure import Figure
    path = os.path.join(measurements["piv"], "PIV", "vectors.npz")
    post = PostProcessing(vh.load_vectors(path, camera=s.CAMERA), PostSettings())
    st = post.statistics()
    fig = post.plot("tke", Figure())
    k = fig.axes[0].collections[0].get_array()
    assert np.isclose(np.nanmax(k), np.nanmax((st["u_rms"] ** 2 + st["v_rms"] ** 2) / 2))
    assert post.integral_time()[1] and "maps" in post.plot("integral_time", Figure()).texts[0].get_text()
    for key in ("integral_length_x", "integral_length_y"):
        assert post.plot(key, Figure()).axes  # A map (NaN where the scale isn't resolved)
    m = post.metrics()
    assert m["median_integral_time_s"] is None and "median_integral_length_x_mm" in m


def test_planar_dissipation_and_ke_scales(measurements):
    """The planar dissipation: exact for fluctuations linear in x and y (their gradients known); the k-ε scales k/ε and
    k^1.5/ε, only in the physical space; the integral length scales of v."""
    from matplotlib.figure import Figure
    from velocihub import turbulence
    rng = np.random.default_rng(3)
    n, x, y = 400, np.linspace(0, 0.01, 9), np.linspace(0, 0.005, 6)
    a, b = rng.normal(size=n), rng.normal(size=n)
    a, b = a - a.mean(), b - b.mean()
    X, Y = np.meshgrid(x, y)
    U = 2.0 + a[:, None, None] * X + b[:, None, None] * Y  # ∂u'/∂x = a, ∂u'/∂y = b
    V = np.zeros_like(U)
    eps = turbulence.planar_dissipation(U, V, x, y, 1e-6, chunk=64)
    assert np.allclose(eps, 1e-6 * (2 * np.mean(a ** 2) + 3 * np.mean(b ** 2)))
    path = os.path.join(measurements["piv"], "PIV", "vectors.npz")
    post = PostProcessing(vh.load_vectors(path, camera=s.CAMERA), PostSettings(nu=1e-6))
    T, L = post.ke_scales()
    st = post.statistics()
    k = (st["u_rms"] ** 2 + st["v_rms"] ** 2) / 2
    ok = np.isfinite(T)
    eps = post.dissipation()
    assert ok.any() and np.allclose(T[ok], k[ok] / eps[ok]) and np.allclose(L[ok], k[ok] ** 1.5 / eps[ok])
    for key in ("ke_time", "ke_length", "integral_length_vx", "integral_length_vy"):
        assert post.plot(key, Figure()).axes, key
    assert r"$\nu$ = 1e-06" in post.plot("ke_time", Figure()).axes[0].get_title()
    sensor = PostProcessing(vh.load_vectors(path, pixels=True), PostSettings())
    assert sensor.ke_scales() == (None, None) and "physical" in sensor.plot("ke_length", Figure()).texts[0].get_text()
    with pytest.raises(ValueError):
        PostSettings(nu=0)


def test_colour_bar_keeps_the_image_size():
    """A colour bar added to an image with a long title (the Colormap button of the page Velocimetry): the image keeps
    its size. Matplotlib's wrapping of the title (wrap=True) made the constrained layout shrink it; the titles are
    broken into lines (title_lines) instead."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from velocihub.preprocessing import title_lines
    title = ("Pair 150 of 15369: images 150 and 151 (original image)\nframe_000149.png\n"
             "No displacements for this pair yet: Run the PIV/PTV (F5) or Run on the pair shown (F6)")
    assert title_lines(title) == title and max(len(line) for line in title_lines("word " * 80).split("\n")) <= 110
    for size in ((17.4, 9.0), (12, 6)):
        fig = Figure(figsize=size, dpi=100, layout="compressed")
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        im = ax.imshow(np.zeros((720, 1280)), extent=(0.5, 1280.5, 720.5, 0.5))
        ax.set_title(title_lines(title), fontsize="medium")
        fig.canvas.draw()
        before = ax.get_window_extent().width
        fig.colorbar(im, ax=ax, label="Intensity (grey levels)")
        fig.canvas.draw()
        assert abs(ax.get_window_extent().width - before) < 2, size
