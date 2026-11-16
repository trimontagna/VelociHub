"""Particle images of the cases of the benchmark, and the test of the background removal.

The images of a case are a time-resolved sequence (the pairs: consecutive frames, the displacement of the flow per
frame): particles seeded at random over the images and a margin around them, advected by the flow (fourth-order
Runge-Kutta, RK_STEPS steps per frame; a particle that leaves the margin, or ends in the body, is seeded again at
random in the margin), drawn as Gaussian spots (diameter about 2.5 px, peak intensity at random) whose brightness
follows the energy of the laser pulse of the frame, over a dark background with camera noise; the body of the flow is
a dark region. render(..., reflections=True) adds stationary features of known shape: the bright rim of the body,
lines (a streak, the reflection of a wall) and circles (a ring, a glare spot), as the reflections of real images
(their brightness follows the laser energy too).

The test measurements (test_workspace): each case as a measurement folder like those of VelociHub: the raw images,
its mask (masks.json) if it has a body, the exact fields of its pairs written as PIV results (Exact/vectors.npz:
post-processable at once), the description of the case (benchmark_case.json); plus the measurement background_removal
(the cylinder with reflections). The folder holds a workspace (VelociHub_workspace.json) of these measurements: the
interface opens it as its workspace (menu Tests).

The background removal test (run_background): the measurement with reflections, the POD background removal of
VelociHub (PODBackgroundRemoval, its mask loaded) with 0 to N modes removed, and how much of the reflections and of
the particles each one keeps: their projections on the known shapes (a matched filter), and the lines and circles a
Hough transform still finds in the mean pre-processed image."""

import json
import os
import shutil
import tempfile

import numpy as np
from PIL import Image

from .flows import FLOWS, HEIGHT, WIDTH
from .measurement import EXACT_FOLDER, write_case_file, write_exact

# Rendering: particles per pixel, margin around the images (px), Runge-Kutta steps per frame, particle image radius
# (Gaussian sigma, px), peak intensity range, background and camera noise (grey levels), laser energy fluctuation
SEEDING, MARGIN, RK_STEPS, SIGMA_P = 0.015, 48, 4, 0.8
PEAK = (110.0, 210.0)
BACKGROUND, CAMERA_NOISE, LASER_JITTER = 12.0, 2.0, 0.05
# Frames of the steady cases, of the turbulence
FRAMES, TURBULENCE_FRAMES = 41, 201
# The folder of the test workspace, its measurement of the background removal test
TEST_WORKSPACE = os.path.join(os.path.expanduser("~"), "VelociHub", "benchmarks", "test_workspace")
BACKGROUND_CASE = "background_removal"


def frames_of(flow):
    return FRAMES if flow.steady else TURBULENCE_FRAMES


################################################################################
# Particles


def _inside_domain(p):
    return ((p[:, 0] > -MARGIN) & (p[:, 0] < WIDTH + MARGIN) & (p[:, 1] > -MARGIN) & (p[:, 1] < HEIGHT + MARGIN))


def _seed(flow, n, rng, margin_only=False):
    """n positions (X, Y up) at random over the images and the margin (only the margin: margin_only), out of the
    body."""
    out = np.empty((0, 2))
    while len(out) < n:
        p = np.column_stack([rng.uniform(-MARGIN, WIDTH + MARGIN, 2 * n), rng.uniform(-MARGIN, HEIGHT + MARGIN, 2 * n)])
        keep = ~flow.solid(p[:, 0], p[:, 1])
        if margin_only:
            keep &= ~((p[:, 0] >= 0) & (p[:, 0] <= WIDTH) & (p[:, 1] >= 0) & (p[:, 1] <= HEIGHT))
        out = np.concatenate([out, p[keep]])
    return out[:n]


def advect(flow, p, t, rng):
    """The particles p (X, Y) from the frame t to t + 1 (Runge-Kutta 4, RK_STEPS steps); those that leave the margin
    or end in the body seeded again in the margin."""
    h = 1.0 / RK_STEPS

    def vel(q, s):
        return np.column_stack(flow.velocity(q[:, 0], q[:, 1], s))
    for i in range(RK_STEPS):
        s = t + i * h
        k1 = vel(p, s)
        k2 = vel(p + h / 2 * k1, s + h / 2)
        k3 = vel(p + h / 2 * k2, s + h / 2)
        k4 = vel(p + h * k3, s + h)
        p = p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    lost = ~_inside_domain(p) | flow.solid(p[:, 0], p[:, 1])
    if lost.any():
        p[lost] = _seed(flow, int(lost.sum()), rng, margin_only=True)
    return p


def draw_particles(p, peaks):
    """The particles (X, Y up) as Gaussian spots on a black image (float)."""
    image = np.zeros((HEIGHT, WIDTH))
    x, y = p[:, 0], HEIGHT - p[:, 1]  # Image coordinates, pixel centres at j + 1/2
    r = int(np.ceil(3 * SIGMA_P))
    ok = (x > -r) & (x < WIDTH + r) & (y > -r) & (y < HEIGHT + r)
    x, y, peaks = x[ok], y[ok], peaks[ok]
    cx, cy = np.floor(x).astype(int), np.floor(y).astype(int)
    for dj in range(-r, r + 1):
        for di in range(-r, r + 1):
            col, row = cx + dj, cy + di
            inside = (col >= 0) & (col < WIDTH) & (row >= 0) & (row < HEIGHT)
            d2 = (col + 0.5 - x) ** 2 + (row + 0.5 - y) ** 2
            np.add.at(image, (row[inside], col[inside]), (peaks * np.exp(-d2 / (2 * SIGMA_P**2)))[inside])
    return image


################################################################################
# Stationary features


def _line(x0, y0, x1, y1, width):
    """A bright line of unit peak between two points (image coordinates), Gaussian across (sigma: width)."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH] + 0.5
    d = np.array([x1 - x0, y1 - y0], float)
    length = np.hypot(*d)
    d /= length
    s = np.clip((xx - x0) * d[0] + (yy - y0) * d[1], 0, length)
    dist = np.hypot(xx - (x0 + s * d[0]), yy - (y0 + s * d[1]))
    return np.exp(-dist**2 / (2 * width**2))


def _ring(cx, cy, radius, width):
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH] + 0.5
    return np.exp(-(np.hypot(xx - cx, yy - cy) - radius) ** 2 / (2 * width**2))


def _spot(cx, cy, sigma):
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH] + 0.5
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


def reflections_of(flow):
    """The stationary features of known shape (image coordinates): {name: (kind, image of unit peak times its
    intensity, geometry)}: the rim of the body, a streak, a wall line, a ring and a glare spot."""
    out = {"streak": ("line", 70 * _line(30, 40, 300, 150, 1.0), (30, 40, 300, 150)),
           "wall line": ("line", 60 * _line(20, HEIGHT - 24, WIDTH - 20, HEIGHT - 24, 1.2),
                         (20, HEIGHT - 24, WIDTH - 20, HEIGHT - 24)),
           "ring": ("circle", 60 * _ring(400, 90, 34, 1.2), (400, 90, 34)),
           "glare": ("spot", 45 * _spot(110, 230, 10), (110, 230, 10))}
    if flow.key == "cylinder":
        cx, cy = flow.centre
        out["rim"] = ("circle", 90 * _ring(cx, HEIGHT - cy, flow.radius + 1.5, 1.2), (cx, HEIGHT - cy, flow.radius))
    return out


def body_image(flow):
    """The body of the flow in the images (a dark region: 0 outside, 1 inside)."""
    from .measurement import solid_pixels
    return solid_pixels(flow).astype(float)


def mask_polygons(flow):
    """The mask of the body of a flow (the polygons of PODBackgroundRemoval.save_masks: pixel centres at integer
    coordinates from 1), a little larger than the body; [] without body."""
    if flow.key == "cylinder":
        cx, cy = flow.centre
        a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        r = flow.radius + 2
        return [[[float(cx + r * np.cos(t) + 0.5), float(HEIGHT - cy + r * np.sin(t) + 0.5)] for t in a]]
    if flow.key == "blasius":
        top = HEIGHT - flow.y_wall - 1 + 0.5
        return [[[0.5, top], [WIDTH + 0.5, top], [WIDTH + 0.5, HEIGHT + 0.5], [0.5, HEIGHT + 0.5]]]
    return []


################################################################################
# Images


def render(flow, folder, frames=None, seed=0, reflections=False, prefix="img"):
    """The images of a flow in folder (prefix_0000.tif...): frames images (frames_of by default). Returns their
    paths and the parts of each image ({"particles": [...], "energy": [...], "positions": the particles of each
    image (X, Y up), "peaks": their peak intensities (before the laser energy), "features": the stationary features
    (float image, before the laser energy)})."""
    frames = frames or frames_of(flow)
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = int(round(SEEDING * (WIDTH + 2 * MARGIN) * (HEIGHT + 2 * MARGIN)))
    p = _seed(flow, n, rng)
    peaks = rng.uniform(*PEAK, n)
    body = body_image(flow)
    yy = np.mgrid[0:HEIGHT, 0:WIDTH][0]
    background = BACKGROUND + 6 * (1 - yy / HEIGHT)
    features = np.zeros((HEIGHT, WIDTH))
    if reflections:
        for _, image, _ in reflections_of(flow).values():
            features += image
    paths, parts = [], {"particles": [], "energy": [], "positions": [], "peaks": peaks, "features": features}
    for k in range(frames):
        if k:
            p = advect(flow, p, k - 1, rng)
        energy = 1 + LASER_JITTER * rng.standard_normal()
        particles = draw_particles(p, peaks) * (1 - body)
        image = background * (1 - body) + 25 * body + energy * (particles + features)
        image += rng.normal(0, CAMERA_NOISE, image.shape)
        path = os.path.join(folder, f"{prefix}_{k:04d}.tif")
        Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8)).save(path)
        paths.append(path)
        parts["particles"].append(particles.astype(np.float32))
        parts["energy"].append(energy)
        parts["positions"].append(p.copy())
    return paths, parts


def write_measurement(flow, folder, reflections=False, seed=0, spacing=16):
    """A test measurement of a flow in folder: its images, its mask (masks.json), its exact fields (Exact), the
    description of its case. Returns the paths of the images and their parts (render)."""
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    paths, parts = render(flow, folder, seed=seed, reflections=reflections)
    polygons = mask_polygons(flow)
    if polygons:
        with open(os.path.join(folder, "masks.json"), "w", encoding="utf-8") as f:
            json.dump({"masks": [{"name": f"{flow.name} body", "enabled": True, "points": poly}
                                 for poly in polygons]}, f, indent=1)
    write_exact(folder, flow, spacing, [k + 0.5 for k in range(len(paths) - 1)], images=paths, name=EXACT_FOLDER)
    write_case_file(folder, flow, {"frames": len(paths), "seeding_ppp": SEEDING, "reflections": reflections,
                                   "mask_file": "masks.json" if polygons else None,
                                   "exact_results": f"{EXACT_FOLDER}/vectors.npz"})
    return paths, parts


def test_workspace(root=None, cases=None, report=None):
    """The test measurements of the cases (all by default) and of the background removal in root (TEST_WORKSPACE by
    default), with their workspace file. Returns the workspace file."""
    from ..workspace import WORKSPACE_FILE, Workspace

    root = os.path.abspath(root or TEST_WORKSPACE)
    report = report or (lambda text: None)
    os.makedirs(root, exist_ok=True)
    for key in cases or list(FLOWS):
        flow = FLOWS[key]()
        report(f"Images: {flow.name} ({frames_of(flow)} frames)")
        write_measurement(flow, os.path.join(root, key), seed=list(FLOWS).index(key))
    report("Images: background removal (cylinder with reflections)")
    write_measurement(FLOWS["cylinder"](), os.path.join(root, BACKGROUND_CASE), reflections=True, seed=99)
    file = os.path.join(root, WORKSPACE_FILE)
    ws = Workspace(file)
    ws.add_root(root)
    ws.save(file)
    return file


################################################################################
# The background removal test


def _projection(image, shape, support):
    """The amplitude of a known shape in an image (matched filter): the least-squares factor of the shape over its
    support, both with their mean removed."""
    a = image[support] - image[support].mean()
    b = shape[support] - shape[support].mean()
    return float(a @ b / (b @ b))


def hough_found(image, kind, geometry):
    """The known line or circle found by a Hough transform in an image (8-bit): True if one is within 4 px (lines:
    both ends within 12 px of the line) and 6° of it."""
    import cv2

    img = cv2.GaussianBlur(np.clip(image, 0, 255).astype(np.uint8), (5, 5), 1.0)
    if kind == "line":
        edges = cv2.Canny(img, 20, 50)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 60, minLineLength=80, maxLineGap=10)
        if lines is None:
            return False
        x0, y0, x1, y1 = geometry
        d = np.array([x1 - x0, y1 - y0], float)
        d /= np.hypot(*d)
        normal = np.array([-d[1], d[0]])
        for a, b, c, e in np.asarray(lines).reshape(-1, 4):
            seg = np.array([c - a, e - b], float)
            seg /= max(np.hypot(*seg), 1e-9)
            angle = np.degrees(np.arccos(min(1.0, abs(seg @ d))))
            dist = max(abs((np.array([a, b]) - [x0, y0]) @ normal), abs((np.array([c, e]) - [x0, y0]) @ normal))
            if angle < 6 and dist < 12:
                return True
        return False
    cx, cy, r = geometry
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=18,
                               minRadius=int(r * 0.8), maxRadius=int(r * 1.25))
    if circles is None:
        return False
    return any(np.hypot(x - cx, y - cy) < 4 and abs(rr - r) < 5 for x, y, rr in np.asarray(circles).reshape(-1, 3))


# The particles measured one by one: isolated (no other within ISOLATED px), their window (px on each side)
ISOLATED, WINDOW = 5.0, 3
# A particle over a reflection: the reflection brighter than ON_FEATURE grey levels at its centre
ON_FEATURE = 10.0
# A particle dimmed: less than DIMMED % of its brightness kept
DIMMED = 50.0


def isolated_particles(positions, flow, mask):
    """The particles of an image (X, Y up) measurable one by one: their window in the image, out of the mask and
    the body, no other particle within ISOLATED px. Returns their indices and image coordinates (x, y down)."""
    from scipy.spatial import cKDTree

    x, y = positions[:, 0], HEIGHT - positions[:, 1]
    r = WINDOW + 1
    inside = (x > r) & (x < WIDTH - r) & (y > r) & (y < HEIGHT - r)
    near = cKDTree(positions).query(positions, k=2)[0][:, 1]
    keep = inside & (near > ISOLATED) & ~flow.solid(positions[:, 0], positions[:, 1])
    idx = np.flatnonzero(keep)
    if mask is not None:
        idx = idx[~mask[y[idx].astype(int), x[idx].astype(int)]]
    return idx, x[idx], y[idx]


def particle_amplitudes(image, x, y):
    """The peak intensity of particles at image positions x, y measured in an image: least-squares amplitude of
    their Gaussian spot (SIGMA_P) plus a constant over their window."""
    cx, cy = np.floor(x).astype(int), np.floor(y).astype(int)
    d = np.arange(-WINDOW, WINDOW + 1)
    rows, cols = cy[:, None, None] + d[None, :, None], cx[:, None, None] + d[None, None, :]
    g = np.exp(-((cols + 0.5 - x[:, None, None]) ** 2 + (rows + 0.5 - y[:, None, None]) ** 2) / (2 * SIGMA_P**2))
    a = image[rows, cols]
    g = g.reshape(len(x), -1)
    a = a.reshape(len(x), -1)
    g0, a0 = g - g.mean(1, keepdims=True), a - a.mean(1, keepdims=True)
    return (g0 * a0).sum(1) / (g0 * g0).sum(1)


def run_background(work=None, modes=(0, 1, 2, 3), report=None, keep=False):
    """The background removal test: the measurement with reflections rendered in work, its POD (the mask of the
    cylinder loaded) and the images processed with each number of modes removed. Returns {"modes": {n: {metric}}},
    the features, and the mean images (raw, and processed for each n) for the report."""
    from ..preprocessing import PODBackgroundRemoval

    report = report or (lambda text: None)
    own = work is None
    work = work or tempfile.mkdtemp(prefix="velocihub_background_")
    flow = FLOWS["cylinder"]()
    try:
        folder = os.path.join(work, BACKGROUND_CASE)
        report("Background removal: images with reflections")
        paths, parts = write_measurement(flow, folder, reflections=True, seed=99)
        feats = reflections_of(flow)
        energy = np.asarray(parts["energy"])
        out = {"modes": {}, "features": {k: (kind, geo) for k, (kind, _, geo) in feats.items()}}
        raw_mean = np.mean([np.asarray(Image.open(p), np.float64) for p in paths], 0)
        out["raw_mean"], out["means"] = raw_mean, {}
        report("Background removal: POD of the images")
        with PODBackgroundRemoval(folder, pattern="img_*.tif", verbose=False) as session:
            session.load_masks(os.path.join(folder, "masks.json"))
            session.evaluate_modes(max(modes) + 2)
            names = session.file_in
            for n in modes:
                loaded = session.load_modes(n)
                images = [np.asarray(session.process_image(os.path.join(session.path_in, name), i, loaded)[0],
                                     np.float64) for i, name in enumerate(names)]
                m = {}
                mask = session.masked_pixels(images[0].shape)
                # The offset added after the removal (automatic by default), out of the images measured below
                offset = float(session.offset_used or 0) if n else 0.0
                if n:
                    m["offset"] = offset
                # The particles kept: the projection of each image on its particles, out of the mask
                support = ~mask
                m["particles_kept"] = 100 * float(np.mean([_projection(img, part, support) / e
                                                           for img, part, e in zip(images, parts["particles"],
                                                                                   energy)]))
                # The particles one by one (the isolated ones): their peak measured in the processed image / their
                # peak in the raw image, over a reflection or not
                ratios, over, where = [], [], []
                for img, pos, e in zip(images, parts["positions"], energy):
                    idx, x, y = isolated_particles(pos, flow, mask)
                    ratios.append(100 * particle_amplitudes(img, x, y) / (parts["peaks"][idx] * e))
                    # The image whose particles are the most dimmed (a mode carrying the particles of a few images)
                    m["dimmed_worst_image"] = max(m.get("dimmed_worst_image", 0.0),
                                                  float(100 * np.mean(ratios[-1] < DIMMED)))
                    over.append(parts["features"][y.astype(int), x.astype(int)] > ON_FEATURE)
                    where.append(np.column_stack([x, y]))
                ratios, over, where = np.concatenate(ratios), np.concatenate(over), np.concatenate(where)
                m["particle_kept_median"] = float(np.median(ratios))
                m["particle_kept_p05"] = float(np.percentile(ratios, 5))
                m["particles_dimmed"] = float(100 * np.mean(ratios < DIMMED))
                m["particle_kept_over_features"] = float(np.median(ratios[over])) if over.any() else np.nan
                m["particles_measured"] = float(len(ratios))
                out.setdefault("particle_ratios", {})[n] = (ratios, over, where)
                if n:
                    # The same without offset (clipped at 0, as before VelociHub 1.2)
                    automatic, session.background_offset = session.background_offset, 0
                    clipped = []
                    for i, (name, pos, e) in enumerate(zip(names, parts["positions"], energy)):
                        img = np.asarray(session.process_image(os.path.join(session.path_in, name), i, loaded)[0],
                                         np.float64)
                        idx, x, y = isolated_particles(pos, flow, mask)
                        clipped.append(100 * particle_amplitudes(img, x, y) / (parts["peaks"][idx] * e))
                    session.background_offset = automatic
                    m["particle_kept_without_offset"] = float(np.median(np.concatenate(clipped)))
                # The pixels at 0 (the processed image cast to the depth of the raw one: its negative values clipped)
                m["clipped_pixels"] = float(100 * np.mean([np.mean(img[support] <= 0) for img in images]))
                mean = np.mean(images, 0)
                # The background left: the mean processed image less the particles (as much of them as kept)
                residual = np.mean([img - offset - m["particles_kept"] / 100 * e * part
                                    for img, part, e in zip(images, parts["particles"], energy)], 0)
                if mask is not None:
                    residual[mask] = np.nan
                # About its own level (the light of the particles averaged over the images, removed with the mean)
                residual -= np.nanmedian(residual)
                out.setdefault("residuals", {})[n] = residual
                m["residual_rms"] = float(np.sqrt(np.nanmean(residual**2)))
                for key, (kind, shape, geo) in feats.items():
                    sup = (shape > 0.05 * shape.max()) & ~mask
                    if not sup.any():
                        continue
                    # The feature kept: its projection in each image, relative to the particles kept (the export may
                    # scale the intensities)
                    kept = np.mean([_projection(img, shape, sup) / e for img, e in zip(images, energy)])
                    m[f"kept_{key}"] = 100 * kept / max(m["particles_kept"] / 100, 1e-9)
                    # Its peak left, in grey levels (at the mean laser energy)
                    m[f"left_{key}"] = m[f"kept_{key}"] / 100 * float(shape.max()) * float(energy.mean())
                    if kind in ("line", "circle"):
                        m[f"hough_{key}"] = float(hough_found(mean, kind, geo))
                shapes = [k for k, (kind, _, _) in feats.items() if kind in ("line", "circle")]
                m["lines_kept"] = max(abs(m[f"kept_{k}"]) for k in shapes)
                m["lines_min_kept"] = min(m[f"kept_{k}"] for k in shapes)
                m["glare_kept"] = abs(m["kept_glare"])
                m["hough_found"] = float(sum(m[f"hough_{k}"] for k in shapes))
                m["hough_total"] = float(len(shapes))
                out["modes"][n] = m
                out["means"][n] = mean
        return out
    finally:
        if own and not keep:
            shutil.rmtree(work, ignore_errors=True)
