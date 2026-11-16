"""PTV of an image sequence with TracTrac (velocihub/piv/tractrac: Heyman 2019, MIT licence), run in its own process by VelociHub
(ptv_engine.py):

    python ptv_worker.py job.json

The job (JSON file) gives the images, the sequence tracked (indices of the images in time order: the pairs of the
PIV are its consecutive frames), the settings, the grid and the key of the computation. The tracking is the loop of
tractrac() for an image sequence, without its plots and files, calling the functions of TracTrac (detection, motion
model) as they are: the motion model is trained over vid_loop passes back and forth, and the tracks are those of the
last pass, forward. A particle tracked at a frame is linked to the frames before and after it (the links checked both
ways and by the motion model): each pair has the links of the particles tracked at its two frames (their
displacement, at the middle of their two positions), kept as they are (links.npz: what SPICY fits) and binned on the
grid (cells of the grid spacing centred on its points): mean displacement of the cell and number of particles. The
cells with fewer than min_particles particles, or rejected by the normalized median test, are invalid.

The state of the tracking (with the random numbers of the detection) is saved every CHECKPOINT frames and when
stopped (file STOP), so that a computation stopped or interrupted continues where it was, with the same results. At
the end: raw.npz (xpx, ypx, u, v, info, count, failed: the binned pairs, as the other engines give them), links.npz (k,
x, y, u, v: the links of each pair k, at the middle of their two positions, and their displacement, px) and tracks.npz
(the particles tracked). The messages are JSON lines on the standard output: progress (phase training,
then pairs), field, log, done, stopped or error.
"""
import glob
import json
import os
import pickle
import sys
import time
import traceback
import warnings

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")  # tractrac.py imports pyplot
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tractrac"))

CHECKPOINT = 100  # Frames tracked between two saves of the state
LATEST_EVERY = 1.0  # Seconds between two fields sent while the pairs are binned
TRAINING_EVERY = 0.5  # Seconds between two messages of progress while the motion model is trained
EPS_MEDIAN = 0.1  # Noise level of the normalized median test (px, Westerweel & Scarano 2005)
SEED = 0  # Of the random numbers of the detection (TracTrac breaks the ties between pixels with them)
# The choices of the settings, as TracTrac numbers them
PEAK_CONV = {"none": 0, "dog": 1, "log": 2, "shi-tomasi": 3}
PEAK_SUBPIX = {"quadratic": 0, "gaussian": 1}
ROWS = ("frame", "id", "x", "y", "u", "v", "ax", "ay", "err")
LINKS = ("k", "x", "y", "u", "v")
# Version of the files of the progress: the progress saved by another version is discarded
FORMAT = 2


def load_greyscale():
    """velocihub/greyscale.py, the reader of the images of every step of VelociHub, loaded from its file: this script
    runs under another interpreter, which may not have VelociHub."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "greyscale.py")
    spec = importlib.util.spec_from_file_location("velocihub_greyscale", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GREYSCALE = load_greyscale()


# The messages go to the standard output; what TracTrac prints (its warnings) is discarded (main)
OUT = sys.stdout


def say(**message):
    print(json.dumps(message), file=OUT, flush=True)


def read(path):
    """Image read as every step of VelociHub reads it (greyscale.read_grey: one channel, the brightness of a colour
    image), as TracTrac takes it: 8 or 16 bit (its imProc scales them to [0, 1)). A float image is taken in [0, 1] if
    its values are, else in grey levels, and converted to 16 bit (clipped to 0-65535)."""
    image = GREYSCALE.read_grey(path)
    if image.dtype in (np.uint8, np.uint16):
        return image
    image = np.nan_to_num(image.astype(np.float64))
    if image.size and image.min() >= 0 and image.max() <= 1:
        image = image * 65535
    return np.clip(np.round(image), 0, 65535).astype(np.uint16)


def thresholds(s, width, height):
    """The parameters of TracTrac (its list th of one dictionary): its defaults, the settings, no plot; as tractrac()
    reads them from its parameter file (numbers as floats) and sets them up."""
    import tractrac as tt

    th, _ = tt.set_default_parameter([{}], width, height)
    th[0].update(vid_loop=s["vid_loop"], BG=int(s["background"]), BGspeed=s["background_speed"],
                 noise=int(s["median_filter"]), noise_size=s["median_size"], peak_th_auto=int(s["auto_threshold"]),
                 peak_th=s["threshold"], peak_neigh=s["peak_neigh"], peak_conv=PEAK_CONV[s["kernel"]],
                 peak_conv_size=s["particle_size"], peak_subpix=PEAK_SUBPIX[s["subpixel"]], peak_minima=int(s["dark"]),
                 motion=int(s["motion"]), motion_av=s["motion_av"], motion_it=s["motion_it"], filter=int(s["filter"]),
                 filter_std=s["filter_std"], filter_time=s["filter_time"], plot=0)
    return tt.setup_parameters([{k: float(v) for k, v in th[0].items()}])


def median_test(u, v, ok, threshold):
    """Normalized median test (Westerweel & Scarano 2005) of the cells ok against their 8 neighbours: the cells whose
    normalized residual of u or v exceeds threshold are invalid, and those with fewer than 2 valid neighbours."""
    h, w = u.shape
    out = np.zeros(u.shape, bool)
    neighbours = []
    for F in (u, v):
        P = np.pad(np.where(ok, F, np.nan), 1, constant_values=np.nan)
        neighbours.append(np.stack([P[1 + i:h + 1 + i, 1 + j:w + 1 + j] for i in (-1, 0, 1) for j in (-1, 0, 1)
                                    if i or j]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for F, N in zip((u, v), neighbours):
            med = np.nanmedian(N, 0)
            spread = np.nanmedian(np.abs(N - med), 0)
            out |= np.abs(F - med) / (spread + EPS_MEDIAN) > threshold
    few = np.isfinite(neighbours[0]).sum(0) < 2
    return ok & ~((out & np.isfinite(out)) | few)


def bin_links(a, b, grid, s):
    """The links of a pair (positions at its first and second frames: a, b) binned on the grid: mean displacement u, v
    of each cell (NaN without particle), the valid cells and the number of particles."""
    x, y, spacing = np.asarray(grid["x"]), np.asarray(grid["y"]), grid["spacing"]
    h, w = len(y), len(x)
    d, m = b - a, (a + b) / 2
    col = np.floor((m[:, 0] - (x[0] - spacing / 2)) / spacing).astype(int)
    row = np.floor((m[:, 1] - (y[0] - spacing / 2)) / spacing).astype(int)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    cell = row[inside] * w + col[inside]
    count = np.bincount(cell, minlength=h * w).reshape(h, w)
    sums = [np.bincount(cell, q[inside], minlength=h * w).reshape(h, w) for q in (d[:, 0], d[:, 1])]
    with np.errstate(invalid="ignore", divide="ignore"):
        u, v = (np.where(count > 0, q / count, np.nan) for q in sums)
    ok = count >= s["min_particles"]
    if s["median_test"] > 0:
        ok = median_test(u, v, ok, s["median_test"])
    return u.astype(np.float32), v.astype(np.float32), ok.astype(np.int16), np.minimum(count, 32767).astype(np.int16)


def pair_links(k, a, b):
    """The links of the pair k (positions a, b at its two frames) as links.npz keeps them: at the middle of their two
    positions, with their displacement (px, x to the right and y down)."""
    a, b = np.asarray(a, np.float64).reshape(-1, 2), np.asarray(b, np.float64).reshape(-1, 2)
    m, d = (a + b) / 2, b - a
    return dict(k=np.full(len(a), k, np.int32), x=m[:, 0].astype(np.float32), y=m[:, 1].astype(np.float32),
                u=d[:, 0].astype(np.float32), v=d[:, 1].astype(np.float32))


def save_latest(work, k, grid, u, v, info):
    """The last pair binned, written whole (replaced at once) so that the interface never reads it half written.
    Returns the file, None if it can't be replaced now (being read)."""
    path, tmp = os.path.join(work, "latest.npz"), os.path.join(work, "latest_tmp.npz")
    xpx, ypx = np.meshgrid(np.asarray(grid["x"], float), np.asarray(grid["y"], float))
    try:
        np.savez(tmp, pair=k, xpx=xpx, ypx=ypx, u=u, v=v, info=info)
        os.replace(tmp, path)
        return path
    except OSError:
        return None


def replace_file(path, write):
    """Write a file whole (write(file object)), then rename it: a file interrupted while written is never read."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


class Tracking:
    """The loop of tractrac() over the frames (forward, then vid_loop passes back and forth), its state saved and
    restored (save, restore)."""

    def __init__(self, job, work):
        import tractrac as tt

        self.tt, self.work, self.job = tt, work, job
        self.files = [job["images"][k] for k in job["sequence"]]
        self.s, self.grid = job["settings"], job["grid"]
        self.n = len(self.files)
        self.state_file = os.path.join(work, "ptv_state.pkl")

    def start(self):
        """Initial state, as tractrac() before its loop: the first two frames, the background, the threshold."""
        tt, n = self.tt, self.n
        I0 = read(self.files[0])
        height, width = I0.shape[:2]
        th = thresholds(self.s, width, height)
        np.random.seed(SEED)
        I0f = tt.imProc(I0, th)
        I1f = tt.imProc(read(self.files[1]), th)
        self.mask = np.ones(I1f.shape, dtype="float32")
        h, w = np.shape(I0f)
        B = np.zeros((h, w), dtype=np.float32)
        if th[0]["BG"] == 1:
            nbgstart = np.minimum(100, n)
            for i in range(nbgstart):
                B = B + tt.imProc(read(self.files[i]), th) / nbgstart
        F0, F1 = I0f * self.mask - B, I1f * self.mask - B
        F0f, x, y, z = tt.feature_detection(F0, th)
        C0 = np.transpose(np.array([x, y]))
        ID0 = np.zeros(C0.shape[0]) + np.nan
        F1f, x, y, z = tt.feature_detection(F1, th)
        C1 = np.transpose(np.array([x, y]))
        # The threshold of the detection in the loop (tractrac() replaces it at each frame by this one, F1f never
        # changing)
        if th[0]["peak_th_auto"] == 1 and not th[0]["peak_conv"] == 3:
            th[0]["peak_th"] = np.mean(F1f) + 0.5 * np.std(F1f)
        self.th = th
        ns01, i1001 = [np.array([]) for _ in range(2)]
        if (len(C0) > 0) & (len(C1) > 0):
            from scipy import spatial as scp

            ns01 = scp.cKDTree(C0).query(C1)
            ns10 = scp.cKDTree(C1).query(C0)
            i1001 = ns10[1][ns01[1]] == np.arange(C1.shape[0])
        empty = np.array([]).reshape((-1, 2))
        self.v = dict(C0=C0, C1=C1, ID0=ID0, X1=empty, um=empty, errU=np.array([]), idgood=[], Xmot=empty,
                      Umot=empty, errmot=np.array([]), errU_th=2.0, ns01=ns01, i1001=i1001, B=B, id_traj=0)
        self.i = 2
        # Links of the pair after the last frame tracked (forward) and the identities of its particles; frames and
        # pairs done (saved by chunks); seconds of the computation
        self.pending, self.prev_ids = (empty, empty), set()
        self.chunks, self.pairs_done, self.seconds = 0, 0, 0.0
        self.detected = np.full(n, -1, np.int32)
        self.tracked = np.full(n, -1, np.int32)
        self.rows, self.binned, self.links = [], [], []

    def sequence(self):
        """Frames read by the loop (N) and their time steps (dt), as tractrac() builds them."""
        n, dt = self.n, np.ones(self.n - 1)
        N, DT = np.arange(n), dt
        for k in range(int(self.th[0]["vid_loop"])):
            if np.mod(k, 2) == 0:
                N, DT = np.hstack((N, np.arange(n - 2, -1, -1))), np.hstack((DT, -dt[::-1]))
            else:
                N, DT = np.hstack((N, np.arange(1, n))), np.hstack((DT, dt[::1]))
        return N, DT

    def save(self):
        """Save the frames tracked since the last save (a chunk) and the state."""
        chunk = os.path.join(self.work, f"ptv_chunk_{self.chunks:05d}.npz")
        rows = {q: np.concatenate([r[q] for r in self.rows]) if self.rows else np.array([]) for q in ROWS}
        binned = {q: np.array([b[q] for b in self.binned]) for q in ("k", "u", "v", "info", "count")}
        links = {q: np.concatenate([lk[q] for lk in self.links]) if self.links else np.array([]) for q in LINKS}
        replace_file(chunk, lambda f: np.savez(f, **{f"row_{q}": a for q, a in rows.items()},
                                               **{f"pair_{q}": a for q, a in binned.items()},
                                               **{f"link_{q}": a for q, a in links.items()}))
        self.chunks += 1
        self.rows, self.binned, self.links = [], [], []
        state = dict(i=self.i, v=self.v, th=self.th, mask_shape=self.mask.shape, pending=self.pending,
                     prev_ids=self.prev_ids, chunks=self.chunks, pairs_done=self.pairs_done, seconds=self.seconds,
                     detected=self.detected, tracked=self.tracked, random=np.random.get_state())
        replace_file(self.state_file, lambda f: pickle.dump(state, f))

    def restore(self):
        """The state saved (False if there is none): the chunks saved after it are discarded."""
        if not os.path.isfile(self.state_file):
            return False
        with open(self.state_file, "rb") as f:
            state = pickle.load(f)
        self.i, self.v, self.th, self.pending = state["i"], state["v"], state["th"], state["pending"]
        self.mask = np.ones(state["mask_shape"], dtype="float32")
        self.prev_ids, self.chunks, self.pairs_done = state["prev_ids"], state["chunks"], state["pairs_done"]
        self.seconds, self.detected, self.tracked = state["seconds"], state["detected"], state["tracked"]
        np.random.set_state(state["random"])
        for path in glob.glob(os.path.join(self.work, "ptv_chunk_*.npz")):
            if int(os.path.basename(path)[10:15]) >= self.chunks:
                os.remove(path)
        self.rows, self.binned, self.links = [], [], []
        return True

    def pair(self, k, a, b):
        """Keep and bin the links of the pair k (positions a, b at its two frames)."""
        u, v, info, count = bin_links(a, b, self.grid, self.s)
        self.binned.append(dict(k=k, u=u, v=v, info=info, count=count))
        self.links.append(pair_links(k, a, b))
        self.pairs_done = k + 1
        return u, v, info

    def step(self, i, N, dt):
        """One iteration of the loop of tractrac() (its frame N[i]). Returns the particles tracked at the middle
        frame (in the last pass: their links before and after it), else None."""
        from scipy import spatial as scp

        tt, th, n = self.tt, self.th, self.n
        V = self.v
        C0, C1, ID0, X1, um, errU, idgood = (V[q] for q in ("C0", "C1", "ID0", "X1", "um", "errU", "idgood"))
        Xmot, Umot, errmot, errU_th, ns01, i1001, B = (V[q] for q in ("Xmot", "Umot", "errmot", "errU_th", "ns01",
                                                                      "i1001", "B"))
        id_traj = V["id_traj"]
        I2f = tt.imProc(read(self.files[N[i]]), th)
        F2 = I2f * self.mask - B
        F2f, x, y, z = tt.feature_detection(F2, th)
        C2 = np.transpose(np.array([x, y]))
        X0 = X2 = A = np.array([]).reshape(-1, 2)
        ID, i_all = [], []
        it = 0
        while it <= th[0]["motion_it"]:
            # Prediction from the motion model
            if (th[0]["motion"] == 1) & (len(idgood) + Xmot.shape[0] > 1) & (len(C1) > 0):
                if len(X1) == 0:
                    [Umotion, errU_filt, Xmot_temp, Umot_temp, errmot_temp] = tt.Propagate_MotionModel_KdTree(
                        C1, np.array([]).reshape(-1, 2), np.array([]).reshape(-1, 2), np.array([]), Xmot, Umot, errmot,
                        th)
                else:
                    [Umotion, errU_filt, Xmot_temp, Umot_temp, errmot_temp] = tt.Propagate_MotionModel_KdTree(
                        C1, X1[idgood, :], um[idgood, :], errU[idgood], Xmot, Umot, errmot, th)
            else:
                Umotion = np.zeros(C1.shape)
                errU_filt = np.zeros(C1.shape[0])
                # tractrac() keeps the motion model of the previous frame then
                Xmot_temp, Umot_temp, errmot_temp = V.get("Xmot_temp", Xmot), V.get("Umot_temp", Umot), V.get(
                    "errmot_temp", errmot)
            if len(C1) > 0:
                t1m = scp.cKDTree(C1 + Umotion * dt[i - 1])
            t2 = scp.cKDTree(C2) if len(C2) > 0 else None
            if (len(C1) > 0) & (len(C2) > 0):
                ns21 = t2.query(C1 + Umotion * dt[i - 1])
                ns12 = t1m.query(C2)
                i1221 = ns12[1][ns21[1]] == np.arange(C1.shape[0])
                i2112 = ns21[1][ns12[1]] == np.arange(C2.shape[0])
            else:
                i1221, i2112, ns12 = [], [], []
            if (len(C0) > 0) & (len(C1) > 0) & (len(C2) > 0):
                i_all = i1001 & i1221
                X0 = C0[ns01[1][i_all], :]
                X1 = C1[i_all, :]
                X2 = C2[ns21[1][i_all], :]
                ID = ID0[ns01[1][i_all]]
                U0 = (X1 - X0) / dt[i - 2]
                U1 = (X2 - X1) / dt[i - 1]
                um = (U0 + U1) / 2.
                if (dt[i - 1] + dt[i - 2]) == 0:
                    A = np.zeros(U0.shape) + np.nan
                else:
                    A = (U1 - U0) / (dt[i - 1] + dt[i - 2]) * 2.
                with np.errstate(divide="ignore", invalid="ignore"):
                    errU = np.maximum(-10., np.log10(np.amax((np.abs((Umotion[i_all, :] - um) * dt[i - 1])), 1)))
                errU[np.isnan(errU)] = 0
                ISgood = errU - errU_filt[i_all] < errU_th
                errU_th = (th[0]["filter_time"] * errU_th + th[0]["filter_std"] * np.std(errU)) / (
                    th[0]["filter_time"] + 1.)
            else:
                i_all, ID, errU, ISgood = [[] for _ in range(4)]
                X0, X1, X2, um, A = [np.array([]).reshape(-1, 2) for _ in range(5)]
                errU_th = 2
            if th[0]["filter"] == 1:
                idgood = np.where(np.asarray(ISgood) == 1)[0]
            elif len(X1) > 0:
                idgood = np.arange(0, len(X1[:, 1]))
            else:
                idgood = []
            if np.isnan(errU_th):
                errU_th = 2
            it += 1
        # New identities of the new tracks
        if len(ID) > 0:
            idnan = np.where(np.isnan(ID[idgood]))[0]
            if len(idnan) > 0:
                ID[idgood[idnan]] = np.arange(0, len(idnan)) + id_traj
                id_traj = id_traj + len(idnan)
        Xmot, Umot, errmot = Xmot_temp, Umot_temp, errmot_temp
        if len(C1) > 0:
            ID0 = np.zeros(C1.shape[0]) + np.nan
            ID0[i_all] = ID
        # The last pass starts: identities from 0
        if i == len(N) - n:
            ID0[i_all] = np.arange(int(sum(i_all)))
            id_traj = sum(i_all)
            A = A + np.nan
        out = None
        if i >= len(N) - n + 2:
            good = np.asarray(idgood, int)
            frame = int(N[i] - np.sign(dt[i - 1]))
            out = (frame, len(C1), dict(frame=np.full(len(good), frame, np.int32), id=ID[good].astype(np.int64)
                                        if len(good) else np.array([], np.int64),
                                        x=X1[good, 0], y=X1[good, 1], u=um[good, 0], v=um[good, 1], ax=A[good, 0],
                                        ay=A[good, 1], err=errU[good] if len(good) else np.array([])),
                   X0[good], X2[good])
        # Background updated
        if th[0]["BG"]:
            r2 = B - I2f
            r = (-np.float32(r2 > th[0]["BGspeed"]) * th[0]["BGspeed"] + np.float32(r2 < -th[0]["BGspeed"])
                 * th[0]["BGspeed"] - np.float32(np.abs(r2) <= th[0]["BGspeed"]) * r2)
            B = np.minimum(np.maximum(B + r, 0), 1)
        V.update(C0=C1, C1=C2, ID0=ID0, X1=X1, um=um, errU=errU, idgood=idgood, Xmot=Xmot, Umot=Umot, errmot=errmot,
                 errU_th=errU_th, ns01=ns12, i1001=i2112, B=B, id_traj=id_traj, Xmot_temp=Xmot_temp,
                 Umot_temp=Umot_temp, errmot_temp=errmot_temp)
        return out


def compute(job, work):
    stop_file = os.path.join(work, "STOP")
    # The progress saved by another computation (other images, sequence or settings) is discarded
    key_file, key = os.path.join(work, "key.json"), {"key": job["key"], "format": FORMAT}
    if os.path.isfile(key_file):
        with open(key_file) as f:
            same = json.load(f) == key
        if not same:
            for name in os.listdir(work):
                if name.startswith(("ptv_", "raw", "tracks", "latest", "links")):
                    os.remove(os.path.join(work, name))
    with open(key_file, "w") as f:
        json.dump(key, f)
    tracking = Tracking(job, work)
    n = tracking.n
    n_pairs = n - 1
    if n < 3:
        raise RuntimeError(f"TracTrac needs at least 3 frames, not {n}")
    if not tracking.restore():
        tracking.start()
    resumed, resumed_seconds = tracking.pairs_done, tracking.seconds
    N, dt = tracking.sequence()
    total = len(N) - 2
    passes = 1 + int(tracking.th[0]["vid_loop"])
    if resumed:
        say(type="progress", phase="pairs", done=resumed, total=n_pairs, rate=None, resumed=True, first=0,
            particles=pair_particles(work, tracking.chunks))
    t0, started, new, said, latest = time.time(), time.time(), 0, 0.0, 0.0
    for i in range(tracking.i, len(N)):
        if os.path.exists(stop_file):
            tracking.i, tracking.seconds = i, tracking.seconds + time.time() - started
            tracking.save()
            say(type="stopped", done=tracking.pairs_done)
            return
        out = tracking.step(i, N, dt)
        new += 1
        rate = (time.time() - t0) / new
        if out is None:
            if time.time() - said >= TRAINING_EVERY:
                said = time.time()
                say(type="progress", phase="training", done=i - 1, total=total, rate=rate,
                    passes=passes, **{"pass": 1 if i < n else 2 + (i - n) // (n - 1)})
        else:
            frame, detected, rows, before, after = out
            tracking.detected[frame], tracking.tracked[frame] = detected, len(rows["x"])
            tracking.rows.append(rows)
            x1 = np.stack([rows["x"], rows["y"]], 1)
            ids = set(rows["id"].tolist())
            # The pair before this frame: the links of the particles tracked at its first frame, and those of the
            # particles tracked from this one only (backward)
            new_ids = np.array([k not in tracking.prev_ids for k in rows["id"].tolist()], bool)
            a = np.concatenate([tracking.pending[0], before[new_ids]])
            b = np.concatenate([tracking.pending[1], x1[new_ids]])
            k = frame - 1
            u, v, info = tracking.pair(k, a, b)
            tracking.pending, tracking.prev_ids = (x1, after), ids
            if frame == n - 2:
                # The last frame: the pair after it too
                say_pair(tracking, k, n_pairs, rate)
                u, v, info = tracking.pair(frame, x1, after)
                k = frame
            say_pair(tracking, k, n_pairs, rate)
            if time.time() - latest >= LATEST_EVERY:
                path = save_latest(work, k, tracking.grid, u, v, info)
                if path is not None:
                    latest = time.time()
                    say(type="field", pair=k, file=path)
        if (i + 1) % CHECKPOINT == 0 and i + 1 < len(N):
            tracking.i, tracking.seconds = i + 1, tracking.seconds + time.time() - started
            started = time.time()
            tracking.save()
    tracking.i, tracking.seconds = len(N), tracking.seconds + time.time() - started
    tracking.save()
    gather(tracking, work)
    say(type="done", failed=0, seconds=round(tracking.seconds - resumed_seconds, 1), resumed=resumed,
        resumed_seconds=round(resumed_seconds, 1))


def say_pair(tracking, k, n_pairs, rate):
    """Progress: the pair k binned and its particles tracked (links binned on the grid; not its valid cells: not a
    reliable quality of a PTV)."""
    binned = tracking.binned[-1]
    say(type="progress", phase="pairs", done=k + 1, total=n_pairs, rate=rate, pair=k,
        particles=int(binned["count"].sum()))


def pair_particles(work, chunks):
    """Particles tracked of the pairs of the chunks saved."""
    particles = []
    for c in range(chunks):
        with np.load(os.path.join(work, f"ptv_chunk_{c:05d}.npz")) as z:
            particles += [int(count.sum()) for count in z["pair_count"]]
    return particles


def gather(tracking, work):
    """The chunks gathered: raw.npz (the binned pairs), links.npz (the links of the pairs) and tracks.npz (the
    particles tracked)."""
    grid, n_pairs = tracking.grid, tracking.n - 1
    xpx, ypx = np.meshgrid(np.asarray(grid["x"], float), np.asarray(grid["y"], float))
    shape = (n_pairs,) + xpx.shape
    pairs = dict(u=np.full(shape, np.nan, np.float32), v=np.full(shape, np.nan, np.float32),
                 info=np.zeros(shape, np.int16), count=np.zeros(shape, np.int16))
    rows, links = {q: [] for q in ROWS}, {q: [] for q in LINKS}
    for c in range(tracking.chunks):
        with np.load(os.path.join(work, f"ptv_chunk_{c:05d}.npz")) as z:
            for k_index, k in enumerate(z["pair_k"].tolist() if z["pair_k"].size else []):
                for q in pairs:
                    pairs[q][k] = z[f"pair_{q}"][k_index]
            for q in ROWS:
                if z[f"row_{q}"].size:
                    rows[q].append(z[f"row_{q}"])
            for q in LINKS:
                if z[f"link_{q}"].size:
                    links[q].append(z[f"link_{q}"])
    dtypes = dict(frame=np.int32, id=np.int64)
    rows = {q: np.concatenate(r).astype(dtypes.get(q, np.float32)) if r else np.array([], dtypes.get(q, np.float32))
            for q, r in rows.items()}
    replace_file(os.path.join(work, "raw.npz"), lambda f: np.savez(f, xpx=xpx, ypx=ypx, failed=np.array([], int),
                                                                   **pairs))
    # Links sorted by pair (stable: in the order tracked within a pair)
    links = {q: np.concatenate(a) if a else np.array([], np.int32 if q == "k" else np.float32)
             for q, a in links.items()}
    order = np.argsort(links["k"], kind="stable")
    replace_file(os.path.join(work, "links.npz"), lambda f: np.savez(f, **{q: a[order] for q, a in links.items()}))
    th = {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in tracking.th[0].items()}
    replace_file(os.path.join(work, "tracks.npz"), lambda f: np.savez(
        f, **rows, detected=tracking.detected, tracked=tracking.tracked,
        sequence=np.asarray(tracking.job["sequence"], np.int32), tractrac=json.dumps(th)))


def main(job_file):
    work = os.path.dirname(os.path.abspath(job_file))
    sys.stdout = open(os.devnull, "w")
    try:
        with open(job_file) as f:
            job = json.load(f)
        compute(job, work)
    except Exception as e:  # Reported to the interface, with the details in error_log.txt
        with open(os.path.join(work, "error_log.txt"), "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M ") + traceback.format_exc() + "\n")
        say(type="error", text=f"{type(e).__name__}: {e} (details in {os.path.join(work, 'error_log.txt')})")


if __name__ == "__main__":
    main(sys.argv[1])
