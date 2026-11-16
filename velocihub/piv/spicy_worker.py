"""Regression of the displacement fields of the PIV/PTV with SPICY, run in its own process by VelociHub (regression.py):

    python spicy_worker.py job.json

The job (JSON file) gives the work folder (with raw.npz, the fields of the PIV/PTV), the settings of SPICY, the status
of the grid points (0: flow, 1: no particles, 2: masked), the width of the images, the key of the regression and
whether the data are particles (PTV). SPICY fits scattered data: after a PTV it fits the links of the particles of each
pair (links.npz of the work folder: each link at the middle of its two positions, with its displacement), never the
binned cells; the links in masked cells or outside the grid are left out. After a PIV it fits the valid vectors of the
flow at the grid points (the PIV gives nothing else). The fit is evaluated at all the grid points with flow; the other
ones keep the values of the PIV (the binned cells of a PTV). The maps are fitted by chunks saved in the folder spicy of
the work folder, so that a regression stopped (file STOP) or interrupted continues from the last chunk saved when it
is started again with the same key; at the end they are gathered in regressed.npz (u, v, info, residual: RMS of the
fit on the data fitted in px, n_rbf, n_points: the data fitted, failed). The messages are JSON lines on the standard
output: progress (phase spicy), log, done, stopped or error.
"""
import glob
import json
import os
import sys
import time
import traceback
import warnings

import numpy as np


def say(**message):
    print(json.dumps(message), flush=True)


# A map needs at least this many data (vectors or particles) to be fitted (otherwise it is kept as the PIV gave it)
MIN_VECTORS = 20


def fit(u, v, info, xpx, ypx, flow, width, s, particles=None):
    """One map fitted with SPICY: (u, v, info, residual, number of RBFs, number of data fitted), the vectors of the
    flow replaced by the regression evaluated there; None if there are too few data. particles: (x, y, u, v) of the
    links of the pair (PTV), fitted instead of the grid vectors; None: the valid vectors of the flow (PIV)."""
    from spicy_vki.spicy import Spicy

    if particles is None:
        ok = flow & (info == 1) & np.isfinite(u) & np.isfinite(v)
        px, py, pu, pv = xpx[ok], ypx[ok], u[ok].astype(np.float64), v[ok].astype(np.float64)
    else:
        px, py, pu, pv = (np.asarray(q, np.float64) for q in particles)
    if px.size < max(MIN_VECTORS, 3 * max(s["n_k"])):
        return None
    sp = Spicy([pu, pv], [px / width, py / width], basis=s["basis"], model="laminar", verbose=0)
    sp.collocation(list(s["n_k"]), method=s["method"], r_mM=[s["r_min"], s["r_max"]], eps_l=s["eps_l"])
    sp.vector_constraints(extra_RBF=False)
    sp.Assembly_Regression(n_hb=s["n_hb"], alpha_div=s["alpha_div"] or None)
    sp.Solve(K_cond=s["k_cond"])
    us, vs = sp.get_sol([xpx[flow] / width, ypx[flow] / width])
    out_u, out_v, out_info = u.copy(), v.copy(), info.copy()
    out_u[flow], out_v[flow], out_info[flow] = us, vs, 1
    fu, fv = sp.get_sol([px / width, py / width])  # At the data fitted
    residual = float(np.sqrt(np.mean((fu - pu) ** 2 + (fv - pv) ** 2)))
    return out_u, out_v, out_info, residual, int(sp.n_b), int(px.size)


class Links:
    """The links of the particles of the pairs (links.npz of the work folder), those fitted: in a cell of the grid
    (cells of the grid spacing centred on its points) that isn't masked."""

    def __init__(self, work, xpx, ypx, status):
        path = os.path.join(work, "links.npz")
        if not os.path.isfile(path):
            raise RuntimeError("the links of the particles (links.npz) are missing: run the PTV again")
        with np.load(path) as z:
            k, x, y, u, v = (z[q] for q in ("k", "x", "y", "u", "v"))
        x0, y0 = float(xpx[0, 0]), float(ypx[0, 0])
        spacing = float(xpx[0, 1] - x0) if xpx.shape[1] > 1 else float(ypx[1, 0] - y0)
        col = np.floor((x - (x0 - spacing / 2)) / spacing).astype(int)
        row = np.floor((y - (y0 - spacing / 2)) / spacing).astype(int)
        h, w = status.shape
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        keep = inside & np.isfinite(x) & np.isfinite(y) & np.isfinite(u) & np.isfinite(v)
        keep[keep] = status[row[keep], col[keep]] != 2
        self.k, self.x, self.y, self.u, self.v = k[keep], x[keep], y[keep], u[keep], v[keep]

    def of(self, k):
        """(x, y, u, v) of the links of the pair k fitted."""
        a, b = np.searchsorted(self.k, [k, k + 1])
        return self.x[a:b], self.y[a:b], self.u[a:b], self.v[a:b]


def main(job_file):
    with open(job_file) as f:
        job = json.load(f)
    work, s, width, chunk, key = job["work"], job["settings"], float(job["width"]), int(job["chunk"]), job["key"]
    status = np.asarray(job["status"])
    flow = status == 0
    stop_file = os.path.join(work, "STOP")
    folder = os.path.join(work, "spicy")
    os.makedirs(folder, exist_ok=True)
    # The chunks of another regression (other PIV or settings) are discarded
    for path in glob.glob(os.path.join(folder, "*.npz")):
        if not os.path.basename(path).startswith(key + "_"):
            os.remove(path)
    with np.load(os.path.join(work, "raw.npz")) as raw:
        xpx, ypx = raw["xpx"].astype(np.float64), raw["ypx"].astype(np.float64)
        U, V, INFO = raw["u"], raw["v"], raw["info"]
    links = Links(work, xpx, ypx, status) if job.get("particles") else None
    n = U.shape[0]
    t0, resumed, resumed_seconds, new = time.time(), 0, 0.0, 0
    warnings.filterwarnings("ignore")
    names = ("u", "v", "info", "residual", "n_rbf", "n_points")
    for start in range(0, n, chunk):
        ks = range(start, min(start + chunk, n))
        path = os.path.join(folder, f"{key}_{start:06d}.npz")
        if os.path.isfile(path):
            with np.load(path) as z:
                resumed += len(ks)
                resumed_seconds += float(z["seconds"])
            say(type="progress", phase="spicy", done=ks.stop, total=n, rate=None, resumed=True)
            continue
        out = {q: [] for q in names + ("failed",)}
        t_chunk = time.time()
        for k in ks:
            if os.path.exists(stop_file):
                say(type="stopped")
                return
            result = fit(U[k], V[k], INFO[k], xpx, ypx, flow, width, s, None if links is None else links.of(k))
            if result is None:
                result = (U[k], V[k], INFO[k], np.nan, 0, 0)
                out["failed"].append(k)
            for q, value in zip(names, result):
                out[q].append(value)
            new += 1
            say(type="progress", phase="spicy", done=k + 1, total=n, rate=(time.time() - t0) / new)
        # Written whole, then renamed: a chunk interrupted while written is never read (discarded at the next start)
        partial = os.path.join(folder, f"partial_{start:06d}.npz")
        np.savez(partial, u=np.array(out["u"], np.float32), v=np.array(out["v"], np.float32),
                 info=np.array(out["info"]), residual=np.array(out["residual"]), n_rbf=np.array(out["n_rbf"]),
                 n_points=np.array(out["n_points"]), failed=np.array(out["failed"], int),
                 seconds=time.time() - t_chunk)
        os.replace(partial, path)
    # All the chunks gathered
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(folder, key + "_*.npz")))]
    try:
        np.savez(os.path.join(work, "regressed.npz"), **{q: np.concatenate([p[q] for p in parts]) for q in names},
                 failed=np.concatenate([p["failed"] for p in parts]).astype(int))
    finally:
        for p in parts:
            p.close()
    say(type="done", seconds=round(time.time() - t0 + resumed_seconds, 1), resumed=resumed)


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as e:
        say(type="error", text=f"SPICY: {type(e).__name__}: {e}")
        traceback.print_exc()
