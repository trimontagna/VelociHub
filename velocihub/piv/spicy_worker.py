"""Regression of the displacement fields of the PIV with SPICY, run in its own process by VelociHub (regression.py):

    python spicy_worker.py job.json

The job (JSON file) gives the work folder (with raw.npz, the fields of the PIV), the settings of SPICY, the status of
the grid points (0: flow), the width of the images and the key of the regression. Each map is fitted on its valid
vectors in the flow (for a PTV, at the mean position of the particles of each cell: cx, cy of raw.npz) and evaluated
at all the grid points with flow; the other ones keep the values of the PIV. The
maps are fitted by chunks saved in the folder spicy of the work folder, so that a regression stopped (file STOP) or
interrupted continues from the last chunk saved when it is started again with the same key; at the end they are
gathered in regressed.npz (u, v, info, residual: RMS of the fit on the valid vectors in px, n_rbf, failed). The
messages are JSON lines on the standard output: progress (phase spicy), log, done, stopped or error.
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


# A map needs at least this many valid vectors to be fitted (otherwise it is kept as the PIV gave it)
MIN_VECTORS = 20


def fit(u, v, info, xpx, ypx, flow, width, s, cx=None, cy=None):
    """One map fitted with SPICY: (u, v, info, residual, number of RBFs), the vectors of the flow replaced by the
    regression; None if it has too few valid vectors. cx, cy: where the vectors are (PTV: the mean position of the
    particles of each cell; None: at the grid points)."""
    from spicy_vki.spicy import Spicy

    ok = flow & (info == 1) & np.isfinite(u) & np.isfinite(v)
    if cx is not None:
        ok &= np.isfinite(cx) & np.isfinite(cy)
    if ok.sum() < max(MIN_VECTORS, 3 * max(s["n_k"])):
        return None
    px, py = (xpx, ypx) if cx is None else (cx.astype(np.float64), cy.astype(np.float64))
    sp = Spicy([u[ok].astype(np.float64), v[ok].astype(np.float64)], [px[ok] / width, py[ok] / width],
               basis=s["basis"], model="laminar", verbose=0)
    sp.collocation(list(s["n_k"]), method=s["method"], r_mM=[s["r_min"], s["r_max"]], eps_l=s["eps_l"])
    sp.vector_constraints(extra_RBF=False)
    sp.Assembly_Regression(n_hb=s["n_hb"], alpha_div=s["alpha_div"] or None)
    sp.Solve(K_cond=s["k_cond"])
    us, vs = sp.get_sol([xpx[flow] / width, ypx[flow] / width])
    out_u, out_v, out_info = u.copy(), v.copy(), info.copy()
    out_u[flow], out_v[flow], out_info[flow] = us, vs, 1
    if cx is None:
        residual = float(np.sqrt(np.mean((out_u[ok] - u[ok]) ** 2 + (out_v[ok] - v[ok]) ** 2)))
    else:
        # At the points fitted
        fu, fv = sp.get_sol([px[ok] / width, py[ok] / width])
        residual = float(np.sqrt(np.mean((fu - u[ok]) ** 2 + (fv - v[ok]) ** 2)))
    return out_u, out_v, out_info, residual, int(sp.n_b)


def main(job_file):
    with open(job_file) as f:
        job = json.load(f)
    work, s, width, chunk, key = job["work"], job["settings"], float(job["width"]), int(job["chunk"]), job["key"]
    flow = np.asarray(job["status"]) == 0
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
        CX, CY = (raw["cx"], raw["cy"]) if "cx" in raw.files else (None, None)
    n = U.shape[0]
    t0, resumed, resumed_seconds, new = time.time(), 0, 0.0, 0
    warnings.filterwarnings("ignore")
    for start in range(0, n, chunk):
        ks = range(start, min(start + chunk, n))
        path = os.path.join(folder, f"{key}_{start:06d}.npz")
        if os.path.isfile(path):
            with np.load(path) as z:
                resumed += len(ks)
                resumed_seconds += float(z["seconds"])
            say(type="progress", phase="spicy", done=ks.stop, total=n, rate=None, resumed=True)
            continue
        out = {q: [] for q in ("u", "v", "info", "residual", "n_rbf", "failed")}
        t_chunk = time.time()
        for k in ks:
            if os.path.exists(stop_file):
                say(type="stopped")
                return
            result = fit(U[k], V[k], INFO[k], xpx, ypx, flow, width, s, *((None, None) if CX is None
                                                                           else (CX[k], CY[k])))
            if result is None:
                result = (U[k], V[k], INFO[k], np.nan, 0)
                out["failed"].append(k)
            for q, value in zip(("u", "v", "info", "residual", "n_rbf"), result):
                out[q].append(value)
            new += 1
            say(type="progress", phase="spicy", done=k + 1, total=n, rate=(time.time() - t0) / new)
        # Written whole, then renamed: a chunk interrupted while written is never read (discarded at the next start)
        partial = os.path.join(folder, f"partial_{start:06d}.npz")
        np.savez(partial, u=np.array(out["u"], np.float32), v=np.array(out["v"], np.float32),
                 info=np.array(out["info"]), residual=np.array(out["residual"]), n_rbf=np.array(out["n_rbf"]),
                 failed=np.array(out["failed"], int), seconds=time.time() - t_chunk)
        os.replace(partial, path)
    # All the chunks gathered
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(folder, key + "_*.npz")))]
    try:
        np.savez(os.path.join(work, "regressed.npz"),
                 **{q: np.concatenate([p[q] for p in parts]) for q in ("u", "v", "info", "residual", "n_rbf")},
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
