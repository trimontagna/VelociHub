"""PIV of image pairs with PaIRS-UniNa, run in its own process by VelociHub (pairs_engine.py):

    python pairs_worker.py job.json

The job (JSON file) gives the images, the pairs (indices of their two images), the settings, the number of pairs
per chunk and the key of the computation. The displacement fields are computed by chunks of pairs saved in the
folder of the job, so that a computation stopped (file STOP created in that folder) or interrupted continues
from the last chunk saved when it is started again with the same key; at the end they are gathered in raw.npz.
Each chunk keeps the seconds its pairs took: the message done gives the seconds of this computation and those of
the chunks resumed. The messages are JSON lines on the standard output: progress, field (the last pair computed,
saved in latest.npz at most every second, to be shown while the computation runs), log, done, stopped or error.

It only needs numpy, Pillow (and velocihub/greyscale.py, loaded from its file) and PaIRS-UniNa (pip install PaIRS-UniNa, Python 3.10 to 3.13): the Python
interpreter running it may differ from the one of the interface. From PIV_PAIRS_batch.py.
"""
import json
import os
import sys
import time
import traceback

import numpy as np


LATEST_EVERY = 1.0  # Seconds between two fields sent while the pairs are computed


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


def say(**message):
    print(json.dumps(message), flush=True)


def median(values):
    """Median of the finite values, None if there are none."""
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def save_latest(work, k, xpx, ypx, u, v, sn, inf):
    """The last pair computed, written whole (replaced at once) so that the interface never reads it half
    written. Returns the file, None if it can't be replaced now (being read)."""
    path, tmp = os.path.join(work, "latest.npz"), os.path.join(work, "latest_tmp.npz")
    try:
        np.savez(tmp, pair=k, xpx=xpx, ypx=ypx, u=u, v=v, sn=sn, info=inf)
        os.replace(tmp, path)
        return path
    except OSError:
        return None


def read(path):
    """Image as floats, read as every step of VelociHub reads it (greyscale.read_grey: the brightness of a colour
    image), tried again if a network or USB drive refuses the read for a moment."""
    return GREYSCALE.read_grey(path).astype(np.float64)


# The settings of the PaIRS process as PaIRS numbers them (pairs_engine.PAIRS_CODES; the job gives them: codes),
# those of its standard process for the jobs of the previous versions, without them
STANDARD = {"image_interpolation": 53, "velocity_interpolation": 52, "peak_interpolation": 0, "median_test": 2,
            "outlier_replacement": 1, "correlation_window": 0, "predictor_window": 0, "max_displacement": -4}
DEFAULTS = {"border": True, "median_kernel": 1, "median_threshold": 2.0, "median_epsilon": 0.5, "sn_threshold": 0.0,
            "cc_threshold": 0.0, "second_peak": True, "predictor_halfwidth": 0}


def make_piv(settings, h, w, codes=None):
    """The PaIRS process (its standard process, as in the PaIRS GUI, with the settings of the engine; codes: the
    choices of the settings as PaIRS numbers them), built from Python."""
    import PaIRS_UniNa.PaIRS_PIV as PaIRS_lib
    from PaIRS_UniNa.Process_Tab import PROpar

    s = {**DEFAULTS, **settings}
    codes = {**STANDARD, **(codes or {})}

    def code(key):
        return int(codes[key])

    wsize = list(s["windows"])
    sp = [max(1, int(round(size * (1 - s["overlap"] / 100)))) for size in wsize]
    PRO = PROpar(top=3, WSize=wsize, WSpac=sp)
    PRO.NIterazioni = s["final_iterations"]
    piv = PaIRS_lib.PIV()
    piv.DefaultValues()
    piv.Inp.FlagNumThreads = 0  # 0 = all CPU cores
    piv.SetVect([np.array(v).astype(np.intc) for v in PRO.Vect])
    piv.Inp.FlagBordo = 1 if s["border"] else 0
    piv.Inp.NIterazioni = PRO.NIterazioni
    piv.Inp.IntIniz = piv.Inp.IntFin = code("image_interpolation")
    piv.Inp.FlagInt = PRO.FlagInt
    piv.Inp.IntCorr = code("peak_interpolation") + 3
    piv.Inp.IntVel = code("velocity_interpolation")
    piv.Inp.FlagValid = code("median_test")
    piv.Inp.SemiDimValid, piv.Inp.SogliaMed = int(s["median_kernel"]), float(s["median_threshold"])
    piv.Inp.ErroreMed = float(s["median_epsilon"])
    piv.Inp.FlagAttivaValSN = (1 if s["sn_threshold"] > 0 else 0) | (2 if s["cc_threshold"] > 0 else 0)
    piv.Inp.SogliaSN = float(s["sn_threshold"]) if s["sn_threshold"] > 0 else PRO.SogliaSN
    piv.Inp.SogliaFcl = float(s["cc_threshold"]) if s["cc_threshold"] > 0 else PRO.SogliaCP
    piv.Inp.FlagValidNog = 0
    piv.Inp.SogliaMedia, piv.Inp.SogliaNumVet = PRO.SogliaMedia, PRO.SogliaNumVet
    # Minimum mean / std per window: the PaIRS defaults (2 and 3 grey levels) reject almost every window of dark
    # images; 0.01 / 0.05 only reject windows without any particle
    piv.Inp.SogliaNoise, piv.Inp.SogliaStd = s["min_mean"], s["min_std"]
    piv.Inp.FlagCorrHart = 0
    piv.Inp.FlagSecMax = 1 if s["second_peak"] else 0
    piv.Inp.FlagCorrezioneVel = code("outlier_replacement")
    piv.Inp.InfoSi, piv.Inp.InfoNo = 1, 0
    piv.Inp.FlagWindowing = code("correlation_window")
    # The windowing of each pass (Astarita 2007): max displacement, weighting windows of the predictor and of the
    # correlation map, half-width of the predictor window, direct correlation
    vw = [np.array(v, dtype=np.intc) for v in (PRO.vMaxDisp, PRO.vFlagCalcVel, PRO.vFlagWindowing,
                                                PRO.vSemiDimCalcVel, PRO.vDC)]
    vw[0][:] = code("max_displacement")
    vw[1][:] = code("predictor_window")
    vw[2][:] = code("correlation_window")
    vw[3][:] = int(s["predictor_halfwidth"])
    vw[4][:] = 0  # No direct correlation, so PaIRS returns the peak ratio
    piv.Inp.FlagDirectCorr = PRO.FlagDirectCorr
    piv.Inp.numInitIt = max(len(wsize), *[len(v) for v in vw])
    piv.SetWindowingVect(vw)
    piv._vw = vw
    piv.Inp.FlagSommaProd = PRO.FlagSommaProd
    piv.Inp.flagAdaptive = 0
    piv.Inp.MaxC, piv.Inp.MinC, piv.Inp.LarMin, piv.Inp.LarMax = PRO.MaxC, PRO.MinC, PRO.LarMin, PRO.LarMax
    piv.Inp.ItAtt = -1000
    piv.Inp.RisX = piv.Inp.RisY = 1.0
    piv.Inp.dt = 1e4  # With RisX = 1: u, v come out in px per image pair
    piv.Inp.FlagLog = 0
    piv.Inp.ImgH, piv.Inp.ImgW = h, w

    class NoStop(PaIRS_lib.PyFunOutPIV):
        def FunOut(self, a, b, o):
            return 0

    piv._cb = NoStop()  # Keep a reference, otherwise PaIRS aborts
    real = np.float64 if PaIRS_lib.SizeOfReal() == 8 else np.float32  # Image type PaIRS expects
    return piv, PaIRS_lib.GetPyFunction(piv._cb), real


def compute(job, work):
    images, pairs, s = job["images"], job["pairs"], job["settings"]
    chunk = job.get("chunk", 100)
    stop_file = os.path.join(work, "STOP")

    # The chunks saved by another computation (other images, pairs or settings) are discarded
    key_file = os.path.join(work, "key.json")
    if os.path.isfile(key_file):
        with open(key_file) as f:
            same = json.load(f) == job["key"]
        if not same:
            for name in os.listdir(work):
                if name.startswith(("chunk_", "background", "raw", "latest")):
                    os.remove(os.path.join(work, name))
    with open(key_file, "w") as f:
        json.dump(job["key"], f)

    h, w = read(images[pairs[0][0]]).shape
    background = 0.0
    if s["subtract_mean"]:
        bfile = os.path.join(work, "background.npy")
        if os.path.isfile(bfile):
            background = np.load(bfile)
        else:
            sel = range(0, len(images), max(1, len(images) // 200))
            background = np.zeros((h, w))
            for c, k in enumerate(sel):
                if os.path.exists(stop_file):
                    say(type="stopped", done=0)
                    return
                background += read(images[k])
                say(type="progress", phase="background", done=c + 1, total=len(sel))
            background /= len(sel)
            np.save(bfile, background)

    try:
        piv, fun, real = make_piv(s, h, w, job.get("codes"))
    except ImportError as e:
        say(type="error", text=f"PaIRS-UniNa can't be imported by {sys.executable}: {e}")
        return

    def img(k):
        return np.ascontiguousarray(np.clip(read(images[k]) - background, 0, None), dtype=real)

    keys = ("u", "v", "sn", "info")
    n = len(pairs)
    res, xpx, ypx, failed, last_error = None, None, None, [], ""
    t0, new, latest = time.time(), 0, 0.0
    # Pairs of the chunks saved by a computation stopped, and the seconds they took (None: unknown, chunks saved
    # without their time)
    resumed, resumed_seconds = 0, 0.0
    cache = {}
    for c in range((n + chunk - 1) // chunk):
        ks = range(c * chunk, min((c + 1) * chunk, n))
        fn = os.path.join(work, f"chunk_{c:04d}.npz")
        if os.path.isfile(fn):
            with np.load(fn) as z:
                arr = {k: z[k] for k in keys}
                xpx, ypx = z["xpx"], z["ypx"]
                fail_c = [int(f) for f in z["failed"]]
                seconds = float(z["seconds"]) if "seconds" in z.files else None
            failed += fail_c
            resumed += len(ks)
            resumed_seconds = None if seconds is None or resumed_seconds is None else resumed_seconds + seconds
            # The quality of the pairs saved, as if they were computed now
            say(type="progress", phase="pairs", done=ks.stop, total=n, rate=None, resumed=True, first=ks.start,
                valid=[None if k in fail_c else float(np.mean(arr["info"][i] == 1) * 100) for i, k in enumerate(ks)],
                peak_ratio=[None if k in fail_c else median(arr["sn"][i]) for i, k in enumerate(ks)])
        else:
            part, fail_c, started = {k: [] for k in keys}, [], time.time()
            for k in ks:
                if os.path.exists(stop_file):
                    say(type="stopped", done=ks.start)
                    return
                a, b = pairs[k]
                # The second image of a pair is often the first one of the next pair
                cache = {i: cache[i] if i in cache else img(i) for i in (a, b)}
                try:
                    piv.SetImg([cache[a], cache[b]])
                    piv.PIV_Run(fun)
                    x, y, u, v, sn, inf = (np.array(q) for q in (piv.x, piv.y, piv.u, piv.v, piv.sn, piv.Info))
                    if xpx is None:
                        xpx, ypx = x, y
                except RuntimeError as e:  # PaIRS gives up on a pair ("not enough correct vectors")
                    fail_c.append(k)
                    last_error = str(e)
                    u = v = sn = inf = None
                for q, F in zip(keys, (u, v, sn, inf)):
                    part[q].append(F)
                new += 1
                say(type="progress", phase="pairs", done=k + 1, total=n, rate=(time.time() - t0) / new, pair=k,
                    valid=None if inf is None else float(np.mean(inf == 1) * 100),
                    peak_ratio=None if sn is None else median(sn))
                if u is not None and time.time() - latest >= LATEST_EVERY:
                    path = save_latest(work, k, xpx, ypx, u, v, sn, inf)
                    if path is not None:
                        latest = time.time()
                        say(type="field", pair=k, file=path)
            if xpx is None:
                raise RuntimeError(f"PaIRS failed on every pair of the first {len(ks)} ({last_error}): "
                                   "check the images")
            arr = {q: np.stack([np.full(xpx.shape, -1 if q == "info" else np.nan, np.float32) if F is None
                                else np.asarray(F, np.float32) for F in part[q]]) for q in keys}
            np.savez(fn, xpx=xpx, ypx=ypx, failed=np.array(fail_c, int), seconds=time.time() - started, **arr)
            failed += fail_c
            if fail_c:
                say(type="log", text=f"{len(fail_c)} pair(s) failed in pairs {ks.start + 1}-{ks.stop}: {last_error}")
        if res is None:
            res = {q: np.full((n,) + xpx.shape, np.nan, np.float32) for q in keys}
        for q in keys:
            res[q][ks.start : ks.stop] = arr[q]
    np.savez(os.path.join(work, "raw.npz"), xpx=xpx, ypx=ypx, failed=np.array(failed, int), **res)
    say(type="done", failed=len(failed), seconds=round(time.time() - t0, 1), resumed=resumed,
        resumed_seconds=None if resumed_seconds is None else round(resumed_seconds, 1))


def main(job_file):
    work = os.path.dirname(os.path.abspath(job_file))
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
