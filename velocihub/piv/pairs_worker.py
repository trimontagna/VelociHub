"""PIV of image pairs with PaIRS-UniNa, run in its own process by VelociHub (pairs_engine.py):

    python pairs_worker.py job.json

The job (JSON file) gives the images, the pairs (indices of their two images), the settings, the number of pairs
per chunk and the key of the computation. The displacement fields are computed by chunks of pairs saved in the
folder of the job, so that a computation stopped (file STOP created in that folder) or interrupted continues
from the last chunk saved when it is started again with the same key; at the end they are gathered in raw.npz.
Each chunk keeps the seconds its pairs took: the message done gives the seconds of this computation and those of
the chunks resumed. The messages are JSON lines on the standard output: progress, field (the last pair computed,
saved in latest.npz at most every second, to be shown while the computation runs), log, done, stopped or error.

It only needs numpy, Pillow and PaIRS-UniNa (pip install PaIRS-UniNa, Python 3.10 to 3.13): the Python
interpreter running it may differ from the one of the interface. From PIV_PAIRS_batch.py.
"""
import json
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image


LATEST_EVERY = 1.0  # Seconds between two fields sent while the pairs are computed


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
    """Image as floats (the mean of the channels of a color image). A network or USB drive sometimes refuses a
    read for a moment: it is tried again."""
    for attempt in range(5):
        try:
            with Image.open(path) as im:
                a = np.asarray(im, dtype=np.float64)
            return a.mean(axis=2) if a.ndim == 3 else a
        except OSError:
            if attempt == 4:
                raise
            time.sleep(3)


def make_piv(settings, h, w):
    """PaIRS "standard" process (as in the PaIRS GUI), built from Python."""
    import PaIRS_UniNa.PaIRS_PIV as PaIRS_lib
    from PaIRS_UniNa.Process_Tab import PROpar

    wsize = list(settings["windows"])
    sp = [max(1, int(round(s * (1 - settings["overlap"] / 100)))) for s in wsize]
    PRO = PROpar(top=3, WSize=wsize, WSpac=sp)
    PRO.NIterazioni = settings["final_iterations"]
    piv = PaIRS_lib.PIV()
    piv.DefaultValues()
    piv.Inp.FlagNumThreads = 0  # 0 = all CPU cores
    piv.SetVect([np.array(v).astype(np.intc) for v in PRO.Vect])
    piv.Inp.FlagBordo = PRO.FlagBordo
    piv.Inp.NIterazioni = PRO.NIterazioni
    piv.Inp.IntIniz, piv.Inp.FlagInt, piv.Inp.IntFin = PRO.IntIniz, PRO.FlagInt, PRO.IntFin
    piv.Inp.IntCorr = PRO.IntCorr + 3
    piv.Inp.IntVel = PRO.IntVel
    piv.Inp.FlagValid = 1 if PRO.TypeMed == 0 else 2
    piv.Inp.SemiDimValid, piv.Inp.SogliaMed, piv.Inp.ErroreMed = PRO.KernMed, PRO.SogliaMed, PRO.ErroreMed
    piv.Inp.FlagAttivaValSN = (1 if PRO.FlagSNTest else 0) | (2 if PRO.FlagCPTest else 0)
    piv.Inp.SogliaSN, piv.Inp.SogliaFcl = PRO.SogliaSN, PRO.SogliaCP
    piv.Inp.FlagValidNog = 0
    piv.Inp.SogliaMedia, piv.Inp.SogliaNumVet = PRO.SogliaMedia, PRO.SogliaNumVet
    # Minimum mean / std per window: the PaIRS defaults (2 and 3 grey levels) reject almost every window of dark
    # images; 0.01 / 0.05 only reject windows without any particle
    piv.Inp.SogliaNoise, piv.Inp.SogliaStd = settings["min_mean"], settings["min_std"]
    piv.Inp.FlagCorrHart = 0
    piv.Inp.FlagSecMax = 1 if PRO.FlagSecMax else 0
    piv.Inp.FlagCorrezioneVel = PRO.FlagCorrezioneVel
    piv.Inp.InfoSi, piv.Inp.InfoNo = 1, 0
    piv.Inp.FlagWindowing = getattr(PRO, "FlagWindowing", 0)
    vw = [np.array(v, dtype=np.intc) for v in (PRO.vMaxDisp, PRO.vFlagCalcVel, PRO.vFlagWindowing,
                                                PRO.vSemiDimCalcVel, PRO.vDC)]
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
        piv, fun, real = make_piv(s, h, w)
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
