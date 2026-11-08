"""Integral scales of the turbulence from the vector fields: the autocorrelation of the velocity fluctuations in time
(at each grid point, over successive maps) and in space (along x or y, over the maps), integrated up to their first
zero crossing.

    T_u(x, y) = ∫_0^τ0 ρ_uu(τ) dτ          ρ_uu(τ) = <u'(t) u'(t + τ)> / σ_u²
    L(x, y)   = ∫_0^r0 ρ_uu(r) dr          ρ_uu(r) = <u'(x) u'(x + r)> / (σ_u(x) σ_u(x + r))

τ0, r0: the first zero crossing of ρ (interpolated between two lags). The gaps of the fields (invalid vectors,
outliers: NaN) are left out of the sums, each lag averaged over the pairs of values it has. A scale is undefined (NaN)
when ρ doesn't cross zero within the lags available (a scale longer than the record, or than the field), when it
crosses zero before MIN_LAGS lags (a scale not resolved by the sampling) or when the lags before the crossing aren't
known (too few pairs). The spatial scale is the mean of the two directions (+r and -r) where both are defined, else
the one defined.
"""

import numpy as np

# A scale is resolved when its autocorrelation stays positive over at least this many lags
MIN_LAGS = 2
# A lag of the autocorrelation needs at least this fraction of the pairs of its first lag (else unknown)
MIN_PAIRS = 0.25
# The lags of the temporal autocorrelation: at most this fraction of the maps (the longer lags are too noisy)
MAX_TIME_LAGS = 0.5


def fluctuations(U):
    """The fluctuations u' = U - mean (over the first axis, the maps; NaN kept) and the mask of the values known."""
    known = np.isfinite(U)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(known, U, np.nan), 0) if known.any() else np.zeros(U.shape[1:])
    return np.where(known, U - mean, 0.0), known


def integral_to_zero(rho, step, min_lags=MIN_LAGS):
    """The integral of autocorrelations (rho: lags first, rho[0] = 1) from lag 0 to their first zero crossing
    (trapezoids, the crossing interpolated linearly), times the step between the lags; NaN where they don't cross zero,
    cross it before min_lags lags, or have unknown lags (NaN) before the crossing."""
    rho = np.asarray(rho, np.float64)
    lags = rho.shape[0]
    out = np.full(rho.shape[1:], np.nan)
    if lags < 2:
        return out
    stop = (rho <= 0) | ~np.isfinite(rho)
    stop[0] = False
    first = np.argmax(stop, 0)
    found = stop.any(0)
    idx = np.where(found, first, 1)
    r2 = np.take_along_axis(rho, idx[None], 0)[0]
    r1 = np.take_along_axis(rho, (idx - 1)[None], 0)[0]
    crossed = found & np.isfinite(r2) & (r2 <= 0) & (idx >= min_lags)
    cum = np.concatenate([np.zeros((1,) + rho.shape[1:]), np.cumsum((rho[1:] + rho[:-1]) / 2, 0)])
    before = np.take_along_axis(cum, (idx - 1)[None], 0)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = r1 / (r1 - r2)
        total = (before + frac * r1 / 2) * step
    out[crossed] = total[crossed]
    return out


def time_autocorrelation(U, max_lag=None):
    """The temporal autocorrelation ρ(τ) of the fluctuations of U (maps, rows, columns) at each grid point, for lags
    of 0 to max_lag maps (MAX_TIME_LAGS of the maps by default): (lags, rows, columns), NaN where a lag has too few
    pairs of values (MIN_PAIRS) or the point no fluctuation. FFT of the fluctuations (the gaps as zeros) and of the
    mask of the values known, row by row."""
    n = U.shape[0]
    max_lag = int(n * MAX_TIME_LAGS) if max_lag is None else min(int(max_lag), n - 1)
    Z, known = fluctuations(U)
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 2))))
    rho = np.full((max_lag + 1,) + U.shape[1:], np.nan)
    for i in range(U.shape[1]):
        F = np.fft.rfft(Z[:, i], nfft, 0)
        M = np.fft.rfft(known[:, i].astype(np.float64), nfft, 0)
        num = np.fft.irfft(F * np.conj(F), nfft, 0)[:max_lag + 1]
        count = np.rint(np.fft.irfft(M * np.conj(M), nfft, 0)[:max_lag + 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.where(count >= np.maximum(MIN_PAIRS * count[0], 2), num / count, np.nan)
            rho[:, i] = R / np.where(R[0] > 0, R[0], np.nan)
    return rho


def space_autocorrelation(U, axis):
    """The spatial autocorrelation coefficient ρ(r) of the fluctuations of U (maps, rows, columns) along an axis of
    the grid (2: x, the columns; 1: y, the rows), from each grid point towards +r (the columns or rows after it):
    (lags, rows, columns), NaN beyond the field and where a lag has too few pairs (MIN_PAIRS). Averaged over the maps
    with both values known, normalised by the standard deviations of both points."""
    Z, known = fluctuations(U)
    Z, known = np.moveaxis(Z, axis, -1), np.moveaxis(known, axis, -1)
    size = Z.shape[-1]
    k = known.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.sqrt((Z ** 2).sum(0) / np.maximum(k.sum(0) - 1, 1))
        sigma[k.sum(0) < 3] = np.nan
    rho = np.full((size,) + Z.shape[1:], np.nan)
    pairs0 = k.sum(0)
    for r in range(size):
        a, b = slice(0, size - r), slice(r, size)
        num = (Z[..., a] * Z[..., b]).sum(0)
        count = (k[..., a] * k[..., b]).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            value = num / np.maximum(count - 1, 1) / (sigma[..., a] * sigma[..., b])
        value[count < np.maximum(MIN_PAIRS * pairs0[..., a], 3)] = np.nan
        rho[r][..., a] = value
    return np.moveaxis(rho, -1, axis)


def integral_length(U, spacing, axis):
    """The integral length scale of the fluctuations of U (maps, rows, columns) along an axis of the grid (2: x, 1: y)
    at each grid point (in the unit of spacing, the step of the grid): the integral of the autocorrelation towards +r
    and towards -r (integral_to_zero), their mean where both are defined, else the one defined."""
    rho = space_autocorrelation(U, axis)
    plus = integral_to_zero(rho, spacing)
    # Towards -r: the autocorrelation of the point r before (the same product), lag by lag
    minus_rho = np.full_like(rho, np.nan)
    size = rho.shape[axis]
    for r in range(size):
        src = [slice(None)] * rho[r].ndim
        dst = [slice(None)] * rho[r].ndim
        src[axis - 1], dst[axis - 1] = slice(0, size - r), slice(r, size)
        minus_rho[r][tuple(dst)] = rho[r][tuple(src)]
    minus = integral_to_zero(minus_rho, spacing)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(plus) & np.isfinite(minus), (plus + minus) / 2, np.where(np.isfinite(plus), plus,
                                                                                               minus))


def planar_dissipation(U, V, x, y, nu, chunk=200):
    """The dissipation rate of the turbulent kinetic energy at each grid point from the in-plane gradients of the
    fluctuations of U and V (maps, rows, columns; m/s), the planar direct estimate (local isotropy for the gradients
    not measured; m²/s³):

        ε = ν [2<(∂u'/∂x)²> + 2<(∂v'/∂y)²> + 3<(∂u'/∂y)²> + 3<(∂v'/∂x)²> + 2<∂u'/∂y ∂v'/∂x>]

    x, y: the positions of the columns and of the rows (m); nu: kinematic viscosity (m²/s). The gradients are central
    differences (one-sided at the edges; NaN next to a gap), each mean over the maps where it is known; the maps by
    chunks (memory)."""
    with np.errstate(invalid="ignore"):
        mean_u, mean_v = np.nanmean(U, 0), np.nanmean(V, 0)
    terms = ((2, "ux", "ux"), (2, "vy", "vy"), (3, "uy", "uy"), (3, "vx", "vx"), (2, "uy", "vx"))
    sums = np.zeros((len(terms),) + U.shape[1:])
    counts = np.zeros((len(terms),) + U.shape[1:])
    for a in range(0, U.shape[0], chunk):
        u, v = U[a:a + chunk] - mean_u, V[a:a + chunk] - mean_v
        d = {"ux": np.gradient(u, x, axis=2), "uy": np.gradient(u, y, axis=1),
             "vx": np.gradient(v, x, axis=2), "vy": np.gradient(v, y, axis=1)}
        for n, (_, p, q) in enumerate(terms):
            product = d[p] * d[q]
            ok = np.isfinite(product)
            sums[n] += np.where(ok, product, 0).sum(0)
            counts[n] += ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    return nu * sum(c * means[n] for n, (c, _, _) in enumerate(terms))


def integral_time(U, interval, max_lag=None):
    """The integral time scale of the fluctuations of U (maps, rows, columns) at each grid point (in the unit of
    interval, the time between successive maps): the integral of its temporal autocorrelation (integral_to_zero)."""
    return integral_to_zero(time_autocorrelation(U, max_lag), interval)
