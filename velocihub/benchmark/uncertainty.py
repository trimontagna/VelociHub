"""The uncertainty of the PTV/PIV predicted from the characteristics of the flow (a priori), against the actual error.

The binned PTV (each grid point: the mean displacement of the N links whose middle falls in its bin of size h): a link
is the displacement at its position plus the measurement noise (sigma_0 on each component). Over the bin, the
displacement varies; to the second order around the grid point,

    u(x) = u + grad(u) . r + 1/2 r . H(u) . r,     r uniform over the bin (h x h),

so the mean of the N links is u plus

- a random error, standard deviation sqrt((sigma_0² + h²/12 |grad u|²) / N): the noise of the links, and the spread of
  the displacement over the bin (the gradient) sampled at N random positions;
- a bias h²/24 (d²u/dx² + d²u/dy²) (the mean of the curvature over the bin: the top-hat filter of the binning).

Both from the flow (here: the exact gradients and curvature; for real data: those of the measured field, and sigma_0
from the dispersion of the links), the number of links of each bin (from the results) and the noise. The prediction is
checked as the uncertainty of a measurement is (Sciacchitano et al. 2015): the RMS uncertainty against the RMS error,
the error over the uncertainty (its random part) of standard deviation 1, and the fraction of the errors within one and
two uncertainties (68 % and 95 % for a Gaussian error).

Not yet modelled: SPICY (a regression: its own filter, from the transfer function measured on the modes, and its
random error from its residual), and the image level (the correlation of the PIV windows, the truncation of the
time step along curved paths, the peak locking)."""

import numpy as np

# Step of the second derivatives of the flow (px)
CURVATURE_STEP = 1.0


def laplacian(flow, X, Y, t):
    """d²/dx² + d²/dy² of U and V of the flow (central differences, step CURVATURE_STEP)."""
    h = CURVATURE_STEP
    c = flow.velocity(X, Y, t)
    s = [flow.velocity(X + h, Y, t), flow.velocity(X - h, Y, t), flow.velocity(X, Y + h, t),
         flow.velocity(X, Y - h, t)]
    return tuple((sum(p[i] for p in s) - 4 * c[i]) / h**2 for i in (0, 1))


def binned_uncertainty(flow, X, Y, t, particles, noise, spacing):
    """The predicted uncertainty of the binned PTV at the grid points (X, Y: px, Y up) of a pair at time t, from the
    links of each bin (particles): random parts sigma_u, sigma_v and biases bias_u, bias_v (px; NaN without link)."""
    ux, uy, vx, vy = flow.gradients(X, Y, t)
    lu, lv = laplacian(flow, X, Y, t)
    N = np.where(np.asarray(particles) > 0, np.asarray(particles, np.float64), np.nan)
    spread = spacing**2 / 12
    return dict(sigma_u=np.sqrt((noise**2 + spread * (ux**2 + uy**2)) / N),
                sigma_v=np.sqrt((noise**2 + spread * (vx**2 + vy**2)) / N),
                bias_u=spacing**2 / 24 * lu, bias_v=spacing**2 / 24 * lv)


def predict(flow, X, Y, status, times, particles, noise, spacing):
    """binned_uncertainty of each pair (arrays pairs x rows x columns), NaN out of the flow."""
    flowing = status == 0
    out = {}
    for k, t in enumerate(times):
        u = binned_uncertainty(flow, X, Y, t, particles[k], noise, spacing)
        for name, a in u.items():
            out.setdefault(name, []).append(np.where(flowing, a, np.nan))
    return {name: np.array(a) for name, a in out.items()}


def uncertainty_metrics(U, V, exact, pred):
    """The predicted uncertainty (predict) against the actual error of fields U, V: unc_ratio (RMS uncertainty of
    the displacement vector / its RMS error), unc_z (standard deviation of the error minus the bias over the random
    part, each component), unc_cover_1 and unc_cover_2 (% of the errors within one and two uncertainties), and the
    arrays of the errors and of the total uncertainties of the components (for the report)."""
    eu, ev = U - exact["U"], V - exact["V"]
    tu = np.sqrt(pred["sigma_u"] ** 2 + pred["bias_u"] ** 2)
    tv = np.sqrt(pred["sigma_v"] ** 2 + pred["bias_v"] ** 2)
    ok = np.isfinite(eu) & np.isfinite(ev) & np.isfinite(tu) & np.isfinite(tv)
    if not ok.any():
        return {}, None
    e = np.concatenate([eu[ok], ev[ok]])
    total = np.concatenate([tu[ok], tv[ok]])
    z = np.concatenate([((eu - pred["bias_u"]) / pred["sigma_u"])[ok], ((ev - pred["bias_v"]) / pred["sigma_v"])[ok]])
    m = {"unc_ratio": float(np.sqrt(np.mean(total**2) / np.mean(e**2))),
         "unc_z": float(np.std(z)),
         "unc_cover_1": float(100 * np.mean(np.abs(e) <= total)),
         "unc_cover_2": float(100 * np.mean(np.abs(e) <= 2 * total)),
         "unc_rms": float(np.sqrt(np.mean(total**2) * 2))}
    return m, (e, total)
