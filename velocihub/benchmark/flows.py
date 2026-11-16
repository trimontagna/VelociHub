"""Analytic flows of the benchmark: displacement fields known exactly, against which VelociHub's results are compared.

Every flow is defined in the frame of the post-processing (sensor space): X to the right and Y upwards from the
bottom-left corner of the images, in pixels, the displacement (U, V) in px per pair (per frame: the pairs are
consecutive frames), at the time t in frames (only the turbulence changes with time). All of them are exactly
divergence-free (they derive from a stream function), so that the divergence of a result measures its error alone and
the divergence penalty of SPICY can be tested on any of them. A flow may have a body (a solid region, masked): no
particle there, the grid points inside it masked.

    flow = FLOWS["cylinder"]()
    U, V = flow.velocity(X, Y)
    W = flow.vorticity(X, Y)        # dV/dX - dU/dY, by central differences of the exact field
    solid = flow.solid(X, Y)        # inside the body
"""

from dataclasses import dataclass, field

import numpy as np

# The images of the benchmark (px): the field of view of every flow
WIDTH, HEIGHT = 512, 320
# Step of the central differences of the exact fields (px): their error is far below that of any result
H_DIFF = 1e-3


class Flow:
    """An analytic flow: velocity(X, Y) in px per pair, its body (solid), its reference velocity u_ref (px per pair)
    and length l_ref (px), a name, a description and its parameters (for the report)."""

    key = ""
    name = ""
    description = ""
    u_ref = 4.0
    l_ref = 64.0

    def velocity(self, X, Y, t=0.0):
        raise NotImplementedError

    def solid(self, X, Y):
        """Inside the body (no flow, masked)."""
        return np.zeros(np.broadcast(np.asarray(X), np.asarray(Y)).shape, bool)

    def parameters(self):
        """The parameters of the flow, for the report: {label: value}."""
        return {}

    def gradients(self, X, Y, t=0.0):
        """dU/dX, dU/dY, dV/dX, dV/dY of the exact field, by central differences (step H_DIFF)."""
        X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        Ue, Ve = self.velocity(X + H_DIFF, Y, t)
        Uw, Vw = self.velocity(X - H_DIFF, Y, t)
        Un, Vn = self.velocity(X, Y + H_DIFF, t)
        Us, Vs = self.velocity(X, Y - H_DIFF, t)
        h2 = 2 * H_DIFF
        return (Ue - Uw) / h2, (Un - Us) / h2, (Ve - Vw) / h2, (Vn - Vs) / h2

    def vorticity(self, X, Y, t=0.0):
        """omega = dV/dX - dU/dY (1/pair)."""
        _, dudy, dvdx, _ = self.gradients(X, Y, t)
        return dvdx - dudy

    def divergence(self, X, Y, t=0.0):
        """dU/dX + dV/dY (1/pair): 0 for every flow of the benchmark (a check of their formulas)."""
        dudx, _, _, dvdy = self.gradients(X, Y, t)
        return dudx + dvdy

    # The flow doesn't change with time
    steady = True


class Uniform(Flow):
    """A uniform displacement, oblique: no structure at all, the baseline (bias, edges, spurious vorticity)."""

    key = "uniform"
    name = "Uniform flow"
    description = ("A uniform, oblique displacement. Anything but a constant in a result is an artefact of the "
                   "method: bias, edge effects, spurious vorticity.")

    def __init__(self, u=4.0, v=1.0):
        self.u, self.v = u, v

    def velocity(self, X, Y, t=0.0):
        shape = np.broadcast(np.asarray(X), np.asarray(Y)).shape
        return np.full(shape, self.u), np.full(shape, self.v)

    def parameters(self):
        return {"U (px/pair)": self.u, "V (px/pair)": self.v}


class Cylinder(Flow):
    """Potential flow around a circular cylinder (doublet in a uniform stream): a masked body, stagnation points,
    a speed of 2 U at its shoulders, strong gradients near its surface, no vorticity."""

    key = "cylinder"
    name = "Potential flow around a cylinder"
    description = ("Inviscid flow around a circular cylinder (uniform stream and doublet), the cylinder masked: "
                   "stagnation points, twice the free-stream speed at its shoulders, strong gradients at its surface, "
                   "and no vorticity anywhere.")

    def __init__(self, u=4.0, radius=40.0, centre=(WIDTH / 2, HEIGHT / 2)):
        self.u_ref, self.radius, self.centre = u, radius, centre
        self.l_ref = radius

    def velocity(self, X, Y, t=0.0):
        x, y = np.asarray(X, np.float64) - self.centre[0], np.asarray(Y, np.float64) - self.centre[1]
        r2 = np.maximum(x * x + y * y, 1e-12)
        a2 = self.radius**2
        U = self.u_ref * (1 - a2 * (x * x - y * y) / r2**2)
        V = -self.u_ref * 2 * a2 * x * y / r2**2
        inside = r2 < a2
        return np.where(inside, 0.0, U), np.where(inside, 0.0, V)

    def solid(self, X, Y):
        x, y = np.asarray(X, np.float64) - self.centre[0], np.asarray(Y, np.float64) - self.centre[1]
        return x * x + y * y < self.radius**2

    def parameters(self):
        return {"U (px/pair)": self.u_ref, "Radius R (px)": self.radius,
                "Centre (px)": f"({self.centre[0]:g}, {self.centre[1]:g})"}


def _log_cosh(a):
    return np.logaddexp(a, -a) - np.log(2.0)


class ShearJet(Flow):
    """A planar jet near its exit: a core of uniform speed between two tanh shear layers that thicken along X (the
    potential core region of the jet measurements). Stream function psi = U/2 delta [ln cosh a - ln cosh b], a, b =
    (Y - Yc +- h) / delta(X): the flow rate of the jet is constant, and the thickening shear layers have a transverse
    displacement V."""

    key = "jet"
    name = "Jet with shear layers"
    description = ("A planar jet near its exit: a uniform core between two tanh shear layers that thicken "
                   "linearly downstream (with the transverse displacement this implies). The thin layers on the "
                   "left are barely resolved by the grid: the test of the resolution of steep gradients.")

    def __init__(self, u=4.0, half_height=64.0, delta0=8.0, growth=0.02, yc=HEIGHT / 2):
        self.u_ref, self.h, self.delta0, self.growth, self.yc = u, half_height, delta0, growth, yc
        self.l_ref = 2 * half_height

    def delta(self, X):
        return self.delta0 + self.growth * np.asarray(X, np.float64)

    def velocity(self, X, Y, t=0.0):
        d = self.delta(X)
        a = (np.asarray(Y, np.float64) - self.yc + self.h) / d
        b = (np.asarray(Y, np.float64) - self.yc - self.h) / d
        U = self.u_ref / 2 * (np.tanh(a) - np.tanh(b))
        # V = -dpsi/dX = -dpsi/ddelta ddelta/dX
        dpsi = self.u_ref / 2 * (_log_cosh(a) - a * np.tanh(a) - _log_cosh(b) + b * np.tanh(b))
        return U, -dpsi * self.growth

    def parameters(self):
        return {"Core U (px/pair)": self.u_ref, "Half-height h (px)": self.h,
                "Shear layer thickness δ (px)": f"{self.delta0:g} + {self.growth:g} X "
                                                f"({self.delta0:g} to {self.delta(WIDTH):g})",
                "Axis Yc (px)": self.yc}


class CellularModes(Flow):
    """A uniform stream with arrays of counter-rotating cells of several wavelengths superposed (Taylor-Green
    vortices: psi = A / k sin(k X + phx) sin(k Y + phy)): the spatial resolution of a method, as the fraction of
    the amplitude of each wavelength it keeps."""

    key = "modes"
    name = "Cellular modes"
    description = ("A uniform stream plus arrays of counter-rotating vortices of four wavelengths, superposed. The "
                   "amplitude a method keeps at each wavelength is its spatial resolution (its transfer function); "
                   "the shortest wavelength is 3 grid spacings.")

    WAVELENGTHS = (320.0, 160.0, 80.0, 48.0)
    PHASES = ((0.3, 1.1), (2.0, 0.4), (1.3, 2.6), (0.7, 1.9))

    def __init__(self, u=3.0, amplitude=1.0):
        self.u0, self.amplitude = u, amplitude
        self.u_ref, self.l_ref = 4.0, 80.0

    def mode(self, i, X, Y):
        """U, V of the mode i alone, of unit amplitude."""
        k = 2 * np.pi / self.WAVELENGTHS[i]
        px, py = self.PHASES[i]
        X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        return np.sin(k * X + px) * np.cos(k * Y + py), -np.cos(k * X + px) * np.sin(k * Y + py)

    def velocity(self, X, Y, t=0.0):
        U = np.full(np.broadcast(np.asarray(X), np.asarray(Y)).shape, self.u0)
        V = np.zeros_like(U)
        for i in range(len(self.WAVELENGTHS)):
            u, v = self.mode(i, X, Y)
            U, V = U + self.amplitude * u, V + self.amplitude * v
        return U, V

    def parameters(self):
        return {"Uniform U (px/pair)": self.u0, "Amplitude of each mode (px/pair)": self.amplitude,
                "Wavelengths (px)": ", ".join(f"{w:g}" for w in self.WAVELENGTHS)}


@dataclass
class _BlasiusSolution:
    """The Blasius function f(eta): f''' + f f'' / 2 = 0, f(0) = f'(0) = 0, f'(inf) = 1, as cubic Hermite splines."""

    eta_max: float = 12.0
    f: object = field(init=False)
    fp: object = field(init=False)
    f_inf: float = field(init=False)

    FPP0 = 0.332057336215196  # f''(0)

    def __post_init__(self):
        from scipy.integrate import solve_ivp
        from scipy.interpolate import CubicHermiteSpline

        eta = np.linspace(0.0, self.eta_max, 2401)
        sol = solve_ivp(lambda t, y: [y[1], y[2], -0.5 * y[0] * y[2]], (0.0, self.eta_max), [0.0, 0.0, self.FPP0],
                        method="DOP853", t_eval=eta, rtol=1e-12, atol=1e-13)
        f, fp, fpp = sol.y
        self.f = CubicHermiteSpline(eta, f, fp)
        self.fp = CubicHermiteSpline(eta, fp, fpp)
        self.f_inf = float(f[-1] - self.eta_max)  # f = eta + f_inf beyond eta_max (-1.7208)

    def values(self, eta):
        """f, f' at eta >= 0."""
        eta = np.asarray(eta, np.float64)
        inside = eta < self.eta_max
        e = np.clip(eta, 0.0, self.eta_max)
        return np.where(inside, self.f(e), eta + self.f_inf), np.where(inside, self.fp(e), 1.0)


_BLASIUS = None


def blasius_solution():
    global _BLASIUS
    if _BLASIUS is None:
        _BLASIUS = _BlasiusSolution()
    return _BLASIUS


class Blasius(Flow):
    """A thick laminar boundary layer on a flat plate (Blasius): the wall at Y = Y_wall (masked below), its leading
    edge x0 upstream of the field of view. eta = (Y - Y_wall) sqrt(U / (nu x)), U f'(eta), V = sqrt(nu U / x) (eta f' -
    f) / 2, with x = X + x0; nu in px^2 per pair. Its displacement and momentum thicknesses are exact: 1.7208 and
    0.6641 sqrt(nu x / U), wall shear U f''(0) sqrt(U / (nu x))."""

    key = "blasius"
    name = "Laminar boundary layer (Blasius)"
    description = ("A thick laminar boundary layer on a flat plate (Blasius solution), the wall masked: viscous "
                   "vorticity concentrated at the wall, a small wall-normal displacement, and integral thicknesses "
                   "known exactly. The boundary layer thickness doubles across the field of view.")

    DISPLACEMENT, MOMENTUM = 1.7207876573, 0.6641147  # delta* and theta over sqrt(nu x / U)

    def __init__(self, u=4.0, y_wall=40.0, x0=170.0, delta99_start=60.0):
        self.u_ref, self.y_wall, self.x0 = u, y_wall, x0
        # nu / U from the thickness at the left edge: delta99 = 4.91 sqrt(nu x / U)
        self.nu_u = (delta99_start / 4.91) ** 2 / x0
        self.l_ref = delta99_start
        self.solution = blasius_solution()

    def scale(self, X):
        """sqrt(nu x / U) (px) at X."""
        return np.sqrt(self.nu_u * (np.asarray(X, np.float64) + self.x0))

    def velocity(self, X, Y, t=0.0):
        g = self.scale(X)
        eta = (np.asarray(Y, np.float64) - self.y_wall) / g
        f, fp = self.solution.values(np.maximum(eta, 0.0))
        U = self.u_ref * fp
        V = self.u_ref * (g / (np.asarray(X, np.float64) + self.x0)) * (eta * fp - f) / 2
        wall = eta <= 0
        return np.where(wall, 0.0, U), np.where(wall, 0.0, V)

    def solid(self, X, Y):
        return np.asarray(Y, np.float64) + 0 * np.asarray(X, np.float64) < self.y_wall

    def delta99(self, X):
        """Thickness where U = 0.99 U_e (px)."""
        return 4.91 * self.scale(X)

    def displacement_thickness(self, X):
        return self.DISPLACEMENT * self.scale(X)

    def momentum_thickness(self, X):
        return self.MOMENTUM * self.scale(X)

    def wall_shear(self, X):
        """dU/dY at the wall (1/pair)."""
        return self.u_ref * self.solution.FPP0 / self.scale(X)

    def parameters(self):
        return {"U (px/pair)": self.u_ref, "Wall Y (px)": self.y_wall, "Leading edge x0 (px left of the image)":
                self.x0, "ν (px²/pair)": round(self.nu_u * self.u_ref, 4),
                "δ99 (px)": f"{4.91 * self.scale(0):.0f} to {4.91 * self.scale(WIDTH):.0f}",
                "δ* (px)": f"{self.displacement_thickness(0):.1f} to {self.displacement_thickness(WIDTH):.1f}"}


class Turbulence(Flow):
    """Synthetic turbulence: a homogeneous, isotropic, divergence-free random field (stream function psi = sum of
    a_n cos(k_n . x + phi_n), M Fourier modes of random directions and phases, their wavenumbers drawn from the
    spectrum E(k) ~ k^4 exp(-2 (k / k_p)^2)), frozen and convected by a uniform stream U0 (Taylor's hypothesis:
    u(x, t) = U0 + u'(x - U0 t)). Its statistics are known from its modes: the variances, the Reynolds stress, the
    spatial autocorrelations (cosine sums) and so the integral length scales to the first zero crossing, the
    integral time scale L_x / U0, and the gradients of the dissipation."""

    key = "turbulence"
    name = "Synthetic turbulence"
    description = ("A homogeneous, isotropic random field built from Fourier modes (divergence-free), frozen and "
                   "convected by a uniform stream: its variances, Reynolds stress, integral length and time scales "
                   "and dissipation are known from its modes. The case of the statistics of the post-processing.")
    steady = False

    def __init__(self, u=4.0, rms=0.6, peak_wavelength=128.0, modes=400, seed=7):
        self.u0, self.u_ref, self.target_rms, self.peak_wavelength = u, u, rms, peak_wavelength
        self.l_ref = peak_wavelength
        rng = np.random.default_rng(seed)
        kp = 2 * np.pi / peak_wavelength
        # Wavenumbers from the spectrum (inverse of its distribution on a fine grid)
        k = np.linspace(1e-4, 6 * kp, 20001)
        E = k**4 * np.exp(-2 * (k / kp) ** 2)
        cdf = np.cumsum(E)
        cdf /= cdf[-1]
        mag = np.interp(rng.random(modes), cdf, k)
        angle = rng.uniform(0, np.pi, modes)
        self.kx, self.ky = mag * np.cos(angle), mag * np.sin(angle)
        self.phase = rng.uniform(0, 2 * np.pi, modes)
        # Each mode the same energy, (a k)^2 / 2: the total u'^2 + v'^2 = 2 rms^2
        self.a = np.sqrt(2 * 2 * rms**2 / modes) / mag

    def fluctuation(self, X, Y):
        """u', v' at X, Y (the frozen field)."""
        X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        shape = np.broadcast(X, Y).shape
        x, y = np.broadcast_to(X, shape).ravel(), np.broadcast_to(Y, shape).ravel()
        u, v = np.zeros(x.size), np.zeros(x.size)
        for start in range(0, len(self.a), 50):
            sl = slice(start, start + 50)
            s = np.sin(np.outer(x, self.kx[sl]) + np.outer(y, self.ky[sl]) + self.phase[sl])
            # psi = a cos(k.x + phi): u = dpsi/dY = -a ky sin, v = -dpsi/dX = a kx sin
            u -= s @ (self.a[sl] * self.ky[sl])
            v += s @ (self.a[sl] * self.kx[sl])
        return u.reshape(shape), v.reshape(shape)

    def velocity(self, X, Y, t=0.0):
        u, v = self.fluctuation(np.asarray(X, np.float64) - self.u0 * np.asarray(t, np.float64), Y)
        return self.u0 + u, v

    # The statistics of the modes (expected values over space and time)

    def variances(self):
        """<u'^2>, <v'^2>, <u'v'> (px^2 per pair^2)."""
        a2 = self.a**2 / 2
        return float(np.sum(a2 * self.ky**2)), float(np.sum(a2 * self.kx**2)), float(-np.sum(a2 * self.kx * self.ky))

    def correlation(self, r, axis, component):
        """The autocorrelation coefficient of u' (or v') for separations r along x or y."""
        a2 = self.a**2 / 2 * (self.ky**2 if component == "u" else self.kx**2)
        k = self.kx if axis == "x" else self.ky
        return np.cos(np.outer(np.asarray(r, np.float64), k)) @ a2 / a2.sum()

    def integral_length(self, axis, component):
        """The integral of the autocorrelation to its first zero crossing (px), as the post-processing defines it."""
        r = np.linspace(0, 4 * self.peak_wavelength, 40001)
        rho = self.correlation(r, axis, component)
        below = np.flatnonzero(rho <= 0)
        end = below[0] if below.size else len(r)
        return float(np.trapezoid(rho[:end], r[:end]))

    def integral_time(self):
        """The integral time scale of u at a point (frames): frozen turbulence, rho(tau) = rho_x(U0 tau)."""
        return self.integral_length("x", "u") / self.u0

    def dissipation(self, nu, h=None):
        """The planar estimate of the dissipation the post-processing computes (turbulence.planar_dissipation),
        nu [2 <ux^2> + 2 <vy^2> + 3 <uy^2> + 3 <vx^2> + 2 <uy vx>] (nu in px^2 per pair: px^2 per pair^3), with the
        exact gradients of the modes, or with h: those of central differences on a grid of spacing h (px), each
        derivative k replaced by sin(k h) / h (what a grid can give at best)."""
        a2 = self.a**2 / 2
        kx, ky = self.kx, self.ky
        dx, dy = (kx, ky) if h is None else (np.sin(kx * h) / h, np.sin(ky * h) / h)
        # u' = -a ky sin(.), v' = a kx sin(.): du/dx = -a ky dx cos, du/dy = -a ky dy cos, dv/dx = a kx dx cos,
        # dv/dy = a kx dy cos
        ux2, uy2 = np.sum(a2 * (ky * dx) ** 2), np.sum(a2 * (ky * dy) ** 2)
        vx2, vy2 = np.sum(a2 * (kx * dx) ** 2), np.sum(a2 * (kx * dy) ** 2)
        uyvx = np.sum(a2 * (-ky * dy) * (kx * dx))
        return float(nu * (2 * ux2 + 2 * vy2 + 3 * uy2 + 3 * vx2 + 2 * uyvx))

    def parameters(self):
        uu, vv, uv = self.variances()
        return {"U0 (px/pair)": self.u0, "RMS of u', v' (px/pair)": f"{np.sqrt(uu):.3f}, {np.sqrt(vv):.3f}",
                "<u'v'> (px²/pair²)": f"{uv:.4f}", "Peak wavelength (px)": self.peak_wavelength,
                "Fourier modes": len(self.a),
                "Integral lengths Lx^u, Ly^u (px)": f"{self.integral_length('x', 'u'):.1f}, "
                                                    f"{self.integral_length('y', 'u'):.1f}",
                "Integral time (frames)": f"{self.integral_time():.2f}"}


# The flows of the benchmark, in the order of the report
FLOWS = {cls.key: cls for cls in (Uniform, Cylinder, ShearJet, CellularModes, Blasius, Turbulence)}
