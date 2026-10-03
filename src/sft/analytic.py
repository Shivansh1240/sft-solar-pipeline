"""Exact solutions used to verify the solver.

* Legendre / spherical-harmonic modes are eigenfunctions of surface diffusion:
  P_l^m(sin lat) cos(m lon) decays as exp(-l (l + 1) eta t / R^2).
* With no decay, any flow u(lat) has an exact steady state for a net (monopolar)
  flux: the transport flux vanishes when eta dB/dlat = u R B, so
  B ~ exp( (R / eta) * int u dlat ).
* A manufactured solution exercises advection, diffusion, decay and sources
  together (method of manufactured solutions).
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import eval_legendre, lpmv

from .constants import R_SUN_M, YEAR_S
from .grid import Grid


def diffusion_rate(l: int, eta_km2s: float) -> float:
    """Decay rate [1/s] of degree-l modes under surface diffusion."""
    return l * (l + 1) * eta_km2s * 1e6 / R_SUN_M**2


def legendre_mode(grid: Grid, l: int) -> np.ndarray:
    """Axisymmetric eigenmode P_l(sin lat) at cell centres."""
    return eval_legendre(l, grid.sin_lat)


def harmonic_mode(grid: Grid, l: int, m: int) -> np.ndarray:
    """Real spherical-harmonic mode P_l^m(sin lat) cos(m lon), shape (n_lat, n_lon)."""
    return lpmv(m, l, grid.sin_lat)[:, None] * np.cos(m * grid.lon)[None, :]


def steady_state_monopole(
    grid: Grid, u0_ms: float, eta_km2s: float, flow="sin2lat", cutoff_deg: float = 75.0
) -> np.ndarray:
    """Exact steady state for net flux, normalised to unit mean field over the sphere."""
    beta = u0_ms * R_SUN_M / (eta_km2s * 1e6)
    lat = grid.lat
    if flow == "sin2lat":
        # int_0^lat sin(2x) dx = sin^2(lat)
        expo = beta * np.sin(lat) ** 2
    elif flow == "sin_cutoff":
        lat0 = np.deg2rad(cutoff_deg)
        x = np.minimum(np.abs(lat), lat0)
        expo = beta * lat0 / np.pi * (1.0 - np.cos(np.pi * x / lat0))
    else:
        raise ValueError("steady state is implemented for 'sin2lat' and 'sin_cutoff'")
    f = np.exp(expo - expo.max())
    return f / (0.5 * np.sum(f * grid.area))


class ManufacturedSolution:
    """B(lat, t) = f(mu) g(t) with mu = sin(lat), for the sin2lat flow.

    In mu the axisymmetric operator is
        adv  = -(2 u0 / R) d/dmu [ mu (1 - mu^2) f ]
        diff = (eta / R^2) d/dmu [ (1 - mu^2) df/dmu ]
    so the source S = dB/dt - (adv + diff - B/tau) is available in closed form.
    The default f is monotonic, so the slope limiter is inactive in the interior.
    """

    def __init__(
        self,
        u0_ms: float,
        eta_km2s: float,
        tau_yr: float | None = None,
        coeffs=(0.3, 1.0, 0.0, 0.5),
        period_yr: float = 1.0,
    ):
        self.u0 = u0_ms
        self.eta = eta_km2s * 1e6
        self.tau = None if tau_yr is None else tau_yr * YEAR_S
        self.f = Polynomial(coeffs)
        self.omega = 2.0 * np.pi / (period_yr * YEAR_S)
        mu = Polynomial([0.0, 1.0])
        one_m_mu2 = Polynomial([1.0, 0.0, -1.0])
        adv = -(2.0 * self.u0 / R_SUN_M) * (mu * one_m_mu2 * self.f).deriv()
        diff = (self.eta / R_SUN_M**2) * (one_m_mu2 * self.f.deriv()).deriv()
        self.spatial_op = adv + diff  # polynomial in mu

    def g(self, t_s):
        return 1.0 + 0.5 * np.sin(self.omega * t_s)

    def dg(self, t_s):
        return 0.5 * self.omega * np.cos(self.omega * t_s)

    def exact(self, grid: Grid, t_yr: float) -> np.ndarray:
        return self.f(grid.sin_lat) * self.g(t_yr * YEAR_S)

    def source(self, grid: Grid):
        """Return S(t_yr) -> array [G/s] at the cell centres of ``grid``."""
        mu = grid.sin_lat
        f, op = self.f(mu), self.spatial_op(mu)
        decay = 0.0 if self.tau is None else 1.0 / self.tau

        def S(t_yr):
            t = t_yr * YEAR_S
            return f * self.dg(t) - self.g(t) * (op - decay * f)

        return S
