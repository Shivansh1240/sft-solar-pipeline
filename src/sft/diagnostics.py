"""Physical diagnostics of a surface field.

Every function takes ``B`` with latitude on axis 0. On a 2-D grid, axis 1 is
longitude. Any further trailing axes (ensemble members, time) are carried
through, so ``axial_dipole(B, grid)`` works on a single field or on a stack.
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_legendre

from .constants import R_SUN_CM
from .grid import Grid


def lon_average(B: np.ndarray, grid: Grid) -> np.ndarray:
    """Longitude-averaged field (identity on an axisymmetric grid)."""
    return B.mean(axis=1) if grid.is_2d else B


def _lat_integral(weights: np.ndarray, Bbar: np.ndarray):
    return np.tensordot(weights, Bbar, axes=(0, 0))


def signed_flux(B: np.ndarray, grid: Grid):
    """Net magnetic flux over the whole sphere [Mx]. Conserved without decay or sources."""
    return 2.0 * np.pi * R_SUN_CM**2 * _lat_integral(grid.area, lon_average(B, grid))


def unsigned_flux(B: np.ndarray, grid: Grid):
    """Total unsigned flux [Mx].

    On an axisymmetric grid this is the unsigned flux of the longitude-averaged
    field, which is a lower bound on the true unsigned flux.
    """
    if grid.is_2d:
        return R_SUN_CM**2 * grid.dlon * _lat_integral(grid.area, np.abs(B).sum(axis=1))
    return 2.0 * np.pi * R_SUN_CM**2 * _lat_integral(grid.area, np.abs(B))


def axial_dipole(B: np.ndarray, grid: Grid):
    """Axial dipole moment D = (3/2) int B sin(lat) cos(lat) dlat [G].

    Normalised so that B = sin(lat) gives D = 1.
    """
    return 1.5 * _lat_integral(grid.dipole_w, lon_average(B, grid))


def polar_field(B: np.ndarray, grid: Grid, cap_deg: float = 60.0):
    """Area-weighted mean field poleward of +/- ``cap_deg`` [G]. Returns (north, south)."""
    Bbar = lon_average(B, grid)
    north = grid.lat_deg > cap_deg
    south = grid.lat_deg < -cap_deg
    wn = grid.area * north
    ws = grid.area * south
    return _lat_integral(wn, Bbar) / wn.sum(), _lat_integral(ws, Bbar) / ws.sum()


def legendre_coefficients(B: np.ndarray, grid: Grid, l_max: int = 20) -> np.ndarray:
    """Coefficients b_l of the expansion  Bbar(lat) = sum_l b_l P_l(sin lat).

    Each P_l is integrated exactly over every cell, using
    int P_l dmu = (P_{l+1} - P_{l-1}) / (2l + 1), so b_1 equals ``axial_dipole``.
    Returns an array of shape (l_max + 1, ...).
    """
    Bbar = lon_average(B, grid)
    mu_f = np.sin(grid.lat_faces)
    ls = np.arange(1, l_max + 1)[:, None]
    prim = eval_legendre(ls + 1, mu_f[None, :]) - eval_legendre(ls - 1, mu_f[None, :])
    w = np.vstack([grid.area[None, :], np.diff(prim, axis=1)]) * 0.5  # (l, n_lat)
    return np.tensordot(w, Bbar, axes=(1, 0))
