"""Spatial operators of the SFT equation on the finite-volume grid.

The equation solved, for the radial field B(lat, lon, t) in the frame that
co-rotates with the Carrington rate, is

    dB/dt = - Omega(lat) dB/dlon
            - 1/(R cos lat) d/dlat [ u(lat) cos(lat) B ]
            + eta/(R^2 cos lat) d/dlat [ cos(lat) dB/dlat ]
            + eta/(R^2 cos^2 lat) d^2B/dlon^2
            - B / tau + S.

Integrating over a latitude cell with weight cos(lat) gives the conservative
update  dB_i/dt = -(F_{i+1/2} - F_{i-1/2}) / (R A_i)  with A_i = int cos(lat) dlat.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .constants import R_SUN_M
from .grid import Grid

# ---------------------------------------------------------------------------
# Slope limiters: given backward (a) and forward (b) differences, return the
# limited slope used in the MUSCL reconstruction.
# ---------------------------------------------------------------------------


def _vanleer(a, b):
    ab = a * b
    return np.where(ab > 0.0, 2.0 * ab / np.where(ab > 0.0, a + b, 1.0), 0.0)


def _minmod(a, b):
    return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _mc(a, b):
    lim = np.minimum(np.minimum(2.0 * np.abs(a), 2.0 * np.abs(b)), 0.5 * np.abs(a + b))
    return np.where(a * b > 0.0, np.sign(a) * lim, 0.0)


LIMITERS = {
    "vanleer": _vanleer,
    "minmod": _minmod,
    "mc": _mc,
    "none": lambda a, b: 0.5 * (a + b),  # unlimited central slope (linear, not TVD)
    "upwind": lambda a, b: np.zeros_like(a),  # first-order donor cell
}


def _as_column(x: np.ndarray, ndim: int) -> np.ndarray:
    """Reshape a latitude vector so it broadcasts against an array with ``ndim`` dims."""
    return x.reshape((-1,) + (1,) * (ndim - 1))


class Advection:
    """Explicit meridional advection with a MUSCL reconstruction.

    ``u_faces`` is the meridional velocity [m/s] at the ``n_lat + 1`` cell faces.
    The operator acts along axis 0 of arrays shaped (n_lat, ...).
    """

    def __init__(self, grid: Grid, u_faces: np.ndarray, limiter: str = "vanleer"):
        if limiter not in LIMITERS:
            raise ValueError(f"unknown limiter {limiter!r}; choose from {sorted(LIMITERS)}")
        self.grid = grid
        self.limiter = limiter
        self._limit = LIMITERS[limiter]
        # Face transport coefficient cos(lat_f) u_f / R  [1/s]; exactly 0 at the poles.
        self.face_vel = grid.cos_faces * np.asarray(u_faces, dtype=float) / R_SUN_M
        self.face_vel[0] = self.face_vel[-1] = 0.0

    def tendency(self, B: np.ndarray) -> np.ndarray:
        nd = B.ndim
        slope = np.zeros_like(B)
        d = np.diff(B, axis=0)
        slope[1:-1] = self._limit(d[:-1], d[1:])
        left = B[:-1] + 0.5 * slope[:-1]  # state just south of interior face
        right = B[1:] - 0.5 * slope[1:]  # state just north of interior face
        v = _as_column(self.face_vel[1:-1], nd)
        flux = np.zeros((B.shape[0] + 1,) + B.shape[1:], dtype=B.dtype)
        flux[1:-1] = v * np.where(v >= 0.0, left, right)
        return -(flux[1:] - flux[:-1]) / _as_column(self.grid.area, nd)

    def cfl(self, dt: float) -> float:
        """Largest fraction of a cell's content advected out in one step of ``dt`` seconds."""
        v = self.face_vel
        out = np.maximum(v[1:], 0.0) + np.maximum(-v[:-1], 0.0)
        return float(np.max(out * dt / self.grid.area))


class DiffusionDecay:
    """Implicit operator L = surface diffusion + linear decay, per longitudinal mode m.

    For mode m the operator is tridiagonal in latitude:
        (L B)_i = lo_i B_{i-1} + diag_{i,m} B_i + up_i B_{i+1}
    with diag_{i,m} = -(lo_i + up_i) - eta m^2 / (R cos lat_i)^2 - 1/tau.

    ``ms`` lists the modes; an axisymmetric problem uses ``ms = [0]``, in which case
    the trailing axis of the state may hold any number of independent columns.
    """

    def __init__(self, grid: Grid, eta: float, tau: float | None = None, ms=(0,)):
        self.grid = grid
        self.eta = float(eta)
        self.tau = None if tau is None or not np.isfinite(tau) else float(tau)
        self.ms = np.asarray(ms, dtype=float)
        k = self.eta / (R_SUN_M**2 * grid.dlat * grid.area)
        self.lo = k * grid.cos_faces[:-1]  # lo[0] = 0 at the south pole
        self.up = k * grid.cos_faces[1:]  # up[-1] = 0 at the north pole
        decay = 0.0 if self.tau is None else 1.0 / self.tau
        lon_term = self.eta * self.ms[None, :] ** 2 / (R_SUN_M * grid.cos_lat[:, None]) ** 2
        self.diag = -(self.lo + self.up)[:, None] - lon_term - decay  # (n_lat, n_modes)

    @property
    def shared_matrix(self) -> bool:
        return len(self.ms) == 1

    def apply(self, U: np.ndarray) -> np.ndarray:
        nd = U.ndim
        out = (_as_column(self.diag[:, 0], nd) if self.shared_matrix else self.diag) * U
        out[1:] += _as_column(self.lo[1:], nd) * U[:-1]
        out[:-1] += _as_column(self.up[:-1], nd) * U[1:]
        return out

    def factorize(self, coef: float):
        """Return a function solving (I - coef L) X = rhs for the current operator."""
        n, nm = self.grid.n_lat, len(self.ms)
        main = 1.0 - coef * self.diag.T.ravel()  # mode-major ordering
        lo = np.tile(self.lo, nm)
        up = np.tile(self.up, nm)
        mat = sp.diags([-coef * lo[1:], main, -coef * up[:-1]], [-1, 0, 1], format="csc")
        lu = splu(mat, permc_spec="NATURAL")

        if self.shared_matrix:

            def solve(rhs):
                shape = rhs.shape
                r = rhs.reshape(n, -1)
                if np.iscomplexobj(r):
                    x = lu.solve(np.ascontiguousarray(r.real)) + 1j * lu.solve(
                        np.ascontiguousarray(r.imag)
                    )
                else:
                    x = lu.solve(np.ascontiguousarray(r))
                return x.reshape(shape)

        else:

            def solve(rhs):
                v = rhs.T.reshape(-1)  # (n_modes * n_lat,), mode-major
                x = lu.solve(np.column_stack([v.real, v.imag]))
                return (x[:, 0] + 1j * x[:, 1]).reshape(nm, n).T

        return solve
