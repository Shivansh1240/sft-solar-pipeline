"""
sft_master_r6.py  ─ Research-grade Surface Flux Transport pipeline
===================================================================
Self-contained single-file implementation.  Covers:
  • Physics  (BMR profiles, meridional flow, differential rotation)
  • Numerics (SSP2(2,2,2) RK-IMEX + van Leer TVD, Thomas solver, spectral, 2-D Strang)
  • Diagnostics (flux conservation, centroid tracking, FWHM, dipole)
  • Visualization (Figure-5 reproduction, butterfly, sensitivity, convergence)
  • AI analysis  (Claude, off by default, opt-in via --ai)
  • CLI          (--quick, --case, --ai, --2d, --sensitivity,
                  --convergence, --compare-schemes, --param-set, --t-end)

Round-6 diagnostic & analysis fixes (applied on top of rounds 1–5)
───────────────────────────────────────────────────────────────────
DIAG-1  profile_fwhm_robust: Two new validity guards before computing FWHM:
          (a) Peak-to-background ratio: peak < 3× 25th-percentile background
              → return NaN.  Prevents meaningless FWHM on flat/noisy profiles.
          (b) Unimodality check: if flux outside ±N/8 cells of the peak exceeds
              30% of peak → return NaN.  Prevents 125° spikes during cross-
              equatorial cancellation when two lobes of comparable amplitude
              appear on opposite sides of the equator.

DIAG-2  flux_conservation: Signed-flux drift normalised by unsigned total flux
        (not signed[0]).  When signed[0]≈0 (equatorial BMR), the old formula
        produced a misleading 0.00% because the denominator collapsed to the
        unsigned fallback but the percent was then essentially 0/large_number.
        New: phi_scale = unsigned_total_0 always.  Also added signed_drift_abs
        in physical units (G·rad) for reference.

DIAG-3  flux_conservation: Per-polarity metric renamed rel_error_pct →
        polarity_cancellation_pct with explicit documentation.  The ~50%
        values seen in BMR runs are PHYSICAL (cross-equatorial diffusive
        cancellation), not a numerical scheme error.  Old key rel_error_pct
        retained for backward compatibility.

DIAG-4  flux_conservation: Added late_time_decay_err_pct — the fraction by
        which per-polarity flux deviates from exp(-t/τ) at t > 1.5 τ, after
        cancellation has completed.  This is the honest metric for validating
        the decay operator; it is NaN when the run is shorter than 1.5 τ.

DIAG-5  SFTDiagnostics.summary: Prints signed_drift_abs alongside the
        percentage, flags polarity_cancellation as physical, and reports
        late_time_decay_err_pct when available.

DIAG-6  plot_diagnostics / plot_flux_budget / plot_all_diagnostics_extended:
        The "Per-polarity decay error" subplot is re-labelled
        "Polarity Cancellation (physical)" with an explanatory subtitle and
        the 1% red threshold removed (it is not a 1%-style numerical target).

DIAG-7  plot_polar_field_evolution: Added text annotation explaining that
        asymmetry index = 1.0 for an antisymmetric dipole field (|N| = |S|,
        opposite signs) is the correct expected result, not a diagnostic bug.

DIAG-8  centroid_tracking: Docstring and returned dict include a warning that
        early-time instantaneous centroid rates are inflated by polarity-
        cancellation artefact and should not be interpreted as physical
        advection speeds.

Round-5 physics fixes (applied on top of rounds 1–4)
─────────────────────────────────────────────────────
BUG-A  RKIMEX1D.step / RKIMEX2D._lat_diff_decay_step:
       Stage-2 k2_E coefficient corrected from (2γ−1)≈−0.414 to γ≈0.293.
BUG-B  Stage-2 explicit L_I now evaluated at U1.
BUG-C  Neumann (zero-flux) boundary conditions at ±89°.
BUG-D  _L_diff_explicit boundary cells use one-sided Neumann flux.

References: Athalathil 2024, Baumann 2006, Cameron & Schüssler 2015,
            Yeates 2023, Pareschi & Russo 2005, van Leer 1977.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d as _interp1d
from scipy.ndimage import gaussian_filter1d

if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz

try:
    import anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False


# ══════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS & PARAMETER SETS
# ══════════════════════════════════════════════════════════════════════════

R_SUN       = 6.96e8
TAU_DECAY   = 5.0 * 3.156e7
ETA_DEFAULT = 500e6
U0_DEFAULT  = 12.5
OMEGA_CAR   = 2 * np.pi / (25.38 * 86400)

PARAM_SETS = {
    "athalathil2024": {
        "eta": 500e6, "u0": 12.5, "tau": 5 * 3.156e7,
        "description": "Athalathil et al. 2024 reference params"},
    "cameron2010": {
        "eta": 250e6, "u0": 11.0, "tau": np.inf,
        "description": "Cameron et al. 2010 (no decay)"},
    "petrovay2019_best": {
        "eta": 625e6, "u0": 12.5, "tau": 7.5 * 3.156e7,
        "description": "Petrovay & Talafha 2019 best-fit"},
    "baumann2004": {
        "eta": 600e6, "u0": 11.0, "tau": 5 * 3.156e7,
        "description": "Baumann et al. 2004"},
    "yeates2023": {
        "eta": 500e6, "u0": 12.5, "tau": 10 * 3.156e7,
        "description": "Yeates et al. 2023 review defaults"},
}


# ══════════════════════════════════════════════════════════════════════════
# 2. CONFIG & GRID
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class SFTConfig:
    eta:     float = ETA_DEFAULT
    u0:      float = U0_DEFAULT
    tau:     float = TAU_DECAY
    R_sun:   float = R_SUN
    dt:      float = 1800.0
    dlat:    float = 1.0
    dlon:    float = 1.0
    lat_min: float = -89.0
    lat_max: float =  89.0
    flow_profile: str = 'van_ballegooijen'
    limiter:      str = 'van_leer'
    include_diff_rot: bool = False
    B0:                float = 14.88
    delta_lat_sep_deg: float = 6.0
    sigma_lat_deg:     float = 3.51
    sigma_lon_deg:     float = 12.0

    def __post_init__(self):
        assert self.flow_profile in ('van_ballegooijen', 'schad', 'multicell')
        assert self.limiter in ('van_leer', 'minmod', 'superbee')
        assert self.dt > 0 and self.eta > 0 and self.u0 > 0
        if np.isfinite(self.tau):
            assert self.tau > 0

    @classmethod
    def from_param_set(cls, name: str, **overrides) -> 'SFTConfig':
        p = PARAM_SETS[name]
        return cls(eta=p['eta'], u0=p['u0'], tau=p['tau'], **overrides)

    @property
    def tau_yr(self) -> float:
        return self.tau / (365.25 * 86400)

    @property
    def eta_km2_s(self) -> float:
        return self.eta / 1e6


@dataclass
class SFTGrid:
    cfg: SFTConfig

    def __post_init__(self):
        c = self.cfg
        self.lat_deg  = np.arange(c.lat_min, c.lat_max + c.dlat, c.dlat)
        self.lon_deg  = np.arange(0.0, 360.0, c.dlon)
        self.lat_rad  = np.deg2rad(self.lat_deg)
        self.lon_rad  = np.deg2rad(self.lon_deg)
        self.dlam     = float(self.lat_rad[1] - self.lat_rad[0])
        self.dphi     = float(self.lon_rad[1] - self.lon_rad[0])
        self.Nlat     = len(self.lat_deg)
        self.Nlon     = len(self.lon_deg)
        self.LON, self.LAT = np.meshgrid(self.lon_deg, self.lat_deg)
        self.cos_lam  = np.cos(self.lat_rad)
        self.lat_faces = np.concatenate([
            [self.lat_rad[0] - 0.5 * self.dlam],
            self.lat_rad + 0.5 * self.dlam
        ])
        self.cos_half = np.cos(self.lat_faces)


# ══════════════════════════════════════════════════════════════════════════
# 3. PHYSICS
# ══════════════════════════════════════════════════════════════════════════

def _mf_vanball(lat_rad: np.ndarray, u0: float, lat0_deg: float = 75.0) -> np.ndarray:
    lam0 = np.deg2rad(lat0_deg)
    u    = u0 * np.sin(2.0 * lat_rad)
    return np.where(np.abs(lat_rad) < lam0, u, 0.0)

def _mf_schad(lat_rad: np.ndarray, u0: float) -> np.ndarray:
    return u0 * np.sin(np.pi * lat_rad / np.deg2rad(90.0))

def _mf_multicell(lat_rad: np.ndarray, u0: float, n_cells: int = 1) -> np.ndarray:
    if n_cells == 1:
        return _mf_vanball(lat_rad, u0)
    elif n_cells == 2:
        return u0 * np.sin(2.0 * lat_rad) * (1.0 - 2.0 * np.sin(lat_rad) ** 6)
    raise ValueError(f"n_cells must be 1 or 2, got {n_cells}")

def meridional_flow(lat_rad: np.ndarray, cfg: SFTConfig) -> np.ndarray:
    return {
        'van_ballegooijen': _mf_vanball,
        'schad':            _mf_schad,
        'multicell':        _mf_multicell,
    }[cfg.flow_profile](lat_rad, cfg.u0)

def differential_rotation_snodgrass(lat_rad: np.ndarray) -> np.ndarray:
    s = np.sin(lat_rad)
    return (2.71e-6 + (-0.503e-6) * s**2 + (-0.422e-6) * s**4) - OMEGA_CAR

def bmr_2d(lat_grid_deg, lon_grid_deg,
           lat0_deg=12.0, lon0_deg=100.0,
           cfg: Optional[SFTConfig] = None,
           delta_lat_sep_deg=6.0, sigma_lat_deg=3.51,
           sigma_lon_deg=12.0, B0=14.88) -> np.ndarray:
    if cfg is not None:
        delta_lat_sep_deg = cfg.delta_lat_sep_deg
        sigma_lat_deg     = cfg.sigma_lat_deg
        sigma_lon_deg     = cfg.sigma_lon_deg
        B0                = cfg.B0
    lat_pos = lat0_deg + delta_lat_sep_deg / 2.0
    lat_neg = lat0_deg - delta_lat_sep_deg / 2.0
    def gauss2d(lat_c, lon_c):
        dlat = lat_grid_deg - lat_c
        dlon = lon_grid_deg - lon_c
        dlon = np.where(dlon >  180, dlon - 360, dlon)
        dlon = np.where(dlon < -180, dlon + 360, dlon)
        return np.exp(-0.5 * (dlat / sigma_lat_deg)**2
                      - 0.5 * (dlon / sigma_lon_deg)**2)
    return B0 * (gauss2d(lat_pos, lon0_deg) - gauss2d(lat_neg, lon0_deg))

def bmr_longitude_averaged(lat_deg, lat0_deg=12.0,
                            cfg: Optional[SFTConfig] = None,
                            delta_lat_sep_deg=6.0, sigma_lat_deg=3.51,
                            sigma_lon_deg=12.0, B0=14.88) -> np.ndarray:
    if cfg is not None:
        delta_lat_sep_deg = cfg.delta_lat_sep_deg
        sigma_lat_deg     = cfg.sigma_lat_deg
        sigma_lon_deg     = cfg.sigma_lon_deg
        B0                = cfg.B0
    amp     = B0 * (sigma_lon_deg / 360.0) * np.sqrt(2 * np.pi)
    lat_pos = lat0_deg + delta_lat_sep_deg / 2.0
    lat_neg = lat0_deg - delta_lat_sep_deg / 2.0
    return amp * (np.exp(-0.5 * ((lat_deg - lat_pos) / sigma_lat_deg)**2)
                - np.exp(-0.5 * ((lat_deg - lat_neg) / sigma_lat_deg)**2))

def total_flux(Br, lat_rad) -> float:
    return float(_trapz(Br * np.cos(lat_rad), lat_rad))

def axial_dipole_moment(Br, lat_rad) -> float:
    return float((3.0 / (4.0 * np.pi)) * 2 * np.pi
                 * _trapz(Br * np.sin(lat_rad) * np.cos(lat_rad), lat_rad))

def flux_centroid(Br, lat_rad, polarity: str = 'positive') -> float:
    """
    Flux-weighted mean latitude of one polarity.

    PHYSICS FIX: Weight by |B| cos(lambda) dl (= area element on sphere),
    NOT by bare |B| dl.  Without cos(lambda), polar regions are counted
    at the same weight as equatorial despite having cos->0 area, which
    artificially pulls the centroid poleward and inflates migration rates.

    centroid = int lambda |B| cos(lambda) dlambda
               ------------------------------------
                  int |B| cos(lambda) dlambda
    """
    cos_lam = np.cos(lat_rad)
    B_mask  = np.abs(Br) * cos_lam * (Br > 0 if polarity == 'positive' else Br < 0)
    denom   = float(_trapz(B_mask, lat_rad))
    if abs(denom) < 1e-30:
        return float('nan')
    return float(np.rad2deg(_trapz(lat_rad * B_mask, lat_rad) / denom))


# ══════════════════════════════════════════════════════════════════════════
# 4. NUMERICS
# ══════════════════════════════════════════════════════════════════════════

def _flux_limiter(r: np.ndarray, kind: str = 'van_leer') -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if kind == 'van_leer':
        return (r + np.abs(r)) / (1.0 + np.abs(r) + 1e-30)
    elif kind == 'minmod':
        return np.clip(r, 0.0, 1.0)
    elif kind == 'superbee':
        return np.maximum(0.0,
               np.maximum(np.minimum(2.0 * r, 1.0),
                          np.minimum(r, 2.0)))
    raise ValueError(f"Unknown limiter: {kind!r}")


def _advection_kernel(B: np.ndarray,
                      u_half: np.ndarray,
                      cos_lam: np.ndarray,
                      dlam: float,
                      R_sun: float,
                      limiter: str = 'van_leer') -> np.ndarray:
    N = len(B)
    Q = B * cos_lam
    Qg = np.zeros(N + 4)
    Qg[2:N+2] = Q
    Qg[0]   = Q[0];  Qg[1]   = Q[0]
    Qg[N+2] = Q[-1]; Qg[N+3] = Q[-1]
    idx = np.arange(N + 1)
    Qmm = Qg[idx];       Qm  = Qg[idx + 1]
    Qp  = Qg[idx + 2];   Qpp = Qg[idx + 3]
    dQ_fwd = Qp - Qm;    dQ_bwd = Qm - Qmm
    sign_L = np.where(dQ_bwd >= 0, 1.0, -1.0)
    r_L    = dQ_fwd / (dQ_bwd + 1e-30 * sign_L)
    phi_L  = _flux_limiter(r_L, limiter)
    Qf_L   = Qm + 0.5 * phi_L * dQ_bwd
    dQ_rr  = Qpp - Qp
    sign_R = np.where(dQ_rr >= 0, 1.0, -1.0)
    r_R    = dQ_fwd / (dQ_rr + 1e-30 * sign_R)
    phi_R  = _flux_limiter(r_R, limiter)
    Qf_R   = Qp - 0.5 * phi_R * dQ_rr
    Qf     = np.where(u_half >= 0, Qf_L, Qf_R)
    F      = u_half * Qf
    div_Q  = (F[1:] - F[:-1]) / dlam
    return -div_Q / (R_sun * cos_lam + 1e-30)


def _advection_kernel_2d(B2d: np.ndarray,
                          u_half: np.ndarray,
                          cos_lam: np.ndarray,
                          dlam: float,
                          R_sun: float,
                          limiter: str = 'van_leer') -> np.ndarray:
    N, M = B2d.shape
    Q = B2d * cos_lam[:, None]
    Qg = np.zeros((N + 4, M))
    Qg[2:N+2, :] = Q
    Qg[0,   :] = Q[0,  :]; Qg[1,   :] = Q[0,  :]
    Qg[N+2, :] = Q[-1, :]; Qg[N+3, :] = Q[-1, :]
    idx = np.arange(N + 1)
    Qmm = Qg[idx,     :]; Qm  = Qg[idx + 1, :]
    Qp  = Qg[idx + 2, :]; Qpp = Qg[idx + 3, :]
    dQ_fwd = Qp - Qm;   dQ_bwd = Qm - Qmm
    sign_L = np.where(dQ_bwd >= 0, 1.0, -1.0)
    r_L    = dQ_fwd / (dQ_bwd + 1e-30 * sign_L)
    phi_L  = _flux_limiter(r_L, limiter)
    Qf_L   = Qm + 0.5 * phi_L * dQ_bwd
    dQ_rr  = Qpp - Qp
    sign_R = np.where(dQ_rr >= 0, 1.0, -1.0)
    r_R    = dQ_fwd / (dQ_rr + 1e-30 * sign_R)
    phi_R  = _flux_limiter(r_R, limiter)
    Qf_R   = Qp - 0.5 * phi_R * dQ_rr
    u      = u_half[:, None]
    Qf     = np.where(u >= 0, Qf_L, Qf_R)
    F      = u * Qf
    div_Q  = (F[1:, :] - F[:-1, :]) / dlam
    return -div_Q / (R_sun * cos_lam[:, None] + 1e-30)


def thomas_solve(a: np.ndarray, b: np.ndarray,
                 c: np.ndarray, d: np.ndarray) -> np.ndarray:
    N   = len(d)
    c_  = np.empty(N); d_ = np.empty(N); x = np.empty(N)
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, N):
        denom = b[i] - a[i] * c_[i - 1]
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i] * d_[i - 1]) / denom
    x[-1] = d_[-1]
    for i in range(N - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]
    return x


def build_diffusion_decay_matrix(N: int, dlam_rad: float, eta: float,
                                  R_sun: float, tau: float, dt: float,
                                  lat_rad: np.ndarray, alpha: float = 0.5):
    """
    Build tridiagonal (I − α dt L_diff + α dt/τ) for Thomas solve.

    BUG-C FIX (round-5): Boundary rows implement Neumann (zero-flux) BC.
      i=0   (south): only Dhp[0] contributes  — no south-face flux.
      i=N-1 (north): only Dhm[-1] contributes — no north-face flux.
    """
    decay_coeff = (alpha * dt / tau) if np.isfinite(tau) else 0.0

    cos_c  = np.cos(lat_rad)
    cos_hp = np.cos(lat_rad + 0.5 * dlam_rad)
    cos_hm = np.cos(lat_rad - 0.5 * dlam_rad)

    D_hp = alpha * dt * eta * cos_hp / (R_sun**2 * dlam_rad**2)
    D_hm = alpha * dt * eta * cos_hm / (R_sun**2 * dlam_rad**2)

    safe_cos = np.where(np.abs(cos_c) > 1e-8, cos_c, 1e-8)

    a = np.zeros(N); b = np.ones(N); c = np.zeros(N)

    # Interior points
    a[1:N-1] = -D_hm[1:N-1] / safe_cos[1:N-1]
    c[1:N-1] = -D_hp[1:N-1] / safe_cos[1:N-1]
    b[1:N-1] = 1.0 + decay_coeff - a[1:N-1] - c[1:N-1]

    # BUG-C FIX — South boundary (i=0): zero south-face flux
    c[0] = -D_hp[0] / safe_cos[0]
    b[0] = 1.0 + decay_coeff - c[0]      # a[0] = 0 already

    # BUG-C FIX — North boundary (i=N-1): zero north-face flux
    a[-1] = -D_hm[-1] / safe_cos[-1]
    b[-1] = 1.0 + decay_coeff - a[-1]    # c[-1] = 0 already

    return a, b, c


# ══════════════════════════════════════════════════════════════════════════
# 5. SOLVERS
# ══════════════════════════════════════════════════════════════════════════

class RKIMEX1D:
    """
    SSP2(2,2,2) / RK-IMEX for 1-D axisymmetric SFT.

    Scheme: Pareschi & Russo (2005)  γ = 1 − 1/√2 ≈ 0.29289.

      Stage 1:
        k1_E = L_E(Bⁿ)
        U1   = (I − γdt L_I)⁻¹ [Bⁿ + γdt k1_E]

      Stage 2 (BUG-A + BUG-B fixes):
        k2_E    = L_E(U1)
        L_I1    = L_diff(U1) − U1/τ       ← at U1, not Bⁿ (BUG-B)
        rhs2    = Bⁿ + dt[(1−γ)k1_E + γ k2_E]   ← γ not (2γ−1) (BUG-A)
                    + dt(1−γ)·L_I1
        Bⁿ⁺¹   = (I − γdt L_I)⁻¹ rhs2

    Boundary conditions (BUG-C): Neumann at ±89°.
    """

    def __init__(self, grid: SFTGrid, cfg: SFTConfig,
                 u_half_override: Optional[np.ndarray] = None):  # FIX: allow custom flow injection
        self.grid    = grid
        self.cfg     = cfg
        if u_half_override is not None:  # FIX: use caller-supplied flow at cell faces
            self.u_half = u_half_override
        else:
            self.u_half = meridional_flow(grid.lat_faces, cfg)
        self.cos_lam = grid.cos_lam
        self._gam    = 1.0 - 1.0 / np.sqrt(2.0)

        self.a, self.b, self.c = build_diffusion_decay_matrix(
            grid.Nlat, grid.dlam, cfg.eta, cfg.R_sun,
            cfg.tau, cfg.dt, grid.lat_rad, alpha=self._gam)

        lat    = grid.lat_rad
        dlam   = grid.dlam
        cos_c  = np.cos(lat)
        cos_hp = np.cos(lat + 0.5 * dlam)
        cos_hm = np.cos(lat - 0.5 * dlam)
        self._safe_cos = np.where(np.abs(cos_c) > 1e-8, cos_c, 1e-8)
        self._Dhp = cfg.eta * cos_hp / (cfg.R_sun**2 * dlam**2)
        self._Dhm = cfg.eta * cos_hm / (cfg.R_sun**2 * dlam**2)

    def _L_adv(self, B: np.ndarray) -> np.ndarray:
        return _advection_kernel(B, self.u_half, self.cos_lam,
                                  self.grid.dlam, self.cfg.R_sun,
                                  self.cfg.limiter)

    def _L_diff_explicit(self, B: np.ndarray) -> np.ndarray:
        """Pure spatial diffusion L_diff(B) with Neumann BC (BUG-D fix)."""
        N   = len(B)
        out = np.zeros(N)
        flux_hp = self._Dhp[1:N-1] * (B[2:N]   - B[1:N-1])
        flux_hm = self._Dhm[1:N-1] * (B[1:N-1] - B[0:N-2])
        out[1:N-1] = (flux_hp - flux_hm) / self._safe_cos[1:N-1]
        out[0]  =  self._Dhp[0]  * (B[1]  - B[0])  / self._safe_cos[0]
        out[-1] = -self._Dhm[-1] * (B[-1] - B[-2]) / self._safe_cos[-1]
        return out

    def _implicit_solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve (I − γdt L_I) B_new = rhs via Thomas. BUG-C: no rhs zeroing."""
        return thomas_solve(self.a, self.b, self.c, rhs.copy())

    def step(self, B: np.ndarray) -> np.ndarray:
        dt  = self.cfg.dt
        gam = self._gam

        k1_E = self._L_adv(B)
        rhs1 = B + gam * dt * k1_E
        U1   = self._implicit_solve(rhs1)

        k2_E      = self._L_adv(U1)
        L_diff_U1 = self._L_diff_explicit(U1)
        decay_U1  = -U1 / self.cfg.tau if np.isfinite(self.cfg.tau) else np.zeros_like(U1)
        L_I_U1    = L_diff_U1 + decay_U1

        rhs2 = (B
                + dt * ((1.0 - gam) * k1_E + gam * k2_E)
                + dt * (1.0 - gam) * L_I_U1)
        return self._implicit_solve(rhs2)

    def evolve(self, B0: np.ndarray, t_end_yr: float,
               save_every_days: float = 30.0,
               verbose: bool = True,
               t_start_yr: float = 0.0):
        t_end_s   = (t_end_yr - t_start_yr) * 365.25 * 86400
        n_steps   = max(1, int(round(t_end_s / self.cfg.dt)))
        save_freq = max(1, int(save_every_days * 86400 / self.cfg.dt))

        B       = B0.copy()
        t_list  = [t_start_yr]
        B_list  = [B.copy()]

        for i in range(1, n_steps + 1):
            B = self.step(B)
            if i % save_freq == 0 or i == n_steps:
                t_yr = t_start_yr + i * self.cfg.dt / (365.25 * 86400)
                t_list.append(t_yr)
                B_list.append(B.copy())
                if verbose:
                    print(f"  t={t_yr:.2f} yr  |B|_max={np.max(np.abs(B)):.4f} G")

        return np.array(t_list), np.array(B_list)


# Alias for backward compatibility
RKIMEX2Stage = RKIMEX1D


class RKIMEX2D:
    """Full 2-D SFT via Strang operator splitting. All round-5 fixes applied."""

    def __init__(self, grid: SFTGrid, cfg: SFTConfig):
        self.grid      = grid
        self.cfg       = cfg
        self.u_half    = meridional_flow(grid.lat_faces, cfg)
        self.cos_lam   = grid.cos_lam
        self._gam      = 1.0 - 1.0 / np.sqrt(2.0)

        self.a_lat, self.b_lat, self.c_lat = build_diffusion_decay_matrix(
            grid.Nlat, grid.dlam, cfg.eta, cfg.R_sun,
            cfg.tau, cfg.dt, grid.lat_rad, alpha=self._gam)

        self.include_diff_rot = cfg.include_diff_rot
        if cfg.include_diff_rot:
            self.omega_dr = differential_rotation_snodgrass(grid.lat_rad)

        safe_cos = np.where(np.abs(grid.cos_lam) > 1e-6, grid.cos_lam, 1e-6)
        self._lon_diff_coeff = (self._gam * cfg.dt * cfg.eta
                                 / (cfg.R_sun * safe_cos)**2
                                 / grid.dphi**2)

        lat    = grid.lat_rad
        dlam   = grid.dlam
        cos_c  = np.cos(lat)
        cos_hp = np.cos(lat + 0.5 * dlam)
        cos_hm = np.cos(lat - 0.5 * dlam)
        self._safe_cos_2d = np.where(np.abs(cos_c) > 1e-8, cos_c, 1e-8)
        self._Dhp_2d = cfg.eta * cos_hp / (cfg.R_sun**2 * dlam**2)
        self._Dhm_2d = cfg.eta * cos_hm / (cfg.R_sun**2 * dlam**2)

    def _lat_advection_step(self, B2d: np.ndarray, dt_sub: float) -> np.ndarray:
        L = _advection_kernel_2d(B2d, self.u_half, self.cos_lam,
                                  self.grid.dlam, self.cfg.R_sun,
                                  self.cfg.limiter)
        return B2d + dt_sub * L

    def _lat_diff_decay_step(self, B2d: np.ndarray) -> np.ndarray:
        dt  = self.cfg.dt
        gam = self._gam
        N   = self.grid.Nlat
        Dhp = self._Dhp_2d
        Dhm = self._Dhm_2d
        sc  = self._safe_cos_2d

        B_new = np.empty_like(B2d)
        for j in range(self.grid.Nlon):
            Bn = B2d[:, j]
            U1 = thomas_solve(self.a_lat, self.b_lat, self.c_lat, Bn.copy())

            L_diff_U1 = np.zeros(N)
            fhp = Dhp[1:N-1] * (U1[2:N]   - U1[1:N-1])
            fhm = Dhm[1:N-1] * (U1[1:N-1] - U1[0:N-2])
            L_diff_U1[1:N-1] = (fhp - fhm) / sc[1:N-1]
            L_diff_U1[0]  =  Dhp[0]  * (U1[1]  - U1[0])  / sc[0]
            L_diff_U1[-1] = -Dhm[-1] * (U1[-1] - U1[-2]) / sc[-1]

            decay_U1 = -U1 / self.cfg.tau if np.isfinite(self.cfg.tau) else np.zeros(N)
            L_I_U1   = L_diff_U1 + decay_U1

            rhs2 = Bn + dt * (1.0 - gam) * L_I_U1
            B_new[:, j] = thomas_solve(self.a_lat, self.b_lat, self.c_lat, rhs2)

        return B_new

    def _lon_diff_step(self, B2d: np.ndarray) -> np.ndarray:
        N     = self.grid.Nlon
        k     = np.arange(N, dtype=float)
        eig   = 2.0 * (np.cos(2.0 * np.pi * k / N) - 1.0) / self.grid.dphi**2
        B_new = np.empty_like(B2d)
        for i in range(self.grid.Nlat):
            if np.abs(self.cos_lam[i]) < 1e-6:
                B_new[i, :] = B2d[i, :]
                continue
            coeff  = self._lon_diff_coeff[i]
            Bhat   = np.fft.rfft(B2d[i, :])
            eig_r  = eig[:N // 2 + 1]
            Bhat  /= (1.0 - coeff * eig_r)
            B_new[i, :] = np.fft.irfft(Bhat, n=N)
        return B_new

    def _diff_rot_step(self, B2d: np.ndarray, dt_sub: float) -> np.ndarray:
        if not self.include_diff_rot:
            return B2d
        N     = self.grid.Nlon
        freqs = np.fft.rfftfreq(N)
        B_new = np.empty_like(B2d)
        for i in range(self.grid.Nlat):
            shift_cells = self.omega_dr[i] * dt_sub / self.grid.dphi
            Bhat  = np.fft.rfft(B2d[i, :])
            phase = np.exp(-2j * np.pi * freqs * shift_cells)
            B_new[i, :] = np.fft.irfft(Bhat * phase, n=N)
        return B_new

    def step(self, B2d: np.ndarray) -> np.ndarray:
        dt = self.cfg.dt
        B  = self._lat_advection_step(B2d, dt * 0.5)
        B  = self._lat_diff_decay_step(B)
        B  = self._lon_diff_step(B)
        B  = self._diff_rot_step(B, dt)
        B  = self._lat_advection_step(B, dt * 0.5)
        return B

    def evolve(self, B0_2d: np.ndarray, t_end_yr: float,
               save_every_days: float = 30.0, verbose: bool = True):
        t_end_s    = t_end_yr * 365.25 * 86400
        n_steps    = max(1, int(round(t_end_s / self.cfg.dt)))
        save_every = max(1, int(save_every_days * 86400 / self.cfg.dt))

        B = B0_2d.copy()
        t_snaps = [0.0]; B_snaps = [B.copy()]

        for i in range(1, n_steps + 1):
            B = self.step(B)
            if i % save_every == 0 or i == n_steps:
                t_yr = i * self.cfg.dt / (365.25 * 86400)
                t_snaps.append(t_yr); B_snaps.append(B.copy())
                if verbose:
                    print(f"  [2D] t={t_yr:.2f} yr  |B|_max={np.max(np.abs(B)):.4f} G")

        return np.array(t_snaps), np.array(B_snaps)


class SpectralSFT1D:
    """
    Exact diffusion+decay in Legendre spectral space + explicit advection.
    Reference solver (not SSP2(2,2,2)).  Neumann BC implicit in Legendre basis.
    """

    def __init__(self, lat_deg, l_max=90, eta=500e6, u0=12.5,
                 tau=5 * 3.156e7, R_sun=6.96e8, dt=1800.0,
                 limiter='van_leer', flow_profile='van_ballegooijen'):
        from scipy.special import legendre as _legendre

        cfg  = SFTConfig(eta=eta, u0=u0, tau=tau, R_sun=R_sun, dt=dt,
                         dlat=float(np.asarray(lat_deg)[1] - np.asarray(lat_deg)[0]),
                         flow_profile=flow_profile, limiter=limiter)
        grid = SFTGrid(cfg)

        self.lat_rad  = grid.lat_rad
        self.cos_lat  = grid.cos_lam
        self.N        = grid.Nlat
        self.l_max    = l_max
        self.eta      = eta; self.tau = tau
        self.R_sun    = R_sun; self.dt = dt
        self.dlam     = grid.dlam
        self.limiter  = limiter
        self.u_face   = meridional_flow(grid.lat_faces, cfg)

        print(f"  [SpectralSFT] Pre-computing Legendre basis l_max={l_max}...")
        self.P = np.zeros((l_max + 1, self.N))
        for l in range(l_max + 1):
            self.P[l] = _legendre(l)(self.cos_lat)

        self.spectral_decay = np.array([
            eta * l * (l + 1) / R_sun**2
            + (1.0 / tau if np.isfinite(tau) else 0.0)
            for l in range(l_max + 1)
        ])

    def _to_spectral(self, B):
        c = np.empty(self.l_max + 1)
        for l in range(self.l_max + 1):
            c[l] = ((2 * l + 1) / 2.0
                    * _trapz(B * self.P[l] * self.cos_lat, self.lat_rad))
        return c

    def _from_spectral(self, c):
        return self.P.T @ c

    def step(self, B):
        L      = _advection_kernel(B, self.u_face, self.cos_lat,
                                    self.dlam, self.R_sun, self.limiter)
        B_half = B + self.dt * L
        c      = self._to_spectral(B_half)
        c     *= np.exp(-self.spectral_decay * self.dt)
        return self._from_spectral(c)

    def evolve(self, B0, t_end_yr, verbose=True, save_interval_days=30):
        n_steps    = max(1, int(round(t_end_yr * 365.25 * 86400 / self.dt)))
        save_every = max(1, int(save_interval_days * 86400 / self.dt))
        B = B0.copy()
        t_snaps = [0.0]; B_snaps = [B.copy()]
        for i in range(1, n_steps + 1):
            B = self.step(B)
            if i % save_every == 0 or i == n_steps:
                t_yr = i * self.dt / (365.25 * 86400)
                t_snaps.append(t_yr); B_snaps.append(B.copy())
                if verbose:
                    print(f"  [Spectral] t={t_yr:.2f} yr  |B|={np.max(np.abs(B)):.4f} G")
        return np.array(t_snaps), np.array(B_snaps)


# ══════════════════════════════════════════════════════════════════════════
# 6. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════

def profile_fwhm(B, lat_rad, polarity='positive') -> float:
    return profile_fwhm_robust(B, lat_rad, polarity=polarity)


def profile_fwhm_robust(B: np.ndarray, lat_rad: np.ndarray,
                         polarity: str = 'positive',
                         smooth_sigma_cells: int = 2,
                         min_peak_to_bg_ratio: float = 3.0,
                         max_far_field_fraction: float = 0.30) -> float:
    """
    FWHM of the one-signed part of B with two validity guards (DIAG-1, round-6).

    Guard (a) — Peak-to-background ratio:
        If peak < min_peak_to_bg_ratio × (25th-percentile of the one-sided
        profile), the profile is too flat or dominated by background noise
        and FWHM is not meaningful → return NaN.

    Guard (b) — Unimodality check:
        Computes flux far from the primary peak (outside ±N/8 cells).
        If that far-field exceeds max_far_field_fraction × peak_val, the
        profile has a substantial secondary lobe (e.g. during cross-equatorial
        cancellation) and FWHM is not well-defined → return NaN.

    These guards prevent the spurious 80–125° FWHM spikes seen in BMR runs
    after t~1 yr when inter-polarity cancellation flattens or splits the
    one-signed distribution.
    """
    B_work = np.where(B > 0, B, 0.0) if polarity == 'positive' \
             else np.where(B < 0, -B, 0.0)
    if smooth_sigma_cells > 0:
        B_sm = gaussian_filter1d(B_work.astype(float),
                                  sigma=smooth_sigma_cells,
                                  mode='constant', cval=0.0)
    else:
        B_sm = B_work.astype(float).copy()

    peak_val = float(B_sm.max())
    if peak_val < 1e-12:
        return float('nan')

    # DIAG-1a: Peak-to-background ratio check
    bg_level = float(np.percentile(B_sm, 25))
    if peak_val < min_peak_to_bg_ratio * (bg_level + 1e-30):
        return float('nan')   # Profile too flat / not above background

    i_peak   = int(np.argmax(B_sm))
    N_arr    = len(B_sm)

    # DIAG-1b: Unimodality check — substantial secondary lobe → NaN
    half_width = max(5, N_arr // 8)
    lo = max(0, i_peak - half_width)
    hi = min(N_arr, i_peak + half_width + 1)
    far_mask = np.ones(N_arr, dtype=bool)
    far_mask[lo:hi] = False
    if B_sm[far_mask].max() > max_far_field_fraction * peak_val:
        return float('nan')   # Multi-modal: FWHM is not well-defined

    half = 0.5 * peak_val

    i_lo = i_peak
    while i_lo > 0 and B_sm[i_lo] > half:
        i_lo -= 1
    if i_lo == 0 and B_sm[0] >= half:
        return float('nan')
    dB_lo   = B_sm[i_lo + 1] - B_sm[i_lo]
    frac_lo = (half - B_sm[i_lo]) / dB_lo if abs(dB_lo) > 1e-30 else 0.0
    lat_lo  = lat_rad[i_lo] + frac_lo * (lat_rad[i_lo + 1] - lat_rad[i_lo])

    i_hi = i_peak
    while i_hi < N_arr - 1 and B_sm[i_hi] > half:
        i_hi += 1
    if i_hi == N_arr - 1 and B_sm[-1] >= half:
        return float('nan')
    dB_hi   = B_sm[i_hi - 1] - B_sm[i_hi]
    frac_hi = (B_sm[i_hi - 1] - half) / dB_hi if abs(dB_hi) > 1e-30 else 0.0
    lat_hi  = lat_rad[i_hi - 1] + frac_hi * (lat_rad[i_hi] - lat_rad[i_hi - 1])

    width_deg = float(np.rad2deg(lat_hi - lat_lo))
    return width_deg if width_deg > 0 else float('nan')


def diffusion_broadening_theory(t_yr, sigma0_deg, eta, R_sun) -> np.ndarray:
    t_s = np.asarray(t_yr) * 365.25 * 86400
    return np.rad2deg(np.sqrt(np.deg2rad(sigma0_deg)**2 + 2 * eta * t_s / R_sun**2))


class SFTDiagnostics:
    def __init__(self, t_yr: np.ndarray, B_snaps: np.ndarray,
                 grid: SFTGrid, cfg: SFTConfig):
        self.t_yr    = t_yr
        self.B       = B_snaps
        self.grid    = grid
        self.cfg     = cfg
        self.lat_rad = grid.lat_rad

    def flux_conservation(self) -> dict:
        """
        Signed and per-polarity flux diagnostics.

        Round-6 fixes (DIAG-2, DIAG-3, DIAG-4):

        DIAG-2: signed_drift_pct is now normalised by unsigned_total_0 (not
          |signed[0]|).  When signed[0]≈0 (equatorial BMR), the old formula
          produced 0.00% because the denominator used unsigned flux but the
          percentage was then 0/large, giving false precision.  The new
          phi_scale = unsigned_total_0 always, so the percentage reflects
          the absolute signed-drift relative to the initial unsigned flux.

        DIAG-3: polarity_cancellation_pct (= old rel_error_pct) compares each
          polarity's unsigned flux against exp(-t/τ) from t=0.  The ~50%
          values seen in BMR runs are PHYSICAL: diffusion drives flux across
          the equator where it cancels against the opposite polarity, depleting
          each polarity faster than exp(-t/τ).  This is NOT a numerical error.
          The old name 'rel_error_pct' is retained as a backward-compat alias.

        DIAG-4: late_time_decay_err_pct measures per-polarity deviation from
          exp(-t/τ) only at t > 1.5 τ, after cancellation has stabilised.
          This is the honest metric for validating the decay operator.
          Returns NaN if the run is shorter than 1.5 τ.
        """
        tau_s = self.cfg.tau
        lat   = self.lat_rad
        t_s   = self.t_yr * 365.25 * 86400

        signed = np.array([_trapz(B * np.cos(lat), lat) for B in self.B])

        if np.isfinite(tau_s):
            exp_factor      = np.exp(-t_s / tau_s)
            signed_expected = signed[0] * exp_factor
        else:
            exp_factor      = np.ones_like(t_s)
            signed_expected = np.full_like(t_s, signed[0])

        unsigned_total_0 = float(_trapz(np.abs(self.B[0]) * np.cos(lat), lat))

        # DIAG-2: Always normalise by unsigned flux for a meaningful percentage
        phi_scale        = unsigned_total_0 + 1e-30
        signed_drift_abs = np.abs(signed - signed_expected)      # G·rad
        signed_drift     = signed_drift_abs / phi_scale * 100.0  # % of |Phi_0|

        pos_flux = np.array([_trapz(np.maximum(B, 0) * np.cos(lat), lat) for B in self.B])
        neg_flux = np.array([_trapz(np.maximum(-B, 0) * np.cos(lat), lat) for B in self.B])
        pos_expected = pos_flux[0] * exp_factor
        neg_expected = neg_flux[0] * exp_factor

        # DIAG-3: Physical cancellation, NOT numerical error
        pos_cancellation  = np.abs(pos_flux - pos_expected) / (pos_flux[0] + 1e-30) * 100.0
        neg_cancellation  = np.abs(neg_flux - neg_expected) / (neg_flux[0] + 1e-30) * 100.0
        combined_cancellation = 0.5 * (pos_cancellation + neg_cancellation)

        # DIAG-4: Late-time numerical decay error
        tau_yr_val = self.cfg.tau_yr if np.isfinite(self.cfg.tau) else 1e30
        late_mask  = self.t_yr > 1.5 * tau_yr_val
        if late_mask.sum() >= 2:
            late_pos    = pos_flux[late_mask]
            late_pos_ex = pos_expected[late_mask]
            late_err    = float(
                np.mean(np.abs(late_pos - late_pos_ex) / (late_pos + 1e-30))) * 100.0
        else:
            late_err = float('nan')

        return {
            't_yr':                          self.t_yr,
            'signed_flux':                   signed,
            'signed_expected':               signed_expected,
            'signed_drift_pct':              signed_drift,
            'signed_drift_abs':              signed_drift_abs,          # G·rad
            'max_signed_drift_pct':          float(np.max(signed_drift)),
            'pos_flux':                      pos_flux,
            'neg_flux':                      neg_flux,
            'flux':                          pos_flux + neg_flux,
            'expected':                      pos_expected + neg_expected,
            # DIAG-3: Renamed — physical cancellation, not scheme error
            'polarity_cancellation_pct':     combined_cancellation,
            'max_polarity_cancellation_pct': float(np.max(combined_cancellation)),
            # Backward-compat aliases
            'rel_error_pct':                 combined_cancellation,
            'max_error_pct':                 float(np.max(combined_cancellation)),
            # DIAG-4: Honest scheme-validation metric
            'late_time_decay_err_pct':       late_err,
        }

    def centroid_tracking(self) -> dict:
        """
        Flux-weighted centroid latitude for each polarity over time.

        DIAG-8 note: Instantaneous centroid migration rates at early times
        (t < ~1 yr for typical BMR parameters) are significantly inflated
        by the polarity-cancellation artefact: as the negative lobe near the
        equator is depleted by cross-equatorial diffusion, the positive-lobe
        centroid shifts rapidly upward not because the fluid advects faster
        but because the weighting distribution changes.  The average rate
        over the full run is more representative of physical transport.
        The 'centroid_artefact_warning' key in the returned dict flags this.
        """
        pos_c = np.array([flux_centroid(B, self.lat_rad, 'positive') for B in self.B])
        neg_c = np.array([flux_centroid(B, self.lat_rad, 'negative') for B in self.B])
        if len(self.t_yr) > 1:
            dt_yr    = np.diff(self.t_yr)
            pos_rate = np.diff(pos_c) / (dt_yr + 1e-30)
            neg_rate = np.diff(neg_c) / (dt_yr + 1e-30)
        else:
            pos_rate = neg_rate = np.array([float('nan')])

        # FIX (Failure 3): Cleaned migration rate excluding cancellation artefact
        rate_threshold = 15.0  # deg/yr — physical meridional flow gives 1-3 deg/yr
        pos_physical = np.where(np.abs(pos_rate) < rate_threshold, pos_rate, np.nan)
        neg_physical = np.where(np.abs(neg_rate) < rate_threshold, neg_rate, np.nan)

        return {
            't_yr':                       self.t_yr,
            'pos_centroid_deg':           pos_c,
            'neg_centroid_deg':           neg_c,
            'pos_migration_rate_deg_yr':  pos_rate,
            'neg_migration_rate_deg_yr':  neg_rate,
            'avg_pos_rate':               float(np.nanmean(np.abs(pos_rate))),
            'avg_neg_rate':               float(np.nanmean(np.abs(neg_rate))),
            # FIX: cleaned rates excluding artefact epochs (>15 deg/yr)
            'pos_rate_cleaned_deg_yr':    pos_physical,
            'neg_rate_cleaned_deg_yr':    neg_physical,
            # Guard: np.nanmean raises RuntimeWarning when ALL values are NaN
            # (happens on short runs where every rate exceeds the threshold)
            'avg_pos_rate_cleaned':       float(np.nan if np.all(np.isnan(pos_physical))
                                               else np.nanmean(np.abs(pos_physical))),
            'avg_neg_rate_cleaned':       float(np.nan if np.all(np.isnan(neg_physical))
                                               else np.nanmean(np.abs(neg_physical))),
            # DIAG-8: flag that early-time instantaneous rates are artefact-inflated
            'centroid_artefact_warning':  (
                'Early instantaneous centroid rates overestimate physical transport '
                'speed due to polarity-cancellation reweighting. '
                'Use avg_pos_rate_cleaned for physical estimates.'),
        }

    def profile_evolution(self) -> dict:
        return {
            't_yr':         self.t_yr,
            'peak_pos':     np.array([np.max(B)  for B in self.B]),
            'peak_neg':     np.array([np.min(B)  for B in self.B]),
            'fwhm_pos_deg': np.array([profile_fwhm(B, self.lat_rad, 'positive')
                                      for B in self.B]),
            'fwhm_neg_deg': np.array([profile_fwhm(B, self.lat_rad, 'negative')
                                      for B in self.B]),
        }

    def dipole_evolution(self) -> dict:
        return {
            't_yr':     self.t_yr,
            'dipole_G': np.array([axial_dipole_moment(B, self.lat_rad) for B in self.B]),
        }

    def summary(self, label: str = '') -> dict:
        fc = self.flux_conservation()
        ct = self.centroid_tracking()
        pe = self.profile_evolution()
        de = self.dipole_evolution()
        print(f"\n{'='*60}")
        print(f"  DIAGNOSTICS: {label}")
        print(f"{'='*60}")
        print(f"  Span:                {self.t_yr[0]:.1f} -> {self.t_yr[-1]:.1f} yr")
        # DIAG-2: Report both absolute and relative signed drift
        print(f"  Signed flux drift:   {fc['max_signed_drift_pct']:.3f}% of |Phi_0|"
              f"  (abs max: {float(np.max(fc['signed_drift_abs'])):.3e} G·rad)")
        # DIAG-3: Flag that polarity_cancellation is physical
        print(f"  Polarity cancelltn:  {fc['max_polarity_cancellation_pct']:.1f}%"
              f"  [PHYSICAL — cross-equatorial diffusive cancellation, not scheme error]")
        # DIAG-4: Late-time decay accuracy
        lt_err = fc['late_time_decay_err_pct']
        if not np.isnan(lt_err):
            print(f"  Late-time decay err: {lt_err:.2f}%"
                  f"  [t > 1.5τ, honest numerical validation]")
        else:
            print(f"  Late-time decay err: N/A  (run shorter than 1.5 τ = "
                  f"{1.5 * self.cfg.tau_yr:.1f} yr)")
        print(f"  Peak amplitude:      {pe['peak_pos'][0]:.3f} G -> "
              f"{pe['peak_pos'][-1]:.4f} G")
        print(f"  FWHM (pos):          {pe['fwhm_pos_deg'][0]:.1f} deg -> "
              f"{pe['fwhm_pos_deg'][-1]:.1f} deg  "
              f"[NaN = profile invalid for FWHM after cancellation]")
        print(f"  Centroid (pos):      {ct['pos_centroid_deg'][0]:.1f} deg -> "
              f"{ct['pos_centroid_deg'][-1]:.1f} deg")
        print(f"  Avg poleward rate:   {ct['avg_pos_rate']:.2f} deg/yr"
              f"  (early-time rates may be artefact-inflated)")
        print(f"  Axial dipole:        {de['dipole_G'][0]:.4f} G -> "
              f"{de['dipole_G'][-1]:.4f} G")
        print(f"{'='*60}\n")
        return {'flux': fc, 'centroid': ct, 'profile': pe, 'dipole': de}


def stability_report(grid: SFTGrid, cfg: SFTConfig):
    dlam  = grid.dlam
    dx    = cfg.R_sun * dlam
    cfl_a = cfg.u0 * cfg.dt / dx
    cfl_d = 4.0 * cfg.eta * cfg.dt / dx**2
    print("=" * 55)
    print("  STABILITY ANALYSIS")
    print("=" * 55)
    print(f"  Grid spacing:         {np.rad2deg(dlam):.1f} deg  ({dx/1e6:.1f} Mm)")
    print(f"  Time step:            {cfg.dt:.0f} s  ({cfg.dt/3600:.1f} hr)")
    print(f"  CFL (advection):      {cfl_a:.5f}  [stable < 1]")
    flag = 'UNSTABLE explicit -> IMEX required' if cfl_d > 1 else 'stable'
    print(f"  CFL (diff explicit):  {cfl_d:.2f}  [{flag}]")
    print(f"  IMEX speedup:         ~{cfl_d / max(cfl_a, 1e-10):.0f}x")
    print(f"  Decay timescale:      {cfg.tau_yr:.1f} yr")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════════
# 7. VALIDATION TESTS
# ══════════════════════════════════════════════════════════════════════════

def run_validation_tests(dlat=1.0, verbose=True) -> dict:
    """
    Six canonical validation tests (updated for Neumann BC / round-5 numerics).
    Tests 2 and 3 are most sensitive to BUG-A/B/C/D fixes.
    """
    lat_deg = np.arange(-89.0, 90.0, dlat)
    lat_rad = np.deg2rad(lat_deg)
    sigma0  = np.deg2rad(4.0)
    lat0    = 0.0
    eta     = 500e6; dt = 1800.0

    if verbose:
        print("\n" + "=" * 60)
        print("  VALIDATION TEST SUITE  (round-6, Neumann BC)")
        print("=" * 60)

    results = {}

    # ── Test 1: Pure decay ────────────────────────────────────────────────
    tau_s  = 5.0 * 365.25 * 86400
    t_end  = 0.5
    B0     = np.exp(-0.5 * ((lat_rad - lat0) / sigma0)**2)
    cfg1   = SFTConfig(eta=1.0, u0=0.001, tau=tau_s, dt=dt, dlat=dlat)
    grid1  = SFTGrid(cfg1)
    B0i    = _interp1d(lat_rad, B0, bounds_error=False, fill_value=0.0)(grid1.lat_rad)
    _, Bd  = RKIMEX1D(grid1, cfg1).evolve(B0i, t_end,
                save_every_days=int(t_end * 365), verbose=False)
    true_exp  = float(np.max(B0i)) * np.exp(-t_end * 365.25 * 86400 / tau_s)
    decay_err = abs(float(np.max(Bd[-1])) - true_exp) / (float(np.max(B0i)) + 1e-30)
    passed1   = decay_err < 1e-3
    results['pure_decay'] = {'passed': passed1, 'max_error': decay_err, 'threshold': 1e-3}
    if verbose:
        print(f"  [1] Pure decay:          {'PASS' if passed1 else 'FAIL'}"
              f"  (err={decay_err:.2e}, tol=1e-3)")

    # ── Test 2: Pure diffusion — Gaussian broadening ──────────────────────
    t_end_d  = 0.3
    t_s_d    = t_end_d * 365.25 * 86400
    cfg2     = SFTConfig(eta=eta, u0=0.001, tau=np.inf, dt=dt, dlat=dlat)
    grid2    = SFTGrid(cfg2)
    B0d      = _interp1d(lat_rad, np.exp(-0.5 * ((lat_rad - lat0) / sigma0)**2),
                         bounds_error=False, fill_value=0.0)(grid2.lat_rad)
    _, Bdiff = RKIMEX1D(grid2, cfg2).evolve(B0d, t_end_d,
                 save_every_days=int(t_end_d * 365), verbose=False)
    sigma_th    = np.sqrt(sigma0**2 + 2.0 * eta * t_s_d / R_SUN**2)
    peak_th     = float(np.max(B0d)) * sigma0 / sigma_th
    diff_err    = abs(float(np.max(Bdiff[-1])) - peak_th) / (peak_th + 1e-30)
    fwhm_th_deg = float(np.rad2deg(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_th))
    fwhm_meas   = profile_fwhm_robust(Bdiff[-1], grid2.lat_rad, 'positive')
    fwhm_err    = (abs(fwhm_meas - fwhm_th_deg) / fwhm_th_deg
                   if not np.isnan(fwhm_meas) else 1.0)
    peaks    = np.max(np.abs(Bdiff), axis=1)
    monotone = bool(np.all(np.diff(peaks) <= 1e-8))
    broadened = (not np.isnan(fwhm_meas)) and \
                (fwhm_meas > np.rad2deg(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma0))
    passed2  = monotone and broadened and diff_err < 0.05 and fwhm_err < 0.05
    results['pure_diffusion'] = {
        'passed': passed2, 'peak_error': diff_err, 'fwhm_error': fwhm_err,
        'monotone': monotone, 'broadened': broadened,
        'fwhm_measured_deg': fwhm_meas, 'fwhm_theory_deg': fwhm_th_deg,
    }
    if verbose:
        print(f"  [2] Pure diffusion:      {'PASS' if passed2 else 'FAIL'}"
              f"  (peak err={diff_err:.2%}, FWHM err={fwhm_err:.2%},"
              f" monotone={monotone}, broadened={broadened})")

    # ── Test 3: Flux conservation under pure advection ────────────────────
    cfg3   = SFTConfig(eta=1.0, u0=12.5, tau=np.inf, dt=dt, dlat=dlat)
    grid3  = SFTGrid(cfg3)
    lat3   = grid3.lat_rad; dlam3 = grid3.dlam
    lat0_c = np.deg2rad(15.0)
    B0f    = np.exp(-0.5 * ((lat3 - lat0_c) / sigma0)**2)
    s0_fc  = float(np.sum(B0f * np.cos(lat3) * dlam3))
    u0_fc  = float(np.sum(np.abs(B0f) * np.cos(lat3) * dlam3))
    norm_fc = max(abs(s0_fc), 0.1 * u0_fc + 1e-30)
    t_fc, B_fc = RKIMEX1D(grid3, cfg3).evolve(B0f, 1.0,
                   save_every_days=90, verbose=False)
    drift_pct = np.array([
        abs(float(np.sum(Bi * np.cos(lat3) * dlam3)) - s0_fc) / norm_fc * 100.0
        for Bi in B_fc
    ])
    passed3 = float(np.max(drift_pct)) < 0.05
    results['flux_conservation'] = {
        'passed': passed3,
        'max_drift_pct': float(np.max(drift_pct)),
        'threshold_pct': 0.05,
    }
    if verbose:
        print(f"  [3] Flux conservation:   {'PASS' if passed3 else 'FAIL'}"
              f"  (signed-flux drift={np.max(drift_pct):.4f}%, tol=0.05%)")

    # ── Test 4: TVD — no new extrema ─────────────────────────────────────
    cfg4   = SFTConfig(eta=1.0, u0=12.5, tau=np.inf, dt=dt, dlat=dlat)
    grid4  = SFTGrid(cfg4)
    B0a    = _interp1d(lat_rad, np.exp(-0.5 * ((lat_rad - lat0) / sigma0)**2),
                       bounds_error=False, fill_value=0.0)(grid4.lat_rad)
    _, Badv = RKIMEX1D(grid4, cfg4).evolve(B0a, 0.5,
                save_every_days=90, verbose=False)
    peak0      = float(np.max(Badv[0]))
    peak_f     = float(np.max(Badv[-1]))
    no_new_max = peak_f <= peak0 * (1.0 + 1e-6)
    passed4    = bool(no_new_max)
    results['pure_advection'] = {
        'passed': passed4, 'no_new_extrema': no_new_max,
        'peak_ratio': peak_f / (peak0 + 1e-30),
    }
    if verbose:
        print(f"  [4] Pure advection (TVD):{'PASS' if passed4 else 'FAIL'}"
              f"  (peak ratio={peak_f/max(peak0,1e-30):.8f}, tol=1+1e-6)")

    # ── Test 5: Poleward migration ────────────────────────────────────────
    tau_s_5 = 5.0 * 365.25 * 86400
    cfg5    = SFTConfig(eta=eta, u0=12.5, tau=tau_s_5, dt=dt, dlat=dlat)
    grid5   = SFTGrid(cfg5)
    B0bmr   = bmr_longitude_averaged(grid5.lat_deg, lat0_deg=12.0, cfg=cfg5)
    _, Bcomb = RKIMEX1D(grid5, cfg5).evolve(B0bmr, 2.0,
                 save_every_days=180, verbose=False)
    c_init  = flux_centroid(Bcomb[0],  grid5.lat_rad, 'positive')
    c_final = flux_centroid(Bcomb[-1], grid5.lat_rad, 'positive')
    rate_deg_yr = (c_final - c_init) / 2.0 \
                  if not (np.isnan(c_final) or np.isnan(c_init)) else 0.0
    migrated = not np.isnan(c_final) and (c_final > c_init)
    rate_ok  = 1.0 < rate_deg_yr < 15.0
    passed5  = bool(migrated and rate_ok)
    results['poleward_migration'] = {
        'passed': passed5, 'c_init': c_init, 'c_final': c_final,
        'rate_deg_yr': rate_deg_yr,
    }
    if verbose:
        print(f"  [5] Poleward migration:  {'PASS' if passed5 else 'FAIL'}"
              f"  ({c_init:.1f} -> {c_final:.1f} deg, rate={rate_deg_yr:.2f} deg/yr)")

    # ── Test 6: RKIMEX1D vs SpectralSFT1D ────────────────────────────────
    try:
        t_end_6  = 0.3
        cfg6     = SFTConfig(eta=eta, u0=0.001, tau=np.inf, dt=dt, dlat=dlat)
        grid6    = SFTGrid(cfg6)
        B0_6     = _interp1d(lat_rad, np.exp(-0.5 * ((lat_rad - lat0) / sigma0)**2),
                              bounds_error=False, fill_value=0.0)(grid6.lat_rad)
        _, B_rk  = RKIMEX1D(grid6, cfg6).evolve(B0_6, t_end_6,
                     save_every_days=int(t_end_6 * 365), verbose=False)
        spec     = SpectralSFT1D(grid6.lat_deg, l_max=90, eta=eta, u0=0.001,
                                  tau=np.inf, dt=dt)
        _, B_sp  = spec.evolve(B0_6, t_end_6,
                                save_interval_days=int(t_end_6 * 365), verbose=False)
        rms_num  = float(np.sqrt(np.mean(B_rk[-1]**2)))
        l2_err   = float(np.sqrt(np.mean((B_rk[-1] - B_sp[-1])**2))) / (rms_num + 1e-30)
        passed6  = l2_err < 0.01
        results['rkimex_vs_spectral'] = {
            'passed': passed6, 'l2_relative_error': l2_err, 'threshold': 0.01,
        }
        if verbose:
            print(f"  [6] RKIMEX vs Spectral:  {'PASS' if passed6 else 'FAIL'}"
                  f"  (L2={l2_err:.4%}, tol=1%)")
    except Exception as exc:
        results['rkimex_vs_spectral'] = {'passed': False, 'error': str(exc)}
        if verbose:
            print(f"  [6] RKIMEX vs Spectral:  SKIP ({exc})")

    n_pass = sum(v['passed'] for v in results.values())
    n_all  = len(results)
    if verbose:
        print(f"\n  Result: {n_pass}/{n_all} tests passed")
        print("=" * 60 + "\n")
    return results


# ══════════════════════════════════════════════════════════════════════════
# 8. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

_STYLE = {
    'pos_color': '#1f77b4', 'neg_color': '#d62728',
    'cmap_field': 'RdBu_r', 'fig_dpi': 150,
}
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10,
                     'axes.titlesize': 10, 'legend.fontsize': 8})


def _safe_vmax(arr, percentile=99, fallback=1e-6):
    raw = np.percentile(np.abs(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)),
                        percentile)
    v = float(raw)
    if not np.isfinite(v) or v < 0:
        v = 0.0
    return max(v, float(fallback))


def _add_colorbar(fig, ax, img, label='B (G)'):
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax, label=label)


def plot_initial_bmr(B_2d, lat_deg, lon_deg, title="Initial BMR", ax=None,
                     vmax=None, lon_range=(50, 150)):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    vmax = vmax or _safe_vmax(B_2d)
    il   = np.searchsorted(lon_deg, lon_range[0])
    ih   = np.searchsorted(lon_deg, lon_range[1])
    img  = ax.pcolormesh(lon_deg[il:ih], lat_deg, B_2d[:, il:ih],
                          cmap='RdBu_r', norm=TwoSlopeNorm(0, vmin=-vmax, vmax=vmax),
                          shading='auto')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Latitude (deg)")
    ax.set_title(title)
    _add_colorbar(ax.figure, ax, img)
    return ax


def plot_lon_avg_profile(B_1d, lat_deg, title="Lon-Avg Profile", ax=None,
                          times_yr=None, B_snapshots=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(3.5, 4))
    if B_snapshots is not None and times_yr is not None:
        cm = plt.cm.plasma
        n  = max(len(times_yr) - 1, 1)
        for i, (t, B) in enumerate(zip(times_yr, B_snapshots)):
            ax.plot(B, lat_deg, color=cm(i / n), lw=1.2, label=f"t={t:.1f} yr")
        ax.legend(fontsize=7)
    else:
        ax.plot(B_1d, lat_deg, color=_STYLE['pos_color'], lw=1.5)
    ax.axvline(0, color='r', lw=0.8, ls='--')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("Br (G)"); ax.set_ylabel("Latitude (deg)"); ax.set_title(title)
    ax.set_ylim(lat_deg[0], lat_deg[-1])
    return ax


def plot_butterfly(t_yr, B_snaps, lat_deg, title="Butterfly Diagram",
                   ax=None, vmax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    B_2d = np.array(B_snaps)
    vmax = vmax or _safe_vmax(B_2d, percentile=98)
    img  = ax.pcolormesh(t_yr, lat_deg, B_2d.T,
                          cmap='RdBu_r',
                          norm=TwoSlopeNorm(0, vmin=-vmax, vmax=vmax),
                          shading='auto')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Latitude (deg)"); ax.set_title(title)
    _add_colorbar(ax.figure, ax, img)
    return ax


def plot_figure5_reproduction(B_2d_c1, lon_avg_c1, t_yr_c1, B_snaps_c1,
                               B_2d_c2, lon_avg_c2, t_yr_c2, B_snaps_c2,
                               lat_deg, lon_deg,
                               save_path="figure5_reproduction.png"):
    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.45)
    cases = [
        (B_2d_c1, lon_avg_c1, t_yr_c1, B_snaps_c1, "a", "b", "c", "lambda0=0 deg"),
        (B_2d_c2, lon_avg_c2, t_yr_c2, B_snaps_c2, "d", "e", "f", "lambda0=12 deg"),
    ]
    for row, (B2d, Bla, t, Bs, la, lb, lc, lbl) in enumerate(cases):
        axes = [fig.add_subplot(gs[row, col]) for col in range(3)]
        plot_initial_bmr(B2d, lat_deg, lon_deg,
                          title=f"({la}) Initial BMR", ax=axes[0])
        plot_lon_avg_profile(Bla, lat_deg,
                              title=f"({lb}) {lbl}", ax=axes[1])
        plot_butterfly(t, Bs, lat_deg, title=f"({lc}) Butterfly", ax=axes[2])
    fig.suptitle("Figure 5 — Athalathil et al. (2024) | SSP2(2,2,2) + van Leer (round-6)",
                 y=1.01, fontsize=12)
    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


def plot_diagnostics(diag_dict: dict, lat_deg, save_path="diagnostics.png"):
    """
    Standard 2×3 diagnostic panel.
    DIAG-3 fix: bottom-right relabelled 'Polarity Cancellation (physical)'
    with an explanatory note replacing the misleading '1% threshold'.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("SFT Diagnostic Analysis", fontsize=13)

    fd  = diag_dict['flux']
    cd  = diag_dict['centroid']
    pd_ = diag_dict['profile']
    dd  = diag_dict['dipole']

    ax = axes[0, 0]
    ax.plot(fd['t_yr'], fd['flux'],     'b-o', ms=3, label='Unsigned total')
    ax.plot(fd['t_yr'], fd['expected'], 'r--',       label='Expected exp(-t/tau)')
    ax.set_yscale('log')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("|Phi|"); ax.set_title("Unsigned Flux")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(fd['t_yr'], fd['signed_flux'],     'darkorange', lw=2, label='Signed Phi(t)')
    ax.plot(fd['t_yr'], fd['signed_expected'], 'k--', lw=1.5, label='Phi(0)*exp(-t/tau)')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Phi_signed")
    ax.set_title(f"Signed Flux Conservation\n"
                 f"(drift max={fd['max_signed_drift_pct']:.2f}% of |Phi_0|)")
    ax.legend()

    ax = axes[0, 2]
    ax.plot(cd['t_yr'], cd['pos_centroid_deg'], _STYLE['pos_color'], lw=2, label='Positive')
    ax.plot(cd['t_yr'], cd['neg_centroid_deg'], _STYLE['neg_color'], lw=2, label='Negative')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Centroid (deg)")
    ax.set_title("Poleward Migration\n(avg rate may be artefact-inflated early)")
    ax.legend()

    ax = axes[1, 0]
    ax.semilogy(pd_['t_yr'], pd_['peak_pos'],
                 _STYLE['pos_color'], lw=2, label='Peak +')
    ax.semilogy(pd_['t_yr'], np.abs(pd_['peak_neg']),
                 _STYLE['neg_color'], lw=2, label='Peak -')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Peak |Br| (G)")
    ax.set_title("Amplitude Decay"); ax.legend()

    ax = axes[1, 1]
    # Mask NaN entries so matplotlib doesn't break line at valid-→-invalid transitions
    fwhm_pos = pd_['fwhm_pos_deg'].copy()
    fwhm_neg = pd_['fwhm_neg_deg'].copy()
    valid_pos = np.isfinite(fwhm_pos)
    valid_neg = np.isfinite(fwhm_neg)
    if valid_pos.any():
        ax.plot(pd_['t_yr'][valid_pos], fwhm_pos[valid_pos],
                _STYLE['pos_color'], lw=2, label='Positive (valid epochs)')
    if valid_neg.any():
        ax.plot(pd_['t_yr'][valid_neg], fwhm_neg[valid_neg],
                _STYLE['neg_color'], lw=2, label='Negative (valid epochs)')
    sigma0_deg = 3.51
    fwhm_th = diffusion_broadening_theory(pd_['t_yr'], sigma0_deg, ETA_DEFAULT, R_SUN) * 2.355
    ax.plot(pd_['t_yr'], fwhm_th, 'k--', lw=1, label='Theory sigma(t) [planar]')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("FWHM (deg)")
    ax.set_title("Diffusion Broadening\n(NaN = multi-modal / non-Gaussian, see DIAG-1)")
    ax.legend()

    ax = axes[1, 2]
    ax.plot(dd['t_yr'], dd['dipole_G'], 'purple', lw=2)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Dipole (G)")
    ax.set_title("Axial Dipole Moment")

    plt.tight_layout()
    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


def plot_sensitivity(sens_results: dict, lat_deg,
                     t_eval_yr=2.0, save_path="sensitivity.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Parameter Sensitivity at t={t_eval_yr} yr", fontsize=12)
    for ax, key, xkey, title in [
        (axes[0], 'eta_sweep', 'eta_label', 'Diffusivity eta'),
        (axes[1], 'u0_sweep',  'u0_label',  'Flow Speed u0'),
        (axes[2], 'tau_sweep', 'tau_label', 'Decay Time tau'),
    ]:
        for r in sens_results[key]:
            ax.plot(r['B'], lat_deg, lw=2, label=r[xkey])
        ax.axvline(0, color='k', lw=0.5, ls='--')
        ax.set_xlabel("Br (G)"); ax.set_ylabel("Lat (deg)")
        ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


def plot_diffusion_evolution(t_snap_yr, B_snaps, lat_deg,
                              save_path="diffusion_evolution.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    cm = plt.cm.viridis; n = max(len(t_snap_yr) - 1, 1)
    for i, (t, B) in enumerate(zip(t_snap_yr, B_snaps)):
        ax.plot(lat_deg, B, color=cm(i / n), lw=2, label=f"t={t:.1f} yr")
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("Latitude (deg)"); ax.set_ylabel("Br (G)")
    ax.set_title("Profile evolution (diffusion + advection)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


def plot_convergence(conv_results: dict, t_end_yr=1.0,
                     save_path="convergence.png"):
    resolutions = sorted(conv_results.keys())
    finest      = resolutions[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Grid Convergence Test (t={t_end_yr} yr)", fontsize=12)

    ax = axes[0]
    cm = plt.cm.cool
    for i, dres in enumerate(resolutions):
        lbl = f"dlat={dres} deg" + (" [ref]" if dres == finest else "")
        ax.plot(conv_results[dres]['lat_deg'], conv_results[dres]['B'],
                color=cm(i / max(len(resolutions) - 1, 1)), lw=2, label=lbl)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("Latitude (deg)"); ax.set_ylabel("Br (G)")
    ax.set_title("Final profiles by resolution"); ax.legend()

    ax = axes[1]
    dxs    = [r for r in resolutions if r != finest]
    errors = [conv_results[r]['L1_vs_finest'] for r in dxs]
    if len(dxs) >= 2 and all(e > 0 for e in errors):
        ax.loglog(dxs, errors, 'bo-', lw=2, ms=7, label='L1 error')
        slope, intercept = np.polyfit(np.log(dxs), np.log(errors), 1)
        x_fit = np.array([min(dxs) * 0.8, max(dxs) * 1.2])
        ax.loglog(x_fit, np.exp(intercept) * x_fit**slope, 'r--', lw=1,
                   label=f'fit O(dlat^{slope:.1f})')
        ax.set_title(f"Convergence: slope={slope:.2f}  (ideal ~2)")
    else:
        ax.set_title("L1 error vs resolution")
    ax.set_xlabel("Grid spacing (deg)"); ax.set_ylabel("L1 error vs finest")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# 8b. EXTENDED VISUALIZATION SUITE
# ══════════════════════════════════════════════════════════════════════════

def plot_energy_spectrum(B_snaps, lat_deg, t_yr, save_path="energy_spectrum.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Latitudinal Power Spectrum Evolution", fontsize=12)
    cm = plt.cm.plasma; n = max(len(t_yr) - 1, 1)

    ax = axes[0]
    for i, (t, B) in enumerate(zip(t_yr, B_snaps)):
        ps    = np.abs(np.fft.rfft(B))**2
        freqs = np.fft.rfftfreq(len(B), d=1.0)
        ax.loglog(freqs[1:], ps[1:], color=cm(i/n), lw=1.2, alpha=0.85,
                  label=f"t={t:.1f} yr")
    ax.set_xlabel("Spatial frequency"); ax.set_ylabel("Power |B-hat|^2")
    ax.set_title("Power spectrum vs time")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[1]
    centroids = []
    for B in B_snaps:
        ps    = np.abs(np.fft.rfft(B))**2
        freqs = np.fft.rfftfreq(len(B), d=1.0)
        valid = freqs > 0
        wc    = float(np.sum(freqs[valid] * ps[valid]) / (np.sum(ps[valid]) + 1e-30))
        centroids.append(wc)
    ax.plot(t_yr, centroids, 'b-o', ms=4, lw=2)
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Spectral centroid")
    ax.set_title("Energy migration to large scales"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


def plot_dipole_quadrupole(diag_dict, save_path="multipole_evolution.png"):
    dd = diag_dict['dipole']; pe = diag_dict['profile']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Multipole Moment Evolution", fontsize=12)

    ax = axes[0]
    ax.plot(dd['t_yr'], dd['dipole_G'], 'purple', lw=2.5)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.fill_between(dd['t_yr'], dd['dipole_G'], alpha=0.2, color='purple')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Dipole moment (G)")
    ax.set_title("Axial Dipole D(t)"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(pe['t_yr'], pe['peak_pos'], _STYLE['pos_color'], lw=2, label='Peak +')
    ax.semilogy(pe['t_yr'], np.abs(pe['peak_neg']), _STYLE['neg_color'], lw=2, label='Peak -')
    t_fine = np.linspace(pe['t_yr'][0], pe['t_yr'][-1], 200)
    amp0   = pe['peak_pos'][0]
    ax.semilogy(t_fine, amp0 * np.exp(-t_fine / 5.0), 'k--', lw=1, label='exp(-t/5yr)')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Peak |Br| (G)")
    ax.set_title("Amplitude Decay")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    fwhm_pos = pe['fwhm_pos_deg'].copy(); fwhm_neg = pe['fwhm_neg_deg'].copy()
    valid_pos = np.isfinite(fwhm_pos); valid_neg = np.isfinite(fwhm_neg)
    if valid_pos.any():
        ax.plot(pe['t_yr'][valid_pos], fwhm_pos[valid_pos],
                _STYLE['pos_color'], lw=2, label='FWHM + (valid)')
    if valid_neg.any():
        ax.plot(pe['t_yr'][valid_neg], fwhm_neg[valid_neg],
                _STYLE['neg_color'], lw=2, label='FWHM - (valid)')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("FWHM (deg)")
    ax.set_title("Profile Broadening\n(NaN = multi-modal, see DIAG-1)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


def plot_meridional_flow_profile(cfg, save_path="meridional_flow.png"):
    lat_deg = np.linspace(-70, 70, 300)
    lat_rad = np.deg2rad(lat_deg)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Meridional Flow Profiles", fontsize=12)

    profiles = {
        'van Ballegooijen (1998)': ('van_ballegooijen', '#1f77b4'),
        'Schad (2013)':            ('schad',            '#ff7f0e'),
        'Multi-cell (2-cell)':     ('multicell',        '#2ca02c'),
    }

    ax = axes[0]
    for label, (prof, col) in profiles.items():
        cfg_tmp = SFTConfig(u0=cfg.u0, flow_profile=prof)
        u = meridional_flow(lat_rad, cfg_tmp)
        ax.plot(lat_deg, u, color=col, lw=2.5, label=label)
    ax.axhline(0, color='k', lw=0.8, ls='--'); ax.axvline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel("Latitude (deg)"); ax.set_ylabel("u (m/s)")
    ax.set_title(f"Flow profiles  (u0={cfg.u0} m/s)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    cfg_vb = SFTConfig(u0=cfg.u0, flow_profile='van_ballegooijen')
    u_vb   = meridional_flow(lat_rad, cfg_vb)
    ax.plot(lat_deg, u_vb, '#1f77b4', lw=2, label='u(lambda)')
    ax.plot(lat_deg, u_vb * np.cos(lat_rad), '#d62728', lw=2, ls='--',
             label='u(lambda)*cos(lambda)')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("Latitude (deg)"); ax.set_ylabel("m/s")
    ax.set_title("van Ballegooijen: u vs u*cos(lambda)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


def plot_detailed_butterfly(t_yr, B_snaps, lat_deg,
                             save_path="butterfly_detailed.png"):
    B_2d = np.array(B_snaps)
    vmax = _safe_vmax(B_2d, percentile=98)
    fig, ax = plt.subplots(figsize=(12, 6))
    img = ax.pcolormesh(t_yr, lat_deg, B_2d.T,
                         cmap='RdBu_r',
                         norm=TwoSlopeNorm(0, vmin=-vmax, vmax=vmax),
                         shading='auto', rasterized=True)
    try:
        lvls = [v for v in [-0.05, -0.02, 0.02, 0.05] if -vmax < v < vmax]
        if lvls:
            ax.contour(t_yr, lat_deg, B_2d.T, levels=lvls,
                        colors=['navy', 'steelblue', 'firebrick', 'darkred'],
                        linewidths=0.8, alpha=0.6)
    except Exception:
        pass
    ax.axhspan(-35, -5, alpha=0.06, color='gold', label='Active region belt')
    ax.axhspan(5,   35, alpha=0.06, color='gold')
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.6)
    cbar = plt.colorbar(img, ax=ax, pad=0.01, shrink=0.95)
    cbar.set_label("Br (G)", fontsize=11)
    ax.set_xlabel("Time (yr)", fontsize=12); ax.set_ylabel("Latitude (deg)", fontsize=12)
    ax.set_title("Butterfly Diagram — Surface Flux Transport (round-6)", fontsize=13)
    ax.set_ylim(lat_deg[0], lat_deg[-1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


def plot_polar_field_evolution(t_yr, B_snaps, lat_deg,
                                polar_lat_cutoff=55.0,
                                save_path="polar_field_evolution.png"):
    """
    DIAG-7 (round-6): Added annotation explaining that asymmetry index = 1.0
    is the physically correct outcome for an antisymmetric dipole field
    (N and S polar caps have equal-magnitude, opposite-sign fields).
    The index (N−S)/(|N|+|S|) = +1 when N > 0, S < 0 regardless of relative
    magnitudes — this is by design and does NOT indicate a numerical problem.
    """
    lat_rad = np.deg2rad(lat_deg)
    B_2d    = np.array(B_snaps)
    lc      = polar_lat_cutoff
    north_mask = lat_deg > lc; south_mask = lat_deg < -lc
    cos_lat = np.cos(lat_rad)

    def polar_avg(B_arr, mask):
        out = []
        for B in B_arr:
            w = cos_lat[mask]
            out.append(float(np.sum(B[mask] * w) / (np.sum(w) + 1e-30)))
        return np.array(out)

    north = polar_avg(B_2d, north_mask); south = polar_avg(B_2d, south_mask)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(f"Polar Cap Field Evolution  (|lat| > {lc} deg)", fontsize=13)

    ax = axes[0]
    ax.plot(t_yr, north, color='navy',     lw=2.5, label='North polar cap')
    ax.plot(t_yr, south, color='firebrick', lw=2.5, label='South polar cap')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.fill_between(t_yr, north, alpha=0.15, color='navy')
    ax.fill_between(t_yr, south, alpha=0.15, color='firebrick')
    ax.set_ylabel("mean Br polar cap (G)")
    ax.set_title("North vs South polar field"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    asym = (north - south) / (np.abs(north) + np.abs(south) + 1e-30)
    ax.plot(t_yr, asym, color='darkorchid', lw=2.5)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.fill_between(t_yr, asym, alpha=0.2, color='darkorchid')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Asymmetry index")
    # DIAG-7: Correct the title to explain what index = 1.0 means
    ax.set_title("Hemispheric Asymmetry  (N−S)/(|N|+|S|)\n"
                 "Index = +1.0 when N > 0 and S < 0 (antisymmetric dipole — physically correct)")
    ax.set_ylim(-1.1, 1.1); ax.grid(True, alpha=0.3)

    # DIAG-7: Annotate at the saturation region
    if float(asym[-1]) > 0.8:
        ax.annotate(
            "Index = 1.0: N and S fields have\nopposite signs (correct dipole structure)",
            xy=(float(t_yr[-1]) * 0.6, 1.0),
            xytext=(float(t_yr[-1]) * 0.3, 0.65),
            fontsize=8, color='darkorchid',
            arrowprops=dict(arrowstyle='->', color='darkorchid', lw=1.2),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


def plot_flux_budget(diag_dict, cfg, save_path="flux_budget.png"):
    """
    DIAG-3 fix: Bottom-right panel relabelled 'Polarity Cancellation (physical)'
    with explanatory subtitle. The 1% red threshold is removed because the
    ~50% values seen in BMR runs are PHYSICAL (cross-equatorial diffusion),
    not numerical errors, so a 1%-style alarm is inappropriate here.
    """
    fc     = diag_dict['flux']; tau_yr = cfg.tau_yr
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Magnetic Flux Budget", fontsize=13)

    ax = axes[0, 0]
    ax.semilogy(fc['t_yr'], fc['pos_flux'], _STYLE['pos_color'], lw=2, label='Phi+')
    ax.semilogy(fc['t_yr'], fc['neg_flux'], _STYLE['neg_color'], lw=2, label='Phi-')
    ax.semilogy(fc['t_yr'], fc['expected'], 'k--', lw=1.5,
                 label=f'Phi0 exp(-t/{tau_yr:.1f}yr)')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("|Phi|")
    ax.set_title("Per-polarity unsigned flux"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(fc['t_yr'], fc['signed_flux'], 'darkorange', lw=2.5)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.fill_between(fc['t_yr'], fc['signed_flux'], alpha=0.2, color='darkorange')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Phi_signed")
    ax.set_title("Signed flux (advection-conservation check)"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    sd = fc.get('signed_drift_pct',
                 np.abs(fc['signed_flux'] - fc['signed_flux'][0])
                 / (np.abs(fc['unsigned_total_0']) + 1e-30) * 100
                 if 'unsigned_total_0' in fc
                 else np.abs(fc['signed_flux'] - fc['signed_flux'][0])
                 / (float(np.max(np.abs(fc['signed_flux']))) + 1e-30) * 100)
    ax.plot(fc['t_yr'], sd, 'teal', lw=2)
    ax.axhline(1.0, color='r', ls='--', lw=1.5, label='1% threshold')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Drift (% of |Phi_0|)")
    ax.set_title(f"Signed flux drift (max={fc.get('max_signed_drift_pct',0):.2f}%)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # DIAG-3: Relabelled — physical cancellation, not scheme error
    ax = axes[1, 1]
    canc = fc.get('polarity_cancellation_pct', fc.get('rel_error_pct'))
    ax.plot(fc['t_yr'], canc, 'steelblue', lw=2)
    # Annotate the physical explanation
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Deviation from exp(-t/tau) (%)")
    ax.set_title("Polarity Cancellation (physical)\n"
                 "High values = cross-equatorial diffusion, not numerical error")
    ax.grid(True, alpha=0.3)

    # Add late-time error annotation if available
    lt_err = fc.get('late_time_decay_err_pct', float('nan'))
    if not np.isnan(lt_err):
        tau_yr_v = cfg.tau_yr if np.isfinite(cfg.tau) else None
        if tau_yr_v is not None:
            late_t = 1.5 * tau_yr_v
            if late_t < float(fc['t_yr'][-1]):
                ax.axvline(late_t, color='green', ls=':', lw=1.5,
                            label=f'1.5τ={late_t:.1f} yr')
                ax.text(late_t + 0.1, float(canc.max()) * 0.5,
                        f'Late-time\nnumerical err:\n{lt_err:.1f}%',
                        fontsize=8, color='green')
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


def plot_case_comparison(results_dict, lat_deg, save_path="case_comparison.png"):
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle("Case Comparison: Equatorial vs Mid-Latitude BMR", fontsize=14)

    colors = {1: '#1f77b4', 2: '#d62728'}
    labels = {1: 'Case 1: lambda0=0 deg', 2: 'Case 2: lambda0=12 deg'}

    ax0 = fig.add_subplot(gs[0, :])
    for cid, res in results_dict.items():
        t_yr    = res['t_yr']; B_snaps = res['B_snaps_1d']
        cm_ = plt.cm.Blues if cid == 1 else plt.cm.Reds
        n_  = max(len(t_yr) - 1, 1)
        for i in [0, len(t_yr)//3, 2*len(t_yr)//3, -1]:
            ax0.plot(lat_deg, B_snaps[i],
                      color=cm_(0.3 + 0.7 * abs(i)/n_), lw=1.5, alpha=0.85,
                      label=(f'C{cid} t={t_yr[i]:.1f}yr' if i in [0, -1] else None))
    ax0.axhline(0, color='k', lw=0.5, ls='--'); ax0.axvline(0, color='k', lw=0.5, ls=':')
    ax0.set_xlabel("Latitude (deg)"); ax0.set_ylabel("Br (G)")
    ax0.set_title("Longitude-averaged profiles at t=0, t~2yr, t~4yr, t=final")
    ax0.legend(fontsize=7, ncol=4); ax0.grid(True, alpha=0.3)

    for col, (cid, res) in enumerate(results_dict.items()):
        ax = fig.add_subplot(gs[1, col])
        B_2d = np.array(res['B_snaps_1d'])
        vmax = _safe_vmax(B_2d, percentile=98)
        img  = ax.pcolormesh(res['t_yr'], lat_deg, B_2d.T,
                              cmap='RdBu_r',
                              norm=TwoSlopeNorm(0, vmin=-vmax, vmax=vmax),
                              shading='auto', rasterized=True)
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ct = res['diag']['centroid']
        ax.plot(ct['t_yr'], ct['pos_centroid_deg'], 'w-', lw=1.5, alpha=0.8)
        ax.plot(ct['t_yr'], ct['neg_centroid_deg'], 'k-', lw=1.5, alpha=0.8)
        ax.set_xlabel("Time (yr)"); ax.set_ylabel("Latitude (deg)")
        ax.set_title(f"{labels.get(cid, f'Case {cid}')} — Butterfly")
        plt.colorbar(img, ax=ax, shrink=0.8, label='Br (G)')

    ax_dip = fig.add_subplot(gs[2, 0]); ax_amp = fig.add_subplot(gs[2, 1])
    for cid, res in results_dict.items():
        col = colors[cid]
        dd  = res['diag']['dipole']; pe = res['diag']['profile']
        ax_dip.plot(dd['t_yr'], dd['dipole_G'], color=col, lw=2.5,
                     label=labels.get(cid, f'Case {cid}'))
        ax_amp.semilogy(pe['t_yr'], pe['peak_pos'], color=col, lw=2.5,
                         label=labels.get(cid, f'Case {cid}'))

    ax_dip.axhline(0, color='k', lw=0.5, ls='--')
    ax_dip.set_xlabel("Time (yr)"); ax_dip.set_ylabel("Dipole (G)")
    ax_dip.set_title("Axial Dipole Evolution"); ax_dip.legend(); ax_dip.grid(True, alpha=0.3)
    ax_amp.set_xlabel("Time (yr)"); ax_amp.set_ylabel("Peak Br (G)")
    ax_amp.set_title("Peak Amplitude Decay"); ax_amp.legend(); ax_amp.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


def plot_numerical_performance(results_dict, lat_deg,
                                save_path="numerical_performance.png"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Numerical Performance Dashboard", fontsize=13)
    colors = {1: '#1f77b4', 2: '#d62728'}
    labels = {1: 'Case 1 (lambda0=0)', 2: 'Case 2 (lambda0=12)'}

    ax = axes[0, 0]
    for cid, res in results_dict.items():
        fc = res['diag']['flux']
        sd = fc.get('signed_drift_pct',
                     np.abs(fc['signed_flux'] - fc['signed_flux'][0])
                     / (np.abs(fc['signed_flux'][0]) + 1e-30) * 100)
        ax.plot(fc['t_yr'], sd, color=colors[cid], lw=2, label=labels.get(cid))
    ax.axhline(1.0, color='r', ls='--', lw=1.5, label='1% target')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Signed flux drift (% of |Phi_0|)")
    ax.set_title("Advection Conservation Quality\n(relative to unsigned initial flux)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for cid, res in results_dict.items():
        ct    = res['diag']['centroid']
        rates = np.abs(ct['pos_migration_rate_deg_yr'])
        t_mid = 0.5 * (ct['t_yr'][1:] + ct['t_yr'][:-1])
        ax.plot(t_mid, rates, color=colors[cid], lw=1, alpha=0.3,
                label=labels.get(cid) + ' (raw)')
        # FIX (Failure 3): plot cleaned rate as the primary line
        cleaned = ct.get('pos_rate_cleaned_deg_yr', rates)
        ax.plot(t_mid, np.abs(cleaned), color=colors[cid], lw=2,
                label=labels.get(cid) + ' (cleaned)')
    ax.axhline(15.0, color='gray', ls=':', lw=1, alpha=0.5, label='15 deg/yr threshold')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Migration rate (deg/yr)")
    ax.set_title("Poleward Migration Rate\n(cleaned: artefact epochs >15 deg/yr masked)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for cid, res in results_dict.items():
        pe = res['diag']['profile']
        fwhm_pos = pe['fwhm_pos_deg'].copy()
        valid_pos = np.isfinite(fwhm_pos)
        if valid_pos.any():
            ax.plot(pe['t_yr'][valid_pos], fwhm_pos[valid_pos],
                    color=colors[cid], lw=2, label=labels.get(cid) + ' (valid)')
    sigma0_deg = 3.51
    if results_dict:
        first_res = list(results_dict.values())[0]
        t_arr     = first_res['diag']['profile']['t_yr']
        fwhm_th   = diffusion_broadening_theory(t_arr, sigma0_deg, ETA_DEFAULT, R_SUN) * 2.355
        ax.plot(t_arr, fwhm_th, 'k--', lw=1.5, label='Theory (planar approx)')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("FWHM (deg)")
    ax.set_title("Diffusion Broadening vs Theory\n(NaN = multi-modal profile, see DIAG-1)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for cid, res in results_dict.items():
        dd = res['diag']['dipole']
        ax.plot(dd['t_yr'], dd['dipole_G'], color=colors[cid], lw=2.5, label=labels.get(cid))
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Axial dipole (G)")
    ax.set_title("Dipole Moment (Solar Cycle Predictor)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


def plot_bmr_initial_conditions(grid, cfg, save_path="bmr_initial.png"):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
    fig.suptitle("Initial Conditions: Bipolar Magnetic Regions", fontsize=14)

    cases_info = [(0, 'Case 1: Equatorial BMR (lambda0=0)'),
                  (12, 'Case 2: Mid-latitude BMR (lambda0=12)')]

    for row, (lat0, title) in enumerate(cases_info):
        B2d  = bmr_2d(grid.LAT, grid.LON, lat0_deg=lat0, cfg=cfg)
        B1d  = bmr_longitude_averaged(grid.lat_deg, lat0_deg=lat0, cfg=cfg)
        vmax = max(float(abs(_safe_vmax(B2d))), 1e-6)

        ax = fig.add_subplot(gs[row, 0])
        il  = np.searchsorted(grid.lon_deg, 40)
        ih  = np.searchsorted(grid.lon_deg, 160)
        img = ax.pcolormesh(grid.lon_deg[il:ih], grid.lat_deg, B2d[:, il:ih],
                             cmap='RdBu_r', norm=TwoSlopeNorm(0, vmin=-vmax, vmax=vmax),
                             shading='auto')
        ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Latitude (deg)")
        ax.set_title(f"2D BMR field\n{title}")
        plt.colorbar(img, ax=ax, shrink=0.8, label='Br (G)')

        ax = fig.add_subplot(gs[row, 1])
        ax.plot(B1d, grid.lat_deg, lw=2.5, color='#1f77b4')
        ax.axvline(0, color='r', lw=0.8, ls='--'); ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.fill_betweenx(grid.lat_deg, B1d, where=B1d > 0, alpha=0.3, color='#1f77b4', label='+polarity')
        ax.fill_betweenx(grid.lat_deg, B1d, where=B1d < 0, alpha=0.3, color='#d62728', label='-polarity')
        ax.set_xlabel("Br (G)"); ax.set_ylabel("Latitude (deg)")
        ax.set_title(f"Lon-averaged profile\n{title}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[row, 2])
        peak_lat_idx = np.argmax(np.abs(B1d))
        peak_lat     = grid.lat_deg[peak_lat_idx]
        ax.plot(grid.lon_deg, B2d[peak_lat_idx, :], lw=2, color='#2ca02c')
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.fill_between(grid.lon_deg, B2d[peak_lat_idx, :],
                         where=B2d[peak_lat_idx, :] > 0, alpha=0.3, color='#1f77b4')
        ax.fill_between(grid.lon_deg, B2d[peak_lat_idx, :],
                         where=B2d[peak_lat_idx, :] < 0, alpha=0.3, color='#d62728')
        ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Br (G)")
        ax.set_title(f"Longitudinal slice at lat={peak_lat:.0f} deg\n{title}")
        ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


def plot_all_diagnostics_extended(diag_dict, cfg, lat_deg, case_label,
                                   B_snaps, t_yr, save_path="diagnostics_extended.png"):
    """
    Extended 3×3 diagnostic panel.
    DIAG-2/3 fix: Signed-flux drift label updated; polarity-cancellation subplot
    clearly labelled as physical process.
    DIAG-1 fix: FWHM subplots only plot valid (finite) epochs.
    """
    fc  = diag_dict['flux']; ct = diag_dict['centroid']
    pe  = diag_dict['profile']; dd = diag_dict['dipole']
    fig, axes = plt.subplots(3, 3, figsize=(17, 14))
    fig.suptitle(f"Extended Diagnostic Panel — {case_label}", fontsize=14)

    ax = axes[0, 0]
    ax.semilogy(fc['t_yr'], fc['pos_flux'], _STYLE['pos_color'], lw=2, label='Phi+')
    ax.semilogy(fc['t_yr'], fc['neg_flux'], _STYLE['neg_color'], lw=2, label='Phi-')
    ax.semilogy(fc['t_yr'], fc['expected'], 'k--', lw=1.2,
                 label=f'exp(-t/{cfg.tau_yr:.1f}yr)')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("|Phi|")
    ax.set_title("Unsigned Flux Budget"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # DIAG-2 fix: renamed label
    ax = axes[0, 1]
    sd = fc.get('signed_drift_pct',
                 np.abs(fc['signed_flux'] - fc['signed_flux'][0])
                 / (float(np.max(np.abs(fc['signed_flux']))) + 1e-30) * 100)
    ax.plot(fc['t_yr'], sd, 'teal', lw=2)
    ax.axhline(1.0, color='r', ls='--', label='1% threshold')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Drift (% of |Phi_0|)")
    ax.set_title(f"Signed Flux Drift (% of |Phi_0|)\nmax={fc.get('max_signed_drift_pct',0):.2f}%")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    B_2d = np.array(B_snaps); vmax = _safe_vmax(B_2d, percentile=98)
    img  = ax.pcolormesh(t_yr, lat_deg, B_2d.T, cmap='RdBu_r',
                          norm=TwoSlopeNorm(0, vmin=-vmax, vmax=vmax),
                          shading='auto', rasterized=True)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    plt.colorbar(img, ax=ax, shrink=0.8)
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Lat (deg)"); ax.set_title("Butterfly Diagram")

    ax = axes[1, 0]
    ax.plot(ct['t_yr'], ct['pos_centroid_deg'], _STYLE['pos_color'], lw=2.5, label='Positive')
    ax.plot(ct['t_yr'], ct['neg_centroid_deg'], _STYLE['neg_color'], lw=2.5, label='Negative')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Centroid lambda (deg)")
    ax.set_title(f"Poleward Migration\n(avg rate: {ct['avg_pos_rate']:.1f} deg/yr)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # DIAG-8: note about centroid rate artefact
    ax = axes[1, 1]
    if len(ct['t_yr']) > 1:
        t_mid = 0.5*(ct['t_yr'][1:]+ct['t_yr'][:-1])
        ax.plot(t_mid, ct['pos_migration_rate_deg_yr'], _STYLE['pos_color'], lw=2, label='+rate')
        ax.plot(t_mid, ct['neg_migration_rate_deg_yr'], _STYLE['neg_color'], lw=2, label='-rate')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Rate (deg/yr)")
    ax.set_title("Instantaneous Migration Rate\n(early spike = artefact, see DIAG-8)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.semilogy(pe['t_yr'], pe['peak_pos'], _STYLE['pos_color'], lw=2.5, label='Peak +')
    ax.semilogy(pe['t_yr'], np.abs(pe['peak_neg']), _STYLE['neg_color'], lw=2.5, label='Peak -')
    t_fine = np.linspace(pe['t_yr'][0], pe['t_yr'][-1], 200)
    ax.semilogy(t_fine, pe['peak_pos'][0]*np.exp(-t_fine/cfg.tau_yr), 'k--', lw=1.2,
                 label=f'exp(-t/{cfg.tau_yr:.1f}yr)')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("Peak |Br| (G)")
    ax.set_title("Peak Amplitude Decay"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # DIAG-1 fix: only plot finite FWHM epochs
    ax = axes[2, 0]
    fwhm_pos = pe['fwhm_pos_deg'].copy(); fwhm_neg = pe['fwhm_neg_deg'].copy()
    vp = np.isfinite(fwhm_pos); vn = np.isfinite(fwhm_neg)
    if vp.any():
        ax.plot(pe['t_yr'][vp], fwhm_pos[vp], _STYLE['pos_color'], lw=2.5, label='FWHM + (valid)')
    if vn.any():
        ax.plot(pe['t_yr'][vn], fwhm_neg[vn], _STYLE['neg_color'], lw=2.5, label='FWHM - (valid)')
    sigma0_deg = 3.51
    fwhm_th = diffusion_broadening_theory(pe['t_yr'], sigma0_deg, ETA_DEFAULT, R_SUN) * 2.355
    ax.plot(pe['t_yr'], fwhm_th, 'k--', lw=1.2, label='Theory (planar)')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("FWHM (deg)")
    ax.set_title("Profile Broadening vs Theory\n(NaN = multi-modal, see DIAG-1)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(dd['t_yr'], dd['dipole_G'], 'purple', lw=2.5)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.fill_between(dd['t_yr'], dd['dipole_G'], alpha=0.2, color='purple')
    ax.set_xlabel("Time (yr)"); ax.set_ylabel("D(t) (G)")
    ax.set_title(f"Axial Dipole Moment\n(final: {dd['dipole_G'][-1]:.4f} G)")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    idx_set = sorted(set([0, len(t_yr)//4, len(t_yr)//2, 3*len(t_yr)//4, len(t_yr)-1]))
    cm2 = plt.cm.viridis
    for rank, i in enumerate(idx_set):
        color = cm2(rank / max(len(idx_set)-1, 1))
        ax.plot(lat_deg, B_snaps[i], color=color, lw=1.8, label=f"t={t_yr[i]:.1f} yr")
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("Latitude (deg)"); ax.set_ylabel("Br (G)")
    ax.set_title("Profile Snapshots"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=_STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# 9. AI ANALYSER  (opt-in via --ai)
# ══════════════════════════════════════════════════════════════════════════

CLAUDE_MODEL = "claude-sonnet-4-20250514"

def _to_serializable(obj):
    if isinstance(obj, np.ndarray):    return obj.tolist()
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, dict):          return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_serializable(i) for i in obj]
    return obj


class SFTAnalyzer:
    SYSTEM = (
        "You are an expert solar physicist specialising in Surface Flux Transport modelling. "
        "Analyse numerical SFT results quantitatively: cite specific values, connect to the "
        "broader solar-physics context, and reference Athalathil 2024, Baumann 2006, "
        "Petrovay & Talafha 2019, Yeates 2023, Cameron & Schüssler 2015. "
        "Format with clear markdown headers and bullet points."
    )

    def __init__(self):
        self.client = anthropic.Anthropic() if _ANTHROPIC_OK else None

    def _prompt(self, diag: dict, params: dict, label: str = '') -> str:
        fc = diag.get('flux', {}); ct = diag.get('centroid', {})
        pd_ = diag.get('profile', {}); dd = diag.get('dipole', {})
        lt_err = fc.get('late_time_decay_err_pct', float('nan'))
        lt_note = (f"{lt_err:.2f}%" if not np.isnan(lt_err)
                   else "N/A (run < 1.5 tau)")
        return (
            f"# SFT Simulation Analysis — {label}\n"
            f"eta={params.get('eta',500e6)/1e6:.0f} km2/s  "
            f"u0={params.get('u0',12.5):.1f} m/s  "
            f"tau={params.get('tau',5*3.156e7)/(365.25*86400):.1f} yr\n\n"
            f"**Diagnostics:**\n"
            f"- Signed flux drift (max) = {fc.get('max_signed_drift_pct',0):.3f} % of |Phi_0|\n"
            f"- Polarity cancellation (max) = {fc.get('max_polarity_cancellation_pct',0):.1f}%"
            f"  [physical cross-equatorial cancellation, not scheme error]\n"
            f"- Late-time numerical decay error = {lt_note}\n"
            f"- Avg poleward migration = {ct.get('avg_pos_rate',0):.2f} deg/yr\n"
            f"- Peak amplitude: "
            f"{float(pd_.get('peak_pos',[0])[0]):.3f} -> "
            f"{float(pd_.get('peak_pos',[0])[-1]):.4f} G\n"
            f"- Axial dipole: "
            f"{float(dd.get('dipole_G',[0])[0]):.4f} -> "
            f"{float(dd.get('dipole_G',[0])[-1]):.4f} G\n\n"
            "Please analyse: transport mechanisms, numerical performance vs literature, "
            "solar cycle implications, and follow-up recommendations."
        )

    def _call(self, prompt: str, max_tokens: int = 2000) -> str:
        if not _ANTHROPIC_OK or not self.client:
            return "AI analysis unavailable — install 'anthropic' package."
        try:
            r = self.client.messages.create(
                model=CLAUDE_MODEL, max_tokens=max_tokens,
                system=self.SYSTEM,
                messages=[{"role": "user", "content": prompt}])
            return r.content[0].text
        except Exception as e:
            return f"AI call failed: {e}"

    def analyze_simulation(self, diag, params, label='') -> str:
        print(f"  [AI] Analysing {label}...")
        return self._call(self._prompt(diag, params, label))

    def compare_cases(self, diag_list, labels, params_list) -> str:
        prompt = "# Multi-Case Comparison\n\n" + "\n---\n".join(
            self._prompt(d, p, l) for d, l, p in zip(diag_list, labels, params_list))
        return self._call(prompt, max_tokens=2500)

    def sensitivity_interpretation(self, sens_results, lat_deg) -> str:
        payload = json.dumps(_to_serializable(sens_results), default=str)[:3000]
        return self._call(f"# SFT Parameter Sensitivity\n{payload}", max_tokens=2000)

    def generate_forecast_narrative(self, dipole_t_yr, dipole_G,
                                     current_cycle: int = 25) -> str:
        prompt = (
            f"# Solar Cycle Forecast — Cycle {current_cycle}\n"
            f"Dipole {float(dipole_G[0]):.4f} -> {float(dipole_G[-1]):.4f} G "
            f"over {float(dipole_t_yr[-1]):.1f} yr.\n"
            "Estimate next cycle amplitude using the polar-field precursor method "
            "and compare with SC25 observations."
        )
        return self._call(prompt, max_tokens=1500)


# ══════════════════════════════════════════════════════════════════════════
# 10. PARAMETER STUDIES
# ══════════════════════════════════════════════════════════════════════════

def sensitivity_study(B0, lat_deg, cfg_base: Optional[SFTConfig] = None,
                      t_eval_yr=2.0,
                      eta_values=None, u0_values=None, tau_values=None) -> dict:
    if cfg_base is None:
        cfg_base = SFTConfig(dlat=float(lat_deg[1] - lat_deg[0]))
    lat_rad = np.deg2rad(lat_deg)

    def _run(c):
        g   = SFTGrid(c)
        B0i = _interp1d(lat_rad, B0, bounds_error=False, fill_value=0.0)(g.lat_rad)
        solver = RKIMEX1D(g, c)
        t_s, B_s = solver.evolve(B0i, t_eval_yr,
                                  save_every_days=int(365.25 * t_eval_yr), verbose=False)
        idx = np.argmin(np.abs(t_s - t_eval_yr))
        return B_s[idx], g.lat_rad

    results = {}
    for eta in (eta_values or [250e6, 500e6, 1000e6]):
        c = SFTConfig(eta=eta, u0=cfg_base.u0, tau=cfg_base.tau, dlat=cfg_base.dlat)
        Bf, lr = _run(c)
        results.setdefault('eta_sweep', []).append({
            'eta': eta, 'eta_label': f'eta={eta/1e6:.0f} km2/s',
            'B': Bf, 'peak': float(np.max(Bf)),
            'fwhm': profile_fwhm(Bf, lr, 'positive'),
            'centroid_pos': flux_centroid(Bf, lr, 'positive'),
        })
    for u0 in (u0_values or [6.25, 12.5, 25.0]):
        c = SFTConfig(eta=cfg_base.eta, u0=u0, tau=cfg_base.tau, dlat=cfg_base.dlat)
        Bf, lr = _run(c)
        results.setdefault('u0_sweep', []).append({
            'u0': u0, 'u0_label': f'u0={u0} m/s',
            'B': Bf, 'peak': float(np.max(Bf)),
            'fwhm': profile_fwhm(Bf, lr, 'positive'),
            'centroid_pos': flux_centroid(Bf, lr, 'positive'),
        })
    for tau_yr in (tau_values or [2.5, 5.0, 10.0]):
        c = SFTConfig(eta=cfg_base.eta, u0=cfg_base.u0,
                      tau=tau_yr * 365.25 * 86400, dlat=cfg_base.dlat)
        Bf, lr = _run(c)
        results.setdefault('tau_sweep', []).append({
            'tau_yr': tau_yr, 'tau_label': f'tau={tau_yr} yr',
            'B': Bf, 'peak': float(np.max(Bf)),
            'fwhm': profile_fwhm(Bf, lr, 'positive'),
        })
    return results


def convergence_test(B0_fine, lat_deg_fine, resolutions=(2.0, 1.0, 0.5),
                     t_end_yr=1.0, cfg_base: Optional[SFTConfig] = None) -> dict:
    if cfg_base is None:
        cfg_base = SFTConfig()
    lat_rad_fine = np.deg2rad(lat_deg_fine)
    results = {}
    for dres in sorted(resolutions):
        c  = SFTConfig(eta=cfg_base.eta, u0=cfg_base.u0,
                       tau=cfg_base.tau, dlat=dres)
        g  = SFTGrid(c)
        B0 = _interp1d(lat_rad_fine, B0_fine,
                       bounds_error=False, fill_value=0.0)(g.lat_rad)
        solver = RKIMEX1D(g, c)
        _, B_s = solver.evolve(B0, t_end_yr,
                               save_every_days=int(365.25 * t_end_yr), verbose=False)
        results[dres] = {'lat_deg': g.lat_deg, 'lat_rad': g.lat_rad, 'B': B_s[-1]}

    finest      = min(resolutions)
    fine_interp = _interp1d(results[finest]['lat_rad'], results[finest]['B'],
                             bounds_error=False, fill_value=0.0)
    for dres in resolutions:
        lr = results[dres]['lat_rad']
        L1 = float(np.mean(np.abs(results[dres]['B'] - fine_interp(lr))))
        results[dres]['L1_vs_finest'] = L1
    return results


# ══════════════════════════════════════════════════════════════════════════
# 11. CASE DEFINITIONS & PIPELINE
# ══════════════════════════════════════════════════════════════════════════

_CASES = {
    1: {'lat0_deg':  0.0, 'lon0_deg': 100.0,
        'label': 'Case 1: Equatorial BMR (lambda0=0 deg)',    'short': 'case1'},
    2: {'lat0_deg': 12.0, 'lon0_deg': 100.0,
        'label': 'Case 2: Mid-latitude BMR (lambda0=12 deg)', 'short': 'case2'},
}


def run_case(case_id: int, grid: SFTGrid, cfg: SFTConfig,
             t_end_years: float = 6.0,
             save_interval_days: float = 30.0,
             verbose: bool = True) -> dict:
    case = _CASES[case_id]
    print(f"\n{'='*60}\n  Running {case['label']}\n  Duration: {t_end_years} yr\n{'='*60}")

    lat0          = case['lat0_deg']
    B_2d_init     = bmr_2d(grid.LAT, grid.LON, lat0_deg=lat0, cfg=cfg)
    B_1d_init     = bmr_longitude_averaged(grid.lat_deg, lat0_deg=lat0, cfg=cfg)

    solver = RKIMEX1D(grid, cfg)
    if verbose:
        stability_report(grid, cfg)

    t0 = time.time()
    t_yr, B_snaps = solver.evolve(B_1d_init, t_end_years,
                                   save_every_days=save_interval_days,
                                   verbose=verbose)
    print(f"  Elapsed: {time.time()-t0:.1f} s  "
          f"({int(t_end_years * 365.25 * 86400 / cfg.dt)} steps)")

    diag_obj    = SFTDiagnostics(t_yr, B_snaps, grid, cfg)
    diag_result = diag_obj.summary(label=case['label'])

    return {
        'case_id':      case_id,
        'label':        case['label'],
        'short':        case['short'],
        't_yr':         t_yr,
        'B_snaps_1d':   B_snaps,
        'B_init_2d':    B_2d_init,
        'lon_avg_init': B_1d_init,
        'B_1d_init':    B_1d_init,
        'diag':         diag_result,
        'params': {'eta': cfg.eta, 'u0': cfg.u0, 'tau': cfg.tau,
                   'dt': cfg.dt, 'dlat': cfg.dlat,
                   'scheme': 'SSP2(2,2,2) RK-IMEX + van Leer TVD (round-6)'},
        'lat0_deg': lat0,
    }


def compare_schemes(grid: SFTGrid, cfg: SFTConfig,
                    lat0_deg: float = 0.0, t_end_yr: float = 2.0):
    B0 = bmr_longitude_averaged(grid.lat_deg, lat0_deg=lat0_deg, cfg=cfg)

    print("\n[Scheme comparison]")
    t0 = time.time()
    t_i, B_i = RKIMEX1D(grid, cfg).evolve(B0, t_end_yr, verbose=False)
    print(f"  RK-IMEX:  {time.time()-t0:.2f} s  peak={np.max(B_i[-1]):.5f} G")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(grid.lat_deg, B0,       'k--', lw=1.5, label='t=0')
    axes[0].plot(grid.lat_deg, B_i[-1],  'b-',  lw=2,   label=f'RK-IMEX t={t_end_yr} yr')
    axes[0].set_xlabel("Lat (deg)"); axes[0].set_ylabel("Br (G)")
    axes[0].set_title("Final profile"); axes[0].legend()
    axes[1].semilogy(t_i, np.max(np.abs(B_i), axis=1), 'b-', lw=2)
    axes[1].set_xlabel("Time (yr)"); axes[1].set_ylabel("Peak |Br| (G)")
    axes[1].set_title("Amplitude evolution")
    plt.tight_layout()
    plt.savefig("scheme_comparison.png", dpi=150, bbox_inches='tight')
    print("  [saved] scheme_comparison.png")
    return {'imex': {'t': t_i, 'B': B_i}}


def run_2d_simulation(grid: SFTGrid, cfg: SFTConfig,
                      lat0_deg: float = 12.0,
                      t_end_years: float = 2.0, verbose: bool = True) -> dict:
    print(f"\n{'='*60}\n  2D SFT  lat0={lat0_deg} deg  {t_end_years} yr\n{'='*60}")
    cfg2d  = SFTConfig(eta=cfg.eta, u0=cfg.u0, tau=cfg.tau,
                       dt=cfg.dt, dlat=cfg.dlat, dlon=cfg.dlon,
                       include_diff_rot=True)
    grid2d = SFTGrid(cfg2d)
    B_init = bmr_2d(grid2d.LAT, grid2d.LON, lat0_deg=lat0_deg, cfg=cfg2d)

    t_yr, B_snaps_2d = RKIMEX2D(grid2d, cfg2d).evolve(
        B_init, t_end_years, save_every_days=60, verbose=verbose)

    vmax = max(_safe_vmax(B_init, percentile=99) * 0.1, 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    idxs = [0, len(B_snaps_2d) // 2, -1]
    lbls = ['t=0', f't={t_end_years/2:.1f} yr', f't={t_end_years:.1f} yr']
    for ax, idx, lbl in zip(axes, idxs, lbls):
        img = ax.pcolormesh(grid2d.lon_deg, grid2d.lat_deg, B_snaps_2d[idx],
                             cmap='RdBu_r',
                             norm=TwoSlopeNorm(0, vmin=-vmax, vmax=vmax),
                             shading='auto')
        ax.set_title(f"2D Field {lbl}")
        ax.set_xlabel("Lon (deg)"); ax.set_ylabel("Lat (deg)")
        _add_colorbar(fig, ax, img)
    plt.suptitle(f"2D SFT (lat0={lat0_deg} deg, diff_rot=True)")
    plt.tight_layout()
    plt.savefig("sft_2d_evolution.png", dpi=150, bbox_inches='tight')
    print("  [saved] sft_2d_evolution.png")
    return {'t_yr': t_yr, 'B_snaps_2d': B_snaps_2d}


# ══════════════════════════════════════════════════════════════════════════
# 12. CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SFT Master Pipeline (round-6, SSP2(2,2,2) + Neumann BC + DIAG fixes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--quick',           action='store_true',
                        help='1 yr, 2 deg grid — fast smoke test')
    parser.add_argument('--case',            type=int, default=0, choices=[0, 1, 2],
                        help='0 = both cases')
    parser.add_argument('--ai',              action='store_true',
                        help='Enable Claude AI analysis (off by default)')
    parser.add_argument('--2d',              dest='run_2d', action='store_true',
                        help='Run full 2D simulation with differential rotation')
    parser.add_argument('--sensitivity',     action='store_true')
    parser.add_argument('--convergence',     action='store_true')
    parser.add_argument('--validation',      action='store_true')
    parser.add_argument('--compare-schemes', dest='compare_schemes', action='store_true')
    parser.add_argument('--param-set',       type=str, default='athalathil2024',
                        choices=list(PARAM_SETS.keys()))
    parser.add_argument('--t-end',           type=float, default=6.0,
                        help='Simulation duration (years)')
    args = parser.parse_args()

    print("\n" + "*"*60)
    print("   SURFACE FLUX TRANSPORT PIPELINE  (round-6)")
    print("   Numerics: SSP2(2,2,2)  gamma=1-1/sqrt(2), Neumann BC")
    print("   Diagnostics (round-6):")
    print("   DIAG-1  FWHM validity guards (peak/bg + unimodality)")
    print("   DIAG-2  Signed drift normalised by unsigned total flux")
    print("   DIAG-3  Polarity cancellation = physical, not scheme error")
    print("   DIAG-4  Late-time decay error metric added")
    print("   DIAG-5  Summary prints abs drift + late-time err")
    print("   DIAG-6  Plot labels corrected for cancellation subplot")
    print("   DIAG-7  Asymmetry index = 1.0 annotated as correct dipole")
    print("   DIAG-8  Centroid artefact warning added")
    print("*"*60)

    dlat  = 2.0 if args.quick else 1.0
    t_end = 1.0 if args.quick else args.t_end
    cfg   = SFTConfig.from_param_set(args.param_set, dlat=dlat)
    grid  = SFTGrid(cfg)
    print(f"\n  Parameter set : {args.param_set}")
    print(f"  {PARAM_SETS[args.param_set]['description']}")
    print(f"  Grid          : {grid.Nlat} lat pts  |  dt={cfg.dt} s  |  "
          f"eta={cfg.eta_km2_s:.0f} km2/s  |  tau={cfg.tau_yr:.1f} yr")

    if args.validation:
        run_validation_tests(dlat=dlat, verbose=True)

    print("\n  Generating initial-condition panel...")
    plot_bmr_initial_conditions(grid, cfg, save_path="bmr_initial.png")
    plot_meridional_flow_profile(cfg, save_path="meridional_flow.png")

    cases_to_run = [1, 2] if args.case == 0 else [args.case]
    results = {}
    for cid in cases_to_run:
        results[cid] = run_case(cid, grid, cfg, t_end_years=t_end)

    if 1 in results and 2 in results:
        r1, r2 = results[1], results[2]
        print("\n  Generating Figure 5 reproduction...")
        plot_figure5_reproduction(
            r1['B_init_2d'], r1['lon_avg_init'], r1['t_yr'], r1['B_snaps_1d'],
            r2['B_init_2d'], r2['lon_avg_init'], r2['t_yr'], r2['B_snaps_1d'],
            grid.lat_deg, grid.lon_deg,
            save_path="figure5_reproduction.png")

    for cid, res in results.items():
        print(f"\n  Generating plots for Case {cid}...")
        t_yr    = res['t_yr']
        B_snaps = res['B_snaps_1d']
        diag    = res['diag']
        label   = res['label']

        plot_diagnostics(diag, grid.lat_deg,
                          save_path=f"diagnostics_case{cid}.png")
        plot_all_diagnostics_extended(
            diag, cfg, grid.lat_deg, label, B_snaps, t_yr,
            save_path=f"diagnostics_extended_case{cid}.png")

        snap_idx = sorted(set([0] + [
            int(np.argmin(np.abs(t_yr - t)))
            for t in [1.5, 3.0, 6.0]
            if np.min(np.abs(t_yr - t)) < 1.0
        ]))
        plot_diffusion_evolution(
            t_yr[snap_idx], [B_snaps[i] for i in snap_idx], grid.lat_deg,
            save_path=f"diffusion_case{cid}.png")
        plot_detailed_butterfly(t_yr, B_snaps, grid.lat_deg,
            save_path=f"butterfly_case{cid}.png")
        plot_polar_field_evolution(t_yr, B_snaps, grid.lat_deg,
            save_path=f"polar_field_case{cid}.png")
        plot_flux_budget(diag, cfg, save_path=f"flux_budget_case{cid}.png")
        plot_dipole_quadrupole(diag, save_path=f"multipole_case{cid}.png")
        stride = max(1, len(t_yr) // 10)
        plot_energy_spectrum(B_snaps[::stride], grid.lat_deg, t_yr[::stride],
            save_path=f"energy_spectrum_case{cid}.png")

    if len(results) > 1:
        print("\n  Generating case comparison panel...")
        plot_case_comparison(results, grid.lat_deg, save_path="case_comparison.png")
        plot_numerical_performance(results, grid.lat_deg,
                                    save_path="numerical_performance.png")

    sens = None
    if args.sensitivity:
        print("\n  Running parameter sensitivity study...")
        cid  = list(results.keys())[0]
        sens = sensitivity_study(results[cid]['B_1d_init'], grid.lat_deg, cfg_base=cfg)
        plot_sensitivity(sens, grid.lat_deg, save_path="sensitivity.png")

    conv = None
    if args.convergence:
        print("\n  Running grid convergence test...")
        cid         = list(results.keys())[0]
        resolutions = (4.0, 2.0, 1.0) if args.quick else (2.0, 1.0, 0.5)
        conv        = convergence_test(
            results[cid]['B_1d_init'], grid.lat_deg,
            resolutions=resolutions,
            t_end_yr=min(t_end, 1.0), cfg_base=cfg)
        plot_convergence(conv, t_end_yr=min(t_end, 1.0), save_path="convergence.png")
        print("\n  Convergence results:")
        for dres in sorted(conv.keys()):
            print(f"    dlat={dres} deg  L1={conv[dres]['L1_vs_finest']:.3e}")

    if args.compare_schemes:
        compare_schemes(grid, cfg)

    if args.run_2d:
        run_2d_simulation(grid, cfg, lat0_deg=12.0, t_end_years=min(t_end, 2.0))

    if args.ai:
        print("\n  Running AI analysis with Claude...")
        analyzer   = SFTAnalyzer()
        ai_reports = {}

        for cid, res in results.items():
            sim_p  = {**res['params'], 'lat0_deg': res['lat0_deg']}
            report = analyzer.analyze_simulation(res['diag'], sim_p, label=res['label'])
            ai_reports[cid] = report
            print(f"\n{'~'*60}\n  AI ANALYSIS — {res['label']}\n{'~'*60}")
            print(report)

        if 1 in results and 2 in results:
            cr = analyzer.compare_cases(
                [results[1]['diag'], results[2]['diag']],
                [results[1]['label'], results[2]['label']],
                [{**results[1]['params'], 'lat0_deg': 0},
                 {**results[2]['params'], 'lat0_deg': 12}])
            print(f"\n{'~'*60}\n  COMPARATIVE ANALYSIS\n{'~'*60}\n{cr}")

        if sens:
            print(f"\n{'~'*60}\n  SENSITIVITY INTERPRETATION\n{'~'*60}")
            print(analyzer.sensitivity_interpretation(sens, grid.lat_deg))

        if 2 in results:
            dd = results[2]['diag']['dipole']
            print(f"\n{'~'*60}\n  CYCLE FORECAST\n{'~'*60}")
            print(analyzer.generate_forecast_narrative(dd['t_yr'], dd['dipole_G']))

        with open("ai_analysis_report.md", 'w') as f:
            for cid, rep in ai_reports.items():
                f.write(f"# Case {cid}: {results[cid]['label']}\n\n{rep}\n\n---\n\n")
        print("  [saved] ai_analysis_report.md")

    # ── Output manifest ───────────────────────────────────────────────────
    always_outputs = ["bmr_initial.png", "meridional_flow.png"]
    if 1 in results and 2 in results:
        always_outputs += ["figure5_reproduction.png",
                           "case_comparison.png", "numerical_performance.png"]
    per_case_outputs = []
    for cid in results:
        per_case_outputs += [
            f"diagnostics_case{cid}.png", f"diagnostics_extended_case{cid}.png",
            f"diffusion_case{cid}.png",   f"butterfly_case{cid}.png",
            f"polar_field_case{cid}.png", f"flux_budget_case{cid}.png",
            f"multipole_case{cid}.png",   f"energy_spectrum_case{cid}.png",
        ]
    optional_outputs = []
    if args.sensitivity:     optional_outputs.append("sensitivity.png")
    if args.convergence:     optional_outputs.append("convergence.png")
    if args.compare_schemes: optional_outputs.append("scheme_comparison.png")
    if args.run_2d:          optional_outputs.append("sft_2d_evolution.png")
    if args.ai:              optional_outputs.append("ai_analysis_report.md")

    all_outputs = always_outputs + per_case_outputs + optional_outputs
    print("\n" + "*"*60)
    print("  PIPELINE COMPLETE")
    print(f"  {len([f for f in all_outputs if os.path.exists(f)])}"
          f"/{len(all_outputs)} output files written:\n")
    for fn in all_outputs:
        tick = "+" if os.path.exists(fn) else "-"
        print(f"    {tick}  {fn}")
    print("*"*60 + "\n")


if __name__ == "__main__":
    main()