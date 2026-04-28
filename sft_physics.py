"""
sft_physics.py
==============
Physical models for Surface Flux Transport simulation.

Implements:
- Meridional circulation profiles (multiple variants)
- Differential rotation (Snodgrass 1983)
- Bipolar Magnetic Region (BMR) initialization (van Ballegooijen 1998 prescription)
- Source term models (Gaussian BMR, ring-doublet, observational)
- Solar constants and parameter sets from literature

References:
- van Ballegooijen et al. 1998 (BMR prescription)
- Snodgrass 1983 (differential rotation)
- Petrovay & Talafha 2019 (parameter optimization)
- Yeates et al. 2023 (review, classical SFT)
- Athalathil et al. 2024 (RK-IMEX reference paper)
"""

import numpy as np

# ─────────────────────────────────────────────
# Solar constants
# ─────────────────────────────────────────────
R_SUN      = 6.96e8          # m  – solar radius
TAU_DECAY  = 5.0 * 3.156e7   # s  – 5-year decay timescale (Baumann 2004)
ETA_DEFAULT = 500e6           # m²/s  = 500 km²/s  (turbulent diffusivity)
U0_DEFAULT  = 12.5            # m/s   – peak meridional flow (Hathaway 2003)
OMEGA_CAR   = 2 * np.pi / (25.38 * 86400)  # rad/s  – Carrington rotation


# ─────────────────────────────────────────────
# Meridional flow profiles
# ─────────────────────────────────────────────

def meridional_flow_vanBall(lat_rad, u0=U0_DEFAULT, lat0_deg=75.0):
    """
    van Ballegooijen et al. 1998 profile used in Athalathil 2024:
        u(λ) = u0 * sin(2λ)    |λ| < λ0
               0               otherwise
    Returns poleward (positive northward) speed in m/s.
    """
    lam0 = np.deg2rad(lat0_deg)
    u = u0 * np.sin(2.0 * lat_rad)
    u = np.where(np.abs(lat_rad) < lam0, u, 0.0)
    return u


def meridional_flow_schad(lat_rad, u0=U0_DEFAULT):
    """
    Schad et al. 2013 helioseismic single-cell profile:
        u(λ) = u0 * sin(π λ / λ_pole)
    Simpler profile sometimes used in full-cycle models.
    """
    u = u0 * np.sin(np.pi * lat_rad / np.deg2rad(90.0))
    return u


def meridional_flow_constant(lat_rad, u0=U0_DEFAULT):
    """
    Constant flow profile used for pure advection testing.
        u(λ) = u0
    """
    return np.full_like(lat_rad, u0, dtype=float)


def meridional_flow_multicell(lat_rad, u0=U0_DEFAULT, n_cells=1):
    """
    Optional multi-cell variant:
    For n_cells=1: standard single-cell (same as schad)
    For n_cells=2: double-cell with return flow near poles
    """
    if n_cells == 1:
        return meridional_flow_vanBall(lat_rad, u0)
    elif n_cells == 2:
        # Two-cell: poleward below ~55°, equatorward above
        u = u0 * np.sin(2.0 * lat_rad) * (1.0 - 2.0 * np.sin(lat_rad) ** 6)
        return u
    else:
        raise ValueError(f"n_cells must be 1 or 2, got {n_cells}")


# ─────────────────────────────────────────────
# Differential rotation
# ─────────────────────────────────────────────

def differential_rotation_snodgrass(lat_rad):
    """
    Snodgrass 1983 differential rotation in Carrington frame (rad/s).
    ω(λ) = A + B sin²(λ) + C sin⁴(λ)
    with A = 2.71e-6, B = -0.503e-6, C = -0.422e-6 rad/s
    Returns angular velocity (rad/s) relative to Carrington frame.
    """
    s = np.sin(lat_rad)
    omega = (2.71e-6 + (-0.503e-6) * s**2 + (-0.422e-6) * s**4)
    # Subtract Carrington rotation rate to get differential (relative)
    return omega - OMEGA_CAR


# ─────────────────────────────────────────────
# BMR initialization (van Ballegooijen 1998)
# ─────────────────────────────────────────────

def bmr_2d(lat_grid_deg, lon_grid_deg,
           lat0_deg=12.0, lon0_deg=100.0,
           delta_lat_sep_deg=6.0,
           sigma_lat_deg=3.51,
           sigma_lon_deg=12.0,
           B0=14.88,
           tilt_deg=0.0):
    """
    2D Gaussian bipolar magnetic region.

    Parameters
    ----------
    lat_grid_deg, lon_grid_deg : 2D arrays (degrees)
    lat0_deg   : center latitude (degrees)
    lon0_deg   : center longitude (degrees)
    delta_lat_sep_deg : latitudinal polarity separation (degrees)
    sigma_lat_deg     : latitudinal Gaussian width per polarity
    sigma_lon_deg     : longitudinal Gaussian width
    B0         : peak field strength (Gauss)
    tilt_deg   : Joy's law tilt angle (degrees, positive = leading polarity equatorward)

    Returns
    -------
    B : 2D array of radial field (Gauss)
    """
    # Positive polarity center (poleward)
    lat_pos = lat0_deg + delta_lat_sep_deg / 2.0
    lat_neg = lat0_deg - delta_lat_sep_deg / 2.0

    def gauss2d(lat_c, lon_c):
        dlat = lat_grid_deg - lat_c
        dlon = lon_grid_deg - lon_c
        # periodic longitude
        dlon = np.where(dlon >  180, dlon - 360, dlon)
        dlon = np.where(dlon < -180, dlon + 360, dlon)
        return np.exp(-0.5 * (dlat / sigma_lat_deg)**2
                      -0.5 * (dlon / sigma_lon_deg)**2)

    B = B0 * (gauss2d(lat_pos, lon0_deg) - gauss2d(lat_neg, lon0_deg))
    return B


def bmr_longitude_averaged(lat_deg, lat0_deg=12.0,
                             delta_lat_sep_deg=6.0,
                             sigma_lat_deg=3.51,
                             sigma_lon_deg=12.0,
                             B0=14.88):
    """
    Longitude-averaged BMR profile ⟨Br⟩(λ) at t=0.
    Analytically the Gaussian in longitude integrates to
    ∝ sigma_lon, so the amplitude scales accordingly.
    """
    # Amplitude after longitude averaging over 360°
    amp = B0 * (sigma_lon_deg / 360.0) * np.sqrt(2 * np.pi)

    lat_pos = lat0_deg + delta_lat_sep_deg / 2.0
    lat_neg = lat0_deg - delta_lat_sep_deg / 2.0

    pos_lobe = np.exp(-0.5 * ((lat_deg - lat_pos) / sigma_lat_deg)**2)
    neg_lobe = np.exp(-0.5 * ((lat_deg - lat_neg) / sigma_lat_deg)**2)

    return amp * (pos_lobe - neg_lobe)


def bmr_ensemble(lat_grid_deg, lon_grid_deg,
                 n_bmr=10, seed=42,
                 lat_range=(5, 35),
                 cycle_hemisphere='N'):
    """
    Generate an ensemble of realistic BMRs for a full-cycle simulation.
    Uses Joy's Law for tilt and Spörer's Law for latitude distribution.
    """
    rng = np.random.default_rng(seed)
    B_total = np.zeros_like(lat_grid_deg)

    for _ in range(n_bmr):
        lat0  = rng.uniform(*lat_range) * (1 if cycle_hemisphere == 'N' else -1)
        lon0  = rng.uniform(0, 360)
        tilt  = 0.5 * lat0 + rng.normal(0, 5)   # Joy's law ~0.5 * lat + scatter
        amp   = rng.uniform(8, 20)               # Gauss
        sep   = rng.uniform(4, 8)                # deg separation
        B_total += bmr_2d(lat_grid_deg, lon_grid_deg,
                          lat0_deg=lat0, lon0_deg=lon0,
                          delta_lat_sep_deg=sep,
                          B0=amp, tilt_deg=tilt)
    return B_total


# ─────────────────────────────────────────────
# Flux diagnostics helpers
# ─────────────────────────────────────────────

def total_flux(Br_lat_avg, lat_rad):
    """
    Compute total unsigned flux:
    Φ = ∫ ⟨Br⟩ sin(λ+90°) dλ  (surface area element on sphere)
    i.e. Φ = ∫ ⟨Br⟩ cos(λ) dλ
    """
    integrand = Br_lat_avg * np.cos(lat_rad)
    return np.trapz(integrand, lat_rad)


def signed_flux(Br_lat_avg, lat_rad):
    """Signed total flux (should be ~0 for closed flux systems)."""
    return np.trapz(Br_lat_avg * np.cos(lat_rad), lat_rad)


def axial_dipole_moment(Br_lat_avg, lat_rad):
    """
    Axial dipole moment D = (3/4π) ∫ ⟨Br⟩ cos(λ) cos(λ) dλ  (in Gauss)
    Key observable for solar cycle prediction.
    """
    integrand = Br_lat_avg * np.cos(lat_rad)**2
    return (3.0 / (4.0 * np.pi)) * 2 * np.pi * np.trapz(integrand, lat_rad)


def flux_centroid(Br_lat_avg, lat_rad, polarity='positive'):
    """
    Track centroid latitude of a given polarity.
    Returns centroid in degrees.
    """
    if polarity == 'positive':
        mask = Br_lat_avg > 0
    else:
        mask = Br_lat_avg < 0
    B_masked = np.abs(Br_lat_avg) * mask
    if B_masked.sum() == 0:
        return np.nan
    centroid_rad = np.trapz(lat_rad * B_masked, lat_rad) / np.trapz(B_masked, lat_rad)
    return np.rad2deg(centroid_rad)


# ─────────────────────────────────────────────
# Parameter sets from literature
# ─────────────────────────────────────────────

PARAM_SETS = {
    "athalathil2024": {
        "eta": 500e6, "u0": 12.5, "tau": 5 * 3.156e7,
        "lat0_cutoff": 75.0, "description": "Athalathil et al. 2024 reference params"
    },
    "cameron2010": {
        "eta": 250e6, "u0": 11.0, "tau": np.inf,
        "lat0_cutoff": 75.0, "description": "Cameron et al. 2010 (no decay)"
    },
    "petrovay2019_best": {
        "eta": 625e6, "u0": 12.5, "tau": 7.5 * 3.156e7,
        "lat0_cutoff": 75.0, "description": "Petrovay & Talafha 2019 best-fit"
    },
    "baumann2004": {
        "eta": 600e6, "u0": 11.0, "tau": 5 * 3.156e7,
        "lat0_cutoff": 75.0, "description": "Baumann et al. 2004"
    },
    "yeates2023": {
        "eta": 500e6, "u0": 12.5, "tau": 10 * 3.156e7,
        "lat0_cutoff": 75.0, "description": "Yeates et al. 2023 review defaults"
    }
}
