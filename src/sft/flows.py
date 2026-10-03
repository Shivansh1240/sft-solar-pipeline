"""Surface flow profiles: meridional circulation and differential rotation.

Sign conventions: latitude increases northward, so a poleward meridional flow
is positive in the north and negative in the south. Longitude increases in the
direction of solar rotation, so a positive rotation rate drifts features to
larger longitude.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .constants import deg_per_day_to_rad_per_s


def flow_sin2lat(lat: np.ndarray, u0: float) -> np.ndarray:
    """u(lat) = u0 sin(2 lat). Smooth, vanishes at the equator and the poles."""
    return u0 * np.sin(2.0 * lat)


def flow_sin_cutoff(lat: np.ndarray, u0: float, cutoff_deg: float = 75.0) -> np.ndarray:
    """u(lat) = u0 sin(pi lat / lat0) for |lat| < lat0, else 0.

    Continuous at lat0 (the sine reaches zero there), with peak speed at lat0 / 2.
    """
    lat0 = np.deg2rad(cutoff_deg)
    u = u0 * np.sin(np.pi * lat / lat0)
    return np.where(np.abs(lat) < lat0, u, 0.0)


FLOW_PROFILES: dict[str, Callable] = {
    "sin2lat": flow_sin2lat,
    "sin_cutoff": flow_sin_cutoff,
}


def meridional_flow(lat: np.ndarray, u0: float, profile="sin2lat", **kwargs) -> np.ndarray:
    """Evaluate a meridional flow profile in m/s.

    ``profile`` is a name from ``FLOW_PROFILES`` or a callable ``f(lat, u0, **kwargs)``.
    """
    fn = FLOW_PROFILES[profile] if isinstance(profile, str) else profile
    return np.asarray(fn(np.asarray(lat, dtype=float), u0, **kwargs), dtype=float)


# Snodgrass & Ulrich (1990), written in the Carrington frame as in the
# Yeates et al. (2023) SFT review: Omega = 0.18 - 2.396 sin^2 - 1.787 sin^4 deg/day.
SNODGRASS_ULRICH_1990 = (0.18, -2.396, -1.787)


def differential_rotation(lat: np.ndarray, coeffs=SNODGRASS_ULRICH_1990) -> np.ndarray:
    """Rotation rate relative to the Carrington frame [rad/s].

    Omega(lat) = A + B sin^2(lat) + C sin^4(lat), with (A, B, C) in deg/day.
    """
    a, b, c = coeffs
    s2 = np.sin(lat) ** 2
    return deg_per_day_to_rad_per_s(a + b * s2 + c * s2**2)
