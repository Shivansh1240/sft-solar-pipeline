"""Physical constants and unit conversions used throughout the package.

Internal units are SI (metres, seconds) except the magnetic field, which is in
Gauss, and magnetic flux, which is reported in Maxwell (G cm^2).
"""

import numpy as np

R_SUN_M = 6.96e8  # solar radius [m]
R_SUN_CM = R_SUN_M * 100.0  # solar radius [cm], used for fluxes in Mx

DAY_S = 86400.0
YEAR_S = 365.25 * DAY_S

DEG = np.pi / 180.0  # degrees -> radians


def deg_per_day_to_rad_per_s(value):
    """Convert an angular velocity from deg/day to rad/s."""
    return np.asarray(value) * DEG / DAY_S
