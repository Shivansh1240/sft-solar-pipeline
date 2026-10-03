"""Flux emergence: bipolar magnetic regions (BMRs), solar-cycle generators, catalogues.

Conventions
-----------
* ``flux_mx`` is the flux of *each* polarity, so a BMR adds zero net flux.
* ``tilt_deg`` > 0 means Joy's-law orientation in either hemisphere: the leading
  polarity sits closer to the equator than the trailing one.
* The leading polarity lies westward, i.e. at larger longitude (the direction of
  rotation). ``leading_sign`` is the sign of the leading polarity.
* Hale's law used here: in odd-numbered cycles the northern leading polarity is
  positive and the southern one negative; even cycles are reversed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .constants import R_SUN_CM
from .grid import Grid

JOY_AMPLITUDE_DEG = 32.1  # Stenflo & Kosovichev (2012): tilt = 32.1 deg * sin(latitude)


def hale_leading_sign(cycle: int, lat_deg: float) -> int:
    """Sign of the leading polarity for solar cycle ``cycle`` in the hemisphere of ``lat_deg``."""
    north = 1 if cycle % 2 == 1 else -1
    return north if lat_deg >= 0 else -north


def joy_tilt_deg(lat_deg, amplitude_deg: float = JOY_AMPLITUDE_DEG):
    """Mean tilt from Joy's law, positive in both hemispheres."""
    return amplitude_deg * np.sin(np.deg2rad(np.abs(lat_deg)))


@dataclass(frozen=True)
class BMR:
    """A bipolar magnetic region inserted instantaneously at ``time_yr``."""

    time_yr: float
    lat_deg: float
    flux_mx: float
    tilt_deg: float
    leading_sign: int
    lon_deg: float = 0.0
    sep_deg: float = 6.0  # heliocentric angle between the two polarity centres
    width_deg: float = 2.0  # Gaussian width (sigma) of each polarity
    member: int = 0  # column index when running an axisymmetric ensemble

    def polarity_centres(self):
        """Return ((lat, lon) of leading, (lat, lon) of trailing), in degrees."""
        h = 0.5 * self.sep_deg
        tilt = np.deg2rad(self.tilt_deg)
        hemi = 1.0 if self.lat_deg >= 0 else -1.0
        dlat = h * np.sin(tilt)
        dlon = h * np.cos(tilt) / max(np.cos(np.deg2rad(self.lat_deg)), 1e-3)
        lead = (self.lat_deg - hemi * dlat, self.lon_deg + dlon)
        trail = (self.lat_deg + hemi * dlat, self.lon_deg - dlon)
        return lead, trail

    def field(self, grid: Grid) -> np.ndarray:
        """Radial field of this BMR on ``grid`` [G]; net flux is exactly zero on the grid."""
        return bmr_field_2d(self, grid) if grid.is_2d else bmr_field_1d(self, grid)


def _normalise(shape: np.ndarray, cell_area_cm2: np.ndarray, flux_mx: float) -> np.ndarray:
    total = float(np.sum(shape * cell_area_cm2))
    if total <= 0.0:
        raise ValueError("BMR polarity is not resolved by the grid (zero integrated shape)")
    return shape * (flux_mx / total)


def bmr_field_1d(bmr: BMR, grid: Grid) -> np.ndarray:
    """Longitude-averaged BMR: a pair of opposite-polarity Gaussian rings in latitude."""
    lead, trail = bmr.polarity_centres()
    lat = grid.lat_deg
    cell = 2.0 * np.pi * R_SUN_CM**2 * grid.area
    out = np.zeros(grid.n_lat)
    for (lat_c, _), sign in ((lead, bmr.leading_sign), (trail, -bmr.leading_sign)):
        g = np.exp(-0.5 * ((lat - lat_c) / bmr.width_deg) ** 2)
        out += sign * _normalise(g, cell, bmr.flux_mx)
    return out


def bmr_field_2d(bmr: BMR, grid: Grid) -> np.ndarray:
    """2-D BMR with van Ballegooijen et al. (1998) polarity shape exp(-2(1 - cos b)/delta^2).

    With delta = sqrt(2) * width this is a Gaussian of width ``width_deg`` for small b.
    """
    lead, trail = bmr.polarity_centres()
    lat = grid.lat[:, None]
    lon = grid.lon[None, :]
    delta = np.sqrt(2.0) * np.deg2rad(bmr.width_deg)
    cell = (R_SUN_CM**2 * grid.dlon * grid.area)[:, None] * np.ones((1, grid.n_lon))
    out = np.zeros((grid.n_lat, grid.n_lon))
    for (lat_c, lon_c), sign in ((lead, bmr.leading_sign), (trail, -bmr.leading_sign)):
        lc, pc = np.deg2rad(lat_c), np.deg2rad(lon_c)
        cos_b = np.sin(lat) * np.sin(lc) + np.cos(lat) * np.cos(lc) * np.cos(lon - pc)
        g = np.exp(-2.0 * (1.0 - cos_b) / delta**2)
        out += sign * _normalise(g, cell, bmr.flux_mx)
    return out


# ---------------------------------------------------------------------------
# Synthetic solar cycle
# ---------------------------------------------------------------------------


def hathaway_profile(t_months, b: float = 56.0, c: float = 0.8, t0: float = -4.0):
    """Hathaway, Wilson & Reichmann (1994) cycle shape, unit amplitude.

    F(t) = x^3 / (exp(x^2) - c),  x = (t - t0) / b,  t in months after minimum.
    Defaults (b = 56, c = 0.8, t0 = -4) fit the average cycle (Hathaway 2015, Living Rev.).
    """
    x = (np.asarray(t_months, dtype=float) - t0) / b
    xp = np.clip(x, 0.0, None)
    return np.where(x > 0.0, xp**3 / (np.exp(xp**2) - c), 0.0)


@dataclass
class SyntheticCycle:
    """Random BMR emergences that follow the statistical laws of a solar cycle.

    Emergence times follow the Hathaway (1994) profile, the mean latitude drifts
    equatorward as lat_start * exp(-t / lat_efold) (spot-zone migration), tilts
    follow Joy's law with optional Gaussian scatter, and polarities follow Hale's law.
    """

    cycle: int = 25
    start_yr: float = 0.0
    length_yr: float = 11.0
    n_bmr: int = 2000
    flux_mx: float = 3e21  # median flux per polarity
    flux_log_sigma: float = 0.0  # lognormal scatter of flux (natural log)
    lat_start_deg: float = 28.0
    lat_efold_months: float = 90.0
    lat_scatter_deg: float = 5.0
    joy_amplitude_deg: float = JOY_AMPLITUDE_DEG
    tilt_scatter_deg: float = 0.0
    sep_deg: float = 6.0
    width_deg: float = 2.0
    hathaway_b: float = 56.0
    hathaway_c: float = 0.8
    hathaway_t0: float = -4.0

    def activity(self, t_yr) -> np.ndarray:
        """Relative emergence rate at ``t_yr`` years after the cycle start."""
        return hathaway_profile(
            12.0 * np.asarray(t_yr), self.hathaway_b, self.hathaway_c, self.hathaway_t0
        )

    def mean_latitude(self, t_yr) -> np.ndarray:
        return self.lat_start_deg * np.exp(-12.0 * np.asarray(t_yr) / self.lat_efold_months)

    def generate(self, seed=None, member: int = 0) -> list[BMR]:
        rng = np.random.default_rng(seed)
        n = self.n_bmr
        # Emergence times by inverse-CDF sampling of the activity profile.
        grid_t = np.linspace(0.0, self.length_yr, 4001)
        cdf = np.cumsum(self.activity(grid_t))
        cdf /= cdf[-1]
        t = np.interp(rng.random(n), cdf, grid_t)

        hemi = np.where(rng.random(n) < 0.5, 1.0, -1.0)
        abs_lat = self.mean_latitude(t) + self.lat_scatter_deg * rng.standard_normal(n)
        lat = hemi * np.clip(abs_lat, 1.0, 60.0)
        lon = 360.0 * rng.random(n)
        flux = self.flux_mx * np.exp(self.flux_log_sigma * rng.standard_normal(n))
        tilt = joy_tilt_deg(lat, self.joy_amplitude_deg)
        tilt = tilt + self.tilt_scatter_deg * rng.standard_normal(n)

        order = np.argsort(t)
        return [
            BMR(
                time_yr=float(self.start_yr + t[i]),
                lat_deg=float(lat[i]),
                lon_deg=float(lon[i]),
                flux_mx=float(flux[i]),
                tilt_deg=float(tilt[i]),
                leading_sign=hale_leading_sign(self.cycle, lat[i]),
                sep_deg=self.sep_deg,
                width_deg=self.width_deg,
                member=member,
            )
            for i in order
        ]


def generate_cycles(template: SyntheticCycle, n_cycles: int, seed=None, member: int = 0):
    """Concatenate ``n_cycles`` consecutive cycles (numbers increase by one each cycle)."""
    rng = np.random.default_rng(seed)
    events: list[BMR] = []
    for k in range(n_cycles):
        cyc = replace(
            template,
            cycle=template.cycle + k,
            start_yr=template.start_yr + k * template.length_yr,
        )
        events.extend(cyc.generate(seed=rng, member=member))
    return events


# ---------------------------------------------------------------------------
# Catalogue input
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ("time_yr", "lat_deg", "flux_mx", "tilt_deg")


def load_catalogue(path, sep_deg: float = 6.0, width_deg: float = 2.0) -> list[BMR]:
    """Load BMRs from a CSV file with a header row.

    Required columns: time_yr, lat_deg, flux_mx, tilt_deg (Joy's-law sense, see module doc).
    Optional columns: lon_deg, sep_deg, width_deg, and either leading_sign (+1/-1)
    or cycle (Hale's law then sets the leading sign).
    """
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    data = np.atleast_1d(data)
    cols = data.dtype.names or ()
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"catalogue {path} is missing columns: {missing}")
    if "leading_sign" not in cols and "cycle" not in cols:
        raise ValueError("catalogue needs a 'leading_sign' or a 'cycle' column")

    events = []
    for row in data:
        if "leading_sign" in cols:
            sign = int(np.sign(row["leading_sign"]))
        else:
            sign = hale_leading_sign(int(row["cycle"]), row["lat_deg"])
        events.append(
            BMR(
                time_yr=float(row["time_yr"]),
                lat_deg=float(row["lat_deg"]),
                flux_mx=float(row["flux_mx"]),
                tilt_deg=float(row["tilt_deg"]),
                leading_sign=sign,
                lon_deg=float(row["lon_deg"]) if "lon_deg" in cols else 0.0,
                sep_deg=float(row["sep_deg"]) if "sep_deg" in cols else sep_deg,
                width_deg=float(row["width_deg"]) if "width_deg" in cols else width_deg,
            )
        )
    return sorted(events, key=lambda e: e.time_yr)
