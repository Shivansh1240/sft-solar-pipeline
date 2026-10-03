"""Finite-volume latitude(-longitude) grid on the sphere.

Latitude cells are uniform in latitude and the outermost cell faces sit exactly
on the poles. Because cos(latitude) vanishes there, the transport flux through
the poles is identically zero: no ghost cells or pole boundary conditions are
needed, and the discrete signed flux is conserved to round-off.
"""

from __future__ import annotations

import numpy as np


class Grid:
    """Cell-centred grid with ``n_lat`` latitude cells and ``n_lon`` longitude cells.

    ``n_lon == 1`` means an axisymmetric (longitude-averaged) problem.

    Attributes (all angles in radians)
    ----------------------------------
    lat_faces : (n_lat + 1,) face latitudes, from -pi/2 to +pi/2
    lat       : (n_lat,) cell-centre latitudes
    dlat      : latitude spacing
    cos_faces : cos(lat_faces), exactly 0 at the poles
    area      : (n_lat,) integral of cos(lat) over each cell; sums to 2
    dipole_w  : (n_lat,) integral of sin(lat) cos(lat) over each cell
    lon       : (n_lon,) cell-centre longitudes, j * 2 pi / n_lon
    dlon      : longitude spacing
    """

    def __init__(self, n_lat: int = 180, n_lon: int = 1):
        if n_lat < 4:
            raise ValueError("n_lat must be >= 4")
        if n_lon < 1:
            raise ValueError("n_lon must be >= 1")
        if n_lon > 1 and n_lon % 2:
            raise ValueError("n_lon must be even for the longitude FFT")
        self.n_lat = int(n_lat)
        self.n_lon = int(n_lon)

        self.dlat = np.pi / n_lat
        self.lat_faces = -0.5 * np.pi + self.dlat * np.arange(n_lat + 1)
        self.lat_faces[-1] = 0.5 * np.pi
        self.lat = 0.5 * (self.lat_faces[:-1] + self.lat_faces[1:])

        self.cos_faces = np.cos(self.lat_faces)
        self.cos_faces[0] = self.cos_faces[-1] = 0.0
        self.cos_lat = np.cos(self.lat)
        self.sin_lat = np.sin(self.lat)

        sin_f = np.sin(self.lat_faces)
        self.area = np.diff(sin_f)
        self.dipole_w = 0.5 * np.diff(sin_f**2)

        self.dlon = 2.0 * np.pi / n_lon
        self.lon = self.dlon * np.arange(n_lon)

    @property
    def is_2d(self) -> bool:
        return self.n_lon > 1

    @property
    def lat_deg(self) -> np.ndarray:
        return np.rad2deg(self.lat)

    @property
    def lon_deg(self) -> np.ndarray:
        return np.rad2deg(self.lon)

    @property
    def n_modes(self) -> int:
        """Number of longitudinal Fourier modes kept by ``numpy.fft.rfft``."""
        return self.n_lon // 2 + 1 if self.is_2d else 1

    def mesh_deg(self):
        """Return (LAT, LON) 2-D arrays in degrees, shape (n_lat, n_lon)."""
        lon, lat = np.meshgrid(self.lon_deg, self.lat_deg)
        return lat, lon

    def __repr__(self) -> str:
        return f"Grid(n_lat={self.n_lat}, n_lon={self.n_lon})"
