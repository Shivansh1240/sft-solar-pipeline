"""sft: a finite-volume Surface Flux Transport solver for the solar surface field."""

from .diagnostics import (
    axial_dipole,
    legendre_coefficients,
    lon_average,
    polar_field,
    signed_flux,
    unsigned_flux,
)
from .flows import differential_rotation, meridional_flow
from .grid import Grid
from .imex import SCHEMES
from .model import RunResult, SFTModel, TransportParams
from .sources import (
    BMR,
    SyntheticCycle,
    generate_cycles,
    hale_leading_sign,
    joy_tilt_deg,
    load_catalogue,
)

__version__ = "1.0.0"

__all__ = [
    "BMR",
    "Grid",
    "RunResult",
    "SCHEMES",
    "SFTModel",
    "SyntheticCycle",
    "TransportParams",
    "axial_dipole",
    "differential_rotation",
    "generate_cycles",
    "hale_leading_sign",
    "joy_tilt_deg",
    "legendre_coefficients",
    "load_catalogue",
    "lon_average",
    "meridional_flow",
    "polar_field",
    "signed_flux",
    "unsigned_flux",
]
