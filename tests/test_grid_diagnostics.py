import numpy as np
import pytest

from sft import Grid, axial_dipole, legendre_coefficients, polar_field, signed_flux, unsigned_flux
from sft.constants import R_SUN_CM


@pytest.mark.parametrize("n_lat", [4, 45, 180, 181])
def test_grid_geometry(n_lat):
    g = Grid(n_lat)
    assert g.lat_faces[0] == -np.pi / 2 and g.lat_faces[-1] == np.pi / 2
    assert g.cos_faces[0] == 0.0 and g.cos_faces[-1] == 0.0
    assert g.area.sum() == pytest.approx(2.0, abs=1e-14)
    assert g.dipole_w.sum() == pytest.approx(0.0, abs=1e-14)
    assert np.all(np.diff(g.lat) > 0)


def test_grid_rejects_bad_sizes():
    with pytest.raises(ValueError):
        Grid(3)
    with pytest.raises(ValueError):
        Grid(10, 7)  # odd longitude count


def test_uniform_field_flux_and_polar_field():
    g = Grid(90)
    B = np.full(g.n_lat, 2.0)
    assert signed_flux(B, g) == pytest.approx(2.0 * 4.0 * np.pi * R_SUN_CM**2, rel=1e-12)
    assert unsigned_flux(-B, g) == pytest.approx(signed_flux(B, g), rel=1e-12)
    north, south = polar_field(B, g)
    assert north == pytest.approx(2.0) and south == pytest.approx(2.0)
    assert axial_dipole(B, g) == pytest.approx(0.0, abs=1e-14)


def test_dipole_is_second_order_accurate():
    errs = [abs(axial_dipole(np.sin(Grid(n).lat), Grid(n)) - 1.0) for n in (45, 90, 180)]
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 1.9)


def test_legendre_coefficients_pick_out_single_mode():
    from scipy.special import eval_legendre

    g = Grid(360)
    c = legendre_coefficients(eval_legendre(3, g.sin_lat), g, l_max=6)
    expected = np.zeros(7)
    expected[3] = 1.0
    assert np.allclose(c, expected, atol=1e-4)
    # b_1 equals the axial dipole moment (both integrate exactly over cells)
    B = np.sin(g.lat) + 0.3 * eval_legendre(2, g.sin_lat)
    assert legendre_coefficients(B, g, 2)[1] == pytest.approx(axial_dipole(B, g), rel=1e-12)


def test_diagnostics_broadcast_over_trailing_axes():
    g = Grid(45)
    B = np.stack([np.sin(g.lat), 2 * np.sin(g.lat)], axis=1)
    d = axial_dipole(B, g)
    assert d.shape == (2,)
    assert d[1] == pytest.approx(2 * d[0])


def test_2d_diagnostics_match_longitude_average():
    g2, g1 = Grid(45, 90), Grid(45)
    rng = np.random.default_rng(3)
    B = rng.standard_normal((45, 90))
    assert axial_dipole(B, g2) == pytest.approx(axial_dipole(B.mean(axis=1), g1))
    assert signed_flux(B, g2) == pytest.approx(signed_flux(B.mean(axis=1), g1))
    # unsigned flux of the full map exceeds that of its longitude average
    assert unsigned_flux(B, g2) > unsigned_flux(B.mean(axis=1), g1)
