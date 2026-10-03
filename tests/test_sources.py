import numpy as np
import pytest

from sft import (
    BMR,
    Grid,
    SyntheticCycle,
    axial_dipole,
    generate_cycles,
    hale_leading_sign,
    joy_tilt_deg,
    load_catalogue,
    signed_flux,
    unsigned_flux,
)
from sft.sources import hathaway_profile


def test_hale_law_table():
    assert hale_leading_sign(23, 10) == 1 and hale_leading_sign(23, -10) == -1
    assert hale_leading_sign(24, 10) == -1 and hale_leading_sign(24, -10) == 1


def test_joy_tilt():
    assert joy_tilt_deg(30.0) == pytest.approx(32.1 * 0.5)
    assert joy_tilt_deg(-30.0) == pytest.approx(joy_tilt_deg(30.0))


@pytest.mark.parametrize("lat", [20.0, -20.0])
def test_polarity_geometry_follows_joy(lat):
    lead, trail = BMR(0, lat, 1e22, 10.0, 1, lon_deg=100.0).polarity_centres()
    assert abs(lead[0]) < abs(trail[0])  # leading polarity nearer the equator
    assert lead[1] > trail[1]  # and westward (larger longitude)


@pytest.mark.parametrize("grid", [Grid(180), Grid(90, 180)])
def test_bmr_is_flux_balanced(grid):
    b = BMR(0, 25.0, 3e21, 12.0, -1, lon_deg=200.0, sep_deg=14.0)  # polarities 7 sigma apart
    B = b.field(grid)
    assert abs(signed_flux(B, grid)) < 1e-9 * b.flux_mx
    if grid.is_2d:
        assert unsigned_flux(B, grid) == pytest.approx(2 * b.flux_mx, rel=0.02)


def test_bmr_1d_is_longitude_average_of_2d_dipole_contribution():
    b = BMR(0, 20.0, 1e22, 15.0, 1, lon_deg=50.0, width_deg=3.0)
    d1 = axial_dipole(b.field(Grid(180)), Grid(180))
    d2 = axial_dipole(b.field(Grid(180, 360)), Grid(180, 360))
    assert d2 == pytest.approx(d1, rel=0.02)


def test_hathaway_profile_shape():
    t = np.linspace(-10, 200, 2101)
    f = hathaway_profile(t)
    assert np.all(f[t <= -4] == 0)
    peak = t[np.argmax(f)]
    assert 40 < peak < 60  # months after minimum, for b = 56
    assert f[-1] < 0.05 * f.max()


def test_synthetic_cycle_statistics_and_reproducibility():
    cyc = SyntheticCycle(cycle=24, n_bmr=3000)
    ev = cyc.generate(seed=5)
    assert len(ev) == 3000
    assert ev == cyc.generate(seed=5)
    t = np.array([e.time_yr for e in ev])
    lat = np.array([e.lat_deg for e in ev])
    assert np.all(np.diff(t) >= 0)
    assert 0.4 < np.mean(lat > 0) < 0.6
    early, late = np.abs(lat[t < 3]).mean(), np.abs(lat[t > 8]).mean()
    assert early > late + 5  # Sporer's law: emergence drifts equatorward
    assert all(e.leading_sign == hale_leading_sign(24, e.lat_deg) for e in ev)


def test_generate_cycles_alternates_parity():
    ev = generate_cycles(SyntheticCycle(cycle=23, n_bmr=50), 3, seed=0)
    for k, cycle in enumerate((23, 24, 25)):
        chunk = [e for e in ev if 11 * k <= e.time_yr < 11 * (k + 1)]
        assert chunk and all(e.leading_sign == hale_leading_sign(cycle, e.lat_deg) for e in chunk)


def test_load_catalogue(tmp_path):
    csv = tmp_path / "bmr.csv"
    csv.write_text(
        "time_yr,lat_deg,lon_deg,flux_mx,tilt_deg,cycle\n"
        "2001.5,-12.0,40.0,2e21,5.0,23\n"
        "2000.1,15.0,300.0,3e21,7.0,23\n"
    )
    ev = load_catalogue(csv)
    assert [e.time_yr for e in ev] == [2000.1, 2001.5]
    assert ev[0].leading_sign == 1 and ev[1].leading_sign == -1
    assert ev[0].lon_deg == 300.0


def test_load_catalogue_explicit_sign_and_errors(tmp_path):
    ok = tmp_path / "ok.csv"
    ok.write_text("time_yr,lat_deg,flux_mx,tilt_deg,leading_sign\n1.0,10,1e21,3,-1\n")
    assert load_catalogue(ok)[0].leading_sign == -1

    missing = tmp_path / "missing.csv"
    missing.write_text("time_yr,lat_deg,flux_mx\n1.0,10,1e21\n")
    with pytest.raises(ValueError, match="missing"):
        load_catalogue(missing)

    nosign = tmp_path / "nosign.csv"
    nosign.write_text("time_yr,lat_deg,flux_mx,tilt_deg\n1.0,10,1e21,3\n")
    with pytest.raises(ValueError, match="leading_sign"):
        load_catalogue(nosign)
