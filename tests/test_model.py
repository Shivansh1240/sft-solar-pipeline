import numpy as np
import pytest

from sft import BMR, Grid, SFTModel, TransportParams
from sft.constants import YEAR_S


def test_run_ends_exactly_at_requested_time():
    g = Grid(45)
    # 0.5 yr / 0.7 d is not a whole number of steps: the last step must be shortened
    r = SFTModel(g, dt_days=0.7).run(np.sin(g.lat), 0.5, save_every_days=30)
    assert r.t_yr[-1] == pytest.approx(0.5, abs=1e-12)
    assert r.t_yr[0] == 0.0
    assert np.all(np.diff(r.t_yr) > 0)


def test_partial_step_matches_decay_exactly():
    g = Grid(20)
    p = TransportParams(u0_ms=0.0, tau_yr=1.0)
    r = SFTModel(g, p, dt_days=0.7).run(np.ones(g.n_lat), 1.0, save_every_days=1e9)
    assert r.B[-1, 0] == pytest.approx(np.exp(-1.0), rel=1e-6)


def test_cfl_guard():
    with pytest.raises(ValueError, match="CFL"):
        SFTModel(Grid(180), TransportParams(u0_ms=20.0), dt_days=60.0)


@pytest.mark.parametrize("kw", [{"scheme": "rk4"}, {"limiter": "superbee2"}])
def test_rejects_unknown_options(kw):
    with pytest.raises(ValueError):
        SFTModel(Grid(20), **kw)


def test_rejects_wrong_field_shape():
    m = SFTModel(Grid(20))
    with pytest.raises(ValueError):
        m.run(np.zeros(21), 0.1)
    with pytest.raises(ValueError):
        SFTModel(Grid(20, 40)).run(np.zeros(20), 0.1)


def test_events_inserted_once_and_counted():
    g = Grid(90)
    events = [
        BMR(0.25, 15.0, 1e22, 8.0, 1),
        BMR(0.5, -15.0, 1e22, 8.0, -1),
        BMR(5.0, 10, 1e22, 5, 1),
    ]
    r = SFTModel(g, TransportParams(u0_ms=0.0, eta_km2s=0.0)).run(
        np.zeros(g.n_lat), 1.0, events=events, save_every_days=1
    )
    assert r.n_events == 2  # the event at 5 yr is outside the run
    i = np.searchsorted(r.t_yr, 0.3)
    expected = events[0].field(g)
    assert np.allclose(r.B[i], expected)  # no transport: the field is just the BMR
    assert np.allclose(r.B[-1], events[0].field(g) + events[1].field(g))


def test_ensemble_columns_are_independent():
    g = Grid(60)
    m = SFTModel(g, TransportParams(tau_yr=5.0))
    ev = [BMR(0.1, 20.0, 1e22, 10.0, 1, member=1)]
    B0 = np.zeros((g.n_lat, 3))
    B0[:, 0] = np.sin(g.lat)
    r = m.run(B0, 1.0, events=ev, save_every_days=1e9)
    single = m.run(np.sin(g.lat), 1.0, save_every_days=1e9)
    assert np.allclose(r.B[-1][:, 0], single.B[-1])
    assert np.abs(r.B[-1][:, 1]).max() > 0
    assert np.all(r.B[-1][:, 2] == 0)
    assert r.dipole().shape == (2, 3)


def test_event_member_requires_ensemble_state():
    g = Grid(30)
    with pytest.raises(ValueError):
        SFTModel(g).run(np.zeros(g.n_lat), 0.5, events=[BMR(0.1, 10, 1e21, 5, 1, member=2)])


def test_linearity_with_linear_limiter():
    g = Grid(60)
    m = SFTModel(g, TransportParams(tau_yr=4.0), limiter="none")
    a = BMR(0, 15, 1e22, 10, 1).field(g)
    b = np.sin(g.lat) ** 3
    ra, rb = m.run(a, 2.0).B[-1], m.run(b, 2.0).B[-1]
    rab = m.run(2.0 * a - 3.0 * b, 2.0).B[-1]
    assert np.allclose(rab, 2.0 * ra - 3.0 * rb, atol=1e-12)


def test_continuous_source_adds_expected_flux():
    from sft import signed_flux

    g = Grid(45)
    rate = 1e-8  # G/s, uniform
    r = SFTModel(g, TransportParams(u0_ms=0.0)).run(
        np.zeros(g.n_lat), 1.0, source=lambda t: np.full(g.n_lat, rate), save_every_days=1e9
    )
    assert np.allclose(r.B[-1], rate * YEAR_S, rtol=1e-12)
    assert signed_flux(r.B[-1], g) > 0


def test_step_matches_run():
    g = Grid(40)
    m = SFTModel(g, TransportParams(tau_yr=3.0), dt_days=2.0)
    B0 = np.sin(g.lat) + 0.2
    assert np.allclose(m.step(B0), m.run(B0, 2.0 / 365.25).B[-1])


def test_2d_keep_2d_false_stores_longitude_average():
    g = Grid(30, 60)
    B0 = BMR(0, 20, 1e22, 10, 1, lon_deg=90, width_deg=4).field(g)
    full = SFTModel(g).run(B0, 0.2, save_every_days=30)
    avg = SFTModel(g).run(B0, 0.2, save_every_days=30, keep_2d=False)
    assert avg.B.shape == (len(avg.t_yr), 30)
    assert not avg.grid.is_2d
    assert np.allclose(avg.B, full.lon_avg())
    assert np.allclose(avg.dipole(), full.dipole())
