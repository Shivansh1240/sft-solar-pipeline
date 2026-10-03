"""Named verification checks with explicit pass criteria.

Each check runs the solver on a problem with a known answer and returns a
:class:`CheckResult`. The same checks back ``sft verify``, the pytest suite and
the published benchmark report, so the three cannot disagree.

``quick=True`` uses coarser grids so the full set runs in well under a minute.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from . import analytic as an
from .constants import YEAR_S
from .diagnostics import axial_dipole
from .flows import differential_rotation
from .grid import Grid
from .model import SFTModel, TransportParams
from .sources import BMR, SyntheticCycle, generate_cycles, hale_leading_sign, joy_tilt_deg


@dataclass
class CheckResult:
    name: str
    description: str
    metric: str
    value: float
    criterion: str
    passed: bool
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["value"] = float(self.value)
        d["passed"] = bool(self.passed)
        return d


def fitted_order(h, err) -> float:
    """Least-squares slope of log(err) against log(h)."""
    return float(np.polyfit(np.log(h), np.log(err), 1)[0])


def _bmr_initial(grid: Grid) -> np.ndarray:
    b = BMR(0.0, 15.0, 1e22, 10.0, 1, sep_deg=8.0, width_deg=4.0)
    return b.field(grid) + 0.5 * np.sin(grid.lat)


# ---------------------------------------------------------------------------
# Conservation and exactness
# ---------------------------------------------------------------------------


def check_flux_conservation_1d(quick: bool = False) -> CheckResult:
    g = Grid(90)
    rng = np.random.default_rng(0)
    B0 = rng.standard_normal(g.n_lat)
    drifts = {}
    for scheme in ("ssp2", "ars2", "euler"):
        r = SFTModel(g, TransportParams(), dt_days=2.0, scheme=scheme).run(
            B0, 2.0 if quick else 10.0, save_every_days=30
        )
        f = r.signed_flux()
        drifts[scheme] = float(np.max(np.abs(f - f[0])) / np.max(np.abs(r.unsigned_flux())))
    worst = max(drifts.values())
    return CheckResult(
        "flux_conservation_1d",
        "Net flux of a random field under advection + diffusion (no decay), all schemes",
        "max |dPhi| / unsigned flux",
        worst,
        "< 1e-12",
        worst < 1e-12,
        {"by_scheme": drifts},
    )


def check_flux_conservation_2d(quick: bool = False) -> CheckResult:
    g = Grid(45, 90) if quick else Grid(90, 180)
    B0 = np.random.default_rng(1).standard_normal((g.n_lat, g.n_lon))
    r = SFTModel(g, TransportParams()).run(B0, 1.0 if quick else 2.0, save_every_days=30)
    f = r.signed_flux()
    drift = float(np.max(np.abs(f - f[0])) / np.max(np.abs(r.unsigned_flux())))
    return CheckResult(
        "flux_conservation_2d",
        "Net flux of a random 2-D field with differential rotation",
        "max |dPhi| / unsigned flux",
        drift,
        "< 1e-12",
        drift < 1e-12,
    )


def check_dipole_normalisation(quick: bool = False) -> CheckResult:
    g = Grid(180)
    err = abs(float(axial_dipole(np.sin(g.lat), g)) - 1.0)
    return CheckResult(
        "dipole_normalisation",
        "Axial dipole of B = sin(lat) on a 1 degree grid",
        "|D - 1|",
        err,
        "< 1e-4",
        err < 1e-4,
    )


def check_decay(quick: bool = False) -> CheckResult:
    g = Grid(45)
    p = TransportParams(u0_ms=0.0, tau_yr=5.0)
    r = SFTModel(g, p, dt_days=1.0).run(np.ones(g.n_lat), 5.0, save_every_days=1e9)
    err = float(np.max(np.abs(r.B[-1] - np.exp(-1.0))) / np.exp(-1.0))
    return CheckResult(
        "exponential_decay",
        "Uniform field with tau = 5 yr after 5 yr (dt = 1 day) vs exp(-1)",
        "relative error",
        err,
        "< 1e-6",
        err < 1e-6,
    )


def check_rotation_exact(quick: bool = False) -> CheckResult:
    g = Grid(45, 90)
    p = TransportParams(eta_km2s=0.0, u0_ms=0.0)
    B0 = np.exp(-(((g.lon[None, :] - np.pi) / 0.3) ** 2)) * np.cos(g.lat)[:, None] ** 2
    r = SFTModel(g, p, dt_days=1.0).run(B0, 0.5, save_every_days=1e9)
    omega = differential_rotation(g.lat)
    m = np.arange(g.n_modes)
    phase = np.exp(-1j * m[None, :] * omega[:, None] * r.t_yr[-1] * YEAR_S)
    exact = np.fft.irfft(np.fft.rfft(B0, axis=1) * phase, n=g.n_lon, axis=1)
    err = float(np.max(np.abs(r.B[-1] - exact)))
    return CheckResult(
        "differential_rotation_exact",
        "Pure differential rotation for 0.5 yr vs the exact shifted field",
        "max abs error [G]",
        err,
        "< 1e-12",
        err < 1e-12,
    )


def check_axisymmetric_consistency(quick: bool = False) -> CheckResult:
    g2, g1 = Grid(45, 90), Grid(45)
    b = BMR(0.0, 20.0, 5e21, 15.0, 1, lon_deg=100.0, sep_deg=8.0, width_deg=4.0)
    B2 = b.field(g2)
    p = TransportParams(tau_yr=5.0)
    r2 = SFTModel(g2, p, limiter="none").run(B2, 1.0, save_every_days=1e9)
    r1 = SFTModel(g1, p, limiter="none").run(B2.mean(axis=1), 1.0, save_every_days=1e9)
    err = float(np.max(np.abs(r2.lon_avg()[-1] - r1.B[-1])) / np.max(np.abs(r1.B[-1])))
    return CheckResult(
        "axisymmetric_consistency",
        "Longitude average of a 2-D run vs the 1-D run (linear scheme, rotation on)",
        "max relative difference",
        err,
        "< 1e-12",
        err < 1e-12,
    )


def check_positivity(quick: bool = False) -> CheckResult:
    g = Grid(90)
    p = TransportParams(eta_km2s=0.0, u0_ms=20.0)
    B0 = np.where(np.abs(g.lat_deg - 10.0) < 8.0, 1.0, 0.0)
    probe = SFTModel(g, p, dt_days=1.0)
    m = SFTModel(g, p, dt_days=0.5 / probe.cfl)  # CFL = 0.5
    r = m.run(B0, 2.0, save_every_days=5)
    worst = float(r.B.min())
    return CheckResult(
        "positivity",
        "Pure advection of a non-negative top-hat (SSP2 + van Leer, CFL 0.5)",
        "min B / max B0",
        worst,
        ">= -1e-12",
        worst >= -1e-12,
    )


def check_hale_joy_sign(quick: bool = False) -> CheckResult:
    g = Grid(90)
    out = {}
    ok = True
    for cycle in (23, 24):
        for lat in (20.0, -20.0):
            b = BMR(0.0, lat, 1e22, joy_tilt_deg(lat), hale_leading_sign(cycle, lat))
            d = float(axial_dipole(b.field(g), g))
            out[f"cycle{cycle}_lat{lat:+.0f}"] = d
            ok &= (d < 0) if cycle % 2 else (d > 0)
    return CheckResult(
        "hale_joy_dipole_sign",
        "A Joy-tilted, Hale-ordered BMR pushes the dipole negative in odd cycles, "
        "positive in even cycles, in both hemispheres",
        "dipole of single BMRs [G]",
        min(abs(v) for v in out.values()),
        "sign pattern (-,-,+,+)",
        bool(ok),
        out,
    )


def check_cycle_reversal(quick: bool = False) -> CheckResult:
    g = Grid(90)
    template = SyntheticCycle(cycle=23, n_bmr=800 if quick else 2000)
    events = generate_cycles(template, 2, seed=7)
    p = TransportParams(tau_yr=5.0)
    r = SFTModel(g, p).run(np.zeros(g.n_lat), 22.0, events=events, save_every_days=27)
    d11 = float(r.dipole()[np.argmin(np.abs(r.t_yr - 11.0))])
    d22 = float(r.dipole()[-1])
    return CheckResult(
        "polar_reversal",
        "Two synthetic cycles (23, 24) from zero field, tau = 5 yr: dipole sign alternates",
        "dipole at end of cycle 23, 24 [G]",
        d11,
        "D(11 yr) < 0 < D(22 yr)",
        d11 < 0 < d22,
        {"D_11yr": d11, "D_22yr": d22},
    )


# ---------------------------------------------------------------------------
# Convergence orders
# ---------------------------------------------------------------------------


def time_convergence(scheme: str, quick: bool = False) -> dict:
    g = Grid(90)
    p = TransportParams(eta_km2s=250.0, u0_ms=15.0, tau_yr=5.0)
    B0 = _bmr_initial(g)
    n_list = (32, 64, 128) if quick else (32, 64, 128, 256)
    ref = SFTModel(g, p, dt_days=365.25 / 8192, scheme=scheme).run(B0, 1.0, save_every_days=1e9)
    dts, errs = [], []
    for n in n_list:
        r = SFTModel(g, p, dt_days=365.25 / n, scheme=scheme).run(B0, 1.0, save_every_days=1e9)
        dts.append(365.25 / n)
        errs.append(float(np.max(np.abs(r.B[-1] - ref.B[-1]))))
    return {"dt_days": dts, "error": errs, "order": fitted_order(dts, errs)}


def _time_order_check(scheme: str, expected: int, quick: bool) -> CheckResult:
    data = time_convergence(scheme, quick)
    lo, hi = expected - 0.1, expected + 0.15
    return CheckResult(
        f"time_order_{scheme}",
        f"Temporal convergence of {scheme} (advection + diffusion + decay, 1 yr, vs dt -> 0)",
        "fitted order",
        data["order"],
        f"in [{lo:.2f}, {hi:.2f}]",
        lo <= data["order"] <= hi,
        data,
    )


def space_convergence_diffusion(quick: bool = False) -> dict:
    ns = (45, 90, 180) if quick else (45, 90, 180, 360)
    l, eta, T = 3, 500.0, 1.0
    errs = []
    for n in ns:
        g = Grid(n)
        r = SFTModel(g, TransportParams(u0_ms=0.0), dt_days=0.25).run(
            an.legendre_mode(g, l), T, save_every_days=1e9
        )
        exact = an.legendre_mode(g, l) * np.exp(-an.diffusion_rate(l, eta) * T * YEAR_S)
        errs.append(float(np.max(np.abs(r.B[-1] - exact))))
    return {"n_lat": list(ns), "error": errs, "order": -fitted_order(ns, errs)}


def space_convergence_mms(quick: bool = False, limiter: str = "vanleer") -> dict:
    ns = (45, 90, 180) if quick else (45, 90, 180, 360)
    mms = an.ManufacturedSolution(u0_ms=15.0, eta_km2s=500.0, tau_yr=5.0)
    p = TransportParams(eta_km2s=500.0, u0_ms=15.0, tau_yr=5.0)
    errs = []
    for n in ns:
        g = Grid(n)
        r = SFTModel(g, p, dt_days=365.25 / 2048, limiter=limiter).run(
            mms.exact(g, 0.0), 1.0, source=mms.source(g), save_every_days=1e9
        )
        e = r.B[-1] - mms.exact(g, 1.0)
        errs.append(float(np.sqrt(0.5 * np.sum(e**2 * g.area))))
    return {"n_lat": list(ns), "error": errs, "order": -fitted_order(ns, errs)}


def space_convergence_steady(quick: bool = False) -> dict:
    ns = (45, 90, 180)
    errs = []
    for n in ns:
        g = Grid(n)
        p = TransportParams(eta_km2s=500.0, u0_ms=5.0)
        r = SFTModel(g, p, dt_days=2.0).run(np.ones(n), 40.0, save_every_days=1e9)
        exact = an.steady_state_monopole(g, 5.0, 500.0, "sin2lat")
        errs.append(float(np.max(np.abs(r.B[-1] - exact)) / exact.max()))
    return {"n_lat": list(ns), "error": errs, "order": -fitted_order(ns, errs)}


def space_convergence_harmonic(quick: bool = False) -> dict:
    ns = (45, 90) if quick else (45, 90, 180)
    l, m, T = 4, 3, 0.5
    errs = []
    for n in ns:
        g = Grid(n, 2 * n)
        p = TransportParams(u0_ms=0.0, rotation=None)
        r = SFTModel(g, p, dt_days=0.25).run(an.harmonic_mode(g, l, m), T, save_every_days=1e9)
        exact = an.harmonic_mode(g, l, m) * np.exp(-an.diffusion_rate(l, 500.0) * T * YEAR_S)
        errs.append(float(np.max(np.abs(r.B[-1] - exact)) / np.max(np.abs(exact))))
    return {"n_lat": list(ns), "error": errs, "order": -fitted_order(ns, errs)}


def _space_order_check(name, description, fn, quick) -> CheckResult:
    data = fn(quick)
    return CheckResult(
        name, description, "fitted order", data["order"], ">= 1.9", data["order"] >= 1.9, data
    )


CHECKS: dict[str, Callable[[bool], CheckResult]] = {
    "flux_conservation_1d": check_flux_conservation_1d,
    "flux_conservation_2d": check_flux_conservation_2d,
    "dipole_normalisation": check_dipole_normalisation,
    "exponential_decay": check_decay,
    "differential_rotation_exact": check_rotation_exact,
    "axisymmetric_consistency": check_axisymmetric_consistency,
    "positivity": check_positivity,
    "hale_joy_dipole_sign": check_hale_joy_sign,
    "polar_reversal": check_cycle_reversal,
    "time_order_ssp2": lambda q: _time_order_check("ssp2", 2, q),
    "time_order_ars2": lambda q: _time_order_check("ars2", 2, q),
    "time_order_euler": lambda q: _time_order_check("euler", 1, q),
    "space_order_diffusion": lambda q: _space_order_check(
        "space_order_diffusion",
        "Diffusion of the Legendre mode P_3 vs exact decay (L-inf)",
        space_convergence_diffusion,
        q,
    ),
    "space_order_mms": lambda q: _space_order_check(
        "space_order_mms",
        "Manufactured solution, full operator with van Leer limiter (L2)",
        space_convergence_mms,
        q,
    ),
    "space_order_steady_state": lambda q: _space_order_check(
        "space_order_steady_state",
        "Advection-diffusion steady state for net flux, sin(2 lat) flow (L-inf)",
        space_convergence_steady,
        q,
    ),
    "space_order_2d_harmonic": lambda q: _space_order_check(
        "space_order_2d_harmonic",
        "2-D diffusion of the harmonic P_4^3(sin lat) cos(3 lon) (L-inf)",
        space_convergence_harmonic,
        q,
    ),
}


def run_all(quick: bool = False, names=None, verbose: bool = True) -> list[CheckResult]:
    results = []
    for name in names or CHECKS:
        res = CHECKS[name](quick)
        results.append(res)
        if verbose:
            flag = "PASS" if res.passed else "FAIL"
            print(f"[{flag}] {res.name:30s} {res.metric} = {res.value:.3g}  ({res.criterion})")
    return results


__all__ = ["CHECKS", "CheckResult", "fitted_order", "run_all"]
