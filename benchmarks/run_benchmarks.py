"""Regenerate everything under results/: verification checks, convergence, runtime,
and reference physical runs, plus results/REPORT.md.

    python benchmarks/run_benchmarks.py            # full run, ~2-3 minutes
    python benchmarks/run_benchmarks.py --quick    # CI smoke version, coarser everything
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import scipy

from sft import BMR, Grid, SFTModel, SyntheticCycle, TransportParams, generate_cycles
from sft import verification as ver
from sft.sources import hale_leading_sign, joy_tilt_deg

ROOT = Path(__file__).resolve().parents[1]
LEGACY = json.loads((Path(__file__).parent / "legacy_baseline.json").read_text())

REFERENCE = TransportParams(eta_km2s=500.0, u0_ms=12.5, tau_yr=5.0)
CYCLE = SyntheticCycle(cycle=23, n_bmr=2000, flux_mx=3e21)
SEED = 2026


def zero_crossings(t, y):
    """Times where y changes sign (linear interpolation)."""
    idx = np.where(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]
    return [float(t[i] - y[i] * (t[i + 1] - t[i]) / (y[i + 1] - y[i])) for i in idx]


def best_wall(fn, repeat=3):
    best, out = np.inf, None
    for _ in range(repeat):
        out = fn()
        best = min(best, out.wall_time_s)
    return best, out


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_verification(quick):
    print("== verification checks")
    results = ver.run_all(quick=quick)
    return [r.to_dict() for r in results]


def section_convergence(quick):
    print("== convergence")
    time_data = {s: ver.time_convergence(s, quick) for s in ("euler", "ssp2", "ars2")}
    space_data = {
        "diffusion, P_3": ver.space_convergence_diffusion(quick),
        "full operator, MMS": ver.space_convergence_mms(quick),
        "advection steady state": ver.space_convergence_steady(quick),
        "2-D harmonic, Y_4^3": ver.space_convergence_harmonic(quick),
    }
    return {"time": time_data, "space": space_data}


def section_accuracy_cost(quick):
    """Error after 1 yr against wall time, per scheme (same problem as the time-order check)."""
    print("== accuracy vs cost")
    g = Grid(90)
    p = TransportParams(eta_km2s=250.0, u0_ms=15.0, tau_yr=5.0)
    B0 = ver._bmr_initial(g)
    ref = SFTModel(g, p, dt_days=365.25 / 8192).run(B0, 1.0, save_every_days=1e9).B[-1]
    n_list = (32, 128, 512) if quick else (32, 64, 128, 256, 512, 1024)
    out = {}
    for scheme in ("euler", "ssp2", "ars2"):
        rows = []
        for n in n_list:
            m = SFTModel(g, p, dt_days=365.25 / n, scheme=scheme)
            wall, r = best_wall(lambda m=m: m.run(B0, 1.0, save_every_days=1e9), repeat=5)
            rows.append(
                {"dt_days": 365.25 / n, "wall_s": wall, "error": float(np.abs(r.B[-1] - ref).max())}
            )
        out[scheme] = rows
    return out


def section_runtime(quick):
    print("== runtime")
    rows = []
    for n in (90, 180) if quick else (90, 180, 360):
        g = Grid(n)
        B0 = BMR(0, 12, 5e21, 10, 1).field(g)
        m = SFTModel(g, REFERENCE)
        wall, r = best_wall(lambda m=m, B0=B0: m.run(B0, 11.0, save_every_days=27))
        rows.append({"case": f"1-D, {n} cells, 11 yr", "n_steps": r.n_steps, "wall_s": wall})
    for n in (90,) if quick else (90, 180):
        g = Grid(n, 2 * n)
        B0 = BMR(0, 12, 5e21, 10, 1, lon_deg=180).field(g)
        m = SFTModel(g, REFERENCE)
        wall, r = best_wall(
            lambda m=m, B0=B0: m.run(B0, 1.0, save_every_days=1e9), repeat=1 if quick else 2
        )
        rows.append(
            {"case": f"2-D, {n} x {2 * n} cells, 1 yr", "n_steps": r.n_steps, "wall_s": wall}
        )
    return {
        "dt_days": 1.0,
        "scheme": "ssp2",
        "rows": rows,
        "legacy": LEGACY["runtime"],
    }


def reference_cycles(n_cycles, quick, params=REFERENCE, n_lat=180):
    tmpl = replace(CYCLE, n_bmr=600 if quick else CYCLE.n_bmr)
    events = generate_cycles(tmpl, n_cycles, seed=SEED)
    g = Grid(90 if quick else n_lat)
    return SFTModel(g, params).run(
        np.zeros(g.n_lat), n_cycles * tmpl.length_yr, events=events, save_every_days=27
    )


def section_cycle_run(quick):
    print("== reference cycle run")
    r = reference_cycles(3, quick)
    d = r.dipole()
    north, south = r.polar_fields()
    ends = {}
    for k, cyc in enumerate((23, 24, 25)):
        i = int(np.argmin(np.abs(r.t_yr - 11.0 * (k + 1))))
        ends[str(cyc)] = {
            "t_yr": float(r.t_yr[i]),
            "dipole_G": float(d[i]),
            "polar_north_G": float(north[i]),
            "polar_south_G": float(south[i]),
        }
    return r, {
        "params": {"eta_km2s": 500.0, "u0_ms": 12.5, "tau_yr": 5.0, "flow": "sin2lat"},
        "n_lat": r.grid.n_lat,
        "n_bmr_per_cycle": r.n_events // 3,
        "wall_s": r.wall_time_s,
        "end_of_cycle": ends,
        "dipole_reversals_yr": zero_crossings(r.t_yr, d),
        "north_polar_reversals_yr": zero_crossings(r.t_yr, north),
    }


def section_sensitivity(quick):
    print("== parameter sensitivity")
    variants = [("reference", {})]
    for eta in (250.0, 750.0):
        variants.append((f"eta = {eta:.0f} km^2/s", {"eta_km2s": eta}))
    for u0 in (8.0, 17.0):
        variants.append((f"u0 = {u0:g} m/s", {"u0_ms": u0}))
    for tau in (None, 10.0):
        variants.append((f"tau = {tau} yr" if tau else "no decay", {"tau_yr": tau}))
    rows = []
    for label, change in variants:
        r = reference_cycles(2, quick, params=replace(REFERENCE, **change), n_lat=90)
        d = r.dipole()
        i11 = int(np.argmin(np.abs(r.t_yr - 11.0)))
        rev = [t for t in zero_crossings(r.t_yr, d) if t > 11.0]
        rows.append(
            {
                "variant": label,
                "D_end_cycle23_G": float(d[i11]),
                "D_end_cycle24_G": float(d[-1]),
                "reversal_in_cycle24_yr": rev[0] if rev else None,
            }
        )
    return rows


def section_ensemble(quick):
    """Spread of the cycle-24 dipole from random emergence and tilt scatter."""
    print("== ensemble")
    n_members = 20 if quick else 200
    g = Grid(90 if quick else 180)
    tmpl = replace(CYCLE, n_bmr=600 if quick else CYCLE.n_bmr)
    model = SFTModel(g, REFERENCE)
    base = model.run(
        np.zeros(g.n_lat), 11.0, events=tmpl.generate(seed=SEED), save_every_days=1e9
    ).B[-1]
    cyc24 = replace(tmpl, cycle=24, start_yr=11.0)
    out, series = {}, {}
    # Lognormal flux scatter keeps the mean flux per BMR fixed: median = mean * exp(-s^2 / 2).
    variants = (
        ("no_tilt_scatter", 0.0, 0.0),
        ("tilt_scatter_15deg", 15.0, 0.0),
        ("tilt_15deg_and_lognormal_flux", 15.0, 1.0),
    )
    for label, scatter, log_sigma in variants:
        rng = np.random.default_rng(SEED + 1)
        member_cycle = replace(
            cyc24,
            tilt_scatter_deg=scatter,
            flux_log_sigma=log_sigma,
            flux_mx=cyc24.flux_mx * np.exp(-0.5 * log_sigma**2),
        )
        events = []
        for k in range(n_members):
            events.extend(member_cycle.generate(seed=rng, member=k))
        B0 = np.repeat(base[:, None], n_members, axis=1)
        r = model.run(B0, 22.0, t_start_yr=11.0, events=events, save_every_days=27)
        D = r.dipole()  # (n_time, members)
        final = D[-1]
        out[label] = {
            "tilt_scatter_deg": scatter,
            "flux_log_sigma": log_sigma,
            "members": n_members,
            "D_start_G": float(D[0, 0]),
            "D_end_mean_G": float(final.mean()),
            "D_end_std_G": float(final.std(ddof=1)),
            "D_end_p05_G": float(np.percentile(final, 5)),
            "D_end_p95_G": float(np.percentile(final, 95)),
            "fraction_reversed": float(np.mean(np.sign(final) != np.sign(D[0, 0]))),
            "wall_s": r.wall_time_s,
        }
        series[label] = (r.t_yr, D)
    return out, series


def section_2d(quick):
    print("== 2-D BMR demo")
    n = 90 if quick else 180
    g = Grid(n, 2 * n)
    lat = 20.0
    bmr = BMR(0.0, lat, 1e22, joy_tilt_deg(lat), hale_leading_sign(24, lat), lon_deg=180.0)
    r = SFTModel(g, TransportParams(eta_km2s=500.0, u0_ms=12.5)).run(
        bmr.field(g), 1.0, save_every_days=9
    )
    r1 = SFTModel(Grid(n), TransportParams(eta_km2s=500.0, u0_ms=12.5)).run(
        bmr.field(g).mean(axis=1), 1.0, save_every_days=9
    )
    rel = float(np.abs(r.dipole()[-1] - r1.dipole()[-1]) / abs(r1.dipole()[-1]))
    return r, {
        "grid": f"{n} x {2 * n}",
        "wall_s": r.wall_time_s,
        "dipole_after_1yr_2d_G": float(r.dipole()[-1]),
        "dipole_after_1yr_1d_G": float(r1.dipole()[-1]),
        "dipole_rel_diff_2d_vs_1d": rel,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def fmt(x, digits=3):
    if x is None:
        return "none"
    if isinstance(x, str):
        return x
    return f"{x:.{digits}g}"


def write_report(path, env, checks, conv, acc, runtime, cycle, sens, ens, demo2d):
    n_pass = sum(c["passed"] for c in checks)
    L = []
    w = L.append
    w("# SFT solver: verification and performance report\n")
    w(
        f"Generated by `benchmarks/run_benchmarks.py` on {env['date']} "
        f"({env['cpu']}, {env['cores']} cores, Python {env['python']}, NumPy {env['numpy']}, "
        f"SciPy {env['scipy']}). Mode: **{env['mode']}**.\n"
    )
    w("## 1. Verification checks\n")
    w(f"**{n_pass} / {len(checks)} passed.** Each check has a known exact answer.\n")
    w("| Check | What is tested | Metric | Value | Pass criterion | Result |")
    w("|---|---|---|---|---|---|")
    for c in checks:
        w(
            f"| `{c['name']}` | {c['description']} | {c['metric']} | {fmt(c['value'])} "
            f"| {c['criterion']} | {'PASS' if c['passed'] else '**FAIL**'} |"
        )
    w("")

    w("## 2. Convergence\n")
    w("![temporal convergence](figures/time_convergence.png)")
    w("![spatial convergence](figures/space_convergence.png)\n")
    w("| Test | Resolutions | Errors | Fitted order |")
    w("|---|---|---|---|")
    for s, d in conv["time"].items():
        w(
            f"| time, `{s}` | dt = {', '.join(fmt(x, 3) for x in d['dt_days'])} d "
            f"| {', '.join(fmt(e, 2) for e in d['error'])} | {d['order']:.2f} |"
        )
    for s, d in conv["space"].items():
        w(
            f"| space, {s} | n_lat = {', '.join(str(x) for x in d['n_lat'])} "
            f"| {', '.join(fmt(e, 2) for e in d['error'])} | {d['order']:.2f} |"
        )
    lg = LEGACY["time_order"]
    w(
        f"\nFor comparison, the previous code's 'SSP2(2,2,2)' stepper measured order "
        f"**{lg['observed_order']}** in time on a similar test (see "
        "`benchmarks/legacy_baseline.json`); its stage coefficients were wrong.\n"
    )

    w("## 3. Accuracy versus cost\n")
    w("Same problem as the temporal check (1-D, 90 cells, 1 yr). Lower-left is better.\n")
    w("![accuracy vs cost](figures/accuracy_vs_cost.png)\n")
    w("| Scheme | dt [d] | Wall [ms] | Max error [G] |")
    w("|---|---|---|---|")
    for s, rows in acc.items():
        for row in rows:
            w(f"| {s} | {row['dt_days']:.3g} | {1e3 * row['wall_s']:.1f} | {row['error']:.2e} |")
    w("")

    w("## 4. Runtime\n")
    w("SSP2, dt = 1 day, best of repeated runs.\n")
    w("![runtime](figures/runtime.png)\n")
    w("| Case | Steps | Wall [s] |")
    w("|---|---|---|")
    for row in runtime["rows"]:
        w(f"| {row['case']} | {row['n_steps']} | {row['wall_s']:.3g} |")
    leg = runtime["legacy"]
    w(
        "| previous code: 1-D, 179 cells, 11 yr (dt = 30 min) | 192,846 "
        f"| ~{leg['1d_11yr_1deg_s']:.0f} |"
    )
    w(
        "| previous code: 2-D, 90 x 180 cells, 1 yr (dt = 30 min) | 17,532 "
        f"| ~{leg['2d_1yr_2deg_s']:.0f} |"
    )
    w(
        "\nPrevious-code timings are extrapolated from shorter runs (see `legacy_baseline.json`). "
        "Most of the speed-up comes from the step size: a correct second-order scheme is "
        "accurate at dt = 1 day, so the 30-minute step is unnecessary. The rest is vectorisation "
        "and a pre-factorised implicit solve.\n"
    )

    w("## 5. Reference solar-cycle run\n")
    p = cycle["params"]
    w(
        f"Three synthetic cycles (23, 24, 25) from zero field: {cycle['n_bmr_per_cycle']} BMRs "
        f"per cycle of 3e21 Mx per polarity, Joy tilt 32.1 deg sin(lat), "
        f"eta = {p['eta_km2s']:.0f} km^2/s, u0 = {p['u0_ms']} m/s, tau = {p['tau_yr']} yr, "
        f"{cycle['n_lat']} latitude cells. Wall time {cycle['wall_s']:.2f} s.\n"
    )
    w("![cycle run](figures/cycle_run.png)\n")
    w("| End of cycle | Axial dipole [G] | North polar field [G] | South polar field [G] |")
    w("|---|---|---|---|")
    for cyc, e in cycle["end_of_cycle"].items():
        w(
            f"| {cyc} (t = {e['t_yr']:.1f} yr) | {e['dipole_G']:+.2f} | "
            f"{e['polar_north_G']:+.2f} | {e['polar_south_G']:+.2f} |"
        )
    w(
        "\nDipole sign changes at t = "
        + ", ".join(f"{t:.1f}" for t in cycle["dipole_reversals_yr"])
        + " yr. These amplitudes come from an uncalibrated synthetic source; they are a "
        "consistency check of the model's behaviour, not a fit to observed polar fields.\n"
    )

    w("## 6. Parameter sensitivity\n")
    w("Two cycles from zero field with identical BMRs; one parameter changed at a time.\n")
    w(
        "| Variant | D, end of cycle 23 [G] | D, end of cycle 24 [G] "
        "| Dipole reversal in cycle 24 [yr] |"
    )
    w("|---|---|---|---|")
    for row in sens:
        rev = row["reversal_in_cycle24_yr"]
        w(
            f"| {row['variant']} | {row['D_end_cycle23_G']:+.2f} | {row['D_end_cycle24_G']:+.2f} "
            f"| {'no reversal' if rev is None else f'{rev:.1f}'} |"
        )
    w(
        "\nWithout a decay term the dipole does not reverse in cycle 24 when starting from zero "
        "field: the result depends on tau (or on an equivalent mechanism) as much as on the "
        "transport parameters.\n"
    )

    w("## 7. Ensemble spread of the cycle-24 dipole\n")
    w(
        "Cycle 24 started from the end state of a fixed cycle-23 run; each member draws its own "
        "BMR times, latitudes and longitudes, with or without Gaussian tilt scatter.\n"
    )
    w("![ensemble](figures/ensemble.png)\n")
    w(
        "With identical BMR fluxes the spread is small because thousands of equal contributions "
        "average out. A lognormal flux distribution (mean flux unchanged) lets a few large "
        "regions dominate, which widens the spread. The real distribution is broad, so the "
        "equal-flux numbers understate the uncertainty of a dipole forecast.\n"
    )
    w(
        "| Tilt scatter | Flux scatter (ln sigma) | Members | D at start [G] "
        "| D at end, mean ± std [G] | 5-95 % [G] | Reversed |"
    )
    w("|---|---|---|---|---|---|---|")
    for e in ens.values():
        w(
            f"| {e['tilt_scatter_deg']:.0f} deg | {e['flux_log_sigma']:g} | {e['members']} "
            f"| {e['D_start_G']:+.2f} "
            f"| {e['D_end_mean_G']:+.2f} ± {e['D_end_std_G']:.2f} "
            f"| {e['D_end_p05_G']:+.2f} to {e['D_end_p95_G']:+.2f} "
            f"| {100 * e['fraction_reversed']:.0f} % |"
        )
    w("")

    w("## 8. 2-D BMR evolution\n")
    w(
        f"One Joy-tilted BMR (1e22 Mx per polarity) at 20 deg N on a {demo2d['grid']} grid with "
        f"Snodgrass & Ulrich (1990) differential rotation, no decay. One year took "
        f"{demo2d['wall_s']:.2f} s. The 2-D and 1-D axial dipoles after one year differ by "
        f"{100 * demo2d['dipole_rel_diff_2d_vs_1d']:.2f} % (the van Leer limiter is nonlinear, "
        "so averaging in longitude does not commute exactly with the scheme; with a linear "
        "scheme the two agree to round-off, see `axisymmetric_consistency`).\n"
    )
    w("![2-D BMR](figures/bmr_2d.png)\n")

    w("## Limitations\n")
    w(
        "- Verification shows the equations are solved correctly; it does not show they match "
        "the Sun. No comparison with observed polar fields or magnetograms is included.\n"
        "- The synthetic source is statistical (Hathaway profile, Joy's law, Hale's law), not a "
        "real BMR record. Use `load_catalogue` for observed emergences.\n"
        "- The 1-D model represents each BMR as a pair of Gaussian rings; it cannot capture "
        "non-axisymmetric effects such as BMR-BMR interaction within a longitude band.\n"
        "- Runtime numbers depend on hardware; ratios are more meaningful than absolute values.\n"
    )
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    from sft import plotting

    t0 = time.perf_counter()
    env = {
        "date": date.today().isoformat(),
        "mode": "quick" if args.quick else "full",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cores": os.cpu_count(),
        "cpu": next(
            (
                ln.split(":", 1)[1].strip()
                for ln in Path("/proc/cpuinfo").read_text().splitlines()
                if ln.startswith("model name")
            ),
            platform.processor() or "unknown CPU",
        )
        if Path("/proc/cpuinfo").exists()
        else platform.processor() or "unknown CPU",
    }

    checks = section_verification(args.quick)
    conv = section_convergence(args.quick)
    acc = section_accuracy_cost(args.quick)
    runtime = section_runtime(args.quick)
    cycle_result, cycle = section_cycle_run(args.quick)
    sens = section_sensitivity(args.quick)
    ens, ens_series = section_ensemble(args.quick)
    result2d, demo2d = section_2d(args.quick)

    print("== figures")
    plotting.plot_time_convergence(conv["time"], figs / "time_convergence.png")
    plotting.plot_space_convergence(conv["space"], figs / "space_convergence.png")
    plotting.plot_accuracy_cost(acc, figs / "accuracy_vs_cost.png")
    rows_1d = [
        (r["case"], r["wall_s"], True) for r in runtime["rows"] if r["case"].startswith("1-D")
    ]
    rows_2d = [
        (r["case"], r["wall_s"], True) for r in runtime["rows"] if r["case"].startswith("2-D")
    ]
    bars = [
        *rows_1d,
        ("previous code: 1-D, 179 cells, 11 yr", LEGACY["runtime"]["1d_11yr_1deg_s"], False),
        *rows_2d,
        ("previous code: 2-D, 90 x 180, 1 yr", LEGACY["runtime"]["2d_1yr_2deg_s"], False),
    ]
    plotting.plot_runtime(bars, figs / "runtime.png")
    plotting.plot_cycle_run(
        cycle_result, figs / "cycle_run.png", title="Reference run: synthetic cycles 23-25"
    )
    labels = {
        "no_tilt_scatter": "random emergence only",
        "tilt_scatter_15deg": "+ tilt scatter 15 deg",
        "tilt_15deg_and_lognormal_flux": "+ tilt and flux scatter",
    }
    plotting.plot_ensemble(
        {labels[k]: v for k, v in ens_series.items()},
        figs / "ensemble.png",
        title="Cycle 24 axial dipole: ensemble spread",
    )
    plotting.plot_2d_snapshots(result2d, figs / "bmr_2d.png", indices=[0, 4, 12, -1])

    summary = {
        "environment": env,
        "checks_passed": sum(c["passed"] for c in checks),
        "checks_total": len(checks),
        "time_orders": {k: v["order"] for k, v in conv["time"].items()},
        "space_orders": {k: v["order"] for k, v in conv["space"].items()},
        "runtime": runtime,
        "cycle_run": cycle,
        "sensitivity": sens,
        "ensemble": ens,
        "bmr_2d": demo2d,
        "benchmark_wall_s": time.perf_counter() - t0,
    }
    (out / "verification.json").write_text(json.dumps(checks, indent=2))
    (out / "convergence.json").write_text(json.dumps({**conv, "accuracy_vs_cost": acc}, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(out / "REPORT.md", env, checks, conv, acc, runtime, cycle, sens, ens, demo2d)
    print(f"done in {summary['benchmark_wall_s']:.0f} s -> {out}")
    if summary["checks_passed"] != summary["checks_total"]:
        raise SystemExit("some verification checks failed")


if __name__ == "__main__":
    main()
