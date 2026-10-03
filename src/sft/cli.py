"""Command-line interface.

sft verify [--quick] [--json out.json]
sft run [--cycles 2] [--first-cycle 23] [--tau 5] ... [--out run.npz] [--plot run.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import numpy as np

from . import verification
from .grid import Grid
from .imex import SCHEMES
from .model import SFTModel, TransportParams
from .operators import LIMITERS
from .sources import SyntheticCycle, generate_cycles, load_catalogue


def _cmd_verify(args) -> int:
    results = verification.run_all(quick=args.quick, names=args.check or None)
    n_pass = sum(r.passed for r in results)
    print(f"\n{n_pass}/{len(results)} checks passed")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in results], fh, indent=2)
    return 0 if n_pass == len(results) else 1


def _cmd_run(args) -> int:
    params = TransportParams(
        eta_km2s=args.eta,
        u0_ms=args.u0,
        tau_yr=None if args.tau <= 0 else args.tau,
        flow=args.flow,
    )
    grid = Grid(args.n_lat, args.n_lon)
    model = SFTModel(grid, params, dt_days=args.dt, scheme=args.scheme, limiter=args.limiter)

    if args.catalogue:
        events = load_catalogue(args.catalogue)
        if not events:
            raise SystemExit(f"no BMRs in {args.catalogue}")
        t0 = events[0].time_yr
        t_end = t0 + args.years if args.years else events[-1].time_yr + 1.0
    else:
        template = SyntheticCycle(
            cycle=args.first_cycle,
            n_bmr=args.n_bmr,
            flux_mx=args.flux,
            tilt_scatter_deg=args.tilt_scatter,
        )
        events = generate_cycles(template, args.cycles, seed=args.seed)
        t0 = 0.0
        t_end = args.years or args.cycles * template.length_yr

    B0 = np.zeros((grid.n_lat, grid.n_lon)) if grid.is_2d else np.zeros(grid.n_lat)
    result = model.run(
        B0, t_end, t_start_yr=t0, events=events, keep_2d=False, save_every_days=args.save_every
    )
    north, south = result.polar_fields()
    dip = result.dipole()
    print(f"{grid}  scheme={args.scheme}  dt={args.dt} d  CFL={model.cfl:.3f}")
    print(f"{result.n_events} BMRs, {result.n_steps} steps in {result.wall_time_s:.2f} s")
    print(
        f"final axial dipole {dip[-1]:+.3f} G, polar fields N {north[-1]:+.3f} G, "
        f"S {south[-1]:+.3f} G"
    )

    if args.out:
        np.savez_compressed(
            args.out,
            t_yr=result.t_yr,
            lat_deg=result.grid.lat_deg,
            butterfly=result.butterfly(),
            dipole=dip,
            polar_north=north,
            polar_south=south,
            params=json.dumps(
                {k: v for k, v in asdict(params).items() if k != "flow_kwargs"}, default=str
            ),
        )
        print(f"saved {args.out}")
    if args.plot:
        from .plotting import plot_cycle_run

        plot_cycle_run(result, args.plot)
        print(f"saved {args.plot}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sft", description="Surface Flux Transport solver")
    sub = ap.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify", help="run the verification checks")
    v.add_argument("--quick", action="store_true", help="coarser grids, ~10 s")
    v.add_argument(
        "--check",
        action="append",
        choices=sorted(verification.CHECKS),
        help="run only this check (repeatable)",
    )
    v.add_argument("--json", help="write results to this JSON file")
    v.set_defaults(func=_cmd_verify)

    r = sub.add_parser("run", help="simulate synthetic cycles or a BMR catalogue")
    r.add_argument("--catalogue", help="CSV of BMRs (see sft.sources.load_catalogue)")
    r.add_argument("--cycles", type=int, default=2)
    r.add_argument("--first-cycle", type=int, default=23)
    r.add_argument("--n-bmr", type=int, default=2000, help="BMRs per cycle")
    r.add_argument("--flux", type=float, default=3e21, help="flux per polarity [Mx]")
    r.add_argument("--tilt-scatter", type=float, default=0.0, help="Joy tilt scatter [deg]")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument(
        "--years",
        type=float,
        default=None,
        help="run length [yr] (default: cycles x 11, or catalogue span + 1 yr)",
    )
    r.add_argument("--n-lat", type=int, default=180)
    r.add_argument("--n-lon", type=int, default=1, help="1 = axisymmetric")
    r.add_argument("--dt", type=float, default=1.0, help="time step [days]")
    r.add_argument("--eta", type=float, default=500.0, help="diffusivity [km^2/s]")
    r.add_argument("--u0", type=float, default=12.5, help="meridional flow [m/s]")
    r.add_argument("--tau", type=float, default=5.0, help="decay time [yr]; <= 0 disables")
    r.add_argument("--flow", default="sin2lat", choices=["sin2lat", "sin_cutoff"])
    r.add_argument("--scheme", default="ssp2", choices=SCHEMES)
    r.add_argument("--limiter", default="vanleer", choices=sorted(LIMITERS))
    r.add_argument("--save-every", type=float, default=27.0, help="snapshot cadence [days]")
    r.add_argument("--out", help="write results to this .npz file")
    r.add_argument("--plot", help="write a butterfly/polar-field figure to this PNG")
    r.set_defaults(func=_cmd_run)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
