"""The SFT model: parameters, time integration loop and run results."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from . import diagnostics as diag
from .constants import DAY_S, YEAR_S
from .flows import SNODGRASS_ULRICH_1990, differential_rotation, meridional_flow
from .grid import Grid
from .imex import IMEXStepper
from .operators import Advection, DiffusionDecay


@dataclass
class TransportParams:
    """Physical transport parameters.

    eta_km2s   : supergranular diffusivity [km^2/s]
    u0_ms      : meridional flow amplitude [m/s]
    tau_yr     : exponential decay time [yr]; None disables decay
    flow       : meridional profile name ("sin2lat", "sin_cutoff") or callable f(lat, u0, **kw)
    flow_kwargs: extra keyword arguments for the flow profile
    rotation   : (A, B, C) in deg/day for Omega = A + B sin^2 + C sin^4 relative to the
                 Carrington frame; None disables differential rotation (2-D runs only)
    """

    eta_km2s: float = 500.0
    u0_ms: float = 12.5
    tau_yr: float | None = None
    flow: str | Callable = "sin2lat"
    flow_kwargs: dict = field(default_factory=dict)
    rotation: tuple[float, float, float] | None = SNODGRASS_ULRICH_1990

    @property
    def eta(self) -> float:
        return self.eta_km2s * 1e6

    @property
    def tau_s(self) -> float | None:
        return None if self.tau_yr is None else self.tau_yr * YEAR_S


@dataclass
class RunResult:
    """Snapshots of a run. ``B`` has time on axis 0, then latitude, then longitude/members."""

    grid: Grid
    t_yr: np.ndarray
    B: np.ndarray
    params: TransportParams
    scheme: str
    dt_days: float
    wall_time_s: float
    n_steps: int
    n_events: int = 0

    def _lat_first(self) -> np.ndarray:
        return np.moveaxis(self.B, 0, -1)

    def lon_avg(self) -> np.ndarray:
        """Longitude-averaged snapshots, shape (n_time, n_lat[, members])."""
        return np.moveaxis(diag.lon_average(self._lat_first(), self.grid), -1, 0)

    def butterfly(self) -> np.ndarray:
        """Time-latitude map of the longitude-averaged field, shape (n_lat, n_time)."""
        Bbar = self.lon_avg()
        if Bbar.ndim != 2:
            raise ValueError("butterfly() needs a single run, not an ensemble")
        return Bbar.T

    def dipole(self) -> np.ndarray:
        return np.moveaxis(np.atleast_1d(diag.axial_dipole(self._lat_first(), self.grid)), -1, 0)

    def signed_flux(self) -> np.ndarray:
        return np.moveaxis(np.atleast_1d(diag.signed_flux(self._lat_first(), self.grid)), -1, 0)

    def unsigned_flux(self) -> np.ndarray:
        return np.moveaxis(np.atleast_1d(diag.unsigned_flux(self._lat_first(), self.grid)), -1, 0)

    def polar_fields(self, cap_deg: float = 60.0):
        n, s = diag.polar_field(self._lat_first(), self.grid, cap_deg)
        return np.moveaxis(np.atleast_1d(n), -1, 0), np.moveaxis(np.atleast_1d(s), -1, 0)


class SFTModel:
    """Surface flux transport on a :class:`Grid`.

    Axisymmetric grids (``n_lon == 1``) evolve the longitude-averaged field; the
    state may carry extra columns (shape (n_lat, k)) for independent ensemble
    members. 2-D grids transform to longitudinal Fourier modes, solve diffusion
    implicitly per mode, and apply differential rotation as an exact phase shift
    in a Strang splitting around each IMEX step.
    """

    def __init__(
        self,
        grid: Grid,
        params: TransportParams | None = None,
        dt_days: float = 1.0,
        scheme: str = "ssp2",
        limiter: str = "vanleer",
    ):
        self.grid = grid
        self.params = params or TransportParams()
        self.dt_days = float(dt_days)
        self.dt = self.dt_days * DAY_S
        self.scheme = scheme
        self.limiter = limiter

        p = self.params
        u_faces = meridional_flow(grid.lat_faces, p.u0_ms, p.flow, **p.flow_kwargs)
        self.advection = Advection(grid, u_faces, limiter)
        ms = np.arange(grid.n_modes)
        self.implicit = DiffusionDecay(grid, p.eta, p.tau_s, ms)
        self.stepper = IMEXStepper(scheme, self.dt, self.implicit)

        self._omega_m = None
        if grid.is_2d and p.rotation is not None:
            omega = differential_rotation(grid.lat, p.rotation)
            self._omega_m = omega[:, None] * ms[None, :]
        self._rot_half = self._rotation_phase(self.dt)

        cfl = self.cfl
        if cfl > 1.0:
            raise ValueError(
                f"advection CFL = {cfl:.2f} > 1 with dt = {dt_days} d; reduce dt_days below "
                f"{dt_days / cfl:.3g} d (CFL <= 0.5 keeps the limiter's positivity guarantee)"
            )

    @property
    def cfl(self) -> float:
        """Advection Courant number of the explicit part (should be <= 0.5)."""
        return self.advection.cfl(self.dt)

    # -- state transforms -------------------------------------------------
    def _to_state(self, B: np.ndarray) -> np.ndarray:
        B = np.array(B, dtype=float)
        if self.grid.is_2d:
            if B.shape != (self.grid.n_lat, self.grid.n_lon):
                raise ValueError(f"2-D field must have shape {(self.grid.n_lat, self.grid.n_lon)}")
            return np.fft.rfft(B, axis=1)
        if B.shape[0] != self.grid.n_lat or B.ndim > 2:
            raise ValueError(
                f"field must have shape ({self.grid.n_lat},) or ({self.grid.n_lat}, k)"
            )
        return B

    def _to_field(self, U: np.ndarray) -> np.ndarray:
        if self.grid.is_2d:
            return np.fft.irfft(U, n=self.grid.n_lon, axis=1)
        return U.copy()

    def _explicit(self, source):
        if self.grid.is_2d:
            n_lon = self.grid.n_lon

            def explicit(U, t_s):
                B = np.fft.irfft(U, n=n_lon, axis=1)
                out = self.advection.tendency(B)
                if source is not None:
                    out = out + source(t_s / YEAR_S)
                return np.fft.rfft(out, axis=1)

        else:

            def explicit(U, t_s):
                out = self.advection.tendency(U)
                if source is not None:
                    out = out + np.reshape(
                        source(t_s / YEAR_S), (U.shape[0],) + (1,) * (U.ndim - 1)
                    )
                return out

        return explicit

    def _rotation_phase(self, dt: float):
        """Exact differential-rotation phase factor for half of a step of ``dt`` seconds."""
        return None if self._omega_m is None else np.exp(-0.5j * dt * self._omega_m)

    def _step_state(self, U, t_s, explicit, stepper=None, rot_half=None):
        stepper = stepper or self.stepper
        rot_half = self._rot_half if stepper is self.stepper else rot_half
        if rot_half is not None:
            U = U * rot_half
        U = stepper.step(U, t_s, explicit)
        if rot_half is not None:
            U = U * rot_half
        return U

    def _insert(self, U, events):
        for ev in events:
            f = ev.field(self.grid)
            if self.grid.is_2d:
                U = U + np.fft.rfft(f, axis=1)
            elif U.ndim == 2:
                U[:, ev.member] += f
            elif ev.member == 0:
                U = U + f
            else:
                raise ValueError("event.member > 0 needs an ensemble state of shape (n_lat, k)")
        return U

    # -- public API -------------------------------------------------------
    def step(self, B: np.ndarray, t_yr: float = 0.0, source=None) -> np.ndarray:
        """Advance a field by one time step (convenience wrapper; ``run`` is faster)."""
        U = self._step_state(self._to_state(B), t_yr * YEAR_S, self._explicit(source))
        return self._to_field(U)

    def run(
        self,
        B0: np.ndarray,
        t_end_yr: float,
        *,
        t_start_yr: float = 0.0,
        events: Iterable = (),
        source: Callable | None = None,
        save_every_days: float = 27.0,
        keep_2d: bool = True,
    ) -> RunResult:
        """Integrate from ``t_start_yr`` to exactly ``t_end_yr``.

        If the interval is not a whole number of steps, the last step is shortened.

        events : BMR-like objects (``time_yr``, ``member``, ``field(grid)``) inserted at
                 the step boundary nearest their time; events outside the run are ignored.
        source : optional continuous source S(t_yr) -> field [G/s], treated explicitly.
        keep_2d: on 2-D grids, store full maps (True) or only longitude averages (False).
        """
        span_steps = (t_end_yr - t_start_yr) * YEAR_S / self.dt
        n_full = int(np.floor(span_steps + 1e-9))
        remainder = (span_steps - n_full) * self.dt
        if remainder < 1e-9 * self.dt:
            remainder = 0.0
        n_steps = n_full + (remainder > 0.0)
        if n_steps < 1:
            raise ValueError("t_end_yr must be after t_start_yr")
        save_every = max(1, int(round(save_every_days * DAY_S / self.dt)))

        schedule: dict[int, list] = defaultdict(list)
        for ev in events:
            k = int(round((ev.time_yr - t_start_yr) * YEAR_S / self.dt))
            if 0 <= k < n_steps:
                schedule[k].append(ev)
        n_events = sum(len(v) for v in schedule.values())

        explicit = self._explicit(source)
        t0_s = t_start_yr * YEAR_S
        U = self._to_state(B0)

        def snapshot(U):
            B = self._to_field(U)
            return B.mean(axis=1) if (self.grid.is_2d and not keep_2d) else B

        times, snaps = [t_start_yr], [snapshot(U)]
        wall0 = time.perf_counter()
        for k in range(n_full):
            if k in schedule:
                U = self._insert(U, schedule[k])
            U = self._step_state(U, t0_s + k * self.dt, explicit)
            if (k + 1) % save_every == 0 or k + 1 == n_steps:
                times.append(t_start_yr + (k + 1) * self.dt / YEAR_S)
                snaps.append(snapshot(U))
        if remainder > 0.0:
            if n_full in schedule:
                U = self._insert(U, schedule[n_full])
            last = IMEXStepper(self.scheme, remainder, self.implicit)
            U = self._step_state(
                U, t0_s + n_full * self.dt, explicit, last, self._rotation_phase(remainder)
            )
            times.append(t_end_yr)
            snaps.append(snapshot(U))
        wall = time.perf_counter() - wall0

        grid = self.grid if (keep_2d or not self.grid.is_2d) else Grid(self.grid.n_lat, 1)
        return RunResult(
            grid=grid,
            t_yr=np.array(times),
            B=np.array(snaps),
            params=self.params,
            scheme=self.scheme,
            dt_days=self.dt_days,
            wall_time_s=wall,
            n_steps=n_steps,
            n_events=n_events,
        )
