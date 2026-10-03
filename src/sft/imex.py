"""Implicit-explicit Runge-Kutta time steppers.

All schemes advance dU/dt = E(U, t) + L U, where E (advection and sources) is
treated explicitly and the linear operator L (diffusion and decay) implicitly.

Schemes
-------
ssp2  : SSP2(2,2,2) of Pareschi & Russo (2005). Explicit part is Heun's SSP
        method; implicit part is L-stable. Second order. Default.
ars2  : ARS(2,2,2) of Ascher, Ruuth & Spiteri (1997). Stiffly accurate, L-stable,
        second order. Its explicit part has a negative weight, so it is not SSP.
euler : IMEX Euler (forward/backward Euler). First order; kept as a baseline.

Both second-order schemes use gamma = 1 - 1/sqrt(2) on the implicit diagonal,
so one factorisation of (I - gamma dt L) serves every stage.
"""

from __future__ import annotations

import numpy as np

GAMMA = 1.0 - 1.0 / np.sqrt(2.0)
DELTA = 1.0 - 1.0 / (2.0 * GAMMA)  # ARS(2,2,2) explicit weight, approximately -0.707

SCHEMES = ("ssp2", "ars2", "euler")
SCHEME_ORDER = {"ssp2": 2, "ars2": 2, "euler": 1}


class IMEXStepper:
    """Advance a state by one step of size ``dt`` seconds with a fixed implicit operator."""

    def __init__(self, scheme: str, dt: float, implicit_op):
        if scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")
        self.scheme = scheme
        self.dt = float(dt)
        self.L = implicit_op
        coef = self.dt if scheme == "euler" else GAMMA * self.dt
        self._solve = implicit_op.factorize(coef)

    def step(self, U: np.ndarray, t: float, explicit) -> np.ndarray:
        """Return U at time t + dt. ``explicit(U, t)`` evaluates E; t is in seconds."""
        dt, solve, L = self.dt, self._solve, self.L.apply

        if self.scheme == "ssp2":
            U1 = solve(U)
            e1, i1 = explicit(U1, t), L(U1)
            U2 = solve(U + dt * e1 + (1.0 - 2.0 * GAMMA) * dt * i1)
            e2, i2 = explicit(U2, t + dt), L(U2)
            return U + 0.5 * dt * (e1 + e2 + i1 + i2)

        if self.scheme == "ars2":
            e0 = explicit(U, t)
            U1 = solve(U + GAMMA * dt * e0)
            e1 = explicit(U1, t + GAMMA * dt)
            rhs = U + dt * (DELTA * e0 + (1.0 - DELTA) * e1) + (1.0 - GAMMA) * dt * L(U1)
            return solve(rhs)

        return solve(U + dt * explicit(U, t))
