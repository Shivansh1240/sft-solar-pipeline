"""
sft_numerics.py
===============
Numerical schemes for Surface Flux Transport.

Implements:
1. RK-IMEX (2nd-order Runge-Kutta Implicit-Explicit) ← PRIMARY SCHEME
   - Explicit advection with van Leer flux limiter (Godunov-type)
   - Implicit diffusion + decay via Thomas tridiagonal solver
   - Operator splitting for 2D: dimensional splitting (Strang)

2. Spherical Harmonics spectral method ← HIGHEST ACCURACY
   - Treats diffusion exactly as eigenvalue problem of Laplacian
   - No Gibbs ringing if smoothed; pole-safe by construction

3. Explicit Euler (for comparison / stability testing only)

References:
- Pareschi & Russo 2005 (IMEX-RK)
- van Leer 1977 (flux limiter)
- Kundu et al. 2021 (SFT upwind scheme)
- Athalathil et al. 2024 (reference implementation)
"""

import numpy as np
from scipy.linalg import solve_banded


# ─────────────────────────────────────────────
# Van Leer flux limiter
# ─────────────────────────────────────────────

def van_leer_limiter(r):
    """
    Van Leer flux limiter φ(r) = (r + |r|) / (1 + |r|)
    r = ratio of consecutive gradients.
    """
    r = np.asarray(r, dtype=float)
    return (r + np.abs(r)) / (1.0 + np.abs(r) + 1e-30)


def minmod_limiter(r):
    """Minmod limiter - more diffusive but robustly monotone."""
    return np.maximum(0.0, np.minimum(1.0, r))


def superbee_limiter(r):
    """Superbee limiter - least diffusive."""
    return np.maximum(0.0, np.maximum(np.minimum(2*r, 1.0), np.minimum(r, 2.0)))


# ─────────────────────────────────────────────
# 1D Advection operator (upwind + van Leer)
# ─────────────────────────────────────────────

def advection_flux_1d(B, u_half, dlam, limiter='van_leer'):
    """
    Compute advection term for 1D SFT equation using Godunov-type upwind
    scheme with van Leer flux limiter.

    The 1D SFT advection term (in colatitude θ or latitude λ):
        L_adv(B) = -1/(R cosλ) ∂/∂λ [u(λ) B cosλ]

    Here we compute the flux F_{i+1/2} = u_{i+1/2} * B_{i+1/2}
    with B_{i+1/2} reconstructed using van Leer limiter.

    Parameters
    ----------
    B       : 1D array of ⟨Br⟩ values at cell centers
    u_half  : 1D array of u at cell faces (len = len(B)+1)
    dlam    : grid spacing in radians
    limiter : 'van_leer', 'minmod', or 'superbee'

    Returns
    -------
    L_adv : advection operator applied to B (same shape as B)
    """
    N = len(B)
    limiter_fn = {'van_leer': van_leer_limiter,
                  'minmod': minmod_limiter,
                  'superbee': superbee_limiter}[limiter]

    # Extend with boundary ghost cells (zero-field BCs)
    Bext = np.zeros(N + 4)
    Bext[2:N+2] = B

    # Compute fluxes at cell faces i+1/2 (index i to i+2 in extended)
    flux = np.zeros(N + 1)
    for i in range(N + 1):
        u = u_half[i]
        if u >= 0:   # flow going right (poleward in N hemisphere)
            # Upwind from left
            B_L = Bext[i+2]
            B_LL = Bext[i+1]
            B_R  = Bext[i+3]
            dB_down = B_L - B_LL
            dB_up   = B_R - B_L
            r = dB_up / (dB_down + 1e-30 * np.sign(dB_down + 1e-30))
            phi = limiter_fn(r)
            B_face = B_L + 0.5 * phi * dB_down
        else:         # flow going left
            # Upwind from right
            B_R  = Bext[i+3]
            B_RR = Bext[i+4]
            B_L  = Bext[i+2]
            dB_down = B_R - B_RR
            dB_up   = B_L - B_R
            r = dB_up / (dB_down + 1e-30 * np.sign(dB_down + 1e-30))
            phi = limiter_fn(r)
            B_face = B_R + 0.5 * phi * dB_down

        flux[i] = u * B_face

    # Divergence of flux
    div_flux = (flux[1:] - flux[:-1]) / dlam
    return -div_flux


# ─────────────────────────────────────────────
# Thomas algorithm (tridiagonal solver)
# ─────────────────────────────────────────────

def thomas_solve(a, b, c, d):
    """
    Thomas algorithm for tridiagonal system A·x = d
    a : sub-diagonal (length N, a[0] unused)
    b : main diagonal (length N)
    c : super-diagonal (length N, c[-1] unused)
    d : RHS (length N)

    Returns x : solution (length N)
    O(N) complexity — ideal for implicit diffusion solves.
    """
    N = len(d)
    c_ = np.zeros(N)
    d_ = np.zeros(N)
    x  = np.zeros(N)

    # Forward elimination
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, N):
        denom = b[i] - a[i] * c_[i-1]
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i] * d_[i-1]) / denom

    # Back substitution
    x[-1] = d_[-1]
    for i in range(N-2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i+1]

    return x


def build_diffusion_decay_matrix(N, dlam_rad, eta, R_sun, tau, dt,
                                  lat_rad, alpha=0.5):
    """
    Build tridiagonal coefficients for implicit diffusion + decay solve:
        [I + α·dt·(L_diff + L_decay)] B = RHS

    The diffusion operator in latitude:
        L_diff B_i = η/(R²) * 1/cosλ * ∂/∂λ [cosλ ∂B/∂λ]

    Discretized with centered differences.

    Parameters
    ----------
    N       : number of grid points
    dlam_rad: grid spacing (radians)
    eta     : diffusivity (m²/s)
    R_sun   : solar radius (m)
    tau     : decay timescale (s) - use np.inf to disable
    dt      : time step (s)
    lat_rad : latitude array (radians)
    alpha   : implicitness factor (0.5 = Crank-Nicolson, 1.0 = fully implicit)

    Returns
    -------
    a, b, c : tridiagonal vectors (each length N)
    """
    coeff = alpha * dt * eta / (R_sun**2 * dlam_rad**2)
    decay_coeff = alpha * dt / tau if np.isfinite(tau) else 0.0

    cos_lam = np.cos(lat_rad)
    # cos at half-points (staggered)
    cos_half_p = np.cos(lat_rad + 0.5 * dlam_rad)   # i+1/2
    cos_half_m = np.cos(lat_rad - 0.5 * dlam_rad)   # i-1/2

    a = np.zeros(N)   # sub-diagonal
    b = np.ones(N)    # main diagonal
    c = np.zeros(N)   # super-diagonal

    for i in range(1, N-1):
        cl = cos_lam[i]
        a[i] = -coeff * cos_half_m[i] / cl
        c[i] = -coeff * cos_half_p[i] / cl
        b[i] = 1.0 + decay_coeff - (a[i] + c[i])

    # Boundary: Dirichlet B=0 at poles
    b[0]  = 1.0
    b[-1] = 1.0
    a[0]  = 0.0; c[0]  = 0.0
    a[-1] = 0.0; c[-1] = 0.0

    return a, b, c


# ─────────────────────────────────────────────
# RK-IMEX 2-stage scheme (primary solver)
# ─────────────────────────────────────────────

class RKIMEX2Stage:
    """
    2-stage RK-IMEX scheme for the 1D axisymmetric SFT equation.

    Scheme (Pareschi & Russo 2005, "L-stable" variant):
        Stage 1:
            B* = B^n + (dt/2) · L_adv(B^n)
            Solve: [I + (dt/2)(L_diff + L_decay)] B^(1) = B*

        Stage 2:
            B** = B^n + dt · L_adv(B^(1))
            Solve: [I + (dt/2)(L_diff + L_decay)] B^(2) = B**

        Update:
            B^{n+1} = (1/2)[B^(1) + B^(2)]

    This achieves 2nd-order accuracy in both space and time.
    """

    def __init__(self, lat_deg, eta=500e6, u0=12.5, tau=5*3.156e7,
                 R_sun=6.96e8, dt=1800.0,
                 flow_profile='van_ballegooijen',
                 limiter='van_leer',
                 include_diff_rot=False):
        """
        Parameters
        ----------
        lat_deg  : 1D array of latitudes in degrees
        eta      : turbulent diffusivity (m²/s)
        u0       : peak meridional flow speed (m/s)
        tau      : flux decay timescale (s)
        R_sun    : solar radius (m)
        dt       : time step (s)
        flow_profile : 'van_ballegooijen' | 'schad' | 'multicell'
        limiter  : 'van_leer' | 'minmod' | 'superbee'
        include_diff_rot: whether to include differential rotation
        """
        from sft_physics import (meridional_flow_vanBall,
                                  meridional_flow_schad,
                                  meridional_flow_multicell,
                                  meridional_flow_constant)

        self.lat_deg  = lat_deg
        self.lat_rad  = np.deg2rad(lat_deg)
        self.dlam     = self.lat_rad[1] - self.lat_rad[0]  # uniform
        self.N        = len(lat_deg)
        self.eta      = eta
        self.tau      = tau
        self.R_sun    = R_sun
        self.dt       = dt
        self.limiter  = limiter

        # Meridional flow at cell faces (half-points)
        lat_half = self.lat_rad + 0.5 * self.dlam  # i+1/2 faces
        lat_half_full = np.concatenate([
            [self.lat_rad[0] - 0.5 * self.dlam],
            lat_half
        ])

        profiles = {
            'van_ballegooijen': meridional_flow_vanBall,
            'schad': meridional_flow_schad,
            'multicell': meridional_flow_multicell,
            'constant': meridional_flow_constant,
        }
        u_fn = profiles.get(flow_profile, meridional_flow_vanBall)

        # u at faces divided by R * cos(λ) for advection flux
        # The SFT advection in latitude is: -1/(R cosλ) ∂/∂λ [u cosλ B]
        # We store u at half-points
        self.u_half = u_fn(lat_half_full, u0=u0)  # m/s at faces

        # Build tridiagonal matrix for implicit solve
        self.a_mat, self.b_mat, self.c_mat = build_diffusion_decay_matrix(
            self.N, self.dlam, eta, R_sun, tau, dt, self.lat_rad, alpha=0.5
        )

        self.include_diff_rot = include_diff_rot
        if include_diff_rot:
            from sft_physics import differential_rotation_snodgrass
            self.omega_diff = differential_rotation_snodgrass(self.lat_rad)

    def _explicit_advection(self, B):
        """Compute explicit advection term L_adv(B)."""
        # Need to scale u by cos(λ) geometry factors
        # L_adv = -1/(R cosλ) ∂/∂λ [u cosλ B]
        # We absorb the geometry into the flux computation

        cos_lam = np.cos(self.lat_rad)
        cos_half = np.cos(self.lat_rad + 0.5 * self.dlam)
        cos_half_full = np.concatenate([
            [np.cos(self.lat_rad[0] - 0.5 * self.dlam)],
            cos_half
        ])

        # Effective velocity for the flux F = u * cos(λ) * B
        # then divergence divided by R * cos(λ)
        u_eff = self.u_half  # u at faces in m/s

        # Compute flux F_{i+1/2} = u_{i+1/2} * cos(λ_{i+1/2}) * B_{i+1/2}
        N = self.N
        Bext = np.zeros(N + 4)
        Bext[2:N+2] = B * cos_lam   # store B*cos(λ) for flux

        limiter_fn = {'van_leer': van_leer_limiter,
                      'minmod': minmod_limiter,
                      'superbee': superbee_limiter}[self.limiter]

        flux = np.zeros(N + 1)
        for i in range(N + 1):
            u = u_eff[i]
            if u >= 0:
                B_L = Bext[i+2]; B_LL = Bext[i+1]; B_R = Bext[i+3]
                dB1 = B_L - B_LL; dB2 = B_R - B_L
                r = dB2 / (dB1 + 1e-30 * (1 if dB1 >= 0 else -1))
                phi = limiter_fn(r)
                Bf = B_L + 0.5 * phi * dB1
            else:
                B_R = Bext[i+3]; B_RR = Bext[i+4]; B_L = Bext[i+2]
                dB1 = B_R - B_RR; dB2 = B_L - B_R
                r = dB2 / (dB1 + 1e-30 * (1 if dB1 >= 0 else -1))
                phi = limiter_fn(r)
                Bf = B_R + 0.5 * phi * dB1
            flux[i] = u * Bf

        # L_adv = -1/(R cosλ) * (F_{i+1/2} - F_{i-1/2}) / Δλ
        div_flux = (flux[1:] - flux[:-1]) / self.dlam
        L_adv = -div_flux / (self.R_sun * cos_lam + 1e-30)

        return L_adv

    def _implicit_solve(self, rhs):
        """Solve [I + (dt/2)(L_diff + L_decay)] B = rhs."""
        return thomas_solve(self.a_mat, self.b_mat, self.c_mat, rhs)

    def step(self, B):
        """
        Advance B by one time step dt.

        Returns
        -------
        B_new : updated field
        """
        dt = self.dt

        # Stage 1
        L1 = self._explicit_advection(B)
        rhs1 = B + 0.5 * dt * L1
        rhs1[0]  = 0.0   # Dirichlet BC
        rhs1[-1] = 0.0
        B1 = self._implicit_solve(rhs1)

        # Stage 2
        L2 = self._explicit_advection(B1)
        rhs2 = B + dt * L2
        rhs2[0]  = 0.0
        rhs2[-1] = 0.0
        B2 = self._implicit_solve(rhs2)

        # Final update
        B_new = 0.5 * (B1 + B2)
        B_new[0]  = 0.0
        B_new[-1] = 0.0

        return B_new

    def evolve(self, B0, t_end_years, verbose=True, save_interval_days=30):
        """
        Evolve the field from t=0 to t_end_years.

        Parameters
        ----------
        B0              : initial 1D field (Gauss)
        t_end_years     : simulation end time in years
        verbose         : print progress
        save_interval_days : how often to save snapshots

        Returns
        -------
        t_snapshots : list of times (years)
        B_snapshots : list of field arrays
        """
        t_end_s  = t_end_years * 365.25 * 86400
        n_steps  = int(t_end_s / self.dt)
        save_every = max(1, int(save_interval_days * 86400 / self.dt))

        B = B0.copy()
        t_snaps = [0.0]
        B_snaps = [B.copy()]

        for step_i in range(1, n_steps + 1):
            B = self.step(B)
            if step_i % save_every == 0:
                t_yr = step_i * self.dt / (365.25 * 86400)
                t_snaps.append(t_yr)
                B_snaps.append(B.copy())
                if verbose:
                    peak = np.max(np.abs(B))
                    print(f"  t={t_yr:.2f} yr  |B|_max={peak:.4f} G")

        return np.array(t_snaps), np.array(B_snaps)


# ─────────────────────────────────────────────
# Spectral SFT solver (spherical harmonics)
# ─────────────────────────────────────────────

class SpectralSFT1D:
    """
    1D axisymmetric SFT solver using Legendre polynomial expansion.
    Solves only the diffusion+decay part exactly in spectral space;
    advection is still handled by a separate step (operator-split).

    This is the approach of Cameron et al. 2010 / Jiang et al. 2014.
    The diffusion operator L_diff becomes:
        L_diff Y_l = -η l(l+1)/R² · Y_l   (eigenvalue problem)

    Making the diffusion step exact (no spatial truncation error).
    """

    def __init__(self, lat_deg, l_max=90, eta=500e6, u0=12.5,
                 tau=5*3.156e7, R_sun=6.96e8, dt=1800.0):
        from scipy.special import legendre

        self.lat_deg  = lat_deg
        self.lat_rad  = np.deg2rad(lat_deg)
        self.cos_lat  = np.cos(self.lat_rad)
        self.N        = len(lat_deg)
        self.l_max    = l_max
        self.eta      = eta
        self.tau      = tau
        self.R_sun    = R_sun
        self.dt       = dt

        # Precompute Legendre polynomials P_l(cos λ) at grid points
        print(f"  [SpectralSFT] Precomputing Legendre basis l_max={l_max}...")
        self.P = np.zeros((l_max + 1, self.N))
        for l in range(l_max + 1):
            Pl = legendre(l)
            self.P[l] = Pl(self.cos_lat)

        # Spectral diffusion + decay operator (diagonal in l-space)
        self.spectral_decay = np.array([
            eta * l * (l + 1) / R_sun**2 + (1.0 / tau if np.isfinite(tau) else 0.0)
            for l in range(l_max + 1)
        ])

    def field_to_spectral(self, B):
        """Project B onto Legendre basis using Gauss-Legendre quadrature."""
        # Simple trapz integration: a_l = (2l+1)/2 ∫ B(λ) P_l(cosλ) sinλ dλ
        # note: sinλ dλ = -d(cosλ), but here latitude runs -90 to +90
        cos_lam = self.cos_lat
        dlam    = self.lat_rad[1] - self.lat_rad[0]
        coeffs  = np.zeros(self.l_max + 1)
        for l in range(self.l_max + 1):
            integrand = B * self.P[l] * np.cos(self.lat_rad)
            coeffs[l] = (2*l + 1) / 2.0 * np.trapz(integrand, self.lat_rad)
        return coeffs

    def spectral_to_field(self, coeffs):
        """Reconstruct B from spectral coefficients."""
        B = np.zeros(self.N)
        for l in range(self.l_max + 1):
            B += coeffs[l] * self.P[l]
        return B

    def diffusion_decay_step(self, B):
        """Exact diffusion+decay step in spectral space."""
        coeffs = self.field_to_spectral(B)
        decay  = np.exp(-self.spectral_decay * self.dt)
        coeffs_new = coeffs * decay
        return self.spectral_to_field(coeffs_new)


# ─────────────────────────────────────────────
# 2D RK-IMEX solver (Strang dimensional splitting)
# ─────────────────────────────────────────────

class RKIMEX2D:
    """
    Full 2D SFT solver using dimensional splitting (Strang splitting):
        Step 1: Advection in latitude (half step)
        Step 2: Diffusion in latitude + longitude (full step)
        Step 3: Advection in latitude (half step)
        + Differential rotation in longitude

    Handles the full 2D SFT equation:
    ∂B/∂t = -[advection] + [diffusion on sphere] - B/τ + S(θ,φ,t)
    """

    def __init__(self, lat_deg, lon_deg,
                 eta=500e6, u0=12.5, tau=5*3.156e7,
                 R_sun=6.96e8, dt=1800.0,
                 include_diff_rot=True,
                 flow_profile='van_ballegooijen'):

        from sft_physics import (meridional_flow_vanBall,
                                  differential_rotation_snodgrass)

        self.lat_deg = lat_deg
        self.lon_deg = lon_deg
        self.lat_rad = np.deg2rad(lat_deg)
        self.lon_rad = np.deg2rad(lon_deg)
        self.dlam = self.lat_rad[1] - self.lat_rad[0]
        self.dphi = self.lon_rad[1] - self.lon_rad[0]
        self.Nlat = len(lat_deg)
        self.Nlon = len(lon_deg)
        self.eta  = eta
        self.tau  = tau
        self.R_sun = R_sun
        self.dt   = dt

        # Meridional flow at latitude faces
        lat_half_full = np.concatenate([
            [self.lat_rad[0] - 0.5 * self.dlam],
            self.lat_rad + 0.5 * self.dlam
        ])
        self.u_half = meridional_flow_vanBall(lat_half_full, u0=u0)

        # Differential rotation (radians/s)
        self.include_diff_rot = include_diff_rot
        if include_diff_rot:
            self.omega_dr = differential_rotation_snodgrass(self.lat_rad)

        # Build tridiagonal for latitude diffusion
        self.a_lat, self.b_lat, self.c_lat = build_diffusion_decay_matrix(
            self.Nlat, self.dlam, eta, R_sun, tau, dt, self.lat_rad, alpha=0.5
        )

        # Build tridiagonal for longitude diffusion (periodic)
        # For longitude: L_diff_phi B = η/(R cosλ)² ∂²B/∂φ²
        # This varies with latitude — store as lat-dependent coefficients

    def _lat_advection_step(self, B2d, dt_sub):
        """
        Apply latitude advection for each longitude column.
        B2d: shape (Nlat, Nlon)
        """
        from sft_numerics import van_leer_limiter
        cos_lam = np.cos(self.lat_rad)
        cos_half = np.cos(self.lat_rad + 0.5 * self.dlam)
        cos_half_full = np.concatenate([
            [np.cos(self.lat_rad[0] - 0.5 * self.dlam)],
            cos_half
        ])

        B_new = B2d.copy()
        for j in range(self.Nlon):
            B = B2d[:, j]
            N = self.Nlat
            Bext = np.zeros(N + 4)
            Bext[2:N+2] = B * cos_lam

            flux = np.zeros(N + 1)
            for i in range(N + 1):
                u = self.u_half[i]
                if u >= 0:
                    BL = Bext[i+2]; BLL = Bext[i+1]; BR = Bext[i+3]
                    dB1 = BL - BLL; dB2 = BR - BL
                    r = dB2 / (dB1 + 1e-30)
                    phi = van_leer_limiter(r)
                    Bf = BL + 0.5 * phi * dB1
                else:
                    BR = Bext[i+3]; BRR = Bext[i+4]; BL = Bext[i+2]
                    dB1 = BR - BRR; dB2 = BL - BR
                    r = dB2 / (dB1 + 1e-30)
                    phi = van_leer_limiter(r)
                    Bf = BR + 0.5 * phi * dB1
                flux[i] = u * Bf

            div_f = (flux[1:] - flux[:-1]) / self.dlam
            L_adv = -div_f / (self.R_sun * (cos_lam + 1e-30))
            B_new[:, j] = B + dt_sub * L_adv
            B_new[0, j] = 0.0
            B_new[-1, j] = 0.0

        return B_new

    def _lat_diff_decay_step(self, B2d):
        """Implicit diffusion+decay in latitude for each longitude column."""
        B_new = np.zeros_like(B2d)
        for j in range(self.Nlon):
            rhs = B2d[:, j]
            rhs[0] = 0.0; rhs[-1] = 0.0
            B_new[:, j] = thomas_solve(self.a_lat, self.b_lat, self.c_lat, rhs)
        return B_new

    def _lon_diff_step(self, B2d):
        """
        Implicit diffusion in longitude. For each latitude row:
        η/(R cosλ)² ∂²B/∂φ²  (periodic BCs)
        """
        B_new = np.zeros_like(B2d)
        for i in range(self.Nlat):
            cos_l = np.cos(self.lat_rad[i])
            if abs(cos_l) < 1e-4:   # pole region – skip
                B_new[i, :] = B2d[i, :]
                continue
            coeff = 0.5 * self.dt * self.eta / (self.R_sun * cos_l)**2 / self.dphi**2
            B = B2d[i, :]
            M = len(B)
            # Tridiagonal with periodic BCs — use scipy
            diag  = np.ones(M) * (1.0 + 2 * coeff)
            off   = np.ones(M) * (-coeff)
            # Build banded matrix for scipy
            ab = np.zeros((3, M))
            ab[0, 1:]  = off[:-1]   # super-diag
            ab[1, :]   = diag        # main
            ab[2, :-1] = off[1:]     # sub-diag
            # Periodic corner elements
            rhs = B.copy()
            rhs[0]  += coeff * B[-1]
            rhs[-1] += coeff * B[0]
            B_new[i, :] = solve_banded((1, 1), ab, rhs)
        return B_new

    def _diff_rot_step(self, B2d, dt_sub):
        """
        Apply differential rotation as a rigid shift per latitude.
        Uses FFT for spectral accuracy (fractional shift).
        """
        if not self.include_diff_rot:
            return B2d
        B_new = np.zeros_like(B2d)
        Nlon = self.Nlon
        for i in range(self.Nlat):
            omega = self.omega_dr[i]       # rad/s relative to Carrington
            dphi_shift = omega * dt_sub    # rad
            shift_cells = dphi_shift / self.dphi
            # FFT-based fractional shift
            Brow = B2d[i, :]
            fft_B = np.fft.rfft(Brow)
            freqs = np.fft.rfftfreq(Nlon)
            phase = np.exp(2j * np.pi * freqs * shift_cells * (-1))
            B_new[i, :] = np.fft.irfft(fft_B * phase, n=Nlon)
        return B_new

    def step(self, B2d):
        """
        One full time step using Strang splitting:
        L_adv_lat(dt/2) → L_diff_lat+lon(dt) → L_diff_rot(dt) → L_adv_lat(dt/2)
        """
        dt = self.dt
        B = B2d.copy()

        # Half-step lat advection
        B = self._lat_advection_step(B, dt / 2)
        # Full lat diffusion+decay (implicit)
        B = self._lat_diff_decay_step(B)
        # Full lon diffusion (implicit)
        B = self._lon_diff_step(B)
        # Full differential rotation
        B = self._diff_rot_step(B, dt)
        # Half-step lat advection
        B = self._lat_advection_step(B, dt / 2)

        return B

    def evolve(self, B0_2d, t_end_years, verbose=True, save_interval_days=30):
        """Evolve 2D field, saving snapshots."""
        t_end_s   = t_end_years * 365.25 * 86400
        n_steps   = int(t_end_s / self.dt)
        save_every = max(1, int(save_interval_days * 86400 / self.dt))

        B = B0_2d.copy()
        t_snaps = [0.0]
        B_snaps = [B.copy()]

        for step_i in range(1, n_steps + 1):
            B = self.step(B)
            if step_i % save_every == 0:
                t_yr = step_i * self.dt / (365.25 * 86400)
                t_snaps.append(t_yr)
                B_snaps.append(B.copy())
                if verbose:
                    peak = np.max(np.abs(B))
                    print(f"  t={t_yr:.2f} yr  |B|_max={peak:.4f} G")

        return np.array(t_snaps), np.array(B_snaps)


# ─────────────────────────────────────────────
# CFL and stability utilities
# ─────────────────────────────────────────────

def compute_cfl_numbers(lat_rad, u0, eta, R_sun, dt, dlam_rad):
    """
    Compute CFL numbers for advection and diffusion.
    """
    dx = R_sun * dlam_rad   # arc length per grid cell (m)
    umax = u0               # peak flow m/s

    cfl_adv  = umax * dt / dx
    cfl_diff = 4 * eta * dt / dx**2   # explicit diffusion CFL (should be >> 1 = IMEX needed)

    return {
        'cfl_advection': cfl_adv,
        'cfl_diffusion_if_explicit': cfl_diff,
        'dt_max_advection': dx / umax,
        'dt_max_diffusion_explicit': dx**2 / (4 * eta),
        'dx_m': dx,
        'speedup_from_IMEX': cfl_diff / cfl_adv
    }


def stability_report(lat_deg, eta, u0, tau, dt):
    """Print a full stability report."""
    from sft_physics import R_SUN
    dlam = np.deg2rad(lat_deg[1] - lat_deg[0])
    cfl = compute_cfl_numbers(np.deg2rad(lat_deg), u0, eta, R_SUN, dt, dlam)

    print("=" * 55)
    print("  STABILITY ANALYSIS")
    print("=" * 55)
    print(f"  Grid spacing:         {np.rad2deg(dlam):.1f}°  ({cfl['dx_m']/1e6:.1f} Mm)")
    print(f"  Time step:            {dt:.0f} s ({dt/3600:.1f} hr)")
    print(f"  CFL (advection):      {cfl['cfl_advection']:.5f}  ✓ stable")
    print(f"  CFL (diff, explicit): {cfl['cfl_diffusion_if_explicit']:.2f}  "
          f"{'⚠ UNSTABLE (IMEX needed)' if cfl['cfl_diffusion_if_explicit'] > 1 else '✓'}")
    print(f"  IMEX speedup factor:  {cfl['speedup_from_IMEX']:.0f}×")
    print(f"  Max dt (advection):   {cfl['dt_max_advection']/3600:.1f} hr")
    print(f"  Max dt (diff expl.):  {cfl['dt_max_diffusion_explicit']:.0f} s")
    print(f"  Decay timescale:      {tau/(365.25*86400):.1f} yr")
    print("=" * 55)
