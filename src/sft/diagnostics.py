"""
sft_diagnostics.py
==================
Comprehensive diagnostic tools for SFT simulations.

Diagnostics implemented:
1. Flux conservation test (Φ vs Φ_0 exp(-t/τ))
2. Centroid tracking (poleward migration rate)
3. Profile FWHM evolution (diffusion diagnostic)
4. Axial dipole moment evolution (cycle forecasting)
5. Parameter sensitivity analysis
6. Butterfly diagram construction
7. Energy dissipation check
8. Convergence testing
"""

import numpy as np
from sft_physics import (total_flux, signed_flux, axial_dipole_moment,
                          flux_centroid, R_SUN)


# ─────────────────────────────────────────────
# 1. Flux conservation
# ─────────────────────────────────────────────

def flux_conservation_check(t_yr, B_snaps, lat_rad, tau_yr=5.0):
    """
    Check exponential flux decay against analytic prediction.

    Returns
    -------
    result : dict with fluxes, expected, errors
    """
    tau_s = tau_yr * 365.25 * 86400
    fluxes = []
    for B in B_snaps:
        fluxes.append(total_flux(B, lat_rad))

    fluxes = np.array(fluxes)
    Phi0 = fluxes[0]
    expected = Phi0 * np.exp(-t_yr * 365.25 * 86400 / tau_s)

    rel_error = np.abs(fluxes - expected) / (np.abs(Phi0) + 1e-30) * 100.0

    return {
        't_yr': t_yr,
        'flux': fluxes,
        'expected': expected,
        'rel_error_pct': rel_error,
        'max_error_pct': np.max(rel_error),
    }


# ─────────────────────────────────────────────
# 2. Centroid tracking
# ─────────────────────────────────────────────

def track_centroids(t_yr, B_snaps, lat_rad):
    """
    Track centroid latitude of positive and negative polarity over time.

    Returns
    -------
    dict with time arrays and centroid latitude arrays
    """
    lat_deg = np.rad2deg(lat_rad)
    pos_centroids = []
    neg_centroids = []

    for B in B_snaps:
        pos_centroids.append(flux_centroid(B, lat_rad, 'positive'))
        neg_centroids.append(flux_centroid(B, lat_rad, 'negative'))

    pos_c = np.array(pos_centroids)
    neg_c = np.array(neg_centroids)

    # Migration rates (deg/yr) using finite differences
    if len(t_yr) > 1:
        dt_yr = np.diff(t_yr)
        pos_rate = np.diff(pos_c) / (dt_yr + 1e-30)
        neg_rate = np.diff(neg_c) / (dt_yr + 1e-30)
    else:
        pos_rate = neg_rate = np.array([np.nan])

    return {
        't_yr': t_yr,
        'pos_centroid_deg': pos_c,
        'neg_centroid_deg': neg_c,
        'pos_migration_rate_deg_yr': pos_rate,
        'neg_migration_rate_deg_yr': neg_rate,
        'avg_pos_rate': np.nanmean(np.abs(pos_rate)),
        'avg_neg_rate': np.nanmean(np.abs(neg_rate)),
    }


# ─────────────────────────────────────────────
# 3. Profile FWHM evolution
# ─────────────────────────────────────────────

def profile_fwhm(B, lat_rad, polarity='positive'):
    """
    Compute FWHM of the profile for a given polarity.
    Returns FWHM in degrees.
    """
    if polarity == 'positive':
        B_abs = np.maximum(B, 0)
    else:
        B_abs = np.maximum(-B, 0)

    if B_abs.max() < 1e-10:
        return np.nan

    half_max = B_abs.max() / 2.0
    above = B_abs >= half_max
    indices = np.where(above)[0]
    if len(indices) < 2:
        return np.nan

    fwhm_rad = lat_rad[indices[-1]] - lat_rad[indices[0]]
    return np.abs(np.rad2deg(fwhm_rad))


def track_profile_evolution(t_yr, B_snaps, lat_rad):
    """
    Track profile peak amplitude and FWHM over time.

    Returns
    -------
    dict with time, peak amplitudes, FWHMs for both polarities
    """
    peak_pos = []; peak_neg = []
    fwhm_pos = []; fwhm_neg = []

    for B in B_snaps:
        peak_pos.append(np.max(B))
        peak_neg.append(np.min(B))
        fwhm_pos.append(profile_fwhm(B, lat_rad, 'positive'))
        fwhm_neg.append(profile_fwhm(B, lat_rad, 'negative'))

    return {
        't_yr': t_yr,
        'peak_pos': np.array(peak_pos),
        'peak_neg': np.array(peak_neg),
        'fwhm_pos_deg': np.array(fwhm_pos),
        'fwhm_neg_deg': np.array(fwhm_neg),
    }


def diffusion_broadening_theory(t_yr, sigma0_deg, eta, R_sun):
    """
    Theoretical profile broadening due to diffusion:
    σ(t)² = σ₀² + 2ηt/R²
    """
    t_s = t_yr * 365.25 * 86400
    sigma0_rad = np.deg2rad(sigma0_deg)
    sigma_rad  = np.sqrt(sigma0_rad**2 + 2 * eta * t_s / R_sun**2)
    return np.rad2deg(sigma_rad)


# ─────────────────────────────────────────────
# 4. Axial dipole moment
# ─────────────────────────────────────────────

def compute_dipole_evolution(t_yr, B_snaps, lat_rad):
    """
    Compute axial dipole moment D(t) = (3/4π) ∫ B cosλ · cosλ dλ · 2π
    Key predictor of next solar cycle amplitude.
    """
    dipoles = []
    for B in B_snaps:
        dipoles.append(axial_dipole_moment(B, lat_rad))
    return {
        't_yr': t_yr,
        'dipole_G': np.array(dipoles),
    }


# ─────────────────────────────────────────────
# 5. Parameter sensitivity analysis
# ─────────────────────────────────────────────

def sensitivity_study(B0, lat_deg, t_eval_yr=2.0,
                       eta_values=None, u0_values=None, tau_values=None):
    """
    Run parameter sensitivity sweeps.

    Returns
    -------
    dict with results for each parameter variation
    """
    from sft_numerics import RKIMEX2Stage
    from sft_physics import R_SUN

    lat_rad = np.deg2rad(lat_deg)
    ref_eta  = 500e6
    ref_u0   = 12.5
    ref_tau  = 5 * 365.25 * 86400

    results = {}

    # Vary eta
    if eta_values is None:
        eta_values = [250e6, 500e6, 1000e6]

    results['eta_sweep'] = []
    for eta in eta_values:
        solver = RKIMEX2Stage(lat_deg, eta=eta, u0=ref_u0, tau=ref_tau,
                               R_sun=R_SUN, dt=1800.0)
        t_snaps, B_snaps = solver.evolve(B0, t_eval_yr, verbose=False,
                                          save_interval_days=365.25*t_eval_yr)
        idx = np.argmin(np.abs(t_snaps - t_eval_yr))
        B_final = B_snaps[idx]
        results['eta_sweep'].append({
            'eta': eta, 'eta_label': f'η={eta/1e6:.0f} km²/s',
            'B': B_final,
            'peak': np.max(B_final),
            'fwhm': profile_fwhm(B_final, lat_rad, 'positive'),
            'centroid_pos': flux_centroid(B_final, lat_rad, 'positive'),
        })

    # Vary u0
    if u0_values is None:
        u0_values = [6.25, 12.5, 25.0]

    results['u0_sweep'] = []
    for u0 in u0_values:
        solver = RKIMEX2Stage(lat_deg, eta=ref_eta, u0=u0, tau=ref_tau,
                               R_sun=R_SUN, dt=1800.0)
        t_snaps, B_snaps = solver.evolve(B0, t_eval_yr, verbose=False,
                                          save_interval_days=365.25*t_eval_yr)
        idx = np.argmin(np.abs(t_snaps - t_eval_yr))
        B_final = B_snaps[idx]
        results['u0_sweep'].append({
            'u0': u0, 'u0_label': f'u₀={u0} m/s',
            'B': B_final,
            'peak': np.max(B_final),
            'fwhm': profile_fwhm(B_final, lat_rad, 'positive'),
            'centroid_pos': flux_centroid(B_final, lat_rad, 'positive'),
        })

    # Vary tau
    if tau_values is None:
        tau_values = [2.5, 5.0, 10.0]  # years

    results['tau_sweep'] = []
    for tau_yr in tau_values:
        tau_s = tau_yr * 365.25 * 86400
        solver = RKIMEX2Stage(lat_deg, eta=ref_eta, u0=ref_u0, tau=tau_s,
                               R_sun=R_SUN, dt=1800.0)
        t_snaps, B_snaps = solver.evolve(B0, t_eval_yr, verbose=False,
                                          save_interval_days=365.25*t_eval_yr)
        idx = np.argmin(np.abs(t_snaps - t_eval_yr))
        B_final = B_snaps[idx]
        results['tau_sweep'].append({
            'tau_yr': tau_yr, 'tau_label': f'τ={tau_yr} yr',
            'B': B_final,
            'peak': np.max(B_final),
            'fwhm': profile_fwhm(B_final, lat_rad, 'positive'),
        })

    return results


# ─────────────────────────────────────────────
# 6. Butterfly diagram
# ─────────────────────────────────────────────

def build_butterfly_diagram(t_yr, B_snaps):
    """
    Build butterfly diagram array (t, lat).

    Returns
    -------
    t_yr   : time array (years)
    B_2d   : 2D array shape (len(t_yr), Nlat)
    """
    return t_yr, np.array(B_snaps)


# ─────────────────────────────────────────────
# 7. Convergence test
# ─────────────────────────────────────────────

def convergence_test(B0_fine, lat_deg_fine, resolutions=(1.0, 0.5, 0.25),
                     t_end_yr=1.0, eta=500e6, u0=12.5, tau=5*3.156e7):
    """
    Grid convergence test: run same simulation at multiple resolutions,
    compute L1 norm differences vs finest grid.
    """
    from sft_numerics import RKIMEX2Stage
    from scipy.interpolate import interp1d

    results = {}
    finest_B = None
    finest_lat = None

    for dres in sorted(resolutions):
        lat_deg = np.arange(-70, 70 + dres, dres)
        lat_rad = np.deg2rad(lat_deg)

        # Interpolate B0 to this resolution
        lat0 = np.deg2rad(lat_deg_fine)
        interp = interp1d(lat0, B0_fine, bounds_error=False, fill_value=0.0)
        B0 = interp(lat_rad)

        solver = RKIMEX2Stage(lat_deg, eta=eta, u0=u0, tau=tau, dt=1800.0)
        t_snaps, B_snaps = solver.evolve(B0, t_end_yr, verbose=False,
                                          save_interval_days=365.25*t_end_yr)
        B_final = B_snaps[-1]

        if finest_B is None and dres == min(resolutions):
            finest_B = B_final
            finest_lat = lat_rad

        results[dres] = {'lat_deg': lat_deg, 'B': B_final}

    # Compute L1 errors vs finest
    fine_interp = interp1d(np.deg2rad(results[min(resolutions)]['lat_deg']),
                            results[min(resolutions)]['B'],
                            bounds_error=False, fill_value=0.0)
    for dres in resolutions:
        B = results[dres]['B']
        lat_r = np.deg2rad(results[dres]['lat_deg'])
        B_ref = fine_interp(lat_r)
        L1 = np.mean(np.abs(B - B_ref))
        results[dres]['L1_vs_finest'] = L1

    return results


# ─────────────────────────────────────────────
# 8. Summary report
# ─────────────────────────────────────────────

def print_diagnostic_summary(t_yr, B_snaps, lat_rad, tau_yr=5.0,
                               label="Simulation"):
    """Print a compact diagnostic summary table."""
    flux_check  = flux_conservation_check(t_yr, B_snaps, lat_rad, tau_yr)
    centroid_tr = track_centroids(t_yr, B_snaps, lat_rad)
    profile_ev  = track_profile_evolution(t_yr, B_snaps, lat_rad)
    dipole_ev   = compute_dipole_evolution(t_yr, B_snaps, lat_rad)

    print(f"\n{'=' * 60}")
    print(f"  DIAGNOSTICS: {label}")
    print(f"{'=' * 60}")
    print(f"  Simulation span:   {t_yr[0]:.1f} → {t_yr[-1]:.1f} yr")
    print(f"  Flux error (max):  {flux_check['max_error_pct']:.3f} %")
    print(f"  Peak amplitude:    {profile_ev['peak_pos'][0]:.3f} G (t=0) → "
          f"{profile_ev['peak_pos'][-1]:.4f} G (final)")
    print(f"  FWHM (pos):        {profile_ev['fwhm_pos_deg'][0]:.1f}° (t=0) → "
          f"{profile_ev['fwhm_pos_deg'][-1]:.1f}° (final)")
    print(f"  Centroid (pos):    {centroid_tr['pos_centroid_deg'][0]:.1f}° → "
          f"{centroid_tr['pos_centroid_deg'][-1]:.1f}°")
    print(f"  Centroid (neg):    {centroid_tr['neg_centroid_deg'][0]:.1f}° → "
          f"{centroid_tr['neg_centroid_deg'][-1]:.1f}°")
    print(f"  Avg migration rate (pos): {centroid_tr['avg_pos_rate']:.1f} °/yr")
    print(f"  Axial dipole:      {dipole_ev['dipole_G'][0]:.4f} G → "
          f"{dipole_ev['dipole_G'][-1]:.4f} G")
    print(f"{'=' * 60}\n")

    return {
        'flux': flux_check,
        'centroid': centroid_tr,
        'profile': profile_ev,
        'dipole': dipole_ev,
    }
