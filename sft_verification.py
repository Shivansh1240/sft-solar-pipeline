import numpy as np
import time

from sft_physics import (
    R_SUN, TAU_DECAY, ETA_DEFAULT, U0_DEFAULT,
    bmr_longitude_averaged
)
from sft_numerics import RKIMEX2Stage
from sft_diagnostics import (
    flux_conservation_check, profile_fwhm, diffusion_broadening_theory, convergence_test
)

def _l2_norm(x):
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x * x)))

def _relative_l2_error(a, b, eps=1e-30):
    a = np.asarray(a); b = np.asarray(b)
    return float(_l2_norm(a - b) / (_l2_norm(b) + eps))

def _shift_profile_linear(lat_deg, B, shift_deg):
    """
    Shift a 1D profile by shift_deg using linear interpolation.
    Values outside the domain are filled with 0 (consistent with Dirichlet BCs).
    """
    lat_deg = np.asarray(lat_deg)
    B = np.asarray(B)
    x = lat_deg
    xp = x - shift_deg
    return np.interp(x, xp, B, left=0.0, right=0.0)

def diffusion_test(dlat=1.0, lat_range=(-89, 89)):
    """
    1. Pure Diffusion Test
    - Turn off advection (u0=0) + decay (tau=inf)
    - Compare with analytic Gaussian spreading FWHM.
    """
    print("[Verification] Running Pure Diffusion Test...")
    lat_deg = np.arange(lat_range[0], lat_range[1] + dlat, dlat)
    lat_rad = np.deg2rad(lat_deg)
    
    sigma0_deg = 3.51
    eta = 500e6
    t_end_yr = 1.0
    
    # Initialize Gaussian BMR near equator to minimize spherical distortion initially
    B0 = bmr_longitude_averaged(lat_deg, lat0_deg=0.0, sigma_lat_deg=sigma0_deg, B0=10.0, delta_lat_sep_deg=10.0)
    
    solver = RKIMEX2Stage(
        lat_deg, eta=eta, u0=0.0, tau=np.inf, 
        flow_profile='constant', dt=1800.0
    )
    
    t_yr, B_snaps = solver.evolve(B0, t_end_yr, verbose=False, save_interval_days=30)
    
    # Compute FWHM of the positive lobe
    fwhm_sim = []
    fwhm_theory = []
    
    # Base FWHM theory relation to sigma (FWHM = 2*sqrt(2*ln(2)) * sigma)
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    
    for i, t in enumerate(t_yr):
        # We track FWHM over time
        f_sim = profile_fwhm(B_snaps[i], lat_rad, 'positive')
        # Using the theoretical broadening relation: sigma(t)^2 = sigma(0)^2 + 2*eta*t/R^2
        sigma_t = diffusion_broadening_theory(t, sigma0_deg, eta, R_SUN)
        f_theo = fwhm_factor * sigma_t
        fwhm_sim.append(f_sim)
        fwhm_theory.append(f_theo)
        
    fwhm_sim = np.array(fwhm_sim)
    fwhm_theory = np.array(fwhm_theory)
    
    # Rel error on the final FWHM
    error_percent = np.abs(fwhm_sim[-1] - fwhm_theory[-1]) / fwhm_theory[-1] * 100.0
    
    # PASS condition: < 5% error.
    passed = bool(error_percent < 5.0)
    
    return {
        "test_name": "Pure Diffusion",
        "expected_fwhm": fwhm_theory.tolist(),
        "simulated_fwhm": fwhm_sim.tolist(),
        "error_percent": float(error_percent),
        "passed": passed,
        "details": f"Diffusion FWHM final theory: {fwhm_theory[-1]:.2f}° vs sim: {fwhm_sim[-1]:.2f}°",
        "params": {"eta_m2_s": float(eta), "u0_m_s": 0.0, "tau_s": float(np.inf), "dt_s": 1800.0, "dlat_deg": float(dlat)},
    }

def advection_test(dlat=0.5, lat_range=(-89, 89)):
    """
    2. Pure Advection Test
    - Turn off diffusion (eta=0) and decay (tau=inf)
    - Initial profile should translate without distortion.
    """
    print("[Verification] Running Pure Advection Test...")
    lat_deg = np.arange(lat_range[0], lat_range[1] + dlat, dlat)
    
    u0 = 10.0 # m/s (constant)
    t_end_yr = 1.0
    
    B0 = bmr_longitude_averaged(lat_deg, lat0_deg=10.0, sigma_lat_deg=3.51, B0=10.0, delta_lat_sep_deg=10.0)
    
    solver = RKIMEX2Stage(
        lat_deg, eta=0.0, u0=u0, tau=np.inf, 
        flow_profile='constant', dt=1800.0
    )
    
    t_yr, B_snaps = solver.evolve(B0, t_end_yr, verbose=False, save_interval_days=30)
    
    B_final = B_snaps[-1]
    
    # Analytic advection distance (ignoring spherical geometry area scaling for a moment, just check peak movement)
    # The actual SFT eq: -1/(R cos lat) d/dlat (u cos lat B). With u=constant:
    # d/dt (B cos lat) = - u/R d/dlat (B cos lat)
    # So B cos lat simply translates at angular velocity u/R.
    dlat_shift_rad = (u0 / R_SUN) * (t_end_yr * 365.25 * 86400)
    dlat_shift_deg = np.rad2deg(dlat_shift_rad)
    
    peak_lat_idx_0 = np.argmax(B0)
    peak_lat_idx_f = np.argmax(B_final)
    
    lat_shift_sim = lat_deg[peak_lat_idx_f] - lat_deg[peak_lat_idx_0]
    
    error_deg = np.abs(lat_shift_sim - dlat_shift_deg)
    error_percent = error_deg / dlat_shift_deg * 100.0 if dlat_shift_deg > 0 else 100.0

    # Shape preservation: compare (B cosλ) against shifted initial (B cosλ)
    cos_lam = np.cos(np.deg2rad(lat_deg))
    Bc0 = B0 * cos_lam
    Bcf = B_final * cos_lam
    Bc0_shift = _shift_profile_linear(lat_deg, Bc0, dlat_shift_deg)
    rel_l2_shape = _relative_l2_error(Bcf, Bc0_shift)

    passed = bool((error_percent < 10.0) and (rel_l2_shape < 0.15))
    
    return {
        "test_name": "Pure Advection",
        "expected_shift_deg": float(dlat_shift_deg),
        "simulated_shift_deg": float(lat_shift_sim),
        "error_percent": float(error_percent),
        "shape_rel_l2_error": float(rel_l2_shape),
        "passed": passed,
        "details": f"Shift expected: {dlat_shift_deg:.2f}° vs sim: {lat_shift_sim:.2f}°; shape rel-L2(B cosλ)={rel_l2_shape:.3f}",
        "params": {"eta_m2_s": 0.0, "u0_m_s": float(u0), "tau_s": float(np.inf), "dt_s": 1800.0, "dlat_deg": float(dlat)},
    }

def decay_test(dlat=1.0, lat_range=(-89, 89)):
    """
    3. Decay-only Test
    - Turn off advection (u0=0) and diffusion (eta=0)
    - Verify B(t) = B0 exp(-t/tau)
    """
    print("[Verification] Running Decay-only Test...")
    lat_deg = np.arange(lat_range[0], lat_range[1] + dlat, dlat)
    
    tau_yr = 5.0
    tau_s = tau_yr * 365.25 * 86400
    t_end_yr = 5.0
    
    B0 = bmr_longitude_averaged(lat_deg, lat0_deg=12.0)
    
    solver = RKIMEX2Stage(
        lat_deg, eta=0.0, u0=0.0, tau=tau_s, 
        flow_profile='constant', dt=1800.0
    )
    
    t_yr, B_snaps = solver.evolve(B0, t_end_yr, verbose=False, save_interval_days=30)
    
    B_final = B_snaps[-1]
    B_final_theory = B0 * np.exp(-t_end_yr / tau_yr)
    
    max_error = np.max(np.abs(B_final - B_final_theory))
    max_B = np.max(np.abs(B0))
    error_percent = (max_error / max_B) * 100.0 if max_B > 0 else 0.0

    # With Crank–Nicolson-style implicit decay, expect small but not machine-zero error.
    passed = bool(error_percent < 0.1)
    
    return {
        "test_name": "Pure Decay",
        "max_error_G": float(max_error),
        "error_percent": float(error_percent),
        "passed": passed,
        "details": f"Max deviation from exact exponential decay: {max_error:.2e} G",
        "params": {"eta_m2_s": 0.0, "u0_m_s": 0.0, "tau_s": float(tau_s), "dt_s": 1800.0, "dlat_deg": float(dlat)},
    }

def symmetry_test(dlat=1.0, lat_range=(-89, 89)):
    """
    4. Symmetry Test
    - Symmetric BMR -> must remain symmetric
    - Detect grid bias / numerical asymmetry
    """
    print("[Verification] Running Symmetry Test...")
    # Need exactly symmetric grid around 0
    lat_deg = np.arange(-90, 90 + dlat, dlat) 
    
    # Initial perfectly antisymmetric field
    B0 = np.zeros_like(lat_deg)
    # B(-lat) = -B(lat)
    for i, lat in enumerate(lat_deg):
        if lat > 0:
            B0[i] = np.exp(-0.5 * ((lat - 20) / 5)**2)
        elif lat < 0:
            B0[i] = -np.exp(-0.5 * ((lat + 20) / 5)**2)
            
    # u0=0 so no asymmetric flow.
    solver = RKIMEX2Stage(
        lat_deg, eta=500e6, u0=0.0, tau=np.inf, 
        flow_profile='constant', dt=1800.0
    )
    
    t_yr, B_snaps = solver.evolve(B0, 2.0, verbose=False, save_interval_days=60)
    B_final = B_snaps[-1]
    
    # Check antisymmetry: B(lat) + B(-lat) should be 0
    asymmetry_error = np.abs(B_final + B_final[::-1])
    max_asymmetry = np.max(asymmetry_error)
    
    # Normalizing by max amplitude
    max_B = np.max(np.abs(B_final))
    error_percent = (max_asymmetry / max_B) * 100.0 if max_B > 0 else 0.0

    # Allow tiny numerical asymmetry; failures here should be "obvious", not floating-point noise.
    passed = bool(error_percent < 1e-6)
    
    return {
        "test_name": "Symmetry Preservation",
        "max_asymmetry_error": float(max_asymmetry),
        "error_percent": float(error_percent),
        "passed": passed,
        "details": f"Max asymmetry magnitude: {max_asymmetry:.2e}",
        "params": {"eta_m2_s": 500e6, "u0_m_s": 0.0, "tau_s": float(np.inf), "dt_s": 1800.0, "dlat_deg": float(dlat)},
    }

def run_convergence_test():
    """
    5. Convergence Test
    - dlat = 2, 1, 0.5
    - Verify error scaling
    """
    print("[Verification] Running Convergence Test...")
    lat_deg_fine = np.arange(-89, 89 + 0.25, 0.25)
    B0_fine = bmr_longitude_averaged(lat_deg_fine, lat0_deg=20.0)
    
    # We use RKIMEX with advection + diffusion
    res = convergence_test(B0_fine, lat_deg_fine, resolutions=(2.0, 1.0, 0.5), t_end_yr=1.0)
    
    l1_2 = res[2.0]['L1_vs_finest']
    l1_1 = res[1.0]['L1_vs_finest']
    l1_05 = res[0.5]['L1_vs_finest']
    
    # Order of convergence = log2(error(2dx)/error(dx))
    order_1 = np.log2(l1_2 / l1_1) if l1_1 > 0 else 0.0
    order_2 = np.log2(l1_1 / l1_05) if l1_05 > 0 else 0.0
    
    # We expect close to 2nd order (Godunov van leer + implicit center difference)
    passed = bool(order_1 > 1.0 and order_2 > 1.0) 
    
    return {
        "test_name": "Grid Convergence",
        "l1_errors": {
            "dlat=2.0": float(l1_2),
            "dlat=1.0": float(l1_1),
            "dlat=0.5": float(l1_05)
        },
        "convergence_rates": [float(order_1), float(order_2)],
        "passed": passed,
        "details": f"Rates: {order_1:.2f}, {order_2:.2f} (Target > 1.0)"
    }

def run_all_verification_tests():
    t0 = time.time()
    results = [
        diffusion_test(),
        advection_test(),
        decay_test(),
        symmetry_test(),
        run_convergence_test()
    ]
    
    total_passed = sum([1 for r in results if r['passed']])
    
    report = {
        "meta": {
            "solver": "RKIMEX2Stage",
            "notes": [
                "All quantities are SI internally (eta in m^2/s, tau in s).",
                "Advection test checks translation of (B cosλ) for constant u, consistent with the discretized geometry term.",
            ],
        },
        "summary": {
            "total_tests": len(results),
            "passed": total_passed,
            "failed": len(results) - total_passed,
            "duration_s": round(time.time() - t0, 2)
        },
        "tests": results
    }
    
    return report

if __name__ == "__main__":
    import json
    report = run_all_verification_tests()
    print(json.dumps(report, indent=2))
