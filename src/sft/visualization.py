"""
sft_visualization.py
====================
Visualization tools for SFT simulation results.

Produces:
- Figure 5 reproduction (6-panel: initial maps, profiles, butterfly diagrams)
- Diagnostic plots (flux conservation, centroid tracking, diffusion evolution)
- Parameter sensitivity plots
- Animated butterfly diagrams
- 2D field maps with colorbars
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for pipeline use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

# ─────────────────────────────────────────────
# Style defaults
# ─────────────────────────────────────────────

STYLE = {
    'pos_color': '#1f77b4',   # blue – positive polarity
    'neg_color': '#d62728',   # red  – negative polarity
    'cmap_field': 'RdBu_r',
    'cmap_butterfly': 'RdBu_r',
    'fig_dpi': 150,
}

def _setup_style():
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': STYLE['fig_dpi'],
    })

_setup_style()


# ─────────────────────────────────────────────
# 1. Initial 2D field map
# ─────────────────────────────────────────────

def plot_initial_bmr(B_2d, lat_deg, lon_deg, title="Initial BMR",
                      ax=None, vmax=None, lon_range=(50, 150)):
    """
    Plot 2D bipolar magnetic region map at t=0.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    vmax = vmax or np.percentile(np.abs(B_2d), 99)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Mask to lon_range for visibility
    i_lo = np.searchsorted(lon_deg, lon_range[0])
    i_hi = np.searchsorted(lon_deg, lon_range[1])
    B_show = B_2d[:, i_lo:i_hi]
    lon_show = lon_deg[i_lo:i_hi]

    img = ax.pcolormesh(lon_show, lat_deg, B_show,
                         cmap=STYLE['cmap_field'], norm=norm, shading='auto')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel("Longitude (in degrees)")
    ax.set_ylabel("Latitude (in degrees)")
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax, label="Magnetic field (Gauss)")

    return ax


# ─────────────────────────────────────────────
# 2. Longitude-averaged profile
# ─────────────────────────────────────────────

def plot_lon_avg_profile(B_1d, lat_deg, title="Longitude-Averaged Profile",
                          ax=None, times_yr=None, B_snapshots=None,
                          color=STYLE['pos_color']):
    """
    Plot ⟨Br⟩(λ) at t=0 (and optionally multiple times).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 4))
    else:
        fig = ax.figure

    if B_snapshots is not None and times_yr is not None:
        cmap = plt.cm.plasma
        for i, (t, B) in enumerate(zip(times_yr, B_snapshots)):
            c = cmap(i / max(len(times_yr) - 1, 1))
            ax.plot(B, lat_deg, color=c, lw=1.2, label=f"t={t:.1f} yr")
        ax.legend(fontsize=7)
    else:
        ax.plot(B_1d, lat_deg, color=color, lw=1.5)

    ax.axvline(0, color='r', lw=0.8, ls='--')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("⟨B r⟩ (Gauss)")
    ax.set_ylabel("Latitude (in degrees)")
    ax.set_title(title)
    ax.set_ylim(lat_deg[0], lat_deg[-1])

    return ax


# ─────────────────────────────────────────────
# 3. Butterfly diagram
# ─────────────────────────────────────────────

def plot_butterfly(t_yr, B_snaps, lat_deg, title="Butterfly Diagram",
                   ax=None, vmax=None):
    """
    Plot time-latitude butterfly diagram of ⟨Br⟩(λ, t).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    B_2d = np.array(B_snaps)   # shape: (Nt, Nlat)
    vmax = vmax or np.percentile(np.abs(B_2d), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    img = ax.pcolormesh(t_yr, lat_deg, B_2d.T,
                         cmap=STYLE['cmap_butterfly'], norm=norm,
                         shading='auto')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("time (years)")
    ax.set_ylabel("Latitude (in degrees)")
    ax.set_title(title)
    ax.set_xlim(t_yr[0], t_yr[-1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax, label="Magnetic field (Gauss)")

    return ax


# ─────────────────────────────────────────────
# 4. Figure 5 reproduction (6-panel)
# ─────────────────────────────────────────────

def plot_figure5_reproduction(
        # Case 1
        B_2d_c1, lon_avg_c1, t_yr_c1, B_snaps_c1,
        # Case 2
        B_2d_c2, lon_avg_c2, t_yr_c2, B_snaps_c2,
        lat_deg, lon_deg,
        save_path="figure5_reproduction.png"):
    """
    Reproduce Figure 5 from Athalathil et al. 2024.
    6-panel layout: rows = cases, columns = initial map / lon-avg / butterfly.
    """
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.45)

    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    cases = [
        (B_2d_c1, lon_avg_c1, t_yr_c1, B_snaps_c1, "a", "b", "c",
         r"$\lambda_0 = 0°$"),
        (B_2d_c2, lon_avg_c2, t_yr_c2, B_snaps_c2, "d", "e", "f",
         r"$\lambda_0 = 12°$"),
    ]

    for row, (B2d, Bla, t_yr, Bsnaps, la, lb, lc, lbl) in enumerate(cases):
        # Panel a/d: initial 2D map
        plot_initial_bmr(B2d, lat_deg, lon_deg,
                          title=f"({la}) initial BMR", ax=axes[row][0])

        # Panel b/e: longitude-averaged profile
        plot_lon_avg_profile(Bla, lat_deg,
                              title=f"({lb}) {lbl}", ax=axes[row][1])

        # Panel c/f: butterfly diagram
        plot_butterfly(t_yr, Bsnaps, lat_deg,
                        title=f"({lc}) butterfly", ax=axes[row][2])

    fig.suptitle("Figure 5 Reproduction — Athalathil et al. (2024)\n"
                 "SFT Pipeline: RK-IMEX + van Leer limiter", y=1.01, fontsize=12)

    plt.savefig(save_path, dpi=STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


# ─────────────────────────────────────────────
# 5. Diagnostic plots
# ─────────────────────────────────────────────

def plot_diagnostics(diag_dict, lat_deg, save_path="diagnostics.png"):
    """
    Plot all diagnostics in a single multi-panel figure.
    diag_dict: output from sft_diagnostics.print_diagnostic_summary
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("SFT Diagnostic Analysis", fontsize=13)

    lat_rad = np.deg2rad(lat_deg)

    # 1. Flux conservation
    ax = axes[0, 0]
    fd = diag_dict['flux']
    ax.plot(fd['t_yr'], np.abs(fd['flux']), 'b-o', ms=3, label='Simulated |Φ|')
    ax.plot(fd['t_yr'], np.abs(fd['expected']), 'r--', lw=1.5, label='Expected e^{-t/τ}')
    ax.set_xlabel("Time (years)"); ax.set_ylabel("|Φ|")
    ax.set_title("Flux Conservation"); ax.legend(); ax.set_yscale('log')

    # 2. Flux error
    ax = axes[0, 1]
    ax.plot(fd['t_yr'], fd['rel_error_pct'], 'g-o', ms=3)
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Relative error (%)")
    ax.set_title(f"Flux Error (max={fd['max_error_pct']:.3f}%)")
    ax.axhline(0.1, color='r', ls='--', label='0.1% threshold')
    ax.legend()

    # 3. Centroid migration
    ax = axes[0, 2]
    cd = diag_dict['centroid']
    ax.plot(cd['t_yr'], cd['pos_centroid_deg'], STYLE['pos_color'], lw=2, label='Positive polarity')
    ax.plot(cd['t_yr'], cd['neg_centroid_deg'], STYLE['neg_color'], lw=2, label='Negative polarity')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Centroid latitude (°)")
    ax.set_title("Poleward Migration"); ax.legend()

    # 4. Profile evolution (peak amplitude)
    ax = axes[1, 0]
    pd = diag_dict['profile']
    ax.semilogy(pd['t_yr'], pd['peak_pos'], STYLE['pos_color'], lw=2, label='Peak positive')
    ax.semilogy(pd['t_yr'], np.abs(pd['peak_neg']), STYLE['neg_color'], lw=2, label='Peak |negative|')
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Peak |⟨Br⟩| (G)")
    ax.set_title("Amplitude Decay"); ax.legend()

    # 5. FWHM evolution
    ax = axes[1, 1]
    ax.plot(pd['t_yr'], pd['fwhm_pos_deg'], STYLE['pos_color'], lw=2, label='Positive')
    ax.plot(pd['t_yr'], pd['fwhm_neg_deg'], STYLE['neg_color'], lw=2, label='Negative')
    # Theory curve
    from sft_diagnostics import diffusion_broadening_theory
    from sft_physics import ETA_DEFAULT, R_SUN
    sigma0 = 3.51 * 2.35  # sigma to FWHM (approx)
    t_theory = pd['t_yr']
    fwhm_theory = np.array([diffusion_broadening_theory(t, sigma0, ETA_DEFAULT, R_SUN) * 2.35
                              for t in t_theory])
    ax.plot(t_theory, fwhm_theory, 'k--', lw=1, label='Diffusion theory')
    ax.set_xlabel("Time (years)"); ax.set_ylabel("FWHM (°)")
    ax.set_title("Profile Broadening (Diffusion)"); ax.legend()

    # 6. Axial dipole moment
    ax = axes[1, 2]
    dd = diag_dict['dipole']
    ax.plot(dd['t_yr'], dd['dipole_G'], 'purple', lw=2)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Axial Dipole Moment (G)")
    ax.set_title("Dipole Evolution")

    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


# ─────────────────────────────────────────────
# 6. Parameter sensitivity plot
# ─────────────────────────────────────────────

def plot_sensitivity(sens_results, lat_deg, t_eval_yr=2.0,
                      save_path="sensitivity.png"):
    """Plot parameter sensitivity results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Parameter Sensitivity Study at t = {t_eval_yr} yr", fontsize=12)

    # Eta sweep
    ax = axes[0]
    for r in sens_results['eta_sweep']:
        ax.plot(r['B'], lat_deg, lw=2, label=r['eta_label'])
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("⟨Br⟩ (Gauss)"); ax.set_ylabel("Latitude (°)")
    ax.set_title("Diffusivity Sensitivity"); ax.legend()

    # u0 sweep
    ax = axes[1]
    for r in sens_results['u0_sweep']:
        ax.plot(r['B'], lat_deg, lw=2, label=r['u0_label'])
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("⟨Br⟩ (Gauss)"); ax.set_ylabel("Latitude (°)")
    ax.set_title("Meridional Flow Sensitivity"); ax.legend()

    # tau sweep
    ax = axes[2]
    for r in sens_results['tau_sweep']:
        ax.plot(r['B'], lat_deg, lw=2, label=r['tau_label'])
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("⟨Br⟩ (Gauss)"); ax.set_ylabel("Latitude (°)")
    ax.set_title("Decay Timescale Sensitivity"); ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig


# ─────────────────────────────────────────────
# 7. Diffusion evolution plot
# ─────────────────────────────────────────────

def plot_diffusion_evolution(t_snap_yr, B_snaps, lat_deg,
                               save_path="diffusion_evolution.png"):
    """
    Plot profile snapshots showing diffusion broadening and decay.
    Like Figure 5 in Diagnostic 1 of the reference report.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.viridis
    n = len(t_snap_yr)
    for i, (t, B) in enumerate(zip(t_snap_yr, B_snaps)):
        color = cmap(i / max(n - 1, 1))
        ax.plot(lat_deg, B, color=color, lw=2, label=f"t={t:.1f} yr")

    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("⟨Br⟩ (G)")
    ax.set_title("Diffusion: latitude profiles over time")
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE['fig_dpi'], bbox_inches='tight')
    print(f"  [saved] {save_path}")
    return fig
