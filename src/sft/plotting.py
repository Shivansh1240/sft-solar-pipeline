"""Figures for runs and benchmark results (needs matplotlib: ``pip install sft-solar[plot]``)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

# Palette tokens (validated: slots 1-3 pass all-pairs CVD checks on the light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
EMPHASIS_OFF = "#b5b3ac"

FIELD_CMAP = LinearSegmentedColormap.from_list(
    "sft_diverging",
    ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3b0af", "#e34948", "#9b2020"],
)


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "font.size": 9,
        }
    )


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _slope_guide(ax, x, y0, order, label):
    x = np.asarray(x, dtype=float)
    y = y0 * (x / x[0]) ** order
    ax.plot(x, y, color=MUTED, lw=0.9, zorder=1)
    ax.annotate(
        label,
        (x[-1], y[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        color=MUTED,
        fontsize=7.5,
        va="center",
    )


def plot_time_convergence(data: dict, path):
    """``data`` maps scheme -> {"dt_days": [...], "error": [...], "order": float}."""
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for i, (scheme, d) in enumerate(data.items()):
        dt, err = np.array(d["dt_days"]), np.array(d["error"])
        ax.loglog(
            dt, err, color=SERIES[i], marker=MARKERS[i], label=f"{scheme} (order {d['order']:.2f})"
        )
        ax.annotate(
            scheme,
            (dt[0], err[0]),
            xytext=(5, 0),
            textcoords="offset points",
            color=INK_2,
            fontsize=8,
            va="center",
        )
    ref = data[next(iter(data))]
    dts = np.array(ref["dt_days"])[::-1]
    errs_all = np.concatenate([np.array(d["error"]) for d in data.values()])
    _slope_guide(ax, dts, errs_all.min() * 0.5, 2, "slope 2")
    _slope_guide(ax, dts, errs_all.max() * 0.08, 1, "slope 1")
    ax.set_xlabel("time step [days]")
    ax.set_ylabel("max error after 1 yr [G]")
    ax.set_title("Temporal convergence")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_space_convergence(data: dict, path):
    """``data`` maps test label -> {"n_lat": [...], "error": [...], "order": float}.

    Errors are divided by the error on the coarsest grid so the tests share one axis.
    """
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for i, (label, d) in enumerate(data.items()):
        n, err = np.array(d["n_lat"]), np.array(d["error"])
        ax.loglog(
            n,
            err / err[0],
            color=SERIES[i],
            marker=MARKERS[i],
            label=f"{label} (order {d['order']:.2f})",
        )
    n_all = sorted({x for d in data.values() for x in d["n_lat"]})
    _slope_guide(ax, n_all, 1.6, -2, "slope -2")
    ax.set_xlabel("latitude cells")
    ax.set_ylabel("error / error on coarsest grid")
    ax.set_title("Spatial convergence")
    ax.legend(loc="lower left")
    return _save(fig, path)


def plot_cycle_run(result, path, cap_deg: float = 60.0, title="Synthetic cycles"):
    """Butterfly diagram, polar fields and axial dipole of a 1-D (or lon-averaged) run."""
    _style()
    fig, axes = plt.subplots(
        3, 1, figsize=(8, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]}
    )
    t = result.t_yr
    bf = result.butterfly()
    vmax = float(np.percentile(np.abs(bf), 99.5)) or 1.0
    ax = axes[0]
    mesh = ax.pcolormesh(
        t,
        result.grid.lat_deg,
        bf,
        cmap=FIELD_CMAP,
        norm=TwoSlopeNorm(0.0, -vmax, vmax),
        shading="auto",
        rasterized=True,
    )
    ax.grid(False)
    ax.set_ylabel("latitude [deg]")
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_title(title)
    cb = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label("longitude-averaged B [G]", color=INK_2, fontsize=8)
    cb.outline.set_visible(False)

    north, south = result.polar_fields(cap_deg)
    ax = axes[1]
    ax.axhline(0.0, color=AXIS, lw=0.8)
    ax.plot(t, north, color=SERIES[0], label=f"north (> {cap_deg:.0f} deg)")
    ax.plot(t, south, color=SERIES[1], label=f"south (< -{cap_deg:.0f} deg)")
    ax.annotate(
        "N",
        (t[-1], north[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        color=INK_2,
        fontsize=8,
        va="center",
    )
    ax.annotate(
        "S",
        (t[-1], south[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        color=INK_2,
        fontsize=8,
        va="center",
    )
    ax.set_ylabel("polar field [G]")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)

    ax = axes[2]
    ax.axhline(0.0, color=AXIS, lw=0.8)
    ax.plot(t, result.dipole(), color=SERIES[0])
    ax.set_ylabel("axial dipole [G]")
    ax.set_xlabel("time [yr]")
    fig.align_ylabels(axes)
    return _save(fig, path)


def plot_ensemble(t_yr, dipoles, path, title="Ensemble of axial dipole"):
    """Spread of the dipole across members: median, 5-95 % band, and end-of-run histogram."""
    _style()
    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [2.2, 1]}, sharey=True
    )
    lo, med, hi = np.percentile(dipoles, [5, 50, 95], axis=1)
    ax.axhline(0.0, color=AXIS, lw=0.8)
    ax.fill_between(t_yr, lo, hi, color=SERIES[0], alpha=0.18, lw=0, label="5-95 % of members")
    ax.plot(t_yr, med, color=SERIES[0], label="median")
    ax.set_xlabel("time [yr]")
    ax.set_ylabel("axial dipole [G]")
    ax.set_title(title)
    ax.legend(loc="upper left")
    final = dipoles[-1]
    axh.hist(
        final,
        bins=25,
        orientation="horizontal",
        color=SERIES[0],
        alpha=0.85,
        edgecolor=SURFACE,
        linewidth=1.0,
    )
    axh.axhline(np.median(final), color=INK_2, lw=0.9)
    axh.set_xlabel("members")
    axh.set_title(f"end of run: {final.mean():.2f} ± {final.std():.2f} G")
    return _save(fig, path)


def plot_runtime(rows, path, title="Wall-clock time (lower is better)"):
    """Horizontal bars on a log axis. ``rows`` is a list of (label, seconds, highlight)."""
    _style()
    fig, ax = plt.subplots(figsize=(7.0, 0.55 * len(rows) + 1.0))
    labels = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows])
    colors = [SERIES[0] if r[2] else EMPHASIS_OFF for r in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=colors, height=0.6)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, vals, strict=True):
        ax.annotate(
            f"{v:.3g} s",
            (v, yi),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=8,
        )
    ax.set_xlim(vals.min() * 0.4, vals.max() * 6)
    ax.set_xlabel("seconds (log scale)")
    ax.set_title(title)
    return _save(fig, path)


def plot_2d_snapshots(result, path, indices=None, lat_window=(-60, 60)):
    """Maps of a 2-D run at a few times."""
    _style()
    B = result.B
    indices = indices if indices is not None else np.linspace(0, len(B) - 1, 4).astype(int)
    vmax = float(np.percentile(np.abs(B[indices[0]]), 99.9)) or 1.0
    fig, axes = plt.subplots(len(indices), 1, figsize=(7.0, 1.9 * len(indices)), sharex=True)
    g = result.grid
    for ax, i in zip(np.atleast_1d(axes), indices, strict=True):
        mesh = ax.pcolormesh(
            g.lon_deg,
            g.lat_deg,
            B[i],
            cmap=FIELD_CMAP,
            norm=TwoSlopeNorm(0.0, -vmax, vmax),
            shading="auto",
            rasterized=True,
        )
        ax.grid(False)
        ax.set_ylim(*lat_window)
        ax.set_ylabel("lat [deg]")
        ax.set_title(f"t = {result.t_yr[i]:.2f} yr", loc="left", fontsize=9)
    np.atleast_1d(axes)[-1].set_xlabel("Carrington longitude [deg]")
    cb = fig.colorbar(mesh, ax=axes, pad=0.01, fraction=0.03)
    cb.set_label(f"B [G] (colour scale clipped at ±{vmax:.0f} G)", color=INK_2, fontsize=8)
    cb.outline.set_visible(False)
    return _save(fig, path)
