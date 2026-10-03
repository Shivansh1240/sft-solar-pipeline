"""Figures for runs and benchmark results (needs matplotlib: ``pip install sft-solar[plot]``)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

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


def _plain_log_ticks(axis, values, fmt="{:g}"):
    """Put ticks exactly at ``values`` on a log axis, labelled as plain numbers."""
    values = sorted(set(float(v) for v in values))
    axis.set_major_locator(FixedLocator(values))
    axis.set_minor_locator(NullLocator())
    axis.set_ticklabels([fmt.format(v) for v in values])


def _label_end(ax, x, y, text, dy=0):
    ax.annotate(
        text,
        (x, y),
        xytext=(5, dy),
        textcoords="offset points",
        color=INK_2,
        fontsize=8,
        va="center",
    )


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
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    offsets = [0, 7, -7, 0]
    for i, (scheme, d) in enumerate(data.items()):
        dt, err = np.array(d["dt_days"]), np.array(d["error"])
        ax.loglog(
            dt, err, color=SERIES[i], marker=MARKERS[i], label=f"{scheme} (order {d['order']:.2f})"
        )
        _label_end(ax, dt[0], err[0], scheme, offsets[i])
    dts = np.sort(np.array(data[next(iter(data))]["dt_days"]))
    errs_all = np.concatenate([np.array(d["error"]) for d in data.values()])
    _slope_guide(ax, dts, errs_all.min() * 0.4, 2, "slope 2")
    _slope_guide(ax, dts, errs_all.max() * 0.06, 1, "slope 1")
    _plain_log_ticks(ax.xaxis, dts, "{:.3g}")
    ax.set_xlim(dts[0] * 0.8, dts[-1] * 1.6)
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
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
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
    _plain_log_ticks(ax.xaxis, n_all, "{:.0f}")
    ax.set_xlabel("latitude cells")
    ax.set_ylabel("error / error on coarsest grid")
    ax.set_title("Spatial convergence")
    ax.legend(loc="lower left")
    return _save(fig, path)


def plot_accuracy_cost(data: dict, path):
    """``data`` maps scheme -> list of {"wall_s", "error", "dt_days"} rows."""
    _style()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    offsets = [0, 7, -7, 0]
    for i, (scheme, rows) in enumerate(data.items()):
        wall = np.array([r["wall_s"] for r in rows]) * 1e3
        err = np.array([r["error"] for r in rows])
        ax.loglog(wall, err, color=SERIES[i], marker=MARKERS[i], label=scheme)
        _label_end(ax, wall[-1], err[-1], scheme, offsets[i])
    ax.set_xlabel("wall-clock time for 1 simulated year [ms]")
    ax.set_ylabel("max error [G]")
    ax.set_title("Accuracy versus cost (lower-left is better)")
    ax.legend(loc="upper right")
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
    _label_end(ax, t[-1], north[-1], "N")
    _label_end(ax, t[-1], south[-1], "S")
    ax.set_ylabel("polar field [G]")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)

    ax = axes[2]
    ax.axhline(0.0, color=AXIS, lw=0.8)
    ax.plot(t, result.dipole(), color=SERIES[0])
    ax.set_ylabel("axial dipole [G]")
    ax.set_xlabel("time [yr]")
    fig.align_ylabels(axes)
    return _save(fig, path)


def plot_ensemble(series: dict, path, title="Ensemble of axial dipole"):
    """``series`` maps label -> (t_yr, D) with D shaped (n_time, members).

    Left: median and 5-95 % band over time. Right: distribution at the final time.
    """
    _style()
    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(8.6, 3.7), gridspec_kw={"width_ratios": [2.0, 1.1]}
    )
    ax.axhline(0.0, color=AXIS, lw=0.8)
    finals = []
    for i, (label, (t, D)) in enumerate(series.items()):
        lo, med, hi = np.percentile(D, [5, 50, 95], axis=1)
        ax.fill_between(t, lo, hi, color=SERIES[i], alpha=0.16, lw=0)
        ax.plot(t, med, color=SERIES[i], label=label)
        finals.append(D[-1])
    ax.set_xlabel("time [yr]")
    ax.set_ylabel("axial dipole [G]")
    ax.set_title(title)
    ax.legend(loc="upper left", title="median and 5-95 % band", title_fontsize=8)

    allf = np.concatenate(finals)
    bins = np.linspace(allf.min(), allf.max(), 24)
    for i, (label, f) in enumerate(zip(series, finals, strict=True)):
        axh.hist(f, bins=bins, color=SERIES[i], alpha=0.55, label=label, edgecolor=SURFACE, lw=0.5)
        axh.axvline(np.median(f), color=SERIES[i], lw=1.2)
    axh.set_xlabel(f"axial dipole at t = {series[label][0][-1]:.0f} yr [G]")
    axh.set_ylabel("members")
    axh.set_title("final distribution")
    return _save(fig, path)


def plot_runtime(rows, path, title="Wall-clock time (lower is better)"):
    """Horizontal bars on a log axis. ``rows`` is a list of (label, seconds, highlight)."""
    _style()
    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(rows) + 1.0))
    labels = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows])
    colors = [SERIES[0] if r[2] else EMPHASIS_OFF for r in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=colors, height=0.62)
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
    ax.set_xlabel("seconds (log scale); blue = this code, grey = previous code")
    ax.set_title(title)
    return _save(fig, path)


def plot_2d_snapshots(result, path, indices=None, lat_window=(-60, 60)):
    """Maps of a 2-D run at a few times, each panel on its own symmetric colour scale."""
    _style()
    B = result.B
    indices = indices if indices is not None else np.linspace(0, len(B) - 1, 4).astype(int)
    fig, axes = plt.subplots(len(indices), 1, figsize=(7.4, 1.9 * len(indices)), sharex=True)
    g = result.grid
    for ax, i in zip(np.atleast_1d(axes), indices, strict=True):
        vmax = float(np.max(np.abs(B[i]))) or 1.0
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
        cb = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025)
        cb.set_label("B [G]", color=INK_2, fontsize=8)
        cb.outline.set_visible(False)
    np.atleast_1d(axes)[-1].set_xlabel("Carrington longitude [deg]")
    return _save(fig, path)
