"""Publication-quality cooperation-rate figures for IPD behavioral study."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Publication styling — exactly matches ipd_logits_viz.py
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "DejaVu Serif"],
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.titleweight":      "bold",
    "axes.labelsize":        10,
    "legend.fontsize":       8.5,
    "legend.frameon":        False,
    "legend.title_fontsize": 9,
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             True,
    "grid.alpha":            0.15,
    "grid.linestyle":        "-",
    "lines.linewidth":       1.6,
    "lines.markersize":      4.5,
})

# Okabe-Ito colorblind-safe palette — one colour per window size
WIN_COLORS  = {2: "#E69F00", 4: "#0072B2", 8: "#009E73"}
WIN_MARKERS = {2: "o",       4: "s",       8: "^"}

# Ocean Dusk palette — one colour per opponent (matches ipd_logits_viz.py OCEAN)
OPP_COLORS = {
    "ALLC":   "#264653",   # deep teal
    "ALLD":   "#E76F51",   # burnt coral
    "TFT":    "#2A9D8F",   # teal
    "TF2T":   "#9B59B6",   # purple (replaces yellow — visible on white)
    "RANDOM": "#F4A261",   # sandy orange
}
OPP_MARKERS = {
    "ALLC":   "o",
    "ALLD":   "s",
    "TFT":    "^",
    "TF2T":   "D",
    "RANDOM": "v",
}

# ICML figure widths
FIG_FULL = 6.75   # inches, full-width

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/ipd_behavioral")
OPPONENTS   = ["ALLC", "ALLD", "TFT", "TF2T", "RANDOM"]
WINDOWS     = [2, 4, 8]

OPP_LABELS = {
    "ALLC":   "Always Cooperate (ALLC)",
    "ALLD":   "Always Defect (ALLD)",
    "TFT":    "Tit-for-Tat (TFT)",
    "TF2T":   "Tit-for-Two-Tats (TF2T)",
    "RANDOM": "Random 50/50",
}

OPP_LABELS_SHORT = {
    "ALLC":   "Always Cooperate",
    "ALLD":   "Always Defect",
    "TFT":    "Tit-for-Tat",
    "TF2T":   "Tit-for-Two-Tats",
    "RANDOM": "Random 50/50",
}

OPP_LINESTYLES = {
    "ALLC":   "-",
    "ALLD":   "-",
    "TFT":    "-",
    "TF2T":   "-",
    "RANDOM": "-",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> dict:
    """
    Returns nested dict: data[opp][window] = {
        "coop":    np.ndarray [n_runs, n_rounds]  (1=C, 0=D)
        "n_runs":  int
        "n_rounds": int
    }
    """
    data: dict = {}
    for opp in OPPONENTS:
        data[opp] = {}
        for w in WINDOWS:
            path = RESULTS_DIR / f"{opp}_w{w}.json"
            if not path.exists():
                print(f"  (skipping {opp} w={w} — file not found)")
                continue
            runs = json.loads(path.read_text())
            coop = np.array(
                [[1 if r["llm_action"] == "C" else 0 for r in run] for run in runs],
                dtype=float,
            )  # [n_runs, n_rounds]
            data[opp][w] = {
                "coop":     coop,
                "n_runs":   coop.shape[0],
                "n_rounds": coop.shape[1],
            }
    n_loaded = sum(len(data[o]) for o in OPPONENTS)
    print(f"Loaded {n_loaded} / {len(OPPONENTS) * len(WINDOWS)} combos")
    return data


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------


def mean_ci95(coop: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-round mean and 95% Wilson-like CI from [n_runs, n_rounds] binary array."""
    n  = coop.shape[0]
    mu = coop.mean(axis=0)
    se = np.sqrt(np.clip(mu * (1 - mu), 0, None) / n)
    z  = 1.96
    return mu, np.clip(mu - z * se, 0, 1), np.clip(mu + z * se, 0, 1)


# ---------------------------------------------------------------------------
# Figure 1: per-opponent panel, lines = window sizes
# ---------------------------------------------------------------------------


def plot_coop_by_opponent(data: dict) -> None:
    """
    5 stacked subplots (one per opponent).
    3 lines per subplot (window 2, 4, 8) with 95% CI ribbons.
    x = round, y = P(Cooperate).
    """
    n_opp = len(OPPONENTS)
    fig, axes = plt.subplots(
        n_opp, 1,
        figsize=(FIG_FULL, 2.6 * n_opp),
        sharex=True,
    )

    for ax, opp in zip(axes, OPPONENTS):
        avail = [w for w in WINDOWS if w in data[opp]]
        if not avail:
            ax.set_visible(False)
            continue

        for w in avail:
            coop      = data[opp][w]["coop"]
            n_rounds  = data[opp][w]["n_rounds"]
            n_runs    = data[opp][w]["n_runs"]
            rounds    = np.arange(1, n_rounds + 1)
            mu, lo, hi = mean_ci95(coop)
            color     = WIN_COLORS[w]
            marker    = WIN_MARKERS[w]
            every     = max(1, n_rounds // 10)

            ax.plot(rounds, mu, color=color, lw=1.6,
                    marker=marker, markevery=every,
                    markersize=3.5, label=f"$w={w}$", zorder=3)
            ax.fill_between(rounds, lo, hi, color=color, alpha=0.15, zorder=2)

        ax.axhline(0.5, color="#888", lw=0.9, ls="--", alpha=0.7, zorder=1)
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.tick_params(labelsize=9)
        ax.set_title(OPP_LABELS[opp], loc="left", pad=4, fontsize=10.5)

    axes[0].legend(title="History window", loc="upper right",
                   handlelength=2.0, ncol=len(WINDOWS))
    axes[-1].set_xlabel("Round")
    fig.text(0.03, 0.5, "$P$(Cooperate)", va="center",
             rotation="vertical", fontsize=10)
    fig.suptitle(
        "Gemma-2-2b-it IPD Cooperation Rate (n=30, T=0.7, 95% CI)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    _save(fig, "coop_rate_by_opponent")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: per-window panel, lines = opponents
# ---------------------------------------------------------------------------


def _rolling(arr: np.ndarray, win: int) -> np.ndarray:
    """Apply a centred rolling mean to a 1-D array, preserving edges."""
    kernel = np.ones(win) / win
    padded = np.pad(arr, win // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def plot_coop_by_window(data: dict) -> None:
    """
    3 side-by-side subplots (one per window).
    5 lines per subplot (one per opponent), smoothed with a rolling mean.
    CI bands omitted to reduce clutter; each opponent uses both colour + linestyle.
    Shared legend below panels.
    """
    avail_windows = [w for w in WINDOWS if any(w in data[o] for o in OPPONENTS)]
    n_win = len(avail_windows)
    if n_win == 0:
        print("No data available for per-window plot.")
        return

    SMOOTH = 9  # rounds rolling-mean window

    fig, axes = plt.subplots(
        n_win, 1,
        figsize=(FIG_FULL, 3.2 * n_win),
        sharex=True, sharey=True,
    )
    if n_win == 1:
        axes = [axes]

    legend_handles = []
    for ax, w in zip(axes, avail_windows):
        for opp in OPPONENTS:
            if w not in data[opp]:
                continue
            coop      = data[opp][w]["coop"]
            n_rounds  = data[opp][w]["n_rounds"]
            rounds    = np.arange(1, n_rounds + 1)
            mu, lo, hi = mean_ci95(coop)
            color     = OPP_COLORS[opp]
            ls        = OPP_LINESTYLES[opp]

            line, = ax.plot(rounds, mu, color=color, lw=1.9,
                            linestyle=ls, label=OPP_LABELS_SHORT[opp], zorder=3)
            ax.fill_between(rounds, lo, hi, color=color, alpha=0.10, zorder=2)
            if ax is axes[0]:
                legend_handles.append(line)

        ax.axhline(0.5, color="#888", lw=0.8, ls="--", alpha=0.6, zorder=1)
        ax.set_ylabel(f"$w = {w}$\n$P$(Coop.)", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.tick_params(labelsize=9)

    axes[-1].set_xlabel("Round")

    axes[0].legend(
        handles=legend_handles,
        loc="upper right",
        ncol=1,
        fontsize=8.5,
        handlelength=2.2,
        frameon=False,
    )
    fig.suptitle(
        "Cooperation Rate by History Window — Gemma-2-2b-it (n=30, T=0.7)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "coop_rate_by_window")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------


def print_summary(data: dict) -> None:
    print(f"\n{'='*68}")
    print(f"{'Opponent':<10} {'w':>3}  {'P(C)':>6}  {'95% CI':>18}  {'n_runs':>6}")
    print(f"{'='*68}")
    for opp in OPPONENTS:
        for w in WINDOWS:
            if w not in data[opp]:
                continue
            coop = data[opp][w]["coop"]
            p    = coop.mean()
            n    = coop.size
            se   = np.sqrt(p * (1 - p) / n)
            lo   = max(0.0, p - 1.96 * se)
            hi   = min(1.0, p + 1.96 * se)
            print(
                f"{opp:<10} {w:>3}  {p:>6.3f}"
                f"  [{lo:.3f}, {hi:.3f}]"
                f"  {data[opp][w]['n_runs']:>6}"
            ) 
        print()


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def _save(fig, stem: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = RESULTS_DIR / f"{stem}.{ext}"
        fig.savefig(path)
        print(f"Saved → {path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_data()
    plot_coop_by_opponent(data)
    plot_coop_by_window(data)
    print_summary(data)
