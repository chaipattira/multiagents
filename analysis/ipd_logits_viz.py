"""Publication-quality figures for IPD DLA sweep."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Publication styling
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif"],
    "font.size":            10,
    "axes.titlesize":       11,
    "axes.titleweight":     "bold",
    "axes.labelsize":       10,
    "legend.fontsize":      8.5,
    "legend.frameon":       False,
    "legend.title_fontsize": 9,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.alpha":           0.15,
    "grid.linestyle":       "-",
    "lines.linewidth":      1.6,
    "lines.markersize":     4.5,
})

# Okabe-Ito colorblind-safe — one colour per window size (2, 4, 8)
WIN_COLORS = ["#E69F00", "#0072B2", "#009E73"]   # orange, blue, green

# Ocean Dusk accent colours — used for cross-opponent top-heads, etc.
OCEAN   = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]

# Figure widths: ICML single-column (3.25 in) and full-width (6.75 in)
FIG_SINGLE = (3.25, 2.5)
FIG_FULL   = (6.75, 2.8)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/ipd_sweep")
OPPONENTS   = ["ALLC", "ALLD", "TFT", "TF2T", "RANDOM"]
WINDOWS     = [2, 4, 8]

OPP_LABELS = {
    "ALLC":   "Always Cooperate",
    "ALLD":   "Always Defect",
    "TFT":    "Tit-for-Tat",
    "TF2T":   "Tit-for-Two-Tats",
    "RANDOM": "Random",
}

_LLM_PAYOFF = {("C", "C"): 3, ("C", "D"): 0, ("D", "C"): 5, ("D", "D"): 1}
_OPP_PAYOFF = {("C", "C"): 3, ("C", "D"): 5, ("D", "C"): 0, ("D", "D"): 1}
_SOCIAL     = {("C", "C"): 6, ("C", "D"): 5, ("D", "C"): 5, ("D", "D"): 2}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def per_player_payoff(llm_acts, opp_acts):
    pairs = list(zip(llm_acts, opp_acts))
    return (
        np.array([_LLM_PAYOFF[p] for p in pairs], dtype=float),
        np.array([_OPP_PAYOFF[p] for p in pairs], dtype=float),
    )


def social_utility(llm_acts, opp_acts):
    return np.array([_SOCIAL[(l, o)] for l, o in zip(llm_acts, opp_acts)], dtype=float)


def _save(fig, stem: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = RESULTS_DIR / f"{stem}.{ext}"
        fig.savefig(path)
        print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> dict:
    data = {}
    for opp in OPPONENTS:
        data[opp] = {}
        for w in WINDOWS:
            path = RESULTS_DIR / f"{opp}_w{w}.json"
            if not path.exists():
                print(f"  (skipping {opp} w={w} — not yet available)")
                continue
            records = json.loads(path.read_text())
            data[opp][w] = {
                "rounds":             [r["round"]      for r in records],
                "logit_diff":         [r["logit_diff"] for r in records],
                "llm_action":         [r["llm_action"] for r in records],
                "opp_action":         [r["opp_action"] for r in records],
                "logit_lens":         np.array([r["logit_lens"]         for r in records]),  # (T, L)
                "head_contributions": np.array([r["head_contributions"] for r in records]),  # (T, L, H)
            }
    n_combos = sum(len(data[o]) for o in OPPONENTS)
    print(f"Loaded {n_combos} / {len(OPPONENTS) * len(WINDOWS)} combos")
    return data


# ---------------------------------------------------------------------------
# Plot 1: Logit diff over rounds
# ---------------------------------------------------------------------------

def plot_logit_diff(data: dict) -> None:
    n = len(OPPONENTS)
    fig, axes = plt.subplots(n, 1, figsize=(6.75, 2.6 * n), sharex=True)

    for ax, opp in zip(axes, OPPONENTS):
        avail = [w for w in WINDOWS if w in data[opp]]
        if not avail:
            ax.set_visible(False)
            continue

        all_ld = [np.array(data[opp][w]["logit_diff"]) for w in avail]
        ylo = min(ld.min() for ld in all_ld)
        yhi = max(ld.max() for ld in all_ld)
        pad = max((yhi - ylo) * 0.12, 0.5)
        ylo -= pad; yhi += pad

        ax.axhspan(0,   yhi, color="#FDDEDE", alpha=0.55, zorder=0, lw=0)
        ax.axhspan(ylo, 0,   color="#DCE9F5", alpha=0.55, zorder=0, lw=0)

        markers = {2: "o", 4: "s", 8: "^"}
        colors  = {2: WIN_COLORS[0], 4: WIN_COLORS[1], 8: WIN_COLORS[2]}
        for w in avail:
            ld     = np.array(data[opp][w]["logit_diff"])
            rounds = np.array(data[opp][w]["rounds"])
            n_pts  = len(rounds)
            ax.plot(rounds, ld, color=colors[w], lw=1.6,
                    marker=markers[w], markevery=max(1, n_pts // 10),
                    markersize=3.5, label=f"$w={w}$", zorder=3)

        ax.axhline(0, color="#888", lw=0.9, ls="--", alpha=0.9, zorder=1)
        ax.set_ylim(ylo, yhi)
        ax.text(0.01, 0.97, OPP_LABELS[opp], transform=ax.transAxes,
                va="top", ha="left", fontsize=10, fontweight="bold", zorder=5,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                          boxstyle="round,pad=0.25"))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.tick_params(labelsize=9)

    axes[0].legend(title="History window", loc="upper right",
                   handlelength=2.0, ncol=3)
    axes[-1].set_xlabel("Round")
    fig.text(0.03, 0.5, "Logit diff (D − C)", va="center",
             rotation="vertical", fontsize=10)
    fig.suptitle("Gemma-2-2b-it · IPD Logit Diff Sweep",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    _save(fig, "logit_diff_sweep")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Logit-lens layer profile  + cross-condition fork
# ---------------------------------------------------------------------------

def plot_logit_lens_profile(data: dict) -> None:
    """
    5 per-opponent panels (mean ± SD across rounds) + 1 summary panel
    showing cross-condition SD per layer (Experiment 0 fork diagnostic).
    """
    markers = {2: "o", 4: "s", 8: "^"}
    colors  = {2: WIN_COLORS[0], 4: WIN_COLORS[1], 8: WIN_COLORS[2]}

    # Pre-compute per-condition means (needed for fork panel)
    cond_means: list[np.ndarray] = []
    for opp in OPPONENTS:
        for w in WINDOWS:
            if w in data[opp]:
                cond_means.append(data[opp][w]["logit_lens"].mean(axis=0))

    cross_sd  = np.stack(cond_means).std(axis=0) if cond_means else None
    n_layers  = len(cond_means[0]) if cond_means else 26
    layers    = np.arange(n_layers)

    n_opp  = len(OPPONENTS)
    ratios = [2.5] * n_opp + [1.2]
    fig, axes = plt.subplots(
        n_opp + 1, 1,
        figsize=(6.75, 2.5 * n_opp + 1.8),
        sharex=True,
        gridspec_kw={"height_ratios": ratios},
    )

    for ax, opp in zip(axes[:n_opp], OPPONENTS):
        avail = [w for w in WINDOWS if w in data[opp]]
        if not avail:
            ax.set_visible(False)
            continue

        all_means = [data[opp][w]["logit_lens"].mean(axis=0) for w in avail]
        all_stds  = [data[opp][w]["logit_lens"].std(axis=0)  for w in avail]

        ylo = min((m - s).min() for m, s in zip(all_means, all_stds))
        yhi = max((m + s).max() for m, s in zip(all_means, all_stds))
        pad = max((yhi - ylo) * 0.12, 0.5)
        ylo -= pad; yhi += pad

        ax.axhspan(0,   yhi, color="#FDDEDE", alpha=0.4, zorder=0, lw=0)
        ax.axhspan(ylo, 0,   color="#DCE9F5", alpha=0.4, zorder=0, lw=0)
        ax.axvspan(17.5, n_layers - 0.5, color="#FFF3CD", alpha=0.4, zorder=0)

        for w, mean, std in zip(avail, all_means, all_stds):
            ax.plot(layers, mean, color=colors[w], lw=1.6,
                    marker=markers[w], markevery=max(1, n_layers // 8),
                    markersize=3.5, label=f"$w={w}$", zorder=3)
            ax.fill_between(layers, mean - std, mean + std,
                            color=colors[w], alpha=0.14, zorder=2)

        ax.axhline(0, color="#888", lw=0.9, ls="--", alpha=0.9, zorder=1)
        ax.set_ylim(ylo, yhi)
        ax.text(0.01, 0.97, OPP_LABELS[opp], transform=ax.transAxes,
                va="top", ha="left", fontsize=10, fontweight="bold", zorder=5,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                          boxstyle="round,pad=0.25"))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.tick_params(labelsize=9)

    axes[0].legend(title="History window", loc="upper right",
                   handlelength=2.0, ncol=3)

    # --- Fork panel (Exp 0): cross-condition SD per layer ---
    ax_fork = axes[n_opp]
    if cross_sd is not None:
        ax_fork.fill_between(layers, 0, cross_sd, color="#56B4E9", alpha=0.22, zorder=1)
        ax_fork.plot(layers, cross_sd, color="#0072B2", lw=1.6, zorder=3)
        ax_fork.axhline(0.2, color="#E76F51", lw=1.0, ls="--", alpha=0.85,
                        label="SD = 0.2")
        ax_fork.axhline(1.0, color="#009E73", lw=1.0, ls="--", alpha=0.85,
                        label="SD = 1.0")
        ax_fork.axvspan(17.5, n_layers - 0.5, color="#FFF3CD", alpha=0.4, zorder=0)
        ax_fork.yaxis.set_major_locator(ticker.MaxNLocator(3))
        ax_fork.legend(loc="upper left", fontsize=8, title="Cross-cond. SD")

    ax_fork.set_ylabel("SD across\nconditions")
    ax_fork.set_xlabel("Layer")
    ax_fork.set_xticks(np.arange(0, n_layers, 2))

    fig.text(0.03, 0.5, "Accumulated logit diff (D − C)", va="center",
             rotation="vertical", fontsize=10)
    fig.suptitle("Where Does the D/C Decision Come From?",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0.06, 0, 1, 1])
    _save(fig, "logit_lens_profile")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Head contribution heatmaps (per opponent)
# ---------------------------------------------------------------------------

def plot_head_contributions(data: dict) -> None:
    for opp in OPPONENTS:
        avail = [w for w in WINDOWS if w in data[opp]]
        if not avail:
            continue

        vmax = max(
            np.abs(data[opp][w]["head_contributions"].mean(axis=0)).max()
            for w in avail
        )

        n_cols = len(avail)
        fig, axes = plt.subplots(1, n_cols,
                                 figsize=(3.0 * n_cols, 7.0),
                                 sharey=True)
        if n_cols == 1:
            axes = [axes]
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])

        for ax, w in zip(axes, avail):
            mean_hc = data[opp][w]["head_contributions"].mean(axis=0)  # (L, H)

            # Determine top-5 cells to annotate (highest |mean contribution|)
            flat_top = np.argsort(np.abs(mean_hc).ravel())[::-1][:5]

            # Build annotation matrix: blank except for top-5
            annot = np.full(mean_hc.shape, "", dtype=object)
            for idx in flat_top:
                r, c = divmod(idx, mean_hc.shape[1])
                annot[r, c] = f"{mean_hc[r, c]:+.2f}"

            ax.grid(False)
            sns.heatmap(
                mean_hc,
                ax=ax,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                center=0,
                annot=annot,
                fmt="",
                annot_kws={"size": 6, "weight": "bold"},
                linewidths=0.4,
                linecolor="#EEEEEE",
                cbar=w == avail[-1],
                cbar_ax=cbar_ax if w == avail[-1] else None,
                cbar_kws={"label": "Mean DLA (D − C direction)", "shrink": 0.8},
            )
            ax.set_title(f"$w={w}$", pad=4)
            ax.set_xlabel("Head")
            ax.tick_params(labelsize=8)
            if w != avail[0]:
                ax.set_ylabel("")
            else:
                ax.set_ylabel("Layer")

        fig.suptitle(
            f"Head DLA Contributions: {OPP_LABELS[opp]}",
            fontsize=11, fontweight="bold",
        ) # (red → Defect, blue → Cooperate; annotated: top-5 by |DLA|)
        plt.subplots_adjust(right=0.90, wspace=0.08)
        _save(fig, f"head_contributions_{opp}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Top-head trajectories over rounds
# ---------------------------------------------------------------------------

def plot_top_heads_over_rounds(data: dict, n_top: int = 5) -> None:
    win_markers = {2: "o", 4: "s", 8: "^"}
    win_colors  = {2: WIN_COLORS[0], 4: WIN_COLORS[1], 8: WIN_COLORS[2]}

    for opp in OPPONENTS:
        avail = [w for w in WINDOWS if w in data[opp]]
        if not avail:
            continue

        pooled = np.concatenate(
            [data[opp][w]["head_contributions"] for w in avail], axis=0
        )  # (T_total, L, H)
        mean_abs  = np.abs(pooled).mean(axis=0)
        flat_top  = np.argsort(mean_abs.ravel())[::-1][:n_top]
        top_heads = [(int(i // mean_abs.shape[1]), int(i % mean_abs.shape[1]))
                     for i in flat_top]
        head_colors = (OCEAN + OCEAN)[:n_top]

        n_rows = len(avail)
        fig, axes = plt.subplots(n_rows, 1,
                                 figsize=(6.75, 2.8 * n_rows),
                                 sharex=True)
        if n_rows == 1:
            axes = [axes]

        for ax, w in zip(axes, avail):
            hc     = data[opp][w]["head_contributions"]  # (T, L, H)
            rounds = np.array(data[opp][w]["rounds"])
            n_pts  = len(rounds)
            marker = win_markers[w]

            for (layer, head), hcolor in zip(top_heads, head_colors):
                series = hc[:, layer, head]
                ax.plot(rounds, series, color=hcolor, lw=1.4,
                        marker=marker, markevery=max(1, n_pts // 10),
                        markersize=3.5,
                        label=f"L{layer}H{head}", zorder=3)

            ax.axhline(0, color="#888", lw=0.8, ls="--", alpha=0.7, zorder=1)
            ax.text(0.01, 0.97, f"$w={w}$", transform=ax.transAxes,
                    va="top", ha="left", fontsize=10, fontweight="bold", zorder=5,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                              boxstyle="round,pad=0.25"))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.tick_params(labelsize=9)

        axes[0].legend(title="Head", loc="upper right",
                       handlelength=1.8, ncol=n_top, fontsize=8)
        axes[-1].set_xlabel("Round")
        fig.text(0.03, 0.5, "DLA contribution (D − C)", va="center",
                 rotation="vertical", fontsize=10)
        fig.suptitle(f"Top-{n_top} Heads by |DLA| — {OPP_LABELS[opp]}",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout(rect=[0.06, 0, 1, 1])
        _save(fig, f"top_heads_{opp}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: Head DLA — condition-invariant vs condition-sensitive decomposition
# ---------------------------------------------------------------------------

def plot_head_condition_decomposition(data: dict) -> None:
    """
    Decompose head DLA contributions into:
      - condition-invariant: cross-condition std per (layer, head) — left panel
      - condition-sensitive: cosine similarity between condition DLA vectors — right panel
    """
    conditions = [(opp, w) for opp in OPPONENTS for w in WINDOWS if w in data[opp]]
    if not conditions:
        return

    # Mean DLA per condition → (n_cond, L, H)
    mean_hc = np.stack([
        data[opp][w]["head_contributions"].mean(axis=0)
        for opp, w in conditions
    ])
    n_cond, n_layers, n_heads = mean_hc.shape

    # Cross-condition std per head: (L, H)
    cross_std = mean_hc.std(axis=0)

    # Annotate top-5 most sensitive heads by name
    annot_std = np.full(cross_std.shape, "", dtype=object)
    for idx in np.argsort(cross_std.ravel())[::-1][:5]:
        r, c = divmod(idx, n_heads)
        annot_std[r, c] = f"L{r}H{c}"

    # Cosine similarity on mean-centered vectors (removes shared D-bias)
    flat      = mean_hc.reshape(n_cond, -1)
    flat      = flat - flat.mean(axis=0)
    norms     = np.linalg.norm(flat, axis=1, keepdims=True)
    flat_norm = flat / np.clip(norms, 1e-8, None)
    cos_sim   = flat_norm @ flat_norm.T

    cond_labels = [f"{opp[:4]}-{w}" for opp, w in conditions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 7.5))

    # --- Left: cross-condition std ---
    ax1.grid(False)
    sns.heatmap(
        cross_std, ax=ax1,
        cmap="Oranges",
        annot=annot_std, fmt="",
        annot_kws={"size": 7, "weight": "bold", "color": "black"},
        linewidths=0.3, linecolor="#EEEEEE",
        cbar_kws={"label": "Cross-condition SD", "shrink": 0.7},
    )
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_title("Cross-condition SD", loc="left")

    # --- Right: cosine similarity ---
    ax2.grid(False)
    sns.heatmap(
        cos_sim, ax=ax2,
        cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f",
        annot_kws={"size": 5.5},
        linewidths=0.4, linecolor="white",
        xticklabels=cond_labels, yticklabels=cond_labels,
        cbar_kws={"label": "Cosine similarity", "shrink": 0.7},
    )
    ax2.set_title("Condition similarity (mean-centred RSA)", loc="left")
    ax2.tick_params(labelsize=6.5)

    fig.suptitle("Head DLA Decomposition", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "head_condition_decomposition")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: Lagged correlation heatmap
# ---------------------------------------------------------------------------

_VARIABLE_OPPS = ["TFT", "TF2T", "RANDOM"]  # ALLC/ALLD are constant → corr undefined


def _lagged_corr(llm: list[str], opp: list[str], max_lag: int) -> np.ndarray:
    llm_bin = np.array([1.0 if a == "C" else 0.0 for a in llm])
    opp_bin = np.array([1.0 if a == "C" else 0.0 for a in opp])
    corrs = []
    for lag in range(1, max_lag + 1):
        x, y = opp_bin[:-lag], llm_bin[lag:]
        corrs.append(np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan)
    return np.array(corrs)


def plot_lagged_corr(data: dict) -> None:
    max_lag = max(WINDOWS)
    cmap    = plt.cm.RdBu_r.copy()
    cmap.set_bad("#CCCCCC")

    fig, axes = plt.subplots(len(_VARIABLE_OPPS), 1,
                             figsize=(6.75, 2.8 * len(_VARIABLE_OPPS)),
                             constrained_layout=True)

    im = None
    for ax, opp in zip(axes, _VARIABLE_OPPS):
        avail = [w for w in WINDOWS if w in data[opp]]
        mat   = np.full((len(avail), max_lag), np.nan)
        for ri, w in enumerate(avail):
            d = data[opp][w]
            mat[ri, :w] = _lagged_corr(d["llm_action"], d["opp_action"], w)

        im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        for ri, w in enumerate(avail):
            for ci in range(max_lag):
                if ci < w:
                    v     = mat[ri, ci]
                    txt   = f"{v:.2f}" if not np.isnan(v) else "n/a"
                    color = "white" if (not np.isnan(v) and abs(v) > 0.55) else "black"
                else:
                    txt, color = "–", "#888888"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

        ax.set_yticks(range(len(avail)))
        ax.set_yticklabels([f"$w={w}$" for w in avail])
        ax.set_xticks(range(max_lag))
        ax.set_xticklabels([str(i + 1) for i in range(max_lag)])
        ax.set_xticks(np.arange(-0.5, max_lag), minor=True)
        ax.set_yticks(np.arange(-0.5, len(avail)), minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="minor", size=0)
        ax.text(0.01, 0.97, OPP_LABELS[opp], transform=ax.transAxes,
                va="top", ha="left", fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                          boxstyle="round,pad=0.25"))

    axes[-1].set_xlabel("Lag (rounds)")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02,
                     label=r"$\mathrm{corr}(\mathrm{opp}[t - \mathrm{lag}],\; \mathrm{llm}[t])$")
    fig.suptitle(
        "Does History Window Matter?",
        fontsize=11, fontweight="bold",
    )
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93))
    _save(fig, "lagged_corr_sweep")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_action_summary(data: dict) -> None:
    for opp in OPPONENTS:
        print(f"\n{'='*65}")
        print(f"Opponent: {OPP_LABELS[opp]}")
        print(f"{'='*65}")

        avail = [w for w in WINDOWS if w in data[opp]]
        c_rates: dict[int, float] = {}
        for w in avail:
            d        = data[opp][w]
            llm      = d["llm_action"]
            opp_acts = d["opp_action"]
            ld       = np.array(d["logit_diff"])
            llm_pay, opp_pay = per_player_payoff(llm, opp_acts)

            c_rate     = llm.count("C") / len(llm)
            c_rates[w] = c_rate

            opp_c_idx = [i for i, a in enumerate(opp_acts) if a == "C"]
            opp_d_idx = [i for i, a in enumerate(opp_acts) if a == "D"]
            p_c_opp_c = (sum(llm[i] == "C" for i in opp_c_idx) / len(opp_c_idx)
                         if opp_c_idx else float("nan"))
            p_c_opp_d = (sum(llm[i] == "C" for i in opp_d_idx) / len(opp_d_idx)
                         if opp_d_idx else float("nan"))

            hc       = data[opp][w]["head_contributions"]     # (T, L, H)
            mean_abs = np.abs(hc).mean(axis=0)
            top_flat = np.argsort(mean_abs.ravel())[::-1][:3]
            top_str  = ", ".join(
                f"L{i // mean_abs.shape[1]}H{i % mean_abs.shape[1]}"
                f"({hc.mean(axis=0).ravel()[i]:+.2f})"
                for i in top_flat
            )

            print(
                f"  w={w}  C-rate={c_rate:.2f}"
                f"  P(C|opp_C)={p_c_opp_c:.2f}  P(C|opp_D)={p_c_opp_d:.2f}"
                f"  ld={ld.mean():+.2f}±{ld.std():.2f}"
                f"  llm_pay={llm_pay.mean():.2f}  opp_pay={opp_pay.mean():.2f}"
                f"\n         top heads: {top_str}"
            )

        if len(avail) >= 2:
            delta = c_rates[avail[-1]] - c_rates[avail[0]]
            print(f"  Δ C-rate (w={avail[0]}→w={avail[-1]}): {delta:+.2f}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_data()
    plot_logit_diff(data)
    plot_logit_lens_profile(data)
    plot_head_contributions(data)
    plot_top_heads_over_rounds(data)
    plot_head_condition_decomposition(data)
    plot_lagged_corr(data)
    print_action_summary(data)
