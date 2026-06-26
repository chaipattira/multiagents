import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.interp import MPL_RC

mpl.rcParams.update(MPL_RC)

RESULTS_DIR = Path("results/ipd_sweep")
OPPONENTS = ["ALLC", "ALLD", "TFT", "TF2T", "RANDOM"]
WINDOWS = [2, 4, 8]
COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]  # window 2, 4, 8

OPP_LABELS = {
    "ALLC":   "Always Cooperate",
    "ALLD":   "Always Defect",
    "TFT":    "Tit-for-Tat",
    "TF2T":   "Tit-for-Two-Tats",
    "RANDOM": "Random",
}

# Standard IPD payoff matrix (R=3, P=1, T=5, S=0)
_LLM_PAYOFF = {("C", "C"): 3, ("C", "D"): 0, ("D", "C"): 5, ("D", "D"): 1}
_OPP_PAYOFF = {("C", "C"): 3, ("C", "D"): 5, ("D", "C"): 0, ("D", "D"): 1}
_SOCIAL     = {("C", "C"): 6, ("C", "D"): 5, ("D", "C"): 5, ("D", "D"): 2}


def softmax_entropy(logit_diff: np.ndarray) -> np.ndarray:
    p_d = 1 / (1 + np.exp(-logit_diff))
    p_c = 1 - p_d
    return -(p_c * np.log(p_c) + p_d * np.log(p_d))


def per_player_payoff(llm_acts: list[str], opp_acts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pairs = list(zip(llm_acts, opp_acts))
    return (
        np.array([_LLM_PAYOFF[p] for p in pairs], dtype=float),
        np.array([_OPP_PAYOFF[p] for p in pairs], dtype=float),
    )


def social_utility(llm_acts: list[str], opp_acts: list[str]) -> np.ndarray:
    return np.array([_SOCIAL[(l, o)] for l, o in zip(llm_acts, opp_acts)], dtype=float)


def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(arr, np.ones(w) / w, mode="same")


def load_data() -> dict:
    data = {}
    for opp in OPPONENTS:
        data[opp] = {}
        for w in WINDOWS:
            path = RESULTS_DIR / f"{opp}_w{w}.json"
            records = json.loads(path.read_text())
            data[opp][w] = {
                "rounds":     [r["round"]      for r in records],
                "logit_diff": [r["logit_diff"] for r in records],
                "llm_action": [r["llm_action"] for r in records],
                "opp_action": [r["opp_action"] for r in records],
            }
    print("Loaded", sum(len(data[o]) for o in OPPONENTS), "combos")
    return data


def plot_logit_diff(data: dict) -> None:
    fig, axes = plt.subplots(len(OPPONENTS), 1, figsize=(14, 4 * len(OPPONENTS)), sharex=True)

    for ax, opp in zip(axes, OPPONENTS):
        all_lo, all_hi = [], []
        for w in WINDOWS:
            ld = np.array(data[opp][w]["logit_diff"])
            H  = softmax_entropy(ld)
            all_lo.append((ld - H).min())
            all_hi.append((ld + H).max())
        ylo, yhi = min(all_lo), max(all_hi)
        pad = max((yhi - ylo) * 0.12, 0.5)
        ylo -= pad; yhi += pad

        ax.axhspan(0,   yhi, color="#ffcccc", alpha=0.4, zorder=0)
        ax.axhspan(ylo, 0,   color="#cce5ff", alpha=0.4, zorder=0)

        for color, w in zip(COLORS, WINDOWS):
            d = data[opp][w]
            rounds = np.array(d["rounds"])
            ld     = np.array(d["logit_diff"])
            H      = softmax_entropy(ld)

            ax.plot(rounds, ld, color=color, lw=1.8, label=f"w={w}", zorder=3)
            ax.fill_between(rounds, ld - H, ld + H, color=color, alpha=0.18, zorder=2)

        ax.axhline(0, color="gray", lw=1.0, ls="--", alpha=0.8, zorder=1)
        ax.set_ylim(ylo, yhi)
        ax.set_title(OPP_LABELS[opp], fontsize=15, loc="left", pad=4, fontweight="bold")
        ax.grid(alpha=0.25, zorder=0)
        ax.tick_params(labelsize=13)

    axes[0].legend(title="History window", fontsize=13, title_fontsize=13, loc="upper right")
    axes[-1].set_xlabel("Round", fontsize=15)

    fig.text(0.04, 0.5, "Logit diff (D − C)", va="center", rotation="vertical", fontsize=15)
    fig.suptitle(
        "Gemma-2-2b-it in Iterated Prisoner's Dilemma",
        fontsize=18, y=1.01, fontweight="bold"
    )
    plt.tight_layout(rect=[0.06, 0, 1, 1])

    out = RESULTS_DIR / "logit_diff_sweep.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


def _lagged_corr(llm: list[str], opp: list[str], max_lag: int) -> np.ndarray:
    """corr(opp_action[t-lag], llm_action[t]) for lag = 1..max_lag. Returns nan if zero variance."""
    llm_bin = np.array([1.0 if a == "C" else 0.0 for a in llm])
    opp_bin = np.array([1.0 if a == "C" else 0.0 for a in opp])
    corrs = []
    for lag in range(1, max_lag + 1):
        x, y = opp_bin[:-lag], llm_bin[lag:]
        corrs.append(np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan)
    return np.array(corrs)


_VARIABLE_OPPS = ["TFT", "TF2T", "RANDOM"]  # ALLC/ALLD have zero variance, always nan


def plot_lagged_corr(data: dict) -> None:
    max_lag = max(WINDOWS)
    opps    = _VARIABLE_OPPS
    cmap = mpl.cm.RdBu_r.copy()
    cmap.set_bad("#cccccc")  # gray for out-of-window cells

    fig, axes = plt.subplots(len(opps), 1, figsize=(12, 3.0 * len(opps)),
                             constrained_layout=True)

    for ax, opp in zip(axes, opps):
        # Build (n_windows × max_lag) matrix; fill only in-window cells
        mat = np.full((len(WINDOWS), max_lag), np.nan)
        for ri, w in enumerate(WINDOWS):
            d = data[opp][w]
            mat[ri, :w] = _lagged_corr(d["llm_action"], d["opp_action"], w)

        im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        # Annotate every cell
        for ri, w in enumerate(WINDOWS):
            for ci in range(max_lag):
                if ci < w:
                    v = mat[ri, ci]
                    txt   = f"{v:.2f}" if not np.isnan(v) else "n/a"
                    color = "white" if (not np.isnan(v) and abs(v) > 0.55) else "black"
                else:
                    txt, color = "–", "#888888"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

        ax.set_yticks(range(len(WINDOWS)))
        ax.set_yticklabels([f"w={w}" for w in WINDOWS], fontsize=13)
        ax.set_xticks(range(max_lag))
        ax.set_xticklabels([str(i + 1) for i in range(max_lag)], fontsize=12)
        ax.set_xticks(np.arange(-0.5, max_lag), minor=True)
        ax.set_yticks(np.arange(-0.5, len(WINDOWS)), minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="minor", size=0)
        ax.set_title(OPP_LABELS[opp], fontsize=14, loc="left", pad=4, fontweight="bold")

    axes[-1].set_xlabel("Lag (rounds)", fontsize=14)
    fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02,
                 label="corr(opp[t−lag], llm[t])")
    fig.suptitle(
        "Gemma-2-2b-it — Lagged Correlation: Does History Window Depth Matter?\n"
        "(gray = lag outside given window)",
        fontsize=15, fontweight="bold", y=1.01,
    )
    out = RESULTS_DIR / "lagged_corr_sweep.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


def print_action_summary(data: dict) -> None:
    for opp in OPPONENTS:
        print(f"\n{'='*65}")
        print(f"Opponent: {OPP_LABELS[opp]}")
        print(f"{'='*65}")

        c_rates: dict[int, float] = {}
        for w in WINDOWS:
            d        = data[opp][w]
            llm      = d["llm_action"]
            opp_acts = d["opp_action"]
            ld       = np.array(d["logit_diff"])
            H        = softmax_entropy(ld)
            llm_pay, opp_pay = per_player_payoff(llm, opp_acts)

            c_rate      = llm.count("C") / len(llm)
            c_rates[w]  = c_rate

            # P(C | opp played C/D)
            opp_c_idx = [i for i, a in enumerate(opp_acts) if a == "C"]
            opp_d_idx = [i for i, a in enumerate(opp_acts) if a == "D"]
            p_c_opp_c = (sum(llm[i] == "C" for i in opp_c_idx) / len(opp_c_idx)
                         if opp_c_idx else float("nan"))
            p_c_opp_d = (sum(llm[i] == "C" for i in opp_d_idx) / len(opp_d_idx)
                         if opp_d_idx else float("nan"))

            # Correlation opp_action[t] → llm_action[t+1]
            opp_bin  = np.array([1 if a == "C" else 0 for a in opp_acts[:-1]])
            llm_next = np.array([1 if a == "C" else 0 for a in llm[1:]])
            corr = (np.corrcoef(opp_bin, llm_next)[0, 1]
                    if len(opp_bin) > 1 else float("nan"))

            print(
                f"  w={w}  C-rate={c_rate:.2f}"
                f"  P(C|opp_C)={p_c_opp_c:.2f}  P(C|opp_D)={p_c_opp_d:.2f}"
                f"  corr(opp[t]→llm[t+1])={corr:+.2f}"
                f"  ld={ld.mean():+.2f}±{ld.std():.2f}"
                f"  H={H.mean():.3f}"
                f"  llm_pay={llm_pay.mean():.2f}  opp_pay={opp_pay.mean():.2f}"
            )

        delta = c_rates[WINDOWS[-1]] - c_rates[WINDOWS[0]]
        print(f"  Δ C-rate (w={WINDOWS[0]}→w={WINDOWS[-1]}): {delta:+.2f}")


if __name__ == "__main__":
    data = load_data()
    plot_logit_diff(data)
    plot_lagged_corr(data)
    print_action_summary(data)
