from __future__ import annotations

import json
from pathlib import Path

import torch as t

from src.interp import get_decision_tokens, get_logit_dir, load_model, run_dla
from src.ipd_game import Action, OPPONENTS, new_game, record_round
from src.ipd_prompts import build_prompt

WINDOWS = [2, 4, 8]
N_ROUNDS = 100
OUT_DIR = Path(__file__).parent.parent / "results" / "ipd_sweep"


def run_sweep(model) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok_c, tok_d = get_decision_tokens(model)
    logit_dir    = get_logit_dir(model, tok_d, tok_c)

    for opp_name, opp_fn in OPPONENTS.items():
        for window_n in WINDOWS:
            game = new_game()
            records = []

            for rnd in range(N_ROUNDS):
                prompt = build_prompt(game, window_n, N_ROUNDS)
                dla    = run_dla(prompt, model, logit_dir)

                logit_lens        = dla["logit_lens"]           # [n_layers]
                head_contributions = dla["head_contributions"]  # [n_layers, n_heads]
                logit_diff        = logit_lens[-1].item()
                action: Action    = "D" if logit_diff > 0 else "C"
                opp_action        = opp_fn(game)
                record_round(game, action, opp_action)

                records.append({
                    "round":             rnd + 1,
                    "llm_action":        action,
                    "opp_action":        opp_action,
                    "logit_diff":        logit_diff,
                    "logit_lens":        logit_lens.tolist(),
                    "head_contributions": head_contributions.tolist(),
                })

            out_path = OUT_DIR / f"{opp_name}_w{window_n}.json"
            out_path.write_text(json.dumps(records, indent=2))
            print(f"Saved {opp_name} w={window_n} → {out_path}")


if __name__ == "__main__":
    model = load_model()
    model.eval()
    t.set_grad_enabled(False)
    run_sweep(model)
