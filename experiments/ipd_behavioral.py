from __future__ import annotations

import json
from pathlib import Path

import torch as t

from src.interp import get_decision_tokens, load_model
from src.ipd_game import Action, OPPONENTS, new_game, record_round
from src.ipd_prompts import build_stripped_prompt

WINDOWS = [2, 4, 8]
N_ROUNDS = 100
OUT_DIR = Path(__file__).parent.parent / "results" / "ipd_sweep"


def run_sweep(model) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok_c, tok_d = get_decision_tokens(model)

    for opp_name, opp_fn in OPPONENTS.items():
        for window_n in WINDOWS:
            game = new_game()
            records = []

            for rnd in range(N_ROUNDS):
                prompt = build_stripped_prompt(game, window_n, N_ROUNDS)
                tokens = model.to_tokens(prompt)
                logits = model(tokens)[0, -1, :]

                logit_c = logits[tok_c].float().item()
                logit_d = logits[tok_d].float().item()
                logit_diff = logit_d - logit_c
                action: Action = "C" if logit_c >= logit_d else "D"
                opp_action = opp_fn(game)
                record_round(game, action, opp_action)

                records.append({
                    "round":      rnd + 1,
                    "llm_action": action,
                    "opp_action": opp_action,
                    "logit_c":    logit_c,
                    "logit_d":    logit_d,
                    "logit_diff": logit_diff,
                })

            out_path = OUT_DIR / f"{opp_name}_w{window_n}.json"
            out_path.write_text(json.dumps(records, indent=2))
            print(f"Saved {opp_name} w={window_n} → {out_path}")


if __name__ == "__main__":
    model = load_model()
    model.eval()
    t.set_grad_enabled(False)
    run_sweep(model)
