from __future__ import annotations

import json
from pathlib import Path

import torch as t
import torch.nn.functional as F

from src.interp import get_decision_tokens, load_model
from src.ipd_game import Action, OPPONENTS, new_game, record_round
from src.ipd_prompts import build_prompt

TEMPERATURE = 0.7
N_RUNS      = 30
N_ROUNDS    = 100
WINDOWS     = [2, 4, 8]
OUT_DIR     = Path(__file__).parent.parent / "results" / "ipd_behavioral"


def sample_action(
    model, tokens: t.Tensor, tok_c: int, tok_d: int
) -> Action:
    logits = model(tokens)[0, -1, :]
    cd_logits = logits[[tok_c, tok_d]].float()
    probs = F.softmax(cd_logits / TEMPERATURE, dim=-1)
    idx = t.multinomial(probs, num_samples=1).item()
    return "C" if idx == 0 else "D"


def run_behavioral(model) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok_c, tok_d = get_decision_tokens(model)

    total = len(OPPONENTS) * len(WINDOWS) * N_RUNS
    done  = 0

    for opp_name, opp_fn in OPPONENTS.items():
        for window in WINDOWS:
            all_runs: list[list[dict]] = []

            for run in range(N_RUNS):
                game    = new_game()
                records: list[dict] = []

                for rnd in range(N_ROUNDS):
                    prompt     = build_prompt(game, window, N_ROUNDS)
                    tokens     = model.to_tokens(prompt)
                    action     = sample_action(model, tokens, tok_c, tok_d)
                    opp_action = opp_fn(game)
                    record_round(game, action, opp_action)
                    records.append({
                        "round":      rnd + 1,
                        "llm_action": action,
                        "opp_action": opp_action,
                    })

                all_runs.append(records)
                done += 1
                print(
                    f"[{done}/{total}]  {opp_name} w={window} run {run+1}/{N_RUNS}",
                    flush=True,
                )

            out_path = OUT_DIR / f"{opp_name}_w{window}.json"
            out_path.write_text(json.dumps(all_runs, indent=2))
            print(f"  → saved {out_path}")


if __name__ == "__main__":
    model = load_model()
    model.eval()
    t.set_grad_enabled(False)
    run_behavioral(model)
