from __future__ import annotations

import random
from typing import Literal

Action = Literal["C", "D"]

PAYOFFS: dict[tuple[Action, Action], tuple[int, int]] = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}


def new_game() -> dict:
    return {"llm_actions": [], "opp_actions": [], "llm_scores": [], "opp_scores": []}


def record_round(game: dict, llm_act: Action, opp_act: Action) -> None:
    p_llm, p_opp = PAYOFFS[(llm_act, opp_act)]
    game["llm_actions"].append(llm_act)
    game["opp_actions"].append(opp_act)
    game["llm_scores"].append(p_llm)
    game["opp_scores"].append(p_opp)


def opp_allc(game: dict) -> Action:
    return "C"


def opp_alld(game: dict) -> Action:
    return "D"


def opp_tft(game: dict) -> Action:
    if not game["llm_actions"]:
        return "C"
    return game["llm_actions"][-1]


def opp_tf2t(game: dict) -> Action:
    acts = game["llm_actions"]
    if len(acts) < 2:
        return "C"
    return "D" if acts[-1] == "D" and acts[-2] == "D" else "C"


def opp_random(game: dict) -> Action:
    return random.choice(["C", "D"])


OPPONENTS: dict[str, callable] = {
    "ALLC":   opp_allc,
    "ALLD":   opp_alld,
    "TFT":    opp_tft,
    "TF2T":   opp_tf2t,
    "RANDOM": opp_random,
}
