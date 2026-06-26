from __future__ import annotations

from .ipd_game import Action

_PAYOFF_BLOCK = (
    "Payoffs:\n"
    "If you play D and opponent plays D, then you get 1 pt and opponent gets 1 pt.\n"
    "If you play D and opponent plays C, then you get 5 pts and opponent gets 0 pts.\n"
    "If you play C and opponent plays D, then you get 0 pts and opponent gets 5 pts.\n"
    "If you play C and opponent plays C, then you get 3 pts and opponent gets 3 pts."
)


def build_stripped_prompt(game: dict, window_n: int, n_rounds: int = 100) -> str:
    n_played = len(game["llm_actions"])
    current_round = n_played + 1

    s = slice(-window_n, None) if window_n > 0 else slice(0, 0)
    llm_hist = game["llm_actions"][s]
    opp_hist = game["opp_actions"][s]

    if llm_hist:
        start = n_played - len(llm_hist) + 1
        history_block = "\n".join(
            f"Round {start + i}: you {la}, opponent {oa}."
            for i, (la, oa) in enumerate(zip(llm_hist, opp_hist))
        )
    else:
        history_block = "(none yet)"

    user_msg = (
        "You are playing a multi-round game against an opponent.\n"
        "At each turn, you and your opponent simultaneously choose one of: {C, D}.\n\n"
        f"{_PAYOFF_BLOCK}\n\n"
        f"You will play a total of {n_rounds} rounds. "
        "Remember: your objective is to get the highest possible number of points in the long run.\n\n"
        f"History of the last {window_n} rounds:\n{history_block}\n\n"
        f"What do you play in round {current_round}? "
        "State your decision as exactly: Action: C  or  Action: D"
    )
    return f"<start_of_turn>user\n{user_msg}\n<end_of_turn>\n<start_of_turn>model\nAction: \n"
