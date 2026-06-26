from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer
from transformer_lens import utilities as utils

from .ipd_prompts import build_prompt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID       = "google/gemma-2-2b-it"
OPP_ACTION_POS = 101  # token position of opponent's action in the 1-round contrast pair

_ACTION_PREFIX = "<start_of_turn>model\nAction: \n"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: t.device | None = None) -> HookedTransformer:
    if device is None:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(
        MODEL_ID,
        dtype=t.bfloat16,
        fold_ln=True,          # folds RMSNorm weights → resid @ W_U is pre-softcap logit
        device=device,
        center_unembed=False,  # Gemma-2 logit softcap is not shift-invariant
    )
    model.cfg.use_attn_result = True
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Token / direction helpers
# ---------------------------------------------------------------------------

def get_decision_tokens(model: HookedTransformer) -> tuple[int, int]:
    """Returns (TOK_C, TOK_D) token IDs for the decision position."""
    tok_c = model.to_tokens(_ACTION_PREFIX + "C", prepend_bos=False)[0, -1].item()
    tok_d = model.to_tokens(_ACTION_PREFIX + "D", prepend_bos=False)[0, -1].item()
    return tok_c, tok_d

def get_logit_dir(model: HookedTransformer, tok_d: int, tok_c: int) -> t.Tensor:
    """D − C unembedding direction. Positive projection = toward defection."""
    return model.W_U[:, tok_d].float() - model.W_U[:, tok_c].float()

# ---------------------------------------------------------------------------
# Contrast pair
# ---------------------------------------------------------------------------

def make_contrast_pair(model: HookedTransformer) -> dict:
    """
    Build and validate the standard 1-round contrast pair.

    Clean   = opp cooperated (opp_R1=C) → model defects.
    Corrupt = opp defected   (opp_R1=D) → model cooperates.

    Asserts both prompts are the same length and differ in exactly 1 token
    at OPP_ACTION_POS.

    Returns a dict with keys:
        clean_prompt, corrupt_prompt,
        clean_tokens, corrupt_tokens,
        tok_c, tok_d, logit_dir,
        opp_action_pos, seq_len.
    """
    tok_c, tok_d   = get_decision_tokens(model)
    _clean_game    = {"llm_actions": ["C"], "opp_actions": ["C"], "llm_scores": [], "opp_scores": []}
    _corrupt_game  = {"llm_actions": ["C"], "opp_actions": ["D"], "llm_scores": [], "opp_scores": []}
    clean_prompt   = build_prompt(_clean_game, window_n=1)
    corrupt_prompt = build_prompt(_corrupt_game, window_n=1)
    clean_tokens   = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)

    assert clean_tokens.shape == corrupt_tokens.shape, "Prompt length mismatch"
    diffs = (clean_tokens != corrupt_tokens).nonzero(as_tuple=False)
    assert len(diffs) == 1, f"Expected 1 token diff, got {len(diffs)}"
    pos = diffs[0, 1].item()
    assert pos == OPP_ACTION_POS, f"Diff at pos {pos}, expected {OPP_ACTION_POS}"

    return dict(
        clean_prompt   = clean_prompt,
        corrupt_prompt = corrupt_prompt,
        clean_tokens   = clean_tokens,
        corrupt_tokens = corrupt_tokens,
        tok_c          = tok_c,
        tok_d          = tok_d,
        logit_dir      = get_logit_dir(model, tok_d, tok_c),
        opp_action_pos = pos,
        seq_len        = clean_tokens.shape[1],
    )

# ---------------------------------------------------------------------------
# DLA
# ---------------------------------------------------------------------------

def run_dla(prompt: str, model: HookedTransformer, logit_dir: t.Tensor) -> dict:
    """
    Single-prompt DLA at the final token position.

    Uses recompute_ln=True throughout to work correctly with fold_ln=True
    (Gemma-2's final LN is folded into W_U, so hook_scale is not populated).

    Returns:
        logit_lens:          [n_layers]      accumulated residual projected onto logit_dir
        layer_contributions: [n_components]  per sublayer (embed / attn / mlp)
        comp_labels:         list[str]       labels for layer_contributions
        head_contributions:  [n_layers, n_heads]  per-head (no LN, standard DLA approximation)
    """
    n_layers = model.cfg.n_layers
    tokens   = model.to_tokens(prompt)

    _needed = (
        {utils.get_act_name("resid_post", l) for l in range(n_layers)}
        | {utils.get_act_name("result",   l) for l in range(n_layers)}
        | {utils.get_act_name("attn_out", l) for l in range(n_layers)}
        | {utils.get_act_name("mlp_out",  l) for l in range(n_layers)}
        | {"hook_embed"}
    )
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in _needed)

    # Logit lens: accumulated residual at each layer, LN-scaled
    resid_stack  = t.stack([cache["resid_post", l][0, -1].float() for l in range(n_layers)])
    scaled_resid = cache.apply_ln_to_stack(
        resid_stack.unsqueeze(1), layer=-1, pos_slice=-1, recompute_ln=True
    ).squeeze(1).float()
    logit_lens = scaled_resid @ logit_dir

    # Per-component contributions, LN-scaled
    per_comp, comp_labels = cache.decompose_resid(
        layer=-1, pos_slice=-1, return_labels=True
    )
    scaled_comp         = cache.apply_ln_to_stack(
        per_comp.float(), layer=-1, pos_slice=-1, recompute_ln=True
    ).float()
    layer_contributions = scaled_comp[:, 0, :] @ logit_dir

    # Per-head: no LN (standard DLA approximation — heads are additive deltas)
    head_contributions = t.stack([
        cache["result", l][0, -1, :, :].float() @ logit_dir
        for l in range(n_layers)
    ])  # [n_layers, n_heads]

    return dict(
        logit_lens          = logit_lens,
        layer_contributions = layer_contributions,
        comp_labels         = comp_labels,
        head_contributions  = head_contributions,
    )
