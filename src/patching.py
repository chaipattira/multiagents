from __future__ import annotations

import torch as t
from tqdm import tqdm
from transformer_lens import utilities as utils


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def logit_diff(logits: t.Tensor, tok_d: int, tok_c: int, pos: int = -1) -> float:
    """logit(D) − logit(C) at pos. Positive = model prefers defection."""
    return (logits[0, pos, tok_d] - logits[0, pos, tok_c]).float().item()


def normalised_recovery(patched_ld: float, corrupt_ld: float, clean_ld: float) -> float:
    """
    How much of the clean signal was recovered?
    0 = still at corrupt baseline, 1 = fully restored to clean.

    For a noising degradation score, call with clean/corrupt swapped:
        normalised_recovery(patched_ld, clean_ld, corrupt_ld)
    which gives 0 = no degradation, 1 = fully degraded.
    """
    return (patched_ld - corrupt_ld) / (clean_ld - corrupt_ld)


# ---------------------------------------------------------------------------
# Low-level hook functions
# ---------------------------------------------------------------------------


def _patch_resid_post(
    act: t.Tensor, hook, pos: int, source_cache
) -> t.Tensor:
    """
    Replace act[0, pos, :] with the cached resid_post at the same layer and pos.
    No-op if pos is out of bounds in the source cache (safe for mismatched seq_lens).
    """
    src = source_cache[utils.get_act_name("resid_post", hook.layer())]
    if pos >= src.shape[1]:
        return act
    act = act.clone()
    act[0, pos, :] = src[0, pos, :]
    return act


def _patch_head_result(
    act: t.Tensor, hook, pos: int, head: int, source_cache
) -> t.Tensor:
    """
    Replace act[0, pos, head, :] with the cached head result at the same layer.
    No-op if pos is out of bounds in the source cache.
    """
    src = source_cache[utils.get_act_name("result", hook.layer())]
    if pos >= src.shape[1]:
        return act
    act = act.clone()
    act[0, pos, head, :] = src[0, pos, head, :]
    return act


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------


def sweep_resid_full(
    model,
    run_tokens: t.Tensor,
    source_cache,
    baseline_ld: float,
    reference_ld: float,
    tok_d: int,
    tok_c: int,
    *,
    show_progress: bool = True,
) -> t.Tensor:
    """
    Patch source_cache activations into run_tokens at every (layer, pos).
    Returns (n_layers, seq_len) normalised scores:
        0 = baseline (no change), 1 = fully at reference.

    For noising: run_tokens=clean, source_cache=corrupt_cache,
                 baseline_ld=clean_ld, reference_ld=corrupt_ld.
    For denoising: run_tokens=corrupt, source_cache=clean_cache,
                   baseline_ld=corrupt_ld, reference_ld=clean_ld.
    """
    n_layers = model.cfg.n_layers
    seq_len  = run_tokens.shape[1]
    scores   = t.zeros(n_layers, seq_len)

    for layer in tqdm(range(n_layers), desc="Layers", disable=not show_progress):
        hook_name = utils.get_act_name("resid_post", layer)
        for pos in range(seq_len):
            def hook_fn(act, hook, _pos=pos, _src=source_cache):
                return _patch_resid_post(act, hook, _pos, _src)

            with t.no_grad():
                patched_ld = logit_diff(
                    model.run_with_hooks(run_tokens, fwd_hooks=[(hook_name, hook_fn)]),
                    tok_d, tok_c,
                )
            scores[layer, pos] = normalised_recovery(patched_ld, baseline_ld, reference_ld)

    return scores


def sweep_heads_at_pos(
    model,
    run_tokens: t.Tensor,
    source_cache,
    patch_pos: int,
    baseline_ld: float,
    reference_ld: float,
    tok_d: int,
    tok_c: int,
    *,
    show_progress: bool = True,
    desc: str = "",
) -> t.Tensor:
    """
    Patch source_cache head results into run_tokens at a single token position,
    sweeping all (layer, head) pairs.
    Returns (n_layers, n_heads) normalised scores.

    For denoising: run_tokens=corrupt, source_cache=clean_cache,
                   baseline_ld=corrupt_ld, reference_ld=clean_ld.
    """
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    out = t.zeros(n_layers, n_heads)

    for layer in tqdm(range(n_layers), desc=desc or f"Heads pos={patch_pos}", disable=not show_progress):
        hook_name = utils.get_act_name("result", layer)
        ca = source_cache[hook_name]
        for head in range(n_heads):
            def hook_fn(act, hook, _h=head, _a=ca, _p=patch_pos):
                act = act.clone()
                act[0, _p, _h, :] = _a[0, _p, _h, :]
                return act
            with t.no_grad():
                patched_ld = logit_diff(
                    model.run_with_hooks(run_tokens, fwd_hooks=[(hook_name, hook_fn)]),
                    tok_d, tok_c,
                )
            out[layer, head] = normalised_recovery(patched_ld, baseline_ld, reference_ld)

    return out


def sweep_mlp_at_pos(
    model,
    run_tokens: t.Tensor,
    source_cache,
    pos: int,
    baseline_ld: float,
    reference_ld: float,
    tok_d: int,
    tok_c: int,
) -> t.Tensor:
    """
    Patch source_cache mlp_out into run_tokens at a single token position across all layers.
    Returns (n_layers,) normalised scores.
    """
    n_layers = model.cfg.n_layers
    scores   = t.zeros(n_layers)

    for layer in range(n_layers):
        hook_name = utils.get_act_name("mlp_out", layer)
        if hook_name not in source_cache:
            continue
        src = source_cache[hook_name][0, pos, :].clone()

        def hook_fn(act, hook, _src=src, _pos=pos):
            act = act.clone()
            act[0, _pos, :] = _src
            return act

        with t.no_grad():
            patched_ld = logit_diff(
                model.run_with_hooks(run_tokens, fwd_hooks=[(hook_name, hook_fn)]),
                tok_d, tok_c,
            )
        scores[layer] = normalised_recovery(patched_ld, baseline_ld, reference_ld)

    return scores


def patch_resid_at_positions(
    model,
    run_tokens: t.Tensor,
    source_cache,
    layer: int,
    positions: list[int],
) -> t.Tensor:
    """
    Patch source_cache resid_post at ONE layer into run_tokens at a SET of
    positions (e.g. "every token in the opponent's history lines"), leaving
    all other positions -- including the final decision token -- untouched
    and computed fresh by the model's own forward pass.

    Returns the patched logits (caller applies logit_diff / normalised_recovery).
    Unlike sweep_resid_at_pos (single position, all layers), this is single
    layer, many positions -- used to test whether a *content region* read
    naturally by later layers carries a causal effect, as opposed to
    directly overwriting the position whose logits we read out.
    """
    hook_name = utils.get_act_name("resid_post", layer)
    src = source_cache[hook_name]

    def hook_fn(act, hook, _positions=positions, _src=src):
        act = act.clone()
        for p in _positions:
            if p < _src.shape[1]:
                act[0, p, :] = _src[0, p, :]
        return act

    with t.no_grad():
        return model.run_with_hooks(run_tokens, fwd_hooks=[(hook_name, hook_fn)])


def patch_component_at_positions(
    model,
    run_tokens: t.Tensor,
    source_cache,
    layer: int,
    positions: list[int],
    *,
    component: str = "resid_post",
) -> t.Tensor:
    """
    Like patch_resid_at_positions, but for any single-tensor-per-layer
    component ("resid_post", "attn_out", "mlp_out", ...). Used to decompose
    a resid_post-level content-region patch into its attention vs. MLP
    contributions -- e.g. is a construction step attention-mediated or
    MLP-mediated at a given (layer, content-region) combination.
    """
    hook_name = utils.get_act_name(component, layer)
    src = source_cache[hook_name]

    def hook_fn(act, hook, _positions=positions, _src=src):
        act = act.clone()
        for p in _positions:
            if p < _src.shape[1]:
                act[0, p, :] = _src[0, p, :]
        return act

    with t.no_grad():
        return model.run_with_hooks(run_tokens, fwd_hooks=[(hook_name, hook_fn)])


def sweep_resid_at_pos(
    model,
    run_tokens: t.Tensor,
    source_cache,
    pos: int,
    baseline_ld: float,
    reference_ld: float,
    tok_d: int,
    tok_c: int,
) -> t.Tensor:
    """
    Patch source_cache into run_tokens at a single token position across all layers.
    Returns (n_layers,) normalised scores.

    Useful for per-round denoising: call once per corrupt-k with opp_pos[k].
    """
    n_layers = model.cfg.n_layers
    scores   = t.zeros(n_layers)

    for layer in range(n_layers):
        hook_name = utils.get_act_name("resid_post", layer)

        def hook_fn(act, hook, _pos=pos, _src=source_cache):
            return _patch_resid_post(act, hook, _pos, _src)

        with t.no_grad():
            patched_ld = logit_diff(
                model.run_with_hooks(run_tokens, fwd_hooks=[(hook_name, hook_fn)]),
                tok_d, tok_c,
            )
        scores[layer] = normalised_recovery(patched_ld, baseline_ld, reference_ld)

    return scores
