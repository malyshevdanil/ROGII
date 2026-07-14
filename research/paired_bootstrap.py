"""Reusable paired well-bootstrap validator, adopted directly from the top-5 competitor's (Tucker
Arrants, 2026-07-14 writeup) methodology: fixed folds are too noisy on this heavy-tailed per-well metric
(a handful of wells dominate SSE), so a single 4-5 fold cross-fit can flip sign just from which outlier
wells land in which fold (exactly what happened with the trellis fold-3 result today). A paired bootstrap
resamples WELLS (not rows) with replacement many times and reports the distribution of the gain, giving a
real confidence interval instead of a single noisy point estimate.
"""
import numpy as np

def paired_bootstrap_gain(per_well_sq_base, per_well_sq_cand, wids, n_boot=2000, seed=0):
    """per_well_sq_base/cand: dict wid -> array of per-row squared errors for that well.
    Returns (mean_gain, ci_lo, ci_hi, frac_positive) on POOLED RMSE gain (base_rmse - cand_rmse) computed
    per bootstrap resample of wells (with replacement), matching the competition's pooled-row metric."""
    rng = np.random.default_rng(seed)
    wids = np.array(wids)
    n = len(wids)
    gains = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(wids, size=n, replace=True)
        sq_base = np.concatenate([per_well_sq_base[w] for w in sample])
        sq_cand = np.concatenate([per_well_sq_cand[w] for w in sample])
        rmse_base = np.sqrt(sq_base.mean())
        rmse_cand = np.sqrt(sq_cand.mean())
        gains[b] = rmse_base - rmse_cand
    mean_gain = float(gains.mean())
    ci_lo, ci_hi = float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))
    frac_positive = float((gains > 0).mean())
    return mean_gain, ci_lo, ci_hi, frac_positive

def shuffled_control(per_well_sq_base, per_well_pred_cand, per_well_true, wids, blend_fn, n_shuffle=200, seed=1):
    """No-op-style control: shuffle the CANDIDATE's per-well predictions across wells (breaking any real
    well-specific signal) and see how often the shuffled version STILL looks like a 'gain' -- if shuffled
    inputs regularly produce gains of similar magnitude to the real candidate, the real result is not
    trustworthy (matches Tucker's duplicate/no-op control)."""
    rng = np.random.default_rng(seed)
    wids = np.array(wids)
    real_sq = {w: (blend_fn(w, per_well_pred_cand[w]) - per_well_true[w]) ** 2 for w in wids}
    real_rmse = np.sqrt(np.concatenate([real_sq[w] for w in wids]).mean())
    base_rmse = np.sqrt(np.concatenate([per_well_sq_base[w] for w in wids]).mean())
    real_gain = base_rmse - real_rmse

    shuf_gains = np.empty(n_shuffle)
    for s in range(n_shuffle):
        perm = rng.permutation(wids)
        shuf_sq = {w: (blend_fn(w, per_well_pred_cand[wp]) - per_well_true[w]) ** 2 for w, wp in zip(wids, perm)}
        shuf_rmse = np.sqrt(np.concatenate([shuf_sq[w] for w in wids]).mean())
        shuf_gains[s] = base_rmse - shuf_rmse
    return real_gain, shuf_gains
