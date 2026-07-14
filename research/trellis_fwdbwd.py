"""Exact forward-backward (sum-product) trellis over a joint (position, dip) grid, replacing the
beam-search decoder's approximate pruning with exact marginal inference -- directly motivated by a
top-5 competitor's writeup (2026-07-14): their trellis scores 13.42 alone (worse than our flat=15.1)
but has only 0.488 error correlation with their main ensemble, and blending it gives a real 7.762->7.699
gain in 10/10 folds. Our own beam-HSMM (approximate beam search over the same category of state space)
showed ZERO blend value in every test this session -- the working hypothesis is that beam-search pruning
lost exactly the information that makes this category of model useful as a diverse ensemble member, and
exact inference (no pruning, full posterior) will behave differently.

Architecture: nodes spaced ~50ft in MD (like the competitor's). State per node = (pos, dip) on a dense
joint grid, pos tracked as TVT+Z (project convention, pos relative offset from a dead-reckoned anchor).
Emission at a node = sum of Gaussian log-lik over all eval rows falling in that node's window, each row's
implied position given by linear extrapolation from the node's own (pos, dip). Transition: pos' ~
N(pos + dip*dm, sigma_pos(node)) with sigma_pos GROWING with distance from the anchor (increasing datum
uncertainty, matching the competitor's prior); dip' ~ N(rho*dip, persist_resid_std) using our OWN fitted
persistence prior (stride3 priors, not copied from anyone). Standard forward-backward computes exact
posterior marginals; output is the posterior-mean position at each node, linearly interpolated back to
the original eval-row MD grid for scoring.
"""
import numpy as np, pandas as pd, glob, os, math
from numba import njit

TRAIN_DIR = 'd:/ROGII/data/train'
PF_GR_SIG_MIN = 10.; PF_GR_SIG_MAX = 60.; PF_GR_SIG_DEF = 30.

# priors fit in stride3_fit_priors.py (pos=TVT+Z convention, post landing-curve fix)
PERSIST_RHO = 0.4725; PERSIST_RESID_STD = 0.0325
DIP_STD = 0.0409

NODE_SPACING = 50.0
POS_HALF_RANGE = 80.0; POS_STEP = 2.5      # -> 65 pos bins
DIP_HALF_RANGE = 0.15; DIP_STEP = 0.015    # -> 21 dip bins
SIGMA_POS_BASE = 1.5; SIGMA_POS_GROWTH = 0.08  # ft, grows per node

@njit(cache=False)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1. - t) + grid[i + 1] * t

@njit(cache=False)
def _logsumexp_update(current, cand):
    if current < -1e17:
        return cand
    if cand < -1e17:
        return current
    if cand > current:
        return cand + math.log1p(math.exp(current - cand))
    else:
        return current + math.log1p(math.exp(cand - current))

@njit(cache=False)
def _trellis_fwdbwd(md_v, z_v, gr_v, gg, vmin, step, gs,
                     node_md, pos_grid, dip_grid, ref_pos, ir0,
                     rho, dip_resid, sigma_pos_base, sigma_pos_growth):
    """ref_pos[t] is a rolling dead-reckoned reference (ls + ir0*(node_md[t]-md0)); pos_grid values are
    OFFSETS from ref_pos[t] at each respective node, so sigma_pos can grow with distance without ever
    needing the absolute grid to widen (the reference itself already tracks the naive continuation)."""
    NP = len(pos_grid); ND = len(dip_grid)
    NS = NP * ND
    T = len(node_md)

    # Emission depends ONLY on the position offset (i), never on the candidate's own dip (j) --
    # using a candidate's own dip to extrapolate WHICH part of the typewell curve its member rows
    # get compared against would let large-|dip| states "scan" more of the (self-similar) typewell
    # within one window and win by aliasing, not genuine fit. Every row is scored against a single
    # smooth ir0-based reference trajectory (ref_pos_row) plus a constant offset pos_grid[i].
    emission_pos = np.zeros((T, NP))
    row_node = np.empty(len(md_v), dtype=np.int64)
    md0 = md_v[0]
    for r in range(len(md_v)):
        k = int((md_v[r] - node_md[0]) / (node_md[1] - node_md[0])) if T > 1 else 0
        if k < 0: k = 0
        if k >= T: k = T - 1
        row_node[r] = k

    # count valid (non-NaN) rows per node once, to AVERAGE (not sum) the per-row log-lik terms --
    # the GR-vs-typewell residual has an autocorrelation length of ~50 samples (same order as one
    # node's row count), so summing them as if independent makes the emission wildly overconfident.
    row_count = np.zeros(T)
    for r in range(len(md_v)):
        if not math.isnan(gr_v[r]):
            row_count[row_node[r]] += 1.0

    for i in range(NP):
        for r in range(len(md_v)):
            gv = gr_v[r]
            if math.isnan(gv): continue
            t = row_node[r]
            ref_pos_row = ref_pos[0] + ir0 * (md_v[r] - md0)
            pos_r = ref_pos_row + pos_grid[i]
            tvt_r = pos_r - z_v[r]
            eg = _interp1(gg, tvt_r, vmin, step)
            d = (gv - eg) / gs
            rc = row_count[t]
            w = 1.0 / max(1.0, rc/10.0) if rc > 0 else 0.0
            emission_pos[t, i] += -0.5 * d * d * w

    emission = np.zeros((T, NS))
    for t in range(T):
        for i in range(NP):
            ev = emission_pos[t, i]
            for j in range(ND):
                emission[t, i * ND + j] = ev

    alpha = np.full((T, NS), -1e18)
    beta = np.full((T, NS), -1e18)

    init_pos_std = sigma_pos_base
    for i in range(NP):
        for j in range(ND):
            s = i * ND + j
            dpos = pos_grid[i]; ddip = dip_grid[j] - ir0
            lp = -0.5 * (dpos / init_pos_std) ** 2 - 0.5 * (ddip / DIP_STD) ** 2
            alpha[0, s] = lp + emission[0, s]

    # forward pass
    for t in range(1, T):
        dm = node_md[t] - node_md[t - 1]
        sigma_pos = sigma_pos_base + sigma_pos_growth * t
        if sigma_pos > (POS_HALF_RANGE / 5.0): sigma_pos = POS_HALF_RANGE / 5.0
        ref_shift = ref_pos[t] - ref_pos[t - 1]  # == ir0*dm
        for i in range(NP):
            for j in range(ND):
                sp = i * ND + j
                a_prev = alpha[t - 1, sp]
                if a_prev < -1e17: continue
                # predicted offset in the NEW node's reference frame
                pred_pos = pos_grid[i] + dip_grid[j] * dm - ref_shift
                pred_dip = rho * dip_grid[j]
                for i2 in range(NP):
                    dpos = pos_grid[i2] - pred_pos
                    if abs(dpos) > 4.5 * sigma_pos: continue
                    lp_pos = -0.5 * (dpos / sigma_pos) ** 2
                    for j2 in range(ND):
                        ddip = dip_grid[j2] - pred_dip
                        if abs(ddip) > 4.5 * dip_resid: continue
                        lp_dip = -0.5 * (ddip / dip_resid) ** 2
                        s2 = i2 * ND + j2
                        cand = a_prev + lp_pos + lp_dip
                        alpha[t, s2] = _logsumexp_update(alpha[t, s2], cand)
        for s2 in range(NS):
            if alpha[t, s2] > -1e17:
                alpha[t, s2] += emission[t, s2]

    # backward pass
    for s in range(NS):
        beta[T - 1, s] = 0.0
    for t in range(T - 2, -1, -1):
        dm = node_md[t + 1] - node_md[t]
        sigma_pos = sigma_pos_base + sigma_pos_growth * (t + 1)
        if sigma_pos > (POS_HALF_RANGE / 5.0): sigma_pos = POS_HALF_RANGE / 5.0
        ref_shift = ref_pos[t + 1] - ref_pos[t]
        for i in range(NP):
            for j in range(ND):
                sp = i * ND + j
                pred_pos = pos_grid[i] + dip_grid[j] * dm - ref_shift
                pred_dip = rho * dip_grid[j]
                acc = -1e18
                for i2 in range(NP):
                    dpos = pos_grid[i2] - pred_pos
                    if abs(dpos) > 4.5 * sigma_pos: continue
                    lp_pos = -0.5 * (dpos / sigma_pos) ** 2
                    for j2 in range(ND):
                        ddip = dip_grid[j2] - pred_dip
                        if abs(ddip) > 4.5 * dip_resid: continue
                        lp_dip = -0.5 * (ddip / dip_resid) ** 2
                        s2 = i2 * ND + j2
                        eb = beta[t + 1, s2]
                        if eb < -1e17: continue
                        em2 = emission[t + 1, s2]
                        cand = lp_pos + lp_dip + em2 + eb
                        acc = _logsumexp_update(acc, cand)
                beta[t, sp] = acc

    # posterior mean position per node (absolute = ref_pos[t] + offset)
    out_pos = np.empty(T)
    for t in range(T):
        best = -1e18
        for s in range(NS):
            v = alpha[t, s] + beta[t, s]
            if v > best: best = v
        wsum = 0.0; psum = 0.0
        for i in range(NP):
            for j in range(ND):
                s = i * ND + j
                v = alpha[t, s] + beta[t, s]
                if v < best - 30.0: continue
                w = math.exp(v - best)
                wsum += w
                psum += w * pos_grid[i]
        out_pos[t] = ref_pos[t] + (psum / wsum if wsum > 0 else 0.0)

    return out_pos, row_node

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)

def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw.TVT_input.notna() & hw.GR.notna()]
    if len(kn) < 20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn.GR.values - np.interp(kn.TVT_input.values, tw_tvt, tw_gr)),
                          PF_GR_SIG_MIN, PF_GR_SIG_MAX))

def load_well(wid):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tw_tvt = tw['TVT'].values.astype(float); tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(kn) < 50 or len(ev) < 50: return None
    return hw, tw_tvt, tw_gr, kn, ev

def run_one(wid):
    r = load_well(wid)
    if r is None: return None
    hw, tw_tvt, tw_gr, kn, ev = r
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    ls = float(kn.TVT_input.iloc[-1] + kn.Z.iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values)
    m = dm > 0
    ir0 = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)

    md_v = ev.MD.values.astype(np.float64); z_v = ev.Z.values.astype(np.float64); gr_v = ev.GR.values.astype(np.float64)
    true_tvt = ev.TVT.values.astype(np.float64)

    node_md = np.arange(md_v[0], md_v[-1] + NODE_SPACING, NODE_SPACING)
    if node_md[-1] < md_v[-1]: node_md = np.append(node_md, md_v[-1])
    pos_grid = np.arange(-POS_HALF_RANGE, POS_HALF_RANGE + POS_STEP, POS_STEP)
    dip_grid = np.arange(-DIP_HALF_RANGE, DIP_HALF_RANGE + DIP_STEP, DIP_STEP)
    ref_pos = ls + ir0 * (node_md - md_v[0])

    out_pos, row_node = _trellis_fwdbwd(
        md_v, z_v, gr_v, gg, gmin, gst, gs,
        node_md, pos_grid, dip_grid, ref_pos, ir0,
        PERSIST_RHO, PERSIST_RESID_STD, SIGMA_POS_BASE, SIGMA_POS_GROWTH)

    pos_at_rows = np.interp(md_v, node_md, out_pos)
    tvt_pred = pos_at_rows - z_v
    return tvt_pred, true_tvt

def pooled_rmse(wid_list):
    sq, n = 0.0, 0
    for wid in wid_list:
        r = run_one(wid)
        if r is None: continue
        pred, true_tvt = r
        sq += np.sum((pred - true_tvt) ** 2); n += len(true_tvt)
    return float(np.sqrt(sq / n)) if n > 0 else np.nan

if __name__ == '__main__':
    import time
    wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
    rng = np.random.default_rng(7)
    sample = list(rng.choice(wids, size=10, replace=False))
    t0 = time.time()
    v = pooled_rmse(sample)
    print(f'trellis fwd-bwd pooled RMSE on {len(sample)} wells: {v:.4f}  ({time.time()-t0:.0f}s, {(time.time()-t0)/len(sample):.1f}s/well)')
