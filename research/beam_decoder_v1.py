"""New architecture, step 2: a Hidden Semi-Markov beam-search decoder over (position, active-dip, age)
triples, replacing the PF's continuous AR(1) diffusion motion model with an explicit piecewise-linear
duration-aware model grounded in our own priors fit from known-zone data (stride3_fit_priors.py, after
fixing the landing-curve contamination bug):

  - segment length ~ LogNormal(mu=4.900, sigma=0.664)  ->  hazard(age) = pdf(age)/survival(age)
  - dip persistence: dip_new | dip_old ~ Normal(rho*dip_old, resid_std),  rho=0.2697, resid_std=0.0301

At each MD step, every beam hypothesis either CONTINUES its current segment (dip held fixed, prob =
1-hazard(age)*dm) or BREAKS into a new segment (prob = hazard(age)*dm) with a new dip proposed from the
persistence-conditional grid. Position is always continuous (no kink in level, only in slope) -- this
matches the human-markup finding (WORKING_NOTE.md S8.7: piecewise-linear, quantized dips). Emission
likelihood (GR vs typewell curve) scores every branch exactly like the PF. Prune to top-K by cumulative
log-score each step (approximate Viterbi / beam search over a duration-aware HSMM).

Architecturally different from the PF in kind: jump process on SLOPE with explicit duration hazard, vs
continuous AR(1) diffusion. Uses the SAME holdout harness (known-zone-conditioned, eval = TVT_input masked
rows scored against the always-present TVT column) as every other test this project, for direct comparison
against flat=15.1 and the production pf_ancc baseline.
"""
import numpy as np, pandas as pd, glob, os, pickle, math
from numba import njit

TRAIN_DIR = 'd:/ROGII/data/train'
PF_GR_SIG_MIN = 10.; PF_GR_SIG_MAX = 60.; PF_GR_SIG_DEF = 30.

# priors fit in stride3_fit_priors.py (post landing-curve fix, tracked on pos=TVT+Z to match
# the PF's own state convention -- NOT raw TVT_input, which the beam decoder v1 first cut got
# wrong and produced RMSE=32 as a result)
LOGLEN_MU = 5.706; LOGLEN_SIGMA = 0.807
PERSIST_RHO = 0.4725; PERSIST_RESID_STD = 0.0325
DIP_STD = 0.0409  # unconditional, used only for the initial dip grid spread

# A single global log-normal segment-length prior turned out too rigid: wells with genuinely
# "wiggly" (multi-reversal) eval trajectories need much shorter effective segments than the
# global median (357ft), but their KNOWN zone alone doesn't reliably signal this in advance
# (e.g. well 46d24a09's known-zone lateral is a single 767ft straight segment, yet its eval
# region reverses direction every ~300-500ft) -- so per-well adaptation from the known zone
# can't fully anticipate it either. Fix: a 2-component mixture hazard (short + long regime)
# fit on a median split of the same 721-segment sample, so the hazard always carries enough
# early mass to catch a surprise reversal without needing to see it coming.
MIX_W = 0.5
LOGLEN_MU_SHORT = 5.057; LOGLEN_SIGMA_SHORT = 0.629
LOGLEN_MU_LONG = 6.356; LOGLEN_SIGMA_LONG = 0.250

K_BEAM = 60
G_GRID = 9  # candidate new dips per breakpoint branch

@njit(cache=False)
def _seed_jit(seed):
    np.random.seed(seed)

@njit(cache=False)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=False)
def _lognorm_pdf(x, mu, sigma):
    if x <= 1e-6: return 0.0
    lx = math.log(x)
    return math.exp(-((lx-mu)**2)/(2.*sigma*sigma)) / (x*sigma*math.sqrt(2.*math.pi))

@njit(cache=False)
def _lognorm_sf(x, mu, sigma):
    if x <= 1e-6: return 1.0
    z = (math.log(x)-mu)/sigma
    return 1.0 - 0.5*(1.0+math.erf(z/math.sqrt(2.)))

@njit(cache=False)
def _hazard(age, mu, sigma):
    sf = _lognorm_sf(age, mu, sigma)
    if sf < 1e-9: return 1.0
    h = _lognorm_pdf(age, mu, sigma) / sf
    return h

@njit(cache=False)
def _hazard_mix(age, w, mu1, sigma1, mu2, sigma2):
    s1 = _lognorm_sf(age, mu1, sigma1); s2 = _lognorm_sf(age, mu2, sigma2)
    f1 = _lognorm_pdf(age, mu1, sigma1); f2 = _lognorm_pdf(age, mu2, sigma2)
    S = w*s1 + (1.-w)*s2
    if S < 1e-9: return 1.0
    f = w*f1 + (1.-w)*f2
    return f / S

@njit(cache=False)
def _beam_decode(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir0,
                  mix_w, loglen_mu_s, loglen_sigma_s, loglen_mu_l, loglen_sigma_l,
                  persist_rho, persist_resid, dip_std,
                  K, G):
    n = len(md_v)
    pos_arr = np.empty((n, K))
    dip_arr = np.empty((n, K))
    age_arr = np.empty((n, K))
    score_arr = np.empty((n, K))
    parent_arr = np.empty((n, K), dtype=np.int64)

    # init: K hypotheses spread around measured ir0
    for k in range(K):
        pos_arr[0, k] = ls
        dip_arr[0, k] = ir0 + dip_std*0.6*np.random.randn()
        age_arr[0, k] = 0.0
        score_arr[0, k] = 0.0
    parent_arr[0, :] = -1
    pm = md_v[0] - 1.0

    # scratch buffers for children: up to K*(1+G)
    max_children = K*(1+G)
    cand_score = np.empty(max_children)
    cand_pos = np.empty(max_children)
    cand_dip = np.empty(max_children)
    cand_age = np.empty(max_children)
    cand_parent = np.empty(max_children, dtype=np.int64)

    for i in range(n):
        dm = md_v[i] - pm
        if dm < 0.01: dm = 0.01
        nc = 0
        for k in range(K):
            if i == 0:
                pos_p = pos_arr[0, k]; dip_p = dip_arr[0, k]; age_p = age_arr[0, k]; score_p = score_arr[0, k]
            else:
                pos_p = pos_arr[i-1, k]; dip_p = dip_arr[i-1, k]; age_p = age_arr[i-1, k]; score_p = score_arr[i-1, k]

            h = _hazard_mix(age_p, mix_w, loglen_mu_s, loglen_sigma_s, loglen_mu_l, loglen_sigma_l)
            p_break = h*dm
            if p_break > 0.98: p_break = 0.98
            if p_break < 1e-6: p_break = 1e-6
            log_cont = math.log(1.0-p_break)
            log_break = math.log(p_break)

            # continue branch
            new_pos = pos_p + dip_p*dm
            new_age = age_p + dm
            emit = 0.0
            gv = gr_v[i]
            if not math.isnan(gv):
                eg = _interp1(gg, new_pos-z_v[i], vmin, step)
                d = (gv-eg)/gs
                emit = -0.5*d*d
            cand_score[nc] = score_p + log_cont + emit
            cand_pos[nc] = new_pos; cand_dip[nc] = dip_p; cand_age[nc] = new_age; cand_parent[nc] = k
            nc += 1

            # break branches: G candidate new dips around persistence-conditional mean
            cmean = persist_rho*dip_p
            for g in range(G):
                z = -3.0 + 6.0*g/(G-1)
                d_new = cmean + z*persist_resid
                logp_dip = -0.5*z*z  # log density up to additive const (const cancels below)
                new_pos2 = pos_p + d_new*dm
                emit2 = 0.0
                if not math.isnan(gv):
                    eg2 = _interp1(gg, new_pos2-z_v[i], vmin, step)
                    d2 = (gv-eg2)/gs
                    emit2 = -0.5*d2*d2
                cand_score[nc] = score_p + log_break + logp_dip + emit2
                cand_pos[nc] = new_pos2; cand_dip[nc] = d_new; cand_age[nc] = 0.0; cand_parent[nc] = k
                nc += 1

        # diverse top-K selection: greedily take candidates in score order, but cap how many
        # can share a (dip, age)-bucket, so a locally-suboptimal-but-structurally-distinct
        # hypothesis (e.g. an early, still-unproven breakpoint) survives instead of the beam
        # collapsing to near-duplicates of the single best-looking trajectory so far.
        order = np.argsort(-cand_score[:nc])
        n_dip_buckets = 80; n_age_buckets = 60
        bucket_count = np.zeros(n_dip_buckets*n_age_buckets, dtype=np.int64)
        cap = 3
        sel = np.empty(K, dtype=np.int64)
        used = np.zeros(nc, dtype=np.bool_)
        nsel = 0
        for t in range(nc):
            if nsel >= K: break
            j = order[t]
            db = int(cand_dip[j]/0.0125) + n_dip_buckets//2
            if db < 0: db = 0
            if db >= n_dip_buckets: db = n_dip_buckets-1
            ab = int(cand_age[j]/40.0)
            if ab >= n_age_buckets: ab = n_age_buckets-1
            bk = db*n_age_buckets + ab
            if bucket_count[bk] < cap:
                sel[nsel] = j; nsel += 1
                bucket_count[bk] += 1
                used[t] = True
        # fill any remaining slots with next-best regardless of bucket cap (single extra pass,
        # using `used` on the ORDER index t, not the candidate index, so no duplicate scan needed)
        if nsel < K:
            for t in range(nc):
                if nsel >= K: break
                if not used[t]:
                    sel[nsel] = order[t]; nsel += 1
        for k in range(K):
            j = sel[k] if k < nsel else sel[nsel-1]
            pos_arr[i, k] = cand_pos[j]; dip_arr[i, k] = cand_dip[j]
            age_arr[i, k] = cand_age[j]; score_arr[i, k] = cand_score[j]
            parent_arr[i, k] = cand_parent[j]
        pm = md_v[i]

    return pos_arr, score_arr, parent_arr

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax+step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)

def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw.TVT_input.notna() & hw.GR.notna()]
    if len(kn) < 20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn.GR.values-np.interp(kn.TVT_input.values, tw_tvt, tw_gr)),
                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))

def load_well(wid):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tw_tvt = tw['TVT'].values.astype(float); tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(kn) < 50 or len(ev) < 50: return None
    return hw, tw_tvt, tw_gr, kn, ev

def run_one(wid, seed, K=K_BEAM, G=G_GRID, weighted=True):
    _seed_jit(seed)
    r = load_well(wid)
    if r is None: return None
    hw, tw_tvt, tw_gr, kn, ev = r
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    ls = float(kn.TVT_input.iloc[-1]+kn.Z.iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values)
    m = dm > 0
    ir0 = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)

    md_v = ev.MD.values.astype(np.float64); z_v = ev.Z.values.astype(np.float64); gr_v = ev.GR.values.astype(np.float64)
    true_tvt = ev.TVT.values.astype(np.float64)

    pos_arr, score_arr, parent_arr = _beam_decode(
        md_v, z_v, gr_v, gg, gmin, gst, gs, ls, ir0,
        MIX_W, LOGLEN_MU_SHORT, LOGLEN_SIGMA_SHORT, LOGLEN_MU_LONG, LOGLEN_SIGMA_LONG,
        PERSIST_RHO, PERSIST_RESID_STD, DIP_STD, K, G)

    n = len(md_v)
    if not weighted:
        # MAP-only: backtrack the single best final beam
        j = int(np.argmax(score_arr[n-1]))
        path = np.empty(n)
        for i in range(n-1, -1, -1):
            path[i] = pos_arr[i, j] - z_v[i]
            j = int(parent_arr[i, j])
        return path, true_tvt

    # soft estimate: backtrack top-min(K,10) final beams, softmax-weight by final score
    KK = min(K, 10)
    final_scores = score_arr[n-1]
    order = np.argsort(-final_scores)[:KK]
    sc = final_scores[order]; sc = sc - sc.max()
    w = np.exp(sc); w = w/w.sum()
    acc = np.zeros(n)
    for wi, j0 in zip(w, order):
        j = int(j0)
        path = np.empty(n)
        for i in range(n-1, -1, -1):
            path[i] = pos_arr[i, j] - z_v[i]
            j = int(parent_arr[i, j])
        acc += wi*path
    return acc, true_tvt

def pooled_rmse(wid_list, seed, K=K_BEAM, G=G_GRID, weighted=True):
    sq, n = 0.0, 0
    for wid in wid_list:
        r = run_one(wid, seed, K, G, weighted)
        if r is None: continue
        pts, true_tvt = r
        sq += np.sum((pts - true_tvt) ** 2); n += len(true_tvt)
    return float(np.sqrt(sq / n)) if n > 0 else np.nan

if __name__ == '__main__':
    wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
    rng = np.random.default_rng(7)
    sample = list(rng.choice(wids, size=30, replace=False))  # small smoke-test sample first

    import time
    t0 = time.time()
    v = pooled_rmse(sample, seed=1, K=K_BEAM, G=G_GRID, weighted=True)
    print(f'beam decoder (K={K_BEAM}, G={G_GRID}, weighted) pooled RMSE on {len(sample)} wells: {v:.4f}  ({time.time()-t0:.0f}s)')

    t0 = time.time()
    v2 = pooled_rmse(sample, seed=1, K=K_BEAM, G=G_GRID, weighted=False)
    print(f'beam decoder (MAP-only) pooled RMSE: {v2:.4f}  ({time.time()-t0:.0f}s)')
