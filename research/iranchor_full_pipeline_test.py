"""Does the ir-anchored PF motion-model fix (confirmed real & reproducible on the isolated tracker, anchor
~0.9-1.2, 15.34->13.3-13.4) still help once blended into the ALREADY-BEST combo (sp45 + WARP-blend(0.30) +
physics-pp)? Cross-fit validated on all 773 wells, seed-averaged (3 seeds per well) using the numba-jit-seed
fix so results are genuinely reproducible.
"""
import numpy as np, pandas as pd, glob, os, pickle
from numba import njit
from scipy.signal import savgol_filter

TRAIN_DIR = 'd:/ROGII/data/train'
PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'
WARP_CACHE = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/research/warp_on_proxy_cache.pkl'

ANCC_ALPHA = 0.998; ANCC_RN = 0.002; ANCC_PN = 0.005; ANCC_IS = 0.3; ANCC_RP = 0.1; ANCC_RR = 0.001
PF_RESAMP = 0.5
PF_GR_SIG_MIN = 10.; PF_GR_SIG_MAX = 60.; PF_GR_SIG_DEF = 30.
IR_ANCHOR = 0.9  # confirmed optimum from the isolated-tracker sweep
N_PARTICLES = 300
SEEDS = [1, 2, 3]

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
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1] = cum[j]+w[j]
    u0 = np.random.uniform(0., 1./N); np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0+j/N
        while ci < N-1 and cum[ci+1] < u: ci += 1
        np2[j] = pos[ci]+rp*np.random.randn(); na[j] = aux[ci]+rv*np.random.randn()
    return np2, na

@njit(cache=False)
def _pf_ancc_ir(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP, ir_anchor):
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
    for j in range(N):
        pos[j] = ls+IS*np.random.randn(); rate[j] = ir+0.01*np.random.randn()
    pts = np.empty(len(md_v)); pm = md_v[0]-1.
    target = ir_anchor * ir
    for i in range(len(md_v)):
        dm = md_v[i]-pm; dm = max(dm, 1.)
        for j in range(N):
            rate[j] = target + ALPHA*(rate[j]-target) + RN*np.random.randn()
            pos[j] += rate[j]*dm+PN*np.random.randn()
            tvt_j = pos[j]-z_v[i]; tvt_j = max(tvt_j, vmin-50.); tvt_j = min(tvt_j, vmin+len(gg)*step+50.)
            pos[j] = tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs
                lk = max(np.exp(-0.5*d*d) if d*d < 600. else 0., 1e-300); w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
        ne = 0.
        for j in range(N): ne += w[j]*w[j]
        if 1./ne < RESAMP*N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1./N
        tv = 0.
        for j in range(N): tv += w[j]*(pos[j]-z_v[i])
        pts[i] = tv; pm = md_v[i]
    return pts

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax+step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)

def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw.TVT_input.notna() & hw.GR.notna()]
    if len(kn) < 20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn.GR.values-np.interp(kn.TVT_input.values, tw_tvt, tw_gr)),
                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))

def load_well_raw(wid):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tw_tvt = tw['TVT'].values.astype(float); tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(kn) < 50 or len(ev) < 50: return None
    return hw, tw_tvt, tw_gr, kn, ev

def ir_anchor_track(wid, seed):
    r = load_well_raw(wid)
    if r is None: return None
    hw, tw_tvt, tw_gr, kn, ev = r
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    ls = float(kn.TVT_input.iloc[-1]+kn.Z.iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values)
    m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    md_v = ev.MD.values.astype(np.float64); z_v = ev.Z.values.astype(np.float64); gr_v = ev.GR.values.astype(np.float64)
    _seed_jit(seed)
    pts = _pf_ancc_ir(md_v, z_v, gr_v, gg, gmin, gst, gs, ls, ir, N_PARTICLES,
                       ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP, IR_ANCHOR)
    md_raw = hw['MD'].values.astype(float); ev_raw = hw['TVT_input'].isna().values
    return np.interp(md_raw[ev_raw], md_v, pts)

print('computing seed-averaged ir-anchored PF track for all wells (this is the expensive step)...')
proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
warp_on_proxy = pickle.load(open(WARP_CACHE, 'rb'))
WIDS = [wid for wid in warp_on_proxy if wid in proxy]
print('wells:', len(WIDS))

ir_track = {}
t0 = __import__('time').time()
for i, wid in enumerate(WIDS):
    px = proxy[wid]
    tracks = []
    for sd in SEEDS:
        t = ir_anchor_track(wid, sd)
        if t is not None and len(t) == len(px['true']):
            tracks.append(t)
    if tracks:
        ir_track[wid] = np.mean(tracks, axis=0)
    if (i+1) % 100 == 0:
        print(f'  {i+1}/{len(WIDS)}  {__import__("time").time()-t0:.0f}s')
print('done computing ir-anchored tracks:', len(ir_track), f'{__import__("time").time()-t0:.0f}s total')
pickle.dump(ir_track, open('iranchor_track_cache.pkl', 'wb'))

def robust_poly_fit(x, y, deg=4, n_iter=4):
    wt = np.ones_like(y)
    coef = np.polyfit(x, y, deg, w=wt)
    for _ in range(n_iter):
        resid = y - np.polyval(coef, x)
        s = np.median(np.abs(resid)) * 1.4826 + 1e-6
        u = resid / (4.685 * s)
        wt = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        coef = np.polyfit(x, y, deg, w=wt + 1e-6)
    return coef

def physics_pp(wid, base_track, beta=0.75, warmup=500, smooth=51):
    px = proxy[wid]; md = px['md']; z = px['z']
    U_raw = base_track + z
    coef = robust_poly_fit(md, U_raw, deg=4)
    U_fit = np.polyval(coef, md)
    md0 = md.min()
    ramp = np.clip((md - md0) / max(warmup, 1e-6), 0, 1) * beta
    U_blend = (1 - ramp) * U_raw + ramp * U_fit
    if smooth >= 5 and smooth % 2 == 1 and len(U_blend) > smooth:
        U_blend = savgol_filter(U_blend, smooth, 2)
    return U_blend - z

def best_combo_track(wid):
    px = proxy[wid]
    base = (1 - 0.30) * px['sp45'] + 0.30 * warp_on_proxy[wid]
    return physics_pp(wid, base)

def combo_with_ir(wid, ir_weight):
    combo = best_combo_track(wid)
    if ir_weight <= 0 or wid not in ir_track:
        return combo
    return (1 - ir_weight) * combo + ir_weight * ir_track[wid]

def pooled(pred_fn, wid_list):
    s = []
    for wid in wid_list:
        p = pred_fn(wid); s.append((p - proxy[wid]['true']) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(s))))

WIDS_IR = [w for w in WIDS if w in ir_track]
print('\n=== full-set sweep: best-combo (WARP+physics-pp) + ir-anchor blend ===')
print('baseline (best combo, no ir-anchor):', pooled(lambda wid: best_combo_track(wid), WIDS_IR))
for w in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
    v = pooled(lambda wid, w=w: combo_with_ir(wid, w), WIDS_IR)
    print(f'  ir_weight={w:.2f}  pooled={v:.4f}')

print('\n=== 5-fold cross-fit ===')
rng = np.random.default_rng(13)
order = np.array(WIDS_IR); rng.shuffle(order)
folds = np.array_split(order, 5)
W_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

def best_w_on(wl):
    best = (1e9, None)
    for w in W_GRID:
        v = pooled(lambda wid, w=w: combo_with_ir(wid, w), wl)
        if v < best[0]: best = (v, w)
    return best

gains = []
for i in range(5):
    test_wl = folds[i]; train_wl = np.concatenate([folds[j] for j in range(5) if j != i])
    b = pooled(lambda wid: best_combo_track(wid), test_wl)
    _, w = best_w_on(train_wl)
    v = pooled(lambda wid, w=w: combo_with_ir(wid, w), test_wl)
    gains.append(b - v)
    print(f'fold {i}: baseline={b:.4f}  ir_weight={w:.2f}  with_ir={v:.4f}  gain={b-v:+.4f}')
print('mean gain:', np.mean(gains), 'positive in', sum(g>0 for g in gains), '/5 folds')
