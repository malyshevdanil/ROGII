"""Stage 2 of the track6 project: multi-scale correlation features, DECOUPLED from sp45 (unlike the
failed GBM-residual attempt). For each eval row, search a window of candidate TVT offsets AROUND pf_ancc's
own estimate (pf_ancc is a real independent prediction, used here only as a loose search CENTER, not as a
value being corrected -- this is architecturally different from "predict residual from sp45"), at several
window sizes, and extract: best-matching offset, peak correlation, and match sharpness (peak vs runner-up)
at each scale. These features plus pf_ancc's own raw position and standard trajectory features feed a GBM
predicting ABSOLUTE position (not a residual), matching the STRIDE writeup's track6 design.
"""
import numpy as np, pandas as pd, pickle, time

TRAIN_DIR = 'd:/ROGII/data/train'
PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'

SEARCH_HALF = 25.0   # ft, search range around pf_ancc's estimate
SEARCH_STEP = 1.0
WIN_SIZES = (11, 25, 51)

def interp_nan(a):
    a = a.copy(); n = len(a); idx = np.arange(n); m = np.isnan(a)
    if m.all(): return np.zeros(n)
    a[m] = np.interp(idx[m], idx[~m], a[~m]); return a

def match_profile(gwin, twk_gr, tw_tvt, center_tvt, search_half, search_step):
    """Return (best_offset, peak_corr, sharpness) for matching gwin against the typewell curve searched
    over [center_tvt - search_half, center_tvt + search_half]."""
    offsets = np.arange(-search_half, search_half + search_step, search_step)
    gw = gwin - gwin.mean()
    gnorm = np.linalg.norm(gw)
    if gnorm < 1e-9:
        return 0.0, 0.0, 0.0
    corrs = np.empty(len(offsets))
    half = len(gwin) // 2
    for k, off in enumerate(offsets):
        tvt_pts = center_tvt + off + np.arange(-half, len(gwin) - half)
        twin = np.interp(tvt_pts, tw_tvt, twk_gr)
        tw_c = twin - twin.mean()
        tnorm = np.linalg.norm(tw_c)
        corrs[k] = float(np.dot(gw, tw_c) / (gnorm * tnorm)) if tnorm > 1e-9 else 0.0
    best_i = int(np.argmax(corrs))
    peak = corrs[best_i]
    tmp = corrs.copy(); tmp[max(0, best_i - 2):best_i + 3] = -2
    runner_up = tmp.max() if len(tmp) > 5 else peak
    sharpness = peak - runner_up
    return float(offsets[best_i]), float(peak), float(sharpness)

def build_well(wid, pf_track, sub_every=5):
    """sub_every: only compute the (expensive) correlation search every N rows, then interpolate --
    keeps runtime manageable while still giving the GBM a locally-varying signal."""
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    km = hw['TVT_input'].notna()
    if km.sum() < 20 or (~km).sum() < 20 or len(tw) < 10: return None
    last = hw[km].iloc[-1]; last_tvt = float(last['TVT_input']); last_MD = float(last['MD']); last_Z = float(last['Z'])
    tw_tvt = tw['TVT'].values.astype(float); tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    gr = interp_nan(hw['GR'].values.astype(float))
    X_ = hw['X'].values.astype(float); Y = hw['Y'].values.astype(float); Z = hw['Z'].values.astype(float); MD = hw['MD'].values.astype(float)
    mdd = np.gradient(MD); mdd[mdd == 0] = 1
    kg = gr[:int(km.sum())]; ktvt = hw.loc[km, 'TVT_input'].values
    twk = np.interp(ktvt, tw_tvt, tw_gr); v = np.isfinite(kg) & np.isfinite(twk)
    a, b = (np.polyfit(kg[v], twk[v], 1) if v.sum() >= 20 else (1., 0.))
    cal = gr * a + b
    head = np.arctan2(np.gradient(Y), np.gradient(X_) + 1e-9)

    ev_idx = np.where((~km).values)[0]
    n_ev = len(ev_idx)
    if len(pf_track) != n_ev: return None
    pf_tvt = pf_track   # pf_ancc_source.run_pf_ancc returns TVT directly (verified against Stage 1's true-holdout test)

    sub_idx = np.arange(0, n_ev, sub_every)
    n_sub = len(sub_idx)
    off_by_ws = {ws: np.zeros(n_sub) for ws in WIN_SIZES}
    peak_by_ws = {ws: np.zeros(n_sub) for ws in WIN_SIZES}
    sharp_by_ws = {ws: np.zeros(n_sub) for ws in WIN_SIZES}
    cal_ev = cal[ev_idx]
    for si, j in enumerate(sub_idx):
        for ws in WIN_SIZES:
            half = ws // 2
            lo = max(0, j - half); hi = min(n_ev, j + half + 1)
            gwin = cal_ev[lo:hi]
            off, peak, sharp = match_profile(gwin, tw_gr, tw_tvt, pf_tvt[j], SEARCH_HALF, SEARCH_STEP)
            off_by_ws[ws][si] = off; peak_by_ws[ws][si] = peak; sharp_by_ws[ws][si] = sharp

    F = {}
    F['md_since'] = (MD - last_MD)[ev_idx]
    F['gr'] = gr[ev_idx]; F['gr_grad'] = np.gradient(gr)[ev_idx]
    F['gr_rstd'] = pd.Series(gr).rolling(21, center=True, min_periods=1).std().fillna(0).values[ev_idx]
    F['z'] = (Z - last_Z)[ev_idx]; F['dzdmd'] = (np.gradient(Z) / mdd)[ev_idx]; F['cal_gr'] = cal[ev_idx]
    F['pf_ancc_tvt'] = pf_tvt
    F['sin_azi'] = np.sin(head)[ev_idx]; F['cos_azi'] = np.cos(head)[ev_idx]
    for ws in WIN_SIZES:
        F['off%d' % ws] = np.interp(np.arange(n_ev), sub_idx, off_by_ws[ws])
        F['peak%d' % ws] = np.interp(np.arange(n_ev), sub_idx, peak_by_ws[ws])
        F['sharp%d' % ws] = np.interp(np.arange(n_ev), sub_idx, sharp_by_ws[ws])

    FEATCOLS = ['md_since', 'gr', 'gr_grad', 'gr_rstd', 'z', 'dzdmd', 'cal_gr', 'pf_ancc_tvt', 'sin_azi', 'cos_azi']
    for ws in WIN_SIZES: FEATCOLS += ['off%d' % ws, 'peak%d' % ws, 'sharp%d' % ws]
    M = np.stack([F[c] for c in FEATCOLS], axis=1).astype(np.float32)
    true = hw['TVT'].values.astype(float)[ev_idx]
    return M, true, FEATCOLS

if __name__ == '__main__':
    t0 = time.time()
    proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
    pf_all = pickle.load(open('track6_pfancc_all773.pkl', 'rb'))
    WIDS = list(pf_all.keys())
    print('wells with pf_ancc:', len(WIDS))

    DATA = {}
    for i, wid in enumerate(WIDS):
        r = build_well(wid, pf_all[wid])
        if r is None: continue
        M, true, FEATCOLS = r
        DATA[wid] = dict(M=M, true=true)
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(WIDS)}  {time.time()-t0:.0f}s', flush=True)
    print(f'built {len(DATA)} wells  {time.time()-t0:.0f}s')
    pickle.dump(dict(DATA=DATA, FEATCOLS=FEATCOLS), open('track6_stage2_features.pkl', 'wb'))
    print('saved track6_stage2_features.pkl')
