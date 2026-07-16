"""Check pf_z (Z-velocity-coupled PF, different motion model than pf_ancc's pure-momentum state) as a
candidate ADDITIONAL decorrelated feature for track6, or standalone blend partner. Same diagnostic
pattern as track6's own Stage 1 (pf_ancc quality check): standalone quality, tail behavior, correlation
with existing signals, on the TRUE 160-well holdout."""
import numpy as np, pandas as pd, pickle, time
import pf_ancc_source as pfs

TRAIN_DIR = 'd:/ROGII/data/train'
PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'

def load(wid):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tw_tvt = tw['TVT'].values.astype(float); tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    ev = hw[hw.TVT_input.isna()]
    if len(ev) < 50: return None
    return hw, tw_tvt, tw_gr, ev

if __name__ == '__main__':
    proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
    VA_WIDS = pickle.load(open('warp_true_holdout_160.pkl', 'rb'))
    WIDS = [w for w in proxy if w in VA_WIDS]
    print('n wells (true holdout):', len(WIDS))

    t0 = time.time()
    per_well = {}
    for i, wid in enumerate(WIDS):
        r = load(wid)
        if r is None: continue
        hw, tw_tvt, tw_gr, ev = r
        pts, std = pfs.run_pf_z(hw, tw_tvt, tw_gr)
        true_tvt = ev.TVT.values.astype(np.float64)
        if len(pts) != len(true_tvt): continue
        px = proxy[wid]
        md_ev = ev.MD.values.astype(np.float64)
        pf_on_proxy = np.interp(px['md'], md_ev, pts)
        per_well[wid] = dict(pf_z=pf_on_proxy, true=px['true'])
        if (i + 1) % 30 == 0:
            print(f'  {i+1}/{len(WIDS)}  {time.time()-t0:.0f}s', flush=True)
    print(f'done {len(per_well)} wells  {time.time()-t0:.0f}s')
    pickle.dump(per_well, open('pfz_true_holdout.pkl', 'wb'))

    rmses = []
    for w, d in per_well.items():
        rmse = np.sqrt(np.mean((d['pf_z'] - d['true']) ** 2))
        rmses.append((w, rmse))
    rmses.sort(key=lambda x: -x[1])
    print('\nworst 10:')
    for w, r in rmses[:10]:
        print(' ', w, round(r, 1))
    print('median:', np.median([r for _, r in rmses]), 'p90:', np.percentile([r for _, r in rmses], 90))

    pooled = np.sqrt(np.mean(np.concatenate([(d['pf_z'] - d['true']) ** 2 for d in per_well.values()])))
    print('\npooled RMSE (pf_z solo, true holdout):', pooled)

    # correlation with pf_ancc and v7blend
    pf_ancc = pickle.load(open('track6_pfancc_true_holdout.pkl', 'rb'))
    warp_on_proxy = pickle.load(open('warp_on_proxy_cache.pkl', 'rb'))
    gru_v7 = pickle.load(open('gru_v7_kfold_oof_preds.pkl', 'rb'))
    from scipy.signal import savgol_filter

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

    def full_combo_track(wid):
        px = proxy[wid]
        base = (1 - 0.30) * px['sp45'] + 0.30 * warp_on_proxy[wid]
        return physics_pp(wid, base)

    common = [w for w in per_well if w in pf_ancc and w in gru_v7 and w in warp_on_proxy]
    pfz_err = np.concatenate([per_well[w]['pf_z'] - per_well[w]['true'] for w in common])
    pfancc_err = np.concatenate([pf_ancc[w]['pf_ancc'] - pf_ancc[w]['true'] for w in common])
    v7blend_err = np.concatenate([(0.7 * full_combo_track(w) + 0.3 * gru_v7[w]) - per_well[w]['true'] for w in common])
    print('\ncorr(pf_z_err, pf_ancc_err):', np.corrcoef(pfz_err, pfancc_err)[0, 1])
    print('corr(pf_z_err, v7blend_err):', np.corrcoef(pfz_err, v7blend_err)[0, 1])
