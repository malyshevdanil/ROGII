"""track6 REBUILD, step 1+2 for ALL 773 wells (not just the 160 holdout): multi-seed (N=16) pf_ancc
ensemble + physics-pp robust-polynomial projection, saved in the same {wid: array} format as
track6_pfancc_all773.pkl so it drops straight into the existing stage2 (correlation features) and
stage6 (spatial features) builders, replacing their old single-seed pf_ancc base.
Diagnostic on the true_holdout_160 already showed: single-seed 13.00 -> N=16 ensemble 11.84 -> +physics-pp
11.59 (still ~1.0ft short of sp45's 10.62, but the OLD track6 GBM+features layer took single-seed pf_ancc
from its own 14.93 down to 11.39, a ~3.5ft gain on a much weaker base -- reapplying that feature value on
this new, stronger 11.59 base is the next step, which needs this base computed for all 773 wells first).
"""
import numpy as np, pandas as pd, pickle, time
import pf_ancc_source as pfs

TRAIN_DIR = 'd:/ROGII/data/train'
PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'
N_SEEDS = 16
WARMUP = 500.; SMOOTH = 51; BETA = 0.75; DEG = 4


def robust_poly_fit(x, y, deg=4, n_iter=4):
    w = np.ones_like(y); coef = np.polyfit(x, y, deg, w=w)
    for _ in range(n_iter):
        resid = y - np.polyval(coef, x)
        s = np.median(np.abs(resid)) * 1.4826 + 1e-6
        u = resid / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        coef = np.polyfit(x, y, deg, w=w + 1e-6)
    return coef


def apply_pp(pred, md, z):
    order = np.argsort(md)
    md_s = md[order]; z_s = z[order]; pred_s = pred[order]
    U = pred_s + z_s
    try:
        coef = robust_poly_fit(md_s, U, deg=DEG)
    except Exception:
        return pred
    Ufit = np.polyval(coef, md_s)
    ramp = np.clip((md_s - md_s.min()) / max(WARMUP, 1e-6), 0, 1) * BETA
    Ublend = (1 - ramp) * U + ramp * Ufit
    from scipy.signal import savgol_filter
    if SMOOTH >= 5 and SMOOTH % 2 == 1 and len(Ublend) > SMOOTH:
        Ublend = savgol_filter(Ublend, SMOOTH, 2)
    new_pred_s = Ublend - z_s
    new_pred = np.empty_like(pred_s); new_pred[order] = new_pred_s
    return new_pred


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
    WIDS = sorted(proxy.keys())
    print('n wells:', len(WIDS), flush=True)

    t0 = time.time()
    base_v2 = {}
    for i, wid in enumerate(WIDS):
        r = load(wid)
        if r is None: continue
        hw, tw_tvt, tw_gr, ev = r
        md_ev = ev.MD.values.astype(np.float64)
        px = proxy[wid]
        md_px = px['md'].astype(np.float64); z_px = px['z'].astype(np.float64)
        seed_preds = []
        for s in range(N_SEEDS):
            pts, std = pfs.run_pf_ancc(hw, tw_tvt, tw_gr)
            pf_on_proxy = np.interp(md_px, md_ev, pts)
            seed_preds.append(pf_on_proxy)
        ens = np.mean(seed_preds, axis=0)
        ens_pp = apply_pp(ens, md_px, z_px)
        base_v2[wid] = ens_pp.astype(np.float32)
        if (i + 1) % 30 == 0:
            print(f'  {i+1}/{len(WIDS)}  {time.time()-t0:.0f}s', flush=True)
            pickle.dump(base_v2, open('track6_pfancc_v2_all773.pkl', 'wb'))
    print(f'done {len(base_v2)} wells  {time.time()-t0:.0f}s', flush=True)
    pickle.dump(base_v2, open('track6_pfancc_v2_all773.pkl', 'wb'))
    print('saved track6_pfancc_v2_all773.pkl', flush=True)
