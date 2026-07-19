"""track6 REBUILD, step 1: is pf_ancc's own standalone weakness (11.39 vs sp45's 10.62 on true_holdout_160)
fixable by simply averaging multiple seeds, the same mechanism that took the base pipeline 7.230->7.096?
pf_ancc has no exposed seed control (numba's internal RNG just keeps advancing across calls, giving
naturally-diverse repeated runs) -- run N_SEEDS independent passes per well, average, and report pooled
RMSE at ensemble sizes 1/2/4/8/16 against sp45 and single-seed pf_ancc, on the true 160-well holdout.
"""
import numpy as np, pandas as pd, pickle, time
import pf_ancc_source as pfs

TRAIN_DIR = 'd:/ROGII/data/train'
PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'
N_SEEDS = 16


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
    WIDS = sorted(VA_WIDS)
    print('n wells (true holdout):', len(WIDS), flush=True)

    t0 = time.time()
    per_well = {}
    for i, wid in enumerate(WIDS):
        r = load(wid)
        if r is None: continue
        hw, tw_tvt, tw_gr, ev = r
        px = proxy[wid]
        md_ev = ev.MD.values.astype(np.float64)
        true_tvt = px['true'].astype(np.float64)
        md_px = px['md'].astype(np.float64)
        seed_preds = []
        for s in range(N_SEEDS):
            pts, std = pfs.run_pf_ancc(hw, tw_tvt, tw_gr)
            pf_on_proxy = np.interp(md_px, md_ev, pts)
            seed_preds.append(pf_on_proxy)
        per_well[wid] = dict(seed_preds=np.stack(seed_preds), true=true_tvt)
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(WIDS)}  {time.time()-t0:.0f}s', flush=True)
            pickle.dump(per_well, open('track6_v2_multiseed_pfancc_holdout.pkl', 'wb'))
    print(f'done {len(per_well)} wells  {time.time()-t0:.0f}s', flush=True)
    pickle.dump(per_well, open('track6_v2_multiseed_pfancc_holdout.pkl', 'wb'))
    print('saved track6_v2_multiseed_pfancc_holdout.pkl', flush=True)

    def pooled(preds, trues):
        return float(np.sqrt(np.mean(np.concatenate([(p - t) ** 2 for p, t in zip(preds, trues)]))))

    sp45_preds = [proxy[w]['sp45'].astype(np.float64) for w in per_well]
    trues = [per_well[w]['true'] for w in per_well]
    print('sp45 pooled:', pooled(sp45_preds, trues))

    for n in [1, 2, 4, 8, 16]:
        ens_preds = [per_well[w]['seed_preds'][:n].mean(axis=0) for w in per_well]
        print(f'pf_ancc N_SEEDS={n:2d} pooled:', pooled(ens_preds, trues))
