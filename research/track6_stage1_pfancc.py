"""Stage 1 of the "build a real track6" multi-day project: establish the foundation signal quality.
pf_ancc (research/pf_ancc_source.py) is a genuinely independent particle filter (own state, own momentum
motion model, 600 particles) -- NOT derived from sp45 in any way, unlike today's failed GBM-residual
attempt. Check: standalone quality, correlation with our current best (full_combo + v7 GRU-refiner blend),
and CRITICALLY the per-well tail behavior (max/worst-decile RMSE) since that's the exact failure mode that
killed beam-HSMM/trellis earlier this session. If pf_ancc itself has a catastrophic tail, that tail will
propagate into ANY GBM/correlation-feature system built on top of it as track6's backbone signal.
"""
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
    np.random.seed(1)
    for i, wid in enumerate(WIDS):
        r = load(wid)
        if r is None: continue
        hw, tw_tvt, tw_gr, ev = r
        pts, std = pfs.run_pf_ancc(hw, tw_tvt, tw_gr)
        true_tvt = ev.TVT.values.astype(np.float64)
        if len(pts) != len(true_tvt): continue
        # align to proxy's md grid for comparison with full_combo/v7
        px = proxy[wid]
        md_ev = ev.MD.values.astype(np.float64)
        pf_on_proxy = np.interp(px['md'], md_ev, pts)
        per_well[wid] = dict(pf_ancc=pf_on_proxy, true=px['true'])
        if (i + 1) % 30 == 0:
            print(f'  {i+1}/{len(WIDS)}  {time.time()-t0:.0f}s', flush=True)
    print(f'done {len(per_well)} wells  {time.time()-t0:.0f}s')
    pickle.dump(per_well, open('track6_pfancc_true_holdout.pkl', 'wb'))

    rmses = []
    for w, d in per_well.items():
        rmse = np.sqrt(np.mean((d['pf_ancc'] - d['true']) ** 2))
        rmses.append((w, rmse))
    rmses.sort(key=lambda x: -x[1])
    print('\nworst 10:')
    for w, r in rmses[:10]:
        print(' ', w, round(r, 1))
    print('median:', np.median([r for _, r in rmses]), 'p90:', np.percentile([r for _, r in rmses], 90))

    pooled = np.sqrt(np.mean(np.concatenate([(d['pf_ancc'] - d['true']) ** 2 for d in per_well.values()])))
    print('\npooled RMSE (pf_ancc solo, true holdout):', pooled)
