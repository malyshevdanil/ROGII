"""Recompute pf_z WITH its particle-spread std (run_pf_z returns (pts, std_), we only saved pts before)
for the true 160-well holdout, to test confidence-weighted standalone blend value -- the same pattern
that worked for neighbor-transfer (fade toward the current best track when uncertain)."""
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
        std_on_proxy = np.interp(px['md'], md_ev, std)
        per_well[wid] = dict(pf_z=pf_on_proxy, pf_z_std=std_on_proxy, true=px['true'])
        if (i + 1) % 30 == 0:
            print(f'  {i+1}/{len(WIDS)}  {time.time()-t0:.0f}s', flush=True)
    print(f'done {len(per_well)} wells  {time.time()-t0:.0f}s')
    pickle.dump(per_well, open('pfz_with_std_holdout.pkl', 'wb'))
    print('saved pfz_with_std_holdout.pkl')
