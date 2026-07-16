"""Compute pf_z for all 773 wells (needed to add as a track6 feature, matching how pf_ancc_all773.py
was built)."""
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
    WIDS = list(proxy.keys())
    print('n wells:', len(WIDS))

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
        per_well[wid] = pf_on_proxy
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(WIDS)}  {time.time()-t0:.0f}s', flush=True)
    print(f'done {len(per_well)} wells  {time.time()-t0:.0f}s')
    pickle.dump(per_well, open('pfz_all773.pkl', 'wb'))
