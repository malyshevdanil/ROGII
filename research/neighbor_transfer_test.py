"""Idea: full-curve neighbor transfer (De DQ / stevenleehans thread) -- "copy the shape of the nearest
well's interpreted TVT curve" via spatial (X,Y) matching + a per-well datum offset, NOT just a smoothed
formation-plane fit (track6's FormationPlaneKNN). Ruled out earlier by geometry (our 3 real test wells'
nearest DIFFERENT neighbor is 311-550ft, outside stevenleehans's "<150ft" sweet spot) -- but that
threshold was HIS empirical finding on HIS data, not a hard physical law. Test directly: for a sample of
true-holdout wells, does transferring the single nearest OTHER well's own structural-elevation curve
(pos = TVT+Z, matching production's own state convention) carry ANY real signal at typical real-world
neighbor distances (not just <150ft)?
"""
import numpy as np, pandas as pd, pickle, glob, os
from scipy.spatial import cKDTree

TRAIN_DIR = 'd:/ROGII/data/train'

def build_well_xy_curve(wid):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    if 'TVT' not in hw.columns: return None
    km = hw['TVT_input'].notna()
    xy = hw[['X', 'Y']].to_numpy(np.float64)
    z = hw['Z'].to_numpy(np.float64)
    tvt_true = hw['TVT'].to_numpy(np.float64)  # known where km, else NaN (only used for eval target)
    tvt_known = hw['TVT_input'].to_numpy(np.float64)
    pos = np.where(km.values, tvt_known, np.nan) + z  # structural elevation, known-zone only for now
    return dict(hw=hw, xy=xy, z=z, pos_known=pos, km=km.values, tvt_true=tvt_true)

def transfer_predict(target_xy, target_z, neighbor_xy, neighbor_pos_full, neighbor_tree, datum_offset):
    """For each target point, find nearest neighbor-well point (by XY), borrow its structural elevation
    (already including the neighbor's own interpolated full pos curve across known+eval), apply the
    datum offset, convert back to TVT."""
    dist, idx = neighbor_tree.query(target_xy, k=1)
    borrowed_pos = neighbor_pos_full[idx] + datum_offset
    pred_tvt = borrowed_pos - target_z
    return pred_tvt, dist

if __name__ == '__main__':
    proxy = pickle.load(open('C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl', 'rb'))['DATA']
    VA_WIDS = pickle.load(open('warp_true_holdout_160.pkl', 'rb'))
    train_wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})

    # build centroid tree for finding nearest-neighbor WELL (not point) first
    cents = []
    for wid in train_wids:
        df = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv', usecols=['X', 'Y'])
        cents.append((wid, df.X.median(), df.Y.median()))
    cent_xy = np.array([[c[1], c[2]] for c in cents]); cent_ids = [c[0] for c in cents]
    cent_tree = cKDTree(cent_xy)

    results = []
    PER_WELL = {}
    for wid in VA_WIDS:
        if wid not in proxy: continue
        i = cent_ids.index(wid) if wid in cent_ids else None
        if i is None: continue
        dist, idx = cent_tree.query(cent_xy[i], k=2)
        nn_wid = cent_ids[idx[1]]; nn_dist = dist[1]

        target = build_well_xy_curve(wid)
        neighbor = build_well_xy_curve(nn_wid)
        if target is None or neighbor is None: continue

        # neighbor's FULL pos curve: known-zone pos is TVT_input+Z (exact); eval-zone pos uses the
        # neighbor's OWN TRUE TVT+Z (legal since neighbor is a fully-labeled train well -- this tests
        # the ORACLE ceiling of the mechanism: "if we had the true neighbor curve, would matching help").
        neighbor_pos_full = np.where(neighbor['km'], neighbor['tvt_true'], neighbor['tvt_true']) + neighbor['z']
        # (both branches equal since tvt_true is available for the whole neighbor well; kept for clarity)
        neighbor_tree = cKDTree(neighbor['xy'])

        # datum offset: match on the KNOWN zone only (legal), via median difference of pos
        kn_idx = np.where(target['km'])[0]
        if len(kn_idx) < 20: continue
        dist_kn, nn_idx_kn = neighbor_tree.query(target['xy'][kn_idx], k=1)
        borrowed_kn = neighbor_pos_full[nn_idx_kn]
        true_kn_pos = target['pos_known'][kn_idx]
        datum_offset = float(np.median(true_kn_pos - borrowed_kn))

        ev_idx = np.where(~target['km'])[0]
        pred_tvt, match_dist = transfer_predict(target['xy'][ev_idx], target['z'][ev_idx],
                                                  neighbor['xy'], neighbor_pos_full, neighbor_tree, datum_offset)
        true_tvt = target['tvt_true'][ev_idx]
        v = np.isfinite(pred_tvt) & np.isfinite(true_tvt)
        if v.sum() < 20: continue
        rmse = np.sqrt(np.mean((pred_tvt[v] - true_tvt[v]) ** 2))
        results.append((wid, nn_wid, nn_dist, rmse, v.sum())); PER_WELL[wid] = dict(pred=pred_tvt, true=true_tvt, ev_idx=ev_idx, v=v)

    print(f'n wells tested: {len(results)}')
    dists = np.array([r[2] for r in results])
    rmses = np.array([r[3] for r in results])
    print(f'neighbor distance: median={np.median(dists):.0f}ft  p25={np.percentile(dists,25):.0f}  p75={np.percentile(dists,75):.0f}')
    print(f'transfer RMSE: median={np.median(rmses):.2f}  mean={np.mean(rmses):.2f}')

    # bucket by neighbor distance to see if closer -> better (oracle ceiling test)
    for lo, hi in [(0, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 1e9)]:
        m = (dists >= lo) & (dists < hi)
        if m.sum() == 0: continue
        print(f'  dist[{lo}-{hi}) n={m.sum()}  median_rmse={np.median(rmses[m]):.2f}')

    pooled_rmse = np.sqrt(np.mean(rmses ** 2))  # rough (not exactly pooled per-row, but indicative)
    print(f'\nrough pooled-ish RMSE (oracle neighbor-transfer): {pooled_rmse:.2f}')
    print('(compare vs pf_ancc solo ~13, full_combo ~9.4, sp45 ~10.6)')
    import pickle
    pickle.dump(PER_WELL, open('neighbor_transfer_holdout.pkl','wb'))
    print('saved neighbor_transfer_holdout.pkl')
