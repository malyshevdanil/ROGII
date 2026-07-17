"""Improve neighbor-transfer (v1: single nearest neighbor, corr=0.193 but tail-risk grows fast beyond
w=0.02) before combining with beam2. Two improvements tested:
(a) IDW across k=3 nearest DIFFERENT wells instead of k=1 (reduces risk from any single bad match).
(b) distance-adaptive per-row confidence: rows whose nearest match is far away get less trust (matches
    the observed pattern: quality degrades sharply beyond ~2000ft).
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
    tvt_true = hw['TVT'].to_numpy(np.float64)
    tvt_known = hw['TVT_input'].to_numpy(np.float64)
    pos = np.where(km.values, tvt_known, np.nan) + z
    return dict(hw=hw, xy=xy, z=z, pos_known=pos, km=km.values, tvt_true=tvt_true)

if __name__ == '__main__':
    proxy = pickle.load(open('C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl', 'rb'))['DATA']
    VA_WIDS = pickle.load(open('warp_true_holdout_160.pkl', 'rb'))
    train_wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})

    cents = []
    for wid in train_wids:
        df = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv', usecols=['X', 'Y'])
        cents.append((wid, df.X.median(), df.Y.median()))
    cent_xy = np.array([[c[1], c[2]] for c in cents]); cent_ids = [c[0] for c in cents]
    cent_tree = cKDTree(cent_xy)

    # pre-build per-well curve cache and XY-point trees (expensive; do once)
    curve_cache = {}
    tree_cache = {}

    def get_curve(wid):
        if wid not in curve_cache:
            curve_cache[wid] = build_well_xy_curve(wid)
        return curve_cache[wid]

    def get_tree(wid):
        if wid not in tree_cache:
            c = get_curve(wid)
            tree_cache[wid] = cKDTree(c['xy']) if c is not None else None
        return tree_cache[wid]

    K_NEIGHBORS = 3
    DIST_SCALE = 1500.0  # ft; confidence weight = exp(-dist/DIST_SCALE)

    PER_WELL = {}
    n_ok = 0
    for wid in VA_WIDS:
        if wid not in proxy: continue
        i = cent_ids.index(wid) if wid in cent_ids else None
        if i is None: continue
        dist, idx = cent_tree.query(cent_xy[i], k=K_NEIGHBORS + 1)
        nn_wids = [cent_ids[j] for j in idx[1:]]  # skip self

        target = get_curve(wid)
        if target is None: continue
        kn_idx = np.where(target['km'])[0]
        if len(kn_idx) < 20: continue
        ev_idx = np.where(~target['km'])[0]

        preds_by_neighbor = []  # (pred_tvt_array, match_dist_array)
        for nn_wid in nn_wids:
            neighbor = get_curve(nn_wid)
            ntree = get_tree(nn_wid)
            if neighbor is None or ntree is None: continue
            neighbor_pos_full = neighbor['tvt_true'] + neighbor['z']

            dist_kn, nn_idx_kn = ntree.query(target['xy'][kn_idx], k=1)
            borrowed_kn = neighbor_pos_full[nn_idx_kn]
            true_kn_pos = target['pos_known'][kn_idx]
            datum_offset = float(np.median(true_kn_pos - borrowed_kn))

            dist_ev, idx_ev = ntree.query(target['xy'][ev_idx], k=1)
            borrowed_pos = neighbor_pos_full[idx_ev] + datum_offset
            pred_tvt = borrowed_pos - target['z'][ev_idx]
            preds_by_neighbor.append((pred_tvt, dist_ev))

        if not preds_by_neighbor: continue

        # IDW combine across the K neighbors (weight = 1/(dist+eps))
        preds_stack = np.stack([p for p, d in preds_by_neighbor], axis=1)  # (n_ev, K)
        dists_stack = np.stack([d for p, d in preds_by_neighbor], axis=1)
        w_idw = 1.0 / (dists_stack + 50.0)
        w_idw /= w_idw.sum(axis=1, keepdims=True)
        pred_idw = (preds_stack * w_idw).sum(axis=1)

        # single-nearest (k=1) for comparison
        pred_k1 = preds_by_neighbor[0][0]
        dist_k1 = preds_by_neighbor[0][1]

        # distance-adaptive confidence (based on nearest match distance)
        conf = np.exp(-dist_k1 / DIST_SCALE)

        true_tvt = target['tvt_true'][ev_idx]
        PER_WELL[wid] = dict(pred_k1=pred_k1, pred_idw=pred_idw, dist_k1=dist_k1, conf=conf, true=true_tvt)
        n_ok += 1

    print(f'n wells: {n_ok}')
    rmse_k1 = np.sqrt(np.mean(np.concatenate([(d['pred_k1']-d['true'])**2 for d in PER_WELL.values()])))
    rmse_idw = np.sqrt(np.mean(np.concatenate([(d['pred_idw']-d['true'])**2 for d in PER_WELL.values()])))
    print(f'pooled RMSE k=1: {rmse_k1:.3f}   IDW(k={K_NEIGHBORS}): {rmse_idw:.3f}')

    pickle.dump(PER_WELL, open('neighbor_transfer_v2_holdout.pkl', 'wb'))
    print('saved neighbor_transfer_v2_holdout.pkl')
