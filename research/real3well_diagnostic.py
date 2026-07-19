"""BIG DIAGNOSTIC: our whole session's paired-bootstrap validation ran on warp_true_holdout_160,
which does NOT contain 2 of our 3 real test wells (00bbac68, 00e12e8b are among the OTHER 613 wells
used to train WARP/GRU-refiner -- only 000d7d20 is a genuine holdout member). So every "should this
help/hurt" call we made this session was blind to how beam2/neighbor/pf_z actually behave on 2/3 of
what determines our real LB score. This script computes each signal DIRECTLY (no training needed for
beam2/pf_z/neighbor -- they are per-well physics/geometry, not fit models) on all 3 real test wells and
checks: (a) is the baseline (gru7 OOF, honest for all 773) already atypically good on all 3, confirming/
extending the single-well 000d7d20 finding; (b) would each decorrelated signal have helped or hurt
SPECIFICALLY on 00bbac68 / 00e12e8b, which were never checked before this script existed.
"""
import numpy as np, pandas as pd, pickle, sys
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/research')
import pf_ancc_source as pfs
import beam2_decoder as b2

TRAIN_DIR = 'd:/ROGII/data/train'
PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'

REAL_TEST_WELLS = ['000d7d20', '00bbac68', '00e12e8b']

b2.EMISSION_WEIGHT = 0.5
PZ_STD_SCALE = 2.614363968372345
NB_K = 3
NB_DIST_SCALE = 1500.0


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load_hw_tw(wid):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    tw_tvt = tw['TVT'].values.astype(float)
    tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    return hw, tw_tvt, tw_gr


def get_neighbor_pred(wid, train_wids, cent_xy, cent_ids, cent_tree, curve_cache, tree_cache):
    def get_curve(w):
        if w not in curve_cache:
            hw = pd.read_csv(f'{TRAIN_DIR}/{w}__horizontal_well.csv')
            if 'TVT' not in hw.columns:
                curve_cache[w] = None
            else:
                km = hw['TVT_input'].notna()
                xy = hw[['X', 'Y']].to_numpy(np.float64)
                z = hw['Z'].to_numpy(np.float64)
                tvt_true = hw['TVT'].to_numpy(np.float64)
                tvt_known = hw['TVT_input'].to_numpy(np.float64)
                pos_known = np.where(km.values, tvt_known, np.nan) + z
                curve_cache[w] = dict(xy=xy, z=z, pos_known=pos_known, km=km.values, tvt_true=tvt_true)
        return curve_cache[w]

    def get_tree(w):
        if w not in tree_cache:
            from scipy.spatial import cKDTree
            c = get_curve(w)
            tree_cache[w] = cKDTree(c['xy']) if c is not None else None
        return tree_cache[w]

    hw_t = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    km_t = hw_t['TVT_input'].notna()
    xy_t = hw_t[['X', 'Y']].to_numpy(np.float64)
    z_t = hw_t['Z'].to_numpy(np.float64)
    kn_idx = np.where(km_t.values)[0]
    ev_idx = np.where(~km_t.values)[0]

    i0 = cent_ids.index(wid)
    dist0, idx0 = cent_tree.query(cent_xy[i0], k=NB_K + 5)
    nn_wids = [cent_ids[j] for j in idx0 if cent_ids[j] != wid][:NB_K]

    kn_pos_t = hw_t.loc[km_t, 'TVT_input'].to_numpy(np.float64) + z_t[kn_idx]
    preds_by_neighbor = []
    for nn_wid in nn_wids:
        neighbor = get_curve(nn_wid)
        ntree = get_tree(nn_wid)
        if neighbor is None or ntree is None:
            continue
        neighbor_pos_full = neighbor['tvt_true'] + neighbor['z']
        dist_kn, idx_kn = ntree.query(xy_t[kn_idx], k=1)
        borrowed_kn = neighbor_pos_full[idx_kn]
        datum_offset = float(np.median(kn_pos_t - borrowed_kn))
        dist_ev, idx_ev = ntree.query(xy_t[ev_idx], k=1)
        borrowed_pos = neighbor_pos_full[idx_ev] + datum_offset
        pred_tvt = borrowed_pos - z_t[ev_idx]
        preds_by_neighbor.append((pred_tvt, dist_ev))

    preds_stack = np.stack([p for p, d in preds_by_neighbor], axis=1)
    dists_stack = np.stack([d for p, d in preds_by_neighbor], axis=1)
    w_idw = 1.0 / (dists_stack + 50.0)
    w_idw /= w_idw.sum(axis=1, keepdims=True)
    pred_idw = (preds_stack * w_idw).sum(axis=1)
    dist_k1 = preds_by_neighbor[0][1]
    conf = np.exp(-dist_k1 / NB_DIST_SCALE)
    md_ev = hw_t.loc[ev_idx, 'MD'].to_numpy(np.float64)
    return md_ev, pred_idw, conf


if __name__ == '__main__':
    proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
    gru7 = pickle.load(open('gru_v7_kfold_oof_preds.pkl', 'rb'))

    import glob, os
    train_wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
    cents = []
    for wid in train_wids:
        df = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv', usecols=['X', 'Y'])
        cents.append((wid, df.X.median(), df.Y.median()))
    cent_xy = np.array([[c[1], c[2]] for c in cents])
    cent_ids = [c[0] for c in cents]
    from scipy.spatial import cKDTree
    cent_tree = cKDTree(cent_xy)
    curve_cache, tree_cache = {}, {}

    for wid in REAL_TEST_WELLS:
        print(f'\n===== {wid} =====')
        px = proxy[wid]
        md_px, true_px = px['md'].astype(np.float64), px['true'].astype(np.float64)
        base_px = gru7[wid].astype(np.float64)
        print(f'  n_eval={len(true_px)}  base(gru7oof)_rmse={rmse(base_px, true_px):.3f}  sp45_rmse={rmse(px["sp45"], true_px):.3f}')

        hw, tw_tvt, tw_gr = load_hw_tw(wid)

        # --- pf_z (with std) ---
        pts, std = pfs.run_pf_z(hw, tw_tvt, tw_gr)
        ev = hw[hw.TVT_input.isna()]
        md_ev = ev.MD.values.astype(np.float64)
        true_ev = ev.TVT.values.astype(np.float64)
        pfz_on_px = np.interp(md_px, md_ev, pts)
        std_on_px = np.interp(md_px, md_ev, std)
        conf_pz = np.exp(-std_on_px / PZ_STD_SCALE)
        for w in [0.05, 0.15, 0.30]:
            eff = w * conf_pz
            blended = (1 - eff) * base_px + eff * pfz_on_px
            print(f'  pf_z  w={w:.2f} (mean_eff={eff.mean():.3f})  blended_rmse={rmse(blended, true_px):.3f}')

        # --- neighbor-transfer ---
        md_nb, pred_nb, conf_nb = get_neighbor_pred(wid, train_wids, cent_xy, cent_ids, cent_tree, curve_cache, tree_cache)
        pred_nb_px = np.interp(md_px, md_nb, pred_nb)
        conf_nb_px = np.interp(md_px, md_nb, conf_nb)
        for w in [0.05, 0.15, 0.30]:
            eff = w * conf_nb_px
            blended = (1 - eff) * base_px + eff * pred_nb_px
            print(f'  neigh w={w:.2f} (mean_eff={eff.mean():.3f})  blended_rmse={rmse(blended, true_px):.3f}')

        # --- beam2 solo ---
        r = b2.decode_well(wid)
        if r is not None:
            pred_b2, true_b2, md_b2 = r
            pred_b2_px = np.interp(md_px, md_b2, pred_b2)
            for w in [0.02, 0.05, 0.15]:
                blended = (1 - w) * base_px + w * pred_b2_px
                print(f'  beam2 w={w:.2f}  blended_rmse={rmse(blended, true_px):.3f}')
        else:
            print('  beam2: decode failed')

    print('\ndone')
