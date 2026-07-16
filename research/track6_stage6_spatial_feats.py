"""Idea 1: add spatial nearest-neighbor formation-plane features to track6, matching STRIDE's actual
track6/track8 design (own PF + correlation features + spatial NN formation-plane features -> GBM). Our
base pipeline already has this exact component (FormationPlaneKNN, IDW-weighted local-plane fit of the
6 formation/contact columns ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA from nearby wells) -- ported here for the
773-well research context (self-well-excluded at fit time, per-well offset correction from the known
zone, same as production)."""
import numpy as np, pandas as pd, pickle, time
from scipy.spatial import cKDTree

TRAIN_DIR = 'd:/ROGII/data/train'
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10

class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            fp = f'{data_dir}/{wid}__horizontal_well.csv'
            try:
                cols = pd.read_csv(fp, nrows=0).columns
                use = [c for c in (["X", "Y"] + FORMATIONS) if c in cols]
                if "X" not in use or "Y" not in use: continue
                df = pd.read_csv(fp, usecols=use).dropna(subset=["X", "Y"])
            except Exception:
                continue
            if len(df) == 0: continue
            row = {"wid": wid, "x": float(df.X.median()), "y": float(df.Y.median())}
            for c in FORMATIONS:
                row[f"{c}_m"] = float(df[c].median()) if (c in df.columns and df[c].notna().any()) else np.nan
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df["wid"])}
        fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64).copy()
        col_mean = np.nanmean(fa, axis=0) if np.isfinite(fa).any() else np.zeros(fa.shape[1])
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        nanpos = np.where(np.isnan(fa)); fa[nanpos] = np.take(col_mean, nanpos[1])
        self.fa = fa
        xy = self.df[["x", "y"]].to_numpy(); self.scale = np.where(xy.std(0) < 1e-3, 1., xy.std(0))
        self.tree = cKDTree(xy / self.scale); self.xa = self.df.x.to_numpy(); self.ya = self.df.y.to_numpy()

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        xy_q = np.atleast_2d(np.asarray(xy_q, float))
        q = xy_q / self.scale; nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap: dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ordr = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1. / (dk + 1e-3), 0.).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]; wx = w * xn; wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]; A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]; A[:, 2, 1] = A[:, 1, 2]; A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9; A[:, 1, 1] += 1e-9; A[:, 2, 2] += 1e-9
        rhs = np.stack([(wx[:, :, None] * fn).sum(1), (wy[:, :, None] * fn).sum(1), (w[:, :, None] * fn).sum(1)], 1)
        try:
            coef = np.linalg.solve(A, rhs)
        except Exception:
            coef = np.zeros((len(q), 3, len(FORMATIONS)))
            for r in range(len(q)):
                try: coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception: pass
        Xq = xy_q[:, 0]; Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)

def seg_b_well(ktvt, kz, form_col):
    bv = ktvt + kz - form_col
    return float(np.median(bv)) if len(bv) else 0.0

if __name__ == '__main__':
    t0 = time.time()
    feat = pickle.load(open('track6_stage2_features.pkl', 'rb'))
    DATA2 = feat['DATA']; FEATCOLS = feat['FEATCOLS']
    WIDS = list(DATA2.keys())
    print('n wells:', len(WIDS))

    FPK = FormationPlaneKNN(WIDS, TRAIN_DIR)
    print('FormationPlaneKNN fit on', len(FPK.df), 'wells,', time.time() - t0, 's')

    new_feats = {}
    n_ok = 0
    for i, wid in enumerate(WIDS):
        hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
        km = hw['TVT_input'].notna()
        kn = hw[km]; ev = hw[~km]
        cols_present = [c for c in FORMATIONS if c in hw.columns]
        if len(cols_present) < len(FORMATIONS) or 'X' not in hw.columns:
            continue
        xy_kn = kn[['X', 'Y']].to_numpy(np.float64)
        xy_ev = ev[['X', 'Y']].to_numpy(np.float64)
        form_kn, _ = FPK.impute(xy_kn, self_wid=wid)
        form_ev, knn_d = FPK.impute(xy_ev, self_wid=wid)
        ktvt = kn['TVT_input'].to_numpy(np.float32); kz = kn['Z'].to_numpy(np.float32)
        z_ev = ev['Z'].to_numpy(np.float32)
        tvtF = {}
        for fi, fn in enumerate(FORMATIONS):
            b = seg_b_well(ktvt, kz, form_kn[:, fi])
            tvtF[fn] = (-z_ev + form_ev[:, fi] + b).astype(np.float32)
        M2 = np.stack([tvtF[fn] for fn in FORMATIONS] + [knn_d], axis=1).astype(np.float32)
        new_feats[wid] = M2
        n_ok += 1
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(WIDS)}  {time.time()-t0:.0f}s', flush=True)
    print(f'built spatial feats for {n_ok} wells  {time.time()-t0:.0f}s')

    NEW_COLS = [f'tvtF_{fn}' for fn in FORMATIONS] + ['knn_d']
    # merge into a new feature matrix aligned with track6_stage2's DATA (skip wells missing spatial feats)
    DATA3 = {}
    for wid in WIDS:
        if wid not in new_feats: continue
        M = DATA2[wid]['M']; M2 = new_feats[wid]
        if len(M) != len(M2): continue
        DATA3[wid] = dict(M=np.concatenate([M, M2], axis=1), true=DATA2[wid]['true'])
    print('wells with matched spatial+corr feats:', len(DATA3))
    pickle.dump(dict(DATA=DATA3, FEATCOLS=FEATCOLS + NEW_COLS), open('track6_stage6_features.pkl', 'wb'))
    print('saved track6_stage6_features.pkl')
    print(f'total time {time.time()-t0:.0f}s')
