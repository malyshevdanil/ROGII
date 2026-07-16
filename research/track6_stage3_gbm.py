"""Stage 3 of the track6 project: GBM predicting ABSOLUTE TVT position (not a residual from sp45 --
that's the critical architectural difference from the earlier failed gbm_track6_style.py attempt),
using the Stage 2 feature set (pf_ancc position + multi-scale correlation match features).
Proper 5-fold GroupKFold (grouped by well) so no well leaks across train/val.
"""
import numpy as np, pandas as pd, pickle, time
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

FEAT_PATH = 'track6_stage2_features.pkl'

if __name__ == '__main__':
    t0 = time.time()
    d = pickle.load(open(FEAT_PATH, 'rb'))
    DATA = d['DATA']; FEATCOLS = d['FEATCOLS']
    WIDS = list(DATA.keys())
    print('n wells:', len(WIDS))

    X_list, y_list, g_list, wid_row = [], [], [], []
    for wid in WIDS:
        M = DATA[wid]['M']; true = DATA[wid]['true']
        X_list.append(M); y_list.append(true)
        g_list.append(np.full(len(true), wid))
        wid_row.append(wid)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)
    print('X shape:', X.shape)

    gkf = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan)
    per_well_sq_gbm = {}
    per_well_sq_pfancc = {}

    pf_col = FEATCOLS.index('pf_ancc_tvt')

    fold_rmse_gbm = []
    fold_rmse_pfancc = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        model = lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.03, num_leaves=31,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=fold, verbose=-1)
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[va_idx])
        oof[va_idx] = pred
        rmse_gbm = np.sqrt(np.mean((pred - y[va_idx])**2))
        rmse_pfancc = np.sqrt(np.mean((X[va_idx, pf_col] - y[va_idx])**2))
        fold_rmse_gbm.append(rmse_gbm); fold_rmse_pfancc.append(rmse_pfancc)
        print(f'fold {fold}: gbm={rmse_gbm:.4f}  pf_ancc={rmse_pfancc:.4f}  n_val={len(va_idx)}', flush=True)

    pooled_gbm = np.sqrt(np.mean((oof - y)**2))
    pooled_pfancc = np.sqrt(np.mean((X[:, pf_col] - y)**2))
    print(f'\npooled RMSE: gbm={pooled_gbm:.4f}  pf_ancc={pooled_pfancc:.4f}')
    print(f'fold RMSEs gbm: {fold_rmse_gbm}')
    print(f'fold RMSEs pf_ancc: {fold_rmse_pfancc}')

    # per-well sq errors for downstream paired bootstrap
    per_well_sq_gbm = {}
    per_well_sq_pfancc = {}
    start = 0
    for wid in WIDS:
        n = len(DATA[wid]['true'])
        sl = slice(start, start + n)
        per_well_sq_gbm[wid] = (oof[sl] - y[sl])**2
        per_well_sq_pfancc[wid] = (X[sl, pf_col] - y[sl])**2
        start += n

    pickle.dump(dict(oof=oof, y=y, groups=groups, WIDS=WIDS,
                      per_well_sq_gbm=per_well_sq_gbm, per_well_sq_pfancc=per_well_sq_pfancc,
                      pooled_gbm=pooled_gbm, pooled_pfancc=pooled_pfancc),
                open('track6_stage3_oof.pkl', 'wb'))
    print('saved track6_stage3_oof.pkl')
    print(f'total time {time.time()-t0:.0f}s')
