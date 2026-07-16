"""Stage 3b: fixed version of the track6 GBM combiner. Diagnosis of the first attempt (predicting
ABSOLUTE TVT position): train/val RMSE gap (11.9 vs 25.8 on fold 0) showed severe overfitting -- per-row
independent trees with no smoothness prior produced predictions 30x jumpier than the true trajectory
(row-to-row std 2.55 vs true 0.013), despite pf_ancc_tvt being the dominant input feature (importance
6601, corr 0.9993 with output) -- the model was overfitting azimuth/position quirks of individual
training wells rather than learning a generalizable correction.
Fix: predict the RESIDUAL (true - pf_ancc_tvt) instead of absolute position (much smaller target scale,
less room for the tree ensemble to swing wildly), with much heavier regularization (fewer/shallower
trees, larger min_child_samples, stronger L1/L2).
"""
import numpy as np, pandas as pd, pickle, time
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

FEAT_PATH = 'track6_stage7_features.pkl'

if __name__ == '__main__':
    t0 = time.time()
    d = pickle.load(open(FEAT_PATH, 'rb'))
    DATA = d['DATA']; FEATCOLS = d['FEATCOLS']
    WIDS = list(DATA.keys())

    X_list, y_list, g_list = [], [], []
    for wid in WIDS:
        M = DATA[wid]['M']; true = DATA[wid]['true']
        X_list.append(M); y_list.append(true); g_list.append(np.full(len(true), wid))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)
    pf_col = FEATCOLS.index('pf_ancc_tvt')
    resid = y - X[:, pf_col]

    gkf = GroupKFold(n_splits=5)
    oof_abs = np.full(len(y), np.nan)
    fold_rmse_gbm, fold_rmse_pfancc = [], []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.02, num_leaves=15,
            min_child_samples=200, subsample=0.7, colsample_bytree=0.6,
            reg_lambda=10.0, reg_alpha=1.0, random_state=fold, verbose=-1)
        model.fit(X[tr_idx], resid[tr_idx])
        pred_resid = model.predict(X[va_idx])
        pred_abs = X[va_idx, pf_col] + pred_resid
        oof_abs[va_idx] = pred_abs
        rmse_gbm = np.sqrt(np.mean((pred_abs - y[va_idx])**2))
        rmse_pfancc = np.sqrt(np.mean((X[va_idx, pf_col] - y[va_idx])**2))
        fold_rmse_gbm.append(rmse_gbm); fold_rmse_pfancc.append(rmse_pfancc)
        print(f'fold {fold}: gbm_resid={rmse_gbm:.4f}  pf_ancc={rmse_pfancc:.4f}  n_val={len(va_idx)}', flush=True)

    pooled_gbm = np.sqrt(np.mean((oof_abs - y)**2))
    pooled_pfancc = np.sqrt(np.mean((X[:, pf_col] - y)**2))
    print(f'\npooled RMSE: gbm_resid={pooled_gbm:.4f}  pf_ancc={pooled_pfancc:.4f}')
    print(f'fold RMSEs gbm_resid: {fold_rmse_gbm}')
    print(f'fold RMSEs pf_ancc: {fold_rmse_pfancc}')

    per_well_sq_gbm, per_well_sq_pfancc = {}, {}
    start = 0
    for wid in WIDS:
        n = len(DATA[wid]['true'])
        sl = slice(start, start + n)
        per_well_sq_gbm[wid] = (oof_abs[sl] - y[sl])**2
        per_well_sq_pfancc[wid] = (X[sl, pf_col] - y[sl])**2
        start += n

    pickle.dump(dict(oof_abs=oof_abs, y=y, groups=groups, WIDS=WIDS,
                      per_well_sq_gbm=per_well_sq_gbm, per_well_sq_pfancc=per_well_sq_pfancc,
                      pooled_gbm=pooled_gbm, pooled_pfancc=pooled_pfancc),
                open('track6_stage7_oof.pkl', 'wb'))
    print('saved track6_stage3b_oof.pkl')
    print(f'total time {time.time()-t0:.0f}s')
