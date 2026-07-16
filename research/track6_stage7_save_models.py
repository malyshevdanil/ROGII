"""Stage 5: train and SAVE the actual 5 fold GBM models (single seed -- Stage 3c's ensemble showed
ensembling added nothing over a single well-regularized model) for deployment, matching the fold split
used for OOF validation exactly. Also saves the pf_ancc_tvt column index and FEATCOLS for the inference
cell to reconstruct features identically."""
import numpy as np, pickle, time
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
    models = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.02, num_leaves=15,
            min_child_samples=200, subsample=0.7, colsample_bytree=0.6,
            reg_lambda=10.0, reg_alpha=1.0, random_state=fold, verbose=-1)
        model.fit(X[tr_idx], resid[tr_idx])
        models.append(model)
        pred_abs = X[va_idx, pf_col] + model.predict(X[va_idx])
        rmse = np.sqrt(np.mean((pred_abs - y[va_idx])**2))
        print(f'fold {fold}: val rmse={rmse:.4f}  {time.time()-t0:.0f}s', flush=True)
        model.booster_.save_model(f'track6v3_gbm_fold{fold}.txt')

    pickle.dump(dict(FEATCOLS=FEATCOLS, pf_col=pf_col), open('track6v3_gbm_meta.pkl', 'wb'))
    print('saved 5 fold models (track6v3_gbm_fold{0-4}.txt) + track6v3_gbm_meta.pkl')
    print(f'total time {time.time()-t0:.0f}s')
