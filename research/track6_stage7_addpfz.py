"""Add pf_z (Z-velocity-coupled PF, different motion model than pf_ancc) as two new track6 features:
its own TVT estimate, and its disagreement with pf_ancc (a self-contained instability signal, not
dependent on any external tracker like sp45/WARP). pf_z showed low correlation with both pf_ancc
(0.49) and the current best blend (0.51) on the true holdout -- a genuinely different failure mode,
worth testing as extra GBM signal."""
import numpy as np, pickle, time

if __name__ == '__main__':
    t0 = time.time()
    feat = pickle.load(open('track6_stage6_features.pkl', 'rb'))
    DATA6 = feat['DATA']; FEATCOLS6 = feat['FEATCOLS']
    pfz_all = pickle.load(open('pfz_all773.pkl', 'rb'))
    pf_col = FEATCOLS6.index('pf_ancc_tvt')

    DATA7 = {}
    n_ok = 0
    for wid, d in DATA6.items():
        if wid not in pfz_all: continue
        M = d['M']; true = d['true']
        pfz = pfz_all[wid]
        if len(pfz) != len(true): continue
        pf_ancc_col = M[:, pf_col]
        disagree = pfz - pf_ancc_col
        M2 = np.concatenate([M, pfz.reshape(-1, 1).astype(np.float32), disagree.reshape(-1, 1).astype(np.float32)], axis=1)
        DATA7[wid] = dict(M=M2, true=true)
        n_ok += 1
    print('wells with pf_z matched:', n_ok, '/', len(DATA6))
    FEATCOLS7 = FEATCOLS6 + ['pf_z_tvt', 'pfz_ancc_disagree']
    pickle.dump(dict(DATA=DATA7, FEATCOLS=FEATCOLS7), open('track6_stage7_features.pkl', 'wb'))
    print('saved track6_stage7_features.pkl,', time.time() - t0, 's')
