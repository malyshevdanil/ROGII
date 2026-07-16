"""Permutation importance on the already-trained GRU-refiner v8 checkpoints (no retraining needed):
for each fold's held-out validation wells, shuffle one feature column across the batch dimension and
measure how much the prediction RMSE (vs true sp45-residual target) degrades. A feature the network
actually relies on should show a real degradation; a feature that's just noise should show ~0 change.
Used to decide whether the 3 new disagreement features (sp45-pf_ancc, sp45-pf_z, pf_ancc-pf_z) are worth
keeping individually, rather than re-running full 5-fold training for each ablation.
"""
import numpy as np, pickle, torch, time
from sklearn.model_selection import GroupKFold
from gru_v8_bignet import build_dataset_v8, GRURefinerV8, N_FEAT, SEQ_LEN, DEV, CFG

FEAT_NAMES = ['md_since', 'gr', 'cal_gr', 'sm5', 'sm15', 'sm41', 'dog1', 'dog2', 'grad', 'rstd', 'gap',
              'z', 'dzdmd', 'fwd_mean', 'fwd_std',
              'tda-20', 'tda-10', 'tda-5', 'tda0', 'tda5', 'tda10', 'tda20', 'sin_azi', 'cos_azi',
              'sp45_minus_pfancc', 'sp45_minus_pfz', 'pfancc_minus_pfz']
assert len(FEAT_NAMES) == N_FEAT

if __name__ == '__main__':
    t0 = time.time()
    proxy = pickle.load(open('C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl', 'rb'))['DATA']
    sp45_lookup = {w: proxy[w]['sp45'] for w in proxy}
    DATA = build_dataset_v8(sp45_lookup, n_cuts=CFG['n_cuts'], seed=CFG['seed'])
    print(f'dataset rebuilt: {len(DATA)} entries ({time.time()-t0:.0f}s)')

    parents = sorted({d['parent'] for d in DATA.values()})
    groups_by_parent = {p: i for i, p in enumerate(parents)}
    keys = list(DATA.keys())
    group_ids = np.array([groups_by_parent[DATA[k]['parent']] for k in keys])
    gkf = GroupKFold(n_splits=CFG['n_folds'])
    dummy_X = np.zeros(len(keys))

    # check features to test: the 3 new ones, plus 2 known-important v7 features as a sanity baseline
    TEST_FEATS = ['sp45_minus_pfancc', 'sp45_minus_pfz', 'pfancc_minus_pfz', 'gr', 'cal_gr', 'fwd_mean']
    test_idx = {f: FEAT_NAMES.index(f) for f in TEST_FEATS}

    rng = np.random.default_rng(0)
    base_losses = []
    perm_losses = {f: [] for f in TEST_FEATS}

    for fold, (tr_i, va_i) in enumerate(gkf.split(dummy_X, groups=group_ids)):
        VA_keys = [keys[i] for i in va_i if DATA[keys[i]]['is_orig']]
        if not VA_keys: continue
        norm = pickle.load(open(f'gru_v8_norm_fold{fold}.pkl', 'rb'))
        mean, std = norm['mean'], norm['std']
        for seed_i in range(CFG['n_models_per_fold']):
            net = GRURefinerV8(N_FEAT, CFG['d'], CFG['layers'], CFG['drop']).to(DEV)
            net.load_state_dict(torch.load(f'gru_v8_fold{fold}_seed{seed_i}.pt', map_location=DEV))
            net.eval()

            Xb = np.stack([(DATA[k]['X'] - mean) / std for k in VA_keys])
            yb = np.stack([DATA[k]['y'] for k in VA_keys])
            with torch.no_grad():
                pred_base = net(torch.tensor(Xb, dtype=torch.float32, device=DEV)).cpu().numpy()
            base_loss = np.sqrt(np.mean((pred_base - yb) ** 2))
            base_losses.append(base_loss)

            for f in TEST_FEATS:
                ci = test_idx[f]
                Xp = Xb.copy()
                perm = rng.permutation(Xp.shape[0])
                Xp[:, :, ci] = Xp[perm, :, ci]
                with torch.no_grad():
                    pred_p = net(torch.tensor(Xp, dtype=torch.float32, device=DEV)).cpu().numpy()
                perm_loss = np.sqrt(np.mean((pred_p - yb) ** 2))
                perm_losses[f].append(perm_loss - base_loss)
        print(f'fold {fold} done  {time.time()-t0:.0f}s', flush=True)

    print(f'\nbaseline RMSE (mean over folds/seeds): {np.mean(base_losses):.4f}')
    print('\npermutation importance (RMSE increase when feature is shuffled -- higher = more important):')
    for f in TEST_FEATS:
        vals = np.array(perm_losses[f])
        print(f'  {f:22s}  mean_increase={vals.mean():+.4f}  std={vals.std():.4f}')
