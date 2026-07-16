"""v8: backlog item 1 from the competitor-discussion ideas (rogii_competitor_ideas_backlog.md) --
add uncertainty/disagreement features to the GRU-refiner, matching the pattern that already gave track6
a real validated gain (pf_ancc-vs-pf_z disagreement, corr(pf_z_err,pf_ancc_err)=0.49 on the true
holdout -- a genuinely different failure mode from the same GR data). v7's 24 features have no PF-derived
signal at all; add 3 new columns: (sp45 - pf_ancc), (sp45 - pf_z), (pf_ancc - pf_z), using the already-
cached track6_pfancc_all773.pkl / pfz_all773.pkl (both aligned to proxy's md grid = hw's own eval rows,
already verified this session). Everything else (architecture, prefix-cut augmentation, 5-fold GroupKFold,
training loop) is unchanged from v7 -- isolating the effect of just these 3 new features.
"""
import numpy as np, pandas as pd, pickle, torch, torch.nn as nn, torch.nn.functional as F, time, glob, os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from gru_v7_bignet import (DEV, SEQ_LEN, TRAIN_DIR, CFG as CFG_V7, interp_nan, build_well_raw,
                             feats_for_cut as feats_for_cut_v7, GRURefinerV7)

CFG = dict(CFG_V7)
N_FEAT = 26  # 24 (v7) + 2 disagreement features (dropped pfancc-pfz, permutation importance +0.15 vs +0.65/+0.63 for the other two)

PF_ANCC_ALL = pickle.load(open('track6_pfancc_all773.pkl', 'rb'))
PF_Z_ALL = pickle.load(open('pfz_all773.pkl', 'rb'))

def build_dataset_v9(sp45_lookup, n_cuts=2, seed=0):
    rng = np.random.default_rng(seed)
    wids = [w for w in sp45_lookup.keys() if w in PF_ANCC_ALL and w in PF_Z_ALL]
    print(f'wells with sp45+pf_ancc+pf_z: {len(wids)} / {len(sp45_lookup)}')
    DATA = {}
    for wid in wids:
        r = build_well_raw(wid)
        if r is None: continue
        hw, tw_tvt, tw_gr, km = r
        ev_mask = hw['TVT_input'].isna().values
        n_known = int(km.sum())
        if n_known < 60: continue
        sp45 = sp45_lookup[wid]; pf_ancc = PF_ANCC_ALL[wid]; pf_z = PF_Z_ALL[wid]
        n_ev = ev_mask.sum()
        if len(sp45) != n_ev or len(pf_ancc) != n_ev or len(pf_z) != n_ev: continue
        disagree = np.stack([sp45 - pf_ancc, sp45 - pf_z], axis=1).astype(np.float32)
        cuts = [1.0]
        for _ in range(n_cuts):
            f = rng.uniform(0.4, 0.9)
            cuts.append(f)
        for ci, cut_frac in enumerate(cuts):
            X, last_tvt = feats_for_cut_v7(hw, tw_tvt, tw_gr, km, cut_frac)
            Xev = X[ev_mask]
            n = Xev.shape[0]
            if n != n_ev: continue
            Xev_full = np.concatenate([Xev, disagree], axis=1)  # (n, 26)
            true = hw['TVT'].values.astype(float)[ev_mask]
            y = true - sp45
            src = np.arange(n); dst = np.linspace(0, n - 1, SEQ_LEN)
            Xr = np.stack([np.interp(dst, src, Xev_full[:, c]) for c in range(N_FEAT)], axis=1).astype(np.float32)
            yr = np.interp(dst, src, y).astype(np.float32)
            key = wid if ci == 0 else f'{wid}__cut{ci}'
            DATA[key] = dict(X=Xr, y=yr, n=n, src_true=true, src_sp45=sp45, parent=wid, is_orig=(ci == 0))
    return DATA

class GRURefinerV9(GRURefinerV7):
    pass  # identical architecture, just a bigger input dim (N_FEAT=27 instead of 24)

def normalize_features(DATA, train_keys):
    Xall = np.concatenate([DATA[k]['X'] for k in train_keys], axis=0)
    mean = Xall.mean(0); std = Xall.std(0) + 1e-6
    return mean, std

def train_one_fold(DATA, TR_keys, VA_keys, seed, save_tag=None):
    mean, std = normalize_features(DATA, TR_keys)
    net = GRURefinerV9(N_FEAT, CFG['d'], CFG['layers'], CFG['drop']).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['epochs'])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    def get_batch(keys, bs):
        idx = rng.choice(len(keys), size=min(bs, len(keys)), replace=False)
        Xb = np.stack([(DATA[keys[i]]['X'] - mean) / std for i in idx])
        yb = np.stack([DATA[keys[i]]['y'] for i in idx])
        if CFG['noise_std'] > 0:
            Xb = Xb + rng.normal(0, CFG['noise_std'], Xb.shape).astype(np.float32)
        return torch.tensor(Xb, dtype=torch.float32, device=DEV), torch.tensor(yb, dtype=torch.float32, device=DEV)

    n_steps = max(1, len(TR_keys) // CFG['bs'])
    for ep in range(CFG['epochs']):
        net.train()
        for _ in range(n_steps):
            Xb, yb = get_batch(TR_keys, CFG['bs'])
            pred = net(Xb)
            loss = F.smooth_l1_loss(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()

    net.eval()
    VA_orig = [k for k in VA_keys if DATA[k]['is_orig']]
    preds = {}
    with torch.no_grad():
        for k in VA_orig:
            d = DATA[k]
            Xn = (d['X'] - mean) / std
            Xt = torch.tensor(Xn[None], dtype=torch.float32, device=DEV)
            r = net(Xt)[0].cpu().numpy()
            n = d['n']; src_dst = np.linspace(0, n - 1, SEQ_LEN)
            pred_full = np.interp(np.arange(n), src_dst, r)
            preds[k] = d['src_sp45'] + pred_full
    if save_tag:
        torch.save(net.state_dict(), f'{save_tag}.pt')
        pickle.dump(dict(mean=mean, std=std), open(f'{save_tag}_norm.pkl', 'wb'))
    return preds, net, mean, std

if __name__ == '__main__':
    from sklearn.model_selection import GroupKFold
    t0 = time.time()
    proxy = pickle.load(open('C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl', 'rb'))['DATA']
    sp45_lookup = {w: proxy[w]['sp45'] for w in proxy}
    DATA = build_dataset_v9(sp45_lookup, n_cuts=CFG['n_cuts'], seed=CFG['seed'])
    print(f'built dataset: {len(DATA)} entries ({time.time()-t0:.0f}s)')

    parents = sorted({d['parent'] for d in DATA.values()})
    groups_by_parent = {p: i for i, p in enumerate(parents)}
    keys = list(DATA.keys())
    group_ids = np.array([groups_by_parent[DATA[k]['parent']] for k in keys])

    gkf = GroupKFold(n_splits=CFG['n_folds'])
    dummy_X = np.zeros(len(keys))
    all_oof = {}
    for fold, (tr_i, va_i) in enumerate(gkf.split(dummy_X, groups=group_ids)):
        TR_keys = [keys[i] for i in tr_i]
        VA_keys = [keys[i] for i in va_i]
        fold_preds_list = []
        for seed_i in range(CFG['n_models_per_fold']):
            seed = CFG['seed'] * 100 + fold * 10 + seed_i
            preds, net, mean, std = train_one_fold(DATA, TR_keys, VA_keys, seed,
                                                     save_tag=f'gru_v9_fold{fold}_seed{seed_i}')
            fold_preds_list.append(preds)
        merged = {}
        for k in fold_preds_list[0]:
            merged[k] = np.mean([fp[k] for fp in fold_preds_list], axis=0)
        all_oof.update(merged)
        rmse_fold = np.sqrt(np.mean(np.concatenate([(merged[k] - DATA[k]['src_true'])**2 for k in merged])))
        print(f'fold {fold}: val rmse={rmse_fold:.4f}  n_val={len(merged)}  {time.time()-t0:.0f}s', flush=True)
        pickle.dump(dict(mean=mean, std=std), open(f'gru_v9_norm_fold{fold}.pkl', 'wb'))

    pooled = np.sqrt(np.mean(np.concatenate([(all_oof[k] - DATA[k]['src_true'])**2 for k in all_oof])))
    print(f'\npooled OOF RMSE (v9, +2 disagreement feats): {pooled:.4f}')
    pickle.dump(all_oof, open('gru_v9_kfold_oof_preds.pkl', 'wb'))
    print(f'saved gru_v9_kfold_oof_preds.pkl, total time {time.time()-t0:.0f}s')
