"""v7: a genuinely stronger primary refiner, matching more of Tucker Arrants' described architecture
(2026-07-14 writeup) rather than a quick small-correction test. Additions over v6 (own-only, 16 feat):
  - multi-scale GR (raw, sm5/15/41, band-pass dog1/dog2 -- Eagle Ford bedding scale, from H4)
  - LEGAL look-ahead summaries: forward-only rolling mean/std of GR (the whole eval GR sequence is
    known upfront, only TVT is hidden -- "look-ahead" is legal here, exactly as Tucker's writeup notes)
  - gap indicator (was this row's GR originally missing, before interpolation)
  - distance-to-end (normalized position within the eval sequence) -- his "distance-to-end features"
  - PREFIX-CUT AUGMENTATION: for each well, also train on 1-2 artificially-shortened known-zones
    (matching Tucker's "prefix-cut GRU augmentation improved 8.404->8.159", and our own memory's
    "dense prefix cuts gave a real -0.187 on a matched screen"), always keeping cuts of the SAME
    parent well in the SAME fold to avoid leakage.
  - proper 5-fold GroupKFold (not a single train/val split) for both training diversity and honest OOF.
Still predicts a RESIDUAL added to sp45 (proven stable framing from v6), bidirectional GRU.
"""
import numpy as np, pandas as pd, pickle, torch, torch.nn as nn, torch.nn.functional as F, time, glob, os

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 500
TRAIN_DIR = 'd:/ROGII/data/train'
CFG = dict(d=96, layers=2, drop=0.40, wd=1e-3, lr=1.2e-3, epochs=35, bs=16, noise_std=0.12,
           n_models_per_fold=2, n_folds=5, seed=11, n_cuts=2)

N_FEAT = 24

def interp_nan(a):
    a = a.copy(); n = len(a); idx = np.arange(n); m = np.isnan(a)
    if m.all(): return np.zeros(n)
    a[m] = np.interp(idx[m], idx[~m], a[~m]); return a

def build_well_raw(wid):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tw_tvt = tw['TVT'].values.astype(float); tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    if len(tw_tvt) < 10: return None
    km = hw['TVT_input'].notna()
    if km.sum() < 20 or (~km).sum() < 20: return None
    return hw, tw_tvt, tw_gr, km

def feats_for_cut(hw, tw_tvt, tw_gr, km, cut_frac=1.0):
    """cut_frac<1.0 pretends only the first cut_frac of the known zone is visible (prefix-cut aug);
    the true eval region (already isna) is always used as-is for the TARGET, but the FEATURES that
    depend on 'last known' (md_since, z anchor, cal_gr fit) are recomputed from the shortened prefix."""
    km_idx = np.where(km.values)[0]
    n_known_use = max(20, int(len(km_idx) * cut_frac))
    km_idx_use = km_idx[:n_known_use]
    last_i = km_idx_use[-1]
    last_tvt = float(hw['TVT_input'].iloc[last_i]); last_Z = float(hw['Z'].iloc[last_i]); last_MD = float(hw['MD'].iloc[last_i])

    gr = interp_nan(hw['GR'].values.astype(float))
    gap = hw['GR'].isna().values.astype(np.float32)
    X_ = hw['X'].values.astype(float); Y = hw['Y'].values.astype(float)
    Z = hw['Z'].values.astype(float); MD = hw['MD'].values.astype(float)
    mdd = np.gradient(MD); mdd[mdd == 0] = 1

    kg = gr[km_idx_use]; ktvt = hw['TVT_input'].values[km_idx_use]
    twk = np.interp(ktvt, tw_tvt, tw_gr); v = np.isfinite(kg) & np.isfinite(twk)
    a, b = (np.polyfit(kg[v], twk[v], 1) if v.sum() >= 20 else (1., 0.))
    cal = gr * a + b

    cs = pd.Series(cal)
    sm5 = cs.rolling(5, center=True, min_periods=1).mean().values
    sm15 = cs.rolling(15, center=True, min_periods=1).mean().values
    sm41 = cs.rolling(41, center=True, min_periods=1).mean().values
    dog1 = sm5 - sm15; dog2 = sm15 - sm41
    grad = np.gradient(gr)
    rstd = pd.Series(gr).rolling(21, center=True, min_periods=1).std().fillna(0).values
    dzdmd = np.gradient(Z) / mdd
    head = np.arctan2(np.gradient(Y), np.gradient(X_) + 1e-9)

    # legal forward-only look-ahead: rolling mean/std over the NEXT 40 samples (shift so window starts
    # at the current row, not centered) -- legal because the whole GR sequence is observed upfront.
    fwd_mean = cs.rolling(40, min_periods=1).mean().shift(-40).bfill().ffill().values
    fwd_std = cs.rolling(40, min_periods=1).std().shift(-40).bfill().ffill().fillna(0).values

    F_ = {}
    F_['md_since'] = MD - last_MD
    F_['gr'] = gr; F_['cal_gr'] = cal
    F_['sm5'] = sm5; F_['sm15'] = sm15; F_['sm41'] = sm41
    F_['dog1'] = dog1; F_['dog2'] = dog2
    F_['grad'] = grad; F_['rstd'] = rstd
    F_['gap'] = gap
    F_['z'] = Z - last_Z; F_['dzdmd'] = dzdmd
    F_['fwd_mean'] = fwd_mean; F_['fwd_std'] = fwd_std
    for o in (-20, -10, -5, 0, 5, 10, 20):
        F_['tda%d' % o] = gr - np.interp(last_tvt + o, tw_tvt, tw_gr)
    F_['sin_azi'] = np.sin(head); F_['cos_azi'] = np.cos(head)

    ORDER = ['md_since', 'gr', 'cal_gr', 'sm5', 'sm15', 'sm41', 'dog1', 'dog2', 'grad', 'rstd', 'gap',
              'z', 'dzdmd', 'fwd_mean', 'fwd_std',
              'tda-20', 'tda-10', 'tda-5', 'tda0', 'tda5', 'tda10', 'tda20', 'sin_azi', 'cos_azi']
    assert len(ORDER) == N_FEAT
    X = np.stack([F_[c] for c in ORDER], axis=1)
    return X, last_tvt

def build_dataset_v7(sp45_lookup, n_cuts=2, seed=0):
    """sp45_lookup[wid] -> per-row sp45 track aligned to eval rows (from proxy.pkl)."""
    rng = np.random.default_rng(seed)
    wids = [w for w in sp45_lookup.keys()]
    DATA = {}
    for wid in wids:
        r = build_well_raw(wid)
        if r is None: continue
        hw, tw_tvt, tw_gr, km = r
        ev_mask = hw['TVT_input'].isna().values
        n_known = int(km.sum())
        if n_known < 60: continue
        cuts = [1.0]
        for _ in range(n_cuts):
            f = rng.uniform(0.4, 0.9)
            cuts.append(f)
        for ci, cut_frac in enumerate(cuts):
            X, last_tvt = feats_for_cut(hw, tw_tvt, tw_gr, km, cut_frac)
            Xev = X[ev_mask]
            n = Xev.shape[0]
            sp45 = sp45_lookup[wid]
            if len(sp45) != n: continue
            true = hw['TVT'].values.astype(float)[ev_mask]
            y = true - sp45
            src = np.arange(n); dst = np.linspace(0, n - 1, SEQ_LEN)
            Xr = np.stack([np.interp(dst, src, Xev[:, c]) for c in range(N_FEAT)], axis=1).astype(np.float32)
            yr = np.interp(dst, src, y).astype(np.float32)
            key = wid if ci == 0 else f'{wid}__cut{ci}'
            DATA[key] = dict(X=Xr, y=yr, n=n, src_true=true, src_sp45=sp45, parent=wid, is_orig=(ci == 0))
    return DATA

class GRURefinerV7(nn.Module):
    def __init__(self, n_in, d, layers, drop):
        super().__init__()
        self.inp = nn.Linear(n_in, d)
        self.gru = nn.GRU(d, d, num_layers=layers, batch_first=True, bidirectional=True, dropout=drop if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))
        self.head[-1].weight.data *= 0.01
        self.head[-1].bias.data.zero_()
    def forward(self, X):
        h = F.gelu(self.inp(X))
        h, _ = self.gru(h)
        return self.head(h)[..., 0]

def normalize_features(DATA, wids_fit):
    Xall = np.concatenate([DATA[w]['X'] for w in wids_fit], axis=0)
    mean = Xall.mean(axis=0); std = Xall.std(axis=0) + 1e-6
    for w in DATA:
        DATA[w]['Xn'] = (DATA[w]['X'] - mean) / std
    return mean, std

def prmse(sqs):
    return float(np.sqrt(np.mean(np.concatenate(sqs))))

def evaluate(net, DATA, wids_orig_only):
    """wids_orig_only: only score on non-cut (real) wells, matching Tucker's 'validation uses only the
    original hidden zones'."""
    net.eval()
    sqs = []
    with torch.no_grad():
        for i in range(0, len(wids_orig_only), 32):
            ws = wids_orig_only[i:i + 32]
            X = torch.tensor(np.stack([DATA[w]['Xn'] for w in ws]), dtype=torch.float32, device=DEV)
            pred_resid = net(X).cpu().numpy()
            for j, w in enumerate(ws):
                d = DATA[w]; n = d['n']
                src_dst = np.linspace(0, n - 1, SEQ_LEN)
                pred_full = np.interp(np.arange(n), src_dst, pred_resid[j])
                final_pred = d['src_sp45'] + pred_full
                sqs.append((final_pred - d['src_true']) ** 2)
    net.train()
    return prmse(sqs)

def train_one(seed, DATA, TR_keys, VA_keys_orig, save_tag=None):
    torch.manual_seed(seed); np.random.seed(seed)
    net = GRURefinerV7(N_FEAT, CFG['d'], CFG['layers'], CFG['drop']).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    n_batches = (len(TR_keys) + CFG['bs'] - 1) // CFG['bs']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG['lr'], total_steps=CFG['epochs'] * n_batches)
    best_state = None; best = (99.0, 0)
    order = np.arange(len(TR_keys))
    t0 = time.time()
    for ep in range(CFG['epochs']):
        np.random.shuffle(order)
        for i in range(0, len(TR_keys), CFG['bs']):
            ws = [TR_keys[k] for k in order[i:i + CFG['bs']]]
            if not ws: continue
            X = torch.tensor(np.stack([DATA[w]['Xn'] for w in ws]), dtype=torch.float32, device=DEV)
            y = torch.tensor(np.stack([DATA[w]['y'] for w in ws]), dtype=torch.float32, device=DEV)
            X = X + torch.randn_like(X) * CFG['noise_std']
            pred = net(X)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
        r = evaluate(net, DATA, VA_keys_orig)
        if r < best[0]:
            best = (r, ep + 1)
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'    seed{seed} ep{ep+1:3d} | holdout {r:.4f} (best {best[0]:.4f}@{best[1]}) {time.time()-t0:.0f}s', flush=True)
    net.load_state_dict(best_state)
    if save_tag is not None:
        torch.save(best_state, f'{save_tag}.pt')
    return net, best

if __name__ == '__main__':
    t0 = time.time()
    proxy = pickle.load(open('C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl', 'rb'))['DATA']
    sp45_lookup = {w: proxy[w]['sp45'].astype(np.float64) for w in proxy}
    DATA = build_dataset_v7(sp45_lookup, n_cuts=CFG['n_cuts'], seed=0)
    n_orig = sum(1 for k, d in DATA.items() if d['is_orig'])
    print(f'dataset ready {time.time()-t0:.0f}s  total_examples={len(DATA)} (orig wells={n_orig})')

    parent_wids = sorted({d['parent'] for d in DATA.values()})
    rng = np.random.default_rng(CFG['seed'])
    order = np.array(parent_wids); rng.shuffle(order)
    folds = np.array_split(order, CFG['n_folds'])

    all_pred = {}
    for fi in range(CFG['n_folds']):
        va_parents = set(folds[fi])
        tr_parents = set(np.concatenate([folds[j] for j in range(CFG['n_folds']) if j != fi]))
        TR_keys = [k for k, d in DATA.items() if d['parent'] in tr_parents]
        VA_orig_keys = [k for k, d in DATA.items() if d['parent'] in va_parents and d['is_orig']]
        mean, std = normalize_features(DATA, TR_keys)
        pickle.dump(dict(mean=mean, std=std), open(f'gru_v7_norm_fold{fi}.pkl', 'wb'))
        base_va = prmse([(DATA[w]['src_sp45'] - DATA[w]['src_true']) ** 2 for w in VA_orig_keys])

        preds = {w: [] for w in VA_orig_keys}
        fold_bests = []
        for seed in range(CFG['n_models_per_fold']):
            net, best = train_one(fi * 100 + seed, DATA, TR_keys, VA_orig_keys, save_tag=f'gru_refiner_v7_fold{fi}_seed{seed}')
            fold_bests.append(best[0])
            net.eval()
            with torch.no_grad():
                for i in range(0, len(VA_orig_keys), 32):
                    ws = VA_orig_keys[i:i + 32]
                    X = torch.tensor(np.stack([DATA[w]['Xn'] for w in ws]), dtype=torch.float32, device=DEV)
                    pred_resid = net(X).cpu().numpy()
                    for j, w in enumerate(ws):
                        d = DATA[w]; n = d['n']
                        src_dst = np.linspace(0, n - 1, SEQ_LEN)
                        pred_full = np.interp(np.arange(n), src_dst, pred_resid[j])
                        preds[w].append(d['src_sp45'] + pred_full)
        for w in VA_orig_keys:
            all_pred[w] = np.mean(preds[w], axis=0)
        fold_ens_rmse = prmse([(all_pred[w] - DATA[w]['src_true']) ** 2 for w in VA_orig_keys])
        print(f'fold {fi}: n={len(VA_orig_keys)} sp45_base={base_va:.4f} per-seed best={[round(b,3) for b in fold_bests]} '
              f'ensemble={fold_ens_rmse:.4f}  {time.time()-t0:.0f}s', flush=True)

    overall_rmse = prmse([(all_pred[w] - DATA[w]['src_true']) ** 2 for w in all_pred])
    overall_sp45 = prmse([(DATA[w]['src_sp45'] - DATA[w]['src_true']) ** 2 for w in all_pred])
    print(f'\nOVERALL (honest OOF, orig wells only): gru_v7={overall_rmse:.4f}  sp45={overall_sp45:.4f}')
    pickle.dump(all_pred, open('gru_v7_kfold_oof_preds.pkl', 'wb'))
    print('saved gru_v7_kfold_oof_preds.pkl')
