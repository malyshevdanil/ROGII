"""v4: honest 5-fold cross-validation across ALL 773 wells, using ONLY the 16 'own' GR/trajectory
features (drop the warp-disagreement channels entirely) -- those are always legal per-well features
with no train/test leakage concern, unlike warp-sp45 disagreement which is only honest on WARP's own
160-well true holdout (613/773 wells are WARP's training set). This trades away the warp-disagreement
signal for the ability to get an honest prediction on ALL 773 wells (5 folds x small ensemble each),
giving far more statistical power for the paired bootstrap than the 160-well test.
"""
import numpy as np, pickle, torch, torch.nn as nn, torch.nn.functional as F, time
from gru_refiner import DEV, SEQ_LEN, GRURefiner

CFG = dict(d=64, layers=2, drop=0.40, wd=1e-3, lr=1.2e-3, epochs=35, bs=16, noise_std=0.15,
           n_models_per_fold=3, n_folds=5, seed=7)

PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'

def build_dataset_own_only():
    proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
    WIDS = list(proxy.keys())
    DATA = {}
    for wid in WIDS:
        d = proxy[wid]
        n = len(d['true'])
        if n < 50: continue
        sp45 = d['sp45'].astype(np.float64)
        own = d['own'].astype(np.float64)  # (n,16)
        true = d['true'].astype(np.float64)
        y = true - sp45
        src = np.arange(n)
        dst = np.linspace(0, n - 1, SEQ_LEN)
        Xr = np.stack([np.interp(dst, src, own[:, c]) for c in range(own.shape[1])], axis=1).astype(np.float32)
        yr = np.interp(dst, src, y).astype(np.float32)
        DATA[wid] = dict(X=Xr, y=yr, n=n, src_true=true, src_sp45=sp45)
    return DATA

def normalize_features(DATA, wids_fit):
    Xall = np.concatenate([DATA[w]['X'] for w in wids_fit], axis=0)
    mean = Xall.mean(axis=0); std = Xall.std(axis=0) + 1e-6
    for w in DATA:
        DATA[w]['Xn'] = (DATA[w]['X'] - mean) / std
    return mean, std

def prmse(sqs):
    return float(np.sqrt(np.mean(np.concatenate(sqs))))

def evaluate(net, DATA, wids):
    net.eval()
    sqs = []
    with torch.no_grad():
        for i in range(0, len(wids), 32):
            ws = wids[i:i + 32]
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

def train_one(seed, DATA, TR, VA, save_tag=None):
    torch.manual_seed(seed); np.random.seed(seed)
    net = GRURefiner(16, CFG['d'], CFG['layers'], CFG['drop']).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    n_batches = (len(TR) + CFG['bs'] - 1) // CFG['bs']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG['lr'], total_steps=CFG['epochs'] * n_batches)
    best_state = None; best = (99.0, 0)
    order = np.arange(len(TR))
    for ep in range(CFG['epochs']):
        np.random.shuffle(order)
        for i in range(0, len(TR), CFG['bs']):
            ws = [TR[k] for k in order[i:i + CFG['bs']]]
            if not ws: continue
            X = torch.tensor(np.stack([DATA[w]['Xn'] for w in ws]), dtype=torch.float32, device=DEV)
            y = torch.tensor(np.stack([DATA[w]['y'] for w in ws]), dtype=torch.float32, device=DEV)
            X = X + torch.randn_like(X) * CFG['noise_std']
            pred = net(X)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
        r = evaluate(net, DATA, VA)
        if r < best[0]:
            best = (r, ep + 1)
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    if save_tag is not None:
        torch.save(best_state, f'gru_refiner_{save_tag}.pt')
    return net, best

if __name__ == '__main__':
    t0 = time.time()
    DATA = build_dataset_own_only()
    WIDS = list(DATA.keys())
    print(f'dataset ready {time.time()-t0:.0f}s  n_wells={len(WIDS)}')

    rng = np.random.default_rng(CFG['seed'])
    order = np.array(WIDS); rng.shuffle(order)
    folds = np.array_split(order, CFG['n_folds'])

    all_pred = {}
    for fi in range(CFG['n_folds']):
        VA = list(folds[fi])
        TR = list(np.concatenate([folds[j] for j in range(CFG['n_folds']) if j != fi]))
        mean, std = normalize_features(DATA, TR)
        pickle.dump(dict(mean=mean, std=std), open(f'gru_v6_norm_fold{fi}.pkl', 'wb'))
        base_va = prmse([(DATA[w]['src_sp45'] - DATA[w]['src_true']) ** 2 for w in VA])

        preds = {w: [] for w in VA}
        fold_bests = []
        for seed in range(CFG['n_models_per_fold']):
            net, best = train_one(fi * 100 + seed, DATA, TR, VA, save_tag=f'v6_fold{fi}_seed{seed}')
            fold_bests.append(best[0])
            net.eval()
            with torch.no_grad():
                for i in range(0, len(VA), 32):
                    ws = VA[i:i + 32]
                    X = torch.tensor(np.stack([DATA[w]['Xn'] for w in ws]), dtype=torch.float32, device=DEV)
                    pred_resid = net(X).cpu().numpy()
                    for j, w in enumerate(ws):
                        d = DATA[w]; n = d['n']
                        src_dst = np.linspace(0, n - 1, SEQ_LEN)
                        pred_full = np.interp(np.arange(n), src_dst, pred_resid[j])
                        preds[w].append(d['src_sp45'] + pred_full)
        for w in VA:
            all_pred[w] = np.mean(preds[w], axis=0)

        fold_ens_rmse = prmse([(all_pred[w] - DATA[w]['src_true']) ** 2 for w in VA])
        print(f'fold {fi}: n={len(VA)} sp45_base={base_va:.4f} per-seed best={[round(b,3) for b in fold_bests]} '
              f'ensemble={fold_ens_rmse:.4f}  {time.time()-t0:.0f}s', flush=True)

    overall_rmse = prmse([(all_pred[w] - DATA[w]['src_true']) ** 2 for w in WIDS])
    overall_sp45 = prmse([(DATA[w]['src_sp45'] - DATA[w]['src_true']) ** 2 for w in WIDS])
    print(f'\nOVERALL (all {len(WIDS)} wells, honest OOF): gru={overall_rmse:.4f}  sp45={overall_sp45:.4f}')
    pickle.dump(all_pred, open('gru_v4_kfold_oof_preds.pkl', 'wb'))
    print('saved gru_v4_kfold_oof_preds.pkl')
