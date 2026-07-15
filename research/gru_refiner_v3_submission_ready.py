"""v3: reduced feature set for SAFE submission integration -- drop beam/pf8 (only available via a
research snapshot, not reliably re-derivable live in the Kaggle notebook without duplicating expensive
PF work) and h4ema (would need a second Kaggle-dataset checkpoint upload). Keep only: 16 'own' GR/
trajectory features + (warp - sp45) disagreement + std(sp45, warp) -- both trivially available at the
point in the real pipeline where the WARP-blend cell already runs (it computes warp predictions in
memory). This is a deliberate fidelity-for-reliability tradeoff: re-validate that the smaller feature
set still shows the same paired-bootstrap signal before building the real submission cell.
"""
import numpy as np, pickle, torch, torch.nn as nn, torch.nn.functional as F, time
from gru_refiner import DEV, SEQ_LEN, GRURefiner

CFG = dict(d=64, layers=2, drop=0.40, wd=1e-3, lr=1.2e-3, epochs=40, bs=16, n_val=160, seed=42,
           noise_std=0.15, n_models=6)

PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'
WARP_CACHE = 'warp_on_proxy_cache.pkl'

def build_dataset_v3():
    proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
    warp_on_proxy = pickle.load(open(WARP_CACHE, 'rb'))
    WIDS = [w for w in proxy if w in warp_on_proxy]
    print('usable wells:', len(WIDS))
    DATA = {}
    for wid in WIDS:
        d = proxy[wid]
        n = len(d['true'])
        if n < 50: continue
        sp45 = d['sp45'].astype(np.float64)
        warp = warp_on_proxy[wid].astype(np.float64)
        own = d['own'].astype(np.float64)
        true = d['true'].astype(np.float64)

        disagree = (warp - sp45)[:, None]
        ens_std = np.stack([sp45, warp], axis=1).std(axis=1, keepdims=True)
        X = np.concatenate([own, disagree, ens_std], axis=1)  # (n, 18)
        y = true - sp45

        src = np.arange(n)
        dst = np.linspace(0, n - 1, SEQ_LEN)
        Xr = np.stack([np.interp(dst, src, X[:, c]) for c in range(X.shape[1])], axis=1).astype(np.float32)
        yr = np.interp(dst, src, y).astype(np.float32)
        DATA[wid] = dict(X=Xr, y=yr, n=n, src_md=d['md'], src_true=true, src_sp45=sp45)
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

def train_one(seed, DATA, TR, VA, tag):
    torch.manual_seed(seed); np.random.seed(seed)
    net = GRURefiner(18, CFG['d'], CFG['layers'], CFG['drop']).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    n_batches = (len(TR) + CFG['bs'] - 1) // CFG['bs']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG['lr'], total_steps=CFG['epochs'] * n_batches)
    best = (99.0, 0)
    order = np.arange(len(TR))
    t0 = time.time()
    for ep in range(CFG['epochs']):
        np.random.shuffle(order)
        tot = 0.0; nb = 0
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
            tot += loss.item(); nb += 1
        r = evaluate(net, DATA, VA)
        if r < best[0]:
            best = (r, ep + 1)
            torch.save(net.state_dict(), f'gru_refiner_{tag}_seed{seed}.pt')
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'  seed{seed} ep{ep+1:3d} loss{tot/nb:.4f} | holdout {r:.4f} (best {best[0]:.4f}@{best[1]}) {time.time()-t0:.0f}s', flush=True)
    print(f'seed{seed} DONE best={best[0]:.4f}@ep{best[1]}')
    return best

if __name__ == '__main__':
    t0 = time.time()
    DATA = build_dataset_v3()
    WIDS = list(DATA.keys())
    rng = np.random.default_rng(CFG['seed']); idx = np.arange(len(WIDS)); rng.shuffle(idx)
    VA = [WIDS[i] for i in idx[:CFG['n_val']]]; TR = [WIDS[i] for i in idx[CFG['n_val']:]]
    mean, std = normalize_features(DATA, TR)
    print(f'dataset ready {time.time()-t0:.0f}s  train={len(TR)} val={len(VA)}')
    print('feature mean/std (for export):')
    print('mean:', mean.tolist())
    print('std:', std.tolist())
    pickle.dump(dict(mean=mean, std=std), open('gru_v3_norm_stats.pkl', 'wb'))

    base_va = prmse([(DATA[w]['src_sp45'] - DATA[w]['src_true']) ** 2 for w in VA])
    print(f'sp45 baseline on VA: {base_va:.4f}')

    results = []
    for seed in range(CFG['n_models']):
        r = train_one(seed, DATA, TR, VA, tag='v3')
        results.append(r)
    print('\nper-seed best:', results)

    preds = {w: [] for w in VA}
    for seed in range(CFG['n_models']):
        net = GRURefiner(18, CFG['d'], CFG['layers'], CFG['drop']).to(DEV)
        net.load_state_dict(torch.load(f'gru_refiner_v3_seed{seed}.pt', map_location=DEV))
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
    sqs_ens = []
    for w in VA:
        ens_pred = np.mean(preds[w], axis=0)
        sqs_ens.append((ens_pred - DATA[w]['src_true']) ** 2)
    ens_rmse = prmse(sqs_ens)
    print(f'\nENSEMBLE ({CFG["n_models"]} seeds, 18-feature reduced) holdout RMSE: {ens_rmse:.4f}  vs sp45={base_va:.4f}')
    pickle.dump({w: np.mean(preds[w], axis=0) for w in VA}, open('gru_refiner_v3_ensemble_preds.pkl', 'wb'))
