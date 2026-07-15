"""v5 PRODUCTION: the "own-only" (16 features, no WARP-dependency) GRU-refiner, which the k-fold test
showed gives a statistically significant bootstrap gain (w=0.05: 95%CI=[+0.003,+0.079], excludes zero)
on the true 160-well holdout -- cleaner and better validated than the earlier WARP-dependent v3. Trained
as a 6-seed ensemble on the STANDARD n_val=160,seed=42 split (matching WARP/H4, for consistency with all
of today's reporting), ready for deployment: no WARP recomputation needed inside the submission cell at
all, simpler and more robust integration.
"""
import numpy as np, pickle, torch, torch.nn as nn, torch.nn.functional as F, time
from gru_refiner import DEV, SEQ_LEN, GRURefiner
from gru_refiner_v4_kfold import build_dataset_own_only

CFG = dict(d=64, layers=2, drop=0.40, wd=1e-3, lr=1.2e-3, epochs=40, bs=16, n_val=160, seed=42,
           noise_std=0.15, n_models=6)

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
    net = GRURefiner(16, CFG['d'], CFG['layers'], CFG['drop']).to(DEV)
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
    DATA = build_dataset_own_only()
    WIDS = list(DATA.keys())
    rng = np.random.default_rng(CFG['seed']); idx = np.arange(len(WIDS)); rng.shuffle(idx)
    VA = [WIDS[i] for i in idx[:CFG['n_val']]]; TR = [WIDS[i] for i in idx[CFG['n_val']:]]
    mean, std = normalize_features(DATA, TR)
    print(f'dataset ready {time.time()-t0:.0f}s  train={len(TR)} val={len(VA)}')
    pickle.dump(dict(mean=mean, std=std), open('gru_v5_norm_stats.pkl', 'wb'))

    base_va = prmse([(DATA[w]['src_sp45'] - DATA[w]['src_true']) ** 2 for w in VA])
    print(f'sp45 baseline on VA: {base_va:.4f}')

    results = []
    for seed in range(CFG['n_models']):
        r = train_one(seed, DATA, TR, VA, tag='v5')
        results.append(r)
    print('\nper-seed best:', results)

    preds = {w: [] for w in VA}
    for seed in range(CFG['n_models']):
        net = GRURefiner(16, CFG['d'], CFG['layers'], CFG['drop']).to(DEV)
        net.load_state_dict(torch.load(f'gru_refiner_v5_seed{seed}.pt', map_location=DEV))
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
    print(f'\nENSEMBLE ({CFG["n_models"]} seeds, own-only 16-feature) holdout RMSE: {ens_rmse:.4f}  vs sp45={base_va:.4f}')
    pickle.dump({w: np.mean(preds[w], axis=0) for w in VA}, open('gru_refiner_v5_ensemble_preds.pkl', 'wb'))
