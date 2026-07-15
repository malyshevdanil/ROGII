"""Bidirectional-GRU REFINEMENT model, directly informed by today's top-5 competitor writeup (Tucker
Arrants): his winning architecture is NOT a from-scratch GR->TVT regressor (which is exactly what our
WARP/H4/MDN/etc. all were, and all capped ~11.3 -- see the NN research history in memory) -- it is a
bidirectional GRU that REFINES an existing PF tracker's output, taking the tracker's own position AND
its disagreement with other tracker variants as input features, not raw GR alone. We have never tried
this framing: every one of our from-scratch NN attempts tried to solve localization directly from GR.

Architecture: input = [16 raw GR/trajectory features ('own', already cached) + 4 disagreement channels
(warp-sp45, h4ema-sp45, beam-sp45, pf8-sp45) + 1 ensemble-std channel] per eval row, resampled to a fixed
length. Bidirectional GRU (2-layer) -> linear head -> predicts a RESIDUAL correction added to sp45 (not
predicting TVT from scratch), so the model only has to learn WHEN to trust/distrust the existing trackers,
not solve GR-matching itself. Trained with the SAME n_val=160,seed=42 holdout as WARP/H4 for consistency
and honest reuse of warp_true_holdout_160.pkl.
"""
import numpy as np, pickle, torch, torch.nn as nn, torch.nn.functional as F, time

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 500
CFG = dict(d=96, layers=2, drop=0.25, wd=3e-4, lr=1.5e-3, epochs=80, bs=16, n_val=160, seed=42)

PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'
WARP_CACHE = 'warp_on_proxy_cache.pkl'
H4_CACHE_ALL = 'h4ema_on_proxy_cache_ALL773.pkl'

def build_dataset():
    proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
    warp_on_proxy = pickle.load(open(WARP_CACHE, 'rb'))
    h4_on_proxy = pickle.load(open(H4_CACHE_ALL, 'rb'))
    WIDS = [w for w in proxy if w in warp_on_proxy and w in h4_on_proxy]
    print('usable wells:', len(WIDS))

    DATA = {}
    for wid in WIDS:
        d = proxy[wid]
        n = len(d['true'])
        if n < 50: continue
        sp45 = d['sp45'].astype(np.float64)
        warp = warp_on_proxy[wid].astype(np.float64)
        h4 = h4_on_proxy[wid].astype(np.float64)
        beam = d['beam'].astype(np.float64)
        pf8 = d['pf8'].astype(np.float64)
        own = d['own'].astype(np.float64)  # (n, 16)
        true = d['true'].astype(np.float64)

        disagree = np.stack([warp - sp45, h4 - sp45, beam - sp45, pf8 - sp45], axis=1)  # (n,4)
        stack_all = np.stack([sp45, warp, h4, beam, pf8], axis=1)
        ens_std = stack_all.std(axis=1, keepdims=True)  # (n,1)

        X = np.concatenate([own, disagree, ens_std], axis=1)  # (n, 21)
        y = (true - sp45)  # residual target

        # resample to fixed length
        src = np.arange(n)
        dst = np.linspace(0, n - 1, SEQ_LEN)
        Xr = np.stack([np.interp(dst, src, X[:, c]) for c in range(X.shape[1])], axis=1).astype(np.float32)
        yr = np.interp(dst, src, y).astype(np.float32)
        DATA[wid] = dict(X=Xr, y=yr, n=n, src_md=d['md'], src_true=true, src_sp45=sp45)
    return DATA

class GRURefiner(nn.Module):
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

if __name__ == '__main__':
    torch.manual_seed(0); np.random.seed(0)
    print('DEV:', DEV)
    t0 = time.time()
    DATA = build_dataset()
    print(f'dataset built {time.time()-t0:.0f}s')

    WIDS = list(DATA.keys())
    rng = np.random.default_rng(CFG['seed']); idx = np.arange(len(WIDS)); rng.shuffle(idx)
    VA = [WIDS[i] for i in idx[:CFG['n_val']]]; TR = [WIDS[i] for i in idx[CFG['n_val']:]]
    print(f'train={len(TR)} val={len(VA)}')

    mean, std = normalize_features(DATA, TR)

    net = GRURefiner(21, CFG['d'], CFG['layers'], CFG['drop']).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    n_batches = (len(TR) + CFG['bs'] - 1) // CFG['bs']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG['lr'], total_steps=CFG['epochs'] * n_batches)

    def batch_tensors(ws):
        X = torch.tensor(np.stack([DATA[w]['Xn'] for w in ws]), dtype=torch.float32, device=DEV)
        y = torch.tensor(np.stack([DATA[w]['y'] for w in ws]), dtype=torch.float32, device=DEV)
        return X, y

    def evaluate(wids):
        net.eval()
        sqs = []
        with torch.no_grad():
            for i in range(0, len(wids), 32):
                ws = wids[i:i + 32]
                X, y = batch_tensors(ws)
                pred_resid = net(X)
                for j, w in enumerate(ws):
                    d = DATA[w]
                    n = d['n']
                    src_dst = np.linspace(0, n - 1, SEQ_LEN)
                    pred_full = np.interp(np.arange(n), src_dst, pred_resid[j].cpu().numpy())
                    final_pred = d['src_sp45'] + pred_full
                    sqs.append((final_pred - d['src_true']) ** 2)
        net.train()
        return prmse(sqs)

    base_va = prmse([( DATA[w]['src_sp45'] - DATA[w]['src_true']) ** 2 for w in VA])
    print(f'sp45 baseline on VA: {base_va:.4f}')

    best = (99.0, 0)
    order = np.arange(len(TR))
    for ep in range(CFG['epochs']):
        np.random.shuffle(order)
        tot = 0.0; nb = 0
        for i in range(0, len(TR), CFG['bs']):
            ws = [TR[k] for k in order[i:i + CFG['bs']]]
            if not ws: continue
            X, y = batch_tensors(ws)
            pred = net(X)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
        r = evaluate(VA)
        if r < best[0]:
            best = (r, ep + 1)
            torch.save(net.state_dict(), 'gru_refiner_best.pt')
        print(f'ep{ep+1:3d} loss{tot/nb:.4f} | holdout RMSE {r:.4f} (best {best[0]:.4f}@{best[1]}) {time.time()-t0:.0f}s', flush=True)

    print(f'\nDONE. sp45 baseline={base_va:.4f}  GRU-refiner best={best[0]:.4f}@ep{best[1]}')
