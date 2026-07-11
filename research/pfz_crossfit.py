"""Honest cross-fit for the pf_z blend weight against the REAL sp45 proxy (not just pf_ancc alone).
Uses the verbatim numba kernels from real_pfz_test.py, seed-averaged (8 seeds, matching production
intent) on as many proxy wells as time allows, split 50/50 for cross-fit."""
import numpy as np, pandas as pd, glob, os, time, pickle
import real_pfz_test as rt

proxy = pickle.load(open('C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl', 'rb'))
D = proxy['DATA']
wids = list(D.keys())
TRAIN_DIR = 'd:/ROGII/data/train'
N_SEEDS = 8
N_WELLS = 300

rng = np.random.default_rng(11); order = np.array(wids); rng.shuffle(order)
use_wids = order[:N_WELLS]

def run_pfz_avg(hw, tw_tvt, tw_gr, n_seeds):
    accum = None
    for s in range(n_seeds):
        pz, _ = rt.run_pf_z(hw, tw_tvt, tw_gr, N=rt.PF_N)
        if accum is None: accum = np.zeros_like(pz, dtype=np.float64)
        accum += pz
    return accum / n_seeds

print('computing pf_z (seed-avg=%d) on %d wells...' % (N_SEEDS, N_WELLS))
t0 = time.time()
records = []
for j, wid in enumerate(use_wids):
    if wid not in D: continue
    try:
        hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
        tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    except Exception:
        continue
    if 'TVT' not in hw.columns: continue
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0: continue
    tw_tvt = tw['TVT'].values.astype(np.float32); tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(np.float32)
    pz_avg = run_pfz_avg(hw, tw_tvt, tw_gr, N_SEEDS)
    d = D[wid]
    if len(pz_avg) != len(d['sp45']):
        continue  # alignment mismatch (proxy arrays may already be eval-only in raw row order; should match)
    records.append(dict(wid=wid, pfz=pz_avg, sp45=d['sp45'], true=d['true']))
    if (j + 1) % 25 == 0: print('  ', j + 1, '/', len(use_wids), '%.0fs' % (time.time() - t0), flush=True)

print('\nusable wells:', len(records))

def pooled(a_weight, wl):
    s = []
    for r in records:
        if r['wid'] not in wl: continue
        pred = (1 - a_weight) * r['sp45'] + a_weight * r['pfz']
        s.append((pred - r['true']) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(s))))

all_wids = [r['wid'] for r in records]
rng2 = np.random.default_rng(3); ord2 = np.array(all_wids); rng2.shuffle(ord2)
H1 = set(ord2[:len(ord2)//2]); H2 = set(ord2[len(ord2)//2:])

print('\nbaseline sp45 (a=0):  H1=%.4f H2=%.4f ALL=%.4f' % (
    pooled(0.0, H1), pooled(0.0, H2), pooled(0.0, set(all_wids))))

grid = np.arange(0.0, 0.61, 0.05)
def best_on(wl):
    best = (99, 0)
    for a in grid:
        v = pooled(a, wl)
        if v < best[0]: best = (v, a)
    return best

b1 = best_on(H1); print('fit on H1: best a=%.2f train=%.4f -> H2=%.4f (H2 baseline %.4f)' % (
    b1[1], b1[0], pooled(b1[1], H2), pooled(0.0, H2)))
b2 = best_on(H2); print('fit on H2: best a=%.2f train=%.4f -> H1=%.4f (H1 baseline %.4f)' % (
    b2[1], b2[0], pooled(b2[1], H1), pooled(0.0, H1)))

print('\nfull sweep (ALL wells):')
for a in grid:
    print('  a=%.2f  pooled=%.4f' % (a, pooled(a, set(all_wids))))
