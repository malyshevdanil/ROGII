"""Cross-fit validation of the beam-decoder blend with sp45: fit the best blend weight on one half
of wells, test on the untouched other half, both directions. This is the project's standing rule
(every isolated-PF-correction idea gets this treatment) -- 5 previous ideas (ir-anchor, gs-widening,
wall-hedge, pf_z, Cauchy) failed it once blended into the full combo. This is the first one to show
strong decorrelation (corr=0.088) with a real full-sample blend gain (7.92->6.86 sp45-only); this
script checks whether that gain is real (survives cross-fit) or a mirage on w.
"""
import numpy as np, pickle, time
from beam_decoder_v1 import run_one as beam_run_one, load_well as beam_load_well

PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'
proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
WIDS = list(proxy.keys())
rng = np.random.default_rng(11)
sample = list(rng.choice(WIDS, size=50, replace=False))

per_well = {}
t0 = time.time()
for i, wid in enumerate(sample):
    px = proxy[wid]
    r = beam_run_one(wid, seed=1, K=800, G=9, weighted=False)
    if r is None: continue
    pts, true_tvt = r
    r2 = beam_load_well(wid)
    if r2 is None: continue
    hw, tw_tvt, tw_gr, kn, ev = r2
    md_v = ev.MD.values.astype(np.float64)
    md_proxy = px['md']
    beam_on_proxy = np.interp(md_proxy, md_v, pts)
    per_well[wid] = dict(beam=beam_on_proxy, sp45=px['sp45'], true=px['true'])
    if (i+1) % 10 == 0:
        print(f'  {i+1}/{len(sample)}  {time.time()-t0:.0f}s')

print(f'\nn wells ok: {len(per_well)}/{len(sample)}  total time: {time.time()-t0:.0f}s')
OK_WIDS = list(per_well.keys())

def pooled_blend(w, wl):
    sq = []
    for wid in wl:
        d = per_well[wid]
        pred = (1-w)*d['sp45'] + w*d['beam']
        sq.append((pred-d['true'])**2)
    return float(np.sqrt(np.mean(np.concatenate(sq))))

W_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

def best_w_on(wl):
    best = (1e9, None)
    for w in W_GRID:
        v = pooled_blend(w, wl)
        if v < best[0]: best = (v, w)
    return best

print('\n=== full-sample sweep ===')
for w in W_GRID:
    print(f'  w={w:.1f}  pooled={pooled_blend(w, OK_WIDS):.4f}')

print('\n=== 2-fold cross-fit ===')
rng2 = np.random.default_rng(13)
order = np.array(OK_WIDS); rng2.shuffle(order)
H1, H2 = list(order[:len(order)//2]), list(order[len(order)//2:])

b1 = best_w_on(H1)
base_h2 = pooled_blend(0.0, H2)
v_h2 = pooled_blend(b1[1], H2)
print(f'fit on H1: best w={b1[1]} (train={b1[0]:.4f}) -> H2 test={v_h2:.4f} vs H2 sp45-only baseline={base_h2:.4f}')

b2 = best_w_on(H2)
base_h1 = pooled_blend(0.0, H1)
v_h1 = pooled_blend(b2[1], H1)
print(f'fit on H2: best w={b2[1]} (train={b2[0]:.4f}) -> H1 test={v_h1:.4f} vs H1 sp45-only baseline={base_h1:.4f}')
