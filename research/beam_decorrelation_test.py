"""Does the new beam-search decoder's error pattern decorrelate from sp45 (the base track used
throughout the production combo)? Even at rough quality-parity with pf_ancc (see beam_decoder_v1.py
results: 14.47 vs pf_ancc's 14.27, same 40-well/4-seed protocol), a genuinely different error
STRUCTURE could still add value in a blend -- that's the real question, not raw solo RMSE.
"""
import numpy as np, pickle, time
from beam_decoder_v1 import run_one as beam_run_one, load_well as beam_load_well

PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'
proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
WIDS = list(proxy.keys())
rng = np.random.default_rng(11)
sample = list(rng.choice(WIDS, size=25, replace=False))

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
    if (i+1) % 5 == 0:
        print(f'  {i+1}/{len(sample)}  {time.time()-t0:.0f}s')

print(f'\nn wells ok: {len(per_well)}/{len(sample)}  total time: {time.time()-t0:.0f}s')

beam_err_all = np.concatenate([d['beam']-d['true'] for d in per_well.values()])
sp45_err_all = np.concatenate([d['sp45']-d['true'] for d in per_well.values()])
print('beam RMSE (on proxy grid):', np.sqrt(np.mean(beam_err_all**2)))
print('sp45 RMSE (on proxy grid):', np.sqrt(np.mean(sp45_err_all**2)))
corr = np.corrcoef(beam_err_all, sp45_err_all)[0, 1]
print('correlation(beam_err, sp45_err):', corr)

def pooled_blend(w):
    sq = []
    for d in per_well.values():
        pred = (1-w)*d['sp45'] + w*d['beam']
        sq.append((pred-d['true'])**2)
    return float(np.sqrt(np.mean(np.concatenate(sq))))

print('\n=== blend sweep: (1-w)*sp45 + w*beam ===')
for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    print(f'  w={w:.1f}  pooled={pooled_blend(w):.4f}')
