"""Run the (bugfixed, w=0.5) beam decoder on the full true 160-well holdout, save per-well predictions
for correlation/blend-value analysis against the current best combo (v7blend)."""
import numpy as np, pickle, time
import beam2_decoder as b2

b2.EMISSION_WEIGHT = 0.5

if __name__ == '__main__':
    t0 = time.time()
    VA_WIDS = pickle.load(open('warp_true_holdout_160.pkl', 'rb'))
    per_well = {}
    n_ok = 0
    for i, wid in enumerate(VA_WIDS):
        r = b2.decode_well(wid)
        if r is None: continue
        pred, true, md = r
        per_well[wid] = dict(pred=pred, true=true, md=md)
        n_ok += 1
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(VA_WIDS)}  {time.time()-t0:.0f}s', flush=True)
    print(f'done {n_ok}/{len(VA_WIDS)} wells  {time.time()-t0:.0f}s')

    pooled = np.sqrt(np.mean(np.concatenate([(d['pred'] - d['true'])**2 for d in per_well.values()])))
    print('pooled RMSE (beam decoder w=0.5, true holdout):', pooled)
    pickle.dump(per_well, open('beam2_true_holdout_w05.pkl', 'wb'))
    print('saved beam2_true_holdout_w05.pkl')
