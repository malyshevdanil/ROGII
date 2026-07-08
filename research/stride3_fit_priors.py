"""New architecture, step 1: fit segment-length and persistence priors from OUR OWN known-zone data
(legal -- known zone has ground truth on every training well). This is the joint (h_len, delta-TVT)
lattice-decode idea (inspired by the category of STRIDE's approach, not copied numbers), grounded in our
own already-established human-markup findings (S8.7: piecewise-linear, quantized dips, structural
breakpoints).

Algorithm: detect breakpoints in each well's KNOWN-zone TVT-vs-MD curve via a simple recursive
change-point split (greedy: keep splitting the biggest-error segment until residual is small), then
extract (segment_length_ft, dip_rate=delta_tvt/segment_length) pairs. Fit:
  - log-normal shape to segment lengths
  - AR(1)-style persistence: dip_rate[i] ~ rho * dip_rate[i-1] + noise, fit rho and noise std
"""
import numpy as np, pandas as pd, glob, os

TRAIN_DIR = 'd:/ROGII/data/train'

def breakpoints_piecewise(md, tvt, max_resid=1.5, min_seg=20.0):
    """Greedy recursive split: split the segment with largest max-residual from its own line fit,
    until every segment's residual is below max_resid or segment is too short to split further."""
    segments = [(0, len(md))]
    def seg_resid(a, b):
        if b - a < 3: return 0.0, None
        x = md[a:b]; y = tvt[a:b]
        c = np.polyfit(x, y, 1)
        r = y - np.polyval(c, x)
        return np.max(np.abs(r)), c
    changed = True
    while changed:
        changed = False
        new_segments = []
        for (a, b) in segments:
            if md[b-1] - md[a] < min_seg * 2:
                new_segments.append((a, b)); continue
            resid, _ = seg_resid(a, b)
            if resid <= max_resid:
                new_segments.append((a, b)); continue
            # find split point minimizing total residual
            best = None
            for k in range(a + 5, b - 5):
                if md[k] - md[a] < min_seg or md[b-1] - md[k] < min_seg: continue
                r1, _ = seg_resid(a, k); r2, _ = seg_resid(k, b)
                score = max(r1, r2)
                if best is None or score < best[0]:
                    best = (score, k)
            if best is None or best[0] >= resid:
                new_segments.append((a, b))
            else:
                new_segments.append((a, best[1])); new_segments.append((best[1], b))
                changed = True
        segments = new_segments
        if len(segments) > 60: break  # safety cap
    return sorted(segments)

def find_landing_index(md, z, win=40, thresh=0.15):
    """Detect where the well flattens out of the build/landing curve into the lateral.
    Inclination proxy = |dZ/dMD|; near 1 while still curving down into the target,
    near 0 once lateral. Return the first index after which the rolling-median
    inclination proxy stays below `thresh` for good, so we can drop the steep
    landing segment and only fit priors on the lateral (which is what the eval
    region actually continues)."""
    if len(md) < win * 2:
        return 0
    dz = np.diff(z) / np.maximum(np.diff(md), 1e-6)
    incl = np.abs(dz)
    roll = pd.Series(incl).rolling(win, min_periods=win).median()
    below = roll < thresh
    # last index where it's still above thresh (i.e. still in the build curve)
    above_idx = np.where(~below.values & ~np.isnan(roll.values))[0]
    if len(above_idx) == 0:
        return 0
    return int(above_idx[-1]) + 1

def process_well(wid):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    kn = hw[hw['TVT_input'].notna()]
    if len(kn) < 100: return None
    md_full = kn['MD'].values.astype(float)
    z_full = kn['Z'].values.astype(float)
    # track pos = TVT+Z, matching the production PF's state convention (pos evolves smoothly;
    # TVT = pos - Z at read-out time), NOT raw TVT_input directly.
    pos_full = kn['TVT_input'].values.astype(float) + z_full
    land_idx = find_landing_index(md_full, z_full)
    md = md_full[land_idx:]; tvt = pos_full[land_idx:]
    if len(md) < 50: return None
    segs = breakpoints_piecewise(md, tvt)
    out = []
    resids = []
    for (a, b) in segs:
        if b - a < 3: continue
        length = md[b-1] - md[a]
        if length < 5: continue
        dtvt = tvt[b-1] - tvt[a]
        out.append((length, dtvt / length))
        x = md[a:b]; y = tvt[a:b]
        c = np.polyfit(x, y, 1)
        resids.extend((y - np.polyval(c, x)).tolist())
    return out, resids

if __name__ == '__main__':
    wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
    rng = np.random.default_rng(0)
    sample = list(rng.choice(wids, size=400, replace=False))

    all_lengths = []
    all_dips = []
    all_resids = []
    persistence_pairs = []  # (dip[i-1], dip[i])
    n_ok = 0
    for wid in sample:
        pr = process_well(wid)
        if pr is None: continue
        r, resids = pr
        if len(r) < 1: continue
        n_ok += 1
        lengths = [x[0] for x in r]; dips = [x[1] for x in r]
        all_lengths.extend(lengths); all_dips.extend(dips); all_resids.extend(resids)
        for i in range(1, len(dips)):
            persistence_pairs.append((dips[i-1], dips[i]))

    all_lengths = np.array(all_lengths); all_dips = np.array(all_dips)
    persistence_pairs = np.array(persistence_pairs)
    print('wells processed:', n_ok, '/', len(sample))
    print('total segments:', len(all_lengths))
    print('\n--- segment length distribution (ft) ---')
    print('median:', np.median(all_lengths), 'mean:', all_lengths.mean(), 'std:', all_lengths.std())
    log_len = np.log(all_lengths)
    print('log-normal fit: mu=%.3f sigma=%.3f (implies median=%.1f ft)' % (log_len.mean(), log_len.std(), np.exp(log_len.mean())))
    for p in [10, 25, 50, 75, 90]:
        print(f'  p{p}: {np.percentile(all_lengths, p):.1f} ft')

    print('\n--- dip rate distribution (ft/ft) ---')
    print('median:', np.median(all_dips), 'mean:', all_dips.mean(), 'std:', all_dips.std())

    print('\n--- persistence: dip[i] ~ rho * dip[i-1] + noise ---')
    x = persistence_pairs[:,0]; y = persistence_pairs[:,1]
    rho = np.sum(x*y) / np.sum(x*x)
    resid = y - rho*x
    print('n pairs:', len(x), 'rho (no intercept):', rho, 'resid std:', resid.std())
    corr = np.corrcoef(x, y)[0,1]
    print('correlation(dip[i-1], dip[i]):', corr)

    all_resids = np.array(all_resids)
    print('\n--- within-segment line-fit residual (local position wiggle) ---')
    print('n points:', len(all_resids), 'std:', all_resids.std(), 'p90 abs:', np.percentile(np.abs(all_resids), 90))

    import pickle
    pickle.dump(dict(log_len_mu=log_len.mean(), log_len_sigma=log_len.std(),
                      dip_std=all_dips.std(), persistence_rho=rho, persistence_resid_std=resid.std(),
                      dip_mean=all_dips.mean(), local_resid_std=float(all_resids.std())),
                open('stride3_priors.pkl', 'wb'))
    print('\nsaved priors to stride3_priors.pkl')
