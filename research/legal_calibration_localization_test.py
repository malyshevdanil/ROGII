"""Tests a specific claim from a new competitor writeup ('Fork the ruler, not the model'):
that a LEGAL heel-calibration (affine gain/offset fit on the known zone only, no eval-zone ground truth)
recovers ~80% localization within ~2 ft, "essentially" matching an oracle's 82% -- i.e. that the datum/
offset IS legally recoverable via GR-calibrated shift-scan matching, contradicting the "unobservable
datum" consensus (ours and 4 other independent writeups: James-Stein a*~0, r=0.03 known->eval, r=-0.08
neighbour corr, etc).

We build our own independent window shift-scan localization test:
 - naive: raw (uncalibrated) eval-zone GR window
 - legal: window calibrated by an affine fit using ONLY the known/heel zone (no eval ground truth)
 - oracle: window calibrated by an affine fit using the true eval-zone GR-vs-typewell-at-true-TVT (cheating,
   upper bound reference only)
For each, shift-scan the window against the typewell TVT-GR curve, take the best-matching (argmin misfit)
center position, and compare to the TRUE center TVT. Report the fraction of wells localized within 2 ft,
across a few window lengths, to see whether we replicate anything like the claimed ~80%/~8% split.
"""
import numpy as np, pandas as pd, glob, os

TRAIN_DIR = 'd:/ROGII/data/train'

def inn(a):
    a = a.copy(); n = len(a); idx = np.arange(n); m = np.isnan(a)
    if m.all(): return np.zeros(n)
    a[m] = np.interp(idx[m], idx[~m], a[~m]); return a

def load_well(wid):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tt = tw['TVT'].values.astype(float); tg = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    if len(tt) < 20: return None
    kn = hw[hw['TVT_input'].notna()]
    ev = hw['TVT_input'].isna().values
    if len(kn) < 50 or ev.sum() < 50: return None
    gr = inn(hw['GR'].values.astype(float))
    true_tvt = hw['TVT'].values.astype(float)
    md = hw['MD'].values.astype(float)
    return dict(wid=wid, tt=tt, tg=tg, kn_gr=kn['GR'].interpolate().bfill().ffill().values.astype(float),
                kn_tvt=kn['TVT_input'].values.astype(float), gr=gr, true_tvt=true_tvt, md=md, ev=ev)

def fit_affine(x, y):
    v = np.isfinite(x) & np.isfinite(y)
    if v.sum() < 20: return 1.0, 0.0
    a, b = np.polyfit(x[v], y[v], 1)
    return a, b

def shift_scan_localize(window_gr, tw_tvt, tw_gr, search_lo, search_hi, n_candidates=400):
    """slide the (already-calibrated) window against the typewell curve; return best-match center TVT."""
    wlen_tvt = None  # window expressed directly in GR-index space; we resample typewell to same length grid
    candidates = np.linspace(search_lo, search_hi, n_candidates)
    best_c, best_err = None, np.inf
    n = len(window_gr)
    half_span = None
    for c in candidates:
        # define a TVT span around candidate center with the SAME number of samples as window,
        # spaced at the typewell's native resolution scale (use a fixed span proportional to window length)
        span = SPAN_FT
        grid = np.linspace(c - span/2, c + span/2, n)
        if grid[0] < tw_tvt.min() or grid[-1] > tw_tvt.max():
            continue
        cand_gr = np.interp(grid, tw_tvt, tw_gr)
        # NOTE: no per-candidate z-normalization here -- that would erase any affine
        # calibration difference between naive/legal/oracle before it can matter.
        # window_gr arrives ALREADY calibrated (or raw, for 'naive'); match in those units directly.
        err = np.mean((cand_gr - window_gr) ** 2)
        if err < best_err:
            best_err = err; best_c = c
    return best_c

WINDOW_LENS = [100, 300, 600]
N_WELLS = 150

wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
rng = np.random.default_rng(0)
sample = rng.choice(wids, size=min(N_WELLS, len(wids)), replace=False)

for WLEN in WINDOW_LENS:
    SPAN_FT = WLEN
    results = {'naive': [], 'legal': [], 'oracle': []}
    used = 0
    for wid in sample:
        w = load_well(wid)
        if w is None: continue
        tt, tg = w['tt'], w['tg']
        ev_idx = np.where(w['ev'])[0]
        if len(ev_idx) < 20: continue
        e0 = ev_idx[0]
        # pick a window starting at the anchor, WLEN ft of MD (approx by sample count via MD spacing)
        md = w['md']
        seg_end = e0
        while seg_end < len(md) - 1 and (md[seg_end] - md[e0]) < WLEN:
            seg_end += 1
        if seg_end - e0 < 10 or seg_end >= len(md): continue
        idx = np.arange(e0, seg_end)
        true_center_tvt = float(np.mean(w['true_tvt'][idx]))
        raw_win = w['gr'][idx]

        # naive: raw GR window
        # legal: affine-calibrate using KNOWN zone only
        kn_twg = np.interp(w['kn_tvt'], tt, tg)
        a_legal, b_legal = fit_affine(w['kn_gr'], kn_twg)
        legal_win = raw_win * a_legal + b_legal

        # oracle: affine-calibrate using the TRUE eval window itself vs typewell-at-true-TVT (cheating)
        true_twg_here = np.interp(w['true_tvt'][idx], tt, tg)
        a_or, b_or = fit_affine(raw_win, true_twg_here)
        oracle_win = raw_win * a_or + b_or

        lo, hi = tt.min() + SPAN_FT/2 + 1, tt.max() - SPAN_FT/2 - 1
        if hi <= lo: continue

        for name, win in [('naive', raw_win), ('legal', legal_win), ('oracle', oracle_win)]:
            est = shift_scan_localize(win, tt, tg, lo, hi)
            if est is None: continue
            results[name].append(abs(est - true_center_tvt))
        used += 1

    print(f'\n=== window length {WLEN} ft, n_wells={used} ===')
    for name in ['naive', 'legal', 'oracle']:
        errs = np.array(results[name])
        if len(errs) == 0:
            print(f'  {name}: no data'); continue
        within2 = float(np.mean(errs <= 2.0)) * 100
        within5 = float(np.mean(errs <= 5.0)) * 100
        print(f'  {name:8s}: n={len(errs):3d}  median_err={np.median(errs):7.2f} ft  '
              f'mean_err={errs.mean():7.2f} ft  within_2ft={within2:5.1f}%  within_5ft={within5:5.1f}%')
