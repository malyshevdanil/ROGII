"""v3: root cause found. Known-zone AVERAGE dip wildly overestimates the local eval-zone TVT span for a
given MD window (e.g. 0.30 ft/ft average dip x 400 ft MD = 120 ft projected span, when the TRUE local span
over that exact window was 1.93 ft -- a ~60x overestimate). That is itself a restatement of "the wall"
(§5): known-zone dip does not persist into the eval zone, so you cannot legally size a shift-scan window
from it either. To fairly isolate the CALIBRATION question (not the dip-projection question, which we
already know fails), size the candidate window using the TRUE span (oracle-informed) IDENTICALLY for all
three conditions -- naive/legal/oracle differ ONLY in GR amplitude calibration, not in window size.
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

def shift_scan_localize(window_gr, span, tw_tvt, tw_gr, search_lo, search_hi, n_candidates=600):
    candidates = np.linspace(search_lo, search_hi, n_candidates)
    best_c, best_err = None, np.inf
    n = len(window_gr)
    for c in candidates:
        grid = np.linspace(c - span/2, c + span/2, n)
        if grid[0] < tw_tvt.min() or grid[-1] > tw_tvt.max():
            continue
        cand_gr = np.interp(grid, tw_tvt, tw_gr)
        err = np.mean((cand_gr - window_gr) ** 2)
        if err < best_err:
            best_err = err; best_c = c
    return best_c

WINDOW_MD_LENS = [200, 400, 800]
N_WELLS = 150

wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
rng = np.random.default_rng(0)
sample = rng.choice(wids, size=min(N_WELLS, len(wids)), replace=False)

for WLEN in WINDOW_MD_LENS:
    results = {'naive': [], 'legal': [], 'oracle': []}
    used, skipped_tiny_span = 0, 0
    for wid in sample:
        w = load_well(wid)
        if w is None: continue
        tt, tg = w['tt'], w['tg']
        ev_idx = np.where(w['ev'])[0]
        if len(ev_idx) < 20: continue
        e0 = ev_idx[0]
        md = w['md']
        seg_end = e0
        while seg_end < len(md) - 1 and (md[seg_end] - md[e0]) < WLEN:
            seg_end += 1
        if seg_end - e0 < 10 or seg_end >= len(md): continue
        idx = np.arange(e0, seg_end)

        true_span = float(w['true_tvt'][idx].max() - w['true_tvt'][idx].min())
        span_est = max(true_span, 1.0)  # oracle-informed span, held IDENTICAL across all 3 conditions
        if true_span < 0.5:
            skipped_tiny_span += 1  # degenerate: well is essentially flat here, skip (no info either way)
            continue

        true_center_tvt = float(np.mean(w['true_tvt'][idx]))
        raw_win = w['gr'][idx]

        kn_twg = np.interp(w['kn_tvt'], tt, tg)
        a_legal, b_legal = fit_affine(w['kn_gr'], kn_twg)
        legal_win = raw_win * a_legal + b_legal

        true_twg_here = np.interp(w['true_tvt'][idx], tt, tg)
        a_or, b_or = fit_affine(raw_win, true_twg_here)
        oracle_win = raw_win * a_or + b_or

        lo, hi = tt.min() + span_est/2 + 0.5, tt.max() - span_est/2 - 0.5
        if hi <= lo: continue

        for name, win in [('naive', raw_win), ('legal', legal_win), ('oracle', oracle_win)]:
            est = shift_scan_localize(win, span_est, tt, tg, lo, hi)
            if est is None: continue
            results[name].append(abs(est - true_center_tvt))
        used += 1

    print(f'\n=== window MD length {WLEN} ft (span = TRUE oracle-informed span, same for all 3 conditions) ===')
    print(f'    n_wells used={used}, skipped (near-zero true span here)={skipped_tiny_span}')
    for name in ['naive', 'legal', 'oracle']:
        errs = np.array(results[name])
        if len(errs) == 0:
            print(f'  {name}: no data'); continue
        within2 = float(np.mean(errs <= 2.0)) * 100
        within5 = float(np.mean(errs <= 5.0)) * 100
        print(f'  {name:8s}: n={len(errs):3d}  median_err={np.median(errs):7.2f} ft  '
              f'mean_err={errs.mean():7.2f} ft  within_2ft={within2:5.1f}%  within_5ft={within5:5.1f}%')
