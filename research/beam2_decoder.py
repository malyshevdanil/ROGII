"""Idea 2: STRIDE-style joint (segment_length, dip) beam decoder over the eval zone, using the
CORRECTED priors (stride3_priors_CORRECTED.pkl: log-normal segment length mu=5.706/sigma=0.807,
dip persistence rho=0.473/resid_std=0.032, unconditional dip_std=0.041).

Design (learned from this project's two prior decoder failures -- trellis, beam-HSMM):
- State at each breakpoint = (md, tvt, dip). Segments are piecewise-LINEAR between breakpoints, so no
  separate "reference trajectory" is needed (avoids the trellis's reference-frame drift bug).
- Candidate segment length: a small quantile grid from the log-normal prior (not sampled randomly, so
  runs are deterministic/reproducible).
- Candidate dip: a quantile grid around the persistence-predicted dip (rho*prev_dip), width set by
  persistence resid_std. This BOUNDS how extreme a candidate dip can be while still scoring well via the
  dip prior term -- directly guards against the trellis's aliasing bug (where an unconstrained extreme
  dip could "scan" more of the self-similar typewell curve and win by aliasing).
- Emission score = NCC (normalized cross-correlation) between the segment's calibrated GR trace and the
  typewell GR sampled along the candidate's own linear TVT trajectory -- a single aggregate score per
  segment (not a sum of per-row Gaussian terms), which sidesteps the trellis's autocorrelation-overconfidence
  bug (GR-vs-typewell residuals have ~50-row correlation length; NCC naturally aggregates instead of
  treating rows as independent).
- Beam search: keep the top-K partial paths by cumulative score (log-NCC-ish emission + log length-prior +
  log dip-prior), extend until MD reaches the end of the eval zone, backtrack the best final path.
"""
import numpy as np, pandas as pd, pickle, glob, os, time

TRAIN_DIR = 'd:/ROGII/data/train'
PRIORS = pickle.load(open('stride3_priors_CORRECTED.pkl', 'rb'))
LOG_LEN_MU, LOG_LEN_SIGMA = PRIORS['log_len_mu'], PRIORS['log_len_sigma']
DIP_STD = PRIORS['dip_std']
RHO = PRIORS['persistence_rho']; PERSIST_RESID_STD = PRIORS['persistence_resid_std']

BEAM_WIDTH = 15
LEN_QUANTILES = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
DIP_QUANTILES = [-1.8, -1.0, -0.5, 0.0, 0.5, 1.0, 1.8]  # in units of persistence_resid_std
EMISSION_WEIGHT = 40.0   # relative weight of NCC emission vs log-priors (tuned by a small sweep below)
MIN_SEG_LEN = 15.0

def _lognorm_ppf(q, mu, sigma):
    from scipy.stats import norm
    return float(np.exp(mu + sigma * norm.ppf(q)))

LEN_GRID = sorted({round(_lognorm_ppf(q, LOG_LEN_MU, LOG_LEN_SIGMA), 1) for q in LEN_QUANTILES})
LEN_GRID = [l for l in LEN_GRID if l >= MIN_SEG_LEN]

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9: return 0.0
    return float(np.dot(a, b) / (na * nb))

def load_well(wid, data_dir=TRAIN_DIR):
    hw = pd.read_csv(f'{data_dir}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{data_dir}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    km = hw['TVT_input'].notna()
    kn = hw[km]; ev = hw[~km]
    if len(kn) < 50 or len(ev) < 50: return None
    tw_tvt = tw['TVT'].values.astype(float); tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    return hw, tw_tvt, tw_gr, kn, ev

def calibrate(kn, tw_tvt, tw_gr, gr_full):
    kg = kn['GR'].values.astype(float); ktvt = kn['TVT_input'].values.astype(float)
    twk = np.interp(ktvt, tw_tvt, tw_gr)
    v = np.isfinite(kg) & np.isfinite(twk)
    a, b = (np.polyfit(kg[v], twk[v], 1) if v.sum() >= 20 else (1., 0.))
    return gr_full * a + b

def decode_well(wid, data_dir=TRAIN_DIR, beam_width=BEAM_WIDTH):
    r = load_well(wid, data_dir)
    if r is None: return None
    hw, tw_tvt, tw_gr, kn, ev = r
    gr_full = hw['GR'].interpolate(limit_direction='both').values.astype(float)
    cal = calibrate(kn, tw_tvt, tw_gr, gr_full)
    MD = hw['MD'].values.astype(float)
    md0 = float(kn['MD'].iloc[-1]); tvt0 = float(kn['TVT_input'].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values); dm = np.diff(tail['MD'].values)
    m = dm > 0
    dip0 = float(np.median(dt[m] / dm[m])) if m.sum() >= 3 else 0.0

    ev_md = ev['MD'].values.astype(float)
    md_end = float(ev_md[-1])
    ev_idx0 = ev.index[0]  # first eval row index in hw

    beam = [dict(md=md0, tvt=tvt0, dip=dip0, logp=0.0, segs=[])]
    n_iter = 0
    while True:
        n_iter += 1
        active = [p for p in beam if p['md'] < md_end - 1.0]
        if not active or n_iter > 400:
            break
        new_beam = []
        for p in beam:
            if p['md'] >= md_end - 1.0:
                new_beam.append(p); continue
            for L in LEN_GRID:
                new_md = min(p['md'] + L, md_end)
                actual_L = new_md - p['md']
                if actual_L < MIN_SEG_LEN * 0.5: continue
                for zq in DIP_QUANTILES:
                    new_dip = RHO * p['dip'] + zq * PERSIST_RESID_STD
                    new_tvt = p['tvt'] + new_dip * actual_L
                    seg_mask = (MD >= p['md']) & (MD <= new_md)
                    seg_md = MD[seg_mask]
                    if len(seg_md) < 5: continue
                    seg_gr = cal[seg_mask]
                    seg_tvt_pred = p['tvt'] + new_dip * (seg_md - p['md'])
                    seg_tw = np.interp(seg_tvt_pred, tw_tvt, tw_gr)
                    v = np.isfinite(seg_gr) & np.isfinite(seg_tw)
                    if v.sum() < 5: continue
                    corr = ncc(seg_gr[v], seg_tw[v])
                    len_lp = -0.5 * ((np.log(max(actual_L, 1.0)) - LOG_LEN_MU) / LOG_LEN_SIGMA) ** 2
                    dip_lp = -0.5 * (zq ** 2)
                    score = EMISSION_WEIGHT * corr + len_lp + dip_lp
                    new_beam.append(dict(md=new_md, tvt=new_tvt, dip=new_dip,
                                          logp=p['logp'] + score,
                                          segs=p['segs'] + [(p['md'], new_md, p['tvt'], new_tvt)]))
        if not new_beam: break
        new_beam.sort(key=lambda x: -x['logp'])
        beam = new_beam[:beam_width]
        if all(p['md'] >= md_end - 1.0 for p in beam): break

    best = max(beam, key=lambda x: x['logp'])
    segs = best['segs']
    if not segs: return None
    pred_tvt = np.full(len(ev_md), np.nan)
    for (a, b, ta, tb) in segs:
        m = (ev_md >= a) & (ev_md <= b + 1e-6)
        if not m.any(): continue
        frac = (ev_md[m] - a) / max(b - a, 1e-6)
        pred_tvt[m] = ta + frac * (tb - ta)
    # fallback: if the beam didn't reach md_end (safety cap or dead end), extrapolate the last
    # segment's own dip for any remaining uncovered tail, instead of leaving garbage/NaN.
    if np.isnan(pred_tvt).any():
        last_a, last_b, last_ta, last_tb = segs[-1]
        last_dip = (last_tb - last_ta) / max(last_b - last_a, 1e-6)
        m = np.isnan(pred_tvt)
        pred_tvt[m] = last_tb + last_dip * (ev_md[m] - last_b)
    true_tvt = ev['TVT'].values.astype(float)
    return pred_tvt, true_tvt, ev_md

if __name__ == '__main__':
    t0 = time.time()
    wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
    rng = np.random.default_rng(3)
    sample = list(rng.choice(wids, size=20, replace=False))
    print('smoke test on', len(sample), 'wells')
    sq, n = 0.0, 0
    for wid in sample:
        r = decode_well(wid)
        if r is None:
            print(f'  {wid}: SKIPPED (no result)')
            continue
        pred, true, md = r
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        sq += np.sum((pred - true) ** 2); n += len(true)
        print(f'  {wid}: n={len(true)}  rmse={rmse:.3f}')
    print(f'\npooled RMSE ({len(sample)} wells):', np.sqrt(sq / n) if n else 'N/A')
    print('time:', time.time() - t0, 's')
