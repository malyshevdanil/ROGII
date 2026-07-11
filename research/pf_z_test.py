"""Test a NEW decorrelation source for the PF ensemble: pf_z, a Z-velocity-coupled motion model,
independently designed (not copied) after the *concept* described in a competitor writeup: instead of
pure momentum persistence on the dip rate (our production PF: rate = MOM*rate + VN*noise), couple the
proposed rate toward beta*dz/dMD + intercept (fit per-well on the known prefix), so the second PF's prior
comes from an INDEPENDENT physical source (trajectory curvature) rather than a re-tuned version of the
same momentum random walk. Measure error correlation vs the production PF and the honest blend gain,
using our own whole-well holdout (not Kaggle), mirroring prep.py's split.
"""
import numpy as np, pandas as pd, glob, os, time

TRAIN_DIR = 'd:/ROGII/data/train'

def run_pf(hw, tw, n_particles=400, seed=42, MOM=0.998, VN=0.002, PN=0.005, mode='base'):
    """mode='base': production near-constant-dip PF. mode='z': Z-velocity-coupled motion model
    (rate pulled toward beta*dzdmd + intercept each step, fit on the known prefix)."""
    tw_s = tw.sort_values('TVT'); tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0: return hw['TVT_input'].values.astype(float).copy()
    last = kn.iloc[-1]; last_tvt = float(last['TVT_input']); last_Z = float(last['Z']); last_MD = float(last['MD'])
    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), 10., 60.))
    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values); dz = np.diff(tail['Z'].values); dm = np.diff(tail['MD'].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    # --- Z-velocity coupling: fit beta,intercept regressing known-zone dS/dMD on dZ/dMD ---
    beta, intercept = 0.0, ir
    if mode == 'z':
        kn_all = kn
        if len(kn_all) > 40:
            kMD = kn_all['MD'].values.astype(float); kZ = kn_all['Z'].values.astype(float)
            kTVT = kn_all['TVT_input'].values.astype(float)
            mdd = np.gradient(kMD); mdd[mdd == 0] = 1
            dzdmd_kn = np.gradient(kZ) / mdd
            dSdmd_kn = np.gradient(kTVT + kZ) / mdd
            v = np.isfinite(dzdmd_kn) & np.isfinite(dSdmd_kn)
            if v.sum() > 30 and np.std(dzdmd_kn[v]) > 1e-6:
                beta, intercept = np.polyfit(dzdmd_kn[v], dSdmd_kn[v], 1)

    N = n_particles; rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z
    pos = ls + 4.5 * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    RP, RR, RESAMP = 0.1, 0.001, 0.5
    md_v = ev['MD'].values.astype(float); z_v = ev['Z'].values.astype(float)
    gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]
    mdd_ev = np.gradient(hw['MD'].values.astype(float))[list(ev.index)] if mode == 'z' else None
    z_grad_ev = np.gradient(hw['Z'].values.astype(float))[list(ev.index)] if mode == 'z' else None

    out_vals = hw['TVT_input'].values.astype(float).copy()
    res = np.empty(len(ev)); prev_MD = last_MD
    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        if mode == 'z':
            local_dzdmd = float(z_grad_ev[i] / max(mdd_ev[i], 1e-6)) if mdd_ev[i] != 0 else 0.0
            local_dzdmd = np.clip(local_dzdmd, -5, 5)
            target_rate = beta * local_dzdmd + intercept
            rate = 0.85 * rate + 0.15 * target_rate + VN * rng.standard_normal(N)
        else:
            rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d ** 2, 600.)); lk = np.maximum(lk, 1e-300)
        w = w * lk; ws = w.sum(); w = w / ws if ws > 0 else np.ones(N) / N
        n_eff = 1.0 / (w ** 2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N); rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N
        res[i] = float(np.dot(w, pos - z_v[i])); prev_MD = md_v[i]
    out_vals[list(ev.index)] = res
    return out_vals

def pf_ensemble(hw, tw, mode, n_seeds=12, n_particles=300):
    preds = [run_pf(hw, tw, n_particles=n_particles, seed=s, mode=mode) for s in range(n_seeds)]
    return np.mean(preds, axis=0)

if __name__ == '__main__':
    wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
    rng = np.random.default_rng(42); idx = np.arange(len(wids)); rng.shuffle(idx)
    VA = [wids[i] for i in idx[:120]]   # subset for speed
    t0 = time.time()
    err_base = []; err_z = []; err_mean = []
    for j, wid in enumerate(VA):
        hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
        tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv')
        if 'TVT' not in hw.columns: continue
        ev_mask = hw['TVT_input'].isna().values
        if ev_mask.sum() < 20: continue
        true = hw['TVT'].values.astype(float)[ev_mask]
        p_base = pf_ensemble(hw, tw, 'base', n_seeds=12, n_particles=300)[ev_mask]
        p_z = pf_ensemble(hw, tw, 'z', n_seeds=12, n_particles=300)[ev_mask]
        err_base.append(p_base - true); err_z.append(p_z - true); err_mean.append((p_base + p_z) / 2 - true)
        if (j + 1) % 20 == 0: print('  ', j + 1, '/', len(VA), '%.0fs' % (time.time() - t0), flush=True)
    eb = np.concatenate(err_base); ez = np.concatenate(err_z); em = np.concatenate(err_mean)
    print('\nwells used:', len(err_base))
    print('pooled RMSE base :', np.sqrt(np.mean(eb ** 2)))
    print('pooled RMSE pf_z :', np.sqrt(np.mean(ez ** 2)))
    print('pooled RMSE mean(base,z):', np.sqrt(np.mean(em ** 2)))
    print('error correlation(base, pf_z):', np.corrcoef(eb, ez)[0, 1])
