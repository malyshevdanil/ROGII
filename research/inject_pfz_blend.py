import json

ROOT = 'd:/ROGII'
SRC = f'{ROOT}/7.178_moreseeds192_warpblend_physicspp.ipynb'
DST = f'{ROOT}/7.178_moreseeds192_full_stack.ipynb'

nb = json.load(open(SRC, encoding='utf-8'))

pfz_cell_src = '''# --- PF_Z BLEND: our own pipeline's Z-velocity-coupled PF, currently unused beyond a GBM feature ------
# Our inherited pipeline already computes a second, independently-motivated particle filter (pf_z) inside
# build_well -- its motion model couples the proposed velocity toward beta*dZ/dMD + intercept (fit per
# well on the known prefix) via a soft Bayesian velocity-consistency likelihood, rather than the main
# tracker's near-constant-dip momentum random walk. Until now pf_z's output only fed the (largely inactive)
# GBM stack as a feature ("pf_z", "pf_z_delta", "pf_vs_z") -- it was never blended directly into the SP45
# prediction. We measured it: on a 160-well holdout, error correlation between pf_ancc and pf_z is only
# 0.47 (vs 0.82-0.94 for every previously-tried decorrelated partner -- process-noise scale, GR-sensitivity,
# typewell-jitter, all §8.3), and blending gives a real single-seed gain (pooled 13.49 -> 12.05 at a=0.4,
# -10.7% relative). This is the single most decorrelated, most promising lever surfaced in this study.
# Seeds are averaged to tame pf_z's own stochastic noise (same variance-reduction logic as the main
# ensemble, §7); the blend weight is conservative relative to the single-seed prototype result because the
# real sp45 baseline here is a well-tuned multi-seed ensemble, not a single pf_ancc run. Placed after the
# WARP blend and physics post-process, before guarded_contact_override (same protection logic).
# Parallelized across test wells via joblib (process-based "loky" backend, matching the main SP45
# stage's own reasoning: our njit kernels are plain, not nogil=True, so they hold the GIL and
# threads would not overlap CPU work) -- keeps this stage's wall-clock cost small and safe under
# the 9-hour Code Requirements regardless of per-core CPU speed on the day's Kaggle instance.
_PFZ_BLEND_A = float(os.environ.get("ROGII_PFZ_BLEND", "0.15"))
_PFZ_SEEDS = int(os.environ.get("ROGII_PFZ_SEEDS", "16"))
if _PFZ_BLEND_A > 0:
    try:
        import numpy as _zn, pandas as _zpd
        from joblib import Parallel, delayed
        from numba import njit as _znjit

        _PFZ_N = 600
        _PFZ_MOM, _PFZ_VN, _PFZ_PN = 0.993, 0.005, 0.01
        _PFZ_GR_SIG_MIN, _PFZ_GR_SIG_MAX, _PFZ_GR_SIG_DEF = 10., 60., 30.
        _PFZ_GR_WIN, _PFZ_GR_WT, _PFZ_RESAMP, _PFZ_ROUGH_P, _PFZ_ROUGH_V = 5, 0.3, 0.5, 0.2, 0.003

        @_znjit
        def _zinterp1(grid, v, vmin, step):
            i = int((v - vmin) / step)
            if i < 0: return grid[0]
            n = len(grid) - 1
            if i >= n: return grid[n]
            t = (v - vmin) / step - i
            return grid[i] * (1. - t) + grid[i + 1] * t

        @_znjit
        def _zresamp(pos, aux, w, N, rp, rv):
            cum = _zn.zeros(N + 1)
            for j in range(N): cum[j + 1] = cum[j] + w[j]
            u0 = _zn.random.uniform(0., 1. / N); np2 = _zn.empty(N); na = _zn.empty(N); ci = 0
            for j in range(N):
                u = u0 + j / N
                while ci < N - 1 and cum[ci + 1] < u: ci += 1
                np2[j] = pos[ci] + rp * _zn.random.randn(); na[j] = aux[ci] + rv * _zn.random.randn()
            return np2, na

        @_znjit
        def _pfz_core(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv, beta, icpt, zsig, N,
                      MOM, VN, PN, GR_WT, RP, RV, RESAMP):
            pos = _zn.empty(N); vel = _zn.empty(N); w = _zn.ones(N) / N
            for j in range(N):
                pos[j] = ip + 0.5 * _zn.random.randn(); vel[j] = iv + 0.02 * _zn.random.randn()
            pts = _zn.empty(len(md_v)); pm = md_v[0] - 1.; pz = z_v[0] - 1.
            for i in range(len(md_v)):
                dm = md_v[i] - pm; dm = max(dm, 1.); dzd = (z_v[i] - pz) / dm; ve = beta * dzd + icpt
                for j in range(N):
                    vel[j] = MOM * vel[j] + VN * _zn.random.randn(); pos[j] += vel[j] * dm + PN * _zn.random.randn()
                    pos[j] = max(pos[j], vmin - 50.); pos[j] = min(pos[j], vmin + len(gg_p) * step + 50.)
                if not _zn.isnan(gr_v[i]):
                    ws = 0.
                    for j in range(N):
                        ep = _zinterp1(gg_p, pos[j], vmin, step); dp = (gr_v[i] - ep) / gs
                        lp = max(_zn.exp(-0.5 * dp * dp) if dp * dp < 600. else 0., 1e-300)
                        if not _zn.isnan(gr_sm_v[i]):
                            es = _zinterp1(gg_s, pos[j], vmin, step); ds = (gr_sm_v[i] - es) / (gs * 1.5)
                            lsm = max(_zn.exp(-0.5 * ds * ds) if ds * ds < 600. else 0., 1e-300)
                            lk = (1. - GR_WT) * lp + GR_WT * lsm
                        else:
                            lk = lp
                        lk = max(lk, 1e-300); w[j] *= lk; ws += w[j]
                    if ws > 0.:
                        for j in range(N): w[j] /= ws
                    else:
                        for j in range(N): w[j] = 1. / N
                ws2 = 0.
                for j in range(N):
                    dv = (vel[j] - ve) / max(zsig * 2., 0.005)
                    lz = max(_zn.exp(-0.5 * dv * dv) if dv * dv < 600. else 0., 1e-300)
                    w[j] *= lz; ws2 += w[j]
                if ws2 > 0.:
                    for j in range(N): w[j] /= ws2
                else:
                    for j in range(N): w[j] = 1. / N
                ne = 0.
                for j in range(N): ne += w[j] * w[j]
                if 1. / ne < RESAMP * N:
                    pos, vel = _zresamp(pos, vel, w, N, RP, RV)
                    for j in range(N): w[j] = 1. / N
                wm = 0.
                for j in range(N): wm += w[j] * pos[j]
                pts[i] = wm; pm = md_v[i]; pz = z_v[i]
            return pts

        def _zgrid(tw_tvt, tw_gr, step=0.2):
            tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
            tvt_g = _zn.arange(tmin, tmax + step, step)
            return _zn.interp(tvt_g, tw_tvt, tw_gr).astype(_zn.float64), float(tmin), float(step)

        def _zgrsig(hw, tw_tvt, tw_gr):
            kn = hw[hw.TVT_input.notna() & hw.GR.notna()]
            if len(kn) < 20: return float(_PFZ_GR_SIG_DEF)
            return float(_zn.clip(_zn.std(kn.GR.values - _zn.interp(kn.TVT_input.values, tw_tvt, tw_gr)),
                                   _PFZ_GR_SIG_MIN, _PFZ_GR_SIG_MAX))

        def _run_pfz_seeds(hw, tw_tvt, tw_gr, n_seeds, N=_PFZ_N):
            gs = _zgrsig(hw, tw_tvt, tw_gr)
            tw_s = _zpd.Series(tw_gr).rolling(_PFZ_GR_WIN, center=True, min_periods=1).mean().values.astype(_zn.float32)
            kna = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
            if len(ev) == 0: return None
            dz_k = _zn.diff(kna.Z.values); dvt = _zn.diff(kna.TVT_input.values); dmd_k = _zn.diff(kna.MD.values)
            m2 = dmd_k > 0
            if m2.sum() >= 10:
                vz = dz_k[m2] / dmd_k[m2]; vt = dvt[m2] / dmd_k[m2]
                A = _zn.column_stack([vz, _zn.ones_like(vz)])
                c, _, _, _ = _zn.linalg.lstsq(A, vt, rcond=None)
                beta, icpt, zsig = float(c[0]), float(c[1]), max(float(_zn.std(vt - (c[0] * vz + c[1]))), 0.001)
            else:
                beta, icpt, zsig = -1., 0., 0.1
            t2 = kna.tail(20); dvt2 = _zn.diff(t2.TVT_input.values); dmd2 = _zn.diff(t2.MD.values); m3 = dmd2 > 0
            iv = float(_zn.median(dvt2[m3] / dmd2[m3])) if m3.sum() >= 3 else 0.
            gg, gmin, gst = _zgrid(tw_tvt, tw_gr); gs2, _, _ = _zgrid(tw_tvt, tw_s)
            gr_sm = hw.GR.rolling(_PFZ_GR_WIN, center=True, min_periods=1).mean()
            md_v = ev.MD.values.astype(_zn.float64); z_v = ev.Z.values.astype(_zn.float64)
            gr_v = ev.GR.values.astype(_zn.float64); gr_sm_v = gr_sm.loc[ev.index].values.astype(_zn.float64)
            ip = float(kna.TVT_input.iloc[-1])
            accum = _zn.zeros(len(ev))
            for s in range(n_seeds):
                _zn.random.seed(s)
                pts = _pfz_core(md_v, z_v, gr_v, gr_sm_v, gg, gs2, gmin, gst, gs, ip, iv, beta, icpt, zsig,
                                 _PFZ_N, _PFZ_MOM, _PFZ_VN, _PFZ_PN, _PFZ_GR_WT, _PFZ_ROUGH_P, _PFZ_ROUGH_V, _PFZ_RESAMP)
                accum += _zn.asarray(pts)
            pred = accum / n_seeds
            out = hw['TVT_input'].values.astype(float).copy()
            out[list(ev.index)] = pred
            return out

        def _pfz_process_well(wid):
            # runs in a worker process (loky backend) -- must be self-contained / picklable.
            try:
                hw_te = _zpd.read_csv(CFG.DATA / "test" / f"{wid}__horizontal_well.csv")
                tw_te = _zpd.read_csv(CFG.DATA / "test" / f"{wid}__typewell.csv").sort_values("TVT")
                tt = tw_te["TVT"].values.astype(_zn.float32)
                tg = tw_te["GR"].fillna(tw_te["GR"].mean()).values.astype(_zn.float32)
                kn = hw_te[hw_te["TVT_input"].notna()]
                if len(kn) < 20 or hw_te["TVT_input"].isna().sum() < 20:
                    return wid, None
                pfz_full = _run_pfz_seeds(hw_te, tt, tg, _PFZ_SEEDS)
                return wid, pfz_full
            except Exception:
                return wid, None

        _zb = _zpd.read_csv(OUT / "submission.csv")
        _zb_well, _zb_ri = split_id(_zb["id"]); _zb["well"] = _zb_well; _zb["row_idx"] = _zb_ri
        _zb_tvt = _zb["tvt"].to_numpy(float).copy()
        _n_pfz = 0
        _test_wells = list_wells("test")
        # process-based parallelism (loky): our njit kernels are plain (no nogil=True), so they
        # hold the GIL -- threads would not overlap CPU work, matching the main SP45 stage's own
        # reasoning (cell that builds build_sp45_candidate).
        _n_workers_pfz = min(CFG.n_jobs, len(_test_wells)) if _test_wells else 1
        _pfz_results = Parallel(n_jobs=_n_workers_pfz, backend="loky")(
            delayed(_pfz_process_well)(wid) for wid in _test_wells
        )
        for _wid, _pfz_full in _pfz_results:
            if _pfz_full is None:
                continue
            _g = _zb[_zb["well"] == _wid]
            _pos = _g.index.to_numpy()             # positional index into _zb / _zb_tvt (RangeIndex from read_csv)
            _ridx = _g["row_idx"].to_numpy()
            _valid = (_ridx >= 0) & (_ridx < len(_pfz_full))
            _pos = _pos[_valid]; _ridx = _ridx[_valid]
            _zb_tvt[_pos] = (1 - _PFZ_BLEND_A) * _zb_tvt[_pos] + _PFZ_BLEND_A * _pfz_full[_ridx]
            _n_pfz += 1
        _zb["tvt"] = _zb_tvt
        _order_ids = _zpd.read_csv(CFG.DATA / "sample_submission.csv")["id"].astype(str)
        _zb = _zb.set_index("id").reindex(_order_ids).reset_index()
        assert _zb["tvt"].notna().all(), "pf_z blend lost ids vs sample"
        _zb[["id", "tvt"]].to_csv(OUT / "submission.csv", index=False)
        POSTPROCESSORS.append(f"pfz_blend_{_PFZ_BLEND_A:.2f}_seeds{_PFZ_SEEDS}")
        print(f"pf_z blend: A={_PFZ_BLEND_A} seeds={_PFZ_SEEDS} applied to {_n_pfz} wells")
    except Exception as _zpz_e:
        print(f"pf_z blend FAILED, skipping safely (submission unchanged): {_zpz_e}")
else:
    print("pf_z blend disabled (A=0)")
'''

def make_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in src.rstrip("\n").split("\n")]}

pp_idx = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and 'PHYSICS POST-PROCESS' in ''.join(c['source']):
        pp_idx = i
        break
assert pp_idx is not None, "physics-pp cell not found"
insert_at = pp_idx + 1
nxt = ''.join(nb['cells'][insert_at]['source'])
assert 'def guarded_contact_override' in nxt, f"unexpected next cell:\n{nxt[:200]}"

nb['cells'].insert(insert_at, make_cell(pfz_cell_src))

json.dump(nb, open(DST, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
import os
print('wrote', DST, 'size(MB)=', round(os.path.getsize(DST) / 1e6, 2), 'cells=', len(nb['cells']))
