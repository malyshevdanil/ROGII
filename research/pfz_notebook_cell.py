"""Submission cell for pf_z confidence-weighted blend: pf_z (Z-velocity-coupled PF, a SECOND independent
particle filter with a different motion model than pf_ancc) blended with confidence based on its OWN
particle-spread std output (fade toward the current best track when the PF is locally uncertain) --
the same pattern that made neighbor-transfer strong. corr(err, current_best_err)=0.623 on the true
160-well holdout (higher/less-decorrelated than beam2 or neighbor, but the CLEANEST validation of the
whole session): paired bootstrap at w=0.05: +0.084ft, 95%CI=[+0.010,+0.164], 98.5% positive -- CI
CLEARLY excludes zero. Tail-risk is the cleanest found this session: max single-well worsening only
+0.84ft (at the combined weight used in deployment). Nearly uncorrelated with neighbor-transfer
(corr=0.026) and moderately correlated with beam2 (corr=0.394) -- still adds real incremental value in
the combined 3-signal blend (see the combined validation: +0.202ft, 95%CI=[+0.091,+0.313], 100%
positive at the conservative weight point).
"""

CELL_PFZ_BLEND = '''# --- PF_Z CONFIDENCE-WEIGHTED BLEND (second independent PF, different motion model) --------------
# pf_z: a genuinely independent particle filter (Z-velocity-coupled motion model, distinct from
# pf_ancc's pure-momentum state) blended with confidence based on its OWN particle-spread std (fades
# toward the current best track when the PF is locally uncertain -- same pattern that made
# neighbor-transfer strong). corr(err, current_best_err)=0.623 on the true 160-well holdout but the
# CLEANEST bootstrap CI of the whole session: w=0.05 gives +0.084ft, 95%%CI=[+0.010,+0.164], 98.5%%
# positive -- clearly excludes zero. Tail-risk max single-well worsening only +0.84ft (combined blend).
# Nearly uncorrelated with neighbor-transfer (0.026), moderately correlated with beam2 (0.394) -- still
# adds real incremental value combined with both (see track6/beam2/neighbor commits for the full story).
_PFZ_BLEND_W = float(os.environ.get("ROGII_PFZ_BLEND", "0.05"))
if _PFZ_BLEND_W > 0:
    try:
        import numpy as _pzp, pandas as _pzpd
        from numba import njit as _pznjit

        _PZ_N = 600
        _PZ_MOM, _PZ_VN, _PZ_PN = 0.993, 0.005, 0.01
        _PZ_GR_WIN, _PZ_GR_WT, _PZ_RESAMP = 5, 0.3, 0.5
        _PZ_ROUGH_P, _PZ_ROUGH_V = 0.2, 0.003
        _PZ_GR_SIG_MIN, _PZ_GR_SIG_MAX, _PZ_GR_SIG_DEF = 10., 60., 30.
        _PZ_STD_SCALE = 2.614363968372345  # ~3x the median particle-spread std on the true holdout

        @_pznjit(cache=True)
        def _pz_interp1(grid, v, vmin, step):
            i = int((v - vmin) / step)
            if i < 0: return grid[0]
            n = len(grid) - 1
            if i >= n: return grid[n]
            t = (v - vmin) / step - i
            return grid[i] * (1. - t) + grid[i + 1] * t

        @_pznjit(cache=True)
        def _pz_resamp(pos, aux, w, N, rp, rv):
            cum = _pzp.zeros(N + 1)
            for j in range(N): cum[j + 1] = cum[j] + w[j]
            u0 = _pzp.random.uniform(0., 1. / N); np2 = _pzp.empty(N); na = _pzp.empty(N); ci = 0
            for j in range(N):
                u = u0 + j / N
                while ci < N - 1 and cum[ci + 1] < u: ci += 1
                np2[j] = pos[ci] + rp * _pzp.random.randn(); na[j] = aux[ci] + rv * _pzp.random.randn()
            return np2, na

        @_pznjit(cache=True)
        def _pz_pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv, beta, icpt, zsig, N,
                     MOM, VN, PN, GR_WT, RP, RV, RESAMP):
            pos = _pzp.empty(N); vel = _pzp.empty(N); w = _pzp.ones(N) / N
            for j in range(N):
                pos[j] = ip + 0.5 * _pzp.random.randn(); vel[j] = iv + 0.02 * _pzp.random.randn()
            pts = _pzp.empty(len(md_v)); std_ = _pzp.empty(len(md_v)); pm = md_v[0] - 1.; pz = z_v[0] - 1.
            for i in range(len(md_v)):
                dm = md_v[i] - pm; dm = max(dm, 1.); dzd = (z_v[i] - pz) / dm; ve = beta * dzd + icpt
                for j in range(N):
                    vel[j] = MOM * vel[j] + VN * _pzp.random.randn(); pos[j] += vel[j] * dm + PN * _pzp.random.randn()
                    pos[j] = max(pos[j], vmin - 50.); pos[j] = min(pos[j], vmin + len(gg_p) * step + 50.)
                if not _pzp.isnan(gr_v[i]):
                    ws = 0.
                    for j in range(N):
                        ep = _pz_interp1(gg_p, pos[j], vmin, step); dp = (gr_v[i] - ep) / gs
                        lp = max(_pzp.exp(-0.5 * dp * dp) if dp * dp < 600. else 0., 1e-300)
                        if not _pzp.isnan(gr_sm_v[i]):
                            es = _pz_interp1(gg_s, pos[j], vmin, step); ds = (gr_sm_v[i] - es) / (gs * 1.5)
                            lsm = max(_pzp.exp(-0.5 * ds * ds) if ds * ds < 600. else 0., 1e-300); lk = (1. - GR_WT) * lp + GR_WT * lsm
                        else: lk = lp
                        lk = max(lk, 1e-300); w[j] *= lk; ws += w[j]
                    if ws > 0.:
                        for j in range(N): w[j] /= ws
                    else:
                        for j in range(N): w[j] = 1. / N
                ws2 = 0.
                for j in range(N):
                    dv = (vel[j] - ve) / max(zsig * 2., 0.005); lz = max(_pzp.exp(-0.5 * dv * dv) if dv * dv < 600. else 0., 1e-300)
                    w[j] *= lz; ws2 += w[j]
                if ws2 > 0.:
                    for j in range(N): w[j] /= ws2
                else:
                    for j in range(N): w[j] = 1. / N
                ne = 0.
                for j in range(N): ne += w[j] * w[j]
                if 1. / ne < RESAMP * N:
                    pos, vel = _pz_resamp(pos, vel, w, N, RP, RV)
                    for j in range(N): w[j] = 1. / N
                wm = 0.
                for j in range(N): wm += w[j] * pos[j]
                pts[i] = wm; va = 0.
                for j in range(N): va += w[j] * (pos[j] - wm) ** 2
                std_[i] = va ** 0.5; pm = md_v[i]; pz = z_v[i]
            return pts, std_

        def _pz_seed_jit_inner(seed):
            _pzp.random.seed(seed)
        _pz_seed_jit = _pznjit(cache=True)(_pz_seed_jit_inner)
        _pz_seed_jit(42)

        def _pz_grid(tw_tvt, tw_gr, step=0.2):
            tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
            tvt_g = _pzp.arange(tmin, tmax + step, step)
            return _pzp.interp(tvt_g, tw_tvt, tw_gr).astype(_pzp.float64), float(tmin), float(step)

        def _pz_gr_sig(hw, tw_tvt, tw_gr):
            kn = hw[hw.TVT_input.notna() & hw.GR.notna()]
            if len(kn) < 20: return float(_PZ_GR_SIG_DEF)
            return float(_pzp.clip(_pzp.std(kn.GR.values - _pzp.interp(kn.TVT_input.values, tw_tvt, tw_gr)),
                                    _PZ_GR_SIG_MIN, _PZ_GR_SIG_MAX))

        def _pz_run_pf_z(hw, tw_tvt, tw_gr, N=_PZ_N):
            gs = _pz_gr_sig(hw, tw_tvt, tw_gr)
            tw_s = _pzpd.Series(tw_gr).rolling(_PZ_GR_WIN, center=True, min_periods=1).mean().values.astype(_pzp.float32)
            kna = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
            if len(ev) == 0: return _pzp.array([]), _pzp.array([])
            dz_k = _pzp.diff(kna.Z.values); dvt = _pzp.diff(kna.TVT_input.values); dmd_k = _pzp.diff(kna.MD.values); m2 = dmd_k > 0
            if m2.sum() >= 10:
                vz = dz_k[m2] / dmd_k[m2]; vt = dvt[m2] / dmd_k[m2]; A = _pzp.column_stack([vz, _pzp.ones_like(vz)])
                c, _, _, _ = _pzp.linalg.lstsq(A, vt, rcond=None)
                beta, icpt, zsig = float(c[0]), float(c[1]), max(float(_pzp.std(vt - (c[0] * vz + c[1]))), 0.001)
            else:
                beta, icpt, zsig = -1., 0., 0.1
            t2 = kna.tail(20); dvt2 = _pzp.diff(t2.TVT_input.values); dmd2 = _pzp.diff(t2.MD.values); m3 = dmd2 > 0
            iv = float(_pzp.median(dvt2[m3] / dmd2[m3])) if m3.sum() >= 3 else 0.
            gg, gmin, gst = _pz_grid(tw_tvt, tw_gr); gs2, _, _ = _pz_grid(tw_tvt, tw_s)
            gr_sm = hw.GR.rolling(_PZ_GR_WIN, center=True, min_periods=1).mean()
            pts, std = _pz_pf_z(ev.MD.values.astype(_pzp.float64), ev.Z.values.astype(_pzp.float64), ev.GR.values.astype(_pzp.float64),
                                 gr_sm.loc[ev.index].values.astype(_pzp.float64), gg, gs2, gmin, gst, gs,
                                 float(kna.TVT_input.iloc[-1]), iv, beta, icpt, zsig, N,
                                 _PZ_MOM, _PZ_VN, _PZ_PN, _PZ_GR_WT, _PZ_ROUGH_P, _PZ_ROUGH_V, _PZ_RESAMP)
            return pts.astype(_pzp.float32), std.astype(_pzp.float32)

        _pz_pred_by_id = {}
        for _wid in list_wells("test"):
            hw = _pzpd.read_csv(CFG.DATA / "test" / f"{_wid}__horizontal_well.csv")
            tw = _pzpd.read_csv(CFG.DATA / "test" / f"{_wid}__typewell.csv").sort_values("TVT")
            tw_tvt = tw["TVT"].values.astype(float); tw_gr = tw["GR"].fillna(tw["GR"].mean()).values.astype(float)
            if len(tw_tvt) < 10: continue
            km = hw["TVT_input"].notna()
            if km.sum() < 20 or hw["TVT_input"].isna().sum() < 20: continue
            pts, std = _pz_run_pf_z(hw, tw_tvt, tw_gr)
            if len(pts) == 0: continue
            ev_idx = _pzp.where(~km.values)[0]
            conf = _pzp.exp(-std / _PZ_STD_SCALE)
            raw_ids = [f"{_wid}_{_r}" for _r in ev_idx]
            for _rid, _pv, _cf in zip(raw_ids, pts, conf):
                _pz_pred_by_id[_rid] = (float(_pv), float(_cf))

        if _pz_pred_by_id:
            _pzb = _pzpd.read_csv(OUT / "submission.csv")
            _pz_tvt = _pzb["tvt"].to_numpy(float).copy()
            _pz_ids = _pzb["id"].astype(str).to_numpy()
            _n_touched6 = 0
            for _i, _id in enumerate(_pz_ids):
                if _id in _pz_pred_by_id:
                    _pv, _cf = _pz_pred_by_id[_id]
                    _eff_w = _PFZ_BLEND_W * _cf
                    _pz_tvt[_i] = (1.0 - _eff_w) * _pz_tvt[_i] + _eff_w * _pv
                    _n_touched6 += 1
            _pzb["tvt"] = _pz_tvt
            _order_ids6 = _pzpd.read_csv(CFG.DATA / "sample_submission.csv")["id"].astype(str)
            _pzb = _pzb.set_index("id").reindex(_order_ids6).reset_index()
            assert _pzb["tvt"].notna().all(), "pfz blend lost ids vs sample"
            _pzb[["id", "tvt"]].to_csv(OUT / "submission.csv", index=False)
            POSTPROCESSORS.append(f"pfz_blend_{_PFZ_BLEND_W:.2f}")
            print(f"pfz blend W={_PFZ_BLEND_W}: touched {_n_touched6} eval rows")
        else:
            print("pfz blend: no usable test wells found (skipped, submission unchanged)")
    except Exception as _pze:
        print(f"pfz blend FAILED, skipping safely (submission unchanged): {_pze}")
else:
    print("pfz blend disabled (W=0)")
'''

if __name__ == '__main__':
    print('CELL_PFZ_BLEND length:', len(CELL_PFZ_BLEND))
