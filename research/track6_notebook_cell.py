"""Submission cell for the track6 project: an independent PF (pf_ancc, own momentum-free state -- NOT
derived from sp45) + multi-scale NCC correlation-search features (decoupled from sp45 -- pf_ancc used
only as a loose search center) -> GBM predicting a RESIDUAL from pf_ancc (absolute-position framing
severely overfit; residual framing + heavy regularization fixed it). Paired bootstrap on the true
160-well holdout: corr(track6_err, v7blend_err)=0.635 (architecturally more independent than the first,
failed sp45-residual attempt at 0.83-0.88), blend gain at w=0.10 = +0.118ft (95%CI touches zero slightly,
94.7% positive), tail-risk checked clean (max single-well worsening +1.77ft, no catastrophic blowups).
Runtime is negligible (~9s total for pf_ancc+corr-search on the real 3-well/14151-row test set).
Inserted AFTER the GRU-refiner blend cell so it blends against the full current-best track.
"""

CELL_TRACK6_BLEND = '''# --- TRACK6 BLEND (independent PF + multi-scale correlation-search features -> GBM residual) ------
# Second independent estimator in the spirit of the STRIDE writeup's "track6/track8": a genuinely
# separate PF (pf_ancc, own state/motion model, NOT derived from sp45) plus NCC correlation-search
# features computed in a window around pf_ancc's own estimate (used only as a loose search center, not
# a value being corrected). A GBM predicts the RESIDUAL from pf_ancc (absolute-position framing
# overfit badly: train/val RMSE gap 11.9 vs 25.8 -- residual + heavy regularization fixed it). Paired
# bootstrap on the true 160-well holdout: corr(track6_err, current_best_err)=0.635 (well below the
# first failed sp45-residual attempt's 0.83-0.88), blend gain at w=0.10 real but modest (+0.118ft,
# 94.7% positive), tail-risk clean (max single-well worsening +1.77ft). Attach the
# 'rogii-track6-gbm' Kaggle Dataset (5 fold LightGBM boosters + track6_gbm_meta.pkl).
_T6_BLEND_W = float(os.environ.get("ROGII_TRACK6_BLEND", "0.10"))
if _T6_BLEND_W > 0:
    try:
        import glob as _t6glob, numpy as _t6np, pandas as _t6pd, pickle as _t6pickle
        from numba import njit as _t6njit
        import lightgbm as _t6lgb

        _T6_ANCC_N = 600
        _T6_ANCC_ALPHA, _T6_ANCC_RN, _T6_ANCC_PN = 0.998, 0.002, 0.005
        _T6_ANCC_IS, _T6_ANCC_RP, _T6_ANCC_RR = 0.3, 0.1, 0.001
        _T6_GR_SIG_MIN, _T6_GR_SIG_MAX, _T6_GR_SIG_DEF = 10., 60., 30.
        _T6_RESAMP = 0.5
        _T6_SEARCH_HALF, _T6_SEARCH_STEP = 25.0, 1.0
        _T6_WIN_SIZES = (11, 25, 51)
        _T6_SUB_EVERY = 5

        @_t6njit(cache=True)
        def _t6_interp1(grid, v, vmin, step):
            i = int((v - vmin) / step)
            if i < 0: return grid[0]
            n = len(grid) - 1
            if i >= n: return grid[n]
            t = (v - vmin) / step - i
            return grid[i] * (1. - t) + grid[i + 1] * t

        @_t6njit(cache=True)
        def _t6_resamp(pos, aux, w, N, rp, rv):
            cum = _t6np.zeros(N + 1)
            for j in range(N): cum[j + 1] = cum[j] + w[j]
            u0 = _t6np.random.uniform(0., 1. / N); np2 = _t6np.empty(N); na = _t6np.empty(N); ci = 0
            for j in range(N):
                u = u0 + j / N
                while ci < N - 1 and cum[ci + 1] < u: ci += 1
                np2[j] = pos[ci] + rp * _t6np.random.randn(); na[j] = aux[ci] + rv * _t6np.random.randn()
            return np2, na

        @_t6njit(cache=True)
        def _t6_pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP):
            pos = _t6np.empty(N); rate = _t6np.empty(N); w = _t6np.ones(N) / N
            for j in range(N):
                pos[j] = ls + IS * _t6np.random.randn(); rate[j] = ir + 0.01 * _t6np.random.randn()
            pts = _t6np.empty(len(md_v)); pm = md_v[0] - 1.
            for i in range(len(md_v)):
                dm = md_v[i] - pm; dm = max(dm, 1.)
                for j in range(N):
                    rate[j] = ALPHA * rate[j] + RN * _t6np.random.randn(); pos[j] += rate[j] * dm + PN * _t6np.random.randn()
                    tvt_j = pos[j] - z_v[i]; tvt_j = max(tvt_j, vmin - 50.); tvt_j = min(tvt_j, vmin + len(gg) * step + 50.)
                    pos[j] = tvt_j + z_v[i]
                if not _t6np.isnan(gr_v[i]):
                    ws = 0.
                    for j in range(N):
                        eg = _t6_interp1(gg, pos[j] - z_v[i], vmin, step); d = (gr_v[i] - eg) / gs
                        lk = max(_t6np.exp(-0.5 * d * d) if d * d < 600. else 0., 1e-300); w[j] *= lk; ws += w[j]
                    if ws > 0.:
                        for j in range(N): w[j] /= ws
                    else:
                        for j in range(N): w[j] = 1. / N
                ne = 0.
                for j in range(N): ne += w[j] * w[j]
                if 1. / ne < RESAMP * N:
                    pos, rate = _t6_resamp(pos, rate, w, N, RP, RR)
                    for j in range(N): w[j] = 1. / N
                tv = 0.
                for j in range(N): tv += w[j] * (pos[j] - z_v[i])
                pts[i] = tv; pm = md_v[i]
            return pts

        def _t6_grid(tw_tvt, tw_gr, step=0.2):
            tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
            tvt_g = _t6np.arange(tmin, tmax + step, step)
            return _t6np.interp(tvt_g, tw_tvt, tw_gr).astype(_t6np.float64), float(tmin), float(step)

        def _t6_gr_sig(hw, tw_tvt, tw_gr):
            kn = hw[hw.TVT_input.notna() & hw.GR.notna()]
            if len(kn) < 20: return float(_T6_GR_SIG_DEF)
            return float(_t6np.clip(_t6np.std(kn.GR.values - _t6np.interp(kn.TVT_input.values, tw_tvt, tw_gr)),
                                     _T6_GR_SIG_MIN, _T6_GR_SIG_MAX))

        def _t6_run_pf_ancc(hw, tw_tvt, tw_gr, N=_T6_ANCC_N):
            gs = _t6_gr_sig(hw, tw_tvt, tw_gr); kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
            if len(ev) == 0: return _t6np.array([])
            ls = float(kn.TVT_input.iloc[-1] + kn.Z.iloc[-1])
            tail = kn.tail(30); dt = _t6np.diff(tail.TVT_input.values); dz = _t6np.diff(tail.Z.values)
            dm = _t6np.diff(tail.MD.values); m = dm > 0
            ir = float(_t6np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
            gg, gmin, gst = _t6_grid(tw_tvt, tw_gr)
            pts = _t6_pf_ancc(ev.MD.values.astype(_t6np.float64), ev.Z.values.astype(_t6np.float64),
                               ev.GR.values.astype(_t6np.float64), gg, gmin, gst, gs, ls, ir, N,
                               _T6_ANCC_ALPHA, _T6_ANCC_RN, _T6_ANCC_PN, _T6_ANCC_IS, _T6_ANCC_RP, _T6_ANCC_RR, _T6_RESAMP)
            return pts.astype(_t6np.float32)

        def _t6_interp_nan(a):
            a = a.copy(); n = len(a); idx = _t6np.arange(n); m = _t6np.isnan(a)
            if m.all(): return _t6np.zeros(n)
            a[m] = _t6np.interp(idx[m], idx[~m], a[~m]); return a

        def _t6_match_profile(gwin, twk_gr, tw_tvt, center_tvt, search_half, search_step):
            offsets = _t6np.arange(-search_half, search_half + search_step, search_step)
            gw = gwin - gwin.mean(); gnorm = _t6np.linalg.norm(gw)
            if gnorm < 1e-9: return 0.0, 0.0, 0.0
            corrs = _t6np.empty(len(offsets)); half = len(gwin) // 2
            for k, off in enumerate(offsets):
                tvt_pts = center_tvt + off + _t6np.arange(-half, len(gwin) - half)
                twin = _t6np.interp(tvt_pts, tw_tvt, twk_gr)
                tw_c = twin - twin.mean(); tnorm = _t6np.linalg.norm(tw_c)
                corrs[k] = float(_t6np.dot(gw, tw_c) / (gnorm * tnorm)) if tnorm > 1e-9 else 0.0
            best_i = int(_t6np.argmax(corrs)); peak = corrs[best_i]
            tmp = corrs.copy(); tmp[max(0, best_i - 2):best_i + 3] = -2
            runner_up = tmp.max() if len(tmp) > 5 else peak
            return float(offsets[best_i]), float(peak), float(peak - runner_up)

        _T6_FEATCOLS = ["md_since", "gr", "gr_grad", "gr_rstd", "z", "dzdmd", "cal_gr", "pf_ancc_tvt", "sin_azi", "cos_azi"]
        for _ws in _T6_WIN_SIZES: _T6_FEATCOLS += ["off%d" % _ws, "peak%d" % _ws, "sharp%d" % _ws]

        def _t6_build_well(hw, tw_tvt, tw_gr, pf_tvt):
            km = hw["TVT_input"].notna()
            last = hw[km].iloc[-1]; last_MD = float(last["MD"]); last_Z = float(last["Z"])
            gr = _t6_interp_nan(hw["GR"].values.astype(float))
            X_ = hw["X"].values.astype(float); Y = hw["Y"].values.astype(float)
            Z = hw["Z"].values.astype(float); MD = hw["MD"].values.astype(float)
            mdd = _t6np.gradient(MD); mdd[mdd == 0] = 1
            kg = gr[:int(km.sum())]; ktvt = hw.loc[km, "TVT_input"].values
            twk = _t6np.interp(ktvt, tw_tvt, tw_gr); v = _t6np.isfinite(kg) & _t6np.isfinite(twk)
            a, b = (_t6np.polyfit(kg[v], twk[v], 1) if v.sum() >= 20 else (1., 0.))
            cal = gr * a + b
            head = _t6np.arctan2(_t6np.gradient(Y), _t6np.gradient(X_) + 1e-9)
            ev_idx = _t6np.where((~km).values)[0]; n_ev = len(ev_idx)
            if len(pf_tvt) != n_ev: return None
            sub_idx = _t6np.arange(0, n_ev, _T6_SUB_EVERY); n_sub = len(sub_idx)
            off_by_ws = {ws: _t6np.zeros(n_sub) for ws in _T6_WIN_SIZES}
            peak_by_ws = {ws: _t6np.zeros(n_sub) for ws in _T6_WIN_SIZES}
            sharp_by_ws = {ws: _t6np.zeros(n_sub) for ws in _T6_WIN_SIZES}
            cal_ev = cal[ev_idx]
            for si, j in enumerate(sub_idx):
                for ws in _T6_WIN_SIZES:
                    half = ws // 2
                    lo = max(0, j - half); hi = min(n_ev, j + half + 1)
                    gwin = cal_ev[lo:hi]
                    off, peak, sharp = _t6_match_profile(gwin, tw_gr, tw_tvt, pf_tvt[j], _T6_SEARCH_HALF, _T6_SEARCH_STEP)
                    off_by_ws[ws][si] = off; peak_by_ws[ws][si] = peak; sharp_by_ws[ws][si] = sharp
            F = {}
            F["md_since"] = (MD - last_MD)[ev_idx]
            F["gr"] = gr[ev_idx]; F["gr_grad"] = _t6np.gradient(gr)[ev_idx]
            F["gr_rstd"] = _t6pd.Series(gr).rolling(21, center=True, min_periods=1).std().fillna(0).values[ev_idx]
            F["z"] = (Z - last_Z)[ev_idx]; F["dzdmd"] = (_t6np.gradient(Z) / mdd)[ev_idx]; F["cal_gr"] = cal[ev_idx]
            F["pf_ancc_tvt"] = pf_tvt
            F["sin_azi"] = _t6np.sin(head)[ev_idx]; F["cos_azi"] = _t6np.cos(head)[ev_idx]
            for ws in _T6_WIN_SIZES:
                F["off%d" % ws] = _t6np.interp(_t6np.arange(n_ev), sub_idx, off_by_ws[ws])
                F["peak%d" % ws] = _t6np.interp(_t6np.arange(n_ev), sub_idx, peak_by_ws[ws])
                F["sharp%d" % ws] = _t6np.interp(_t6np.arange(n_ev), sub_idx, sharp_by_ws[ws])
            M = _t6np.stack([F[c] for c in _T6_FEATCOLS], axis=1).astype(_t6np.float32)
            return M, ev_idx

        def _t6_find(pattern):
            hits = _t6glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
            if not hits:
                _local_dir = os.environ.get("ROGII_TRACK6_CKPT_DIR", "track6_checkpoint")
                if os.path.isdir(_local_dir):
                    hits = _t6glob.glob(os.path.join(_local_dir, pattern))
            return hits

        _t6_model_paths = sorted(_t6_find("track6_gbm_fold*.txt"))
        _t6_meta_paths = _t6_find("track6_gbm_meta.pkl")
        if not _t6_model_paths or not _t6_meta_paths:
            raise FileNotFoundError(
                "track6 GBM checkpoints/meta not found -- attach the 'rogii-track6-gbm' Kaggle Dataset as an input.")
        _t6_meta = _t6pickle.load(open(_t6_meta_paths[0], "rb"))
        _t6_pf_col = _t6_meta["pf_col"]
        _t6_models = [_t6lgb.Booster(model_file=_p) for _p in _t6_model_paths]

        _t6_pred_by_id = {}
        for _wid in list_wells("test"):
            hw = _t6pd.read_csv(CFG.DATA / "test" / f"{_wid}__horizontal_well.csv")
            tw = _t6pd.read_csv(CFG.DATA / "test" / f"{_wid}__typewell.csv").sort_values("TVT")
            tw_tvt = tw["TVT"].values.astype(float); tw_gr = tw["GR"].fillna(tw["GR"].mean()).values.astype(float)
            if len(tw_tvt) < 10: continue
            km = hw["TVT_input"].notna()
            if km.sum() < 20 or hw["TVT_input"].isna().sum() < 20: continue
            pf_tvt = _t6_run_pf_ancc(hw, tw_tvt, tw_gr)
            if len(pf_tvt) == 0: continue
            _r = _t6_build_well(hw, tw_tvt, tw_gr, pf_tvt)
            if _r is None: continue
            M, ev_idx = _r
            resid_preds = _t6np.mean([m.predict(M) for m in _t6_models], axis=0)
            t6_abs = M[:, _t6_pf_col] + resid_preds
            raw_ids = [f"{_wid}_{_r2}" for _r2 in ev_idx]
            for _rid, _pv in zip(raw_ids, t6_abs):
                _t6_pred_by_id[_rid] = float(_pv)

        if _t6_pred_by_id:
            _t6b = _t6pd.read_csv(OUT / "submission.csv")
            _t6_tvt = _t6b["tvt"].to_numpy(float).copy()
            _t6_ids = _t6b["id"].astype(str).to_numpy()
            _n_touched3 = 0
            for _i, _id in enumerate(_t6_ids):
                if _id in _t6_pred_by_id:
                    _t6_tvt[_i] = (1.0 - _T6_BLEND_W) * _t6_tvt[_i] + _T6_BLEND_W * _t6_pred_by_id[_id]
                    _n_touched3 += 1
            _t6b["tvt"] = _t6_tvt
            _order_ids3 = _t6pd.read_csv(CFG.DATA / "sample_submission.csv")["id"].astype(str)
            _t6b = _t6b.set_index("id").reindex(_order_ids3).reset_index()
            assert _t6b["tvt"].notna().all(), "track6 blend lost ids vs sample"
            _t6b[["id", "tvt"]].to_csv(OUT / "submission.csv", index=False)
            POSTPROCESSORS.append(f"track6_blend_{_T6_BLEND_W:.2f}")
            print(f"track6 blend W={_T6_BLEND_W}: touched {_n_touched3} eval rows, {len(_t6_models)} fold models")
        else:
            print("track6 blend: no usable test wells found (skipped, submission unchanged)")
    except Exception as _t6e:
        print(f"track6 blend FAILED, skipping safely (submission unchanged): {_t6e}")
else:
    print("track6 blend disabled (W=0)")
'''

if __name__ == '__main__':
    print('CELL_TRACK6_BLEND length:', len(CELL_TRACK6_BLEND))
