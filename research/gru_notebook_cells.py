"""Generates the two notebook cell source strings for the GRU-refiner submission integration:
CELL_SNAPSHOT (insert BEFORE the WARP-blend cell) and CELL_GRU_BLEND (insert AFTER physics-pp,
before the final audit cell). Both follow the existing notebook's defensive self-contained pattern
(env-var toggle, try/except-wrapped, never crashes the submission if anything goes wrong).
"""

CELL_SNAPSHOT = '''# --- SNAPSHOT pre-WARP submission (needed by the GRU-refiner blend stage later) -----------------
# The GRU-refiner (see the cell after physics post-process) was validated as: gru_pred = sp45 + a
# learned residual, blended against the FINAL post-physics-pp signal. It needs the pre-WARP-blend
# "sp45" value as its anchor/disagreement-feature reference, matching exactly how it was validated
# offline (proxy sp45 baseline = 10.62, identical to this WARP-blend cell's own docstring number).
try:
    import shutil as _snap_shutil
    _snap_shutil.copy(OUT / "submission.csv", OUT / "pre_warp_snapshot.csv")
except Exception as _snap_e:
    print(f"pre-WARP snapshot FAILED (GRU-refiner stage will skip safely later): {_snap_e}")
'''

CELL_GRU_BLEND = '''# --- GRU-REFINER BLEND ---------------------------------------------------------------------------
# Directly informed by a top-5 competitor's writeup (2026-07-14): their winning architecture is a
# bidirectional-GRU that REFINES an existing PF tracker's output (taking the tracker's own position
# and its DISAGREEMENT with another tracker as input features), not a from-scratch GR->TVT regressor
# like every one of our own from-scratch NN attempts (WARP/H4/MDN/etc, all capped ~11.3 on holdout).
# This is the first NN in this project framed as a refiner rather than a predictor. Architecture:
# 16 raw GR/trajectory features + (warp-sp45) disagreement + std(sp45,warp) -> bidirectional GRU
# (2-layer) -> small-init linear head -> residual correction added to sp45. Trained as a 6-seed
# ensemble (matching the competitor's own "six-model GRU ensemble" diversity mechanism) with the
# SAME n_val=160,seed=42 holdout as the WARP checkpoint. Validated with a proper paired well-bootstrap
# (not a handful of fixed folds -- this project's own methodology lesson from today): mean_gain
# +0.03 to +0.08 ft blended against the full combo, frac_positive 88-95% across w in [0.1,0.3], CI
# still touching zero at n=160 (same order as the competitor's own confirmed-but-tiny 0.06ft gain,
# which needed his full 773-well bootstrap to clear significance) -- promising but NOT yet a fully
# confirmed win, hence this real-LB test. Attach the Kaggle Dataset with the 6 seed checkpoints +
# gru_v3_norm_stats.pkl as an input; this cell finds them under /kaggle/input/ automatically.
_GRU_BLEND_W = float(os.environ.get("ROGII_GRU_BLEND", "0.15"))
if _GRU_BLEND_W > 0:
    try:
        import glob as _gglob, torch as _gtorch, torch.nn as _gnn, torch.nn.functional as _gF
        import numpy as _gnp, pandas as _gpd, pickle as _gpickle

        _GDEV = "cuda" if _gtorch.cuda.is_available() else "cpu"
        _G_SEQLEN = 500
        _G_D, _G_LAYERS = 64, 2

        class _GRURefiner(_gnn.Module):
            def __init__(s, n_in, d, layers):
                super().__init__()
                s.inp = _gnn.Linear(n_in, d)
                s.gru = _gnn.GRU(d, d, num_layers=layers, batch_first=True, bidirectional=True)
                s.head = _gnn.Sequential(_gnn.Linear(2 * d, d), _gnn.GELU(), _gnn.Dropout(0.0), _gnn.Linear(d, 1))
            def forward(s, X):
                h = _gF.gelu(s.inp(X))
                h, _ = s.gru(h)
                return s.head(h)[..., 0]

        _gru_ckpts = sorted(_gglob.glob("/kaggle/input/**/gru_refiner_v3_seed*.pt", recursive=True))
        _norm_path = None
        for _p in _gglob.glob("/kaggle/input/**/gru_v3_norm_stats.pkl", recursive=True):
            _norm_path = _p; break
        if not _gru_ckpts or _norm_path is None:
            _local_dir = os.environ.get("ROGII_GRU_CKPT_DIR", "gru_refiner_checkpoint")
            if os.path.isdir(_local_dir):
                _gru_ckpts = sorted(_gglob.glob(os.path.join(_local_dir, "gru_refiner_v3_seed*.pt")))
                _local_norm = os.path.join(_local_dir, "gru_v3_norm_stats.pkl")
                if os.path.exists(_local_norm): _norm_path = _local_norm
        if not _gru_ckpts or _norm_path is None:
            raise FileNotFoundError(
                "GRU-refiner checkpoints/norm-stats not found -- attach the "
                "'rogii-gru-refiner' Kaggle Dataset as an input.")
        _gnorm = _gpickle.load(open(_norm_path, "rb"))
        _gmean, _gstd = _gnorm["mean"], _gnorm["std"]

        _gnets = []
        for _ckpt in _gru_ckpts:
            _net = _GRURefiner(18, _G_D, _G_LAYERS).to(_GDEV)
            _net.load_state_dict(_gtorch.load(_ckpt, map_location=_GDEV))
            _net.eval()
            _gnets.append(_net)

        def _g_inan(a):
            a = a.copy(); m = _gnp.isnan(a); i = _gnp.arange(len(a))
            if m.all(): return _gnp.zeros(len(a))
            a[m] = _gnp.interp(i[m], i[~m], a[~m]); return a

        # -- own (16) GR/trajectory features, matching research/full_pipeline_proxy.py's well_feats --
        _G_OFFS = (-20, -10, -5, 0, 5, 10, 20)
        def _g_own_feats(hw, tw_tvt, tw_gr, last_tvt, last_Z):
            gr = _g_inan(hw["GR"].values.astype(float))
            X = hw["X"].values.astype(float); Y = hw["Y"].values.astype(float)
            Z = hw["Z"].values.astype(float); MD = hw["MD"].values.astype(float)
            mdd = _gnp.gradient(MD); mdd[mdd == 0] = 1
            km = hw["TVT_input"].notna()
            kg = gr[:int(km.sum())]; ktvt = hw.loc[km, "TVT_input"].values
            twk = _gnp.interp(ktvt, tw_tvt, tw_gr); v = _gnp.isfinite(kg) & _gnp.isfinite(twk)
            a, b = (_gnp.polyfit(kg[v], twk[v], 1) if v.sum() >= 20 else (1., 0.))
            head = _gnp.arctan2(_gnp.gradient(Y), _gnp.gradient(X) + 1e-9)
            F = {}
            F["md_since"] = MD - float(hw.loc[km, "MD"].iloc[-1]); F["gr"] = gr; F["gr_grad"] = _gnp.gradient(gr)
            F["gr_rstd"] = _gpd.Series(gr).rolling(21, center=True, min_periods=1).std().fillna(0).values
            F["z"] = Z - last_Z; F["dzdmd"] = _gnp.gradient(Z) / mdd; F["cal_gr"] = gr * a + b
            for o in _G_OFFS: F["tda%d" % o] = gr - _gnp.interp(last_tvt + o, tw_tvt, tw_gr)
            F["sin_azi"] = _gnp.sin(head); F["cos_azi"] = _gnp.cos(head)
            OWN_ORDER = ["md_since", "gr", "gr_grad", "gr_rstd", "z", "dzdmd", "cal_gr",
                         "tda-20", "tda-10", "tda-5", "tda0", "tda5", "tda10", "tda20", "sin_azi", "cos_azi"]
            return _gnp.stack([F[c] for c in OWN_ORDER], axis=1)  # (n,16)

        # -- WARP inference, recomputed independently here for self-containment (small/fast) --
        _W_KS, _W_ES, _W_TG, _W_D2, _W_MAXSTEP = 60, 360, 256, 128, 1.5
        class _GWEnc(_gnn.Module):
            def __init__(s, nin, d):
                super().__init__(); s.inp = _gnn.Conv1d(nin, d, 5, padding=2)
                s.blocks = _gnn.ModuleList([_gnn.Sequential(
                    _gnn.Conv1d(d, d, 3, padding=dl, dilation=dl), _gnn.GroupNorm(8, d), _gnn.GELU(), _gnn.Dropout(0.0),
                    _gnn.Conv1d(d, d, 3, padding=dl, dilation=dl), _gnn.GroupNorm(8, d), _gnn.GELU())
                    for dl in (1, 2, 4, 8, 16)])
                s.out = _gnn.Conv1d(d, d, 1)
            def forward(s, x):
                h = _gF.gelu(s.inp(x))
                for blk in s.blocks: h = h + blk(h)
                return s.out(h)
        class _GWarpNet(_gnn.Module):
            def __init__(s, d):
                super().__init__(); s.he = _GWEnc(4, d); s.te = _GWEnc(1, d)
                s.q = _gnn.Conv1d(d, d, 1); s.k = _gnn.Conv1d(d, d, 1); s.vv = _gnn.Conv1d(d, d, 1); s.sc = d ** -0.5
                s.head = _gnn.Sequential(_gnn.Linear(2 * d, d), _gnn.GELU(), _gnn.Dropout(0.0), _gnn.Linear(d, 1))
            def forward(s, H, G, lt):
                h = s.he(H); t = s.te(G)
                Q = s.q(h).transpose(1, 2); K = s.k(t).transpose(1, 2); V = s.vv(t).transpose(1, 2)
                att = _gtorch.softmax(Q @ K.transpose(1, 2) * s.sc, dim=2); ctx = att @ V
                x = _gtorch.cat([h.transpose(1, 2), ctx], dim=2)
                dt = _gtorch.tanh(s.head(x)[..., 0]) * _W_MAXSTEP
                tvt = _gtorch.cumsum(dt, 1)
                return tvt - tvt[:, _W_KS - 1:_W_KS] + lt[:, None]

        _warp_ckpt_path = None
        for _p in _gglob.glob("/kaggle/input/**/best_warp.pt", recursive=True):
            _warp_ckpt_path = _p; break
        if _warp_ckpt_path is None:
            _local_w = os.environ.get("ROGII_WARP_CKPT", "best_warp.pt")
            if os.path.exists(_local_w): _warp_ckpt_path = _local_w
        if _warp_ckpt_path is None:
            raise FileNotFoundError("best_warp.pt not found for the GRU-refiner's WARP-disagreement feature.")
        _gwnet = _GWarpNet(_W_D2).to(_GDEV)
        _gwnet.load_state_dict(_gtorch.load(_warp_ckpt_path, map_location=_GDEV)); _gwnet.eval()

        _sp45_snap = _gpd.read_csv(OUT / "pre_warp_snapshot.csv")
        _sp45_by_id = dict(zip(_sp45_snap["id"].astype(str), _sp45_snap["tvt"].astype(float)))

        _gru_pred_by_id = {}
        for _wid in list_wells("test"):
            hw = _gpd.read_csv(CFG.DATA / "test" / f"{_wid}__horizontal_well.csv")
            tw = _gpd.read_csv(CFG.DATA / "test" / f"{_wid}__typewell.csv").sort_values("TVT")
            tt = tw["TVT"].values.astype(float); tg = tw["GR"].fillna(tw["GR"].mean()).values.astype(float)
            if len(tt) < 10: continue
            kn = hw[hw["TVT_input"].notna()]
            if len(kn) < 20 or hw["TVT_input"].isna().sum() < 20: continue
            last_tvt = float(kn["TVT_input"].iloc[-1]); last_Z = float(kn["Z"].iloc[-1])
            ev_mask = hw["TVT_input"].isna().values
            ei = _gnp.where(ev_mask)[0]
            if len(ei) < 5: continue
            n = len(hw)
            raw_ids = [f"{_wid}_{_r}" for _r in ei]
            sp45_ev = _gnp.array([_sp45_by_id.get(_rid, _gnp.nan) for _rid in raw_ids])
            if not _gnp.isfinite(sp45_ev).all(): continue

            own16 = _g_own_feats(hw, tt, tg, last_tvt, last_Z)[ei]  # (n_eval,16)

            # WARP prediction on this well (small, recomputed for self-containment)
            gr = _g_inan(hw["GR"].values.astype(float))
            kn_gr = kn["GR"].interpolate().bfill().ffill().values.astype(float)
            twk = _gnp.interp(kn["TVT_input"].values, tt, tg); v = _gnp.isfinite(kn_gr) & _gnp.isfinite(twk)
            a, b = (_gnp.polyfit(kn_gr[v], twk[v], 1) if v.sum() >= 20 else (1., 0.)); cal = gr * a + b
            Z = hw["Z"].values.astype(float); MD = hw["MD"].values.astype(float)
            mdd = _gnp.gradient(MD); mdd[mdd == 0] = 1
            grad = _gnp.gradient(gr); rstd = _gpd.Series(gr).rolling(21, center=True, min_periods=1).std().fillna(0).values
            dz = _gnp.gradient(Z) / mdd
            e0 = ei[0]; ks = _gnp.arange(max(0, e0 - 400), e0)
            if len(ks) < 5: continue
            g_tvt = _gnp.linspace(tt.min(), tt.max(), _W_TG); g_gr = _gnp.interp(g_tvt, tt, tg)
            gm, gs2 = float(g_gr.mean()), float(g_gr.std() + 1e-6)
            caln = (cal - gm) / gs2; gradn = grad / gs2; rstdn = rstd / gs2
            dzn = (dz - dz.mean()) / (dz.std() + 1e-6)
            kd = _gnp.linspace(ks[0], ks[-1], _W_KS); ed = _gnp.linspace(e0, n - 1, _W_ES)
            dst = _gnp.concatenate([kd, ed]); src = _gnp.arange(n)
            Rw = lambda x: _gnp.interp(dst, src, x).astype(_gnp.float32)
            Hw = _gnp.stack([Rw(caln), Rw(gradn), Rw(rstdn), Rw(dzn)]).astype(_gnp.float32)
            with _gtorch.no_grad():
                Ht = _gtorch.tensor(Hw[None], dtype=_gtorch.float32, device=_GDEV)
                Gt = _gtorch.tensor(((g_gr - gm) / gs2).astype(_gnp.float32)[None, None, :_W_TG], dtype=_gtorch.float32, device=_GDEV)
                Lt = _gtorch.tensor([last_tvt], dtype=_gtorch.float32, device=_GDEV)
                pred_grid = _gwnet(Ht, Gt, Lt)[0].cpu().numpy()
            md_ev_grid = Rw(MD)[_W_KS:]; pred_ev_grid = pred_grid[_W_KS:]
            order_w = _gnp.argsort(md_ev_grid)
            md_raw_ev = MD[ei]
            warp_ev = _gnp.interp(md_raw_ev, md_ev_grid[order_w], pred_ev_grid[order_w])

            disagree = (warp_ev - sp45_ev)[:, None]
            ens_std = _gnp.stack([sp45_ev, warp_ev], axis=1).std(axis=1, keepdims=True)
            Xrow = _gnp.concatenate([own16, disagree, ens_std], axis=1)  # (n_eval,18)
            Xn = (Xrow - _gmean) / _gstd

            n_eval = len(ei)
            src_idx = _gnp.arange(n_eval); dst_idx = _gnp.linspace(0, n_eval - 1, _G_SEQLEN)
            Xr = _gnp.stack([_gnp.interp(dst_idx, src_idx, Xn[:, c]) for c in range(Xn.shape[1])], axis=1).astype(_gnp.float32)

            resid_preds = []
            with _gtorch.no_grad():
                Xt = _gtorch.tensor(Xr[None], dtype=_gtorch.float32, device=_GDEV)
                for _net in _gnets:
                    r = _net(Xt)[0].cpu().numpy()
                    resid_preds.append(r)
            resid_mean = _gnp.mean(resid_preds, axis=0)
            resid_full = _gnp.interp(src_idx, dst_idx, resid_mean)
            gru_ev = sp45_ev + resid_full
            for _rid, _pv in zip(raw_ids, gru_ev):
                _gru_pred_by_id[_rid] = float(_pv)

        if _gru_pred_by_id:
            _gb = _gpd.read_csv(OUT / "submission.csv")
            _gb_tvt = _gb["tvt"].to_numpy(float).copy()
            _gb_ids = _gb["id"].astype(str).to_numpy()
            _n_touched2 = 0
            for _i, _id in enumerate(_gb_ids):
                if _id in _gru_pred_by_id:
                    _gb_tvt[_i] = (1.0 - _GRU_BLEND_W) * _gb_tvt[_i] + _GRU_BLEND_W * _gru_pred_by_id[_id]
                    _n_touched2 += 1
            _gb["tvt"] = _gb_tvt
            _order_ids2 = _gpd.read_csv(CFG.DATA / "sample_submission.csv")["id"].astype(str)
            _gb = _gb.set_index("id").reindex(_order_ids2).reset_index()
            assert _gb["tvt"].notna().all(), "gru-refiner blend lost ids vs sample"
            _gb[["id", "tvt"]].to_csv(OUT / "submission.csv", index=False)
            POSTPROCESSORS.append(f"gru_refiner_blend_{_GRU_BLEND_W:.2f}")
            print(f"GRU-refiner blend W={_GRU_BLEND_W}: touched {_n_touched2} eval rows (device={_GDEV})")
        else:
            print("GRU-refiner blend: no usable test wells found (skipped, submission unchanged)")
    except Exception as _ge:
        print(f"GRU-refiner blend FAILED, skipping safely (submission unchanged): {_ge}")
else:
    print("GRU-refiner blend disabled (W=0)")
'''

if __name__ == '__main__':
    print('CELL_SNAPSHOT length:', len(CELL_SNAPSHOT))
    print('CELL_GRU_BLEND length:', len(CELL_GRU_BLEND))
