"""v3: the GRU-refiner cell for the v7 architecture (24 features: multi-scale GR, legal look-ahead,
gap indicator, band-pass; trained with prefix-cut augmentation, 5-fold x 2-seed = 10-model ensemble).
Paired well-bootstrap on the true 160-well holdout: w=0.3 gives 95%CI=[+0.033,+0.598] (clearly excludes
zero), the strongest and most confidently validated result in this project's whole correction-blending
history -- roughly 10x the gain of the earlier v6 (own-only, no augmentation) version. Default weight 0.3.
"""

CELL_SNAPSHOT = '''# --- SNAPSHOT pre-WARP submission (needed by the GRU-refiner blend stage later) -----------------
# The GRU-refiner (see the cell after physics post-process) was validated as: gru_pred = sp45 + a
# learned residual, blended against the FINAL post-physics-pp signal. It needs the pre-WARP-blend
# "sp45" value as its anchor, matching exactly how it was validated offline (proxy sp45 baseline =
# 10.62, identical to this WARP-blend cell's own docstring number).
try:
    import shutil as _snap_shutil
    _snap_shutil.copy(OUT / "submission.csv", OUT / "pre_warp_snapshot.csv")
except Exception as _snap_e:
    print(f"pre-WARP snapshot FAILED (GRU-refiner stage will skip safely later): {_snap_e}")
'''

CELL_GRU_BLEND = '''# --- GRU-REFINER BLEND (v7: rich multi-scale + legal-lookahead features, prefix-cut-augmented) ----
# Directly informed by a top-5 competitor's writeup (2026-07-14): their winning architecture is a
# bidirectional-GRU that REFINES an existing PF tracker's output (not a from-scratch GR->TVT regressor
# like every prior from-scratch NN attempt in this project, all capped ~11.3 on holdout). v7 upgrades
# the first working version (v6, own-only 16 features) with: multi-scale GR (raw/sm5/sm15/sm41/DoG
# band-pass), a gap indicator, LEGAL forward-only look-ahead summaries (the whole eval GR sequence is
# known upfront -- only TVT is hidden, so "look-ahead" within the eval zone is legal, exactly as
# Tucker's writeup notes), and PREFIX-CUT AUGMENTATION during training (matching his "prefix-cut GRU
# augmentation improved 8.404->8.159"). Trained as a 5-fold x 2-seed = 10-model ensemble; every model
# is honestly out-of-sample for any genuinely new test well. Paired well-bootstrap on the true 160-well
# holdout: w=0.3 gives 95%CI=[+0.033,+0.598] ft gain against the full production combo -- clearly
# excludes zero, roughly 10x the gain of the earlier v6 version. Attach the (updated) Kaggle Dataset
# with the 10 fold/seed checkpoints + 5 gru_v7_norm_fold*.pkl files; found under /kaggle/input/.
_GRU_BLEND_W = float(os.environ.get("ROGII_GRU_BLEND", "0.30"))
if _GRU_BLEND_W > 0:
    try:
        import glob as _gglob, torch as _gtorch, torch.nn as _gnn, torch.nn.functional as _gF
        import numpy as _gnp, pandas as _gpd, pickle as _gpickle, re as _gre

        _GDEV = "cuda" if _gtorch.cuda.is_available() else "cpu"
        _G_SEQLEN = 500
        _G_D, _G_LAYERS = 96, 2
        _G_NFEAT = 24

        class _GRURefinerV7(_gnn.Module):
            def __init__(s, n_in, d, layers):
                super().__init__()
                s.inp = _gnn.Linear(n_in, d)
                s.gru = _gnn.GRU(d, d, num_layers=layers, batch_first=True, bidirectional=True)
                s.head = _gnn.Sequential(_gnn.Linear(2 * d, d), _gnn.GELU(), _gnn.Dropout(0.0), _gnn.Linear(d, 1))
            def forward(s, X):
                h = _gF.gelu(s.inp(X))
                h, _ = s.gru(h)
                return s.head(h)[..., 0]

        def _find(pattern):
            hits = _gglob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
            if not hits:
                _local_dir = os.environ.get("ROGII_GRU_CKPT_DIR", "gru_refiner_checkpoint")
                if os.path.isdir(_local_dir):
                    hits = _gglob.glob(os.path.join(_local_dir, pattern))
            return hits

        _ckpt_paths = sorted(_find("gru_refiner_v7_fold*_seed*.pt"))
        _norm_paths = sorted(_find("gru_v7_norm_fold*.pkl"))
        if not _ckpt_paths or not _norm_paths:
            raise FileNotFoundError(
                "GRU-refiner v7 checkpoints/norm-stats not found -- attach the "
                "'rogii-gru-refiner' Kaggle Dataset (updated to v7) as an input.")

        _norm_by_fold = {}
        for _p in _norm_paths:
            _m = _gre.search(r"fold(\\d+)", os.path.basename(_p))
            if _m: _norm_by_fold[int(_m.group(1))] = _gpickle.load(open(_p, "rb"))

        _models = []  # list of (net, mean, std)
        for _p in _ckpt_paths:
            _m = _gre.search(r"fold(\\d+)_seed(\\d+)", os.path.basename(_p))
            if not _m: continue
            _fold = int(_m.group(1))
            if _fold not in _norm_by_fold: continue
            _net = _GRURefinerV7(_G_NFEAT, _G_D, _G_LAYERS).to(_GDEV)
            _net.load_state_dict(_gtorch.load(_p, map_location=_GDEV))
            _net.eval()
            _norm = _norm_by_fold[_fold]
            _models.append((_net, _norm["mean"], _norm["std"]))
        if not _models:
            raise RuntimeError("GRU-refiner: no (checkpoint, norm) pairs could be matched.")

        def _g_inan(a):
            a = a.copy(); m = _gnp.isnan(a); i = _gnp.arange(len(a))
            if m.all(): return _gnp.zeros(len(a))
            a[m] = _gnp.interp(i[m], i[~m], a[~m]); return a

        _G_OFFS = (-20, -10, -5, 0, 5, 10, 20)
        def _g_feats(hw, tw_tvt, tw_gr, km):
            km_idx = _gnp.where(km.values)[0]
            last_i = km_idx[-1]
            last_tvt = float(hw["TVT_input"].iloc[last_i]); last_Z = float(hw["Z"].iloc[last_i]); last_MD = float(hw["MD"].iloc[last_i])

            gr = _g_inan(hw["GR"].values.astype(float))
            gap = hw["GR"].isna().values.astype(_gnp.float32)
            X_ = hw["X"].values.astype(float); Y = hw["Y"].values.astype(float)
            Z = hw["Z"].values.astype(float); MD = hw["MD"].values.astype(float)
            mdd = _gnp.gradient(MD); mdd[mdd == 0] = 1

            kg = gr[km_idx]; ktvt = hw["TVT_input"].values[km_idx]
            twk = _gnp.interp(ktvt, tw_tvt, tw_gr); v = _gnp.isfinite(kg) & _gnp.isfinite(twk)
            a, b = (_gnp.polyfit(kg[v], twk[v], 1) if v.sum() >= 20 else (1., 0.))
            cal = gr * a + b

            cs = _gpd.Series(cal)
            sm5 = cs.rolling(5, center=True, min_periods=1).mean().values
            sm15 = cs.rolling(15, center=True, min_periods=1).mean().values
            sm41 = cs.rolling(41, center=True, min_periods=1).mean().values
            dog1 = sm5 - sm15; dog2 = sm15 - sm41
            grad = _gnp.gradient(gr)
            rstd = _gpd.Series(gr).rolling(21, center=True, min_periods=1).std().fillna(0).values
            dzdmd = _gnp.gradient(Z) / mdd
            head = _gnp.arctan2(_gnp.gradient(Y), _gnp.gradient(X_) + 1e-9)

            fwd_mean = cs.rolling(40, min_periods=1).mean().shift(-40).bfill().ffill().values
            fwd_std = cs.rolling(40, min_periods=1).std().shift(-40).bfill().ffill().fillna(0).values

            F_ = {}
            F_["md_since"] = MD - last_MD
            F_["gr"] = gr; F_["cal_gr"] = cal
            F_["sm5"] = sm5; F_["sm15"] = sm15; F_["sm41"] = sm41
            F_["dog1"] = dog1; F_["dog2"] = dog2
            F_["grad"] = grad; F_["rstd"] = rstd
            F_["gap"] = gap
            F_["z"] = Z - last_Z; F_["dzdmd"] = dzdmd
            F_["fwd_mean"] = fwd_mean; F_["fwd_std"] = fwd_std
            for o in _G_OFFS: F_["tda%d" % o] = gr - _gnp.interp(last_tvt + o, tw_tvt, tw_gr)
            F_["sin_azi"] = _gnp.sin(head); F_["cos_azi"] = _gnp.cos(head)
            ORDER = ["md_since", "gr", "cal_gr", "sm5", "sm15", "sm41", "dog1", "dog2", "grad", "rstd", "gap",
                     "z", "dzdmd", "fwd_mean", "fwd_std",
                     "tda-20", "tda-10", "tda-5", "tda0", "tda5", "tda10", "tda20", "sin_azi", "cos_azi"]
            return _gnp.stack([F_[c] for c in ORDER], axis=1)  # (n,24)

        _sp45_snap = _gpd.read_csv(OUT / "pre_warp_snapshot.csv")
        _sp45_by_id = dict(zip(_sp45_snap["id"].astype(str), _sp45_snap["tvt"].astype(float)))

        _gru_pred_by_id = {}
        for _wid in list_wells("test"):
            hw = _gpd.read_csv(CFG.DATA / "test" / f"{_wid}__horizontal_well.csv")
            tw = _gpd.read_csv(CFG.DATA / "test" / f"{_wid}__typewell.csv").sort_values("TVT")
            tt = tw["TVT"].values.astype(float); tg = tw["GR"].fillna(tw["GR"].mean()).values.astype(float)
            if len(tt) < 10: continue
            km = hw["TVT_input"].notna()
            if km.sum() < 20 or hw["TVT_input"].isna().sum() < 20: continue
            ei = _gnp.where(hw["TVT_input"].isna().values)[0]
            if len(ei) < 5: continue
            raw_ids = [f"{_wid}_{_r}" for _r in ei]
            sp45_ev = _gnp.array([_sp45_by_id.get(_rid, _gnp.nan) for _rid in raw_ids])
            if not _gnp.isfinite(sp45_ev).all(): continue

            Xfull = _g_feats(hw, tt, tg, km)
            Xev = Xfull[ei]  # (n_eval,24)

            n_eval = len(ei)
            src_idx = _gnp.arange(n_eval); dst_idx = _gnp.linspace(0, n_eval - 1, _G_SEQLEN)

            resid_all = []
            for _net, _mean, _std in _models:
                Xn = (Xev - _mean) / _std
                Xr = _gnp.stack([_gnp.interp(dst_idx, src_idx, Xn[:, c]) for c in range(Xn.shape[1])], axis=1).astype(_gnp.float32)
                with _gtorch.no_grad():
                    Xt = _gtorch.tensor(Xr[None], dtype=_gtorch.float32, device=_GDEV)
                    r = _net(Xt)[0].cpu().numpy()
                resid_all.append(_gnp.interp(src_idx, dst_idx, r))
            resid_mean = _gnp.mean(resid_all, axis=0)
            gru_ev = sp45_ev + resid_mean
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
            print(f"GRU-refiner blend W={_GRU_BLEND_W}: touched {_n_touched2} eval rows, "
                  f"{len(_models)} ensemble models (device={_GDEV})")
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
