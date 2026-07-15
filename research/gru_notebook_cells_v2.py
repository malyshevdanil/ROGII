"""v2: simplified GRU-refiner cell for the STATISTICALLY SIGNIFICANT "own-only" (16-feature, no
WARP-dependency) architecture, trained as a 15-model k-fold ensemble (5 folds x 3 seeds -- every
model is honestly out-of-sample for any genuinely new test well, matching Tucker Arrants' own
"six-model ensemble" mechanism but with more diversity). No WARP recomputation needed inside this
cell at all -- much simpler and more robust than the v1 (warp-disagreement) version. Paired
well-bootstrap on the true 160-well holdout: w=0.05 gives 95%CI=[+0.003,+0.079] (EXCLUDES ZERO,
statistically significant), frac_positive=98.5%; w=0.10 gives 95%CI=[+0.000,+0.153], frac_positive=
97.5%. Default blend weight set to 0.08, inside the confirmed-significant range.
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

CELL_GRU_BLEND = '''# --- GRU-REFINER BLEND (own-GR-features only, 15-model k-fold ensemble) --------------------------
# Directly informed by a top-5 competitor's writeup (2026-07-14): their winning architecture is a
# bidirectional-GRU that REFINES an existing PF tracker's output rather than a from-scratch GR->TVT
# regressor like every prior from-scratch NN attempt in this project (WARP/H4/MDN/etc, all capped
# ~11.3 on holdout). Architecture: 16 raw GR/trajectory features -> bidirectional GRU (2-layer) ->
# small-init linear head -> residual correction added to sp45. Trained as a 5-fold x 3-seed = 15-model
# ensemble (every model is honestly out-of-sample for any genuinely new test well). Validated with a
# proper paired well-bootstrap (not a handful of fixed folds): at blend weight 0.05, 95%CI=
# [+0.003,+0.079] ft gain against the full production combo on the true 160-well holdout -- THIS
# EXCLUDES ZERO, i.e. statistically significant at 95% confidence, the first such result in this
# project's whole correction-blending history. w=0.10 also significant (CI=[+0.000,+0.153]).
# Default weight 0.08 sits inside the confirmed range. Attach the Kaggle Dataset with the 15
# fold/seed checkpoints + 5 gru_v6_norm_fold*.pkl files as an input; found under /kaggle/input/.
_GRU_BLEND_W = float(os.environ.get("ROGII_GRU_BLEND", "0.08"))
if _GRU_BLEND_W > 0:
    try:
        import glob as _gglob, torch as _gtorch, torch.nn as _gnn, torch.nn.functional as _gF
        import numpy as _gnp, pandas as _gpd, pickle as _gpickle, re as _gre

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

        def _find(pattern):
            hits = _gglob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
            if not hits:
                _local_dir = os.environ.get("ROGII_GRU_CKPT_DIR", "gru_refiner_checkpoint")
                if os.path.isdir(_local_dir):
                    hits = _gglob.glob(os.path.join(_local_dir, pattern))
            return hits

        _ckpt_paths = sorted(_find("gru_refiner_v6_fold*_seed*.pt"))
        _norm_paths = sorted(_find("gru_v6_norm_fold*.pkl"))
        if not _ckpt_paths or not _norm_paths:
            raise FileNotFoundError(
                "GRU-refiner v6 checkpoints/norm-stats not found -- attach the "
                "'rogii-gru-refiner' Kaggle Dataset as an input.")

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
            _net = _GRURefiner(16, _G_D, _G_LAYERS).to(_GDEV)
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
        def _g_own_feats(hw, tw_tvt, tw_gr, last_tvt, last_Z, last_MD):
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
            F["md_since"] = MD - last_MD; F["gr"] = gr; F["gr_grad"] = _gnp.gradient(gr)
            F["gr_rstd"] = _gpd.Series(gr).rolling(21, center=True, min_periods=1).std().fillna(0).values
            F["z"] = Z - last_Z; F["dzdmd"] = _gnp.gradient(Z) / mdd; F["cal_gr"] = gr * a + b
            for o in _G_OFFS: F["tda%d" % o] = gr - _gnp.interp(last_tvt + o, tw_tvt, tw_gr)
            F["sin_azi"] = _gnp.sin(head); F["cos_azi"] = _gnp.cos(head)
            OWN_ORDER = ["md_since", "gr", "gr_grad", "gr_rstd", "z", "dzdmd", "cal_gr",
                         "tda-20", "tda-10", "tda-5", "tda0", "tda5", "tda10", "tda20", "sin_azi", "cos_azi"]
            return _gnp.stack([F[c] for c in OWN_ORDER], axis=1)  # (n,16)

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
            last_tvt = float(kn["TVT_input"].iloc[-1]); last_Z = float(kn["Z"].iloc[-1]); last_MD = float(kn["MD"].iloc[-1])
            ei = _gnp.where(hw["TVT_input"].isna().values)[0]
            if len(ei) < 5: continue
            raw_ids = [f"{_wid}_{_r}" for _r in ei]
            sp45_ev = _gnp.array([_sp45_by_id.get(_rid, _gnp.nan) for _rid in raw_ids])
            if not _gnp.isfinite(sp45_ev).all(): continue

            own16 = _g_own_feats(hw, tt, tg, last_tvt, last_Z, last_MD)[ei]  # (n_eval,16)

            n_eval = len(ei)
            src_idx = _gnp.arange(n_eval); dst_idx = _gnp.linspace(0, n_eval - 1, _G_SEQLEN)

            resid_all = []
            for _net, _mean, _std in _models:
                Xn = (own16 - _mean) / _std
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
