"""Submission cell for neighbor-transfer: a purely spatial signal (zero GR involved) that borrows the
nearest OTHER train well's own structural-elevation curve (pos = TVT+Z), IDW-combined across k=3
neighbors, with distance-adaptive confidence (fades toward the current best track when the match is far
-- matches the observed pattern that transfer quality degrades sharply beyond ~2000ft). Validated on the
true 160-well holdout: corr(err, current_best_err)=0.387 (confidence-weighted version), paired bootstrap
at w=0.05: +0.146ft, 95%CI=[-0.037,+0.296], 94.7% positive, tail-risk max single-well worsening +3.17ft
at w=0.05 with a healthy 2.5x improvement/worsening ratio.

CRITICAL: this competition has train/test well-ID overlap (the guarded_contact_override finding) -- our
real test wells are literal copies of specific named train wells. The neighbor search MUST explicitly
exclude the target well's OWN ID from the candidate donor pool (by name, not just "skip nearest"),
otherwise it would find itself at ~zero distance and leak the true answer directly. This mirrors the
exact self_wid bug already found and fixed in track6.
"""

CELL_NEIGHBOR_BLEND = '''# --- NEIGHBOR-TRANSFER BLEND (pure spatial signal, zero GR involved) -----------------------------
# Borrows the nearest OTHER train well's own structural-elevation curve (pos = TVT+Z) via IDW across
# k=3 spatial neighbors, with distance-adaptive confidence (fades toward the current best track when
# the nearest match is far -- transfer quality degrades sharply beyond ~2000ft on the true holdout).
# corr(err, current_best_err)=0.387 on the true 160-well holdout, paired bootstrap at w=0.05:
# +0.146ft, 95%CI=[-0.037,+0.296], 94.7%% positive, tail-risk max worsening +3.17ft (2.5x
# improvement/worsening ratio). CRITICAL: this competition has train/test well-ID overlap
# (guarded_contact_override) -- the neighbor search explicitly excludes the target well's OWN ID from
# the donor pool by name, not just by array index, to avoid leaking the true answer via a
# ~zero-distance self-match.
_NEIGH_BLEND_W = float(os.environ.get("ROGII_NEIGHBOR_BLEND", "0.05"))
if _NEIGH_BLEND_W > 0:
    try:
        import glob as _nbglob, numpy as _nbnp, pandas as _nbpd
        from scipy.spatial import cKDTree as _NbTree

        _NB_K = 3
        _NB_DIST_SCALE = 1500.0

        _nb_train_wids = sorted({_p.stem.split("__")[0] for _p in (CFG.DATA / "train").glob("*__horizontal_well.csv")})
        _nb_cents = []
        for _wid2 in _nb_train_wids:
            try:
                _df = _nbpd.read_csv(CFG.DATA / "train" / f"{_wid2}__horizontal_well.csv", usecols=["X", "Y"])
                _nb_cents.append((_wid2, float(_df.X.median()), float(_df.Y.median())))
            except Exception:
                continue
        _nb_cent_xy = _nbnp.array([[c[1], c[2]] for c in _nb_cents])
        _nb_cent_ids = [c[0] for c in _nb_cents]
        _nb_cent_tree = _NbTree(_nb_cent_xy)

        _nb_curve_cache = {}
        _nb_tree_cache = {}

        def _nb_get_curve(wid):
            if wid not in _nb_curve_cache:
                _hw = _nbpd.read_csv(CFG.DATA / "train" / f"{wid}__horizontal_well.csv")
                if "TVT" not in _hw.columns:
                    _nb_curve_cache[wid] = None
                else:
                    _km = _hw["TVT_input"].notna()
                    _xy = _hw[["X", "Y"]].to_numpy(_nbnp.float64)
                    _z = _hw["Z"].to_numpy(_nbnp.float64)
                    _tvt_true = _hw["TVT"].to_numpy(_nbnp.float64)
                    _tvt_known = _hw["TVT_input"].to_numpy(_nbnp.float64)
                    _pos_known = _nbnp.where(_km.values, _tvt_known, _nbnp.nan) + _z
                    _nb_curve_cache[wid] = dict(xy=_xy, z=_z, pos_known=_pos_known, km=_km.values, tvt_true=_tvt_true)
            return _nb_curve_cache[wid]

        def _nb_get_tree(wid):
            if wid not in _nb_tree_cache:
                _c = _nb_get_curve(wid)
                _nb_tree_cache[wid] = _NbTree(_c["xy"]) if _c is not None else None
            return _nb_tree_cache[wid]

        _nb_pred_by_id = {}
        for _wid in list_wells("test"):
            _hw_t = _nbpd.read_csv(CFG.DATA / "test" / f"{_wid}__horizontal_well.csv")
            _km_t = _hw_t["TVT_input"].notna()
            _xy_t = _hw_t[["X", "Y"]].to_numpy(_nbnp.float64)
            _z_t = _hw_t["Z"].to_numpy(_nbnp.float64)
            _kn_idx = _nbnp.where(_km_t.values)[0]
            _ev_idx = _nbnp.where(~_km_t.values)[0]
            if len(_kn_idx) < 20 or len(_ev_idx) < 5: continue

            if _wid in _nb_cent_ids:
                _i0 = _nb_cent_ids.index(_wid)
                _dist0, _idx0 = _nb_cent_tree.query(_nb_cent_xy[_i0], k=_NB_K + 5)
            else:
                _dist0, _idx0 = _nb_cent_tree.query(_xy_t.mean(axis=0), k=_NB_K + 5)
            # CRITICAL: exclude the target's OWN well ID by name (train/test ID overlap in this comp)
            _nn_wids = [_nb_cent_ids[_j] for _j in _idx0 if _nb_cent_ids[_j] != _wid][:_NB_K]

            _kn_pos_t = _hw_t.loc[_km_t, "TVT_input"].to_numpy(_nbnp.float64) + _z_t[_kn_idx]
            _preds_by_neighbor = []
            for _nn_wid in _nn_wids:
                _neighbor = _nb_get_curve(_nn_wid)
                _ntree = _nb_get_tree(_nn_wid)
                if _neighbor is None or _ntree is None: continue
                _neighbor_pos_full = _neighbor["tvt_true"] + _neighbor["z"]

                _dist_kn, _idx_kn = _ntree.query(_xy_t[_kn_idx], k=1)
                _borrowed_kn = _neighbor_pos_full[_idx_kn]
                _datum_offset = float(_nbnp.median(_kn_pos_t - _borrowed_kn))

                _dist_ev, _idx_ev = _ntree.query(_xy_t[_ev_idx], k=1)
                _borrowed_pos = _neighbor_pos_full[_idx_ev] + _datum_offset
                _pred_tvt = _borrowed_pos - _z_t[_ev_idx]
                _preds_by_neighbor.append((_pred_tvt, _dist_ev))

            if not _preds_by_neighbor: continue
            _preds_stack = _nbnp.stack([p for p, d in _preds_by_neighbor], axis=1)
            _dists_stack = _nbnp.stack([d for p, d in _preds_by_neighbor], axis=1)
            _w_idw = 1.0 / (_dists_stack + 50.0)
            _w_idw /= _w_idw.sum(axis=1, keepdims=True)
            _pred_idw = (_preds_stack * _w_idw).sum(axis=1)
            _dist_k1 = _preds_by_neighbor[0][1]
            _conf = _nbnp.exp(-_dist_k1 / _NB_DIST_SCALE)

            _raw_ids = [f"{_wid}_{_r}" for _r in _ev_idx]
            for _rid, _pv, _cf in zip(_raw_ids, _pred_idw, _conf):
                _nb_pred_by_id[_rid] = (float(_pv), float(_cf))

        if _nb_pred_by_id:
            _nbb = _nbpd.read_csv(OUT / "submission.csv")
            _nb_tvt = _nbb["tvt"].to_numpy(float).copy()
            _nb_ids = _nbb["id"].astype(str).to_numpy()
            _n_touched5 = 0
            for _i, _id in enumerate(_nb_ids):
                if _id in _nb_pred_by_id:
                    _pv, _cf = _nb_pred_by_id[_id]
                    _eff_w = _NEIGH_BLEND_W * _cf
                    _nb_tvt[_i] = (1.0 - _eff_w) * _nb_tvt[_i] + _eff_w * _pv
                    _n_touched5 += 1
            _nbb["tvt"] = _nb_tvt
            _order_ids5 = _nbpd.read_csv(CFG.DATA / "sample_submission.csv")["id"].astype(str)
            _nbb = _nbb.set_index("id").reindex(_order_ids5).reset_index()
            assert _nbb["tvt"].notna().all(), "neighbor blend lost ids vs sample"
            _nbb[["id", "tvt"]].to_csv(OUT / "submission.csv", index=False)
            POSTPROCESSORS.append(f"neighbor_blend_{_NEIGH_BLEND_W:.2f}")
            print(f"neighbor blend W={_NEIGH_BLEND_W}: touched {_n_touched5} eval rows")
        else:
            print("neighbor blend: no usable test wells found (skipped, submission unchanged)")
    except Exception as _nbe:
        print(f"neighbor blend FAILED, skipping safely (submission unchanged): {_nbe}")
else:
    print("neighbor blend disabled (W=0)")
'''

if __name__ == '__main__':
    print('CELL_NEIGHBOR_BLEND length:', len(CELL_NEIGHBOR_BLEND))
