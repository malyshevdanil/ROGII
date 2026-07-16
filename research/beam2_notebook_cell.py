"""Submission cell for the beam decoder blend: a genuinely independent architecture (discrete joint
(segment_length, dip) beam search with log-normal length prior + persistence-dip prior + NCC emission),
re-opened after finding and fixing 2 real bugs in the original prototype (n_iter safety cap hit before
covering the eval zone at higher emission weights; uninitialized np.empty() tail left garbage/NaN
instead of falling back to the last segment's own dip). After the fix: corr(err, current_best_err)=0.354
on the true 160-well holdout -- the LOWEST correlation of any blend partner found this session (lower
than pf_z's 0.49-0.51) -- and the cleanest tail-risk profile (max single-well worsening only +0.67ft at
w=0.02, vs track6's best of +1.41ft). Paired bootstrap at w=0.02: +0.036ft, 95%CI=[-0.009,+0.081],
93.8% positive. Caveat: the effect size (+0.036ft, ~0.4% of the ~9.34ft base) is SMALLER than track6's
own +0.15ft effect, which failed to transfer on all 3 real submissions -- real-world transfer here is
uncertain, but the architecture is the most genuinely decorrelated found this session. No external
Kaggle dataset needed -- the 4 priors are baked in as constants (deterministic, no RNG at all, so no
seeding concerns unlike track6's PF).
"""

CELL_BEAM2_BLEND = '''# --- BEAM2 BLEND (independent joint segment-length/dip beam decoder) ------------------------------
# A genuinely different architecture from every other component in this pipeline: a discrete beam
# search over (segment_length, dip) hypotheses using a log-normal length prior + AR(1) dip-persistence
# prior fit from our own known-zone data, with NCC emission against the typewell. Re-opened after
# finding 2 real bugs in the first prototype (iteration cap hit before covering the eval zone at higher
# emission weights; uninitialized tail left as garbage). corr(err, current_best_err)=0.354 on the true
# 160-well holdout -- the lowest correlation of any blend partner tested this session -- with the
# cleanest tail-risk profile found (max single-well worsening +0.67ft at w=0.02). Paired bootstrap:
# +0.036ft, 95%CI=[-0.009,+0.081], 93.8% positive. No external dataset needed -- deterministic, no RNG.
_BEAM2_BLEND_W = float(os.environ.get("ROGII_BEAM2_BLEND", "0.02"))
if _BEAM2_BLEND_W > 0:
    try:
        import numpy as _b2np, pandas as _b2pd

        _B2_LOG_LEN_MU, _B2_LOG_LEN_SIGMA = 5.706, 0.807
        _B2_RHO, _B2_PERSIST_RESID_STD = 0.4725, 0.0325
        _B2_BEAM_WIDTH = 15
        _B2_LEN_QUANTILES = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
        _B2_DIP_QUANTILES = [-1.8, -1.0, -0.5, 0.0, 0.5, 1.0, 1.8]
        _B2_EMISSION_WEIGHT = 0.5
        _B2_MIN_SEG_LEN = 15.0

        def _b2_lognorm_ppf(q, mu, sigma):
            from scipy.stats import norm
            return float(_b2np.exp(mu + sigma * norm.ppf(q)))

        _B2_LEN_GRID = sorted({round(_b2_lognorm_ppf(q, _B2_LOG_LEN_MU, _B2_LOG_LEN_SIGMA), 1) for q in _B2_LEN_QUANTILES})
        _B2_LEN_GRID = [l for l in _B2_LEN_GRID if l >= _B2_MIN_SEG_LEN]

        def _b2_ncc(a, b):
            a = a - a.mean(); b = b - b.mean()
            na = _b2np.linalg.norm(a); nb = _b2np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9: return 0.0
            return float(_b2np.dot(a, b) / (na * nb))

        def _b2_calibrate(kn, tw_tvt, tw_gr, gr_full):
            kg = kn["GR"].values.astype(float); ktvt = kn["TVT_input"].values.astype(float)
            twk = _b2np.interp(ktvt, tw_tvt, tw_gr)
            v = _b2np.isfinite(kg) & _b2np.isfinite(twk)
            a, b = (_b2np.polyfit(kg[v], twk[v], 1) if v.sum() >= 20 else (1., 0.))
            return gr_full * a + b

        def _b2_decode_well(hw, tw_tvt, tw_gr, beam_width=_B2_BEAM_WIDTH):
            km = hw["TVT_input"].notna()
            kn = hw[km]; ev = hw[~km]
            if len(kn) < 50 or len(ev) < 50: return None
            gr_full = hw["GR"].interpolate(limit_direction="both").values.astype(float)
            cal = _b2_calibrate(kn, tw_tvt, tw_gr, gr_full)
            MD = hw["MD"].values.astype(float)
            md0 = float(kn["MD"].iloc[-1]); tvt0 = float(kn["TVT_input"].iloc[-1])
            tail = kn.tail(30)
            dt = _b2np.diff(tail["TVT_input"].values); dm = _b2np.diff(tail["MD"].values)
            m = dm > 0
            dip0 = float(_b2np.median(dt[m] / dm[m])) if m.sum() >= 3 else 0.0

            ev_md = ev["MD"].values.astype(float)
            md_end = float(ev_md[-1])

            beam = [dict(md=md0, tvt=tvt0, dip=dip0, logp=0.0, segs=[])]
            n_iter = 0
            while True:
                n_iter += 1
                active = [p for p in beam if p["md"] < md_end - 1.0]
                if not active or n_iter > 400:
                    break
                new_beam = []
                for p in beam:
                    if p["md"] >= md_end - 1.0:
                        new_beam.append(p); continue
                    for L in _B2_LEN_GRID:
                        new_md = min(p["md"] + L, md_end)
                        actual_L = new_md - p["md"]
                        if actual_L < _B2_MIN_SEG_LEN * 0.5: continue
                        for zq in _B2_DIP_QUANTILES:
                            new_dip = _B2_RHO * p["dip"] + zq * _B2_PERSIST_RESID_STD
                            new_tvt = p["tvt"] + new_dip * actual_L
                            seg_mask = (MD >= p["md"]) & (MD <= new_md)
                            seg_md = MD[seg_mask]
                            if len(seg_md) < 5: continue
                            seg_gr = cal[seg_mask]
                            seg_tvt_pred = p["tvt"] + new_dip * (seg_md - p["md"])
                            seg_tw = _b2np.interp(seg_tvt_pred, tw_tvt, tw_gr)
                            v = _b2np.isfinite(seg_gr) & _b2np.isfinite(seg_tw)
                            if v.sum() < 5: continue
                            corr = _b2_ncc(seg_gr[v], seg_tw[v])
                            len_lp = -0.5 * ((_b2np.log(max(actual_L, 1.0)) - _B2_LOG_LEN_MU) / _B2_LOG_LEN_SIGMA) ** 2
                            dip_lp = -0.5 * (zq ** 2)
                            score = _B2_EMISSION_WEIGHT * corr + len_lp + dip_lp
                            new_beam.append(dict(md=new_md, tvt=new_tvt, dip=new_dip,
                                                  logp=p["logp"] + score,
                                                  segs=p["segs"] + [(p["md"], new_md, p["tvt"], new_tvt)]))
                if not new_beam: break
                new_beam.sort(key=lambda x: -x["logp"])
                beam = new_beam[:beam_width]
                if all(p["md"] >= md_end - 1.0 for p in beam): break

            best = max(beam, key=lambda x: x["logp"])
            segs = best["segs"]
            if not segs: return None
            pred_tvt = _b2np.full(len(ev_md), _b2np.nan)
            for (a, b, ta, tb) in segs:
                mm = (ev_md >= a) & (ev_md <= b + 1e-6)
                if not mm.any(): continue
                frac = (ev_md[mm] - a) / max(b - a, 1e-6)
                pred_tvt[mm] = ta + frac * (tb - ta)
            if _b2np.isnan(pred_tvt).any():
                last_a, last_b, last_ta, last_tb = segs[-1]
                last_dip = (last_tb - last_ta) / max(last_b - last_a, 1e-6)
                mm = _b2np.isnan(pred_tvt)
                pred_tvt[mm] = last_tb + last_dip * (ev_md[mm] - last_b)
            return pred_tvt, ev.index.values

        _b2_pred_by_id = {}
        for _wid in list_wells("test"):
            hw = _b2pd.read_csv(CFG.DATA / "test" / f"{_wid}__horizontal_well.csv")
            tw = _b2pd.read_csv(CFG.DATA / "test" / f"{_wid}__typewell.csv").sort_values("TVT")
            tw_tvt = tw["TVT"].values.astype(float); tw_gr = tw["GR"].fillna(tw["GR"].mean()).values.astype(float)
            if len(tw_tvt) < 10: continue
            _r = _b2_decode_well(hw, tw_tvt, tw_gr)
            if _r is None: continue
            pred_tvt, ev_idx = _r
            raw_ids = [f"{_wid}_{_i}" for _i in ev_idx]
            for _rid, _pv in zip(raw_ids, pred_tvt):
                _b2_pred_by_id[_rid] = float(_pv)

        if _b2_pred_by_id:
            _b2b = _b2pd.read_csv(OUT / "submission.csv")
            _b2_tvt = _b2b["tvt"].to_numpy(float).copy()
            _b2_ids = _b2b["id"].astype(str).to_numpy()
            _n_touched4 = 0
            for _i, _id in enumerate(_b2_ids):
                if _id in _b2_pred_by_id:
                    _b2_tvt[_i] = (1.0 - _BEAM2_BLEND_W) * _b2_tvt[_i] + _BEAM2_BLEND_W * _b2_pred_by_id[_id]
                    _n_touched4 += 1
            _b2b["tvt"] = _b2_tvt
            _order_ids4 = _b2pd.read_csv(CFG.DATA / "sample_submission.csv")["id"].astype(str)
            _b2b = _b2b.set_index("id").reindex(_order_ids4).reset_index()
            assert _b2b["tvt"].notna().all(), "beam2 blend lost ids vs sample"
            _b2b[["id", "tvt"]].to_csv(OUT / "submission.csv", index=False)
            POSTPROCESSORS.append(f"beam2_blend_{_BEAM2_BLEND_W:.2f}")
            print(f"beam2 blend W={_BEAM2_BLEND_W}: touched {_n_touched4} eval rows")
        else:
            print("beam2 blend: no usable test wells found (skipped, submission unchanged)")
    except Exception as _b2e:
        print(f"beam2 blend FAILED, skipping safely (submission unchanged): {_b2e}")
else:
    print("beam2 blend disabled (W=0)")
'''

if __name__ == '__main__':
    print('CELL_BEAM2_BLEND length:', len(CELL_BEAM2_BLEND))
