import json

ROOT = 'd:/ROGII'
SRC = f'{ROOT}/7.178_diversepf_moreseeds192.ipynb'   # the 7.080 best, untouched
DST = f'{ROOT}/7.178_moreseeds192_warpblend.ipynb'
# WARP checkpoint is NOT embedded (Kaggle caps kernel source at 1MB) -- it is attached as the
# Kaggle Dataset "malyshevdanil/rogii-warp-checkpoint" and read from /kaggle/input/ at runtime.

nb = json.load(open(SRC, encoding='utf-8'))

# ---------------------------------------------------------------------------
# Cell A: WARP decorrelation blend. Insert AFTER the sp45+fleongg final blend
# ("final blend written"), BEFORE guarded_contact_override (so the override
# still re-protects near-exact contact-resolved wells by overwriting them).
# ---------------------------------------------------------------------------
warp_cell_src = '''# --- WARP DECORRELATION BLEND ------------------------------------------------------------------
# Validated on the full-pipeline proxy (160 held-out wells, cross-fit BOTH directions of an 80/80
# split): a small continuity-anchored CNN (WARP -- predicts dTVT, integrates via cumsum anchored at
# last_known_tvt, reads the typewell via cross-attention, deliberately does NOT trust GR the way the
# PF/GBM stack does) has error correlation only 0.52 with the tuned sp45/fleongg blend. Blending the
# two gives a real, cross-validated gain (proxy: sp45-only 10.62 -> blended ~9.8-10.2 depending on
# weight, improved on BOTH halves of the split, never worse). A "shrink toward last_tvt" wall-hedge
# was ALSO tested combined with this blend and found REDUNDANT/HARMFUL once WARP is included (its
# own continuity anchor already captures that effect: joint-optimal hedge weight collapsed to 0 on
# both cross-fits) -- so this stage stands alone, it does not stack with any separate hedge. Placed
# BEFORE guarded_contact_override so any well the override resolves near-exactly (~0.01 RMSE) stays
# protected (override runs after and overwrites this stage's output there). The WARP checkpoint was
# trained offline on the 773 labeled training wells (whole-well holdout, seed 42). Kaggle caps kernel
# SOURCE at 1MB, so the ~4.4MB checkpoint cannot be embedded in the notebook -- attach the Kaggle
# Dataset "malyshevdanil/rogii-warp-checkpoint" as an input (same way as the fleongg/ravaghi
# datasets); this cell finds best_warp.pt under /kaggle/input/ automatically.
# Runs on CPU or GPU -- the model is small and sequences short, so inference is seconds either way.
_WARP_BLEND_A = float(os.environ.get("ROGII_WARP_BLEND", "0.15"))
if _WARP_BLEND_A > 0:
    try:
        import glob as _glob, torch as _wtorch, torch.nn as _wnn, torch.nn.functional as _wF
        import numpy as _wnp, pandas as _wpd

        _WDEV = "cuda" if _wtorch.cuda.is_available() else "cpu"
        _W_KS, _W_ES, _W_TG, _W_D, _W_MAXSTEP = 60, 360, 256, 128, 1.5

        class _WEnc(_wnn.Module):
            def __init__(s, nin, d):
                super().__init__(); s.inp = _wnn.Conv1d(nin, d, 5, padding=2)
                s.blocks = _wnn.ModuleList([_wnn.Sequential(
                    _wnn.Conv1d(d, d, 3, padding=dl, dilation=dl), _wnn.GroupNorm(8, d), _wnn.GELU(), _wnn.Dropout(0.0),
                    _wnn.Conv1d(d, d, 3, padding=dl, dilation=dl), _wnn.GroupNorm(8, d), _wnn.GELU())
                    for dl in (1, 2, 4, 8, 16)])
                s.out = _wnn.Conv1d(d, d, 1)
            def forward(s, x):
                h = _wF.gelu(s.inp(x))
                for b in s.blocks: h = h + b(h)
                return s.out(h)

        class _WarpNet(_wnn.Module):
            def __init__(s, d):
                super().__init__(); s.he = _WEnc(4, d); s.te = _WEnc(1, d)
                s.q = _wnn.Conv1d(d, d, 1); s.k = _wnn.Conv1d(d, d, 1); s.vv = _wnn.Conv1d(d, d, 1); s.sc = d ** -0.5
                s.head = _wnn.Sequential(_wnn.Linear(2 * d, d), _wnn.GELU(), _wnn.Dropout(0.0), _wnn.Linear(d, 1))
            def forward(s, H, G, lt):
                h = s.he(H); t = s.te(G)
                Q = s.q(h).transpose(1, 2); K = s.k(t).transpose(1, 2); V = s.vv(t).transpose(1, 2)
                att = _wtorch.softmax(Q @ K.transpose(1, 2) * s.sc, dim=2); ctx = att @ V
                x = _wtorch.cat([h.transpose(1, 2), ctx], dim=2)
                dt = _wtorch.tanh(s.head(x)[..., 0]) * _W_MAXSTEP
                tvt = _wtorch.cumsum(dt, 1)
                return tvt - tvt[:, _W_KS - 1:_W_KS] + lt[:, None]

        _warp_ckpt_path = None
        for _p in _glob.glob("/kaggle/input/**/best_warp.pt", recursive=True):
            _warp_ckpt_path = _p; break
        if _warp_ckpt_path is None:
            _local_p = os.environ.get("ROGII_WARP_CKPT", "best_warp.pt")
            if os.path.exists(_local_p): _warp_ckpt_path = _local_p
        if _warp_ckpt_path is None:
            raise FileNotFoundError(
                "best_warp.pt not found -- attach the 'rogii-warp-checkpoint' Kaggle Dataset "
                "(malyshevdanil/rogii-warp-checkpoint) as an input, or set ROGII_WARP_CKPT locally.")
        _wsd = _wtorch.load(_warp_ckpt_path, map_location=_WDEV)
        _wnet = _WarpNet(_W_D).to(_WDEV); _wnet.load_state_dict(_wsd); _wnet.eval()

        def _w_inan(a):
            a = a.copy(); m = _wnp.isnan(a); i = _wnp.arange(len(a))
            if m.all(): return _wnp.zeros(len(a))
            a[m] = _wnp.interp(i[m], i[~m], a[~m]); return a

        def _w_build(wid):
            # NOTE: real test files have NO "TVT" ground-truth column (only MD,X,Y,Z,GR,TVT_input) --
            # this function must never reference hw["TVT"].
            hw = _wpd.read_csv(CFG.DATA / "test" / f"{wid}__horizontal_well.csv")
            tw = _wpd.read_csv(CFG.DATA / "test" / f"{wid}__typewell.csv").sort_values("TVT")
            tt = tw["TVT"].values.astype(float); tg = tw["GR"].fillna(tw["GR"].mean()).values.astype(float)
            if len(tt) < 10: return None
            kn = hw[hw["TVT_input"].notna()]
            if len(kn) < 20 or hw["TVT_input"].isna().sum() < 20: return None
            last_tvt = float(kn["TVT_input"].iloc[-1])
            n = len(hw); gr = _w_inan(hw["GR"].values.astype(float))
            kn_gr = kn["GR"].interpolate().bfill().ffill().values.astype(float)
            twk = _wnp.interp(kn["TVT_input"].values, tt, tg)
            v = _wnp.isfinite(kn_gr) & _wnp.isfinite(twk)
            a, b = (_wnp.polyfit(kn_gr[v], twk[v], 1) if v.sum() >= 20 else (1., 0.))
            cal = gr * a + b
            Z = hw["Z"].values.astype(float); MD = hw["MD"].values.astype(float)
            mdd = _wnp.gradient(MD); mdd[mdd == 0] = 1
            grad = _wnp.gradient(gr)
            rstd = _wpd.Series(gr).rolling(21, center=True, min_periods=1).std().fillna(0).values
            dz = _wnp.gradient(Z) / mdd
            ev = hw["TVT_input"].isna().values.astype(float); ei = _wnp.where(ev > 0.5)[0]
            if len(ei) < 5: return None
            e0 = ei[0]; ks = _wnp.arange(max(0, e0 - 400), e0)
            if len(ks) < 5: return None
            g_tvt = _wnp.linspace(tt.min(), tt.max(), _W_TG); g_gr = _wnp.interp(g_tvt, tt, tg)
            gm, gs = float(g_gr.mean()), float(g_gr.std() + 1e-6)
            caln = (cal - gm) / gs; gradn = grad / gs; rstdn = rstd / gs
            kd = _wnp.linspace(ks[0], ks[-1], _W_KS); ed = _wnp.linspace(e0, n - 1, _W_ES)
            dst = _wnp.concatenate([kd, ed]); src = _wnp.arange(n)
            R = lambda x: _wnp.interp(dst, src, x).astype(_wnp.float32)
            return dict(wid=wid, H=_wnp.stack([R(caln), R(gradn), R(rstdn), R(dz)]).astype(_wnp.float32),
                        gn=((g_gr - gm) / gs).astype(_wnp.float32), last_tvt=last_tvt,
                        md_grid=R(MD), md_raw=MD, ev_raw=hw["TVT_input"].isna().values)

        _wells_feat = {}; _dz_all = []
        for _wid in list_wells("test"):
            _b = _w_build(_wid)
            if _b is not None:
                _wells_feat[_wid] = _b; _dz_all.append(_b["H"][3])

        if _wells_feat:
            _dzcat = _wnp.concatenate(_dz_all); _dzm, _dzs = _dzcat.mean(), _dzcat.std() + 1e-6
            _warp_pred_by_id = {}
            with _wtorch.no_grad():
                for _wid, _b in _wells_feat.items():
                    _H = _b["H"].copy(); _H[3] = (_H[3] - _dzm) / _dzs
                    _Ht = _wtorch.tensor(_H[None], dtype=_wtorch.float32, device=_WDEV)
                    _Gt = _wtorch.tensor(_b["gn"][None, None, :_W_TG], dtype=_wtorch.float32, device=_WDEV)
                    _Lt = _wtorch.tensor([_b["last_tvt"]], dtype=_wtorch.float32, device=_WDEV)
                    _pred_grid = _wnet(_Ht, _Gt, _Lt)[0].cpu().numpy()
                    _md_ev = _b["md_grid"][_W_KS:]; _pred_ev = _pred_grid[_W_KS:]
                    _order = _wnp.argsort(_md_ev)
                    _md_raw_ev = _b["md_raw"][_b["ev_raw"]]
                    _warp_interp = _wnp.interp(_md_raw_ev, _md_ev[_order], _pred_ev[_order])
                    _raw_idx = _wnp.where(_b["ev_raw"])[0]
                    for _ridx, _pv in zip(_raw_idx, _warp_interp):
                        _warp_pred_by_id[f"{_wid}_{_ridx}"] = float(_pv)

            _wb = _wpd.read_csv(OUT / "submission.csv")
            _wb_tvt = _wb["tvt"].to_numpy(float).copy()
            _wb_ids = _wb["id"].astype(str).to_numpy()
            _n_touched = 0
            for _i, _id in enumerate(_wb_ids):
                if _id in _warp_pred_by_id:
                    _wb_tvt[_i] = (1.0 - _WARP_BLEND_A) * _wb_tvt[_i] + _WARP_BLEND_A * _warp_pred_by_id[_id]
                    _n_touched += 1
            _wb["tvt"] = _wb_tvt
            _order_ids = _wpd.read_csv(CFG.DATA / "sample_submission.csv")["id"].astype(str)
            _wb = _wb.set_index("id").reindex(_order_ids).reset_index()
            assert _wb["tvt"].notna().all(), "warp-blend lost ids vs sample"
            _wb[["id", "tvt"]].to_csv(OUT / "submission.csv", index=False)
            POSTPROCESSORS.append(f"warp_blend_{_WARP_BLEND_A:.2f}")
            print(f"WARP-blend A={_WARP_BLEND_A}: touched {_n_touched} eval rows across {len(_wells_feat)} wells "
                  f"(device={_WDEV})")
        else:
            print("WARP-blend: no usable test wells found (skipped, submission unchanged)")
    except Exception as _e:
        print(f"WARP-blend FAILED, skipping safely (submission unchanged): {_e}")
else:
    print("WARP-blend disabled (A=0)")
'''

# ---------------------------------------------------------------------------
# Cell B: adversarial-validation diagnostic (train vs test). Non-destructive,
# never touches submission.csv, wrapped so any failure cannot break the run.
# Placed early (right after the cell that defines list_wells / split_id).
# ---------------------------------------------------------------------------
adv_cell_src = '''# --- ADVERSARIAL VALIDATION (diagnostic only, never touches submission.csv) --------------------
# Trains a classifier to distinguish train vs test wells on simple per-well summary features. A
# high AUC would mean the test set is distributionally distinguishable from train -- i.e. some
# features the pipeline relies on may not generalize, a risk factor for private-LB shake-up. This
# is purely informational: it writes adv_val_report.json to the output folder and does not alter
# the submission. Wrapped in try/except so a failure here can never break the real submission.
try:
    import lightgbm as _advlgb
    from sklearn.model_selection import StratifiedKFold as _AdvSKF
    from sklearn.metrics import roc_auc_score as _adv_auc

    def _adv_feat(wid, split):
        hw = pd.read_csv(CFG.DATA / split / f"{wid}__horizontal_well.csv")
        tw = pd.read_csv(CFG.DATA / split / f"{wid}__typewell.csv")
        kn = hw[hw["TVT_input"].notna()] if "TVT_input" in hw.columns else hw.iloc[:0]
        ev = hw[hw["TVT_input"].isna()] if "TVT_input" in hw.columns else hw
        gr = hw["GR"].dropna(); md = hw["MD"]; z = hw["Z"] if "Z" in hw.columns else pd.Series(dtype=float)
        f = dict(n_rows=len(hw), n_known=len(kn), n_eval=len(ev),
                 md_span=md.max() - md.min(), gr_mean=gr.mean(), gr_std=gr.std(),
                 gr_min=gr.min(), gr_max=gr.max(),
                 z_mean=z.mean() if len(z) else np.nan, z_std=z.std() if len(z) else np.nan,
                 z_span=(z.max() - z.min()) if len(z) else np.nan,
                 tw_rows=len(tw),
                 tw_tvt_span=(tw["TVT"].max() - tw["TVT"].min()) if "TVT" in tw.columns else np.nan,
                 tw_gr_mean=tw["GR"].mean() if "GR" in tw.columns else np.nan,
                 tw_gr_std=tw["GR"].std() if "GR" in tw.columns else np.nan,
                 known_frac=len(kn) / max(1, len(hw)))
        if "X" in hw.columns and "Y" in hw.columns:
            f["x_mean"] = hw["X"].mean(); f["y_mean"] = hw["Y"].mean()
            f["xy_span"] = float(np.hypot(hw["X"].max() - hw["X"].min(), hw["Y"].max() - hw["Y"].min()))
        f["last_tvt"] = float(kn["TVT_input"].iloc[-1]) if len(kn) else np.nan
        return f

    _adv_train_w = list_wells("train"); _adv_test_w = list_wells("test")
    _rows = []; _labels = []
    for _w in _adv_train_w:
        try: _rows.append(_adv_feat(_w, "train")); _labels.append(0)
        except Exception: pass
    for _w in _adv_test_w:
        try: _rows.append(_adv_feat(_w, "test")); _labels.append(1)
        except Exception: pass
    _advX = pd.DataFrame(_rows); _advy = np.array(_labels)
    _adv_result = dict(n_train=len(_adv_train_w), n_test=len(_adv_test_w))
    if len(set(_advy)) < 2 or min((_advy == 0).sum(), (_advy == 1).sum()) < 10:
        _adv_result["auc"] = None
        _adv_result["verdict"] = "too few wells of one class for a meaningful AUC (expected on the local test stub)"
    else:
        _adv_oof = np.zeros(len(_advy))
        _adv_skf = _AdvSKF(n_splits=5, shuffle=True, random_state=0)
        for _tri, _vai in _adv_skf.split(_advX, _advy):
            _advm = _advlgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                            min_child_samples=10, verbosity=-1)
            _advm.fit(_advX.iloc[_tri], _advy[_tri])
            _adv_oof[_vai] = _advm.predict_proba(_advX.iloc[_vai])[:, 1]
        _auc = float(_adv_auc(_advy, _adv_oof))
        _advm_full = _advlgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                             min_child_samples=10, verbosity=-1)
        _advm_full.fit(_advX, _advy)
        _adv_imp = sorted(zip(_advX.columns, _advm_full.feature_importances_), key=lambda t: -t[1])[:10]
        _adv_result["auc"] = _auc
        _adv_result["top_features"] = [(n, int(i)) for n, i in _adv_imp]
        _adv_result["verdict"] = ("distribution shift risk (AUC>0.6): consider dropping top features"
                                   if _auc > 0.6 else "no strong shift detected (AUC<=0.6)")
    with open(OUT / "adv_val_report.json", "w", encoding="utf-8") as _f:
        json.dump(_adv_result, _f, indent=2, default=str)
    print("adversarial validation:", _adv_result)
except Exception as _adv_e:
    print(f"adversarial validation skipped (non-fatal): {_adv_e}")
'''

def make_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in src.rstrip("\n").split("\n")]}

# find insertion points
blend_idx = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and 'final blend written' in ''.join(c['source']):
        blend_idx = i + 1
        break
assert blend_idx is not None, "blend cell not found"
nxt = ''.join(nb['cells'][blend_idx]['source'])
assert 'guarded_contact_override' in nxt or 'FORM_REF_PRIORITY' in nxt, f"unexpected cell after blend:\n{nxt[:200]}"

helpers_idx = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and 'def list_wells' in ''.join(c['source']):
        helpers_idx = i + 1
        break
assert helpers_idx is not None, "helpers cell (list_wells) not found"

# insert WARP-blend first (later index), then adv-val (earlier index) so indices don't shift wrongly
nb['cells'].insert(blend_idx, make_cell(warp_cell_src))
nb['cells'].insert(helpers_idx, make_cell(adv_cell_src))

json.dump(nb, open(DST, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
import os
print('wrote', DST, 'size(MB)=', round(os.path.getsize(DST) / 1e6, 2), 'cells=', len(nb['cells']))
