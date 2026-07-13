import json, copy
SRC='d:/ROGII/7.178_diversepf_moreseeds192.ipynb'
DST='d:/ROGII/7.178_moreseeds192_wallhedge.ipynb'
nb=json.load(open(SRC,encoding='utf-8'))

# locate the cell that writes the final blend submission (cell 15 region) -> insert AFTER it,
# BEFORE guarded_contact_override so override overwrites & protects near-exact wells.
ins_idx=None
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code' and 'final blend written' in ''.join(c['source']):
        ins_idx=i+1; break
assert ins_idx is not None, "blend cell not found"
# sanity: next code cell should be the override
nxt=''.join(nb['cells'][ins_idx]['source'])
assert 'guarded_contact_override' in nxt or 'FORM_REF_PRIORITY' in nxt, f"unexpected next cell:\n{nxt[:200]}"

hedge = '''# --- WALL HEDGE: shrink eval predictions toward last known TVT on the UNCERTAIN wells ----------
# Validated on the full-pipeline proxy (773 wells, 5 independent splits + full set): the pipeline
# loses to the flat baseline on ~19% of wells -- the sub-seismic-fault "wall" wells whose dip shift
# the self-similar GR log cannot localize. A mild pull toward last_known_tvt hedges those. Every
# split showed monotone improvement at A in [0.05,0.15] (e.g. full set 10.826 -> 10.713 at 0.10),
# so this is variance-reduction (same family as the decorrelated ensemble that transferred to LB),
# NOT a new GR signal (which never transferred). Placed BEFORE guarded_contact_override so that any
# well the override resolves near-exactly (~0.01 RMSE) OVERWRITES the hedge and stays protected;
# the hedge therefore only touches the wells the pipeline is least confident about.
_WALL_HEDGE_A = float(os.environ.get("ROGII_WALL_HEDGE", "0.07"))
if _WALL_HEDGE_A > 0:
    _wh = pd.read_csv(OUT / "submission.csv")
    _wh_well, _wh_ri = split_id(_wh["id"]); _wh["well"] = _wh_well
    _last = {}
    for _wid in _wh["well"].unique():
        _lt = np.nan
        for _dir in ("test", "train"):
            _p = CFG.DATA / _dir / f"{_wid}__horizontal_well.csv"
            if _p.exists():
                _ti = pd.read_csv(_p, usecols=["TVT_input"])["TVT_input"].dropna()
                if len(_ti):
                    _lt = float(_ti.iloc[-1]); break
        _last[_wid] = _lt
    _lt_arr = _wh["well"].map(_last).to_numpy(float)
    _msk = np.isfinite(_lt_arr)
    _tvt = _wh["tvt"].to_numpy(float).copy()
    _tvt[_msk] = (1.0 - _WALL_HEDGE_A) * _tvt[_msk] + _WALL_HEDGE_A * _lt_arr[_msk]
    _wh["tvt"] = _tvt
    _order = pd.read_csv(CFG.DATA / "sample_submission.csv")["id"].astype(str)
    _wh = _wh.set_index("id").reindex(_order).reset_index()
    assert _wh["tvt"].notna().all(), "wall-hedge lost ids vs sample"
    _wh[["id", "tvt"]].to_csv(OUT / "submission.csv", index=False)
    POSTPROCESSORS.append(f"wall_hedge_shrink_{_WALL_HEDGE_A:.2f}")
    print(f"wall-hedge A={_WALL_HEDGE_A}: pulled {int(_msk.sum())} eval rows toward last_tvt "
          f"(override will re-protect near-exact wells downstream)")
else:
    print("wall-hedge disabled (A=0)")
'''

cell={"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
      "source":[l+"\n" for l in hedge.rstrip("\n").split("\n")]}
nb['cells'].insert(ins_idx, cell)
json.dump(nb, open(DST,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
print("inserted hedge cell at index", ins_idx, "-> total", len(nb['cells']), "cells")
print("wrote", DST)
