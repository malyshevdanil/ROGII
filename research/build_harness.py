"""Build file 65: an honest offline harness around file51's pipeline.

Why this exists: the 3 visible test wells are byte-identical duplicates of train wells, so the
contact-override resolves them exactly and every preview/self-check number measures a leak rather
than skill. This wraps the real pipeline so it runs against leak-free held-out train wells instead,
and scores every intermediate stage against the withheld truth -- one run gives an honest read on
which layers actually help, with no competition submission spent.
"""
import json

SCRATCH = r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad'
nb = json.load(open(r'D:\ROGII\51_gru_theirbase_blend.ipynb', encoding='utf-8'))
cells = nb['cells']


def code_cell(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': src.splitlines(keepends=True)}


builder = r'''# ============== HONEST OFFLINE HARNESS: synthetic leak-free held-out test set ===================
# The 3 real visible test wells are byte-identical duplicates of train wells (verified: MD/GR/Z
# allclose 1e-9, train.TVT == test.TVT_input on every known row), so the contact-override resolves
# them EXACTLY (RMSE 0.0000) and preview scores measure a leak, not skill -- which is why preview
# never tracked the real LB and every idea cost a real submission to test.
# This cell builds a leak-free stand-in: take N train wells, mask them exactly like real test wells
# (contiguous known prefix, 20-33% of rows, formation columns and TVT dropped), place them in a
# synthetic test/ dir, and REMOVE them from the train/ pool so no duplicate can be looked up.
# Everything downstream runs unmodified; the withheld TVT is kept aside for scoring at the end.
import os, glob, random
from pathlib import Path
import numpy as np, pandas as pd

HARNESS_N_WELLS  = 12
HARNESS_SEED     = 7
HARNESS_KNOWN_LO = 0.204   # observed known-prefix fractions on the real test wells:
HARNESS_KNOWN_HI = 0.326   # 0.204 / 0.273 / 0.326
TEST_SCHEMA      = ['MD', 'X', 'Y', 'Z', 'GR', 'TVT_input']

_SRC  = Path(COMPETITION_DATA_ROOT)
_HARN = Path('/tmp/rogii_harness')          # /tmp, not /kaggle/working: keeps it out of kernel output
for _sub in ('train', 'test'):
    (_HARN / _sub).mkdir(parents=True, exist_ok=True)

_all_wells = sorted(Path(p).name.replace('__horizontal_well.csv', '')
                    for p in glob.glob(str(_SRC / 'train' / '*__horizontal_well.csv')))
_rng = random.Random(HARNESS_SEED)
HARNESS_WELLS = sorted(_rng.sample(_all_wells, HARNESS_N_WELLS))
print('harness: %d train wells available, holding out %d' % (len(_all_wells), HARNESS_N_WELLS))

_truth_rows, _sample_ids = [], []
for _w in HARNESS_WELLS:
    _hw = pd.read_csv(_SRC / 'train' / ('%s__horizontal_well.csv' % _w))
    _n = len(_hw)
    _k = int(round(_n * _rng.uniform(HARNESS_KNOWN_LO, HARNESS_KNOWN_HI)))
    _tvt = _hw['TVT'].to_numpy(float)
    _masked = _hw.copy()
    _masked['TVT_input'] = np.where(np.arange(_n) < _k, _tvt, np.nan)
    _masked[TEST_SCHEMA].to_csv(_HARN / 'test' / ('%s__horizontal_well.csv' % _w), index=False)
    pd.read_csv(_SRC / 'train' / ('%s__typewell.csv' % _w)).to_csv(
        _HARN / 'test' / ('%s__typewell.csv' % _w), index=False)
    for _i in range(_k, _n):
        _sample_ids.append('%s_%d' % (_w, _i))
        _truth_rows.append(('%s_%d' % (_w, _i), float(_tvt[_i])))
    print('  %s: n=%d known=%d (%.1f%%) eval=%d' % (_w, _n, _k, 100.0 * _k / _n, _n - _k))

_held = set(HARNESS_WELLS)
_linked = 0
for _w in _all_wells:
    if _w in _held:
        continue                                # the point of the harness: no lookup-able duplicate
    for _suf in ('__horizontal_well.csv', '__typewell.csv'):
        _src_f = _SRC / 'train' / ('%s%s' % (_w, _suf))
        _dst_f = _HARN / 'train' / ('%s%s' % (_w, _suf))
        if _dst_f.exists() or not _src_f.exists():
            continue
        try:
            os.symlink(_src_f, _dst_f)
        except Exception:
            _dst_f.write_bytes(_src_f.read_bytes())
    _linked += 1

pd.DataFrame({'id': _sample_ids, 'tvt': 0.0}).to_csv(_HARN / 'sample_submission.csv', index=False)
pd.DataFrame(_truth_rows, columns=['id', 'true_tvt']).to_csv('/kaggle/working/harness_truth.csv',
                                                             index=False)

COMPETITION_DATA_ROOT = str(_HARN)          # every downstream stage now reads the leak-free set
print('harness: train pool=%d wells (held-out removed), eval rows=%d' % (_linked, len(_sample_ids)))
print('harness: COMPETITION_DATA_ROOT ->', COMPETITION_DATA_ROOT)
'''

cells.insert(1, code_cell(builder))

# --- make the GBM/ridge anchor honest for the held-out wells ----------------------------------
# The LGB/CatBoost trainers are loaded from ravaghi's cached artifact, so train_df CANNOT be filtered
# (trainer.oof_preds is a stored array sized to the full 773-well train set -- filtering desynchronises
# it from y and the ridge stage dies on a shape mismatch).
# It also does not need to be: trainer.oof_preds are OUT-OF-FOLD predictions under GroupKFold BY WELL,
# so each well's OOF value comes from a fold-model that never saw that well -- honest by construction.
# The one genuinely leaked path is trainer.predict(X_test), which uses models fitted on all 773 wells
# including our held-out ones. So instead of filtering, overwrite the ridge anchor for the held-out
# wells with their honest OOF counterparts, matched by row id.
honest_anchor = r'''# HARNESS: replace the (leaked) ridge test predictions with honest out-of-fold values ------------
import numpy as _ha_np, pandas as _ha_pd

_ha_held = set(str(w) for w in globals().get('HARNESS_WELLS', []))
if _ha_held:
    _ha_oof = _ha_np.asarray(ridge_oof_preds, dtype=float).ravel()
    assert len(_ha_oof) == len(train_df), 'ridge_oof_preds does not align with train_df'
    _ha_map = _ha_pd.Series(_ha_oof, index=train_df['id'].astype(str).to_numpy())
    _ha_map = _ha_map[~_ha_map.index.duplicated(keep='first')]

    _ha_test_ids = test_df['id'].astype(str).to_numpy()
    _ha_vals = _ha_map.reindex(_ha_test_ids).to_numpy(dtype=float)
    _ha_found = int(_ha_np.isfinite(_ha_vals).sum())
    print('harness: honest-anchor direct id coverage %d/%d test rows (%.1f%%)'
          % (_ha_found, len(_ha_test_ids), 100.0 * _ha_found / max(len(_ha_test_ids), 1)))
    if _ha_found < 0.90 * len(_ha_test_ids):
        raise RuntimeError('harness: OOF anchor covers only %d of %d test rows -- too low to repair, '
                           'train_df/test_df id formats likely differ' % (_ha_found, len(_ha_test_ids)))

    # build_dataset does not emit a row for every well row (rolling-window warm-up etc.), so a few
    # percent of test rows have no direct OOF counterpart. Fill those by interpolating along MD order
    # from the SAME well's honest OOF values -- still leak-free, no fitted model involved.
    _ha_well = test_df['well'].astype(str).to_numpy()
    _ha_row = test_df['id'].astype(str).str.rsplit('_', n=1).str[-1].astype(int).to_numpy()
    for _w in sorted(set(_ha_well)):
        _m = _ha_np.flatnonzero(_ha_well == _w)
        if _m.size == 0:
            continue
        _order = _m[_ha_np.argsort(_ha_row[_m])]
        _r, _v = _ha_row[_order], _ha_vals[_order]
        _ok = _ha_np.isfinite(_v)
        if _ok.sum() < 2:
            raise RuntimeError('harness: well %s has %d honest anchor values, cannot interpolate'
                               % (_w, int(_ok.sum())))
        _ha_vals[_order] = _ha_np.interp(_r, _r[_ok], _v[_ok])
    assert _ha_np.isfinite(_ha_vals).all(), 'harness: anchor still has gaps after interpolation'
    print('harness: filled %d remaining rows by within-well interpolation'
          % (len(_ha_test_ids) - _ha_found))

    _ha_before = _ha_np.asarray(ridge_test_preds, dtype=float).ravel().copy()
    ridge_test_preds = _ha_vals
    print('harness: swapped leaked -> honest anchor, mean abs shift %.4f'
          % float(_ha_np.abs(ridge_test_preds - _ha_before).mean()))
'''
i_ridge = next(i for i, c in enumerate(cells)
               if 'ridge_test_preds = ridge_trainer.predict' in ''.join(c['source']))
cells.insert(i_ridge + 1, code_cell(honest_anchor))
print('inserted honest-anchor swap after ridge cell', i_ridge)


# --- make the second (PF / sp45 / learned-trajectory) section read the harness root ------------
# That section defines its OWN _find_data() with the competition path hardcoded, so it silently
# evaluated the real 3 test wells (14151 rows) while the anchor section above used the harness set
# (59554 rows) -- the two then collided in the sp45/learned blend with "Blend id mismatch".
i_fd = next(i for i, c in enumerate(cells) if 'def _find_data' in ''.join(c['source']))
src_fd = ''.join(cells[i_fd]['source'])
old_head = ('def _find_data():\n'
            '    for c in ["/kaggle/input/competitions/rogii-wellbore-geology-prediction",\n'
            '              "/kaggle/input/rogii-wellbore-geology-prediction"]:\n'
            '        if Path(c).exists() and (Path(c)/"train").exists():')
new_head = ('def _find_data():\n'
            '    # HARNESS: honour the data root chosen at the top of this notebook first, so this\n'
            '    # section evaluates the same held-out wells as the anchor section.\n'
            '    for c in [globals().get("COMPETITION_DATA_ROOT", ""),\n'
            '              "/kaggle/input/competitions/rogii-wellbore-geology-prediction",\n'
            '              "/kaggle/input/rogii-wellbore-geology-prediction"]:\n'
            '        if c and Path(c).exists() and (Path(c)/"train").exists():')
assert old_head in src_fd, 'unexpected _find_data() shape in cell %d' % i_fd
cells[i_fd]['source'] = src_fd.replace(old_head, new_head).splitlines(keepends=True)
print('patched _find_data() to honour the harness root in cell', i_fd)

# --- neutralise Q0522 under the harness --------------------------------------------------------
# Q0522 is a hand-tuned constant shift for one specific REAL test well (00e12e8b) and hard-raises on
# any deviation from its recorded sha/stats/row-count. None of that applies to held-out wells, so it
# would abort the harness for no analytical loss -- skip it when the harness is active.
i_q = next(i for i, c in enumerate(cells) if "_EX_LABEL = 'Q0522'" in ''.join(c['source']))
body = ''.join(cells[i_q]['source'])
assert '"""' not in body and "'''" not in body, 'Q0522 cell has triple-quoted text; indenting is unsafe'
guarded = ('# HARNESS: skip the well-specific Q0522 hedge (targets real test well 00e12e8b only).\n'
           'if globals().get("HARNESS_WELLS"):\n'
           '    print("harness: Q0522 skipped -- hand-tuned hedge for a real test well, absent here")\n'
           'else:\n'
           + '\n'.join(('    ' + ln) if ln.strip() else ln for ln in body.split('\n')))
cells[i_q]['source'] = guarded.splitlines(keepends=True)
print('guarded the Q0522 cell for harness runs at cell', i_q)


def snapshot(tag):
    return code_cell(
        "# HARNESS snapshot: %s\n"
        "import shutil as _snap_sh\n"
        "from pathlib import Path as _SnapPath\n"
        "_snap_w = _SnapPath('/kaggle/working')\n"
        "try:\n"
        "    _snap_sh.copy(_snap_w / 'submission.csv', _snap_w / 'harness_stage_%s.csv')\n"
        "    print('harness snapshot: %s')\n"
        "except Exception as _snap_e:\n"
        "    print('harness snapshot %s skipped:', _snap_e)\n" % (tag, tag, tag, tag))


i_beam = next(i for i, c in enumerate(cells)
              if ''.join(c['source']).strip().startswith('# --- PORTABLE BEAM2'))
tags = ['00_pre_blends', '01_after_beam2', '02_after_neighbor', '03_after_warp', '04_after_gru']
cells.insert(i_beam, snapshot(tags[0]))
for k in range(1, 5):
    cells.insert(i_beam + 2 * k, snapshot(tags[k]))
print('inserted 5 stage snapshots around the blend cells at', i_beam)

scorer = r'''# ================ HARNESS SCORING: honest per-stage RMSE on leak-free wells =====================
import glob as _sc_glob, os as _sc_os
import numpy as _sc_np, pandas as _sc_pd

_truth = _sc_pd.read_csv('/kaggle/working/harness_truth.csv',
                         dtype={'id': 'string'}).set_index('id')['true_tvt']

def _sc_eval(path):
    df = _sc_pd.read_csv(path, dtype={'id': 'string'})
    if 'tvt' not in df.columns or 'id' not in df.columns:
        return None
    df['true'] = df['id'].astype(str).map(_truth)
    df = df.dropna(subset=['true'])
    if df.empty:
        return None
    err = df['tvt'].astype(float).to_numpy() - df['true'].astype(float).to_numpy()
    pooled = float(_sc_np.sqrt(_sc_np.mean(err ** 2)))
    per = (df.assign(w=df['id'].astype(str).str.rsplit('_', n=1).str[0], e2=err ** 2)
             .groupby('w')['e2'].mean().pow(0.5))
    return pooled, float(per.mean()), per

_paths = sorted(_sc_glob.glob('/kaggle/working/harness_stage_*.csv'))
_paths += sorted(_sc_glob.glob('/kaggle/working/submission*.csv'))

print('=' * 78)
print('%-48s %9s %9s' % ('stage', 'pooled', 'perwell'))
print('=' * 78)
_rows = []
for _p in _paths:
    try:
        _r = _sc_eval(_p)
    except Exception:
        _r = None
    if _r is None:
        continue
    print('%-48s %9.4f %9.4f' % (_sc_os.path.basename(_p), _r[0], _r[1]))
    _rows.append(dict(stage=_sc_os.path.basename(_p), pooled_rmse=_r[0], per_well_rmse=_r[1]))
print('=' * 78)
_sc_pd.DataFrame(_rows).to_csv('/kaggle/working/harness_scores.csv', index=False)

_final = _sc_eval('/kaggle/working/submission.csv')
if _final is not None:
    print('\nper-well RMSE of the FINAL submission:')
    print(_final[2].sort_values(ascending=False).to_string())
'''
cells.append(code_cell(scorer))

nb['cells'] = cells
out = SCRATCH + r'\rogii_65_kernel\65_offline_harness.ipynb'
json.dump(nb, open(out, 'w', encoding='utf-8'))
print('total cells:', len(cells))

import ast
bad = 0
for i, c in enumerate(cells):
    if c['cell_type'] != 'code':
        continue
    try:
        ast.parse(''.join(c['source']))
    except SyntaxError:
        bad += 1
print('cells failing local py311 parse (py312 f-strings expected):', bad)
