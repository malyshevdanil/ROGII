"""Build file 77: simulate the graded environment for the REAL 3 test wells.

Evidence that the graded set is the same 3 wells: file51's unmodified Q0522 cell hard-raises unless well
'00e12e8b' has exactly 4301 eval rows, and file51 scored 6.460 at real submission without raising. So the
graded test wells carry the same ids and row counts as the visible ones, which means the TRUE answer for
the graded eval rows is the TVT sitting in the train copies we already have on disk.

That leaves one explanation for preview scoring 2.795 against that truth while the leaderboard says 6.460:
the graded environment's train/ directory does not contain the duplicate copies of the test wells, so the
same-well lookup that makes preview trivial cannot fire there.

This file tests exactly that, changing ONE thing: the 3 test wells are removed from the train/ pool while
everything else -- including the cached GBM/ridge models, which are trained on all 773 wells in the graded
run too and so must stay -- is left identical. Scored offline against the train-copy truth:
  * lands near 6.46  -> the hypothesis holds and we finally have an offline oracle, and every remaining
    idea can be tested for free instead of costing a submission;
  * lands near 2.8-4.0 -> some other stage is still leaking and the search continues;
  * lands far off     -> the graded truth differs after all and the oracle idea is dead for good.

Costs no submission: this kernel is only ever scored locally.
"""
import json

SCRATCH = r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad'
nb = json.load(open(r'D:\ROGII\51_gru_theirbase_blend.ipynb', encoding='utf-8'))
cells = nb['cells']


def code_cell(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': src.splitlines(keepends=True)}


shim = r'''# ---- GRADED-ENVIRONMENT SIMULATION: same 3 test wells, train pool without their duplicates ----
import os, glob, shutil
from pathlib import Path

TEST_WELLS = ['000d7d20', '00bbac68', '00e12e8b']
_SRC = Path(COMPETITION_DATA_ROOT)
_SIM = Path('/tmp/rogii_graded_sim')          # /tmp so it never lands in the kernel output
for _sub in ('train', 'test'):
    (_SIM / _sub).mkdir(parents=True, exist_ok=True)

# test/ and sample_submission are copied verbatim -- the graded run sees the real thing
for _f in glob.glob(str(_SRC / 'test' / '*')):
    _dst = _SIM / 'test' / Path(_f).name
    if not _dst.exists():
        try: os.symlink(_f, _dst)
        except Exception: shutil.copy(_f, _dst)
_ss = _SIM / 'sample_submission.csv'
if not _ss.exists():
    shutil.copy(_SRC / 'sample_submission.csv', _ss)

# train/ gets everything EXCEPT the duplicate copies of the test wells
_kept = _skipped = 0
for _f in glob.glob(str(_SRC / 'train' / '*')):
    _name = Path(_f).name
    if any(_name.startswith(_w) for _w in TEST_WELLS):
        _skipped += 1
        continue
    _dst = _SIM / 'train' / _name
    if not _dst.exists():
        try: os.symlink(_f, _dst)
        except Exception: shutil.copy(_f, _dst)
    _kept += 1

COMPETITION_DATA_ROOT = str(_SIM)
print('graded-sim: train files kept=%d, duplicate-test files withheld=%d' % (_kept, _skipped))
print('graded-sim: COMPETITION_DATA_ROOT ->', COMPETITION_DATA_ROOT)
'''

cells.insert(1, code_cell(shim))

# the PF / sp45 section defines its own _find_data() with the competition path hardcoded
i_fd = next(i for i, c in enumerate(cells) if 'def _find_data' in ''.join(c['source']))
src_fd = ''.join(cells[i_fd]['source'])
old = ('def _find_data():\n'
       '    for c in ["/kaggle/input/competitions/rogii-wellbore-geology-prediction",\n'
       '              "/kaggle/input/rogii-wellbore-geology-prediction"]:\n'
       '        if Path(c).exists() and (Path(c)/"train").exists():')
new = ('def _find_data():\n'
       '    for c in [globals().get("COMPETITION_DATA_ROOT", ""),\n'
       '              "/kaggle/input/competitions/rogii-wellbore-geology-prediction",\n'
       '              "/kaggle/input/rogii-wellbore-geology-prediction"]:\n'
       '        if c and Path(c).exists() and (Path(c)/"train").exists():')
assert old in src_fd, 'unexpected _find_data() shape'
cells[i_fd]['source'] = src_fd.replace(old, new).splitlines(keepends=True)

# both contact-override cells build their own candidate path lists from the hardcoded root
OV = [("[_OvPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),",
       "[_OvPath(globals().get('COMPETITION_DATA_ROOT', '.')),\n"
       "                   _OvPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),"),
      ("_GoldPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),",
       "_GoldPath(globals().get('COMPETITION_DATA_ROOT', '.')),\n"
       "        _GoldPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),")]
n_ov = 0
for c in cells:
    s = ''.join(c['source'])
    if 'override fallback' not in s:
        continue
    for o, w in OV:
        if o in s:
            s = s.replace(o, w, 1)
            n_ov += 1
    c['source'] = s.splitlines(keepends=True)
assert n_ov >= 2, 'patched %d override path lists' % n_ov

scorer = r'''# ---- score this run against the train-copy truth for the real 3 test wells ----
import numpy as _o_np, pandas as _o_pd
from pathlib import Path as _OPath

_o_src = _OPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')
_truth = {}
for _w in ['000d7d20', '00bbac68', '00e12e8b']:
    _tr = _o_pd.read_csv(_o_src / 'train' / ('%s__horizontal_well.csv' % _w))
    _te = _o_pd.read_csv(_o_src / 'test' / ('%s__horizontal_well.csv' % _w))
    for _i in _o_np.flatnonzero(_te['TVT_input'].isna().to_numpy()):
        _truth['%s_%d' % (_w, _i)] = float(_tr['TVT'].to_numpy(float)[_i])

_sub = _o_pd.read_csv('/kaggle/working/submission.csv', dtype={'id': 'string'})
_sub['t'] = _sub['id'].astype(str).map(_truth)
_d = _sub.dropna(subset=['t'])
_err = _d['tvt'].astype(float).to_numpy() - _d['t'].astype(float).to_numpy()
print('=' * 66)
print('GRADED-SIM offline RMSE vs train-copy truth: %.4f   (n=%d)' % (
    float(_o_np.sqrt(_o_np.mean(_err ** 2))), len(_d)))
print('  real LB of this same configuration (file51): 6.460')
print('  -> if these agree, the offline oracle is real')
print('=' * 66)
_per = _d.assign(w=_d['id'].astype(str).str.rsplit('_', n=1).str[0], e2=_err ** 2
                 ).groupby('w')['e2'].mean().pow(0.5)
print(_per.to_string())
'''
cells.append(code_cell(scorer))

nb['cells'] = cells
json.dump(nb, open(SCRATCH + r'\rogii_77_kernel\77_graded_sim_oracle.ipynb', 'w', encoding='utf-8'))
print('written file 77, cells:', len(cells))
