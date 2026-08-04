"""Build files 79-83: five independent levers, four of them at the normal runtime budget.

file78 showed the seed-branch hedge only fires because the PF is under-sampled: at 512 seeds the minority
mode's mass falls to 0.236, under the mechanism's own _BH_MIN_MASS=0.25, and it declines by itself. But
raising the budget costs 38 minutes of preview and the submission-scoring pass runs far slower than
preview (file73 timed out at the normal budget), so the finding is better delivered by moving the
threshold instead of the sample size -- identical outcome, zero extra runtime.

  79  _BH_MIN_MASS 0.25 -> 0.30      hedge declines (mass is 0.279 at the normal budget); no runtime cost
  80  visible-prefix profile         'balanced' -> 'conservative'. Scored offline in the graded-sim run,
                                     conservative 4.031 vs balanced 4.834 -- the profile we ship was the
                                     worst of the three. Offline truth is not predictive of the LB, but a
                                     0.8 gap on an untouched knob is worth one slot.
  81  hedge aimed at the LOW mode    every fresh piece of evidence says the low mode is right (better
                                     sampling puts 0.764 there; the visible truth sits there) yet the
                                     mechanism, and the whole public lineage, shifts UP toward the midpoint
  82  sp45_blend_weight 0.60 -> 0.70 never tuned; the pipeline's own variants improve monotonically toward
                                     0.60 offline, so the optimum may lie past the end of its own sweep
  83  seeds 128 -> 256               half the variance at ~20 min, the compromise between file78's finding
                                     and the timeout risk

All five are built on file51 (6.460) so each differs from a known baseline by one thing.
"""
import json
import os

SCRATCH = r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad'
BASE = r'D:\ROGII\51_gru_theirbase_blend.ipynb'

# Q0522 hard-raises when no branch was applied; that is the intended outcome for 79/81, so make the
# stage optional exactly as file78 did rather than fatal.
def make_q0522_optional(nb):
    i = next(i for i, c in enumerate(nb['cells']) if "_EX_LABEL = 'Q0522'" in ''.join(c['source']))
    body = ''.join(nb['cells'][i]['source'])
    assert '"""' not in body and "'''" not in body
    guarded = ('try:\n'
               + '\n'.join(('    ' + l) if l.strip() else l for l in body.split('\n')) + '\n'
               "except Exception as _q_exc:\n"
               "    print('Q0522 skipped:', _q_exc)\n")
    nb['cells'][i]['source'] = guarded.splitlines(keepends=True)


VARIANTS = [
    ('79', '79_minmass_030', [('_BH_MIN_MASS = 0.25', '_BH_MIN_MASS = 0.30')], True),
    ('80', '80_vp_conservative', [("SUBMISSION_PROFILE = 'vp_balanced_modelpkg_005'",
                                   "SUBMISSION_PROFILE = 'vp_conservative_final'")], False),
    ('81', '81_hedge_to_low_mode', [('_target = 0.5 * (_low + _high)', '_target = _low')], False),
    ('82', '82_sp45_w070', [('sp45_blend_weight=0.60', 'sp45_blend_weight=0.70')], False),
    ('83', '83_seeds_256', [('SP45_SELECTOR_N_SEEDS = 128', 'SP45_SELECTOR_N_SEEDS = 256'),
                            ('PF_SEEDS = 128', 'PF_SEEDS = 256')], True),
]

for num, name, repl, optional_q in VARIANTS:
    nb = json.load(open(BASE, encoding='utf-8'))
    counts = {o: 0 for o, _ in repl}
    for c in nb['cells']:
        s = ''.join(c['source'])
        orig = s
        for old, new in repl:
            if old in s:
                s = s.replace(old, new)
                counts[old] += 1
        if s != orig:
            c['source'] = s.splitlines(keepends=True)
    missing = [o for o, n in counts.items() if n == 0]
    if missing:
        print('  !! %s: NOT FOUND -> %s' % (name, missing))
        continue
    if optional_q:
        make_q0522_optional(nb)
    outdir = os.path.join(SCRATCH, 'rogii_%s_kernel' % num)
    os.makedirs(outdir, exist_ok=True)
    json.dump(nb, open(os.path.join(outdir, name + '.ipynb'), 'w', encoding='utf-8'))
    print('  OK %s  %s' % (name, {o: n for o, n in counts.items()}))
