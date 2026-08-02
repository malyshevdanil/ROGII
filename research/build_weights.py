"""Build files 68/69: file51 with beam2+neighbor blend weights scaled up.

Harness evidence (12 leak-free held-out wells): beam2 and neighbor are the only two components where
the harness and the real leaderboard AGREE in direction (both say they help), and their weights were
inherited from our own pipeline and never tuned for this base. Scaling both together is monotonically
better on the harness up to a broad optimum near x6:
    x1 (production) 8.1402 | x2 8.0027 | x3 7.8964 | x4 7.8225 | x6 7.7752 | x8 7.8633
Leave-one-well-out picks m in 5.5-6.5 in 9 of 12 folds, and a fixed multiplier beats production on
7-8 of 12 held-out wells -- moderate confidence, not the 12/12 the WARP/GRU verdict had (and that one
failed to transfer). So submit two points, a conservative x3 and the harness optimum x6, and let the
real leaderboard map the response curve rather than trusting the harness optimum outright.

Built on file51 (6.460) so the ONLY difference from a known baseline is these two weights.
"""
import json

SCRATCH = r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad'
BASE = r'D:\ROGII\51_gru_theirbase_blend.ipynb'

for mult, out in [(3.0, r'\rogii_68_kernel\68_beam2neigh_x3.ipynb'),
                  (6.0, r'\rogii_69_kernel\69_beam2neigh_x6.ipynb')]:
    nb = json.load(open(BASE, encoding='utf-8'))
    b2, ng = round(0.0109 * mult, 6), round(0.0316 * mult, 6)
    hits = 0
    for c in nb['cells']:
        s = ''.join(c['source'])
        if 'PORTABLE_BEAM2_BLEND_W = 0.0109' in s:
            s = s.replace('PORTABLE_BEAM2_BLEND_W = 0.0109',
                          'PORTABLE_BEAM2_BLEND_W = %s' % b2)
            hits += 1
        if 'PORTABLE_NEIGHBOR_BLEND_W = 0.0316' in s:
            s = s.replace('PORTABLE_NEIGHBOR_BLEND_W = 0.0316',
                          'PORTABLE_NEIGHBOR_BLEND_W = %s' % ng)
            hits += 1
        c['source'] = s.splitlines(keepends=True)
    assert hits == 2, 'expected to patch both weights, patched %d' % hits
    json.dump(nb, open(SCRATCH + out, 'w', encoding='utf-8'))
    print('x%g -> beam2=%s neighbor=%s' % (mult, b2, ng))
