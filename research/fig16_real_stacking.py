"""Fig 16 -- real-LB confirmation of the stacking/placement results (all numbers are actual
Kaggle public-LB scores from real submissions, not proxy/offline numbers)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi':150,'savefig.dpi':150,'font.size':11,
    'axes.grid':True,'grid.alpha':0.25,'axes.spines.top':False,'axes.spines.right':False,
    'figure.facecolor':'white','axes.facecolor':'white','font.family':'DejaVu Sans'})

OUT = 'd:/ROGII/figures'
BASE = 7.080

groups = ['WARP\nblend alone', 'WARP +\nphysics-pp', 'wall-hedge\n(shrink)', 'WARP+pp+pf_z\n(full stack)']
before = [6.882, 6.881, 7.252, 7.515]   # placed before gold-calibration
after  = [6.846, 6.836, 7.304, 7.446]   # placed after gold-calibration (the "fix")

x = np.arange(len(groups))
w = 0.34

fig, ax = plt.subplots(figsize=(11, 5.2))
b1 = ax.bar(x - w/2, before, w, label='before gold-calibration (original placement)', color='#8c9bab', alpha=0.9)
b2 = ax.bar(x + w/2, after,  w, label='after gold-calibration (placement fix)', color='#1f77b4', alpha=0.95)

# highlight the new best and the two failures
b2[1].set_color('#2ca02c')   # WARP+pp after-gold = new best
b1[3].set_color('#8c1d1d'); b2[3].set_color('#c0392b')
b1[2].set_color('#8c1d1d'); b2[2].set_color('#c0392b')

for i, v in enumerate(before):
    ax.text(x[i]-w/2, v+0.02, f'{v:.3f}', ha='center', fontsize=9)
for i, v in enumerate(after):
    ax.text(x[i]+w/2, v+0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

ax.axhline(BASE, color='k', ls='--', lw=1.4)
ax.text(3.65, BASE-0.03, f'incumbent baseline {BASE:.3f}', fontsize=9, ha='right', va='top')

ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10)
ax.set_ylabel('public LB pooled RMSE (ft) — lower is better')
ax.set_ylim(6.6, 7.65)
ax.legend(fontsize=9, loc='upper left')
ax.set_title('Fig. 16 — Real-LB confirmation: WARP+physics-pp is a new best (6.836);\n'
             'wall-hedge and the 3-way stack fail regardless of pipeline placement.')
fig.tight_layout()
fig.savefig(f'{OUT}/fig16_real_stacking.png', bbox_inches='tight')
print('wrote fig16_real_stacking')
