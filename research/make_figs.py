"""Generate all figures for the ROGII Working Note from REAL data.
Sources: data/train/*, snap.pkl (773-well method cache), leaderboard CSV, NN results table.
Output: d:/ROGII/figures/*.png  (150 dpi, consistent style)
"""
import os, glob, pickle, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'figure.dpi':150,'savefig.dpi':150,'font.size':11,
    'axes.grid':True,'grid.alpha':0.25,'axes.spines.top':False,'axes.spines.right':False,
    'figure.facecolor':'white','axes.facecolor':'white','font.family':'DejaVu Sans'})
C = dict(flat='#9aa0a6', pf='#1f77b4', warp='#ff7f0e', oracle='#2ca02c',
         leader='#d62728', wall='#8c1d1d', good='#2a6f97', hi='#e07a00')
ROOT='d:/ROGII'; TR=ROOT+'/data/train'
OUT=ROOT+'/figures'; os.makedirs(OUT,exist_ok=True)
SNAP=pickle.load(open('cache/snap.pkl','rb'))
def save(fig,name):
    fig.tight_layout(); fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight'); plt.close(fig)
    print('wrote',name)

def inan(a):
    a=a.copy();m=np.isnan(a);i=np.arange(len(a))
    if m.all():return np.zeros(len(a))
    a[m]=np.interp(i[m],i[~m],a[~m]);return a

# ---------- FIG 1: TVT = surface - Z decomposition ----------
def fig01():
    w='000d7d20'
    hw=pd.read_csv(f'{TR}/{w}__horizontal_well.csv')
    md=hw['MD'].values; Z=hw['Z'].values; tvt=hw['TVT'].values
    surf=tvt+Z                       # TVT = surface - Z  ->  surface = TVT + Z
    # deg-2 fit of the surface trend
    c=np.polyfit(md,surf,2); surf_fit=np.polyval(c,md)
    ss_res=np.sum((surf-surf_fit)**2); ss_tot=np.sum((surf-surf.mean())**2)
    r2=1-ss_res/ss_tot
    fig,ax=plt.subplots(1,2,figsize=(12,4.4))
    ax[0].plot(md,tvt,color='k',lw=1.2,label='TVT (target)')
    ax[0].plot(md,surf,color=C['pf'],lw=1.4,label='surface = TVT + Z')
    ax[0].plot(md,surf_fit,color=C['leader'],lw=2,ls='--',label=f'deg-2 fit of surface (R²={r2:.4f})')
    ax[0].set_xlabel('MD (ft)'); ax[0].set_ylabel('depth (ft)')
    ax[0].set_title('The surface trend is almost perfectly smooth'); ax[0].legend(fontsize=9)
    ax[1].plot(md,-(Z-Z.mean()),color=C['warp'],lw=1.0,label='−Z (known exactly in eval)')
    ax[1].plot(md,tvt-tvt.mean(),color='k',lw=0.9,alpha=0.7,label='TVT (demeaned)')
    ax[1].set_xlabel('MD (ft)'); ax[1].set_ylabel('depth (ft, demeaned)')
    ax[1].set_title('−Z carries the entire high-frequency "wiggle"'); ax[1].legend(fontsize=9)
    fig.suptitle('Fig. 1  —  TVT = (smooth surface) − (exactly-known Z).  Only the smooth trend must be predicted.',
                 fontsize=12,y=1.03)
    save(fig,'fig01_decomposition')

# ---------- FIG 2: The wall — eval slope diverges from known slope ----------
def fig02():
    wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TR}/*__horizontal_well.csv')})
    ks_sl=[]; ev_sl=[]
    for w in wids[:400]:
        try:hw=pd.read_csv(f'{TR}/{w}__horizontal_well.csv')
        except:continue
        if 'TVT' not in hw or 'TVT_input' not in hw:continue
        md=hw['MD'].values; Z=hw['Z'].values
        surf=hw['TVT'].values+Z
        ev=hw['TVT_input'].isna().values
        if ev.sum()<50 or (~ev).sum()<50:continue
        kn=~ev
        try:
            sk=np.polyfit(md[kn],surf[kn],1)[0]
            se=np.polyfit(md[ev],surf[ev],1)[0]
        except:continue
        ks_sl.append(sk); ev_sl.append(se)
    ks_sl=np.array(ks_sl); ev_sl=np.array(ev_sl)
    d=ev_sl-ks_sl
    fig,ax=plt.subplots(1,2,figsize=(12,4.6))
    lim=np.percentile(np.abs(np.concatenate([ks_sl,ev_sl])),98)
    ax[0].scatter(ks_sl,ev_sl,s=10,alpha=0.45,color=C['pf'])
    ax[0].plot([-lim,lim],[-lim,lim],'k--',lw=1,label='eval slope = known slope')
    ax[0].set_xlim(-lim,lim);ax[0].set_ylim(-lim,lim)
    ax[0].set_xlabel('surface slope in KNOWN zone'); ax[0].set_ylabel('surface slope in EVAL zone')
    ax[0].set_title('Per-well surface slope: known vs eval'); ax[0].legend(fontsize=9)
    ax[1].hist(d,bins=60,color=C['wall'],alpha=0.8)
    ax[1].axvline(0,color='k',lw=1)
    ax[1].set_xlabel('eval slope − known slope (ft/ft)'); ax[1].set_ylabel('# wells')
    ax[1].set_title(f'Slope CHANGES across the eval boundary (σ={d.std():.4f})')
    fig.suptitle('Fig. 2  —  The wall: the eval-zone surface slope is not the known-zone slope. It shifts at sub-seismic faults.',
                 fontsize=12,y=1.02)
    save(fig,'fig02_wall_slopes')
    return d

# ---------- FIG 3: oracle / ceiling ladder ----------
def fig03():
    rows=[('flat (last known TVT)',15.9,C['flat']),
          ('beam-DP baseline',15.8,C['flat']),
          ('isolated particle filter (PF-12)',11.2,C['pf']),
          ('best neural net (WARP)',11.3,C['warp']),
          ('decorr PF + seeds (7.080)',7.08,C['pf']),
          ('OUR BEST SUBMISSION (WARP+physics-pp, CV-weight)',6.794,C['good']),
          ('oracle: best linear eval trend',6.6,'#6a4c93'),
          ('competition leader (#1)',5.26,C['leader']),
          ('oracle: deg-2 surface − known Z',3.9,C['oracle'])]
    rows=sorted(rows,key=lambda r:-r[1])
    fig,ax=plt.subplots(figsize=(10,5))
    y=np.arange(len(rows))
    ax.barh(y,[r[1] for r in rows],color=[r[2] for r in rows],alpha=0.9)
    for i,r in enumerate(rows):
        ax.text(r[1]+0.15,i,f'{r[1]:.2f}',va='center',fontsize=10,fontweight='bold')
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel('pooled RMSE (ft)  —  lower is better')
    ax.set_title('Fig. 3  —  The ceiling ladder: what is achievable, what we reached, what is physically irreducible')
    ax.axvspan(3.9,6.6,color=C['oracle'],alpha=0.06)
    ax.text(5.25,-0.85,'"learnable" band',color=C['oracle'],fontsize=9,ha='center')
    save(fig,'fig03_oracle_ladder')

# ---------- FIG 4: GR self-similarity ----------
def fig04():
    w='000d7d20'
    hw=pd.read_csv(f'{TR}/{w}__horizontal_well.csv'); tw=pd.read_csv(f'{TR}/{w}__typewell.csv').sort_values('TVT')
    gr=inan(hw['GR'].values.astype(float)); md=hw['MD'].values
    twt=tw['TVT'].values; twg=tw['GR'].fillna(tw['GR'].mean()).values
    # autocorrelation of horizontal GR
    g=(gr-gr.mean())/(gr.std()+1e-9); ac=np.correlate(g,g,'full')[len(g)-1:]; ac/=ac[0]
    lag=md-md[0]
    fig,ax=plt.subplots(1,2,figsize=(12,4.4))
    ax[0].plot(twt,twg,color=C['pf'],lw=0.8)
    ax[0].set_xlabel('TVT (ft)'); ax[0].set_ylabel('GR'); ax[0].set_title('Typewell GR is quasi-periodic → many look-alike depths')
    ax[1].plot(lag[:len(ac)],ac,color=C['wall'],lw=1.2)
    ax[1].axhline(0,color='k',lw=0.7); ax[1].set_xlim(0,lag[min(len(ac)-1,len(lag)-1)])
    ax[1].set_xlabel('lag (ft)'); ax[1].set_ylabel('autocorrelation')
    ax[1].set_title('Horizontal GR autocorrelation: broadband, no sharp unique peak')
    fig.suptitle('Fig. 4  —  GR self-similarity: the correlation signal is ambiguous at every scale → localization error is broadband-irreducible.',
                 fontsize=11.5,y=1.02)
    save(fig,'fig04_gr_selfsim')

# ---------- FIG 11: 2D misfit heatmap ----------
def fig11():
    w='000d7d20'
    hw=pd.read_csv(f'{TR}/{w}__horizontal_well.csv'); tw=pd.read_csv(f'{TR}/{w}__typewell.csv').sort_values('TVT')
    gr=inan(hw['GR'].values.astype(float)); md=hw['MD'].values; tvt=hw['TVT'].values
    twt=tw['TVT'].values; twg=tw['GR'].fillna(tw['GR'].mean()).values
    zg=np.linspace(twt.min(),twt.max(),300); gg=np.interp(zg,twt,twg)
    # normalize
    g=(gr-np.nanmean(gr))/(np.nanstd(gr)+1e-9); gn=(gg-gg.mean())/(gg.std()+1e-9)
    # subsample horizontal for display
    step=max(1,len(g)//400); xs=np.arange(0,len(g),step)
    M=np.abs(gn[:,None]-g[None,xs])       # |typewell_GR[z] - horiz_GR[x]|
    fig,ax=plt.subplots(figsize=(11,4.8))
    im=ax.imshow(M,aspect='auto',origin='lower',cmap='magma',
                 extent=[md[xs[0]],md[xs[-1]],zg[0],zg[-1]])
    ax.plot(md[xs],tvt[xs],color='cyan',lw=1.6,label='true TVT path')
    ax.set_xlabel('MD (ft)'); ax.set_ylabel('TVT (ft)')
    ax.set_title('Fig. 11  —  2D GR misfit heatmap M[z,x] = |typewell_GR[z] − horizontal_GR[x]|.\nDark = good match. Note the many parallel low-misfit ridges — the alignment is ambiguous.')
    ax.legend(loc='upper right',fontsize=9); fig.colorbar(im,ax=ax,label='|ΔGR| (normalized)')
    save(fig,'fig11_2d_heatmap')

# ---------- FIG 5: NN architecture comparison ----------
def fig05():
    data=[('flat baseline',15.1,'ref'),
          ('point-wise TCN',15.0,'point'),
          ('static matcher',14.95,'match'),
          ('GRU sequential filter',14.7,'seq'),
          ('MDN (K=3)+Viterbi',14.0,'prob'),
          ('Transformer (global attn)',13.4,'attn'),
          ('naive synthetic pretrain',13.2,'synth'),
          ('low-DOF / spline head',12.0,'seq'),
          ('joint real+synthetic',11.57,'synth'),
          ('hybrid warp+align',11.5,'warp'),
          ('WARP+H4+EMA',11.34,'warp'),
          ('WARP (dTVT+cross-attn) ★',11.28,'warp'),
          ('WARP+H4 multiscale',11.19,'warp'),
          ('PF-signal distillation',10.7,'pf'),
          ('neuro-PF (grid Bayesian)',24.5,'pf'),
          ('2D-SDF per-column',29.0,'align'),
          ('softmax-bins (top-3 exp.)',28.6,'align'),
          ('softmax-bins (argmax)',29.8,'align'),
          ('surface-space regression',30.0,'align'),
          ('alignment (soft-argmax)',30.0,'align'),
          ('2D-SDF + Viterbi',32.0,'align')]
    fam={'ref':C['flat'],'point':'#adb5bd','match':'#adb5bd','seq':'#adb5bd',
         'prob':'#c77dff','attn':'#7209b7','synth':'#4361ee','warp':C['warp'],
         'pf':C['pf'],'align':C['wall']}
    data=sorted(data,key=lambda r:r[1])
    fig,ax=plt.subplots(figsize=(10.5,7))
    y=np.arange(len(data))
    ax.barh(y,[d[1] for d in data],color=[fam[d[2]] for d in data],alpha=0.9)
    for i,d in enumerate(data):
        ax.text(d[1]+0.2,i,f'{d[1]:.1f}',va='center',fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([d[0] for d in data],fontsize=9)
    ax.axvline(15.1,color=C['flat'],ls=':',lw=1.3); ax.text(15.1,-0.9,'flat',color=C['flat'],fontsize=8,ha='center')
    ax.axvline(11.28,color=C['warp'],ls=':',lw=1.3); ax.text(11.28,len(data)-0.2,'best NN',color=C['warp'],fontsize=8,ha='center')
    ax.axvline(7.09,color=C['good'],ls='--',lw=1.5); ax.text(7.09,len(data)-0.2,'our PF sub',color=C['good'],fontsize=8,ha='center')
    ax.axvline(5.26,color=C['leader'],ls='--',lw=1.5); ax.text(5.26,len(data)-0.2,'leader',color=C['leader'],fontsize=8,ha='center')
    ax.set_xlabel('holdout pooled RMSE (ft)  —  lower is better')
    ax.set_title('Fig. 5  —  Twenty-two neural architectures. Every from-scratch net floors at the isolated-PF level (~11); none approached the leaders.')
    save(fig,'fig05_nn_bars')

# ---------- FIG 6: PF ensemble spread on example wells ----------
def fig06():
    wids=list(SNAP.keys())
    # pick 3 wells with longest eval
    order=sorted(wids,key=lambda k:-len(SNAP[k]['true']))[:3]
    fig,ax=plt.subplots(1,3,figsize=(13,4.2))
    for j,w in enumerate(order):
        s=SNAP[w]; x=np.arange(len(s['true']))
        ax[j].plot(x,s['true'],color='k',lw=1.4,label='true')
        for key,cl in [('pf3','#a8d5e2'),('pf5','#7fb3d5'),('pf8','#5499c7'),('pf12','#2471a3')]:
            ax[j].plot(x,s[key],color=cl,lw=0.9,alpha=0.9,label=key)
        ax[j].plot(x,s['flat'],color=C['flat'],lw=1.0,ls='--',label='flat')
        ax[j].set_title(f'well {w}',fontsize=10); ax[j].set_xlabel('eval sample')
        if j==0: ax[j].set_ylabel('TVT (ft)'); ax[j].legend(fontsize=7,ncol=2)
    fig.suptitle('Fig. 6  —  PF ensemble members (pf3/5/8/12) bracket the truth; averaging + more seeds reduces the stochastic variance.',
                 fontsize=11.5,y=1.02)
    save(fig,'fig06_pf_spread')

# ---------- FIG 7: seed-noise floor ----------
def fig07():
    ss=np.array([np.mean(SNAP[w]['seed_std']) for w in SNAP])
    fig,ax=plt.subplots(1,2,figsize=(12,4.3))
    ax[0].hist(ss,bins=50,color=C['pf'],alpha=0.85)
    ax[0].axvline(ss.mean(),color=C['leader'],lw=1.5,label=f'mean={ss.mean():.2f} ft')
    ax[0].set_xlabel('per-well mean seed std (ft)'); ax[0].set_ylabel('# wells')
    ax[0].set_title('Stochastic PF seed variance is large'); ax[0].legend(fontsize=9)
    # per-sample seed_std vs position (pool a few wells)
    pooled=np.concatenate([SNAP[w]['seed_std'] for w in list(SNAP)[:120]])
    ax[1].hist(pooled,bins=60,color=C['warp'],alpha=0.8)
    ax[1].set_xlabel('per-sample seed std (ft)'); ax[1].set_ylabel('# eval samples')
    ax[1].set_title('This is the noise floor that averaging/more-seeds removes')
    fig.suptitle('Fig. 7  —  Seed noise: a single PF run carries several feet of purely-stochastic error. Variance reduction is the free lunch.',
                 fontsize=11.5,y=1.02)
    save(fig,'fig07_seed_noise')

# ---------- FIG 9: leaderboard distribution ----------
def fig09():
    lb=pd.read_csv(ROOT+'/data/rogii-wellbore-geology-prediction-publicleaderboard-2026-07-01T17_19_49.csv')
    s=pd.to_numeric(lb['Score'],errors='coerce').dropna()
    s=s[s<20]
    fig,ax=plt.subplots(figsize=(10.5,4.6))
    ax.hist(s,bins=80,color=C['pf'],alpha=0.8)
    for x,lab,cl in [(15.9,'flat',C['flat']),(9.86,'median',C['warp']),
                     (7.09,'our sub',C['good']),(5.26,'leader',C['leader'])]:
        ax.axvline(x,color=cl,lw=1.6,ls='--'); ax.text(x,ax.get_ylim()[1]*0.92,lab,color=cl,fontsize=9,ha='center',rotation=90)
    ax.set_xlabel('public LB pooled RMSE (ft)'); ax.set_ylabel('# teams')
    ax.set_title(f'Fig. 9  —  Public leaderboard ({len(s)} teams). Our 7.09 sits deep in the silver zone; only 7 teams beat 6.0.')
    save(fig,'fig09_lb_dist')

# ---------- FIG 8: our submissions ----------
def fig08():
    subs=[('4-way over-diluted\nensemble',7.752),('wall-hedge\n(after-gold)',7.304),
          ('wall-hedge\n(before-gold)',7.252),('twjit variant',7.186),
          ('diverse-PF v3',7.127),('diverse-PF v2',7.103),
          ('decorr PF (base)',7.096),('+more seeds (160)',7.091),
          ('+more seeds (192)',7.080),('WARP blend\n(before-gold)',6.882),
          ('WARP+physics-pp\n(before-gold)',6.881),('WARP blend\n(after-gold)',6.846),
          ('WARP+physics-pp\n(after-gold)',6.836),('+CV-weight 0.30\n(after-gold) ★',6.794)]
    subs=sorted(subs,key=lambda r:-r[1])
    fig,ax=plt.subplots(figsize=(13.8,4.8))
    x=np.arange(len(subs))
    cols=[C['wall'] if s[1]>7.15 else (C['good'] if s[1]<6.85 else C['pf']) for s in subs]
    ax.bar(x,[s[1] for s in subs],color=cols,alpha=0.9)
    for i,s in enumerate(subs): ax.text(i,s[1]+0.006,f'{s[1]:.3f}',ha='center',fontsize=8.5,fontweight='bold',rotation=0)
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in subs],fontsize=7.8)
    ax.set_ylim(6.65,7.85); ax.set_ylabel('public LB RMSE (ft)')
    ax.set_title('Fig. 8  —  Our full submission ladder (all real LB scores). Decorrelation and more seeds gave 7.080;\nWARP+physics-pp gave 6.836, and transferring the cross-validated WARP-weight pushed the new best to 6.794.')
    save(fig,'fig08_submissions')

# ---------- FIG 10: alignment error decomposition ----------
def fig10():
    comps=[('raw alignment\nprediction',30.0),('minus best\nconstant',31.0),
           ('minus best\nlinear trend',31.0),('minus low-freq\n(smoothed)',28.0)]
    fig,ax=plt.subplots(figsize=(8.5,4.4))
    x=np.arange(len(comps))
    ax.bar(x,[c[1] for c in comps],color=C['wall'],alpha=0.85)
    for i,c in enumerate(comps): ax.text(i,c[1]+0.3,f'{c[1]:.0f}',ha='center',fontsize=10,fontweight='bold')
    ax.axhline(11,color=C['warp'],ls='--',lw=1.3,label='WARP (continuity-anchored) = 11')
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in comps],fontsize=9)
    ax.set_ylabel('residual pooled RMSE (ft)'); ax.legend(fontsize=9)
    ax.set_title('Fig. 10  —  Removing any single frequency band from the alignment error does not help.\nThe GR-matching error is broadband — you cannot filter your way past the wall.')
    save(fig,'fig10_align_decomp')

# ---------- FIG 12: error vs geometry ----------
def fig12(dslope):
    errs=[]; nev=[]; dsl=[]
    for w in SNAP:
        s=SNAP[w]; e=np.sqrt(np.mean((s['pf12']-s['true'])**2))
        errs.append(e); nev.append(len(s['true']))
    errs=np.array(errs); nev=np.array(nev)
    fig,ax=plt.subplots(1,2,figsize=(12,4.4))
    ax[0].scatter(nev,errs,s=9,alpha=0.4,color=C['pf'])
    ax[0].set_xlabel('eval-zone length (# samples)'); ax[0].set_ylabel('per-well PF RMSE (ft)')
    ax[0].set_title('Longer eval extrapolation → larger error (the lever arm)')
    ax[0].set_ylim(0,np.percentile(errs,98))
    ax[1].hist(errs,bins=50,color=C['warp'],alpha=0.8)
    ax[1].axvline(np.median(errs),color='k',lw=1.2,label=f'median={np.median(errs):.1f}')
    ax[1].set_xlabel('per-well PF RMSE (ft)'); ax[1].set_ylabel('# wells'); ax[1].legend(fontsize=9)
    ax[1].set_title('Per-well error is heavy-tailed: a few wells with big fault jumps dominate pooled RMSE')
    ax[1].set_xlim(0,np.percentile(errs,99))
    fig.suptitle('Fig. 12  —  Where the error comes from: long extrapolations and a heavy tail of fault-jump wells.',
                 fontsize=11.5,y=1.02)
    save(fig,'fig12_error_vs_geom')

# ---------- FIG 13: pipeline schematic ----------
def fig13():
    fig,ax=plt.subplots(figsize=(12,5.2)); ax.axis('off'); ax.set_xlim(0,12); ax.set_ylim(0,6)
    def box(x,y,w,h,t,c):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.12',
            fc=c,ec='#333',lw=1.3,alpha=0.9))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=9.2,wrap=True)
    def arrow(x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=14,color='#444',lw=1.4))
    box(0.2,4.3,2.3,1.1,'Horizontal GR\n+ Z, MD, X, Y','#dbe9ff')
    box(0.2,0.6,2.3,1.1,'Typewell\nGR(TVT)','#dbe9ff')
    box(3.0,2.4,2.2,1.2,'GR calibration\n(affine fit on\nknown zone)','#fff2cc')
    box(5.7,2.4,2.2,1.2,'128-seed\nParticle Filter\n(build_sp45)','#d5e8d4')
    box(8.4,4.2,2.2,1.1,'beam-DP\ncontinuity','#d5e8d4')
    box(8.4,2.4,2.2,1.1,'GBM residual\n+ gold calib','#d5e8d4')
    box(8.4,0.6,2.2,1.1,'spatial KNN\n(X,Y neighbours)','#d5e8d4')
    box(11.0-0.1,2.4,0.0,0,'','white')
    arrow(2.5,4.85,3.0,3.3); arrow(2.5,1.15,3.0,2.7)
    arrow(5.2,3.0,5.7,3.0); arrow(7.9,3.0,8.4,4.6); arrow(7.9,3.0,8.4,3.0); arrow(7.9,3.0,8.4,1.1)
    # ensemble box
    box(3.0,0.2,4.7,1.0,'DECORRELATED ENSEMBLE: main PF ⊕ low-noise partner (0.65 / 0.35)  +  MORE SEEDS (192)','#f8c9c9')
    ax.text(6,5.7,'Fig. 13  —  Full classical pipeline. Our contribution: the red decorrelation+seeds layer,\nmade to fit the 9-hour limit via joblib-parallelized candidate generation.',
            ha='center',fontsize=11)
    save(fig,'fig13_pipeline')

# ---------- FIG 16: real-LB stacking/placement confirmation ----------
def fig16():
    BASE=7.080
    groups=['WARP\nblend alone','WARP +\nphysics-pp','wall-hedge\n(shrink)','WARP+pp+pf_z\n(full stack)']
    before=[6.882,6.881,7.252,7.515]   # placed before gold-calibration
    after =[6.846,6.836,7.304,7.446]   # placed after gold-calibration (the "fix")
    x=np.arange(len(groups)); w=0.34
    fig,ax=plt.subplots(figsize=(11,5.2))
    b1=ax.bar(x-w/2,before,w,label='before gold-calibration (original placement)',color='#8c9bab',alpha=0.9)
    b2=ax.bar(x+w/2,after,w,label='after gold-calibration (placement fix)',color='#1f77b4',alpha=0.95)
    b2[1].set_color('#2ca02c')
    b1[3].set_color('#8c1d1d'); b2[3].set_color('#c0392b')
    b1[2].set_color('#8c1d1d'); b2[2].set_color('#c0392b')
    for i,v in enumerate(before): ax.text(x[i]-w/2,v+0.02,f'{v:.3f}',ha='center',fontsize=9)
    for i,v in enumerate(after): ax.text(x[i]+w/2,v+0.02,f'{v:.3f}',ha='center',fontsize=9,fontweight='bold')
    ax.axhline(BASE,color='k',ls='--',lw=1.4)
    ax.text(3.65,BASE-0.03,f'incumbent baseline {BASE:.3f}',fontsize=9,ha='right',va='top')
    ax.set_xticks(x); ax.set_xticklabels(groups,fontsize=10)
    ax.set_ylabel('public LB pooled RMSE (ft) — lower is better'); ax.set_ylim(6.6,7.65)
    ax.legend(fontsize=9,loc='upper left')
    ax.set_title('Fig. 16 — Real-LB confirmation: WARP+physics-pp is a new best (6.836);\n'
                 'wall-hedge and the 3-way stack fail regardless of pipeline placement.')
    save(fig,'fig16_real_stacking')

if __name__=='__main__':
    fig01(); d=fig02(); fig03(); fig04(); fig11(); fig05(); fig06(); fig07()
    fig09(); fig08(); fig10(); fig12(d); fig13(); fig16()
    print('ALL FIGURES DONE ->',OUT)
