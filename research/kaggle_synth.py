"""ROGII — SYNTHETIC PRETRAINING (leaders' direction 4, the key enabler vs overfitting on 773 wells).
Generate synthetic horizontal wells from the real typewells (true TVT known), pretrain WARP on them,
fine-tune on the 773 real wells, eval on the real whole-well holdout. Point-head WARP first to isolate
the synthetic-pretrain effect (beat WARP's 11?). Kaggle: GPU T4 x2, Internet off, competition dataset."""
import numpy as np, pandas as pd, glob, os, time, copy, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0); DEV='cuda' if torch.cuda.is_available() else 'cpu'
CFG=dict(d=96, KSTEPS=60, EVSTEPS=360, TWGRID=256, drop=0.15, wd=3e-4, lr=1.5e-3, bs=32,
         pretrain_steps=4000, finetune_epochs=60, maxstep=1.5, smooth=0.05,
         aug_caljit=0.12, aug_noise=0.07, aug_warp=0.35, w_known=0.5, ema_decay=0.995, n_val=160, seed=42)
TG=CFG['TWGRID']; KSTEPS=CFG['KSTEPS']; EVSTEPS=CFG['EVSTEPS']; L=KSTEPS+EVSTEPS
c=glob.glob('/kaggle/input/**/*__horizontal_well.csv',recursive=True); _t=[p for p in c if 'train' in p.lower()]; c=_t if _t else c
TRAIN_DIR=os.path.dirname(c[0]) if c else 'data/train'; print('DEV=%s data=%s'%(DEV,TRAIN_DIR),flush=True)
def inan(a):
    a=a.copy();m=np.isnan(a);i=np.arange(len(a))
    if m.all():return np.zeros(len(a))
    a[m]=np.interp(i[m],i[~m],a[~m]);return a
# ---- typewell bank (for the generator) + real wells (build_well) ----
def load_tw(wid):
    tw=pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    tt=tw['TVT'].values.astype(float); tg=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    if len(tt)<30: return None
    g_tvt=np.linspace(tt.min(),tt.max(),TG); g_gr=np.interp(g_tvt,tt,tg)
    return dict(g_tvt=g_tvt.astype(np.float32),g_gr=g_gr.astype(np.float32),gm=float(g_gr.mean()),gs=float(g_gr.std()+1e-6))
def build_real(wid):
    hw=pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv'); tw=pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tt=tw['TVT'].values.astype(float); tg=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    if len(tt)<10: return None
    kn=hw[hw['TVT_input'].notna()]
    if len(kn)<20 or hw['TVT_input'].isna().sum()<20: return None
    last=kn.iloc[-1]; last_tvt=float(last['TVT_input'])
    n=len(hw); gr=inan(hw['GR'].values.astype(float)); kg=kn['GR'].interpolate().bfill().ffill().values.astype(float)
    twk=np.interp(kn['TVT_input'].values,tt,tg); v=np.isfinite(kg)&np.isfinite(twk); a,b=(np.polyfit(kg[v],twk[v],1) if v.sum()>=20 else (1.,0.)); cg=gr*a+b
    Z=hw['Z'].values.astype(float); MD=hw['MD'].values.astype(float); mdd=np.gradient(MD); mdd[mdd==0]=1; dz=np.gradient(Z)/mdd
    ev=hw['TVT_input'].isna().values.astype(float); ei=np.where(ev>0.5)[0]
    if len(ei)<5: return None
    e0=ei[0]; ks=np.arange(max(0,e0-400),e0)
    if len(ks)<5: return None
    g_tvt=np.linspace(tt.min(),tt.max(),TG); g_gr=np.interp(g_tvt,tt,tg); gm,gs=float(g_gr.mean()),float(g_gr.std()+1e-6); caln=(cg-gm)/gs
    true=hw['TVT'].values.astype(float)
    kd=np.linspace(ks[0],ks[-1],KSTEPS); ed=np.linspace(e0,n-1,EVSTEPS); dst=np.concatenate([kd,ed]); src=np.arange(n)
    R=lambda x: np.interp(dst,src,x).astype(np.float32); evr=np.concatenate([np.zeros(KSTEPS),np.ones(EVSTEPS)]).astype(np.float32)
    return dict(H=np.stack([R(caln),R(np.gradient(caln)),R(dz)]).astype(np.float32),
                gn=((g_gr-gm)/gs).astype(np.float32), tvt=R(true), ev=evr, last_tvt=last_tvt)
print('loading real + typewell bank...',flush=True); t0=time.time()
wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
TWBANK=[x for x in (load_tw(w) for w in wids) if x]
DATA=[b for b in (build_real(w) for w in wids) if b is not None]
d2=np.concatenate([w['H'][2] for w in DATA]); DZM,DZS=d2.mean(),d2.std()+1e-6
for w in DATA: w['H'][2]=(w['H'][2]-DZM)/DZS
rng=np.random.default_rng(CFG['seed']); idx=np.arange(len(DATA)); rng.shuffle(idx)
VA=[DATA[i] for i in idx[:CFG['n_val']]]; TR=[DATA[i] for i in idx[CFG['n_val']:]]
print('typewells=%d real=%d (%d/%d) %.0fs'%(len(TWBANK),len(DATA),len(TR),len(VA),time.time()-t0),flush=True)
def prmse(s): return float(np.sqrt(np.mean(np.concatenate(s))))
print('BASELINE flat=%.3f (WARP~11)'%prmse([(w['last_tvt']-w['tvt'][w['ev']>0.5])**2 for w in VA]),flush=True)

# ---- SYNTHETIC GENERATOR: sample a random TVT trajectory, GR=typewell_GR(TVT)+noise ----
def gen_synth(B):
    Hs=np.zeros((B,3,L),np.float32); GN=np.zeros((B,TG),np.float32); LT=np.zeros(B,np.float32); TVT=np.zeros((B,L),np.float32)
    for b in range(B):
        tw=TWBANK[np.random.randint(len(TWBANK))]; g_tvt=tw['g_tvt']; g_gr=tw['g_gr']; gm,gs=tw['gm'],tw['gs']
        tlo,thi=g_tvt[0],g_tvt[-1]; span=thi-tlo
        start=np.random.uniform(tlo+0.15*span,thi-0.15*span)
        # dip: piecewise-constant segments (few faults) + slow drift + rare bimodal jumps
        nseg=np.random.randint(1,5); bps=np.sort(np.random.choice(np.arange(1,L),nseg,replace=False))
        dip=np.zeros(L); seg_dips=np.random.randn(nseg+1)*np.random.uniform(0.02,0.08)
        si=0
        for i in range(L):
            if si<len(bps) and i>=bps[si]: si+=1
            dip[i]=seg_dips[si]
        step=dip+np.random.randn(L)*0.02                            # per-step TVT change
        traj=start+np.cumsum(step-step.mean())                      # de-mean so it drifts around start
        if np.random.rand()<0.4:                                    # rare bimodal ±15ft shift on a stretch
            j=np.random.randint(KSTEPS,L-30); traj[j:]+=np.random.choice([-1,1])*np.random.uniform(8,18)
        traj=np.clip(traj,tlo+2,thi-2)
        gr=np.interp(traj,g_tvt,g_gr); caln=(gr-gm)/gs+np.random.randn(L)*np.random.uniform(0.35,0.8)   # WEAK/ambiguous GR (force continuity-learning, match real self-similarity)
        sca=1+np.random.randn()*0.1; sh=np.random.randn()*0.1; caln=caln*sca+sh                          # random miscalibration
        surf=np.linspace(0,np.random.randn()*20,L)+np.cumsum(np.random.randn(L)*0.01)                    # smooth surface
        Z=surf-traj; dz=(np.gradient(Z)-DZM)/DZS
        Hs[b]=np.stack([caln,np.gradient(caln),dz]); GN[b]=(g_gr-gm)/gs; LT[b]=traj[KSTEPS-1]; TVT[b]=traj
    ev=np.concatenate([np.zeros(KSTEPS),np.ones(EVSTEPS)]).astype(np.float32)
    to=lambda a: torch.tensor(a,dtype=torch.float32,device=DEV)
    return to(Hs),to(GN)[:,None],to(LT),to(TVT),to(np.tile(ev,(B,1)))

class Enc(nn.Module):
    def __init__(s,nin,d,drop):
        super().__init__(); s.inp=nn.Conv1d(nin,d,5,padding=2)
        s.bl=nn.ModuleList([nn.Sequential(nn.Conv1d(d,d,3,padding=dl,dilation=dl),nn.GroupNorm(8,d),nn.GELU(),nn.Dropout(drop),
            nn.Conv1d(d,d,3,padding=dl,dilation=dl),nn.GroupNorm(8,d),nn.GELU()) for dl in (1,2,4,8,16)]); s.o=nn.Conv1d(d,d,1)
    def forward(s,x):
        h=F.gelu(s.inp(x))
        for b in s.bl: h=h+b(h)
        return s.o(h)
class WARP(nn.Module):
    def __init__(s,d,drop):
        super().__init__(); s.he=Enc(3,d,drop); s.te=Enc(1,d,drop); s.q=nn.Conv1d(d,d,1); s.k=nn.Conv1d(d,d,1); s.vv=nn.Conv1d(d,d,1); s.sc=d**-0.5
        s.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1)); s.head[-1].weight.data*=0.01; s.head[-1].bias.data.zero_()
    def forward(s,H,G,lt):
        h=s.he(H); t=s.te(G); ctx=torch.softmax(s.q(h).transpose(1,2)@s.k(t).transpose(1,2).transpose(1,2)*s.sc,2)@s.vv(t).transpose(1,2)
        dt=torch.tanh(s.head(torch.cat([h.transpose(1,2),ctx],2))[...,0])*CFG['maxstep']
        tvt=torch.cumsum(dt,1); return tvt-tvt[:,KSTEPS-1:KSTEPS]+lt[:,None]
def loss_fn(pred,tvt,ev):
    m=ev+CFG['w_known']*(1-ev); l=(F.smooth_l1_loss(pred,tvt,reduction='none',beta=8.0)*m).sum()/m.sum()
    d2=pred[:,2:]-2*pred[:,1:-1]+pred[:,:-2]; return l+CFG['smooth']*(d2*d2*m[:,2:]).sum()/m[:,2:].sum()
def batch_real(ws,train=False):
    H=np.stack([w['H'] for w in ws]).copy(); gn=np.stack([w['gn'][:TG] for w in ws]).copy(); tvt=np.stack([w['tvt'] for w in ws]).copy(); ev=np.stack([w['ev'] for w in ws])
    if train:
        sc=1+np.random.randn(len(ws),1,1)*CFG['aug_caljit']; sh=np.random.randn(len(ws),1,1)*CFG['aug_caljit']
        H[:,0:1]=H[:,0:1]*sc+sh; gn=gn*sc[:,:,0]+sh[:,:,0]; H=H+np.random.randn(*H.shape)*CFG['aug_noise']
        if CFG['aug_warp']>0:
            m=EVSTEPS; src=np.arange(m)
            for bi in range(len(ws)):
                d=np.cumsum(np.abs(1+np.random.randn(m)*CFG['aug_warp'])); d=(d-d[0])/(d[-1]-d[0])*(m-1)
                for ch in range(3): H[bi,ch,KSTEPS:]=np.interp(d,src,H[bi,ch,KSTEPS:])
                tvt[bi,KSTEPS:]=np.interp(d,src,tvt[bi,KSTEPS:])
    to=lambda a: torch.tensor(a,dtype=torch.float32,device=DEV)
    return to(H),to(gn)[:,None],to(np.array([w['last_tvt'] for w in ws])),to(tvt),to(ev)
def evaluate(net):
    net.eval(); e=[]
    with torch.no_grad():
        for i in range(0,len(VA),CFG['bs']):
            H,G,lt,tvt,ev=batch_real(VA[i:i+CFG['bs']]); p=net(H,G,lt); e.append(((p-tvt)[ev>0.5]).cpu().numpy())
    net.train(); return float(np.sqrt(np.mean(np.concatenate(e)**2)))
net=WARP(CFG['d'],CFG['drop']).to(DEV)
EP=CFG['finetune_epochs']; SYNW=CFG.get('synw',0.5)   # weight of synthetic loss (as augmentation/regularizer)
opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG['lr'],total_steps=EP*((len(TR)+CFG['bs']-1)//CFG['bs']))
ema=copy.deepcopy(net); [p.requires_grad_(False) for p in ema.parameters()]; best=(99,0); order=np.arange(len(TR))
print('=== JOINT TRAIN: real + synthetic-as-augmentation (synw=%.2f, %d ep) ==='%(SYNW,EP),flush=True); t0=time.time()
for ep in range(EP):
    np.random.shuffle(order)
    for i in range(0,len(TR),CFG['bs']):
        H,G,lt,tvt,ev=batch_real([TR[j] for j in order[i:i+CFG['bs']]],True); loss=loss_fn(net(H,G,lt),tvt,ev)
        Hs,Gs,lts,tvts,evs=gen_synth(CFG['bs']); loss=loss+SYNW*loss_fn(net(Hs,Gs,lts),tvts,evs)   # synthetic as regularizer
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        with torch.no_grad():
            for pe,pn in zip(ema.parameters(),net.parameters()): pe.mul_(CFG['ema_decay']).add_(pn,alpha=1-CFG['ema_decay'])
            for be,bn in zip(ema.buffers(),net.buffers()): be.copy_(bn)
    if (ep+1)%3==0 or ep==EP-1:
        re=evaluate(ema); rr=evaluate(net)
        if re<best[0]: best=(re,ep+1)
        print('  ep%2d | raw %.3f | EMA %.3f (best %.3f@%d) %.0fs'%(ep+1,rr,re,best[0],best[1],time.time()-t0),flush=True)
print('DONE JOINT real+synth EMA best %.3f (flat15 WARP11 PF7 Tucker5.4)'%best[0],flush=True)
