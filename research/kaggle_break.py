"""ROGII BREAKTHROUGH attempt: alignment -> surface -> low-pass -> minus Z.
Alignment (soft-argmax over typewell) gives absolute TVT that is PER-POINT (no cumsum drift) so its error
is high-freq jitter. surface=align_TVT+Z is then LOW-PASSED (removes the HF alignment noise, keeps the
smooth true surface) and Z subtracted back exactly (restores the wiggle). Premise (sim): HF-noise 30 ->
lowpass surface 3. Kaggle: GPU T4 x2 (NOT P100), Internet off, competition dataset only."""
import numpy as np, pandas as pd, glob, os, time, copy, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0); DEV='cuda' if torch.cuda.is_available() else 'cpu'
CFG=dict(d=128, KSTEPS=60, EVSTEPS=360, TWGRID=256, drop=0.15, wd=3e-4, lr=1.5e-3, epochs=120, bs=24,
         lp_k=81, w_mono=0.02, w_ent=0.004, aug_caljit=0.12, aug_noise=0.06, aug_warp=0.35, aug_twjit=0.06,
         w_known=0.5, ema_decay=0.995, n_val=160, seed=42, tag='break')
TG=CFG['TWGRID']; KSTEPS=CFG['KSTEPS']; EVSTEPS=CFG['EVSTEPS']; SEQL=KSTEPS+EVSTEPS
c=glob.glob('/kaggle/input/**/*__horizontal_well.csv',recursive=True); _t=[p for p in c if 'train' in p.lower()]; c=_t if _t else c
TRAIN_DIR=os.path.dirname(c[0]) if c else 'data/train'; print('DEV=%s data=%s'%(DEV,TRAIN_DIR),flush=True)
def inan(a):
    a=a.copy();n=len(a);i=np.arange(n);m=np.isnan(a)
    if m.all():return np.zeros(n)
    a[m]=np.interp(i[m],i[~m],a[~m]);return a
def build(wid):
    hw=pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv'); tw=pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tt=tw['TVT'].values.astype(float); tg=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    if len(tt)<10: return None
    kn=hw[hw['TVT_input'].notna()]
    if len(kn)<20 or hw['TVT_input'].isna().sum()<20: return None
    last=kn.iloc[-1]; last_tvt=float(last['TVT_input']); last_Z=float(last['Z'])
    n=len(hw); gr=inan(hw['GR'].values.astype(float)); kg=kn['GR'].interpolate().bfill().ffill().values.astype(float)
    twk=np.interp(kn['TVT_input'].values,tt,tg); v=np.isfinite(kg)&np.isfinite(twk); a,b=(np.polyfit(kg[v],twk[v],1) if v.sum()>=20 else (1.,0.)); cg=gr*a+b
    Z=hw['Z'].values.astype(float); MD=hw['MD'].values.astype(float); mdd=np.gradient(MD); mdd[mdd==0]=1
    grad=np.gradient(gr); rstd=pd.Series(gr).rolling(21,center=True,min_periods=1).std().fillna(0).values; dzdmd=np.gradient(Z)/mdd
    ev=hw['TVT_input'].isna().values.astype(float); ei=np.where(ev>0.5)[0]
    if len(ei)<5: return None
    e0=ei[0]; ks=np.arange(max(0,e0-400),e0)
    if len(ks)<5: return None
    g_tvt=np.linspace(tt.min(),tt.max(),TG); g_gr=np.interp(g_tvt,tt,tg); gm,gs=float(g_gr.mean()),float(g_gr.std()+1e-6)
    cs=pd.Series((cg-gm)/gs); caln=cs.values; sm15=cs.rolling(15,center=True,min_periods=1).mean().values; sm41=cs.rolling(41,center=True,min_periods=1).mean().values
    dog1=cs.rolling(5,center=True,min_periods=1).mean().values-sm15; dog2=sm15-sm41
    true=hw['TVT'].values.astype(float); zrel=Z-last_Z
    kd=np.linspace(ks[0],ks[-1],KSTEPS); ed=np.linspace(e0,n-1,EVSTEPS); dst=np.concatenate([kd,ed]); src=np.arange(n)
    R=lambda x: np.interp(dst,src,x).astype(np.float32); evr=np.concatenate([np.zeros(KSTEPS),np.ones(EVSTEPS)]).astype(np.float32)
    return dict(H=np.stack([R(caln),R(sm15),R(sm41),R(dog1),R(dog2),R(grad/gs),R(rstd/gs),R(dzdmd)]).astype(np.float32),
                gn=((g_gr-gm)/gs).astype(np.float32), g_tvt=g_tvt.astype(np.float32), zrel=R(zrel).astype(np.float32),
                tvt=R(true), ev=evr, last_tvt=last_tvt)
print('building...',flush=True); t0=time.time()
wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
DATA=[b for b in (build(w) for w in wids) if b is not None]
d7=np.concatenate([w['H'][7] for w in DATA]); m7,s7=d7.mean(),d7.std()+1e-6
for w in DATA: w['H'][7]=(w['H'][7]-m7)/s7
rng=np.random.default_rng(CFG['seed']); idx=np.arange(len(DATA)); rng.shuffle(idx)
VA=[DATA[i] for i in idx[:CFG['n_val']]]; TR=[DATA[i] for i in idx[CFG['n_val']:]]
print('built %d (%d/%d) %.0fs'%(len(DATA),len(TR),len(VA),time.time()-t0),flush=True)
def prmse(s): return float(np.sqrt(np.mean(np.concatenate(s))))
flat=[(w['last_tvt']-w['tvt'][w['ev']>0.5])**2 for w in VA]
line=[np.polyval(np.polyfit(np.arange((w['ev']>0.5).sum()),w['tvt'][w['ev']>0.5],1),np.arange((w['ev']>0.5).sum()))-w['tvt'][w['ev']>0.5] for w in VA]
print('BASELINES flat=%.3f line-oracle=%.3f'%(prmse(flat),prmse([l**2 for l in line])),flush=True)

class Enc(nn.Module):
    def __init__(s,nin,d,drop):
        super().__init__(); s.inp=nn.Conv1d(nin,d,5,padding=2)
        s.bl=nn.ModuleList([nn.Sequential(nn.Conv1d(d,d,3,padding=dl,dilation=dl),nn.GroupNorm(8,d),nn.GELU(),nn.Dropout(drop),
            nn.Conv1d(d,d,3,padding=dl,dilation=dl),nn.GroupNorm(8,d),nn.GELU()) for dl in (1,2,4,8,16)]); s.o=nn.Conv1d(d,d,1)
    def forward(s,x):
        h=F.gelu(s.inp(x))
        for b in s.bl: h=h+b(h)
        return s.o(h)
class BreakNet(nn.Module):
    def __init__(s,d,drop):
        super().__init__(); s.he=Enc(8,d,drop); s.te=Enc(1,d,drop); s.q=nn.Conv1d(d,d,1); s.k=nn.Conv1d(d,d,1)
        s.log_temp=nn.Parameter(torch.tensor(np.log(10.).astype('float32')))
    def forward(s,H,G,g_tvt,zrel):
        h=s.he(H); t=s.te(G); Q=F.normalize(s.q(h).transpose(1,2),dim=2); K=F.normalize(s.k(t).transpose(1,2),dim=2)
        M=(Q@K.transpose(1,2))*F.softplus(s.log_temp); A=torch.softmax(M,2)
        idx=torch.arange(A.shape[2],device=A.device).float(); pos=A@idx
        align=(A*g_tvt[:,None,:]).sum(2)                      # (B,L) absolute per-point TVT (HF-noisy, no drift)
        surf=align+zrel                                       # surface proxy (=TVT+Z up to const)
        kk=CFG['lp_k']; pad=kk//2
        surf_lp=F.avg_pool1d(F.pad(surf[:,None],(pad,pad),mode='replicate'),kk,1,0)[:,0]   # LOW-PASS (replicate-pad, no zero artifact)
        tvt=surf_lp-zrel                                      # subtract exact Z -> restore the wiggle
        return tvt,pos,A
def make(seed):
    torch.manual_seed(seed); net=BreakNet(CFG['d'],CFG['drop']).to(DEV)
    opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG['lr'],total_steps=CFG['epochs']*((len(TR)+CFG['bs']-1)//CFG['bs']))
    return net,opt,sch
def batch(ws,train=False):
    H=np.stack([w['H'] for w in ws]).copy(); gn=np.stack([w['gn'][:TG] for w in ws]).copy()
    tvt=np.stack([w['tvt'] for w in ws]).copy(); ev=np.stack([w['ev'] for w in ws]); zr=np.stack([w['zrel'] for w in ws]).copy()
    if train:
        sc=1+np.random.randn(len(ws),1,1)*CFG['aug_caljit']; sh=np.random.randn(len(ws),1,1)*CFG['aug_caljit']
        H[:,0:3]=H[:,0:3]*sc+sh; H[:,3:7]=H[:,3:7]*sc; gn=gn*sc[:,:,0]+sh[:,:,0]; H=H+np.random.randn(*H.shape)*CFG['aug_noise']
        gn=gn+np.random.randn(*gn.shape)*CFG['aug_twjit']
        if CFG['aug_warp']>0:
            m=EVSTEPS; src=np.arange(m)
            for bi in range(len(ws)):
                d=np.cumsum(np.abs(1+np.random.randn(m)*CFG['aug_warp'])); d=(d-d[0])/(d[-1]-d[0])*(m-1)
                for ch in range(H.shape[1]): H[bi,ch,KSTEPS:]=np.interp(d,src,H[bi,ch,KSTEPS:])
                tvt[bi,KSTEPS:]=np.interp(d,src,tvt[bi,KSTEPS:]); zr[bi,KSTEPS:]=np.interp(d,src,zr[bi,KSTEPS:])
    to=lambda a: torch.tensor(a,dtype=torch.float32,device=DEV)
    return (to(H),to(gn)[:,None],to(np.stack([w['g_tvt'][:TG] for w in ws])),to(zr),to(tvt),to(ev))
def evaluate(mdl):
    mdl.eval(); e=[]
    with torch.no_grad():
        for i in range(0,len(VA),CFG['bs']):
            H,G,gt,zr,tvt,ev=batch(VA[i:i+CFG['bs']]); p,_,_=mdl(H,G,gt,zr); e.append(((p-tvt)[ev>0.5]).cpu().numpy())
    mdl.train(); return float(np.sqrt((np.concatenate(e)**2).mean()))
net,opt,sch=make(CFG['seed']); ema=copy.deepcopy(net); [p.requires_grad_(False) for p in ema.parameters()]; edec=CFG['ema_decay']
print('training BREAK (align->surface->lowpass->-Z) | params=%.0fk'%(sum(p.numel() for p in net.parameters())/1e3),flush=True)
t0=time.time(); best=(99,0); bestema=(99,0); order=np.arange(len(TR))
for ep in range(CFG['epochs']):
    np.random.shuffle(order)
    for i in range(0,len(TR),CFG['bs']):
        H,G,gt,zr,tvt,ev=batch([TR[j] for j in order[i:i+CFG['bs']]],True); p,pos,A=net(H,G,gt,zr)
        m=ev+CFG['w_known']*(1-ev); loss=(F.smooth_l1_loss(p,tvt,reduction='none',beta=8.0)*m).sum()/m.sum()
        loss=loss+CFG['w_mono']*(F.relu(pos[:,:-1]-pos[:,1:])*m[:,1:]).sum()/m[:,1:].sum()
        loss=loss+CFG['w_ent']*(-(A*torch.log(A+1e-8)).sum(2)*m).sum()/m.sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        with torch.no_grad():
            for pe,pn in zip(ema.parameters(),net.parameters()): pe.mul_(edec).add_(pn,alpha=1-edec)
            for be,bn in zip(ema.buffers(),net.buffers()): be.copy_(bn)
    if (ep+1)%2==0 or ep==CFG['epochs']-1:
        r=evaluate(net); re=evaluate(ema)
        if r<best[0]: best=(r,ep+1)
        if re<bestema[0]: bestema=(re,ep+1)
        print('  ep%2d | raw %.3f (best %.3f) | EMA %.3f (best %.3f@%d) %.0fs'%(ep+1,r,best[0],re,bestema[0],bestema[1],time.time()-t0),flush=True)
print('DONE raw %.3f | EMA %.3f (flat~15, line~6.6, PF~7, Tucker~5) — BREAKTHROUGH if <7'%(best[0],bestema[0]),flush=True)
