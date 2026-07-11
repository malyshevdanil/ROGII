"""ROGII — self-contained GPU training (Kaggle-ready).
Neural grid Bayesian filter (PF) with learned drift, trained end-to-end to predict eval-zone TVT.
Ranked on a whole-well holdout (pooled RMSE). Auto-detects Kaggle vs local data path and GPU.
Paste as one Kaggle cell (Accelerator: GPU). Prints baselines + per-epoch holdout RMSE."""
import numpy as np, pandas as pd, glob, os, time, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV='cuda' if torch.cuda.is_available() else 'cpu'

# ---------- config (scale these up on Kaggle's fast GPU) ----------
CFG=dict(d=96, KSTEPS=60, EVSTEPS=360, TWGRID=256, drop=0.1, wd=1e-4, lr=2e-3, epochs=60, bs=16,
         emis_temp=2.0, trans_w=6.0, maxdrift=2.0, aug_caljit=0.08, aug_noise=0.05, w_known=0.3,
         n_val=160, seed=42)
TG=CFG['TWGRID']; KSTEPS=CFG['KSTEPS']; EVSTEPS=CFG['EVSTEPS']; SEQL=KSTEPS+EVSTEPS

# ---------- data path autodetect (Kaggle or local) ----------
c=glob.glob('/kaggle/input/**/*__horizontal_well.csv',recursive=True)
_tr=[p for p in c if 'train' in p.lower()]      # prefer the train/ folder, not test/ (test has no TVT labels)
c=_tr if _tr else c
TRAIN_DIR=os.path.dirname(c[0]) if c else 'data/train'
print('DEV=%s | data=%s'%(DEV,TRAIN_DIR),flush=True)

def interp_nan(a):
    a=a.copy();n=len(a);idx=np.arange(n);m=np.isnan(a)
    if m.all():return np.zeros(n)
    a[m]=np.interp(idx[m],idx[~m],a[~m]);return a

def build_well(wid):
    hw=pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw=pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tw_tvt=tw['TVT'].values.astype(float); tw_gr=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    if len(tw_tvt)<10: return None
    kn=hw[hw['TVT_input'].notna()]
    if len(kn)<20 or hw['TVT_input'].isna().sum()<20: return None
    last=kn.iloc[-1]; last_tvt=float(last['TVT_input'])
    n=len(hw); gr=interp_nan(hw['GR'].values.astype(float))
    kn_gr=kn['GR'].interpolate().bfill().ffill().values.astype(float)
    tw_at_k=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr); v=np.isfinite(kn_gr)&np.isfinite(tw_at_k)
    a,b=(np.polyfit(kn_gr[v],tw_at_k[v],1) if v.sum()>=20 else (1.,0.))
    cal_gr=gr*a+b
    Z=hw['Z'].values.astype(float); MD=hw['MD'].values.astype(float)
    mdd=np.gradient(MD); mdd[mdd==0]=1
    grad=np.gradient(gr); rstd=pd.Series(gr).rolling(21,center=True,min_periods=1).std().fillna(0).values
    dzdmd=np.gradient(Z)/mdd; md_since=(MD-float(last['MD']))
    ev=hw['TVT_input'].isna().values.astype(float); ei=np.where(ev>0.5)[0]
    if len(ei)<5: return None
    e0=ei[0]; ksrc=np.arange(max(0,e0-400),e0)
    if len(ksrc)<5: return None
    g_tvt=np.linspace(tw_tvt.min(),tw_tvt.max(),TG); g_gr=np.interp(g_tvt,tw_tvt,tw_gr)
    gm,gs=float(g_gr.mean()),float(g_gr.std()+1e-6)
    caln=(cal_gr-gm)/gs; gradn=grad/gs; rstdn=rstd/gs
    true=hw['TVT'].values.astype(float)
    # resample known-tail -> KSTEPS, eval -> EVSTEPS
    kd=np.linspace(ksrc[0],ksrc[-1],KSTEPS); ed=np.linspace(e0,n-1,EVSTEPS); dst=np.concatenate([kd,ed]); src=np.arange(n)
    R=lambda x: np.interp(dst,src,x).astype(np.float32)
    md_r=R(md_since); dstep=np.diff(md_r,prepend=md_r[0]); dstep[0]=dstep[1]
    tvt_r=R(true); j_true=np.clip(np.round((tvt_r-g_tvt[0])/(g_tvt[-1]-g_tvt[0])*(TG-1)),0,TG-1).astype(np.int64)
    last_j=int(np.clip(round((last_tvt-g_tvt[0])/(g_tvt[-1]-g_tvt[0])*(TG-1)),0,TG-1))
    evr=np.concatenate([np.zeros(KSTEPS),np.ones(EVSTEPS)]).astype(np.float32)
    return dict(H=np.stack([R(caln),R(gradn),R(rstdn),((R(dzdmd)-0)/1.0)]).astype(np.float32),
                gn=((g_gr-gm)/gs).astype(np.float32), gtvt=g_tvt.astype(np.float32), tvt=tvt_r,
                jtrue=j_true, last_j=last_j, ev=evr, last_tvt=last_tvt)

print('building features...',flush=True); t0=time.time()
wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
DATA=[b for b in (build_well(w) for w in wids) if b is not None]
# normalize dzdmd channel (idx 3) globally
d3=np.concatenate([w['H'][3] for w in DATA]); m3,s3=d3.mean(),d3.std()+1e-6
for w in DATA: w['H'][3]=(w['H'][3]-m3)/s3
rng=np.random.default_rng(CFG['seed']); idx=np.arange(len(DATA)); rng.shuffle(idx)
VA=[DATA[i] for i in idx[:CFG['n_val']]]; TR=[DATA[i] for i in idx[CFG['n_val']:]]
print('built %d wells (%d train / %d val) %.0fs'%(len(DATA),len(TR),len(VA),time.time()-t0),flush=True)

# ---------- baselines (targets) ----------
def prmse(sqs): return float(np.sqrt(np.mean(np.concatenate(sqs))))
flat=[(w['last_tvt']-w['tvt'][w['ev']>0.5])**2 for w in VA]
line=[]
for w in VA:
    m=w['ev']>0.5; x=np.arange(m.sum()); y=w['tvt'][m]
    A=np.polyfit(x,y,1); line.append((np.polyval(A,x)-y)**2)
print('BASELINES  flat=%.3f  per-well-line-oracle=%.3f  (target: beat line -> reach the wiggle ~5)'%(
    prmse(flat),prmse(line)),flush=True)

# ---------- model: neural grid Bayesian filter with learned drift ----------
class Enc(nn.Module):
    def __init__(s,nin,d,drop):
        super().__init__(); s.inp=nn.Conv1d(nin,d,5,padding=2)
        s.blocks=nn.ModuleList([nn.Sequential(
            nn.Conv1d(d,d,3,padding=dl,dilation=dl),nn.GroupNorm(8,d),nn.GELU(),nn.Dropout(drop),
            nn.Conv1d(d,d,3,padding=dl,dilation=dl),nn.GroupNorm(8,d),nn.GELU()) for dl in (1,2,4,8,16)])
        s.out=nn.Conv1d(d,d,1)
    def forward(s,x):
        h=F.gelu(s.inp(x))
        for b in s.blocks: h=h+b(h)
        return s.out(h)
class PFGrid(nn.Module):
    def __init__(s,d,drop):
        super().__init__(); s.he=Enc(4,d,drop); s.te=Enc(1,d,drop)
        s.log_transw=nn.Parameter(torch.tensor(np.log(CFG['trans_w']).astype('float32')))
        s.log_gain=nn.Parameter(torch.tensor(np.log(np.expm1(0.12)).astype('float32')))
        s.drift_head=nn.Sequential(nn.Conv1d(d,d,1),nn.GELU(),nn.Conv1d(d,1,1))
        s.register_buffer('koff',torch.arange(-40,41).float())
    def shift(s,p,dd):
        B,TGn=p.shape; ar=torch.arange(TGn,device=p.device).float()[None,:]
        i=ar-dd[:,None]; i0=i.floor(); fr=i-i0; i0l=i0.long()
        v0=((i0l>=0)&(i0l<TGn)).float(); v1=((i0l+1>=0)&(i0l+1<TGn)).float()
        return torch.gather(p,1,i0l.clamp(0,TGn-1))*(1-fr)*v0+torch.gather(p,1,(i0l+1).clamp(0,TGn-1))*fr*v1
    def transition(s,p):
        w=F.softplus(s.log_transw)+1.0; k=torch.exp(-0.5*(s.koff/w)**2); k=k/k.sum()
        return F.conv1d(p[:,None],k[None,None],padding=len(s.koff)//2)[:,0]
    def forward(s,H,G,jinit):
        B,_,L=H.shape; he_raw=s.he(H); he=F.normalize(he_raw,dim=1); te=F.normalize(s.te(G),dim=1)
        E=F.softplus(s.log_gain)*torch.einsum('bdl,bdt->blt',he,te)/CFG['emis_temp']
        Eexp=torch.exp(E-E.max(2,keepdim=True).values)
        drift=torch.tanh(s.drift_head(he_raw)[:,0])*CFG['maxdrift']
        p=torch.zeros(B,TG,device=H.device)
        p[torch.arange(B),jinit]=1.0; p=s.transition(p)+1e-8; p=p/p.sum(1,keepdim=True); outs=[]
        for t in range(L):
            p=s.transition(p); p=s.shift(p,drift[:,t])+1e-8; p=p/p.sum(1,keepdim=True)
            p=p*Eexp[:,t]+1e-8; p=p/p.sum(1,keepdim=True); outs.append(p)
        return torch.stack(outs,1)

net=PFGrid(CFG['d'],CFG['drop']).to(DEV)
opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG['lr'],total_steps=CFG['epochs']*((len(TR)+CFG['bs']-1)//CFG['bs']))
def batch(ws,train=False):
    H=np.stack([w['H'] for w in ws]); gn=np.stack([w['gn'][:TG] for w in ws])
    if train:
        sc=1+np.random.randn(len(ws),1,1)*CFG['aug_caljit']; sh=np.random.randn(len(ws),1,1)*CFG['aug_caljit']
        H[:,0:1]=H[:,0:1]*sc+sh; gn=gn*sc[:,:,0]+sh[:,:,0]; H=H+np.random.randn(*H.shape)*CFG['aug_noise']
    to=lambda a: torch.tensor(a,dtype=torch.float32,device=DEV)
    return (to(H),to(gn)[:,None],to(np.stack([w['gtvt'][:TG] for w in ws])),
            torch.tensor([w['last_j'] for w in ws],device=DEV),
            torch.tensor(np.stack([w['jtrue'] for w in ws]),device=DEV),
            to(np.stack([w['tvt'] for w in ws])),to(np.stack([w['ev'] for w in ws])))
def evaluate():
    net.eval(); e=[]
    with torch.no_grad():
        for i in range(0,len(VA),CFG['bs']):
            ws=VA[i:i+CFG['bs']]; H,G,gtvt,ji,jt,tvt,ev=batch(ws)
            pred=(net(H,G,ji)*gtvt[:,None]).sum(2); e.append(((pred-tvt)[ev>0.5]).cpu().numpy())
    net.train(); return float(np.sqrt((np.concatenate(e)**2).mean()))
print('training | params=%.0fk'%(sum(p.numel() for p in net.parameters())/1e3),flush=True)
t0=time.time(); best=(99,0); order=np.arange(len(TR))
for ep in range(CFG['epochs']):
    np.random.shuffle(order); tot=0;nb=0
    for i in range(0,len(TR),CFG['bs']):
        ws=[TR[j] for j in order[i:i+CFG['bs']]]; H,G,gtvt,ji,jt,tvt,ev=batch(ws,True)
        P=net(H,G,ji); pred=(P*gtvt[:,None]).sum(2)
        nll=-torch.log(P.clamp_min(1e-8)).gather(2,jt[:,:,None]).squeeze(2)
        m=ev+CFG['w_known']*(1-ev)
        loss=(nll*m).sum()/m.sum()+0.01*(F.smooth_l1_loss(pred,tvt,reduction='none',beta=8.0)*m).sum()/m.sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step()
        tot+=loss.item(); nb+=1
    if (ep+1)%2==0 or ep==CFG['epochs']-1:
        r=evaluate();
        if r<best[0]: best=(r,ep+1); torch.save(net.state_dict(),'best_pfgrid.pt')
        print('  ep%2d loss%.3f | holdout pooled RMSE %.3f (best %.3f@%d) gain=%.2f %.0fs'%(
            ep+1,tot/nb,r,best[0],best[1],F.softplus(net.log_gain).item(),time.time()-t0),flush=True)
print('DONE best holdout RMSE %.3f @ep%d (flat~15.1, line-oracle~6.6, target<6.6)'%(best[0],best[1]),flush=True)
