import numpy as np, pandas as pd, glob, os, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})
OFFS=[-20,-10,-5,0,5,10,20]
FEATS=['gr','gr_sm','gr_grad','gr_rstd','cal_gr','z','dzdmd','dxdmd','dydmd','md_since','tvtin','mask_kn']+['tda%d'%o for o in OFFS]
NF=len(FEATS); GR_CH=[0,1,4]+list(range(12,NF))  # GR-derived channels for calibration jitter
def interp_nan(a):
    a=a.copy();n=len(a);idx=np.arange(n);m=np.isnan(a)
    if m.all():return np.zeros(n)
    a[m]=np.interp(idx[m],idx[~m],a[~m]);return a
def build(wid, stride=4):
    hw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv'))
    tw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')).sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tw_tvt=tw['TVT'].values.astype(float); tw_gr=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    kn=hw[hw['TVT_input'].notna()]
    if len(kn)<20 or hw['TVT_input'].isna().sum()<20: return None
    last=kn.iloc[-1]; last_tvt=float(last['TVT_input']); last_Z=float(last['Z']); last_MD=float(last['MD'])
    n=len(hw); gr=interp_nan(hw['GR'].values.astype(float))
    gr_sm=pd.Series(gr).rolling(11,center=True,min_periods=1).mean().values
    gr_rstd=pd.Series(gr).rolling(21,center=True,min_periods=1).std().fillna(0).values
    kn_gr=kn['GR'].interpolate().bfill().ffill().values.astype(float)
    tw_at_k=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr); v=np.isfinite(kn_gr)&np.isfinite(tw_at_k)
    a,b=(np.polyfit(kn_gr[v],tw_at_k[v],1) if v.sum()>=20 else (1.,0.))
    Z=hw['Z'].values.astype(float);X=hw['X'].values.astype(float);Y=hw['Y'].values.astype(float);MD=hw['MD'].values.astype(float)
    mdd=np.gradient(MD);mdd[mdd==0]=1
    feat=np.zeros((n,NF),np.float32)
    feat[:,0]=gr;feat[:,1]=gr_sm;feat[:,2]=np.gradient(gr);feat[:,3]=gr_rstd;feat[:,4]=gr*a+b
    feat[:,5]=Z-last_Z;feat[:,6]=np.gradient(Z)/mdd;feat[:,7]=np.gradient(X)/mdd;feat[:,8]=np.gradient(Y)/mdd
    feat[:,9]=(MD-last_MD)/1000.;feat[:,10]=np.nan_to_num(hw['TVT_input'].values.astype(float)-last_tvt,nan=0.0)
    feat[:,11]=hw['TVT_input'].notna().values.astype(float)
    for j,o in enumerate(OFFS): feat[:,12+j]=gr-np.interp(last_tvt+o,tw_tvt,tw_gr)
    target=hw['TVT'].values.astype(float)-last_tvt; ev=hw['TVT_input'].isna().values.astype(np.float32)
    sel=np.arange(0,n,stride)
    return feat[sel],target[sel].astype(np.float32),ev[sel],wid
print('building...',flush=True); t0=time.time()
data=[b for b in (build(w) for w in well_ids) if b is not None]
print('wells',len(data),'%.0fs'%(time.time()-t0),flush=True)
allf=np.concatenate([d[0] for d in data],0); mu=allf.mean(0); sd=allf.std(0)+1e-6; del allf

def augment(fn):  # fn: standardized (L,NF) numpy
    f=fn.copy(); L=f.shape[0]
    sc=1.0+0.06*np.random.randn(); off=0.12*np.random.randn()
    f[:,GR_CH]=f[:,GR_CH]*sc+off
    f+=0.05*np.random.randn(*f.shape).astype(np.float32)
    # mild time-warp: resample to L' then back to L
    if np.random.rand()<0.5:
        w=1.0+0.08*(np.random.rand()-0.5)*2
        Lp=max(8,int(L*w)); xi=np.linspace(0,L-1,Lp);
        f2=np.stack([np.interp(np.linspace(0,L-1,L),np.linspace(0,L-1,Lp),np.interp(xi,np.arange(L),f[:,c])) for c in range(NF)],1)
        f=f2.astype(np.float32)
    return f

class MTP(nn.Module):
    def __init__(self,nf,h=64,K=2,drop=0.2):
        super().__init__(); self.K=K
        self.inp=nn.Conv1d(nf,h,1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(h,h,3,padding=p,dilation=p),nn.GELU(),nn.Dropout(drop)) for p in [1,2,4,8,16,32]])
        self.heads=nn.Conv1d(h,K,1); self.modepool=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(h,K))
    def forward(self,x):
        z=self.inp(x)
        for blk in self.blocks: z=z+blk(z)
        return self.heads(z).squeeze(0), self.modepool(z).squeeze(0)
val_ids=set(d[3] for d in data[-120:])
train=[d for d in data if d[3] not in val_ids]; val=[d for d in data if d[3] in val_ids]
print('train',len(train),'val',len(val),flush=True)
model=MTP(NF); opt=torch.optim.Adam(model.parameters(),lr=8e-4,weight_decay=3e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30)
def stdz(f): return (f-mu)/sd
def evaluate():
    model.eval(); se=0;cnt=0;fse=0
    with torch.no_grad():
        for f,t,m,wid in val:
            mm=torch.tensor(m)>0
            if mm.sum()<5: continue
            traj,logit=model(torch.tensor(stdz(f),dtype=torch.float32).T.unsqueeze(0)); p=torch.softmax(logit,0)
            pm=(p[:,None]*traj).sum(0); tt=torch.tensor(t)
            se+=float(((pm[mm]-tt[mm])**2).sum()); cnt+=int(mm.sum()); fse+=float((tt[mm]**2).sum())
    return (se/cnt)**0.5,(fse/cnt)**0.5
t0=time.time(); best=1e9
for ep in range(30):
    model.train(); order=np.random.permutation(len(train)); tot=0
    for k in order:
        f,t,m,wid=train[k]
        fa=augment(stdz(f))
        # time-warp changed length -> resample target/mask to match
        if fa.shape[0]!=len(t):
            L0=len(t); L1=fa.shape[0]; xi=np.linspace(0,L0-1,L1)
            t2=np.interp(xi,np.arange(L0),t); m2=(np.interp(xi,np.arange(L0),m)>0.5).astype(np.float32)
        else:
            t2,m2=t,m
        mm=torch.tensor(m2)>0
        if mm.sum()<5: continue
        traj,logit=model(torch.tensor(fa,dtype=torch.float32).T.unsqueeze(0)); tt=torch.tensor(t2)
        errs=torch.stack([((traj[j][mm]-tt[mm])**2).mean() for j in range(model.K)])
        win=int(torch.argmin(errs).item()); reg=errs[win]
        cls=F.cross_entropy(logit.unsqueeze(0),torch.tensor([win]))
        loss=reg+0.5*cls
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tot+=float(reg.detach())
    sched.step()
    pm,fr=evaluate(); best=min(best,pm)
    print('ep %2d reg %.1f | VAL pm %.3f (flat %.3f) best=%.3f %.0fs'%(ep+1,tot/len(train),pm,fr,best,time.time()-t0),flush=True)
print('DONE MTP+AUG best val %.3f (flat %.3f)  [no-aug MTP was 15.073]'%(best,fr),flush=True)
