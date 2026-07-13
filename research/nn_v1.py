import numpy as np, pandas as pd, glob, os, time
import torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

OFFS=[-20,-10,-5,0,5,10,20]
FEATS=['gr','gr_sm','gr_grad','gr_rstd','cal_gr','z','dzdmd','dxdmd','dydmd','md_since','tvtin','mask_kn']+['tda%d'%o for o in OFFS]
NF=len(FEATS)

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
    n=len(hw)
    gr=interp_nan(hw['GR'].values.astype(float))
    gr_sm=pd.Series(gr).rolling(11,center=True,min_periods=1).mean().values
    gr_grad=np.gradient(gr)
    gr_rstd=pd.Series(gr).rolling(21,center=True,min_periods=1).std().fillna(0).values
    kn_gr=kn['GR'].interpolate().bfill().ffill().values.astype(float)
    tw_at_k=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr); v=np.isfinite(kn_gr)&np.isfinite(tw_at_k)
    if v.sum()>=20: a,b=np.polyfit(kn_gr[v],tw_at_k[v],1)
    else: a,b=1.0,0.0
    cal_gr=gr*a+b
    Z=hw['Z'].values.astype(float); X=hw['X'].values.astype(float); Y=hw['Y'].values.astype(float); MD=hw['MD'].values.astype(float)
    mdd=np.gradient(MD); mdd[mdd==0]=1
    dzdmd=np.gradient(Z)/mdd; dxdmd=np.gradient(X)/mdd; dydmd=np.gradient(Y)/mdd
    md_since=(MD-last_MD)/1000.0
    tvtin=np.nan_to_num(hw['TVT_input'].values.astype(float)-last_tvt,nan=0.0)
    mask_kn=hw['TVT_input'].notna().values.astype(float)
    feat=np.zeros((n,NF),np.float32)
    feat[:,0]=gr; feat[:,1]=gr_sm; feat[:,2]=gr_grad; feat[:,3]=gr_rstd; feat[:,4]=cal_gr
    feat[:,5]=Z-last_Z; feat[:,6]=dzdmd; feat[:,7]=dxdmd; feat[:,8]=dydmd; feat[:,9]=md_since
    feat[:,10]=tvtin; feat[:,11]=mask_kn
    for j,o in enumerate(OFFS):
        feat[:,12+j]=gr-np.interp(last_tvt+o,tw_tvt,tw_gr)
    tvt=hw['TVT'].values.astype(float)
    target=tvt-last_tvt
    ev=hw['TVT_input'].isna().values
    loss_mask=ev.astype(np.float32)
    # stride to cap length
    sel=np.arange(0,n,stride)
    return feat[sel], target[sel].astype(np.float32), loss_mask[sel], wid

# normalize features globally
print('building features...',flush=True)
data=[]; t0=time.time()
for i,wid in enumerate(well_ids):
    r=build(wid)
    if r is not None: data.append(r)
    if (i+1)%150==0: print('%d/%d %.0fs'%(i+1,len(well_ids),time.time()-t0),flush=True)
print('wells with data:',len(data),flush=True)

allfeat=np.concatenate([d[0] for d in data],0)
mu=allfeat.mean(0); sd=allfeat.std(0)+1e-6
del allfeat

def to_tensor(d):
    f,t,m,wid=d
    f=(f-mu)/sd
    return torch.tensor(f,dtype=torch.float32), torch.tensor(t,dtype=torch.float32), torch.tensor(m,dtype=torch.float32)

class CNN(nn.Module):
    def __init__(self,nf,h=96):
        super().__init__()
        self.inp=nn.Conv1d(nf,h,1)
        self.blocks=nn.ModuleList()
        for d in [1,2,4,8,16,32,1]:
            self.blocks.append(nn.Sequential(nn.Conv1d(h,h,3,padding=d,dilation=d),nn.GELU(),nn.BatchNorm1d(h)))
        self.out=nn.Conv1d(h,1,1)
    def forward(self,x):  # x: (B, nf, L)
        h=self.inp(x)
        for blk in self.blocks: h=h+blk(h)
        return self.out(h).squeeze(1)  # (B, L)

# GroupKFold: hold out last 120 wells for validation
val_ids=set(w for *_,w in [d for d in data][-120:])
train=[d for d in data if d[3] not in val_ids]
val=[d for d in data if d[3] in val_ids]
print('train wells',len(train),'val wells',len(val),flush=True)

dev='cpu'
model=CNN(NF).to(dev)
opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=25)

def val_rmse():
    model.eval(); se=0.0; cnt=0; flat_se=0.0
    with torch.no_grad():
        for d in val:
            f,t,m=to_tensor(d)
            pred=model(f.T.unsqueeze(0)).squeeze(0)
            mm=m>0
            se+=float(((pred[mm]-t[mm])**2).sum()); cnt+=int(mm.sum())
            flat_se+=float((t[mm]**2).sum())
    return (se/cnt)**0.5, (flat_se/cnt)**0.5

t0=time.time()
for ep in range(25):
    model.train(); order=np.random.permutation(len(train)); tot=0.0
    for k in order:
        f,t,m=to_tensor(train[k])
        pred=model(f.T.unsqueeze(0)).squeeze(0)
        mm=m>0
        if mm.sum()<5: continue
        loss=((pred[mm]-t[mm])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss)
    sched.step()
    if (ep+1)%2==0 or ep==0:
        vr,fr=val_rmse()
        print('ep %2d  train_mse %.2f  VAL RMSE %.3f  (flat baseline %.3f)  %.0fs'%(ep+1,tot/len(train),vr,fr,time.time()-t0),flush=True)
torch.save({'model':model.state_dict(),'mu':mu,'sd':sd},r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad\nn_v1.pt')
print('saved. total %.0fs'%(time.time()-t0),flush=True)
