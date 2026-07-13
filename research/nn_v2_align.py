import numpy as np, pandas as pd, glob, os, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

HSTRIDE=6      # horizontal downsample
TWGRID=384     # typewell resampled grid size

def interp_nan(a):
    a=a.copy();n=len(a);idx=np.arange(n);m=np.isnan(a)
    if m.all():return np.zeros(n)
    a[m]=np.interp(idx[m],idx[~m],a[~m]);return a

def build(wid):
    hw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv'))
    tw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')).sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tw_tvt=tw['TVT'].values.astype(float); tw_gr=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    kn=hw[hw['TVT_input'].notna()]
    if len(kn)<20 or hw['TVT_input'].isna().sum()<20 or len(tw_tvt)<10: return None
    last=kn.iloc[-1]; last_tvt=float(last['TVT_input']); last_Z=float(last['Z']); last_MD=float(last['MD'])
    # resample typewell to fixed grid over its TVT range
    g_tvt=np.linspace(tw_tvt.min(),tw_tvt.max(),TWGRID)
    g_gr=np.interp(g_tvt,tw_tvt,tw_gr)
    # horizontal features
    gr=interp_nan(hw['GR'].values.astype(float))
    kn_gr=kn['GR'].interpolate().bfill().ffill().values.astype(float)
    tw_at_k=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr); v=np.isfinite(kn_gr)&np.isfinite(tw_at_k)
    a,b=(np.polyfit(kn_gr[v],tw_at_k[v],1) if v.sum()>=20 else (1.0,0.0))
    cal_gr=gr*a+b
    Z=hw['Z'].values.astype(float); MD=hw['MD'].values.astype(float)
    mdd=np.gradient(MD); mdd[mdd==0]=1; dzdmd=np.gradient(Z)/mdd
    md_since=(MD-last_MD)/1000.0
    mask_kn=hw['TVT_input'].notna().values.astype(float)
    tvt=hw['TVT'].values.astype(float)
    # slope prior from known tail
    tail=kn.tail(50);
    if len(tail)>=5:
        sl=np.polyfit(tail['MD'].values,tail['TVT_input'].values,1)[0]
    else: sl=0.0
    prior=last_tvt+sl*(MD-last_MD)   # physical linear prior per position
    n=len(hw); sel=np.arange(0,n,HSTRIDE)
    hfeat=np.stack([gr,cal_gr,np.gradient(gr),Z-last_Z,dzdmd,md_since,mask_kn,
                    np.nan_to_num(hw['TVT_input'].values.astype(float)-last_tvt,nan=0.0)],1)[sel]
    return dict(wid=wid, hfeat=hfeat.astype(np.float32),
                g_gr=g_gr.astype(np.float32), g_tvt=(g_tvt-last_tvt).astype(np.float32),
                prior=(prior[sel]-last_tvt).astype(np.float32),
                target=(tvt[sel]-last_tvt).astype(np.float32),
                mask_ev=(1.0-mask_kn[sel]).astype(np.float32),
                mask_all=np.ones(len(sel),np.float32))

print('building...',flush=True); t0=time.time()
data=[]
for i,wid in enumerate(well_ids):
    r=build(wid)
    if r is not None: data.append(r)
print('wells',len(data),'%.0fs'%(time.time()-t0),flush=True)

# normalize hfeat + g_gr globally
allh=np.concatenate([d['hfeat'] for d in data],0); hmu=allh.mean(0); hsd=allh.std(0)+1e-6; del allh
allg=np.concatenate([d['g_gr'] for d in data],0); gmu=float(allg.mean()); gsd=float(allg.std()+1e-6); del allg
TVSCALE=50.0  # scale for tvt-space (delta)

class Aligner(nn.Module):
    def __init__(self, hf=8, d=64):
        super().__init__()
        # horizontal encoder (dilated conv)
        self.hin=nn.Conv1d(hf,d,1)
        self.hblocks=nn.ModuleList([nn.Sequential(nn.Conv1d(d,d,3,padding=p,dilation=p),nn.GELU()) for p in [1,2,4,8,16]])
        self.hq=nn.Conv1d(d,d,1)
        # typewell encoder
        self.tin=nn.Conv1d(1,d,1)
        self.tblocks=nn.ModuleList([nn.Sequential(nn.Conv1d(d,d,3,padding=p,dilation=p),nn.GELU()) for p in [1,2,4,8]])
        self.tk=nn.Conv1d(d,d,1)
        self.log_lambda=nn.Parameter(torch.tensor(0.0))  # locality prior strength
        self.d=d
    def forward(self, h, tgr, gtvt, prior):
        # h: (1,hf,L) ; tgr:(1,1,K); gtvt:(K,) delta ; prior:(L,) delta
        hz=self.hin(h)
        for blk in self.hblocks: hz=hz+blk(hz)
        q=self.hq(hz).squeeze(0).T           # (L,d)
        tz=self.tin(tgr)
        for blk in self.tblocks: tz=tz+blk(tz)
        k=self.tk(tz).squeeze(0).T           # (K,d)
        scores=(q@k.T)/ (self.d**0.5)        # (L,K)
        lam=torch.exp(self.log_lambda)
        loc=-lam*((gtvt.view(1,-1)-prior.view(-1,1))/TVSCALE)**2   # locality prior (L,K)
        attn=torch.softmax(scores+loc, dim=1)                       # (L,K)
        pred=attn@gtvt                                              # (L,) expected delta-tvt
        return pred

def pack(d):
    h=torch.tensor((d['hfeat']-hmu)/hsd,dtype=torch.float32).T.unsqueeze(0)
    tgr=torch.tensor((d['g_gr']-gmu)/gsd,dtype=torch.float32).view(1,1,-1)
    gtvt=torch.tensor(d['g_tvt'],dtype=torch.float32)
    prior=torch.tensor(d['prior'],dtype=torch.float32)
    tgt=torch.tensor(d['target'],dtype=torch.float32)
    mev=torch.tensor(d['mask_ev'],dtype=torch.float32)
    mall=torch.tensor(d['mask_all'],dtype=torch.float32)
    return h,tgr,gtvt,prior,tgt,mev,mall

val_ids=set(d['wid'] for d in data[-120:])
train=[d for d in data if d['wid'] not in val_ids]; val=[d for d in data if d['wid'] in val_ids]
print('train',len(train),'val',len(val),flush=True)

model=Aligner(); opt=torch.optim.Adam(model.parameters(),lr=8e-4,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20)

def val_rmse():
    model.eval(); se=0;cnt=0;fse=0
    with torch.no_grad():
        for d in val:
            h,tgr,gtvt,prior,tgt,mev,mall=pack(d)
            pred=model(h,tgr,gtvt,prior)
            m=mev>0
            se+=float(((pred[m]-tgt[m])**2).sum()); cnt+=int(m.sum()); fse+=float((tgt[m]**2).sum())
    return (se/cnt)**0.5,(fse/cnt)**0.5

best=1e9; t0=time.time()
for ep in range(20):
    model.train(); order=np.random.permutation(len(train)); tot=0
    for k in order:
        h,tgr,gtvt,prior,tgt,mev,mall=pack(train[k])
        pred=model(h,tgr,gtvt,prior)
        # supervise BOTH zones: known (dense, teaches matching) + eval; weight eval higher
        w=mall+2.0*mev
        loss=((pred-tgt)**2*w).sum()/w.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tot+=float(loss.detach())
    sched.step()
    vr,fr=val_rmse(); best=min(best,vr)
    print('ep %2d train %.2f  VAL RMSE %.3f (flat %.3f) lam=%.2f best=%.3f %.0fs'%(
        ep+1,tot/len(train),vr,fr,float(torch.exp(model.log_lambda)),best,time.time()-t0),flush=True)
print('DONE best val %.3f'%best,flush=True)
