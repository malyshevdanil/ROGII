"""Experiment 1: regularized dilated TCN, per-point TVT-delta regression.
Ranked on the shared whole-well holdout (prep.pooled_rmse_eval). Configurable for fast iteration."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, sys, os, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import prep
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(4)

CFG=dict(ch=48, layers=[1,2,4,8,16,32], drop=0.15, wd=1e-4, lr=2e-3, epochs=40, bs=8,
         aug_caljit=0.08, aug_noise=0.05, aug_chdrop=0.1, huber=8.0, sup_known=0.0, tag='tcn_base')
# allow CLI overrides: key=value
for a in sys.argv[1:]:
    k,v=a.split('='); CFG[k]=type(CFG[k])(v) if k in CFG and not isinstance(CFG[k],list) else v

data=prep.load(); tr,va=prep.split(data)
# per-channel normalization from TRAIN points
allf=np.concatenate([w['feat'] for w in tr],0)
mu=allf.mean(0); sd=allf.std(0)+1e-6
mu[9]=0; sd[9]=1  # keep mask_kn as 0/1
def norm(f): return (f-mu)/sd
# target scale (delta) for loss stability
tgt_all=np.concatenate([w['target'][w['ev']>0.5] for w in tr]); TSD=float(tgt_all.std())
print('target delta std=%.2f (scale of the problem)'%TSD,flush=True)

GR_CH=prep.GR_CH
def make_batch(wells, train=False):
    L=max(len(w['feat']) for w in wells)
    B=len(wells); NF=prep.NF
    X=np.zeros((B,NF,L),np.float32); Y=np.zeros((B,L),np.float32); M=np.zeros((B,L),np.float32); PAD=np.zeros((B,L),np.float32)
    for i,w in enumerate(wells):
        f=norm(w['feat']).copy(); n=len(f)
        y=w['target'].copy(); ev=w['ev'].copy()
        if train:
            # GR calibration jitter: affine on GR-derived (normalized) channels
            s=1.0+np.random.randn()*CFG['aug_caljit']; sh=np.random.randn()*CFG['aug_caljit']
            f[:,GR_CH]=f[:,GR_CH]*s+sh
            f=f+np.random.randn(*f.shape).astype(np.float32)*CFG['aug_noise']
            if CFG['aug_chdrop']>0:
                dropm=(np.random.rand(NF)<CFG['aug_chdrop']); dropm[9]=False
                f[:,dropm]=0.0
        X[i,:,:n]=f.T; Y[i,:n]=y; PAD[i,:n]=1.0
        m=ev.copy()
        if CFG['sup_known']>0: m=np.maximum(m, CFG['sup_known']*w['feat'][:,9])  # light known-point supervision
        M[i,:n]=m
    return (torch.from_numpy(X),torch.from_numpy(Y/TSD),torch.from_numpy(M),torch.from_numpy(PAD))

class Block(nn.Module):
    def __init__(s,c,d,drop):
        super().__init__(); s.c1=nn.Conv1d(c,c,3,padding=d,dilation=d); s.c2=nn.Conv1d(c,c,3,padding=d,dilation=d)
        s.n1=nn.GroupNorm(8,c); s.n2=nn.GroupNorm(8,c); s.dp=nn.Dropout(drop)
    def forward(s,x):
        h=F.gelu(s.n1(s.c1(x))); h=s.dp(h); h=F.gelu(s.n2(s.c2(h))); return x+h
class TCN(nn.Module):
    def __init__(s,nf,c,dils,drop):
        super().__init__(); s.inp=nn.Conv1d(nf,c,1); s.blocks=nn.ModuleList([Block(c,d,drop) for d in dils])
        s.head=nn.Sequential(nn.Conv1d(c,c,1),nn.GELU(),nn.Dropout(drop),nn.Conv1d(c,1,1))
    def forward(s,x):
        h=s.inp(x)
        for b in s.blocks: h=b(h)
        return s.head(h).squeeze(1)

dev='cpu'; net=TCN(prep.NF,CFG['ch'],CFG['layers'],CFG['drop']).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG['lr'],total_steps=CFG['epochs']*((len(tr)+CFG['bs']-1)//CFG['bs']))
hub=CFG['huber']/TSD

def evaluate():
    net.eval(); preds=[]
    with torch.no_grad():
        for w in va:
            X,_,_,_=make_batch([w],train=False)
            p=net(X.to(dev))[0,:len(w['feat'])].cpu().numpy()*TSD
            preds.append(p)
    net.train(); return prep.pooled_rmse_eval(preds,va)

print('training %s | params=%.0fk'%(CFG['tag'],sum(p.numel() for p in net.parameters())/1e3),flush=True)
t0=time.time(); best=(99,0)
order=np.arange(len(tr))
for ep in range(CFG['epochs']):
    np.random.shuffle(order); net.train(); tot=0;nb=0
    for i in range(0,len(tr),CFG['bs']):
        wells=[tr[j] for j in order[i:i+CFG['bs']]]
        X,Y,M,PAD=make_batch(wells,train=True); X,Y,M=X.to(dev),Y.to(dev),M.to(dev)
        p=net(X); err=p-Y
        loss=(F.huber_loss(p,Y,reduction='none',delta=hub)*M).sum()/(M.sum()+1e-6)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step()
        tot+=loss.item();nb+=1
    if (ep+1)%4==0 or ep==CFG['epochs']-1:
        r,npts=evaluate()
        if r<best[0]: best=(r,ep+1)
        print('  ep%2d loss%.3f | holdout pooled RMSE %.3f  (best %.3f @ep%d) %.0fs'%(ep+1,tot/nb,r,best[0],best[1],time.time()-t0),flush=True)
r,_=evaluate()
print('FINAL %s holdout pooled RMSE %.3f | best %.3f'%(CFG['tag'],r,best[0]),flush=True)
# log
logp=os.path.join(os.path.dirname(os.path.abspath(__file__)),'RESULTS.jsonl')
open(logp,'a').write(json.dumps(dict(tag=CFG['tag'],best=best[0],final=r,cfg={k:v for k,v in CFG.items()},t=time.time()-t0))+'\n')
