"""Experiment 2: differentiable GR<->typewell matcher (neural DTW-ish).
Predict each horizontal point's TVT via soft-argmax over the typewell grid, with a CONTINUITY PRIOR
(softmax biased toward last_tvt -> can't diverge, worst case ~flat) and SUPERVISED known-zone alignment
(we know true TVT on known pts -> teaches GR->position). Ranked on the shared whole-well holdout."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, sys, os, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import prep
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

CFG=dict(d=32, hlayers=[1,2,4,8,16], tlayers=[1,2,4,8], drop=0.1, wd=1e-4, lr=2e-3, epochs=40, bs=4,
         temp=1.0, smooth=0.5, w_known=0.3, iters=3, smoothk=41, aug_caljit=0.08, aug_noise=0.04, aug_warp=0.0, tag='matcher_base')
for a in sys.argv[1:]:
    k,v=a.split('=');
    if k in CFG and not isinstance(CFG[k],list): CFG[k]=type(CFG[k])(v)
    else: CFG[k]=v

data=prep.load(); tr,va=prep.split(data)
HCH=[0,1,2,3,4,5,6,7,9]   # gr,gr_sm,gr_grad,gr_rstd,cal_gr,z_rel,dzdmd,md_since,mask_kn (NO tvtin -> no leak)
CALIDX=HCH.index(4)       # position of cal_gr within HCH
# global norm for geometry-ish channels (5,6,7); GR channels standardized per-well by typewell stats
gf=np.concatenate([w['feat'] for w in tr],0); GMU=gf.mean(0); GSD=gf.std(0)+1e-6

def prep_well(w, train=False):
    f=w['feat'][:,HCH].astype(np.float32).copy()
    g=w['g_gr'].astype(np.float32).copy(); gtvt=w['g_tvt'].astype(np.float32)
    # per-well standardize GR channels (cols 0,1,4->cal_gr) and typewell by typewell GR stats
    gm,gs=float(g.mean()),float(g.std()+1e-6)
    for c in [0,1,4]:  # raw gr, gr_sm, cal_gr are in GR units
        f[:,HCH.index(c)]=(f[:,HCH.index(c)]-gm)/gs
    f[:,HCH.index(2)]=f[:,HCH.index(2)]/gs   # gr_grad scale
    f[:,HCH.index(3)]=f[:,HCH.index(3)]/gs   # gr_rstd scale
    gn=(g-gm)/gs
    # geometry channels 5,6,7 global norm
    for c in [5,6,7]:
        i=HCH.index(c); f[:,i]=(f[:,i]-GMU[c])/GSD[c]
    if train:
        s=1.0+np.random.randn()*CFG['aug_caljit']; sh=np.random.randn()*CFG['aug_caljit']
        for c in [0,1,4]: f[:,HCH.index(c)]=f[:,HCH.index(c)]*s+sh
        gn=gn*s+sh
        f=f+np.random.randn(*f.shape).astype(np.float32)*CFG['aug_noise']
    return f, gn, gtvt

class Enc(nn.Module):
    def __init__(s,nin,d,dils,drop):
        super().__init__(); s.inp=nn.Conv1d(nin,d,1); s.L=nn.ModuleList()
        for dl in dils:
            s.L.append(nn.Sequential(nn.Conv1d(d,d,3,padding=dl,dilation=dl),nn.GroupNorm(4,d),nn.GELU(),nn.Dropout(drop)))
    def forward(s,x):
        h=s.inp(x)
        for l in s.L: h=h+l(h)
        return h   # (B,d,Lt)

class Matcher(nn.Module):
    def __init__(s,nh,d,hd,td,drop):
        super().__init__(); s.he=Enc(nh,d,hd,drop); s.te=Enc(2,d,td,drop)
        s.beta=nn.Parameter(torch.tensor(1.0)); s.logW=nn.Parameter(torch.tensor(np.log(60.).astype('float32')))
        s.temp=CFG['temp']
    def forward(s,H,G,gtvt,last_tvt):
        # H:(1,nh,n) G:(1,2,Tg) gtvt:(Tg,) last_tvt scalar
        he=s.he(H)[0].T                # (n,d)
        te=s.te(G)[0].T                # (Tg,d)
        he=F.normalize(he,dim=1); te=F.normalize(te,dim=1)
        A=(he@te.T)/s.temp             # (n,Tg) similarity logits
        W=F.softplus(s.logW)+5.0; beta=F.softplus(s.beta)
        n=he.shape[0]
        center=torch.full((n,),float(last_tvt))   # start at flat anchor
        k=CFG['smoothk']; pad=k//2
        tvt=center
        for _ in range(CFG['iters']):
            prior=-beta*((gtvt[None,:]-center[:,None])/W)**2   # window travels with center
            w=torch.softmax(A+prior,dim=1)
            tvt=w@gtvt                 # (n,)
            # re-center on a low-pass (continuity) of the current estimate, anchored to start at last_tvt
            c=F.avg_pool1d(tvt[None,None],k,stride=1,padding=pad)[0,0][:n]
            center=c
        return tvt

dev='cpu'; net=Matcher(len(HCH),CFG['d'],CFG['hlayers'],CFG['tlayers'],CFG['drop']).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
steps=CFG['epochs']*len(tr); sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG['lr'],total_steps=steps)

def run_well(w,train=False):
    f,gn,gtvt=prep_well(w,train)
    H=torch.from_numpy(f.T[None]); G=torch.from_numpy(np.stack([gn,np.gradient(gn)])[None].astype(np.float32))
    gt=torch.from_numpy(gtvt); tvt=net(H,G,gt,float(w['last_tvt']))
    return tvt, gt

def evaluate():
    net.eval(); preds=[]
    with torch.no_grad():
        for w in va:
            tvt,_=run_well(w,False); preds.append((tvt.numpy()-w['last_tvt']))
    net.train(); return prep.pooled_rmse_eval(preds,va)

print('training %s | params=%.0fk'%(CFG['tag'],sum(p.numel() for p in net.parameters())/1e3),flush=True)
t0=time.time(); best=(99,0); order=np.arange(len(tr))
for ep in range(CFG['epochs']):
    np.random.shuffle(order); net.train(); tot=0;nb=0; opt.zero_grad()
    for cnt,j in enumerate(order):
        w=tr[j]; tvt,gt=run_well(w,True)
        true=torch.from_numpy((w['target']+w['last_tvt']).astype(np.float32))
        ev=torch.from_numpy(w['ev']); kn=1.0-ev
        m=ev+CFG['w_known']*kn
        mse=(F.smooth_l1_loss(tvt,true,reduction='none',beta=8.0)*m).sum()/(m.sum()+1e-6)
        d2=tvt[2:]-2*tvt[1:-1]+tvt[:-2]; sm=(d2*d2).mean()
        loss=(mse+CFG['smooth']*sm)/CFG['bs']
        loss.backward(); tot+=loss.item()*CFG['bs']; nb+=1
        if (cnt+1)%CFG['bs']==0:
            nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step(); opt.zero_grad()
    if (ep+1)%4==0 or ep==CFG['epochs']-1:
        r,npts=evaluate();
        if r<best[0]: best=(r,ep+1)
        print('  ep%2d loss%.3f | holdout RMSE %.3f (best %.3f@%d) beta=%.2f W=%.0f %.0fs'%(
            ep+1,tot/nb,r,best[0],best[1],F.softplus(net.beta).item(),(F.softplus(net.logW)+5).item(),time.time()-t0),flush=True)
r,_=evaluate(); print('FINAL %s holdout RMSE %.3f | best %.3f'%(CFG['tag'],r,best[0]),flush=True)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'RESULTS.jsonl'),'a').write(
    json.dumps(dict(tag=CFG['tag'],best=best[0],final=r,cfg={k:v for k,v in CFG.items()},t=time.time()-t0))+'\n')
