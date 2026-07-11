"""Phase 2.2: differentiable GRID Bayesian filter (neural PF) — on GPU.
Full posterior over 256 typewell positions (multi-hypothesis -> won't collapse to flat, models bimodality).
predict: diffuse belief via a learned continuity transition (conv). update: multiply by LEARNED GR
emission (CNN window-match), renormalize (can't diverge). Output = posterior-mean TVT. Trained end-to-end.
This is the neural analog of the PF that gets 7.096. Ranked on the shared whole-well holdout."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, sys, os, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import prep
torch.manual_seed(0); np.random.seed(0)
DEV='cuda' if torch.cuda.is_available() else 'cpu'

CFG=dict(d=64, KSTEPS=60, EVSTEPS=360, drop=0.1, wd=1e-4, lr=2e-3, epochs=50, bs=10,
         emis_temp=2.0, trans_w=6.0, aug_caljit=0.08, aug_noise=0.05, smooth=0.0, w_known=0.3, tag='pfgrid')
for a in sys.argv[1:]:
    k,v=a.split('=')
    if k in CFG and not isinstance(CFG[k],(list,tuple)): CFG[k]=type(CFG[k])(v)
KSTEPS=CFG['KSTEPS']; EVSTEPS=CFG['EVSTEPS']; SEQL=KSTEPS+EVSTEPS; TG=prep.TWGRID
data=prep.load(); tr,va=prep.split(data)
gf=np.concatenate([w['feat'] for w in tr],0); D6M,D6S=gf[:,6].mean(),gf[:,6].std()+1e-6

def seqs(w):
    f=w['feat']; g=w['g_gr']; gm,gs=float(g.mean()),float(g.std()+1e-6)
    ev=w['ev']; ei=np.where(ev>0.5)[0]
    if len(ei)<5: return None
    e0=ei[0]; ksrc=np.arange(max(0,e0-400),e0)
    if len(ksrc)<5: return None
    caln=(f[:,4]-gm)/gs; grad=f[:,2]/gs; rstd=f[:,3]/gs; dzdmd=(f[:,6]-D6M)/D6S
    true=w['target']+w['last_tvt']
    kd=np.linspace(ksrc[0],ksrc[-1],KSTEPS); ed=np.linspace(e0,len(ev)-1,EVSTEPS); dst=np.concatenate([kd,ed])
    src=np.arange(len(caln)); R=lambda a: np.interp(dst,src,a).astype(np.float32)
    gtvt=w['g_tvt'].astype(np.float32)
    # true position index on the typewell grid (nearest), for supervision & init
    tvt_r=R(true); j_true=np.clip(np.round((tvt_r-gtvt[0])/(gtvt[-1]-gtvt[0])*(TG-1)),0,TG-1).astype(np.int64)
    last_j=int(np.clip(round((w['last_tvt']-gtvt[0])/(gtvt[-1]-gtvt[0])*(TG-1)),0,TG-1))
    evr=np.concatenate([np.zeros(KSTEPS),np.ones(EVSTEPS)]).astype(np.float32)
    return dict(H=np.stack([R(caln),R(grad),R(rstd),R(dzdmd)]).astype(np.float32),   # (4,L)
                gn=((g-gm)/gs).astype(np.float32), gtvt=gtvt, tvt=tvt_r, jtrue=j_true,
                last_j=last_j, ev=evr, last_tvt=float(w['last_tvt']))
TR=[s for s in (seqs(w) for w in tr) if s]; VA=[s for s in (seqs(w) for w in va) if s]
print('device=%s | train %d val %d | seq %d grid %d'%(DEV,len(TR),len(VA),SEQL,TG),flush=True)

class Enc(nn.Module):
    def __init__(s,nin,d,drop):
        super().__init__()
        s.inp=nn.Conv1d(nin,d,5,padding=2)
        s.blocks=nn.ModuleList([nn.Sequential(
            nn.Conv1d(d,d,3,padding=dl,dilation=dl),nn.GroupNorm(8,d),nn.GELU(),nn.Dropout(drop),
            nn.Conv1d(d,d,3,padding=dl,dilation=dl),nn.GroupNorm(8,d),nn.GELU()) for dl in (1,2,4,8)])
        s.out=nn.Conv1d(d,d,1)
    def forward(s,x):
        h=F.gelu(s.inp(x))
        for b in s.blocks: h=h+b(h)
        return s.out(h)   # (B,d,L)

class PFGrid(nn.Module):
    def __init__(s,d,drop):
        super().__init__(); s.he=Enc(4,d,drop); s.te=Enc(1,d,drop)
        s.log_transw=nn.Parameter(torch.tensor(np.log(CFG['trans_w']).astype('float32')))
        # learnable emission gain, init small -> starts ~continuity-only (>=flat, never worse), ramps where GR helps
        s.log_gain=nn.Parameter(torch.tensor(np.log(np.expm1(0.12)).astype('float32')))
        # DRIFT head: per-step directional shift of the belief (velocity/dip) — without this the filter can
        # only diffuse symmetrically and can't recover the per-well slope (caps at flat, not the line oracle 6.6).
        s.drift_head=nn.Sequential(nn.Conv1d(d,d,1),nn.GELU(),nn.Conv1d(d,1,1))
        s.maxdrift=CFG.get('maxdrift',3.0)
        s.emis_temp=CFG['emis_temp']
        s.register_buffer('koff',torch.arange(-40,41).float())   # transition-kernel offsets
    def shift(s,p,d):
        # shift belief p:(B,TG) by d grid-cells (B,), differentiable linear interp
        B,TG=p.shape; ar=torch.arange(TG,device=p.device).float()[None,:]
        idx=ar-d[:,None]; i0=idx.floor(); fr=(idx-i0)
        i0l=i0.long(); v0=((i0l>=0)&(i0l<TG)).float(); v1=((i0l+1>=0)&(i0l+1<TG)).float()
        g0=torch.gather(p,1,i0l.clamp(0,TG-1)); g1=torch.gather(p,1,(i0l+1).clamp(0,TG-1))
        return g0*(1-fr)*v0+g1*fr*v1
    def transition(s,p):
        # diffuse belief p:(B,TG) with a Gaussian kernel (continuity); width learnable
        w=F.softplus(s.log_transw)+1.0
        k=torch.exp(-0.5*(s.koff/w)**2); k=k/k.sum()
        return F.conv1d(p[:,None],k[None,None],padding=len(s.koff)//2)[:,0]
    def forward(s,H,G,jinit):
        B,_,L=H.shape
        he_raw=s.he(H)                         # (B,d,L)
        he=F.normalize(he_raw,dim=1)
        te=F.normalize(s.te(G),dim=1)          # (B,d,TG)
        E=torch.einsum('bdl,bdt->blt',he,te)/s.emis_temp   # (B,L,TG) emission logits
        E=F.softplus(s.log_gain)*E                         # learnable gain (init small = start near continuity-only)
        Eexp=torch.exp(E-E.max(dim=2,keepdim=True).values)
        drift=torch.tanh(s.drift_head(he_raw)[:,0])*s.maxdrift   # (B,L) per-step directional shift (velocity)
        # init belief peaked at last known position
        p=torch.zeros(B,TG,device=H.device); p[torch.arange(B),jinit]=1.0
        p=s.transition(p)+1e-8; p=p/p.sum(1,keepdim=True)
        outs=[]
        for t in range(L):
            p=s.transition(p)                  # predict: diffuse (continuity)
            p=s.shift(p,drift[:,t])+1e-8       # predict: drift (velocity/dip) -- recovers the per-well slope
            p=p/p.sum(1,keepdim=True)
            p=p*Eexp[:,t]+1e-8                 # update (GR likelihood)
            p=p/p.sum(1,keepdim=True)          # normalize
            outs.append(p)
        P=torch.stack(outs,1)                  # (B,L,TG)
        return P

net=PFGrid(CFG['d'],CFG['drop']).to(DEV)
opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG['lr'],total_steps=CFG['epochs']*((len(TR)+CFG['bs']-1)//CFG['bs']))

def make_batch(wells,train=False):
    B=len(wells)
    H=np.stack([w['H'] for w in wells]); gn=np.stack([w['gn'][:TG] for w in wells])
    if train:
        sca=1+np.random.randn(B,1,1)*CFG['aug_caljit']; sh=np.random.randn(B,1,1)*CFG['aug_caljit']
        H[:,0:1]=H[:,0:1]*sca+sh; gn=gn*sca[:,:,0]+sh[:,:,0]
        H=H+np.random.randn(*H.shape)*CFG['aug_noise']
    H=torch.tensor(H,dtype=torch.float32,device=DEV)
    G=torch.tensor(gn,dtype=torch.float32,device=DEV)[:,None]
    gtvt=torch.tensor(np.stack([w['gtvt'][:TG] for w in wells]),device=DEV)
    jinit=torch.tensor([w['last_j'] for w in wells],device=DEV)
    jtrue=torch.tensor(np.stack([w['jtrue'] for w in wells]),device=DEV)
    tvt=torch.tensor(np.stack([w['tvt'] for w in wells]),device=DEV)
    ev=torch.tensor(np.stack([w['ev'] for w in wells]),device=DEV)
    return H,G,gtvt,jinit,jtrue,tvt,ev

def evaluate():
    net.eval(); errs=[]
    with torch.no_grad():
        for i in range(0,len(VA),CFG['bs']):
            wells=VA[i:i+CFG['bs']]; H,G,gtvt,jinit,jtrue,tvt,ev=make_batch(wells)
            P=net(H,G,jinit); pred=(P*gtvt[:,None]).sum(2)      # posterior-mean TVT (B,L)
            m=ev>0.5
            errs.append(((pred-tvt)[m]).cpu().numpy())
    net.train(); e=np.concatenate(errs); return float(np.sqrt((e**2).mean()))

print('training %s | params=%.0fk'%(CFG['tag'],sum(p.numel() for p in net.parameters())/1e3),flush=True)
t0=time.time(); best=(99,0); order=np.arange(len(TR))
for ep in range(CFG['epochs']):
    np.random.shuffle(order); net.train(); tot=0;nb=0
    for i in range(0,len(TR),CFG['bs']):
        wells=[TR[j] for j in order[i:i+CFG['bs']]]; H,G,gtvt,jinit,jtrue,tvt,ev=make_batch(wells,True)
        P=net(H,G,jinit); pred=(P*gtvt[:,None]).sum(2)
        logp=torch.log(P.clamp_min(1e-8))
        nll=-logp.gather(2,jtrue[:,:,None]).squeeze(2)          # -log posterior at true pos
        m=ev+CFG['w_known']*(1-ev)
        loss=(nll*m).sum()/m.sum() + 0.01*(F.smooth_l1_loss(pred,tvt,reduction='none',beta=8.0)*m).sum()/m.sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step()
        tot+=loss.item(); nb+=1
    if (ep+1)%3==0 or ep==CFG['epochs']-1:
        r=evaluate();
        if r<best[0]: best=(r,ep+1)
        print('  ep%2d loss%.3f | holdout RMSE %.3f (best %.3f@%d) transW=%.1f gain=%.2f %.0fs'%(
            ep+1,tot/nb,r,best[0],best[1],(F.softplus(net.log_transw)+1).item(),F.softplus(net.log_gain).item(),time.time()-t0),flush=True)
r=evaluate(); print('FINAL %s holdout RMSE %.3f | best %.3f'%(CFG['tag'],r,best[0]),flush=True)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'RESULTS.jsonl'),'a').write(
    json.dumps(dict(tag=CFG['tag'],best=best[0],final=r,cfg=CFG,t=time.time()-t0,dev=DEV))+'\n')
