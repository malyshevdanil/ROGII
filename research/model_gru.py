"""Phase 2.1: sequential GRU state-estimator (neural filter).
Steps along MD from a known-tail through the eval zone. At each step the observation is the GR residual
vs the typewell sampled at the CURRENT predicted position (autoregressive, differentiable interp) ->
the net learns to nudge a BOUNDED dip step. Teacher-forced on the known tail (true position known).
Anchored at last_tvt (worst case ~flat). Ranked on the shared whole-well holdout."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, sys, os, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import prep
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

CFG=dict(hid=64, drop=0.1, wd=1e-4, lr=3e-3, epochs=30, bs=8, ktail=150, maxstep=1.2,
         offs=(-15,-8,-4,0,4,8,15), aug_caljit=0.08, aug_noise=0.04, ssamp=0.0, smooth=0.02, tag='gru_base')
for a in sys.argv[1:]:
    k,v=a.split('=')
    if k in CFG and not isinstance(CFG[k],(list,tuple)): CFG[k]=type(CFG[k])(v)
    else: CFG[k]=v
OFFS=np.array(CFG['offs'],np.float32); NOFF=len(OFFS)

data=prep.load(); tr,va=prep.split(data)
KSTEPS=40; EVSTEPS=180; SEQL=KSTEPS+EVSTEPS   # fixed resampled length (speed on CPU)
gf=np.concatenate([w['feat'] for w in tr],0); D6M,D6S=gf[:,6].mean(),gf[:,6].std()+1e-6
def rs(a,idx_src,idx_dst): return np.interp(idx_dst,idx_src,a).astype(np.float32)
def seqs(w):
    f=w['feat']; g=w['g_gr']; gm,gs=float(g.mean()),float(g.std()+1e-6)
    ev=w['ev']; ei=np.where(ev>0.5)[0]
    if len(ei)<5: return None
    e0=ei[0]; ksrc=np.arange(max(0,e0-400),e0); esrc=np.arange(e0,len(ev))
    if len(ksrc)<5: return None
    caln=(f[:,4]-gm)/gs; grad=f[:,2]/gs; rstd=f[:,3]/gs; dzdmd=(f[:,6]-D6M)/D6S
    md=f[:,7]*1000.; true=w['target']+w['last_tvt']; mkn=f[:,9]
    # resample known-tail -> KSTEPS, eval -> EVSTEPS (uniform in source index within each region)
    kd=np.linspace(ksrc[0],ksrc[-1],KSTEPS); ed=np.linspace(esrc[0],esrc[-1],EVSTEPS)
    dst=np.concatenate([kd,ed]); src=np.arange(len(caln))
    def R(a): return rs(a,src,dst)
    md_r=R(md); dstep=np.diff(md_r,prepend=md_r[0]); dstep[0]=dstep[1]
    evr=np.concatenate([np.zeros(KSTEPS),np.ones(EVSTEPS)]).astype(np.float32)
    mknr=1.0-evr
    return dict(caln=R(caln),grad=R(grad),rstd=R(rstd),dzdmd=R(dzdmd),dstep=(dstep/50.).astype(np.float32),
                true=R(true),ev=evr,mkn=mknr,gn=((g-gm)/gs).astype(np.float32),
                tw_min=float(w['g_tvt'][0]),tw_max=float(w['g_tvt'][-1]),last_tvt=float(w['last_tvt']),n=SEQL)
TR=[s for s in (seqs(w) for w in tr) if s is not None]
VA=[s for s in (seqs(w) for w in va) if s is not None]
print('resampled: train %d val %d, seq len %d'%(len(TR),len(VA),SEQL),flush=True)

def clip_seq(s):  # last ktail known steps + all eval steps
    ev=s['ev']; ei=np.where(ev>0.5)[0]
    if len(ei)==0: return None
    start=max(0, ei[0]-CFG['ktail'])
    return start
INP=NOFF+6  # obs(NOFF) + caln,grad,rstd,dzdmd,dstep,mkn

class GRUFilter(nn.Module):
    def __init__(s,inp,hid,drop):
        super().__init__(); s.cell=nn.GRUCell(inp,hid); s.drop=nn.Dropout(drop)
        s.head=nn.Sequential(nn.Linear(hid,hid),nn.GELU(),nn.Linear(hid,1))
        s.head[-1].bias.data.zero_(); s.head[-1].weight.data*=0.1
    def sample_tw(s,pos,gn,tw_min,tw_max):
        # pos:(B,) gn:(B,Tg) -> residual profile at pos+OFFS : (B,NOFF)
        B,Tg=gn.shape; dt=(tw_max-tw_min)/(Tg-1)                 # (B,)
        off=torch.from_numpy(OFFS)[None,:]                       # (1,NOFF)
        p=pos[:,None]+off                                        # (B,NOFF)
        idx=(p-tw_min[:,None])/dt[:,None]
        idx=idx.clamp(0,Tg-1-1e-4); i0=idx.floor().long(); fr=idx-i0.float()
        g0=torch.gather(gn,1,i0); g1=torch.gather(gn,1,(i0+1).clamp(max=Tg-1))
        return (1-fr)*g0+fr*g1                                   # (B,NOFF) typewell GR at pos+offs
    def forward(s,batch,train=False,ssamp=0.0):
        (caln,grad,rstd,dzdmd,dstep,true,ev,mkn,gn,tw_min,tw_max,last_tvt,PAD,L)=batch
        B=caln.shape[0]; h=torch.zeros(B,s.cell.hidden_size)
        pos=last_tvt.clone(); preds=[]
        for t in range(L):
            tw_at=s.sample_tw(pos,gn,tw_min,tw_max)              # (B,NOFF)
            obs=tw_at-caln[:,t:t+1]                              # residual profile (B,NOFF)
            x=torch.cat([obs,caln[:,t:t+1],grad[:,t:t+1],rstd[:,t:t+1],dzdmd[:,t:t+1],dstep[:,t:t+1],mkn[:,t:t+1]],1)
            h=s.cell(x,h); h=s.drop(h)
            step=torch.tanh(s.head(h)[:,0])*CFG['maxstep']
            newpos=pos+step*PAD[:,t]
            preds.append(newpos)
            # teacher forcing on known steps (mkn=1): use true position next; on eval use own (with optional scheduled sampling)
            tf=mkn[:,t]
            if train and ssamp>0:
                tf=tf*(torch.rand(B)<1.0).float()  # keep known TF; eval always own
            pos=tf*true[:,t]+(1-tf)*newpos
        return torch.stack(preds,1)                              # (B,L) predicted TVT

def make_batch(wells):
    starts=[clip_seq(s) for s in wells]
    segs=[];
    for s,st in zip(wells,starts):
        sl=slice(st,s['n']); segs.append({k:(s[k][sl] if isinstance(s[k],np.ndarray) and s[k].ndim==1 and len(s[k])==s['n'] else s[k]) for k in s})
    L=max(len(x['caln']) for x in segs); B=len(segs)
    def pad(key):
        A=np.zeros((B,L),np.float32)
        for i,x in enumerate(segs): A[i,:len(x[key])]=x[key]
        return torch.from_numpy(A)
    PAD=np.zeros((B,L),np.float32)
    for i,x in enumerate(segs): PAD[i,:len(x['caln'])]=1.0
    Tg=len(wells[0]['gn'])
    gn=np.zeros((B,Tg),np.float32)
    for i,x in enumerate(segs):
        gg=x['gn']; gn[i,:len(gg)]=gg[:Tg] if len(gg)>=Tg else np.pad(gg,(0,Tg-len(gg)))
    return (pad('caln'),pad('grad'),pad('rstd'),pad('dzdmd'),pad('dstep'),pad('true'),pad('ev'),pad('mkn'),
            torch.from_numpy(gn),torch.tensor([x['tw_min'] for x in segs]),torch.tensor([x['tw_max'] for x in segs]),
            torch.tensor([x['last_tvt'] for x in segs]),torch.from_numpy(PAD),L), segs

net=GRUFilter(INP,CFG['hid'],CFG['drop'])
opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG['lr'],total_steps=CFG['epochs']*((len(TR)+CFG['bs']-1)//CFG['bs']))

def evaluate():
    net.eval(); preds=[]
    with torch.no_grad():
        for i in range(0,len(VA),CFG['bs']):
            wells=VA[i:i+CFG['bs']]; batch,segs=make_batch(wells)
            out=net(batch,train=False)
            for b,x in enumerate(segs):
                n=len(x['caln']); p=out[b,:n].numpy()
                # map back to full-well: only eval positions matter; build delta vs last_tvt on eval steps
                ev=x['ev']>0.5; preds.append((p[ev]-x['last_tvt'], x['true'][ev]-x['last_tvt']))
    net.train()
    e2=np.concatenate([(a-b)**2 for a,b in preds]); return float(np.sqrt(e2.mean()))

print('training %s | params=%.0fk INP=%d'%(CFG['tag'],sum(p.numel() for p in net.parameters())/1e3,INP),flush=True)
t0=time.time(); best=(99,0); order=np.arange(len(TR))
for ep in range(CFG['epochs']):
    np.random.shuffle(order); net.train(); tot=0;nb=0
    for i in range(0,len(TR),CFG['bs']):
        wells=[TR[j] for j in order[i:i+CFG['bs']]]
        # light augmentation on GR channels
        if CFG['aug_caljit']>0 or CFG['aug_noise']>0:
            wells=[dict(w) for w in wells]
            for w in wells:
                sca=1.0+np.random.randn()*CFG['aug_caljit']; sh=np.random.randn()*CFG['aug_caljit']
                w['caln']=w['caln']*sca+sh+np.random.randn(len(w['caln'])).astype(np.float32)*CFG['aug_noise']
                w['gn']=w['gn']*sca+sh
        batch,segs=make_batch(wells)
        out=net(batch,train=True,ssamp=CFG['ssamp'])
        true=batch[5]; ev=batch[6]; mkn=batch[7]; PAD=batch[12]
        m=(ev+0.3*mkn)*PAD
        loss=(F.smooth_l1_loss(out,true,reduction='none',beta=8.0)*m).sum()/(m.sum()+1e-6)
        d2=out[:,2:]-2*out[:,1:-1]+out[:,:-2]; loss=loss+CFG['smooth']*(d2*d2*PAD[:,2:]).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step()
        tot+=loss.item(); nb+=1
    if (ep+1)%3==0 or ep==CFG['epochs']-1:
        r=evaluate();
        if r<best[0]: best=(r,ep+1)
        print('  ep%2d loss%.3f | holdout RMSE %.3f (best %.3f@%d) %.0fs'%(ep+1,tot/nb,r,best[0],best[1],time.time()-t0),flush=True)
r=evaluate(); print('FINAL %s holdout RMSE %.3f | best %.3f'%(CFG['tag'],r,best[0]),flush=True)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'RESULTS.jsonl'),'a').write(
    json.dumps(dict(tag=CFG['tag'],best=best[0],final=r,cfg={k:v for k,v in CFG.items()},t=time.time()-t0))+'\n')
