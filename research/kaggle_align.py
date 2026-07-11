"""ROGII — self-contained GPU training (Kaggle-ready). WARP model (stable, replaces the dead grid-PF).
Predicts per-point dTVT (a DERIVATIVE), integrates via cumsum anchored at the last known TVT -> well-
conditioned (fixes the flat-collapse of absolute regression) and can't diverge (bounded steps). Reads
the typewell as a 'ruler' via cross-attention. Ranked on a whole-well holdout (pooled RMSE).
Kaggle: Accelerator=GPU T4 x2 (NOT P100), Internet off, only the competition dataset."""
import numpy as np, pandas as pd, glob, os, time, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV='cuda' if torch.cuda.is_available() else 'cpu'
CFG=dict(d=128, KSTEPS=60, EVSTEPS=360, TWGRID=256, drop=0.2, wd=3e-4, lr=1.2e-3, epochs=140, bs=24,
         maxstep=1.5, lp_k=41, w_mono=0.02, w_ent=0.005, w_smooth=0.05,
         aug_caljit=0.12, aug_noise=0.07, aug_warp=0.35, aug_twjit=0.06,
         w_known=0.5, n_val=160, seed=42, tag='hybrid')
TG=CFG['TWGRID']; KSTEPS=CFG['KSTEPS']; EVSTEPS=CFG['EVSTEPS']; SEQL=KSTEPS+EVSTEPS
c=glob.glob('/kaggle/input/**/*__horizontal_well.csv',recursive=True)
_tr=[p for p in c if 'train' in p.lower()]; c=_tr if _tr else c
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
    a,b=(np.polyfit(kn_gr[v],tw_at_k[v],1) if v.sum()>=20 else (1.,0.)); cal_gr=gr*a+b
    Z=hw['Z'].values.astype(float); MD=hw['MD'].values.astype(float); mdd=np.gradient(MD); mdd[mdd==0]=1
    grad=np.gradient(gr); rstd=pd.Series(gr).rolling(21,center=True,min_periods=1).std().fillna(0).values
    dzdmd=np.gradient(Z)/mdd; md_since=(MD-float(last['MD']))
    ev=hw['TVT_input'].isna().values.astype(float); ei=np.where(ev>0.5)[0]
    if len(ei)<5: return None
    e0=ei[0]; ksrc=np.arange(max(0,e0-400),e0)
    if len(ksrc)<5: return None
    g_tvt=np.linspace(tw_tvt.min(),tw_tvt.max(),TG); g_gr=np.interp(g_tvt,tw_tvt,tw_gr)
    gm,gs=float(g_gr.mean()),float(g_gr.std()+1e-6); caln=(cal_gr-gm)/gs; gradn=grad/gs; rstdn=rstd/gs
    true=hw['TVT'].values.astype(float)
    kd=np.linspace(ksrc[0],ksrc[-1],KSTEPS); ed=np.linspace(e0,n-1,EVSTEPS); dst=np.concatenate([kd,ed]); src=np.arange(n)
    R=lambda x: np.interp(dst,src,x).astype(np.float32)
    evr=np.concatenate([np.zeros(KSTEPS),np.ones(EVSTEPS)]).astype(np.float32)
    return dict(H=np.stack([R(caln),R(gradn),R(rstdn),R(dzdmd)]).astype(np.float32),
                gn=((g_gr-gm)/gs).astype(np.float32), g_tvt=g_tvt.astype(np.float32),
                tvt=R(true), ev=evr, last_tvt=last_tvt)
print('building features...',flush=True); t0=time.time()
wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
DATA=[b for b in (build_well(w) for w in wids) if b is not None]
d3=np.concatenate([w['H'][3] for w in DATA]); m3,s3=d3.mean(),d3.std()+1e-6
for w in DATA: w['H'][3]=(w['H'][3]-m3)/s3
rng=np.random.default_rng(CFG['seed']); idx=np.arange(len(DATA)); rng.shuffle(idx)
VA=[DATA[i] for i in idx[:CFG['n_val']]]; TR=[DATA[i] for i in idx[CFG['n_val']:]]
print('built %d wells (%d/%d) %.0fs'%(len(DATA),len(TR),len(VA),time.time()-t0),flush=True)
def prmse(sqs): return float(np.sqrt(np.mean(np.concatenate(sqs))))
flat=[(w['last_tvt']-w['tvt'][w['ev']>0.5])**2 for w in VA]
line=[]
for w in VA:
    m=w['ev']>0.5; x=np.arange(m.sum()); y=w['tvt'][m]; A=np.polyfit(x,y,1); line.append((np.polyval(A,x)-y)**2)
print('BASELINES flat=%.3f line-oracle=%.3f (target: beat line -> wiggle ~5)'%(prmse(flat),prmse(line)),flush=True)

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
class HybridNet(nn.Module):
    """WARP (continuity, ~11 but drifts) + ALIGNMENT (absolute global anchor, GR-matched but noisy ~30).
    final = warp corrected toward the LOW-PASSED alignment -> continuity keeps local accuracy, the
    low-passed absolute anchor kills warp's long-range drift. Aims to beat both (11 and 30)."""
    def __init__(s,d,drop):
        super().__init__(); s.he=Enc(4,d,drop); s.te=Enc(1,d,drop)
        s.q=nn.Conv1d(d,d,1); s.k=nn.Conv1d(d,d,1)
        s.log_temp=nn.Parameter(torch.tensor(np.log(10.).astype('float32')))
        s.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1))
        s.head[-1].weight.data*=0.01; s.head[-1].bias.data.zero_()
        s.gate=nn.Parameter(torch.tensor(-4.0))               # sigmoid(-4)~0.02: start ~pure WARP, ramp anchor as align trains
    def forward(s,H,G,g_tvt,last_tvt):
        h=s.he(H); t=s.te(G)                                  # (B,d,L),(B,d,TG)
        Q=F.normalize(s.q(h).transpose(1,2),dim=2); K=F.normalize(s.k(t).transpose(1,2),dim=2)
        M=(Q@K.transpose(1,2))*F.softplus(s.log_temp); A=torch.softmax(M,dim=2)   # (B,L,TG)
        idx=torch.arange(A.shape[2],device=A.device).float(); pos=A@idx
        align_tvt=(A*g_tvt[:,None,:]).sum(2)                  # (B,L) absolute aligned TVT (noisy)
        ctx=A@t.transpose(1,2)                                # (B,L,d) aligned typewell context
        dtvt=torch.tanh(s.head(torch.cat([h.transpose(1,2),ctx],2))[...,0])*CFG['maxstep']
        warp=torch.cumsum(dtvt,1); warp=warp-warp[:,KSTEPS-1:KSTEPS]+last_tvt[:,None]   # continuity, anchored
        k=CFG['lp_k']; align_lp=F.avg_pool1d(align_tvt[:,None],k,1,k//2)[:,0]           # low-pass anchor (kills ~30 noise)
        final=warp+torch.sigmoid(s.gate)*(align_lp-warp)      # correct warp drift toward global anchor
        return final,pos,A
net=HybridNet(CFG['d'],CFG['drop']).to(DEV)
opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG['lr'],total_steps=CFG['epochs']*((len(TR)+CFG['bs']-1)//CFG['bs']))
def batch(ws,train=False):
    H=np.stack([w['H'] for w in ws]).copy(); gn=np.stack([w['gn'][:TG] for w in ws]).copy()
    tvt=np.stack([w['tvt'] for w in ws]).copy(); ev=np.stack([w['ev'] for w in ws])
    if train:
        # GR calibration jitter + noise (shape/scale robustness)
        sc=1+np.random.randn(len(ws),1,1)*CFG['aug_caljit']; sh=np.random.randn(len(ws),1,1)*CFG['aug_caljit']
        H[:,0:1]=H[:,0:1]*sc+sh; gn=gn*sc[:,:,0]+sh[:,:,0]; H=H+np.random.randn(*H.shape)*CFG['aug_noise']
        gn=gn+np.random.randn(*gn.shape)*CFG['aug_twjit']                     # typewell-ruler jitter
        # TIME-WARP the eval region (+target) -> learn RELATIVE alignment, not memorized positions
        if CFG.get('aug_warp',0)>0:
            m=EVSTEPS; src=np.arange(m)
            for bi in range(len(ws)):
                d=np.cumsum(np.abs(1+np.random.randn(m)*CFG['aug_warp'])); d=(d-d[0])/(d[-1]-d[0])*(m-1)
                for ch in range(H.shape[1]): H[bi,ch,KSTEPS:]=np.interp(d,src,H[bi,ch,KSTEPS:])
                tvt[bi,KSTEPS:]=np.interp(d,src,tvt[bi,KSTEPS:])
    to=lambda a: torch.tensor(a,dtype=torch.float32,device=DEV)
    return (to(H),to(gn)[:,None],to(np.stack([w['g_tvt'][:TG] for w in ws])),
            to(np.array([w['last_tvt'] for w in ws])),to(tvt),to(ev))
def evaluate():
    net.eval(); e=[]
    with torch.no_grad():
        for i in range(0,len(VA),CFG['bs']):
            H,G,gtvt,lt,tvt,ev=batch(VA[i:i+CFG['bs']]); pred,_,_=net(H,G,gtvt,lt); e.append(((pred-tvt)[ev>0.5]).cpu().numpy())
    net.train(); return float(np.sqrt((np.concatenate(e)**2).mean()))
print('training HYBRID | params=%.0fk'%(sum(p.numel() for p in net.parameters())/1e3),flush=True)
t0=time.time(); best=(99,0); order=np.arange(len(TR))
for ep in range(CFG['epochs']):
    np.random.shuffle(order); tot=0;nb=0
    for i in range(0,len(TR),CFG['bs']):
        H,G,gtvt,lt,tvt,ev=batch([TR[j] for j in order[i:i+CFG['bs']]],True); pred,pos,A=net(H,G,gtvt,lt)
        m=ev+CFG['w_known']*(1-ev)
        loss=(F.smooth_l1_loss(pred,tvt,reduction='none',beta=8.0)*m).sum()/m.sum()
        mono=F.relu(pos[:,:-1]-pos[:,1:])                              # monotone alignment
        loss=loss+CFG['w_mono']*(mono*m[:,1:]).sum()/m[:,1:].sum()
        ent=-(A*torch.log(A+1e-8)).sum(2)                             # sharpen alignment -> localize
        loss=loss+CFG['w_ent']*(ent*m).sum()/m.sum()
        d2=pred[:,2:]-2*pred[:,1:-1]+pred[:,:-2]                       # keep final smooth/stable
        loss=loss+CFG['w_smooth']*(d2*d2*m[:,2:]).sum()/m[:,2:].sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step()
        tot+=loss.item(); nb+=1
    if (ep+1)%2==0 or ep==CFG['epochs']-1:
        r=evaluate()
        if r<best[0]: best=(r,ep+1); torch.save(net.state_dict(),'best_warp.pt')
        print('  ep%2d loss%.3f | holdout pooled RMSE %.3f (best %.3f@%d) %.0fs'%(ep+1,tot/nb,r,best[0],best[1],time.time()-t0),flush=True)
print('DONE best %.3f @ep%d (flat~15, line~6.6, target<6.6)'%(best[0],best[1]),flush=True)
