"""ROGII — MDN approach (per leaders' strategy). TCN + cross-attention to typewell + Mixture-Density head.
Instead of MSE (collapses to the mean between the ±15ft Eagle Ford modes -> flat), predict K Gaussian
components per point (means/vars/weights) and train NLL. Inference = VITERBI over the K modes with a
continuity transition -> coherent mode-selected trajectory (no mean-collapse). Reduced features + strong
aug. Kaggle: GPU T4 x2 (NOT P100), Internet off, competition dataset only."""
import numpy as np, pandas as pd, glob, os, time, copy, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0); DEV='cuda' if torch.cuda.is_available() else 'cpu'
CFG=dict(d=96, K=3, KSTEPS=60, EVSTEPS=360, TWGRID=256, drop=0.15, wd=3e-4, lr=1.5e-3, epochs=120, bs=24,
         max_off=90., min_sig=2., trans=0.02, aug_caljit=0.12, aug_noise=0.07, aug_warp=0.35, aug_hflip=0.5,
         w_known=0.5, ema_decay=0.995, n_val=160, seed=42, tag='mdn')
TG=CFG['TWGRID']; KSTEPS=CFG['KSTEPS']; EVSTEPS=CFG['EVSTEPS']; K=CFG['K']
c=glob.glob('/kaggle/input/**/*__horizontal_well.csv',recursive=True); _t=[p for p in c if 'train' in p.lower()]; c=_t if _t else c
TRAIN_DIR=os.path.dirname(c[0]) if c else 'data/train'; print('DEV=%s data=%s'%(DEV,TRAIN_DIR),flush=True)
def inan(a):
    a=a.copy();m=np.isnan(a);i=np.arange(len(a))
    if m.all():return np.zeros(len(a))
    a[m]=np.interp(i[m],i[~m],a[~m]);return a
def build(wid):
    hw=pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv'); tw=pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tt=tw['TVT'].values.astype(float); tg=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    if len(tt)<10: return None
    kn=hw[hw['TVT_input'].notna()]
    if len(kn)<20 or hw['TVT_input'].isna().sum()<20: return None
    last=kn.iloc[-1]; last_tvt=float(last['TVT_input'])
    n=len(hw); gr=inan(hw['GR'].values.astype(float)); kg=kn['GR'].interpolate().bfill().ffill().values.astype(float)
    twk=np.interp(kn['TVT_input'].values,tt,tg); v=np.isfinite(kg)&np.isfinite(twk); a,b=(np.polyfit(kg[v],twk[v],1) if v.sum()>=20 else (1.,0.)); cg=gr*a+b
    Z=hw['Z'].values.astype(float); MD=hw['MD'].values.astype(float); mdd=np.gradient(MD); mdd[mdd==0]=1; dzdmd=np.gradient(Z)/mdd
    ev=hw['TVT_input'].isna().values.astype(float); ei=np.where(ev>0.5)[0]
    if len(ei)<5: return None
    e0=ei[0]; ks=np.arange(max(0,e0-400),e0)
    if len(ks)<5: return None
    g_tvt=np.linspace(tt.min(),tt.max(),TG); g_gr=np.interp(g_tvt,tt,tg); gm,gs=float(g_gr.mean()),float(g_gr.std()+1e-6)
    caln=(cg-gm)/gs; true=hw['TVT'].values.astype(float)
    kd=np.linspace(ks[0],ks[-1],KSTEPS); ed=np.linspace(e0,n-1,EVSTEPS); dst=np.concatenate([kd,ed]); src=np.arange(n)
    R=lambda x: np.interp(dst,src,x).astype(np.float32); evr=np.concatenate([np.zeros(KSTEPS),np.ones(EVSTEPS)]).astype(np.float32)
    # REDUCED features (leaders' advice): calibrated GR, its gradient, dZ/dMD  (3 channels)
    return dict(H=np.stack([R(caln),R(np.gradient(caln)),R(dzdmd)]).astype(np.float32),
                gn=((g_gr-gm)/gs).astype(np.float32), tvt=R(true), ev=evr, last_tvt=last_tvt)
print('building...',flush=True); t0=time.time()
wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
DATA=[b for b in (build(w) for w in wids) if b is not None]
d2=np.concatenate([w['H'][2] for w in DATA]); mm,ss=d2.mean(),d2.std()+1e-6
for w in DATA: w['H'][2]=(w['H'][2]-mm)/ss
rng=np.random.default_rng(CFG['seed']); idx=np.arange(len(DATA)); rng.shuffle(idx)
VA=[DATA[i] for i in idx[:CFG['n_val']]]; TR=[DATA[i] for i in idx[CFG['n_val']:]]
print('built %d (%d/%d) %.0fs'%(len(DATA),len(TR),len(VA),time.time()-t0),flush=True)
def prmse(s): return float(np.sqrt(np.mean(np.concatenate(s))))
print('BASELINES flat=%.3f'%prmse([(w['last_tvt']-w['tvt'][w['ev']>0.5])**2 for w in VA]),flush=True)

class Enc(nn.Module):
    def __init__(s,nin,d,drop):
        super().__init__(); s.inp=nn.Conv1d(nin,d,5,padding=2)
        s.bl=nn.ModuleList([nn.Sequential(nn.Conv1d(d,d,3,padding=dl,dilation=dl),nn.GroupNorm(8,d),nn.GELU(),nn.Dropout(drop),
            nn.Conv1d(d,d,3,padding=dl,dilation=dl),nn.GroupNorm(8,d),nn.GELU()) for dl in (1,2,4,8,16)]); s.o=nn.Conv1d(d,d,1)
    def forward(s,x):
        h=F.gelu(s.inp(x))
        for b in s.bl: h=h+b(h)
        return s.o(h)
class MDN(nn.Module):
    def __init__(s,d,drop):
        super().__init__(); s.he=Enc(3,d,drop); s.te=Enc(1,d,drop); s.q=nn.Conv1d(d,d,1); s.k=nn.Conv1d(d,d,1); s.vv=nn.Conv1d(d,d,1); s.sc=d**-0.5
        s.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,3*K))   # K*(off,logsig,logit)
        s.head[-1].weight.data*=0.01; s.head[-1].bias.data.zero_()
    def forward(s,H,G,last_tvt):
        h=s.he(H); t=s.te(G); Q=s.q(h).transpose(1,2); Kk=s.k(t).transpose(1,2); V=s.vv(t).transpose(1,2)
        ctx=torch.softmax(Q@Kk.transpose(1,2)*s.sc,2)@V; x=torch.cat([h.transpose(1,2),ctx],2)
        o=s.head(x); B,L,_=o.shape; o=o.view(B,L,K,3)
        base=torch.linspace(-30.,30.,K,device=o.device)[None,None,:]       # break symmetry: modes at distinct offsets
        mu=last_tvt[:,None,None]+base+torch.tanh(o[...,0])*CFG['max_off']   # (B,L,K)
        sig=CFG['min_sig']+(18.-CFG['min_sig'])*torch.sigmoid(o[...,1])     # CAP variance (prevent inflation-collapse)
        logpi=F.log_softmax(o[...,2],dim=2)
        return mu,sig,logpi
def mdn_nll(mu,sig,logpi,y):   # y:(B,L)
    z=(y[...,None]-mu)/sig; comp=logpi-0.5*z*z-torch.log(sig)-0.9189
    return -torch.logsumexp(comp,dim=2)                                    # (B,L)
def viterbi(mu,sig,logpi,trans):  # numpy, per well: pick coherent mode path
    L,Kk=mu.shape; em=logpi                                                # (L,K) prefer high weight
    D=em[0].copy(); bp=np.zeros((L,Kk),np.int32)
    for i in range(1,L):
        # transition cost = -(mu jump)^2 * trans (prefer smooth)
        tc=-trans*(mu[i][None,:]-mu[i-1][:,None])**2                       # (Kprev,Kcur)
        tot=D[:,None]+tc; bp[i]=np.argmax(tot,0); D=em[i]+tot[bp[i],np.arange(Kk)]
    path=np.zeros(L,np.int32); path[-1]=np.argmax(D)
    for i in range(L-1,0,-1): path[i-1]=bp[i,path[i]]
    return mu[np.arange(L),path]
def make(seed):
    torch.manual_seed(seed); net=MDN(CFG['d'],CFG['drop']).to(DEV)
    opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG['lr'],total_steps=CFG['epochs']*((len(TR)+CFG['bs']-1)//CFG['bs']))
    return net,opt,sch
def batch(ws,train=False):
    H=np.stack([w['H'] for w in ws]).copy(); gn=np.stack([w['gn'][:TG] for w in ws]).copy()
    tvt=np.stack([w['tvt'] for w in ws]).copy(); ev=np.stack([w['ev'] for w in ws])
    if train:
        sc=1+np.random.randn(len(ws),1,1)*CFG['aug_caljit']; sh=np.random.randn(len(ws),1,1)*CFG['aug_caljit']
        H[:,0:1]=H[:,0:1]*sc+sh; gn=gn*sc[:,:,0]+sh[:,:,0]; H=H+np.random.randn(*H.shape)*CFG['aug_noise']
        if CFG['aug_warp']>0:
            m=EVSTEPS; src=np.arange(m)
            for bi in range(len(ws)):
                d=np.cumsum(np.abs(1+np.random.randn(m)*CFG['aug_warp'])); d=(d-d[0])/(d[-1]-d[0])*(m-1)
                for ch in range(H.shape[1]): H[bi,ch,KSTEPS:]=np.interp(d,src,H[bi,ch,KSTEPS:])
                tvt[bi,KSTEPS:]=np.interp(d,src,tvt[bi,KSTEPS:])
    to=lambda a: torch.tensor(a,dtype=torch.float32,device=DEV)
    return to(H),to(gn)[:,None],to(np.array([w['last_tvt'] for w in ws])),to(tvt),to(ev)
def evaluate(net):
    net.eval(); e=[]
    with torch.no_grad():
        for i in range(0,len(VA),CFG['bs']):
            ws=VA[i:i+CFG['bs']]; H,G,lt,tvt,ev=batch(ws); mu,sig,lp=net(H,G,lt)
            mu=mu.cpu().numpy(); lp=lp.cpu().numpy(); sg=sig.cpu().numpy(); tv=tvt.cpu().numpy(); em=ev.cpu().numpy()>0.5
            for b in range(len(ws)):
                path=viterbi(mu[b],sg[b],lp[b],CFG['trans']); e.append((path[em[b]]-tv[b][em[b]])**2)
    net.train(); return float(np.sqrt(np.mean(np.concatenate(e))))
net,opt,sch=make(CFG['seed']); ema=copy.deepcopy(net); [p.requires_grad_(False) for p in ema.parameters()]; edec=CFG['ema_decay']
print('training MDN K=%d | params=%.0fk'%(K,sum(p.numel() for p in net.parameters())/1e3),flush=True)
t0=time.time(); best=(99,0); bestema=(99,0); order=np.arange(len(TR))
for ep in range(CFG['epochs']):
    np.random.shuffle(order)
    for i in range(0,len(TR),CFG['bs']):
        H,G,lt,tvt,ev=batch([TR[j] for j in order[i:i+CFG['bs']]],True); mu,sig,lp=net(H,G,lt)
        m=ev+CFG['w_known']*(1-ev); nll=mdn_nll(mu,sig,lp,tvt); loss=(nll*m).sum()/m.sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        with torch.no_grad():
            for pe,pn in zip(ema.parameters(),net.parameters()): pe.mul_(edec).add_(pn,alpha=1-edec)
            for be,bn in zip(ema.buffers(),net.buffers()): be.copy_(bn)
    if (ep+1)%2==0 or ep==CFG['epochs']-1:
        r=evaluate(net); re=evaluate(ema)
        if r<best[0]: best=(r,ep+1)
        if re<bestema[0]: bestema=(re,ep+1)
        print('  ep%2d | raw %.3f (best %.3f) | EMA %.3f (best %.3f@%d) %.0fs'%(ep+1,r,best[0],re,bestema[0],bestema[1],time.time()-t0),flush=True)
print('DONE raw %.3f | EMA %.3f (flat~15, WARP~11, PF~7, Tucker~5.4) — MDN+Viterbi'%(best[0],bestema[0]),flush=True)
