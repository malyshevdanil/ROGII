"""ROGII — TRANSFORMER (global attention). All prior attempts were CNN (convs are LOCAL -> blind to
long-range continuity). A transformer sees the WHOLE wellbore: self-attention captures long-range
continuity, cross-attention to the typewell does differentiable alignment. LayerNorm (fixed-length, no
padding). WARP-style dTVT head (continuity anchor). Kaggle: GPU T4 x2, Internet off, competition dataset."""
import numpy as np, pandas as pd, glob, os, time, copy, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0); DEV='cuda' if torch.cuda.is_available() else 'cpu'
CFG=dict(d=128, nhead=8, nlayer=4, KSTEPS=60, EVSTEPS=360, TWGRID=256, drop=0.15, wd=3e-4, lr=1.2e-3,
         epochs=100, bs=24, maxstep=1.5, smooth=0.05, aug_caljit=0.12, aug_noise=0.07, aug_warp=0.35,
         w_known=0.5, ema=0.995, n_val=160, seed=42)
TG=CFG['TWGRID']; KSTEPS=CFG['KSTEPS']; EVSTEPS=CFG['EVSTEPS']; L=KSTEPS+EVSTEPS
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
    Z=hw['Z'].values.astype(float); MD=hw['MD'].values.astype(float); mdd=np.gradient(MD); mdd[mdd==0]=1; dz=np.gradient(Z)/mdd
    ev=hw['TVT_input'].isna().values.astype(float); ei=np.where(ev>0.5)[0]
    if len(ei)<5: return None
    e0=ei[0]; ks=np.arange(max(0,e0-400),e0)
    if len(ks)<5: return None
    g_tvt=np.linspace(tt.min(),tt.max(),TG); g_gr=np.interp(g_tvt,tt,tg); gm,gs=float(g_gr.mean()),float(g_gr.std()+1e-6)
    caln=(cg-gm)/gs; true=hw['TVT'].values.astype(float)
    kd=np.linspace(ks[0],ks[-1],KSTEPS); ed=np.linspace(e0,n-1,EVSTEPS); dst=np.concatenate([kd,ed]); src=np.arange(n)
    R=lambda x: np.interp(dst,src,x).astype(np.float32)
    return dict(H=np.stack([R(caln),R(np.gradient(caln)),R(dz)]).astype(np.float32),
                gn=((g_gr-gm)/gs).astype(np.float32), tvt=R(true), ev=np.concatenate([np.zeros(KSTEPS),np.ones(EVSTEPS)]).astype(np.float32), last_tvt=last_tvt)
print('building...',flush=True); t0=time.time()
wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
DATA=[b for b in (build(w) for w in wids) if b is not None]
d2=np.concatenate([w['H'][2] for w in DATA]); mm,ss=d2.mean(),d2.std()+1e-6
for w in DATA: w['H'][2]=(w['H'][2]-mm)/ss
rng=np.random.default_rng(CFG['seed']); idx=np.arange(len(DATA)); rng.shuffle(idx)
VA=[DATA[i] for i in idx[:CFG['n_val']]]; TR=[DATA[i] for i in idx[CFG['n_val']:]]
print('built %d (%d/%d) %.0fs'%(len(DATA),len(TR),len(VA),time.time()-t0),flush=True)
def prmse(s): return float(np.sqrt(np.mean(np.concatenate(s))))
print('flat=%.3f (WARP~11)'%prmse([(w['last_tvt']-w['tvt'][w['ev']>0.5])**2 for w in VA]),flush=True)

def pos_enc(n,d,dev):
    p=torch.zeros(n,d,device=dev); pos=torch.arange(n,device=dev).float()[:,None]
    dv=torch.exp(torch.arange(0,d,2,device=dev).float()*(-np.log(10000)/d))
    p[:,0::2]=torch.sin(pos*dv); p[:,1::2]=torch.cos(pos*dv); return p
class Block(nn.Module):
    def __init__(s,d,h,drop):
        super().__init__(); s.sa=nn.MultiheadAttention(d,h,dropout=drop,batch_first=True)
        s.ca=nn.MultiheadAttention(d,h,dropout=drop,batch_first=True)
        s.n1=nn.LayerNorm(d); s.n2=nn.LayerNorm(d); s.n3=nn.LayerNorm(d)
        s.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Dropout(drop),nn.Linear(2*d,d))
    def forward(s,x,mem):
        x=x+s.sa(s.n1(x),s.n1(x),s.n1(x))[0]                   # self-attn: long-range continuity
        x=x+s.ca(s.n2(x),mem,mem)[0]                           # cross-attn to typewell: alignment
        return x+s.ff(s.n3(x))
class TF(nn.Module):
    def __init__(s,d,h,nl,drop):
        super().__init__(); s.hin=nn.Linear(3,d); s.tin=nn.Linear(1,d)
        s.blocks=nn.ModuleList([Block(d,h,drop) for _ in range(nl)])
        s.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1))
        s.head[-1].weight.data*=0.01; s.head[-1].bias.data.zero_(); s.d=d
    def forward(s,H,G,lt):
        B=H.shape[0]; x=s.hin(H.transpose(1,2))+pos_enc(H.shape[2],s.d,H.device)[None]   # (B,L,d)
        mem=s.tin(G.transpose(1,2))+pos_enc(G.shape[2],s.d,H.device)[None]               # (B,TG,d)
        for blk in s.blocks: x=blk(x,mem)
        dt=torch.tanh(s.head(x)[...,0])*CFG['maxstep']; tvt=torch.cumsum(dt,1)
        return tvt-tvt[:,KSTEPS-1:KSTEPS]+lt[:,None]
def batch(ws,train=False):
    H=np.stack([w['H'] for w in ws]).copy(); gn=np.stack([w['gn'][:TG] for w in ws]).copy(); tvt=np.stack([w['tvt'] for w in ws]).copy(); ev=np.stack([w['ev'] for w in ws])
    if train:
        sc=1+np.random.randn(len(ws),1,1)*CFG['aug_caljit']; sh=np.random.randn(len(ws),1,1)*CFG['aug_caljit']
        H[:,0:1]=H[:,0:1]*sc+sh; gn=gn*sc[:,:,0]+sh[:,:,0]; H=H+np.random.randn(*H.shape)*CFG['aug_noise']
        if CFG['aug_warp']>0:
            m=EVSTEPS; src=np.arange(m)
            for bi in range(len(ws)):
                d=np.cumsum(np.abs(1+np.random.randn(m)*CFG['aug_warp'])); d=(d-d[0])/(d[-1]-d[0])*(m-1)
                for ch in range(3): H[bi,ch,KSTEPS:]=np.interp(d,src,H[bi,ch,KSTEPS:])
                tvt[bi,KSTEPS:]=np.interp(d,src,tvt[bi,KSTEPS:])
    to=lambda a: torch.tensor(a,dtype=torch.float32,device=DEV)
    return to(H),to(gn)[:,None],to(np.array([w['last_tvt'] for w in ws])),to(tvt),to(ev)
def evaluate(net):
    net.eval(); e=[]
    with torch.no_grad():
        for i in range(0,len(VA),CFG['bs']):
            H,G,lt,tvt,ev=batch(VA[i:i+CFG['bs']]); p=net(H,G,lt); e.append(((p-tvt)[ev>0.5]).cpu().numpy())
    net.train(); return float(np.sqrt(np.mean(np.concatenate(e)**2)))
net=TF(CFG['d'],CFG['nhead'],CFG['nlayer'],CFG['drop']).to(DEV); opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG['lr'],total_steps=CFG['epochs']*((len(TR)+CFG['bs']-1)//CFG['bs']))
ema=copy.deepcopy(net); [p.requires_grad_(False) for p in ema.parameters()]
print('training TRANSFORMER | params=%.0fk'%(sum(p.numel() for p in net.parameters())/1e3),flush=True)
t0=time.time(); best=(99,0); order=np.arange(len(TR))
for ep in range(CFG['epochs']):
    np.random.shuffle(order)
    for i in range(0,len(TR),CFG['bs']):
        H,G,lt,tvt,ev=batch([TR[j] for j in order[i:i+CFG['bs']]],True); p=net(H,G,lt)
        m=ev+CFG['w_known']*(1-ev); l=(F.smooth_l1_loss(p,tvt,reduction='none',beta=8.0)*m).sum()/m.sum()
        d2=p[:,2:]-2*p[:,1:-1]+p[:,:-2]; loss=l+CFG['smooth']*(d2*d2*m[:,2:]).sum()/m[:,2:].sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        with torch.no_grad():
            for pe,pn in zip(ema.parameters(),net.parameters()): pe.mul_(CFG['ema']).add_(pn,alpha=1-CFG['ema'])
            for be,bn in zip(ema.buffers(),net.buffers()): be.copy_(bn)
    if (ep+1)%3==0 or ep==CFG['epochs']-1:
        re=evaluate(ema); rr=evaluate(net)
        if re<best[0]: best=(re,ep+1)
        print('  ep%2d | raw %.3f EMA %.3f (best %.3f@%d) %.0fs'%(ep+1,rr,re,best[0],best[1],time.time()-t0),flush=True)
print('DONE TRANSFORMER EMA best %.3f (flat15 WARP11 PF7 Tucker5.4)'%best[0],flush=True)
