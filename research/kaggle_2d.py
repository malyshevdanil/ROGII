"""ROGII — 2D MISFIT-HEATMAP + SDF (the geosteering-image approach).
Build a 2D image M[z,x] = typewell_GR[z] - horizontal_GR[x] (rows=typewell TVT depth, cols=MD step).
The true TVT is a CONTINUOUS zero-contour on M. A 2D U-Net predicts the Signed Distance Function to that
contour (dense supervision over all pixels), and the continuity of the contour resolves GR self-similarity
(what 1D conv couldn't). Inference: per column, the SDF zero-crossing -> z* -> TVT. Kaggle: GPU T4 x2."""
import numpy as np, pandas as pd, glob, os, time, copy, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0); DEV='cuda' if torch.cuda.is_available() else 'cpu'
CFG=dict(TG=192, L=384, KSTEPS=48, d=48, drop=0.1, wd=3e-4, lr=1.5e-3, epochs=60, bs=8,
         sdf_clip=40., w_pos=1.0, w_sdf=1.0, aug_caljit=0.12, aug_noise=0.06, ema=0.995, n_val=160, seed=42)
TG=CFG['TG']; L=CFG['L']; KSTEPS=CFG['KSTEPS']; EV=L-KSTEPS
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
    ev=hw['TVT_input'].isna().values.astype(float); ei=np.where(ev>0.5)[0]
    if len(ei)<5: return None
    e0=ei[0]; ks=np.arange(max(0,e0-400),e0)
    if len(ks)<5: return None
    g_tvt=np.linspace(tt.min(),tt.max(),TG); g_gr=np.interp(g_tvt,tt,tg); gm,gs=float(g_gr.mean()),float(g_gr.std()+1e-6)
    caln=(cg-gm)/gs; true=hw['TVT'].values.astype(float)
    kd=np.linspace(ks[0],ks[-1],KSTEPS); ed=np.linspace(e0,n-1,EV); dst=np.concatenate([kd,ed]); src=np.arange(n)
    R=lambda x: np.interp(dst,src,x).astype(np.float32)
    tvt_r=R(true); zstar=np.clip((tvt_r-g_tvt[0])/(g_tvt[-1]-g_tvt[0])*(TG-1),0,TG-1)   # true depth-index path
    evr=np.concatenate([np.zeros(KSTEPS),np.ones(EV)]).astype(np.float32)
    return dict(caln=R(caln).astype(np.float32), gn=((g_gr-gm)/gs).astype(np.float32), g_tvt=g_tvt.astype(np.float32),
                zstar=zstar.astype(np.float32), tvt=tvt_r, ev=evr, last_tvt=last_tvt)
print('building...',flush=True); t0=time.time()
wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
DATA=[b for b in (build(w) for w in wids) if b is not None]
rng=np.random.default_rng(CFG['seed']); idx=np.arange(len(DATA)); rng.shuffle(idx)
VA=[DATA[i] for i in idx[:CFG['n_val']]]; TR=[DATA[i] for i in idx[CFG['n_val']:]]
print('built %d (%d/%d) %.0fs'%(len(DATA),len(TR),len(VA),time.time()-t0),flush=True)
def prmse(s): return float(np.sqrt(np.mean(np.concatenate(s))))
print('flat=%.3f (WARP~11)'%prmse([(w['last_tvt']-w['tvt'][w['ev']>0.5])**2 for w in VA]),flush=True)

ZC=torch.arange(TG,device=DEV).float()
def make_batch(ws,train=False):
    B=len(ws); caln=np.stack([w['caln'] for w in ws]); gn=np.stack([w['gn'][:TG] for w in ws])
    zstar=np.stack([w['zstar'] for w in ws]); ev=np.stack([w['ev'] for w in ws]); tvt=np.stack([w['tvt'] for w in ws])
    if train:
        sc=1+np.random.randn(B,1)*CFG['aug_caljit']; sh=np.random.randn(B,1)*CFG['aug_caljit']
        caln=caln*sc+sh+np.random.randn(*caln.shape)*CFG['aug_noise']; gn=gn*sc+sh
    caln=torch.tensor(caln,dtype=torch.float32,device=DEV); gn=torch.tensor(gn,dtype=torch.float32,device=DEV)
    # 2D misfit image M[b, z, x] = typewell_gr[z] - horizontal_gr[x]  (B,TG,L)
    M=gn[:,:,None]-caln[:,None,:]
    Ma=M.abs()
    zc=(ZC[None,:,None]/TG*2-1).expand(B,TG,L)                    # depth positional channel
    img=torch.stack([M,Ma,zc],1)                                 # (B,3,TG,L)
    zst=torch.tensor(zstar,dtype=torch.float32,device=DEV)       # (B,L) true depth path
    sdf=(ZC[None,:,None]-zst[:,None,:]).clamp(-CFG['sdf_clip'],CFG['sdf_clip'])   # (B,TG,L) signed dist to contour
    return img,zst,sdf,torch.tensor(ev,device=DEV),torch.tensor(tvt,device=DEV),torch.tensor(np.stack([w['g_tvt'] for w in ws]),dtype=torch.float32,device=DEV)

def gn_(c): return nn.GroupNorm(min(8,c),c)
class UNet(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.e1=nn.Sequential(nn.Conv2d(3,d,3,padding=1),gn_(d),nn.GELU(),nn.Conv2d(d,d,3,padding=1),gn_(d),nn.GELU())
        s.e2=nn.Sequential(nn.Conv2d(d,2*d,3,padding=1),gn_(2*d),nn.GELU(),nn.Conv2d(2*d,2*d,3,padding=1),gn_(2*d),nn.GELU())
        s.e3=nn.Sequential(nn.Conv2d(2*d,4*d,3,padding=1),gn_(4*d),nn.GELU(),nn.Conv2d(4*d,4*d,3,padding=1),gn_(4*d),nn.GELU())
        s.u2=nn.Sequential(nn.Conv2d(6*d,2*d,3,padding=1),gn_(2*d),nn.GELU())
        s.u1=nn.Sequential(nn.Conv2d(3*d,d,3,padding=1),gn_(d),nn.GELU())
        s.out=nn.Conv2d(d,1,1)
    def forward(s,x):
        x1=s.e1(x); x2=s.e2(F.max_pool2d(x1,2)); x3=s.e3(F.max_pool2d(x2,2))
        u2=s.u2(torch.cat([F.interpolate(x3,size=x2.shape[-2:],mode='nearest'),x2],1))
        u1=s.u1(torch.cat([F.interpolate(u2,size=x1.shape[-2:],mode='nearest'),x1],1))
        return s.out(u1)[:,0]                                    # (B,TG,L) predicted SDF
def pos_from_sdf(sdf_hat):                                       # per column -> soft depth via softmin of |sdf|
    w=torch.softmax(-sdf_hat.abs()*0.5,dim=1)                    # (B,TG,L)
    return (w*ZC[None,:,None]).sum(1)                            # (B,L) z*
def viterbi_decode(sdf_hat,trans=0.06,band=15):                 # CONTINUITY-constrained contour (uses 2D structure!)
    B,TGn,Ln=sdf_hat.shape; cost=sdf_hat.abs().cpu().numpy(); out=np.zeros((B,Ln),np.float32)
    zc=np.arange(TGn); off=np.arange(-band,band+1)
    for b in range(B):
        c=cost[b]; D=c[:,0].copy(); bp=np.zeros((TGn,Ln),np.int32)
        for x in range(1,Ln):
            # for each z, best prev z' within band minimizing D_prev + trans*(z-z')^2
            src=zc[:,None]+off[None,:]; ok=(src>=0)&(src<TGn); src_c=np.clip(src,0,TGn-1)
            cand=D[src_c]+trans*(off[None,:]**2); cand[~ok]=1e18
            bi=np.argmin(cand,1); bp[:,x]=zc+off[bi]; D=c[:,x]+cand[zc,bi]
        z=int(np.argmin(D)); path=np.zeros(Ln,np.int32); path[-1]=z
        for x in range(Ln-1,0,-1): z=int(np.clip(bp[z,x],0,TGn-1)); path[x-1]=z
        out[b]=path
    return torch.tensor(out,device=sdf_hat.device)
def to_tvt(zsoft,g_tvt):
    z0=zsoft.clamp(0,TG-1.001); i0=z0.long(); fr=z0-i0.float()
    a=torch.gather(g_tvt,1,i0); bb=torch.gather(g_tvt,1,(i0+1).clamp(max=TG-1)); return a*(1-fr)+bb*fr
def evaluate(net):
    net.eval(); e=[]
    with torch.no_grad():
        for i in range(0,len(VA),CFG['bs']):
            ws=VA[i:i+CFG['bs']]; img,zst,sdf,ev,tvt,gt=make_batch(ws)
            z=viterbi_decode(net(img)); pr=to_tvt(z,gt); m=ev>0.5   # CONTINUITY decode (resolves GR ambiguity)
            e.append(((pr-tvt)[m]).cpu().numpy())
    net.train(); return float(np.sqrt(np.mean(np.concatenate(e)**2)))
net=UNet(CFG['d']).to(DEV); opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG['lr'],total_steps=CFG['epochs']*((len(TR)+CFG['bs']-1)//CFG['bs']))
ema=copy.deepcopy(net); [p.requires_grad_(False) for p in ema.parameters()]
print('training 2D-SDF U-Net | params=%.0fk'%(sum(p.numel() for p in net.parameters())/1e3),flush=True)
t0=time.time(); best=(99,0); order=np.arange(len(TR))
for ep in range(CFG['epochs']):
    np.random.shuffle(order)
    for i in range(0,len(TR),CFG['bs']):
        ws=[TR[j] for j in order[i:i+CFG['bs']]]; img,zst,sdf,ev,tvt,gt=make_batch(ws,True)
        sh=net(img); z=pos_from_sdf(sh); pr=to_tvt(z,gt)
        m=ev+0.5*(1-ev)
        Lsdf=((sh-sdf)**2).mean()
        Lpos=(F.smooth_l1_loss(pr,tvt,reduction='none',beta=8.0)*m).sum()/m.sum()
        dz=z[:,1:]-z[:,:-1]; Lsm=(dz*dz*m[:,1:]).sum()/m[:,1:].sum()          # smooth contour (continuity)
        loss=CFG['w_sdf']*Lsdf*0.02+CFG['w_pos']*Lpos+0.05*Lsm
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        with torch.no_grad():
            for pe,pn in zip(ema.parameters(),net.parameters()): pe.mul_(CFG['ema']).add_(pn,alpha=1-CFG['ema'])
            for be,bn in zip(ema.buffers(),net.buffers()): be.copy_(bn)
    if (ep+1)%4==0 or ep==CFG["epochs"]-1:
        re=evaluate(ema); rr=0.0  # EMA-only (Viterbi eval is CPU-slow)
        if re<best[0]: best=(re,ep+1)
        print('  ep%2d Lsdf%.2f Lpos%.2f | raw %.3f EMA %.3f (best %.3f@%d) %.0fs'%(ep+1,Lsdf.item(),Lpos.item(),rr,re,best[0],best[1],time.time()-t0),flush=True)
print('DONE 2D-SDF EMA best %.3f (flat15 WARP11 PF7 Tucker5.4) — CONTOUR approach'%best[0],flush=True)
