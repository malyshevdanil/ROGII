"""ROGII idea-2 TRANSFER GATE: does synthetic pretrain with REALISTIC (bootstrapped) GR noise transfer
to the real holdout, while NAIVE gaussian noise does not? Same WarpNet as kaggle_warp.py.
Synthetic TVT path uses our own finding: piecewise-linear, ~quantized dips, rare fault jumps.
Run modes (env NOISE): 'boot' (real residual bank) | 'gauss' (white) | 'real' (train on real, ref)."""
import numpy as np, pandas as pd, glob, os, time, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV='cuda' if torch.cuda.is_available() else 'cpu'
NOISE=os.environ.get('NOISE','boot'); EPOCHS=int(os.environ.get('EPOCHS','60'))
CFG=dict(d=128, KSTEPS=60, EVSTEPS=360, TWGRID=256, drop=0.2, wd=3e-4, lr=1.2e-3, bs=24,
         maxstep=1.5, smooth=0.06, w_known=0.5, n_val=160, seed=42)
TG=CFG['TWGRID']; KS=CFG['KSTEPS']; ES=CFG['EVSTEPS']; L=KS+ES
TRAIN_DIR=next((d for d in ('data/train','d:/ROGII/data/train') if os.path.isdir(d)),'data/train')
def inn(a):
    a=a.copy();n=len(a);idx=np.arange(n);m=np.isnan(a)
    if m.all():return np.zeros(n)
    a[m]=np.interp(idx[m],idx[~m],a[~m]);return a
def build_well(wid):
    hw=pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv'); tw=pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tt=tw['TVT'].values.astype(float); tg=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    if len(tt)<10: return None
    kn=hw[hw['TVT_input'].notna()]
    if len(kn)<20 or hw['TVT_input'].isna().sum()<20: return None
    last=kn.iloc[-1]; last_tvt=float(last['TVT_input'])
    n=len(hw); gr=inn(hw['GR'].values.astype(float))
    kn_gr=kn['GR'].interpolate().bfill().ffill().values.astype(float)
    twk=np.interp(kn['TVT_input'].values,tt,tg); v=np.isfinite(kn_gr)&np.isfinite(twk)
    a,b=(np.polyfit(kn_gr[v],twk[v],1) if v.sum()>=20 else (1.,0.)); cal=gr*a+b
    Z=hw['Z'].values.astype(float); MD=hw['MD'].values.astype(float); mdd=np.gradient(MD); mdd[mdd==0]=1
    grad=np.gradient(gr); rstd=pd.Series(gr).rolling(21,center=True,min_periods=1).std().fillna(0).values
    dz=np.gradient(Z)/mdd
    ev=hw['TVT_input'].isna().values.astype(float); ei=np.where(ev>0.5)[0]
    if len(ei)<5: return None
    e0=ei[0]; ks=np.arange(max(0,e0-400),e0)
    if len(ks)<5: return None
    g_tvt=np.linspace(tt.min(),tt.max(),TG); g_gr=np.interp(g_tvt,tt,tg); gm,gs=float(g_gr.mean()),float(g_gr.std()+1e-6)
    caln=(cal-gm)/gs; gradn=grad/gs; rstdn=rstd/gs; true=hw['TVT'].values.astype(float)
    kd=np.linspace(ks[0],ks[-1],KS); ed=np.linspace(e0,n-1,ES); dst=np.concatenate([kd,ed]); src=np.arange(n)
    R=lambda x: np.interp(dst,src,x).astype(np.float32)
    evr=np.concatenate([np.zeros(KS),np.ones(ES)]).astype(np.float32)
    # residual in sequence space: caln - perfect(true)  (perfect from THIS typewell)
    perf=(np.interp(true,tt,tg)-gm)/gs
    resid=(R(caln)-R(perf)).astype(np.float32)
    return dict(H=np.stack([R(caln),R(gradn),R(rstdn),R(dz)]).astype(np.float32),
                gn=((g_gr-gm)/gs).astype(np.float32), tvt=R(true), ev=evr, last_tvt=last_tvt,
                resid=resid, tt=tt, tg=tg, gm=gm, gs=gs, dzr=R(dz).astype(np.float32))
print('building real wells...',flush=True); t0=time.time()
wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
DATA=[b for b in (build_well(w) for w in wids) if b is not None]
d3=np.concatenate([w['H'][3] for w in DATA]); m3,s3=d3.mean(),d3.std()+1e-6
for w in DATA:
    w['H'][3]=(w['H'][3]-m3)/s3; w['dzr']=(w['dzr']-m3)/s3
rng=np.random.default_rng(CFG['seed']); idx=np.arange(len(DATA)); rng.shuffle(idx)
VA=[DATA[i] for i in idx[:CFG['n_val']]]; TRr=[DATA[i] for i in idx[CFG['n_val']:]]
print('built %d (%d/%d) %.0fs | NOISE=%s EPOCHS=%d'%(len(DATA),len(TRr),len(VA),time.time()-t0,NOISE,EPOCHS),flush=True)
def prmse(sqs): return float(np.sqrt(np.mean(np.concatenate(sqs))))
print('flat=%.3f'%prmse([(w['last_tvt']-w['tvt'][w['ev']>0.5])**2 for w in VA]),flush=True)

# ---- banks for synthetic generation (from TRAIN wells only) ----
RESID=[w['resid'] for w in TRr]                              # real noise bank (EVSTEPS,), autocorr~50
TWS=[(w['tt'],w['tg'],w['gm'],w['gs']) for w in TRr]         # typewell bank
DZR=[w['dzr'] for w in TRr]                                  # geometry channel bank
# real dTVT scale (eval) to make synthetic paths realistic
_dt=np.concatenate([np.diff(w['tvt'][w['ev']>0.5]) for w in TRr]); DTSD=float(np.std(_dt))
print('real eval dTVT std=%.4f (synthetic paths matched to this)'%DTSD,flush=True)

def gen_synth(nb):
    """generate nb synthetic wells matching the WarpNet input format."""
    out=[]
    for _ in range(nb):
        ti=np.random.randint(len(TWS)); tt,tg,gm,gs=TWS[ti]
        # piecewise-linear TVT path: quantized-ish dips + rare fault jumps (our human-markup finding)
        last_tvt=float(np.random.uniform(tt.min()+50,tt.max()-50))
        dips=np.zeros(L)
        nseg=np.random.randint(2,7); bps=np.sort(np.random.choice(np.arange(KS,L),nseg,replace=False))
        cur=np.random.randn()*DTSD*1.3; p=0
        for bp in list(bps)+[L]:
            dips[p:bp]=cur; cur=cur+np.random.randn()*DTSD*1.1
            if np.random.rand()<0.25: cur+=np.random.choice([-1,1])*np.random.uniform(2,6)*DTSD  # fault
            p=bp
        path=np.cumsum(dips); path=path-path[KS-1]+last_tvt
        path=np.clip(path,tt.min()+5,tt.max()-5)
        perf=(np.interp(path,tt,tg)-gm)/gs
        if NOISE=='boot':
            nz=RESID[np.random.randint(len(RESID))].copy()
            if np.random.rand()<0.5: nz=nz[::-1].copy()                      # augment noise bank
            nz=nz*np.random.uniform(0.7,1.2)
        else:                                                               # gauss: white, matched variance
            sd=np.median([np.std(r) for r in RESID[:50]]); nz=np.random.randn(L).astype(np.float32)*sd
        caln=(perf+nz).astype(np.float32)
        gradn=np.gradient(caln).astype(np.float32)
        rstdn=pd.Series(caln).rolling(21,center=True,min_periods=1).std().fillna(0).values.astype(np.float32)
        dz=DZR[np.random.randint(len(DZR))].astype(np.float32)
        H=np.stack([caln,gradn,rstdn,dz]).astype(np.float32)
        evr=np.concatenate([np.zeros(KS),np.ones(ES)]).astype(np.float32)
        out.append(dict(H=H,gn=((np.interp(np.linspace(tt.min(),tt.max(),TG),tt,tg)-gm)/gs).astype(np.float32),
                        tvt=path.astype(np.float32),ev=evr,last_tvt=last_tvt))
    return out

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
class WarpNet(nn.Module):
    def __init__(s,d,drop):
        super().__init__(); s.he=Enc(4,d,drop); s.te=Enc(1,d,drop)
        s.q=nn.Conv1d(d,d,1); s.k=nn.Conv1d(d,d,1); s.vv=nn.Conv1d(d,d,1); s.sc=d**-0.5
        s.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1))
        s.head[-1].weight.data*=0.01; s.head[-1].bias.data.zero_()
    def forward(s,H,G,lt):
        h=s.he(H); t=s.te(G)
        Q=s.q(h).transpose(1,2); K=s.k(t).transpose(1,2); V=s.vv(t).transpose(1,2)
        att=torch.softmax(Q@K.transpose(1,2)*s.sc,dim=2); ctx=att@V
        x=torch.cat([h.transpose(1,2),ctx],dim=2)
        dt=torch.tanh(s.head(x)[...,0])*CFG['maxstep']; tvt=torch.cumsum(dt,1)
        return tvt-tvt[:,KS-1:KS]+lt[:,None]
def batch(ws):
    to=lambda a: torch.tensor(np.asarray(a),dtype=torch.float32,device=DEV)
    H=np.stack([w['H'] for w in ws]); gn=np.stack([w['gn'][:TG] for w in ws])
    return to(H),to(gn)[:,None],to([w['last_tvt'] for w in ws]),to(np.stack([w['tvt'] for w in ws])),to(np.stack([w['ev'] for w in ws]))
net=WarpNet(CFG['d'],CFG['drop']).to(DEV)
opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
STEPS_PER=200
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG['lr'],total_steps=EPOCHS*STEPS_PER)
def evaluate():
    net.eval(); e=[]
    with torch.no_grad():
        for i in range(0,len(VA),CFG['bs']):
            H,G,lt,tvt,ev=batch(VA[i:i+CFG['bs']]); p=net(H,G,lt); e.append(((p-tvt)[ev>0.5]).cpu().numpy())
    net.train(); return float(np.sqrt((np.concatenate(e)**2).mean()))
print('training WARP on %s synthetic | transfer to REAL holdout'%NOISE,flush=True)
t0=time.time(); best=(99,0)
for ep in range(EPOCHS):
    tot=0
    for _ in range(STEPS_PER):
        ws=gen_synth(CFG['bs']) if NOISE!='real' else [TRr[j] for j in np.random.randint(0,len(TRr),CFG['bs'])]
        H,G,lt,tvt,ev=batch(ws); p=net(H,G,lt)
        m=ev+CFG['w_known']*(1-ev)
        loss=(F.smooth_l1_loss(p,tvt,reduction='none',beta=8.0)*m).sum()/m.sum()
        d2=p[:,2:]-2*p[:,1:-1]+p[:,:-2]; loss=loss+CFG['smooth']*(d2*d2*m[:,2:]).sum()/m[:,2:].sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step(); tot+=loss.item()
    if (ep+1)%3==0 or ep==EPOCHS-1:
        r=evaluate();
        if r<best[0]: best=(r,ep+1)
        print('  ep%2d loss%.3f | REAL holdout %.3f (best %.3f@%d) %.0fs'%(ep+1,tot/STEPS_PER,r,best[0],best[1],time.time()-t0),flush=True)
print('DONE NOISE=%s best REAL holdout %.3f (flat~15, WARP-on-real~11.3; GATE: boot<gauss & boot->~11-13 = idea2 viable)'%(NOISE,best[0]),flush=True)
