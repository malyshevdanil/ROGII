"""Idea 2: pure classification over typewell depth bins (softmax + cross-entropy, argmax decode).
Literal reading of the idea: NO continuity anchor, NO cumsum -- each step independently predicts
P(bin_k | local GR window) via softmax over TG discretized typewell-TVT bins, decoded via argmax
(handles bimodality without mean-collapse, unlike soft-argmax). Same Enc/data pipeline as kaggle_warp.py
for a fair architecture comparison. Also reports top-3-expectation decode as the user suggested."""
import numpy as np, pandas as pd, glob, os, time, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0); DEV='cuda' if torch.cuda.is_available() else 'cpu'
CFG=dict(d=128, KSTEPS=60, EVSTEPS=360, TWGRID=256, drop=0.2, wd=3e-4, lr=1.2e-3, epochs=45, bs=24,
         aug_caljit=0.12, aug_noise=0.07, n_val=160, seed=42)
TG=CFG['TWGRID']; KS=CFG['KSTEPS']; ES=CFG['EVSTEPS']
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
    tw_lo,tw_hi=tt.min(),tt.max()
    g_tvt=np.linspace(tw_lo,tw_hi,TG); g_gr=np.interp(g_tvt,tt,tg); gm,gs=float(g_gr.mean()),float(g_gr.std()+1e-6)
    caln=(cal-gm)/gs; gradn=grad/gs; rstdn=rstd/gs; true=hw['TVT'].values.astype(float)
    kd=np.linspace(ks[0],ks[-1],KS); ed=np.linspace(e0,n-1,ES); dst=np.concatenate([kd,ed]); src=np.arange(n)
    R=lambda x: np.interp(dst,src,x).astype(np.float32)
    evr=np.concatenate([np.zeros(KS),np.ones(ES)]).astype(np.float32)
    true_r=R(true)
    true_bin=np.clip(((true_r-tw_lo)/(tw_hi-tw_lo)*(TG-1)).round().astype(np.int64),0,TG-1)
    return dict(H=np.stack([R(caln),R(gradn),R(rstdn),R(dz)]).astype(np.float32),
                gn=((g_gr-gm)/gs).astype(np.float32), tvt=true_r, ev=evr, last_tvt=last_tvt,
                bin=true_bin, tw_lo=tw_lo, tw_hi=tw_hi, tg_vals=g_tvt.astype(np.float32))
print('building...',flush=True); t0=time.time()
wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
DATA=[b for b in (build_well(w) for w in wids) if b is not None]
d3=np.concatenate([w['H'][3] for w in DATA]); m3,s3=d3.mean(),d3.std()+1e-6
for w in DATA: w['H'][3]=(w['H'][3]-m3)/s3
rng=np.random.default_rng(CFG['seed']); idx=np.arange(len(DATA)); rng.shuffle(idx)
VA=[DATA[i] for i in idx[:CFG['n_val']]]; TR=[DATA[i] for i in idx[CFG['n_val']:]]
print('built %d (%d/%d) %.0fs'%(len(DATA),len(TR),len(VA),time.time()-t0),flush=True)
def prmse(sqs): return float(np.sqrt(np.mean(np.concatenate(sqs))))
print('flat=%.3f (WARP anchor ~11.0)'%prmse([(w['last_tvt']-w['tvt'][w['ev']>0.5])**2 for w in VA]),flush=True)

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
class ClsNet(nn.Module):
    """Pure classification: NO continuity anchor, NO cumsum. Cross-attn to typewell then per-step
    softmax over TG absolute typewell bins. Literal reading of idea 2."""
    def __init__(s,d,drop):
        super().__init__(); s.he=Enc(4,d,drop); s.te=Enc(1,d,drop)
        s.q=nn.Conv1d(d,d,1); s.k=nn.Conv1d(d,d,1); s.vv=nn.Conv1d(d,d,1); s.sc=d**-0.5
        s.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,TG))
    def forward(s,H,G):
        h=s.he(H); t=s.te(G)
        Q=s.q(h).transpose(1,2); K=s.k(t).transpose(1,2); V=s.vv(t).transpose(1,2)
        att=torch.softmax(Q@K.transpose(1,2)*s.sc,dim=2); ctx=att@V
        x=torch.cat([h.transpose(1,2),ctx],dim=2)
        return s.head(x)                                    # (B,L,TG) logits
net=ClsNet(CFG['d'],CFG['drop']).to(DEV)
opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['wd'])
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG['lr'],total_steps=CFG['epochs']*((len(TR)+CFG['bs']-1)//CFG['bs']))
def batch(ws,train=False):
    H=np.stack([w['H'] for w in ws]).copy(); gn=np.stack([w['gn'][:TG] for w in ws]).copy()
    bins=np.stack([w['bin'] for w in ws]).copy(); ev=np.stack([w['ev'] for w in ws])
    tvt=np.stack([w['tvt'] for w in ws]).copy()
    if train:
        sc=1+np.random.randn(len(ws),1,1)*CFG['aug_caljit']; sh=np.random.randn(len(ws),1,1)*CFG['aug_caljit']
        H[:,0:1]=H[:,0:1]*sc+sh; gn=gn*sc[:,:,0]+sh[:,:,0]; H=H+np.random.randn(*H.shape)*CFG['aug_noise']
    to=lambda a: torch.tensor(a,dtype=torch.float32 if a.dtype!=np.int64 else torch.long,device=DEV)
    return to(H),to(gn)[:,None],to(bins),to(ev),to(tvt)
def decode(logits,ws,mode='argmax'):
    # logits (B,L,TG); return decoded TVT (B,L) using each well's own tg_vals mapping
    if mode=='argmax':
        idx=logits.argmax(-1).cpu().numpy()
    else:  # top3 expectation
        p=torch.softmax(logits,-1)
        top=torch.topk(p,3,dim=-1)
        idx=None
    out=np.zeros(logits.shape[:2],dtype=np.float32)
    prob=torch.softmax(logits,-1).cpu().numpy() if mode!='argmax' else None
    for bi,w in enumerate(ws):
        tgv=w['tg_vals']
        if mode=='argmax':
            out[bi]=tgv[idx[bi]]
        else:
            p3v,p3i=torch.topk(torch.softmax(logits[bi],-1),3,dim=-1)
            p3v=p3v.cpu().numpy(); p3i=p3i.cpu().numpy()
            vals=tgv[p3i]                                    # (L,3)
            w_=p3v/p3v.sum(-1,keepdims=True)
            out[bi]=(vals*w_).sum(-1)
    return out
def evaluate(mode='argmax'):
    net.eval(); e=[]
    with torch.no_grad():
        for i in range(0,len(VA),CFG['bs']):
            ws=VA[i:i+CFG['bs']]; H,G,bins,ev,tvt=batch(ws)
            logits=net(H,G); dec=decode(logits,ws,mode)
            m=(ev>0.5).cpu().numpy()
            e.append(((dec-tvt.cpu().numpy())[m])**2)
    net.train(); return float(np.sqrt(np.mean(np.concatenate(e))))
print('training pure-classification net | params=%.0fk'%(sum(p.numel() for p in net.parameters())/1e3),flush=True)
t0=time.time(); best=(99,0,'')
order=np.arange(len(TR))
for ep in range(CFG['epochs']):
    np.random.shuffle(order); tot=0; nb=0
    for i in range(0,len(TR),CFG['bs']):
        ws=[TR[j] for j in order[i:i+CFG['bs']]]
        H,G,bins,ev,tvt=batch(ws,True); logits=net(H,G)
        loss=F.cross_entropy(logits.reshape(-1,TG),bins.reshape(-1),reduction='none')
        w_=(ev+0.3*(1-ev)).reshape(-1)
        loss=(loss*w_).sum()/w_.sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step()
        tot+=loss.item(); nb+=1
    if (ep+1)%3==0 or ep==CFG['epochs']-1:
        ra=evaluate('argmax'); rt=evaluate('top3')
        if ra<best[0]: best=(ra,ep+1,'argmax')
        if rt<best[0]: best=(rt,ep+1,'top3')
        print('  ep%2d loss%.3f | argmax %.3f top3exp %.3f (best %.3f@%d-%s) %.0fs'%(
            ep+1,tot/nb,ra,rt,best[0],best[1],best[2],time.time()-t0),flush=True)
print('DONE pure-classification best %.3f @ep%d[%s] (flat~15, WARP-continuity~11.0, target: beat 11)'%best,flush=True)
