"""Idea 3: does WARP (continuity-anchored, doesn't trust GR) decorrelate with the GR-trusting
pipeline (sp45) enough to give a free blend gain? Load best_warp.pt, run on the same 160-well
holdout as proxy.pkl, measure error correlation, sweep blend weight."""
import numpy as np, pandas as pd, glob, os, pickle, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0); DEV='cuda' if torch.cuda.is_available() else 'cpu'
CFG=dict(d=128, KSTEPS=60, EVSTEPS=360, TWGRID=256, drop=0.2, maxstep=1.5, n_val=160, seed=42)
TG=CFG['TWGRID']; KS=CFG['KSTEPS']; ES=CFG['EVSTEPS']
TRAIN_DIR='d:/ROGII/data/train'
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
    return dict(wid=wid,H=np.stack([R(caln),R(gradn),R(rstdn),R(dz)]).astype(np.float32),
                gn=((g_gr-gm)/gs).astype(np.float32), tvt=R(true), ev=evr, last_tvt=last_tvt,
                md=R(MD), dz_raw=R(dz))
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
    def forward(s,H,G,lt):
        h=s.he(H); t=s.te(G)
        Q=s.q(h).transpose(1,2); K=s.k(t).transpose(1,2); V=s.vv(t).transpose(1,2)
        att=torch.softmax(Q@K.transpose(1,2)*s.sc,dim=2); ctx=att@V
        x=torch.cat([h.transpose(1,2),ctx],dim=2)
        dt=torch.tanh(s.head(x)[...,0])*CFG['maxstep']; tvt=torch.cumsum(dt,1)
        return tvt-tvt[:,KS-1:KS]+lt[:,None]

# same holdout split as kaggle_warp.py / kaggle_synth2.py (seed 42)
wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
DATA=[b for b in (build_well(w) for w in wids) if b is not None]
d3=np.concatenate([w['dz_raw'] for w in DATA]); m3,s3=d3.mean(),d3.std()+1e-6
for w in DATA: w['H'][3]=(w['dz_raw']-m3)/s3
rng=np.random.default_rng(CFG['seed']); idx=np.arange(len(DATA)); rng.shuffle(idx)
VA=[DATA[i] for i in idx[:CFG['n_val']]]
print('holdout wells:',len(VA))

net=WarpNet(CFG['d'],CFG['drop']).to(DEV)
sd=torch.load('d:/ROGII/best_warp.pt',map_location=DEV)
net.load_state_dict(sd); net.eval()
warp_pred={}
with torch.no_grad():
    for i in range(0,len(VA),16):
        ws=VA[i:i+16]
        H=torch.tensor(np.stack([w['H'] for w in ws]),dtype=torch.float32,device=DEV)
        G=torch.tensor(np.stack([w['gn'][:TG] for w in ws]),dtype=torch.float32,device=DEV)[:,None]
        lt=torch.tensor([w['last_tvt'] for w in ws],dtype=torch.float32,device=DEV)
        p=net(H,G,lt).cpu().numpy()
        for w,pp in zip(ws,p): warp_pred[w['wid']]=pp

# load proxy (sp45 = pipeline core, on the SAME wids but proxy has its own arrays)
proxy=pickle.load(open('C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl','rb'))['DATA']

def pooled_from(preds_by_wid):
    s=[]
    for wid,p in preds_by_wid.items():
        if wid not in proxy: continue
        w=proxy[wid]; s.append((np.asarray(p,float)-w['true'])**2)
    return float(np.sqrt(np.mean(np.concatenate(s))))

# align WARP's eval-region resampled grid predictions back onto proxy's raw-index true/sp45 arrays via MD interp
common=[w for w in VA if w['wid'] in proxy]
print('common with proxy:',len(common))
warp_on_proxy_grid={}
errs_sp45=[]; errs_warp=[]
for w in common:
    wid=w['wid']; pxw=proxy[wid]
    ev_mask = w['ev']>0.5
    md_ev = w['md'][ev_mask]; warp_ev = warp_pred[wid][ev_mask]
    # proxy true/sp45 are on raw hw index; need MD for raw index too -> reload MD raw quickly
    hw=pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv'); md_raw=hw['MD'].values.astype(float)
    ev_raw=hw['TVT_input'].isna().values
    md_raw_ev=md_raw[ev_raw]
    warp_interp=np.interp(md_raw_ev, md_ev, warp_ev)
    warp_on_proxy_grid[wid]=warp_interp
    errs_sp45.append(pxw['sp45']-pxw['true']); errs_warp.append(warp_interp-pxw['true'])
errs_sp45=np.concatenate(errs_sp45); errs_warp=np.concatenate(errs_warp)
print('pooled sp45 (this well subset):', float(np.sqrt(np.mean(errs_sp45**2))))
print('pooled warp (this well subset):', float(np.sqrt(np.mean(errs_warp**2))))
print('error correlation sp45 vs warp:', float(np.corrcoef(errs_sp45,errs_warp)[0,1]))

print('\n=== blend sweep: (1-a)*sp45 + a*warp ===')
for a in [0.0,0.02,0.05,0.08,0.1,0.15,0.2,0.3,0.5]:
    preds={wid: (1-a)*proxy[wid]['sp45'] + a*warp_on_proxy_grid[wid] for wid in warp_on_proxy_grid}
    print('  a=%.2f  pooled=%.4f'%(a,pooled_from(preds)))
