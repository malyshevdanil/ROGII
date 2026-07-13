import numpy as np, pandas as pd, glob, os, time, sys
sys.path.insert(0, r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad')
import beam_mod
from beam_mod import beam_search, BEAMS

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def prmse(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);return float(np.sqrt(np.mean((a-b)**2)))

# --- production selector constants (from the notebook) ---
SELECTOR_N_EVAL_THRESHOLD=4840.0
SELECTOR_Z_SPAN_THRESHOLDS=(136.73000000000016,185.5133333333342)
SELECTOR_BIN_VARIANTS={0:'pf_scale_5_hold_0.2',1:'pf_scale_3_hold_0.15',2:'pf_scale_12_beam_0.2_hold_0.15',
    3:'pf_scale_5_hold_0.15',4:'pf_scale_5_beam_0.05_hold_0.05',5:'pf_scale_12_beam_0.2_hold_0.05'}
SELECTOR_GLOBAL_VARIANT='pf_scale_8_hold_0.2'
SCALES=(3.0,5.0,8.0,12.0)

def selector_code(hw):
    em=hw['TVT_input'].isna().to_numpy(); n_eval=float(em.sum())
    ze=hw.loc[em,'Z'].values.astype(float); z_span=float(np.nanmax(ze)-np.nanmin(ze)) if len(ze) else 0.0
    n_bin=int(n_eval>SELECTOR_N_EVAL_THRESHOLD); z_bin=int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS,z_span,side='right'))
    code=n_bin+2*z_bin
    return SELECTOR_BIN_VARIANTS.get(code,SELECTOR_GLOBAL_VARIANT)

def parse_variant(name):
    parts=name.split('_'); scale=float(parts[2]); bw=0.0; hw_=0.0
    if 'beam' in parts: bw=float(parts[parts.index('beam')+1])
    if 'hold' in parts: hw_=float(parts[parts.index('hold')+1])
    return scale,bw,hw_

def apply_variant(name, pf_by_scale, tvt_beam, last_known):
    scale,bw,hw_=parse_variant(name)
    base=pf_by_scale.get('pf_scale_%g'%scale, pf_by_scale['pf_scale_8'])
    pred=(1-bw)*base+bw*tvt_beam
    pred=(1-hw_)*pred+hw_*last_known
    return pred

def run_pf(hw,tw,seed,N=200):
    tw_s=tw.sort_values('TVT'); tw_tvt=tw_s['TVT'].values.astype(float); tw_gr=tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]; last=kn.iloc[-1]
    tw_at_k=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr); gs=float(np.clip(np.nanstd(kn['GR'].fillna(0).values-tw_at_k),10.,60.))
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values); dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.0
    rng=np.random.default_rng(seed); ls=float(last['TVT_input'])+float(last['Z'])
    pos=ls+4.5*rng.standard_normal(N); rate=ir+0.01*rng.standard_normal(N); w=np.ones(N)/N
    MOM=0.998;VN=0.002;PN=0.005;RP=0.1;RR=0.001;RESAMP=0.5
    md_v=ev['MD'].values.astype(float); z_v=ev['Z'].values.astype(float)
    gr_v=hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)[ev.index.to_numpy()]
    res=np.empty(len(ev)); prev=float(last['MD']); ll=0.0
    for i in range(len(ev)):
        ds=max(md_v[i]-prev,1.0); rate=MOM*rate+VN*rng.standard_normal(N); pos=pos+rate*ds+PN*rng.standard_normal(N)
        tvt_p=np.clip(pos-z_v[i],tw_tvt[0]-100,tw_tvt[-1]+100); pos=tvt_p+z_v[i]
        eg=np.interp(tvt_p,tw_tvt,tw_gr); d=(gr_v[i]-eg)/gs; lk=np.maximum(np.exp(-0.5*np.minimum(d*d,600.)),1e-300)
        avg=float((w*lk).sum()); ll+=np.log(max(avg,1e-300)); w=w*lk; ws=w.sum(); w=w/ws if ws>0 else np.ones(N)/N
        if 1.0/(w*w).sum()<RESAMP*N:
            cum=np.cumsum(w);u0=rng.uniform(0,1.0/N);idx=np.clip(np.searchsorted(cum,u0+np.arange(N)/N),0,N-1)
            pos=pos[idx]+RP*rng.standard_normal(N);rate=rate[idx]+RR*rng.standard_normal(N);w=np.ones(N)/N
        res[i]=float(np.dot(w,pos-z_v[i])); prev=md_v[i]
    return res, ll

def pf_by_scale(hw,tw,n_seeds=24):
    preds=[];liks=[]
    for s in range(n_seeds):
        r,ll=run_pf(hw,tw,s); preds.append(r); liks.append(ll)
    preds=np.stack(preds,0); liks=np.array(liks); ln=liks-liks.max()
    out={}
    for sc in SCALES:
        wt=np.exp(ln/sc); wt/=wt.sum(); out['pf_scale_%g'%sc]=(wt[:,None]*preds).sum(0)
    return out

def beam_mean(hw,tw):
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    last_tvt=float(kn.iloc[-1]['TVT_input'])
    tw_s=tw.sort_values('TVT'); tw_tvt=tw_s['TVT'].values.astype(float); tw_gr=tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    gr_all=hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)
    hgr=gr_all[ev.index.to_numpy()]
    beams=[beam_search(hgr,tw_tvt,tw_gr,last_tvt,bs,mc,es,r) for (bs,mc,es,r,tag) in BEAMS]
    return np.stack(beams,0).mean(0)

def load(wid):
    return (pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv')),
            pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')))

# warm numba
_hw,_tw=load(well_ids[0]); beam_mean(_hw,_tw)
NW=50
wells=[]
for wid in well_ids[-NW:]:
    hw,tw=load(wid); km=hw['TVT_input'].notna()
    if km.sum()<30 or (~km).sum()<30: continue
    wells.append((hw,tw,hw['TVT'].values[np.where(~km.values)[0]]))
print('wells:',len(wells),flush=True)

strategies={'production_bin_selector':[], 'global_scale8_hold0.2':[], 'global_scale12_hold0.2':[], 'global_scale12_nohold':[], 'global_scale8_nohold':[]}
truth=[]
t0=time.time()
for k,(hw,tw,true) in enumerate(wells):
    pbs=pf_by_scale(hw,tw); bm=beam_mean(hw,tw)
    last_known=float(hw[hw['TVT_input'].notna()]['TVT_input'].iloc[-1])
    truth.append(true)
    strategies['production_bin_selector'].append(apply_variant(selector_code(hw),pbs,bm,last_known))
    strategies['global_scale8_hold0.2'].append(apply_variant('pf_scale_8_hold_0.2',pbs,bm,last_known))
    strategies['global_scale12_hold0.2'].append(apply_variant('pf_scale_12_hold_0.2',pbs,bm,last_known))
    strategies['global_scale12_nohold'].append(pbs['pf_scale_12'])
    strategies['global_scale8_nohold'].append(pbs['pf_scale_8'])
    if (k+1)%10==0: print('%d wells %.0fs'%(k+1,time.time()-t0),flush=True)
truth=np.concatenate(truth)
print('--- selector strategies, pooled RMSE ---',flush=True)
for name,v in strategies.items():
    print('%-26s RMSE=%.3f'%(name,prmse(truth,np.concatenate(v))),flush=True)
print('total %.0fs'%(time.time()-t0),flush=True)
