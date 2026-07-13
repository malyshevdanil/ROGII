import numpy as np, pandas as pd, glob, os, time, sys
sys.path.insert(0, r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad')
import beam_mod
from beam_mod import beam_search, BEAMS

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def prmse(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);return float(np.sqrt(np.mean((a-b)**2)))

def run_beam_ensemble_custom(hw, tw, calibrate=False):
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return None,None
    last_tvt=float(kn.iloc[-1]['TVT_input'])
    tw_s=tw.sort_values('TVT'); tw_tvt=tw_s['TVT'].values.astype(float); tw_gr=tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    gr_all=hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)
    if calibrate:
        kn_gr=kn['GR'].values.astype(float); tw_at_k=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr)
        v=np.isfinite(kn_gr)&np.isfinite(tw_at_k)
        if v.sum()>=20:
            a,b=np.polyfit(kn_gr[v],tw_at_k[v],1); gr_all=gr_all*a+b
    hgr=gr_all[ev.index.to_numpy()]
    beams=[beam_search(hgr,tw_tvt,tw_gr,last_tvt,bs,mc,es,r) for (bs,mc,es,r,tag) in BEAMS]
    beam_mean=np.stack(beams,0).mean(0)
    return beam_mean, ev.index.to_numpy()

def load(wid):
    return (pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv')),
            pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')))

# warm up numba
_hw,_tw=load(well_ids[0])
run_beam_ensemble_custom(_hw,_tw,calibrate=False)
print('numba warmed',flush=True)

NW=60
wells=[]
for wid in well_ids[-NW:]:
    hw,tw=load(wid); km=hw['TVT_input'].notna()
    if km.sum()<30 or (~km).sum()<30: continue
    wells.append((hw,tw,hw['TVT'].values[np.where(~km.values)[0]]))
print('wells:',len(wells),flush=True)

t0=time.time()
t_all,raw_all,cal_all=[],[],[]
for k,(hw,tw,true) in enumerate(wells):
    braw,ei=run_beam_ensemble_custom(hw,tw,calibrate=False)
    bcal,_=run_beam_ensemble_custom(hw,tw,calibrate=True)
    t_all.append(true); raw_all.append(braw); cal_all.append(bcal)
    if (k+1)%20==0: print('%d wells %.0fs'%(k+1,time.time()-t0),flush=True)
t_all=np.concatenate(t_all)
print('--- Real beam-search ensemble: raw vs heel-calibrated GR ---',flush=True)
print('beam RAW GR         RMSE=%.3f'%prmse(t_all,np.concatenate(raw_all)),flush=True)
print('beam CALIBRATED GR  RMSE=%.3f'%prmse(t_all,np.concatenate(cal_all)),flush=True)
print('total %.0fs'%(time.time()-t0),flush=True)
