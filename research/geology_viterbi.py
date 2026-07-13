import numpy as np, pandas as pd, glob, os, time
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')
TRAIN_DIR='data/train'
wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
def inan(a):
    a=a.copy();n=len(a);i=np.arange(n);m=np.isnan(a)
    if m.all():return np.zeros(n)
    a[m]=np.interp(i[m],i[~m],a[~m]);return a
def prmse(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))

# feature for GR->formation: gr + rolling + grad
def grfeat(gr):
    s=pd.Series(gr)
    return np.stack([gr,s.rolling(5,center=True,min_periods=1).mean().values,
        s.rolling(15,center=True,min_periods=1).mean().values,np.gradient(gr),
        s.rolling(15,center=True,min_periods=1).std().fillna(0).values],1)

# --- train GR->formation classifier on typewells NOT in the eval set (last 60 held out) ---
holdout=set(wids[-60:])
rows_X=[];rows_y=[];labels=set()
for wid in wids:
    if wid in holdout: continue
    tw=pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'Geology' not in tw.columns: continue
    g=tw.dropna(subset=['Geology'])
    if len(g)<30: continue
    gr=g['GR'].fillna(g['GR'].mean()).values.astype(float)
    rows_X.append(grfeat(gr)); rows_y.extend(g['Geology'].values); labels.update(g['Geology'].unique())
labels=sorted(labels); lab2i={l:i for i,l in enumerate(labels)}
X=np.concatenate(rows_X); y=np.array([lab2i[v] for v in rows_y])
print('training formation classifier...',flush=True); t0=time.time()
clf=lgb.LGBMClassifier(n_estimators=150,num_leaves=31,learning_rate=0.05,n_jobs=4,verbose=-1)
clf.fit(X,y); print('trained %.0fs, %d formations'%(time.time()-t0,len(labels)),flush=True)

# --- DP alignment with optional formation term ---
def dp_align(cal_ev, tw_tvt, tw_gr, tw_form_idx, start, form_post, lam, es=100., mc=15., maxstep=3, band=120):
    n=len(cal_ev); nt=len(tw_gr)
    lo=max(0,start-band); hi=min(nt,start+n+band); idxs=np.arange(lo,hi); m=len(idxs)
    INF=1e18; ar=np.arange(m)
    # precompute formation-mismatch cost per (eval i, grid pos) is heavy; approximate: cost_form[i, gridpos]
    # form_post: (n, n_labels); tw_form_idx: (nt,) label index or -1
    tw_lab=tw_form_idx[idxs]  # (m,)
    def emis(i):
        e=(tw_gr[idxs]-cal_ev[i])**2/es
        if lam>0:
            valid=tw_lab>=0
            fp=np.zeros(m); fp[valid]=form_post[i, tw_lab[valid]]
            e=e + lam*(-np.log(np.clip(fp,1e-4,1.0)))
        return e
    prev=emis(0)+np.abs(idxs-start)*mc; back=np.zeros((n,m),np.int32)
    for i in range(1,n):
        e=emis(i); cur=np.full(m,INF); bk=np.zeros(m,np.int32)
        for d in range(-maxstep,maxstep+1):
            src=ar-d; ok=(src>=0)&(src<m); cand=np.full(m,INF); cand[ok]=prev[src[ok]]+abs(d)*mc
            tot=cand+e; better=tot<cur; cur=np.where(better,tot,cur); bk=np.where(better,src,bk)
        prev=cur;back[i]=bk
    j=int(np.argmin(prev));path=np.zeros(n,np.int32);path[n-1]=j
    for i in range(n-1,0,-1):j=back[i,j];path[i-1]=j
    return tw_tvt[idxs[path]]

allt=[];graw=[];gform=[]
t0=time.time()
for k,wid in enumerate(sorted(holdout)):
    hw=pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw=pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns or 'Geology' not in tw.columns: continue
    km=hw['TVT_input'].notna()
    if km.sum()<30 or (~km).sum()<30: continue
    tw_tvt=tw['TVT'].values.astype(float); tw_gr=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    tw_form=tw['Geology'].map(lambda x: lab2i.get(x,-1)).fillna(-1).values.astype(int)
    gr=inan(hw['GR'].values.astype(float))
    kn=hw[km]; kg=gr[:km.sum()]; twk=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr); v=np.isfinite(kg)&np.isfinite(twk)
    a,b=(np.polyfit(kg[v],twk[v],1) if v.sum()>=20 else (1.,0.))
    ei=np.where(~km.values)[0]; cal_ev=(gr*a+b)[ei]
    last_tvt=float(kn['TVT_input'].iloc[-1]); start=int(np.clip(np.searchsorted(tw_tvt,last_tvt),0,len(tw_tvt)-1))
    fp=clf.predict_proba(grfeat(gr)[ei])  # (n_ev, n_labels)
    true=hw['TVT'].values[ei]
    raw=dp_align(cal_ev,tw_tvt,tw_gr,tw_form,start,fp,lam=0.0)
    frm=dp_align(cal_ev,tw_tvt,tw_gr,tw_form,start,fp,lam=3.0)
    allt.append(true);graw.append(raw);gform.append(frm)
    if (k+1)%15==0: print('%d %.0fs'%(k+1,time.time()-t0),flush=True)
allt=np.concatenate(allt)
print('\nDP GR-only            %.3f'%prmse(allt,np.concatenate(graw)),flush=True)
print('DP GR + formation     %.3f  (Geology-constrained)'%prmse(allt,np.concatenate(gform)),flush=True)
print('total %.0fs'%(time.time()-t0),flush=True)
