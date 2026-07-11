"""Idea 1 (revisited properly): confidence-gated shrinkage. Train a LightGBM meta-model to predict
|sp45-true| per eval point from the 16 'own' features (GroupKFold by well, no leakage), then use the
OUT-OF-FOLD predicted error to gate per-point shrink toward last_tvt. Compare vs the flat-alpha wall-hedge."""
import pickle, numpy as np
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

proxy=pickle.load(open('../proxy.pkl','rb')); D=proxy['DATA']
wids=list(D.keys())
X=[]; y=[]; grp=[]; sp45=[]; true=[]; lt=[]; wid_of=[]
for w in wids:
    d=D[w]; n=len(d['true'])
    X.append(d['own']); y.append(np.abs(d['sp45']-d['true']))
    sp45.append(d['sp45']); true.append(d['true']); lt.append(d['lt']*np.ones(n))
    grp.extend([w]*n); wid_of.extend([w]*n)
X=np.concatenate(X); y=np.concatenate(y); sp45=np.concatenate(sp45); true=np.concatenate(true); lt=np.concatenate(lt)
grp=np.array(grp)
print('total points',len(X),'wells',len(wids))

gkf=GroupKFold(n_splits=5)
oof_pred=np.zeros(len(X))
for fold,(tri,vai) in enumerate(gkf.split(X,y,groups=grp)):
    m=lgb.LGBMRegressor(n_estimators=300,num_leaves=31,learning_rate=0.05,min_child_samples=50,
                         subsample=0.8,colsample_bytree=0.8,verbosity=-1)
    m.fit(X[tri],y[tri])
    oof_pred[vai]=m.predict(X[vai])
    print(' fold%d done, train %d val %d'%(fold,len(tri),len(vai)))

print('corr(oof predicted |err|, true |err|) = %.3f'%np.corrcoef(oof_pred,y)[0,1])

def pooled(pred, mask=None):
    t = true if mask is None else true[mask]
    return float(np.sqrt(np.mean((pred-t)**2)))
print('baseline sp45 pooled:', round(pooled(sp45),4))
print('baseline flat(lt) pooled:', round(pooled(lt),4))

# whole-set split for honest alpha-mapping fit (train half wells, eval other half)
uniq=np.array(wids); rng=np.random.default_rng(7); rng.shuffle(uniq)
trw=set(uniq[:len(uniq)//2]); vaw=set(uniq[len(uniq)//2:])
tr_mask=np.array([w in trw for w in wid_of]); va_mask=~tr_mask

def gated_shrink(scale,amax,pred_err,mask=None):
    a=np.clip(pred_err/scale,0,amax)
    out=(1-a)*sp45+a*lt
    return out
print('\n=== confidence-gated shrink: sweep (scale,amax), fit on TRAIN wells, eval on VAL wells ===')
best=(99,None)
for scale in [5,8,10,15,20,30]:
    for amax in [0.2,0.35,0.5,0.7,1.0]:
        out=gated_shrink(scale,amax,oof_pred)
        tr_score=pooled(out[tr_mask],tr_mask)
        if tr_score<best[0]: best=(tr_score,(scale,amax))
print('best (scale,amax) on TRAIN =',best[1],'train pooled=%.4f'%best[0])
scale,amax=best[1]
out=gated_shrink(scale,amax,oof_pred)
print('VAL pooled with gated shrink:', round(pooled(out[va_mask],va_mask),4), ' | VAL baseline sp45:', round(pooled(sp45[va_mask],va_mask),4))

print('\n=== compare: simple GLOBAL alpha shrink (wall-hedge, no meta-model) on the SAME split ===')
best2=(99,0)
for a in [0,0.02,0.05,0.08,0.1,0.15,0.2,0.3]:
    out=(1-a)*sp45+a*lt
    tr=pooled(out[tr_mask],tr_mask)
    if tr<best2[0]: best2=(tr,a)
out=(1-best2[1])*sp45+best2[1]*lt
print('best global a on TRAIN=%.2f -> VAL pooled=%.4f (baseline %.4f)'%(best2[1],pooled(out[va_mask],va_mask),pooled(sp45[va_mask],va_mask)))

print('\n=== feature importance (meta-model, last fold) ===')
names=['md_since','gr','gr_grad','gr_rstd','z','dzdmd','cal_gr','tda-20','tda-10','tda-5','tda0','tda5','tda10','tda20','sin_azi','cos_azi']
imp=m.feature_importances_
for n,i in sorted(zip(names,imp),key=lambda t:-t[1])[:8]: print('  %-10s %d'%(n,i))
