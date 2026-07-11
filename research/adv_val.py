"""Idea 4: adversarial validation train-vs-test. Standalone, non-destructive diagnostic (writes a
JSON, never touches submission.csv). NOTE: locally CFG.DATA/'test' is a 3-well STUB (confirmed
identical to 3 train wells) -- this can only give a REAL answer when run inside an actual Kaggle
submission, where the hidden test set replaces the stub at scoring time. Smoke-tested here on the
stub to confirm the code runs; the printed AUC below is meaningless (train==test wells) by design."""
import numpy as np, pandas as pd, glob, os, json
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA_DIR = next((d for d in ('data','d:/ROGII/data') if os.path.isdir(d)), 'data')

def list_wells(split):
    return sorted({os.path.basename(f).split('__')[0]
                   for f in glob.glob(f'{DATA_DIR}/{split}/*__horizontal_well.csv')})

def well_features(wid, split):
    hw = pd.read_csv(f'{DATA_DIR}/{split}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{DATA_DIR}/{split}/{wid}__typewell.csv')
    kn = hw[hw['TVT_input'].notna()] if 'TVT_input' in hw.columns else hw.iloc[:0]
    ev = hw[hw['TVT_input'].isna()] if 'TVT_input' in hw.columns else hw
    gr = hw['GR'].dropna()
    md = hw['MD']
    z = hw['Z'] if 'Z' in hw.columns else pd.Series(dtype=float)
    feat = dict(
        n_rows=len(hw), n_known=len(kn), n_eval=len(ev),
        md_min=md.min(), md_max=md.max(), md_span=md.max()-md.min(),
        gr_mean=gr.mean(), gr_std=gr.std(), gr_min=gr.min(), gr_max=gr.max(),
        z_mean=z.mean() if len(z) else np.nan, z_std=z.std() if len(z) else np.nan,
        z_span=(z.max()-z.min()) if len(z) else np.nan,
        tw_rows=len(tw), tw_tvt_span=(tw['TVT'].max()-tw['TVT'].min()) if 'TVT' in tw.columns else np.nan,
        tw_gr_mean=tw['GR'].mean() if 'GR' in tw.columns else np.nan,
        tw_gr_std=tw['GR'].std() if 'GR' in tw.columns else np.nan,
        known_frac=len(kn)/max(1,len(hw)),
    )
    if 'X' in hw.columns and 'Y' in hw.columns:
        feat['x_mean']=hw['X'].mean(); feat['y_mean']=hw['Y'].mean()
        feat['xy_span']=float(np.hypot(hw['X'].max()-hw['X'].min(), hw['Y'].max()-hw['Y'].min()))
    if len(kn):
        feat['last_tvt']=float(kn['TVT_input'].iloc[-1])
    else:
        feat['last_tvt']=np.nan
    return feat

def run():
    train_w = list_wells('train'); test_w = list_wells('test')
    print(f'train wells={len(train_w)} test wells={len(test_w)}')
    rows=[]; labels=[]
    for w in train_w:
        try: rows.append(well_features(w,'train')); labels.append(0)
        except Exception as e: print('skip train',w,e)
    for w in test_w:
        try: rows.append(well_features(w,'test')); labels.append(1)
        except Exception as e: print('skip test',w,e)
    X = pd.DataFrame(rows); y = np.array(labels)
    if y.sum()==0 or y.sum()==len(y):
        print('WARNING: only one class present (train==test stub locally) -- AUC is meaningless here.')
    skf = StratifiedKFold(n_splits=min(5, max(2, y.sum())), shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    try:
        for tri, vai in skf.split(X, y):
            m = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                    min_child_samples=10, verbosity=-1)
            m.fit(X.iloc[tri], y[tri])
            oof[vai] = m.predict_proba(X.iloc[vai])[:,1]
        auc = roc_auc_score(y, oof) if len(set(y))>1 else float('nan')
    except Exception as e:
        print('CV failed (likely too few test wells locally):', e); auc=float('nan')
    m_full = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                 min_child_samples=10, verbosity=-1)
    m_full.fit(X, y)
    imp = sorted(zip(X.columns, m_full.feature_importances_), key=lambda t: -t[1])
    result = dict(n_train=len(train_w), n_test=len(test_w), auc=float(auc) if auc==auc else None,
                  top_features=[(n,int(i)) for n,i in imp[:10]],
                  verdict=('LOCAL STUB -- rerun inside a real Kaggle submission for a meaningful AUC'
                           if len(set(y))<2 or len(test_w)<20 else
                           ('distribution shift risk (AUC>0.6): consider dropping top features'
                            if auc>0.6 else 'no strong shift detected (AUC<=0.6): features look stable')))
    print(json.dumps(result, indent=2, default=str))
    return result

if __name__=='__main__':
    run()
