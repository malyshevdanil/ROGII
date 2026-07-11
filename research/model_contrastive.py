"""Phase 1.2 premise-test: can a LEARNED window matcher localize position better than naive GR-Gaussian?
Contrastive (InfoNCE): a horizontal GR window must pick its correct typewell window (centered at true TVT)
among all typewell windows. Eval = localization RMSE on HELD-OUT known points (true TVT known there).
If << ~31ft (memory's pure-GR oracle) -> learned likelihood beats naive -> integrate into PF (worth it).
If ~31ft+ -> GR self-similarity wall confirmed -> matching direction dead."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, sys, os, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import prep
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

CFG=dict(W=33, d=48, ppw=24, epochs=20, bs=16, lr=2e-3, wd=1e-4, drop=0.1, temp=0.1, tag='contrastive')
for a in sys.argv[1:]:
    k,v=a.split('=');
    if k in CFG: CFG[k]=type(CFG[k])(v)
W=CFG['W']; HW=W//2
data=prep.load(); tr,va=prep.split(data)

def prep_well(w):
    f=w['feat']; g=w['g_gr']; gm,gs=float(g.mean()),float(g.std()+1e-6)
    caln=((f[:,4]-gm)/gs).astype(np.float32)          # horizontal calibrated GR, typewell-normalized
    gn=((g-gm)/gs).astype(np.float32)                 # typewell GR grid, normalized
    gtvt=w['g_tvt'].astype(np.float32); dt=(gtvt[-1]-gtvt[0])/(len(gtvt)-1)
    tvt_at=(w['target']+w['last_tvt']).astype(np.float32)   # true TVT per horizontal point
    mkn=f[:,9]>0.5
    return caln,gn,gtvt,dt,tvt_at,mkn
TR=[prep_well(w) for w in tr]; VA=[prep_well(w) for w in va]

class Enc(nn.Module):
    def __init__(s,d,drop):
        super().__init__()
        s.net=nn.Sequential(nn.Conv1d(1,d,5,padding=2),nn.GroupNorm(4,d),nn.GELU(),nn.Dropout(drop),
                            nn.Conv1d(d,d,5,padding=2),nn.GroupNorm(4,d),nn.GELU(),nn.Dropout(drop),
                            nn.Conv1d(d,d,5,padding=2),nn.GroupNorm(4,d),nn.GELU())
    def forward(s,x):            # x:(B,1,W) -> (B,d)
        h=s.net(x); return F.normalize(h.mean(-1),dim=1)
he=Enc(CFG['d'],CFG['drop']); te=Enc(CFG['d'],CFG['drop'])
opt=torch.optim.AdamW(list(he.parameters())+list(te.parameters()),lr=CFG['lr'],weight_decay=CFG['wd'])

def tw_windows(gn):             # all length-W windows of the typewell grid -> (Tg-W+1, W)
    Tg=len(gn); idx=np.arange(Tg-W+1)[:,None]+np.arange(W)[None,:]
    return gn[idx], np.arange(HW,Tg-HW)   # windows, center grid-index of each

def sample_batch(wells, ppw):
    Hs=[]; Ts=[]; TGT=[]; owner=[]
    tw_emb_input=[]; centers=[]
    for wi,(caln,gn,gtvt,dt,tvt_at,mkn) in enumerate(wells):
        tw_win,ctr_idx=tw_windows(gn)                      # (M,W),(M,)
        ctr_tvt=gtvt[ctr_idx]
        tw_emb_input.append(tw_win); centers.append(ctr_tvt)
        # pick known points with a valid horizontal window
        n=len(caln); ok=np.where(mkn & (np.arange(n)>=HW) & (np.arange(n)<n-HW))[0]
        if len(ok)==0: continue
        pick=np.random.choice(ok, min(ppw,len(ok)), replace=False)
        for p in pick:
            Hs.append(caln[p-HW:p+HW+1])
            # target = index of typewell window whose center TVT is nearest true tvt_at[p]
            j=int(np.argmin(np.abs(ctr_tvt-tvt_at[p]))); TGT.append(j); owner.append(wi)
    return Hs,tw_emb_input,centers,TGT,owner

def run_epoch(train=True):
    src=TR if train else VA
    he.train(train); te.train(train)
    tot=0; nb=0; loc_err=[]
    order=np.arange(len(src));
    if train: np.random.shuffle(order)
    for i in range(0,len(src),CFG['bs']):
        wells=[src[j] for j in order[i:i+CFG['bs']]]
        Hs,tw_in,centers,TGT,owner=sample_batch(wells,CFG['ppw'] if train else 40)
        if len(Hs)==0: continue
        H=torch.from_numpy(np.stack(Hs)[:,None].astype(np.float32))
        with torch.set_grad_enabled(train):
            hemb=he(H)                                     # (P,d)
            # encode each well's typewell windows once
            temb=[];
            for tw in tw_in:
                temb.append(te(torch.from_numpy(tw[:,None].astype(np.float32))))
            loss=0.0; cnt=0
            for p in range(len(Hs)):
                wi=owner[p]; T=temb[wi]                     # (M,d)
                sim=(hemb[p:p+1]@T.T).squeeze(0)/CFG['temp']  # (M,)
                loss=loss+F.cross_entropy(sim[None],torch.tensor([TGT[p]]))
                cnt+=1
                if not train:
                    j=int(sim.argmax()); loc_err.append((centers[wi][j]-centers[wi][TGT[p]]))
            loss=loss/max(cnt,1)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item(); nb+=1
    if not train:
        le=np.array(loc_err); return tot/max(nb,1), float(np.sqrt(np.mean(le**2)))
    return tot/max(nb,1), None

print('training %s W=%d | params=%.0fk'%(CFG['tag'],W,sum(p.numel() for p in list(he.parameters())+list(te.parameters()))/1e3),flush=True)
t0=time.time(); best=99
for ep in range(CFG['epochs']):
    trl,_=run_epoch(True)
    if (ep+1)%2==0 or ep==CFG['epochs']-1:
        _,loc=run_epoch(False); best=min(best,loc)
        print('  ep%2d trainCE %.3f | val localization RMSE %.2f ft (best %.2f) %.0fs'%(ep+1,trl,loc,best,time.time()-t0),flush=True)
print('FINAL %s val localization RMSE %.2f ft | best %.2f'%(CFG['tag'],loc,best),flush=True)
print('(reference: memory says pure-GR oracle registration ~31ft; flat/continuity is what PF adds on top.)',flush=True)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'RESULTS.jsonl'),'a').write(
    json.dumps(dict(tag=CFG['tag'],loc_rmse=best,cfg=CFG,t=time.time()-t0))+'\n')
