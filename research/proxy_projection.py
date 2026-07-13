import numpy as np, pickle, time
P=pickle.load(open(r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad\proxy.pkl','rb'))
DATA=P['DATA']; gbm=P['gbm']; wids=P['wids']
def prmse(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
allt=np.concatenate([DATA[w]['true'] for w in wids])
blend={w:0.7*DATA[w]['sp45']+0.3*gbm[w] for w in wids}

def robfit(md,y,deg):
    x=np.asarray(md,float); x0=x[0]; xs=max(x.max()-x.min(),1e-6); xk=(x-x0)/xs
    if len(x)<deg+2: return np.asarray(y,float).copy()
    c=np.polyfit(xk,y,deg)
    for _ in range(4):
        r=y-np.polyval(c,xk); sc=np.median(np.abs(r))*1.4826+1e-6
        c=np.polyfit(xk,y,deg,w=1.0/(1.0+(r/(2*sc))**2))
    return np.polyval(c,xk)

def piecewise_linear(md,y,n_seg):
    # fit continuous piecewise-linear with n_seg segments (equal MD breakpoints), robust
    x=np.asarray(md,float); yy=np.asarray(y,float); n=len(x)
    if n<n_seg*3: return robfit(md,y,1)
    bps=np.linspace(x.min(),x.max(),n_seg+1)
    # design matrix: hinge basis
    cols=[np.ones(n),x]
    for k in bps[1:-1]:
        cols.append(np.maximum(x-k,0))
    Xd=np.stack(cols,1)
    w=np.ones(n)
    for _ in range(4):
        W=np.sqrt(w)[:,None]
        coef,*_=np.linalg.lstsq(Xd*W,yy*np.sqrt(w),rcond=None)
        r=yy-Xd@coef; sc=np.median(np.abs(r))*1.4826+1e-6; w=1.0/(1.0+(r/(2*sc))**2)
    return Xd@coef

def project(mode, blend_w):
    out={}
    for w in wids:
        d=DATA[w]; pred=blend[w]; surf=pred+d['z']; anchor=d['lt']+d['z'][0]
        s=surf-anchor
        if mode=='deg2': fit=robfit(d['md'],s,2)
        elif mode=='deg4': fit=robfit(d['md'],s,4)
        elif mode=='deg6': fit=robfit(d['md'],s,6)
        elif mode=='pl2': fit=piecewise_linear(d['md'],s,2)
        elif mode=='pl3': fit=piecewise_linear(d['md'],s,3)
        elif mode=='pl5': fit=piecewise_linear(d['md'],s,5)
        elif mode=='pl8': fit=piecewise_linear(d['md'],s,8)
        fit=(fit+anchor)-d['z']
        out[w]=(1-blend_w)*pred+blend_w*fit
    return prmse(allt,np.concatenate([out[w] for w in wids]))

print('baseline blend (no proj)  %.3f'%prmse(allt,np.concatenate([blend[w] for w in wids])),flush=True)
print('\n--- projection mode x blend weight ---',flush=True)
t0=time.time()
for mode in ['deg2','deg4','deg6','pl2','pl3','pl5','pl8']:
    row=[]
    for bw in [0.5,0.75,0.9]:
        row.append('%s=%.3f'%(('bw%.2f'%bw),project(mode,bw)))
    print('%-6s  '%mode+'  '.join(row),flush=True)
print('time %.0fs'%(time.time()-t0),flush=True)
