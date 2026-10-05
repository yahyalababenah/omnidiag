import sys, json, numpy as np
sys.path.insert(0,'/home/yahia/omnidiag_experiments/heart/scripts')
from p1_data import load_repo, SITES
from p4_common import encode
from glore import SiteHandle, fit_glore
from p8_10b_items_3_6 import frozen_l3_transform, newton_fit, RIDGE_LAMBDA
from sklearn.metrics import roc_auc_score, brier_score_loss
sig=lambda z:1/(1+np.exp(-np.clip(z,-35,35)))
raw=load_repo(); enc=encode(raw,cp_map=None); site=raw['site'].values; y=raw['HeartDisease'].values.astype(float)
out={}
out['prev']={s:float(y[site==s].mean()) for s in SITES}; out['n']={s:int((site==s).sum()) for s in SITES}
# full-data basis for convergence/mechanics experiments
tf=frozen_l3_transform(); X=tf.fit_transform(enc,y); p=X.shape[1]; out['p']=int(p)
# ---- optimum references (ridge same as GLORE)
D=np.column_stack([(site==s).astype(float) for s in SITES])
th,_=newton_fit(np.hstack([D,X]),y,ridge_lambda=RIDGE_LAMBDA,ridge_from=4); beta_star=th[4:]; alpha_star=th[:4]
th1,_=newton_fit(np.hstack([np.ones((len(y),1)),X]),y,ridge_lambda=RIDGE_LAMBDA,ridge_from=1); pooled_theta=th1
out['alpha_star']=dict(zip(SITES,map(float,alpha_star))); out['pooled_intercept']=float(th1[0])
# ---- GLORE convergence trace
sites=[SiteHandle(s,X[site==s],y[site==s]) for s in SITES]
r=fit_glore(sites,ridge_lambda=RIDGE_LAMBDA,tol=1e-12)
out['glore_ll']=r['loglik_history']; out['glore_iters']=r['n_iter']
# distance to optimum per iteration: rerun manually
dist=[]
for k in range(1,r['n_iter']+1):
    ss=[SiteHandle(s,X[site==s],y[site==s]) for s in SITES]
    rr=fit_glore(ss,ridge_lambda=RIDGE_LAMBDA,max_iter=k,tol=0)
    dist.append(float(np.linalg.norm(rr['beta']-beta_star)))
out['glore_dist']=dist
# ---- FedAvg (naive: shared intercept) simulation, full-batch local GD E steps, lr
def fedavg(Xs,ys,rounds=200,E=5,lr=0.5,intercept_local=False,Xte=None):
    K=len(Xs); n=np.array([len(v) for v in ys]); w=n/n.sum()
    beta=np.zeros(Xs[0].shape[1]); a=np.zeros(K) if intercept_local else np.zeros(1)
    hist=[]
    for t in range(rounds):
        bs=[];as_=[]
        for k in range(K):
            b=beta.copy(); ak=(a[k] if intercept_local else a[0])
            for _ in range(E):
                pr=sig(ak+Xs[k]@b); g=(ys[k]-pr)
                b=b+lr*(Xs[k].T@g/n[k]-RIDGE_LAMBDA*b/n[k]); ak=ak+lr*g.mean()
            bs.append(b); as_.append(ak)
        beta=sum(w[k]*bs[k] for k in range(K))
        if intercept_local: a=np.array(as_)
        else: a=np.array([sum(w[k]*as_[k] for k in range(K))])
        hist.append((beta.copy(),a.copy()))
    return hist
Xs=[X[site==s] for s in SITES]; ys=[y[site==s] for s in SITES]
for name,loc in (('fedavg_shared',False),('fedavg_localint',True)):
    h=fedavg(Xs,ys,rounds=300,intercept_local=loc)
    out[name+'_dist']=[float(np.linalg.norm(b-beta_star)) for b,_ in h]
    out[name+'_ll']=[]
    for b,a in h:
        ll=0
        for k in range(4):
            ak=a[k] if loc else a[0]; e=ak+Xs[k]@b; ll+=(ys[k]*e-np.logaddexp(0,e)).sum()
        out[name+'_ll'].append(float(ll))
    b,a=h[-1]
    out[name+'_calib']={s:dict(pred=float(np.mean(sig((a[k] if loc else a[0])+Xs[k]@b))),obs=float(ys[k].mean())) for k,s in enumerate(SITES)}
# non-IID drift: vary local epochs E, rounds to reach ||beta-beta*||<0.1 ... record final dist after 100 rounds
out['fedavg_E_sweep']={}
for E in (1,5,20,50):
    h=fedavg(Xs,ys,rounds=100,E=E,intercept_local=True)
    out['fedavg_E_sweep'][E]=[float(np.linalg.norm(b-beta_star)) for b,_ in h]
# ---- clean LOSO: methods
res={m:{} for m in ('local_only','pooled_shared_int','fedavg_shared','glore_local_int','glore_coldstart')}
brier={m:{} for m in res}
for held in SITES:
    trs=[s for s in SITES if s!=held]; tr=np.isin(site,trs); te=site==held
    t=frozen_l3_transform(); Xtr=t.fit_transform(enc[tr],y[tr]); Xte=t.transform(enc[te]); ytr=y[tr]; yte=y[te]; st=site[tr]
    # glore
    hs=[SiteHandle(s,Xtr[st==s],ytr[st==s]) for s in trs]; g=fit_glore(hs,ridge_lambda=RIDGE_LAMBDA)
    bg=g['beta']; pp=ytr.mean()
    # local intercept (needs held-out labels: warm start) and cold start (pooled-prevalence matched)
    a=0.
    for _ in range(50):
        pr=sig(a+Xte@bg); d=(yte-pr).sum()/max((pr*(1-pr)).sum(),1e-9); a+=d
        if abs(d)<1e-10: break
    acold=np.mean([h.alpha for h in hs])
    res['glore_local_int'][held]=roc_auc_score(yte,sig(a+Xte@bg)); brier['glore_local_int'][held]=brier_score_loss(yte,sig(a+Xte@bg))
    res['glore_coldstart'][held]=roc_auc_score(yte,sig(acold+Xte@bg)); brier['glore_coldstart'][held]=brier_score_loss(yte,sig(acold+Xte@bg))
    # pooled shared intercept (central)
    t1,_=newton_fit(np.hstack([np.ones((len(ytr),1)),Xtr]),ytr,ridge_lambda=RIDGE_LAMBDA,ridge_from=1)
    pr=sig(t1[0]+Xte@t1[1:]); res['pooled_shared_int'][held]=roc_auc_score(yte,pr); brier['pooled_shared_int'][held]=brier_score_loss(yte,pr)
    # fedavg shared
    h=fedavg([Xtr[st==s] for s in trs],[ytr[st==s] for s in trs],rounds=300,intercept_local=False)
    b,a2=h[-1]; pr=sig(a2[0]+Xte@b); res['fedavg_shared'][held]=roc_auc_score(yte,pr); brier['fedavg_shared'][held]=brier_score_loss(yte,pr)
    # local only: best-case, trained on a single other site (mean over the 3)
    la=[];lb=[]
    for s in trs:
        m=st==s; t2,_=newton_fit(np.hstack([np.ones((m.sum(),1)),Xtr[m]]),ytr[m],ridge_lambda=RIDGE_LAMBDA,ridge_from=1)
        pr=sig(t2[0]+Xte@t2[1:]); la.append(roc_auc_score(yte,pr)); lb.append(brier_score_loss(yte,pr))
    res['local_only'][held]=float(np.mean(la)); brier['local_only'][held]=float(np.mean(lb))
out['loso_auc']={m:{k:float(v) for k,v in d.items()} for m,d in res.items()}
out['loso_brier']={m:{k:float(v) for k,v in d.items()} for m,d in brier.items()}
# communication cost
K=4; out['comm']=dict(floats_per_site_per_iter=int(1+p+1+p+p*p), p=int(p), bytes_per_iter_all_sites=int(K*(2+2*p+p*p)*8), raw_bytes=int(X.size*8+len(y)*8))
json.dump(out,open('fl_exp.json','w'),indent=1)
print(json.dumps({k:out[k] for k in ('prev','n','p','alpha_star','pooled_intercept','glore_iters','loso_auc','loso_brier','comm','fedavg_shared_calib','fedavg_localint_calib')},indent=1))
print('glore dist',out['glore_dist']); print('fedavg shared dist@10,50,300',[out['fedavg_shared_dist'][i] for i in (9,49,299)]); print('fedavg local dist',[out['fedavg_localint_dist'][i] for i in (9,49,299)])
