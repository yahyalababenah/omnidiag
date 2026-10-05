import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
o=json.load(open('fl_exp.json')); S=list(o['prev']); lab=['Cleveland','Hungarian','Switzerland','Long Beach']
C=dict(glore='#1b9e77',fed='#d95f02',fedl='#7570b3',pool='#666666',loc='#e7298a')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25})
def save(f,n): f.tight_layout(); f.savefig(f'fig{n}.png',dpi=140); plt.close(f)
# 1 prevalence & intercepts
f,ax=plt.subplots(1,2,figsize=(10,4)); x=np.arange(4)
ax[0].bar(x,[o['prev'][s]*100 for s in S],color=C['glore']); ax[0].set_xticks(x,lab,rotation=15); ax[0].set_ylabel('Disease prevalence %'); ax[0].set_title('Label shift between sites')
for i,s in enumerate(S): ax[0].text(i,o['prev'][s]*100+1,f"n={o['n'][s]}",ha='center',fontsize=9)
ax[1].bar(x,[o['alpha_star'][s] for s in S],color=C['glore'],label='per-site $\\alpha_s$ (GLORE)'); ax[1].axhline(o['pooled_intercept'],color=C['fed'],ls='--',label=f"single shared intercept = {o['pooled_intercept']:.2f}")
ax[1].set_xticks(x,lab,rotation=15); ax[1].set_ylabel('Intercept (logit)'); ax[1].set_title('Intercepts: one number cannot fit all'); ax[1].legend(fontsize=9)
save(f,1)
# 2 convergence
f,ax=plt.subplots(figsize=(7,4.4)); it=np.arange(1,8)
ax.semilogy(it,np.maximum(o['glore_dist'],1e-16),'o-',color=C['glore'],label='GLORE (Newton): 7 rounds')
r=np.arange(1,301); ax.semilogy(r,o['fedavg_localint_dist'],color=C['fedl'],label='FedAvg + local intercept (E=5)'); ax.semilogy(r,o['fedavg_shared_dist'],color=C['fed'],label='FedAvg shared intercept (E=5)')
ax.set_xscale('log'); ax.set_xlabel('Communication rounds'); ax.set_ylabel(r'$\|\beta-\beta^*\|_2$'); ax.set_title('Distance to the exact optimum'); ax.legend(fontsize=9)
save(f,2)
# 3 calibration
f,ax=plt.subplots(figsize=(7,4.4)); w=.27
ax.bar(x-w,[o['prev'][s]*100 for s in S],w,color='#bbbbbb',label='Observed'); ax.bar(x,[o['fedavg_shared_calib'][s]['pred']*100 for s in S],w,color=C['fed'],label='FedAvg shared intercept')
ax.bar(x+w,[o['fedavg_localint_calib'][s]['pred']*100 for s in S],w,color=C['glore'],label='Per-site intercept')
ax.set_xticks(x,lab,rotation=15); ax.set_ylabel('Mean predicted risk %'); ax.set_title('Calibration-in-the-large (in-sample, per site)'); ax.legend(fontsize=9)
save(f,3)
# 4/5 LOSO
M=[('local_only','Local-only',C['loc']),('pooled_shared_int','Pooled central',C['pool']),('fedavg_shared','FedAvg',C['fed']),('glore_local_int','GLORE',C['glore'])]
for n,key,ttl,yl in ((4,'loso_auc','Leave-one-site-out AUC (clean basis)','AUC'),(5,'loso_brier','Leave-one-site-out Brier (lower=better)','Brier')):
    f,ax=plt.subplots(figsize=(9,4.4)); w=.2
    for j,(m,l,c) in enumerate(M):
        v=[o[key][m][s] for s in S]; ax.bar(x+(j-1.5)*w,v,w,color=c,label=f'{l} (mean {np.mean(v):.3f})')
    ax.set_xticks(x,lab); ax.set_ylabel(yl); ax.set_title(ttl)
    if key=='loso_auc': ax.set_ylim(.6,.9)
    ax.legend(fontsize=8,ncol=2); save(f,n)
# 6 client drift
f,ax=plt.subplots(figsize=(7,4.4))
for E,c in zip(('1','5','20','50'),plt.cm.viridis([.1,.4,.65,.9])): ax.semilogy(o['fedavg_E_sweep'][E],color=c,label=f'E={E} local steps')
ax.set_xlabel('Round'); ax.set_ylabel(r'$\|\beta-\beta^*\|_2$'); ax.set_title('FedAvg client drift (non-IID sites)'); ax.legend(fontsize=9)
save(f,6)
# 7 comm
f,ax=plt.subplots(figsize=(6,4)); cm=o['comm']; v=[cm['raw_bytes']/1e3,cm['bytes_per_iter_all_sites']/1e3,cm['bytes_per_iter_all_sites']*7/1e3]
ax.bar(['Raw data\n(4 sites)','GLORE\n1 iteration','GLORE\n7 iterations'],v,color=[C['pool'],C['glore'],C['glore']]); ax.set_ylabel('KB'); ax.set_title('Bytes on the wire')
for i,t in enumerate(v): ax.text(i,t+2,f'{t:.0f}',ha='center')
save(f,7)
for m,_,_ in M: print(m,np.mean(list(o['loso_auc'][m].values())).round(4),np.mean(list(o['loso_brier'][m].values())).round(4))
print('cold brier',np.mean(list(o['loso_brier']['glore_coldstart'].values())))
