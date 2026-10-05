import sys, json, numpy as np
sys.path.insert(0,'/home/yahia/omnidiag_experiments/heart/scripts')
from p1_data import load_repo, SITES
from p4_common import encode
from glore import SiteHandle, fit_glore
from p8_10b_items_3_6 import frozen_l3_transform, RIDGE_LAMBDA
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
np.set_printoptions(precision=3,suppress=True,linewidth=200)
raw=load_repo(); enc=encode(raw,cp_map=None); site=raw['site'].values; y=raw['HeartDisease'].values.astype(float)
X=frozen_l3_transform().fit_transform(enc,y); p=X.shape[1]
sites=[SiteHandle(s,X[site==s],y[site==s]) for s in SITES]
res={}
for tag,(alpha0,beta0) in {'iter0 (theta=0)':(0.,np.zeros(p))}.items():
    st=[s.local_stats(beta0) for s in sites]
    print('=== ',tag)
    for s,t in zip(SITES,st):
        ev=np.linalg.eigvalsh(-t['h_bb'])
        print(f"{s:14s} n={len(s)}  g_alpha={t['g_alpha']:8.2f}  h_aa={t['h_aa']:8.2f}  |g_beta|={np.linalg.norm(t['g_beta']):7.2f}  |H_bb|_F={np.linalg.norm(t['h_bb']):7.2f}  eig(-H_bb) min/max={ev.min():.3f}/{ev.max():.2f}")
    print('g_beta first 6 per site:'); 
    for s,t in zip(SITES,st): print(f"  {s:14s}",t['g_beta'][:6])
    print('H_bb[:4,:4] per site:')
    for s,t in zip(SITES,st): print(f"  {s}\n",t['h_bb'][:4,:4])
    G=sum(t['g_beta'] for t in st); H=sum(t['h_bb'] for t in st)
    # centralized on pooled rows with the same per-site alpha (all 0)
    p0=np.full(len(y),.5); r=y-p0; w=.25
    Gc=X.T@r; Hc=-(X*w).T@X
    print('sum_site g_beta - central g_beta  max abs =',np.abs(G-Gc).max(),'| H diff max abs =',np.abs(H-Hc).max())
    # one Newton step from the sums
    K=4; dim=K+p; Hf=np.zeros((dim,dim)); g=np.zeros(dim)
    for k,t in enumerate(st): Hf[k,k]=t['h_aa']; Hf[k,K:]=t['h_ab']; Hf[K:,k]=t['h_ab']; g[k]=t['g_alpha']
    Hf[K:,K:]=H-RIDGE_LAMBDA*np.eye(p); g[K:]=G-RIDGE_LAMBDA*beta0
    d=np.linalg.solve(-Hf,g); print('Newton step: d_alpha =',d[:K],'| |d_beta| =',np.linalg.norm(d[K:]))
# at convergence
r=fit_glore(sites,ridge_lambda=RIDGE_LAMBDA,tol=1e-12); b=r['beta']
st=[s.local_stats(b) for s in sites]
print('=== converged (7 iters)')
G=sum(t['g_beta'] for t in st)-RIDGE_LAMBDA*b
for s,t in zip(SITES,st): print(f"{s:14s} g_alpha={t['g_alpha']:+.2e} |g_beta|={np.linalg.norm(t['g_beta']):7.3f}")
print('global |g_beta| (after ridge) =',np.linalg.norm(G),' sum g_alpha_s per site ~0 each')
# plot: heatmaps of H_bb per site at theta=0 and bar of g_beta
for s_ in sites: s_.alpha=0.0  # fit_glore leaves converged alphas on the handles
st0=[s_.local_stats(np.zeros(p)) for s_ in sites]
f,ax=plt.subplots(2,4,figsize=(16,7))
for k,(s,t) in enumerate(zip(SITES,st0)):
    im=ax[0,k].imshow(-t['h_bb'],cmap='viridis'); ax[0,k].set_title(f'{s}: $-H_{{\\beta\\beta}}$ (n={sites[k].n})'); plt.colorbar(im,ax=ax[0,k],fraction=.046)
    ax[1,k].bar(range(p),t['g_beta'],color='#1b9e77'); ax[1,k].set_title(f'{s}: $g_\\beta$ (θ=0)  $g_\\alpha$={t["g_alpha"]:.1f}'); ax[1,k].set_xlabel('feature index')
f.tight_layout(); f.savefig('fig8_grad_hess.png',dpi=130)
f,ax=plt.subplots(figsize=(7,4)); 
nm=[f'{s}' for s in SITES]; gs=[np.linalg.norm(t['g_beta']) for t in st0]; ax.bar(nm,gs,color='#1b9e77'); ax.set_ylabel(r'$\|g_\beta\|_2$ at θ=0'); ax.set_title('Per-site gradient norm; sum = global gradient')
f.tight_layout(); f.savefig('fig9_gnorm.png',dpi=130)
