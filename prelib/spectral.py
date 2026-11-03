"""Heat-kernel diagnostics using zero-mode-deflated stochastic Lanczos quadrature."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh_tridiagonal
from .common import run_dir,save_table,write_json
from .graphs import network_data
from .metrics import adjacency


def lanczos_measure(L,v,steps,zero=None):
    v=np.asarray(v,float)
    if zero is not None:v=v-zero*np.dot(zero,v)
    norm2=float(v@v)
    if norm2<1e-22:return np.array([]),np.array([]),0.
    q=v/np.sqrt(norm2);basis=np.empty((min(steps,len(v)),len(v)));alpha=[];beta=[];last=np.zeros(len(v));b=0.
    for j in range(min(steps,len(v))):
        basis[j]=q;z=L@q-b*last;a=float(q@z);z=z-a*q
        # Full reorthogonalization prevents ghost Ritz values on modular graphs.
        Q=basis[:j+1]
        for _ in range(2):z-=Q.T@(Q@z)
        if zero is not None:z-=zero*np.dot(zero,z)
        next_b=float(np.linalg.norm(z));alpha.append(a)
        if next_b<1e-12 or j+1==min(steps,len(v)):break
        beta.append(next_b);last,q=q,z/next_b;b=next_b
    if len(alpha)==1:return np.asarray(alpha),np.array([1.]),norm2
    values,U=eigh_tridiagonal(np.asarray(alpha),np.asarray(beta))
    return values,U[0]**2,norm2


def heat_slq(B,taus,probes=16,steps=48,seed=1):
    B=sparse.csr_matrix(B,dtype=float);n=B.shape[0];taus=np.asarray(taus,float)
    degree=np.asarray(B.sum(1)).ravel()
    if n==0:raise ValueError('Empty component.')
    if n==1:
        return pd.DataFrame({'tau':taus,'return_probability':1.,'return_low':1.,'return_high':1.,
            'entropy':0.,'entropy_normalized':0.,'entropy_se':0.,'heat_capacity':0.,'heat_se':0.}),np.array([0.]),np.array([1.])
    if (B-B.T).nnz or (B.data.size and B.data.min()<0):raise ValueError('Laplacian input must be symmetric and nonnegative.')
    if sparse.csgraph.connected_components(B,directed=False)[0]!=1:raise ValueError('Use one connected component for the deflated heat estimator.')
    if np.any(degree<=0):raise ValueError('Heat diagnostics require one connected, nontrivial component.')
    inv=1/np.sqrt(degree);L=sparse.eye(n,format='csr')-sparse.diags(inv)@B@sparse.diags(inv)
    zero=np.sqrt(degree);zero/=np.linalg.norm(zero)
    rng=np.random.default_rng(seed);z0=[];z1=[];z2=[];density_x=[0.];density_w=[1.]
    for _ in range(probes):
        v=rng.normal(size=n);v-=zero*np.dot(zero,v)
        v*=np.sqrt((n-1)/max(float(v@v),1e-30))
        lam,w,norm2=lanczos_measure(L,v,steps,zero)
        lam=np.clip(lam,0,2);weight=w*norm2
        exp=np.exp(-taus[:,None]*lam[None,:])
        z0.append(exp@weight);z1.append(exp@(weight*lam));z2.append(exp@(weight*lam**2))
        density_x.extend(lam.tolist());density_w.extend((weight/probes).tolist())
    z0,z1,z2=map(np.asarray,(z0,z1,z2))
    def values(a,b,c):
        Z=1+a;E=b/Z;S=np.log(Z)+taus*E;C=taus**2*(c/Z-E**2)
        return Z,S,np.maximum(C,0)
    Z,S,C=values(z0.mean(0),z1.mean(0),z2.mean(0))
    seZ=z0.std(0,ddof=1)/np.sqrt(probes) if probes>1 else np.zeros(len(taus))
    if probes>1:
        leave=[]
        for i in range(probes):leave.append(values((z0.sum(0)-z0[i])/(probes-1),(z1.sum(0)-z1[i])/(probes-1),(z2.sum(0)-z2[i])/(probes-1))[1:])
        s,c=np.asarray(leave).transpose(1,0,2)
        seS=np.sqrt((probes-1)/probes*np.sum((s-s.mean(0))**2,axis=0))
        seC=np.sqrt((probes-1)/probes*np.sum((c-c.mean(0))**2,axis=0))
    else:seS=seC=np.zeros(len(taus))
    result=pd.DataFrame({'tau':taus,'return_probability':Z/n,'return_low':np.maximum(1/n,(Z-1.96*seZ)/n),
      'return_high':(Z+1.96*seZ)/n,'entropy':S,'entropy_normalized':S/np.log(n),'entropy_se':seS,
      'heat_capacity':C,'heat_se':seC})
    return result,np.asarray(density_x),np.asarray(density_w)/n


def run_spectral(cfg):
    from .diffusion import run_spectral as run
    return run(cfg)
