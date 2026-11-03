"""Interpretable heat-return and spectral-entropy profiles with convergence gates."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse,linalg
from .common import run_dir,save_table,write_json
from .graphs import network_data
from .metrics import adjacency
from .spectral import lanczos_measure


def _values(moments,taus,n):
    z=1+moments[:,0].mean(0);s=np.log(z)+taus*moments[:,1].mean(0)/z
    return z/n,s/np.log(n)


def _moments(L,zero,taus,probes,steps,seed):
    rng=np.random.default_rng(seed);n=L.shape[0];out=np.empty((probes,2,len(taus)))
    for i in range(probes):
        v=rng.normal(size=n);v-=zero*(zero@v);v*=np.sqrt((n-1)/(v@v))
        lam,w,norm=lanczos_measure(L,v,steps,zero);lam=np.clip(lam,0,2)
        heat=np.exp(-taus[:,None]*lam);out[i,0]=heat@(w*norm);out[i,1]=heat@(w*norm*lam)
    return out


def diffusion_profile(B,taus,cfg):
    B=sparse.csr_matrix(B,dtype=float);n=len(B.indptr)-1;taus=np.asarray(taus,float)
    if n<2:raise ValueError('Diffusion comparison needs a nontrivial connected component.')
    degree=np.asarray(B.sum(1)).ravel()
    if sparse.csgraph.connected_components(B,directed=False)[0]!=1 or np.any(degree<=0):
        raise ValueError('Use the largest connected undirected component.')
    inv=1/np.sqrt(degree);L=sparse.eye(n,format='csr')-sparse.diags(inv)@B@sparse.diags(inv)
    if n<=cfg.get('spectral_exact_limit',1400):
        lam=np.clip(linalg.eigvalsh(L.toarray(),check_finite=False),0,2);lam[0]=0.
        w=np.exp(-taus[:,None]*lam);Z=w.sum(1);S=np.log(Z)+taus*(w@lam)/Z
        p=Z/n;s=S/np.log(n);low=high=p;se=np.zeros(len(taus));es=ep=np.zeros(len(taus));method='dense spectrum'
    else:
        zero=np.sqrt(degree);zero/=np.linalg.norm(zero)
        r=cfg['slq_probes'];rc=cfg.get('spectral_coarse_probes',max(2,r//2));seed=cfg['seed']+77
        fine=_moments(L,zero,taus,r,cfg['slq_steps'],seed)
        coarse=_moments(L,zero,taus,rc,cfg.get('spectral_coarse_steps',max(8,cfg['slq_steps']//2)),seed)
        p,s=_values(fine,taus,n);pm,sm=_values(fine[:rc],taus,n);pc,sc=_values(coarse,taus,n)
        es=np.maximum(np.abs(sm-sc),np.abs(s-sm))
        ep=np.maximum(np.abs(pm-pc),np.abs(p-pm))/np.maximum(p,1/n)
        sep=fine[:,0].std(0,ddof=1)/np.sqrt(r)/n
        low=np.maximum(1/n,p-1.96*sep);high=np.minimum(1,p+1.96*sep)
        leave=np.array([_values(np.delete(fine,i,axis=0),taus,n)[1] for i in range(r)])
        se=np.sqrt((r-1)/r*np.sum((leave-leave.mean(0))**2,axis=0));method='deflated SLQ'
    passed=(es<=cfg.get('spectral_entropy_tolerance',.025))&(ep<=cfg.get('spectral_return_tolerance',.10))
    supported=np.logical_and.accumulate(passed)
    return pd.DataFrame({'tau':taus,'return_probability':p,'return_low':low,'return_high':high,
      'entropy_normalized':s,'entropy_normalized_se':se,'entropy_step_probe_delta':es,
      'return_step_probe_relative_delta':ep,'supported':supported,'method':method,
      'component_nodes':n,'excess_return':(n*p-1)/(n-1)})


def run_spectral(cfg):
    out=run_dir(cfg);nodes,e,v,days=network_data(cfg);A=adjacency(e[['s','t']].to_numpy(),len(nodes));B=(A+A.T).tocsr()
    _,label=sparse.csgraph.connected_components(B,directed=False);idx=np.flatnonzero(label==np.bincount(label).argmax())
    taus=np.geomspace(cfg['spectral_tau_min'],cfg['spectral_tau_max'],cfg['spectral_tau_points'])
    result=diffusion_profile(B[idx][:,idx],taus,cfg)
    save_table(result,out/'tables/diffusion_profile',latex=False)
    supported=result.loc[result.supported]
    end=float(supported.tau.max()) if len(supported) else None
    write_json(out/'models/diffusion_summary.json',{'full_nodes':len(nodes),'component_nodes':len(idx),
      'method':result.method.iloc[0],'supported_tau_max':end,'tau_max_requested':float(taus[-1]),
      'operator':'symmetric normalized Laplacian on largest undirected component',
      'uncertainty':'probe variability; step/probe changes reported separately'})
    return result
