"""Sparse DAG subgraphs, mixing information and descriptive network observables."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


def adjacency(edges,n,dtype='float64'):
    e=np.asarray(edges,dtype='int64').reshape(-1,2)
    return sparse.csr_matrix((np.ones(len(e),dtype=dtype),(e[:,0],e[:,1])),shape=(n,n))


def safe_corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.std(x[ok])<1e-14 or np.std(y[ok])<1e-14:return np.nan
    return float(np.corrcoef(x[ok],y[ok])[0,1])


def gini(x):
    x=np.sort(np.asarray(x,float));x=x[np.isfinite(x)&(x>=0)]
    return float((2*np.dot(np.arange(1,len(x)+1),x)/(len(x)*x.sum())-(len(x)+1)/len(x))) if len(x) and x.sum()>0 else 0.


def distribution_entropy(counts):
    x=np.asarray(counts,float);x=x[x>0]
    p=x/x.sum() if x.sum() else x
    return float(-np.sum(p*np.log(p)))


def ccdf(x,positive=False):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if positive:x=x[x>0]
    values,count=np.unique(x,return_counts=True)
    return values,np.cumsum(count[::-1])[::-1]/max(1,len(x))


def cdf(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    values,count=np.unique(x,return_counts=True)
    return values,np.cumsum(count)/max(1,len(x))


def edge_distances(e,vectors,batch=4096):
    e=np.asarray(e,dtype=int).reshape(-1,2);result=np.full(len(e),np.nan)
    for a in range(0,len(e),batch):
        b=e[a:a+batch];u=np.asarray(vectors[b[:,0]],float);v=np.asarray(vectors[b[:,1]],float)
        good=np.isfinite(u).all(axis=1)&np.isfinite(v).all(axis=1)
        result[a+np.flatnonzero(good)]=np.clip(1-np.einsum('ij,ij->i',u[good],v[good]),0,2)
    return result


def local_diversity(e,vectors,communities,min_coverage=.8,min_semantic_coverage=None):
    n=len(vectors);e=np.asarray(e,dtype=int).reshape(-1,2)
    count=np.bincount(e[:,0],minlength=n)
    c=np.asarray(communities,dtype=int);K=max(1,int(c.max(initial=-1)+1))
    known=c[e[:,1]]>=0;valid=np.isfinite(vectors).all(axis=1)
    valid_targets=valid[e[:,1]]
    mapped=np.bincount(e[known,0],minlength=n)
    C=np.bincount(e[known,0]*K+c[e[known,1]],minlength=n*K).reshape(n,K)
    cov=np.divide(mapped,count,out=np.zeros(n),where=count>0)
    B=np.full(n,np.nan);good=(mapped>=2)&(cov>=min_coverage)
    B[good]=1-(C[good].astype(float)**2).sum(axis=1)/mapped[good]**2
    A=adjacency(e[valid_targets],n,dtype='float32')
    V=np.asarray(vectors) if valid.all() else np.nan_to_num(np.asarray(vectors),nan=0.,posinf=0.,neginf=0.)
    nv=np.bincount(e[valid_targets,0],minlength=n);active=np.flatnonzero(nv)
    sums=np.asarray(A[active]@V,dtype='float64');norm2=np.zeros(n)
    norm2[active]=np.einsum('ij,ij->i',sums,sums)
    vc=np.divide(nv,count,out=np.zeros(n),where=count>0)
    S=np.full(n,np.nan);semantic_min=min_coverage if min_semantic_coverage is None else min_semantic_coverage
    good=(nv>=2)&(vc>=semantic_min)
    S[good]=np.clip((nv[good]**2-norm2[good])/(nv[good]*(nv[good]-1)),0,2)
    return B,S,cov,vc


def mixing(e,c):
    e=np.asarray(e,dtype=int).reshape(-1,2);c=np.asarray(c,dtype=int)
    K=max(1,int(c.max(initial=-1)+1));good=(c[e[:,0]]>=0)&(c[e[:,1]]>=0)
    counts=np.bincount(c[e[good,0]]*K+c[e[good,1]],minlength=K*K).reshape(K,K)
    p=counts/max(1,counts.sum());row=p.sum(axis=1);col=p.sum(axis=0)
    expected=row[:,None]*col[None,:];mask=p>0
    mi=float(np.sum(p[mask]*np.log(p[mask]/expected[mask]))) if mask.any() else np.nan
    denom=distribution_entropy(row)+distribution_entropy(col)
    nmi=2*mi/denom if denom>0 else np.nan
    cross=float(np.mean(c[e[good,0]]!=c[e[good,1]])) if good.any() else np.nan
    return counts,p,mi,nmi,cross,good


def partition_nmi(a,b):
    a=np.asarray(a);b=np.asarray(b);good=(a>=0)&(b>=0)
    if not good.any():return np.nan
    _,i=np.unique(a[good],return_inverse=True);_,j=np.unique(b[good],return_inverse=True)
    p=np.bincount(i*(j.max()+1)+j,minlength=(i.max()+1)*(j.max()+1)).reshape(i.max()+1,j.max()+1).astype(float)
    p/=p.sum();r=p.sum(1);c=p.sum(0);q=r[:,None]*c[None,:];ok=p>0
    I=np.sum(p[ok]*np.log(p[ok]/q[ok]));h=distribution_entropy(r)+distribution_entropy(c)
    return float(2*I/h) if h>0 else 1.


def dag_subgraphs(e,n,batch=256):
    A=adjacency(e,n);kin=np.asarray(A.sum(axis=0)).ravel();kout=np.asarray(A.sum(axis=1)).ravel()
    transitive=0.;degeneracy=0.
    for a in range(0,n,batch):
        W=A[a:a+batch]@A
        transitive+=float(W.multiply(A[a:a+batch]).sum())
        degeneracy+=float(np.sum(W.data*(W.data-1)/2))
    chain=float(np.dot(kin,kout));outfork=float(np.sum(kout*(kout-1)/2));infork=float(np.sum(kin*(kin-1)/2))
    return {'030T':transitive,'021D':max(0.,outfork-transitive),'021U':max(0.,infork-transitive),
            '021C':max(0.,chain-transitive),'two_paths':chain,
            'transitive_closure':transitive/chain if chain>0 else np.nan,'path_degeneracy':degeneracy}


def transitive_triangles(e,n,batch=256):
    """Count only transitive triples; avoid unused path-multiplicity work."""
    A=adjacency(e,n);total=0.
    for a in range(0,n,batch):total+=float((A[a:a+batch]@A).multiply(A[a:a+batch]).sum())
    return total


def core_statistics(e,vectors,communities,batch=256,subgraphs_cache=None):
    """Primary observables, without dense reference-diversity calculations."""
    e=np.asarray(e,dtype='int32').reshape(-1,2);n=len(vectors);c=np.asarray(communities)
    ko=np.bincount(e[:,0],minlength=n);ki=np.bincount(e[:,1],minlength=n)
    d=edge_distances(e,vectors);known=(c[e[:,0]]>=0)&(c[e[:,1]]>=0);valid=np.isfinite(d)
    cross=known&(c[e[:,0]]!=c[e[:,1]]);within=known&~cross
    mean=lambda mask:float(d[mask].mean()) if mask.any() else np.nan
    wc=mean(within&valid);cc=mean(cross&valid)
    sub=dag_subgraphs(e,n,batch) if subgraphs_cache is None else subgraphs_cache
    return {'nodes':n,'edges':len(e),'max_indegree':int(ki.max(initial=0)),
      'indegree_gini':gini(ki),'indegree_variance':float(ki.var()),
      'cross_edge_share':float(cross.sum()/known.sum()) if known.any() else np.nan,
      'mean_edge_distance':mean(valid),'mean_edge_distance_mapped':mean(valid&known),
      'within_mean_distance':wc,'cross_mean_distance':cc,'cross_minus_within_distance':cc-wc,
      'mapped_edges':int(known.sum()),'valid_distance_edges':int(valid.sum()),
      'mapped_edge_fraction':float(known.mean()) if len(e) else np.nan,**sub}


def rich_club(e,degree,thresholds):
    e=np.asarray(e,dtype=int).reshape(-1,2);rows=[]
    for k in thresholds:
        hit=np.asarray(degree)>k;nr=int(hit.sum());m=int(np.sum(hit[e[:,0]]&hit[e[:,1]]))
        rows.append({'threshold':float(k),'club_nodes':nr,'club_edges':m,'phi':2*m/(nr*(nr-1)) if nr>=2 else np.nan})
    return pd.DataFrame(rows)


def network_statistics(e,vectors,communities,years,min_coverage=.8,motif_batch=256,subgraphs_cache=None,min_semantic_coverage=None):
    e=np.asarray(e,dtype='int32').reshape(-1,2);n=len(vectors)
    ko=np.bincount(e[:,0],minlength=n);ki=np.bincount(e[:,1],minlength=n);degree=ko+ki
    d=edge_distances(e,vectors);_,p,mi,nmi,cross,known=mixing(e,communities)
    cross_mask=np.zeros(len(e),bool);cross_mask[known]=communities[e[known,0]]!=communities[e[known,1]]
    geom=np.isfinite(d)&known
    gap=float(d[geom&cross_mask].mean()-d[geom&~cross_mask].mean()) if (geom&cross_mask).any() and (geom&~cross_mask).any() else np.nan
    B,S,bc,sc=local_diversity(e,vectors,communities,min_coverage,min_semantic_coverage)
    good=np.isfinite(B)&np.isfinite(S)&(ko>=2);corr=np.nan
    if good.sum()>=10:
        f=pd.DataFrame({'B':B[good],'S':S[good],'year':np.asarray(years)[good],
                        'count_bin':np.floor(np.log2(ko[good])).astype(int)})
        centered=f[['B','S']]-f.groupby(['year','count_bin'])[['B','S']].transform('mean')
        corr=safe_corr(centered.B,centered.S)
    sub=dag_subgraphs(e,n,motif_batch) if subgraphs_cache is None else {k:subgraphs_cache[k] for k in ['030T','021D','021U','021C','two_paths','transitive_closure','path_degeneracy']}
    valid_degree=degree.sum();internal=np.trace(p)
    # Undirected modularity of the frozen partition on mapped edges only.
    if known.any():
        K=p.shape[0];stub=np.bincount(np.r_[communities[e[known,0]],communities[e[known,1]]],minlength=K)
        Q=float(internal-np.sum((stub/(2*known.sum()))**2))
    else:Q=np.nan
    return {'nodes':n,'edges':len(e),'max_indegree':int(ki.max(initial=0)),'indegree_gini':gini(ki),
      'indegree_variance':float(ki.var()),'cross_edge_share':cross,'mean_edge_distance':float(np.nanmean(d)) if np.isfinite(d).any() else np.nan,
      'cross_minus_within_distance':gap,'ref_B_S_centered_correlation':corr,
      'mixing_mutual_information':mi,'mixing_NMI':nmi,'frozen_modularity':Q,
      'out_in_assortativity':safe_corr(ko[e[:,0]],ki[e[:,1]]) if len(e) else np.nan,
      'sources_with_two_references':int((ko>=2).sum()),'mapped_edge_fraction':float(known.mean()) if len(e) else np.nan,
      **sub}
