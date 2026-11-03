"""Reverse-insertion site failure and incremental bond percolation."""
from __future__ import annotations
import heapq
import numpy as np
import pandas as pd
from scipy import sparse
from .common import run_dir,save_table
from .graphs import network_data
from .metrics import adjacency


class Components:
    def __init__(self,n):
        self.parent=np.arange(n);self.size=np.zeros(n,dtype='int64');self.active=np.zeros(n,bool)
        self.heap=[];self.count=0;self.sum_sq=0
    def find(self,a):
        while self.parent[a]!=a:
            self.parent[a]=self.parent[self.parent[a]];a=int(self.parent[a])
        return a
    def activate(self,a):
        if self.active[a]:return
        self.active[a]=True;self.parent[a]=a;self.size[a]=1;self.count+=1;self.sum_sq+=1
        heapq.heappush(self.heap,(-1,int(a)))
    def union(self,a,b):
        if not self.active[a] or not self.active[b]:return
        a=self.find(int(a));b=self.find(int(b))
        if a==b:return
        if self.size[a]<self.size[b]:a,b=b,a
        x,y=int(self.size[a]),int(self.size[b]);self.sum_sq+=2*x*y
        self.parent[b]=a;self.size[a]=x+y;heapq.heappush(self.heap,(-int(x+y),int(a)))
    def _clean(self):
        while self.heap:
            negative,a=self.heap[0]
            if self.parent[a]==a and self.active[a] and self.size[a]==-negative:break
            heapq.heappop(self.heap)
    def observe(self):
        self._clean()
        if not self.heap:return 0,0,0.
        first=heapq.heappop(self.heap);largest=-first[0];self._clean()
        second=-self.heap[0][0] if self.heap else 0;heapq.heappush(self.heap,first)
        rest=self.count-largest;chi=(self.sum_sq-largest*largest)/rest if rest else 0.
        return int(largest),int(second),float(chi)


def site_curve(edges,n,removal_order,grid):
    e=np.asarray(edges,dtype=int).reshape(-1,2);B=adjacency(e,n);B=(B+B.T).tocsr()
    order=np.asarray(removal_order,int)
    if len(order)!=n or len(np.unique(order))!=n:raise ValueError('Invalid node removal order.')
    wanted={}
    for q in grid:wanted.setdefault(n-int(round(float(q)*n)),[]).append(float(q))
    uf=Components(n);rows=[]
    def record():
        if uf.count not in wanted:return
        L,S,chi=uf.observe()
        for q in wanted[uf.count]:rows.append({'fraction':q,'active_nodes':uf.count,'largest':L,'second':S,
             'giant_fraction':L/max(1,n),'second_fraction':S/max(1,n),'survivor_giant_fraction':L/max(1,uf.count),'susceptibility':chi})
    record()
    for node in order[::-1]:
        uf.activate(node)
        for nb in B.indices[B.indptr[node]:B.indptr[node+1]]:
            if uf.active[nb]:uf.union(node,int(nb))
        record()
    return pd.DataFrame(rows).sort_values('fraction').reset_index(drop=True)


def bond_curve(edges,n,edge_order,grid):
    e=np.asarray(edges,dtype=int).reshape(-1,2);order=np.asarray(edge_order,int);m=len(e)
    if len(order)!=m or len(np.unique(order))!=m:raise ValueError('Invalid edge insertion order.')
    wanted={}
    for p in grid:wanted.setdefault(int(round(float(p)*m)),[]).append(float(p))
    uf=Components(n)
    for node in range(n):uf.activate(node)
    rows=[]
    def record(k):
        if k not in wanted:return
        L,S,chi=uf.observe()
        for p in wanted[k]:rows.append({'fraction':p,'retained_edges':k,'largest':L,'second':S,
          'giant_fraction':L/max(1,n),'second_fraction':S/max(1,n),'susceptibility':chi})
    record(0)
    for k,j in enumerate(order,1):uf.union(*map(int,e[j]));record(k)
    return pd.DataFrame(rows).sort_values('fraction').reset_index(drop=True)


def run_percolation(cfg):
    out=run_dir(cfg);nodes,frame,vec,days=network_data(cfg);n=len(nodes);e=frame[['s','t']].to_numpy()
    degree=nodes.degree.to_numpy();c=nodes.community.to_numpy();known=(c[e[:,0]]>=0)&(c[e[:,1]]>=0)
    cross=known&(c[e[:,0]]!=c[e[:,1]])
    bridge=np.bincount(e[cross].ravel(),minlength=n)
    rng=np.random.default_rng(cfg['seed']+411);grid=np.linspace(0,1,cfg['percolation_points']);rows=[]
    for strategy in ['Random','Degree']:
        repeats=cfg['robustness_repetitions'] if strategy in ['Random','Degree'] else 1
        for rep in range(repeats):
            tie=rng.random(n)
            order=rng.permutation(n) if strategy=='Random' else np.lexsort((tie,-(degree if strategy=='Degree' else bridge)))
            q=site_curve(e,n,order,grid);q['family']='site';q['strategy']=strategy;q['repetition']=rep;rows.append(q)
    # Bond comparisons use the same finite-distance edge universe in all arms.
    ok=np.isfinite(frame.distance.to_numpy());be=e[ok];distance=frame.distance.to_numpy()[ok]
    for strategy in ['Random','Similar first','Distant first']:
        repeats=cfg['robustness_repetitions'] if strategy in ['Random','Degree'] else 1
        for rep in range(repeats):
            tie=rng.random(len(be))
            order=rng.permutation(len(be)) if strategy=='Random' else np.lexsort((tie,distance if strategy=='Similar first' else -distance))
            q=bond_curve(be,n,order,grid);q['family']='bond';q['strategy']=strategy;q['repetition']=rep;rows.append(q)
    result=pd.concat(rows,ignore_index=True)
    for family,g in result.groupby('family'):
        for strategy,h in g.groupby('strategy'):
            for rep,k in h.groupby('repetition'):
                k=k.sort_values('fraction');base=k.giant_fraction.iloc[0 if family=='site' else -1]
                result.loc[k.index,'relative_giant']=k.giant_fraction/max(base,1e-30)
    summary=result.groupby(['family','strategy','fraction'],sort=False).agg(
        giant_mean=('giant_fraction','mean'),giant_low=('giant_fraction',lambda x:x.quantile(.025)),giant_high=('giant_fraction',lambda x:x.quantile(.975)),
        relative_mean=('relative_giant','mean'),relative_low=('relative_giant',lambda x:x.quantile(.025)),relative_high=('relative_giant',lambda x:x.quantile(.975)),
        second_mean=('second_fraction','mean')).reset_index()
    save_table(result.drop(columns='susceptibility'),out/'tables/percolation_curves',latex=False)
    save_table(summary,out/'tables/percolation_summary',latex=False)
    stats=[]
    for (family,strategy,rep),g in result.groupby(['family','strategy','repetition']):
        g=g.sort_values('fraction');x=g.fraction.to_numpy();y=g.relative_giant.to_numpy()
        xp=y[::-1] if family=='site' else y;fp=x[::-1] if family=='site' else x
        # Interpolate only inside a crossed grid bracket; ties use their first occurrence.
        hit=np.flatnonzero(y<=.5) if family=='site' else np.flatnonzero(y>=.5)
        at=np.nan
        if len(hit):
            j=hit[0]
            at=float(x[j]) if j==0 or y[j]==y[j-1] else float(x[j-1]+(.5-y[j-1])*(x[j]-x[j-1])/(y[j]-y[j-1]))
        area=float(np.sum(np.diff(x)*(y[:-1]+y[1:])/2))
        stats.append({'family':family,'strategy':strategy,'repetition':rep,'half_fraction':at,'relative_AUC':area})
    save_table(pd.DataFrame(stats),out/'tables/connectivity_summary',latex=False)
    return summary
