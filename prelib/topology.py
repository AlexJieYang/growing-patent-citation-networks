from __future__ import annotations
from collections import deque
import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from .common import shared_dir,run_dir,domain_config,save_table,write_json
from .graphs import network_data
from .metrics import adjacency,core_statistics
from .percolation import Components


def annual_snapshots(nodes,e):
    n=len(nodes);array=e[['s','t']].to_numpy(dtype=int);source=array[:,0]
    order=np.argsort(source,kind='stable');ptr=np.r_[0,np.cumsum(np.bincount(source,minlength=n))]
    year=nodes.grant_year.to_numpy(dtype=int);uf=Components(n);rows=[];stop=0;M=0
    for y in range(int(year.min()),int(year.max())+1):
        upper=int(np.searchsorted(year,y,side='right'))
        for node in range(stop,upper):
            uf.activate(node)
            for pos in order[ptr[node]:ptr[node+1]]:uf.union(node,int(array[pos,1]));M+=1
        stop=upper;L,S,chi=uf.observe()
        rows.append({'year':y,'nodes':upper,'edges':M,'mean_degree':2*M/max(1,upper),'giant_fraction':L/max(1,upper),
                     'second_fraction':S/max(1,upper),'susceptibility':chi,'new_nodes':int(np.sum(year==y)),
                     'new_edges':int(np.sum(year[source]==y))})
    return pd.DataFrame(rows)


def make_layout(G,nodes,cfg,out):
    if not len(G):return
    largest=max(nx.connected_components(G),key=len)
    degree=dict(G.degree());seed=max(largest,key=lambda u:(degree[u],-u))
    queue=deque([seed]);seen={seed};chosen=[]
    while queue and len(chosen)<cfg['layout_nodes']:
        u=queue.popleft();chosen.append(u)
        for nb in sorted(G[u],key=lambda x:(-degree[x],x)):
            if nb not in seen:seen.add(nb);queue.append(nb)
    if not chosen:return
    view=G.subgraph(chosen).copy()
    xy=nx.spring_layout(view,seed=cfg['seed'],iterations=cfg['layout_iterations'])
    idx=np.array(sorted(view.nodes()),dtype='int32');loc={u:i for i,u in enumerate(idx)}
    es=np.asarray([(loc[u],loc[v]) for u,v in view.edges()],dtype='int32').reshape(-1,2)
    if len(es)>cfg['layout_edges']:
        rng=np.random.default_rng(cfg['seed']);es=es[rng.choice(len(es),cfg['layout_edges'],replace=False)]
    np.savez_compressed(out/'data/network_view.npz',nodes=idx,xy=np.array([xy[u] for u in idx]),edges=es)


def run_topology(cfg):
    summaries=[]
    for spec in cfg['domains']:
        dc=domain_config(cfg,spec['id']);out=run_dir(dc);nodes,e,v,days=network_data(dc)
        array=e[['s','t']].to_numpy();n=len(nodes);A=adjacency(array,n);B=(A+A.T).tocsr()
        G=nx.from_scipy_sparse_array(B,edge_attribute='weight')
        # No reciprocals in a strict citation DAG, so the projection is simple.
        core=nx.core_number(G);clustering=nx.clustering(G)
        nodes['coreness']=[core[i] for i in range(n)];nodes['clustering']=[clustering[i] for i in range(n)]
        degree=nodes.degree.to_numpy(float)
        nodes['neighbor_degree']=np.divide(B@degree,degree,out=np.full(n,np.nan),where=degree>0)
        nodes.to_parquet(out/'data/node_statistics.parquet',index=False)
        stat=core_statistics(array,v,nodes.community.to_numpy(),cfg['motif_batch_rows'])
        labels=sparse.csgraph.connected_components(B,directed=False)[1];sizes=np.bincount(labels)
        stat.update({'domain':spec['id'],'label':spec['label'],'mean_degree':float(degree.mean()),
          'DAG_density':len(e)/(n*(n-1)/2) if n>1 else 0.,'isolates':int(np.sum(degree==0)),
          'giant_fraction':float(sizes.max(initial=0)/max(1,n)),'components':len(sizes),'max_core':max(core.values(),default=0),
          'mean_clustering':float(nodes.clustering.mean()),
          'mapped_node_fraction':float(nodes.community.ge(0).mean()),'valid_vector_fraction':float(nodes.vector_valid.mean())})
        summaries.append(stat)
        save_table(annual_snapshots(nodes,e),out/'tables/annual_snapshots',latex=False)
        save_table(pd.DataFrame([{'motif':k,'count':stat[k]} for k in ['021D','021U','021C','030T']]),out/'tables/induced_connected_triads',latex=False)
        make_layout(G,nodes,cfg,out)
        write_json(out/'models/topology_summary.json',{k:(None if isinstance(x,float) and not np.isfinite(x) else x) for k,x in stat.items()})
    result=pd.DataFrame(summaries);save_table(result,shared_dir(cfg)/'tables/topology_summary')
    return result
