from __future__ import annotations
import numpy as np
import pandas as pd
from .common import run_dir,shared_dir,domain_config,save_table,write_json,cutoff_date
from .prepare import load_data
from .historical import frozen_communities
from .metrics import local_diversity,edge_distances


def network_data(cfg):
    out=run_dir(cfg)
    nodes=pd.read_parquet(out/'data/network_nodes.parquet')
    edges=pd.read_parquet(out/'data/network_edges.parquet')
    v=np.load(out/'data/network_vectors.npy',mmap_mode='r',allow_pickle=False)
    days=nodes.patent_date.to_numpy(dtype='datetime64[D]').astype('int64')
    if not np.array_equal(nodes.model_pos,np.arange(len(nodes))) or len(v)!=len(nodes):raise ValueError('Network index mismatch.')
    if np.any(days[1:]<days[:-1]) or np.any(days[edges.t.to_numpy()]>=days[edges.s.to_numpy()]):raise ValueError('Chronology mismatch.')
    return nodes,edges,v,days


def build_networks(cfg):
    out=shared_dir(cfg);nodes,v=load_data(cfg)
    refs=pd.read_parquet(out/'data/reference_edges.parquet');inside=pd.read_parquet(out/'data/internal_edges.parquet')
    scopes=[]
    for spec in cfg['domains']:
        dc=domain_config(cfg,spec['id']);dest=run_dir(dc)
        mask=nodes.is_model & nodes.domain.eq(spec['id']).fillna(False)
        selected=nodes.loc[mask].sort_values(['patent_date','patent_id']).reset_index(drop=True).copy()
        dr=refs.loc[refs.source.isin(selected.patent_id)]
        mapping,members,summary=frozen_communities(nodes,dr,cfg)
        members.to_csv(dest/'data/frozen_communities.csv',index=False)
        selected['community']=selected.tech_code.map(mapping).fillna(-1).astype('int32')
        selected['model_pos']=np.arange(len(selected),dtype='int32')
        pos=selected.set_index('patent_id').model_pos
        e=inside.loc[inside.source.isin(pos.index)&inside.target.isin(pos.index)].copy()
        e['s']=e.source.map(pos).astype('int32');e['t']=e.target.map(pos).astype('int32')
        e=e.sort_values(['s','t']).reset_index(drop=True)
        vec=np.asarray(v[selected.vector_row.to_numpy()],dtype='float32')
        array=e[['s','t']].to_numpy(dtype='int32');ki=np.bincount(e.t,minlength=len(selected));ko=np.bincount(e.s,minlength=len(selected))
        selected['in_degree']=ki;selected['out_degree']=ko;selected['degree']=ki+ko
        B,S,bc,sc=local_diversity(array,vec,selected.community.to_numpy(),cfg['min_community_coverage'],cfg['min_semantic_coverage'])
        selected['B']=B;selected['S']=S;selected['community_coverage']=bc;selected['semantic_coverage']=sc
        e['distance']=edge_distances(array,vec)
        e['age_years']=(e.source_date-e.target_date).dt.days/365.25
        mapped=(selected.community.to_numpy()[e.s]>=0)&(selected.community.to_numpy()[e.t]>=0)
        e['cross_community']=np.where(mapped,selected.community.to_numpy()[e.s]!=selected.community.to_numpy()[e.t],np.nan)
        selected.to_parquet(dest/'data/network_nodes.parquet',index=False)
        e.to_parquet(dest/'data/network_edges.parquet',index=False);np.save(dest/'data/network_vectors.npy',vec)
        summary.update({'domain':spec['id'],'label':spec['label'],'nodes':len(selected),'edges':len(e),
                        'mapped_nodes':int(selected.community.ge(0).sum()),'valid_vectors':int(selected.vector_valid.sum()),
                        'mapped_edges':int(mapped.sum()),'last_year':int(selected.grant_year.max())})
        scopes.append(summary)
        write_json(dest/'models/network_scope.json',summary)
    result=pd.DataFrame(scopes);save_table(result,out/'tables/network_scopes')
    return result
