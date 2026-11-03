from __future__ import annotations
import numpy as np
import pandas as pd
from .common import run_dir,save_table,write_json
from .graphs import network_data
from .metrics import core_statistics,mixing,dag_subgraphs
from .swaps import TemporalSwapChain,permutation_strata,stratified_permutation,empirical_two_sided

METRICS=['cross_edge_share','mean_edge_distance','transitive_closure','path_degeneracy']


def run_null_models(cfg):
    out=run_dir(cfg);nodes,frame,v,days=network_data(cfg);years=nodes.grant_year.to_numpy(int);C=nodes.community.to_numpy(int)
    array=frame[['s','t']].to_numpy(dtype='int32')
    mask=(years[array[:,0]]>=cfg['community_freeze_year'])&((days[array[:,0]]-days[array[:,1]])<=cfg['max_reference_age_years']*365.25)
    moving=array[mask];fixed=array[~mask]
    if not len(moving):raise ValueError('No eligible edges for temporal rewiring.')
    chain=TemporalSwapChain(moving,days,years,cfg['seed']+101,max_age_days=cfg['max_reference_age_years']*365.25,proposal_batch=cfg['null_proposal_batch'])
    burn=chain.step(int(cfg['null_burnin_proposals_per_edge']*len(moving)))
    if burn['proposal_acceptance']<cfg['minimum_null_acceptance']:raise RuntimeError('Null burn-in acceptance too low; experiment stopped.')
    observed=core_statistics(array,v,C,cfg['motif_batch_rows'])
    original=set(map(tuple,moving));strata=permutation_strata(nodes,cfg['community_freeze_year'])
    kinds=cfg.get('null_models',['Topology'])
    rng=np.random.default_rng(cfg['seed']+102);rows=[];trace=[];mix_samples=[]
    for rep in range(cfg['null_repetitions']):
        change=chain.step(int(cfg['null_between_proposals_per_edge']*len(moving)))
        if change['proposal_acceptance']<cfg['minimum_null_acceptance']:raise RuntimeError('Null movement acceptance too low; no selective draw deletion.')
        graph=np.vstack([fixed,chain.edges])
        topology=core_statistics(graph,v,C,cfg['motif_batch_rows'])
        permutation=np.arange(len(nodes))
        if any(k in kinds for k in ['Semantics','Both']):
            permutation=stratified_permutation(nodes,rng,strata=strata)
        vp=v[permutation] if any(k in kinds for k in ['Semantics','Both']) else v
        for kind in kinds:
            if kind=='Topology':stat=topology
            elif kind=='Semantics':stat=core_statistics(array,vp,C,cfg['motif_batch_rows'],
                {k:observed[k] for k in ['030T','021D','021U','021C','two_paths','transitive_closure','path_degeneracy']})
            else:stat=core_statistics(graph,vp,C,cfg['motif_batch_rows'],
                {k:topology[k] for k in ['030T','021D','021U','021C','two_paths','transitive_closure','path_degeneracy']})
            rows.append({'null_model':kind,'repetition':rep,**stat})
        trace.append({'repetition':rep,**change,'edge_overlap':len(chain.edge_set&original)/len(original),
                      'permuted_node_fraction':float(np.mean(permutation!=np.arange(len(nodes))))})
        mix_samples.append(mixing(graph,C)[1])
    samples=pd.DataFrame(rows);summary=[]
    for kind,g in samples.groupby('null_model',sort=False):
        for metric in METRICS:
            values=g[metric].dropna();sd=values.std(ddof=1);obs=observed[metric]
            good=np.isfinite(sd) and sd>1e-14
            summary.append({'null_model':kind,'statistic':metric,'observed':obs,'null_mean':values.mean(),'null_sd':sd,
                'draws':len(values),'standardized_difference':(obs-values.mean())/sd if good else np.nan,
                'empirical_p_two_sided':empirical_two_sided(obs,values),'tail_resolution_per_side':1/(len(values)+1),
                'lag1_autocorrelation':values.autocorr(1) if len(values)>3 and good else np.nan,'degenerate_null':not good})
    result=pd.DataFrame(summary)
    save_table(samples,out/'tables/null_draws',latex=False)
    save_table(result,out/'tables/null_summary',latex=False)
    save_table(pd.DataFrame(trace),out/'tables/null_trace',latex=False)
    obs_mix=mixing(array,C)[1];null_mix=np.mean(mix_samples,axis=0)
    np.savez_compressed(out/'data/mixing_null.npz',observed=obs_mix,null_mean=null_mix,lift=np.divide(obs_mix,null_mix,out=np.full_like(obs_mix,np.nan),where=null_mix>0))
    write_json(out/'models/null_design.json',{'scope':'full internal graph; only post-freeze, age-eligible edges rewired',
       'moving_edges':len(moving),'fixed_edges':len(fixed),'burnin':burn,'draws':cfg['null_repetitions'],
       'clock':'fixed proposals including rejections','semantic_strata':'grant year and original CPC subgroup; valid vectors only',
       'primary_statistics':METRICS,'ensembles':kinds})
    return result
