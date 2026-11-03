"""Three nested chronological growth mechanisms and direct model discrepancies."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from .common import run_dir,cutoff_date,save_table,write_json
from .graphs import network_data
from .kinetics import date_groups
from .metrics import edge_distances,gini,transitive_triangles,ccdf


def physical_models(cfg):
    return {'PA_Age':(0.,0.),'Geometry':(cfg['physical_semantic_cost'],0.),
            'Additive':(cfg['physical_semantic_cost'],cfg['physical_cross_cost'])}


def target_collision(counts):
    """Probability that two distinct newly generated citations share a target."""
    k=np.asarray(counts,dtype=np.float64);m=k.sum()
    return float(np.dot(k,k-1)/(m*(m-1))) if m>1 else np.nan


def simulate_growth(cfg):
    out=run_dir(cfg);nodes,frame,v,days=network_data(cfg);n=len(nodes)
    c=nodes.community.to_numpy();a=frame[['s','t']].to_numpy(dtype='int32');s,t=a.T
    start=int(np.datetime64(cfg['simulation_start_date'],'D').astype('int64'))
    end=int(min(pd.Timestamp(cfg['simulation_end_date']),cutoff_date(cfg)).to_numpy().astype('datetime64[D]').astype('int64'))
    births=np.flatnonzero((days>=start)&(days<=end))
    if not len(births):raise ValueError('No patents in the prespecified growth window.')
    if len(births)>cfg['max_simulation_births']:raise MemoryError('Growth births exceed the configured budget.')
    valid=nodes.vector_valid.to_numpy()&np.isfinite(v).all(1)&(c>=0)
    pool=np.flatnonzero(valid);age=(days[s]-days[t])/365.25
    window=(days[s]>=start)&(days[s]<=end)
    modeled=window&valid[s]&valid[t]&(age<=cfg['max_reference_age_years'])
    observed=a[modeled];m=np.bincount(observed[:,0],minlength=n)
    if not len(observed):raise ValueError('No eligible new citations in the growth window.')
    history=a[days[s]<start];fixed_inside=a[window&~modeled]
    fixed=np.vstack([history,fixed_inside]);last=int(np.searchsorted(days,end,side='right'))
    cache={};pairs=0
    for source in np.flatnonzero(m):
        lo=np.searchsorted(days[pool],days[source]-cfg['max_reference_age_years']*365.25)
        hi=np.searchsorted(days[pool],days[source]);risk=pool[lo:hi].astype('int32')
        pairs+=len(risk)
        if pairs>cfg['max_simulation_pairs']:raise MemoryError('Cached growth candidate pairs exceed the configured budget.')
        if len(risk)<m[source]:raise ValueError('Observed modeled out-degree exceeds the risk universe.')
        distance=np.clip(1-np.asarray(v[risk])@v[source],0,2).astype('float32')
        cache[int(source)]=(risk,distance,(c[risk]!=c[source]),((days[source]-days[risk])/365.25).astype('float32'))
    fixed_order=np.argsort(fixed_inside[:,0],kind='stable')
    fixed_ptr=np.r_[0,np.cumsum(np.bincount(fixed_inside[:,0],minlength=n))]
    baseline_triads=transitive_triangles(fixed,last,cfg['motif_batch_rows'])
    observed_distance=edge_distances(observed,v);curves=[]

    def summarize(new,model,rep):
        distance=edge_distances(new,v);ki=np.bincount(new[:,1],minlength=last)
        extra=transitive_triangles(np.vstack([fixed,new]),last,cfg['motif_batch_rows'])-baseline_triads
        stats={'model':model,'repetition':rep,'scope':'new_edges','nodes':last,'edges':len(new),
          'cross_edge_share':float(np.mean(c[new[:,0]]!=c[new[:,1]])),
          'mean_edge_distance':float(distance.mean()),'indegree_variance':float(ki.var()),
          'indegree_gini':gini(ki),'target_collision':target_collision(ki),
          'effective_targets':float(1/np.sum((ki/len(new))**2)),
          'new_transitive_triads':int(round(extra)),'triads_per_new_edge':float(extra/len(new)),
          'distance_W1':float(wasserstein_distance(distance,observed_distance))}
        x=np.unique(np.r_[np.arange(1,min(20,len(new)+1)),
                        np.rint(np.geomspace(min(20,len(new)),len(new),64)).astype(int)])
        tail=np.cumsum(np.bincount(ki,minlength=len(new)+1)[::-1])[::-1]
        p=tail[x]/max(1,np.sum(ki>0))
        curves.extend({'model':model,'repetition':rep,'metric':'positive_new_indegree','x':u,'probability':q} for u,q in zip(x,p))
        bins=np.linspace(0,2,101);count=np.histogram(distance,bins)[0];p=np.cumsum(count)/len(new)
        curves.extend({'model':model,'repetition':rep,'metric':'distance','x':u,'probability':q} for u,q in zip(bins[1:],p))
        return stats

    rows=[summarize(observed,'Observed',-1)]
    # Independent, reproducible streams indexed by mechanism and realization.
    streams=np.random.SeedSequence(cfg['seed']+620).spawn(len(physical_models(cfg))*cfg['simulation_repetitions'])
    stream=0
    for model,(beta,kappa) in physical_models(cfg).items():
        for rep in range(cfg['simulation_repetitions']):
            rng=np.random.default_rng(streams[stream]);stream+=1
            degree=np.bincount(history[:,1],minlength=n).astype('int64');new=[]
            for left,right in zip(*date_groups(days[births])):
                pending=[]
                for source in births[left:right]:
                    if m[source]:
                        risk,d,cross,age=cache[int(source)]
                        logw=cfg['physical_alpha']*np.log1p(degree[risk])-age/cfg['physical_age_tau_years']-beta*d-kappa*cross
                        weight=np.exp(np.maximum(logw-logw.max(),-700));weight/=weight.sum()
                        targets=rng.choice(risk,size=int(m[source]),replace=False,p=weight)
                        new.extend((int(source),int(target)) for target in targets);pending.extend(targets)
                    current=fixed_inside[fixed_order[fixed_ptr[source]:fixed_ptr[source+1]]]
                    if len(current):pending.extend(current[:,1])
                np.add.at(degree,np.asarray(pending,dtype=int),1)
            new=np.asarray(new,dtype='int32').reshape(-1,2)
            if not np.array_equal(np.bincount(new[:,0],minlength=n),m) or np.any(days[new[:,1]]>=days[new[:,0]]):
                raise ValueError('Growth constraints failed.')
            rows.append(summarize(new,model,rep))
            if rep==0:pd.DataFrame(np.vstack([fixed,new]),columns=['source_pos','target_pos']).to_parquet(out/f'models/simulation_{model}_example.parquet',index=False)
    result=pd.DataFrame(rows);save_table(result,out/'tables/growth_simulations',latex=False)
    save_table(pd.DataFrame(curves),out/'tables/growth_edge_curves',latex=False)
    summary=[]
    for model,g in result.groupby('model',sort=False):
        for metric in ['cross_edge_share','mean_edge_distance','target_collision','triads_per_new_edge','distance_W1']:
            q=g[metric].dropna();summary.append({'model':model,'statistic':metric,'mean':q.mean(),
                'low':q.quantile(.025),'high':q.quantile(.975),'realizations':len(q)})
    save_table(pd.DataFrame(summary),out/'tables/growth_summary',latex=False)
    write_json(out/'models/growth_experiment.json',{'start':str(pd.Timestamp(start,unit='D').date()),
      'end':str(pd.Timestamp(end,unit='D').date()),'modeled_new_edges':len(observed),
      'modeled_sources':int((m>0).sum()),'fixed_edges':len(fixed),'candidate_pairs':pairs,
      'models':{name:{'beta':b,'kappa':k} for name,(b,k) in physical_models(cfg).items()},
      'alpha':cfg['physical_alpha'],'age_tau_years':cfg['physical_age_tau_years'],
      'repetitions':cfg['simulation_repetitions'],'uncertainty':'variation among simulated realizations',
      'triad_definition':'T(fixed + modeled new edges) - T(fixed); all comparisons share the same fixed graph'})
    return result
