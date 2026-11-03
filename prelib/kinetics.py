"""Risk-set opportunity enrichment and censored first-passage statistics; no fitting."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .common import run_dir,shared_dir,cutoff_date,save_table,write_json,domain_config
from .graphs import network_data
from .prepare import load_data

FEATURES=['log_degree_plus_one','age_decades','semantic_distance','cross_community','cross_x_distance']


def date_groups(days):
    if not len(days):return np.array([],int),np.array([],int)
    start=np.r_[0,np.flatnonzero(days[1:]!=days[:-1])+1]
    return start,np.r_[start[1:],len(days)]


def build_opportunities(cfg):
    out=run_dir(cfg);nodes,edges,v,days=network_data(cfg)
    n=len(nodes);s=edges.s.to_numpy();t=edges.t.to_numpy();year=nodes.grant_year.to_numpy(int);c=nodes.community.to_numpy(int)
    valid=nodes.vector_valid.to_numpy()&np.isfinite(v).all(1);known=valid&(c>=0)
    age=(days[s]-days[t])/365.25
    eligible=(year[s]>=cfg['growth_start_year'])&(age<=cfg['max_reference_age_years'])
    split=np.where(year[s]<=cfg['train_end_year'],0,np.where(year[s]<=cfg['validation_end_year'],1,2))
    names=['train','validation','test'];K=cfg['negative_controls'];rng=np.random.default_rng(cfg['seed']+91)
    cache={};selected={};summary=[]
    for mode in ['full','known']:
        selected[mode]=np.zeros(len(edges),bool)
        for i,name in enumerate(names):
            available=np.flatnonzero(eligible&(split==i)&(True if mode=='full' else known[s]&known[t]))
            amount=min(len(available),cfg['max_choice_events_per_split']);pick=rng.choice(available,amount,replace=False)
            selected[mode][pick]=True
            cache[mode,name]={'X':np.empty((amount,K+1,5),dtype='float32'),'alternatives':np.empty((amount,K+1),dtype='int32'),
                 'source':np.empty(amount,dtype='int32'),'target':np.empty(amount,dtype='int32'),
                 'risk_size':np.empty(amount,dtype='int32'),'year':np.empty(amount,dtype='int16'),'count':0,'available':len(available)}
    order=np.argsort(s,kind='stable');ptr=np.r_[0,np.cumsum(np.bincount(s,minlength=n))];degree=np.zeros(n,dtype='int64')
    positions={'full':np.arange(n),'known':np.flatnonzero(known)}
    rank={m:np.full(n,-1,dtype=int) for m in positions}
    for m,pos in positions.items():rank[m][pos]=np.arange(len(pos))
    for a,b in zip(*date_groups(days)):
        day=days[a]
        for mode,pool in positions.items():
            lo=np.searchsorted(days[pool],day-cfg['max_reference_age_years']*365.25);hi=np.searchsorted(days[pool],day)
            for source in range(a,b):
                for edge in order[ptr[source]:ptr[source+1]][selected[mode][order[ptr[source]:ptr[source+1]]]]:
                    target=int(t[edge]);r=rank[mode][target]
                    if hi-lo<K+1 or not lo<=r<hi:continue
                    offset=rng.choice(hi-lo-1,K,replace=False)
                    alternatives=np.r_[target,pool[lo+offset+(offset>=r-lo)]]
                    d=np.full(K+1,np.nan)
                    if valid[source]:
                        ok=valid[alternatives]
                        d[ok]=np.clip(1-np.asarray(v[alternatives[ok]])@v[source],0,2)
                    cross=np.where((c[alternatives]>=0)&(c[source]>=0),c[alternatives]!=c[source],np.nan)
                    x=np.column_stack([np.log1p(degree[alternatives]),(day-days[alternatives])/3652.5,d,cross,cross*d])
                    rec=cache[mode,names[split[edge]]];j=rec['count']
                    rec['X'][j]=x;rec['alternatives'][j]=alternatives
                    for key,value in [('source',source),('target',target),('risk_size',hi-lo),('year',year[source])]:rec[key][j]=value
                    rec['count']+=1
        np.add.at(degree,t[order[ptr[a]:ptr[b]]],1)
    for (mode,name),rec in cache.items():
        count=rec.pop('count');available=rec.pop('available')
        arrays={k:x[:count] for k,x in rec.items()}
        if mode=='known' and not np.isfinite(arrays['X']).all():raise ValueError('Known opportunity universe is not finite.')
        np.savez_compressed(out/f'data/opportunities_{mode}_{name}.npz',**arrays)
        summary.append({'mode':mode,'split':name,'available_events':available,'sampled_events':count,
                        'citing_patents':len(np.unique(arrays['source'])),'controls':K})
    result=pd.DataFrame(summary);save_table(result,out/'tables/opportunity_samples',latex=False)
    return result


def opportunity_estimates(bins,risk,source,nbins,repetitions=199,seed=1):
    """Chosen plus uniformly sampled nonchosen alternatives give an HT risk sum."""
    bins=np.asarray(bins,int);risk=np.asarray(risk,float);source=np.asarray(source)
    E,K1=bins.shape;K=K1-1
    if not E:return pd.DataFrame({'bin':np.arange(nbins),'observed':0.,'expected_uniform':0.,'enrichment':np.nan,'low':np.nan,'high':np.nan})
    weight=np.repeat(((risk-1)/(K*risk))[:,None],K1,axis=1);weight[:,0]=1/risk
    ok=(bins>=0)&(bins<nbins)
    event=np.repeat(np.arange(E),K1).reshape(E,K1)
    expected=np.bincount((event*nbins+np.maximum(bins,0))[ok],weights=weight[ok],minlength=E*nbins).reshape(E,nbins)
    observed=np.zeros((E,nbins));good=(bins[:,0]>=0)&(bins[:,0]<nbins)
    observed[np.flatnonzero(good),bins[good,0]]=1
    ids,inverse=np.unique(source,return_inverse=True);gO=np.zeros((len(ids),nbins));gE=np.zeros_like(gO)
    np.add.at(gO,inverse,observed);np.add.at(gE,inverse,expected)
    O=observed.sum(0);Q=expected.sum(0);ratio=np.divide(O,Q,out=np.full(nbins,np.nan),where=Q>0)
    boot=[];rng=np.random.default_rng(seed)
    for _ in range(repetitions):
        mult=np.bincount(rng.integers(0,len(ids),size=len(ids)),minlength=len(ids))
        a=mult@gO;b=mult@gE;boot.append(np.divide(a,b,out=np.full(nbins,np.nan),where=b>0))
    boot=np.asarray(boot).reshape(-1,nbins);low=np.full(nbins,np.nan);high=low.copy()
    for j in range(nbins):
        values=boot[:,j];values=values[np.isfinite(values)]
        if len(values):low[j],high[j]=np.quantile(values,[.025,.975])
    insufficient=(Q<5)|(O<3)|(len(ids)<5)
    ratio[insufficient]=np.nan;low[insufficient]=np.nan;high[insufficient]=np.nan
    return pd.DataFrame({'bin':np.arange(nbins),'observed':O,'expected_uniform':Q,'enrichment':ratio,'low':low,'high':high})


def run_enrichment(cfg):
    out=run_dir(cfg);rows=[]
    specs=[('degree','full',np.array([-.5,.5,1.5,2.5,4.5,8.5,16.5,32.5,64.5,np.inf])),
           ('age','full',np.array([0,.25,.5,1,2,5,10,20,30.001])),
           ('distance','known',np.array([0,.1,.2,.3,.4,.5,.6,.8,1.,1.4,2.0001]))]
    for split in ['train','validation','test']:
        for feature,mode,edges in specs:
            d=dict(np.load(out/f'data/opportunities_{mode}_{split}.npz'));X=d['X']
            raw=np.rint(np.expm1(X[:,:,0])) if feature=='degree' else X[:,:,1]*10 if feature=='age' else X[:,:,2]
            bins=np.searchsorted(edges,raw,side='right')-1
            bins[~np.isfinite(raw)]=-1;nb=len(edges)-1
            if feature=='distance':bins=bins+nb*np.nan_to_num(X[:,:,3],nan=0).astype(int);nb*=2
            q=opportunity_estimates(bins,d['risk_size'],d['source'],nb,cfg['bootstrap_repetitions'],cfg['seed']+27)
            for j,row in q.iterrows():
                k=int(row['bin'])%(len(edges)-1);left,right=edges[k:k+2]
                center=(left+right)/2 if np.isfinite(right) else max(1,left*1.5)
                rows.append({'feature':feature,'mode':mode,'split':split,'center':center,'left':left,'right':right,
                      'cross_community':int(row['bin'])//(len(edges)-1) if feature=='distance' else -1,**row.to_dict()})
    result=pd.DataFrame(rows);save_table(result,out/'tables/opportunity_enrichment',latex=False)
    surfaces=[];degree_edges=np.array([-.5,.5,1.5,3.5,7.5,15.5,31.5,np.inf]);age_edges=np.array([0,1,2,5,10,20,30.001])
    for split in ['train','validation','test']:
        d=dict(np.load(out/f'data/opportunities_full_{split}.npz'));X=d['X']
        k=np.searchsorted(degree_edges,np.rint(np.expm1(X[:,:,0])),side='right')-1
        a=np.searchsorted(age_edges,10*X[:,:,1],side='right')-1
        nk=len(degree_edges)-1;na=len(age_edges)-1
        q=opportunity_estimates(a*nk+k,d['risk_size'],d['source'],nk*na,0,cfg['seed'])
        q['degree_bin']=q['bin']%nk;q['age_bin']=q['bin']//nk;q['split']=split;surfaces.append(q)
    save_table(pd.concat(surfaces,ignore_index=True),out/'tables/degree_age_enrichment',latex=False)
    return result


def kaplan_meier(duration,event):
    duration=np.asarray(duration,float);event=np.asarray(event,int)
    ok=np.isfinite(duration)&(duration>=0);duration=duration[ok];event=event[ok]
    survival=1.;greenwood=0.;rows=[{'days':0.,'survival':1.,'low':1.,'high':1.,'at_risk':len(duration),'events':0}]
    for t in np.unique(duration):
        risk=int(np.sum(duration>=t));deaths=int(np.sum((duration==t)&(event==1)))
        if deaths:
            survival*=1-deaths/risk
            if risk>deaths:greenwood+=deaths/(risk*(risk-deaths))
        if survival==0:lo=hi=0.
        elif survival==1:lo=hi=1.
        else:
            s=np.sqrt(greenwood)/abs(np.log(survival));z=np.log(-np.log(survival))
            lo=np.exp(-np.exp(z+1.96*s));hi=np.exp(-np.exp(z-1.96*s))
        rows.append({'days':t,'survival':survival,'low':lo,'high':hi,'at_risk':risk,'events':deaths})
    return pd.DataFrame(rows)


def run_cohort_statistics(cfg):
    out=shared_dir(cfg);global_nodes,v=load_data(cfg);fwd=pd.read_parquet(out/'data/forward_edges.parquet')
    index=global_nodes.set_index('patent_id');T=cutoff_date(cfg);rows=[];kmrows=[];patients=[]
    for spec in cfg['domains']:
        nodes,e,vec,days=network_data(domain_config(cfg,spec['id']))
        focal=nodes.loc[nodes.is_focal].copy();ids=pd.Index(focal.patent_id);grants=focal.set_index('patent_id').patent_date
        ends=grants+pd.DateOffset(years=cfg['primary_outcome_years']);censor=grants+pd.DateOffset(years=cfg['outcome_years']);censor=censor.clip(upper=T)
        f=fwd.loc[fwd.target.isin(ids)].copy()
        f=f.loc[f.source_date.le(f.target.map(censor))]
        within=f.loc[f.source_date.le(f.target.map(ends))].groupby('target').size()
        focal['complete_window']=focal.patent_date.add(pd.DateOffset(years=cfg['primary_outcome_years'])).le(T)
        focal['citations_fixed_window']=focal.patent_id.map(within).fillna(0).where(focal.complete_window)
        for year,g in focal.groupby('grant_year'):
            x=g.citations_fixed_window.dropna()
            rows.append({'domain':spec['id'],'year':int(year),'patents':len(g),'mature':int(g.complete_window.sum()),
                'mean_fixed_window_citations':x.mean(),'median_fixed_window_citations':x.median(),
                'q25':x.quantile(.25),'q75':x.quantile(.75),'B_available':int(g.B.notna().sum()),'S_available':int(g.S.notna().sum())})
        f['source_field']=f.source.map(index.cpc_subclass);f['target_field']=f.target.map(index.cpc_subclass)
        known=f.source_field.notna()&f.target_field.notna()
        first=f.loc[known&f.source_field.ne(f.target_field)].groupby('target').source_date.min()
        unknown=f.loc[~known].groupby('target').source_date.min()
        pt=focal[['patent_id','grant_year','patent_date']].copy();pt['event_date']=first.reindex(ids).to_numpy(dtype='datetime64[ns]')
        # Empty datetime mappings are reindexed explicitly for pandas compatibility.
        if first.empty:pt['event_date']=pd.Series(pd.NaT,index=pt.index,dtype='datetime64[ns]')
        unk=unknown.reindex(ids).to_numpy(dtype='datetime64[ns]');pt['unknown_date']=unk
        pt['censor_date']=pt.patent_id.map(censor)
        pt['usable']=pt.patent_id.map(index.cpc_subclass).notna() & (pt.unknown_date.isna() | (pt.event_date.notna()&pt.event_date.le(pt.unknown_date)))
        pt['duration']=(pt.event_date.fillna(pt.censor_date)-pt.patent_date).dt.days.astype(float)
        pt['event']=pt.event_date.notna().astype(int);pt['domain']=spec['id'];patients.append(pt)
        for cohort,mask in [('2000-2014',pt.grant_year.le(2014)),('2015-2018',pt.grant_year.between(2015,2018)),('2019-2025',pt.grant_year.ge(2019))]:
            sub=pt.loc[mask&pt.usable&pt.duration.ge(0)];q=kaplan_meier(sub.duration,sub.event);q['domain']=spec['id'];q['cohort']=cohort;q['sample_n']=len(sub);kmrows.append(q)
    result=pd.DataFrame(rows);save_table(result,out/'tables/cohort_statistics',latex=False)
    save_table(pd.concat(kmrows,ignore_index=True),out/'tables/first_passage_KM',latex=False)
    pd.concat(patients,ignore_index=True).to_csv(out/'data/first_passage_patents.csv',index=False)
    return result
