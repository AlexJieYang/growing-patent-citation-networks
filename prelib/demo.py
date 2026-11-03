"""Fast two-domain synthetic fixture through 2025-12-31. Never empirical evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import ROOT,write_json


def create_demo_source(destination=None,seed=731,per_year=72):
    default = destination is None
    destination = Path(destination or ROOT/"outputs/_demo_source")
    core = destination/"02_core";emb = destination/"06_embedding_metrics/patentsberta_v1/02_cache"
    core.mkdir(parents=True,exist_ok=True);emb.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(seed);dim=32;years=np.repeat(np.arange(1985,2026),per_year);N=len(years)
    local = np.tile(np.arange(per_year),41)
    domain_type = np.where(local%8<3,0,np.where(local%8<6,1,2))
    codes = ["G06N3/0409","G06N3/0418","G06N3/042","G06N3/061","G06N3/063","G06N3/067","G06N3/082","G06N3/084","G06N3/088"]+[f"H01S3/{j:02d}" for j in range(1,10)]+["G06F16/00","H04L9/00","G02F1/00"]
    ci=np.empty(N,dtype=int)
    for d,lo,hi in [(0,0,9),(1,9,18),(2,18,21)]:ci[domain_type==d]=rng.integers(lo,hi,size=np.sum(domain_type==d))
    latent=np.r_[np.repeat(np.arange(3),3),np.repeat(np.arange(3),3),np.arange(3)]
    prototype=rng.normal(size=(3,dim));prototype/=np.linalg.norm(prototype,axis=1,keepdims=True)
    vectors=prototype[latent[ci]]+rng.normal(scale=.14,size=(N,dim));vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
    dates=pd.to_datetime(pd.Series(years).astype(str)+"-01-01")+pd.to_timedelta((local//2)*(350/max(1,per_year//2)),unit="D")
    dates.iloc[-1]=pd.Timestamp("2025-12-31")
    days=dates.to_numpy(dtype="datetime64[D]").astype("int64");degree=np.zeros(N,dtype=int)
    ids=4_000_000+np.arange(N)*7;edge_s=[];edge_t=[]
    for i in range(N):
        hi=np.searchsorted(days,days[i],side="left");lo=np.searchsorted(days,days[i]-25*365.25)
        if hi<=lo:continue
        previous=np.arange(lo,hi);d=np.clip(1-vectors[previous]@vectors[i],0,2)
        cross=latent[ci[previous]]!=latent[ci[i]]
        logw=.65*np.log1p(degree[previous])-(days[i]-days[previous])/365.25/13-1.8*d+.5*cross-.9*cross*d
        probability=np.exp(logw-logw.max());probability/=probability.sum()
        targets=rng.choice(previous,size=min(len(previous),max(1,int(rng.poisson(7)))),replace=False,p=probability)
        edge_s.extend([ids[i]]*len(targets));edge_t.extend(ids[targets]);np.add.at(degree,targets,1)
    raw_codes=np.array(codes,dtype=object)[ci]
    df=pd.DataFrame({"patent_id":ids,"patent_date":dates,"cpc_group":raw_codes,
                     "cpc_subclass":[x[:4] for x in raw_codes]})
    df["filing_date"]=df.patent_date-pd.to_timedelta(rng.integers(150,900,size=N),unit="D")
    df["filing_year"]=df.filing_date.dt.year.astype("int16");df["grant_year"]=years.astype("int16")
    df["grant_lag_years"]=(df.patent_date-df.filing_date).dt.days/365.25
    df[["patent_id","patent_date","filing_date","filing_year","grant_year","grant_lag_years"]].sample(frac=1,random_state=3).to_parquet(core/"patent_dates.parquet",index=False)
    fields=df[["patent_id","cpc_subclass","cpc_group"]].copy()
    # Canonicalization must also handle zero-padded main groups and spaces.
    fields.loc[fields.index[::97],"cpc_group"]=fields.loc[fields.index[::97],"cpc_group"].str.replace("G06N3/","G06N 003/",regex=False)
    fields.to_parquet(core/"patent_primary_field.parquet",index=False)
    secondary=fields.loc[(df.grant_year>=2000)&(domain_type==0)].head(4).copy();secondary["cpc_group"]="H01S3/04";secondary["cpc_subclass"]="H01S"
    pd.concat([fields,secondary],ignore_index=True).to_parquet(core/"patent_cpc.parquet",index=False)
    e=pd.DataFrame({"patent_id":edge_s,"citation_patent_id":edge_t}).sample(frac=1,random_state=4)
    e.to_parquet(core/"citation.parquet",index=False);r=rng.random(len(e))
    e.loc[r<.72].to_parquet(core/"citation_applicant.parquet",index=False)
    e.loc[r>.65].to_parquet(core/"citation_examiner.parquet",index=False)
    order=rng.permutation(N);matrix=vectors[order].astype("float32");matrix[0]=0;matrix[1,0]=np.nan
    np.save(emb/"vectors.npy",matrix)
    pd.DataFrame({"patent_id":ids[order],"embedding_row":np.arange(N,dtype="int64")}).to_parquet(emb/"index.parquet",index=False)
    cfg=json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
    cfg.update({"run_name":"demo_physics","source_root":"outputs/_demo_source" if default else str(destination.resolve()),
      "synthetic_data":True,"domains":[{"id":"neural","label":"Neural networks","include":["G06N3/02*","G06N3/04*","G06N3/06*","G06N3/08*","G06N3/09*","G06N3/10*"],"exclude":[]},
                                         {"id":"laser","label":"Lasers","include":["H01S3"],"exclude":[]}],
      "max_nodes_per_domain":5000,"max_total_model_nodes":10000,"max_boundary_nodes":15000,"max_kept_edges":100000,
      "negative_controls":8,"max_choice_events_per_split":800,"null_repetitions":9,"null_min_repetitions":5,
      "null_burnin_proposals_per_edge":10,"null_between_proposals_per_edge":5,"simulation_repetitions":3,
      "max_simulation_births":300,
      "source_observation_end":"2025-12-31", "reuse_prepared_if_available":False, "prepared_data_root":"",
      "slq_probes":8,"slq_steps":32,"spectral_coarse_probes":4,"spectral_coarse_steps":16,"spectral_exact_limit":1200,"robustness_repetitions":5,"layout_nodes":220,"layout_edges":700,
      "bootstrap_repetitions":49,"spectral_tau_points":45})
    # Keep the empirical 600-dpi setting unchanged; the synthetic check uses 300 dpi.
    cfg['figure_dpi']=300
    write_json(ROOT/"config_demo.json",cfg)
    write_json(destination/"SYNTHETIC_ONLY.json",{"synthetic":True,"nodes":N,"edges":len(e),"through":"2025-12-31"})
    return cfg
