"""Checks of executable notebooks, data alignment, chronology and saved figures.

Data-dependent checks skip until ``python run_all.py --demo`` has completed.
"""
from pathlib import Path
import ast
import json
import xml.etree.ElementTree as ET
import re
import numpy as np
import pandas as pd
import pytest
import nbformat
from PIL import Image
from prelib.common import ROOT,load_config,domain_config,run_dir,initialize_run
from prelib.graphs import network_data


@pytest.fixture(scope='module')
def demo_cfg():
    path=ROOT/'config_demo.json'
    if not path.exists():pytest.skip('Run --demo before data-dependent checks.')
    cfg=load_config(path)
    if not (run_dir(cfg)/'figures/Main_Fig05_Connectivity.svg').is_file():
        pytest.skip('Run --demo through the figure stage.')
    return cfg


def test_notebooks_are_python_and_parseable():
    paths=sorted(ROOT.glob('0*.ipynb'));assert len(paths)==6
    for p in paths:
        nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
        assert nb.metadata.kernelspec.language=='python'
        for cell in nb.cells:
            if cell.cell_type=='code':ast.parse(cell.source,filename=p.name)
    assert not list(ROOT.rglob('*.R')) and not list(ROOT.rglob('*.Rmd'))
    for p in (ROOT/'prelib').glob('*.py'):
        tree=ast.parse(p.read_text(encoding='utf-8'))
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
            elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
        assert not any(x.split('.')[0] in {'statsmodels','rpy2','duckdb','sqlalchemy'} for x in imports)


def test_reset_cannot_delete_prepared_input(tmp_path,monkeypatch):
    import prelib.common as common
    monkeypatch.setattr(common,'ROOT',tmp_path)
    out=tmp_path/'outputs/new_run';out.mkdir(parents=True)
    sentinel=out/'original_data';sentinel.write_text('keep')
    cfg={'run_name':'new_run','prepared_data_root':str(out)}
    with pytest.raises(ValueError):initialize_run(cfg)
    assert sentinel.read_text()=='keep'
    with pytest.raises(ValueError):initialize_run({'run_name':'../escape'})
    with pytest.raises(ValueError):initialize_run({'run_name':'new_run','reuse_prepared_if_available':True,'reuse_prepared_run':'new_run'})
    assert sentinel.exists()


def test_full_census_and_vectors_survive_missing_attributes(demo_cfg):
    shared=pd.read_parquet(run_dir(demo_cfg)/'data/nodes.parquet')
    for spec in demo_cfg['domains']:
        cfg=domain_config(demo_cfg,spec['id']);nodes,e,v,days=network_data(cfg)
        wanted=shared.loc[shared.is_model&shared.domain.eq(spec['id']).fillna(False),'patent_id']
        assert set(wanted)==set(nodes.patent_id)
        assert len(nodes)==1107
        valid=np.isfinite(v).all(1)
        np.testing.assert_allclose(np.linalg.norm(v[valid],axis=1),1,atol=2e-6)
        assert np.all(days[e.t]<days[e.s])
        assert not e.duplicated(['s','t']).any()


def test_prepared_cache_checks_census_and_followup(demo_cfg,tmp_path,monkeypatch):
    import prelib.common as common
    import prelib.prepare as prepare
    candidate=run_dir(demo_cfg).resolve()
    raw=(ROOT/demo_cfg['source_root']).resolve()
    cfg={**demo_cfg,'run_name':'import_check','prepared_data_root':str(candidate),'source_root':str(raw)}
    monkeypatch.setattr(common,'ROOT',tmp_path)
    monkeypatch.setattr(prepare,'ROOT',tmp_path)
    result=prepare.prepare_data(cfg)
    assert result['domain_nodes']==2214
    original=pd.read_parquet(candidate/'data/nodes.parquet')
    copied=pd.read_parquet(run_dir(cfg)/'data/nodes.parquet')
    np.testing.assert_array_equal(original.patent_id,copied.patent_id)
    too_long={**cfg,'run_name':'long_check','outcome_years':6}
    with pytest.raises(ValueError,match='follow-up'):prepare.prepare_data(too_long)
    changed={**cfg,'run_name':'census_check','domains':[dict(d) for d in cfg['domains']]}
    changed['domains'][0]['include']=['G06N3/0409']
    with pytest.raises(ValueError,match='census'):prepare.prepare_data(changed)


def test_config_rejects_truncated_primary_followup(tmp_path):
    cfg=load_config(ROOT/'config.json');cfg['primary_outcome_years']=6;cfg['outcome_years']=5
    p=tmp_path/'bad_config.json';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='window'):load_config(p)


def test_opportunity_degree_uses_all_strictly_past_edges(demo_cfg):
    for spec in demo_cfg['domains']:
        cfg=domain_config(demo_cfg,spec['id']);out=run_dir(cfg);nodes,e,v,days=network_data(cfg)
        for mode in ['full','known']:
            for split in ['train','validation','test']:
                with np.load(out/f'data/opportunities_{mode}_{split}.npz') as d:
                    assert len(d['source'])>0
                    np.testing.assert_array_equal(d['target'],d['alternatives'][:,0])
                    for j in np.linspace(0,len(d['source'])-1,min(12,len(d['source']))).astype(int):
                        source=int(d['source'][j]);alternative=d['alternatives'][j]
                        assert len(np.unique(alternative))==len(alternative)
                        assert np.all(days[alternative]<days[source])
                        prior=e.loc[days[e.s.to_numpy()]<days[source]]
                        degree=np.bincount(prior.t,minlength=len(nodes))
                        np.testing.assert_allclose(np.expm1(d['X'][j,:,0]),degree[alternative],atol=4e-4)
                        age=(days[source]-days[alternative])/365.25
                        np.testing.assert_allclose(10*d['X'][j,:,1],age,atol=5e-6)
                        pool=(days<days[source])&(days>=days[source]-cfg['max_reference_age_years']*365.25)
                        if mode=='known':pool&=nodes.vector_valid.to_numpy()&nodes.community.ge(0).to_numpy()
                        assert d['risk_size'][j]==pool.sum()


def test_simulations_preserve_outdegrees_fixed_edges_and_chronology(demo_cfg):
    for spec in demo_cfg['domains']:
        cfg=domain_config(demo_cfg,spec['id']);out=run_dir(cfg);nodes,e,v,days=network_data(cfg)
        end=pd.Timestamp(cfg['simulation_end_date']).to_numpy().astype('datetime64[D]').astype(int)
        observed=e.loc[days[e.s.to_numpy()]<=end,['s','t']].to_numpy()
        dates=nodes.patent_date.to_numpy();start=np.datetime64(cfg['simulation_start_date'])
        c=nodes.community.to_numpy();valid=nodes.vector_valid.to_numpy()&(c>=0)
        ar=e[['s','t']].to_numpy();age=(days[ar[:,0]]-days[ar[:,1]])/365.25
        modeled=(dates[ar[:,0]]>=start)&(days[ar[:,0]]<=end)&valid[ar[:,0]]&valid[ar[:,1]]&(age<=cfg['max_reference_age_years'])
        fixed=set(map(tuple,ar[(days[ar[:,0]]<=end)&~modeled]))
        for p in (out/'models').glob('simulation_*_example.parquet'):
            sample=pd.read_parquet(p).to_numpy()
            np.testing.assert_array_equal(np.bincount(sample[:,0],minlength=len(nodes)),np.bincount(observed[:,0],minlength=len(nodes)))
            assert np.all(days[sample[:,1]]<days[sample[:,0]])
            assert fixed.issubset(set(map(tuple,sample)))
            assert len(set(map(tuple,sample)))==len(sample)
        curves=pd.read_csv(out/'tables/growth_edge_curves.csv')
        for _,g in curves.groupby(['model','repetition','metric']):
            assert g.probability.between(0,1+1e-12).all()
            delta=g.sort_values('x').probability.diff().dropna()
            assert (delta>=-1e-12).all() if g.metric.iloc[0]=='distance' else (delta<=1e-12).all()


def test_null_scope_and_fixed_observables(demo_cfg):
    for spec in demo_cfg['domains']:
        out=run_dir(domain_config(demo_cfg,spec['id']))
        nodes,e,_,_=network_data(domain_config(demo_cfg,spec['id']))
        draws=pd.read_csv(out/'tables/null_draws.csv')
        assert len(draws)==len(demo_cfg['null_models'])*demo_cfg['null_repetitions']
        assert draws.nodes.eq(len(nodes)).all() and draws.edges.eq(len(e)).all()
        for key in ['max_indegree','indegree_variance','indegree_gini']:
            assert np.ptp(draws[key])<1e-10
        temporal=draws.loc[draws.null_model.eq('Topology')].set_index('repetition')
        trace=pd.read_csv(out/'tables/null_trace.csv')
        assert (trace.fixed_proposals==trace.attempts).all()
        assert trace.proposal_acceptance.ge(demo_cfg['minimum_null_acceptance']).all()


def test_publication_panels_and_complete_exports(demo_cfg):
    folder=run_dir(demo_cfg)/'figures'
    # Only published plates count; interrupted exports may leave temporary files.
    files=sorted([*folder.glob('Main_Fig*.svg'), *folder.glob('SI_Fig*.svg')])
    assert len(files)==10
    for p in files:
        doc=ET.parse(p);texts=[''.join(t.itertext()) for t in doc.iter() if t.tag.endswith('}text')]
        panels=[t for t in texts if re.match(r'^\([a-z]\)$',t)]
        assert len(panels) in {4,6},(p.name,len(panels))
        assert 'SYNTHETIC DATA' in texts
        with Image.open(p.with_suffix('.png')) as im:
            im.load();assert im.width>=2000 and im.height>=1400
        pdf=p.with_suffix('.pdf').read_bytes()
        assert pdf.startswith(b'%PDF-') and b'%%EOF' in pdf[-1024:]
