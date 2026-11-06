"""Notebook-equivalent execution entry point."""
from __future__ import annotations
import time
from .common import (load_config, initialize_run, set_active_config, domain_config,
                     shared_dir, write_json, environment_report)
from .prepare import prepare_data
from .graphs import build_networks
from .topology import run_topology
from .kinetics import build_opportunities, run_enrichment
from .diffusion import run_spectral
from .percolation import run_percolation
from .growth import simulate_growth
from .nulls import run_null_models
from .publication import make_figures


def run_all(config_path='config.json', demo=False, *, source_root=None, run_name=None):
    if demo:
        from .demo import create_demo_source
        create_demo_source()
        config_path='config_demo.json'
    cfg=load_config(config_path)
    if source_root is not None:
        if demo:
            raise ValueError('--source-root cannot be combined with the synthetic demo.')
        cfg['source_root']=str(source_root)
    if run_name is not None:
        cfg['run_name']=str(run_name)
    # Verify configured inputs before resetting an existing result directory.
    from .preflight import check_input_files
    check_input_files(cfg)
    initialize_run(cfg)
    active=shared_dir(cfg).parent/'active_run_config.json'
    write_json(active,cfg)
    set_active_config(str(active))
    write_json(shared_dir(cfg)/'models/config_used.json',cfg)
    write_json(shared_dir(cfg)/'models/environment.json',environment_report(cfg))
    stages=[('Preparation',prepare_data),('Network scopes',build_networks),
            ('Topology',run_topology)]
    for name,fun in stages:
        start=time.perf_counter();fun(cfg)
        print(f'{name}: complete ({time.perf_counter()-start:.1f} s)',flush=True)
    for spec in cfg['domains']:
        dc=domain_config(cfg,spec['id'])
        for fun in [build_opportunities,run_enrichment,run_spectral,run_percolation,
                    simulate_growth,run_null_models]:
            start=time.perf_counter();fun(dc)
            print(f'{spec["id"]}: {fun.__name__} complete ({time.perf_counter()-start:.1f} s)',flush=True)
    figures=make_figures(cfg)
    print(f'Complete: {shared_dir(cfg)}',flush=True)
    return figures
