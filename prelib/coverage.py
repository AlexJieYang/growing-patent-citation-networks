import pandas as pd
from .common import shared_dir, write_json


def resolve_coverage(cfg, paths, metadata_max_grant):
    requested=pd.Timestamp(cfg['citation_observation_end']).normalize()
    if not cfg.get('source_observation_end'):
        raise ValueError('请在00中填写已经核实的source_observation_end，不能将目标日期当作数据覆盖日期。')
    source=pd.Timestamp(cfg['source_observation_end']).normalize()
    maximum=pd.Timestamp(metadata_max_grant).normalize()
    if pd.isna(maximum) or pd.isna(source):raise ValueError('授权日期或来源覆盖日期无效。')
    effective=min(requested,source,maximum)
    result={'requested_observation_end':str(requested.date()),'source_observation_end':str(source.date()),
            'metadata_max_grant':str(maximum.date()),'effective_observation_end':str(effective.date()),
            'requested_end_available':bool(effective==requested)}
    write_json(shared_dir(cfg)/'data/observation_window.json',result)
    return effective,result
