import numpy as np
import pandas as pd
import networkx as nx
from .common import main_group

def frozen_communities(nodes, refs, cfg):
    """Only pre-freeze edges. Labels remain fixed in every subsequent analysis and simulation."""
    fields = nodes.set_index("patent_id").tech_code
    year = refs.source_date.dt.year
    h = refs.loc[year.between(cfg["community_freeze_year"] - cfg["community_history_years"],
                              cfg["community_freeze_year"] - 1)].copy()
    h["s_code"], h["t_code"] = h.source.map(fields), h.target.map(fields)
    h = h.dropna(subset=["s_code", "t_code"])
    graph = nx.Graph()
    graph.add_nodes_from(sorted(set(h.s_code) | set(h.t_code)))
    different = h.loc[h.s_code.ne(h.t_code)]
    pairs = different.groupby(["s_code", "t_code"]).size()
    for (a, b), weight in pairs.items():
        old = graph.get_edge_data(a, b, {}).get("weight", 0)
        graph.add_edge(a, b, weight=old + int(weight))
    if graph.number_of_edges() == 0:
        raise ValueError("冻结期没有跨技术代码连接；请扩大历史题域或调整冻结年。")
    groups = nx.community.louvain_communities(
        graph, weight="weight", resolution=cfg["community_resolution"], seed=cfg["seed"])
    groups = sorted([sorted(x) for x in groups], key=lambda x: x[0])
    exact = {code: c for c, group in enumerate(groups) for code in group}
    # Parent rules use unique baseline patents, never later patent frequencies,
    # outcomes, or semantic vectors. Exact code labels always take priority.
    baseline_ids=pd.Index(pd.concat([h.source,h.target],ignore_index=True).unique())
    baseline=nodes.loc[nodes.patent_id.isin(baseline_ids),['patent_id','tech_code']].drop_duplicates('patent_id')
    baseline['parent']=main_group(baseline.tech_code)
    baseline['community']=baseline.tech_code.map(exact)
    counts=baseline.dropna().groupby(['parent','community']).size().rename('n').reset_index()
    parents={}
    for parent,q in counts.groupby('parent',sort=True):
        winner=q.sort_values(['n','community'],ascending=[False,True]).iloc[0]
        n=int(q.n.sum());purity=float(winner.n/n)
        if n>=cfg.get('community_parent_min_patents',10) and purity>=cfg.get('community_parent_min_purity',.6):
            parents[parent]=(int(winner.community),purity,n)
    mapping=dict(exact);records=[]
    all_codes=pd.Series(sorted(nodes.tech_code.dropna().unique()),dtype='string')
    for code,parent in zip(all_codes,main_group(all_codes)):
        if code in exact:
            records.append({'tech_code':code,'community':exact[code],'assignment':'exact','parent':parent})
        elif pd.notna(parent) and parent in parents:
            community,purity,n=parents[parent];mapping[code]=community
            records.append({'tech_code':code,'community':community,'assignment':'baseline_parent',
                            'parent':parent,'parent_purity':purity,'baseline_parent_n':n})
    members=pd.DataFrame(records)
    summary = {"freeze_year": cfg["community_freeze_year"], "history_edges": len(h),
             "code_nodes": graph.number_of_nodes(), "code_edges": graph.number_of_edges(),
             "n_communities": len(groups), "resolution": cfg["community_resolution"],
             "labels_frozen_after_baseline": True,
             "classification_basis": "current CPC assigned retrospectively"}
    if len(groups) < 2:
        raise ValueError("仅有一个历史社群，无法识别跨社群项。调整题域或社群分辨率后使用新的 run_name。")
    return mapping, members, summary


