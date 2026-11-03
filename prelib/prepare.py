"""Bounded, read-only ingestion of the user's existing Parquet/NPY core.

Full citation files are scanned in batches, never loaded as a full DataFrame.
The model is a domain census, not a random induced-node sample. Reference and
forward-citation boundaries are retained separately from the internal network.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .cpc import domain_mask
from .coverage import resolve_coverage
from .common import ROOT, main_group, normalize_code, run_dir, shared_dir, save_table, write_json


def schema(path):
    return pq.ParquetFile(path).schema_arrow.names


def batches(path, columns, batch_rows=200_000):
    missing = sorted(set(columns) - set(schema(path)))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        yield batch.to_pandas()


def strict_ids(series):
    if pd.api.types.is_float_dtype(series):
        raise TypeError("专利ID不得经过浮点数；请使用原始 int64/string ID。")
    if series.isna().any():
        raise ValueError("核心关系中发现空专利ID。")
    if not pd.api.types.is_integer_dtype(series):
        if not series.astype("string").str.fullmatch(r"\d+").all():
            raise ValueError("本流程只接受核心层 numeric utility-grant patent_id。")
    return series.astype("int64")


def read_ids(path, ids, columns, cfg):
    keep = pd.Index(np.asarray(list(ids), dtype="int64"))
    parts = []
    for df in batches(path, columns, cfg["batch_rows"]):
        df["patent_id"] = strict_ids(df["patent_id"])
        part = df.loc[df.patent_id.isin(keep)]
        if len(part):
            parts.append(part.copy())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def unique_dimension(df, name):
    if df.patent_id.duplicated().any():
        raise ValueError(f"{name}: 专利主键重复，停止以防多对多连接。")
    return df


def _metadata(paths, ids, cfg):
    dates = read_ids(paths["dates"], ids,
                     ["patent_id", "patent_date", "filing_date"], cfg)
    unique_dimension(dates, "patent_dates")
    fields = read_ids(paths["fields"], ids,
                      ["patent_id", "cpc_subclass", "cpc_group"], cfg)
    unique_dimension(fields, "patent_primary_field")
    out = pd.DataFrame({"patent_id": sorted(ids)}).merge(
        dates, on="patent_id", how="left", validate="one_to_one").merge(
        fields, on="patent_id", how="left", validate="one_to_one")
    out["patent_date"] = pd.to_datetime(out.patent_date, errors="coerce")
    out["filing_date"] = pd.to_datetime(out.filing_date, errors="coerce")
    out["grant_year"] = out.patent_date.dt.year.astype("Int16")
    out["cpc_subclass"] = normalize_code(out.cpc_subclass)
    out["tech_code"] = normalize_code(out.cpc_group)
    out["tech_main_group"] = main_group(out.tech_code)
    # A missing exact CPC subgroup is not silently replaced by a coarser class.
    out.loc[~out.tech_code.str.contains("/", na=False), "tech_code"] = pd.NA
    return out


def input_paths(cfg):
    root = Path(cfg["source_root"]).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    suffix = "" if cfg["edge_source"] == "all" else "_" + cfg["edge_source"]
    return {"dates": root / "02_core/patent_dates.parquet",
            "fields": root / "02_core/patent_primary_field.parquet",
            "edges": root / f"02_core/citation{suffix}.parquet",
            "cpc": root / "02_core/patent_cpc.parquet",
            "index": root / cfg["embedding_index"],
            "vectors": root / cfg["embedding_vectors"]}


def select_domains(cfg, paths):
    selection_path = paths["fields"] if cfg["cpc_match_mode"] == "primary" else paths["cpc"]
    selected = {d["id"]: set() for d in cfg["domains"]}
    for f in batches(selection_path, ["patent_id", "cpc_subclass", "cpc_group"], cfg["batch_rows"]):
        f["patent_id"] = strict_ids(f.patent_id)
        codes = normalize_code(f.cpc_group).fillna(normalize_code(f.cpc_subclass))
        for spec in cfg["domains"]:
            selected[spec["id"]].update(f.loc[domain_mask(codes, spec), "patent_id"].tolist())
    ids = set().union(*selected.values())
    if not ids:
        raise ValueError("CPC筛选为空，请检查选择器。")
    parts, maximum = [], pd.NaT
    for f in batches(paths["dates"], ["patent_id", "patent_date"], cfg["batch_rows"]):
        f["patent_id"] = strict_ids(f.patent_id)
        f["patent_date"] = pd.to_datetime(f.patent_date, errors="coerce")
        value = f.patent_date.max()
        if pd.notna(value): maximum = value if pd.isna(maximum) else max(maximum, value)
        if f.patent_id.isin(ids).any(): parts.append(f.loc[f.patent_id.isin(ids)].copy())
    dates = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["patent_id", "patent_date"])
    unique_dimension(dates, "selected dates")
    cutoff, coverage = resolve_coverage(cfg, paths, maximum)
    dates = dates.loc[dates.patent_date.dt.year.between(cfg["network_start_year"], cfg["network_end_year"])
                      & dates.patent_date.le(cutoff)]
    membership = pd.concat([pd.DataFrame({"patent_id": sorted(v), "domain": k}) for k,v in selected.items()], ignore_index=True)
    membership = membership.merge(dates, on="patent_id", validate="many_to_one")
    overlap_n = int(membership.loc[membership.patent_id.duplicated(False), "patent_id"].nunique())
    if overlap_n and cfg["overlap_policy"] == "error":
        raise ValueError(f"{overlap_n}件专利跨领域重复。请修改筛选，或显式选择 overlap_policy=priority。")
    membership = membership.drop_duplicates("patent_id", keep="first")
    count = membership.groupby("domain").size().reindex(selected, fill_value=0).rename("nodes").reset_index()
    save_table(count, shared_dir(cfg) / "tables/domain_sizes")
    if count.nodes.eq(0).any(): raise ValueError("至少一个领域为空；见领域规模表。")
    if count.nodes.gt(cfg["max_nodes_per_domain"]).any() or len(membership) > cfg["max_total_model_nodes"]:
        raise MemoryError("领域节点超过预算，请缩小CPC选择器；不会随机删节点或截断年份。")
    coverage["overlapping_ids_resolved"] = overlap_n
    return membership.reset_index(drop=True), cutoff, coverage


def collect_edges(cfg, paths, model_ids, focal_ids):
    parts, scanned, kept = [], 0, 0
    model_ids, focal_ids = pd.Index(model_ids), pd.Index(focal_ids)
    for df in batches(paths["edges"], ["patent_id", "citation_patent_id"],
                      cfg["batch_rows"]):
        df["patent_id"] = strict_ids(df.patent_id)
        df["citation_patent_id"] = strict_ids(df.citation_patent_id)
        scanned += len(df)
        hit = df.patent_id.isin(model_ids) | df.citation_patent_id.isin(focal_ids)
        if hit.any():
            p = df.loc[hit].rename(columns={"patent_id": "source", "citation_patent_id": "target"})
            parts.append(p.copy())
            kept += len(p)
        if kept > cfg["max_kept_edges"]:
            raise MemoryError("保留的边超过 max_kept_edges。请缩小焦点题域/年份；不会截断数据。")
    if not parts:
        raise ValueError("该题域没有任何引用关系。")
    edges = pd.concat(parts, ignore_index=True)
    before = len(edges)
    edges = edges.drop_duplicates(["source", "target"]).reset_index(drop=True)
    return edges, {"scanned_citation_rows": scanned, "retained_before_dedup": before,
                   "duplicate_pairs_removed": before - len(edges)}


def load_selected_vectors(cfg, paths, nodes, destination):
    raw = np.load(paths["vectors"], mmap_mode="r", allow_pickle=False)
    if raw.ndim != 2:
        raise ValueError("embedding_vectors 必须是二维 NPY。")
    estimate = len(nodes) * raw.shape[1] * 4 / 1024**3
    if estimate > cfg["max_vector_gb"]:
        raise MemoryError(f"选中向量将占用 {estimate:.2f} GiB，超过 max_vector_gb。")
    row_col = cfg["embedding_row_column"]
    idx = read_ids(paths["index"], nodes.patent_id,
                   ["patent_id", row_col], cfg)
    unique_dimension(idx, "embedding index")
    if idx[row_col].isna().any() or not pd.api.types.is_integer_dtype(idx[row_col]):
        raise TypeError("embedding_row 必须是无缺失整数。")
    if idx[row_col].duplicated().any():
        raise ValueError("不同专利映射到同一向量行，需核验生成时的ID清单。")
    if len(idx) and (idx[row_col].min() < 0 or idx[row_col].max() >= len(raw)):
        raise IndexError("向量行映射越界。")
    index = nodes[["patent_id"]].merge(idx, on="patent_id", how="left", validate="one_to_one")
    matrix = np.lib.format.open_memmap(destination, mode="w+", dtype="float32",
                                       shape=(len(nodes), raw.shape[1]))
    matrix[:] = np.nan
    valid = np.zeros(len(nodes), dtype=bool)
    matched = np.flatnonzero(index[row_col].notna().to_numpy())
    for start in range(0, len(matched), 4096):
        pos = matched[start:start + 4096]
        block = np.asarray(raw[index.iloc[pos][row_col].astype("int64").to_numpy()], dtype="float32")
        norms = np.linalg.norm(block.astype("float64"), axis=1)
        good = np.isfinite(block).all(axis=1) & np.isfinite(norms) & (norms > 0)
        matrix[pos[good]] = block[good] / norms[good, None]
        valid[pos[good]] = True
    matrix.flush()
    del matrix, raw
    nodes["vector_row"] = np.arange(len(nodes), dtype="int32")
    nodes["vector_valid"] = valid
    return {"dimension": int(np.load(destination, mmap_mode="r").shape[1]),
            "selected_vector_GiB": estimate, "matched_ids": len(matched),
            "valid_vectors": int(valid.sum()), "selected_nodes": len(nodes),
            "invalid_or_missing_vectors": int(len(nodes)-valid.sum())}


def prepare_raw(cfg):
    out = run_dir(cfg)
    paths = input_paths(cfg)
    for key in ["dates", "fields", "edges", "index", "vectors"]:
        if not paths[key].is_file():
            raise FileNotFoundError(f"{key}: {paths[key]}。请修改 00 Notebook 的路径配置。")
    domain, cutoff, coverage = select_domains(cfg, paths)
    mature = domain.patent_date.add(pd.DateOffset(years=cfg["primary_outcome_years"])).le(cutoff)
    focal_mask = domain.patent_date.dt.year.between(cfg["focal_start_year"], cfg["focal_end_year"])
    focal_ids = set(domain.loc[focal_mask, "patent_id"])
    if not focal_ids:
        raise ValueError("没有焦点专利。检查领域与年份；近期专利允许进入但固定窗口结果会设为NA。")
    model_ids = set(domain.patent_id)
    edges, summary = collect_edges(cfg, paths, model_ids, focal_ids)
    all_ids = model_ids | set(edges.source) | set(edges.target)
    if len(all_ids) > cfg["max_boundary_nodes"]:
        raise MemoryError(f"含边界的节点数 {len(all_ids):,} 超过 max_boundary_nodes。请缩小题域。")
    nodes = _metadata(paths, all_ids, cfg)
    indexed = nodes.set_index("patent_id")
    gs = edges.source.map(indexed.patent_date)
    gt = edges.target.map(indexed.patent_date)
    no_dates = gs.isna() | gt.isna()
    self_edge = edges.source.eq(edges.target)
    same_day = gs.eq(gt)
    reverse_time = gt.gt(gs)
    summary.update({"edges_missing_dates": int(no_dates.sum()), "self_edges": int(self_edge.sum()),
                  "same_day_edges_excluded": int(same_day.sum()),
                  "reverse_time_edges_excluded": int(reverse_time.sum())})
    edges = edges.loc[~no_dates & ~self_edge & gt.lt(gs) & gs.le(cutoff)].copy()
    edges = edges.sort_values(["source", "target"]).reset_index(drop=True)
    edges["source_date"] = edges.source.map(indexed.patent_date)
    edges["target_date"] = edges.target.map(indexed.patent_date)
    refs = edges.loc[edges.source.isin(model_ids)].copy()
    forward = edges.loc[edges.target.isin(focal_ids)].copy()
    end_dates = indexed.loc[list(focal_ids), "patent_date"] + pd.DateOffset(years=cfg["outcome_years"])
    forward = forward.loc[forward.source_date.le(forward.target.map(end_dates))].copy()
    internal = refs.loc[refs.target.isin(model_ids)].copy()
    used = model_ids | set(refs.target) | set(forward.source)
    nodes = nodes.loc[nodes.patent_id.isin(used)].sort_values("patent_id").reset_index(drop=True)
    nodes["domain"] = nodes.patent_id.map(domain.set_index("patent_id").domain).astype("string")
    nodes["is_model"] = nodes.patent_id.isin(model_ids)
    nodes["is_focal"] = nodes.patent_id.isin(focal_ids)
    va = load_selected_vectors(cfg, paths, nodes, out / "data/vectors.npy")
    save_table(pd.DataFrame([va]), out / "tables/vector_coverage")
    nodes.to_parquet(out / "data/nodes.parquet", index=False)
    for name, frame in [("reference_edges", refs), ("forward_edges", forward), ("internal_edges", internal)]:
        frame.to_parquet(out / f"data/{name}.parquet", index=False)
    summary.update({"domain_census_nodes": len(model_ids), "focal_nodes_including_recent": len(focal_ids), "mature_focal_nodes": int((focal_mask & mature).sum()),
                  "recent_focal_nodes_retained": int((focal_mask & ~mature).sum()),
                  "boundary_included_nodes": len(nodes), "reference_edges": len(refs),
                  "internal_edges": len(internal), "forward_edges": len(forward),
                  "edge_source": cfg["edge_source"],
                  "citation_observation_end": str(cutoff.date()), "coverage": coverage,
                  "citation_scans": 1,
                  "estimand": "CPC-selected domain census; internal growth; all-field citation outcomes",
                  "classification_vintage": "current CPC, retrospective reconstruction"})
    write_json(out/'data/prepared_scope.json',{'edge_source':cfg['edge_source'],
        'max_outcome_years':cfg['outcome_years']})
    save_table(pd.DataFrame([{"item": k, "value": v} for k, v in summary.items()
                             if isinstance(v, (int, str))]), out / "tables/sample_overview")
    gc.collect()
    return summary


def load_data(cfg):
    out = shared_dir(cfg)
    nodes = pd.read_parquet(out / "data/nodes.parquet")
    vectors = np.load(out / "data/vectors.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(nodes.vector_row.to_numpy(), np.arange(len(nodes))):
        raise ValueError("节点行顺序与 vector_row 不一致。")
    if len(vectors) != len(nodes):
        raise ValueError("节点数与选中向量不一致。")
    return nodes, vectors


def prepare_data(cfg):
    """Read raw Parquet once, or import a deliberately selected prepared dataset."""
    import shutil,json
    selected=cfg.get('prepared_data_root','')
    candidate=Path(selected).expanduser() if selected else ROOT/'outputs'/cfg.get('reuse_prepared_run','')
    if selected and not candidate.is_absolute():candidate=ROOT/candidate
    if selected and not (candidate/'data/nodes.parquet').is_file():
        raise FileNotFoundError(f'Prepared data not found: {candidate}')
    reuse=bool(selected) or (cfg.get('reuse_prepared_if_available',False) and (candidate/'data/nodes.parquet').is_file())
    if not reuse:return prepare_raw(cfg)
    out=run_dir(cfg)
    if candidate.resolve()==out.resolve():raise ValueError('The prepared input and new result directories must differ.')
    if cfg['cpc_match_mode']!='primary':raise ValueError('Reusing prepared data is supported for unchanged primary-CPC scopes only.')
    source=candidate/'data'
    required=['nodes.parquet','reference_edges.parquet','forward_edges.parquet','internal_edges.parquet','vectors.npy','observation_window.json']
    for name in required:
        if not (source/name).is_file():raise FileNotFoundError(source/name)
    obs=json.loads((source/'observation_window.json').read_text(encoding='utf-8'))
    end=min(pd.Timestamp(cfg['citation_observation_end']),pd.Timestamp(cfg['source_observation_end']),pd.Timestamp(obs['metadata_max_grant']))
    if end!=pd.Timestamp(obs['effective_observation_end']):raise ValueError('Observation coverage changed; prepare from raw data.')
    nodes=pd.read_parquet(source/'nodes.parquet')
    model=nodes.loc[nodes.is_model]
    paths=input_paths(cfg)
    if not paths['fields'].is_file() or not paths['dates'].is_file():
        raise FileNotFoundError('CPC/date metadata are required to verify an imported domain census; citation rescanning is not needed.')
    expected,new_end,_=select_domains(cfg,paths)
    actual=set(zip(model.patent_id.astype('int64'),model.domain.astype(str)))
    wanted=set(zip(expected.patent_id.astype('int64'),expected.domain.astype(str)))
    if actual!=wanted or new_end!=end:raise ValueError('Prepared census differs from requested CPC/date scope; prepare from raw data.')
    scope_file=source/'prepared_scope.json'
    prepared_scope=None
    if scope_file.is_file():
        prepared_scope=json.loads(scope_file.read_text(encoding='utf-8'))
    old_overview=candidate/'tables/sample_overview.csv'
    if prepared_scope is None and old_overview.is_file():
        old=pd.read_csv(old_overview)
        if {'item','value'}.issubset(old.columns):
            declared=old.loc[old.item.eq('edge_source'),'value']
            # Legacy caches are supported only for the supplied five-year workflow.
            if len(declared):prepared_scope={'edge_source':str(declared.iloc[0]),'max_outcome_years':5}
    if prepared_scope is None:
        raise ValueError('Prepared citation provenance is missing; use raw preparation.')
    if prepared_scope['edge_source']!=cfg['edge_source']:raise ValueError('Citation source changed.')
    if cfg['outcome_years']>prepared_scope['max_outcome_years']:
        raise ValueError('Prepared forward edges do not cover the requested follow-up window.')
    expected_focal=model.patent_date.dt.year.between(cfg['focal_start_year'],cfg['focal_end_year'])
    if not np.array_equal(model.is_focal.to_numpy(),expected_focal.to_numpy()):raise ValueError('Focal years changed.')
    for spec in cfg['domains']:
        part=model.loc[model.domain.eq(spec['id']).fillna(False)]
        if part.empty or not domain_mask(part.tech_code.fillna(part.cpc_subclass),spec).all():
            raise ValueError('Prepared CPC scope is incompatible; prepare from raw data.')
    if set(model.domain.dropna())!={d['id'] for d in cfg['domains']}:raise ValueError('Prepared domains differ.')
    if not model.patent_date.dt.year.between(cfg['network_start_year'],cfg['network_end_year']).all():raise ValueError('Network years changed.')
    if not nodes.loc[nodes.is_focal].patent_date.dt.year.between(cfg['focal_start_year'],cfg['focal_end_year']).all():raise ValueError('Focal years changed.')
    if model.groupby('domain').size().gt(cfg['max_nodes_per_domain']).any() or len(model)>cfg['max_total_model_nodes']:raise MemoryError('Prepared sample exceeds node budget.')
    if len(nodes)>cfg['max_boundary_nodes']:raise MemoryError('Prepared boundaries exceed budget.')
    v=np.load(source/'vectors.npy',mmap_mode='r',allow_pickle=False)
    if len(v)!=len(nodes) or not np.array_equal(nodes.vector_row,np.arange(len(nodes))):raise ValueError('Prepared vector mapping invalid.')
    if v.nbytes>cfg['max_vector_gb']*1024**3:raise MemoryError('Prepared vectors exceed budget.')
    for name in required[1:]:shutil.copy2(source/name,out/'data'/name)
    write_json(out/'data/prepared_scope.json',prepared_scope)
    columns=['patent_id','patent_date','filing_date','grant_year','cpc_subclass','cpc_group','tech_code',
             'tech_main_group','domain','is_model','is_focal','vector_row','vector_valid']
    nodes[[x for x in columns if x in nodes]].to_parquet(out/'data/nodes.parquet',index=False)
    sizes=model.groupby('domain').size().rename('nodes').reset_index()
    save_table(sizes,out/'tables/domain_sizes')
    save_table(pd.DataFrame([{'dimension':v.shape[1],'selected_nodes':len(nodes),
       'valid_vectors':int(nodes.vector_valid.sum()),'invalid_or_missing_vectors':int((~nodes.vector_valid).sum())}]),out/'tables/vector_coverage')
    result={'reused_prepared_data':True,'domain_nodes':len(model),'boundary_nodes':len(nodes),
            'scope_requirement':'CPC, years, edge source and maximum outcome window must match the prepared run.'}
    save_table(pd.DataFrame([result]),out/'tables/sample_overview')
    return result
