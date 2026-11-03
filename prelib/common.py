from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def json_default(x):
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, (Path, pd.Timestamp)):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=json_default, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def load_config(path=None):
    if path is None:
        path = os.environ.get("PRE_CONFIG")
        active = ROOT / "outputs/active_config.json"
        if not path and active.exists(): path = json.loads(active.read_text(encoding="utf-8"))["config_file"]
    path = Path(path or "config.json")
    if not path.is_absolute():
        path = ROOT / path
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", cfg["run_name"]):
        raise ValueError("run_name须以字母开头，仅使用字母、数字、下划线和短横线。")
    if cfg["edge_source"] not in {"all", "applicant", "examiner"}:
        raise ValueError("edge_source must be all/applicant/examiner")
    if not (cfg["network_start_year"] < cfg["community_freeze_year"]
            <= cfg["growth_start_year"] <= cfg["train_end_year"]
            < cfg["validation_end_year"] < cfg["network_end_year"]):
        raise ValueError("检查历史、训练、验证和测试年份的先后关系。")
    if cfg["focal_start_year"] < cfg["community_freeze_year"]:
        raise ValueError("焦点队列不得早于固定社群的冻结年份。")
    for k in ['max_nodes_per_domain','max_total_model_nodes','max_boundary_nodes','max_kept_edges',
              'batch_rows','negative_controls','max_choice_events_per_split','null_repetitions',
              'simulation_repetitions','slq_probes','slq_steps','robustness_repetitions']:
        if int(cfg[k])<1:raise ValueError(f'{k} must be positive')
    if not 0.5<cfg['community_parent_min_purity']<=1:raise ValueError('Invalid parent purity')
    if not 0<cfg['min_semantic_coverage']<=1 or not 0<cfg['min_community_coverage']<=1:raise ValueError('Invalid coverage')
    if pd.Timestamp(cfg['simulation_start_date'])>pd.Timestamp(cfg['simulation_end_date']):raise ValueError('Invalid simulation dates')
    ids=[d['id'] for d in cfg['domains']]
    if not ids or len(ids)!=len(set(ids)) or any(not re.fullmatch(r'[a-z][a-z0-9_]*',x) for x in ids):raise ValueError('Invalid domain IDs')
    if cfg['cpc_match_mode'] not in {'primary','any'} or cfg['overlap_policy'] not in {'error','priority'}:raise ValueError('Invalid CPC selection')
    if cfg['null_min_repetitions']>cfg['null_repetitions']:raise ValueError('Insufficient null draws')
    if not 1<=cfg['primary_outcome_years']<=cfg['outcome_years']:
        raise ValueError('The forward-edge collection window must cover the primary outcome window.')
    if cfg['physical_age_tau_years']<=0 or cfg['max_reference_age_years']<=0:
        raise ValueError('Age scales must be positive.')
    if not 0<=cfg['minimum_null_acceptance']<1:raise ValueError('Invalid null acceptance threshold.')
    if cfg['spectral_tau_min']<=0 or cfg['spectral_tau_min']>=cfg['spectral_tau_max']:
        raise ValueError('Invalid diffusion-scale range.')
    if not set(cfg.get('null_models',['Topology'])) <= {'Topology','Semantics','Both'}:
        raise ValueError('Unknown null ensemble.')
    if not cfg.get('null_models',['Topology']):raise ValueError('Choose at least one null ensemble.')
    if 'Topology' not in cfg.get('null_models',['Topology']):raise ValueError('The primary figures require the Topology ensemble.')
    if cfg.get('spectral_coarse_probes',1)>cfg['slq_probes'] or cfg.get('spectral_coarse_steps',1)>cfg['slq_steps']:
        raise ValueError('The spectral convergence reference must use no more probes/steps than the final estimate.')
    if cfg['slq_probes']<2 or cfg.get('spectral_coarse_probes',2)<2:
        raise ValueError('At least two probes are needed for probe uncertainty.')
    if cfg['spectral_tau_points']<2 or cfg['percolation_points']<3:
        raise ValueError('Use at least two diffusion scales and three connectivity fractions.')
    if min(cfg['physical_semantic_cost'],cfg['physical_cross_cost'])<0:
        raise ValueError('The nested cost mechanisms require nonnegative semantic/community costs.')
    return cfg


def set_active_config(path):
    write_json(ROOT / "outputs/active_config.json", {"config_file": str(path)})


def run_dir(cfg, initialize=True):
    out = ROOT / "outputs" / cfg["run_name"]
    if cfg.get("_domain_id"):
        out = out / "domains" / cfg["_domain_id"]
    if initialize:
        out.mkdir(parents=True, exist_ok=True)
        for sub in ["data", "tables", "figures", "models"]:
            (out / sub).mkdir(exist_ok=True)
    return out


def initialize_run(cfg):
    """Start at 00: replace only this project's named result directory."""
    if cfg.get("_domain_id") or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*",cfg["run_name"]):
        raise ValueError("只能从共享项目配置开始新计算。")
    parent=(ROOT/"outputs").resolve()
    out=ROOT/"outputs"/cfg["run_name"]
    if out.is_symlink() or out.resolve().parent != parent:
        raise ValueError("结果路径必须直接位于当前项目outputs目录中。")
    if cfg.get('prepared_data_root'):
        source=Path(cfg['prepared_data_root']).expanduser()
        if not source.is_absolute():source=ROOT/source
        if source.resolve()==out.resolve() or out.resolve() in source.resolve().parents:
            raise ValueError('Prepared input cannot be located inside the result directory being reset.')
    if cfg.get('reuse_prepared_if_available') and cfg.get('reuse_prepared_run')==cfg['run_name']:
        raise ValueError('A new run must not overwrite its prepared input run.')
    if out.exists():shutil.rmtree(out)
    return run_dir(cfg)


def shared_dir(cfg, initialize=True):
    return run_dir({k:v for k,v in cfg.items() if not k.startswith("_")}, initialize)


def domain_config(cfg, domain_id):
    if domain_id not in [d["id"] for d in cfg["domains"]]:
        raise ValueError(domain_id)
    return {**cfg, "_domain_id": domain_id}


def cutoff_date(cfg):
    if "run_name" not in cfg:
        return pd.Timestamp(cfg["citation_observation_end"])
    path = shared_dir(cfg) / "data/observation_window.json"
    if path.exists():
        return pd.Timestamp(json.loads(path.read_text(encoding="utf-8"))["effective_observation_end"])
    return pd.Timestamp(cfg["citation_observation_end"])


def save_table(df, path, latex=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    if latex:
        df.to_latex(path.with_suffix(".tex"), index=False, escape=True,
                    float_format=lambda v: f"{v:.4g}")
    return df


def environment_report(cfg):
    versions = {}
    for p in ["numpy", "pandas", "pyarrow", "scipy", "networkx",
              "matplotlib", "nbformat", "nbclient"]:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = "not installed"
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": versions,
              "synthetic_data": bool(cfg.get("synthetic_data", False))}
    return report


def require_outputs(cfg, *names):
    out = run_dir(cfg)
    for name in names:
        if not (out / name).exists():
            raise FileNotFoundError(f"缺少 {name}；请先运行前序 Notebook。")
    return out


def normalize_code(values):
    s = values.astype("string").str.upper().str.replace(r"\s+", "", regex=True)
    parts = s.str.extract(r"^([A-HY]\d{2}[A-Z])0*(\d+)(/.*)?$")
    good = parts[0].notna()
    main = parts[1].str.lstrip("0").replace("", "0")
    s.loc[good] = parts[0][good] + main[good] + parts[2][good].fillna("")
    return s


def main_group(values):
    # G06N 003/08 and G06N3/08 both become G06N3.
    code = normalize_code(values)
    parts = code.str.extract(r"^([A-HY]\d{2}[A-Z])0*(\d+)/")
    return parts[0] + parts[1].str.lstrip("0").replace("", "0")
