"""Nonmutating checks before resetting a named analysis output directory."""
from __future__ import annotations

from pathlib import Path

from .common import ROOT
from .prepare import input_paths


def check_input_files(cfg):
    """Require the files used by raw ingestion or an explicitly selected cache."""
    paths = input_paths(cfg)
    selected = cfg.get('prepared_data_root')
    candidate = Path(selected) if selected else ROOT / 'outputs' / cfg.get('reuse_prepared_run', '')
    if selected and not candidate.is_absolute():
        candidate = ROOT / candidate
    reuse = bool(selected) or (
        cfg.get('reuse_prepared_if_available', False)
        and (candidate / 'data/nodes.parquet').is_file()
    )
    required = [paths['dates'], paths['fields']]
    if cfg['cpc_match_mode'] != 'primary':
        required.append(paths['cpc'])
    if reuse:
        required.extend(candidate / 'data' / name for name in [
            'nodes.parquet', 'reference_edges.parquet', 'forward_edges.parquet',
            'internal_edges.parquet', 'vectors.npy', 'observation_window.json',
        ])
    else:
        required.extend(paths[key] for key in ['edges', 'index', 'vectors'])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'Required input files are missing; set --source-root or use config.local.json.\n'
            + '\n'.join(missing)
        )
