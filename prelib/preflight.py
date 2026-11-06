"""Nonmutating checks before resetting a named analysis output directory."""
from __future__ import annotations

from pathlib import Path

from .common import ROOT, input_paths


def required_input_files(cfg):
    """Return the files used by raw ingestion or the selected prepared cache."""
    paths = input_paths(cfg)
    selected = cfg.get('prepared_data_root')
    candidate = Path(selected).expanduser() if selected else ROOT / 'outputs' / cfg.get('reuse_prepared_run', '')
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
        provenance = [candidate / 'data/prepared_scope.json',
                      candidate / 'tables/sample_overview.csv']
        present = [path for path in provenance if path.is_file()]
        required.extend(present or provenance[:1])
    else:
        required.extend(paths[key] for key in ['edges', 'index', 'vectors'])
    return required


def check_input_files(cfg):
    """Require each input before starting the configured workflow."""
    required = required_input_files(cfg)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'Required input files are missing; set --source-root or use config.local.json.\n'
            + '\n'.join(missing)
        )
