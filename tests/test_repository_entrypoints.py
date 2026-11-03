"""Entry-point checks that protect existing results when configuration is wrong."""
import json

import pytest

from prelib.common import ROOT


def test_missing_raw_inputs_do_not_reset_existing_results(tmp_path, monkeypatch):
    import prelib.common as common
    from prelib.pipeline import run_all

    cfg = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    cfg.update(source_root=str(tmp_path / 'missing'), run_name='keep_results')
    path = tmp_path / 'config.local.json'
    path.write_text(json.dumps(cfg), encoding='utf-8')
    sentinel = tmp_path / 'outputs/keep_results/result.csv'
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('existing result', encoding='utf-8')
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError, match='Required input files'):
        run_all(str(path))
    assert sentinel.read_text(encoding='utf-8') == 'existing result'


def test_missing_selected_cache_does_not_reset_existing_results(tmp_path, monkeypatch):
    import prelib.common as common
    from prelib.pipeline import run_all

    cfg = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    cfg.update(source_root=str(tmp_path), prepared_data_root=str(tmp_path / 'missing_cache'), run_name='keep_results')
    path = tmp_path / 'config.local.json'
    path.write_text(json.dumps(cfg), encoding='utf-8')
    sentinel = tmp_path / 'outputs/keep_results/result.csv'
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('existing result', encoding='utf-8')
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError, match='Required input files'):
        run_all(str(path))
    assert sentinel.exists()
