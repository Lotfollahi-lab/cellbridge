import json

import pandas as pd
import pytest

from cellbridge.inference.calibration import report as reporting
from cellbridge.inference.calibration.metrics import METRICS
from cellbridge.inference.calibration.run import METHODS


@pytest.fixture
def hest_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, '_plot', lambda *args: None)
    monkeypatch.setattr(reporting, '_write_markdown', lambda *args: None)
    cases = []
    for m, method in enumerate(METHODS):
        for section in range(14):
            sample = f'section{section:02d}'
            context = dict(method=method, dataset='hest', task='all', sample_id=sample)
            key = f'{method}/{sample}'
            cases.append(dict(context, key=key))
            directory = tmp_path / 'runs' / key
            directory.mkdir(parents=True)
            audit = dict(context, train_ids_sha256=sample, validation_ids_sha256=sample,
                         test_identity_sha256=sample, classes_sha256='classes')
            (directory / 'audit.json').write_text(json.dumps(audit))
            rows = [dict(variant=variant, **{metric: section / 100 + m / 100 for metric in METRICS})
                    for variant in ['raw', 'temperature_scaled']]
            pd.DataFrame(rows).to_csv(directory / 'metrics.csv', index=False)
            pd.DataFrame(columns=['variant', 'bootstrap', *METRICS]).to_csv(directory / 'bootstrap.csv', index=False)
            pd.DataFrame(columns=['n_bins', 'n']).to_csv(directory / 'reliability.csv', index=False)
    (tmp_path / 'cases.json').write_text(json.dumps(cases))
    return tmp_path


def test_hest_equal_section_mean_and_paired_intervals(hest_audit):
    reporting.report(hest_audit)
    summary = pd.read_csv(hest_audit / 'summary.csv')
    ours = summary.query("method == 'CellBridge' and variant == 'raw' and metric == 'ece_15'").iloc[0]
    assert ours.estimate == pytest.approx(.065)
    differences = pd.read_csv(hest_audit / 'paired_differences.csv')
    difference = differences.query("comparison == 'CellBridge - UNI2-h' and variant == 'raw' and metric == 'ece_15'").iloc[0]
    assert difference.estimate == pytest.approx(-.01)
    assert difference.ci_2_5 == pytest.approx(-.01)
    assert difference.ci_97_5 == pytest.approx(-.01)
    scaling = differences.query("method == 'CellBridge' and comparison_type == 'temperature_scaling' and metric == 'ece_15'").iloc[0]
    assert scaling.estimate == 0
    assert scaling.ci_2_5 == 0
    assert scaling.ci_97_5 == 0


def test_report_rejects_different_test_populations(hest_audit):
    path = hest_audit / 'runs' / 'CellBridge' / 'section00' / 'audit.json'
    audit = json.loads(path.read_text())
    audit['test_identity_sha256'] = 'different'
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='test_identity_sha256'):
        reporting.report(hest_audit)


def test_report_rejects_incomplete_comparison(hest_audit):
    (hest_audit / 'runs' / 'CellBridge' / 'section00' / 'audit.json').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        reporting.report(hest_audit)
