import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.bias import (compute_statistical_parity, compute_disparate_impact,
                      compute_reweighing_weights, compute_intersectional_bias)
from src.explainability import (positive_class_shap, shap_by_group,
                                detect_proxy_variables, explain_single_prediction)
from src.model import find_fair_threshold, mcnemar_test
from src.preprocessing import run_preprocessing, NUMERICAL_COLS

ROOT = Path(__file__).resolve().parents[1]


def test_known_fairness_rates():
    groups = pd.Series(['A'] * 4 + ['B'] * 4)
    pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    assert compute_statistical_parity(pred, groups)['spd'] == .5
    assert compute_disparate_impact(pred, groups)['worst_di'] == .3333
    assert compute_disparate_impact(np.zeros(8), groups)['worst_di'] is None


def test_reweighing_reproduces_independent_marginals():
    groups = pd.Series(['A'] * 4 + ['B'] * 4)
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    weights = compute_reweighing_weights(y, groups)
    assert weights.mean() == pytest.approx(1)
    for group in groups.unique():
        for label in [0, 1]:
            mask = (groups == group) & (y == label)
            assert weights[mask].sum() / weights.sum() == pytest.approx(.25)


def test_shap_labels_and_positive_class_survive_export(tmp_path):
    values = np.arange(24).reshape(4, 3, 2)
    expected = values[:, :, 1]
    np.testing.assert_equal(positive_class_shap(values), expected)
    np.testing.assert_equal(positive_class_shap([values[:, :, 0], expected]), expected)
    frame = shap_by_group(values, None, np.array(['A', 'A', 'B', 'B']), ['age', 'job', 'hours'])
    path = tmp_path / 'shap.csv'
    frame.to_csv(path, index=False)
    assert pd.read_csv(path)['feature'].tolist() == ['age', 'job', 'hours']
    np.testing.assert_equal(frame['A'], expected[:2].mean(axis=0))


def test_single_explanation_modern_shap():
    class Explainer:
        def shap_values(self, X):
            return np.array([[[1, 2], [3, 4]]])
    result = explain_single_prediction(Explainer(), pd.DataFrame([[5, 6]]), ['a', 'b'])
    assert dict(zip(result.feature, result.shap_value)) == {'a': 2, 'b': 4}


def test_proxy_is_spearman_and_empty_is_valid():
    X = pd.DataFrame({'x': [1, 2, 3, 100], 'constant': [1] * 4})
    sensitive = pd.DataFrame({'gender': [0, 0, 1, 1]})
    result = detect_proxy_variables(X, sensitive, X.columns, threshold=0)
    assert result.iloc[0].spearman_corr == round(spearmanr(X.x, sensitive.gender).statistic, 4)
    empty = detect_proxy_variables(X, sensitive, X.columns, threshold=1.1)
    assert empty.empty and 'feature' in empty.columns


def test_tiny_intersections_have_valid_empty_schema():
    result = compute_intersectional_bias([0], [1], pd.DataFrame({'gender_raw': ['A'], 'race_raw': ['B']}))
    assert result.empty and 'positive_rate' in result.columns


def test_threshold_infeasible_is_explicit():
    result = find_fair_threshold(np.array([0, 1]), np.array([.6, .4]), pd.Series(['A', 'B']), accuracy_floor=1)
    assert result['feasible'] is False
    assert result['threshold'] == .5
    with pytest.raises(ValueError):
        find_fair_threshold([], [], pd.Series(dtype=str), target_metric='unsupported')


def test_mcnemar_equal_discordances_and_real_boolean():
    result = mcnemar_test(np.array([0, 1]), np.array([1, 0]), np.array([0, 0]))
    assert result['chi2'] == 0
    assert result['p_value'] == 1
    assert result['significant'] is False
    assert json.loads(json.dumps(result))['significant'] is False


def test_three_way_split_is_disjoint_and_scaler_fits_training_only():
    result = run_preprocessing(ROOT / 'data/adult.data', ROOT / 'data/adult.test', validation_size=.2)
    train, test, _, _, _, _, features, scaler, val, _, _ = result
    assert not set(train.index) & set(test.index)
    assert not set(train.index) & set(val.index)
    assert not set(test.index) & set(val.index)
    assert len(train) + len(test) + len(val) == 45222
    np.testing.assert_allclose(train[NUMERICAL_COLS].mean(), 0, atol=1e-10)
    assert scaler.n_samples_seen_ == len(train)
    assert train.columns.tolist() == test.columns.tolist() == val.columns.tolist() == features


def test_saved_outputs_have_real_metrics_and_labels():
    result = json.loads((ROOT / 'results/mitigation_comparison.json').read_text())
    for scenario in ['original', 'reweighing', 'threshold']:
        assert 0 <= result[scenario]['gender_tpr_gap'] <= 1
        assert 0 <= result[scenario]['gender_fpr_gap'] <= 1
    assert isinstance(result['reweighing']['mcnemar']['significant'], bool)
    names = pd.read_csv(ROOT / 'results/shap_by_group.csv').feature
    assert set(names) == set(pd.read_csv(ROOT / 'results/shap_importance.csv').feature)


def test_dashboard_widgets_render_without_exceptions():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / 'dashboard/streamlit_app.py')).run(timeout=60)
    assert not app.exception
    assert len(app.tabs) == 6
    app.radio[0].set_value('Race').run()
    assert not app.exception
    app.slider[0].set_value(5).run()
    assert not app.exception
    app.slider[0].set_value(14).run()
    assert not app.exception


def test_json_writer_preserves_types_and_rejects_invalid_results(tmp_path, monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline, 'RESULTS_DIR', str(tmp_path))
    pipeline.save_json({'flag': np.bool_(False), 'count': np.int64(3)}, 'result.json')
    path = tmp_path / 'result.json'
    assert json.loads(path.read_text()) == {'flag': False, 'count': 3}
    before = path.read_text()
    with pytest.raises(ValueError):
        pipeline.save_json({'invalid': float('nan')}, 'result.json')
    assert path.read_text() == before


def test_dashboard_missing_results_stops_with_helpful_message(tmp_path):
    from streamlit.testing.v1 import AppTest
    dashboard = tmp_path / 'dashboard'
    dashboard.mkdir()
    path = dashboard / 'streamlit_app.py'
    path.write_text((ROOT / 'dashboard/streamlit_app.py').read_text(encoding='utf-8'), encoding='utf-8')
    app = AppTest.from_file(str(path)).run(timeout=60)
    assert not app.exception
    assert any('Results not found' in error.value for error in app.error)
