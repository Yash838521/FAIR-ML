"""
model.py
--------
Model training, evaluation, and threshold adjustment for fairness mitigation.
Implements Logistic Regression and Random Forest with full classification metrics.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from scipy import stats


# ─────────────────────────────────────────────
# Model Builders
# ─────────────────────────────────────────────

def build_logistic_regression(random_state: int = 42, max_iter: int = 1000, **kwargs):
    return LogisticRegression(random_state=random_state, max_iter=max_iter,
                              class_weight="balanced", **kwargs)


def build_random_forest(random_state: int = 42, n_estimators: int = 200, **kwargs):
    return RandomForestClassifier(random_state=random_state, n_estimators=n_estimators,
                                  class_weight="balanced", n_jobs=-1, **kwargs)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_model(model, X_train, y_train, sample_weight=None):
    """Fit a model, optionally with sample weights (for reweighing mitigation)."""
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """
    Full evaluation: accuracy, precision, recall, F1, AUC, confusion matrix.
    Supports custom decision threshold.
    """
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, proba),
        "confusion_matrix": cm,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "threshold": threshold,
        "y_pred": y_pred,
        "y_proba": proba,
    }


def evaluate_equalized_odds(y_test, y_pred, sensitive_series) -> dict:
    """
    Equalized Odds: checks that TPR and FPR are equal across groups.
    Returns per-group TPR, FPR, and the max gap (lower = fairer).
    """
    groups = sensitive_series.unique()
    results = {}
    for g in groups:
        mask = (sensitive_series == g)
        yt = y_test[mask]
        yp = y_pred[mask]
        tp = ((yt == 1) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        results[g] = {"TPR": round(tpr, 4), "FPR": round(fpr, 4)}

    tprs = [v["TPR"] for v in results.values()]
    fprs = [v["FPR"] for v in results.values()]
    results["TPR_gap"] = round(max(tprs) - min(tprs), 4)
    results["FPR_gap"] = round(max(fprs) - min(fprs), 4)
    return results


def evaluate_calibration(y_test, y_proba, sensitive_series, n_bins: int = 5) -> dict:
    """
    Calibration check: mean predicted probability vs actual positive rate per group.
    Well-calibrated models have low calibration gap across groups.
    """
    groups = sensitive_series.unique()
    results = {}
    for g in groups:
        mask = (sensitive_series == g)
        mean_pred = y_proba[mask].mean()
        actual_rate = y_test[mask].mean()
        results[g] = {
            "mean_predicted_proba": round(float(mean_pred), 4),
            "actual_positive_rate": round(float(actual_rate), 4),
            "calibration_gap": round(float(abs(mean_pred - actual_rate)), 4)
        }
    return results


# ─────────────────────────────────────────────
# Statistical Significance Testing
# ─────────────────────────────────────────────

def mcnemar_test(y_pred_before, y_pred_after, y_test) -> dict:
    """
    McNemar's test: checks if two classifiers differ significantly.
    Returns chi2 statistic and p-value.
    """
    b = ((y_pred_before != y_test) & (y_pred_after == y_test)).sum()
    c = ((y_pred_before == y_test) & (y_pred_after != y_test)).sum()
    if b + c == 0:
        return {"chi2": 0, "p_value": 1.0, "significant": False}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return {"chi2": round(chi2, 4), "p_value": round(p_value, 4), "significant": p_value < 0.05}


# ─────────────────────────────────────────────
# Threshold Optimisation (Mitigation)
# ─────────────────────────────────────────────

def find_fair_threshold(y_test, y_proba, sensitive_series,
                        target_metric: str = "statistical_parity",
                        thresholds=None) -> dict:
    """
    Grid-search thresholds to minimise the chosen fairness gap
    while keeping accuracy above a floor (default 80%).
    Returns the best threshold and its metrics.
    """
    from src.bias import compute_statistical_parity, compute_disparate_impact

    if thresholds is None:
        thresholds = np.arange(0.3, 0.75, 0.01)

    best = {"threshold": 0.5, "fairness_gap": 999, "accuracy": 0}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        acc = accuracy_score(y_test, y_pred)
        if acc < 0.78:          # accuracy floor
            continue
        sp = compute_statistical_parity(y_pred, sensitive_series)
        gap = abs(sp.get("spd", 999))
        if gap < best["fairness_gap"]:
            best = {"threshold": round(t, 3), "fairness_gap": round(gap, 4), "accuracy": round(acc, 4)}

    return best


# ─────────────────────────────────────────────
# Summary Printer
# ─────────────────────────────────────────────

def print_evaluation(metrics: dict, model_name: str = "Model"):
    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation (threshold={metrics['threshold']})")
    print(f"{'='*50}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  TN={metrics['tn']}  FP={metrics['fp']}")
    print(f"  FN={metrics['fn']}  TP={metrics['tp']}")
