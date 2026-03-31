"""
pipeline.py
-----------
End-to-end pipeline: preprocess → train → evaluate → bias → explain → mitigate.
Saves all results to results/ as JSON + CSV for Streamlit dashboard.
"""

import os
import sys
import json
import urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing  import run_preprocessing
from src.model          import (
    build_logistic_regression, build_random_forest,
    train_model, evaluate_model, mcnemar_test,
    find_fair_threshold
)
from src.bias import (
    full_bias_report,
    compute_intersectional_bias,
    compute_reweighing_weights
)
from src.explainability import (
    get_shap_explainer,
    compute_shap_values,
    global_feature_importance,
    shap_by_group,
    detect_proxy_variables
)

# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TRAIN_PATH = os.path.join(DATA_DIR, "adult.data")
TEST_PATH  = os.path.join(DATA_DIR, "adult.test")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_URL_TRAIN = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
DATA_URL_TEST  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"


# ─────────────────────────────────────────────
def ensure_data():
    for path, url in [(TRAIN_PATH, DATA_URL_TRAIN), (TEST_PATH, DATA_URL_TEST)]:
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)} ...")
            urllib.request.urlretrieve(url, path)


# ─────────────────────────────────────────────
def convert(o):
   

    if isinstance(o, (np.integer, np.int64)):
        return int(o)

    if isinstance(o, (np.floating, np.float64)):
        return float(o)

    if isinstance(o, np.ndarray):
        return o.tolist()

    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")

    if isinstance(o, pd.Series):
        return o.tolist()

    # sklearn / complex objects fallback
    if hasattr(o, "__dict__"):
        return str(o)

    return str(o)


# ─────────────────────────────────────────────
def save_json(obj, filename):
    path = os.path.join(RESULTS_DIR, filename)

    try:
        # deep-safe conversion (kills circular references)
        safe_obj = json.loads(json.dumps(obj, default=convert))
    except Exception:
        safe_obj = str(obj)

    with open(path, "w") as f:
        json.dump(safe_obj, f, indent=2)

    print(f"  Saved {filename}")


# ─────────────────────────────────────────────
def run():
    print("\n[1/6] Preprocessing data …")
    ensure_data()

    (X_train, X_test, y_train, y_test,
     sens_train, sens_test, feature_names, scaler) = run_preprocessing(
        TRAIN_PATH, TEST_PATH
    )

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(feature_names)}")

    # ───────────────────────── MODELS ─────────────────────────
    print("\n[2/6] Training models …")

    lr = train_model(build_logistic_regression(), X_train, y_train)
    rf = train_model(build_random_forest(), X_train, y_train)

    lr_metrics = evaluate_model(lr, X_test, y_test)
    rf_metrics = evaluate_model(rf, X_test, y_test)

    model_comparison = {
        "LogisticRegression": {
            k: v for k, v in lr_metrics.items()
            if k not in ("y_pred", "y_proba", "confusion_matrix")
        },
        "RandomForest": {
            k: v for k, v in rf_metrics.items()
            if k not in ("y_pred", "y_proba", "confusion_matrix")
        },
    }

    model_comparison["LogisticRegression"]["confusion_matrix"] = lr_metrics["confusion_matrix"].tolist()
    model_comparison["RandomForest"]["confusion_matrix"] = rf_metrics["confusion_matrix"].tolist()

    save_json(model_comparison, "model_metrics.json")

    # ───────────────────────── BIAS ─────────────────────────
    print("\n[3/6] Bias detection …")

    sens_test = sens_test.reset_index(drop=True)
    y_test    = y_test.reset_index(drop=True)

    bias_gender = full_bias_report(
        y_test, lr_metrics["y_pred"], lr_metrics["y_proba"],
        sens_test, sensitive_col="gender_raw", sensitive_binary="gender"
    )

    bias_race = full_bias_report(
        y_test, lr_metrics["y_pred"], lr_metrics["y_proba"],
        sens_test, sensitive_col="race_raw", sensitive_binary="race_binary"
    )

    intersect_df = compute_intersectional_bias(
        y_test, lr_metrics["y_pred"], sens_test,
        col1="gender_raw", col2="race_raw"
    )

    save_json({"gender": bias_gender, "race": bias_race}, "bias_before.json")
    intersect_df.to_csv(os.path.join(RESULTS_DIR, "intersectional_bias.csv"), index=False)

    # ───────────────────────── EXPLAINABILITY ─────────────────────────
    print("\n[4/6] SHAP explainability …")

    explainer = get_shap_explainer(rf, X_train, model_type="tree")
    shap_vals, X_shap = compute_shap_values(explainer, X_test, model_type="tree")

    importance_df = global_feature_importance(shap_vals, feature_names)
    importance_df.to_csv(os.path.join(RESULTS_DIR, "shap_importance.csv"), index=False)

    sens_shap = sens_test.iloc[:len(X_shap)]["gender_raw"]

    group_shap_df = shap_by_group(shap_vals, X_shap, sens_shap, feature_names)
    group_shap_df.to_csv(os.path.join(RESULTS_DIR, "shap_by_group.csv"), index=False)

    proxy_df = detect_proxy_variables(X_test, sens_test, feature_names, threshold=0.2)
    proxy_df.to_csv(os.path.join(RESULTS_DIR, "proxy_variables.csv"), index=False)

    # ───────────────────────── MITIGATION ─────────────────────────
    print("\n[5/6] Bias mitigation …")

    weights = compute_reweighing_weights(y_train, sens_train["gender_raw"])

    lr_rw = train_model(build_logistic_regression(), X_train, y_train, sample_weight=weights)
    lr_rw_metrics = evaluate_model(lr_rw, X_test, y_test)

    best_thresh = find_fair_threshold(
        y_test, lr_metrics["y_proba"], sens_test["gender_raw"]
    )

    lr_thresh_metrics = evaluate_model(
        lr, X_test, y_test, threshold=best_thresh["threshold"]
    )

    mc_rw = mcnemar_test(lr_metrics["y_pred"], lr_rw_metrics["y_pred"], y_test.values)
    mc_th = mcnemar_test(lr_metrics["y_pred"], lr_thresh_metrics["y_pred"], y_test.values)

    bias_after_rw = full_bias_report(
        y_test, lr_rw_metrics["y_pred"], lr_rw_metrics["y_proba"],
        sens_test, "gender_raw", "gender"
    )

    bias_after_th = full_bias_report(
        y_test, lr_thresh_metrics["y_pred"], lr_thresh_metrics["y_proba"],
        sens_test, "gender_raw", "gender"
    )

    mitigation_results = {
        "original": {
            "accuracy": float(lr_metrics["accuracy"]),
            "f1": float(lr_metrics["f1"]),
            "roc_auc": float(lr_metrics["roc_auc"]),
            "gender_spd": float(bias_gender["statistical_parity"]["spd"]),
        },
        "reweighing": {
            "accuracy": float(lr_rw_metrics["accuracy"]),
            "f1": float(lr_rw_metrics["f1"]),
            "roc_auc": float(lr_rw_metrics["roc_auc"]),
            "gender_spd": float(bias_after_rw["statistical_parity"]["spd"]),
            "mcnemar": mc_rw,
        },
        "threshold": {
            "accuracy": float(lr_thresh_metrics["accuracy"]),
            "f1": float(lr_thresh_metrics["f1"]),
            "roc_auc": float(lr_thresh_metrics["roc_auc"]),
            "threshold_used": float(best_thresh["threshold"]),
            "gender_spd": float(bias_after_th["statistical_parity"]["spd"]),
            "mcnemar": mc_th,
        }
    }

    save_json(mitigation_results, "mitigation_comparison.json")

    print("\n[6/6] Pipeline complete ✓")
    print(f"Results saved in: {RESULTS_DIR}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    run()