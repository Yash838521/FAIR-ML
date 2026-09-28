"""
explainability.py
-----------------
SHAP-based model explainability:
  - Global feature importance (mean |SHAP|)
  - Per-group SHAP comparison to detect proxy bias
  - Individual prediction explanation
  - Proxy variable detection: features correlated with sensitive attributes
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr


def positive_class_shap(values):
    """Normalize legacy list and modern (samples, features, classes) outputs."""
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")
    return values


# ─────────────────────────────────────────────
# SHAP Explainer Factory
# ─────────────────────────────────────────────

def get_shap_explainer(model, X_background, model_type: str = "tree"):
    """
    Returns the appropriate SHAP explainer for the model type.
    model_type: 'tree' for RandomForest, 'linear' for LogisticRegression.
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model, data=X_background[:200])
    else:
        background = shap.sample(X_background, 100)
        f = lambda x: model.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(f, background)
    return explainer


def compute_shap_values(explainer, X, max_samples: int = 500,
                         model_type: str = "tree"):

    X_sample = X.iloc[:max_samples] if len(X) > max_samples else X

    #  Ensure numeric stability
    X_sample = X_sample.astype(np.float64)

    if model_type == "tree":
        shap_vals = explainer.shap_values(
            X_sample,
            check_additivity=False  
        )

        # binary classification fix
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

    else:
        shap_vals = explainer.shap_values(X_sample)

    return positive_class_shap(shap_vals), X_sample

# ─────────────────────────────────────────────
# Global Feature Importance
# ─────────────────────────────────────────────

def global_feature_importance(shap_values: np.ndarray,
                               feature_names: list) -> pd.DataFrame:
    """
    Mean absolute SHAP value per feature (robust version)
    """

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # binary classification fix

    # ensure 2D
    shap_values = np.atleast_2d(shap_values)

    mean_abs = np.mean(np.abs(shap_values), axis=0)

    mean_abs = np.ravel(mean_abs)  # force 1D

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    })

    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df


# ─────────────────────────────────────────────
# Per-Group SHAP Analysis (Proxy Bias Detection)
# ─────────────────────────────────────────────

def shap_by_group(shap_vals, X, sensitive_attr, feature_names):
    """
    Computes mean absolute SHAP per group.
    Handles binary/multiclass SHAP outputs safely.
    """

    shap_vals = positive_class_shap(shap_vals)
    results = {}

    groups = np.unique(sensitive_attr)

    for g in groups:
        idx = np.where(sensitive_attr == g)[0]

        group_shap = shap_vals[idx]

        # Case 1: (samples, features)
        if len(group_shap.shape) == 2:
            mean_vals = np.mean(np.abs(group_shap), axis=0)

        # Case 2: (samples, features, classes)
        elif len(group_shap.shape) == 3:
            # collapse classes first
            mean_vals = np.mean(np.abs(group_shap), axis=(0, 2))

        else:
            raise ValueError(f"Unexpected SHAP shape: {group_shap.shape}")

        results[g] = mean_vals

    # Convert safely into DataFrame
    df = pd.DataFrame(results, index=feature_names)

    df.index.name = "feature"
    return df.reset_index()
# ─────────────────────────────────────────────
# Proxy Variable Detection
# ─────────────────────────────────────────────

def detect_proxy_variables(X: pd.DataFrame,
                            sensitive_df: pd.DataFrame,
                            feature_names: list,
                            threshold: float = 0.3) -> pd.DataFrame:
    """
    Detect features that are highly correlated with sensitive attributes.
    Uses Spearman rank correlation.
    Returns a table of (feature, sensitive_attr, correlation) sorted by |corr|.
    """
    records = []
    for s_col in ["gender", "race_binary"]:
        if s_col not in sensitive_df.columns:
            continue
        s = sensitive_df[s_col].values
        for feat in feature_names:
            if feat in sensitive_df.columns:
                continue
            x = X[feat].values
            if np.unique(x).size < 2 or np.unique(s).size < 2:
                continue
            corr = spearmanr(x, s).statistic
            if abs(corr) >= threshold:
                records.append({
                    "feature": feat,
                    "sensitive_attr": s_col,
                    "spearman_corr": round(corr, 4),
                    "abs_corr": round(abs(corr), 4),
                })

    df = pd.DataFrame(records, columns=["feature", "sensitive_attr", "spearman_corr", "abs_corr"]).sort_values("abs_corr", ascending=False)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# Individual Explanation
# ─────────────────────────────────────────────

def explain_single_prediction(explainer, X_single: pd.DataFrame,
                               feature_names: list,
                               model_type: str = "tree") -> pd.DataFrame:
    """
    SHAP explanation for a single instance.
    Returns a DataFrame of (feature, value, shap_value) sorted by |shap|.
    """
    if model_type == "tree":
        sv = explainer.shap_values(X_single)
        if isinstance(sv, list):
            sv = sv[1]
        sv = positive_class_shap(sv).flatten()
    else:
        sv = explainer.shap_values(X_single).flatten()

    df = pd.DataFrame({
        "feature": feature_names,
        "value": X_single.values.flatten(),
        "shap_value": sv,
    })
    df["abs_shap"] = df["shap_value"].abs()
    return df.sort_values("abs_shap", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# Matplotlib Plot Helpers (for Streamlit)
# ─────────────────────────────────────────────

def plot_global_importance(importance_df: pd.DataFrame,
                           top_n: int = 15,
                           title: str = "Global Feature Importance (SHAP)") -> plt.Figure:
    df = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["feature"][::-1], df["mean_abs_shap"][::-1], color="#4F8EF7")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_shap_group_comparison(group_df: pd.DataFrame,
                                top_n: int = 10) -> plt.Figure:
    df = group_df.head(top_n)
    group_cols = [c for c in df.columns if c not in ("feature", "abs_diff")]
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4F8EF7", "#F76C6C", "#6CF7A8", "#F7C94F"]
    for i, col in enumerate(group_cols):
        ax.bar(x + i * width, df[col], width, label=col, color=colors[i % len(colors)])

    ax.set_xticks(x + width * (len(group_cols) - 1) / 2)
    ax.set_xticklabels(df["feature"], rotation=30, ha="right")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title("Per-Group SHAP Comparison (Proxy Bias Detection)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig
