"""
bias.py
-------
Comprehensive fairness metrics:
  - Statistical Parity Difference (SPD)
  - Disparate Impact (DI)
  - Equalized Odds (TPR gap / FPR gap)
  - Calibration gap
  - Intersectional bias analysis (gender X race)
  - Reweighing sample weights for bias mitigation
"""

import numpy as np
import pandas as pd
from itertools import combinations


# ─────────────────────────────────────────────
# Core Fairness Metrics
# ─────────────────────────────────────────────

def compute_statistical_parity(y_pred, sensitive_series) -> dict:
    """
    Statistical Parity Difference (SPD):
    P(ŷ=1 | privileged) - P(ŷ=1 | unprivileged)
    SPD = 0 is ideal; |SPD| < 0.1 is generally acceptable.
    """
    groups = sensitive_series.unique()
    rates = {}
    for g in groups:
        mask = (sensitive_series == g)
        rates[str(g)] = float(y_pred[mask].mean())

    vals = list(rates.values())
    spd = max(vals) - min(vals)

    return {
        "group_rates": rates,
        "spd": round(spd, 4),
        "max_group": max(rates, key=rates.get),
        "min_group": min(rates, key=rates.get),
        "is_fair": abs(spd) < 0.1,
    }


def compute_disparate_impact(y_pred, sensitive_series,
                              privileged_value=None) -> dict:
    """
    Disparate Impact (DI) = P(ŷ=1 | unprivileged) / P(ŷ=1 | privileged)
    DI >= 0.8 is the 80% rule (acceptable range).
    """
    groups = sensitive_series.unique()
    rates = {}
    for g in groups:
        mask = (sensitive_series == g)
        rates[str(g)] = float(y_pred[mask].mean())

    if privileged_value is None:
        priv = max(rates, key=rates.get)
    else:
        priv = str(privileged_value)

    unpriv_rates = {k: v for k, v in rates.items() if k != priv}
    if not unpriv_rates:
        return {"di": 1.0, "is_fair": True}

    di_values = {}
    for k, v in unpriv_rates.items():
        di = v / rates[priv] if rates[priv] > 0 else 0
        di_values[k] = round(di, 4)

    worst_di = min(di_values.values())

    return {
        "group_rates": rates,
        "privileged_group": priv,
        "di_per_group": di_values,
        "worst_di": round(worst_di, 4),
        "is_fair": worst_di >= 0.8,
    }


def compute_equalized_odds(y_test, y_pred, sensitive_series) -> dict:
    """
    Equalized Odds: equal TPR and FPR across groups.
    Returns per-group TPR/FPR and the gap (lower gap = fairer).
    """
    groups = sensitive_series.unique()
    results = {}
    for g in groups:
        mask = (sensitive_series == g)
        yt = np.array(y_test)[mask]
        yp = np.array(y_pred)[mask]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # precision per group
        results[str(g)] = {
            "TPR": round(tpr, 4),
            "FPR": round(fpr, 4),
            "PPV": round(ppv, 4),
            "support": int(mask.sum()),
        }

    tprs = [v["TPR"] for v in results.values()]
    fprs = [v["FPR"] for v in results.values()]
    tpr_gap = round(max(tprs) - min(tprs), 4)
    fpr_gap = round(max(fprs) - min(fprs), 4)

    return {
        "per_group": results,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "is_fair_tpr": tpr_gap < 0.1,
        "is_fair_fpr": fpr_gap < 0.1,
    }


def compute_calibration(y_test, y_proba, sensitive_series) -> dict:
    """
    Calibration: mean predicted probability vs actual positive rate per group.
    A well-calibrated model has small calibration gaps across groups.
    """
    groups = sensitive_series.unique()
    results = {}
    for g in groups:
        mask = sensitive_series == g
        pred_mean = float(np.array(y_proba)[mask].mean())
        actual_rate = float(np.array(y_test)[mask].mean())
        results[str(g)] = {
            "mean_predicted": round(pred_mean, 4),
            "actual_rate": round(actual_rate, 4),
            "gap": round(abs(pred_mean - actual_rate), 4),
        }
    gaps = [v["gap"] for v in results.values()]
    return {
        "per_group": results,
        "max_calibration_gap": round(max(gaps), 4),
        "is_calibrated": max(gaps) < 0.05,
    }


# ─────────────────────────────────────────────
# Intersectional Fairness (gender × race)
# ─────────────────────────────────────────────

def compute_intersectional_bias(y_test, y_pred, sens_df: pd.DataFrame,
                                 col1: str = "gender_raw",
                                 col2: str = "race_raw") -> pd.DataFrame:
    """
    Compute positive prediction rate, TPR, and FPR for every
    combination of col1 × col2 (e.g., Male-White, Female-Black …).
    Returns a DataFrame sorted by positive_rate ascending.
    """
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)

    records = []
    intersect = sens_df[col1].astype(str) + " + " + sens_df[col2].astype(str)
    for grp in sorted(intersect.unique()):
        mask = (intersect == grp).values
        n = mask.sum()
        if n < 30:          # skip tiny cells
            continue
        yt = y_test_arr[mask]
        yp = y_pred_arr[mask]
        pos_rate = float(yp.mean())
        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        records.append({
            "group": grp,
            "n": n,
            "positive_rate": round(pos_rate, 4),
            "TPR": round(tpr, 4),
            "FPR": round(fpr, 4),
        })

    df = pd.DataFrame(records).sort_values("positive_rate")
    return df


# ─────────────────────────────────────────────
# Bias Mitigation: Reweighing
# ─────────────────────────────────────────────

def compute_reweighing_weights(y_train, sensitive_series) -> np.ndarray:
    """
    Reweighing (Kamiran & Calders, 2012):
    Assigns sample weights so that every (group, label) cell has
    equal expected weight, removing statistical dependence between
    the sensitive attribute and the target.

    w(group, label) = P(group) * P(label) / P(group, label)
    """
    y = np.array(y_train)
    s = np.array(sensitive_series)
    n = len(y)

    groups = np.unique(s)
    labels = np.unique(y)

    weights = np.ones(n)
    for g in groups:
        for l in labels:
            mask = (s == g) & (y == l)
            if mask.sum() == 0:
                continue
            p_g = mask_to_prob(s == g, n)
            p_l = mask_to_prob(y == l, n)
            p_gl = mask_to_prob(mask, n)
            w = (p_g * p_l) / p_gl if p_gl > 0 else 1.0
            weights[mask] = w

    # Normalise so sum equals n
    weights = weights / weights.mean()
    return weights


def mask_to_prob(mask, n):
    return mask.sum() / n


# ─────────────────────────────────────────────
# Full Bias Report
# ─────────────────────────────────────────────

def full_bias_report(y_test, y_pred, y_proba, sens_test: pd.DataFrame,
                     sensitive_col: str = "gender_raw",
                     sensitive_binary: str = "gender") -> dict:
    """
    Compute all fairness metrics for one sensitive attribute.
    Returns a nested dict ready for display.
    """
    s_raw   = sens_test[sensitive_col].reset_index(drop=True)
    s_bin   = sens_test[sensitive_binary].reset_index(drop=True)
    yt      = np.array(y_test)
    yp      = np.array(y_pred)
    yproba  = np.array(y_proba)

    return {
        "statistical_parity": compute_statistical_parity(yp, s_raw),
        "disparate_impact":   compute_disparate_impact(yp, s_raw),
        "equalized_odds":     compute_equalized_odds(yt, yp, s_raw),
        "calibration":        compute_calibration(yt, yproba, s_raw),
    }


if __name__ == "__main__":
    # Quick sanity check with synthetic data
    np.random.seed(42)
    n = 1000
    gender = np.random.choice(["Male", "Female"], n)
    y_true = np.random.randint(0, 2, n)
    # Biased predictions: Males get positive more often
    y_pred = np.where(gender == "Male",
                      np.random.choice([0, 1], n, p=[0.4, 0.6]),
                      np.random.choice([0, 1], n, p=[0.7, 0.3]))

    s = pd.Series(gender)
    sp = compute_statistical_parity(y_pred, s)
    di = compute_disparate_impact(y_pred, s)
    eo = compute_equalized_odds(y_true, y_pred, s)

    print("Statistical Parity:", sp)
    print("Disparate Impact:",   di)
    print("Equalized Odds:",     eo)
