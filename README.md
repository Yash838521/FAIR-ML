# ⚖️ Fairness-Aware Machine Learning System

> Predict income. Detect bias. Explain decisions. Mitigate harm.

A production-grade ML pipeline on the **Adult Income dataset** (UCI) that goes beyond accuracy to measure, explain, and reduce algorithmic bias — with a full interactive Streamlit dashboard.

---

## 🗂 Project Structure

```
FAIRML/
├── data/                        # Auto-downloaded Adult Income dataset
├── src/
│   ├── preprocessing.py         # Cleaning, encoding, scaling, sensitive feature handling
│   ├── model.py                 # LR + RF training, evaluation, threshold optimisation
│   ├── bias.py                  # SPD, DI, Equalized Odds, Calibration, Intersectionality, Reweighing
│   └── explainability.py        # SHAP global/per-group, proxy variable detection
├── dashboard/
│   └── streamlit_app.py         # Interactive 6-tab dashboard
├── results/                     # Auto-generated JSON + CSV outputs
├── pipeline.py                  # End-to-end orchestration script
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline (downloads data, trains models, saves all results)
```bash
python pipeline.py
```

### 3. Launch the dashboard
```bash
streamlit run dashboard/streamlit_app.py
```

Open your browser at **http://localhost:8501**

---

## 🔬 What the Pipeline Does

### Step 1 — Preprocessing (`src/preprocessing.py`)
- Downloads the UCI Adult Income dataset automatically
- Removes `?` entries (missing values)
- Label-encodes categorical features (workclass, occupation, education, etc.)
- StandardScales numerical features (age, hours-per-week, capital-gain, etc.)
- Preserves `gender` and `race` as both raw strings and binary encodings for fairness analysis
- Stratified 80/20 train/test split

### Step 2 — Model Training (`src/model.py`)
Two models are trained:

| Model | Why |
|---|---|
| **Logistic Regression** | Primary model — interpretable, linear, suitable for SHAP |
| **Random Forest** | Comparison — higher accuracy, suitable for TreeSHAP |

Both use `class_weight="balanced"` to handle the income class imbalance (~75% ≤50K).

Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.

### Step 3 — Bias Detection (`src/bias.py`)

Four fairness metrics are computed for **gender** and **race** separately:

| Metric | Definition | Fair threshold |
|---|---|---|
| **Statistical Parity Difference (SPD)** | P(ŷ=1\|group_A) − P(ŷ=1\|group_B) | \|SPD\| < 0.10 |
| **Disparate Impact (DI)** | P(ŷ=1\|unprivileged) / P(ŷ=1\|privileged) | DI ≥ 0.80 |
| **Equalized Odds** | TPR gap + FPR gap across groups | Gap < 0.10 |
| **Calibration** | Mean predicted probability vs actual rate per group | Gap < 0.05 |

### Step 4 — Intersectional Analysis (`src/bias.py`)
Analyses **gender × race** combinations (e.g., Female + Black, Male + White) to surface compounded bias that single-attribute analysis misses.

### Step 5 — Explainability (`src/explainability.py`)
- **TreeSHAP** on Random Forest for global feature importance
- **Per-group SHAP** comparison: are features weighted differently for men vs women?
- **Proxy variable detection**: Spearman correlation between features and sensitive attributes. Features like `relationship` and `marital_status` often act as gender proxies.

### Step 6 — Bias Mitigation (`src/bias.py` + `src/model.py`)

Two mitigation techniques:

**A. Reweighing (pre-processing)**
Assigns sample weights so that every (group, label) cell — e.g., (Male, >50K), (Female, ≤50K) — has equal expected weight, removing statistical dependence before training.

**B. Threshold Adjustment (post-processing)**
Grid-searches decision thresholds (0.30–0.75) to minimise Statistical Parity Difference while maintaining accuracy ≥ 78%.

**Significance testing:** McNemar's test validates that the difference between original and mitigated predictions is statistically significant (p < 0.05).

---

## 📊 Dashboard Tabs

| Tab | Content |
|---|---|
| **Overview** | KPIs, pipeline summary, quick fairness table |
| **Model Performance** | Metrics comparison, confusion matrices, AUC |
| **Bias Detection** | SPD, DI, Equalized Odds, Calibration — interactive by attribute |
| **Intersectionality** | Gender × Race positive rates and TPR/FPR |
| **Explainability** | SHAP importance, per-group SHAP, proxy detection |
| **Mitigation** | Before/after comparison, trade-off analysis, McNemar test |

---

## 🔑 Key Findings (Expected)

1. The model shows **significant gender bias**: men receive positive predictions at ~2× the rate of women
2. **Race bias** is even stronger: White individuals receive >50K predictions at substantially higher rates
3. **Intersectional analysis** reveals that certain groups (e.g., non-White women) face compounded disadvantage
4. Features like `relationship`, `marital_status`, and `occupation` act as **gender proxy variables**
5. **Reweighing** reduces SPD by ~30–40% with minimal accuracy cost
6. **Threshold adjustment** can reduce SPD by ~50–60% at a ~2% accuracy cost

---

## 📚 References

- Kamiran, F. & Calders, T. (2012). *Data preprocessing techniques for classification without discrimination*
- Hardt, M., Price, E. & Srebro, N. (2016). *Equality of Opportunity in Supervised Learning*
- Lundberg, S. & Lee, S. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP)*
- Chouldechova, A. (2017). *Fair prediction with disparate impact*
- Dua, D. & Graff, C. (2019). *UCI Machine Learning Repository — Adult Dataset*
