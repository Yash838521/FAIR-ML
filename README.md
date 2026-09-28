# FAIR-ML: Fairness-Aware Machine Learning

An end-to-end responsible machine-learning project that trains income classifiers,
measures group-level performance disparities, explains model behaviour with SHAP,
and evaluates mitigation trade-offs.

[Live demo](https://fair-ml.streamlit.app) | Dataset: UCI Adult Income

## Project overview

The pipeline processes 48,842 Adult Income records and compares Logistic Regression
with Random Forest. It evaluates predictive performance alongside fairness metrics
for gender and race, including statistical parity difference, disparate impact,
equalized-odds gaps and calibration. It also includes intersectional analysis,
SHAP-based explainability, proxy-variable checks, reweighing and threshold
adjustment.

## Reproduced results after the audit repair

| Experiment | Accuracy | ROC-AUC | Gender SPD | TPR gap | FPR gap |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8482 | 0.9091 | Not audited | — | — |
| Logistic Regression | 0.7718 | 0.8475 | 0.3472 | 0.3733 | 0.2337 |
| LR + reweighing | 0.7502 | 0.8277 | 0.1002 | 0.0236 | 0.0066 |
| LR + shared threshold | 0.8165 | 0.8475 | 0.1878 | 0.2808 | 0.0856 |

These results use 28,941 training, 7,236 validation,
and 9,045 test records after complete-case cleaning (45,222 retained
from 48,842 source records). The original UCI files are pooled and re-split, not
evaluated using their original partition. Seed: 42. Python: 3.11.

The shared threshold (0.74) is selected only on validation data
to minimise gender SPD with a validation accuracy floor of 0.78. Reweighing reduces
gender SPD by 71.1% relative to the baseline on this test split. These are
trade-offs, not a claim that bias is eliminated. The main fairness audit and
mitigation concern Logistic Regression; the SHAP explanations concern Random Forest.
McNemar's test compares paired error rates and does not test fairness improvement.

Important limitations: nominal features use ordinal codes; gender and race_binary
are included as inputs; group-mean calibration is not a full reliability analysis;
Spearman association is only a proxy screening signal; intersectional groups with
fewer than 30 test examples are omitted; uncertainty intervals are not reported.
The 0.10/0.80/0.05 audit tolerances are illustrative, not universal or legal standards.

## Pipeline

1. Download and clean the UCI Adult Income data.
2. Encode categorical variables and scale numerical features without test-set leakage.
3. Train Logistic Regression and Random Forest classifiers.
4. Evaluate accuracy, precision, recall, F1, ROC-AUC and confusion matrices.
5. Measure fairness by gender, race and intersectional groups.
6. Generate global and group-level SHAP explanations.
7. Compare reweighing and threshold-adjustment mitigation strategies.
8. Save structured JSON and CSV outputs for the Streamlit dashboard.

## Repository structure

```text
FAIR-ML/
|-- data/                         # Downloaded dataset files
|-- dashboard/
|   `-- streamlit_app.py          # Interactive analysis dashboard
|-- results/                      # Reproducible JSON and CSV outputs
|-- src/
|   |-- bias.py                   # Fairness metrics and mitigation
|   |-- explainability.py         # SHAP and proxy-variable analysis
|   |-- model.py                  # Training and evaluation
|   `-- preprocessing.py          # Leakage-aware data preparation
|-- pipeline.py                   # End-to-end orchestration
`-- requirements.txt
```

## Run locally

Use Python 3.11 with the pinned dependencies. The dashboard can run immediately
from the committed results; running the full pipeline regenerates them and takes
several minutes, mainly for SHAP.

```bash
git clone https://github.com/Yash838521/FAIR-ML.git
cd FAIR-ML
python -m venv .venv
```

Activate the environment, then run:

```bash
pip install -r requirements.txt
python pipeline.py
streamlit run dashboard/streamlit_app.py
```

The dashboard will be available at `http://localhost:8501`.

## Dashboard

The six dashboard views cover:

- model performance and confusion matrices
- gender and race fairness metrics
- intersectional group comparisons
- global and group-level SHAP explanations
- proxy-variable analysis
- mitigation results and accuracy trade-offs

## Responsible-use note

This is an educational analysis of a public benchmark dataset. The Adult Income
data is dated and contains historical social inequalities. Model outputs must not
be used for employment, credit or other consequential decisions. Fairness metrics
are diagnostic tools and do not establish that a model is fair in every context.

## References

- Kamiran, F. and Calders, T. (2012), data preprocessing for classification without discrimination
- Hardt, M., Price, E. and Srebro, N. (2016), equality of opportunity in supervised learning
- Lundberg, S. and Lee, S. (2017), SHAP
- UCI Machine Learning Repository, Adult dataset

## Verification and maintenance

```bash
pip install pytest
python -m pytest -q
pip check
```

GitHub Actions runs regression tests, including Streamlit widget checks. Tests
cover three-way split isolation, known fairness examples, SHAP class-axis and
feature labels, Spearman correlation, empty tables, threshold infeasibility,
McNemar edge cases, strict JSON types, and missing-result behaviour.

Streamlit Community Cloud should run `dashboard/streamlit_app.py` from `main`.
Commit regenerated `results/` files with experiment changes. Community Cloud may
sleep after inactivity; use its wake button before a presentation. A healthy app
does not guarantee that the hosting service will always remain available.
