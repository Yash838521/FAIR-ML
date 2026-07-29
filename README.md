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

## Achieved results

| Experiment | Accuracy | ROC-AUC | Gender SPD |
|---|---:|---:|---:|
| Random Forest | 0.8583 | 0.9089 | - |
| Logistic Regression | 0.7736 | 0.8477 | 0.3506 |
| Logistic Regression with reweighing | 0.7512 | 0.8291 | 0.1087 |
| Logistic Regression with threshold adjustment | 0.8177 | 0.8477 | 0.1870 |

Reweighing reduced gender statistical parity difference from 0.3506 to 0.1087,
while accuracy moved from 0.7736 to 0.7512. These results demonstrate a measurable
fairness-performance trade-off rather than claiming that bias was eliminated.

The Random Forest model achieved the strongest predictive performance. SHAP
analysis was used to explain behaviour across 14 model features, while group and
proxy analyses examined how sensitive and correlated variables affected outcomes.

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
