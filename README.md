<div align="center">

# `fairles` — Shapley meets Rawls

### An integrated framework for measuring and explaining unfairness

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


</div>

---

> *Shapley meets Rawls: an integrated framework for measuring and explaining unfairness.*


This library implements the **two-stage ESL (Efficient-Symmetric-Linear) attribution framework** for auditing and explaining group fairness in binary classifiers. It bridges two fields that are usually treated separately — **Shapley-based explainability** and **group fairness criteria** — into a single, unified pipeline.

---

## How it works — at a glance

![Two-stage ESL overview](/image/summary.png)

**The core idea:** a classifier is *fair* if and only if each demographic group contributes equally to the classification metric (Theorem 3.1). When fairness is violated, we recursively decompose each group's contribution over features to pinpoint which ones drive the disparity.

| Stage | Question answered | Output |
|:------|:-----------------|:-------|
| **Stage 1** | *Is the classifier fair?* | Group-ESL values + Z-test |
| **Stage 2** | *Which features cause unfairness?* | Feature contributions  per group + majority voting + Z-test |
| **Fair variant** | *Did mitigation work?* | Repeat with AIF360 Equalized Odds post-processing |

To ensure that conclusions do not depend on a single attribution method, the framework runs all five ESL values  — **Shapley**, **Equal Surplus (ES)**, **Solidarity**, **Consensus**, and **LSP** — independently on each feature. Each value casts a vote: "unfair" (reject) or "fair" (fail to reject). A feature is declared **unfair** when at least 3 out of 5 ESL values reject the null hypothesis. This majority rule makes the analysis robust to the idiosyncrasies of any single method.

![Majority voting](/image/majority_voting.png)
---

## Repository structure

```
shapley-rawls-fairness/
│
├── les/                          # Core library
│   ├── __init__.py               # exports LES, LESfair
│   ├── LES.py                    # Standard ESL attribution + variance
│   └── LES_Fair.py               # Fair ESL attribution (AIF360)
│
├── les_plot/                     # Visualization (SHAP-style plots)
│   ├── __init__.py
│   └── plot_contributions.py     # les_group_bar, les_bar, les_summary_dot, les_majority_voting
│
├── notebooks/
│   └── census_income_demo.ipynb  #  Full reproduction
│
├── docs/img/                     # Diagrams and figures for documentation
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/your-username/shapley-rawls-fairness.git
cd shapley-rawls-fairness
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

**Dependencies:** numpy, scipy, scikit-learn, pandas, matplotlib, joblib, aif360, xgboost, lightgbm, ucimlrepo.

---

## Quick start 

Open the notebook and run all cells:

```bash
jupyter notebook notebooks/census_income_demo.ipynb
```

The notebook reproduces every result from the paper on the **Census Income dataset** (UCI, Kohavi 1996) — from data loading to the final majority voting table. It covers:

1. Data preprocessing and soft-voting ensemble training
2. First-stage ESL group attribution (all 5 methods)
3. Group-level fairness Z-test
4. Second-stage feature attribution with parallelism
5. Asymptotic variance and feature-level Z-tests
6. Equalized Odds mitigation + verification
7. All  plots

---

## Use case — Census Income gender fairness audit

This walkthrough follows the paper's empirical application. The task: predict whether income exceeds $50K, with **gender** as the sensitive attribute.

### Step 1 — Prepare data

The sensitive attribute must always be the **last column**:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Features: age, education-num, hours-per-week, marital-status, sex (LAST)
X = full_data[['age', 'education-num', 'hours-per-week', 'marital-status', 'sex']]
y = (full_data['income'] == '>50K').astype(int)

# Encode categorical variables
le_marstat = LabelEncoder()
le_sex = LabelEncoder()
X = X.copy()
X['marital-status'] = le_marstat.fit_transform(X['marital-status'])
X['sex'] = le_sex.fit_transform(X['sex'])
y = LabelEncoder().fit_transform(y).ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.3, random_state=42
)
Xshp  = X_train.to_numpy()
yshp  = y_train.copy()
Xshpt = X_test.to_numpy()
yshpt = y_test.copy()

gender_col_index = X_test.columns.get_loc('sex')



np.save("Xshp.npy",  Xshp)
np.save("yshp.npy",  yshp)
np.save("Xshpt.npy", Xshpt)
np.save("yshpt.npy", yshpt)
np.save("gender_idx.npy", np.array([gender_col_index]))

```

### Step 2 — Train a classifier

Any scikit-learn compatible classifier works:

```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

model = VotingClassifier(estimators=[
    ('dt', DecisionTreeClassifier(class_weight='balanced')),
    ('xgb', XGBClassifier(scale_pos_weight=3.0)),
    ('lr', LogisticRegression(class_weight='balanced', max_iter=500)),
    ('rf', RandomForestClassifier(class_weight='balanced', n_estimators=200)),
], voting='soft')
model.fit(X_train, y_train)
```

### Step 3 — First-stage ESL (group contributions)

```python
from LES import LES

# Instantiate one LES object per ESL value
attribution     = LES(model=m, method='shapley')
attributionES   = LES(model=m, method='ES_LES')
attributionSol  = LES(model=m, method='solidarity')
attributionCs   = LES(model=m, method='consensus')
attributionLSP  = LES(model=m, method='LSP')
attributionESLin= LES(model=m, method='ES')

# --- Shapley ---
results_level1 = attribution.fit_FS(Xshp, yshp, Xshpt, yshpt)
# --- ES  ---
resultsesLin_level1 = attributionESLin.fit_FS(Xshp, yshp, Xshpt, yshpt)
# --- Solidarity ---
resultsol_level1 = attributionSol.fit_FS(Xshp, yshp, Xshpt, yshpt)
# --- Consensus ---
resultscs_level1 = attributionCs.fit_FS(Xshp, yshp, Xshpt, yshpt)
# --- LSP ---
resultsLSp_level1 = attributionLSP.fit_FS(Xshp, yshp, Xshpt, yshpt)

phi_dict = {
    "Shapley": np.asarray(results_level1).ravel(),
    "ES": np.asarray(resultses_level1).ravel(),
    "Solidarity": np.asarray(resultsol_level1).ravel(),
    "Consensus": np.asarray(resultscs_level1).ravel(),
    "LSP": np.asarray(resultsLSp_level1).ravel()
}

```
Output (paper Table 1):
```
Shapley       Men: 1.047   Women: 0.565
ES            Men: 1.047   Women: 0.565
Solidarity    Men: 0.926   Women: 0.685
Consensus     Men: 1.047   Women: 0.565
LSP           Men: 1.047   Women: 0.565
```

**Interpretation:** Men receive ~65% of the TPR improvement, women only ~35%. The classifier is unfair.

### Step 4 — Visualise group contributions

```python
import importlib
import plot_contributions
import plot_contributions as ples

fig, ax = ples.les_group_bar(
    phi_dict,
    group_labels=["Women", "Men"]
)
fig.savefig("group contribution", dpi=300, bbox_inches="tight")
plt.show()
```


![group contribution overview](/image/group_contribution.png)



### Step 6 — Second-stage feature attribution

```python
# Parallel computation for all 5 ESL methods
attributionSecdp  = LES(model=voting_clf, method='shapleysecd_parallel')
attributionSecdp1 = LES(model=voting_clf, method='ESsecond_parallel')
attributionSecdp2 = LES(model=voting_clf, method='solidaritysecond_parallel')
attributionSecdp3 = LES(model=voting_clf, method='consensussecond_parallel')
attributionSecdp4 = LES(model=voting_clf, method='LSPsecond_parallel')

resp = attributionSecdp.fit_parallel(
    Xshp, yshp, Xshpt, yshpt, n_jobs=56
)
resp1 = attributionSecdp1.fit_parallel(
    Xshp, yshp, Xshpt, yshpt, n_jobs=32
)
resp2 = attributionSecdp2.fit_parallel(
    Xshp, yshp, Xshpt, yshpt, n_jobs=32
)
resp3 = attributionSecdp3.fit_parallel(
    Xshp, yshp, Xshpt, yshpt, n_jobs=32
)
resp4 = attributionSecdp4.fit_parallel(
    Xshp, yshp, Xshpt, yshpt, n_jobs=32
)
```

### Step 7 — Visualise feature contributions

```python
import importlib
import plot_contributions
import plot_contributions as ples

contrib_women_dict = {
    "Shapley": np.asarray(resp[0]).ravel(),
    "ES": np.asarray(resp1[0]).ravel(),
    "Solidarity": np.asarray(resp2[0]).ravel(),
    "Consensus": np.asarray(resp3[0]).ravel(),
    "LSP": np.asarray(resp4[0]).ravel(),
}

contrib_men_dict = {
    "Shapley": np.asarray(resp[1]).ravel(),
    "ES": np.asarray(resp1[1]).ravel(),
    "Solidarity": np.asarray(resp2[1]).ravel(),
    "Consensus": np.asarray(resp3[1]).ravel(),
    "LSP": np.asarray(resp4[1]).ravel(),
}

feature_names = column_names[:gender_col_index] 
fig, axes = ples.les_summary_dot(
    contrib_women_dict,
    contrib_men_dict,
    feature_names,
    group_labels=["Women", "Men"]
)
fig.savefig("feature contribution", dpi=300, bbox_inches="tight")
plt.show()
```


![feature  contribution by group ](/image/feature_contribution.png)

**Reading the plot:** each panel is one ESL method. Orange dots = women's contribution, blue dots = men's. When dots are far apart, the feature contributes differently across groups → source of unfairness.

### Step 8 — Majority voting

```python
import importlib
import plot_contributions
import plot_contributions as ples

feature_name = ["age","edu-num","h_per_w","mart-status"] 
fig, ax = ples.les_majority_voting(
    test_results,
    feature_name,
    alpha=0.05,
)
fig.savefig("majority_voting.png", dpi=400, bbox_inches="tight")
plt.show()
```

![Majority voting ](/image/majority_voting.png)

**Result:** Age, Hours/Week, and Marital status are flagged as unfair by majority voting. Removing Marital status alone would increase the fairness gap — it acts as a proxy for gender. This finding goes beyond prior work that identified only proxy features.

### Step 9 — Fair mitigation (optional)

```python
from les import LESfair

fair_attr = LESfair(model=model, method='FairESadj')
phi_fair = fair_attr.fit_FSF(
    X_train, y_train, X_test, y_test,
    column_names=feature_names + ['sex'],
    label='income', attribute='sex'
)
# After EqOdds: Men 51.7% / Women 48.3% → gap ≈ 0 → fair
```

---

## Plot gallery

The `les_plot` module provides four SHAP-style visualizations:

| Function | Purpose | Stage |
|:---------|:--------|:------|
| `les_group_bar(phi_dict)` | Grouped horizontal bars: one pair per ESL method | 1 |
| `les_bar(contrib_dict, features)` | Ranked feature bars for one group, all methods | 2 |
| `les_summary_dot(cw, cm, features)` | Dot plot per method — women vs men per feature | 2 |
| `les_majority_voting(tests, features)` | Heatmap with ✓/✗ decisions + majority row | 2 |

All functions return `(fig, ax)` and accept `save_path` for export.

---

## Key convention

> **The sensitive attribute must always be the last column** of every feature matrix `X` / `Xt` passed to the library.

`group_column` defaults to `X.shape[1] - 1` and can be omitted in most calls.

```python
# Correct: sex is the last column
X = df[['age', 'education-num', 'hours-per-week', 'marital-status', 'sex']]
```

---

## API reference

### `LES` — Standard ESL attribution

```python
from les import LES

attr = LES(model=clf, method='shapley')
```

**First-stage methods** (pass to `method=`):
`'shapley'`, `'ES'`, `'ES_LES'`, `'solidarity'`, `'consensus'`, `'LSP'`

```python
phi = attr.fit_FS(X_train, y_train, X_test, y_test, metric='TPR')
# phi.shape = (1, 2) → [women, men]
```

**Second-stage methods** (parallel):
`'shapleysecd_parallel'`, `'ESsecd_parallel'`, `'solidaritysecd_parallel'`, `'consensussecd_parallel'`, `'LSPsecd_parallel'`

```python
attr2 = LES(model=clf, method='shapleysecd_parallel')
contrib_women, contrib_men = attr2.fit_parallel(X_train, y_train, X_test, y_test, n_jobs=8)
```

**Variance and tests:**
`'VAR_par'`, `'VAR_par_1'`, `'VAR_par_2'` — parallelised variance estimation for asymptotic Z-tests.

**Fairness test (static method):**

```python
result = LES.fairness_test(b1, b2, phi_men, phi_women, p, n1, n2)
# Returns: {'z_stat': float, 'p_value': float, 'ci_lower': float, 'ci_upper': float}
```

### `LESfair` — Fair ESL attribution

```python
from les import LESfair

fair = LESfair(model=clf, method='FairESadj')
phi = fair.fit_FSF(X_train, y_train, X_test, y_test,
                   column_names=names, label='income', attribute='sex')
```

Applies AIF360 **Equalized Odds post-processing** before computing ESL values.

---



## Supported fairness criteria

The framework works with any metric expressible as a group ratio:

| Criterion | Metric | Definition |
|:----------|:-------|:-----------|
| Independence (IND) | Selection Rate | P(Ŷ=1\|A=g) |
| Separation (SEP) | TPR + FPR | P(Ŷ=1\|Y=1,A=g) + P(Ŷ=1\|Y=0,A=g) |
| Equal Opportunity | TPR only | P(Ŷ=1\|Y=1,A=g) |
| Sufficiency (SUF) | PPV + NPV | P(Y=1\|Ŷ=1,A=g) + P(Y=1\|Ŷ=0,A=g) |

Change the metric via the `metric=` parameter: `'TPR'`, `'FPR'`, `'PPV'`, `'NPV'`.

---


## License

MIT — see [LICENSE](LICENSE).
