"""Telco churn: which customers will leave, why, and who to contact first.

Split 60/20/20 (train / validation / test, stratified). Models are compared by 5-fold CV on train+validation;
the decision threshold is chosen on validation only; the test set is scored once.
Writes results/metrics.json, results/drivers.csv and results/*.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
R = {}

df = pd.read_csv(os.path.join(ROOT, "data", "telco_churn.csv"))
R["rows_raw"] = len(df)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
R["rows_blank_total_charges_tenure0"] = int(df["TotalCharges"].isna().sum())
df = df.dropna(subset=["TotalCharges"])  # all 11 are brand-new customers (tenure 0) with no bill yet
y = (df.pop("Churn") == "Yes").astype(int).values
df = df.drop(columns=["customerID"])
R["rows_used"], R["churn_rate"] = len(df), round(float(y.mean()), 4)

# Descriptive: churn rate by contract and tenure band
R["churn_by_contract"] = df.assign(c=y).groupby("Contract")["c"].mean().round(3).to_dict()
band = pd.cut(df["tenure"], [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
R["churn_by_tenure_months"] = df.assign(c=y).groupby(band, observed=True)["c"].mean().round(3).to_dict()

num = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat = [c for c in df.columns if c not in num]
prep = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat)])
models = {
    "logistic": Pipeline([("prep", prep), ("m", LogisticRegression(max_iter=2000, C=1.0))]),
    "gradient_boosting": Pipeline([("prep", prep), ("m", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                                         max_leaf_nodes=15, random_state=0))]),
}

X_trv, X_te, y_trv, y_te = train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=0.25, stratify=y_trv, random_state=42)

cv = StratifiedKFold(5, shuffle=True, random_state=0)
R["cv5_roc_auc"] = {k: [round(float(s.mean()), 4), round(float(s.std()), 4)]
                    for k, m in models.items() for s in [cross_val_score(m, X_trv, y_trv, cv=cv, scoring="roc_auc")]}
best = max(R["cv5_roc_auc"], key=lambda k: R["cv5_roc_auc"][k][0])
# Prefer the simpler, explainable model unless the other is clearly better (> 0.005 AUC)
if best != "logistic" and R["cv5_roc_auc"][best][0] - R["cv5_roc_auc"]["logistic"][0] <= 0.005:
    best = "logistic"
R["selected_model"] = best

model = models[best].fit(X_tr, y_tr)
p_va = model.predict_proba(X_va)[:, 1]
# Threshold: highest precision that still recalls at least 60% of churners, chosen on validation
cands = [(precision_score(y_va, p_va >= t, zero_division=0), t) for t in np.arange(0.05, 0.95, 0.01)
         if recall_score(y_va, p_va >= t) >= 0.60]
thr = round(float(max(cands)[1]), 2)
R["threshold_from_validation"] = thr

model = models[best].fit(X_trv, y_trv)  # refit on train+validation, then score test once
p = model.predict_proba(X_te)[:, 1]
order = np.argsort(-p)
top20 = order[: int(0.2 * len(p))]
R["test"] = {
    "n": int(len(y_te)), "roc_auc": round(float(roc_auc_score(y_te, p)), 4),
    "pr_auc": round(float(average_precision_score(y_te, p)), 4),
    "precision_at_threshold": round(float(precision_score(y_te, p >= thr)), 4),
    "recall_at_threshold": round(float(recall_score(y_te, p >= thr)), 4),
    "top20pct_share_of_churners_caught": round(float(y_te[top20].sum() / y_te.sum()), 4),
    "top20pct_churn_rate": round(float(y_te[top20].mean()), 4),
}

if best == "logistic":
    names = model.named_steps["prep"].get_feature_names_out()
    coef = pd.DataFrame({"feature": names, "coefficient": model.named_steps["m"].coef_[0]})
    coef["odds_ratio"] = np.exp(coef["coefficient"])
    coef = coef.reindex(coef["coefficient"].abs().sort_values(ascending=False).index)
    coef.round(4).to_csv(os.path.join(OUT, "drivers.csv"), index=False)
    top = coef.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"].str.replace("cat__", "").str.replace("num__", ""), top["coefficient"],
            color=["#c0392b" if c > 0 else "#2e86c1" for c in top["coefficient"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Logistic coefficient (red raises churn odds, blue lowers them)")
    ax.set_title("Strongest churn drivers")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "drivers.png"), dpi=150)

fpr, tpr, _ = roc_curve(y_te, p)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(fpr, tpr, lw=2, label=f"{best} (test AUC {R['test']['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "roc_test.png"), dpi=150)

with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(R, f, indent=2)
print(json.dumps(R, indent=2))
