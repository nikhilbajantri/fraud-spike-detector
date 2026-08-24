"""Train the fraud classifier, evaluate honestly on a held-out test set,
run a cost-sensitive threshold sweep, and persist model + report.

Run once:  python train.py

Outputs:
  model/model.joblib   - trained HistGradientBoostingClassifier
  model/report.json    - every metric / curve / cost number the app renders
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_download import ensure_data
from features import FEATURE_COLS, engineer

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
REPORT_PATH = MODEL_DIR / "report.json"

# --- Cost model (illustrative assumptions, documented in README + app Tab 3) ---
FN_COST_FIXED = 25.0   # dispute/chargeback admin overhead per missed fraud
FP_COST = 3.50         # cost of wrongly flagging a legit transaction
TP_REVIEW_COST = 8.0   # ops cost of manually reviewing a caught fraud


def expected_cost(y_true, y_proba, amounts, threshold):
    """Total expected cost at a decision threshold. Missed fraud also loses the
    transaction amount, not just the fixed admin fee."""
    pred = (y_proba >= threshold).astype(int)
    fn_mask = (pred == 0) & (y_true == 1)
    fp_mask = (pred == 1) & (y_true == 0)
    tp_mask = (pred == 1) & (y_true == 1)
    return float(
        fn_mask.sum() * FN_COST_FIXED
        + amounts[fn_mask].sum()
        + fp_mask.sum() * FP_COST
        + tp_mask.sum() * TP_REVIEW_COST
    )


def metrics_at(y_true, y_proba, threshold):
    pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    ensure_data()
    csv = Path(__file__).parent / "data" / "creditcard.csv"
    print("Loading data ...")
    df = pd.read_csv(csv)
    df_eng = engineer(df)
    X = df_eng[FEATURE_COLS]
    y = df_eng["Class"].astype(int)
    amounts = df_eng["Amount"].values

    X_train, X_test, y_train, y_test, amt_train, amt_test = train_test_split(
        X, y, amounts, test_size=0.30, random_state=42, stratify=y
    )
    y_test_np = y_test.to_numpy()
    amt_test = np.asarray(amt_test)
    print(f"Train {len(X_train):,}  Test {len(X_test):,}  "
          f"Fraud rate {y.mean():.4%}")

    # --- Primary model ---
    print("Training HistGradientBoostingClassifier ...")
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.08,
        class_weight="balanced", random_state=42,
        early_stopping=True, validation_fraction=0.1,
    )
    model.fit(X_train, y_train)
    train_secs = time.time() - t0
    proba = model.predict_proba(X_test)[:, 1]

    pr_auc = float(average_precision_score(y_test_np, proba))
    roc_auc = float(roc_auc_score(y_test_np, proba))
    print(f"  trained in {train_secs:.1f}s  PR-AUC {pr_auc:.4f}  ROC-AUC {roc_auc:.4f}")

    # --- Logistic-regression baseline for the comparison table ---
    print("Training logistic-regression baseline ...")
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    )
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_pr_auc = float(average_precision_score(y_test_np, lr_proba))
    lr_roc_auc = float(roc_auc_score(y_test_np, lr_proba))
    lr_default = metrics_at(y_test_np, lr_proba, 0.5)

    # --- Cost-sensitive threshold sweep ---
    print("Sweeping thresholds for cost-optimal operating point ...")
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    sweep = []
    for t in thresholds:
        m = metrics_at(y_test_np, proba, t)
        m["cost"] = expected_cost(y_test_np, proba, amt_test, t)
        sweep.append(m)

    cost_nothing = expected_cost(y_test_np, proba, amt_test, 1.01)   # flag nothing
    cost_everything = expected_cost(y_test_np, proba, amt_test, 0.0)  # flag all
    best = min(sweep, key=lambda m: m["cost"])
    op_threshold = best["threshold"]

    default_metrics = metrics_at(y_test_np, proba, 0.5)
    default_metrics["cost"] = expected_cost(y_test_np, proba, amt_test, 0.5)
    op_metrics = dict(best)

    savings_pct = 100.0 * (cost_nothing - op_metrics["cost"]) / cost_nothing

    # --- Feature importance (HistGBC has no .feature_importances_) ---
    print("Computing permutation importance (this takes ~30s) ...")
    # Subsample the test set to keep it fast; scoring metric = average precision.
    n_imp = min(20000, len(X_test))
    imp_idx = np.random.RandomState(42).choice(len(X_test), n_imp, replace=False)
    perm = permutation_importance(
        model, X_test.iloc[imp_idx], y_test.iloc[imp_idx],
        scoring="average_precision", n_repeats=5, random_state=42, n_jobs=-1,
    )
    perm_imp = sorted(
        [{"feature": FEATURE_COLS[i], "importance": float(perm.importances_mean[i])}
         for i in range(len(FEATURE_COLS))],
        key=lambda d: d["importance"], reverse=True,
    )

    # --- Curves (subsampled for compact JSON) ---
    prec, rec, _ = precision_recall_curve(y_test_np, proba)
    fpr, tpr, _ = roc_curve(y_test_np, proba)

    def thin(a, n=300):
        a = np.asarray(a, dtype=float)
        if len(a) <= n:
            return a.tolist()
        idx = np.linspace(0, len(a) - 1, n).astype(int)
        return a[idx].tolist()

    # --- Curated sample of test transactions for the live scorer ---
    # Mix of fraud + legit across the score range; store raw rows + score.
    test_idx = X_test.index
    sample_rows = []
    # fraud examples
    fraud_pos = [i for i in range(len(y_test_np)) if y_test_np[i] == 1]
    fraud_pick = fraud_pos[:12]
    # legit spread across score quantiles
    legit_pos = [i for i in range(len(y_test_np)) if y_test_np[i] == 0]
    legit_sorted = sorted(legit_pos, key=lambda i: proba[i])
    legit_pick = [legit_sorted[int(q * (len(legit_sorted) - 1))]
                  for q in np.linspace(0.05, 0.999, 18)]
    picks = sorted(set(fraud_pick + legit_pick))
    for i in picks:
        row = df_eng.loc[test_idx[i]]
        sample_rows.append({
            "id": int(test_idx[i]),
            "score": float(proba[i]),
            "label": int(y_test_np[i]),
            "Amount": float(row["Amount"]),
            "Hour": int(row["Hour"]),
            "features": {c: float(row[c]) for c in FEATURE_COLS},
        })

    report = {
        "dataset": {
            "name": "Kaggle Credit Card Fraud Detection (ULB)",
            "rows": int(len(df)),
            "frauds": int(y.sum()),
            "fraud_rate": float(y.mean()),
            "n_features": len(FEATURE_COLS),
            "test_rows": int(len(X_test)),
            "test_frauds": int(y_test_np.sum()),
        },
        "model": {
            "name": "HistGradientBoostingClassifier",
            "params": {"max_iter": 300, "max_depth": 6, "learning_rate": 0.08,
                       "class_weight": "balanced"},
            "train_secs": round(train_secs, 1),
            "pr_auc": pr_auc, "roc_auc": roc_auc,
        },
        "baseline": {
            "name": "LogisticRegression (balanced)",
            "pr_auc": lr_pr_auc, "roc_auc": lr_roc_auc,
            "precision": lr_default["precision"], "recall": lr_default["recall"],
            "f1": lr_default["f1"],
        },
        "cost_model": {
            "FN_COST_FIXED": FN_COST_FIXED, "FP_COST": FP_COST,
            "TP_REVIEW_COST": TP_REVIEW_COST,
            "note": "Illustrative assumptions, configurable; not a real merchant P&L.",
        },
        "operating_point": op_metrics,
        "default_point": default_metrics,
        "cost_baselines": {
            "flag_nothing": cost_nothing,
            "flag_everything": cost_everything,
            "default_0_5": default_metrics["cost"],
            "cost_optimal": op_metrics["cost"],
            "savings_pct_vs_nothing": savings_pct,
        },
        "threshold_sweep": sweep,
        "pr_curve": {"precision": thin(prec), "recall": thin(rec)},
        "roc_curve": {"fpr": thin(fpr), "tpr": thin(tpr)},
        "sample_transactions": sample_rows,
        "permutation_importance": perm_imp,
        "feature_cols": FEATURE_COLS,
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Summary ===")
    print(f"PR-AUC {pr_auc:.4f} | ROC-AUC {roc_auc:.4f}")
    print(f"Operating threshold {op_threshold:.2f}: "
          f"precision {op_metrics['precision']:.3f} "
          f"recall {op_metrics['recall']:.3f}")
    print(f"Cost savings vs flag-nothing: {savings_pct:.1f}%")
    print(f"Saved -> {MODEL_PATH} , {REPORT_PATH}")


if __name__ == "__main__":
    main()
