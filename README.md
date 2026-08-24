# Fraud-Spike Detector — AI Risk Manager

**Razorpay Buildathon · Track 2 (AI Risk Manager)**

A transaction-level fraud classifier paired with a real-time **spike monitor**
that alarms on coordinated fraud bursts — with a cost-sensitive operating
threshold and honest false-positive-cost accounting. **Strictly defense-only.**

- **Live demo:** _<fill in your https://…streamlit.app URL>_
- **Repo:** _<fill in your GitHub URL>_

---

## 1. Problem

Fraud, chargebacks and returns eat directly into merchant margin, and the cost
is asymmetric: a missed fraud loses the whole transaction plus dispute overhead,
while over-blocking legitimate customers burns real money in lost sales and
support load. The Track 2 bar is explicit — a working detector with **measured
precision and recall on a held-out test set, including false-positive cost**.
This project delivers exactly that, plus a burst monitor for the "spike" case: a
coordinated attack shows up as a *rate* anomaly, not just isolated high-risk
transactions.

## 2. Approach

Two layers:

1. **Transaction classifier** — scores each transaction's fraud probability.
   This is what produces the precision/recall numbers.
2. **Spike / burst monitor** — a control-chart on top of the classifier that
   watches the *flagged-transaction rate* over rolling time buckets and raises an
   alarm when it exceeds a z-score bound of its trailing baseline.

**Model:** `HistGradientBoostingClassifier(class_weight="balanced")` on the full
284K-row training data. We benchmarked alternatives and chose this deliberately:

| Approach | PR-AUC (test) | ROC-AUC | Train time | Verdict |
|---|---|---|---|---|
| Logistic Regression (balanced) | ~0.70 | ~0.97 | 2s | Baseline in the comparison table |
| GradientBoosting on aggressive undersample | ~0.22 | ~0.96 | 10–40s | Trap — ROC-AUC looks fine, PR-AUC collapses |
| RandomForest (300 trees) | ~0.82 | ~0.96 | ~160s | Best raw score, slow, large model |
| **HistGradientBoosting (full data)** | **0.781** | **0.947** | **~18s** | **Primary — near-RF accuracy, trains in seconds** |

At a 0.17% base rate, **PR-AUC is the metric that matters** — ROC-AUC is
flattering under extreme class imbalance. `HistGradientBoostingClassifier` has no
`.feature_importances_`, so explainability uses `permutation_importance` on the
test set (top signal: `V14`, `V12`, `V10` — consistent with the published
literature on this dataset).

## 3. Data

**Kaggle Credit Card Fraud Detection (ULB Machine Learning Group)** — a real,
widely-cited benchmark: 284,807 European cardholder transactions over two days,
492 frauds (**0.172%** fraud rate). Features `V1`–`V28` are **PCA components** of
the original (confidential) attributes — deliberately opaque; `Time` and `Amount`
are the only raw fields. We do **not** overclaim interpretability the anonymised
data does not support (no invented merchant/geo labels).

Engineered features on top of `V1`–`V28`: `Hour` (recovered daily cycle from
cumulative seconds), `LogAmount`, and a robust (median/IQR) amount z-score — 32
features total. Split: **stratified 70/30**, `random_state=42`; the test set is
touched only for final evaluation, never for threshold tuning.

The 98MB CSV is gitignored and **self-provisions on first run**
(`data_download.py` pulls it from a raw GitHub mirror), so the deployed app boots
with no manual data step.

## 4. Metrics (held-out test set)

| | PR-AUC | ROC-AUC | Precision@op | Recall@op | F1@op |
|---|---|---|---|---|---|
| HistGradientBoosting | **0.781** | 0.947 | 0.665 | 0.818 | 0.733 |
| LogisticRegression (baseline) | ~0.70 | ~0.97 | — | — | — |

**Cost model (illustrative, configurable — not a real merchant P&L):**

- Missed fraud (FN): **$25** admin overhead **+ the lost transaction amount**
- Wrongly flagged legit (FP): **$3.50** (friction, lost sale, support)
- Reviewing a caught fraud (TP): **$8** ops cost

We sweep every threshold `0.01→0.99`, compute total expected cost, and pick the
minimum — **not** the default 0.5.

- **Cost-optimal threshold ≈ 0.64** → precision **0.665**, recall **0.818**
- **~63% total-cost savings vs. doing nothing** (flag-nothing baseline)

The app's Metrics tab renders the PR curve, ROC curve, the expected-cost curve
with the optimal point marked, and the confusion matrix at the operating point.

## 5. Architecture

```
fraud-risk-manager/
├── app.py            # Streamlit app — 3 tabs (scorer / spike monitor / metrics)
├── train.py          # trains, evaluates, cost-sweeps, saves model + report
├── features.py       # shared feature engineering (train + serve parity)
├── data_download.py  # self-provisions the dataset on first run
├── model/
│   ├── model.joblib  # trained classifier
│   └── report.json   # all metrics / curves / cost numbers the app renders
├── data/creditcard.csv   # gitignored; auto-downloaded
├── requirements.txt
└── README.md
```

**Run locally:**

```bash
pip install -r requirements.txt
python train.py        # downloads data, trains, writes model/ (~1 min)
streamlit run app.py
```

`app.py` reads the precomputed `model/` artifacts and never retrains on boot.

**Deploy (Streamlit Community Cloud):** push to GitHub → https://share.streamlit.io
→ New app → pick repo/branch → `app.py` → deploy. Public `*.streamlit.app` URL in
a couple of minutes.

## 6. Defense-only statement

This system **only scores, flags, and alerts**. It takes no autonomous adverse
action beyond flagging a transaction for human review. It exposes no capability to
construct fraud, evade detection, or generate synthetic fraudulent instruments.
The "inject burst" demo control produces **clearly labelled synthetic** data for
alarm testing only — it models no real attack technique. Anything offense-capable
is out of scope by design.

## 7. Limitations (stated up front)

- Dataset is 2013 European card data, **not** Indian BFSI data — production use
  would require recalibration for distribution shift.
- `V1`–`V28` are PCA components; real semantic explainability is limited by the
  anonymisation.
- Cost-model dollar figures are **illustrative assumptions**, not calibrated to a
  real merchant's P&L — all three are configurable in `train.py`.
- The spike monitor uses a simple z-score control-chart baseline; a production
  system would tune window size and seasonality per merchant.
