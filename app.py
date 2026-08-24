"""Fraud-Spike Detector — AI Risk Manager (Razorpay Buildathon, Track 2).

Streamlit app, three tabs:
  1. Live Transaction Scorer   - score a real held-out transaction, explain it
  2. Fraud-Spike Monitor       - control-chart alarm on flagged-rate bursts
  3. Metrics & Methodology     - honest metrics, cost curve, defense-only statement

Reads model/model.joblib + model/report.json produced by train.py. It never
retrains on boot; if the model is missing it trains once and caches.
"""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_download import ensure_data
from features import FEATURE_COLS, engineer

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "model.joblib"
REPORT_PATH = ROOT / "model" / "report.json"

st.set_page_config(page_title="Fraud-Spike Detector", page_icon="🛡️", layout="wide")


# ----------------------------------------------------------------------------
# Loading (cached so reboots are instant)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_report():
    if not MODEL_PATH.exists() or not REPORT_PATH.exists():
        ensure_data()
        subprocess.run([sys.executable, str(ROOT / "train.py")], check=True)
    model = joblib.load(MODEL_PATH)
    with open(REPORT_PATH) as f:
        report = json.load(f)
    return model, report


@st.cache_data
def load_test_stream(op_threshold):
    """Load a contiguous time-window of held-out-like transactions for the
    spike monitor. We reconstruct the test split with the same seed, sort by
    Time, and score every row once."""
    ensure_data()
    df = pd.read_csv(ROOT / "data" / "creditcard.csv")
    df_eng = engineer(df)
    from sklearn.model_selection import train_test_split
    X = df_eng[FEATURE_COLS]
    y = df_eng["Class"].astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    model, _ = load_model_and_report()
    proba = model.predict_proba(X_test)[:, 1]
    stream = pd.DataFrame({
        "Time": df_eng.loc[X_test.index, "Time"].values,
        "Amount": df_eng.loc[X_test.index, "Amount"].values,
        "score": proba,
        "label": y_test.values,
    }).sort_values("Time").reset_index(drop=True)
    stream["flagged"] = (stream["score"] >= op_threshold).astype(int)
    return stream


model, report = load_model_and_report()
OP = report["operating_point"]["threshold"]

st.title("🛡️ Fraud-Spike Detector")
st.caption(
    "AI Risk Manager · Track 2 — transaction-level fraud classifier + "
    "real-time burst monitor. Strictly defense-only: scores and flags for "
    "review, takes no autonomous adverse action."
)

tab1, tab2, tab3 = st.tabs(
    ["🔍 Live Scorer", "📈 Spike Monitor", "📊 Metrics & Methodology"]
)


# ----------------------------------------------------------------------------
# TAB 1 — Live Transaction Scorer
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Score a held-out transaction")
    samples = report["sample_transactions"]

    c_pick, c_btn = st.columns([4, 1])
    labels = [
        f"#{s['id']} · ${s['Amount']:.2f} · {'FRAUD' if s['label'] else 'legit'} "
        f"· hr {s['Hour']} · score {s['score']:.3f}"
        for s in samples
    ]
    if "pick_idx" not in st.session_state:
        st.session_state.pick_idx = 0
    with c_btn:
        st.write("")
        st.write("")
        if st.button("🎲 Random", use_container_width=True):
            st.session_state.pick_idx = int(np.random.randint(len(samples)))
    with c_pick:
        idx = st.selectbox(
            "Pick a real test transaction (fraud + legit across the score range):",
            range(len(samples)), format_func=lambda i: labels[i],
            index=st.session_state.pick_idx, key="pick_select",
        )
    st.session_state.pick_idx = idx
    s = samples[idx]

    thr = st.slider(
        "Decision threshold", 0.01, 0.99, float(round(OP, 2)), 0.01,
        help=f"Cost-optimal operating threshold from train.py is {OP:.2f}.",
    )

    score = s["score"]
    decision = "FLAG for review" if score >= thr else "Allow"
    truth = "FRAUD" if s["label"] == 1 else "Legitimate"

    g1, g2 = st.columns([1, 1])
    with g1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score * 100,
            number={"suffix": "%"},
            title={"text": "Fraud risk score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#d62728" if score >= thr else "#2ca02c"},
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": thr * 100,
                },
                "steps": [
                    {"range": [0, thr * 100], "color": "#e8f5e9"},
                    {"range": [thr * 100, 100], "color": "#ffebee"},
                ],
            },
        ))
        gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge, use_container_width=True)
    with g2:
        st.metric("Model decision", decision)
        st.metric("Ground truth (held-out label)", truth)
        st.metric("Transaction amount", f"${s['Amount']:.2f}")
        if (score >= thr) == (s["label"] == 1):
            st.success("✓ Model decision matches the true label.")
        else:
            st.warning("✗ Mismatch at this threshold (a FP or FN).")

    # Live precision/recall/cost at the chosen threshold from the sweep table
    sweep = {round(m["threshold"], 2): m for m in report["threshold_sweep"]}
    near = sweep.get(round(thr, 2)) or min(
        report["threshold_sweep"], key=lambda m: abs(m["threshold"] - thr)
    )
    st.markdown("**Test-set performance at this threshold**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{near['precision']:.3f}")
    m2.metric("Recall", f"{near['recall']:.3f}")
    m3.metric("F1", f"{near['f1']:.3f}")
    m4.metric("Expected cost", f"${near['cost']:,.0f}")

    # Feature contribution — global permutation importance (honest, PCA labels)
    st.markdown("**Top contributing features (global permutation importance)**")
    imp = report.get("permutation_importance")
    if imp:
        top = imp[:8]
        bar = go.Figure(go.Bar(
            x=[t["importance"] for t in top][::-1],
            y=[t["feature"] for t in top][::-1],
            orientation="h", marker_color="#1f77b4",
        ))
        bar.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Δ average-precision when shuffled")
        st.plotly_chart(bar, use_container_width=True)
    st.caption(
        "Features V1–V28 are PCA components of the original (confidential) "
        "transaction attributes — labels are intentionally opaque; we do not "
        "invent merchant/geo semantics the data does not contain."
    )


# ----------------------------------------------------------------------------
# TAB 2 — Fraud-Spike Monitor
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Control-chart monitor on the flagged-transaction rate")
    st.caption(
        "Replays held-out transactions in time order. A coordinated attack "
        "shows up as a *burst* of flags, not isolated high scores — this is "
        "what makes it a fraud-**spike** detector."
    )

    stream = load_test_stream(OP)

    c1, c2, c3 = st.columns(3)
    bucket_min = c1.slider("Rolling bucket (minutes)", 1, 15, 5)
    z_alarm = c2.slider("Alarm z-score", 2.0, 5.0, 3.0, 0.5)
    inject = c3.checkbox("Inject simulated attack burst (synthetic demo)")

    bucket_sec = bucket_min * 60
    s2 = stream.copy()
    t0 = s2["Time"].min()
    s2["bucket"] = ((s2["Time"] - t0) // bucket_sec).astype(int)

    if inject:
        # Clearly-labelled SYNTHETIC injection, distinct from real replayed data.
        mid = int(s2["bucket"].max() * 0.6)
        n_inj = 40
        inj_time = t0 + mid * bucket_sec + np.random.randint(0, bucket_sec, n_inj)
        inj = pd.DataFrame({
            "Time": inj_time,
            "Amount": np.random.uniform(50, 400, n_inj),
            "score": np.random.uniform(OP + 0.05, 0.99, n_inj),
            "label": 1,
            "flagged": 1,
            "synthetic": 1,
        })
        s2["synthetic"] = 0
        s2 = pd.concat([s2, inj], ignore_index=True).sort_values("Time")
        s2["bucket"] = ((s2["Time"] - t0) // bucket_sec).astype(int)
    else:
        s2["synthetic"] = 0

    grp = s2.groupby("bucket").agg(
        rate=("flagged", "mean"),
        n=("flagged", "size"),
        t_mid=("Time", "mean"),
    ).reset_index()
    # Trailing rolling baseline (mean/std of prior buckets) -> z-score alarm
    grp["roll_mean"] = grp["rate"].shift(1).rolling(6, min_periods=3).mean()
    grp["roll_std"] = grp["rate"].shift(1).rolling(6, min_periods=3).std()
    grp["z"] = (grp["rate"] - grp["roll_mean"]) / grp["roll_std"].replace(0, np.nan)
    grp["alarm"] = grp["z"] >= z_alarm
    grp["upper"] = grp["roll_mean"] + z_alarm * grp["roll_std"]
    grp["hr"] = grp["t_mid"] / 3600.0

    n_alarm = int(grp["alarm"].sum())
    if n_alarm:
        st.error(f"🚨 {n_alarm} spike alarm(s) fired "
                 f"(flagged-rate exceeded {z_alarm:.1f}σ of trailing baseline).")
    else:
        st.success("No spike alarms — flagged-rate within statistical bounds.")

    # Rate line + alarm band
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grp["hr"], y=grp["rate"], mode="lines+markers",
                             name="Flagged rate", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=grp["hr"], y=grp["upper"], mode="lines",
                             name=f"{z_alarm:.1f}σ alarm band",
                             line=dict(color="#ff7f0e", dash="dash")))
    al = grp[grp["alarm"]]
    fig.add_trace(go.Scatter(x=al["hr"], y=al["rate"], mode="markers",
                             name="ALARM", marker=dict(color="#d62728", size=13,
                             symbol="x")))
    fig.update_layout(height=320, xaxis_title="Time (hours since window start)",
                      yaxis_title="Fraction of transactions flagged",
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Underlying per-transaction scatter
    fig2 = go.Figure()
    real = s2[s2["synthetic"] == 0]
    fig2.add_trace(go.Scattergl(
        x=real["Time"] / 3600.0, y=real["score"], mode="markers",
        name="Replayed transactions",
        marker=dict(size=4, color=real["label"], colorscale=[[0, "#9ecae1"],
                    [1, "#d62728"]], showscale=False, opacity=0.6),
    ))
    if inject:
        syn = s2[s2["synthetic"] == 1]
        fig2.add_trace(go.Scattergl(
            x=syn["Time"] / 3600.0, y=syn["score"], mode="markers",
            name="SYNTHETIC injected burst",
            marker=dict(size=7, color="#ff7f0e", symbol="triangle-up"),
        ))
    fig2.add_hline(y=OP, line_dash="dot", line_color="gray",
                   annotation_text=f"threshold {OP:.2f}")
    fig2.update_layout(height=300, xaxis_title="Time (hours)",
                       yaxis_title="Fraud risk score",
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Red points = true fraud in the real replayed stream. Orange "
               "triangles (if enabled) are a clearly-labelled synthetic demo "
               "injection, not real data.")


# ----------------------------------------------------------------------------
# TAB 3 — Metrics & Methodology
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Honest metrics on the held-out test set")
    ds = report["dataset"]
    st.markdown(
        f"**Dataset:** {ds['name']} — {ds['rows']:,} transactions, "
        f"{ds['frauds']} frauds ({ds['fraud_rate']:.3%} base rate). "
        f"Held-out test set: {ds['test_rows']:,} rows, {ds['test_frauds']} frauds. "
        f"Stratified 70/30 split, `random_state=42`; test set used only for final "
        f"evaluation, never for threshold tuning."
    )

    mdl, base, op = report["model"], report["baseline"], report["operating_point"]
    comp = pd.DataFrame([
        {"Model": mdl["name"], "PR-AUC": mdl["pr_auc"], "ROC-AUC": mdl["roc_auc"],
         "Precision@op": op["precision"], "Recall@op": op["recall"], "F1@op": op["f1"]},
        {"Model": base["name"], "PR-AUC": base["pr_auc"], "ROC-AUC": base["roc_auc"],
         "Precision@op": base["precision"], "Recall@op": base["recall"],
         "F1@op": base["f1"]},
    ])
    st.dataframe(comp.style.format({
        "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}", "Precision@op": "{:.3f}",
        "Recall@op": "{:.3f}", "F1@op": "{:.3f}"}), use_container_width=True)
    st.caption("Primary metric at a 0.17% base rate is **PR-AUC**, not ROC-AUC — "
               "ROC-AUC looks flattering on extreme class imbalance.")

    cc1, cc2 = st.columns(2)
    with cc1:
        pr = report["pr_curve"]
        f = go.Figure(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines",
                                 line=dict(color="#1f77b4")))
        f.update_layout(title="Precision-Recall curve", xaxis_title="Recall",
                        yaxis_title="Precision", height=320,
                        margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, use_container_width=True)
    with cc2:
        rc = report["roc_curve"]
        f = go.Figure(go.Scatter(x=rc["fpr"], y=rc["tpr"], mode="lines",
                                 line=dict(color="#2ca02c")))
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                               line=dict(color="gray", dash="dash"),
                               showlegend=False))
        f.update_layout(title="ROC curve", xaxis_title="False-positive rate",
                        yaxis_title="True-positive rate", height=320,
                        margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, use_container_width=True)

    st.markdown("### 💰 Cost model — the false-positive-cost bar")
    cm = report["cost_model"]
    cb = report["cost_baselines"]
    st.markdown(
        f"Assumptions (illustrative, configurable): missed fraud costs "
        f"**${cm['FN_COST_FIXED']:.0f}** admin **+ the lost amount**, a wrongly "
        f"flagged legit transaction costs **${cm['FP_COST']:.2f}**, reviewing a "
        f"caught fraud costs **${cm['TP_REVIEW_COST']:.0f}**. We sweep every "
        f"threshold and pick the one that minimises total expected cost."
    )

    sweep = pd.DataFrame(report["threshold_sweep"])
    fc = go.Figure()
    fc.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["cost"], mode="lines",
                            name="Expected cost", line=dict(color="#d62728")))
    fc.add_vline(x=op["threshold"], line_dash="dash", line_color="black",
                 annotation_text=f"cost-optimal {op['threshold']:.2f}")
    fc.add_hline(y=cb["flag_nothing"], line_dash="dot", line_color="gray",
                 annotation_text="flag nothing")
    fc.update_layout(title="Expected cost vs. decision threshold",
                     xaxis_title="Threshold", yaxis_title="Total expected cost ($)",
                     height=340, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fc, use_container_width=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Flag nothing", f"${cb['flag_nothing']:,.0f}")
    k2.metric("Default (0.5)", f"${cb['default_0_5']:,.0f}")
    k3.metric(f"Cost-optimal ({op['threshold']:.2f})", f"${cb['cost_optimal']:,.0f}")
    k4.metric("Savings vs nothing", f"{cb['savings_pct_vs_nothing']:.1f}%")

    st.markdown("### Confusion matrix at the operating threshold")
    cmx = go.Figure(go.Heatmap(
        z=[[op["tn"], op["fp"]], [op["fn"], op["tp"]]],
        x=["Pred: legit", "Pred: fraud"], y=["True: legit", "True: fraud"],
        text=[[f"TN {op['tn']:,}", f"FP {op['fp']:,}"],
              [f"FN {op['fn']:,}", f"TP {op['tp']:,}"]],
        texttemplate="%{text}", colorscale="Blues", showscale=False,
    ))
    cmx.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(cmx, use_container_width=True)

    st.markdown("### 🛡️ Defense-only compliance statement")
    st.info(
        "This system **only scores, flags, and alerts**. It takes no autonomous "
        "adverse action beyond flagging a transaction for human review. It exposes "
        "no capability to construct fraud, evade detection, or generate synthetic "
        "fraudulent instruments. The 'inject burst' control produces clearly "
        "labelled synthetic demo data for alarm testing only — it does not model "
        "or teach any real attack technique."
    )

    st.markdown("### Limitations (stated up front)")
    st.markdown(
        "- Dataset is 2013 European card transactions, **not** Indian BFSI data — "
        "distribution shift would require recalibration for production.\n"
        "- V1–V28 are PCA components; real semantic explainability is limited by "
        "the anonymisation.\n"
        "- Cost-model dollar figures are **illustrative assumptions**, not "
        "calibrated to a real merchant's P&L.\n"
        "- The spike monitor's control-chart bounds are a simple z-score baseline; "
        "a production system would tune window size and seasonality per merchant."
    )
