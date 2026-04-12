import pandas as pd
import streamlit as st
from src.database import (
    init_db,
    get_batch_history,
    get_batch_records,
    get_drift_alerts,
    get_summary_stats,
)
from src.model import churn_model
from src.logger import get_logger
from src.config import (
    LOG_FILE,
    TRAINING_STATS,
    GREEN,
    BLOOD_RED,
    LIGHT_RED,
    LIGHT_YELLOW,
    LIGHT_GREEN,
    ORANGE,
    GREY,
)

logger = get_logger(__name__)

st.set_page_config(
    page_title="Monitoring",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()


@st.cache_resource
def load_model():
    churn_model.load()
    return churn_model


try:
    load_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

st.title("📺 Monitoring Dashboard")
st.caption("Audit trail & Drift alerts & Batch history & Customer logs")
st.divider()

stats = get_summary_stats()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Batches", f"{stats.get('total_batches', 0) or 0:,}")
c2.metric("Customers Scored", f"{stats.get('total_customers', 0) or 0:,}")
c3.metric("Predicted Churned", f"{stats.get('total_churned', 0) or 0:,}")
c4.metric(
    "Avg Churn Rate",
    (
        f"{stats.get('avg_churn_rate', 0) or 0:.1%}"
        if stats.get("avg_churn_rate")
        else "-"
    ),
)
c5.metric("High Risk Flagged", f"{stats.get('total_high_risk', 0) or 0:,}")
st.divider()

st.subheader("Batch History")
batches = get_batch_history(10)

if not batches:
    st.info("No predictions yet - go to Live Predictor and upload a file")
else:
    bdf = pd.DataFrame(batches)
    bdf["uploaded_at"] = pd.to_datetime(bdf["uploaded_at"]).dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    bdf["churn_rate"] = bdf["churn_rate"].map(lambda x: f"{x:.1%}")

    def style_status(v):
        if v == "success":
            return f"color:{GREEN};font-weight:700"
        return f"color:{BLOOD_RED};font-weight:700"

    batch_cols = [
        "uploaded_at",
        "filename",
        "total_customers",
        "predicted_churn",
        "churn_rate",
        "high_risk",
        "medium_risk",
        "low_risk",
        "processing_ms",
        "status",
    ]
    batch_cols = [c for c in batch_cols if c in bdf.columns]

    st.dataframe(
        bdf[batch_cols].style.map(style_status, subset=["status"]),
        width="content",
        height=260,
        hide_index=True,
    )

    st.subheader("Customer-Level Drill Down")
    opts = {
        f"{b['uploaded_at'][:16]} | {b['filename']}": b["batch_id"] for b in batches
    }
    sel = st.selectbox("Select batch", list(opts.keys()))
    bid = opts[sel]
    recs = get_batch_records(bid)

    if recs:
        rdf = pd.DataFrame(recs)
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers", len(rdf))
        c2.metric("Churned", int((rdf["churn_prediction"] == "Yes").sum()))
        c3.metric("High Risk", int(rdf["risk_tier"].str.contains("High").sum()))

        def style_risk(v):
            if "High" in str(v):
                return f"background-color:{LIGHT_RED};color:{BLOOD_RED};font-weight:700"
            if "Medium" in str(v):
                return f"background-color:{LIGHT_YELLOW};color:{ORANGE};font-weight:700"
            if "Low" in str(v):
                return f"background-color:{LIGHT_GREEN};color:{GREEN};font-weight:700"
            return ""

        def style_churn(v):
            if v == "Yes":
                return f"color:{BLOOD_RED};font-weight:700"
            if v == "No":
                return f"color:{GREEN};font-weight:700"
            return ""

        rec_cols = [
            "customer_id",
            "churn_prediction",
            "churn_probability",
            "risk_tier",
            "tenure",
            "monthly_charges",
            "contract",
            "internet_service",
            "predicted_at",
        ]
        rec_cols = [c for c in rec_cols if c in rdf.columns]

        st.dataframe(
            rdf[rec_cols]
            .style.map(style_risk, subset=["risk_tier"])
            .map(style_churn, subset=["churn_prediction"])
            .format(
                {
                    "churn_probability": "{:.1%}",
                    "monthly_charges": "${:.2f}",
                    "tenure": "{:.0f}",
                }
            ),
            width="content",
            height=340,
            hide_index=True,
        )

        st.download_button(
            "Download batch records",
            rdf.to_csv(index=False).encode(),
            f"batch_{str(bid)[:8]}.csv",
            "text/csv",
        )
    else:
        st.info("No customer records found for this batch.")

st.divider()


st.subheader("Drift Alerts")
alerts = get_drift_alerts(limit=100)

if not alerts:
    st.success("No drift alerts recorded yet")
else:
    adf = pd.DataFrame(alerts)
    adf["alerted_at"] = pd.to_datetime(adf["alerted_at"]).dt.strftime("%Y-%m-%d %H:%M")

    def style_severity(v):
        if v == "HIGH":
            return f"background-color:{LIGHT_RED};color:{BLOOD_RED};font-weight:700"
        if v == "MEDIUM":
            return f"background-color:{LIGHT_YELLOW};color:{ORANGE};font-weight:700"
        return f"color:{GREY}"

    alert_cols = [
        "alerted_at",
        "feature",
        "alert_type",
        "severity",
        "training_value",
        "upload_value",
        "deviation",
    ]
    alert_cols = [c for c in alert_cols if c in adf.columns]

    st.dataframe(
        adf[alert_cols].style.map(style_severity, subset=["severity"]),
        width="content",
        height=260,
        hide_index=True,
    )

st.divider()

st.subheader("Training Reference Stats")
ts = TRAINING_STATS

if ts:
    numeric_stats = {k: v for k, v in ts.items() if isinstance(v, dict) and "mean" in v}
    if numeric_stats:
        rows = [
            {
                "Feature": feat,
                "Mean": round(s["mean"], 3),
                "Std": round(s["std"], 3),
                "Min": round(s["min"], 3),
                "Max": round(s["max"], 3),
                "Missing %": f'{s["missing_rate"]:.2%}',
            }
            for feat, s in numeric_stats.items()
        ]
        st.dataframe(pd.DataFrame(rows), width="content", hide_index=True)

    if "churn_rate" in ts:
        st.info(
            f"Training churn rate: **{ts['churn_rate']:.1%}** - baseline for drift comparison"
        )
else:
    st.info("Training stats unavailable - run src/extract_training_stats.py")

st.divider()

st.subheader("Application Logs (Last 100 Lines)")
if LOG_FILE.exists():
    lines = open(LOG_FILE).readlines()
    st.code("".join(lines[-100:]), language="text")
else:
    st.info("Logs appear after first prediction")
