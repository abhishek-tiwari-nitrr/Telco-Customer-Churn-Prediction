import io
import pandas as pd
import streamlit as st
from src.database import init_db, log_batch, log_predictions, log_drift_alert
from src.model import churn_model, PredictionOutput
from src.logger import get_logger
from src.config import (
    MODEL_VERSION,
    LIGHT_RED,
    RED,
    LIGHT_YELLOW,
    ORANGE,
    LIGHT_GREEN,
    GREEN,
    BLOOD_RED,
)
from src.preprocessing import validate_and_clean
from src.monitoring import detect_drift, DriftReport

logger = get_logger(__name__)

st.set_page_config(
    page_title="Live Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()


@st.cache_resource
def load_model():
    churn_model.load()
    return churn_model


@st.cache_data
def read_upload(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    return pd.read_csv(buf) if filename.endswith(".csv") else pd.read_excel(buf)


@st.cache_data
def run_drift(_df: pd.DataFrame) -> DriftReport:
    return detect_drift(_df)


@st.cache_data
def run_predictions(_cleaned: pd.DataFrame, customer_ids) -> PredictionOutput:
    return churn_model.predict(_cleaned, customer_ids)


def style_risk(v):
    if "High" in str(v):
        return f"background-color:{LIGHT_RED};color:{RED};font-weight:700"
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


def style_prob(v):
    try:
        f = float(v)
        if f >= 0.7:
            return f"color:{BLOOD_RED};font-weight:700"
        if f >= 0.4:
            return f"color:{ORANGE};font-weight:700"
        return f"color:{GREEN};font-weight:700"
    except:
        return ""


st.title("🚀 Live Churn Predictor")
st.caption("Upload CSV or Excel -> validate -> predict -> download")
st.divider()

try:
    model = load_model()
    st.success("Model loaded and ready", icon="✅")
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

st.subheader("01 | Upload File")
uploaded = st.file_uploader(
    "Upload CSV or Excel", type=["csv", "xlsx", "xls"], label_visibility="collapsed"
)
with st.expander("Expected columns"):
    st.code(
        "customerID (optional), gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, \nOnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, \nPaymentMethod, MonthlyCharges, TotalCharges"
    )

if not uploaded:
    st.info("Upload a CSV or Excel file to begin")
    st.stop()

try:
    raw = uploaded.read()
    df = read_upload(raw, uploaded.name)
    logger.info(f"Uploaded: {uploaded.name} | rows={len(df)}")
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.success(f"Loaded **{uploaded.name}** - {len(df):,} rows & {len(df.columns)} columns")

st.subheader("02 | Validation")
cleaned, val = validate_and_clean(df)

for w in val.warnings:
    st.warning(w)

if not val.is_valid:
    for e in val.errors:
        st.error(e)
    log_batch(
        uploaded.name,
        len(df),
        0,
        0,
        0,
        0,
        0,
        MODEL_VERSION,
        0,
        "failed",
        "; ".join(val.errors),
    )
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Total Rows", f"{val.original_rows:,}")
c2.metric("Valid Rows", f"{val.cleaned_rows:,}")
c3.metric("Warnings", str(len(val.warnings)))
st.success("Validation passed - data ready for prediction")

st.subheader("03 | Predict")
if st.button("⚡ Run Churn Prediction", type="primary"):
    with st.spinner("Running predictions..."):
        try:
            out = run_predictions(cleaned, val.customer_ids)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            logger.error(f"Prediction error: {e}")
            st.stop()

        drift = run_drift(cleaned)
        batch_id = log_batch(
            uploaded.name,
            out.total,
            out.predicted_churn,
            out.churn_rate,
            out.high_risk,
            out.medium_risk,
            out.low_risk,
            MODEL_VERSION,
            out.processing_ms,
        )

        records = [
            {
                "customer_id": str(r.get("customer_id", "")),
                "churn_prediction": r["churn_prediction"],
                "churn_probability": float(r["churn_probability"]),
                "risk_tier": r["risk_tier"],
                "tenure": float(r.get("tenure", 0) or 0),
                "monthlycharges": float(r.get("monthlycharges", 0) or 0),
                "totalcharges": float(r.get("totalcharges", 0) or 0),
                "contract": str(r.get("contract", "")),
                "internetservice": str(r.get("internetservice", "")),
                "paymentmethod": str(r.get("paymentmethod", "")),
            }
            for _, r in out.result_df.iterrows()
        ]
        log_predictions(batch_id, records)

        for a in drift.alerts:
            log_drift_alert(
                batch_id,
                a.feature,
                a.alert_type,
                a.training_value,
                a.upload_value,
                a.deviation,
                a.severity,
            )

        st.session_state.update(
            {
                "pred_out": out,
                "pred_drift": drift,
                "pred_bid": batch_id,
                "pred_fname": uploaded.name,
            }
        )

if "pred_out" not in st.session_state:
    st.info("Run a prediction to see results here.")
    st.stop()

out = st.session_state["pred_out"]
drift = st.session_state["pred_drift"]
bid = st.session_state["pred_bid"]

st.subheader("04 | Summary")
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Total", f"{out.total:,}")
c2.metric("Churned", f"{out.predicted_churn:,}")
c3.metric("Churn Rate", f"{out.churn_rate:.1%}")
c4.metric("High Risk", f"{out.high_risk:,}")
c5.metric("Medium Risk", f"{out.medium_risk:,}")
c6.metric("Low Risk", f"{out.low_risk:,}")
c7.metric("Time", f"{out.processing_ms}ms")
st.caption(f"batch_id: {bid}")

if drift.alerts:
    st.subheader("05 | Drift Alerts")
    for a in drift.alerts:
        if a.severity == "HIGH":
            st.error(f"HIGH | {a.feature} | {a.message}")
        elif a.severity == "MEDIUM":
            st.warning(f"MEDIUM | {a.feature} | {a.message}")
        else:
            st.info(f"LOW | {a.feature} | {a.message}")
else:
    st.success("No data drift detected - upload matches training distribution")

st.subheader("06 | Results")
res = out.result_df

f1, f2, f3 = st.columns([2, 2, 4])
cf = f1.selectbox("Filter Churn", ["All", "Yes", "No"])
rf = f2.selectbox("Filter Risk", ["All", "High Risk", "Medium Risk", "Low Risk"])
sr = f3.text_input("Search customer ID", "")

filt = res.copy()
if cf != "All":
    filt = filt[filt["churn_prediction"] == cf]
if rf != "All":
    filt = filt[filt["risk_tier"].str.contains(rf.split()[0])]
if sr and "customer_id" in filt.columns:
    filt = filt[filt["customer_id"].astype(str).str.contains(sr, case=False)]

st.caption(f"Showing {len(filt):,} of {out.total:,} rows")

st.dataframe(
    filt.style.map(style_risk, subset=["risk_tier"])
    .map(style_churn, subset=["churn_prediction"])
    .map(style_prob, subset=["churn_probability"])
    .format({"churn_probability": "{:.1%}"}),
    width="content",
    height=420,
    hide_index=True,
)

st.subheader("07 | Download")
fname = st.session_state["pred_fname"].rsplit(".", 1)[0]
c1, c2 = st.columns(2)

with c1:
    st.download_button(
        "Full Results (CSV)",
        res.to_csv(index=False).encode(),
        f"{fname}_predictions.csv",
        "text/csv",
        width="stretch",
    )
with c2:
    hr = res[res["risk_tier"].str.contains("High")]
    st.download_button(
        f"High Risk Only ({out.high_risk:,})",
        hr.to_csv(index=False).encode(),
        f"{fname}_high_risk.csv",
        "text/csv",
        width="stretch",
    )
