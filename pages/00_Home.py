import streamlit as st
from src.database import get_summary_stats

stats = get_summary_stats()

st.caption("END-TO-END ML PROJECT - V1.0.0")
st.title("🛰️ Telco Customer Churn Prediction System")

st.write(
    "A production-grade ML application — EDA, 25-model benchmark, SMOTE, hyperparameter tuning, SHAP explainability, drift detection and customer-level audit logging."
)

cols = st.columns(5)
with cols[0]:
    st.badge("AdaBoost Classifier", icon="🤖")
with cols[1]:
    st.badge("SMOTE Oversampling", icon="⚖️")
with cols[2]:
    st.badge("AUC-ROC 0.8403", icon="📈")
with cols[3]:
    st.badge("Recall 0.7727", icon="🎯")
with cols[4]:
    st.badge("scikit-learn + SHAP", icon="🔍")

st.caption("Prepared by: Abhishek Tiwari")
st.divider()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("🧑‍🤝‍🧑 Training Customers", "7,043")
c2.metric("🗂️ Features", "21")
c3.metric("📉 Training Churn Rate", "26.5%")
c4.metric("📊 Best AUC-ROC", "0.8403")
c5.metric("🎯 Best Recall", "0.7727")
c6.metric("🔄 Batches Run", str(stats.get("total_batches", 0)))
