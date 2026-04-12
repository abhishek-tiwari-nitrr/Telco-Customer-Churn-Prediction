import streamlit as st
from src.config import ASSETS_DIR

st.set_page_config(page_title="EDA Report", layout="wide")

st.title("📜 Exploratory Data Analysis Report")
st.divider()

st.header("Dataset Overview")
st.image(ASSETS_DIR / "slide_1.png", width="stretch")

st.header("Data Quality Assessment")
st.image(ASSETS_DIR / "slide_2.png", width="stretch")

st.header("Descriptive Statistics")
st.image(ASSETS_DIR / "slide_3.png", width="stretch")

st.header("Univariate Analysis")
col1, col2 = st.columns(2)
with col1:
    st.image(ASSETS_DIR / "slide_4.png", width="stretch")
with col2:
    st.image(ASSETS_DIR / "slide_5.png", width="stretch")


st.header("Bivariate Analysis")
col1, col2 = st.columns(2)
with col1:
    st.image(ASSETS_DIR / "slide_6.png", width="stretch")
with col2:
    st.image(ASSETS_DIR / "slide_7.png", width="stretch")
col1, col2 = st.columns(2)
with col1:
    st.image(ASSETS_DIR / "slide_8.png", width="stretch")

st.header("Multivariate Analysis")
st.image(ASSETS_DIR / "slide_9.png", width="stretch")

st.header("Key Insights & Next Steps")
st.image(ASSETS_DIR / "slide_10.png", width="stretch")
