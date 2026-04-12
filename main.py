import streamlit as st
from src.database import init_db
from src.model import churn_model
from src.logger import get_logger
from src.config import APP_TITLE, APP_ICON

logger = get_logger(__name__)

# MUST BE FIRST
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
)


@st.cache_resource
def startup():
    init_db()
    churn_model.load()
    logger.info("App startup complete")
    return True


startup()

home_page = st.Page("pages/00_Home.py", title="Home", icon="🏠")
eda_page = st.Page("pages/01_EDA_Report.py", title="EDA Report", icon="📜")
live_predictor_page = st.Page(
    "pages/02_Live_Predictor.py", title="Live Predictor", icon="🚀"
)
monitor_page = st.Page("pages/03_Monitoring.py", title="Monitoring", icon="📺")

pg = st.navigation([home_page, eda_page, live_predictor_page, monitor_page])
pg.run()
