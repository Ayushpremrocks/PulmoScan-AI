import streamlit as st

import config
import ui

st.set_page_config(page_title="PulmoScan AI", page_icon=":material/pulmonology:", layout="wide")
st.logo(str(config.BASE_DIR / "assets" / "logo.svg"), size="large")

page = st.navigation(
    [
        st.Page(
            "app_pages/analyze.py",
            title="Analyze",
            icon=":material/pulmonology:",
            default=True,
            help="Analyze one chest X-ray with either or both models, with a Grad-CAM heatmap and a downloadable report.",
        ),
        st.Page(
            "app_pages/batch.py",
            title="Batch",
            icon=":material/stacks:",
            help="Screen many X-rays at once and download the results as CSV.",
        ),
        st.Page(
            "app_pages/evaluation.py",
            title="Evaluation",
            icon=":material/analytics:",
            help="Test-set confusion matrix, ROC curve and a fair matched-sensitivity comparison.",
        ),
        st.Page(
            "app_pages/about.py",
            title="About",
            icon=":material/info:",
            help="Project summary, benchmark results, method and limitations.",
        ),
    ],
    position="top",
)

page.run()

st.divider()
st.caption(f":material/info: {ui.DISCLAIMER}")
