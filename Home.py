import streamlit as st

import ui

st.set_page_config(page_title="PulmoScan AI", page_icon="🫁", layout="wide")
ui.load_styles()

page = st.navigation(
    [
        st.Page(
            "app_pages/overview.py",
            title="Overview",
            icon=":material/home:",
            default=True,
            help="Project summary and benchmark results for both models.",
        ),
        st.Page(
            "app_pages/densenet121.py",
            title="DenseNet121",
            icon=":material/trophy:",
            help="Analyze an X-ray with the high-sensitivity champion model, with a Grad-CAM heatmap.",
        ),
        st.Page(
            "app_pages/resnet50.py",
            title="ResNet50",
            icon=":material/bar_chart:",
            help="Analyze an X-ray with the ResNet50 baseline model, with a Grad-CAM heatmap.",
        ),
        st.Page(
            "app_pages/compare.py",
            title="Compare",
            icon=":material/compare_arrows:",
            help="Run both models on the same X-ray and compare predictions and heatmaps.",
        ),
        st.Page(
            "app_pages/evaluation.py",
            title="Model evaluation",
            icon=":material/query_stats:",
            help="Test-set confusion matrix, ROC curve and a fair matched-sensitivity comparison.",
        ),
    ],
    position="top",
)

page.run()

st.divider()
st.caption(
    ":material/info: For education and research only. PulmoScan AI is not a medical device "
    "and is not a substitute for diagnosis by a qualified clinician."
)
