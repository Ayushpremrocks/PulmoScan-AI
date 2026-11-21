import pandas as pd
import streamlit as st

import config
import ui

ui.hero("🫁 PulmoScan AI", "Pneumonia screening from chest X-rays: a comparison of two deep learning models")

st.markdown(
    "**Author:** Ayush Prem  \n"
    "**Goal:** a screening tool that misses as few pneumonia cases as possible (few false negatives), "
    "while being honest about the cost in false alarms."
)

col_goal, col_tradeoff = st.columns(2)
with col_goal:
    st.info(
        "In screening, **sensitivity (recall)** matters most: sending a sick patient home is dangerous, "
        "while a false alarm leads to a follow-up check.",
        title="Why sensitivity",
        icon=":material/pulmonology:",
    )
with col_tradeoff:
    st.warning(
        "Raising sensitivity always costs **specificity**: more healthy patients get flagged. "
        "A model is only better if it catches more cases *at the same false-alarm rate*.",
        title="The trade-off",
        icon=":material/compare_arrows:",
    )

st.header("Benchmark results")
st.markdown(
    f"Both models were fine-tuned from ImageNet weights and tested on the same held-out set of "
    f"**{config.TEST_SET_SIZE} X-rays** ({config.TEST_SET_PNEUMONIA} pneumonia, {config.TEST_SET_NORMAL} normal)."
)

for col, spec in zip(st.columns(len(config.MODELS)), config.MODELS.values()):
    r = spec.reported
    missed = round((1 - r.sensitivity) * config.TEST_SET_PNEUMONIA)
    false_alarms = round((1 - r.specificity) * config.TEST_SET_NORMAL)
    with col, st.container(border=True):
        st.subheader(spec.name)
        st.caption(f"{spec.role} · threshold {r.threshold:.2f}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Sensitivity", f"{r.sensitivity:.1%}", help="Share of pneumonia cases caught.")
        m2.metric("Specificity", f"{r.specificity:.0%}", help="Share of healthy patients correctly cleared.")
        m3.metric("ROC-AUC", f"{r.roc_auc:.3f}", help="Threshold-independent ranking quality (1.0 is perfect).")
        st.markdown(
            f"Missed about **{missed} of {config.TEST_SET_PNEUMONIA}** pneumonia cases and flagged about "
            f"**{false_alarms} of {config.TEST_SET_NORMAL}** healthy patients."
        )

table = pd.DataFrame(
    {
        spec.name: {
            "Decision threshold": f"{spec.reported.threshold:.2f}",
            "Sensitivity (recall)": f"{spec.reported.sensitivity:.1%}",
            "Specificity": f"{spec.reported.specificity:.0%}",
            "Precision": f"{spec.reported.precision:.0%}",
            "ROC-AUC": f"{spec.reported.roc_auc:.3f}",
        }
        for spec in config.MODELS.values()
    }
)
st.table(table, alt="Test-set metrics for each model")

st.markdown(
    "**How to read this.** DenseNet121 is the better *screening* choice: at its threshold it misses only "
    "one pneumonia case. That comes from a low threshold (0.30), and it flags about half of healthy patients. "
    "The two models rank X-rays about equally well (ROC-AUC 0.961 vs 0.964), so a fair comparison fixes the "
    "sensitivity and compares false alarms. The **Model evaluation** page does exactly that."
)
st.caption(
    "These numbers come from the training notebook. The thresholds were tuned while looking at the test set, "
    "so they are slightly optimistic."
)

st.header("Try it")
c1, c2, c3, c4 = st.columns(4)
with c1, st.container(border=True, height="stretch"):
    st.markdown("**DenseNet121**  \nThe high-sensitivity champion, with Grad-CAM.")
    st.page_link("app_pages/densenet121.py", label="Open", icon=":material/trophy:")
with c2, st.container(border=True, height="stretch"):
    st.markdown("**ResNet50**  \nThe baseline model, with Grad-CAM.")
    st.page_link("app_pages/resnet50.py", label="Open", icon=":material/bar_chart:")
with c3, st.container(border=True, height="stretch"):
    st.markdown("**Compare**  \nBoth models side by side on one X-ray.")
    st.page_link("app_pages/compare.py", label="Open", icon=":material/compare_arrows:")
with c4, st.container(border=True, height="stretch"):
    st.markdown("**Model evaluation**  \nConfusion matrix and ROC curve at any threshold.")
    st.page_link("app_pages/evaluation.py", label="Open", icon=":material/query_stats:")

st.divider()
st.caption("Tech stack: PyTorch · Torchvision · Streamlit · OpenCV (Grad-CAM) · Altair")
