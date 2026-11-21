import pandas as pd
import streamlit as st

import config
import ui

ui.page_header(
    "About PulmoScan AI",
    ":material/info:",
    "Pneumonia screening from chest X-rays, comparing two fine-tuned CNNs.",
)

st.markdown(
    "PulmoScan AI flags chest X-rays that may show pneumonia. It's built as a **screening** tool, so it "
    "prioritises **sensitivity**: missing a sick patient is worse than an extra follow-up check. "
    "Each prediction comes with a **Grad-CAM heatmap** showing which regions drove it."
)

st.subheader("Results on the test set")
st.caption(
    f"{config.TEST_SET_SIZE} held-out X-rays ({config.TEST_SET_PNEUMONIA} pneumonia, "
    f"{config.TEST_SET_NORMAL} normal)."
)
for col, spec in zip(st.columns(len(config.MODELS)), config.MODELS.values()):
    r = spec.reported
    missed = round((1 - r.sensitivity) * config.TEST_SET_PNEUMONIA)
    false_alarms = round((1 - r.specificity) * config.TEST_SET_NORMAL)
    with col, st.container(border=True):
        st.markdown(f"**{spec.name}**")
        st.caption(f"{spec.role} · threshold {r.threshold:.2f}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Sensitivity", f"{r.sensitivity:.1%}", help="Share of pneumonia cases caught.")
        m2.metric("Specificity", f"{r.specificity:.0%}", help="Share of healthy patients correctly cleared.")
        m3.metric("ROC-AUC", f"{r.roc_auc:.3f}", help="Ranking quality across all thresholds (1.0 is perfect).")
        st.caption(f"Missed ~{missed} of {config.TEST_SET_PNEUMONIA} cases · ~{false_alarms} of {config.TEST_SET_NORMAL} false alarms")

st.markdown(
    "DenseNet121 is the screening pick: at its low threshold it missed just one case, at the cost of flagging "
    "about half of healthy patients. Both models rank X-rays about equally well (similar ROC-AUC), so most of the "
    "difference comes from the threshold. The **Evaluation** page compares them at matched sensitivity."
)

st.subheader("How it works")
st.table(
    pd.DataFrame(
        {
            "Step": ["Data", "Models", "Training", "Explainability", "App"],
            "Details": [
                "Kaggle Chest X-Ray Images (Pneumonia): 5,216 training and 624 test X-rays",
                "DenseNet121 and ResNet50, ImageNet-pretrained, new 512-unit classification head",
                "Class-weighted cross-entropy, augmentation (flip, rotation), then fine-tuning of the last block",
                "Grad-CAM on the final feature block (DenseNet `norm5`, ResNet `layer4`)",
                "Streamlit, PyTorch on CPU, cached models and predictions",
            ],
        }
    ),
    hide_index=True,
    alt="Project pipeline",
)

st.subheader("Limitations")
st.markdown(
    "- The decision thresholds were tuned while looking at the test set, so reported metrics are slightly optimistic.\n"
    "- Training and validation were split by image, not by patient, so the same patient may appear in both.\n"
    "- In this dataset, normal and pneumonia X-rays differ in size and framing. Check heatmaps for attention "
    "outside the lungs, which suggests the model is using shortcuts.\n"
    "- Scores are softmax outputs, not calibrated probabilities.\n"
    "- Trained on paediatric X-rays from one source, so it may not generalise to adults or other hospitals."
)

st.subheader("Author")
st.markdown(
    "**Ayush Prem** · [GitHub](https://github.com/ayushpremrocks) · "
    "[LinkedIn](https://www.linkedin.com/in/ayushpremrocks) · "
    "[Source code](https://github.com/ayushpremrocks/PulmoScan-AI)"
)
