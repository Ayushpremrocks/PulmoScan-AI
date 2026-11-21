import altair as alt
import pandas as pd
import streamlit as st

import config
import metrics
import ui

ui.hero("📈 Model evaluation", "How the threshold trades missed cases against false alarms")


@st.cache_data
def load_predictions(path: str, modified: float) -> pd.DataFrame:
    # `modified` makes the cache refresh when the file is regenerated.
    return pd.read_csv(path)


path = config.EVAL_PREDICTIONS_PATH
predictions = load_predictions(str(path), path.stat().st_mtime) if path.exists() else None
model_keys = list(config.MODELS)

has_data = predictions is not None and {"label", *model_keys} <= set(predictions.columns)

if not has_data:
    st.info(
        "Generate per-image test-set predictions to make this page interactive. Download the "
        "[Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), "
        "then run:\n\n"
        "```\npython scripts/evaluate.py --data-dir path/to/chest_xray/test\n```\n\n"
        f"This writes `{path.relative_to(config.BASE_DIR).as_posix()}`. Reload this page afterwards.",
        title="No evaluation data yet",
        icon=":material/info:",
    )
    st.subheader("Results reported in the training notebook")
    st.table(
        pd.DataFrame(
            {
                spec.name: {
                    "Threshold": f"{spec.reported.threshold:.2f}",
                    "Sensitivity": f"{spec.reported.sensitivity:.1%}",
                    "Specificity": f"{spec.reported.specificity:.0%}",
                    "Precision": f"{spec.reported.precision:.0%}",
                    "ROC-AUC": f"{spec.reported.roc_auc:.3f}",
                }
                for spec in config.MODELS.values()
            }
        ),
        alt="Test-set metrics reported in the training notebook",
    )
else:
    labels = predictions["label"].to_numpy()
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    st.caption(f"Test set: {len(labels)} X-rays ({n_pos} pneumonia, {n_neg} normal).")

    with st.sidebar:
        thresholds = {key: ui.threshold_slider(key, key=f"eval_{key}_threshold") for key in model_keys}

    # ---- Metrics at the chosen thresholds ----
    st.header("At your chosen thresholds")
    for col, key in zip(st.columns(len(model_keys)), model_keys):
        spec = config.MODELS[key]
        c = metrics.confusion_at(labels, predictions[key].to_numpy(), thresholds[key])
        with col, st.container(border=True):
            st.subheader(spec.name)
            st.caption(f"Threshold {thresholds[key]:.2f}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sensitivity", f"{c.sensitivity:.1%}")
            m2.metric("Specificity", f"{c.specificity:.1%}")
            m3.metric("Precision", f"{c.precision:.1%}")
            m4.metric("Accuracy", f"{c.accuracy:.1%}")
            st.table(
                pd.DataFrame(
                    {"Predicted normal": [c.tn, c.fn], "Predicted pneumonia": [c.fp, c.tp]},
                    index=["Actually normal", "Actually pneumonia"],
                ),
                alt=f"{spec.name} confusion matrix",
            )
            st.markdown(f"**{c.fn}** missed cases · **{c.fp}** false alarms")

    # ---- ROC curve ----
    st.header("ROC curve")
    roc_frames, points = [], []
    for key in model_keys:
        spec = config.MODELS[key]
        probs = predictions[key].to_numpy()
        auc = metrics.roc_auc(labels, probs)
        roc = metrics.roc_curve(labels, probs)
        roc["model"] = f"{spec.name} (AUC {auc:.3f})"
        roc_frames.append(roc)
        c = metrics.confusion_at(labels, probs, thresholds[key])
        points.append({"model": roc["model"].iloc[0], "fpr": 1 - c.specificity, "tpr": c.sensitivity, "threshold": thresholds[key]})

    roc_df = pd.concat(roc_frames, ignore_index=True)
    base_color = alt.Color("model:N", title="Model")
    lines = (
        alt.Chart(roc_df)
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X("fpr:Q", title="False positive rate (1 − specificity)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("tpr:Q", title="Sensitivity", scale=alt.Scale(domain=[0, 1])),
            color=base_color,
            tooltip=[
                "model:N",
                alt.Tooltip("threshold:Q", format=".3f"),
                alt.Tooltip("tpr:Q", title="Sensitivity", format=".1%"),
                alt.Tooltip("fpr:Q", title="False positive rate", format=".1%"),
            ],
        )
    )
    markers = (
        alt.Chart(pd.DataFrame(points))
        .mark_point(size=140, filled=True)
        .encode(
            x="fpr:Q",
            y="tpr:Q",
            color=base_color,
            tooltip=["model:N", alt.Tooltip("threshold:Q", title="Your threshold", format=".2f")],
        )
    )
    st.altair_chart(lines + markers, alt="ROC curves for both models with markers at the selected thresholds")
    st.caption("Dots mark the thresholds chosen in the sidebar. Up and to the left is better.")

    # ---- Fair comparison ----
    st.header("Fair comparison at matched sensitivity")
    st.markdown(
        "Comparing models at different thresholds can make either one look better. Here each model gets "
        "the highest threshold that still reaches the target sensitivity, so you can compare false alarms directly."
    )
    target = st.slider("Target sensitivity", 0.90, 1.0, 0.98, 0.005, format="%.3f", key="eval_target_sensitivity")
    rows = []
    for key in model_keys:
        probs = predictions[key].to_numpy()
        t = metrics.threshold_for_sensitivity(labels, probs, target)
        c = metrics.confusion_at(labels, probs, t)
        rows.append(
            {
                "Model": config.MODELS[key].name,
                "Threshold": round(t, 3),
                "Sensitivity": c.sensitivity,
                "Specificity": c.specificity,
                "Missed cases": c.fn,
                "False alarms": c.fp,
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={
            "Sensitivity": st.column_config.NumberColumn(format="percent"),
            "Specificity": st.column_config.NumberColumn(format="percent"),
        },
        alt="Specificity and false alarms for each model at the target sensitivity",
    )
    st.caption(
        "Choosing a threshold on the test set itself is optimistic. In a real study you'd fix it on a validation set first."
    )
