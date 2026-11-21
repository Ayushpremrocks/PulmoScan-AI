import streamlit as st

import config
import ui

ui.hero("⚖️ Compare models", "Run both models on the same X-ray")

with st.sidebar:
    thresholds = {
        key: ui.threshold_slider(key, key=f"compare_{key}_threshold") for key in config.MODELS
    }

picked = ui.image_input(key_prefix="compare")
if picked is not None:
    image_bytes, caption = picked
    image = ui.checked_image(image_bytes, key_prefix="compare")
    if image is not None:
        with st.spinner("Running both models…"):
            probabilities = {key: ui.pneumonia_probability(key, image_bytes) for key in config.MODELS}
        verdicts = {key: probabilities[key] > thresholds[key] for key in config.MODELS}

        col_image, *col_models = st.columns(1 + len(config.MODELS))
        with col_image:
            st.image(ui.display_copy(image), caption=caption, width="stretch", alt="Chest X-ray being analyzed")
        for col, key in zip(col_models, config.MODELS):
            with col:
                ui.render_result(key, probabilities[key], thresholds[key])

        if len(set(verdicts.values())) == 1:
            st.info("Both models agree on this X-ray.", icon=":material/check_circle:")
        else:
            flagged = [config.MODELS[k].name for k, v in verdicts.items() if v]
            st.warning(
                f"The models disagree: only {', '.join(flagged)} flags pneumonia. "
                "In a screening workflow a disagreement would be sent for human review.",
                icon=":material/compare_arrows:",
            )

        st.subheader("Where each model looks")
        if st.toggle("Show Grad-CAM overlays", value=True, key="compare_show_gradcam"):
            for col, key in zip(st.columns(len(config.MODELS)), config.MODELS):
                with col:
                    ui.render_gradcam(key, image_bytes, image, side_by_side=False)
            st.caption(ui.GRADCAM_EXPLANATION)
