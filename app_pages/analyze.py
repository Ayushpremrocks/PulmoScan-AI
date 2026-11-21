from datetime import datetime

import pandas as pd
import streamlit as st

import config
import inference
import ui

ui.page_header(
    "Analyze an X-ray",
    ":material/pulmonology:",
    "Pick a sample or upload a chest X-ray, then review the prediction and where the model looked.",
)

left, right = st.columns([1, 2], gap="large")

with left:
    with st.container(border=True):
        picked = ui.image_input("analyze")
        choice = ui.model_picker("analyze_model")
    with st.expander("Settings", icon=":material/tune:"):
        thresholds = {k: ui.threshold_slider(k, key=f"analyze_{k}_threshold") for k in config.MODELS}
        opacity = st.slider("Heatmap opacity", 0.1, 0.9, 0.4, 0.05, key="analyze_opacity")

with right:
    if picked is None:
        st.info("Upload a chest X-ray (JPEG or PNG) to start.", icon=":material/upload:")
    elif (image := ui.checked_image(picked[0], "analyze")) is not None:
        image_bytes, name = picked
        keys = ui.selected_models(choice)

        with st.spinner("Analyzing…"):
            probs = {k: ui.pneumonia_probability(k, image_bytes) for k in keys}

        for col, k in zip(st.columns(len(keys)), keys):
            with col:
                ui.result_card(k, probs[k], thresholds[k])

        if len({ui.is_pneumonia(probs[k], thresholds[k]) for k in keys}) > 1:
            st.warning(
                "The models disagree. In a screening workflow this X-ray would go to a radiologist.",
                icon=":material/compare_arrows:",
            )

        view = st.segmented_control(
            "View",
            ["heatmap", "original", "enhanced"],
            format_func={
                "heatmap": ":material/layers: Heatmap",
                "original": ":material/image: Original",
                "enhanced": ":material/contrast: Enhanced contrast",
            }.get,
            default="heatmap",
            required=True,
            key="analyze_view",
        )
        shown = ui.display_copy(image)
        overlays = {}
        if view == "original":
            st.image(shown, caption=name, width="stretch", alt="Chest X-ray")
        elif view == "enhanced":
            st.image(
                inference.enhance_contrast(shown),
                caption=f"{name} · contrast enhanced (CLAHE, display only)",
                width="stretch",
                alt="Chest X-ray with enhanced contrast",
            )
        else:
            with st.spinner("Generating heatmap…"):
                overlays = {
                    k: inference.overlay_heatmap(ui.gradcam_map(k, image_bytes), shown, alpha=opacity) for k in keys
                }
            for col, k in zip(st.columns(len(keys)), keys):
                col.image(
                    overlays[k],
                    caption=f"{config.MODELS[k].name} Grad-CAM",
                    width="stretch",
                    alt=f"{config.MODELS[k].name} Grad-CAM heatmap over the X-ray",
                )
            st.caption(ui.GRADCAM_EXPLANATION)

        file_id = ui.image_id(image_bytes)
        with st.container(horizontal=True):
            st.download_button(
                "Report",
                ui.build_report(name, probs, thresholds),
                file_name=f"pulmoscan_report_{file_id}.txt",
                mime="text/plain",
                icon=":material/download:",
                on_click="ignore",
            )
            for k, overlay in overlays.items():
                st.download_button(
                    f"{config.MODELS[k].name} heatmap",
                    inference.to_png_bytes(overlay),
                    file_name=f"pulmoscan_{k}_heatmap_{file_id}.png",
                    mime="image/png",
                    icon=":material/download:",
                    on_click="ignore",
                )

        now = datetime.now().strftime("%H:%M:%S")
        for k in keys:
            ui.record_history(
                {
                    "Time": now,
                    "Image": name,
                    "Image ID": file_id,
                    "Model": config.MODELS[k].name,
                    "Score": probs[k],
                    "Threshold": thresholds[k],
                    "Result": ui.verdict(probs[k], thresholds[k]),
                }
            )

history = st.session_state.get("history", [])
if history:
    with st.expander(f"Session history ({len(history)})", icon=":material/history:"):
        df = pd.DataFrame(history)
        st.dataframe(
            df.iloc[::-1],
            hide_index=True,
            column_order=["Time", "Image", "Model", "Score", "Threshold", "Result"],
            column_config={
                "Score": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "Threshold": st.column_config.NumberColumn(format="%.2f"),
            },
            alt="Analyses run in this session",
        )
        with st.container(horizontal=True):
            st.download_button(
                "Download CSV",
                df.drop(columns="Image ID").to_csv(index=False),
                file_name="pulmoscan_history.csv",
                mime="text/csv",
                icon=":material/download:",
                on_click="ignore",
            )
            st.button("Clear", icon=":material/delete:", on_click=ui.clear_history, type="tertiary")
