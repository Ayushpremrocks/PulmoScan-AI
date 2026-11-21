import pandas as pd
import streamlit as st
from PIL import UnidentifiedImageError

import config
import inference
import ui

ui.page_header(
    "Batch analysis",
    ":material/stacks:",
    "Screen many X-rays at once and export the results as CSV.",
)

with st.container(border=True):
    source = st.segmented_control(
        "Images",
        ["samples", "upload"],
        format_func={
            "samples": ":material/photo_library: All 20 sample cases",
            "upload": ":material/upload: Upload files",
        }.get,
        default="samples",
        required=True,
        key="batch_source",
    )
    files: list[tuple[str, bytes, str | None]] = []  # (name, bytes, true label)
    if source == "upload":
        uploads = st.file_uploader(
            "Chest X-rays",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_upload",
            label_visibility="collapsed",
        )
        files = [(f.name, f.getvalue(), None) for f in uploads or []]
    else:
        files = [
            (name, (config.SAMPLES_DIR / name).read_bytes(), ui.sample_true_label(name))
            for name in ui.sample_files()
        ]

    choice = ui.model_picker("batch_model")
    with st.expander("Thresholds", icon=":material/tune:"):
        thresholds = {k: ui.threshold_slider(k, key=f"batch_{k}_threshold") for k in config.MODELS}

    run = st.button(
        f"Analyze {len(files)} image{'s' if len(files) != 1 else ''}",
        type="primary",
        icon=":material/play_arrow:",
        disabled=not files,
        key="batch_run",
    )

if run:
    keys = ui.selected_models(choice)
    rows, skipped = [], []
    progress = st.progress(0.0, text="Starting…")
    for i, (name, data, truth) in enumerate(files, start=1):
        progress.progress(i / len(files), text=f"Analyzing {name} ({i}/{len(files)})")
        try:
            image = inference.open_image(data)
        except (UnidentifiedImageError, OSError):
            skipped.append(name)
            continue
        row = {
            "Preview": ui.thumbnail_data_uri(image),
            "Image": ui.sample_label(name) if truth else name,
            "Truth": truth,
            "Warnings": " ".join(inference.xray_warnings(image)) or None,
        }
        for k in keys:
            p = ui.pneumonia_probability(k, data)
            row[f"{config.MODELS[k].name} score"] = p
            row[f"{config.MODELS[k].name} result"] = ui.verdict(p, thresholds[k])
        rows.append(row)
    progress.empty()
    st.session_state["batch_results"] = {"rows": rows, "skipped": skipped, "keys": keys}

results = st.session_state.get("batch_results")
if results and results["rows"]:
    df = pd.DataFrame(results["rows"])
    keys = results["keys"]
    has_truth = df["Truth"].notna().all()

    st.subheader("Summary")
    for col, k in zip(st.columns(len(keys)), keys):
        name = config.MODELS[k].name
        flagged = int((df[f"{name} result"] == "Pneumonia suspected").sum())
        with col, st.container(border=True):
            st.markdown(f"**{name}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Flagged", f"{flagged} / {len(df)}")
            if has_truth:
                correct = (df[f"{name} result"].str.startswith("Pneumonia") == (df["Truth"] == "Pneumonia"))
                missed = int(((df["Truth"] == "Pneumonia") & ~df[f"{name} result"].str.startswith("Pneumonia")).sum())
                m2.metric("Correct", f"{correct.mean():.0%}")
                m3.metric("Missed cases", missed)
            else:
                m2.metric("Normal", len(df) - flagged)

    if results["skipped"]:
        st.warning(f"Skipped unreadable files: {', '.join(results['skipped'])}", icon=":material/error:")

    column_config = {
        "Preview": st.column_config.ImageColumn(width="small"),
        "Truth": st.column_config.TextColumn(help="Known diagnosis (sample cases only)"),
    }
    for k in keys:
        column_config[f"{config.MODELS[k].name} score"] = st.column_config.ProgressColumn(
            format="percent", min_value=0, max_value=1
        )
    st.dataframe(
        df.dropna(axis=1, how="all"),
        hide_index=True,
        column_config=column_config,
        row_height=64,
        alt="Batch analysis results",
    )
    st.download_button(
        "Download CSV",
        df.drop(columns="Preview").to_csv(index=False),
        file_name="pulmoscan_batch_results.csv",
        mime="text/csv",
        icon=":material/download:",
        on_click="ignore",
    )
    st.caption("Results use the thresholds that were set when you pressed Analyze.")
