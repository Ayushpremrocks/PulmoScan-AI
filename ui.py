"""Streamlit building blocks shared by the app pages."""

from __future__ import annotations

import re

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

import config
import inference

DISPLAY_MAX_PX = 900


# ---------- Cached model work ----------


@st.cache_resource(show_spinner="Loading model…")
def get_model(model_key: str):
    spec = config.MODELS[model_key]
    return inference.load_model(model_key, spec.weights_path)


@st.cache_data(max_entries=64, show_spinner=False)
def pneumonia_probability(model_key: str, image_bytes: bytes) -> float:
    image = inference.open_image(image_bytes)
    return float(inference.predict_proba(get_model(model_key), inference.preprocess(image))[0])


@st.cache_data(max_entries=64, show_spinner=False)
def gradcam_map(model_key: str, image_bytes: bytes) -> np.ndarray:
    model = get_model(model_key)
    tensor = inference.preprocess(inference.open_image(image_bytes))
    return inference.grad_cam(model, inference.gradcam_target_layer(model, model_key), tensor)


# ---------- Page chrome ----------


def load_styles() -> None:
    st.html(f"<style>{config.STYLES_PATH.read_text(encoding='utf-8')}</style>")


def hero(title: str, subtitle: str) -> None:
    st.html(f'<div><h1 class="hero-text">{title}</h1><h3 class="subheading">{subtitle}</h3></div>')


def threshold_slider(model_key: str, key: str) -> float:
    spec = config.MODELS[model_key]
    return st.slider(
        f"{spec.name} decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=spec.default_threshold,
        step=0.05,
        key=key,
        help=(
            "An X-ray is flagged as pneumonia when the model's score is above this value. "
            "Lower values catch more cases but raise more false alarms. "
            f"The tuned value for {spec.name} is {spec.default_threshold:.2f}."
        ),
    )


# ---------- Image input ----------


def _sample_sort_key(name: str) -> tuple[str, int]:
    match = re.match(r"([a-z]+)_(\d+)", name)
    return (match.group(1), int(match.group(2))) if match else (name, 0)


def _sample_label(name: str) -> str:
    match = re.match(r"([a-z]+)_(\d+)", name)
    return f"{match.group(1).capitalize()} case {match.group(2)}" if match else name


def sample_files() -> list[str]:
    if not config.SAMPLES_DIR.exists():
        return []
    names = [p.name for p in config.SAMPLES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return sorted(names, key=_sample_sort_key)


def image_input(key_prefix: str) -> tuple[bytes, str] | None:
    """Let the user upload an X-ray or pick a sample. Returns (image bytes, caption)."""
    with st.container(border=True):
        source = st.segmented_control(
            "Image source",
            ["sample", "upload"],
            format_func={"sample": ":material/image: Sample case", "upload": ":material/upload: Upload"}.get,
            default="sample",
            required=True,
            key=f"{key_prefix}_source",
        )
        if source == "upload":
            uploaded = st.file_uploader(
                "Chest X-ray", type=["jpg", "jpeg", "png"], key=f"{key_prefix}_upload"
            )
            if uploaded is None:
                st.caption("Upload a frontal chest X-ray (JPEG or PNG).")
                return None
            return uploaded.getvalue(), uploaded.name

        samples = sample_files()
        if not samples:
            st.error(f"No sample images found in `{config.SAMPLES_DIR.name}/`.")
            return None
        name = st.selectbox(
            "Sample case",
            samples,
            format_func=_sample_label,
            key=f"{key_prefix}_sample",
            help="Images from the held-out test set. The file name gives the true diagnosis.",
        )
        return (config.SAMPLES_DIR / name).read_bytes(), f"Sample: {_sample_label(name)}"


def checked_image(image_bytes: bytes, key_prefix: str) -> Image.Image | None:
    """Decode the image and run plausibility checks. Returns None if analysis should stop."""
    try:
        image = inference.open_image(image_bytes)
    except (UnidentifiedImageError, OSError):
        st.error("This file couldn't be read as an image. Upload a JPEG or PNG chest X-ray.")
        return None

    problems = inference.xray_warnings(image)
    if problems:
        st.warning(
            "\n".join(f"- {p}" for p in problems)
            + "\n\nThe models were trained only on chest X-rays. For anything else their output is meaningless.",
            title="This may not be a chest X-ray",
            icon=":material/warning:",
        )
        if not st.checkbox("Analyze anyway", key=f"{key_prefix}_override"):
            return None
    return image


def display_copy(image: Image.Image) -> Image.Image:
    small = image.copy()
    small.thumbnail((DISPLAY_MAX_PX, DISPLAY_MAX_PX))
    return small


# ---------- Results ----------


def render_result(model_key: str, probability: float, threshold: float) -> None:
    spec = config.MODELS[model_key]
    with st.container(border=True):
        st.markdown(f"**{spec.name}**")
        if probability > threshold:
            st.error(
                "Findings consistent with pneumonia. Refer for clinical review.",
                title="Pneumonia suspected",
                icon=":material/warning:",
            )
        else:
            st.success(
                "Score is below the decision threshold.",
                title="Normal",
                icon=":material/check_circle:",
            )
        st.metric(
            "Pneumonia score",
            f"{probability:.1%}",
            help="The model's softmax output. It is not a calibrated clinical probability.",
        )
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Flagged as pneumonia when the score is above {threshold:.0%}.")


def render_gradcam(model_key: str, image_bytes: bytes, image: Image.Image, *, side_by_side: bool = True) -> None:
    with st.spinner("Generating attention map…"):
        heatmap = gradcam_map(model_key, image_bytes)
    shown = display_copy(image)
    overlay = inference.overlay_heatmap(heatmap, shown)
    name = config.MODELS[model_key].name
    if side_by_side:
        raw = inference.colorize_heatmap(heatmap, shown.size)
        c1, c2 = st.columns(2)
        c1.image(raw, caption="Attention map", width="stretch", alt=f"{name} Grad-CAM attention map")
        c2.image(overlay, caption="Overlay on X-ray", width="stretch", alt=f"{name} Grad-CAM overlay on the X-ray")
    else:
        st.image(overlay, caption=f"{name} Grad-CAM overlay", width="stretch", alt=f"{name} Grad-CAM overlay on the X-ray")


GRADCAM_EXPLANATION = (
    "Red areas push the model towards **pneumonia**; blue areas have little influence. "
    "The map is computed on a 7×7 grid and upscaled, so it shows rough regions, not exact lesions. "
    "Attention outside the lungs (labels, borders) is a warning sign that the model is using shortcuts."
)


def model_page(model_key: str) -> None:
    """Full single-model analysis page."""
    spec = config.MODELS[model_key]
    with st.sidebar:
        threshold = threshold_slider(model_key, key=f"{model_key}_page_threshold")

    picked = image_input(key_prefix=model_key)
    if picked is None:
        return
    image_bytes, caption = picked
    image = checked_image(image_bytes, key_prefix=model_key)
    if image is None:
        return

    with st.spinner("Analyzing lung patterns…"):
        probability = pneumonia_probability(model_key, image_bytes)

    col_image, col_result = st.columns(2)
    with col_image:
        st.image(display_copy(image), caption=caption, width="stretch", alt="Chest X-ray being analyzed")
    with col_result:
        render_result(model_key, probability, threshold)

    st.subheader("Explainable AI (Grad-CAM)")
    if st.toggle("Show heatmap analysis", value=True, key=f"{model_key}_show_gradcam"):
        render_gradcam(model_key, image_bytes, image)
        st.caption(GRADCAM_EXPLANATION)
    st.caption(f"{spec.name}: {spec.role}.")
