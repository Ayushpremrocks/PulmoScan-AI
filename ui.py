"""Streamlit building blocks shared by the app pages."""

from __future__ import annotations

import base64
import hashlib
import io
import re
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

import config
import inference

DISPLAY_MAX_PX = 900
MODEL_CHOICES = ["densenet121", "resnet50", "both"]
DISCLAIMER = "For education and research only. Not a medical device and not a substitute for a clinician's diagnosis."


# ---------- Cached model work ----------


@st.cache_resource(show_spinner="Loading model…")
def get_model(model_key: str):
    spec = config.MODELS[model_key]
    return inference.load_model(model_key, spec.weights_path)


@st.cache_data(max_entries=256, show_spinner=False)
def pneumonia_probability(model_key: str, image_bytes: bytes) -> float:
    image = inference.open_image(image_bytes)
    return float(inference.predict_proba(get_model(model_key), inference.preprocess(image))[0])


@st.cache_data(max_entries=64, show_spinner=False)
def gradcam_map(model_key: str, image_bytes: bytes) -> np.ndarray:
    model = get_model(model_key)
    tensor = inference.preprocess(inference.open_image(image_bytes))
    return inference.grad_cam(model, inference.gradcam_target_layer(model, model_key), tensor)


# ---------- Small helpers ----------


def model_label(choice: str) -> str:
    return "Both models" if choice == "both" else config.MODELS[choice].name


def selected_models(choice: str) -> list[str]:
    return list(config.MODELS) if choice == "both" else [choice]


def is_pneumonia(probability: float, threshold: float) -> bool:
    return probability > threshold


def verdict(probability: float, threshold: float) -> str:
    return "Pneumonia suspected" if is_pneumonia(probability, threshold) else "Normal"


def image_id(image_bytes: bytes) -> str:
    return hashlib.sha1(image_bytes).hexdigest()[:12]


def display_copy(image: Image.Image) -> Image.Image:
    small = image.copy()
    small.thumbnail((DISPLAY_MAX_PX, DISPLAY_MAX_PX))
    return small


def thumbnail_data_uri(image: Image.Image, size: int = 96) -> str:
    thumb = image.copy()
    thumb.thumbnail((size, size))
    buffer = io.BytesIO()
    thumb.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def page_header(title: str, icon: str, subtitle: str) -> None:
    st.title(title, icon=icon)
    st.caption(subtitle)


# ---------- Settings ----------


def threshold_slider(model_key: str, key: str) -> float:
    spec = config.MODELS[model_key]
    return st.slider(
        f"{spec.name} threshold",
        min_value=0.0,
        max_value=1.0,
        value=spec.default_threshold,
        step=0.05,
        key=key,
        help=(
            "An X-ray is flagged when the model's pneumonia score is above this value. "
            "Lower catches more cases but raises more false alarms. "
            f"Tuned value: {spec.default_threshold:.2f}."
        ),
    )


def model_picker(key: str) -> str:
    return st.segmented_control(
        "Model",
        MODEL_CHOICES,
        format_func=model_label,
        default="densenet121",
        required=True,
        key=key,
        help="DenseNet121 is tuned for high sensitivity. ResNet50 is the baseline.",
    )


# ---------- Image input ----------


def _sample_sort_key(name: str) -> tuple[str, int]:
    match = re.match(r"([a-z]+)_(\d+)", name)
    return (match.group(1), int(match.group(2))) if match else (name, 0)


def sample_label(name: str) -> str:
    match = re.match(r"([a-z]+)_(\d+)", name)
    return f"{match.group(1).capitalize()} case {match.group(2)}" if match else name


def sample_true_label(name: str) -> str | None:
    if name.startswith("pneumonia"):
        return "Pneumonia"
    if name.startswith("normal"):
        return "Normal"
    return None


def sample_files() -> list[str]:
    if not config.SAMPLES_DIR.exists():
        return []
    names = [p.name for p in config.SAMPLES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return sorted(names, key=_sample_sort_key)


def image_input(key_prefix: str) -> tuple[bytes, str] | None:
    """Upload an X-ray or pick a sample. Returns (image bytes, display name)."""
    source = st.segmented_control(
        "Image",
        ["sample", "upload"],
        format_func={"sample": ":material/photo_library: Sample", "upload": ":material/upload: Upload"}.get,
        default="sample",
        required=True,
        key=f"{key_prefix}_source",
    )
    if source == "upload":
        uploaded = st.file_uploader(
            "Chest X-ray",
            type=["jpg", "jpeg", "png"],
            key=f"{key_prefix}_upload",
            label_visibility="collapsed",
        )
        if uploaded is None:
            return None
        return uploaded.getvalue(), uploaded.name

    samples = sample_files()
    if not samples:
        st.error(f"No sample images found in `{config.SAMPLES_DIR.name}/`.")
        return None
    name = st.selectbox(
        "Sample case",
        samples,
        format_func=sample_label,
        key=f"{key_prefix}_sample",
        label_visibility="collapsed",
        help="X-rays from the held-out test set. The name gives the true diagnosis.",
    )
    return (config.SAMPLES_DIR / name).read_bytes(), sample_label(name)


def checked_image(image_bytes: bytes, key_prefix: str) -> Image.Image | None:
    """Decode the image and run plausibility checks. Returns None if analysis should stop."""
    try:
        image = inference.open_image(image_bytes)
    except (UnidentifiedImageError, OSError):
        st.error("This file couldn't be read as an image. Upload a JPEG or PNG chest X-ray.", icon=":material/error:")
        return None

    problems = inference.xray_warnings(image)
    if problems:
        st.warning(
            "\n".join(f"- {p}" for p in problems)
            + "\n\nThe models were trained only on chest X-rays, so results for other images are meaningless.",
            title="This may not be a chest X-ray",
            icon=":material/warning:",
        )
        if not st.checkbox("Analyze anyway", key=f"{key_prefix}_override"):
            return None
    return image


# ---------- Results ----------


def result_card(model_key: str, probability: float, threshold: float) -> None:
    spec = config.MODELS[model_key]
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**{spec.name}**")
            st.space("stretch")
            if is_pneumonia(probability, threshold):
                st.badge("Pneumonia suspected", icon=":material/warning:", color="red")
            else:
                st.badge("Normal", icon=":material/check_circle:", color="green")
        st.metric(
            "Pneumonia score",
            f"{probability:.1%}",
            help="The model's softmax output. It isn't a calibrated clinical probability.",
        )
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Flagged above {threshold:.0%} · {spec.role}")


def build_report(name: str, results: dict[str, float], thresholds: dict[str, float]) -> str:
    lines = [
        "PulmoScan AI: screening report",
        "=" * 34,
        f"Date:  {datetime.now():%Y-%m-%d %H:%M}",
        f"Image: {name}",
        "",
    ]
    for key, probability in results.items():
        spec = config.MODELS[key]
        lines += [
            f"{spec.name}",
            f"  Result:          {verdict(probability, thresholds[key])}",
            f"  Pneumonia score: {probability:.1%}",
            f"  Threshold:       {thresholds[key]:.2f}",
            "",
        ]
    lines += [DISCLAIMER]
    return "\n".join(lines)


GRADCAM_EXPLANATION = (
    "Warm colours show the regions that push the model towards **pneumonia**. "
    "The map is computed on a 7×7 grid and upscaled, so it shows rough areas, not exact lesions. "
    "Heat outside the lungs (labels, edges) suggests the model is relying on shortcuts."
)


# ---------- Session history ----------


def record_history(entry: dict) -> None:
    """Add an analysis to this browser session's history, skipping exact repeats."""
    history = st.session_state.setdefault("history", [])
    key = (entry["Image ID"], entry["Model"], entry["Threshold"])
    if any((h["Image ID"], h["Model"], h["Threshold"]) == key for h in history):
        return
    history.append(entry)


def clear_history() -> None:
    st.session_state["history"] = []
