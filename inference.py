"""Model loading, preprocessing, prediction and Grad-CAM.

This module has no Streamlit imports so command-line scripts can reuse it.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMAGE_SIZE = 224
NORMAL, PNEUMONIA = 0, 1

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _classifier_head(in_features: int) -> nn.Sequential:
    # Must match the head used in training exactly, or the weights won't load.
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 2),
    )


def build_model(arch: str) -> nn.Module:
    if arch == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = _classifier_head(model.classifier.in_features)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = _classifier_head(model.fc.in_features)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


def load_model(arch: str, weights_path: Path) -> nn.Module:
    model = build_model(arch)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    # Inference only. Freezing the weights also means Grad-CAM never writes
    # parameter gradients, so one cached model can be shared between sessions.
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    # The last feature block before global pooling.
    if arch == "densenet121":
        return model.features.norm5
    if arch == "resnet50":
        return model.layer4
    raise ValueError(f"Unknown architecture: {arch}")


def open_image(data: bytes) -> Image.Image:
    """Decode image bytes to RGB. Raises PIL.UnidentifiedImageError for non-images."""
    image = Image.open(io.BytesIO(data))
    image.load()
    return image.convert("RGB")


def preprocess(image: Image.Image) -> torch.Tensor:
    return _TRANSFORM(image).unsqueeze(0)


def predict_proba(model: nn.Module, batch: torch.Tensor) -> np.ndarray:
    """Return P(pneumonia) for each image in the batch."""
    with torch.inference_mode():
        logits = model(batch)
        return torch.softmax(logits, dim=1)[:, PNEUMONIA].cpu().numpy()


def grad_cam(
    model: nn.Module, layer: nn.Module, tensor: torch.Tensor, class_idx: int = PNEUMONIA
) -> np.ndarray:
    """Grad-CAM map (values 0-1) showing which regions push the model towards `class_idx`."""
    store: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        store["activations"] = output.detach().clone()
        # A tensor hook (not a module backward hook) copes with the in-place
        # ReLU that DenseNet applies right after norm5.
        output.register_hook(lambda grad: store.__setitem__("gradients", grad.detach().clone()))

    handle = layer.register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():
            # Weights are frozen, so make the input require grad to build a graph.
            x = tensor.clone().requires_grad_(True)
            score = model(x)[0, class_idx]
            score.backward()
    finally:
        handle.remove()

    if "activations" not in store or "gradients" not in store:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    activations = store["activations"][0]
    weights = store["gradients"][0].mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * activations).sum(dim=0)).cpu().numpy()
    peak = cam.max()
    if peak > 0:
        cam = cam / peak
    return cam.astype(np.float32)


def colorize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Upscale a Grad-CAM map to (width, height) with bicubic smoothing and apply the JET colormap."""
    resized = cv2.resize(heatmap, size, interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0, 1)
    colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_heatmap(heatmap: np.ndarray, image: Image.Image, alpha: float = 0.4) -> np.ndarray:
    img = np.asarray(image)
    colored = colorize_heatmap(heatmap, (img.shape[1], img.shape[0]))
    return cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)


def enhance_contrast(image: Image.Image) -> np.ndarray:
    """CLAHE contrast enhancement, a common radiology viewing aid. Display only, not used by the models."""
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def to_png_bytes(array: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR) if array.ndim == 3 else array)
    if not ok:
        raise ValueError("Could not encode image")
    return buffer.tobytes()


def xray_warnings(image: Image.Image) -> list[str]:
    """Cheap checks that an image could plausibly be a chest X-ray.

    These are heuristics, not a classifier: they catch obvious mistakes
    (photos, screenshots, tiny images) before the model gives a confident
    answer about something it was never trained on.
    """
    warnings: list[str] = []
    width, height = image.size

    small = np.asarray(image.resize((128, 128)), dtype=np.float32)
    channel_spread = (
        np.abs(small[..., 0] - small[..., 1]).mean() + np.abs(small[..., 1] - small[..., 2]).mean()
    )
    if channel_spread > 8:
        warnings.append("It's a colour image. Chest X-rays are greyscale.")

    ratio = width / height
    if not 0.5 <= ratio <= 2.2:
        warnings.append(f"Its shape is unusual for a chest X-ray (width/height = {ratio:.2f}).")

    if min(width, height) < 150:
        warnings.append(f"It's very small ({width}×{height} px), so detail will be lost.")

    return warnings
