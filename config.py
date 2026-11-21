"""Project-wide paths and model settings: the single source of truth for numbers shown in the app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / "test_samples"
STYLES_PATH = BASE_DIR / "styles" / "styles.css"
EVAL_PREDICTIONS_PATH = BASE_DIR / "evaluation" / "test_predictions.csv"

TEST_SET_SIZE = 624
TEST_SET_PNEUMONIA = 390
TEST_SET_NORMAL = 234


@dataclass(frozen=True)
class ReportedMetrics:
    """Test-set results recorded in the training notebook.

    Note: the thresholds were picked while looking at the test set, so these
    numbers are slightly optimistic. Regenerate them with scripts/evaluate.py.
    """

    threshold: float
    sensitivity: float
    specificity: float
    precision: float
    roc_auc: float


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    role: str
    weights_path: Path
    default_threshold: float
    reported: ReportedMetrics


MODELS: dict[str, ModelSpec] = {
    "densenet121": ModelSpec(
        key="densenet121",
        name="DenseNet121",
        role="Champion: high-sensitivity screening",
        weights_path=BASE_DIR / "pneumonia_densenet_final.pth",
        default_threshold=0.30,
        reported=ReportedMetrics(
            threshold=0.30,
            sensitivity=389 / 390,
            specificity=0.51,
            precision=0.77,
            roc_auc=0.961,
        ),
    ),
    "resnet50": ModelSpec(
        key="resnet50",
        name="ResNet50",
        role="Baseline: standard transfer learning",
        weights_path=BASE_DIR / "pneumonia_resnet_final.pth",
        default_threshold=0.70,
        reported=ReportedMetrics(
            threshold=0.70,
            sensitivity=0.97,
            specificity=0.74,
            precision=0.86,
            roc_auc=0.964,
        ),
    ),
}
