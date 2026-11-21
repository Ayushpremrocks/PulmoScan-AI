"""Score both models on a labelled test folder and save per-image predictions.

Usage:
    python scripts/evaluate.py --data-dir path/to/chest_xray/test

The folder must contain NORMAL/ and PNEUMONIA/ subfolders (the layout of the
Kaggle "Chest X-Ray Images (Pneumonia)" dataset). The output CSV powers the
app's Model evaluation page.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import inference  # noqa: E402
import metrics  # noqa: E402

CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def collect_files(data_dir: Path) -> list[tuple[Path, int]]:
    files = []
    for folder, label in CLASSES.items():
        class_dir = data_dir / folder
        if not class_dir.is_dir():
            sys.exit(f"Missing folder: {class_dir}")
        files += [(p, label) for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    return files


def score(model: torch.nn.Module, paths: list[Path], batch_size: int) -> list[float]:
    probs: list[float] = []
    for start in range(0, len(paths), batch_size):
        batch = torch.cat(
            [inference.preprocess(inference.open_image(p.read_bytes())) for p in paths[start : start + batch_size]]
        )
        probs.extend(inference.predict_proba(model, batch).tolist())
        print(f"  {min(start + batch_size, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder with NORMAL/ and PNEUMONIA/")
    parser.add_argument("--out", type=Path, default=config.EVAL_PREDICTIONS_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    files = collect_files(args.data_dir)
    paths = [p for p, _ in files]
    df = pd.DataFrame(
        {"file": [f"{p.parent.name}/{p.name}" for p in paths], "label": [label for _, label in files]}
    )
    print(f"Found {len(df)} images ({df.label.sum()} pneumonia, {(df.label == 0).sum()} normal).")

    for key, spec in config.MODELS.items():
        print(f"Scoring with {spec.name}…")
        model = inference.load_model(key, spec.weights_path)
        df[key] = score(model, paths, args.batch_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}\n")

    labels = df["label"].to_numpy()
    for key, spec in config.MODELS.items():
        probs = df[key].to_numpy()
        c = metrics.confusion_at(labels, probs, spec.default_threshold)
        print(
            f"{spec.name:12s} threshold={spec.default_threshold:.2f}  "
            f"sensitivity={c.sensitivity:.3f}  specificity={c.specificity:.3f}  "
            f"precision={c.precision:.3f}  AUC={metrics.roc_auc(labels, probs):.3f}  "
            f"(missed {c.fn}, false alarms {c.fp})"
        )


if __name__ == "__main__":
    main()
