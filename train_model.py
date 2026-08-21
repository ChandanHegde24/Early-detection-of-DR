"""
RetinaGuard — Diabetic Retinopathy Model Training Script

Trains deep learning CNN model on the dataset in `raw_combined` and the
clinical biomarker stacking ensemble model. Saves all artifacts to `saved_models/`.

Examples:
    # 1. Quick test on CPU (~5-10 mins with 150 samples per class):
    python train_model.py --fast

    # 2. Fast CPU training with MobileNetV2 / EfficientNetB0 on custom subset:
    python train_model.py --subset 300 --epochs 5 --backbone EfficientNetB0

    # 3. Full dataset training (all 22,629 images):
    python train_model.py --epochs 12 --fine-tune-epochs 4
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.train import train_cnn_pipeline, train_biomarker_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="RetinaGuard — Train DR Deep Learning & Biomarker AI Models."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="raw_combined",
        help="Directory containing the 5 class folders (0, 1, 2, 3, 4)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="EfficientNetB0",
        choices=["EfficientNetB0", "EfficientNetV2B0", "EfficientNetB3", "MobileNetV2", "ResNet50"],
        help="Backbone CNN architecture (default: EfficientNetB0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for head training (default: 10)",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32; use 16 for lower memory on CPU)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Max images per class to train on (e.g. 200 for fast CPU testing). Omit for full dataset.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a quick verification pass (150 images per class, 3 head epochs, 1 fine-tune epoch)",
    )
    parser.add_argument(
        "--skip-biomarker",
        action="store_true",
        help="Skip clinical biomarker model training and train CNN only",
    )
    parser.add_argument(
        "--skip-cnn",
        action="store_true",
        help="Skip CNN training and train biomarker model only",
    )

    args = parser.parse_args()

    if args.fast:
        print("[FAST MODE ENABLED] Running lightweight verification configuration...")
        subset = 150
        epochs = 3
        fine_tune_epochs = 1
        batch_size = 16
    else:
        subset = args.subset
        epochs = args.epochs
        fine_tune_epochs = args.fine_tune_epochs
        batch_size = args.batch_size

    print("\n" + "=" * 60)
    print("RETINAGUARD AI MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"* Dataset directory:   {args.data_dir}")
    print(f"* CNN Backbone:        {args.backbone}")
    print(f"* Head Epochs:         {epochs}")
    print(f"* Fine-tune Epochs:    {fine_tune_epochs}")
    print(f"* Batch Size:          {batch_size}")
    print(f"* Subset per class:    {subset if subset else 'Full Dataset (all images)'}")
    print("=" * 60 + "\n")

    # 1. Train Clinical Biomarker Model
    if not args.skip_biomarker:
        train_biomarker_pipeline()

    # 2. Train Retinal CNN Model
    if not args.skip_cnn:
        train_cnn_pipeline(
            data_dir=args.data_dir,
            backbone=args.backbone,
            epochs=epochs,
            fine_tune_epochs=fine_tune_epochs,
            batch_size=batch_size,
            subset_per_class=subset,
        )

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL MODELS TRAINED & SAVED TO 'saved_models/'")
    print("=" * 60)
    print("Files created:")
    print("  * saved_models/cnn_weights.h5      (CNN model weights)")
    print("  * saved_models/biomarker_model.pkl (Clinical stacking ensemble)")
    print("  * saved_models/biomarker_scaler.pkl(Feature scaler)")
    print("  * saved_models/features.json       (Feature names)")
    print("\nNext steps:")
    print("  1. Start FastAPI backend:  uvicorn api.main:app --reload --port 8000")
    print("  2. Start Next.js frontend: cd frontend && npm run dev")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
