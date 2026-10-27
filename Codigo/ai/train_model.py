"""
Training Script — Transfer Learning Pipeline
===============================================
This script is executed by the ML Engineer / Data Scientist to:
  1. Download and prepare the medical image dataset.
  2. Build the Transfer Learning model (MobileNetV2 + Fine-Tuning).
  3. Train the model.
  4. Evaluate and generate performance metrics.
  5. Save the model and register it in the Model Registry.

This process runs OUTSIDE the clinical inference flow (Milestone 1 Correction #5).

Usage:
    python ai/train_model.py [--epochs 10] [--version 1.0.0]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger, get_logger
from config.settings import DEFAULT_MODEL_NAME

setup_logger()
logger = get_logger("TrainingScript")


def main():
    parser = argparse.ArgumentParser(
        description="Train the Medical Image Classification model using Transfer Learning"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--version", type=str, default="1.0.0", help="Model version string")
    parser.add_argument("--fine-tune-layers", type=int, default=20, help="Number of base layers to fine-tune")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Transfer Learning Training Pipeline")
    logger.info(f"  Model: {DEFAULT_MODEL_NAME} v{args.version}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info("=" * 60)

    # Step 1: Initialize database (needed for Model Registry)
    from infrastructure.database import init_database
    init_database()

    # Step 2: Prepare dataset
    from ai.transfer_learning import TransferLearningTrainer

    trainer = TransferLearningTrainer()

    logger.info("\n[*] Step 1/5: Preparing dataset...")
    dataset_info = trainer.prepare_dataset()
    logger.info(f"Dataset ready: {dataset_info}")

    # Step 3: Build model
    logger.info("\n[*] Step 2/5: Building model architecture...")
    trainer.build_model(fine_tune_layers=args.fine_tune_layers)

    # Step 4: Train
    logger.info("\n[*] Step 3/5: Training model...")
    history = trainer.train(epochs=args.epochs)
    logger.info(f"Training results: {history}")

    # Step 5: Evaluate
    logger.info("\n[*] Step 4/5: Evaluating model...")
    metrics = trainer.evaluate()
    logger.info(f"Evaluation metrics: {metrics}")

    # Step 6: Save model
    logger.info("\n[*] Step 5/5: Saving model and registering in Model Registry...")
    model_path = trainer.save_model(version=args.version)

    # Step 7: Register in Model Registry
    from infrastructure.model_registry import ModelRegistry
    registry = ModelRegistry()

    registered = registry.register_version(
        model_name=DEFAULT_MODEL_NAME,
        version=args.version,
        file_path=model_path,
        metrics=metrics,
        created_by="ML Engineer",
        description=f"Transfer Learning model (MobileNetV2 + Fine-Tuning, {args.epochs} epochs)",
    )

    # Step 8: Approve and activate
    registry.approve_version(DEFAULT_MODEL_NAME, args.version)
    registry.activate_version(DEFAULT_MODEL_NAME, args.version)

    logger.info("\n" + "=" * 60)
    logger.info("  [+] Training Pipeline Complete!")
    logger.info(f"  Model: {DEFAULT_MODEL_NAME} v{args.version}")
    logger.info(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
    logger.info(f"  Precision: {metrics.get('precision', 'N/A')}")
    logger.info(f"  Recall: {metrics.get('recall', 'N/A')}")
    logger.info(f"  F1-Score: {metrics.get('f1_score', 'N/A')}")
    logger.info(f"  Saved to: {model_path}")
    logger.info(f"  Status: APPROVED and ACTIVE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
