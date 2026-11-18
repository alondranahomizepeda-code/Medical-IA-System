"""
Training Script — Transfer Learning Pipeline
===============================================
This script is executed by the ML Engineer / Data Scientist to:
  1. Load and prepare the spinal radiograph dataset.
  2. Build the Transfer Learning model (MobileNetV2 + Rescaling(2,-1) + Spatial Head).
  3. Train the model with balanced class weights.
  4. Evaluate on the held-out test split and enforce the automatic Quality Gate.
  5. Save the model and register it in the Model Registry (approving/activating ONLY if it passes the Quality Gate).

This process runs OUTSIDE the clinical inference flow (Milestone 1 Correction #5).

Usage:
    python ai/train_model.py [--epochs 12] [--version 1.1.0] [--dataset <path>]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger, get_logger
from config.settings import DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION

setup_logger()
logger = get_logger("TrainingScript")


def main():
    parser = argparse.ArgumentParser(
        description="Train the Medical Image Classification model using Transfer Learning"
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--version", type=str, default=DEFAULT_MODEL_VERSION, help="Model version string")
    parser.add_argument("--fine-tune-layers", type=int, default=30, help="Number of base layers to fine-tune")
    parser.add_argument("--dataset", type=str, default=None, help="Optional path to dataset")
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
    dataset_info = trainer.prepare_dataset(dataset_path=args.dataset)
    logger.info(f"Dataset ready: {dataset_info}")

    # Step 3: Build model
    logger.info("\n[*] Step 2/5: Building model architecture...")
    trainer.build_model(fine_tune_layers=args.fine_tune_layers)

    # Step 4: Train
    logger.info("\n[*] Step 3/5: Training model...")
    history = trainer.train(epochs=args.epochs)
    logger.info(f"Training results: {history}")

    # Step 5: Evaluate
    logger.info("\n[*] Step 4/5: Evaluating model & running Quality Gate...")
    metrics = trainer.evaluate()
    logger.info(f"Evaluation metrics: {metrics}")

    # Step 6: Save model
    logger.info("\n[*] Step 5/5: Saving model and registering in Model Registry...")
    model_path = trainer.save_model(version=args.version, metrics=metrics)

    # Step 7: Register in Model Registry
    from infrastructure.model_registry import ModelRegistry
    registry = ModelRegistry()

    registry.register_version(
        model_name=DEFAULT_MODEL_NAME,
        version=args.version,
        file_path=model_path,
        metrics=metrics,
        created_by="ML Engineer",
        description=f"Transfer Learning model (MobileNetV2 + Rescaling(2,-1), {args.epochs} epochs)",
    )

    # Step 8: Approve and activate ONLY if the automatic Quality Gate passed (Question 10)
    gate = metrics.get("quality_gate", {})
    if gate.get("approved", False):
        registry.approve_version(DEFAULT_MODEL_NAME, args.version)
        registry.activate_version(DEFAULT_MODEL_NAME, args.version)
        status_str = "APPROVED and ACTIVE (Quality Gate Passed ✓)"
    else:
        status_str = f"REGISTERED ONLY — NOT APPROVED (Quality Gate Failed: {gate.get('reasons')})"

    logger.info("\n" + "=" * 60)
    logger.info("  [+] Training Pipeline Complete!")
    logger.info(f"  Model:             {DEFAULT_MODEL_NAME} v{args.version}")
    logger.info(f"  Accuracy:          {metrics.get('accuracy', 'N/A')}")
    logger.info(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 'N/A')}")
    logger.info(f"  Macro Precision:   {metrics.get('precision', 'N/A')}")
    logger.info(f"  Macro Recall:      {metrics.get('recall', 'N/A')}")
    logger.info(f"  Macro F1-Score:    {metrics.get('f1_score', 'N/A')}")
    logger.info(f"  Saved to:          {model_path}")
    logger.info(f"  Status:            {status_str}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
