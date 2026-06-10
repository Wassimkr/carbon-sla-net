"""Phase 5a: Behavior Cloning pre-training script.

Run:
    python experiments/run_bc_training.py           # production config
    python experiments/run_bc_training.py --fast    # fast config for testing
"""

from __future__ import annotations

import sys
from pathlib import Path
# When run as a script, add the project root to sys.path so that
# 'experiments', 'agent', 'env', etc. are importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path

from agent.bc_trainer import BCTrainer, BCTrainingResult
from experiments.configs.main_training import (
    TrainingConfig,
    get_default_config,
    get_fast_config,
)


def run_bc_training(config: TrainingConfig) -> BCTrainingResult:
    """Run BC pre-training and return a summary result.

    Parameters
    ----------
    config:
        A :class:`TrainingConfig` controlling all BC hyperparameters.

    Returns
    -------
    :class:`~agent.bc_trainer.BCTrainingResult` with loss curves and the
    path to the saved best checkpoint.

    Raises
    ------
    FileNotFoundError
        If ``config.bc_dataset_path`` does not exist.
    """
    print("=== Phase 5a: Behavior Cloning Pre-training ===")

    dataset_path = Path(config.bc_dataset_path)
    if not dataset_path.exists():
        print(f"ERROR: BC dataset not found at {dataset_path}")
        raise FileNotFoundError(
            f"BC dataset not found: {dataset_path}. "
            "Run milp/oracle_runner.py to generate it first."
        )

    print(f"  Dataset : {dataset_path}")
    print(f"  Epochs  : {config.bc_epochs}  (patience={config.bc_patience})")
    print(f"  Batch   : {config.bc_batch_size}  LR={config.bc_learning_rate}")
    print(f"  Dropout : {config.bc_dropout}")
    print(f"  Out dir : {config.bc_checkpoint_dir}/{config.bc_run_name}_best.pt")
    print()

    trainer = BCTrainer(
        obs_dim=74,
        n_actions=11,
        dropout=config.bc_dropout,
        learning_rate=config.bc_learning_rate,
    )

    result = trainer.train(
        dataset_path=str(dataset_path),
        epochs=config.bc_epochs,
        batch_size=config.bc_batch_size,
        patience=config.bc_patience,
        checkpoint_dir=config.bc_checkpoint_dir,
        run_name=config.bc_run_name,
    )

    print()
    print("BC training complete.")
    print(f"  Best epoch     : {result.best_epoch}/{result.total_epochs_run}")
    print(f"  Best val_loss  : {result.best_val_loss:.4f}")
    print(f"  Best val_acc   : {result.best_val_accuracy:.3f}")
    print(f"  Checkpoint     : {result.checkpoint_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 5a: BC pre-training for Carbon-SLA-Net"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Use fast config for testing"
    )
    args = parser.parse_args()

    config = get_fast_config() if args.fast else get_default_config()
    run_bc_training(config)
