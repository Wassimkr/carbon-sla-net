"""Behavior cloning trainer for Carbon-SLA-Net Phase 3a.

Loads an (obs, action) dataset produced by oracle_runner.generate_bc_dataset(),
trains CarbonSLANetPolicy with cross-entropy loss, and provides a weight-transfer
utility to initialise a MaskablePPO model from the BC checkpoint.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from agent.policy_net import CarbonSLANetPolicy


@dataclass
class BCTrainingResult:
    """Summary of a completed BC training run."""

    train_losses: List[float]       # one entry per epoch run
    val_losses: List[float]
    val_accuracies: List[float]
    best_val_loss: float
    best_val_accuracy: float
    best_epoch: int                 # 1-indexed epoch with best val_loss
    total_epochs_run: int
    checkpoint_path: str


class BCTrainer:
    """Supervised behavior-cloning trainer.

    Trains a :class:`~agent.policy_net.CarbonSLANetPolicy` on
    ``(obs, action)`` pairs collected by the MILP oracle.  Includes early
    stopping, cosine LR annealing, and a weight-transfer helper for PPO
    warm-starting.
    """

    def __init__(
        self,
        obs_dim: int = 74,
        n_actions: int = 11,
        hidden_sizes: tuple[int, ...] = (256, 256),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        device: str = "auto",
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.policy = CarbonSLANetPolicy(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_dataset(
        self, dataset_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and validate a BC dataset pickle file.

        Returns
        -------
        ``(obs_tensor, action_tensor)`` with shapes ``(N, 74)`` and ``(N,)``.
        """
        with open(dataset_path, "rb") as fh:
            raw: list = pickle.load(fh)

        obs_list = [pair[0] for pair in raw]
        act_list = [int(pair[1]) for pair in raw]

        obs_arr = np.stack(obs_list).astype(np.float32)
        act_arr = np.array(act_list, dtype=np.int64)

        # Validate actions
        if np.any((act_arr < 0) | (act_arr >= self.n_actions)):
            bad = act_arr[(act_arr < 0) | (act_arr >= self.n_actions)]
            raise ValueError(
                f"Actions out of [0, {self.n_actions - 1}]: {bad.tolist()[:5]}"
            )

        # Validate / clip observations
        if np.any(obs_arr < 0.0) or np.any(obs_arr > 1.0):
            warnings.warn(
                "Observation values outside [0, 1] detected; clipping.",
                RuntimeWarning,
                stacklevel=2,
            )
            obs_arr = np.clip(obs_arr, 0.0, 1.0)

        return (
            torch.from_numpy(obs_arr),
            torch.from_numpy(act_arr),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset_path: str,
        epochs: int = 50,
        batch_size: int = 256,
        val_fraction: float = 0.1,
        patience: int = 10,
        checkpoint_dir: str = "checkpoints/bc",
        run_name: str = "bc_run",
    ) -> BCTrainingResult:
        """Train with cross-entropy loss and cosine LR decay.

        Uses a temporal train/val split (no shuffling before split) to
        evaluate generalisation on later episodes.  Saves the best checkpoint
        and restores it on early stopping.
        """
        obs_t, act_t = self.load_dataset(dataset_path)
        N = len(obs_t)

        n_val = max(1, int(N * val_fraction)) if val_fraction > 0 else 0
        n_train = N - n_val

        train_obs = obs_t[:n_train]
        train_act = act_t[:n_train]
        val_obs = obs_t[n_train:].to(self.device)
        val_act = act_t[n_train:].to(self.device)

        train_ds = TensorDataset(train_obs, train_act)
        eff_batch = min(batch_size, n_train)
        train_loader = DataLoader(train_ds, batch_size=eff_batch, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        checkpoint_path = self.get_checkpoint_path(checkpoint_dir, run_name)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        train_losses: List[float] = []
        val_losses: List[float] = []
        val_accuracies: List[float] = []
        best_val_loss = float("inf")
        best_epoch = 1
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # ---- train ----
            self.policy.train()
            epoch_loss = 0.0
            for obs_b, act_b in train_loader:
                obs_b = obs_b.to(self.device)
                act_b = act_b.to(self.device)
                optimizer.zero_grad()
                logits = self.policy.get_action_logits(obs_b)
                loss = criterion(logits, act_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(obs_b)
            train_loss = epoch_loss / max(n_train, 1)

            # ---- validation ----
            self.policy.eval()
            with torch.no_grad():
                val_logits = self.policy.get_action_logits(val_obs)
                val_loss = criterion(val_logits, val_act).item()
                preds = val_logits.argmax(dim=1)
                val_acc = (preds == val_act).float().mean().item()

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.3f} | "
                f"lr={current_lr:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.policy.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    # Restore best weights before returning
                    self.policy.load_state_dict(
                        torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                    )
                    break

        return BCTrainingResult(
            train_losses=train_losses,
            val_losses=val_losses,
            val_accuracies=val_accuracies,
            best_val_loss=best_val_loss,
            best_val_accuracy=float(val_accuracies[best_epoch - 1]),
            best_epoch=best_epoch,
            total_epochs_run=len(train_losses),
            checkpoint_path=checkpoint_path,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_checkpoint_path(self, checkpoint_dir: str, run_name: str) -> str:
        return str(Path(checkpoint_dir) / f"{run_name}_best.pt")

    def transfer_weights_to_sb3(
        self,
        ppo_model,
        checkpoint_path: str,
    ) -> int:
        """Copy BC encoder weights into a MaskablePPO model by shape matching.

        Loads the checkpoint into a fresh :class:`CarbonSLANetPolicy`, then
        iterates over the PPO policy's named parameters and copies the first
        unmatched BC parameter with the same shape (excluding LayerNorm
        tensors, identified by having 'norm' in their name).

        Returns
        -------
        Number of parameter tensors successfully transferred.
        """
        bc_policy = CarbonSLANetPolicy(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        )
        bc_policy.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )

        bc_params = dict(bc_policy.named_parameters())
        ppo_params = dict(ppo_model.policy.named_parameters())

        transferred = 0
        used_bc_names: set[str] = set()

        for ppo_name, ppo_param in ppo_params.items():
            for bc_name, bc_param in bc_params.items():
                if bc_name in used_bc_names:
                    continue
                if bc_param.shape != ppo_param.shape:
                    continue
                is_weight = "weight" in bc_name and "weight" in ppo_name
                is_bias = "bias" in bc_name and "bias" in ppo_name
                if not (is_weight or is_bias):
                    continue
                if "norm" in bc_name or "norm" in ppo_name:
                    continue
                ppo_param.data.copy_(bc_param.data)
                transferred += 1
                used_bc_names.add(bc_name)
                break

        n_ppo = len(ppo_params)
        print(f"BC → PPO: transferred {transferred}/{n_ppo} parameter tensors")

        if transferred == 0:
            warnings.warn(
                "No parameters transferred — check architecture alignment.",
                RuntimeWarning,
                stacklevel=2,
            )

        return transferred
