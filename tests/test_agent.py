"""Tests for Phase 3: policy network, BC trainer, weight transfer, RL trainer."""

from __future__ import annotations

import os
import pickle
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO

from agent.policy_net import CarbonSLANetPolicy
from agent.bc_trainer import BCTrainer, BCTrainingResult
from agent.rl_trainer import RLTrainer, RLTrainingResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _random_obs(batch: int = 4) -> torch.Tensor:
    return torch.rand(batch, 74)


def _synthetic_dataset(n: int = 50, n_actions: int = 11) -> list:
    rng = np.random.default_rng(0)
    obs = rng.random((n, 74)).astype(np.float32)
    actions = rng.integers(0, n_actions, size=n)
    return [(obs[i], int(actions[i])) for i in range(n)]


def _save_dataset(dataset: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(dataset, fh)


# ===========================================================================
# 1–8: Policy network tests
# ===========================================================================

def test_policy_init():
    """1. CarbonSLANetPolicy initialises without error."""
    policy = CarbonSLANetPolicy(obs_dim=74, n_actions=11)
    assert isinstance(policy, torch.nn.Module)


def test_policy_forward_shapes():
    """2. forward() returns (logits, value) with correct shapes."""
    policy = CarbonSLANetPolicy()
    obs = _random_obs(4)
    logits, value = policy(obs)
    assert logits.shape == (4, 11)
    assert value.shape == (4, 1)


def test_policy_get_action_logits_shape():
    """3. get_action_logits() returns shape (B, n_actions)."""
    policy = CarbonSLANetPolicy()
    logits = policy.get_action_logits(_random_obs(8))
    assert logits.shape == (8, 11)


def test_policy_get_value_shape():
    """4. get_value() returns shape (B, 1)."""
    policy = CarbonSLANetPolicy()
    v = policy.get_value(_random_obs(8))
    assert v.shape == (8, 1)


@pytest.mark.parametrize("batch", [1, 32])
def test_policy_batch_sizes(batch):
    """5. Output shapes are correct for batch size 1 and 32."""
    policy = CarbonSLANetPolicy()
    logits, value = policy(_random_obs(batch))
    assert logits.shape == (batch, 11)
    assert value.shape == (batch, 1)


def test_policy_no_nan_inf():
    """6. No NaN or Inf in outputs for random obs in [0, 1]."""
    policy = CarbonSLANetPolicy()
    obs = _random_obs(16)
    logits, value = policy(obs)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_policy_deterministic_zero_dropout():
    """7. With dropout=0.0 and eval mode, same input → same output."""
    policy = CarbonSLANetPolicy(dropout=0.0)
    policy.eval()
    obs = _random_obs(4)
    out1 = policy.get_action_logits(obs)
    out2 = policy.get_action_logits(obs)
    assert torch.allclose(out1, out2)


def test_policy_has_parameters():
    """8. Policy has trainable parameters."""
    policy = CarbonSLANetPolicy()
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    assert n_params > 0


# ===========================================================================
# 9–18: BC trainer tests
# ===========================================================================

def test_bc_trainer_init():
    """9. BCTrainer initialises without error."""
    trainer = BCTrainer(device="cpu")
    assert isinstance(trainer, BCTrainer)
    assert isinstance(trainer.policy, CarbonSLANetPolicy)


def test_bc_load_dataset_shapes(tmp_path):
    """10. load_dataset() returns tensors of correct shapes."""
    ds = _synthetic_dataset(60)
    path = str(tmp_path / "ds.pkl")
    _save_dataset(ds, path)
    trainer = BCTrainer(device="cpu")
    obs_t, act_t = trainer.load_dataset(path)
    assert obs_t.shape == (60, 74)
    assert act_t.shape == (60,)
    assert obs_t.dtype == torch.float32
    assert act_t.dtype == torch.int64


def test_bc_load_dataset_invalid_action(tmp_path):
    """11. load_dataset() raises ValueError for out-of-range action."""
    ds = _synthetic_dataset(10)
    ds[3] = (ds[3][0], 99)  # invalid action
    path = str(tmp_path / "ds_bad.pkl")
    _save_dataset(ds, path)
    trainer = BCTrainer(device="cpu")
    with pytest.raises(ValueError, match="Actions out of"):
        trainer.load_dataset(path)


def test_bc_load_dataset_clips_obs(tmp_path):
    """12. load_dataset() clips obs and emits a RuntimeWarning."""
    rng = np.random.default_rng(1)
    obs = rng.random((10, 74)).astype(np.float32)
    obs[0, 0] = 1.5  # out of range
    ds = [(obs[i], 0) for i in range(10)]
    path = str(tmp_path / "ds_clip.pkl")
    _save_dataset(ds, path)
    trainer = BCTrainer(device="cpu")
    with pytest.warns(RuntimeWarning, match="clipping"):
        obs_t, _ = trainer.load_dataset(path)
    assert float(obs_t.max()) <= 1.0


def test_bc_train_runs(tmp_path):
    """13. train() runs at least 1 epoch on 50 pairs without error."""
    ds = _synthetic_dataset(50)
    ds_path = str(tmp_path / "ds.pkl")
    _save_dataset(ds, ds_path)
    trainer = BCTrainer(device="cpu")
    result = trainer.train(
        dataset_path=ds_path,
        epochs=2,
        batch_size=16,
        checkpoint_dir=str(tmp_path / "ckpt"),
        run_name="test",
    )
    assert result.total_epochs_run >= 1


def test_bc_train_returns_result(tmp_path):
    """14. train() returns a BCTrainingResult with all required fields."""
    ds = _synthetic_dataset(50)
    ds_path = str(tmp_path / "ds.pkl")
    _save_dataset(ds, ds_path)
    trainer = BCTrainer(device="cpu")
    result = trainer.train(
        dataset_path=ds_path,
        epochs=2,
        batch_size=16,
        checkpoint_dir=str(tmp_path / "ckpt"),
        run_name="test",
    )
    assert isinstance(result, BCTrainingResult)
    assert hasattr(result, "train_losses")
    assert hasattr(result, "val_losses")
    assert hasattr(result, "val_accuracies")
    assert hasattr(result, "best_val_loss")
    assert hasattr(result, "best_val_accuracy")
    assert hasattr(result, "best_epoch")
    assert hasattr(result, "total_epochs_run")
    assert hasattr(result, "checkpoint_path")


def test_bc_train_saves_checkpoint(tmp_path):
    """15. train() saves a checkpoint file that exists on disk."""
    ds = _synthetic_dataset(50)
    ds_path = str(tmp_path / "ds.pkl")
    _save_dataset(ds, ds_path)
    trainer = BCTrainer(device="cpu")
    result = trainer.train(
        dataset_path=ds_path,
        epochs=2,
        batch_size=16,
        checkpoint_dir=str(tmp_path / "ckpt"),
        run_name="test",
    )
    assert os.path.isfile(result.checkpoint_path)


def test_bc_train_early_stopping(tmp_path):
    """16. patience=1 stops training early on a tiny dataset."""
    ds = _synthetic_dataset(50)
    ds_path = str(tmp_path / "ds.pkl")
    _save_dataset(ds, ds_path)
    trainer = BCTrainer(device="cpu")
    result = trainer.train(
        dataset_path=ds_path,
        epochs=20,
        batch_size=16,
        patience=1,
        checkpoint_dir=str(tmp_path / "ckpt"),
        run_name="early_test",
    )
    assert result.total_epochs_run < 20


def test_bc_best_val_accuracy_range(tmp_path):
    """17. BCTrainingResult.best_val_accuracy is in [0.0, 1.0]."""
    ds = _synthetic_dataset(50)
    ds_path = str(tmp_path / "ds.pkl")
    _save_dataset(ds, ds_path)
    trainer = BCTrainer(device="cpu")
    result = trainer.train(
        dataset_path=ds_path,
        epochs=2,
        batch_size=16,
        checkpoint_dir=str(tmp_path / "ckpt"),
        run_name="test",
    )
    assert 0.0 <= result.best_val_accuracy <= 1.0


def test_bc_train_losses_length(tmp_path):
    """18. BCTrainingResult.train_losses has length == total_epochs_run."""
    ds = _synthetic_dataset(50)
    ds_path = str(tmp_path / "ds.pkl")
    _save_dataset(ds, ds_path)
    trainer = BCTrainer(device="cpu")
    result = trainer.train(
        dataset_path=ds_path,
        epochs=3,
        batch_size=16,
        checkpoint_dir=str(tmp_path / "ckpt"),
        run_name="test",
    )
    assert len(result.train_losses) == result.total_epochs_run


# ===========================================================================
# 19–21: Weight transfer tests
# ===========================================================================

@pytest.fixture(scope="module")
def bc_checkpoint(tmp_path_factory):
    """Train for 1 epoch and return the checkpoint path."""
    tmp = tmp_path_factory.mktemp("bc_ckpt")
    ds = _synthetic_dataset(80)
    ds_path = str(tmp / "ds.pkl")
    _save_dataset(ds, ds_path)
    trainer = BCTrainer(device="cpu")
    result = trainer.train(
        dataset_path=ds_path,
        epochs=1,
        batch_size=16,
        checkpoint_dir=str(tmp / "ckpt"),
        run_name="transfer_test",
    )
    return result.checkpoint_path


def test_transfer_returns_positive(bc_checkpoint):
    """19. transfer_weights_to_sb3() returns an integer > 0."""
    from env.cloud_edge_env import CloudEdgeEnv
    env = CloudEdgeEnv(N=10, T=4, seed=0)
    model = MaskablePPO(
        "MlpPolicy", env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0,
    )
    trainer = BCTrainer(device="cpu")
    n = trainer.transfer_weights_to_sb3(model, bc_checkpoint)
    assert isinstance(n, int)
    assert n > 0


def test_transfer_changes_params(bc_checkpoint):
    """20. After transfer, at least one PPO parameter differs from random init."""
    from env.cloud_edge_env import CloudEdgeEnv
    env = CloudEdgeEnv(N=10, T=4, seed=0)

    # Record random-init values
    model_before = MaskablePPO(
        "MlpPolicy", env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0, seed=1,
    )
    before = {
        name: param.data.clone()
        for name, param in model_before.policy.named_parameters()
    }

    # Transfer
    model_after = MaskablePPO(
        "MlpPolicy", env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0, seed=1,
    )
    trainer = BCTrainer(device="cpu")
    trainer.transfer_weights_to_sb3(model_after, bc_checkpoint)

    changed = any(
        not torch.equal(before[name], param.data)
        for name, param in model_after.policy.named_parameters()
    )
    assert changed, "No parameter changed after transfer"


def test_transfer_warns_no_match(tmp_path):
    """21. transfer_weights_to_sb3() warns when no params match."""
    # Use a real BC checkpoint
    ckpt_path = str(tmp_path / "tiny_best.pt")
    trainer = BCTrainer(obs_dim=74, n_actions=11, device="cpu")
    torch.save(trainer.policy.state_dict(), ckpt_path)

    # Fake PPO whose policy exposes zero named parameters → guaranteed 0 transfers
    class _FakePolicy:
        def named_parameters(self):
            return iter([])

    class _FakePPO:
        policy = _FakePolicy()

    with pytest.warns(RuntimeWarning, match="No parameters transferred"):
        n = trainer.transfer_weights_to_sb3(_FakePPO(), ckpt_path)
    assert n == 0


# ===========================================================================
# 22–27: RL trainer tests
# ===========================================================================

@pytest.fixture(scope="module")
def small_rl_trainer():
    return RLTrainer(N=10, T=4, n_envs=2, seed=0, checkpoint_dir="checkpoints/ppo_test", log_dir="results/ppo_test")


def test_rl_trainer_init(small_rl_trainer):
    """22. RLTrainer initialises without error."""
    assert isinstance(small_rl_trainer, RLTrainer)


def test_rl_build_model(small_rl_trainer):
    """23. build_model() returns a MaskablePPO instance."""
    model = small_rl_trainer.build_model()
    assert isinstance(model, MaskablePPO)
    # Close envs to avoid resource leak
    if small_rl_trainer.train_envs is not None:
        small_rl_trainer.train_envs.close()
        small_rl_trainer.train_envs = None


@pytest.fixture(scope="module")
def rl_training_result(small_rl_trainer):
    """Run a minimal training loop once and reuse across tests."""
    return small_rl_trainer.train(
        total_timesteps=1024,
        eval_freq=512,
        n_eval_episodes=2,
        model_name="pytest_run",
    )


def test_rl_train_completes(rl_training_result):
    """24. train() completes without error for small parameters."""
    assert isinstance(rl_training_result, RLTrainingResult)


def test_rl_checkpoint_exists(rl_training_result, small_rl_trainer):
    """25. A checkpoint file exists in checkpoint_dir after training."""
    ckpt_dir = Path(small_rl_trainer.checkpoint_dir)
    zips = list(ckpt_dir.rglob("*.zip"))
    assert len(zips) > 0, f"No .zip checkpoints found in {ckpt_dir}"


def test_rl_evaluate_has_mean_reward(rl_training_result, small_rl_trainer):
    """26. evaluate_model() returns a dict with key 'mean_reward'."""
    model_path = rl_training_result.best_model_path
    if not os.path.isfile(model_path):
        # Fall back to any available checkpoint
        ckpt_dir = Path(small_rl_trainer.checkpoint_dir)
        zips = list(ckpt_dir.rglob("*.zip"))
        pytest.skip("No model zip found for evaluation") if not zips else None
        model_path = str(zips[0])

    result = small_rl_trainer.evaluate_model(
        model_path=model_path, n_episodes=2, seed_offset=20_000
    )
    assert "mean_reward" in result


def test_rl_evaluate_inference_ms(rl_training_result, small_rl_trainer):
    """27. evaluate_model() mean_inference_ms is a positive finite float."""
    model_path = rl_training_result.best_model_path
    if not os.path.isfile(model_path):
        ckpt_dir = Path(small_rl_trainer.checkpoint_dir)
        zips = list(ckpt_dir.rglob("*.zip"))
        if not zips:
            pytest.skip("No model zip found for evaluation")
        model_path = str(zips[0])

    result = small_rl_trainer.evaluate_model(
        model_path=model_path, n_episodes=2, seed_offset=20_000
    )
    ms = result["mean_inference_ms"]
    assert isinstance(ms, float)
    assert ms > 0.0
    assert np.isfinite(ms)
