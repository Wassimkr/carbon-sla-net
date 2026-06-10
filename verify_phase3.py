"""Phase 3 verification script for Carbon-SLA-Net.

Run with:  python verify_phase3.py

Exercises the policy network, BC trainer, weight transfer, and PPO pipeline.
No tracebacks should occur.

The ``if __name__ == '__main__'`` guard is required on macOS (which uses the
'spawn' multiprocessing start method by default).  Without it, SubprocVecEnv
worker processes re-import this file as ``__main__`` and re-execute the body,
causing an infinite recursion crash.  On Linux ('fork' default) the guard is
not strictly necessary but is harmless.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from agent.policy_net import CarbonSLANetPolicy
from agent.bc_trainer import BCTrainer
from agent.rl_trainer import RLTrainer
from milp.oracle_runner import generate_bc_dataset
from sb3_contrib import MaskablePPO
from env.cloud_edge_env import CloudEdgeEnv


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


if __name__ == "__main__":

    # ---------------------------------------------------------------------------
    # 1. Policy smoke test
    # ---------------------------------------------------------------------------
    section("1. Policy network forward pass (batch=8)")
    policy = CarbonSLANetPolicy(obs_dim=74, n_actions=11)
    obs_batch = torch.rand(8, 74)
    logits, value = policy(obs_batch)
    print(f"  logits shape : {tuple(logits.shape)}")
    print(f"  value shape  : {tuple(value.shape)}")
    assert logits.shape == (8, 11), "unexpected logits shape"
    assert value.shape == (8, 1), "unexpected value shape"
    nan_ok = torch.isfinite(logits).all() and torch.isfinite(value).all()
    print(f"  No NaN/Inf   : {'PASS' if nan_ok else 'FAIL'}")

    # ---------------------------------------------------------------------------
    # 2. BC dataset check / generation
    # ---------------------------------------------------------------------------
    section("2. BC dataset availability check")
    BC_PATH = "data/bc_dataset/bc_500.pkl"

    if os.path.isfile(BC_PATH):
        import pickle
        with open(BC_PATH, "rb") as fh:
            _ds = pickle.load(fh)
        print(f"  Found {BC_PATH} with {len(_ds)} pairs — skipping generation")
    else:
        print(f"  {BC_PATH} not found — generating 30-episode sample …")
        _ds = generate_bc_dataset(
            n_episodes=30,
            N=100,
            T=8,
            output_path=BC_PATH,
            time_limit=120.0,
            verbose=True,
        )
        print(f"  Generated {len(_ds)} pairs → {BC_PATH}")

    assert len(_ds) > 0, "BC dataset is empty — check MILP solver"

    # ---------------------------------------------------------------------------
    # 3. BC training mini-run (5 epochs)
    # ---------------------------------------------------------------------------
    section("3. BC training mini-run (5 epochs)")
    trainer = BCTrainer(device="auto", dropout=0.1)
    bc_result = trainer.train(
        dataset_path=BC_PATH,
        epochs=5,
        batch_size=256,
        checkpoint_dir="checkpoints/bc",
        run_name="verify_run",
    )
    print(f"  epochs run   : {bc_result.total_epochs_run}")
    print(f"  best epoch   : {bc_result.best_epoch}")
    print(f"  best val_loss: {bc_result.best_val_loss:.4f}")
    print(f"  best val_acc : {bc_result.best_val_accuracy:.3f}")

    # ---------------------------------------------------------------------------
    # 4. Checkpoint integrity
    # ---------------------------------------------------------------------------
    section("4. BC checkpoint integrity")
    ckpt_path = bc_result.checkpoint_path
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    expected_keys = {"encoder.0.linear.weight", "encoder.0.linear.bias",
                     "encoder.1.linear.weight", "encoder.1.linear.bias",
                     "actor.weight", "critic.weight"}
    found_keys = set(state_dict.keys())
    missing = expected_keys - found_keys
    print(f"  Checkpoint path : {ckpt_path}")
    print(f"  Total keys      : {len(found_keys)}")
    print(f"  Required present: {'PASS' if not missing else 'FAIL — missing: ' + str(missing)}")

    # ---------------------------------------------------------------------------
    # 5. Weight transfer to MaskablePPO
    # ---------------------------------------------------------------------------
    section("5. Weight transfer BC → MaskablePPO")
    env_for_model = CloudEdgeEnv(N=50, T=8, seed=0)
    ppo_model = MaskablePPO(
        "MlpPolicy",
        env_for_model,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0,
    )
    n_transferred = trainer.transfer_weights_to_sb3(ppo_model, ckpt_path)
    print(f"  Transferred tensors: {n_transferred}")
    assert n_transferred >= 4, (
        f"Expected ≥ 4 transfers (2 weight + 2 bias for encoder), got {n_transferred}"
    )
    print(f"  Transfer ≥ 4    : PASS")

    # ---------------------------------------------------------------------------
    # 6. RL smoke test
    # ---------------------------------------------------------------------------
    section("6. RL training smoke test (N=50, T=8, n_envs=2, 2048 steps)")
    rl = RLTrainer(
        N=50,
        T=8,
        n_envs=2,
        seed=42,
        log_dir="results/ppo",
        checkpoint_dir="checkpoints/ppo",
    )
    rl_result = rl.train(
        total_timesteps=2048,
        eval_freq=1024,
        n_eval_episodes=5,
        bc_checkpoint_path=ckpt_path,
        model_name="verify_run",
    )
    print(f"  total_timesteps : {rl_result.total_timesteps}")
    print(f"  best_mean_reward: {rl_result.best_mean_reward:.4f}")
    print(f"  training_time_s : {rl_result.training_time_s:.2f}")
    print(f"  best_model_path : {rl_result.best_model_path}")

    # ---------------------------------------------------------------------------
    # 7. Evaluation smoke test
    # ---------------------------------------------------------------------------
    section("7. Evaluation smoke test (5 episodes)")
    model_path = rl_result.best_model_path
    if not os.path.isfile(model_path):
        from pathlib import Path as _Path
        zips = list(_Path("checkpoints/ppo").rglob("*.zip"))
        assert zips, "No .zip checkpoint found after training"
        model_path = str(zips[0])
        print(f"  (best_model not found, using fallback: {model_path})")

    eval_metrics = rl.evaluate_model(
        model_path=model_path,
        n_episodes=5,
        seed_offset=20_000,
    )
    print(f"  mean_reward      : {eval_metrics['mean_reward']:.4f}")
    print(f"  mean_energy_wh   : {eval_metrics['mean_energy_wh']:.2f}")
    print(f"  mean_carbon_gco2 : {eval_metrics['mean_carbon_gco2']:.4f}")
    print(f"  mean_inference_ms: {eval_metrics['mean_inference_ms']:.2f}")
    assert eval_metrics["mean_inference_ms"] > 0.0, "Inference time must be positive"
    print(f"  inference_ms > 0 : PASS")

    # ---------------------------------------------------------------------------
    print("\nPhase 3 complete. Policy network, BC trainer, and PPO fine-tuning pipeline verified.")
