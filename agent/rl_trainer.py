"""MaskablePPO fine-tuning pipeline for Carbon-SLA-Net Phase 3b.

Builds vectorised training environments, optionally initialises from a BC
checkpoint, and runs MaskablePPO with EvalCallback + CheckpointCallback.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import tensorboard  # noqa: F401
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from agent.bc_trainer import BCTrainer
from env.cloud_edge_env import CloudEdgeEnv


_DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "energy": 0.3,
    "carbon": 0.3,
    "sla": 0.3,
    "renewable": 0.1,
}


@dataclass
class RLTrainingResult:
    """Summary of a MaskablePPO training run."""

    total_timesteps: int
    final_mean_reward: float
    best_mean_reward: float
    best_model_path: str
    training_time_s: float
    n_eval_episodes: int


class RLTrainer:
    """MaskablePPO trainer with optional BC warm-start.

    Creates vectorised training environments via SubprocVecEnv and a single
    evaluation environment.  Supports warm-starting from a BC checkpoint via
    :meth:`BCTrainer.transfer_weights_to_sb3`.
    """

    def __init__(
        self,
        N: int = 100,
        T: int = 8,
        n_envs: int = 4,
        reward_weights: Optional[Dict[str, float]] = None,
        workload_pattern: str = "uniform",
        renewable_condition: str = "sunny",
        log_dir: str = "results/ppo",
        checkpoint_dir: str = "checkpoints/ppo",
        seed: int = 42,
    ) -> None:
        self.N = N
        self.T = T
        self.n_envs = n_envs
        self.reward_weights = dict(_DEFAULT_REWARD_WEIGHTS)
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)
        self.workload_pattern = workload_pattern
        self.renewable_condition = renewable_condition
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed

        # Single eval environment (seed far from training seeds)
        self.eval_env = CloudEdgeEnv(
            N=N,
            T=T,
            reward_weights=self.reward_weights,
            workload_pattern=workload_pattern,
            renewable_condition=renewable_condition,
            seed=seed + 9999,
        )

        self.train_envs = None
        self.model: Optional[MaskablePPO] = None

    # ------------------------------------------------------------------
    # Environment factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_env(
        N: int,
        T: int,
        seed: int,
        reward_weights: Dict[str, float],
        workload_pattern: str,
        renewable_condition: str,
    ):
        """Return a zero-argument factory suitable for SubprocVecEnv."""
        def _init() -> CloudEdgeEnv:
            return CloudEdgeEnv(
                N=N,
                T=T,
                reward_weights=reward_weights,
                workload_pattern=workload_pattern,
                renewable_condition=renewable_condition,
                seed=seed,
            )
        return _init

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(
        self,
        bc_checkpoint_path: Optional[str] = None,
        model_name: str = "carbon_sla_net",
    ) -> MaskablePPO:
        """Build a MaskablePPO model on vectorised training environments.

        Optionally warm-starts from *bc_checkpoint_path* by transferring
        weights via :meth:`BCTrainer.transfer_weights_to_sb3`.

        The rollout buffer always contains 2048 total steps regardless of
        n_envs.  If n_envs does not divide 2048 evenly, n_steps falls back
        to 512.
        """
        # Close any previously created envs to avoid resource leaks
        if self.train_envs is not None:
            self.train_envs.close()

        factories = [
            self._make_env(
                self.N, self.T,
                self.seed + i,
                self.reward_weights,
                self.workload_pattern,
                self.renewable_condition,
            )
            for i in range(self.n_envs)
        ]
        os.makedirs(self.log_dir, exist_ok=True)
        monitor_path = os.path.join(self.log_dir, model_name)
        self.train_envs = VecMonitor(SubprocVecEnv(factories), monitor_path)

        n_steps = 2048 // self.n_envs if 2048 % self.n_envs == 0 else 512

        self.model = MaskablePPO(
            policy="MlpPolicy",
            env=self.train_envs,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            tensorboard_log=self.log_dir if _TENSORBOARD_AVAILABLE else None,
            verbose=0,
            seed=self.seed,
        )

        if bc_checkpoint_path is not None:
            trainer = BCTrainer(device="cpu")
            trainer.transfer_weights_to_sb3(self.model, bc_checkpoint_path)

        return self.model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int = 500_000,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 20,
        bc_checkpoint_path: Optional[str] = None,
        model_name: str = "carbon_sla_net",
    ) -> RLTrainingResult:
        """Run MaskablePPO training with eval and checkpoint callbacks.

        Parameters
        ----------
        total_timesteps:
            Total environment steps across all envs.
        eval_freq:
            Steps between evaluations (in total env steps).
        n_eval_episodes:
            Episodes per evaluation call.
        bc_checkpoint_path:
            Optional BC checkpoint for warm-starting.
        model_name:
            Prefix for checkpoint files.

        Returns
        -------
        :class:`RLTrainingResult` with training summary.
        """
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        self.build_model(bc_checkpoint_path, model_name=model_name)

        best_model_dir = str(
            Path(self.checkpoint_dir) / f"{model_name}_best"
        )

        eval_cb = MaskableEvalCallback(
            eval_env=self.eval_env,
            best_model_save_path=best_model_dir,
            log_path=self.log_dir,
            eval_freq=max(eval_freq // self.n_envs, 1),
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=0,
        )

        ckpt_cb = CheckpointCallback(
            save_freq=max(100_000 // self.n_envs, 1),
            save_path=self.checkpoint_dir,
            name_prefix=model_name,
            verbose=0,
        )

        t0 = time.perf_counter()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList([eval_cb, ckpt_cb]),
            progress_bar=False,
            reset_num_timesteps=True,
        )
        training_time_s = time.perf_counter() - t0

        self.train_envs.close()
        self.train_envs = None

        best_mean = float(eval_cb.best_mean_reward) if hasattr(eval_cb, "best_mean_reward") else float("nan")
        last_mean = float(eval_cb.last_mean_reward) if hasattr(eval_cb, "last_mean_reward") else float("nan")

        return RLTrainingResult(
            total_timesteps=total_timesteps,
            final_mean_reward=last_mean,
            best_mean_reward=best_mean,
            best_model_path=str(Path(best_model_dir) / "best_model.zip"),
            training_time_s=training_time_s,
            n_eval_episodes=n_eval_episodes,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model_path: str,
        n_episodes: int = 50,
        seed_offset: int = 20_000,
    ) -> Dict:
        """Load a saved model and evaluate on fresh episodes.

        Parameters
        ----------
        model_path:
            Path to a ``.zip`` checkpoint saved by MaskablePPO.
        n_episodes:
            Number of evaluation episodes.
        seed_offset:
            Starting seed; must be far from training seeds to prevent
            data leakage between training and evaluation distributions.

        Returns
        -------
        Dict with mean/std reward and per-metric means.
        """
        model = MaskablePPO.load(model_path)

        rewards: List[float] = []
        energies: List[float] = []
        carbons: List[float] = []
        sla_viols: List[float] = []
        re_useds: List[float] = []
        inference_ms_list: List[float] = []

        for ep in range(n_episodes):
            env = CloudEdgeEnv(
                N=self.N,
                T=self.T,
                reward_weights=self.reward_weights,
                workload_pattern=self.workload_pattern,
                renewable_condition=self.renewable_condition,
                seed=seed_offset + ep,
            )
            obs, _ = env.reset(seed=seed_offset + ep)
            episode_reward = 0.0
            terminal_info: Dict = {}
            t0 = time.perf_counter()

            done = False
            while not done:
                masks = env.action_masks()
                action, _ = model.predict(
                    obs, action_masks=masks, deterministic=True
                )
                obs, reward, terminated, truncated, info = env.step(int(action))
                episode_reward += reward
                terminal_info = info
                done = terminated or truncated

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rewards.append(episode_reward)
            energies.append(float(terminal_info.get("energy_wh", 0.0)))
            carbons.append(float(terminal_info.get("carbon_gco2", 0.0)))
            sla_viols.append(float(terminal_info.get("sla_violation", 0.0)))
            re_useds.append(float(terminal_info.get("re_used_wh", 0.0)))
            inference_ms_list.append(elapsed_ms)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_energy_wh": float(np.mean(energies)),
            "mean_carbon_gco2": float(np.mean(carbons)),
            "mean_sla_violation": float(np.mean(sla_viols)),
            "mean_re_used_wh": float(np.mean(re_useds)),
            "mean_inference_ms": float(np.mean(inference_ms_list)),
            "n_episodes": n_episodes,
        }
