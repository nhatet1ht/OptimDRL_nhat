"""
2-phase PPO training with action-based reward shaping.

Problem: with uniform policy, reward is independent of action (pcell never changes
because HO never completes). Advantage A(s,a) ≈ 0 → gradient ≈ 0.

Fix: add a small immediate reward based on what BS the agent points to:
  r_shaped = alpha * sinr_norm[action]

This provides gradient signal BEFORE any HO completes:
  - action = best_BS -> r_shaped ≈ alpha*1.0 (high)
  - action = worst_BS -> r_shaped ≈ alpha*0.0 (low)
  - Advantage A(s, best_BS) > A(s, worst_BS) -> policy shifts toward best_BS
  - Consistently picking best_BS for 5 steps triggers HO -> large SINR reward
  - Positive feedback loop kicks in

alpha = 0.1 (10% of reward constant C=0.95) to not override the main signal.

This is potential-based shaping: F(s,a) = alpha*sinr_norm[a] where 0<=sinr_norm<=1.
"""

import os
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
from ho_optim_drl.gym_env import HandoverEnvPPO
import ho_optim_drl.utils as ut

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "ppo_shaped"
SHAPING_ALPHA = 0.1  # small enough not to override SINR reward


class HandoverEnvShaped(HandoverEnvPPO):
    """HandoverEnvPPO with action-based reward shaping."""

    def _get_reward(self) -> float:
        """Standard reward plus small shaping term on chosen action."""
        reward = super()._get_reward()

        # Shaping: encourage pointing to high-SINR BS
        sinr_norm = self.sinr_norm_list[self.dataset_idx][self.t, :]
        action = self.s_action[-1]
        reward += SHAPING_ALPHA * float(sinr_norm[action])

        return reward


def load_datasets(config, data_dir, speed_list):
    rsrp_files = dl.get_filenames(data_dir, "rsrp")
    sinr_files = dl.get_filenames(data_dir, "sinr")
    rsrp_files, sinr_files, _ = ut.filenames_speed_filter(rsrp_files, sinr_files, speed_list)

    rsrp_list, sinr_list, sinr_norm_list = [], [], []
    for rf, sf in zip(rsrp_files, sinr_files):
        rsrp_db, sinr_db = dl.load_preprocess_dataset(config, data_dir, rf, sf)
        sinr_norm = ut.clipnorm(sinr_db, config.sinr_lower_clip, config.sinr_upper_clip)
        rsrp_list.append(rsrp_db)
        sinr_list.append(sinr_db)
        sinr_norm_list.append(sinr_norm)
    return rsrp_list, sinr_list, sinr_norm_list


def make_model(env, config, tensorboard_log=None):
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=config.net_arch, vf=config.net_arch),
    )
    return PPO(
        "MlpPolicy",
        env,
        ent_coef=config.ent_coef,
        learning_rate=5e-5,
        verbose=1,
        policy_kwargs=policy_kwargs,
        n_steps=config.n_steps_per_update,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        tensorboard_log=tensorboard_log,
        device=torch.device("cpu"),
    )


def main():
    config = Config()
    data_dir = os.path.join(ROOT_PATH, "data", "processed")
    rsrp_list, sinr_list, sinr_norm_list = load_datasets(config, data_dir, [30, 50])

    sim_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_log = os.path.join(ROOT_PATH, "results", "tensorboard", MODEL_NAME, sim_id)

    # ── Phase 1: learn HO strategy, no PP termination ─────────────────────────
    print("\n" + "=" * 60)
    print(f"PHASE 1 (shaped): terminate_on_pp=False, alpha={SHAPING_ALPHA}")
    print("=" * 60)
    config.terminate_on_pp = False
    config.terminate_on_rlf = True

    HandoverEnvPPO.reset_cls()
    env1 = HandoverEnvShaped(config, rsrp_list, sinr_list, sinr_norm_list)
    check_env(env1, warn=True)

    model = make_model(env1, config, tensorboard_log=tb_log)
    phase1_steps = config.n_steps_total // 2  # 2.5M

    model.learn(total_timesteps=phase1_steps, progress_bar=True, tb_log_name="phase1")
    print(f"Phase 1 done: {phase1_steps:,} steps")

    # ── Phase 2: add PP termination ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PHASE 2 (shaped): terminate_on_pp=True, alpha={SHAPING_ALPHA}")
    print("=" * 60)
    config.terminate_on_pp = True

    HandoverEnvPPO.reset_cls()
    env2 = HandoverEnvShaped(config, rsrp_list, sinr_list, sinr_norm_list)
    model.set_env(env2)
    model.lr_schedule = lambda _: config.lr * 0.2  # constant 1e-5

    phase2_steps = config.n_steps_total - phase1_steps  # 2.5M
    model.learn(
        total_timesteps=phase2_steps,
        progress_bar=True,
        tb_log_name="phase2",
        reset_num_timesteps=True,
    )
    print(f"Phase 2 done: {phase2_steps:,} steps")

    # ── Save ──────────────────────────────────────────────────────────────────
    model_dir = os.path.join(ROOT_PATH, "results", "models", MODEL_NAME)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model")
    model.save(model_path)
    print(f"\nModel saved: {model_path}.zip")

    return 0


if __name__ == "__main__":
    main()
