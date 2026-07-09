"""Two-phase PPO training matching the paper's description.

Phase 1: RLF terminates episode, PP does NOT. Agent focuses on learning HO strategies.
Phase 2: Both RLF and PP terminate episode. Agent learns to avoid excessive HOs.

NOTE on t_ho_prep modification:
  Reducing t_ho_prep from 5→3 to increase early HO frequency backfires:
  it causes too many RLF terminations early on, making rewards too noisy to converge.
  The original t_ho_prep=5 is kept throughout.
"""

import os
from datetime import datetime

import torch
from rl_zoo3 import linear_schedule
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
from ho_optim_drl.gym_env import HandoverEnvPPO
import ho_optim_drl.utils as ut

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "ppo_2phase"


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
        learning_rate=linear_schedule(config.lr),
        verbose=1,
        policy_kwargs=policy_kwargs,
        n_steps=config.n_steps_per_update,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        tensorboard_log=tensorboard_log,
        device=torch.device("cpu"),  # MLP runs faster on CPU
    )


def main():
    config = Config()
    data_dir = os.path.join(ROOT_PATH, "data", "processed")

    rsrp_list, sinr_list, sinr_norm_list = load_datasets(config, data_dir, [30, 50])

    sim_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_log = os.path.join(ROOT_PATH, "results", "tensorboard", MODEL_NAME, sim_id)

    # ── Phase 1: learn HO strategies, no PP termination ──────────────────────
    print("\n=== PHASE 1: terminate_on_pp=False, terminate_on_rlf=True ===")
    config.terminate_on_pp = False
    config.terminate_on_rlf = True
    config.t_ho_prep = 5  # keep original

    env1 = HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)
    check_env(env1, warn=True)

    model = make_model(env1, config, tensorboard_log=tb_log)
    phase1_steps = config.n_steps_total // 2  # 2.5M

    model.learn(total_timesteps=phase1_steps, progress_bar=True, tb_log_name="phase1")
    print(f"Phase 1 done: {phase1_steps:,} steps")

    # ── Phase 2: add PP termination, fresh lr schedule ────────────────────────
    print("\n=== PHASE 2: terminate_on_pp=True, terminate_on_rlf=True ===")
    config.terminate_on_pp = True

    env2 = HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)
    model.set_env(env2)

    # Use constant small lr for phase 2 fine-tuning (avoid linear schedule restart issue)
    model.lr_schedule = lambda _: config.lr * 0.2  # 1e-5 constant

    phase2_steps = config.n_steps_total - phase1_steps  # 2.5M
    model.learn(
        total_timesteps=phase2_steps,
        progress_bar=True,
        tb_log_name="phase2",
        reset_num_timesteps=True,  # reset counter so lr schedule works properly
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
