"""
3-phase curriculum training to solve the credit assignment problem.

Phase 0 (1M steps): t_ho_prep=1, t_ho_exec=1, no RLF/PP termination
  - Near-instant HO: any action immediately affects serving BS within 2 steps
  - P(HO trigger per step) = 80% -> agent learns "pick best SINR BS" quickly
  - terminate_on_rlf=False: no early episode end, agent sees full reward signal

Phase 1 (2M steps): t_ho_prep=3, t_ho_exec=2, RLF terminates, no PP
  - Transition to realistic timing; policy already knows which BS to prefer

Phase 2 (2M steps): t_ho_prep=5, t_ho_exec=4, full paper params (RLF+PP)
  - Full paper constraints: agent fine-tunes timing and PP avoidance
"""

import os
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
from ho_optim_drl.gym_env import HandoverEnvPPO
import ho_optim_drl.utils as ut

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "ppo_curriculum"


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


def make_env(config, rsrp_list, sinr_list, sinr_norm_list):
    HandoverEnvPPO.reset_cls()
    return HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)


def make_model(env, config, tensorboard_log=None, lr=5e-5):
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=config.net_arch, vf=config.net_arch),
    )
    return PPO(
        "MlpPolicy",
        env,
        ent_coef=config.ent_coef,
        learning_rate=lr,
        verbose=1,
        policy_kwargs=policy_kwargs,
        n_steps=config.n_steps_per_update,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        tensorboard_log=tensorboard_log,
        device=torch.device("cpu"),
    )


def set_phase(config, t_ho_prep, t_ho_exec, terminate_rlf, terminate_pp):
    config.t_ho_prep = t_ho_prep
    config.t_ho_exec = t_ho_exec
    config.terminate_on_rlf = terminate_rlf
    config.terminate_on_pp = terminate_pp


def main():
    config = Config()
    data_dir = os.path.join(ROOT_PATH, "data", "processed")
    rsrp_list, sinr_list, sinr_norm_list = load_datasets(config, data_dir, [30, 50])

    sim_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_log = os.path.join(ROOT_PATH, "results", "tensorboard", MODEL_NAME, sim_id)

    # ── Phase 0: Near-instant HO, no penalties ────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 0: t_ho_prep=1, t_ho_exec=1, terminate_on_rlf=False")
    print("  P(HO trigger/step) = 80% -> immediate reward feedback")
    print("=" * 60)
    set_phase(config, t_ho_prep=1, t_ho_exec=1, terminate_rlf=False, terminate_pp=False)

    env0 = make_env(config, rsrp_list, sinr_list, sinr_norm_list)
    check_env(env0, warn=True)
    model = make_model(env0, config, tensorboard_log=tb_log, lr=5e-5)

    phase0_steps = 1_000_000
    model.learn(total_timesteps=phase0_steps, progress_bar=True, tb_log_name="phase0")
    print(f"Phase 0 done: {phase0_steps:,} steps")

    # checkpoint
    ckpt_dir = os.path.join(ROOT_PATH, "results", "models", MODEL_NAME)
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(os.path.join(ckpt_dir, "model_phase0"))
    print("Checkpoint: model_phase0.zip")

    # ── Phase 1: Intermediate delay, RLF terminates ───────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: t_ho_prep=3, t_ho_exec=2, terminate_on_rlf=True")
    print("  Agent adapts timing from 1->3 steps; knows which BS to pick")
    print("=" * 60)
    set_phase(config, t_ho_prep=3, t_ho_exec=2, terminate_rlf=True, terminate_pp=False)

    env1 = make_env(config, rsrp_list, sinr_list, sinr_norm_list)
    model.set_env(env1)
    model.lr_schedule = lambda _: 3e-5

    phase1_steps = 2_000_000
    model.learn(
        total_timesteps=phase1_steps,
        progress_bar=True,
        tb_log_name="phase1",
        reset_num_timesteps=True,
    )
    print(f"Phase 1 done: {phase1_steps:,} steps")
    model.save(os.path.join(ckpt_dir, "model_phase1"))
    print("Checkpoint: model_phase1.zip")

    # ── Phase 2: Full paper params ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: t_ho_prep=5, t_ho_exec=4, terminate_on_rlf=True, terminate_on_pp=True")
    print("  Full paper constraints: fine-tune timing + PP avoidance")
    print("=" * 60)
    set_phase(config, t_ho_prep=5, t_ho_exec=4, terminate_rlf=True, terminate_pp=True)

    env2 = make_env(config, rsrp_list, sinr_list, sinr_norm_list)
    model.set_env(env2)
    model.lr_schedule = lambda _: 1e-5

    phase2_steps = 2_000_000
    model.learn(
        total_timesteps=phase2_steps,
        progress_bar=True,
        tb_log_name="phase2",
        reset_num_timesteps=True,
    )
    print(f"Phase 2 done: {phase2_steps:,} steps")

    # ── Save final model ──────────────────────────────────────────────────────
    model_path = os.path.join(ckpt_dir, "model")
    model.save(model_path)
    print(f"\nFinal model saved: {model_path}.zip")

    return 0


if __name__ == "__main__":
    main()
