"""
Evaluate intermediate checkpoints with their original training config.

Useful for diagnosing: "did Phase 0 actually learn good behavior?"
Each checkpoint is evaluated with the t_ho_prep/t_ho_exec that was used
during that training phase, not the paper's final params.

Also provides a diagnostic tool to understand where learning broke down.
"""

import os

import numpy as np
from stable_baselines3 import PPO

from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
from ho_optim_drl.gym_env import HandoverEnvPPO
from ho_optim_drl.gym_env.ho_env_ppo import test_ppo_model
import ho_optim_drl.utils as ut

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USE_SPEEDS = [30, 50, 70, 90]


def load_datasets(config, data_dir, speed_list):
    rsrp_files = dl.get_filenames(data_dir, "rsrp")
    sinr_files = dl.get_filenames(data_dir, "sinr")
    rsrp_files, sinr_files, speeds = ut.filenames_speed_filter(
        rsrp_files, sinr_files, speed_list
    )
    rsrp_list, sinr_list, sinr_norm_list = [], [], []
    for rf, sf in zip(rsrp_files, sinr_files):
        rsrp_db, sinr_db = dl.load_preprocess_dataset(config, data_dir, rf, sf)
        sinr_norm = ut.clipnorm(sinr_db, config.sinr_lower_clip, config.sinr_upper_clip)
        rsrp_list.append(rsrp_db)
        sinr_list.append(sinr_db)
        sinr_norm_list.append(sinr_norm)
    return rsrp_list, sinr_list, sinr_norm_list, speeds


def eval_model(label, model_path, config, rsrp_list, sinr_list, sinr_norm_list, speeds):
    print(f"\n  Evaluating: {label}")
    HandoverEnvPPO.reset_cls()
    env = HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)
    model = PPO.load(model_path, env=env, tensorboard_log=None)

    result = ut.get_result_container(speeds)
    for i in range(env.n_datasets):
        test_ppo_model(env, model, i)
        stats = env.get_statistics()
        spd = speeds[i]
        result["sinr_connected"][spd].extend(env.ho_procedure.sinr_timeline)
        result["sinr_max"][spd].extend(list(np.max(env.sinr_list[env.dataset_idx], axis=1)))
        result["n_ho"][spd].append(stats["num_ho_exe_started"])
        result["n_pp"][spd].append(stats["num_pp"])
        result["n_rlf"][spd].append(stats["num_rlf"])

    metrics = {"speed": [], "r_rel": [], "pp_rate": [], "rlf_rate": []}
    for spd in np.unique(speeds):
        sinr_c = np.array(result["sinr_connected"][spd])
        sinr_m = np.array(result["sinr_max"][spd])
        sinr_c_lin = 10 ** (sinr_c / 10)
        sinr_m_lin = 10 ** (sinr_m / 10)
        sinr_c_lin[np.isnan(sinr_c_lin)] = 0
        sinr_m_lin[np.isnan(sinr_m_lin)] = 0
        r_mean = np.mean(config.bw * np.log2(1 + sinr_c_lin))
        r_max = np.mean(config.bw * np.log2(1 + sinr_m_lin))
        n_ho = np.mean(result["n_ho"][spd]) if result["n_ho"][spd] else 1
        n_pp = np.mean(result["n_pp"][spd]) if result["n_pp"][spd] else 0
        n_rlf = np.mean(result["n_rlf"][spd]) if result["n_rlf"][spd] else 0
        metrics["speed"].append(int(spd))
        metrics["r_rel"].append(r_mean / r_max if r_max > 0 else 0)
        metrics["pp_rate"].append(n_pp / n_ho if n_ho > 0 else 0)
        metrics["rlf_rate"].append(n_rlf / n_ho if n_ho > 0 else 0)

    return metrics


def print_metrics(label, metrics):
    print(f"\n  {label}")
    for i, spd in enumerate(metrics["speed"]):
        g = metrics["r_rel"][i] * 100
        pp = metrics["pp_rate"][i] * 100
        rlf = metrics["rlf_rate"][i] * 100
        print(f"    {spd} km/h: G={g:.3f}%  PP={pp:.1f}%  RLF={rlf:.2f}%")


def main():
    data_dir = os.path.join(ROOT_PATH, "data", "processed")

    print("=" * 70)
    print("CHECKPOINT DIAGNOSTIC: eval each model with its own training config")
    print("=" * 70)

    # ── 1. Curriculum Phase 0: t_ho_prep=1, t_ho_exec=1, no RLF/PP ────────────
    ckpt0 = os.path.join(ROOT_PATH, "results", "models", "ppo_curriculum", "model_phase0.zip")
    if os.path.exists(ckpt0):
        print("\n[1] Curriculum Phase 0 checkpoint (t_ho_prep=1, t_ho_exec=1)")
        print("    Question: did agent really learn to pick best BS?")
        config = Config()
        config.t_ho_prep = 1
        config.t_ho_exec = 1
        config.terminate_on_rlf = False
        config.terminate_on_pp = False
        rsrp_list, sinr_list, sinr_norm_list, speeds = load_datasets(config, data_dir, USE_SPEEDS)
        m = eval_model("Curriculum Phase 0 (t_ho_prep=1)", ckpt0, config,
                       rsrp_list, sinr_list, sinr_norm_list, speeds)
        print_metrics("Curriculum Phase 0 @ (t_ho_prep=1, t_ho_exec=1):", m)
    else:
        print(f"\n[1] SKIP: {ckpt0} not found")

    # ── 2. Curriculum Phase 1: t_ho_prep=3, t_ho_exec=2 ──────────────────────
    ckpt1 = os.path.join(ROOT_PATH, "results", "models", "ppo_curriculum", "model_phase1.zip")
    if os.path.exists(ckpt1):
        print("\n[2] Curriculum Phase 1 checkpoint (t_ho_prep=3, t_ho_exec=2)")
        print("    Question: how much did the transition to t_ho_prep=3 hurt?")
        config = Config()
        config.t_ho_prep = 3
        config.t_ho_exec = 2
        config.terminate_on_rlf = True
        config.terminate_on_pp = False
        rsrp_list, sinr_list, sinr_norm_list, speeds = load_datasets(config, data_dir, USE_SPEEDS)
        m = eval_model("Curriculum Phase 1 (t_ho_prep=3)", ckpt1, config,
                       rsrp_list, sinr_list, sinr_norm_list, speeds)
        print_metrics("Curriculum Phase 1 @ (t_ho_prep=3, t_ho_exec=2):", m)
    else:
        print(f"\n[2] SKIP: {ckpt1} not found")

    # ── 3. BC + Curriculum Phase 0 (if exists) ───────────────────────────────
    bc_ckpt0 = os.path.join(ROOT_PATH, "results", "models", "ppo_bc_curriculum", "model_phase0.zip")
    if os.path.exists(bc_ckpt0):
        print("\n[3] BC+Curriculum Phase 0 (t_ho_prep=1, t_ho_exec=1)")
        config = Config()
        config.t_ho_prep = 1
        config.t_ho_exec = 1
        config.terminate_on_rlf = False
        config.terminate_on_pp = False
        rsrp_list, sinr_list, sinr_norm_list, speeds = load_datasets(config, data_dir, USE_SPEEDS)
        m = eval_model("BC+Curriculum Phase 0", bc_ckpt0, config,
                       rsrp_list, sinr_list, sinr_norm_list, speeds)
        print_metrics("BC+Curriculum Phase 0:", m)
    else:
        print(f"\n[3] SKIP: {bc_ckpt0} not found (run train_ppo_bc_curriculum first)")

    # ── 4. BC + Curriculum final (if exists) ─────────────────────────────────
    bc_final = os.path.join(ROOT_PATH, "results", "models", "ppo_bc_curriculum", "model.zip")
    if os.path.exists(bc_final):
        print("\n[4] BC+Curriculum final (t_ho_prep=5, t_ho_exec=4 — paper params)")
        config = Config()
        config.t_ho_prep = 5
        config.t_ho_exec = 4
        config.terminate_on_rlf = False
        config.terminate_on_pp = False
        rsrp_list, sinr_list, sinr_norm_list, speeds = load_datasets(config, data_dir, USE_SPEEDS)
        m = eval_model("BC+Curriculum final", bc_final, config,
                       rsrp_list, sinr_list, sinr_norm_list, speeds)
        print_metrics("BC+Curriculum final @ paper params:", m)
    else:
        print(f"\n[4] SKIP: {bc_final} not found")

    print("\n" + "=" * 70)
    print("Interpretation guide:")
    print("  If Phase 0 @ t_ho_prep=1 shows G~99% -> concept works, transition is problem")
    print("  If Phase 0 @ t_ho_prep=1 shows G~50% -> even Phase 0 didn't fully converge")
    print("  If Phase 1 @ t_ho_prep=3 shows G<Phase 0 -> transition causes regression")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    main()
