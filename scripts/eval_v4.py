"""Evaluate all V4 models: bc_paper, bc_highent, bc_noterm, bc_2m."""

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
    rsrp_files, sinr_files, speeds = ut.filenames_speed_filter(rsrp_files, sinr_files, speed_list)
    rsrp_list, sinr_list, sinr_norm_list = [], [], []
    for rf, sf in zip(rsrp_files, sinr_files):
        rsrp_db, sinr_db = dl.load_preprocess_dataset(config, data_dir, rf, sf)
        sinr_norm = ut.clipnorm(sinr_db, config.sinr_lower_clip, config.sinr_upper_clip)
        rsrp_list.append(rsrp_db)
        sinr_list.append(sinr_db)
        sinr_norm_list.append(sinr_norm)
    return rsrp_list, sinr_list, sinr_norm_list, speeds


def eval_model(label, model_path, config, rsrp_list, sinr_list, sinr_norm_list, speeds):
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
        status = "OK" if g > 90 else ("~" if g > 50 else "FAIL")
        print(f"    {spd} km/h: G={g:.1f}%  PP={pp:.1f}%  RLF={rlf:.1f}%  [{status}]")


def main():
    data_dir = os.path.join(ROOT_PATH, "data", "processed")

    # Paper params for eval (no termination in test mode)
    config = Config()
    config.t_ho_prep = 5
    config.t_ho_exec = 4
    config.terminate_on_rlf = False
    config.terminate_on_pp = False

    rsrp_list, sinr_list, sinr_norm_list, speeds = load_datasets(config, data_dir, USE_SPEEDS)

    print("=" * 70)
    print("V4 EVAL: all models evaluated with paper params (t_ho_prep=5, exec=4)")
    print("=" * 70)

    models = [
        ("bc_paper",   "BC + paper params directly"),
        ("bc_highent", "BC + curriculum + ent_coef=0.1"),
        ("bc_noterm",  "BC + curriculum + no termination"),
        ("bc_2m",      "BC + curriculum + 2M steps/phase"),
    ]

    for model_name, description in models:
        model_path = os.path.join(ROOT_PATH, "results", "models", f"ppo_{model_name}", "model.zip")
        if not os.path.exists(model_path):
            print(f"\n[{model_name}] SKIP: model not found at {model_path}")
            continue
        print(f"\n[{model_name}] {description}")
        try:
            m = eval_model(description, model_path, config,
                           rsrp_list, sinr_list, sinr_norm_list, speeds)
            print_metrics(description, m)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 70)
    print("Reference: paper model G~99.8%, PP<1%, RLF<1%")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    main()
