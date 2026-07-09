"""Evaluate and compare: 3GPP baseline, original PPO, and 2-phase PPO."""

import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
from ho_optim_drl.gym_env import HandoverEnvPPO
from ho_optim_drl.gym_env.ho_env_ppo import test_ppo_model
from ho_optim_drl.gym_env.ho_env_3gpp import HandoverEnv3GPP
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


def eval_ppo(label, model_path, config, rsrp_list, sinr_list, sinr_norm_list, speeds):
    """Evaluate a PPO model and return per-speed metrics."""
    print(f"\n{'='*50}")
    print(f"Evaluating: {label}")
    print(f"{'='*50}")

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

    return _compute_metrics(result, config, speeds)


def _compute_metrics(result, config, speeds):
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


def print_comparison(results_dict):
    speeds = results_dict[next(iter(results_dict))]["speed"]
    print("\n" + "=" * 80)
    print(f"{'COMPARISON RESULTS':^80}")
    print("=" * 80)

    header = f"{'Speed (km/h)':<14}" + "".join(f"{k:^22}" for k in results_dict)
    print(header)
    print("-" * 80)

    for i, spd in enumerate(speeds):
        row = f"{spd:<14}"
        for metrics in results_dict.values():
            r = metrics["r_rel"][i] * 100
            pp = metrics["pp_rate"][i] * 100
            rlf = metrics["rlf_rate"][i] * 100
            row += f"  G={r:.3f}% PP={pp:.1f}% RLF={rlf:.2f}%"
        print(row)

    print("=" * 80)
    print("G = relative average rate (higher is better)")
    print("PP = ping-pong rate (lower is better)")
    print("RLF = radio link failure rate (lower is better)")


def main():
    config = Config()
    data_dir = os.path.join(ROOT_PATH, "data", "processed")
    rsrp_list, sinr_list, sinr_norm_list, speeds = load_datasets(config, data_dir, USE_SPEEDS)

    results = {}

    # Ensure eval config always uses full paper params (no early termination)
    config.t_ho_prep = 5
    config.t_ho_exec = 4
    config.terminate_on_pp = False
    config.terminate_on_rlf = False

    # ── All models to evaluate: label -> path ─────────────────────────────────
    model_specs = {
        "PPO (original)":   os.path.join(ROOT_PATH, "results", "models", "ppo_model",      "model"),
        "PPO (2-phase)":    os.path.join(ROOT_PATH, "results", "models", "ppo_2phase",      "model"),
        "PPO (curriculum)": os.path.join(ROOT_PATH, "results", "models", "ppo_curriculum",  "model"),
        "PPO (shaped)":     os.path.join(ROOT_PATH, "results", "models", "ppo_shaped",      "model"),
        "PPO (imitation)":  os.path.join(ROOT_PATH, "results", "models", "ppo_imitation",   "model"),
        "PPO (multiseed)":  os.path.join(ROOT_PATH, "results", "models", "ppo_multiseed",   "model"),
    }

    for label, model_path in model_specs.items():
        if os.path.exists(model_path + ".zip"):
            results[label] = eval_ppo(
                label, model_path,
                config, rsrp_list, sinr_list, sinr_norm_list, speeds
            )
        else:
            print(f"[SKIP] {label}: {model_path}.zip not found")

    print("\n[INFO] For 3GPP baseline: uv run python run.py validate_3gpp")

    # ── Print comparison ──────────────────────────────────────────────────────
    if results:
        print_comparison(results)

        # Save results
        out_dir = os.path.join(ROOT_PATH, "results", "metrics")
        os.makedirs(out_dir, exist_ok=True)
        for name, metrics in results.items():
            fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_eval.csv"
            ut.write_to_csv(os.path.join(out_dir, fname), metrics)
            print(f"Saved: results/metrics/{fname}")

    return 0


if __name__ == "__main__":
    main()
