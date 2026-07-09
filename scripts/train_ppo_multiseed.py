"""
Multiple random seeds to probe whether convergence is seed-sensitive.

Hypothesis (from RESEARCH_NOTES.md sec 6.1):
  The original paper may have used a lucky seed where random initialization
  created slight bias toward one BS -> positive feedback -> convergence.
  With 5 seeds, probability that at least 1 converges is much higher.

Protocol:
  - 5 seeds: [0, 42, 123, 777, 1234]
  - Each: standard 2-phase training (same as train_ppo_2phase.py)
  - Quick entropy probe after 500K steps to detect early convergence
  - Best model (lowest final entropy) is saved as the main model
  - All checkpoints saved individually for analysis

Convergence indicator:
  entropy < 1.0 after 500K steps -> policy is specializing (promising)
  entropy ≈ 1.61 after 500K steps -> uniform policy (stuck), continue anyway
"""

import os
import random
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
MODEL_NAME = "ppo_multiseed"

SEEDS = [0, 42, 123, 777, 1234]
PROBE_STEPS = 300_000   # quick entropy check after this many steps
TOTAL_STEPS = 2_000_000  # 2M/seed × 5 seeds = 10M total (~110 min)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def make_model(env, config, seed):
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
        seed=seed,
        device=torch.device("cpu"),
    )


def get_policy_entropy(model, n_bs=5):
    """Estimate policy entropy by sampling a few observations."""
    entropies = []
    for best_bs in range(n_bs):
        s_bs = np.zeros(n_bs, dtype=np.float32)
        s_bs[best_bs] = 1.0
        sinr_norm = np.zeros(n_bs, dtype=np.float32)
        sinr_norm[best_bs] = 0.8
        obs = np.concatenate([s_bs, sinr_norm, [0.0]])
        dist = model.policy.get_distribution(
            torch.FloatTensor(obs).unsqueeze(0)
        )
        probs = dist.distribution.probs.detach().numpy()[0]
        probs = np.clip(probs, 1e-10, 1.0)
        h = -np.sum(probs * np.log(probs))
        entropies.append(h)
    return float(np.mean(entropies))


def train_one_seed(seed, config, rsrp_list, sinr_list, sinr_norm_list, model_dir):
    print(f"\n{'='*60}")
    print(f"SEED {seed}: Starting 2-phase training")
    print("=" * 60)

    set_all_seeds(seed)

    # Phase 1
    config.terminate_on_pp = False
    config.terminate_on_rlf = True
    config.t_ho_prep = 5
    config.t_ho_exec = 4

    HandoverEnvPPO.reset_cls()
    env1 = HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)

    model = make_model(env1, config, seed=seed)

    # Probe: train 500K steps first, check entropy
    model.learn(total_timesteps=PROBE_STEPS, progress_bar=True, tb_log_name=f"seed{seed}_probe")
    probe_entropy = get_policy_entropy(model)
    print(f"  Seed {seed}: entropy after {PROBE_STEPS//1000}K steps = {probe_entropy:.4f}")
    print(f"  {'PROMISING (entropy < 1.0)' if probe_entropy < 1.0 else 'Stuck (entropy ~1.61)'}")

    # Continue Phase 1 to completion
    phase1_total = TOTAL_STEPS // 2  # 1M
    remaining_p1 = phase1_total - PROBE_STEPS
    if remaining_p1 > 0:
        model.learn(
            total_timesteps=remaining_p1,
            progress_bar=True,
            tb_log_name=f"seed{seed}_phase1",
            reset_num_timesteps=False,
        )

    # Phase 2
    config.terminate_on_pp = True
    HandoverEnvPPO.reset_cls()
    env2 = HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)
    model.set_env(env2)
    model.lr_schedule = lambda _: config.lr * 0.2  # constant 1e-5

    model.learn(
        total_timesteps=TOTAL_STEPS // 2,  # 1M
        progress_bar=True,
        tb_log_name=f"seed{seed}_phase2",
        reset_num_timesteps=True,
    )

    final_entropy = get_policy_entropy(model)
    print(f"  Seed {seed}: final entropy = {final_entropy:.4f}")

    # Save individual seed model
    seed_path = os.path.join(model_dir, f"model_seed{seed}")
    model.save(seed_path)
    print(f"  Saved: {seed_path}.zip")

    return model, final_entropy


def main():
    config = Config()
    data_dir = os.path.join(ROOT_PATH, "data", "processed")
    rsrp_list, sinr_list, sinr_norm_list = load_datasets(config, data_dir, [30, 50])

    model_dir = os.path.join(ROOT_PATH, "results", "models", MODEL_NAME)
    os.makedirs(model_dir, exist_ok=True)

    best_model = None
    best_entropy = float("inf")
    best_seed = None

    results = {}

    for seed in SEEDS:
        model, entropy = train_one_seed(
            seed, config, rsrp_list, sinr_list, sinr_norm_list, model_dir
        )
        results[seed] = entropy

        if entropy < best_entropy:
            best_entropy = entropy
            best_model = model
            best_seed = seed

    # Summary
    print("\n" + "=" * 60)
    print("MULTI-SEED SUMMARY")
    print("=" * 60)
    for seed, h in results.items():
        marker = " <- BEST" if seed == best_seed else ""
        print(f"  Seed {seed:5d}: final entropy = {h:.4f}{marker}")

    print(f"\nBest seed: {best_seed} (entropy={best_entropy:.4f})")
    print(f"Max entropy (uniform): 1.6094")

    if best_model is not None:
        best_path = os.path.join(model_dir, "model")
        best_model.save(best_path)
        print(f"Best model saved: {best_path}.zip")

    return 0


if __name__ == "__main__":
    main()
