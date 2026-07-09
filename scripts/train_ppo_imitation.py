"""
Imitation Learning (Behavioral Cloning) + PPO fine-tuning.

Stage 1 - Behavioral Cloning:
  Oracle policy: always choose argmax(sinr_norm) = BS with best signal.
  Synthetic dataset: for each training timestep t:
    obs = [one_hot(best_bs), sinr_norm[t,:], 0.0]  (pp_flag=0)
    oracle_action = argmax(sinr_norm[t,:])
  Train actor (pi branch) with CrossEntropy loss.
  Result: policy learns "pick best SINR BS" deterministically.

Stage 2 - PPO fine-tuning (2-phase):
  Starting from a near-deterministic best-BS policy:
  - Phase 1 (2.5M): terminate_on_rlf=True, terminate_on_pp=False
    Agent has rich HO experience from day 1 -> learns timing
  - Phase 2 (2.5M): add PP termination -> learns PP avoidance

Key differences from baseline:
  - Policy starts from informed (not random) initialization
  - First rollout already contains completed HOs -> real gradient signal
  - ent_coef reduced to 0.01 to preserve learned structure

Why it works:
  Random policy: 0.16% HO completion rate -> credit assignment fails
  BC policy: ~80% HO completion rate from step 1 -> gradient flows immediately
"""

import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
from ho_optim_drl.gym_env import HandoverEnvPPO
import ho_optim_drl.utils as ut

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "ppo_imitation"

BC_EPOCHS = 20          # Behavioral cloning training epochs
BC_BATCH_SIZE = 512     # BC mini-batch size
BC_LR = 3e-4            # BC learning rate
PPO_ENT_COEF = 0.01    # Reduced entropy (preserve BC-learned policy structure)


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


def build_bc_dataset(sinr_norm_list):
    """Build (obs, oracle_action) dataset from all training data.

    Oracle: always choose BS with highest SINR.
    Observation: [one_hot(best_bs), sinr_norm, pp=0]
    This is a synthetic dataset; doesn't run through the env step.
    """
    obs_list = []
    action_list = []

    n_bs = sinr_norm_list[0].shape[1]

    for sinr_norm in sinr_norm_list:
        T = sinr_norm.shape[0]
        for t in range(T):
            sn = sinr_norm[t, :]
            best_bs = int(np.argmax(sn))

            # obs = [s_BS (one-hot), s_SINR, s_PP=0]
            s_bs = np.zeros(n_bs, dtype=np.float32)
            s_bs[best_bs] = 1.0
            obs = np.concatenate([s_bs, sn.astype(np.float32), [0.0]])

            obs_list.append(obs)
            action_list.append(best_bs)

    obs_arr = np.stack(obs_list, axis=0)         # (N, 11)
    action_arr = np.array(action_list, dtype=np.int64)  # (N,)
    print(f"BC dataset: {len(obs_arr):,} samples from {len(sinr_norm_list)} datasets")
    return obs_arr, action_arr


def behavioral_cloning(model, obs_arr, action_arr, device="cpu"):
    """Pre-train actor network with behavioral cloning (CrossEntropy)."""
    print(f"\n{'='*60}")
    print(f"STAGE 1: Behavioral Cloning ({BC_EPOCHS} epochs, lr={BC_LR})")
    print(f"  Dataset: {len(obs_arr):,} samples")
    print(f"  Oracle: argmax(sinr_norm) -> should reach accuracy ~100%")
    print("=" * 60)

    obs_tensor = torch.FloatTensor(obs_arr).to(device)
    act_tensor = torch.LongTensor(action_arr).to(device)

    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=BC_BATCH_SIZE, shuffle=True)

    # Only train actor (pi branch) — value network stays at random init
    bc_params = (
        list(model.policy.mlp_extractor.policy_net.parameters())
        + list(model.policy.action_net.parameters())
    )
    optimizer = torch.optim.Adam(bc_params, lr=BC_LR)

    model.policy.train()

    for epoch in range(BC_EPOCHS):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for obs_batch, act_batch in loader:
            # Forward pass through actor branch
            features = model.policy.extract_features(
                obs_batch, model.policy.features_extractor
            )
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            logits = model.policy.action_net(latent_pi)

            loss = F.cross_entropy(logits, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(obs_batch)
            preds = logits.argmax(dim=1)
            total_correct += (preds == act_batch).sum().item()
            total_samples += len(obs_batch)

        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples
        print(f"  Epoch {epoch+1:2d}/{BC_EPOCHS}: loss={avg_loss:.4f}, acc={accuracy:.1f}%")

    model.policy.eval()
    print("Behavioral cloning complete.")


def make_model(env, config, ent_coef, tensorboard_log=None):
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=config.net_arch, vf=config.net_arch),
    )
    return PPO(
        "MlpPolicy",
        env,
        ent_coef=ent_coef,
        learning_rate=config.lr,
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

    # ── Stage 1: Behavioral Cloning ───────────────────────────────────────────
    # Create initial PPO model (with Phase 1 config for BC)
    config.terminate_on_pp = False
    config.terminate_on_rlf = True

    HandoverEnvPPO.reset_cls()
    env_bc = HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)
    model = make_model(env_bc, config, ent_coef=PPO_ENT_COEF, tensorboard_log=tb_log)

    # Build oracle dataset and pre-train
    obs_arr, action_arr = build_bc_dataset(sinr_norm_list)
    behavioral_cloning(model, obs_arr, action_arr, device="cpu")

    # Check policy behavior after BC
    print("\nPolicy check after BC:")
    for best_bs in range(5):
        s_bs = np.zeros(5, dtype=np.float32)
        s_bs[best_bs] = 1.0
        sinr_norm_fake = np.zeros(5, dtype=np.float32)
        sinr_norm_fake[best_bs] = 1.0
        obs = np.concatenate([s_bs, sinr_norm_fake, [0.0]])
        action, _ = model.predict(obs, deterministic=True)
        dist = model.policy.get_distribution(
            torch.FloatTensor(obs).unsqueeze(0)
        )
        probs = dist.distribution.probs.detach().numpy()[0]
        print(f"  best_BS={best_bs}: action={action}, probs={np.round(probs, 3)}")

    # ── Stage 2: PPO Fine-tuning, Phase 1 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2, PHASE 1: PPO fine-tuning, terminate_on_pp=False")
    print(f"  ent_coef={PPO_ENT_COEF} (low to preserve BC policy structure)")
    print("=" * 60)

    model.set_env(env_bc)
    phase1_steps = config.n_steps_total // 2  # 2.5M
    model.learn(
        total_timesteps=phase1_steps,
        progress_bar=True,
        tb_log_name="ppo_phase1",
        reset_num_timesteps=True,
    )
    print(f"PPO Phase 1 done: {phase1_steps:,} steps")

    # Checkpoint
    model_dir = os.path.join(ROOT_PATH, "results", "models", MODEL_NAME)
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model_phase1"))
    print("Checkpoint: model_phase1.zip")

    # ── Stage 2: PPO Fine-tuning, Phase 2 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2, PHASE 2: PPO fine-tuning, terminate_on_pp=True")
    print("=" * 60)
    config.terminate_on_pp = True

    HandoverEnvPPO.reset_cls()
    env2 = HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)
    model.set_env(env2)
    model.lr_schedule = lambda _: config.lr * 0.2  # constant 1e-5

    phase2_steps = config.n_steps_total - phase1_steps  # 2.5M
    model.learn(
        total_timesteps=phase2_steps,
        progress_bar=True,
        tb_log_name="ppo_phase2",
        reset_num_timesteps=True,
    )
    print(f"PPO Phase 2 done: {phase2_steps:,} steps")

    # ── Save final model ──────────────────────────────────────────────────────
    model_path = os.path.join(model_dir, "model")
    model.save(model_path)
    print(f"\nFinal model saved: {model_path}.zip")

    return 0


if __name__ == "__main__":
    main()
