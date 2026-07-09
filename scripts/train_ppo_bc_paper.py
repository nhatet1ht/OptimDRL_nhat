"""
BC init + paper params directly (no curriculum).

Key hypothesis: BC gives the right initialization, then paper's original
hyperparams (ent_coef=0.1, lr=5e-5, t_ho_prep=5) can learn from there.

Previous bc_curriculum used ent_coef=0.01 which killed exploration (policy
converged to "never HO"). Paper uses ent_coef=0.1 — 10x higher.
"""

import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO

from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
from ho_optim_drl.gym_env import HandoverEnvPPO
import ho_optim_drl.utils as ut

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "ppo_bc_paper"

BC_EPOCHS = 30
BC_BATCH_SIZE = 512
BC_LR = 3e-4


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
    obs_list, action_list = [], []
    n_bs = sinr_norm_list[0].shape[1]
    for sinr_norm in sinr_norm_list:
        T = sinr_norm.shape[0]
        for t in range(T):
            sn = sinr_norm[t, :]
            best_bs = int(np.argmax(sn))
            s_bs = np.zeros(n_bs, dtype=np.float32)
            s_bs[best_bs] = 1.0
            obs = np.concatenate([s_bs, sn.astype(np.float32), [0.0]])
            obs_list.append(obs)
            action_list.append(best_bs)
    return np.stack(obs_list), np.array(action_list, dtype=np.int64)


def behavioral_cloning(model, obs_arr, action_arr):
    print(f"\n{'='*60}")
    print(f"BEHAVIORAL CLONING: {BC_EPOCHS} epochs, lr={BC_LR}")
    print(f"  {len(obs_arr):,} samples, oracle = argmax(sinr_norm)")
    print("=" * 60)

    obs_t = torch.FloatTensor(obs_arr)
    act_t = torch.LongTensor(action_arr)
    loader = DataLoader(TensorDataset(obs_t, act_t), batch_size=BC_BATCH_SIZE, shuffle=True)

    bc_params = (
        list(model.policy.mlp_extractor.policy_net.parameters())
        + list(model.policy.action_net.parameters())
    )
    optimizer = torch.optim.Adam(bc_params, lr=BC_LR)
    model.policy.train()

    for epoch in range(BC_EPOCHS):
        total_loss, total_correct, n = 0.0, 0, 0
        for obs_b, act_b in loader:
            features = model.policy.extract_features(obs_b, model.policy.features_extractor)
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            logits = model.policy.action_net(latent_pi)
            loss = F.cross_entropy(logits, act_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(obs_b)
            total_correct += (logits.argmax(1) == act_b).sum().item()
            n += len(obs_b)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}: loss={total_loss/n:.5f}  acc={100*total_correct/n:.1f}%")

    model.policy.eval()


def main():
    config = Config()
    # Paper params: t_ho_prep=5, t_ho_exec=4, ent_coef=0.1, lr=5e-5
    # (all already defaults in Config)

    data_dir = os.path.join(ROOT_PATH, "data", "processed")
    rsrp_list, sinr_list, sinr_norm_list = load_datasets(config, data_dir, [30, 50])

    sim_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_log = os.path.join(ROOT_PATH, "results", "tensorboard", MODEL_NAME, sim_id)
    model_dir = os.path.join(ROOT_PATH, "results", "models", MODEL_NAME)
    os.makedirs(model_dir, exist_ok=True)

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=config.net_arch, vf=config.net_arch),
    )

    print(f"\n{'='*60}")
    print("BC + PAPER PARAMS (no curriculum)")
    print(f"  t_ho_prep={config.t_ho_prep}, t_ho_exec={config.t_ho_exec}")
    print(f"  ent_coef={config.ent_coef}, lr={config.lr}")
    print(f"  terminate_rlf={config.terminate_on_rlf}, terminate_pp={config.terminate_on_pp}")
    print("=" * 60)

    HandoverEnvPPO.reset_cls()
    env = HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)

    model = PPO(
        "MlpPolicy",
        env,
        ent_coef=config.ent_coef,
        learning_rate=config.lr,
        verbose=1,
        policy_kwargs=policy_kwargs,
        n_steps=config.n_steps_per_update,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        tensorboard_log=tb_log,
        device=torch.device("cpu"),
    )

    obs_arr, action_arr = build_bc_dataset(sinr_norm_list)
    behavioral_cloning(model, obs_arr, action_arr)

    print(f"\nPPO training: {config.n_steps_total:,} steps with paper params")
    model.learn(
        total_timesteps=config.n_steps_total,
        progress_bar=True,
        tb_log_name="ppo",
        reset_num_timesteps=True,
    )

    final_path = os.path.join(model_dir, "model")
    model.save(final_path)
    print(f"\nModel saved: {final_path}.zip")
    return 0


if __name__ == "__main__":
    main()
