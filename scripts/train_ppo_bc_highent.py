"""
BC + Gradual Curriculum + ent_coef=0.1 (paper's entropy coefficient).

Fixes the main bug in train_ppo_bc_curriculum.py which used ent_coef=0.01.
Paper uses ent_coef=0.1 — 10x higher, forcing exploration throughout training.

Speed: SubprocVecEnv with N_ENVS parallel environments (~N_ENVS x faster).
"""

import functools
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
from ho_optim_drl.gym_env import HandoverEnvPPO
import ho_optim_drl.utils as ut

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "ppo_bc_highent"

BC_EPOCHS = 30
BC_BATCH_SIZE = 512
BC_LR = 3e-4
PPO_ENT_COEF = 0.1
N_ENVS = 4

PHASES = [
    # (t_ho_prep, t_ho_exec, terminate_rlf, terminate_pp, lr, label)
    (1, 1, False, False, 3e-5,  "prep=1 exec=1 no-RLF no-PP"),
    (2, 1, True,  False, 2e-5,  "prep=2 exec=1 RLF no-PP"),
    (3, 2, True,  False, 1.5e-5,"prep=3 exec=2 RLF no-PP"),
    (4, 3, True,  True,  1e-5,  "prep=4 exec=3 RLF PP"),
    (5, 4, True,  True,  5e-6,  "prep=5 exec=4 RLF PP (paper params)"),
]
STEPS_PER_PHASE = 1_000_000


# Module-level so SubprocVecEnv can pickle it
def _make_env(t_ho_prep, t_ho_exec, terminate_rlf, terminate_pp,
              rsrp_list, sinr_list, sinr_norm_list):
    cfg = Config()
    cfg.t_ho_prep = t_ho_prep
    cfg.t_ho_exec = t_ho_exec
    cfg.terminate_on_rlf = terminate_rlf
    cfg.terminate_on_pp = terminate_pp
    return HandoverEnvPPO(cfg, rsrp_list, sinr_list, sinr_norm_list)


def make_vec_env(config, rsrp_list, sinr_list, sinr_norm_list, n_envs):
    HandoverEnvPPO.reset_cls()
    fns = [
        functools.partial(_make_env, config.t_ho_prep, config.t_ho_exec,
                          config.terminate_on_rlf, config.terminate_on_pp,
                          rsrp_list, sinr_list, sinr_norm_list)
        for _ in range(n_envs)
    ]
    try:
        vec = SubprocVecEnv(fns)
        print(f"  SubprocVecEnv: {n_envs} parallel envs")
        return vec
    except Exception as e:
        print(f"  SubprocVecEnv failed ({e}), using DummyVecEnv")
        return DummyVecEnv(fns)


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


def behavioral_cloning(model, obs_arr, action_arr, device):
    print(f"\n{'='*60}")
    print(f"BEHAVIORAL CLONING: {BC_EPOCHS} epochs, lr={BC_LR}")
    print(f"  {len(obs_arr):,} samples, device={device}")
    print("=" * 60)

    obs_t = torch.FloatTensor(obs_arr).to(device)
    act_t = torch.LongTensor(action_arr).to(device)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, N_ENVS: {N_ENVS}")

    config = Config()
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

    # Create initial single env for model construction + BC
    t_prep, t_exec, rlf, pp, lr, _ = PHASES[0]
    config.t_ho_prep = t_prep
    config.t_ho_exec = t_exec
    config.terminate_on_rlf = rlf
    config.terminate_on_pp = pp

    HandoverEnvPPO.reset_cls()
    env0 = HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)

    model = PPO(
        "MlpPolicy",
        env0,
        ent_coef=PPO_ENT_COEF,
        learning_rate=PHASES[0][4],
        verbose=1,
        policy_kwargs=policy_kwargs,
        n_steps=config.n_steps_per_update,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        tensorboard_log=tb_log,
        device=device,
    )

    obs_arr, action_arr = build_bc_dataset(sinr_norm_list)
    behavioral_cloning(model, obs_arr, action_arr, device)

    vec_env = None
    for phase_idx, (t_prep, t_exec, rlf, pp, lr, label) in enumerate(PHASES):
        print(f"\n{'='*60}")
        print(f"PHASE {phase_idx}: {label}")
        print(f"  t_ho_prep={t_prep}, t_ho_exec={t_exec}, RLF={rlf}, PP={pp}, lr={lr:.0e}, ent={PPO_ENT_COEF}")
        print("=" * 60)

        config.t_ho_prep = t_prep
        config.t_ho_exec = t_exec
        config.terminate_on_rlf = rlf
        config.terminate_on_pp = pp

        if vec_env is not None:
            vec_env.close()
        vec_env = make_vec_env(config, rsrp_list, sinr_list, sinr_norm_list, N_ENVS)
        model.set_env(vec_env)
        model.lr_schedule = lambda _, _lr=lr: _lr

        model.learn(
            total_timesteps=STEPS_PER_PHASE,
            progress_bar=True,
            tb_log_name=f"phase{phase_idx}",
            reset_num_timesteps=True,
        )

        ckpt_path = os.path.join(model_dir, f"model_phase{phase_idx}")
        model.save(ckpt_path)
        print(f"Phase {phase_idx} done. Checkpoint: model_phase{phase_idx}.zip")

    if vec_env is not None:
        vec_env.close()

    final_path = os.path.join(model_dir, "model")
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}.zip")
    return 0


if __name__ == "__main__":
    main()
