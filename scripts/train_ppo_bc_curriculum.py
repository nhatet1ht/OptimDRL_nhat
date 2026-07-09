"""
BC + Gradual Curriculum: the most promising approach after 6 failed attempts.

Key lessons from previous experiments:
  - Standard 2-phase (t_ho_prep=5): stuck at uniform (H=1.61) through 5M steps
  - Curriculum 1->3->5: Phase 0 worked (H dropped to 1.07!) but Phase 2 reverted to uniform
  - Imitation (BC+PPO): BC perfect (100% acc) but PP penalty killed HO behavior
  - Root cause: each time t_ho_prep increases, credit assignment resets

This approach combines what worked:
  1. BC init: policy starts knowing "pick best SINR BS" (100% oracle accuracy)
  2. Gradual t_ho_prep: 1->2->3->4->5 in small steps (no sudden jumps)
  3. Gradual t_ho_exec: 1->1->2->3->4 (short disconnect time early)
  4. Gradual PP introduction: only in last 2 phases
  5. Low ent_coef=0.01: preserve BC-learned structure throughout
  6. Decreasing lr: 3e-5 -> 1e-5 -> 5e-6 per phase

Phase schedule (1M steps each, 5M total):
  Phase 0: prep=1, exec=1, no RLF, no PP   -> instant HO, warm up BC policy
  Phase 1: prep=2, exec=1, RLF, no PP      -> 2 steps commit, short exec
  Phase 2: prep=3, exec=2, RLF, no PP      -> 3 steps commit
  Phase 3: prep=4, exec=3, RLF, PP         -> 4 steps, PP introduced
  Phase 4: prep=5, exec=4, RLF, PP         -> full paper params
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
MODEL_NAME = "ppo_bc_curriculum"

BC_EPOCHS = 30
BC_BATCH_SIZE = 512
BC_LR = 3e-4
PPO_ENT_COEF = 0.01

PHASES = [
    # (t_ho_prep, t_ho_exec, terminate_rlf, terminate_pp, lr, label)
    (1, 1, False, False, 3e-5, "prep=1 exec=1 no-RLF no-PP"),
    (2, 1, True,  False, 2e-5, "prep=2 exec=1 RLF no-PP"),
    (3, 2, True,  False, 1.5e-5, "prep=3 exec=2 RLF no-PP"),
    (4, 3, True,  True,  1e-5, "prep=4 exec=3 RLF PP"),
    (5, 4, True,  True,  5e-6, "prep=5 exec=4 RLF PP (paper params)"),
]
STEPS_PER_PHASE = 1_000_000


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


def check_policy(model, n_bs=5):
    print("\n  Policy check (should be deterministic best-BS):")
    for best_bs in range(n_bs):
        s_bs = np.zeros(n_bs, dtype=np.float32)
        s_bs[best_bs] = 1.0
        sn = np.zeros(n_bs, dtype=np.float32)
        sn[best_bs] = 0.9
        obs = np.concatenate([s_bs, sn, [0.0]])
        action, _ = model.predict(obs, deterministic=True)
        dist = model.policy.get_distribution(torch.FloatTensor(obs).unsqueeze(0))
        probs = dist.distribution.probs.detach().numpy()[0]
        h = -np.sum(np.clip(probs, 1e-10, 1) * np.log(np.clip(probs, 1e-10, 1)))
        print(f"    best_BS={best_bs}: action={action}, H={h:.4f}, probs={np.round(probs,3)}")


def make_env(config, rsrp_list, sinr_list, sinr_norm_list):
    HandoverEnvPPO.reset_cls()
    return HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)


def make_initial_model(env, config, tensorboard_log=None):
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=config.net_arch, vf=config.net_arch),
    )
    return PPO(
        "MlpPolicy",
        env,
        ent_coef=PPO_ENT_COEF,
        learning_rate=PHASES[0][4],
        verbose=1,
        policy_kwargs=policy_kwargs,
        n_steps=config.n_steps_per_update,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        tensorboard_log=tensorboard_log,
        device=torch.device("cpu"),
    )


def main(start_phase: int = 0):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-phase", type=int, default=start_phase,
                        help="Resume from this phase (0=from scratch, 1-4=load prev checkpoint)")
    args, _ = parser.parse_known_args()
    start_phase = args.start_phase

    config = Config()
    data_dir = os.path.join(ROOT_PATH, "data", "processed")
    rsrp_list, sinr_list, sinr_norm_list = load_datasets(config, data_dir, [30, 50])

    sim_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_log = os.path.join(ROOT_PATH, "results", "tensorboard", MODEL_NAME, sim_id)

    model_dir = os.path.join(ROOT_PATH, "results", "models", MODEL_NAME)
    os.makedirs(model_dir, exist_ok=True)

    if start_phase == 0:
        # ── Stage 1: Behavioral Cloning ───────────────────────────────────────
        t_ho_prep, t_ho_exec, terminate_rlf, terminate_pp, lr, _ = PHASES[0]
        config.t_ho_prep = t_ho_prep
        config.t_ho_exec = t_ho_exec
        config.terminate_on_rlf = terminate_rlf
        config.terminate_on_pp = terminate_pp

        env0 = make_env(config, rsrp_list, sinr_list, sinr_norm_list)
        model = make_initial_model(env0, config, tensorboard_log=tb_log)

        obs_arr, action_arr = build_bc_dataset(sinr_norm_list)
        behavioral_cloning(model, obs_arr, action_arr)
        check_policy(model)
    else:
        # ── Resume: load checkpoint from previous phase ────────────────────────
        prev_ckpt = os.path.join(model_dir, f"model_phase{start_phase - 1}.zip")
        print(f"\n{'='*60}")
        print(f"RESUMING from Phase {start_phase} (loading {prev_ckpt})")
        print("=" * 60)
        t_prep, t_exec, rlf, pp, lr, label = PHASES[start_phase]
        config.t_ho_prep = t_prep
        config.t_ho_exec = t_exec
        config.terminate_on_rlf = rlf
        config.terminate_on_pp = pp
        env_resume = make_env(config, rsrp_list, sinr_list, sinr_norm_list)
        model = PPO.load(prev_ckpt, env=env_resume, tensorboard_log=tb_log)

    # ── Stage 2: Gradual Curriculum ───────────────────────────────────────────
    for phase_idx, (t_prep, t_exec, rlf, pp, lr, label) in enumerate(PHASES):
        if phase_idx < start_phase:
            print(f"Phase {phase_idx}: SKIPPED (already done)")
            continue
        print(f"\n{'='*60}")
        print(f"PHASE {phase_idx}: {label}")
        print(f"  t_ho_prep={t_prep}, t_ho_exec={t_exec}, RLF={rlf}, PP={pp}, lr={lr:.0e}")
        print("=" * 60)

        config.t_ho_prep = t_prep
        config.t_ho_exec = t_exec
        config.terminate_on_rlf = rlf
        config.terminate_on_pp = pp

        env = make_env(config, rsrp_list, sinr_list, sinr_norm_list)
        if phase_idx == 0:
            # First phase: use the env model was created with (already set above)
            model.set_env(env)
        else:
            model.set_env(env)

        model.lr_schedule = lambda _, _lr=lr: _lr

        model.learn(
            total_timesteps=STEPS_PER_PHASE,
            progress_bar=True,
            tb_log_name=f"phase{phase_idx}",
            reset_num_timesteps=True,
        )
        print(f"Phase {phase_idx} done: {STEPS_PER_PHASE:,} steps")

        ckpt_path = os.path.join(model_dir, f"model_phase{phase_idx}")
        model.save(ckpt_path)
        print(f"Checkpoint: model_phase{phase_idx}.zip")

        check_policy(model)

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = os.path.join(model_dir, "model")
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}.zip")

    return 0


if __name__ == "__main__":
    main()
