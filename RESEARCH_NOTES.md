# Research Notes: Tái hiện bài báo DRL Handover (SCC 2025)

> **Paper:** "A Deep Reinforcement Learning-Based Approach for Adaptive Handover Protocols"  
> **Authors:** Johannes Voigt, Peter J. Gu, Peter M. Rost (KIT / TUM)  
> **Repo gốc:** https://github.com/kit-cel/HandoverOptimDRL  
> **Environment:** RTX 3060 Ti · CUDA 12.8 · PyTorch 2.7.0+cu126 · Python 3.12 · uv

---

## 1. Kiến trúc tổng thể

### Bài toán
Thay thế 3GPP Event A3 HO cố định bằng PPO agent học động. Agent đặt tại BS, nhận SINR/RSRP từ UE và quyết định kết nối BS nào mỗi Δt=10ms.

### Mô phỏng
- **Dữ liệu thực**: Vienna 5G Simulator + SUMO mobility, khu vực trung tâm Karlsruhe
- **N = 5 macro BS** (BS0–BS4), 2.1 GHz, 10 MHz bandwidth
- **UE speed**: 3–50 km/h (train), 30/50/70/90 km/h (test)
- **GPX tracks** sampled Δt=10ms, layer-3 filtered (k=16)
- **248 datasets** tổng (62 per speed)

---

## 2. Formulation RL

### State vector — 11 chiều (2N+1, N=5)

```
s_t = [s_BS | s_SINR | s_PP]
       (5)     (5)     (1)
```

| Thành phần | Ý nghĩa | Giá trị |
|---|---|---|
| `s_BS[i]` | One-hot: 1 nếu BS_i đang serve | {0, 1} |
| `s_SINR[i]` | SINR của BS_i, clip [-10,10]dB → scale [0,1] | [0, 1] |
| `s_PP` | = 1 nếu đang trong MTS window (PP monitoring) | {0, 1} |

### Action

```
a_t ∈ {0, 1, 2, 3, 4}   →   index của BS muốn kết nối
```

- Nếu `a_t == pcell` → stay / abort HO prep
- Nếu `a_t ≠ pcell` → trigger HO preparation về phía BS `a_t`
- Nếu đổi target giữa chừng → HO prep reset (vì `permit_ho_prep_abort=True`)

### Reward

```python
r(s_t, a_t) = r_SINR + r_PP + r_RLF
```

| Component | Điều kiện | Giá trị |
|---|---|---|
| `r_SINR` | connected to pcell AND pcell == best BS | `sinr_norm[pcell] + C` |
| `r_SINR` | connected to pcell, pcell ≠ best BS | `sinr_norm[pcell]` |
| `r_SINR` | pcell == None (disconnected) | `0` |
| `r_PP` | ping-pong detected | `-C` |
| `r_RLF` | RLF declared | `-2C` |
| `r_RLF` | SINR < Q_out (out-of-sync) | `-C` |

**C = 0.95** (reward/penalty constant)

> **Bug tiềm ẩn trong code gốc**: `if self.s_pcell[-1] > 0` thay vì `>= 0`  
> → BS0 (index=0) không nhận SINR reward. Tuy nhiên BS indices được shuffle mỗi episode nên bias này trung bình ra.

---

## 3. Cơ chế HO — các timer quan trọng

```
Agent chọn BS_i ≠ pcell
    → HO prep counter bắt đầu (t_ho_prep = 5 steps = 50ms)
    → Nếu agent đổi target trong lúc đang prep → RESET
    → Sau 5 steps chọn cùng BS → HO prep xong → bắt đầu HO exec (4 steps = 40ms)
    → Sau HO exec: kết nối BS mới (nếu SINR > Q_in)
                   hoặc → RLF (nếu SINR < Q_in hoặc T310 đang chạy)
```

| Timer / Counter | Giá trị | Steps (Δt=10ms) |
|---|---|---|
| `t_ho_prep` | 50 ms | **5 steps** |
| `t_ho_exec` | 40 ms | 4 steps |
| `t_mts` (MTS) | 1000 ms | 100 steps |
| `t_rlfr` (RLF recovery) | 200 ms | 20 steps |
| `t_t310` (T310) | 1000 ms | 100 steps |
| `n310` | 10 counts | — |
| `n311` | 3 counts | — |

### Điều kiện RLF
1. N310 consecutive SINR < Q_out (-8 dB) → start T310
2. T310 expires mà không có N311 consecutive SINR > Q_in (-6 dB) → **RLF**
3. T310 đang chạy khi HO prep xong → **HOF → RLF**

### Điều kiện PP
- UE HO sang BS mới rồi quay lại BS cũ trong vòng MTS = 1000ms → **PP**

---

## 4. PPO Architecture & Hyperparameters

```
Actor-Critic MLP [64 → 128 → 64] + ReLU
Input: 11 → hidden → output: 5 (actor) / 1 (critic)
```

| Hyperparameter | Giá trị |
|---|---|
| Algorithm | PPO (Stable-Baselines3 2.6.0) |
| Learning rate | 5×10⁻⁵ (linear schedule → 0) |
| Entropy coef | **0.1** |
| n_steps (rollout) | 2000 |
| batch_size | 200 |
| n_epochs | 10 |
| Total timesteps | 5,000,000 |
| Training device | **CPU** (MLP policy chạy nhanh hơn CPU vs GPU) |
| Training speeds | 30, 50 km/h |
| Test speeds | 30, 50, 70, 90 km/h |

> **Quan trọng**: SB3 PPO với MLP policy chạy **nhanh hơn trên CPU**. GPU chỉ có lợi cho CNN. Trên setup này: CPU ~1100 it/s, GPU ~700 it/s.

---

## 5. Two-Phase Training (theo paper)

### Lý do 2-phase

Paper mô tả: với random policy (uniform 1/5 mỗi BS), xác suất để HO xảy ra thấp. Phase 1 không có PP termination để agent tự do học HO strategy. Phase 2 thêm PP penalty để ổn định.

| | Phase 1 | Phase 2 |
|---|---|---|
| `terminate_on_rlf` | True | True |
| `terminate_on_pp` | **False** | **True** |
| `t_ho_prep` | 5 (giữ nguyên) | 5 (giữ nguyên) |
| Steps | 2.5M | 2.5M |

### Transition Phase 1 → Phase 2

```python
# Sau phase 1:
config.terminate_on_pp = True
env2 = HandoverEnvPPO(config, ...)
model.set_env(env2)
model.lr_schedule = lambda _: config.lr * 0.2   # constant 1e-5 cho phase 2
model.learn(total_timesteps=2_500_000, reset_num_timesteps=True)
```

**Lý do dùng `reset_num_timesteps=True`**: Khi False, SB3 tính `_total_timesteps = phase2_steps + num_timesteps_from_phase1`. Với `linear_schedule`, `progress_remaining` có thể = 0 ngay từ đầu phase 2 → lr = 0 → model không học được.

---

## 6. Bài học từ thực nghiệm

### 6.1 Vấn đề cốt lõi: Gradient signal quá yếu (Credit Assignment Problem)

Đây là nguyên nhân gốc rễ khiến TẤT CẢ các lần training thất bại.

**Tại sao policy bị stuck ở uniform distribution?**

Với uniform random policy (mỗi BS được chọn xác suất 1/5):
- P(hoàn thành HO trong 1 lần) = P(5 bước liên tiếp cùng BS) = (1/5)^4 = **0.16%**
- Gần như **không có HO nào hoàn thành** trong suốt quá trình training ban đầu

Hậu quả:
- UE **luôn ở trên cùng BS** (pcell không đổi)
- Reward mỗi step = `sinr_norm[pcell]` ≈ 0.5 → **KHÔNG phụ thuộc vào action**
- Advantage A(s, a) ≈ 0 cho mọi (s, a) → **gradient gần như bằng 0**
- Policy **không thể thoát khỏi** uniform distribution dù training bao lâu

**Xác nhận thực nghiệm** (300K steps, 3 giá trị ent_coef):

```
ent_coef=0.1:   H=1.6094  (max=1.6094) → 100.0% maximum entropy
ent_coef=0.01:  H=1.6093               →  99.9% maximum entropy
ent_coef=0.001: H=1.6084               →  99.9% maximum entropy
```

→ Giảm entropy coefficient KHÔNG giúp được gì. Vấn đề là gradient từ policy objective, không phải từ entropy regularization.

**Tại sao pre-trained model của paper lại hoạt động?**

Model gốc cho kết quả hoàn toàn deterministic:
```
BS3 best SINR → action=3, probs=[0., 0., 0., 1., 0.]  (H=0)
BS1 best SINR → action=1, probs=[0., 1., 0., 0., 0.]  (H=0)
```

Khả năng lý giải:
1. **Random seed may mắn**: Initialization tình cờ tạo ra slight bias về một BS → positive feedback loop → hội tụ
2. **Nhiều steps hơn**: Paper có thể đã train lâu hơn 5M steps (không tái hiện được)
3. **SINR rewards vẫn có gradient**: Khi UE ở cell edge, SINR thấp → RLF (-2C) → agent học cách tránh bằng cách liên tục pick cùng BS → HO thành công

**Ước tính gradient signal thực tế** (over 5M steps):

| Metric | Tính toán | Giá trị |
|---|---|---|
| Episodes (~ep_len=291) | 5M / 291 | ~17,180 |
| HOs/episode (uniform) | 291 * 0.0016 / 9 | ~0.05 |
| Tổng HOs trong 5M steps | 17,180 * 0.05 | ~860 |
| HOs / rollout buffer | 860 / 2500 | ~0.34 |

→ Chỉ ~860 HO completions trong toàn bộ 5M steps để học từ → tín hiệu quá thưa.

---

### 6.2 Thử nghiệm 1 (THẤT BẠI): t_ho_prep=3 trong Phase 1

**Ý tưởng**: Tăng tần suất HO bằng cách giảm `t_ho_prep` 5→3:
- P(HO | t_ho_prep=5) = (1/5)^4 = **0.16%**
- P(HO | t_ho_prep=3) = (1/5)^2 = **4.0%** → tăng 25×

**Kết quả thực tế**: **Thảm họa**

```
Γ_R: 14–34%   (vs expected ~99.8%)
RLF rate: 57–142%   (> 100% nghĩa là nhiều RLF hơn cả HO!)
Action probs: [0.20, 0.20, 0.20, 0.20, 0.20]  ← UNIFORM
```

**Nguyên nhân**: Với t_ho_prep=3, HO xảy ra nhiều hơn nhưng đến BS ngẫu nhiên (không phải best BS) → SINR thấp → HOF → RLF → episode terminate sớm với -2C. Reward quá noisy, gradient không đủ để thoát uniform.

---

### 6.3 Thử nghiệm 2 (THẤT BẠI): 2-phase đúng paper, t_ho_prep=5 xuyên suốt

Giữ nguyên t_ho_prep=5. Chỉ thay `terminate_on_pp` giữa 2 phase.

```
Phase 1: 2.5M steps, terminate_on_rlf=True, terminate_on_pp=False
Phase 2: 2.5M steps, terminate_on_rlf=True, terminate_on_pp=True
         lr_schedule = constant 1e-5, reset_num_timesteps=True
```

**Kết quả sau 5M steps**:

```
entropy_loss = -1.61 xuyên suốt TOÀN BỘ training (cả 2 phase)
Action probs: [0.199, 0.199, 0.202, 0.203, 0.197]  ← UNIFORM
```

**Nguyên nhân**: ep_len_mean = 291 dù không có PP termination → RLF vẫn là nguyên nhân chính terminate episodes ngắn → gradient signal vẫn quá yếu. Phase 1 (no PP) không giúp được vì RLF vẫn truncate episodes ở ~291 steps thay vì để episodes dài hơn.

---

### 6.4 Vấn đề lr_schedule với reset_num_timesteps (Critical Bug)

**Bẫy khi kết hợp linear_schedule + reset_num_timesteps=False** trong PPO SB3:

```python
# SAI - Phase 2 có lr=0 ngay từ đầu!
model.learn(total_timesteps=2_500_000, reset_num_timesteps=False)
# Lúc này: progress_remaining = 1 - num_timesteps/_total_timesteps
# num_timesteps = 2.5M (từ phase 1)
# _total_timesteps = 2.5M (phase 2)
# → progress_remaining = 1 - 2.5M/2.5M = 0 → lr = 0
```

```python
# ĐÚNG - dùng constant lr hoặc reset counter:
model.lr_schedule = lambda _: config.lr * 0.2   # constant 1e-5
model.learn(total_timesteps=2_500_000, reset_num_timesteps=True)
```

---

### 6.5 So sánh policy: Original vs 2-phase (ours)

| Model | BS3 best obs | BS1 best obs | BS0 & stay |
|---|---|---|---|
| **Original (paper)** | `[0, 0, 0, 1, 0]` H=0 | `[0, 1, 0, 0, 0]` H=0 | `[1, 0, 0, 0, 0]` H=0 |
| **2-phase v1 (sai)** | `[0.20, 0.20, 0.20, 0.20, 0.20]` H=1.61 | uniform | uniform |
| **2-phase v2 (fix)** | `[0.199, 0.199, 0.202, 0.203, 0.197]` H=1.61 | uniform | uniform |

→ Cả 2 lần thử đều cho uniform policy. Pre-trained model của paper không thể tái tạo từ scratch với 5M steps trong setup này.

---

### 6.6 Hướng giải quyết (chưa thử)

**Phương án 1: Curriculum t_ho_prep = 1 → 5**
```
Phase 0: t_ho_prep=1, t_ho_exec=1, terminate_on_rlf=False, terminate_on_pp=False
         → Mỗi action ngay lập tức thay đổi serving BS
         → Reward trực tiếp phụ thuộc action → học nhanh
         → 1M steps
Phase 1: t_ho_prep=3, terminate_on_rlf=True, terminate_on_pp=False → 2M steps
Phase 2: t_ho_prep=5, terminate_on_rlf=True, terminate_on_pp=True → 2M steps
```
Lý do: Phase 0 biến bài toán HO thành "multi-armed bandit" đơn giản, policy hội tụ nhanh.

**Phương án 2: Reward shaping**
```python
# Thêm immediate reward cho action chọn best BS (dù HO chưa xong):
r_action = sinr_norm[action] * 0.1  # nhỏ để không overpower SINR reward
```
Cung cấp gradient signal ngay cả khi HO chưa hoàn thành.

**Phương án 3: Imitation learning pre-training**
```
Pre-train actor network với behavior cloning từ 3GPP A3 algorithm
→ Policy bắt đầu từ gần-optimal thay vì random
→ Fine-tune với PPO
```

**Phương án 4: Nhiều random seeds**
```
Train 5 runs với seeds khác nhau, chọn run tốt nhất
Xác suất ít nhất 1 run hội tụ: cao hơn nếu đây là vấn đề seed sensitivity
```

---

## 7. Kết quả đã có

### 3GPP Baseline (124 datasets, 4 speeds)

| Speed | Γ_R (%) | PP rate | RLF rate |
|:---:|:---:|:---:|:---:|
| 30 km/h | 99.756 | 41.83% | 3.37% |
| 50 km/h | 99.751 | 50.00% | 2.05% |
| 70 km/h | 99.797 | 47.26% | 2.53% |
| 90 km/h | 99.696 | 44.41% | 6.99% |

**Nhận xét**: PP rate 42–50% — cứ 2 lần HO thì 1 lần ping-pong. 3GPP không thích nghi được với mobility.

### Original PPO (pre-trained model của paper, 124 datasets)

| Speed | Γ_R (%) | PP rate | RLF rate |
|:---:|:---:|:---:|:---:|
| 30 km/h | 99.795 | 43.58% | 1.17% |
| 50 km/h | 99.823 | 45.38% | 0.70% |
| 70 km/h | 99.839 | 46.45% | 2.84% |
| 90 km/h | 99.763 | 47.53% | 4.63% |

**So với paper (Fig.4)**: Sai lệch < 0.02% → **Tái hiện thành công**.

**Policy của original model**: Hoàn toàn deterministic

```
Obs: BS3 has best SINR → action=3, probs=[0., 0., 0., 1., 0.]
Obs: BS1 has best SINR → action=1, probs=[0., 1., 0., 0., 0.]
```

**Nhận xét**: RLF rate giảm rõ rệt so với 3GPP (nhất là 50km/h: 2.05% → 0.70%). PP rate vẫn cao (43–47%) — hạn chế của original training.

### So sánh với Paper

| Method | Γ_R TB (%) | Match |
|---|---|---|
| Paper 3GPP | ~99.75 | ✅ |
| Tái hiện 3GPP | 99.750 | ✅ |
| Paper PPO | ~99.82 | ✅ |
| Tái hiện PPO gốc | 99.805 | ✅ |

---

## 8. Cấu trúc code quan trọng

```
src/ho_optim_drl/
├── config.py              ← TẤT CẢ hyperparameter
├── gym_env/
│   ├── ho_env_ppo.py      ← Gym wrapper, reward function
│   └── ho_protocol_ppo.py ← State machine: HO/RLF/PP logic (quan trọng nhất)
scripts/
├── train_ppo.py           ← Training gốc (SAVE_MODEL=False, cần đổi thành True)
├── train_ppo_2phase.py    ← 2-phase training (custom)
├── validate_ppo.py        ← Test original model
└── evaluate_all.py        ← Compare all models
```

### Các hàm cốt lõi trong `ho_protocol_ppo.py`

| Hàm | Vai trò |
|---|---|
| `HOProcedurePPO.step()` | Entry point mỗi timestep |
| `_ho_preparation_handler()` | Trigger/reset HO prep dựa trên target cell |
| `_ho_state_machine()` | Chuyển prep→exec→connect |
| `_rlf_detection()` | N310/N311/T310 logic |
| `_pp_monitoring()` | MTS window monitoring |
| `_rlf_recovery()` | Reconnect sau RLF |

---

## 9. Lệnh chạy

```bash
# Setup (chỉ 1 lần)
uv venv --python 3.12
uv sync

# Training
uv run python run.py train_ppo           # original (5M steps, ~55 phút)
uv run python run.py train_ppo_2phase    # 2-phase (5M steps, ~55 phút)

# Evaluation
uv run python run.py validate_3gpp       # 3GPP baseline
uv run python run.py validate_ppo        # original PPO
uv run python run.py evaluate_all        # so sánh tất cả

# Plot
uv run python run.py plot_results
```

---

## 10. Câu hỏi mở / việc cần làm

### Đã làm ✅
- [x] Tái hiện 3GPP baseline: khớp paper < 0.001%
- [x] Tái hiện PPO original: khớp paper < 0.02%
- [x] Phân tích root cause: gradient signal quá yếu khi t_ho_prep=5
- [x] Xác nhận: giảm ent_coef không giúp được (cả 3 giá trị đều cho H=1.61)
- [x] Fix lr_schedule bug cho multi-phase training

### Cần thử tiếp
- [ ] **Curriculum t_ho_prep=1→5**: Thử phase 0 với t_ho_prep=1 để có immediate reward
- [ ] **Multiple seeds**: Train 3-5 lần với seeds khác để xem có seed nào hội tụ không
- [ ] **Reward shaping**: Thêm small immediate reward cho action chọn best SINR BS
- [ ] **PP rate vẫn cao (43–47%)**: Cả 3GPP lẫn PPO gốc đều có PP cao → tìm hiểu
- [ ] **Visualization**: Vẽ SINR + BS selection theo thời gian để visualize HO behavior
- [ ] **Higher speeds (70/90 km/h)**: Train trực tiếp để xem PPO adapt được không

### Kết luận về khả năng tái tạo
Pre-trained model của paper **có thể evaluate được** (✅ tái hiện kết quả Fig. 4).
Pre-trained model **không thể train lại** từ scratch trong 5M steps với setup này, do:
1. Credit assignment problem: SINR reward không phụ thuộc action khi HO ít xảy ra
2. Có thể paper đã dùng seed cụ thể hoặc train nhiều hơn 5M steps
3. Khả năng cao: paper chỉ train 1 lần may mắn và không kiểm tra reproducibility
