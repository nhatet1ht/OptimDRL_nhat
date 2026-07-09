# Ghi chú tái hiện bài báo: DRL-based Adaptive Handover Protocols (SCC 2025)

**Paper:** "A Deep Reinforcement Learning-Based Approach for Adaptive Handover Protocols"  
**Repo gốc:** https://github.com/kit-cel/HandoverOptimDRL  
**GPU:** RTX 3060 Ti (8GB VRAM) — đủ để train, code tự detect CUDA

---

## 1. Tóm tắt bài toán

Bài báo tối ưu **timing và target BS selection** trong handover 5G NR dùng PPO, thay thế 3GPP event-based HO cố định. Agent được đặt ở BS, nhận thông tin từ UE và ra quyết định HO.

**Mục tiêu:**
- Tối đa hóa **average data rate** (relative rate Γ_R)
- Giảm **Radio Link Failure (RLF)** và **Ping-Pong (PP)**

---

## 2. Môi trường mô phỏng

| Thông số | Giá trị |
|---|---|
| Khu vực | Trung tâm thành phố Karlsruhe (49°00'17.6"N, 8°22'38.3"E – 49°00'40.3"N, 8°23'44.2"E) |
| Số BS (N) | **5** (BS0–BS4), macro BS |
| Tần số (fc) | 2.1 GHz (IMT-2000 n1) |
| Băng thông | 10 MHz |
| Bước thời gian (Δt) | **10 ms** |
| Tốc độ UE | 3–50 km/h (train), thêm 70, 90 km/h (test) |
| Simulator | Vienna 5G System Level Simulator + SUMO |
| Fading model | UMa 5G (line-of-sight + shadowing từ 3D building data) |
| SINR/RSRP | Layer 1 filtered + Layer 3 (L3, k=16) filtered |

---

## 3. Định nghĩa HO, PP, RLF

### Radio Link Failure (RLF)
- N310=10 lần liên tiếp SINR < Q_out (-8 dB) → khởi động timer T310 (1000 ms = 100 steps)
- Nếu T310 hết hạn mà chưa có N311=3 lần SINR > Q_in (-6 dB) → **RLF**
- Sau RLF: UE disconnect, chờ recovery 200 ms (20 steps)

### Handover Failure (HOF → RLF)
- Nếu T310 đang chạy khi HO prep xong → HOF → RLF
- Nếu T310 hết hạn trước khi HO execution xong → HOF → RLF

### Ping-Pong (PP)
- UE switch sang BS mới rồi quay lại BS cũ trong khoảng MTS = 1000 ms (100 steps) → **PP**

---

## 4. Các timer quan trọng

| Timer/Counter | Giá trị gốc | Số steps (Δt=10ms) |
|---|---|---|
| HO prep time (t_ho_prep) | 50 ms | **5 steps** |
| HO exec time (t_ho_exec) | 40 ms | 4 steps |
| MTS (Minimum Time of Stay) | 1000 ms | 100 steps |
| RLF recovery (t_rlfr) | 200 ms | 20 steps |
| T310 timer | 1000 ms | 100 steps |
| N310 counter | 10 | - |
| N311 counter | 3 | - |

### Cơ chế HO prep (quan trọng!)

```
Agent chọn target BS ≠ pcell
    → HO prep counter bắt đầu đếm
    → Nếu agent đổi target BS giữa chừng → prep RESET (nếu permit_ho_prep_abort=True)
    → Sau đúng 5 steps chọn cùng 1 BS → HO prep hoàn tất → bắt đầu HO exec (4 steps)
    → Sau HO exec: kết nối target BS (nếu SINR > Q_in)
```

**Vì vậy điều kiện thực tế để HO xảy ra = agent phải chọn cùng 1 BS trong 5 steps liên tiếp.**

---

## 5. State, Action, Reward

### State (2N+1 = 11 chiều, N=5)

```
s_t = [s_BS (5 dims), s_SINR (5 dims), s_PP (1 dim)]
```

| Thành phần | Mô tả |
|---|---|
| `s_BS` | One-hot vector chỉ BS đang serve (1 tại pcell, 0 còn lại) |
| `s_SINR` | SINR của từng BS, clip về [-10, 10] dB rồi scale về [0, 1] |
| `s_PP` | = 1 nếu `t - t_HO < MTS` (đang trong window PP detection), = 0 otherwise |

### Action

```
a_t ∈ {0, 1, 2, 3, 4}  →  index của BS muốn kết nối
```

Nếu agent chọn BS khác pcell → trigger HO prep. Nếu chọn pcell → giữ nguyên / abort prep.

### Reward

```
r(s_t, a_t) = r_SINR + r_PP + r_RLF
```

| Thành phần | Công thức |
|---|---|
| r_SINR | = normalized_SINR[pcell] + C nếu pcell là BS tốt nhất, else = normalized_SINR[pcell] |
| r_PP | = -C nếu PP detected, else 0 |
| r_RLF | = -2C nếu RLF detected; -C nếu SINR < Q_out (out-of-sync); else 0 |

**C = 0.95** (reward/penalty constant)

### Metric đánh giá

```
Γ_R = R̄ / R̄_max  ∈ [0, 1]
```

- R̄ = tốc độ trung bình thực tế của UE (dùng Shannon capacity với SINR tại BS đang serve)
- R̄_max = tốc độ lý tưởng nếu UE luôn connect BS tốt nhất và không có HO delay

---

## 6. Kiến trúc PPO

```
Actor-Critic MLP:
  - Input: 11 neurons (state)
  - Hidden: [64, 128, 64] với ReLU
  - Actor output: 5 neurons (softmax → action probabilities)
  - Critic output: 1 neuron (value function)
```

| Hyperparameter | Giá trị |
|---|---|
| Algorithm | PPO (Stable-Baselines3) |
| Learning rate | 5e-5 (linear schedule) |
| Entropy coef | 0.1 |
| n_steps_per_update | 2000 |
| batch_size | 200 |
| n_epochs | 10 |
| total_timesteps | 5,000,000 |
| discount γ | default SB3 (0.99) |

---

## 7. Two-Phase Training (gốc từ bài báo)

### Phase 1
- `terminate_on_rlf = True`, `terminate_on_pp = False`
- Episode kết thúc khi: RLF xảy ra hoặc đạt max timestep T
- Mục đích: agent học HO strategy mà không bị phạt PP

### Phase 2
- `terminate_on_rlf = True`, `terminate_on_pp = True`
- Episode kết thúc khi: RLF hoặc PP xảy ra hoặc đạt max T
- Mục đích: thêm penalty cho HO thừa, học policy ổn định hơn

### Vấn đề ở đầu training (nhận xét từ người đi trước)

Tại thời điểm ban đầu, agent chọn BS ngẫu nhiên với xác suất đều (1/5 cho mỗi BS):
- Xác suất 5 lần liên tiếp chọn cùng 1 BS = (1/5)^4 = **0.16%**
- → HO gần như không bao giờ xảy ra → agent không học được HO behavior

### Giải pháp (modification đề xuất)

Chia phase 1 thành 2 sub-phase bằng cách thay đổi `t_ho_prep`:

| Sub-phase | t_ho_prep | Số steps cần liên tiếp | Xác suất ngẫu nhiên |
|---|---|---|---|
| Phase 1a | 30 ms | **3 steps** | (1/5)^2 = **4%** |
| Phase 2 | 50 ms | **5 steps** | (1/5)^4 = 0.16% |

Cách implement: thay `config.t_ho_prep` trong quá trình training.

---

## 8. Cấu trúc source code

```
HandoverOptimDRL/
├── run.py                          # Entry point: train_ppo / validate_ppo / validate_3gpp / plot_results
├── src/ho_optim_drl/
│   ├── config.py                   # Tất cả hyperparameter (Config dataclass)
│   ├── dataloader.py               # Load RSRP/SINR từ file .mat (data/processed/)
│   ├── utils.py                    # clipnorm, speed filter, csv writer...
│   └── gym_env/
│       ├── ho_env_ppo.py           # Gym environment (HandoverEnvPPO)
│       ├── ho_protocol_ppo.py      # State machine: HO, RLF, PP logic (HOProcedurePPO)
│       ├── ho_env_3gpp.py          # 3GPP baseline environment
│       └── ho_protocol_3gpp.py     # 3GPP baseline protocol (Event A3)
├── scripts/
│   ├── train_ppo.py                # Training script
│   ├── validate_ppo.py             # Testing PPO model
│   ├── validate_3gpp.py            # Testing 3GPP baseline
│   └── plot_results.py             # Reproduce paper plots
└── data/processed/                 # RSRP/SINR .mat files (tốc độ 30,50,70,90 km/h)
```

### File quan trọng nhất

- `ho_protocol_ppo.py`: **Toàn bộ logic HO state machine**
  - `HOProcedurePPO.step()`: chạy 1 timestep
  - `_ho_preparation_handler()`: xử lý HO prep (reset nếu đổi target)
  - `_ho_state_machine()`: chuyển từ prep → exec → connect
  - `_rlf_detection()`: N310/N311/T310 logic
  - `_pp_monitoring()`: MTS monitoring
- `ho_env_ppo.py`: Gym wrapper, reward function (`_get_reward()`)
- `config.py`: Thay đổi hyperparameter tại đây

---

## 9. Chạy thử

```bash
# Cài đặt
pip install -e .

# Training
python run.py train_ppo

# Validation
python run.py validate_ppo

# 3GPP baseline
python run.py validate_3gpp

# Plot kết quả
python run.py plot_results
```

Model được save tại: `results/models/ppo_sweep/<run_name>/`  
Model để validate: `results/models/ppo_model/model`

---

## 10. Implement 2-phase training với modification

```python
# Phase 1a: t_ho_prep = 3 steps (30ms)
config.t_ho_prep = 3  # 30ms / 10ms
config.terminate_on_pp = False
config.terminate_on_rlf = True
# Train khoảng 2.5M steps...

# Phase 2: t_ho_prep = 5 steps (50ms) - gốc
config.t_ho_prep = 5  # 50ms / 10ms
config.terminate_on_pp = True
config.terminate_on_rlf = True
# Train tiếp 2.5M steps...
```

Lưu ý: phải tạo lại environment sau khi thay đổi config vì `HOProcedurePPO.__init__` đọc config tại thời điểm khởi tạo.

---

## 11. Kết quả bài báo (benchmark)

Trên dataset test (30–50 km/h train, 30–90 km/h test):

| Protocol | Γ_R (%) | Nhận xét |
|---|---|---|
| 3GPP HO | ~99.6–99.8% | Giảm ở tốc độ cao |
| PPO HO | ~99.7–99.9% | Ổn định hơn ở tốc độ cao |

PPO cải thiện đáng kể ở vận tốc cao (>50 km/h) vì timing HO linh hoạt hơn.  
SINR sau HO (dashed line trong Fig. 5) của PPO cao hơn 3GPP → HO sang BS tốt hơn.

---

## 12. Lưu ý GPU RTX 3060 Ti

- Code tự detect CUDA: `device=torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- 5M timesteps với MLP nhỏ [64,128,64] → training rất nhanh (~30–60 phút)
- VRAM 8GB là quá đủ cho model size này
- Có thể tăng `n_steps_per_update` hoặc `batch_size` nếu muốn tận dụng GPU hơn
