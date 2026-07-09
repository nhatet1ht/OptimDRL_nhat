# Thử nghiệm V4: Các biến thể BC+Curriculum (2026-06-24)

> Tiếp nối V3. Sau khi phát hiện bc_curriculum dùng ent_coef=0.01 (quá thấp so với paper=0.1),
> thử 4 biến thể với ent_coef=0.1 và các thay đổi cấu trúc khác nhau.

---

## Lý do thực hiện V4

Từ kết quả V3 (bc_curriculum, ent_coef=0.01):
- Phase 0: G=38-50%, PP=0%, RLF=0% → policy học "never HO"
- Final: G=21-37%, PP=0%, RLF=225-295% → policy HO nhưng thất bại liên tục

**Root cause phát hiện**: `ent_coef=0.01` quá thấp → entropy collapse nhanh → policy stuck
"never HO". Paper dùng `ent_coef=0.1` (10x cao hơn).

---

## 4 phương án V4

### V4.1 — bc_paper: BC init + paper params trực tiếp

**Script**: `scripts/train_ppo_bc_paper.py`

**Thiết kế**:
- BC init: 30 epochs, oracle = argmax(sinr_norm), 100% accuracy
- PPO: paper params trực tiếp (t_ho_prep=5, t_ho_exec=4, ent_coef=0.1, lr=5e-5)
- Không curriculum, 5M steps
- Hypothesis: BC giải quyết cold-start, paper params đủ tốt từ đó

**Kết quả training**:
- entropy_loss dao động -1.34 → -1.54 (không bị uniform hoàn toàn)
- ep_len_mean = 5400 (dài → ít RLF/PP termination ở đầu)
- fps = ~2084

**Kết quả eval (paper params t_ho_prep=5, t_ho_exec=4)**:

| Speed | G_R (%) | PP rate (%) | RLF rate (%) |
|:---:|:---:|:---:|:---:|
| 30 | 24.1 | 0.0 | 250.0 |
| 50 | 33.2 | 0.0 | 490.0 |
| 70 | 31.1 | 0.0 | 325.0 |
| 90 | 30.6 | 0.0 | 361.5 |

**Nhận xét**: PP=0% nhưng RLF=250-490% → cố HO nhưng thất bại liên tục.
Target BS thay đổi trong 9 timestep (5+4) → RLF.

---

### V4.2 — bc_highent: BC + Curriculum + ent_coef=0.1

**Script**: `scripts/train_ppo_bc_highent.py`

**Thiết kế**:
- BC init: 30 epochs
- 5-phase curriculum (1→2→3→4→5 prep): giống V3 nhưng **ent_coef=0.1**
- SubprocVecEnv n_envs=4, GPU
- 1M steps/phase = 5M tổng

**Kết quả eval**:

| Speed | G_R (%) | PP rate (%) | RLF rate (%) |
|:---:|:---:|:---:|:---:|
| 30 | 9.1 | 0.0 | 225.0 |
| 50 | 21.8 | 14.3 | 285.7 |
| 70 | 26.6 | 23.8 | 228.6 |
| 90 | 27.9 | 5.3 | 278.9 |

**Nhận xét**: G thấp nhất trong V4. Ent cao hơn → explore nhiều HO hơn → nhiều RLF
hơn. Có PP=14-24% ở 50-70 km/h (policy đang ping-pong). Kết quả tệ nhất.

---

### V4.3 — bc_noterm: BC + Curriculum + không terminate

**Script**: `scripts/train_ppo_bc_noterm.py`

**Thiết kế**:
- BC init: 30 epochs
- 5-phase curriculum
- **terminate_on_rlf=False, terminate_on_pp=False** xuyên suốt (chỉ reward penalty)
- ent_coef=0.1, SubprocVecEnv n_envs=4, GPU
- Rationale: Không terminate → policy học từ hậu quả thay vì bị reset

**Kết quả eval**:

| Speed | G_R (%) | PP rate (%) | RLF rate (%) |
|:---:|:---:|:---:|:---:|
| 30 | 38.8 | 0.0 | 0.0 |
| 50 | 41.3 | 0.0 | 0.0 |
| 70 | 50.0 | 0.0 | 0.0 |
| 90 | 44.3 | 0.0 | 0.0 |

**Nhận xét**: Giống hệt Imitation V2 (PP=0%, RLF=0%, G=38-50%). Policy học
"never HO" vì không bị terminate nhưng HO cost (disconnect) vẫn âm → safer không HO.

---

### V4.4 — bc_2m: BC + Curriculum + 2M steps/phase

**Script**: `scripts/train_ppo_bc_2m.py`

**Thiết kế**:
- BC init: 30 epochs
- 5-phase curriculum với **2M steps/phase** (10M tổng)
- ent_coef=0.1, SubprocVecEnv n_envs=4, GPU

**Kết quả eval**:

| Speed | G_R (%) | PP rate (%) | RLF rate (%) |
|:---:|:---:|:---:|:---:|
| 30 | 22.9 | 0.0 | 250.0 |
| 50 | 35.1 | 0.0 | 490.0 |
| 70 | 31.1 | 0.0 | 350.0 |
| 90 | 30.6 | 0.0 | 361.5 |

**Nhận xét**: Gần giống bc_paper (cùng pattern HO+RLF). Thêm steps không giúp được.
RLF pattern giống hệt nhau → cùng fundamental problem.

---

## Tổng hợp V4

| Model | G_R TB | PP TB | RLF TB | Pattern |
|---|:---:|:---:|:---:|---|
| bc_paper | 29.8% | 0% | 357% | HO badly |
| bc_highent | 21.4% | 10.9% | 254% | HO badly + PP |
| bc_noterm | **43.5%** | 0% | 0% | Never HO |
| bc_2m | 29.9% | 0% | 363% | HO badly |

**Kết luận**: Hai nhóm kết quả rõ ràng:
1. Policy "never HO": G=38-50%, an toàn nhưng không theo kịp BS tốt nhất
2. Policy "HO badly": G=22-35%, cố HO nhưng RLF=250-490%

Không phương án nào vượt qua được G=50%.

---

## Phân tích root cause sau V4

**Vấn đề cốt lõi**: Với t_ho_prep=5 + t_ho_exec=4 = **9 timesteps delay**,
UE di chuyển nhanh (30-90 km/h × 100ms/step) → target BS thay đổi trong khi HO
đang thực hiện → RLF không thể tránh trừ khi policy **dự đoán được BS tốt nhất
sau 9 bước**.

**Tại sao paper model (G=99.8%) hoạt động?**
- Có thể được train với trick/seed đặc biệt không được document
- Hoặc có thể dự đoán tốt nhờ pattern SINR trong data cụ thể
- Cần phân tích thêm behavior của paper model

**Hướng tiếp theo**:
1. Thêm SINR look-ahead vào observation (obs[t+1..t+9])
2. Phân tích paper model: tần suất HO, timing, độ chính xác dự đoán
3. Thử reward shaping mạnh hơn: r = sinr_norm[action, t+9]
