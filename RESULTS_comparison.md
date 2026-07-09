# Kết quả so sánh tổng hợp: Tất cả phương án đã thử

> Bài báo: *A Deep Reinforcement Learning-Based Approach for Adaptive Handover Protocols* (SCC 2025, KIT)
> Metric chính: **G_R** = relative average rate (Gamma_R trong paper)
> PP rate = ping-pong rate, RLF rate = radio link failure rate

---

## 1. Mục tiêu: Kết quả từ bài báo gốc (Fig. 4)

| Speed (km/h) | G_R 3GPP (%) | G_R PPO (%) |
|:---:|:---:|:---:|
| 30 | ~99.75 | ~99.80 |
| 50 | ~99.77 | ~99.83 |
| 70 | ~99.78 | ~99.85 |
| 90 | ~99.70 | ~99.80 |

---

## 2. Baseline (tai hien thanh cong)

### 3GPP Baseline

| Speed | G_R (%) | PP rate (%) | RLF rate (%) |
|:---:|:---:|:---:|:---:|
| 30 | **99.756** | 41.83 | 3.37 |
| 50 | **99.751** | 50.00 | 2.05 |
| 70 | **99.797** | 47.26 | 2.53 |
| 90 | **99.696** | 44.41 | 6.99 |

### PPO goc (pre-trained model cua paper)

| Speed | G_R (%) | PP rate (%) | RLF rate (%) |
|:---:|:---:|:---:|:---:|
| 30 | **99.795** | 43.58 | 1.17 |
| 50 | **99.823** | 45.38 | 0.70 |
| 70 | **99.839** | 46.45 | 2.84 |
| 90 | **99.763** | 47.53 | 4.63 |

---

## 3. V1 — Training tu scratch (THAT BAI)

### PPO 2-phase (paper params, t_ho_prep=5)

| Speed | G_R (%) | PP (%) | RLF (%) | Policy |
|:---:|:---:|:---:|:---:|:---:|
| 30 | 22.2 | 0.0 | 193.3 | Uniform H=1.61 |
| 50 | 10.1 | 0.0 | 185.7 | Uniform H=1.61 |
| 70 | 16.5 | 4.5 | 204.5 | Uniform H=1.61 |
| 90 | 29.4 | 0.0 | 196.2 | Uniform H=1.61 |

Root cause: Credit assignment. P(HO/step)=0.16% -> gradient ~= 0.

---

## 4. V2 — 4 phuong an moi (2026-06-23)

### 4.1 Curriculum t_ho_prep = 1 -> 3 -> 5

| Speed | G_R (%) | PP (%) | RLF (%) |
|:---:|:---:|:---:|:---:|
| 30 | 29.5 | 97.1 | 68.2 |
| 50 | 12.6 | 95.7 | 64.5 |
| 70 | 23.7 | 95.9 | 63.6 |
| 90 | 25.3 | 95.7 | 68.9 |

Phase 0 entropy -1.61 -> -1.07 (BREAKTHROUGH), Phase 2 ve lai -1.61. PP=97%.

### 4.2 Reward Shaping alpha=0.1

| Speed | G_R (%) | PP (%) | RLF (%) |
|:---:|:---:|:---:|:---:|
| 30 | 36.5 | 25.0 | 40.0 |
| 50 | 22.7 | 41.3 | 173.9 |
| 70 | 23.8 | 34.7 | 124.5 |
| 90 | 17.8 | 26.3 | 122.8 |

Entropy stuck -1.61. Alpha qua nho.

### 4.3 Imitation Learning (BC + PPO)

| Speed | G_R (%) | PP (%) | RLF (%) | Pattern |
|:---:|:---:|:---:|:---:|:---:|
| 30 | 38.8 | 0.0 | 0.0 | Never HO |
| 50 | 41.3 | 0.0 | 0.0 | Never HO |
| 70 | 50.0 | 0.0 | 0.0 | Never HO |
| 90 | 44.3 | 0.0 | 0.0 | Never HO |

BC 100% accuracy sau 2 epochs. PPO hoc tranh HO de tranh PP penalty.

### 4.4 Multiple Seeds (5 seeds x 2M steps)

Tat ca 5 seeds: entropy = 1.609-1.610 (uniform). Problem la systematic.

---

## 5. V3 — BC + Gradual Curriculum (2026-06-24)

**Script**: `train_ppo_bc_curriculum.py` — ent_coef=0.01, 5 phases, 1M steps/phase

### Diagnostic — phase 0 cua old curriculum (t_ho_prep=1)

| Speed | G_R (%) | PP (%) | RLF (%) |
|:---:|:---:|:---:|:---:|
| 30 | 35.8 | 81.5 | 48.7 |
| 50 | 22.1 | 90.9 | 61.1 |
| 70 | 23.4 | 85.0 | 63.7 |
| 90 | 17.5 | 83.7 | 63.5 |

### Diagnostic — BC+Curriculum Phase 0 (t_ho_prep=1, ent=0.01)

| Speed | G_R (%) | PP (%) | RLF (%) | Pattern |
|:---:|:---:|:---:|:---:|:---:|
| 30 | 38.8 | 0.0 | 0.0 | Never HO |
| 50 | 41.3 | 0.0 | 0.0 | Never HO |
| 70 | 50.0 | 0.0 | 0.0 | Never HO |
| 90 | 44.3 | 0.0 | 0.0 | Never HO |

### BC+Curriculum Final (paper params)

| Speed | G_R (%) | PP (%) | RLF (%) |
|:---:|:---:|:---:|:---:|
| 30 | 31.3 | 0.0 | 225.0 |
| 50 | 37.4 | 0.0 | 294.7 |
| 70 | 25.8 | 0.0 | 235.0 |
| 90 | 21.1 | 0.0 | 239.1 |

Root cause xac nhan: ent_coef=0.01 -> policy stuck "never HO" hoac HO badly.

---

## 6. V4 — Cac bien the BC+Curriculum (2026-06-24)

**Cai tien chinh**: ent_coef=0.1 (paper value), SubprocVecEnv n_envs=4, GPU

### 6.1 bc_paper — BC + paper params truc tiep

| Speed | G_R (%) | PP (%) | RLF (%) |
|:---:|:---:|:---:|:---:|
| 30 | 24.1 | 0.0 | 250.0 |
| 50 | 33.2 | 0.0 | 490.0 |
| 70 | 31.1 | 0.0 | 325.0 |
| 90 | 30.6 | 0.0 | 361.5 |

### 6.2 bc_highent — BC + Curriculum + ent_coef=0.1

| Speed | G_R (%) | PP (%) | RLF (%) |
|:---:|:---:|:---:|:---:|
| 30 | 9.1 | 0.0 | 225.0 |
| 50 | 21.8 | 14.3 | 285.7 |
| 70 | 26.6 | 23.8 | 228.6 |
| 90 | 27.9 | 5.3 | 278.9 |

### 6.3 bc_noterm — BC + Curriculum + khong terminate

| Speed | G_R (%) | PP (%) | RLF (%) | Pattern |
|:---:|:---:|:---:|:---:|:---:|
| 30 | 38.8 | 0.0 | 0.0 | Never HO |
| 50 | 41.3 | 0.0 | 0.0 | Never HO |
| 70 | 50.0 | 0.0 | 0.0 | Never HO |
| 90 | 44.3 | 0.0 | 0.0 | Never HO |

### 6.4 bc_2m — BC + Curriculum + 2M steps/phase

| Speed | G_R (%) | PP (%) | RLF (%) |
|:---:|:---:|:---:|:---:|
| 30 | 22.9 | 0.0 | 250.0 |
| 50 | 35.1 | 0.0 | 490.0 |
| 70 | 31.1 | 0.0 | 350.0 |
| 90 | 30.6 | 0.0 | 361.5 |

---

## 7. Tong hop tat ca phuong an

| # | Method | G_R TB (%) | PP TB (%) | RLF TB (%) | Status |
|---|---|:---:|:---:|:---:|---|
| 0 | **Paper PPO (original)** | **99.81** | 45.7 | 2.3 | TARGET |
| 0 | 3GPP baseline | 99.75 | 45.9 | 3.7 | Reference |
| 1 | PPO 2-phase (paper params) | 19.6 | 1.1 | 195.0 | FAIL - uniform |
| 2 | Curriculum 1->3->5 | 22.8 | 96.1 | 66.3 | FAIL - PP=97% |
| 3 | Reward shaping alpha=0.1 | 25.2 | 31.8 | 115.0 | FAIL - uniform |
| 4 | Imitation (BC+PPO) | 43.6 | 0.0 | 0.0 | PARTIAL - never HO |
| 5 | Multiseed (5 seeds x 2M) | 20.7 | 49.4 | 93.2 | FAIL - uniform |
| 6 | BC+Curriculum ent=0.01 | 28.9 | 0.0 | 248.5 | FAIL - HO badly |
| 7 | bc_paper (BC+paper params) | 29.8 | 0.0 | 356.6 | FAIL - HO badly |
| 8 | bc_highent (BC+ent=0.1) | 21.4 | 10.9 | 254.6 | FAIL - HO badly |
| 9 | bc_noterm (BC+no-term) | 43.5 | 0.0 | 0.0 | PARTIAL - never HO |
| 10 | bc_2m (BC+2M steps) | 29.9 | 0.0 | 362.9 | FAIL - HO badly |

---

## 8. Ket luan tong quat

**Hai trang thai policy hoc duoc — deu khong dat muc tieu:**

- **"Never HO"** (imitation, bc_noterm, bc+curriculum phase 0): G~43-50%
  - PP=0%, RLF=0% — an toan
  - Khong theo kip BS tot nhat khi UE di chuyen
  - Xuat hien khi: terminate_on_pp=True hoac chi co reward penalty (khong terminate)

- **"HO badly"** (bc_paper, bc_2m, bc+curriculum final): G~22-35%
  - RLF=250-490% — that bai lien tuc
  - Xuat hien khi: ent_coef cao + t_ho_prep lon
  - Nguyen nhan: 9 timestep delay (5+4) -> target BS thay doi -> RLF

**Ly do paper model (G=99.8%) hoat dong tot:**
- Co the duoc train voi initialization dac biet
- Hoac co the du doan duoc BS tot nhat sau 9 buoc tu pattern SINR
- Chua xac dinh duoc mechanism chinh xac

**Huong nghien cuu tiep theo:**
1. Phan tich paper model: timing HO, do chinh xac du doan
2. Them SINR look-ahead vao observation (t+1 den t+9)
3. Reward shaping du bao tuong lai: r = sinr_norm[action, t+9]
4. Kiem tra lai environment code co bug khong

---

*Cap nhat lan cuoi: 2026-06-24*
