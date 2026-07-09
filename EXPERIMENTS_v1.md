# Thử nghiệm V1: Baseline và 2 lần thất bại đầu tiên

> Mục tiêu: Tái hiện kết quả training PPO từ scratch khớp với bài báo (G_R ≈ 99.8%)

---

## Kết quả baseline (tái hiện thành công)

### 3GPP Baseline (validate_3gpp)

| Speed (km/h) | G_R (%) | PP rate (%) | RLF rate (%) |
|:---:|:---:|:---:|:---:|
| 30 | 99.756 | 41.83 | 3.37 |
| 50 | 99.751 | 50.00 | 2.05 |
| 70 | 99.797 | 47.26 | 2.53 |
| 90 | 99.696 | 44.41 | 6.99 |

### PPO gốc (model của paper, validate_ppo)

| Speed (km/h) | G_R (%) | PP rate (%) | RLF rate (%) |
|:---:|:---:|:---:|:---:|
| 30 | 99.795 | 43.58 | 1.17 |
| 50 | 99.823 | 45.38 | 0.70 |
| 70 | 99.839 | 46.45 | 2.84 |
| 90 | 99.763 | 47.53 | 4.63 |

**Kết luận**: Tái hiện evaluation thành công (sai số < 0.02% so với paper Fig. 4) ✅

Policy của model gốc: hoàn toàn deterministic
```
best_BS=3 -> action=3, probs=[0., 0., 0., 1., 0.]  (H=0)
best_BS=1 -> action=1, probs=[0., 1., 0., 0., 0.]  (H=0)
```

---

## Thử nghiệm 1: t_ho_prep=3 trong Phase 1 ❌

**Script**: `scripts/train_ppo_2phase.py` (phiên bản đầu với t_ho_prep=3)

**Config**:
- Phase 1: t_ho_prep=3, terminate_on_pp=False, terminate_on_rlf=True
- Phase 2: t_ho_prep=5, terminate_on_pp=True, terminate_on_rlf=True
- Total: 5M steps

**Lý do thử**: Với t_ho_prep=3, P(HO/step) = (1/5)² = 4.0% thay vì 0.16% → tăng 25×

**Kết quả training**:
```
entropy_loss = -1.61 xuyên suốt (uniform)
```

**Kết quả eval**:
- G_R = 14–34%, RLF rate = 57–142%

**Phân tích**:
Với t_ho_prep=3, HO xảy ra nhiều hơn nhưng đến BS ngẫu nhiên (không phải best BS) → SINR thấp → HOF → RLF. Reward quá noisy, gradient không đủ để thoát uniform.

---

## Thử nghiệm 2: 2-phase đúng paper (t_ho_prep=5) ❌

**Script**: `scripts/train_ppo_2phase.py` (phiên bản hiện tại)

**Config**:
- Phase 1: t_ho_prep=5, terminate_on_pp=False, terminate_on_rlf=True → 2.5M steps
- Phase 2: t_ho_prep=5, terminate_on_pp=True, terminate_on_rlf=True → 2.5M steps
- lr Phase 2: constant 1e-5 (fix bug lr_schedule)

**Kết quả training**:
```
entropy_loss = -1.61 xuyên suốt TOÀN BỘ 5M steps
Action probs: [0.199, 0.199, 0.202, 0.203, 0.197]  ← UNIFORM
```

**Kết quả eval**: Chưa evaluate riêng (từ evaluate_all: G≈22%, RLF>185%)

**Root cause xác định**:

Với uniform random policy:
- P(5 bước liên tiếp cùng BS) = (1/5)⁴ = **0.16%**
- Trong 5M steps: chỉ ~860 HO completions
- UE hầu như không bao giờ HO → pcell không đổi
- reward ≈ sinr_norm[pcell] = const (**không phụ thuộc action**)
- Advantage A(s,a) ≈ 0 → **gradient ≈ 0**
- Policy không thể thoát khỏi uniform dù train bao lâu

**Lưu ý quan trọng**: Bug lr_schedule với reset_num_timesteps
```python
# SAI: lr=0 ngay từ đầu Phase 2
model.learn(total_timesteps=2_500_000, reset_num_timesteps=False)

# ĐÚNG: reset counter để linear schedule hoạt động
model.lr_schedule = lambda _: config.lr * 0.2  # constant 1e-5
model.learn(total_timesteps=2_500_000, reset_num_timesteps=True)
```

---

## Entropy check với nhiều ent_coef (300K steps mỗi cái)

```
ent_coef=0.1:   H=1.6094  (max=1.6094) → 100.0% maximum entropy
ent_coef=0.01:  H=1.6093               →  99.9% maximum entropy
ent_coef=0.001: H=1.6084               →  99.9% maximum entropy
```

**Kết luận**: Giảm entropy coefficient KHÔNG giúp được. Vấn đề là gradient từ policy objective, không phải từ entropy regularization.

---

## Kết luận V1

Credit assignment problem là root cause. Với t_ho_prep=5:
- Reward không phụ thuộc action → Advantage ≈ 0 → gradient ≈ 0
- Không thể train từ scratch với hyperparameters của paper trong 5M steps
