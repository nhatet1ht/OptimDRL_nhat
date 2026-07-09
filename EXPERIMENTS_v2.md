# Thử nghiệm V2: 4 phương án mới (2026-06-23)

> Sau khi xác định root cause là credit assignment problem, thử 4 phương án giải quyết.

---

## Phương án 1: Curriculum t_ho_prep = 1 → 3 → 5

**Script**: `scripts/train_ppo_curriculum.py`

**Ý tưởng**: Phase 0 với t_ho_prep=1 biến bài toán thành multi-armed bandit đơn giản:
- P(HO trigger/step) = 80% (thay vì 0.16%)
- Reward phụ thuộc action ngay lập tức
- Agent học "pick best SINR BS" nhanh

**Config**:
| Phase | t_ho_prep | t_ho_exec | terminate_rlf | terminate_pp | Steps |
|---|---|---|---|---|---|
| 0 | 1 (10ms) | 1 (10ms) | False | False | 1M |
| 1 | 3 (30ms) | 2 (20ms) | True | False | 2M |
| 2 | 5 (50ms) | 4 (40ms) | True | True | 2M |

**Kết quả training - Phase 0** ← BREAKTHROUGH!:
```
40K steps: entropy -1.29 (so với -1.61 stuck trước đây)
ep_rew_mean: -1290 → -232 → +62 → +335 → +495 → +3660
Phase 0 end: entropy = -1.07, ep_rew = +3660
```
**Policy đang học! Đây là lần đầu tiên từ trước đến nay thấy entropy giảm.**

**Kết quả training - Phase 1**:
```
entropy: -1.07 → -1.58 (REGRESSION)
ep_len = 185 (RLF đang terminate sớm)
ep_rew = 180
```

**Kết quả training - Phase 2**:
```
entropy = -1.61 (về lại UNIFORM!)
approx_kl = 1e-7 (policy gần như không thay đổi)
ep_rew = 127-128, ep_len = 283
```

**Kết quả eval (với t_ho_prep=5 — paper params)**:

| Speed | G_R (%) | PP rate (%) | RLF rate (%) |
|:---:|:---:|:---:|:---:|
| 30 | 29.5 | 97.1 | 68.2 |
| 50 | 12.6 | 95.7 | 64.5 |
| 70 | 23.7 | 95.9 | 63.6 |
| 90 | 25.3 | 95.7 | 68.9 |

**Phân tích thất bại**:
- Phase 0 thành công: policy học "luôn chọn best BS"
- Khi chuyển sang Phase 2 (t_ho_prep=5): credit assignment problem quay lại
- Policy reverted về uniform (entropy -1.61)
- Eval với t_ho_prep=5: uniform policy + PP termination → 97% ping-pong
- Lý do PP=97%: số ít HO completions (0.16%) thường là "ngẫu nhiên" → UE muốn HO lại ngay

**Checkpoint Phase 0 đã lưu**: `results/models/ppo_curriculum/model_phase0.zip`

---

## Phương án 2: Reward Shaping r += α·sinr_norm[action]

**Script**: `scripts/train_ppo_reward_shaped.py`

**Ý tưởng**: Thêm immediate reward dựa trên action chọn:
```python
r_shaped = alpha * sinr_norm[action]  # alpha = 0.1
```
- action = best_BS → r_shaped ≈ 0.1 × 1.0 = 0.1 (cao)
- action = worst_BS → r_shaped ≈ 0.1 × 0.0 = 0.0 (thấp)
- Advantage A(s, best_BS) > A(s, other_BS) → policy học chọn best BS
- Positive feedback: chọn best BS → commit 5 steps → HO → SINR reward lớn

**Config**:
- Phase 1: terminate_on_pp=False → 2.5M steps
- Phase 2: terminate_on_pp=True → 2.5M steps
- Subclass `HandoverEnvShaped(HandoverEnvPPO)` override `_get_reward()`

**Kết quả training**:
```
entropy = -1.61 sau toàn bộ 5M steps (THẤT BẠI)
```

**Kết quả eval**:

| Speed | G_R (%) | PP rate (%) | RLF rate (%) |
|:---:|:---:|:---:|:---:|
| 30 | 36.5 | 25.0 | 40.0 |
| 50 | 22.7 | 41.3 | 173.9 |
| 70 | 23.8 | 34.7 | 124.5 |
| 90 | 17.8 | 26.3 | 122.8 |

**Phân tích thất bại**:
- alpha=0.1 quá nhỏ để break uniform distribution
- Gradient từ SINR reward (không phụ thuộc action) vẫn dominant
- Với 5M steps và entropy penalty=0.1: r_shaped contribution bị lấn át

**Ghi chú**: Có thể thử alpha=0.5 hoặc alpha=1.0 trong tương lai. Risk: destabilize training nếu alpha quá lớn.

---

## Phương án 3: Imitation Learning (BC + PPO)

**Script**: `scripts/train_ppo_imitation.py`

**Ý tưởng**:
1. Stage 1 - Behavioral Cloning: Pre-train actor với oracle policy (argmax(sinr))
2. Stage 2 - PPO Fine-tuning: Bắt đầu từ informed policy thay vì random

**BC Dataset**: 348,400 samples từ 57 datasets training
- obs = [one_hot(best_bs), sinr_norm, pp=0]
- oracle_action = argmax(sinr_norm)

**Config BC**:
- 20 epochs, batch_size=512, lr=3e-4
- Train chỉ actor (pi branch), không train critic

**Kết quả BC**:
```
Epoch  1: loss=0.2061, acc=98.9%
Epoch  2: loss=0.0004, acc=100.0%
Epoch 3-20: loss≈0.0000, acc=100.0%
```

Policy sau BC:
```
best_BS=0: action=0, probs=[1., 0., 0., 0., 0.]
best_BS=1: action=1, probs=[0., 1., 0., 0., 0.]
...  (hoàn toàn deterministic!)
```

**Config PPO Fine-tuning**:
- ent_coef = 0.01 (giảm để preserve BC structure)
- Phase 1: 2.5M steps, terminate_on_pp=False
- Phase 2: 2.5M steps, terminate_on_pp=True, lr constant 1e-5

**Kết quả training PPO**:
```
Phase 2 end:
  entropy = -2.77e-5 ≈ 0  (near-deterministic policy!)
  approx_kl = 0.0 (policy frozen!)
```

**Kết quả eval**:

| Speed | G_R (%) | PP rate (%) | RLF rate (%) |
|:---:|:---:|:---:|:---:|
| 30 | 38.8 | **0.0** | **0.0** |
| 50 | 41.3 | **0.0** | **0.0** |
| 70 | 50.0 | **0.0** | **0.0** |
| 90 | 44.3 | **0.0** | **0.0** |

**Phân tích**:
- PP=0%, RLF=0% → tốt về kết quả phụ
- Nhưng G=38-50% → chứng tỏ policy "không bao giờ HO"
- UE kẹt ở BS ban đầu (argmax RSRP tại t=0), không chuyển khi channel thay đổi
- Lý do: PPO Phase 2 với terminate_on_pp=True → mọi HO đều bị PP penalty → agent học "tránh HO hoàn toàn"

**Điểm tích cực**:
- BC hoạt động hoàn hảo (100% accuracy sau 2 epochs)
- ent_coef=0.01 giữ được structure deterministic qua PPO
- Zero PP và RLF cho thấy potential nếu timing được học đúng

---

## Phương án 4: Multiple Seeds

**Script**: `scripts/train_ppo_multiseed.py`

**Ý tưởng**: Paper có thể đã dùng lucky seed. Thử 5 seeds × 2M steps.

**Seeds**: [0, 42, 123, 777, 1234]

**Kết quả**:
```
Seed     0: entropy = 1.6093 (uniform)
Seed    42: entropy = 1.6094 (uniform)
Seed   123: entropy = 1.6094 (uniform)
Seed   777: entropy = 1.6093 (uniform)
Seed  1234: entropy = 1.6090 (uniform, "best")
```

**Kết luận**: Credit assignment problem là **systematic**, không phải do seed.
Với 2M steps standard 2-phase training, tất cả seeds đều stuck uniform.

---

## So sánh tổng hợp V2

| Model | G_R TB (%) | PP rate TB (%) | RLF rate TB (%) | Status |
|---|:---:|:---:|:---:|---|
| Paper PPO (original) | **99.8** | 45.7 | 2.3 | ✅ TARGET |
| 3GPP baseline | 99.75 | 45.9 | 3.7 | ✅ Baseline |
| PPO 2-phase (paper) | ~17 | ~1 | ~190 | ❌ |
| **Curriculum** | **22.7** | **96.1** | **66.3** | ❌ PP quá cao |
| Reward shaped | 25.2 | 31.8 | 115 | ❌ |
| **Imitation** | **43.6** | **0.0** | **0.0** | ⚠️ Never HO |
| Multiseed | 20.7 | 49.4 | 93.2 | ❌ |

---

## Phân tích tại sao tất cả thất bại

### Curriculum: Phase transition problem
```
Phase 0 (prep=1): entropy 1.61 → 1.07 ✅ Đang học!
Phase 1 (prep=3): entropy 1.07 → 1.58 ⚠️ Regression
Phase 2 (prep=5): entropy 1.58 → 1.61 ❌ Back to uniform
```
Root cause: Mỗi khi t_ho_prep tăng, credit assignment problem xuất hiện lại.
Policy không thể "remember" gì từ phase trước nếu reward signal mất đi.

### Imitation: PP penalty kills HO
```
BC: policy = "always pick argmax(sinr)" → deterministic ✅
PPO Phase 1 (no PP): agent HOs frequently following BC policy ✅  
PPO Phase 2 (PP terminate): every HO → PP penalty → agent learns "avoid HO" ❌
```
Root cause: BC init tốt nhưng PP termination quá aggressive → policy collapse sang "never HO".
Solution needed: gradual PP introduction (đang thử trong V3).

---

## Bài học từ V2

1. **Credit assignment** là vấn đề chính, không phải seed hay ent_coef
2. **Curriculum Phase 0 work!** — entropy thực sự giảm khi t_ho_prep=1
3. **BC work!** — 100% accuracy, deterministic policy sau 2 epochs
4. **PP termination quá sớm** phá hỏng imitation approach
5. **Transition phải mượt hơn** — không thể jump từ prep=1 thẳng lên prep=5
