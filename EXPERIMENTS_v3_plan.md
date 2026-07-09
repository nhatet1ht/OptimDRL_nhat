# Thử nghiệm V3: BC + Gradual Curriculum (2026-06-24)

> Phương án kết hợp tốt nhất từ V2: BC init + curriculum mượt + exec ngắn

---

## Thiết kế phương án

**Script**: `scripts/train_ppo_bc_curriculum.py`

### Vấn đề cần giải quyết từ V2:
1. Curriculum V1: Phase 0 học được nhưng transition lên prep=5 quá đột ngột → policy reset
2. Imitation: BC tốt nhưng PP penalty kills HO behavior  
3. Cần: smooth transition + giới thiệu PP từ từ + exec ngắn để giảm HOF

### Giải pháp:
```
BC init (oracle: argmax sinr, 30 epochs) 
→ Phase 0: prep=1, exec=1, no-RLF, no-PP   → 1M steps (warm up)
→ Phase 1: prep=2, exec=1, RLF, no-PP      → 1M steps
→ Phase 2: prep=3, exec=2, RLF, no-PP      → 1M steps
→ Phase 3: prep=4, exec=3, RLF, PP         → 1M steps (PP introduced)
→ Phase 4: prep=5, exec=4, RLF, PP         → 1M steps (paper params)
Total: 5M PPO steps
```

### Key design choices:
- **ent_coef=0.01**: preserve BC structure (vs 0.1 trong paper)
- **t_ho_exec giảm dần từ 1→4**: ít disconnection time ở đầu → ít HOF → smoother learning  
- **PP chỉ từ Phase 3**: agent học HO trước khi học tránh PP
- **lr giảm dần**: 3e-5 → 2e-5 → 1.5e-5 → 1e-5 → 5e-6
- **1M steps/phase**: đủ để adapt nhưng không quá nhiều để drift

---

## Eval Checkpoint Diagnostic

**Script**: `scripts/eval_checkpoints.py`

Trước khi biết kết quả final, eval Phase 0 checkpoint với đúng config của nó:

**Câu hỏi**: Phase 0 curriculum (entropy=-1.07) có thực sự học HO tốt không?
- Nếu G_R ≈ 99% với t_ho_prep=1 → concept đúng, vấn đề chỉ là transition
- Nếu G_R thấp → ngay cả Phase 0 cũng chưa học đủ tốt

---

## Kết quả diagnostic (eval Phase 0)

*Cập nhật sau khi eval chạy xong*

### Curriculum Phase 0 @ t_ho_prep=1, t_ho_exec=1

| Speed | G_R (%) | PP rate (%) | RLF rate (%) | Nhận xét |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | |

### Curriculum Phase 1 @ t_ho_prep=3, t_ho_exec=2

| Speed | G_R (%) | PP rate (%) | RLF rate (%) | Nhận xét |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | |

---

## Kết quả BC + Gradual Curriculum

*Cập nhật sau khi training xong*

### Policy check sau mỗi phase

| Phase | config | entropy | Policy behavior |
|---|---|---|---|
| BC init | - | ~0 | Deterministic best-BS |
| Phase 0 | prep=1 | TBD | |
| Phase 1 | prep=2 | TBD | |
| Phase 2 | prep=3 | TBD | |
| Phase 3 | prep=4 | TBD | |
| Phase 4 | prep=5 | TBD | |

### Kết quả eval (paper params: t_ho_prep=5, t_ho_exec=4)

| Speed | G_R (%) | PP rate (%) | RLF rate (%) |
|---|---|---|---|
| 30 | TBD | TBD | TBD |
| 50 | TBD | TBD | TBD |
| 70 | TBD | TBD | TBD |
| 90 | TBD | TBD | TBD |

---

## Tiêu chí đánh giá thành công

| Chỉ số | Thất bại | Chấp nhận | Tốt | Xuất sắc |
|---|---|---|---|---|
| G_R | < 50% | 50-90% | 90-99% | > 99% |
| PP rate | > 50% | 20-50% | 5-20% | < 5% |
| RLF rate | > 50% | 10-50% | 1-10% | < 1% |

---

## Phân tích sau thử nghiệm

*Cập nhật sau khi có kết quả*

### Nếu thất bại:
Các phương án còn lại:
1. **Tăng số steps**: 2M/phase thay vì 1M (10M tổng)
2. **Reward shaping mạnh hơn**: alpha=0.5 kết hợp với curriculum
3. **Không dùng PP termination**: chỉ dùng PP penalty trong reward, không terminate
4. **Longer BC**: 100 epochs hoặc curriculum BC (dạy prep=1 trước, rồi prep=5)

### Nếu thành công:
- Document hyperparameters chi tiết
- So sánh với paper Fig. 4
- Tạo plot tương tự paper
