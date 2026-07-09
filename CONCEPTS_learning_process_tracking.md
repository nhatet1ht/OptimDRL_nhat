# Khái niệm cần nhớ: Lý thuyết RL + Cách track learning process

> File này để ôn lại các khái niệm đứng sau `COMPREHENSIVE_REPORT.md` / `RESULTS_AND_ANALYSIS.pdf` —
> không lặp lại nội dung domain (HO/RLF/PP) đã có trong report, mà tập trung vào phần
> **lý thuyết RL** và **quy trình track training** mà bạn đã dùng nhưng dễ quên vì nó nằm
> rải rác trong log files, không nằm trong code.

---

## 0. Bức tranh tổng thể: 3 lớp log bạn đã dùng

```
┌─────────────────────────────────────────────────────────────────┐
│ Lớp 1: TRAINING-TIME LOG (tự động, theo thời gian thực)         │
│   SB3 + TensorBoard ghi mỗi rollout trong lúc model.learn()      │
│   → entropy_loss, ep_rew_mean, ep_len_mean, approx_kl,           │
│     clip_fraction, loss, value_loss, explained_variance          │
│   File: results/tensorboard/<sweep_name>/<run_name>/             │
├─────────────────────────────────────────────────────────────────┤
│ Lớp 2: CHECKPOINT DIAGNOSTIC LOG (theo từng giai đoạn training)  │
│   Lưu model riêng sau mỗi phase (model_phase0.zip, phase1...)    │
│   → eval_checkpoints.py load lại, eval bằng đúng config phase đó │
│   → trả lời: "tại mốc này, policy đã học được hành vi gì?"       │
├─────────────────────────────────────────────────────────────────┤
│ Lớp 3: FINAL RESULT LOG (chỉ kết quả cuối, sau khi train xong)   │
│   → G_R / PP rate / RLF rate theo từng tốc độ (30/50/70/90 km/h) │
│   → RESULTS_comparison.md                                        │
└─────────────────────────────────────────────────────────────────┘
```

Report của bạn mạnh vì nó dùng **cả 3 lớp** để chẩn đoán *nguyên nhân* thất bại, chứ
không chỉ báo cáo *hiện tượng* thất bại (G_R thấp). Đây là điểm khác biệt với cách
"dò code + lý thuyết" mà Lâm/cô làm.

---

## 1. Khái niệm RL nền tảng cần nhớ

### 1.1 MDP và các thành phần
- **State** $s_t$, **Action** $a_t$, **Reward** $r_t$, **Policy** $\pi_\theta(a|s)$
- **Return** $G_t = \sum_k \gamma^k r_{t+k}$ — tổng reward chiết khấu từ thời điểm $t$
- **Value function** $V(s)$ — kỳ vọng return nếu ở state $s$ và theo policy hiện tại
- **Advantage** $A(s,a) = Q(s,a) - V(s)$ — action này tốt hơn/kém hơn trung bình bao nhiêu

### 1.2 Policy Gradient — vì sao "credit assignment problem" là gốc rễ
Policy gradient cập nhật theo hướng:
$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t)]$$

**Điểm mấu chốt:** nếu $A(s,a) \approx 0$ với mọi $(s,a)$ — tức là reward không phân biệt
được action nào tốt hơn action nào — thì gradient $\approx 0$ và **policy không update dù
train bao nhiêu steps đi nữa**. Đây chính là cơ chế đứng sau phát hiện lớn nhất của report
(mục 7 trong `COMPREHENSIVE_REPORT.md`): với $t_{ho\_prep}=5$, P(HO thành công) chỉ 0.16%/step
→ hầu hết thời gian reward không phụ thuộc action → advantage ≈ 0 → **credit assignment problem**.

> Ghi nhớ cách suy luận: **HO hiếm xảy ra → reward gần như hằng số theo action → advantage
> triệt tiêu → gradient triệt tiêu → policy stuck**. Đây là chuỗi lý luận bạn cần trình bày
> lại được mà không cần nhìn báo cáo.

### 1.3 PPO (Proximal Policy Optimization)
- Thuật toán **on-policy**: thu thập rollout bằng policy hiện tại → cập nhật → vứt bỏ dữ liệu cũ
- Objective bị **clip** để tránh update quá xa policy cũ trong 1 bước:
$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$
  trong đó $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$
- **Vì sao PPO phù hợp bài toán này:** ổn định hơn REINFORCE thuần, nhưng vẫn là policy
  gradient nên **vẫn bị credit assignment problem** nếu tín hiệu quá thưa.

### 1.4 Entropy và ent_coef — công cụ kiểm soát exploration
- Entropy của phân phối action: $H(\pi(\cdot|s)) = -\sum_a \pi(a|s)\log\pi(a|s)$
- Với 5 action (N=5 BS) uniform: $H_{max} = \ln(5) = 1.6094$ — **con số này chính là "entropy
  = -1.61" (SB3 report `entropy_loss` = -H) mà report nhắc đi nhắc lại**. Bất cứ khi nào bạn
  thấy `entropy_loss` dính chặt ở -1.6094, nghĩa là **policy hoàn toàn ngẫu nhiên đều, chưa
  học được gì**, bất kể ent_coef bao nhiêu.
- `ent_coef` (config: `ent_coef=0.1`, xem `src/ho_optim_drl/config.py:57`) là hệ số cộng thêm
  entropy bonus vào loss để khuyến khích exploration — **nhưng nó không tạo ra gradient signal
  mới**, chỉ làm chậm việc policy sụp thành deterministic. Report đã chứng minh thực nghiệm:
  giảm ent_coef từ 0.1 → 0.001 **không giúp thoát uniform** vì vấn đề nằm ở advantage ≈ 0,
  không nằm ở entropy regularization.

### 1.5 Behavioral Cloning (BC) — vì sao dùng để "mồi" policy
- BC là supervised learning: học $\pi(a|s)$ trực tiếp từ oracle action (ở đây là
  `argmax(sinr_norm)` — luôn chọn BS SINR cao nhất) bằng cross-entropy loss.
- Mục đích: cho policy một **điểm khởi đầu tốt** (không phải random init) trước khi PPO
  fine-tune, để né vấn đề "reward quá thưa nên PPO không tự tìm ra được vùng tốt".
- Bài học quan trọng bạn rút ra: BC giải quyết được cold-start, nhưng **không giải quyết
  được credit assignment trong giai đoạn PPO fine-tune sau đó** — đó là lý do các phương án
  BC+PPO vẫn thất bại theo 2 pattern "Never HO" / "HO badly" (mục 13 trong COMPREHENSIVE_REPORT.md).

---

## 2. Glossary các chỉ số TensorBoard/SB3 — đây là phần cốt lõi của "track learning process"

Đây là bảng bạn cần thuộc để đọc log lúc train (Lớp 1) mà không cần tra lại:

| Chỉ số SB3 | Ý nghĩa | Cách đọc để chẩn đoán |
|---|---|---|
| `rollout/ep_rew_mean` | Reward trung bình mỗi episode | Tăng đều = đang học; đứng yên/dao động quanh 1 mức = chưa học được gì mới |
| `rollout/ep_len_mean` | Độ dài trung bình episode (số steps trước khi terminate) | Ngắn bất thường = terminate sớm do RLF/PP liên tục → episode không đủ dài để tích lũy tín hiệu học |
| `train/entropy_loss` | = $-H(\pi)$, càng âm càng "chắc chắn" (ít ngẫu nhiên) | Dính ở $-\ln(N)$ (VD -1.6094 với N=5) = uniform, chưa học; giảm dần (VD -1.61→-1.07) = đang phân hoá hành vi |
| `train/approx_kl` | KL divergence ước lượng giữa policy mới và cũ sau mỗi update | Quá lớn = update quá mạnh (rủi ro mất ổn định); quá nhỏ liên tục = policy gần như không đổi |
| `train/clip_fraction` | Tỷ lệ samples bị clip trong PPO objective | Cao = nhiều update bị "kìm lại" bởi clip, có thể cần giảm learning rate |
| `train/explained_variance` | Value function dự đoán return tốt đến đâu (1 = hoàn hảo, 0 = như đoán mò, âm = tệ hơn đoán mò) | Gần 0 hoặc âm dai dẳng = critic cũng không học được gì → củng cố thêm giả thuyết reward quá thưa |
| `train/value_loss` | MSE loss của critic | Không giảm = critic không cải thiện dự đoán return |
| `train/policy_gradient_loss` | Loss objective chính của actor | Gần 0 dai dẳng = gradient signal yếu (liên hệ trực tiếp credit assignment problem) |
| `train/learning_rate` | LR thực tế tại thời điểm đó (nếu dùng schedule) | Dùng để bắt bug như bug lr_schedule + reset_num_timesteps=False đã tìm thấy (mục 5.4 COMPREHENSIVE_REPORT.md) — LR tụt về 0 sớm hơn dự kiến |

**Cách mở log của bạn:**
```bash
tensorboard --logdir results/tensorboard
```
rồi so sánh nhiều run cùng lúc (TensorBoard tự overlay các run trong cùng thư mục con) —
đây là cách bạn phát hiện ra sự khác biệt giữa các phase (VD entropy Phase 0 giảm nhưng
Phase 1/2 lại tăng trở lại — mục 9.1 COMPREHENSIVE_REPORT.md).

---

## 3. Quy trình "track learning process" — tái tạo lại từng bước

Đây là quy trình cụ thể bạn đã làm, viết lại thành checklist để dùng lại / dạy Lâm:

### Bước 1 — Thiết lập log ngay từ lúc khai báo model
```python
model = PPO(
    "MlpPolicy", env,
    ent_coef=config.ent_coef,
    tensorboard_log=tb_log,      # bắt buộc để có Lớp 1
    verbose=1,                    # in ra console mỗi rollout
    ...
)
```
Nếu quên `tensorboard_log`, bạn sẽ **chỉ có Lớp 3** (kết quả cuối) — không thể chẩn đoán
được gì khi training thất bại, chỉ biết "nó thất bại" chứ không biết "thất bại kiểu gì".

### Bước 2 — Lưu checkpoint tại các mốc có ý nghĩa (không chỉ lưu model cuối)
```python
model.learn(total_timesteps=STEPS_PER_PHASE, ...)
model.save(f"model_phase{phase_idx}.zip")   # lưu SAU khi học xong 1 phase
```
Đây là điều **script gốc của paper không làm đúng** (bug ở mục 15.2 COMPREHENSIVE_REPORT.md —
save trước khi train). Việc bạn chủ động lưu theo từng phase là điều kiện tiên quyết để có Lớp 2.

### Bước 3 — Eval riêng từng checkpoint bằng đúng config lúc nó được train
Xem `scripts/eval_checkpoints.py` — ý tưởng cốt lõi:
```python
config.t_ho_prep = 1        # dùng config của PHASE ĐÓ, không phải config cuối cùng
config.terminate_on_rlf = False
model = PPO.load("model_phase0.zip", env=env)
# eval, tính G_R / PP rate / RLF rate cho riêng checkpoint này
```
**Vì sao quan trọng:** nếu eval checkpoint phase 0 bằng config cuối (t_ho_prep=5), bạn sẽ
không biết được model *tại thời điểm đó* thực sự đã học được gì, vì môi trường test khác
môi trường nó được train. Đây là lỗi dễ mắc nhất khi debug RL.

### Bước 4 — Đọc số liệu theo trình tự thời gian, không chỉ theo giá trị cuối
Cách bạn viết trong report (VD mục 9.1):
```
40K steps:  entropy = -1.29
Phase 0:    entropy: -1.61 → -1.07
Phase 1:    entropy: -1.07 → -1.58   (REGRESSION)
Phase 2:    entropy: -1.58 → -1.61   (về lại UNIFORM)
```
Đây là kỹ năng chính: **đọc quỹ đạo (trajectory) của chỉ số qua thời gian**, không chỉ đọc
điểm cuối. Một model có G_R thấp nhưng entropy *đã từng* giảm rồi tăng lại trở nên rất
thông tin — nó cho biết vấn đề nằm ở *transition giữa các phase*, không nằm ở bản thân ý
tưởng ban đầu.

### Bước 5 — Đối chiếu hành vi cụ thể (probe policy), không chỉ nhìn số liệu tổng hợp
Kỹ thuật nhỏ đã dùng (mục 9.3 COMPREHENSIVE_REPORT.md):
```python
for best_bs in range(N):
    obs = make_obs_where_best_bs_is(best_bs)
    probs = model.policy.get_distribution(obs).distribution.probs
    print(f"best_BS={best_bs}: action={action}, H={entropy:.4f}, probs={probs}")
```
Đây là cách "hỏi trực tiếp" policy: cho nó 1 state cụ thể (biết trước BS nào tốt nhất),
xem nó trả lời action gì. Cách này phát hiện ra pattern "Never HO" (luôn chọn giữ nguyên
BS hiện tại bất kể best BS là gì) mà chỉ nhìn G_R/PP/RLF tổng hợp sẽ không thấy rõ.

---

## 4. Hai pattern chẩn đoán bạn cần nhớ để giải thích nhanh cho người khác

| Pattern | Dấu hiệu số liệu | Nguyên nhân gốc |
|---|---|---|
| **Uniform / chưa học gì** | `entropy_loss` dính ở $-\ln(N)$, `ep_rew_mean` không đổi, `explained_variance` ≈ 0 | Credit assignment: reward không phụ thuộc action đủ thường xuyên |
| **"Never HO"** | PP=0%, RLF=0%, nhưng G_R thấp (38-50%) | Mọi HO đều có rủi ro bị penalty (PP hoặc disconnect) → policy học "an toàn nhất là không làm gì" |
| **"HO badly"** | RLF rate rất cao (250-490%) | Policy chọn HO nhưng không dự đoán được BS tốt nhất sau độ trễ 9 timesteps (5 prep + 4 exec) |

---

## 5. Việc cần chuẩn bị cho buổi thảo luận với cô/Lâm

- [ ] Mở `tensorboard --logdir results/tensorboard` sống, chỉ 1-2 run tiêu biểu (VD `ppo_curriculum` vs `ppo_bc_curriculum`), demo cách đọc `entropy_loss` / `ep_rew_mean` theo thời gian
- [ ] Giải thích chuỗi suy luận credit assignment: P(HO)=0.16% → advantage≈0 → gradient≈0 (mục 1.2 ở trên)
- [ ] Cho xem `scripts/eval_checkpoints.py` như ví dụ cụ thể của Lớp 2 (đánh giá checkpoint giữa chừng)
- [ ] Demo nhanh kỹ thuật "probe policy" (mục 3, Bước 5) — cho input cụ thể, xem action probs
- [ ] Nhấn mạnh: 2 pattern "Never HO" / "HO badly" không phải đoán mò — suy ra được từ đọc cả 3 lớp log cùng lúc, không chỉ từ G_R cuối cùng

---

*Tài liệu này bổ sung cho `COMPREHENSIVE_REPORT.md` (chi tiết domain + kết quả) và
`RESEARCH_NOTES.md` / `NOTES_paper_reproduction.md` (ghi chú gốc lúc mới bắt đầu tái hiện paper).*
