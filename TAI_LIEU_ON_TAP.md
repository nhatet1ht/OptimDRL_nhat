# Tài liệu ôn tập: DRL cho tối ưu Handover trong 5G NR

> Mục đích: tài liệu **một file duy nhất** để đọc từ đầu đến cuối, hiểu tường tận từ lý
> thuyết → bài toán → code → cách test → toàn bộ hành trình 11 lần thử nghiệm và **vì sao**
> mỗi thử nghiệm được thiết kế như vậy. Viết để bạn có thể tự tin trả lời khi cô hỏi, không
> chỉ đọc thuộc kết quả mà hiểu được chuỗi suy luận đứng sau.
>
> Các file khác trong repo (`COMPREHENSIVE_REPORT.md`, `RESEARCH_NOTES.md`,
> `CONCEPTS_learning_process_tracking.md`, `EXPERIMENTS_v1-v4.md`, `RESULTS_comparison.md`)
> là nguồn dữ liệu gốc, chi tiết hơn, dùng để tra cứu số liệu chính xác. File này là bản
> **tổng hợp học tập**, sắp xếp lại theo trình tự sư phạm (lý thuyết trước, ứng dụng sau,
> hành trình suy luận xuyên suốt) thay vì trình tự thời gian viết báo cáo.
>
> Xem thêm **[`PIPELINES.md`](./PIPELINES.md)** — sơ đồ trực quan (Mermaid, tự render trên
> GitHub) cho data/training/evaluation pipeline, state machine HO/RLF/PP, và vòng lặp
> phương pháp luận của 11 thí nghiệm.

---

## Mục lục

- [Phần I — Lý thuyết nền tảng](#phần-i--lý-thuyết-nền-tảng)
- [Phần II — Bài toán Handover cụ thể](#phần-ii--bài-toán-handover-cụ-thể)
- [Phần III — Triển khai & cách test](#phần-iii--triển-khai--cách-test)
- [Phần IV — Hành trình thực nghiệm: chuỗi suy luận](#phần-iv--hành-trình-thực-nghiệm-chuỗi-suy-luận)
- [Phần V — Chuẩn bị trả lời câu hỏi](#phần-v--chuẩn-bị-trả-lời-câu-hỏi)

---

## Phần I — Lý thuyết nền tảng

### 1.1 Vì sao cần Handover thông minh?

Trong 5G NR, UE (thiết bị đầu cuối) di chuyển liên tục nên phải đổi trạm phục vụ (BS) khi
tín hiệu trạm hiện tại yếu đi. Chuẩn 3GPP dùng **Event A3**: một luật cố định — nếu RSRP
của BS lân cận vượt RSRP của BS đang phục vụ quá một ngưỡng (offset) trong một khoảng thời
gian TTT (Time-To-Trigger), thì HO được kích hoạt.

Vấn đề của luật cố định: ngưỡng/TTT không đổi theo tốc độ di chuyển hay điều kiện kênh, dễ
gây **Ping-Pong** (nhảy qua nhảy lại giữa 2 BS ở biên cell) và không tối ưu ở tốc độ cao.

**Ý tưởng của bài báo:** thay luật cố định bằng một **agent PPO** — agent quan sát SINR các
BS xung quanh và tự quyết định kết nối BS nào ở mỗi bước 10ms, học từ dữ liệu thay vì theo
luật fix cứng.

### 1.2 MDP — ngôn ngữ toán học của bài toán

Bài toán ra quyết định tuần tự được mô hình hoá là **Markov Decision Process (MDP)**, gồm:

| Ký hiệu | Ý nghĩa | Trong bài toán HO |
|---|---|---|
| $s_t$ | State tại bước $t$ | SINR các BS + BS đang serve + cờ ping-pong |
| $a_t$ | Action tại bước $t$ | Chọn BS muốn kết nối (0..4) |
| $r_t$ | Reward nhận được | Thưởng SINR tốt, phạt RLF/PP |
| $\pi_\theta(a\|s)$ | Policy — xác suất chọn action $a$ khi ở state $s$ | Mạng neural PPO |
| $G_t = \sum_k \gamma^k r_{t+k}$ | Return — tổng reward chiết khấu từ $t$ trở đi | |
| $V(s) = \mathbb{E}[G_t \| s_t=s]$ | Value function — kỳ vọng return nếu ở $s$ và theo policy | Đầu ra "critic" |
| $Q(s,a)$ | Kỳ vọng return nếu ở $s$, chọn $a$, rồi theo policy | |
| $A(s,a) = Q(s,a) - V(s)$ | **Advantage** — action này tốt hơn/kém trung bình bao nhiêu | |

Mục tiêu học: tìm $\theta$ tối đa hoá $J(\theta) = \mathbb{E}[G_0]$.

### 1.3 Policy Gradient — và vì sao nó là gốc rễ của mọi thất bại trong dự án này

Công thức cập nhật policy gradient (nền tảng của PPO):

$$\nabla_\theta J(\theta) = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t)\big]$$

**Đọc công thức này theo trực giác:** nếu action $a_t$ tốt hơn trung bình ($A>0$), tăng xác
suất chọn nó; nếu tệ hơn ($A<0$), giảm xác suất. Nhưng nếu $A(s,a) \approx 0$ với **mọi**
cặp $(s,a)$ — tức là reward không phân biệt được action nào tốt hơn action nào — thì cả biểu
thức $\approx 0$, và **policy không cập nhật gì cả, dù train bao nhiêu bước đi nữa.**

Đây gọi là **Credit Assignment Problem**: hệ thống không biết "quy công" (assign credit) cho
action nào vì reward không đủ phụ thuộc vào action. Toàn bộ 11 lần thí nghiệm trong dự án
này xoay quanh việc chẩn đoán và cố gắng giải quyết đúng vấn đề này (xem Phần IV).

Ghi nhớ chuỗi suy luận ngắn gọn để trình bày:

> **HO hiếm xảy ra → reward gần như hằng số bất kể action → advantage ≈ 0 → gradient ≈ 0
> → policy đứng yên mãi mãi (dù train 5 triệu bước).**

### 1.4 PPO (Proximal Policy Optimization)

PPO là thuật toán **on-policy**: thu thập một lô dữ liệu (rollout) bằng policy hiện tại, cập
nhật vài epoch, rồi **vứt bỏ** dữ liệu đó (khác với DQN off-policy dùng replay buffer).

Đóng góp chính của PPO: giới hạn (clip) mức độ thay đổi policy trong một lần cập nhật để
tránh update quá đà làm sụp policy:

$$L^{CLIP}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\,\hat A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\Big]$$

trong đó $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ là tỉ lệ xác
suất giữa policy mới/cũ.

**Điểm quan trọng cần nhớ:** PPO ổn định hơn REINFORCE thuần, nhưng **vẫn là một dạng policy
gradient** — nên nó **vẫn bị Credit Assignment Problem** như mục 1.3 nếu tín hiệu quá thưa.
Không có "phép màu" nào trong PPO tự giải quyết được vấn đề reward không phụ thuộc action.

Kiến trúc dùng trong bài: **Actor-Critic**, một mạng MLP dùng chung phần thân, tách 2 đầu ra:
- **Actor** (policy head): 5 neuron → softmax → $\pi(a|s)$
- **Critic** (value head): 1 neuron → $V(s)$

### 1.5 Entropy và `ent_coef` — công cụ kiểm soát khám phá (exploration)

Entropy của phân phối action: $H(\pi(\cdot|s)) = -\sum_a \pi(a|s)\log\pi(a|s)$.

Với $N=5$ action, phân phối **đều tuyệt đối** (uniform, mỗi action xác suất $1/5$) có entropy
tối đa: $H_{max} = \ln(5) = 1.6094$.

**Đây là con số quan trọng nhất để đọc log training trong dự án này.** SB3 report giá trị
`entropy_loss` $= -H(\pi)$. Bất cứ khi nào bạn thấy `entropy_loss` dính chặt ở **-1.6094**,
nghĩa là **policy hoàn toàn ngẫu nhiên đều, chưa học được gì** — bất kể train bao lâu.

`ent_coef` là hệ số cộng thêm bonus entropy vào loss để khuyến khích action đa dạng (khám
phá), **nhưng nó không tự tạo ra gradient signal mới** — nó chỉ làm chậm tốc độ policy sụp
thành deterministic. Thực nghiệm trong dự án (mục IV.0) đã chứng minh: hạ `ent_coef` từ 0.1
xuống 0.001 **không giúp thoát uniform**, vì vấn đề nằm ở $A(s,a)\approx 0$ (mục 1.3), không
nằm ở entropy regularization. Đây là một phát hiện quan trọng cần nói được rõ ràng.

### 1.6 Behavioral Cloning (BC) — "mồi" policy bằng supervised learning

BC là học có giám sát thuần tuý: học $\pi(a|s)$ trực tiếp bằng cross-entropy loss từ một
**oracle action** — trong dự án này, oracle = `argmax(sinr_norm)` (luôn chọn BS có SINR cao
nhất tại thời điểm hiện tại).

**Mục đích:** cho policy một điểm khởi đầu tốt (thay vì random init), để né việc PPO phải tự
mò ra vùng tốt trong không gian policy chỉ bằng reward quá thưa.

**Giới hạn quan trọng cần nhớ:** BC giải quyết được vấn đề *cold-start* (khởi đầu ngẫu
nhiên), nhưng **không giải quyết được Credit Assignment Problem trong giai đoạn PPO fine-tune
sau đó**. Đây là lý do mọi phương án BC+PPO trong dự án vẫn thất bại theo 1 trong 2 mẫu hình
"Never HO" hoặc "HO badly" (xem mục IV.2 – IV.4).

---

## Phần II — Bài toán Handover cụ thể

### 2.1 Môi trường mô phỏng

| Thông số | Giá trị |
|---|---|
| Khu vực | Trung tâm Karlsruhe (dữ liệu thật, không phải kênh giả lập lý thuyết) |
| Số BS (N) | **5** (BS0–BS4), macro BS |
| Tần số / băng thông | 2.1 GHz / 10 MHz |
| Bước thời gian $\Delta t$ | **10 ms** |
| Tốc độ UE train / test | 3–50 km/h / 30, 50, 70, 90 km/h |
| Simulator | Vienna 5G System Level Simulator + SUMO (mobility) |
| Dữ liệu | Pre-computed SINR/RSRP lưu file `.mat`, nạp lại khi train (**trace-driven**, không mô phỏng radio online) |

### 2.2 Ba khái niệm cốt lõi: HO, RLF, Ping-Pong

**Handover (HO)** không xảy ra tức thì — đi qua 2 giai đoạn:

```
Agent chọn BS_target ≠ BS hiện tại (pcell)
    │
    ▼
[HO Preparation] t_ho_prep = 5 bước = 50ms
    │ Phải CHỌN LIÊN TỤC CÙNG target trong 5 bước liên tiếp
    │ Đổi ý giữa chừng → RESET bộ đếm prep
    ▼
[HO Execution] t_ho_exec = 4 bước = 40ms  (UE mất kết nối tạm thời)
    ▼
Kết nối BS mới (nếu SINR > Q_in) HOẶC → HOF → RLF
```

→ Điều kiện thực tế: agent phải "cam kết" (commit) cùng 1 target trong 5 bước liên tiếp.
Với policy ngẫu nhiên đều (uniform):

$$P(\text{HO hoàn thành}) = \left(\frac{1}{5}\right)^{4} = 0.16\%\ \text{mỗi bước}$$

(Số mũ là 4 chứ không phải 5, vì bước đầu tiên đã "chọn" target — chỉ cần 4 bước tiếp theo
giữ nguyên lựa chọn đó.)

**Radio Link Failure (RLF)** — mất kết nối hoàn toàn, 2 con đường:
1. SINR < $Q_{out}=-8$dB liên tục 10 lần (N310) → khởi động timer T310 (100 bước) → nếu hết
   hạn mà chưa có 3 lần liên tiếp (N311) SINR > $Q_{in}=-6$dB → **RLF**.
2. T310 đang chạy khi HO prep hoàn tất, hoặc hết hạn giữa lúc HO exec → **HOF → RLF**.

**Ping-Pong (PP)** — HO sang BS mới rồi quay lại BS cũ trong vòng MTS = 1000ms (100 bước).

| Timer / Counter | Giá trị | Số bước ($\Delta t=10$ms) |
|---|:---:|:---:|
| $t_{ho\_prep}$ | 50 ms | **5** |
| $t_{ho\_exec}$ | 40 ms | **4** |
| MTS | 1000 ms | 100 |
| RLF recovery | 200 ms | 20 |
| T310 | 1000 ms | 100 |
| N310 / N311 | 10 / 3 lần | — |

**Điểm mấu chốt kỹ thuật:** tổng độ trễ HO = $5+4=9$ bước = 90ms. Trong 90ms đó UE đã di
chuyển và kênh truyền đã đổi khác — nghĩa là chọn "target tốt nhất tại thời điểm quyết định"
chưa chắc còn tốt "tại thời điểm 9 bước sau khi HO xong". Đây chính là nguồn gốc của mẫu hình
thất bại "HO badly" (mục IV.4).

### 2.3 State — Action — Reward

**State** ($2N+1=11$ chiều với $N=5$):

$$s_t = \underbrace{[s_{BS,0..4}]}_{\text{one-hot, ai đang serve}} \| \underbrace{[s_{SINR,0..4}]}_{\text{SINR normalize }[0,1]} \| \underbrace{[s_{PP}]}_{\text{đang trong MTS window?}}$$

Normalize: $s_{SINR,i} = \dfrac{\text{clip}(\text{SINR}_i,-10,10)+10}{20}$.

**Action:** $a_t \in \{0,1,2,3,4\}$ — chỉ số BS muốn kết nối. Chọn khác pcell → trigger HO
prep; chọn lại pcell giữa lúc đang prep → abort.

**Reward** — $r_t = r_{SINR} + r_{PP} + r_{RLF}$, với $C=0.95$:

| Thành phần | Điều kiện | Giá trị |
|---|---|:---:|
| $r_{SINR}$ | pcell kết nối, pcell = best BS | $s_{SINR}[\text{pcell}] + C$ |
| $r_{SINR}$ | pcell kết nối, pcell ≠ best BS | $s_{SINR}[\text{pcell}]$ |
| $r_{SINR}$ | Disconnected | 0 |
| $r_{PP}$ | Ping-Pong | $-C$ |
| $r_{RLF}$ | RLF | $-2C$ |
| $r_{RLF}$ | out-of-sync (SINR < $Q_{out}$) | $-C$ |

**Vì sao thiết kế reward như vậy tạo ra Credit Assignment Problem:** khi UE không HO (phần
lớn thời gian, vì P(HO)=0.16%), $r_t = s_{SINR}[\text{pcell}]$ — hoàn toàn **không phụ thuộc
action** đang chọn (vì pcell không đổi bất kể agent "muốn" chọn BS nào). Đây là chỗ nối trực
tiếp với công thức 1.3.

**Metric đánh giá — Relative Average Rate $\Gamma_R$:**

$$\Gamma_R = \frac{\bar R}{\bar R_{max}} \in [0,1], \quad R_t = B\log_2(1+\text{SINR}_{pcell,t}),\quad \bar R_{max} = B\,\mathbb{E}_t[\log_2(1+\max_i \text{SINR}_{i,t})]$$

Tức là: tốc độ thực tế đạt được so với tốc độ lý tưởng nếu UE luôn kết nối BS tốt nhất và
không có độ trễ HO. $\Gamma_R=99\%$ nghĩa là gần tối ưu; $\Gamma_R=40\%$ nghĩa là rất tệ.

### 2.4 Kiến trúc mạng & hyperparameter PPO (theo paper)

```
Input 11 → Linear(11→64)+ReLU → Linear(64→128)+ReLU → Linear(128→64)+ReLU
    ├── Actor: Linear(64→5) + Softmax
    └── Critic: Linear(64→1)
```

| Hyperparameter | Giá trị |
|---|---|
| Learning rate | $5\times10^{-5}$ (linear schedule → 0) |
| `ent_coef` | **0.1** |
| n_steps / batch_size / n_epochs | 2000 / 200 / 10 |
| $\gamma$ | 0.99 |
| Total timesteps | 5,000,000 |
| Device | **CPU** (MLP nhỏ chạy nhanh hơn CPU so với GPU do overhead chuyển tensor) |

**Two-phase training** (theo mô tả của paper):

| | Phase 1 | Phase 2 |
|---|:---:|:---:|
| `terminate_on_pp` | False | True |
| `terminate_on_rlf` | True | True |
| Steps | 2.5M | 2.5M |

Ý tưởng: Phase 1 cho agent tự do khám phá HO mà không sợ bị phạt PP; Phase 2 mới thêm áp lực
tránh PP thừa.

---

## Phần III — Triển khai & cách test

### 3.1 Cấu trúc code

```
HandoverOptimDRL/
├── run.py                          # CLI: train_ppo, validate_ppo, evaluate_all, ...
├── src/ho_optim_drl/
│   ├── config.py                   # TẤT CẢ hyperparameter (dataclass Config)
│   ├── dataloader.py                # Load RSRP/SINR từ .mat
│   ├── utils.py                     # clipnorm, speed filter, csv writer
│   └── gym_env/
│       ├── ho_env_ppo.py            # Gym wrapper + reward function _get_reward()
│       ├── ho_protocol_ppo.py       # State machine HO/RLF/PP — FILE QUAN TRỌNG NHẤT
│       ├── ho_env_3gpp.py / ho_protocol_3gpp.py   # Baseline 3GPP Event A3
├── scripts/
│   ├── train_ppo.py                 # Script gốc (có 2 bug lớn, xem mục 4.1)
│   ├── train_ppo_2phase.py / _curriculum.py / _reward_shaped.py / _imitation.py / _multiseed.py
│   ├── train_ppo_bc_curriculum.py   # V3
│   ├── train_ppo_bc_paper.py / _bc_highent.py / _bc_noterm.py / _bc_2m.py   # V4
│   ├── eval_checkpoints.py / eval_v4.py / evaluate_all.py
│   └── validate_ppo.py / validate_3gpp.py / plot_results.py
└── data/processed/                  # .mat SINR/RSRP theo tốc độ
```

Hàm cốt lõi trong `ho_protocol_ppo.py` (file quan trọng nhất — toàn bộ logic HO/RLF/PP nằm
ở đây):

| Hàm | Vai trò |
|---|---|
| `HOProcedurePPO.step()` | Entry point mỗi timestep |
| `_ho_preparation_handler()` | Trigger/reset HO prep |
| `_ho_state_machine()` | Chuyển trạng thái prep → exec → connect |
| `_rlf_detection()` | Logic N310/N311/T310 |
| `_pp_monitoring()` | MTS window |
| `_rlf_recovery()` | Reconnect sau RLF |

### 3.2 Cách chạy train / test

```bash
# Cài đặt (uv)
uv venv --python 3.12 && uv sync

# Train qua run.py (các lệnh đã đăng ký sẵn)
uv run python run.py train_ppo               # gốc — CẨN THẬN: có 2 bug, xem mục 4.1
uv run python run.py train_ppo_2phase        # V1
uv run python run.py train_ppo_curriculum    # V2 phương án 1
uv run python run.py train_ppo_reward_shaped # V2 phương án 2
uv run python run.py train_ppo_imitation     # V2 phương án 3
uv run python run.py train_ppo_multiseed     # V2 phương án 4
uv run python run.py train_ppo_bc_curriculum # V3

# Các script V4 (bc_paper/bc_highent/bc_noterm/bc_2m) chưa được đăng ký vào run.py,
# chạy trực tiếp bằng:
uv run python scripts/train_ppo_bc_paper.py
uv run python scripts/train_ppo_bc_highent.py
uv run python scripts/train_ppo_bc_noterm.py
uv run python scripts/train_ppo_bc_2m.py

# Đánh giá
uv run python run.py validate_3gpp     # baseline 3GPP
uv run python run.py validate_ppo      # PPO gốc (pre-trained)
uv run python run.py evaluate_all      # so sánh original vs 2-phase
uv run python run.py eval_checkpoints  # eval checkpoint giữa training (xem 3.4)
uv run python scripts/eval_v4.py       # eval các model V4
uv run python run.py plot_results      # vẽ lại hình như paper
```

**Cách tính $\Gamma_R$/PP/RLF khi eval:** load model đã train, chạy qua toàn bộ (hoặc một
tập con) 124 dataset test theo từng tốc độ (30/50/70/90 km/h), với `test_deterministic_actions
=True` (action = argmax xác suất, không sample) để đánh giá policy đã hội tụ thế nào, rồi
tính trung bình $\Gamma_R$, PP rate, RLF rate trên toàn bộ trajectory.

### 3.3 Ba lớp log — cách "track" quá trình học, không chỉ đọc kết quả cuối

```
Lớp 1 — TRAINING-TIME LOG (real-time, qua TensorBoard)
  Ghi mỗi rollout trong lúc model.learn(): entropy_loss, ep_rew_mean, ep_len_mean,
  approx_kl, clip_fraction, value_loss, explained_variance
  → tensorboard --logdir results/tensorboard

Lớp 2 — CHECKPOINT DIAGNOSTIC LOG (theo từng giai đoạn training)
  Lưu model riêng sau mỗi phase (model_phase0.zip, phase1.zip, ...)
  eval_checkpoints.py load lại, eval bằng ĐÚNG config của phase đó
  → trả lời "tại mốc này, policy đã học được hành vi gì?"

Lớp 3 — FINAL RESULT LOG (chỉ số cuối cùng)
  Γ_R / PP rate / RLF rate theo từng tốc độ, sau khi train xong toàn bộ
```

**Vì sao dùng cả 3 lớp quan trọng:** nếu chỉ nhìn Lớp 3, bạn chỉ biết "thất bại" chứ không
biết "thất bại kiểu gì" hay "thất bại ở giai đoạn nào". Đây chính là cách các thất bại trong
Phần IV được chẩn đoán ra nguyên nhân cụ thể, không phải đoán mò.

Bảng đọc chỉ số TensorBoard (thuộc lòng bảng này để đọc log mà không cần tra lại):

| Chỉ số | Ý nghĩa | Cách đọc |
|---|---|---|
| `rollout/ep_rew_mean` | Reward TB / episode | Tăng đều = đang học; dao động quanh 1 mức = chưa học gì mới |
| `rollout/ep_len_mean` | Độ dài TB episode | Ngắn bất thường = terminate sớm do RLF/PP liên tục |
| `train/entropy_loss` | $=-H(\pi)$ | Dính ở $-\ln(N)=-1.6094$ = uniform, chưa học gì |
| `train/approx_kl` | KL(policy mới, cũ) | Quá nhỏ liên tục = policy gần như không đổi |
| `train/explained_variance` | Critic dự đoán return tốt đến đâu (1=hoàn hảo, 0=đoán mò) | Gần 0/âm dai dẳng = củng cố giả thuyết reward quá thưa |
| `train/policy_gradient_loss` | Loss chính của actor | Gần 0 dai dẳng = liên hệ trực tiếp Credit Assignment Problem |
| `train/learning_rate` | LR thực tế (nếu có schedule) | Dùng để bắt bug lr_schedule (mục 4.1) |

### 3.4 Kỹ thuật "probe policy" — hỏi trực tiếp policy thay vì chỉ nhìn số liệu tổng hợp

```python
for best_bs in range(N):
    obs = make_obs_where_best_bs_is(best_bs)   # tạo state giả: biết trước BS nào tốt nhất
    probs = model.policy.get_distribution(obs).distribution.probs
    print(f"best_BS={best_bs}: action={action}, H={entropy:.4f}, probs={probs}")
```

Kỹ thuật này cho input cụ thể vào policy và xem nó trả lời gì — phát hiện được mẫu hình
"Never HO" (luôn giữ nguyên bất kể best BS là gì) mà chỉ nhìn $\Gamma_R$/PP/RLF tổng hợp sẽ
không thấy rõ ràng bằng.

---

## Phần IV — Hành trình thực nghiệm: chuỗi suy luận

Đây là phần quan trọng nhất để trả lời câu "phân tích như thế nào để nghĩ ra cách thử đó" —
mỗi mục đi theo khung: **Quan sát → Giả thuyết → Thiết kế (vì sao thiết kế vậy) → Kết quả →
Phân tích → Câu hỏi mở ra cho bước tiếp theo.**

### IV.0 — Bước khởi đầu: tái hiện baseline & phát hiện bug

**Việc làm đầu tiên** (trước khi train bất cứ gì từ đầu): chạy `validate_3gpp` và
`validate_ppo` với model pre-trained của paper để xác nhận **evaluation pipeline đúng**.

| Method | $\Gamma_R$ TB (%) | So với paper |
|---|:---:|:---:|
| 3GPP baseline (tái hiện) | 99.75 | ✅ khớp |
| PPO pre-trained (tái hiện) | 99.81 | ✅ khớp (sai số < 0.02%) |

→ Kết luận quan trọng: **pipeline evaluation đáng tin cậy**. Nếu sau này train ra $\Gamma_R$
thấp, đó là do quá trình training, không phải do đo sai.

**Trong lúc đọc code để chuẩn bị train**, phát hiện 2 bug nghiêm trọng trong script gốc
`train_ppo.py` — đây là lý do quan trọng khiến việc "chạy thẳng script gốc" sẽ luôn thất bại
về mặt thực hành, độc lập với vấn đề thuật toán:

1. **`SAVE_MODEL = False`** mặc định → train xong không lưu model nào cả.
2. **`model.save()` được gọi TRƯỚC `model.learn()`** → dù bật save, file lưu ra là model
   **chưa hề train** (chỉ có random init).
3. (phụ) Default `config.terminate_on_pp = True` → script gốc chạy nguyên 5M bước với config
   Phase 2, không hề implement Phase 1 như paper mô tả bằng lời.
4. (phụ) Bug `if self.s_pcell[-1] > 0` (đáng lẽ `>= 0`) khiến BS0 không bao giờ nhận SINR
   reward — nhưng vì BS index được xáo trộn (shuffle) mỗi episode nên bias này trung bình ra,
   không phải nguyên nhân chính của các thất bại sau này.

→ **Bài học phương pháp luận:** trước khi kết luận "thuật toán không hoạt động", luôn phải
loại trừ khả năng "code có bug" trước. Đây là lý do 4 bug trên được liệt kê và sửa **trước
khi** bắt đầu phân tích nguyên nhân thuật toán.

### IV.1 — V1: Giả thuyết đầu tiên (đơn giản nhất) và thất bại đầu tiên

**Giả thuyết đơn giản nhất có thể nghĩ ra:** có lẽ chỉ cần fix bug + implement đúng 2-phase
training như paper mô tả là đủ để tái hiện.

**Thử nghiệm 1A — giảm `t_ho_prep` 5→3 ở Phase 1:**

*Lý luận thiết kế:* $P(\text{HO}|t_{ho\_prep}=5)=0.16\%$ rất thấp; hạ xuống
$t_{ho\_prep}=3 \Rightarrow P=(1/5)^2=4\%$, tăng 25 lần. Kỳ vọng: HO xảy ra thường xuyên hơn
→ có nhiều tín hiệu học hơn.

*Kết quả:* $\Gamma_R \approx 10$–$29\%$, RLF rate $\approx 186$–$205\%$ — **tệ hơn cả không
làm gì**.

*Phân tích tại sao thất bại:* HO xảy ra nhiều hơn đúng như dự đoán, nhưng đến **BS ngẫu
nhiên** (vì policy vẫn gần như uniform) — không phải BS tốt. SINR sau HO thấp → HOF → RLF.
Tăng tần suất HO không giúp gì nếu HO đó là ngẫu nhiên; ngược lại tạo thêm nhiều lần thất bại
(RLF) → reward càng âm càng nhiều.

**Thử nghiệm 1B — giữ nguyên `t_ho_prep=5`, chỉ đổi `terminate_on_pp` giữa 2 phase (đúng như
paper mô tả):**

*Kết quả:* `entropy_loss = -1.61` (uniform tuyệt đối) xuyên suốt **toàn bộ** 5M bước, cả 2
phase.

*Phân tích:* `ep_len_mean ≈ 291` — RLF vẫn là nguyên nhân chính khiến episode kết thúc sớm,
dù Phase 1 không terminate vì PP. Việc "không phạt PP" không giải quyết được gốc rễ, vì HO
gần như không bao giờ xảy ra để mà bị PP.

**→ Kết luận rút ra sau V1 (bước ngoặt quan trọng):** cả 2 cách chỉnh nhẹ tham số đều thất
bại theo 2 kiểu khác nhau, nhưng cùng quy về một nguyên nhân — chính là **Credit Assignment
Problem** đã trình bày ở mục 1.3/2.3: với `t_ho_prep=5`, HO gần như không xảy ra, nên reward
gần như không phụ thuộc action, nên gradient ≈ 0. Đây là lúc phân tích chuyển từ "thử tham số"
sang "chẩn đoán nguyên nhân gốc rễ" một cách có hệ thống.

**Xác nhận thực nghiệm cho giả thuyết Credit Assignment** (chạy riêng 300K bước, thử 3 giá
trị `ent_coef`):

| `ent_coef` | Entropy cuối | Kết luận |
|:---:|:---:|---|
| 0.1 | 1.6094 (100% max) | Hoàn toàn uniform |
| 0.01 | 1.6093 | Gần như uniform |
| 0.001 | 1.6084 | Vẫn uniform |

→ Nếu là vấn đề exploration (entropy quá cao khiến agent "lười" hội tụ), hạ `ent_coef` phải
giúp được. Nó **không giúp** → xác nhận vấn đề nằm ở gradient/advantage, đúng như lý thuyết ở
mục 1.3, chứ không phải do exploration.

### IV.2 — V2: 4 hướng giải quyết, mỗi hướng nhắm vào một khía cạnh khác nhau của Credit Assignment

Sau khi xác định rõ root cause (P(HO) quá thấp → advantage ≈ 0), câu hỏi logic tiếp theo là:
**"làm sao để tạo ra tín hiệu học đủ mạnh mà vẫn cuối cùng hội tụ về đúng tham số bài
báo?"** Bốn hướng được thử **song song vì mỗi hướng tấn công vấn đề theo một cơ chế khác
nhau** — đây là cách thiết kế thí nghiệm để tối đa hoá thông tin thu được trong một vòng thử:

| # | Hướng | Cơ chế tấn công Credit Assignment |
|---|---|---|
| 1 | **Curriculum** $t_{ho\_prep}$: 1→3→5 | Tăng $P(\text{HO})$ **thật** bằng cách đổi độ khó bài toán tạm thời (không đổi reward) |
| 2 | **Reward shaping** | Thêm reward phụ thuộc action *ngay lập tức*, không cần đợi HO hoàn thành |
| 3 | **Imitation (BC + PPO)** | Bỏ qua việc học từ reward thưa — học trực tiếp từ oracle bằng supervised learning trước |
| 4 | **Multiple seeds** | Loại trừ giả thuyết "chỉ là do seed xui" |

**Hướng 1 — Curriculum 1→3→5** (`scripts/train_ppo_curriculum.py`):

*Lý luận:* nếu $t_{ho\_prep}=1$, HO xảy ra ngay khi agent đổi ý — bài toán biến thành gần
như **multi-armed bandit** (chọn action → thấy hệ quả ngay), loại bỏ hoàn toàn độ trễ. Đây
là môi trường lý tưởng để policy học "quy tắc chọn BS tốt" trước, rồi mới "kéo dài" khả năng
commit dần dần lên $t_{ho\_prep}=5$.

*Kết quả Phase 0 ($t_{ho\_prep}=1$):* **breakthrough đầu tiên trong toàn dự án** — entropy
giảm thật: $-1.61 \to -1.07$, ep_rew tăng đều $-1290 \to +3660$. Đây là lần đầu tiên quan sát
được policy *thực sự* học được gì.

*Nhưng Phase 1 ($t_{ho\_prep}=3$):* entropy **quay ngược lại** $-1.07\to-1.58$. Phase 2
($t_{ho\_prep}=5$): entropy về hẳn $-1.61$ — mất sạch những gì đã học.

*Phân tích:* mỗi lần $t_{ho\_prep}$ tăng, Credit Assignment Problem **xuất hiện trở lại** vì
$P(\text{HO})$ lại giảm. Policy không có cơ chế nào để "giữ lại" tri thức đã học khi tín hiệu
huấn luyện biến mất — nó chỉ đơn giản trôi dạt trở lại uniform vì không còn gradient để neo
lại. Eval cuối cùng (với $t_{ho\_prep}=5$): PP rate **97%** — cực cao, vì policy uniform +
không bị phạt PP trong lúc train đủ lâu để hình thành "phản xạ tránh PP".

**Hướng 2 — Reward Shaping** (`scripts/train_ppo_reward_shaped.py`):

*Lý luận:* thêm trực tiếp $r_{shaped} = \alpha \cdot s_{SINR}[a_t]$, $\alpha=0.1$ — thưởng
ngay lập tức nếu action trùng với BS có SINR cao, không cần đợi HO hoàn thành mới có reward.
Về lý thuyết, điều này tạo advantage khác 0 ngay từ bước đầu.

*Kết quả:* entropy vẫn dính ở $-1.61$ sau toàn bộ 5M bước.

*Phân tích:* $\alpha=0.1$ quá nhỏ so với reward SINR gốc (vốn không phụ thuộc action và có
độ lớn tương đương $\approx 0.5$–$1.4$) — gradient từ phần thưởng "thưởng đều" vẫn áp đảo,
nhấn chìm tín hiệu nhỏ từ $r_{shaped}$.

**Hướng 3 — Imitation Learning (BC + PPO)** (`scripts/train_ppo_imitation.py`):

*Lý luận:* nếu PPO không tự tìm ra được policy tốt từ reward thưa, hãy **cho nó biết trước**
policy tốt là gì bằng supervised learning, rồi mới fine-tune bằng PPO.

*Stage 1 — BC:* dataset 348,400 mẫu, oracle = argmax(SINR). Kết quả: **100% accuracy sau chỉ
2 epoch** — bài toán "chọn BS tốt nhất tại thời điểm hiện tại" là bài toán supervised rất dễ
(không có yếu tố thời gian/độ trễ).

*Stage 2 — PPO fine-tune* (`ent_coef=0.01`, thấp hơn để giữ cấu trúc deterministic từ BC):
Phase 1 (không phạt PP) → Phase 2 (phạt PP).

*Kết quả:* PP=0%, RLF=0% (rất tốt!) nhưng $\Gamma_R$ chỉ 38–50%.

*Phân tích — mẫu hình "Never HO" xuất hiện lần đầu:* policy học được "an toàn nhất là không
làm gì". Vì Phase 2 phạt PP cho **mọi** HO có rủi ro ping-pong, và agent (đã bắt đầu từ BC,
khá "tự tin") học ra rằng cách chắc chắn nhất để tránh phạt PP là **không bao giờ HO**. Đây
là một local optimum hợp lý về mặt reward-engineering nhưng không phải hành vi mong muốn.

**Hướng 4 — Multiple Seeds** (`scripts/train_ppo_multiseed.py`):

*Lý luận:* loại trừ khả năng model gốc của paper chỉ là seed may mắn.

*Kết quả:* cả 5 seed (0, 42, 123, 777, 1234) × 2M bước đều cho entropy $\approx 1.609$–$1.610$
— **tất cả đều uniform**.

*Kết luận:* Credit Assignment Problem là **hệ thống** (systematic), không phải hiện tượng
ngẫu nhiên phụ thuộc seed — loại bỏ hẳn một giả thuyết, thu hẹp không gian tìm kiếm nguyên
nhân.

**→ Tổng kết sau V2 — hai điểm tích cực được giữ lại để dùng cho V3:**
1. **Curriculum Phase 0 chứng minh được khái niệm hoạt động** — chỉ là transition giữa các
   phase quá đột ngột (nhảy thẳng 1→3→5, thiếu bước đệm).
2. **BC chứng minh học được policy tốt gần như tức thì** — chỉ là PPO fine-tune sau đó phá
   hỏng nó vì PP penalty quá gắt.

Hai phát hiện này trực tiếp định hướng thiết kế V3: **kết hợp BC (khởi đầu tốt) với curriculum
mượt hơn (nhiều bước đệm, giới thiệu PP từ từ)**.

### IV.3 — V3: Kết hợp có chủ đích dựa trên bài học của V2

**Thiết kế** (`scripts/train_ppo_bc_curriculum.py`) — mỗi lựa chọn dưới đây trực tiếp trả lời
một vấn đề cụ thể quan sát được ở V2:

| Vấn đề quan sát ở V2 | Giải pháp tương ứng trong V3 |
|---|---|
| Curriculum nhảy 1→3→5 quá đột ngột → mất tri thức | Chia nhỏ thành **5 phase**: prep = 1→2→3→4→5, mỗi phase 1M bước |
| PP penalty giết chết HO behavior ngay khi bật | PP chỉ được bật từ **Phase 3** (học HO trước, học tránh PP sau) |
| HO thất bại (RLF) làm nhiễu tín hiệu học ở early phase | `t_ho_exec` cũng giảm dần 1→4 — ít thời gian mất kết nối hơn ở đầu → ít HOF hơn |
| Cần giữ cấu trúc deterministic từ BC | `ent_coef=0.01` (thấp, giữ policy gần deterministic) |

*Kết quả Phase 0 (BC + $t_{ho\_prep}=1$, `ent_coef=0.01`):* PP=0%, RLF=0%, $\Gamma_R$
38–50% — **lại là "Never HO"**, giống hệt kết quả Imitation ở V2.

*Kết quả cuối (paper params, $t_{ho\_prep}=5$):* PP=0%, nhưng **RLF nhảy vọt lên
225–295%** — chuyển sang mẫu hình khác: **"HO badly"**.

*Phân tích — phát hiện quan trọng nhất của V3:* `ent_coef=0.01` (chọn để "giữ cấu trúc BC")
lại chính là nguyên nhân khiến entropy **sụp đổ quá nhanh** — policy trở nên gần như
deterministic quá sớm, trước khi kịp khám phá đủ để học cách tránh RLF khi $t_{ho\_prep}$
tăng dần. Paper dùng `ent_coef=0.1` — cao hơn 10 lần. Đây là **giả thuyết mới** cho vòng thử
tiếp theo: có lẽ chính lựa chọn hạ `ent_coef` để "bảo toàn BC" đã phản tác dụng.

### IV.4 — V4: Kiểm định giả thuyết `ent_coef` bằng 4 biến thể có kiểm soát

**Thiết kế thí nghiệm kiểu "kiểm soát biến" (controlled variation):** giữ nguyên khung
BC+Curriculum của V3, chỉ đổi **một yếu tố mỗi lần** để cô lập nguyên nhân:

| Biến thể | Thay đổi so với V3 | Câu hỏi muốn trả lời |
|---|---|---|
| `bc_paper` | Bỏ hẳn curriculum, BC + tham số paper trực tiếp, `ent_coef=0.1` | BC một mình có đủ không nếu dùng đúng `ent_coef` của paper? |
| `bc_highent` | Giữ curriculum V3, chỉ đổi `ent_coef: 0.01→0.1` | Giả thuyết `ent_coef` thấp là nguyên nhân — đúng không? |
| `bc_noterm` | Bỏ hết `terminate_on_rlf`/`terminate_on_pp`, chỉ dùng penalty trong reward | Việc "terminate episode" (thay vì chỉ phạt điểm) có phải là nguyên nhân ép policy về "Never HO" không? |
| `bc_2m` | Tăng 1M→2M bước/phase (10M tổng) | Có phải chỉ cần train lâu hơn là đủ? |

**Kết quả:**

| Biến thể | $\Gamma_R$ TB (%) | PP (%) | RLF (%) | Mẫu hình |
|---|:---:|:---:|:---:|---|
| bc_paper | 29.8 | 0 | 357 | HO badly |
| bc_highent | 21.4 | 11 | 255 | HO badly + PP xuất hiện lại |
| bc_noterm | **43.5** | 0 | 0 | Never HO |
| bc_2m | 29.9 | 0 | 363 | HO badly (giống hệt bc_paper) |

**Phân tích từng câu hỏi:**
- `bc_highent` bác bỏ một phần giả thuyết "chỉ cần tăng `ent_coef`": entropy cao hơn → agent
  explore nhiều HO hơn → **nhiều RLF hơn** (vì vẫn không giải quyết được vấn đề 9-bước
  look-ahead), và PP quay trở lại (11–24%) vì policy kém chắc chắn hơn. Tăng `ent_coef` không
  phải liều thuốc, nó chỉ đổi kiểu thất bại.
- `bc_noterm` xác nhận: vấn đề "Never HO" **không phải do cơ chế terminate episode** — dù bỏ
  hẳn terminate, chỉ dùng penalty liên tục, policy vẫn chọn "an toàn" giống hệt trước. Nghĩa
  là nguyên nhân sâu hơn: bản thân HO luôn có "cái giá" tức thời (mất kết nối khi exec) trong
  khi lợi ích chỉ đến sau — đây lại là một dạng khác của Credit Assignment (delayed reward).
- `bc_2m` cho kết quả **giống hệt** `bc_paper` dù train gấp đôi số bước → bác bỏ giả thuyết
  "chỉ cần train lâu hơn là đủ". Vấn đề không phải thiếu dữ liệu, mà là **cấu trúc bài toán**
  (độ trễ 9 bước giữa quyết định và hệ quả) mà MLP thuần không có cách nào "nhìn thấy trước".

**→ Kết luận cuối cùng sau 11 lần thử:** chỉ tồn tại **2 trạng thái hội tụ ổn định**, không
có trạng thái nào đạt gần target 99.8%:

```
┌─────────────────────┐        ┌──────────────────────┐
│     "Never HO"       │        │      "HO Badly"       │
│  Γ_R = 38–50%         │        │  Γ_R = 22–35%          │
│  PP = 0%, RLF = 0%    │        │  PP = 0–11%, RLF=250-490%│
│  Xuất hiện khi:       │        │  Xuất hiện khi:        │
│  HO luôn có "cái giá" │        │  ent_coef cao hơn      │
│  tức thời không bù đắp│        │  → HO xảy ra nhưng     │
│  được bởi lợi ích chỉ │        │    không dự đoán được  │
│  đến sau              │        │    BS tốt sau 9 bước   │
└─────────────────────┘        └──────────────────────┘
              🎯 TARGET 99.8% — chỉ pre-trained model đạt được
```

**Vì sao pre-trained model của paper thoát được cả 2 bẫy này — 4 giả thuyết theo xác suất
giảm dần** (không giả thuyết nào được xác nhận chắc chắn, vì paper không công bố đủ thông
tin):

1. **(Khả năng cao nhất)** Code có sẵn cấu hình **WandB Bayesian sweep** cho `ent_coef` ∈
   {0.001, 0.01, 0.1} × `rew_const` ∈ {0.8, 0.9, 1.0} — 9 tổ hợp. Nếu paper chạy sweep và
   chọn model tốt nhất trong 9 lần thử, xác suất tổng hợp có ít nhất 1 lần hội tụ cao hơn hẳn
   một lần chạy đơn lẻ.
2. Paper có thể train nhiều hơn 5,000,000 bước ghi trong config mặc định (không document số
   bước thực tế).
3. Seed may mắn kết hợp với trajectory cụ thể có nhiều đoạn "cell edge" rõ ràng, tạo positive
   feedback loop sớm.
4. (khả năng thấp) Paper chỉ chạy 1 lần và không kiểm tra reproducibility — vấn đề khá phổ
   biến trong các paper ML.

### IV.5 — Hướng đi tiếp theo (chưa thử, nhưng có cơ sở lý thuyết rõ)

Xuất phát trực tiếp từ chẩn đoán "MLP không có look-ahead cho độ trễ 9 bước" (mục 2.2 và
IV.4):

1. **Thêm SINR look-ahead vào observation:** $s_t^{new}$ gồm cả $\text{SINR}(t{+}1..t{+}9)$
   — vì dữ liệu SINR tương lai đã được pre-compute sẵn trong dataset, việc "cho agent nhìn
   trước" là khả thi trong lúc train (dù không thực tế khi triển khai online).
2. **Reward shaping hướng tương lai:** $r_{shaped} = \alpha \cdot s_{SINR}[a_t, t{+}9]$ —
   thưởng trực tiếp cho việc chọn BS sẽ tốt *sau* 9 bước, giải quyết đúng vấn đề credit
   assignment thay vì né tránh nó.
3. **Dùng kiến trúc có memory (LSTM/GRU)** thay MLP thuần, để mạng tự học biểu diễn thời gian
   thay vì chỉ ánh xạ $s_t \to a_t$ tại từng bước độc lập.

---

## Phần V — Chuẩn bị trả lời câu hỏi

### 5.1 Tóm tắt 1 phút (nếu cô chỉ hỏi 1 câu "vậy kết luận là gì?")

> "Em tái hiện thành công phần **evaluation** của paper — baseline 3GPP và model pre-trained
> đều khớp kết quả trong paper với sai số dưới 0.02%. Nhưng khi thử **train lại from scratch**
> theo đúng mô tả của paper (11 cấu hình khác nhau, gồm cả sửa 2 bug quan trọng trong code
> gốc), tất cả đều không đạt được hiệu năng của paper. Em xác định được nguyên nhân gốc rễ là
> **Credit Assignment Problem**: vì HO chỉ xảy ra khi agent chọn liên tục cùng 1 BS trong 5
> bước (~0.16% xác suất với policy ngẫu nhiên), reward gần như không phụ thuộc action trong
> phần lớn thời gian, nên gradient của PPO gần như bằng 0 và policy không học được gì. Em đã
> thử 4 hướng giải quyết khác nhau (curriculum, reward shaping, imitation learning, multiple
> seeds) và các biến thể của chúng, mỗi lần đều dùng kết quả trước để định hướng cho lần thử
> sau — nhưng chỉ hội tụ vào 2 trạng thái không mong muốn: 'không bao giờ HO' hoặc 'HO nhưng
> luôn thất bại'. Kết luận: paper thiếu thông tin quan trọng (seed, số bước thực tế, có sweep
> hyperparameter hay không) khiến việc tái hiện từ scratch không khả thi với thông tin hiện
> có — đây không phải do em làm sai, mà là giới hạn về documentation của paper gốc."

### 5.2 Câu hỏi khả năng cao sẽ được hỏi + gợi ý trả lời

**Q: "Tại sao không đơn giản là tăng `ent_coef` hoặc train lâu hơn?"**
→ Đã thử cả hai (mục IV.4: `bc_highent`, `bc_2m`) — tăng entropy chỉ đổi từ "Never HO" sang
"HO badly" (không cải thiện), train gấp đôi số bước cho kết quả gần như y hệt. Vấn đề nằm ở
**cấu trúc bài toán** (advantage ≈ 0 khi HO hiếm, và độ trễ 9-bước không quan sát được), không
phải thiếu compute.

**Q: "Sao biết chắc là Credit Assignment Problem, không phải bug nào khác trong code?"**
→ Ba dòng bằng chứng độc lập: (1) tính toán lý thuyết $P(\text{HO})=0.16\%$ khớp với quan sát
`ep_len_mean` dài bất thường mà HO gần như không xảy ra; (2) giảm `ent_coef` không giúp gì
(loại trừ giả thuyết "chỉ là exploration kém"); (3) khi cố tình hạ `t_ho_prep` xuống 1 (Phase
0 của curriculum), entropy **thực sự giảm** — chứng minh cơ chế policy gradient hoạt động
bình thường một khi có đủ tín hiệu, chỉ là bình thường nó không có đủ tín hiệu.

**Q: "Vậy có phải paper báo cáo kết quả không đúng sự thật?"**
→ Không kết luận vậy được — vì đã tái hiện **evaluation** khớp paper < 0.02% bằng chính
pre-trained model của họ. Vấn đề là paper **không công bố đủ chi tiết training** (seed, số
bước thực tế, có chạy sweep hay không) để người khác tái hiện được quá trình training — đây
là vấn đề reproducibility phổ biến trong nhiều paper ML, không phải paper gian dối.

**Q: "Nếu có thêm thời gian, em sẽ làm gì tiếp theo?"**
→ Trả lời bằng mục IV.5: thêm SINR look-ahead vào observation, hoặc reward shaping hướng
tương lai — cả hai đều nhắm thẳng vào nguyên nhân đã xác định (MLP không nhìn thấy được hệ
quả của quyết định sau 9 bước).

**Q: "Em track quá trình training như thế nào để biết nó thất bại ở đâu, không phải chỉ nhìn
kết quả cuối?"**
→ Trình bày mục 3.3 (3 lớp log) và demo `tensorboard --logdir results/tensorboard`, chỉ ra
đường entropy đi lên/xuống theo từng phase — đây là bằng chứng trực quan mạnh nhất.

### 5.3 Checklist chuẩn bị trước buổi trình bày

- [ ] Mở `tensorboard --logdir results/tensorboard`, chọn sẵn 1–2 run tiêu biểu để demo trực
      tiếp cách đọc `entropy_loss` / `ep_rew_mean` theo thời gian.
- [ ] Thuộc chuỗi suy luận Credit Assignment (mục 1.3) — có thể viết ra giấy không cần nhìn.
- [ ] Chuẩn bị sẵn bảng tổng hợp 11 kết quả (mục IV.4, dùng `RESULTS_comparison.md` để tra số
      chính xác nếu cần).
- [ ] Có thể giải thích được 2 bug trong code gốc (mục IV.0) — đây là điểm cho thấy đã đọc
      code kỹ, không chỉ chạy script có sẵn.
- [ ] Demo nhanh kỹ thuật "probe policy" (mục 3.4) nếu có thời gian — rất trực quan để minh
      hoạ "Never HO" là gì.
