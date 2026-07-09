# Sơ đồ Pipeline: DRL Handover Optimization

> Bổ sung trực quan cho `TAI_LIEU_ON_TAP.md`. Các sơ đồ dùng cú pháp **Mermaid** — GitHub tự
> render thành hình khi xem file này trên web (không cần cài gì thêm). Nếu xem bằng editor
> không hỗ trợ Mermaid (VD Notepad), đọc phần text thô cũng vẫn hiểu được luồng.

---

## 1. Kiến trúc tổng thể — Data → Environment → Agent

```mermaid
flowchart TD
    A["data/processed/*.mat<br/>RSRP + SINR theo tốc độ<br/>(30/50/70/90 km/h)"] --> B["dataloader.py<br/>load_preprocess_dataset()"]
    B --> C["HandoverEnvPPO<br/>ho_env_ppo.py<br/>(Gym wrapper)"]
    C --> D["HOProcedurePPO<br/>ho_protocol_ppo.py<br/>state machine HO / RLF / PP"]
    D --> E["_get_reward()<br/>r = r_SINR + r_PP + r_RLF"]
    C --> F["State s_t<br/>[s_BS | s_SINR | s_PP] — 11 chiều"]
    F --> G["PPO Agent (Stable-Baselines3)<br/>Actor-Critic MLP [64,128,64]"]
    G -- "action a_t (chọn BS)" --> C
    E --> G
    G --> H["model.zip<br/>(checkpoint)"]
    G --> I["TensorBoard log<br/>entropy_loss, ep_rew_mean, ..."]
```

**Đọc sơ đồ:** dữ liệu SINR/RSRP đã pre-compute (không mô phỏng radio online) được nạp vào
môi trường Gym; state machine trong `ho_protocol_ppo.py` quyết định HO/RLF/PP có xảy ra hay
không dựa trên action của agent; reward được tính rồi phản hồi ngược lại PPO để cập nhật
policy — đúng vòng lặp MDP kinh điển (state → action → reward → state mới).

---

## 2. Training pipeline (một vòng rollout–update của PPO)

```mermaid
flowchart TD
    A["env.reset()<br/>pcell = argmax(RSRP tại t=0)"] --> B["Thu thập rollout<br/>n_steps = 2000 bước"]
    B --> C["env.step(a_t)"]
    C --> D["HOProcedurePPO.step()<br/>cập nhật state machine"]
    D --> E["reward r_t"]
    E --> F{"Đủ 2000 bước?"}
    F -- Chưa --> B
    F -- Đủ --> G["PPO update<br/>n_epochs=10, batch_size=200<br/>tối đa hoá L^CLIP"]
    G --> H["Ghi log TensorBoard<br/>(entropy_loss, ep_rew_mean, approx_kl...)"]
    G --> I{"Đủ total_timesteps<br/>của phase này?"}
    I -- Chưa --> B
    I -- Đủ --> J["model.save('model_phaseX.zip')"]
    J --> K{"Còn phase tiếp theo?"}
    K -- Có --> L["Đổi config<br/>(t_ho_prep / terminate_on_pp / lr_schedule)<br/>tạo env mới, model.set_env()"]
    L --> B
    K -- Không --> M["Training xong<br/>→ sang Evaluation pipeline"]
```

**Điểm cần nhớ khi giải thích:** bước `L` (đổi config giữa các phase) chính là cơ chế
**curriculum** — mỗi lần đổi phase phải tạo lại environment vì `HOProcedurePPO.__init__` đọc
config tại thời điểm khởi tạo, không tự cập nhật nếu config đổi sau đó.

---

## 3. Multi-phase / Curriculum training theo thời gian (ví dụ V3 — BC + Gradual Curriculum)

```mermaid
flowchart LR
    BC["BC init<br/>oracle = argmax(SINR)<br/>30 epochs, 100% acc"] --> P0
    P0["Phase 0<br/>prep=1, exec=1<br/>no-RLF, no-PP<br/>1M steps"] --> P1
    P1["Phase 1<br/>prep=2, exec=1<br/>RLF, no-PP<br/>1M steps"] --> P2
    P2["Phase 2<br/>prep=3, exec=2<br/>RLF, no-PP<br/>1M steps"] --> P3
    P3["Phase 3<br/>prep=4, exec=3<br/>RLF, PP bật<br/>1M steps"] --> P4
    P4["Phase 4<br/>prep=5, exec=4<br/>paper params<br/>1M steps"] --> EVAL["Eval cuối<br/>(t_ho_prep=5)"]
```

Đây là dạng tổng quát của "curriculum learning": độ khó bài toán (`t_ho_prep`) và mức phạt
(`terminate_on_pp`) tăng dần từng bước, thay vì nhảy thẳng vào cấu hình khó nhất (paper params)
ngay từ đầu — vì nhảy thẳng khiến Credit Assignment Problem xuất hiện ngay lập tức (xem V1).

---

## 4. HO Protocol — State Machine (trái tim của `ho_protocol_ppo.py`)

```mermaid
stateDiagram-v2
    [*] --> Connected: reset() — pcell = argmax(RSRP)

    Connected --> Connected: action == pcell (giữ nguyên)
    Connected --> HO_Prep: action != pcell (bắt đầu prep)

    HO_Prep --> HO_Prep: action == target (đếm tiếp)
    HO_Prep --> Connected: đổi target giữa chừng (reset prep)
    HO_Prep --> HO_Exec: đủ t_ho_prep=5 bước liên tiếp cùng target
    HO_Prep --> RLF: T310 đang chạy khi prep xong (HOF)

    HO_Exec --> Connected: sau t_ho_exec=4 bước, SINR(target) > Q_in
    HO_Exec --> RLF: T310 hết hạn giữa lúc exec (HOF)

    Connected --> OutOfSync: SINR < Q_out, N310=10 lần liên tiếp
    OutOfSync --> Connected: SINR > Q_in, N311=3 lần liên tiếp (trước khi T310 hết hạn)
    OutOfSync --> RLF: T310 (100 bước) hết hạn mà chưa đủ N311

    RLF --> Disconnected: bắt đầu đếm t_rlfr = 20 bước
    Disconnected --> Connected: reconnect sau t_rlfr

    note right of HO_Exec
        Trong lúc Exec: UE mất kết nối
        tạm thời (disconnected),
        không nhận reward SINR
    end note

    note right of Connected
        Ping-Pong (PP): nếu HO quay
        lại BS cũ trong vòng MTS=100
        bước sau khi vừa HO xong
        → phạt -C, không đổi state
    end note
```

**Vì sao sơ đồ này quan trọng nhất khi giải thích cho cô:** toàn bộ 11 lần thí nghiệm thất
bại đều xoay quanh việc policy có vượt qua được nhánh `Connected → HO_Prep → HO_Exec →
Connected` hay không (Credit Assignment Problem — nhánh này hiếm khi được đi qua với policy
ngẫu nhiên), và nếu đi qua được thì có kẹt ở nhánh `→ RLF` hay không (vấn đề dự đoán 9-bước).

---

## 5. Evaluation pipeline (tính Γ_R / PP rate / RLF rate)

```mermaid
flowchart TD
    A["model.zip (đã train)"] --> B["PPO.load(model, env)"]
    B --> C["for speed in [30,50,70,90] km/h:"]
    C --> D["for dataset_idx in datasets_của_speed:"]
    D --> E["env.set_dataset_idx(dataset_idx)"]
    E --> F["rollout với<br/>test_deterministic_actions=True<br/>(action = argmax xác suất, không sample)"]
    F --> G["Đếm sự kiện: #HO, #PP, #RLF"]
    F --> H["Tính R_t = B·log2(1+SINR_pcell,t)<br/>mỗi bước"]
    H --> I["R̄ = trung bình R_t<br/>R̄_max = B·E[log2(1+max_i SINR_i,t)]"]
    I --> J["Γ_R = R̄ / R̄_max"]
    G --> K["PP rate = #PP / #HO<br/>RLF rate = #RLF / #HO"]
    J --> L["Bảng kết quả theo speed"]
    K --> L
    L --> M["So sánh với baseline 3GPP<br/>và target paper (Γ_R≈99.8%)"]
```

Script tương ứng: `validate_ppo.py` (model gốc), `evaluate_all.py` (so sánh nhiều model),
`eval_checkpoints.py` (eval checkpoint giữa chừng bằng đúng config lúc nó được train),
`eval_v4.py` (eval các biến thể V4).

---

## 6. Vòng lặp phương pháp luận — cách 11 thí nghiệm nối tiếp nhau

```mermaid
flowchart TD
    A["Quan sát log<br/>(entropy_loss dính -1.61,<br/>ep_len bất thường...)"] --> B["Giả thuyết nguyên nhân<br/>(vd: Credit Assignment)"]
    B --> C["Thiết kế thí nghiệm mới<br/>đổi CÓ KIỂM SOÁT 1 biến số<br/>so với lần trước"]
    C --> D["Train + ghi log 3 lớp<br/>(TensorBoard / checkpoint / final eval)"]
    D --> E{"Entropy có giảm thật?<br/>Γ_R có tăng không?"}
    E -- "Thất bại theo pattern cũ" --> F["Loại trừ giả thuyết<br/>(vd: không phải do seed)"]
    E -- "Thất bại theo pattern MỚI" --> G["Thu hẹp giả thuyết<br/>(vd: 'Never HO' → do PP penalty)"]
    E -- "Có tín hiệu học (breakthrough)" --> H["Giữ lại thành phần đã work<br/>(vd: Curriculum Phase 0, hoặc BC)"]
    F --> B
    G --> B
    H --> C
```

**Áp dụng thực tế vào 4 vòng V1→V4:**

| Vòng | Input (mang từ vòng trước) | Output (mang sang vòng sau) |
|---|---|---|
| V1 | Giả thuyết đơn giản: chỉ cần fix bug + đúng 2-phase | Xác nhận Credit Assignment Problem là root cause |
| V2 | 4 hướng tấn công root cause theo cơ chế khác nhau | Curriculum Phase 0 work + BC work (nhưng PPO fine-tune phá hỏng) |
| V3 | Kết hợp BC + curriculum mượt (5 phase thay vì 3) | Phát hiện `ent_coef=0.01` có thể là nguyên nhân entropy collapse sớm |
| V4 | 4 biến thể kiểm soát biến để test giả thuyết `ent_coef` | Bác bỏ "chỉ cần ent_coef cao hơn" và "chỉ cần train lâu hơn"; chốt lại 2 pattern thất bại cuối cùng |

---

*Xem `TAI_LIEU_ON_TAP.md` để có phần giải thích chi tiết bằng chữ đi kèm mỗi sơ đồ.*
