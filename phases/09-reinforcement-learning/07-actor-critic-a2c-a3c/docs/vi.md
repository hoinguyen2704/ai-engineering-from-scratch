# Actor-Critic — A2C và A3C

> REINFORCE bị nhiễu. Hãy thêm một critic để học `V̂(s)`, trừ nó ra khỏi return, và bạn sẽ nhận được một advantage có cùng kỳ vọng nhưng phương sai thấp hơn nhiều. Đó chính là actor-critic. A2C chạy đồng bộ; A3C chạy trên các luồng (threads). Cả hai đều là mô hình tư duy cho mọi phương pháp deep-RL hiện đại.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Time:** ~75 phút

## Vấn đề

Vanilla REINFORCE hoạt động được, nhưng phương sai của nó rất tệ. Monte Carlo returns `G_t` có thể dao động hơn 10 lần giữa các tập (episodes). Việc nhân nhiễu đó với `∇ log π` và lấy trung bình tạo ra một bộ ước lượng gradient cần hàng ngàn tập để di chuyển policy một khoảng cách mà bạn có thể đạt được với ít cập nhật DQN hơn nhiều.

Phương sai đến từ việc sử dụng raw returns. Nếu bạn trừ đi một baseline `b(s_t)` — bất kỳ hàm nào của state, bao gồm cả một giá trị đã học — kỳ vọng sẽ không thay đổi và phương sai sẽ giảm. Baseline khả thi tốt nhất là `V̂(s_t)`. Bây giờ, đại lượng nhân với `∇ log π` chính là *advantage*:

`A(s, a) = G - V̂(s)`

Một hành động được coi là tốt nếu nó tạo ra return cao hơn trung bình; và tệ nếu thấp hơn. REINFORCE với một critic đã học chính là *actor-critic*. Critic cung cấp cho actor một "người thầy" có phương sai thấp. Đây là nền tảng của mọi phương pháp deep-policy sau năm 2015 (A2C, A3C, PPO, SAC, IMPALA).

## Khái niệm

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**Hai mạng, một hàm loss chung:**

- **Actor** `π_θ(a | s)`: policy. Được lấy mẫu để thực hiện hành động. Được huấn luyện bằng policy gradient.
- **Critic** `V_φ(s)`: ước tính return kỳ vọng từ state. Được huấn luyện để tối thiểu hóa `(V_φ(s) - target)²`.

**Advantage.** Hai dạng tiêu chuẩn:

- *MC advantage:* `A_t = G_t - V_φ(s_t)`. Không chệch (unbiased), phương sai cao hơn.
- *TD advantage:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. Có chệch (sử dụng `V_φ`), phương sai thấp hơn nhiều. Còn được gọi là *TD residual* `δ_t`.

**n-step advantage.** Nội suy giữa hai dạng trên:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` là TD thuần túy. `n = ∞` là MC. Hầu hết các triển khai sử dụng `n = 5` cho Atari, `n = 2048` cho PPO trên MuJoCo.

**Generalized Advantage Estimation (GAE).** Schulman và cộng sự (2016) đã đề xuất một trung bình trọng số theo hàm mũ trên tất cả các n-step advantages:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

với `λ ∈ [0, 1]`. `λ = 0` là TD (phương sai thấp, bias cao). `λ = 1` là MC (phương sai cao, không chệch). `λ = 0.95` là mặc định của năm 2026 — hãy tinh chỉnh cho đến khi đạt được mức bias/variance mong muốn.

**A2C: synchronous advantage actor-critic.** Thu thập `T` bước trên `N` môi trường song song. Tính toán advantage cho mỗi bước. Cập nhật actor và critic trên batch kết hợp. Lặp lại. Đây là phiên bản đơn giản và dễ mở rộng hơn của A3C.

**A3C: asynchronous advantage actor-critic.** Mnih và cộng sự (2016). Tạo ra `N` luồng worker, mỗi luồng chạy một môi trường. Mỗi worker tính toán gradient cục bộ trên rollout của riêng nó, sau đó cập nhật không đồng bộ vào một tham số chung (parameter server). Không cần replay buffer — các worker tự khử tương quan bằng cách chạy các quỹ đạo khác nhau. A3C đã chứng minh rằng bạn có thể huấn luyện trên CPU ở quy mô lớn. Vào năm 2026, A2C dựa trên GPU (các môi trường song song theo batch) chiếm ưu thế vì GPU cần các batch lớn.

**Hàm loss kết hợp.**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

Ba thành phần: policy-gradient loss, value regression, entropy bonus. `c_v ~ 0.5`, `c_e ~ 0.01` là các điểm khởi đầu kinh điển.

```figure
actor-critic
```

## Xây dựng

### Bước 1: một critic

Linear critic `V_φ(s) = w · features(s)` được cập nhật bằng MSE:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

Trên một môi trường dạng bảng (tabular env), critic hội tụ trong vài trăm tập. Trên Atari, hãy thay thế linear critic bằng một CNN trunk chung + value head.

### Bước 2: n-step advantage

Với một rollout có độ dài `T` và một giá trị cuối cùng được bootstrap `V(s_T)`:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` là mục tiêu của critic. `advantages` là đại lượng nhân với `∇ log π`.

### Bước 3: cập nhật kết hợp

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy, một rollout cho mỗi lần cập nhật, tốc độ học (learning rates) riêng biệt cho actor và critic.

### Bước 4: song song hóa (A3C vs A2C)

- **A3C:** khởi chạy `N` luồng. Mỗi luồng chạy môi trường riêng và thực hiện forward pass riêng. Định kỳ đẩy các cập nhật gradient lên master chung. Không cần khóa (lock) trên master — các xung đột (races) là chấp nhận được, chúng chỉ thêm nhiễu.
- **A2C:** chạy `N` instance môi trường trong một tiến trình duy nhất, xếp chồng các quan sát thành một batch `[N, obs_dim]`, thực hiện forward pass và backward pass theo batch. Tận dụng GPU tốt hơn, có tính xác định, dễ suy luận hơn. Đây là mặc định vào năm 2026.

Mã nguồn mẫu của chúng ta là đơn luồng để dễ hiểu; việc viết lại thành A2C theo batch chỉ mất ba dòng numpy.

## Các cạm bẫy

- **Critic bias trước khi có actor gradient.** Nếu critic là ngẫu nhiên, baseline của nó không có thông tin và bạn đang huấn luyện trên nhiễu thuần túy. Hãy khởi động (warm up) critic trong vài trăm bước trước khi bật policy gradient, hoặc sử dụng tốc độ học actor chậm.
- **Advantage normalization.** Chuẩn hóa các advantage về zero-mean/unit-std cho mỗi batch. Giúp ổn định huấn luyện đáng kể với chi phí gần như bằng không.
- **Shared trunk.** Sử dụng bộ trích xuất đặc trưng chung cho actor và critic trên đầu vào là hình ảnh. Các head riêng biệt. Các đặc trưng chung được hưởng lợi từ cả hai hàm loss.
- **On-policy contract.** A2C tái sử dụng dữ liệu cho đúng một lần cập nhật. Nếu nhiều hơn, gradient của bạn sẽ bị chệch (hiệu chỉnh importance-sampling là thứ mà PPO bổ sung).
- **Entropy collapse.** Nếu không có `c_e > 0`, policy sẽ trở nên gần như xác định sau vài trăm lần cập nhật và ngừng khám phá.
- **Reward scale.** Độ lớn của advantage phụ thuộc vào thang đo phần thưởng. Hãy chuẩn hóa phần thưởng (ví dụ: chia cho running-std) để có độ lớn gradient nhất quán giữa các tác vụ.

## Sử dụng

A2C/A3C hiếm khi là lựa chọn cuối cùng vào năm 2026 nhưng chúng là kiến trúc mà mọi phương pháp sau này đều tinh chỉnh:

| Phương pháp | Liên quan đến A2C |
|--------|----------------|
| PPO | A2C + clipped importance ratio cho các cập nhật nhiều epoch |
| IMPALA | A3C + hiệu chỉnh off-policy V-trace |
| SAC (Phase 9 · 07) | Off-policy A2C với soft-value critic (bài học tiếp theo) |
| GRPO (Phase 9 · 12) | A2C không có critic — group-relative advantage |
| DPO | A2C được thu gọn thành hàm loss xếp hạng ưu tiên, không lấy mẫu |
| AlphaStar / OpenAI Five | A2C với league training + imitation pre-training |

Nếu bạn thấy "advantage" trong một bài báo năm 2026, hãy nghĩ đến actor-critic.

## Triển khai

Lưu dưới dạng `outputs/skill-actor-critic-trainer.md`:

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## Bài tập

1. **Dễ.** Huấn luyện actor-critic với MC advantage (`G_t - V(s_t)`) trên 4×4 GridWorld. So sánh hiệu quả lấy mẫu với REINFORCE-with-running-mean-baseline từ Bài 06.
2. **Trung bình.** Chuyển sang TD-residual advantage (`r + γ V(s') - V(s)`). Đo phương sai của các batch advantage. Nó giảm bao nhiêu?
3. **Khó.** Triển khai GAE(λ). Quét `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`. Vẽ biểu đồ return cuối cùng so với hiệu quả lấy mẫu. Đâu là điểm cân bằng bias/variance cho tác vụ này?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Actor | "Mạng policy" | `π_θ(a\|s)`, được cập nhật bởi policy gradient. |
| Critic | "Mạng giá trị" | `V_φ(s)`, được cập nhật bằng MSE regression tới returns / TD targets. |
| Advantage | "Tốt hơn trung bình bao nhiêu" | `A(s, a) = Q(s, a) - V(s)` hoặc các bộ ước lượng của nó. Hệ số nhân cho `∇ log π`. |
| TD residual | "δ" | `δ_t = r + γ V(s') - V(s)`; ước lượng advantage một bước. |
| GAE | "Núm xoay nội suy" | Tổng trọng số theo hàm mũ của n-step advantages, tham số hóa bởi `λ`. |
| A2C | "Actor-critic đồng bộ" | Batch qua các môi trường; một bước gradient cho mỗi rollout. |
| A3C | "Actor-critic không đồng bộ" | Các luồng worker đẩy gradient lên server tham số chung. Bài báo gốc; ít phổ biến vào năm 2026. |
| Bootstrap | "Sử dụng V tại chân trời" | Cắt ngắn rollout, thêm `γ^n V(s_{t+n})` để đóng tổng. |

## Đọc thêm

- [Mnih và cộng sự (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C, bài báo gốc về async actor-critic.
- [Schulman và cộng sự (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE.
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — nền tảng; kết hợp với Ch. 9 về xấp xỉ hàm khi critic là một mạng thần kinh.
- [Espeholt và cộng sự (2018). IMPALA](https://arxiv.org/abs/1802.01561) — actor-critic phân tán có khả năng mở rộng với hiệu chỉnh off-policy V-trace.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — các triển khai A2C/PPO sản xuất đáng đọc.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — kết quả hội tụ nền tảng cho phân rã actor-critic hai thang thời gian.