# Monte Carlo Methods — Học từ các tập (episodes) hoàn chỉnh

> Dynamic programming cần một mô hình. Monte Carlo không cần gì ngoài các tập (episodes). Chạy policy, quan sát các return, và tính trung bình chúng. Đây là ý tưởng đơn giản nhất trong RL — và là chìa khóa mở ra mọi thứ phía sau.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75 phút

## Vấn đề

Dynamic programming rất thanh lịch, nhưng nó giả định rằng bạn có thể truy vấn `P(s' | s, a)` cho mọi trạng thái và hành động. Hầu như không có gì trong thế giới thực hoạt động theo cách đó. Một robot không thể tính toán phân phối trên các pixel camera sau khi tác động một lực vào khớp. Một thuật toán định giá không thể tích phân trên mọi phản ứng có thể có của khách hàng. Một LLM không thể liệt kê tất cả các khả năng tiếp nối sau một token.

Bạn cần một phương pháp chỉ yêu cầu khả năng *lấy mẫu* (sample) từ môi trường. Chạy policy. Nhận một quỹ đạo `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`. Sử dụng nó để ước tính các giá trị. Đó chính là Monte Carlo.

Sự chuyển dịch từ DP sang MC có ý nghĩa triết học quan trọng: chúng ta chuyển từ *mô hình đã biết + backup chính xác* sang *rollout được lấy mẫu + return trung bình*. Phương sai tăng lên, nhưng khả năng áp dụng lại bùng nổ. Mọi thuật toán RL sau bài học này — TD, Q-learning, REINFORCE, PPO, GRPO — về bản chất đều là các bộ ước lượng Monte Carlo, đôi khi có thêm bootstrapping ở phía trên.

## Khái niệm

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**Ý tưởng cốt lõi, trong một dòng:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` trong đó `G^{(i)}(s)` là các return quan sát được sau khi ghé thăm `s` dưới policy `π`.

**First-visit vs every-visit MC.** Với một tập (episode) ghé thăm trạng thái `s` nhiều lần, first-visit MC chỉ tính return từ lần ghé thăm đầu tiên; every-visit MC tính tất cả các lần ghé thăm. Cả hai đều không chệch (unbiased) trong giới hạn. First-visit đơn giản hơn để phân tích (các mẫu iid). Every-visit sử dụng nhiều dữ liệu hơn trên mỗi tập và thường hội tụ nhanh hơn trong thực tế.

**Trung bình lũy tiến (Incremental mean).** Thay vì lưu trữ tất cả các return, hãy cập nhật giá trị trung bình chạy:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

Sắp xếp lại: `V_new = V_old + α · (target - V_old)` với `α = 1/n`. Thay thế `1/n` bằng một hằng số bước `α ∈ (0, 1)` và bạn sẽ có một bộ ước lượng MC không dừng (non-stationary) giúp theo dõi các thay đổi trong `π`. Bước đi đó chính là toàn bộ sự chuyển dịch từ MC sang TD đến mọi thuật toán RL hiện đại.

**Khám phá (Exploration) giờ là một vấn đề.** DP chạm đến mọi trạng thái bằng cách liệt kê. MC chỉ thấy các trạng thái mà policy ghé thăm. Nếu `π` là tất định, toàn bộ các vùng của không gian trạng thái sẽ không bao giờ được lấy mẫu, và các ước tính giá trị của chúng sẽ mãi ở mức 0. Ba cách khắc phục, theo thứ tự lịch sử:

1. **Exploring starts.** Bắt đầu mỗi tập từ một cặp (s, a) ngẫu nhiên. Đảm bảo độ bao phủ; không thực tế (bạn không thể "reset" một robot vào một trạng thái tùy ý).
2. **ε-greedy.** Hành động theo kiểu greedy với Q hiện tại, nhưng với xác suất `ε` hãy chọn một hành động ngẫu nhiên. Tất cả các cặp trạng thái-hành động sẽ được lấy mẫu tiệm cận.
3. **Off-policy MC.** Thu thập dữ liệu dưới một behavior policy `μ`, học về target policy `π` thông qua importance sampling. Phương sai cao, nhưng đây là cầu nối đến các phương pháp replay-buffer như DQN.

**Monte Carlo Control.** Đánh giá → cải thiện → đánh giá, giống như policy iteration, nhưng việc đánh giá dựa trên lấy mẫu:

1. Chạy `π`, nhận một tập.
2. Cập nhật `Q(s, a)` từ các return quan sát được.
3. Làm cho `π` trở thành ε-greedy đối với `Q`.
4. Lặp lại.

Hội tụ về `Q*` và `π*` với xác suất 1 trong các điều kiện nhẹ (mỗi cặp được ghé thăm vô số lần, `α` thỏa mãn Robbins-Monro).

```figure
epsilon-greedy
```

## Xây dựng

### Bước 1: rollout → danh sách (s, a, r)

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

Không có mô hình, chỉ có `env.reset()` và `env.step(s, a)`. Giao diện giống như môi trường gym nhưng đã được tinh giản.

### Bước 2: tính toán các return (quét ngược)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

Một lượt quét, `O(T)`. Công thức đệ quy ngược `G_t = r_{t+1} + γ G_{t+1}` tránh việc phải tính tổng lại.

### Bước 3: đánh giá first-visit MC

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

Ba dòng thực hiện công việc: đánh dấu trạng thái đã thấy ở lần ghé thăm đầu tiên, tăng biến đếm, cập nhật trung bình lũy tiến.

### Bước 4: ε-greedy MC control (on-policy)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Bước 5: so sánh với tiêu chuẩn vàng DP

Ước tính MC của bạn về `V^π` sẽ khớp với kết quả DP từ Bài 02 khi số tập → ∞. Trong thực tế: 50.000 tập trên 4×4 GridWorld sẽ đưa bạn đến trong khoảng `~0.1` so với đáp án DP.

## Các cạm bẫy

- **Các tập vô hạn.** MC yêu cầu các tập phải *kết thúc*. Nếu policy của bạn có thể lặp vô hạn, hãy giới hạn `max_steps` và coi giới hạn đó là thất bại ngầm định. GridWorld với một policy ngẫu nhiên thường xuyên bị timeout — điều đó là bình thường, chỉ cần đảm bảo bạn đếm nó một cách chính xác.
- **Phương sai.** MC sử dụng các return đầy đủ. Trên các tập dài, phương sai rất lớn — một phần thưởng không may mắn ở cuối sẽ làm thay đổi `V(s_0)` một lượng tương ứng. Các phương pháp TD (Bài 04) cắt giảm điều này bằng cách bootstrapping.
- **Độ bao phủ trạng thái.** Greedy MC trên một Q mới với các giá trị bằng nhau sẽ chỉ thử một hành động duy nhất. Bạn *phải* khám phá (ε-greedy, exploring starts, UCB).
- **Các policy không dừng.** Nếu `π` thay đổi (như trong MC control), các return cũ là từ một policy khác. Constant-α MC xử lý điều này; sample-average MC thì không.
- **Off-policy importance sampling.** Các trọng số `π(a|s)/μ(a|s)` nhân lên trên một quỹ đạo. Phương sai bùng nổ theo chiều dài horizon. Hãy giới hạn bằng per-decision weighted IS hoặc chuyển sang TD.

## Sử dụng

Vai trò của các phương pháp Monte Carlo vào năm 2026:

| Trường hợp sử dụng | Tại sao chọn MC |
|----------|--------|
| Trò chơi ngắn (blackjack, poker) | Các tập kết thúc tự nhiên; return rất sạch. |
| Đánh giá offline một policy đã ghi lại | Trung bình các return đã chiết khấu trên các quỹ đạo đã lưu. |
| Monte Carlo Tree Search (AlphaZero) | Các rollout MC từ các lá cây hướng dẫn việc lựa chọn. |
| Đánh giá LLM RL | Tính phần thưởng trung bình trên các hoàn thiện được lấy mẫu cho một policy nhất định. |
| Ước tính baseline trong PPO | Mục tiêu advantage `A_t = G_t - V(s_t)` sử dụng một `G_t` MC. |
| Giảng dạy RL | Thuật toán đơn giản nhất thực sự hoạt động — loại bỏ bootstrapping để thấy cốt lõi. |

Các thuật toán deep-RL hiện đại (PPO, SAC) nội suy giữa MC thuần túy (return đầy đủ) và TD thuần túy (bootstrap một bước) thông qua các return `n`-bước hoặc GAE. Cả hai điểm cuối đều là các trường hợp của cùng một bộ ước lượng.

## Triển khai

Lưu dưới dạng `outputs/skill-mc-evaluator.md`:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## Bài tập

1. **Dễ.** Triển khai đánh giá first-visit MC của policy ngẫu nhiên đồng nhất trên 4×4 GridWorld. Chạy 10.000 tập. Vẽ đồ thị `V(0,0)` theo số lượng tập so với đáp án DP.
2. **Trung bình.** Triển khai ε-greedy MC control với `ε ∈ {0.01, 0.1, 0.3}`. So sánh return trung bình sau 20.000 tập. Đường cong trông như thế nào? Sự đánh đổi giữa bias và phương sai nằm ở đâu?
3. **Khó.** Triển khai *off-policy* MC với importance sampling: thu thập dữ liệu dưới policy ngẫu nhiên đồng nhất `μ`, ước tính `V^π` cho policy tối ưu tất định `π`. So sánh IS đơn thuần vs per-decision IS vs weighted IS. Cái nào có phương sai thấp nhất?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Monte Carlo | "Lấy mẫu ngẫu nhiên" | Ước tính kỳ vọng bằng cách lấy trung bình trên các mẫu iid từ phân phối. |
| Return `G_t` | "Phần thưởng tương lai" | Tổng các phần thưởng đã chiết khấu từ bước `t` đến cuối tập: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| First-visit MC | "Đếm mỗi trạng thái một lần" | Chỉ lần ghé thăm đầu tiên trong một tập đóng góp vào ước tính giá trị. |
| Every-visit MC | "Sử dụng tất cả các lần ghé thăm" | Mọi lần ghé thăm đều đóng góp; hơi chệch nhưng hiệu quả lấy mẫu cao hơn. |
| ε-greedy | "Nhiễu khám phá" | Chọn hành động greedy với xác suất `1-ε`; hành động ngẫu nhiên với xác suất `ε`. |
| Importance sampling | "Sửa lỗi do lấy mẫu từ phân phối sai" | Trọng số lại các return bằng tích `π(a\|s)/μ(a\|s)` để ước tính `V^π` từ dữ liệu `μ`. |
| On-policy | "Học từ dữ liệu của chính mình" | Target policy = behavior policy. Vanilla MC, PPO, SARSA. |
| Off-policy | "Học từ dữ liệu của người khác" | Target policy ≠ behavior policy. Importance-sampled MC, Q-learning, DQN. |

## Đọc thêm

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — tài liệu chuẩn mực.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — phân tích first-visit vs every-visit.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — off-policy MC và kiểm soát phương sai.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — các bộ ước lượng IS phương sai thấp hiện đại.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — minh chứng thực nghiệm quy mô lớn đầu tiên về việc MC/TD self-play hội tụ đến trình độ siêu nhân; tiền thân khái niệm cho mọi bài học trong nửa sau của giai đoạn này.