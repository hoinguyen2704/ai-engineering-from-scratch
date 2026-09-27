# Policy Gradient — REINFORCE from Scratch

> Đừng ước tính giá trị nữa. Hãy tham số hóa trực tiếp policy, tính gradient của lợi nhuận kỳ vọng và bước lên phía trước. Williams (1992) đã viết điều đó trong một định lý duy nhất. Đây là lý do tại sao PPO, GRPO và mọi vòng lặp RL cho LLM tồn tại.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 phút

## Vấn đề

Q-learning và DQN tham số hóa hàm *giá trị*. Bạn chọn hành động bằng `argmax Q`. Điều đó ổn với các hành động và trạng thái rời rạc. Nó sẽ thất bại khi hành động là liên tục (ví dụ: `argmax` trên một mô-men xoắn 10 chiều?) hoặc khi bạn muốn một policy ngẫu nhiên (`argmax` được xây dựng theo hướng tất định).

Policy gradient tham số hóa trực tiếp *policy*. `π_θ(a | s)` là một mạng thần kinh xuất ra một phân phối xác suất trên các hành động. Lấy mẫu từ đó để thực hiện hành động. Tính gradient của lợi nhuận kỳ vọng theo `θ`. Bước lên phía trước. Không cần `argmax`. Không cần đệ quy Bellman. Chỉ cần gradient ascent trên `J(θ) = E_{π_θ}[G]`.

Định lý REINFORCE (Williams 1992) cho bạn biết gradient này có thể tính toán được: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. Chạy một tập (episode). Tính lợi nhuận. Nhân với `∇ log π_θ(a | s)` tại mỗi bước. Lấy trung bình. Gradient-ascent. Xong.

Mọi thuật toán LLM-RL vào năm 2026 — PPO, DPO, GRPO — đều là sự tinh chỉnh của REINFORCE. Việc nắm vững nó là điều kiện tiên quyết cho phần còn lại của giai đoạn này, cũng như cho Phase 10 · 07 (triển khai RLHF) và Phase 10 · 08 (DPO).

## Khái niệm

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**Định lý policy gradient.** Với bất kỳ policy `π_θ` nào được tham số hóa bởi `θ`:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

trong đó `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` là lợi nhuận chiết khấu từ bước `t`. Kỳ vọng được tính trên toàn bộ quỹ đạo `τ` được lấy mẫu từ `π_θ`.

**Chứng minh rất ngắn gọn.** Lấy đạo hàm của `J(θ) = Σ_τ P(τ; θ) G(τ)` dưới dấu kỳ vọng. Sử dụng `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (thủ thuật log-derivative). Phân tích `log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms that do not depend on θ`. Các thành phần môi trường sẽ triệt tiêu. Hai dòng đại số sẽ cho bạn định lý này.

**Các thủ thuật giảm phương sai.** REINFORCE cơ bản có phương sai rất lớn — lợi nhuận thì nhiễu, `∇ log π` thì nhiễu, tích của chúng lại càng nhiễu hơn. Hai cách khắc phục tiêu chuẩn:

1. **Trừ baseline.** Thay thế `G_t` bằng `G_t - b(s_t)` cho bất kỳ baseline `b(s_t)` nào không phụ thuộc vào `a_t`. Điều này không làm chệch hướng (unbiased) vì `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`. Lựa chọn điển hình: `b(s_t) = V̂(s_t)` được học bởi một critic → actor-critic (Bài 07).
2. **Reward-to-go.** Thay thế `Σ_t G_t · ∇ log π_θ(a_t | s_t)` bằng `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`. Chỉ các lợi nhuận tương lai mới quan trọng đối với một hành động nhất định — các phần thưởng trong quá khứ chỉ đóng góp nhiễu có giá trị trung bình bằng không.

Kết hợp lại, bạn có:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

đây chính là REINFORCE với baseline — tổ tiên trực tiếp của A2C (Bài 07) và PPO (Bài 08).

**Tham số hóa policy bằng Softmax.** Đối với các hành động rời rạc, lựa chọn tiêu chuẩn là:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

trong đó `f_θ` là bất kỳ mạng thần kinh nào xuất ra điểm số cho mỗi hành động. Gradient có dạng rất gọn:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

tức là, điểm số của hành động đã chọn trừ đi giá trị kỳ vọng của nó dưới policy hiện tại.

**Policy Gaussian cho hành động liên tục.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)` có dạng đóng. Đó là tất cả những gì SAC trong Phase 9 · 07 cần.

```figure
policy-gradient-landscape
```

## Xây dựng

### Bước 1: Mạng policy softmax

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Sử dụng policy tuyến tính (một vector trọng số cho mỗi hành động) cho môi trường dạng bảng. Đối với Atari, hãy thay bằng CNN và giữ nguyên lớp softmax.

### Bước 2: Lấy mẫu và log-probability

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Bước 3: Rollout với log-probs được lưu lại

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Bước 4: Cập nhật REINFORCE

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

Gradient `∇ log π(a|s) = e_a - π(·|s)` (onehot của `a` trừ đi các xác suất) là trái tim của policy gradient softmax. Hãy ghi nhớ nó vào bộ nhớ cơ bắp.

### Bước 5: Baselines

Giá trị trung bình trượt của `G` qua các tập gần đây là đủ để giảm phương sai cho GridWorld 4×4; mất khoảng 500 tập để hội tụ. Nâng cấp baseline thành một `V̂(s)` được học và bạn sẽ có actor-critic.

## Các cạm bẫy

- **Exploding gradients.** Lợi nhuận có thể rất lớn. Luôn chuẩn hóa `G` thành `~N(0, 1)` trên toàn batch trước khi nhân với `∇ log π`.
- **Entropy collapse.** Policy hội tụ về một hành động gần như tất định quá sớm, ngừng khám phá, dẫn đến bị kẹt. Cách khắc phục: thêm entropy bonus `β · H(π(·|s))` vào hàm mục tiêu.
- **Phương sai cao.** REINFORCE cơ bản cần hàng ngàn tập. Baseline critic (Bài 07) hoặc trust region của TRPO/PPO (Bài 08) là cách khắc phục tiêu chuẩn.
- **Kém hiệu quả về mẫu.** On-policy nghĩa là bạn vứt bỏ mọi transition sau một lần cập nhật. Các hiệu chỉnh off-policy thông qua importance sampling giúp tận dụng lại dữ liệu, nhưng phải trả giá bằng phương sai (tỷ lệ trong PPO là một trọng số IS đã được cắt tỉa).
- **Gradient không dừng (Non-stationary).** Cùng một gradient từ 100 tập trước sử dụng `π` cũ. Các phương pháp on-policy cập nhật sau mỗi vài lần rollout vì lý do này.
- **Phân bổ tín dụng (Credit assignment).** Nếu không có reward-to-go, các phần thưởng quá khứ sẽ tạo ra nhiễu. Luôn sử dụng reward-to-go.

## Sử dụng

Vào năm 2026, REINFORCE hiếm khi được chạy trực tiếp nhưng công thức gradient của nó có mặt ở khắp mọi nơi:

| Trường hợp sử dụng | Phương pháp dẫn xuất |
|----------|---------------|
| Điều khiển liên tục | PPO / SAC với Gaussian policy |
| LLM RLHF | PPO với KL penalty, chạy trên policy cấp token |
| LLM reasoning (DeepSeek) | GRPO — REINFORCE với baseline tương đối theo nhóm, không có critic |
| Đa tác nhân | Centralized-critic REINFORCE (MADDPG, COMA) |
| Robot hành động rời rạc | A2C, A3C, PPO |
| Cài đặt chỉ dựa trên ưu tiên | DPO — REINFORCE được viết lại thành hàm mất mát preference-likelihood, không lấy mẫu |

Khi bạn đọc `loss = -advantage * log_prob` trong một script huấn luyện năm 2026, đó chính là REINFORCE với baseline. Toàn bộ các bài báo (DPO, GRPO, RLOO) đều là các thủ thuật giảm phương sai dựa trên dòng công thức này.

## Triển khai

Lưu dưới dạng `outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Bài tập

1. **Dễ.** Triển khai REINFORCE trên GridWorld 4×4 với policy softmax tuyến tính. Huấn luyện trong 1.000 tập mà không có baseline. Vẽ đường cong học tập; đo phương sai (độ lệch chuẩn của lợi nhuận).
2. **Trung bình.** Thêm baseline trung bình trượt. Huấn luyện lại. So sánh hiệu quả mẫu và phương sai với bản chạy cơ bản. Baseline giúp giảm bao nhiêu bước để hội tụ?
3. **Khó.** Thêm entropy bonus `β · H(π)`. Quét giá trị `β ∈ {0, 0.01, 0.1, 1.0}`. Vẽ lợi nhuận cuối cùng và entropy của policy. Đâu là điểm tối ưu cho tác vụ này?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Policy gradient | "Huấn luyện policy trực tiếp" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; dẫn xuất từ thủ thuật log-derivative. |
| REINFORCE | "Thuật toán PG gốc" | Williams (1992); lợi nhuận Monte Carlo nhân với log-policy gradient. |
| Log-derivative trick | "Ước tính hàm điểm" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; giúp gradient của các kỳ vọng trở nên khả thi. |
| Baseline | "Giảm phương sai" | Bất kỳ `b(s)` nào được trừ khỏi `G`; không chệch vì `E[b · ∇ log π] = 0`. |
| Reward-to-go | "Chỉ tính lợi nhuận tương lai" | `G_t^{from t}` thay vì toàn bộ `G_0`; chính xác và phương sai thấp hơn. |
| Entropy bonus | "Khuyến khích khám phá" | Thành phần `+β · H(π(·\|s))` giúp policy không bị suy giảm (collapse). |
| On-policy | "Huấn luyện trên những gì vừa thấy" | Kỳ vọng gradient theo policy hiện tại — không thể tái sử dụng dữ liệu cũ trực tiếp. |
| Advantage | "Tốt hơn trung bình bao nhiêu" | `A(s, a) = G(s, a) - V(s)`; đại lượng có dấu mà REINFORCE-với-baseline nhân vào. |

## Đọc thêm

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — bài báo gốc về REINFORCE.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — định lý policy-gradient hiện đại với xấp xỉ hàm.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — trình bày trong sách giáo khoa.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — giải thích sư phạm rõ ràng với mã nguồn PyTorch.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — giảm phương sai và góc nhìn natural-gradient kết nối REINFORCE với họ trust-region (TRPO, PPO).