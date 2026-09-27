# MDPs, States, Actions & Rewards

> Một Markov Decision Process bao gồm năm thành phần: trạng thái (states), hành động (actions), chuyển đổi (transitions), phần thưởng (rewards) và chiết khấu (discount). Mọi thứ trong RL — Q-learning, PPO, DPO, GRPO — đều tối ưu hóa dựa trên cấu trúc này. Hãy học nó một lần, và bạn sẽ hiểu phần còn lại của reinforcement learning một cách dễ dàng.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**Time:** ~45 minutes

## The Problem

Bạn đang viết một bot chơi cờ. Hoặc một hệ thống lập kế hoạch tồn kho. Hoặc một tác nhân giao dịch. Hoặc vòng lặp PPO để huấn luyện một mô hình suy luận. Bốn lĩnh vực khác nhau, nhưng có một sự thật đáng ngạc nhiên: cả bốn đều quy về cùng một đối tượng toán học.

Supervised learning cung cấp cho bạn các cặp `(x, y)` và yêu cầu bạn khớp một hàm số. Reinforcement learning không cung cấp nhãn — chỉ có một luồng các trạng thái, các hành động bạn đã thực hiện và một phần thưởng vô hướng. Nước đi đó có thắng ván cờ không? Quyết định nhập hàng có tiết kiệm chi phí không? Giao dịch đó có mang lại lợi nhuận không? Token mà LLM vừa tạo ra có dẫn đến phần thưởng cao hơn từ bộ đánh giá (judge) không?

Bạn không thể học từ luồng dữ liệu này cho đến khi bạn chính thức hóa nó. "Những gì tôi thấy", "những gì tôi đã làm", "điều gì xảy ra tiếp theo", "điều đó tốt như thế nào" — mỗi thứ đều phải trở thành một đối tượng mà bạn có thể suy luận. Sự chính thức hóa đó chính là Markov Decision Process. Mọi thuật toán RL trong giai đoạn này, bao gồm cả các vòng lặp RLHF và GRPO ở cuối, đều tối ưu hóa dựa trên cấu trúc này.

## The Concept

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**Năm đối tượng.**

- **States** `S`. Mọi thứ tác nhân cần để đưa ra quyết định. Trong GridWorld, đó là ô lưới. Trong cờ vua, đó là bàn cờ. Trong LLM, đó là cửa sổ ngữ cảnh (context window) cộng với bất kỳ bộ nhớ nào.
- **Actions** `A`. Các lựa chọn. Di chuyển lên/xuống/trái/phải. Thực hiện một nước đi. Phát ra một token.
- **Transitions** `P(s' | s, a)`. Với trạng thái `s` và hành động `a`, đây là phân phối xác suất về trạng thái tiếp theo. Có tính tất định trong cờ vua, ngẫu nhiên trong tồn kho, và gần như tất định trong giải mã LLM.
- **Rewards** `R(s, a, s')`. Tín hiệu vô hướng. Thắng = +1, thua = -1. Doanh thu trừ chi phí. Số hạng tỷ lệ log-likelihood trong GRPO.
- **Discount** `γ ∈ [0, 1)`. Mức độ quan trọng của phần thưởng tương lai so với hiện tại. `γ = 0.99` mang lại tầm nhìn khoảng ~100 bước; `γ = 0.9` mang lại ~10 bước.

**Tính chất Markov** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. Tương lai chỉ phụ thuộc vào trạng thái hiện tại. Nếu không, biểu diễn trạng thái chưa đầy đủ — đây không phải là lỗi của phương pháp, mà là lỗi của trạng thái.

**Policies và returns.** Một policy `π(a | s)` ánh xạ các trạng thái tới các phân phối hành động. Return `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` là tổng chiết khấu của các phần thưởng tương lai. Value `V^π(s) = E[G_t | s_t = s]` là return kỳ vọng bắt đầu từ `s` theo policy `π`. Q-value `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` là return kỳ vọng bắt đầu với một hành động cụ thể. Mọi thuật toán RL đều ước tính một trong hai giá trị này, sau đó cải thiện `π` tương ứng.

**Các phương trình Bellman.** Các phương trình điểm cố định mà mọi thứ trong giai đoạn này đều sử dụng:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

Chúng chia return kỳ vọng thành "phần thưởng của bước này" cộng với "giá trị chiết khấu của nơi bạn đặt chân đến". Có tính đệ quy. Mọi thuật toán trong Phase 9 đều lặp lại phương trình này để hội tụ (dynamic programming), lấy mẫu từ nó (Monte Carlo), hoặc bootstrap nó một bước (temporal difference).

```figure
discount-horizon
```

## Build It

### Step 1: một MDP tất định nhỏ

Một GridWorld 4×4. Tác nhân bắt đầu ở góc trên bên trái, kết thúc ở góc dưới bên phải, phần thưởng -1 mỗi bước, các hành động `{up, down, left, right}`. Xem `code/main.py`.

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

Năm dòng. Đó là toàn bộ môi trường. Các chuyển đổi tất định, hình phạt bước cố định, trạng thái kết thúc hấp thụ.

### Step 2: triển khai một policy

Một policy là một hàm từ trạng thái đến phân phối hành động. Đơn giản nhất: ngẫu nhiên đồng nhất.

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

Chạy policy ngẫu nhiên 1000 lần. Return trung bình nằm trong khoảng -60 đến -80 cho bàn cờ 4×4 này. Return tối ưu là -6 (đường đi thẳng xuống-phải). Thu hẹp khoảng cách đó là mục tiêu của toàn bộ Phase 9.

### Step 3: tính toán `V^π` chính xác thông qua phương trình Bellman

Đối với các MDP nhỏ, phương trình Bellman là một hệ phương trình tuyến tính. Liệt kê các trạng thái, áp dụng kỳ vọng, lặp lại cho đến khi các giá trị không thay đổi nữa.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

Đây là đánh giá policy lặp (iterative policy evaluation). Đây là thuật toán đầu tiên trong Sutton & Barto và là nền tảng lý thuyết của mọi phương pháp RL theo sau.

### Step 4: `γ` là một siêu tham số có ý nghĩa vật lý

Tầm nhìn hiệu dụng (effective horizon) xấp xỉ `1 / (1 - γ)`. `γ = 0.9` → 10 bước. `γ = 0.99` → 100 bước. `γ = 0.999` → 1000 bước.

Nếu quá thấp, tác nhân sẽ hành động thiển cận. Nếu quá cao, việc phân bổ tín dụng (credit assignment) sẽ trở nên nhiễu, vì nhiều bước đầu tiên cùng chịu trách nhiệm cho phần thưởng ở tương lai xa. RLHF cho LLM thường sử dụng `γ = 1` vì các tập (episodes) ngắn và có giới hạn. Các tác vụ điều khiển sử dụng `0.95–0.99`. Các trò chơi chiến lược dài hạn sử dụng `0.999`.

## Pitfalls

- **Trạng thái phi Markov.** Nếu bạn cần ba quan sát gần nhất để quyết định, "trạng thái" không chỉ là quan sát hiện tại. Cách khắc phục: xếp chồng các khung hình (DQN trên Atari xếp chồng 4 khung hình) hoặc sử dụng trạng thái hồi quy (LSTM/GRU trên các quan sát).
- **Phần thưởng thưa thớt (Sparse rewards).** Phần thưởng chỉ khi thắng khiến việc học gần như không thể trong không gian trạng thái lớn. Hãy định hình phần thưởng (tín hiệu trung gian) hoặc bootstrap bằng bắt chước (Phase 9 · 09).
- **Hack phần thưởng.** Tối ưu hóa một phần thưởng thay thế (proxy) thường tạo ra hành vi bệnh lý. Tác nhân đua thuyền của OpenAI đã xoay vòng tròn để thu thập vật phẩm mãi mãi thay vì hoàn thành cuộc đua. Luôn xác định phần thưởng từ kết quả mục tiêu, không phải từ proxy.
- **Sai lệch chiết khấu.** `γ = 1` trên một tác vụ có tầm nhìn vô hạn làm cho mọi giá trị trở nên vô hạn. Luôn giới hạn bằng tầm nhìn hữu hạn hoặc `γ < 1`.
- **Quy mô phần thưởng.** Phần thưởng {+100, -100} so với {+1, -1} tạo ra các policy tối ưu giống hệt nhau nhưng độ lớn gradient khác nhau đáng kể. Hãy chuẩn hóa về khoảng `[-1, 1]` trước khi đưa vào PPO/DQN.

## Use It

Stack 2026 giảm thiểu mọi pipeline RL về một MDP trước khi chạm vào mã nguồn:

| Tình huống | Trạng thái | Hành động | Phần thưởng | γ |
|-----------|-------|--------|--------|---|
| Điều khiển (di chuyển, thao tác) | Góc khớp + vận tốc | Mô-men xoắn liên tục | Định hình theo tác vụ | 0.99 |
| Trò chơi (cờ vua, Go, poker) | Bàn cờ + lịch sử | Nước đi hợp lệ | Thắng=+1 / thua=-1 | 1.0 (hữu hạn) |
| Tồn kho / định giá | Hàng tồn + nhu cầu | Số lượng đặt hàng | Doanh thu - chi phí | 0.95 |
| RLHF cho LLMs | Token ngữ cảnh | Token tiếp theo | Điểm mô hình phần thưởng ở cuối | 1.0 (tập ~200 token) |
| GRPO cho suy luận | Prompt + phản hồi một phần | Token tiếp theo | Bộ xác minh 0/1 ở cuối | 1.0 |

Hãy viết năm bộ tuple trước khi viết bất kỳ vòng lặp huấn luyện nào. Hầu hết các báo cáo lỗi "RL không hoạt động" đều bắt nguồn từ việc xây dựng MDP bị sai ngay trên giấy.

## Ship It

Lưu dưới dạng `outputs/skill-mdp-modeler.md`:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## Exercises

1. **Dễ.** Triển khai GridWorld 4×4 và triển khai policy ngẫu nhiên trong `code/main.py`. Chạy 10.000 tập. Báo cáo giá trị trung bình và độ lệch chuẩn của return. So sánh với return tối ưu (-6).
2. **Trung bình.** Chạy `policy_evaluation` với `γ ∈ {0.5, 0.9, 0.99}` cho policy ngẫu nhiên đồng nhất. In `V` dưới dạng lưới 4×4 cho mỗi trường hợp. Giải thích tại sao các giá trị trạng thái gần trạng thái kết thúc tăng nhanh hơn với `γ` lớn hơn.
3. **Khó.** Làm cho GridWorld trở nên ngẫu nhiên: mỗi hành động có thể trượt sang hướng lân cận với xác suất `p = 0.1`. Đánh giá lại policy đồng nhất. Liệu `V[start]` có tốt hơn hay tệ hơn? Tại sao?

## Key Terms

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| MDP | "Thiết lập reinforcement learning" | Tuple `(S, A, P, R, γ)` thỏa mãn tính chất Markov. |
| State | "Những gì tác nhân thấy" | Thống kê đầy đủ cho động lực tương lai theo lớp policy đã chọn. |
| Policy | "Hành vi của tác nhân" | Phân phối có điều kiện `π(a \| s)` hoặc ánh xạ tất định `s → a`. |
| Return | "Tổng phần thưởng" | Tổng chiết khấu `Σ γ^t r_t` từ bước hiện tại. |
| Value | "Trạng thái tốt như thế nào" | Return kỳ vọng theo `π` bắt đầu từ `s`. |
| Q-value | "Hành động tốt như thế nào" | Return kỳ vọng theo `π` bắt đầu từ `s` với hành động đầu tiên `a`. |
| Bellman equation | "Đệ quy quy hoạch động" | Phân rã điểm cố định của giá trị / Q thành phần thưởng một bước cộng với giá trị kế tiếp đã chiết khấu. |
| Discount `γ` | "Tương lai so với hiện tại" | Trọng số hình học cho phần thưởng tương lai xa; tầm nhìn hiệu dụng `~1/(1-γ)`. |

## Further Reading

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — sách giáo khoa. Chương 3 bao gồm MDP và phương trình Bellman; Chương 1 thúc đẩy giả thuyết phần thưởng làm nền tảng cho mọi bài học sau đó.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — nguồn gốc của phương trình Bellman.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — tài liệu nhập môn MDP ngắn gọn từ góc độ deep-RL.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — tài liệu tham khảo về nghiên cứu vận hành (operations-research) về MDP và các phương pháp giải chính xác.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — cách dẫn xuất MDP sạch nhất như một chuyên ngành của quy hoạch động.