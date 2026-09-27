# Dynamic Programming — Policy Iteration & Value Iteration

> Dynamic programming là RL với sự "gian lận". Bạn đã biết trước các hàm chuyển trạng thái (transition) và hàm phần thưởng (reward); bạn chỉ cần lặp lại phương trình Bellman cho đến khi `V` hoặc `π` không còn thay đổi nữa. Đây là tiêu chuẩn vàng mà mọi phương pháp dựa trên lấy mẫu (sampling-based) đều cố gắng tiệm cận.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75 minutes

## The Problem

Bạn có một MDP với mô hình đã biết: bạn có thể truy vấn `P(s' | s, a)` và `R(s, a, s')` cho bất kỳ cặp trạng thái-hành động nào. Một người quản lý kho hàng biết phân phối nhu cầu. Một trò chơi bàn cờ có các bước chuyển xác định. Một gridworld chỉ là bốn dòng code Python. Bạn đang sở hữu một *mô hình*.

Model-free RL (Q-learning, PPO, REINFORCE) được phát minh cho trường hợp bạn không có mô hình — bạn chỉ có thể lấy mẫu từ môi trường. Nhưng khi bạn đã có mô hình, có những phương pháp nhanh hơn và tốt hơn: dynamic programming. Bellman đã thiết kế chúng vào năm 1957. Chúng vẫn định nghĩa thế nào là sự chính xác: khi người ta nói "chính sách tối ưu cho MDP này", họ có ý nói đến chính sách mà DP sẽ trả về.

Bạn cần chúng vào năm 2026 vì ba lý do. Thứ nhất, mọi môi trường dạng bảng (tabular) trong nghiên cứu RL (GridWorld, FrozenLake, CliffWalking) đều được giải bằng DP để tạo ra chính sách tiêu chuẩn vàng. Thứ hai, các giá trị chính xác cho phép bạn *debug* các phương pháp lấy mẫu: nếu ước tính của Q-learning cho `V*(s_0)` sai lệch 30% so với kết quả của DP, thì Q-learning của bạn đang có lỗi. Thứ ba, các phương pháp offline RL và lập kế hoạch hiện đại (MCTS, tìm kiếm của AlphaZero, model-based RL trong Phase 9 · 10) đều lặp lại một Bellman backup trên một mô hình đã học hoặc đã cho trước.

## The Concept

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**Hai thuật toán, cả hai đều là lặp điểm cố định (fixed-point iteration) trên phương trình Bellman.**

**Policy iteration.** Luân phiên hai bước cho đến khi chính sách không còn thay đổi.

1. *Đánh giá (Evaluation):* với chính sách `π`, tính toán `V^π` bằng cách áp dụng liên tục `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` cho đến khi hội tụ.
2. *Cải thiện (Improvement):* với `V^π`, làm cho `π` trở nên tham lam (greedy) đối với `V^π`: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

Sự hội tụ được đảm bảo vì (a) mỗi bước cải thiện hoặc giữ nguyên `π` hoặc tăng nghiêm ngặt `V^π` cho một trạng thái nào đó, (b) không gian các chính sách xác định là hữu hạn. Thông thường hội tụ trong khoảng 5–20 vòng lặp ngoài ngay cả với không gian trạng thái lớn.

**Value iteration.** Gộp bước đánh giá và cải thiện thành một lượt quét duy nhất. Áp dụng phương trình *tối ưu* Bellman:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

Lặp lại cho đến khi `max_s |V_{new}(s) - V(s)| < ε`. Trích xuất chính sách ở cuối bằng cách chọn hành động tham lam. Nhanh hơn đáng kể trên mỗi vòng lặp — không có vòng lặp đánh giá bên trong — nhưng thường cần nhiều vòng lặp hơn để hội tụ.

**Generalized policy iteration (GPI).** Khung lý thuyết thống nhất. Hàm giá trị và chính sách bị khóa trong một vòng lặp cải thiện hai chiều; bất kỳ phương pháp nào thúc đẩy cả hai hướng tới sự nhất quán lẫn nhau (async value iteration, modified policy iteration, Q-learning, actor-critic, PPO) đều là một trường hợp của GPI.

**Tại sao `γ < 1` quan trọng.** Toán tử Bellman là một `γ`-contraction trong chuẩn sup: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. Tính chất contraction ngụ ý một điểm cố định duy nhất và sự hội tụ hình học. Bỏ `γ < 1` và bạn sẽ mất đi sự đảm bảo — bạn cần một chân trời hữu hạn (finite horizon) hoặc một trạng thái kết thúc hấp thụ (absorbing terminal state).

```figure
value-iteration-gamma
```

## Build It

### Step 1: xây dựng mô hình GridWorld MDP

Sử dụng GridWorld 4×4 tương tự từ Bài 01. Chúng ta thêm một biến thể ngẫu nhiên: với xác suất `0.1`, tác nhân sẽ trượt sang một hướng vuông góc ngẫu nhiên.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` trả về một danh sách các `(s', r, p)`. Đây là toàn bộ mô hình.

### Step 2: đánh giá chính sách (policy evaluation)

Với chính sách `π(s) = {action: prob}`, lặp lại phương trình Bellman cho đến khi `V` không còn thay đổi:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Step 3: cải thiện chính sách (policy improvement)

Thay thế `π` bằng chính sách tham lam đối với `V`. Nếu `π` không thay đổi, hãy trả về — chúng ta đã đạt đến tối ưu.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Step 4: kết hợp chúng lại

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

Sự hội tụ điển hình trên 4×4: 4–6 vòng lặp ngoài. Trả về `V*(0,0) ≈ -6` và một chính sách giúp giảm số bước đi một cách nghiêm ngặt.

### Step 5: value iteration (phiên bản một vòng lặp)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

Cùng một điểm cố định, ít dòng code hơn.

## Pitfalls

- **Quên xử lý các trạng thái kết thúc.** Nếu bạn áp dụng Bellman cho một trạng thái hấp thụ, nó vẫn sẽ chọn một "hành động tốt nhất" mà không thay đổi gì cả. Hãy bảo vệ bằng `if s == terminal: V[s] = 0`.
- **Chuẩn sup vs hội tụ L2.** Sử dụng `max |V_new - V|`, không phải trung bình. Đảm bảo lý thuyết nằm trên chuẩn sup.
- **Cập nhật tại chỗ (in-place) vs đồng bộ.** Cập nhật `V[s]` tại chỗ (Gauss-Seidel) hội tụ nhanh hơn so với việc dùng một dict `V_new` riêng biệt (Jacobi). Code thực tế thường sử dụng cập nhật tại chỗ.
- **Ràng buộc chính sách (Policy ties).** Nếu hai hành động có giá trị Q bằng nhau, `argmax` có thể phá vỡ ràng buộc khác nhau mỗi vòng lặp, gây ra hiện tượng dao động trong kiểm tra "policy stable". Hãy sử dụng cách phá vỡ ràng buộc ổn định (hành động đầu tiên theo thứ tự cố định).
- **Bùng nổ không gian trạng thái.** DP có độ phức tạp `O(|S| · |A|)` mỗi lượt quét. Hoạt động tốt lên đến khoảng 10⁷ trạng thái. Vượt quá mức đó, bạn cần xấp xỉ hàm (Phase 9 · 05 trở đi).

## Use It

Vào năm 2026, DP là cơ sở cho sự chính xác và là vòng lặp bên trong của các bộ lập kế hoạch:

| Use case | Method |
|----------|--------|
| Giải chính xác một tabular MDP nhỏ | Value iteration (đơn giản hơn) hoặc policy iteration (ít bước ngoài hơn) |
| Xác minh cài đặt Q-learning / PPO | So sánh với DP-optimal V* trên một môi trường đồ chơi |
| Model-based RL (Phase 9 · 10) | Bellman backup trên một mô hình chuyển trạng thái đã học |
| Lập kế hoạch trong AlphaZero / MuZero | Monte Carlo Tree Search = async Bellman backup |
| Offline RL (CQL, IQL) | Conservative Q-iteration — DP với hình phạt trên các hành động OOD |

Mỗi khi ai đó nói "hàm giá trị tối ưu", họ có ý nói "điểm cố định của DP". Khi bạn thấy `V*` hoặc `Q*` trong một bài báo, hãy hình dung vòng lặp này.

## Ship It

Lưu dưới dạng `outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## Exercises

1. **Dễ.** Chạy value iteration trên 4×4 GridWorld với `γ ∈ {0.9, 0.99}`. Cần bao nhiêu lượt quét cho đến khi `max |ΔV| < 1e-6`? In `V*` dưới dạng lưới 4×4.
2. **Trung bình.** So sánh policy iteration và value iteration trên GridWorld *ngẫu nhiên* (xác suất trượt `0.1`). Đếm: số lượt quét, thời gian thực, `V*(0,0)` cuối cùng. Cái nào hội tụ nhanh hơn về số vòng lặp? Về thời gian thực?
3. **Khó.** Xây dựng modified policy iteration: trong bước đánh giá, chỉ chạy `k` lượt quét thay vì chạy đến khi hội tụ. Vẽ biểu đồ sai số `V*(0,0)` so với `k` cho `k ∈ {1, 2, 5, 10, 50}`. Đường cong này cho bạn biết gì về sự đánh đổi giữa đánh giá và cải thiện?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Policy iteration | "DP algorithm" | Luân phiên đánh giá (`V^π`) và cải thiện (tham lam `π` đối với `V^π`) cho đến khi chính sách không thay đổi. |
| Value iteration | "Faster DP" | Bellman optimality backup áp dụng trong một lượt quét; hội tụ về `V*` theo cấp số nhân. |
| Bellman operator | "The recursion" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; một `γ`-contraction trong chuẩn sup. |
| Contraction | "Why DP converges" | Bất kỳ toán tử `T` nào với `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` đều có một điểm cố định duy nhất. |
| GPI | "Everything is DP" | Generalized Policy Iteration: bất kỳ phương pháp nào thúc đẩy `V` và `π` đến sự nhất quán lẫn nhau. |
| Synchronous update | "Jacobi-style" | Sử dụng `V` cũ trong suốt lượt quét; dễ phân tích nhưng chậm hơn. |
| In-place update | "Gauss-Seidel-style" | Sử dụng `V` ngay khi nó được cập nhật; hội tụ nhanh hơn trong thực tế. |

## Further Reading

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — trình bày chuẩn mực về policy iteration và value iteration.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — xử lý nghiêm ngặt các lập luận về contraction-mapping.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — modified policy iteration và phân tích hội tụ của nó.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — bài báo gốc về policy iteration.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — cầu nối từ DP đến approximate-DP / deep RL được sử dụng trong mọi bài học tiếp theo.