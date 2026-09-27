# Temporal Difference — Q-Learning & SARSA

> Monte Carlo đợi cho đến khi kết thúc episode. TD cập nhật sau mỗi bước bằng cách bootstrapping ước tính giá trị tiếp theo. Q-learning là off-policy và lạc quan; SARSA là on-policy và thận trọng. Cả hai đều chỉ cần một dòng code. Cả hai đều là nền tảng cho mọi phương pháp deep-RL trong giai đoạn này.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**Time:** ~75 phút

## Vấn đề

Monte Carlo hoạt động tốt nhưng có hai yêu cầu đắt đỏ. Nó cần các episode phải kết thúc, và chỉ cập nhật sau khi đã có return cuối cùng. Nếu episode của bạn dài 1.000 bước, MC phải đợi 1.000 bước mới cập nhật được bất cứ thứ gì. Nó có phương sai cao, độ chệch thấp và chậm trong thực tế.

Dynamic programming có đặc điểm ngược lại — các bản sao lưu (backups) bootstrapped có phương sai bằng không — nhưng yêu cầu phải biết trước mô hình.

Temporal difference (TD) learning nằm ở giữa. Từ một bước chuyển `(s, a, r, s')`, tạo ra một mục tiêu một bước `r + γ V(s')` và điều chỉnh `V(s)` hướng về phía đó. Không cần mô hình. Không cần các episode hoàn chỉnh. Có độ chệch do sử dụng `V` xấp xỉ ở vế phải, nhưng phương sai thấp hơn đáng kể so với MC và cho phép cập nhật trực tuyến (online) ngay từ bước đầu tiên.

Đây là điểm tựa mà toàn bộ RL hiện đại — DQN, A2C, PPO, SAC — đều xoay quanh. Phần còn lại của Phase 9 là các lớp xấp xỉ hàm và các thủ thuật được xây dựng dựa trên bản cập nhật TD một bước mà bạn sẽ viết trong bài học này.

## Khái niệm

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**Bản cập nhật TD(0) cho V:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

Đại lượng trong ngoặc là TD error `δ = r + γ V(s') - V(s)`. Nó là phiên bản trực tuyến của `G_t - V(s_t)` trong MC. Sự hội tụ yêu cầu `α` thỏa mãn điều kiện Robbins-Monro (`Σ α = ∞`, `Σ α² < ∞`) và tất cả các trạng thái phải được ghé thăm vô số lần.

**Q-learning.** Một phương pháp TD off-policy cho điều khiển:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` giả định rằng chính sách *greedy* sẽ được tuân theo từ `s'` trở đi, bất kể tác nhân thực sự thực hiện hành động nào. Việc tách biệt này giúp Q-learning học được `Q*` trong khi tác nhân khám phá thông qua ε-greedy. Mnih và cộng sự (2015) đã chuyển đổi điều này thành deep Q-learning trên Atari (Bài học 05).

**SARSA.** Một phương pháp TD on-policy:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

Tên gọi này xuất phát từ bộ `(s, a, r, s', a')`. SARSA sử dụng hành động `a'` mà tác nhân *thực sự* thực hiện tiếp theo, không phải hành động greedy `argmax`. Hội tụ về `Q^π` cho bất kỳ chính sách ε-greedy `π` nào đang chạy, và trong giới hạn `ε → 0`, nó trở thành `Q*`.

**Sự khác biệt trong bài toán cliff-walking.** Trong bài toán cliff-walking kinh điển (rơi xuống vách đá = phần thưởng -100), Q-learning học được con đường tối ưu dọc theo mép vách đá nhưng thỉnh thoảng vẫn chịu phạt trong quá trình khám phá. SARSA học một con đường an toàn hơn, cách xa vách đá một bước vì nó tính đến nhiễu khám phá vào giá trị Q của mình. Với quá trình huấn luyện, cả hai đều đạt đến tối ưu tại `ε → 0`. Trong thực tế, điều này rất quan trọng: khi việc khám phá thực sự diễn ra trong quá trình triển khai, hành vi của SARSA sẽ thận trọng hơn.

**Expected SARSA.** Thay thế `Q(s', a')` bằng giá trị kỳ vọng của nó theo `π`:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

Phương sai thấp hơn SARSA (không lấy mẫu `a'`), mục tiêu on-policy tương tự. Thường là lựa chọn mặc định trong các giáo trình hiện đại.

**n-step TD và TD(λ).** Nội suy giữa TD(0) và MC bằng cách đợi `n` bước trước khi thực hiện bootstrapping. `n=1` là TD, `n=∞` là MC. TD(λ) lấy trung bình trên tất cả `n` với trọng số hình học `(1-λ)λ^{n-1}`. Hầu hết deep-RL sử dụng `n` trong khoảng từ 3 đến 20.

```figure
qlearning-gridworld
```

## Xây dựng

### Bước 1: SARSA trên chính sách ε-greedy

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

Tám dòng. Sự khác biệt *duy nhất* so với Q-learning là dòng mục tiêu.

### Bước 2: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` tách biệt mục tiêu khỏi hành vi. Ký hiệu đó chính là sự khác biệt giữa on-policy và off-policy.

### Bước 3: đường cong học tập (learning curves)

Theo dõi return trung bình trên mỗi 100 episode. Q-learning hội tụ nhanh hơn trên GridWorld đơn giản có tính tất định; SARSA thận trọng hơn trên cliff-walking. Trên GridWorld 4×4 trong `code/main.py`, cả hai đều đạt gần tối ưu sau khoảng 2.000 episode với `α=0.1, ε=0.1`.

### Bước 4: so sánh với kết quả DP

Chạy value iteration (Bài học 02) để có `Q*`. Kiểm tra `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`. Một tác nhân TD dạng bảng (tabular) tốt sẽ đạt được kết quả trong khoảng `~0.5` trên GridWorld 4×4 sau 10.000 episode.

## Các cạm bẫy

- **Giá trị Q khởi tạo rất quan trọng.** Khởi tạo lạc quan (`Q = 0` cho một tác vụ có phần thưởng âm) khuyến khích khám phá. Khởi tạo bi quan có thể bẫy một chính sách greedy mãi mãi.
- **Lịch trình α.** `α` hằng số là ổn cho các bài toán không dừng (non-stationary). `α_n = 1/n` giảm dần cho sự hội tụ về lý thuyết nhưng quá chậm trong thực tế — hãy cố định `α` trong `[0.05, 0.3]` và theo dõi đường cong học tập.
- **Lịch trình ε.** Bắt đầu cao (`ε=1.0`), giảm dần về `ε=0.05`. "GLIE" (greedy in the limit with infinite exploration) là điều kiện hội tụ.
- **Max bias trong Q-learning.** Toán tử `max` bị chệch lên trên khi `Q` có nhiễu. Dẫn đến đánh giá quá cao — Double Q-learning của Hasselt (được DDQN sử dụng trong Bài học 05) khắc phục điều này bằng hai bảng Q.
- **Episode không kết thúc.** TD có thể học mà không cần trạng thái kết thúc, nhưng bạn cần giới hạn số bước hoặc xử lý bootstrap chính xác tại điểm giới hạn. Tiêu chuẩn: coi giới hạn là không kết thúc, tiếp tục bootstrapping.
- **Hash trạng thái.** Nếu các trạng thái là tuple/tensor, hãy sử dụng khóa có thể hash (tuple, không phải list; tuple các số thực đã làm tròn, không phải giá trị thô).

## Sử dụng

Bối cảnh TD năm 2026:

| Tác vụ | Phương pháp | Lý do |
|------|--------|--------|
| Môi trường dạng bảng nhỏ | Q-learning | Học chính sách tối ưu trực tiếp. |
| On-policy quan trọng về an toàn | SARSA / Expected SARSA | Thận trọng trong quá trình khám phá. |
| Trạng thái chiều cao | DQN (Phase 9 · 05) | Q-function mạng thần kinh với replay và target net. |
| Hành động liên tục | SAC / TD3 (Phase 9 · 07) | Cập nhật TD trên Q-network; policy net xuất ra hành động. |
| LLM RL (dựa trên reward-model) | PPO / GRPO (Phase 9 · 08, 12) | Actor-critic với lợi thế kiểu TD thông qua GAE. |
| Offline RL | CQL / IQL (Phase 9 · 08) | Q-learning với chính quy hóa thận trọng. |

Chín mươi phần trăm "RL" bạn đọc trong các bài báo năm 2026 là sự mở rộng của Q-learning hoặc SARSA. Hãy hiểu rõ bản cập nhật dạng bảng trong lòng bàn tay trước khi đọc sâu hơn.

## Triển khai

Lưu dưới dạng `outputs/skill-td-agent.md`:

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## Bài tập

1. **Dễ.** Triển khai Q-learning và SARSA trên GridWorld 4×4. Vẽ đường cong học tập (return trung bình mỗi 100 episode) trong 2.000 episode. Phương pháp nào hội tụ nhanh hơn?
2. **Trung bình.** Xây dựng môi trường cliff-walking (4×12, hàng cuối là vách đá với phần thưởng -100 và reset về điểm bắt đầu). So sánh các chính sách cuối cùng của Q-learning và SARSA. Chụp ảnh các con đường mà mỗi phương pháp chọn. Phương pháp nào đi gần vách đá hơn?
3. **Khó.** Triển khai Double Q-learning. Trên GridWorld có phần thưởng nhiễu (nhiễu Gaussian σ=5 thêm vào phần thưởng mỗi bước), hãy chứng minh Q-learning đánh giá quá cao `V*(0,0)` một lượng đáng kể trong khi Double Q-learning thì không.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| TD error | "Tín hiệu cập nhật" | `δ = r + γ V(s') - V(s)`, phần dư bootstrapped. |
| TD(0) | "TD một bước" | Cập nhật sau mỗi bước chuyển chỉ sử dụng ước tính của trạng thái tiếp theo. |
| Q-learning | "RL off-policy cơ bản" | Cập nhật TD với `max` trên các hành động trạng thái tiếp theo; học `Q*` bất kể chính sách hành vi. |
| SARSA | "Q-learning on-policy" | Cập nhật TD sử dụng hành động tiếp theo thực tế; học `Q^π` cho π ε-greedy hiện tại. |
| Expected SARSA | "SARSA phương sai thấp" | Thay thế `a'` được lấy mẫu bằng kỳ vọng của nó theo π. |
| GLIE | "Lịch trình khám phá đúng" | Greedy in the Limit with Infinite Exploration; cần thiết cho sự hội tụ của Q-learning. |
| Bootstrapping | "Sử dụng ước tính hiện tại trong mục tiêu" | Điều phân biệt TD với MC. Nguồn gốc của độ chệch nhưng giảm phương sai đáng kể. |
| Maximization bias | "Q-learning đánh giá quá cao" | `max` trên các ước tính nhiễu bị chệch lên trên; được khắc phục bởi Double Q-learning. |

## Đọc thêm

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — bài báo gốc và chứng minh hội tụ.
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0), SARSA, Q-learning, Expected SARSA.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — khắc phục maximization bias.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — động lực của Expected SARSA.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — bài báo đặt tên cho SARSA (lúc đó gọi là "modified connectionist Q-learning").
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — tổng quát hóa TD(0) thành TD(n), con đường từ Q-learning đến eligibility traces và sau đó là GAE trong PPO.