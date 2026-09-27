# Multi-Agent RL

> RL đơn tác nhân (single-agent RL) giả định môi trường là tĩnh. Khi đặt hai tác nhân học tập vào cùng một thế giới, giả định đó bị phá vỡ: mỗi tác nhân là một phần trong môi trường của tác nhân kia, và cả hai đều đang thay đổi. Multi-agent RL là tập hợp các kỹ thuật giúp quá trình học hội tụ khi giả định Markov không còn đúng nữa.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~45 phút

## Vấn đề

Một robot học cách điều hướng trong phòng là bài toán RL đơn tác nhân. Một đội bóng đá thì không. AlphaStar đối đầu với các đối thủ trong StarCraft thì không. Một thị trường với các tác nhân đấu thầu thì không. Hai chiếc xe thương lượng tại ngã tư thì không. Nhiều bài toán thực tế với tương tác nhiều bên đều không phải là bài toán đơn tác nhân.

Trong mọi bối cảnh đa tác nhân, từ góc nhìn của bất kỳ tác nhân nào, các tác nhân khác *đều là* một phần của môi trường. Khi chúng học và thay đổi hành vi, môi trường trở nên không tĩnh (non-stationary). Tính chất Markov — "trạng thái tiếp theo chỉ phụ thuộc vào trạng thái hiện tại và hành động của tôi" — bị vi phạm vì trạng thái tiếp theo còn phụ thuộc vào những gì *các tác nhân khác* đã chọn, và chính sách của chúng là những mục tiêu di động.

Điều này phá vỡ các bằng chứng hội tụ dạng bảng (đảm bảo của Q-learning giả định một môi trường tĩnh). Nó cũng phá vỡ cả deep RL ngây thơ: các tác nhân đuổi theo nhau trong các vòng lặp và không bao giờ hội tụ về một chính sách ổn định. Bạn cần các kỹ thuật đặc thù cho đa tác nhân: huấn luyện tập trung / thực thi phi tập trung (centralized training / decentralized execution), các baseline phản thực tế (counterfactual baselines), league play, và self-play.

Các ứng dụng năm 2026: bầy đàn robot, điều phối giao thông, đội xe tự hành, trình mô phỏng thị trường, hệ thống LLM đa tác nhân (Phase 16), và bất kỳ trò chơi nào có nhiều hơn một người chơi thông minh.

## Khái niệm

![Four MARL regimes: indep, centralized critic, self-play, league](../assets/marl.svg)

**Hình thức hóa: Markov Game.** Một sự tổng quát hóa của MDP: các trạng thái `S`, một hành động chung `a = (a_1, …, a_n)`, chuyển đổi `P(s' | s, a)`, và phần thưởng cho mỗi tác nhân `R_i(s, a, s')`. Mỗi tác nhân `i` tối đa hóa lợi nhuận của riêng mình theo chính sách `π_i` của chính nó. Nếu phần thưởng giống hệt nhau, đó là **hợp tác hoàn toàn (fully cooperative)**. Nếu là tổng bằng không, đó là **đối kháng (adversarial)**. Nếu hỗn hợp, đó là **tổng quát (general-sum)**.

**Các thách thức cốt lõi:**

- **Tính không tĩnh (Non-stationarity).** `P(s' | s, a_i)` từ góc nhìn của tác nhân `i` phụ thuộc vào `π_{-i}`, vốn đang thay đổi.
- **Phân bổ tín dụng (Credit assignment).** Với phần thưởng chia sẻ, tác nhân nào đã tạo ra nó?
- **Phối hợp khám phá (Exploration coordination).** Các tác nhân phải khám phá các chiến lược bổ sung cho nhau, thay vì khám phá trùng lặp cùng một trạng thái.
- **Khả năng mở rộng (Scalability).** Không gian hành động chung tăng theo cấp số nhân với `n`.
- **Khả năng quan sát một phần (Partial observability).** Mỗi tác nhân chỉ nhìn thấy quan sát của riêng mình; trạng thái toàn cục bị ẩn.

**Bốn chế độ thống trị:**

**1. Independent Q-learning / independent PPO (IQL, IPPO).** Mỗi tác nhân học Q hoặc chính sách của riêng mình, coi những tác nhân khác là một phần của môi trường. Đơn giản, đôi khi hiệu quả (đặc biệt với experience replay đóng vai trò như một thủ thuật làm mượt mô hình tác nhân). Hội tụ lý thuyết: không có. Trong thực tế: ổn với các tác vụ liên kết lỏng lẻo, tệ với các tác vụ liên kết chặt chẽ.

**2. Huấn luyện tập trung, thực thi phi tập trung (CTDE).** Mô hình hiện đại phổ biến nhất. Mỗi tác nhân có *chính sách* `π_i` riêng dựa trên quan sát cục bộ `o_i` — thực thi phi tập trung tiêu chuẩn khi triển khai. Trong quá trình *huấn luyện*, một critic tập trung `Q(s, a_1, …, a_n)` dựa trên trạng thái toàn cục đầy đủ và hành động chung. Ví dụ:
- **MADDPG** (Lowe và cộng sự 2017): DDPG với một critic tập trung cho mỗi tác nhân.
- **COMA** (Foerster và cộng sự 2017): baseline phản thực tế — đặt câu hỏi "phần thưởng của tôi sẽ là bao nhiêu nếu tôi thực hiện hành động `a'` thay thế?" — giúp cô lập đóng góp của cá nhân.
- **MAPPO** / **IPPO** với critic chia sẻ (Yu và cộng sự 2022): PPO với hàm giá trị tập trung. Thống trị vào năm 2026 cho MARL hợp tác.
- **QMIX** (Rashid và cộng sự 2018): phân rã giá trị — `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` với trộn đơn điệu (monotonic mixing).

**3. Self-play.** Hai bản sao của cùng một tác nhân chơi với nhau. Chính sách của đối thủ *chính là* chính sách của tôi từ một snapshot trong quá khứ. AlphaGo / AlphaZero / MuZero. OpenAI Five. Hoạt động tốt nhất cho các trò chơi tổng bằng không; tín hiệu huấn luyện là đối xứng.

**4. League play.** Một phần mở rộng của self-play cho các môi trường tổng quát / đối kháng: duy trì một quần thể các chính sách quá khứ và hiện tại, lấy mẫu đối thủ từ giải đấu, huấn luyện chống lại họ. Thêm các "exploiter" (chuyên đánh bại người giỏi nhất hiện tại) và "main exploiter" (chuyên đánh bại các exploiter). AlphaStar (StarCraft II). Cần thiết khi trò chơi có các chu kỳ chiến lược kiểu "kéo-búa-bao".

**Giao tiếp.** Cho phép các tác nhân gửi các thông điệp đã học `m_i` cho nhau. Hoạt động trong các bối cảnh hợp tác. Foerster và cộng sự (2016) đã chỉ ra rằng giao tiếp giữa các tác nhân có thể vi phân (differentiable) có thể được huấn luyện end-to-end. Ngày nay, các hệ thống đa tác nhân dựa trên LLM (Phase 16) về cơ bản giao tiếp bằng ngôn ngữ tự nhiên.

```figure
f3-marl-orbit
```

## Xây dựng

Bài học này sử dụng GridWorld 6×6 với hai tác nhân hợp tác. Chúng bắt đầu ở các góc đối diện và phải đạt được mục tiêu chung. Phần thưởng chia sẻ: `-1` mỗi bước khi một trong hai tác nhân vẫn đang di chuyển, `+10` khi cả hai cùng đến nơi. Xem `code/main.py`.

### Bước 1: môi trường đa tác nhân

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

Không gian hành động *chung* là `|A|² = 16`. Trạng thái toàn cục là hai vị trí.

### Bước 2: independent Q-learning

Mỗi tác nhân chạy bảng Q riêng dựa trên trạng thái chung. Tại mỗi bước: cả hai chọn hành động ε-greedy, thu thập chuyển đổi chung, mỗi tác nhân cập nhật Q của riêng mình với phần thưởng chia sẻ.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

Hoạt động trên tác vụ này vì phần thưởng dày đặc và đồng nhất. Thất bại trên các tác vụ liên kết chặt chẽ (ví dụ: nơi một tác nhân phải *đợi* tác nhân kia).

### Bước 3: Q tập trung với cập nhật phân rã giá trị

Sử dụng một Q trên các hành động chung `Q(s, a_1, a_2)`. Cập nhật từ phần thưởng chia sẻ. Phi tập trung hóa khi thực thi bằng cách biên hóa (marginalizing): `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. Đánh đổi không gian hành động chung theo cấp số nhân để lấy một cái nhìn toàn cục *chính xác*.

### Bước 4: self-play đơn giản (đối kháng 2 tác nhân)

Cùng một tác nhân, hai vai trò. Huấn luyện tác nhân A chống lại tác nhân B; sau `K` tập, sao chép trọng số của A vào B. Huấn luyện đối xứng, tiến bộ nhất quán. Công thức AlphaZero ở quy mô nhỏ.

## Các cạm bẫy

- **Replay không tĩnh.** Experience replay với các tác nhân độc lập tệ hơn so với đơn tác nhân vì các chuyển đổi cũ được tạo ra bởi các đối thủ hiện đã lỗi thời. Khắc phục: dán nhãn lại hoặc trọng số theo độ gần đây.
- **Sự mơ hồ trong phân bổ tín dụng.** Phần thưởng chia sẻ sau một tập dài; không có cách rõ ràng để nói tác nhân nào đã đóng góp. Khắc phục: baseline phản thực tế (COMA), hoặc định hình phần thưởng (reward shaping) cho mỗi tác nhân.
- **Trôi chính sách / đuổi theo.** Phản ứng tốt nhất của mỗi tác nhân thay đổi theo mỗi cập nhật của tác nhân kia. Khắc phục: critic tập trung, tốc độ học chậm, hoặc đóng băng từng tác nhân một.
- **Hack phần thưởng thông qua phối hợp.** Các tác nhân tìm thấy các khai thác phối hợp mà người thiết kế không lường trước được. Các tác nhân đấu giá hội tụ về mức giá thầu bằng không. Khắc phục: thiết kế phần thưởng cẩn thận, các ràng buộc hành vi.
- **Trùng lặp khám phá.** Cả hai tác nhân đều khám phá cùng các cặp trạng thái-hành động. Khắc phục: thưởng entropy cho mỗi tác nhân, hoặc điều kiện hóa vai trò (role-conditioning).
- **Chu kỳ giải đấu.** Self-play thuần túy có thể bị mắc kẹt trong một chu kỳ thống trị. Khắc phục: league play với các đối thủ đa dạng.
- **Bùng nổ mẫu.** `n` tác nhân × không gian trạng thái × hành động chung. Xấp xỉ bằng xấp xỉ hàm; không gian hành động phân tách (một đầu ra chính sách cho mỗi tác nhân).

## Sử dụng

Bản đồ ứng dụng MARL năm 2026:

| Lĩnh vực | Phương pháp | Ghi chú |
|--------|--------|-------|
| Điều hướng / thao tác hợp tác | MAPPO / QMIX | CTDE; critic chia sẻ + các actor phi tập trung. |
| Trò chơi hai người (cờ vua, Go, poker) | Self-play với MCTS (AlphaZero) | Tổng bằng không; huấn luyện đối xứng. |
| Nhiều người chơi phức tạp (Dota, StarCraft) | League play + tiền huấn luyện bắt chước | OpenAI Five, AlphaStar. |
| Đội xe tự hành | CTDE MAPPO / PPO với attention | Quan sát một phần; quy mô đội thay đổi. |
| Thị trường đấu giá | Cân bằng lý thuyết trò chơi + RL | Mean-field RL khi `n` → ∞. |
| Hệ thống đa tác nhân LLM (Phase 16) | Giao tiếp ngôn ngữ tự nhiên + điều kiện hóa vai trò | Vòng lặp RL ở lớp lập kế hoạch tác nhân. |

Vào năm 2026, lĩnh vực tăng trưởng lớn nhất của MARL là dựa trên LLM: các bầy đàn tác nhân mô hình ngôn ngữ thương lượng, tranh luận, xây dựng phần mềm. RL xuất hiện dưới dạng tối ưu hóa ưu tiên trên các đầu ra *cấp quỹ đạo*, không phải cấp token (Phase 16 · 03).

## Triển khai

Lưu dưới dạng `outputs/skill-marl-architect.md`:

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## Bài tập

1. **Dễ.** Huấn luyện independent Q-learning trên GridWorld hợp tác 2 tác nhân. Cần bao nhiêu tập để lợi nhuận trung bình > 0? Vẽ đường cong học tập chung.
2. **Trung bình.** Thêm một tác vụ "phối hợp": mục tiêu chỉ đạt được khi cả hai tác nhân bước vào đó cùng một lượt. Independent Q có hội tụ không? Điều gì bị phá vỡ?
3. **Khó.** Triển khai một critic tập trung cho huấn luyện kiểu MAPPO và so sánh tốc độ hội tụ với independent PPO trên tác vụ phối hợp.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Markov game | "Multi-agent MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; mỗi tác nhân có phần thưởng riêng. |
| CTDE | "Centralized training, decentralized execution" | Critic chung tại thời điểm huấn luyện; chính sách của mỗi tác nhân chỉ dùng quan sát cục bộ. |
| IPPO | "Independent PPO" | Mỗi tác nhân chạy PPO riêng biệt. Baseline đơn giản; thường bị đánh giá thấp. |
| MAPPO | "Multi-agent PPO" | PPO với hàm giá trị tập trung dựa trên trạng thái toàn cục. |
| QMIX | "Monotonic value decomposition" | `Q_tot = f_monotone(Q_1, …, Q_n)` cho phép argmax phi tập trung. |
| COMA | "Counterfactual multi-agent" | Advantage = Q của tôi trừ đi Q kỳ vọng khi biên hóa hành động của tôi. |
| Self-play | "Agent vs past self" | Một tác nhân, hai vai trò; tiêu chuẩn cho trò chơi tổng bằng không. |
| League play | "Population training" | Lưu trữ các chính sách quá khứ, lấy mẫu đối thủ từ nhóm; xử lý các chu kỳ chiến lược. |

## Đọc thêm

- [Lowe và cộng sự (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — CTDE với critic tập trung.
- [Foerster và cộng sự (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — baseline phản thực tế cho phân bổ tín dụng.
- [Rashid và cộng sự (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — phân rã giá trị với tính đơn điệu.
- [Yu và cộng sự (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO mạnh mẽ đáng ngạc nhiên cho MARL.
- [Vinyals và cộng sự (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — league play ở quy mô lớn.
- [Silver và cộng sự (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — self-play thuần túy trong trò chơi tổng bằng không.
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — bao gồm phần xử lý ngắn của sách giáo khoa về bối cảnh đa tác nhân và vấn đề không tĩnh mà CTDE được thiết kế để giải quyết.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — khảo sát bao gồm MARL hợp tác, cạnh tranh và hỗn hợp với các kết quả hội tụ.