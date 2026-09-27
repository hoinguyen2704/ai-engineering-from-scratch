# Deep Q-Networks (DQN)

> 2013: Mnih đã huấn luyện một mạng Q-learning trên các pixel thô, đánh bại mọi tác nhân RL cổ điển trên bảy trò chơi Atari. 2015: mở rộng lên 49 trò chơi, được xuất bản trên Nature, khơi nguồn cho kỷ nguyên deep-RL. DQN là Q-learning cộng với ba thủ thuật giúp cho việc xấp xỉ hàm trở nên ổn định.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75 phút

## Vấn đề

Tabular Q-learning cần một giá trị Q riêng biệt cho mỗi cặp (trạng thái, hành động). Một bàn cờ vua có khoảng 10⁴³ trạng thái. Một khung hình Atari là 210×160×3 = 100,800 đặc trưng. Tabular RL sẽ thất bại ở quy mô hàng nghìn trạng thái, chứ chưa nói đến hàng tỷ.

Giải pháp khá rõ ràng khi nhìn lại: thay thế bảng Q bằng một mạng thần kinh, `Q(s, a; θ)`. Nhưng cái "rõ ràng khi nhìn lại" đó đã mất hàng thập kỷ. Việc xấp xỉ hàm ngây thơ với Q-learning sẽ phân kỳ dưới "bộ ba chết chóc" (deadly triad) — xấp xỉ hàm + bootstrapping + học off-policy. Mnih và cộng sự (2013, 2015) đã xác định ba thủ thuật kỹ thuật giúp ổn định quá trình học:

1. **Experience replay** giúp loại bỏ sự tương quan giữa các chuyển đổi.
2. **Target network** giúp đóng băng mục tiêu bootstrap.
3. **Reward clipping** giúp chuẩn hóa độ lớn của gradient.

DQN trên Atari là lần đầu tiên một kiến trúc duy nhất với một bộ siêu tham số duy nhất giải quyết được hàng chục bài toán điều khiển từ các pixel thô. Mọi thứ mà "deep-RL" xây dựng kể từ đó — DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57 — đều được xây dựng dựa trên nền tảng ba thủ thuật này.

## Khái niệm

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**Mục tiêu.** DQN tối thiểu hóa hàm mất mát TD một bước trên một hàm Q thần kinh:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = mạng online, được cập nhật mỗi bước bằng gradient descent. `θ^-` = mạng mục tiêu (target network), được sao chép định kỳ từ `θ` (mỗi ~10,000 bước). `D` = bộ đệm phát lại (replay buffer) chứa các chuyển đổi trong quá khứ.

**Ba thủ thuật, theo thứ tự quan trọng:**

**Experience replay.** Một vòng đệm (ring buffer) chứa `~10⁶` chuyển đổi. Mỗi bước huấn luyện sẽ lấy mẫu một minibatch ngẫu nhiên đồng nhất. Điều này phá vỡ sự tương quan theo thời gian (các khung hình liên tiếp gần như giống hệt nhau), cho phép mạng học từ các chuyển đổi hiếm hoi có phần thưởng nhiều lần, và loại bỏ sự tương quan giữa các cập nhật gradient liên tiếp. Nếu không có nó, on-policy TD với một mạng thần kinh sẽ phân kỳ trên Atari.

**Target network.** Việc sử dụng cùng một mạng `Q(·; θ)` ở cả hai phía của phương trình Bellman khiến mục tiêu thay đổi sau mỗi lần cập nhật — giống như "đuổi theo cái đuôi của chính mình". Giải pháp: giữ một mạng thứ hai `Q(·; θ^-)` với trọng số đóng băng. Mỗi `C` bước, sao chép `θ → θ^-`. Điều này ổn định mục tiêu hồi quy trong hàng nghìn bước gradient mỗi lần. Soft updates `θ^- ← τ θ + (1-τ) θ^-` (được sử dụng trong DDPG, SAC) là một biến thể mượt mà hơn.

**Reward clipping.** Độ lớn phần thưởng của Atari thay đổi từ 1 đến hơn 1000. Việc cắt (clipping) về `{-1, 0, +1}` ngăn chặn bất kỳ trò chơi đơn lẻ nào làm chủ gradient. Điều này sai nếu độ lớn phần thưởng quan trọng; nhưng lại ổn với Atari nơi chỉ có dấu của phần thưởng là quan trọng.

**Double DQN.** Hasselt (2016) khắc phục độ chệch tối đa (maximization bias): sử dụng mạng online để *chọn* hành động, mạng mục tiêu để *đánh giá* hành động đó.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

Đây là sự thay thế trực tiếp và nhất quán tốt hơn. Hãy sử dụng nó theo mặc định.

**Các cải tiến khác (Rainbow, 2017):** prioritized replay (lấy mẫu các chuyển đổi có TD-error cao nhiều hơn), kiến trúc dueling (tách biệt các đầu ra `V(s)` và advantage), noisy networks (khám phá được học), n-step returns, distributional Q (C51/QR-DQN), multi-step bootstrapping. Mỗi cải tiến thêm vài phần trăm; các lợi ích này cộng dồn với nhau.

```figure
f3-dqn-stability
```

## Xây dựng

Mã nguồn ở đây chỉ sử dụng thư viện chuẩn, không dùng numpy — chúng ta sử dụng một MLP một lớp ẩn tự viết trên một GridWorld liên tục nhỏ, vì vậy mỗi bước huấn luyện chỉ chạy trong vài micro giây. Thuật toán này giống hệt với Atari DQN ở quy mô lớn.

### Bước 1: replay buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

~50,000 dung lượng cho Atari; 5,000 là đủ cho môi trường đồ chơi của chúng ta.

### Bước 2: một mạng Q nhỏ (MLP thủ công)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Forward pass: linear → ReLU → linear. Đó là toàn bộ mạng.

### Bước 3: cập nhật DQN

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

Hình dạng của nó là Q-learning từ Bài 04 với hai khác biệt: (a) chúng ta thực hiện backprop qua một `Q(·; θ)` có thể vi phân thay vì lập chỉ mục một bảng, (b) mục tiêu sử dụng `Q(·; θ^-)`.

### Bước 4: vòng lặp ngoài

Đối với mỗi tập (episode), thực hiện ε-greedy trên `Q(·; θ)`, đẩy các chuyển đổi vào bộ đệm, lấy mẫu một minibatch, thực hiện một bước gradient, đồng bộ định kỳ `θ^- ← θ`. Mô hình:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

Trên GridWorld nhỏ của chúng ta với trạng thái one-hot 16 chiều, tác nhân học được chính sách gần tối ưu trong khoảng 500 tập. Trên Atari, hãy mở rộng quy mô này lên 200 triệu khung hình và thêm một bộ trích xuất đặc trưng CNN.

## Các cạm bẫy

- **Deadly triad.** Xấp xỉ hàm + off-policy + bootstrapping có thể phân kỳ. DQN giảm thiểu điều này bằng target net + replay; đừng loại bỏ bất kỳ cái nào.
- **Exploration.** ε phải giảm dần, thường từ 1.0 xuống 0.01 trong khoảng 10% đầu tiên của quá trình huấn luyện. Nếu không khám phá đủ sớm, Q-net sẽ hội tụ vào một lưu vực cục bộ.
- **Overestimation.** `max` trên Q nhiễu bị chệch lên trên. Luôn sử dụng Double DQN trong môi trường thực tế.
- **Reward scale.** Cắt hoặc chuẩn hóa phần thưởng; độ lớn gradient tỷ lệ thuận với độ lớn phần thưởng.
- **Replay buffer coldstart.** Đừng huấn luyện cho đến khi bộ đệm có vài nghìn chuyển đổi. Các gradient sớm trên khoảng 20 mẫu sẽ gây ra overfitting.
- **Tần suất đồng bộ mục tiêu.** Quá thường xuyên ≈ không có target net; quá thưa thớt ≈ mục tiêu cũ. Atari DQN sử dụng 10,000 bước môi trường. Quy tắc ngón tay cái: đồng bộ mỗi ~1/100 thời gian huấn luyện.
- **Tiền xử lý quan sát.** Atari DQN xếp chồng 4 khung hình để tạo trạng thái Markov. Bất kỳ môi trường nào có thông tin vận tốc đều cần xếp chồng khung hình hoặc trạng thái tái phát (recurrent state).

## Sử dụng

Vào năm 2026, DQN hiếm khi là thuật toán hiện đại nhất nhưng vẫn là thuật toán off-policy tham chiếu:

| Tác vụ | Phương pháp lựa chọn | Tại sao không phải DQN? |
|------|------------------|--------------|
| Discrete-action giống Atari | Rainbow DQN hoặc Muesli | Cùng khung, nhiều thủ thuật hơn. |
| Điều khiển liên tục | SAC / TD3 (Phase 9 · 07) | DQN không có mạng chính sách. |
| On-policy / thông lượng cao | PPO (Phase 9 · 08) | Không có replay buffer; dễ mở rộng hơn. |
| Offline RL | CQL / IQL / Decision Transformer | Mục tiêu Q bảo thủ, không bị bùng nổ bootstrapping. |
| Không gian hành động rời rạc lớn (recommender) | DQN với action embedding, hoặc IMPALA | Tốt; việc trang trí (decoration) quan trọng. |
| LLM RL | PPO / GRPO | Cấp độ chuỗi, không phải cấp độ bước; hàm mất mát khác. |

Các bài học vẫn còn giá trị. Replay và target network xuất hiện trong SAC, TD3, DDPG, SAC-X, bộ đệm tự chơi của AlphaZero và mọi phương pháp offline RL. Reward clipping vẫn tồn tại dưới dạng chuẩn hóa lợi thế (advantage normalization) trong PPO. Kiến trúc này chính là bản thiết kế.

## Triển khai

Lưu dưới dạng `outputs/skill-dqn-trainer.md`:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Vẽ đường cong lợi nhuận theo từng tập. Cần bao nhiêu tập để giá trị trung bình vượt quá -10?
2. **Trung bình.** Vô hiệu hóa target network (sử dụng mạng online cho cả hai phía của mục tiêu Bellman). Đo lường sự mất ổn định của quá trình huấn luyện — lợi nhuận có dao động hoặc phân kỳ không?
3. **Khó.** Thêm Double DQN: sử dụng mạng online để chọn `argmax a'`, mạng mục tiêu để đánh giá. So sánh độ chệch của `Q(s_0, best_a)` so với `V*(s_0)` thực tế sau 1,000 tập với và không có Double DQN trên một GridWorld có phần thưởng nhiễu.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| DQN | "Deep Q-learning" | Q-learning với hàm Q thần kinh, replay buffer và target network. |
| Experience replay | "Chuyển đổi xáo trộn" | Vòng đệm được lấy mẫu đồng nhất mỗi bước gradient; loại bỏ tương quan dữ liệu. |
| Target network | "Bootstrap đóng băng" | Bản sao định kỳ của Q được sử dụng trong mục tiêu Bellman; ổn định huấn luyện. |
| Deadly triad | "Tại sao RL phân kỳ" | Xấp xỉ hàm + bootstrapping + off-policy = không đảm bảo hội tụ. |
| Double DQN | "Sửa lỗi chệch tối đa" | Mạng online chọn hành động, mạng mục tiêu đánh giá nó. |
| Dueling DQN | "Các đầu V và A" | Phân tách Q = V + A - mean(A); cùng đầu ra, luồng gradient tốt hơn. |
| Rainbow | "Tất cả các thủ thuật" | DDQN + PER + dueling + n-step + noisy + distributional trong một. |
| PER | "Prioritized Replay" | Lấy mẫu chuyển đổi tỷ lệ thuận với độ lớn TD-error. |

## Đọc thêm

- [Mnih và cộng sự (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — bài báo hội thảo NeurIPS 2013 khởi đầu cho deep RL.
- [Mnih và cộng sự (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — bài báo trên Nature, DQN 49 trò chơi.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN.
- [Wang và cộng sự (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN.
- [Hessel và cộng sự (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — bài báo về các thủ thuật xếp chồng.
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — giải thích hiện đại rõ ràng.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — giáo trình xử lý "bộ ba chết chóc" (xấp xỉ hàm + bootstrapping + off-policy) mà target network và replay buffer của DQN được thiết kế để thuần hóa.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — DQN tham chiếu một tệp được sử dụng trong các nghiên cứu ablation; nên đọc cùng với phiên bản tự viết của bài học này.