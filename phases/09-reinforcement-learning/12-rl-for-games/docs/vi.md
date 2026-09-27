# RL cho Games — AlphaZero, MuZero và Kỷ nguyên LLM-Reasoning

> 1992: TD-Gammon đánh bại các nhà vô địch con người ở trò backgammon bằng TD thuần túy. 2016: AlphaGo đánh bại Lee Sedol. 2017: AlphaZero thống trị cờ vua, shogi và cờ vây từ con số không. 2024: DeepSeek-R1 chứng minh công thức tương tự, với GRPO thay thế PPO, hoạt động hiệu quả trên khả năng suy luận. Games là thước đo thúc đẩy mọi đột phá trong giai đoạn này.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**Time:** ~120 phút

## Vấn đề

Games có mọi thứ mà RL cần. Phần thưởng rõ ràng (thắng/thua). Các tập (episodes) vô hạn (tự chơi lại). Mô phỏng hoàn hảo (trò chơi *chính là* trình mô phỏng). Không gian hành động rời rạc hoặc liên tục nhỏ. Cấu trúc đa tác nhân (multi-agent) buộc phải có sự mạnh mẽ đối kháng (adversarial robustness).

Và games là nơi mọi đột phá lớn của RL được kiểm chứng. TD-Gammon (backgammon, 1992). Atari-DQN (2013). AlphaGo (2016). AlphaZero (2017). OpenAI Five (Dota 2, 2019). AlphaStar (StarCraft II, 2019). MuZero (mô hình học được, 2019). AlphaTensor (phép nhân ma trận, 2022). AlphaDev (thuật toán sắp xếp, 2023). DeepSeek-R1 (suy luận toán học, 2025) — minh chứng mới nhất cho thấy các kỹ thuật RL trong game hoạt động hiệu quả trên văn bản.

Capstone này khảo sát ba kiến trúc mang tính bước ngoặt — AlphaZero, MuZero và GRPO — thông qua một lăng kính thống nhất: **tự chơi (self-play) + tìm kiếm (search) + cải thiện chính sách (policy improvement)**. Mỗi kiến trúc đều tổng quát hóa kiến trúc trước đó; đặc biệt GRPO là công thức của AlphaZero được áp dụng cho suy luận LLM, với các token là hành động và xác minh toán học là tín hiệu thắng.

## Khái niệm

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**Vòng lặp thống nhất.**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017).** Silver và cộng sự. Với một trò chơi (cờ vua, shogi, cờ vây) có luật chơi đã biết:

- Mạng policy-value: một tháp `f_θ(s) → (p, v)`. `p` là phân phối xác suất tiên nghiệm (prior) trên các nước đi hợp lệ. `v` là kết quả trò chơi dự kiến.
- Monte Carlo Tree Search (MCTS): tại mỗi nước đi, mở rộng một cây các khả năng tiếp theo. Sử dụng `(p, v)` làm tiên nghiệm + bootstrap. Chọn các nút bằng UCB (PUCT): `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`.
- Tự chơi (Self-play): tác nhân đấu với tác nhân. Tại nước đi `t`, phân phối lượt truy cập MCTS `π_t` trở thành mục tiêu huấn luyện chính sách.
- Hàm mất mát: `L = (v - z)² - π · log p + c · ||θ||²`. `z` là kết quả trò chơi (+1 / 0 / -1).

Không kiến thức con người. Không heuristic thủ công. Một công thức duy nhất đã làm chủ cờ vua, shogi và cờ vây sau vài chục triệu ván tự chơi mỗi loại.

**MuZero (2019).** Schrittwieser và cộng sự. Loại bỏ yêu cầu phải biết luật chơi.

- Thay vì một môi trường cố định, hãy học một *mô hình động lực học tiềm ẩn (latent dynamics model)* `(h, g, f)`:
  - `h(s)`: mã hóa quan sát thành trạng thái tiềm ẩn.
  - `g(s_latent, a)`: dự đoán trạng thái tiềm ẩn tiếp theo + phần thưởng.
  - `f(s_latent)`: dự đoán chính sách tiên nghiệm + giá trị.
- MCTS chạy trong *không gian tiềm ẩn đã học*. Cùng một cách tìm kiếm, cùng một vòng lặp huấn luyện.
- Hoạt động trên cờ vây, cờ vua, shogi *và* Atari — một thuật toán, không cần biết luật.

**Stochastic MuZero (2022).** Thêm động lực học ngẫu nhiên và các nút cơ hội; mở rộng sang các trò chơi loại backgammon.

**Muesli, Gumbel MuZero (2022-2024).** Cải tiến về hiệu quả lấy mẫu và tìm kiếm tất định.

**GRPO (2024-2025).** Công thức của DeepSeek-R1. Cùng một vòng lặp theo kiểu AlphaZero, áp dụng cho suy luận mô hình ngôn ngữ:

- "Game": trả lời một bài toán/lập trình/suy luận. "Thắng" = trình xác minh (test case vượt qua, đáp án số khớp) trả về 1.
- Policy: LLM. Hành động: các token. Trạng thái: prompt + phản hồi cho đến hiện tại.
- Không có critic (kiểu PPO V_φ). Thay vào đó, với mỗi prompt, lấy mẫu `G` phản hồi từ policy. Tính phần thưởng cho mỗi phản hồi. Sử dụng **lợi thế tương đối theo nhóm (group-relative advantage)** `A_i = (r_i - mean_r) / std_r` làm tín hiệu cho cập nhật kiểu REINFORCE.
- Phạt KL với chính sách tham chiếu để ngăn chặn trôi dạt (giống RLHF).
- Hàm mất mát đầy đủ:

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

Không mô hình phần thưởng, không critic, không MCTS. Đường cơ sở tương đối theo nhóm thay thế cả ba. Đạt hoặc vượt chất lượng PPO-RLHF trên các benchmark suy luận với chi phí tính toán thấp hơn nhiều.

**Công thức R1 đầy đủ.** DeepSeek-R1 (DeepSeek 2025) là hai mô hình trong cùng một bài báo:

- **R1-Zero.** Bắt đầu từ mô hình cơ sở DeepSeek-V3. Không SFT. Áp dụng GRPO trực tiếp với hai thành phần phần thưởng: *phần thưởng độ chính xác* (dựa trên luật — đáp án cuối cùng có phân tích ra số đúng không / code có vượt qua unit test không) và *phần thưởng định dạng* (phản hồi có bao bọc chuỗi suy nghĩ trong thẻ `<think>…</think>` không). Qua hàng ngàn bước, độ dài phản hồi trung bình tăng từ ~100 lên ~10,000 token và điểm benchmark toán học leo lên mức gần bằng o1-preview. Mô hình học cách suy luận từ con số không. Nhược điểm: chuỗi suy nghĩ thường khó đọc, trộn lẫn ngôn ngữ và thiếu sự trau chuốt về phong cách.
- **R1.** Khắc phục các vấn đề về khả năng đọc của R1-Zero với quy trình bốn giai đoạn:
  1. **Cold-start SFT.** Thu thập vài ngàn bản trình diễn CoT dài với định dạng sạch. Supervised-finetune mô hình cơ sở trên đó. Điều này tạo ra điểm khởi đầu dễ đọc.
  2. **GRPO định hướng suy luận.** Áp dụng GRPO với phần thưởng độ chính xác + định dạng cộng thêm phần thưởng *nhất quán ngôn ngữ* để ngăn chặn việc trộn ngôn ngữ.
  3. **Lấy mẫu từ chối + SFT vòng 2.** Lấy mẫu ~600K quỹ đạo suy luận từ checkpoint RL, chỉ giữ lại những quỹ đạo có đáp án cuối đúng và CoT dễ đọc, kết hợp với ~200K ví dụ SFT không suy luận (viết lách, QA, tự nhận thức). Fine-tune lại mô hình cơ sở.
  4. **GRPO toàn phổ.** Thêm một vòng RL nữa bao gồm cả suy luận (phần thưởng dựa trên luật) và căn chỉnh chung (phần thưởng dựa trên sở thích về sự hữu ích/vô hại).

Kết quả đạt mức o1 trên AIME và MATH-500 với trọng số mở, và đủ nhỏ để chưng cất (distill). Bài báo cũng phát hành sáu mô hình dày đặc đã chưng cất (Qwen-1.5B đến Llama-70B) bằng cách SFT trên các dấu vết suy luận của R1 — không dùng RL ở mô hình học sinh. Việc chưng cất từ một giáo viên RL mạnh luôn vượt trội hơn so với RL từ đầu ở quy mô của học sinh.

**Tại sao GRPO thay vì PPO cho suy luận.** Ba lý do trong bài báo DeepSeekMath (tháng 2/2024): (1) không cần huấn luyện mạng giá trị, giảm một nửa bộ nhớ; (2) đường cơ sở nhóm xử lý tự nhiên phần thưởng thưa thớt ở cuối quỹ đạo mà các tác vụ suy luận tạo ra; (3) chuẩn hóa theo từng prompt giúp các lợi thế có thể so sánh được giữa các bài toán có độ khó khác nhau, điều mà critic đơn lẻ của PPO không làm được.

**Không tìm kiếm vs Dựa trên tìm kiếm.** Games đã phân nhánh:

- *Trò chơi thông tin hoàn hảo với chân trời dài* (cờ vây, cờ vua): vẫn dựa trên tìm kiếm. AlphaZero / MuZero thống trị.
- *Suy luận LLM*: chưa có MCTS trong sản xuất; GRPO trên các lượt chạy đầy đủ, best-of-N cho tính toán suy luận. Các mô hình phần thưởng quy trình (PRM) gợi ý rằng tìm kiếm theo từng bước đang được thêm trở lại.

```figure
f3-selfplay-ladder
```

## Xây dựng

Mã trong `code/main.py` triển khai **GRPO thu nhỏ** — một bandit với nhiều nhóm mẫu. Thuật toán giống hệt như trên LLM; chỉ có policy và môi trường là đơn giản hơn. Nó dạy về *hàm mất mát* và *lợi thế tương đối theo nhóm*, vốn là sự đổi mới của năm 2025.

### Bước 1: môi trường xác minh nhỏ

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

Trong GRPO thực tế, trình xác minh chạy unit test hoặc kiểm tra đẳng thức toán học.

### Bước 2: policy: softmax trên K token trả lời mỗi prompt

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

Tương đương với đầu ra lớp cuối của một LLM được điều kiện hóa trên một prompt.

### Bước 3: lấy mẫu nhóm và lợi thế tương đối theo nhóm

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Lợi thế tương đối theo nhóm là thủ thuật của DeepSeek năm 2024. Không cần critic. "Đường cơ sở" là trung bình nhóm, và chuẩn hóa sử dụng độ lệch chuẩn nhóm.

### Bước 4: so sánh với đường cơ sở REINFORCE (không giá trị)

Cùng thiết lập, cùng tính toán, REINFORCE thuần túy. GRPO hội tụ nhanh hơn và ổn định hơn.

### Bước 5: quan sát entropy và KL

Các chẩn đoán giống như RLHF: KL trung bình so với tham chiếu, entropy chính sách, phần thưởng theo thời gian. Khi các chỉ số này ổn định, quá trình huấn luyện hoàn tất.

## Các cạm bẫy

- **Hack phần thưởng thông qua chơi game xác minh.** GRPO kế thừa rủi ro của RLHF: nếu trình xác minh sai hoặc có thể khai thác, LLM sẽ tìm ra lỗ hổng. Các trình xác minh mạnh mẽ (nhiều test case, chứng minh hình thức) rất quan trọng.
- **Kích thước nhóm quá nhỏ.** Phương sai của đường cơ sở nhóm đi theo `1/√G`. Dưới `G = 4`, tín hiệu lợi thế bị nhiễu; lựa chọn tiêu chuẩn là `G = 8` đến `64`.
- **Thiên kiến độ dài.** Các phản hồi LLM có độ dài khác nhau có log-xác suất khác nhau. Hãy chuẩn hóa theo số lượng token, hoặc sử dụng log-prob cấp chuỗi, hoặc cắt bớt đến độ dài tối đa.
- **Chu kỳ tự chơi thuần túy.** Huấn luyện kiểu AlphaZero có thể bị mắc kẹt trong các vòng lặp thống trị trên các trò chơi tổng quát. Được giảm thiểu bằng các nhóm đối thủ đa dạng (league play, Bài 10).
- **Không khớp giữa tìm kiếm và chính sách.** AlphaZero huấn luyện chính sách để bắt chước đầu ra tìm kiếm. Nếu mạng chính sách quá nhỏ để đại diện cho phân phối của tìm kiếm, quá trình huấn luyện sẽ bị đình trệ.
- **Sàn tính toán.** MuZero / AlphaZero cần tính toán khổng lồ. Một lần ablation thường tốn hàng trăm giờ GPU. Các bản demo thu nhỏ tồn tại (ví dụ: AlphaZero trên Connect Four) để học tập.
- **Độ bao phủ của trình xác minh.** Các unit test vượt qua cho một giải pháp lỗi sẽ củng cố lỗi đó. Hãy thiết kế các trình xác minh bắt được các trường hợp biên.

## Sử dụng

Bối cảnh game-RL năm 2026, theo lĩnh vực:

| Lĩnh vực | Phương pháp thống trị |
|--------|-----------------|
| Trò chơi bảng tổng bằng không hai người (cờ vây, cờ vua, shogi) | AlphaZero / MuZero / KataGo |
| Trò chơi bài thông tin không hoàn hảo (poker) | CFR + deep learning (DeepStack, Libratus, Pluribus) |
| Atari / trò chơi pixel | Muesli / MuZero / IMPALA-PPO |
| Chiến lược đa người chơi lớn (Dota, StarCraft) | PPO + tự chơi + league (OpenAI Five, AlphaStar) |
| Suy luận toán/code LLM | GRPO (DeepSeek-R1, Qwen-RL, các bản sao mở) |
| Căn chỉnh LLM | DPO / RLHF-PPO (không phải GRPO; xác minh là sở thích không phải có thể kiểm chứng) |
| Robotics | PPO + DR (không phải game-RL, nhưng sử dụng cùng công cụ policy-gradient) |
| Các bài toán tổ hợp | Các biến thể AlphaZero (AlphaTensor, AlphaDev) |

*Công thức* — tự chơi, cải thiện tăng cường tìm kiếm, chưng cất chính sách — trải dài trên văn bản, pixel và điều khiển vật lý. GRPO là phiên bản trẻ nhất; nhiều phiên bản khác đang đến.

## Triển khai

Lưu dưới dạng `outputs/skill-game-rl-designer.md`:

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## Bài tập

1. **Dễ.** Triển khai GRPO bandit trong `code/main.py`. Huấn luyện trên 2 prompt × 4 token trả lời mỗi prompt. Hội tụ trong < 1,000 cập nhật với `G=8`.
2. **Trung bình.** Cắm PPO (clipped) và REINFORCE thuần túy vào. So sánh hiệu quả lấy mẫu và phương sai phần thưởng với GRPO trên cùng một bandit.
3. **Khó.** Mở rộng thành "chuỗi suy luận" độ dài 2: tác nhân phát ra hai token và trình xác minh thưởng cho cặp đó. Đo lường cách GRPO xử lý việc gán tín dụng (credit assignment) trên các chuỗi hai bước. (Gợi ý: tính lợi thế nhóm trên *toàn bộ chuỗi*, truyền bá cho cả hai vị trí token.)

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| MCTS | "Tìm kiếm cây với mạng đã học" | Monte Carlo Tree Search; chọn UCB1/PUCT với các tiên nghiệm `(p, v)` đã học. |
| AlphaZero | "Tự chơi + MCTS" | Mạng policy-value được huấn luyện để khớp với lượt truy cập MCTS và kết quả trò chơi. |
| MuZero | "AlphaZero mô hình đã học" | Vòng lặp tương tự nhưng trong không gian tiềm ẩn thông qua động lực học đã học. |
| GRPO | "PPO không critic" | Group Relative Policy Optimization; REINFORCE với đường cơ sở trung bình nhóm + KL. |
| PUCT | "UCB của AlphaZero" | `Q + c · p · √N / (1 + N_a)` — cân bằng ước tính giá trị với tiên nghiệm. |
| Tự chơi | "Tác nhân đấu với bản thân quá khứ" | Tiêu chuẩn cho tổng bằng không; tín hiệu huấn luyện đối xứng. |
| League play | "Tự chơi dựa trên quần thể" | Quá khứ + hiện tại + các tác nhân khai thác được lấy mẫu làm đối thủ. |
| Phần thưởng xác minh | "RL có thể xác minh" | Phần thưởng đến từ trình kiểm tra tất định (test vượt qua, đáp án khớp). |
| Phần thưởng quy trình | "PRM" | Chấm điểm từng bước suy luận, không chỉ đáp án cuối cùng. |

## Đọc thêm

- [Silver và cộng sự (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver và cộng sự (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser và cộng sự (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals và cộng sự (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — bài báo giới thiệu GRPO và đường cơ sở tương đối theo nhóm.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — công thức R1 bốn giai đoạn đầy đủ cộng với ablation R1-Zero.
- [Brown và cộng sự (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR + deep-learning ở quy mô lớn.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — bài báo khởi đầu tất cả.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — tài liệu tham khảo sản xuất để áp dụng GRPO với các hàm phần thưởng tùy chỉnh.
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — bản sao mở của công thức R1 ở nhiều quy mô.
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — khung giáo trình cho tự chơi, tìm kiếm và "phần thưởng được thiết kế" mà R1 hiện thực hóa ở quy mô LLM.