# Proximal Policy Optimization (PPO)

> A2C loại bỏ mỗi rollout sau một lần cập nhật. PPO bao bọc policy gradient trong một tỷ lệ quan trọng (importance ratio) được cắt bớt (clipped), cho phép bạn thực hiện hơn 10 epoch trên cùng một dữ liệu mà không làm policy bị bùng nổ. Schulman và cộng sự (2017). Đây vẫn là thuật toán policy-gradient mặc định vào năm 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75 phút

## Vấn đề

A2C (Bài 07) là on-policy: gradient `E_{π_θ}[A · ∇ log π_θ]` yêu cầu dữ liệu được lấy mẫu từ `π_θ` hiện tại. Chỉ cần thực hiện một lần cập nhật, `π_θ` sẽ thay đổi; dữ liệu bạn vừa dùng đã trở thành off-policy. Nếu tái sử dụng nó, gradient của bạn sẽ bị chệch (biased).

Rollout rất tốn kém. Trên Atari, một rollout qua 8 env × 128 bước = 1024 chuyển đổi và mất khoảng mười giây thời gian môi trường. Việc loại bỏ dữ liệu đó sau một bước gradient là rất lãng phí.

Trust Region Policy Optimization (TRPO, Schulman 2015) là giải pháp khắc phục đầu tiên: ràng buộc mỗi lần cập nhật sao cho KL divergence giữa policy cũ và mới nằm dưới `δ`. Về lý thuyết thì rất sạch sẽ, nhưng yêu cầu giải conjugate-gradient cho mỗi lần cập nhật. Không ai dùng TRPO vào năm 2026 nữa.

PPO (Schulman và cộng sự 2017) thay thế ràng buộc trust-region cứng nhắc bằng một mục tiêu (objective) được cắt bớt đơn giản. Chỉ cần thêm một dòng code. Mười epoch cho mỗi rollout. Không cần conjugate gradient. Đảm bảo lý thuyết đủ tốt. Chín năm sau, nó vẫn là thuật toán policy-gradient mặc định cho mọi thứ, từ MuJoCo đến RLHF.

## Khái niệm

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**Tỷ lệ quan trọng (Importance ratio).**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

Đây là tỷ lệ khả năng xảy ra (likelihood ratio) của policy mới so với policy đã thu thập dữ liệu. `r_t = 1` nghĩa là không có thay đổi. `r_t = 2` nghĩa là policy mới có khả năng thực hiện `a_t` cao gấp đôi so với policy cũ.

**Hàm mục tiêu thay thế được cắt bớt (Clipped surrogate).**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

Hai thành phần:

- Nếu advantage `A_t > 0` và tỷ lệ cố gắng vượt quá `1 + ε`, việc cắt bớt sẽ làm phẳng gradient — không đẩy một hành động tốt vượt quá `+ε` so với xác suất cũ.
- Nếu advantage `A_t < 0` và tỷ lệ cố gắng vượt quá `1 - ε` (nghĩa là chúng ta sẽ làm cho một hành động tồi trở nên khả thi hơn so với mức giảm đã cắt bớt của nó), việc cắt bớt sẽ giới hạn gradient — không đẩy một hành động tồi xuống dưới `-ε`.

`min` xử lý hướng còn lại: nếu tỷ lệ đã di chuyển theo hướng *có lợi*, bạn vẫn nhận được gradient (không cắt bớt ở phía có thể gây hại cho bạn).

Thông thường `ε = 0.2`. Vẽ đồ thị hàm mục tiêu theo `r_t`: một hàm tuyến tính từng đoạn với phần mái phẳng ở "phía tốt" và phần sàn phẳng ở "phía tồi".

**Hàm loss PPO đầy đủ.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

Cấu trúc actor-critic giống như A2C. Ba hệ số, thường là `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`.

**Vòng lặp huấn luyện.**

1. Thu thập `N × T` chuyển đổi qua `N` môi trường song song, mỗi môi trường `T` bước.
2. Tính toán các advantage (GAE), đóng băng chúng như các hằng số.
3. Đóng băng `π_{θ_old}` như một bản snapshot của `π_θ` hiện tại.
4. Trong `K` epoch, cho mỗi minibatch `(s, a, A, V_target, log π_old(a|s))`:
   - Tính `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`.
   - Áp dụng `L^{CLIP}` + value loss + entropy.
   - Bước gradient.
5. Loại bỏ rollout. Quay lại bước 1.

`K = 10` và minibatch 64 là bộ siêu tham số tiêu chuẩn. PPO rất mạnh mẽ: các con số chính xác hiếm khi quan trọng trong phạm vi ±50%.

**Biến thể KL-penalty.** Bài báo gốc đề xuất một giải pháp thay thế sử dụng KL penalty thích ứng: `L = L^{PG} - β · KL(π_θ || π_old)` với `β` được điều chỉnh dựa trên KL quan sát được. Phiên bản cắt bớt (clipping) đã trở nên phổ biến hơn; biến thể KL vẫn tồn tại trong RLHF (nơi KL so với policy tham chiếu là một ràng buộc riêng biệt mà bạn luôn muốn có).

```figure
ppo-clip
```

## Xây dựng

### Bước 1: ghi lại `log π_old(a | s)` tại thời điểm rollout

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

Snapshot được thực hiện một lần, tại thời điểm rollout. Nó không thay đổi trong các epoch cập nhật.

### Bước 2: tính toán GAE advantages (Bài 07)

Giống như A2C. Chuẩn hóa trên toàn bộ batch.

### Bước 3: cập nhật clipped surrogate

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

Mô hình "cắt bớt → gradient bằng 0" là trái tim của PPO. Nếu policy mới đã trôi quá xa theo hướng có lợi, quá trình cập nhật sẽ dừng lại.

### Bước 4: value và entropy

Thêm MSE tiêu chuẩn vào mục tiêu critic và phần thưởng entropy vào actor, giống như A2C.

### Bước 5: chẩn đoán

Ba điều cần theo dõi mỗi lần cập nhật:

- **Mean KL** `E[log π_old - log π_θ]`. Nên nằm trong khoảng `[0, 0.02]`. Nếu vượt quá `0.1`, hãy giảm `K_EPOCHS` hoặc `LR`.
- **Clip fraction** — tỷ lệ các mẫu có tỷ lệ nằm ngoài `[1-ε, 1+ε]`. Nên là `~0.1-0.3`. Nếu `~0`, việc cắt bớt không bao giờ kích hoạt → tăng `LR` hoặc `K_EPOCHS`. Nếu `~0.5+`, bạn đang over-fitting rollout → giảm chúng xuống.
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`. Chỉ số chất lượng critic. Nên tăng dần về 1 khi critic học được.

## Các cạm bẫy

- **Hệ số clip không được tinh chỉnh.** `ε = 0.2` là tiêu chuẩn thực tế. Tăng lên `0.1` làm cho các cập nhật quá rụt rè; `0.3+` gây ra sự mất ổn định.
- **Quá nhiều epoch.** `K > 20` thường gây mất ổn định vì policy trôi quá xa so với `π_old`. Hãy giới hạn epoch, đặc biệt đối với các mạng lớn.
- **Không chuẩn hóa phần thưởng.** Quy mô phần thưởng lớn sẽ ăn vào phạm vi clip. Hãy chuẩn hóa phần thưởng (running std) trước khi tính advantage.
- **Quên chuẩn hóa advantage.** Chuẩn hóa zero-mean/unit-std theo batch là tiêu chuẩn. Bỏ qua nó sẽ làm hỏng PPO trên hầu hết các benchmark.
- **Learning rate không giảm.** PPO hưởng lợi từ việc giảm LR tuyến tính về 0. LR hằng số thường tệ hơn.
- **Lỗi toán học tỷ lệ quan trọng.** Luôn sử dụng `exp(log_new - log_old)` để ổn định số học, không phải `new / old`.
- **Sai dấu gradient.** Tối đa hóa surrogate = *tối thiểu hóa* `-L^{CLIP}`. Dấu bị đảo ngược là lỗi PPO phổ biến nhất.

## Sử dụng

PPO là thuật toán RL mặc định của năm 2026 trên một số lượng đáng ngạc nhiên các lĩnh vực:

| Trường hợp sử dụng | Biến thể PPO |
|----------|-------------|
| MuJoCo / điều khiển robot | PPO với Gaussian policy, GAE(0.95) |
| Atari / trò chơi rời rạc | PPO với categorical policy, rollout 128 bước |
| RLHF cho LLM | PPO với KL penalty so với mô hình tham chiếu, phần thưởng từ RM ở cuối phản hồi |
| Tác nhân trò chơi quy mô lớn | IMPALA + PPO (AlphaStar, OpenAI Five) |
| LLM suy luận | GRPO (Bài 12) — biến thể PPO không có critic |
| Dữ liệu chỉ ưu tiên | DPO — rút gọn dạng đóng của PPO+KL, không lấy mẫu trực tuyến |

*Hình dạng loss* của PPO — clipped surrogate + value + entropy — là khung sườn cho DPO, GRPO và gần như mọi pipeline RLHF.

## Triển khai

Lưu dưới dạng `outputs/skill-ppo-trainer.md`:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Bài tập

1. **Dễ.** Chạy PPO trên 4×4 GridWorld với `ε=0.2, K=4`. So sánh hiệu quả lấy mẫu với A2C (một epoch mỗi rollout) tại các bước env tương đương.
2. **Trung bình.** Quét `K ∈ {1, 4, 10, 30}`. Vẽ đồ thị lợi nhuận theo bước env và theo dõi KL trung bình mỗi lần cập nhật. Tại `K` nào thì KL bùng nổ trên tác vụ này?
3. **Khó.** Thay thế clipped surrogate bằng KL penalty thích ứng (`β` nhân đôi nếu `KL > 2·target`, chia đôi nếu `KL < target/2`). So sánh lợi nhuận cuối cùng, độ ổn định và tính chất không cần clip.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Importance ratio | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`; độ lệch so với policy đã thu thập dữ liệu. |
| Clipped surrogate | "Mẹo chính của PPO" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; gradient phẳng sau khi cắt ở phía có lợi. |
| Trust region | "Mục tiêu của TRPO / PPO" | Giới hạn KL của mỗi lần cập nhật để đảm bảo cải thiện đơn điệu. |
| KL penalty | "Soft trust region" | PPO thay thế: `L - β · KL(π_θ \|\| π_old)`. `β` thích ứng. |
| Clip fraction | "Tần suất kích hoạt clip" | Chẩn đoán — nên là 0.1-0.3; nằm ngoài phạm vi nghĩa là chưa tinh chỉnh. |
| Multi-epoch training | "Tái sử dụng dữ liệu" | K epoch trên mỗi rollout; đánh đổi chi phí phương sai lấy hiệu quả lấy mẫu. |
| On-policy-ish | "Gần như on-policy" | PPO về danh nghĩa là on-policy nhưng K>1 epoch sử dụng dữ liệu hơi off-policy một cách an toàn. |
| PPO-KL | "PPO khác" | Biến thể KL-penalty; dùng trong RLHF nơi KL-so-với-tham-chiếu đã là một ràng buộc. |

## Đọc thêm

- [Schulman và cộng sự (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — bài báo gốc.
- [Schulman và cộng sự (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO, tiền thân của PPO.
- [Andrychowicz và cộng sự (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — phân tích mọi siêu tham số PPO.
- [Ouyang và cộng sự (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT; công thức PPO trong RLHF.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — giải thích hiện đại, rõ ràng với PyTorch.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — PPO file đơn tham chiếu được nhiều bài báo sử dụng.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — công thức sản xuất cho PPO trên các mô hình ngôn ngữ; đọc cùng Bài 09 (RLHF).
- [Engstrom và cộng sự (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — bài báo về "37 tối ưu hóa cấp code"; mẹo PPO nào thực sự quan trọng và mẹo nào chỉ là truyền miệng.