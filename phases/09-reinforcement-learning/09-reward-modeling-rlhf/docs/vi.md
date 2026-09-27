# Reward Modeling & RLHF

> Con người không thể viết một hàm phần thưởng (reward function) cho "phản hồi hỗ trợ tốt", nhưng họ có thể so sánh hai phản hồi và chọn cái tốt hơn. Hãy huấn luyện một reward model dựa trên các so sánh đó, sau đó thực hiện RL cho language model dựa trên mô hình này. Christiano 2017. InstructGPT 2022. Công thức đã biến GPT-3 thành ChatGPT. Đến năm 2026, phương pháp này chủ yếu được thay thế bằng DPO — nhưng tư duy cốt lõi vẫn được giữ nguyên.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45 phút

## Vấn đề

Bạn đã huấn luyện một language model dựa trên mục tiêu dự đoán token tiếp theo. Nó viết tiếng Anh đúng ngữ pháp. Nó cũng nói dối, lan man và từ chối việc từ chối. Bạn không thể sửa lỗi này bằng cách pretraining thêm — dữ liệu web chính là vấn đề, không phải là liều thuốc chữa trị.

Bạn muốn một *scalar reward* (phần thưởng vô hướng) cho biết "phản hồi A tốt hơn phản hồi B cho chỉ dẫn X". Việc viết hàm phần thưởng đó bằng tay là bất khả thi. "Sự hữu ích" không phải là một biểu thức dạng đóng (closed-form) trên các token. Nhưng con người có thể so sánh hai đầu ra và đánh dấu ưu tiên. Việc này rất rẻ để thu thập ở quy mô lớn.

RLHF (Christiano và cộng sự 2017; Ouyang và cộng sự 2022) chuyển đổi các ưu tiên thành một reward model, sau đó tối ưu hóa LM thông qua PPO dựa trên phần thưởng đó. Gồm ba bước: SFT → RM → PPO. Đây là công thức đã tạo nên ChatGPT, Claude, Gemini và mọi LLM được căn chỉnh (aligned-LLM) khác trong giai đoạn 2023–2025.

Vào năm 2026, bước PPO chủ yếu được thay thế bằng DPO (Phase 10 · 08) vì nó rẻ hơn và hiệu quả gần như tương đương cho việc tinh chỉnh căn chỉnh. Tuy nhiên, thành phần *reward model* vẫn là nền tảng cho mọi bộ lấy mẫu Best-of-N, mọi pipeline RL-from-verifiable-rewards và mọi mô hình suy luận sử dụng process reward model. Hiểu RLHF là bạn hiểu toàn bộ hệ thống căn chỉnh.

## Khái niệm

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**Giai đoạn 1: Supervised Fine-Tuning (SFT).** Bắt đầu từ một base model đã được pretrained. Fine-tune trên các ví dụ do con người viết về hành vi mục tiêu (phản hồi theo chỉ dẫn, trả lời hữu ích, v.v.). Kết quả: một mô hình `π_SFT` có *thiên kiến hướng tới hành vi tốt* nhưng vẫn có không gian hành động không giới hạn.

**Giai đoạn 2: Huấn luyện Reward Model.**

- Thu thập các cặp phản hồi `(y_+, y_-)` cho các prompt `x`, được con người dán nhãn là "y_+ được ưu tiên hơn y_-."
- Huấn luyện một reward model `R_φ(x, y)` để gán điểm cao hơn cho `y_+`.
- Hàm mất mát: **Bradley-Terry pairwise logistic**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ là hàm sigmoid. Sự khác biệt về phần thưởng ngụ ý log-odds của ưu tiên. BT đã là tiêu chuẩn từ năm 1952 (Bradley-Terry) và là lựa chọn thống trị trong RLHF hiện đại.

- `R_φ` thường được khởi tạo từ mô hình SFT với một scalar head ở trên cùng. Sử dụng cùng kiến trúc transformer; một lớp tuyến tính duy nhất xuất ra phần thưởng.

**Giai đoạn 3: PPO dựa trên RM với hình phạt KL.**

- Khởi tạo policy có thể huấn luyện `π_θ` từ `π_SFT`. Giữ một mô hình *tham chiếu* `π_ref = π_SFT` đã đóng băng.
- Phần thưởng ở cuối phản hồi `y`:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  Hình phạt KL ngăn cản `π_θ` trôi dạt tùy tiện khỏi `π_SFT` — nó là một *bộ điều chuẩn* (regularizer), không phải là một vùng tin cậy cứng. `β` thường là `0.01`-`0.05`.
- Chạy PPO (Bài 08) với phần thưởng này. Các lợi thế (advantages) được tính toán trên quỹ đạo cấp token, nhưng RM chỉ chấm điểm toàn bộ phản hồi.

**Tại sao cần KL?** Nếu không có nó, PPO sẽ vui vẻ tìm ra các chiến lược "hack phần thưởng" — RM chỉ được huấn luyện trên các phần hoàn thiện trong phân phối (in-distribution). Một phản hồi ngoài phân phối (out-of-distribution) có thể đạt điểm cao hơn bất kỳ phản hồi nào do con người viết. KL giữ cho `π_θ` nằm gần đa tạp nơi RM được huấn luyện. Đây là nút điều chỉnh quan trọng nhất trong RLHF.

**Tình trạng năm 2026:**

- **DPO** (Rafailov 2023): đại số dạng đóng giúp gộp Giai đoạn 2+3 thành một hàm mất mát có giám sát duy nhất trên dữ liệu ưu tiên. Không RM, không PPO. Chất lượng tương đương trên các benchmark căn chỉnh với chi phí tính toán thấp hơn nhiều. Được đề cập trong Phase 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO với baseline tương đối theo nhóm thay vì critic, phần thưởng từ một *verifier* (chạy code / khớp đáp án toán học) thay vì RM do con người huấn luyện. Thống trị cho các mô hình suy luận. Được đề cập trong Phase 9 · 12.
- **Process reward models (PRMs):** chấm điểm các giải pháp từng phần (mỗi bước suy luận), được sử dụng trong cả biến thể RLHF và GRPO cho suy luận.
- **Constitutional AI / RLAIF:** sử dụng một LLM đã được căn chỉnh để tạo ra các ưu tiên thay vì con người. Mở rộng quy mô ngân sách ưu tiên.

```figure
reward-model
```

## Xây dựng

Bài học này sử dụng các "prompt" và "phản hồi" tổng hợp nhỏ được biểu diễn dưới dạng chuỗi. RM là một bộ chấm điểm tuyến tính trên biểu diễn bag-of-tokens. Không có LLM thực sự nào ở đây — *hình dạng* của pipeline mới là điều quan trọng, không phải quy mô. Xem `code/main.py`.

### Bước 1: dữ liệu ưu tiên tổng hợp

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

Trong RLHF thực tế, bước này được thay thế bằng các nhãn từ con người. Hình dạng — `(prompt, preferred_response, rejected_response)` — là giống hệt nhau.

### Bước 2: Bradley-Terry reward model

Điểm tuyến tính: `R(x, y) = w · bag(y)`. Huấn luyện để giảm thiểu log-loss cặp Bradley-Terry:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

Sau vài trăm lần cập nhật, `w` gán trọng số dương cho các token từ tốt và trọng số âm cho các token xấu.

### Bước 3: Policy kiểu PPO trên RM

Policy đồ chơi của chúng ta tạo ra một token duy nhất từ từ vựng. Chúng ta chấm điểm token đó theo RM, tính `log π_θ(token | prompt)`, thêm hình phạt KL-to-reference và áp dụng PPO surrogate đã cắt (clipped).

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Bước 4: giám sát KL

Theo dõi `KL(π_θ || π_ref)` trung bình mỗi lần cập nhật. Nếu nó vượt quá `~5-10`, policy đã trôi dạt quá xa khỏi `π_SFT` — `β` thấp hơn đang tăng hoặc việc hack phần thưởng đang bắt đầu. Đây là chẩn đoán hàng đầu trong RLHF thực tế.

### Bước 5: công thức sản xuất với TRL

Khi bạn đã hiểu pipeline đồ chơi, đây là vòng lặp tương tự như cách một người dùng thư viện thực tế viết. [TRL](https://huggingface.co/docs/trl) của Hugging Face là triển khai tham chiếu — `RewardTrainer` cho Giai đoạn 2 và `PPOTrainer` (với KL-to-reference được tích hợp sẵn) cho Giai đoạn 3.

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

Ba điều thư viện làm cho bạn. `adap_kl_ctrl=True` triển khai lịch trình β thích ứng: nếu KL quan sát được vượt quá `target_kl`, β tăng gấp đôi; nếu dưới một nửa, β giảm một nửa. Mô hình tham chiếu được đóng băng theo quy ước — bạn không được vô tình chia sẻ tham số với `policy`. Và value head nằm trên cùng backbone với policy (`AutoModelForCausalLMWithValueHead` gắn một scalar MLP head), đó là lý do tại sao TRL báo cáo `policy/kl` và `value/loss` riêng biệt.

## Các cạm bẫy

- **Tối ưu hóa quá mức / hack phần thưởng.** RM không hoàn hảo; `π_θ` tìm ra các phản hồi đối nghịch đạt điểm cao nhưng lại tệ. Triệu chứng: phần thưởng tăng vô hạn trong khi điểm đánh giá của con người đi ngang hoặc giảm. Cách sửa: dừng sớm, tăng `β`, mở rộng dữ liệu huấn luyện RM.
- **Hack độ dài.** Các RM được huấn luyện trên các phản hồi hữu ích thường ngầm định thưởng cho độ dài. Policy học cách đệm phản hồi. Cách khắc phục: phần thưởng chuẩn hóa theo độ dài, hoặc RLAIF với một RM nhận biết độ dài.
- **RM quá nhỏ.** RM cần phải lớn ít nhất bằng policy. Một RM quá nhỏ không thể chấm điểm trung thực các đầu ra của policy.
- **Điều chỉnh KL.** β quá thấp → trôi dạt và hack phần thưởng. β quá cao → policy hầu như không thay đổi. Mẹo tiêu chuẩn là sử dụng β *thích ứng* nhắm mục tiêu KL cố định mỗi bước.
- **Nhiễu dữ liệu ưu tiên.** ~30% nhãn của con người bị nhiễu hoặc mơ hồ. Hiệu chỉnh bằng cách huấn luyện RM trên dữ liệu đã lọc sự đồng thuận hoặc sử dụng nhiệt độ (temperature) trên BT.
- **Các vấn đề off-policy.** Dữ liệu PPO hơi lệch khỏi policy (off-policy) sau epoch đầu tiên. Giám sát tỷ lệ cắt (clip fraction) như trong Bài 08.

## Sử dụng

RLHF vào năm 2026 được phân lớp:

| Lớp | Mục tiêu | Phương pháp |
|-------|--------|--------|
| Tuân thủ chỉ dẫn, hữu ích, vô hại | Căn chỉnh | DPO (Phase 10 · 08) được ưu tiên hơn RLHF-PPO. |
| Độ chính xác suy luận (toán, code) | Năng lực | GRPO với phần thưởng verifier (Phase 9 · 12). |
| Tác vụ đa bước dài hạn | Tác nhân | PPO / GRPO với process reward models qua các bước. |
| Hành vi an toàn / từ chối | An toàn | RLHF-PPO với RM an toàn riêng biệt, hoặc Constitutional AI. |
| Best-of-N tại thời điểm suy luận | Căn chỉnh nhanh | Sử dụng RM tại thời điểm giải mã; không cần huấn luyện policy. |
| Chưng cất phần thưởng | Tính toán suy luận | Huấn luyện một "reward head" nhỏ trên một LM đã đóng băng. |

RLHF là phương pháp *chủ đạo* trong giai đoạn 2022–2024. Đến năm 2026, các pipeline căn chỉnh sản xuất ưu tiên DPO, chỉ dùng PPO cho các bước cần RM chuyên sâu hoặc quan trọng về an toàn.

## Triển khai

Lưu dưới dạng `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## Bài tập

1. **Dễ.** Huấn luyện reward model Bradley-Terry trong `code/main.py` trên 500 cặp ưu tiên tổng hợp. Đo độ chính xác theo cặp trên 100 cặp giữ lại. Nên vượt quá 90%.
2. **Trung bình.** Chạy vòng lặp PPO-RLHF đồ chơi với `β ∈ {0.0, 0.1, 1.0}`. Với mỗi vòng, vẽ biểu đồ điểm RM so với KL-to-reference qua các lần cập nhật. Chạy nào bị hack phần thưởng?
3. **Khó.** Triển khai DPO (hàm mất mát xác suất ưu tiên dạng đóng) trên cùng dữ liệu ưu tiên và so sánh với pipeline RLHF-PPO về chi phí tính toán sử dụng và điểm RM cuối cùng đạt được.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| RLHF | "Alignment RL" | Pipeline ba giai đoạn SFT + RM + PPO (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | "Mạng chấm điểm" | Hàm vô hướng đã học được khớp với các ưu tiên theo cặp qua Bradley-Terry. |
| Bradley-Terry | "Pairwise logistic loss" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; mục tiêu RM tiêu chuẩn. |
| KL penalty | "Giữ gần tham chiếu" | `β · KL(π_θ \|\| π_ref)` trong phần thưởng; bộ điều chuẩn chống hack phần thưởng. |
| Reward hacking | "Định luật Goodhart" | Policy khai thác các lỗ hổng RM; triệu chứng: phần thưởng tăng, đánh giá con người đi ngang. |
| RLAIF | "Ưu tiên do AI dán nhãn" | RLHF nơi các nhãn đến từ một LM khác thay vì con người. |
| PRM | "Process Reward Model" | Chấm điểm các bước suy luận từng phần; được sử dụng trong các pipeline suy luận. |
| Constitutional AI | "Phương pháp của Anthropic" | Các ưu tiên do AI tạo ra được hướng dẫn bởi các quy tắc rõ ràng. |

## Đọc thêm

- [Christiano và cộng sự (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — bài báo khởi đầu RLHF.
- [Ouyang và cộng sự (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — công thức đằng sau ChatGPT.
- [Stiennon và cộng sự (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — RLHF sớm cho tóm tắt văn bản.
- [Rafailov và cộng sự (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO; mặc định sau RLHF vào năm 2026.
- [Bai và cộng sự (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF và vòng lặp tự phê bình.
- [Bài báo RLHF của Anthropic (Bai và cộng sự 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — bài báo HH.
- [Thư viện TRL của Hugging Face](https://huggingface.co/docs/trl) — `RewardTrainer` và `PPOTrainer` trong sản xuất. Đọc mã nguồn trainer để biết chi tiết về adaptive-KL và value-head.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) bởi Lambert, Castricato, von Werra, Havrilla — hướng dẫn chuẩn về pipeline ba giai đoạn với sơ đồ.
- [von Werra và cộng sự (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — thư viện; `examples/` có các script RLHF end-to-end cho Llama, Mistral và Qwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — quan điểm về giả thuyết phần thưởng; điều kiện tiên quyết thiết yếu để suy nghĩ về hack phần thưởng.