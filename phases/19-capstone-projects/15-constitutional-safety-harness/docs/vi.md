# Capstone 15 — Constitutional Safety Harness + Red-Team Range

> Các bộ phân loại Constitutional Classifiers của Anthropic, Llama Guard 4 của Meta, ShieldGemma-2 của Google, NVIDIA Nemotron 3 Content Safety và X-Guard cho khả năng bao phủ đa ngôn ngữ đã định hình nên stack phân loại an toàn năm 2026. garak, PyRIT, NVIDIA Aegis và promptfoo đã trở thành các công cụ đánh giá đối kháng tiêu chuẩn. NeMo Guardrails v0.12 kết nối chúng thành một pipeline sản xuất. Capstone này kết hợp tất cả lại với nhau: một bộ khung an toàn phân lớp xung quanh một ứng dụng mục tiêu, một tác nhân red-team tự động chạy hơn 6 họ tấn công và một quy trình tự phê bình (self-critique) dựa trên hiến pháp tạo ra sự thay đổi về độ an toàn có thể đo lường được.

**Type:** Capstone
**Languages:** Python (safety pipeline, red team), YAML (policy configs)
**Prerequisites:** Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 18 (ethics, safety, alignment)
**Phases exercised:** P10 · P11 · P13 · P14 · P18
**Time:** 25 hours

## Problem

Ranh giới của an toàn LLM vào năm 2026 không nằm ở việc liệu các bộ phân loại có hoạt động hay không (chúng hoạt động, ở mức độ tương đối) mà là cách kết hợp chúng một cách chính xác xung quanh một ứng dụng sản xuất mà không gây ra tình trạng từ chối quá mức (over-refusing) hoặc để lại những lỗ hổng rõ ràng. Llama Guard 4 xử lý các vi phạm chính sách bằng tiếng Anh. X-Guard (132 ngôn ngữ) xử lý các jailbreak đa ngôn ngữ. ShieldGemma-2 phát hiện các cuộc tấn công prompt injection dựa trên hình ảnh. NVIDIA Nemotron 3 Content Safety bao phủ các danh mục doanh nghiệp. Constitutional Classifiers của Anthropic là một phương pháp riêng biệt được sử dụng trong quá trình huấn luyện thay vì phục vụ (serving).

Sự tiến hóa của các cuộc tấn công cũng rất quan trọng. PAIR và TAP tự động hóa việc khám phá jailbreak. GCG chạy các cuộc tấn công hậu tố dựa trên gradient. Các cuộc tấn công đa vòng (multi-turn) và chuyển đổi mã (code-switch) khai thác bộ nhớ của tác nhân. Bất kỳ LLM nào được triển khai đều cần một phạm vi red-team — garak và PyRIT là các trình điều khiển chính tắc — cộng với các biện pháp giảm thiểu được ghi chép và các phát hiện được chấm điểm CVSS.

Bạn sẽ củng cố một ứng dụng mục tiêu (một mô hình 8B đã được tinh chỉnh hướng dẫn hoặc một trong các chatbot RAG từ các capstone khác), chạy hơn 6 họ tấn công chống lại nó và tạo ra phép đo độ an toàn trước/sau khi thực hiện.

## Concept

Pipeline an toàn bao gồm năm lớp. **Input sanitize**: loại bỏ các ký tự độ rộng bằng 0, giải mã base64/rot13, chuẩn hóa Unicode. **Policy layer**: NeMo Guardrails v0.12 (off-domain, toxicity, PII extraction). **Classifier gate**: Llama Guard 4 cho đầu vào, X-Guard cho ngôn ngữ không phải tiếng Anh, ShieldGemma-2 cho đầu vào hình ảnh. **Model**: LLM mục tiêu. **Output filter**: Llama Guard 4 cho đầu ra, Presidio PII scrub, thực thi trích dẫn nếu có. **HITL tier**: các đầu ra được gắn cờ rủi ro cao sẽ chuyển đến hàng đợi Slack.

Phạm vi red-team chạy trên một bộ lập lịch. PAIR và TAP tự động khám phá các jailbreak. GCG chạy các cuộc tấn công hậu tố dựa trên gradient. Các cuộc tấn công mã hóa ASCII / base64 / rot13. Các cuộc tấn công đa vòng (áp dụng nhân vật, khai thác bộ nhớ). Các cuộc tấn công chuyển đổi mã (trộn tiếng Anh với tiếng Swahili hoặc tiếng Thái). Mỗi lần chạy tạo ra một tệp kết quả có cấu trúc với điểm CVSS và mốc thời gian tiết lộ.

Quy trình tự phê bình dựa trên hiến pháp là một sự can thiệp trong thời gian huấn luyện. Lấy 1.000 prompt thử nghiệm độc hại, yêu cầu mô hình soạn thảo phản hồi, phê bình nó dựa trên một hiến pháp bằng văn bản (các quy tắc không gây hại) và huấn luyện lại trên vòng lặp phê bình. Đo lường sự thay đổi về độ an toàn trước/sau trên một tập đánh giá (eval) tách biệt.

## Architecture

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## Stack

- Safety classifiers: Llama Guard 4, ShieldGemma-2, NVIDIA Nemotron 3 Content Safety, X-Guard
- Guardrail framework: NeMo Guardrails v0.12 + OPA
- Red-team drivers: garak (NVIDIA), PyRIT (Microsoft Azure), NVIDIA Aegis, promptfoo
- Jailbreak agents: PAIR (Chao et al., 2023), Tree-of-Attacks (TAP), GCG suffix
- Constitutional training: Anthropic-style self-critique loop + SFT on critiques
- PII scrub: Presidio
- Target: an 8B instruction-tuned model or one of the other capstones' RAG chatbots

```figure
cf-safety-stack
```

## Build It

1. **Target setup.** Thiết lập một mô hình 8B đã được tinh chỉnh hướng dẫn trên vLLM (hoặc tái sử dụng chatbot RAG từ một capstone khác). Đây là ứng dụng đang được kiểm thử.

2. **Safety pipeline wrap.** Kết nối pipeline năm lớp xung quanh mục tiêu. Xác minh từng lớp có thể quan sát được riêng lẻ (span mỗi lớp trong Langfuse).

3. **Classifier coverage.** Tải Llama Guard 4, X-Guard (đa ngôn ngữ), ShieldGemma-2 (hình ảnh). Chạy từng cái trên một tập dữ liệu nhỏ đã được gắn nhãn để thiết lập các đường cơ sở (baselines).

4. **Red-team scheduler.** Lập lịch cho garak, PyRIT, một tác nhân PAIR, một tác nhân TAP, một trình chạy GCG, một kẻ tấn công đa vòng và một kẻ tấn công chuyển đổi mã. Mỗi cái chạy trên một hàng đợi riêng.

5. **Attack suite.** Sáu họ tấn công: (1) PAIR automated jailbreak, (2) TAP tree-of-attacks, (3) GCG gradient suffix, (4) ASCII / base64 / rot13 encoding, (5) multi-turn persona, (6) multilingual code-switch. Báo cáo tỷ lệ thành công trên mỗi họ.

6. **Constitutional self-critique.** Tuyển chọn 1.000 prompt thử nghiệm độc hại. Với mỗi prompt, mục tiêu soạn thảo một phản hồi. Một LLM phê bình sẽ chấm điểm dựa trên một hiến pháp bằng văn bản ("không gây hại", "trích dẫn bằng chứng", "từ chối các yêu cầu bất hợp pháp"). Các prompt mà người phê bình phản đối sẽ được viết lại; mục tiêu tinh chỉnh trên các cặp cải tiến từ phê bình. Đo lường sự thay đổi về độ an toàn trước/sau trên một tập eval tách biệt.

7. **Over-refusal measurement.** Theo dõi tỷ lệ dương tính giả trên một bộ prompt lành tính (ví dụ: XSTest). Mục tiêu phải duy trì sự hữu ích trên các câu hỏi lành tính.

8. **CVSS scoring.** Đối với mỗi jailbreak thành công, chấm điểm trên CVSS 4.0 (vector tấn công, độ phức tạp, tác động). Tạo mốc thời gian tiết lộ và kế hoạch giảm thiểu.

9. **Range automation.** Mọi thứ ở trên chạy trên cron; các phát hiện ghi vào hàng đợi; các cảnh báo hồi quy về việc từ chối quá mức được gửi đến Slack.

## Use It

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## Ship It

`outputs/skill-safety-harness.md` là sản phẩm bàn giao. Một pipeline an toàn phân lớp cấp độ sản xuất cộng với một phạm vi red-team có thể tái lập với các chỉ số thay đổi về độ an toàn trước/sau.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Độ bao phủ bề mặt tấn công | 6+ họ tấn công được thực hiện, 2+ ngôn ngữ |
| 20 | Đánh đổi dương tính thật / dương tính giả | Tỷ lệ chặn tấn công so với tỷ lệ vượt qua lành tính XSTest |
| 20 | Độ lệch tự phê bình | Độ an toàn trước/sau trên tập eval tách biệt |
| 20 | Tài liệu và tiết lộ | Các phát hiện được chấm điểm CVSS với mốc thời gian |
| 15 | Tự động hóa và khả năng lặp lại | Mọi thứ chạy trên cron với cảnh báo |
| **100** | | |

## Exercises

1. Chạy plugin của garak về prompt-injection trên một chatbot RAG và so sánh tỷ lệ thành công của cuộc tấn công khi có và không có lớp lọc đầu ra (output-filter).

2. Thêm họ tấn công thứ bảy: indirect prompt injection thông qua các tài liệu được truy xuất. Đo lường khả năng phòng thủ bổ sung cần thiết.

3. Triển khai chế độ "từ chối kèm trợ giúp": khi guardrail chặn, mục tiêu cung cấp một câu trả lời liên quan an toàn hơn thay vì từ chối thẳng thừng. Đo lường độ lệch XSTest.

4. Khoảng trống bao phủ đa ngôn ngữ: tìm một ngôn ngữ mà X-Guard hoạt động kém hiệu quả. Đề xuất một tập dữ liệu tinh chỉnh nhắm vào ngôn ngữ đó.

5. Chạy quy trình tự phê bình dựa trên hiến pháp trên mô hình 30B và đo lường xem độ lệch có mở rộng quy mô hay không.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Layered safety | "Defense in depth" | Nhiều guardrail ở đầu vào, cổng, đầu ra, HITL |
| Llama Guard 4 | "Meta's safety classifier" | Bộ phân loại nội dung đầu vào/đầu ra tham chiếu năm 2026 |
| PAIR | "Jailbreak agent" | Bài báo (Chao et al.) về khám phá jailbreak do LLM điều khiển |
| TAP | "Tree-of-Attacks" | Biến thể tìm kiếm cây của PAIR |
| GCG | "Greedy coordinate gradient" | Tấn công hậu tố đối kháng dựa trên gradient |
| Constitutional self-critique | "Anthropic-style training" | Mục tiêu soạn thảo -> người phê bình chấm điểm -> viết lại -> huấn luyện lại |
| XSTest | "Benign probe set" | Benchmark cho hồi quy từ chối quá mức |
| CVSS 4.0 | "Severity score" | Chấm điểm lỗ hổng tiêu chuẩn cho các phát hiện an toàn |

## Further Reading

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — tài liệu tham khảo thời gian huấn luyện
- [Meta Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — bộ phân loại đầu vào/đầu ra năm 2026
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — an toàn đa phương thức + hình ảnh
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — tài liệu tham khảo doanh nghiệp
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — an toàn đa ngôn ngữ 132 ngôn ngữ
- [garak](https://github.com/NVIDIA/garak) — bộ công cụ red-team của NVIDIA
- [PyRIT](https://github.com/Azure/PyRIT) — khung red-team của Microsoft
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — khung guardrail
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — bài báo về tác nhân jailbreak