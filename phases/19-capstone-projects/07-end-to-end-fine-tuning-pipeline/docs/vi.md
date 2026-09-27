# Capstone 07 — Pipeline Fine-Tuning End-to-End (Dữ liệu đến SFT đến DPO đến Phục vụ)

> Một mô hình 8B được huấn luyện trên dữ liệu của riêng bạn, được căn chỉnh DPO dựa trên các ưu tiên của bạn, được lượng tử hóa, giải mã suy đoán (speculative-decoded) và phục vụ ở mức $/1M tokens có thể đo lường được. Stack mở năm 2026 bao gồm Axolotl v0.8, TRL 0.15, Unsloth để lặp lại, GPTQ/AWQ/GGUF để lượng tử hóa, vLLM 0.7 với EAGLE-3 để phục vụ. Capstone này yêu cầu chạy toàn bộ pipeline một cách có thể tái lập — đầu vào là YAML, đầu ra là endpoint phục vụ — và xuất bản model card theo Khung Mở Mô hình (Model Openness Framework) 2026.

**Type:** Capstone
**Languages:** Python (pipeline), YAML (configs), Bash (scripts)
**Prerequisites:** Phase 2 (ML), Phase 3 (DL), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**Time:** 35 giờ

## Vấn đề

Mọi đội ngũ AI chuyên nghiệp vào năm 2026 đều duy trì một pipeline fine-tuning sẵn sàng. Không phải vì họ phát hành một mô hình nền tảng tiên phong, mà vì sự thích ứng hạ nguồn (downstream adaptation) — SFT theo miền, DPO dựa trên các ưu tiên được gắn nhãn, các bản nháp chưng cất cho giải mã suy đoán, phục vụ với EAGLE-3 — là nơi tạo ra những lợi thế có thể đo lường được. Axolotl v0.8 xử lý các cấu hình SFT đa GPU. TRL 0.15 xử lý DPO và GRPO. Unsloth giúp bạn lặp lại nhanh trên một GPU. vLLM 0.7 với EAGLE-3 đẩy thông lượng giải mã lên gấp 2-3 lần mà không làm giảm chất lượng. Công cụ đã có sẵn; kỹ năng nằm ở các tệp YAML, vệ sinh dữ liệu và kỷ luật đánh giá.

Bạn sẽ chạy một mô hình nền 8B (Llama 3.3, Qwen3, hoặc Gemma 3) qua SFT sau đó là DPO trên dữ liệu đặc thù của tác vụ, lượng tử hóa để phục vụ và đo lường mức tăng so với lm-evaluation-harness, RewardBench-2, MT-Bench-v2 và MMLU-Pro. Bạn sẽ tạo ra một model card theo Khung Mở Mô hình 2026. Mục tiêu là khả năng tái lập — một lệnh duy nhất chạy lại toàn bộ pipeline từ đầu đến cuối.

## Khái niệm

Pipeline có năm giai đoạn. **Dữ liệu**: khử trùng lặp (MinHash / Datatrove), lọc chất lượng (bộ phân loại kiểu Nemotron-CC), xóa PII, kiểm tra vệ sinh phân tách (split-hygiene) chống nhiễm bẩn benchmark công khai. **SFT**: Axolotl YAML, ZeRO-3 trên 8xH100, lịch trình cosine, đóng gói chuỗi (packed sequences), 2-3 epoch. **DPO hoặc GRPO**: cấu hình TRL, 1 epoch, các cặp ưu tiên được gắn nhãn bởi con người hoặc đánh giá bởi mô hình, điều chỉnh beta. **Lượng tử hóa**: GPTQ + AWQ + GGUF để linh hoạt khi triển khai. **Phục vụ**: vLLM 0.7 với các đầu giải mã suy đoán EAGLE-3 (hoặc SGLang với SpecForge), triển khai K8s, HPA dựa trên độ trễ hàng đợi.

Các thử nghiệm cắt bỏ (ablations) là kết quả cần bàn giao: chỉ SFT so với SFT+DPO so với SFT+GRPO trên ba benchmark đặc thù. Chỉ số phục vụ: tokens/s ở batch 1 / 8 / 32, tỷ lệ chấp nhận EAGLE-3, $/1M tokens. Đánh giá an toàn: tỷ lệ vượt qua Llama Guard 4. Model card: đánh giá thiên kiến, các seed tái lập, giấy phép dữ liệu.

## Kiến trúc

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## Stack

- Dữ liệu: Datatrove để khử trùng lặp, bộ phân loại Nemotron-CC cho chất lượng, Presidio cho PII
- Nền tảng: Llama 3.3 8B, Qwen3 14B, hoặc Gemma 3 12B
- SFT: Axolotl v0.8 với ZeRO-3, Flash Attention 3, đóng gói chuỗi
- Điều chỉnh ưu tiên: TRL 0.15 cho DPO hoặc GRPO; Unsloth cho lặp lại trên một GPU
- Lượng tử hóa: GPTQ (Marlin), AWQ, GGUF qua llama.cpp
- Phục vụ: vLLM 0.7 với giải mã suy đoán EAGLE-3 (hoặc SGLang 0.4 + SpecForge)
- Đánh giá: lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro
- Đánh giá an toàn: Llama Guard 4, ShieldGemma-2
- Hạ tầng: Kubernetes + NVIDIA device plugin, HPA trên chỉ số độ trễ hàng đợi
- Khả năng quan sát: W&B cho huấn luyện, Langfuse cho suy luận

```figure
ce-finetune-stages
```

## Xây dựng

1. **Pipeline dữ liệu.** Chạy khử trùng lặp Datatrove trên corpus thô. Áp dụng bộ phân loại chất lượng kiểu Nemotron-CC. Presidio xóa PII. Viết các tập train/val với seed rõ ràng.

2. **Kiểm tra nhiễm bẩn.** Đối với mỗi tập validation, tính toán MinHash so với các tập kiểm tra MMLU-Pro, MT-Bench-v2, RewardBench-2. Loại bỏ bất kỳ sự trùng lặp nào.

3. **Axolotl SFT.** YAML với ZeRO-3, FA3, đóng gói chuỗi. 2-3 epoch trên 8xH100. Ghi log vào W&B.

4. **TRL DPO / GRPO.** Lấy checkpoint SFT, chạy một epoch DPO trên các cặp ưu tiên (hoặc GRPO với phần thưởng có thể kiểm chứng trên toán/code). Quét tham số beta.

5. **Lượng tử hóa.** Tạo ba bản lượng tử: GPTQ-INT4-Marlin, AWQ-INT4, GGUF-Q4_K_M cho llama.cpp. Ghi lại kích thước và thông lượng danh nghĩa.

6. **Phục vụ với giải mã suy đoán.** Cấu hình vLLM 0.7 với các đầu nháp EAGLE-3 được huấn luyện qua Red Hat Speculators. Đo tỷ lệ chấp nhận và độ trễ đuôi (tail latency) ở batch 1 / 8 / 32. Báo cáo $/1M tokens so với Anthropic / OpenAI trên cùng một đánh giá.

7. **Ma trận đánh giá.** Chạy lm-eval-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro trên mô hình nền, chỉ SFT, SFT+DPO, SFT+GRPO. Tạo một bảng tổng hợp.

8. **Đánh giá an toàn.** Tỷ lệ vượt qua Llama Guard 4 trên tập dev. Bộ lọc đầu ra ShieldGemma-2.

9. **Model card.** Mẫu MOF 2026: dữ liệu, huấn luyện, đánh giá, an toàn, giấy phép, phần tái lập với các tệp YAML và commit SHA.

## Sử dụng

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## Vận chuyển

`outputs/skill-finetuning-pipeline.md` mô tả kết quả bàn giao. Một lệnh duy nhất chạy dữ liệu qua SFT, qua DPO, qua lượng tử hóa, qua phục vụ, qua đánh giá và xuất ra một model card + endpoint phục vụ.

| Trọng số | Tiêu chí | Cách đo lường |
|:-:|---|---|
| 25 | Delta đánh giá so với nền | Mức tăng đo được trên các tác vụ mục tiêu (MMLU-Pro, MT-Bench-v2, tác vụ đặc thù) |
| 20 | Khả năng tái lập pipeline | Một lệnh chạy lại từ đầu đến cuối với các seed giống hệt |
| 20 | Vệ sinh dữ liệu | Tỷ lệ khử trùng lặp, độ phủ xóa PII, kiểm tra nhiễm bẩn đạt yêu cầu |
| 20 | Hiệu quả phục vụ | tokens/s tại bs=1/8/32, tỷ lệ chấp nhận EAGLE-3, $/1M tokens |
| 15 | Model card + đánh giá an toàn | Độ hoàn thiện MOF 2026 + tỷ lệ vượt qua Llama Guard 4 |
| **100** | | |

## Bài tập

1. Chạy chỉ SFT so với SFT+DPO so với SFT+GRPO trên cùng một benchmark đặc thù. Báo cáo phương pháp ưu tiên nào thắng và thắng bao nhiêu.

2. Thay thế Llama 3.3 8B bằng Qwen3 14B. Đo $/1M tokens ở chất lượng tương đương.

3. Đo tỷ lệ chấp nhận EAGLE-3 trên dữ liệu miền so với ShareGPT chung. Báo cáo delta và ý nghĩa của nó đối với ngân sách độ trễ.

4. Tiêm 1% nhiễm bẩn (rò rỉ câu trả lời MMLU-Pro vào dữ liệu huấn luyện) và chạy lại đánh giá. Quan sát độ chính xác MMLU-Pro tăng phi thực tế. Xây dựng một cổng CI kiểm tra nhiễm bẩn để phát hiện điều này.

5. Thêm LoRA SFT như một giải pháp thay thế cho fine-tune toàn bộ. Đo khoảng cách chất lượng ở mức bộ nhớ thấp hơn 10 lần.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Axolotl | "SFT trainer" | Trình huấn luyện thống nhất dựa trên YAML cho SFT, DPO và chưng cất |
| TRL | "Preference tuner" | Thư viện Hugging Face cho DPO, GRPO, PPO trên LLM |
| GRPO | "Group-relative policy optimization" | Công thức RL của DeepSeek R1 với phần thưởng có thể kiểm chứng |
| EAGLE-3 | "Speculative decoding draft" | Các đầu nháp dự đoán trước N token; vLLM xác minh với mô hình mục tiêu |
| MOF | "Model Openness Framework" | Tiêu chuẩn 2026 để xếp hạng các bản phát hành mô hình về dữ liệu, mã, giấy phép |
| Contamination check | "Split hygiene" | Phát hiện dựa trên MinHash về việc rò rỉ tập kiểm tra vào tập huấn luyện |
| Acceptance rate | "EAGLE / MTP metric" | Tỷ lệ các token nháp được mô hình mục tiêu chấp nhận |

## Đọc thêm

- [Tài liệu Axolotl](https://axolotl-ai-cloud.github.io/axolotl/) — trình huấn luyện SFT / DPO tham chiếu
- [Tài liệu TRL](https://huggingface.co/docs/trl) — các triển khai tham chiếu DPO và GRPO
- [Unsloth](https://github.com/unslothai/unsloth) — tham chiếu lặp lại trên một GPU
- [Bài báo DeepSeek R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — phương pháp luận GRPO
- [Tài liệu vLLM + EAGLE-3](https://docs.vllm.ai) — stack phục vụ tham chiếu
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — trình huấn luyện giải mã suy đoán thay thế
- [Model Openness Framework 2026](https://isocpp.org/) — tiêu chuẩn xếp hạng phát hành mở
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — trình chạy đánh giá chuẩn