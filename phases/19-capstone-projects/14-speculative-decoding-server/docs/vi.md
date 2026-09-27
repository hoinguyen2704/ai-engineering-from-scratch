# Capstone 14 — Speculative-Decoding Inference Server

> Speculative decoding — một phương pháp trong đó một mô hình dự thảo (draft model) giá rẻ đề xuất các token và mô hình mục tiêu (target model) xác thực chúng trong một lượt truyền — hiện đã trở thành một kỹ thuật tối ưu hóa sẵn sàng cho sản xuất, không còn là một thủ thuật nghiên cứu. EAGLE-3 trong vLLM 0.7 mang lại thông lượng gấp 2.5-3 lần trên lưu lượng thực tế. P-EAGLE (AWS 2026) đã đẩy khả năng suy luận song song (parallel speculation) đi xa hơn nữa. SpecForge của SGLang đã huấn luyện các draft head ở quy mô lớn. Hub Speculators của Red Hat đã xuất bản các bản dự thảo được căn chỉnh (aligned drafts) cho các mô hình mở phổ biến. TensorRT-LLM đã đưa speculative decoding trở thành tính năng hạng nhất trên NVIDIA. Stack phục vụ sản xuất năm 2026 là vLLM hoặc SGLang với các bản dự thảo dòng EAGLE, lượng tử hóa FP8 hoặc INT4 và HPA dựa trên thời gian chờ đợi trong hàng đợi (queue-wait). Capstone này yêu cầu bạn phục vụ hai mô hình mở với thông lượng gấp 2.5 lần trở lên so với baseline cùng báo cáo đầy đủ về độ trễ đuôi (tail-latency).

**Type:** Capstone
**Languages:** Python (serving), C++ / CUDA (kernel inspection), YAML (configs)
**Prerequisites:** Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 17 (infrastructure)
**Phases exercised:** P3 · P7 · P10 · P17
**Time:** 30 hours

## Problem

Speculative decoding đã trở thành một mặt hàng phổ biến vào năm 2026. Các draft head EAGLE-3 được huấn luyện trên các trạng thái ẩn (hidden states) của mô hình mục tiêu và dự đoán trước N token; mô hình mục tiêu xác thực tất cả trong một lượt truyền duy nhất. Tỷ lệ chấp nhận từ 60-80% giúp tăng thông lượng end-to-end lên gấp 2-3 lần. vLLM 0.7 tích hợp tính năng này một cách nguyên bản. SGLang + SpecForge cung cấp cho bạn pipeline huấn luyện. Red Hat's Speculators xuất bản các bản dự thảo được căn chỉnh cho Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B.

Kỹ năng nằm ở khâu vận hành phục vụ (serving operations), không phải ở mô hình. Tỷ lệ chấp nhận sẽ thay đổi theo phân phối lưu lượng (ShareGPT so với code so với dữ liệu miền). Độ trễ đuôi khi bị từ chối (rejection) sẽ tệ hơn so với khi không sử dụng speculation — bạn phải báo cáo p99 ở nhiều kích thước batch khác nhau, không chỉ là tokens/sec ở trạng thái ổn định. Chi phí trên mỗi 1 triệu token so với API của Anthropic / OpenAI là đòn bẩy cho sự uy tín.

## Concept

Speculative decoding có hai lớp. Một mô hình **dự thảo** (EAGLE-3 head, ngram, hoặc mô hình mục tiêu nhỏ hơn) đề xuất k token ứng viên mỗi bước. Mô hình **mục tiêu** xác thực tất cả k token trong một lượt truyền; bất kỳ tiền tố nào được chấp nhận sẽ thay thế đường dẫn greedy. Tỷ lệ chấp nhận phụ thuộc vào sự căn chỉnh giữa mô hình dự thảo và mục tiêu cũng như phân phối đầu vào.

EAGLE-3 vượt trội hơn các bản dự thảo ngram trên hầu hết lưu lượng truy cập. P-EAGLE chạy suy luận song song cho các cây dự thảo sâu hơn. Sự đánh đổi: Độ trễ P99 khi bị từ chối sẽ cao hơn vì lượt xác thực lớn hơn. Cấu hình phục vụ phải báo cáo độ trễ theo từng nhóm kích thước batch để làm nổi bật điều này.

Việc triển khai được thực hiện trên Kubernetes. vLLM 0.7 chạy một bản sao (replica) trên mỗi GPU hoặc tensor-parallel shard. HPA tự động mở rộng quy mô dựa trên thời gian chờ đợi trong hàng đợi thay vì CPU. Các phương pháp lượng tử hóa FP8 (Marlin) và INT4 (AWQ) giữ bộ nhớ GPU trong phạm vi của H100 / H200. Báo cáo end-to-end bao gồm thông lượng, tỷ lệ chấp nhận, p50/p99 ở batch 1/8/32 và chi phí $/1M token.

## Architecture

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## Stack

- Serving: vLLM 0.7 hoặc SGLang 0.4
- Speculative methods: EAGLE-3 draft heads, P-EAGLE parallel speculation, ngram fallback
- Draft training: SpecForge (SGLang) hoặc Red Hat Speculators
- Target models: Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B
- Quantization: FP8 (Marlin), INT4 AWQ
- Deployment: Kubernetes + NVIDIA device plugin; HPA on queue-wait metric
- Eval: ShareGPT, MT-Bench-v2, GSM8K, HumanEval for domain-spread acceptance measurement
- Reference: TensorRT-LLM speculative decoding for a vendor baseline

```figure
cf-spec-decode
```

## Build It

1. **Chuẩn bị mô hình mục tiêu.** Chọn Llama 3.3 70B. Lượng tử hóa sang FP8 thông qua Marlin. Triển khai trên vLLM 0.7 trên 1xH100 (hoặc 2x tensor-parallel).

2. **Nguồn dự thảo.** Lấy một draft head EAGLE-3 đã được căn chỉnh từ Red Hat Speculators (hoặc huấn luyện một cái thông qua SpecForge). Tải vào cấu hình speculative-decoding của vLLM.

3. **Số liệu baseline.** Trước khi speculation: tokens/s ở batch 1/8/32, độ trễ p50/p99, mức sử dụng GPU. Công bố kết quả.

4. **Kích hoạt EAGLE-3.** Bật cấu hình; chạy lại cùng một benchmark. Báo cáo mức tăng tốc, tỷ lệ chấp nhận, độ lệch độ trễ đuôi p99.

5. **P-EAGLE.** Kích hoạt suy luận song song; đo lường cây dự thảo sâu hơn so với EAGLE-3 tuần tự. Báo cáo điểm uốn nơi P-EAGLE giúp ích hoặc gây hại.

6. **Lưu lượng miền.** Chạy ShareGPT so với HumanEval so với lưu lượng truy cập chuyên biệt qua cùng một server. Đo lường tỷ lệ chấp nhận theo từng phân phối. Xác định khi nào các bản dự thảo bị lệch (drift).

7. **Mô hình mục tiêu thứ hai.** Chạy cùng pipeline trên Qwen3-Coder-30B MoE. Dự thảo sẽ khó hơn (do nhiễu định tuyến MoE). Báo cáo kết quả.

8. **K8s HPA.** Triển khai trên K8s với HPA theo dõi `queue_wait_ms`. Chứng minh khả năng mở rộng khi tải tăng gấp ba.

9. **So sánh chi phí.** Tính toán $/1M token so với Anthropic Claude Sonnet 4.7 và OpenAI GPT-5.4 trên cùng một eval. Công bố kết quả.

## Use It

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## Ship It

`outputs/skill-inference-server.md` mô tả các sản phẩm bàn giao. Một stack phục vụ đã được đo lường với speculative decoding, báo cáo benchmark đầy đủ và triển khai K8s.

| Trọng số | Tiêu chí | Cách đo lường |
|:-:|---|---|
| 25 | Mức tăng tốc so với baseline | Thông lượng 2.5x+ với chất lượng tương đương trên hai mô hình |
| 20 | Tỷ lệ chấp nhận trên lưu lượng thực tế | Báo cáo tỷ lệ chấp nhận theo từng phân phối |
| 20 | Kỷ luật độ trễ đuôi P99 | p99 ở batch 1/8/32 có và không có speculation |
| 20 | Vận hành | Triển khai K8s, HPA dựa trên queue-wait, rollout mượt mà |
| 15 | Viết báo cáo và phương pháp luận | Giải thích rõ ràng những gì đã thay đổi và tại sao |
| **100** | | |

## Exercises

1. Đo lường sự suy giảm tỷ lệ chấp nhận khi bản dự thảo chậm hơn mô hình mục tiêu một phiên bản (ví dụ: Llama 3.3 -> 3.4 drift). Xây dựng cảnh báo giám sát.

2. Triển khai ngram-fallback: nếu tỷ lệ chấp nhận của EAGLE-3 giảm xuống dưới ngưỡng, hãy chuyển sang dự thảo ngram. Báo cáo sự cải thiện về độ tin cậy.

3. Chạy thí nghiệm MoE có kiểm soát: cùng Qwen3-Coder-30B với nhiễu định tuyến được tiêm vào so với không có. Đo lường độ nhạy chấp nhận của dự thảo.

4. Mở rộng sang H200 (141 GB). Báo cáo dung lượng mô hình trên mỗi replica đạt được và liệu bạn có thể phục vụ Llama 3.3 70B không lượng tử hóa hay không.

5. Benchmark speculative decoding của TensorRT-LLM trên cùng phần cứng H100. Báo cáo nơi nó thắng so với vLLM.

## Key Terms

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Draft model | "Speculator" | Mô hình nhỏ đề xuất N token để mô hình mục tiêu xác thực |
| EAGLE-3 | "Kiến trúc dự thảo 2026" | Draft head được huấn luyện trên các trạng thái ẩn của mục tiêu; ~75% chấp nhận |
| P-EAGLE | "Suy luận song song" | Cây các nhánh dự thảo được xác thực trong một lượt truyền mục tiêu |
| Acceptance rate | "Tỷ lệ trúng" | Tỷ lệ các token dự thảo được chấp nhận mà không cần lấy mẫu lại |
| Quantization | "FP8 / INT4" | Trọng số độ chính xác thấp hơn để chứa nhiều mô hình hơn trong bộ nhớ GPU |
| Queue wait | "Chỉ số HPA" | Thời gian một yêu cầu chờ trong hàng đợi trước khi bắt đầu suy luận |
| Speculators hub | "Dự thảo căn chỉnh" | Hub Neural Magic của Red Hat chứa các dự thảo EAGLE cho các mô hình mở phổ biến |

## Further Reading

- [Tài liệu vLLM EAGLE và P-EAGLE](https://docs.vllm.ai) — stack phục vụ tham chiếu
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — bài báo về speculative decoding song song + tích hợp
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — pipeline huấn luyện draft-head
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — hub dự thảo căn chỉnh
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — giải pháp thay thế từ nhà cung cấp
- [Kiến trúc phục vụ Fireworks.ai](https://fireworks.ai/blog) — tham chiếu thương mại
- [Bài báo EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — bài báo về phương pháp
- [Kho lưu trữ vLLM](https://github.com/vllm-project/vllm) — mã nguồn và benchmark