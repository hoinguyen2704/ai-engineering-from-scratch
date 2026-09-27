# Phục vụ LLM đa vùng và tính cục bộ của KV Cache

> Cân bằng tải round-robin gây hại trực tiếp cho suy luận LLM có sử dụng cache. Một yêu cầu không được gửi đến node đang giữ prefix của nó sẽ phải trả toàn bộ chi phí prefill — khoảng 800 ms ở P50 đối với prompt dài, so với ~80 ms khi cache hit. Năm 2026, mô hình sản xuất tiêu chuẩn là bộ định tuyến nhận biết cache (vLLM Router bằng Rust, llm-d router) – bộ định tuyến này tiêu thụ các sự kiện KV-cache và định tuyến dựa trên khớp prefix-hash. Nghiên cứu gần đây (GORGO) đưa độ trễ mạng liên vùng thành một tham số rõ ràng trong mục tiêu định tuyến. Các dịch vụ "suy luận liên vùng" thương mại (Bedrock cross-region inference, GKE multi-cluster gateways) coi suy luận là một hộp đen — chúng xử lý tính sẵn sàng, không phải TTFT. JPMorgan và Mayo Clinic đã thực hiện chuyển đổi dự phòng (failover) us-east-1 vào tháng 11 năm 2024 trong khoảng ~22 phút. Thực tế về DR (Khôi phục thảm họa): 32% các thất bại DR của LLM xảy ra do các đội ngũ sao lưu trọng số (weights) nhưng quên các tệp tokenizer hoặc cấu hình lượng tử hóa (quantization configs).

**Type:** Học tập
**Languages:** Python (stdlib, trình mô phỏng bộ định tuyến nhận biết prefix-cache)
**Prerequisites:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 phút

## Mục tiêu học tập

- Giải thích lý do tại sao cân bằng tải round-robin làm hỏng suy luận có cache và định lượng mức phạt TTFT.
- Vẽ sơ đồ bộ định tuyến nhận biết cache: đầu vào (sự kiện KV-cache), thuật toán (khớp prefix-hash), cơ chế giải quyết xung đột (sử dụng GPU).
- Nêu tên nguyên nhân gây thất bại DR chiếm 32% đối với LLM (thiếu tệp tokenizer / cấu hình lượng tử hóa) và liệt kê danh sách kiểm tra DR gồm ba tệp.
- Phân biệt các dịch vụ liên vùng thương mại (Bedrock CRI, GKE Multi-Cluster Gateway) với định tuyến nhận biết KV.

## Vấn đề

Dịch vụ của bạn chạy tại us-east-1, us-west-2 và eu-west-1. Bạn đặt một ALB phía trước với round-robin. Tỷ lệ cache hit của prefix trong môi trường sản xuất giảm xuống còn 8%. TTFT P50 tăng gấp ba. Nhật ký vLLM của bạn cho thấy mọi yêu cầu đều phải trả chi phí prefill đầy đủ.

Round-robin là tối ưu cho các dịch vụ không trạng thái (stateless). Suy luận LLM về bản chất là có trạng thái (stateful) — KV cache mã hóa mọi thứ mà mô hình đã thấy. Định tuyến mù quáng là định tuyến vào sai cache.

Ngoài ra, nhóm của bạn có một kế hoạch DR. Bạn sao lưu trọng số mô hình lên S3 liên vùng. Một sự cố khu vực xảy ra; bạn cố gắng chuyển đổi dự phòng; bản sao từ chối khởi động. Bạn đã quên rằng tokenizer.json, cấu hình lượng tử hóa và cấu hình RoPE scaling nằm trong một bucket riêng mà bạn chưa đồng bộ.

Phục vụ LLM đa vùng là vấn đề về cache, vấn đề về định tuyến và vấn đề về vệ sinh DR — không phải vấn đề về bộ cân bằng tải.

## Khái niệm

### Định tuyến nhận biết cache (Cache-aware routing)

Yêu cầu đến kèm theo một prompt. Bộ định tuyến băm prefix (ví dụ: 512 token đầu tiên); nó hỏi từng bản sao "bạn có lưu trữ prefix này không?". Các bản sao xuất bản các sự kiện KV-cache trên kênh pub/sub khi chúng cấp phát và giải phóng các block. Bộ định tuyến chọn bản sao có khớp, nếu không có thì chuyển sang cơ chế giải quyết xung đột dựa trên mức sử dụng GPU.

**vLLM Router** (Rust, stack sản xuất 2026): đăng ký các sự kiện `kv.cache.block_added`, duy trì ánh xạ prefix-hash → chỉ số bản sao, định tuyến với tra cứu O(1). Chuyển sang chọn hàng đợi ngắn nhất khi không có khớp.

**llm-d router**: cùng mô hình, native cho Kubernetes. Xuất bản sự kiện thông qua ControlPlane API.

**SGLang RadixAttention** (Phase 17 · 06) là tương đương trong nội bộ bản sao. Định tuyến giữa các bản sao là bước thượng nguồn nghiêm ngặt.

### Các con số

TTFT P50 trên prompt 2K-token, Llama 3.3 70B FP8, H100:
- Cache hit (cùng bản sao, prefix đã lưu trú): ~80 ms.
- Cache miss (prefill lạnh): ~800 ms.

Khoảng cách 10 lần. Nếu bộ định tuyến của bạn đạt 60-80% prefix cache hit trên các bản sao, bạn đạt hiệu suất xấp xỉ một bản sao đơn lẻ ở công suất N bản sao. Nếu đạt 10%, bạn đạt hiệu suất xấp xỉ mở rộng quy mô ngây thơ.

### Liên vùng có một ràng buộc mới — độ trễ mạng

RTT liên vùng:
- us-east-1 ↔ us-west-2: ~65 ms.
- us-east-1 ↔ eu-west-1: ~75 ms.
- us-east-1 ↔ ap-southeast-1: ~220 ms.

Nếu định tuyến đưa một yêu cầu từ us-east-1 đến một prefix "nóng" ở ap-southeast-1, phần prefill tiết kiệm được (800 → 80 ms) sẽ bị lu mờ bởi 440 ms khứ hồi. GORGO (nghiên cứu 2026) làm rõ điều này — tối thiểu hóa `prefill_time + network_latency` một cách tổng thể, không chỉ riêng prefill. Thông thường, giải pháp là giữ định tuyến trong khu vực trừ khi đối với các prefix đa MB khổng lồ nơi prefill chiếm ưu thế.

### "Suy luận liên vùng" thương mại không giúp ích ở đây

AWS Bedrock cross-region inference tự động định tuyến yêu cầu sang các vùng khác khi áp lực công suất tăng cao. Nó tối ưu hóa tính sẵn sàng, không phải TTFT, và coi suy luận là một hộp đen. GKE Multi-Cluster Gateway cũng tương tự — chuyển đổi dự phòng ở cấp dịch vụ, không nhận biết KV cache.

Bạn vẫn cần một bộ định tuyến nhận biết cache ở tầng ứng dụng ngay cả khi sử dụng các dịch vụ này. Chúng xử lý trường hợp "us-east-1 đang gặp sự cố". Định tuyến nhận biết cache xử lý trường hợp TTFT.

### Vệ sinh DR — vấn đề 32% thiếu tệp

Thống kê năm 2026 được trích dẫn rộng rãi: 32% các thất bại DR của LLM xảy ra do các đội ngũ sao lưu trọng số nhưng quên:

- `tokenizer.json` hoặc `tokenizer.model`
- Cấu hình lượng tử hóa (`quantize_config.json`, AWQ scales, GPTQ zero-points)
- Cấu hình đặc thù của mô hình (RoPE scaling, attention masks, chat templates)
- Cấu hình engine (`vllm_config.yaml`, sampling defaults, LoRA adapter manifests)

Giải pháp là một bản kê khai DR tối thiểu gồm ba tệp:

1. Tất cả các tệp trong repo mô hình HF (trọng số + cấu hình + tokenizer).
2. Cấu hình phục vụ đặc thù của engine.
3. Bản kê khai triển khai (K8s YAML, Dockerfile, dependency lock).

Thêm vào đó: thực hiện diễn tập DR hàng quý. Cuộc diễn tập us-east-1 của JPMorgan đạt thời gian phục hồi 22 phút vào tháng 11 năm 2024 chỉ vì kịch bản đã được diễn tập.

### Cư trú dữ liệu (Data residency) là trực giao

PHI của khách hàng EU không được rời khỏi EU. Nếu bộ định tuyến nhận biết cache của bạn gửi một yêu cầu từ Paris đến us-east-1 để khớp prefix, bạn đã vi phạm GDPR bất kể mức tăng TTFT. Hãy phân vùng các bộ định tuyến theo ranh giới cư trú trước khi tối ưu hóa cho cache.

### Các con số bạn nên nhớ

- Khoảng cách TTFT giữa cache hit và miss: ~10x (80 ms so với 800 ms trên prompt 2K).
- RTT liên vùng US-EU: ~75 ms.
- Thất bại DR: 32% thiếu tokenizer/cấu hình lượng tử hóa.
- Chuyển đổi dự phòng JPMorgan us-east-1 tháng 11 năm 2024: 22 phút (SLA 30 phút).

```figure
cache-aware-router
```

## Sử dụng

`code/main.py` mô phỏng ba chiến lược định tuyến (round-robin, cache-aware regional, cache-aware global) trên khối lượng công việc đa vùng. Báo cáo tỷ lệ cache hit, TTFT P50/P99 và chi phí liên vùng.

## Triển khai

Bài học này tạo ra `outputs/skill-multi-region-router.md`. Dựa trên các vùng, ràng buộc cư trú và SLA, hãy thiết kế một kế hoạch định tuyến.

## Bài tập

1. Chạy `code/main.py`. Ở độ dài prompt nào thì định tuyến liên vùng vượt qua định tuyến chỉ trong nội bộ vùng, với RTT 75 ms?
2. Tỷ lệ cache hit của bạn giảm từ 70% xuống 12%. Hãy chẩn đoán ba nguyên nhân có thể xảy ra và các quan sát có thể xác nhận từng nguyên nhân.
3. Thiết kế một bản kê khai DR cho mô hình 70B AWQ-quantized được phục vụ trong vLLM với 5 LoRA adapter. Liệt kê mọi tệp và cấu hình.
4. Lập luận xem liệu Bedrock cross-region inference có "đủ" cho một công ty fintech với các SLO TTFT nghiêm ngặt hay không. Trích dẫn các hành vi cụ thể.
5. Một yêu cầu từ Paris khớp với một prefix ở us-east-1. Bạn có định tuyến nó không? Hãy viết chính sách.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Cache-aware routing | "smart LB" | Định tuyến dựa trên khớp prefix-hash đến bản sao giữ KV-cache |
| KV-cache events | "cache pub-sub" | Các bản sao xuất bản thêm/xóa block; bộ định tuyến lập chỉ mục |
| Prefix hash | "cache key" | Băm của N token đầu tiên được dùng làm khóa tra cứu bộ định tuyến |
| GORGO | "cross-region routing research" | arXiv 2602.11688; độ trễ mạng là tham số rõ ràng |
| Cross-region inference | "Bedrock CRI" | Sản phẩm AWS; chuyển đổi dự phòng tính sẵn sàng, không nhận biết TTFT |
| DR manifest | "the backup list" | Mọi tệp cần thiết để khôi phục — không chỉ trọng số |
| Data residency | "GDPR boundary" | Ràng buộc pháp lý về vùng nào được xem dữ liệu người dùng |
| RTT | "round-trip time" | Độ trễ mạng; 75 ms US-EU, 220 ms US-APAC |
| LLM-aware LB | "cache-hit LB" | Bộ định tuyến nhận biết cache như một danh mục sản phẩm |

## Đọc thêm

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — tái sử dụng KV-cache liên vùng với tham số độ trễ mạng.
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — tài liệu về chuyển đổi dự phòng tính sẵn sàng.
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — mã nguồn bộ định tuyến nhận biết cache.