# Lựa chọn Stack Quan sát (Observability) cho LLM

> Thị trường quan sát năm 2026 chia thành hai loại. Các nền tảng phát triển (LangSmith, Langfuse, Comet Opik) tích hợp giám sát với đánh giá (evals), quản lý prompt và phát lại phiên (session replays). Các công cụ Gateway/instrumentation (Helicone, SigNoz, OpenLLMetry, Phoenix) tập trung vào telemetry. Langfuse là lõi mã nguồn mở (MIT) với sự cân bằng tốt (miễn phí 50K sự kiện/tháng trên cloud). Phoenix là công cụ native OpenTelemetry theo giấy phép Elastic License 2.0 — tuyệt vời để trực quan hóa drift/RAG, nhưng không phải là backend sản xuất bền vững. Arize AX sử dụng tích hợp Iceberg/Parquet zero-copy, tuyên bố rẻ hơn 100 lần so với các hệ thống quan sát nguyên khối. LangSmith dẫn đầu cho LangChain/LangGraph, giá $39/người dùng/tháng, chỉ hỗ trợ self-host cho bản Enterprise. Helicone dựa trên proxy với thời gian thiết lập 15-30 phút, miễn phí 100K yêu cầu/tháng, nhưng độ sâu về trace cho agent còn hạn chế. Mô hình sản xuất phổ biến: Gateway (Helicone/Portkey) + nền tảng đánh giá (Phoenix/TruLens) được kết nối bởi OpenTelemetry.

**Type:** Học tập
**Languages:** Python (stdlib, trình mô phỏng lấy mẫu trace)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 phút

## Mục tiêu học tập

- Phân biệt các nền tảng phát triển (tích hợp: evals + prompts + sessions) với các công cụ gateway/telemetry (chỉ có traces + metrics).
- Ánh xạ sáu công cụ chính (Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik) với giấy phép, giá cả và trường hợp sử dụng tối ưu.
- Giải thích mô hình kết nối OpenTelemetry cho phép kết hợp công cụ gateway với nền tảng đánh giá riêng biệt.
- Nêu tên yếu tố khác biệt về chi phí năm 2026 (cách tiếp cận zero-copy của Arize AX so với ingest nguyên khối) và hệ số nhân khoảng 100 lần.

## Vấn đề

Bạn đã triển khai một tính năng LLM. Nó hoạt động. Nhưng bạn không có khả năng quan sát các lỗi prompt, vòng lặp công cụ, độ trễ, chi phí tăng đột biến hoặc tỷ lệ hit của prompt-cache. Bạn Google "LLM observability" và nhận được tám công cụ đều tuyên bố giải quyết cùng một vấn đề ở ba mức giá khác nhau.

Chúng không giải quyết cùng một vấn đề. LangSmith trả lời "tại sao LangGraph này chạy thất bại?". Phoenix trả lời "pipeline RAG của tôi có bị drift không?". Helicone trả lời "ứng dụng nào đang tiêu tốn nhiều token?". Langfuse trả lời "tôi có thể tự host toàn bộ hệ thống không?". Các công cụ khác nhau, đối tượng khác nhau.

Việc lựa chọn dựa trên bốn trục: stack (LangChain? SDK thô? đa nhà cung cấp?), khả năng chấp nhận giấy phép (chỉ MIT? Elastic OK? thương mại ổn?), ngân sách (gói miễn phí? $100/mo? $1000/tháng?), và khả năng self-host (bắt buộc? nên có? không bao giờ?).

## Khái niệm

### Hai danh mục

**Nền tảng phát triển** tích hợp quan sát với đánh giá, quản lý prompt, phiên bản tập dữ liệu, phát lại phiên. Bạn chạy thử nghiệm, xem prompt nào hiệu quả, kiểm tra hồi quy tập dữ liệu của prompt mới so với các phiên bản cũ. LangSmith, Langfuse, Comet Opik.

**Công cụ Gateway/telemetry** đo lường các lệnh gọi inference — prompt, phản hồi, token, độ trễ, model, chi phí. Helicone, SigNoz, OpenLLMetry, Phoenix. Tối giản. Có thể kết hợp với công cụ đánh giá riêng biệt thông qua OpenTelemetry.

### Langfuse — Cân bằng OSS

- Lõi Apache / MIT; self-host qua Docker.
- Gói cloud miễn phí: 50K sự kiện/tháng. Trả phí: $29/tháng cho nhóm.
- Đánh giá, quản lý prompt, traces, tập dữ liệu. Độ bao phủ hợp lý cho cả bốn tính năng của nền tảng phát triển.
- Điểm tối ưu: bạn muốn các tính năng đẳng cấp LangSmith nhưng phải self-host hoặc giữ giấy phép OSS.

### Phoenix (Arize) — ưu tiên telemetry, native OpenTelemetry

- Elastic License 2.0; self-host đơn giản.
- Xuất sắc trong việc trực quan hóa RAG và drift. Các biểu đồ phân tán không gian embedding được tích hợp sẵn.
- Không được thiết kế như backend sản xuất bền vững — chủ yếu là quan sát trong quá trình phát triển.
- Điểm tối ưu: phát triển pipeline RAG, gỡ lỗi drift, kết hợp với gateway riêng cho sản xuất.

### Arize AX — cuộc chơi quy mô

- Thương mại. Tích hợp data lake zero-copy qua Iceberg/Parquet.
- Tuyên bố rẻ hơn ~100 lần so với quan sát nguyên khối (đẳng cấp Datadog) ở quy mô lớn. Cách tính: bạn lưu trữ trace trong Parquet của riêng mình trên S3; Arize đọc trực tiếp.
- Điểm tối ưu: >10M traces/ngày, có sẵn data lake, muốn dashboard chuyên biệt cho LLM mà không phải trả giá theo kiểu Datadog.

### LangSmith — Ưu tiên LangChain/LangGraph

- Thương mại, $39/người dùng/tháng. Chỉ self-host cho bản Enterprise.
- Tốt nhất trong phân khúc cho các stack LangChain và LangGraph. Nếu bạn không dùng các công cụ này, nó ít hấp dẫn hơn.
- Điểm tối ưu: nhóm cam kết sử dụng LangChain, sẵn sàng trả phí.

### Helicone — proxy tối giản (MVP)

- Thiết lập 15-30 phút bằng cách thay thế `OPENAI_API_BASE` của bạn bằng proxy Helicone.
- Giấy phép MIT; miễn phí 100K yêu cầu/tháng, trả phí từ $20/tháng.
- Bao gồm failover, caching, giới hạn tốc độ — đóng vai trò như một gateway.
- Độ sâu về trace cho agent/đa bước còn hạn chế.
- Điểm tối ưu: khởi động nhanh, ứng dụng đơn stack, cần cả gateway + quan sát trong một.

### Opik (Comet) — Nền tảng phát triển OSS

- Apache 2.0, hoàn toàn OSS.
- Bộ tính năng tương tự Langfuse với di sản từ Comet.
- Điểm tối ưu: các nhóm ML đã dùng Comet, muốn quan sát LLM trên cùng một giao diện.

### SigNoz — APM đầy đủ ưu tiên OpenTelemetry

- Apache 2.0. Xử lý APM chung cộng với LLM thông qua OpenTelemetry.
- Điểm tối ưu: quan sát thống nhất trên các dịch vụ và lệnh gọi LLM.

### Chất kết dính: OpenTelemetry + GenAI semantic conventions

OpenTelemetry đã công bố các quy ước ngữ nghĩa GenAI vào cuối năm 2025 (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`). Các công cụ tiêu thụ OTel có thể tương tác với nhau. Mô hình sản xuất đang nổi lên:

1. Phát OTel với các quy ước GenAI từ mọi lệnh gọi LLM.
2. Định tuyến đến gateway (Helicone / Portkey) cho hoạt động hàng ngày.
3. Gửi song song đến nền tảng đánh giá (Phoenix / Langfuse) để kiểm tra hồi quy.
4. Lưu trữ trong data lake (Iceberg) để phân tích dài hạn qua Arize AX hoặc DuckDB.

### Cái bẫy: đo lường sai tầng

Đo lường bên trong framework agent (ví dụ: thêm trace LangSmith) sẽ gắn chặt bạn vào framework đó. Đo lường ở tầng HTTP/OpenAI-SDK (thông qua OpenLLMetry hoặc gateway) sẽ có tính di động cao hơn.

### Lấy mẫu (Sampling) — bạn không thể giữ tất cả

Ở mức >1M yêu cầu/ngày, chi phí lưu trữ toàn bộ trace sẽ đắt hơn cả chi phí gọi LLM. Lấy mẫu theo quy tắc: 100% lỗi, 100% chi phí cao, 5% thành công. Luôn giữ dữ liệu tổng hợp; giữ dữ liệu thô cho các trường hợp hiếm gặp.

### Các con số cần nhớ

- Langfuse cloud miễn phí: 50K sự kiện/tháng.
- LangSmith: $39/người dùng/tháng.
- Helicone miễn phí: 100K yêu cầu/tháng.
- Tuyên bố của Arize AX: rẻ hơn ~100 lần so với hệ thống nguyên khối ở quy mô lớn.
- Quy ước GenAI của OpenTelemetry: ra mắt 2025, phổ biến rộng rãi 2026.

```figure
i4-otel-glue
```

## Sử dụng

`code/main.py` mô phỏng một ngày với 1M trace qua các chiến lược lưu trữ (100% ingest, lấy mẫu, lấy mẫu + lỗi). Báo cáo chi phí lưu trữ và những gì bị mất trong mỗi trường hợp.

## Triển khai

Bài học này tạo ra `outputs/skill-observability-stack.md`. Dựa trên stack, quy mô, ngân sách, giấy phép, chọn (các) công cụ phù hợp.

## Bài tập

1. Nhóm của bạn dùng LangChain và muốn quan sát OSS self-hosted. Chọn Langfuse hoặc Opik và giải thích lý do.
2. Với 5M trace/ngày và báo giá Datadog là $150K/tháng, hãy tính điểm hòa vốn cho Arize AX.
3. Thiết kế một bộ thuộc tính OpenTelemetry GenAI mà quy định của tổ chức bạn nên bắt buộc trên mọi lệnh gọi LLM.
4. Tranh luận xem liệu chỉ riêng Phoenix có đủ cho sản xuất không. Khi nào thì nó không đủ?
5. Helicone có độ trễ proxy 20ms. Với P99 TTFT là 300ms, mức đó có chấp nhận được không? Nếu SLA là 100ms thì sao?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| OpenLLMetry | "OTel cho LLMs" | Instrumentation OpenTelemetry mã nguồn mở cho LLMs |
| GenAI conventions | "Thuộc tính OTel" | Tên thuộc tính OTel tiêu chuẩn cho các lệnh gọi LLM |
| LangSmith | "Quan sát LangChain" | Nền tảng thương mại tích hợp với hệ sinh thái LangChain |
| Langfuse | "LangSmith OSS" | OSS MIT với bộ tính năng tương tự |
| Phoenix | "Công cụ dev Arize" | Nền tảng dev/eval native OpenTelemetry |
| Arize AX | "Quan sát quy mô" | Quan sát thương mại zero-copy Iceberg/Parquet |
| Helicone | "Quan sát proxy" | Proxy HTTP thu thập telemetry LLM + tính năng gateway |
| Opik | "Comet LLM" | Nền tảng dev OSS Apache 2.0 từ Comet |
| Session replay | "Chạy lại trace" | Phát lại toàn bộ phiên agent với các lệnh gọi công cụ |
| Eval | "Kiểm tra offline" | Chạy model/prompt ứng viên trên tập dữ liệu đã gán nhãn |

## Đọc thêm

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Phân tích thay thế Arize AX](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Thiết lập Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Tài liệu Arize Phoenix](https://docs.arize.com/phoenix)
- [Tài liệu Helicone](https://docs.helicone.ai/)