# Capstone 11 — LLM Observability & Eval Dashboard

> Langfuse đã chuyển sang mô hình open-core. Arize Phoenix đã công bố các ánh xạ semconv GenAI cho năm 2026. Helicone và Braintrust đều tập trung mạnh vào việc phân bổ chi phí theo từng người dùng. OpenLLMetry của Traceloop đã trở thành SDK instrumentation tiêu chuẩn thực tế. Cấu trúc hệ thống sản xuất bao gồm ClickHouse cho các trace, Postgres cho metadata, Next.js cho UI, và một đội ngũ các job đánh giá (DeepEval, RAGAS, LLM-judge) chạy trên các trace được lấy mẫu. Hãy xây dựng một hệ thống tự lưu trữ (self-hosted), thu thập dữ liệu từ ít nhất bốn nhóm SDK và chứng minh khả năng phát hiện một lỗi hồi quy (regression) được cài cắm trong vòng chưa đầy năm phút.

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (thu thập dữ liệu + đánh giá), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (công cụ), Phase 17 (cơ sở hạ tầng), Phase 18 (an toàn)
**Phases exercised:** P11 · P13 · P17 · P18
**Time:** 25 giờ

## Vấn đề

Mọi đội ngũ AI vận hành lưu lượng truy cập thực tế vào năm 2026 đều duy trì một mặt phẳng quan sát (observability plane) song song với mô hình. Phân bổ chi phí. Phát hiện ảo tưởng (hallucination). Giám sát độ lệch (drift). Tín hiệu bẻ khóa (jailbreak). Bảng điều khiển SLO. Cảnh báo rò rỉ PII. Các tài liệu tham khảo mã nguồn mở — Langfuse, Phoenix, OpenLLMetry — đã hội tụ về các quy ước ngữ nghĩa (semantic conventions) GenAI của OpenTelemetry làm lược đồ thu thập dữ liệu. Giờ đây, bạn có thể instrument OpenAI, Anthropic, Google, LangChain, LlamaIndex và vLLM bằng một SDK duy nhất và gửi các span tương thích.

Bạn sẽ xây dựng một bảng điều khiển tự lưu trữ thu thập dữ liệu từ ít nhất bốn nhóm SDK, chạy một tập hợp nhỏ các job đánh giá trên các trace được lấy mẫu, phát hiện độ lệch và gửi cảnh báo. Tiêu chuẩn đo lường: với một lỗi hồi quy được cố tình cài cắm (một prompt bắt đầu tạo ra PII), bảng điều khiển sẽ phát hiện và kích hoạt cảnh báo trong vòng chưa đầy năm phút.

## Khái niệm

Thu thập dữ liệu thông qua OTLP HTTP. SDK tạo ra các span theo GenAI-semconv: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.id`, `llm.prompts`, `llm.completions`. Các span được lưu vào ClickHouse để phân tích theo cột; metadata (người dùng, phiên làm việc, ứng dụng) được lưu vào Postgres.

Các đánh giá chạy dưới dạng job hàng loạt trên các trace được lấy mẫu. DeepEval chấm điểm độ trung thực (faithfulness), độc hại (toxicity) và mức độ liên quan của câu trả lời (answer relevance). RAGAS chấm điểm các chỉ số truy xuất khi trace chứa ngữ cảnh truy xuất. Các LLM-judge tùy chỉnh thực hiện các kiểm tra đặc thù theo miền (rò rỉ PII, phản hồi ngoài chính sách). Các kết quả đánh giá được ghi ngược lại vào cùng ClickHouse dưới dạng các span đánh giá được liên kết với trace cha.

Phát hiện độ lệch theo dõi sự phân phối trong không gian embedding theo thời gian (PSI hoặc KL divergence trên các embedding của prompt) cộng với xu hướng điểm đánh giá. Cảnh báo được gửi đến Prometheus Alertmanager và sau đó là Slack / PagerDuty. UI sử dụng Next.js 15 với Recharts.

## Kiến trúc

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## Stack

- Thu thập dữ liệu: OpenTelemetry SDKs + GenAI semantic conventions; truyền tải OTLP HTTP
- Collector: OpenTelemetry Collector với bộ xử lý tail-sampling (để kiểm soát chi phí)
- Lưu trữ: ClickHouse cho các span, Postgres cho metadata, S3 để lưu trữ sự kiện thô
- Đánh giá: DeepEval, RAGAS 0.2, Arize Phoenix evaluator pack, LLM-judge tùy chỉnh
- Độ lệch: PSI / KL trên các embedding prompt được gộp lại (sentence-transformers) hàng tuần
- Cảnh báo: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + server actions
- Các SDK được hỗ trợ sẵn: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

```figure
ce-otel-drift
```

## Xây dựng

1. **Cấu hình Collector.** OpenTelemetry Collector với bộ thu OTLP HTTP, bộ lấy mẫu đuôi (tail-sampler) giữ lại 100% các trace lỗi và 10% các trace thành công, cùng các exporter gửi đến ClickHouse và S3.

2. **Lược đồ ClickHouse.** Bảng `spans` với các cột phản ánh GenAI semconv: `gen_ai_system`, `gen_ai_request_model`, `input_tokens`, `output_tokens`, `latency_ms`, `prompt_hash`, `trace_id`, `parent_span_id`, cộng với một trường JSON cho các payload dài. Thêm các chỉ mục phụ theo user_id và app_id.

3. **Kiểm tra độ bao phủ SDK.** Viết một ứng dụng client nhỏ sử dụng từng SDK (OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM) với tính năng tự động instrument của OpenLLMetry. Xác minh rằng mỗi SDK tạo ra các span GenAI chuẩn được lưu vào ClickHouse.

4. **Job đánh giá.** Một job theo lịch trình đọc các trace được lấy mẫu trong 15 phút gần nhất và chạy các đánh giá của DeepEval về độ trung thực, độc hại và mức độ liên quan. Kết quả đầu ra là các span đánh giá được liên kết với trace cha.

5. **LLM-judge tùy chỉnh.** Một bộ đánh giá rò rỉ PII: với một phản hồi, gọi một guard LLM để chấm điểm khả năng rò rỉ PII. Các phản hồi có điểm cao sẽ được đưa vào hàng đợi phân loại.

6. **Phát hiện độ lệch.** Job hàng tuần tính toán PSI giữa các embedding prompt được gộp lại của tuần này và đường cơ sở của 4 tuần trước đó. Nếu PSI vượt ngưỡng, gửi cảnh báo.

7. **Bảng điều khiển.** Next.js 15 với các trang: tổng quan (span/giây, chi phí/người dùng, độ trễ p95), trace (tìm kiếm + waterfall), đánh giá (xu hướng độ trung thực, độc hại), độ lệch (PSI theo thời gian), cảnh báo.

8. **Chuỗi cảnh báo.** Prometheus exporter đọc các tổng hợp điểm đánh giá và phân vị độ trễ; Alertmanager định tuyến đến Slack cho các cảnh báo và PagerDuty cho các vi phạm nghiêm trọng.

9. **Kiểm tra hồi quy.** Cài cắm một lỗi: chatbot được đánh giá bắt đầu làm rò rỉ các SSN giả với tỷ lệ 1%. Đo lường MTTR: từ khi lỗi được triển khai đến khi có cảnh báo trên Slack.

## Sử dụng

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## Triển khai

`outputs/skill-llm-observability.md` là sản phẩm bàn giao. Với một ứng dụng LLM, bảng điều khiển sẽ thu thập các trace của nó, chạy đánh giá, cảnh báo về độ lệch và hiển thị bảng phân tích chi phí/người dùng trong Next.js.

| Trọng số | Tiêu chí | Cách đo lường |
|:-:|---|---|
| 25 | Độ bao phủ lược đồ trace | Số lượng nhóm SDK tạo ra các span GenAI chuẩn (mục tiêu: 6+) |
| 20 | Độ chính xác đánh giá | Điểm DeepEval / RAGAS so với tập dữ liệu được gán nhãn thủ công |
| 20 | UX bảng điều khiển | MTTR đối với lỗi hồi quy được cài cắm (mục tiêu dưới 5 phút) |
| 20 | Chi phí / quy mô | Thu thập dữ liệu ổn định ở mức 1k span/giây mà không bị tồn đọng |
| 15 | Cảnh báo + phát hiện độ lệch | Chuỗi Prometheus/Alertmanager được thực thi từ đầu đến cuối |
| **100** | | |

## Bài tập

1. Thêm instrumentation tùy chỉnh cho framework Haystack. Xác minh các span chuẩn được lưu vào ClickHouse với các thuộc tính `gen_ai.*` chính xác.

2. Thay thế DeepEval bằng các bộ đánh giá của Phoenix trên cùng các trace đó. Đo lường độ lệch điểm số giữa hai công cụ đánh giá.

3. Tối ưu hóa bộ phát hiện độ lệch: tính toán PSI theo từng app-id thay vì toàn cục. Hiển thị các dấu vết độ lệch theo từng ứng dụng.

4. Thêm trang "tác động người dùng": chi phí trên mỗi người dùng và tỷ lệ lỗi trên mỗi người dùng với các biểu đồ sparkline.

5. Xây dựng chính sách lấy mẫu đuôi (tail-sampling) giữ lại 100% các trace có độ độc hại > 0.5 cộng với 10% mẫu phân tầng của phần còn lại. Đo lường độ lệch lấy mẫu (sampling bias) được tạo ra.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| GenAI semconv | "Thuộc tính OTel LLM" | Đặc tả OpenTelemetry 2025 cho các thuộc tính span LLM (system, model, tokens) |
| Tail sampling | "Lấy mẫu sau trace" | Collector quyết định giữ hoặc loại bỏ trace sau khi nó hoàn tất (có thể xem trước lỗi) |
| PSI | "Chỉ số ổn định dân số" | Chỉ số độ lệch so sánh hai phân phối; > 0.2 thường báo hiệu độ lệch đáng kể |
| LLM-judge | "Đánh giá bằng mô hình" | Một LLM chấm điểm đầu ra của một LLM khác dựa trên một bộ tiêu chí (độ trung thực, độc hại, PII) |
| Tail-sampling policy | "Quy tắc giữ" | Quy tắc quyết định trace nào được lưu trữ so với loại bỏ; dựa trên lỗi + tỷ lệ lấy mẫu |
| Eval span | "Trace đánh giá liên kết" | Span con chứa điểm đánh giá được liên kết với span gọi LLM gốc |
| Cost per user | "Kinh tế đơn vị" | Chi phí đô la được phân bổ cho một user_id trong một khoảng thời gian; chỉ số sản phẩm quan trọng |

## Đọc thêm

- [Langfuse](https://github.com/langfuse/langfuse) — nền tảng quan sát open-core tham chiếu
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — tham chiếu thay thế với hỗ trợ độ lệch mạnh mẽ
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — nhóm SDK tự động instrument
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — lược đồ thu thập dữ liệu
- [Helicone](https://www.helicone.ai) — nền tảng quan sát được lưu trữ thay thế
- [Braintrust](https://www.braintrust.dev) — nền tảng ưu tiên đánh giá thay thế
- [Tài liệu ClickHouse](https://clickhouse.com/docs) — kho lưu trữ span dạng cột
- [DeepEval](https://github.com/confident-ai/deepeval) — thư viện đánh giá