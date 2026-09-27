# OpenTelemetry GenAI — Tracing Tool Calls End-to-End

> Một agent gọi năm công cụ, ba MCP server và hai sub-agent. Bạn cần một trace duy nhất xuyên suốt toàn bộ quá trình này. Các quy ước ngữ nghĩa (semantic conventions) GenAI của OpenTelemetry (các thuộc tính ổn định từ v1.37 trở lên) là tiêu chuẩn của năm 2026, được hỗ trợ nguyên bản bởi Datadog, Langfuse, Arize Phoenix, OpenLLMetry và AgentOps. Bài học này liệt kê các thuộc tính bắt buộc, hướng dẫn phân cấp span (agent → LLM → tool) và cung cấp một stdlib span emitter mà bạn có thể tích hợp vào bất kỳ OTel exporter nào.

**Type:** Build
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~75 phút

## Mục tiêu học tập

- Nắm được các thuộc tính OTel GenAI bắt buộc cho một LLM span và một tool-execution span.
- Xây dựng hệ thống phân cấp trace bao phủ vòng lặp agent, lệnh gọi LLM, lệnh gọi công cụ và điều phối MCP client.
- Quyết định nội dung nào cần thu thập (opt-in) so với nội dung cần ẩn (mặc định).
- Phát hành các span tới một collector cục bộ (Jaeger, Langfuse) mà không cần viết lại mã nguồn công cụ.

## Vấn đề

Một lỗi được gỡ từ tháng 2 năm 2026: người dùng báo cáo "agent của tôi đôi khi mất 30 giây để phản hồi; những lúc khác chỉ mất 3 giây." Không có trace nào cả. Logs hiển thị lệnh gọi LLM, nhưng không thấy việc điều phối công cụ, không thấy round-trip của MCP server, cũng không thấy sub-agent. Bạn chỉ có thể đoán. Cuối cùng bạn phát hiện ra: một MCP server thỉnh thoảng bị treo do cold-start.

Nếu không có end-to-end tracing, bạn không thể tìm ra nguyên nhân này. OTel GenAI giải quyết vấn đề đó.

Các quy ước đã được thống nhất trong giai đoạn 2025-2026 dưới nhóm quy ước ngữ nghĩa OpenTelemetry. Chúng định nghĩa các tên thuộc tính ổn định để Datadog, Langfuse, Phoenix, OpenLLMetry và AgentOps đều phân tích cùng một loại span. Chỉ cần instrument một lần; gửi tới bất kỳ backend nào.

## Khái niệm

### Phân cấp Span

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

Toàn bộ quá trình nằm dưới một trace id duy nhất. Các span id liên kết các mối quan hệ cha-con.

### Các thuộc tính bắt buộc

Theo semconv 2025-2026:

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`.
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`.
- `gen_ai.request.model` — chuỗi model được yêu cầu (ví dụ: `"gpt-4o-2024-08-06"`).
- `gen_ai.response.model` — model thực tế được phục vụ.
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id` — id phản hồi của nhà cung cấp để đối chiếu.

Đối với tool spans:

- `gen_ai.tool.name` — định danh công cụ.
- `gen_ai.tool.call.id` — id lệnh gọi cụ thể.
- `gen_ai.tool.description` — mô tả công cụ (tùy chọn).

Đối với agent spans:

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### Các loại Span (Span kinds)

- `SpanKind.CLIENT` cho các lệnh gọi vượt qua ranh giới tiến trình (LLM provider, MCP server).
- `SpanKind.INTERNAL` cho các bước vòng lặp của chính agent và việc thực thi công cụ.

### Thu thập nội dung Opt-in

Theo mặc định, các span mang theo số liệu (metrics) và thời gian — không bao gồm prompt hoặc completion. Các payload lớn và PII bị tắt theo mặc định. Thiết lập `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` và các biến môi trường thu thập nội dung cụ thể để bao gồm nội dung. Hãy xem xét kỹ lưỡng trước khi kích hoạt trong môi trường production.

### Sự kiện trên Span (Events on spans)

Các sự kiện ở cấp độ token có thể được thêm vào dưới dạng span events:

- `gen_ai.content.prompt` — các tin nhắn đầu vào.
- `gen_ai.content.completion` — các tin nhắn đầu ra.
- `gen_ai.content.tool_call` — lệnh gọi công cụ như đã ghi lại.

Các sự kiện được sắp xếp theo thời gian trong một span để phát lại chi tiết.

### Exporters

Các OTel span xuất ra:

- **Jaeger / Tempo.** OSS, on-prem.
- **Langfuse.** Chuyên biệt cho LLM-observability; trực quan hóa việc sử dụng token.
- **Arize Phoenix.** Kết hợp Evals + tracing.
- **Datadog.** Thương mại; phân tích nguyên bản các thuộc tính `gen_ai.*`.
- **Honeycomb.** Hướng cột; thân thiện với truy vấn.

Tất cả đều sử dụng OTLP, định dạng truyền tải. Mã nguồn của bạn không cần quan tâm đến điều này.

### Lan truyền qua MCP

Khi một MCP client gọi một server, hãy inject header W3C traceparent vào request. HTTP hỗ trợ streaming hỗ trợ các header tiêu chuẩn. Stdio không mang theo header HTTP nguyên bản; lộ trình năm 2026 của đặc tả thảo luận về việc thêm trường `_meta.traceparent` vào các lệnh gọi JSON-RPC.

Cho đến khi tính năng đó được phát hành: hãy bao gồm traceparent trong `_meta` của mọi request theo cách thủ công. Server sẽ ghi lại trace id.

### Metrics

Bên cạnh các span, GenAI semconv định nghĩa các số liệu:

- `gen_ai.client.token.usage` — histogram.
- `gen_ai.client.operation.duration` — histogram.
- `gen_ai.tool.execution.duration` — histogram.

Sử dụng các số liệu này cho các bảng điều khiển không cần chi tiết từng lệnh gọi.

### Lớp AgentOps

AgentOps (thành lập năm 2024) chuyên về khả năng quan sát GenAI. Nó bao bọc các framework phổ biến (LangGraph, Pydantic AI, CrewAI) để tự động phát hành các OTel span. Hữu ích nếu stack của bạn sử dụng một framework được hỗ trợ; nếu không, hãy sử dụng instrumentation thủ công.

```figure
t3-span-waterfall
```

## Sử dụng

`code/main.py` phát hành các span theo định dạng OTel ra stdout (ở định dạng giống OTLP-JSON) cho một agent gọi LLM, điều phối hai công cụ và thực hiện một round-trip MCP. Không có exporter thực tế nào — bài học tập trung vào hình dạng span và tập hợp thuộc tính. Dán đầu ra vào một trình xem tương thích OTLP hoặc chỉ cần đọc nó.

Những điều cần chú ý:

- Trace id được chia sẻ trên tất cả các span.
- Các liên kết cha-con được mã hóa thông qua `parentSpanId`.
- Các thuộc tính `gen_ai.*` bắt buộc đã được điền.
- Thu thập nội dung bị tắt theo mặc định; một kịch bản sẽ bật nó thông qua biến môi trường.

## Triển khai

Bài học này tạo ra `outputs/skill-otel-genai-instrumentation.md`. Với một codebase agent, kỹ năng này tạo ra một kế hoạch instrumentation: nơi thêm span, thuộc tính nào cần điền và exporter nào cần nhắm tới.

## Bài tập

1. Chạy `code/main.py`. Đếm số lượng span và xác định cái nào là CLIENT, cái nào là INTERNAL.

2. Bật thu thập nội dung (biến môi trường) và xác nhận các sự kiện `gen_ai.content.prompt` và `gen_ai.content.completion` xuất hiện. Lưu ý các tác động đối với PII.

3. Thêm số liệu thực thi công cụ `gen_ai.tool.execution.duration` và phát hành nó dưới dạng mẫu histogram cho mỗi lệnh gọi.

4. Lan truyền một traceparent từ một span agent cha vào trường `_meta.traceparent` của một request MCP. Xác minh rằng MCP server sẽ thấy cùng một trace id.

5. Đọc đặc tả OTel GenAI semconv. Xác định một thuộc tính được liệt kê trong semconv mà mã nguồn của bài học này KHÔNG phát hành. Hãy thêm nó vào.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | Tiêu chuẩn mở cho traces, metrics, logs |
| GenAI semconv | "GenAI semantic conventions" | Tên thuộc tính ổn định cho các span LLM / tool / agent |
| `gen_ai.*` | "The attribute namespace" | Tất cả thuộc tính GenAI đều chia sẻ tiền tố này |
| Span | "Timed operation" | Một đơn vị công việc có điểm bắt đầu, kết thúc và các thuộc tính |
| Trace | "Cross-span ancestry" | Cây các span chia sẻ chung một trace id |
| SpanKind | "CLIENT / SERVER / INTERNAL" | Gợi ý về hướng của span |
| OTLP | "OpenTelemetry Line Protocol" | Định dạng truyền tải cho các exporter |
| Opt-in content | "Prompt / completion capture" | Tắt theo mặc định; dùng biến môi trường để bật |
| traceparent | "W3C header" | Lan truyền ngữ cảnh trace qua các dịch vụ |
| Exporter | "Backend-specific shipper" | Thành phần gửi span tới Jaeger / Datadog / v.v. |

## Đọc thêm

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — các quy ước chính thống cho span, metrics và events GenAI
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — danh sách thuộc tính span cho LLM và thực thi công cụ
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — span `invoke_agent` ở cấp độ agent
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — nguồn sự thật trên GitHub
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — hướng dẫn tích hợp vào môi trường production