# OpenTelemetry GenAI Semantic Conventions

> GenAI SIG của OpenTelemetry (ra mắt tháng 4 năm 2024) định nghĩa lược đồ tiêu chuẩn cho telemetry của agent. Tên span, thuộc tính và các quy tắc thu thập nội dung được thống nhất giữa các nhà cung cấp để các trace của agent có ý nghĩa đồng nhất trên Datadog, Grafana, Jaeger và Honeycomb.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 24 (Observability Platforms)
**Time:** ~60 phút

## Mục tiêu học tập

- Gọi tên các danh mục span GenAI: model/client, agent, tool.
- Phân biệt `invoke_agent` CLIENT và INTERNAL span và thời điểm áp dụng mỗi loại.
- Liệt kê các thuộc tính GenAI cấp cao nhất: tên nhà cung cấp, model yêu cầu, ID nguồn dữ liệu.
- Giải thích hợp đồng thu thập nội dung: opt-in, `OTEL_SEMCONV_STABILITY_OPT_IN`, khuyến nghị tham chiếu bên ngoài.

## Vấn đề

Mỗi nhà cung cấp tự đặt tên span riêng. Các đội ngũ vận hành (Ops) cuối cùng phải xây dựng các dashboard riêng cho từng framework. GenAI SIG của OpenTelemetry giải quyết vấn đề này bằng cách định nghĩa một tiêu chuẩn duy nhất mà toàn bộ hệ sinh thái hướng tới.

## Khái niệm

### Danh mục span

1. **Model / client spans.** Bao gồm các lệnh gọi LLM thô. Được phát ra bởi các SDK của nhà cung cấp (Anthropic, OpenAI, Bedrock) và các adapter model của framework.
2. **Agent spans.** `create_agent` (khi agent được khởi tạo) và `invoke_agent` (khi nó chạy).
3. **Tool spans.** Mỗi lần gọi tool là một span; được kết nối với agent span theo quan hệ cha-con.

### Đặt tên agent span

- Tên span: `invoke_agent {gen_ai.agent.name}` nếu được đặt tên; dự phòng là `invoke_agent`.
- Loại span (Span kind):
  - **CLIENT** — cho các dịch vụ agent từ xa (OpenAI Assistants API, Bedrock Agents).
  - **INTERNAL** — cho các framework agent chạy trong tiến trình (LangChain, CrewAI, local ReAct).

### Các thuộc tính chính

- `gen_ai.provider.name` — `anthropic`, `openai`, `aws.bedrock`, `google.vertex`.
- `gen_ai.request.model` — ID của model.
- `gen_ai.response.model` — model đã phân giải (có thể khác với yêu cầu do định tuyến).
- `gen_ai.agent.name` — định danh của agent.
- `gen_ai.operation.name` — `chat`, `completion`, `invoke_agent`, `tool_call`.
- `gen_ai.data_source.id` — cho RAG: corpus hoặc kho lưu trữ nào đã được tham vấn.

Các quy ước cụ thể theo công nghệ tồn tại cho Anthropic, Azure AI Inference, AWS Bedrock, OpenAI.

### Thu thập nội dung

Quy tắc mặc định: các instrumentation KHÔNG NÊN thu thập đầu vào/đầu ra theo mặc định. Việc thu thập là opt-in thông qua:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

Mô hình sản xuất được khuyến nghị: lưu trữ nội dung bên ngoài (S3, kho log của bạn), ghi lại các tham chiếu trên span (ID con trỏ, không phải văn bản thô). Đây là biện pháp phòng thủ chống "content-poisoning" từ Bài 27 được tích hợp vào khả năng quan sát (observability).

### Tính ổn định

Hầu hết các quy ước vẫn đang ở dạng thử nghiệm tính đến tháng 3 năm 2026. Hãy opt-in vào bản xem trước ổn định với:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ ánh xạ các thuộc tính GenAI một cách tự nhiên vào lược đồ LLM Observability của nó. Các backend khác (Grafana, Honeycomb, Jaeger) hỗ trợ các thuộc tính thô.

### Những sai lầm thường gặp

- **Thu thập toàn bộ prompt trong span.** PII, bí mật, dữ liệu khách hàng trong các trace mà đội ngũ vận hành có thể đọc được. Hãy lưu trữ bên ngoài.
- **Không có `gen_ai.provider.name`.** Các dashboard đa nhà cung cấp sẽ bị lỗi khi thiếu thông tin quy thuộc (attribution).
- **Span không có liên kết cha.** Các tool span bị mồ côi. Luôn luôn truyền ngữ cảnh (context).
- **Không thiết lập opt-in tính ổn định.** Các thuộc tính của bạn có thể bị đổi tên khi nâng cấp backend.

```figure
ae-genai-span-tree
```

## Xây dựng

`code/main.py` triển khai một bộ phát span stdlib phù hợp với các quy ước GenAI:

- `Span` với lược đồ thuộc tính GenAI.
- `Tracer` với `start_span`, các ngữ cảnh lồng nhau.
- Một agent chạy theo kịch bản phát ra: `create_agent`, `invoke_agent` (INTERNAL), các span cho từng tool, `chat` span cho các lệnh gọi LLM.
- Chế độ thu thập nội dung lưu trữ prompt bên ngoài và ghi lại ID trên các span.

Chạy nó:

```
python3 code/main.py
```

Đầu ra: một cây span với tất cả các thuộc tính GenAI bắt buộc, và một "kho lưu trữ bên ngoài" hiển thị các tham chiếu nội dung đã opt-in.

## Sử dụng

- **Datadog LLM Observability** (v1.37+) ánh xạ các thuộc tính một cách tự nhiên.
- **Langfuse / Phoenix / Opik** (Bài 24) — tự động instrument hệ sinh thái.
- **Jaeger / Honeycomb / Grafana Tempo** — các OTel trace thô; xây dựng dashboard từ các thuộc tính GenAI.
- **Self-hosted** — chạy OTel Collector với một bộ xử lý GenAI.

## Triển khai

`outputs/skill-otel-genai.md` kết nối các OTel GenAI span vào một agent hiện có với các mặc định thu thập nội dung và lưu trữ tham chiếu bên ngoài.

## Bài tập

1. Instrument vòng lặp ReAct ở Bài 01 của bạn với `invoke_agent` (INTERNAL) + các span cho từng tool. Gửi đến một instance Jaeger.
2. Thêm thu thập nội dung ở chế độ "chỉ tham chiếu": các prompt gửi vào SQLite, các thuộc tính span chỉ chứa ID dòng.
3. Đọc đặc tả cho `gen_ai.data_source.id`. Kết nối nó vào tìm kiếm Mem0 ở Bài 09 của bạn.
4. Thiết lập `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` và xác minh rằng các thuộc tính của bạn không bị đổi tên bởi collector.
5. Xây dựng một dashboard: "lỗi tool nào tương quan với model nào" chỉ từ các thuộc tính GenAI.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| GenAI SIG | "Nhóm GenAI của OpenTelemetry" | Nhóm làm việc của OTel định nghĩa lược đồ |
| invoke_agent | "Agent span" | Tên của span đại diện cho một lần chạy agent |
| CLIENT span | "Lệnh gọi từ xa" | Span cho một lệnh gọi đến dịch vụ agent từ xa |
| INTERNAL span | "Trong tiến trình" | Span cho một lần chạy agent trong tiến trình |
| gen_ai.provider.name | "Nhà cung cấp" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "Nguồn RAG" | Corpus/kho lưu trữ nào đã cung cấp kết quả truy xuất |
| Content capture | "Ghi log prompt" | Thu thập tin nhắn theo dạng opt-in; lưu trữ bên ngoài trong môi trường prod |
| Stability opt-in | "Chế độ xem trước" | Biến môi trường để cố định các quy ước thử nghiệm |

## Đọc thêm

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — đặc tả kỹ thuật
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — mặc định có các GenAI span
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — tích hợp sẵn OTel span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — truyền ngữ cảnh trace W3C