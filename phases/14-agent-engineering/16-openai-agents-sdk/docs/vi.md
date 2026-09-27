# OpenAI Agents SDK: Handoffs, Guardrails, Tracing

> OpenAI Agents SDK là framework đa tác nhân (multi-agent) gọn nhẹ được xây dựng trên Responses API. Bao gồm năm thành phần cơ bản: Agent, Handoff, Guardrail, Session, Tracing. Handoff là các công cụ có tên `transfer_to_<agent>`. Guardrail sẽ kích hoạt dựa trên đầu vào hoặc đầu ra. Tracing được bật theo mặc định.

**Type:** Học + Xây dựng
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 phút

## Mục tiêu học tập

- Nêu tên năm thành phần cơ bản của OpenAI Agents SDK.
- Giải thích về handoff: tại sao chúng được mô hình hóa như các công cụ, mô hình nhìn thấy tên dưới dạng nào và ngữ cảnh được chuyển giao như thế nào.
- Phân biệt guardrail đầu vào, guardrail đầu ra và guardrail công cụ; giải thích `run_in_parallel` so với chế độ chặn (blocking mode).
- Triển khai runtime bằng stdlib với handoff + guardrail + tracing kiểu span.

## Vấn đề

Các Agent không thể ủy quyền một cách sạch sẽ thường dẫn đến việc nhồi nhét mọi thứ vào một prompt duy nhất. Các Agent không có guardrail sẽ làm rò rỉ PII, tạo ra đầu ra vi phạm chính sách hoặc lặp vô tận. SDK của OpenAI hệ thống hóa ba thành phần cơ bản giúp việc xây dựng hệ thống đa tác nhân trở nên khả thi.

## Khái niệm

### Năm thành phần cơ bản

1. **Agent.** LLM + hướng dẫn + công cụ + handoff.
2. **Handoff.** Ủy quyền cho một agent khác. Được biểu diễn với mô hình như một công cụ có tên `transfer_to_<agent_name>`.
3. **Guardrail.** Kiểm chứng trên đầu vào (chỉ agent đầu tiên), đầu ra (chỉ agent cuối cùng) hoặc việc gọi công cụ (mỗi function tool).
4. **Session.** Lịch sử hội thoại tự động qua các lượt tương tác.
5. **Tracing.** Các span tích hợp sẵn cho việc tạo nội dung LLM, gọi công cụ, handoff, guardrail.

### Handoff dưới dạng công cụ

Mô hình nhìn thấy `transfer_to_billing_agent` trong danh sách công cụ của nó. Việc gọi công cụ này báo hiệu cho runtime:

1. Sao chép ngữ cảnh hội thoại (hoặc thu gọn nó thông qua `nest_handoff_history` beta).
2. Khởi tạo agent mục tiêu với các hướng dẫn của nó.
3. Tiếp tục quá trình chạy với agent mục tiêu.

Đây là mô hình supervisor (Bài 13 / Bài 28) được thương mại hóa.

### Guardrail

Ba loại:

- **Guardrail đầu vào.** Chạy trên đầu vào của agent đầu tiên. Từ chối các yêu cầu không an toàn hoặc nằm ngoài phạm vi trước khi bất kỳ lệnh gọi LLM nào diễn ra.
- **Guardrail đầu ra.** Chạy trên đầu ra của agent cuối cùng. Bắt lỗi rò rỉ PII, vi phạm chính sách, phản hồi sai định dạng.
- **Guardrail công cụ.** Chạy trên mỗi function tool. Kiểm chứng đối số, kiểm tra quyền hạn, kiểm toán việc thực thi.

Chế độ:

- **Song song (Parallel)** (mặc định). LLM của guardrail chạy cùng lúc với LLM chính. Độ trễ thấp hơn. Nếu bị kích hoạt (trip), công việc của LLM chính sẽ bị loại bỏ (lãng phí token).
- **Chặn (Blocking)** (`run_in_parallel=False`). LLM của guardrail chạy trước. Nếu bị kích hoạt, không có token nào bị lãng phí cho lệnh gọi chính.

Các tripwire sẽ kích hoạt `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`.

### Tracing

Được bật theo mặc định. Mỗi lần tạo LLM, gọi công cụ, handoff và guardrail đều phát ra một span. `OPENAI_AGENTS_DISABLE_TRACING=1` dùng để tắt tính năng này. `add_trace_processor(processor)` đẩy các span đến backend của bạn cùng với backend của OpenAI.

### Sessions

`Session` lưu trữ lịch sử hội thoại trong một backend (SQLite, Redis, tùy chỉnh). `Runner.run(agent, input, session=session)` tự động tải và nối thêm dữ liệu.

### Những điểm sai lầm thường gặp

- **Handoff drift.** Agent A chuyển cho Agent B, sau đó Agent B chuyển ngược lại Agent A. Hãy thêm bộ đếm số bước (hop counter).
- **Bỏ qua guardrail.** Guardrail công cụ chỉ kích hoạt trên các function tool; các công cụ tích hợp sẵn (đọc file, lấy dữ liệu web) cần chính sách riêng.
- **Over-tracing.** Nội dung nhạy cảm trong các span. Hãy kết hợp với các quy tắc thu thập nội dung OTel GenAI (Bài 23) — lưu trữ bên ngoài, tham chiếu bằng ID.

```figure
ae-agent-handoff
```

## Xây dựng

`code/main.py` triển khai cấu trúc SDK bằng stdlib:

- `Agent`, `FunctionTool`, `Handoff` (dưới dạng một function tool với ngữ nghĩa chuyển giao).
- `Runner` với guardrail đầu vào/đầu ra/công cụ, điều phối handoff và bộ đếm số bước.
- Một bộ phát span đơn giản để hiển thị cấu trúc trace.
- Một agent phân loại (triage agent) chuyển giao cho bộ phận thanh toán hoặc hỗ trợ dựa trên truy vấn của người dùng; guardrail kích hoạt trên một đầu vào.

Chạy nó:

```
python3 code/main.py
```

Trace hiển thị hai lần handoff thành công, một lần kích hoạt guardrail đầu vào và một cây span phản ánh những gì SDK thực tế phát ra.

## Sử dụng

- **OpenAI Agents SDK** cho các sản phẩm ưu tiên OpenAI.
- **Claude Agent SDK** (Bài 17) cho các sản phẩm ưu tiên Claude.
- **LangGraph** (Bài 13) khi bạn muốn trạng thái rõ ràng và khả năng khôi phục bền vững.
- **Tùy chỉnh** khi bạn cần kiểm soát chính xác (giọng nói, đa nhà cung cấp, triển khai liên kết).

## Triển khai

`outputs/skill-agents-sdk-scaffold.md` tạo khung cho một ứng dụng Agents SDK với agent phân loại, handoff, guardrail đầu vào/đầu ra/công cụ, lưu trữ session và bộ xử lý trace.

## Bài tập

1. Thêm bộ đếm số bước handoff: từ chối sau N lần chuyển giao. Trace hành vi này.
2. Triển khai `nest_handoff_history` như một tùy chọn — thu gọn các tin nhắn trước đó thành một bản tóm tắt trước khi chuyển giao.
3. Viết một guardrail đầu ra dạng chặn (blocking). So sánh độ trễ trên các prompt kích hoạt nó so với các prompt vượt qua.
4. Kết nối `add_trace_processor` với một trình ghi log JSON. Nó phát ra cấu trúc gì cho mỗi span?
5. Đọc tài liệu SDK. Chuyển đổi ứng dụng stdlib của bạn sang `openai-agents-python`. Bạn đã mô hình hóa sai ở đâu?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Agent | "LLM + hướng dẫn" | Loại Agent trong SDK; sở hữu công cụ và handoff |
| Handoff | "Chuyển giao" | Công cụ mà mô hình gọi để ủy quyền cho agent khác |
| Guardrail | "Kiểm tra chính sách" | Kiểm chứng trên đầu vào / đầu ra / gọi công cụ |
| Tripwire | "Kích hoạt guardrail" | Ngoại lệ được đưa ra khi guardrail từ chối |
| Session | "Lưu trữ lịch sử" | Bộ nhớ hội thoại được duy trì giữa các lần chạy |
| Tracing | "Spans" | Khả năng quan sát tích hợp trên LLM + công cụ + handoff + guardrail |
| Blocking guardrail | "Kiểm tra tuần tự" | Guardrail chạy trước; không lãng phí token khi bị kích hoạt |
| Parallel guardrail | "Kiểm tra đồng thời" | Guardrail chạy song song; độ trễ thấp hơn, lãng phí token khi bị kích hoạt |

## Đọc thêm

- [Tài liệu OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — các thành phần cơ bản, handoff, guardrail, tracing
- [Tổng quan Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — đối tác mang phong cách Claude
- [Anthropic, Xây dựng các Agent hiệu quả](https://www.anthropic.com/research/building-effective-agents) — khi nào nên sử dụng handoff
- [Quy ước ngữ nghĩa OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — tiêu chuẩn mà các span của Agents SDK ánh xạ tới