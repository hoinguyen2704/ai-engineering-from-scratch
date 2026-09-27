# A2A — Giao thức Agent-to-Agent

> MCP là giao thức agent-to-tool. A2A (Agent2Agent) là giao thức agent-to-agent — một giao thức mở cho phép các agent "hộp đen" (opaque) được xây dựng trên các framework khác nhau có thể cộng tác với nhau. Được Google phát hành vào tháng 4 năm 2025, chuyển giao cho Linux Foundation vào tháng 6 năm 2025, đạt phiên bản v1.0 vào tháng 4 năm 2026 với hơn 150 đơn vị ủng hộ bao gồm AWS, Cisco, Microsoft, Salesforce, SAP và ServiceNow. Giao thức này đã hấp thụ ACP của IBM và bổ sung phần mở rộng thanh toán AP2. Bài học này sẽ hướng dẫn về Agent Card, vòng đời Task và hai phương thức transport binding.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 phút

## Mục tiêu học tập

- Phân biệt các trường hợp sử dụng giữa agent-to-tool (MCP) và agent-to-agent (A2A).
- Xuất bản một Agent Card tại `/.well-known/agent.json` với metadata về kỹ năng và endpoint.
- Tìm hiểu vòng đời của Task (submitted → working → input-required → completed / failed / canceled / rejected).
- Sử dụng Messages với các Parts (text, file, data) và Artifacts làm đầu ra.

## Vấn đề

Một agent dịch vụ khách hàng cần ủy quyền việc viết báo cáo cho một agent chuyên gia viết lách. Các tùy chọn trước khi có A2A:

- Custom REST API: Hoạt động được nhưng mỗi cặp agent lại là một giải pháp riêng lẻ.
- Shared codebase: Yêu cầu hai agent phải chạy cùng một framework.
- MCP: Không phù hợp vì MCP dùng để gọi công cụ (tools), không phải để hai agent cộng tác trong khi vẫn bảo toàn tư duy nội bộ "hộp đen" của mỗi agent.

A2A lấp đầy khoảng trống này. Nó mô hình hóa tương tác dưới dạng một agent gửi Task cho agent khác, với vòng đời, tin nhắn và các artifact. Trạng thái nội bộ của agent được gọi vẫn là "hộp đen" — người gọi chỉ thấy các chuyển đổi trạng thái task và kết quả đầu ra cuối cùng.

A2A là giao thức "cho phép các agent trên các framework khác nhau trò chuyện với nhau". Nó không thay thế MCP; cả hai bổ trợ cho nhau.

## Khái niệm

### Agent Card

Mỗi agent tuân thủ A2A sẽ xuất bản một card tại `/.well-known/agent.json`:

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

Việc khám phá dựa trên URL: lấy card, tìm hiểu URL của endpoint A2A, liệt kê các kỹ năng (skills).

### Signed Agent Cards (AP2)

Phần mở rộng AP2 (tháng 9 năm 2025) bổ sung chữ ký số vào Agent Cards. Nhà xuất bản ký vào card của chính mình bằng JWT; người tiêu dùng sẽ xác thực. Điều này ngăn chặn việc giả mạo.

### Vòng đời Task

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

Các client khởi tạo với `tasks/send`. Agent được gọi sẽ chuyển đổi qua các trạng thái; client đăng ký nhận cập nhật trạng thái qua SSE hoặc polling.

### Messages và Parts

Một tin nhắn mang theo một hoặc nhiều Parts:

- `text` — nội dung văn bản thuần túy.
- `file` — blob base64 với mimeType.
- `data` — payload JSON có kiểu (dữ liệu đầu vào có cấu trúc cho agent được gọi).

Ví dụ:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

Đầu ra là các Artifacts, không phải chuỗi thô. Một Artifact là một đầu ra có tên và kiểu dữ liệu:

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts có thể được truyền tải (stream) dưới dạng các chunk. Người gọi sẽ tích lũy chúng.

### Hai transport bindings

1. **JSON-RPC over HTTP.** Endpoint `/a2a`, POST cho các yêu cầu, SSE tùy chọn cho streaming. Đây là binding mặc định.
2. **gRPC.** Dành cho môi trường doanh nghiệp nơi gRPC là tiêu chuẩn.

Cả hai binding đều mang cùng một cấu trúc tin nhắn logic.

### Bảo toàn tính "hộp đen" (Opacity)

Nguyên tắc thiết kế cốt lõi: trạng thái nội bộ của agent được gọi là "hộp đen". Người gọi chỉ thấy trạng thái task và các artifact. Chuỗi tư duy (chain-of-thought), các lệnh gọi công cụ, việc ủy quyền cho agent con của agent được gọi — tất cả đều vô hình. Điều này khác với MCP, nơi các lệnh gọi công cụ là minh bạch.

Lý do: A2A cho phép các đối thủ cạnh tranh cộng tác mà không tiết lộ nội bộ. A2A có thể là "gọi agent dịch vụ khách hàng này" mà người gọi không cần biết agent đó thực hiện dịch vụ như thế nào.

### Dòng thời gian

- **2025-04-09.** Google công bố A2A.
- **2025-06-23.** Chuyển giao cho Linux Foundation.
- **2025-08.** Hấp thụ ACP của IBM.
- **2025-09.** Ra mắt phần mở rộng AP2 (Agent Payments).
- **2026-04.** Phát hành v1.0 với hơn 150 tổ chức hỗ trợ.

### Mối quan hệ với MCP

| Khía cạnh | MCP | A2A |
|-----------|-----|-----|
| Trường hợp sử dụng | Agent-to-tool | Agent-to-agent |
| Tính minh bạch | Các lệnh gọi công cụ minh bạch | Tư duy nội bộ "hộp đen" |
| Người gọi điển hình | Agent runtime | Một agent khác |
| Trạng thái | Kết quả gọi công cụ | Task với vòng đời |
| Xác thực | OAuth 2.1 (Phase 13 · 16) | Agent Cards ký bằng JWT (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

Sử dụng MCP khi bạn muốn gọi một công cụ cụ thể. Sử dụng A2A khi bạn muốn ủy quyền toàn bộ một task cho một agent khác. Nhiều hệ thống sản xuất sử dụng cả hai: một agent sử dụng MCP cho lớp công cụ và A2A cho lớp cộng tác.

```figure
a2a-task-lifecycle
```

## Sử dụng

`code/main.py` triển khai một harness A2A tối giản: một agent nghiên cứu xuất bản card của nó, một agent viết lách nhận `tasks/send` với các phần bao gồm PDF và hướng dẫn văn bản, chuyển đổi qua các trạng thái working → input_required → working → completed, và trả về một artifact văn bản. Tất cả đều dùng thư viện chuẩn; sử dụng transport trong bộ nhớ để tập trung vào cấu trúc tin nhắn.

Những điểm cần chú ý:

- Cấu trúc JSON của Agent Card.
- Gán ID cho Task và các chuyển đổi trạng thái.
- Tin nhắn với các phần có kiểu hỗn hợp.
- Nhánh input-required giữa task.
- Trả về Artifact khi hoàn thành.

## Triển khai

Bài học này tạo ra `outputs/skill-a2a-agent-spec.md`. Với một agent mới cần được các agent khác gọi, kỹ năng này tạo ra JSON của Agent Card, schema kỹ năng và bản thiết kế endpoint.

## Bài tập

1. Chạy `code/main.py`. Theo dõi toàn bộ vòng đời Task, bao gồm cả trạng thái tạm dừng input-required khi agent được gọi yêu cầu làm rõ thông tin.

2. Thêm một Agent Card có chữ ký. Ký bằng HMAC trên JSON chuẩn hóa của card. Viết trình xác thực và xác nhận nó thất bại nếu card bị thay đổi.

3. Triển khai task streaming: agent viết lách phát ra ba chunk artifact tăng dần qua SSE và người gọi tích lũy chúng.

4. Thiết kế một agent A2A bao bọc (wrap) một MCP server. Ánh xạ mỗi công cụ MCP thành một kỹ năng A2A. Lưu ý các đánh đổi — tính "hộp đen" nào bị mất đi?

5. Đọc thông báo A2A v1.0 và xác định một tính năng chưa được triển khai bởi bất kỳ framework nào tính đến tháng 4 năm 2026. (Gợi ý: liên quan đến việc ủy quyền task đa chặng - multi-hop).

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| A2A | "Giao thức Agent-to-Agent" | Giao thức mở cho cộng tác agent "hộp đen" |
| Agent Card | "`.well-known/agent.json`" | Metadata đã xuất bản mô tả kỹ năng và endpoint của agent |
| Skill | "Đơn vị có thể gọi" | Một thao tác được đặt tên mà agent hỗ trợ (tương tự công cụ MCP) |
| Task | "Đơn vị ủy quyền" | Một mục công việc có vòng đời và artifact cuối cùng |
| Message | "Đầu vào Task" | Mang theo các Parts (văn bản, tệp, dữ liệu) |
| Part | "Chunk có kiểu" | Phần tử `text` / `file` / `data` của tin nhắn |
| Artifact | "Đầu ra Task" | Đầu ra có tên, có kiểu được trả về khi hoàn thành |
| AP2 | "Giao thức thanh toán Agent" | Phần mở rộng Agent Cards có chữ ký cho sự tin cậy và thanh toán |
| Opacity | "Cộng tác hộp đen" | Nội bộ của agent được gọi bị ẩn khỏi người gọi |
| Input-required | "Tạm dừng Task" | Trạng thái vòng đời khi agent cần thêm thông tin |

## Đọc thêm

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — đặc tả A2A chính thức
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — các triển khai tham chiếu và SDK
- [Linux Foundation — Thông cáo báo chí ra mắt A2A](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — chuyển giao quản trị tháng 6 năm 2025
- [Google Cloud — Nâng cấp giao thức A2A](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — lộ trình và động lực đối tác
- [Google Dev — Cột mốc A2A 1.0](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — ghi chú phát hành v1.0 và hướng dẫn tương thích ngược