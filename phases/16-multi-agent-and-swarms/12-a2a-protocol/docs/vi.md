# A2A — Giao thức Agent-to-Agent

> Google công bố A2A vào tháng 4 năm 2025; đến tháng 4 năm 2026, đặc tả kỹ thuật đã đạt phiên bản https://a2a-protocol.org/latest/specification/ và được hơn 150 tổ chức hỗ trợ. A2A là sự bổ sung theo chiều ngang cho MCP (Bài 13): trong khi MCP mang tính chiều dọc (agent ↔ công cụ), thì A2A mang tính ngang hàng (agent ↔ agent). Nó định nghĩa Agent Card (khám phá), các tác vụ với artifact (văn bản, dữ liệu cấu trúc, video), vòng đời tác vụ ẩn (opaque) và xác thực. Các hệ thống sản xuất ngày càng kết hợp MCP với A2A. Google Cloud đã tích hợp hỗ trợ A2A vào Vertex AI Agent Builder trong giai đoạn 2025-2026.

**Type:** Học + Xây dựng
**Languages:** Python (stdlib, `http.server`, `json`)
**Prerequisites:** Giai đoạn 16 · 04 (Primitive Model)
**Time:** ~75 phút

## Vấn đề

Agent của bạn cần gọi một agent khác trên một hệ thống khác. Làm thế nào? Bạn có thể mở một HTTP endpoint, định nghĩa một JSON schema tùy chỉnh và hy vọng phía bên kia hiểu được nó. Mỗi cặp agent trở thành một tích hợp tùy chỉnh.

A2A là giao thức truyền tin phổ quát cho yêu cầu đó. Khám phá tiêu chuẩn, mô hình tác vụ tiêu chuẩn, truyền tải tiêu chuẩn, artifact tiêu chuẩn. Giống như HTTP+REST nhưng dành cho các agent với tư cách là thực thể hạng nhất.

## Khái niệm

### Bốn yếu tố cốt lõi

**Agent Card.** Một tài liệu JSON tại `/.well-known/agent.json` mô tả agent: tên, kỹ năng, endpoint, các phương thức hỗ trợ, yêu cầu xác thực. Việc khám phá diễn ra bằng cách đọc thẻ này.

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task (Tác vụ).** Đơn vị công việc. Một đối tượng bất đồng bộ, có trạng thái với vòng đời: `submitted → working → completed / failed / canceled`. Client gửi một tác vụ, sau đó polling hoặc đăng ký nhận cập nhật.

**Artifact.** Loại kết quả được tạo ra bởi một tác vụ. Văn bản, JSON cấu trúc, hình ảnh, video, âm thanh. Các artifact được định kiểu để các phương thức khác nhau trở thành thực thể hạng nhất.

**Opaque lifecycle (Vòng đời ẩn).** A2A không quy định *cách thức* agent từ xa giải quyết tác vụ. Client chỉ thấy các chuyển đổi trạng thái và artifact; việc triển khai được tự do sử dụng bất kỳ framework nào.

### Phân biệt MCP và A2A

- **MCP** (Bài 13): agent ↔ công cụ. Agent đọc/ghi thông qua JSON-RPC tới một tool server. Mặc định là không trạng thái (stateless).
- **A2A**: agent ↔ agent. Giao thức ngang hàng; cả hai bên đều là các agent với khả năng suy luận riêng.

Các hệ thống đa agent trong sản xuất sử dụng cả hai. Một peer A2A sẽ gọi các công cụ MCP ở phía của nó. Sự phân tách này giữ cho hai mối quan tâm được tách biệt rõ ràng.

### Luồng khám phá

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

Hoặc với streaming: Đăng ký SSE tới `/tasks/{id}/events` để nhận cập nhật đẩy (push updates).

### Xác thực

A2A hỗ trợ ba mô hình phổ biến:

- **Bearer token** — OAuth2 hoặc opaque.
- **mTLS** — mutual TLS; các tổ chức chứng minh danh tính cho nhau.
- **Signed requests** — HMAC trên payload.

Xác thực được khai báo trong Agent Card; client khám phá và tuân thủ.

### 150+ tổ chức vào tháng 4 năm 2026

Việc áp dụng trong doanh nghiệp đã thúc đẩy quy mô của A2A. Điểm nhấn: A2A đã trở thành cách thức để các hệ thống agent doanh nghiệp vượt qua các ranh giới tin cậy. Google Cloud đã phát hành hỗ trợ A2A cho Vertex AI Agent Builder; Microsoft Agent Framework cũng hỗ trợ nó; hầu hết các framework lớn (LangGraph, CrewAI, AutoGen) đều cung cấp các adapter cho A2A.

### Khi nào A2A chiếm ưu thế

- **Các cuộc gọi liên tổ chức.** Agent tại công ty A gọi agent tại công ty B. Nếu không có A2A, mỗi cặp kết nối là một hợp đồng tùy chỉnh.
- **Các framework không đồng nhất.** Agent LangGraph gọi agent CrewAI gọi agent Python tùy chỉnh. A2A chuẩn hóa mọi thứ.
- **Artifact được định kiểu.** Kết quả video, JSON cấu trúc, âm thanh — tất cả đều là thực thể hạng nhất.
- **Tác vụ chạy dài.** Vòng đời ẩn + polling giúp các tác vụ kéo dài hàng giờ trở nên đơn giản.

### Khi nào A2A gặp khó khăn

- **Các cuộc gọi vi mô nhạy cảm với độ trễ.** Vòng đời của A2A là bất đồng bộ. Việc gọi agent-to-agent dưới một mili giây không phù hợp; hãy sử dụng RPC trực tiếp.
- **Các agent gắn kết chặt chẽ trong cùng tiến trình.** Nếu cả hai agent chạy trong cùng một tiến trình Python, việc round-trip qua HTTP của A2A là quá mức cần thiết.
- **Các nhóm nhỏ.** Chi phí đặc tả là có thật; các agent chỉ dùng nội bộ có thể không cần sự hình thức này.

### A2A so với ACP, ANP, NLIP

Một số đặc tả liên quan đã xuất hiện trong giai đoạn 2024-2026:

- **ACP** (IBM/Linux Foundation) — tiền thân của A2A, phạm vi hẹp hơn.
- **ANP** (Agent Network Protocol) — tập trung vào khám phá ngang hàng, ưu tiên phi tập trung.
- **NLIP** (Ecma Natural Language Interaction Protocol, chuẩn hóa tháng 12 năm 2025) — loại nội dung ngôn ngữ tự nhiên.

A2A là giao thức ngang hàng được áp dụng rộng rãi nhất tính đến tháng 4 năm 2026. Xem arXiv:2505.02279 (Liu và cộng sự, "A Survey of Agent Interoperability Protocols") để so sánh.

```figure
sw-agent-card-discovery
```

## Xây dựng

`code/main.py` triển khai một server và client A2A tối giản sử dụng `http.server` và JSON. Server:

- mở `/.well-known/agent.json`,
- chấp nhận `POST /tasks`,
- quản lý trạng thái tác vụ,
- trả về artifact tại `GET /tasks/{id}`.

Client:

- lấy Agent Card,
- gửi một tác vụ,
- polling cho đến khi hoàn thành,
- đọc artifact.

Chạy:

```
python3 code/main.py
```

Script khởi động server trong một luồng nền, sau đó chạy client tương tác với nó. Bạn sẽ thấy toàn bộ luồng: khám phá, gửi, polling, artifact.

## Sử dụng

`outputs/skill-a2a-integrator.md` thiết kế một tích hợp A2A: nội dung Agent Card, schema tác vụ, lựa chọn xác thực, streaming so với polling.

## Triển khai

Danh sách kiểm tra:

- **Ghim phiên bản đặc tả.** A2A vẫn đang phát triển; Agent Card nên khai báo phiên bản giao thức.
- **Tạo tác vụ idempotent.** Các lần gửi trùng lặp (do thử lại mạng) chỉ nên tạo ra một tác vụ duy nhất.
- **Schema cho artifact.** Khai báo các định dạng mà agent trả về; người tiêu dùng nên thực hiện kiểm chứng.
- **Giới hạn tốc độ + xác thực.** A2A hướng ra công chúng; hãy áp dụng bảo mật web tiêu chuẩn.
- **Dead-letter cho các tác vụ thất bại.** Kiểm tra các mẫu theo thời gian để tìm ra các loại lỗi lặp lại.

## Bài tập

1. Chạy `code/main.py`. Xác nhận client khám phá được server và nhận đúng artifact.
2. Thêm kỹ năng thứ hai vào server (ví dụ: "tóm tắt"). Cập nhật Agent Card. Viết một client chọn kỹ năng dựa trên loại tác vụ.
3. Triển khai một endpoint streaming SSE: `/tasks/{id}/events` phát ra các thay đổi trạng thái. Client cần làm gì khác biệt?
4. Đọc đặc tả A2A (https://a2a-protocol.org/latest/specification/). Xác định ba điều mà đặc tả yêu cầu nhưng bản demo này chưa triển khai.
5. So sánh A2A (khám phá qua Agent Card) với MCP (liệt kê khả năng phía server qua `listTools`). Đánh đổi giữa các agent tự mô tả và việc thăm dò khả năng là gì?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| A2A | "Agent-to-agent" | Giao thức ngang hàng để các agent gọi các agent khác qua các hệ thống. Google 2025. |
| Agent Card | "Danh thiếp của agent" | JSON tại `/.well-known/agent.json` mô tả kỹ năng, endpoint, xác thực. |
| Task | "Đơn vị công việc" | Đối tượng có trạng thái bất đồng bộ với vòng đời; artifact được tạo khi hoàn thành. |
| Artifact | "Kết quả" | Đầu ra được định kiểu: văn bản, JSON cấu trúc, hình ảnh, video, âm thanh. Phương tiện hạng nhất. |
| Opaque lifecycle | "Cách giải quyết là việc của agent" | Client thấy các chuyển đổi trạng thái; server tự do chọn framework/công cụ. |
| Discovery | "Tìm kiếm agent" | `GET /.well-known/agent.json` trả về thẻ (card). |
| MCP vs A2A | "Công cụ vs ngang hàng" | MCP: chiều dọc agent ↔ công cụ. A2A: chiều ngang agent ↔ agent. |
| ACP / ANP / NLIP | "Các giao thức anh em" | Các đặc tả lân cận; A2A là giao thức được áp dụng nhiều nhất năm 2026. |

## Đọc thêm

- [Đặc tả A2A](https://a2a-protocol.org/latest/specification/) — đặc tả chính thức
- [Blog Google Developers — Thông báo A2A](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — bài đăng ra mắt tháng 4 năm 2025
- [GitHub repo của A2A](https://github.com/a2aproject/A2A) — các triển khai tham chiếu và SDK
- [Liu và cộng sự — Khảo sát về các giao thức tương tác Agent](https://arxiv.org/html/2505.02279v1) — so sánh MCP, ACP, A2A, ANP