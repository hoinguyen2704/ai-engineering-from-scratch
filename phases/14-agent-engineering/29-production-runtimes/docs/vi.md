# Production Runtimes: Queue, Event, Cron

> Các agent trong môi trường production vận hành trên sáu hình thái runtime: request-response, streaming, durable execution, queue-based background, event-driven và scheduled. Hãy chọn hình thái trước khi chọn framework. Khả năng quan sát (observability) là yếu tố then chốt ở mọi hình thái.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Voice)
**Time:** ~60 minutes

## Mục tiêu học tập

- Liệt kê sáu hình thái runtime trong production và khớp mỗi hình thái với một framework / mô hình sản phẩm.
- Giải thích tại sao durable execution (LangGraph) lại quan trọng đối với các tác vụ dài hạn (long-horizon).
- Mô tả runtime event-driven và thời điểm phù hợp để sử dụng Claude Managed Agents.
- Giải thích tại sao khả năng quan sát lại là yếu tố then chốt đối với các agent đa bước.

## Vấn đề

Các agent trong môi trường production thường gặp lỗi mà Jupyter notebook không thể hiển thị: timeout mạng ở bước 37, người dùng ngắt kết nối giữa chừng khi đang gọi thoại, cron job bị dừng khi khởi động lại máy, hoặc background worker bị tràn bộ nhớ. Hình thái runtime sẽ quyết định những lỗi nào có thể khắc phục được.

## Khái niệm

### Request-response

- HTTP đồng bộ. Người dùng chờ đợi cho đến khi hoàn tất.
- Chỉ khả thi cho các tác vụ ngắn (<30s).
- Stacks: Agno (Python + FastAPI), Mastra (TypeScript + Express/Hono/Fastify/Koa).
- Observability: HTTP access logs tiêu chuẩn + OTel spans.

### Streaming

- SSE hoặc WebSocket cho đầu ra tiến triển (progressive output).
- LiveKit mở rộng điều này sang WebRTC cho voice/video (Bài 22).
- Stacks: bất kỳ framework nào hỗ trợ streaming + frontend xử lý SSE/WS.
- Observability: timing theo từng chunk, độ trễ token đầu tiên, độ trễ đuôi (tail latency).

### Durable execution

- Trạng thái được checkpoint sau mỗi bước; tự động tiếp tục khi gặp lỗi.
- Mô hình actor của AutoGen v0.4 cô lập lỗi cho từng agent (Bài 14).
- Điểm khác biệt cốt lõi của LangGraph (Bài 13).
- Cần thiết khi số lượng bước không xác định và chi phí khôi phục cao.

### Queue-based / background

- Job đi vào hàng đợi, worker lấy job, kết quả trả về qua webhook hoặc pub/sub.
- Cần thiết cho các agent dài hạn (hàng chục đến hàng trăm bước mỗi tác vụ, theo thông báo về computer use của Anthropic).
- Stacks: Celery (Python), BullMQ (Node), SQS + Lambda (AWS), tùy chỉnh.
- Observability: độ sâu hàng đợi, phân phối độ trễ theo từng job, kích thước DLQ.

### Event-driven

- Các agent đăng ký nhận trigger: email mới, PR được mở, cron chạy.
- Claude Managed Agents hỗ trợ sẵn điều này (Bài 17).
- CrewAI Flows (Bài 15) cấu trúc các quy trình làm việc tất định (deterministic) theo hướng sự kiện.
- Observability: nguồn trigger, độ trễ từ sự kiện đến khi bắt đầu, độ trễ của agent.

### Scheduled

- Các agent dạng cron chạy định kỳ.
- Kết hợp với durable execution để một lần chạy ban đêm bị lỗi có thể tiếp tục ở lần tick tiếp theo.
- Stacks: Kubernetes CronJob + một framework bền vững; các dịch vụ host (Render cron, Vercel cron).

### Các mô hình triển khai năm 2026

- **CrewAI Flows** cho production hướng sự kiện.
- **Agno** stateless FastAPI cho các microservice Python.
- **Mastra** server adapters (Express, Hono, Fastify, Koa) để nhúng.
- **Pipecat Cloud / LiveKit Cloud** cho voice được quản lý (Bài 22).
- **Claude Managed Agents** cho các tác vụ async dài hạn được host sẵn.

### Khả năng quan sát là yếu tố then chốt

Nếu không có OpenTelemetry GenAI spans (Bài 23) cộng với backend Langfuse/Phoenix/Opik (Bài 24), bạn không thể debug một agent đa bước bị lỗi ở bước 40. Đây không phải là tùy chọn trong production. Đó là sự khác biệt giữa "chúng ta debug nhanh chóng" và "chúng ta phải chạy lại từ đầu với nhiều log hơn".

### Nơi các runtime production thất bại

- **Chọn sai hình thái.** Chọn request-response cho một tác vụ kéo dài 5 phút. Người dùng ngắt kết nối; worker bị quá tải; các lần thử lại (retry) chồng chất.
- **Không có DLQ.** Queue worker không có hàng đợi thư chết (dead-letter queue). Các job bị lỗi sẽ biến mất hoàn toàn.
- **Công việc nền không minh bạch.** Agent chạy nền mà không xuất trace. Lỗi trở nên vô hình cho đến khi người dùng báo cáo.
- **Bỏ qua trạng thái bền vững (durable state).** Bất kỳ lần chạy nào > 30 giây mà bạn không thể chấp nhận việc khởi động lại đều cần durable execution.

```figure
wb-runtime-shapes
```

## Xây dựng

`code/main.py` là một bản demo đa hình thái sử dụng stdlib:

- Endpoint request-response (hàm thuần).
- Handler streaming (generator).
- Worker dựa trên hàng đợi với DLQ.
- Registry kích hoạt sự kiện.
- Scheduler dạng cron.

Chạy nó:

```bash
python3 code/main.py
```

Đầu ra: năm trace hiển thị hành vi của từng hình thái trên cùng một tác vụ. Cùng một logic agent, nhưng các lớp vỏ bên ngoài khác nhau. Durable execution (hình thái thứ sáu) được đề cập cụ thể trong Bài 13 với checkpointing của LangGraph.

## Sử dụng

- **Request-response** cho UX kiểu chat.
- **Streaming** cho phản hồi tiến triển.
- **Durable** cho các tác vụ dài hạn.
- **Queue** cho batch / async / chạy dài.
- **Event** cho tính phản ứng của agent.
- **Cron** cho các công việc bảo trì (tối ưu bộ nhớ, đánh giá, báo cáo chi phí).

## Triển khai

`outputs/skill-runtime-shape.md` chọn một hình thái runtime cho tác vụ và thiết lập các yêu cầu về khả năng quan sát.

## Bài tập

1. Chuyển đổi vòng lặp ReAct từ Bài 01 của bạn sang cả sáu hình thái trong stack của bạn. Hình thái nào phù hợp với bề mặt sản phẩm nào?
2. Thêm DLQ vào bản demo dựa trên hàng đợi. Mô phỏng 10% job bị lỗi; hiển thị kích thước DLQ.
3. Viết một agent đánh giá (eval) kích hoạt bằng cron chạy hàng đêm dựa trên 20 trace hàng đầu trong ngày của bạn.
4. Triển khai streaming với backpressure: nếu client chậm, hãy tạm dừng agent. Điều này tương tác thế nào với ngân sách lượt (turn budget)?
5. Đọc tài liệu về Claude Managed Agents. Khi nào bạn nên chuyển một agent dài hạn tự host sang dạng managed?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Request-response | "Đồng bộ" | Người dùng chờ đợi; chỉ dành cho tác vụ ngắn |
| Streaming | "SSE / WS" | Đầu ra tiến triển; UX tốt hơn; độ trễ có thể quan sát theo từng chunk |
| Durable execution | "Tiếp tục từ lỗi" | Trạng thái được checkpoint; khởi động lại từ bước cuối |
| Queue-based | "Job nền" | Producer / pool worker / DLQ |
| Event-driven | "Dựa trên trigger" | Agent phản ứng với các sự kiện bên ngoài |
| DLQ | "Hàng đợi thư chết" | Nơi chứa các job bị lỗi |
| Claude Managed Agents | "Harness được host" | Async dài hạn do Anthropic host với caching + compaction |

## Đọc thêm

- [Tổng quan về LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — chi tiết về durable execution
- [Tổng quan về Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) — async dài hạn được host
- [Anthropic, Giới thiệu về computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "hàng chục đến hàng trăm bước mỗi tác vụ"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — cô lập lỗi theo mô hình actor