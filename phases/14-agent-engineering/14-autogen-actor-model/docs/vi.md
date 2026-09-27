# Actor Model cho Agents — Tin nhắn bất đồng bộ và Runtime định kiểu

> Agents dưới dạng actors: trao đổi tin nhắn bất đồng bộ, trình xử lý hướng sự kiện, cô lập lỗi, tính đồng thời tự nhiên. AutoGen v0.4 (Microsoft Research, tháng 1 năm 2025) đã thiết kế lại việc điều phối agent dựa trên mô hình này; framework hiện đang ở chế độ bảo trì, với Microsoft Agent Framework (bản xem trước công khai tháng 10 năm 2025) là phiên bản kế nhiệm cho môi trường production.

**Type:** Học + Xây dựng
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 phút

## Mục tiêu học tập

- Mô tả mô hình actor: agents là các actors, tin nhắn là phương thức IPC duy nhất, cô lập lỗi trên mỗi actor.
- Kể tên ba lớp API của AutoGen v0.4 — Core, AgentChat, Extensions — và mục đích của từng lớp.
- Giải thích lý do tại sao việc tách biệt việc gửi tin nhắn khỏi việc xử lý lại mang lại khả năng cô lập lỗi và tính đồng thời tự nhiên.
- Triển khai một runtime actor bằng stdlib trong Python và chuyển đổi luồng code-review giữa hai agent sang mô hình này.

## Vấn đề

Hầu hết các framework agent đều là đồng bộ: một agent tạo ra, một agent tiêu thụ, trong một call stack. Các lỗi sẽ làm sập toàn bộ stack. Tính đồng thời được thêm vào một cách gượng ép. Việc phân tán đòi hỏi phải viết lại code.

Câu trả lời của AutoGen v0.4: mô hình actor. Mỗi agent là một actor với một hộp thư đến (inbox) riêng. Tin nhắn là tương tác duy nhất. Runtime tách biệt việc gửi khỏi việc xử lý. Lỗi được cô lập trong một actor. Tính đồng thời là mặc định. Phân tán chỉ đơn giản là thay đổi phương thức truyền tải (transport).

## Khái niệm

### Actors

Một actor có:

- Trạng thái riêng (không bao giờ bị can thiệp trực tiếp từ bên ngoài).
- Một hộp thư đến (hàng đợi tin nhắn).
- Một trình xử lý (handler): `receive(message) -> effects` nơi các hiệu ứng có thể là "trả lời", "gửi cho actor khác", "tạo actor mới", "cập nhật trạng thái", "dừng chính mình".

Hai actors không thể chia sẻ bộ nhớ. Chúng chỉ có thể gửi tin nhắn cho nhau.

### Ba lớp API

AutoGen v0.4 chia bề mặt API thành ba phần:

1. **Core.** Framework actor cấp thấp. `AgentRuntime`, `Agent`, `Message`, `Topic`. Trao đổi tin nhắn bất đồng bộ, hướng sự kiện.
2. **AgentChat.** API cấp cao dựa trên tác vụ (thay thế cho ConversableAgent của v0.2). `AssistantAgent`, `UserProxyAgent`, `RoundRobinGroupChat`, `SelectorGroupChat`.
3. **Extensions.** Các tích hợp — OpenAI, Anthropic, Azure, công cụ, bộ nhớ.

### Tại sao việc tách biệt lại quan trọng

Trong mô hình v0.2, việc gọi `agent_a.chat(agent_b)` một cách đồng bộ sẽ chặn agent_a cho đến khi agent_b phản hồi. Trong v0.4, `send(agent_b, msg)` đặt tin nhắn vào inbox của agent_b và trả về ngay lập tức. Runtime sẽ gửi tin nhắn sau đó. Ba hệ quả:

- **Cô lập lỗi.** Agent B bị crash không làm sập Agent A — runtime bắt lỗi trong trình xử lý của B và quyết định cách xử lý (ghi log, thử lại, dead-letter).
- **Tính đồng thời tự nhiên.** Nhiều tin nhắn được gửi cùng lúc; các actors xử lý inbox của chúng một cách đồng thời.
- **Sẵn sàng cho phân tán.** Inbox + transport là cùng một trừu tượng bất kể actor nằm trong cùng tiến trình hay trên một host khác.

### Các cấu trúc liên kết (Topologies)

- **RoundRobinGroupChat.** Các agents thay phiên nhau theo vòng lặp cố định.
- **SelectorGroupChat.** Một agent chọn lọc (selector) quyết định ai sẽ thực hiện tiếp theo dựa trên ngữ cảnh hội thoại.
- **Magentic-One.** Nhóm đa agent tham chiếu cho việc duyệt web, thực thi code, xử lý tệp. Được xây dựng trên AgentChat.

### Khả năng quan sát (Observability)

Hỗ trợ OpenTelemetry được tích hợp sẵn. Mỗi tin nhắn phát ra một span; các lệnh gọi công cụ mang theo các thuộc tính `gen_ai.*` theo quy ước ngữ nghĩa GenAI OTel 2026 (Bài 23).

### Trạng thái: chế độ bảo trì

Đầu năm 2026: AutoGen v0.7.x ổn định cho nghiên cứu và tạo mẫu. Microsoft đã chuyển trọng tâm phát triển sang Microsoft Agent Framework, phiên bản kế nhiệm cho production (bản xem trước công khai ngày 1 tháng 10 năm 2025; bản 1.0 GA dự kiến vào cuối quý 1 năm 2026). Các mô hình AutoGen được chuyển đổi dễ dàng — mô hình actor là ý tưởng bền vững.

```figure
actor-mailbox
```

## Xây dựng

`code/main.py` triển khai một runtime actor bằng stdlib:

- `Message` — payload định kiểu với `sender`, `recipient`, `topic`, `body`.
- `Actor` — trừu tượng với `receive(message, runtime)`.
- `Runtime` — vòng lặp sự kiện với hàng đợi chia sẻ, phân phối, cô lập lỗi.
- Bản demo hai actor: `ReviewerAgent` đánh giá code, `ChecklistAgent` chạy danh sách kiểm tra; chúng trao đổi tin nhắn cho đến khi đạt được sự đồng thuận.

Chạy nó:

```
python3 code/main.py
```

Dấu vết (trace) hiển thị việc gửi tin nhắn, một lỗi mô phỏng trong một actor không làm sập actor kia, và sự hội tụ về một phán quyết chung.

## Sử dụng

- **AutoGen v0.4/v0.7** (bảo trì) — ổn định cho nghiên cứu, tạo mẫu, các mô hình đa agent.
- **Microsoft Agent Framework** — phiên bản kế nhiệm cho production (bản xem trước công khai tháng 10 năm 2025); cùng ý tưởng mô hình actor trong một API mới mẻ.
- **LangGraph swarm topology** (Bài 13) — mô hình tương tự thông qua việc chuyển giao công cụ chia sẻ.
- **Custom actor runtime** — khi bạn cần phương thức truyền tải cụ thể (NATS, RabbitMQ, gRPC).

## Triển khai

`outputs/skill-actor-runtime.md` tạo ra một runtime actor tối giản cộng với một mẫu nhóm (RoundRobin hoặc Selector) cho một tác vụ đa agent cụ thể.

## Bài tập

1. Thêm hàng đợi dead-letter: khi một trình xử lý gặp lỗi, hãy lưu tin nhắn lỗi đó để con người kiểm tra. Hàng đợi DLQ bị kích hoạt bao nhiêu lần trong ví dụ của bạn?
2. Triển khai `SelectorGroupChat`: một actor chọn lọc sẽ chọn ai xử lý tin nhắn tiếp theo dựa trên trạng thái hội thoại.
3. Thêm phương thức truyền tải phân tán: thay thế hàng đợi trong tiến trình bằng máy chủ JSON-over-HTTP để các actors có thể chạy trong các tiến trình riêng biệt.
4. Kết nối một OTel span cho mỗi tin nhắn (hoặc một stand-in không hoạt động). Phát ra `gen_ai.agent.name`, `gen_ai.operation.name` theo Bài 23.
5. Đọc bài viết về kiến trúc của AutoGen v0.4. Chuyển đổi ví dụ của bạn sang API `autogen_core` thực tế. Bạn đã bỏ qua điều gì quan trọng trong môi trường production?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Actor | "Agent" | Trạng thái riêng + inbox + trình xử lý; không chia sẻ bộ nhớ |
| Message | "Event" | Payload định kiểu; cách duy nhất để các actors tương tác |
| Inbox | "Mailbox" | Hàng đợi tin nhắn chờ xử lý của mỗi actor |
| Runtime | "Agent host" | Vòng lặp sự kiện định tuyến tin nhắn và cô lập lỗi |
| Topic | "Channel" | Tuyến đường publish-subscribe được đặt tên giữa các actors |
| Fault isolation | "Let it crash" | Một actor bị lỗi không làm sập các actor khác |
| RoundRobinGroupChat | "Fixed-rotation team" | Các agents thay phiên nhau theo thứ tự |
| SelectorGroupChat | "Context-routed team" | Selector chọn ai thực hiện tiếp theo |
| Magentic-One | "Reference team" | Nhóm đa agent cho web + code + tệp |

## Đọc thêm

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — bài viết về thiết kế lại
- [Tổng quan về LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — giải pháp thay thế dạng đồ thị
- [Quy ước ngữ nghĩa GenAI của OpenTelemetry](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — các spans mà AutoGen phát ra theo mặc định