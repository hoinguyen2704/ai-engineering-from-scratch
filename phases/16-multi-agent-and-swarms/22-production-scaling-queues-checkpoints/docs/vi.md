# Production Scaling — Queues, Checkpoints, Durability

> Việc mở rộng các hệ thống đa tác nhân (multi-agent) lên hàng nghìn lượt chạy đồng thời đòi hỏi **durable execution** (thực thi bền vững) — kết hợp giữa hàng đợi công việc (work queues) và các điểm kiểm tra (checkpoints), để bất kỳ worker nào cũng có thể tiếp tục một lượt chạy sau khi xảy ra sự cố, miễn là có cơ chế xử lý lease, các side effect có tính lũy đẳng (idempotent) và khả năng phát lại (replay) có tính tất định. Runtime của LangGraph là ví dụ tham chiếu: nó ghi lại một checkpoint sau mỗi super-step được khóa bởi `thread_id` (mặc định là Postgres); khi worker gặp sự cố, nó sẽ giải phóng lease và một worker khác sẽ tiếp tục công việc. Các tác nhân có thể "ngủ" vô thời hạn trong khi chờ đợi phản hồi từ con người. **MegaAgent** (arXiv:2408.09955) vận hành một hàng đợi producer-consumer cho mỗi tác nhân với ba trạng thái (Idle / Processing / Response) và cơ chế điều phối hai lớp (chat nội bộ nhóm + chat quản trị liên nhóm). **Fiber/async** vượt trội hơn so với mô hình thread-per-job khi xử lý streaming LLM: các thread thường nhàn rỗi 99% thời gian để chờ token, trong khi các fiber chủ động nhường quyền (yield) khi thực hiện I/O. Quan điểm đối lập: Ashpreet Bedi trong "Scaling Agentic Software" lập luận rằng chỉ cần **FastAPI + Postgres** là đủ cho đến khi tải thực tế chứng minh điều ngược lại — các kiến trúc đơn giản thường hiệu quả hơn mong đợi. Bài học này xây dựng một nhật ký checkpoint bền vững, hàng đợi công việc cho mỗi tác nhân với các chuyển đổi trạng thái, bản demo so sánh async-vs-thread, và đúc kết quy tắc thực dụng "bắt đầu đơn giản".

**Type:** Learn + Build
**Languages:** Python (stdlib, `asyncio`, `sqlite3`)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)
**Time:** ~75 phút

## Problem

Một hệ thống đa tác nhân nguyên mẫu hoạt động trên một máy tính xách tay với ba tác nhân trong vòng lặp sự kiện (event loop) nằm trên bộ nhớ. Khi bạn chuyển sang môi trường production:

- Các tác nhân đôi khi chạy trong nhiều giờ (nghiên cứu dài hạn, chờ đợi phản hồi từ con người).
- Các tiến trình worker bị crash. Việc khởi động lại sẽ làm mất trạng thái.
- Tải đỉnh điểm gấp 10 lần mức trung bình; bạn cần khả năng mở rộng theo chiều ngang (horizontal scaling).
- Người dùng trả phí theo lượt chạy tác nhân; bạn cần ngữ nghĩa "exactly-once" (chính xác một lần) để tính phí.

Vòng lặp sự kiện trong bộ nhớ không giải quyết được các vấn đề này. Bạn cần một lớp thực thi bền vững bên dưới. Các tùy chọn tiêu chuẩn năm 2026 bao gồm:

1. Workflow engine có checkpoint (Temporal, LangGraph runtime).
2. Message queue với kho lưu trữ trạng thái (Postgres + SQS/RabbitMQ).
3. Framework mô hình Actor (producer-consumer cho mỗi tác nhân của MegaAgent).
4. Tự xây dựng FastAPI + Postgres (lập luận của Bedi).

Bài học này sẽ xây dựng phiên bản thu nhỏ của từng loại.

## Concept

### Durable execution, mô hình thực thi

Một engine thực thi bền vững sẽ lưu trữ toàn bộ trạng thái chương trình sau mỗi "bước" (super-step, theo thuật ngữ của LangGraph). Khi xảy ra sự cố:

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

Các yêu cầu để cơ chế này hoạt động:

- **Trạng thái có thể tuần tự hóa (Serializable state).** Tất cả trạng thái của tác nhân phải có khả năng lưu trữ. Các function closure chứa kết nối cơ sở dữ liệu đang hoạt động sẽ không thể tồn tại sau khi crash.
- **Khả năng tiếp tục có tính tất định (Deterministic resume).** Với cùng một trạng thái và cùng đầu vào, tác nhân tạo ra các hành động giống hệt nhau (hoặc ủy quyền cho một oracle tất định bên ngoài để gọi LLM).
- **Side effect có tính lũy đẳng (Idempotent side effects).** Các lệnh gọi bên ngoài (gọi công cụ, thanh toán) phải có tính lũy đẳng hoặc sử dụng khóa chống trùng lặp (deduplication key).

LangGraph ghi checkpoint sau mỗi super-step; Temporal ghi sau mỗi activity; Restate sử dụng nhật ký dựa trên sự kiện (event-sourced journals). Cả ba đều triển khai cùng một mô hình.

### Runtime checkpoint-per-step

Runtime của LangGraph là ví dụ thực tế: mỗi tác nhân có một `thread_id`; trạng thái là một typed dict; mỗi super-step ghi một hàng vào bảng checkpoints. Khi tiếp tục, runtime sẽ phát lại từ checkpoint cuối cùng thay vì bắt đầu từ đầu. Các tác nhân có thể `interrupt()` trong khi chờ đợi phản hồi từ con người; runtime sẽ lưu trữ và giải phóng worker. Khi có phản hồi, bất kỳ worker nào cũng có thể tiếp tục.

Đây là thiết kế production tiêu chuẩn vào tháng 4 năm 2026.

### Hàng đợi cho mỗi tác nhân của MegaAgent

arXiv:2408.09955 mô tả một thử nghiệm quy mô: hàng nghìn tác nhân đồng thời trong một cụm. Kiến trúc:

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

Cơ chế điều phối hai lớp cho phép các cuộc hội thoại nội bộ nhóm diễn ra dày đặc trong khi liên lạc giữa các nhóm vẫn thưa thớt — mô hình được sử dụng để giữ chi phí tuyến tính khi có hàng nghìn tác nhân.

### Async vs thread-per-job

Các lệnh gọi LLM bị giới hạn bởi I/O. Một thread chờ token tiếp theo sẽ nhàn rỗi 99% thời gian. Các thread tiêu tốn khoảng 1MB RAM mỗi thread; với 10.000 lệnh gọi đồng thời, đó là 10GB chỉ dành cho stack.

Các Fiber (Python `asyncio`, Go goroutines, Rust `tokio`) chủ động nhường quyền khi thực hiện I/O. 10.000 lệnh gọi tương tự có thể nằm gọn trong một tiến trình. Ở quy mô tác nhân LLM, async không phải là một tối ưu hóa — nó là kiến trúc.

Ngoại lệ: Các tác vụ xử lý hậu kỳ nặng về CPU (embedding, kỹ thuật tokenizer) vẫn cần thread hoặc tiến trình. Hãy tách biệt lớp I/O khỏi lớp CPU của bạn.

### Quan điểm đối lập của Bedi

"Scaling Agentic Software" (Ashpreet Bedi, 2026) lập luận rằng hầu hết các đội ngũ đều kỹ thuật hóa quá mức trước khi đo lường tải thực tế. Mặc định thực dụng:

- FastAPI + Postgres.
- Mỗi lượt chạy tác nhân là một hàng; trạng thái được cập nhật tại chỗ với optimistic concurrency.
- Các tác vụ nền thông qua `pg_notify` hoặc một worker Celery đơn giản.
- Chính sách thử lại (retry) trong mã ứng dụng.

Đối với tải dưới ~100 lượt chạy tác nhân đồng thời trên các tác vụ có thể quản lý được, đây thường là tất cả những gì bạn cần. Hãy nâng cấp khi bạn đo lường thấy hệ thống bị quá tải.

Quy tắc: chỉ áp dụng các framework thực thi bền vững khi bạn gặp vấn đề cụ thể mà các kiến trúc đơn giản không thể giải quyết. Việc áp dụng sớm chỉ làm lãng phí thời gian vào các thủ tục không mang lại hiệu quả.

### Ngữ nghĩa Exactly-once

Đối với các lượt chạy tác nhân có trả phí, bạn cần "exactly-once effective" (giao hàng ít nhất một lần + consumer lũy đẳng). Các bước kỹ thuật:

- **Khóa Dedup cho mỗi lượt chạy.** Bao gồm nó trong mọi lệnh gọi side-effect.
- **Mô hình Outbox.** Các side effect ghi vào một bảng trước, sau đó một tiến trình riêng biệt sẽ thực thi chúng. Cả hai bước đều phải lũy đẳng.
- **Giao dịch bù trừ (Compensating transactions).** Khi một side effect thành công nhưng việc ghi nhật ký theo dõi thất bại, hãy lên lịch cho một giao dịch bù trừ.

Đây là các mô hình kỹ thuật cơ sở dữ liệu, không phải dành riêng cho LLM. Thuế LLM duy nhất là các lệnh gọi LLM chậm; mọi thứ khác đều là hệ thống phân tán tiêu chuẩn.

### Rainbow deployment

Hệ thống nghiên cứu đa tác nhân của Anthropic sử dụng "rainbow deployments": nhiều phiên bản runtime tác nhân chạy đồng thời để các tác nhân chạy dài hạn không bị ngắt quãng mỗi khi triển khai code mới. Canary các phiên bản mới trên một phần lưu lượng; loại bỏ các phiên bản cũ khi các tác nhân của chúng hoàn tất.

Đây là tiêu chuẩn cho các hệ thống có trạng thái chạy dài hạn; sự thích nghi năm 2026 là các tác nhân có thể sống hàng giờ, vì vậy các chu kỳ triển khai phải đáp ứng được điều đó.

### Danh sách kiểm tra production tiêu chuẩn

- Trạng thái bền vững (checkpoints, snapshots, hoặc outbox + nhật ký có thể phát lại).
- Side effect lũy đẳng.
- Lớp I/O async cho các lệnh gọi LLM.
- Giao hàng ít nhất một lần với dedup.
- Rainbow/canary deployment cho các workload có trạng thái.
- Khả năng quan sát (Observability): dấu vết (trace) cho mỗi tác nhân, kiểm toán super-step, bộ đếm thử lại.

```figure
sw-checkpoint-replay
```

## Build It

`code/main.py` triển khai:

- `CheckpointStore` — Nhật ký checkpoint dựa trên SQLite với các khóa thread-id. Mỗi super-step thêm một hàng.
- `run_with_checkpoint(agent, thread_id)` — mô phỏng một sự cố giữa chừng; worker thứ hai tiếp tục từ checkpoint cuối cùng.
- `AgentQueue` — máy trạng thái Idle / Processing / Response cho mỗi tác nhân với một hàng đợi công việc nhỏ.
- `demo_async_vs_threads()` — chạy 500 "lệnh gọi LLM" mô phỏng đồng thời thông qua asyncio và thông qua các thread; báo cáo thời gian thực và bộ nhớ đỉnh (ước tính).

Chạy:

```
python3 code/main.py
```

Kết quả mong đợi: việc tiếp tục từ checkpoint thành công sau khi mô phỏng crash; phiên bản async xử lý 500 lệnh gọi đồng thời trong < 1s; phiên bản thread mất vài giây và sử dụng bộ nhớ nhiều hơn gấp nhiều lần cho mỗi đơn vị đồng thời.

## Use It

`outputs/skill-scaling-advisor.md` đưa ra lời khuyên về lựa chọn thực thi bền vững: FastAPI + Postgres, LangGraph runtime, Temporal, hoặc tùy chỉnh. Được hiệu chỉnh theo tải, nhu cầu lưu giữ trạng thái và tần suất triển khai.

## Ship It

Các bước củng cố production tiêu chuẩn:

- **Bắt đầu đơn giản (Quy tắc của Bedi).** FastAPI + Postgres cho đến khi bạn đo lường thấy nó thất bại.
- **Đo lường mọi thứ trước khi tối ưu hóa.** Biểu đồ độ trễ mỗi lượt chạy, thời gian mỗi bước, số lần thử lại, phân loại lỗi.
- **Mô hình Outbox cho side effect.** Đặc biệt là thanh toán và các lệnh gọi API bên ngoài.
- **Rainbow deploys.** Không bao giờ ngắt các lượt chạy tác nhân đang thực hiện trong quá trình triển khai.
- **Áp dụng các engine thực thi bền vững (Temporal / LangGraph / Restate) khi** bạn gặp các vấn đề cụ thể: chờ đợi phản hồi từ con người kéo dài hàng giờ, điều phối liên vùng, các chính sách thử lại/bù trừ phức tạp.
- **Async cho lớp I/O.** Chỉ sử dụng thread cho các tác vụ xử lý hậu kỳ nặng về CPU.

## Exercises

1. Chạy `code/main.py`. Xác nhận việc tiếp tục từ checkpoint hoạt động; đo lường sự khác biệt về khả năng đồng thời giữa async và thread.
2. Triển khai bảng **outbox**: mọi lệnh gọi công cụ ghi vào outbox trước, sau đó một goroutine/task riêng biệt thực thi. Xác minh tính lũy đẳng bằng cách chạy lệnh gọi công cụ hai lần.
3. Mô phỏng **rainbow deploy**: hai phiên bản runtime đồng thời; định tuyến một nửa số thread_id mới cho mỗi phiên bản; xác nhận rằng các thread đang chạy trên phiên bản cũ không bị gián đoạn.
4. Đọc tài liệu runtime của LangGraph (được liên kết bên dưới). Xác định các tính năng nào của runtime sẽ mất nhiều thời gian nhất để sao chép trong phiên bản FastAPI + Postgres tự xây dựng. Đó có phải là lý do để áp dụng, hay bạn có thể trì hoãn?
5. Đọc MegaAgent (arXiv:2408.09955) Phần 3. Cơ chế điều phối hai lớp (chat nội bộ nhóm + chat quản trị liên nhóm) được nêu rõ. Hãy phác thảo cách bạn sẽ ánh xạ điều này vào một message queue với hai họ hàng đợi.

## Key Terms

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Durable execution | "Lưu trữ trạng thái chương trình" | Engine ghi trạng thái sau mỗi super-step; phục hồi sau crash có tính tất định. |
| Super-step | "Ranh giới giao dịch" | Đơn vị công việc giữa các checkpoint. Thuật ngữ của LangGraph. |
| thread_id | "Định danh lượt chạy tác nhân" | Khóa liên kết các checkpoint và logic tiếp tục. |
| Idempotency | "An toàn để thử lại" | Lặp lại một side effect tạo ra kết quả tương tự như một lần thử. |
| Outbox pattern | "Tách biệt side effect" | Ghi ý định vào bảng; một executor riêng biệt thực hiện và đánh dấu hoàn thành. |
| At-least-once delivery | "Có thể trùng lặp" | Ngữ nghĩa message queue; khóa dedup giúp consumer đạt được hiệu quả exactly-once. |
| Rainbow deploy | "Các phiên bản chồng lấp" | Nhiều phiên bản runtime đồng thời trong các workload chạy dài hạn. |
| Async fiber | "Nhường quyền chủ động" | Đồng thời ở chế độ người dùng; rẻ hơn so với thread cho các tải bị giới hạn bởi I/O. |
| Checkpoint | "Ảnh chụp trạng thái" | Trạng thái đã tuần tự hóa tại ranh giới super-step; khóa để tiếp tục. |

## Further Reading

- [LangChain — Runtime đằng sau các tác nhân sâu trong production](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — Thiết kế runtime LangGraph
- [MegaAgent](https://arxiv.org/abs/2408.09955) — Hàng đợi producer-consumer cho mỗi tác nhân; điều phối hai lớp tại hàng nghìn tác nhân đồng thời
- [Matrix](https://arxiv.org/abs/2511.21686) — Framework phi tập trung với message queue làm nền tảng điều phối
- [Tài liệu Temporal](https://docs.temporal.io/) — Workflow engine tham chiếu cho thực thi bền vững
- [Anthropic — Hệ thống nghiên cứu đa tác nhân](https://www.anthropic.com/engineering/multi-agent-research-system) — Các bài học production bao gồm rainbow deployment