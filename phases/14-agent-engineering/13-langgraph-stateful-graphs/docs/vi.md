# Stateful Graph Orchestration — Durable Execution and Checkpoints

> Agent là một máy trạng thái (state machine); các node là các hàm; các cạnh (edges) là các chuyển đổi; trạng thái được checkpoint sau mỗi node. Khôi phục từ bất kỳ lỗi nào tại checkpoint thành công gần nhất. LangGraph là tài liệu tham khảo năm 2026 cho mô hình điều phối trạng thái cấp thấp này.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 phút

## Mục tiêu học tập

- Mô tả mô hình cốt lõi của LangGraph: máy trạng thái với trạng thái định kiểu (typed state), các node hàm, các cạnh điều kiện và checkpoint sau mỗi node.
- Nêu tên bốn khả năng mà tài liệu nhấn mạnh: thực thi bền bỉ (durable execution), streaming, human-in-the-loop, bộ nhớ toàn diện.
- Giải thích ba cấu trúc liên kết điều phối mà LangGraph hỗ trợ: supervisor, peer-to-peer (swarm), phân cấp (subgraphs lồng nhau).
- Triển khai một state graph bằng stdlib với trạng thái định kiểu, các cạnh điều kiện và chu kỳ checkpoint/resume.

## Vấn đề

Các agent và workflow có chung một vấn đề: khi một tiến trình 40 bước bị lỗi ở bước 38, bạn muốn khôi phục từ bước 38 chứ không phải bắt đầu lại từ đầu. Các mô hình trạng thái hạng hai khiến người vận hành phải loay hoay với các cơ chế thử lại (retry) xung quanh một thư viện vốn giả định các tiến trình chạy mới hoàn toàn.

Câu trả lời thiết kế của LangGraph: trạng thái là một đối tượng định kiểu hạng nhất, các thay đổi là tường minh và các checkpoint được lưu trữ sau mỗi node. Việc khôi phục chỉ là một lệnh gọi `load_state(session_id)`.

## Khái niệm

### Đồ thị (The graph)

Một đồ thị được định nghĩa bởi:

- **Loại trạng thái (State type).** Một dict định kiểu (hoặc Pydantic model) mà mọi node đều đọc và thay đổi.
- **Các node.** Các hàm thuần túy `(state) -> state_update`. Các cập nhật được hợp nhất vào trạng thái sau khi hàm trả về.
- **Các cạnh (Edges).** Các chuyển đổi trực tiếp hoặc có điều kiện giữa các node.
- **Điểm vào và điểm ra.** Các node sentinel `START` và `END` đánh dấu ranh giới.

Ví dụ: một agent với các node `classify`, `refund`, `bug`, `sales`, `done` — một workflow định tuyến dưới dạng đồ thị.

### Thực thi bền bỉ (Durable execution)

Sau khi mỗi node trả về, runtime sẽ tuần tự hóa (serialize) trạng thái và ghi vào một checkpointer (SQLite, Postgres, Redis, tùy chỉnh). Khi lỗi xảy ra tại bước N, runtime có thể `resume(session_id)` và tiếp tục từ bước N+1 với trạng thái chính xác.

Tài liệu LangGraph nêu rõ các người dùng sản xuất quan trọng: Klarna, Uber, J.P. Morgan. Điểm mấu chốt không phải là hình dạng đồ thị; mà là hình dạng đồ thị cộng với checkpointing giúp việc phục hồi trở nên rẻ và hiệu quả.

### Streaming

Mỗi node có thể trả về kết quả từng phần (yield). Đồ thị truyền các sự kiện delta theo từng node tới người gọi để giao diện người dùng cập nhật khi đồ thị chạy.

### Human-in-the-loop

Kiểm tra và sửa đổi trạng thái giữa các node. Các triển khai: tạm dừng trước một node quan trọng, hiển thị trạng thái cho con người, chấp nhận sửa đổi, sau đó tiếp tục. Checkpointer giúp việc này trở nên dễ dàng vì trạng thái đã được tuần tự hóa.

### Bộ nhớ (Memory)

Ngắn hạn (trong một lần chạy — lịch sử hội thoại trong trạng thái) và dài hạn (giữa các lần chạy — bền bỉ thông qua checkpointer cộng với một kho lưu trữ dài hạn riêng biệt). LangGraph tích hợp với các hệ thống bộ nhớ ngoài (Mem0, tùy chỉnh) thông qua các công cụ (tools).

### Ba cấu trúc liên kết

1. **Supervisor.** LLM định tuyến trung tâm điều phối đến các subagent chuyên biệt. `create_supervisor()` trong `langgraph-supervisor` (mặc dù nhóm LangChain năm 2026 khuyến nghị thực hiện việc này thông qua các tool call trực tiếp để kiểm soát ngữ cảnh tốt hơn).
2. **Swarm / peer-to-peer.** Các agent bàn giao trực tiếp thông qua bề mặt công cụ chia sẻ. Không có bộ định tuyến trung tâm.
3. **Phân cấp (Hierarchical).** Các supervisor quản lý các sub-supervisor, được triển khai dưới dạng các subgraph lồng nhau.

### Nơi mô hình này đi sai hướng

- **Checkpoints quá nhỏ.** Chỉ checkpoint các lượt hội thoại sẽ khiến trạng thái công cụ và các ghi chép bộ nhớ không thể khôi phục. Trạng thái đầy đủ phải được tuần tự hóa.
- **Các node không tất định (Non-deterministic).** Việc khôi phục giả định rằng đầu vào của node tạo ra cùng một cập nhật trạng thái. Các hạt giống ngẫu nhiên (random seeds), thời gian thực (wall-clock), các API bên ngoài phải được ghi lại.
- **Lạm dụng các cạnh điều kiện.** Một đồ thị với mọi cạnh đều có điều kiện là một máy trạng thái không thể suy luận được. Ưu tiên các chuỗi tuyến tính với các nhánh thỉnh thoảng xuất hiện.

```figure
langgraph-state
```

## Xây dựng

`code/main.py` triển khai một stateful graph bằng stdlib:

- `State` — một dict định kiểu với `messages`, `step`, `route`, `output`, `human_approval`.
- `Node` — hàm có thể gọi nhận trạng thái và trả về một dict cập nhật.
- `StateGraph` — các node + cạnh + cạnh điều kiện + chạy + khôi phục.
- `SQLiteCheckpointer` (giả lập trong bộ nhớ) — tuần tự hóa trạng thái sau mỗi node; `load(session_id)` khôi phục.
- Một đồ thị demo: phân loại -> nhánh (hoàn tiền / lỗi / bán hàng) -> cổng con người -> gửi.

Chạy nó:

```
python3 code/main.py
```

Dấu vết (trace) cho thấy lần chạy đầu tiên bị lỗi tại cổng con người, sự bền bỉ được duy trì, sau đó việc khôi phục tạo ra kết quả cuối cùng.

## Sử dụng

- **LangGraph** — tài liệu tham khảo, sẵn sàng cho sản xuất. Sử dụng `create_react_agent`, `create_supervisor`, hoặc tự xây dựng đồ thị của riêng bạn.
- **AutoGen v0.4** (Bài 14) — mô hình actor thay thế cho các kịch bản có tính đồng thời cao.
- **Claude Agent SDK** (Bài 17) — bộ công cụ quản lý với lưu trữ phiên tích hợp.
- **Tùy chỉnh** — khi bạn cần kiểm soát chính xác hình dạng trạng thái hoặc backend checkpointer.

## Triển khai

`outputs/skill-state-graph.md` tạo ra một state graph theo hình dạng LangGraph trong bất kỳ runtime mục tiêu nào với checkpointing và khôi phục được tích hợp sẵn.

## Bài tập

1. Thêm một cạnh điều kiện từ `classify` đến `end` khi độ tin cậy phân loại dưới một ngưỡng. Khôi phục quá trình chạy sau khi con người đặt `route` theo cách thủ công.
2. Thay thế giả lập kiểu SQLite bằng một checkpointer SQLite thực tế. Đo lường chi phí tuần tự hóa theo từng bước.
3. Triển khai các cạnh song song: hai node chạy đồng thời, hợp nhất bởi một reducer tùy chỉnh. Trạng thái bất biến (immutable state) mang lại lợi ích gì ở đây?
4. Đọc tài liệu tham khảo `langgraph-supervisor`. Chuyển đổi bản demo sang `create_supervisor`. So sánh hình dạng dấu vết.
5. Thêm streaming: mỗi node trả về trạng thái từng phần trong khi chạy. In các delta khi chúng đến.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| State graph | "Agent là máy trạng thái" | Trạng thái định kiểu + node + cạnh + reducer |
| Checkpointer | "Backend bền bỉ" | Tuần tự hóa trạng thái sau mỗi node; cho phép khôi phục |
| Reducer | "Hợp nhất trạng thái" | Hàm kết hợp trạng thái hiện tại với cập nhật của node |
| Cạnh điều kiện | "Nhánh" | Cạnh được chọn bởi một hàm của trạng thái |
| Subgraph | "Đồ thị lồng nhau" | Một đồ thị được sử dụng như một node bên trong đồ thị khác |
| Thực thi bền bỉ | "Khôi phục từ lỗi" | Khởi động lại tại node thành công cuối cùng với trạng thái chính xác |
| Supervisor | "LLM định tuyến" | Bộ điều phối trung tâm cho các subagent chuyên biệt |
| Swarm | "Agent P2P" | Các agent bàn giao qua công cụ chia sẻ; không có bộ định tuyến trung tâm |

## Đọc thêm

- [Tổng quan về LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — tài liệu tham khảo
- [Tham khảo langgraph-supervisor](https://reference.langchain.com/python/langgraph/supervisor/) — API mô hình supervisor
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — giải pháp thay thế mô hình actor
- [Tổng quan về Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — lưu trữ phiên và subagent