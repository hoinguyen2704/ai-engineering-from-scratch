# Các mô hình điều phối (Orchestration Patterns): Supervisor, Swarm, Hierarchical

> Bốn mô hình điều phối xuất hiện lặp đi lặp lại trong các framework năm 2026: supervisor-worker, swarm / peer-to-peer, hierarchical, và debate. Lời khuyên từ Anthropic: "Vấn đề là xây dựng hệ thống phù hợp với nhu cầu của bạn." Hãy bắt đầu đơn giản; chỉ thêm cấu trúc liên kết (topology) khi một agent đơn lẻ cộng với năm mô hình workflow là không đủ.

**Type:** Học + Xây dựng
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 25 (Multi-Agent Debate)
**Time:** ~60 phút

## Mục tiêu học tập

- Gọi tên bốn mô hình điều phối phổ biến và biết khi nào nên áp dụng từng mô hình.
- Mô tả khuyến nghị của LangChain năm 2026: supervision dựa trên tool-call so với các thư viện supervisor.
- Giải thích quy tắc "xây dựng hệ thống phù hợp" của Anthropic và cách nó quyết định việc lựa chọn topology.
- Triển khai cả bốn mô hình bằng stdlib với một LLM được lập trình sẵn.

## Vấn đề

Các đội ngũ thường tìm đến "multi-agent" trước khi thực sự cần đến nó. Bốn mô hình xuất hiện lặp lại trong các framework; một khi bạn có thể gọi tên chúng, bạn có thể chọn đúng mô hình — hoặc bỏ qua hoàn toàn việc thiết lập topology.

## Khái niệm

### Supervisor-worker

- Một LLM điều phối trung tâm (central routing LLM) phân phối công việc cho các agent chuyên gia.
- Quyết định: quay lại chính nó, chuyển giao cho chuyên gia, hoặc kết thúc.
- Các chuyên gia không nói chuyện với nhau; mọi luồng điều hướng đều đi qua supervisor.

Framework: LangGraph `create_supervisor`, Anthropic orchestrator-workers, CrewAI Hierarchical Process.

**Khuyến nghị của LangChain năm 2026:** thực hiện supervision thông qua các tool call trực tiếp thay vì `create_supervisor`. Điều này mang lại khả năng kiểm soát context engineering tinh vi hơn — bạn quyết định chính xác những gì mỗi chuyên gia nhìn thấy.

### Swarm / peer-to-peer

- Các agent chuyển giao công việc trực tiếp thông qua bề mặt công cụ (tool surface) được chia sẻ.
- Không có bộ điều phối trung tâm.
- Độ trễ thấp hơn supervisor (ít bước trung gian hơn).
- Khó suy luận hơn (không có điểm kiểm soát duy nhất).

Framework: LangGraph swarm topology, OpenAI Agents SDK handoffs (khi tất cả các agent có thể chuyển giao cho tất cả các agent khác).

### Hierarchical

- Các supervisor quản lý các sub-supervisor, những người này lại quản lý các worker.
- Được triển khai dưới dạng các subgraph lồng nhau trong LangGraph; các crew lồng nhau trong CrewAI.
- Có khả năng mở rộng cho số lượng agent lớn với cái giá phải trả là độ phức tạp vận hành.

Khi nào bạn cần: khi ngân sách context của một supervisor đơn lẻ không thể chứa mô tả của tất cả các chuyên gia.

### Debate

- Các bên đề xuất song song + phản biện chéo lặp đi lặp lại (Bài 25).
- Không hẳn là điều phối — thiên về xác minh hơn — nhưng xuất hiện như một lựa chọn topology trong các framework.

### Autonomous crews vs deterministic flows

CrewAI chính thức hóa hai chế độ triển khai:

- **Flow** cho tự động hóa dựa trên sự kiện mang tính xác định (điểm bắt đầu được khuyến nghị cho production).
- **Crew** cho sự cộng tác dựa trên vai trò tự chủ.

Điều này độc lập với bốn mô hình trên nhưng ánh xạ tới topology: Flow thường là supervisor hoặc hierarchical; Crew thường là supervisor với một LLM router.

### Lời khuyên từ Anthropic

"Thành công trong lĩnh vực LLM không phải là xây dựng hệ thống tinh vi nhất. Đó là xây dựng hệ thống phù hợp với nhu cầu của bạn."

Thứ tự quyết định:

1. Single agent + workflow patterns (Bài 12) — hãy bắt đầu từ đây.
2. Supervisor-worker — khi bạn có 2-4 chuyên gia.
3. Swarm — khi độ trễ quan trọng hơn sự rõ ràng trong suy luận.
4. Hierarchical — chỉ khi ngân sách context của supervisor bị quá tải.
5. Debate — khi độ chính xác quan trọng hơn chi phí.

### Nơi mô hình này đi chệch hướng

- **Tư duy ưu tiên topology.** "Chúng ta cần multi-agent" trước khi xác định vấn đề mà multi-agent giải quyết là gì.
- **Chuyển giao qua lại trong swarm.** A -> B -> A -> B. Hãy sử dụng bộ đếm bước nhảy (hop counters).
- **Hệ thống phân cấp giả.** Ba tầng chỉ vì "doanh nghiệp"; thực tế chỉ có hai đội. Hãy gộp lại.

```figure
orchestration-pattern
```

## Xây dựng

`code/main.py` triển khai cả bốn mô hình trong stdlib với một LLM được lập trình sẵn:

- `Supervisor` — bộ điều phối trung tâm.
- `Swarm` — peer-to-peer với các chuyển giao trực tiếp.
- `Hierarchical` — các supervisor của supervisor.
- `Debate` — các bên đề xuất song song + phản biện.

Mỗi mô hình xử lý cùng một tác vụ ba mục đích (hoàn tiền / lỗi / bán hàng). Hình dạng của trace sẽ khác nhau.

Chạy nó:

```
python3 code/main.py
```

Đầu ra: trace theo từng mô hình + số lượng thao tác. Supervisor là sạch nhất; swarm là ngắn nhất; hierarchical là sâu nhất; debate là tốn kém nhất.

## Sử dụng

- **LangGraph** cho supervisor và hierarchical (subgraph lồng nhau).
- **OpenAI Agents SDK** cho handoffs-as-tools (dạng supervisor).
- **CrewAI Flow** cho production mang tính xác định.
- **Custom** cho debate hoặc khi bạn muốn kiểm soát chính xác.

## Triển khai

`outputs/skill-orchestration-picker.md` chọn một topology và triển khai nó.

## Bài tập

1. Chuyển đổi một supervisor-worker thành swarm bằng cách loại bỏ router. Điều gì bị hỏng? Điều gì cải thiện?
2. Thêm bộ đếm bước nhảy vào swarm: từ chối sau 3 lần chuyển giao. Nó có bắt được việc lặp lại A->B->A không?
3. Xây dựng hệ thống phân cấp hai tầng cho một domain có 12 chuyên gia. Ngân sách context sẽ thất bại ở đâu nếu không có lồng nhau?
4. Profile bốn mô hình trên một workload có hình dạng production. Mô hình nào thắng trên chỉ số nào (độ trễ, chi phí, độ chính xác, khả năng debug)?
5. Đọc bài viết "Building Effective Agents" của Anthropic. Ánh xạ từng luồng production của bạn vào một trong bốn mô hình. Có cái nào không ánh xạ rõ ràng không?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Supervisor-worker | "Router + chuyên gia" | LLM trung tâm điều phối cho chuyên gia; họ không nói chuyện với nhau |
| Swarm | "Peer-to-peer" | Chuyển giao trực tiếp qua các công cụ chia sẻ; không có router trung tâm |
| Hierarchical | "Supervisor của supervisor" | Subgraph lồng nhau cho số lượng lớn agent |
| Debate | "Đề xuất + phản biện" | Các bên đề xuất song song, phản biện chéo (Bài 25) |
| Tool-call-based supervision | "Supervisor không thư viện" | Triển khai supervisor dưới dạng tool call trực tiếp để kiểm soát context |
| Crew | "Đội tự chủ" | Chế độ cộng tác dựa trên vai trò của CrewAI |
| Flow | "Workflow xác định" | Chế độ production dựa trên sự kiện của CrewAI |

## Đọc thêm

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — năm mô hình + agent vs workflow
- [Tổng quan về LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor, swarm, hierarchical
- [Tài liệu CrewAI](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — mô hình debate