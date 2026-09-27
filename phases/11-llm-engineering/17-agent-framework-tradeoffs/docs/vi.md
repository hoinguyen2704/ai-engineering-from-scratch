# Các đánh đổi trong Framework Agent — Điều phối theo Đồ thị, Vai trò và Tác nhân

> Mọi framework đều quảng cáo cùng một bản demo (agent nghiên cứu viết báo cáo) và che giấu cùng một lỗi (xung đột giữa schema trạng thái và lớp điều phối). Hãy chọn framework có các abstraction khớp với hình thái vấn đề của bạn; mọi thứ khác chỉ là phần keo dán mà bạn sẽ phải viết lại hai lần.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 16 (LangGraph)
**Time:** ~45 phút

## Vấn đề

Bạn có một tác vụ cần nhiều hơn một lần gọi LLM. Có thể đó là một quy trình nghiên cứu (lập kế hoạch, tìm kiếm, tóm tắt, trích dẫn). Có thể đó là một pipeline đánh giá code (phân tích diff, phê bình, sửa lỗi, xác thực). Hoặc có thể là một trợ lý đa lượt giúp đặt chuyến bay, viết email và nộp báo cáo chi phí. Bạn chọn một framework.

Ba ngày sau, bạn phát hiện ra các abstraction của framework bị rò rỉ. CrewAI cung cấp cho bạn các vai trò nhưng gây khó khăn khi "nhà nghiên cứu" cần chuyển một kế hoạch có cấu trúc cho "người viết". AutoGen cung cấp cuộc trò chuyện giữa các agent nhưng không có trạng thái hạng nhất (first-class state), vì vậy checkpoint của bạn chỉ là một file pickle của nhật ký hội thoại. LangGraph cung cấp đồ thị trạng thái nhưng buộc bạn phải đặt tên cho mọi chuyển đổi trước khi biết agent sẽ làm gì. Agno cung cấp abstraction đơn agent nhưng sẽ "gào thét" khi bạn cố gắng phân nhánh ra ba worker chạy đồng thời.

Giải pháp không phải là "chọn framework tốt nhất". Đó là khớp abstraction cốt lõi của framework với hình thái vấn đề của bạn. Bài học này sẽ vẽ ra bản đồ đó.

## Khái niệm

![Agent framework matrix: core abstraction vs problem shape](../assets/framework-matrix.svg)

Bốn framework thống trị bối cảnh năm 2026. Các abstraction cốt lõi của chúng không giống nhau.

| Framework | Abstraction cốt lõi | Phù hợp nhất | Ít phù hợp nhất |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — trạng thái định kiểu, các node, các cạnh điều kiện, checkpointer. | Các quy trình làm việc có trạng thái rõ ràng và ngắt quãng có sự tham gia của con người; các agent sản xuất cần gỡ lỗi theo kiểu du hành thời gian. | Brainstorming tự do, theo vai trò mà cấu trúc liên kết chưa được xác định. |
| **CrewAI** | `Crew` — các vai trò (mục tiêu, bối cảnh), các tác vụ, quy trình (tuần tự hoặc phân cấp). | Các quy trình làm việc nhập vai hoặc theo persona với kế hoạch tuyến tính/phân cấp ngắn. | Bất cứ thứ gì có trạng thái phức tạp ngoài lịch sử lượt của crew; phân nhánh phức tạp. |
| **AutoGen** | Cặp `ConversableAgent` — hai hoặc nhiều agent trò chuyện theo lượt cho đến khi đạt điều kiện thoát. | *Đối thoại* đa agent (giáo viên-học sinh, người đề xuất-người phê bình, tác nhân-người đánh giá) nơi tư duy nảy sinh từ cuộc trò chuyện. | Các quy trình làm việc tất định với DAG đã biết; bất cứ thứ gì cần trạng thái bền vững qua các lần khởi động lại. |
| **Agno** | `Agent` — một LLM đơn lẻ + công cụ + bộ nhớ, có thể kết hợp thành các đội. | Xây dựng nhanh các agent đơn lẻ và các đội nhẹ; hỗ trợ đa phương thức mạnh mẽ và các driver lưu trữ tích hợp sẵn. | Các đồ thị phân nhánh sâu, rõ ràng với các bộ giảm (reducer) tùy chỉnh. |

### "Abstraction" thực sự nghĩa là gì

Abstraction cốt lõi của một framework là thứ bạn vẽ lên bảng trắng khi trình bày kiến trúc.

- **LangGraph** → bạn vẽ một đồ thị. Các node là các bước, các cạnh là các chuyển đổi, và đối tượng trạng thái tại mỗi điểm đều được định kiểu. Mô hình tư duy là một máy trạng thái (state machine).
- **CrewAI** → bạn vẽ một sơ đồ tổ chức. Mỗi vai trò có một mô tả công việc và một người quản lý điều phối các tác vụ. Mô hình tư duy là một nhóm nhỏ các chuyên gia.
- **AutoGen** → bạn vẽ một tin nhắn Slack. Hai agent nhắn tin cho nhau; một agent thứ ba tham gia nếu bạn cần người điều phối. Mô hình tư duy là trò chuyện.
- **Agno** → bạn vẽ một hộp đơn lẻ với các công cụ gắn kèm. Đặt các hộp cạnh nhau để tạo thành một đội. Mô hình tư duy là "agent với đầy đủ tiện ích đi kèm".

### Câu hỏi về trạng thái

Trạng thái là nơi hầu hết các lựa chọn framework thất bại trong môi trường sản xuất.

- **LangGraph.** Trạng thái định kiểu (`TypedDict` hoặc mô hình Pydantic), các bộ giảm (reducer) theo từng trường, checkpointer hạng nhất (SQLite/Postgres/Redis). Việc khôi phục, ngắt quãng và du hành thời gian là miễn phí. *(Xem Phase 11 · 16.)*
- **CrewAI.** Trạng thái luân chuyển dưới dạng chuỗi giữa các tác vụ thông qua trường `context`, hoặc có cấu trúc thông qua `output_pydantic`. Không có kho lưu trữ bền vững cho mỗi crew ngay từ đầu; bạn phải tự thêm vào nếu crew cần tồn tại sau khi khởi động lại.
- **AutoGen.** Trạng thái là lịch sử trò chuyện và bất kỳ `context` nào do người dùng định nghĩa. Bản ghi hội thoại được lưu giữ; trạng thái quy trình làm việc tùy ý thì không, trừ khi bạn viết các adapter.
- **Agno.** Các driver lưu trữ tích hợp sẵn (SQLite, Postgres, Mongo, Redis, DynamoDB) gắn vào một `Agent` thông qua `storage=` — các phiên hội thoại và bộ nhớ người dùng được lưu giữ tự động. Không phải là một checkpointer đồ thị đầy đủ; mà là một kho lưu trữ phiên.

### Câu hỏi về phân nhánh

Mọi agent không tầm thường đều phân nhánh. Ai quyết định nhánh đó mới là vấn đề.

- **LangGraph** — bạn quyết định, thông qua các cạnh điều kiện. Định tuyến là một hàm Python với các nhánh được đặt tên. Các nhánh là hạng nhất trong đồ thị đã biên dịch; checkpointer ghi lại nhánh nào đã được chọn.
- **CrewAI** — người quản lý quyết định ở chế độ phân cấp; ở chế độ tuần tự, bạn quyết định tại thời điểm xây dựng. Định tuyến ẩn trong danh sách tác vụ; không có "if" hạng nhất bên ngoài prompt của người quản lý.
- **AutoGen** — các agent quyết định thông qua trò chuyện. Phân nhánh nảy sinh từ việc ai sẽ nói tiếp theo. `GroupChatManager` chọn người nói tiếp theo; bạn có thể tự viết một `speaker_selection_method` nhưng mặc định là do LLM điều khiển.
- **Agno** — agent quyết định bằng cách chọn công cụ nào để gọi tiếp theo. Các đội có chế độ điều phối/định tuyến/cộng tác; việc phân nhánh ngoài phạm vi đó là trách nhiệm của nhà phát triển.

### Câu hỏi về khả năng quan sát (observability)

- **LangGraph** — OpenTelemetry thông qua LangSmith hoặc bất kỳ bộ xuất OTel nào. Mỗi chuyển đổi node là một trace span; các checkpoint đóng vai trò như các trace có thể phát lại. LangSmith là tùy chọn hạng nhất; Langfuse/Phoenix cũng có các adapter.
- **CrewAI** — OpenTelemetry hạng nhất từ cuối năm 2025; tích hợp với Langfuse, Phoenix, Opik, AgentOps.
- **AutoGen** — Tích hợp OpenTelemetry thông qua `autogen-core`; AgentOps và Opik có các connector. Độ chi tiết của trace là theo từng tin nhắn của agent, không phải theo từng node.
- **Agno** — cờ `monitoring=True` tích hợp sẵn cộng với các bộ xuất OpenTelemetry; tích hợp chặt chẽ với Langfuse cho các trace phiên.

### Chi phí và độ trễ

Cả bốn framework đều thêm chi phí trên mỗi lần gọi (logic framework, xác thực, tuần tự hóa). Thứ tự tăng dần của chi phí: Agno ≈ LangGraph < CrewAI ≈ AutoGen. Sự khác biệt bị chi phối bởi việc framework thực hiện bao nhiêu định tuyến LLM bổ sung. Người quản lý phân cấp của CrewAI tiêu tốn token để quyết định ai sẽ thực hiện tiếp theo; `GroupChatManager` của AutoGen cũng vậy. LangGraph chỉ tiêu tốn token ở những nơi bạn viết `llm.invoke`. Đường dẫn đơn agent của Agno rất nhẹ.

Khi chi phí mỗi lần chạy là quan trọng, hãy ưu tiên định tuyến rõ ràng (các cạnh của LangGraph, `speaker_selection_method` của AutoGen) thay vì định tuyến do LLM chọn.

### Khả năng tương tác

- **LangGraph** ↔ các công cụ, retriever, LLM của **LangChain**. Adapter MCP hạng nhất (các công cụ được nhập dưới dạng máy chủ MCP).
- **CrewAI** ↔ các công cụ kế thừa từ `BaseTool`; các công cụ LangChain, LlamaIndex và MCP đều có thể thích ứng. Ủy quyền giữa các crew thông qua `allow_delegation=True`.
- **AutoGen** → `FunctionTool` bao bọc bất kỳ callable nào của Python; có sẵn adapter MCP. Kết nối chặt chẽ với hệ sinh thái AG2 cho các mô hình agent-to-agent.
- **Agno** → decorator `@tool` hoặc lớp con BaseTool; adapter MCP; các công cụ có thể được chia sẻ giữa các agent và các đội.

## Kỹ năng

> Bạn có thể giải thích, trong một câu, tại sao một framework nhất định lại phù hợp với một vấn đề agent nhất định.

Danh sách kiểm tra trước khi xây dựng:

1. **Vẽ hình thái.** Đây có phải là một đồ thị (trạng thái định kiểu, các chuyển đổi có tên)? Một vở kịch vai trò (các chuyên gia chuyển giao công việc)? Một cuộc trò chuyện (các agent nói chuyện cho đến khi xong)? Một agent đơn lẻ với các công cụ?
2. **Quyết định ai phân nhánh.** Phân nhánh do nhà phát triển quyết định → LangGraph. Do agent quản lý quyết định → CrewAI phân cấp. Phân nhánh nảy sinh từ trò chuyện → AutoGen. Do gọi công cụ quyết định → Agno.
3. **Kiểm tra ngân sách trạng thái.** Bạn có cần khôi phục từ checkpoint? Du hành thời gian? Ngắt quãng bởi con người giữa chừng? Nếu có, LangGraph là mặc định; các phiên của Agno bao gồm trạng thái trong phạm vi hội thoại.
4. **Kiểm tra ngân sách chi phí.** Định tuyến do LLM chọn tốn thêm token mỗi lượt. Nếu agent chạy hàng nghìn lần mỗi ngày, hãy ưu tiên định tuyến rõ ràng.
5. **Dự trù chi phí framework.** Mỗi framework là một phụ thuộc khác. Nếu tác vụ chỉ là hai lần gọi LLM và một công cụ, hãy viết 30 dòng Python thuần; không có framework nào rẻ hơn là không dùng framework nào.

Hãy từ chối việc sử dụng framework trước khi bạn có thể vẽ được đồ thị, sơ đồ tổ chức, cuộc trò chuyện hoặc hộp agent. Hãy từ chối chọn một framework buộc bạn phải chiến đấu với mô hình trạng thái của nó cho thứ mà bạn thực sự cần.

## Ma trận quyết định

| Hình thái vấn đề | Framework ưu tiên | Tại sao |
|---------------|---------------------|-----|
| DAG quy trình làm việc với trạng thái định kiểu, phê duyệt của con người, chạy dài hạn | LangGraph | Trạng thái hạng nhất, checkpointer, ngắt quãng, du hành thời gian. |
| Pipeline nghiên cứu / viết với các vai trò riêng biệt | CrewAI (tuần tự) hoặc các subgraph của LangGraph | Vai trò-trên-tác vụ rất rẻ để thể hiện trong CrewAI; mở rộng với LangGraph khi phân nhánh trở nên phức tạp. |
| Đối thoại người đề xuất-người phê bình hoặc giáo viên-học sinh | AutoGen | Trò chuyện hai agent là hình thái tự nhiên của nó. |
| Agent đơn lẻ với công cụ, phiên, bộ nhớ | Agno | Thiết lập mỏng nhất, lưu trữ và bộ nhớ tích hợp sẵn. |
| Hàng nghìn fanout song song với các bộ giảm | LangGraph + `Send` | Framework duy nhất có API điều phối song song hạng nhất. |
| Prototype nhanh, không cam kết framework | Python thuần + SDK nhà cung cấp | Không framework là framework nhanh nhất. |

```figure
l5-framework-fit
```

## Bài tập

1. **Dễ.** Thực hiện cùng một tác vụ — "nghiên cứu trụ sở của Anthropic, viết bản tóm tắt 200 từ, trích dẫn nguồn" — và triển khai nó trong LangGraph (bốn node: lập kế hoạch, tìm kiếm, viết, trích dẫn) và trong CrewAI (ba vai trò: nhà nghiên cứu, người viết, biên tập viên). Báo cáo chi phí token mỗi lần chạy và số dòng code.
2. **Trung bình.** Xây dựng cùng tác vụ đó trong AutoGen (trò chuyện nhà nghiên cứu ↔ người viết, biên tập viên tham gia qua `GroupChat`) và Agno (một agent đơn lẻ với `search_tools` và `write_tools`, cộng với kho lưu trữ phiên). Xếp hạng bốn cách triển khai dựa trên (a) chi phí mỗi lần chạy, (b) khả năng khôi phục sau khi crash, (c) khả năng chèn phê duyệt của con người trước bước viết.
3. **Khó.** Xây dựng một script cây quyết định `pick_framework.py` nhận mô tả vấn đề ngắn (JSON: `{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`) và trả về đề xuất kèm lý do trong một câu. Xác minh nó trên sáu trường hợp bạn tự thiết kế.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Orchestration | "Cách các agent phối hợp" | Lớp quyết định node/vai trò/agent nào chạy tiếp theo. |
| Durable state | "Khôi phục sau khi khởi động lại" | Trạng thái tồn tại sau khi tiến trình chết, gắn với checkpoint hoặc kho lưu trữ phiên. |
| LLM-selected routing | "Để mô hình quyết định" | Một LLM lập kế hoạch chọn bước tiếp theo mỗi lượt; linh hoạt nhưng tốn token cho mỗi quyết định. |
| Explicit routing | "Nhà phát triển quyết định" | Một hàm Python hoặc cạnh tĩnh chọn bước tiếp theo; rẻ và có thể kiểm toán. |
| Crew | "Một đội CrewAI" | Vai trò + tác vụ + quy trình (tuần tự hoặc phân cấp) được đóng gói thành một đơn vị có thể chạy. |
| GroupChat | "Trò chuyện đa agent của AutoGen" | Một cuộc hội thoại được quản lý giữa N agent với bộ chọn người nói. |
| Team (Agno) | "Agno đa agent" | Chế độ định tuyến / điều phối / cộng tác trên một tập hợp các agent. |
| StateGraph | "Đồ thị của LangGraph" | Abstraction trạng thái định kiểu, node, cạnh điều kiện, checkpointer. |

## Đọc thêm

- [Tài liệu LangGraph](https://langchain-ai.github.io/langgraph/) — StateGraph, checkpointers, ngắt quãng, du hành thời gian.
- [Tài liệu CrewAI](https://docs.crewai.com/) — Crews, Flows, Agents, Tasks, Processes.
- [Tài liệu AutoGen](https://microsoft.github.io/autogen/) — ConversableAgent, GroupChat, teams, tools.
- [Tài liệu Agno](https://docs.agno.com/) — Agent, Team, Workflow, storage, memory.
- [Anthropic — Building effective agents (Tháng 12/2024)](https://www.anthropic.com/research/building-effective-agents) — thư viện mẫu (chuỗi prompt, định tuyến, song song hóa, điều phối-worker, đánh giá-tối ưu) không phụ thuộc framework.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — vòng lặp mà mọi framework đều trang trí thêm.
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — tài liệu thiết kế của AutoGen.
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — nền tảng nhập vai mà các persona stack kiểu CrewAI xây dựng dựa trên đó.
- Phase 11 · 16 (LangGraph) — framework mà bài học này lấy làm chuẩn so sánh.
- Phase 11 · 19 (Reflexion) — một mẫu ánh xạ rõ ràng tới LangGraph nhưng lại khó khăn với CrewAI.
- Phase 11 · 22 (Production observability) — cách đo lường bất kỳ framework nào bạn chọn.