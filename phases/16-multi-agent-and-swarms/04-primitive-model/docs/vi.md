# Mô hình Nguyên thủy Đa tác nhân (Multi-Agent Primitive Model)

> Chỉ với bốn nguyên thủy — tác nhân (agent), bàn giao (handoff), trạng thái chia sẻ (shared state), và điều phối viên (orchestrator) — chúng ta có thể bao quát toàn bộ không gian thiết kế bốn chiều. Các framework đa tác nhân lớn ra mắt trong năm 2026 (AutoGen, LangGraph, CrewAI, OpenAI Agents SDK, Microsoft Agent Framework) đều là các điểm nằm trong không gian này. Bài học này sẽ xây dựng chúng từ con số không, chạy một hệ thống mô phỏng trên cả bốn, sau đó ánh xạ mọi framework lớn lên cùng các trục này để bạn có thể hiểu bất kỳ bản phát hành mới nào chỉ trong một đoạn văn.

**Type:** Học tập
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 phút

## Vấn đề

Cứ mỗi sáu tháng lại có một framework đa tác nhân mới ra đời. AutoGen năm 2023. CrewAI năm 2024. LangGraph và OpenAI Swarm năm 2024. Google ADK vào tháng 4 năm 2025. Microsoft Agent Framework RC vào tháng 2 năm 2026. Mỗi thông cáo báo chí đều khẳng định họ là "trừu tượng hóa đúng đắn".

Nếu bạn cố gắng học từng cái một, bạn sẽ kiệt sức. Các API trông khác nhau. Tài liệu không thống nhất về định nghĩa của một "tác nhân". Một framework gọi bộ nhớ chia sẻ là "bảng đen" (blackboard), một cái khác gọi là "nhóm tin nhắn" (message pool), cái thứ ba gọi là "StateGraph". Bạn bắt đầu nghi ngờ rằng lĩnh vực này chỉ đang chạy theo phong trào.

Thực tế không phải vậy. Bên dưới lớp vỏ marketing, bốn nguyên thủy này rất ổn định. Hãy học chúng một lần, bạn sẽ đọc hiểu mọi framework mới chỉ trong một đoạn văn.

## Khái niệm

### Bốn nguyên thủy

1. **Agent** — một system prompt cộng với danh sách công cụ. Không trạng thái (stateless); mỗi lần chạy đều bắt đầu từ system prompt và lịch sử tin nhắn hiện tại.
2. **Handoff** — sự chuyển giao quyền kiểm soát có cấu trúc từ tác nhân này sang tác nhân khác. Về mặt kỹ thuật, đây là một lệnh gọi công cụ trả về một tác nhân mới hoặc một cạnh đồ thị tuân theo một điều kiện.
3. **Shared state** — bất kỳ cấu trúc dữ liệu nào mà nhiều tác nhân có thể đọc (đôi khi là ghi). Nhóm tin nhắn, bảng đen, kho lưu trữ key-value, bộ nhớ vector.
4. **Orchestrator** — thực thể quyết định ai sẽ là người tiếp theo lên tiếng. Các tùy chọn: đồ thị tường minh (xác định), bộ chọn người nói bằng LLM (mềm dẻo), lệnh gọi handoff của người nói trước đó (OpenAI Swarm), hoặc bộ lập lịch trên hàng đợi (kiến trúc swarm).

Đó là toàn bộ không gian thiết kế. Mỗi framework chọn các giá trị mặc định cho từng trục; phần còn lại chỉ là cú pháp bề mặt.

### Cách mọi framework năm 2026 ánh xạ vào mô hình này

| Framework | Agent | Handoff | Shared state | Orchestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | công cụ trả về Agent | vấn đề của người gọi | lệnh gọi handoff tiếp theo của LLM |
| AutoGen v0.4 / AG2 | `ConversableAgent` | bộ chọn người nói trong GroupChat | nhóm tin nhắn | hàm chọn (LLM hoặc round-robin) |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | đầu ra Task được xâu chuỗi | LLM quản lý hoặc thứ tự tĩnh |
| LangGraph | hàm node | cạnh đồ thị + điều kiện | `StateGraph` reducer | đồ thị, xác định |
| Microsoft Agent Framework | tác nhân + mẫu điều phối | tùy theo mẫu | luồng / ngữ cảnh | tùy theo mẫu |
| Google ADK | tác nhân + thẻ A2A | tác vụ A2A | các artifact A2A | host quyết định |

Sự khác biệt bề mặt trông rất lớn. Nhưng bên dưới: chỉ là bốn núm xoay giống nhau.

### Tại sao điều này quan trọng

Khi bạn đã thấy các nguyên thủy, việc so sánh framework trở thành một danh sách kiểm tra ngắn gọn:

- Orchestrator tin tưởng LLM để định tuyến (Swarm) hay ghim việc định tuyến trong mã nguồn (LangGraph)?
- Shared state là toàn bộ lịch sử (GroupChat) hay đã được chiếu (StateGraph reducer)?
- Các tác nhân có thể sửa đổi prompt của nhau (quản lý CrewAI) hay chỉ có thể bàn giao (Swarm)?

Ba câu hỏi đó trả lời 80% việc framework nào phù hợp với một vấn đề cụ thể. Bạn sẽ ngừng tìm kiếm "framework đa tác nhân tốt nhất" và bắt đầu thiết kế cho trục mà bạn thực sự quan tâm.

### Sự thấu hiểu về tính không trạng thái (stateless)

Mọi nguyên thủy ngoại trừ shared state đều không có trạng thái. Agent là một hàm của (prompt, công cụ). Handoff là một lệnh gọi hàm. Orchestrator là một bộ lập lịch. **Thứ duy nhất có trạng thái trong hệ thống là shared state.** Đó là nơi chứa tất cả các lỗi thú vị: nhiễm độc bộ nhớ (Bài 15), thứ tự tin nhắn, quản lý phiên bản, tranh chấp ghi.

Các framework ẩn shared state (Swarm) đẩy vấn đề cho người gọi. Các framework tập trung hóa nó (LangGraph checkpoint, AutoGen pool) làm cho nó có thể kiểm tra được nhưng lại chuyển chi phí điều phối sang việc triển khai shared state.

### Giải phẫu một nguyên thủy đơn lẻ

#### Agent

```
Agent = (system_prompt, tools, model, optional_name)
```

Không bộ nhớ. Không trạng thái. Hai tác nhân có cùng system prompt và công cụ là có thể thay thế cho nhau. Mọi thứ trông giống như trạng thái của từng tác nhân thực chất đều nằm trong shared state hoặc giao thức handoff.

#### Handoff

```
Handoff = (from_agent, to_agent, reason, payload)
```

Ba cách triển khai chiếm ưu thế:

- **Trả về hàm** — công cụ trả về tác nhân tiếp theo. Đây là mô hình OpenAI Swarm. Các tác nhân mang theo thông tin định tuyến trong lược đồ công cụ của chúng.
- **Cạnh đồ thị** — LangGraph. Các cạnh mang tính khai báo. LLM tạo ra một giá trị; một điều kiện sẽ chọn node tiếp theo.
- **Chọn người nói** — AutoGen GroupChat. Một hàm chọn (đôi khi chính là một lệnh gọi LLM) đọc nhóm tin nhắn và chọn người nói tiếp theo.

#### Shared state

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

Tối thiểu là một danh sách các tin nhắn. Thường là nhiều hơn: các artifact có cấu trúc (đầu ra Task của CrewAI), ngữ cảnh có kiểu (LangGraph reducers), bộ nhớ ngoài (MCP, vector DB).

Hai cấu trúc liên kết: **nhóm đầy đủ** (mọi tác nhân thấy mọi tin nhắn) và **được chiếu** (các tác nhân thấy một góc nhìn theo vai trò). Nhóm đầy đủ đơn giản nhưng mở rộng kém. Nhóm được chiếu mở rộng tốt nhưng yêu cầu thiết kế lược đồ từ trước.

#### Orchestrator

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

Bốn kiểu:

- **Tĩnh** — đồ thị được cố định tại thời điểm xây dựng (LangGraph xác định, CrewAI tuần tự).
- **LLM chọn** — một LLM đọc nhóm tin nhắn và chọn người nói tiếp theo (AutoGen, CrewAI phân cấp).
- **Dựa trên handoff** — tác nhân hiện tại quyết định bằng cách gọi công cụ handoff (Swarm).
- **Dựa trên hàng đợi** — các worker lấy việc từ hàng đợi chia sẻ; không có người nói tiếp theo tường minh (kiến trúc swarm, Matrix).

### Những gì thay đổi giữa các framework

Khi các nguyên thủy đã được cố định, các quyết định thiết kế còn lại là:

- **Chiến lược bộ nhớ** — checkpoint tạm thời so với bền vững (LangGraph checkpointer).
- **Ranh giới an toàn** — ai có thể phê duyệt một handoff (con người trong vòng lặp).
- **Hạch toán chi phí** — ngân sách token cho mỗi tác nhân.
- **Khả năng quan sát** — theo dõi các handoff, lưu trữ trạng thái để phát lại.

Tất cả đều có thể triển khai trên nền tảng các nguyên thủy. Không cái nào trong số đó là nguyên thủy mới.

```figure
a5-primitive-radar
```

## Xây dựng

`code/main.py` triển khai bốn nguyên thủy trong khoảng 150 dòng Python tiêu chuẩn. Không có LLM thực sự — mỗi tác nhân là một chính sách được viết kịch bản để tập trung vào cấu trúc điều phối.

Tệp này xuất ra:

- `Agent` — một dataclass gồm tên, system prompt, công cụ, hàm chính sách.
- `Handoff` — một hàm trả về một tác nhân mới.
- `SharedState` — một nhóm tin nhắn an toàn với luồng (thread-safe).
- `Orchestrator` — ba biến thể: `StaticOrchestrator`, `HandoffOrchestrator`, `LLMSelectorOrchestrator` (mô phỏng).

Bản demo chạy cùng một quy trình ba tác nhân (nghiên cứu → viết → đánh giá) qua cả ba loại orchestrator và in nhóm tin nhắn ở cuối. Bạn có thể thấy đầu ra chỉ khác nhau ở *ai là người chọn tiếp theo*; các tác nhân và shared state là giống hệt nhau giữa các lần chạy.

Chạy nó:

```
python3 code/main.py
```

Đầu ra mong đợi: ba lần chạy orchestrator, mỗi lần một kiểu. Mỗi lần in ra nhóm tin nhắn cuối cùng. Lần chạy dựa trên handoff đạt đến ít tác nhân hơn nếu nhà nghiên cứu quyết định rằng công việc đã hoàn thành sớm — đó là sự đánh đổi định tuyến của LLM ở quy mô nhỏ.

## Sử dụng

`outputs/skill-primitive-mapper.md` là một kỹ năng đọc bất kỳ codebase hoặc tài liệu framework đa tác nhân nào và trả về ánh xạ bốn nguyên thủy. Hãy chạy nó trên một bản phát hành framework mới để hiểu nhanh trong một đoạn văn trước khi đọc sâu vào tài liệu.

## Triển khai

Trước khi áp dụng một framework mới, hãy viết ánh xạ nguyên thủy cho nó. Nếu bạn không thể, tài liệu đó chưa đầy đủ hoặc framework đang phát minh ra một nguyên thủy thứ năm (hiếm gặp — hãy kiểm tra xem có kiểu shared state nào bạn chưa từng thấy không).

Ghim ánh xạ đó vào tài liệu kiến trúc của bạn. Khi một thành viên mới tham gia, hãy gửi cho họ ánh xạ trước khi gửi tài liệu API. Khi các phiên bản framework thay đổi, hãy diff ánh xạ, không phải changelog.

## Bài tập

1. Chạy `code/main.py` ba lần với các chính sách tác nhân khác nhau. Quan sát cách lựa chọn orchestrator thay đổi tác nhân nào được chạy.
2. Triển khai loại orchestrator thứ tư: dựa trên hàng đợi, nơi các tác nhân thăm dò shared state để tìm việc. Bế tắc (deadlock) nào có thể xảy ra và làm thế nào để phát hiện nó?
3. Lấy LangGraph quickstart (https://docs.langchain.com/oss/python/langgraph/workflows-agents) và viết lại nó dưới dạng bốn nguyên thủy. Những trừu tượng nào của LangGraph ánh xạ 1:1 và cái nào là trình bao bọc tiện ích?
4. Đọc sách hướng dẫn OpenAI Swarm (https://developers.openai.com/cookbook/examples/orchestrating_agents). Xác định nguyên thủy nào trong bốn nguyên thủy mà Swarm làm cho thuận tiện nhất, và cái nào nó đẩy cho người gọi.
5. Tìm một framework trong bảng này ẩn hoàn toàn shared state. Giải thích điều gì sẽ hỏng khi các tác nhân cần phối hợp qua các handoff mà không đọc lại lịch sử.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|-----------|---------------|------------------|
| Agent | "Một LLM với công cụ" | Một bộ ba `(system_prompt, tools, model)`. Không trạng thái. |
| Handoff | "Chuyển giao quyền kiểm soát" | Một lệnh gọi có cấu trúc đặt tên tác nhân tiếp theo và payload tùy chọn. Ba cách triển khai: trả về hàm, cạnh đồ thị, chọn người nói. |
| Shared state | "Bộ nhớ" / "ngữ cảnh" | Phần duy nhất có trạng thái của hệ thống đa tác nhân. Nhóm tin nhắn hoặc bảng đen. |
| Orchestrator | "Điều phối viên" | Người quyết định ai chạy tiếp theo. Đồ thị tĩnh, bộ chọn LLM, dựa trên handoff, hoặc dựa trên hàng đợi. |
| Nguyên thủy | "Trừu tượng hóa" | Một trong bốn trục mà mọi framework tham số hóa. Không phải tính năng của framework. |
| Nhóm tin nhắn | "Lịch sử chat chia sẻ" | Shared state toàn bộ lịch sử. Dễ suy luận, mở rộng kém. |
| Trạng thái được chiếu | "Góc nhìn theo phạm vi" | Góc nhìn theo vai trò vào shared state. Mở rộng tốt, yêu cầu thiết kế lược đồ. |
| Chọn người nói | "Ai nói tiếp theo" | Mẫu orchestrator nơi một hàm (thường là LLM) chọn tác nhân tiếp theo từ một nhóm. |

## Đọc thêm

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — cách diễn đạt rõ ràng nhất về điều phối dựa trên handoff
- [Tài liệu ổn định AutoGen](https://microsoft.github.io/autogen/stable/) — GroupChat + chọn người nói là tài liệu tham khảo cho điều phối do LLM chọn
- [Quy trình làm việc và tác nhân LangGraph](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — điều phối cạnh đồ thị và shared state dựa trên reducer
- [Giới thiệu CrewAI](https://docs.crewai.com/en/introduction) — các tác nhân theo vai trò-mục tiêu-bối cảnh, quy trình Tuần tự / Phân cấp
- [AG2 (cộng đồng tiếp nối AutoGen)](https://github.com/ag2ai/ag2) — dòng AutoGen v0.2 trực tiếp sau khi Microsoft chuyển v0.4 sang bảo trì