# Handoffs and Routines — Stateless Orchestration

> OpenAI's Swarm (tháng 10 năm 2024) đã chắt lọc việc điều phối đa tác nhân (multi-agent orchestration) thành hai nguyên lý cơ bản: **routines** (hướng dẫn + công cụ dưới dạng system prompt) và **handoffs** (một công cụ trả về một Agent khác). Không có máy trạng thái (state machine), không có DSL phân nhánh — LLM thực hiện điều hướng bằng cách gọi đúng công cụ handoff. OpenAI Agents SDK (tháng 3 năm 2025) là phiên bản kế nhiệm cho môi trường production. Bản thân Swarm vẫn là tài liệu tham khảo khái niệm rõ ràng nhất — toàn bộ mã nguồn của nó chỉ gói gọn trong vài trăm dòng. Mô hình này trở nên phổ biến vì bề mặt API chỉ đơn giản là "agent = prompt + tools; handoff = hàm trả về agent." Hạn chế: không lưu trạng thái (stateless), vì vậy bộ nhớ là vấn đề mà người gọi (caller) phải tự giải quyết.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 phút

## Vấn đề

Mọi framework đa tác nhân đều muốn bạn học DSL của riêng họ: các node và edge của LangGraph, các crew và task của CrewAI, GroupChat và manager của AutoGen. Các DSL này là những lớp trừu tượng thực sự, nhưng chúng làm cho mọi thứ trở nên cồng kềnh hơn mức cần thiết.

Swarm đi theo hướng ngược lại: tận dụng khả năng gọi công cụ (tool-calling) mà mô hình đã có sẵn. Handoffs trở thành các lệnh gọi công cụ. Bộ điều phối (orchestrator) chính là tác nhân đang nắm giữ cuộc hội thoại. Máy trạng thái được ẩn chứa trong các system prompt của các tác nhân.

## Khái niệm

### Hai nguyên lý cơ bản

**Routine.** Một system prompt xác định vai trò và các công cụ khả dụng của một tác nhân. Hãy coi nó như một tập hợp các hướng dẫn có phạm vi cụ thể: "bạn là tác nhân phân loại; nếu người dùng hỏi về hoàn tiền, hãy chuyển giao (handoff) cho tác nhân hoàn tiền."

**Handoff.** Một công cụ mà tác nhân có thể gọi để trả về một đối tượng Agent mới. Runtime của Swarm phát hiện giá trị trả về là Agent và chuyển đổi tác nhân đang hoạt động cho lượt tiếp theo.

Đó là toàn bộ sự trừu tượng.

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

System prompt của tác nhân phân loại khiến nó chọn đúng handoff dựa trên tin nhắn của người dùng. Việc gọi công cụ của LLM thực hiện công việc điều hướng.

### Tại sao nó trở nên phổ biến

- **API nhỏ gọn.** Chỉ cần học hai khái niệm.
- **Tận dụng những gì mô hình đã làm tốt.** Gọi công cụ đã là tiêu chuẩn công nghiệp trên các nhà cung cấp.
- **Không gánh nặng về máy trạng thái.** Bạn không cần mô tả đồ thị; các prompt của tác nhân tự mô tả việc chúng sẽ chuyển giao cho ai.

### Sự đánh đổi về tính stateless

Swarm hoàn toàn không lưu trạng thái giữa các lần chạy. Framework giữ lịch sử tin nhắn trong một lần chạy, nhưng nó không lưu trữ bất cứ thứ gì. Bộ nhớ, tính liên tục, các tác vụ chạy dài — tất cả đều là vấn đề của người gọi.

Trong môi trường production (OpenAI Agents SDK, tháng 3 năm 2025), đây là một trong những thay đổi chính: SDK bổ sung tính năng quản lý phiên (session management), guardrails và tracing tích hợp sẵn trong khi vẫn giữ nguyên nguyên lý handoff.

### Khi nào nên dùng Swarm/handoffs

- **Mô hình phân loại (Triage patterns).** Tác nhân tuyến đầu điều hướng người dùng đến chuyên gia.
- **Chuyển giao dựa trên kỹ năng.** "Nếu tác vụ cần code, hãy gọi coder; nếu cần nghiên cứu, hãy gọi researcher."
- **Các cuộc hội thoại ngắn, có giới hạn.** Hỗ trợ khách hàng, FAQ-to-ticket, các quy trình làm việc đơn giản.

### Khi nào Swarm gặp khó khăn

- **Các phiên dài với bộ nhớ chia sẻ.** Handoffs đặt lại trạng thái cuộc hội thoại về prompt của tác nhân mới cộng với lịch sử. Không có trạng thái bền vững giữa các tác nhân nếu không có bộ nhớ do người gọi quản lý.
- **Thực thi song song.** Handoff chỉ diễn ra từng cái một — tác nhân đang hoạt động sẽ chuyển đổi. Tính song song đòi hỏi người gọi phải điều phối nhiều lần chạy Swarm.
- **Kiểm toán và phát lại (Audit and replay).** Các lần chạy stateless rất khó để phát lại chính xác; lựa chọn handoff của LLM không mang tính tất định.

### OpenAI Agents SDK (tháng 3 năm 2025)

Phiên bản kế nhiệm cho production bổ sung:

- **Trạng thái phiên (Session state).** Luồng (thread) bền vững giữa các lần chạy.
- **Guardrails.** Các hook để xác thực đầu vào/đầu ra.
- **Tracing.** Mọi lệnh gọi công cụ và handoff đều được ghi lại.
- **Handoff filters.** Kiểm soát ngữ cảnh nào được chuyển giao khi handoff.

Nguyên lý handoff vẫn tồn tại; các yếu tố công thái học cho production được thêm vào xung quanh nó.

### Swarm vs GroupChat

Cả hai đều sử dụng điều hướng dựa trên LLM, nhưng chúng khác nhau ở **ai là người chọn tiếp theo**:

- GroupChat: một bộ chọn (hàm hoặc LLM) chọn người nói tiếp theo từ bên ngoài.
- Swarm: tác nhân hiện tại chọn người kế nhiệm bằng cách gọi công cụ handoff.

Swarm là "tác nhân quyết định bước tiếp theo"; GroupChat là "quản lý quyết định bước tiếp theo". Quyết định của Swarm nằm trong lệnh gọi công cụ của tác nhân đang hoạt động; quyết định của GroupChat nằm trong `GroupChatManager`.

```figure
sw-handoff-routing
```

## Xây dựng

`code/main.py` triển khai Swarm từ đầu: một dataclass Agent, cơ chế handoff (công cụ trả về Agent), và vòng lặp chạy (run loop) phát hiện việc chuyển đổi tác nhân.

Demo: một tác nhân phân loại điều hướng đến các chuyên gia hoàn tiền, bán hàng hoặc hỗ trợ. Mỗi chuyên gia có các công cụ riêng. Vòng lặp chạy in ra mỗi lần handoff.

Chạy:

```
python3 code/main.py
```

## Sử dụng

`outputs/skill-handoff-designer.md` thiết kế cấu trúc handoff cho một tác vụ cụ thể: những tác nhân nào tồn tại, chúng có thể gọi những handoff nào, ngữ cảnh nào được chuyển giao.

## Triển khai

Danh sách kiểm tra:

- **Ghi nhật ký handoff.** Mỗi handoff ghi lại một sự kiện trace với tác nhân nguồn, tác nhân đích, và bản chụp ngữ cảnh.
- **Quy tắc chuyển giao ngữ cảnh.** Quyết định những gì được chuyển khi handoff: toàn bộ lịch sử (tốn kém), N tin nhắn cuối cùng, hoặc một bản tóm tắt.
- **Guardrail khi handoff.** Việc handoff cho một chuyên gia với quyền công cụ khác phải được xác thực — nếu không, prompt injection có thể ép buộc các handoff không mong muốn.
- **Phát hiện vòng lặp.** Hai tác nhân chuyển qua lại cho nhau là lỗi phổ biến; hãy phát hiện bằng cách kiểm tra vòng lặp K bước gần nhất.
- **Tác nhân dự phòng (Fallback agent).** Nếu mục tiêu handoff không tồn tại, hãy quay lại một mặc định an toàn.

## Bài tập

1. Chạy `code/main.py`, phân loại đến tác nhân hoàn tiền. Xác nhận tác nhân hoạt động ở lượt thứ hai là refund.
2. Thêm quy tắc phát hiện vòng lặp: nếu cùng hai tác nhân đã chuyển giao 3 lần liên tiếp, hãy buộc thoát. Thiết kế phương án dự phòng.
3. Đọc tài liệu OpenAI Agents SDK về handoff filters. Triển khai phiên bản "tóm tắt khi handoff": tác nhân gửi đi nén ngữ cảnh thành một bản tóm tắt dạng gạch đầu dòng trước khi tác nhân nhận tiếp quản.
4. So sánh handoff của Swarm với bộ chọn GroupChatManager. Mô hình nào làm cho prompt injection trở nên tồi tệ hơn, và tại sao?
5. Đọc Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents). Xác định một quyết định thiết kế rõ ràng mà Swarm thực hiện nhưng OpenAI Agents SDK đã thay đổi hoặc giữ lại.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Routine | "Prompt của tác nhân" | System prompt + danh sách công cụ. Xác định vai trò và các handoff khả dụng. |
| Handoff | "Chuyển sang tác nhân khác" | Một công cụ mà tác nhân đang hoạt động có thể gọi để trả về một Agent mới. Runtime chuyển đổi tác nhân hoạt động. |
| Stateless | "Không có bộ nhớ giữa các lần chạy" | Swarm không lưu trữ bất cứ thứ gì; bộ nhớ là trách nhiệm của người gọi. |
| Active agent | "Ai đang nói" | Tác nhân hiện đang nắm giữ cuộc hội thoại. Handoff thay đổi điều này. |
| Context transfer | "Cái gì được chuyển khi handoff" | Chính sách về lịch sử mà tác nhân nhận được thấy: đầy đủ, N tin nhắn cuối, hoặc tóm tắt. |
| Handoff loop | "Tác nhân ping-pong" | Chế độ lỗi khi hai tác nhân liên tục chuyển qua lại cho nhau. |
| OpenAI Agents SDK | "Swarm cho production" | Phiên bản kế nhiệm tháng 3 năm 2025; thêm phiên, guardrails, tracing trên nền tảng nguyên lý handoff. |
| Handoff filter | "Cổng kiểm soát chuyển giao" | Tính năng SDK để kiểm tra và sửa đổi ngữ cảnh tại ranh giới handoff. |

## Đọc thêm

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — tài liệu tham khảo chính thức
- [OpenAI Swarm repo](https://github.com/openai/swarm) — triển khai gốc, được giữ làm tài liệu tham khảo khái niệm
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — phiên bản kế nhiệm cho production với các phiên và tracing
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — cách các subagent của Claude Code sử dụng mô hình giống handoff thông qua `Task`