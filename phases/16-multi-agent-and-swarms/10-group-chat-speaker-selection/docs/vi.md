# Group Chat và Speaker Selection

> Điều phối hội thoại chia sẻ (Shared-conversation orchestration) đặt N agent vào trong một cuộc hội thoại; một hàm chọn (LLM, round-robin, hoặc tùy chỉnh) sẽ quyết định ai là người nói tiếp theo. Đây là nguyên mẫu của hội thoại đa agent (multi-agent) nổi trội — các agent không biết vai trò của mình trong một đồ thị tĩnh, chúng chỉ phản ứng với nhóm tin nhắn chung. AutoGen GroupChat và AG2 GroupChat là các bản triển khai tham chiếu: ngữ nghĩa GroupChat của AutoGen v0.2 được bảo tồn trong nhánh AG2; AutoGen v0.4 đã viết lại nó dưới dạng mô hình actor hướng sự kiện (event-driven actor model). Microsoft đã đưa AutoGen vào chế độ bảo trì vào tháng 2 năm 2026 và hợp nhất nó với Semantic Kernel thành Microsoft Agent Framework (RC tháng 2 năm 2026). Nguyên mẫu GroupChat vẫn tồn tại trong cả AG2 và Microsoft Agent Framework — học một lần, dùng mọi nơi.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Vấn đề

Các đồ thị tĩnh (LangGraph) rất tuyệt vời khi quy trình làm việc đã được biết trước. Các cuộc hội thoại thực tế không tĩnh: đôi khi coder hỏi reviewer, đôi khi hỏi researcher, đôi khi hỏi writer. Việc hardcode mọi khả năng chuyển giao sẽ tạo ra sự bùng nổ các cạnh (edge explosion). Bạn muốn *các agent phản ứng với một nhóm tin nhắn chung*, với một hàm nào đó quyết định ai sẽ nói tiếp theo.

Đó chính xác là những gì AutoGen GroupChat thực hiện.

## Khái niệm

### Hình thái

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

Mỗi agent đều nhìn thấy mọi tin nhắn. Một hàm chọn được gọi ở mỗi lượt để chọn người nói tiếp theo.

### Ba kiểu chọn (selector)

**Round-robin.** Chu kỳ cố định. Có tính xác định. Quy mô tăng tuyến tính theo N nhưng bỏ qua ngữ cảnh — một coder vẫn đến lượt ngay cả khi chủ đề là đánh giá pháp lý.

**LLM-selected.** Một lệnh gọi đến LLM để đọc nhóm tin nhắn gần đây và trả về người nói tiếp theo phù hợp nhất. Nhận biết ngữ cảnh nhưng chậm: mỗi lượt đều thêm một lệnh gọi LLM. Đây là mặc định của AutoGen.

**Custom.** Một hàm Python với bất kỳ logic nào bạn muốn. Điển hình: LLM-selected với các quy tắc dự phòng (ví dụ: "luôn để người kiểm chứng nói sau coder").

### API ConversableAgent

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` chứa hàm chọn. Khi một agent hoàn thành lượt của mình, manager sẽ gọi hàm chọn, hàm này trả về agent tiếp theo. Vòng lặp tiếp tục cho đến khi đạt điều kiện kết thúc.

### Kết thúc (Termination)

Ba mô hình phổ biến:

- **Max rounds.** Giới hạn cứng về tổng số lượt.
- **Token "TERMINATE".** Các agent có thể phát ra một tin nhắn đánh dấu; manager sẽ dừng lại khi một tin nhắn như vậy xuất hiện.
- **Kiểm tra đạt mục tiêu (Goal-reached check).** Một trình kiểm chứng nhẹ chạy mỗi lượt và dừng cuộc trò chuyện khi hoàn thành.

### Nguồn gốc: các nhánh và sự hợp nhất

Vào đầu năm 2025, Microsoft bắt đầu viết lại lớn AutoGen (v0.4) dựa trên mô hình actor hướng sự kiện. Cộng đồng đã fork ngữ nghĩa GroupChat của AutoGen v0.2 thành AG2, bảo tồn API mà những người dùng sớm đã tích hợp.

Vào tháng 2 năm 2026, Microsoft thông báo AutoGen sẽ chuyển sang chế độ bảo trì, với mô hình actor hướng sự kiện được hợp nhất vào **Microsoft Agent Framework** (RC tháng 2 năm 2026, hiện đã hợp nhất với Semantic Kernel). Khái niệm GroupChat tồn tại trong cả hai lộ trình; chi tiết triển khai khác nhau. AG2 là upstream ưu tiên cho mã tương thích với v0.2.

### Khi nào GroupChat phù hợp

- **Hội thoại nổi trội (Emergent conversations).** Bạn không muốn nối dây trước mọi khả năng người nói tiếp theo.
- **Tác vụ trộn vai trò.** Coder hỏi researcher, researcher hỏi archivist, archivist hỏi lại coder. Luồng không phải là một DAG.
- **Giải quyết vấn đề mang tính khám phá.** Hãy nghĩ về "cuộc họp động não", không phải "dây chuyền lắp ráp".

### Khi nào nó thất bại

- **Tính xác định nghiêm ngặt.** Hàm chọn LLM có thể không nhất quán. Cùng một prompt, các lần chạy khác nhau, người nói tiếp theo khác nhau.
- **Sycophancy cascades (Hiệu ứng xu nịnh).** Các agent nghe theo bất cứ ai nói tự tin nhất. Hãy counter-prompt một cách rõ ràng.
- **Context bloat (Phình ngữ cảnh).** Mỗi agent đọc mọi tin nhắn; sau 10 lượt, ngữ cảnh trở nên khổng lồ. Sử dụng projection (Bài 15) để giới hạn phạm vi xem.
- **Hot speakers.** Một agent thống trị cuộc hội thoại vì hàm chọn ưu tiên các chuyên môn của nó. Hãy đưa sự cân bằng người nói vào làm một tính năng của hàm chọn.

### Group chat vs supervisor

Cùng các nguyên mẫu, các mặc định khác nhau:

- Supervisor: một agent lập kế hoạch và những người khác thực thi. Hàm chọn là "hỏi người lập kế hoạch xem phải làm gì".
- Group chat: tất cả các agent là ngang hàng; hàm chọn là một hàm trên nhóm tin nhắn chung.

Cả hai đều sử dụng bốn nguyên mẫu từ Bài 04. Group chat mặc định sử dụng điều phối LLM-selected và trạng thái chia sẻ toàn bộ nhóm.

```figure
swarm-speaker
```

## Xây dựng

`code/main.py` triển khai một GroupChat từ đầu bằng stdlib. Ba agent (coder, reviewer, manager), các biến thể round-robin và LLM-selected, và kết thúc dựa trên token `TERMINATE`.

Bản demo in ra bản ghi cuộc hội thoại cộng với dấu vết quyết định của hàm chọn cho cả hai biến thể.

Chạy:

```
python3 code/main.py
```

## Sử dụng

`outputs/skill-groupchat-selector.md` cấu hình một hàm chọn GroupChat cho một tác vụ nhất định — round-robin vs LLM-selected vs custom, và các đầu vào của hàm chọn (tin nhắn gần đây, chuyên môn của agent, số lượt) cần sử dụng.

## Triển khai

Danh sách kiểm tra:

- **Giới hạn Max rounds.** Luôn luôn. 10-20 cho các tác vụ điển hình.
- **Chỉ số cân bằng người nói.** Theo dõi số lượt mỗi agent; cảnh báo khi sự mất cân bằng vượt quá ngưỡng.
- **Token kết thúc.** `TERMINATE` hoặc một agent kiểm chứng chuyên dụng.
- **Projection hoặc bộ nhớ có phạm vi.** Sau khoảng 10 tin nhắn, hãy cân nhắc cung cấp cho mỗi agent một cái nhìn có phạm vi để ngăn chặn phình ngữ cảnh.
- **Ghi nhật ký hàm chọn.** Đối với các biến thể LLM-selected, hãy ghi lại cả đầu vào của hàm chọn và lựa chọn của nó. Nếu không, việc gỡ lỗi là không thể.

## Bài tập

1. Chạy `code/main.py`. So sánh cuộc hội thoại giữa round-robin và LLM-selected. Agent nào thống trị trong mỗi trường hợp?
2. Thêm quy tắc "max-speaks-per-agent" vào hàm chọn. Nó ảnh hưởng thế nào đến bản ghi?
3. Triển khai kết thúc khi đạt mục tiêu: dừng khi reviewer trả về "approved". Nó kích hoạt bao nhiêu lần trước khi đạt giới hạn lượt?
4. Đọc tài liệu ổn định của AutoGen về GroupChat (https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html). Xác định hàm chọn mặc định được sử dụng bởi `GroupChatManager`.
5. Đọc repo AG2 (https://github.com/ag2ai/ag2) và so sánh GroupChat v0.2 của nó với phiên bản hướng sự kiện v0.4. Thuộc tính cụ thể nào (thông lượng, khả năng chịu lỗi, khả năng kết hợp) mà v0.4 bổ sung?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|----------------|------------------------|
| GroupChat | "Các agent trong một phòng chat" | Nhóm tin nhắn chia sẻ + hàm chọn. Nguyên mẫu AutoGen / AG2. |
| Speaker selection | "Ai nói tiếp theo" | Hàm chọn agent tiếp theo. Round-robin, LLM-selected, hoặc tùy chỉnh. |
| GroupChatManager | "Chủ trì cuộc họp" | Thành phần AutoGen sở hữu hàm chọn và lặp qua các lượt. |
| ConversableAgent | "Agent cơ sở" | Lớp cơ sở AutoGen; một agent có thể gửi và nhận tin nhắn. |
| Termination token | "Từ khóa 'dừng'" | Chuỗi đánh dấu (thường là `TERMINATE`) kết thúc cuộc trò chuyện. |
| Hot speaker | "Một agent thống trị" | Chế độ thất bại khi hàm chọn liên tục chọn cùng một agent. |
| Context bloat | "Nhóm tin nhắn tăng không giới hạn" | Mỗi agent đọc mọi tin nhắn trước đó; ngữ cảnh tăng theo lượt. |
| Projection | "Cái nhìn có phạm vi" | Cái nhìn theo vai trò vào nhóm tin nhắn chung để ngăn phình ngữ cảnh. |

## Đọc thêm

- [Tài liệu group chat AutoGen](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — bản triển khai tham chiếu
- [Repo AG2](https://github.com/ag2ai/ag2) — sự tiếp nối cộng đồng của AutoGen v0.2
- [Tài liệu Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/) — người kế nhiệm hợp nhất, RC tháng 2 năm 2026
- [Ghi chú phát hành AutoGen v0.4](https://microsoft.github.io/autogen/stable/) — chi tiết về việc viết lại mô hình actor hướng sự kiện