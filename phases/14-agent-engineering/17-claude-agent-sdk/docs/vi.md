# The Harness as a Library — Subagents and Session Store

> Một harness mà bạn có thể import: các công cụ tích hợp sẵn, subagent để cô lập ngữ cảnh, các hook, truyền trace W3C, và duy trì phiên làm việc (session persistence). Claude Agent SDK là ví dụ tham chiếu — dạng thư viện của harness Claude Code — và Claude Managed Agents là giải pháp thay thế được host sẵn cho các tác vụ async chạy dài hạn.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time:** ~75 minutes

## Learning Objectives

- Giải thích sự khác biệt giữa Anthropic Client SDK (API thô) và Claude Agent SDK (dạng harness).
- Mô tả subagent — song song hóa và cô lập ngữ cảnh — và khi nào nên sử dụng chúng.
- Nêu tên bề mặt session store của Python SDK (`append`, `load`, `list_sessions`, `delete`, `list_subkeys`) và vai trò của `--session-mirror`.
- Triển khai một harness stdlib với các công cụ tích hợp, tạo subagent với ngữ cảnh cô lập, các lifecycle hook, và một session store.

## The Problem

Một LLM API thô chỉ cung cấp cho bạn một vòng lặp request-response. Một agent trong môi trường production cần thực thi công cụ, kết nối MCP server, các lifecycle hook, tạo subagent, duy trì phiên làm việc, và truyền trace. Claude Agent SDK cung cấp cấu trúc này dưới dạng thư viện — cùng một harness mà Claude Code sử dụng, được mở ra cho các agent tùy chỉnh.

## The Concept

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`).** Messages API thô. Bạn tự quản lý vòng lặp, công cụ và trạng thái.
- **Agent SDK (`claude-agent-sdk`).** Thực thi công cụ tích hợp, kết nối MCP, các hook, tạo subagent, session store. Vòng lặp Claude Code dưới dạng thư viện.

### Built-in tools

SDK cung cấp sẵn hơn 10 công cụ: đọc/ghi file, shell, grep, glob, web fetch, và nhiều hơn nữa. Các công cụ tùy chỉnh được đăng ký thông qua giao diện tool-schema tiêu chuẩn.

### Subagents

Hai mục đích được Anthropic ghi nhận:

1. **Parallelization (Song song hóa).** Chạy các công việc độc lập đồng thời. "Tìm file test cho mỗi module trong số 20 module này" là 20 tác vụ subagent song song.
2. **Context isolation (Cô lập ngữ cảnh).** Các subagent sử dụng cửa sổ ngữ cảnh riêng của chúng; chỉ kết quả mới được trả về cho orchestrator. Ngân sách ngữ cảnh của orchestrator được bảo toàn.

Các bổ sung gần đây của Python SDK: `list_subagents()`, `get_subagent_messages()` để đọc transcript của subagent.

### Session store

Đồng bộ giao thức với TypeScript:

- `append(session_id, message)` — thêm một lượt hội thoại.
- `load(session_id)` — khôi phục hội thoại.
- `list_sessions()` — liệt kê.
- `delete(session_id)` — với cascade đến các session của subagent.
- `list_subkeys(session_id)` — liệt kê các subagent key.

`--session-mirror` (CLI flag) phản chiếu transcript ra một file bên ngoài khi nó đang stream, phục vụ cho việc debug.

### Hooks

Các lifecycle hook bạn có thể đăng ký:

- `PreToolUse`, `PostToolUse` — kiểm soát hoặc kiểm tra các lời gọi công cụ.
- `SessionStart`, `SessionEnd` — thiết lập và dọn dẹp.
- `UserPromptSubmit` — xử lý input của người dùng trước khi model nhìn thấy.
- `PreCompact` — chạy trước khi nén ngữ cảnh (context compaction).
- `Stop` — dọn dẹp khi agent thoát.
- `Notification` — các cảnh báo kênh phụ (side-channel).

Hooks là cách mà pro-workflow (tham chiếu giáo trình Phase 14) và các hệ thống tương tự thêm vào các hành vi xuyên suốt.

### W3C trace context

Các OTel span đang hoạt động trên caller sẽ truyền vào subprocess CLI thông qua các header W3C trace context. Toàn bộ trace đa tiến trình sẽ hiển thị như một trace duy nhất trong backend của bạn.

### Claude Managed Agents

Giải pháp thay thế được host sẵn (beta header `managed-agents-2026-04-01`). Các tác vụ async chạy dài hạn, tích hợp sẵn prompt caching, tích hợp sẵn compaction. Đánh đổi quyền kiểm soát để lấy hạ tầng được quản lý.

### Where this pattern goes wrong

- **Subagent over-spawn.** Tạo 100 subagent cho 100 tác vụ nhỏ. Overhead sẽ chiếm ưu thế. Hãy batch chúng lại.
- **Hook creep.** Mỗi team thêm hook; thời gian khởi động tăng vọt. Hãy review các hook hàng quý.
- **Session bloat.** Các session tích tụ; kích thước tăng lên. Sử dụng `list_sessions` + chính sách hết hạn.

```figure
ae-subagent-isolation
```

## Build It

`code/main.py` triển khai cấu trúc SDK trong stdlib:

- `Tool`, `ToolRegistry` với các `read_file`, `write_file`, `list_dir` tích hợp sẵn.
- `Subagent` — ngữ cảnh riêng tư, chạy cô lập, kết quả được trả về.
- `SessionStore` — append, load, list, delete, list_subkeys.
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`.
- Một bản demo: agent chính tạo 3 subagent song song (mỗi cái đều cô lập), tổng hợp kết quả, duy trì session.

Chạy nó:

```
python3 code/main.py
```

Trace cho thấy sự cô lập ngữ cảnh của subagent (kích thước ngữ cảnh của orchestrator vẫn được giới hạn), thực thi hook, và duy trì session.

## Use It

- **Claude Agent SDK** cho các sản phẩm ưu tiên Claude muốn có cấu trúc harness của Claude Code.
- **Claude Managed Agents** cho các tác vụ async chạy dài hạn được host sẵn.
- **OpenAI Agents SDK** (Bài 16) cho các đối tác ưu tiên OpenAI.
- **LangGraph + custom tools** nếu bạn muốn một máy trạng thái (state machine) dạng đồ thị.

## Ship It

`outputs/skill-claude-agent-scaffold.md` tạo khung cho một ứng dụng Claude Agent SDK với subagent, hook, session store, đính kèm MCP server, và truyền trace W3C.

## Exercises

1. Thêm một subagent spawner batch 20 tác vụ thành các nhóm 5 subagent song song. Đo kích thước ngữ cảnh của orchestrator so với cách chạy một tác vụ mỗi lần.
2. Triển khai một hook `PreToolUse` để giới hạn tốc độ (rate-limit) các lời gọi `write_file` (5 lần mỗi phút cho mỗi session). Trace hành vi này.
3. Kết nối `list_subkeys` để render cây subagent. Cấu trúc lồng sâu trông như thế nào?
4. Chuyển đổi bản demo sang package Python `claude-agent-sdk` thực tế. Những gì thay đổi về việc đăng ký công cụ?
5. Đọc tài liệu về Claude Managed Agents. Khi nào bạn nên chuyển từ self-hosted sang managed?

## Key Terms

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Agent SDK | "Claude Code dưới dạng thư viện" | Cấu trúc harness: công cụ, MCP, hook, subagent, session store |
| Subagent | "Agent con" | Ngữ cảnh riêng, ngân sách riêng; kết quả được đẩy lên trên |
| Session store | "DB hội thoại" | Lưu, tải, liệt kê, xóa các lượt hội thoại với cascade subagent |
| Hook | "Callback vòng đời" | Trước/sau công cụ, session, gửi prompt, nén, dừng |
| W3C trace context | "Trace xuyên tiến trình" | Parent span truyền vào subprocess CLI |
| Managed Agents | "Harness được host" | Tác vụ async chạy dài hạn do Anthropic host |
| `--session-mirror` | "Phản chiếu transcript" | Ghi các lượt hội thoại vào file bên ngoài khi chúng đang stream |
| MCP server | "Bề mặt công cụ" | Nguồn công cụ/tài nguyên bên ngoài được gắn vào agent |

## Further Reading

- [Tổng quan về Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — dạng thư viện của Claude Code
- [Anthropic, Xây dựng agent với Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — các mô hình production
- [Tổng quan về Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) — giải pháp thay thế được host sẵn
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — đối tác tương đương