# Capstone 01 — Terminal-Native Coding Agent

> Đến năm 2026, hình thái của một coding agent đã trở nên ổn định. Đó là một bộ khung TUI, một trạng thái kế hoạch (stateful plan), một bề mặt công cụ được sandbox, và một vòng lặp thực hiện lập kế hoạch, hành động, quan sát, và phục hồi. Claude Code, Cursor 3, và OpenCode đều trông giống hệt nhau khi nhìn từ xa. Capstone này yêu cầu bạn xây dựng một agent từ đầu đến cuối — đầu vào là CLI, đầu ra là pull request — và đo lường nó so với mini-swe-agent và Live-SWE-agent trên SWE-bench Pro. Bạn sẽ hiểu tại sao phần khó nhất không phải là gọi model, mà là vòng lặp công cụ, sandbox, và giới hạn chi phí cho một phiên chạy 50 lượt.

**Type:** Capstone
**Languages:** TypeScript / Bun (harness), Python (eval scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and protocols), Phase 14 (agents), Phase 15 (autonomous systems), Phase 17 (infrastructure)
**Phases exercised:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Time:** 35 giờ

## Vấn đề

Coding agent đã trở thành danh mục ứng dụng AI chiếm ưu thế vào năm 2026. Claude Code (Anthropic), Cursor 3 với Composer 2 và Agent Tabs (Cursor), Amp (Sourcegraph), OpenCode (112k stars), Factory Droids, và Google Jules đều phát hành các biến thể của cùng một kiến trúc: một bộ khung terminal, bề mặt công cụ được cấp quyền, sandbox, và vòng lặp lập kế hoạch-hành động-quan sát được xây dựng xung quanh một frontier model. Ranh giới này rất hẹp — Live-SWE-agent đạt 79.2% trên SWE-bench Verified với Opus 4.5 — nhưng kỹ năng kỹ thuật thì rất rộng. Hầu hết các chế độ lỗi không phải do model sai, mà là do sự mất ổn định của vòng lặp công cụ, nhiễm độc ngữ cảnh (context poisoning), chi phí token vượt kiểm soát, và các thao tác hệ thống tệp mang tính phá hủy.

Bạn không thể suy luận về các agent này từ bên ngoài. Bạn phải tự xây dựng một cái, chứng kiến vòng lặp bị treo ở lượt thứ 47 khi ripgrep trả về 8MB kết quả, và xây dựng lại lớp cắt tỉa (truncation layer). Đó chính là mục đích của capstone này.

## Khái niệm

Bộ khung (harness) có bốn bề mặt. **Plan** duy trì một đối tượng trạng thái kiểu TodoWrite mà model viết lại sau mỗi lượt. **Act** điều phối các lệnh gọi công cụ (read, edit, run, search, git). **Observe** thu thập stdout / stderr / mã thoát, cắt tỉa, và đưa bản tóm tắt ngược lại. **Recover** xử lý các lỗi công cụ mà không làm tràn cửa sổ ngữ cảnh hoặc lặp vô tận. Hình thái năm 2026 bổ sung thêm một thứ: **hooks**. `PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `Notification`, `Stop`, và `PreCompact` — các điểm mở rộng có thể cấu hình nơi người vận hành chèn chính sách, telemetry, và các rào chắn (guardrails).

Sandbox sử dụng E2B hoặc Daytona. Mỗi tác vụ chạy trong một devcontainer mới với git worktree được gắn ở chế độ đọc-ghi. Bộ khung không bao giờ chạm vào hệ thống tệp của máy chủ. Worktree sẽ bị xóa sau khi thành công hoặc thất bại. Kiểm soát chi phí được thực thi ở ba lớp: giới hạn token mỗi lượt, ngân sách đô la mỗi phiên, và giới hạn lượt cứng (thường là 50). Lớp quan sát là các OpenTelemetry spans với các quy ước ngữ nghĩa GenAI, được gửi đến một instance Langfuse tự lưu trữ.

## Kiến trúc

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## Stack

- Harness runtime: Bun 1.2 + Ink 5 (React-in-terminal)
- Model access: OpenRouter unified API với Claude Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro, Opus 4.5 (cho các tác vụ khó nhất)
- Tool transport: Model Context Protocol StreamableHTTP (MCP 2026 revision)
- Sandbox: E2B sandboxes (JS SDK) hoặc Daytona devcontainers
- Code search: ripgrep subprocess, tree-sitter parsers cho 17 ngôn ngữ (đã biên dịch trước)
- Isolation: `git worktree add` mỗi tác vụ, dọn dẹp khi thành công / thất bại
- Eval harness: SWE-bench Pro (tập con đã xác minh) + Terminal-Bench 2.0 + 30 tác vụ holdout của riêng bạn
- Observability: OpenTelemetry SDK với `gen_ai.*` semconv → Langfuse tự lưu trữ
- PR posting: GitHub App với token phạm vi hẹp, giới hạn trong repo mục tiêu

```figure
ce-agent-loop
```

## Xây dựng

1. **TUI và vòng lặp lệnh.** Scaffold một dự án Bun với Ink. Chấp nhận `agent run <repo> "<task>"`. In ra chế độ xem chia đôi: bảng kế hoạch (trên), luồng gọi công cụ (giữa), ngân sách token (dưới). Thêm tính năng hủy bằng Ctrl-C, kích hoạt hook `SessionEnd` trước khi thoát.

2. **Trạng thái kế hoạch.** Định nghĩa schema TodoWrite có kiểu (các mục pending / in_progress / done kèm ghi chú). Model viết lại toàn bộ trạng thái mỗi lượt dưới dạng một lệnh gọi công cụ — không để nó thay đổi dần dần. Lưu trạng thái kế hoạch vào `.agent/state.json` để có thể khôi phục khi bị treo.

3. **Bề mặt công cụ.** Định nghĩa sáu công cụ: `read_file`, `edit_file` (kèm xem trước diff), `ripgrep`, `tree_sitter_symbols`, `run_shell` (kèm timeout), `git` (trạng thái / diff / commit / push). Phơi bày qua MCP StreamableHTTP để bộ khung không phụ thuộc vào transport. Mọi công cụ đều trả về đầu ra đã cắt tỉa (giới hạn 4k token mỗi lần gọi).

4. **Bao bọc Sandbox.** Mỗi tác vụ tạo ra một E2B sandbox. `git worktree add -b agent/$TASK_ID` một nhánh mới. Tất cả các lệnh gọi công cụ thực thi bên trong sandbox. Hệ thống tệp máy chủ không thể truy cập được.

5. **Hooks.** Triển khai tất cả tám loại hook năm 2026. Kết nối ít nhất bốn hook do người dùng viết: (a) `PreToolUse` rào chắn lệnh phá hủy chặn `rm -rf` bên ngoài worktree, (b) `PostToolUse` kế toán token, (c) `SessionStart` khởi tạo ngân sách, (d) `Stop` ghi lại gói trace cuối cùng.

6. **Vòng lặp đánh giá.** Clone một tập con 30 vấn đề của SWE-bench Pro Python. Chạy bộ khung của bạn trên từng vấn đề. So sánh với mini-swe-agent (baseline tối thiểu) về pass@1, số lượt mỗi tác vụ, và chi phí mỗi tác vụ. Ghi kết quả vào `eval/results.jsonl`.

7. **Kiểm soát chi phí.** Các giới hạn cứng: 50 lượt, 200k ngữ cảnh, $5 mỗi tác vụ. Hook `PreCompact` tóm tắt các lượt cũ thành một khối trạng thái trước đó tại mốc 150k, giải phóng không gian cho các quan sát mới mà không làm mất kế hoạch.

8. **Đăng PR.** Khi thành công, bước cuối cùng là `git push` + một lệnh gọi GitHub API mở PR với kế hoạch và tóm tắt diff trong phần nội dung.

## Sử dụng

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## Phát hành

Kỹ năng cần đạt được nằm ở `outputs/skill-terminal-coding-agent.md`. Với đường dẫn repo và mô tả tác vụ, nó chạy toàn bộ vòng lặp lập kế hoạch-hành động-quan sát trong sandbox và trả về URL PR cùng một gói trace. Rubric cho capstone này:

| Trọng số | Tiêu chí | Cách đo lường |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 so với baseline | Bộ khung của bạn so với mini-swe-agent trên 30 tác vụ Python khớp |
| 20 | Độ rõ ràng của kiến trúc | Phân tách plan/act/observe, bề mặt hook, schema công cụ — đối chiếu với bố cục Live-SWE-agent |
| 20 | An toàn | Kiểm tra thoát sandbox, nhắc quyền, rào chắn lệnh phá hủy vượt qua red-team |
| 20 | Khả năng quan sát | Độ đầy đủ của trace (100% lệnh gọi công cụ được ghi lại), kế toán token mỗi lượt |
| 15 | UX nhà phát triển | Khởi động lạnh < 2s, khôi phục khi treo tiếp tục kế hoạch, Ctrl-C hủy giữa chừng sạch sẽ |
| **100** | | |

## Bài tập

1. Thay đổi model hỗ trợ từ Claude Sonnet 4.7 sang Qwen3-Coder-30B chạy trên vLLM. So sánh pass@1 và chi phí mỗi tác vụ. Báo cáo nơi model mở hoạt động kém hiệu quả.

2. Thêm một sub-agent `reviewer` đọc diff trước khi đăng PR và có thể yêu cầu vòng lặp sửa đổi. Đo lường xem các đánh giá dương tính giả có làm giảm tỷ lệ pass SWE-bench xuống dưới baseline đơn agent hay không (gợi ý: thường là có).

3. Stress-test sandbox: viết một tác vụ cố gắng `curl` một URL bên ngoài và một tác vụ ghi bên ngoài worktree. Xác nhận cả hai đều bị chặn bởi hook PreToolUse. Ghi log các nỗ lực này.

4. Triển khai tóm tắt `PreCompact` với một model nhỏ hơn (Haiku 4.5). Đo lường mức độ trung thực của kế hoạch bị mất đi ở mức nén 3x.

5. Thay thế transport MCP StreamableHTTP bằng stdio. Benchmark độ trễ khởi động lạnh và mỗi lần gọi. Chọn phương án tối ưu cho việc sử dụng cục bộ.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Harness | "Vòng lặp agent" | Mã bao quanh model điều phối công cụ, duy trì trạng thái kế hoạch và thực thi ngân sách |
| Hook | "Trình lắng nghe sự kiện agent" | Tập lệnh do người dùng viết chạy trên một trong tám sự kiện vòng đời bởi bộ khung |
| Worktree | "Git sandbox" | Một bản checkout git liên kết tại đường dẫn riêng; có thể xóa mà không ảnh hưởng clone chính |
| TodoWrite | "Trạng thái kế hoạch" | Danh sách các mục pending/in_progress/done có kiểu mà model viết lại mỗi lượt |
| StreamableHTTP | "MCP transport" | Bản sửa đổi MCP 2026: kết nối HTTP tồn tại lâu với streaming hai chiều; thay thế SSE |
| Token ceiling | "Ngân sách ngữ cảnh" | Giới hạn token đầu vào+đầu ra mỗi lượt hoặc mỗi phiên; kích hoạt nén hoặc chấm dứt |
| pass@1 | "Tỷ lệ pass lần đầu" | Tỷ lệ các tác vụ SWE-bench được giải quyết ngay lần chạy đầu tiên mà không cần thử lại |

## Đọc thêm

- [Tài liệu Claude Code](https://docs.anthropic.com/en/docs/claude-code) — bộ khung tham chiếu từ Anthropic
- [Cursor 3 changelog](https://cursor.com/changelog) — ghi chú sản phẩm Agent Tabs và Composer 2
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — baseline tối thiểu để so sánh bộ khung SWE-bench
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 79.2% SWE-bench Verified với Opus 4.5
- [OpenCode](https://opencode.ai) — bộ khung mở, 112k stars
- [Bảng xếp hạng SWE-bench Pro](https://www.swebench.com) — đánh giá mà capstone này nhắm tới
- [Lộ trình Model Context Protocol 2026](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, metadata khả năng
- [Quy ước ngữ nghĩa OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — schema span cho lệnh gọi công cụ và sử dụng token