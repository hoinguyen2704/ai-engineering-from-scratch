# Capstone 16 — GitHub Issue-to-PR Autonomous Agent

> Gán nhãn issue, nhận PR — đây là hình thái sản phẩm năm 2026 cho các tác nhân (agent) lập trình tự động: chạy agent trong một cloud sandbox, xác minh các bài kiểm thử (test) đã vượt qua, và tạo một PR sẵn sàng để review kèm theo lý do thực hiện. AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud, và Google Jules đều đang triển khai mô hình này. Những phần khó nhất bao gồm: tự động tái tạo môi trường build của repo, ngăn chặn rò rỉ thông tin xác thực (credential), thực thi ngân sách theo từng repo, và đảm bảo agent không thể force-push. Capstone này xây dựng phiên bản tự lưu trữ (self-hosted) và so sánh về chi phí cũng như tỷ lệ thành công với các giải pháp thay thế được lưu trữ (hosted).

**Type:** Capstone
**Languages:** Python (agent), TypeScript (GitHub App), YAML (Actions)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P11 · P13 · P14 · P15 · P17
**Time:** 30 giờ

## Vấn đề

Async cloud coding agent là một danh mục sản phẩm riêng biệt so với các tác nhân lập trình tương tác (capstone 01). UX của nó là một nhãn (label) trên GitHub. Bạn gán nhãn cho một issue `@agent fix this`, một worker sẽ khởi chạy trong cloud sandbox, clone repo, chạy các bài test, chỉnh sửa file, xác minh, và mở một PR với lý do của agent trong phần nội dung. Không có vòng lặp tương tác, không có terminal. AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud, Google Jules, và Factory Droids đều đang hội tụ về mô hình này.

Các thách thức kỹ thuật rất cụ thể: tái tạo môi trường (agent phải build repo từ đầu mà không có dev image được cache), các bài test không ổn định (flaky tests - phải được chạy lại hoặc cô lập), phạm vi quyền hạn của thông tin xác thực (một GitHub App với các quyền chi tiết tối thiểu), thực thi ngân sách theo repo mỗi ngày, và chính sách không force-push. Capstone này đo lường tỷ lệ thành công, chi phí, và độ an toàn so với các giải pháp thay thế được lưu trữ.

## Khái niệm

Trigger là một GitHub webhook (nhãn issue hoặc bình luận PR). Một dispatcher sẽ đưa công việc vào hàng đợi (enqueue) tới ECS Fargate hoặc Lambda. Worker kéo repo vào một sandbox Daytona hoặc E2B với một Dockerfile chung được suy luận từ repo (ngôn ngữ, framework). Agent chạy một vòng lặp mini-swe-agent hoặc SWE-agent v2 với Claude Opus 4.7 hoặc GPT-5.4-Codex. Nó lặp lại các bước: đọc code, đề xuất sửa lỗi, áp dụng bản vá (patch), chạy test.

Xác minh là bước kiểm soát (gating). CI đầy đủ phải vượt qua trong sandbox trước khi PR được mở. Độ lệch độ bao phủ (coverage delta) được tính toán; nếu âm vượt quá ngưỡng, PR vẫn được mở nhưng sẽ bị gắn nhãn `needs-review`. Agent đăng lý do thực hiện dưới dạng mô tả PR kèm theo một luồng `@agent` mà người review có thể ping để theo dõi thêm.

Độ an toàn được kiểm soát thông qua hai bề mặt GitHub khác nhau: App cung cấp một token cài đặt ngắn hạn với `workflows: read` và phạm vi nội dung repo/PR hẹp; bảo vệ nhánh (branch protection - không phải quyền của app) thực thi chính sách "không ghi trực tiếp vào `main`" và "không force-push" — app không bao giờ được thêm vào danh sách bỏ qua (bypass list). Quyền truy cập chỉ đọc theo đường dẫn (path-scoped read-only) tới `.github/workflows` không phải là một primitive thực sự của GitHub App, vì vậy danh sách cho phép (allow-list) của agent đối với việc chỉnh sửa file phải được thực thi tại worker. Các giới hạn ngân sách theo repo mỗi ngày được thực thi tại dispatcher (ví dụ: tối đa 5 PR mỗi repo mỗi ngày, $20 mỗi PR).

## Kiến trúc

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## Stack

- Trigger: GitHub App với token chi tiết; bộ nhận webhook qua Lambda hoặc Fly.io
- Worker: ECS Fargate task (hoặc GitHub Actions self-hosted runner)
- Sandbox: Daytona devcontainer hoặc E2B sandbox cho mỗi task
- Vòng lặp Agent: mini-swe-agent baseline hoặc SWE-agent v2 trên Claude Opus 4.7 / GPT-5.4-Codex
- Retrieval: tree-sitter repo-map + ripgrep
- Xác minh: CI đầy đủ trong sandbox + cổng kiểm soát độ lệch độ bao phủ
- Khả năng quan sát (Observability): Langfuse với lưu trữ trace theo từng PR được liên kết từ nội dung PR
- Ngân sách: giới hạn chi phí hàng ngày theo repo; số lượng PR tối đa mỗi repo mỗi ngày

```figure
cf-issue-to-pr
```

## Xây dựng

1. **GitHub App.** Token cài đặt chi tiết: issues read+write, pull_requests write, contents read+write, workflows read. Bảo vệ nhánh (bề mặt duy nhất có thể làm điều này) thực thi "không push trực tiếp vào `main`" và "không force-push"; app không nằm trong danh sách bỏ qua. Worker thực thi "không ghi dưới `.github/workflows`" như một bước kiểm tra allow-list trên diff được đề xuất, vì quyền của GitHub App không được giới hạn theo đường dẫn.

2. **Webhook receiver.** Hàm Lambda chấp nhận webhook nhãn issue / bình luận PR. Lọc theo nhãn `@agent fix this`. Đưa vào hàng đợi SQS.

3. **Dispatcher.** Lấy task từ SQS. Thực thi ngân sách theo repo mỗi ngày. Khởi chạy một ECS Fargate task với URL repo, nội dung issue, và một sandbox Daytona mới.

4. **Suy luận môi trường.** Phát hiện ngôn ngữ (Python, Node, Go, Rust) và trình quản lý gói (uv, pnpm, go mod, cargo). Tạo Dockerfile ngay lập tức nếu chưa có.

5. **Vòng lặp Agent.** mini-swe-agent hoặc SWE-agent v2 với Claude Opus 4.7. Công cụ: ripgrep, tree-sitter repo-map, read_file, edit_file, run_tests, git. Giới hạn cứng: chi phí $20, thời gian thực 30 phút, 30 lượt agent.

6. **Xác minh.** Sau khi vòng lặp kết thúc, chạy bộ test đầy đủ trong sandbox. Tính toán độ lệch độ bao phủ qua jacoco / coverage.py. Nếu CI đỏ: dừng lại, không mở PR. Nếu độ bao phủ giảm hơn 2%: mở PR với nhãn `needs-review`.

7. **Đăng PR.** Push nhánh agent. Mở PR qua GitHub API với: tiêu đề, lý do, tóm tắt diff, URL trace, chi phí, số lượt.

8. **Vệ sinh thông tin xác thực.** Worker chạy với token cài đặt GitHub App ngắn hạn. Logs được làm sạch các bí mật trước khi lưu trữ.

9. **Đánh giá (Eval).** 30 issue nội bộ được chọn lọc với độ khó khác nhau. Đo lường tỷ lệ thành công, chất lượng PR (kích thước diff, phong cách, độ bao phủ), chi phí, độ trễ. So sánh với Cursor Background Agents và AWS Remote SWE Agents trên cùng các issue đó.

## Sử dụng

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## Triển khai

`outputs/skill-issue-to-pr.md` là sản phẩm bàn giao. Một GitHub App + async cloud worker biến các issue được gắn nhãn thành các PR sẵn sàng để review với chi phí giới hạn và thông tin xác thực được kiểm soát.

| Trọng số | Tiêu chí | Cách đo lường |
|:-:|---|---|
| 25 | Tỷ lệ thành công trên 30 issue | Thành công end-to-end (CI xanh + độ bao phủ OK) |
| 20 | Chất lượng PR | Kích thước diff, độ lệch độ bao phủ, tuân thủ phong cách |
| 20 | Chi phí và độ trễ mỗi issue | $ và thời gian thực mỗi PR |
| 20 | Độ an toàn | Token được kiểm soát, ngân sách theo repo, không force-push, vệ sinh credential |
| 15 | UX người vận hành | Bình luận lý do, khả năng thử lại, theo dõi qua @-mention |
| **100** | | |

## Bài tập

1. Thêm chế độ "fix flaky test": nhãn `@agent stabilize-flake TestX` chạy test 50 lần trong sandbox và đề xuất thay đổi tối thiểu để ổn định nó.

2. So sánh chi phí với Cursor Background Agents trên ba issue chung. Báo cáo công cụ nào thắng ở đâu.

3. Triển khai bảng điều khiển ngân sách: chi phí theo repo mỗi ngày, chi phí theo người dùng. Cảnh báo khi có bất thường.

4. Xây dựng chế độ "dry-run" mở một bản nháp PR mà không chạy CI, để người review có thể kiểm tra kế hoạch với chi phí thấp.

5. Thêm chính sách lưu giữ: các nhánh PR cũ hơn 7 ngày mà không được merge sẽ tự động bị xóa.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| GitHub App | "Định danh bot có phạm vi" | App với quyền chi tiết + token cài đặt ngắn hạn |
| Async cloud agent | "Tác nhân nền" | Worker không tương tác chạy trong cloud sandbox, không phải terminal |
| Environment inference | "Tổng hợp Dockerfile" | Phát hiện ngôn ngữ + trình quản lý gói, tạo Dockerfile nếu thiếu |
| Verification | "CI trong sandbox" | Chạy bộ test đầy đủ bên trong worker trước khi mở PR |
| Coverage delta | "Bảo toàn độ bao phủ" | Thay đổi % độ bao phủ test từ nhánh gốc sang nhánh agent |
| Per-repo budget | "Giới hạn hàng ngày" | Giới hạn $ và số lượng PR được thực thi tại dispatcher |
| Rationale | "Giải thích nội dung PR" | Tóm tắt của agent về những gì đã thay đổi và tại sao; bắt buộc trong nội dung PR |

## Đọc thêm

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — tài liệu tham khảo chuẩn về async cloud agent
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — tài liệu tham khảo CLI
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — giải pháp thương mại thay thế
- [OpenAI Codex (cloud)](https://openai.com/codex) — đối thủ cạnh tranh được lưu trữ
- [Google Jules](https://jules.google) — phiên bản được lưu trữ của Google
- [Factory Droids](https://www.factory.ai) — tài liệu tham khảo thương mại thay thế
- [Tài liệu GitHub App](https://docs.github.com/en/apps) — định danh bot có phạm vi
- [Daytona cloud sandboxes](https://daytona.io) — sandbox tham khảo