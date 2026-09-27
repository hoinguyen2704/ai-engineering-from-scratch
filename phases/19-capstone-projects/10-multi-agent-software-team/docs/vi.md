# Capstone 10 — Đội ngũ Kỹ thuật Phần mềm Đa tác nhân (Multi-Agent)

> Hình thái của một đội ngũ kỹ thuật đa tác nhân vào năm 2026 đã dần định hình: một kiến trúc sư lập kế hoạch, N lập trình viên làm việc trên các worktree song song, một người đánh giá (reviewer) kiểm soát chất lượng, và một người kiểm thử (tester) xác thực. Kiến trúc "factory" của SWE-AF, kỹ thuật prompting dựa trên vai trò của MetaGPT, đồ thị tác nhân có kiểu (typed actor graph) của AutoGen 0.4, Devin của Cognition, và Droids của Factory đều độc lập đi đến cùng một mô hình này. Các worktree song song chuyển đổi thời gian thực (wall-clock) thành thông lượng (throughput). Trạng thái chia sẻ và các giao thức bàn giao (handoff) trở thành bề mặt dễ xảy ra lỗi. Capstone này yêu cầu bạn xây dựng đội ngũ, đánh giá trên SWE-bench Pro, và báo cáo những điểm bàn giao nào bị lỗi và tần suất xảy ra.

**Type:** Capstone
**Languages:** Python / TypeScript (tác nhân), Shell (các script cho worktree)
**Prerequisites:** Phase 11 (Kỹ thuật LLM), Phase 13 (Công cụ), Phase 14 (Tác nhân), Phase 15 (Tự hành), Phase 16 (Đa tác nhân), Phase 17 (Cơ sở hạ tầng)
**Phases exercised:** P11 · P13 · P14 · P15 · P16 · P17
**Time:** 40 giờ

## Vấn đề

Việc lập trình bằng tác nhân đơn lẻ (single-agent) gặp giới hạn với các tác vụ lớn. Không phải vì tác nhân đó yếu, mà vì ngữ cảnh 200k-token không thể chứa đồng thời kế hoạch kiến trúc, bốn phần codebase song song, bình luận của người đánh giá và kết quả kiểm thử. Các factory đa tác nhân chia nhỏ vấn đề: kiến trúc sư sở hữu kế hoạch, các lập trình viên thực hiện trong các worktree song song, người đánh giá kiểm soát, và người kiểm thử xác thực. Kiến trúc "factory" của SWE-AF, các vai trò của MetaGPT, đồ thị tác nhân có kiểu của AutoGen — cả ba cách tiếp cận đều mô tả cùng một hình thái.

Bề mặt dễ xảy ra lỗi chính là khâu bàn giao. Kiến trúc sư lập kế hoạch mà các lập trình viên không thể thực hiện. Các lập trình viên tạo ra các diff xung đột. Người đánh giá phê duyệt một bản sửa lỗi bị ảo giác. Người kiểm thử chạy kiểm thử trong khi lập trình viên vẫn đang viết code. Bạn sẽ xây dựng một trong những đội ngũ này, chạy trên 50 vấn đề của SWE-bench Pro, theo dõi mọi khâu bàn giao và xuất bản báo cáo phân tích sau sự cố (post-mortem).

## Khái niệm

Các vai trò là những tác nhân có kiểu (typed agents). **Kiến trúc sư** (Claude Opus 4.7) đọc vấn đề, viết kế hoạch và chia nhỏ thành các tác vụ con với các giao diện rõ ràng. **Các lập trình viên** (Claude Sonnet 4.7, N thực thể song song, mỗi thực thể trong một `git worktree` + sandbox Daytona) thực hiện các tác vụ con một cách độc lập. **Người đánh giá** (GPT-5.4) đọc diff đã hợp nhất và phê duyệt hoặc yêu cầu thay đổi cụ thể. **Người kiểm thử** (Gemini 2.5 Pro) chạy bộ kiểm thử trong môi trường cô lập và báo cáo kết quả đạt/không đạt kèm theo các artifact.

Giao tiếp thông qua một bảng tác vụ chia sẻ (dựa trên file hoặc Redis). Mỗi vai trò tiêu thụ các tác vụ mà nó được phép xử lý. Các khâu bàn giao là các thông điệp theo giao thức A2A. Các vấn đề phối hợp: giải quyết xung đột hợp nhất (vai trò điều phối hoặc hợp nhất ba chiều tự động), đồng bộ hóa trạng thái chia sẻ (kế hoạch bị đóng băng khi lập trình viên bắt đầu; việc lập kế hoạch lại là các sự kiện riêng biệt), và kiểm soát của người đánh giá (người đánh giá không thể phê duyệt các thay đổi của chính mình hoặc các thay đổi do chính họ đề xuất).

Sự khuếch đại token là chi phí ẩn. Mỗi ranh giới vai trò đều thêm các prompt tóm tắt và ngữ cảnh bàn giao. Một lượt chạy 40 bước của tác nhân đơn lẻ trở thành 160 lượt tổng cộng qua bốn vai trò. Rubric đánh giá cụ thể cân nhắc hiệu quả token so với baseline tác nhân đơn lẻ vì câu hỏi không phải là "đa tác nhân có hoạt động không" mà là "nó có hiệu quả về chi phí không".

## Kiến trúc

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## Stack

- Điều phối: LangGraph với trạng thái chia sẻ + các sub-graph cho mỗi tác nhân
- Nhắn tin: Giao thức A2A (Google 2025) cho các thông điệp có kiểu giữa các tác nhân
- Mô hình: Opus 4.7 (kiến trúc sư), Sonnet 4.7 (lập trình viên), GPT-5.4 (người đánh giá), Gemini 2.5 Pro (người kiểm thử)
- Cô lập worktree: `git worktree add` cho mỗi lập trình viên + sandbox Daytona
- Điều phối hợp nhất: hợp nhất ba chiều tùy chỉnh + giải quyết xung đột qua LLM
- Đánh giá: SWE-bench Pro (50 vấn đề), các kịch bản SWE-AF, HumanEval++ cho unit test
- Khả năng quan sát: Langfuse với các span gắn thẻ vai trò, kế toán token theo từng tác nhân
- Triển khai: K8s với mỗi vai trò là một Deployment riêng biệt + HPA trên backlog

```figure
ce-team-handoff
```

## Xây dựng

1. **Bảng tác vụ.** JSONL dựa trên file với các thông điệp có kiểu: `plan_request`, `subtask`, `diff_ready`, `review_needed`, `test_needed`, `approved`, `rejected`, `replan_needed`. Các tác nhân đăng ký theo thẻ.

2. **Kiến trúc sư.** Đọc vấn đề GitHub, chạy Opus 4.7 với template kế hoạch yêu cầu các giao diện tác vụ con rõ ràng (các file bị tác động, các hàm public, tác động kiểm thử). Xuất ra một `plan_request` với DAG các tác vụ con.

3. **Các lập trình viên.** N worker song song, mỗi worker nhận một tác vụ con từ bảng. Mỗi worker tạo một nhánh `git worktree add` mới cộng với một sandbox Daytona. Thực hiện tác vụ con. Xuất ra `diff_ready` với patch + các thay đổi kiểm thử.

4. **Điều phối hợp nhất.** Khi tất cả lập trình viên hoàn thành, thực hiện hợp nhất ba chiều N nhánh vào một nhánh staging. Giải quyết xung đột qua LLM chỉ khi có sự chồng chéo ở cấp độ file.

5. **Người đánh giá.** GPT-5.4 đọc diff đã hợp nhất. Không thể phê duyệt các diff do chính mình tạo ra. Xuất ra `approved` (no-op) hoặc `review_feedback` với các yêu cầu thay đổi cụ thể được định tuyến lại cho lập trình viên liên quan.

6. **Người kiểm thử.** Gemini 2.5 Pro chạy bộ kiểm thử trong sandbox sạch. Ghi lại các artifact. Xuất ra `test_passed` hoặc `test_failed` với stacktrace. Các kiểm thử thất bại sẽ quay lại lập trình viên sở hữu tác vụ con đó.

7. **Kế toán bàn giao.** Mỗi thông điệp đi qua ranh giới vai trò đều có một span trong Langfuse với kích thước payload và mô hình được sử dụng. Tính toán sự khuếch đại token theo từng tác vụ con (coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens).

8. **Đánh giá.** Chạy trên 50 vấn đề SWE-bench Pro. So sánh pass@1 và $-trên-mỗi-vấn-đề-được-giải-quyết so với baseline tác nhân đơn lẻ (một Sonnet 4.7 trong một worktree duy nhất).

9. **Phân tích sau sự cố.** Với mỗi vấn đề thất bại, xác định khâu bàn giao bị lỗi (kế hoạch quá mơ hồ, xung đột hợp nhất, người đánh giá phê duyệt sai, người kiểm thử bị lỗi flake). Tạo biểu đồ tần suất lỗi bàn giao.

## Sử dụng

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## Xuất bản

`outputs/skill-multi-agent-team.md` là sản phẩm bàn giao. Với một URL vấn đề và mức độ song song, đội ngũ tạo ra một PR sẵn sàng hợp nhất với kế toán token theo từng vai trò.

| Trọng số | Tiêu chí | Cách đo lường |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | Tập con 50 vấn đề đã khớp, pass@1 |
| 20 | Tăng tốc song song | Thời gian thực so với baseline tác nhân đơn lẻ |
| 20 | Chất lượng đánh giá | Tỷ lệ phê duyệt sai trên probe lỗi được tiêm vào |
| 20 | Hiệu quả token | Tổng token trên mỗi vấn đề được giải quyết so với tác nhân đơn lẻ |
| 15 | Kỹ thuật phối hợp | Giải quyết xung đột hợp nhất, biểu đồ tần suất lỗi bàn giao |
| **100** | | |

## Bài tập

1. Tiêm một lỗi rõ ràng vào diff giữa chừng (thêm `return None` trước phần thân chính). Đo tỷ lệ phê duyệt sai của người đánh giá. Tinh chỉnh prompt của người đánh giá cho đến khi tỷ lệ phê duyệt sai dưới 5%.

2. Giảm xuống còn hai lập trình viên (kiến trúc sư + lập trình viên + người đánh giá + người kiểm thử, lập trình viên chạy hai tác vụ con tuần tự). So sánh thời gian thực và tỷ lệ đạt.

3. Thay thế điều phối viên hợp nhất bằng ràng buộc một người ghi (các tác vụ con chạm vào các tập file rời rạc). Đo lường gánh nặng lập kế hoạch lên kiến trúc sư.

4. Đổi người đánh giá từ GPT-5.4 sang Claude Opus 4.7. Đo tỷ lệ phê duyệt sai và chênh lệch chi phí token.

5. Thêm vai trò thứ năm: người viết tài liệu (Haiku 4.5). Sau khi đánh giá, nó tạo ra một mục changelog. Đo lường xem chất lượng tài liệu có xứng đáng với chi phí token bỏ ra thêm hay không.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Parallel worktree | "Nhánh cô lập" | `git worktree add` tạo ra một cây làm việc mới cho mỗi lập trình viên |
| Task board | "Bus thông điệp chia sẻ" | Lưu trữ file hoặc Redis các thông điệp có kiểu mà các tác nhân đăng ký |
| Handoff | "Ranh giới vai trò" | Bất kỳ thông điệp nào đi từ ngữ cảnh của vai trò này sang vai trò khác |
| Token amplification | "Chi phí đa tác nhân" | Tổng token qua các vai trò / token tác nhân đơn lẻ cho cùng tác vụ |
| A2A protocol | "Tác nhân-đến-tác nhân" | Đặc tả của Google năm 2025 cho các thông điệp có kiểu giữa các tác nhân |
| Merge coordinator | "Người tích hợp" | Thành phần chạy hợp nhất ba chiều và điều phối xung đột |
| False approval | "Ảo giác người đánh giá" | Người đánh giá phê duyệt một diff có lỗi đã biết |

## Đọc thêm

- [Kiến trúc factory SWE-AF](https://github.com/Agent-Field/SWE-AF) — factory đa tác nhân tham chiếu năm 2026
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — framework đa tác nhân dựa trên vai trò
- [AutoGen v0.4](https://github.com/microsoft/autogen) — framework tác nhân có kiểu của Microsoft
- [Cognition AI (Devin)](https://cognition.ai) — sản phẩm tham chiếu
- [Factory Droids](https://www.factory.ai) — sản phẩm tham chiếu thay thế
- [Giao thức Google A2A](https://a2a-protocol.org/latest/) — đặc tả nhắn tin giữa các tác nhân
- [Tài liệu git worktree](https://git-scm.com/docs/git-worktree) — nền tảng cô lập
- [SWE-bench Pro](https://www.swebench.com) — mục tiêu đánh giá