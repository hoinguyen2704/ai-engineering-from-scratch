# Long-Running Background Agents: Durable Execution

> Các agent dài hạn trong môi trường production không chạy trong `while True`. Mọi lời gọi LLM đều trở thành một activity với cơ chế checkpoint, retry và replay. Việc tích hợp OpenAI Agents SDK của Temporal đã đạt trạng thái GA vào tháng 3 năm 2026. Claude Code Routines (của Anthropic) thực thi các lời gọi Claude Code theo lịch trình mà không cần một tiến trình cục bộ chạy thường trực. Các phiên làm việc (session) sẽ tạm dừng khi chờ phản hồi từ con người, tồn tại qua các lần deploy và tiếp tục từ checkpoint mới nhất được định danh bởi `thread_id`. Đằng sau sự tiện dụng mới này là một mô hình cũ — workflow orchestration — với một đầu vào mới: các lời gọi LLM được coi là các activity không tất định (non-deterministic) và phải được replay một cách tất định khi khôi phục.

**Type:** Learn
**Languages:** Python (stdlib, minimal durable-execution state machine)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~60 minutes

## Vấn đề

Hãy xem xét một agent chạy trong bốn giờ. Nó gọi ba công cụ, nhắc người dùng hai lần và thực hiện bốn mươi lời gọi LLM. Giữa chừng, máy chủ đang chạy nó bị khởi động lại. Điều gì sẽ xảy ra?

- Trong một vòng lặp `while True` ngây thơ: mọi thứ đều mất. Quá trình chạy bắt đầu lại từ đầu. Ba lời gọi công cụ (với các tác động thực tế) được thực thi lại. Người dùng bị nhắc lại những thứ họ đã phê duyệt. Bốn mươi lời gọi LLM bị tính phí lại.
- Với durable execution: quá trình chạy tiếp tục từ checkpoint gần nhất. Các activity đã hoàn thành không bị thực thi lại; kết quả của chúng được replay từ nhật ký bền vững (durable log). Người dùng không phải phê duyệt lại những gì họ đã duyệt. Các lời gọi LLM đã thực hiện không bị tính phí lại.

Đây là cùng một mô hình mà các workflow engine đã triển khai trong một thập kỷ qua (Temporal, Cadence, Uber's Cherami). Điểm mới ở đây là các lời gọi LLM giờ đây được coi là một loại activity — không tất định, tốn kém, có tác động phụ — và chúng khớp hoàn hảo với mô hình này.

Chủ đề xuyên suốt của bài học: độ tin cậy của các tác vụ dài hạn sẽ suy giảm (METR quan sát thấy "sự suy giảm 35 phút" — tỷ lệ thành công giảm theo hàm bậc hai so với thời gian thực hiện). Durable execution cho phép các tiến trình chạy lâu hơn mức mà hồ sơ độ tin cậy hỗ trợ, đây là một cách mới để thất bại an toàn nếu thiết kế đúng và thất bại không an toàn nếu thiết kế sai.

## Khái niệm

### Activities, workflows và replay

- **Workflow**: mã orchestration tất định. Định nghĩa trình tự các activity, các nhánh, các điểm chờ. Phải là tất định để có thể replay từ nhật ký sự kiện mà không gây ra sự khác biệt bất ngờ.
- **Activity**: một đơn vị công việc không tất định, có khả năng thất bại. Lời gọi LLM, lời gọi công cụ, ghi file, yêu cầu HTTP. Mỗi activity được ghi lại cùng với đầu vào và (khi hoàn thành) đầu ra của nó.
- **Event log**: kho lưu trữ bền vững. Mọi sự kiện bắt đầu, hoàn thành, thất bại, thử lại của activity và mọi quyết định của workflow đều được ghi lại.
- **Replay**: khi khôi phục, mã workflow chạy lại từ đầu; mọi activity đã hoàn thành sẽ trả về kết quả đã ghi mà không cần thực thi lại. Chỉ những activity chưa hoàn thành mới thực sự được chạy.

Cấu trúc này tương tự như cách React render lại dựa trên virtual DOM, hoặc Git xây dựng lại cây làm việc từ các commit. Tính tất định trong orchestrator là yếu tố giúp cho durable execution trở nên hiệu quả.

### Tại sao các lời gọi LLM phù hợp với mô hình này

Các lời gọi LLM có đặc điểm:
- Không tất định (temperature > 0; ngay cả temperature 0 cũng có thể sai lệch giữa các phiên bản model).
- Tốn kém (tiền bạc và độ trễ).
- Có khả năng thất bại (giới hạn tốc độ, timeout).
- Có tác động phụ (nếu chúng gọi các công cụ).

Đây chính xác là hồ sơ của một activity. Việc bao bọc mỗi lời gọi LLM thành một activity giúp bạn có được cơ chế retry với exponential backoff, checkpoint qua các lần khởi động lại và một dấu vết (trace) có thể replay để debug.

### Checkpoint được định danh bởi `thread_id`

LangGraph, Microsoft Agent Framework, Cloudflare Durable Objects và Claude Code Routines đều hội tụ về cùng một dạng API: một `thread_id` (hoặc tương đương) định danh phiên làm việc; mỗi sự thay đổi trạng thái được lưu vào backend (mặc định là PostgreSQL, SQLite cho dev, Redis cho cache); khi resume, hệ thống sẽ đọc checkpoint mới nhất.

Việc lựa chọn backend rất quan trọng:

- **PostgreSQL**: bền vững, có thể truy vấn, tồn tại qua các lần deploy. Mặc định cho LangGraph.
- **SQLite**: chỉ dùng cho local-dev; mất dữ liệu khi chuyển host.
- **Redis**: nhanh nhưng tạm thời trừ khi cấu hình AOF/snapshot.
- **Cloudflare Durable Objects**: phân tán một cách minh bạch; phạm vi theo một khóa duy nhất; tồn tại từ vài giờ đến vài tuần.

### Human-input là một trạng thái hạng nhất

Mô hình Propose-then-commit (Bài 15) yêu cầu một trạng thái "đang chờ con người" bền vững. Workflow tạm dừng, hàng đợi bên ngoài giữ yêu cầu đang chờ xử lý, và một sự phê duyệt sẽ resume workflow chính xác từ điểm đó. Nếu không có tính bền vững, đây chỉ là nỗ lực tốt nhất (best-effort); với nó, một sự phê duyệt qua đêm sẽ được ghi nhận và workflow sẽ tiếp tục vào buổi sáng.

### Sự suy giảm 35 phút

METR quan sát thấy rằng mọi lớp agent được đo lường đều cho thấy độ tin cậy suy giảm sau khoảng ~35 phút hoạt động liên tục. Việc tăng gấp đôi thời gian tác vụ làm tăng tỷ lệ thất bại lên khoảng bốn lần. Durable execution không khắc phục điều này; nó cho phép bạn chạy lâu hơn mức mà hồ sơ độ tin cậy hỗ trợ. Mô hình an toàn là kết hợp tính bền vững với các checkpoint yêu cầu HITL (Human-in-the-loop) mới khi quay lại, và với các công tắc ngắt ngân sách (Bài 13) để giới hạn tổng chi phí tính toán bất kể thời gian thực.

### Khi nào durable execution là câu trả lời sai

- Các tác vụ chạy ngắn hơn vài phút mà không có sự can thiệp của con người. Chi phí vận hành > lợi ích.
- Truy xuất thông tin chỉ đọc (read-only).
- Các tác vụ mà tính đúng đắn yêu cầu thực hiện end-to-end trong một context window (một số tác vụ suy luận; một số tác vụ tạo nội dung one-shot).

```figure
memory-consolidation
```

## Sử dụng

`code/main.py` triển khai một engine durable-execution tối giản trong stdlib Python. Nó hỗ trợ:

- Decorator `@activity` ghi lại đầu vào và đầu ra vào một JSON event log.
- Một hàm workflow sắp xếp các activity.
- Một hàm `run_or_replay(workflow, event_log)` replay các activity đã hoàn thành mà không thực thi lại chúng.

Driver mô phỏng một workflow gồm ba activity, crash giữa chừng và hiển thị (a) cơ chế retry ngây thơ thực thi lại mọi thứ so với (b) cơ chế replay chỉ chạy activity còn thiếu.

## Triển khai

`outputs/skill-durable-execution-review.md` xem xét một bản triển khai agent dài hạn được đề xuất để đảm bảo hình thái durable-execution chính xác: các activity, tính tất định, checkpoint backend, trạng thái human-input và chính sách HITL-on-resume.

## Bài tập

1. Chạy `code/main.py`. Quan sát sự khác biệt về số lần thực thi activity giữa retry ngây thơ và replay. Thay đổi điểm crash và cho thấy số lần replay thay đổi tương ứng.

2. Chuyển đổi engine đồ chơi sang sử dụng `thread_id` một cách rõ ràng. Mô phỏng hai phiên làm việc đồng thời chia sẻ cùng một engine và xác nhận rằng nhật ký sự kiện của chúng không bị xung đột.

3. Chọn một activity trong engine đồ chơi. Giới thiệu một yếu tố không tất định (ví dụ: dấu thời gian wall-clock bên trong một quyết định workflow). Chứng minh sự phân kỳ khi replay. Giải thích cách các engine thực tế xử lý vấn đề này (đăng ký tác động phụ, các API `Workflow.now()`).

4. Đọc bài viết "Runtime behind production deep agents" của LangChain. Liệt kê mọi trạng thái mà runtime lưu trữ và nêu tên chế độ thất bại mà mỗi trạng thái đó giải quyết.

5. Thiết kế chính sách checkpoint cho một tác vụ lập trình tự động kéo dài 6 giờ. Bạn sẽ checkpoint ở đâu? Việc resume-on-crash trông như thế nào? Điều gì yêu cầu HITL mới?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| Workflow | "Script của agent" | Mã orchestration tất định; có thể replay từ event log |
| Activity | "Một bước" | Đơn vị không tất định (lời gọi LLM, lời gọi công cụ); được ghi lại trước và sau khi chạy |
| Event log | "Kho lưu trữ" | Bản ghi bền vững của mọi sự thay đổi trạng thái |
| Replay | "Tiếp tục" | Chạy lại workflow; các activity đã hoàn thành trả về kết quả đã ghi mà không thực thi lại |
| Checkpoint | "Điểm lưu" | Trạng thái được lưu trữ định danh bởi thread_id; ưu tiên trạng thái mới nhất khi resume |
| thread_id | "Khóa phiên" | Định danh xác định phạm vi trạng thái bền vững |
| 35-minute degradation | "Suy giảm độ tin cậy" | METR: tỷ lệ thành công giảm theo hàm bậc hai so với thời gian thực hiện |
| Non-determinism | "Sai lệch khi replay" | Thời gian thực, ngẫu nhiên, đầu ra LLM; phải được đăng ký như một tác động phụ |

## Đọc thêm

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — ngân sách, lượt chạy và ngữ nghĩa resume.
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — hình thái RequestInfoEvent.
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — các yêu cầu runtime cụ thể.
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — hình thái activity cho các lời gọi LLM.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — tài liệu tham khảo về sự suy giảm 35 phút.