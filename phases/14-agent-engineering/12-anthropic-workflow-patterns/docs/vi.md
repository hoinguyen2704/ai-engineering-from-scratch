# Các Workflow Pattern của Anthropic: Đơn giản hơn Phức tạp

> Schluntz và Zhang (Anthropic, tháng 12 năm 2024) phân biệt giữa workflow (các đường dẫn được xác định trước) và agent (sử dụng công cụ linh hoạt). Năm workflow pattern bao quát hầu hết các trường hợp. Hãy bắt đầu với các API call trực tiếp. Chỉ thêm agent khi các bước không thể dự đoán trước.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu tên năm workflow pattern của Anthropic: prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer.
- Giải thích sự khác biệt giữa agent và workflow cũng như chi phí kỹ thuật của từng loại.
- Xác định khi nào nên chọn workflow thay vì agent (và ngược lại).
- Triển khai cả năm pattern bằng stdlib với một scripted LLM.

## Vấn đề

Các đội ngũ thường tìm đến các framework multi-agent cho những vấn đề chỉ cần một function call đơn giản. Chi phí là có thật: các framework thêm vào những lớp trung gian làm mờ các prompt, che giấu luồng điều khiển (control flow) và dẫn đến sự phức tạp không cần thiết. Bài viết tháng 12 năm 2024 của Schluntz và Zhang là lời phản biện được trích dẫn nhiều nhất trong ngành: hãy bắt đầu đơn giản, chỉ thêm sự phức tạp khi nó thực sự xứng đáng với chi phí bỏ ra.

## Khái niệm

### Workflow vs Agent

- **Workflow.** Các LLM và công cụ được điều phối thông qua các đường dẫn code được xác định trước. Kỹ sư làm chủ đồ thị (graph).
- **Agent.** Các LLM tự điều hướng công cụ và thực hiện các bước của riêng chúng một cách linh hoạt. Mô hình làm chủ đồ thị.

Cả hai đều có vị trí riêng. Workflow rẻ hơn, nhanh hơn và dễ debug hơn. Agent giải quyết các vấn đề mở nhưng khiến các chế độ lỗi (failure modes) khó suy luận hơn.

### Augmented LLM

Nền tảng cho cả năm pattern: một LLM với ba khả năng được tích hợp — search (truy xuất), tools (hành động), memory (lưu trữ). Bất kỳ API call nào cũng có thể sử dụng những khả năng này.

### Năm workflow pattern

1. **Prompt chaining.** Đầu ra của call 1 là đầu vào của call 2. Sử dụng khi tác vụ có sự phân tách tuyến tính rõ ràng. Có thể tùy chọn các cổng lập trình (programmatic gates) giữa các bước.

2. **Routing.** Một LLM phân loại sẽ chọn LLM hoặc công cụ hạ nguồn nào để gọi. Sử dụng khi các đầu vào khác biệt hoàn toàn về loại cần được xử lý khác nhau (hỗ trợ cấp 1 vs hoàn tiền vs lỗi vs bán hàng).

3. **Parallelization.** Chạy N LLM call đồng thời, sau đó tổng hợp kết quả. Hai hình thức: sectioning (các phần khác nhau) và voting (cùng một prompt, chạy N lần, lấy đa số/tổng hợp).

4. **Orchestrator-workers.** Một LLM điều phối (orchestrator) quyết định linh hoạt các worker (cũng là các LLM) nào cần chạy và tổng hợp đầu ra của chúng. Tương tự như agent loop nhưng orchestrator không lặp vô hạn.

5. **Evaluator-optimizer.** Một LLM đề xuất câu trả lời, một LLM khác đánh giá nó. Lặp lại cho đến khi evaluator thông qua. Đây là Self-Refine (Bài 05) được tổng quát hóa.

### Khi nào workflow vượt trội hơn agent

- **Tác vụ có thể dự đoán.** Nếu bạn có thể liệt kê các bước, bạn nên làm vậy.
- **Tác vụ bị giới hạn chi phí.** Workflow có số bước giới hạn; agent có thể mất kiểm soát.
- **Tác vụ bị ràng buộc bởi tuân thủ.** Kiểm toán viên muốn đọc đồ thị, không phải suy luận nó từ các quỹ đạo (trajectories).

### Khi nào agent vượt trội hơn workflow

- **Nghiên cứu mở.** Khi bước tiếp theo phụ thuộc vào những gì bước trước đó trả về.
- **Tác vụ có độ dài biến thiên.** Công việc kéo dài từ vài phút đến vài giờ mà không biết trước số bước.
- **Lĩnh vực mới.** Khi bạn chưa biết workflow đúng là gì — hãy khám phá trước, hệ thống hóa sau.

### Người bạn đồng hành về context engineering

"Effective context engineering for AI agents" (Anthropic 2025) chính thức hóa kỷ luật liên quan: cửa sổ 200k là một ngân sách, không phải là một cái thùng chứa. Những gì cần bao gồm, khi nào cần nén, khi nào để context phát triển. Được đề cập chi tiết trong bài học về nén context ở Phase 14 (bài học 06 trước đây trong chương trình này trước khi đánh số lại).

```figure
workflow-chain
```

## Xây dựng

`code/main.py` triển khai tất cả năm workflow pattern dựa trên một `ScriptedLLM`:

- `prompt_chain(input, steps)` — tuần tự.
- `route(input, classifier, handlers)` — phân loại + điều phối.
- `parallel_vote(prompt, n, aggregator)` — N lần chạy, tổng hợp.
- `orchestrator_workers(task, workers)` — orchestrator chọn worker.
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — lặp cho đến khi đạt yêu cầu.

Chạy nó:

```
python3 code/main.py
```

Mỗi pattern sẽ in ra dấu vết (trace) của nó. Tổng số dòng code mỗi pattern là ~10-15; chi phí của một framework được đo bằng hàng nghìn dòng.

## Sử dụng

- Sử dụng API call trực tiếp cho hầu hết các tác vụ.
- Chỉ dùng framework khi pattern thực sự cần trạng thái bền vững (LangGraph), tính đồng thời theo mô hình actor (AutoGen v0.4), hoặc tạo khuôn mẫu vai trò (CrewAI).
- Tìm đến Claude Agent SDK khi bạn muốn hình dạng của Claude Code harness mà không cần phải xây dựng lại từ đầu.

## Triển khai

`outputs/skill-workflow-picker.md` chọn pattern phù hợp cho một mô tả tác vụ nhất định, bao gồm lý do quyết định và lộ trình refactor sang agent nếu workflow không đáp ứng đủ.

## Bài tập

1. Triển khai routing với ngưỡng tin cậy (confidence threshold). Dưới ngưỡng -> leo thang lên con người. Ngưỡng này nên đặt ở đâu cho trường hợp hỗ trợ khách hàng cấp 1?
2. Thêm timeout vào `parallel_vote`. Điều gì xảy ra khi một call bị treo? Làm thế nào để tổng hợp khi thiếu các kết quả bầu chọn?
3. Biến `evaluator_optimizer` thành một bandit: giữ lại 2 kết quả đầu ra tốt nhất qua các lần lặp để một kết quả tốt muộn không bị ghi đè bởi một kết quả xấu sau đó.
4. Kết hợp prompt chaining với routing: một router chọn một trong ba chuỗi. Đo lường chi phí token so với một giải pháp big-prompt duy nhất.
5. Chọn một trong các tính năng sản phẩm của bạn. Vẽ đồ thị workflow. Đếm số bước. Liệu một agent có thực sự tốt hơn ở đây không?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Workflow | "Luồng xác định trước" | Đồ thị các LLM và tool call do kỹ sư sở hữu |
| Agent | "AI tự trị" | Đồ thị do mô hình sở hữu; điều hướng công cụ linh hoạt |
| Augmented LLM | "LLM với công cụ" | LLM + search + tools + memory; đơn vị nguyên tử |
| Prompt chaining | "Các call tuần tự" | Đầu ra của call N là đầu vào của call N+1 |
| Routing | "Điều phối phân loại" | Chọn chuỗi/mô hình xử lý đầu vào |
| Parallelization | "Fan out" | N call đồng thời; tổng hợp bằng sectioning hoặc voting |
| Orchestrator-workers | "Agent điều phối" | LLM điều phối chọn các LLM chuyên gia một cách linh hoạt |
| Evaluator-optimizer | "Người đề xuất + giám khảo" | Lặp cho đến khi evaluator thông qua; Self-Refine tổng quát |

## Đọc thêm

- [Anthropic, Building Effective Agents (Tháng 12 năm 2024)](https://www.anthropic.com/research/building-effective-agents) — năm workflow pattern
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — kỷ luật đồng hành
- [Tổng quan về LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — khi nào các đồ thị có trạng thái xứng đáng với chi phí
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — pattern orchestrator-workers, được thương mại hóa