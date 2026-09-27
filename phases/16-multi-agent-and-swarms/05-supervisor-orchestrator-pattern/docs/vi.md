# Supervisor / Orchestrator-Worker Pattern

> Một tác nhân chính (lead agent) lập kế hoạch và ủy quyền; các tác nhân chuyên biệt (workers) thực thi trong các ngữ cảnh song song và báo cáo kết quả. Đây là mô hình đứng sau hệ thống Research của Anthropic (Claude Opus 4 làm lead, Sonnet 4 làm subagent), đạt mức tăng trưởng +90,2% so với Opus 4 đơn lẻ trong các bài đánh giá nghiên cứu nội bộ. Bài viết kỹ thuật của Anthropic báo cáo rằng 80% sự biến thiên trên BrowseComp được giải thích chỉ bằng lượng token sử dụng — mô hình đa tác nhân chiến thắng chủ yếu vì mỗi subagent có một cửa sổ ngữ cảnh (context window) mới. Bài học này xây dựng mô hình supervisor từ các nguyên lý cơ bản và bao quát các bài học kỹ thuật năm 2026 từ các triển khai thực tế.

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 phút

## Vấn đề

Nghiên cứu là tác vụ điển hình mà các hệ thống đơn tác nhân thường thất bại. Bạn hỏi "điều gì đã thay đổi trong các hệ thống đa tác nhân giữa năm 2023 và 2026?" Một tác nhân đơn lẻ đọc năm bài báo một cách tuần tự, làm đầy một nửa ngữ cảnh của nó với văn bản của chúng, và sau đó phải suy luận về tất cả chúng cùng lúc. Nó quên mất bài báo đầu tiên khi đọc đến bài thứ năm. Nó không thể thực hiện song song.

Mô hình supervisor giải quyết vấn đề này: một lead agent lập kế hoạch tìm kiếm, ủy quyền từng câu hỏi phụ cho một worker, và tổng hợp kết quả. Mỗi worker có cửa sổ 200k-token riêng cho một câu hỏi hẹp. Lead không bao giờ nhìn thấy các bài báo gốc — chỉ thấy các bản tóm tắt từ worker.

Hệ thống Research thực tế của Anthropic báo cáo mức tăng +90,2% trong các bài đánh giá nghiên cứu nội bộ so với một Opus 4 đơn lẻ. Bài viết tương tự lưu ý rằng 80% sự biến thiên của BrowseComp được giải thích chỉ bằng *lượng token sử dụng*. Ngữ cảnh mới cho mỗi subagent là cơ chế chính.

## Khái niệm

### Mô hình này

```
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

Lead không bao giờ đọc tài liệu gốc. Các worker không bao giờ nhìn thấy công việc của nhau cho đến khi lead tổng hợp. Mỗi mũi tên là một sự bàn giao với một artifact (sản phẩm) hẹp.

### Tại sao nó hiệu quả

Ba cơ chế:

1. **Ngữ cảnh mới cho mỗi subagent.** Một worker khám phá "di sản FIPA-ACL" không phải mang theo 40k token mà lead đã dùng để lập kế hoạch. Nó nhận được cửa sổ 200k cho một câu hỏi duy nhất.
2. **Chuyên môn hóa thông qua prompt.** Prompt của lead là "phân rã và tổng hợp", không phải "nghiên cứu". Prompt của mỗi worker rất hẹp: "tìm những gì đã thay đổi trong X". Các prompt tập trung tạo ra các đầu ra tập trung.
3. **Tính song song.** Các worker chạy đồng thời. Thời gian thực tế (wall-clock time) xấp xỉ `max(worker_times) + plan + synthesis`, thay vì `sum(worker_times)`.

### Các bài học kỹ thuật (Anthropic 2025)

Bài viết của Anthropic liệt kê một số bài học thực tế vẫn còn phù hợp trong năm 2026:

- **Quy mô nỗ lực theo độ phức tạp của truy vấn.** Truy vấn đơn giản: một tác nhân, 3-10 lần gọi công cụ. Truy vấn phức tạp: 10+ tác nhân. Lead phải ước tính điều này, không phải người gọi.
- **Rộng rồi hẹp.** Phân rã thành các câu hỏi phụ rộng trước, sau đó tạo thêm nhiều worker cho mỗi câu hỏi phụ nếu câu trả lời cần độ sâu.
- **Triển khai Rainbow (Rainbow deployments).** Các tác nhân chạy dài hạn và có trạng thái. Blue-green truyền thống không hoạt động. Anthropic sử dụng rainbow: triển khai dần dần các phiên bản mới trong khi các phiên bản cũ dần cạn kiệt.
- **Lượng token chiếm ưu thế.** Đa tác nhân tốn khoảng ~15× lượng token so với đơn tác nhân. Chỉ chạy nó khi giá trị của tác vụ xứng đáng với chi phí.

### Bước ngoặt hướng đồ thị (Graph-native)

LangGraph ban đầu xuất xưởng một thư viện `langgraph-supervisor` với một helper `create_supervisor` cấp cao. Năm 2025, LangChain chuyển khuyến nghị sang việc triển khai mô hình supervisor thông qua gọi công cụ (tool-calling) trực tiếp, vì các lệnh gọi công cụ cho phép kiểm soát tốt hơn *những gì supervisor nhìn thấy* (kỹ thuật ngữ cảnh). Thư viện vẫn hoạt động; tài liệu hiện khuyến nghị hình thức gọi công cụ.

### Các chế độ thất bại

- **Lead ảo tưởng về kế hoạch.** Nếu lead tạo ra các câu hỏi phụ không phân rã được câu hỏi thực tế, các worker sẽ thực hiện nghiên cứu chính xác trên mục tiêu sai.
- **Worker khám phá quá mức.** Nếu không có ranh giới phạm vi rõ ràng, các worker sẽ đi chệch khỏi câu hỏi phụ được giao và làm ô nhiễm bước tổng hợp.
- **Xung đột tổng hợp.** Hai worker trả về các sự kiện mâu thuẫn. Lead phải yêu cầu lại (thêm một vòng) hoặc ghi chú rõ ràng về sự bất đồng. Việc âm thầm chọn một bên là thất bại tồi tệ nhất: người dùng không bao giờ biết sự bất đồng đã xảy ra.

### Khi nào supervisor không phù hợp

- **Tác vụ tuần tự.** Nếu bước 2 thực sự cần đầu ra của bước 1, tính song song không mang lại lợi ích gì. Hãy sử dụng pipeline (CrewAI Sequential, LangGraph linear graph).
- **Truy vấn đơn giản.** Tác nhân đơn lẻ xử lý chúng nhanh hơn và rẻ hơn. Hãy sử dụng bước kiểm tra "quy mô nỗ lực" của lead trước khi tạo worker.
- **Tính xác định nghiêm ngặt.** Supervisor sử dụng việc ủy quyền do LLM lựa chọn. Các đồ thị tĩnh (static graphs) tốt hơn khi việc kiểm toán/phát lại quan trọng hơn khả năng thích ứng.

```figure
supervisor-hierarchy
```

## Xây dựng

`code/main.py` triển khai một supervisor gồm ba worker song song sử dụng `threading`. Lead phân rã một truy vấn thành các câu hỏi phụ, các worker chạy đồng thời trên mỗi câu hỏi phụ, và lead tổng hợp. Không có LLM thực sự — các worker được viết kịch bản để mô phỏng việc tìm nạp và tóm tắt.

Cấu trúc chính:

- `Lead.plan(query)` chia một truy vấn thành 3 câu hỏi phụ.
- `Worker.run(sub_q)` trả về một bản tóm tắt giả (có thể là bất kỳ tác nhân sử dụng công cụ nào trong thực tế).
- `Lead.run(query)` khởi chạy các worker trong các luồng, kết nối và tổng hợp.

Chạy:

```
python3 code/main.py
```

Đầu ra hiển thị kế hoạch, các dấu vết worker song song với dấu thời gian bắt đầu/kết thúc, và bản tổng hợp cuối cùng. Bạn có thể thấy lợi ích về thời gian thực tế: ba worker 0,3 giây chạy trong ~0,35 giây, thay vì 0,9 giây.

## Sử dụng

`outputs/skill-supervisor-designer.md` nhận một truy vấn người dùng và tạo ra một thiết kế mô hình supervisor: system prompt cho lead, vai trò của worker, quy tắc phân rã câu hỏi phụ, và mẫu tổng hợp. Hãy sử dụng điều này trước khi xây dựng một hệ thống tác nhân kiểu nghiên cứu mới.

## Triển khai

Danh sách kiểm tra trước khi triển khai mô hình supervisor:

- **Ghép nối mô hình.** Lead sử dụng mô hình cấp suy luận (hạng Opus, hạng `o3`). Worker sử dụng mô hình nhanh hơn, rẻ hơn (Sonnet, `o4-mini`).
- **Thời gian chờ của worker.** Bất kỳ worker nào vượt quá 2× thời gian chạy trung bình sẽ bị ngắt; lead sẽ khởi tạo lại với phạm vi hẹp hơn hoặc tiếp tục mà không có nó.
- **Giới hạn token mỗi worker.** Giới hạn cứng (ví dụ: 10× đầu vào tổng hợp dự kiến) ngăn chặn một worker chạy quá đà làm hỏng ngân sách.
- **Khả năng quan sát (Observability).** Theo dõi kế hoạch của lead, các lệnh gọi công cụ của từng worker, và quá trình tổng hợp. Đây là cơ sở cho bất kỳ việc gỡ lỗi hậu kỳ nào.
- **Triển khai Rainbow.** Các tác nhân chạy dài hạn có trạng thái cần sự chuyển đổi phiên bản dần dần, không phải thay thế nóng.

## Bài tập

1. Chạy `code/main.py`, sau đó sửa đổi lead để tạo 5 worker thay vì 3. Quan sát hiệu ứng thời gian thực tế. Tại số lượng worker nào thì chi phí tạo (spawn overhead) vượt quá mức tiết kiệm được từ tính song song trong bản demo này?
2. Triển khai thời gian chờ cho worker: ngắt bất kỳ worker nào chạy lâu hơn 0,5 giây và để lead tổng hợp các kết quả còn lại. Bạn cần khả năng quan sát nào để biết một worker đã bị cắt?
3. Thêm bước phát hiện xung đột vào quá trình tổng hợp của lead: nếu hai worker trả về các câu trả lời mâu thuẫn, lead ghi chú sự bất đồng thay vì chọn một bên. Làm thế nào để bạn phát hiện mâu thuẫn mà không cần gọi LLM?
4. Đọc bài viết kỹ thuật về hệ thống Research của Anthropic. Liệt kê ba thực tiễn mà bản demo này cần áp dụng để chạy trong môi trường thực tế.
5. So sánh `create_supervisor` của LangGraph (cũ) với khuyến nghị gọi công cụ mới. Cái nào cho bạn khả năng kiểm soát tốt hơn đối với những gì supervisor nhìn thấy? Tại sao Anthropic chỉ truyền các câu trả lời phụ và không truyền ngữ cảnh worker thô vào quá trình tổng hợp?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Supervisor | "Lead agent" | Một tác nhân điều phối lập kế hoạch, ủy quyền và tổng hợp. Bản thân nó không thực hiện công việc. |
| Worker | "Subagent" | Một tác nhân tập trung được supervisor gọi với phạm vi hẹp và cửa sổ ngữ cảnh riêng. |
| Orchestrator-worker | "Supervisor pattern" | Cùng một thứ, tên gọi khác nhau. Tài liệu năm 2026 sử dụng cả hai. |
| Fresh context | "Clean window" | Ngữ cảnh của worker bắt đầu từ system prompt và câu hỏi được giao của nó, không phải lịch sử của lead. |
| Rainbow deployment | "Gradual rollout" | Các tác nhân có trạng thái chạy dài hạn cần quy trình rút cạn và thay thế theo phiên bản, không phải blue-green. |
| Token dominance | "Context is the variable" | 80% sự biến thiên trong đánh giá nghiên cứu đến từ tổng số token được sử dụng, không phải lựa chọn mô hình, theo Anthropic. |
| Scale effort | "Match agent count to complexity" | Lead ước tính độ khó truy vấn, tạo 1 hoặc 10+ worker tương ứng. |
| Synthesis conflict | "Workers disagree" | Hai worker trả về các sự kiện mâu thuẫn; lead phải làm nổi bật sự bất đồng, không được âm thầm chọn một bên. |

## Đọc thêm

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — tài liệu tham khảo thực tế cho mô hình supervisor
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — supervisor gọi công cụ hiện là hình thức được khuyến nghị
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — helper cũ, vẫn được sử dụng trong thực tế năm 2026
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — biến thể supervisor dựa trên bàn giao (handoff)