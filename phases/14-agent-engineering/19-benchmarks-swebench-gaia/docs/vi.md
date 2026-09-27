# Benchmarks: SWE-bench, GAIA, AgentBench

> Ba bộ benchmark đóng vai trò trụ cột trong việc đánh giá agent vào năm 2026. SWE-bench kiểm tra khả năng vá lỗi code. GAIA kiểm tra khả năng sử dụng công cụ của một agent tổng quát. AgentBench kiểm tra khả năng suy luận trong nhiều môi trường khác nhau. Hãy nắm vững cấu trúc, vấn đề nhiễm dữ liệu (contamination) và những gì các benchmark này không đo lường được.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 06 (Tool Use)
**Time:** ~60 minutes

## Learning Objectives

- Nêu tên bộ kiểm thử (test harness) của SWE-bench (FAIL_TO_PASS) và giải thích tại sao nó lại sử dụng unit test làm cổng kiểm soát.
- Giải thích lý do tồn tại của SWE-bench Verified (OpenAI, 500 tác vụ) và những gì nó loại bỏ.
- Mô tả thiết kế của GAIA: đơn giản với con người, khó với AI; ba cấp độ khó.
- Nêu tên tám môi trường của AgentBench và rào cản chính đối với các LLM mã nguồn mở.
- Tóm tắt phát hiện về sự nhiễm dữ liệu của SWE-bench+ và các hệ quả của nó.

## The Problem

Các bảng xếp hạng (leaderboard) cho bạn biết mô hình nào thắng trên một benchmark cụ thể. Chúng không cho bạn biết:

- Benchmark đó có bị nhiễm dữ liệu hay không (lời giải nằm trong dữ liệu huấn luyện, rò rỉ tập kiểm tra).
- Benchmark đó có đo lường đúng thứ bạn quan tâm hay không (code so với duyệt web so với tổng quát).
- Bộ đánh giá có mạnh mẽ hay không (khớp AST, kiểm tra trạng thái, đánh giá bởi con người).

Hãy nắm rõ ba benchmark trụ cột này và các chế độ lỗi của chúng trước khi trích dẫn một con số.

## The Concept

### SWE-bench (Jimenez et al., ICLR 2024 oral)

- 2.294 issue thực tế từ GitHub của 12 repo Python phổ biến.
- Agent nhận được: codebase tại commit trước khi sửa lỗi + mô tả issue bằng ngôn ngữ tự nhiên.
- Agent tạo ra: một bản vá (patch).
- Bộ đánh giá: áp dụng bản vá, chạy bộ test của repo. Bản vá phải làm cho các test FAIL_TO_PASS chuyển sang trạng thái PASS (trước đó fail, giờ pass) mà không làm hỏng các test PASS_TO_PASS.

SWE-agent (Yang et al., 2024) đạt 12,5% khi ra mắt nhờ chú trọng vào giao diện agent-máy tính (các lệnh chỉnh sửa file, cú pháp tìm kiếm mà mô hình hiểu được).

### SWE-bench Verified

OpenAI, tháng 8 năm 2024. Tập con gồm 500 tác vụ được con người chọn lọc. Loại bỏ các issue mơ hồ, các test không đáng tin cậy và các tác vụ mà cách sửa không rõ ràng. Đây là benchmark chính để trả lời câu hỏi: "Agent của bạn có tạo ra bản vá thực tế không?"

### Contamination

- Hơn 94% các issue trong SWE-bench có trước thời điểm cắt dữ liệu (cutoff) của hầu hết các mô hình.
- **SWE-bench+** phát hiện 32,67% các bản vá thành công bị rò rỉ lời giải trong văn bản issue (mô hình đã thấy cách sửa trong phần mô tả), và 31,08% đáng ngờ do độ bao phủ test yếu.
- Verified sạch hơn nhưng không hoàn toàn không bị nhiễm dữ liệu.

Hệ quả thực tế: một mô hình đạt 50% trên SWE-bench có thể chỉ đạt 35% trên SWE-bench+. Luôn báo cáo cả hai nếu bạn tuyên bố hiệu suất trên SWE-bench.

### GAIA (Mialon et al., tháng 11 năm 2023)

- 466 câu hỏi; 300 câu được giữ lại cho bảng xếp hạng riêng tư tại huggingface.co/gaia-benchmark.
- Triết lý thiết kế: "đơn giản về mặt khái niệm với con người (92%) nhưng khó với AI (GPT-4 với plugin: 15%)."
- Kiểm tra khả năng suy luận, đa phương thức, web, sử dụng công cụ.
- Ba cấp độ khó; Cấp độ 3 đòi hỏi chuỗi công cụ dài qua nhiều phương thức.

GAIA là thứ bạn chạy để đo lường "khả năng tổng quát". Đừng nhầm lẫn với các benchmark chuyên biệt về code.

### AgentBench (Liu et al., ICLR 2024)

- 8 môi trường bao gồm code (Bash, DB, KG), trò chơi (Alfworld, LTP), web (WebShop, Mind2Web) và tạo nội dung mở.
- Đa lượt (multi-turn), khoảng 4k-13k lượt mỗi split.
- Phát hiện chính: suy luận dài hạn, ra quyết định và tuân thủ hướng dẫn là những rào cản khiến các LLM mã nguồn mở chưa bắt kịp các mô hình thương mại.

### Những gì các benchmark này không đo lường

- Chi phí vận hành thực tế (token, thời gian thực).
- Hành vi an toàn trong các điều kiện đối nghịch.
- Hiệu suất trên domain của riêng bạn (hãy sử dụng các đánh giá của riêng bạn, Lesson 30).
- Các lỗi ở phần đuôi (benchmark lấy trung bình; người vận hành sản phẩm quan tâm đến 1% tệ nhất).

### Nơi việc đánh giá benchmark đi sai hướng

- **Ám ảnh bởi một con số duy nhất.** SWE-bench 50% cho bạn biết ít hơn nhiều so với phân phối chi phí + số bước P50/P75/P95.
- **Các tuyên bố bị nhiễm dữ liệu.** Báo cáo SWE-bench mà không đề cập đến Verified hoặc SWE-bench+ là gây hiểu lầm.
- **Benchmark là mục tiêu phát triển.** Tối ưu hóa cho benchmark sẽ làm lệch hướng khỏi tính hữu dụng trong sản xuất.

```figure
ae-swebench-gate
```

## Build It

`code/main.py` triển khai một bộ harness mô phỏng SWE-bench:

- Các tác vụ sửa lỗi tổng hợp (3 tác vụ).
- Một "agent" được viết kịch bản để đề xuất các bản vá.
- Một trình chạy test kiểm tra FAIL_TO_PASS (lỗi đã được sửa) và PASS_TO_PASS (không có gì bị hỏng).
- Một bộ phân loại độ khó kiểu GAIA dựa trên độ sâu phân rã câu hỏi.

Chạy nó:

```
python3 code/main.py
```

Kết quả đầu ra hiển thị tỷ lệ giải quyết theo từng tác vụ + theo độ khó và làm rõ các quy tắc đánh giá.

## Use It

- **SWE-bench Verified** cho các agent về code. Luôn báo cáo điểm Verified.
- **GAIA** cho các agent tổng quát. Sử dụng split bảng xếp hạng riêng tư.
- **AgentBench** để so sánh đa môi trường.
- **Custom evals** (Lesson 30) cho hình thái thực tế của sản phẩm bạn.

## Ship It

`outputs/skill-benchmark-harness.md` xây dựng một bộ harness kiểu SWE-bench cho bất kỳ cặp codebase-tác vụ nào với cơ chế cổng FAIL_TO_PASS / PASS_TO_PASS.

## Exercises

1. Chuyển bộ harness mô phỏng sang chạy trên một repo thực tế (chọn một repo của bạn). Viết 3 test FAIL_TO_PASS cho các lỗi đã biết.
2. Thêm chỉ số đếm số bước (step-count). Trên 3 tác vụ của bạn, agent cần bao nhiêu bước để giải quyết?
3. Đọc bài báo SWE-bench+. Triển khai kiểm tra rò rỉ lời giải (khớp mẫu văn bản issue với diff).
4. Tải xuống một câu hỏi GAIA từ split công khai. Truy vết những gì một agent cấp độ GPT-4 sẽ làm. Nó cần những công cụ gì?
5. Đọc phân tích theo môi trường của AgentBench. Môi trường nào phản ánh bề mặt sản phẩm của bạn? "SOTA" ở đó trông như thế nào?

## Key Terms

| Thuật ngữ | Cách mọi người nói | Ý nghĩa thực tế |
|------|----------------|------------------------|
| SWE-bench | "Benchmark cho code agent" | 2.294 issue GitHub; bản vá phải làm thay đổi các test FAIL_TO_PASS |
| SWE-bench Verified | "SWE-bench sạch" | 500 tác vụ được con người chọn lọc, OpenAI |
| FAIL_TO_PASS | "Cổng sửa lỗi" | Các test trước đó fail và phải pass sau khi vá |
| PASS_TO_PASS | "Cổng không hồi quy" | Các test đã pass và phải tiếp tục pass |
| GAIA | "Benchmark tổng quát" | 466 câu hỏi đa công cụ, dễ với người / khó với AI |
| AgentBench | "Benchmark đa môi trường" | 8 môi trường; đa lượt, tầm nhìn dài hạn |
| Contamination | "Rò rỉ tập huấn luyện" | Các tác vụ benchmark có mặt trong dữ liệu huấn luyện mô hình |
| SWE-bench+ | "Kiểm toán nhiễm dữ liệu" | 32,67% rò rỉ lời giải được tìm thấy trong các bản vá thành công của SWE-bench |

## Further Reading

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — benchmark gốc
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — tập con được chọn lọc
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — benchmark tổng quát
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — bộ suite đa môi trường