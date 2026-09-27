# Eval-Driven Agent Development

> Hướng dẫn từ Anthropic: "hãy bắt đầu với các prompt đơn giản, tối ưu hóa chúng bằng cách đánh giá toàn diện, và chỉ thêm các hệ thống agent đa bước khi thực sự cần thiết." Đánh giá không phải là bước cuối cùng. Đó là vòng lặp bên ngoài thúc đẩy mọi lựa chọn khác trong Phase 14.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Tất cả các bài trong Phase 14.
**Time:** ~60 phút

## Mục tiêu học tập

- Gọi tên ba tầng đánh giá — static benchmarks, custom offline, online production — và mục đích của từng tầng.
- Giải thích vòng lặp chặt chẽ giữa evaluator và optimizer.
- Mô tả phương pháp thực hành tốt nhất năm 2026: các eval nằm cùng thư mục với code, chạy trong CI, và chặn các PR.
- Kết nối mọi bài học trong Phase 14 với trường hợp đánh giá (eval case) mà nó tạo ra.

## Vấn đề

Các agent thường vượt qua các bản demo. Nhưng chúng thất bại trong môi trường production theo những cách mà demo không thể dự đoán được. Các benchmark trả lời câu hỏi "model này có năng lực tổng quát không?" chứ không phải "agent này có đang gửi đúng các bản vá cho sản phẩm của tôi không?". Câu trả lời là: đánh giá ở ba tầng, chạy liên tục, với mọi guardrail và quy tắc đã học được ánh xạ tới một eval case.

## Khái niệm

### Ba tầng đánh giá

1. **Static benchmarks** — SWE-bench Verified cho code (Bài 19), WebArena/OSWorld cho duyệt web / desktop (Bài 20), GAIA cho tổng quát (Bài 19), BFCL V4 cho việc sử dụng tool (Bài 06). Sử dụng để so sánh giữa các model và kiểm soát hồi quy (regression gating). Sự nhiễm bẩn dữ liệu là có thật: SWE-bench+ đã phát hiện 32.67% dữ liệu giải pháp bị rò rỉ. Luôn báo cáo các điểm số đã được Verified / kiểm định.

2. **Custom offline evals** — hình thái sản phẩm của bạn:
   - LLM-as-judge (Langfuse, Phoenix, Opik — Bài 24).
   - Dựa trên thực thi (chạy bản vá, kiểm tra các test).
   - Dựa trên quỹ đạo (so sánh chuỗi hành động với dữ liệu chuẩn; OSWorld-Human cho thấy các agent hàng đầu đạt hiệu suất gấp 1.4-2.7 lần so với dữ liệu chuẩn).

3. **Online evals** — môi trường production:
   - Phát lại phiên làm việc (Langfuse).
   - Cảnh báo kích hoạt bởi guardrail (Bài 16, 21).
   - Theo dõi chi phí / độ trễ trên mỗi bước (Bài 23 OTel spans).

### Evaluator-optimizer (Anthropic)

Vòng lặp chặt chẽ:

1. Proposer tạo ra kết quả đầu ra.
2. Evaluator đánh giá.
3. Tinh chỉnh cho đến khi evaluator thông qua.

Đây là Self-Refine (Bài 05) được tổng quát hóa. Bất kỳ luồng agent nào bạn quan tâm đều có thể bao bọc trong evaluator-optimizer để đảm bảo độ tin cậy.

### Phương pháp thực hành tốt nhất năm 2026

- Evals nằm cùng thư mục với code.
- Chạy trong CI trên mỗi PR.
- Chặn merge dựa trên điểm số eval (ví dụ: "không hồi quy > 5% so với nhánh main").
- Mọi guardrail đều ánh xạ tới một eval case.
- Mọi quy tắc đã học (Reflexion, pro-workflow learn-rule) đều ánh xạ tới một trường hợp thất bại.

### Kết nối Phase 14

Mỗi bài học trong Phase 14 tạo ra các eval case:

| Bài học | Eval case được tạo ra |
|--------|------------------------|
| 01 Agent Loop | Cạn kiệt ngân sách, guard cho vòng lặp vô tận |
| 02 ReWOO | Planner lập kế hoạch lại chính xác khi một tool thất bại |
| 03 Reflexion | Các phản hồi đã học được áp dụng khi thử lại |
| 05 Self-Refine/CRITIC | Judge thông qua kết quả đầu ra đã tinh chỉnh |
| 06 Tool Use | Ép kiểu đối số hoạt động; các tool không xác định bị từ chối |
| 07-10 Memory | Trích dẫn truy xuất khớp với nguồn; dữ liệu cũ làm mất hiệu lực |
| 12 Workflow Patterns | Mỗi pattern tạo ra kết quả đầu ra chính xác |
| 13 LangGraph | Resume tái tạo trạng thái chính xác |
| 14 AutoGen Actors | DLQ bắt được các handler bị crash |
| 16 OpenAI Agents SDK | Guardrail kích hoạt trên các input đúng |
| 17 Claude Agent SDK | Kết quả subagent trả về cho orchestrator |
| 19-20 Benchmarks | Điểm SWE-bench Verified, tỷ lệ thành công WebArena, hiệu suất OSWorld |
| 21 Computer Use | An toàn trên mỗi bước bắt được DOM bị tiêm vào |
| 23 OTel | Spans phát ra các thuộc tính bắt buộc |
| 26 Failure Modes | Các detector gắn thẻ các lỗi đã biết |
| 27 Prompt Injection | PVE từ chối các truy xuất bị nhiễm độc |
| 28 Orchestration | Supervisor định tuyến đến đúng chuyên gia |
| 29 Runtime Shapes | DLQ xử lý N% thất bại |

Nếu bộ eval của bạn có các trường hợp cho từng mục trên, bạn đã bao phủ toàn bộ Phase 14.

### Nơi mà eval-driven development thất bại

- **Không có baseline.** Evals mà không có dữ liệu "tốt nhất gần nhất" (last-known-good) sẽ không thể đọc được. Hãy lưu trữ các baseline.
- **LLM-judge không có căn cứ.** Các judge cũng có thể bị ảo giác. Pattern CRITIC (Bài 05) — judge dựa trên các tool bên ngoài.
- **Quá khớp (Over-fitting) với evals.** Tối ưu hóa cho eval làm lệch hướng khỏi tính hữu dụng trong production. Hãy xoay vòng các trường hợp.
- **Evals không ổn định (Flaky).** Các trường hợp không xác định gây ra báo động giả. Hãy ghim các seed, chụp nhanh trạng thái.

```figure
ae-eval-three-layers
```

## Xây dựng

`code/main.py` là một bộ khung eval chuẩn:

- Đăng ký trường hợp với các danh mục (benchmark, custom, online).
- Một agent được viết kịch bản để kiểm thử.
- Vòng lặp evaluator-optimizer: đề xuất, đánh giá, tinh chỉnh cho đến khi đạt hoặc hết số vòng tối đa.
- CI gate: tổng hợp tỷ lệ đạt + hồi quy so với baseline.

Chạy nó:

```
python3 code/main.py
```

Đầu ra: đạt/thất bại trên mỗi trường hợp, cờ hồi quy, quyết định của CI gate.

## Sử dụng

- Viết các eval case trong cùng repo với code agent của bạn.
- Chạy chúng trên mỗi PR thông qua CI.
- Làm thất bại bản build nếu có hồi quy.
- Theo dõi tỷ lệ đạt theo thời gian.
- Gắn mọi lỗi production vào một trường hợp mới.

## Triển khai

`outputs/skill-eval-suite.md` xây dựng một bộ eval ba tầng cho sản phẩm agent với các CI gate và theo dõi hồi quy.

## Bài tập

1. Lấy một trong những lỗi production của bạn. Viết một eval case tái tạo nó. Agent của bạn có vượt qua nó bây giờ không?
2. Xây dựng một rubric LLM-judge cho lĩnh vực của bạn với ba chiều (thực tế, giọng điệu, phạm vi). Chấm điểm 50 phiên làm việc.
3. Kết nối bộ eval vào CI. Làm thất bại bản build nếu hồi quy >=5%.
4. Thêm một chỉ số hiệu quả quỹ đạo: agent đã thực hiện bao nhiêu bước so với một quỹ đạo chuẩn?
5. Ánh xạ mọi bài học Phase 14 vào một eval case trong bộ của bạn. Có thiếu cái nào không? Đó là khoảng trống cần lấp đầy.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Static benchmark | "Off-the-shelf eval" | SWE-bench, GAIA, AgentBench, WebArena, OSWorld |
| Custom offline eval | "Domain eval" | LLM-as-judge / exec / quỹ đạo trên hình thái sản phẩm của bạn |
| Online eval | "Production eval" | Phát lại phiên, cảnh báo guardrail, theo dõi chi phí/độ trễ |
| Evaluator-optimizer | "Propose-judge-refine" | Lặp lại cho đến khi judge thông qua |
| CI gate | "Merge blocker" | Làm thất bại bản build khi có hồi quy eval |
| Baseline | "Last-known-good" | Điểm số tham chiếu để phát hiện hồi quy |
| Trajectory efficiency | "Steps over gold" | Số bước của agent chia cho số bước tối thiểu của chuyên gia |

## Đọc thêm

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — "bắt đầu đơn giản, tối ưu hóa với evals"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — benchmark được tuyển chọn
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — benchmark sử dụng tool
- [Langfuse docs](https://langfuse.com/) — evals + phát lại phiên trong thực tế