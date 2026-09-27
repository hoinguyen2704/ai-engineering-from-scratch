# Load Testing LLM APIs — Tại sao k6 và Locust lại đưa ra kết quả sai lệch

> Các công cụ kiểm thử tải truyền thống không được thiết kế cho phản hồi dạng streaming, độ dài đầu ra biến thiên, các chỉ số ở cấp độ token, hoặc tình trạng bão hòa GPU. Hai cái bẫy thường khiến hầu hết các đội ngũ gặp khó khăn. Bẫy GIL: Việc đo lường ở cấp độ token của Locust chạy quá trình token hóa dưới Python GIL, vốn cạnh tranh với việc tạo request khi có độ đồng thời cao; tình trạng tồn đọng token hóa sau đó làm tăng độ trễ giữa các token (inter-token latency) được báo cáo — client của bạn mới là nút thắt cổ chai, không phải server. Bẫy đồng nhất prompt: các prompt giống hệt nhau trong một vòng lặp chỉ kiểm tra một điểm trên phân phối token; lưu lượng thực tế có độ dài biến thiên và các prefix match đa dạng. LLMPerf giải quyết vấn đề này với `--mean-input-tokens` + `--stddev-input-tokens`. Ánh xạ công cụ năm 2026: Các công cụ chuyên biệt cho LLM (GenAI-Perf, LLMPerf, LLM-Locust, guidellm) để đảm bảo độ chính xác ở cấp độ token; **k6 v2026.1.0** + **k6 Operator 1.0 GA (Tháng 9/2025)** — hỗ trợ streaming, phân tán native trên Kubernetes thông qua các CRD TestRun/PrivateLoadZone, tốt nhất cho các cổng CI/CD; Vegeta cho việc bão hòa tốc độ cố định bằng Go; Locust 2.43.3 chỉ dùng với extension LLM-Locust cho streaming. Các mô hình tải: steady-state (trạng thái ổn định), ramp (tăng dần), spike (tăng đột biến - kiểm tra autoscaling), soak (kiểm tra rò rỉ bộ nhớ).

**Type:** Build
**Languages:** Python (stdlib, toy realistic-prompt generator + latency collector)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~75 phút

## Mục tiêu học tập

- Giải thích hai anti-pattern (bẫy GIL, bẫy đồng nhất prompt) khiến các công cụ kiểm thử tải thông thường đưa ra kết quả sai lệch cho LLM API.
- Chọn công cụ phù hợp cho từng mục đích: LLMPerf (chạy benchmark), k6 + streaming extension (cổng CI), guidellm (tổng hợp quy mô lớn), GenAI-Perf (tham chiếu từ NVIDIA).
- Thiết kế bốn mô hình tải (steady, ramp, spike, soak) và nêu tên chế độ lỗi mà mỗi mô hình phát hiện được.
- Xây dựng phân phối prompt thực tế sử dụng giá trị trung bình (mean) + độ lệch chuẩn (stddev) của số lượng input token thay vì độ dài cố định.

## Vấn đề

Bạn đã kiểm thử endpoint LLM bằng k6 với 500 người dùng đồng thời. Nó vẫn hoạt động tốt. Bạn triển khai. Trong môi trường production với 200 người dùng thực tế, dịch vụ bị sập — P99 TTFT tăng vọt, GPU bị quá tải.

Hai điều đã xảy ra. Thứ nhất, k6 gửi 500 prompt giống hệt nhau — việc gộp request (request-coalescing) và prefix caching khiến hệ thống trông như đang xử lý 500 lượt giải mã đồng thời trong khi thực tế chỉ xử lý một. Thứ hai, k6 không theo dõi độ trễ giữa các token trên các phản hồi streaming theo cách mà mắt người cảm nhận; nó nhìn thấy một kết nối HTTP, không phải 500 token đến ở các khoảng thời gian khác nhau.

Kiểm thử tải cho LLM là một lĩnh vực chuyên biệt.

## Khái niệm

### Bẫy GIL (Locust)

Locust sử dụng Python và chạy quá trình token hóa ở phía client dưới GIL. Khi độ đồng thời cao, bộ token hóa bị xếp hàng chờ sau quá trình tạo request. Độ trễ giữa các token được báo cáo bao gồm cả độ trễ do tồn đọng token hóa ở phía client. Bạn nghĩ server chậm; thực tế là do bộ kiểm thử.

Giải pháp: Extension LLM-Locust chuyển quá trình token hóa sang các tiến trình riêng biệt, hoặc sử dụng bộ kiểm thử bằng ngôn ngữ biên dịch (k6, LLMPerf sử dụng tokenizers.rs).

### Bẫy đồng nhất prompt

Tất cả các công cụ kiểm thử tải phổ biến đều cho phép bạn cấu hình một prompt duy nhất. Trong một bài kiểm tra lặp lại 10.000 lần, cùng một prompt chính xác đó được gửi đi mỗi lần. Server nhìn thấy cùng một prefix mỗi lần — tỷ lệ cache hit của prefix đạt gần 100%, thông lượng trông rất tốt.

Giải pháp: lấy mẫu từ một phân phối prompt. LLMPerf sử dụng `--mean-input-tokens 500 --stddev-input-tokens 150` — độ dài đa dạng, nội dung đa dạng.

### Bốn mô hình tải

1. **Steady-state** — RPS không đổi trong 30-60 phút. Phát hiện: các lỗi hồi quy hiệu năng cơ bản.
2. **Ramp** — tăng RPS tuyến tính từ 0 đến mục tiêu trong 15 phút. Phát hiện: điểm giới hạn năng lực, các bất thường khi khởi động.
3. **Spike** — tăng đột ngột 3-10x RPS trong 2 phút rồi quay lại. Phát hiện: độ trễ autoscaling, bão hòa hàng đợi, tác động của cold-start.
4. **Soak** — trạng thái ổn định trong 4-8 giờ. Phát hiện: rò rỉ bộ nhớ, trôi dạt connection-pool, tràn dữ liệu quan sát (observability).

### Ánh xạ công cụ năm 2026

**LLMPerf** (Anyscale) — Dùng Python nhưng token hóa dựa trên Rust. Prompt theo mean/stddev. Hỗ trợ streaming. Lựa chọn mặc định tốt nhất cho các bài chạy hiệu năng.

**NVIDIA GenAI-Perf** — Tham chiếu từ NVIDIA. Sử dụng Triton client; độ bao phủ chỉ số toàn diện. Lưu ý ITL của nó không bao gồm TTFT; trong khi LLMPerf thì có. Hai công cụ tạo ra TPOT khác nhau cho cùng một server.

**LLM-Locust** (TrueFoundry) — Extension của Locust giúp khắc phục bẫy GIL. Sử dụng DSL quen thuộc của Locust + các chỉ số streaming.

**guidellm** — benchmark tổng hợp quy mô lớn.

**k6 v2026.1.0** + **k6 Operator 1.0 GA (Tháng 9/2025)**:
- Bản thân k6 (Go, biên dịch, không có GIL) đã thêm các chỉ số hỗ trợ streaming.
- k6 Operator sử dụng các CRD TestRun / PrivateLoadZone cho kiểm thử phân tán native trên Kubernetes.
- Tốt nhất cho các cổng CI/CD và kiểm thử SLA.

**Vegeta** — Go, đơn giản hơn k6. Bão hòa HTTP ở tốc độ không đổi. Không chuyên cho LLM nhưng tốt cho việc kiểm thử gateway / rate-limit.

**Locust 2.43.3 bản gốc** — mắc bẫy GIL đối với LLM. Chỉ dùng được với extension LLM-Locust.

### Cổng SLA trong CI

Chạy k6 trên PR với:

- 30-50 lần lặp ở mức RPS cơ sở.
- Cổng: P50/P95 TTFT, 5xx < 5%, TPOT dưới ngưỡng.
- Ngắt build nếu vi phạm.

### Phân phối prompt thực tế

Xây dựng từ các mẫu lưu lượng thực tế (nếu có) hoặc từ các phân phối đã công bố (ví dụ: prompt ShareGPT cho chat, HumanEval cho code). Cung cấp giá trị mean + stddev cho LLMPerf. Tránh bằng mọi giá việc lặp lại với một prompt duy nhất.

### Các con số cần ghi nhớ

- k6 Operator 1.0 GA: Tháng 9/2025.
- k6 v2026.1.0: các chỉ số hỗ trợ streaming.
- Một lần chạy LLMPerf điển hình: 100-1000 request ở độ đồng thời X.
- Cổng CI điển hình: 30-50 lần lặp mỗi PR.
- Bốn mô hình: steady, ramp, spike, soak.

```figure
load-pattern-waves
```

## Sử dụng

`code/main.py` mô phỏng một bài kiểm tra tải với phân phối prompt thực tế, đo lường TPOT hiệu dụng và minh họa bẫy đồng nhất prompt.

## Triển khai

Bài học này tạo ra `outputs/skill-load-test-plan.md`. Dựa trên khối lượng công việc và SLA, chọn công cụ và thiết kế bốn mô hình tải.

## Bài tập

1. Chạy `code/main.py`. So sánh phân phối đồng nhất vs thực tế — khoảng cách nằm ở đâu?
2. Viết script k6 cho một cổng CI: TTFT P95 < 800 ms ở 100 người dùng đồng thời, thời gian chạy 5 phút.
3. Bài kiểm tra soak của bạn cho thấy bộ nhớ tăng 50 MB/giờ. Nêu ba nguyên nhân và công cụ đo lường để phân biệt giữa chúng.
4. Kiểm tra spike từ 10 RPS lên 100 RPS. Thời gian phục hồi dự kiến là bao nhiêu nếu Karpenter + vLLM production-stack đang được sử dụng (Phase 17 · 03 + 18)?
5. GenAI-Perf báo cáo TPOT=6ms; LLMPerf báo cáo TPOT=11ms trên cùng một server. Giải thích.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| LLMPerf | "bộ kiểm thử LLM" | Công cụ benchmark của Anyscale, hỗ trợ streaming |
| GenAI-Perf | "công cụ NVIDIA" | Bộ kiểm thử tham chiếu của NVIDIA |
| LLM-Locust | "Locust cho LLM" | Extension của Locust khắc phục bẫy GIL |
| guidellm | "benchmark tổng hợp" | Công cụ tổng hợp quy mô lớn |
| k6 Operator | "K8s k6" | k6 phân tán dựa trên CRD |
| Bẫy GIL | "overhead client Python" | Tồn đọng token hóa làm tăng độ trễ báo cáo |
| Bẫy đồng nhất prompt | "lời nói dối về prompt đơn" | Vòng lặp với cùng một prompt làm cache hit, tăng thông lượng ảo |
| Steady-state | "tải không đổi" | RPS phẳng trong N phút |
| Ramp | "tăng tuyến tính" | 0 đến mục tiêu trong thời gian xác định |
| Spike | "kiểm tra bùng nổ" | Tăng đột biến rồi quay lại |
| Soak | "kiểm tra dài hạn" | Nhiều giờ để phát hiện rò rỉ |

## Đọc thêm

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)