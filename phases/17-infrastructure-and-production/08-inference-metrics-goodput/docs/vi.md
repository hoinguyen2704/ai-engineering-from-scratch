# Inference Metrics — TTFT, TPOT, ITL, Goodput, P99

> Bốn chỉ số quyết định liệu một hệ thống triển khai inference có đang hoạt động hiệu quả hay không. TTFT bao gồm prefill cộng với thời gian chờ (queue) và mạng. TPOT (tương đương với ITL) là chi phí giải mã (decode) bị giới hạn bởi bộ nhớ trên mỗi token. Độ trễ end-to-end (E2E) là TTFT cộng với TPOT nhân với độ dài đầu ra. Throughput là số lượng token mỗi giây được tổng hợp trên toàn bộ hệ thống. Tuy nhiên, chỉ số quan trọng nhất đối với sản phẩm là goodput — tỷ lệ các yêu cầu đáp ứng đồng thời mọi SLO. Throughput cao nhưng goodput thấp đồng nghĩa với việc bạn đang xử lý các token không bao giờ đến tay người dùng đúng hạn. Các con số tham chiếu cho Llama-3.1-8B-Instruct trên TRT-LLM vào năm 2026: TTFT trung bình 162 ms, TPOT trung bình 7.33 ms, E2E trung bình 1,093 ms. Luôn báo cáo P50, P90, P99 — đừng bao giờ chỉ dùng giá trị trung bình. Và hãy cẩn thận với cái bẫy đo lường: GenAI-Perf loại trừ TTFT khỏi phép tính ITL, trong khi LLMPerf lại bao gồm nó; hai công cụ này đưa ra kết quả TPOT khác nhau cho cùng một lần chạy.

**Type:** Learn
**Languages:** Python (stdlib, toy percentile calculator and goodput reporter)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals)
**Time:** ~60 phút

## Mục tiêu học tập

- Định nghĩa chính xác TTFT, TPOT, ITL, E2E, throughput và goodput, đồng thời nêu tên thành phần mà mỗi chỉ số đo lường.
- Giải thích tại sao giá trị trung bình (mean) là thống kê sai lầm cho việc phục vụ LLM và cách đọc P50/P90/P99.
- Xây dựng một SLO đa ràng buộc (ví dụ: TTFT<500 ms VÀ TPOT<15 ms VÀ E2E<2 s) và tính toán goodput dựa trên đó.
- Nêu tên hai công cụ benchmark đưa ra kết quả TPOT khác nhau cho cùng một lần chạy và giải thích lý do.

## Vấn đề

"Throughput của chúng tôi là 15,000 token mỗi giây." Vậy thì sao? Nếu 40% yêu cầu vượt quá 2 giây end-to-end, người dùng sẽ rời bỏ phiên làm việc. Chỉ riêng throughput không cho bạn biết liệu sản phẩm có đang hoạt động tốt hay không.

Inference có nhiều trục độ trễ và mỗi trục thất bại theo cách khác nhau. Prefill bị giới hạn bởi tính toán (compute-bound) và tỷ lệ thuận với độ dài prompt. Decode bị giới hạn bởi bộ nhớ (memory-bound) và tỷ lệ thuận với batch size. Độ trễ hàng đợi (queuing delay) là vấn đề vận hành. Mạng là vấn đề về khoảng cách vật lý. Bạn cần các chỉ số riêng biệt cho từng loại, cần các phân vị (percentiles) và cần một chỉ số tổng hợp duy nhất để trả lời câu hỏi "người dùng có nhận được những gì họ mong đợi không" — đó chính là goodput.

## Khái niệm

### TTFT — time to first token

`TTFT = queue_time + network_request + prefill_time`

Prefill chiếm ưu thế khi prompt dài. Trên Llama-3.3-70B FP8 chạy trên H100, một prompt 32k mất khoảng 800 ms chỉ riêng cho prefill. Thời gian chờ (queue time) là hành vi của bộ lập lịch dưới tải. Yêu cầu mạng là thời gian truyền tải bao gồm cả TLS. TTFT là độ trễ mà người dùng thấy trước khi bất kỳ nội dung nào được stream về.

### TPOT / ITL — inter-token latency

Nhiều tên gọi cho cùng một đại lượng. `TPOT` (thời gian cho mỗi token đầu ra), `ITL` (độ trễ giữa các token), `decode latency per token` — tất cả đều là một. Đó là khoảng thời gian giữa các token liên tiếp được stream sau token đầu tiên.

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

Trên cùng stack Llama-3.3-70B H100 với chunked prefill, TPOT trung bình khoảng 7 ms. Nếu không có chunked prefill, trong quá trình prefill dài của một chuỗi lân cận, TPOT có thể tăng vọt lên 50 ms. Hãy theo dõi P99, đừng nhìn vào giá trị trung bình.

### E2E latency

`E2E = TTFT + TPOT * output_tokens + network_response`

Đối với các đầu ra dài (>500 token), E2E bị chi phối bởi TPOT. Đối với các đầu ra ngắn với prompt dài, E2E bị chi phối bởi TTFT. Hãy báo cáo E2E có điều kiện theo độ dài đầu ra.

### Throughput

`throughput = total_output_tokens / elapsed_time`

Chỉ số tổng hợp. Cho biết hiệu suất của toàn bộ hệ thống. Không cho biết tình trạng của từng yêu cầu riêng lẻ.

### Goodput — chỉ số bạn thực sự quan tâm

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO là một đa ràng buộc. Một yêu cầu được coi là "tốt" chỉ khi mọi ràng buộc đều được thỏa mãn. Goodput là tỷ lệ phần trăm các yêu cầu đó. Throughput cao với goodput 60% là thất bại. Throughput thấp hơn với goodput 99% mới là mục tiêu.

Vào năm 2026, goodput là chỉ số được sử dụng trong các bài nộp MLPerf Inference v6.0 và trong việc theo dõi SLA nội bộ tại các nhà cung cấp nền tảng AI.

### Tại sao giá trị trung bình là thống kê sai lầm

Phân phối độ trễ của LLM bị lệch phải (right-skewed). Một batch decode với một yêu cầu prefill dài lân cận có thể gửi 500 token với TPOT ~7 ms và 20 token với TPOT ~60 ms. TPOT trung bình là 9 ms. TPOT P99 là 65 ms. Người dùng thường xuyên gặp phải mức P99 — đó là lý do tại sao họ rời đi.

Luôn báo cáo bộ ba (P50, P90, P99). Đối với trải nghiệm người dùng, P99 là chỉ số bạn cần tối ưu hóa.

### Các con số tham chiếu — Llama-3.1-8B-Instruct trên TRT-LLM, 2026

- TTFT trung bình: 162 ms
- TPOT trung bình: 7.33 ms
- E2E trung bình: 1,093 ms
- TPOT P99: dao động 10-25 ms tùy thuộc vào cấu hình chunked-prefill.

Đây là các điểm tham chiếu được NVIDIA công bố. Chúng thay đổi theo kích thước mô hình (70B sẽ gấp 3-5 lần), phần cứng (H100 so với B200 ~3 lần) và tải trọng.

### Cái bẫy đo lường

Hai trong số các công cụ benchmark phổ biến nhất năm 2026 đưa ra kết quả TPOT khác nhau cho cùng một lần chạy:

- **NVIDIA GenAI-Perf**: loại trừ TTFT khỏi phép tính ITL. ITL bắt đầu từ token thứ 2.
- **LLMPerf**: bao gồm TTFT. ITL bắt đầu từ token thứ 1.

Đối với một yêu cầu có TTFT 500 ms và 100 token đầu ra trong tổng thời gian decode 700 ms, GenAI-Perf báo cáo `ITL = 700/99 = 7.07 ms`, LLMPerf báo cáo `ITL = 1200/100 = 12.00 ms`. Việc chọn công cụ sẽ làm thay đổi con số.

Luôn nêu rõ công cụ nào được sử dụng. Luôn công bố định nghĩa.

### Xây dựng một SLO

Một SLO hợp lý hướng tới người dùng cho mô hình chat 70B vào năm 2026:

- TTFT P99 <= 800 ms.
- TPOT P99 <= 25 ms.
- E2E P99 <= 3 s cho các đầu ra <300 token.
- Mục tiêu goodput >= 99%.

Các SLO doanh nghiệp thường thắt chặt TTFT (200-400 ms) và nới lỏng E2E. Vấn đề là phải ghi lại chúng, đo lường cả ba và theo dõi goodput như một chỉ số tổng hợp duy nhất.

### Cách đo lường

- Chạy lưu lượng thực tế hoặc lưu lượng tổng hợp thực tế (LLMPerf với `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`).
- Nhắm mục tiêu gấp 2 lần concurrency đỉnh cho lần chạy benchmark.
- Chạy 30-50 lần lặp, lấy các phân vị của mẫu kết hợp.
- Công bố kèm tên công cụ, phiên bản công cụ, mô hình, phần cứng, concurrency và phân phối prompt.

```figure
throughput-latency
```

## Sử dụng

`code/main.py` là một công cụ tính goodput đơn giản. Tạo một phân phối độ trễ tổng hợp, áp dụng SLO và tính toán goodput. Nó cũng cho thấy sự khác biệt về TPOT giữa GenAI-Perf và LLMPerf trên cùng một trace.

## Triển khai

Bài học này tạo ra `outputs/skill-slo-goodput-gate.md`. Với một workload và SLO nhất định, nó tạo ra một công thức benchmark sẵn sàng cho CI/CD, giúp kiểm soát việc triển khai dựa trên goodput thay vì throughput.

## Bài tập

1. Chạy `code/main.py`. Tạo một phân phối với 1% spike ở phần đuôi. Goodput thay đổi thế nào khi bạn thắt chặt TPOT P99 từ 30 ms xuống 15 ms?
2. Một nhà cung cấp báo giá "15,000 tok/s trên Llama 3.3 70B H100". Hãy nêu ba câu hỏi cần đặt ra trước khi tin tưởng con số đó.
3. Tại sao chunked prefill bảo vệ được TPOT P99 nhưng không bảo vệ được TPOT trung bình?
4. Xây dựng một SLO người dùng cho trợ lý giọng nói (token đầu tiên được nghe thấy, không phải đọc). Chỉ số nào là dễ thấy nhất đối với người dùng?
5. Đọc README của LLMPerf và tài liệu của GenAI-Perf. Xác định ba chỉ số khác mà các công cụ này không thống nhất với nhau.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| TTFT | "time to first token" | Queue + mạng + prefill; bị chi phối bởi prefill khi prompt dài |
| TPOT | "time per output token" | Chi phí decode bị giới hạn bởi bộ nhớ trên mỗi token sau token đầu tiên |
| ITL | "inter-token latency" | Giống TPOT trong hầu hết các công cụ (không phải tất cả — xem GenAI-Perf) |
| E2E | "end to end" | TTFT + TPOT * độ dài đầu ra; cộng thêm mạng phía phản hồi |
| Throughput | "tok/s" | Hiệu suất hệ thống; vô dụng nếu không có các phân vị độ trễ |
| Goodput | "SLO-met rate" | Tỷ lệ các yêu cầu đáp ứng đồng thời mọi ràng buộc SLO |
| P99 | "tail" | Độ trễ xấu nhất trong 100 trường hợp; chỉ số trải nghiệm người dùng |
| SLO multi-constraint | "the joint" | Phép AND của cả ba giới hạn độ trễ; yêu cầu thất bại nếu vi phạm bất kỳ ràng buộc nào |
| GenAI-Perf vs LLMPerf | "the tool trap" | Các công cụ không thống nhất về việc ITL có bao gồm TTFT hay không |

## Đọc thêm

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — định nghĩa chuẩn về TTFT, ITL, TPOT.
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — các định nghĩa thay thế và công thức đo lường.
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — đo lường ứng dụng trên các hệ thống triển khai thực tế.
- [LLMPerf](https://github.com/ray-project/llmperf) — benchmark mã nguồn mở dựa trên Ray.
- [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md) — công cụ benchmark của NVIDIA.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — benchmark dựa trên goodput được ngành công nghiệp chấp nhận.