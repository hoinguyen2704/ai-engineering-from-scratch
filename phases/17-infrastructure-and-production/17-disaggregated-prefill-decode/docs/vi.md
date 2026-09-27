# Disaggregated Prefill/Decode — NVIDIA Dynamo và llm-d

> Prefill bị giới hạn bởi tính toán (compute-bound); decode bị giới hạn bởi bộ nhớ (memory-bound). Chạy cả hai trên cùng một GPU gây lãng phí tài nguyên. Disaggregation (phân tách) chia chúng thành các pool riêng biệt và chuyển KV cache giữa chúng qua NIXL (RDMA/InfiniBand hoặc TCP fallback). NVIDIA Dynamo (công bố tại GTC 2025, bản 1.0 GA) nằm phía trên vLLM/SGLang/TRT-LLM — Planner Profiler + SLA Planner của nó tự động điều chỉnh tỷ lệ prefill:decode để đáp ứng các SLO. NVIDIA công bố mức tăng throughput trong phạm vi này — developer.nvidia.com (2025-06) cho thấy mức cải thiện ~6x cho DeepSeek-R1 MoE trên GB200 NVL72 + Dynamo trong chế độ độ trễ trung bình, và trang sản phẩm Dynamo (developer.nvidia.com, không ghi ngày) quảng cáo mức throughput MoE lên tới 50x trên GB300 NVL72 + Dynamo so với Hopper. Con số "30x" là tổng hợp từ cộng đồng trên toàn bộ stack Blackwell + Dynamo + DeepSeek-R1; chúng tôi chưa tìm thấy nguồn chính thống nào khẳng định chính xác 30x, vì vậy hãy coi đó là một tuyên bố mang tính định hướng. llm-d (Red Hat + AWS) là Kubernetes-native: prefill / decode / router là các Service độc lập với HPA theo từng vai trò. llm-d 0.5 bổ sung hierarchical KV offloading, cache-aware LoRA routing, UCCL networking, và scale-to-zero. Về kinh tế: tổng hợp nội bộ từ nhiều tiết lộ của khách hàng cho thấy mức tiết kiệm 30–40% trên $2M-class inference spend (i.e., $600-800K/năm khi chuyển từ serving tập trung (colocated) sang phân tách với Dynamo ở cùng mức SLA; con số $2M→$600-800K cụ thể là một tổng hợp nội bộ, không phải là một nghiên cứu điển hình được công bố — hãy sử dụng nó như một mốc tham chiếu về độ lớn, không phải là trích dẫn tham khảo. Các prompt ngắn (<512 token, output ngắn) không bù đắp được chi phí chuyển đổi.

**Type:** Learn
**Languages:** Python (stdlib, toy disaggregated-vs-colocated simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 08 (Inference Metrics)
**Time:** ~75 phút

## Mục tiêu học tập

- Giải thích lý do tại sao prefill và decode có các phân bổ GPU tối ưu khác nhau và định lượng sự lãng phí khi chạy tập trung (colocation).
- Vẽ sơ đồ kiến trúc phân tách: prefill pool, decode pool, chuyển KV qua NIXL, router.
- Nêu điều kiện khi việc phân tách KHÔNG mang lại hiệu quả (prompt ngắn, output ngắn).
- Phân biệt NVIDIA Dynamo (stack-above) với llm-d (Kubernetes-native) và áp dụng từng loại vào ngữ cảnh vận hành phù hợp.

## Vấn đề

Bạn chạy Llama 3.3 70B trên 8 H100. Với workload hỗn hợp (prompt dài + output ngắn), các GPU sẽ nhàn rỗi trong quá trình decode vì phần lớn tính toán đã dành cho prefill. Với workload khác (prompt ngắn + output dài), điều ngược lại xảy ra. Việc chạy chung prefill + decode có nghĩa là bạn đang cung cấp dư thừa cho cả hai.

Tác động ngân sách: 20-40% thời gian GPU bị lãng phí vào tài nguyên không phù hợp. Bạn đang mua sức mạnh tính toán H100 để chạy decode bị giới hạn bởi bộ nhớ, hoặc mua băng thông HBM của H100 để chạy prefill bị giới hạn bởi tính toán. Cả hai đều là sự lãng phí đắt đỏ.

Phân tách chia prefill và decode thành các pool riêng biệt được định cỡ theo điểm nghẽn của từng loại. KV cache được chuyển từ prefill pool sang decode pool thông qua kết nối băng thông cao.

## Khái niệm

### Tại sao các điểm nghẽn lại khác nhau

**Prefill** — chạy Transformer trên toàn bộ prompt đầu vào trong một lần forward. Các phép nhân ma trận chiếm ưu thế; bị giới hạn bởi tính toán (compute-bound). H100 FP8 cung cấp ~2000 TFLOPS throughput hữu ích. Hiệu suất batch tốt — một lần forward xử lý nhiều token.

**Decode** — tạo từng token một, đọc toàn bộ trọng số trong mỗi lần lặp. Bị giới hạn bởi băng thông bộ nhớ (memory-bandwidth-bound). HBM3 cung cấp ~3 TB/s. Hiệu suất batch chỉ tốt ở mức concurrency cao — trọng số được đọc được khấu hao trên toàn bộ batch.

Khi chạy chung: bạn mua các GPU được tối ưu cho cả hai. H100 tốt cho cả hai nhưng chi phí là như nhau. Ở quy mô lớn, bạn muốn prefill pool trên H100 / thiên về tính toán; decode pool trên H200 / thiên về bộ nhớ, hoặc với kỹ thuật quantization mạnh mẽ.

### Kiến trúc

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL là giao thức truyền tải liên nút (inter-node) của NVIDIA. Sử dụng RDMA/InfiniBand khi có sẵn, nếu không sẽ fallback về TCP. Độ trễ truyền tải là có thật — thường là 20-80 ms cho KV cache của một prompt 4K-token trên 70B FP8. Đây là lý do tại sao các prompt ngắn không biện minh được cho việc phân tách: chi phí truyền tải vượt quá mức tiết kiệm.

### Dynamo vs llm-d

**NVIDIA Dynamo** (công bố tại GTC 2025, 1.0 GA):
- Nằm phía trên vLLM, SGLang, TRT-LLM với vai trò điều phối (orchestrator).
- Planner Profiler đo lường workload, SLA Planner tự động cấu hình tỷ lệ prefill:decode.
- Core bằng Rust, khả năng mở rộng bằng Python.
- Mức tăng throughput: NVIDIA báo cáo 6x cho DeepSeek-R1 MoE trên GB200 NVL72 + Dynamo trong chế độ độ trễ trung bình (developer.nvidia.com, 2025-06); các báo cáo cộng đồng về "lên tới 30x" trên toàn bộ stack Blackwell + Dynamo + DeepSeek-R1 thiếu nguồn chính thống và nên được coi là mang tính định hướng.
- GB300 NVL72 + Dynamo: throughput MoE lên tới 50x so với Hopper theo trang sản phẩm Dynamo (developer.nvidia.com, không ghi ngày).

**llm-d** (Red Hat + AWS, Kubernetes-native):
- Prefill / decode / router là các Kubernetes Service độc lập.
- HPA theo từng vai trò với tín hiệu từ độ sâu hàng đợi (prefill) / mức sử dụng KV (decode).
- `topologyConstraint packDomain: rack` đóng gói các nhóm prefill+decode trên cùng một rack để truyền KV băng thông cao.
- llm-d 0.5 (2026): hierarchical KV offloading, cache-aware LoRA routing, UCCL networking, scale-to-zero.

Sử dụng Dynamo nếu bạn muốn một trình điều phối stack-above được quản lý. Sử dụng llm-d nếu bạn muốn các primitive Kubernetes-native và cam kết với hệ sinh thái CNCF.

### Kinh tế

Tổng hợp nội bộ (không phải nghiên cứu điển hình đơn lẻ — mốc tham chiếu về độ lớn):

- $2M/năm chi phí inference cho serving tập trung.
- Chuyển sang phân tách với Dynamo.
- Cùng khối lượng request, cùng SLA độ trễ P99.
- Mức tiết kiệm được báo cáo: $600K–$800K/năm (giảm 30–40%).
- Không cần phần cứng mới.

Chúng tôi tổng hợp con số này từ nhiều tiết lộ của khách hàng thay vì một nghiên cứu điển hình đơn lẻ có thể trích dẫn; điểm dữ liệu công bố gần nhất là TTFT nhanh hơn 2x / throughput cao hơn 61% của Baseten với Dynamo KV routing (baseten.co, 2025-10), và dự báo của VAST + CoreWeave về mức tăng 60–130% tokens/$ ở tỷ lệ KV hit 40–60% (vastdata.com, 2025-12). Mức tiết kiệm đến từ việc định cỡ đúng cho từng pool; các workload nặng về prefill (RAG với prefix 8K+) hưởng lợi nhiều hơn so với các workload cân bằng.

### Khi nào KHÔNG NÊN phân tách

- Prompt < 512 token và output < 200 token: chi phí truyền tải chiếm ưu thế so với mức tăng.
- Cluster nhỏ (< 4 GPU): không đủ sự đa dạng pool.
- Đội ngũ không thể vận hành hai pool GPU với việc scaling theo vai trò: Dynamo có hỗ trợ nhưng không hề đơn giản.
- Không có hạ tầng RDMA: chi phí truyền tải TCP nặng nề hơn.

### Router tích hợp với Phase 17 · 11

Các router phân tách có nhận thức về KV-cache (Phase 17 · 11). Một request rơi vào decode pool đang giữ prefix của nó — nếu không khớp, nó sẽ chảy qua prefill → decode. Tỷ lệ hit và sự phân tách kết hợp với nhau — router nhận thức cache xác định liệu có cần thực hiện prefill mới hay không.

### MoE trên Blackwell là nơi có những con số thực tế

GB300 NVL72 + Dynamo cho thấy throughput MoE gấp 50x so với các baseline Hopper. MoE expert routing nặng về tính toán ở prefill nhưng nặng về bộ nhớ ở decode (expert caches), vì vậy phân tách là một chiến thắng kép. Việc serving các mô hình frontier năm 2026 chủ yếu là MoE (DeepSeek-V3, các biến thể GPT-5 tương lai).

### Những con số bạn cần nhớ

Các con số benchmark thay đổi — NVIDIA và stack inference công bố kết quả cập nhật hàng quý. Hãy kiểm tra lại trước khi trích dẫn.

- DeepSeek-R1 trên GB200 NVL72 + Dynamo: ~6x throughput so với baseline trong chế độ độ trễ trung bình (developer.nvidia.com, 2025-06); các tuyên bố cộng đồng "lên tới 30x" trên toàn bộ stack Blackwell + Dynamo là các tổng hợp định hướng không có nguồn chính thống đơn lẻ.
- GB300 NVL72 + Dynamo: throughput MoE lên tới 50x so với Hopper (developer.nvidia.com, không ghi ngày).
- Mốc tiết kiệm (tổng hợp nội bộ, không phải nghiên cứu điển hình đơn lẻ): $600-800K/year off a $2M chi phí hàng năm ở cùng mức SLA.
- Ngưỡng phân tách: prompt >512 token + output >200 token.
- Chuyển KV qua NIXL: 20-80 ms cho 4K-prompt KV trên 70B FP8.

```figure
prefill-decode-split
```

## Sử dụng

`code/main.py` mô phỏng serving tập trung so với phân tách. Báo cáo throughput, chi phí mỗi request, và điểm giao cắt độ dài prompt.

## Triển khai

Bài học này tạo ra `outputs/skill-disaggregation-decider.md`. Dựa trên workload và cluster, quyết định xem có nên phân tách hay không.

## Bài tập

1. Chạy `code/main.py`. Tại độ dài prompt nào thì phân tách vượt trội hơn tập trung?
2. Thiết kế prefill pool và decode pool cho một dịch vụ RAG với độ dài prefix P99 là 8K, output 300.
3. Dynamo vs llm-d: chọn một cho shop thuần Kubernetes không có ưu tiên runtime Python.
4. Tính chi phí chuyển KV: 4K prefill trên 70B FP8 = ~500 MB KV. Tại RDMA 100 GB/s, truyền = 5 ms. Tại TCP 10 GB/s = 50 ms. Điều nào quan trọng đối với SLA của bạn?
5. MoE expert routing thay đổi các mẫu truy cập KV. Phân tách hoạt động như thế nào với MoE kích hoạt các expert khác nhau cho mỗi token?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Disaggregated serving | "tách prefill/decode" | Các pool GPU riêng biệt cho mỗi giai đoạn |
| NIXL | "NVIDIA transport" | Truyền KV liên nút của Dynamo (RDMA/TCP) |
| NVIDIA Dynamo | "the orchestrator" | Trình điều phối stack-above cho vLLM/SGLang/TRT-LLM |
| llm-d | "Kubernetes native" | Stack phân tách K8s của Red Hat + AWS |
| Planner Profiler | "Dynamo auto-config" | Đo lường workload, cấu hình tỷ lệ pool |
| SLA Planner | "Dynamo policy" | Tự động khớp tỷ lệ prefill:decode để đáp ứng SLO |
| `packDomain: rack` | "llm-d topology" | Đóng gói prefill+decode trên cùng rack để KV nhanh |
| UCCL | "unified collective" | Lớp mạng llm-d 0.5 cho scale-to-zero |
| MoE expert routing | "expert per token" | Mẫu DeepSeek-V3; phân tách giúp ích |

## Đọc thêm

- [NVIDIA — Giới thiệu Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference trên Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [Blog về Disaggregated Serving của TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [GitHub của llm-d](https://github.com/llm-d/llm-d)
- [Ghi chú phát hành llm-d 0.5](https://github.com/llm-d/llm-d/releases)