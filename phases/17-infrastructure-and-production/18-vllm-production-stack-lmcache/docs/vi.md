# Production Serving Stack — KV Offloading và Cache-Aware Routing

> Một production serving stack kết nối router, các engine và khả năng quan sát (observability) vào một triển khai Kubernetes duy nhất — và coi KV cache như một tài nguyên có thể nằm ngoài GPU. KV offloading trích xuất KV cache ra khỏi bộ nhớ GPU và tái sử dụng nó trên các truy vấn và engine (CPU DRAM, sau đó là disk/Ceph). Production-stack của vLLM là triển khai tham chiếu; LMCache là lớp offloading. vLLM 0.11.0 KV Offloading Connector (tháng 1 năm 2026) giúp quá trình này trở nên bất đồng bộ và có thể cắm vào thông qua Connector API (v0.9.0+). Đường dẫn offload thường được ẩn khỏi đường dẫn yêu cầu, mặc dù cache miss và việc quảng bá (promotion) có thể làm tăng độ trễ end-to-end. LMCache có giá trị ngay cả khi không có shared prefix — khi GPU hết slot KV, các yêu cầu bị preempt có thể được khôi phục từ CPU thay vì phải tính toán lại prefill. Các benchmark đã công bố trên 16x H100 (80GB HBM) trên 4 a3-highgpu-4g: khi KV cache vượt quá HBM, cả native CPU offload và LMCache đều cải thiện đáng kể throughput; ở mức KV footprint thấp, tất cả các cấu hình đều khớp với baseline với mức overhead nhỏ.

**Type:** Learn
**Languages:** Python (stdlib, toy KV-spill simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang/RadixAttention)
**Time:** ~60 phút

## Mục tiêu học tập

- Vẽ sơ đồ các lớp vLLM production-stack: router, engines, KV offload, observability.
- Giải thích KV Offloading Connector API (v0.9.0+) và cách đường dẫn bất đồng bộ trong bản 0.11.0 ẩn đi độ trễ offload.
- Định lượng thời điểm LMCache CPU-DRAM hữu ích (KV > HBM) so với khi nó gây thêm overhead (KV đủ nhỏ để vừa HBM).
- Lựa chọn giữa native vLLM CPU offload và LMCache connector dựa trên các ràng buộc triển khai.

## Vấn đề

Việc serving bằng vLLM của bạn cho thấy GPU đang ở mức 100% HBM với các sự kiện preemption xảy ra bất cứ khi nào concurrency tăng lên. Các yêu cầu bị đẩy ra (evicted), xếp hàng lại, và bạn phải re-prefill cùng một prompt 2K-token bốn lần trong một phút. Tài nguyên tính toán GPU bị lãng phí cho các lần prefill dư thừa; goodput thấp hơn nhiều so với throughput thô.

Việc thêm nhiều GPU tốn kém theo tuyến tính. Việc thêm HBM là không thể. Nhưng CPU DRAM lại rẻ — một socket có 512 GB+ với độ trễ kém hơn HBM nhiều bậc nhưng vẫn ổn cho KV cache "ấm tạm thời".

LMCache trích xuất KV cache vào CPU DRAM để các yêu cầu bị preempt có thể khôi phục nhanh chóng, và các prefix lặp lại trên các engine có thể chia sẻ cache mà không cần mỗi engine phải re-prefill.

## Khái niệm

### vLLM production-stack

`github.com/vllm-project/production-stack` là triển khai Kubernetes tham chiếu:

- **Router** — cache-aware (Phase 17 · 11). Tiêu thụ các sự kiện KV.
- **Engines** — các vLLM worker. Một worker cho mỗi GPU hoặc mỗi nhóm TP/PP.
- **KV cache offload** — triển khai LMCache hoặc connector native.
- **Observability** — Prometheus scrape, Grafana dashboards, OTel traces.
- **Control plane** — service discovery, cấu hình, rolling updates.

Được đóng gói dưới dạng Helm chart + operator.

### KV Offloading Connector API (v0.9.0+)

vLLM 0.9.0 đã giới thiệu Connector API cho các KV cache backend có thể cắm vào. Engine của bạn offload các block tới connector; connector lưu trữ chúng (RAM, disk, object storage, LMCache). Khi yêu cầu cần một block, connector sẽ tải nó trở lại.

vLLM 0.11.0 (tháng 1 năm 2026) bổ sung đường dẫn offload bất đồng bộ — việc offload có thể diễn ra trong nền để engine không bị chặn trong các trường hợp thông thường. Độ trễ và throughput end-to-end vẫn phụ thuộc vào hình thái workload, tỷ lệ KV cache hit và áp lực hệ thống; các ghi chú của vLLM lưu ý rằng offload bằng custom-kernel có thể làm giảm throughput ở tỷ lệ hit thấp và việc lập lịch bất đồng bộ có các vấn đề tương tác đã biết với speculative decoding.

### Native CPU offload so với LMCache

**Native vLLM CPU offload**: cục bộ tại engine. Lưu trữ các KV block trong host RAM. Triển khai nhanh, không tốn network hop. Không chia sẻ giữa các engine.

**LMCache connector**: quy mô cụm. Lưu trữ các block trong một LMCache server dùng chung (CPU DRAM + tầng Ceph/S3). Các block có thể truy cập bởi bất kỳ engine nào. Đã có benchmark trên 16x H100.

Chọn native khi một engine đơn lẻ gặp áp lực HBM. Chọn LMCache khi nhiều engine chia sẻ prefix (RAG với các system prompt chung, multi-tenant với các template dùng chung).

### Hành vi benchmark

Thử nghiệm trên 16x H100 (80 GB HBM) phân bổ trên 4 a3-highgpu-4g:

- KV footprint thấp (prompt ngắn, concurrency thấp): tất cả các cấu hình khớp với baseline, LMCache thêm ~3-5% overhead.
- Footprint trung bình: LMCache bắt đầu giúp ích cho việc tái sử dụng prefix giữa các engine.
- KV vượt quá HBM: cả native CPU offload và LMCache đều cải thiện đáng kể throughput; LMCache đạt mức tăng lớn hơn nhờ chia sẻ giữa các engine.

### Khi nào LMCache mang tính quyết định

- Serving đa người thuê (multi-tenant) nơi các system prompt được chia sẻ giữa các tenant.
- RAG nơi các đoạn tài liệu lặp lại giữa các truy vấn.
- Các biến thể fine-tuned (LoRA) trên cùng một base model nơi việc tái sử dụng KV của base model giúp cắt giảm công việc dư thừa.
- Workload có tần suất preemption cao: khôi phục từ CPU rẻ hơn so với re-prefill.

### Khi nào KHÔNG nên bật

- Áp lực HBM nhỏ — bạn phải trả phí overhead mà không nhận được lợi ích.
- Context ngắn (<1K token) — thời gian truyền tải > thời gian re-prefill.
- Workload đơn người thuê, đơn prompt — không có sự tái sử dụng để khai thác.

### Tích hợp với disaggregated serving

Phase 17 · 17 disaggregated serving + LMCache mang lại hiệu quả cộng hưởng: các KV transfer từ prefill pool sang decode pool sẽ nằm trong LMCache nếu không được sử dụng; các truy vấn tiếp theo sẽ lấy từ LMCache. Cache-aware router ở Phase 17 · 11 có thể định tuyến đến engine có cache cục bộ HOẶC cache chia sẻ qua LMCache khớp với yêu cầu.

### Các con số cần nhớ

- vLLM 0.9.0: Connector API được phát hành.
- vLLM 0.11.0 (tháng 1 năm 2026): đường dẫn offload bất đồng bộ; tác động đến độ trễ end-to-end phụ thuộc vào workload, tỷ lệ KV hit và áp lực hệ thống (không phải là sự đảm bảo tuyệt đối).
- Benchmark 16x H100: LMCache giúp ích khi KV footprint vượt quá HBM.
- Áp lực HBM nhỏ: 3-5% overhead mà không có lợi ích.

```figure
zero-sharding
```

## Sử dụng

`code/main.py` mô phỏng một workload có tần suất preemption cao với và không có LMCache. Báo cáo số lần re-prefill đã tránh được, mức tăng throughput và điểm hòa vốn của việc sử dụng HBM.

## Triển khai

Bài học này tạo ra `outputs/skill-vllm-stack-decider.md`. Dựa trên hình thái workload và triển khai vLLM, quyết định chọn native, LMCache hay không chọn gì cả.

## Bài tập

1. Chạy `code/main.py`. Tại mức sử dụng HBM nào thì LMCache bắt đầu mang lại hiệu quả?
2. Một tenant chia sẻ một system prompt 6K-token cho 200 truy vấn/giờ. Tính toán mức tiết kiệm dự kiến của LMCache cho mỗi tenant.
3. LMCache server là một điểm lỗi duy nhất (single point of failure). Hãy thiết kế chiến lược HA (replicas, fallback về native).
4. LMCache lưu trữ vào Ceph trên ổ đĩa cơ học. Đối với 4K-token KV ở 70B FP8 (500 MB), thời gian đọc so với re-prefill là bao nhiêu?
5. Hãy lập luận liệu đường dẫn bất đồng bộ của vLLM 0.11.0 có thực sự "miễn phí" không — overhead ẩn ở đâu?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|-----------|----------------------|------------------|
| Production-stack | "triển khai tham chiếu" | Kubernetes Helm chart + operator của vLLM |
| Connector API | "giao diện KV backend" | Giao diện KV store có thể cắm vào của vLLM 0.9.0+ |
| Native CPU offload | "tràn bộ nhớ cục bộ" | Lưu trữ KV trong host RAM của cùng một engine |
| LMCache | "KV cache cụm" | Server KV cache liên engine trên CPU DRAM + disk |
| 0.11.0 async | "offload không chặn" | Offload được ẩn phía sau luồng engine |
| Preemption | "đẩy ra để lấy chỗ" | Xáo trộn KV cache khi HBM đầy |
| Prefix reuse | "cùng system prompt" | Nhiều truy vấn chia sẻ phần đầu; cache hit |
| Ceph tier | "tầng đĩa" | Lưu trữ bền vững bên dưới DRAM trong phân cấp cache |

## Đọc thêm

- [vLLM Blog — KV Offloading Connector (tháng 1 năm 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator.
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Triển khai Connector.
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — chi tiết về đường dẫn bất đồng bộ.