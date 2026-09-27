# Tự động mở rộng GPU trên Kubernetes — Karpenter, KAI Scheduler, Gang Scheduling

> Ba lớp, không phải một. Karpenter cung cấp node một cách linh hoạt (dưới một phút, nhanh hơn 40% so với Cluster Autoscaler). KAI Scheduler xử lý gang scheduling, nhận thức cấu trúc liên kết (topology awareness) và các hàng đợi phân cấp — nó ngăn chặn bẫy phân bổ một phần 7-trên-8, nơi bảy node chờ đợi và lãng phí tài nguyên vì thiếu một GPU. Các bộ tự động mở rộng ở cấp ứng dụng (NVIDIA Dynamo Planner, llm-d Workload Variant Autoscaler) mở rộng dựa trên các tín hiệu cụ thể của suy luận (inference) — độ sâu hàng đợi, mức sử dụng KV cache — chứ không phải chu kỳ hoạt động (duty cycle) của CPU/DCGM. Bẫy HPA cổ điển là `DCGM_FI_DEV_GPU_UTIL` là một phép đo chu kỳ hoạt động: 100% có thể là 10 yêu cầu hoặc 100. vLLM phân bổ trước bộ nhớ KV cache, vì vậy bộ nhớ không bao giờ kích hoạt việc giảm quy mô (scale-down). Bài học này dạy bạn cách kết hợp ba lớp và tránh chính sách `WhenEmptyOrUnderutilized` mặc định của Karpenter, vốn sẽ chấm dứt các tác vụ GPU đang chạy giữa chừng khi đang suy luận.

**Type:** Learn
**Languages:** Python (stdlib, toy queue-depth autoscaler simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (Serving Engine Internals)
**Time:** ~75 minutes

## Mục tiêu học tập

- Vẽ sơ đồ ba lớp tự động mở rộng (cung cấp node, gang scheduling, cấp ứng dụng) và nêu tên công cụ được sử dụng ở mỗi lớp.
- Giải thích tại sao `DCGM_FI_DEV_GPU_UTIL` là tín hiệu HPA sai cho vLLM và nêu tên hai giải pháp thay thế (độ sâu hàng đợi, mức sử dụng KV cache).
- Mô tả gang scheduling và chế độ lỗi phân bổ một phần mà KAI Scheduler ngăn chặn (7 trên 8 GPU nhàn rỗi).
- Nêu tên chính sách hợp nhất của Karpenter (`WhenEmptyOrUnderutilized`) gây chấm dứt các tác vụ GPU đang chạy và nêu giải pháp thay thế an toàn cho năm 2026.

## Vấn đề

Nhóm của bạn vận hành dịch vụ phục vụ LLM trên Kubernetes. Bạn thiết lập HPA với `DCGM_FI_DEV_GPU_UTIL` làm tín hiệu. Dịch vụ đạt mức sử dụng 100% trong giờ cao điểm. HPA không bao giờ mở rộng quy mô — nó cho rằng bạn đã đầy tải. Bạn thêm một bản sao (replica) theo cách thủ công; TTFT giảm xuống. HPA vẫn không mở rộng. Tín hiệu đang đánh lừa bạn.

Mặt khác, bạn sử dụng Cluster Autoscaler cho các node. Một prompt 1 triệu token đến lúc 2 giờ sáng; cụm (cluster) mất 3 phút để cung cấp một node và yêu cầu bị quá thời gian (timeout).

Lại một trường hợp khác, bạn triển khai mô hình 70B yêu cầu 8 GPU trên 2 node. Cụm có 7 GPU trống và 1 GPU nằm rải rác trên 3 node. Cluster Autoscaler cung cấp một node cho 1 GPU còn thiếu. Bảy node chờ đợi 4 phút trong khi lãng phí tiền bạc để Kubernetes khởi động GPU cuối cùng.

Ba lớp, ba chế độ lỗi khác nhau. Tự động mở rộng nhận thức GPU vào năm 2026 không phải là "bật HPA lên". Đó là việc kết hợp cung cấp node, gang scheduling và tự động mở rộng theo tín hiệu ứng dụng.

## Khái niệm

### Lớp 1 — cung cấp node (Karpenter)

Karpenter theo dõi các pod đang chờ và cung cấp node trong khoảng ~45-60 giây (Cluster Autoscaler thường mất 90-120 giây cho các node GPU). Nó chọn các loại instance một cách linh hoạt theo ràng buộc `NodePool` — nếu pod của bạn cần 8 H100 và cụm không có node phù hợp, Karpenter sẽ cung cấp trực tiếp một node thay vì mở rộng một nhóm hiện có.

**Bẫy hợp nhất (consolidation trap)**: `consolidationPolicy: WhenEmptyOrUnderutilized` mặc định của Karpenter rất nguy hiểm cho các nhóm GPU. Nó sẽ chấm dứt một node GPU đang chạy để di chuyển các pod sang một instance có kích thước phù hợp và rẻ hơn. Đối với các workload suy luận, điều đó có nghĩa là trục xuất các yêu cầu đang chạy và tải lại mô hình 70B trên node mới. Tổn thất là hàng phút công suất cộng với các yêu cầu bị lỗi.

Cài đặt an toàn cho các nhóm GPU:

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

Cho phép Karpenter hợp nhất các node thực sự trống sau một giờ nhưng không bao giờ trục xuất một tác vụ đang chạy.

### Lớp 2 — gang scheduling (KAI Scheduler)

KAI Scheduler (dự án "Karp" sau đó được đổi tên) xử lý những gì kube-scheduler mặc định không làm được:

**Gang scheduling** — lập lịch tất cả hoặc không có gì. Một pod suy luận phân tán yêu cầu 8 GPU, hoặc cả 8 cùng bắt đầu hoặc không có gì cả. Nếu không có điều này, bạn sẽ gặp bẫy phân bổ một phần: 7 trong 8 pod bắt đầu, chờ đợi vô thời hạn, lãng phí tiền bạc.

**Nhận thức cấu trúc liên kết (Topology awareness)** — biết GPU nào chia sẻ NVLink, GPU nào nằm trên cùng một rack, GPU nào có InfiniBand giữa chúng. Đặt các pod cho phù hợp. Một workload song song tensor DeepSeek-V3 67B phải nằm trong một miền NVLink; KAI Scheduler tôn trọng điều đó.

**Hàng đợi phân cấp (Hierarchical queues)** — nhiều nhóm cạnh tranh cho cùng một nhóm GPU với mức độ ưu tiên và hạn ngạch. Việc ưu tiên của Nhóm A có thể bị chiếm quyền bởi tác vụ huấn luyện của Nhóm B chỉ khi các quy tắc ưu tiên cho phép.

KAI được triển khai cùng với kube-scheduler như một bộ lập lịch phụ; bạn chú thích (annotate) các workload để sử dụng nó. Cả Ray và vLLM production-stack đều tích hợp.

### Lớp 3 — tín hiệu cấp ứng dụng

**Bẫy HPA**: `DCGM_FI_DEV_GPU_UTIL` là một chỉ số chu kỳ hoạt động — nó đo lường xem GPU có đang thực hiện công việc tại mỗi khoảng thời gian lấy mẫu hay không. Mức sử dụng 100% có thể có nghĩa là 10 yêu cầu đồng thời hoặc 100; GPU vẫn bận rộn như nhau. Mở rộng dựa trên chu kỳ hoạt động là mở rộng một cách mù quáng.

Tệ hơn nữa, vLLM và các engine tương tự phân bổ trước bộ nhớ KV cache (lên đến `--gpu-memory-utilization`). Mức sử dụng bộ nhớ duy trì gần 90% ngay cả khi chỉ có một yêu cầu. HPA dựa trên bộ nhớ không bao giờ giảm quy mô.

**Các tín hiệu thay thế năm 2026**:

- Độ sâu hàng đợi (số lượng yêu cầu đang chờ prefill).
- Mức sử dụng KV cache (phần trăm các khối được phân bổ cho các chuỗi đang hoạt động).
- P99 TTFT trên mỗi bản sao (tín hiệu SLA của bạn).
- Goodput (số yêu cầu đáp ứng tất cả SLO mỗi giây).

NVIDIA Dynamo Planner và llm-d Workload Variant Autoscaler tiêu thụ các tín hiệu này và mở rộng các bản sao. Chúng thay thế hoàn toàn HPA cho việc phục vụ LLM.

### Khi nào sử dụng cái gì

| Quyết định mở rộng | Công cụ |
|----------------|------|
| Thêm/xóa node | Karpenter |
| Lập lịch tác vụ đa GPU | KAI Scheduler |
| Thêm/xóa bản sao | Dynamo Planner / llm-d WVA (hoặc HPA tùy chỉnh dựa trên độ sâu hàng đợi) |
| Chọn loại GPU | Karpenter NodePool |
| Chiếm quyền ưu tiên thấp | KAI Scheduler queues |

### Prefill/decode phân tách làm mọi thứ phức tạp hơn

Nếu bạn chạy prefill/decode phân tách (Phase 17 · 17), bạn có hai lớp pod với các trình kích hoạt mở rộng khác nhau: các pod prefill mở rộng dựa trên độ sâu hàng đợi, các pod decode mở rộng dựa trên áp lực KV cache. llm-d hiển thị chúng dưới dạng các `Services` riêng biệt với HPA theo vai trò. Đừng cố gắng đặt một HPA duy nhất cho cả hai.

### Cold start cũng quan trọng ở đây

Giảm thiểu cold-start (Phase 17 · 10) là nơi thời gian cung cấp node trở nên hiển thị với người dùng. Thời gian khởi động 45-60 giây của Karpenter cộng với việc tải mô hình 20GB và khởi tạo engine có nghĩa là một yêu cầu từ con số không mất 2-5 phút. Hãy giữ một nhóm ấm (`min_workers=1`) cho các đường dẫn quan trọng đối với SLO, hoặc sử dụng checkpointing kiểu Modal ở lớp ứng dụng.

### Các con số bạn nên nhớ

- Cung cấp node Karpenter: ~45-60s so với Cluster Autoscaler ~90-120s (node GPU).
- KAI Scheduler ngăn chặn lãng phí phân bổ một phần — bẫy 7-trên-8.
- `DCGM_FI_DEV_GPU_UTIL` làm tín hiệu HPA: bị lỗi; hãy sử dụng độ sâu hàng đợi hoặc mức sử dụng KV.
- Karpenter `WhenEmptyOrUnderutilized`: chấm dứt các tác vụ GPU đang chạy. Sử dụng `WhenEmpty + consolidateAfter: 1h` cho suy luận.

```figure
autoscaling
```

## Sử dụng nó

`code/main.py` mô phỏng một bộ tự động mở rộng ba lớp trên một workload GPU bùng nổ. So sánh HPA ngây thơ (chu kỳ hoạt động), HPA độ sâu hàng đợi và mở rộng theo gang-scheduled của KAI. Báo cáo các yêu cầu chưa được đáp ứng, số phút GPU nhàn rỗi và điểm số tổng hợp.

## Triển khai nó

Bài học này tạo ra `outputs/skill-gpu-autoscaler-plan.md`. Dựa trên cấu trúc liên kết cụm, hình dạng workload và SLO, nó thiết kế một kế hoạch tự động mở rộng ba lớp.

## Bài tập

1. Chạy `code/main.py`. Dưới một workload bùng nổ, HPA chu kỳ hoạt động ngây thơ làm rơi bao nhiêu yêu cầu mà HPA độ sâu hàng đợi bắt được? Sự khác biệt đến từ đâu?
2. Thiết kế một Karpenter NodePool cho một cụm phục vụ Llama 3.3 70B FP8 trên H100 SXM5. Chỉ định `capacity-type`, `disruption.consolidationPolicy`, `consolidateAfter` và một taint để giữ các workload không phải GPU khỏi các node này.
3. Nhóm của bạn báo cáo rằng các triển khai bị kẹt ở trạng thái Pending vì "GPU khả dụng nhưng pod không lập lịch". Hãy chẩn đoán — đó là Karpenter, kube-scheduler hay KAI Scheduler? Những chỉ số nào xác nhận điều đó?
4. Chọn một tín hiệu để tự động mở rộng các pod prefill phân tách và một tín hiệu khác cho các pod decode. Hãy biện minh cho cả hai.
5. Tính toán chi phí của bẫy hợp nhất `WhenEmptyOrUnderutilized` trên một dịch vụ sản xuất 24x7 trung bình có 60 sự kiện làm rơi yêu cầu mỗi ngày tại P99 TTFT > 10s.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Karpenter | "bộ cung cấp node" | Bộ tự động mở rộng node Kubernetes; cung cấp dưới một phút |
| Cluster Autoscaler | "bộ mở rộng cũ" | Tiền thân của bộ tự động mở rộng node Kubernetes; chậm hơn, dựa trên nhóm |
| KAI Scheduler | "bộ lập lịch GPU" | Bộ lập lịch phụ cho gang + topology + hàng đợi |
| Gang scheduling | "tất cả hoặc không có gì" | Lập lịch N pod nguyên tử hoặc hoãn tất cả chúng |
| Topology awareness | "nhận thức rack" | Đặt pod dựa trên vị trí NVLink/IB/rack |
| `DCGM_FI_DEV_GPU_UTIL` | "mức sử dụng GPU" | Chỉ số chu kỳ hoạt động; KHÔNG phải tín hiệu mở rộng cho LLM |
| Queue depth | "yêu cầu đang chờ" | Tín hiệu HPA chính xác cho mở rộng dựa trên prefill |
| KV cache utilization | "áp lực bộ nhớ" | Tín hiệu HPA chính xác cho mở rộng dựa trên decode |
| Consolidation | "hợp nhất Karpenter" | Chấm dứt node để chuyển sang loại instance rẻ hơn |
| `WhenEmpty + 1h` | "hợp nhất an toàn" | Chính sách không trục xuất các tác vụ GPU đang chạy |

## Đọc thêm

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — tài liệu thiết kế và ví dụ cấu hình.
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — ngữ nghĩa chính sách hợp nhất và các mặc định an toàn cho GPU.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — các tín hiệu mở rộng của Dynamo Planner.
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — mô hình tích hợp Ray.
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — hướng dẫn cụ thể cho Kubernetes được quản lý.
- [llm-d GitHub](https://github.com/llm-d/llm-d) — thiết kế Workload Variant Autoscaler.