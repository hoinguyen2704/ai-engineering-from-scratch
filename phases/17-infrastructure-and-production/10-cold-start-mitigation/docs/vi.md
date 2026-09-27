# Giảm thiểu Cold Start cho Serverless LLM

> Một image mô hình 20 GB mất từ 5-10 phút (đối với 7B) đến hơn 20 phút (đối với 70B) để chuyển từ trạng thái cold sang sẵn sàng phục vụ. Trong thế giới serverless thực thụ, đó không phải là khởi động — đó là sự cố ngừng hoạt động (outage). Các biện pháp giảm thiểu hoạt động ở năm lớp: node image được nạp sẵn (Bottlerocket trên AWS, kiến trúc dual-volume), model streaming (NVIDIA Run:ai Model Streamer, tích hợp sẵn trong vLLM), GPU memory snapshots (checkpoint của Modal, khởi động lại nhanh hơn tới 10 lần), warm pool (`min_workers=1`), tải theo tầng (pipeline NVMe→DRAM→HBM của ServerlessLLM, giảm độ trễ 10-200 lần), và di chuyển trực tiếp (live migration) bằng cách chuyển input tokens (KB) thay vì KV cache (GB). Modal công bố mức cold start tối thiểu là 2-4 giây; Baseten mặc định 5-10 giây, dưới 1 giây nếu có pre-warming. Bài học này dạy bạn cách đo lường, lập ngân sách và xếp chồng năm lớp này.

**Type:** Learn
**Languages:** Python (stdlib, toy cold-start path simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~60 phút

## Mục tiêu học tập

- Liệt kê năm lớp giảm thiểu cold-start và nêu tên một công cụ hoặc mô hình tại mỗi lớp.
- Tính toán tổng thời gian cold-start bằng tổng của (cấp phát node) + (tải trọng số) + (nạp trọng số vào HBM) + (khởi tạo engine) cho một mô hình 70B.
- Giải thích tại sao live migration chuyển input tokens (KB) thay vì KV cache (GB) và hình phạt (penalty) là gì (tính toán lại).
- Nêu sự đánh đổi của warm-pool (trả tiền cho GPU nhàn rỗi hoặc chấp nhận cold-start kéo dài) và ngưỡng SLA mà tại đó `min_workers > 0` trở thành bắt buộc.

## Vấn đề

Endpoint LLM serverless của bạn tự động scale về 0 vào ban đêm. Lúc 8 giờ sáng, lưu lượng truy cập tăng đột biến. Request đầu tiên phải chờ trong khi:

1. Karpenter cấp phát một GPU node: 45-60 giây.
2. Container kéo một image 30 GB chứa các trọng số: 120-300 giây.
3. Engine nạp trọng số vào HBM: 45-120 giây tùy thuộc vào kích thước mô hình và tốc độ lưu trữ.
4. vLLM hoặc TRT-LLM khởi tạo CUDA graphs, KV cache pool, tokenizer: 10-30 giây.

Tổng cộng: 220-510 giây (khoảng 3-8 phút) trước khi nhận lại được một token. SLA của bạn là 2 giây. Bạn triển khai warm-pool (`min_workers=1`) và vấn đề dường như biến mất — nhưng giờ đây bạn phải trả tiền cho một GPU nhàn rỗi 24/7. Nếu dịch vụ của bạn có 5 sản phẩm, mỗi sản phẩm có một warm replica, thì đó là 5 × 24 × 30 = 3.600 GPU-giờ/tháng bất kể có người dùng nào gọi hay không.

Giảm thiểu cold-start là cách để duy trì tính kinh tế của serverless trong khi vẫn đạt được độ trễ gần như luôn bật (always-on).

## Khái niệm

### Lớp 1 — Node image được nạp sẵn (Bottlerocket)

Trên AWS, kiến trúc dual-volume của Bottlerocket tách biệt OS khỏi dữ liệu. Hãy snapshot volume dữ liệu với container image đã được kéo sẵn; tham chiếu ID snapshot trong `EC2NodeClass` của bạn. Các node mới khởi động với trọng số đã có sẵn trên NVMe cục bộ — các bước 2 và một phần của bước 3 sẽ biến mất. Hoạt động nguyên bản với Karpenter. Mức tiết kiệm điển hình: 2-4 phút mỗi lần cold start cho các mô hình lớn.

Tương đương trên GCP: custom VM images với các container layer được nạp sẵn. Trên Azure: snapshot managed disk với mô hình tương tự.

### Lớp 2 — Model streaming (Run:ai Model Streamer)

Thay vì nạp toàn bộ file trước khi trả lời request đầu tiên, hãy stream các trọng số vào bộ nhớ GPU theo từng layer và bắt đầu xử lý ngay khi transformer block đầu tiên đã sẵn sàng. NVIDIA Run:ai Model Streamer được tích hợp sẵn trong vLLM 2026. Hoạt động với S3, GCS và NVMe cục bộ. Giảm thời gian nạp trọng số xuống khoảng một nửa đối với các mô hình lớn bằng cách chồng lấp I/O với quá trình thiết lập tính toán.

### Lớp 3 — GPU memory snapshots (Modal)

Modal thực hiện checkpoint trạng thái GPU (trọng số, CUDA graphs, vùng KV cache) sau lần nạp đầu tiên. Các lần khởi động lại sau đó sẽ deserialize trực tiếp vào HBM — nhanh hơn 10 lần so với việc khởi tạo lại. Đây là thứ gần nhất với việc "khởi động một GPU warm trong 2 giây". Đánh đổi: các snapshot phụ thuộc vào GPU-topology, vì vậy nếu Karpenter di chuyển bạn sang một SKU khác, bạn phải thực hiện checkpoint lại.

### Lớp 4 — Warm pools (min_workers=1)

Biện pháp giảm thiểu đơn giản nhất: giữ một replica luôn sẵn sàng. Chi phí là giá thuê GPU theo giờ 24/7. Phép tính này rất khắc nghiệt với các mô hình nhỏ (bạn trả $0.85-$1.50/giờ để tránh cold start 30 giây) và có lợi với các mô hình lớn (trả 4$/giờ để tránh cold start 5 phút). Ngưỡng SLA mà tại đó warm pool trở thành bắt buộc: thường là TTFT P99 < 60 giây trên mô hình 70B+.

### Lớp 5 — Tải theo tầng (ServerlessLLM)

ServerlessLLM coi lưu trữ là một hệ thống phân cấp: NVMe (nhanh nhưng lớn), DRAM (trung bình nhưng theo tầng), HBM (nhỏ nhưng tức thời). Trọng số được nạp sẵn vào DRAM; nạp theo yêu cầu vào HBM. Tài liệu báo cáo giảm độ trễ 10-200 lần khi tải cold so với cách nạp từ disk vào HBM thông thường. Việc áp dụng trong thực tế còn sớm nhưng đã có các tích hợp với vLLM.

### Lớp 6 — Live migration (mô hình bổ sung)

Khi một node không khả dụng (spot eviction, node drain), mô hình truyền thống là cold-start một replica khác và drain hàng đợi request. Live migration di chuyển các input tokens (kilobytes) đến đích đã nạp sẵn mô hình và tính toán lại KV cache tại đích đó. Việc tính toán lại rẻ hơn so với việc chuyển hàng GB KV cache qua mạng. Áp dụng cho các triển khai phân tán (disaggregated).

### Phép tính warm-pool

Đối với một dịch vụ có SLA P99 TTFT là 2 giây, câu hỏi không phải là "có hay không dùng warm pool" mà là "bao nhiêu warm replica, và đường dẫn nào cần chúng".

- Các đường dẫn tương tác giá trị cao (live chat, voice agent): `min_workers=1-2`.
- Các đường dẫn batch nền (phân loại hàng đêm): chấp nhận scale-to-zero, cold start 5-10 phút là có thể chấp nhận được.
- Gói cao cấp: `min_workers` cho mỗi khách hàng với dung lượng chuyên dụng.

### Đo lường trước khi tối ưu hóa

Giải phẫu cold-start cho mô hình 70B trên một node mới (minh họa):

| Giai đoạn | Thời gian | Biện pháp giảm thiểu |
|-----------|----------|----------------------|
| Cấp phát node | 50s | Bottlerocket + image nạp sẵn, warm pool |
| Kéo image | 180s | Volume dữ liệu nạp sẵn (loại bỏ) |
| Trọng số vào HBM | 75s | Model streamer (giảm một nửa); GPU snapshot (loại bỏ) |
| Khởi tạo engine | 20s | Persistent CUDA graph cache |
| Forward đầu tiên | 3s | Độ trễ vốn có tối thiểu |
| **Tổng cold** | **328s** | |
| **Tổng với giảm thiểu** | **~15s** | Giảm 22 lần |

### Các con số bạn nên nhớ

- Modal cold start: 2-4 giây (với GPU snapshots).
- Baseten default cold start: 5-10 giây; dưới 1 giây với pre-warming.
- Cold start 70B thô: 3-8 phút.
- Run:ai Model Streamer: tăng tốc nạp trọng số ~2 lần.
- ServerlessLLM tiered loading: giảm độ trễ 10-200 lần (theo số liệu tài liệu).

```figure
cold-start-pipeline
```

## Sử dụng

`code/main.py` mô hình hóa một đường dẫn cold-start có và không có từng biện pháp giảm thiểu. Báo cáo tổng thời gian cold-start, chi phí warm-pool và tỷ lệ request hòa vốn mà tại đó warm pool tự chi trả cho chính nó.

## Triển khai

Bài học này tạo ra `outputs/skill-cold-start-planner.md`. Dựa trên SLA, kích thước mô hình và hình thái lưu lượng, chọn các biện pháp giảm thiểu để xếp chồng lên nhau.

## Bài tập

1. Chạy `code/main.py`. Tính toán tỷ lệ request hòa vốn mà tại đó một warm replica rẻ hơn so với việc trả thuế cold-start thông qua việc drop thêm request tại SLO.
2. Bạn triển khai mô hình 13B với SLA P99 TTFT là 3 giây. Chọn ngăn xếp giảm thiểu tối thiểu (ít lớp nhất) để đạt được điều đó.
3. Việc nạp sẵn Bottlerocket loại bỏ việc kéo image nhưng trọng số vẫn nạp từ snapshot vào HBM. Tính toán thời gian thực cho mô hình 70B nếu NVMe hỗ trợ snapshot đọc ở tốc độ 7 GB/s.
4. Nhà cung cấp serverless của bạn cung cấp GPU snapshots (Modal) và nhóm của bạn từ chối vì "snapshots làm rò rỉ PII". Hãy tranh luận cả hai phía — rủi ro thực tế là gì, và biện pháp giảm thiểu là gì (ephemeral snapshots, mã hóa, cách ly namespace)?
5. Thiết kế chính sách warm-pool theo tầng: bao nhiêu warm replica cho người dùng trả phí, người dùng dùng thử và khối lượng công việc batch? Hãy trình bày phép tính.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|-----------|----------------------|------------------|
| Cold start | "khoảng dừng lớn" | Thời gian từ request đến token đầu tiên trên một replica mới |
| Warm pool | "tối thiểu luôn bật" | `min_workers >= 1` để giữ ít nhất một replica sẵn sàng |
| Pre-seeded image | "baked AMI" | Node image với trọng số container đã có sẵn |
| Bottlerocket | "AWS node OS" | OS tối ưu cho container của AWS với hỗ trợ snapshot dual-volume |
| Model streamer | "tải streaming" | Chồng lấp I/O trọng số với thiết lập tính toán |
| GPU snapshot | "checkpoint vào HBM" | Serialize trạng thái GPU sau khi nạp; deserialize khi khởi động lại |
| Tiered loading | "NVMe + DRAM + HBM" | Phân cấp các tầng lưu trữ; nạp theo yêu cầu |
| Live migration | "di chuyển tokens" | Chuyển input (KB), tính toán lại KV tại đích |
| `min_workers` | "warm replicas" | Số lượng keep-alive tối thiểu của serverless |
| Scale-to-zero | "serverless toàn diện" | Không tốn phí khi nhàn rỗi; chấp nhận toàn bộ thuế cold-start |

## Đọc thêm

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Các benchmark đã công bố và kiến trúc checkpoint của Modal.
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — Mô hình snapshot volume dữ liệu nạp sẵn.
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — Chồng lấp nạp trọng số với thiết lập tính toán.
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — Sách hướng dẫn pre-warming.
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — Thiết kế tải theo tầng.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Live migration cho các triển khai phân tán.