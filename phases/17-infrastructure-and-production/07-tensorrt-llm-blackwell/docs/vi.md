# Hardware-Specialized Inference Compilation — FP8 và NVFP4 trên Blackwell

> Biên dịch suy luận chuyên biệt cho phần cứng đánh đổi tính di động để lấy thông lượng (throughput), và TensorRT-LLM — chỉ dành riêng cho NVIDIA, được tinh chỉnh cho Blackwell — là ví dụ rõ ràng nhất cho thấy sự đánh đổi này mang lại hiệu quả. Trên GB200 NVL72 với Dynamo orchestration, SemiAnalysis InferenceX đã đo được $0.012 per million tokens on a 120B model in Q1-Q2 2026, against $0.09/M trên H100 + vLLM — một khoảng cách kinh tế gấp 7 lần. Stack này là sự kết hợp của ba chế độ dấu phẩy động: FP8 vẫn rất quan trọng đối với KV cache và các attention kernel vì nó có dải động (dynamic range) cần thiết; NVFP4 (4-bit microscaling) xử lý trọng số (weights) và kích hoạt (activations); multi-token prediction (MTP) và disaggregated prefill/decode bổ sung thêm 2-3 lần hiệu năng nữa. Hỗ trợ model Day-0 tải trực tiếp trọng số FP4 mà không cần chuyển đổi sau huấn luyện (post-training conversion). Điểm cần lưu ý cho các đội ngũ kỹ thuật năm 2026: TRT-LLM là mã nguồn mở nhưng dành riêng cho NVIDIA — chuyên biệt cho CUDA và Blackwell — vì vậy việc áp dụng nó đồng nghĩa với việc đánh đổi tính di động để lấy thông lượng. Hãy tính toán kỹ lưỡng trên tập hợp các model và phần cứng của bạn trước khi quyết định.

**Type:** Learn
**Languages:** Python (stdlib, toy FP8/NVFP4 memory and cost calculator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 10 · 13 (Quantization)
**Time:** ~75 phút

## Mục tiêu học tập

- Giải thích lý do tại sao FP8 vẫn rất quan trọng đối với KV cache và attention ngay cả khi trọng số ở định dạng NVFP4.
- Tính toán dung lượng HBM cần thiết cho một frontier model ở định dạng BF16, FP8 và NVFP4, đồng thời lý giải nguồn gốc của việc tiết kiệm tài nguyên.
- Liệt kê các tính năng chuyên biệt của Blackwell mà TRT-LLM khai thác (FP4 day-0, MTP, disaggregated serving, các primitive all-to-all).
- Quyết định khi nào việc bị khóa vào hệ sinh thái NVIDIA của TRT-LLM xứng đáng với khoảng cách chi phí gấp 7 lần so với vLLM trên Hopper.

## Vấn đề

Biên giới của kinh tế học suy luận (inference economics) vào năm 2026 là "bao nhiêu token trên mỗi đô la". Câu trả lời phụ thuộc vào bốn lựa chọn xếp chồng: thế hệ phần cứng (Hopper H100/H200 so với Blackwell B200/GB200), độ chính xác (BF16 → FP8 → NVFP4), serving engine (vLLM so với SGLang so với TRT-LLM), và orchestration (cơ bản so với disaggregated so với Dynamo).

Trên Hopper với vLLM, một model MoE 120B chạy ở mức ~$0.09 per million tokens. On Blackwell with TRT-LLM + Dynamo, the same model runs at ~$0.012 — rẻ hơn 7 lần. Một phần của khoảng cách đó đến từ phần cứng (Blackwell có thông lượng LLM trên mỗi GPU cao gấp 11-15 lần so với Hopper). Một phần đến từ stack: trọng số FP4, MTP draft, disaggregated prefill/decode, và NVLink 5 all-to-all cho việc giao tiếp giữa các chuyên gia (expert) trong MoE.

Bạn không thể sao chép điều này bên ngoài stack của NVIDIA. Đó là sự đánh đổi — tính di động để lấy hiệu quả kinh tế. Hiểu được lựa chọn stack nào đóng góp bao nhiêu phần trăm vào khoảng cách này chính là mục tiêu của bài học này.

## Khái niệm

### Tại sao FP8 vẫn là tiêu chuẩn tối thiểu cho KV cache

Một sai lầm phổ biến vào năm 2026: cho rằng NVFP4 áp dụng cho mọi nơi. Thực tế không phải vậy. KV cache cần FP8 (8-bit floating point) vì nó lưu trữ các khóa (keys) và giá trị (values) của attention, vốn trải rộng trên một dải động lớn. Việc lượng tử hóa KV xuống FP4 gây ra mất mát độ chính xác nghiêm trọng — phần đuôi của phân phối bị mất và các điểm số attention bị sụp đổ. Các bit số mũ (exponent bits) của FP8 cung cấp cho KV cache dải động mà nó cần.

NVFP4 (2025-2026) áp dụng cho trọng số và kích hoạt. Microscaling: mỗi khối trọng số có hệ số tỷ lệ riêng để các khối nhỏ có thể trải rộng trên các dải động khác nhau mà không bị mất mát do tỷ lệ trên toàn tensor. Đối với kích hoạt, FP4 vẫn ổn vì các kích hoạt có dải giá trị nhỏ trong một lớp.

Cấu hình Blackwell điển hình:

- Trọng số: NVFP4 (4-bit microscaling).
- Kích hoạt: NVFP4.
- KV cache: FP8.
- Attention accumulator: FP32 (đảm bảo tính ổn định của softmax).

### Các primitive chuyên biệt của Blackwell mà TRT-LLM sử dụng

- **Trọng số FP4 Day-0**: các nhà cung cấp model phân phối trọng số FP4 trực tiếp; TRT-LLM tải mà không cần chuyển đổi sau huấn luyện. Không cần bước AWQ / GPTQ cho FP4.
- **Multi-token prediction (MTP)**: ý tưởng tương tự như EAGLE (Phase 17 · 05) nhưng được tích hợp vào bản build của TRT-LLM.
- **Disaggregated serving**: prefill và decode trên các pool GPU riêng biệt, KV cache được truyền qua NVLink hoặc InfiniBand. Ý tưởng tương tự như Dynamo (Phase 17 · 20).
- **All-to-all communication primitives**: NVLink 5 cắt giảm độ trễ giao tiếp giữa các chuyên gia MoE xuống 3 lần so với Hopper. Các MoE kernel của TRT-LLM được tinh chỉnh cho điều này.
- **NVFP4 + MXFP8 microscaling**: xử lý hệ số tỷ lệ được tăng tốc phần cứng trên Blackwell Tensor Cores.

### Các con số bạn nên ghi nhớ

- HGX B200 ở mức $0.02/M token trên GPT-OSS-120B thông qua TRT-LLM.
- GB200 NVL72 ở mức $0.012/M token thông qua Dynamo (điều phối TRT-LLM).
- H100 + vLLM ≈ $0.09/M token trên khối lượng công việc tương đương.
- Tăng 2.8 lần thông lượng trong ba tháng cập nhật TRT-LLM (2026).
- Thông lượng LLM trên mỗi GPU gấp 11-15 lần, Blackwell so với Hopper.
- MLPerf Inference v6.0 (Tháng 4 năm 2026): Blackwell thống trị mọi tác vụ được gửi.

### Cái giá thực sự của FP4 đối với chất lượng

NVFP4 rất mạnh tay. Trên các khối lượng công việc đòi hỏi suy luận cao (chain-of-thought, toán học, tạo mã với ngữ cảnh dài), trọng số FP4 làm giảm chất lượng rõ rệt. Hiệu chuẩn theo khối (per-block calibration) giúp giảm thiểu nhưng không loại bỏ hoàn toàn. Các đội ngũ triển khai model suy luận thường sử dụng trọng số FP8 + kích hoạt FP4 như một sự thỏa hiệp, hoặc gắn bó với H200 cùng FP8 toàn diện.

Quy tắc: luôn xác thực chất lượng tác vụ trên tập eval của bạn trước khi cam kết sử dụng trọng số NVFP4.

### Tại sao đây là quyết định bị khóa vào NVIDIA

TRT-LLM là C++ + CUDA + các kernel đóng. Các model cần được biên dịch cho một SKU GPU cụ thể. Không hỗ trợ AMD, không Intel, không ARM. Nếu chiến lược hạ tầng của bạn là đa nhà cung cấp, TRT-LLM không phải là lựa chọn cho tầng phục vụ bằng TRT-LLM — bạn vẫn có thể phục vụ từ vLLM trên phần cứng hỗn hợp. Nếu bạn chỉ dùng NVIDIA, khoảng cách 7 lần sẽ bù đắp cho sự ràng buộc này.

### Công thức thực tế năm 2026

Đối với hóa đơn suy luận hàng năm trên 100 triệu USD, việc chạy trên Hopper + vLLM sẽ lãng phí 7-10 lần chi phí. Hãy chuyển các khối lượng công việc chiếm tỷ trọng chi phí lớn sang Blackwell + TRT-LLM + Dynamo. Giữ tầng thử nghiệm trên H100 + vLLM để đảm bảo tốc độ lặp lại model. Xác thực chất lượng trên mỗi model đã chuyển đổi sang NVFP4 trước khi đưa vào sản xuất.

### Lợi ích của disaggregation

Disaggregated serving của TRT-LLM (các pool prefill và decode riêng biệt) được đề cập chi tiết trong Phase 17 · 20. Trên Blackwell, các hệ số nhân xếp chồng lên nhau: trọng số FP4 × tốc độ MTP × vị trí disaggregated × định tuyến nhận biết cache. Con số 7 lần giả định toàn bộ stack này.

```figure
pipeline-parallel
```

## Sử dụng

`code/main.py` tính toán dung lượng HBM, thông lượng decode (trong chế độ giới hạn bộ nhớ), và $/M-token cho một model trên ba stack: H100 + BF16 + vLLM, H100 + FP8 + vLLM, B200 + NVFP4/FP8 + TRT-LLM. Hãy chạy nó để thấy hiệu ứng cộng hưởng và tỷ trọng đóng góp của mỗi thay đổi vào khoảng cách chi phí.

## Triển khai

Bài học này tạo ra `outputs/skill-trtllm-blackwell-advisor.md`. Với một khối lượng công việc, kích thước model và khối lượng token hàng năm, nó quyết định xem stack Blackwell + TRT-LLM có xứng đáng với việc bị khóa vào NVIDIA hay không.

## Bài tập

1. Chạy `code/main.py`. Trên một model MoE 120B với 30% tham số hoạt động, hãy tính toán thông lượng decode bị giới hạn bởi băng thông bộ nhớ trên H100 BF16, H100 FP8 và B200 NVFP4/FP8. Bước nhảy lớn nhất đến từ đâu?
2. Một khách hàng chi 2 triệu USD/năm cho H100 + vLLM. Số lượng GPU Blackwell hòa vốn cần mua để khấu hao việc chuyển đổi sang TRT-LLM trong 12 tháng là bao nhiêu, với khoảng cách kinh tế gấp 7 lần?
3. Bạn thấy độ chính xác giảm 3 điểm trên MATH sau khi chuyển đổi trọng số NVFP4. Hãy nêu hai lộ trình phục hồi: một ưu tiên chất lượng (giữ trọng số FP8), một ưu tiên chi phí (hiệu chuẩn với dữ liệu trong miền).
4. Đọc kết quả suy luận MLPerf v6.0. Tác vụ nào có khoảng cách Blackwell-so-với-Hopper nhỏ nhất, và tại sao?
5. Tính toán HBM cần thiết cho một model 405B với trọng số NVFP4 + KV cache FP8 ở ngữ cảnh 128k. Nó có vừa trên một node GB200 NVL72 duy nhất không?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| FP8 | "eight-bit float" | Dấu phẩy động 8-bit; dùng cho KV cache và attention do dải động |
| NVFP4 | "four-bit micro" | Định dạng FP microscaling 4-bit của NVIDIA; trọng số và kích hoạt trên Blackwell |
| MXFP8 | "MX eight" | Biến thể Microscaling FP8; tăng tốc phần cứng trên Blackwell Tensor Cores |
| Day-0 FP4 | "ship FP4 weights" | Nhà cung cấp model phát hành trọng số đã ở định dạng FP4; không cần bước chuyển đổi sau huấn luyện |
| MTP | "multi-token prediction" | Draft suy luận suy đoán (speculative-decoding) tích hợp của TRT-LLM (Phase 17 · 05) |
| Disaggregated serving | "split prefill/decode" | Prefill và decode trên các pool GPU riêng biệt; KV được truyền qua NVLink/IB |
| All-to-all | "MoE expert comm" | Mô hình giao tiếp định tuyến token đến các GPU chuyên gia; NVLink 5 cắt giảm 3 lần |
| InferenceX | "SemiAnalysis inference bench" | Benchmark chi phí trên mỗi token được chấp nhận trong ngành năm 2026 |

## Đọc thêm

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — Kết quả MLPerf tháng 4 năm 2026.
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all và các MoE kernel.
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — tài liệu engine chính thức.
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — disaggregated orchestration phía trên TRT-LLM.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — bộ benchmark công bố các con số của Blackwell.