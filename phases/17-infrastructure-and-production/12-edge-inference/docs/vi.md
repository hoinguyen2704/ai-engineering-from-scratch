# Edge Inference — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> Ràng buộc cốt lõi của edge inference là băng thông bộ nhớ, không phải năng lực tính toán. Mobile DRAM đạt mức 50-90 GB/s; trong khi datacenter HBM3 đạt 2-3 TB/s — một khoảng cách 30-50 lần. Quá trình giải mã (decode) bị giới hạn bởi bộ nhớ nên khoảng cách này mang tính quyết định. Năm 2026, thị trường phân hóa thành bốn hướng. Apple M4/A18 Neural Engine đạt đỉnh 38 TOPS với bộ nhớ thống nhất (không cần copy dữ liệu giữa CPU↔NPU). Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon đạt 45 TOPS. WebGPU + WebLLM chạy Llama 3.1 8B (Q4) ở mức ~41 tok/s trên M3 Max (khoảng 70-80% so với native); 17.6k sao trên GitHub, API tương thích OpenAI, độ phủ di động ~70-75%. NVIDIA Jetson Orin Nano Super (8GB) phù hợp với Llama 3.2 3B / Phi-3; AGX Orin chạy gpt-oss-20b qua vLLM ở mức ~40 tok/s; Jetson T4000 (JetPack 7.1) mạnh gấp 2 lần AGX Orin. TensorRT Edge-LLM hỗ trợ EAGLE-3, NVFP4, chunked prefill — được trình diễn tại CES 2026 bởi Bosch, ThunderSoft, MediaTek.

**Type:** Learn
**Languages:** Python (stdlib, toy bandwidth-bound decode simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 09 (Production Quantization)
**Time:** ~60 phút

## Mục tiêu học tập

- Giải thích lý do tại sao inference LLM trên thiết bị di động bị giới hạn bởi băng thông bộ nhớ và năng lực tính toán chỉ là yếu tố phụ.
- Liệt kê bốn mục tiêu edge (Apple ANE, Qualcomm Hexagon, WebGPU/WebLLM, NVIDIA Jetson) và khớp từng mục tiêu với một trường hợp sử dụng.
- Nêu tên khoảng cách về độ phủ của WebGPU năm 2026 (Firefox Android đang bắt kịp) và sự xuất hiện trên Safari iOS 26.
- Chọn định dạng lượng tử hóa cho từng mục tiêu (Core ML INT4 + FP16 cho ANE, QNN INT8/INT4 cho Hexagon, WebGPU Q4 cho trình duyệt, NVFP4 cho Jetson Thor).

## Vấn đề

Một khách hàng muốn có chatbot trên thiết bị: ưu tiên giọng nói, mặc định riêng tư, hoạt động ngoại tuyến. Trên MacBook Pro M3 Max, Llama 3.1 8B Q4 chạy ở mức ~55 tok/s — ổn. Trên iPhone 16 Pro, cùng mô hình đó chạy ở mức 3 tok/s — không ổn. Trên điện thoại Android tầm trung với Snapdragon 8 Gen 3, đạt 7 tok/s. Trong trình duyệt qua WebGPU trên Chrome Android v121+, đạt 4-8 tok/s tùy thiết bị.

Sự chênh lệch về thông lượng không phải là vấn đề porting. Đó là sự kết hợp giữa khoảng cách băng thông, định dạng lượng tử hóa và khả năng truy cập NPU từ user-space. Edge inference vào năm 2026 là bốn bài toán khác nhau với bốn giải pháp khác nhau.

## Khái niệm

### Băng thông là giới hạn thực sự

Quá trình giải mã đọc toàn bộ tập trọng số cho mỗi token. Một mô hình 7B ở định dạng Q4 có dung lượng 3.5 GB. Đọc 3.5 GB ở tốc độ 50 GB/s mất 70 ms — giới hạn lý thuyết là ~14 tok/s. Ở mức 90 GB/s (mobile DRAM cao cấp), giới hạn tăng lên ~25 tok/s. Không có năng lực tính toán nào giúp cải thiện con số này.

Datacenter HBM3 ở mức 3 TB/s đọc cùng 3.5 GB trong 1.2 ms — giới hạn là 830 tok/s. Cùng mô hình, cùng trọng số. Hệ thống bộ nhớ khác nhau.

### Apple Neural Engine (M4 / A18)

- Lên đến 38 TOPS. Bộ nhớ thống nhất (CPU và ANE dùng chung một vùng nhớ) — không tốn chi phí copy.
- Truy cập qua Core ML + `.mlmodel` compiled models, hoặc qua Metal Performance Shaders (MPS) thông qua PyTorch.
- Llama.cpp Metal backend sử dụng MPS, không trực tiếp dùng ANE; ANE native yêu cầu chuyển đổi sang Core ML.
- Con đường thực tế tốt nhất cho ứng dụng iOS năm 2026: Core ML với trọng số INT4 + activations FP16.

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- Lên đến 45 TOPS. Tích hợp với CPU và GPU trong SoC nhưng có vùng nhớ riêng biệt.
- QNN (Qualcomm Neural Network) SDK và AI Hub cung cấp khả năng chuyển đổi từ PyTorch/ONNX.
- Các chat template, Llama 3.2, Phi-3 đều được cung cấp dưới dạng artifact hạng nhất trên AI Hub.

### Intel / AMD NPUs (Lunar Lake, Ryzen AI 300)

- 40-50 TOPS. Phần mềm đi sau Apple/Qualcomm; OpenVINO đang cải thiện nhưng vẫn là thị trường ngách.
- Tốt nhất cho các ứng dụng copilot trên Windows ARM; native trên máy tính để bàn AMD/Intel cho nhu cầu local-first.

### WebGPU + WebLLM

- Chạy mô hình trong trình duyệt qua WebGPU compute shaders; không cần cài đặt.
- Llama 3.1 8B Q4 đạt ~41 tok/s trên M3 Max — khoảng 70-80% so với native thông qua cùng backend.
- 17.6k sao trên GitHub cho WebLLM; JS API tương thích OpenAI; Apache 2.0.
- Độ phủ 2026: Chrome Android v121+, Safari iOS 26 GA, Firefox Android đang bắt kịp. Độ phủ di động tổng thể ~70-75%.

### Dòng NVIDIA Jetson

- Orin Nano Super (8GB): phù hợp với Llama 3.2 3B, Phi-3 ở tốc độ tok/s tốt.
- AGX Orin: chạy gpt-oss-20b qua vLLM ở mức ~40 tok/s.
- Thor / T4000 (JetPack 7.1): hiệu năng gấp 2 lần AGX Orin, hỗ trợ EAGLE-3 và NVFP4.
- TensorRT Edge-LLM (2026) hỗ trợ giải mã suy đoán EAGLE-3, trọng số NVFP4, chunked prefill — các tối ưu hóa từ datacenter được port sang edge.

### Lựa chọn lượng tử hóa theo mục tiêu

| Mục tiêu | Định dạng | Ghi chú |
|--------|--------|-------|
| Apple ANE | Trọng số INT4 + activations FP16 | Đường dẫn chuyển đổi Core ML |
| Qualcomm Hexagon | QNN INT8 / INT4 | Các bộ chuyển đổi AI Hub |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | Sử dụng `mlc_llm convert_weight` + `.wasm` đã biên dịch; GGUF không được hỗ trợ |
| Jetson Orin Nano | Q4 GGUF hoặc TRT-LLM INT4 | Bị giới hạn bởi bộ nhớ |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Đường dẫn Edge-LLM |

### Cái bẫy context dài trên edge

Context 128K của Llama 3.1 là tính năng dành cho datacenter. Trên điện thoại có 8 GB RAM, mô hình 4 GB + 2 GB KV cache cho 32K token + overhead hệ điều hành = OOM (hết bộ nhớ). Các triển khai trên edge giữ context ở mức 4K-8K trừ khi chấp nhận lượng tử hóa KV mạnh (Q4 KV).

### Giọng nói là ứng dụng sát thủ (killer app)

Các tác nhân giọng nói nhạy cảm với độ trễ (token đầu tiên < 500 ms). Inference cục bộ loại bỏ hoàn toàn độ trễ mạng. Kết hợp với chuyển đổi giọng nói thành văn bản (các biến thể Whisper Turbo chạy trên edge) và edge inference trở thành vòng lặp giọng nói chất lượng sản xuất.

### Các con số cần ghi nhớ

- Apple M4 / A18 ANE: 38 TOPS.
- Qualcomm Hexagon SD X Elite: 45 TOPS.
- WebLLM M3 Max: ~41 tok/s trên Llama 3.1 8B Q4.
- AGX Orin: ~40 tok/s trên gpt-oss-20b qua vLLM.
- Khoảng cách băng thông datacenter-edge: 30-50x.
- Độ phủ di động của WebGPU: ~70-75% (Firefox Android đang chậm lại).

```figure
edge-bandwidth-pipe
```

## Sử dụng

`code/main.py` tính toán các giới hạn thông lượng giải mã lý thuyết từ toán học bị giới hạn bởi băng thông trên các mục tiêu edge. So sánh với các benchmark quan sát được và làm nổi bật nơi nào băng thông, chứ không phải năng lực tính toán, là nút thắt cổ chai.

## Triển khai

Bài học này tạo ra `outputs/skill-edge-target-picker.md`. Dựa trên nền tảng (iOS/Android/trình duyệt/Jetson), mô hình và ngân sách độ trễ/bộ nhớ, chọn định dạng lượng tử hóa và quy trình chuyển đổi.

## Bài tập

1. Chạy `code/main.py`. Đối với mô hình 7B ở định dạng Q4 trên Snapdragon 8 Gen 3 (băng thông ~77 GB/s), hãy tính giới hạn giải mã. So sánh với mức 6-8 tok/s quan sát được — runtime có hiệu quả không?
2. WebGPU trên Android yêu cầu Chrome v121+. Thiết kế phương án dự phòng cho các trình duyệt cũ hơn — phía máy chủ thông qua cùng API tương thích OpenAI.
3. Ứng dụng iOS của bạn cần streaming với context 4K. Sự kết hợp mô hình/định dạng nào cho phép bạn duy trì dưới 4 GB bộ nhớ khả dụng trên iPhone 16?
4. Jetson AGX Orin chạy gpt-oss-20b ở mức 40 tok/s. Jetson Nano chỉ phù hợp với mô hình 3B. Nếu sản phẩm của bạn nhắm đến cả hai, làm thế nào để thống nhất stack inference?
5. Lập luận xem liệu "WebLLM đã sẵn sàng cho sản xuất vào năm 2026 hay chưa." Trích dẫn độ phủ, hiệu năng và khoảng cách của Firefox Android.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| ANE | "Apple neural engine" | NPU trên thiết bị trong dòng M và A; bộ nhớ thống nhất |
| Hexagon | "Qualcomm NPU" | NPU của Snapdragon; QNN SDK để truy cập |
| WebGPU | "browser GPU" | API GPU trình duyệt chuẩn W3C; Chrome/Safari 2026 |
| WebLLM | "browser LLM runtime" | Dự án MLC-LLM; Apache 2.0; JS tương thích OpenAI |
| Jetson | "NVIDIA edge" | Dòng Orin Nano / AGX / Thor / T4000 |
| TRT Edge-LLM | "edge TensorRT" | Bản port edge của TensorRT-LLM năm 2026; EAGLE-3 + NVFP4 |
| Unified memory | "shared pool" | CPU và NPU nhìn thấy cùng RAM; không tốn chi phí copy |
| Bandwidth-bound | "memory limited" | Quá trình giải mã bị giới hạn bởi tốc độ đọc trọng số (bytes/sec) |
| Core ML | "Apple conversion" | Framework của Apple cho các mô hình ANE-native |
| QNN | "Qualcomm stack" | Qualcomm Neural Network SDK |

## Đọc thêm

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — bối cảnh và các benchmark.
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor.
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — thông báo về bản port edge 2026.
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — thiết kế và benchmark.
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — chuyển đổi ANE-native.
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — các mô hình đã chuyển đổi sẵn cho Hexagon.