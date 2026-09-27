# Production Quantization — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

> Định dạng lượng tử hóa (quantization) không phải là lựa chọn vạn năng — nó phụ thuộc vào phần cứng, engine phục vụ (serving engine) và khối lượng công việc. GGUF Q4_K_M hoặc Q5_K_M thống trị trên CPU và thiết bị edge, được phân phối thông qua llama.cpp và Ollama. GPTQ chiếm ưu thế trong vLLM khi bạn cần chạy multi-LoRA trên cùng một base model. AWQ với các kernel Marlin-AWQ đạt khoảng 741 tok/s trên model lớp 7B với chỉ số Pass@1 tốt nhất ở định dạng INT4 — đây là tiêu chuẩn cho sản xuất tại datacenter vào năm 2026. FP8 vẫn là lựa chọn trung hòa trên Hopper, Ada và Blackwell — gần như không mất mát dữ liệu và được hỗ trợ rộng rãi. NVFP4 và MXFP4 (microscaling trên Blackwell) rất mạnh mẽ nhưng yêu cầu xác thực theo từng block. Có hai cái bẫy mà các đội ngũ thường gặp phải: tập dữ liệu hiệu chuẩn (calibration dataset) phải khớp với miền dữ liệu triển khai, và KV cache là thành phần tách biệt với lượng tử hóa trọng số — bài học từ AWQ "model của tôi giờ chỉ còn 4 GB" thường quên mất 10-30 GB KV cache ở các batch size thực tế.

**Type:** Learn
**Languages:** Python (stdlib, toy memory and throughput comparison across formats)
**Prerequisites:** Phase 10 · 13 (Quantization foundations), Phase 17 · 04 (Serving Engine Internals)
**Time:** ~75 minutes

## Mục tiêu học tập

- Liệt kê sáu định dạng lượng tử hóa trong sản xuất và ưu điểm của chúng vào năm 2026.
- Chọn định dạng dựa trên phần cứng (CPU vs GPU, Hopper vs Blackwell), engine (vLLM, TRT-LLM, llama.cpp) và khối lượng công việc (chat thông thường, suy luận, multi-LoRA).
- Tính toán bộ nhớ trọng số được tiết kiệm và KV cache không bị ảnh hưởng cho một định dạng đã chọn.
- Chỉ ra sai lầm về tập dữ liệu hiệu chuẩn làm suy giảm chất lượng model lượng tử hóa trên dữ liệu miền chuyên biệt.

## Vấn đề

Lượng tử hóa giúp giảm bộ nhớ và băng thông HBM, đây chính xác là những gì quá trình giải mã (decode) cần. Một model 70B ở định dạng FP16 chiếm 140 GB trọng số. Lượng tử hóa trọng số xuống INT4 (AWQ hoặc GPTQ) sẽ đưa model về 35 GB — vừa vặn trong một card H100 và còn dư chỗ cho KV cache, điều này rất quan trọng vì ở mức 128 chuỗi đồng thời với context 2k, riêng KV cache đã chiếm 20-30 GB.

Nhưng lượng tử hóa không miễn phí. Lượng tử hóa quá mức sẽ làm giảm chất lượng, đặc biệt là trong các tác vụ đòi hỏi suy luận cao. Các định dạng khác nhau hoạt động với các engine khác nhau. Phần cứng khác nhau hỗ trợ các độ chính xác khác nhau ở cấp độ phần cứng. "Vườn thú" định dạng năm 2026 là có thật và bạn không thể sao chép lựa chọn của người khác — bạn phải chọn dựa trên stack của mình.

## Khái niệm

### Sáu định dạng chính

| Định dạng | Bits | Ưu điểm | Engines |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU, edge, laptop | llama.cpp, Ollama |
| GPTQ | 4-8 | Multi-LoRA trên vLLM | vLLM, TGI |
| AWQ | 4 | Datacenter GPU production | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell datacenter | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell multi-user | TRT-LLM |
| NVFP4 | 4 | Blackwell multi-user | TRT-LLM |

### GGUF — mặc định cho CPU/edge

GGUF là một định dạng tệp, không hẳn là một lược đồ lượng tử hóa — nó đóng gói các biến thể K-quant (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0) trong một container. Q4_K_M và Q5_K_M là các mặc định sản xuất — chất lượng gần như BF16 ở mức 4-5 bits. Đây là lựa chọn tốt nhất cho CPU hoặc phục vụ tại edge vì llama.cpp là engine suy luận CPU nhanh nhất hiện nay.

Hình phạt về throughput trong vLLM: ~93 tok/s trên 7B — định dạng này không được tối ưu cho các kernel GPU. Chỉ sử dụng GGUF khi mục tiêu triển khai là CPU/edge.

### GPTQ — multi-LoRA trong vLLM

GPTQ là thuật toán lượng tử hóa sau huấn luyện (post-training) với một bước hiệu chuẩn. Các kernel Marlin giúp nó chạy nhanh trên GPU (tăng tốc 2.6x so với GPTQ không dùng Marlin). ~712 tok/s trên 7B.

Điểm thắng lợi độc đáo: GPTQ-Int4 hỗ trợ các adapter LoRA trong vLLM. Nếu bạn đang phục vụ một base model cộng với 10-50 biến thể tinh chỉnh (mỗi biến thể là một LoRA), GPTQ là con đường dành cho bạn. NVFP4 hiện chưa hỗ trợ LoRA tính đến đầu năm 2026.

### AWQ — mặc định cho datacenter GPU

Activation-aware Weight Quantization. Bảo vệ khoảng 1% trọng số quan trọng nhất trong quá trình lượng tử hóa. Các kernel Marlin-AWQ: tăng tốc 10.9x so với cách tiếp cận thông thường. ~741 tok/s trên 7B, Pass@1 tốt nhất trong các định dạng INT4.

Chọn AWQ cho việc phục vụ GPU mới trừ khi bạn cần multi-LoRA (GPTQ) hoặc FP4 mạnh mẽ trên Blackwell (NVFP4).

### FP8 — lựa chọn trung hòa đáng tin cậy

8-bit floating point. Gần như không mất mát dữ liệu. Được hỗ trợ rộng rãi. Hopper Tensor Cores tăng tốc FP8 ở cấp độ phần cứng. Blackwell cũng kế thừa điều này. FP8 là mặc định an toàn năm 2026 khi chất lượng là yếu tố không thể thỏa hiệp (suy luận, y tế, tạo mã). Tiết kiệm bộ nhớ chỉ bằng một nửa so với INT4 nhưng rủi ro về chất lượng thấp hơn nhiều.

### MXFP4 / NVFP4 — sự mạnh mẽ trên Blackwell

Microscaling FP4. Mỗi block trọng số có hệ số tỷ lệ riêng. Rất mạnh mẽ nhưng yêu cầu tăng tốc phần cứng trên Blackwell Tensor Cores. Giảm một nửa số byte trên mỗi token so với FP8 — lợi ích kinh tế trong Phase 17 · 07.

Lưu ý:
- Chưa hỗ trợ LoRA (đầu năm 2026).
- Chất lượng giảm rõ rệt trên các khối lượng công việc đòi hỏi suy luận cao.
- Cần xác thực trên tập eval của bạn cho từng model.

### Cái bẫy hiệu chuẩn

AWQ và GPTQ yêu cầu tập dữ liệu hiệu chuẩn — thường là C4 hoặc WikiText. Đối với các model chuyên biệt (code, y tế, pháp lý), việc hiệu chuẩn trên văn bản web chung khiến thuật toán đưa ra quyết định sai lầm về việc trọng số nào cần bảo vệ. Chỉ số Pass@1 trên HumanEval có thể giảm vài điểm.

Cách khắc phục: hiệu chuẩn trên dữ liệu trong miền (in-domain). Hàng trăm mẫu dữ liệu chuyên biệt thường là đủ. Kiểm tra trên tập eval trước khi triển khai.

### Cái bẫy KV cache

AWQ thu nhỏ trọng số xuống 4 bits. KV cache là thành phần tách biệt và vẫn giữ ở định dạng FP16/FP8. Đối với model 70B với AWQ:

- Trọng số: ~35 GB (INT4 từ 140 GB).
- KV cache ở mức 128 đồng thời × 2k context: ~20 GB.
- Activations: ~5 GB.
- Tổng cộng: ~60 GB — vừa vặn trên H100 80GB.

Việc ngây thơ cho rằng "tôi đã lượng tử hóa model xuống 4 GB" sẽ bỏ qua 30-50 GB còn lại. Hãy lập ngân sách HBM một cách tổng thể.

Ngoài ra, lượng tử hóa KV cache (FP8 KV hoặc INT8 KV) là một lựa chọn khác với những đánh đổi riêng — nó ảnh hưởng trực tiếp đến độ chính xác của attention và không phải là lợi ích miễn phí.

### AWQ INT4 nguy hiểm cho suy luận

Chain-of-thought, toán học, tạo mã với context dài — những tác vụ này bị ảnh hưởng rõ rệt bởi lượng tử hóa mạnh. AWQ INT4 mất khoảng 3-5 điểm trên MATH. Đối với các khối lượng công việc đòi hỏi suy luận cao, hãy sử dụng FP8 hoặc BF16; chấp nhận chi phí bộ nhớ.

### Hướng dẫn lựa chọn năm 2026

- Phục vụ CPU/edge: GGUF Q4_K_M. Xong.
- Phục vụ GPU, chat thông thường, không LoRA: AWQ.
- Phục vụ GPU, multi-LoRA: GPTQ với Marlin.
- Khối lượng công việc suy luận: FP8.
- Datacenter Blackwell, chất lượng đã xác thực: NVFP4 + FP8 KV.
- Không chắc chắn: chạy eval 1.000 mẫu trên mỗi định dạng ứng viên.

```figure
gpu-memory-breakdown
```

## Sử dụng

`code/main.py` tính toán dung lượng bộ nhớ (trọng số + KV + activations) và throughput tương đối trên sáu định dạng cho nhiều kích thước model khác nhau. Cho thấy nơi KV cache chiếm ưu thế, nơi nén trọng số mang lại hiệu quả và nơi FP8 là lựa chọn an toàn.

## Triển khai

Bài học này tạo ra `outputs/skill-quantization-picker.md`. Dựa trên phần cứng, kích thước model, loại công việc và mức độ chấp nhận chất lượng, chọn một định dạng và lập kế hoạch hiệu chuẩn/xác thực.

## Bài tập

1. Chạy `code/main.py`. Đối với model 70B ở mức 128 đồng thời với context 2k, hãy tính tổng HBM cho mỗi định dạng. Định dạng nào cho phép bạn chạy trên một card H100 80GB?
2. Bạn có một model coding 7B. Hãy chọn một định dạng và giải thích lý do. Nếu bạn sai về mức độ chấp nhận chất lượng, lộ trình phục hồi là gì?
3. Tính toán kích thước tập dữ liệu hiệu chuẩn cần thiết để hiệu chuẩn AWQ cho một model y tế. Tại sao nhiều dữ liệu hơn không phải lúc nào cũng tốt hơn?
4. Đọc bài báo hoặc ghi chú phát hành về kernel Marlin-AWQ. Giải thích trong ba câu tại sao AWQ đạt 741 tok/s trên 7B trong khi GPTQ thô đạt ~712.
5. Khi nào thì nên kết hợp trọng số AWQ với FP8 KV cache thay vì giữ KV ở BF16?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| GGUF | "định dạng llama.cpp" | Định dạng tệp đóng gói các biến thể K-quant; mặc định cho CPU/edge |
| Q4_K_M | "Q4 K M" | 4-bit K-quant medium; mặc định GGUF cho sản xuất |
| GPTQ | "gee pee tee q" | INT4 sau huấn luyện với hiệu chuẩn; hỗ trợ LoRA trong vLLM |
| AWQ | "a w q" | INT4 nhận biết kích hoạt; kernel Marlin; Pass@1 tốt nhất ở INT4 |
| Marlin kernels | "kernel INT4 nhanh" | Kernel CUDA tùy chỉnh cho INT4 trên Hopper; tăng tốc 10x |
| FP8 | "số thực tám bit" | Độ chính xác mặc định an toàn trên Hopper/Ada/Blackwell |
| MXFP4 / NVFP4 | "microscaling four" | FP 4-bit trên Blackwell với hệ số tỷ lệ theo block |
| Calibration dataset | "dữ liệu cal" | Văn bản đầu vào dùng để chọn tham số lượng tử hóa; phải khớp với miền dữ liệu |
| KV cache quantization | "KV INT8" | Lựa chọn tách biệt với trọng số; ảnh hưởng đến độ chính xác của attention |

## Đọc thêm

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — các benchmark so sánh.
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — số liệu throughput theo định dạng.
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — cách chọn định dạng.
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — các định dạng và cờ hỗ trợ.
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — công thức AWQ gốc.
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — công thức GPTQ gốc.