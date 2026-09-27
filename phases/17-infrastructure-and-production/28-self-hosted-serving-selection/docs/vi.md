# Lựa chọn Serving tự lưu trữ (Self-Hosted) — Kết hợp Engine với Phần cứng và Quy mô

> Việc lựa chọn engine phụ thuộc vào phần cứng, quy mô và hệ sinh thái — không phải dựa trên bảng xếp hạng. Bốn engine thống trị việc inference tự lưu trữ vào năm 2026: llama.cpp, Ollama, vLLM, SGLang, trong khi TGI đang dần bị loại bỏ ở chế độ bảo trì. **llama.cpp** nhanh nhất trên CPU — hỗ trợ model rộng nhất, kiểm soát hoàn toàn về quantization và threading. **Ollama** là lựa chọn cài đặt một lệnh cho laptop của lập trình viên, chậm hơn khoảng 15-30% so với llama.cpp (do Go + CGo + tuần tự hóa HTTP), và có khoảng cách throughput gấp 3 lần dưới tải thực tế. **TGI đã chuyển sang chế độ bảo trì từ ngày 11 tháng 12 năm 2025** — chỉ sửa lỗi, throughput thô chậm hơn khoảng 10% so với vLLM nhưng trước đây có khả năng quan sát (observability) và tích hợp hệ sinh thái HF tốt nhất. Trạng thái bảo trì này khiến nó trở thành một lựa chọn rủi ro về lâu dài — SGLang hoặc vLLM là các lựa chọn mặc định an toàn hơn cho các dự án mới. **vLLM** là lựa chọn mặc định cho production đa năng — v0.15.1 (tháng 2 năm 2026) bổ sung PyTorch 2.10, RTX Blackwell SM120, và tối ưu hóa H200. **SGLang** là chuyên gia cho các tác vụ agentic đa lượt (multi-turn) / ưu tiên prefix — với hơn 400.000 GPU đang chạy trong production (xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS). Ràng buộc phần cứng: Ưu tiên CPU → llama.cpp. AMD / không phải NVIDIA → vLLM là con đường được hỗ trợ mạnh mẽ nhất (TRT-LLM bị khóa với NVIDIA). Mô hình pipeline năm 2026: dev = Ollama, staging = llama.cpp, prod = vLLM hoặc SGLang. Các engine sử dụng các định dạng trọng số khác nhau — GGUF cho dòng llama.cpp, HF safetensors cho các engine GPU — vì vậy việc chuyển đổi định dạng có thể cần thiết giữa các giai đoạn.

**Type:** Learn
**Languages:** Python (stdlib, engine-decision tree walker)
**Prerequisites:** Tất cả các bài học trong Phase 17 về engine (04, 06, 07, 09, 18)
**Time:** ~45 phút

## Mục tiêu học tập

- Chọn engine dựa trên phần cứng (CPU / AMD / NVIDIA Hopper / Blackwell), quy mô (1 người dùng / 100 / 10.000) và khối lượng công việc (chat tổng quát / agent / ngữ cảnh dài).
- Nắm được trạng thái bảo trì của TGI năm 2026 (11 tháng 12 năm 2025) và lý do tại sao nó khiến các dự án mới ưu tiên vLLM hoặc SGLang.
- Mô tả pipeline dev/staging/prod, bao gồm vị trí của việc chuyển đổi định dạng GGUF sang safetensors giữa các giai đoạn.
- Giải thích tại sao "Ưu tiên CPU" dẫn đến llama.cpp và "AMD" loại trừ TRT-LLM.

## Vấn đề

Nhóm của bạn bắt đầu một dự án LLM tự lưu trữ mới. Một kỹ sư đề xuất Ollama, người khác đề xuất vLLM, người thứ ba hỏi "TGI chẳng phải hoạt động ngay lập tức sao?". Cả ba đều đúng trong các ngữ cảnh khác nhau. Không ai đúng cho tất cả.

Vào năm 2026, cây quyết định rất quan trọng: phần cứng trước, quy mô thứ hai, khối lượng công việc thứ ba. Và một sự kiện cụ thể năm 2025 — TGI chuyển sang chế độ bảo trì ngày 11 tháng 12 — đã thay đổi lựa chọn mặc định cho các dự án mới.

## Khái niệm

### Năm engine chính

| Engine | Tốt nhất cho | Ghi chú |
|--------|----------|-------|
| **llama.cpp** | CPU / edge / phụ thuộc tối thiểu / hỗ trợ model rộng nhất | Nhanh nhất trên CPU, kiểm soát toàn diện |
| **Ollama** | Laptop dev, người dùng đơn lẻ, cài đặt một lệnh | Chậm hơn 15-30% so với llama.cpp; khoảng cách throughput prod gấp 3 lần |
| **TGI** | Hệ sinh thái HF, các ngành được quản lý | **Chế độ bảo trì từ 11/12/2025** |
| **vLLM** | Production đa năng, 100+ người dùng | Mặc định cho production; v0.15.1 tháng 2/2026 |
| **SGLang** | Tác vụ agentic đa lượt, khối lượng công việc prefix-heavy | 400.000+ GPU trong production |

### Quyết định dựa trên phần cứng

**Ưu tiên CPU** → llama.cpp. Ollama cũng hoạt động nhưng chậm hơn. Không có engine nào khác cạnh tranh được trên CPU.

**AMD GPU** → vLLM là con đường được hỗ trợ mạnh mẽ nhất (hỗ trợ AMD ROCm). SGLang cũng hoạt động. TRT-LLM bị khóa với NVIDIA, nên bị loại.

**NVIDIA Hopper (H100 / H200)** → vLLM hoặc SGLang hoặc TRT-LLM. Cả ba đều là hàng đầu.

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM dẫn đầu về throughput (Phase 17 · 07). vLLM và SGLang theo sát phía sau.

**Apple Silicon (M-series)** → llama.cpp (Metal). Ollama bao bọc engine này.

### Quyết định dựa trên quy mô

**1 người dùng / dev cục bộ** → Ollama. Một lệnh, token đầu tiên trong vài giây.

**10-100 người dùng / nhóm nhỏ** → vLLM single-GPU.

**100-10k người dùng / production** → vLLM production-stack (Phase 17 · 18) hoặc SGLang.

**10k+ người dùng / doanh nghiệp** → vLLM production-stack + disaggregated (Phase 17 · 17) + LMCache (Phase 17 · 18).

### Quyết định dựa trên khối lượng công việc

**Chat tổng quát / Q&A** → vLLM thắng nhờ tính mặc định rộng rãi.

**Agentic đa lượt (công cụ, lập kế hoạch, bộ nhớ)** → RadixAttention của SGLang (Phase 17 · 06) chiếm ưu thế.

**RAG với việc tái sử dụng prefix nặng** → SGLang.

**Tạo mã (Code generation)** → vLLM tốt; SGLang tốt hơn một chút về cache.

**Ngữ cảnh dài (128K+)** → vLLM + chunked prefill; SGLang + tiered KV.

### Cái bẫy bảo trì của TGI

Hugging Face TGI đã chuyển sang chế độ bảo trì vào ngày 11 tháng 12 năm 2025 — chỉ sửa lỗi từ nay về sau. Trước đây: khả năng quan sát hàng đầu, tích hợp hệ sinh thái HF tốt nhất (model cards, công cụ an toàn), chậm hơn vLLM một chút về throughput thô.

Đối với các dự án mới vào năm 2026: hãy tránh TGI. Các triển khai TGI hiện tại có thể tiếp tục nhưng nên sớm di chuyển. SGLang và vLLM là các lựa chọn mặc định an toàn hơn.

### Mô hình pipeline

Dev (Ollama) → staging (llama.cpp) → prod (vLLM). Các engine sử dụng các định dạng trọng số khác nhau — GGUF cho dòng llama.cpp, HF safetensors cho các engine GPU — vì vậy việc chuyển đổi định dạng có thể nằm giữa các giai đoạn. Các kỹ sư lặp lại nhanh trên laptop; staging phản ánh quantization của production; prod là mục tiêu phục vụ.

### Lưu ý về Ollama

Ollama rất tốt cho dev. Nó không tốt cho production chia sẻ: tuần tự hóa HTTP của Go gây thêm overhead, quản lý concurrency đơn giản hơn vLLM, hỗ trợ OpenTelemetry còn chậm. Hãy sử dụng Ollama nơi nó tỏa sáng — một người dùng, một lệnh — và chuyển sang vLLM cho môi trường chia sẻ.

### Tự lưu trữ vs Managed là một quyết định riêng biệt

Phase 17 · 01 (managed hyperscalers), · 02 (inference platforms) bao gồm các dịch vụ managed. Bài học này giả định bạn đã quyết định tự lưu trữ. Lý do tự lưu trữ: lưu trú dữ liệu, fine-tune tùy chỉnh, tổng chi phí sở hữu ở quy mô lớn, model chuyên biệt không có sẵn trên các dịch vụ hosted.

### Các con số bạn cần nhớ

- Chế độ bảo trì TGI: 11 tháng 12 năm 2025.
- vLLM v0.15.1: Tháng 2 năm 2026; PyTorch 2.10; hỗ trợ Blackwell SM120.
- Quy mô production của SGLang: 400.000+ GPU.
- Khoảng cách throughput của Ollama so với llama.cpp: chậm hơn 15-30%; gấp 3 lần dưới tải prod.

```figure
data-parallel
```

## Sử dụng nó

`code/main.py` là một trình duyệt cây quyết định: dựa trên phần cứng + quy mô + khối lượng công việc, chọn một engine và giải thích lý do.

## Triển khai nó

Bài học này tạo ra `outputs/skill-engine-picker.md`. Dựa trên các ràng buộc, chọn một engine và viết kế hoạch di chuyển.

## Bài tập

1. Chạy `code/main.py` với phần cứng / quy mô / khối lượng công việc của bạn. Kết quả đầu ra có khớp với trực giác của bạn không?
2. Hạ tầng của bạn là 12 H100 và 8 MI300X AMD. Chọn engine nào? Tại sao TRT-LLM bị loại?
3. Một nhóm muốn sử dụng TGI vào năm 2026 vì "đó là thứ chúng tôi biết". Hãy lập luận cho trường hợp di chuyển.
4. Từ Ollama dev sang vLLM prod: những gì thay đổi về quantization, cấu hình và khả năng quan sát?
5. Sản phẩm RAG với độ dài prefix P99 là 8K và khả năng tái sử dụng cao giữa các tenant. Chọn một engine và kết hợp nó với Phase 17 · 11 + 18.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|----------------|------------------------|
| llama.cpp | "cái chạy trên CPU" | Hỗ trợ model rộng nhất, nhanh nhất trên CPU |
| Ollama | "cái cho laptop" | Cài đặt một lệnh, throughput mức dev |
| TGI | "serving của HF" | Chế độ bảo trì từ tháng 12/2025 |
| vLLM | "mặc định" | Baseline production rộng rãi năm 2026 |
| SGLang | "cái cho agent" | Prefix-heavy, RadixAttention |
| TRT-LLM | "bị khóa với NVIDIA" | Dẫn đầu throughput Blackwell, chỉ NVIDIA |
| GGUF | "định dạng llama.cpp" | Các biến thể K-quant đi kèm |
| Production-stack | "vLLM K8s" | Triển khai tham chiếu Phase 17 · 18 |
| Pipeline pattern | "dev→stage→prod" | Ollama → llama.cpp → vLLM; định dạng trọng số khác nhau theo engine |

## Đọc thêm

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — So sánh toàn diện các Engine Inference LLM](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 lựa chọn thay thế vLLM tốt nhất 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [Thông báo bảo trì TGI](https://github.com/huggingface/text-generation-inference) — ghi chú phát hành.
- [Ghi chú phát hành vLLM v0.15.1](https://github.com/vllm-project/vllm/releases)