# Multimodal RAG và Cross-Modal Retrieval

> Vision-native document RAG chỉ là một phần nhỏ. Multimodal RAG trong môi trường production có phạm vi rộng hơn — truy xuất trên nhiều định dạng văn bản, hình ảnh, âm thanh và video cho các quy trình như lập kế hoạch chuyến đi ("tìm cho tôi một quán brunch thuần chay yên tĩnh có ánh sáng tự nhiên"), phân loại y tế ("chấn thương nào khớp với ảnh này + các ghi chú này"), thương mại điện tử ("trang phục tương tự như ảnh selfie này, đúng size của tôi"), và dịch vụ hiện trường ("chẩn đoán âm thanh động cơ này cộng với ảnh của bộ phận đó"). Ba báo cáo khảo sát năm 2025 — Abootorabi và cộng sự, Mei và cộng sự, Zhao và cộng sự — đã hệ thống hóa các bài toán con: cross-modal retrieval, retrieval fusion, generation grounding, và multimodal evaluation. Bài học này sẽ đọc các báo cáo khảo sát đó và thiết kế một pipeline production.

**Type:** Build
**Languages:** Python (stdlib, cross-modal retriever với fusion + grounded generator)
**Prerequisites:** Phase 12 · 23 (ColPali), Phase 11 (RAG basics)
**Time:** ~180 phút

## Mục tiêu học tập

- Thiết kế cross-modal retrieval: văn bản → hình ảnh, hình ảnh → văn bản, âm thanh → video, v.v.
- So sánh ba chiến lược fusion: score fusion, attention-based fusion, MoE fusion.
- Giải thích generation grounding: việc "trích dẫn nguồn" trông như thế nào khi các nguồn là sự kết hợp của nhiều modality.
- Nêu tên ba báo cáo khảo sát multimodal RAG tiêu biểu năm 2025 và hệ thống phân loại bài toán con của chúng.

## Vấn đề

Single-modality RAG là một mô hình đã được giải quyết: nhúng truy vấn, nhúng các đoạn văn bản (chunks), truy xuất, và đưa vào LLM. Multimodal RAG yêu cầu:

1. Nhiều đầu truy xuất (mỗi modality cần các embedding trong một không gian tương thích).
2. Kết hợp (fusion) các kết quả truy xuất từ nhiều modality.
3. Generation grounding trích dẫn nguồn từ nhiều modality.
4. Các chỉ số đánh giá bao phủ tín hiệu cross-modal.

Tất cả các báo cáo khảo sát năm 2025 đều đi đến cùng một hệ thống phân loại.

## Khái niệm

### Cross-modal retrieval

Truy xuất các tài liệu thuộc modality B dựa trên truy vấn thuộc modality A. Ba mô hình:

1. Không gian nhúng chia sẻ (Shared embedding space). CLIP và CLAP tạo ra các embedding văn bản + hình ảnh / văn bản + âm thanh trong một không gian chung. Cosine similarity giữa các modality hoạt động trực tiếp. Bị giới hạn bởi các cặp dữ liệu được huấn luyện cho CLIP.

2. Per-modality encoder + translation. Encoder văn bản + encoder hình ảnh + một module dịch nhỏ để ánh xạ giữa các không gian. Sen2Sen của Gupta và cộng sự và các thiết kế khác năm 2024. Linh hoạt nhưng làm tăng độ phức tạp.

3. VLM làm encoder. Sử dụng các hidden states của VLM làm biểu diễn truy xuất. Bất kỳ modality nào mà VLM hỗ trợ đều hoạt động được. Chất lượng cao hơn, chi phí đắt hơn.

Lựa chọn: CLIP / SigLIP 2 cho văn bản+hình ảnh; CLAP cho văn bản+âm thanh; VLM-hidden-states cho cross-modal ở chất lượng cao nhất.

### Chiến lược Fusion

Bạn đã truy xuất được 10 kết quả: 5 hình ảnh, 3 đoạn văn bản, 2 đoạn âm thanh. Bạn hợp nhất chúng như thế nào?

Score fusion (rẻ nhất). Mỗi modality có retriever riêng, mỗi cái trả về điểm số. Chuẩn hóa điểm số trong nội bộ modality rồi cộng lại. Đơn giản, thường hiệu quả.

Attention-based fusion. Nối tất cả các mục đã truy xuất, để một mạng attention nhỏ gán trọng số cho chúng. Cần huấn luyện.

MoE fusion. Mạng gating điều hướng đến các chuyên gia (experts) chuyên biệt cho từng modality. Các loại truy vấn khác nhau sẽ được điều hướng khác nhau — một câu hỏi về hình ảnh sẽ gán trọng số cao hơn cho hình ảnh.

Mặc định trong production: score fusion với một chút thiên vị (bias) đối với modality chủ đạo của truy vấn. Nâng cấp lên MoE nếu A/B testing cho thấy kết quả vượt trội rõ rệt trên domain của bạn.

### Generation grounding

LLM nên trích dẫn mục truy xuất nào đã dẫn đến mỗi khẳng định. Đối với đa phương thức:

- Nguồn văn bản: trích dẫn tiêu chuẩn `[1]`.
- Nguồn hình ảnh: `[img 3]` kèm theo chú thích ngắn.
- Âm thanh: `[audio 2 at 0:34]`.

Huấn luyện generator với dữ liệu nhận thức grounding: mỗi khẳng định trong mục tiêu huấn luyện được gắn thẻ với chỉ số nguồn. Khi inference, mô hình tự nhiên xuất ra các trích dẫn.

### Các báo cáo khảo sát năm 2025

Abootorabi và cộng sự (arXiv:2502.08826, "Ask in Any Modality"): hệ thống phân loại cho multimodal RAG. Bao gồm truy xuất, fusion, tạo văn bản. Phạm vi bao phủ rộng nhất.

Mei và cộng sự (arXiv:2504.08748, "A Survey of Multimodal RAG"): tập trung vào các benchmark tác vụ con và các chế độ lỗi (failure modes). Hữu ích cho việc thiết kế đánh giá.

Zhao và cộng sự (arXiv:2503.18016): báo cáo khảo sát tập trung vào thị giác (vision). Mạnh về các công trình thuộc họ ColPali.

Đọc cả ba báo cáo này sẽ cung cấp cho bạn trạng thái kỹ thuật mới nhất tính đến mùa xuân 2025. Hầu hết các bài toán con vẫn đang là vấn đề mở.

### MuRAG — bài báo nền tảng

MuRAG (Chen và cộng sự, 2022) là multimodal RAG đầu tiên. Truy xuất hình ảnh + văn bản từ một KB đa phương thức, tạo câu trả lời. Đã chứng minh tính khả thi trước làn sóng VLM. Các hệ thống hiện đại (REACT, VisRAG, M3DocRAG) đều xây dựng dựa trên nó.

### Ví dụ về lập kế hoạch chuyến đi trong production

Truy vấn: "tìm cho tôi một quán brunch thuần chay yên tĩnh có ánh sáng tự nhiên."

Pipeline:

1. Phân tách truy vấn. "yên tĩnh" → từ khóa âm thanh/đánh giá; "brunch thuần chay" → mục thực đơn; "ánh sáng tự nhiên" → đặc điểm hình ảnh.
2. Truy xuất theo từng modality:
   - Truy xuất văn bản trên các đánh giá: "brunch thuần chay, không gian yên tĩnh."
   - Truy xuất hình ảnh trên ảnh nhà hàng: "ánh sáng tự nhiên, thoáng đãng."
   - Truy xuất âm thanh trên các đoạn âm thanh môi trường: "decibel thấp, không nhạc."
3. Hợp nhất điểm số. Mỗi nhà hàng có một điểm số tổng hợp.
4. Top-k nhà hàng → VLM generator với tất cả bằng chứng → trả lời kèm trích dẫn.

Điều này vượt xa text-RAG. Mỗi modality bổ sung tín hiệu mà văn bản đơn thuần bỏ lỡ.

### Agentic multimodal RAG

Multi-hop: nếu lần truy xuất đầu tiên không trả về câu trả lời có độ tin cậy cao, LLM sẽ tái cấu trúc và truy xuất lại. Các mô hình Agentic RAG từ Phase 14 áp dụng ở đây. Ví dụ:

- Truy xuất top-10 ban đầu → LLM hỏi "quá ồn, lọc lấy <40 dB" → truy xuất lại.
- Truy xuất hình ảnh → LLM thấy một cái có thực đơn → truy xuất văn bản thực đơn → trả lời.

Tăng độ phức tạp nhưng xử lý được các truy vấn mà truy xuất một lần (single-shot) không làm được.

### Đánh giá

Đánh giá đa phương thức vẫn còn non trẻ. Các proxy phổ biến:

- Recall@k theo từng modality.
- Độ chính xác của top-k đã hợp nhất.
- Sự hài lòng tổng thể do con người đánh giá (end-to-end).
- Theo tác vụ cụ thể (hoàn thành đặt chỗ, thực hiện mua hàng).

Chưa có benchmark tiêu chuẩn nào bao trùm tất cả các modality. Hầu hết các bài báo đánh giá trên các tác vụ cụ thể theo domain.

```figure
contrastive-matrix
```

## Sử dụng

`code/main.py`:

- Ba mock retriever (văn bản, hình ảnh, âm thanh) hoạt động trên một corpus nhà hàng chung.
- Score fusion kết hợp điểm số modality với các trọng số có thể cấu hình.
- Một generator stub xuất ra câu trả lời cuối cùng kèm trích dẫn.
- Một vòng lặp agentic đơn giản tái cấu trúc truy vấn nếu độ tin cậy thấp.

## Triển khai

Bài học này tạo ra `outputs/skill-multimodal-rag-designer.md`. Với một đặc tả sản phẩm có luồng truy vấn đa phương thức, thiết kế các retriever, fusion, generator và đánh giá.

## Bài tập

1. Đề xuất một hệ thống multimodal RAG cho phân loại y tế: truy vấn = ảnh chấn thương + văn bản triệu chứng. Những modality nào truy xuất từ KB nào?

2. Score fusion là một tổng có trọng số đơn giản. Nó có chế độ lỗi nào mà MoE fusion tránh được?

3. Đọc hệ thống phân loại của Abootorabi và cộng sự (Mục 3). Ba bài toán con tiêu biểu là gì và chúng ánh xạ thế nào vào sản phẩm bạn đã chọn?

4. Thiết kế đặc tả đánh giá cho multimodal RAG lập kế hoạch chuyến đi. Những chỉ số nào bao phủ recall hình ảnh, recall âm thanh và độ chính xác tổng hợp?

5. Agentic multi-hop RAG có chi phí độ trễ cho mỗi vòng lặp. Ở mức độ khó truy vấn nào thì mức tăng độ chính xác sẽ biện minh cho độ trễ đó?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Cross-modal retrieval | "Truy vấn một modality, lấy kết quả khác" | Truy vấn văn bản lấy hình ảnh; truy vấn hình ảnh lấy văn bản; yêu cầu không gian chung hoặc bộ dịch |
| Score fusion | "Kết hợp điểm số" | Tổng có trọng số của điểm truy xuất từng modality; fusion đơn giản nhất |
| MoE fusion | "Chuyên gia điều hướng modality" | Mạng gating chọn điểm số của modality nào để tin tưởng cho mỗi truy vấn |
| Grounded generation | "Trích dẫn nguồn" | Mỗi khẳng định trong câu trả lời được gắn thẻ với chỉ số nguồn |
| MuRAG | "Multimodal RAG đầu tiên" | Bài báo năm 2022 thiết lập mô hình multimodal RAG |
| Agentic multi-hop | "Tái cấu trúc và thử lại" | LLM truy vấn lại các retriever khi độ tin cậy của lần đầu thấp |

## Đọc thêm

- [Abootorabi và cộng sự — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei và cộng sự — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao và cộng sự — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen và cộng sự — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu và cộng sự — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)