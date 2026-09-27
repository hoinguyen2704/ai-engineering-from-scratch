# ColPali và Vision-Native Document RAG

> RAG truyền thống phân tích PDF thành văn bản, chia nhỏ thành các đoạn (chunks), nhúng (embed) các đoạn đó và lưu trữ vào vector database. Mỗi bước đều làm mất đi tín hiệu quan trọng: OCR làm mất dữ liệu biểu đồ, việc chia đoạn làm đứt gãy các hàng trong bảng, và text embedding bỏ qua các hình ảnh. ColPali (Faysse và cộng sự, tháng 7 năm 2024) đặt ra một câu hỏi đơn giản hơn: tại sao phải trích xuất văn bản làm gì? Hãy nhúng trực tiếp hình ảnh trang tài liệu thông qua PaliGemma, sử dụng cơ chế tương tác muộn (late interaction) kiểu ColBERT để truy xuất, và giữ lại toàn bộ bố cục, hình ảnh, phông chữ và định dạng mà tài liệu mang lại. Các kết quả đo lường đã công bố cho thấy độ chính xác end-to-end cao hơn 20-40% so với text-RAG trên các tài liệu giàu hình ảnh. ColQwen2, ColSmol và VisRAG đã mở rộng mô hình này. Bài học này sẽ tìm hiểu luận điểm về vision-native RAG và xây dựng một bộ lập chỉ mục (indexer) nhỏ theo phong cách ColPali.

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Time:** ~180 phút

## Mục tiêu học tập

- Giải thích sự khác biệt giữa truy xuất bi-encoder (một vector cho mỗi tài liệu) và truy xuất tương tác muộn (nhiều vector cho mỗi tài liệu).
- Mô tả thao tác MaxSim của ColBERT và cách ColPali tổng quát hóa nó từ các token văn bản sang các patch hình ảnh.
- Xây dựng một bộ lập chỉ mục nhỏ kiểu ColPali: trang → patch embeddings → MaxSim trên các embedding của truy vấn → top-k trang.
- So sánh bộ tạo (generator) ColPali + Qwen2.5-VL với text-RAG + GPT-4 trong trường hợp sử dụng hóa đơn / báo cáo tài chính.

## Vấn đề

Text-RAG trên PDF loại bỏ hầu hết thông tin của tài liệu. Tăng trưởng doanh thu quý 3 trong một báo cáo tài chính thường nằm ở biểu đồ; các phát hiện trong báo cáo y tế nằm ở các hình ảnh có chú thích; khối chữ ký trong hợp đồng pháp lý là một thực tế về bố cục, không phải thực tế văn bản.

Quy trình text-RAG:

1. PDF → văn bản thông qua OCR / pdftotext.
2. Văn bản → các đoạn 300-500 token.
3. Đoạn → nhúng bi-encoder (một vector).
4. Truy vấn người dùng → nhúng → cosine similarity → top-k đoạn.
5. Các đoạn + truy vấn → LLM.

Năm bước gây mất mát dữ liệu. Biểu đồ không được ghi lại. Bảng bị chia cắt giữa các đoạn. Bố cục nhiều cột bị làm phẳng. Chú thích hình ảnh biến mất.

Giải pháp của ColPali: bỏ qua OCR, nhúng trực tiếp hình ảnh trang. Sử dụng tương tác muộn kiểu ColBERT để truy xuất để mô hình có thể chú ý đến các patch chi tiết tại thời điểm truy vấn.

## Khái niệm

### ColBERT (2020)

ColBERT (Khattab & Zaharia, arXiv:2004.12832) là một phương pháp truy xuất văn bản. Thay vì một vector cho mỗi tài liệu, nó tạo ra một vector cho mỗi token. Tại thời điểm truy vấn:

- Các token truy vấn có embedding riêng (N_q vector).
- Các token tài liệu có embedding (N_d vector, thường được lưu cache).
- Điểm số = tổng các giá trị lớn nhất của độ tương đồng cosine giữa các token truy vấn và token tài liệu: Σ_i max_j cos(q_i, d_j).

Đây là thao tác MaxSim. Mỗi token truy vấn "chọn" token tài liệu khớp nhất với nó. Điểm số cuối cùng là tổng của các giá trị này.

Ưu điểm: khả năng thu hồi mạnh, xử lý ngữ nghĩa ở cấp độ thuật ngữ. Nhược điểm: N_d vector cho mỗi tài liệu, chi phí lưu trữ đắt đỏ.

### ColPali

ColPali (Faysse và cộng sự, arXiv:2407.01449) áp dụng mô hình ColBERT cho hình ảnh.

- Mỗi trang được PaliGemma (ViT + ngôn ngữ) mã hóa thành các patch embeddings: N_p vector cho mỗi trang.
- Mỗi truy vấn người dùng (văn bản) được mã hóa thành các embedding token truy vấn: N_q vector.
- Điểm số = Σ_i max_j cos(q_i, p_j), tức là MaxSim giữa các token văn bản truy vấn và các patch hình ảnh trang.
- Truy xuất top-k trang dựa trên tổng điểm.

Tại thời điểm nạp tài liệu: nhúng mọi trang bằng PaliGemma, lưu trữ tất cả patch embeddings. Tại thời điểm truy vấn: nhúng các token truy vấn, tính MaxSim với tất cả các embedding trang đã lưu, trả về top-k trang.

Ưu điểm: hiệu suất end-to-end vượt trội hơn text-RAG 20-40% trên các tài liệu giàu hình ảnh. Mỗi patch-vector nắm bắt được bố cục và nội dung cục bộ.

Nhược điểm: N_p patch × 4-byte float × D-dim vector cho mỗi trang = lưu trữ tăng nhanh. Được giảm thiểu bằng cách lượng tử hóa PQ / OPQ.

### ColQwen2 và ColSmol

ColQwen2 (illuin-tech, 2024-2025) thay thế PaliGemma bằng Qwen2-VL. Bộ mã hóa cơ sở tốt hơn, truy xuất tốt hơn.

ColSmol là biến thể quy mô nhỏ hơn cho mục đích sử dụng cục bộ / edge. Một bộ truy xuất ColSmol với khoảng 1B tham số có thể chạy trên GPU phổ thông.

### VisRAG

VisRAG (Yu và cộng sự, arXiv:2410.10594) là một biến thể khác: thay vì MaxSim trên các patch, nó gộp (pool) mỗi trang thành một vector duy nhất bằng VLM rồi truy xuất bằng bi-encoder. Lập chỉ mục nhanh hơn + lưu trữ nhỏ hơn, nhưng khả năng thu hồi yếu hơn.

Sự đánh đổi giữa chất lượng và chi phí: ColPali cho chất lượng, VisRAG cho quy mô.

### M3DocRAG

M3DocRAG (Cho và cộng sự, arXiv:2411.04952) mở rộng truy xuất đa phương thức sang suy luận đa trang, đa tài liệu. Truy xuất các trang trên nhiều tài liệu, tổng hợp ngữ cảnh đa trang cho VLM.

### ViDoRe — bộ benchmark

Bộ benchmark đồng hành của ColPali. Visual Document Retrieval Evaluation. Các tác vụ bao gồm báo cáo tài chính, bài báo khoa học, tài liệu hành chính, hồ sơ y tế, hướng dẫn sử dụng. Chỉ số: nDCG@5.

ColPali-v1 đạt khoảng 80% nDCG@5 trên ViDoRe; text-RAG trên cùng tài liệu đạt khoảng 50-60%.

### Quy trình RAG end-to-end

Đối với RAG vision-native:

1. Nạp dữ liệu: PDF → hình ảnh trang → mã hóa PaliGemma → lưu trữ tất cả patch embeddings.
2. Truy vấn: văn bản người dùng → embedding token truy vấn → MaxSim với tất cả các trang đã lập chỉ mục → top-k trang.
3. Tạo: hình ảnh top-k trang + truy vấn → VLM (Qwen2.5-VL hoặc Claude) → câu trả lời.

Không cần OCR. Hình ảnh, biểu đồ, phông chữ, bố cục đều được đưa vào câu trả lời.

### Tính toán lưu trữ

Một báo cáo tài chính 50 trang với 729 patch mỗi trang và embedding 128-dim:

- ColPali: 50 * 729 * 128 * 4 byte = ~18 MB thô, ~4 MB sau khi dùng PQ.
- Text-RAG: 50 đoạn * 768-dim * 4 byte = ~150 kB.

ColPali tốn lưu trữ gấp ~30 lần cho mỗi tài liệu. Ở quy mô lớn, OPQ / PQ giúp giảm xuống còn ~5-10 lần, thường là mức chấp nhận được.

### Khi nào text-RAG vẫn thắng

- Các tài liệu thuần văn bản không có tín hiệu bố cục (bài viết wiki, nhật ký trò chuyện). Text-RAG đơn giản hơn và rẻ hơn về lưu trữ.
- Các kho lưu trữ hàng triệu trang nơi lưu trữ là yếu tố chi phí chính.
- Các yêu cầu pháp lý nghiêm ngặt đòi hỏi văn bản OCR có thể trích xuất được đi kèm với việc truy xuất.

Đối với mọi thứ khác vào năm 2026 — báo cáo tài chính, bài báo khoa học, hợp đồng pháp lý, hồ sơ y tế, tài liệu UX — vision-native RAG chiếm ưu thế.

```figure
mm-maxsim
```

## Sử dụng

`code/main.py`:

- Bộ mã hóa patch đồ chơi: ánh xạ một "trang" (lưới nhỏ các vector đặc trưng) thành một mảng các patch embeddings.
- Bộ tính điểm MaxSim: tính điểm kiểu ColBERT giữa tập embedding token truy vấn và tập patch trang.
- Lập chỉ mục 5 trang đồ chơi, chạy 3 truy vấn, trả về top-k kèm điểm số.

## Triển khai

Bài học này tạo ra `outputs/skill-vision-rag-designer.md`. Với một dự án RAG tài liệu, hãy chọn ColPali / ColQwen2 / VisRAG / text-RAG và tính toán dung lượng lưu trữ.

## Bài tập

1. Một báo cáo thường niên 200 trang với 729 patch mỗi trang, embedding 128-dim, float 4-byte. Tính dung lượng lưu trữ thô và lưu trữ nén PQ (8x).

2. MaxSim là Σ_i max_j cos(q_i, p_j). Tổng này nắm bắt được điều gì mà độ tương đồng trung bình đơn giản không làm được?

3. ColPali lập chỉ mục các trang dưới dạng tập hợp patch. Điều gì sẽ thay đổi nếu chúng ta lập chỉ mục ở cấp độ từ (như ColBERT)? Những sự đánh đổi là gì?

4. Thiết kế quy trình end-to-end cho kho dữ liệu 1 triệu trang với ngân sách độ trễ 500ms mỗi truy vấn. Chọn ColQwen2 / VisRAG và giải thích lý do.

5. Đọc M3DocRAG (arXiv:2411.04952). Mô tả mô hình chú ý đa trang và cách nó khác với truy xuất ColPali đơn trang.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Late interaction | "Kiểu ColBERT" | Truy xuất sử dụng embedding theo token hoặc theo patch + MaxSim, không phải một vector tài liệu duy nhất |
| MaxSim | "Max-over-patches" | Với mỗi token truy vấn, chọn token tài liệu có độ tương đồng cao nhất; tổng hợp qua các truy vấn |
| Bi-encoder | "Single-vector" | Một vector cho mỗi tài liệu; nhanh hơn nhưng mất đi độ chi tiết |
| Multi-vector | "Many-vectors-per-doc" | Lưu trữ N_p vector cho mỗi tài liệu / trang; chi phí lưu trữ tăng nhưng khả năng thu hồi cải thiện |
| Patch embedding | "Page feature" | Một vector cho mỗi patch hình ảnh từ bộ mã hóa VLM, được lưu cache theo trang |
| ViDoRe | "Vision doc bench" | Bộ benchmark của ColPali cho truy xuất tài liệu hình ảnh |
| PQ quantization | "Product quantization" | Nén giúp duy trì độ tương đồng vector trong khi giảm dung lượng lưu trữ ~8x |

## Đọc thêm

- [Faysse và cộng sự — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu và cộng sự — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho và cộng sự — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)