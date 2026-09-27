# Hiểu về Tài liệu và Sơ đồ (Document and Diagram Understanding)

> Tài liệu không phải là ảnh chụp. Một tệp PDF, bài báo khoa học, hóa đơn hay biểu mẫu viết tay đều có bố cục, bảng biểu, sơ đồ, chú thích, tiêu đề và cấu trúc ngữ nghĩa mà việc hiểu ảnh thông thường không thể nắm bắt được. Stack tiền-VLM là một pipeline: Tesseract OCR + LayoutLMv3 + các heuristic trích xuất bảng. Làn sóng VLM đã thay thế điều đó bằng các mô hình không cần OCR — Donut (2022), Nougat (2023), DocLLM (2023) — xuất ra markup có cấu trúc trực tiếp. Đến năm 2026, biên giới công nghệ chỉ đơn giản là "đưa ảnh trang tài liệu vào Claude Opus 4.7 ở độ phân giải gốc 2576px", và kết quả markup có cấu trúc sẽ có được miễn phí. Bài học này đọc hiểu quá trình phát triển ba giai đoạn của AI tài liệu.

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180 phút

## Mục tiêu học tập

- Giải thích ba giai đoạn của AI tài liệu: Pipeline OCR, không cần OCR (OCR-free), và VLM-native.
- Mô tả ba luồng đầu vào của LayoutLMv3: văn bản, bố cục (bbox), các patch ảnh, với cơ chế masking thống nhất.
- So sánh Donut (OCR-free, ảnh → markup), Nougat (bài báo khoa học → LaTeX), DocLLM (tạo sinh có nhận thức bố cục), PaliGemma 2 (VLM-native).
- Chọn mô hình tài liệu cho một tác vụ mới (hóa đơn, bài báo khoa học, biểu mẫu viết tay, biên lai tiếng Trung).

## Vấn đề

"Hiểu tệp PDF này" khó hơn vẻ ngoài của nó. Thông tin nằm ở:

- Nội dung văn bản (90% tín hiệu).
- Bố cục (tiêu đề, chú thích, thanh bên, định dạng hai cột).
- Bảng biểu (hàng, cột, ô gộp).
- Hình ảnh và sơ đồ.
- Chú thích viết tay.
- Phông chữ và kiểu chữ (tiêu đề so với nội dung).

OCR thô xuất ra văn bản và làm mất đi phần còn lại. Một hệ thống quan tâm đến hóa đơn cần biết "Tổng cộng: $1,245" đến từ góc dưới bên phải, chứ không phải từ một chú thích.

## Khái niệm

### Giai đoạn 1 — Pipeline OCR (trước 2021)

Stack cổ điển:

1. PDF → ảnh cho mỗi trang.
2. Tesseract (hoặc OCR thương mại) trích xuất văn bản với bounding box cho từng từ.
3. Bộ phân tích bố cục xác định các khối (tiêu đề, bảng, đoạn văn).
4. Bộ nhận diện cấu trúc bảng phân tích các bảng.
5. Quy tắc miền + regex trích xuất các trường.

Hoạt động tốt với văn bản in sạch. Thất bại với chữ viết tay, bản quét bị lệch, bảng phức tạp, các ngôn ngữ không phải tiếng Anh. Mỗi chế độ lỗi đều yêu cầu một đường dẫn ngoại lệ tùy chỉnh.

### TrOCR (2021)

TrOCR (Li và cộng sự, arXiv:2109.10282) đã thay thế CNN-CTC cổ điển của Tesseract bằng một transformer encoder-decoder được huấn luyện trên ảnh văn bản tổng hợp + thực tế. Một chiến thắng rõ ràng trên văn bản viết tay và đa ngôn ngữ. Vẫn là một pipeline (phát hiện rồi đến TrOCR rồi đến bố cục), nhưng bước OCR đã cải thiện đáng kể.

### Giai đoạn 2 — Không cần OCR (2022-2023)

Các mô hình không cần OCR đầu tiên cho biết: bỏ qua hoàn toàn việc phát hiện, ánh xạ pixel ảnh trực tiếp sang đầu ra có cấu trúc.

Donut (Kim và cộng sự, arXiv:2111.15664):
- Encoder-decoder transformer, encoder là Swin-B.
- Đầu ra là JSON cho hiểu biểu mẫu, markdown cho tóm tắt, hoặc bất kỳ schema đặc thù tác vụ nào.
- Không OCR, không bố cục, không phát hiện.

Nougat (Blecher và cộng sự, arXiv:2308.13418):
- Được huấn luyện đặc biệt trên các bài báo khoa học.
- Đầu ra là LaTeX / markdown.
- Xử lý phương trình, bố cục nhiều cột, hình ảnh.
- Mô hình mà mọi trình phân tích arXiv đều gọi.

Đây là các chuyên gia, không phải mô hình tổng quát. Donut trên bài báo khoa học sẽ thất bại; Nougat trên hóa đơn sẽ thất bại.

### LayoutLMv3 (2022)

Một hướng đi khác. LayoutLMv3 (Huang và cộng sự, arXiv:2204.08387) vẫn giữ OCR nhưng bổ sung hiểu biết về bố cục:

- Ba luồng đầu vào: token văn bản OCR, bounding box 2D cho mỗi token, các patch ảnh.
- Mục tiêu huấn luyện masked trên cả ba phương thức (văn bản bị mask, patch bị mask, bố cục bị mask).
- Tác vụ hạ nguồn: phân loại, trích xuất thực thể, QA bảng.

LayoutLMv3 là đỉnh cao của hiểu tài liệu dựa trên OCR. Mạnh về biểu mẫu và hóa đơn. Yêu cầu OCR ở thượng nguồn. Độ chính xác tiền-VLM tốt nhất trên các benchmark tài liệu tiêu chuẩn.

### DocLLM (2023)

DocLLM (Wang và cộng sự, arXiv:2401.00908) là người anh em tạo sinh của LayoutLM. Tạo ra các câu trả lời tự do dựa trên các token bố cục. Tốt hơn cho QA trên tài liệu; vẫn phụ thuộc vào đầu vào OCR.

### Giai đoạn 3 — VLM-native (2024+)

Các VLM năm 2024 đã trở nên đủ tốt để thay thế hoàn toàn pipeline. Đưa ảnh toàn trang ở độ phân giải cao vào VLM, đặt câu hỏi, nhận câu trả lời.

- LLaVA-NeXT 336-tile AnyRes hoạt động cho các tài liệu nhỏ.
- Qwen2.5-VL với độ phân giải động xử lý 2048+ pixel một cách tự nhiên.
- Claude Opus 4.7 hỗ trợ tài liệu 2576px.
- PaliGemma 2 (tháng 4 năm 2025) huấn luyện đặc biệt cho tài liệu + chữ viết tay.

Khoảng cách giữa VLM-native và pipeline OCR đã thu hẹp nhanh chóng. Đến năm 2026, VLM-native thắng thế về:

- Văn bản trong cảnh (viết tay + in ấn, hỗn hợp ngôn ngữ).
- Bảng phức tạp với các ô gộp.
- Phương trình toán học nhúng trong văn bản.
- Hình ảnh có chú thích văn bản.

Các pipeline OCR vẫn thắng thế về:

- Khối lượng công việc quét thuần túy ở quy mô lớn nơi độ trễ mỗi trang là quan trọng.
- Độ tin cậy của pipeline (lỗi xác định so với ảo giác của VLM).
- Môi trường được quản lý yêu cầu đầu ra OCR có thể kiểm toán.

### Biên giới Claude 4.7 / GPT-5

Ở đầu vào gốc 2576-pixel, các VLM biên giới thực hiện hiểu tài liệu với độ chính xác gần bằng con người. Các con số benchmark từ đầu năm 2026:

- DocVQA: Claude 4.7 ~95.1, PaliGemma 2 ~88.4, Nougat ~77.3, LayoutLMv3 pipeline ~83.
- ChartQA: Claude 4.7 ~92.2, GPT-4V ~78.
- VisualMRC: Claude 4.7 ~94.

Khoảng cách mô hình đóng chủ yếu là độ phân giải và quy mô base-LLM. Các mô hình mở ở mức 7B kém hơn vài điểm nhưng đang bắt kịp.

### Phương trình toán học và đầu ra LaTeX

Các bài báo khoa học cần đầu ra LaTeX chính xác cho các phương trình. Nougat đã được huấn luyện về điều này. Các VLM được huấn luyện với mục tiêu LaTeX (Qwen2.5-VL-Math, các dẫn xuất của Nougat) tạo ra LaTeX có thể sử dụng được. Nếu không có huấn luyện LaTeX rõ ràng, các VLM tạo ra các bản phiên âm dễ đọc nhưng không chính xác.

Đối với các pipeline bài báo khoa học năm 2026: kết hợp Nougat trên PDF, sau đó là VLM trên các trang khó.

### Chữ viết tay

Vẫn là tác vụ phụ khó nhất. Hỗn hợp in ấn + viết tay (ghi chú của bác sĩ, biểu mẫu đã điền) là nơi các pipeline OCR vẫn đánh bại VLM về chi phí. Các VLM chỉ dành cho chữ viết tay đang cải thiện (Claude 4.7, PaliGemma 2).

### Công thức năm 2026

Cho một dự án AI tài liệu mới:

- Hóa đơn in ấn thuần túy ở quy mô lớn: LayoutLMv3 + quy tắc, tiết kiệm chi phí.
- Tài liệu hỗn hợp (khoa học + viết tay + biểu mẫu): VLM-native (PaliGemma 2 hoặc Qwen2.5-VL).
- Nhập liệu toàn bộ arXiv: Nougat cho toán học, VLM cho hình ảnh.
- Quy định pháp lý: Pipeline OCR + VLM validator để kiểm tra chéo.

```figure
mm-doc-layout
```

## Sử dụng

`code/main.py`:

- Một tokenizer nhận thức bố cục đơn giản: với các cặp (văn bản, bbox), tạo ra đầu vào kiểu LayoutLMv3.
- Một trình tạo schema tác vụ kiểu Donut: mẫu JSON cho biểu mẫu.
- So sánh ngân sách token mỗi trang giữa pipeline OCR, Donut, Nougat và VLM-native.

## Triển khai

Bài học này tạo ra `outputs/skill-document-ai-stack-picker.md`. Với một dự án AI tài liệu (miền, quy mô, chất lượng, quy định), chọn lựa giữa pipeline OCR, chuyên gia không cần OCR, và VLM-native.

## Bài tập

1. Dự án của bạn là 10 triệu hóa đơn mỗi ngày. Stack nào giảm thiểu chi phí mỗi trang mà không làm mất độ chính xác?

2. Tại sao LayoutLMv3 vượt trội hơn các VLM thuần CLIP trên QA biểu mẫu nhưng lại kém hơn ở văn bản trong cảnh? Luồng bbox từ bỏ điều gì?

3. Nougat tạo ra LaTeX. Hãy đề xuất một trường hợp kiểm thử mà đầu ra VLM-native đánh bại Nougat về độ trung thực của LaTeX, và một trường hợp mà Nougat thắng.

4. Đọc bài báo PaliGemma 2 (Google, 2024). Sự bổ sung dữ liệu huấn luyện chính nào đã nâng cao độ chính xác tài liệu so với PaliGemma 1?

5. Thiết kế một hệ thống lai an toàn về mặt quy định: Pipeline OCR là chính, VLM là kiểm tra chéo phụ. Bạn giải quyết sự bất đồng như thế nào?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| OCR pipeline | "Kiểu Tesseract" | Stack theo giai đoạn: phát hiện -> OCR -> bố cục -> quy tắc; xác định, dễ vỡ |
| OCR-free | "Kiểu Donut" | Transformer ảnh-sang-đầu ra bỏ qua OCR rõ ràng; mô hình đơn lẻ |
| Layout-aware | "LayoutLM" | Đầu vào bao gồm tọa độ bbox cho mỗi token; masking thống nhất qua các phương thức |
| VLM-native | "Frontier VLM" | Đưa ảnh trang trực tiếp vào VLM Claude/GPT/Qwen ở độ phân giải cao; không pipeline |
| DocVQA | "Doc benchmark" | Tiêu chuẩn VQA tài liệu; điểm số được trích dẫn nhiều nhất |
| Markup output | "LaTeX / MD" | Định dạng đầu ra có cấu trúc thay vì văn bản tự do; cho phép tự động hóa hạ nguồn |

## Đọc thêm

- [Li và cộng sự — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher và cộng sự — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang và cộng sự — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim và cộng sự — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang và cộng sự — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)