# Watermarking — SynthID, Stable Signature, C2PA

> Ba công nghệ định hình nguồn gốc nội dung do AI tạo ra vào năm 2026. SynthID (Google DeepMind) — công nghệ watermarking hình ảnh ra mắt tháng 8/2023, văn bản+video tháng 5/2024 (Gemini + Veo), văn bản được mở mã nguồn vào tháng 10/2024 thông qua Responsible GenAI Toolkit, bộ phát hiện đa phương tiện hợp nhất vào tháng 11/2025 cùng với Gemini 3 Pro. Watermarking văn bản điều chỉnh xác suất lấy mẫu next-token một cách không thể nhận thấy; watermark hình ảnh/video có khả năng chống chịu với nén, cắt, bộ lọc, thay đổi tốc độ khung hình. Stable Signature (Fernandez và cộng sự, ICCV 2023, arXiv:2303.15435) — tinh chỉnh (fine-tune) bộ giải mã latent diffusion để mỗi đầu ra chứa một thông điệp cố định; hình ảnh tạo ra bị cắt (10% nội dung) vẫn được phát hiện >90% ở mức FPR<1e-6. Nghiên cứu tiếp theo "Stable Signature is Unstable" (arXiv:2405.07145, tháng 5/2024) — việc fine-tuning có thể loại bỏ watermark trong khi vẫn bảo toàn chất lượng. C2PA — tiêu chuẩn metadata có chữ ký mật mã, chống giả mạo (C2PA 2.2 Explainer 2025). Watermarking và C2PA bổ trợ cho nhau: metadata có thể bị xóa nhưng mang lại nguồn gốc phong phú hơn; watermark tồn tại qua quá trình chuyển mã nhưng mang ít thông tin hơn.

**Type:** Build
**Languages:** Python (stdlib, token-watermark embed + detect)
**Prerequisites:** Phase 10 · 04 (sampling), Phase 01 · 09 (information theory)
**Time:** ~75 phút

## Mục tiêu học tập

- Mô tả watermarking cấp độ token (phong cách SynthID-text) và cơ chế phát hiện.
- Mô tả Stable Signature và cuộc tấn công loại bỏ năm 2024 đã phá vỡ nó.
- Nêu vai trò của C2PA và lý do tại sao nó bổ trợ cho watermarking.
- Mô tả các hạn chế chính: tín hiệu đặc thù theo mô hình, khả năng chống chịu khi diễn giải lại (paraphrase), và các cuộc tấn công bảo toàn ý nghĩa (arXiv:2508.20228).

## Vấn đề

Giai đoạn 2023-2024 chứng kiến deepfake và nội dung do AI tạo ra thâm nhập vào bối cảnh chính trị và người tiêu dùng trên quy mô lớn. Watermarking là tín hiệu nguồn gốc kỹ thuật được đề xuất: đánh dấu các thế hệ nội dung tại thời điểm tạo ra, sau đó phát hiện chúng. Bằng chứng năm 2025: không có watermark nào là bền vững tuyệt đối, nhưng khi kết hợp với metadata C2PA, sự kết hợp này cung cấp một câu chuyện về nguồn gốc có thể sử dụng được.

## Khái niệm

### Watermarking văn bản (phong cách SynthID-text)

Cơ chế của Kirchenbauer và cộng sự năm 2023, được Google đưa vào sản xuất:

1. Tại mỗi bước giải mã, băm (hash) K token trước đó để tạo ra một phân vùng giả ngẫu nhiên của từ vựng thành các tập "xanh" và "đỏ".
2. Thiên vị lấy mẫu về phía tập xanh bằng cách thêm δ vào các logit xanh.
3. Nội dung tạo ra chứa nhiều token xanh hơn so với xác suất ngẫu nhiên.

Phát hiện: băm lại mỗi tiền tố, đếm các token xanh trong nội dung tạo ra, tính z-score. Z-score >0 đối với văn bản có watermark, ~0 đối với văn bản do con người viết.

Đặc tính:
- Không thể nhận thấy đối với người đọc (δ đủ nhỏ để mất mát chất lượng là không đáng kể).
- Có thể phát hiện nếu có quyền truy cập vào hàm phân vùng từ vựng.
- Không bền vững với việc diễn giải lại (paraphrase) — viết lại văn bản sẽ phá hủy tín hiệu.

SynthID-text được mở mã nguồn vào tháng 10/2024 thông qua Responsible GenAI Toolkit của Google.

### Stable Signature (hình ảnh)

Fernandez và cộng sự, ICCV 2023. Fine-tune bộ giải mã latent diffusion để mỗi hình ảnh tạo ra chứa một thông điệp nhị phân cố định được nhúng trong biểu diễn latent. Việc phát hiện được thực hiện bằng cách giải mã từ latent với một bộ giải mã thần kinh. Hình ảnh bị cắt (còn 10% nội dung) vẫn được phát hiện >90% ở mức FPR<1e-6.

Tháng 5/2024, "Stable Signature is Unstable" (arXiv:2405.07145): việc fine-tune bộ giải mã sẽ loại bỏ watermark trong khi vẫn bảo toàn chất lượng hình ảnh. Việc fine-tune đối kháng sau khi tạo ra rất rẻ; khả năng chống chịu đối kháng của watermark bị hạn chế.

### Bộ phát hiện hợp nhất SynthID (tháng 11/2025)

Cùng với Gemini 3 Pro: một bộ phát hiện đa phương tiện đọc các tín hiệu SynthID từ văn bản, hình ảnh, âm thanh và video trong một API duy nhất. Hợp nhất hệ sinh thái nguồn gốc của Google.

### C2PA

Liên minh về Nguồn gốc và Tính xác thực của Nội dung (Coalition for Content Provenance and Authenticity). Tiêu chuẩn metadata có chữ ký mật mã, chống giả mạo. C2PA 2.2 Explainer (2025). Một manifest C2PA ghi lại các tuyên bố về nguồn gốc (ai tạo, khi nào, các biến đổi nào) được ký bởi khóa của người tạo.

Bổ trợ cho watermarking:
- Metadata có thể bị xóa; watermark thì không (dễ dàng).
- Metadata phong phú (chuỗi nguồn gốc đầy đủ); watermark chỉ mang các bit thông tin.
- C2PA phụ thuộc vào sự chấp nhận của nền tảng; watermark được nhúng tự động.

Google tích hợp cả hai trong Tìm kiếm, Quảng cáo và "About this image".

### Hạn chế

- **Đặc thù theo mô hình.** Watermark SynthID chỉ đánh dấu các nội dung từ các mô hình có hỗ trợ SynthID. Nội dung từ một mô hình không có SynthID sẽ không có watermark, vì vậy "không có tín hiệu SynthID" không phải là bằng chứng của tính xác thực.
- **Diễn giải lại (Paraphrase).** Watermark văn bản không tồn tại được qua quá trình diễn giải lại bảo toàn ý nghĩa.
- **Tấn công biến đổi.** arXiv:2508.20228 (2025) cho thấy các cuộc tấn công bảo toàn ý nghĩa có thể phá hủy cả watermark văn bản và nhiều watermark hình ảnh.
- **Loại bỏ bằng fine-tune.** Theo "Stable Signature is Unstable", việc fine-tune sau khi tạo ra sẽ loại bỏ các watermark đã nhúng.

### EU AI Act Điều 50

Bộ quy tắc minh bạch về dán nhãn nội dung do AI tạo ra (dự thảo đầu tiên tháng 12/2025, dự thảo thứ hai tháng 3/2026, dự kiến hoàn thiện tháng 6/2026 theo [trang trạng thái của Ủy ban Châu Âu](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)). Bộ quy tắc vẫn đang ở dạng dự thảo tính đến tháng 4/2026 và mốc thời gian có thể thay đổi. Đây là lớp quy định yêu cầu lớp kỹ thuật. Deepfake phải được dán nhãn.

### Vị trí trong Phase 18

Bài 22-23 nói về những gì mô hình phát ra (dữ liệu riêng tư, tín hiệu nguồn gốc). Bài 27 bao gồm quản trị dữ liệu huấn luyện. Bài 24 là khung pháp lý yêu cầu các biện pháp kỹ thuật này.

```figure
an-watermark-greenlist
```

## Sử dụng

`code/main.py` xây dựng một watermark văn bản mẫu. Các token là số nguyên 0..N-1; lấy mẫu có watermark thiên vị về phía tập xanh được xác định bởi hàm băm. Một bộ phát hiện tính toán z-score của token xanh. Bạn có thể quan sát việc phát hiện ở các văn bản 1000 token, xem việc diễn giải lại phá hủy tín hiệu như thế nào, và đo tỷ lệ dương tính giả trên văn bản do con người viết.

## Triển khai

Bài học này tạo ra `outputs/skill-provenance-audit.md`. Với một nội dung được triển khai kèm tuyên bố nguồn gốc, nó kiểm định: cơ chế watermark (nếu có), chuỗi ký C2PA (nếu có), khả năng chống chịu đối kháng của từng loại, và phạm vi bao phủ theo từng phương thức.

## Bài tập

1. Chạy `code/main.py`. Báo cáo z-score cho văn bản 1000 token có watermark so với văn bản do con người viết. Xác định tỷ lệ dương tính giả tại ngưỡng tin cậy 95%.

2. Thực hiện một cuộc tấn công diễn giải lại thay thế 30% token bằng từ đồng nghĩa. Đo lại z-score.

3. Đọc Kirchenbauer và cộng sự 2023, Mục 6 về độ bền vững. Tại sao watermark văn bản thất bại khi diễn giải lại nhưng watermark hình ảnh lại tồn tại được khi cắt?

4. Thiết kế một hệ thống triển khai sử dụng SynthID-text + metadata C2PA. Mô tả chuỗi nguồn gốc mà người tiêu dùng nhìn thấy. Xác định một chế độ lỗi của từng thành phần.

5. Kết quả "Stable Signature is Unstable" năm 2024 cho thấy fine-tuning loại bỏ watermark hình ảnh. Thiết kế một kiểm soát triển khai hạn chế cuộc tấn công này — ví dụ, yêu cầu các bản phát hành checkpoint đã fine-tune phải có chữ ký.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| SynthID | "Watermark của Google" | Tín hiệu nguồn gốc đa phương thức; văn bản, hình ảnh, âm thanh, video |
| Token watermark | "Phong cách Kirchenbauer" | Watermark văn bản lấy mẫu thiên vị, phát hiện qua z-score token xanh |
| Stable Signature | "Watermark hình ảnh" | Watermark bộ giải mã đã fine-tune; ICCV 2023 |
| C2PA | "Tiêu chuẩn metadata" | Metadata nguồn gốc có chữ ký mật mã, chống giả mạo |
| Paraphrase robustness | "Diễn giải lại có phá vỡ nó không" | Đặc tính watermark văn bản; hiện tại còn hạn chế |
| Fine-tune removal | "Gỡ watermark đối kháng" | Cuộc tấn công loại bỏ watermark hình ảnh qua fine-tune bộ giải mã |
| Cross-modal detector | "SynthID hợp nhất" | API hợp nhất tháng 11/2025 trên các phương thức |

## Đọc thêm

- [Kirchenbauer và cộng sự — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — cơ chế token-watermark
- [Fernandez và cộng sự — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — bài báo về watermark hình ảnh
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — cuộc tấn công loại bỏ
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — watermark đa phương thức
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — tiêu chuẩn metadata