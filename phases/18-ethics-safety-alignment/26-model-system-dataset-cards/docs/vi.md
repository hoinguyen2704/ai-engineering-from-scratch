# Model, System, and Dataset Cards

> Ba định dạng tài liệu giúp cấu trúc tính minh bạch của AI. Model Cards (Mitchell và cộng sự, 2019) — giống như nhãn dinh dưỡng cho các mô hình: dữ liệu huấn luyện, phân tích định lượng phân tách, các cân nhắc về đạo đức, các lưu ý; chỉ có 0,3% các model card trên Hugging Face ghi lại các cân nhắc về đạo đức (Oreamuno và cộng sự, 2023). Datasheets for Datasets (Gebru và cộng sự, 2018, CACM) — động lực, thành phần, quy trình thu thập, dán nhãn, phân phối, bảo trì; tương tự như datasheet trong lĩnh vực điện tử. Data Cards (Pushkarna và cộng sự, Google 2022) — chi tiết phân lớp theo mô-đun (telescopic, periscopic, microscopic) đóng vai trò là các đối tượng ranh giới (boundary objects) cho nhiều đối tượng độc giả khác nhau. Các phát triển giai đoạn 2024-2025: tạo tự động thông qua LLM (CardGen, Liu và cộng sự, 2024); chi tiết trong model card tương quan với mức tăng lượt tải xuống lên tới 29% trên HF (Liang và cộng sự, 2024); các chứng thực có thể kiểm chứng (Laminator, Duddu và cộng sự, 2024); bổ sung báo cáo bền vững về carbon/nước (Jouneaux và cộng sự, tháng 7 năm 2025); các loại thẻ quy định theo EU/ISO đang dần xuất hiện. System Cards (Sidhpurwala 2024; tính minh bạch cấp hệ thống của Meta; "Blueprints of Trust" arXiv:2509.20394) — tài liệu hệ thống AI toàn diện (end-to-end) bao gồm các khả năng bảo mật, bảo vệ chống prompt-injection, phát hiện rò rỉ dữ liệu, và sự phù hợp với các giá trị nhân văn.

**Type:** Build
**Languages:** Python (stdlib, model-card + datasheet + system-card generator)
**Prerequisites:** Phase 18 · 18 (safety frameworks), Phase 18 · 24 (regulatory)
**Time:** ~60 phút

## Mục tiêu học tập

- Mô tả model card gốc của Mitchell và cộng sự (2019) và datasheet của Gebru và cộng sự (2018).
- Mô tả cấu trúc phân lớp telescopic/periscopic/microscopic của Data Cards.
- Mô tả System Cards và phạm vi bao phủ toàn diện (end-to-end) của chúng.
- Nêu ba phát triển trong giai đoạn 2024-2025 (tạo tự động, chứng thực có thể kiểm chứng, báo cáo bền vững).

## Vấn đề

Các khung quy định (Bài 24) và chính sách an toàn phòng thí nghiệm (Bài 18) đều yêu cầu tài liệu hóa. Các định dạng tài liệu đã phát triển từ cấp độ mô hình (model cards) sang cấp độ tập dữ liệu (datasheets) và cấp độ hệ thống (system cards). Mỗi định dạng giải quyết một phạm vi minh bạch khác nhau. Các công việc về tự động hóa và chứng thực có thể kiểm chứng trong giai đoạn 2024-2025 giải quyết vấn đề áp dụng lâu nay.

## Khái niệm

### Model Cards (Mitchell và cộng sự, 2019)

Các phần chính:
- Chi tiết mô hình.
- Mục đích sử dụng.
- Các yếu tố (yếu tố nhân khẩu học hoặc môi trường liên quan để đánh giá).
- Các chỉ số (metrics).
- Dữ liệu đánh giá.
- Dữ liệu huấn luyện.
- Phân tích định lượng (phân tách theo các yếu tố).
- Các cân nhắc về đạo đức.
- Các lưu ý và khuyến nghị.

Vấn đề áp dụng: Kiểm toán của Oreamuno và cộng sự (2023) về các model card trên Hugging Face cho thấy chỉ có 0,3% tài liệu ghi lại các cân nhắc về đạo đức.

### Datasheets for Datasets (Gebru và cộng sự, 2018)

Tương tự như datasheet điện tử. Các phần chính:
- Động lực (tại sao tập dữ liệu được tạo ra).
- Thành phần (nội dung bên trong).
- Quy trình thu thập (cách thức tập hợp).
- Dán nhãn (nếu có).
- Sử dụng (dự định, bị cấm, rủi ro).
- Phân phối.
- Bảo trì.

Được xuất bản trên CACM 2021. Datasheet là tài liệu thượng nguồn; model card phụ thuộc vào độ chính xác của datasheet.

### Data Cards (Pushkarna và cộng sự, Google 2022)

Chi tiết phân lớp theo mô-đun. Ba cấp độ thu phóng:
- **Telescopic.** Tóm tắt cấp cao cho người không chuyên.
- **Periscopic.** Tổng quan cấp trung cho các kỹ sư ML.
- **Microscopic.** Tài liệu chi tiết cấp tính năng cho kiểm toán viên.

Khung đối tượng ranh giới: các độc giả khác nhau trích xuất thông tin khác nhau từ cùng một tài liệu.

### System Cards

Phạm vi: hệ thống AI toàn diện bao gồm mô hình + ngăn xếp an toàn (safety stack) + ngữ cảnh triển khai. Các phần thường bao gồm:
- Khả năng bảo mật.
- Bảo vệ chống prompt-injection.
- Phát hiện rò rỉ dữ liệu.
- Sự phù hợp với các giá trị nhân văn đã nêu.
- Ứng phó sự cố.

Sidhpurwala 2024 và công việc về tính minh bạch cấp hệ thống của Meta. "Blueprints of Trust" (arXiv:2509.20394) chính thức hóa System Card như một phần bổ sung ở lớp triển khai cho Model Cards.

### Các phát triển giai đoạn 2024-2025

- **CardGen (Liu và cộng sự, 2024).** Tạo model card tự động thông qua LLM; báo cáo độ khách quan cao hơn nhiều so với các thẻ do con người viết trên các trường tiêu chuẩn của Mitchell 2019.
- **Tương quan lượt tải xuống (Liang và cộng sự, 2024).** Các model card chi tiết tương quan với tỷ lệ tải xuống cao hơn tới 29% trên HF — áp lực áp dụng hiện nay là do thị trường thúc đẩy, không chỉ do tuân thủ.
- **Laminator (Duddu và cộng sự, 2024).** Chứng thực có thể kiểm chứng thông qua phần cứng TEE / chữ ký mật mã — cho phép model card mang theo bằng chứng xác thực, không chỉ là một tuyên bố.
- **Tính bền vững (Jouneaux và cộng sự, tháng 7 năm 2025).** Bổ sung cho dấu chân carbon, nước và năng lượng tính toán; các tiêu chuẩn ISO mới nổi.
- **Regulatory cards.** Chương Minh bạch trong Bộ quy tắc thực hành GPAI của Đạo luật AI EU (Bài 24) yêu cầu model card như một tài liệu tuân thủ.

### Vị trí trong Giai đoạn 18

Bài 24-25 là các lớp quy định và CVE. Bài 26 là lớp tài liệu. Bài 27 là quản trị dữ liệu huấn luyện, là thượng nguồn của datasheet. Bài 28 là hệ sinh thái nghiên cứu tạo ra các đánh giá được tham chiếu trong các thẻ.

```figure
an-card-scopes
```

## Sử dụng

`code/main.py` tạo ra một model card, datasheet và system card tối giản cho một bản triển khai thử nghiệm. Mỗi loại tuân theo cấu trúc phần chính tắc. Bạn có thể kiểm tra định dạng và so sánh ba phạm vi này.

## Triển khai

Bài học này tạo ra `outputs/skill-card-audit.md`. Với một model card, datasheet hoặc system card, nó kiểm toán mức độ bao phủ của các phần, sự phân tách số liệu và liệu các chứng thực có thể kiểm chứng có hiện diện hay không.

## Bài tập

1. Chạy `code/main.py`. Kiểm tra các thẻ đã tạo. Xác định các phần còn yếu (chỉ chứa trình giữ chỗ) và chỉ định bằng chứng nào sẽ củng cố chúng.

2. Mở rộng model card với phân tích định lượng phân tách trên hai nhóm nhân khẩu học (Bài 20).

3. Đọc Oreamuno và cộng sự (2023) về tỷ lệ áp dụng 0,3%. Đề xuất một thay đổi cấu trúc đối với đặc tả model card để tăng tỷ lệ áp dụng các cân nhắc về đạo đức.

4. Laminator (Duddu và cộng sự, 2024) sử dụng TEE cho các chứng thực có thể kiểm chứng. Thiết kế một trường model card mang theo chứng thực mật mã của kết quả đánh giá và mô tả vai trò của người xác minh.

5. Viết một System Card (System Card, không phải Model Card) cho một trong các dự án trước đây của bạn hoặc một bản triển khai giả định. Xác định phần có giá trị cao nhất đối với các kiểm toán viên bên thứ ba.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Model Card | "thẻ Mitchell" | Tài liệu tiêu chuẩn của Mitchell và cộng sự 2019 cho các mô hình ML |
| Datasheet | "datasheet Gebru" | Tài liệu tiêu chuẩn của Gebru và cộng sự 2018 cho các tập dữ liệu |
| Data Card | "thẻ Pushkarna" | Tài liệu dữ liệu phân lớp theo mô-đun của Google 2022 |
| System Card | "thẻ triển khai" | Tài liệu hệ thống AI toàn diện bao gồm ngăn xếp an toàn |
| Boundary object | "nhiều độc giả, một tài liệu" | Khung Data Cards: cùng một tài liệu phục vụ nhiều đối tượng khác nhau |
| Verifiable attestation | "chứng thực Laminator" | Bằng chứng mật mã hoặc TEE đính kèm với một tuyên bố trong tài liệu |
| Sustainability field | "dấu chân carbon / nước" | Bổ sung mới năm 2025 cho kế toán môi trường |

## Đọc thêm

- [Mitchell và cộng sự — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — model card chính tắc
- [Gebru và cộng sự — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — bài báo về datasheet
- [Pushkarna và cộng sự — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — tài liệu dữ liệu phân lớp
- [Sidhpurwala và cộng sự — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — chính thức hóa System Card