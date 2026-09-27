# Tuân thủ — SOC 2, HIPAA, GDPR, PCI-DSS, EU AI Act, ISO 42001

> Việc bao phủ đa khung tiêu chuẩn (multi-framework) là điều kiện tiên quyết để chốt các hợp đồng doanh nghiệp vào năm 2026. **EU AI Act**: có hiệu lực từ ngày 1 tháng 8 năm 2024. Hầu hết các yêu cầu đối với hệ thống rủi ro cao sẽ được thực thi từ ngày 2 tháng 8 năm 2026. Mức phạt lên tới 15 triệu Euro hoặc 3% tổng doanh thu hàng năm toàn cầu đối với các nghĩa vụ hệ thống rủi ro cao (Điều 99(4)); lên tới 35 triệu Euro hoặc 7% đối với các hành vi AI bị cấm (Điều 99(3)). Áp dụng trên toàn cầu nếu phục vụ người dùng tại EU. **Colorado AI Act**: có hiệu lực từ ngày 30 tháng 6 năm 2026 (bị hoãn từ tháng 2 năm 2026 theo SB25B-004) — yêu cầu đánh giá tác động đối với các hệ thống rủi ro cao, quyền kháng cáo các quyết định của AI. Virginia cũng có quy định tương tự cho các lĩnh vực tín dụng/việc làm/nhà ở/giáo dục. **SOC 2 Type II**: yêu cầu B2B AI trên thực tế (Type II, không phải Type I, đối với fintech). **GDPR**: mức phạt cụ thể về AI lớn nhất được ghi nhận là 30,5 triệu Euro đối với Clearview AI (Cơ quan bảo vệ dữ liệu Hà Lan, tháng 9 năm 2024); Cơ quan Garante của Ý đã phạt OpenAI 15 triệu Euro vào tháng 12 năm 2024 (sau đó bị hủy bỏ khi kháng cáo vào tháng 3 năm 2026). Việc ẩn danh PII theo thời gian thực tại lớp inference là tiêu chuẩn có thể bảo vệ được; việc làm sạch dữ liệu sau xử lý (post-processing) là không đủ. **HIPAA**: bắt buộc đối với lĩnh vực chăm sóc sức khỏe — không được gửi PHI đến các dịch vụ AI bên ngoài nếu không có BAA. **PCI-DSS**: việc bao phủ lớp tương tác AI đòi hỏi cấu hình + thỏa thuận hợp đồng, không phải tự động. **ISO 42001**: tiêu chuẩn quản trị AI mới nổi, đang trở thành yêu cầu mua sắm ngày càng phổ biến bên cạnh ISO 27001. Hồ sơ tham chiếu: OpenAI duy trì SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS cho các thành phần thanh toán của ChatGPT. Việc ánh xạ chéo giữa các khung giúp giảm bớt áp lực kiểm toán: các kiểm soát truy cập có thể ánh xạ qua ISO 27001 A.5.15-5.18, GDPR Điều 32, HIPAA §164.312(a).

**Type:** Học tập
**Languages:** (Python tùy chọn — tuân thủ là chính sách + quy trình, không phải mã nguồn)
**Prerequisites:** Giai đoạn 17 · 25 (Bảo mật), Giai đoạn 17 · 13 (Khả năng quan sát)
**Time:** ~60 phút

## Mục tiêu học tập

- Liệt kê bảy khung tuân thủ năm 2026 liên quan đến các sản phẩm LLM và khớp từng khung với phân khúc khách hàng tương ứng.
- Trích dẫn mốc thời gian thực thi EU AI Act (có hiệu lực tháng 8 năm 2024; thực thi rủi ro cao tháng 8 năm 2026) và mức trần phạt hai cấp (15 triệu Euro / 3% cho nghĩa vụ rủi ro cao, 35 triệu Euro / 7% cho các hành vi bị cấm).
- Giải thích lý do tại sao việc làm sạch PII sau xử lý là không đủ đối với GDPR và nêu tên việc ẩn danh tại lớp inference thời gian thực là tiêu chuẩn có thể bảo vệ được.
- Mô tả việc ánh xạ kiểm soát chéo giữa các khung (ví dụ: kiểm soát truy cập ánh xạ tới ISO 27001 A.5.15-5.18 + GDPR Điều 32 + HIPAA §164.312(a)).

## Vấn đề

Bộ phận mua sắm của một khách hàng doanh nghiệp yêu cầu SOC 2 Type II, GDPR, HIPAA BAA, ISO 27001 và "Tuyên bố tuân thủ EU AI Act". Nhóm của bạn hiện có SOC 2 Type I. Bạn còn sáu tháng nữa mới đạt Type II và chưa bắt đầu hồ sơ theo Điều 30 của GDPR.

Việc bao phủ đa khung không phải là vấn đề riêng của LLM — đó là vấn đề của doanh nghiệp SaaS, với các lớp bổ sung dành riêng cho LLM. Các nhóm mua sắm vào năm 2026 muốn một ma trận với mỗi hàng là một khung và mỗi cột là một kiểm soát, thay vì một tệp PDF.

## Khái niệm

### Bảy khung tuân thủ

| Khung | Phạm vi | Yêu cầu cụ thể cho LLM |
|-----------|-------|--------------------------|
| SOC 2 Type II | Cơ sở B2B SaaS | Kiểm soát quy trình được kiểm toán trong 6-12 tháng |
| HIPAA | Y tế Hoa Kỳ | BAA bắt buộc; PHI không được rời khỏi hạ tầng nếu không có thỏa thuận ký kết |
| GDPR | Người dùng EU | Ẩn danh PII thời gian thực; quyền chủ thể dữ liệu; hồ sơ Điều 30 |
| PCI-DSS | Dữ liệu thanh toán | Cấu hình + hợp đồng cho AI tiếp xúc với thanh toán |
| EU AI Act | Phục vụ người dùng EU | Phân loại mức độ rủi ro; hệ thống rủi ro cao: đánh giá sự phù hợp, tài liệu, ghi nhật ký |
| Colorado AI Act | Phục vụ cư dân CO | Đánh giá tác động; quyền kháng cáo |
| ISO 42001 | Quản trị AI | Mới nổi; đi kèm với ISO 27001 |

### Mốc thời gian EU AI Act

- 1 tháng 8 năm 2024: có hiệu lực.
- 2 tháng 2 năm 2025: thực thi các hành vi AI bị cấm.
- 2 tháng 8 năm 2026: thực thi đối với các hệ thống rủi ro cao (đánh giá sự phù hợp, tài liệu, ghi nhật ký).
- Tháng 8 năm 2027: hệ thống rủi ro cao trong các sản phẩm thuộc luật hài hòa.

Các mức rủi ro: Không thể chấp nhận (bị cấm), Rủi ro cao (sự phù hợp + ghi nhật ký), Rủi ro hạn chế (minh bạch), Rủi ro tối thiểu (không ràng buộc). Hầu hết B2B LLM SaaS thuộc mức rủi ro hạn chế; rủi ro cao áp dụng cho việc làm, tín dụng, giáo dục, thực thi pháp luật, di cư, dịch vụ thiết yếu.

Các mức phạt (Điều 99): lên tới 15 triệu Euro hoặc 3% doanh thu hàng năm toàn cầu đối với vi phạm nghĩa vụ hệ thống rủi ro cao (Điều 99(4)); lên tới 35 triệu Euro hoặc 7% đối với các hành vi AI bị cấm (Điều 99(3)); mức nào cao hơn sẽ được áp dụng.

### GDPR — ẩn danh thời gian thực là tiêu chuẩn

Làm sạch sau xử lý (ẩn danh PII sau khi LLM đã thấy dữ liệu) không phải là tư thế có thể bảo vệ được — mô hình đã thấy dữ liệu đó rồi. Ẩn danh tại lớp inference thời gian thực là tiêu chuẩn của năm 2026:

- Nhận diện thực thể trước khi gọi LLM.
- Token hóa nhất quán (phương pháp Mesh) để bảo toàn ngữ nghĩa.
- Chỉ lưu trữ các prompt đã được ẩn danh + dữ liệu thô nếu có sự đồng ý.

Thực thi gần đây: 30,5 triệu Euro đối với Clearview AI (Cơ quan bảo vệ dữ liệu Hà Lan, tháng 9 năm 2024) là mức phạt GDPR cụ thể về AI lớn nhất cho đến nay; 15 triệu Euro đối với OpenAI (Cơ quan Garante của Ý, tháng 12 năm 2024) là mức phạt cụ thể về LLM lớn nhất, mặc dù đã bị hủy bỏ khi kháng cáo vào tháng 3 năm 2026 và phán quyết vẫn đang được xem xét thêm. Các tuyên bố về làm sạch sau xử lý đã thất bại trong các cuộc kiểm toán.

### HIPAA — BAA không phải là tùy chọn

Bạn không thể gửi PHI đến các dịch vụ AI bên ngoài nếu không có Thỏa thuận cộng tác viên kinh doanh (BAA) đã ký. Cả ba nền tảng LLM của các nhà cung cấp đám mây lớn (Bedrock, Azure OpenAI, Vertex) đều cung cấp BAA. API trực tiếp của OpenAI cung cấp BAA. API trực tiếp của Anthropic cung cấp BAA. Hãy xác nhận trước khi gửi PHI.

### SOC 2 Type II

Type I: các kiểm soát được thiết kế và lập tài liệu.
Type II: các kiểm soát vận hành hiệu quả trong 6-12 tháng.

Mua sắm B2B vào năm 2026 mặc định yêu cầu Type II. Type I chỉ là bước khởi đầu; Type II là cánh cửa quyết định.

Các yếu tố thúc đẩy kiểm toán phổ biến: nhật ký truy cập (ai đã thấy gì), quản lý thay đổi (được triển khai như thế nào), đánh giá rủi ro (hàng quý), ứng phó sự cố (đã được kiểm thử chưa?). Nhật ký kiểm toán từ Giai đoạn 17 · 25 có thể tái sử dụng trực tiếp.

### Ánh xạ chéo giữa các khung

Một chính sách kiểm soát truy cập có thể đáp ứng nhiều kiểm soát của các khung khác nhau:

| Kiểm soát | Các khung |
|---------|-----------|
| Ghi nhật ký truy cập | ISO 27001 A.5.15-5.18, GDPR Điều 32, HIPAA §164.312(a) |
| Quản lý thay đổi | ISO 27001 A.8.32, PCI DSS Yêu cầu 6, Phạm vi thông báo vi phạm HIPAA |
| Mã hóa khi truyền tải | ISO 27001 A.8.24, GDPR Điều 32, HIPAA §164.312(e) |
| Quản lý bí mật | ISO 27001 A.8.19, PCI DSS Yêu cầu 8, SOC 2 CC6.1 |

Các công cụ tuân thủ (Drata, Vanta, Secureframe) tự động hóa việc ánh xạ này. Rất đáng giá khi triển khai ở quy mô lớn.

### ISO 42001 — mới nổi

Được công bố cuối năm 2023. Yêu cầu mua sắm ngày càng tăng bên cạnh ISO 27001. Khung quản trị AI bao gồm quản lý rủi ro, chất lượng dữ liệu, tính minh bạch, sự giám sát của con người.

### Hồ sơ tham chiếu của OpenAI

OpenAI duy trì SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS cho các thành phần thanh toán của ChatGPT. Đó là mức tiêu chuẩn doanh nghiệp vào năm 2026.

### Các con số cần ghi nhớ

- Mức phạt EU AI Act: lên tới 15 triệu Euro / 3% (nghĩa vụ rủi ro cao, Điều 99(4)); lên tới 35 triệu Euro / 7% (hành vi bị cấm, Điều 99(3)).
- Thực thi rủi ro cao EU AI Act: 2 tháng 8 năm 2026.
- Mức phạt GDPR cụ thể về AI lớn nhất được ghi nhận: 30,5 triệu Euro, Clearview AI (Cơ quan bảo vệ dữ liệu Hà Lan, tháng 9 năm 2024).
- Mức phạt GDPR cụ thể về LLM lớn nhất: 15 triệu Euro, OpenAI (Cơ quan Garante của Ý, tháng 12 năm 2024; bị hủy bỏ khi kháng cáo tháng 3 năm 2026).
- Khoảng thời gian SOC 2 Type II: 6-12 tháng vận hành các kiểm soát.
- Ngày hiệu lực Colorado AI Act: 30 tháng 6 năm 2026 (bị hoãn từ tháng 2 năm 2026 bởi SB25B-004).

```figure
i4-control-matrix
```

## Sử dụng

`code/main.py` là một bảng tính ánh xạ tuân thủ bằng Python — dựa trên một kiểm soát, liệt kê các khung mà nó đáp ứng.

## Triển khai

Bài học này tạo ra `outputs/skill-compliance-matrix.md`. Dựa trên phân khúc khách hàng và địa lý, chỉ định các khung và kiểm soát cần thiết.

## Bài tập

1. Khách hàng doanh nghiệp đầu tiên của bạn yêu cầu SOC 2 Type II, HIPAA BAA, tuyên bố EU AI Act. Tư thế tuân thủ tối thiểu khả thi để giành được hợp đồng là gì?
2. Phân loại ba sản phẩm LLM giả định theo các mức rủi ro của EU AI Act. Điều gì thay đổi ở mức rủi ro cao?
3. Bạn vô tình gửi PHI đến một nhà cung cấp mà không có BAA. Hãy thực hiện quy trình ứng phó sự cố.
4. Tranh luận liệu ISO 42001 có "cần thiết vào năm 2026" đối với một nhà cung cấp AI tầm trung hay không.
5. Ánh xạ các trường nhật ký kiểm toán LLM của bạn (Giai đoạn 17 · 25) tới ít nhất ba kiểm soát khung.

## Thuật ngữ chính

| Thuật ngữ | Cách mọi người nói | Ý nghĩa thực tế |
|------|----------------|------------------------|
| SOC 2 Type II | "kiểm soát được kiểm toán" | Các kiểm soát vận hành trong 6-12 tháng, được chứng thực độc lập |
| HIPAA BAA | "hợp đồng y tế" | Thỏa thuận cộng tác viên kinh doanh; bắt buộc đối với PHI |
| GDPR | "quyền riêng tư EU" | Ẩn danh PII thời gian thực là tiêu chuẩn có thể bảo vệ được năm 2026 |
| EU AI Act | "quy tắc AI EU" | Thực thi rủi ro cao tháng 8 năm 2026; 15 triệu Euro / 3% (nghĩa vụ rủi ro cao) — 35 triệu Euro / 7% (hành vi bị cấm) |
| Colorado AI Act | "luật AI tiểu bang Mỹ" | Hiệu lực 30 tháng 6 năm 2026 (bị hoãn bởi SB25B-004); đánh giá tác động |
| ISO 42001 | "quản trị AI" | Khung mới nổi cho rủi ro + minh bạch AI |
| ISO 27001 | "bảo mật ISMS" | Cơ sở hệ thống quản lý an toàn thông tin |
| Đánh giá sự phù hợp | "gói tài liệu EU AI" | Yêu cầu rủi ro cao: tài liệu, kiểm thử, ghi nhật ký |
| Ánh xạ chéo khung | "một kiểm soát, nhiều khung" | Một chính sách duy nhất đáp ứng nhiều kiểm soát khung |

## Đọc thêm

- [Bảo mật và quyền riêng tư của OpenAI](https://openai.com/security-and-privacy/) — hồ sơ tuân thủ tham chiếu.
- [GuardionAI — Tuân thủ LLM 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — Hướng dẫn kiểm toán SOC 2 Type 2 năm 2026: 10 kiểm soát AI](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [Văn bản chính thức EU AI Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — nguồn chính.
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — nguồn chính.
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — tiêu chuẩn hệ thống quản lý AI.