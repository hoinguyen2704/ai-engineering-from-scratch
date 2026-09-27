# Data Provenance và Quản trị Dữ liệu Huấn luyện

> EU AI Act yêu cầu các tiêu chuẩn từ chối (opt-out) có thể đọc được bằng máy cho GPAI trước tháng 8 năm 2025 (thông qua ngoại lệ TDM của Chỉ thị Bản quyền EU). California AB 2013 (được ký năm 2024) — quy định về tính minh bạch của dữ liệu huấn luyện Generative AI yêu cầu các nhà phát triển công bố bản tóm tắt các tập dữ liệu với 12 trường bắt buộc. Sự đồng bộ của DPA năm 2025 về lợi ích hợp pháp: Irish DPC (21 tháng 5 năm 2025) chấp nhận việc Meta huấn luyện LLM trên nội dung công khai của người dùng trưởng thành tại EU/EEA với các biện pháp bảo vệ sau ý kiến của EDPB; Tòa án Cấp cao Cologne (23 tháng 5 năm 2025) bác bỏ lệnh cấm; Hamburg DPA hủy bỏ tính khẩn cấp; UK ICO (23 tháng 9 năm 2025) đưa ra phản hồi pháp lý tích cực đối với các biện pháp bảo vệ huấn luyện AI của LinkedIn (tính minh bạch, opt-out đơn giản hóa, kéo dài thời gian phản đối) và tiếp tục giám sát — không phải là sự phê duyệt chính thức. ANPD của Brazil (2 tháng 7 năm 2024) đã đình chỉ việc xử lý của Meta do thiếu minh bạch thông tin; biện pháp phòng ngừa này đã được dỡ bỏ vào ngày 30 tháng 8 năm 2024 sau khi Meta nộp kế hoạch tuân thủ. Vấn đề bất khả nghịch cốt lõi: các khung đồng ý cookie được thiết kế cho việc theo dõi thời gian thực, có thể đảo ngược; một khi dữ liệu đã nằm trong trọng số mô hình, việc xóa bỏ triệt để là không thể — không có quyền xóa bỏ (right-to-erasure) thực tế theo GDPR cho các mạng thần kinh đã được huấn luyện. Cửa sổ tuân thủ nằm ở thời điểm thu thập. Data Provenance Initiative (dataprovenance.org, Longpre, Mahari, Lee và cộng sự, "Consent in Crisis", tháng 7 năm 2024): kiểm toán quy mô lớn cho thấy sự suy giảm nhanh chóng của dữ liệu chung (data commons) cho AI khi các nhà xuất bản thêm các hạn chế robots.txt.

**Type:** Learn
**Languages:** Python (stdlib, trình tạo khung 12 trường theo California AB 2013)
**Prerequisites:** Phase 18 · 24 (quy định), Phase 18 · 26 (thẻ mô hình)
**Time:** ~60 phút

## Mục tiêu Học tập

- Mô tả 12 trường bắt buộc của California AB 2013 về tính minh bạch của dữ liệu huấn luyện Generative AI.
- Nêu quan điểm của DPA năm 2025 về việc huấn luyện LLM dựa trên lợi ích hợp pháp (Irish DPC, UK ICO, Hamburg, Cologne).
- Mô tả vấn đề bất khả nghịch: tại sao quyền xóa bỏ của GDPR không có tương đương thực tế cho các mạng thần kinh đã được huấn luyện.
- Nêu phát hiện "Consent in Crisis" của Data Provenance Initiative.

## Vấn đề

Quản trị dữ liệu huấn luyện là thượng nguồn của mọi thẻ mô hình (Bài 26) và nghĩa vụ pháp lý (Bài 24). Trong giai đoạn 2024-2025, bối cảnh pháp lý đã hợp nhất dựa trên ba nguyên tắc: cơ sở hạ tầng opt-out, công bố theo từng tập dữ liệu và các điều chỉnh về lợi ích hợp pháp cho dữ liệu công khai. Các nhà cung cấp không tuân thủ tại thời điểm thu thập sẽ không thể khắc phục ở hạ nguồn.

## Khái niệm

### California AB 2013

Được ký năm 2024. Tài liệu phải được đăng vào hoặc trước ngày 1 tháng 1 năm 2026 cho các hệ thống được phát hành vào hoặc sau ngày 1 tháng 1 năm 2022. Mục 3111(a) yêu cầu các nhà phát triển công bố bản tóm tắt cấp cao về các tập dữ liệu được sử dụng trong huấn luyện với 12 mục theo luật định:
1. Nguồn hoặc chủ sở hữu của các tập dữ liệu.
2. Mô tả cách các tập dữ liệu thúc đẩy mục đích dự định của hệ thống AI.
3. Số lượng điểm dữ liệu trong các tập dữ liệu (chấp nhận phạm vi chung; ước tính cho các tập dữ liệu động).
4. Mô tả các loại điểm dữ liệu (loại nhãn cho các tập dữ liệu có nhãn; đặc điểm chung cho dữ liệu không nhãn).
5. Liệu các tập dữ liệu có bao gồm bất kỳ dữ liệu nào được bảo vệ bởi bản quyền, nhãn hiệu, bằng sáng chế hay hoàn toàn thuộc phạm vi công cộng.
6. Liệu các tập dữ liệu có được mua hoặc cấp phép hay không.
7. Liệu các tập dữ liệu có bao gồm thông tin cá nhân hay không (theo Cal. Civ. Code §1798.140(v)).
8. Liệu các tập dữ liệu có bao gồm thông tin người tiêu dùng tổng hợp hay không (theo Cal. Civ. Code §1798.140(b)).
9. Việc làm sạch, xử lý hoặc sửa đổi khác bởi nhà phát triển, với mục đích dự định.
10. Khoảng thời gian dữ liệu được thu thập, có thông báo nếu việc thu thập đang diễn ra.
11. Ngày các tập dữ liệu được sử dụng lần đầu trong quá trình phát triển.
12. Liệu hệ thống có sử dụng hoặc liên tục sử dụng tạo dữ liệu tổng hợp (synthetic data) hay không.

Mục 12 (dữ liệu tổng hợp) là điểm mới so với các bảng dữ liệu của Gebru và cộng sự năm 2018. Mục 7 (thông tin cá nhân) kích hoạt các nghĩa vụ của Đạo luật Quyền riêng tư (CPRA). Đạo luật miễn trừ các hệ thống an ninh/toàn vẹn, vận hành máy bay và an ninh quốc gia chỉ dành cho liên bang (Mục 3111(b)).

### EU AI Act (Bài 24) và opt-out TDM

Ngoại lệ khai thác văn bản và dữ liệu (TDM) của Chỉ thị Bản quyền EU cho phép huấn luyện trên nội dung công khai trừ khi chủ sở hữu quyền từ chối. Chương Bản quyền trong Bộ quy tắc thực hành GPAI của EU AI Act yêu cầu các nhà cung cấp GPAI tôn trọng các tín hiệu opt-out có thể đọc được bằng máy (robots.txt, tuyên bố "No AI Training" của C2PA, v.v.).

### Sự hội tụ của DPA năm 2025 về lợi ích hợp pháp

Irish DPC (21 tháng 5 năm 2025): Kế hoạch của Meta về việc huấn luyện trên nội dung công khai của người dùng trưởng thành tại EU/EEA đã được chấp nhận với các biện pháp bảo vệ sau ý kiến của EDPB. Tòa án Cấp cao Cologne (23 tháng 5 năm 2025) bác bỏ lệnh cấm đối với Meta: opt-out là đủ. Hamburg DPA hủy bỏ thủ tục khẩn cấp để đảm bảo tính nhất quán trên toàn EU. UK ICO (23 tháng 9 năm 2025) đã đưa ra phản hồi pháp lý tích cực — không phải là sự phê duyệt chính thức — đối với việc LinkedIn tiếp tục huấn luyện AI với các biện pháp bảo vệ tương tự và giám sát liên tục.

Nguyên tắc hội tụ: lợi ích hợp pháp có thể biện minh cho việc huấn luyện trên nội dung công khai của bên thứ nhất với cơ chế opt-out. Không cần sự đồng ý.

### ANPD của Brazil (Tháng 6 năm 2024)

Đã đình chỉ việc Meta xử lý dữ liệu người dùng Brazil để huấn luyện AI do thiếu minh bạch thông tin. Kết quả khác với các DPA của EU — ANPD ưu tiên tính minh bạch hơn tính hợp lệ của lợi ích hợp pháp.

### Vấn đề bất khả nghịch

Đồng ý cookie được thiết kế cho việc theo dõi thời gian thực, có thể đảo ngược. Dữ liệu huấn luyện thì khác: một khi dữ liệu đi vào trọng số mô hình, việc xóa bỏ triệt để là không thể. Huấn luyện lại từ đầu là cách khắc phục hoàn toàn duy nhất, và nó cực kỳ tốn kém.

Các biện pháp khắc phục một phần:
- **Unlearning.** Loại bỏ xấp xỉ; được đo lường bằng MIA (Bài 22).
- **Định vị dựa trên hàm ảnh hưởng (Influence function-based localization).** Xác định các trọng số bị ảnh hưởng nhiều nhất bởi dữ liệu; cập nhật có chọn lọc.
- **Fine-tune-suppression.** Huấn luyện mô hình từ chối các đầu ra bắt nguồn từ dữ liệu đó.

Không có cách nào giải quyết triệt để vấn đề. Cửa sổ tuân thủ nằm ở thời điểm thu thập.

### Data Provenance Initiative

dataprovenance.org. Longpre, Mahari, Lee và cộng sự "Consent in Crisis" (tháng 7 năm 2024): kiểm toán quy mô lớn về dữ liệu chung huấn luyện AI. Phát hiện: các nhà xuất bản đang thêm các hạn chế robots.txt với tốc độ ngày càng tăng. Dữ liệu chung có thể huấn luyện công khai đang co lại nhanh chóng. Giai đoạn 2023 -> 2024 chứng kiến khoảng 25% các nguồn huấn luyện hàng đầu thêm một số hạn chế. Hệ quả: khả năng cung cấp dữ liệu huấn luyện trong tương lai phụ thuộc vào các mô hình thu thập mới (cấp phép, tạo dữ liệu tổng hợp, khuyến khích tham gia).

### Vị trí trong Phase 18

Bài 26 là tài liệu cấp mô hình. Bài 27 là quản trị cấp tập dữ liệu. Cùng nhau, chúng xác định lớp minh bạch. Bài 28 vạch ra hệ sinh thái nghiên cứu làm việc về các câu hỏi này.

```figure
an-provenance-oneway
```

## Sử dụng

`code/main.py` tạo ra một khung tóm tắt tập dữ liệu 12 trường tuân thủ California AB 2013 cho một tập dữ liệu thử nghiệm. Bạn có thể điền vào các trường và quan sát những trường nào kích hoạt các nghĩa vụ về quyền riêng tư hoặc bản quyền.

## Triển khai

Bài học này tạo ra `outputs/skill-provenance-check.md`. Với một tập dữ liệu được sử dụng trong huấn luyện, nó kiểm tra phạm vi bao phủ 12 trường của AB 2013, sự tuân thủ cơ sở hạ tầng opt-out, sự đồng bộ với DPA và đánh giá rủi ro bất khả nghịch.

## Bài tập

1. Chạy `code/main.py`. Tạo bản tóm tắt 12 trường cho một tập dữ liệu thử nghiệm và xác định những trường nào chưa được chỉ định rõ ràng.

2. Opt-out TDM của Chỉ thị Bản quyền EU có thể đọc được bằng máy. Hãy đề xuất một định dạng tiêu chuẩn cho tín hiệu opt-out và so sánh nó với robots.txt và "No AI Training" của C2PA.

3. Đọc "Consent in Crisis" của Data Provenance Initiative (tháng 7 năm 2024). Mô tả ba danh mục nội dung bị hạn chế nhanh nhất và lập luận về một hệ quả kinh tế.

4. Sự đồng bộ của DPA năm 2025 chấp nhận lợi ích hợp pháp cho việc huấn luyện trên nội dung công khai. Hãy xây dựng một kịch bản mà lợi ích hợp pháp sẽ không đủ và xác định cơ sở pháp lý mà nhà cung cấp cần thay thế.

5. Phác thảo một bản kê khai nguồn gốc dữ liệu huấn luyện (training-data-provenance manifest) kết hợp với các trường của AB 2013 và chuỗi nguồn gốc được ký bởi C2PA cho mỗi tập dữ liệu. Xác định một rào cản kỹ thuật và một rào cản pháp lý.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| AB 2013 | "luật California" | Tính minh bạch dữ liệu huấn luyện Generative AI; 12 trường bắt buộc |
| Ngoại lệ TDM | "khai thác văn bản và dữ liệu" | Ngoại lệ dữ liệu huấn luyện của Chỉ thị Bản quyền EU với opt-out |
| Lợi ích hợp pháp | "cơ sở của EU" | Cơ sở Điều 6 GDPR có thể biện minh cho việc huấn luyện trên nội dung công khai |
| Tín hiệu Opt-out | "không huấn luyện (đọc được bằng máy)" | robots.txt, C2PA "No AI Training", TDM.Reservation |
| Bất khả nghịch | "không thể bỏ huấn luyện" | Dữ liệu trong trọng số mô hình không thể xóa bỏ triệt để |
| Unlearning | "loại bỏ xấp xỉ" | Các can thiệp sau huấn luyện để giảm sự phụ thuộc của mô hình vào dữ liệu cụ thể |
| Consent in Crisis | "kiểm toán DPI" | Phát hiện tháng 7 năm 2024 về việc tăng tốc các hạn chế robots.txt |

## Đọc thêm

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — Luật minh bạch dữ liệu huấn luyện Generative AI
- [EU AI Act + Bộ quy tắc thực hành GPAI (Bài 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — Chương Bản quyền
- [Longpre, Mahari, Lee và cộng sự — Consent in Crisis (dataprovenance.org, tháng 7 năm 2024)](https://www.dataprovenance.org/consent-in-crisis-paper) — Kiểm toán DPI
- [IAPP — Các sửa đổi GDPR trong EU Digital Omnibus (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — bối cảnh pháp lý