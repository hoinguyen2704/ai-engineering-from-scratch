# Anthropic Responsible Scaling Policy v3.0

> RSP v3.0 có hiệu lực từ ngày 24 tháng 2 năm 2026, thay thế cho chính sách năm 2023. Chính sách này áp dụng cơ chế giảm thiểu rủi ro hai tầng: những gì Anthropic sẽ thực hiện đơn phương so với những gì được định khung là khuyến nghị toàn ngành (bao gồm các tiêu chuẩn bảo mật RAND SL-4). Chính sách bổ sung Frontier Safety Roadmaps (Lộ trình An toàn Tiên phong) và Risk Reports (Báo cáo Rủi ro) dưới dạng các tài liệu thường trực thay vì các kết quả bàn giao một lần. Loại bỏ cam kết tạm dừng từ năm 2023. Giới thiệu ngưỡng AI R&D-4: một khi vượt qua ngưỡng này, Anthropic phải công bố một bản đánh giá khẳng định xác định các rủi ro sai lệch (misalignment) và các biện pháp giảm thiểu. Claude Opus 4.6 chưa vượt qua ngưỡng này. Trong thông báo về v3.0, Anthropic tuyên bố rằng "việc tự tin loại trừ khả năng này đang trở nên khó khăn". SaferAI đã xếp hạng RSP 2023 ở mức 2.2; họ đã hạ bậc v3.0 xuống 1.9, đưa Anthropic vào nhóm RSP "yếu" cùng với OpenAI và DeepMind. Các ngưỡng định tính đã thay thế cho các cam kết định lượng của năm 2023; việc loại bỏ điều khoản tạm dừng là sự thoái lui rõ rệt nhất.

**Type:** Learn
**Languages:** Python (stdlib, RSP threshold decision engine)
**Prerequisites:** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)
**Time:** ~45 phút

## Vấn đề

Các phòng thí nghiệm tiên phong (frontier labs) công bố các chính sách mở rộng quy mô (scaling policies) vừa là tài liệu kỹ thuật, vừa là tài liệu quản trị, vừa là tín hiệu gửi tới các cơ quan quản lý. RSP v3.0 là tài liệu hiện tại của Anthropic. Việc đọc kỹ tài liệu này rất quan trọng, không phải vì việc tuân thủ nó là bắt buộc (thực tế là không), mà vì cách định khung của nó định hình cách một phòng thí nghiệm hình dung về rủi ro thảm họa và cách họ truyền đạt các đánh đổi với công chúng.

Sự khác biệt giữa v3.0 và v2.0 là đơn vị phân tích hữu ích. Những gì đã được thêm vào: Frontier Safety Roadmaps, Risk Reports, ngưỡng AI R&D-4. Những gì đã bị loại bỏ: cam kết tạm dừng năm 2023. Những gì đã được định khung lại: lịch trình giảm thiểu rủi ro hai tầng được chia giữa hành động đơn phương của Anthropic và khuyến nghị toàn ngành. Đánh giá bên ngoài — SaferAI — đã hạ điểm từ 2.2 (v2) xuống 1.9 (v3.0). Đây là cách một chính sách mở rộng quy mô có thể trở nên kém nghiêm ngặt hơn trong khi trông có vẻ chỉn chu hơn.

## Khái niệm

### Lịch trình giảm thiểu rủi ro hai tầng

- **Các hành động đơn phương của Anthropic**: những gì Anthropic sẽ thực hiện bất kể các phòng thí nghiệm khác làm gì. Việc huấn luyện sẽ dừng lại khi vượt quá một ngưỡng nhất định, các biện pháp bảo mật cụ thể, các cổng triển khai cụ thể.
- **Các khuyến nghị toàn ngành**: những gì Anthropic cho rằng ngành nên thực hiện một cách tập thể. Bao gồm các tiêu chuẩn bảo mật RAND SL-4. Đây không phải là các cam kết từ phía Anthropic; chúng là các vận động chính sách.

Cấu trúc hai tầng không có trong v2. Điều này có nghĩa là người đọc cần xem xét mỗi cam kết nằm ở cột nào. Một biện pháp bảo mật trong cột "khuyến nghị toàn ngành" không phải là lời hứa của Anthropic; đó là hy vọng của Anthropic.

### Ngưỡng AI R&D-4

Đây là cấp độ năng lực mà RSP v3.0 xác định là ngưỡng quan trọng tiếp theo. Cụ thể: một mô hình có thể tự động hóa một phần đáng kể nghiên cứu AI với chi phí cạnh tranh. Một khi Anthropic tin rằng một mô hình đã vượt qua ngưỡng này, họ phải công bố một bản đánh giá khẳng định xác định các rủi ro sai lệch và các biện pháp giảm thiểu trước khi tiếp tục mở rộng quy mô.

Claude Opus 4.6 chưa vượt qua ngưỡng này theo thông báo v3.0. Tài liệu bổ sung: "việc tự tin loại trừ khả năng này đang trở nên khó khăn". Cách diễn đạt đó rất quan trọng; nó thừa nhận rằng ngưỡng này đủ gần để trở thành một mối quan tâm thực tế, không phải là một giới hạn suy đoán.

Bài học 6 (Automated Alignment Research) và Bài học 7 (Recursive Self-Improvement) đóng góp trực tiếp vào ngưỡng này. Việc các nhà nghiên cứu về căn chỉnh tự động (automated alignment) vượt qua các tiêu chuẩn chất lượng nghiên cứu là bằng chứng cho thấy ngưỡng AI R&D-4 đang đến gần.

### Frontier Safety Roadmaps và Risk Reports

v3.0 nâng cấp hai loại tài liệu thành các tài liệu thường trực:

- **Frontier Safety Roadmap**: tài liệu hướng tới tương lai mô tả công việc an toàn đã lên kế hoạch, kỳ vọng về năng lực và nghiên cứu giảm thiểu rủi ro.
- **Risk Report**: tài liệu hồi cứu về các mô hình cụ thể sau khi phát hành, mô tả năng lực quan sát được và rủi ro còn lại.

Cả hai đều được công khai. Cả hai đều được cập nhật theo định kỳ đã công bố. Tiện ích ở đây là: người đọc có thể theo dõi cách những gì Anthropic nói họ sẽ làm trong Roadmap so sánh với những gì họ báo cáo trong Risk Report.

### Loại bỏ điều khoản tạm dừng

RSP 2023 bao gồm một cam kết tạm dừng rõ ràng: nếu một mô hình vượt qua các ngưỡng năng lực cụ thể, việc huấn luyện sẽ tạm dừng cho đến khi các biện pháp giảm thiểu được áp dụng. v3.0 thay thế việc tạm dừng rõ ràng bằng một công thức nhẹ nhàng hơn (công bố bản đánh giá khẳng định, tiếp tục nếu các biện pháp giảm thiểu là đầy đủ). SaferAI và các nhà phân tích khác đã chỉ trích trực tiếp điều này như là sự thoái lui mạnh mẽ nhất trong tài liệu mới.

Lập luận chính sách cho sự thay đổi này: các ngưỡng định lượng vào năm 2023 hóa ra không thể đạt được theo các tiêu chuẩn năng lực của năm 2026 vì bản thân các tiêu chuẩn đó đã được điều chỉnh lại. Lập luận phản bác: một điều khoản tạm dừng trong chính sách mở rộng quy mô là một cơ chế cam kết; việc loại bỏ nó làm mất đi uy tín của chính sách.

### Sự hạ bậc của SaferAI

SaferAI là một tổ chức độc lập xếp hạng các tài liệu theo phong cách RSP. Xếp hạng công khai của họ: RSP 2023 của Anthropic đạt 2.2 (trên thang điểm 4.0 là RSP tốt nhất hiện tại và 1.0 là danh nghĩa). v3.0 đạt 1.9. Điều này đã đưa Anthropic từ mức "trung bình" xuống "yếu", gia nhập nhóm yếu cùng với OpenAI và DeepMind.

Các yếu tố hạ bậc theo SaferAI:
- Các ngưỡng định tính thay thế cho các ngưỡng định lượng.
- Cam kết tạm dừng bị loại bỏ.
- Các biện pháp giảm thiểu cho ngưỡng AI R&D-4 được mô tả là "bản đánh giá khẳng định" thay vì các biện pháp cụ thể.
- Các cơ chế đánh giá phụ thuộc vào Nhóm Cố vấn An toàn của Anthropic, với sự giám sát độc lập hạn chế.

### Bài học này không nói về điều gì

Đây không phải là bài học về tuân thủ. RSP v3.0 không phải là một quy định; không có gì buộc Anthropic phải tuân theo nó. Bài học này nằm ở việc đọc tài liệu với sự cụ thể và hoài nghi mà nó xứng đáng nhận được. Các chính sách mở rộng quy mô là tín hiệu công khai chính mà các phòng thí nghiệm tiên phong phát ra về tư thế đối với rủi ro thảm họa. Đọc hiểu chúng tốt là một kỹ năng thực tế cho bất kỳ ai có công việc phụ thuộc vào các năng lực tiên phong.

```figure
a5-rsp-ladder
```

## Sử dụng

`code/main.py` triển khai một công cụ ra quyết định nhỏ phản ánh hình thái đánh giá ngưỡng của RSP: với một mô hình ứng viên và một tập hợp các phép đo năng lực, hãy trả về việc ngưỡng AI R&D-4 có bị vượt qua hay không, các phần cần thiết của bản đánh giá khẳng định và liệu việc triển khai có thể tiếp tục hay không. Nó được thiết kế đơn giản một cách có chủ đích; mục đích là làm cho logic của tài liệu trở nên rõ ràng.

## Triển khai

`outputs/skill-scaling-policy-review.md` đánh giá một chính sách mở rộng quy mô (Anthropic, OpenAI, DeepMind hoặc nội bộ) dựa trên tài liệu tham khảo v3.0: cấu trúc hai tầng, các ngưỡng, cam kết tạm dừng, đánh giá độc lập.

## Bài tập

1. Chạy `code/main.py`. Nhập ba mô hình tổng hợp ở các cấp độ năng lực khác nhau. Xác nhận bộ đánh giá ngưỡng hoạt động như mong đợi và tạo ra mẫu bản đánh giá khẳng định phù hợp.

2. Đọc toàn bộ RSP v3.0 (32 trang). Xác định mọi cam kết nằm trong tầng "khuyến nghị toàn ngành". Những cam kết nào trong số đó sẽ là "hành động đơn phương của Anthropic" trong v2?

3. Đọc phương pháp xếp hạng RSP của SaferAI. Tái tạo điểm số 1.9 của họ cho v3.0 bằng cách áp dụng thang điểm của họ vào tài liệu. Hàng nào trong thang điểm đã dẫn đến việc hạ bậc nhiều nhất?

4. Cam kết tạm dừng năm 2023 đã bị loại bỏ. Hãy đề xuất một cam kết thay thế giúp duy trì uy tín của chính sách trong khi vẫn thừa nhận vấn đề điều chỉnh lại tiêu chuẩn năm 2026.

5. So sánh RSP v3.0 với OpenAI Preparedness Framework v2 (Bài học 20). Chọn một lĩnh vực mà v3.0 mạnh hơn. Chọn một lĩnh vực mà Preparedness Framework mạnh hơn.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| RSP | "Chính sách mở rộng quy mô của Anthropic" | Responsible Scaling Policy; v3.0 có hiệu lực từ 24/02/2026 |
| AI R&D-4 | "Ngưỡng tự động hóa nghiên cứu" | Năng lực tự động hóa nghiên cứu AI đáng kể với chi phí cạnh tranh |
| Affirmative case | "Biện minh an toàn" | Lập luận được công bố rằng các rủi ro đã được xác định và các biện pháp giảm thiểu là đầy đủ |
| Frontier Safety Roadmap | "Kế hoạch tương lai" | Tài liệu thường trực về công việc an toàn đã lên kế hoạch và các năng lực dự kiến |
| Risk Report | "Hồi cứu về một mô hình" | Tài liệu thường trực về năng lực quan sát được và rủi ro còn lại sau khi phát hành |
| Two-tier mitigation | "Đơn phương so với ngành" | Các cam kết của Anthropic so với các khuyến nghị của ngành, được tách biệt |
| Pause commitment | "Điều khoản 2023" | Lời hứa rõ ràng về việc tạm dừng huấn luyện; đã bị loại bỏ trong v3.0 |
| SaferAI rating | "Điểm RSP độc lập" | Thang điểm của bên thứ ba; v3.0 đạt 1.9 (v2 là 2.2) |

## Đọc thêm

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — toàn bộ chính sách 32 trang.
- [Anthropic — Thông báo RSP v3.0](https://www.anthropic.com/news/responsible-scaling-policy-v3) — tóm tắt các thay đổi từ v2.
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — tài liệu thường trực được liên kết từ RSP v3.0.
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — hồi cứu về mô hình tiên phong hiện tại.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — kết nối AI R&D-4 với quyền tự chủ được đo lường.