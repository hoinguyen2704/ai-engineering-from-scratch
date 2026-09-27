# WMDP và Đánh giá năng lực sử dụng kép (Dual-Use)

> Li và cộng sự, "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning" (ICML 2024, arXiv:2403.03218). 4.157 câu hỏi trắc nghiệm thuộc các lĩnh vực an ninh sinh học (1.520), an ninh mạng (2.225) và hóa học (412). Các câu hỏi nằm trong "vùng vàng" (yellow zone) — kiến thức hỗ trợ gần, được lọc qua quy trình đánh giá bởi nhiều chuyên gia và tuân thủ các quy định pháp lý ITAR/EAR. Mục đích kép: đánh giá đại diện (proxy) cho năng lực sử dụng kép và là chuẩn mực cho việc "unlearning" (phương pháp RMU đi kèm giúp giảm hiệu suất WMDP trong khi vẫn bảo toàn năng lực tổng quát). Tường thuật thực địa giai đoạn 2024-2025: các đánh giá ban đầu của OpenAI/Anthropic vào năm 2024 báo cáo "mức độ cải thiện nhẹ" so với tìm kiếm trên internet; đến tháng 4 năm 2025, Preparedness Framework v2 của OpenAI cho biết các mô hình đang "tiệm cận mức hỗ trợ đáng kể cho những người mới bắt đầu tạo ra các mối đe dọa sinh học đã biết." Thử nghiệm thu thập vũ khí sinh học của Anthropic cho thấy mức cải thiện 2,53 lần, không đủ để loại trừ ASL-3.

**Type:** Learn
**Languages:** Python (stdlib, WMDP-shaped uplift evaluation harness)
**Prerequisites:** Phase 18 · 16 (red-team tooling), Phase 14 (agent engineering)
**Time:** ~60 phút

## Mục tiêu học tập

- Mô tả ba lĩnh vực của WMDP, số lượng câu hỏi và tiêu chí lọc "vùng vàng".
- Giải thích RMU và lý do tại sao WMDP vừa là một chuẩn đánh giá vừa là chuẩn mực cho unlearning.
- Mô tả tường thuật về mức độ cải thiện (uplift) giai đoạn 2024-2025: "cải thiện nhẹ" -> "tiệm cận" -> "không đủ để loại trừ ASL-3."
- Phân biệt giữa mức độ cải thiện tương đối đối với người mới và năng lực tuyệt đối của chuyên gia.

## Vấn đề

Năng lực sử dụng kép là vấn đề đo lường trong mọi khung an toàn biên (frontier safety framework) của các phòng thí nghiệm (Bài 18). Câu hỏi đặt ra là: liệu mô hình X có thực sự nâng cao khả năng của một người mới trong việc gây ra thiệt hại quy mô lớn về sinh học, hóa học hoặc mạng hay không? Việc đo lường trực tiếp (yêu cầu mô hình thực hiện hành vi gây hại) là bất hợp pháp và phi đạo đức. Việc đo lường đại diện cần một chuẩn mực mà mô hình không thể từ chối (để có được các con số năng lực trung thực) nhưng các câu hỏi trong đó không phải là các tài liệu hướng dẫn gây hại.

## Khái niệm

### "Vùng vàng" (The "yellow zone")

Các câu hỏi đòi hỏi kiến thức hỗ trợ gần về một quy trình gây hại mà không phải là công thức tổng hợp trực tiếp. Ví dụ: "Chất phản ứng nào xúc tác cho bước 4 của [quy trình đã công bố]?" thay vì "làm thế nào để tôi tạo ra [hợp chất nguy hiểm]?". Mỗi câu hỏi được xem xét bởi nhiều chuyên gia trong lĩnh vực; được lọc để tuân thủ kiểm soát xuất khẩu ITAR/EAR.

Tổng cộng 4.157 câu hỏi:
- An ninh sinh học: 1.520
- An ninh mạng: 2.225
- Hóa học: 412

Định dạng trắc nghiệm. Các mô hình trả lời mà không bị yêu cầu hỗ trợ bất cứ điều gì; năng lực có thể được đo lường mà không cần kích hoạt hành vi gây hại.

### RMU — Representation Misdirection for Unlearning

Phương pháp unlearning đi kèm. Được áp dụng cho LLaMa-2-7B, giúp giảm điểm số WMDP xuống gần mức ngẫu nhiên trong khi vẫn bảo toàn MMLU và các chuẩn mực năng lực tổng quát khác trong phạm vi vài phần trăm. Phương pháp được công bố này là chuẩn mực unlearning cho mọi bài báo về unlearning sinh-hóa-mạng sau đó.

### Tường thuật về mức độ cải thiện (uplift) giai đoạn 2024-2025

Ba giai đoạn:

1. **2024 "cải thiện nhẹ."** Các đánh giá Preparedness/RSP ban đầu của OpenAI và Anthropic báo cáo những lợi thế nhỏ so với tìm kiếm trên internet đối với những người mới thực hiện các tác vụ liên quan đến sinh học. Cách diễn đạt công khai: các mô hình biên có hỗ trợ, nhưng không đáng kể hơn Google.

2. **Tháng 4 năm 2025 "tiệm cận."** Preparedness Framework v2 của OpenAI báo cáo các mô hình đang "tiệm cận mức hỗ trợ đáng kể cho những người mới bắt đầu tạo ra các mối đe dọa sinh học đã biết." Đây không phải là tuyên bố về năng lực — mà là cảnh báo rằng ngưỡng đó đã rất gần.

3. **Thử nghiệm thu thập vũ khí sinh học năm 2025 của Anthropic.** Nghiên cứu có kiểm soát với những người tham gia là người mới, đo lường mức độ thành công tương đối ở các tác vụ giai đoạn thu thập. Báo cáo mức cải thiện 2,53 lần. Không đủ để loại trừ ASL-3 (Bài 18) — ngưỡng cho cấp độ 3 của Chính sách mở rộng quy mô có trách nhiệm (Responsible Scaling Policy) của Anthropic đã đạt được hoặc tiệm cận.

### Người mới (novice-relative) vs Chuyên gia (expert-absolute)

Một sự phân biệt quan trọng:

- **Mức độ cải thiện tương đối đối với người mới.** Mô hình giúp ích bao nhiêu cho một người không chuyên? Mang tính nhân bội. Lợi thế tương đối cao vì người mới biết rất ít; ngay cả thông tin khiêm tốn cũng giúp ích.
- **Năng lực tuyệt đối của chuyên gia.** Mô hình tạo ra bao nhiêu thông tin ở mức nỗ lực tối đa? Một chuyên gia có thể trích xuất nhiều hơn một người mới. Trần năng lực tuyệt đối là rất cao.

Các trường hợp an toàn (Bài 18) nhắm vào cả hai: "mô hình không thể cung cấp cho người mới đủ mức cải thiện để thực hiện" cộng với "chuyên gia không thể trích xuất thông tin từ mô hình mà chưa được công bố."

### Bẫy đo lường

WMDP là một đại diện năng lực, không phải là phép đo triển khai. Một mô hình đạt điểm cao trên WMDP có thể hoặc không thể bị người mới khai thác trong thực tế, tùy thuộc vào:
- Khả năng chống lại sự kích hoạt (khó khăn như thế nào để lấy được năng lực mà không kích hoạt các bộ lọc an toàn)
- Kiến thức ngầm (năng lực đòi hỏi kỹ năng phòng thí nghiệm, không phải thông tin)
- Rào cản thực thi (mua sắm, thiết bị)

Thử nghiệm thu thập vũ khí sinh học năm 2025 của Anthropic bổ sung lớp kích hoạt của người mới lên trên năng lực kiểu WMDP: nó đo lường sự thành công thực tế của tác vụ, không phải năng lực trắc nghiệm.

### Vị trí trong Giai đoạn 18

Các bài 12-16 là công cụ tấn công và phòng thủ trên đầu ra của mô hình. Bài 17 là lớp năng lực sử dụng kép — phép đo mà các khung an toàn biên (Bài 18) đánh giá. Bài 30 khép lại lộ trình với bằng chứng về mức độ cải thiện trong lĩnh vực mạng/sinh/hóa/hạt nhân năm 2026.

```figure
al-wmdp-yellow-zone
```

## Sử dụng

`code/main.py` xây dựng một bộ công cụ đánh giá mô phỏng WMDP. Một mô hình giả định được kiểm tra trên các câu hỏi được phân loại theo danh mục; điểm số trên mỗi lĩnh vực được báo cáo. Một biện pháp can thiệp unlearning đơn giản (xóa bỏ biểu diễn đặc thù của lĩnh vực) làm giảm điểm số; bạn có thể đo lường sự đánh đổi so với năng lực tổng quát.

## Triển khai

Bài học này tạo ra `outputs/skill-wmdp-eval.md`. Với một tuyên bố về năng lực sử dụng kép ("mô hình của chúng tôi không hỗ trợ đáng kể cho vũ khí sinh học"), nó kiểm toán: những chuẩn mực nào đã được chạy, lộ trình từ chối nào được sử dụng để đánh giá (hoàn thành thô so với kiểm soát chính sách), và liệu các nghiên cứu về kích hoạt của người mới có bổ sung cho kết quả trắc nghiệm hay không.

## Bài tập

1. Chạy `code/main.py`. Báo cáo độ chính xác theo từng lĩnh vực trước và sau bước unlearning giả định. Giải thích sự đánh đổi về năng lực tổng quát.

2. Bổ sung vào WMDP giả định một lĩnh vực thứ tư (ví dụ: phóng xạ). Chỉ định hai loại câu hỏi minh họa trong vùng vàng. Giải thích tại sao việc tạo ra các câu hỏi như vậy khó hơn việc thêm các câu hỏi kiểu MMLU.

3. Đọc WMDP 2024 Phần 5 (phương pháp RMU). Phác thảo một phương pháp unlearning đơn giản hơn (ví dụ: triệt tiêu các neuron top-k cho nội dung lĩnh vực) và mô tả chi phí năng lực tổng quát dự kiến.

4. Thử nghiệm thu thập vũ khí sinh học năm 2025 của Anthropic báo cáo mức cải thiện 2,53 lần. Mô tả hai cách con số này có thể bị chệch lên trên (quy mô mẫu người mới, độ trung thực của tác vụ) và hai cách chệch xuống dưới (trần kích hoạt, kiểm soát an toàn mô hình).

5. Trình bày rõ ràng những gì một trường hợp an toàn cho ASL-3 yêu cầu ngoài việc vượt qua bài kiểm tra unlearning WMDP. Nêu tên ít nhất hai nghiên cứu kích hoạt bổ sung.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| WMDP | "chuẩn mực sử dụng kép" | 4.157 câu hỏi trắc nghiệm về sinh/mạng/hóa trong vùng vàng |
| Vùng vàng | "hỗ trợ nhưng không tổng hợp" | Kiến thức gần với năng lực gây hại mà không phải là công thức tổng hợp |
| RMU | "chuẩn mực unlearning" | Representation Misdirection for Unlearning; giảm điểm WMDP, bảo toàn năng lực tổng quát |
| Mức cải thiện tương đối | "giúp người không chuyên bao nhiêu" | Lợi thế nhân bội so với tìm kiếm internet hiện tại cho người mới |
| Năng lực tuyệt đối | "trần cho chuyên gia" | Thông tin tối đa có thể trích xuất từ mô hình bởi một chuyên gia có động cơ |
| Tác vụ giai đoạn thu thập | "các bước trước tổng hợp" | Mua sắm, thiết bị, giấy phép — những phần sớm nhất của lộ trình gây hại |
| ITAR/EAR | "tuân thủ kiểm soát xuất khẩu" | Khung pháp lý hạn chế việc công bố một số kiến thức hỗ trợ nhất định |

## Đọc thêm

- [Li và cộng sự — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218) — bài báo về chuẩn mực và RMU
- [OpenAI — Preparedness Framework v2 (15/04/2025)](https://openai.com/index/updating-our-preparedness-framework/) — ngôn ngữ "tiệm cận"
- [Anthropic — Responsible Scaling Policy v3.0 (02/2026)](https://www.anthropic.com/responsible-scaling-policy) — ngưỡng sinh học ASL-3 và kết quả thử nghiệm thu thập
- [DeepMind — Frontier Safety Framework v3.0 (09/2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL về mức độ cải thiện sinh học