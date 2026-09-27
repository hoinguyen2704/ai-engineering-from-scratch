# Rủi ro sử dụng kép (Dual-Use) — Nâng cao năng lực trong các lĩnh vực Cyber, Sinh học, Hóa học, Hạt nhân

> Bức tranh về rủi ro sử dụng kép năm 2026, xét theo từng lĩnh vực. Sinh học/Hóa học: Bài 17 đề cập đến WMDP; thử nghiệm thu thập vũ khí sinh học của Anthropic (mức tăng 2,53 lần) và cảnh báo trong Preparedness Framework v2 tháng 4 năm 2025 của OpenAI ("đang ở ngưỡng có thể hỗ trợ đáng kể cho những người mới bắt đầu tạo ra các mối đe dọa sinh học đã biết") đánh dấu điểm bùng phát. Cyber (báo cáo tháng 11 năm 2025 của Anthropic): Các tác nhân nhà nước có liên hệ với Trung Quốc đã sử dụng công cụ lập trình tác tử (agentic coding tool) của Claude để tự động hóa tới 90% một chiến dịch tấn công mạng, với sự can thiệp của con người chỉ trong 4-6 bước; chương trình thí điểm "truy cập tin cậy" (trusted access) của OpenAI cung cấp quyền truy cập năng lực cho các tổ chức an ninh đã qua kiểm duyệt để thực hiện các công việc phòng thủ sử dụng kép. Sự xói mòn khoảng cách thực thi trong hóa học/sinh học: hàng rào phòng thủ cổ điển từng là "chỉ riêng việc tiếp cận thông tin là không đủ". Các mô hình tiên phong có khả năng thị giác (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1) có thể quan sát video tại phòng thí nghiệm ướt (wet-lab) và đưa ra hiệu chỉnh theo thời gian thực. Tháng 12 năm 2025: OpenAI đã chứng minh GPT-5 lặp lại các thí nghiệm tại phòng thí nghiệm ướt, đạt được mức cải thiện hiệu suất 79 lần thông qua tối ưu hóa giao thức dựa trên AI. Mô hình người mới so với chuyên gia: AI mang lại mức tăng tương đối lớn hơn cho người mới nhưng mang lại năng lực tuyệt đối lớn hơn cho các chuyên gia.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 18 (safety frameworks), Phase 18 · 28 (ecosystem)
**Time:** ~75 phút

## Mục tiêu học tập

- Mô tả câu chuyện về sự nâng cao năng lực sinh học giai đoạn 2024-2025: "nâng cao nhẹ" -> "đang ở ngưỡng" -> "mức tăng 2,53 lần không đủ để loại trừ ASL-3."
- Mô tả báo cáo cyber tháng 11 năm 2025 của Anthropic: tự động hóa bởi các tác nhân liên kết với Trung Quốc lên tới 90% một chiến dịch tấn công mạng.
- Mô tả sự xói mòn khoảng cách thực thi trong hóa học/sinh học: hiệu chỉnh theo thời gian thực các thí nghiệm tại phòng thí nghiệm ướt thông qua thị giác máy tính.
- Nêu rõ sự bất đối xứng giữa mức tăng tương đối của người mới và năng lực tuyệt đối của chuyên gia, cùng ý nghĩa của nó đối với việc xây dựng các trường hợp an toàn (safety-case).

## Vấn đề

Bài 17 là phương pháp luận đo lường. Bài 30 là trạng thái đo lường năm 2026. Bức tranh đã thay đổi đáng kể từ năm 2024 đến cuối năm 2025: mỗi lĩnh vực đều vượt qua một ngưỡng mà các khung quản trị năm 2024 chưa dự đoán được.

## Khái niệm

### Câu chuyện về sự nâng cao năng lực sinh học/hóa học

Ba giai đoạn (nhắc lại từ Bài 17 để đảm bảo tính nhất quán):

1. **2024 "nâng cao nhẹ."** Các đánh giá Preparedness/RSP ban đầu báo cáo lợi thế nhỏ của người mới so với việc tìm kiếm trên internet.
2. **Tháng 4 năm 2025 "đang ở ngưỡng."** OpenAI PF v2 cảnh báo rằng các mô hình "đang ở ngưỡng có thể hỗ trợ đáng kể cho những người mới bắt đầu tạo ra các mối đe dọa sinh học đã biết."
3. **Thử nghiệm thu thập vũ khí sinh học năm 2025 của Anthropic.** Nghiên cứu có kiểm soát trên người mới; mức tăng 2,53 lần trong các tác vụ giai đoạn thu thập; không đủ để loại trừ ASL-3.

Sự thay đổi mang tính định tính: "nhẹ" đã tiến hóa thành "có khả năng hỗ trợ" trong vòng mười tám tháng, ngay cả khi không có đột phá về năng lực.

### Sự xói mòn khoảng cách thực thi trong hóa học/sinh học

Phòng thủ lịch sử: thông tin là cần thiết nhưng không đủ; kỹ năng thực thi giao thức là rào cản đối với người mới. Các mô hình tiên phong năm 2025 với khả năng thị giác đã phá vỡ một phần rào cản này:

- **Hiệu chỉnh giao thức theo thời gian thực.** GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1 có thể quan sát video tại phòng thí nghiệm ướt và gắn cờ các lỗi ngay trong quá trình thực hiện.
- **Chứng minh của OpenAI tháng 12 năm 2025.** GPT-5 lặp lại các thí nghiệm tại phòng thí nghiệm ướt đạt được mức cải thiện hiệu suất 79 lần thông qua tối ưu hóa giao thức.

Ý nghĩa: kỹ năng thực thi như một hàng rào phòng thủ đang bị xói mòn. Các rào cản về mua sắm và thiết bị vẫn còn, nhưng khoảng cách về kiến thức ngầm (tacit-knowledge) đang thu hẹp lại.

### Nâng cao năng lực Cyber (Tháng 11 năm 2025)

Báo cáo tháng 11 năm 2025 của Anthropic: Các tác nhân nhà nước liên kết với Trung Quốc đã sử dụng công cụ lập trình tác tử của Claude để tự động hóa 80-90% một chiến dịch tấn công mạng. Sự can thiệp của con người chỉ cần thiết trong 4-6 bước.

Ý nghĩa:
- Lập trình tác tử (agentic coding) là nguyên mẫu tự động hóa tấn công. Hỗ trợ cyber bằng AI trước đây bị giới hạn ở cấp độ đoạn mã; các quy trình làm việc tác tử tích hợp trinh sát, khai thác, hậu khai thác và đánh cắp dữ liệu.
- 4-6 bước can thiệp của con người là nút thắt cổ chai; các bước tiến năng lực trong tương lai sẽ giảm con số đó.
- Sử dụng kép trong phòng thủ: Chương trình thí điểm "truy cập tin cậy" của OpenAI cung cấp cho các tổ chức an ninh đã qua kiểm duyệt (các công ty ứng phó sự cố uy tín, chính phủ) quyền truy cập năng lực để phòng thủ. Sự bất đối xứng trong quyền truy cập sẽ có lợi cho bên phòng thủ nếu chương trình thí điểm được mở rộng.

### Hạt nhân

Đây là lĩnh vực ít được phân tích nhất trong bốn lĩnh vực CBRN trong các tài liệu công khai. Mô hình đe dọa khác biệt: việc thu thập vật liệu phân hạch chiếm ưu thế về độ khó, không phải thông tin. Việc nâng cao năng lực của AI ở tầng thông tin mang lại lợi ích hạn chế cho người mới trong thực tế. Không có báo cáo nào từ các phòng thí nghiệm lớn giai đoạn 2024-2025 xác định được ngưỡng cụ thể nào bị vượt qua trong lĩnh vực hạt nhân.

### Người mới so với chuyên gia

Một mô hình xuyên suốt cả bốn lĩnh vực:

- **Mức tăng tương đối của người mới.** Cao. Có tính nhân bội. Theo báo cáo sinh học năm 2025 của Anthropic, mức tăng là 2,53 lần.
- **Năng lực tuyệt đối của chuyên gia.** Trần năng lực cao. Một chuyên gia khai thác được nhiều hơn người mới vì chuyên gia biết cách đặt câu hỏi và cách diễn giải kết quả.

Ý nghĩa đối với các trường hợp an toàn: chỉ giải quyết mức tăng của người mới (thông qua bộ lọc đầu vào, từ chối, xử lý sự không chắc chắn) là không đủ để kiểm soát năng lực tuyệt đối của chuyên gia. Cần các biện pháp bổ sung: làm cứng quá trình gợi mở (elicitation-hardening), xóa bỏ năng lực (capability unlearning - Bài 17) và các giao thức kiểm soát (Bài 10).

### Tổng hợp liên lĩnh vực

| Lĩnh vực | 2024 | 2025 | Điểm bùng phát |
|---|---|---|---|
| Sinh học | nâng cao nhẹ | tăng 2,53 lần, tiếp cận ASL-3 | tự động hóa giai đoạn thu thập |
| Hóa học | nâng cao nhẹ | xói mòn khoảng cách thực thi qua thị giác | hiệu chỉnh phòng thí nghiệm ướt thời gian thực |
| Cyber | hỗ trợ mã nguồn | tự động hóa chiến dịch 80-90% | lập trình tác tử |
| Hạt nhân | hạn chế | hạn chế | nút thắt tiếp cận vật liệu vẫn tồn tại |

Ba lĩnh vực đã vượt qua các ngưỡng. Một lĩnh vực vẫn bị giới hạn bởi các rào cản phi thông tin.

### Vị trí của bài học này trong Giai đoạn 18

Bài 30 là bài học tổng kết: bức tranh sử dụng kép hiện tại mà mọi bài học trước đó đều góp phần đo lường, hạn chế hoặc quản trị. Các bài 17-18 cung cấp phương pháp đo lường và khung quản trị; các bài 12-16 cung cấp công cụ đánh giá; các bài 24-25 cung cấp tầng quy định và công bố thông tin; bài 28 cung cấp hệ sinh thái nghiên cứu. Bài 30 là nơi các bằng chứng được tập hợp lại.

```figure
an-uplift-asymmetry
```

## Sử dụng

Không có mã nguồn. Hãy đọc báo cáo về mối đe dọa cyber tháng 11 năm 2025 của Anthropic, bản cập nhật Preparedness Framework v2 tháng 4 năm 2025 của OpenAI và báo cáo tổng kết AI x Bio năm 2025 của Hội đồng Rủi ro Chiến lược (Council on Strategic Risks).

## Triển khai

Bài học này tạo ra `outputs/skill-dual-use-triage.md`. Với một tuyên bố về năng lực hoặc báo cáo sự cố năm 2026, nó sẽ phân loại trên bốn lĩnh vực và xác định liệu tuyên bố đó có ảnh hưởng đến mức tăng tương đối của người mới, năng lực tuyệt đối của chuyên gia, hay cả hai.

## Bài tập

1. Đọc báo cáo cyber tháng 11 năm 2025 của Anthropic. Liệt kê 4-6 bước can thiệp của con người và lập luận xem bước nào sẽ được tự động hóa đầu tiên trong mô hình thế hệ tiếp theo.

2. Khoảng cách thực thi hóa học/sinh học đang bị xói mòn thông qua thị giác. Hãy thiết kế một đánh giá đo lường sự nâng cao kiến thức ngầm mà không vi phạm các ranh giới ITAR/EAR.

3. Sự nâng cao năng lực hạt nhân dường như bị giới hạn bởi việc tiếp cận vật liệu. Hãy lập luận ủng hộ và phản đối quan điểm cho rằng một đột phá AI trong tương lai có thể thay đổi nút thắt này.

4. Xây dựng một trường hợp an toàn (ba trụ cột của Bài 18) cho một mô hình tiên phong có năng lực cyber nhằm giới hạn mức tăng năng lực của cả người mới và chuyên gia.

5. Chọn một trong bốn lĩnh vực và viết một đoạn dự báo cho năm 2027 dựa trên quỹ đạo 2024-2025. Xác định bằng chứng có thể bác bỏ dự báo của bạn.

## Thuật ngữ chính

| Thuật ngữ | Cách hiểu thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Uplift | "AI giúp kẻ tấn công" | Mức tăng năng lực của kẻ tấn công nhờ sự hỗ trợ của AI |
| Novice-relative uplift | "nhân bội" | Mức độ AI giúp người mới so với trạng thái hiện tại |
| Expert-absolute capability | "trần năng lực" | Năng lực tối đa mà chuyên gia có thể khai thác từ mô hình |
| Execution gap | "làm vs biết" | Phòng thủ lịch sử: kỹ năng phòng thí nghiệm ướt ngầm định ngăn cản người mới |
| Agentic coding | "tấn công tự chủ" | Thực thi tác vụ cyber tự chủ nhiều bước |
| Acquisition phase | "các bước tiền tổng hợp" | Các giai đoạn mua sắm, thiết bị, giấy phép của mối đe dọa sinh học |
| Trusted access | "thí điểm chỉ dành cho bên phòng thủ" | Chương trình năm 2025 của OpenAI cung cấp quyền truy cập năng lực cho bên phòng thủ đã qua kiểm duyệt |

## Đọc thêm

- [Anthropic — Báo cáo mối đe dọa cyber tháng 11 năm 2025](https://www.anthropic.com/news/disrupting-AI-espionage) — Tự động hóa chiến dịch liên kết với Trung Quốc
- [OpenAI — Preparedness Framework v2 (15 tháng 4, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — sinh học "đang ở ngưỡng"
- [Anthropic — RSP v3.0 (Tháng 2 năm 2026)](https://www.anthropic.com/responsible-scaling-policy) — các ngưỡng sinh học ASL-3
- [Council on Strategic Risks — Tổng kết AI x Bio năm 2025](https://councilonstrategicrisks.org/2025/12/22/2025-aixbio-wrapped-a-year-in-review-and-projections-for-2026/) — tổng hợp cuối năm