# Các Khung Quản trị An toàn Tiên phong — RSP, PF, FSF

> Ba khung quản trị của các phòng thí nghiệm lớn định hình nền quản trị ngành AI tiên phong năm 2026. Chính sách Quy mô Trách nhiệm (Responsible Scaling Policy - RSP) v3.0 của Anthropic (tháng 2 năm 2026) giới thiệu các Cấp độ An toàn AI (ASL-1 đến ASL-5+), được mô phỏng theo các cấp độ an toàn sinh học, với ASL-3 được kích hoạt từ tháng 5 năm 2025 cho các mô hình liên quan đến CBRN. Khung Chuẩn bị (Preparedness Framework - PF) v2 của OpenAI (tháng 4 năm 2025) xác định năm tiêu chí cho các năng lực được theo dõi và tách biệt Báo cáo Năng lực khỏi Báo cáo Biện pháp Bảo vệ. Khung An toàn Tiên phong (Frontier Safety Framework - FSF) v3.0 của DeepMind (tháng 9 năm 2025) giới thiệu các Cấp độ Năng lực Quan trọng (Critical Capability Levels - CCL), bao gồm một CCL mới về Thao túng Có hại (Harmful Manipulation). Cả ba khung hiện đều bao gồm các điều khoản điều chỉnh theo đối thủ cạnh tranh, cho phép trì hoãn nếu các phòng thí nghiệm ngang hàng phát hành sản phẩm mà không có các biện pháp bảo vệ tương đương. Sự đồng bộ giữa các phòng thí nghiệm mang tính cấu trúc chứ không phải thuật ngữ: "Capability Thresholds" (Ngưỡng năng lực), "High Capability thresholds" (Ngưỡng năng lực cao) và "Critical Capability Levels" (Cấp độ năng lực quan trọng) đều biểu thị các cấu trúc tương đương.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 07-09 (deception failures)
**Time:** ~75 phút

## Mục tiêu Học tập

- Mô tả cấu trúc cấp độ ASL của Anthropic và những gì đã kích hoạt ASL-3.
- Liệt kê năm tiêu chí trong Khung Chuẩn bị v2 của OpenAI cho các năng lực được theo dõi.
- Mô tả cấu trúc Cấp độ Năng lực Quan trọng của DeepMind và CCL về Thao túng Có hại.
- Giải thích các điều khoản điều chỉnh theo đối thủ cạnh tranh và lý do tại sao chúng quan trọng đối với động lực chạy đua.
- Định nghĩa một "safety case" (hồ sơ an toàn) và mô tả cấu trúc ba trụ cột (giám sát, tính không rõ ràng, tính không có năng lực).

## Vấn đề

Các bài học 7-17 đã thiết lập rằng sự lừa dối là có thể xảy ra, năng lực lưỡng dụng là tồn tại và việc đánh giá có những giới hạn. Một phòng thí nghiệm sở hữu mô hình có năng lực tiên phong cần một cấu trúc quản trị nội bộ để:
- Xác định các ngưỡng khi nào cần các biện pháp bảo vệ mới.
- Xác định các đánh giá bắt buộc trước khi mở rộng quy mô.
- Mô tả hình thức của một hồ sơ an toàn (safety case).
- Xử lý vấn đề động lực chạy đua (nếu đối thủ phát hành mà không có biện pháp bảo vệ, bạn sẽ làm gì?).

Ba khung quản trị năm 2025-2026 là những tiêu chuẩn hiện đại nhất — dù chưa hoàn hảo, đang phát triển và đủ đồng bộ giữa các phòng thí nghiệm để câu hỏi quản trị hiện nay là liệu các khung này có đầy đủ hay không, thay vì liệu chúng có tồn tại hay không.

## Khái niệm

### Chính sách Quy mô Trách nhiệm (RSP) v3.0 của Anthropic (Tháng 2 năm 2026)

Cấu trúc ASL:
- ASL-1: không phải mô hình tiên phong (được bao hàm bởi các mô hình cơ sở yếu hơn mức tiên phong).
- ASL-2: mức cơ sở tiên phong hiện tại; được triển khai với các biện pháp bảo vệ thông thường.
- ASL-3: rủi ro lạm dụng thảm khốc cao hơn đáng kể; các năng lực liên quan đến CBRN. Được kích hoạt từ tháng 5 năm 2025.
- ASL-4: AI R&D-2 vượt ngưỡng; các mô hình có thể tự động hóa nghiên cứu AI cấp độ đầu vào.
- ASL-5+: AI R&D nâng cao; các mô hình tăng tốc đáng kể việc mở rộng quy mô hiệu quả.

Điểm mới trong v3.0:
- Lộ trình An toàn Tiên phong (công khai dưới dạng đã biên tập).
- Báo cáo Rủi ro (hàng quý, một số được đánh giá bởi bên ngoài).
- AI R&D được phân tách thành AI R&D-2 và AI R&D-4.
- Khi vượt qua AI R&D-4, bắt buộc phải có một hồ sơ an toàn khẳng định, xác định các rủi ro sai lệch từ các mô hình theo đuổi các mục tiêu sai lệch.

### Khung Chuẩn bị (PF) v2 của OpenAI (15 tháng 4 năm 2025)

Năm tiêu chí cho các năng lực được theo dõi:
- **Plausible (Hợp lý).** Tồn tại mô hình đe dọa hợp lý.
- **Measurable (Có thể đo lường).** Có thể thực hiện đánh giá thực nghiệm.
- **Severe (Nghiêm trọng).** Tác hại lớn.
- **Net-new (Mới hoàn toàn).** Không phải là rủi ro đã tồn tại từ trước được mở rộng quy mô.
- **Instantaneous-or-irremediable (Tức thời hoặc không thể khắc phục).** Tác hại xảy ra nhanh hoặc không thể đảo ngược.

Các năng lực đáp ứng cả năm tiêu chí trên sẽ được theo dõi. Các năng lực khác thì không.

Cấu trúc khác của PF v2:
- Tách biệt Báo cáo Năng lực (mô hình có thể làm gì) khỏi Báo cáo Biện pháp Bảo vệ (các kiểm soát hiện có).
- Nhóm Cố vấn An toàn đánh giá.
- Lãnh đạo phê duyệt; Ủy ban An toàn & Bảo mật của Hội đồng quản trị giám sát.
- "Điều khoản điều chỉnh": OpenAI có thể giảm các yêu cầu nếu một phòng thí nghiệm khác phát hành sản phẩm mà không có các biện pháp bảo vệ tương đương.

### Khung An toàn Tiên phong (FSF) v3.0 của DeepMind (Tháng 9 năm 2025)

Các Cấp độ Năng lực Quan trọng (CCLs) theo lĩnh vực:
- Nâng cao năng lực vũ khí sinh học (Bioweapon Uplift)
- Nâng cao năng lực tấn công mạng (Cyber Uplift)
- Tăng tốc nghiên cứu ML R&D
- Thao túng Có hại (mới trong v3.0): các mô hình có thể thay đổi đáng kể niềm tin/hành vi trong các bối cảnh có rủi ro cao.

v2.0 (tháng 2 năm 2025) đã thêm phần Căn chỉnh Lừa dối (Deceptive Alignment) và các cấp độ bảo mật cao hơn cho các CCL về ML R&D.

### Sự đồng bộ giữa các phòng thí nghiệm

- Anthropic: "Capability Thresholds" (Ngưỡng năng lực).
- DeepMind: "Critical Capability Levels" (Cấp độ năng lực quan trọng).
- OpenAI: "High Capability thresholds" (Ngưỡng năng lực cao).

Không có thuật ngữ tiêu chuẩn ngành. Tuy nhiên, chúng đồng bộ về cấu trúc: ba cấp độ năng lực tiên phong với các tiêu chí đánh giá được công bố. Cả ba đều bao gồm các điều khoản điều chỉnh theo đối thủ cạnh tranh (từ năm 2025 trở đi).

### Hồ sơ an toàn (Safety cases)

Một hồ sơ an toàn là một lập luận bằng văn bản cho thấy việc triển khai là an toàn ở mức chấp nhận được trong các giả định trường hợp xấu nhất. Cấu trúc tiêu chuẩn nhắm vào ba trụ cột:

- **Monitoring (Giám sát).** Chúng ta có thể phát hiện hành vi xấu nếu nó xảy ra không?
- **Illegibility (Tính không rõ ràng).** Mô hình có thiếu khả năng thực hiện một kế hoạch mạch lạc để gây hại không?
- **Incapability (Tính không có năng lực).** Mô hình có thiếu năng lực để gây ra tác hại đó không?

Các hồ sơ an toàn khác nhau nhắm vào các trụ cột khác nhau. Đối với hồ sơ CBRN ASL-3, tính không có năng lực (thông qua việc xóa bỏ kiến thức - unlearning) là mục tiêu chính. Đối với căn chỉnh lừa dối, giám sát và tính không rõ ràng là các mục tiêu. Đối với nâng cao năng lực tấn công mạng, cả ba trụ cột đều liên quan.

### Vấn đề động lực chạy đua

Các điều khoản điều chỉnh theo đối thủ cạnh tranh gây nhiều tranh cãi. Những người chỉ trích cho rằng chúng tạo ra một cuộc đua xuống đáy: nếu cả ba phòng thí nghiệm đều giảm yêu cầu khi đối thủ vi phạm, trạng thái cân bằng sẽ chuyển sang hướng vi phạm. Những người ủng hộ lập luận rằng giải pháp thay thế (các biện pháp bảo vệ đơn phương) sẽ tạo ra kết quả tồi tệ hơn nếu phòng thí nghiệm vi phạm ít chú trọng đến an toàn hơn.

UK AISI, US CAISI và Văn phòng AI của EU (Bài học 24) là các đối tác quản trị bên ngoài. Các khung của phòng thí nghiệm là tự nguyện; các khung pháp lý đang dần hình thành.

### Vị trí trong Giai đoạn 18

Các bài học 17-18 là lớp đo lường và quản trị nằm trên các phân tích về sự lừa dối và red-team. Các bài học 19-24 bao gồm phúc lợi, định kiến, quyền riêng tư, đóng dấu bản quyền (watermarking) và cấu trúc pháp lý. Bài học 28 lập bản đồ hệ sinh thái nghiên cứu (MATS, Redwood, Apollo, METR) giúp vận hành các đánh giá.

```figure
al-asl-ladder
```

## Sử dụng

Không có mã cho bài học này. Hãy đọc ba nguồn tài liệu chính: RSP v3.0, PF v2, FSF v3.0. Hãy lập bản đồ cấu trúc cấp độ của mỗi phòng thí nghiệm với các phòng thí nghiệm khác và xác định một ngưỡng mà mỗi phòng thí nghiệm định nghĩa nhưng các phòng thí nghiệm khác thì không.

## Triển khai

Bài học này tạo ra `outputs/skill-framework-diff.md`. Với một khung an toàn hoặc ghi chú phát hành, nó so sánh các định nghĩa ngưỡng, các đánh giá bắt buộc và cấu trúc hồ sơ an toàn của khung đó với RSP v3.0, PF v2, FSF v3.0 và gắn cờ các khoảng trống giữa các phòng thí nghiệm.

## Bài tập

1. Đọc RSP v3.0, PF v2 và FSF v3.0. Lập bảng về ngưỡng CBRN, ngưỡng AI R&D và đánh giá bắt buộc trước khi triển khai của mỗi phòng thí nghiệm.

2. Điều khoản điều chỉnh theo đối thủ cạnh tranh có trong cả ba khung (từ 2025+). Hãy viết một đoạn văn lập luận ủng hộ và một đoạn văn lập luận phản đối. Xác định giả định mà mỗi quan điểm dựa vào.

3. Thiết kế một hồ sơ an toàn cho một mô hình vượt qua ngưỡng AI R&D-4 của Anthropic. Nêu tên các bằng chứng mà mỗi trụ cột trong ba trụ cột (giám sát, tính không rõ ràng, tính không có năng lực) yêu cầu.

4. FSF v3.0 của DeepMind giới thiệu CCL về Thao túng Có hại. Đề xuất ba phép đo thực nghiệm cho thấy một mô hình đã vượt qua ngưỡng này.

5. Đọc tài liệu "Common Elements of Frontier AI Safety Policies" (2025) của METR. Nêu tên ba sự hội tụ mạnh nhất giữa các phòng thí nghiệm và hai sự khác biệt lớn nhất.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| RSP | "Khung của Anthropic" | Chính sách Quy mô Trách nhiệm; các cấp độ ASL; v3.0 tháng 2 năm 2026 |
| PF | "Khung của OpenAI" | Khung Chuẩn bị; năm tiêu chí; v2 tháng 4 năm 2025 |
| FSF | "Khung của DeepMind" | Khung An toàn Tiên phong; các CCL; v3.0 tháng 9 năm 2025 |
| ASL-3 | "Tương đương cấp độ an toàn sinh học 3" | Cấp độ của Anthropic cho các năng lực liên quan đến CBRN; kích hoạt tháng 5 năm 2025 |
| CCL | "Cấp độ năng lực quan trọng" | Cấu trúc ngưỡng của DeepMind; theo từng lĩnh vực |
| Safety case | "Lập luận chính thức" | Lập luận bằng văn bản rằng việc triển khai là an toàn ở mức chấp nhận được trong trường hợp xấu nhất |
| Adjustment clause | "Cho phép vi phạm do đối thủ" | Điều khoản khung cho việc giảm yêu cầu nếu đối thủ phát hành mà không có biện pháp bảo vệ tương đương |

## Đọc thêm

- [Anthropic — Responsible Scaling Policy v3.0 (Tháng 2 năm 2026)](https://www.anthropic.com/responsible-scaling-policy) — các cấp độ ASL, lộ trình, phân tách AI R&D
- [OpenAI — Updating the Preparedness Framework (15 tháng 4 năm 2025)](https://openai.com/index/updating-our-preparedness-framework/) — năm tiêu chí, điều khoản điều chỉnh
- [DeepMind — Strengthening our Frontier Safety Framework (Tháng 9 năm 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0, Thao túng Có hại
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — so sánh giữa các phòng thí nghiệm