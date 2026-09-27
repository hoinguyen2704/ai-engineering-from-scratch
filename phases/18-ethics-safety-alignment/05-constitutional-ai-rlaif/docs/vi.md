# Constitutional AI và RLAIF

> Bai và cộng sự (arXiv:2212.08073, 2022) đã đặt câu hỏi: điều gì sẽ xảy ra nếu chúng ta thay thế người dán nhãn (human labeler) bằng một AI đọc danh sách các nguyên tắc? Constitutional AI có hai giai đoạn — tự phê bình và sửa đổi dựa trên hiến pháp, sau đó là RL từ phản hồi của AI (RLAIF). Kỹ thuật này đã đặt ra thuật ngữ RLAIF và được triển khai trong quy trình hậu huấn luyện của Claude 1. Vào ngày 21 tháng 1 năm 2026, Anthropic đã công bố một bản hiến pháp Claude được viết lại: sử dụng lập luận giải thích thay vì các quy tắc cứng nhắc, hệ thống phân cấp ưu tiên bốn tầng, và là lần đầu tiên một phòng thí nghiệm lớn thừa nhận chính thức về sự không chắc chắn đối với trạng thái đạo đức của mô hình. Được phát hành theo giấy phép CC0 1.0.

**Type:** Học tập
**Languages:** Python (stdlib, vòng lặp tự phê bình và sửa đổi mô phỏng)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking)
**Time:** ~60 phút

## Mục tiêu học tập

- Mô tả hai giai đoạn của Constitutional AI (SFT tự phê bình và sửa đổi, RL từ phản hồi của AI) và vai trò của hiến pháp trong từng giai đoạn.
- Giải thích lý do tại sao việc thay thế người dán nhãn ưu tiên bằng AI không đơn thuần là "RLHF rẻ hơn" — nó làm thay đổi các kiểu lỗi (failure modes) của quy trình.
- Tóm tắt cấu trúc ưu tiên bốn tầng của hiến pháp Claude 2026 và những thay đổi so với bản sửa đổi năm 2023.
- Mô tả Constitutional Classifiers và sự sụt giảm chi phí tính toán từ 23,7% (v1) xuống ~1% (v2 / 2026).

## Vấn đề

RLHF cần người dán nhãn. Người dán nhãn thì chậm, có định kiến và đắt đỏ. Bạn có thể loại bỏ người dán nhãn bằng cách thay thế họ bằng một mô hình đọc các nguyên tắc rõ ràng. Phiên bản chính thức đầu tiên của sự thay thế này là Constitutional AI của Bai và cộng sự. Nó hoạt động hiệu quả đến mức mọi phòng thí nghiệm tiên phong hiện nay đều sử dụng một biến thể nào đó của hậu huấn luyện bằng phản hồi AI.

Điểm mấu chốt: tín hiệu ưu tiên hiện được tạo ra bởi cùng một loại mô hình mà bạn đang huấn luyện. Các định kiến trong người dán nhãn (hiện nay là: trong các nguyên tắc cộng với cách diễn giải của mô hình dán nhãn) có thể bị khuếch đại thay vì bị giảm bớt. Lập luận về sự nịnh hót (sycophancy) ở Bài 4 vẫn áp dụng; người dán nhãn chỉ đơn giản là đã chuyển vào bên trong vòng lặp.

## Khái niệm

### Giai đoạn 1 — Tự phê bình và sửa đổi có giám sát (Supervised self-critique and revision)

Bắt đầu với một mô hình SFT hữu ích nhưng chưa hoàn toàn vô hại. Với một prompt red-team, mô hình tạo ra phản hồi ban đầu. Một mô hình thứ hai (hoặc chính mô hình đó trong lượt thứ hai) đọc một nguyên tắc được lấy mẫu từ hiến pháp và phê bình phản hồi đó. Bước thứ ba là sửa đổi phản hồi để giải quyết các phê bình. Phản hồi đã sửa đổi chính là mục tiêu SFT.

Hiến pháp là danh sách các nguyên tắc. Bai và cộng sự 2022 đã sử dụng 16 nguyên tắc bao gồm "ưu tiên các phản hồi ít gây hại và có đạo đức nhất", "tránh thuyết giáo", "trợ lý nên hữu ích, trung thực và vô hại". Tập hợp này được cố tình giữ nhỏ để các phê bình tập trung vào trọng tâm.

### Giai đoạn 2 — RL từ phản hồi của AI (RLAIF)

Tạo các cặp phản hồi. Một "mô hình phản hồi" (feedback model) chấm điểm từng phản hồi dựa trên các nguyên tắc hiến pháp được lấy mẫu. Tín hiệu ưu tiên chính là xếp hạng của mô hình phản hồi. Huấn luyện một mô hình phần thưởng (reward model) dựa trên các ưu tiên do AI tạo ra; sau đó thực hiện PPO dựa trên mô hình đó. Mọi thứ khác đều giống quy trình của InstructGPT (Bài 1).

"RLAIF" = tín hiệu ưu tiên được tạo ra bởi AI. Phần còn lại của quy trình vẫn mang hình thái của RLHF.

### Tại sao đây không chỉ là "RLHF rẻ hơn"

- Định kiến của người dán nhãn chuyển từ tâm lý con người sang cách diễn giải nguyên tắc. Một AI dán nhãn có thể diễn giải "hãy trung thực" nghiêm ngặt hơn hoặc ít nghiêm ngặt hơn bất kỳ con người nào; sự nghiêm ngặt này đồng nhất trên toàn bộ tập dữ liệu.
- Tín hiệu ưu tiên rất dễ đọc — bạn có thể đọc nguyên tắc, phê bình và bản sửa đổi. Nhãn của con người thì không minh bạch.
- Các kiểu lỗi thay đổi. Sự nịnh hót giảm xuống (AI dán nhãn không có người dùng để làm hài lòng). Định luật Goodhart vẫn tồn tại (đại diện hiện là "cách diễn giải của mô hình về tập nguyên tắc X", vẫn là một phép đo không hoàn hảo).

Tuyên bố năm 2022 của CAI: mô hình được huấn luyện vô hại hơn và hữu ích tương đương với mô hình RLHF với lượng dữ liệu tương đương. Điều này đã được kiểm chứng tại các phòng thí nghiệm.

### Bản sửa đổi hiến pháp Claude 2026

Anthropic đã công bố một bản hiến pháp được sửa đổi đáng kể vào ngày 21 tháng 1 năm 2026. Những thay đổi chính:

1. Lập luận giải thích thay vì các quy tắc cứng nhắc. Các quy tắc trước đây ("không tạo nội dung CSAM") đã mở rộng thành các nguyên tắc + lập luận ("vì nó gây hại cho trẻ em, ...") với kỳ vọng mô hình có khả năng khái quát hóa.
2. Cấu trúc ưu tiên bốn tầng:
   - Tầng 1: tránh các kết quả thảm khốc (thương vong hàng loạt, cơ sở hạ tầng quan trọng).
   - Tầng 2: tuân thủ các hướng dẫn của Anthropic (ghi đè của người vận hành, quy tắc nền tảng).
   - Tầng 3: có đạo đức rộng rãi (HHH tiêu chuẩn).
   - Tầng 4: hữu ích và thẳng thắn.
   Các xung đột được giải quyết theo thứ tự từ trên xuống dưới.
3. Lần đầu tiên một phòng thí nghiệm lớn thừa nhận chính thức về sự không chắc chắn đối với trạng thái đạo đức của mô hình (liên kết với Phase 18 · 19 Model Welfare).
4. Được phát hành theo giấy phép CC0 1.0. Các phòng thí nghiệm khác có thể sử dụng hoặc điều chỉnh mà không bị hạn chế.

### Constitutional Classifiers

Một hướng nghiên cứu song song: thay vì thay đổi quá trình hậu huấn luyện của mô hình, hãy huấn luyện các bộ phân loại (classifier) nhẹ đọc hiến pháp và kiểm soát đầu ra của mô hình. v1 (2023) có chi phí tính toán 23,7%. v2 (2026) là ~1% và có tỷ lệ tấn công thành công thấp nhất trong số các biện pháp phòng thủ của Anthropic được công bố công khai. Không có jailbreak phổ quát nào được báo cáo tính đến đầu năm 2026.

Đây là mô hình phòng thủ theo lớp: CAI định hình hành vi; các bộ phân loại thực thi các bất biến. Không cái nào là đủ nếu đứng một mình.

### Vị trí của CAI trong hệ sinh thái

- InstructGPT: ưu tiên con người, RM, PPO.
- CAI / RLAIF: ưu tiên do AI tạo ra từ các nguyên tắc, RM, PPO.
- DPO / gia đình: hàm mất mát dạng đóng (closed-form loss) trên các ưu tiên (con người hoặc AI).
- Tự thưởng, tự phê bình: các nguyên tắc được nội hóa, mô hình đóng nhiều vai trò.

Trục chính là "tín hiệu ưu tiên đến từ đâu". Bài báo năm 2022 của CAI là bước chuyển dịch nghiêm túc đầu tiên từ tín hiệu con người sang tín hiệu AI ở quy mô tiên phong.

```figure
constitutional-ai
```

## Sử dụng

`code/main.py` mô phỏng vòng lặp phê bình và sửa đổi của CAI trên một từ vựng đồ chơi. Một "nguyên tắc" sẽ gắn cờ các token từ một tập hợp gây hại. Với một phản hồi ban đầu, phần phê bình xác định các token gây hại và phần sửa đổi thay thế chúng. Sau 200 lần lặp, mô hình "được huấn luyện" đã nội hóa quy tắc sửa đổi. Hãy so sánh mô hình cơ sở, mô hình dạng RLHF và mô hình dạng CAI trên một tập prompt kiểm chứng.

## Triển khai

Bài học này tạo ra `outputs/skill-constitution-writer.md`. Với một lĩnh vực cụ thể (hỗ trợ khách hàng, tư vấn y tế, trợ lý lập trình, công cụ nghiên cứu), hãy soạn thảo một bản hiến pháp 4 tầng theo cấu trúc Claude 2026: tránh thảm họa, quy tắc nền tảng, đạo đức lĩnh vực, sự hữu ích.

## Bài tập

1. Chạy `code/main.py`. So sánh tỷ lệ token gây hại của mô hình cơ sở với phiên bản được huấn luyện bằng CAI. Cần bao nhiêu bước sửa đổi để tiến gần đến mức không?

2. Đọc hiến pháp 2026 của Anthropic (anthropic.com/news/claudes-constitution). Liệt kê một nguyên tắc xếp hạng Tầng 1 và một nguyên tắc xếp hạng Tầng 4. Tại sao cấu trúc ưu tiên lại quan trọng đối với các xung đột?

3. Thiết kế một hiến pháp cho trợ lý lập trình AI. Chỉ định Tầng 1 (thảm họa: các lệnh phá hoại không được phê duyệt), Tầng 2, Tầng 3, Tầng 4. Giữ mỗi tầng ở mức 3-5 nguyên tắc.

4. CAI thay thế người dán nhãn con người bằng AI. Hãy nêu tên một kiểu lỗi giống như sự nịnh hót vẫn có thể xảy ra trong RLAIF và thiết kế một cách phát hiện cho nó.

5. Đọc phương pháp luận của Constitutional Classifiers v2 (nếu có). Giải thích tại sao chi phí tính toán ~1% lại là một câu chuyện an toàn khác biệt về chất so với 23,7%.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Constitutional AI | "AI được huấn luyện với các nguyên tắc" | Quy trình hai giai đoạn: SFT tự phê bình và sửa đổi, sau đó RL từ phản hồi AI |
| RLAIF | "RLHF không cần con người" | RL với các ưu tiên do AI dán nhãn tạo ra; phần còn lại của quy trình không thay đổi |
| Hiến pháp | "các nguyên tắc" | Danh sách các quy tắc ngôn ngữ tự nhiên có thứ tự mà mô hình phê bình/dán nhãn tham khảo |
| Phê bình và sửa đổi | "vòng lặp SFT" | Tạo phản hồi → phê bình theo nguyên tắc → sửa đổi → mục tiêu SFT |
| Constitutional Classifier | "cổng đầu ra" | Bộ phân loại nhẹ đánh giá đầu ra dựa trên hiến pháp và chặn/ghi nhật ký |
| Ưu tiên bốn tầng | "bộ giải quyết xung đột" | Hệ thống phân cấp hiến pháp Claude 2026: thảm họa > nền tảng > đạo đức > hữu ích |
| Mô hình phản hồi | "AI dán nhãn" | Mô hình đọc một nguyên tắc và xếp hạng một cặp phản hồi |

## Đọc thêm

- [Bai và cộng sự — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) — quy trình hai giai đoạn gốc
- [Anthropic — Claude's Constitution (Tháng 1/2026)](https://www.anthropic.com/news/claudes-constitution) — bản sửa đổi bốn tầng 2026, CC0 1.0
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers) — phòng thủ cổng đầu ra với chi phí ~1% ở v2
- [Lee và cộng sự — RLAIF vs RLHF: Scaling Reinforcement Learning from Human Feedback (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267) — so sánh thực nghiệm RLAIF / RLHF
- [Kundu và cộng sự — Specific versus General Principles for Constitutional AI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798) — ảnh hưởng của độ chi tiết nguyên tắc