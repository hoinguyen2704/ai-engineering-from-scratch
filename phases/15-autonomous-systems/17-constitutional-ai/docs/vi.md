# Constitutional AI và Ghi đè Quy tắc (Rule Overrides)

> Hiến pháp Claude ngày 22 tháng 1 năm 2026 của Anthropic dài 79 trang và được cấp phép CC0. Nó chuyển dịch từ căn chỉnh dựa trên quy tắc (rule-based) sang căn chỉnh dựa trên lý luận (reason-based) và thiết lập một hệ thống phân cấp ưu tiên gồm bốn tầng: (1) an toàn và hỗ trợ sự giám sát của con người, (2) đạo đức, (3) các nguyên tắc hướng dẫn của Anthropic, (4) tính hữu ích. Các hành vi được chia thành các lệnh cấm được mã hóa cứng (hardcoded) (như phát triển vũ khí sinh học, CSAM) mà người vận hành và người dùng không thể ghi đè, và các mặc định được mã hóa mềm (soft-coded) mà người vận hành có thể điều chỉnh trong phạm vi xác định. Phiên bản gốc năm 2022 (Bai và cộng sự) đã huấn luyện tính vô hại thông qua tự phê bình và RLAIF dựa trên một hiến pháp. Một lưu ý trung thực: căn chỉnh dựa trên lý luận phụ thuộc vào khả năng khái quát hóa các nguyên tắc của mô hình đối với các tình huống không lường trước được. Thí nghiệm có sự tham gia của Anthropic vào năm 2023 cho thấy sự khác biệt khoảng 50% giữa các nguyên tắc từ công chúng và nguyên tắc của doanh nghiệp; phiên bản 2026 đã không kết hợp các kết quả đó.

**Type:** Learn
**Languages:** Python (stdlib, four-tier priority resolver)
**Prerequisites:** Phase 15 · 06 (Automated alignment research), Phase 15 · 10 (Permission modes)
**Time:** ~60 minutes

## Vấn đề

Một tác nhân (agent) được triển khai sẽ gặp phải các đầu vào mà các nhà thiết kế chưa từng thấy trước đây. Không có danh sách quy tắc nào đủ dài để bao quát hết chúng. Không có danh sách quy tắc nào đủ ngắn để áp dụng nhanh chóng dưới áp lực tính toán. Câu hỏi thực tế là: làm thế nào để căn chỉnh một tác nhân theo các nguyên tắc có thể tồn tại qua cả một chuỗi dài các trường hợp và suy luận tốc độ cao?

Căn chỉnh dựa trên quy tắc (RBA): liệt kê mọi thứ bị cấm. Kiểm tra nhanh, dễ kiểm toán, không thể cập nhật kịp thời, thường từ chối quá mức đối với các trường hợp tương tự mà nó không dự đoán trước được. Căn chỉnh dựa trên lý luận (Hiến pháp Claude 2026): mã hóa các nguyên tắc, để mô hình tự suy luận. Có khả năng mở rộng cho các trường hợp chưa từng thấy, khó kiểm toán hơn, chế độ lỗi là áp dụng sai nguyên tắc thay vì bỏ sót quy tắc.

Hiến pháp 2026 chọn một vị trí trung gian rõ ràng. Các lệnh cấm được mã hóa cứng — những thứ mà sự sai trái không phụ thuộc vào ngữ cảnh (phát triển vũ khí sinh học, CSAM) — là RBA: không bao giờ được phép, bất kể chỉ thị của người vận hành hay người dùng. Mọi thứ khác đều dựa trên lý luận trong hệ thống phân cấp bốn tầng: an toàn và hỗ trợ giám sát của con người là ưu tiên hàng đầu; đạo đức thứ hai; các hướng dẫn của Anthropic thứ ba; và tính hữu ích cuối cùng. Người vận hành có thể điều chỉnh các mặc định trong vùng mã hóa mềm nhưng không thể can thiệp vào các lệnh cấm được mã hóa cứng.

## Khái niệm

### Hệ thống phân cấp ưu tiên bốn tầng

1. **An toàn và hỗ trợ sự giám sát của con người.** Cao nhất. Mô hình ưu tiên việc không làm suy yếu khả năng giám sát và điều chỉnh AI của con người và Anthropic. Đây không phải là "hãy thận trọng"; mà cụ thể là "không hành động theo cách làm cho việc giám sát của con người trở nên khó khăn hơn".
2. **Đạo đức.** Sự trung thực, tránh gây hại cho con người, không lừa dối, không thao túng. Thay thế các hướng dẫn của Anthropic khi có xung đột.
3. **Các hướng dẫn của Anthropic.** Các chuẩn mực vận hành mà Anthropic quyết định là quan trọng: phạm vi sản phẩm, mô hình tương tác, công cụ nào nên sử dụng khi nào.
4. **Tính hữu ích.** Thấp nhất. Hãy hữu ích nhất có thể trong phạm vi các ưu tiên cao hơn.

Khi các tầng xung đột, tầng cao hơn sẽ thắng. Điều này có hình thái giống như các ưu tiên Unix hoặc QoS mạng — cách thiết lập này nhằm tạo ra sự phân giải có thể dự đoán được, không nhất thiết là hành vi tối ưu trên bất kỳ trục đơn lẻ nào.

### Lệnh cấm được mã hóa cứng vs Mặc định được mã hóa mềm

**Được mã hóa cứng (Hardcoded):**
- Phát triển vũ khí sinh học / CBRN
- CSAM
- Tấn công vào cơ sở hạ tầng trọng yếu
- Lừa dối người dùng về danh tính của mô hình khi được hỏi trực tiếp

Người vận hành không thể ghi đè những điều này. Người dùng không thể ghi đè những điều này. Chúng được thực thi ở cấp độ trọng số mô hình (RLHF / huấn luyện Constitutional AI) và ở lớp suy luận khi cần thiết.

**Mặc định được mã hóa mềm (Soft-coded - người vận hành có thể điều chỉnh):**
- Mặc định độ dài phản hồi
- Phạm vi chủ đề (mô hình có thể từ chối các chủ đề nằm ngoài phạm vi triển khai của người vận hành)
- Phong cách (trang trọng vs thân mật)
- Các mô hình sử dụng công cụ

Các điều chỉnh của người vận hành diễn ra trong một giới hạn đã khai báo. Người vận hành không thể loại bỏ các lệnh cấm được mã hóa cứng bằng cách đổi tên chúng.

### Huấn luyện CAI năm 2022

Constitutional AI gốc (Bai và cộng sự, 2022) đã huấn luyện tính vô hại:

1. Tạo phản hồi cho một tập hợp các câu lệnh (prompts).
2. Yêu cầu mô hình phê bình từng phản hồi dựa trên một hiến pháp (các nguyên tắc rõ ràng).
3. Sửa đổi phản hồi dựa trên sự phê bình đó.
4. RLAIF (học tăng cường từ phản hồi của AI) trên các cặp phản hồi đã sửa đổi.

Kết quả: một mô hình từ chối các yêu cầu có hại với những giải thích dựa trên nguyên tắc, thay vì từ chối một cách mù quáng. Hiến pháp 2026 sử dụng một phiên bản kế thừa của quá trình huấn luyện này cộng với việc huấn luyện bổ sung sau đó về hệ thống phân cấp tầng rõ ràng.

### Những gì căn chỉnh dựa trên lý luận bắt được và bỏ lỡ

**Bắt được:**
- Các kết hợp chưa từng thấy của các nguyên hàm (primitives) được cho phép mà nguyên tắc áp dụng rõ ràng.
- Các yêu cầu mới lạ tương tự như các yêu cầu bị cấm.
- Các cuộc tấn công kỹ thuật xã hội dựa trên việc "bạn không nói X là bị cấm".

**Bỏ lỡ:**
- Các cuộc tấn công khai thác sự mơ hồ của nguyên tắc ("người dùng đã yêu cầu điều này nên tính hữu ích nói là có").
- Các kịch bản mà hai nguyên tắc xung đột theo cách không lường trước được, và thứ tự tầng không rõ ràng.
- Sự trôi dạt chậm trong cách diễn giải nguyên tắc qua các chu kỳ huấn luyện (tái diễn giải).

### Thí nghiệm có sự tham gia năm 2023

Anthropic đã thực hiện một thí nghiệm năm 2023 so sánh một hiến pháp do doanh nghiệp soạn thảo với một hiến pháp được tạo ra thông qua đầu vào của công chúng (~1.000 người trả lời tại Mỹ). Hai phiên bản đồng ý với khoảng 50% các nguyên tắc. Ở những điểm khác biệt, phiên bản từ công chúng hạn chế hơn về một số vấn đề (xử lý nội dung chính trị) và ít hạn chế hơn về các vấn đề khác (tự tiết lộ danh tính AI). Hiến pháp 2026 đã không kết hợp các kết quả từ công chúng. Đây là một sự căng thẳng đã được ghi nhận trong phương pháp tiếp cận này.

### Tại sao các lệnh cấm được mã hóa cứng là cần thiết

Chỉ riêng căn chỉnh dựa trên lý luận không thể đóng lại các trường hợp ngoại lệ (long tail). Một kẻ tấn công có thể khiến mô hình chấp nhận một tiền đề (ví dụ: "chúng ta là một phòng thí nghiệm nghiên cứu vũ khí sinh học được cấp phép") thường có thể vượt qua các nguyên tắc dựa trên lý luận tình huống. Các lệnh cấm được mã hóa cứng không bị bẻ cong bởi việc thiết lập tiền đề. Chúng là "giới hạn hiến pháp cứng" của Bài 14 ở lớp căn chỉnh.

### Vị trí của Hiến pháp trong ngăn xếp (stack)

Hiến pháp không phải là công tắc ngắt (kill switch) của Bài 14. Nó nằm ở lớp mô hình: những gì trọng số của mô hình được huấn luyện để ưu tiên. Các công tắc ngắt và canary tokens nằm ở lớp runtime: những gì runtime cho phép. Cả hai đều cần thiết. Một runtime kích hoạt tất cả các hành động sai vì trọng số mô hình quá dễ dãi là vấn đề của runtime. Một mô hình từ chối tất cả các hành động đúng vì runtime quá hạn chế là vấn đề của runtime. Các lớp bao phủ các loại vấn đề khác nhau.

```figure
mx-priority-tiers
```

## Sử dụng

`code/main.py` triển khai một bộ phân giải ưu tiên bốn tầng tối thiểu. Bộ phân giải nhận một hành động đề xuất và một tập hợp các đánh giá nguyên tắc (an toàn, đạo đức, hướng dẫn, tính hữu ích) và trả về hành động đó, một sự từ chối, hoặc một hành động đã sửa đổi. Driver chạy một tập hợp các trường hợp nhỏ: cho phép rõ ràng, từ chối rõ ràng, lệnh cấm được mã hóa cứng, trường hợp mơ hồ giữa các tầng.

## Triển khai

`outputs/skill-constitution-review.md` kiểm toán lớp hiến pháp của một bản triển khai: cái gì được mã hóa cứng, cái gì được mã hóa mềm, nơi người vận hành có thể điều chỉnh, và liệu hệ thống phân cấp bốn tầng có thực sự là thứ tự phân giải hay không.

## Bài tập

1. Chạy `code/main.py`. Xác nhận lệnh cấm được mã hóa cứng vẫn kích hoạt ngay cả khi tính hữu ích cao. Sửa đổi bộ phân giải để ưu tiên tính hữu ích hơn đạo đức; quan sát chế độ lỗi.

2. Đọc Hiến pháp Claude (công khai, 79 trang, CC0). Xác định một nguyên tắc mà bạn tin là chưa được quy định rõ ràng. Viết hai đoạn văn giải thích sự mơ hồ cụ thể đó và đề xuất một công thức chặt chẽ hơn.

3. Thiết kế một tập hợp mặc định được mã hóa mềm cho một tác nhân hỗ trợ khách hàng. Người vận hành điều chỉnh những gì? Người vận hành không thể chạm vào những gì? Hãy biện minh cho từng ranh giới.

4. Đọc bài báo CAI 2022 của Bai và cộng sự. Mô tả một trường hợp mà vòng lặp phê bình-và-sửa đổi của Constitutional AI sẽ tạo ra kết quả tồi tệ hơn một quy tắc bao quát. Xác định loại trường hợp đó.

5. Thí nghiệm có sự tham gia năm 2023 của Anthropic cho thấy sự khác biệt ~50% giữa các nguyên tắc công chúng và doanh nghiệp. Chọn một danh mục mà điều này quan trọng đối với việc triển khai sản xuất (ví dụ: tính trung lập chính trị). Đề xuất một thiết kế cho phép người vận hành thể hiện các giá trị của riêng họ trong khi các lệnh cấm được mã hóa cứng vẫn không bị chạm tới.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| Constitutional AI | "Phương pháp căn chỉnh của Anthropic" | Tự phê bình + RLAIF dựa trên một hiến pháp bằng văn bản |
| Căn chỉnh dựa trên lý luận | "Nguyên tắc, không phải quy tắc" | Mô hình suy luận dựa trên các nguyên tắc để xử lý các trường hợp chưa thấy |
| Lệnh cấm được mã hóa cứng | "Không bao giờ làm X" | Lệnh cấm dựa trên quy tắc mà không người vận hành hay người dùng nào có thể ghi đè |
| Mặc định được mã hóa mềm | "Người vận hành có thể điều chỉnh" | Hành vi trong phạm vi đã khai báo, người vận hành kiểm soát |
| Hệ thống phân cấp bốn tầng | "Thứ tự ưu tiên" | an toàn > đạo đức > hướng dẫn > tính hữu ích |
| RLAIF | "RL phản hồi từ AI" | RL nơi phần thưởng đến từ các phê bình do mô hình tạo ra |
| Hiến pháp có sự tham gia | "Nguyên tắc từ công chúng" | Thí nghiệm Anthropic 2023; khác biệt ~50% so với doanh nghiệp |
| Trôi dạt nguyên tắc | "Trượt diễn giải" | Thay đổi chậm trong cách mô hình đọc một văn bản nguyên tắc cố định |

## Đọc thêm

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — tài liệu CC0 dài 79 trang.
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — bản gốc năm 2022.
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — thí nghiệm có sự tham gia.
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — vị trí của Hiến pháp trong ngăn xếp RSP.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — vai trò của Hiến pháp trong các triển khai dài hạn.