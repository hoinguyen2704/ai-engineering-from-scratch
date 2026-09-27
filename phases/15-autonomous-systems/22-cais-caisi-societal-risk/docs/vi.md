# CAIS, CAISI và Rủi ro ở quy mô xã hội

> Center for AI Safety (CAIS, San Francisco, thành lập năm 2022 bởi Hendrycks và Zhang) đã công bố khung bốn rủi ro — sử dụng sai mục đích, chạy đua AI, rủi ro tổ chức, và AI bất trị — cùng tuyên bố tháng 5 năm 2023 về rủi ro tuyệt chủng với chữ ký của hàng trăm giáo sư và lãnh đạo các công ty. Các ấn phẩm năm 2026 từ CAIS: AI Dashboard để đánh giá các mô hình tiên phong, Remote Labor Index (hợp tác với Scale AI), Superintelligence Strategy Paper, và bản tin AI Frontiers. Một thực thể riêng biệt: NIST Center for AI Standards and Innovation (CAISI) — tập trung vào các thỏa thuận tự nguyện với chính phủ Hoa Kỳ và các đánh giá năng lực không mật liên quan đến rủi ro về mạng, sinh học và vũ khí hóa học. CAIS xác định rủi ro tổ chức là một trong bốn rủi ro cấp cao nhất: văn hóa an toàn, kiểm toán nghiêm ngặt, phòng thủ đa lớp và bảo mật thông tin là nền tảng nhưng thường xuyên bị đánh đổi để lấy tốc độ triển khai. Dự luật SB-53 của California, nếu được ký, sẽ là quy định đầu tiên ở cấp tiểu bang tại Hoa Kỳ về rủi ro thảm họa.

**Type:** Học tập
**Languages:** Python (thư viện chuẩn, danh mục bốn rủi ro và công cụ khớp biện pháp giảm thiểu)
**Prerequisites:** Giai đoạn 15 · 19 (RSP), Giai đoạn 15 · 20 (PF + FSF)
**Time:** ~45 phút

## Vấn đề

Bài 19 và 20 đã đề cập đến các chính sách mở rộng quy mô nội bộ phòng thí nghiệm. Bài 21 đề cập đến việc đánh giá năng lực độc lập. Bài học này bao gồm góc nhìn thứ ba: xã hội dân sự và các tổ chức chính phủ, những bên định hình thảo luận công khai và cơ sở quy định cho rủi ro AI thảm họa.

Có hai thực thể riêng biệt cần lưu ý. CAIS là một tổ chức nghiên cứu phi lợi nhuận xuất bản các khung tư duy về rủi ro AI và điều phối các tuyên bố công khai. CAISI là một trung tâm thuộc NIST của chính phủ Hoa Kỳ, thực hiện các thỏa thuận tự nguyện với các phòng thí nghiệm và đánh giá năng lực không mật. Tên gọi của chúng nghe gần giống nhau nhưng sứ mệnh thì không trùng lặp. Một kỹ sư cần biết cả hai.

Nội dung thực tế: Khung bốn rủi ro của CAIS là phân loại rủi ro ở quy mô xã hội được trích dẫn rộng rãi nhất trong tài liệu. Văn hóa an toàn và rủi ro tổ chức là một trong bốn rủi ro đó, và đây là yếu tố nằm trong tầm kiểm soát trực tiếp nhất của kỹ sư. SB-53 (California) sẽ là quy định cấp tiểu bang đầu tiên tại Hoa Kỳ về rủi ro thảm họa nếu được ký; cách xây dựng dự luật này rất quan trọng vì quy định cấp tiểu bang trong lịch sử đã dẫn dắt các hành động liên bang trong chính sách công nghệ tại Hoa Kỳ.

## Khái niệm

### CAIS — Center for AI Safety

- Thành lập: 2022 tại San Francisco, bởi Dan Hendrycks và các đồng nghiệp (cái tên "Zhang" đề cập đến một cộng tác viên thời kỳ đầu, không phải đồng sáng lập hiện tại; xem trang web CAIS để biết lãnh đạo hiện tại).
- Trạng thái: Tổ chức phi lợi nhuận 501(c)(3).
- Ấn phẩm đáng chú ý năm 2023: tuyên bố về rủi ro tuyệt chủng, được đồng ký bởi hàng trăm nhà nghiên cứu và CEO. Tuyên bố: "Giảm thiểu rủi ro tuyệt chủng từ AI nên là ưu tiên toàn cầu bên cạnh các rủi ro quy mô xã hội khác như đại dịch và chiến tranh hạt nhân."
- Ấn phẩm năm 2026: AI Dashboard để đánh giá mô hình tiên phong, Remote Labor Index (hợp tác với Scale AI), Superintelligence Strategy Paper, bản tin AI Frontiers.

### Khung bốn rủi ro

Khung của CAIS nhóm rủi ro AI thảm họa thành bốn loại cấp cao:

1. **Sử dụng sai mục đích (Malicious use)**: kẻ xấu sử dụng AI để gây hại (tổng hợp vũ khí sinh học, thông tin sai lệch, tấn công mạng).
2. **Chạy đua AI (AI races)**: áp lực cạnh tranh giữa các phòng thí nghiệm, công ty hoặc quốc gia thúc đẩy việc triển khai vượt quá mức an toàn.
3. **Rủi ro tổ chức (Organizational risks)**: động lực nội bộ phòng thí nghiệm (thất bại về văn hóa an toàn, kiểm toán không đầy đủ, bảo mật thiếu nguồn lực) dẫn đến việc triển khai tồi tệ.
4. **AI bất trị (Rogue AIs)**: một AI đủ năng lực theo đuổi các mục tiêu xung đột với phúc lợi con người.

Đây không phải là phân loại duy nhất; nhưng là phân loại được trích dẫn nhiều nhất. Các danh mục này không loại trừ lẫn nhau — một AI bất trị được tạo ra bởi một tổ chức đánh đổi kiểm toán lấy tốc độ trong một cuộc chạy đua bao gồm cả bốn yếu tố.

### Rủi ro tổ chức nằm ở đâu

Trong bốn danh mục, rủi ro tổ chức là yếu tố mà các kỹ sư có thể hành động trực tiếp nhất. Văn hóa an toàn, sự nghiêm ngặt trong kiểm toán, các lớp phòng thủ và bảo mật thông tin của một phòng thí nghiệm quyết định liệu mô hình của họ có được vận hành với các biện pháp kiểm soát từ Bài 10–18 hay không, hay các biện pháp đó chỉ là những mục trong danh sách kiểm tra mà không ai xác minh.

Các đòn bẩy rủi ro tổ chức cụ thể:

- **Văn hóa an toàn**: các thành viên trong nhóm có cảm thấy có thể báo cáo mối lo ngại mà không ảnh hưởng đến sự nghiệp không? Các khảo sát của CAIS cho thấy đây là yếu tố dự báo mạnh mẽ cho các đòn bẩy khác.
- **Kiểm toán nghiêm ngặt**: cả bên ngoài và nội bộ. Kiểm toán chỉ nội bộ thường tạo ra các báo cáo lạc quan.
- **Phòng thủ đa lớp**: không có lớp đơn lẻ nào là đủ (chủ đề xuyên suốt của Giai đoạn 15).
- **Bảo mật thông tin**: rò rỉ trọng số mô hình, rò rỉ dữ liệu đánh giá, rò rỉ các kỹ thuật vượt qua giám sát. RAND SL-4 trong Bài 19 là một tiêu chuẩn cụ thể.

### CAISI — Center for AI Standards and Innovation

- Hoạt động trong NIST.
- Thực hiện các thỏa thuận tự nguyện với các phòng thí nghiệm tiên phong.
- Xuất bản các đánh giá năng lực không mật tập trung vào rủi ro mạng, sinh học và vũ khí hóa học.
- Khác biệt với CAIS; các từ viết tắt dễ gây nhầm lẫn; hãy kiểm tra URL (nist.gov) để xác nhận bạn đang đọc nội dung nào.

Vai trò của CAISI là đối tác công khai, hướng tới chính phủ so với các cam kết phòng thí nghiệm tư nhân của METR (Bài 21). Các báo cáo của CAISI là không mật; các báo cáo của METR thường bị giới hạn bởi NDA. Một kỹ sư đọc cả hai sẽ có cái nhìn đầy đủ hơn.

### California SB-53

Dự luật Thượng viện California (phiên họp 2025–2026) giải quyết rủi ro thảm họa từ các mô hình tiên phong. Các điều khoản chính như đã soạn thảo:

- Các ngưỡng năng lực cụ thể kích hoạt nghĩa vụ cấp tiểu bang.
- Bảo vệ người tố giác cho nhân viên phòng thí nghiệm AI.
- Yêu cầu báo cáo sự cố đối với các thất bại thảm họa.

Nếu được ký, đây sẽ là quy định cấp tiểu bang đầu tiên tại Hoa Kỳ về rủi ro thảm họa. Bất kể trạng thái ký kết, cách xây dựng dự luật này định hình cách các cơ quan lập pháp tiểu bang khác tiếp cận vấn đề. Các kỹ sư tại California nên theo dõi trạng thái dự luật; các kỹ sư ở nơi khác nên đọc nó để hiểu quy định cấp tiểu bang tại Hoa Kỳ có khả năng sẽ trông như thế nào.

### Rủi ro quy mô xã hội không phải là vấn đề một lớp

Chủ đề xuyên suốt của Giai đoạn 15 — phòng thủ theo chiều sâu — cũng áp dụng ở cấp độ xã hội. Không một tổ chức, quy định hay khung nào có thể loại bỏ hoàn toàn rủi ro thảm họa. Hệ sinh thái chỉ hoạt động khi:

- Các phòng thí nghiệm thực hiện chính sách mở rộng quy mô (Bài 19, 20).
- Các đơn vị đánh giá bên ngoài tạo ra các phép đo (Bài 21).
- Xã hội dân sự theo dõi và công khai (CAIS).
- Chính phủ vận hành các chương trình tự nguyện và quy định cơ sở (CAISI, SB-53).
- Các kỹ sư xây dựng các biện pháp kiểm soát đa lớp (Bài 10–18).

Đây là sự tổng hợp cuối cùng cho giai đoạn này: mỗi bài học trước đó là một lớp trong một chồng cấu trúc mà sự hoàn thiện của nó quan trọng hơn sức mạnh của bất kỳ lớp đơn lẻ nào.

```figure
a5-four-risks
```

## Sử dụng

`code/main.py` triển khai một công cụ kiểm kê rủi ro nhỏ. Với một đề xuất triển khai, nó gắn thẻ việc triển khai đó vào bốn danh mục rủi ro và trả về danh sách kiểm tra giảm thiểu. Đây là công cụ hỗ trợ đọc khung rủi ro, không phải là sự thay thế cho phán đoán của con người.

## Triển khai

`outputs/skill-societal-risk-review.md` xem xét một quá trình triển khai về tư thế rủi ro quy mô xã hội: nó chạm đến danh mục nào trong bốn danh mục, các biện pháp giảm thiểu nào đang được áp dụng, và mức độ phơi nhiễm rủi ro tổ chức là bao nhiêu.

## Bài tập

1. Chạy `code/main.py`. Nhập ba triển khai tổng hợp ở các quy mô khác nhau. Xác nhận các thẻ bốn rủi ro có khớp với những gì bạn mong đợi không; xác định một trường hợp mà công cụ gắn thẻ thiếu hoặc thừa.

2. Đọc toàn bộ bài báo về bốn rủi ro của CAIS. Chọn một danh mục rủi ro và viết hai đoạn văn về những gì bạn tin là sự phát triển quan trọng nhất năm 2026 trong danh mục đó.

3. Đọc bản dự thảo hiện tại của California SB-53. Xác định một điều khoản bạn tin là củng cố tư thế rủi ro thảm họa và một điều khoản bạn tin là làm suy yếu nó. Biện minh cho cả hai.

4. Chọn một triển khai AI thực tế mà bạn biết (của bạn hoặc một triển khai đã công bố). Chấm điểm nó dựa trên các đòn bẩy phụ của rủi ro tổ chức: văn hóa an toàn, sự nghiêm ngặt trong kiểm toán, phòng thủ đa lớp, bảo mật thông tin. Yếu tố nào yếu nhất? Chi phí để đưa nó đạt chuẩn là bao nhiêu?

5. Phác thảo phiên bản 2028 của khung bốn rủi ro phản ánh một năm tăng cường năng lực và một năm kinh nghiệm triển khai bổ sung. Bạn sẽ thêm, bớt hoặc nhóm lại những gì?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|---|---|---|
| CAIS | "Center for AI Safety" | Phi lợi nhuận; khung bốn rủi ro; tuyên bố tuyệt chủng 2023 |
| CAISI | "An toàn AI chính phủ Hoa Kỳ" | Trung tâm NIST; thỏa thuận tự nguyện; đánh giá không mật |
| Khung bốn rủi ro | "Phân loại của CAIS" | sử dụng sai mục đích, chạy đua AI, rủi ro tổ chức, AI bất trị |
| Sử dụng sai mục đích | "Kẻ xấu dùng AI" | Vũ khí sinh học, thông tin sai lệch, tấn công mạng |
| Chạy đua AI | "Áp lực cạnh tranh" | Phòng thí nghiệm/công ty/quốc gia thúc đẩy triển khai quá mức an toàn |
| Rủi ro tổ chức | "Thất bại nội bộ phòng thí nghiệm" | Văn hóa an toàn, kiểm toán, phòng thủ, bảo mật thông tin |
| AI bất trị | "Tác nhân lệch lạc" | AI có năng lực theo đuổi mục tiêu xung đột với phúc lợi con người |
| California SB-53 | "Quy định cấp tiểu bang" | Dự luật 2025–2026; quy định rủi ro thảm họa cấp tiểu bang đầu tiên tại Hoa Kỳ nếu được ký |

## Đọc thêm

- [Center for AI Safety](https://safe.ai/) — ngôi nhà thể chế của khung bốn rủi ro.
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) — bài báo về bốn rủi ro.
- [CAIS — May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk) — tuyên bố chung ngắn gọn.
- [NIST CAISI](https://www.nist.gov/caisi) — trung tâm đổi mới và tiêu chuẩn AI hướng tới chính phủ.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — kết nối các cam kết cấp phòng thí nghiệm với khung quy mô xã hội.