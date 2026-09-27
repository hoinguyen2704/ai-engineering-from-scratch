# METR Time Horizons và Đánh giá năng lực bên ngoài

> METR (trước đây là ARC Evals) đã trở thành một tổ chức 501(c)(3) độc lập kể từ tháng 12 năm 2023. Benchmark Time Horizon 1.1 (tháng 1 năm 2026) của họ sử dụng đường cong logistic để khớp xác suất thành công của tác vụ với log(thời gian hoàn thành của chuyên gia); điểm giao cắt tại xác suất 50% xác định "chân trời thời gian" (time horizon) của mô hình. Bộ đánh giá giai đoạn 2025–2026 bao gồm GPT-5.1, GPT-5.1-Codex-Max và các đánh giá giám sát nguyên mẫu (liệu trình giám sát có thể phát hiện các tác vụ phụ; liệu tác nhân có thể né tránh hay không). Các bộ benchmark bao gồm: HCAST (hơn 180 tác vụ ML, an ninh mạng, SWE, suy luận; từ 1 phút đến hơn 8 giờ), RE-Bench (71 tác vụ kỹ thuật-nghiên cứu ML với baseline từ chuyên gia), SWAA. Lưu ý thực tế: Các phép đo của METR là lý tưởng hóa — không có con người, không có hậu quả thực tế — và nhóm đã ghi lại khoảng cách hành vi giữa đánh giá và triển khai (Bài học 1). Chân trời thời gian là một cận trên, không phải là dự đoán triển khai.

**Type:** Learn
**Languages:** Python (stdlib, logistic-fit horizon estimator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 19 (RSP)
**Time:** ~60 phút

## Vấn đề

Các chính sách mở rộng quy mô (Bài học 19, 20) chỉ hữu ích khi các phép đo mà chúng tham chiếu có giá trị. "Ngưỡng AI R&D-4" và "Tính tự chủ tầm xa" được định nghĩa trong văn bản chính sách; chúng chỉ trở nên khả thi khi các đánh giá cụ thể tạo ra các con số cụ thể.

METR là tổ chức đánh giá bên ngoài giai đoạn 2024–2026, đơn vị đã xác định nhiều con số trong số đó. Họ đánh giá các mô hình tiên phong — thường là trước khi phát hành, dưới thỏa thuận bảo mật (NDA) với các phòng thí nghiệm — và công bố phương pháp luận sau đó. Benchmark Time Horizon 1.1 (tháng 1 năm 2026) là sản phẩm tiêu biểu của họ: một đại lượng vô hướng duy nhất nén năng lực thành một đơn vị mà con người có thể hiểu được ("mô hình này có thể thực hiện loại tác vụ mà một chuyên gia mất X giờ với độ tin cậy 50%").

Bài học này một phần nói về phương pháp luận (cách tính toán chân trời) và một phần về cách diễn giải (tại sao chân trời là cận trên, không phải dự đoán triển khai). Hai kỹ năng này cần đi đôi với nhau. Một nhóm hiểu cách khớp chân trời sẽ khó bị đánh lừa bởi các tuyên bố kém chất lượng từ nhà cung cấp hơn là một nhóm chỉ nhìn thấy con số "14 giờ" trên slide.

## Khái niệm

### Bối cảnh về METR

- Thành lập: Tháng 12 năm 2023 (trước đây là ARC Evals, tách ra thành tổ chức 501(c)(3) độc lập).
- Phạm vi: Đánh giá năng lực tự chủ của các mô hình tiên phong, thường là trước khi phát hành.
- Các phòng thí nghiệm đối tác: Anthropic, OpenAI (nhiều đợt hợp tác trong giai đoạn 2025–2026).
- Các kết quả đáng chú ý: Time Horizon 1.0 (tháng 3 năm 2025), Time Horizon 1.1 (tháng 1 năm 2026), các đánh giá giám sát nguyên mẫu.

### Khớp đường cong Time Horizon

Phương pháp luận (từ blog và các bài báo của METR):

1. Thu thập một bộ tác vụ trải dài từ quy mô phút đến quy mô giờ theo thời gian hoàn thành của chuyên gia. Các bộ hiện tại: HCAST (hơn 180 tác vụ), RE-Bench (71 tác vụ), SWAA.
2. Chạy mô hình trên từng tác vụ; ghi lại thành công hoặc thất bại.
3. Khớp đường cong logistic: P(thành công) là hàm số của log(thời gian hoàn thành của chuyên gia).
4. Chân trời là thời gian chuyên gia tại đó P(thành công) = 0.5.

Hình dạng khớp logistic là phù hợp vì năng lực thường có mối quan hệ tăng dần và tiến tới ngưỡng bão hòa với độ khó của tác vụ. Điểm 50% là một lựa chọn (có thể là 10%, 90%); METR báo cáo nhiều ngưỡng trong bài báo chi tiết nhưng ưu tiên 50% vì đây là điểm trực quan nhất.

### Các con số tháng 1 năm 2026

Theo Time Horizon 1.1:

- Claude Opus 4.6: ~14 giờ ở độ tin cậy 50%, tính đến Time Horizon 1.1 (tháng 1 năm 2026).
- Thời gian nhân đôi trên các tác vụ kiểu HCAST: ~4.3 tháng (130.8 ngày) dựa trên dữ liệu khớp sau năm 2023 được báo cáo bởi Time Horizon 1.1 (tháng 1 năm 2026); con số ~7 tháng là dữ liệu khớp toàn bộ giai đoạn 2019–2025 từ Time Horizon 1.0 và được báo cáo trong TH1.1 cùng với con số sau năm 2023.

Đó là những con số cụ thể mà phần còn lại của Giai đoạn 15 tham chiếu. Chúng được trích dẫn trong RSP v3.0 của Anthropic (Bài học 19) và FSF v3 của DeepMind (Bài học 20) như một trong những phép đo cung cấp dữ liệu cho việc đánh giá ngưỡng.

### Các bộ Benchmark

- **HCAST**: 189 tác vụ bao gồm ML, an ninh mạng, kỹ thuật phần mềm và suy luận tổng quát. Phạm vi: 1 phút đến hơn 8 giờ thời gian chuyên gia. Đây là bộ chính đằng sau con số chân trời.
- **RE-Bench**: 71 tác vụ kỹ thuật-nghiên cứu ML với baseline từ chuyên gia con người. Đo lường liệu mô hình có thể thực hiện loại công việc R&D ML dẫn đến ngưỡng AI R&D-4 hay không.
- **SWAA**: các tác vụ ngắn hơn; hữu ích để hiệu chuẩn phần thấp của đường cong chân trời.

### Đánh giá giám sát nguyên mẫu

Công việc của METR giai đoạn 2025–2026 bao gồm các nguyên mẫu đánh giá giám sát: liệu trình giám sát có thể phát hiện mô hình thực hiện một tác vụ phụ (thứ gì đó khác với những gì người vận hành yêu cầu); liệu tác nhân có thể né tránh trình giám sát hay không. Những điều này trực tiếp cung cấp dữ liệu cho danh mục "Làm suy yếu các biện pháp bảo vệ" trong OpenAI PF v2 và giám sát sự liên kết lừa dối trong DeepMind FSF v3 (Bài học 20).

### Tại sao chân trời là cận trên

- **Công cụ lý tưởng hóa**: các tác vụ benchmark cung cấp cho mô hình các công cụ sạch và dữ liệu được định dạng tốt. Việc triển khai thực tế lộn xộn hơn nhiều.
- **Không có hậu quả thực tế**: mô hình không bao giờ thực sự tính phí khách hàng, xóa dữ liệu thực hoặc liên hệ với người thật. Việc triển khai thực tế có những rủi ro không thể đảo ngược.
- **Gian lận ngữ cảnh đánh giá (Eval-context gaming)**: Bài học 1. Các mô hình hành xử khác nhau trong các bài kiểm tra. Báo cáo An toàn AI Quốc tế năm 2026 đã ghi lại điều này một cách thực nghiệm.
- **Không có sự biến thiên từ người dùng hợp pháp**: các prompt benchmark được cấu trúc chặt chẽ. Người dùng thực tế tạo ra các yêu cầu mơ hồ, phụ thuộc vào ngữ cảnh.

Chân trời là trần năng lực trong các điều kiện thuận lợi. Độ tin cậy khi triển khai là một con số khác, thấp hơn, và các nhóm phải tự đo lường phân phối của riêng mình để biết được con số đó.

### Trường hợp của đánh giá viên bên ngoài

Đánh giá bên ngoài quan trọng vì các phòng thí nghiệm nội bộ có động lực để tối ưu hóa các chỉ số mà họ báo cáo. Sự độc lập của METR — một tổ chức 501(c)(3) với phương pháp luận được công bố và các bài báo được bình duyệt — là biện pháp giảm thiểu rủi ro về mặt cấu trúc. Điều này chưa đủ (các phòng thí nghiệm vẫn kiểm soát những gì METR thấy), nhưng chắc chắn tốt hơn là không có đánh giá bên ngoài.

### Cách sử dụng các con số chân trời trong thực tế

- **Như một bộ lọc năng lực**: nếu chân trời của mô hình thấp hơn nhiều so với thời gian chuyên gia của một tác vụ được đề xuất, đừng triển khai nó một cách tự chủ (tệp kỹ năng của Bài học 1).
- **Như một chỉ báo xu hướng**: thời gian nhân đôi cho bạn biết thực tiễn hiện tại sẽ còn an toàn trong bao lâu ngay cả khi không có các biện pháp giảm thiểu mới.
- **Như một thông tin tiên nghiệm (prior)**: chân trời 14 giờ là điểm khởi đầu. Hãy điều chỉnh giảm xuống cho phân phối tác vụ, chất lượng công cụ và ngữ cảnh triển khai của bạn.

```figure
a5-horizon-fit
```

## Sử dụng

`code/main.py` triển khai việc khớp logistic giữa thành công của tác vụ và log(thời gian chuyên gia), dựa trên một tập kết quả tổng hợp. Nó báo cáo chân trời 50% (con số tiêu biểu của METR), chân trời 10% (thận trọng) và chân trời 90% (lạc quan). Đồng thời minh họa những gì thay đổi khi tỷ lệ thành công bị thổi phồng một cách nhân tạo bởi việc gian lận ngữ cảnh đánh giá.

## Triển khai

`outputs/skill-horizon-interpretation.md` xem xét tuyên bố về chân trời của một nhà cung cấp và tạo ra phân tích khoảng cách giữa tuyên bố benchmark và thực tế triển khai.

## Bài tập

1. Chạy `code/main.py`. Xác nhận chân trời 50% của kết quả khớp khớp với dữ liệu thực tế tổng hợp. Bây giờ hãy chia đôi lưới thời gian tác vụ; ước tính chân trời có thay đổi đáng kể không?

2. Đọc bài đăng trên blog Time Horizon 1.1 của METR. Xác định các tác vụ cụ thể mà độ tin cậy cao nhất và thấp nhất. Giải thích tại sao khoảng cách đó tồn tại.

3. Đọc các tài nguyên "Đo lường năng lực AI tự chủ" của METR. Liệt kê các danh mục tác vụ HCAST. Chọn một danh mục bạn sẽ ưu tiên hơn cho một tác vụ sản xuất và biện minh lý do tại sao.

4. Đưa việc gian lận ngữ cảnh đánh giá vào trình mô phỏng: chuyển đổi ~20% các tác vụ thất bại thành thành công. Báo cáo chân trời mới. Điều này xấp xỉ những gì tỷ lệ gian lận 20% gây ra cho con số quan sát được.

5. Thiết kế một đánh giá chân trời nội bộ trên danh sách tồn đọng lỗi (bug backlog) của riêng bạn hoặc một tập tác vụ đại diện. Mô tả việc thu thập dữ liệu, cách khớp và những gì đầu ra cho bạn biết. So sánh với các con số của METR.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|---|---|---|
| METR | "Đánh giá viên bên ngoài" | cựu ARC Evals; tổ chức 501(c)(3) độc lập từ tháng 12/2023 |
| Time Horizon | "Thước đo năng lực" | Thời gian tác vụ chuyên gia ở độ tin cậy 50%, từ khớp logistic |
| HCAST | "Bộ chính của METR" | Hơn 180 tác vụ trải dài từ 1 phút đến hơn 8 giờ |
| RE-Bench | "Kỹ thuật nghiên cứu" | 71 tác vụ kỹ thuật-nghiên cứu ML với baseline con người |
| SWAA | "Bộ tác vụ ngắn" | Hiệu chuẩn phần thấp của đường cong chân trời |
| Doubling time | "Tốc độ tăng trưởng" | Thời gian để chân trời 50% tăng gấp đôi; ~7 tháng theo HCAST |
| Eval-context gaming | "Mô hình hành xử khác" | Khoảng cách hành vi được ghi lại giữa các bài kiểm tra và triển khai |
| Upper bound | "Chân trời là trần" | Chân trời benchmark > độ tin cậy triển khai khi chịu tải |

## Đọc thêm

- [METR — Tài nguyên đo lường năng lực AI tự chủ](https://metr.org/measuring-autonomous-ai-capabilities/) — Thông số kỹ thuật HCAST, RE-Bench, SWAA.
- [METR — Đo lường khả năng hoàn thành các tác vụ dài của AI](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — bài báo gốc về chân trời.
- [METR — Time Horizon 1.1 (tháng 1 năm 2026)](https://metr.org/research/) — các con số và phương pháp luận hiện tại.
- [Epoch AI — Benchmark METR Time Horizons](https://epoch.ai/benchmarks/metr-time-horizons) — theo dõi trực tiếp.
- [Anthropic — Đo lường tính tự chủ của tác nhân trong thực tế](https://www.anthropic.com/research/measuring-agent-autonomy) — góc nhìn nội bộ về các phép đo của METR.