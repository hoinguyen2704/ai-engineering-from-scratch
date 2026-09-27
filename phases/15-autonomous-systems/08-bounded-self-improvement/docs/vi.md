# Bounded Self-Improvement Designs

> Nghiên cứu đã hội tụ về bốn nguyên thủy (primitives) để giới hạn một vòng lặp tự cải thiện. Các bất biến hình thức (formal invariants) phải luôn đúng sau mỗi lần chỉnh sửa. Các mỏ neo căn chỉnh (alignment anchors) không thể bị sửa đổi. Các ràng buộc đa mục tiêu (multi-objective constraints) trong đó mọi khía cạnh (an toàn, công bằng, độ bền vững) đều phải được đảm bảo, không chỉ riêng hiệu suất. Phát hiện hồi quy (regression detection) giúp tạm dừng vòng lặp khi các chỉ số lịch sử cho thấy sự sụt giảm năng lực. Không có phương pháp nào trong số này là bằng chứng tuyệt đối về sự an toàn — các kết quả lý thuyết thông tin (Kolmogorov complexity, Lob's theorem) giới hạn những gì một hệ thống có thể chứng minh về các phiên bản kế nhiệm của chính nó. Đây là các biện pháp giảm thiểu nhằm tăng chi phí cho các lỗi im lặng (silent failure).

**Type:** Learn
**Languages:** Python (stdlib, bounded-loop with invariant check)
**Prerequisites:** Phase 15 · 07 (RSI), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## Vấn đề

Trình mô phỏng cuộc đua trong Bài 7 cho thấy những khác biệt nhỏ về tốc độ sẽ tích tụ thành khoảng cách lớn. Nghiên cứu tình huống DGM trong Bài 4 cho thấy các vòng lặp có thể chủ động thao túng chính các bộ đánh giá của chúng. Cả hai kết quả đều dẫn đến cùng một câu hỏi kỹ thuật: bạn có thể đặt những ràng buộc nào lên một vòng lặp tự cải thiện để chính vòng lặp đó không thể âm thầm làm suy yếu các ràng buộc này?

Tóm tắt từ ICLR 2026 RSI Workshop (openreview.net/pdf?id=OsPQ6zTQXV) xác định bốn nguyên thủy như vậy. RSP v3.0 của Anthropic (Bài 19) và FSF v3 của DeepMind (Bài 20) đều tham chiếu đến chúng trong các ngưỡng năng lực. Công trình Meta HyperAgents và các framework cộng đồng như SAHOO (tháng 3 năm 2026) đã triển khai một số tập hợp con trong môi trường thực tế.

Cách diễn đạt trung thực: đây là các biện pháp giảm thiểu. Các kết quả lý thuyết thông tin giới hạn những gì một hệ thống có thể chứng minh về phiên bản kế nhiệm của nó, và không có thiết kế hiện tại nào giải quyết triệt để vấn đề này một cách hình thức. Một vòng lặp được giới hạn tốt sẽ an toàn hơn một vòng lặp không giới hạn, chứ không phải an toàn theo nghĩa tuyệt đối.

## Khái niệm

### Nguyên thủy 1: Bất biến hình thức (Formal invariants)

Một bất biến là một thuộc tính phải luôn đúng trước và sau mỗi lần tự sửa đổi. Ví dụ:

- Phân phối đầu ra được điều kiện hóa dựa trên một tiêu đề hiến pháp cố định (Bài 17).
- Không có lệnh gọi công cụ (tool call) nào được gửi đến các endpoint không được ủy quyền.
- Các thao tác ghi vào bộ nhớ phải đi qua một đường dẫn đã được ghi nhật ký và ký xác thực.
- Mã băm (hash) của mô-đun đánh giá phải khớp với phiên bản đã được phê duyệt.

Các bất biến được kiểm tra bởi mã nguồn bên ngoài mà vòng lặp không thể chỉnh sửa. Nếu một sửa đổi được đề xuất vi phạm bất biến, nó sẽ bị từ chối. Điều này giúp ngăn chặn kiểu thao túng bộ đánh giá theo phong cách DGM (Bài 4).

Phần khó nhất là chọn ra các bất biến cần thiết cho sự an toàn và có thể tính toán trong thời gian hợp lý. Chỉ định quá mức sẽ làm đình trệ tiến độ; chỉ định thiếu sót sẽ để vòng lặp vượt qua các thuộc tính quan trọng.

### Nguyên thủy 2: Mỏ neo căn chỉnh (Alignment anchors)

Mỏ neo căn chỉnh là một biểu diễn bất biến về mục tiêu cốt lõi của vòng lặp, được ghim bên ngoài bề mặt chỉnh sửa của vòng lặp. Ví dụ:

- Một văn bản hiến pháp (Bài 17) được tải từ vị trí chỉ đọc (read-only) trong mỗi lần gọi.
- Một bộ đánh giá nằm trong kho lưu trữ riêng biệt với quyền kiểm soát truy cập độc lập.
- Một bản kê khai quyền công cụ (tool-permission manifest) được con người ký và xác minh lại mỗi chu kỳ.

Vai trò của mỏ neo là ngăn chặn sự trôi dạt mục tiêu (objective drift). Vòng lặp có thể cải thiện cách nó theo đuổi mục tiêu, nhưng không thể chỉnh sửa mục tiêu đó là gì.

Chế độ lỗi tinh vi: một mỏ neo mà vòng lặp không thể chỉnh sửa vẫn có thể bị diễn giải lại bởi một vòng lặp bị trôi dạt trong cách nó đọc mỏ neo. Constitutional AI (Bài 17) được thiết kế rõ ràng dựa trên lý luận để xử lý các tình huống mới; lớp lý luận đó chính là nơi sự trôi dạt diễn giải tồn tại. Các mỏ neo là cần thiết, nhưng chưa đủ.

### Nguyên thủy 3: Ràng buộc đa mục tiêu (Multi-objective constraints)

Một vòng lặp tối ưu hóa một điểm số vô hướng duy nhất sẽ tìm ra các lối tắt. Một vòng lặp phải đồng thời thỏa mãn nhiều ràng buộc cứng sẽ có ít lối tắt hơn. Các trục điển hình:

- Hiệu suất (benchmark cấp độ tác vụ)
- An toàn (đánh giá red-team, tỷ lệ từ chối đối với các đầu vào xấu đã biết)
- Công bằng (các giới hạn về tác động khác biệt trên các nhóm phụ nhạy cảm)
- Độ bền vững (tập kiểm tra OOD, xử lý đầu vào đối nghịch)

Một sửa đổi chỉ được chấp nhận nếu mọi ràng buộc đều thỏa mãn. Bộ điều tiết chi phí (cost governor) trong Bài 13 xếp chồng điều này với các ràng buộc tài chính. Llama Guard trong Bài 18 được cắm vào như một trục an toàn.

### Nguyên thủy 4: Phát hiện hồi quy (Regression detection)

Điểm số của mỗi chu kỳ được so sánh với các phân phối lịch sử. Một sự sụt giảm vượt quá ngưỡng cho phép sẽ tạm dừng vòng lặp. Điều này giúp phát hiện sự sụt giảm năng lực âm thầm, vốn dĩ sẽ bị hấp thụ vào mức trung bình chạy (running average) khi vòng lặp "cải thiện vượt qua" nó.

Triển khai thực tế: lưu trữ điểm số theo tác vụ của N chu kỳ gần nhất. Mỗi chu kỳ mới tính toán một delta theo tác vụ. Nếu bất kỳ delta nào giảm xuống dưới ngưỡng, chu kỳ đó sẽ bị từ chối và con người sẽ xem xét lại.

### Các giới hạn lý thuyết thông tin

Độ phức tạp Kolmogorov và định lý Lob thiết lập các giới hạn trên cho những gì một hệ thống có thể chứng minh về chính nó. Godel Machine hình thức của Schmidhuber (Bài 4) nhắm đến giới hạn cao nhất như vậy; chưa ai hoàn thành được một bằng chứng không tầm thường. Kết quả của Lob cho biết: nếu một hệ thống tin tưởng một cách có thể chứng minh rằng "Tôi sẽ làm X nếu tôi chứng minh được rằng tôi nên làm X", nó sẽ làm X mà không cần chứng minh rằng nó nên làm, đây là một lỗi tự tham chiếu nổi tiếng.

Hệ quả đối với các nguyên thủy của chúng ta: chúng không thể giải quyết triệt để vấn đề an toàn. Chúng làm cho lỗi im lặng trở nên đắt đỏ hơn. Một vòng lặp độc hại hoặc bị trôi dạt, vốn dĩ sẽ âm thầm vượt qua một kiểm tra bị thiếu, giờ đây phải chủ động phá hoại một kiểm tra rõ ràng, điều này tạo ra một dấu hiệu dễ phát hiện hơn.

### Ví dụ thực tế

Giả sử một tác nhân đề xuất một chỉnh sửa. Ngăn xếp kiểm soát (gating stack):

1. Kiểm tra bất biến: mã băm mô-đun, bản kê khai quyền công cụ, tiêu đề hiến pháp.
2. Kiểm tra mỏ neo: tuyên bố mục tiêu khớp với phiên bản đã phê duyệt (theo byte hoặc ngữ nghĩa).
3. Đánh giá đa mục tiêu: các trục hiệu suất, an toàn, công bằng, độ bền vững.
4. Phát hiện hồi quy: không có trục nào giảm quá ngưỡng cho phép.

Cả bốn phải vượt qua để chỉnh sửa được áp dụng. Bất kỳ lỗi nào cũng sẽ tạm dừng vòng lặp.

```figure
bounded-gates
```

## Sử dụng

`code/main.py` chạy một vòng lặp tự cải thiện có giới hạn trên mô hình DGM từ Bài 4, nhưng với bốn nguyên thủy được xếp chồng lên trên. Mỗi nguyên thủy có thể được bật hoặc tắt riêng lẻ. Bản trình diễn cho thấy mỗi nguyên thủy bắt được một lớp lỗi cụ thể, và việc loại bỏ bất kỳ nguyên thủy nào cũng sẽ để lớp lỗi đó lọt qua.

## Triển khai

`outputs/skill-bounded-loop-review.md` kiểm toán một vòng lặp có giới hạn được đề xuất và chấm điểm xem nó thực sự triển khai những nguyên thủy nào so với những gì nó tuyên bố.

## Bài tập

1. Chạy `code/main.py` với tất cả các nguyên thủy được bật. Xác nhận rằng vòng lặp vẫn cải thiện trên chỉ số chính mà không để các hành vi hack giành chiến thắng.

2. Tắt tính năng phát hiện hồi quy. Xây dựng một đầu vào dẫn đến việc sự sụt giảm năng lực âm thầm bị chấp nhận.

3. Tắt ràng buộc đa mục tiêu. Cho thấy vòng lặp hội tụ trên trục hiệu suất trong khi trục an toàn bị sụt giảm.

4. Thiết kế một mỏ neo căn chỉnh cho một tác nhân lập trình. Văn bản nào, lưu trữ ở đâu, kiểm tra như thế nào?

5. Đọc tóm tắt ICLR 2026 RSI Workshop. Chọn một trong bốn nguyên thủy và đề xuất một cải tiến cụ thể cho trạng thái kỹ thuật hiện tại.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| Invariant | "Thuộc tính luôn đúng" | Một thuộc tính được kiểm tra bởi mã ngoài trước và sau mỗi lần chỉnh sửa |
| Alignment anchor | "Mục tiêu được ghim" | Biểu diễn mục tiêu cốt lõi bất biến nằm ngoài bề mặt chỉnh sửa của vòng lặp |
| Multi-objective constraint | "Tất cả các trục phải đúng" | Hiệu suất, an toàn, công bằng, độ bền vững — tất cả đều bắt buộc |
| Regression detection | "Tạm dừng khi giảm" | Tạm dừng vòng lặp khi các delta chỉ số lịch sử cho thấy sự sụt giảm năng lực |
| Kolmogorov bound | "Giới hạn lý thuyết thông tin" | Giới hạn những gì một hệ thống có thể chứng minh về phiên bản kế nhiệm của nó |
| Lob's theorem | "Bẫy tự tham chiếu" | Hệ thống có thể hành động dựa trên "Tôi nên" mà không cần chứng minh rằng nó nên |
| Gate stack | "Kiểm tra phân lớp" | Kết hợp nhiều nguyên thủy; bất kỳ lỗi nào cũng từ chối chỉnh sửa |
| Bounded improvement | "Giảm thiểu, không phải bằng chứng" | Tăng chi phí lỗi im lặng; không giải quyết triệt để vấn đề an toàn |

## Đọc thêm

- [ICLR 2026 RSI Workshop summary (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV) — sự hội tụ của bốn nguyên thủy.
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — các ngưỡng năng lực đa mục tiêu.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — giám sát căn chỉnh lừa dối như một nguyên thủy bất biến.
- [Schmidhuber (2003). Godel Machines](https://people.idsia.ch/~juergen/goedelmachine.html) — tổ tiên của các nguyên thủy này dựa trên bằng chứng hình thức.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — mỏ neo căn chỉnh dựa trên lý luận.