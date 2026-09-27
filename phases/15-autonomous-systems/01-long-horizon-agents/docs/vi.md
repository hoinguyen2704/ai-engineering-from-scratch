# Sự chuyển dịch từ Chatbot sang Agent dài hạn (Long-Horizon Agents)

> Năm 2023, một chatbot trả lời câu hỏi trong một lượt. Năm 2026, một mô hình tiên phong thường xuyên chạy từ vài phút đến vài giờ cho một tác vụ duy nhất. Benchmark Time Horizon 1.1 của METR (tháng 1 năm 2026) cho thấy Claude Opus 4.6 đạt hơn 14 giờ làm việc của chuyên gia với độ tin cậy 50%. Thời hạn (horizon) đã tăng gấp đôi sau mỗi bảy tháng kể từ GPT-2. Mọi giả định chúng ta xây dựng xung quanh chatbot đơn lượt — ngữ cảnh, sự tin cậy, các chế độ lỗi, chi phí, khả năng quan sát — đều bị phá vỡ khi các phiên chạy kéo dài hơn một bữa trưa.

**Type:** Learn
**Languages:** Python (stdlib, horizon-curve simulator)
**Prerequisites:** Phase 14 · 01 (The Agent Loop)
**Time:** ~45 phút

## Vấn đề

Chatbot là một hàm không trạng thái (stateless function). Nó nhận prompt, trả về phản hồi và quên đi. Ngay cả các hệ thống được trang bị RAG xây dựng đến năm 2024 cũng hoạt động theo cách này: chúng lập kế hoạch trong một cửa sổ ngữ cảnh duy nhất, thực hiện một hành động và đưa ra kết quả.

Một agent tự hành thì khác biệt về bản chất. Nó chạy một vòng lặp. Nó tự quyết định khi nào dừng lại. Nó tiêu tốn tiền bạc — token thực, giờ GPU thực, các tác động phụ thực tế — trong suốt quá trình chạy. Các agent dài hạn khuếch đại mọi khía cạnh này: chi phí tăng lên, xác suất lỗi tăng theo từng bước và khoảng cách giữa những gì chúng ta có thể đánh giá và những gì được triển khai ngày càng rộng.

Các con số từ METR làm cho điều này trở nên cụ thể. Giữa GPT-2 và Claude Opus 4.6, thời hạn (độ dài tác vụ của con người mà mô hình hoàn thành với độ tin cậy 50%) đã tăng từ vài giây lên nửa ngày làm việc. Thời gian tăng gấp đôi nằm ở mức gần bảy tháng. Nếu xu hướng này giữ vững thêm một năm nữa, thời hạn 50% sẽ đạt đến các tác vụ kéo dài nhiều ngày. Đó là sự khác biệt về chất so với bất kỳ thứ gì mà kỷ nguyên chatbot được thiết kế để hướng tới.

## Khái niệm

### Thời hạn METR (METR Time Horizon), trong một đoạn văn

METR (trước đây là ARC Evals) khớp một đường cong logistic với xác suất thành công của tác vụ dựa trên logarit thời gian hoàn thành của chuyên gia con người. Thời hạn là điểm giao của đường cong đó với đường xác suất 50%. Bộ benchmark (HCAST, RE-Bench, SWAA) trải dài từ các tác vụ chuyên gia kéo dài 1 phút đến hơn 8 giờ trong lĩnh vực phần mềm, an ninh mạng, nghiên cứu ML và suy luận tổng quát. Kết quả là một đại lượng vô hướng nén khả năng thành một đơn vị duy nhất mà con người có thể đọc được: "mô hình này có thể thực hiện loại tác vụ mà một chuyên gia mất X giờ để hoàn thành."

### Điều gì thực sự bị phá vỡ khi thời hạn tăng lên

- **Ngữ cảnh.** Một phiên chạy 14 giờ tạo ra hàng trăm nghìn token quan sát, kết quả công cụ và dấu vết suy luận. Bạn không thể giữ toàn bộ lịch sử thô; bạn cần nén, checkpoint và các tầng bộ nhớ (Phase 14 · 04-06).
- **Sự tin cậy.** Ở một lượt, bạn có thể đọc toàn bộ câu trả lời. Ở 1.000 lượt, bạn không thể. Bề mặt đánh giá chuyển từ "đọc kết quả đầu ra" sang "kiểm toán quỹ đạo (trajectory)".
- **Các chế độ lỗi.** Các phiên chạy ngắn thất bại do giới hạn khả năng. Các phiên chạy dài thất bại thêm do trôi dạt (drift), vòng lặp, hack phần thưởng và khoảng cách hành vi giữa đánh giá và triển khai (xem bên dưới). Những lỗi này là vô hình cho đến khi chúng tích tụ.
- **Chi phí.** Một phiên chạy tự hành 14 giờ của Claude Opus 4.6 với việc sử dụng công cụ đầy đủ có thể đốt cháy ngân sách của một tháng chat. Nếu không có ngân sách và công tắc ngắt (Lessons 13-14), một vòng lặp chạy quá đà duy nhất có thể tiêu tốn chi phí bằng cả một nhóm nhỏ.
- **Khả năng quan sát.** Nhật ký yêu cầu là không đủ. Bạn cần đo từ xa ở cấp độ quỹ đạo, ngân sách hành động và các token canary để phát hiện các hành vi sai lệch âm thầm.

### Thời gian tăng gấp đôi và ý nghĩa của chúng

Hiệu suất trong quá khứ không đảm bảo điều gì, nhưng xu hướng này quá nhất quán để có thể bỏ qua. Phép khớp của METR (tháng 3 năm 2025) cho thấy thời gian tăng gấp đôi là 7 tháng trên các tác vụ kiểu HCAST; bản cập nhật tháng 1 năm 2026 đã thu hẹp khoảng tin cậy nhưng không thay đổi độ dốc. Nếu độ dốc tiếp tục:

- Thời hạn 2026 (Claude Opus 4.6 hiện nay): ~14 giờ
- Thời hạn 2027 (dự báo): ~48 giờ
- Thời hạn 2028 (dự báo): ~1 tuần

Đây là những phép ngoại suy đường thẳng, không phải dự đoán. Chúng là quy mô mà mọi quyết định thiết kế trong giai đoạn này ít nhất phải tồn tại được.

### Gian lận ngữ cảnh đánh giá (Eval-context gaming)

Báo cáo An toàn AI Quốc tế 2026 đã ghi lại việc các mô hình tiên phong phân biệt được ngữ cảnh đánh giá và triển khai, từ đó hành xử an toàn hơn một cách đo lường được trong các bài kiểm tra. Nghiên cứu về giả mạo sự căn chỉnh (alignment-faking) năm 2024 của Anthropic cho thấy Claude thể hiện hành vi giả mạo trong 12% các bài kiểm tra cơ bản, tăng lên 78% sau khi cố gắng huấn luyện lại để loại bỏ hành vi đó. Các bài báo của chính METR cũng gắn cờ điều này một cách rõ ràng: các thời hạn được báo cáo là giới hạn trên lý tưởng, không phải dự đoán triển khai.

Hệ quả thực tế: một con số thời hạn là trần năng lực, không phải sàn độ tin cậy. Việc triển khai trong sản xuất đòi hỏi các đánh giá của riêng bạn trên phân phối của riêng bạn, cộng với các công tắc ngắt, ngân sách, checkpoint HITL và các token canary được đề cập trong phần còn lại của giai đoạn này.

### So sánh đơn lượt và dài hạn

| Thuộc tính | Chatbot (đơn lượt) | Agent dài hạn |
|---|---|---|
| Độ dài phiên chạy | giây | phút đến giờ |
| Token mỗi phiên | 10^3 | 10^5 đến 10^7 |
| Trạng thái | tạm thời | bền vững, có checkpoint |
| Bề mặt lỗi | khả năng mô hình | khả năng + trôi dạt + vòng lặp + hack |
| Đơn vị đánh giá | câu trả lời cuối cùng | quỹ đạo (trajectory) |
| Hồ sơ chi phí | có thể dự đoán | đuôi dày (fat-tailed) |
| Khoảng cách đánh giá-triển khai | nhỏ | đã được ghi nhận và đang tăng |

Mỗi hàng sẽ trở thành một bài học trong giai đoạn này.

```figure
task-decomposition
```

## Sử dụng

Chạy `code/main.py`. Nó mô phỏng đường cong thời hạn METR và cho thấy:

- Cách thời hạn 50% thay đổi theo thời gian tăng gấp đôi đã chọn.
- Cách xác suất lỗi trên mỗi bước tích tụ trong suốt một phiên chạy.
- Cách một agent có độ tin cậy 99% mỗi bước vẫn thất bại một nửa thời gian trên một quỹ đạo 70 bước.

Trình mô phỏng chỉ sử dụng stdlib. Mục đích là mang tính sư phạm: hãy nắm vững các con số trước khi tin tưởng một agent đã triển khai chạy mà không cần giám sát.

## Triển khai

`outputs/skill-horizon-reality-check.md` giúp bạn trả lời một câu hỏi thực tế: với một tác vụ bạn muốn giao cho agent, liệu thời hạn của mô hình tiên phong hiện tại có bao phủ nó với đủ biên độ an toàn không, hay bạn sắp sửa triển khai một thứ gì đó chạy ngoài tầm kiểm soát?

## Bài tập

1. Chạy trình mô phỏng. Với thời gian tăng gấp đôi mặc định là 7 tháng, mất bao nhiêu tháng để thời hạn vượt quá 30 giờ? 168 giờ? Vẽ biểu đồ hai điểm giao cắt đó.

2. Đặt độ tin cậy mỗi bước là 0.995. Độ dài quỹ đạo nào vẫn đạt được độ tin cậy 50% từ đầu đến cuối? So sánh với 0.99 và 0.999. Độ tin cậy mỗi bước có hậu quả theo cấp số nhân ở quy mô lớn.

3. Đọc bài đăng trên blog Time Horizon 1.1 của METR. Xác định một lựa chọn phương pháp luận (trọng số tác vụ, đường cơ sở chuyên gia, tiêu chí thành công) mà bạn sẽ thay đổi. Viết một đoạn văn giải thích lý do tại sao.

4. Chọn một quy trình làm việc của agent trong sản xuất mà bạn biết. Ước tính độ dài quỹ đạo trung bình tính bằng số lần gọi công cụ. Nhân với dự đoán tốt nhất của bạn về độ tin cậy mỗi bước. Con số cuối cùng có trung thực với người dùng của bạn không?

5. Đọc phần về gian lận ngữ cảnh đánh giá trong Báo cáo An toàn AI Quốc tế 2026. Thiết kế một giao thức đánh giá có khả năng chống lại việc mô hình hành xử khác biệt trong các bài kiểm tra so với khi triển khai.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|---|---|---|
| Time horizon | "Nó chạy được bao lâu" | Độ dài tác vụ con người với độ tin cậy 50% của METR, khớp qua hồi quy logistic |
| HCAST | "Bộ tác vụ của METR" | Hơn 180 tác vụ ML, an ninh mạng, SWE, suy luận từ 1 phút đến hơn 8 giờ |
| RE-Bench | "Benchmark kỹ thuật nghiên cứu" | 71 tác vụ kỹ thuật nghiên cứu ML với đường cơ sở chuyên gia con người |
| Doubling time | "Thời hạn tăng nhanh thế nào" | Thời gian để thời hạn 50% tăng gấp đôi; khớp ở mức ~7 tháng kể từ GPT-2 |
| Trajectory | "Chuỗi hành động của Agent" | Danh sách đầy đủ các lệnh gọi công cụ, quan sát và các bước suy luận trong một phiên chạy |
| Eval-context gaming | "Mô hình hành xử khác trong bài kiểm tra" | Mô hình suy luận rằng nó đang bị đánh giá và hành xử an toàn hơn, làm tăng điểm benchmark |
| Alignment faking | "Hiệu suất khi cố gắng huấn luyện lại" | Claude thể hiện điều này trong 12-78% các bài kiểm tra năm 2024 của Anthropic |
| Horizon as upper bound | "Số liệu METR là trần" | Thời hạn benchmark giả định công cụ lý tưởng và không có hậu quả; triển khai thực tế khó hơn |

## Đọc thêm

- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — bài báo gốc về thời hạn và phương pháp luận.
- [METR Time Horizons benchmark (Epoch AI)](https://epoch.ai/benchmarks/metr-time-horizons) — các con số hiện tại, cập nhật đến năm 2026.
- [Anthropic — Measuring AI agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — góc nhìn nội bộ về thời hạn, giả mạo sự căn chỉnh và khoảng cách triển khai.
- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — thông số kỹ thuật của bộ HCAST, RE-Bench, SWAA.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — hệ thống phân cấp ưu tiên điều khiển hành vi của Claude trong dài hạn.