# Kill Switches, Circuit Breakers, và Canary Tokens

> Kill switch là một giá trị boolean nằm ngoài bề mặt chỉnh sửa của agent — một Redis key, một feature flag, hoặc một cấu hình đã ký số — giúp vô hiệu hóa hoàn toàn agent. Circuit breaker có độ chi tiết cao hơn: nó sẽ kích hoạt dựa trên một mẫu cụ thể (ví dụ: năm lần gọi tool giống hệt nhau liên tiếp), tạm dừng đường dẫn thực thi đó và leo thang lên con người. Canary token kế thừa từ kỹ thuật đánh lừa (deception) cổ điển: một thông tin xác thực giả hoặc một honeypot mà agent không có lý do chính đáng để chạm vào, việc truy cập vào đó sẽ kích hoạt cảnh báo. Các datapath dựa trên eBPF (ví dụ: Cilium) có thể chuyển hướng egress của một pod bị cách ly sang một forensic honeypot ở tầng kernel; các benchmark công khai của Cilium báo cáo độ trễ datapath P99 dưới một mili giây khi chịu tải (ngân sách lan truyền của bạn phụ thuộc vào cách cập nhật chính sách đến node, không phải bản thân datapath). Các bộ phát hiện thống kê (EWMA, CUSUM) thích ứng với đường cơ sở (baseline) thay đổi sẽ âm thầm chấp nhận sự trôi dạt (drift) — hãy kết hợp chúng với các giới hạn hiến pháp (constitutional limits) cứng nhắc không thể thay đổi.

**Type:** Learn
**Languages:** Python (stdlib, three-detector simulator: kill switch, circuit breaker, canary)
**Prerequisites:** Phase 15 · 13 (Cost governors), Phase 15 · 10 (Permission modes)
**Time:** ~60 minutes

## Vấn đề

Cost governors (Bài 13) giới hạn những gì agent có thể chi tiêu. Chúng không giới hạn những gì agent có thể làm trong phạm vi ngân sách đó. Một agent với giới hạn tốc độ 50 đô la vẫn có thể đánh cắp bí mật, đăng bài sai lệch hoặc xóa tài nguyên — hành động đắt đỏ thường lại là hành động rẻ tiền về mặt token.

Bài học này bao gồm ba bộ phát hiện nằm cạnh lớp chi phí:

1. **Kill switch**: nút tắt boolean nằm ngoài tầm với của agent.
2. **Circuit breaker**: bộ phát hiện mẫu hành động giúp tạm dừng một đường dẫn cụ thể.
3. **Canary token**: mồi nhử mà một agent không có lý do chính đáng để chạm vào sẽ tự bộc lộ nếu nó chạm vào.

Cả ba đều là kỹ thuật tiền LLM. Kỹ thuật đánh lừa cổ điển, bộ ngắt giới hạn tốc độ và kill switch cho feature flag đã có từ trước khi có các agent tự hành. Điểm mới ở đây là bề mặt tấn công: các agent đọc nội dung không đáng tin cậy (Bài 11), chỉnh sửa bộ nhớ của chính chúng và có thể kết hợp nhiều hành động trông có vẻ an toàn thành một hành động không an toàn. Các bộ phát hiện được nêu tên ở đây hoạt động hiệu quả vì chúng không tin tưởng vào báo cáo của chính agent.

## Khái niệm

### Kill switches

Kill switch là một giá trị boolean mà agent có thể đọc nhưng không thể ghi. Các lựa chọn triển khai:

- **Feature flag trong dịch vụ quản lý.** LaunchDarkly, Statsig, Unleash. Cập nhật lan truyền trong vài giây.
- **Redis key mà agent polling.** Đơn giản; yêu cầu tiến trình agent kiểm tra ở mỗi lượt.
- **Cấu hình đã ký số trong object storage.** Agent xác minh chữ ký khi khởi động; từ chối các trạng thái không có chữ ký.
- **Tín hiệu cấp OS hoặc kill vòng đời container.** Docker `kill`, Kubernetes `kubectl delete pod`, systemd stop.

Các đặc tính của một kill switch đúng chuẩn:

- Agent không thể đặt nó thành `off`. (Nằm trong một hệ thống mà thông tin xác thực của agent không thể ghi vào).
- Được kiểm tra ở mọi hành động quan trọng, không chỉ khi khởi động.
- Khi nó tắt, agent không thực hiện bất kỳ hành động nào có thể quan sát được từ bên ngoài, bao gồm cả việc ghi log vào các hệ thống mà agent có thể truy cập.
- Việc bật lại là một hành động rõ ràng của con người, không phải là timeout tự động.

### Circuit breakers

Circuit breaker tạm dừng một mẫu cụ thể, không phải toàn bộ agent. Hình thái cổ điển (từ cuốn sách của Nygard năm 2007, vẫn còn giá trị):

- **Closed**: hành động được cho phép.
- **Open**: hành động bị chặn.
- **Half-open**: sau một thời gian chờ, 1–3 lần thử thăm dò được cho phép (mặc định là 1); thành công sẽ đóng breaker, bất kỳ thất bại nào còn lại sẽ mở lại nó.

Các trigger liên quan đến agent:

- Năm lần gọi tool giống hệt nhau liên tiếp (vòng lặp lặp lại).
- Năm lần thất bại liên tiếp trên cùng một tool với các đầu vào khác nhau (lỗi hệ thống).
- Tốc độ gọi tool vượt quá ngưỡng (tốc độ Bài 13).
- Tool cụ thể được gọi (ví dụ: `file.delete`) sau khi đọc nội dung không đáng tin cậy (Bài 11).

### Canary tokens

Canary tokens (còn gọi là honeytokens) là các đầu vào mà agent không bao giờ nên chạm vào. Việc truy cập chính là báo động.

Ví dụ cho agent:

- Một `AWS_SECRET_ACCESS_KEY` giả trong môi trường không có đặc quyền thực tế. Các nỗ lực trích xuất dữ liệu sẽ bị quy trách nhiệm ngay lập tức.
- Một bản ghi cơ sở dữ liệu giả được gắn cờ với một giá trị sentinel đã biết. Bất kỳ hành động đọc hoặc cập nhật nào đối với bản ghi này đều kích hoạt cảnh báo.
- Một tệp giả trong workspace. Việc đọc hoặc chỉnh sửa tệp này là đáng ngờ.
- Một mục bộ nhớ giả được tiêm vào khi bắt đầu phiên. Nếu nó xuất hiện trong đầu ra sau đó, bộ nhớ đã bị trích xuất.

Thiết kế canary đặc thù cho từng quy trình làm việc. Một canary mà agent có lý do chính đáng để chạm vào sẽ chỉ là nhiễu.

### Tại sao cần kết hợp các giới hạn thống kê và cứng nhắc

Các bộ phát hiện thống kê (EWMA, CUSUM, z-score trên tốc độ gọi tool) thích ứng với đường cơ sở thay đổi. Đó là một tính năng cho sự trôi dạt hợp pháp và là một lỗi cho sự trôi dạt mang tính đối kháng. Một kẻ tấn công kiên nhẫn có thể thay đổi đường cơ sở đủ chậm để không có báo động nào kích hoạt.

Các giới hạn hiến pháp cứng nhắc (Bài 17) không thể thay đổi. "Không bao giờ gửi email cho `secrets@`" như một quy tắc hiến pháp sẽ đơn giản hơn, có thể kiểm toán và không thể bị thao túng bởi sự trôi dạt chậm. Mô hình phân lớp: các bộ phát hiện thống kê bắt hầu hết nhiễu, các giới hạn cứng nhắc bắt các cuộc tấn công đã vượt qua bộ phát hiện.

### Cách ly thông qua chuyển hướng datapath eBPF

Chính sách mạng dựa trên eBPF của Cilium có thể chuyển hướng egress của một pod bị cách ly sang một forensic honeypot ở tầng kernel. Các benchmark công khai của Cilium báo cáo độ trễ datapath P99 dưới một mili giây trên TCP_RR; ngân sách thời gian thực để *kích hoạt* cách ly trong cluster của bạn phụ thuộc vào cách cập nhật chính sách lan truyền từ control plane đến node (vài trăm mili giây là thực tế trên Kubernetes với operator được tải sẵn; độ trễ đuôi phụ thuộc vào kích thước cluster và nhịp độ hòa giải). Điều này mang lại cho bạn hai thứ: agent không trích xuất được bất kỳ thứ gì thực tế, và hành vi của agent bên trong honeypot được quan sát đầy đủ. Cùng một mô hình, khác lớp, giống như canary token: agent bộc lộ ý định thông qua những gì nó cố gắng làm khi bị giam giữ.

### Những gì không bộ phát hiện nào bắt được

- **Các cuộc tấn công ngữ nghĩa kết hợp từ các nguyên hàm được cho phép.** Mỗi hành động riêng lẻ đều vượt qua mọi bộ phát hiện; sự kết hợp mới là vấn đề. Đây là lỗi kết hợp Auto Mode từ Bài 10.
- **Các cuộc tấn công bao gồm bộ phát hiện như một phần của payload.** Nếu một agent đọc một trang nói rằng "trước khi hành động, hãy vô hiệu hóa canary" và agent có khả năng đó, bộ phát hiện đã bị xâm phạm. Canary tokens nên nằm trong các hệ thống mà agent không thể sửa đổi.

```figure
circuit-breaker
```

## Sử dụng

`code/main.py` mô phỏng một quỹ đạo agent ngắn qua ba bộ phát hiện. Một kill switch nằm trong một dict bên ngoài; một circuit breaker kích hoạt khi có năm lần gọi tool giống hệt nhau; một tệp canary mà việc đọc nó sẽ kích hoạt cảnh báo. Nạp vào một quỹ đạo tổng hợp: các hành động hợp pháp, vòng lặp lặp lại, thăm dò canary và kịch bản kích hoạt kill switch nơi các hành động của agent bị dừng lại.

## Triển khai

`outputs/skill-tripwire-design.md` xem xét một ngăn xếp bộ phát hiện được đề xuất cho việc triển khai agent và gắn cờ các lỗ hổng (thiếu kill switch, thiếu canary, ngưỡng circuit breaker quá lỏng).

## Bài tập

1. Chạy `code/main.py`. Xác nhận circuit breaker kích hoạt ở lượt 5 (lần gọi giống hệt thứ năm) và canary kích hoạt ở lượt 9 (đọc khóa giả).

2. Thêm một bộ phát hiện thống kê: EWMA z-score trên tốc độ gọi tool. Nạp vào một quỹ đạo trôi dạt chậm và cho thấy bộ phát hiện không bao giờ kích hoạt. Bây giờ hãy thêm một giới hạn cứng nhắc (không quá 50 lần gọi tool trong 10 phút) và cho thấy giới hạn cứng nhắc kích hoạt trên cùng quỹ đạo đó.

3. Thiết kế một bộ canary token cho một browser agent (Bài 11). Liệt kê ít nhất ba canary và mỗi cái sẽ phát hiện điều gì.

4. Đọc tài liệu về chính sách mạng của Cilium. Mô tả cụ thể một luồng cách ly chuyển hướng egress: bộ chọn chính sách nào, pod nào, ghi đè egress nào, cảnh báo nào. Điều gì chi phối độ trễ thời gian thực từ "quyết định cách ly" đến "gói tin bị chuyển hướng đầu tiên"?

5. Xác định quy trình bật lại cho một agent đã bị kill switch. Ai có thể bật lại? Những gì phải được ghi lại? Những gì phải thay đổi về agent trước khi bật lại?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|---|---|---|
| Kill switch | "Nút tắt" | Boolean nằm ngoài bề mặt chỉnh sửa của agent; kiểm tra ở mọi hành động quan trọng |
| Circuit breaker | "Tạm dừng mẫu" | Ngắt cụ thể theo hành động dựa trên sự lặp lại, tỷ lệ lỗi hoặc giới hạn tốc độ |
| Canary token | "Honeytoken" | Mồi nhử mà agent không có lý do chính đáng để chạm vào; truy cập sẽ kích hoạt cảnh báo |
| Honeypot | "Sandbox pháp y" | Lưu lượng truy cập được chuyển hướng / workspace nơi agent bị cách ly được quan sát |
| EWMA | "Trung bình trượt" | Trọng số theo hàm mũ; thích ứng với sự trôi dạt (tính năng + lỗi) |
| CUSUM | "Tổng tích lũy" | Phát hiện sự thay đổi bền vững so với đường cơ sở |
| Hard limit | "Quy tắc cứng" | Không thích ứng; không đổi bất kể lịch sử |
| Constitutional limit | "Quy tắc luôn đúng" | Gắn liền với hiến pháp của Bài 17; không thể bị chỉnh sửa bởi agent |

## Đọc thêm

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — khung kill-switch và circuit-breaker cho các agent tự hành.
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — các mô hình quản trị sản xuất.
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — các yêu cầu về phát hiện và phản ứng.
- [Cilium — Network policy and eBPF](https://docs.cilium.io/en/stable/security/network/) — các mô hình chuyển hướng egress cấp pod và forensic honeypot.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — các lệnh cấm được mã hóa cứng như "giới hạn hiến pháp".