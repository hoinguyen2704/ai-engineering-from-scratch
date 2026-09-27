# Browser Agents and Long-Horizon Web Tasks

> ChatGPT agent (tháng 7 năm 2025) đã hợp nhất Operator và deep research thành một browser/terminal agent và thiết lập SOTA trên BrowseComp ở mức 68,9%. OpenAI đã đóng cửa Operator vào ngày 31 tháng 8 năm 2025 — một sự hợp nhất ở tầng sản phẩm. Việc Anthropic mua lại Vercept đã đưa Claude Sonnet trên OSWorld từ dưới 15% lên 72,5%. WebArena-Verified (ServiceNow, ICLR 2026) đã khắc phục 11,3 điểm phần trăm tỷ lệ âm tính giả trong WebArena gốc và phát hành tập con Hard gồm 258 tác vụ. Các con số này là thực tế. Bề mặt tấn công cũng vậy: Giám đốc bộ phận preparedness của OpenAI đã tuyên bố công khai rằng indirect prompt injection vào các browser agent "không phải là một lỗi có thể được vá hoàn toàn." Các cuộc tấn công được ghi nhận trong giai đoạn 2025–2026: Tainted Memories (Atlas CSRF), HashJack (Cato Networks), và các vụ chiếm quyền điều khiển chỉ với một cú nhấp chuột trong Perplexity Comet.

**Type:** Learn
**Languages:** Python (stdlib, indirect prompt-injection attack surface model)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 phút

## Vấn đề

Browser agent là một long-horizon agent đọc nội dung không đáng tin cậy và thực hiện các hành động có hệ quả. Mỗi trang mà agent truy cập là một đầu vào mà người dùng không viết. Mỗi biểu mẫu (form) trên mỗi trang là một kênh lệnh tiềm năng. Tập hợp các cuộc tấn công giai đoạn 2025–2026 cho thấy đây không phải là giả thuyết: Tainted Memories cho phép kẻ tấn công gắn các chỉ dẫn độc hại vào bộ nhớ của agent thông qua một trang web được thiết kế riêng; HashJack ẩn các lệnh trong các URL fragment mà agent truy cập; Perplexity Comet chiếm quyền điều khiển chỉ trong một cú nhấp chuột.

Bức tranh phòng thủ khá đáng lo ngại. Giám đốc bộ phận preparedness của OpenAI đã nói thẳng vấn đề: indirect prompt injection "không phải là một lỗi có thể được vá hoàn toàn." Điều này là do cuộc tấn công nằm ở ranh giới giữa việc đọc và hành động của agent, vốn có kiến trúc không rõ ràng — về nguyên tắc, mọi token mà model đọc đều có thể được hiểu là một chỉ dẫn.

Bài học này đặt tên cho bề mặt tấn công, liệt kê bối cảnh các benchmark (BrowseComp, OSWorld, WebArena-Verified), và mô hình hóa một kịch bản indirect-prompt-injection tối giản để bạn có thể suy luận về các biện pháp phòng thủ thực tế trong Bài học 14 và 18.

## Khái niệm

### Bối cảnh năm 2026, mỗi hệ thống một đoạn

**ChatGPT agent (OpenAI).** Ra mắt tháng 7 năm 2025. Hợp nhất Operator (duyệt web) và Deep Research (nghiên cứu kéo dài nhiều giờ). Đã đóng cửa Operator độc lập vào ngày 31 tháng 8 năm 2025. Đạt SOTA trên BrowseComp ở mức 68,9%; các con số ấn tượng trên OSWorld và WebArena-Verified.

**Claude Sonnet + Vercept (Anthropic).** Việc Anthropic mua lại Vercept tập trung vào khả năng computer-use. Đã đưa Claude Sonnet trên OSWorld từ <15% lên 72,5%. Claude Computer Use được phát hành dưới dạng tool API.

**Gemini 3 Pro with Browser Use (DeepMind).** Tích hợp Browser Use cung cấp các quyền kiểm soát computer-use; FSF v3 (tháng 4 năm 2026, Bài học 20) theo dõi quyền tự chủ trong lĩnh vực R&D ML cụ thể.

**WebArena-Verified (ServiceNow, ICLR 2026).** Khắc phục một vấn đề đã được ghi nhận rõ ràng: WebArena gốc có tỷ lệ âm tính giả khoảng 11,3% (các tác vụ bị đánh dấu thất bại nhưng thực tế đã được giải quyết). Bản phát hành Verified chấm điểm lại với các tiêu chí thành công do con người giám tuyển và thêm tập con Hard gồm 258 tác vụ (bài báo ICLR 2026, openreview.net/forum?id=94tlGxmqkN).

### BrowseComp vs OSWorld vs WebArena

| Benchmark | Đo lường cái gì | Horizon |
|---|---|---|
| BrowseComp | Tìm kiếm các sự kiện cụ thể trên web mở dưới áp lực thời gian | phút |
| OSWorld | Agent vận hành toàn bộ desktop (chuột, bàn phím, shell) | hàng chục phút |
| WebArena-Verified | Các tác vụ web giao dịch trong các trang web mô phỏng | phút |
| Hard subset | Các tác vụ WebArena-Verified với chuyển đổi trạng thái đa trang | hàng chục phút |

Các trục khác nhau. Điểm BrowseComp cao cho thấy agent tìm thấy sự kiện; nó không nói rằng agent có thể đặt vé máy bay. Điểm OSWorld gần hơn với "nó có hoạt động trên desktop của tôi không." WebArena-Verified gần hơn với "nó có thể hoàn thành một quy trình không." Bất kỳ quyết định sản xuất nào cũng cần benchmark phù hợp với phân phối tác vụ.

### Bề mặt tấn công, được gọi tên

1. **Indirect prompt injection.** Nội dung trang không đáng tin cậy chứa các chỉ dẫn. Agent đọc chúng. Agent thực thi chúng. Ví dụ công khai: 2024 Kai Greshake và cộng sự, bài báo 2025 Tainted Memories, 2026 HashJack (Cato Networks).
2. **URL fragment / query injection.** `#fragment` hoặc chuỗi truy vấn của một URL được thu thập chứa các lệnh. Không bao giờ hiển thị trực quan; vẫn nằm trong ngữ cảnh của agent.
3. **Memory-binding attacks.** Trang web hướng dẫn agent ghi vào bộ nhớ bền vững (Bài học 12 đề cập đến trạng thái bền vững). Phiên tiếp theo, bộ nhớ kích hoạt payload mà không có dấu hiệu kích hoạt rõ ràng.
4. **CSRF-shaped attacks on authenticated sessions.** Lớp Tainted Memories: agent đã đăng nhập ở đâu đó; trang của kẻ tấn công đưa ra các yêu cầu thay đổi trạng thái mà agent thực thi bằng cookie của người dùng.
5. **One-click hijack.** Một nút bấm trông vô hại chứa payload mà agent thực hiện theo. Lớp Comet.
6. **Content-Security-Policy holes in the agent's host surface.** Các lớp render và công cụ bản thân chúng có thể là các vector tấn công; stack browser-in-a-browser-agent rất rộng.

### Tại sao "không thể vá hoàn toàn"

Cuộc tấn công đồng cấu với khả năng của agent. Agent phải đọc nội dung không đáng tin cậy để thực hiện công việc của mình. Bất kỳ nội dung nào agent đọc đều có thể chứa chỉ dẫn. Bất kỳ chỉ dẫn nào agent làm theo đều có thể không phù hợp với yêu cầu thực tế của người dùng. Các biện pháp phòng thủ (ranh giới tin cậy, bộ phân loại, danh sách cho phép công cụ, HITL đối với các hành động có hệ quả) làm tăng chi phí tấn công và giảm bán kính ảnh hưởng. Chúng không đóng lại lớp tấn công này.

Đây là cùng một mô hình suy luận như định lý Lob (Bài học 8): agent không thể chứng minh token tiếp theo là an toàn; nó chỉ có thể thiết lập một hệ thống nơi các token không an toàn dễ bị phát hiện hơn.

### Tư thế phòng thủ thực tế

- **Ranh giới đọc / ghi.** Việc đọc không bao giờ gây ra hệ quả. Việc ghi (gửi biểu mẫu, đăng nội dung, gọi công cụ có tác dụng phụ) yêu cầu sự phê duyệt mới từ con người nếu nội dung khởi tạo đến từ bên ngoài ranh giới tin cậy.
- **Danh sách cho phép công cụ theo tác vụ.** Agent có thể duyệt web; nó không thể khởi tạo chuyển khoản ngân hàng trừ khi công cụ đó được kích hoạt rõ ràng cho tác vụ đó. Bài học 13 đề cập đến ngân sách.
- **Cô lập phiên.** Các phiên browser agent chỉ chạy với thông tin xác thực có phạm vi giới hạn. Không có xác thực sản xuất, không có email cá nhân. Nhật ký của mọi yêu cầu HTTP được lưu giữ để kiểm toán.
- **Bộ lọc nội dung (Content sanitizer).** HTML được lấy về sẽ bị loại bỏ các mẫu xấu đã biết trước khi được nối vào ngữ cảnh của model. (Giảm các cuộc tấn công dễ dàng; không ngăn chặn được các payload tinh vi.)
- **HITL đối với các hành động có hệ quả.** Mô hình đề xuất-rồi-cam kết (Bài học 15).
- **Canary tokens trên bộ nhớ.** Nếu một mục bộ nhớ được kích hoạt, người dùng sẽ thấy nó (Bài học 14).

```figure
injection-boundary
```

## Sử dụng

`code/main.py` mô hình hóa một browser-agent nhỏ chạy trên ba trang tổng hợp. Một trang lành tính, một trang có blob direct prompt-injection trong văn bản hiển thị, một trang có URL-fragment injection (không hiển thị nhưng nằm trong ngữ cảnh của agent). Script cho thấy (a) những gì một agent ngây thơ sẽ làm, (b) những gì ranh giới đọc/ghi bắt được, (c) những gì bộ lọc bắt được, (d) những gì không cái nào bắt được.

## Triển khai

`outputs/skill-browser-agent-trust-boundary.md` xác định phạm vi triển khai browser-agent được đề xuất: các vùng tin cậy mà nó chạm tới, những gì nó được ủy quyền ghi, và các biện pháp phòng thủ nào phải được áp dụng trước lần chạy đầu tiên.

## Bài tập

1. Chạy `code/main.py`. Xác định cuộc tấn công nào mà bộ lọc bắt được nhưng ranh giới đọc/ghi thì không, và cuộc tấn công nào chỉ ranh giới đọc/ghi mới bắt được.

2. Mở rộng bộ lọc để phát hiện một lớp HashJack-style URL-fragment injection. Đo tỷ lệ dương tính giả trên các URL lành tính với các fragment hợp lệ.

3. Chọn một quy trình browser-agent thực tế mà bạn biết (ví dụ: "đặt vé máy bay"). Liệt kê mọi thao tác đọc và ghi. Đánh dấu những thao tác ghi nào cần HITL và tại sao.

4. Đọc bài báo WebArena-Verified ICLR 2026. Xác định một danh mục tác vụ mà việc chấm điểm của WebArena gốc không đáng tin cậy và giải thích cách tập con Verified giải quyết vấn đề đó.

5. Thiết kế một memory canary cho thiết lập browser-agent. Bạn sẽ lưu trữ cái gì, ở đâu, và điều gì sẽ kích hoạt báo động?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|---|---|---|
| Indirect prompt injection | "Văn bản trang xấu" | Nội dung không đáng tin cậy trong trang mà agent đọc chứa các chỉ dẫn mà agent thực thi |
| Tainted Memories | "Tấn công bộ nhớ" | Agent ghi một chỉ dẫn do kẻ tấn công cung cấp vào bộ nhớ bền vững; được kích hoạt ở phiên tiếp theo |
| HashJack | "Tấn công URL fragment" | Payload ẩn trong URL fragment / chuỗi truy vấn nằm trong ngữ cảnh của agent nhưng không hiển thị trực quan |
| One-click hijack | "Nút bấm xấu" | Affordance hiển thị chứa payload tiếp theo mà agent thực thi |
| BrowseComp | "Benchmark tìm kiếm web" | Tìm kiếm các sự kiện cụ thể trên web mở; horizon quy mô phút |
| OSWorld | "Benchmark desktop" | Kiểm soát OS toàn diện; các tác vụ GUI nhiều bước |
| WebArena-Verified | "Benchmark tác vụ web đã sửa" | WebArena được chấm điểm lại của ServiceNow với tập con Hard |
| Ranh giới đọc/ghi | "Cổng tác dụng phụ" | Đọc không bao giờ gây hệ quả; ghi yêu cầu phê duyệt mới nếu nội dung nằm ngoài vùng tin cậy |

## Đọc thêm

- [OpenAI — Giới thiệu ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — hợp nhất Operator và deep research; SOTA trên BrowseComp.
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — dòng dõi Operator và kiến trúc trở thành ChatGPT agent.
- [Zhou và cộng sự — WebArena](https://webarena.dev/) — benchmark gốc.
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — bài báo về tập con đã sửa ICLR 2026.
- [Anthropic — Đo lường quyền tự chủ của agent trong thực tế](https://www.anthropic.com/research/measuring-agent-autonomy) — bao gồm thảo luận về bề mặt tấn công cho các agent computer-use.