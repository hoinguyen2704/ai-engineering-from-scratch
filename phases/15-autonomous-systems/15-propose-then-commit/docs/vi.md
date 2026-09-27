# Human-in-the-Loop: Propose-Then-Commit

> Sự đồng thuận năm 2026 về HITL rất cụ thể. Nó không phải là "tác nhân (agent) hỏi, người dùng nhấn Approve". Đó là propose-then-commit (đề xuất rồi mới thực thi): hành động được đề xuất sẽ được lưu trữ vào một kho lưu trữ bền vững (durable store) với một idempotency key; được hiển thị cho người đánh giá cùng với mục đích, nguồn gốc dữ liệu (data lineage), các quyền bị tác động, phạm vi ảnh hưởng (blast radius) và kế hoạch khôi phục (rollback plan); chỉ được thực thi sau khi có xác nhận đồng ý; và được kiểm chứng sau khi thực thi để đảm bảo tác dụng phụ đã thực sự xảy ra. `interrupt()` của LangGraph cùng với cơ chế checkpointing của PostgreSQL, `RequestInfoEvent` của Microsoft Agent Framework và `waitForApproval()` của Cloudflare đều triển khai cùng một mô hình này. Chế độ thất bại điển hình là phê duyệt kiểu "đóng dấu cao su" (rubber-stamp): nhấn "Approve?" mà không thực sự xem xét. Biện pháp giảm thiểu được ghi nhận là challenge-and-response (thách thức và phản hồi) với một danh sách kiểm tra (checklist) rõ ràng.

**Type:** Learn
**Languages:** Python (stdlib, propose-then-commit state machine with idempotency)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 14 (Tripwires)
**Time:** ~60 phút

## Vấn đề

Một agent thực hiện một hành động. Người dùng phải quyết định: phê duyệt hay không. Nếu quyết định diễn ra tức thì, đó có lẽ không phải là một sự xem xét kỹ lưỡng. Nếu quyết định có cấu trúc, nó sẽ chậm nhưng đáng tin cậy. Câu hỏi kỹ thuật ở đây là làm thế nào để biến việc xem xét có cấu trúc thành con đường ít trở ngại nhất.

Mô hình HITL thời kỳ 2023 là một lời nhắc đồng bộ: "Agent muốn gửi email đến X với nội dung Y — phê duyệt không?" Người dùng nhấn Approve. Mọi người cảm thấy hệ thống an toàn. Trên thực tế, bề mặt này dễ bị "đóng dấu cao su": người dùng phê duyệt nhanh, các phê duyệt không dự báo được nhiều điều, và khi agent gặp sự cố, dấu vết kiểm toán (audit trail) cho thấy một lịch sử phê duyệt dài mà người dùng không thể nhớ lại.

Mô hình năm 2026 — propose-then-commit — chuyển HITL sang một nền tảng bền vững, đính kèm siêu dữ liệu (metadata) có cấu trúc và yêu cầu cam kết xác nhận. Mọi SDK agent được quản lý đều cung cấp một phiên bản: `interrupt()` của LangGraph, `RequestInfoEvent` của Microsoft Agent Framework, `waitForApproval()` của Cloudflare. Tên API có thể khác nhau; nhưng hình thái thì không.

## Khái niệm

### Máy trạng thái propose-then-commit

1. **Propose (Đề xuất).** Agent tạo ra một hành động đề xuất. Được lưu trữ vào một kho bền vững (PostgreSQL, Redis, Durable Object). Bao gồm:
   - intent (mục đích: tại sao agent làm điều này)
   - data lineage (nguồn gốc dữ liệu: nguồn nào dẫn đến đề xuất này)
   - permissions touched (các quyền bị tác động: phạm vi/tệp/endpoint nào)
   - blast radius (phạm vi ảnh hưởng: trường hợp xấu nhất là gì)
   - rollback plan (kế hoạch khôi phục: nếu đã thực thi, làm thế nào để hoàn tác)
   - idempotency key (khóa tính lũy đẳng: duy nhất cho mỗi đề xuất; việc gửi lại sẽ trả về cùng một bản ghi)
2. **Surface (Hiển thị).** Người đánh giá xem đề xuất cùng với tất cả siêu dữ liệu. Người đánh giá phải là con người (không phải agent tự đánh giá chính nó).
3. **Commit (Cam kết).** Xác nhận đồng ý. Hành động được thực thi.
4. **Verify (Kiểm chứng).** Sau khi thực thi, tác dụng phụ được đọc lại và xác nhận. Nếu bước kiểm chứng thất bại, hệ thống đang ở trạng thái lỗi đã biết và cảnh báo sẽ được kích hoạt.

### Idempotency key (Khóa tính lũy đẳng)

Nếu không có idempotency key, việc thử lại sau một lỗi tạm thời có thể thực thi hành động đã phê duyệt hai lần. Ví dụ cụ thể: người dùng phê duyệt "chuyển 100 đô la từ A sang B." Mạng bị chập chờn. Workflow thử lại. Người dùng chỉ phê duyệt một lần nhưng việc chuyển tiền lại thực thi hai lần. Idempotency key gắn kết việc phê duyệt với một tác dụng phụ duy nhất; lần thực thi thứ hai sẽ là một no-op (không thực hiện gì cả).

Đây là cùng một mô hình tính lũy đẳng mà các API của Stripe và AWS sử dụng. Việc tái sử dụng nó cho các phê duyệt của agent được nêu rõ trong tài liệu của Microsoft Agent Framework.

### Độ bền vững: tại sao các phê duyệt tồn tại lâu hơn các tiến trình

Phòng chờ phê duyệt là một phần trạng thái mà agent không sở hữu. Workflow bị tạm dừng (Bài 12). Khi phê duyệt đến, workflow tiếp tục từ đúng điểm đó. Đây là lý do tại sao LangGraph kết hợp `interrupt()` với checkpointing của PostgreSQL thay vì chỉ lưu trạng thái trong bộ nhớ — một phê duyệt sau hai ngày vẫn tìm thấy workflow ở trạng thái nguyên vẹn.

### Phê duyệt "đóng dấu cao su" và biện pháp giảm thiểu challenge-and-response

Giao diện mặc định cho HITL (nút "Approve" / "Reject") tạo ra các phê duyệt nhanh mà không có sự xem xét thực sự. Biện pháp giảm thiểu được ghi nhận: một danh sách kiểm tra challenge-and-response yêu cầu các câu trả lời khẳng định cho các câu hỏi cụ thể trước khi nút Approve được kích hoạt. Hình thái cụ thể:

- "Bạn có hiểu tài nguyên nào bị tác động không? [ ]"
- "Bạn đã xác minh phạm vi ảnh hưởng có thể chấp nhận được chưa? [ ]"
- "Bạn có kế hoạch khôi phục nếu việc này thất bại không? [ ]"

Đây không phải là quan liêu vô nghĩa — mà là một cơ chế ép buộc. Người đánh giá không thể tích vào các ô này sẽ yêu cầu làm rõ (leo thang) hoặc từ chối (mặc định an toàn). Nghiên cứu về an toàn agent của Anthropic chỉ rõ HITL dựa trên danh sách kiểm tra là một biện pháp giảm thiểu cho các mô hình phê duyệt "đóng dấu cao su".

### Điều gì được coi là quan trọng (consequential)

Không phải mọi hành động đều cần propose-then-commit. Hướng dẫn năm 2026:

- **Các hành động quan trọng** (luôn cần HITL): ghi không thể đảo ngược, giao dịch tài chính, liên lạc ra bên ngoài, thay đổi cơ sở dữ liệu sản xuất, các thao tác hệ thống tệp mang tính phá hủy.
- **Các hành động có thể đảo ngược** (đôi khi cần HITL): chỉnh sửa tệp cục bộ, thay đổi môi trường staging, ghi có thể đảo ngược với kế hoạch khôi phục rõ ràng.
- **Đọc và kiểm tra** (không bao giờ cần HITL): đọc tệp, liệt kê tài nguyên, gọi API chỉ đọc.

### Kiểm chứng sau hành động

"Commit đã chạy" không có nghĩa là "tác dụng phụ đã xảy ra." Sự phân mảnh mạng và các điều kiện tranh chấp (race conditions) có thể tạo ra một workflow tưởng rằng đã thành công trong khi backend không lưu trữ được. Bước kiểm chứng sẽ đọc lại tài nguyên mục tiêu sau khi commit để xác nhận. Đây là cùng một mô hình như các giao dịch cơ sở dữ liệu với các mệnh đề `RETURNING` hoặc AWS `GetObject` sau `PutObject`.

### Đạo luật AI của EU, Điều 14

Điều 14 yêu cầu sự giám sát hiệu quả của con người đối với các hệ thống AI rủi ro cao tại EU. "Hiệu quả" không phải là để trang trí. Ngôn ngữ quy định loại trừ cụ thể các mô hình "đóng dấu cao su". Propose-then-commit với challenge-and-response là hình thái vượt qua được sự giám sát của Điều 14 trong tài liệu tuân thủ của Microsoft Agent Governance Toolkit.

```figure
mx-propose-then-commit
```

## Sử dụng

`code/main.py` triển khai một máy trạng thái propose-then-commit trong Python stdlib. Kho lưu trữ bền vững là một tệp JSON. Idempotency key là một hash của (thread_id, action_signature). Driver mô phỏng ba trường hợp: luồng phê duyệt sạch, thử lại sau lỗi tạm thời (không được thực thi hai lần) và mặc định "đóng dấu cao su" so với luồng challenge-and-response.

## Triển khai

`outputs/skill-hitl-design.md` xem xét một workflow HITL được đề xuất cho hình thái propose-then-commit và gắn cờ các siêu dữ liệu bị thiếu, tính lũy đẳng, kiểm chứng hoặc các lớp challenge-and-response.

## Bài tập

1. Chạy `code/main.py`. Xác nhận rằng việc thử lại một đề xuất đã được phê duyệt sử dụng bản ghi bền vững và không thực thi lại. Bây giờ hãy thay đổi idempotency key để bao gồm dấu thời gian và chứng minh việc thử lại sẽ thực thi hai lần.

2. Mở rộng bản ghi đề xuất với trường `rollback`. Mô phỏng một quá trình thực thi mà bước kiểm chứng thất bại. Cho thấy quá trình rollback tự động kích hoạt.

3. Đọc tài liệu `RequestInfoEvent` của Microsoft Agent Framework. Xác định một trường siêu dữ liệu mà API bao gồm nhưng engine mẫu còn thiếu. Thêm nó vào và giải thích nó bảo vệ chống lại điều gì.

4. Thiết kế một danh sách kiểm tra challenge-and-response cho một hành động cụ thể (ví dụ: "đăng lên tài khoản Twitter công khai"). Ba câu hỏi nào người đánh giá phải trả lời? Tại sao lại là ba câu hỏi đó?

5. Chọn một trường hợp mà lời nhắc "Approve?" đồng bộ là đủ (không cần kho lưu trữ bền vững). Giải thích tại sao và nêu tên loại rủi ro mà bạn đang chấp nhận.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|---|---|---|
| Propose-then-commit | "Phê duyệt hai giai đoạn" | Đề xuất được lưu trữ + cam kết xác nhận + kiểm chứng |
| Idempotency key | "Token an toàn khi thử lại" | Duy nhất cho mỗi đề xuất; lần thực thi thứ hai là no-op |
| Data lineage | "Nó đến từ đâu" | Nội dung nguồn cụ thể dẫn đến đề xuất |
| Blast radius | "Trường hợp xấu nhất" | Phạm vi ảnh hưởng nếu hành động gặp sự cố |
| Rubber-stamp | "Phê duyệt nhanh" | Nhấn "Approve" mà không xem xét thực sự |
| Challenge-and-response | "Danh sách kiểm tra ép buộc" | Người đánh giá phải xác nhận các câu hỏi cụ thể |
| RequestInfoEvent | "Nguyên thủy của MS Agent Framework" | Yêu cầu HITL bền vững với siêu dữ liệu có cấu trúc |
| `interrupt()` / `waitForApproval()` | "Nguyên thủy của Framework" | Các tương đương của LangGraph / Cloudflare với cùng hình thái |

## Đọc thêm

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`, phê duyệt bền vững.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` và Durable Objects.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL như một biện pháp giảm thiểu rủi ro dài hạn.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — cơ sở quy định cho các hệ thống rủi ro cao.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — khung hiến pháp về giám sát.