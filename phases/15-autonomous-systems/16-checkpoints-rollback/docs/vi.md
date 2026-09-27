# Checkpoints và Rollback

> Mọi quá trình chuyển đổi trạng thái đồ thị (graph-state transition) đều được lưu trữ bền vững. Khi một worker gặp sự cố, lease của nó hết hạn và một worker khác sẽ tiếp quản tại checkpoint gần nhất. Cloudflare Durable Objects duy trì trạng thái trong nhiều giờ hoặc nhiều tuần. Cơ chế Propose-then-commit (Bài 15) xác định kế hoạch rollback cho từng hành động. Việc xác minh sau hành động (post-action verification) giúp khép kín vòng lặp. Điều 14 của EU AI Act yêu cầu bắt buộc phải có sự giám sát hiệu quả của con người đối với các hệ thống rủi ro cao — trên thực tế, điều này có nghĩa là các checkpoint phải có khả năng truy vấn, các quy trình rollback phải được diễn tập và dấu vết kiểm toán (audit trail) phải tồn tại sau khi triển khai. Chế độ lỗi nghiêm trọng: nếu không có khóa idempotency (tính lũy đẳng) và kiểm tra điều kiện tiên quyết (precondition check), việc thử lại sau một lỗi tạm thời có thể thực thi hai lần một hành động đã được phê duyệt. Xác minh sau hành động chính là bước phát hiện ra điều này.

**Type:** Learn
**Languages:** Python (stdlib, checkpoint and rollback state machine)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 15 (Propose-then-commit)
**Time:** ~60 minutes

## Vấn đề

Durable execution (Bài 12) giúp một agent bị treo có thể tiếp tục hoạt động. Propose-then-commit (Bài 15) giúp một hành động đã được phê duyệt có thể kiểm toán được. Bài học này kết hợp cả hai: điều gì sẽ xảy ra khi một hành động đã được phê duyệt thực thi một phần, bị treo và sau đó tiếp tục? Khi nào thì rollback chạy và dựa trên trạng thái nào?

Các hệ thống thực tế xử lý việc này theo những cách khác nhau:

- **LangGraph** checkpoint mọi quá trình chuyển đổi trạng thái đồ thị vào PostgreSQL. Khi worker gặp sự cố, lease được giải phóng và một worker khác tiếp tục tại checkpoint gần nhất. Các workflow tạm dừng tại `interrupt()`, bản thân nó cũng được lưu trữ bền vững.
- **Cloudflare Durable Objects** duy trì trạng thái theo từng khóa trong nhiều giờ hoặc nhiều tuần. Đặt tính toán cùng vị trí với lưu trữ cho hành động đã được phê duyệt.
- **Microsoft Agent Framework** cung cấp các nguyên hàm `Checkpoint` trong workflow API; việc phát lại (replay) cộng với tính lũy đẳng (idempotency) sẽ xử lý các lần thử lại.

Trong mọi trường hợp, sự kết hợp thực sự hiệu quả là: khóa idempotency (ngăn chặn thực thi hai lần) + kiểm tra điều kiện tiên quyết (trạng thái vẫn đúng như những gì chúng ta đã phê duyệt) + xác minh sau hành động (tác dụng phụ thực sự đã xảy ra) + rollback khi xác minh thất bại.

## Khái niệm

### Mọi quá trình chuyển đổi đều được lưu trữ

Quá trình chuyển đổi trạng thái đồ thị là bất kỳ bước nào di chuyển workflow từ trạng thái này sang trạng thái khác. Các triển khai sơ khai chỉ lưu trữ tại các điểm commit cụ thể; các triển khai thực tế lưu trữ mọi quá trình chuyển đổi. Chi phí (thêm một vài lần ghi) là rất nhỏ so với lợi ích về độ tin cậy (việc phát lại có thể bắt đầu ở bất cứ đâu, khôi phục lease chính xác).

### Khôi phục Lease

Khi một worker gặp sự cố, workflow không bị mất; lease (một yêu cầu tồn tại ngắn hạn rằng worker này đang thực thi một lượt chạy) chỉ đơn giản là hết hạn. Một worker khác sẽ lấy checkpoint gần nhất và tiếp tục. Cơ chế lease là thứ cho phép các hệ thống thực tế tồn tại qua các đợt triển khai luân phiên (rolling deploys) mà không làm mất công việc đang thực hiện.

### Idempotency và các điều kiện tiên quyết

Chỉ riêng idempotency là không đủ. Hãy xem xét: một workflow được phê duyệt để "chuyển $100 from A to B when balance > $1000". Workflow được commit, bị treo giữa chừng khi thực thi và sau đó tiếp tục. Nếu chỉ kiểm tra khóa idempotency và quá trình thực thi tiếp tục, việc chuyển tiền sẽ chạy một lần (đúng). Nhưng hãy xem xét rằng giữa lúc treo và lúc tiếp tục, số dư của A giảm xuống còn $500 thông qua một workflow khác. Kiểm tra idempotency vẫn vượt qua; nhưng điều kiện tiên quyết thì không. Nếu không có kiểm tra điều kiện tiên quyết, chúng ta sẽ gây ra tình trạng thấu chi.

Mọi hành động quan trọng đều cần cả hai:

- **Khóa Idempotency**: ngăn chặn thực thi hai lần.
- **Kiểm tra điều kiện tiên quyết**: xác nhận trạng thái vẫn nhất quán với những gì đã được phê duyệt.

### Xác minh sau hành động

"Công cụ trả về 200" không phải là xác minh. Xác minh thực tế là đọc lại trạng thái mục tiêu và xác nhận tác dụng phụ thực sự đã xảy ra. Các mẫu hình:

- Cập nhật cơ sở dữ liệu: `UPDATE ... RETURNING *` sau đó khẳng định hàng được trả về khớp với trạng thái dự định.
- Gửi email: kiểm tra thư mục đã gửi để tìm ID tin nhắn sau khi gửi.
- Ghi tệp: đọc lại tệp và băm (hash) nó.
- Gọi API: thực hiện `GET` trên tài nguyên mục tiêu.

Nếu xác minh thất bại, workflow đang ở trạng thái lỗi đã biết. Rollback sẽ được kích hoạt.

### Kế hoạch Rollback

Mọi hành động quan trọng trong propose-then-commit (Bài 15) đều mang theo một kế hoạch rollback. Các loại:

- **In-band rollback**: đảo ngược tác dụng phụ trực tiếp (`DELETE` sau `INSERT`, `Send-correction-email` sau khi gửi).
- **Giao dịch bù trừ (Compensating transaction)**: một hành động mới giúp vô hiệu hóa hành động gốc (mẫu hình SAGA tiêu chuẩn).
- **Out-of-band rollback**: cảnh báo cho con người, tạm dừng workflow, để lại trạng thái lỗi để điều tra.

Rollback dạng no-op ("chúng tôi không thể hoàn tác việc này") phải được nêu rõ trong đề xuất. Các hành động không có rollback yêu cầu HITL (Human-in-the-loop) mạnh mẽ hơn tại thời điểm commit (thử thách và phản hồi trong Bài 15).

### Đọc hiểu Điều 14 của EU AI Act trong vận hành

Điều 14 yêu cầu "sự giám sát hiệu quả của con người" đối với các hệ thống rủi ro cao. Về mặt vận hành, những người triển khai hiểu điều đó là:

- Các checkpoint có thể truy vấn được bởi kiểm toán viên.
- Các quy trình rollback được diễn tập (kiểm tra end-to-end ít nhất một lần).
- Dấu vết kiểm toán tồn tại sau khi triển khai (backend checkpoint không phải là tạm thời).
- Các xác minh thất bại phải được cảnh báo, không chỉ ghi nhật ký âm thầm.

Một workflow bị treo giữa chừng khi commit, tiếp tục và hoàn thành tác dụng phụ mà không có lộ trình xác minh + rollback sẽ không vượt qua bài kiểm tra Điều 14.

### Chế độ lỗi nghiêm trọng: thực thi hai lần

Sự cố phổ biến nhất trong không gian này:

1. Hành động được phê duyệt, khóa idempotency k.
2. Commit bắt đầu, thực thi, trả về 200.
3. Workflow bị treo trước khi lưu trạng thái "đã commit".
4. Workflow tiếp tục; thấy "đã phê duyệt nhưng chưa commit"; thực thi lại.
5. Tác dụng phụ kích hoạt hai lần.

Cách giảm thiểu: lưu trữ ý định "đang thực hiện" (in-flight) trước khi thực thi, thực thi với khóa idempotency, sau đó chỉ đánh dấu "đã commit" sau khi xác minh sau hành động thành công. Nếu hành động kích hoạt và việc ghi trạng thái thất bại, bạn biết cần phải xác minh và (nếu cần) kích hoạt lại. Nếu việc ghi trạng thái thành công và hành động thất bại, bạn xác minh và kích hoạt chính xác một lần thông qua lộ trình khôi phục.

```figure
checkpoint-replay
```

## Sử dụng

`code/main.py` triển khai một workflow có checkpoint với idempotency, điều kiện tiên quyết, xác minh và rollback. Driver mô phỏng bốn kịch bản: chạy sạch, thử lại sau khi treo (idempotency bắt được), điều kiện tiên quyết thất bại (workflow hủy bỏ mà không kích hoạt), xác minh thất bại (rollback kích hoạt).

## Triển khai

`outputs/skill-rollback-rehearsal.md` thiết kế một bài kiểm tra diễn tập rollback cho một workflow đã đề xuất và kiểm toán backend checkpoint để đảm bảo tính bền vững của dấu vết kiểm toán.

## Bài tập

1. Chạy `code/main.py`. Xác minh bốn kịch bản. Đối với trường hợp treo trong khi commit, hãy xác nhận hành động kích hoạt chính xác một lần qua các lần thử lại.

2. Sửa đổi mẫu hình "đánh dấu hoàn thành trước, sau đó mới thực hiện" để việc ghi trạng thái kích hoạt sau hành động. Chạy lại kịch bản treo. Đo lường xem có bao nhiêu hành động trùng lặp được kích hoạt.

3. Thiết kế kế hoạch rollback cho một hành động sản xuất cụ thể (ví dụ: "đăng lên kênh Slack"). Phân loại là in-band, bù trừ hoặc out-of-band. Giải thích lý do cho lựa chọn đó.

4. Chọn một workflow bạn biết. Xác định mọi quá trình chuyển đổi trạng thái. Đánh dấu mỗi cái với yêu cầu về độ bền (lưu trữ / không lưu trữ). Đếm những cái bạn hiện không lưu trữ.

5. Kiểm tra diễn tập rollback: thiết kế một bài kiểm tra end-to-end chạy một workflow thực tế, làm treo nó và xác nhận lộ trình rollback kích hoạt. Bài kiểm tra khẳng định điều gì?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| Checkpoint | "Điểm lưu" | Mọi quá trình chuyển đổi trạng thái đồ thị được lưu vào kho lưu trữ bền vững |
| Lease | "Yêu cầu worker" | Yêu cầu tồn tại ngắn hạn rằng một worker đang thực thi lượt chạy; hết hạn khi treo |
| Precondition | "Cổng trạng thái" | Khẳng định rằng trạng thái vẫn nhất quán với hành động đã phê duyệt |
| Post-action verify | "Kiểm tra đọc lại" | Xác nhận tác dụng phụ thực sự đã xảy ra trong hệ thống mục tiêu |
| In-band rollback | "Hoàn tác trực tiếp" | Đảo ngược tác dụng phụ bằng thao tác nghịch đảo |
| Compensating transaction | "Hoàn tác SAGA" | Một hành động mới giúp vô hiệu hóa hành động gốc |
| Mark-as-done-first | "Thứ tự ghi trạng thái" | Lưu trạng thái đã commit trước khi trả về từ commit |
| Article 14 | "Giám sát con người EU AI Act" | Vận hành: checkpoint có thể truy vấn, rollback được diễn tập, dấu vết kiểm toán |

## Đọc thêm

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — các nguyên hàm checkpoint và khôi phục lease.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — Durable Objects như một nền tảng trạng thái.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — cơ sở quy định.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — khung độ tin cậy cho các workflow dài hạn.
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — hình dạng workflow cho Claude Code Routines.