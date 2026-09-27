# Action Budgets, Iteration Caps, and Cost Governors

> Chi phí LLM hàng tháng của một đại lý thương mại điện tử quy mô trung bình đã tăng vọt từ $1,200 to $4.800 sau khi nhóm của họ kích hoạt kỹ năng "theo dõi đơn hàng". Đó không phải là lỗi định giá. Đó là một đại lý đã tìm thấy một vòng lặp mới và liên tục chi tiêu trong đó. Microsoft's Agent Governance Toolkit (ngày 2 tháng 4 năm 2026) đã hệ thống hóa các biện pháp phòng thủ chống lại loại lỗi này: `max_tokens` trên mỗi yêu cầu, ngân sách token và đô la trên mỗi tác vụ, giới hạn theo ngày/tháng, giới hạn số lần lặp, định tuyến mô hình theo tầng, prompt caching, context windowing, các điểm kiểm soát HITL (Human-in-the-loop) đối với các hành động đắt đỏ, và công tắc ngắt (kill switch) khi vi phạm ngân sách. Anthropic's Claude Code Agent SDK cũng cung cấp các nguyên tắc tương tự dưới những tên gọi khác. Các giới hạn về tốc độ tài chính (financial velocity limits) — ví dụ: cắt quyền truy cập nếu chi tiêu >$50 trong 10 phút — giúp phát hiện các vòng lặp nhanh hơn so với các giới hạn hàng tháng.

**Type:** Learn
**Languages:** Python (stdlib, layered cost-governor simulator)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 12 (Durable execution)
**Time:** ~60 minutes

## Vấn đề

Các đại lý tự hành tiêu tốn tiền thật trong mỗi lượt thực hiện. Phản hồi tồi của một chatbot chỉ là một câu trả lời tệ; vòng lặp tồi của một đại lý là một hóa đơn. Thuật ngữ được ngành công nghiệp ghi nhận cho chế độ lỗi này là "Denial of Wallet" (Từ chối ví tiền) — đại lý tiếp tục suy luận, tiếp tục gọi công cụ, tiếp tục tính phí, và không có gì ngăn cản nó vì không có gì được thiết kế để làm việc đó.

Giải pháp không nằm ở một con số duy nhất. Đó là một chồng các giới hạn ở các quy mô thời gian và độ chi tiết khác nhau: trên mỗi yêu cầu, mỗi tác vụ, mỗi giờ, mỗi ngày, mỗi tháng. Một chồng giới hạn được thiết kế tốt sẽ phát hiện một vòng lặp chạy quá mức trong vài phút, một sự rò rỉ chậm trong vài giờ và một bản phát hành lỗi trong một ngày. Chính chồng giới hạn này giúp duy trì ngân sách khi đại lý hoạt động dài hạn và tự hành.

Đây là một bài học kỹ thuật: toán học thì tầm thường, nhưng kỷ luật mới là nơi các đội ngũ thất bại. Danh sách các giới hạn dưới đây đều được nêu tên trong Microsoft Agent Governance Toolkit hoặc tài liệu của Anthropic Claude Code Agent SDK.

## Khái niệm

### Chồng quản trị chi phí (cost-governor stack)

1. **`max_tokens` trên mỗi yêu cầu.** Đơn giản. Ngăn chặn bất kỳ lệnh gọi nào tạo ra một completion không giới hạn.
2. **Ngân sách token trên mỗi tác vụ.** Trong toàn bộ quá trình chạy, không vượt quá N token. Dừng cứng khi đạt giới hạn.
3. **Ngân sách đô la trên mỗi tác vụ.** Tương tự như token nhưng tính bằng tiền tệ. `max_budget_usd` trong Claude Code.
4. **Giới hạn số lần gọi công cụ.** Không quá N lần gọi `WebFetch`, N lần gọi `shell_exec`, v.v.
5. **Giới hạn số lần lặp (`max_turns`).** Tổng số lần lặp của vòng lặp đại lý; ngăn chặn các vòng lặp suy luận vô hạn.
6. **Giới hạn theo phút / giờ / ngày / tháng.** Các cửa sổ trượt (rolling windows). Phát hiện rò rỉ ở các quy mô thời gian khác nhau.
7. **Giới hạn tốc độ tài chính.** Ví dụ: "nếu chi tiêu vượt quá $50 trong 10 phút, hãy cắt quyền truy cập". Phát hiện sự tiêu tốn dựa trên vòng lặp trước khi các giới hạn hàng tháng được kích hoạt.
8. **Định tuyến mô hình theo tầng.** Mặc định sử dụng mô hình nhỏ hơn; chỉ nâng cấp lên mô hình lớn hơn khi bộ phân loại đánh giá tác vụ đó là cần thiết.
9. **Prompt caching.** System prompt và ngữ cảnh ổn định được lưu trữ trong bộ nhớ đệm của nhà cung cấp; chi phí token khi gửi lại gần như bằng không.
10. **Context windowing.** Nén / tóm tắt để giữ ngữ cảnh hoạt động dưới một ngưỡng; giảm trực tiếp chi phí token.
11. **Điểm kiểm soát HITL cho các hành động đắt đỏ.** Trước một hành động được biết là đắt đỏ (gọi công cụ dài, tải xuống lớn, nâng cấp mô hình tốn kém), yêu cầu sự xác nhận của con người.
12. **Công tắc ngắt khi vi phạm ngân sách.** Phiên làm việc bị hủy bỏ khi bất kỳ giới hạn nào bị vi phạm. Giới hạn được ghi lại; yêu cầu một quy trình kích hoạt lại riêng biệt.

### Tại sao cần cả chồng giới hạn thay vì một giới hạn duy nhất

Một giới hạn hàng tháng chỉ phát hiện ra một đại lý chạy quá mức sau khi ví tiền đã cạn. Một giới hạn trên mỗi yêu cầu không phát hiện được gì ở cấp độ phiên. Các chế độ lỗi khác nhau đòi hỏi các quy mô thời gian khác nhau:

- **Vòng lặp chạy quá mức** (đại lý bị kẹt trong vòng lặp thử lại 5 giây): được phát hiện bởi giới hạn tốc độ.
- **Rò rỉ chậm** (đại lý thực hiện công việc gấp ~2 lần dự kiến trên mỗi tác vụ): được phát hiện bởi giới hạn hàng ngày.
- **Bản phát hành lỗi** (phiên bản mới sử dụng gấp 5 lần token): được phát hiện bởi giới hạn hàng tuần / hàng tháng.
- **Sự gia tăng hợp pháp** (nhu cầu thực, không phải lỗi): được phát hiện bởi giới hạn giờ / ngày với nhật ký rõ ràng.

### Bề mặt ngân sách của harness

Claude Code Agent SDK cung cấp (tài liệu công khai):

- `max_turns` — giới hạn số lần lặp.
- `max_budget_usd` — giới hạn đô la; phiên làm việc bị hủy bỏ khi vi phạm.
- `allowed_tools` / `disallowed_tools` — danh sách cho phép và danh sách chặn công cụ.
- Các điểm hook trước khi sử dụng công cụ để kế toán chi phí tùy chỉnh.

Kết hợp với thang đo chế độ quyền hạn (Bài học 10). Một phiên `autoMode` không có `max_budget_usd` là sự tự hành không được quản lý. Anthropic khẳng định rõ ràng rằng Auto Mode yêu cầu các biện pháp kiểm soát ngân sách; bộ phân loại là độc lập với chi phí.

### EU AI Act, OWASP Agentic Top 10

Microsoft's Agent Governance Toolkit bao gồm các yêu cầu của OWASP Agentic Top 10 và EU AI Act Điều 14 (sự giám sát của con người). Để triển khai sản xuất tại EU, việc ghi nhật ký và thực thi giới hạn là bắt buộc.

### Trường hợp $1,200 → $4.800 đã quan sát

Trường hợp thực tế trong tài liệu của Microsoft: một đại lý thương mại điện tử có chi phí hàng tháng tăng gấp ba lần sau khi một công cụ mới được thêm vào. Công cụ này cho phép đại lý thăm dò trạng thái đơn hàng trong mỗi phiên. Không có phát hiện vòng lặp. Không có giới hạn trên mỗi công cụ. Không có cảnh báo về sự tăng trưởng theo tuần. Giải pháp là giới hạn trên mỗi công cụ cộng với cảnh báo tăng trưởng hàng ngày. Đây là một khuôn mẫu: mỗi bề mặt công cụ mới là một vòng lặp tiềm năng mới; mỗi công cụ mới cần giới hạn riêng và cảnh báo riêng.

```figure
cost-governor-stack
```

## Sử dụng

`code/main.py` mô phỏng một quá trình chạy của đại lý với và không có chồng quản trị chi phí theo lớp. Đại lý mô phỏng trôi vào vòng lặp thăm dò sau một vài lượt; chồng giới hạn theo lớp phát hiện nó trong cửa sổ tốc độ, trong khi một giới hạn hàng tháng duy nhất sẽ không kích hoạt cho đến nhiều ngày sau đó.

## Triển khai

`outputs/skill-agent-budget-audit.md` kiểm tra chồng quản trị chi phí của một bản triển khai đại lý được đề xuất và gắn cờ các lớp còn thiếu.

## Bài tập

1. Chạy `code/main.py`. Xác nhận giới hạn tốc độ kích hoạt trước giới hạn số lần lặp trên quỹ đạo vòng lặp thăm dò. Bây giờ hãy tắt giới hạn tốc độ và đo lường xem đại lý "chi tiêu" bao nhiêu trước khi giới hạn số lần lặp bắt kịp nó.

2. Thiết kế một bộ giới hạn trên mỗi công cụ cho một đại lý trình duyệt (Bài học 11). Công cụ nào cần giới hạn chặt chẽ nhất? Công cụ nào có thể chạy không giới hạn mà không gặp rủi ro?

3. Đọc tài liệu Microsoft Agent Governance Toolkit. Liệt kê mọi loại giới hạn mà bộ công cụ nêu tên. Ánh xạ từng loại với một trong các chế độ lỗi (vòng lặp chạy quá mức, rò rỉ chậm, bản phát hành lỗi, sự gia tăng đột biến).

4. Định giá cho một lần chạy không giám sát qua đêm cho một tác vụ thực tế (ví dụ: "phân loại 50 vấn đề trong một repo"). Đặt `max_budget_usd` ở mức gấp 2 lần ước tính của bạn. Hãy giải thích lý do cho con số gấp 2 đó.

5. `max_budget_usd` của Claude Code kích hoạt dựa trên chi phí tổng hợp của phiên. Thiết kế một giới hạn tốc độ bổ sung mà bạn sẽ thực thi từ bên ngoài. Điều gì kích hoạt việc cắt quyền truy cập, và việc kích hoạt lại trông như thế nào?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói gì | Ý nghĩa thực sự |
|---|---|---|
| Denial of Wallet | "Hóa đơn chạy quá mức" | Vòng lặp đại lý tạo ra chi tiêu mà không có giới hạn để dừng lại |
| max_tokens | "Giới hạn trên mỗi yêu cầu" | Trần cho kích thước của một completion đơn lẻ |
| max_turns | "Giới hạn số lần lặp" | Trần cho số lần lặp của vòng lặp đại lý trong một phiên |
| max_budget_usd | "Công tắc ngắt đô la" | Giới hạn chi phí phiên; hủy bỏ khi vi phạm |
| Velocity limit | "Giới hạn tốc độ" | Giới hạn chi tiêu trên mỗi cửa sổ thời gian ngắn (ví dụ: $50 / 10 phút) |
| Tiered routing | "Mô hình nhỏ trước" | Mặc định mô hình rẻ; chỉ nâng cấp khi bộ phân loại yêu cầu |
| Prompt caching | "Cached system prompt" | Bộ nhớ đệm phía nhà cung cấp giảm chi phí token gửi lại xuống gần bằng không |
| HITL checkpoint | "Cổng phê duyệt của con người" | Yêu cầu sự xác nhận của con người trước hành động đắt đỏ |

## Đọc thêm

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`, `max_budget_usd`, danh sách cho phép công cụ.
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — các điểm kiểm soát quản trị chi phí.
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — các biện pháp kiểm soát chi phí phía nhà cung cấp.
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) — cơ chế caching.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — hồ sơ chi phí cho các đại lý dài hạn.