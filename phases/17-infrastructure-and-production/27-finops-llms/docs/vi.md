# FinOps cho LLM — Kinh tế đơn vị và Phân bổ đa người thuê (Multi-Tenant)

> FinOps truyền thống không còn phù hợp với chi phí LLM. Chi phí ở đây là các giao dịch token, không phải thời gian hoạt động của tài nguyên. Các thẻ (tags) không khớp — một lệnh gọi API là một giao dịch, không phải một tài sản. Các quyết định kỹ thuật (thiết kế prompt, cửa sổ ngữ cảnh, độ dài đầu ra) chính là các quyết định tài chính. Sách lược năm 2026 có ba chiều phân bổ cần được thiết lập ngay từ ngày đầu: theo người dùng (`user_id`) để định giá theo chỗ ngồi và mở rộng, theo tác vụ (`task_id` + `route`) để xác định chi phí bề mặt sản phẩm và ưu tiên, theo người thuê (`tenant_id`) để tính kinh tế đơn vị và gia hạn. Bốn lớp token — prompt, tool, memory, response — nếu gộp chung một giỏ sẽ che giấu chi phí thực. Thang thực thi cho các sản phẩm đa người thuê: giới hạn tốc độ (rate limit) theo người thuê (2-3 lần đỉnh dự kiến, trả về 429 + retry-after rõ ràng); hạn mức chi tiêu hàng ngày (1.5-3 lần trần hợp đồng; kích hoạt thắt chặt tốc độ + cảnh báo); công tắc ngắt (kill switch) khi z-score chi tiêu > 4 (tự động tạm dừng + thông báo cho on-call). Các mô hình phân bổ: gắn thẻ và tổng hợp, kết nối telemetry (trace-ID → thanh toán; độ chính xác cao nhất), lấy mẫu và ngoại suy, phân bổ dựa trên mô hình, event-sourced, streaming thời gian thực. Chỉ số đơn vị: chi phí cho mỗi truy vấn được giải quyết, chi phí cho mỗi artifact được tạo — không phải $/M token. Gắn thẻ hồi tố luôn thiếu sót; hãy thực hiện tại thời điểm tạo yêu cầu.

**Type:** Learn
**Languages:** Python (stdlib, toy cost-attribution simulator with kill switch)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 14 (Caching)
**Time:** ~60 phút

## Mục tiêu học tập

- Giải thích lý do tại sao FinOps truyền thống (thẻ + tầng) không hiệu quả với chi phí LLM và nêu tên ba chiều phân bổ mới.
- Liệt kê bốn lớp token (prompt, tool, memory, response) và lý do tại sao việc thanh toán gộp chung một giỏ lại che giấu chi phí.
- Thiết kế thang thực thi (giới hạn tốc độ → hạn mức chi tiêu → công tắc ngắt) cho sản phẩm đa người thuê.
- Chọn chỉ số đơn vị (chi phí cho mỗi truy vấn/artifact được giải quyết) thay vì $/M token.

## Vấn đề

Hóa đơn của bạn là 40.000 USD. Bạn không biết:
- Người thuê nào đã chi tiêu số tiền đó.
- Tính năng sản phẩm nào đã thúc đẩy chi phí đó.
- Liệu có người dùng cá nhân nào đang lạm dụng hay không.
- Liệu sự phình to của prompt, các lệnh gọi tool, hay sự khuếch đại bộ nhớ là nguyên nhân gây ra chi phí.

Việc gắn thẻ và tổng hợp phía nhà cung cấp hoạt động tốt với tài nguyên đám mây (EC2, S3) nơi các thẻ lan truyền đến các dòng chi phí. Các lệnh gọi API LLM không tự động gắn thẻ — bạn phải đóng dấu người dùng/tác vụ/người thuê tại điểm gọi và duy trì nó. Phân bổ hồi tố luôn bỏ sót các trường hợp biên.

## Khái niệm

### Ba chiều phân bổ

**Theo người dùng** (`user_id`): ai đang tiêu tốn bao nhiêu. Thúc đẩy định giá theo chỗ ngồi, các cuộc thảo luận mở rộng, xác định người dùng quyền lực (power users).

**Theo tác vụ** (`task_id` + `route`): bề mặt sản phẩm nào tốn bao nhiêu. Thúc đẩy ưu tiên tính năng, các quyết định loại bỏ các tính năng đắt đỏ.

**Theo người thuê** (`tenant_id`): khách hàng nào mang lại lợi nhuận. Thúc đẩy kinh tế đơn vị, định giá gia hạn, ngưỡng tầng dịch vụ.

Hãy thiết lập cả ba tại điểm gọi ngay từ ngày đầu. Phân bổ hồi tố luôn tệ hơn.

### Bốn lớp token

| Lớp | Ví dụ | % điển hình của tổng số |
|-------|---------|---------------------|
| Prompt | system + user input | 40-60% |
| Tool | kết quả tool-call được phản hồi lại | 20-40% (tác vụ agent) |
| Memory | hội thoại trước đó / tài liệu đã truy xuất | 10-30% |
| Response | đầu ra của mô hình | 10-30% |

Việc gộp cả bốn lớp này lại khiến việc tối ưu hóa trở nên mù quáng. Hãy tách chúng ra trong lược đồ phân bổ của bạn.

### Thang thực thi

1. **Giới hạn tốc độ (Rate limit)** theo người thuê. 2-3 lần đỉnh dự kiến. Trả về 429 với `Retry-After`. Người thuê thấy sự hạn chế; không có hóa đơn bất ngờ.

2. **Hạn mức chi tiêu hàng ngày** theo người thuê. 1.5-3 lần trần hợp đồng. Kích hoạt: thắt chặt giới hạn tốc độ + cảnh báo cho bộ phận chăm sóc khách hàng.

3. **Công tắc ngắt (Kill switch)** khi z-score chi tiêu > 4 so với mức cơ sở của người thuê. Tự động tạm dừng người thuê; thông báo cho on-call; leo thang lên ops + CS.

### Các mô hình phân bổ

- **Gắn thẻ và tổng hợp**: đóng dấu metadata vào header; tổng hợp sau. Đơn giản; thô sơ.
- **Kết nối telemetry**: kết nối các trace với thanh toán thông qua trace ID. Độ chính xác cao nhất. Điều mà các đội ngũ chuyên nghiệp thực hiện.
- **Lấy mẫu + ngoại suy**: lấy mẫu 5-10%, nhân lên. Hiệu quả về chi phí cho việc ước tính chi tiêu; bỏ sót các trường hợp đuôi.
- **Phân bổ dựa trên mô hình**: hồi quy để suy luận trình điều khiển chi phí. Dành cho dữ liệu cũ không có thẻ.
- **Event-sourced**: chi phí dưới dạng sự kiện trong luồng (Kafka / Kinesis). Thời gian thực.
- **Streaming thời gian thực**: cập nhật bảng điều khiển dưới một giây.

### Chi phí cho mỗi X là chỉ số đơn vị

$/M token là ngôn ngữ của nhà cung cấp. Chỉ số sản phẩm:

- Chi phí cho mỗi ticket hỗ trợ được giải quyết.
- Chi phí cho mỗi bài viết được tạo.
- Chi phí cho mỗi tác vụ agent thành công.
- Chi phí cho mỗi phút phiên người dùng.

Gắn chi phí với kết quả sản phẩm. Nếu không, việc tối ưu hóa sẽ không có điểm tựa.

### Hình dạng trace phân bổ chi phí

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

Phát ra trên mỗi lệnh gọi. Lưu trữ trong data lake. Tổng hợp theo từng chiều. Stack quan sát Phase 17 · 13 là nơi chứa dữ liệu này.

### Stack tiết kiệm tích lũy

Stack: cache + batch + route + gateway. Với cả bốn:
- Cache L2 (Phase 17 · 14): đầu vào rẻ hơn ~10 lần.
- Batch (Phase 17 · 15): giảm 50%.
- Định tuyến đến mô hình rẻ hơn (Phase 17 · 16): giảm 60% chi phí.
- Hiệu quả gateway (Phase 17 · 19): dự phòng + thử lại.

Trường hợp tốt nhất: ~5-10% so với mức cơ sở ban đầu. Hầu hết các đội ngũ chỉ áp dụng 2-3 đòn bẩy; rất ít người áp dụng cả bốn.

### Những con số bạn cần nhớ

- Các chiều phân bổ: theo người dùng, theo tác vụ, theo người thuê.
- Bốn lớp token: prompt, tool, memory, response.
- Công tắc ngắt: z-score chi tiêu > 4.
- Chỉ số đơn vị: chi phí cho mỗi truy vấn được giải quyết, không phải $/M token.
- Tối ưu hóa tích lũy: có thể đạt ~5-10% mức cơ sở.

```figure
i4-spend-ladder
```

## Sử dụng

`code/main.py` mô phỏng một dịch vụ LLM đa người thuê với thang thực thi ba tầng. Chèn một người thuê lạm dụng và chứng minh công tắc ngắt hoạt động.

## Triển khai

Bài học này tạo ra `outputs/skill-finops-plan.md`. Dựa trên sản phẩm và quy mô, thiết kế lược đồ phân bổ và thang thực thi.

## Bài tập

1. Chạy `code/main.py`. Tại z-score nào thì công tắc ngắt kích hoạt? Làm thế nào để bạn chọn ngưỡng đó?
2. Thiết kế bảng điều khiển chi phí theo người thuê, theo tác vụ. 5 chế độ xem bạn xây dựng đầu tiên là gì?
3. Người thuê lớn nhất của bạn đang có kinh tế đơn vị âm. Đề xuất ba biện pháp can thiệp theo thứ tự tác động đến khách hàng.
4. Tính chi phí cho mỗi ticket được giải quyết cho một sản phẩm hỗ trợ: 3M token/ticket, ~800 ticket/ngày, mức giá cache GPT-5.
5. Tranh luận xem liệu gắn thẻ hồi tố có bao giờ hiệu quả không. Khi nào thì nó có thể chấp nhận được?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Phân bổ theo người dùng | "chi phí cấp người dùng" | `user_id` được đóng dấu trên mỗi lệnh gọi |
| Phân bổ theo tác vụ | "chi phí tính năng" | `task_id` + `route` xác định bề mặt sản phẩm |
| Phân bổ theo người thuê | "chi phí khách hàng" | `tenant_id`; thúc đẩy kinh tế đơn vị |
| Bốn lớp token | "các lớp chi phí" | prompt + tool + memory + response |
| Giới hạn tốc độ | "bộ bảo vệ 429" | Trần giới hạn theo người thuê tại gateway |
| Hạn mức chi tiêu hàng ngày | "trần hàng ngày" | Ngân sách phạm vi người thuê với cảnh báo |
| Công tắc ngắt | "tự động tạm dừng" | z-score chi tiêu > 4 kích hoạt tự động đình chỉ |
| Chi phí cho mỗi đơn vị giải quyết | "chỉ số đơn vị sản phẩm" | Chi phí gắn với kết quả sản phẩm, không phải token |
| Kết nối telemetry | "trace-to-billing" | Mô hình phân bổ độ chính xác cao nhất |
| Tối ưu hóa tích lũy | "cache+batch+route+gateway" | Tiết kiệm tích lũy xuống ~5-10% mức cơ sở |

## Đọc thêm

- [FinOps Foundation — Tổng quan FinOps cho AI](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Hướng dẫn chi phí cho mỗi đơn vị 2026](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — Phân bổ chi phí LLM Agent 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Quản lý LLM trong Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)