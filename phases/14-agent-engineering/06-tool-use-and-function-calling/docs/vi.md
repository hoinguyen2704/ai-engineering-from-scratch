# Tool Use và Function Calling

> Toolformer (Schick và cộng sự, 2023) đã khởi đầu cho việc tự chú thích công cụ (self-supervised tool annotation). Berkeley Function Calling Leaderboard V4 (Patil và cộng sự, 2025) thiết lập tiêu chuẩn cho năm 2026: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination. Single-turn đã được giải quyết. Bộ nhớ (memory), ra quyết định động (dynamic decision-making) và chuỗi công cụ dài hạn (long-horizon tool chains) thì chưa.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 phút

## Mục tiêu học tập

- Giải thích tín hiệu huấn luyện tự giám sát của Toolformer: chỉ giữ lại các chú thích công cụ khi việc thực thi làm giảm loss của token tiếp theo.
- Liệt kê năm danh mục đánh giá của BFCL V4 và ý nghĩa của từng danh mục.
- Triển khai một registry công cụ sử dụng stdlib với kiểm tra schema, ép kiểu đối số (argument coercion) và sandbox thực thi.
- Chẩn đoán ba vấn đề mở của năm 2026: chuỗi công cụ dài hạn, ra quyết định động và bộ nhớ.

## Vấn đề

Việc sử dụng công cụ thời kỳ đầu đặt câu hỏi: liệu model có thể dự đoán đúng một function call? Việc sử dụng công cụ hiện đại đặt câu hỏi: liệu model có thể xâu chuỗi các công cụ qua 40 bước, với bộ nhớ, với khả năng quan sát một phần, với khả năng phục hồi sau lỗi công cụ, mà không ảo tưởng (hallucinate) ra các công cụ không tồn tại?

Toolformer đã thiết lập nền tảng: các model có thể học cách gọi công cụ bằng phương pháp tự giám sát. BFCL V4 xác định mục tiêu đánh giá cho năm 2026. Khoảng cách giữa chúng chính là không gian mà các agent sản xuất (production agents) đang tồn tại.

## Khái niệm

### Toolformer (Schick và cộng sự, NeurIPS 2023)

Ý tưởng: để model tự chú thích tập dữ liệu tiền huấn luyện của chính nó với các ứng viên API call. Với mỗi ứng viên, hãy thực thi nó. Chỉ giữ lại chú thích nếu việc bao gồm kết quả công cụ làm giảm loss trên token tiếp theo. Fine-tune trên tập dữ liệu đã lọc.

Các công cụ được bao phủ: máy tính, hệ thống QA, công cụ tìm kiếm, trình dịch, lịch. Tín hiệu tự giám sát hoàn toàn dựa trên việc công cụ có giúp dự đoán văn bản hay không — không cần nhãn từ con người.

Kết quả về quy mô: việc sử dụng công cụ xuất hiện ở quy mô lớn. Các model nhỏ hơn bị ảnh hưởng tiêu cực bởi các chú thích công cụ; các model lớn hơn thì được hưởng lợi. Đây là lý do tại sao các model tiên phong năm 2026 có khả năng sử dụng công cụ mạnh mẽ, trong khi hầu hết các model 7B cần fine-tune chuyên biệt để đạt độ tin cậy.

### Berkeley Function Calling Leaderboard V4 (Patil và cộng sự, ICML 2025)

BFCL là tiêu chuẩn đánh giá thực tế cho năm 2026. Cấu trúc của V4:

- **Agentic (40%)** — các quỹ đạo agent đầy đủ: bộ nhớ, multi-turn, các quyết định động.
- **Multi-Turn (30%)** — các cuộc hội thoại tương tác với chuỗi công cụ.
- **Live (10%)** — các prompt thực tế do người dùng gửi (phân phối khó hơn).
- **Non-Live (10%)** — các trường hợp kiểm thử tổng hợp.
- **Hallucination (10%)** — phát hiện khi nào không nên gọi công cụ.

V3 đã giới thiệu đánh giá dựa trên trạng thái (state-based evaluation): sau một chuỗi công cụ, kiểm tra trạng thái thực tế của API (ví dụ: "file đã được tạo chưa?") thay vì khớp với AST của các lời gọi công cụ. V4 đã thêm các danh mục tìm kiếm web, bộ nhớ và độ nhạy định dạng.

Phát hiện chính năm 2026: function calling single-turn gần như đã được giải quyết. Các lỗi tập trung vào bộ nhớ (duy trì ngữ cảnh qua các lượt), ra quyết định động (chọn công cụ dựa trên kết quả trước đó), chuỗi dài hạn (lệch hướng sau hơn 20 bước) và phát hiện ảo tưởng (từ chối gọi khi không có công cụ phù hợp).

### Tool schema

Mỗi nhà cung cấp đều có một schema. Chúng khác nhau về chi tiết nhưng có chung hình dạng:

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic sử dụng `input_schema` trực tiếp. OpenAI sử dụng `function.parameters`. Cả hai đều chấp nhận JSON Schema. Các mô tả (descriptions) đóng vai trò quan trọng — model đọc chúng để chọn đúng công cụ. Các mô tả công cụ tồi là nguyên nhân gốc rễ số 1 dẫn đến lỗi chọn sai công cụ.

### Kiểm tra đối số (Argument validation)

Đừng tin bất kỳ lời gọi công cụ nào. Hãy kiểm tra:

1. **Ép kiểu (Type coercion).** Model có thể trả về chuỗi "5" trong khi schema yêu cầu int. Hãy ép kiểu nếu không gây mơ hồ; từ chối nếu không thể.
2. **Kiểm tra Enum.** Nếu schema yêu cầu `status in {"open", "closed"}` và model phát ra `"in_progress"`, hãy từ chối với một lỗi mô tả rõ ràng.
3. **Các trường bắt buộc.** Thiếu trường bắt buộc -> trả về quan sát lỗi ngay lập tức cho model, không để crash.
4. **Kiểm tra định dạng.** Ngày tháng, email, URL — hãy kiểm tra bằng các trình phân tích cú pháp cụ thể, không dùng regex.

Mỗi lỗi kiểm tra nên trả về một quan sát có cấu trúc để model có thể thử lại với định dạng đúng.

### Parallel tool calls

Các nhà cung cấp hiện đại hỗ trợ gọi công cụ song song trong một lượt của assistant. Vòng lặp:

1. Model phát ra 3 lời gọi công cụ với các `tool_use_id` riêng biệt.
2. Runtime thực thi chúng (song song nếu độc lập).
3. Mỗi kết quả được trả về dưới dạng khối `tool_result` được liên kết bởi `tool_use_id`.

Quy tắc kỹ thuật: coi các ID liên kết là thành phần quan trọng. Nếu tráo đổi chúng, bạn sẽ nhận được kết quả định tuyến sai từ công cụ này sang công cụ khác.

### Sandboxing

Thực thi công cụ là ranh giới của sandbox. Xem Bài 09 để biết chi tiết. Tóm tắt: mỗi công cụ nên chỉ định bề mặt đọc/ghi, truy cập mạng, timeout, giới hạn bộ nhớ. Việc sử dụng `run_shell(cmd)` chung chung là một dấu hiệu cảnh báo; `git_status()` cụ thể sẽ an toàn hơn.

```figure
tool-routing
```

## Build It

`code/main.py` triển khai một registry công cụ theo chuẩn sản xuất:

- Trình kiểm tra tập con JSON Schema (chỉ dùng stdlib).
- Đăng ký công cụ với mô tả, schema đầu vào, timeout và executor.
- Ép kiểu đối số và kiểm tra enum.
- Điều phối công cụ song song với ID liên kết.
- Quan sát lỗi dưới dạng chuỗi có cấu trúc.

Chạy nó:

```
python3 code/main.py
```

Trace cho thấy một mini agent gọi ba công cụ trong một lượt, với một lời gọi bị lỗi cố ý và bị từ chối kèm theo lỗi mô tả để model có thể xử lý.

## Use It

Mỗi nhà cung cấp đều có schema công cụ riêng — Anthropic, OpenAI, Gemini, Bedrock. Hãy sử dụng lớp chuyển đổi (OpenAI Agents SDK, Vercel AI SDK, LangChain tool adapter) nếu bạn cần đa nhà cung cấp. BFCL là benchmark tham chiếu — hãy chạy nó với agent của bạn trước khi phát hành nếu việc sử dụng công cụ là trọng tâm của sản phẩm.

## Ship It

`outputs/skill-tool-registry.md` tạo ra một danh mục công cụ, schema và registry cho một miền tác vụ nhất định. Bao gồm các kiểm tra chất lượng mô tả (mô tả của mỗi công cụ có cho model biết khi nào nên sử dụng nó không?).

## Bài tập

1. Thêm một công cụ "no-op" cho phép model từ chối sử dụng bất kỳ công cụ nào khác một cách rõ ràng. Đo lường trên một bài kiểm tra ảo tưởng kiểu BFCL.
2. Triển khai ép kiểu đối số cho int-as-string và float-as-string. Việc ép kiểu bắt đầu che giấu các lỗi thực sự ở đâu?
3. Thêm timeout cho mỗi công cụ và một circuit breaker (từ chối công cụ trong 60 giây sau 3 lần thất bại liên tiếp). Điều này thay đổi cách model phục hồi như thế nào?
4. Đọc mô tả BFCL V4. Chọn một danh mục (ví dụ: "multi-turn") và chạy 10 prompt ví dụ qua agent của bạn. Báo cáo tỷ lệ thành công.
5. Chuyển trình kiểm tra stdlib sang Pydantic hoặc Zod. Pydantic/Zod đã bắt được những gì mà bản demo bỏ lỡ?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|-----------|----------------------|------------------|
| Function calling | "Tool use" | Gọi công cụ với đầu ra có cấu trúc và schema đã được kiểm chứng |
| Toolformer | "Self-supervised tool annotation" | Schick 2023 — giữ lại các lời gọi công cụ có kết quả làm giảm loss của token tiếp theo |
| BFCL | "Berkeley Function Calling Leaderboard" | Benchmark 2026: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination |
| Tool schema | "Function signature for the model" | Tên, mô tả, JSON Schema của các đối số |
| tool_use_id | "Correlation ID" | Liên kết lời gọi công cụ với kết quả của nó; thiết yếu cho điều phối song song |
| Hallucination detection | "Know when not to call" | Danh mục V4: từ chối gọi khi không có công cụ nào phù hợp |
| Argument coercion | "String-to-int repair" | Sửa lỗi nhỏ cho các trường hợp không khớp schema có thể dự đoán; từ chối nếu mơ hồ |
| Sandboxing | "Tool execution boundary" | Bề mặt đọc/ghi, mạng, timeout, giới hạn bộ nhớ cho mỗi công cụ |

## Đọc thêm

- [Schick và cộng sự, Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — tự chú thích công cụ
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — benchmark đánh giá 2026
- [Anthropic, Tài liệu về Tool use](https://platform.claude.com/docs/en/agent-sdk/overview) — schema công cụ sản xuất trong Claude Agent SDK
- [Tài liệu OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — kiểu công cụ function và Guardrails