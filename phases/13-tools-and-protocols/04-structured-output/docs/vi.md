# Structured Output — JSON Schema, Pydantic, Zod, Constrained Decoding

> "Yêu cầu model trả về JSON một cách lịch sự" thường thất bại từ 5 đến 15 phần trăm thời gian, ngay cả trên các model tiên tiến nhất. Structured outputs giải quyết vấn đề này bằng constrained decoding: model bị ngăn chặn hoàn toàn khỏi việc tạo ra các token vi phạm schema. Chế độ strict mode của OpenAI, schema-typed tool use của Anthropic, `responseSchema` của Gemini, `output_type` của Pydantic AI, và `.parse` của Zod là năm hình thức thể hiện của cùng một ý tưởng. Bài học này sẽ xây dựng bộ kiểm tra schema (schema validator) và hợp đồng strict-mode mà các kỹ sư sẽ sử dụng cho mọi pipeline trích xuất dữ liệu trong môi trường production.

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 phút

## Mục tiêu học tập

- Viết JSON Schema 2020-12 cho mục tiêu trích xuất dữ liệu bằng cách sử dụng các ràng buộc phù hợp (enum, min/max, required, pattern).
- Giải thích lý do tại sao strict mode và constrained decoding mang lại các đảm bảo khác biệt so với việc "validate sau khi tạo".
- Phân biệt ba chế độ lỗi: parse error, schema violation, model refusal.
- Triển khai pipeline trích xuất dữ liệu với cơ chế xử lý lỗi và từ chối (refusal) có định kiểu (typed).

## Vấn đề

Một agent đọc email đơn hàng cần chuyển đổi văn bản tự do thành `{customer, line_items, total_usd}`. Có ba cách tiếp cận.

**Cách một: prompt yêu cầu JSON.** "Trả lời bằng JSON với các trường customer, line_items, total_usd." Hoạt động từ 85 đến 95 phần trăm thời gian trên các model tiên tiến. Thất bại theo sáu cách: thiếu dấu ngoặc nhọn, thừa dấu phẩy, sai kiểu dữ liệu, trường dữ liệu bị ảo tưởng (hallucinated), bị cắt bớt tại giới hạn token, hoặc chứa văn bản thừa như "Đây là JSON của bạn:".

**Cách hai: validate sau khi tạo.** Tạo văn bản tự do, parse, kiểm tra với schema, và thử lại nếu thất bại. Đáng tin cậy nhưng tốn kém — bạn phải trả phí cho mỗi lần thử lại, và các lỗi cắt bớt dữ liệu sẽ tốn thêm một lượt gọi cho mỗi lần xảy ra.

**Cách ba: constrained decoding.** Nhà cung cấp thực thi schema ngay tại thời điểm giải mã (decode). Các token không hợp lệ sẽ bị loại bỏ khỏi phân phối lấy mẫu (sampling distribution). Đầu ra được đảm bảo parse thành công và đảm bảo hợp lệ với schema. Lỗi chỉ còn một dạng duy nhất: từ chối (model quyết định rằng đầu vào không khớp với schema).

Mọi nhà cung cấp model tiên tiến năm 2026 đều triển khai một dạng của cách tiếp cận thứ ba.

- **OpenAI.** `response_format: {type: "json_schema", strict: true}` cộng với `refusal` trong phản hồi nếu model từ chối.
- **Anthropic.** Thực thi schema trên các đầu vào `tool_use`; `stop_reason: "refusal"` không tồn tại, nhưng `end_turn` không có tool call là tín hiệu nhận biết.
- **Gemini.** `responseSchema` ở cấp độ request; vào năm 2026, Gemini triển khai các ràng buộc ngữ pháp ở cấp độ token cho các kiểu dữ liệu được chọn.
- **Pydantic AI.** `output_type=InvoiceModel` phát ra một `RunResult` có cấu trúc được định kiểu theo `InvoiceModel`.
- **Zod (TypeScript).** Bộ parser runtime kiểm tra đầu ra của nhà cung cấp dựa trên Zod schema; kết hợp với `beta.chat.completions.parse` của OpenAI.

Điểm chung: khai báo schema một lần, thực thi từ đầu đến cuối.

## Khái niệm

### JSON Schema 2020-12 — ngôn ngữ chung

Mọi nhà cung cấp đều chấp nhận JSON Schema 2020-12. Các cấu trúc bạn sử dụng nhiều nhất:

- `type`: một trong các kiểu `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`.
- `properties`: bản đồ ánh xạ tên trường tới subschema.
- `required`: danh sách các tên trường bắt buộc phải có.
- `enum`: tập hợp cố định các giá trị cho phép.
- `minimum` / `maximum` (số), `minLength` / `maxLength` / `pattern` (chuỗi).
- `items`: subschema áp dụng cho mọi phần tử trong mảng.
- `additionalProperties`: `false` cấm các trường bổ sung (mặc định thay đổi tùy theo chế độ).

Strict mode của OpenAI bổ sung ba yêu cầu: mọi thuộc tính phải được liệt kê trong `required`, `additionalProperties: false` ở mọi nơi, và không có `$ref` chưa được giải quyết. Nếu vi phạm, API sẽ trả về lỗi 400 ngay tại thời điểm request.

### Pydantic, binding cho Python

Pydantic v2 tạo JSON Schema từ các model dạng dataclass thông qua `model_json_schema()`. Pydantic AI bao bọc điều này để bạn viết:

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

và framework agent sẽ dịch schema sang strict mode của OpenAI, `input_schema` của Anthropic, hoặc `responseSchema` của Gemini tại biên. Đầu ra của model trả về dưới dạng một instance `Invoice` đã được định kiểu. Các lỗi validation sẽ raise `ValidationError` với đường dẫn lỗi chi tiết.

### Zod, binding cho TypeScript

Zod (`z.object({customer: z.string(), ...})`) là tương đương trong TS. OpenAI Node SDK cung cấp `zodResponseFormat(Invoice)`, giúp dịch sang payload JSON Schema của API.

### Từ chối (Refusals)

Strict mode không thể ép buộc model phải trả lời. Nếu đầu vào không thể khớp với schema ("email là một bài thơ, không phải hóa đơn"), model sẽ phát ra một trường `refusal` chứa lý do. Code của bạn phải xử lý điều này như một kết quả hợp lệ, không phải là lỗi. Việc từ chối cũng hữu ích như một tín hiệu an toàn: một model được yêu cầu trích xuất số thẻ tín dụng từ một email chứa nội dung bảo mật sẽ trả về sự từ chối kèm theo lý do an toàn.

### Constrained decoding trong thực tế

Các triển khai mã nguồn mở sử dụng ba kỹ thuật.

1. **Grammar-based decoding** (`outlines`, `guidance`, `lm-format-enforcer`): xây dựng một máy trạng thái hữu hạn (DFA) từ schema; tại mỗi bước, che (mask) các logit của các token vi phạm FSM.
2. **Logit masking với JSON parser**: chạy một streaming JSON parser song song với model; tại mỗi bước, tính toán tập hợp các token hợp lệ tiếp theo.
3. **Speculative decoding với verifier**: một model dự thảo (draft model) giá rẻ đề xuất các token, verifier thực thi schema.

Các nhà cung cấp thương mại chọn một trong các cách này phía sau hậu trường. Trạng thái công nghệ năm 2026 cho thấy phương pháp này nhanh hơn so với việc tạo văn bản thông thường đối với các đầu ra có cấu trúc ngắn và tốc độ tương đương đối với các đầu ra dài.

### Ba chế độ lỗi

1. **Parse error.** Đầu ra không phải là JSON hợp lệ. Không thể xảy ra trong strict mode. Vẫn có thể xảy ra với các nhà cung cấp không hỗ trợ strict mode.
2. **Schema violation.** Đầu ra parse thành công nhưng vi phạm schema. Không thể xảy ra trong strict mode. Phổ biến ở các chế độ khác.
3. **Refusal.** Model từ chối. Phải được xử lý như một kết quả có định kiểu.

### Chiến lược thử lại (Retry)

Khi bạn không ở trong strict mode (Anthropic tool use, OpenAI không strict, Gemini cũ), mô hình khôi phục là:

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

Một lần thử lại thường là đủ. Ba lần thử lại sẽ bắt được các lỗi của model yếu. Vượt quá ba lần là dấu hiệu của một schema tồi: model không thể đáp ứng nó cho một số đầu vào, và prompt hoặc schema cần được sửa đổi.

### Hỗ trợ model nhỏ

Constrained decoding hoạt động tốt trên các model nhỏ. Một model mã nguồn mở 3B tham số với việc thực thi ngữ pháp (grammar enforcement) hoạt động hiệu quả hơn một model 70B tham số với prompt thông thường trong các tác vụ có cấu trúc. Đây là lý do chính khiến structured outputs quan trọng đối với production: nó tách biệt độ tin cậy khỏi kích thước model.

```figure
constrained-decoding
```

## Sử dụng

`code/main.py` cung cấp một bộ kiểm tra JSON Schema 2020-12 tối giản trong stdlib (types, required, enum, min/max, pattern, items, additionalProperties). Nó bao bọc một schema `Invoice` và chạy một đầu ra LLM giả lập qua bộ kiểm tra, minh họa các đường dẫn lỗi parse, vi phạm schema và từ chối. Hãy thay thế đầu ra giả lập bằng phản hồi thực tế của bất kỳ nhà cung cấp nào trong môi trường production.

Những điều cần lưu ý:

- Bộ kiểm tra trả về một danh sách `[ValidationError]` có định kiểu với đường dẫn và thông báo lỗi. Đó là định dạng bạn muốn hiển thị cho prompt thử lại.
- Nhánh từ chối (refusal) KHÔNG thử lại. Nó ghi log và trả về một sự từ chối có định kiểu. Phase 14 · 09 sử dụng các sự từ chối như một tín hiệu an toàn.
- Kiểm tra `additionalProperties: false` kích hoạt trên đầu vào kiểm thử đối nghịch, cho thấy lý do tại sao strict mode ngăn chặn các trường dữ liệu bị ảo tưởng.

## Triển khai

Bài học này tạo ra `outputs/skill-structured-output-designer.md`. Với một mục tiêu trích xuất văn bản tự do (hóa đơn, ticket hỗ trợ, sơ yếu lý lịch, v.v.), kỹ năng này tạo ra một JSON Schema 2020-12 tương thích với strict-mode và một Pydantic model phản chiếu nó, với cơ chế xử lý từ chối và thử lại được thiết lập sẵn.

## Bài tập

1. Chạy `code/main.py`. Thêm một trường hợp kiểm thử thứ tư có `total_usd` là một số âm. Xác nhận bộ kiểm tra từ chối nó với đường dẫn ràng buộc `minimum`.

2. Mở rộng bộ kiểm tra để hỗ trợ `oneOf` với discriminator. Trường hợp phổ biến: `line_item` là sản phẩm hoặc dịch vụ, được gắn nhãn bởi `kind`. Strict mode có các quy tắc tinh tế ở đây; hãy kiểm tra hướng dẫn structured outputs của OpenAI.

3. Viết cùng một Invoice schema dưới dạng Pydantic BaseModel và so sánh đầu ra `model_json_schema()` với schema tự viết. Xác định trường duy nhất mà Pydantic đặt mặc định mà phiên bản tự viết bỏ qua.

4. Đo lường tỷ lệ từ chối. Xây dựng mười đầu vào không nên trích xuất được (lời bài hát, chứng minh toán học, email trống) và chạy chúng qua một nhà cung cấp thực tế với strict mode. Đếm số lần từ chối so với các đầu ra bị ảo tưởng. Đây là cơ sở thực tế cho các lần thử lại nhận biết từ chối.

5. Đọc hướng dẫn structured outputs của OpenAI từ đầu đến cuối. Xác định một cấu trúc mà nó cấm rõ ràng trong strict mode trong khi JSON Schema thông thường cho phép. Sau đó thiết kế một schema sử dụng cấu trúc bị cấm đó một cách không cần thiết và refactor nó để tương thích với strict mode.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "The schema spec" | Dialect schema IETF-draft mà mọi nhà cung cấp hiện đại đều hỗ trợ |
| Strict mode | "Guaranteed schema" | Flag của OpenAI thực thi schema thông qua constrained decoding |
| Constrained decoding | "Logit masking" | Thực thi tại thời điểm giải mã, che các token tiếp theo không hợp lệ |
| Refusal | "Model declines" | Kết quả có định kiểu khi đầu vào không khớp với schema |
| Parse error | "Invalid JSON" | Đầu ra không parse được thành JSON; không thể xảy ra trong strict mode |
| Schema violation | "Wrong shape" | Đã parse được nhưng vi phạm kiểu / required / enum / range |
| `additionalProperties: false` | "No extras allowed" | Cấm các trường không xác định; bắt buộc trong OpenAI strict |
| Pydantic BaseModel | "Typed output" | Lớp Python phát ra và kiểm tra JSON Schema |
| Zod schema | "TypeScript output type" | Schema runtime TS để kiểm tra đầu ra của nhà cung cấp |
| Grammar enforcement | "Open-weights constrained decode" | Che logit dựa trên FSM, như trong outlines / guidance |

## Đọc thêm

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict mode, refusals, và các yêu cầu về schema
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — Bài đăng ra mắt tháng 8 năm 2024 giải thích về đảm bảo giải mã
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — các binding output_type có định kiểu serialize sang từng nhà cung cấp
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — đặc tả chuẩn
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — ghi chú triển khai doanh nghiệp và các lưu ý về strict-mode