# Function Calling Deep Dive — OpenAI, Anthropic, Gemini

> Ba nhà cung cấp tiên phong đã hội tụ về cùng một vòng lặp gọi công cụ (tool-call loop) vào năm 2024 và sau đó khác biệt hóa ở mọi khía cạnh còn lại. OpenAI sử dụng `tools` và `tool_calls`. Anthropic sử dụng các khối `tool_use` và `tool_result`. Gemini sử dụng `functionDeclarations` và cơ chế tương quan unique-id. Bài học này so sánh sự khác biệt giữa ba nhà cung cấp để mã nguồn của bạn không bị lỗi khi chuyển đổi giữa các nền tảng.

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01 (giao diện công cụ)
**Time:** ~75 phút

## Mục tiêu học tập

- Nêu được ba điểm khác biệt về hình thái giữa các payload gọi hàm của OpenAI, Anthropic và Gemini (khai báo, lời gọi, kết quả).
- Chuyển đổi một khai báo công cụ sang định dạng của cả ba nhà cung cấp và dự đoán nơi các ràng buộc strict-mode sẽ khác biệt.
- Sử dụng `tool_choice` ở mỗi nhà cung cấp để ép buộc, cấm hoặc tự động chọn các lời gọi công cụ.
- Nắm vững các giới hạn cứng của từng nhà cung cấp (số lượng công cụ, độ sâu schema, độ dài đối số) và các chữ ký lỗi mà mỗi bên phát ra khi vi phạm giới hạn.

## Vấn đề

Hình thái của một yêu cầu gọi hàm khác nhau tùy theo nhà cung cấp. Ba ví dụ cụ thể từ các stack sản xuất năm 2026:

**OpenAI Chat Completions / Responses API.** Bạn truyền `tools: [{type: "function", function: {name, description, parameters, strict}}]`. Phản hồi của mô hình chứa `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`, trong đó `arguments` là một chuỗi JSON mà bạn phải phân tích cú pháp. Strict mode (`strict: true`) thực thi việc tuân thủ schema thông qua giải mã có ràng buộc.

**Anthropic Messages API.** Bạn truyền `tools: [{name, description, input_schema}]`. Phản hồi trả về dưới dạng `content: [{type: "text"}, {type: "tool_use", id, name, input}]`. `input` đã được phân tích cú pháp (là một object, không phải chuỗi). Bạn phản hồi bằng một tin nhắn `user` mới chứa khối `{type: "tool_result", tool_use_id, content}`.

**Google Gemini API.** Bạn truyền `tools: [{functionDeclarations: [{name, description, parameters}]}]` (nằm trong `functionDeclarations`). Phản hồi đến dưới dạng `candidates[0].content.parts: [{functionCall: {name, args, id}}]`, trong đó `id` là duy nhất trong Gemini 3 trở lên để tương quan các lời gọi song song. Bạn phản hồi bằng `{functionResponse: {name, id, response}}`.

Cùng một vòng lặp. Tên trường khác nhau, cấu trúc lồng nhau khác nhau, quy ước chuỗi-so-với-object khác nhau, cơ chế tương quan khác nhau. Một đội ngũ viết agent thời tiết trên OpenAI sẽ mất hai ngày để chuyển sang Anthropic và thêm một ngày cho Gemini chỉ để xử lý phần kết nối (plumbing).

Bài học này xây dựng một bộ chuyển đổi thống nhất ba định dạng thành một khai báo công cụ chuẩn và định tuyến tại biên. Phase 13 · 17 khái quát hóa mô hình này thành một LLM gateway.

## Khái niệm

### Cấu trúc chung

Mọi nhà cung cấp đều cần năm yếu tố:

1. **Danh sách công cụ.** Tên, mô tả và schema đầu vào cho mỗi công cụ.
2. **Lựa chọn công cụ.** Ép buộc một công cụ cụ thể, cấm công cụ hoặc để mô hình tự quyết định.
3. **Phát hành lời gọi.** Đầu ra có cấu trúc nêu tên công cụ và các đối số.
4. **Id lời gọi.** Tương quan phản hồi với đúng lời gọi (quan trọng đối với xử lý song song).
5. **Tiêm kết quả.** Một tin nhắn hoặc khối liên kết kết quả trở lại lời gọi.

### So sánh sự khác biệt về hình thái, theo từng trường

| Khía cạnh | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| Bao bì khai báo | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Trường Schema | `parameters` | `input_schema` | `parameters` |
| Container phản hồi | `tool_calls[]` trên tin nhắn assistant | `content[]` thuộc loại `tool_use` | `parts[]` thuộc loại `functionCall` |
| Loại đối số | JSON dạng chuỗi | object đã phân tích | object đã phân tích |
| Định dạng Id | `call_...` (OpenAI tạo) | `toolu_...` (Anthropic) | UUID (Gemini 3+) |
| Khối kết quả | role `tool`, `tool_call_id` | `user` với `tool_result`, `tool_use_id` | `functionResponse` với `id` tương ứng |
| Ép buộc công cụ | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Cấm công cụ | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Strict schema | `strict: true` | schema-là-schema (luôn thực thi) | `responseSchema` ở cấp độ yêu cầu |

### Các giới hạn bạn sẽ thực sự gặp phải

- **OpenAI.** 128 công cụ mỗi yêu cầu. Độ sâu schema là 5. Chuỗi đối số <= 8192 byte. Strict mode yêu cầu không có `$ref`, không có `oneOf`/`anyOf`/`allOf` chồng lấp, mọi thuộc tính phải được liệt kê trong `required`.
- **Anthropic.** 64 công cụ mỗi yêu cầu. Độ sâu schema về lý thuyết là không giới hạn nhưng thực tế là 10. Không có cờ strict-mode; schema là một hợp đồng và mô hình thường tuân thủ.
- **Gemini.** 64 hàm mỗi yêu cầu. Các loại schema là tập con của OpenAPI 3.0 (hơi khác so với JSON Schema 2020-12). Các lời gọi song song có unique-id từ Gemini 3.

### Hành vi `tool_choice`

Ba chế độ mà mọi bên đều hỗ trợ, nhưng được đặt tên khác nhau.

- **Auto.** Mô hình tự chọn công cụ hoặc văn bản. Mặc định.
- **Required / Any.** Mô hình bắt buộc phải gọi ít nhất một công cụ.
- **None.** Mô hình không được phép gọi công cụ.

Cộng thêm một chế độ độc nhất cho mỗi nhà cung cấp:

- **OpenAI.** Ép buộc một công cụ cụ thể theo tên.
- **Anthropic.** Ép buộc một công cụ cụ thể theo tên; cờ `disable_parallel_tool_use` phân biệt giữa đơn lẻ và đa công cụ.
- **Gemini.** `mode: "VALIDATED"` định tuyến mọi phản hồi qua bộ kiểm tra schema bất kể ý định của mô hình.

### Lời gọi song song

`parallel_tool_calls: true` của OpenAI (mặc định) phát ra nhiều lời gọi trong một tin nhắn assistant. Bạn thực thi tất cả và phản hồi bằng một tin nhắn tool-role theo lô chứa một mục cho mỗi `tool_call_id`. Anthropic trước đây chỉ hỗ trợ gọi đơn lẻ; `disable_parallel_tool_use: false` (mặc định từ Claude 3.5) cho phép gọi đa công cụ. Gemini 2 cho phép gọi song song nhưng không cung cấp id ổn định; Gemini 3 thêm UUID để các phản hồi không theo thứ tự có thể tương quan chính xác.

### Streaming

Cả ba đều hỗ trợ streaming các lời gọi công cụ. Định dạng truyền tải khác nhau:

- **OpenAI.** Các đoạn delta của `tool_calls[i].function.arguments` đến dần dần. Bạn tích lũy cho đến khi có `finish_reason: "tool_calls"`.
- **Anthropic.** Các sự kiện block-start / block-delta / block-stop. Các đoạn `input_json_delta` mang theo các đối số một phần.
- **Gemini.** `streamFunctionCallArguments` (mới trong Gemini 3) phát ra các đoạn với `functionCallId` để nhiều lời gọi song song có thể xen kẽ nhau.

Phase 13 · 03 đi sâu vào việc lắp ráp lại các lời gọi song song + streaming. Bài học này tập trung vào khai báo và hình thái lời gọi đơn lẻ.

### Lỗi và sửa lỗi

Các lỗi đối số không hợp lệ cũng trông khác nhau.

- **OpenAI (không strict).** Mô hình trả về `arguments: "{bad json}"`, việc phân tích JSON của bạn thất bại, bạn tiêm một thông báo lỗi và gọi lại.
- **OpenAI (strict).** Việc xác thực xảy ra trong quá trình giải mã; JSON không hợp lệ là không thể xảy ra nhưng `refusal` có thể xuất hiện.
- **Anthropic.** `input` có thể chứa các trường không mong đợi; schema chỉ mang tính chất tham khảo. Hãy xác thực ở phía server.
- **Gemini.** Đặc điểm của OpenAPI 3.0: `enum` trên các trường object bị bỏ qua một cách âm thầm; hãy tự xác thực.

### Mô hình bộ chuyển đổi (Translator pattern)

Một khai báo công cụ chuẩn trong mã của bạn trông như thế này (bạn chọn hình thái):

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

Ba hàm nhỏ sẽ chuyển đổi nó sang ba hình thái của nhà cung cấp. Bộ khung trong `code/main.py` thực hiện chính xác điều này, sau đó chạy thử một lời gọi công cụ giả qua hình thái phản hồi của từng nhà cung cấp. Không cần mạng — bài học này dạy về hình thái, không phải HTTP.

Các đội ngũ sản xuất bao bọc bộ chuyển đổi này trong `AbstractToolset` (Pydantic AI), `UniversalToolNode` (LangGraph), hoặc `BaseTool` (LlamaIndex). Phase 13 · 17 cung cấp một gateway hiển thị API theo hình thái OpenAI trước bất kỳ nhà cung cấp nào trong ba bên.

```figure
function-call-args
```

## Sử dụng

`code/main.py` định nghĩa một dataclass `Tool` chuẩn và ba bộ chuyển đổi phát ra JSON khai báo cho OpenAI, Anthropic và Gemini. Sau đó, nó phân tích một phản hồi giả lập của nhà cung cấp theo từng hình thái thành cùng một đối tượng lời gọi chuẩn, chứng minh rằng ngữ nghĩa là giống hệt nhau bên dưới lớp vỏ. Hãy chạy nó và so sánh ba khai báo cạnh nhau.

Những điều cần chú ý:

- Ba khối khai báo chỉ khác nhau ở bao bì và tên trường.
- Ba khối phản hồi khác nhau ở nơi chứa lời gọi (`tool_calls` cấp cao nhất, khối `content[]`, mục `parts[]`).
- Một hàm `canonical_call()` trích xuất `{id, name, args}` từ cả ba hình thái phản hồi.

## Triển khai

Bài học này tạo ra `outputs/skill-provider-portability-audit.md`. Với một tích hợp gọi hàm cho một nhà cung cấp, kỹ năng này tạo ra một bản kiểm toán tính di động: nhà cung cấp nào dựa vào giới hạn nào, trường nào cần đổi tên, và điều gì sẽ hỏng khi chuyển sang nhà cung cấp khác.

## Bài tập

1. Chạy `code/main.py` và xác minh rằng ba JSON khai báo của nhà cung cấp đều serialize cùng một đối tượng `Tool` cơ sở. Sửa đổi công cụ chuẩn để thêm một tham số enum và xác nhận chỉ bộ chuyển đổi Gemini cần xử lý đặc điểm của OpenAPI.

2. Thêm một trình phân tích `ListToolsResponse` cho mỗi nhà cung cấp để trích xuất danh sách công cụ mà mô hình trả về sau một lời gọi `list_tools` hoặc khám phá. OpenAI không có tính năng này một cách tự nhiên; hãy lưu ý sự bất đối xứng này.

3. Triển khai chuyển đổi `tool_choice`: ánh xạ một `ToolChoice(mode="force", tool_name="x")` chuẩn sang cả ba hình thái nhà cung cấp. Sau đó ánh xạ `mode="any"` và `mode="none"`. Kiểm tra bảng so sánh của bài học.

4. Chọn một trong ba nhà cung cấp và đọc hướng dẫn gọi hàm của họ từ đầu đến cuối. Tìm một trường trong đặc tả schema của họ mà hai bên còn lại không hỗ trợ. Ứng viên: `strict` của OpenAI, `disable_parallel_tool_use` của Anthropic, `function_calling_config.allowed_function_names` của Gemini.

5. Viết một vector kiểm thử: một lời gọi công cụ có các đối số vi phạm schema đã khai báo. Chạy nó qua bộ xác thực của từng nhà cung cấp (bộ stdlib trong Bài 01 sẽ đóng vai trò proxy) và ghi lại lỗi nào được kích hoạt. Ghi lại nhà cung cấp nào bạn sẽ sử dụng trong sản xuất vì tính nghiêm ngặt.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Function calling | "Tool use" | API cấp nhà cung cấp để phát hành lời gọi công cụ có cấu trúc |
| Tool declaration | "Tool spec" | Tên + mô tả + payload đầu vào JSON Schema |
| `tool_choice` | "Force / forbid" | Các chế độ auto / required / none / specific-name |
| Strict mode | "Schema enforcement" | Cờ của OpenAI ràng buộc giải mã để khớp với schema |
| Khối `tool_use` | "Anthropic's call shape" | Khối nội dung nội dòng với id, tên, đầu vào |
| Phần `functionCall` | "Gemini's call shape" | Một mục `parts[]` chứa tên, đối số và id |
| Arguments-as-string | "Stringified JSON" | OpenAI trả về đối số dưới dạng chuỗi JSON, không phải object |
| Parallel tool calls | "Fan-out in one turn" | Nhiều lời gọi công cụ trong một tin nhắn assistant |
| Refusal | "Model declines" | Khối từ chối chỉ có trong strict-mode thay vì một lời gọi |
| OpenAPI 3.0 subset | "Gemini schema quirk" | Gemini sử dụng phương ngữ giống JSON-Schema với những khác biệt nhỏ |

## Đọc thêm

- [OpenAI — Hướng dẫn gọi hàm](https://platform.openai.com/docs/guides/function-calling) — tài liệu tham khảo chuẩn bao gồm strict mode và lời gọi song song
- [Anthropic — Tổng quan về sử dụng công cụ](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — ngữ nghĩa khối `tool_use` và `tool_result`
- [Google — Gọi hàm Gemini](https://ai.google.dev/gemini-api/docs/function-calling) — lời gọi song song, unique id và tập con OpenAPI
- [Vertex AI — Tham chiếu gọi hàm](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — bề mặt doanh nghiệp của Gemini
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — chi tiết thực thi schema strict-mode