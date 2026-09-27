# Xây dựng MCP Server: Python và TypeScript không trạng thái (Stateless)

> Một MCP server hiện đại không ghi nhớ các lần bắt tay (handshake). Nó xác thực metadata trên mỗi yêu cầu, chạy một trình xử lý (handler) và trả về một kết quả có kiểu dữ liệu cụ thể.

**Type:** Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 13, Lesson 06
**Time:** ~85 phút

## Mục tiêu học tập

- Triển khai `server/discover` bắt buộc cho MCP `2026-07-28`.
- Xác thực phiên bản giao thức và khả năng của client trên mỗi yêu cầu.
- Hiển thị các công cụ (tools), tài nguyên (resources) và lời nhắc (prompts) với thứ tự danh sách xác định.
- Trả về `resultType`, định danh server và các gợi ý bộ nhớ đệm (cache hints) trên các kết quả chính xác.
- Phục vụ cùng một hợp đồng không trạng thái qua stdio phân tách bằng dòng mới trong Python và TypeScript.

## Vấn đề

Một server lưu trữ khả năng của client sau tin nhắn đầu tiên thì dễ xây dựng nhưng khó vận hành. Cùng một tiến trình có thể phục vụ các client tuần tự. Một yêu cầu từ xa có thể rơi vào một worker khác. Việc khai báo khả năng cũ (stale) có thể làm rò rỉ hành vi qua các ranh giới ủy quyền.

MCP `2026-07-28` giải quyết phần giao thức của vấn đề đó bằng cách làm cho mọi yêu cầu tự mô tả. Ứng dụng của bạn vẫn có thể giữ các ghi chú bền vững, công việc hoặc các handle trạng thái rõ ràng. Điều nó không thể giữ là trạng thái giao thức ẩn làm thay đổi cách một yêu cầu sau đó được giải mã.

Bài học này xây dựng một server ghi chú hai lần. Các phiên bản Python và TypeScript chỉ sử dụng thư viện tiêu chuẩn của chúng cho lõi giao thức. Cả hai đều hiển thị cùng các phương thức và thực thi cùng một hợp đồng truyền tin.

## Khái niệm

### Vòng lặp điều phối hiện đại

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

Ba quy tắc stdio vẫn quan trọng:

- Chỉ ghi các tin nhắn JSON-RPC vào stdout. Gửi chẩn đoán đến stderr.
- Phân tách các tin nhắn bằng dòng mới và flush mỗi phản hồi.
- Thoát ngay lập tức khi stdin đạt EOF.

Vòng đời tiến trình là vòng đời truyền tải. Nó không phải là một phiên MCP hiện đại.

### Xác thực yêu cầu

Mỗi yêu cầu phải có:

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Hai trường đầu tiên là bắt buộc. `clientInfo` được khuyến nghị. Xác thực hình dạng định danh hiện có, nhưng không coi đó là xác thực (authentication).

Nếu phiên bản không được hỗ trợ, trả về mã `-32022` với `requested` và `supported`. Metadata yêu cầu bị thiếu là tham số không hợp lệ, mã `-32602`. Không bao giờ điền các trường bị thiếu từ một lệnh gọi trước đó.

### Khám phá bắt buộc

Các server hiện đại phải triển khai `server/discover`. Một kết quả khám phá đầy đủ bao gồm các phiên bản hiện đại được hỗ trợ, khả năng, hướng dẫn tùy chọn, gợi ý bộ nhớ đệm và định danh server trong kết quả `_meta`:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

Khám phá không mở khóa server. Một client có thể gọi `tools/list` mà không cần gọi khám phá vì `tools/list` đã mang cùng metadata yêu cầu.

### Công cụ (Tools)

`tools/list` trả về một danh sách xác định các mô tả công cụ. Thứ tự ổn định cải thiện việc lưu bộ nhớ đệm phản hồi và giữ cho ngữ cảnh mô hình ổn định. Kết quả cũng yêu cầu `ttlMs` và `cacheScope`.

`tools/call` trả về các khối nội dung và `isError`. Sử dụng lỗi JSON-RPC khi phong bì giao thức hoặc tham số phương thức không hợp lệ. Sử dụng `isError: true` khi một lệnh gọi công cụ hợp lệ chạy nhưng bản thân công cụ đó thất bại.

Các chú thích công cụ vẫn là gợi ý, không phải thực thi:

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

Host nên sử dụng chúng để xác nhận và trình bày. Server vẫn phải thực thi ủy quyền thực tế.

### Tài nguyên (Resources)

`resources/list` trả về các mô tả URI ổn định. `resources/read` trả về nội dung có kiểu. Cả hai đều có thể lưu bộ nhớ đệm trong `2026-07-28`, vì vậy cả hai đều bao gồm `ttlMs` và `cacheScope`.

Sử dụng `cacheScope: "private"` cho dữ liệu ghi chú cụ thể của người dùng. Một bộ nhớ đệm chia sẻ không được sử dụng lại phản hồi riêng tư qua các ngữ cảnh ủy quyền.

Việc phân phối thay đổi hiện đại không sử dụng `resources/subscribe`. Một client mở `subscriptions/listen` và yêu cầu `resourceSubscriptions` hoặc các danh mục thay đổi danh sách. Bài học 10 xây dựng luồng đó.

### Lời nhắc (Prompts)

`prompts/list` có thể lưu bộ nhớ đệm và xác định. `prompts/get` hiển thị một lời nhắc có tên với các đối số. Kết quả lời nhắc được hiển thị là hoàn chỉnh, nhưng nó không phải là một trong các danh sách có thể lưu bộ nhớ đệm hoặc kết quả đọc yêu cầu gợi ý bộ nhớ đệm.

### Mọi kết quả thành công đều được định kiểu

Các ví dụ sử dụng một wrapper cho mọi thành công:

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

Các handler danh sách, đọc và khám phá thêm `ttlMs` cộng với `cacheScope`. Việc tập trung wrapper này ngăn handler bỏ qua các trường kết quả hiện đại một cách âm thầm.

### Không có yêu cầu do server khởi tạo

Một server hiện đại có thể gửi thông báo liên quan đến yêu cầu của client, hoặc thông báo trên luồng `subscriptions/listen` do client mở. Nó không được gửi yêu cầu JSON-RPC của riêng mình.

Khi một handler cần lấy mẫu, gợi ý hoặc đầu vào gốc, nó trả về kết quả `input_required`. Client thực hiện các yêu cầu đầu vào được nhúng và thử lại phương thức gốc với một id yêu cầu mới. Bài học 11 bao gồm mô hình Yêu cầu Đa vòng lặp (Multi Round-Trip Request) đó.

### Tương thích kế thừa rõ ràng

Một server đa thời đại cũng có thể triển khai bắt tay `2025-11-25` trên một nhánh kế thừa tách biệt rõ ràng. Nó chọn hành vi hiện đại khi các trường `_meta` hiện đại bắt buộc có mặt và hành vi kế thừa khi nhận được `initialize`.

Không đưa yêu cầu `2026-07-28` qua đường dẫn bắt tay kế thừa. Không đóng dấu các trường `resultType` hiện đại lên các kết quả khởi tạo kế thừa. Mã trong bài học này cố tình chỉ dành cho hiện đại để các bất biến của nó vẫn hiển thị.

```figure
t3-dispatch-loop
```

## Sử dụng

Chạy demo và các bài kiểm tra hữu hạn của server Python:

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

Chạy cổng TypeScript với trình chạy TypeScript:

```bash
npx tsx main.ts --demo
```

Demo gửi `server/discover`, liệt kê từng nguyên thủy, gọi các công cụ và hiển thị lỗi phiên bản không được hỗ trợ. Mỗi yêu cầu hiện đại lặp lại metadata. Mỗi thành công bao gồm định danh server.

## Triển khai

Bài học này cung cấp `outputs/skill-mcp-server-scaffolder.md`. Nó tạo ra một kế hoạch server hiện đại với hợp đồng khám phá, xác thực theo yêu cầu, các danh sách có thể lưu bộ nhớ đệm xác định và một bộ chuyển đổi kế thừa cô lập tùy chọn.

## Bài tập

1. Xóa khả năng khỏi một yêu cầu và chứng minh server không sử dụng lại khai báo của yêu cầu trước đó.
2. Đảo ngược thứ tự `TOOLS`, `PROMPTS` và chèn ghi chú. Xác nhận tất cả kết quả danh sách vẫn ổn định.
3. Thêm công cụ `notes_delete` mang tính phá hủy và yêu cầu kiểm tra ủy quyền bên trong executor. Giữ `destructiveHint` chỉ như một gợi ý UX.
4. Thêm `resources/templates/list` với `ttlMs`, `cacheScope` và thứ tự xác định.
5. Xây dựng một bộ chuyển đổi kế thừa riêng cho `2025-11-25`. Thêm các bài kiểm tra chứng minh một yêu cầu hiện đại không bao giờ đi vào đó.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa |
|------|---------|
| Stateless server | Xử lý mỗi yêu cầu từ metadata của chính nó mà không cần bộ nhớ phiên giao thức |
| `server/discover` | Phương thức hiện đại bắt buộc quảng bá các phiên bản và khả năng |
| Complete result | Kết quả hiện đại thành công với `resultType: "complete"` |
| Cacheable result | Kết quả khám phá, danh sách hoặc đọc tài nguyên với `ttlMs` và `cacheScope` |
| Deterministic list | Cùng một registry logic tạo ra cùng một thứ tự mục |
| Server identity | `io.modelcontextprotocol/serverInfo` được khuyến nghị trong kết quả `_meta` |
| Tool error | Lệnh gọi công cụ hợp lệ trả về nội dung với `isError: true` |
| Protocol error | Yêu cầu JSON-RPC hoặc MCP không hợp lệ được trả về qua `error` |

## Đọc thêm

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)