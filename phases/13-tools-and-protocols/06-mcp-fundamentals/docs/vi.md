# Các nguyên lý cơ bản của MCP: Yêu cầu không trạng thái (Stateless) và JSON-RPC

> MCP hiện đại không có bắt tay (handshake) và không có phiên giao thức (protocol session). Mỗi yêu cầu phải mang đủ siêu dữ liệu (metadata) để có thể được hiểu, ủy quyền, định tuyến và thử lại một cách độc lập.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 01 through 05
**Time:** ~55 minutes

## Mục tiêu học tập

- Phân biệt các nguyên hàm (primitives) phía server của MCP với các tính năng phía client.
- Xây dựng các yêu cầu và phản hồi JSON-RPC 2.0 hợp lệ cho MCP `2026-07-28`.
- Đính kèm phiên bản giao thức, khả năng của client (client capabilities) và định danh client vào mọi yêu cầu.
- Sử dụng `server/discover` và xử lý `UnsupportedProtocolVersionError` mà không cần bắt tay.
- Truy vết một yêu cầu độc lập từ khâu xác thực cho đến khi có kết quả hoàn chỉnh.

## Vấn đề

Một server MCP có thể nhận hai yêu cầu liên tiếp từ các client khác nhau, với các khả năng khác nhau, trên cùng một tiến trình hoặc HTTP worker. Nếu server ghi nhớ những gì yêu cầu trước đó đã khai báo, nó có thể áp dụng sai quyền hoặc trả về định dạng dữ liệu không chính xác.

MCP `2026-07-28` loại bỏ sự mơ hồ đó. Cốt lõi của giao thức là không trạng thái (stateless). Server phải quyết định cách xử lý yêu cầu hiện tại dựa trên chính yêu cầu đó, chứ không phải dựa trên lịch sử kết nối.

Điều này thay đổi mô hình tư duy. Trình tự cũ là kết nối trước, bắt tay sau, thực hiện thao tác cuối cùng. Trình tự hiện đại đơn giản hơn:

1. Client gửi một yêu cầu tự mô tả.
2. Server xác thực phiên bản và khả năng của yêu cầu đó.
3. Server xử lý phương thức.
4. Server trả về kết quả có kiểu dữ liệu cụ thể hoặc lỗi JSON-RPC.

Yêu cầu tiếp theo sẽ lặp lại cùng quy trình này từ đầu.

## Khái niệm

### Các nguyên hàm của Server

Các server MCP cung cấp ba nguyên hàm chính:

1. **Tools** là các hành động do model điều khiển, được khám phá bằng `tools/list` và được gọi bằng `tools/call`.
2. **Resources** là dữ liệu được định địa chỉ bằng URI, được khám phá bằng `resources/list` và truy xuất bằng `resources/read`.
3. **Prompts** là các mẫu có thể tái sử dụng, được khám phá bằng `prompts/list` và được render bằng `prompts/get`.

Roots, sampling và logging vẫn nằm trong lược đồ `2026-07-28` để tương thích ngược, nhưng chúng đã bị loại bỏ (deprecated). Các triển khai mới nên sử dụng đầu vào tool hoặc resource rõ ràng cho roots, API trực tiếp của nhà cung cấp model cho sampling, và stderr hoặc OpenTelemetry cho logging. Elicitation vẫn khả dụng thông qua Multi Round-Trip Requests, nơi server trả về một yêu cầu đầu vào và client thử lại thao tác gốc. Một server hiện đại không bao giờ tự khởi tạo một yêu cầu JSON-RPC độc lập.

### Các gói tin JSON-RPC

MCP sử dụng JSON-RPC 2.0:

- Request: `{jsonrpc, id, method, params}`
- Response: `{jsonrpc, id, result}` hoặc `{jsonrpc, id, error}`
- Notification: `{jsonrpc, method, params}` không có `id`

Yêu cầu `id` tương ứng với một phản hồi. Nó không tạo ra một phiên giao thức.

### Siêu dữ liệu yêu cầu bắt buộc

Mọi yêu cầu hiện đại đều mang một đối tượng `_meta` bên trong `params`:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Phiên bản giao thức và khả năng của client là bắt buộc. Định danh client được khuyến nghị sử dụng. Đây là dữ liệu hiển thị và gỡ lỗi do client tự báo cáo, không phải là thông tin xác thực bảo mật.

Server không được phép suy luận bất kỳ giá trị nào trong số này từ một yêu cầu trước đó, một tiến trình stdio, một kết nối HTTP hoặc chỉ từ header truyền tải.

### Kết quả hoàn chỉnh và định danh server

Mọi kết quả hiện đại thành công đều bao gồm `resultType`. Một kết quả cuối cùng thông thường sử dụng `"complete"`. Các server cũng nên tự định danh trong siêu dữ liệu kết quả:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "resultType": "complete",
    "tools": [],
    "ttlMs": 30000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "notes-server",
        "version": "1.0.0"
      }
    }
  }
}
```

`tools/list`, `resources/list`, `prompts/list`, `resources/templates/list`, `resources/read` và `server/discover` là các kết quả có thể lưu vào bộ nhớ đệm (cacheable). Chúng bao gồm `ttlMs` và `cacheScope`. Giá trị mặc định an toàn là `ttlMs: 0` và `cacheScope: "private"`. Các mục trong danh sách nên có thứ tự xác định để các phản hồi tương đương tạo ra các khóa cache ổn định và ngữ cảnh model ổn định.

### Khám phá mà không cần bắt tay

Mỗi server hiện đại phải triển khai `server/discover`. Client có thể gọi phương thức này trước bất kỳ phương thức nào khác để truy xuất:

- `supportedVersions`
- `capabilities` của server
- `instructions` tùy chọn
- Định danh server trong kết quả `_meta`
- Các gợi ý cache

Khám phá rất hữu ích, nhưng nó không phải là một cổng chặn. Client có thể gửi `tools/list` trước vì yêu cầu đó đã mang theo phiên bản giao thức và khả năng của nó.

Nếu phiên bản được yêu cầu không được hỗ trợ, server sẽ trả về mã JSON-RPC `-32022` cùng với:

```json
{
  "requested": "2027-01-01",
  "supported": ["2026-07-28"]
}
```

Client sẽ chọn một phiên bản hiện đại được hỗ trợ chung và thử lại với một id yêu cầu JSON-RPC mới.

### Vòng đời của một yêu cầu

Truy vết một yêu cầu hiện đại theo thứ tự sau:

1. Phân tích cú pháp một gói tin JSON-RPC.
2. Xác nhận `jsonrpc` là `"2.0"`, một `id` tồn tại, `method` là một chuỗi và `params` là một đối tượng.
3. Yêu cầu chuỗi phiên bản và đối tượng khả năng trong `params._meta`; siêu dữ liệu bị thiếu hoặc sai định dạng sẽ là `-32602`.
4. Tại ranh giới HTTP, so sánh phiên bản, phương thức và các header tên áp dụng với phần thân (body). Sự không khớp là `-32020` ngay cả khi một trong hai giá trị phiên bản không được hỗ trợ.
5. Sau khi xác lập sự tương đương, từ chối phiên bản đã khớp nhưng không được hỗ trợ với `-32022`.
6. Kiểm tra các khả năng bắt buộc, sau đó định tuyến theo `method` và xác thực các đối số cụ thể của phương thức.
7. Xác thực và ủy quyền thao tác cụ thể trước khi trình xử lý (handler) của nó chạy.
8. Trả về kết quả hoàn chỉnh với định danh server.
9. Xóa bỏ siêu dữ liệu giao thức phạm vi yêu cầu.

Thứ tự đó ngăn chặn việc hai thành phần diễn giải các lệnh gọi khác nhau. Một gateway không được phép ủy quyền `Mcp-Name: notes.read` trong khi origin thực thi `params.name: notes.delete`. Nó cũng giữ cho đầu vào sai định dạng, nhầm lẫn header, đàm phán phiên bản, lỗi khả năng, ủy quyền và lỗi trình xử lý là các bằng chứng riêng biệt.

Việc đóng stdin hoặc một phản hồi HTTP sẽ kết thúc hoạt động truyền tải. Nó không chấm dứt một phiên giao thức vì MCP hiện đại không có phiên giao thức.

### Tương thích ngược rõ ràng

Các phiên bản đến `2025-11-25` sử dụng `initialize`, `notifications/initialized`, các khả năng phạm vi kết nối và, trên Streamable HTTP cũ, các phiên giao thức tùy chọn. Hành vi đó vẫn liên quan khi một client đa thời đại (dual-era) giao tiếp với một server cũ.

Hãy giữ các thời đại tách biệt. Một yêu cầu hiện đại được xác định bởi siêu dữ liệu bắt buộc trên mỗi yêu cầu. Một kết nối cũ chỉ được chọn thông qua đường dẫn dự phòng đã được tài liệu hóa. Không gửi `initialize` làm mặc định cho một server `2026-07-28`.

Do đó, "không trạng thái" có ý nghĩa cụ thể theo thời đại. Trong `2026-07-28`, đó là một bất biến của giao thức: mọi yêu cầu thông thường đều có thể diễn giải độc lập và không tồn tại phiên MCP nào. Trong các phiên bản đến `2025-11-25`, việc khởi tạo và các khả năng đã đàm phán thuộc về một kết nối, vì vậy một bộ chuyển đổi tương thích có thể giữ lại trạng thái kết nối cũ đó. Một triển khai đa thời đại không phải là một máy trạng thái cho phép. Nó là một lõi hiện đại không trạng thái bên cạnh một bộ chuyển đổi cũ biệt lập, với quyết định lựa chọn rõ ràng trước khi bất kỳ trình phân tích cú pháp nào chạy.

Không định nghĩa nào cấm trạng thái ứng dụng bền vững. Một quy trình làm việc, tác vụ hoặc bản nháp có thể tồn tại phía sau một handle ẩn trong kho lưu trữ chia sẻ. Client gửi handle đó như đầu vào thông thường, và mỗi bản sao sẽ xác thực và ủy quyền việc sử dụng nó. Ngữ cảnh giao thức không được phép rò rỉ vào kho lưu trữ đó như một sự thay thế cho phiên đã bị loại bỏ.

```figure
mcp-tool-call
```

## Sử dụng

`code/main.py` xây dựng, xác thực, truy vết và điều phối các thông điệp MCP hiện đại mà không cần framework. Chạy:

```bash
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Theo dõi ba bất biến trong đầu ra:

- Mọi yêu cầu lặp lại các trường `_meta` của nó.
- Mọi kết quả thành công đều là `resultType: "complete"` và bao gồm định danh server.
- Kết quả danh sách được sắp xếp theo thứ tự xác định và có các gợi ý cache rõ ràng.

## Triển khai

Bài học này xuất bản `outputs/skill-mcp-handshake-tracer.md`. Tên tệp lịch sử vẫn ổn định, nhưng artifact hiện là một trình truy vết yêu cầu không trạng thái. Nó kiểm tra từng thông điệp một cách độc lập và chỉ dán nhãn lưu lượng bắt tay cũ khi nó thực sự hiện diện.

## Bài tập

1. Thay đổi phiên bản giao thức của một yêu cầu thành `2027-01-01`. Xác nhận mã lỗi là `-32022` và dữ liệu quảng bá phiên bản được hỗ trợ.
2. Xóa `io.modelcontextprotocol/clientCapabilities` khỏi yêu cầu thứ hai. Xác nhận server không tái sử dụng các khả năng từ yêu cầu đầu tiên.
3. Đảo ngược registry tool trong bộ nhớ. Xác nhận `tools/list` vẫn trả về cùng một thứ tự xác định.
4. Thay đổi `cacheScope` từ `public` thành `private`. Giải thích ngữ cảnh ủy quyền nào có thể tái sử dụng phản hồi trong mỗi trường hợp.
5. Thêm một bài kiểm tra bỏ qua `clientInfo` tùy chọn. Yêu cầu vẫn phải hợp lệ vì định danh client là được khuyến nghị, không phải bắt buộc.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa |
|------|---------|
| Stateless protocol | Mọi yêu cầu cung cấp siêu dữ liệu cần thiết để diễn giải nó |
| Request metadata | Phiên bản, khả năng của client và định danh client được khuyến nghị trong `params._meta` |
| `server/discover` | Phương thức server bắt buộc cho các phiên bản, khả năng, hướng dẫn và định danh |
| `resultType` | Bộ phân biệt trên mọi kết quả hiện đại thành công |
| Cacheable result | Kết quả bao gồm các gợi ý `ttlMs` và `cacheScope` bắt buộc |
| Protocol era | Siêu dữ liệu trên mỗi yêu cầu hiện đại hoặc khởi tạo phạm vi kết nối cũ |
| Transport lifetime | Thời gian tồn tại của tiến trình, kết nối hoặc luồng phản hồi, không phải trạng thái phiên giao thức |
| `-32022` | Lỗi phiên bản giao thức không được hỗ trợ với các phiên bản được yêu cầu và được hỗ trợ |

## Đọc thêm

- [Kiến trúc MCP](https://modelcontextprotocol.io/specification/2026-07-28/architecture)
- [Giao thức cơ sở MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [Khám phá Server MCP](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [Nhật ký thay đổi MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog)