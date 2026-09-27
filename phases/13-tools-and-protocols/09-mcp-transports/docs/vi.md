# MCP Transports: stdio và Stateless Streamable HTTP

> Transport mang các thông điệp MCP. Nó không cung cấp trạng thái giao thức bị thiếu. Trong `2026-07-28`, stdio cục bộ và Streamable HTTP từ xa đều mang các yêu cầu tự mô tả.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 07 and 08
**Time:** ~65 phút

## Mục tiêu học tập

- Chọn stdio cho các tiến trình con cục bộ và Streamable HTTP cho các dịch vụ mạng.
- Triển khai hợp đồng Streamable HTTP hiện đại chỉ dùng POST, một endpoint duy nhất.
- Đối chiếu và xác thực các header phiên bản, phương thức và tên MCP với phần thân JSON-RPC.
- Phân phối SSE theo phạm vi yêu cầu và các luồng `subscriptions/listen` tồn tại lâu dài một cách chính xác.
- Di chuyển các triển khai dựa trên phiên và HTTP+SSE cũ mà không trình bày hành vi cũ như là hiện đại.

## Vấn đề

Các bản sửa đổi Streamable HTTP trước đây đã kết hợp việc thương lượng giao thức với hành vi kết nối và phiên. Một máy chủ có thể tạo `Mcp-Session-Id`, hiển thị luồng GET độc lập, chấp nhận DELETE để chấm dứt phiên và tiếp tục SSE với `Last-Event-ID`.

MCP `2026-07-28` loại bỏ các cơ chế đó khỏi luồng hiện đại. Mỗi yêu cầu có thể đến bất kỳ worker nào đang hoạt động vì phiên bản giao thức và khả năng của client đều nằm trong phần thân yêu cầu. Các HTTP header đối chiếu các trường đã chọn để định tuyến và áp dụng chính sách, nhưng máy chủ xác thực các header đó với phần thân trước khi thực thi.

Kết quả là dễ mở rộng và dễ suy luận hơn. Điều này cũng có nghĩa là một máy chủ dạy về transport năm 2025 như hiện tại là đang dạy sai mô hình lỗi và bảo mật.

## Khái niệm

### stdio

Binding stdio dành cho tiến trình con do client khởi chạy:

- Client ghi một thông điệp JSON-RPC UTF-8 trên mỗi dòng vào stdin.
- Máy chủ ghi một thông điệp JSON-RPC UTF-8 trên mỗi dòng vào stdout.
- Máy chủ ghi các chẩn đoán vào stderr.
- Máy chủ thoát ngay lập tức khi stdin EOF.
- Mọi yêu cầu hiện đại đều mang phiên bản và khả năng của client trong `params._meta`.

Tiến trình có thể tồn tại qua nhiều lệnh gọi, nhưng nó không phải là một phiên giao thức hiện đại. Nếu nó thoát đột ngột, các yêu cầu đang thực hiện sẽ bị mất. Hãy khởi động lại tiến trình, khám phá lại, liệt kê lại, mở lại các đăng ký và thử lại các thao tác an toàn với các id yêu cầu mới.

### Streamable HTTP vào ngày 28-07-2026

Một máy chủ hiện đại hiển thị một endpoint MCP duy nhất, chẳng hạn như `/mcp`, chấp nhận POST.

Mỗi yêu cầu hoặc thông báo JSON-RPC là một HTTP POST mới. Phần thân chứa một thông điệp JSON-RPC. Client không gửi phản hồi JSON-RPC cho máy chủ.

Đối với một yêu cầu, máy chủ trả về:

- `Content-Type: application/json` với một phản hồi JSON-RPC; hoặc
- `Content-Type: text/event-stream` với các thông báo liên quan đến yêu cầu đó, theo sau là phản hồi JSON-RPC cuối cùng.

Đối với một thông báo được chấp nhận, máy chủ trả về `202 Accepted` mà không có phần thân.

Client quảng bá cả hai loại phản hồi:

```http
Accept: application/json, text/event-stream
```

### POST-only nghĩa là chỉ POST

Streamable HTTP hiện đại không có luồng GET độc lập và không có endpoint phiên DELETE.

- `GET /mcp` trả về `405 Method Not Allowed`.
- `DELETE /mcp` trả về `405 Method Not Allowed`.
- `Mcp-Session-Id` bị bỏ qua và không bao giờ được tạo hoặc phản hồi lại.
- `Last-Event-ID` bị bỏ qua vì các luồng hiện đại không thể tiếp tục (resumable).

Nếu một luồng theo phạm vi yêu cầu bị ngắt trước khi có phản hồi cuối cùng, client đã mất yêu cầu đang thực hiện đó. Nó có thể đưa ra một yêu cầu mới với id JSON-RPC mới khi việc thử lại là an toàn. Nó không được cố gắng tiếp tục luồng.

### Xác thực Origin

Máy chủ xác thực `Origin` trên các kết nối đến để ngăn chặn DNS rebinding. Nếu header có mặt và không được cho phép rõ ràng, hãy trả về `403 Forbidden`. Một client không phải trình duyệt có thể bỏ qua `Origin`, điều mà các quy tắc transport chính thức cho phép.

Các máy chủ cục bộ nên bind vào `127.0.0.1`, không phải mọi giao diện. Các dịch vụ mạng vẫn cần xác thực và ủy quyền trên mọi yêu cầu. Xác thực Origin không phải là xác thực danh tính.

Sử dụng khớp chính xác origin sau khi cấu hình chuẩn hóa. Các kiểm tra tiền tố như `origin.startswith("https://trusted.example")` là không an toàn vì chúng có thể chấp nhận các hậu tố do kẻ tấn công kiểm soát.

### Các HTTP metadata header bắt buộc

Mỗi yêu cầu POST hiện đại bao gồm:

```http
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes_search
```

Quy tắc header:

- `MCP-Protocol-Version` là bắt buộc và phải bằng `params._meta.io.modelcontextprotocol/protocolVersion`.
- `Mcp-Method` là bắt buộc và phải bằng `method` của JSON-RPC.
- `Mcp-Name` là bắt buộc đối với `tools/call`, `resources/read` và `prompts/get`.
- `Mcp-Name` bằng `params.name`, hoặc `params.uri` cho `resources/read`.
- Các giá trị header phân biệt chữ hoa chữ thường mặc dù tên header thì không.

Các giá trị `Mcp-Name` không an toàn hoặc không phải ASCII sử dụng sentinel Base64 UTF-8 chính xác:

```text
=?base64?{Base64EncodedValue}?=
```

Máy chủ giải mã giá trị đó trước khi so sánh với phần thân.

Các header đối chiếu bị thiếu, sai định dạng hoặc không khớp sẽ trả về HTTP `400` với mã JSON-RPC `-32020`. Nếu header và phần thân đồng ý về một phiên bản mà máy chủ không hỗ trợ, hãy trả về HTTP `400` với `-32022` và dữ liệu lỗi chính xác như `{"supported":["2026-07-28"],"requested":"2027-01-01"}`.

Một phương thức hiện đại không xác định trả về HTTP `404` với JSON-RPC `-32601`. Phần thân JSON-RPC rất quan trọng vì một client đa thời đại sử dụng nó để phân biệt lỗi hiện đại với lỗi endpoint cũ.

### SSE theo phạm vi yêu cầu

Máy chủ có thể chọn SSE cho một yêu cầu chạy dài:

```text
POST tools/call id=41
  <- notifications/progress related to id=41
  <- notifications/progress related to id=41
  <- JSON-RPC response id=41
stream closes
```

Máy chủ không được gửi các yêu cầu JSON-RPC độc lập trên luồng này. Các tương tác lấy mẫu, gợi ý và gốc sử dụng kết quả Yêu cầu Đa vòng (Multi Round-Trip Request). Việc đóng luồng phản hồi sẽ hủy yêu cầu đó.

Không thêm id sự kiện SSE để phát lại. Việc tiếp tục `Last-Event-ID` không phải là một phần của bản sửa đổi hiện đại.

### Các thay đổi tồn tại lâu dài sử dụng subscriptions/listen

Các thông báo thay đổi sử dụng một yêu cầu do client mở, không phải GET độc lập:

```json
{
  "jsonrpc": "2.0",
  "id": "listen-1",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true,
      "resourceSubscriptions": ["notes://note-1"]
    },
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

Phản hồi POST là một luồng SSE tồn tại lâu dài. Thông điệp giao thức đầu tiên của nó là `notifications/subscriptions/acknowledged`. Việc xác nhận, mọi thông báo thay đổi và kết quả cuối cùng mang `io.modelcontextprotocol/subscriptionId` trong `_meta`, bằng với id yêu cầu listen. Máy chủ có thể phát ra các bình luận SSE như là keepalive. Khi luồng bị ngắt, client phát hành lại `subscriptions/listen` với một id yêu cầu mới và tìm nạp lại dữ liệu bị ảnh hưởng.

`resources/subscribe` và `resources/unsubscribe` thuộc về kỷ nguyên cũ. Không sử dụng chúng trên một kết nối hiện đại.

### Trạng thái ứng dụng rõ ràng

Việc loại bỏ các phiên giao thức không cấm các quy trình làm việc có trạng thái. Máy chủ có thể tạo một handle trạng thái mờ và trả về nó như một kết quả công cụ bình thường. Client truyền handle đó như một đối số rõ ràng trong các lệnh gọi sau này.

Bind các handle vào chủ thể đã xác thực, làm cho chúng không thể đoán trước, hết hạn chúng và ủy quyền mọi lần sử dụng. Điều này làm cho trạng thái hiển thị ở lớp ứng dụng thay vì ẩn nó trong sự gắn kết transport.

Lỗi do trạng thái bản sao ẩn gây ra là về mặt cơ học:

1. Yêu cầu A đến bản sao 1 và tạo một bản nháp trong bộ nhớ của tiến trình đó.
2. Phản hồi không trả về handle bản nháp vì việc triển khai giả định kết nối xác định bản nháp.
3. Yêu cầu B là một POST mới và đến bản sao 2.
4. Bản sao 2 có metadata giao thức hợp lệ nhưng không có cách nào để đặt tên hoặc tải bản nháp, vì vậy quy trình làm việc thất bại hoặc đọc sai đối tượng cục bộ.
5. Định tuyến dính (sticky routing) có vẻ khắc phục triệu chứng cho đến khi khởi động lại, triển khai, lập lịch lại hoặc chuyển đổi dự phòng di chuyển yêu cầu tiếp theo.

Ranh giới đúng có hai phần. Ngữ cảnh giao thức nằm trong mỗi yêu cầu. Trạng thái ứng dụng bền vững nằm trong một kho lưu trữ chia sẻ dưới một handle do máy chủ tạo ra được trả về cho client. Lệnh gọi tiếp theo cung cấp handle đó, bất kỳ bản sao nào cũng tải cùng một bản ghi và việc ủy quyền ràng buộc bản ghi đó với chủ thể và tenant đã xác thực. Bộ nhớ bản sao có thể lưu trữ bản ghi, nhưng nó không thể là bản sao duy nhất cần thiết cho tính chính xác.

Chọn cơ chế trạng thái theo thời gian tồn tại. Các biến cục bộ yêu cầu có thể phục vụ một lệnh gọi. Một sự tiếp nối MRTR ngắn có thể sử dụng `requestState` được bảo vệ tính toàn vẹn. Một bản nháp hoặc tác vụ bền vững cần một handle rõ ràng cộng với tính bền vững chia sẻ, hết hạn, kiểm soát đồng thời và tính lũy đẳng. Không đối tượng nào trong số đó là một phiên giao thức MCP.

### Khả năng tương thích đa thời đại HTTP

Một client hỗ trợ cả máy chủ hiện đại và cũ sẽ thử POST hiện đại trước. Nếu nhận được HTTP `400`, `404` hoặc `405`, nó sẽ kiểm tra phần thân:

- Một lỗi JSON-RPC hiện đại được nhận dạng chứng minh máy chủ là hiện đại. Hãy sửa yêu cầu hoặc thử lại phiên bản được quảng bá. Không hạ cấp.
- Một phần thân trống hoặc phản hồi không được nhận dạng có thể chỉ ra một máy chủ HTTP+SSE cũ. Chỉ khi đó mới thử endpoint GET cũ và mong đợi sự kiện `endpoint` cũ của nó.

Máy chủ có thể hỗ trợ cả hai kỷ nguyên trong quá trình di chuyển bằng cách định tuyến metadata hiện đại đến triển khai chỉ POST hiện đại và giữ lại các endpoint cũ riêng biệt cho các client cũ. Không bao giờ mô tả hành vi GET, DELETE, id phiên hoặc phát lại cũ như là một phần của `2026-07-28`.

```figure
tp-transport-handshake
```

## Sử dụng

`code/main.py` triển khai một máy chủ Streamable HTTP hiện đại, hữu hạn với thư viện chuẩn Python. Nó xác thực Origin và các header đối chiếu, bỏ qua các header phiên đã loại bỏ, trả về JSON cho các lệnh gọi bình thường và minh họa một luồng SSE `subscriptions/listen` hữu hạn.

```bash
cd code
python3 main.py --probe
python3 -m unittest discover tests -v
```

Công cụ kiểm tra:

- Origin không hợp lệ bị từ chối;
- Khám phá thành công mà không cần id phiên;
- `Mcp-Session-Id` và `Last-Event-ID` bị bỏ qua;
- Header không khớp trả về `-32020`;
- Phiên bản không được hỗ trợ trả về `-32022` với dữ liệu `supported` và `requested` chính xác;
- Một thông báo không id được chấp nhận trả về HTTP `202` không có phần thân;
- GET và DELETE trả về `405`;
- `subscriptions/listen` là một luồng phản hồi POST mà việc xác nhận, thông báo và kết quả cuối cùng của nó mang id đăng ký của nó.

## Phát hành

Bài học này phát hành `outputs/skill-mcp-transport-migrator.md`. Nó loại bỏ các phiên giao thức hiện đại, thêm xác thực header-phần thân, thay thế GET độc lập bằng `subscriptions/listen` và giữ cho bất kỳ cầu nối cũ nào tách biệt rõ ràng.

## Bài tập

1. Xóa `Mcp-Method` khỏi một POST. Xác nhận HTTP `400` và lỗi `-32020`.
2. Gửi header và phiên bản phần thân khớp nhau `2027-01-01`. Xác nhận HTTP `400`, lỗi `-32022` và dữ liệu `{"supported":["2026-07-28"],"requested":"2027-01-01"}` chính xác.
3. Gửi một sentinel Base64 `Mcp-Name` cho một URI tài nguyên không phải ASCII. Xác nhận giá trị đã giải mã được so sánh với `params.uri`.
4. Ngắt luồng listen hữu hạn trước phản hồi cuối cùng của nó. Phát hành lại nó với một id JSON-RPC mới và tìm nạp lại các công cụ.
5. Thêm một handle quy trình làm việc rõ ràng vào công cụ ping. Bind nó vào một chủ thể ủy quyền mà không sử dụng sự gắn kết kết nối.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa |
|------|---------|
| stdio | JSON-RPC phân cách bằng dòng mới qua tiến trình con do client khởi chạy |
| Streamable HTTP | Endpoint duy nhất nơi mỗi thông điệp hiện đại là một POST mới |
| Request-scoped SSE | Luồng phản hồi POST chứa các thông báo liên quan và phản hồi cuối cùng |
| `subscriptions/listen` | Yêu cầu POST tồn tại lâu dài cho các thông báo thay đổi đã chọn |
| Header mismatch | HTTP `400` và JSON-RPC `-32020` khi các header đối chiếu không khớp với phần thân |
| Origin validation | Phòng thủ DNS-rebinding cho các kết nối đến, không phải xác thực |
| Explicit state handle | Token ứng dụng được truyền như một đối số thông thường thay vì trạng thái phiên ẩn |
| Legacy bridge | Hành vi kỷ nguyên trước tách biệt chỉ được giữ lại để tương thích |

## Đọc thêm

- [MCP Transport Overview](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)