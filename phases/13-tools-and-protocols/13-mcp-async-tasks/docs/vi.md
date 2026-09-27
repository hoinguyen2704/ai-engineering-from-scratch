# Tiện ích mở rộng MCP Tasks: Công việc bền vững trên lõi không trạng thái (Stateless)

> MCP không trạng thái (stateless) không có nghĩa là mọi thao tác phải hoàn thành trong một yêu cầu duy nhất. Tiện ích mở rộng Tasks chính thức cung cấp một định danh bền vững (durable handle) cho các công việc chạy dài. Một server có thể trả về định danh đó từ `tools/call`, bất kỳ instance nào cũng có thể phản hồi `tasks/get`, và dữ liệu đầu vào từ client được gửi qua `tasks/update` mà không cần khôi phục các phiên giao thức (protocol sessions).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 11 (stateless MRTR), Phase 13 · 12 (elicitation)
**Time:** ~90 phút

## Mục tiêu học tập

- Phân biệt giữa transport giao thức không trạng thái và trạng thái công việc (task state) bền vững của ứng dụng.
- Đàm phán tiện ích mở rộng `io.modelcontextprotocol/tasks` trong các capabilities theo yêu cầu và `server/discover`.
- Trả về `CreateTaskResult` do server chỉ định với `resultType: "task"` chỉ sau khi đã tạo xong dữ liệu bền vững.
- Poll bằng `tasks/get`, cung cấp dữ liệu đầu vào cho task bằng `tasks/update`, và yêu cầu hủy bỏ hợp tác (cooperative cancellation) bằng `tasks/cancel`.
- Loại bỏ các giả định cũ về `tasks/status`, `tasks/result`, và `tasks/list`.
- Đăng ký nhận thông báo task tùy chọn thông qua `subscriptions/listen` trên luồng SSE phản hồi POST.
- Mô hình hóa đúng cách thời hạn task (expiry), khôi phục sau khi khởi động lại, khử trùng lặp khóa đầu vào (input-key deduplication), và các lỗi thực thi.

## Tại sao Tasks là một tiện ích mở rộng

Tasks lần đầu xuất hiện như một tính năng lõi thử nghiệm vào ngày 25-11-2025. Bản thiết kế lại tháng 7 năm 2026 chuyển chúng vào tiện ích mở rộng `io.modelcontextprotocol/tasks` chính thức để client và server có thể chọn tham gia vào vòng đời bổ sung mà không cần mở rộng giao thức lõi cho tất cả mọi người.

Đặc tả tiện ích mở rộng vẫn là một bản nháp dù đây là nơi chính thức hiện tại cho Tasks. Hãy ghim phiên bản tiện ích mở rộng được SDK của bạn hỗ trợ, chạy các kịch bản tuân thủ, và tách biệt các bộ điều hợp wire (wire adapters) khỏi domain worker và lưu trữ của bạn.

Sử dụng task khi thao tác có một hoặc nhiều thuộc tính sau:

- Nó có thể kéo dài hơn thời gian chờ (timeout) của một yêu cầu thông thường.
- Một hàng đợi worker hoặc hệ thống job bên ngoài đã sở hữu việc thực thi.
- Client cần khôi phục sau khi chính nó khởi động lại.
- Thao tác tạm dừng để chờ đầu vào từ người dùng hoặc model trong khi thực thi.
- Hủy bỏ và truy xuất kết quả bền vững là yêu cầu của sản phẩm.

Đừng tạo task cho một tra cứu xác định (deterministic lookup) đơn giản. Định danh, tính bền vững, polling, thời hạn và hủy bỏ là những độ phức tạp thực sự.

## Lõi không trạng thái, Ứng dụng có trạng thái

MCP 2026-07-28 loại bỏ `initialize`, `notifications/initialized`, các phiên giao thức, và `Mcp-Session-Id`. Điều đó không cấm các sản phẩm có trạng thái.

Một task id là trạng thái ứng dụng rõ ràng:

- Server lưu trữ nó trước khi trả về.
- Client có thể lưu trữ nó và poll lại sau khi khởi động lại.
- Id có thể định tuyến đến bất kỳ bản sao (replica) nào được hỗ trợ bởi cùng một kho lưu trữ bền vững.
- Quyền hạn được kiểm tra trên mọi phương thức task.
- Thời hạn và xóa được xác định bởi các trường của task, không phải thời gian tồn tại của transport.

Điều này khác biệt về mặt vận hành so với trạng thái ẩn gắn liền với một kết nối.

Giữ bốn vòng đời riêng biệt:

| Trạng thái | Vòng đời | Thuộc về đâu |
|---|---|---|
| Metadata giao thức | Một yêu cầu | `params._meta`, được xác thực lại trên mỗi lần gọi |
| Công việc transport | Một yêu cầu stdio hoặc phản hồi HTTP | Điều phối viên đang thực thi với thời hạn giới hạn |
| Tiếp nối MRTR | Một chuỗi thử lại | `requestState` được bảo vệ tính toàn vẹn, cộng với các điều khiển phát lại khi cần |
| Task bền vững | Qua các yêu cầu, bản sao, khởi động lại và kết nối lại | Kho lưu trữ ứng dụng chia sẻ được khóa bởi một `taskId` đã xác thực |

Việc di chuyển bản ghi task vào bộ nhớ tiến trình không làm cho MCP trở nên có trạng thái. Nó làm cho ứng dụng trở nên không đáng tin cậy. Giao thức vẫn không trạng thái, nhưng một `tasks/get` sau đó được định tuyến đến một bản sao khác không thể khôi phục bản ghi. Hãy lưu trữ trước khi trả về định danh, sau đó làm cho mọi phương thức task giải quyết cùng một bản ghi chia sẻ dưới các kiểm tra về tenant và principal.

## Đàm phán Capability

Client quảng bá hỗ trợ trên mọi yêu cầu đủ điều kiện:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "extensions": {
        "io.modelcontextprotocol/tasks": {}
      }
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "lesson-client",
      "version": "1.0.0"
    }
  }
}
```

Server trả về chính xác `supportedVersions`, các capabilities, `ttlMs`, và `cacheScope` từ `server/discover`, với cùng tiện ích mở rộng trong capabilities. Vì nó quảng bá các công cụ, nó cũng triển khai `tools/list` bắt buộc. Kết quả đó trả về một bộ mô tả `generate_report` xác định, đối tượng hợp lệ `inputSchema`, `resultType: "complete"`, metadata định danh server, và các gợi ý cache công khai.

Một phương thức task từ client không khai báo tiện ích mở rộng sẽ trả về `-32021`, Missing Required Client Capability, với `data.requiredCapabilities` được đặt thành `{"extensions":{"io.modelcontextprotocol/tasks":{}}}`. Một chuỗi giao thức không được hỗ trợ trả về `-32022` với dữ liệu `supported` và `requested` chính xác; một phiên bản bị thiếu hoặc không phải chuỗi sẽ trả về `-32602`.

Một phong bì (envelope) không có `id` JSON-RPC là một thông báo. Người nhận có thể xử lý nó, nhưng nó không phát ra kết quả hoặc lỗi JSON-RPC. Một bộ điều hợp HTTP Streamable trả về `202 Accepted` không có nội dung cho một thông báo được chấp nhận.

Hiện tại, chỉ `tools/call` hỗ trợ thực thi tăng cường task. Hãy thiết kế trừu tượng nội bộ của bạn để các loại yêu cầu trong tương lai không yêu cầu viết lại bộ lưu trữ.

## Tạo Task do Server chỉ định

Cờ client cũ `params._meta.task.required` đã không còn. Client khai báo hỗ trợ tiện ích mở rộng, sau đó server quyết định xem một `tools/call` cụ thể có trở thành task hay không.

Yêu cầu:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "generate_report",
    "arguments": {"size": "large"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Phản hồi:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "task",
    "taskId": "tsk_786512e29e0d",
    "status": "working",
    "statusMessage": "Preparing report outline.",
    "createdAt": "2026-08-21T10:30:00Z",
    "lastUpdatedAt": "2026-08-21T10:30:00Z",
    "ttlMs": 900000,
    "pollIntervalMs": 1000
  }
}
```

Server không được trả về định danh này cho đến khi một `tasks/get` cho id đó có thể giải quyết. Trong một kho lưu trữ nhất quán cuối cùng (eventually consistent), hãy đợi khả năng đọc hiển thị trước khi trả lời. Nếu không, client có thể nhận được một id trông có vẻ hợp lệ và ngay lập tức nhận được lỗi "not found".

Phản hồi task là không được yêu cầu theo nghĩa client không yêu cầu chế độ task. Nó không phải là không được đàm phán: yêu cầu hiện tại vẫn phải quảng bá tiện ích mở rộng.

## Hình dạng của Task

Mỗi task mang theo:

- `taskId`: định danh ổn định do server tạo ra;
- `status`: `working`, `input_required`, `completed`, `cancelled`, hoặc `failed`;
- `createdAt` và `lastUpdatedAt`: dấu thời gian ISO 8601;
- `ttlMs`: thời hạn từ khi tạo, hoặc `null` nếu không có giới hạn quảng bá;
- `pollIntervalMs` tùy chọn: nhịp độ polling tối thiểu được đề xuất hiện tại của server;
- `statusMessage` tùy chọn: ngữ cảnh dành cho người dùng hoặc model.

Các trường cụ thể theo trạng thái chỉ xuất hiện khi liên quan:

- `input_required` bao gồm `inputRequests`.
- `completed` bao gồm hình dạng `result` của yêu cầu gốc.
- `failed` bao gồm một đối tượng `error` JSON-RPC.

Client nên tôn trọng `pollIntervalMs`. Server có thể giới hạn tốc độ polling mạnh hơn và có thể thay đổi khoảng thời gian trong suốt vòng đời của task.

## Poll với `tasks/get`

Client yêu cầu một snapshot hiện tại:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/get
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tasks/get",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Bản thân `tasks/get` đã hoàn thành, vì vậy kết quả của nó luôn có `resultType: "complete"`. Task lồng nhau vẫn có thể có `status: "working"` hoặc `status: "input_required"`.

Sự phân biệt này ngăn chặn một lỗi parser phổ biến:

```text
result.resultType = complete    means the tasks/get RPC finished
result.status = working        means the represented job is still running
```

Không có lệnh gọi `tasks/result`. Khi task hoàn thành, phản hồi `tasks/get` tiếp theo sẽ in-line `CallToolResult` gốc dưới `result`:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "completed",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:34:12Z",
  "ttlMs": 900000,
  "result": {
    "resultType": "complete",
    "content": [
      {"type": "text", "text": "Generated large report with approved outline."}
    ],
    "structuredContent": {"size": "large", "approved": true},
    "isError": false,
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "tasks-demo",
        "version": "1.0.0"
      }
    }
  },
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "tasks-demo",
      "version": "1.0.0"
    }
  }
}
```

`resultType` bên ngoài cho biết RPC `tasks/get` đã hoàn thành. `result.resultType` lồng nhau cho biết lệnh gọi công cụ gốc đã hoàn thành. Bộ phân biệt lồng nhau đó là bắt buộc. `CallToolResult` lồng nhau CŨNG NÊN mang theo `io.modelcontextprotocol/serverInfo` của riêng nó; bài học này bao gồm nó thay vì lưu trữ một payload không định kiểu.

Không có `tasks/list`. Các server không phiên (sessionless) không thể suy luận một cách an toàn task nào thuộc về danh sách phạm vi kết nối. Các ứng dụng cần lịch sử nên hiển thị một công cụ domain được ủy quyền với các bộ lọc và quy tắc sở hữu rõ ràng.

## Đầu vào trong khi thực thi Task

Đầu vào task và MRTR lõi trông giống nhau nhưng sử dụng các phần tiếp nối (continuations) khác nhau.

### Đầu vào cần thiết trước khi tạo task

Trả về `resultType: "input_required"` lõi từ `tools/call` gốc. Client hoàn thành nó và thử lại lệnh gọi gốc đó. Chỉ tạo task sau khi các vòng MRTR đồng bộ đó kết thúc.

### Đầu vào cần thiết sau khi tạo task

Đặt task thành `input_required`. `tasks/get` hiển thị `inputRequests` đang chờ xử lý, và client gửi phản hồi qua `tasks/update`. Client không thử lại `tools/call` gốc.

Snapshot:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "input_required",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:31:00Z",
  "ttlMs": 900000,
  "inputRequests": {
    "approve_outline": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Approve the generated report outline?",
        "requestedSchema": {
          "type": "object",
          "properties": {"approved": {"type": "boolean"}},
          "required": ["approved"]
        }
      }
    }
  }
}
```

Cập nhật:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/update
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tasks/update",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "inputResponses": {
      "approve_outline": {
        "action": "accept",
        "content": {"approved": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Phản hồi thành công là một xác nhận trống cộng với `resultType: "complete"`. Thay đổi trạng thái có thể nhất quán cuối cùng, vì vậy client tiếp tục polling hoặc lắng nghe.

Mỗi khóa `inputRequests` phải là duy nhất cho toàn bộ vòng đời task. Các snapshot `tasks/get` lặp lại có thể hiển thị cùng một khóa đang chờ xử lý; client khử trùng lặp UI và server bỏ qua các phản hồi cho các khóa không xác định, bị thay thế hoặc đã được thực hiện. Một bản cập nhật một phần có thể để task ở trạng thái `input_required` cho đến khi tất cả các khóa bắt buộc được trả lời.

## Hủy bỏ là sự hợp tác

`tasks/cancel` báo hiệu ý định và trả về một xác nhận hoàn thành trống. Xác nhận đó không đảm bảo worker đã dừng lại. Công việc có thể kết thúc trước, bỏ qua việc hủy bỏ, hoặc chuyển đổi sau đó.

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/cancel
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tasks/cancel",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Đối với cả ba phương thức task, `Mcp-Name` phản chiếu `params.taskId`. Nó không lặp lại tên phương thức JSON-RPC. `code/main.py` tập trung quy tắc này trong `make_http_request`.

Worker trong bài học tôn trọng việc hủy bỏ ngay lập tức, làm cho các lệnh gọi lặp lại trở nên idempotent. Một client sản xuất vẫn phải coi việc hủy bỏ là hợp tác thay vì suy luận trạng thái task cuối cùng từ xác nhận.

Đừng sử dụng `notifications/cancelled` để hủy một task. Thông báo đó thuộc về việc hủy yêu cầu, không phải Tasks bền vững.

Sự phân biệt này quan trọng tại ranh giới định tuyến. Hủy yêu cầu nhắm mục tiêu vào một thao tác JSON-RPC đang thực thi hoặc phản hồi HTTP phạm vi yêu cầu của nó. Nếu `tools/call` đã trả về `resultType: "task"`, yêu cầu đó đã hoàn tất và việc đóng transport của nó không thể đặt tên hoặc dừng công việc bền vững. `tasks/cancel` là một RPC được ủy quyền mới. Nó mang theo `params.taskId`, phản chiếu id đó trong `Mcp-Name`, giải quyết backend sở hữu task, ghi lại ý định hủy bỏ hợp tác, và trả về xác nhận mà không khẳng định worker đã dừng.

Do đó, một gateway phải giữ các bộ điều phối yêu cầu và các tuyến đường task trong các bảng khác nhau. Bảng yêu cầu có thể biến mất khi phản hồi kết thúc. Tuyến đường task phải tồn tại cho đến trạng thái cuối cùng và hết hạn lưu giữ. [Bài học 29: Độ tin cậy, Hủy bỏ và Kiểm soát luồng MCP](../../29-mcp-reliability-cancellation-and-flow-control/docs/en.md) xây dựng các quy tắc về cuộc đua, timeout, idempotency, backpressure và thử lại cho cả hai đường dẫn.

## Thông báo tùy chọn

Polling là cơ sở. Client muốn cập nhật đẩy (push updates) sẽ gửi `subscriptions/listen` với các task id. Đối với HTTP Streamable, đây là một POST mà phản hồi của nó là một luồng SSE phạm vi yêu cầu. Không có luồng sự kiện GET độc lập và không có phiên giao thức nào để duy trì.

Server xác nhận các id được chấp nhận với `notifications/subscriptions/acknowledged` và sau đó có thể gửi các snapshot đầy đủ qua `notifications/tasks`. Xác nhận và mọi thông báo task mang theo `io.modelcontextprotocol/subscriptionId` trong `_meta`, bằng với id yêu cầu `subscriptions/listen`. Mỗi thông báo task khác tương đương với những gì `tasks/get` sẽ trả về tại thời điểm đó.

Client vẫn phải khai báo tiện ích mở rộng Tasks. Họ nên kết nối lại và tiếp tục từ các task id bền vững thay vì phụ thuộc vào phát lại sự kiện hoặc `Last-Event-ID`.

## Ngữ nghĩa lỗi

Sử dụng hai lớp lỗi một cách chính xác.

### Lỗi giao thức

Các tham số phương thức không hợp lệ hoặc một task id không xác định trả về lỗi JSON-RPC, thường là `-32602`. Thiếu hỗ trợ tiện ích mở rộng trả về `-32021` với đối tượng capability bắt buộc.

### Kết quả thực thi Task

- Một kết quả công cụ bình thường với `isError: true` vẫn là một task `completed` vì lệnh gọi công cụ đã tạo ra kết quả được xác định của nó.
- Một lỗi JSON-RPC trong quá trình thực thi trì hoãn làm cho task trở thành `failed` và lưu trữ lỗi JSON-RPC đó dưới `error`.
- Sự từ chối của người dùng có thể tạo ra `cancelled`, một kết quả từ chối đã hoàn thành, hoặc một kết quả an toàn cụ thể theo domain khác. Hãy ghi lại lựa chọn đó.

## Độ bền, Thời hạn và Quyền sở hữu

Lưu trữ ít nhất task id, trạng thái, dấu thời gian, ttl, khoảng thời gian poll, quyền sở hữu thao tác gốc, kết quả hoặc lỗi, các yêu cầu đầu vào đang chờ xử lý, và tất cả các khóa đầu vào đã cấp.

Khóa lưu trữ phải bao gồm hoặc giải quyết một tenant và principal có thẩm quyền. Biết một task id không được cấp quyền truy cập. Kiểm tra quyền sở hữu trên mỗi `tasks/get`, `tasks/update`, `tasks/cancel`, và đăng ký.

`ttlMs` được đo từ khi tạo và có thể thay đổi. Client có thể coi nó như một điểm dừng khi một task đã ngừng tạo ra các cập nhật có thể quan sát được. Server có thể thất bại và sau đó xóa một task đã hết hạn. Đừng mô tả nó như một lời hứa giữ lại kết quả đã hoàn thành trong nhiều mili giây sau khi hoàn thành.

Sử dụng các ghi chép nguyên tử (atomic writes) hoặc giao dịch. Bài học ghi vào một tệp tạm thời và đổi tên nó một cách nguyên tử. Một dịch vụ đa bản sao nên sử dụng kho lưu trữ bền vững chia sẻ và một lease worker hoặc kiểm soát đồng thời tương đương.

```figure
tp-task-lifecycle
```

## Xây dựng nó

`code/main.py` triển khai một dịch vụ task xác định:

- `server/discover` trả về `supportedVersions`, gợi ý cache, và tiện ích mở rộng Tasks.
- `tools/list` trả về một bộ mô tả `generate_report` xác định, có thể cache với một lược đồ đầu vào hợp lệ.
- `tools/call` tạo và lưu trữ task trước khi trả về `resultType: "task"`.
- Một instance dịch vụ mới tải lại cùng một task, chứng minh khả năng khôi phục sau khi khởi động lại.
- `tasks/get` trả về các snapshot task đầy đủ.
- Worker di chuyển từ `working` sang `input_required`.
- `tasks/update` chấp nhận một phản hồi biểu mẫu và trả về một xác nhận hoàn thành trống.
- Worker lưu trữ một `CallToolResult` lồng nhau với `resultType` và định danh server của riêng nó, sau đó chuyển sang `completed`.
- `tasks/cancel` là idempotent trong triển khai này.
- Trình xây dựng HTTP đặt `Mcp-Name` thành `params.taskId` cho `tasks/get`, `tasks/update`, và `tasks/cancel`.
- Các trình trợ giúp thông báo sử dụng `notifications/subscriptions/acknowledged` và `notifications/tasks`, cả hai đều được gắn thẻ với id yêu cầu lắng nghe.
- Các thông báo không có id không tạo ra phản hồi JSON-RPC.

Worker tiến triển một cách rõ ràng thay vì ngủ trong một luồng nền. Điều đó làm cho mọi chuyển đổi trạng thái trở nên xác định và giữ cho ví dụ giao thức tách biệt khỏi cơ chế hàng đợi.

## Sử dụng nó

Từ thư mục gốc của repository:

```bash
cd phases/13-tools-and-protocols/13-mcp-async-tasks/code
python3 main.py
python3 -m unittest discover tests -v
```

Chuỗi kết quả mong đợi:

```text
id=0 resultType=complete status=ack
id=1 resultType=task status=working
id=2 resultType=complete status=working
id=3 resultType=complete status=input_required
id=4 resultType=complete status=ack
id=5 resultType=complete status=completed
```

Cũng xác minh rằng `tasks/status`, `tasks/result`, và `tasks/list` trả về method-not-found trong dịch vụ hiện đại.
Xác minh rằng `tools/list` là xác định và mọi phương thức task HTTP hiện tại đều phản chiếu task id của nó thông qua `Mcp-Name`.

## Gửi nó

`outputs/skill-task-store-designer.md` hiện tạo ra một thiết kế nhận thức tiện ích mở rộng: đàm phán capability, tạo bền vững-trước-khi-trả-về, các phương thức hiện tại, luồng cập nhật đầu vào, quyền sở hữu, thời hạn, hủy bỏ, đăng ký, và di chuyển từ các phương thức thử nghiệm đã bị loại bỏ.

## Bài tập

1. Thêm khóa đầu vào thứ hai đang chờ xử lý. Gửi một `tasks/update` một phần và chứng minh task vẫn ở trạng thái `input_required` cho đến khi cả hai khóa được trả lời.
2. Thêm quyền sở hữu tenant vào kho lưu trữ và từ chối một task id hợp lệ được trình bày bởi principal xác thực sai.
3. Thêm một lease worker với thời hạn. Chứng minh rằng hai instance dịch vụ không thể hoàn thành cùng một task đồng thời.
4. Triển khai bộ điều hợp SSE phản hồi POST cho `subscriptions/listen`. Không thêm GET, `Last-Event-ID`, hoặc tiêu đề phiên.
5. Thêm dọn dẹp hết hạn. Phân biệt một task đã hết hạn với một task id bị định dạng sai mà không làm rò rỉ sự tồn tại giữa các tenant.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa trong tiện ích mở rộng hiện tại |
|------|----------------------------------|
| Tiện ích mở rộng Tasks | Capability `io.modelcontextprotocol/tasks` tùy chọn cho công việc async bền vững |
| `CreateTaskResult` | Phản hồi `resultType: "task"` do server chỉ định cho một yêu cầu đủ điều kiện |
| `tasks/get` | Poll một snapshot task hiện tại đầy đủ, bao gồm kết quả cuối cùng hoặc đầu vào đang chờ xử lý |
| `tasks/update` | Gửi phản hồi cho `inputRequests` đang chờ xử lý của task |
| `tasks/cancel` | Xác nhận ý định hủy bỏ hợp tác |
| `input_required` | Trạng thái task cho biết đầu vào client đang chờ xử lý |
| `pollIntervalMs` | Độ trễ tối thiểu do server đề xuất trước khi poll lần tiếp theo |
| `ttlMs` | Thời hạn được đo từ khi tạo task |
| Bền vững-trước-khi-trả-về | Quy tắc rằng task id phải giải quyết trước khi handle của nó được gửi |
| `notifications/tasks` | Snapshot task đầy đủ tùy chọn được gửi trên phản hồi SSE đã đăng ký |

## Khả năng tương thích cũ

Bề mặt thử nghiệm 25-11-2025 đã sử dụng tăng cường task do client yêu cầu, `tasks/status`, `tasks/result`, và `tasks/list` tùy chọn. Chỉ giữ những tên đó bên trong một bộ điều hợp cũ đã ghim. Một client hiện tại sử dụng capability tiện ích mở rộng, chấp nhận các handle do server chỉ định, poll `tasks/get`, cung cấp đầu vào với `tasks/update`, và đọc kết quả cuối cùng từ snapshot task.

## Đọc thêm

- [Tiện ích mở rộng MCP Tasks chính thức](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
- [MCP 2026-07-28 Yêu cầu đa vòng (Multi Round-Trip Requests)](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 HTTP có thể truyền phát (Streamable HTTP)](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)