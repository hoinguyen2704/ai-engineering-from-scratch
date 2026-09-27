# Stateless MCP Gateways và Registry Admission

> Một gateway cần làm cho mọi route trở nên tường minh. Giao thức 2026-07-28 cung cấp cho nó các ranh giới về phương thức, tên, phiên bản, khả năng, định danh, bộ nhớ đệm và dấu vết mà không cần phiên truyền tải (transport session).

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13 · 15 (security), Phase 13 · 16 (authorization)
**Time:** ~75 phút

## Mục tiêu học tập

- Tổng hợp nhiều MCP server phía sau một endpoint 2026-07-28 mà không cần session affinity.
- Xác thực metadata và routing header của từng request trước khi áp dụng chính sách hoặc chuyển tiếp.
- Hợp nhất các tool với namespace ổn định, thứ tự xác định, descriptor pin, RBAC và bộ nhớ đệm riêng tư.
- Coi các bản ghi registry là bằng chứng khám phá (discovery evidence) nhưng vẫn yêu cầu chính sách admission.
- Định tuyến chính xác SSE theo phạm vi request, `subscriptions/listen`, các lần thử lại MRTR và các lệnh gọi extension Tasks.
- Cô lập các hỗ trợ handshake và session cũ khỏi đường dẫn hiện đại.

## Vấn đề

Kết nối một client trực tiếp với một server rất đơn giản. Một triển khai lớn hơn cần câu trả lời nhất quán cho những câu hỏi khó hơn:

- Những server nào được phép?
- Principal nào có thể xem và gọi từng tool?
- Điều gì xảy ra khi hai backend cùng hiển thị một tên?
- Các thay đổi descriptor được xem xét như thế nào?
- Giới hạn tốc độ (rate limit) và sự kiện kiểm toán (audit event) được áp dụng ở đâu?
- Bất kỳ instance nào có thể xử lý request tiếp theo không?

Một gateway nằm giữa client và các backend MCP server. Nó trình bày một MCP endpoint duy nhất, áp dụng chính sách xuyên suốt và chuyển tiếp các request đã được phê duyệt.

Các thiết kế gateway cũ thường ghép kênh (multiplex) một session client thành nhiều session backend và viết lại `Mcp-Session-Id`. Đó là thiết kế tương thích kế thừa. Core 2026-07-28 không có session giao thức.

## Khái niệm

### Đường dẫn gateway hiện đại

Đối với mỗi request:

1. Xác thực principal từ authorization của transport.
2. Xác thực `MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name` và `params._meta`.
3. Ủy quyền cho principal, resource, phương thức, tool và các đối số.
4. Áp dụng chính sách về descriptor, registry, tốc độ và dữ liệu.
5. Tạo một request mới độc lập cho backend đã chọn.
6. Xác thực kết quả từ backend và trả về kết quả của gateway.
7. Ghi lại sự kiện kiểm toán mà không ghi log các bí mật.

Không bước nào cần một session giao thức ẩn. Trạng thái ứng dụng vẫn có thể tồn tại trong cơ sở dữ liệu, các handle tường minh, Tasks hoặc trạng thái MRTR được bảo vệ tính toàn vẹn.

### Chính sách runtime là quyết định chính của gateway

Admission quyết định phiên bản backend nào có thể đi vào gateway. Nó không ủy quyền cho một lệnh gọi trực tiếp. Đối với mỗi request, gateway tính toán lại chính sách từ principal đã xác thực, issuer và resource, tenant, phương thức và tên khớp, các đối số đã chuẩn hóa, descriptor pin được chấp nhận, trạng thái sức khỏe backend hiện tại, sự giao thoa khả năng, phân loại dữ liệu, trạng thái tốc độ và bất kỳ sự phê duyệt ràng buộc hành động nào.

Thứ tự này rất quan trọng. Một bản ghi Registry có thể vẫn hoạt động trong khi vai trò của người dùng bị thu hồi. Một descriptor có thể vẫn được ghim trong khi đối số đích vượt qua ranh giới tenant. Một backend có thể vẫn được phê duyệt trong khi chính sách sự cố cách ly các lệnh gọi thay đổi trạng thái. Do đó, chính sách runtime là quyết định cho phép hoặc từ chối chính, với bằng chứng từ Registry và descriptor là các đầu vào.

Không lưu vào bộ nhớ đệm quyết định cho phép (allow) dựa trên kết nối hoặc định danh session đã bị xóa. Nếu chính sách không khả dụng, hãy tuân theo chính sách thất bại đã khai báo theo loại hoạt động. Mặc định an toàn là từ chối (fail closed) đối với các thay đổi trạng thái và các lệnh đọc nhạy cảm, trong khi các đường dẫn đọc công khai đã được phê duyệt rõ ràng có thể sử dụng chính sách cũ (last-known) trong thời gian ngắn chỉ khi mô hình rủi ro cho phép. Ghi lại phiên bản chính sách và đường dẫn thất bại đã đưa ra quyết định, sau đó xác thực kết quả backend trước khi trả về.

### Một endpoint POST duy nhất

Streamable HTTP hiện đại gửi từng thông điệp JSON-RPC qua POST:

```text
POST /mcp
Authorization: Bearer <gateway-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.search
Accept: application/json, text/event-stream
```

Gateway có thể trả về JSON hoặc SSE theo phạm vi request cho POST đó. GET và DELETE trả về 405 cho các request hiện đại. `Mcp-Session-Id` và `Last-Event-ID` không tạo ra thẩm quyền, affinity hoặc hành vi phát lại (replay).

Giá trị header và body phải khớp nhau. Từ chối sự không khớp với `-32020` trước khi tìm kiếm backend. Điều này cho phép các bộ cân bằng tải, gateway và bộ giới hạn tốc độ định tuyến mà không cần phân tích toàn bộ body trong khi vẫn bảo toàn tính toàn vẹn end-to-end.

Xác thực theo đúng một thứ tự: các loại JSON-RPC và metadata, sự bằng nhau của header và body, sau đó là hỗ trợ cho phiên bản khớp. Sự không khớp trả về HTTP 400 với `-32020`. Nếu header và body đồng ý về một phiên bản không được hỗ trợ, trả về HTTP 400 với `-32022` và `data` chính xác là `{"supported":["2026-07-28"],"requested":"<actual>"}`. Một phương thức không xác định trả về HTTP 404 với `-32601`.

`ProtocolError` mang theo `data` tùy chọn, và gateway tuần tự hóa nó vào đối tượng lỗi JSON-RPC. Một thông báo (notification) không có `id`, vì vậy nó không bao giờ nhận được thành công hoặc lỗi JSON-RPC. Một thông báo HTTP được chấp nhận trả về 202 với body trống.

### Triển khai discovery ở mọi lớp

Gateway triển khai `server/discover` cho các client. Nó cũng khám phá từng backend để biết các phiên bản giao thức, khả năng và extension.

Kết quả gateway ví dụ:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": true}
  },
  "ttlMs": 30000,
  "cacheScope": "private",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "enterprise-gateway",
      "version": "2.0.0"
    }
  }
}
```

Chỉ quảng bá sự giao thoa khả năng mà gateway có thể đảm bảo end-to-end. Một tính năng backend không tự động an toàn để hiển thị. Một tính năng gateway không có đường dẫn backend thì không hữu ích để quảng bá.

`serverInfo` là dữ liệu hiển thị và chẩn đoán tự báo cáo. Không sử dụng nó làm bằng chứng registry hoặc nhà xuất bản.

### Khả năng client theo từng request

Mỗi request được chuyển tiếp cần một phong bì `_meta` hiện tại:

```json
{
  "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientCapabilities": {},
  "io.modelcontextprotocol/clientInfo": {
    "name": "enterprise-gateway",
    "version": "1.0.0"
  }
}
```

Không sao chép mù quáng các khả năng của client bên ngoài vào backend. Gateway là client của backend. Chỉ quảng bá các tính năng mà gateway sẽ điều phối chính xác.

### Namespacing xác định

Hợp nhất các tool của backend dưới các tên công khai ổn định:

```text
notes.search
notes.create
issues.list
issues.open
```

Duy trì ánh xạ từ tên công khai đến backend và tên tool gốc. Không bao giờ chọn tên trùng lặp đầu tiên hoặc cuối cùng. Một tên công khai là một phần của hợp đồng phê duyệt và kiểm toán, vì vậy thay đổi nó là một quá trình di chuyển.

`tools/list` phải mang tính xác định. Khi khả năng hiển thị khác nhau theo principal, trả về `cacheScope: private`. Một `ttlMs` có giới hạn làm giảm tải discovery của backend mà không cho phép danh sách dành riêng cho người dùng bị rò rỉ qua các ngữ cảnh ủy quyền.

Mỗi descriptor tool được hiển thị bao gồm tên ổn định, mô tả và `inputSchema` gốc đối tượng. Namespacing không thể loại bỏ các trường descriptor bắt buộc. Kết quả danh sách đầy đủ cũng bao gồm `resultType`, metadata định danh server và các gợi ý bộ nhớ đệm.

### Ghim các descriptor đã phê duyệt

Tại thời điểm admission, chuẩn hóa descriptor đầy đủ và lưu trữ digest của nó dưới tên công khai đủ điều kiện. Tại thời điểm liệt kê và gọi, so sánh descriptor trực tiếp với digest đã phê duyệt.

Nếu nó thay đổi:

- Xóa nó khỏi `tools/list`.
- Từ chối các lệnh gọi trực tiếp.
- Phát ra sự kiện kiểm toán.
- Yêu cầu chính sách hoặc sự phê duyệt lại của con người trước khi cập nhật pin.

Gateway là một điểm thực thi trung tâm hữu ích, nhưng nó không biến một descriptor lần đầu nhìn thấy thành một descriptor an toàn. Việc xem xét ban đầu vẫn là cần thiết.

### Registries giúp khám phá, không phải quyết định

Một `server.json` Registry cung cấp metadata xuất bản. Một bản ghi được hỗ trợ bởi gói có thể trông như thế này:

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "com.example/notes",
  "description": "Example notes MCP server.",
  "version": "1.0.0",
  "packages": [
    {
      "registryType": "npm",
      "identifier": "@example/notes-mcp",
      "version": "1.0.0",
      "transport": {"type": "stdio"}
    }
  ]
}
```

Metadata xuất bản không mang quyết định bảo mật của gateway. Giữ bằng chứng về nhà xuất bản đã xác minh và nguồn gốc trong trạng thái admission riêng biệt:

```json
{
  "registryName": "com.example/notes",
  "registryVersion": "1.0.0",
  "publisher": {"namespace": "com.example", "status": "verified"},
  "provenance": {
    "source": "registry.modelcontextprotocol.io",
    "recordId": "com.example/notes@1.0.0"
  },
  "admission": {"status": "approved", "reviewedBy": "gateway-policy"}
}
```

Gateway kiểm tra hình dạng `server.json` và kết nối nó với trạng thái bên ngoài đó. Gateway vẫn cần một chính sách admission.

Đối với mỗi backend được thừa nhận, ghi lại:

- Định danh registry và bản ghi chính xác.
- Namespace nhà xuất bản đã xác minh hoặc bằng chứng tên miền.
- Transport và endpoint được cho phép.
- Phiên bản được ghim hoặc chính sách nâng cấp được phê duyệt.
- Artifact hoặc descriptor digest.
- Issuer và resource ủy quyền.
- Người đánh giá, thời gian phê duyệt và thời hạn.

Không chấp nhận một server chỉ vì tên hiển thị của nó giống với một sản phẩm quen thuộc. Không coi sự hiện diện trong registry là một đánh giá bảo mật vận hành. Các server riêng tư có thể được thừa nhận thông qua cùng một lược đồ bằng chứng ngay cả khi chúng không bao giờ xuất hiện trong registry công khai.

Bài học này triển khai đường nối gateway: kết nối bằng chứng xuất bản với admission cục bộ trước khi một backend có thể định tuyến. [Lesson 30: MCP Registry Supply Chain, Admission, Drift, and Rollback](../../30-mcp-registry-supply-chain-and-drift/docs/en.md) xây dựng toàn bộ control plane cho bằng chứng namespace chính xác, nguồn gốc artifact, pin bất biến, descriptor drift trực tiếp, đối chiếu trạng thái Registry, sổ cái admission chống giả mạo và rollback dựa trên bằng chứng. Giữ trạng thái chuỗi cung ứng đó tách biệt với quyết định runtime theo từng request ở trên.

### Điều phối thông tin xác thực

Gateway xác thực những người gọi của nó và xác thực riêng với các backend. Thông tin xác thực backend không bao giờ được gửi đến client.

Giữ các ràng buộc này tường minh:

```text
outer principal -> gateway role and policy
backend issuer + resource -> backend registration and token
```

Không bao giờ chuyển token gateway bên ngoài cho backend. Không bao giờ sử dụng lại token backend tại một issuer hoặc resource khác. Nếu một tool hoạt động thay mặt cho người dùng cuối, hãy bảo toàn sự ủy quyền đó bằng một mô hình trao đổi hoặc claims được thiết kế thay vì mạo danh người dùng bằng thông tin xác thực dịch vụ dùng chung.

### Giới hạn tốc độ không cần session

Giới hạn khóa theo principal đã xác thực, issuer, resource, tool công khai, lớp chi phí và cửa sổ thời gian. Id session không tồn tại và sẽ dễ dàng xoay vòng ngay cả khi nó tồn tại.

Áp dụng xác thực giá rẻ trước khi tiêu tốn công việc đắt đỏ. Quyết định xem các lệnh gọi bị từ chối có tính vào giới hạn lạm dụng, hạn ngạch kinh doanh hay cả hai.

### Kiểm toán chuỗi quyết định

Ghi lại đủ để tái tạo một lệnh gọi:

- Định danh request và trace.
- Principal và issuer đã xác thực.
- Tool công khai và route backend.
- Phiên bản descriptor pin.
- Quyết định chính sách và lý do.
- Độ trễ và lớp kết quả.
- Vòng MRTR hoặc định danh task khi áp dụng.

Che các bearer token, mã ủy quyền, refresh token, bí mật thô và các đối số nhạy cảm không cần thiết.

### SSE theo phạm vi request

Một POST bình thường có thể trả về SSE theo phạm vi request khi công việc được truyền tải trong một request đó. Đóng luồng phản hồi sẽ hủy request HTTP hiện đại đang thực hiện đó.

Không tạo luồng GET riêng biệt và không hứa hẹn phát lại Last-Event-ID. Đó là các giả định transport cũ.

### Thông báo thay đổi lâu dài

Đối với các thông báo thay đổi danh sách và tài nguyên, một client hiện tại gửi `subscriptions/listen` qua POST và nhận phản hồi SSE. Các bộ lọc thông báo sử dụng các trường phẳng chính xác `toolsListChanged`, `promptsListChanged`, `resourcesListChanged` và `resourceSubscriptions`:

```json
{
  "jsonrpc": "2.0",
  "id": "listen-tools",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Sự kiện đầu tiên xác nhận tập hợp con được hỗ trợ. Định danh đăng ký của nó là id JSON-RPC của request đã mở luồng:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": "listen-tools"
    },
    "notifications": {
      "toolsListChanged": true
    }
  }
}
```

Gateway sau đó chỉ chuyển tiếp các loại thay đổi đã được xác nhận. Mỗi thông báo trên luồng đó mang cùng `io.modelcontextprotocol/subscriptionId` trong `params._meta`. Không có phát lại tự động hoặc tự động lắng nghe lại. Khi kết nối lại, client mở lại đăng ký và làm mới các danh sách mà nó dựa vào. Một lần đóng duyên dáng do server khởi tạo trả về kết quả hoàn chỉnh cuối cùng được gắn thẻ với cùng id đăng ký.

Đường dẫn hiện đại thay thế `resources/subscribe`, `resources/unsubscribe` và luồng GET độc lập không được yêu cầu. Chỉ giữ những thứ đó trong đường dẫn cũ được kiểm soát phiên bản.

### MRTR thông qua gateway

Khi một backend trả về `resultType: input_required`, gateway có thể chuyển tiếp kết quả đó chỉ khi client bên ngoài hỗ trợ request đầu vào cần thiết. Bảo toàn `requestState` từng byte trừ khi gateway cố tình chấm dứt và phát hành lại tương tác.

Client thử lại tool công khai gốc với id JSON-RPC mới và `inputResponses`. Gateway ủy quyền lại cho lần thử lại, kiểm tra cùng route công khai, sau đó chuyển tiếp một request backend mới. Nó không được giả định rằng một vòng trước đó đã cấp sự phê duyệt không giới hạn.

### Định tuyến extension Tasks

Tasks là một extension chính thức được xác định bởi `io.modelcontextprotocol/tasks`. Chúng không phải là sự thay thế session cốt lõi.

Client khai báo extension bên trong các khả năng client theo từng request, và gateway quảng bá nó trong discovery chỉ khi nó có thể bảo toàn vòng đời end-to-end. Đối với một `tools/call` được hỗ trợ, chỉ backend mới quyết định trả về kết quả thông thường hay `resultType: task`. Một kết quả task mang theo `taskId`, `status`, dấu thời gian, `ttlMs` và `pollIntervalMs` tùy chọn trực tiếp trong kết quả. Task phải có thể đọc được một cách bền vững trước khi kết quả đó được gửi đi.

Gateway ghi lại principal đã xác thực và route backend cho định danh task mờ (opaque). Các lệnh gọi `tasks/get`, `tasks/update` và `tasks/cancel` tiếp theo sử dụng `params.taskId` làm `Mcp-Name`, cung cấp cho các trung gian một khóa định tuyến. `tasks/get` trả về `resultType: complete` với trạng thái task hiện tại và nội dòng kết quả cuối cùng hoặc lỗi giao thức ở trạng thái kết thúc. `tasks/update` gửi `inputResponses` có khóa cho đầu vào task đang chờ xử lý và trả về xác nhận hoàn thành trống. `tasks/cancel` là một ý định hợp tác với xác nhận hoàn thành trống, không phải là sự đảm bảo rằng công việc sẽ dừng lại.

Không triển khai các phương thức `tasks/list` hoặc `tasks/result` mới. Chúng thuộc về mô hình thử nghiệm cũ. Một task cần đầu vào sẽ hiển thị các request nhúng hoàn chỉnh thông qua `tasks/get`; client trả lời chúng thông qua `tasks/update`, không phải bằng cách thử lại lệnh gọi tool gốc. Client vẫn thăm dò ở khoảng thời gian được đề xuất; việc tạo task vẫn do server điều hướng.

Trạng thái route task bền vững là dữ liệu ứng dụng được khóa bởi handle task, không phải session giao thức.

### Ranh giới tương thích

Nếu gateway phải phục vụ một client hoặc backend cũ:

- Phát hiện kỷ nguyên một cách tường minh.
- Giữ khởi tạo, session transport, luồng GET, đăng ký tài nguyên và từ vựng task cũ bên trong một adapter kế thừa.
- Không bao giờ làm rò rỉ id session cũ vào định tuyến hoặc ủy quyền hiện đại.
- Ưu tiên thăm dò discovery có giới hạn và chính sách dự phòng tường minh hơn là hạ cấp âm thầm.

```figure
t3-gateway-funnel
```

## Xây dựng

`code/main.py` triển khai một gateway giao thức trong tiến trình và hai server backend. Mỗi backend nhận một request giao thức hiện tại mới. Gateway cung cấp discovery, `tools/list` xác định được lọc bởi người dùng, định tuyến theo namespace, `server.json` Registry cộng với trạng thái admission bên ngoài, descriptor pin, RBAC, giới hạn tốc độ theo principal, quyết định kiểm toán và xác nhận SSE `subscriptions/listen` được mô hình hóa.

Mô hình nhận các body request đã phân tích, routing header và định danh bearer đã xác thực. Nó không phải là một adapter HTTP hoàn chỉnh và không phân tích `Content-Type` hoặc toàn bộ hợp đồng `Accept`. Kết nối nó với adapter Streamable HTTP của Lesson 09, yêu cầu `Content-Type: application/json` và giá trị `Accept` chứa cả `application/json` và `text/event-stream`.

Chạy nó:

```bash
cd phases/13-tools-and-protocols/17-mcp-gateways-and-registries
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Bản demo in ra id request bên ngoài và id request backend mới để có thể thấy bước nhảy không trạng thái.

## Sử dụng

Thay thế các đối tượng backend trong tiến trình bằng các client giao thức hiện tại thực tế. Giữ nguyên các đường nối:

- Bản ghi admission trước khi kết nối.
- Discovery backend trước khi hiển thị khả năng.
- Tên công khai đủ điều kiện trước khi ủy quyền.
- Descriptor pin trước khi liệt kê hoặc gọi.
- Metadata theo từng request mới trước khi chuyển tiếp.
- Xác thực kết quả trước khi trả về.

## Vận chuyển

Bài học này vận chuyển `outputs/skill-gateway-bootstrap.md`. Nó tạo ra một thiết kế gateway hiện đại bao gồm ingress, discovery, admission, namespace, ủy quyền, bộ nhớ đệm, streaming, đăng ký, MRTR, Tasks, khả năng quan sát và cô lập kế thừa.

## Bài tập

1. Thêm trace context vào metadata request bên ngoài và được chuyển tiếp, đồng thời ghi lại sự tương quan trong sự kiện kiểm toán.
2. Thêm một backend có khả năng Tasks và định tuyến `tasks/get` theo id task trong `Mcp-Name`.
3. Thay đổi một descriptor backend và chứng minh cả discovery và lệnh gọi trực tiếp đều bị chặn.
4. Thêm một khả năng server dành riêng cho principal và giải thích tại sao discovery phải được lưu vào bộ nhớ đệm một cách riêng tư.
5. Viết một giao diện adapter kế thừa mà không thêm bất kỳ trạng thái kế thừa nào vào lớp `Gateway` hiện đại.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa |
|------|---------|
| MCP gateway | Server chính sách và định tuyến giữa client và các backend MCP server |
| Admission record | Bằng chứng và quyết định chính sách cho phép một backend vào gateway |
| Qualified tool name | Route công khai ổn định như `notes.search` |
| Descriptor pin | Digest đã phê duyệt được kiểm tra trong quá trình discovery và điều phối |
| Private cache scope | Kết quả được lưu vào bộ nhớ đệm giới hạn trong một ngữ cảnh ủy quyền |
| Request-scoped SSE | Luồng phản hồi đính kèm với một request POST |
| `subscriptions/listen` | Luồng SSE do client mở cho các thông báo thay đổi lâu dài đã chọn |
| Task route | Ánh xạ ứng dụng từ id task mờ đến backend của nó |
| Legacy adapter | Ranh giới được kiểm soát phiên bản tường minh cho hành vi handshake và session cũ |

## Đọc thêm

- [Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [Yêu cầu server.json của Registry chính thức](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)