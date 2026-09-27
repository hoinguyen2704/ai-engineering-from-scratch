# Model Context Protocol (MCP)

> MCP cung cấp cho một AI host một giao thức duy nhất để khám phá và gọi các công cụ (tools), tài nguyên (resources) và lời nhắc (prompts). Bản sửa đổi 2026-07-28 làm cho giao thức đó trở nên phi trạng thái (stateless): ngữ cảnh về khả năng (capability) và phiên bản (version) được truyền theo mọi yêu cầu, thay vì thông qua bắt tay (handshake) gắn liền với kết nối.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 03 (Structured Outputs)
**Time:** ~75 phút

## Mục tiêu học tập

- Phân biệt được MCP host, client, server, transport và server primitive.
- Xây dựng một yêu cầu JSON-RPC với siêu dữ liệu (metadata) theo yêu cầu của MCP 2026-07-28.
- Sử dụng `server/discover` để kiểm tra các phiên bản, danh tính và khả năng.
- Trả về các kết quả có kiểu dữ liệu (typed) và nhận biết bộ nhớ đệm (cache-aware) từ các công cụ, tài nguyên và lời nhắc.
- Giải thích cách MCP phi trạng thái hiện đại tương tác với các server thời kỳ bắt tay.
- Lựa chọn các ranh giới an toàn về trạng thái, transport và phê duyệt cho một server.

## Vấn đề

Ứng dụng của bạn cần truy vấn cơ sở dữ liệu, thao tác lịch và trình đọc tệp. Nếu không có một giao thức chung, mỗi AI host đều cần các đoạn mã kết nối (glue code) tùy chỉnh cho việc khám phá, gọi hàm, xử lý lỗi, transport và ủy quyền cho cùng những khả năng đó.

MCP giúp giảm bớt ma trận tích hợp này. Một server xuất bản một bề mặt JSON-RPC tiêu chuẩn. Một client tuân thủ có thể khám phá bề mặt đó, trình bày nó cho mô hình hoặc người dùng, gọi nó và diễn giải kết quả mà không cần bộ điều hợp (adapter) dành riêng cho server.

Ranh giới quan trọng rất dễ bị bỏ lỡ. MCP tiêu chuẩn hóa giao tiếp. Nó không quyết định công cụ nào mô hình nên gọi, không làm cho nội dung không đáng tin cậy trở nên an toàn, hay biến một yêu cầu phi trạng thái thành trạng thái ứng dụng bền vững. Host và server của bạn vẫn là bên chịu trách nhiệm cho những quyết định đó.

## Khái niệm

![MCP host, stateless request, and server primitives](../assets/mcp-architecture.svg)

### Ba server primitive

1. **Tools** là các hành động có thể gọi được. Mỗi tool có tên, mô tả, đầu vào JSON Schema và trình xử lý (handler).
2. **Resources** là nội dung được đặt tên, định địa chỉ bằng URI mà client có thể đọc.
3. **Prompts** là các mẫu có thể tái sử dụng mà host có thể hiển thị cho người dùng.

Host chính là ứng dụng AI. Một MCP client bên trong host đó sẽ giao tiếp với một server. Transport mang các thông điệp JSON-RPC giữa chúng.

### Các yêu cầu phi trạng thái thay thế bắt tay

MCP 2026-07-28 loại bỏ `initialize` và `notifications/initialized`. Nó cũng loại bỏ các phiên (session) ở cấp độ giao thức. Mọi yêu cầu đều mang theo ngữ cảnh cần thiết để diễn giải nó trong `params._meta`:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Phiên bản giao thức và khả năng của client là bắt buộc. Danh tính của client được khuyến nghị. Một `_meta` bị thiếu, một trường bắt buộc bị thiếu hoặc một trường bắt buộc có sai kiểu dữ liệu sẽ bị coi là sai định dạng và trả về Invalid Params (`-32602`). Một chuỗi phiên bản đúng định dạng nhưng server không hỗ trợ sẽ trả về `UnsupportedProtocolVersionError` (`-32022`). Server có thể xử lý một yêu cầu hợp lệ mà không cần khôi phục bản ghi đàm phán trước đó.

Phi trạng thái không có nghĩa là một ứng dụng không bao giờ có thể duy trì trạng thái. Nó có nghĩa là trạng thái đó không bị ẩn sau một kết nối MCP hoặc `Mcp-Session-Id`. Nếu một quy trình làm việc cần sự liên tục, server sẽ tạo ra một handle mờ (opaque handle) và client sẽ truyền handle đó như một đối số công cụ thông thường trong các lần gọi sau. Việc ủy quyền vẫn phải được kiểm tra trên mọi yêu cầu.

### Khám phá và lựa chọn phiên bản

Mọi server hiện đại đều triển khai `server/discover`. Kết quả sẽ quảng bá các phiên bản được hỗ trợ, khả năng và danh tính của server:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "complete",
    "supportedVersions": ["2026-07-28"],
    "capabilities": {
      "tools": {},
      "resources": {},
      "prompts": {}
    },
    "ttlMs": 3600000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "demo-server",
        "version": "1.0.0"
      }
    }
  }
}
```

Client có thể gọi trực tiếp một phương thức khác và xử lý lỗi phiên bản, nhưng việc khám phá giúp hiển thị khả năng và lựa chọn phiên bản trở nên rõ ràng. Một phiên bản không được hỗ trợ sẽ trả về `UnsupportedProtocolVersionError` với mã `-32022`. Dữ liệu của nó chứa `supported`, một mảng các bản sửa đổi của server, và `requested`, bản sửa đổi bị từ chối.

Trên stdio, một client hỗ trợ cả hai thời kỳ sẽ thăm dò bằng `server/discover`. Một kết quả khám phá hoặc một lỗi hiện đại được nhận diện như `UnsupportedProtocolVersionError` sẽ xác định một server hiện đại. Bất kỳ lỗi hoặc timeout nào không được nhận diện là hiện đại sẽ cho phép quay lại luồng `initialize` của ngày 25-11-2025. Hành vi cũ là mã tương thích, không phải mặc định hiện đại.

### Kết quả là rõ ràng

Mọi kết quả cốt lõi của 2026-07-28 đều có `resultType`:

- `complete` nghĩa là thao tác đã hoàn tất.
- `input_required` nghĩa là server cần thêm một vòng lặp thông qua mô hình Multi Round-Trip Requests. Các server cốt lõi chỉ có thể trả về điều này từ `tools/call`, `resources/read` hoặc `prompts/get`.

Client phải coi một kết quả cũ bỏ qua `resultType` là đã hoàn tất.

Server nên bao gồm `io.modelcontextprotocol/serverInfo` trong `_meta` của mọi kết quả. Danh tính này do server tự báo cáo và dùng để hiển thị, ghi nhật ký và gỡ lỗi, không dùng cho các quyết định bảo mật.

Các kết quả liệt kê và đọc cũng mang theo `ttlMs` và `cacheScope`. Thứ tự `tools/list` xác định cộng với gợi ý về độ tươi mới (freshness hint) cho phép client lưu bộ nhớ đệm khám phá một cách an toàn và cải thiện độ ổn định của bộ nhớ đệm lời nhắc. `cacheScope: public` cho phép lưu bộ nhớ đệm chia sẻ; `private` giới hạn việc tái sử dụng trong ngữ cảnh gọi.

### Định dạng truyền tin và transport

MCP sử dụng JSON-RPC 2.0 qua stdio hoặc Streamable HTTP.

- Một yêu cầu có `jsonrpc`, `id`, `method` và `params`.
- Một phản hồi có `id` khớp và hoặc `result` hoặc `error`.
- Một thông báo (notification) không có `id` và không mong đợi phản hồi.

Streamable HTTP hiện đại hiển thị một endpoint chấp nhận POST. Mỗi thông điệp JSON-RPC nhận được một POST riêng. Một POST yêu cầu nhận được hoặc một đối tượng JSON hoặc một luồng Server-Sent Events (SSE) theo phạm vi yêu cầu kết thúc bằng phản hồi cuối cùng. Một POST thông báo được chấp nhận sẽ nhận được HTTP 202 mà không có nội dung phản hồi; bản sửa đổi cốt lõi này không định nghĩa bất kỳ thông báo client-to-server nào qua Streamable HTTP.

Không có luồng MCP GET độc lập, endpoint phiên DELETE, `Mcp-Session-Id` hoặc phát lại `Last-Event-ID` trong 2026-07-28. Các thông báo thay đổi dài hạn sử dụng một POST `subscriptions/listen` mà phản hồi của nó vẫn mở dưới dạng luồng SSE.

### Đầu vào của client mà không cần yêu cầu do server khởi tạo

Các bản sửa đổi cũ hơn cho phép server gửi các yêu cầu như `sampling/createMessage`, `roots/list` hoặc `elicitation/create` qua luồng. Giao thức hiện tại sử dụng Multi Round-Trip Requests thay thế. Một lệnh gọi công cụ, đọc tài nguyên hoặc lấy lời nhắc hợp lệ sẽ trả về `resultType: input_required` với ít nhất một trong số `inputRequests` hoặc `requestState`. Client thu thập bất kỳ đầu vào nào được yêu cầu, thử lại phương thức gốc với một ID JSON-RPC mới và `inputResponses` tương ứng, đồng thời lặp lại chính xác `requestState` khi được cung cấp. Nếu không có `inputRequests` nào, lần thử lại sẽ bỏ qua `inputResponses`.

Roots, Sampling và Logging vẫn hoạt động nhưng đã bị phản đối (deprecated), vì vậy các triển khai mới không nên áp dụng chúng. Các yêu cầu Roots hoặc Sampling hiện có sẽ truyền bên trong `inputRequests` của MRTR, không bao giờ là các yêu cầu JSON-RPC độc lập từ server đến client. Hãy ưu tiên các tham số tệp hoặc thư mục rõ ràng, URI tài nguyên, cấu hình server và tích hợp trực tiếp với nhà cung cấp mô hình. Sử dụng stderr cho chẩn đoán stdio và OpenTelemetry cho đo lường sản xuất.

```figure
mcp-nxm-collapse
```

## Xây dựng

### Bước 1: đăng ký bề mặt server

Việc đăng ký vẫn đơn giản ngay cả khi hợp đồng yêu cầu đã thay đổi:

```python
server = MCPServer("demo-server")

@server.tool(
    "add",
    "Add two integers.",
    {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    }
)
def add(a: int, b: int) -> dict:
    return {"sum": a + b}
```

Triển khai được cung cấp trong `code/main.py` cũng đăng ký một tài nguyên và lời nhắc. Nó cố tình sử dụng thư viện tiêu chuẩn để bạn có thể thấy từng phong bì (envelope) thay vì ủy quyền giao thức cho một SDK.

### Bước 2: đính kèm siêu dữ liệu vào mọi yêu cầu

```python
def request(method, params=None):
    body_params = dict(params or {})
    body_params["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": "demo-client",
            "version": "1.0.0"
        }
    }
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": body_params
    }
```

Đừng chỉ lưu bộ nhớ đệm siêu dữ liệu này trong một đối tượng kết nối. Server xác thực nó trên mỗi yêu cầu.

### Bước 3: tùy chọn khám phá trước khi liệt kê

Gọi `server/discover`, chọn một phiên bản được hỗ trợ, sau đó gọi `tools/list`. Một lệnh gọi `tools/list` trực tiếp cũng hợp lệ nếu bạn đã biết phiên bản và có thể xử lý `-32022`.

Bản demo trả về danh sách công cụ theo thứ tự tên và đính kèm `ttlMs`, `cacheScope`, `resultType` và danh tính server. Một lệnh gọi công cụ trả về một kết quả hoàn chỉnh, không thể lưu bộ nhớ đệm vì đầu ra của nó có thể phụ thuộc vào trạng thái hiện tại.

### Bước 4: ánh xạ cùng một yêu cầu sang HTTP

Một POST `tools/call` từ xa bao gồm các tiêu đề phản chiếu nội dung JSON-RPC:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: application/json, text/event-stream
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: add
```

Tiêu đề `MCP-Protocol-Version` phải khớp với phiên bản trong `_meta`. `Mcp-Method` là bắt buộc trên mọi yêu cầu JSON-RPC và phải khớp với `method`. `Mcp-Name` chỉ bắt buộc đối với `tools/call`, `resources/read` và `prompts/get`, nơi nó phải khớp với tên công cụ, URI tài nguyên hoặc tên lời nhắc. Một tiêu đề bắt buộc bị thiếu hoặc không khớp sẽ trả về HTTP 400 với mã `HeaderMismatch` `-32020`.

### Bước 5: thực thi an toàn bên ngoài trạng thái giao thức

- Xác thực ủy quyền và đối tượng trên mọi yêu cầu HTTP.
- Ràng buộc các server cục bộ với localhost và xác thực `Origin` trên Streamable HTTP.
- Đánh dấu các công cụ gây đột biến bằng `destructiveHint: true` và yêu cầu sự phê duyệt của host.
- Truyền phạm vi thư mục và tệp một cách rõ ràng thay vì phụ thuộc vào các Roots đã bị phản đối.
- Coi tài nguyên và đầu ra của công cụ là dữ liệu không đáng tin cậy.
- Giữ stdout dành riêng cho JSON-RPC dưới stdio; ghi chẩn đoán vào stderr.

## Sử dụng

Chạy bài học từ thư mục của nó:

```bash
python3 code/main.py
cd code
python3 -m unittest discover tests -v
```

Dòng đầu tiên sẽ báo cáo việc khám phá `demo-server` tại giao thức `2026-07-28`. Sau đó kiểm tra `MCPClient.request`: nó tái tạo `_meta` cho mỗi lần gọi. Xóa siêu dữ liệu khỏi một yêu cầu và quan sát server từ chối nó.

## Phát hành

`outputs/skill-mcp-server-designer.md` biến một miền thành một thiết kế MCP phi trạng thái. Cổng chấp nhận của nó yêu cầu kết quả khám phá, chính sách siêu dữ liệu trên mỗi yêu cầu, danh sách nhận biết bộ nhớ đệm xác định, các handle trạng thái rõ ràng, tiêu đề transport, ủy quyền và các quy tắc phê duyệt.

## Tiếp tục tìm hiểu sâu về MCP

Bài học này cung cấp cho bạn mô hình giao thức. Giai đoạn 13 biến bốn ranh giới sản xuất thành các bài học xây dựng và xác minh riêng biệt:

1. [MCP Tool Contracts and Content](../../../13-tools-and-protocols/28-mcp-tool-contracts-and-content/docs/en.md) bao gồm các lược đồ đầu vào đóng, nội dung có cấu trúc, siêu dữ liệu định tuyến, phân trang mờ, ủy quyền hoàn thành và sự khác biệt giữa lỗi giao thức và lỗi miền công cụ.
2. [MCP Reliability, Cancellation, and Flow Control](../../../13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/docs/en.md) bao gồm hủy yêu cầu, hủy tác vụ bền vững, thời hạn, tính lũy đẳng (idempotency), kiểm soát luồng (backpressure), bộ đệm proxy và hành vi kết nối lại.
3. [MCP Registry Supply Chain, Admission, Drift, and Rollback](../../../13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/docs/en.md) bao gồm bằng chứng không gian tên, nguồn gốc hiện vật, ghim bất biến, trôi dạt trực tiếp, trạng thái Registry, bằng chứng nhập học và khôi phục.
4. [MCP Conformance Engineering](../../../13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/docs/en.md) bao gồm các bản ghi dây dẫn vàng và tiêu cực, các kỷ nguyên phiên bản nghiêm ngặt, sự khác biệt SDK, bằng chứng proxy, biên tập, cổng sức khỏe và khôi phục phát hành.

Hãy làm theo chúng theo thứ tự khi server sẽ vượt qua ranh giới nhóm hoặc ranh giới tin cậy. Cùng nhau, chúng chuyển từ "phương thức hoạt động" sang "hợp đồng vẫn an toàn và có thể chẩn đoán thông qua triển khai".

## Bài tập

1. Thêm một công cụ `subtract` và xác nhận `tools/list` vẫn được sắp xếp theo thứ tự bảng chữ cái.
2. Xóa khóa phiên bản giao thức và xác minh Invalid Params (`-32602`). Sau đó gửi phiên bản đúng định dạng nhưng không được hỗ trợ `2025-11-25`, xác minh `-32022`, xác nhận `requested` lặp lại bản sửa đổi đó và chọn từ `supported`.
3. Thêm một `draftId` do server tạo vào một thao tác tạo, sau đó yêu cầu nó làm đối số để cập nhật. Giải thích tại sao đó là trạng thái ứng dụng thay vì một phiên giao thức.
4. Trả về `input_required` từ một công cụ cần người dùng xác nhận. Thử lại lệnh gọi gốc với một ID mới, một mục `inputResponses` và chính xác `requestState` thay vì phát minh ra một yêu cầu JSON-RPC từ server đến client.
5. Phác thảo một client stdio hai thời kỳ. Coi một kết quả hoặc lỗi hiện đại được nhận diện là hiện đại, và chỉ cho phép quay lại `initialize` đối với lỗi hoặc timeout không được nhận diện.

## Thuật ngữ chính

| Thuật ngữ | Cách mọi người nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| MCP | "Giao thức công cụ cho LLM" | Giao thức JSON-RPC để khám phá server, công cụ, tài nguyên, lời nhắc và tiện ích mở rộng |
| Host | "Ứng dụng AI" | Sở hữu mô hình và giao diện người dùng, gắn một hoặc nhiều MCP client |
| Client | "Bộ kết nối" | Nói MCP với một server thay mặt cho host |
| Stateless MCP | "Không phiên" | Mọi yêu cầu mang theo phiên bản và khả năng; không có trạng thái giao thức nào được khóa bởi kết nối |
| `server/discover` | "Thăm dò khả năng" | Phương thức server bắt buộc quảng bá các phiên bản, khả năng và danh tính |
| `resultType` | "Trạng thái kết quả" | Đánh dấu một kết quả là `complete` hoặc `input_required` |
| State handle | "ID quy trình làm việc" | Định danh ứng dụng do server tạo được truyền dưới dạng đối số thông thường |
| Streamable HTTP | "Transport từ xa" | Một endpoint POST với phản hồi JSON hoặc SSE theo phạm vi yêu cầu |
| MRTR | "Hỏi và thử lại" | Yêu cầu đầu vào được nhúng trong một kết quả, theo sau là việc thử lại thao tác gốc |

## Đọc thêm

- [Các thay đổi chính của MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [Khám phá server MCP](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Các tính năng MCP bị phản đối](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)