# Xây dựng MCP Client: Khám phá, Định tuyến và Dự phòng Dual-Era

> Một MCP client hiện đại lặp lại hợp đồng của nó trên mỗi yêu cầu. Quyết định tương thích khó khăn nhất là biết khi nào một server cũ thực sự đã lỗi thời và khi nào một server hiện đại đang báo cáo một lỗi có thể sửa chữa được.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lesson 07
**Time:** ~85 minutes

## Mục tiêu học tập

- Xây dựng mọi yêu cầu MCP `2026-07-28` với metadata hiện tại.
- Thăm dò các server stdio bằng `server/discover` và chọn phiên bản được hỗ trợ lẫn nhau.
- Ủy quyền thăm dò legacy có giới hạn chỉ cho các peer được cho phép rõ ràng (allowlisted).
- Chấp nhận một kỷ nguyên (era) legacy chỉ sau khi xác thực kết quả `initialize` dương tính cho một bản sửa đổi được hỗ trợ.
- Hợp nhất danh sách công cụ (tool) một cách tất định mà không ghi đè âm thầm các xung đột.
- Định tuyến các lệnh gọi đến peer sở hữu từng công cụ mà không cần tạo ra các phiên giao thức (protocol sessions).

## Vấn đề

Một agent host thường giao tiếp với nhiều hơn một MCP server. Nó phải khám phá từng server, hợp nhất các danh mục công cụ, giải quyết các tên trùng lặp, định tuyến các lệnh gọi và phục hồi sau lỗi truyền tải.

Bản sửa đổi `2026-07-28` làm cho trạng thái ổn định trở nên đơn giản hơn vì mỗi yêu cầu đều độc lập. Tính tương thích làm cho quá trình khởi động trở nên tinh tế hơn. Một client có thể gặp phải:

- một server hiện đại hỗ trợ phiên bản ưu tiên;
- một server hiện đại trả về lỗi phiên bản hoặc lỗi header đã nhận diện;
- một server legacy chưa từng nghe đến `server/discover`;
- một server legacy giữ im lặng cho đến khi nhận được `initialize`.

Việc coi mọi lỗi thăm dò là legacy rất nguy hiểm. Một yêu cầu hiện đại bị lỗi định dạng, server bị quá tải, tiến trình bị chết và server cũ đều có thể tạo ra cùng một kết quả timeout hoặc đóng kết nối. Những tín hiệu đó là mơ hồ. Client phải kết hợp ý định rõ ràng của người vận hành với bằng chứng giao thức xác thực trước khi chọn kỷ nguyên legacy.

## Khái niệm

### Một peer, không phải một phiên giao thức

Giữ một bản ghi peer truyền tải cho mỗi tiến trình server hoặc endpoint:

- handle truyền tải hoặc hàm gửi;
- kỷ nguyên và phiên bản giao thức đã chọn;
- các khả năng (capabilities) của server được khám phá gần nhất;
- danh sách công cụ tất định gần nhất;
- các id yêu cầu đang chờ xử lý để đối chiếu;
- trạng thái sức khỏe truyền tải.

Đây là việc ghi chép của client. Nó không phải là trạng thái phiên giao thức. Trên MCP hiện đại, server vẫn nhận được phiên bản và khả năng hiện tại trên mỗi yêu cầu.

### Xây dựng mọi yêu cầu hiện đại từ đầu

```python
def modern_request(request_id, method, params, version, capabilities):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {
            **params,
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": version,
                "io.modelcontextprotocol/clientCapabilities": capabilities,
                "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
            },
        },
    }
```

Đừng đính kèm metadata một lần vào đối tượng kết nối và giả định rằng nó đã đến đích. Hãy đóng dấu và kiểm tra yêu cầu đã được tuần tự hóa cuối cùng.

### Khám phá hiện đại

`server/discover` trả về các phiên bản được hỗ trợ, khả năng của server, hướng dẫn, gợi ý bộ nhớ đệm và danh tính server được đề xuất. Client chọn phiên bản hiện đại cao nhất được hỗ trợ lẫn nhau.

Khám phá là tùy chọn đối với client chỉ hỗ trợ hiện đại, nhưng được khuyến nghị trên stdio. Một số server legacy chấp nhận một thao tác trước khi khởi tạo, vì vậy việc gửi `tools/list` trước có thể tạo ra một kết quả thành công mơ hồ. `server/discover` tạo ra một ranh giới kỷ nguyên rõ ràng.

### Thăm dò tương thích stdio

Một client stdio dual-era gửi `server/discover` với metadata hiện đại ưu tiên của nó trước bất kỳ yêu cầu nào khác. Có ba loại kết quả:

1. **DiscoverResult.** Server là hiện đại. Chọn phiên bản được hỗ trợ lẫn nhau và tiếp tục với metadata cho mỗi yêu cầu.
2. **Lỗi hiện đại đã nhận diện.** Server là hiện đại. Đối với `-32022`, chọn từ `data.supported` và thử lại với một id yêu cầu mới. Đối với lỗi header hoặc khả năng, hãy sửa yêu cầu. Không gửi `initialize`.
3. **Tín hiệu mơ hồ.** Một lỗi JSON-RPC không xác định, timeout, đóng kết nối hoặc phản hồi trống không xác định được kỷ nguyên. Hãy đóng kết nối trừ khi peer đó được cấu hình cho tương thích legacy.

Các lỗi giao thức hiện đại được nhận diện bao gồm:

- `-32020` HeaderMismatch
- `-32021` MissingRequiredClientCapability
- `-32022` UnsupportedProtocolVersion

Các lỗi hiện đại được nhận diện vẫn là hiện đại ngay cả khi peer nằm trong danh sách cho phép legacy. Khi một server chứng minh rằng nó hiểu từ vựng lỗi hiện đại, việc gửi `initialize` sẽ là một sự hạ cấp.

Đừng coi `-32601` là bằng chứng legacy dương tính. Nó chỉ làm cho một peer được cho phép rõ ràng đủ điều kiện cho một lần thăm dò legacy. Quy tắc tương tự áp dụng cho timeout, đóng kết nối hoặc phản hồi trống.

### Danh sách cho phép (Allowlisting) là ý định của người vận hành, không phải bằng chứng

Tương thích legacy phải là một thuộc tính rõ ràng của một cấu hình peer được ghim:

```python
client.add_server("archive", archive_transport, allow_legacy=True)
```

Ràng buộc lựa chọn đó với lệnh hoặc endpoint đã cấu hình. Không sử dụng ký tự đại diện cho phép một server tùy ý tự chọn các ngữ nghĩa yếu hơn. Một peer không có `allow_legacy=True` sẽ thất bại sau kết quả khám phá mơ hồ và không bao giờ nhận được `initialize`.

Danh sách cho phép cấp quyền để thăm dò. Nó không chọn kỷ nguyên. Client gửi một `initialize` dưới thời hạn do truyền tải thực thi, sau đó yêu cầu tất cả các điều sau:

- một phản hồi JSON-RPC `2.0` với id yêu cầu khớp;
- chính xác một `result` và không có `error`;
- một `protocolVersion` trong tập hợp bản sửa đổi legacy đã cấu hình của client;
- một trường `capabilities` có giá trị là đối tượng;
- một đối tượng `serverInfo` với các trường `name` và `version` là chuỗi không trống.

Timeout, đóng kết nối, phản hồi lỗi, kết quả sai định dạng, id không khớp hoặc bản sửa đổi không được hỗ trợ sẽ dẫn đến thất bại. Chỉ một kết quả dương tính hợp lệ về cấu trúc mới chọn kỷ nguyên legacy. Mã nguồn truyền `legacy_probe_timeout_ms` cho bộ điều hợp truyền tải; một bộ điều hợp stdio hoặc HTTP thực tế phải thực thi thời hạn đó thay vì chỉ ghi lại nó.

Lưu vào bộ nhớ đệm kỷ nguyên đã chọn cho peer truyền tải. Đừng thăm dò lại trước mỗi lệnh gọi.

### Legacy là một nhánh tương thích

Khi quá trình thăm dò có giới hạn trả về bằng chứng legacy dương tính hợp lệ, client sử dụng phiên bản legacy đã chọn chính xác như được định nghĩa bởi bản sửa đổi đó:

1. Xác minh phong bì phản hồi và id đối chiếu.
2. Xác minh bản sửa đổi đã thương lượng nằm trong tập hợp legacy đã cấu hình.
3. Ghi lại các khả năng đã xác thực và danh tính server.
4. Gửi `notifications/initialized` chỉ sau khi tất cả các kiểm tra vượt qua.
5. Sử dụng các hình dạng yêu cầu legacy cho vòng đời truyền tải đó.

Nhánh này tồn tại để tương tác với các peer đã biết. Nó không phải là thiết kế mặc định cho các server mới hoặc yêu cầu mới. Nếu truyền tải khởi động lại hoặc endpoint thay đổi, hãy hủy bộ nhớ đệm peer-era và thương lượng lại.

### Khám phá và lưu trữ công cụ

Đối với mỗi peer đang hoạt động, hãy gọi `tools/list`. Kết quả hiện đại bao gồm `resultType`, `ttlMs` và `cacheScope`. Tôn trọng gợi ý độ tươi mới trong ngữ cảnh ủy quyền chính xác. Tìm nạp lại sau khi hết hạn hoặc sự kiện thay đổi danh sách đã đăng ký.

Client phải coi việc thiếu `resultType` từ một server legacy là `"complete"`. Không yêu cầu các trường bộ nhớ đệm hiện đại trên phản hồi từ một kỷ nguyên đã thương lượng trước đó.

Server nên trả về thứ tự tất định. Client cũng nên sắp xếp trước khi hợp nhất để thứ tự registry cục bộ không phụ thuộc vào thời gian khởi động tiến trình.

### Hợp nhất không gian tên an toàn với xung đột

Hai server có thể cùng hiển thị `search`. Chọn một chính sách đã khai báo:

1. **Tiền tố khi xung đột.** Giữ tên chính tắc đầu tiên và hiển thị các xung đột sau đó dưới dạng `<server>/<tool>`.
2. **Từ chối khi xung đột.** Không tải bản sao và hiển thị lỗi cấu hình rõ ràng.
3. **Ghi đè âm thầm.** Không bao giờ sử dụng cái này. Nó che giấu server nào nhận hành động do model chọn.

Lưu trữ cả tên chính tắc và tên cục bộ. Model nhìn thấy tên chính tắc. `tools/call` gửi đi sử dụng tên cục bộ mà server sở hữu đã khai báo.

### Định tuyến một lệnh gọi

Định tuyến là một tra cứu thuần túy:

```text
canonical tool name
  -> peer name + local tool name
  -> new JSON-RPC request id
  -> modern request metadata or explicit legacy shape
  -> matching response id
```

Không gửi lệnh gọi khi truyền tải sở hữu nó không khả dụng. Kết nối lại hoặc khởi động lại truyền tải, sau đó chạy lại khám phá và `tools/list`. Các yêu cầu hiện đại đang thực hiện bị mất trên truyền tải bị hỏng có thể được thử lại với một id JSON-RPC mới khi chính sách an toàn của thao tác cho phép.

### Thông báo và đăng ký

Các thay đổi danh sách và tài nguyên hiện đại chỉ đến trên luồng `subscriptions/listen` do client mở. Client gửi bộ lọc thông báo, đợi `notifications/subscriptions/acknowledged` và đối chiếu các sự kiện với id yêu cầu lắng nghe trong metadata thông báo.

Khi ngắt kết nối, mở một yêu cầu lắng nghe mới và tìm nạp lại các danh sách hoặc tài nguyên liên quan. Các luồng hiện đại không tiếp tục với `Last-Event-ID`.

### Không có yêu cầu do server khởi tạo

Các server hiện đại không gọi client bằng các yêu cầu JSON-RPC độc lập để lấy mẫu, gợi ý hoặc gốc. Chúng trả về `input_required` và client thử lại yêu cầu ban đầu sau khi thực hiện các yêu cầu đầu vào được nhúng.

Đừng chặn trình đọc phản hồi của peer trong khi thực hiện đầu vào. Duy trì đối chiếu và tạo một id JSON-RPC mới cho lần thử lại.

```figure
tp-client-merge
```

## Sử dụng

`code/main.py` sử dụng các hàm peer trong tiến trình để các quyết định giao thức vẫn hiển thị. Nó kết nối với hai peer hiện đại và một peer legacy được cho phép có chủ đích, sau đó hợp nhất và định tuyến các công cụ của chúng. Callable truyền tải nhận một ngân sách timeout để nhánh tương thích không thể che giấu một lần thăm dò không giới hạn.

```bash
cd code
python3 main.py
python3 -m unittest discover tests -v
```

Các bài kiểm tra chứng minh các ranh giới mà các bản demo thông thường bỏ lỡ:

- các yêu cầu hiện đại lặp lại metadata;
- `-32022` thử lại khám phá hiện đại mà không cần khởi tạo;
- các lỗi hiện đại được nhận diện không bao giờ hạ cấp, ngay cả đối với peer trong danh sách cho phép;
- timeout, đóng kết nối, phản hồi trống và lỗi không xác định không kích hoạt `initialize` nếu không có danh sách cho phép;
- một peer trong danh sách cho phép chỉ trở thành legacy sau kết quả `initialize` hợp lệ, được hỗ trợ;
- các kết quả legacy sai định dạng và không được hỗ trợ khiến peer không khả dụng;
- một kỷ nguyên đã chọn thành công được lưu vào bộ nhớ đệm cho vòng đời truyền tải.

## Ship It

Bài học này cung cấp `outputs/skill-mcp-client-harness.md`. Nó xây dựng khung đóng dấu yêu cầu hiện đại, thương lượng kỷ nguyên stdio, hợp nhất không gian tên tất định, định tuyến và một nhánh tương thích legacy thất bại an toàn (fail-closed).

## Bài tập

1. Làm cho một server giả trả về `-32022` mà không có phiên bản nào được hỗ trợ lẫn nhau. Xác nhận client thất bại thay vì gửi `initialize`.
2. Cho phép một server legacy giả, làm cho lần thăm dò `initialize` có giới hạn của nó bị timeout và chứng minh peer vẫn ở trạng thái `unknown` và không khả dụng.
3. Thêm danh sách công cụ `cacheScope: "private"` cho hai ngữ cảnh ủy quyền. Xác nhận client không bao giờ chia sẻ kết quả được lưu trong bộ nhớ đệm của ngữ cảnh này với ngữ cảnh kia.
4. Thay đổi chính sách xung đột thành từ chối và làm cho quá trình khởi động thất bại với cả hai tên peer trong lỗi.
5. Thêm một trình mô phỏng `subscriptions/listen` hữu hạn. Khi mất luồng, hãy lắng nghe lại với một id yêu cầu mới và tìm nạp lại các công cụ.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa |
|------|---------|
| Peer | Bản ghi phía client cho một truyền tải server và dữ liệu đã khám phá của nó |
| Kỷ nguyên giao thức | Metadata hiện đại cho mỗi yêu cầu hoặc ngữ nghĩa khởi tạo legacy |
| Thăm dò khám phá | `server/discover` ban đầu được sử dụng để xác định kỷ nguyên stdio |
| Lỗi hiện đại được nhận diện | Lỗi chứng minh hành vi hiện đại và cấm dự phòng legacy |
| Danh sách cho phép legacy | Cấu hình của người vận hành cho phép một lần thăm dò tương thích có giới hạn cho một peer được ghim |
| Bằng chứng legacy dương tính | Kết quả `initialize` hợp lệ, được đối chiếu cho một bản sửa đổi legacy được hỗ trợ rõ ràng |
| Không gian tên hợp nhất | Tên công cụ chính tắc trên tất cả các peer đang hoạt động |
| Chính sách xung đột | Quy tắc tiền tố hoặc từ chối cho các tên công cụ trùng lặp |
| Bộ nhớ đệm kỷ nguyên | Hành vi hiện đại hoặc legacy đã chọn được lưu trữ cho một peer truyền tải |
| Phục hồi truyền tải | Khởi động lại hoặc kết nối lại, khám phá lại, liệt kê lại và thử lại an toàn với một id mới |

## Đọc thêm

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Versioning](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)