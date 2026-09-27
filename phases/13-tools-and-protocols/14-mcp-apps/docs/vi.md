# MCP Apps trên Stateless Protocol

> Một kết quả tương tác vẫn là một trao đổi tool và resource của MCP. Bản core 2026-07-28 làm cho việc trao đổi đó trở nên khép kín, trong khi phần mở rộng Apps bổ sung bề mặt trình duyệt được sandbox.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Time:** ~75 phút

## Mục tiêu học tập

- Quảng bá MCP Apps thông qua `server/discover` và các khả năng mở rộng theo từng request.
- Khai báo một resource `ui://` trên một tool trước khi tool đó được gọi.
- Trả về kết quả tool và resource hoàn chỉnh trên giao thức stateless 2026-07-28.
- Tách biệt thông điệp bridge `ui/initialize` của Apps khỏi quá trình handshake core MCP đã bị loại bỏ.
- Áp dụng xác thực nguồn gốc, sandboxing, CSP và quyền truy cập tối thiểu (least-privilege).

## Vấn đề

Một kết quả văn bản có thể mô tả một dòng thời gian (timeline). Nó không thể cung cấp cho người dùng một dòng thời gian mà họ có thể lọc, kiểm tra hoặc thao tác.

MCP Apps giải quyết vấn đề trình bày bằng một phần mở rộng tùy chọn. Định nghĩa tool trỏ đến một resource `ui://`. Host có thể tìm nạp và xem xét resource đó trước khi tool chạy, hiển thị nó trong một iframe được sandbox và điều phối mọi hành động của ứng dụng thông qua một bridge JSON-RPC.

Giao thức core đã thay đổi vào ngày 2026-07-28. Đừng bao bọc một App trong vòng đời kết nối cũ:

- Không có request `initialize` hoặc thông báo `notifications/initialized` ở core.
- Không có header `Mcp-Session-Id`.
- Mọi request đều mang phiên bản giao thức và khả năng của client trong `params._meta`.
- Server triển khai `server/discover` để client có thể kiểm tra các phiên bản, khả năng core và các phần mở rộng.
- Mỗi kết quả thành công đều có một discriminator `resultType`.
- Streamable HTTP sử dụng một POST cho mỗi request. Các entrypoint GET và DELETE hiện đại trả về 405.

Bridge của Apps vẫn có một phương thức tên là `ui/initialize`. Nó thuộc về dialect postMessage của iframe. Nó không tạo lại một phiên MCP core.

## Khái niệm

### Hai giao thức, một tính năng

Giữ các lớp tách biệt rõ ràng:

1. Core MCP mang `server/discover`, `tools/list`, `tools/call`, `resources/list` và `resources/read`.
2. Phần mở rộng MCP Apps khai báo UI và định nghĩa bridge từ iframe đến host.
3. Các quy tắc sandbox của trình duyệt giới hạn những gì UI có thể truy cập.

Định danh phần mở rộng là `io.modelcontextprotocol/ui`. Cả hai bên đều chọn tham gia (opt-in). Client gửi hỗ trợ phần mở rộng bên trong đối tượng capabilities trên mỗi request:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "server/discover",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/ui": {}
        }
      },
      "io.modelcontextprotocol/clientInfo": {
        "name": "timeline-host",
        "version": "1.0.0"
      }
    }
  }
}
```

`clientInfo` được khuyến nghị cho mục đích chẩn đoán. Đây là dữ liệu tự báo cáo, không phải là danh tính xác thực.

### Khám phá trước khi hiển thị

Kết quả khám phá của server quảng bá phần mở rộng:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {},
    "resources": {},
    "extensions": {
      "io.modelcontextprotocol/ui": {}
    }
  },
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "timeline-app-server",
      "version": "2.0.0"
    }
  }
}
```

Server phải hỗ trợ khám phá. Client không bị buộc phải gọi khám phá trước mỗi hành động vì mỗi hành động mang theo các khả năng riêng của nó.

### Khai báo UI trên định nghĩa tool

Hợp đồng Apps hiện đại gắn UI vào tool trong `tools/list`:

```json
{
  "name": "notes_timeline",
  "description": "Render a timeline of notes.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  },
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline.html"
    }
  }
}
```

Đây là metadata được thiết kế để có trước khi gọi. Host có thể tải trước, lưu vào bộ nhớ đệm và kiểm tra bảo mật HTML trước khi kết quả yêu cầu hiển thị nó. Các khóa metadata phẳng cũ hơn có thể được chấp nhận bởi mã tương thích, nhưng các server mới nên phát hành dạng `_meta.ui.resourceUri` lồng nhau.

`tools/list` có thể lưu vào bộ nhớ đệm trong core hiện tại. Bao gồm thứ tự xác định, `ttlMs` và `cacheScope`. Sử dụng `private` khi các tool hiển thị thay đổi theo người dùng hoặc token.

### Trả về dữ liệu, sau đó để host gắn view

Việc gọi tool trả về nội dung thông thường cộng với dữ liệu có cấu trúc:

```json
{
  "resultType": "complete",
  "content": [
    {"type": "text", "text": "Timeline ready."}
  ],
  "structuredContent": {
    "notes": [
      {"id": "note-1", "title": "Discover", "created": "2026-07-28"}
    ]
  },
  "isError": false
}
```

Host đã biết view nào thuộc về tool nào. Tránh tạo ra một khối nội dung mới chỉ để lặp lại URI.

### Phục vụ ứng dụng như một resource

Server quảng bá `resources` trong quá trình khám phá, vì vậy nó cũng triển khai thao tác `resources/list` bắt buộc. Danh sách xác định của nó bao gồm URI chuẩn, tên ổn định, mô tả và loại MIME. Kết quả danh sách bao gồm `resultType`, metadata danh tính server, `ttlMs` và `cacheScope`, giống như danh sách tool xác định.

Host gửi `resources/read`. Trên Streamable HTTP, request có:

```text
POST /mcp
MCP-Protocol-Version: 2026-07-28
Mcp-Method: resources/read
Mcp-Name: ui://notes/timeline.html
```

Các giá trị header và body JSON-RPC phải khớp nhau. Sự không khớp là lỗi giao thức `-32020`.

Kết quả chứa resource HTML và các gợi ý bộ nhớ đệm:

```json
{
  "resultType": "complete",
  "contents": [
    {
      "uri": "ui://notes/timeline.html",
      "mimeType": "text/html;profile=mcp-app",
      "text": "<!doctype html>...",
      "_meta": {
        "ui": {
          "csp": {
            "connectDomains": [],
            "resourceDomains": [],
            "frameDomains": [],
            "baseUriDomains": []
          },
          "permissions": {}
        }
      }
    }
  ],
  "ttlMs": 60000,
  "cacheScope": "public"
}
```

### Lưu trữ resource UI dưới dạng nội dung thực thi

Một resource App không thể thay thế cho văn bản thông thường. Mục bộ nhớ đệm của nó có thể thực thi mã bridge, hiển thị dữ liệu tool và yêu cầu các hành động do host điều phối. Gán khóa theo URI `ui://` chuẩn, danh tính và phiên bản server được thừa nhận, digest nội dung resource và ngữ cảnh ủy quyền khi `cacheScope` là riêng tư. Không bao giờ sử dụng lại resource App riêng tư giữa các chủ thể vì HTML hoặc metadata chính sách của nó có thể khác nhau ngay cả khi URI giống hệt nhau.

Hủy bỏ mục nhập khi `ttlMs` của nó hết hạn, ràng buộc `_meta.ui.resourceUri` của tool thay đổi, phiên bản server hoặc ghim descriptor được thừa nhận thay đổi, hoặc một đăng ký thay đổi resource được xác nhận gọi tên URI đó. Tìm nạp lại và áp dụng lại CSP và kiểm tra quyền trước khi mount lại. Một iframe cũ không được giữ các quyền rộng hơn chỉ vì phiên bản resource mới chưa tải xong.

### Từ chối sự mơ hồ trên đường truyền trước chính sách tính năng

Việc xác thực có thứ tự rõ ràng. Đầu tiên, xác thực hình dạng JSON-RPC và yêu cầu metadata giao thức dạng chuỗi cộng với bản đồ khả năng client dạng đối tượng. Tiếp theo, so sánh các header định tuyến với body. Chỉ sau đó mới quyết định xem phiên bản giao thức khớp có được hỗ trợ hay không. Thứ tự này ngăn chặn proxy và server diễn giải các request khác nhau.

| Điều kiện | HTTP | Lỗi JSON-RPC |
|-----------|------|----------------|
| Phiên bản, phương thức hoặc tên trong header và body không khớp | 400 | `-32020` |
| Header và body đồng ý về một phiên bản không được hỗ trợ | 400 | `-32022`, với `data` chính xác là `{"supported":["2026-07-28"],"requested":"<actual>"}` |
| `resources/read` thiếu khả năng mở rộng Apps | 400 | `-32021`, với `data.requiredCapabilities.extensions.io.modelcontextprotocol/ui` |
| Phương thức không xác định | 404 | `-32601` |

Một thông báo JSON-RPC không có `id`, vì vậy server không bao giờ phát ra phản hồi JSON-RPC cho nó. Một thông báo HTTP được chấp nhận trả về 202 với body trống. Một lỗi có thể thay đổi trạng thái HTTP, nhưng nó vẫn không thể tạo body lỗi JSON-RPC cho một thông báo.

### Sandbox là một ranh giới, không phải là một phán quyết tin cậy

Host kiểm soát iframe. App không thể trực tiếp đọc cookie, local storage hoặc DOM trang của host. Mọi công việc đặc quyền phải đi qua bridge.

Sử dụng các mặc định sau:

- Để trống tất cả danh sách miền CSP, sau đó chỉ thêm các nguồn gốc mà App cần. Sử dụng `connectDomains` cho fetch, XHR và WebSocket; sử dụng `resourceDomains` cho scripts, styles, images và fonts.
- Gộp mã và dữ liệu khi thực tế.
- Không yêu cầu quyền camera, micro hoặc vị trí trừ khi một tính năng hiển thị cần nó.
- Ghim `postMessage` vào đúng nguồn gốc ngang hàng và từ chối các sự kiện từ mọi nguồn gốc khác.
- Đối xử với các đối số tool, kết quả tool, văn bản resource và thông điệp bridge như dữ liệu đầu vào không đáng tin cậy.
- Giữ sự đồng ý của người dùng trong host. Iframe không thể tự phê duyệt hành động mang tính hệ quả của chính nó.

Đừng sao chép thuộc tính `sandbox` cố định từ hướng dẫn vào mọi host. Host phải chọn các cờ dựa trên mô hình nguồn gốc của App và thiết kế cách ly của riêng nó.

Một miền được cho phép vẫn là một đường dẫn exfiltration. `connectDomains: ["https://api.example.com"]` có nghĩa là bất kỳ script nào thực thi bên trong App đều có thể gửi dữ liệu được phép đến đó. Khớp nguồn gốc chính xác ngăn chặn sự nhầm lẫn đích đến, nhưng nó không quyết định liệu payload có phù hợp hay không. Giữ quyền truy cập kết nối trống theo mặc định, tránh đặt bearer token trong iframe, proxy các thao tác hẹp thông qua host khi thực tế, giới hạn kích thước phản hồi và request, đồng thời kiểm tra hành động người dùng nào đã gây ra mỗi request gửi đi. Đối xử với `resourceDomains` tách biệt với `connectDomains`; quyền tải font hoặc script không nên cấp quyền tải lên dữ liệu tùy ý.

### Bridge Apps có vòng đời riêng

Bridge Apps là một dialect JSON-RPC qua `postMessage`. Nó có thể trao đổi các thông báo `ui/initialize` và `ui/*` và có thể proxy các phương thức trông giống core như `tools/call`.

View gửi `ui/initialize` với `appInfo` và một đối tượng `appCapabilities`. Host trả về các khả năng và ngữ cảnh host của nó. Chỉ sau phản hồi đó, View mới gửi `ui/notifications/initialized`. Host phải đợi thông báo Apps này trước khi gửi thông điệp đến View.

Handshake cục bộ đó tạo ra một bridge giữa một iframe và một khung host. Nó không thương lượng phiên bản giao thức MCP, tạo trạng thái server hoặc tạo phiên vận chuyển. Lưu ý tiền tố chính xác: core `notifications/initialized` đã bị loại bỏ, trong khi Apps `ui/notifications/initialized` vẫn còn. Một request core được tạo bởi một cuộc gọi tool được bridge là một request khép kín mới với id JSON-RPC mới và metadata request đầy đủ.

### Ngữ cảnh host, hành động và thu hồi

Host vẫn là cơ quan có thẩm quyền sau khi khởi tạo bridge. View có thể yêu cầu hành động tool, điều hướng, sử dụng clipboard hoặc hiệu ứng đặc quyền khác chỉ thông qua một khả năng mà host đã quảng bá. Host xác thực request đã nhập, người dùng hiện tại, mục tiêu và các đối số, áp dụng chính sách phê duyệt và có thể từ chối nó. Một cú nhấp chuột và một thông điệp bridge hợp lệ thể hiện ý định; không cái nào cấp thẩm quyền.

Đối xử với chủ đề (theme), kích thước và khả năng truy cập như ngữ cảnh host thay đổi thay vì các đầu vào render một lần:

- Áp dụng các token màu sắc và kiểu chữ do host cung cấp, sau đó phản ứng khi tùy chọn chủ đề hoặc độ tương phản thay đổi.
- Để View báo cáo kích thước mong muốn, nhưng để host giới hạn và áp dụng kích thước iframe để nội dung không thể thoát khỏi bố cục hoặc tạo các lớp phủ lừa đảo.
- Bảo toàn thứ tự bàn phím, tiêu điểm hiển thị, tên có thể truy cập, trạng thái trình đọc màn hình, độ tương phản đủ, thu phóng và hành vi giảm chuyển động bên trong iframe.
- Kiểm tra lại việc chuyển tiêu điểm giữa các điều khiển host và điều khiển View sau khi thay đổi kích thước và render lại.

Các khả năng có thể bị thu hồi khi App đang mở vì người dùng thay đổi tài khoản, chính sách thay đổi, server bị cách ly hoặc host thu hẹp sự đồng ý. Kiểm tra khả năng và ủy quyền tại thời điểm hành động, không chỉ trong quá trình `ui/initialize`. Khi thu hồi, từ chối các cuộc gọi đặc quyền đang chờ xử lý, dừng hoạt động mạng không còn phù hợp với chính sách, xóa trạng thái render nhạy cảm và mount lại hoặc quay lại văn bản khi chính resource UI không còn được thừa nhận. View phải xử lý việc từ chối như một kết quả bình thường, không thử lại cho đến khi host đồng ý.

### Fallback là một phần của hợp đồng

Một server nhận biết Apps vẫn có thể phục vụ các host không quảng bá phần mở rộng UI:

- Trả về cùng một tool mà không có `_meta.ui` trong `tools/list`.
- Giữ một kết quả văn bản hữu ích cho `tools/call`.
- Từ chối `resources/read` cho UI với lỗi thiếu khả năng.
- Không bao giờ giả định iframe tồn tại khi quyết định xem tool đã hoàn thành hay chưa.

```figure
t3-ui-sandbox
```

## Xây dựng

`code/main.py` xây dựng một mô hình giao thức trong tiến trình nhỏ mà không cần SDK. Nó xác thực envelope request hiện tại và các giá trị định tuyến Streamable HTTP, quảng bá Apps thông qua `server/discover`, liệt kê các tool và resource, thực thi tool và phục vụ một resource HTML khép kín.

Mô hình nhận các body và header định tuyến đã được phân tích cú pháp. Nó không phải là một bộ điều hợp HTTP hoàn chỉnh và không phân tích `Content-Type` hoặc `Accept`. Sử dụng Bài học 09 cho bộ điều hợp Streamable HTTP đầy đủ yêu cầu `Content-Type: application/json` và một giá trị `Accept` chứa cả `application/json` và `text/event-stream`.

Chạy nó:

```bash
cd phases/13-tools-and-protocols/14-mcp-apps
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Kiểm tra bốn điều trong đầu ra:

1. Mỗi cuộc gọi là độc lập.
2. Mỗi request có khả năng `_meta`.
3. `resources/list` trả về một descriptor ổn định trước bất kỳ lần đọc resource nào.
4. Mỗi kết quả có `resultType` và metadata danh tính server.
5. Không có định danh phiên core nào xuất hiện.

## Sử dụng

Bắt đầu với `server/discover`. Xác nhận `io.modelcontextprotocol/ui` xuất hiện trong bản đồ phần mở rộng server. Sau đó gọi `tools/list` hai lần, một lần với khả năng Apps và một lần không có nó. Phản hồi đầu tiên khai báo resource. Phản hồi thứ hai vẫn là một tool chỉ văn bản có thể sử dụng được.

Đọc `ui://notes/timeline.html`. Tìm kiếm trong HTML cho `hostOrigin` và guard `event.origin`. Hai dòng đó là bằng chứng hiển thị tối thiểu cho thấy bridge không sử dụng mục tiêu wildcard.

## Ship nó

Bài học này ship `outputs/skill-mcp-apps-spec.md`. Sử dụng nó để xem xét hợp đồng App trước khi viết mã framework. Nó buộc tác giả phải nêu rõ envelope core hiện tại, thương lượng phần mở rộng, fallback, resource UI, chính sách bộ nhớ đệm, CSP, quyền, phương thức bridge và ranh giới đồng ý.

## Bài tập

1. Thay đổi khả năng client thành một bản đồ phần mở rộng trống. Xác nhận `tools/list` giữ lại tool nhưng xóa ràng buộc UI.
2. Gửi `Mcp-Name: ui://notes/other.html` với một body đọc dòng thời gian. Xác nhận lỗi `-32020`.
3. Thay đổi resource thành `cacheScope: private`. Mô tả điều kiện cụ thể của người dùng biện minh cho nó.
4. Di chuyển script đến `https://static.example.com/app.js`. Thêm nguồn gốc đó vào `resourceDomains` và giải thích rủi ro chuỗi cung ứng mới.
5. Thêm một tool `notes_open` và định tuyến cú nhấp chuột qua host. Giữ sự phê duyệt của người dùng trong host.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa |
|------|---------|
| MCP Apps | Phần mở rộng tùy chọn cho HTML tương tác được render bởi host MCP |
| `io.modelcontextprotocol/ui` | Định danh phần mở rộng được quảng bá bởi cả hai bên |
| `ui://` | Lược đồ resource cho mẫu UI của App |
| `text/html;profile=mcp-app` | Loại MIME cho HTML của MCP App |
| `server/discover` | RPC hiện tại để khám phá giao thức và khả năng |
| `resources/list` | Phương thức liệt kê resource bắt buộc khi server quảng bá resource |
| `resultType` | Discriminator bắt buộc cho các kết quả thành công hiện đại |
| `ui/initialize` | Request bridge Apps đầu tiên, tách biệt khỏi khởi tạo core đã bị loại bỏ |
| `ui/notifications/initialized` | Thông báo sẵn sàng của Apps View được gửi sau khi host phản hồi |
| CSP | Chính sách trình duyệt hạn chế scripts, styles, images và nguồn gốc mạng |
| Text fallback | Hành vi tool được giữ lại cho host không hỗ trợ Apps |

## Đọc thêm

- [Giao thức cơ sở MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [Tổng quan về MCP Apps](https://modelcontextprotocol.io/extensions/apps/overview)
- [Hướng dẫn xây dựng MCP Apps](https://modelcontextprotocol.io/extensions/apps/build)
- [Ma trận hỗ trợ phần mở rộng chính thức](https://modelcontextprotocol.io/extensions/client-matrix)