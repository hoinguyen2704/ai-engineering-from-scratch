# Tài nguyên và Prompt MCP: Context có thể định địa chỉ cho các Stateless Server

> Các Tool thực hiện thao tác. Tài nguyên (Resource) cung cấp nội dung có thể định địa chỉ. Prompt đóng gói các mẫu tin nhắn do người dùng lựa chọn. Một MCP server tốt sẽ giữ cho các hợp đồng này tách biệt và có thể dự đoán được.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lesson 07 (Xây dựng một MCP Server), Phase 13, Lesson 09 (MCP Transports)
**Time:** ~60 phút

## Mục tiêu học tập

- Lựa chọn giữa Tool, Resource và Prompt dựa trên ý định của người dùng.
- Quảng bá bề mặt Resource và Prompt thông qua `server/discover` bắt buộc.
- Xây dựng các kết quả `resources/list` và `prompts/list` có tính tất định (deterministic).
- Áp dụng `ttlMs` và `cacheScope` mà không làm rò rỉ dữ liệu người dùng.
- Trả về lỗi JSON-RPC `-32602` cho URI tài nguyên không hợp lệ hoặc không xác định.
- Mở một luồng phản hồi POST `subscriptions/listen` và tương quan mọi sự kiện theo ID đăng ký (subscription ID).
- Coi nội dung tài nguyên và mẫu prompt là đầu ra không đáng tin cậy từ server.

## Bắt đầu với người tiêu dùng (Consumer)

Cách dễ nhất để sử dụng sai MCP là bắt đầu bằng mã triển khai. Một truy vấn cơ sở dữ liệu trở thành Tool vì các hàm rất quen thuộc. Một quy trình làm việc có thể tái sử dụng trở thành Resource vì nó được lưu trữ trong một tệp. Một Prompt trở thành chính sách ẩn vì host có thể chèn nó vào.

Hãy bắt đầu với việc ai là người chọn và họ mong đợi điều gì.

| Primitive | Ý định chính | Chủ sở hữu lựa chọn | Kết quả điển hình |
|---|---|---|---|
| Tool | Thực hiện một thao tác | Model hoặc ứng dụng | Kết quả hành động có cấu trúc |
| Resource | Đọc nội dung tại một URI | Host, ứng dụng hoặc người dùng | Nội dung văn bản hoặc nhị phân |
| Prompt | Bắt đầu quy trình tin nhắn có thể tái sử dụng | Người dùng thông qua giao diện host | Một hoặc nhiều tin nhắn prompt |

Một ghi chú tại `notes://note-1` là một Resource vì nó là nội dung có thể định địa chỉ. `delete_note` là một Tool vì nó thay đổi trạng thái. `review_note` là một Prompt vì người dùng chọn một quy trình đánh giá đã chuẩn bị sẵn.

Đừng chỉ vì muốn trông đầy đủ mà hiển thị một thao tác dưới cả ba dạng. Mỗi bề mặt bổ sung đều cần khám phá, ủy quyền, bộ nhớ đệm, xử lý lỗi, kiểm thử và tài liệu.

## Stateless Envelope 2026-07-28

Bài học này nhắm vào bản sửa đổi giao thức MCP `2026-07-28`. Không có bắt tay khởi tạo hoặc phiên giao thức trong cấu hình này. Mỗi yêu cầu mang theo phiên bản giao thức và khả năng của client trong các khóa `_meta` dành riêng.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Một server phải triển khai `server/discover`. Kết quả của nó quảng bá các phiên bản được hỗ trợ, khả năng về Resource và Prompt, danh tính triển khai và các gợi ý bộ nhớ đệm (cache hints). Client có thể gọi trực tiếp một phương thức khác, nhưng việc khám phá cung cấp cho nó một snapshot ổn định trước khi xây dựng giao diện người dùng.

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "resources": {"listChanged": true, "subscribe": true},
    "prompts": {"listChanged": true}
  },
  "ttlMs": 3600000,
  "cacheScope": "public"
}
```

Một kết quả bình thường khai báo `"resultType": "complete"`. Phản hồi `_meta` xác định triển khai phục vụ với `io.modelcontextprotocol/serverInfo`. Thông tin này hữu ích cho việc chẩn đoán. Nó không phải là danh tính xác thực. Một yêu cầu mang bản sửa đổi không được hỗ trợ sẽ trả về `-32022` với cả bản sửa đổi được yêu cầu và các bản sửa đổi mà server hỗ trợ.

Hợp đồng stateless thay đổi bản năng thiết kế của bạn. Một danh sách không thể phụ thuộc vào một lệnh gọi trước đó trên cùng một kết nối. Ủy quyền có thể thay đổi tập hợp hiển thị vì thông tin xác thực là đầu vào của yêu cầu, nhưng lịch sử kết nối thì không được phép.

## Resource là các hợp đồng URI ổn định

Resource là nội dung được xác định bởi một URI. Hãy thiết kế URI trước khi viết trình xử lý (handler).

Các thuộc tính URI tốt:

- Đủ ổn định để đánh dấu (bookmark) hoặc chuyển giữa các yêu cầu.
- Có không gian tên (namespace) thuộc về domain của server.
- Độc lập với ID tiến trình hoặc kết nối.
- Được xác thực trước khi truy cập lưu trữ.
- Được ủy quyền trên mỗi lần đọc.

`notes://note-1` tốt hơn `note-1` vì không gian tên của nó rõ ràng. Một file server có thể sử dụng các URI `file://`, nhưng nó vẫn phải kiểm tra các ranh giới thư mục đã cấu hình sau khi phân giải symlink và các phân đoạn tương đối.

`resources/list` trả về các tài nguyên hiện có thể nhìn thấy đối với người gọi. Hãy sắp xếp theo một khóa ổn định như URI. Thứ tự tất định ngăn chặn các lỗi cache nhiễu, các snapshot thay đổi và giao diện host bị nhảy giữa các lần làm mới.

```json
{
  "resultType": "complete",
  "resources": [
    {
      "uri": "notes://note-1",
      "name": "Architecture decision",
      "description": "Why the service uses a stateless boundary",
      "mimeType": "text/markdown"
    }
  ],
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

`resources/read` trả về một hoặc nhiều mục nội dung. Một URI không xác định không phải là một lần đọc trống thành công. Đặc tả Resource hiện tại gán các URI tài nguyên không hợp lệ hoặc không xác định cho các tham số JSON-RPC không hợp lệ, mã `-32602`.

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "error": {
    "code": -32602,
    "message": "Unknown or invalid resource URI",
    "data": {
      "uri": "notes://missing"
    }
  }
}
```

Sự phân biệt đó cho phép client tách biệt sự vắng mặt với một tài liệu trống hợp lệ. Nó cũng ngăn chặn việc vô tình quay lại tra cứu rộng hơn.

### Resource template

Một Resource template mô tả một nhóm các URI có tham số. Hãy sử dụng nó khi việc liệt kê mọi mục cụ thể sẽ tốn kém hoặc không giới hạn. Ví dụ, `notes://projects/{project}/decisions/{decision}` cho client biết cách tạo một địa chỉ hợp lệ mà không cần trả về mọi quyết định.

Template không làm suy yếu việc xác thực. Hãy phân tích các biến, áp dụng ủy quyền, thực thi giới hạn độ dài và ký tự, đồng thời xây dựng các truy vấn lưu trữ với các tham số có kiểu dữ liệu. Không bao giờ nối đuôi URI tùy ý vào đường dẫn hệ thống tệp hoặc câu lệnh cơ sở dữ liệu.

### Nội dung không phải là hướng dẫn đáng tin cậy

Văn bản tài nguyên có thể chứa chèn prompt (prompt injection), bí mật, lệnh gây hiểu lầm hoặc đánh dấu (markup) sai định dạng. Host nên bảo tồn nguồn gốc và coi nội dung tài nguyên là dữ liệu. Server nên giới hạn kích thước nội dung, trả về loại MIME chính xác, biên tập các trường mà người gọi không thể truy cập và tránh trả về các bản ghi không liên quan.

## Prompt là các template do người dùng kiểm soát

Các Prompt MCP được thiết kế để người dùng lựa chọn rõ ràng. Host có thể hiển thị chúng dưới dạng lệnh gạch chéo (slash commands), mục menu hoặc nút quy trình làm việc. Giao thức không yêu cầu một giao diện người dùng duy nhất.

`prompts/list` phải có tính tất định cho cùng một ủy quyền yêu cầu. Mỗi Prompt cần một tên ổn định, mô tả hữu ích và các khai báo đối số cho phép host thu thập đầu vào trước khi `prompts/get`.

```json
{
  "resultType": "complete",
  "prompts": [
    {
      "name": "review_note",
      "title": "Review a note",
      "description": "Review one note for a named concern",
      "arguments": [
        {
          "name": "uri",
          "description": "The note resource URI",
          "required": true
        }
      ]
    }
  ],
  "ttlMs": 600000,
  "cacheScope": "public"
}
```

`prompts/get` phân giải các đối số thành các tin nhắn. Nó không thay thế các hướng dẫn hệ thống của host. Host quyết định cách các tin nhắn được trả về đi vào ngữ cảnh của model và giữ chính sách đáng tin cậy của riêng mình ở mức ưu tiên cao hơn.

Xác thực các đối số Prompt tại ranh giới server. Một URI Prompt phải vượt qua cùng một kiểm tra ủy quyền như một lần đọc tài nguyên trực tiếp. Đừng biến Prompt thành một kênh phụ để vượt qua quyền truy cập tài nguyên.

## Cache hints là một phần của tính chính xác

`ttlMs` cho client biết kết quả có thể được sử dụng lại trong bao lâu. `cacheScope` mô tả ai có thể chia sẻ giá trị đã lưu trong cache đó.

| Phạm vi | Ý nghĩa | Sử dụng điển hình |
|---|---|---|
| `public` | Có thể được sử dụng lại giữa các người dùng khi ủy quyền cho phép | Danh mục Prompt công khai |
| `private` | Gắn liền với người dùng yêu cầu hoặc ngữ cảnh thông tin xác thực | Nội dung ghi chú thuộc sở hữu người dùng |

Chọn TTL dựa trên tốc độ thay đổi của dữ liệu và thiệt hại của việc dữ liệu cũ (staleness). Năm phút có thể phù hợp với danh mục Prompt công khai. Một lần đọc ghi chú riêng tư có thể sử dụng một phút.

MCP chỉ định nghĩa `public` và `private` là các giá trị `cacheScope`. Đối với kết quả chứa bí mật hoặc thay đổi nhanh chóng, hãy trả về `cacheScope: "private"` với `ttlMs: 0`, sau đó áp dụng bất kỳ quy tắc no-store nghiêm ngặt nào trong chính sách cache của host. Bản thân `no-store` không phải là một giá trị `cacheScope` của MCP.

Cache hints không bao giờ thay thế ủy quyền. Khóa cache phải bao gồm mọi chiều của yêu cầu làm thay đổi khả năng hiển thị, bao gồm tenant, người dùng, phạm vi, ngôn ngữ và con trỏ phân trang. Nếu một cache chia sẻ không thể thể hiện các chiều đó một cách an toàn, hãy sử dụng `private` với TTL bằng 0 và chính sách no-store ở cấp host.

## Đăng ký sử dụng luồng phản hồi do Client mở

Mô hình đăng ký hiện đại thay thế RPC `resources/subscribe` trước đây và điểm cuối sự kiện HTTP GET cũ.

Client gửi `subscriptions/listen` như một yêu cầu JSON-RPC bình thường. Qua Streamable HTTP, đây là một POST mà phản hồi của nó vẫn mở dưới dạng luồng SSE. Đối tượng `notifications` là một danh sách cho phép (allowlist). Server không được gửi các loại thông báo không được yêu cầu.

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "subscriptions/listen",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    },
    "notifications": {
      "resourcesListChanged": true,
      "promptsListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

ID yêu cầu là ID đăng ký. Trước bất kỳ sự kiện nào được yêu cầu, server gửi `notifications/subscriptions/acknowledged`. Bộ lọc của nó chỉ chứa tập con mà server đã chấp nhận.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "notifications": {
      "resourcesListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

Mỗi sự kiện sau đó trên luồng đó đều mang cùng metadata.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "uri": "notes://note-1"
  }
}
```

Thông báo cho biết tài nguyên đã thay đổi. Client đọc lại nó thông qua `resources/read`, tùy thuộc vào ủy quyền hiện tại. Nó không giả định rằng sự kiện chứa tài liệu mới.

Một vài đăng ký có thể chia sẻ một kênh stdio. ID đăng ký cho phép client phân tách chúng. Qua HTTP, việc đóng luồng phản hồi sẽ hủy đăng ký. Một server kết thúc luồng một cách duyên dáng sẽ trả về phản hồi `resultType: "complete"` cuối cùng tương quan với yêu cầu ban đầu.

Đừng sử dụng luồng đăng ký như một phiên giao thức. Một lần đọc sau đó vẫn là một yêu cầu hoàn chỉnh có thể tiếp cận bất kỳ instance server nào đang hoạt động.

```figure
t3-primitive-sort
```

## Interactive Lab

Sử dụng hình vẽ để phân loại năm khả năng từ một trình theo dõi dự án: chi tiết vấn đề, tạo vấn đề, mẫu đánh giá sprint, chính sách dự án và đóng vấn đề. Sau đó quyết định danh sách nào có thể được cache công khai, lần đọc nào phải giữ riêng tư và tài nguyên nào xứng đáng nhận thông báo cập nhật.

Đối với mỗi phân loại, hãy đặt tên cho người chọn. Nếu model thực hiện một hành động, hãy sử dụng Tool. Nếu host đọc nội dung được định địa chỉ bằng URI, hãy sử dụng Resource. Nếu người dùng bắt đầu một quy trình tin nhắn đã chuẩn bị sẵn, hãy sử dụng Prompt.

## Practice Lab

Chạy trình mô phỏng từ thư mục gốc của repository:

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

Kiểm tra transcript theo thứ tự này:

1. Xác nhận `server/discover` quảng bá bản sửa đổi hiện tại và cả hai khả năng.
2. Xác nhận cả hai kết quả danh sách đều được sắp xếp và sử dụng `resultType: "complete"`.
3. Xác nhận kết quả danh sách và đọc mang các gợi ý cache có chủ đích.
4. Thay đổi URI đọc thành `notes://missing` và quan sát `-32602`.
5. Xác nhận xác nhận đăng ký đi trước sự kiện tài nguyên.
6. Xác nhận sự kiện và việc đóng duyên dáng đều mang ID đăng ký `5`.

Model Python không mở kết nối HTTP thực. Nó đại diện cho các tin nhắn mà một SDK phải đặt trên luồng phản hồi có phạm vi yêu cầu. Sử dụng SDK chính thức để đóng khung và vận chuyển trong môi trường production.

## Shipped Artifact

`outputs/skill-primitive-splitter.md` là một đánh giá thiết kế có thể tái sử dụng cho việc lựa chọn primitive MCP. Nó hiện kiểm tra việc khám phá tất định, phạm vi cache, hành vi URI không hợp lệ và các bộ lọc đăng ký hiện đại.

Bài học cũng cung cấp `assets/primitive-split.svg`, một phiên bản tĩnh của ranh giới primitive và đăng ký để nghiên cứu ngoại tuyến.

## Verify It

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

Kết quả mong đợi: chương trình chính in ra một transcript JSON và lệnh kiểm tra báo cáo ít nhất mười hai bài kiểm tra vượt qua.

## Capstone Connection

Sử dụng hợp đồng này khi server capstone của bạn hiển thị kiến thức có thể định địa chỉ bên cạnh các hành động. Bao gồm một snapshot danh mục tất định, một lần đọc tài nguyên được ủy quyền, một lần phân giải Prompt, một trường hợp URI không hợp lệ và một transcript đăng ký.

Bằng chứng của bạn nên cho thấy rằng không có danh sách nào phụ thuộc vào lịch sử kết nối và một sự kiện đăng ký không bao giờ cấp quyền truy cập vào tài nguyên cơ bản.

## Exercises

1. Thêm một template tài nguyên `notes://projects/{project}/notes/{id}` và xác thực cả hai biến.
2. Thêm phân trang vào `resources/list` trong khi vẫn duy trì thứ tự tất định.
3. Thay đổi một tài nguyên thành `cacheScope: "private"` với `ttlMs: 0`, thêm chính sách no-store ở cấp host và giải thích mối đe dọa biện minh cho cả hai kiểm soát.
4. Thêm một đăng ký thay đổi danh sách Prompt và chứng minh không có sự kiện nào được gửi khi bộ lọc bỏ qua `promptsListChanged`.
5. Tạo hai đăng ký đồng thời và chứng minh mỗi sự kiện mang đúng ID yêu cầu.
6. Thêm một chủ thể ủy quyền vào trình xử lý đọc và chứng minh một mục cache không thể vượt qua các chủ thể.

## Key Terms

- **Resource:** Nội dung được định địa chỉ bằng URI do MCP server hiển thị.
- **Prompt:** Mẫu tin nhắn do người dùng kiểm soát do MCP server hiển thị.
- **Deterministic list:** Kết quả khám phá với thành viên và thứ tự ổn định cho cùng các đầu vào yêu cầu.
- **`ttlMs`:** Thời gian tươi mới của cache tính bằng mili giây.
- **`cacheScope`:** Ranh giới chia sẻ cho một kết quả được cache.
- **`subscriptions/listen`:** Một yêu cầu tồn tại lâu dài có luồng phản hồi cung cấp các thông báo được lọc rõ ràng.
- **Subscription ID:** ID yêu cầu lắng nghe ban đầu, được lặp lại trong metadata thông báo.
- **Invalid parameters:** Lỗi JSON-RPC `-32602`, được sử dụng cho URI tài nguyên không hợp lệ hoặc không xác định.
- **Unsupported protocol version:** Lỗi JSON-RPC `-32022`, bao gồm các bản sửa đổi `supported` và `requested`.
- **`server/discover`:** Phương thức server bắt buộc trả về các bản sửa đổi được hỗ trợ, khả năng, danh tính và các gợi ý cache tùy chọn.

## Đọc thêm

- [MCP 2026-07-28 Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP 2026-07-28 Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP 2026-07-28 Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Caching](https://modelcontextprotocol.io/specification/2026-07-28/basic/utilities/caching)