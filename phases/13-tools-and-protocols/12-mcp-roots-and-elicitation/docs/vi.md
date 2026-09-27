# Phạm vi tường minh và Elicitation không trạng thái (Stateless)

> Roots đã bị loại bỏ (deprecated) trong MCP 2026-07-28 và chưa bao giờ là một sandbox bảo mật. Hãy đặt phạm vi (scope) vào các đối số công cụ (tool arguments) hiển thị hoặc URI tài nguyên, ủy quyền nó trên server và sử dụng MRTR khi một công cụ thực sự cần đầu vào từ người dùng. Người dùng thấy quyết định, model thấy handle và bất kỳ instance server nào cũng có thể xử lý việc thử lại (retry).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 11 (stateless MRTR)
**Time:** ~60 phút

## Mục tiêu học tập

- Thay thế Roots đã bị loại bỏ bằng các tham số workspace tường minh, URI tài nguyên hoặc cấu hình server.
- Tách biệt các gợi ý phạm vi (scope hints) khỏi việc ủy quyền, kiểm soát đường dẫn (path containment) và sandboxing hệ điều hành.
- Cung cấp chế độ form `elicitation/create` thông qua kết quả MRTR `input_required`.
- Quảng bá hỗ trợ elicitation trong các capabilities của client theo từng request và từ chối các chế độ không được hỗ trợ.
- Xác thực `accept`, `decline` và `cancel` như các kết quả riêng biệt.
- Ràng buộc xác nhận hành động phá hủy với một principal đã xác thực, các đối số gốc, tập hợp ứng viên và thời hạn.

## Hai vấn đề trông có vẻ giống nhau

Một công cụ ghi chú nhận được yêu cầu: "Xóa báo cáo TPS cũ."

Server phải trả lời hai câu hỏi khác nhau.

1. Workspace nào có thể bị ảnh hưởng bởi thao tác này?
2. Người dùng muốn xóa ghi chú nào trong ba ghi chú khớp với yêu cầu?

Câu hỏi đầu tiên là về phạm vi và ủy quyền. Câu hỏi thứ hai là về việc làm rõ tương tác (interactive disambiguation). Việc trộn lẫn chúng dẫn đến các thiết kế nguy hiểm, chẳng hạn như coi một thư mục do client cung cấp là bằng chứng cho thấy người gọi có quyền xóa mọi thứ bên trong nó.

## Roots là bề mặt di chuyển (Migration Surface)

Các bản sửa đổi MCP trước đây cho phép client quảng bá Roots và thông báo cho server khi danh sách thay đổi. Roots chỉ là hướng dẫn mang tính thông tin. Chúng không hạn chế những gì tiến trình server có thể đọc, không ủy quyền cho người gọi và không tạo ra sandbox hệ điều hành.

MCP 2026-07-28 loại bỏ `roots/list` và `notifications/roots/list_changed` đối với các thiết kế mới. Hãy ưu tiên một trong các thay thế tường minh sau:

- Một đối số công cụ `workspaceUri` hoặc `directory` khi phạm vi thay đổi theo từng lệnh gọi.
- Một URI tài nguyên khi thao tác đã nhắm mục tiêu vào một tài nguyên cụ thể.
- Cấu hình server khi một deployment sở hữu một workspace cố định.
- Một sandbox tiến trình hoặc hệ thống tệp bị cô lập (jailed filesystem) khi mã nguồn phải được đảm bảo về mặt kỹ thuật không thể thoát ra ngoài.

Nếu một tích hợp 2026-07-28 hiện có vẫn cần `roots/list` trong giai đoạn chuyển đổi, server sẽ nhúng nó vào MRTR `inputRequests`. Nó không được gửi một yêu cầu ngược (reverse request) trực tiếp. Đó là một bộ chuyển đổi di chuyển (migration adapter); các trình xử lý mới nên chấp nhận phạm vi tường minh thay thế.

Model có thể thấy và lặp lại một handle tường minh. Phạm vi phiên truyền tải (transport-session scope) ẩn khó kiểm tra, phát lại, kiểm toán và định tuyến hơn.

### Quy tắc ba lớp

Một URI tường minh không tự ủy quyền cho chính nó. Hãy thực thi cả ba lớp:

1. **Ủy quyền (Authorization):** Principal đã xác thực này có được phép sử dụng workspace này không?
2. **Kiểm soát (Containment):** URI mục tiêu đã chuẩn hóa có nằm trong ranh giới workspace được ủy quyền không?
3. **Sandbox:** Hệ điều hành có thể ngăn chặn một server bị xâm nhập thoát ra ngoài hay không?

Server đang chạy duy trì một danh sách cho phép (allowlist) các URI workspace được ủy quyền, chuẩn hóa các đường dẫn được mã hóa phần trăm, kiểm tra ranh giới thành phần đường dẫn thực và kiểm tra lại sự kiểm soát ngay trước khi xóa.

Các kiểm tra tiền tố chuỗi (string-prefix) ngây thơ là sai lầm:

```text
allowed:   file:///work/notes
attacker:  file:///work/notes-evil/secret.md
traversal: file:///work/notes/%2e%2e/private.md
```

Cả hai đường dẫn độc hại đều bắt đầu bằng một chuỗi gây hiểu lầm. Hãy chuẩn hóa trước, sau đó so sánh các thành phần đường dẫn. Một server hệ thống tệp trong môi trường production cũng phải phòng thủ trước các cuộc tấn công liên kết tượng trưng (symbolic-link races) và ngữ nghĩa đường dẫn cụ thể của từng nền tảng.

## Elicitation vẫn tồn tại, nhưng cách thức phân phối đã thay đổi

Elicitation là tính năng client hiện tại để thu thập đầu vào của người dùng trong quá trình `tools/call`, `prompts/get` hoặc `resources/read`. Tên phương thức vẫn là `elicitation/create`. Điều thay đổi là hướng của luồng dữ liệu.

Một server 2026-07-28 không gửi yêu cầu JSON-RPC ngược. Nó trả về một `InputRequiredResult`:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "delete_choice": {
        "method": "elicitation/create",
        "params": {
          "mode": "form",
          "message": "Choose one matching note and confirm deletion.",
          "requestedSchema": {
            "type": "object",
            "properties": {
              "note_id": {
                "type": "string",
                "enum": ["note-3", "note-7", "note-14"]
              },
              "confirm": {"type": "boolean"}
            },
            "required": ["note_id", "confirm"]
          }
        }
      }
    },
    "requestState": "integrity-protected-delete-state"
  }
}
```

Host sẽ hiển thị form. Người dùng có thể chấp nhận, từ chối rõ ràng hoặc bỏ qua. Sau đó, client sẽ thử lại `tools/call` gốc với một id mới:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes_delete",
    "arguments": {
      "workspaceUri": "file:///Users/alice/Documents/Notes",
      "title": "TPS report"
    },
    "inputResponses": {
      "delete_choice": {
        "action": "accept",
        "content": {"note_id": "note-14", "confirm": true}
      }
    },
    "requestState": "integrity-protected-delete-state",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

Không có phiên giao thức nào giữa hai lệnh gọi. Server xác minh trạng thái được phản hồi (echoed state), xác thực phản hồi dựa trên schema mong đợi, kiểm tra xem ghi chú đã chọn có nằm trong tập hợp ứng viên đã ký hay không, ủy quyền lại workspace, kiểm tra lại sự kiểm soát và sau đó thực hiện xóa.

## Đàm phán Capability theo từng Request

Một client hỗ trợ elicitation chế độ form sẽ khai báo:

```json
{
  "io.modelcontextprotocol/clientCapabilities": {
    "elicitation": {"form": {}}
  }
}
```

Một capability elicitation trống, `"elicitation": {}`, vẫn tương đương với hỗ trợ chỉ-form để đảm bảo tính tương thích. `"elicitation": {"form": {}}` tường minh cũng hỗ trợ chế độ form. Khai báo chỉ-URL, `"elicitation": {"url": {}}`, thì không. Server không được nhúng một chế độ vắng mặt trong các capabilities của request hiện tại, ngay cả khi một request trước đó đã quảng bá nó.

Mỗi request cũng mang theo `io.modelcontextprotocol/protocolVersion`. Một phiên bản bị thiếu hoặc không phải chuỗi sẽ trả về `-32602`. Một chuỗi không được hỗ trợ sẽ trả về `-32022` với dữ liệu `supported` và `requested` chính xác. Hỗ trợ elicitation bị thiếu hoặc chỉ-URL sẽ trả về `-32021` với `data.requiredCapabilities` được đặt thành `{"elicitation":{"form":{}}}`.

Một envelope không có JSON-RPC `id` là một thông báo (notification). Hãy xử lý nó mà không phát ra phản hồi thành công hoặc lỗi JSON-RPC. Trên Streamable HTTP, một thông báo được chấp nhận sẽ nhận được `202 Accepted` không có nội dung.

`clientInfo` nên được bao gồm để chẩn đoán, nhưng nó là thông tin tự báo cáo và không thể xác định người dùng để ủy quyền.

Server triển khai `server/discover` và trả về `supportedVersions`, các capabilities, `ttlMs` và `cacheScope` với `resultType: "complete"`. Nó không quảng bá Roots cho thiết kế hiện đại này. Vì nó quảng bá các công cụ, nó cũng triển khai `tools/list` bắt buộc. Kết quả đó trả về descriptor `notes_delete` xác định, đối tượng `inputSchema` hợp lệ, metadata định danh server và các gợi ý cache công khai.

## Chế độ Form

Chế độ form sử dụng một JSON Schema hạn chế được thiết kế cho các hộp thoại dễ sử dụng. Gốc là một đối tượng và các thuộc tính của nó là các trường nguyên thủy phẳng hoặc các mảng enum được hỗ trợ. Các đối tượng lồng sâu và các schema tài liệu đa năng không thuộc về hộp thoại xác nhận.

Sử dụng chế độ form cho:

- chọn một trong nhiều ứng viên;
- xác nhận một thao tác phá hủy;
- thu thập các tùy chọn không nhạy cảm;
- thu thập một số lượng nhỏ các giá trị mà người dùng, không phải model, phải quyết định.

Không sử dụng chế độ form cho mật khẩu, API key, access token hoặc thông tin xác thực thanh toán. Những bí mật đó sẽ đi qua MCP client và có thể lọt vào logs hoặc ngữ cảnh của model.

Server xác thực lại nội dung được trả về. Việc xác thực form phía client cải thiện UX nhưng không tạo ra sự tin tưởng.

## Chế độ URL

Chế độ URL gửi một URL web an toàn cho một tương tác ngoài băng tần (out-of-band):

```json
{
  "method": "elicitation/create",
  "params": {
    "mode": "url",
    "message": "Connect the report service to continue.",
    "url": "https://mcp.example.com/connect/report-service"
  }
}
```

Sử dụng nó khi thông tin nhạy cảm phải đi trực tiếp đến một luồng web do server kiểm soát, chẳng hạn như ủy quyền bên thứ ba. Client hiển thị toàn bộ đích đến và nhận sự đồng ý trước khi mở nó. Nó không được phép prefetch URL.

Phản hồi `accept` có nghĩa là người dùng đã đồng ý mở URL. Nó không chứng minh luồng bên ngoài đã hoàn tất. Khi thử lại, server kiểm tra trạng thái của chính nó và hoàn tất hoặc trả về một kết quả `input_required` khác.

Elicitation URL không phải là sự thay thế cho việc ủy quyền giữa MCP client và MCP server. Nó dành cho một tương tác bên ngoài mà MCP server cần thực hiện thay mặt người dùng. Server phải ràng buộc người dùng trình duyệt với cùng một principal đã xác thực đã bắt đầu thao tác MCP.

## Các nhánh phản hồi

Hãy coi các hành động là các quyết định sản phẩm, không phải bí danh:

| Hành động | Ý nghĩa | Hành vi server an toàn |
|--------|---------|----------------------|
| `accept` | Người dùng đã gửi tương tác | Xác thực nội dung và tiếp tục |
| `decline` | Người dùng từ chối rõ ràng | Trả về kết quả từ chối hoàn chỉnh, không lỗi |
| `cancel` | Người dùng bỏ qua hoặc không thể hoàn thành | Dừng an toàn và cho phép thử lại sau |

Không bao giờ diễn giải nội dung bị thiếu là sự đồng ý. Không bao giờ chuyển đổi việc từ chối thành một vòng lặp nhắc nhở lặp đi lặp lại.

## Bảo vệ trạng thái MRTR phá hủy

Danh sách ứng viên không thể chỉ tồn tại trong một lời nhắc hoặc giá trị Base64 không được ký. Client kiểm soát mọi thứ nó gửi lại.

Bài học này ký một payload trạng thái bao gồm:

- principal đã xác thực;
- phương thức khởi tạo;
- digest của `workspaceUri` và `title`;
- các id ghi chú được phép hiển thị trong form;
- giai đoạn thao tác;
- thời hạn ngắn.

Trước khi thay đổi, server cũng kiểm tra bản ghi ghi chú trực tiếp. Điều này giúp bắt được các cuộc đua xóa (deletion races) và mục tiêu bị di chuyển ra ngoài workspace sau khi form được hiển thị.

Đối với một hành động tài chính hoặc không thể đảo ngược một lần, HMAC đơn thuần không ngăn được trạng thái hợp lệ bị phát lại trong thời hạn của nó. Hãy lưu trữ và tiêu thụ một nonce chính xác một lần trong một kho lưu trữ phát lại (replay store) được chia sẻ bởi mọi instance trình xử lý. Bài học này chèn một kho lưu trữ có giới hạn, được cắt tỉa theo TTL và giữ yêu cầu nguyên tử của nó trong khi thực hiện xóa trong bộ nhớ. Một cơ sở dữ liệu production nên kết hợp yêu cầu nonce và thay đổi trong một giao dịch hoặc ranh giới ghi có điều kiện tương đương.

Xác thực tương tác trước khi yêu cầu nonce. Một phản hồi bị lỗi hoặc `cancel` không thực hiện thay đổi nào và để trạng thái có thể thử lại cho đến khi hết hạn. Một `decline` tường minh là kết thúc, vì vậy bài học sẽ tiêu thụ nonce mà không xóa bất cứ thứ gì.

```figure
t3-roots-boundary
```

## Xây dựng

`code/main.py` minh họa một công cụ `notes_delete` hiện đại:

- `tools/list` trả về một descriptor xác định, có thể cache với workspace và schema tiêu đề bắt buộc.
- Phạm vi là một đối số `workspaceUri` tường minh.
- Cấu hình server ủy quyền workspace đó cho principal của bài học.
- Chuẩn hóa URI từ chối sự nhầm lẫn tiền tố và duyệt đường dẫn được mã hóa.
- Mọi thao tác xóa phá hủy đều yêu cầu elicitation chế độ form.
- Elicitation truyền đi bên trong `resultType: "input_required"`.
- `requestState` đã ký ràng buộc danh sách ứng viên chính xác và các đối số gốc.
- Kho lưu trữ phát lại được chèn vào từ chối cùng một trạng thái đã chấp nhận hoặc từ chối trên các instance server.
- Việc thử lại sử dụng một id request mới và trả về `resultType: "complete"`.

Kho lưu trữ dữ liệu nằm trong bộ nhớ để hành vi giao thức dễ kiểm tra. Các quy tắc bảo mật vẫn giữ nguyên với cơ sở dữ liệu.

## Sử dụng

Từ gốc repository:

```bash
cd phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/code
python3 main.py
python3 -m unittest discover tests -v
```

Các điểm kiểm tra dự kiến:

- Khám phá quảng bá các công cụ không có Roots.
- Khám phá công cụ trả về `notes_delete` với `resultType`, định danh server và gợi ý cache.
- Request id `1` trả về form trong `inputRequests.delete_choice`.
- Request id `2` phản hồi trạng thái đã ký và hoàn tất việc xóa.
- Một đường dẫn tiền tố và một đường dẫn duyệt được mã hóa đều không vượt qua kiểm tra kiểm soát.
- Một tiêu đề đã thay đổi không thể sử dụng lại trạng thái xác nhận gốc.
- Một sự từ chối để lại ghi chú không thay đổi.
- Hai đối tượng server chia sẻ ghi chú và trạng thái phát lại không thể cùng thực hiện một xác nhận.
- Các khai báo form trống và tường minh hoạt động, trong khi hỗ trợ chỉ-URL trả về các yêu cầu form `-32021` chính xác.
- Các lỗi phiên bản không được hỗ trợ sử dụng hình dạng dữ liệu `-32022` chính xác.
- Một thông báo không có id không tạo ra phản hồi JSON-RPC.

## Phát hành

`outputs/skill-elicitation-form-designer.md` thiết kế phạm vi tường minh, kiểm tra ủy quyền, form MRTR, các nhánh phản hồi và ràng buộc trạng thái. Nó từ chối coi Roots đã bị loại bỏ là một sandbox hoặc thu thập bí mật thông qua chế độ form.

## Bài tập

1. Thay thế kho lưu trữ phát lại trong bộ nhớ bằng SQLite. Sử dụng một giao dịch để yêu cầu nonce và xóa ghi chú, sau đó chứng minh hai tiến trình không thể cùng commit.
2. Thêm đàm phán capability `url` và luồng thiết lập ngoài băng tần. Giữ thông tin xác thực bên thứ ba ra khỏi `inputResponses`.
3. Thay thế bản đồ ghi chú trong bộ nhớ bằng cơ sở dữ liệu SQLite tạm thời. Kiểm tra lại ủy quyền và sự kiểm soát bên trong giao dịch thay đổi.
4. Thêm chính sách liên kết tượng trưng cho việc triển khai hệ thống tệp thực. Giải thích tại sao việc kiểm soát từ vựng URI đơn thuần không thể ngăn chặn việc thoát ra qua symlink.
5. Thiết kế một bộ chuyển đổi 2025-11-25 ánh xạ đầu ra của trình xử lý MRTR hiện đại sang elicitation do server khởi tạo cũ. Giữ nó tách biệt khỏi trình xử lý hiện tại.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa trong 2026-07-28 |
|------|------------------------|
| Roots | Gợi ý workspace mang tính thông tin đã bị loại bỏ, không phải ủy quyền hay sandboxing |
| Phạm vi tường minh | Handle workspace, thư mục hoặc tài nguyên hiển thị trong các đối số request |
| Kiểm soát (Containment) | Kiểm tra thành phần đường dẫn đã chuẩn hóa giúp giữ mục tiêu trong ranh giới |
| Elicitation | Tính năng client để thu thập đầu vào người dùng trong thao tác MCP |
| Chế độ Form | Đầu vào người dùng có cấu trúc trong băng tần sử dụng schema phẳng hạn chế |
| Chế độ URL | Tương tác ngoài băng tần cho các luồng công việc nhạy cảm hoặc bên ngoài |
| MRTR | Kết quả yêu cầu đầu vào không trạng thái theo sau bởi một lần thử lại mới |
| `requestState` | Trạng thái mờ được phản hồi chính xác và được kiểm tra tính toàn vẹn bởi server |
| Từ chối (Decline) | Sự từ chối rõ ràng của người dùng |
| Hủy (Cancel) | Bỏ qua hoặc tương tác không hoàn chỉnh mà không có sự chấp thuận |

## Tương thích cũ

Đối với một peer được ghim ở 2025-11-25, `roots/list`, `notifications/roots/list_changed` và `elicitation/create` do server khởi tạo trực tiếp vẫn có thể tồn tại. Hãy gắn nhãn bộ chuyển đổi đó là legacy. Không cho phép danh sách Root cũ vượt qua ủy quyền server và không mang các giả định phiên giao thức vào trình xử lý hiện đại.

## Đọc thêm

- [MCP 2026-07-28 Elicitation](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Roots deprecation](https://modelcontextprotocol.io/specification/2026-07-28/client/roots)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)