# Bảo mật MCP: Metadata bị nhiễm độc, Định tuyến và Trạng thái MRTR

> Stateless (phi trạng thái) không có nghĩa là trustless (không cần tin cậy). Điều đó có nghĩa là mọi yêu cầu đều phơi bày các bằng chứng mà server và gateway cần để xác thực cuộc gọi một cách độc lập.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~60 phút

## Mục tiêu học tập

- Coi mô tả công cụ (tool descriptions), chú thích, thông tin client và thông tin server là dữ liệu không đáng tin cậy.
- Phát hiện metadata bị nhiễm độc, thay đổi descriptor và xung đột tên giữa các server.
- Xác thực metadata yêu cầu 2026-07-28 và các header định tuyến Streamable HTTP.
- Bảo vệ MRTR `requestState` khỏi việc bị can thiệp và ràng buộc xác nhận với các đối số chính xác.
- Áp dụng ủy quyền và giới hạn tốc độ (rate limits) cho một principal, không phải cho một phiên giao thức đã bị xóa.

## Vấn đề

Một model đọc mô tả công cụ để quyết định nên gọi cái gì. Một bộ định tuyến (router) đọc tên công cụ để quyết định gửi yêu cầu đến đâu. Một người dùng đọc nhãn để quyết định phê duyệt cái gì. Một descriptor độc hại có thể nhắm mục tiêu vào cả ba.

Hướng dẫn bảo mật MCP chính thức rất trực tiếp: các mô tả và chú thích nên được coi là không đáng tin cậy trừ khi chúng đến từ một server đáng tin cậy. Ngay cả khi đó, sự tin cậy trong triển khai có thể thay đổi. Một bản cập nhật server, gói bị xâm nhập, lỗi registry hoặc lỗi hợp nhất gateway có thể làm thay đổi những gì model nhìn thấy.

Giao thức hiện tại cũng thay đổi ranh giới bảo mật. Trong phiên bản 2026-07-28, không có bắt tay cốt lõi (core handshake) và không có phiên truyền tải (transport session). Một thiết kế bảo mật chỉ dựa vào `Mcp-Session-Id` để phê duyệt, giới hạn tốc độ hoặc lịch sử kiểm toán không phải là một thiết kế hiện tại.

## Khái niệm

### Bảy bề mặt tấn công cần kiểm tra

Sử dụng danh sách cụ thể thay vì hướng dẫn mơ hồ là phải cẩn thận.

1. **Metadata poisoning (Nhiễm độc metadata):** Một mô tả chứa các hướng dẫn không liên quan đến hành vi công cụ đã khai báo.
2. **Descriptor rug pull:** Một tên, mô tả, schema hoặc chú thích đã được phê duyệt trước đó bị thay đổi.
3. **Cross-server shadowing (Che khuất giữa các server):** Hai backend cùng hiển thị một tên công cụ không đủ điều kiện (unqualified name) và bộ định tuyến chọn một cái một cách âm thầm.
4. **Header and body confusion:** `Mcp-Method` hoặc `Mcp-Name` không khớp với yêu cầu JSON-RPC.
5. **Capability escalation (Leo thang quyền hạn):** Một peer tuyên bố một extension hoặc tính năng client và server nhầm lẫn tuyên bố đó là sự ủy quyền.
6. **MRTR state tampering:** Một client thay đổi `requestState`, trả lời một câu hỏi khác hoặc tái sử dụng xác nhận với các đối số khác nhau.
7. **Supply-chain identity confusion:** Một tên hiển thị quen thuộc bị coi là bằng chứng về danh tính của nhà xuất bản hoặc server.

Các bề mặt này chồng chéo lên nhau. Hash pinning giúp ích với các thay đổi descriptor nhưng không chứng minh được rằng descriptor đầu tiên là an toàn. Quét tĩnh (static scanning) bắt được các cụm từ rõ ràng nhưng không bắt được các hướng dẫn tinh vi. Namespacing ngăn chặn một lớp xung đột nhưng không ngăn được một server có namespace độc hại. Hãy xếp chồng các biện pháp kiểm soát.

### Envelope yêu cầu hiện tại là bằng chứng, không phải danh tính

Mọi yêu cầu 2026-07-28 đều chứa:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "elicitation": {"form": {}}
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "security-lab",
      "version": "1.0.0"
    }
  }
}
```

Xác thực phiên bản và hình dạng capability trên mọi yêu cầu. Sử dụng các capability để chọn hình dạng phản hồi tương thích. Không sử dụng `clientInfo` như một principal đã xác thực. Nó là thông tin tự báo cáo.

Cảnh báo tương tự áp dụng cho `io.modelcontextprotocol/serverInfo` trong metadata kết quả. Nó hữu ích cho nhật ký và gỡ lỗi. Nó không phải là chứng chỉ, bằng chứng registry hoặc quyết định ủy quyền.

### Xác thực định tuyến trước chính sách

Đối với `tools/call`, Streamable HTTP bao gồm:

```text
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.export
```

Phương thức header phải bằng phương thức body. Tên header phải bằng `params.name`. Từ chối sự bất đồng với `-32020` trước khi chọn backend, áp dụng RBAC hoặc tiêu thụ token giới hạn tốc độ.

Thứ tự này loại bỏ một sự mơ hồ phổ biến: một thành phần ủy quyền cho body trong khi thành phần khác định tuyến theo header.

Việc xác thực đường truyền tuân theo một trình tự chính xác. Xác thực các kiểu JSON-RPC và metadata, so sánh các giá trị header với body, sau đó kiểm tra xem phiên bản khớp có được hỗ trợ hay không. Một header không khớp sẽ trả về HTTP 400 với `-32020`. Nếu header và body đồng ý về một phiên bản không được hỗ trợ, trả về HTTP 400 với `-32022` và `data` chính xác là `{"supported":["2026-07-28"],"requested":"<actual>"}`. Một phương thức không xác định trả về HTTP 404 với `-32601`.

Mỗi đối tượng lỗi bao gồm `data` tùy chọn khi hợp đồng cần thông tin khôi phục có cấu trúc. Một thông báo (notification) không có `id`, vì vậy nó không bao giờ nhận được phản hồi thành công hoặc lỗi JSON-RPC. Một thông báo HTTP được chấp nhận sẽ trả về 202 với body trống.

### Ghim toàn bộ descriptor

Chỉ riêng hash mô tả sẽ bỏ sót các thay đổi về schema và chú thích. Hãy chuẩn hóa và hash các trường descriptor mà người dùng đã phê duyệt:

```python
normalized = json.dumps(tool, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(normalized.encode()).hexdigest()
```

Lưu trữ digest dưới một khóa đủ điều kiện (qualified key) như `notes.export`, cùng với bằng chứng nhà xuất bản và thời gian phê duyệt bên ngoài ví dụ này.

Mỗi khi làm mới:

- Khóa không xác định: cách ly cho đến khi xem xét.
- Cùng khóa, digest khác: cách ly như một vụ "rug pull" cho đến khi được phê duyệt lại.
- Tên không đủ điều kiện trùng lặp: yêu cầu namespacing xác định.
- Scanner hit: chặn và xem xét toàn bộ descriptor.

Sự bình đẳng của hash chứng minh tính ổn định, không phải tính an toàn. Một descriptor bị nhiễm độc vẫn sẽ bị nhiễm độc khi được ghim hoàn hảo.

### Quét tĩnh là một cái bẫy

Các mẫu đơn giản có thể gắn cờ các thẻ vai trò, ghi đè hướng dẫn, che giấu, truy cập bí mật và các đích mạng bị che khuất. Chúng đủ rẻ cho thời điểm cài đặt và CI.

Chúng không phải là bằng chứng ngữ nghĩa. Một mô tả an toàn có thể chứa một cụm từ bị gắn cờ trong một cảnh báo hợp lệ. Một mô tả độc hại có thể tránh mọi cụm từ. Hãy coi kết quả của scanner là bằng chứng xem xét, không phải là điểm số vô tội tự động.

### Namespace trước khi hợp nhất

Giả sử hai server đều hiển thị `search`. Đừng bao giờ để thứ tự khám phá quyết định cái nào thắng.

```text
notes.search
issues.search
```

Tên đủ điều kiện là tên gateway công khai. Ghi lại ánh xạ backend một cách riêng biệt. Các tên ổn định giúp việc phê duyệt, kiểm toán, hash pins và định tuyến `Mcp-Name` tham chiếu đến cùng một đối tượng.

### Capability là các tuyên bố tương thích

`clientCapabilities` trên mỗi yêu cầu cho server biết các tính năng giao thức nào mà client có thể xử lý. Nó không cấp cho client quyền truy cập vào các công cụ, dữ liệu hoặc hành động.

Ủy quyền vẫn đến từ principal đã xác thực và chính sách tài nguyên. Trình tự là:

1. Xác thực thông tin đăng nhập truyền tải.
2. Xác thực phiên bản, header và hình dạng yêu cầu.
3. Kiểm tra tính tương thích của capability.
4. Ủy quyền principal, công cụ, tài nguyên và đối số.
5. Thực thi hoặc yêu cầu đầu vào từ người dùng.

### Bảo vệ xác nhận MRTR phi trạng thái

Một công cụ quan trọng có thể cần sự xác nhận của người dùng. MCP hiện tại sử dụng Multi Round-Trip Requests thay vì callback từ server đến client.

Phản hồi đầu tiên:

```json
{
  "resultType": "input_required",
  "inputRequests": {
    "confirm": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Export notes to archive?",
        "requestedSchema": {
          "type": "object",
          "properties": {
            "confirm": {"type": "boolean"}
          },
          "required": ["confirm"]
        }
      }
    }
  },
  "requestState": "opaque-integrity-protected-value"
}
```

Client lấy đầu vào và thử lại phương thức gốc với một id JSON-RPC mới:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes.export",
    "arguments": {"query": "private", "destination": "archive"},
    "requestState": "opaque-integrity-protected-value",
    "inputResponses": {
      "confirm": {
        "action": "accept",
        "content": {"confirm": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

Mỗi giá trị `inputRequests` là một yêu cầu nhúng hoàn chỉnh với `method` và `params`. Khóa của nó phải khớp với mục tương ứng trong `inputResponses`. Một biểu mẫu gợi ý (form elicitation) sử dụng `requestedSchema` ở gốc đối tượng và client phải đã khai báo capability gợi ý biểu mẫu trước khi server yêu cầu nó.

Capability hiện tại có hai khai báo biểu mẫu hợp lệ. `{"elicitation":{}}` hỗ trợ ngầm định gợi ý biểu mẫu, trong khi `{"elicitation":{"form":{}}}` tuyên bố nó một cách rõ ràng. Một khai báo chỉ có URL như `{"elicitation":{"url":{}}}` không hỗ trợ yêu cầu biểu mẫu. Server trả về HTTP 400 với `-32021` và `data.requiredCapabilities` bằng `{"elicitation":{"form":{}}}`.

Coi `requestState` là đầu vào thù địch. Ký hoặc mã hóa nó, xác thực nó và ràng buộc nó với phương thức, công cụ, đối số chính xác, mục đích, thời hạn, principal và một nonce dùng một lần khi việc phát lại (replay) là quan trọng. Mã bài học sử dụng HMAC và khớp đối số chính xác để làm cho ranh giới trở nên rõ ràng.

Sổ cái nonce không được nằm trong một đối tượng gateway. Model có thể chạy được sẽ tiêm một kho lưu trữ phát lại (replay store) có giới hạn, được cắt tỉa TTL, có thể được chia sẻ bởi nhiều instance gateway. Yêu cầu nguyên tử của nó là ranh giới thực thi: chỉ một sự chấp nhận đã được xác thực hoặc sự từ chối cuối cùng rõ ràng mới tiêu thụ trạng thái. Một phản hồi bị lỗi hoặc `cancel` sẽ không thực thi gì cả và vẫn có thể thử lại cho đến khi hết hạn. Một đội ngũ sản xuất cần yêu cầu có điều kiện tương tự trong bộ lưu trữ bền vững được chia sẻ.

Không lưu trữ ngữ cảnh xác nhận ẩn trong một phiên giao thức. Bất kỳ instance server nào cũng có thể xác thực việc thử lại.

### Quy tắc hai cho các cuộc gọi rủi ro cao

Phân loại một cuộc gọi theo ba trục:

- Nó tiêu thụ đầu vào không đáng tin cậy.
- Nó có thể truy cập dữ liệu nhạy cảm.
- Nó gây ra một hành động bên ngoài quan trọng.

Một bước tự động duy nhất không nên kết hợp cả ba. Hãy chia nhỏ nó, giảm đặc quyền hoặc yêu cầu đầu vào rõ ràng từ người dùng thông qua MRTR. Đây là một heuristic thiết kế, không phải là một capability của giao thức.

### Giảm quyền hạn trước khi thực thi

Tính phi trạng thái đơn thuần không phải là sự an toàn. Nó loại bỏ lịch sử giao thức ẩn, nhưng một yêu cầu tự chứa vẫn có thể yêu cầu một trình xử lý (handler) có quyền hạn quá mức để làm rò rỉ dữ liệu hoặc thực hiện thay đổi không thể đảo ngược. Sự an toàn đến từ việc giảm quyền hạn tại mỗi ranh giới:

1. **Typed verb:** Hiển thị một thao tác có giới hạn như `archive_note`, không phải là công cụ `run` hoặc `request` chung chung có thể thể hiện các quyền hạn không liên quan.
2. **Validated arguments:** Sử dụng schema đóng khi thực tế, từ chối các trường không xác định, chuẩn hóa định danh một lần, giới hạn kích thước và xác thực đích, tenant và quyền sở hữu tài nguyên trước khi đánh giá chính sách.
3. **Current authorization:** Ràng buộc principal đã xác thực với động từ, tài nguyên, môi trường và các đối số đã chuẩn hóa chính xác. Các chú thích công cụ và capability của client không cấp quyền hạn này.
4. **Action-bound approval:** Đối với một cuộc gọi quan trọng, ràng buộc sự phê duyệt với digest của động từ đã nhập và các đối số đã chuẩn hóa, cộng với principal, thời hạn và chính sách dùng một lần. Bất kỳ trường nào bị thay đổi đều yêu cầu một quyết định mới.
5. **First-class refusal:** Mô hình hóa sự từ chối, phê duyệt hết hạn, người dùng từ chối và đích không an toàn như các kết quả thông thường không thực thi tác dụng phụ. Không chuyển đổi sự từ chối thành một công cụ dự phòng yếu hơn.
6. **Redacted audit evidence:** Ghi lại ai đã hỏi, descriptor và phiên bản chính sách nào đã được sử dụng, mục tiêu đã chuẩn hóa nào được ủy quyền, tại sao quyết định cho phép hoặc từ chối và liệu việc thực thi đã bắt đầu hay chưa. Lưu trữ digest hoặc các giá trị đã biên tập thay vì bí mật.

Mỗi bước thu hẹp những gì thành phần tiếp theo có thể làm. Trình xử lý cuối cùng sẽ nhận được một lệnh miền đã được xác thực, không phải văn bản model thô cộng với thông tin đăng nhập rộng rãi. Lặp lại toàn bộ chuỗi khi thử lại MRTR, cập nhật tác vụ hoặc cuộc gọi được chuyển tiếp bởi gateway. Một sự phê duyệt trước đó không biến các yêu cầu sau này thành lưu lượng phiên đáng tin cậy.

### Các đường dẫn tương tác hiện tại và cũ

Roots, Sampling và Logging đã bị loại bỏ đối với các triển khai 2026-07-28 mới. Gateway có thể giữ lại mã kênh yêu cầu cũ chỉ như một đường dẫn tương thích được kiểm soát theo phiên bản.

Đừng xây dựng biện pháp phòng thủ mới xung quanh bộ giới hạn lấy mẫu theo phiên. Áp dụng hạn ngạch cho principal đã xác thực, nhà phát hành, tài nguyên, công cụ và cửa sổ thời gian. Đối với công việc tương tác hiện tại, hãy kiểm tra các yêu cầu và phản hồi đầu vào MRTR.

### Kiểm tra truyền tải phi trạng thái

- Chấp nhận các thông điệp MCP hiện đại tại endpoint POST duy nhất.
- Trả về 405 cho GET và DELETE hiện đại.
- Không tạo hoặc phụ thuộc vào `Mcp-Session-Id`.
- Bỏ qua các header phiên và phát lại cũ như các đầu vào quyền hạn.
- Trả về JSON hoặc SSE theo phạm vi yêu cầu cho POST đó.
- Chỉ sử dụng `subscriptions/listen` cho các thông báo thay đổi dài hạn đã chọn tham gia.

```figure
tp-tool-poisoning
```

## Xây dựng nó

`code/main.py` triển khai một mô hình gateway bảo mật trong tiến trình nhỏ. Nó chuẩn hóa và ghim các descriptor công cụ đầy đủ, báo cáo việc nhiễm độc metadata và che khuất, xác thực envelope yêu cầu hiện đại và các giá trị định tuyến, đồng thời thực hiện xuất khẩu xác nhận hai vòng với `requestState` đã ký và một kho lưu trữ phát lại được chia sẻ đã tiêm.

Model bắt đầu sau khi một adapter HTTP đã phân tích cú pháp body JSON và các header định tuyến. Nó không xác thực `Content-Type` hoặc `Accept`. Kết nối cùng một bộ điều phối với adapter Streamable HTTP hoàn chỉnh của Bài học 09, yêu cầu `Content-Type: application/json` và giá trị `Accept` chứa cả `application/json` và `text/event-stream`.

Chạy nó:

```bash
cd phases/13-tools-and-protocols/15-mcp-security-tool-poisoning
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Mẫu này cố tình làm biến đổi một descriptor. Scanner và so sánh digest tạo ra các phát hiện độc lập. Sau đó, việc xuất khẩu chứng minh phản hồi `input_required` và việc thử lại phi trạng thái.

## Sử dụng nó

Thay thế `SAFE_TOOLS` bằng một snapshot đã chuẩn hóa từ các server đã phê duyệt của riêng bạn. Giữ thông tin đăng nhập và bí mật bên ngoài snapshot. Xem xét mọi descriptor mới hoặc đã thay đổi trước khi cập nhật digest của nó.

Tại gateway, chạy các kiểm tra tương tự trong quá trình khám phá và một lần nữa trước khi gửi đi. Một bộ nhớ cache có thể giảm công việc khám phá, nhưng một sự phê duyệt được lưu trong cache phải hết hạn hoặc bị vô hiệu hóa khi descriptor thay đổi.

## Gửi nó

Bài học này gửi `outputs/skill-mcp-threat-model.md`. Nó tạo ra một mô hình đe dọa giao thức hiện tại trên các ranh giới metadata, định tuyến, capability, ủy quyền, MRTR, bộ nhớ cache, registry và tính tương thích.

## Bài tập

1. Ràng buộc principal đã xác thực và quyết định ủy quyền hiện tại với trạng thái MRTR đã niêm phong, sau đó từ chối thử lại dưới một principal khác.
2. Thay thế kho lưu trữ phát lại trong bộ nhớ bằng một chèn có điều kiện bền vững và chứng minh hai tiến trình không thể cùng yêu cầu một nonce.
3. Tiêm một lỗi sau khi yêu cầu phát lại nhưng trước khi xuất khẩu mô phỏng. Xác định và kiểm tra quy tắc giao dịch hoặc tính lũy đẳng (idempotency) giúp việc khôi phục an toàn.
4. Thay đổi `inputSchema` của một công cụ mà không thay đổi mô tả của nó. Xác nhận việc ghim toàn bộ descriptor sẽ bắt được nó.
5. Thêm một chính sách từ chối bộ nhớ cache công khai khi `tools/list` khác nhau theo principal.
6. Mô hình hóa một server cũ phía sau gateway. Đặt tất cả hành vi bắt tay và phiên đằng sau một nhánh tương thích `2025-11-25` rõ ràng.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa |
|------|---------|
| Metadata poisoning | Các hướng dẫn hoặc tuyên bố lừa đảo được nhúng trong descriptor công cụ |
| Rug pull | Thay đổi đối với descriptor đã được phê duyệt trước đó |
| Tool shadowing | Định tuyến mơ hồ do tên không đủ điều kiện trùng lặp |
| Header mismatch | Sự bất đồng giữa header định tuyến và body JSON-RPC, lỗi `-32020` |
| Hash pin | Digest của toàn bộ descriptor đã được phê duyệt |
| MRTR | Mẫu phản hồi và thử lại phi trạng thái cho đầu vào do server yêu cầu |
| `requestState` | Giá trị khứ hồi mờ đục phải được coi là đầu vào không đáng tin cậy |
| Capability declaration | Tuyên bố về tính tương thích của giao thức, không phải ủy quyền |
| Implicit form support | Một đối tượng capability `elicitation` trống, tương đương với hỗ trợ biểu mẫu |
| Qualified tool name | Tên gateway ổn định như `notes.search` |

## Đọc thêm

- [Hướng dẫn bảo mật và tin cậy MCP](https://modelcontextprotocol.io/specification/2026-07-28#security-and-trust--safety)
- [Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Truyền tải Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Các tính năng không còn được dùng (Deprecated)](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)