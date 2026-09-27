# Ủy quyền MCP: CIMD, Issuer Binding, PKCE và Step-Up

> Một yêu cầu MCP từ xa là không trạng thái (stateless), nhưng việc ủy quyền của nó không ẩn danh. Hãy gắn mọi thông tin xác thực với issuer đã tạo ra nó và mọi token với tài nguyên nhận nó.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 15 (security)
**Time:** ~90 minutes

## Mục tiêu học tập

- Khám phá các authorization server thông qua metadata của tài nguyên được bảo vệ.
- Ưu tiên Client ID Metadata Documents thay vì Dynamic Client Registration đã lỗi thời.
- Khai báo đúng `application_type` khi không thể tránh khỏi đường dẫn tương thích DCR.
- Xác thực `iss` của phản hồi ủy quyền và cô lập thông tin xác thực theo issuer.
- Sử dụng PKCE, resource indicators, xác thực audience và các scope tăng dần.
- Gửi các yêu cầu MCP 2026-07-28 đã được ủy quyền mà không cần phiên giao thức.

## Vấn đề

Một MCP server từ xa có thể đọc các bản ghi riêng tư, ghi vào các hệ thống bên ngoài hoặc kích hoạt các tác vụ tốn kém. Xác thực cho nó biết ai đã cung cấp thông tin xác thực. Ủy quyền phải trả lời được:

- Authorization server nào đã cấp thông tin xác thực?
- Token này dành cho tài nguyên MCP nào?
- Client và redirect URI nào đã hoàn tất luồng?
- Người dùng đã phê duyệt những thao tác nào?
- Yêu cầu cụ thể này có còn phù hợp với sự phê duyệt đó không?

Cấu hình ủy quyền 2026-07-28 thắt chặt việc đăng ký client và xử lý issuer. Nó ưu tiên Client ID Metadata Documents, loại bỏ Dynamic Client Registration, yêu cầu `application_type` chính xác trên DCR, xác thực phản hồi issuer theo RFC 9207 và cấm tái sử dụng thông tin xác thực giữa các issuer.

Các quy tắc này bổ sung cho lõi không trạng thái. Chúng không khôi phục lại cơ chế bắt tay lõi hoặc `Mcp-Session-Id`.

## Khái niệm

### Biết ba vai trò

- **MCP client:** gửi yêu cầu thay mặt cho chủ sở hữu tài nguyên.
- **MCP resource server:** chấp nhận access token và phục vụ endpoint MCP.
- **Authorization server:** xác thực chủ sở hữu tài nguyên, thu thập sự đồng ý và cấp token.

Resource server và authorization server có thể được vận hành cùng nhau, nhưng hãy giữ các định danh và trách nhiệm xác thực của chúng tách biệt.

### Ủy quyền áp dụng cho HTTP

Đặc tả ủy quyền MCP áp dụng cho các transport dựa trên HTTP. Một server stdio cục bộ chạy dưới ranh giới tin cậy của tiến trình và hệ điều hành. Đừng thêm luồng OAuth trình duyệt giả vào stdio chỉ để cho cân xứng.

Đối với Streamable HTTP từ xa, hãy gửi bearer token trong header `Authorization` trên mọi yêu cầu. Không bao giờ đặt nó trong URL.

### Bắt đầu với metadata của tài nguyên được bảo vệ

Resource server xuất bản metadata theo RFC 9728:

```json
{
  "resource": "https://notes.example.com/mcp",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:delete", "notes:read", "notes:write"]
}
```

Client bắt đầu từ URL tài nguyên MCP, tìm nạp tài liệu này, chọn một authorization server được quảng cáo, sau đó tìm nạp metadata OAuth hoặc OpenID Connect của server đó.

Giữ nguyên đường dẫn tài nguyên khi xây dựng URL well-known theo RFC 9728. Đối với tài nguyên `https://notes.example.com/mcp`, bài học này sử dụng `https://notes.example.com/.well-known/oauth-protected-resource/mcp`. Việc bỏ hậu tố `/mcp` có thể chọn nhầm metadata cho một tài nguyên được bảo vệ khác trên cùng origin.

Đừng đoán authorization server từ hostname. Đừng theo một issuer được phát hiện từ nội dung lỗi chưa được xác thực. Hãy duy trì chính sách về những issuer mà client sẵn sàng tin tưởng.

### Xác thực metadata của authorization server

Metadata nên hiển thị các endpoint và các kiểm soát được hỗ trợ:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "code_challenge_methods_supported": ["S256"],
  "authorization_response_iss_parameter_supported": true,
  "client_id_metadata_document_supported": true
}
```

Yêu cầu S256 cho PKCE. Ghi lại chuỗi issuer chính xác. Giá trị chính xác đó trở thành khóa để đăng ký và lưu trữ token.

### Tuân theo thứ tự ưu tiên đăng ký

Sử dụng thông tin client đã đăng ký trước khi client đã có mối quan hệ rõ ràng với issuer được chọn. Nếu không, hãy ưu tiên Client ID Metadata Documents khi authorization server quảng cáo hỗ trợ. Chỉ sử dụng DCR như một phương án dự phòng tương thích đã lỗi thời, sau đó nhắc nhập thông tin client nếu không có cơ chế nào trong số đó khả dụng.

### Ưu tiên Client ID Metadata Documents

Client ID Metadata Document cung cấp cho authorization server một URL HTTPS vừa là định danh client vừa là vị trí metadata của nó:

```json
{
  "client_id": "https://client.example.com/oauth/metadata.json",
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

Authorization server tìm nạp và xác thực tài liệu. `client_id` phải là một URL HTTPS có đường dẫn, và giá trị bên trong tài liệu phải bằng chính xác URL đó. Các trường tài liệu bắt buộc là `client_id`, `client_name` và `redirect_uris`. `application_type` xuất hiện trong ví dụ này nhưng không phải là yêu cầu CIMD. Việc sử dụng bắt buộc mới của nó dành riêng cho đường dẫn DCR.

Hãy coi việc tìm nạp tài liệu là một thao tác nhạy cảm với SSRF. Phân giải và xác thực đích đến, từ chối loopback, private, link-local và các địa chỉ không được phép khác, kiểm tra lại sau khi chuyển hướng và thay đổi DNS, giới hạn chuyển hướng, byte và thời gian, yêu cầu JSON và chỉ lưu cache theo các kiểm soát cache HTTP đã xác thực. Coi `client_name` và các trường hiển thị khác là văn bản không đáng tin cậy.

CIMD loại bỏ nhu cầu tạo định danh động mới cho mỗi lần liên hệ đầu tiên. Nó không loại bỏ việc xác thực redirect URI, chính sách issuer hoặc sự đồng ý của người dùng.

### DCR là đường dẫn tương thích

Dynamic Client Registration vẫn khả dụng cho các authorization server cũ hơn, nhưng nó đã lỗi thời đối với các triển khai MCP mới.

Khi sử dụng DCR, hãy khai báo `application_type`:

```json
{
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

- Các client desktop, mobile, dòng lệnh và loopback sử dụng `native`.
- Các ứng dụng trình duyệt được lưu trữ từ xa sử dụng `web` và các chuyển hướng HTTPS từ xa.

Việc bỏ qua trường này có thể mặc định thành `web` trong triển khai đăng ký OpenID Connect và làm cho redirect loopback hợp lệ bị lỗi.

Giữ mã DCR đằng sau một quyết định dự phòng rõ ràng. Đừng tự động dự phòng sau một lỗi xác thực CIMD tùy ý. Điều đó có thể biến một lỗi bảo mật thành một đường dẫn đăng ký yếu hơn.

### Gắn thông tin xác thực với issuer

Lưu trữ tài liệu đăng ký do issuer cấp dưới chính xác issuer đó:

```text
issuer_credentials[issuer] = pre_registered_or_dcr_client
tokens[(issuer, resource)] = access_token
```

Nếu việc khám phá tài nguyên được bảo vệ thay đổi từ `https://auth-one.example` sang `https://auth-two.example`, hãy đánh giá lại sự tin tưởng. Không bao giờ gửi client secret, DCR client id, registration access token, refresh token hoặc access token của issuer đầu tiên cho issuer thứ hai. Các client đã đăng ký trước và DCR phải sử dụng thông tin xác thực được cấp cho issuer mới.

Một CIMD client id khác biệt vì nó là một URL HTTPS tự lưu trữ, không phải thông tin xác thực do authorization server cấp. Cùng một URL CIMD có tính di động: một issuer tin cậy mới sẽ tìm nạp và xác thực tài liệu mà không cần đăng ký lại DCR. Các phản hồi ủy quyền và token vẫn được xác thực và lưu trữ dưới issuer mới.

### Authorization code với PKCE

Luồng tương tác là:

1. Tạo một `code_verifier` có độ entropy cao.
2. Dẫn xuất `code_challenge` S256.
3. Gửi yêu cầu ủy quyền với `client_id`, `redirect_uri`, `scope`, `code_challenge` và `resource` chính xác.
4. Nhận phản hồi ủy quyền chứa `code` và, khi được cung cấp, `iss`.
5. Xác thực `iss` so với issuer đã ghi lại chính xác trước khi sử dụng bất kỳ trường phản hồi nào.
6. Trao đổi code với `code_verifier`, cùng redirect URI và cùng `resource`.
7. Lưu trữ token kết quả dưới `(issuer, resource)`.

Tham số `resource` từ RFC 8707 xuất hiện trong cả yêu cầu ủy quyền và yêu cầu token. Nó xác định URI server MCP chính tắc.

### Xác thực `iss` chính xác

RFC 9207 ngăn chặn việc nhầm lẫn phản hồi ủy quyền từ issuer này với phản hồi từ issuer khác.

Khi `iss` hiện diện, hãy so sánh nó với issuer đã ghi lại mà không thực hiện case folding, thay đổi dấu gạch chéo cuối, loại bỏ cổng mặc định hoặc chuẩn hóa percent-encoding. Nếu không khớp, đừng hành động dựa trên code hoặc thậm chí hiển thị chi tiết lỗi do kẻ tấn công kiểm soát từ phản hồi đó.

Một authorization server bao gồm `iss` sẽ quảng cáo `authorization_response_iss_parameter_supported: true`. Các client hiện tại vẫn xác thực `iss` hiện diện ngay cả khi quảng cáo đó bị thiếu.

### Xác thực audience tại MCP server

Resource server chỉ chấp nhận các token được cấp cho chính nó:

```text
token.issuer == configured_authorization_server
token.audience == canonical_mcp_resource
```

Các token không hợp lệ, hết hạn, sai issuer hoặc sai audience sẽ nhận mã 401. MCP server không được chấp nhận hoặc chuyển tiếp token dành cho dịch vụ khác.

### Yêu cầu scope hiện tại nhỏ nhất

Bắt đầu với scope cần thiết ngay bây giờ. Nếu một công cụ sau đó yêu cầu nhiều hơn, server sẽ trả về 403 với một thách thức scope có thẩm quyền:

```text
WWW-Authenticate: Bearer error="insufficient_scope",
  scope="notes:delete",
  resource_metadata="https://notes.example.com/.well-known/oauth-protected-resource/mcp"
```

Client giải thích quyền mới, nhận sự đồng ý, thực hiện luồng ủy quyền mới với tập hợp scope kết hợp và thử lại yêu cầu MCP với một JSON-RPC id mới.

Đừng giả định scope được thách thức là một tập con của `scopes_supported`. Thách thức đó là có thẩm quyền cho thao tác hiện tại.

### Ủy quyền và luồng MCP không trạng thái

Một lệnh gọi công cụ đã ủy quyền vẫn mang theo toàn bộ envelope yêu cầu hiện tại:

```text
POST /mcp
Authorization: Bearer <access-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.delete
```

```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "notes.delete",
    "arguments": {"id": "note-7"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "oauth-lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Token ủy quyền cho principal. Metadata yêu cầu thương lượng hành vi giao thức. Không cái nào thay thế cái nào.

Xác thực luồng theo thứ tự cố định: các loại JSON-RPC và metadata, sự bình đẳng giữa header và body, sau đó là hỗ trợ giao thức. Một sự không khớp về routing hoặc version-header sẽ trả về HTTP 400 với `-32020`. Nếu header và body đồng ý về một phiên bản không được hỗ trợ, trả về HTTP 400 với `-32022` và `data` chính xác là `{"supported":["2026-07-28"],"requested":"<actual>"}`. Một phương thức không xác định trả về HTTP 404 với `-32601`.

Mọi lỗi yêu cầu, bao gồm 401 token không hợp lệ và 403 scope không đủ, là một JSON-RPC error envelope với `id` của yêu cầu gốc. Thông tin khôi phục có cấu trúc thuộc về `data` lỗi tùy chọn; `WWW-Authenticate` vẫn là một HTTP response header. Một thông báo không có `id`, vì vậy nó không nhận được body JSON-RPC. Một thông báo HTTP được chấp nhận trả về 202 với body trống.

Server triển khai `server/discover` và quảng cáo các công cụ, vì vậy nó cũng triển khai phương thức `tools/list` bắt buộc. Các mô tả công cụ của nó có tên, mô tả và giá trị `inputSchema` gốc đối tượng ổn định. Danh sách này là tất định và trả về `resultType`, metadata định danh server, `ttlMs` bị giới hạn và `cacheScope`. Việc khám phá và danh sách công cụ độc lập với người dùng có thể khả dụng trước khi ủy quyền. Áp dụng chính sách bình thường và bộ nhớ cache riêng tư nếu một trong hai thay đổi theo principal.

### Không chuyển tiếp token

MCP server không được chuyển tiếp MCP access token của client đến một API hạ nguồn. Hãy lấy một token hạ nguồn riêng biệt với audience phù hợp hoặc sử dụng thiết kế trao đổi token rõ ràng. Xác thực audience chỉ hoạt động khi các dịch vụ từ chối các token được cấp cho người khác.

### Refresh token

Refresh token là tùy chọn. Khi được cấp, hãy lưu trữ chúng một cách bảo mật và khóa chúng theo issuer và tài nguyên. Đừng giả định chúng tồn tại. Hãy xoay vòng chúng khi authorization server hỗ trợ xoay vòng và phát hiện việc tái sử dụng các giá trị đã bị vô hiệu hóa.

```figure
t3-scope-stepup
```

## Xây dựng

`code/main.py` là một trình mô phỏng giao thức và ủy quyền trong tiến trình. Nó triển khai khám phá tài nguyên được bảo vệ, metadata authorization server, đăng ký CIMD, dự phòng DCR có kiểm soát phiên bản, kiểm tra loại ứng dụng, PKCE, xác thực issuer, token gắn với tài nguyên, scope step-up, `server/discover`, `tools/list` và một yêu cầu công cụ không trạng thái.

Mô hình nhận các body yêu cầu đã phân tích cú pháp và các header định tuyến. Nó không phải là một bộ chuyển đổi HTTP hoàn chỉnh và không phân tích `Content-Type` hoặc `Accept`. Kết nối nó với bộ chuyển đổi Streamable HTTP của Bài 09, yêu cầu `Content-Type: application/json` và giá trị `Accept` chứa cả `application/json` và `text/event-stream`.

Chạy nó:

```bash
cd phases/13-tools-and-protocols/16-mcp-security-oauth-2-1
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Đầu ra hiển thị quá trình khám phá trước, đăng ký CIMD, một lần đọc thông thường, hai lần scope step-up riêng biệt và lưu trữ thông tin xác thực theo issuer.

## Sử dụng

Ánh xạ các đối tượng mô phỏng vào các thành phần sản xuất:

- `ResourceServer.protected_resource_metadata` trở thành endpoint RFC 9728.
- `AuthorizationServer.metadata` trở thành khám phá RFC 8414 hoặc OpenID Connect.
- `Client.enroll` trở thành phân giải CIMD cộng với một nhánh tương thích DCR rõ ràng.
- Thông tin xác thực client do issuer cấp và `tokens_by_issuer_resource` trở thành các bản ghi được mã hóa. Một URL CIMD có thể vẫn di động trong khi kết quả ủy quyền của nó vẫn bị ràng buộc với issuer.
- `ResourceServer.handle` trở thành middleware xác thực các header MCP hiện tại, token và scope công cụ trước khi điều phối trong khi giữ mọi lỗi yêu cầu trong một JSON-RPC envelope khớp.

## Phát hành

Bài học này phát hành `outputs/skill-oauth-scope-planner.md`. Nó hiện thiết kế thứ tự ưu tiên đăng ký, lưu trữ thông tin xác thực bị ràng buộc với issuer, loại ứng dụng, PKCE, resource indicators, các thách thức scope và ranh giới yêu cầu không trạng thái hiện tại.

## Bài tập

1. Thêm xoay vòng refresh-token và từ chối việc tái sử dụng refresh token trước đó.
2. Thêm một danh sách cho phép issuer (allowlist). Khi thay đổi issuer, chỉ tái sử dụng URL CIMD di động; từ chối tất cả thông tin xác thực và token do issuer trước đó cấp.
3. Thêm thời hạn cho authorization code và xác nhận việc trao đổi muộn sẽ thất bại.
4. Xây dựng một biến thể web client với chuyển hướng HTTPS từ xa và so sánh metadata DCR của nó với client gốc.
5. Thêm tài nguyên thứ hai dưới cùng issuer. Xác nhận access token của nó không thể được sử dụng tại tài nguyên đầu tiên.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa |
|------|---------|
| Protected-resource metadata | Tài liệu RFC 9728 xác định tài nguyên và các authorization server |
| CIMD | Tài liệu metadata HTTPS có URL là định danh OAuth client |
| DCR | Đăng ký client động đã lỗi thời được giữ lại để tương thích |
| `application_type` | `native` hoặc `web`, được sử dụng để xác thực các quy tắc redirect URI |
| PKCE | Verifier và S256 challenge bảo vệ authorization code bị chặn |
| `iss` | Định danh issuer phản hồi ủy quyền RFC 9207 |
| Resource indicator | Tham số RFC 8707 gắn yêu cầu token với một tài nguyên MCP |
| Audience | Tài nguyên mà token có hiệu lực |
| Step-up | Sự đồng ý mới và cấp token cho một scope thao tác hiện tại bổ sung |
| Issuer-bound credentials | Các bản ghi đăng ký và token được cô lập bởi issuer authorization server chính xác |

## Đọc thêm

- [Đặc tả ủy quyền MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [RFC 9728: OAuth 2.0 Protected Resource Metadata](https://www.rfc-editor.org/rfc/rfc9728)
- [RFC 8707: Resource Indicators for OAuth 2.0](https://www.rfc-editor.org/rfc/rfc8707)
- [RFC 9207: OAuth 2.0 Authorization Server Issuer Identification](https://www.rfc-editor.org/rfc/rfc9207)
- [Bản thảo OAuth Client ID Metadata Document](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/)