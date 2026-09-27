# Xác thực MCP trong môi trường Production: Đăng ký gắn kết với Issuer và Token

> Bài học 16 đã xây dựng máy trạng thái OAuth 2.1. Bài học này củng cố các ranh giới production cho MCP 2026-07-28: ưu tiên Tài liệu Metadata Client ID, Dynamic Registration chỉ còn là cơ chế tương thích lỗi thời, xác thực issuer trong phản hồi ủy quyền, thông tin xác thực client được gắn với issuer, làm mới JWKS, và token được ghim theo audience trên mọi yêu cầu không trạng thái (stateless).
>
> **Ghi chú đặc tả (2026-07-28):** Dynamic Client Registration (DCR) đã bị loại bỏ để thay thế bằng Tài liệu Metadata Client ID (CIMD). DCR vẫn tồn tại như một cơ chế tương thích. Khi sử dụng, client phải khai báo đúng `application_type`. Client phải xác thực giá trị `iss` theo RFC 9207 hiện có và không bao giờ được tái sử dụng thông tin xác thực giữa các issuer của authorization-server.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 16 (Máy trạng thái OAuth 2.1), Phase 13 · 17 (Gateways)
**Time:** ~90 phút

## Mục tiêu học tập

- Khám phá authorization server thông qua metadata RFC 8414 và xác minh hợp đồng.
- Đăng ký thông qua Tài liệu Metadata Client ID và cô lập DCR lỗi thời như một phương án dự phòng.
- Xác thực `iss` theo RFC 9207, đăng ký khóa theo issuer của authorization-server, và ghim token vào tài nguyên theo cặp issuer và tài nguyên.
- Cache và làm mới các khóa JWKS theo lịch trình để việc xác minh chữ ký không bị gián đoạn khi xoay vòng khóa.
- Ghim token vào một tài nguyên MCP duy nhất bằng cách sử dụng chỉ báo tài nguyên RFC 8707 và từ chối việc tái sử dụng confused-deputy.
- Lựa chọn giữa xác thực JWT hoặc token introspection, xác định độ tươi mới của việc thu hồi, và xử lý an toàn khi các phụ thuộc danh tính không khả dụng.
- Tách biệt authorization server, resource server và client để mỗi bên chỉ thực thi các kiểm tra của riêng mình.
- Kiểm tra authorization server dựa trên danh sách kiểm tra triển khai và từ chối việc đăng ký hoặc tái sử dụng token không an toàn.

## Vấn đề

Trình mô phỏng của Bài học 16 chạy OAuth 2.1 trong bộ nhớ. Môi trường production có ba lỗ hổng vận hành mà trình mô phỏng chỉ chạy trong bộ nhớ không thấy được.

Lỗ hổng đầu tiên là việc đăng ký và cô lập thông tin xác thực. Một tổ chức thực tế có thể chạy hàng trăm server MCP và hàng ngàn client MCP. Bản sửa đổi 2026-07-28 ưu tiên **Tài liệu Metadata Client ID**: client sử dụng một URL HTTPS với đường dẫn mà nó kiểm soát làm định danh, và authorization server sẽ lấy metadata đó. Dynamic registration theo RFC 7591 chỉ còn là đường dẫn tương thích lỗi thời. Khi bắt buộc phải dùng DCR, yêu cầu phải khai báo đúng `application_type`. Client lưu trữ các đăng ký theo issuer của authorization-server và các access token theo cặp `(issuer, resource)`. Thay đổi issuer đồng nghĩa với việc đăng ký mới, và tài nguyên khác đồng nghĩa với một token được ghim audience riêng biệt.

Lỗ hổng thứ hai là xoay vòng khóa. Việc xác thực JWT phụ thuộc vào các khóa ký của authorization server, được công bố dưới dạng JSON Web Key Set (JWKS). Authorization server xoay vòng các khóa này theo lịch trình (thường là hàng giờ, đôi khi nhanh hơn trong trường hợp ứng phó sự cố). Một server MCP lấy JWKS một lần khi khởi động sẽ xác thực tốt cho đến khi cửa sổ xoay vòng xảy ra — sau đó mọi yêu cầu sẽ thất bại cho đến khi khởi động lại. Production cần thiết lập JWKS như một giá trị được cache với một tác vụ làm mới để ghi đè cache trước khi các khóa cũ hết hạn, cộng với việc lấy lại khóa khi cache miss trong trường hợp token được ký bởi một khóa mới hơn cache xuất hiện.

Lỗ hổng thứ ba là ghim audience. Bài học 16 đã giới thiệu các chỉ báo tài nguyên RFC 8707. Trong production, chỉ báo đó trở thành một yêu cầu kiểm tra claim cứng trên mọi yêu cầu. Server MCP so sánh `token.aud` với URL tài nguyên chuẩn của chính nó và từ chối các trường hợp không khớp bằng mã HTTP 401. Đây là biện pháp phòng thủ duy nhất chống lại việc một server MCP thượng nguồn (hoặc một client độc hại giữ token dành cho một server khác) phát lại token đó đối với một server khác trong cùng một lưới tin cậy.

Bài học này ánh xạ từng lỗ hổng vào một phần cụ thể của bề mặt tấn công. Tài liệu metadata là một endpoint HTTP. Làm mới cache JWKS là một tác vụ theo lịch trình cộng với cache key-value. Xác thực JWT là một quy trình mà resource server chạy trước khi điều phối bất kỳ công cụ nào. Hãy giữ ba vai trò tách biệt và mỗi vai trò chỉ thực thi các kiểm tra mà nó sở hữu: authorization server phát hành và xoay vòng khóa, resource server cache và xác thực, client khám phá và đăng ký.

## Phạm vi: Thực thi Production sau Bài học 16

[Bài học 16: Bảo mật MCP với OAuth 2.1](../../16-mcp-security-oauth-2-1/docs/en.md) sở hữu máy trạng thái authorization-code, PKCE, khám phá tài nguyên được bảo vệ, chỉ báo tài nguyên và các quyết định về scope. Bài học này không định nghĩa luồng OAuth thứ hai. Nó bắt đầu sau khi các hợp đồng đó tồn tại và đặt câu hỏi làm thế nào một resource server đã triển khai có thể tiếp tục thực thi chúng trong quá trình xoay vòng khóa, xác thực token mờ (opaque-token), thu hồi, lỗi phụ thuộc, triển khai và ứng phó sự cố.

Ranh giới production hẹp hơn và mang tính vận hành hơn:

- Đường dẫn JWT xác minh issuer được ghim, thuật toán, khóa chữ ký, audience, các claim thời gian và scope trên mọi yêu cầu trong khi làm mới JWKS một cách an toàn.
- Đường dẫn opaque-token gọi endpoint introspection đã xác thực của issuer và xác thực trạng thái active được trả về, audience hoặc tài nguyên, thời hạn, subject và scope.
- Chính sách thu hồi xác định mức độ nhanh chóng mà một thông tin xác thực phải ngừng hoạt động và cache nào có thể trì hoãn thực tế đó.
- Chính sách lỗi quyết định điều gì xảy ra khi cơ sở hạ tầng khám phá, JWKS, introspection hoặc thu hồi không khả dụng.
- Nhật ký bằng chứng ghi lại metadata issuer, bộ khóa hoặc phản hồi introspection, các claim của token, phiên bản chính sách và lý do từ chối đã dẫn đến kết quả mà không lưu trữ token.

Sự phân biệt này giúp các bài học có thể kết hợp được. Bài học 16 chứng minh luồng. Bài học 18 chứng minh rằng một token vẫn đáng tin cậy, hoặc bị từ chối, sau khi nó đến được đường dẫn yêu cầu MCP thực tế.

## Khái niệm

### RFC 8414 — Metadata của Authorization Server OAuth

Một tài liệu tại `/.well-known/oauth-authorization-server` mô tả mọi thứ mà một client cần:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "client_id_metadata_document_supported": true,
  "registration_endpoint": "https://auth.example.com/register",
  "authorization_response_iss_parameter_supported": true,
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

Một client được cung cấp URL tài nguyên MCP sẽ thực hiện chuỗi khám phá: `oauth-protected-resource` từ RFC 9728 (tài liệu của resource server) chỉ định issuer, sau đó `oauth-authorization-server` (RFC này) chỉ định mọi endpoint. Client không bao giờ được hard-code URL ủy quyền.

Đối với một định danh tài nguyên có đường dẫn, hãy chèn phân đoạn well-known trước đường dẫn đó. Ví dụ: `https://mcp.example.com/team/server` phân giải metadata tài nguyên được bảo vệ tại `https://mcp.example.com/.well-known/oauth-protected-resource/team/server`. Việc thêm `/.well-known/...` sau đường dẫn tài nguyên là không chính xác.

Hợp đồng bạn xác minh trước khi tin tưởng một IdP cho MCP:

- `code_challenge_methods_supported` bao gồm `S256` (PKCE theo RFC 7636). Đặc tả rất rõ ràng: nếu trường này **vắng mặt**, authorization server không hỗ trợ PKCE và client **PHẢI** từ chối tiếp tục.
- `grant_types_supported` bao gồm `authorization_code` và từ chối `password` và `implicit`.
- Ít nhất một đường dẫn đăng ký khả dụng: `client_id_metadata_document_supported: true` (CIMD, được ưu tiên), một client đã đăng ký trước, hoặc `registration_endpoint` (tương thích RFC 7591 lỗi thời).
- Nếu `authorization_response_iss_parameter_supported` là true, client yêu cầu giá trị `iss` theo RFC 9207 được trả về và so sánh chính xác với issuer đã ghi lại trước khi chuyển hướng.
- `response_types_supported` chính xác là `["code"]` cho OAuth 2.1.

Nếu `S256` bị thiếu, server MCP từ chối triển khai với IdP này — không có chế độ suy giảm cho PKCE. Nếu *không* có đường dẫn đăng ký nào được quảng bá và bạn không có `client_id` đã đăng ký trước, bạn cũng không thể đăng ký; manifest triển khai bị sai, không phải mã nguồn.

### RFC 9728 (tóm tắt) — Metadata tài nguyên được bảo vệ

Bài học 16 đã đề cập đến RFC 9728. Điểm khác biệt trong production: tài liệu này là nơi duy nhất client tìm kiếm các authorization server được tin tưởng bởi *server MCP này*. Một server MCP có thể chấp nhận token từ nhiều IdP (một cho nhân viên, một cho đối tác). RFC 9728 khai báo tập hợp đó; RFC 8414 ghi lại những gì mỗi IdP hỗ trợ.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Tài liệu Metadata Client ID (mặc định được khuyến nghị)

CIMD đảo ngược việc đăng ký từ *đẩy* sang *kéo*. Thay vì yêu cầu authorization server đúc một `client_id`, client sử dụng một URL HTTPS mà nó kiểm soát **như là** `client_id` của nó. URL phân giải thành một tài liệu metadata JSON; authorization server sẽ lấy nó theo yêu cầu trong luồng OAuth. Sự tin tưởng bắt nguồn từ DNS: nếu nhà vận hành server tin tưởng `app.example.com`, họ tin tưởng client được phục vụ từ `https://app.example.com/client.json`. Không có vòng lặp đăng ký, không có không gian tên `client_id` để cạn kiệt, không có trạng thái trên mỗi server cần đồng bộ.

Tài liệu metadata mà client lưu trữ:

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

Giá trị `client_id` trong tài liệu **PHẢI** bằng với URL mà nó được phục vụ (authorization server xác minh điều này; các trường hợp không khớp sẽ bị từ chối). Authorization server quảng bá hỗ trợ với `client_id_metadata_document_supported: true` trong metadata RFC 8414 của nó.

Đối với hợp đồng CIMD hiện tại, `client_id`, `client_name`, và một mảng `redirect_uris` không trống là bắt buộc. Định danh client là một URL HTTPS tuyệt đối có đường dẫn. `application_type` có thể được bao gồm, nhưng không phải là trường CIMD bắt buộc. Đừng sao chép yêu cầu DCR cho `application_type` vào đường dẫn CIMD được ưu tiên.

Hai sự thật bảo mật mà đặc tả nêu rõ:

- **SSRF.** Authorization server lấy một URL do kẻ tấn công cung cấp. Nó phải phòng thủ chống lại giả mạo yêu cầu phía server (không lấy dữ liệu từ các endpoint nội bộ/quản trị).
- **Giả mạo localhost.** Chỉ riêng CIMD không thể ngăn kẻ tấn công cục bộ yêu cầu URL metadata của một client hợp pháp và gắn kết bất kỳ chuyển hướng `localhost` nào. Authorization server **PHẢI** hiển thị rõ ràng hostname của URI chuyển hướng trong quá trình đồng ý và **NÊN** cảnh báo về các chuyển hướng chỉ có `localhost`.

Vì CIMD không cần trạng thái phía server, không có registrar nào cần thiết lập như cách DCR yêu cầu. Phía client là chỉ đọc: phục vụ tài liệu metadata của bạn từ một endpoint HTTPS tĩnh và để authorization server tự lấy.

Nếu nhà vận hành authorization server đã cung cấp định danh client, hãy sử dụng đăng ký theo phạm vi issuer đó trước khi thử đăng ký tự động. Nếu không, hãy ưu tiên CIMD. Chỉ sử dụng DCR lỗi thời khi issuer không thể sử dụng đăng ký trước hoặc CIMD.

### RFC 7591: đăng ký tương thích lỗi thời

DCR đã bị loại bỏ trong bản sửa đổi 2026-07-28. Chỉ giữ nó cho các authorization server không thể tiêu thụ CIMD và nơi việc đăng ký trước là không thực tế. Một client tương thích gửi:

```json
POST /register
Content-Type: application/json

{
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

Server phản hồi với `client_id` và một `registration_access_token` cho các cập nhật sau này:

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`application_type` không phải là trang trí. Một client desktop loopback khai báo `native`; một client được host trên server khai báo `web` và sử dụng các URI chuyển hướng HTTPS. `token_endpoint_auth_method: none` là mặc định đúng cho một client native công cộng. Nó chỉ nhận được `client_id`, với PKCE cung cấp bằng chứng sở hữu.

Ba cạm bẫy trong production:

- Endpoint đăng ký phải giới hạn tốc độ theo IP nguồn. Nếu không, một tác nhân thù địch sẽ viết kịch bản hàng triệu đăng ký giả và làm cạn kiệt không gian tên `client_id`. Chạy kiểm tra giới hạn tốc độ trước khi registrar xử lý yêu cầu.
- `software_statement` (một JWT đã ký bảo lãnh cho client) được yêu cầu bởi một số IdP doanh nghiệp. Bản mô phỏng của bài học bỏ qua nó; production cần thiết lập một bước xác minh từ chối các đăng ký không chữ ký từ bất kỳ thứ gì khác ngoài URI chuyển hướng localhost.
- `registration_access_token` phải được lưu trữ dưới dạng băm, không phải văn bản thuần. Việc đánh cắp token này có nghĩa là kẻ tấn công có thể viết lại các URI chuyển hướng của client.

### RFC 8707 (tóm tắt) — Chỉ báo tài nguyên

Bài học 16 đã thiết lập hình dạng. Quy tắc production: mọi yêu cầu token bao gồm `resource=<canonical-mcp-url>`, và server MCP xác minh `token.aud` khớp với URL tài nguyên của chính nó trên mọi cuộc gọi. URI chuẩn là định danh *cụ thể nhất* cho server: nó sử dụng scheme và host viết thường, không có fragment, và theo quy ước không có dấu gạch chéo ở cuối. Thành phần đường dẫn **không** bị lược bỏ theo quy tắc — đặc tả giữ lại nó khi cần thiết để định danh một server MCP cá nhân. `https://mcp.example.com`, `https://mcp.example.com/mcp`, `https://mcp.example.com:8443`, và `https://mcp.example.com/server/mcp` đều là các URI chuẩn hợp lệ. Chọn một URI cho mỗi server và ghim `aud` chính xác vào đó. (Bản mô phỏng của bài học này sử dụng các audience bare-host như `https://notes.example.com` cho ngắn gọn; một triển khai host nhiều server MCP dưới một origin sẽ phân biệt chúng bằng đường dẫn.)

### RFC 7636 (tóm tắt) — PKCE

PKCE là bắt buộc trong OAuth 2.1. Luồng authorization-code của bài học luôn mang theo `code_challenge` và `code_verifier`. Server từ chối bất kỳ yêu cầu token nào không có verifier hoặc có verifier không băm ra challenge đã lưu trữ.

### Hồ sơ ủy quyền MCP 2026-07-28

Bản sửa đổi MCP hiện tại giữ ranh giới resource-server OAuth trong khi làm cho vận chuyển MCP trở nên stateless. Không có phiên giao thức nào để cache quyết định danh tính. Do đó, lớp ủy quyền xác thực từng yêu cầu một cách độc lập:

- Triển khai metadata tài nguyên được bảo vệ RFC 9728, và cung cấp vị trí của nó thông qua header `WWW-Authenticate: Bearer resource_metadata="..."` trên phản hồi 401 **hoặc** URI well-known `/.well-known/oauth-protected-resource` (SEP-985 làm cho header trở nên tùy chọn với phương án dự phòng well-known). Trường `authorization_servers` trong metadata **PHẢI** đặt tên ít nhất một server.
- Chỉ chấp nhận token thông qua `Authorization: Bearer ...` trên **mọi** yêu cầu — không bao giờ trong chuỗi truy vấn, không bao giờ chỉ xác thực khi bắt đầu phiên.
- Xác thực `aud`, `iss`, `exp`, và các scope bắt buộc trên mỗi yêu cầu. Server **PHẢI** xác thực rằng token được phát hành cụ thể cho nó (audience); một `aud` bị thiếu hoặc không khớp sẽ bị từ chối, không bao giờ được coi là wildcard.
- Khi có 401/403, trả về `WWW-Authenticate: Bearer` mang theo `error=...`, tham số `resource_metadata="<PRM-URL>"` (URL của tài liệu metadata, *không phải* tài nguyên bare), và `scope="..."` trên `insufficient_scope` (403). Lưu ý: tham số là `resource_metadata`, một con trỏ khám phá — không có tham số `resource` trong challenge.
- Khám phá authorization-server chấp nhận **hoặc** metadata OAuth RFC 8414 **hoặc** OpenID Connect Discovery 1.0; client phải thử cả hai hậu tố well-known theo thứ tự ưu tiên.
- Client (không phải server) phòng thủ chống lại **tấn công mix-up**: nó ghi lại `issuer` dự kiến trước khi chuyển hướng và xác thực giá trị `iss` được trả về trong phản hồi ủy quyền thực tế (RFC 9207) trước khi đổi mã. Chỉ riêng PKCE không ngăn được mix-up, vì client trao `code_verifier` của nó cho bất kỳ token endpoint nào mà nó được dẫn tới.
- Một thông tin xác thực client thuộc về một issuer authorization-server. Nếu khám phá phân giải thành một issuer khác, client sẽ đăng ký lại thay vì trình bày `client_id`, token đăng ký hoặc access token cũ.
- CIMD là cơ chế đăng ký được ưu tiên. DCR đã lỗi thời; một yêu cầu DCR tương thích vẫn khai báo đúng `application_type`.

Bản dự thảo OAuth 2.1 là nền tảng; RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD là bề mặt; đặc tả MCP là hồ sơ.

### Danh sách kiểm tra khả năng triển khai

Các bảng tính năng của nhà cung cấp nhanh chóng trở nên lỗi thời. Thay vào đó, hãy kiểm tra metadata được trả về bởi authorization server mà bạn thực sự sẽ triển khai. Cổng kiểm soát mang tính cơ học:

| Kiểm tra | Quyết định bắt buộc |
|---|---|
| Issuer đã khám phá | Issuer HTTPS chính xác theo chính sách |
| PKCE | `S256` được quảng bá; nếu không, dừng lại |
| Đăng ký | CIMD được ưu tiên, chấp nhận đăng ký trước, DCR chỉ là tương thích lỗi thời |
| Phản hồi ủy quyền | Xác thực `iss` theo RFC 9207 khi có mặt hoặc được quảng bá |
| Ghim tài nguyên | Yêu cầu token mang theo `resource`; resource server yêu cầu `aud` khớp |
| Lưu trữ thông tin xác thực | Khóa ID client và thông tin xác thực đăng ký theo issuer; khóa access token theo issuer cộng tài nguyên |
| Tương thích DCR | Khai báo `native` hoặc `web`; từ chối các URI chuyển hướng không phù hợp với loại ứng dụng đã khai báo |

Đừng suy luận sự hỗ trợ từ tên sản phẩm hoặc gói giá. Hãy ghi lại tài liệu đã khám phá trong bằng chứng triển khai và thất bại đóng (fail closed) khi một trường bắt buộc bị thiếu.

### Mô hình làm mới JWKS (xoay vòng tại AS, làm mới tại resource server)

Hãy giữ hai động từ tách biệt, vì việc nhầm lẫn chúng là một lỗi production thực sự:

- **Xoay vòng (Rotate)** là những gì *authorization server* làm: đúc một khóa ký mới, công bố nó trong JWKS, sau đó thu hồi khóa cũ. Resource server không tham gia vào việc này và không thể thực hiện — nó không giữ các khóa riêng tư của IdP.
- **Làm mới (Refresh)** là những gì *resource server* làm: re-`GET` JWKS đã công bố vào cache của nó. Đó là hành động JWKS duy nhất mà resource server thực hiện.

Chế độ thất bại trong production là cache cũ. Giải quyết nó bằng một tác vụ làm mới theo lịch trình cộng với cache key-value. Resource server chạy một tác vụ (cron, timer, bất cứ thứ gì runtime của bạn cung cấp) mà, theo một khoảng thời gian cố định, lấy `<issuer>/.well-known/jwks.json` và ghi đè `cache[issuer] = {keys, fetched_at}`. Trình xác thực đọc từ cache đó. Một token có `kid` bị thiếu trong cache sẽ kích hoạt **một** lần làm mới đồng bộ như một phương án dự phòng, sau đó kiểm tra lại. Điều này xử lý hai trường hợp cùng lúc: làm mới theo lịch trình, và các cửa sổ chồng lấp khóa nơi một token được ký bởi một khóa hoàn toàn mới xuất hiện trước lần làm mới theo lịch trình tiếp theo.

Phương án dự phòng **phải là lấy lại (re-fetch), không bao giờ là xoay vòng**. Nếu bạn nối đường dẫn cache-miss với một tác vụ xoay vòng và đúc khóa, hai điều sẽ hỏng: (1) đúc một khóa mới tạo ra `kid` vẫn không khớp với token, vì vậy việc tra cứu vẫn thất bại; và (2) một kẻ tấn công phun các token với các giá trị `kid` ngẫu nhiên sẽ buộc một loạt các lần tạo khóa không giới hạn — một cuộc tấn công DoS tự gây ra. Việc lấy lại là idempotent, vì vậy một `kid` giả mạo tốn nhiều nhất một lần lấy dữ liệu lãng phí.

Hình dạng cache:

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

Hai khóa cùng một lúc là trạng thái ổn định. Authorization servers xoay vòng bằng cách giới thiệu khóa tiếp theo (`k_2026_04`) trước khi thu hồi khóa trước đó (`k_2026_03`), vì vậy các token được phát hành theo khóa cũ vẫn hợp lệ cho đến khi chúng hết hạn. Cache giữ hợp nhất; trình xác thực chọn theo `kid`.

### Quy trình xác thực

Server MCP chạy xác thực trước khi điều phối bất kỳ công cụ nào. Hình dạng `code/main.py` sử dụng:

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` giải mã JWT, phân giải khóa ký từ cache JWKS (làm mới một lần khi miss), xác minh chữ ký, sau đó kiểm tra `iss` so với danh sách cho phép, `aud` so với tài nguyên chuẩn của server này, `exp`, và scope bắt buộc — trả về challenge `WWW-Authenticate` khi thất bại lần đầu. Giữ nó là một quy trình duy nhất trên resource server có nghĩa là mọi điểm vào (mọi cuộc gọi công cụ, mọi vận chuyển) đều đi qua các kiểm tra giống nhau; không có đường dẫn nào đến được công cụ mà không xác thực trước.

### Opaque token sử dụng introspection, không phải đoán mò

Không phải mọi access token đều là JWT. Nếu issuer ghi lại một opaque token, resource server không thể giải mã nó thành các claim đáng tin cậy. Nó gửi token đến endpoint introspection RFC 7662 của issuer qua một backchannel đã xác thực và yêu cầu `active: true`, ngữ cảnh issuer dự kiến, audience hoặc tài nguyên MCP chính xác, các claim thời gian chưa hết hạn, và các scope được yêu cầu bởi công cụ cụ thể.

Cache introspection theo issuer, một digest token một chiều, và tài nguyên MCP. Không bao giờ sử dụng token rõ ràng làm nhãn log hoặc cache. Ràng buộc một mục cache tích cực bằng thời điểm sớm nhất của thời hạn token, hướng dẫn cache của issuer, và mục tiêu độ tươi mới thu hồi của triển khai. Giữ cache tiêu cực đủ ngắn để một token mới phát hành không bị coi là không hoạt động sai lệch. Một kết quả cho một tài nguyên không thể ủy quyền cho tài nguyên khác ngay cả khi chuỗi opaque token giống hệt nhau.

Đừng chọn chế độ xác thực từ nội dung token do kẻ tấn công kiểm soát. Ghim hành vi JWT so với introspection vào metadata issuer đã xác thực và cấu hình triển khai. Trên đường dẫn JWT, ghim các thuật toán được chấp nhận và `jwks_uri` đáng tin cậy; không bao giờ theo một URL khóa hoặc thuật toán được chọn chỉ bởi header token.

### Thu hồi là một hợp đồng về độ tươi mới

RFC 7009 cho phép client yêu cầu authorization server thu hồi token. Yêu cầu đó không xóa các bản sao đã được cache bởi mọi resource server. Xác định độ trễ thu hồi tối đa có thể chấp nhận được và làm cho mọi cache tôn trọng nó.

Các triển khai opaque-token có thể đạt được sự thu hồi chặt chẽ hơn bằng cách introspect trên mỗi cuộc gọi rủi ro cao hoặc sử dụng cache tích cực ngắn. Các triển khai JWT tự chứa thường kết hợp thời hạn access-token ngắn với thu hồi refresh-token, thu hồi khóa cho các sự cố toàn issuer, và một danh sách đen tùy chọn theo subject, phiên hoặc token-id cho việc từ chối cục bộ khẩn cấp. Một JWT đã ký vẫn hợp lệ về mặt mật mã cho đến khi hết hạn trừ khi resource server có bằng chứng thu hồi bên ngoài hiện tại.

Đăng xuất, vô hiệu hóa tài khoản, rút lại sự đồng ý và ứng phó sự cố là các kích hoạt khác nhau nhưng phải hội tụ vào một tuyên bố có thể đo lường được: sau tối đa cửa sổ thu hồi đã khai báo, mọi bản sao đều từ chối thông tin xác thực. Kiểm tra tuyên bố đó thông qua bộ cân bằng tải, không chỉ đối với một tiến trình đang chạy.

### Lỗi phụ thuộc cần một quyết định đã khai báo

Không bao giờ ứng biến chính sách khả dụng bên trong trình xử lý ngoại lệ.

| Lỗi | Hành vi production an toàn |
|---|---|
| Làm mới JWKS theo lịch trình thất bại, `kid` đã biết vẫn nằm trong cache có giới hạn vẫn hợp lệ | Tiếp tục chỉ trong cửa sổ stale-on-error đã khai báo và phát ra bằng chứng sức khỏe suy giảm |
| Token có `kid` không xác định và lần làm mới được phép duy nhất thất bại | Từ chối; không bao giờ chấp nhận chữ ký không thể xác minh |
| Introspection không khả dụng | Thất bại đóng cho các cuộc gọi được bảo vệ; không chuyển đổi lỗi mạng thành `active: true` |
| Metadata tài nguyên được bảo vệ hoặc issuer thay đổi bất ngờ | Dừng đăng ký và lấy token mới; chỉ giữ cấu hình được ghim rõ ràng, chưa hết hạn theo chính sách sự cố có giới hạn |
| Endpoint thu hồi không khả dụng | Báo cáo đăng xuất hoặc thu hồi là không hoàn tất, giữ lại thông tin xác thực cục bộ như không thể sử dụng khi có thể, và không tuyên bố thu hồi toàn cầu đã thành công |
| Nguồn đồng hồ hoặc loại claim không hợp lệ | Từ chối thay vì mở rộng độ lệch cho đến khi token vượt qua |

Phân loại các lỗi tách biệt với thông tin xác thực không hợp lệ. Sự cố phụ thuộc là một lỗi vận hành với chính sách sức khỏe và thử lại. Chữ ký, issuer, audience, thời hạn hoặc scope xấu là một sự từ chối ủy quyền. Không bên nào đến được trình xử lý công cụ, và không bên nào được làm rò rỉ nội dung token vào bằng chứng kiểm toán.

### Hướng dẫn phát lại audience (hạn chế đặc quyền access-token)

Server A (`notes.example.com`) và Server B (`tasks.example.com`) đều đăng ký với cùng một authorization server. Server A bị xâm nhập. Kẻ tấn công lấy một token ghi chú của người dùng và phát lại nó đối với Server B.

Trình xác thực của Server B:

1. Giải mã JWT, lấy JWKS theo `kid`, xác minh chữ ký.
2. Kiểm tra `iss` so với `authorization_servers` trong metadata tài nguyên được bảo vệ của nó. (Đạt — cùng IdP.)
3. Kiểm tra `aud == "https://tasks.example.com"`. (Thất bại — `aud` của token là `https://notes.example.com`.)
4. Trả về 401 với `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`.

Claim audience là biện pháp phòng thủ duy nhất chống lại cuộc tấn công này ở lớp giao thức. Bỏ qua nó vì hiệu suất là sai lầm production phổ biến nhất; trình xác thực phải chạy trên mọi yêu cầu, không chỉ khi bắt đầu phiên. Đặc tả gọi đây là **hạn chế đặc quyền access-token**: một server MCP `MUST` từ chối bất kỳ token nào không nêu tên nó trong audience.

> **Ghi chú đặt tên.** Đặc tả dành thuật ngữ *confused deputy* cho một vấn đề liên quan nhưng khác biệt: một server MCP đóng vai trò là **proxy** OAuth cho API bên thứ ba, sử dụng ID client tĩnh, chuyển tiếp token mà không có sự đồng ý của người dùng trên mỗi client. Ghim audience sửa lỗi phát lại ở trên; sửa lỗi confused-deputy là sự đồng ý trên mỗi client **cộng với** việc không bao giờ chuyển token đến cho các API thượng nguồn (server MCP `MUST` lấy token thượng nguồn riêng biệt của chính nó).

### Tấn công mix-up (phòng thủ phía client mà server không thể cung cấp)

Một client nói chuyện với nhiều authorization server trong suốt vòng đời của nó. Một AS độc hại có thể cố gắng làm cho client đổi mã ủy quyền của một AS trung thực tại endpoint token của kẻ tấn công. Ghim audience không giúp ích gì ở đây — cuộc tấn công xảy ra trước khi bất kỳ token nào tồn tại. Sự phòng thủ nằm ở client (RFC 9207):

1. Trước khi chuyển hướng, client ghi lại `issuer` dự kiến từ metadata AS đã xác thực.
2. Trên phản hồi ủy quyền, client so sánh tham số `iss` được trả về với issuer đã ghi lại đó (so sánh chuỗi đơn giản, không chuẩn hóa) trước khi gửi mã đi bất cứ đâu.
3. Không khớp (hoặc `iss` vắng mặt khi AS quảng bá `authorization_response_iss_parameter_supported`) → từ chối, và thậm chí không hiển thị các trường `error`.

Chỉ riêng PKCE không ngăn được mix-up, vì client trao `code_verifier` của nó cho bất kỳ token endpoint nào mà nó được dẫn tới. Đây là lý do tại sao đặc tả ghi lại issuer trên mỗi yêu cầu cùng với verifier PKCE và `state`.

### Các chế độ thất bại

- **JWKS cũ.** Trình xác thực từ chối các token hợp lệ sau khi AS xoay vòng khóa. Giải pháp là mô hình cron-refresh + cache-miss-refetch ở trên. Không bao giờ cache JWKS mà không có tác vụ làm mới.
- **Xoay vòng làm phương án dự phòng.** Nối đường dẫn cache-miss với xoay vòng và đúc khóa thay vì lấy lại là một lỗi thực sự: nó không bao giờ tạo ra `kid` bị thiếu, và nó biến các giá trị `kid` do kẻ tấn công kiểm soát thành một cuộc tấn công DoS tạo khóa. Phương án dự phòng phải là `refresh-jwks` idempotent.
- **Thiếu claim `aud`.** Một số IdP mặc định bỏ qua `aud` trừ khi `resource` có mặt trong yêu cầu token. Trình xác thực phải từ chối các token thiếu `aud`, không coi sự vắng mặt là wildcard.
- **Mix-up qua kiểm tra `iss` bị thiếu.** Một client không xác thực tham số phản hồi ủy quyền `iss` theo RFC 9207 so với issuer mà nó đã ghi lại trước khi chuyển hướng có thể bị dẫn dắt đổi mã của một AS trung thực tại endpoint token của kẻ tấn công. Đây là lỗi phía client; resource server không thể bù đắp cho nó.
- **Cuộc đua nâng cấp scope.** Hai luồng step-up đồng thời cho cùng một người dùng đều có thể thành công và tạo ra hai access token với các scope khác nhau. Trình xác thực phải sử dụng token được trình bày trên yêu cầu, không tra cứu "scope hiện tại của người dùng" — điều đó tạo ra cửa sổ TOCTOU.
- **Đánh cắp token đăng ký.** Một `registration_access_token` bị rò rỉ cho phép kẻ tấn công viết lại các URI chuyển hướng. Băm chúng khi lưu trữ; yêu cầu client trình bày văn bản thuần trên mỗi cập nhật; xoay vòng khi nghi ngờ.
- **`iss` không được ghim.** Một trình xác thực chấp nhận bất kỳ `iss` nào cho phép kẻ tấn công thiết lập authorization server của riêng họ, đăng ký một client cho audience mục tiêu, và phát hành token. Danh sách `authorization_servers` trong metadata tài nguyên được bảo vệ là danh sách cho phép; hãy thực thi nó.
- **Va chạm cache thông tin xác thực hoặc token.** Một client chỉ khóa các đăng ký theo tài nguyên có thể trình bày danh tính của một authorization server này cho một server khác. Một client chỉ khóa access token theo issuer có thể phát lại token tại audience sai. Khóa các đăng ký theo issuer đã xác thực, khóa access token theo `(issuer, resource)`, và đăng ký lại bất cứ khi nào issuer thay đổi.

```figure
t3-jwks-rotate
```

## Sử dụng

`code/main.py` đi qua luồng production đầy đủ với Python stdlib và ba vai trò: `AuthorizationServer`, `ResourceServer`, và `Client`. Luồng:

Từ thư mục gốc của repository, chạy:

```bash
cd phases/13-tools-and-protocols/18-mcp-auth-production
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Lệnh đầu tiên in ra bản ghi đăng ký gắn kết với issuer và xác thực token. Lệnh thứ hai báo cáo mười tám kiểm tra vượt qua. Không lệnh nào mở listener mạng hoặc ghi thông tin xác thực.

1. Authorization server công bố metadata RFC 8414 tại `/.well-known/oauth-authorization-server`.
2. Client MCP gọi endpoint metadata và kiểm tra các tùy chọn đăng ký của nó (⟦PROTECT_1