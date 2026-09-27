# MCP Conformance Engineering: Versioning, Evidence, and Operations

> Một server không được coi là tuân thủ (conformant) chỉ vì "happy path" hoạt động thông qua một SDK. Sự tuân thủ nằm ở tầng wire (đường truyền), tại các ranh giới phiên bản, thông qua các trung gian và trong quá trình rollback.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 17 (gateways), Phase 13 · 30 (registry admission)
**Time:** ~100 phút

## Mục tiêu học tập

- Chuyển đổi các quy tắc MCP quy phạm thành các transcript wire (dữ liệu truyền tải) dạng "golden" và "negative".
- Giữ hành vi `2026-07-28` nghiêm ngặt tách biệt với cơ chế fallback kế thừa (legacy) có giới hạn.
- Phân biệt các trường bổ sung (additive unknown fields) với một `resultType` không xác định không hợp lệ.
- So sánh bằng chứng JSON-RPC thô với chế độ xem đã chuẩn hóa của SDK.
- Chứng minh tính toàn vẹn của header và body thông qua một ranh giới proxy thực tế.
- Kiểm soát các bản phát hành (release) bằng transcript đã được làm sạch (redacted), bằng chứng về sức khỏe (health) và rollback.

## Vấn đề

Client của bạn gọi `tools/list` thông qua một SDK và nhận được các tool. Bài kiểm tra tích hợp (integration test) vượt qua.

Kết quả đó để lại những câu hỏi quan trọng chưa được trả lời:

- Request có mang theo metadata giao thức hiện đại cho mỗi request không?
- `MCP-Protocol-Version`, `Mcp-Method` và `Mcp-Name` có khớp với body JSON-RPC không?
- Response có chứa `resultType` hợp lệ trên wire không, hay SDK đã tự tổng hợp ra nó?
- Client có bảo toàn một trường bổ sung trong tương lai không?
- Một lỗi hiện đại đã được nhận diện có vô tình kích hoạt handshake legacy không?
- Proxy có bảo toàn trạng thái gốc và lỗi JSON-RPC không?
- Notification serializer có phát ra response bị cấm không?
- Bộ phận vận hành có thể chứng minh lý do tại sao một bản phát hành được thúc đẩy hoặc rollback mà không cần lưu trữ bí mật không?

Sự tuân thủ là một tập hợp các bất biến (invariants) có thể quan sát được. Hãy xây dựng một bộ harness ghi lại các bất biến đó trước khi traffic production phải tự khám phá chúng.

```figure
mcp-conformance-operations
```

## Bắt đầu với các kỷ nguyên phiên bản (Version Eras)

MCP `2026-07-28` sử dụng metadata tự chứa cho mỗi request. Một request hiện đại mang theo `params._meta.io.modelcontextprotocol/protocolVersion` và `params._meta.io.modelcontextprotocol/clientCapabilities`. Các khóa có namespace chính xác là rất quan trọng; các alias `protocolVersion` hoặc `clientCapabilities` trần trụi là không hợp lệ. Khi các header định tuyến được phản chiếu (mirrored) xuất hiện tại ranh giới HTTP, giá trị của chúng phải khớp với body JSON-RPC. Các kết quả thành công hiện đại mang theo `resultType`.

Các phiên bản cho đến `2025-11-25` sử dụng kỷ nguyên khởi tạo sớm hơn. Một kết quả legacy không có `resultType` chỉ được hiểu là hoàn tất sau khi client đã chọn kỷ nguyên sớm hơn đó.

Đừng tạo một validator cho phép (permissive) chấp nhận cả hai hình thái cùng một lúc. Hãy sử dụng hai nhánh:

| Nhánh | Bằng chứng đầu vào | Thiếu `resultType` | Khởi tạo |
|---|---|---|---|
| Hiện đại | `server/discover` thành công hoặc response hiện đại được nhận diện | Không hợp lệ | Không phải đường dẫn mặc định |
| Legacy | Danh sách cho phép đã cấu hình cộng với kết quả `initialize` legacy hợp lệ sau một lần thăm dò hiện đại không kết luận được | Được hiểu là hoàn tất | Bắt buộc bởi kỷ nguyên đó |

Sự tách biệt này ngăn chặn một peer hiện đại bị lỗi nhận được sự xác thực lỏng lẻo hơn.

### Chế độ Strict (Nghiêm ngặt)

Chế độ Strict yêu cầu bằng chứng về hành vi hiện đại. Một `server/discover` thành công chứng minh nhánh hiện đại. Một lỗi JSON-RPC hiện đại được nhận diện cũng chứng minh điều đó. Hãy sửa request hoặc dừng lại. Không bao giờ hạ cấp (downgrade) vì server trả về `-32020`, `-32021` hoặc `-32022`.

### Chế độ Fallback

Chế độ Fallback thực hiện một lần thăm dò hiện đại có giới hạn. Timeout, phản hồi trống, kết nối bị đóng hoặc response không được nhận diện là không kết luận được. Nó không chứng minh rằng peer là legacy. Chỉ một endpoint được cấu hình hoặc nằm trong danh sách cho phép tương thích mới có thể nhận một lần thăm dò legacy có giới hạn, và client chỉ chọn nhánh legacy sau khi xác thực kết quả `initialize` của lần thăm dò đó và phiên bản legacy đã thương lượng.

Fallback không phải là "thử legacy sau bất kỳ lỗi nào". Một lỗi hiện đại được nhận diện chứa thông tin sửa lỗi hữu ích. Hạ cấp sau khi gặp lỗi đó có thể che giấu sự không khớp header, thiếu khai báo khả năng (capability), hoặc phiên bản không được hỗ trợ.

Điều này ngăn chặn kẻ tấn công, sự cố hoặc proxy lọc ép buộc hạ cấp bằng cách loại bỏ response hiện đại. Hãy ghi lại chính sách endpoint, quan sát hiện đại không kết luận được, bằng chứng legacy tích cực chính xác và kỷ nguyên đã chọn cùng nhau.

Ghi lại kỷ nguyên đã chọn bên cạnh mỗi transcript. Nếu không có sự thật đó, một trường bị thiếu có thể trông có vẻ chấp nhận được trong một lần chạy thử nghiệm này nhưng lại không hợp lệ trong lần chạy khác.

## Xây dựng kho lưu trữ Transcript

Một fixture transcript ghi lại những gì đã đi qua ranh giới, không chỉ là lệnh gọi SDK:

```json
{
  "name": "golden-modern-list",
  "era": "modern",
  "headers": {
    "MCP-Protocol-Version": "2026-07-28",
    "Mcp-Method": "tools/list"
  },
  "request": {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {
      "_meta": {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {}
      }
    }
  },
  "responseStatus": 200,
  "responseBody": {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
      "resultType": "complete",
      "tools": []
    }
  }
}
```

Giữ hai loại fixture.

### Golden transcripts

Golden transcripts chứng minh hành vi được chấp nhận:

- Khám phá hiện đại hoặc request phương thức với metadata và header khớp nhau
- Kết quả hoàn chỉnh với các trường bắt buộc
- Kết quả `input_required` khi phương thức có thể yêu cầu thêm đầu vào
- Kết quả mở rộng chỉ sau khi khả năng tương ứng đã được quảng bá
- Kết quả legacy không có `resultType`, nhưng chỉ trong kỷ nguyên legacy đã chọn
- Xử lý thông báo (notification) mà không có response JSON-RPC

Một golden transcript phải chính xác, không cần quá lớn. Hãy giữ các ID và timestamp biến đổi ở dạng xác định hoặc chuẩn hóa chúng trước khi so sánh.

### Negative transcripts

Negative transcripts chứng minh hành vi từ chối:

- Không khớp header và body
- Thiếu các khả năng (capabilities) cho mỗi request
- Phiên bản giao thức khớp không được hỗ trợ
- Thiếu `resultType` hiện đại
- `resultType` không xác định hoặc không được quảng bá
- `jsonrpc` response khác với `2.0` hoặc ID có giá trị hoặc kiểu JSON khác
- Response chứa cả `result` và `error`, hoặc không chứa cả hai
- Lỗi không có `code` kiểu số nguyên và `message` kiểu chuỗi
- Lỗi giao thức đã biết được ánh xạ tới HTTP status sai
- Response được phát ra cho một thông báo
- Envelope JSON-RPC bị lỗi
- Proxy làm sụp đổ một lỗi giao thức

Đối với mỗi trường hợp negative, hãy khẳng định ranh giới từ chối và mã lỗi ổn định. "Lệnh gọi thất bại" là quá yếu. Một mã 500 do proxy tạo ra và một `-32020` từ gốc đều có thể trông giống như thất bại trong khi kể cho người vận hành những câu chuyện hoàn toàn khác nhau.

Fixture không khớp header phải bao gồm response JSON-RPC HTTP 400 thực tế của server với ID request khớp và mã lỗi `-32020`. Thực thi điều đó tự động bất cứ khi nào validator cục bộ quan sát thấy `HeaderMismatch`; đừng biến việc xác minh response thành một cờ fixture tùy chọn. Một trường hợp với HTTP 500 và không có body sẽ thất bại ngay cả khi mã từ chối cục bộ là đúng. Một harness dừng lại sau khi validator request của chính nó ném ra lỗi chỉ mới kiểm tra chính nó, chứ không phải hành vi wire của server.

Dự án tuân thủ MCP chính thức rất hữu ích như một bộ kiểm tra bên ngoài và tài liệu tham khảo có phiên bản. Hãy giữ cả các transcript cục bộ của bạn. Chúng ghi lại proxy, SDK, xác thực, phần mở rộng và đường dẫn phát hành của bạn, điều mà một bộ kiểm tra chung không thể biết.

## Giá trị Header phải khớp với Body RPC

Trong Streamable HTTP hiện đại, các trung gian có thể định tuyến hoặc thực thi chính sách bằng cách sử dụng các header được phản chiếu. Body JSON-RPC vẫn là nguồn sự thật của giao thức. Sự không khớp là một thất bại về tính toàn vẹn, không phải là gợi ý để chọn một giá trị.

Xác thực theo thứ tự này:

1. Phân tích và xác thực envelope JSON-RPC và các kiểu metadata.
2. So sánh `MCP-Protocol-Version` với `params._meta.io.modelcontextprotocol/protocolVersion`.
3. So sánh `Mcp-Method` với `method`.
4. Khi phương thức có tên định tuyến, so sánh `Mcp-Name` với giá trị body tương ứng.
5. Sau khi thiết lập sự bình đẳng, quyết định xem phiên bản khớp và tập hợp khả năng có được hỗ trợ hay không.

Thứ tự này phân biệt `-32020` không khớp với `-32022` phiên bản không được hỗ trợ. Nó cũng ngăn chặn gateway ủy quyền tên header trong khi gốc thực thi một tên body khác.

Tên trường HTTP không phân biệt chữ hoa chữ thường, trong khi giá trị của chúng phân biệt chữ hoa chữ thường. Chuẩn hóa tên header trước khi tra cứu và từ chối các bản sao xung đột. Đối với `Mcp-Name` không an toàn, không phải ASCII, hoặc có khoảng trắng ở đầu/cuối, hãy giải mã sentinel UTF-8 `=?base64?{Base64EncodedValue}?=` chính xác trước khi so sánh nó với body. Từ chối sentinel không đầy đủ, Base64 không hợp lệ, UTF-8 không hợp lệ hoặc giá trị không an toàn thô với `-32020`. Khoảng trắng bao quanh thô là không hợp lệ ngay cả khi body chứa các ký tự tương tự vì giá trị đó yêu cầu mã hóa sentinel trước khi truyền tải.

Một trung gian có thể từ chối HTTP bị lỗi trước khi request đến server MCP, vì vậy thất bại của nó có thể là lỗi HTTP không có JSON-RPC. Ghi lại xem sự từ chối đến từ trung gian hay từ gốc. Server MCP gốc nên sử dụng hợp đồng lỗi giao thức khi nó xử lý một request JSON-RPC hợp lệ.

## Các trường không xác định không phải là kết quả không xác định

Khả năng tương thích tiến (forward compatibility) yêu cầu hai quy tắc khác nhau.

### Các trường không xác định bổ sung (Additive)

Các đối tượng kết quả và bản đồ `_meta` có thể có thêm các trường. Một validator nên bảo toàn hoặc bỏ qua một trường bổ sung tùy theo vai trò của nó, trừ khi trường đó vi phạm hợp đồng đã đặt trước. Mẫu này giữ toàn bộ kết quả thô trong bằng chứng và chấp nhận `futureHint` bên cạnh một kết quả đã biết.

Nếu bạn là một proxy trong suốt, việc bảo toàn một trường không xác định thường an toàn hơn là loại bỏ nó. Nếu bạn là một ứng dụng client, việc bỏ qua nó có thể hợp lệ. Bài kiểm tra vi sai (differential test) của bạn vẫn nên tiết lộ rằng SDK đã bỏ qua nó để hành vi đó là có chủ ý.

### `resultType` không xác định

`resultType` là một bộ phân biệt (discriminator). Các kết quả hiện đại cốt lõi sử dụng `complete` hoặc `input_required`. Một phần mở rộng chỉ có thể thêm một giá trị khác khi khả năng của nó đã được quảng bá. Ví dụ, phần mở rộng Tasks có thể thêm `task` trong ngữ cảnh khả năng đã thương lượng đó.

Một bộ phân biệt không xác định hoặc không được quảng bá không thể được coi là hoàn tất một cách an toàn. Client không biết vòng đời mà nó sẽ loại bỏ. Hãy từ chối nó.

Do đó, cùng một response thô có thể chứa một trường không xác định chấp nhận được và một kiểu kết quả không xác định không chấp nhận được. Hãy kiểm tra cả hai trường hợp.

Bộ phân biệt chỉ là lớp đầu tiên. Xác thực payload cụ thể của phương thức sau đó. Một kết quả `tools/list` hoàn chỉnh cần một mảng `tools` mà các bộ mô tả của nó có tên không trống duy nhất, mô tả hữu ích và các giá trị `inputSchema` gốc đối tượng. Một kết quả `task` chỉ hợp lệ cho một `tools/call` đủ điều kiện với khả năng Tasks và yêu cầu `taskId`, trạng thái đã biết, timestamp tạo và cập nhật, và `ttlMs`, cộng với khoảng thời gian thăm dò tùy chọn hợp lệ. Một kết quả `completion/complete` hoàn chỉnh yêu cầu một đối tượng `completion` với không quá 100 giá trị chuỗi, một số nguyên không âm tùy chọn `total` không nhỏ hơn các giá trị được trả về, và một Boolean tùy chọn `hasMore`. Một `resultType` được viết đúng chính tả không thể làm cho một payload bị lỗi trở nên tuân thủ.

## Bất biến thông báo (Notification Invariant)

Một thông báo JSON-RPC không có `id`. Người nhận không được gửi response thành công hoặc lỗi JSON-RPC.

Đối với một hình thái thông báo HTTP được chấp nhận, harness mong đợi một HTTP `202` với body trống. MCP `2026-07-28` không định nghĩa thông báo client-to-server cốt lõi nào qua Streamable HTTP. Mẫu này sử dụng một thông báo phần mở rộng khóa học có namespace chỉ để kiểm tra bất biến serializer một chiều. Đừng trình bày nó như một phương thức cốt lõi mới.

Kiểm tra serializer, không chỉ handler. Một handler có thể trả về `None` trong khi middleware bao bọc nó trong một đối tượng thành công JSON. Ghi lại các byte đầu ra cuối cùng.

## Thêm một SDK Differential

SDK thường biến các đối tượng wire thành các kiểu ngôn ngữ thuận tiện. Điều đó hữu ích, nhưng một đối tượng đã chuẩn hóa không thể chứng minh những gì đã nhận được.

Đối với mỗi fixture rủi ro cao, hãy ghi lại:

1. Trạng thái thô, header và response body trước khi giải mã SDK.
2. Giá trị trả về hoặc ngoại lệ đã chuẩn hóa bởi SDK.
3. Dự báo ngữ nghĩa mong đợi cho kỷ nguyên đã chọn.
4. Các trường được nâng lên, tổng hợp, loại bỏ hoặc thay đổi bởi SDK.

Mẫu này cho phép SDK loại bỏ các sổ sách wire đã biết như `resultType`, `_meta`, `ttlMs` và `cacheScope` trong khi so sánh payload ứng dụng. Nó báo cáo một `futureHint` bị loại bỏ vì trường ngữ nghĩa không xác định đó đã biến mất.

Đừng cho rằng mọi sự khác biệt đều là lỗi SDK. Mục đích là làm cho sự biến đổi trở nên rõ ràng. Quyết định xem thành phần của bạn là một endpoint ứng dụng, có thể bỏ qua một trường bổ sung, hay một trung gian trong suốt, nên bảo toàn nó.

Chạy bài kiểm tra vi sai đối với mọi SDK và phiên bản bạn phát hành. Nếu hai SDK chuẩn hóa cùng một transcript khác nhau, chính sách phát hành nên nêu rõ hành vi nào là chấp nhận được thay vì chọn đầu ra thuận tiện nhất sau khi sự việc đã xảy ra.

## Ghi lại bằng chứng Proxy

Hầu hết các lỗi MCP trong production xảy ra trên nhiều hơn một tiến trình. Ghi lại ba góc nhìn:

| Góc nhìn | Bằng chứng tối thiểu |
|---|---|
| Ingress | request headers, JSON-RPC body, content type, authenticated route, receive time |
| Origin | forwarded headers and body digest, origin status, response headers and body |
| Egress | client-visible status, headers, body, and send time |

Mẫu này phát hiện hai biến đổi phổ biến:

- một lỗi JSON-RPC HTTP 400 hoặc 404 từ gốc trở thành lỗi 500 chung của proxy
- body JSON-RPC egress khác với body gốc

Thêm các khẳng định cụ thể cho việc triển khai về content type, `Accept`, nén, SSE phạm vi request, cache headers và trace correlation. Ghi lại cả hai phía của việc kết thúc TLS khi chính sách cho phép. Không bao giờ ghi lại thông tin xác thực chỉ để chứng minh đường dẫn.

## Làm sạch (Redact) trước khi bằng chứng rời khỏi bộ nhớ

Làm sạch là một phần của hoạt động tuân thủ, không phải là công việc dọn dẹp sau đó. Áp dụng nó trước khi tuần tự hóa, băm, ghi nhật ký, tạo artifact kiểm tra hoặc tải lên lỗi.

Mẫu này thực hiện case-fold tên khóa và loại bỏ các dấu phân cách trước khi khớp, sau đó thay thế đệ quy các giá trị dưới các khóa như `Authorization`, `Cookie`, `Set-Cookie`, `X-Api-Key`, `accessToken`, `clientSecret`, `registrationAccessToken`, `token`, `password`, `secret` và `api_key`. Việc chuẩn hóa và danh sách đen (denylist) phải sử dụng cùng một hình thức để các biến thể camelCase, có dấu gạch nối, dấu gạch dưới và dấu chấm không thể vượt qua chính sách của nhau. Một bộ thu thập production nên thêm chính sách đối số cụ thể cho phương thức, vì một khóa vô hại như `query` vẫn có thể chứa dữ liệu cá nhân hoặc dữ liệu được quản lý.

Băm gói bằng chứng đã làm sạch. Chỉ giữ các bản ghi thô trong một hệ thống ngắn hạn được phê duyệt khi một cuộc điều tra cụ thể yêu cầu chúng. Một digest chứng minh gói đã làm sạch nào đã thúc đẩy quyết định; nó không tiết lộ giá trị đã bị loại bỏ.

## Biến Sức khỏe và Rollback thành một phần của Cổng kiểm soát (Gate)

Sự tuân thủ giao thức là cần thiết nhưng không đủ để phát hành. Một ứng viên tuân thủ vẫn có thể bị timeout, rò rỉ bộ nhớ hoặc quá tải một phụ thuộc.

Xác định cửa sổ sức khỏe trước khi triển khai:

- số lượng mẫu tối thiểu
- tỷ lệ lỗi tối đa
- phân vị độ trễ tối đa
- giới hạn bão hòa hoặc tài nguyên
- thời gian quan sát
- so sánh với baseline đã thừa nhận

Xác định bằng chứng rollback trước khi triển khai:

- phiên bản trước chính xác
- digest bằng chứng thừa nhận
- SHA-256 artifact và descriptor pins
- trạng thái Registry hiện tại
- kết quả sức khỏe hiện tại
- quy trình khôi phục tuyến đường
- một chứng thực (attestation) về các trường chính xác đó từ một danh tính bộ điều khiển phát hành đáng tin cậy

Yêu cầu mục tiêu rollback đó phải được xác minh và khỏe mạnh trước khi thúc đẩy, không chỉ sau khi ứng viên thất bại. Một bản phát hành thành công mà không có đường dẫn khôi phục khả dụng là không sẵn sàng cho production.

Nếu một ứng viên thất bại và mục tiêu rollback thiếu bằng chứng đó, hãy giữ traffic thay vì đoán. "Rollback về bất cứ thứ gì ở đó" không phải là một kiểm soát vận hành.

Đừng giảm sự sẵn sàng xuống các kiểm tra truthiness như phiên bản không trống, `healthy: "yes"`, hoặc một chuỗi bằng chứng tùy ý. Mẫu này yêu cầu các kiểu chính xác, trạng thái hoạt động, ba digest SHA-256, người ký đáng tin cậy và chứng thực HMAC-SHA-256 hợp lệ trên toàn bộ payload rollback. Khóa demo xác định của nó là một fixture không bí mật. Hãy tiêm một khóa được bảo vệ, kết quả xác minh KMS, hoặc bộ xác minh chứng thực khóa công khai tại ranh giới phát hành trong production.

Cổng phát hành cũng từ chối transcript trống, SDK differential, hoặc bằng chứng proxy. Mỗi nguồn phải mang các digest bằng chứng hợp lệ. Một cửa sổ sức khỏe xanh không thể lấp đầy một ranh giới chưa bao giờ được quan sát.

## Xây dựng nó

Chạy harness thư viện tiêu chuẩn:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
```

Bản demo chạy chính xác mười lăm golden và negative transcript, bao gồm các kết quả hoàn tất hợp lệ và bị lỗi, so sánh kết quả thô với chế độ xem SDK, kiểm tra một proxy đã làm sụp đổ lỗi gốc, đánh giá sức khỏe, xác thực bằng chứng rollback và chọn mục tiêu đó.

Hình thái mong đợi:

```json
{
  "transcriptsPassed": 15,
  "transcriptsTotal": 15,
  "sdkDroppedFields": ["futureHint"],
  "proxyIssues": [
    "proxy collapsed a protocol error into HTTP 500",
    "proxy changed the origin JSON-RPC body"
  ],
  "releaseAction": "rollback",
  "evidenceDigest": "..."
}
```

Đọc `code/main.py` theo thứ tự này:

1. `validate_request()` thực thi các quy tắc request và header cụ thể cho kỷ nguyên.
2. `validate_result()` tách biệt các bộ phân biệt legacy bị thiếu, giá trị hiện đại hợp lệ, phần mở rộng và giá trị không xác định.
3. `select_era()` thực hiện chính sách fallback nghiêm ngặt và có giới hạn.
4. `run_transcript()` đánh giá các fixture golden và negative.
5. `compare_sdk_view()` phơi bày sự khác biệt về chuẩn hóa.
6. `inspect_proxy()` so sánh bằng chứng ingress, origin và egress.
7. `redact()` loại bỏ các bí mật rõ ràng trước khi băm bằng chứng.
8. `rollback_evidence_ready()` xác thực các trường pin chính xác và chứng thực phát hành đáng tin cậy.
9. `ReleaseGate.evaluate()` kết hợp bằng chứng tuân thủ, SDK, proxy, sức khỏe và rollback không trống.

## Sử dụng nó

Chạy harness tại bốn điểm:

1. Trên mỗi thay đổi triển khai với một bộ điều hợp kiểm tra trong tiến trình.
2. Đối với các binary client và server đã xây dựng qua transport thực tế.
3. Thông qua proxy hoặc gateway đã triển khai trong môi trường staging.
4. Trong quá trình triển khai canary với bằng chứng sức khỏe và rollback trực tiếp.

Giữ các tên trường hợp ổn định trên các lớp. `negative-header-body-mismatch` phải có nghĩa là cùng một bất biến trong các báo cáo đơn vị, end-to-end, proxy và canary. Digest bằng chứng sẽ khác vì ranh giới đã thay đổi; yêu cầu thì không.

Lưu trữ các lược đồ fixture trong kiểm soát phiên bản. Lưu trữ bằng chứng chạy đã làm sạch trong hệ thống phát hành của bạn. Lưu trữ các bản ghi thô ngắn hạn chỉ dưới các kiểm soát truy cập sự cố.

## Interactive Lab

### Lab A: chứng minh ranh giới kỷ nguyên

Từ thư mục `code`, mở Python:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/code
python3 -q
```

Chạy:

```python
from main import *
validate_result({"tools": []}, "legacy")
validate_result({"tools": []}, "modern")
```

Lệnh gọi legacy suy ra `complete`. Lệnh gọi hiện đại ném ra `ProtocolViolation`. Bây giờ hãy kiểm tra fallback:

```python
select_era({"kind": "timeout"}, "fallback")
select_era(
    {"kind": "timeout"},
    "fallback",
    legacy_allowed=True,
    legacy_evidence={"kind": "initialize_success", "protocolVersion": LEGACY_VERSION},
)
select_era({"kind": "jsonrpc_error", "code": -32021}, "fallback")
```

Lần timeout đầu tiên thất bại vì sự im lặng không phải là bằng chứng legacy. Lần gọi thứ hai chỉ chọn legacy vì cấu hình cho phép và một kết quả khởi tạo legacy hợp lệ đã được quan sát. Lỗi thiếu khả năng được nhận diện chứng minh nhánh hiện đại.

### Lab B: trường bổ sung so với bộ phân biệt

```python
validate_result({"resultType": "complete", "tools": [], "futureHint": True}, "modern")
validate_result({"resultType": "future_mode", "tools": []}, "modern")
```

Kết quả đầu tiên bảo toàn `futureHint`. Kết quả thứ hai bị từ chối vì bộ phân biệt vòng đời không xác định.

### Lab C: kiểm tra biến đổi SDK

```python
compare_sdk_view(
    {"resultType": "complete", "tools": [], "futureHint": {"mode": "new"}},
    {"tools": []},
)
```

Quyết định xem thành phần của bạn có thể bỏ qua `futureHint` hay phải chuyển tiếp nó. Viết lựa chọn đó vào chính sách phát hành. Đừng xóa differential một cách âm thầm.

### Lab D: sửa proxy

Sửa đổi trao đổi demo để egress bảo toàn trạng thái và body gốc. Chạy lại `python3 main.py`. Các vấn đề proxy sẽ biến mất, nhưng SDK differential vẫn chặn việc thúc đẩy. Sau đó bao gồm `futureHint` trong chế độ xem SDK và quan sát hành động thay đổi thành `promote` khi mọi nguồn bằng chứng đều vượt qua.

## Practice Lab

Thêm các transcript SSE phạm vi request vào harness.

Yêu cầu:

- Ghi lại trạng thái response, content type, các sự kiện SSE theo thứ tự và kết thúc luồng.
- Chứng minh mỗi sự kiện JSON-RPC có kết quả hoặc lỗi cụ thể cho kỷ nguyên hợp lệ.
- Thêm một trường hợp negative cho proxy đệm toàn bộ luồng trước khi chuyển tiếp.
- Thêm một trường hợp negative cho sự kiện SSE có id JSON-RPC khác với request.
- Làm sạch dữ liệu sự kiện trước khi viết bằng chứng.
- Bao gồm thời lượng luồng, độ trễ sự kiện đầu tiên và số lượng sự kiện trong cửa sổ sức khỏe.
- Làm cho cổng phát hành chỉ chọn một mục tiêu rollback có bằng chứng khi luồng thất bại.

Thành công có nghĩa là cùng một trường hợp chạy trực tiếp và thông qua proxy, với một báo cáo xác định ranh giới chính xác đã thay đổi hành vi.

## Shipped Artifact

Bài học này cung cấp `outputs/skill-mcp-conformance-release-gate.md`. Sử dụng nó để biến một thay đổi server, client, gateway hoặc SDK thành một ma trận tuân thủ có phiên bản và quyết định phát hành. Artifact yêu cầu bằng chứng wire thô, các trường hợp negative, lựa chọn kỷ nguyên rõ ràng, SDK differential, bằng chứng proxy, làm sạch, ngưỡng sức khỏe và bằng chứng rollback.

## Xác minh nó

Chạy demo và bộ kiểm tra xác định:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Xác minh phải chứng minh:

- mọi transcript golden và negative đi kèm đều đạt được kết quả mong đợi
- các request hiện đại yêu cầu các khóa metadata có namespace chính xác
- tên header HTTP được khớp không phân biệt chữ hoa chữ thường và các giá trị `Mcp-Name` được mã hóa được giải mã chính xác
- sự không khớp header và body trả về mã không khớp hiện đại
- phiên bản response, ID, tính độc quyền của kết quả hoặc lỗi, hình thái lỗi và ánh xạ HTTP được xác thực
- các yêu cầu payload tool-list, task và completion cụ thể cho phương thức được thực thi
- mọi `HeaderMismatch` được quan sát yêu cầu một response JSON-RPC HTTP 400 `-32020` thực tế
- khoảng trắng `Mcp-Name` thô bị từ chối trong khi khoảng trắng được mã hóa sentinel chính xác được round-trip
- `resultType` bị thiếu chỉ hợp lệ trong kỷ nguyên legacy đã chọn
- các trường bổ sung tồn tại sau xác thực thô trong khi các kiểu kết quả không xác định thất bại
- các kiểu kết quả phần mở rộng yêu cầu khả năng đã quảng bá của chúng
- các lỗi hiện đại được nhận diện không bao giờ gây ra fallback legacy
- các thông báo không tạo ra response JSON-RPC
- việc loại bỏ sổ sách SDK và mất trường ngữ nghĩa được phân biệt
- sự sụp đổ lỗi proxy được phát hiện và thông tin xác thực được làm sạch đệ quy trên các biến thể camelCase và dấu phân cách
- việc thúc đẩy yêu cầu bằng chứng transcript, SDK, proxy và vận hành khỏe mạnh không trống
- việc thúc đẩy và rollback đều yêu cầu một mục tiêu rollback được xác thực, ghim, hoạt động, khỏe mạnh

## Các chế độ thất bại trong Production

| Thất bại | Những gì bài kiểm tra yếu báo cáo | Những gì harness phải chứng minh |
|---|---|---|
| SDK tổng hợp một bộ phân biệt bị thiếu | “tools/list passed” | Kết quả hiện đại thô thiếu `resultType` và không hợp lệ |
| Client hạ cấp sau `-32021` | “legacy retry worked” | Lỗi hiện đại được nhận diện cấm fallback |
| Kiểu kết quả không xác định được coi là hoàn tất | “response parsed” | Bộ phân biệt vòng đời không được quảng bá bị từ chối |
| Proxy ủy quyền một tool và gốc thực thi một tool khác | “request reached server” | `Mcp-Name` bằng với tên định tuyến body tại mỗi chặng |
| Harness ném lỗi trước khi đọc response server | “header mismatch test passed” | Response HTTP 400 và JSON-RPC `-32020` được ghi lại và xác thực |
| Proxy biến 400 gốc thành 500 chung | “upstream error” | Trạng thái và body JSON-RPC gốc và egress được bảo toàn |
| Notification middleware phát ra `{result: null}` | “handler returned none” | Body egress cuối cùng trống và không có response JSON-RPC tồn tại |
| SDK loại bỏ một trường bổ sung | “typed objects match” | Chế độ xem thô và đã chuẩn hóa cho thấy trường bị loại bỏ chính xác |
| Artifact thất bại làm rò rỉ bearer token | “debug bundle uploaded” | Làm sạch xảy ra trước khi băm, ghi nhật ký hoặc tải lên |
| Kiểu khóa thông tin xác thực vượt qua làm sạch | “denylist contains api_key” | Các biến thể camelCase và dấu phân cách chia sẻ một hình thức danh sách đen chuẩn |
| Canary không có mẫu nhưng có vẻ khỏe mạnh | “zero errors” | Số lượng mẫu tối thiểu được thực thi |
| Rollback chọn một bản build không xác định | “previous deployment restored” | Phiên bản mục tiêu, digest thừa nhận, pins, trạng thái và sức khỏe hiện diện |

## Quy tắc vận hành

Kiểm tra các byte bạn gửi, các byte mà mọi trung gian chuyển tiếp, ngữ nghĩa mà mỗi SDK phơi bày và các bằng chứng mà bộ phận vận hành sẽ sử dụng dưới áp lực. Khả năng tương thích là một nhánh rõ ràng. Rollback là một hành động phát hành được hỗ trợ bởi bằng chứng. Không cái nào trong số đó nên là tác dụng phụ ngẫu nhiên của một trình phân tích cú pháp lỏng lẻo.

## Đọc thêm

- [MCP 2026-07-28 base protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP version negotiation](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Official MCP conformance project](https://github.com/modelcontextprotocol/conformance)