# Độ tin cậy, Hủy bỏ và Kiểm soát luồng trong MCP

> Một request ID chỉ có tác dụng liên kết các thông điệp. Nó không làm cho một side effect trở nên an toàn, không dừng một worker, hay bảo vệ một stream khỏi một consumer chậm chạp.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 09 và 13
**Time:** ~120 phút

## Mục tiêu học tập

- Triển khai tín hiệu hủy bỏ (cancellation signal) chính xác cho stdio và Streamable HTTP.
- Giải quyết các tình trạng tranh chấp (race condition) giữa hoàn thành và hủy bỏ mà không gửi thông điệp sau khi đã hủy.
- Tách biệt việc hủy bỏ request khỏi ngữ nghĩa `tasks/cancel` bền vững.
- Xây dựng các quyết định thử lại (retry) dựa trên side effect và idempotency key rõ ràng.
- Giới hạn các hàng đợi tiến trình (progress queue) trong khi vẫn bảo toàn các phản hồi cuối cùng.
- Khôi phục stream thông qua kết nối lại, truy xuất lại và backoff có jitter.

## Vấn đề

"Happy path" (luồng chạy thành công) thường che giấu những lỗi hệ thống phân tán đắt giá nhất.

Một client gọi một tool. Server bắt đầu thực hiện công việc. Tiến trình (progress) được gửi đi. Một proxy đệm stream đó. Client đạt đến giới hạn thời gian chờ (timeout) và ngắt kết nối. Server hoàn thành công việc chỉ một mili giây sau đó. Client thử lại với một JSON-RPC id mới. Mutation chạy hai lần.

Mọi thành phần đều hoạt động đúng cục bộ. Hệ thống thất bại toàn cục.

MCP định nghĩa hành vi của thông điệp và transport, nhưng ứng dụng của bạn vẫn phải chịu trách nhiệm về:

- Ngân sách thời gian (time budgets);
- Idempotency nghiệp vụ;
- Hàng đợi có giới hạn;
- Phân loại thử lại;
- Trạng thái tác vụ bền vững (durable task state);
- Chính sách kết nối lại và truy xuất lại.

Bài học này tích hợp các quyết định đó vào một trình mô phỏng tất định (deterministic simulator). Không có sleep, socket hay lỗi ngẫu nhiên. Bạn kiểm soát trực tiếp thứ tự sự kiện hủy bỏ. Một bài kiểm tra trên luồng đồng bộ sẽ buộc hai ledger client cạnh tranh cho cùng một idempotency key.

## Hủy bỏ Request phụ thuộc vào Transport

Mục đích là như nhau trên mọi transport: client không còn cần kết quả của request đang thực hiện nữa. Tín hiệu trên đường truyền thì khác nhau.

### stdio

stdio sử dụng một kênh hai chiều dùng chung. Client gửi một thông báo:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/cancelled",
  "params": {
    "requestId": 41,
    "reason": "User closed the operation"
  }
}
```

Thông báo này là "fire-and-forget" (gửi và quên). Server không phát ra phản hồi JSON-RPC cho nó.

Server nên dừng công việc, giải phóng tài nguyên và tránh gửi phản hồi cho request đã bị hủy. Nó có thể bỏ qua việc hủy bỏ khi request đó không xác định, đã hoàn thành hoặc không thể dừng một cách an toàn.

Các thông báo hủy bỏ bị lỗi, không xác định hoặc đã hoàn thành sẽ bị bỏ qua. Việc biến những tình trạng tranh chấp đó thành lỗi mới sẽ tạo ra nhiều tình trạng tranh chấp hơn.

### Streamable HTTP

Streamable HTTP hiện đại cung cấp cho mỗi request một stream phản hồi HTTP hoặc SSE riêng. Client hủy bỏ bằng cách đóng stream phản hồi của request đó.

Đừng POST `notifications/cancelled` cho một request HTTP thông thường. Việc đóng stream chính là tín hiệu hủy bỏ.

Khi server nhận thấy kết nối bị ngắt, nó nên dừng công việc và không được gửi thêm thông điệp nào cho request đó.

### Hủy bỏ do Server gửi có phạm vi hẹp

Server không sử dụng `notifications/cancelled` để hủy các lệnh gọi tùy ý của client. Trên stdio, việc hủy bỏ do server gửi chỉ dành riêng cho việc chấm dứt một request `subscriptions/listen`. Hãy giữ luồng đó tách biệt với việc hủy bỏ request thông thường của client.

## Hủy bỏ là một cuộc đua (Race)

Hai thứ tự sự kiện đều hợp lệ.

### Hủy bỏ thắng

```text
request starts
client sends cancellation signal
server marks request cancelled
worker reaches completion
server suppresses the response
```

### Hoàn thành thắng

```text
request starts
worker commits the result
server sends the response
cancellation arrives late
server ignores the late notification
```

Client cũng phải bỏ qua phản hồi muộn cho một request mà nó đã từ bỏ. Độ trễ mạng có nghĩa là không bên nào có thể chứng minh bên kia đã quan sát thấy sự kiện nào trước.

```figure
mcp-reliability-race
```

`RequestCoordinator` của bài học lưu trữ một trạng thái cuối cùng. `complete()` không trả về phản hồi sau khi hủy bỏ. Một yêu cầu hủy bỏ muộn không thể thay đổi một bản ghi đã hoàn thành.

## Timeout cần hai đồng hồ

Một bộ đếm thời gian không hoạt động (inactivity timer) là không đủ.

Sử dụng hai giới hạn:

1. **Idle timeout:** Khoảng thời gian request có thể không tạo ra hoạt động hữu ích nào.
2. **Maximum timeout:** Ngân sách thời gian thực tuyệt đối kể từ khi bắt đầu request.

Tiến trình có thể reset đồng hồ idle. Nó không bao giờ được phép xóa deadline tối đa.

```text
start: 0 ms
progress: 400 ms
progress: 800 ms
progress: 1200 ms
idle timeout: 500 ms
maximum timeout: 2000 ms
```

Tại 1500 ms, request vẫn hoạt động vì tiến trình mới nhất chỉ mới 300 ms. Tại 2000 ms, deadline tối đa sẽ hủy nó ngay cả khi một sự kiện tiến trình khác vừa đến lúc 1999 ms.

Tiến trình là tùy chọn. Server có thể chấp nhận một token tiến trình và không phát ra cập nhật nào. Đừng bao giờ biến sự hiện diện của một token thành một timeout vô hạn.

Các giá trị tiến trình MCP phải tăng dần. Các thông báo sẽ dừng sau khi hoàn thành hoặc hủy bỏ. Hãy giới hạn tốc độ tiến trình để một worker nhanh không thể làm tràn transport.

## Hủy bỏ Request không phải là `tasks/cancel`

Các cơ chế này giải quyết các vòng đời khác nhau.

| Cơ chế | Mục tiêu | Tín hiệu | Ý nghĩa của thành công |
|-----------|--------|--------|--------------------|
| Hủy bỏ request trên stdio | Một RPC đang thực hiện | `notifications/cancelled` | Client đã từ bỏ request; server nên dừng nếu khả thi |
| Hủy bỏ request trên HTTP | Một stream phản hồi đang thực hiện | Đóng stream | Client đã từ bỏ request; server nên dừng nếu khả thi |
| `tasks/cancel` | Một Task bền vững | Request MCP thông thường | Server đã xác nhận ý định hủy bỏ |

Kết quả `tasks/cancel` thành công không chứng minh được worker đã dừng. Task có thể vẫn ở trạng thái `working` cho đến khi một checkpoint của worker quan sát thấy cờ này. Công việc có thể hoàn thành trước checkpoint đó.

Đừng xóa trạng thái Task bền vững khi kết nối HTTP đóng. Lý do để tạo một Task là vòng đời của nó tồn tại lâu hơn một request và một kết nối.

## JSON-RPC ID mới không phải là Idempotency

JSON-RPC id liên kết các request và response. Chúng không xác định một nghiệp vụ.

Giả sử một client gửi một khoản phí với id `41`, mất phản hồi và thử lại với id `42`. Server thấy hai thông điệp khác nhau. Nếu không có khóa ứng dụng, nó không thể biết chúng đại diện cho cùng một giao dịch thanh toán.

Một idempotency key xác định ý định nghiệp vụ:

```json
{
  "name": "charge_account",
  "arguments": {
    "account": "acct-7",
    "cents": 1200,
    "idempotencyKey": "checkout-7"
  }
}
```

Server lưu trữ:

- Khóa;
- Dấu vân tay (fingerprint) của các đối số thao tác;
- Kết quả đã cam kết.

Cùng một khóa và cùng các đối số sẽ trả về kết quả đã lưu. Cùng một khóa với các đối số khác nhau sẽ bị từ chối. Điều này ngăn chặn việc vô tình tái sử dụng khóa làm thay đổi một nghiệp vụ khác.

### Ranh giới ledger phải nguyên tử và bền vững

Chuỗi này không an toàn:

```text
check key
run mutation
store result
```

Hai worker có thể cùng thấy một khóa bị thiếu và cả hai đều chạy mutation. Một sự cố sau khi thực hiện nhưng trước khi lưu sẽ tạo ra sự mơ hồ tương tự khi thử lại.

Bài học sử dụng một ledger SQLite dựa trên tệp. `BEGIN IMMEDIATE` tuần tự hóa việc kiểm tra khóa, hiệu ứng nghiệp vụ mô phỏng, bộ đếm thực thi và kết quả đã lưu vào một giao dịch duy nhất. Do đó, hai kết nối ledger độc lập tranh chấp với cùng một khóa sẽ quan sát thấy một kết quả đã cam kết và một lần thực thi. Việc đóng và mở lại ledger sẽ giữ bản ghi đó.

Mỗi giá trị trả về được tái tạo từ JSON đã lưu. Người gọi không bao giờ nhận được đối tượng có thể thay đổi (mutable) do ledger nắm giữ, vì vậy việc thay đổi một dictionary được trả về không thể làm hỏng các kết quả phát lại (replay) sau này.

Hiệu ứng nghiệp vụ của trình mô phỏng là bộ đếm biên nhận và thực thi bên trong cùng một giao dịch SQLite. Một khoản thanh toán, triển khai hoặc lệnh gọi API bên ngoài thực tế không trở nên nguyên tử chỉ bằng cách ghi vào một bảng cục bộ. Sản xuất cần một giao dịch cơ sở dữ liệu chia sẻ bền vững, một transactional outbox hoặc một nhà cung cấp thượng nguồn thực thi cùng một idempotency key. Chỉ riêng khóa tiến trình (process lock) không bảo vệ được nhiều bản sao hoặc tồn tại sau khi khởi động lại.

### Ma trận thử lại

Phân loại các lần thử lại trước khi triển khai chúng.

| Loại | Ví dụ | Quy tắc thử lại |
|------|---------|------------|
| An toàn | Đọc tất định không có side effect | Thử lại với JSON-RPC id mới sau khi hiểu ranh giới lỗi |
| Có điều kiện | Mutation với idempotency key bền vững | Thử lại với cùng khóa và các đối số giống hệt |
| Không an toàn | Mutation không có khử trùng lặp nghiệp vụ | Không tự động thử lại; hãy đối soát trước |

Các chú thích tool như `readOnlyHint` và `idempotentHint` vẫn là những gợi ý không đáng tin cậy. Hợp đồng ứng dụng và việc triển khai server quyết định tính an toàn của việc thử lại.

## Backpressure là một phần của tính đúng đắn

Một nhà sản xuất SSE có thể tạo ra tiến trình nhanh hơn mức client, proxy hoặc mạng có thể tiêu thụ. Một hàng đợi không giới hạn sẽ biến sự chậm chạp thành cạn kiệt bộ nhớ.

Sử dụng hàng đợi có giới hạn và xác định những gì có thể mất.

Tiến trình có thể thay thế được. Một giá trị tiến trình sau thay thế giá trị trước cho cùng một token. Một phản hồi JSON-RPC cuối cùng không thể thay thế được.

Bộ đệm của bài học áp dụng chính sách này:

1. Gộp (coalesce) các tiến trình liền kề cho cùng một token.
2. Loại bỏ tiến trình cũ nhất khi đạt đến dung lượng.
3. Đánh dấu stream là cần truy xuất lại có thẩm quyền.
4. Bảo toàn phản hồi cuối cùng.
5. Từ chối trạng thái mà việc bảo toàn phản hồi cuối cùng đòi hỏi phải loại bỏ một phản hồi cuối cùng khác.

Đây là sự mất mát có giới hạn với khả năng phục hồi rõ ràng. Mất mát âm thầm không phải là một chiến lược.

### Đệm Proxy

Server có thể stream chính xác trong khi reverse proxy giữ các sự kiện trong bộ đệm.

Đối với phản hồi SSE, hãy gửi:

```http
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

Đặc tả Streamable HTTP 2026 khuyến nghị `X-Accel-Buffering: no` để các proxy tương thích phân phối sự kiện ngay lập tức.

Đối với các stream tồn tại lâu và yên tĩnh, hãy định kỳ phát ra một bình luận SSE:

```text
:
```

Client bỏ qua các dòng bình luận. Các trung gian thấy lưu lượng truy cập và ít có khả năng đóng kết nối nhàn rỗi.

Keepalive không phải là tiến trình. Đừng reset timeout nhàn rỗi ngữ nghĩa của một thao tác chỉ vì một bình luận transport đã đến.

## Kết nối lại nghĩa là Truy xuất lại

Streamable HTTP hiện đại không hỗ trợ SSE có thể tiếp tục thông qua `Last-Event-ID`.

Sau khi một stream `subscriptions/listen` bị ngắt:

1. Mở một request lắng nghe mới với một JSON-RPC id mới.
2. Khôi phục bộ lọc đăng ký mong muốn.
3. Truy xuất lại các tool, resource, prompt hoặc Task bị ảnh hưởng từ các phương thức có thẩm quyền.
4. Khử trùng lặp trạng thái ứng dụng bằng các định danh ổn định.
5. Không phát lại một mutation không an toàn chỉ vì phản hồi của nó đã bị mất.

Kế hoạch phục hồi mẫu đặt `sendLastEventId` thành false một cách rõ ràng và liệt kê các resource cần truy xuất lại.

### Ngăn chặn hiệu ứng "đàn gia súc" khi kết nối lại

Nếu 10.000 client kết nối lại chính xác tại một giây, server phục hồi sẽ thất bại lần nữa.

Sử dụng exponential backoff với jitter và giới hạn. Bài học tính toán jitter tất định từ client id và số lần thử để các bài kiểm tra vẫn có thể tái lập:

```text
attempt 0: up to 250 ms
attempt 1: up to 500 ms
attempt 2: up to 1000 ms
...
cap: 8000 ms
```

Sản xuất có thể sử dụng tính ngẫu nhiên an toàn về mật mã hoặc thời gian chạy. Bất biến ở đây là sự phân phối, không phải một công thức cụ thể.

## Xây dựng nó

`code/main.py` xây dựng năm thành phần độ tin cậy nhỏ.

### `RequestCoordinator`

- bắt đầu một request đang thực hiện với các deadline nhàn rỗi và tối đa;
- phát ra các thông báo tiến trình đơn điệu;
- tạo ra tín hiệu hủy bỏ stdio hoặc HTTP chính xác;
- bỏ qua các thông báo hủy bỏ không hợp lệ;
- làm cho các cuộc đua hủy bỏ và hoàn thành trở nên rõ ràng;
- dành riêng việc hủy bỏ do server gửi cho các đăng ký stdio.

### `MutationLedger`

- chứng minh rằng hai JSON-RPC id thực thi hai lần nếu không có khóa nghiệp vụ;
- sử dụng giao dịch SQLite dựa trên tệp để kiểm tra khóa, hiệu ứng mô phỏng, bộ đếm thực thi và cam kết kết quả;
- khử trùng lặp các đối số khớp dưới một idempotency key qua các kết nối ledger độc lập;
- từ chối một khóa được tái sử dụng với các đối số khác nhau;
- trả về các bản sao phòng thủ và bảo toàn các bản ghi đã cam kết khi mở lại.

### `DurableTaskService`

- xác nhận một yêu cầu hủy bỏ;
- giữ Task ở trạng thái `working` cho đến khi có checkpoint worker;
- chứng minh tại sao xác nhận không phải là trạng thái cuối cùng.

### `BoundedSseBuffer`

- gộp hoặc loại bỏ tiến trình khi chịu áp lực;
- ghi lại rằng cần truy xuất lại có thẩm quyền;
- không bao giờ loại bỏ phản hồi cuối cùng.

### Các trình trợ giúp phục hồi

- trả về các header SSE an toàn cho proxy và các bình luận keepalive;
- tạo kế hoạch kết nối lại và truy xuất lại;
- lan tỏa các lần thử lại với exponential backoff và jitter tất định.

## Sử dụng nó

Từ thư mục gốc của repository:

```bash
cd phases/13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/code
python3 main.py
python3 -m unittest discover tests -v
```

Bản demo chạy cả hai phía của cuộc đua trung tâm, thực thi một mutation được khử trùng lặp theo giao dịch trong một ledger tạm thời dựa trên tệp, làm quá tải bộ đệm tiến trình có giới hạn và cho thấy một Task bền vững chuyển từ hủy bỏ được xác nhận sang hủy bỏ do worker quan sát thấy.

## Phòng thí nghiệm tương tác

Chạy bốn thứ tự sự kiện mà không cần thêm sleep.

1. Bắt đầu request `A`, hủy nó, sau đó gọi `complete()`.
2. Bắt đầu request `B`, hoàn thành nó, sau đó gửi hủy bỏ.
3. Bắt đầu request `C`, phát tiến trình trước mỗi deadline nhàn rỗi, sau đó vượt qua deadline tối đa.
4. Bắt đầu request `D` qua Streamable HTTP và đóng stream phản hồi của nó.

Ghi lại cho mỗi kịch bản:

- trạng thái request cuối cùng;
- liệu có tồn tại phản hồi cuối cùng hay không;
- tín hiệu hủy bỏ được đặt trên đường truyền;
- sự kiện nào client nên bỏ qua.

Sau đó thay đổi `D` thành stdio. Thao tác giống hệt nhau, nhưng tín hiệu hủy bỏ phải thay đổi.

## Phòng thí nghiệm thực hành

Thêm một mutation `reserve_inventory` vào `MutationLedger`.

Yêu cầu:

1. Khóa liên kết SKU, số lượng, tenant và tên thao tác.
2. Một lần thử lại với cùng khóa và cùng đối số trả về đặt chỗ đầu tiên.
3. Một lần thử lại với số lượng đã thay đổi sẽ thất bại mà không có đặt chỗ khác.
4. Một lần thực thi đã cam kết nhưng mất phản hồi có thể được đối soát theo khóa.
5. Kết quả không ghi lại dữ liệu bí mật hoặc thanh toán.
6. Tự động thử lại bị vô hiệu hóa khi client không cung cấp khóa.
7. Thêm một lần ngắt đăng ký mô phỏng và truy xuất lại bản ghi hàng tồn kho trước khi quyết định làm gì tiếp theo.
8. Bắt đầu hai kết nối ledger tại một rào cản và gửi cùng một khóa đồng thời. Khẳng định một đặt chỗ đã được cam kết.
9. Thay đổi đối tượng đặt chỗ được trả về đầu tiên. Phát lại khóa và chứng minh kết quả đã lưu không thay đổi.
10. Đóng và mở lại tệp ledger, sau đó đối soát đặt chỗ theo khóa.

Hãy giữ cho phòng thí nghiệm trung thực: nếu hàng tồn kho nằm trong một dịch vụ khác, hãy giải thích liệu dịch vụ đó có chấp nhận cùng một idempotency key hay liệu một transactional outbox có bắc cầu cho cam kết cục bộ đến hiệu ứng từ xa hay không.

## Artifact được vận chuyển

`outputs/skill-mcp-reliability-reviewer.md` là một kỹ năng đánh giá độ tin cậy phẳng. Cung cấp cho nó một thao tác MCP, transport, chính sách timeout, hành vi thử lại, chính sách hàng đợi và kế hoạch phục hồi. Nó trả về một bảng đua, phân loại thử lại, ranh giới idempotency, kiểm tra kiểm soát luồng và các fixture lỗi.

## Xác minh nó

Bài học hoàn thành khi các tuyên bố này là đúng:

- Hủy bỏ stdio gửi `notifications/cancelled` và không nhận được phản hồi.
- Hủy bỏ Streamable HTTP đóng stream request và không gửi POST hủy bỏ.
- Hủy-trước-khi-hoàn-thành ngăn chặn phản hồi cuối cùng.
- Hoàn-thành-trước-khi-hủy bảo toàn phản hồi và bỏ qua việc hủy bỏ muộn.
- Tiến trình có thể reset timeout nhàn rỗi nhưng không bao giờ là timeout tối đa.
- Chỉ riêng một JSON-RPC id mới sẽ thực thi lại mutation.
- Một idempotency key và các đối số giống hệt nhau thực thi một lần dưới một cuộc đua hai kết nối đồng thời.
- Một bản ghi đã cam kết tồn tại sau khi mở lại và phát lại trả về một bản sao phòng thủ.
- Thay đổi một kết quả được trả về không thể làm thay đổi kết quả đã lưu.
- Bộ đệm có giới hạn duy trì trong dung lượng và bảo toàn phản hồi cuối cùng.
- Kết nối lại sử dụng một request mới, không gửi `Last-Event-ID` và truy xuất lại trạng thái bị ảnh hưởng.
- Xác nhận `tasks/cancel` để Task ở trạng thái không kết thúc cho đến khi worker quan sát thấy nó.

## Các chế độ thất bại trong sản xuất

| Thất bại | Triệu chứng quan sát được | Phản hồi đúng |
|---------|--------------------|------------------|
| Client HTTP POST thông báo hủy bỏ | Server và client không đồng ý về vòng đời request | Đóng stream phản hồi SSE của request |
| Server phản hồi sau khi đã chấp nhận hủy bỏ | Client nhận được kết quả muộn không thể sử dụng | Dừng công việc và ngăn chặn các thông điệp tiếp theo khi hủy bỏ thắng |
| Tiến trình reset mọi deadline | Công việc bị treo tồn tại mãi mãi | Giữ một timeout tối đa tuyệt đối riêng biệt |
| RPC id mới được coi là khử trùng lặp | Phí, triển khai hoặc xóa chạy hai lần | Thêm một idempotency key ứng dụng bền vững |
| Kiểm tra khóa và hiệu ứng tách biệt | Các worker đồng thời cùng thấy khóa bị thiếu | Cam kết yêu cầu khóa, bản ghi hiệu ứng và kết quả nguyên tử |
| Ledger trong bộ nhớ được sử dụng qua các bản sao | Khởi động lại hoặc worker khác quên các cam kết trước đó | Sử dụng lưu trữ bền vững chia sẻ hoặc idempotency thượng nguồn |
| Kết quả có thể thay đổi được lưu trữ trả về trực tiếp | Mutation của người gọi làm hỏng các lần phát lại sau này | Tuần tự hóa các kết quả đã cam kết và trả về các bản sao phòng thủ |
| Khóa được tái sử dụng với các đối số đã thay đổi | Một khóa bí danh cho hai ý định nghiệp vụ | Lưu trữ và so sánh dấu vân tay đối số |
| Hàng đợi tiến trình không giới hạn | Bộ nhớ tăng với consumer chậm | Gộp và loại bỏ tiến trình có thể thay thế trong một giới hạn |
| Phản hồi cuối cùng bị loại bỏ dưới áp lực | Client không thể biết kết quả request | Dự trữ dung lượng hoặc trục xuất tiến trình, không bao giờ là phản hồi cuối cùng |
| Proxy đệm SSE | Tiến trình đến theo đợt hoặc sau timeout | Vô hiệu hóa đệm và cấu hình timeout proxy tương thích |
| Giả định `Last-Event-ID` | Client tiếp tục từ trạng thái server không hỗ trợ | Kết nối lại với một request mới và truy xuất lại |
| Mọi client kết nối lại ngay lập tức | Phục hồi tạo ra một sự cố khác | Sử dụng exponential backoff có giới hạn với jitter |
| Xác nhận Task được coi là hủy bỏ cuối cùng | Worker tiếp tục chạy sau khi UI nói đã dừng | Poll Task cho đến khi có trạng thái kết thúc |

## Kết nối Capstone

Capstone hệ sinh thái tool nên coi độ tin cậy là bằng chứng có thể thực thi, không phải là một đoạn văn trong sơ đồ kiến trúc.

Yêu cầu các artifact sau:

- một bản ghi cuộc đua hủy bỏ cho mỗi transport;
- một bảng thử lại cho mỗi mutation được phơi bày;
- một bản ghi idempotency-key và fixture không khớp;
- một bản ghi đồng thời cùng khóa, kiểm tra mở lại và kiểm tra bí danh mutation;
- một kết quả quá tải bộ đệm có giới hạn;
- các header SSE reverse-proxy và chính sách nhàn rỗi;
- một kế hoạch kết nối lại đặt tên các phương thức truy xuất lại có thẩm quyền;
- một dấu vết hủy bỏ Task bền vững khi capstone sử dụng Tasks.

Một request xanh trong một tiến trình cục bộ chỉ chứng minh được "happy path". Capstone sẵn sàng cho sản xuất khi các phản hồi bị mất, hủy bỏ muộn, consumer chậm và đàn gia súc kết nối lại có các kết quả tất định.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa |
|------|---------|
| Hủy bỏ request | Sự từ bỏ một request MCP đang thực hiện |
| Cuộc đua hủy bỏ | Sự cạnh tranh giữa các sự kiện hoàn thành kết thúc và hủy bỏ |
| Idle timeout | Giới hạn kể từ hoạt động request hữu ích cuối cùng |
| Maximum timeout | Giới hạn tuyệt đối kể từ khi bắt đầu request, không bị ảnh hưởng bởi tiến trình |
| Idempotency key | Định danh ứng dụng khử trùng lặp một ý định nghiệp vụ |
| Atomic ledger | Ranh giới bền vững cam kết yêu cầu khóa, bản ghi hiệu ứng và kết quả như một đơn vị |
| Backpressure | Kiểm soát được áp dụng khi nhà sản xuất vượt quá người tiêu dùng |
| Gộp tiến trình | Thay thế tiến trình cũ hơn bằng một giá trị có thẩm quyền mới hơn |
| Truy xuất lại | Đọc lại trạng thái hiện tại sau một khoảng trống stream |
| Jitter | Sự thay đổi có chủ ý làm lan tỏa các lần thử lại theo thời gian |

## Đọc thêm

- [MCP Cancellation](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/cancellation)
- [MCP Progress](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/progress)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Tasks Extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)