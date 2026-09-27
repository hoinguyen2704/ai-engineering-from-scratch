# Hợp đồng Công cụ MCP và Nội dung

> Một công cụ chỉ an toàn để tự động hóa khi việc khám phá, đối số, kết quả, phân trang và siêu dữ liệu truyền tải cùng đồng ý về một hợp đồng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 13, Bài học 07, 09 và 10
**Time:** ~120 phút

## Mục tiêu học tập

- Định nghĩa đầu vào và đầu ra của công cụ bằng JSON Schema 2020-12.
- Xác thực các kết quả có cấu trúc mà không giả định chúng là các đối tượng JSON.
- Lựa chọn giữa văn bản, hình ảnh, âm thanh, liên kết tài nguyên và tài nguyên nhúng.
- Từ chối các định nghĩa `x-mcp-header` không an toàn trước khi công cụ đến được mô hình.
- Mã hóa các giá trị tiêu đề tham số và xác minh sự tương đương chính xác giữa tiêu đề và nội dung.
- Duyệt qua phân trang con trỏ mà không cần diễn giải các giá trị con trỏ.
- Giới hạn và ủy quyền các gợi ý `completion/complete`.

## Vấn đề

Việc gọi một hàm Python rất dễ dàng. Việc gọi một khả năng từ xa thông qua một AI host là một vấn đề về hợp đồng.

Máy chủ xuất bản một bộ mô tả. Máy khách biến bộ mô tả đó thành ngữ cảnh mô hình và giao diện người dùng. Mô hình tạo ra các đối số. Một cổng (gateway) có thể định tuyến yêu cầu từ các tiêu đề được phản chiếu. Máy chủ thực thi công cụ. Sau đó, máy khách quyết định xem kết quả có đủ an toàn và hợp lệ để trả về cho mô hình hay không.

Một ranh giới yếu sẽ làm hỏng toàn bộ chuỗi.

Hãy xem xét năm lỗi sau:

- Bộ mô tả nói kết quả là một đối tượng, nhưng máy chủ trả về một mảng.
- Máy khách dừng phân trang khi `nextCursor` là một chuỗi trống.
- Một tham số token được phản chiếu vào tiêu đề HTTP và trở nên hiển thị với các bên trung gian.
- Một giá trị định tuyến Unicode được gửi dưới dạng tiêu đề thô, sau đó cổng và nguồn diễn giải các byte khác nhau.
- Một điểm cuối hoàn thiện (completion endpoint) gợi ý một môi trường sản xuất cho một người gọi không thể truy cập nó.

Không lỗi nào trong số này được khắc phục bằng cách nhắc lệnh (prompting) tốt hơn. Chúng đòi hỏi các hợp đồng giao thức và ứng dụng rõ ràng.

## Quy trình Hợp đồng

Hãy coi mỗi lần gọi công cụ là năm cổng:

1. **Khám phá (Discover).** Đọc danh sách công cụ có tính xác định và được phân trang.
2. **Tiếp nhận (Admit).** Xác thực từng bộ mô tả và áp dụng chính sách bảo mật cục bộ.
3. **Gọi (Invoke).** Xác thực các đối số và xây dựng siêu dữ liệu truyền tải.
4. **Thực thi (Execute).** Chạy trình xử lý và phân loại lỗi một cách chính xác.
5. **Tiêu thụ (Consume).** Xác thực các khối nội dung và đầu ra có cấu trúc trước khi mô hình sử dụng.

```figure
mcp-contract-pipeline
```

Host sở hữu các cổng tiếp nhận và tiêu thụ. Máy chủ không thể buộc máy khách tin tưởng vào các chú thích, lược đồ hoặc đầu ra của nó.

## JSON Schema là một ranh giới thời gian chạy

Trong MCP `2026-07-28`, `inputSchema` và `outputSchema` sử dụng JSON Schema. Khi `$schema` vắng mặt, phương ngữ mặc định là 2020-12.

Lược đồ đầu vào phải là một đối tượng lược đồ. Một công cụ không có đối số vẫn nên nói rõ những gì nó chấp nhận:

```json
{
  "type": "object",
  "additionalProperties": false
}
```

Điều này nghiêm ngặt hơn `{ "type": "object" }`, vốn chấp nhận các thuộc tính tùy ý.

Lược đồ đầu ra là tùy chọn. Khi một máy chủ xuất bản một lược đồ, mọi kết quả công cụ hoàn chỉnh đều cam kết trả về `structuredContent` phù hợp, bao gồm cả các kết quả có `isError: true`. Cờ lỗi phân loại kết quả thực thi; nó không miễn trừ hợp đồng đầu ra đã xuất bản. Máy khách nên xác thực kết quả thay vì tin tưởng vào bộ mô tả.

### Nội dung có cấu trúc là bất kỳ giá trị JSON nào

Đừng mã hóa cứng `structuredContent` dưới dạng từ điển. Nó có thể là:

- một đối tượng;
- một mảng;
- một chuỗi;
- một số;
- một boolean;
- `null`.

Công cụ này trả về một mảng:

```json
{
  "name": "tag_catalog",
  "inputSchema": {
    "type": "object",
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "array",
    "items": {"type": "string"}
  }
}
```

Kết quả thành công của nó là hợp lệ:

```json
{
  "resultType": "complete",
  "content": [
    {
      "type": "text",
      "text": "[\"contracts\", \"mcp\", \"stateless\"]"
    }
  ],
  "structuredContent": ["contracts", "mcp", "stateless"],
  "isError": false
}
```

Để tương thích, các kết quả có cấu trúc cũng nên bao gồm JSON được tuần tự hóa trong một khối văn bản. Văn bản không phải là nguồn xác thực. `structuredContent` mới là nguồn xác thực.

### Một trình xác thực nhỏ vẫn dạy về ranh giới

Bài học sử dụng một tập hợp con JSON Schema có chủ đích vì nó nằm trong thư viện tiêu chuẩn của Python. Nó kiểm tra các cơ chế được sử dụng bởi các công cụ mẫu:

- các kiểu object, array, string, integer, number, boolean và null;
- các thuộc tính bắt buộc;
- `additionalProperties: false`;
- các mục trong mảng;
- các giá trị enum;
- độ dài chuỗi tối thiểu.

Đây không phải là sự thay thế cho một trình xác thực sản xuất hoàn chỉnh. Bài học có thể tái sử dụng là nơi diễn ra việc xác thực: sau khi khám phá đối với các bộ mô tả, trước khi thực thi đối với các đối số và trước khi tiêu thụ đối với các kết quả có cấu trúc.

## Các khối nội dung mang các chi phí khác nhau

Mảng `content` có thể kết hợp nhiều loại nội dung.

| Loại | Sử dụng cho | Ranh giới chính |
|------|------------|---------------|
| `text` | Tóm tắt dễ đọc cho con người và mô hình | Coi văn bản là đầu ra không đáng tin cậy |
| `image` | Bằng chứng hình ảnh được mã hóa dưới dạng base64 | Xác thực loại phương tiện và kích thước |
| `audio` | Đầu ra được nói hoặc ghi âm được mã hóa dưới dạng base64 | Xác thực loại phương tiện và giới hạn thời lượng |
| `resource_link` | Một URI mà máy khách có thể tìm nạp sau này | Ủy quyền lại việc đọc tài nguyên sau đó |
| `resource` | Dữ liệu được nhúng trực tiếp vào kết quả | Thực thi giới hạn tải trọng và nội dung ngay bây giờ |

Một liên kết tài nguyên không phải là bằng chứng cho thấy tài nguyên xuất hiện trong `resources/list`. Đó là một tham chiếu được trả về bởi lệnh gọi công cụ này. Máy khách vẫn áp dụng chính sách tài nguyên của mình khi nó theo dõi URI.

Một tài nguyên nhúng tránh được một chuyến khứ hồi khác nhưng làm tăng kích thước phản hồi hiện tại. Sử dụng liên kết cho các tệp lớn hoặc các tệp thay đổi độc lập. Sử dụng tài nguyên nhúng cho các bằng chứng nhỏ phải di chuyển cùng với kết quả.

Kết quả `evidence_bundle` của bài học bao gồm cả năm loại. Máy khách xác thực từng khối trước khi chấp nhận kết quả.

## `x-mcp-header` là siêu dữ liệu định tuyến

Một thuộc tính bên trong `inputSchema` có thể khai báo `x-mcp-header`. Qua Streamable HTTP, máy khách phản chiếu đối số đó vào `Mcp-Param-{name}`.

```json
{
  "region": {
    "type": "string",
    "x-mcp-header": "Region"
  }
}
```

Với `region: "eu-west"`, phương tiện truyền tải có thể phát ra:

```http
Mcp-Param-Region: eu-west
```

Chú thích tồn tại để bộ cân bằng tải, cổng hoặc công cụ chính sách có thể định tuyến mà không cần phân tích cú pháp nội dung JSON. Đây không phải là nơi để đặt thông tin xác thực.

Giao thức hạn chế chú thích:

- tên tiêu đề không trống và tuân theo cú pháp token tên trường HTTP;
- tên tiêu đề là duy nhất không phân biệt chữ hoa chữ thường;
- kiểu thuộc tính là chuỗi, số nguyên hoặc boolean;
- `number` không được phép;
- chú thích chỉ xuất hiện trên một thành viên trực tiếp của `inputSchema.properties`;
- các giá trị số nguyên nằm trong khoảng từ `-9007199254740991` đến `9007199254740991`.

Quy tắc vị trí là cú pháp và đóng khi lỗi (fail-closed). Hãy duyệt toàn bộ cây lược đồ, không chỉ các thuộc tính mà trình xác thực của bạn tình cờ hiểu được. Từ chối một chú thích nằm dưới `properties` của một đối tượng lồng nhau, một nhánh `oneOf`, `items`, một định nghĩa đạt được bằng `$ref` hoặc bất kỳ lược đồ đầu ra nào. Việc giải quyết một tham chiếu không biến nút được tham chiếu thành một thuộc tính cấp cao nhất trực tiếp.

Bài học này thêm một chính sách triển khai: từ chối các bộ mô tả phản chiếu các tên như `password`, `secret`, `token`, `api_key` hoặc `authorization`. Đặc tả chính thức khuyên các tác giả máy chủ không nên phản chiếu các tham số nhạy cảm. Máy khách có thể biến lời khuyên đó thành một quy tắc tiếp nhận cứng.

Kiểm tra tên tiêu đề, không phải giá trị của nó. Mã mẫu ghi lại `Mcp-Param-Region` trong khi giữ `eu-west` bên ngoài sự kiện kiểm toán.

### Mã hóa các giá trị trước khi xây dựng tiêu đề HTTP

Một giá trị tham số có thể di chuyển dưới dạng văn bản thuần túy chỉ khi nó là một chuỗi không trống gồm các ký tự ASCII hiển thị từ `!` đến `~` và không giống với sentinel mã hóa. Mọi thứ khác sử dụng dạng chính xác này:

```text
=?base64?{Base64UTF8}?=
```

`Base64UTF8` là base64 tiêu chuẩn trên các byte UTF-8 chính xác. Không cắt tỉa, chuẩn hóa hoặc thay thế giá trị trước. Mã hóa Unicode, chuỗi trống, dấu cách, tab, ký tự điều khiển, CR hoặc LF, khoảng trắng ở đầu hoặc cuối và bất kỳ giá trị nào bắt đầu bằng `=?base64?`. Việc mã hóa lại một giá trị trông giống sentinel là điều cho phép người nhận khôi phục văn bản gốc theo nghĩa đen thay vì giải mã nó dưới dạng cú pháp truyền tải.

Các giá trị Boolean hiển thị dưới dạng `true` hoặc `false` viết thường. Các số nguyên hiển thị ở cơ số 10 và phải nằm trong phạm vi số nguyên an toàn của JavaScript. Các giá trị nằm ngoài phạm vi đó sẽ bị từ chối thay vì bị làm tròn bởi một bên trung gian.

### Máy chủ kiểm tra bản sao được phản chiếu

Việc tạo tiêu đề chỉ là một nửa công việc của máy khách. Tại ranh giới Streamable HTTP, máy chủ phải:

1. tìm các tên `Mcp-Param-*` được công nhận mà không phân biệt chữ hoa chữ thường của tên tiêu đề;
2. giải mã dạng sentinel base64 chính xác khi có mặt;
3. so sánh văn bản đã giải mã với đối số nội dung JSON tương ứng một cách chính xác;
4. từ chối một tiêu đề được công nhận bị thiếu, trùng lặp, không mong đợi, sai định dạng hoặc không khớp trước khi gửi đi.

Việc từ chối là HTTP `400` với mã lỗi JSON-RPC `-32020`. Cả giá trị nội dung lẫn dạng tiêu đề được mã hóa của nó đều không thuộc về bản ghi kiểm toán. Chỉ ghi lại tên tiêu đề được công nhận và danh mục từ chối.

`code/main.py` mô hình hóa ranh giới này một cách trực tiếp. [Bài học 09](../../09-mcp-transports/) bao gồm thứ tự xác thực Streamable HTTP rộng hơn, bao gồm cả phương thức và sự tương đương phiên bản giao thức.

## Các con trỏ phân trang là mờ đục (Opaque)

Các thao tác liệt kê MCP sử dụng phân trang con trỏ. Máy chủ chọn kích thước trang và định dạng con trỏ. Máy khách nhận được một quyết định:

```python
if result.get("nextCursor") is None:
    break
cursor = result["nextCursor"]
```

Đừng viết thế này:

```python
if not result.get("nextCursor"):
    break
```

Một chuỗi trống là một con trỏ hợp lệ. Tính đúng đắn (truthiness) sẽ dừng lại quá sớm.

Máy khách không được giải mã con trỏ, tăng nó, so sánh nó với một con trỏ trước đó để sắp xếp hoặc suy ra số trang. Máy chủ có thể ký một con trỏ, ràng buộc nó với một phiên bản danh mục hoặc ánh xạ nó vào trạng thái riêng tư. Đó là chi tiết triển khai của máy chủ.

Máy chủ mẫu cố tình trả về `""` sau trang đầu tiên. Máy khách phải gửi chính xác giá trị đó trong yêu cầu thứ hai. Dấu vết của nó là:

```text
<first request with no cursor>
<second request with cursor "">
```

Các con trỏ không hợp lệ tạo ra JSON-RPC invalid params, mã `-32602`.

## Hoàn thiện (Completion) là một bề mặt ủy quyền

`completion/complete` cung cấp các gợi ý cho các đối số nhắc lệnh và đối số mẫu tài nguyên. Nó hữu ích cho các biểu mẫu tương tác, nhưng nó có thể làm rò rỉ các tên mà các phương thức liệt kê thông thường bảo vệ.

Một yêu cầu hoàn thiện đặt tên cho một tham chiếu và đối số đang được hoàn thiện:

```json
{
  "method": "completion/complete",
  "params": {
    "ref": {
      "type": "ref/prompt",
      "name": "deployment_review"
    },
    "argument": {
      "name": "environment",
      "value": "st"
    }
  }
}
```

Kết quả trả về tối đa 100 giá trị và có thể báo cáo `total` cộng với `hasMore`.

Áp dụng cùng ranh giới ủy quyền được sử dụng bởi lời nhắc hoặc tài nguyên được tham chiếu. Một nhà phân tích trong mẫu nhận được `development` và `staging`. Chỉ một người vận hành mới có thể nhận được `production`.

Hoàn thiện sản xuất cũng cần:

- xác thực đầu vào;
- lọc nhận thức người gọi;
- khử nhiễu yêu cầu (debouncing) trong máy khách;
- giới hạn tốc độ trong máy chủ;
- số lượng kết quả bị giới hạn;
- nhật ký không làm lộ các giá trị gợi ý nhạy cảm.

Hoàn thiện là sự hỗ trợ, không phải là bỏ qua khám phá.

## Hai lớp lỗi

Giữ các lỗi giao thức tách biệt với các lỗi thực thi công cụ.

Sử dụng lỗi JSON-RPC khi yêu cầu MCP không thể được gửi đi một cách chính xác:

- tên công cụ không xác định;
- hình dạng yêu cầu sai định dạng;
- thiếu siêu dữ liệu yêu cầu;
- con trỏ không hợp lệ.

Sử dụng kết quả công cụ hoàn chỉnh với `isError: true` khi lệnh gọi đã đến được công cụ và công cụ báo cáo một lỗi có thể hành động:

- nguồn báo cáo không khả dụng;
- ngày nằm ngoài phạm vi được hỗ trợ;
- quy tắc kinh doanh từ chối thao tác được yêu cầu.

Các mô hình thường có thể sửa lỗi thực thi công cụ. Chúng không thể sửa một máy chủ vi phạm lược đồ đầu ra của chính nó.

Nếu công cụ khai báo một lược đồ đầu ra, hãy mô hình hóa một lỗi có thể hành động bên trong lược đồ đó. Lỗi `route_report` mẫu trả về khu vực được yêu cầu của nó với `accepted: false`, cùng với văn bản lỗi dễ đọc và `isError: true`.

## Xây dựng nó

`code/main.py` xây dựng cả hai phía của ranh giới với thư viện tiêu chuẩn Python.

Máy chủ triển khai:

- xác thực siêu dữ liệu MCP cho mỗi yêu cầu;
- `server/discover` với các khả năng công cụ và hoàn thiện;
- phân trang `tools/list` có tính xác định;
- bốn bộ mô tả công cụ, bao gồm một bộ phải bị từ chối;
- đầu ra có cấu trúc mảng;
- mọi loại khối nội dung công cụ hiện tại;
- cổng tương đương Streamable HTTP giải mã các tiêu đề tham số được công nhận và trả về HTTP `400` cộng với JSON-RPC `-32020` khi không khớp;
- hoàn thiện được ủy quyền và giới hạn tốc độ.

Máy khách triển khai:

- tiếp nhận bộ mô tả;
- xác thực vị trí `x-mcp-header` toàn cây và chính sách trường nhạy cảm;
- mã hóa giá trị ASCII hiển thị thuần túy hoặc UTF-8 base64 chính xác;
- vòng lặp con trỏ mờ đục theo sau một chuỗi trống;
- xác thực đối số và kết quả;
- xác thực khối nội dung;
- các sự kiện kiểm toán tiêu đề chứa tên nhưng không chứa giá trị.

Bộ mô tả cố tình không an toàn là dữ liệu giảng dạy. Nó chứng minh rằng một công cụ bị từ chối không ngăn cản các công cụ hợp lệ tải.

## Sử dụng nó

Từ thư mục gốc của kho lưu trữ:

```bash
cd phases/13-tools-and-protocols/28-mcp-tool-contracts-and-content/code
python3 main.py
python3 -m unittest discover tests -v
```

Bản demo in ra các công cụ được tiếp nhận, bộ mô tả bị từ chối, cả hai yêu cầu phân trang, nội dung mảng có cấu trúc, các loại khối nội dung, tên tiêu đề được phản chiếu, liệu giá trị có yêu cầu mã hóa hay không, trạng thái tương đương HTTP và các giá trị hoàn thiện được lọc theo người gọi.

## Phòng thí nghiệm tương tác

Mở `code/main.py` và định vị `TOOLS`.

1. Thay đổi `tag_catalog.outputSchema.type` từ `array` thành `object`.
2. Chạy bản demo. Máy khách sẽ từ chối mảng được trả về.
3. Khôi phục lược đồ.
4. Giữ `nextCursor` của trang đầu tiên là `""`, sau đó làm cho trang cuối cùng trả về `nextCursor: None` thay vì bỏ qua trường này.
5. Chạy các bài kiểm tra và so sánh dấu vết con trỏ.
6. Thêm `x-mcp-header: "Authorization"` vào một thuộc tính chuỗi.
7. Xác nhận việc tiếp nhận bộ mô tả từ chối nó trước khi gọi.
8. Thử các giá trị `region` chứa Unicode, ký tự xuống dòng, dấu cách xung quanh và văn bản theo nghĩa đen `=?base64?SGVsbG8=?=`. Giải mã từng tiêu đề được phát ra và chứng minh giá trị gốc tồn tại chính xác.
9. Di chuyển chú thích dưới `oneOf`, `items` hoặc một định nghĩa `$ref`. Xác nhận từng bộ mô tả bị từ chối ngay cả khi nhánh đó không bao giờ được bản demo sử dụng.
10. Xóa tiêu đề được công nhận hoặc thay đổi giá trị đã giải mã của nó. Xác nhận ranh giới HTTP trả về trạng thái `400` và mã JSON-RPC `-32020`.

Điểm mấu chốt không phải là ghi nhớ một hình dạng JSON. Đó là quan sát từng cổng thất bại tại ranh giới sở hữu nó.

## Phòng thí nghiệm thực hành

Mở rộng phòng thí nghiệm hợp đồng với một công cụ `search_evidence`.

Yêu cầu:

1. Lược đồ đầu vào của nó chấp nhận `query`, `limit` và một trường định tuyến `region` an toàn.
2. Lược đồ đầu ra của nó là một mảng các đối tượng với `uri`, `title` và `score`.
3. Kết quả bao gồm văn bản tương thích và một liên kết tài nguyên cho mỗi mục.
4. Các đối số từ chối các thuộc tính không xác định.
5. `limit` bị giới hạn bởi xác thực ứng dụng.
6. Một người gọi không có quyền truy cập vào một URI sẽ không bao giờ thấy URI đó thông qua hoàn thiện hoặc đầu ra công cụ.
7. Các bài kiểm tra bao gồm điểm số không phù hợp, chú thích tiêu đề không hợp lệ và danh sách hai trang.
8. Các bài kiểm tra giá trị tiêu đề bao gồm ASCII hiển thị, Unicode, ký tự điều khiển, khoảng trắng, văn bản trông giống sentinel và cả hai giới hạn số nguyên an toàn của JavaScript.
9. Thiết bị cố định HTTP chấp nhận tên tiêu đề không phân biệt chữ hoa chữ thường nhưng từ chối các giá trị được công nhận bị thiếu hoặc không khớp với trạng thái `400` và mã `-32020`.

## Sản phẩm được vận chuyển

`outputs/skill-mcp-contract-reviewer.md` là một kỹ năng đánh giá phẳng, có thể tái sử dụng. Cung cấp cho nó một bộ mô tả công cụ, kết quả mẫu, hành vi phân trang và chính sách hoàn thiện. Nó trả về một quyết định tiếp nhận, kế hoạch xác thực kết quả, chính sách tiêu đề và các bài kiểm tra lỗi cụ thể.

## Xác minh nó

Bài học hoàn tất khi các tuyên bố sau là đúng:

- `tools/list` trả về cùng một thứ tự logic trong các lần gọi lặp lại.
- Máy khách thực hiện yêu cầu thứ hai khi `nextCursor` là `""`.
- Bộ mô tả tiêu đề nhạy cảm không an toàn bị loại trừ trong khi các công cụ khác vẫn khả dụng.
- Một mảng vượt qua lược đồ đầu ra mảng của nó.
- Một đối tượng thất bại với cùng lược đồ mảng đó.
- Các kết quả lỗi không thể bỏ qua hoặc vi phạm lược đồ đầu ra đã xuất bản.
- Các khối văn bản, hình ảnh, âm thanh, liên kết tài nguyên và tài nguyên nhúng được xác thực.
- Các sự kiện kiểm toán tiêu đề chứa tên và không chứa giá trị.
- ASCII hiển thị thuần túy vẫn là thuần túy; Unicode, điều khiển, đệm, trống và các giá trị trông giống sentinel quay vòng thông qua mã hóa UTF-8 base64 chính xác.
- Các số nguyên được phản chiếu nằm ngoài phạm vi an toàn của JavaScript bị từ chối.
- Các chú thích dưới `oneOf`, `items`, đối tượng lồng nhau, định nghĩa `$ref` hoặc lược đồ đầu ra bị từ chối trong quá trình tiếp nhận.
- Tên tiêu đề được công nhận không phân biệt chữ hoa chữ thường chỉ vượt qua khi giá trị đã giải mã khớp chính xác với nội dung; các bản sao bị thiếu hoặc không khớp tạo ra HTTP `400` và JSON-RPC `-32020`.
- Hoàn thiện nhà phân tích không bao giờ trả về `production`.
- Một lỗi công cụ sử dụng `isError: true`; một lệnh gọi giao thức sai định dạng sử dụng JSON-RPC `error`.

## Các chế độ lỗi sản xuất

| Lỗi | Những gì người học thấy | Phản hồi chính xác |
|---------|-----------------------|------------------|
| Máy khách giả định đầu ra đối tượng | Các mảng hợp lệ thất bại hoặc bị bao bọc âm thầm | Xác thực dựa trên lược đồ đã xuất bản mà không có các kiểu chỉ đối tượng |
| Con trỏ trống được coi là false | Các trang cuối biến mất | Tiếp tục bất cứ khi nào `nextCursor` có mặt và không null |
| Giá trị nhạy cảm được phản chiếu | Bí mật xuất hiện trong proxy, WAF hoặc dữ liệu dấu vết | Từ chối bộ mô tả và giữ bí mật trong dữ liệu yêu cầu được bảo vệ |
| Unicode hoặc khoảng trắng thô được phản chiếu | Cổng và nguồn không đồng ý hoặc giá trị bị chuẩn hóa | Sử dụng mã hóa sentinel UTF-8 base64 chính xác và so sánh sau khi giải mã |
| Chú thích ẩn trong một nhánh lược đồ | Máy khách bỏ lỡ siêu dữ liệu định tuyến trong khi tiếp nhận | Duyệt toàn bộ cây lược đồ và chỉ cho phép các thuộc tính cấp cao nhất trực tiếp |
| Số nguyên lớn được phản chiếu | Bên trung gian JavaScript làm tròn giá trị định tuyến | Từ chối các giá trị nằm ngoài phạm vi số nguyên an toàn của JavaScript |
| Tiêu đề và nội dung không khớp | Cổng định tuyến một mục tiêu trong khi nguồn thực thi mục tiêu khác | Từ chối trước khi gửi đi với HTTP `400` và JSON-RPC `-32020` |
| Lược đồ đầu ra bị bỏ qua | Mã hạ nguồn tiêu thụ cấu trúc bị hỏng | Xác thực trước khi mô hình hoặc ứng dụng sử dụng |
| Liên kết tài nguyên được tin tưởng tự động | Người gọi theo dõi một URI không được ủy quyền | Ủy quyền lại mọi lần đọc tài nguyên |
| Hoàn thiện chia sẻ các gợi ý toàn cầu | Tên người thuê bị ẩn rò rỉ | Lọc theo người gọi, tham chiếu và ủy quyền |
| Chú thích công cụ được coi là chính sách | Thao tác phá hoại bỏ qua xác nhận | Thực thi ủy quyền và phê duyệt bên ngoài chú thích |
| Một công cụ sai định dạng làm hỏng khám phá | Toàn bộ máy chủ trở nên không khả dụng | Từ chối bộ mô tả xấu và tiếp nhận các công cụ hợp lệ một cách độc lập |

## Kết nối Capstone

Capstone Giai đoạn 13 cần một cổng có thể hợp nhất các công cụ từ nhiều máy chủ. Bài học này cung cấp cốt lõi tiếp nhận của nó.

Sử dụng sản phẩm để chấm điểm bốn phần bằng chứng capstone:

- khám phá phân trang có tính xác định và đầy đủ;
- xác thực bộ mô tả trước khi phơi bày mô hình;
- đầu ra có cấu trúc được xác thực cộng với các khối nội dung bị giới hạn;
- siêu dữ liệu hoàn thiện và định tuyến bảo tồn các ranh giới ủy quyền.

Đừng tuyên bố khả năng tương thích cổng chỉ từ một `tools/call` thành công. Ghi lại bộ mô tả, dấu vết trang, tập hợp công cụ được tiếp nhận, tập hợp công cụ bị từ chối và một kết quả được xác thực.

## Các thuật ngữ chính

| Thuật ngữ | Ý nghĩa |
|------|---------|
| `inputSchema` | Đối tượng JSON Schema định nghĩa các đối số công cụ được chấp nhận |
| `outputSchema` | JSON Schema tùy chọn định nghĩa `structuredContent` |
| `structuredContent` | Bất kỳ giá trị JSON nào được tạo ra bởi một kết quả công cụ |
| Khối nội dung | Văn bản, hình ảnh, âm thanh, liên kết tài nguyên hoặc tài nguyên nhúng được định kiểu |
| `x-mcp-header` | Chú thích lược đồ phản chiếu một đối số nguyên thủy vào siêu dữ liệu Streamable HTTP |
| Con trỏ mờ đục | Token phân trang do máy chủ cấp mà máy khách không diễn giải giá trị |
| Tham chiếu hoàn thiện | Tên lời nhắc hoặc URI/mẫu tài nguyên có đối số đang được hoàn thiện |
| Tiếp nhận (Admission) | Quyết định của máy khách để phơi bày hoặc từ chối một bộ mô tả được khám phá |

## Đọc thêm

- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Completion](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/completion)
- [MCP Pagination](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination)
- [MCP Streamable HTTP Parameter Headers](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#custom-headers-from-tool-parameters)