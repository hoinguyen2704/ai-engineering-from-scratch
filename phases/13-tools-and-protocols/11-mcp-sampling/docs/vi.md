# MCP Model Input: Sampling Migration và Stateless MRTR

> MCP 2026-07-28 ngừng hỗ trợ (deprecate) Sampling cho các thiết kế mới và loại bỏ kênh yêu cầu từ server đến client. Nếu một workflow hiện có vẫn cần model của client, server sẽ trả về kết quả `input_required` và client sẽ thực hiện lại yêu cầu gốc với output từ model. Vòng lặp suy luận (reasoning loop) trở nên tường minh, có giới hạn và không trạng thái (stateless) ở tầng giao thức.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 phút

## Mục tiêu học tập

- Giải thích lý do tại sao Sampling bị ngừng hỗ trợ trong MCP 2026-07-28 và chọn mặc định tích hợp model trực tiếp cho các server mới.
- Triển khai workflow tương thích truyền tải `sampling/createMessage` thông qua Multi Round-Trip Requests (MRTR).
- Đưa phiên bản giao thức và các capability của client vào mọi đối tượng yêu cầu `_meta`.
- Trả về `resultType: "input_required"` và thực hiện lại phương thức gốc với một JSON-RPC id mới.
- Bảo vệ tính toàn vẹn của `requestState` và ràng buộc nó với principal, phương thức, đối số và thời hạn.
- Giới hạn các vòng lặp có sự hỗ trợ của model bằng kiểm tra capability, phê duyệt, xác thực phản hồi và giới hạn số vòng lặp.

## Quyết định trước khi thực hiện giao thức

Một công cụ như `summarize_repo` cần hai loại công việc:

1. Công việc tất định (deterministic): liệt kê file, đọc các file được phép, xác thực đường dẫn và tập hợp nội dung.
2. Công việc của model: chọn các file đại diện và tổng hợp tóm tắt.

Hiện bạn có hai kiến trúc hợp lệ.

### Server mới: tích hợp trực tiếp với nhà cung cấp model

Đây là mặc định hiện tại. Server sở hữu việc lựa chọn model, thông tin xác thực, ngân sách, cơ chế thử lại (retries) và khả năng quan sát (observability). Nó trả về một kết quả `tools/call` thông thường cho MCP client.

Hãy chọn cách này khi server đã là một dịch vụ được host hoặc khi hành vi model có thể dự đoán được quan trọng hơn việc sử dụng model của host.

### Workflow Sampling hiện có: di chuyển sang MRTR

Sampling vẫn tồn tại trong giai đoạn ngừng hỗ trợ. Một server nhắm đến phiên bản 2026-07-28 không thể gửi yêu cầu `sampling/createMessage` trực tiếp ngược lại cho client. Thay vào đó, nó nhúng yêu cầu đó vào một `InputRequiredResult`.

Chỉ chọn đường dẫn tương thích này khi việc sử dụng model và thông tin xác thực của client là yêu cầu thực tế của sản phẩm. Hãy ghi lại kế hoạch loại bỏ vì các triển khai mới không nên áp dụng Sampling đã bị ngừng hỗ trợ.

## Hợp đồng không trạng thái (Stateless Contract)

Giao thức tháng 7 năm 2026 không có trao đổi `initialize`, không có `notifications/initialized` và không có `Mcp-Session-Id`. Mọi yêu cầu đều mang thông tin từng tồn tại trong quá trình bắt tay (handshake):

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Server xác thực phiên bản trên mỗi yêu cầu. Phiên bản bị thiếu hoặc không phải là chuỗi sẽ dẫn đến lỗi params, `-32602`. Một chuỗi không được hỗ trợ sẽ trả về `-32022` với dữ liệu chính xác `{"supported":["2026-07-28"],"requested":"<client version>"}`. Một capability Sampling bị thiếu sẽ trả về `-32021` với `data.requiredCapabilities` được đặt thành `{"sampling":{}}`.

Một phong bì (envelope) không có JSON-RPC `id` là một thông báo (notification). Người nhận có thể xử lý nó, nhưng nó không phát ra phản hồi thành công cũng như phản hồi lỗi. Một bộ chuyển đổi Streamable HTTP trả về `202 Accepted` không có body cho một thông báo được chấp nhận.

Server cũng triển khai `server/discover` với khóa `supportedVersions` chính xác, các capability, `ttlMs` và `cacheScope` để client có thể tìm hiểu và cache hợp đồng server trước khi gọi công cụ. Vì quá trình khám phá (discovery) quảng bá `tools`, server cũng triển khai `tools/list` bắt buộc. Descriptor `summarize_repo` tất định của nó bao gồm đối tượng hợp lệ `inputSchema`, `resultType: "complete"`, metadata định danh server và các gợi ý cache công khai.

Mỗi kết quả hiện đại thành công đều có một bộ phân biệt (discriminator):

- `resultType: "complete"` nghĩa là thao tác đã hoàn tất.
- `resultType: "input_required"` nghĩa là client phải thực hiện các yêu cầu được nhúng và thử lại.
- Các phần mở rộng có thể định nghĩa thêm các loại kết quả. Phần mở rộng Tasks thêm `"task"` trong Bài 13.

## Một vòng MRTR

Server không thể gọi client trong khi đang xử lý yêu cầu. Thay vào đó, nó trả về kết quả này:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "pick_files": {
        "method": "sampling/createMessage",
        "params": {
          "messages": [
            {
              "role": "user",
              "content": {
                "type": "text",
                "text": "Choose three representative files and return a JSON array."
              }
            }
          ],
          "systemPrompt": "Return only the requested value.",
          "modelPreferences": {
            "costPriority": 0.8,
            "intelligencePriority": 0.2
          },
          "maxTokens": 400
        }
      }
    },
    "requestState": "opaque-integrity-protected-value"
  }
}
```

Client xác minh rằng nó hỗ trợ Sampling, áp dụng các chính sách phê duyệt và model của mình, sau đó nhận phản hồi từ model. Sau đó, nó gửi một yêu cầu mới với một JSON-RPC id khác:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "inputResponses": {
      "pick_files": {
        "role": "assistant",
        "content": {
          "type": "text",
          "text": "[\"README.md\", \"server.py\", \"docs/intro.md\"]"
        },
        "model": "host-model",
        "stopReason": "endTurn"
      }
    },
    "requestState": "opaque-integrity-protected-value",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}}
    }
  }
}
```

Việc thử lại không phải là sự tiếp nối của một phiên giao thức. Đó là một yêu cầu mới lặp lại phương thức và đối số gốc, chỉ thêm `inputResponses` của vòng hiện tại và lặp lại `requestState` từng byte một.

MRTR chỉ được cho phép trên `tools/call`, `prompts/get` và `resources/read`. Server không được trả về `input_required` từ các phương thức không liên quan.

## Trạng thái đa vòng (Multi-Round State)

Bài học này cần hai lần gọi model:

1. `pick_files` trả về một mảng JSON.
2. `summary` trả về văn bản cuối cùng.

Mỗi lần thử lại chỉ mang theo các phản hồi cho vòng đó. Do đó, server đưa phase và dữ liệu trung gian đã xác thực vào `requestState` tiếp theo.

Hãy coi giá trị đó là do kẻ tấn công kiểm soát. Việc ký tên vào một tên phase thô là không đủ. Hãy ràng buộc trạng thái với:

- principal đã xác thực, không phải `clientInfo` do tự báo cáo;
- phương thức khởi tạo;
- một bản tóm tắt (digest) của các đối số gốc;
- thời hạn ngắn;
- phase hiện tại và các giá trị trung gian đã xác thực.

Sử dụng HMAC khi không yêu cầu tính bảo mật. Sử dụng mã hóa xác thực (authenticated encryption) khi client không được phép đọc trạng thái. Từ chối chữ ký xấu, giá trị hết hạn, principal đã thay đổi hoặc đối số đã thay đổi với `-32602`.

Client không được phân tích hoặc sửa đổi `requestState`. Công việc duy nhất của nó là lặp lại chính xác chuỗi đó khi thử lại.

## Ưu tiên Model là các gợi ý

`costPriority`, `speedPriority` và `intelligencePriority` là các ưu tiên độc lập. Chúng không phải là một phân phối xác suất và không cần phải cộng lại bằng một. Client có thể bỏ qua chúng vì client sở hữu chính sách model.

Giữ `includeContext` ở `"none"` nếu bạn duy trì luồng Sampling cũ. Các chế độ context khác làm tăng rủi ro rò rỉ và bản thân chúng cũng đã bị ngừng hỗ trợ. Hãy truyền context tường minh tối thiểu trong yêu cầu.

## Các bất biến an toàn (Safety Invariants)

Client là ranh giới tin cậy cho các yêu cầu Sampling được nhúng.

- Hiển thị cho người dùng biết server đang yêu cầu model làm gì khi chính sách yêu cầu phê duyệt.
- Giới hạn các vòng MRTR. Nếu không, một server độc hại có thể tạo ra vòng lặp tiêu tốn model.
- Xác thực mọi phản hồi sampling trước khi sử dụng nó làm tên file, URL hoặc đầu vào công cụ.
- Giới hạn byte và token cho mỗi vòng.
- Từ chối một yêu cầu đầu vào không được khai báo trong các capability hiện tại của client.
- Giữ output của model nằm ngoài các quyết định ủy quyền.
- Ghi log phương thức khởi tạo và khóa yêu cầu đầu vào mà không ghi log nội dung prompt nhạy cảm.

`clientInfo` và `serverInfo` là metadata hiển thị và chẩn đoán. Không bao giờ sử dụng cả hai làm định danh đã xác thực.

```figure
t3-sampling-flip
```

## Xây dựng

`code/main.py` triển khai luồng hai vòng đầy đủ mà không cần gói của bên thứ ba:

- `server/discover` trả về `supportedVersions`, quảng bá hỗ trợ công cụ và trả về các gợi ý cache.
- `tools/list` trả về một descriptor `summarize_repo` tất định, có thể cache với schema đầu vào đối tượng.
- `tools/call` xác thực metadata cho mỗi yêu cầu.
- Kết quả đầu tiên nhúng `sampling/createMessage` để chọn file.
- Lần thử lại đầu tiên xác thực kết quả model và nhúng yêu cầu thứ hai.
- `requestState` được bảo vệ bằng HMAC mang phase giữa các yêu cầu độc lập.
- Kết quả cuối cùng sử dụng `resultType: "complete"`.

Model host giả lập làm cho ví dụ trở nên tất định. Chỉ thay thế `fake_host_model` khi kết nối với host thực. Máy trạng thái phía server nên giữ tính tất định và có thể kiểm thử.

## Sử dụng

Từ thư mục gốc của repository:

```bash
cd phases/13-tools-and-protocols/11-mcp-sampling/code
python3 main.py
python3 -m unittest discover tests -v
```

Các checkpoint dự kiến:

- Discovery trả về kết quả đầy đủ với `ttlMs` và `cacheScope`.
- Khám phá công cụ trả về cùng một descriptor đã sắp xếp với `resultType`, định danh server và gợi ý cache.
- Các capability bị thiếu và phiên bản không được hỗ trợ sử dụng dữ liệu lỗi `-32021` và `-32022` chính xác.
- Một thông báo không có id không tạo ra phản hồi JSON-RPC.
- Các id yêu cầu là `[1, 2, 3]`, chứng minh mỗi vòng MRTR là độc lập.
- Hai kết quả đầu tiên là `input_required`.
- Kết quả cuối cùng là `complete` và chứa các file đã chọn cộng với tóm tắt.
- Thay đổi các đối số gốc khi thử lại sẽ làm thất bại kiểm tra trạng thái yêu cầu.

## Phát hành

`outputs/skill-sampling-loop-designer.md` hiện là một công cụ lập kế hoạch di chuyển. Trước tiên, nó quyết định xem Sampling có nên bị loại bỏ để chuyển sang tích hợp model trực tiếp hay không. Nếu yêu cầu tính tương thích, nó tạo ra các vòng MRTR, ràng buộc trạng thái, cổng capability, ngân sách, xác thực và kế hoạch loại bỏ.

## Bài tập

1. Thay đổi phản hồi chọn file thành JSON không hợp lệ. Xác nhận server trả về `-32602` thay vì tin tưởng output của model.
2. Thay đổi `audience` giữa lần gọi đầu tiên và lần thử lại. Giải thích tại sao trạng thái đã niêm phong (sealed state) chặn việc tái sử dụng giữa các yêu cầu.
3. Thêm vòng thứ ba yêu cầu host phê bình bản tóm tắt. Mang bản tóm tắt trước đó bên trong trạng thái đã ký và giới hạn toàn bộ luồng ở ba vòng.
4. Loại bỏ Sampling bằng cách thay thế callback host giả bằng một bộ chuyển đổi model do server sở hữu. Liệt kê những trách nhiệm về phê duyệt, thanh toán và khả năng quan sát nào chuyển sang server.
5. Thêm kiểm tra thời hạn sử dụng một giá trị trạng thái đã quá hạn một giây.

## Thuật ngữ chính

| Thuật ngữ | Ý nghĩa trong 2026-07-28 |
|------|------------------------|
| Sampling | Tính năng đã ngừng hỗ trợ yêu cầu model của client thực hiện hoàn thiện |
| MRTR | Mô hình thử lại không trạng thái cho đầu vào client cần thiết trong một yêu cầu |
| `InputRequiredResult` | Kết quả với `resultType: "input_required"` |
| `inputRequests` | Bản đồ do server chỉ định về các yêu cầu elicitation, sampling hoặc roots được nhúng |
| `inputResponses` | Kết quả client của vòng hiện tại được khóa như `inputRequests` |
| `requestState` | Trạng thái server mờ (opaque) được client lặp lại chính xác và server xác minh |
| `resultType` | Bộ phân biệt bắt buộc cho các kết quả MCP hiện đại |
| Direct model integration | Thay thế được khuyến nghị cho các server mới cần suy luận model |
| Capability gate | Quy tắc ngăn chặn gửi yêu cầu nhúng mà client không quảng bá |
| Loop budget | Số vòng, token, byte, thời gian và chi phí tối đa cho phép cho thao tác |

## Tương thích kế thừa

Một client bị ghim ở phiên bản 2025-11-25 vẫn có thể sử dụng luồng `sampling/createMessage` do server khởi tạo cũ hơn qua kết nối trực tiếp. Chỉ giữ hành vi đó trong một bộ chuyển đổi dành riêng cho phiên bản. Đừng biến đường dẫn có phiên (sessionful) thành kiến trúc cho server 2026-07-28.

Các SDK chính thức có thể dịch các trình xử lý `input_required` hiện đại cho các peer cũ hơn. Shim đó là một ranh giới tương thích, không phải là sự cho phép để thêm logic phụ thuộc vào phiên mới.

## Đọc thêm

- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP Sampling deprecation](https://modelcontextprotocol.io/seps/2577-deprecate-roots-sampling-and-logging)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)