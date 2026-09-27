# Chuỗi cung ứng Registry MCP: Tiếp nhận (Admission), Trôi dạt (Drift) và Hoàn tác (Rollback)

> Một mục nhập trong registry cho bạn biết những gì nhà phát hành đã khai báo. Việc tiếp nhận vào môi trường production chứng minh những gì bạn đã tải về, những gì bạn quan sát được, những gì bạn đã phê duyệt và những gì bạn có thể khôi phục một cách an toàn.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 17 (gateways và registries), Phase 13 · 18 (xác thực production)
**Time:** ~90 phút

## Mục tiêu học tập

- Phân tách việc xuất bản lên Registry, nguồn gốc gói (package provenance), khám phá runtime và phê duyệt cục bộ.
- Xác minh namespace của MCP server mà không cần tin tưởng vào tên nằm trong chính bản ghi của nó.
- Ghim (pin) các bằng chứng về xuất bản bất biến, nguồn thực thi, nguồn gốc và mô tả trạng thái trực tiếp.
- Phát hiện các thay đổi trạng thái registry và sự trôi dạt runtime sau khi tiếp nhận.
- Hoàn tác (rollback) định tuyến về phiên bản đã được tiếp nhận trước đó mà không cần viết lại lịch sử.
- Duy trì sổ cái tiếp nhận (admission ledger) chống giả mạo, giải thích cho mọi quyết định.

## Vấn đề

Bạn tìm thấy `com.example/inventory` trong một registry. Mô tả của nó có vẻ đúng. Gói của nó tồn tại. Server phản hồi `server/discover`.

Đó không phải là một sự thật duy nhất. Đó là một chuỗi các sự thật từ các cơ quan có thẩm quyền khác nhau:

1. Một nhà phát hành đã xác thực cho một namespace đã gửi một bản ghi.
2. Một registry gói đã phục vụ một artifact với định danh và digest cụ thể.
3. Một endpoint đang chạy đã báo cáo phiên bản giao thức, các khả năng, công cụ và thông tin server chẩn đoán.
4. Tổ chức của bạn đã quyết định rằng sự kết hợp chính xác này là được phép.

Việc gộp các sự thật đó thành "nó nằm trong registry, vì vậy hãy tin tưởng nó" tạo ra một điểm mù trong chuỗi cung ứng. Một bản xuất bản hợp lệ vẫn có thể bị phản đối (deprecated). Một tag gói có thể trỏ đến một artifact không mong muốn nếu bạn không ghim digest của nó. Một server có thể thêm một công cụ phá hoại sau khi đã được đánh giá. Một thao tác rollback có thể âm thầm chọn một phiên bản chưa từng được tiếp nhận.

Giải pháp là một bộ điều khiển tiếp nhận (admission controller) với bằng chứng tại mọi ranh giới.

## Registry là một chỉ mục, không phải hệ thống phê duyệt của bạn

Registry MCP chính thức lưu trữ metadata của server. Bản ghi `server.json` của nó đặt tên cho một phiên bản server và khai báo một hoặc nhiều gói hoặc endpoint từ xa. Các quy tắc xuất bản bổ sung xác thực namespace, kiểm tra quyền sở hữu gói, các quy tắc registry hạn chế và vị trí metadata nhà phát hành hẹp.

Các kiểm soát đó trả lời các câu hỏi về xuất bản. Chính sách production của bạn vẫn trả lời các câu hỏi về triển khai:

| Ranh giới | Câu hỏi | Chủ sở hữu bằng chứng |
|---|---|---|
| Namespace | Nhà phát hành có được phép sử dụng tên này không? | Xác thực Registry cộng với đầu vào namespace đã xác minh của bạn |
| Bản ghi | Nhà phát hành đã khai báo gì cho phiên bản này? | Digest `server.json` bất biến |
| Nguồn thực thi | Gói hoặc endpoint từ xa nào sẽ thực thi? | Các trường nguồn đã khai báo, kết quả xác minh quyền sở hữu, phương thức vận chuyển và digest đáng tin cậy |
| Runtime | Endpoint hiện đang hiển thị những gì? | `server/discover` và các mô tả công cụ |
| Tiếp nhận | Chính sách của bạn có phê duyệt tập hợp chính xác này không? | Pin cục bộ và mục nhập sổ cái |
| Vận hành | Nó có còn an toàn không, và cái gì có thể thay thế nó? | Kiểm tra trôi dạt, đồng bộ trạng thái, sức khỏe và lộ trình rollback |

Phiên bản schema của Registry và phiên bản giao thức MCP là độc lập. Một bản ghi có thể sử dụng schema server `2025-12-11` đã xuất bản trong khi server trực tiếp hỗ trợ MCP `2026-07-28`. Đừng bao giờ suy diễn cái này từ cái kia.

```figure
mcp-registry-admission
```

## Bảy kiểm soát trong một quyết định tiếp nhận

### 1. Xác minh Namespace

Các tên Registry chính thức sử dụng các namespace đã xác thực. Một domain đã xác minh có thể ánh xạ tới một tiền tố domain đảo ngược. Ví dụ, quyền kiểm soát `example.com` có thể thiết lập `com.example/*`.

Đừng chấp nhận kiểm tra tiền tố chuỗi:

```python
server_name.startswith("com.example")
```

Điều đó cũng chấp nhận cả `com.exampleevil/tool`. Hãy tách tên tại `/`, yêu cầu một slug không rỗng và so sánh chính xác phân đoạn namespace. Quan trọng hơn, hãy truyền namespace đã xác minh vào quá trình tiếp nhận từ kết quả xác thực. Đừng lấy niềm tin từ bản ghi không đáng tin cậy.

Các namespace được hỗ trợ bởi GitHub và các namespace được hỗ trợ bởi domain sử dụng các đường dẫn xác thực khác nhau. Hãy chuẩn hóa cả hai đường dẫn thành một đầu vào tiếp nhận: chuỗi namespace đã xác minh chính xác.

### 2. Kết nối nguồn gốc (Provenance join)

Đối với một bản ghi gói, khai báo và artifact đã tải về phải khớp nhau trên các trường rõ ràng:

- Loại registry gói
- Định danh gói
- Phiên bản gói
- Kết quả xác minh quyền sở hữu
- Digest của artifact đã tải xuống

Cũng cần xác thực phương thức vận chuyển gói đã khai báo. Một bản ghi chỉ có endpoint từ xa là hợp lệ và không được từ chối vì thiếu gói. Đối với nguồn từ xa, hãy kết hợp URL đã khai báo và loại vận chuyển với quyền sở hữu endpoint đã xác minh độc lập và digest của kết nối đáng tin cậy hoặc bằng chứng triển khai.

Mã nguồn bài học hỗ trợ cả hai loại nguồn và băm nguồn đã chọn cùng với nguồn Registry, tên server, phiên bản Registry, digest bản ghi và digest bằng chứng. Digest nguồn gốc kết quả là một con trỏ nhỏ gọn đến toàn bộ tập hợp bằng chứng. Nó không thay thế cho việc lưu giữ bằng chứng.

Không bao giờ chấp nhận một digest chỉ được cung cấp bởi artifact mà bạn đang cố gắng xác minh. Hãy tính toán nó tại một ranh giới tải về đáng tin cậy, hoặc nhận nó từ một dịch vụ gói mà bạn xác thực kết quả kiểm tra.

### 3. Ghim quyết định, không chỉ phiên bản

Các phiên bản Registry là các định danh xuất bản duy nhất. Metadata đã xuất bản là bất biến. Một bản ghi thay đổi yêu cầu một phiên bản mới. Semantic versioning được khuyến nghị, nhưng Registry không yêu cầu và không chấp nhận các dải phiên bản.

Điều này có nghĩa là `^1.4` không phải là một pin tiếp nhận. "Latest" cũng vậy. Một pin hữu ích bao gồm:

```json
{
  "server": "com.example/inventory",
  "version": "1.0.0",
  "recordDigest": "...",
  "source": {"kind": "package", "registryType": "pypi"},
  "sourceDigest": "...",
  "toolsetDigest": "...",
  "provenanceDigest": "...",
  "registryStatus": "active"
}
```

Việc ghim nhiều lớp cho phép bạn xác định ranh giới nào đã thay đổi. Một thay đổi digest bản ghi dưới cùng một phiên bản Registry là lỗi toàn vẹn Registry. Một thay đổi digest nguồn dưới cùng một tọa độ gói hoặc triển khai từ xa là lỗi toàn vẹn nguồn thực thi. Một thay đổi digest bộ công cụ là trôi dạt runtime.

### 4. Phát hiện trôi dạt trực tiếp (Live drift detection)

Quá trình tiếp nhận nên quan sát server thực sự sẽ nhận lưu lượng truy cập. Gọi `server/discover`, liệt kê hoặc lấy các mô tả công cụ được hiển thị thông qua đường dẫn đáng tin cậy của bạn và xác minh:

- `2026-07-28` nằm trong `supportedVersions`
- tất cả các khả năng bắt buộc cục bộ đều hiện diện
- mọi mô tả công cụ đều có định danh và bề mặt schema bắt buộc
- digest mô tả đã chuẩn hóa khớp với pin đã tiếp nhận trong các lần kiểm tra sau

Kết quả tùy chọn `_meta["io.modelcontextprotocol/serverInfo"]` là ngữ cảnh hiển thị, nhật ký và gỡ lỗi do server tự báo cáo. Hãy ghi lại nó như bằng chứng chẩn đoán, nhưng không bao giờ sử dụng nó để thiết lập namespace, quyền sở hữu gói, quyền sở hữu endpoint, tiếp nhận hoặc bất kỳ quyết định bảo mật nào khác. Một bí danh `serverInfo` trực tiếp bên ngoài `_meta` không phải là trường hợp đồng và không nên được đưa vào bằng chứng chẩn đoán.

Chỉ chuẩn hóa các trường không có ý nghĩa về thứ tự. Mẫu này sắp xếp danh sách công cụ theo tên ổn định trước khi băm, vì vậy một thay đổi thứ tự danh sách vô hại không gây ra trôi dạt. Nó không loại bỏ các trường mô tả. Một công cụ mới, schema thay đổi, mô tả thay đổi hoặc chú thích mới sẽ làm thay đổi pin.

Mẫu này coi các mô tả bị lỗi và bất kỳ thay đổi digest mô tả nào là trôi dạt, cách ly pin, xóa lộ trình hoạt động của nó và chặn phiên bản đó làm mục tiêu rollback. Một chính sách production có thể cho phép thay đổi biên tập chỉ thông qua đánh giá mới, vì các mô tả ảnh hưởng đến việc lựa chọn công cụ của model. Metadata "thẩm mỹ" có thể làm thay đổi hành vi của tác nhân.

### 5. Trạng thái Registry là trạng thái trực tiếp

API Registry đính kèm một đối tượng `_meta` cấp phản hồi bên cạnh mỗi bản ghi server. Các trường do Registry quản lý nằm dưới `_meta["io.modelcontextprotocol.registry/official"]`. Truyền đối tượng `_meta` phản hồi vào quá trình tiếp nhận và đọc `_meta["io.modelcontextprotocol.registry/official"].status`. Giá trị `_meta.status` trực tiếp không phải là hình thái chính thức trên đường truyền. Đừng nhầm lẫn metadata phản hồi với `_meta` của chính bản ghi xuất bản. Trạng thái có thể là:

- `active`: được trả về mặc định và đủ điều kiện để tiếp nhận cục bộ
- `deprecated`: vẫn có thể khám phá với cảnh báo, nhưng không còn là lựa chọn tự động an toàn
- `deleted`: bị ẩn theo mặc định trong khi bản ghi lịch sử của nó vẫn khả dụng thông qua các chế độ xem đã xóa hoặc gia tăng

Đồng bộ trạng thái sau khi tiếp nhận. Nếu một phiên bản đang hoạt động bị phản đối hoặc xóa, hãy cách ly pin của nó và ngừng định tuyến công việc mới đến đó. Giữ lại bằng chứng. Việc xóa khỏi danh sách mặc định không phải là sự cho phép để xóa dấu vết kiểm toán của bạn.

Metadata tùy chỉnh do nhà phát hành cung cấp chỉ thuộc về `_meta.io.modelcontextprotocol.registry/publisher-provided` trong bản ghi xuất bản. Metadata phản hồi do Registry quản lý là riêng biệt. Đừng để nhà phát hành tự thiết lập trạng thái chính thức của riêng họ.

### 6. Rollback nghĩa là khôi phục lộ trình

Một bản xuất bản bất biến không bị chỉnh sửa trong quá trình rollback. Rollback chọn một pin đã được tiếp nhận trước đó, hiện đang đủ điều kiện và thay đổi lộ trình hoạt động.

Một mục tiêu an toàn phải:

1. Có một bản ghi tiếp nhận đã hoàn thành.
2. Vẫn có trạng thái Registry hoạt động theo chính sách của bạn.
3. Không bị cách ly bởi bằng chứng runtime hoặc bảo mật.
4. Vẫn phân giải được đến gói đã ghim và tập hợp mô tả trực tiếp.
5. Vượt qua các kiểm tra sức khỏe hiện tại.

Mẫu này tập trung vào ba điều kiện đầu tiên. Một bộ hòa giải (reconciler) thực tế nên tải lại gói và kiểm tra lại endpoint trực tiếp trước khi kích hoạt.

### 7. Gắn thêm sổ cái tiếp nhận (Admission ledger)

Cơ sở dữ liệu tiếp nhận cho biết cái gì đang hoạt động. Sổ cái giải thích lý do tại sao.

Mỗi mục nhập mẫu chứa một chuỗi, thời gian, sự kiện, server, phiên bản, kết quả, lý do, bằng chứng, hash của mục nhập trước đó và hash của chính nó. Thay đổi một kết quả cũ sẽ phá vỡ xác minh của mục nhập đó và mọi liên kết sau đó.

Đây là cơ chế chống giả mạo, không phải là chống giả mạo một cách kỳ diệu. Hãy neo các đầu sổ cái định kỳ vào một miền tin cậy riêng biệt, chẳng hạn như metadata phát hành đã ký hoặc lưu trữ ghi một lần. Hạn chế người có thể gắn thêm. Giữ các token ủy quyền, thông tin xác thực gói, đối số công cụ và dữ liệu endpoint riêng tư ra khỏi bằng chứng.

## Xây dựng

Bộ điều khiển có thể chạy nằm trong `code/main.py`. Nó chỉ sử dụng thư viện tiêu chuẩn của Python.

Bắt đầu với bản demo hữu hạn:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
```

Bản demo thực hiện năm thao tác:

1. Tiếp nhận `1.0.0` với namespace, nguồn gốc gói, giao thức, khả năng và công cụ khớp nhau.
2. Tiếp nhận `1.1.0` và làm cho nó hoạt động.
3. Quan sát một công cụ xóa bất ngờ tại runtime.
4. Quan sát trạng thái Registry cho `1.1.0` trở thành `deprecated`.
5. Khôi phục định tuyến về pin `1.0.0` vẫn đang được tiếp nhận.

Hình thái mong đợi:

```json
{
  "admitted": [true, true],
  "driftAllowed": false,
  "rollbackAllowed": true,
  "activeVersion": "1.0.0",
  "ledgerValid": true
}
```

Đọc phần triển khai theo thứ tự này:

1. `namespace_for_domain()` và `namespace_matches()` thiết lập thẩm quyền đặt tên chính xác.
2. `digest()` và `normalized_tools()` tạo ra bằng chứng tất định.
3. `RegistryAdmissionController.admit()` kết hợp xuất bản, nguồn gốc, runtime và chính sách.
4. `check_live()` so sánh một quan sát mới với pin.
5. `observe_registry_status()` cách ly các phiên bản có trạng thái Registry thay đổi.
6. `rollback()` chỉ kích hoạt một mục tiêu đủ điều kiện đã được tiếp nhận trước đó.
7. `AdmissionLedger.verify()` phát hiện các thay đổi đối với lịch sử đã ghi.

## Sử dụng

Đặt bộ điều khiển giữa khám phá và định tuyến:

```text
Registry sync -> artifact verifier -> live discovery -> admission controller -> route table
                                               |                 |
                                               v                 v
                                          evidence store    admission ledger
```

Sử dụng các định danh riêng biệt cho các công việc này. Một worker đồng bộ Registry cần quyền đọc metadata. Một bộ xác minh artifact cần quyền tải gói. Một bộ hòa giải lộ trình cần quyền kích hoạt một pin đã phê duyệt. Không ai trong số họ cần mọi thông tin xác thực.

Làm cho trạng thái triển khai trở nên rõ ràng. "Đã phê duyệt" nghĩa là bằng chứng đã vượt qua chính sách. "Đang hoạt động" nghĩa là lộ trình hiện đang chọn nó. "Đã cách ly" nghĩa là nó không thể nhận công việc mới. "Đã thay thế" nghĩa là một phiên bản được tiếp nhận khác đang hoạt động. Đừng mã hóa cả bốn ý nghĩa vào một Boolean.

Chạy tiếp nhận trước khi hiển thị server trong `tools/list`. Nếu không, một client có thể khám phá một công cụ trong khoảng thời gian giữa xuất bản và đánh giá chính sách.

## Lab tương tác

Bạn sẽ quan sát từng ranh giới thất bại một.

### Lab A: xung đột namespace

Mở một shell Python từ thư mục mã nguồn:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/code
python3 -q
```

Sau đó chạy:

```python
from main import namespace_matches
namespace_matches("com.example/inventory", "com.example")
namespace_matches("com.exampleevil/inventory", "com.example")
```

Kết quả đầu tiên là `True`; kết quả thứ hai là `False`. Thay thế so sánh chính xác bằng `startswith` cục bộ và quan sát lý do tại sao tên thứ hai vượt qua ranh giới. Khôi phục so sánh chính xác trước khi tiếp tục.

### Lab B: trôi dạt mô tả (descriptor drift)

```python
from main import *
times = iter(f"2026-08-21T12:00:{n:02d}+00:00" for n in range(10))
c = RegistryAdmissionController(clock=lambda: next(times))
meta = {OFFICIAL_META_KEY: {"status": "active"}}
c.admit(sample_record("1.0.0"), meta, "com.example", evidence_for("1.0.0"), sample_live("1.0.0"))
c.check_live("com.example/inventory", "1.0.0", sample_live("1.0.0", True))
```

Kiểm tra các lý do và trạng thái lộ trình. Gói và bản ghi Registry không thay đổi. Bề mặt công cụ runtime đã thay đổi, vì vậy bộ điều khiển đã cách ly và hủy kích hoạt pin. Đây là lý do tại sao kiểm soát chuỗi cung ứng phải tiếp tục sau khi cài đặt.

### Lab C: trạng thái và rollback

Tiếp nhận `1.1.0`, đánh dấu nó là deprecated và thử cả hai mục tiêu rollback:

```python
c.admit(sample_record("1.1.0"), meta, "com.example", evidence_for("1.1.0"), sample_live("1.1.0"))
c.observe_registry_status("com.example/inventory", "1.1.0", "deprecated")
c.rollback("com.example/inventory", "1.1.0", "unsafe retry")
c.rollback("com.example/inventory", "1.0.0", "restore known release")
c.ledger.verify()
```

Mục tiêu bị cách ly bị từ chối. Pin hoạt động trước đó được chấp nhận. Sổ cái vẫn hợp lệ.

## Lab thực hành

Mở rộng bộ điều khiển với cổng phê duyệt hai người.

Yêu cầu:

- Lưu trữ các phê duyệt dưới dạng tham chiếu bằng chứng đã ký, không phải tên có thể thay đổi trong pin.
- Yêu cầu hai định danh người đánh giá khác nhau cho một bộ công cụ chứa công cụ có `destructiveHint: true`.
- Từ chối các định danh người đánh giá trùng lặp.
- Bảo tồn nỗ lực tiếp nhận ban đầu trong sổ cái khi phê duyệt chưa hoàn tất.
- Thêm các bài kiểm tra cho không, một, trùng lặp và hai phê duyệt riêng biệt.
- Không ghi nhật ký chữ ký, thông tin xác thực hoặc toàn bộ đối số công cụ riêng tư.

Thành công nghĩa là một công cụ phá hoại không thể hoạt động cho đến khi cả hai định danh phê duyệt chính xác bản ghi, gói và digest bộ công cụ.

## Artifact đã xuất xưởng

Bài học này xuất xưởng `outputs/skill-mcp-registry-admission.md`. Sử dụng nó như một runbook phẳng, có thể tái sử dụng khi xem xét phiên bản Registry mới hoặc điều tra sự trôi dạt. Nó xác định các đầu vào, quy tắc từ chối, gói bằng chứng, hòa giải trạng thái và bằng chứng rollback mà không phụ thuộc vào tên lớp mẫu.

## Xác minh

Chạy bản demo và bộ kiểm tra tất định:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Xác minh sẽ chứng minh:

- các ranh giới namespace chính xác từ chối các tiền tố trông giống nhau
- chỉ trạng thái Registry có namespace chính thức mới có thể làm cho một phiên bản đủ điều kiện
- bằng chứng gói và từ xa chưa xác minh hoặc không khớp bị từ chối
- metadata nhà phát hành không thể mạo danh metadata do Registry quản lý
- thứ tự công cụ được chuẩn hóa mà không che giấu các thay đổi mô tả
- các cấu trúc gói và công cụ bị lỗi từ chối một cách an toàn
- `serverInfo` vẫn mang tính chẩn đoán và không bao giờ cung cấp thẩm quyền tiếp nhận
- trôi dạt mô tả gây cách ly, hủy kích hoạt và chặn rollback về pin
- thay đổi trạng thái cách ly các pin đang hoạt động
- rollback không thể chọn một phiên bản bị cách ly hoặc không xác định
- phát hiện giả mạo sổ cái

## Các chế độ thất bại trong Production

| Thất bại | Tại sao nó xảy ra | Phản hồi bắt buộc |
|---|---|---|
| Tên trông hợp lệ nhưng namespace chưa bao giờ được xác thực | Chính sách tin tưởng văn bản bản ghi | Từ chối cho đến khi bộ xác minh namespace đáng tin cậy cung cấp tiền tố chính xác |
| Cùng tọa độ gói trả về byte mới | Nguồn upstream thay đổi hoặc phân phối bị xâm phạm | Ngừng kích hoạt, giữ lại cả hai digest, điều tra ranh giới tải về |
| "Latest" thay đổi mà không cần đánh giá | Lựa chọn thả nổi thoát khỏi pin | Chỉ phân giải các phiên bản và digest đã tiếp nhận chính xác |
| Công cụ mới xuất hiện sau khi phê duyệt | Trôi dạt runtime hoặc triển khai khác | Cách ly lộ trình và nắm bắt quan sát mô tả mới |
| Phiên bản deprecated vẫn hoạt động | Thiếu hoặc chậm trễ đồng bộ trạng thái | Hòa giải trạng thái theo lịch trình và trước khi kích hoạt |
| Bản ghi đã xóa biến mất khỏi đồng bộ mặc định | Client chỉ yêu cầu các bản ghi đang hoạt động | Sử dụng hòa giải gia tăng hoặc nhận biết xóa và bảo tồn lịch sử cục bộ |
| Mục tiêu rollback chưa bao giờ được tiếp nhận | Kiểm soát lộ trình và trạng thái phê duyệt bị ngắt kết nối | Từ chối rollback và chạy tiếp nhận mới cho mục tiêu |
| Sổ cái xác minh cục bộ sau khi kẻ tấn công viết lại tất cả mục nhập | Chuỗi hash không có neo bên ngoài | Xuất bản các đầu sổ cái đã ký lên một miền tin cậy riêng biệt |
| Bằng chứng chứa bearer token hoặc đối số công cụ | Nhật ký sao chép toàn bộ yêu cầu | Biên tập tại thời điểm thu thập và chỉ lưu trữ bằng chứng tối thiểu |

## Quy tắc vận hành

Xuất bản trả lời "định danh này có thể xuất bản tên này không?" Tiếp nhận trả lời "chúng ta sẽ thực thi artifact chính xác này và hiển thị hành vi chính xác này không?" Giữ các quyết định đó tách biệt, ghim mọi kết nối và làm cho rollback chọn bằng chứng thay vì bộ nhớ.

## Đọc thêm

- [Yêu cầu server.json của Registry chính thức](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [Hợp đồng OpenAPI của Registry chính thức](https://registry.modelcontextprotocol.io/openapi.yaml)
- [Khám phá server MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)