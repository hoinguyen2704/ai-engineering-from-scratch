# Khám phá kỹ năng và Tiết lộ lũy tiến (Skill Discovery and Progressive Disclosure)

> Một kỹ năng trở nên hữu ích trước khi phần thân của nó được tải. Tên và mô tả của nó giành được một vị trí trong danh mục; các tệp sâu hơn của nó chỉ nhận được ngữ cảnh khi tác vụ chạm đến chúng.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 22 (Agent Skills: Portable Contract and Runtime Boundary)
**Time:** ~105 phút

## Mục tiêu học tập

- Xây dựng quy trình khám phá hệ thống tệp (filesystem) giúp tách biệt phạm vi, xác thực, chính sách xung đột và xuất bản danh mục.
- Giải thích ba cấp độ tiết lộ: siêu dữ liệu danh mục, hướng dẫn hoạt động và tài nguyên dành riêng cho tác vụ.
- Thiết kế các tham chiếu để một agent có thể truy cập trực tiếp vào chi tiết cần thiết mà không cần tải toàn bộ gói.
- Lập ngân sách không gian danh mục độc lập với ngữ cảnh kỹ năng đang hoạt động.
- Từ chối việc duyệt đường dẫn (path traversal) và thoát khỏi symlink khi một kỹ năng đọc tài nguyên của chính nó.

## Vấn đề

Agent của bạn đã cài đặt 200 kỹ năng. Việc tải mọi `SKILL.md`, tệp tham chiếu, tập lệnh và mẫu khi bắt đầu phiên làm việc sẽ làm chìm tác vụ hiện tại trong các thủ tục không liên quan. Việc không tải gì cả sẽ buộc người dùng phải ghi nhớ chính xác các đường dẫn hệ thống tệp.

Giải pháp thỏa hiệp thông thường là một danh mục: hiển thị cho mô hình một mô tả định danh và định tuyến nhỏ gọn cho mỗi kỹ năng đủ điều kiện, sau đó chỉ tải toàn bộ phần thân sau khi đã chọn. Điều đó tạo ra hai vấn đề kỹ thuật mới.

Thứ nhất, khám phá không chỉ là tìm kiếm tệp đệ quy. Các kỹ năng có thể tồn tại ở phạm vi dự án, người dùng, quản trị viên, plugin hoặc tích hợp sẵn. Hai gói có thể trùng tên. Một symlink có thể trỏ ra ngoài thư mục gốc đáng tin cậy. Một gói bị lỗi có thể tiêu tốn không gian danh mục hoặc trở nên không thể gọi được.

Thứ hai, tiết lộ lũy tiến có thể trở thành sự nhầm lẫn lũy tiến. Nếu `SKILL.md` nói "đọc hướng dẫn liên quan" và gói đó chứa mười hai hướng dẫn, mô hình phải đoán. Nếu mỗi hướng dẫn trỏ đến ba tệp nữa, việc tải sẽ trở thành một quá trình duyệt đồ thị không giới hạn.

Một runtime tốt làm cho việc khám phá trở nên xác định và việc tiết lộ trở nên có chủ đích.

## Khái niệm

### Khám phá là một quy trình biên dịch (compiler pipeline)

Hãy coi hệ thống tệp là đầu vào nguồn. Không xuất bản các đường dẫn thô trực tiếp cho mô hình.

```figure
skill-discovery-pipeline
```

Mỗi giai đoạn nên tạo ra dữ liệu có cấu trúc và các lỗi có cấu trúc. Nhật ký khám phá nên trả lời được:

- Những thư mục gốc nào đã được tìm kiếm?
- Những ứng viên nào đã được tìm thấy?
- Những ứng viên nào bị từ chối và tại sao?
- Gói nào thắng trong một cuộc xung đột?
- Những mục danh mục nào đã bị rút ngắn hoặc bỏ qua do ngân sách?

Nếu không có bằng chứng đó, việc chẩn đoán "mô hình không sử dụng kỹ năng của tôi" là gần như không thể.

### Phạm vi là chính sách runtime

Đặc tả di động (portable specification) định nghĩa một gói kỹ năng, không phải một đường dẫn cài đặt phổ quát hay thứ tự ưu tiên. Host quyết định nơi nó tìm kiếm.

Một runtime chung có thể sử dụng các phạm vi sau:

| Phạm vi | Ví dụ thư mục gốc | Quyền sở hữu dự kiến |
|---|---|---|
| Workspace | `<repo>/.agents/skills/` | Người duy trì dự án |
| User | `<user-data>/skills/` | Một nhà phát triển |
| Administrator | `<system>/skills/` | Chính sách máy hoặc tổ chức |
| Plugin | Một gói plugin đã ký | Nhà xuất bản và cài đặt plugin |
| Built-in | Gói runtime | Nhà cung cấp runtime |

Tính đến tháng 8 năm 2026, Codex ghi lại việc khám phá dự án từ `$CWD/.agents/skills` thông qua các thư mục tổ tiên lên đến thư mục gốc của kho lưu trữ, cộng với các vị trí người dùng, quản trị viên và tích hợp sẵn. Nó hỗ trợ các thư mục kỹ năng được liên kết bằng symlink. Các tên trùng lặp có thể xuất hiện cả hai thay vì bị hợp nhất. Đó là các hành vi của Codex, không phải yêu cầu của `SKILL.md`; hãy xác minh [tài liệu kỹ năng Codex](https://learn.chatgpt.com/docs/build-skills) hiện tại khi viết một adapter.

Không bao giờ tự ý tạo ra thứ tự ưu tiên từ tên thư mục. Hãy khai báo nó như một chính sách và kiểm thử nó. Lab bài học sử dụng một số nguyên thứ hạng rõ ràng cho mỗi `Scope` để cùng một tập hợp ứng viên luôn phân giải theo cùng một cách.

### Xung đột cần định danh ngoài `name`

Hai gói có tên `release-readiness` có thể là hợp lệ. Một gói có thể là ghi đè ở workspace và một gói là mặc định của người dùng. Do đó, một mục danh mục cần ít nhất:

```json
{
  "name": "release-readiness",
  "description": "Inspect a release candidate for this repository.",
  "scope": "workspace",
  "source": "/repo/.agents/skills/release-readiness",
  "selected": true
}
```

Các chính sách xung đột phổ biến bao gồm:

| Chính sách | Lợi ích | Rủi ro |
|---|---|---|
| Giữ mọi ứng viên | Không có gì bị ẩn | Mô hình thấy các tên mơ hồ |
| Phạm vi ưu tiên cao nhất thắng | Gọi đơn giản | Một gói cục bộ có thể che khuất gói đáng tin cậy |
| Từ chối trùng lặp | Không có sự che khuất ngầm | Các ghi đè hợp lệ ngừng hoạt động |
| Định danh theo nguồn | Định danh rõ ràng | Tên hiển thị với người dùng trở nên dài hơn |

Chọn một chính sách cho host. Lưu giữ các ứng viên bị từ chối hoặc bị che khuất trong phần chẩn đoán ngay cả khi chúng vắng mặt trong danh mục mô hình.

### Ba cấp độ tiết lộ

Đặc tả Agent Skills mô tả việc tải theo giai đoạn. Chìa khóa là mỗi cấp độ có một mục đích khác nhau.

```figure
skill-disclosure-levels
```

#### Cấp độ 1: siêu dữ liệu danh mục

Mô hình cần đủ thông tin để phân biệt kỹ năng với các kỹ năng lân cận. Đặc tả ước tính khoảng 100 token cho mỗi mục danh mục, nhưng việc tuần tự hóa và token hóa thực tế thuộc về host.

Một mô tả hữu ích có hai mệnh đề:

```yaml
description: Validate a release candidate and produce a readiness report. Use when the user asks whether a version, tag, or package is ready to publish.
```

Mệnh đề đầu tiên nêu khả năng. Mệnh đề thứ hai nêu ranh giới kích hoạt. Bài học 25 đánh giá ranh giới này với các prompt tích cực và gần đúng.

#### Cấp độ 2: hướng dẫn hoạt động

Sau khi kích hoạt, phần thân nên hoạt động như một bản đồ và một quy trình. Đặc tả khuyến nghị giữ `SKILL.md` dưới 500 dòng. Đó là một tín hiệu thiết kế, không phải mục tiêu để lấp đầy.

Phần thân nên chứa:

- ranh giới tác vụ;
- quy trình làm việc mặc định;
- các điều kiện nhánh;
- tham chiếu trực tiếp đến các tệp sâu hơn;
- hợp đồng công cụ và tập lệnh;
- hành vi thất bại và dừng;
- đầu ra mong đợi và xác minh của nó.

Đừng di chuyển quy trình làm việc trung tâm vào một tham chiếu chỉ để làm cho tệp mục nhập ngắn gọn. Việc kích hoạt phải cung cấp cho mô hình đủ ngữ cảnh để bắt đầu một cách chính xác.

#### Cấp độ 3: tài nguyên hỗ trợ

Các tham chiếu cung cấp văn bản hoặc dữ liệu. Các tập lệnh cung cấp tính toán xác định. Tài sản được sao chép, điền vào hoặc chuyển đổi thành các sản phẩm bàn giao thay vì được coi là hướng dẫn.

| Thư mục | Mô hình có đọc không? | Mô hình có thực thi không? | Nội dung điển hình |
|---|:---:|:---:|---|
| `references/` | Có, khi cần | Không | lược đồ, chính sách, hướng dẫn miền |
| `scripts/` | Có thể kiểm tra | Thông qua công cụ được phép | trình xác thực, bộ chuyển đổi, bộ thu thập |
| `assets/` | Chỉ khi hữu ích | Không | mẫu, fixtures, hình ảnh, tệp khởi đầu |

Những tên này là quy ước, không phải khả năng ma thuật. Host vẫn cần quyền truy cập tệp và một công cụ thực thi.

### Tham chiếu theo nhánh tốt hơn việc đổ dữ liệu theo chủ đề

Viết tệp mục nhập như một bản đồ quyết định:

```markdown
## Choose the path

- For a Python package, read `references/python-release.md`.
- For a container image, read `references/container-release.md`.
- For a documentation-only release, read `references/docs-release.md`.
- If the release combines artifact types, read only the guides for those artifacts.
```

Điều này cung cấp cho mỗi tham chiếu một điều kiện tải có thể quan sát được. "Đọc `references/` để biết thêm" thì không.

Giữ đồ thị tham chiếu nông. Hướng dẫn chính thức khuyến nghị các liên kết trực tiếp từ `SKILL.md` và tránh các chuỗi sâu. Một bước nhảy làm cho khả năng tiếp cận có thể kiểm thử được và giảm khả năng một ràng buộc cần thiết không bao giờ đi vào ngữ cảnh.

```figure
skill-reference-map
```

### Ngân sách danh mục và ngữ cảnh hoạt động là các ngân sách khác nhau

Để `c_i` là chi phí danh mục đã tuần tự hóa của kỹ năng `i`, `B_c` là ngân sách danh mục, `b_j` là chi phí phần thân hoạt động và `r_k` là các tài nguyên thực sự được tải.

```text
catalog_cost = sum(c_i for every published skill)
active_cost = sum(b_j for every activated skill) + sum(r_k for every disclosed resource)
```

Việc giảm một ngân sách không tự động giảm ngân sách kia. Các mô tả ngắn có thể tiết kiệm không gian danh mục trong khi phần thân 900 dòng được kích hoạt vẫn làm quá tải tác vụ. Việc chia nhỏ phần thân thành các tham chiếu chỉ có thể giảm chi phí hoạt động khi runtime và các hướng dẫn thực sự tránh tải các nhánh không liên quan.

Codex hiện lập ngân sách cho danh sách kỹ năng ban đầu ở mức 2 phần trăm cửa sổ ngữ cảnh khi biết kích thước cửa sổ ngữ cảnh. Giá trị 8.000 ký tự chỉ là phương án dự phòng khi không biết kích thước đó; nó không phải là mức trần thứ hai kết hợp với quy tắc 2 phần trăm. Khi danh mục vượt quá ngân sách áp dụng, các mô tả có thể bị rút ngắn hoặc bỏ qua. Hãy coi những con số đó là chính sách hiện tại của Codex, không phải là thuộc tính của tiêu chuẩn Agent Skills.

### Đường dẫn tài nguyên là một ranh giới tin cậy

Một kỹ năng chỉ nên đọc các tệp bên trong gói của nó. Kiểm tra tiền tố chuỗi theo nghĩa đen là không đủ:

```text
references/../../../../.ssh/config
references/external-link -> /private/company-secrets
```

Phân giải thư mục gốc của gói và ứng viên bằng ngữ nghĩa hệ thống tệp, từ chối các đầu vào tuyệt đối và xác minh rằng ứng viên đã phân giải vẫn nằm dưới thư mục gốc đã phân giải. Quyết định xem symlink có được phép trước khi khám phá hay không. Nếu được phép, hãy kiểm tra mục tiêu đã phân giải mỗi lần.

```figure
skill-resource-containment
```

Sự chứa đựng đường dẫn không thiết lập sự tin cậy nội dung. Một tham chiếu hợp lệ trong gói vẫn có thể chứa các hướng dẫn độc hại. Bài học 26 xử lý mối đe dọa đó.

### Việc tải phải có thể quan sát được

Ghi lại các sự kiện tiết lộ mà không ghi lại các bí mật:

```json
{
  "event": "skill.resource.loaded",
  "skill": "release-readiness",
  "resource": "references/python-release.md",
  "reason": "candidate contains pyproject.toml",
  "bytes": 2840
}
```

Lý do biến một lựa chọn ngữ cảnh thành bằng chứng có thể xem xét. Nó cũng giúp xác định các hướng dẫn khiến agent tải mọi tệp "đề phòng trường hợp cần đến".

## Xây dựng (Build It)

`code/main.py` xây dựng một công cụ khám phá và tiết lộ xác định.

Bề mặt khám phá bao gồm:

- `Scope` cho siêu dữ liệu nguồn và thứ tự ưu tiên;
- `SkillCandidate` cho một ứng viên hệ thống tệp chưa được xác thực;
- `discover_scope(scope)` để liệt kê các thư mục kỹ năng ngay lập tức;
- `resolve_collisions(candidates, precedence)` để áp dụng một chính sách đã khai báo;
- `CatalogEntry` và `build_catalog(...)` để xuất bản siêu dữ liệu có giới hạn;
- `CatalogBudget` để tính toán các mục đã tuần tự hóa mà không giả định ký tự là các token phổ quát.

Bề mặt tiết lộ bao gồm:

- `load_skill_body(entry, ...)` cho kích hoạt Cấp độ 2;
- `validate_reference(skill_dir, reference)` cho sự chứa đựng đường dẫn;
- `load_reference(...)` cho các lần đọc Cấp độ 3 có giới hạn.

Chạy lab:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Khối này yêu cầu một bản sao cục bộ và phân giải thư mục gốc của kho lưu trữ từ bất kỳ thư mục làm việc nào bên trong bản sao đó.

Bản demo tạo ra các phạm vi dự án và người dùng tạm thời, chèn một xung đột, xây dựng một danh mục dưới ngân sách nhỏ một cách cố ý, kích hoạt một kỹ năng và thử cả việc đọc tham chiếu hợp lệ và thoát khỏi duyệt đường dẫn. Không có tệp vĩnh viễn nào được cài đặt.

### Tại sao khám phá lại nông

`discover_scope` kiểm tra các thư mục con ngay lập tức cho `SKILL.md`. Nó không đệ quy coi mọi `SKILL.md` lồng nhau là một gói riêng biệt. Điều này bảo toàn ranh giới gói và tránh vô tình xuất bản các ví dụ hoặc fixtures bên trong một kỹ năng đã cài đặt.

### Tại sao lab không phân tích cú pháp YAML tùy ý

Lab hỗ trợ frontmatter vô hướng cần thiết cho danh mục của nó. Một runtime sản xuất nên sử dụng trình phân tích cú pháp YAML an toàn với lược đồ rõ ràng, giới hạn kích thước và vô hiệu hóa việc xây dựng đối tượng tùy chỉnh. "Chỉ stdlib" là một ràng buộc giảng dạy, không phải sự cho phép để tự ý tạo ra một phương ngữ YAML một phần một cách âm thầm.

## Sử dụng (Use It)

Áp dụng danh sách kiểm tra này cho bất kỳ adapter khám phá nào:

1. Liệt kê mọi thư mục gốc đã cấu hình và ai có thể ghi vào đó.
2. Nêu rõ liệu các gói được liên kết bằng symlink có được phép hay không.
3. Xác thực tên gói, tên thư mục, siêu dữ liệu bắt buộc và kích thước phần thân mục nhập.
4. Bảo toàn nguồn và phạm vi trong định danh nội bộ.
5. Khai báo và kiểm thử hành vi tên trùng lặp.
6. Đo lường chính xác danh mục đã tuần tự hóa được gửi đến mô hình.
7. Ghi lại lý do tại sao một phần thân hoặc tài nguyên được tải.
8. Giữ các lần đọc tài nguyên bên trong thư mục gốc của gói đã phân giải.
9. Thất bại rõ ràng khi một tệp được tham chiếu bị thiếu.
10. Xây dựng lại danh mục khi các cài đặt hoặc chính sách thay đổi.

## Xuất bản (Ship It)

Bài học này tạo ra gói `skill-catalog-builder`. Nó quét các thư mục gốc được sắp xếp rõ ràng, từ chối các tệp mục nhập được liên kết bằng symlink và sự không khớp giữa tên-thư mục, phân giải các xung đột chéo phạm vi, từ chối các bản sao có cùng thứ tự ưu tiên và khớp siêu dữ liệu đã chọn vào các ngân sách mục nhập, mô tả và ký tự tuần tự hóa đã khai báo.

Báo cáo JSON của nó chứa các mục đã chọn, các ứng viên bị che khuất, các mục bị bỏ qua, lỗi xác thực, thứ tự ưu tiên và việc sử dụng ngân sách. Việc tải phần thân và tham chiếu vẫn là các hoạt động runtime riêng biệt, vì vậy trình xây dựng danh mục không thực thi các tập lệnh hoặc đưa toàn bộ gói vào ngữ cảnh.

## Bài tập

1. Thêm một phạm vi plugin và đặt nó giữa thứ tự ưu tiên của người dùng và tích hợp sẵn. Chứng minh kết quả xung đột bằng một bài kiểm tra.
2. Thay đổi chính sách xung đột từ ưu tiên cao nhất sang tên đủ điều kiện. Bảo toàn cả hai mục trong danh mục.
3. Thêm giới hạn kích thước byte vào `load_reference`. Kiểm tra một tệp chính xác tại giới hạn và một tệp vượt quá một byte.
4. Tạo hai mô tả nghe gần như giống hệt nhau. Viết lại chúng để các ranh giới kích hoạt không chồng chéo.
5. Thêm một manifest chứa các mã băm cho mọi tham chiếu và tập lệnh. Phát hiện tài nguyên đã sửa đổi trước khi tải nó.
6. Cài đặt công cụ cho bản demo để báo cáo số lượng byte Cấp độ 1, Cấp độ 2 và Cấp độ 3 riêng biệt.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói gì | Nó thực sự có nghĩa là gì |
|---|---|---|
| Skill discovery | "Tìm mọi SKILL.md" | Tìm kiếm các phạm vi đã cấu hình, xác thực gói, đính kèm nguồn gốc và áp dụng chính sách |
| Skill catalog | "Danh sách các kỹ năng đã cài đặt" | Siêu dữ liệu định tuyến nhỏ gọn hiển thị với mô hình cho các gói đủ điều kiện |
| Collision policy | "Bản sao nào thắng" | Một quy tắc đã khai báo cho các ứng viên cùng tên từ các nguồn khác nhau |
| Progressive disclosure | "Tải lười (lazy loading)" | Tiếp nhận ngữ cảnh theo giai đoạn từ danh mục đến phần thân đến các tài nguyên theo nhánh |
| Reference graph | "Các tệp được liên kết bởi kỹ năng" | Cấu trúc tài nguyên có thể tiếp cận và các điều kiện tải của nó |
| Path containment | "Ở trong thư mục" | Xác minh các mục tiêu tài nguyên đã phân giải vẫn nằm bên trong thư mục gốc của gói đã phân giải |

## Đọc thêm

- [Đặc tả Agent Skills](https://agentskills.io/specification) về hình dạng gói và các cấp độ tiết lộ lũy tiến.
- [Tối ưu hóa mô tả kỹ năng](https://agentskills.io/skill-creation/optimizing-descriptions) cho siêu dữ liệu định tuyến danh mục.
- [Các phương pháp hay nhất về Agent Skills](https://agentskills.io/skill-creation/best-practices) cho các tham chiếu trực tiếp và kích thước tệp mục nhập.
- [OpenAI: Xây dựng kỹ năng](https://learn.chatgpt.com/docs/build-skills) cho các phạm vi khám phá và giới hạn danh mục Codex hiện tại.