# Kỹ năng Agent: Hợp đồng di động và Ranh giới Runtime

> Một kỹ năng (skill) không phải là một prompt dài với tên tệp hay hơn. Đó là một gói có thể khám phá được bao gồm các hướng dẫn, tài nguyên và các trình trợ giúp có thể thực thi, đi vào ngữ cảnh của agent thông qua một hợp đồng runtime.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 01 (Giao diện Tool), Phase 13 · 05 (Thiết kế Schema cho Tool)
**Time:** ~90 phút

## Mục tiêu học tập

- Định nghĩa kỹ năng của agent mà không nhầm lẫn nó với prompt, hướng dẫn trong repository, tool, hook, subagent hoặc plugin.
- Đọc hợp đồng di động `SKILL.md` và tách biệt nó khỏi các phần mở rộng dành riêng cho runtime.
- Giải thích các giai đoạn vòng đời riêng biệt: khám phá, lựa chọn, kích hoạt, tải tài nguyên, sử dụng tool và xác minh.
- Xác thực một gói kỹ năng trước khi runtime đưa nó vào danh mục của agent.
- Lựa chọn giữa kỹ năng, MCP tool, hook, subagent hoặc code thông thường cho một tác vụ cụ thể.

## Mười phút để thành công bước đầu

Hãy thực hiện việc này trước khi đọc phần giải thích dài. Bạn sẽ tạo một kỹ năng nhỏ, cài đặt gói reviewer hoàn chỉnh vào một agent host thực tế, gọi nó, xác minh kết quả và gỡ bỏ nó. Điều này chứng minh vòng đời với một kết quả có thể quan sát được.

### Kiểm tra trước cho lab real-host

Checkpoint real-host yêu cầu Node.js, `npx`, Python 3, một host có khả năng hỗ trợ kỹ năng và quyền ghi vào dự án hoặc phạm vi người dùng mà bạn chọn trong trình cài đặt. Hãy xác minh các lệnh cục bộ trước:

```bash
node --version
npx --version
python3 --version
```

Quyết định host và phạm vi bạn sẽ sử dụng trước khi cài đặt. Nếu bất kỳ yêu cầu nào không khả dụng, hãy đọc bài học này trên trang web hoặc tiếp tục với bài tập gói thủ công bên dưới. Cách thay thế đó dạy về hợp đồng, nhưng nó không chứng minh được việc khám phá host, gọi lệnh, thực thi script đi kèm hoặc hành vi gỡ cài đặt. Hãy giữ các quan sát đó ở trạng thái chờ xử lý.

### 1. Bắt đầu trong một thư mục làm việc trống

Chạy các lệnh này từ bất kỳ thư mục cha nào nơi bạn lưu trữ công việc học tập:

```bash
mkdir -p agent-skills-first-run
cd agent-skills-first-run
TARGET_ROOT="$(pwd -P)"
printf 'TARGET_ROOT=%s\n' "$TARGET_ROOT"
ls -A
```

Lệnh cuối cùng không được in ra bất cứ thứ gì. Nếu nó in ra các tệp, hãy chọn một thư mục trống khác để phần đánh giá có ranh giới rõ ràng.

Tạo một thư mục cho kỹ năng đầu tiên của bạn:

```bash
mkdir -p my-first-skill
```

Tạo `my-first-skill/SKILL.md` với nội dung sau:

```markdown
---
name: my-first-skill
description: Turn rough meeting notes into a compact decision record when the user asks to capture a technical decision.
---

# Decision record

Extract the decision, context, alternatives, owner, and next review date.
If the notes do not contain a decision, ask one clarifying question instead
of inventing one.
```

Xác minh rằng bạn đã tạo tệp trong thư mục dự định:

```bash
test -f my-first-skill/SKILL.md
```

Không có đầu ra và mã thoát 0 nghĩa là tệp đã tồn tại.

### 2. Cài đặt gói reviewer hoàn chỉnh

Ở lại `agent-skills-first-run` và chạy:

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-contract-reviewer --full-depth
```

Chọn agent host và phạm vi bạn đang sử dụng. Trình cài đặt sẽ liệt kê `skill-contract-reviewer` và đích đến mà nó đã ghi. `--full-depth` là bắt buộc vì kỹ năng của bài học này là một gói lồng nhau với các tham chiếu, một script và một tài sản (asset).

Đặt `SKILL_ROOT` thành thư mục tuyệt đối được trình cài đặt báo cáo. Đó phải là thư mục chứa `SKILL.md` đã cài đặt, không phải thư mục nguồn bài học và không phải không gian làm việc hiện tại:

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-contract-reviewer" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\n' "$SKILL_ROOT"
```

Nếu phiên agent đã mở, hãy bắt đầu một phiên mới hoặc sử dụng lệnh quét lại kỹ năng của host đó. Đừng giả định rằng mọi host đều tự động tải lại danh mục của nó.

### 3. Gọi lệnh một cách tường minh

Trong agent đã cài đặt, với `agent-skills-first-run` là thư mục làm việc, hãy sử dụng cú pháp được host đó hỗ trợ:

| Host | Gọi lệnh tường minh |
|---|---|
| Codex | `skill-contract-reviewer`, hoặc chọn nó từ `/skills`, sau đó cung cấp yêu cầu đánh giá |
| Claude Code | `/skill-contract-reviewer` theo sau là yêu cầu đánh giá |
| Portable fallback | `Use skill-contract-reviewer to review the target package.` |

Sử dụng các giá trị tuyệt đối được in cho `SKILL_ROOT` và `TARGET_ROOT` trong yêu cầu. Yêu cầu host mở rộng chúng trước khi thực thi và hiển thị lệnh đã phân giải chính xác, không phải lệnh phụ thuộc vào thư mục làm việc của tiến trình:

```text
Use skill-contract-reviewer to review <TARGET_ROOT>/my-first-skill. The installed bundle root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/check_skill.py <TARGET_ROOT>/my-first-skill. Before running it, show the fully resolved argv. Return the validation report, selected primitives, and one sentence for each selection. Include the resolved script path, resolved target path, cwd, argv, and exit code as execution evidence.
```

Lệnh đã phân giải sẽ có dạng này, không còn placeholder nào:

```bash
python3 "/absolute/install/path/skill-contract-reviewer/scripts/check_skill.py" \
  "/absolute/workspace/path/agent-skills-first-run/my-first-skill"
```

Một kết quả thành công có cả ba thuộc tính:

1. Host tìm thấy `skill-contract-reviewer` theo tên.
2. Reviewer đọc hợp đồng gói và chạy trình xác thực đi kèm.
3. Phản hồi chứa báo cáo xác thực không có lỗi cấu trúc cho mẫu, cộng với lựa chọn nguyên thủy (primitive) có căn cứ.

Bằng chứng thực thi cũng phải nêu tên đường dẫn script, đường dẫn đích, cwd, vector đối số chính xác và mã thoát. Một báo cáo trôi chảy mà không có các trường đó không chứng minh được rằng script đi kèm đã chạy.

Nếu host báo cáo rằng kỹ năng không khả dụng, hãy xác minh đích cài đặt, quét lại hoặc khởi động lại một lần và thử lại yêu cầu tường minh. Đừng viết lại mô tả kỹ năng để che giấu lỗi cài đặt.

### 4. Thăm dò lựa chọn ngầm định

Bắt đầu một lượt agent mới và nhập cùng một tác vụ mà không nêu tên kỹ năng:

```text
Review <TARGET_ROOT>/my-first-skill as a reusable agent package and tell me whether its package contract is valid.
```

Nếu host hiển thị các kỹ năng đã chọn, hãy ghi lại xem nó có chọn `skill-contract-reviewer` hay không. Nếu host không hiển thị định tuyến, hãy đánh dấu lựa chọn ngầm định là chưa được xác minh. Việc gọi lệnh tường minh là phương án dự phòng di động.

### 5. Dọn dẹp

Chỉ xóa gói reviewer đã cài đặt:

```bash
npx skills remove skill-contract-reviewer
```

Chọn cùng host và phạm vi đã sử dụng trong quá trình cài đặt. Sau khi quét lại hoặc phiên mới, một yêu cầu tường minh cho `skill-contract-reviewer` sẽ báo cáo rằng nó không khả dụng. Giữ `my-first-skill` cho các bài học sau, hoặc xóa thư mục lab sau khi bạn hoàn thành lộ trình.

## Vấn đề

Giả sử nhóm của bạn có một quy trình phát hành đáng tin cậy. Nó tìm các thay đổi đã hợp nhất, kiểm tra ghi chú di chuyển, cập nhật changelog, chạy lệnh đóng gói và tạo danh sách kiểm tra đánh giá.

Việc đưa quy trình đó vào một prompt duy nhất khiến nó dễ dán nhưng khó vận hành. Prompt không có danh tính ổn định, không có quy tắc khám phá, không có ranh giới tài nguyên, không có hình dạng gói có thể kiểm thử và không có câu trả lời cho các câu hỏi cơ bản: Ai được phép gọi nó? Khi nào model nên chọn nó? Nó có thể chạy những script nào? Những tệp nào được tin cậy? Điều gì còn sót lại khi ngữ cảnh bị nén?

Sai lầm ngược lại là coi mọi hướng dẫn có thể tái sử dụng là một kỹ năng. Các quy ước repository, tự động hóa tất yếu, công cụ bên ngoài, hook sự kiện và các agent được ủy quyền giải quyết các vấn đề khác nhau. Việc đóng gói tất cả chúng vào `SKILL.md` tạo ra một thư mục trông có vẻ di động trong khi lại phụ thuộc vào hành vi không được ghi lại của một host cụ thể.

Nhiệm vụ kỹ thuật đầu tiên là phân loại. Hãy quyết định artifact đó là gì trước khi bạn quyết định cách đóng gói nó.

## Khái niệm

### Kỹ năng mã hóa kiến thức quy trình

Kỹ năng của agent là một thư mục có điểm vào là `SKILL.md`. Tệp đầu vào chứa YAML frontmatter theo sau là các hướng dẫn Markdown. Thư mục cũng có thể chứa các tham chiếu, script và tài sản.

```figure
skill-package-anatomy
```

Thư mục, chứ không phải tệp Markdown đơn lẻ, mới là đơn vị có thể triển khai. Một `SKILL.md` được sao chép mà thiếu các tham chiếu là một gói bị hỏng ngay cả khi frontmatter của nó được phân tích cú pháp thành công.

### Các khái niệm trừu tượng lân cận

| Artifact | Công việc chính | Được tải hoặc chạy khi | Không nên mạo danh |
|---|---|---|---|
| Prompt | Định hình một tương tác model | Được ứng dụng hoặc người dùng đưa vào | Một gói có phiên bản với tài nguyên |
| Hướng dẫn repository | Giải thích các quy tắc hiện hành của codebase | Một runtime coding đi vào phạm vi đó | Một quy trình tác vụ có thể tái sử dụng |
| Kỹ năng agent | Cung cấp kiến thức quy trình có thể tái sử dụng | Kích hoạt tường minh hoặc ngầm định | Một ranh giới ủy quyền cứng |
| MCP tool | Phơi bày một khả năng từ xa có kiểu | Model hoặc ứng dụng gọi nó | Một quy trình vận hành chi tiết |
| Hook | Chạy logic tất yếu khi có sự kiện | Sự kiện được khai báo xảy ra | Định tuyến model xác suất |
| Subagent | Ủy quyền công việc với ngữ cảnh và trạng thái riêng | Một trình điều phối tạo hoặc gọi nó | Một gói hướng dẫn tĩnh |
| Plugin | Phân phối một phần mở rộng runtime lớn hơn | Host cài đặt hoặc kích hoạt nó | Chính hợp đồng kỹ năng di động |
| Thư viện kỹ năng đã học | Lưu trữ hành vi được khám phá qua kinh nghiệm | Một chính sách truy xuất chương trình hoặc quỹ đạo trước đó | Một gói `SKILL.md` dựa trên tiêu chuẩn |

Một kỹ năng phát hành có thể cho agent biết cách kiểm tra bản phát hành. Một MCP server có thể phơi bày registry phát hành. Một hook có thể cấm đẩy code trực tiếp. Một subagent có thể kiểm toán ứng viên một cách độc lập. Những phần này kết hợp với nhau vì chúng giữ các trách nhiệm khác nhau.

### Từ "kỹ năng" đặt tên cho hai ý tưởng khác nhau

Các hệ thống nghiên cứu đôi khi gọi một chương trình đã học, quỹ đạo thành công hoặc đoạn chính sách dành riêng cho môi trường là một kỹ năng. Một agent có thể tạo ra các artifact này trong quá trình khám phá, truy xuất chúng theo sự tương đồng của tác vụ, thực thi chúng và sửa đổi thư viện từ phản hồi. Phase 14 · 10 xây dựng loại thư viện học tập suốt đời đó.

Kỹ năng Agent trong lộ trình nhỏ này thì khác. Đó là một gói được tác giả tạo ra với hợp đồng hệ thống tệp được khai báo, metadata danh mục, tiết lộ lũy tiến, gọi lệnh thông qua runtime và các tool do host kiểm soát. Nó có thể được tạo hoặc cải thiện bởi một agent, nhưng việc học không bắt buộc đối với định dạng này.

| Chiều | Gói Kỹ năng Agent | Thư viện kỹ năng đã học |
|---|---|---|
| Đơn vị chính | Thư mục `SKILL.md` | Chương trình, chính sách, quỹ đạo hoặc bản ghi bộ nhớ |
| Tạo | Được tác giả tạo, tạo tự động hoặc tuyển chọn | Thường được khám phá từ kinh nghiệm môi trường |
| Lựa chọn | Mô tả danh mục cộng với chính sách runtime | Truy xuất hoặc chính sách dựa trên trạng thái tác vụ |
| Thực thi | Model làm theo hướng dẫn và gọi tool của host | Môi trường chạy một hành vi hoặc artifact code đã lưu |
| Tính di động | Hợp đồng gói có thể vượt qua các host tương thích | Thường gắn liền với một môi trường và không gian hành động |
| Đánh giá | Định tuyến, artifact, an toàn và tính tương thích của host | Phần thưởng, tỷ lệ thành công, chuyển đổi và tăng trưởng thư viện |

Cả hai ý tưởng đều đóng gói năng lực có thể tái sử dụng. Chúng không nên chia sẻ các tuyên bố thực thi chỉ vì chúng chia sẻ một cái tên.

### Cốt lõi di động

Đặc tả Kỹ năng Agent yêu cầu hai trường frontmatter:

```yaml
---
name: release-readiness
description: Inspect a release candidate when the user asks whether a version is ready to publish.
---
```

`name` là định danh ổn định. Nó phải thỏa mãn các quy tắc đặt tên của đặc tả và khớp với thư mục cha. `description` vừa là tài liệu vừa là metadata định tuyến. Nó nên cho biết kỹ năng làm gì và khi nào nó áp dụng.

Các trường tùy chọn di động là:

| Trường | Mục đích | Lưu ý về tính di động |
|---|---|---|
| `license` | Nêu các điều khoản cho gói | Đặc tả cốt lõi |
| `compatibility` | Nêu các yêu cầu môi trường | Đặc tả cốt lõi |
| `metadata` | Mang dữ liệu mở rộng dạng chuỗi | Đặc tả cốt lõi |
| `allowed-tools` | Gợi ý các tool đã được phê duyệt trước | Thử nghiệm; hỗ trợ của host khác nhau |

Phần thân Markdown chứa các hướng dẫn vận hành. Nó nên xác định quy trình, các điểm quyết định, hành vi khi thất bại và các đường dẫn trực tiếp đến các tài nguyên hỗ trợ.

```markdown
# Release readiness

Use this workflow for a release candidate, not for ordinary development builds.

1. Read `references/release-policy.md`.
2. Run `python3 scripts/inspect_release.py --format json`.
3. Stop if the report contains a blocking failure.
4. Produce the checklist from `assets/release-checklist.md`.
5. Ask for approval before any publish or tag action.
```

### Các phần mở rộng runtime là lớp thứ hai

Một số host chấp nhận frontmatter bổ sung hoặc cấu hình đi kèm. Các trường đó có thể hữu ích, nhưng chúng không tự động di động.

| Hành vi | Ví dụ phần mở rộng host | Cốt lõi di động? |
|---|---|:---:|
| Ẩn kỹ năng khỏi định tuyến model trong khi vẫn cho phép gọi lệnh trực tiếp | `disable-model-invocation` | Không |
| Ẩn kỹ năng khỏi menu lệnh của người dùng trong khi cho phép định tuyến model | `user-invocable` | Không |
| Hiển thị trợ giúp đối số trong menu lệnh | `argument-hint` | Không |
| Chạy kỹ năng trong ngữ cảnh được ủy quyền | `context`, `agent` | Không |
| Ghim cài đặt model hoặc suy luận | `model`, `effort` | Không |
| Đăng ký tự động hóa vòng đời | `hooks` | Không |
| Vô hiệu hóa gọi lệnh ngầm định trong Codex | Chính sách `agents/openai.yaml` | Không |

Hãy coi mỗi phần mở rộng như một bộ chuyển đổi (adapter). Giữ cho quy trình cốt lõi hợp lệ mà không cần nó, ghi lại phương án dự phòng và kiểm thử host tiêu thụ nó. Một runtime có thể bỏ qua một trường không xác định, từ chối nó hoặc bảo toàn nó mà không thực hiện hành vi đó.

### Frontmatter là metadata có thể thực thi

Metadata thay đổi hành vi hệ thống trước khi phần thân kỹ năng được đọc.

- Một `name` bị lỗi có thể khiến việc khám phá thất bại.
- Một `description` mơ hồ có thể định tuyến sai các yêu cầu.
- Một cờ chỉ dành cho con người có thể xóa kỹ năng khỏi danh mục của model.
- Một quyền cho phép tool có thể thay đổi việc host có yêu cầu cấp phép hay không.
- Một cài đặt ngữ cảnh có thể di chuyển việc thực thi sang một phiên agent riêng biệt.

Hãy xem xét frontmatter như code cấu hình. Xác thực nó, lập phiên bản nó và đưa hành vi của nó vào các đánh giá (evals).

### Vòng đời kỹ năng

```figure
skill-runtime-lifecycle
```

Mỗi mũi tên là một ranh giới với các chế độ thất bại riêng.

1. **Khám phá** tìm các gói khả thi ở các vị trí đã cấu hình.
2. **Xác thực** từ chối các gói bị lỗi hoặc không an toàn trước khi xuất bản danh mục.
3. **Lập danh mục** phơi bày một `name` và `description` nhỏ gọn, không phải toàn bộ gói.
4. **Lựa chọn** quyết định xem kỹ năng có liên quan hay không.
5. **Kích hoạt** tải phần thân vào ngữ cảnh mà model có thể nhìn thấy.
6. **Tiết lộ** chỉ đọc các tham chiếu hoặc tài sản khi một nhánh yêu cầu chúng.
7. **Thực thi** sử dụng các tool của host theo các quy tắc cấp phép và cách ly của host.
8. **Xác minh** kiểm tra artifact được tạo ra một cách độc lập với tuyên bố của model.

Việc gộp các giai đoạn này gây ra các mô hình tư duy sai lệch. Một kỹ năng được khám phá không có nghĩa là nó đang hoạt động. Một kỹ năng đang hoạt động không có nghĩa là nó được ủy quyền để làm mọi thứ nó mô tả. Một lệnh gọi tool được cho phép không phải là bằng chứng cho thấy kết quả là chính xác.

### Kỹ năng và tool là trực giao

MCP trả lời câu hỏi: "Ứng dụng này có thể gọi những khả năng nào và schema của chúng là gì?" Một kỹ năng trả lời câu hỏi: "Agent nên tiếp cận loại tác vụ này như thế nào?"

```figure
skill-tool-orthogonality
```

Kỹ năng có thể nêu tên một tool, nhưng host sở hữu registry khả năng thực tế. Nếu tool vắng mặt, kỹ năng nên nêu phương án dự phòng hoặc thất bại một cách rõ ràng. Nó không bao giờ được ngụ ý rằng việc đặt tên cho một khả năng sẽ tạo ra nó.

### Kỹ năng và hướng dẫn repository là các phạm vi khác nhau

Hướng dẫn repository mô tả môi trường bạn đang ở: các lệnh, quy ước, tệp được tạo và ranh giới. Một kỹ năng cung cấp quy trình có thể tái sử dụng cho một tác vụ có thể xảy ra trên nhiều repository.

Khi cả hai cùng áp dụng, yêu cầu của người dùng đang hoạt động và các quy tắc repository sẽ ràng buộc kỹ năng. Một kỹ năng tái cấu trúc chung không được ghi đè lên quy tắc repository cấm chỉnh sửa các tệp được tạo tự động.

### Các kỹ năng không nhập (import) lẫn nhau

Một kỹ năng có thể hướng dẫn agent gọi một kỹ năng khác, nhưng đây không phải là import ở cấp độ ngôn ngữ. Kỹ năng thứ hai vẫn phải trải qua quá trình khám phá runtime, tính đủ điều kiện, kích hoạt, cấp phép và xử lý ngữ cảnh.

Hãy viết các phụ thuộc chéo giữa các kỹ năng dưới dạng các cạnh quy trình có thể quan sát được:

```markdown
After producing the candidate changelog, invoke the `release-risk-review` skill.
Pass the candidate path and require a blocking or non-blocking verdict.
If that skill is unavailable, stop and report the missing dependency.
```

Điều này làm cho sự phụ thuộc có thể kiểm thử và cho phép host thực thi chính sách.

## Xây dựng nó

`code/main.py` triển khai một trình xác thực hướng tiêu chuẩn nhỏ và một bộ chọn artifact. Nó chỉ sử dụng stdlib để mọi quy tắc đều có thể nhìn thấy.

Trình xác thực phơi bày:

- `parse_frontmatter(text)` để tách metadata khỏi phần thân.
- `validate_skill_text(text, directory_name, allowed_runtime_extensions=())` để kiểm tra các trường bắt buộc, đặt tên, các phần mở rộng không xác định, sự hiện diện của phần thân và các giới hạn di động.
- `ValidationIssue` và `SkillReport` để trả về bằng chứng có cấu trúc thay vì một boolean mờ đục.
- `FrontmatterSyntaxError` cho đầu vào không thể được diễn giải một cách an toàn.

Bộ chọn phơi bày `TaskShape` và `select_primitives(task)`. Nó ánh xạ nhu cầu của một tác vụ tới code thông thường, hướng dẫn repository, kỹ năng, hook, subagent hoặc MCP tool.

Chạy lab:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/22-skills-and-agent-sdks
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Khối lệnh này yêu cầu một bản clone cục bộ và phải bắt đầu từ bất kỳ đâu bên trong bản clone đó để `git rev-parse --show-toplevel` có thể phân giải gốc repository.

Bản demo in ra JSON cho một kỹ năng di động hợp lệ, một kỹ năng mở rộng host, một gói không hợp lệ và một số quyết định về hình dạng tác vụ. Kiểm tra các mã lỗi. Một trình xác thực gói nên giải thích cách sửa một artifact mà không cần đoán thay cho tác giả.

### Thứ tự xác thực rất quan trọng

Xác thực các sự kiện cấu trúc rẻ tiền trước các quy tắc nội dung sâu hơn:

```figure
skill-validation-order
```

Thứ tự này ngăn chặn các lỗi thứ cấp làm lu mờ bất biến bị hỏng đầu tiên.

## Sử dụng nó

Trước khi viết một kỹ năng, hãy điền vào thẻ quyết định này:

| Câu hỏi | Nếu có | Nguyên thủy khả thi |
|---|---|---|
| Điều này có cần sự phán đoán của model có thể tái sử dụng qua nhiều bước không? | Quy trình ổn định nhưng các quyết định thay đổi | Kỹ năng |
| Điều này phải xảy ra mỗi khi một sự kiện kích hoạt? | Thiếu một lần thực thi là không thể chấp nhận được | Hook hoặc code ứng dụng |
| Model có cần một khả năng bên ngoài với đầu vào có kiểu không? | Thao tác nằm ngoài ngữ cảnh model | Tool hoặc MCP server |
| Công việc có cần ngữ cảnh, trạng thái hoặc quyền sở hữu bị cô lập không? | Một worker riêng biệt trả về kết quả bị giới hạn | Subagent |
| Hướng dẫn này có dành riêng cho một repository không? | Nó mô tả các lệnh và ràng buộc cục bộ | Hướng dẫn repository |
| Một tương tác là đủ? | Không cần vòng đời gói | Prompt |

Nhiều quy trình sản xuất sử dụng nhiều hơn một hàng. Thẻ này ngăn chặn một artifact giả vờ cung cấp mọi thuộc tính.

## Vận chuyển nó

Bài học này tạo ra gói `skill-contract-reviewer` dưới `outputs/`. Nó bao gồm:

- một `SKILL.md` di động đánh giá một gói kỹ năng được đề xuất;
- danh sách kiểm tra tham chiếu cho hợp đồng di động và lựa chọn nguyên thủy;
- một script xác thực tất yếu;
- các fixture hình dạng tác vụ bao gồm prompts, kỹ năng, tools, hooks, code thông thường và subagents.

Cài đặt gói đầy đủ, không chỉ tệp đầu vào của nó:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 scripts/install_skills.py /tmp/aiefs-skills --phase 13 --type skill
```

Trình cài đặt khóa học báo cáo từng kỹ năng Phase 13 được sao chép và ghi `/tmp/aiefs-skills/manifest.json`. Đích đến sạch sẽ này kiểm tra hình dạng gói; vòng lặp thành công đầu tiên ở trên kiểm tra việc khám phá và gọi lệnh trong một host thực tế.

Các bài học sau làm sâu sắc thêm từng giai đoạn vòng đời. Bài học 24 xây dựng việc khám phá và tiết lộ lũy tiến. Bài học 25 xây dựng chính sách gọi lệnh và định tuyến. Bài học 26 tách biệt quyền hạn khỏi sandboxing. Bài học 27 biến toàn bộ gói thành một artifact phát hành đã được đánh giá.

## Bài tập

1. Phân loại năm quy trình từ nhóm của riêng bạn bằng cách sử dụng `TaskShape`. Bảo vệ mọi trường hợp bạn chọn nhiều hơn một nguyên thủy.
2. Thêm các kiểm thử ranh giới chứng minh rằng giá trị `compatibility` 500 ký tự vượt qua và giá trị 501 ký tự thất bại như một lỗi đặc tả.
3. Thêm một phần mở rộng runtime vào danh sách cho phép. Viết một bài kiểm thử chứng minh cùng một tệp vẫn có thể phân biệt được với một kỹ năng chỉ dành cho di động.
4. Chia một prompt 400 dòng thành `SKILL.md`, một tham chiếu, một hợp đồng script và một mẫu đầu ra. Giữ cho mọi tệp chịu trách nhiệm cho một loại thông tin.
5. Thiết kế một phản hồi thất bại cho một kỹ năng tham chiếu đến một MCP tool không khả dụng. Không được âm thầm thay thế bằng một tool có quyền hạn rộng hơn.
6. Xem xét một kỹ năng hiện có và dán nhãn mọi câu là định tuyến, quy trình, chính sách, con trỏ tham chiếu hoặc hợp đồng đầu ra. Di chuyển bất cứ thứ gì không thuộc về nó.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|---|---|---|
| Kỹ năng Agent | "Một prompt đã lưu" | Một thư mục có thể khám phá gồm các hướng dẫn quy trình và tài nguyên tùy chọn |
| Cốt lõi di động | "Các trường mọi runtime chia sẻ" | Hợp đồng được xác định bởi đặc tả Kỹ năng Agent |
| Phần mở rộng Runtime | "Frontmatter bổ sung" | Cấu hình dành riêng cho host mà hành vi của nó yêu cầu một adapter tương thích |
| Kích hoạt | "Kỹ năng đã chạy" | Phần thân kỹ năng đã đi vào ngữ cảnh model có thể nhìn thấy; việc thực thi có thể đến sau |
| Phụ thuộc kỹ năng | "Nhập một kỹ năng khác" | Một cạnh gọi lệnh thông qua runtime với các kiểm tra tính khả dụng và chính sách |
| Hợp đồng Tool | "Một schema hàm" | Đầu vào, đầu ra, quyền hạn, tác dụng phụ, lỗi và bằng chứng cho một khả năng |

## Đọc thêm

- [Đặc tả Kỹ năng Agent](https://agentskills.io/specification) cho thư mục di động và hợp đồng frontmatter.
- [Các phương pháp hay nhất về Kỹ năng Agent](https://agentskills.io/skill-creation/best-practices) cho phạm vi, hướng dẫn và tổ chức tài nguyên.
- [OpenAI: Xây dựng kỹ năng](https://learn.chatgpt.com/docs/build-skills) cho hành vi khám phá và gọi lệnh Codex hiện tại.
- [Kỹ năng Claude Code](https://code.claude.com/docs/en/skills) cho các phần mở rộng về gọi lệnh, đối số, tool và ngữ cảnh được ủy quyền của một runtime.