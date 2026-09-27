# Skill Invocation and Routing

> Invocation (kích hoạt) là một quyết định về thẩm quyền theo sau bởi một quyết định về mức độ liên quan. Một mô tả tốt giúp model lựa chọn; một chính sách tốt quyết định liệu lựa chọn đó có được phép hay không.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 24 (Skill Discovery and Progressive Disclosure)
**Time:** ~105 minutes

## Learning Objectives

- Phân biệt giữa kích hoạt chủ động bởi người dùng (explicit user invocation), kích hoạt ngầm bởi model (implicit model invocation), kích hoạt bởi ứng dụng (application invocation) và kích hoạt giữa các skill (skill-to-skill invocation).
- Mô hình hóa khả năng hiển thị với con người và khả năng được chọn bởi model như các chiều chính sách độc lập.
- Viết các mô tả định tuyến (routing descriptions) với các trigger tích cực và ranh giới cho các trường hợp gần đúng (near-miss).
- Tách biệt các giai đoạn: xác định tính hợp lệ (eligibility), lựa chọn (selection), kích hoạt (activation), gắn tham số (argument binding) và thực thi (execution) trong các trace và test.
- Điều chỉnh các trường kích hoạt đặc thù của runtime mà không trình bày chúng như các frontmatter có tính di động.

## The Problem

Bạn cài đặt một skill `database-migration`. Người dùng có thể chạy nó bằng tên, nhưng model cũng nhìn thấy mô tả của nó và chọn nó khi ai đó đặt một câu hỏi chung về cơ sở dữ liệu. Sau đó, skill này đề xuất thay đổi schema cho một tác vụ vốn chỉ cần giải thích.

Bạn thêm `user-invocable: false` với hy vọng chặn người dùng chạy nó thủ công. Ở một runtime khác, trường này bị bỏ qua. Bạn thêm `disable-model-invocation: true` với hy vọng skill sẽ biến mất hoàn toàn. Trong runtime hiểu trường này, người dùng vẫn có thể kích hoạt nó một cách chủ động.

Không có gì sai với tên các trường. Model mới là thứ sai. "Người dùng có thể nhìn thấy nó", "model có thể chọn nó", "ứng dụng có thể tải trước nó" và "các công cụ bên trong nó có thể thực thi" là những sự thật riêng biệt. Một biến boolean duy nhất gọi là `invocable` không thể diễn đạt tất cả chúng.

Định tuyến (routing) có một chế độ lỗi thứ hai. Nếu mô tả mơ hồ, nhiều skill sẽ trở nên hợp lý. Nếu mô tả bị nhồi nhét từ khóa, các tác vụ không liên quan sẽ kích hoạt chúng. Danh mục (catalog) là một giao diện xác suất: đủ gọn để vừa vặn, đủ cụ thể để định tuyến.

## The Concept

### Năm kênh có thể bắt đầu vòng đời

| Tác nhân | Hình thái kích hoạt | Cách dùng phổ biến | Rủi ro chính |
|---|---|---|---|
| Người dùng | Gọi tên skill trong UI hoặc prompt | Lựa chọn quy trình làm việc chủ động | Người dùng kỳ vọng sự sẵn có hoặc thẩm quyền mà host không cấp |
| Model hoặc agent tự hành | Chọn một mục trong danh mục từ ngữ cảnh tác vụ | Quy trình chuyên gia tự động | Định tuyến dương tính giả (false-positive) |
| Ứng dụng | Kích hoạt hoặc tải trước skill qua code runtime | Quy trình sản phẩm cố định | Sự phụ thuộc ẩn vào một host |
| Skill hoặc subagent khác | Yêu cầu một skill cụ thể làm phụ thuộc quy trình | Kết hợp (Composition) | Vòng lặp, thiếu phụ thuộc hoặc rò rỉ ngữ cảnh |
| Bộ công cụ đánh giá (harness) | Kích hoạt một skill cụ thể trong kịch bản cố định | Đo lường có thể lặp lại | Kiểm thử skill trong khi vô tình bỏ qua chính sách sản xuất đang nghiên cứu |

Đặc tả Agent Skills di động định nghĩa gói (package). Nó không chuẩn hóa một UI dòng lệnh, cờ định tuyến ngầm, API ứng dụng hay vòng đời subagent chung cho tất cả.

### Năm giai đoạn kích hoạt

```figure
skill-invocation-stages
```

Sử dụng các từ ngữ này một cách chính xác:

- **Eligible (Hợp lệ)** nghĩa là chính sách cho phép tác nhân này yêu cầu skill.
- **Selected (Được chọn)** nghĩa là người dùng đã gọi tên nó hoặc bộ định tuyến đánh giá nó là phù hợp.
- **Activated (Được kích hoạt)** nghĩa là các hướng dẫn của nó đã đi vào ngữ cảnh làm việc.
- **Executing (Đang thực thi)** nghĩa là agent bắt đầu công việc của model hoặc công cụ theo các hướng dẫn đó.
- **Completed (Hoàn thành)** nghĩa là đầu ra đáp ứng một kiểm tra thành công độc lập.

Một trace chỉ ghi lại `skill_used=true` sẽ che giấu ranh giới nơi xảy ra lỗi.

### Kích hoạt bởi người dùng và model tạo thành ma trận 2x2

| Người dùng có thể gọi | Model có thể gọi | Chế độ | Ví dụ phù hợp |
|:---:|:---:|---|---|
| Có | Có | Chia sẻ | Giải thích code, lập kế hoạch kiểm thử, xem xét tài liệu |
| Có | Không | Chỉ người dùng | Chuẩn bị xuất bản, xuất hóa đơn, kế hoạch dọn dẹp phá hủy |
| Không | Có | Chỉ model | Hướng dẫn phong cách nội bộ, tham chiếu miền, quy trình hỗ trợ tự động |
| Không | Không | Vô hiệu hoặc chỉ ứng dụng | Triển khai theo giai đoạn, gói đã lỗi thời, tải trước theo chương trình |

Ma trận này là một mô hình chính sách, không phải YAML tiêu chuẩn.

Một host hiện tại sử dụng `disable-model-invocation: true` cho hàng chỉ người dùng và `user-invocable: false` cho hàng chỉ model. Mặc định là cả hai. Một host khác sử dụng `agents/openai.yaml` với `allow_implicit_invocation: false` để giữ kích hoạt chủ động trong khi vô hiệu hóa lựa chọn ngầm. Đây là các bộ điều hợp (adapter) runtime. Các host không xác định có thể bỏ qua chúng.

Chi tiết gây nhầm lẫn rất quan trọng: `user-invocable: false` không có nghĩa là "model không thể sử dụng cái này". Nó loại bỏ kích hoạt trực tiếp từ người dùng trong host định nghĩa nó. `disable-model-invocation: true` không có nghĩa là "skill bị vô hiệu hóa". Nó loại bỏ lựa chọn do model khởi xướng trong khi vẫn giữ quyền truy cập chủ động của người dùng.

### Kích hoạt chủ động (Explicit) dựa trên định danh

Một kích hoạt chủ động cung cấp định danh trực tiếp:

```text
/release-readiness v2.4.0
```

hoặc:

```text
release-readiness check v2.4.0 without publishing
```

Các giao diện Codex hiện tại ghi lại `/skills` cho việc lựa chọn và tên skill thuần túy trong các yêu cầu kích hoạt chủ động. Claude Code ghi lại `/skill-name` và việc mở rộng tham số đặc thù của host. Cú pháp chính xác, khả năng hiển thị menu, quy tắc trích dẫn và mở rộng biến thuộc về host.

Một yêu cầu chủ động vẫn phải vượt qua chính sách. Việc gọi tên một skill không được phép bỏ qua các quyền bị thiếu, ràng buộc không gian làm việc, cổng phê duyệt hoặc sự cô lập runtime.

### Kích hoạt ngầm (Implicit) dựa trên mô tả

Đối với định tuyến ngầm, model ban đầu nhìn thấy metadata của danh mục thay vì toàn bộ nội dung. Do đó, mô tả chính là giao diện định tuyến của skill.

Yếu:

```yaml
description: Helps with releases.
```

Quá rộng:

```yaml
description: Use for release, version, package, build, deploy, publish, tag, changelog, GitHub, CI, or software tasks.
```

Có giới hạn:

```yaml
description: Inspect an already prepared release candidate and produce a readiness report. Use when the user asks whether a version, tag, package, or image is ready to publish; do not use for ordinary build failures or feature development.
```

Phiên bản có giới hạn chứa:

1. **Khả năng:** kiểm tra một ứng viên đã chuẩn bị.
2. **Đầu ra:** báo cáo mức độ sẵn sàng.
3. **Ranh giới tích cực:** hỏi liệu một artifact phát hành đã sẵn sàng chưa.
4. **Ranh giới tiêu cực:** các bản build thông thường và phát triển nằm ngoài phạm vi.

Các ranh giới tiêu cực hữu ích khi hai skill gần nhau chia sẻ từ vựng. Chúng không thay thế cho các đánh giá near-miss.

### Định tuyến là phân loại với tùy chọn từ chối (abstain)

Đối với một skill `s` và yêu cầu `x`, hãy tưởng tượng một điểm số định tuyến:

```text
score(s, x) = capability_match + trigger_match + context_match - exclusion_match - ambiguity_penalty
```

Việc chấm điểm chính xác có thể là quyết định của LLM thay vì tính toán số học. Nguyên tắc kỹ thuật vẫn giữ nguyên: lựa chọn phải vượt qua ngưỡng và một skill cạnh tranh. Khi bằng chứng yếu, hãy từ chối.

```figure
skill-routing-abstention
```

Đối với các skill có tác động cao, định tuyến ngầm có thể không phù hợp ngay cả với mô tả mạnh. Hãy sử dụng chính sách chỉ người dùng khi chi phí của một kết quả dương tính giả vượt quá sự tiện lợi của việc lựa chọn tự động.

### Tính hợp lệ (Eligibility) phải đi trước xếp hạng

Đừng chấm điểm mọi skill được tìm thấy, chọn kết quả khớp nhất rồi mới kiểm tra chính sách của skill đó. Một kết quả khớp hàng đầu bị chặn sẽ ngăn cản sai lầm một ứng viên hợp lệ có điểm thấp hơn được xem xét.

Sử dụng thứ tự này cho định tuyến ngầm:

1. Lọc các skill được tìm thấy theo tác nhân yêu cầu và bộ điều hợp host đang hoạt động.
2. Chỉ chấm điểm các ứng viên hợp lệ.
3. Chọn kết quả khớp hợp lệ mạnh nhất nếu nó vượt qua ngưỡng và các quy tắc về sự mơ hồ.
4. Từ chối khi không có ứng viên nào hợp lệ hoặc không có điểm số hợp lệ nào đủ mạnh.

Giả sử `incident-triage` đạt `0.80` nhưng phần mở rộng host của nó vô hiệu hóa kích hoạt bởi model. `incident-review` đạt `0.55` và cho phép kích hoạt bởi model. Bộ định tuyến nên đánh giá `incident-review` là ứng viên hợp lệ tốt nhất. Nó không nên chọn `incident-triage`, từ chối nó và dừng lại.

Thứ tự này cũng giúp các thay đổi chính sách không làm thay đổi ý nghĩa của điểm số liên quan. Tính hợp lệ xác định tập hợp lựa chọn. Mức độ liên quan xếp hạng tập hợp đó.

### Đánh giá định tuyến cần các trường hợp gần đúng (near misses)

Các trường hợp tích cực chứng minh độ nhớ (recall):

```json
{"prompt":"Is version 2.4.0 ready to publish?","expected":"release-readiness"}
```

Các trường hợp tiêu cực rõ ràng chứng minh độ chính xác cơ bản (precision):

```json
{"prompt":"Explain rotary position embeddings.","expected":null}
```

Các trường hợp gần đúng (near misses) phơi bày chất lượng ranh giới:

```json
{"prompt":"Why did today's package build fail?","expected":"build-diagnostics"}
```

Trường hợp gần đúng chia sẻ `package` và `build` với skill phát hành nhưng thuộc về nơi khác. Một tập hợp định tuyến chỉ gồm các trường hợp tích cực rõ ràng và tiêu cực không liên quan sẽ làm quá mức chất lượng.

### Tham số có ba biểu diễn

Một tham số kích hoạt vượt qua nhiều ranh giới:

```figure
skill-argument-boundaries
```

Tại mỗi ranh giới, hãy bảo toàn ý định mà không coi văn bản là code.

- Trình phân tích cú pháp của host quyết định cú pháp lệnh và trích dẫn.
- Skill nhận văn bản hoặc biến đã gắn theo quy tắc của host.
- Các hướng dẫn xác thực các giá trị bắt buộc và mặc định.
- Một tool call chuyển đổi các giá trị thành một schema có kiểu và xác thực lại chúng.

Đừng nội suy các tham số thô vào các lệnh shell. Ưu tiên một script được gọi với vector tham số hoặc một MCP tool có kiểu.

### Kích hoạt bởi ứng dụng là sự điều phối chủ động

Một sản phẩm có thể kích hoạt một skill vì quy trình làm việc của nó đã biết loại tác vụ. Ví dụ, một dịch vụ xem xét pull-request có thể tải trước `pull-request-risk-review` sau khi người dùng nhấn Review.

Điều này loại bỏ sự không chắc chắn trong định tuyến nhưng tạo ra sự phụ thuộc vào API runtime. Hãy giữ bộ điều hợp đó bên ngoài phần thân di động:

```figure
skill-host-adapter
```

Skill vẫn nên dễ hiểu khi được mở bởi một client tuân thủ khác.

### Kích hoạt skill-to-skill là một cạnh giống công cụ

Giả sử `release-readiness` yêu cầu `security-change-review` khi các file phụ thuộc thay đổi.

Bên gọi nên cung cấp:

- định danh skill mục tiêu;
- tác vụ có giới hạn và đường dẫn artifact;
- hợp đồng phản hồi mong đợi;
- lý do kích hoạt;
- phương án dự phòng nếu không khả dụng;
- độ sâu tối đa hoặc quy tắc vòng lặp.

```json
{
  "target_skill": "security-change-review",
  "task": "Review dependency changes in the candidate diff",
  "inputs": ["artifacts/release.diff"],
  "expected": "risk-report.json",
  "max_depth": 2
}
```

Skill thứ hai không được dán mù quáng vào skill thứ nhất. Host quyết định cách kích hoạt nó và liệu nó có chia sẻ ngữ cảnh, chạy trong một fork hay trả về qua kết quả công cụ hay không.

### Vòng đời ngữ cảnh là đặc thù của host

Sau khi kích hoạt, phần thân skill có thể vẫn nằm trong cuộc hội thoại, được tóm tắt trong quá trình nén hoặc chạy trong một ngữ cảnh được ủy quyền. Các quyền công cụ có thể kéo dài một lượt trong khi các hướng dẫn tồn tại lâu hơn. Một subagent có thể nhận skill mà không có toàn bộ lịch sử của cha.

Đừng viết một skill phụ thuộc vào giả định vòng đời vô hình. Hãy đặt các đầu ra bền vững vào file hoặc trạng thái có kiểu, làm cho việc nhập lại trở nên an toàn và nêu rõ những gì phải tải lại sau khi bị gián đoạn.

```markdown
On resume, read `artifacts/release-readiness.json` if it exists.
Revalidate the candidate commit before continuing.
Do not repeat an external write whose idempotency key is already recorded.
```

## Build It

`code/main.py` triển khai chính sách và định tuyến như các bộ điều hợp riêng biệt.

Mô hình bao gồm:

- `Actor` cho các tác nhân gọi là người dùng, model, agent tự hành, ứng dụng, skill và harness;
- `SkillMetadata` cho định danh định tuyến;
- `InvocationPolicy` cho ma trận người dùng/model;
- `InvocationRequest` và `InvocationDecision` cho các đầu vào và kết quả có thể truy vết;
- `CorePolicyAdapter` cho hành vi di động không có phần mở rộng host;
- `ExtensionPolicyAdapter` cho các trường runtime được công nhận;
- `build_invocation_matrix(policy)` cho chế độ xem 2x2;
- `route_request(skills, request, adapter)` cho lọc hợp lệ trước khi xếp hạng liên quan, lựa chọn và từ chối.

Chạy nó:

```bash
cd phases/13-tools-and-protocols/25-skill-invocation-and-routing
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Bản demo in ra một ma trận và các quyết định cho các kênh người dùng chủ động, model ngầm, agent tự hành, ứng dụng, kết hợp skill và harness. Kết quả bộ điều hợp mở rộng của nó cho thấy một kết quả khớp từ vựng hàng đầu bị chặn được loại bỏ trước khi một lựa chọn thay thế hợp lệ được xếp hạng. Nó cũng bao gồm các danh sách cho phép tên chính xác. Không cần API model. Bộ định tuyến tất định tồn tại để làm cho các ranh giới chính sách có thể kiểm tra được, không phải để khẳng định rằng khớp từ vựng tái tạo định tuyến model sản xuất.

### Tại sao các bộ điều hợp lõi và mở rộng lại tách biệt

Nếu một trình phân tích cú pháp gán ý nghĩa cho mọi trường frontmatter được quan sát, nó sẽ âm thầm thúc đẩy các quy ước runtime thành một tiêu chuẩn giả. Các bộ điều hợp tách biệt buộc bên gọi phải nêu tên ngữ nghĩa host nào đang hoạt động.

`CorePolicyAdapter` chỉ sử dụng chính sách do ứng dụng cung cấp. `ExtensionPolicyAdapter` nhận ra một tập hợp các trường host rõ ràng và ghi lại trường nào đã thay đổi quyết định.

## Use It

Viết một hợp đồng kích hoạt trước khi xuất bản một skill:

```yaml
actors:
  human: allow
  model: deny
  application: allow
  skill: deny
explicit_name: release-readiness
arguments:
  candidate: required
  publish: fixed_false
ambiguity: ask_user
missing_dependency: stop
context:
  durable_state: artifacts/release-readiness.json
  max_composition_depth: 2
```

Hợp đồng này là tài liệu thiết kế cho các bộ điều hợp và kiểm thử. Nó không phải là frontmatter `SKILL.md` di động trừ khi một tiêu chuẩn áp dụng nó một cách rõ ràng.

## Ship It

Bài học này tạo ra gói `skill-invocation-router`. Nó bao gồm tham chiếu mô hình kích hoạt, chính sách host ví dụ và một CLI không thực thi đánh giá một yêu cầu của người dùng, model, agent tự hành, ứng dụng, kết hợp skill hoặc harness và trả về một quyết định JSON với kênh, bộ điều hợp, điểm số và lý do.

CLI một yêu cầu là một thăm dò chính sách, không phải là đánh giá trigger đầy đủ. Sử dụng thiết kế tích cực và gần đúng được dán nhãn trong Bài 27 để tính toán số lượng nhầm lẫn, độ chính xác, độ nhớ và độ ổn định khi chạy lặp lại.

## Exercises

1. Tạo tất cả bốn hàng của ma trận người dùng/model và viết một trường hợp sử dụng hợp lệ cho mỗi hàng.
2. Thêm kích hoạt chỉ ứng dụng vào `CorePolicyAdapter`. Chứng minh rằng các tác nhân gọi là người dùng và model vẫn bị từ chối.
3. Viết mười trường hợp gần đúng cho một skill triển khai. Mỗi prompt phải chia sẻ từ vựng với skill trong khi thuộc về một quy trình làm việc khác.
4. Thêm một biên độ mơ hồ giữa hai điểm số định tuyến hàng đầu. Trả về `ask` khi biên độ quá nhỏ.
5. Thêm độ sâu kết hợp tối đa cho các yêu cầu skill-to-skill và phát hiện một vòng lặp hai skill.
6. Chạy cùng một tập hợp được dán nhãn qua các bộ điều hợp lõi và mở rộng. Giải thích mọi quyết định đã thay đổi.

## Key Terms

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|---|---|---|
| Explicit invocation | "Slash command" | Tác nhân cung cấp định danh skill trực tiếp, tuân theo chính sách |
| Implicit invocation | "Model chọn" | Bộ định tuyến chọn từ metadata danh mục hợp lệ dựa trên ngữ cảnh tác vụ |
| User-invocable | "Con người có thể dùng" | Menu đặc thù của host hoặc thuộc tính kích hoạt trực tiếp, không phải trường lõi |
| Model-invocable | "Agent có thể dùng" | Tính hợp lệ cho lựa chọn model ngầm theo chính sách host |
| Invocation adapter | "Trình phân tích frontmatter" | Code ánh xạ các trường và API của host vào một mô hình chính sách đã khai báo |
| Near miss | "Hard negative" | Một yêu cầu không kích hoạt giống với các đầu vào dự kiến của skill |
| Abstention | "Không skill nào được chọn" | Kết quả định tuyến có chủ đích khi bằng chứng vắng mặt hoặc mơ hồ |

## Further Reading

- [Tối ưu hóa mô tả skill](https://agentskills.io/skill-creation/optimizing-descriptions) cho các trigger tích cực, tính cụ thể và đánh giá.
- [Đánh giá skill](https://agentskills.io/skill-creation/evaluating-skills) cho thiết kế đánh giá trigger và đầu ra.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills) cho các điều khiển kích hoạt chủ động và ngầm của Codex hiện tại.
- [Claude Code skills](https://code.claude.com/docs/en/skills) cho `user-invocable`, `disable-model-invocation`, tham số và ngữ cảnh được ủy quyền của một host.