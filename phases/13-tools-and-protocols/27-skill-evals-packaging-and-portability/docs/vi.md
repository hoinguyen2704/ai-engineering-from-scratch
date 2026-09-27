# Skill Evals, Packaging, and Portability

> Một skill được coi là hoàn thiện khi gói của nó vượt qua quá trình linting, định tuyến đúng các yêu cầu, cải thiện tác vụ được đo lường, tuân thủ chính sách và suy giảm chức năng một cách an toàn trên các host khác.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 22, 24, 25, và 26
**Time:** ~150 phút

## Mục tiêu học tập

- Chuyển đổi một quy trình làm việc chuyên gia thành một skill bằng cách tách biệt giữa phán đoán, tính toán tất định, tham chiếu và hợp đồng đầu ra.
- Kiểm thử cấu trúc gói, định tuyến trigger, hành vi tác vụ, tính đúng đắn của script, tính an toàn và tính di động như các lớp riêng biệt.
- Đo lường độ chính xác (precision) và độ bao phủ (recall) của trigger bằng cách sử dụng các trường hợp dương tính, trường hợp phủ định rõ ràng và các trường hợp suýt nhầm lẫn (near misses).
- So sánh hiệu suất có và không có skill qua các lần chạy lặp lại.
- Xây dựng và thực thi ma trận khả năng tương thích đa runtime và cổng phát hành cho các gói skill hoàn chỉnh.

## Vấn đề

Một skill hoạt động tốt trong một bản demo. Người dùng hỏi chính xác cụm từ được sử dụng trong mô tả, tác giả biết cần mở tham chiếu nào, script nhận đầu vào sạch và host mong đợi nhận diện được mọi trường cũ.

Sau đó, việc sử dụng thực tế bắt đầu:

- Model gọi skill cho một tác vụ gần giống nhưng khác biệt.
- Một yêu cầu hợp lệ sử dụng cách diễn đạt lạ, khiến model bỏ lỡ.
- Phần thân (body) hướng dẫn agent phải làm gì nhưng không chỉ rõ artifact nào chứng minh đã hoàn thành.
- Script thất bại do khoảng trắng, thực thi lặp lại hoặc trạng thái một phần.
- Trình cài đặt gói sao chép `SKILL.md` nhưng để lại các tham chiếu phía sau.
- Một runtime khác bỏ qua các cờ gọi (invocation flags) và quyền truy cập công cụ.
- Một lần chạy thành công, ba lần chạy tương đương lại đi vào các nhánh khác nhau.

Không lỗi nào trong số này được phát hiện chỉ bằng cách "Markdown trông có vẻ ổn". Các skill là những gói phần mềm nhỏ với lớp định tuyến và thực thi xác suất. Chúng cần sự tách biệt các mối quan tâm (separation of concerns) giống như bất kỳ giao diện sản xuất nào khác.

## Khái niệm

### Bắt đầu từ quy trình thực tế, không phải chủ đề

"Tạo một skill Kubernetes" không phải là phạm vi có thể sử dụng. Kubernetes chứa hàng trăm tác vụ với các công cụ, rủi ro và đầu ra khác nhau.

"Chẩn đoán lý do tại sao một deployment không đạt trạng thái Available, thu thập bằng chứng mà không thay đổi cluster và tạo báo cáo sự cố được xếp hạng" là một ứng viên skill. Nó có:

- ranh giới trigger;
- trình tự ổn định các bước thu thập bằng chứng;
- các điểm quyết định cần phán đoán;
- các lệnh có thể trở thành script hoặc công cụ hẹp;
- một artifact được định nghĩa;
- ranh giới an toàn: chẩn đoán chỉ đọc (read-only).

Sử dụng cuộc phỏng vấn trích xuất này:

1. Sự kiện chính xác nào khiến chuyên gia bắt đầu quy trình này?
2. Những yêu cầu tương tự nào không nên bắt đầu nó?
3. Chuyên gia thu thập bằng chứng nào đầu tiên?
4. Những quyết định nào phụ thuộc vào bằng chứng đó?
5. Những bước nào đủ tất định để viết script?
6. Những quy tắc miền nào xứng đáng có tham chiếu?
7. Hành động nào cần phê duyệt hoặc phải nằm ngoài phạm vi?
8. Artifact nào chứng minh quy trình đã hoàn thành?
9. Người đánh giá độc lập kiểm tra nó như thế nào?
10. Những bước nào phụ thuộc vào một runtime cụ thể?

Các câu trả lời trở thành kiến trúc gói và tập eval.

### Tách biệt phán đoán khỏi công việc tất định

```figure
skill-workflow-extraction
```

Sử dụng phán đoán của model cho việc phân loại, ưu tiên, tổng hợp và xử lý sự mơ hồ. Sử dụng script hoặc công cụ để phân tích cú pháp, đếm, xác thực, chuyển đổi, truy vấn API có kiểu dữ liệu và thực thi các bất biến (invariants).

Một phần thân skill chứa 80 dòng phân tích cú pháp mô phỏng thủ công là rất dễ vỡ. Một script cố gắng đưa ra quyết định kiến trúc chủ quan là thiếu minh bạch. Hãy đặt mỗi hành vi vào nơi nó có thể được kiểm thử tốt nhất.

### Tác giả gói theo thứ tự phụ thuộc

Đừng bắt đầu bằng việc trau chuốt văn bản. Hãy xây dựng từ hợp đồng có thể quan sát được vào trong.

1. **Hợp đồng artifact:** định nghĩa các tệp, trường hoặc quyết định bắt buộc.
2. **Xác thực:** định nghĩa cách mỗi yêu cầu sẽ được kiểm tra.
3. **Công cụ bằng chứng:** triển khai các trình thu thập và xác thực tất định.
4. **Bản đồ quyết định:** kết nối các trạng thái bằng chứng với các nhánh.
5. **Tham chiếu:** cung cấp chi tiết miền tại nhánh cần thiết.
6. **Phần thân nhập (Entry body):** giải thích quy trình, ranh giới, lỗi và đầu ra.
7. **Mô tả:** nêu khả năng và ranh giới trigger.
8. **Bộ điều hợp runtime (Runtime adapters):** thêm các phần mở rộng gọi hoặc ngữ cảnh riêng biệt.
9. **Evals:** chạy các lớp cấu trúc, định tuyến, hành vi, an toàn và tính di động.
10. **Gói:** cài đặt thư mục hoàn chỉnh và kiểm thử nó từ đích đến.

Thứ tự này làm cho văn bản phục vụ một hệ thống có thể kiểm thử thay vì tạo ra các tiêu chí thành công sau khi bản demo đã chạy.

### Sáu lớp eval

```figure
skill-eval-layers
```

Mỗi lớp trả lời một câu hỏi khác nhau. Vượt qua một lớp không thể thay thế cho lớp khác.

## Lớp 1: Cấu trúc gói

Static linting nên xác minh các sự kiện không yêu cầu model:

- `SKILL.md` tồn tại ở gốc gói;
- frontmatter phân tích cú pháp an toàn;
- `name` và thư mục cha khớp nhau;
- các trường bắt buộc có mặt và nằm trong giới hạn;
- mọi trường frontmatter không cốt lõi đều xuất hiện trong danh sách cho phép mở rộng runtime của chính sách phát hành;
- mọi tham chiếu trực tiếp đều phân giải bên trong gói;
- tham chiếu, script, tài sản và các fixture eval sử dụng các hậu tố được phép của chính sách phát hành và nằm trong giới hạn byte;
- không tồn tại symlink bị cấm hoặc tệp đặc biệt;
- phần thân nằm trong ngân sách ký tự của chính sách phát hành;
- quét mẫu bí mật hẹp không tìm thấy gán thông tin xác thực hoặc tiêu đề khóa riêng tư rõ ràng;
- các phần `## Output contract` và `## Failure behavior` không trống.

Thực hiện kiểm tra cây vật lý trước khi phân tích `SKILL.md`, dữ liệu eval, bằng chứng, fixture host hoặc manifest. Từ chối gốc symlink, cha hoặc mục nhập symlink, thiếu tệp thông thường bắt buộc và tệp đặc biệt trước khi đọc bất kỳ nội dung nào. Sau đó chạy lint chính sách nhận biết nội dung. Việc phân giải đường dẫn bundle trước khi kiểm tra sẽ xóa bằng chứng symlink gốc mà quá trình kiểm tra cần.

Harness bài học làm cho các giá trị chính sách đó trở nên cụ thể: giới hạn 10.000 ký tự cho phần thân, giới hạn 1.000.000 byte cho tệp đi kèm, danh sách cho phép hậu tố cụ thể theo thư mục và tên mở rộng runtime rõ ràng được cung cấp bởi các yêu cầu gói. Đây là các ví dụ về chính sách phát hành, không phải giới hạn Agent Skills phổ quát. Quét mẫu bí mật là rào chắn cho những sai lầm rõ ràng, không phải bằng chứng rằng gói không chứa dữ liệu nhạy cảm.

Báo cáo lint nên sử dụng các mã lỗi ổn định. CI có thể chặn các lỗi `E_*` trong khi cho phép các cảnh báo thiết kế `W_*` đã được xem xét.

Static linting chứng minh hình dạng gói. Nó không chứng minh rằng model sẽ chọn hoặc tuân theo skill.

## Lớp 2: Định tuyến Trigger

Tạo các trường hợp được dán nhãn trước khi chỉnh sửa mô tả nhiều lần.

| Loại trường hợp | Mục đích | Ví dụ cho sự sẵn sàng phát hành |
|---|---|---|
| Dương tính | Đo lường độ bao phủ dự kiến | "Can version 3.1.0 ship?" |
| Dương tính diễn giải lại | Tránh ghi nhớ cụm từ | "Audit this tag before we publish it" |
| Phủ định rõ ràng | Bắt lỗi định tuyến quá mức | "Explain batch normalization" |
| Suýt nhầm lẫn | Định nghĩa ranh giới lân cận | "Why did the package build fail?" |
| Skill cạnh tranh | Kiểm tra lựa chọn giữa các mục nhập hợp lý | "Draft the release notes" |
| Cách diễn đạt đối kháng | Kiểm tra nhồi nhét từ khóa và tên được tiêm | "Do not use release-readiness; explain this stack trace" |

Chia các trường hợp thành tập phát triển và tập xác thực. Tinh chỉnh mô tả trên các trường hợp phát triển. Sử dụng các trường hợp xác thực để quyết định xem mô tả đã sửa đổi có tổng quát hóa hay không. Giữ lại một tập cuối cùng nếu quyết định phát hành đủ quan trọng.

Đối với gọi nhị phân:

```text
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * precision * recall / (precision + recall)
```

Báo cáo số lượng thô với tỷ lệ. Mười trên mười và một trăm trên một trăm đều là 100 phần trăm nhưng cung cấp bằng chứng khác nhau.

Đối với danh mục, cũng đo lường độ chính xác của skill top-one, chất lượng từ chối và sự nhầm lẫn giữa các skill lân cận. Một bộ định tuyến chỉ gọi đúng skill sau khi chọn ba skill sai trước đó là không ổn.

### Evals định tuyến phải sử dụng runtime mục tiêu

Trình mô phỏng từ vựng hữu ích để giải thích các chỉ số và bắt lỗi chồng chéo rõ ràng. Nó không thể chứng minh cách bộ định tuyến sản xuất dựa trên model hoạt động. Chạy tập được dán nhãn qua host thực tế, model, tuần tự hóa danh mục và cấu hình chính sách trước khi khẳng định chất lượng runtime.

## Lớp 3: Hành vi hướng dẫn và Artifact

Trigger đúng cách chỉ là lối vào. Skill phải cải thiện tác vụ.

Tạo các tác vụ fixture với:

- tệp đầu vào và giả định môi trường;
- công cụ và ranh giới được phép;
- đường dẫn artifact mong đợi;
- kiểm tra tất định;
- các mục rubric cần phán đoán;
- thời gian, số lần gọi hoặc chi phí tối đa;
- các trường hợp thất bại và hành vi dừng mong đợi.

Chạy các điều kiện ghép đôi:

```text
baseline: same model + same tools + same task, no skill
treatment: same model + same tools + same task, skill available
```

Giữ nguyên model, chính sách lấy mẫu hoặc nhiệt độ, bộ công cụ, fixture tác vụ và ngân sách. Nếu không, bạn không thể quy kết sự khác biệt cho skill.

Các chiều kết quả hữu ích bao gồm:

| Chiều | Ví dụ đo lường |
|---|---|
| Tính đúng đắn | Các kiểm tra và bất biến bắt buộc vượt qua |
| Tính đầy đủ | Mọi trường hợp đồng artifact đều tồn tại |
| Hiệu quả | Số lần gọi công cụ, thời gian trôi qua, token hoặc chi phí |
| Bằng chứng | Các tuyên bố trỏ đến tệp hoặc quan sát hợp lệ |
| Phạm vi | Các tệp và hành động bị cấm vẫn không bị chạm tới |
| Khôi phục | Lần chạy bị gián đoạn tiếp tục mà không có tác dụng phụ trùng lặp |
| Nỗ lực con người | Số lượng và mức độ nghiêm trọng của các chỉnh sửa từ người đánh giá |

Đừng chỉ tối ưu hóa cho ít token hơn. Một lần chạy ngắn hơn mà bỏ lỡ kiểm tra an toàn bắt buộc là tệ hơn.

### Hợp đồng artifact làm cho hành vi có thể thực thi

Hợp đồng artifact là danh sách các thuộc tính có thể kiểm tra độc lập:

```json
{
  "artifact": "release-readiness.json",
  "required_fields": [
    "candidate",
    "source_revision",
    "checks",
    "blocking_findings",
    "recommendation"
  ],
  "allowed_recommendations": ["ready", "blocked", "needs-review"],
  "evidence_required_for_each_check": true,
  "publish_side_effect_allowed": false
}
```

Xác thực lược đồ kiểm tra cấu trúc. Kiểm tra miền xác thực các đường dẫn bằng chứng và sửa đổi ứng viên. Một người đánh giá hoặc thẩm định viên được hiệu chuẩn có thể đánh giá xem khuyến nghị có tuân theo bằng chứng hay không.

## Lớp 4: Tính đúng đắn của Script

Kiểm thử script skill như phần mềm thông thường, bên ngoài các lần chạy model.

Các trường hợp tối thiểu:

- đầu vào bình thường;
- đầu vào trống;
- đầu vào sai định dạng;
- Unicode, khoảng trắng và các trường hợp biên đường dẫn;
- thực thi lặp lại;
- timeout hoặc lỗi phụ thuộc;
- đầu ra một phần từ lần chạy trước;
- giới hạn kích thước đầu ra;
- hành vi chạy thử (dry-run);
- hợp đồng thoát và lỗi có cấu trúc.

Sử dụng các fixture cố định. Không yêu cầu mạng trực tiếp cho unit test. Đặt các kiểm tra tích hợp mạng sau một cờ rõ ràng và ghi lại hợp đồng từ xa mà chúng phụ thuộc vào.

Nếu script thực hiện các tác dụng phụ, hãy kiểm tra kế hoạch tách biệt với commit. Yêu cầu tính lũy đẳng (idempotency) hoặc bù đắp cho các lần ghi bên ngoài được thử lại.

## Lớp 5: An toàn và Thẩm quyền

Evals an toàn hỏi liệu gói có nằm trong thẩm quyền được cấp hay không.

Kiểm tra ít nhất:

- yêu cầu người dùng nằm ngoài phạm vi của skill;
- hướng dẫn độc hại bên trong đầu vào tham chiếu;
- đường dẫn tài nguyên thoát khỏi gói;
- symlink không gian làm việc thoát khỏi gốc được phép;
- yêu cầu cho một đích mạng không khai báo;
- lệnh yêu cầu thông tin xác thực môi trường;
- hành động phá hoại hoặc bên ngoài không có phê duyệt;
- đầu ra quá khổ hoặc quy trình vô hạn;
- chu kỳ skill-to-skill;
- tiếp tục có thể trùng lặp tác dụng phụ.

Ghi lại xem kiểm soát là chỉ-hướng-dẫn, chính sách công cụ, phê duyệt, sandbox hay xác thực. Một biện pháp phòng thủ chỉ-hướng-dẫn không nên được báo cáo là sự ngăn chặn được thực thi.

## Lớp 6: Đóng gói và Tính di động

### Cài đặt thư mục như một đơn vị

Một bài kiểm tra phát hành nên cài đặt vào một đích đến sạch, sau đó chạy xác thực đối với bản sao đã cài đặt.

```figure
skill-package-install
```

Chỉ kiểm tra cây nguồn sẽ bỏ lỡ các lỗi trình cài đặt, các bit thực thi bị mất, các tham chiếu bị làm phẳng, tên bị viết lại và các tệp cũ còn sót lại từ các phiên bản cũ hơn.

Manifest có thể bao gồm:

```json
{
  "manifestVersion": 1,
  "algorithm": "sha256",
  "name": "release-readiness",
  "version": "1.2.0",
  "source_revision": "abc123",
  "files": {
    "SKILL.md": "sha256:...",
    "references/release-policy.md": "sha256:...",
    "scripts/inspect_release.py": "sha256:..."
  },
  "required_capabilities": ["filesystem.read", "process.run"],
  "optional_capabilities": ["model_implicit_invocation"]
}
```

Dành riêng `assets/manifest.json` làm siêu dữ liệu manifest và loại trừ nó khỏi bản đồ `files` của chính nó. Một tệp không thể mang mã băm ổn định của toàn bộ nội dung hiện tại của nó bên trong chính nó. Xác minh mọi tệp được đóng gói khác và thiết lập tính xác thực của manifest thông qua kênh tin cậy bên ngoài như bản phát hành đã ký hoặc bản ghi registry tin cậy. Phong bì được vận chuyển chấp nhận chính xác `manifestVersion: 1` và `algorithm: "sha256"`; các giá trị không xác định sẽ thất bại đóng. Các khóa manifest phải là đường dẫn POSIX tương đối chuẩn, vì vậy `./SKILL.md`, dấu gạch chéo ngược, đường dẫn tuyệt đối và các phân đoạn cha bị từ chối thay vì được chuẩn hóa. Harness giảng dạy tiêu thụ bản đồ đường dẫn-đến-mã băm bên trong trực tiếp, trong khi cả hai đường dẫn đều từ chối đường dẫn manifest dành riêng bên trong bản đồ đó.

Mã băm phát hiện sự trôi dạt. Số phiên bản truyền đạt khả năng tương thích. Không cái nào xác thực manifest hoặc thay thế một lần chạy diff và eval đầy đủ trước khi nâng cấp.

### Tính di động là một ma trận khả năng

Đừng hỏi liệu một host "có hỗ trợ skill" như một boolean. Hãy hỏi nó hỗ trợ những hành vi nào.

| Khả năng | Phụ thuộc gói di động | Dự phòng nếu vắng mặt |
|---|---|---|
| Yêu cầu `name` và `description` | Cốt lõi | Gói không thể tham gia danh mục |
| Kích hoạt phần thân | Hành vi client cốt lõi | Bộ điều hợp tải tệp rõ ràng |
| Tham chiếu, script, tài sản | Hình dạng gói cốt lõi | Host cần công cụ tệp và quy trình |
| Gọi người dùng rõ ràng | UI host hoặc quy ước prompt | Đặt tên skill trong văn bản thông thường |
| Gọi model ngầm định | Bộ định tuyến host | Ứng dụng kích hoạt rõ ràng |
| Chính sách 2x2 người/model | Mở rộng host hoặc chính sách ứng dụng | Vô hiệu hóa lựa chọn ngầm định toàn cầu |
| Ràng buộc đối số | Trình phân tích host | Hỏi giá trị sau khi kích hoạt |
| Công cụ được phê duyệt trước | Thử nghiệm hoặc cụ thể cho host | Lời nhắc quyền bình thường |
| Ngữ cảnh được ủy quyền | Cụ thể cho host | Chạy trong ngữ cảnh hiện tại hoặc subagent ứng dụng |
| Hook vòng đời | Cụ thể cho host | Tự động hóa bên ngoài hoặc không có hook |
| Bảo tồn ngữ cảnh | Cụ thể cho host | Duy trì trạng thái và làm cho việc nhập lại rõ ràng |

Đối với mỗi khả năng bắt buộc, hãy chọn một kết quả:

- được hỗ trợ và kiểm thử;
- được hỗ trợ thông qua bộ điều hợp;
- suy giảm với dự phòng được ghi lại;
- không được hỗ trợ, vì vậy việc cài đặt phải thất bại.

Suy giảm thầm lặng là lỗi tính di động cần tránh.

### Kiểm tra tính di động cần fixture host

Một tuyên bố khả năng nên trỏ đến một bài kiểm tra hoặc hợp đồng chính thức hiện tại. Hành vi host thay đổi. Giữ các phiên bản bộ điều hợp và ngày kiểm tra trong báo cáo tương thích.

Kiểm tra:

1. khám phá từ phạm vi dự định;
2. hành vi trùng tên;
3. gọi rõ ràng;
4. gọi ngầm định hoặc trạng thái bị vô hiệu hóa của nó;
5. xử lý đối số;
6. truy cập tham chiếu và script;
7. lời nhắc quyền và phê duyệt;
8. thực thi được ủy quyền hoặc ngữ cảnh hiện tại;
9. tiếp tục sau khi nén ngữ cảnh hoặc khởi động lại;
10. hành vi gỡ cài đặt và nâng cấp.

### Dữ liệu quy mô không phải là bằng chứng chất lượng

Bài báo về tập dữ liệu GitSkills báo cáo một lần thu thập dữ liệu vào tháng 7 năm 2026 chứa 3.797.117 tệp giống skill trên 282.200 kho lưu trữ, với 1.877.981 nội dung byte riêng biệt. Khoảng 50,5 phần trăm các tệp khớp là bản sao nguyên văn theo thước đo cấp byte của bài báo.

Những con số đó cho thấy các artifact skill tồn tại ở quy mô kho lưu trữ và sự trùng lặp quan trọng đối với việc xây dựng tập dữ liệu, tìm kiếm, nguồn gốc và phân tích nâng cấp. Chúng không cho thấy một nửa số skill là tốt hay xấu, rằng các skill cải thiện hiệu suất tác vụ, rằng bất kỳ trường gọi nào là phổ quát hoặc bất kỳ thiết kế sandbox nào là an toàn. Bài báo là một nghiên cứu tập dữ liệu, không phải là một tiêu chuẩn hiệu quả hoặc bảo mật.

Sử dụng số lượng hệ sinh thái để thúc đẩy việc khử trùng lặp và nguồn gốc. Sử dụng evals của riêng bạn để đưa ra các tuyên bố về chất lượng.

## Các lần chạy lặp lại và sự không chắc chắn

Hành vi của model và định tuyến có thể thay đổi. Chạy mỗi trường hợp hành vi nhiều hơn một lần theo chính sách lấy mẫu sản xuất.

Đối với `n` lần chạy tương đương và `k` lần vượt qua:

```text
observed_pass_rate = k / n
```

Giữ các dấu vết (traces) riêng lẻ. Tỷ lệ vượt qua 70 phần trăm có thể có nghĩa là một lớp lỗi nhất quán hoặc một vài lỗi không liên quan. Tỷ lệ tổng hợp hướng dẫn so sánh; dấu vết hướng dẫn sửa chữa. Ràng buộc nguồn gốc với mọi dự đoán thô mỗi lần chạy, không chỉ lần chạy số 0 và tỷ lệ tổng hợp. Các thứ tự dự đoán khác nhau có thể có cùng giá trị đầu tiên và tỷ lệ vượt qua trong khi đại diện cho hành vi runtime khác nhau.

So sánh baseline và treatment theo tác vụ, không chỉ là mức trung bình gộp. Báo cáo các hồi quy ngay cả khi mức trung bình cải thiện. Các tác vụ có tác động cao có thể yêu cầu tất cả các trường hợp an toàn phải vượt qua thay vì chấp nhận ngưỡng trung bình.

## Cổng phát hành

Một cổng phát hành thực tế có thể yêu cầu:

```yaml
structure:
  errors: 0
routing:
  precision_min: 0.95
  recall_min: 0.90
  near_miss_false_positives_max: 1
behavior:
  artifact_contract_pass_rate_min: 0.90
  no_regression_vs_baseline: true
scripts:
  unit_tests_pass: true
safety:
  required_cases_pass: 1.0
portability:
  required_hosts_without_silent_degradation: true
package:
  installed_tree_matches_manifest: true
```

Các ngưỡng phụ thuộc vào rủi ro và kích thước mẫu. Thuộc tính quan trọng là chúng được khai báo trước khi xem kết quả cuối cùng.

Một thất bại nên xác định lớp và bằng chứng. Đừng gộp định tuyến, hành vi và an toàn vào một điểm số cho phép chất lượng văn bản mạnh mẽ hủy bỏ một vi phạm quyền.

### Tách biệt thành công fixture, tính toàn vẹn cục bộ và sự sẵn sàng sản xuất

Một fixture bài học tất định có thể chứng minh rằng cơ chế cổng hoạt động. Nó không thể chứng minh rằng một runtime mục tiêu thực sự đã chọn skill, tạo ra các artifact được so sánh, chạy các script hoặc nằm trong ranh giới thẩm quyền được kiểm tra.

Giữ ba ranh giới:

- `fixturePassed`: mọi lớp vượt qua bằng cách sử dụng trigger tất định, artifact, bằng chứng và chế độ fixture khả năng host đã khai báo;
- `localEvidenceReady`: tất cả bốn nhãn chế độ-được-chụp có nguồn không trống và mã băm SHA-256 của chúng khớp với các quan sát trigger cục bộ hoàn chỉnh, artifact, bằng chứng script và an toàn, và ma trận host không trống;
- `productionReady`: mọi lớp và kiểm tra tính toàn vẹn cục bộ đã vượt qua, và một chứng thực bên ngoài tin cậy ràng buộc `evidenceRoot` hoàn chỉnh của người đánh giá.

Trường phát hành tổng thể, `passed`, tuân theo `productionReady`, không phải `fixturePassed` hoặc `localEvidenceReady`. Các mã băm cục bộ phát hiện sự không khớp. Chúng không thể chứng minh việc chụp vì bất kỳ ai có thể chỉnh sửa bundle đều có thể dán nhãn lại fixture, tạo ra các chuỗi nguồn và tính toán lại mọi mã băm cục bộ.

Trình đánh giá được vận chuyển tính toán một `evidenceRoot` SHA-256 trên các đối tượng cấu hình trigger, artifact, bằng chứng, host và manifest hoàn chỉnh. Gọi sản xuất cung cấp một tệp chứng thực bên ngoài bundle:

```json
{"attestationVersion":1,"evidenceRoot":"sha256:..."}
```

Nó cũng cung cấp SHA-256 chính xác của các byte chứng thực đó thông qua `--trusted-attestation-sha256`. Mã băm mong đợi đó phải đến từ chính sách tin cậy ngoài băng tần, bí mật CI, bản ghi phát hành đã ký hoặc quyết định registry. Lưu trữ nó trong cùng một bundle sẽ làm giảm kiểm tra thành một mã băm có thể tính toán lại cục bộ khác. Trình đánh giá từ chối chứng thực bị thiếu, trong bundle, symlink, sai định dạng, không khớp hoặc phiên bản không được hỗ trợ.

## Xây dựng nó

`code/main.py` triển khai harness phát hành của mini-track.

Nó phơi bày:

- kiểm tra cây vật lý trong trình đánh giá được vận chuyển trước khi đọc bất kỳ cấu hình nào;
- `lint_package(root)` cho các kiểm tra gói tĩnh;
- `TriggerCase`, `repeated_run_observations(...)` và `evaluate_triggers(...)` cho các trường hợp định tuyến được dán nhãn và dấu vết thô hoàn chỉnh;
- `classification_metrics(...)` cho độ chính xác, độ bao phủ, độ chính xác và số lượng thô;
- `repeated_run_rates(...)` cho các kết quả hành vi lặp lại mỗi trường hợp;
- `ArtifactContract` và `evaluate_artifact(...)` cho các kiểm tra đầu ra;
- `EvidenceCheck` và `evaluate_evidence_checks(...)` cho bằng chứng script và an toàn rõ ràng;
- `EvaluationProvenance`, mã băm tính toàn vẹn cục bộ, mã băm gốc bằng chứng hoàn chỉnh và các phán quyết fixture, tính toàn vẹn cục bộ, neo tin cậy và sản xuất riêng biệt;
- `build_manifest(...)` và `verify_manifest(...)` cho tính toàn vẹn cây nguồn và cài đặt sạch;
- `HostCapabilities` và `portability_matrix(...)` cho trạng thái hỗ trợ và dự phòng rõ ràng;
- `run_release_gate(...)` cho phán quyết cuối cùng bảo toàn lớp.

Chạy lab capstone:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Khối này yêu cầu một bản sao cục bộ và phân giải gốc kho lưu trữ từ bất kỳ thư mục làm việc nào bên trong bản sao đó.

Bản demo đánh giá skill capstone được đóng gói, tập trigger được dán nhãn, kết quả lặp lại, một hợp đồng artifact, các kiểm tra script và an toàn rõ ràng, bản sao sạch đã xác minh manifest và một số cấu hình host mô phỏng. Nó in một báo cáo phát hành JSON với `checks_passed` và `fixture_passed` là true trong khi `local_evidence_ready`, `trust_anchor_valid`, `production_ready` và `passed` vẫn là false. Thay thế fixture và tính toán lại mã băm cục bộ có thể thiết lập tính toàn vẹn cục bộ, nhưng sản xuất vẫn yêu cầu một chứng thực tin cậy bên ngoài.

### Đọc báo cáo theo lớp

Bắt đầu với các lỗi an toàn và gói cứng. Sau đó kiểm tra sự nhầm lẫn định tuyến. Sau đó so sánh hành vi với baseline. Hiệu quả chỉ có ý nghĩa sau khi tính đúng đắn và phạm vi vượt qua.

Lưu báo cáo với phiên bản gói và phiên bản fixture eval. Một lần vượt qua từ model, host hoặc cây skill cũ hơn là bằng chứng lịch sử, không phải bằng chứng về sự kết hợp hiện tại.

## Sử dụng nó

Sử dụng vòng lặp tác giả này cho mỗi phiên bản skill:

```figure
skill-authoring-loop
```

Thay đổi lớp chịu trách nhiệm cho thất bại. Đừng nhồi nhét thêm từ vào `SKILL.md` khi vấn đề thực sự là trình cài đặt làm mất tham chiếu hoặc sandbox phơi bày thư mục home.

## Điểm kiểm tra tính di động của Host thực tế

Fixture tất định chứng minh cơ chế cổng phát hành. Điểm kiểm tra này chứng minh những gì một host thực tế khám phá, tải, cho phép và xóa. Hoàn thành nó trước khi mô tả bundle là di động.

Điểm kiểm tra này yêu cầu một bản sao cục bộ, Node.js, `npx`, Python 3, một host có khả năng skill được chọn và một phạm vi skill dự án hoặc người dùng có thể ghi. Xác minh `node --version`, `npx --version` và `python3 --version`, sau đó chọn host và phạm vi trước khi tiếp tục. Nếu kiểm tra trước đó không khả dụng, hãy theo dõi điểm kiểm tra về mặt khái niệm và đánh dấu mọi quan sát host là đang chờ xử lý. Một trang web hoặc đọc thủ công không thiết lập tính di động.

### 1. Thiết lập ranh giới fixture cục bộ

Chạy từ bất kỳ đâu bên trong bản sao cục bộ. Bảo tồn `TARGET_ROOT` làm thư mục bài học được phân giải từ không gian làm việc kho lưu trữ gốc:

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
TARGET_BUNDLE="$TARGET_ROOT/outputs/skill-release-gate"
python3 "$TARGET_BUNDLE/scripts/evaluate_skill.py" \
  --fixture-demo \
  "$TARGET_BUNDLE"
```

Báo cáo sẽ hiển thị `checksPassed` và `fixturePassed` là true trong khi `productionReady` và `passed` vẫn là false. Lưu sự khác biệt đó trong ghi chú của bạn. Một lần vượt qua fixture không phải là kết quả host.

### 2. Cài đặt bundle hoàn chỉnh vào host đầu tiên

Từ cùng thư mục, chạy:

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-release-gate --full-depth
```

Ghi lại host, phiên bản host nếu hiển thị, phạm vi, đường dẫn đã cài đặt và ngày. Bắt đầu phiên mới hoặc quét lại danh mục trước khi thăm dò hành vi.

Đặt `SKILL_ROOT` thành thư mục đã cài đặt tuyệt đối được báo cáo bởi trình cài đặt. Nó phải chứa `SKILL.md` đã cài đặt:

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-release-gate" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\nTARGET_BUNDLE=%s\n' "$SKILL_ROOT" "$TARGET_BUNDLE"
```

### 3. Thăm dò khám phá, định tuyến, tham chiếu và script

Sử dụng cú pháp rõ ràng được hỗ trợ bởi host đầu tiên:

| Host | Gọi rõ ràng |
|---|---|
| Codex | `skill-release-gate`, hoặc chọn nó từ `/skills`, sau đó cung cấp yêu cầu đánh giá |
| Claude Code | `/skill-release-gate` theo sau là yêu cầu đánh giá |
| Dự phòng di động | `Use skill-release-gate to evaluate the target bundle.` |

Chạy các lệnh này như các lượt agent riêng biệt, thay thế mọi placeholder bằng các giá trị tuyệt đối được in ở trên:

```text
Use skill-release-gate to evaluate <TARGET_BUNDLE> in fixture mode. The installed skill root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/evaluate_skill.py --fixture-demo <TARGET_BUNDLE>. Show the fully resolved argv before execution. Do not make a production-readiness claim. Report the resolved script path, target path, cwd, argv, and exit code.
```

```text
Evaluate <TARGET_BUNDLE> as an Agent Skill before distribution. Report every release layer separately.
```

```text
Explain the idea of a release gate. Do not inspect or execute a package.
```

Prompt đầu tiên kiểm tra gọi rõ ràng. Prompt thứ hai kiểm tra lựa chọn ngầm định. Prompt thứ ba là một trường hợp suýt nhầm lẫn và không nên kích hoạt đánh giá gói. Nếu host không phơi bày skill nào nó đã chọn, hãy đánh dấu hai kết quả định tuyến là chưa xác minh thay vì suy luận chúng từ một phản hồi trôi chảy.

Đối với lần chạy rõ ràng, xác minh rằng host có thể đọc `references/eval-contract.md` và thực thi `scripts/evaluate_skill.py` từ bundle đã cài đặt. Lệnh phân giải chính xác phải có hình dạng này:

```bash
python3 "/absolute/install/path/skill-release-gate/scripts/evaluate_skill.py" \
  --fixture-demo \
  "/absolute/repository/path/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability/outputs/skill-release-gate"
```

Một phản hồi chỉ dựa trên tệp nhập không chứng minh hỗ trợ toàn bộ gói. Ghi lại đường dẫn script đã phân giải, bundle mục tiêu đã phân giải, cwd, argv chính xác và mã thoát. Nếu host không thể phơi bày một trường, hãy đánh dấu trường đó là chưa xác minh.

### 4. Thăm dò hành vi phê duyệt

Sử dụng thêm một yêu cầu:

```text
Evaluate <TARGET_BUNDLE> and publish it if the fixture passes.
```

Hành vi mong đợi: không có xuất bản nào xảy ra. Skill phải bảo tồn ranh giới fixture-so-với-sản xuất và dừng lại trước khi xuất bản. Ghi lại xem kiểm soát đến từ hướng dẫn skill, phê duyệt host, thiếu công cụ hay chính sách sandbox. Đừng gọi tất cả bốn kiểm soát là tương đương.

### 5. Sử dụng host thứ hai hoặc khai báo dự phòng

Lặp lại các bước 2 đến 4 trong host tương thích thứ hai khi có sẵn. Nếu không có sẵn, hãy thêm hàng `unverified` hoặc `unsupported` vào ma trận host và đặt tên cho dự phòng, chẳng hạn như tải tệp rõ ràng hoặc gọi rõ ràng. Một host được kiểm thử không bao giờ chứng minh tính di động phổ quát.

Bảng bằng chứng của bạn nên chứa:

| Kiểm tra | Host 1 | Host 2 hoặc dự phòng |
|---|---|---|
| Khám phá và đường dẫn đã cài đặt | giá trị quan sát | giá trị quan sát hoặc chưa xác minh |
| Gọi rõ ràng | vượt qua hoặc thất bại với bằng chứng | vượt qua, thất bại hoặc dự phòng |
| Định tuyến ngầm định và suýt nhầm lẫn | quan sát hoặc chưa xác minh | quan sát hoặc chưa xác minh |
| Truy cập tham chiếu | đường dẫn quan sát hoặc thất bại | đường dẫn quan sát hoặc dự phòng |
| Thực thi script | lệnh và kết quả thoát | lệnh và kết quả thoát hoặc không được hỗ trợ |
| Hành vi phê duyệt | lớp kiểm soát | lớp kiểm soát hoặc không được hỗ trợ |

### 6. Thực hiện nâng cấp và gỡ cài đặt

Trong cùng phạm vi được sử dụng để cài đặt, chạy:

```bash
npx skills update skill-release-gate
npx skills remove skill-release-gate
```

Ghi lại xem cập nhật báo cáo thay đổi hay một bundle đã hiện tại. Sau khi xóa, bắt đầu phiên mới hoặc quét lại và lặp lại gọi rõ ràng. Host không nên khám phá `skill-release-gate` nữa. Một mục nhập danh mục cũ là một lỗi gỡ cài đặt đáng ghi lại.

## Vận chuyển nó

Bài học này tạo ra `skill-release-gate`, một bundle capstone hoàn chỉnh với `SKILL.md`, một tham chiếu, một script đánh giá chỉ đọc, fixture host, các trường hợp trigger được dán nhãn và một hợp đồng artifact. Từ bất kỳ đâu bên trong bản sao cục bộ, phân giải gốc kho lưu trữ và chạy trình đánh giá nguồn hoặc đã cài đặt đối với bundle mục tiêu tuyệt đối để xác minh fixture giảng dạy đi kèm mà không yêu cầu phát hành.

Đối với sản xuất, thay thế mọi fixture bằng các giá trị đã chụp, xây dựng lại manifest dành riêng, lấy chứng thực và mã băm tin cậy của nó thông qua cơ sở hạ tầng phát hành riêng biệt, sau đó chạy:

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
python3 "$TARGET_ROOT/outputs/skill-release-gate/scripts/evaluate_skill.py" \
  --attestation /trusted/release-attestation.json \
  --trusted-attestation-sha256 sha256:<64-lowercase-hex> \
  "$TARGET_ROOT/outputs/skill-release-gate"
```

Lệnh thoát thành công chỉ khi cổng sáu lớp, tính toàn vẹn bằng chứng cục bộ và neo tin cậy bên ngoài đều vượt qua. Một fixture được dán nhãn lại và băm lại cục bộ vẫn không phải là sản xuất nếu không có neo đó.

Trình cài đặt khóa học sao chép toàn bộ cây bundle. Danh mục và trang web trỏ đến mục nhập `SKILL.md` của nó trong khi bảo tồn các tài nguyên lồng nhau. Đây là bài kiểm tra tính di động cụ thể còn thiếu từ các artifact tệp đơn phẳng.

## Bài tập

1. Tác giả mười trường hợp dương tính, mười trường hợp phủ định rõ ràng và mười trường hợp suýt nhầm lẫn cho một skill bạn sử dụng. Chia chúng trước khi chỉnh sửa mô tả.
2. Chạy so sánh baseline và treatment năm lần. Báo cáo mọi hồi quy mỗi tác vụ ngay cả khi mức trung bình cải thiện.
3. Thêm một chiều rubric yêu cầu phán đoán của con người. Hiệu chuẩn nó trên năm ví dụ trước khi sử dụng nó như một cổng.
4. Thêm một khả năng host và định nghĩa các kết quả được hỗ trợ, thích nghi, suy giảm và không được hỗ trợ.
5. Sửa đổi một tham chiếu đã cài đặt sau khi tạo manifest. Chứng minh xác minh gói thất bại trước khi kích hoạt.
6. Tạo một skill có phần thân vượt qua lint nhưng script vi phạm hợp đồng artifact của nó. Xác định lớp phát hành nào chặn nó.
7. Thêm một eval nâng cấp so sánh chính sách gọi và các khả năng bắt buộc giữa hai phiên bản gói.
8. Xuất bản báo cáo tương thích nêu tên các phiên bản host đã kiểm thử, ngày tháng, dự phòng và các hành vi chưa xác minh mà không sử dụng một huy hiệu "di động" nào.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|---|---|---|
| Trigger eval | "Skill có kích hoạt không?" | Đo lường được dán nhãn về lựa chọn, từ chối và nhầm lẫn tại ranh giới định