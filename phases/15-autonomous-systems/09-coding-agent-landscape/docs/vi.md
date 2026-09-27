# Bối cảnh các Coding Agent tự hành (2026)

> SWE-bench Verified đã tăng từ 4% lên 80,9% trong chưa đầy ba năm. Cùng một model Claude Sonnet 4.5 đạt 43,2% trên SWE-agent v1 và 59,8% trên Cline autonomous — phần scaffolding (khung hỗ trợ) bao quanh model hiện nay quan trọng không kém gì bản thân model đó. OpenHands (trước đây là OpenDevin) là nền tảng mã nguồn mở theo giấy phép MIT hoạt động tích cực nhất và vòng lặp CodeAct của nó thực thi các hành động Python trực tiếp trong sandbox thay vì gọi các tool JSON. Các con số tiêu đề thường che giấu một vấn đề về phương pháp luận: 161 trong số 500 tác vụ của SWE-bench Verified chỉ yêu cầu thay đổi 1–2 dòng code, trong khi SWE-bench Pro (các tác vụ yêu cầu thay đổi trên 10 dòng) chỉ đạt mức 23–59% đối với cùng các model tiên tiến nhất.

**Type:** Learn
**Languages:** Python (stdlib, so sánh CodeAct với JSON tool-call)
**Prerequisites:** Phase 14 · 07 (Sử dụng công cụ), Phase 15 · 01 (Agent tầm nhìn dài hạn)
**Time:** ~45 phút

## Vấn đề

"Coding agent nào tốt nhất" là một câu hỏi sai. Câu hỏi đúng phải là: trên một phân phối tác vụ phù hợp với công việc của tôi, với hệ thống scaffolding mà tôi sẽ chạy trong môi trường production, tôi nhận được độ tin cậy end-to-end là bao nhiêu?

Từ năm 2022 đến 2026, lĩnh vực này đã rút ra bài học rằng scaffolding — lớp truy xuất (retrieval layer), bộ lập kế hoạch (planner), sandbox, vòng lặp chỉnh sửa-xác minh (edit-verify loop), định dạng phản hồi — là thành phần chịu tải chính. Claude Sonnet 4.5 trên SWE-agent v1 đạt 43,2% trên SWE-bench Verified; cùng model đó bên trong scaffolding tự hành của Cline đạt 59,8%. Chênh lệch tuyệt đối 16,6 điểm với cùng trọng số. Base model chỉ là một thành phần; vòng lặp mới là sản phẩm hoàn chỉnh.

Vấn đề đi kèm là sự bão hòa của các benchmark đang che giấu các bước lùi. SWE-bench Verified gần như đã bão hòa, và phần đuôi của các tác vụ dễ (161 trong 500 tác vụ yêu cầu ≤2 dòng) đã kéo điểm số tổng thể lên cao. Chất lượng thực tế được đo lường tốt hơn trên các phân phối như SWE-bench Pro (thay đổi trên 10 dòng), nơi các hệ thống dẫn đầu vẫn chỉ nằm trong khoảng 23–59%.

## Khái niệm

### SWE-bench, tóm tắt trong một đoạn

SWE-bench (Jimenez và cộng sự) lấy các issue thực tế từ GitHub cùng với các bản vá (patch) chuẩn và yêu cầu agent tạo ra một bản vá giúp bộ test suite vượt qua. SWE-bench Verified (OpenAI, 2024) là một tập con gồm 500 tác vụ được con người kiểm duyệt, đã loại bỏ các tác vụ mơ hồ và bị lỗi. SWE-bench Pro là phiên bản kế nhiệm khó hơn — các tác vụ yêu cầu thay đổi trên 10 dòng, nơi các agent tiên tiến hiện nay chỉ đạt 23–59%.

### Đường cong phát triển từ 2022 → 2026 thực sự cho thấy điều gì

- **2022**: các model nghiên cứu đạt khoảng 4% trên SWE-bench thô.
- **2024**: GPT-4 + scaffolding kiểu Devin đạt khoảng 14%; SWE-agent đạt khoảng 12%.
- **2025**: Claude 3.5/3.7 Sonnet bên trong Aider và SWE-agent đẩy mức điểm lên phạm vi 40–55%.
- **2026**: Claude Sonnet 4.5 và các đối thủ tiên tiến đạt 70–80%+ trên SWE-bench Verified. Bảng xếp hạng của Epoch AI theo dõi điều này trực tiếp.

Độ dốc này đến từ ba nguồn cộng hưởng: base model tốt hơn, scaffolding tốt hơn (CodeAct, reflection, vòng lặp xác minh) và benchmark tốt hơn (Verified loại bỏ nhiễu).

### CodeAct so với JSON tool calls

OpenHands (All-Hands-AI, arXiv:2407.16741, trước đây là OpenDevin) đã đặt cược vào một kiến trúc cụ thể: thay vì model phát ra các JSON tool call để host giải mã và thực thi, model sẽ phát ra code Python và một kernel kiểu Jupyter sẽ chạy nó trong sandbox. Agent có thể lặp qua các tệp, xâu chuỗi các công cụ và tự bắt lỗi (exception) ngay trong một hành động.

Sự đánh đổi:

- **JSON tool calls**: mỗi hành động là một lượt; dễ kiểm tra (audit); khả năng kết hợp hạn chế; an toàn theo mặc định vì mỗi lệnh gọi đều đi qua một bộ xác thực rõ ràng.
- **CodeAct**: một hành động có thể là cả một chương trình; có tính kết hợp cao; yêu cầu một sandbox được bảo mật (OpenHands sử dụng Docker isolation); các chế độ lỗi bao gồm bất cứ thứ gì mà runtime của sandbox cho phép.

Cả hai kiến trúc đều đang được sử dụng trong production. CodeAct chiếm ưu thế trong các nền tảng mở (OpenHands, smolagents). JSON tool calls vẫn chiếm ưu thế trong các dịch vụ được quản lý (Anthropic Managed Agents, OpenAI Assistants) nơi nhà cung cấp kiểm soát trình thực thi.

### Các Scaffold trong bối cảnh năm 2026

| Scaffold | Giấy phép | Mô hình thực thi | Đặc điểm nổi bật |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | CodeAct trong Docker | Nền tảng mở hoạt động tích cực nhất; có thể phát lại luồng sự kiện |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | Scaffolding end-to-end đầu tiên cho SWE-bench |
| Aider | Apache-2 | edit-via-diff trong repo cục bộ | Scaffolding tối giản, độ ổn định hồi quy mạnh |
| Cline | Apache-2 | VS Code agent với chính sách công cụ | Scaffolding mở đạt điểm cao nhất trên Sonnet 4.5 |
| Devin (Cognition) | Độc quyền | Managed VM + planner | Danh mục sản phẩm "AI software engineer" đầu tiên |
| Claude Code | Độc quyền | Chế độ cấp quyền + routines | Bài 10 bao quát chi tiết về vòng lặp agent |

### Tại sao scaffolding lại chiếm ưu thế

Một quá trình coding là một quỹ đạo dài hạn (Bài 1). Độ tin cậy tích lũy qua từng bước. Ba nơi mà scaffolding mang lại giá trị:

1. **Truy xuất (Retrieval)**: tìm đúng tệp để đọc là nút thắt cổ chai thầm lặng. ACI của SWE-agent, file-index của OpenHands và repo-map của Aider đều giải quyết vấn đề này.
2. **Vòng lặp xác minh (Verifier loop)**: chạy test, đọc stack trace và thử lại là yếu tố tạo ra sự khác biệt hơn 10 điểm trên SWE-bench.
3. **Kiểm soát lỗi (Failure containment)**: một sandbox có khả năng rollback khi có lỗi sẽ ngăn chặn thiệt hại tích lũy. Cùng một model với và không có vòng lặp xác minh trông giống như hai sản phẩm khác nhau.

### Sự bão hòa của benchmark và phân phối thực tế

Các tác giả của OpenHands và Epoch AI đều lưu ý rằng SWE-bench Verified có phần đuôi dễ: 161 trong 500 tác vụ chỉ cần thay đổi 1–2 dòng. Điểm số cao một phần được thúc đẩy bởi phần đuôi này. SWE-bench Pro giới hạn ở các thay đổi trên 10 dòng và trả về điểm số trong khoảng 23–59% ngay cả đối với các hệ thống tiên tiến. Phân phối trong production của bạn gần như chắc chắn gần với Pro hơn là Verified.

Hệ quả cho việc chọn agent: hãy chạy một tập con giống như Pro từ chính backlog lỗi của bạn. Điểm số quan trọng là điểm số trên các tác vụ đại diện cho những gì bạn thực sự triển khai.

```figure
a5-scaffold-delta
```

## Sử dụng

`code/main.py` so sánh hai scaffold agent mẫu trên một phân phối tác vụ nhỏ cố định:

1. Một scaffold **JSON tool-call** thực hiện một hành động mỗi lượt.
2. Một scaffold **CodeAct** có thể phát ra một đoạn mã Python nhỏ cho mỗi hành động.

Cả hai đều sử dụng một "model" giả (các quy tắc tất định) để việc so sánh tách biệt được scaffold khỏi chất lượng model. Kết quả cho thấy scaffold CodeAct giải quyết được nhiều tác vụ hơn trong ít lượt hơn, với cái giá là phạm vi ảnh hưởng (blast radius) trên mỗi hành động lớn hơn.

## Triển khai

`outputs/skill-scaffold-audit.md` giúp bạn kiểm tra một scaffold coding-agent trước khi áp dụng: chất lượng truy xuất, sự hiện diện của bộ xác minh, tính cô lập của sandbox và mức độ phù hợp giữa benchmark với phân phối thực tế.

## Bài tập

1. Chạy `code/main.py`. Mỗi scaffold mất bao nhiêu lượt trên cùng một tập tác vụ? Phạm vi ảnh hưởng trên mỗi hành động của từng loại là bao nhiêu?

2. Đọc bài báo về OpenHands (arXiv:2407.16741). Bài báo lập luận rằng CodeAct vượt trội hơn JSON tool calls trên các tác vụ phức tạp. Hãy xác định một chế độ lỗi mà bài báo thừa nhận và viết một câu về thời điểm chế độ đó sẽ chiếm ưu thế trong production.

3. Chọn một tác vụ từ backlog lỗi của bạn yêu cầu thay đổi trên 10 dòng trên hai tệp. Ước tính xác suất thành công end-to-end cho một model tiên tiến dưới (a) JSON tool calls và (b) CodeAct. Giải thích lý do cho sự chênh lệch.

4. SWE-bench Verified có 161 tác vụ thay đổi 1–2 dòng trên một tệp duy nhất. Hãy xây dựng một điểm số loại trừ các tác vụ này. Bảng xếp hạng sẽ thay đổi như thế nào?

5. Đọc "Introducing SWE-bench Verified" (OpenAI). Giải thích phương pháp cụ thể được sử dụng để loại bỏ các tác vụ mơ hồ và nêu tên một danh mục mà quá trình kiểm duyệt này có thể bỏ sót.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| SWE-bench | "Benchmark lập trình" | Các issue GitHub thực tế với bản vá chuẩn và test suite |
| SWE-bench Verified | "Tập con đã làm sạch" | 500 tác vụ được con người kiểm duyệt, vẫn còn phần đuôi dễ |
| SWE-bench Pro | "Tập con khó hơn" | Thay đổi trên 10 dòng; các hệ thống tiên tiến đạt 23–59% |
| CodeAct | "Code-as-action" | Agent phát ra Python; kernel kiểu Jupyter thực thi trong sandbox |
| JSON tool call | "Function calling" | Mỗi hành động là một payload JSON cấu trúc được xác thực trước khi chạy |
| Scaffold | "Agent framework" | Truy xuất + lập kế hoạch + thực thi + vòng lặp xác minh bao quanh base model |
| ACI (Agent-Computer Interface) | "Định dạng của SWE-agent" | Tập lệnh được thiết kế cho công thái học của LLM, không phải shell cho người |
| Verifier loop | "Test-and-retry" | Chạy test, đọc kết quả, sửa bản vá; cải thiện độ tin cậy lớn nhất ngoài model |

## Đọc thêm

- [Jimenez và cộng sự — SWE-bench](https://www.swebench.com/) — benchmark và phương pháp luận gốc.
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — cách tập con được kiểm duyệt được xây dựng.
- [Wang và cộng sự — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — kiến trúc CodeAct và thiết kế luồng sự kiện.
- [Epoch AI — Bảng xếp hạng SWE-bench](https://epoch.ai/benchmarks) — điểm số được theo dõi trực tiếp.
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — khung đánh giá độ tin cậy của coding-agent dài hạn.