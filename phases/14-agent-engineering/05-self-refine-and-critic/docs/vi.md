# Self-Refine và CRITIC: Cải thiện đầu ra lặp lại

> Self-Refine (Madaan và cộng sự, 2023) sử dụng một LLM trong ba vai trò — tạo (generate), phản hồi (feedback), tinh chỉnh (refine) — trong một vòng lặp. Mức tăng trung bình: +20 tuyệt đối trên 7 tác vụ. CRITIC (Gou và cộng sự, 2023) củng cố bước phản hồi bằng cách định tuyến việc xác minh thông qua các công cụ bên ngoài. Vào năm 2026, mô hình này được tích hợp trong mọi framework dưới dạng "evaluator-optimizer" (Anthropic) hoặc vòng lặp guardrail (OpenAI Agents SDK).

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu được ba prompt của Self-Refine (generate, feedback, refine) và giải thích tại sao lịch sử (history) lại quan trọng đối với prompt refine.
- Giải thích hiểu biết cốt lõi của CRITIC: LLM không đáng tin cậy trong việc tự xác minh nếu thiếu cơ sở thực tế bên ngoài.
- Triển khai vòng lặp Self-Refine bằng stdlib với lịch sử và một trình xác minh bên ngoài tùy chọn.
- Ánh xạ mô hình này vào quy trình "evaluator-optimizer" của Anthropic và các output guardrail của OpenAI Agents SDK.

## Vấn đề

Một agent tạo ra câu trả lời gần đúng. Có thể một dòng code bị lỗi cú pháp. Có thể một bản tóm tắt quá dài. Có thể một kế hoạch bỏ sót một trường hợp biên (edge case). Điều bạn muốn là: agent tự phê bình đầu ra của chính nó, sau đó sửa lỗi.

Self-Refine cho thấy điều này hiệu quả với một mô hình duy nhất, không cần dữ liệu huấn luyện, không cần RL. Nhưng có một vấn đề: LLM rất tệ trong việc tự xác minh các sự kiện khó. CRITIC đưa ra giải pháp — định tuyến bước xác minh thông qua các công cụ bên ngoài (tìm kiếm, trình thông dịch code, máy tính, trình chạy kiểm thử).

Cùng với nhau, hai bài báo này định nghĩa tiêu chuẩn mặc định cho năm 2026 về cải thiện lặp lại: tạo, xác minh (bên ngoài nếu có thể), tinh chỉnh, dừng lại khi trình xác minh thông qua.

## Khái niệm

### Self-Refine (Madaan và cộng sự, NeurIPS 2023)

Một LLM, ba vai trò:

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

Chi tiết quan trọng: `refine` nhìn thấy toàn bộ lịch sử — tất cả các đầu ra và phê bình trước đó — vì vậy nó không lặp lại sai lầm. Bài báo đã thực hiện thử nghiệm loại bỏ (ablation): nếu bỏ lịch sử, chất lượng giảm mạnh.

Điểm nhấn: Cải thiện tuyệt đối +20 trung bình trên 7 tác vụ (toán, code, viết tắt, hội thoại) bao gồm cả GPT-4. Không huấn luyện, không công cụ bên ngoài, chỉ một mô hình.

### CRITIC (Gou và cộng sự, arXiv:2305.11738, v4 tháng 2 năm 2024)

Điểm yếu của Self-Refine: bước phản hồi là một LLM tự chấm điểm chính nó. Đối với các tuyên bố thực tế, điều này không đáng tin cậy (một ảo giác thường trông có vẻ thuyết phục đối với chính mô hình đã tạo ra nó). CRITIC thay thế `feedback(task, output)` bằng `verify(task, output, tools)`, trong đó `tools` bao gồm:

- Công cụ tìm kiếm cho các tuyên bố thực tế.
- Trình thông dịch code để kiểm tra tính đúng đắn của code.
- Máy tính cho các phép tính số học.
- Trình xác minh chuyên biệt (unit test, trình kiểm tra kiểu, linter).

Trình xác minh tạo ra một phê bình có cấu trúc dựa trên kết quả của công cụ. Sau đó, trình tinh chỉnh sẽ điều chỉnh dựa trên phê bình này.

Điểm nhấn: CRITIC vượt trội hơn Self-Refine trong các tác vụ thực tế vì phê bình có cơ sở. Đối với các tác vụ không có trình xác minh bên ngoài (viết sáng tạo, định dạng), CRITIC trở về dạng Self-Refine.

### Điều kiện dừng

Hai hình thái phổ biến:

1. **Trình xác minh thông qua.** Kiểm thử bên ngoài trả về thành công. Ưu tiên khi có sẵn (unit test, trình kiểm tra kiểu, xác nhận guardrail).
2. **Không có phản hồi.** Mô hình nói "đầu ra ổn". Rẻ hơn nhưng không đáng tin cậy; hãy kết hợp với giới hạn số lần lặp tối đa.

Mặc định năm 2026: kết hợp cả hai. "Dừng nếu trình xác minh thông qua HOẶC mô hình nói ổn VÀ số lần lặp >= 2 HOẶC số lần lặp >= số lần lặp tối đa."

### Evaluator-Optimizer (Anthropic, 2024)

Bài đăng tháng 12 năm 2024 của Anthropic gọi đây là một trong năm mô hình quy trình làm việc. Hai vai trò:

- Evaluator: chấm điểm đầu ra và đưa ra phê bình.
- Optimizer: sửa đổi đầu ra dựa trên phê bình.

Lặp lại cho đến khi evaluator thông qua. Đây chính là Self-Refine/CRITIC trong khung tư duy của Anthropic. Chi tiết kỹ thuật quan trọng mà Anthropic bổ sung: prompt của evaluator và optimizer nên khác biệt đáng kể để mô hình không chỉ đơn thuần "đóng dấu xác nhận".

### Output guardrails của OpenAI Agents SDK

OpenAI Agents SDK triển khai mô hình này dưới dạng "output guardrails". Guardrail là một trình xác thực chạy trên đầu ra cuối cùng của agent. Nếu guardrail kích hoạt (nâng `OutputGuardrailTripwireTriggered`), đầu ra sẽ bị từ chối và agent có thể thử lại. Guardrails có thể gọi công cụ (kiểu CRITIC) hoặc là các hàm thuần túy (kiểu Self-Refine).

### Những cạm bẫy năm 2026

- **Vòng lặp "đóng dấu xác nhận" (Rubber-stamp loops).** Cùng một mô hình thực hiện tạo và phê bình với cùng một phong cách prompt sẽ dẫn đến kết quả "trông có vẻ ổn với tôi". Hãy sử dụng các prompt có cấu trúc khác nhau, hoặc một mô hình nhỏ, rẻ tiền hơn để phê bình.
- **Tinh chỉnh quá mức.** Mỗi lần tinh chỉnh đều làm tăng độ trễ và số lượng token. Hãy đặt ngân sách từ 1-3 lần; sau đó, hãy chuyển sang đánh giá của con người.
- **CRITIC trên các tác vụ tầm thường.** Nếu không có trình xác minh bên ngoài, CRITIC sẽ thoái hóa thành Self-Refine; đừng trả chi phí độ trễ cho một trình xác minh sơ sài.

```figure
self-refine
```

## Xây dựng

`code/main.py` triển khai Self-Refine và CRITIC trên một tác vụ đồ chơi: tạo một danh sách gạch đầu dòng ngắn dựa trên một chủ đề. Trình xác minh kiểm tra định dạng (3 gạch đầu dòng, mỗi dòng dưới 60 ký tự). CRITIC bổ sung một "trình xác minh thực tế" bên ngoài để phạt các ảo giác đã biết.

Các thành phần:

- `generate` — trình tạo theo kịch bản.
- `feedback` — tự phê bình kiểu LLM.
- `verify_external` — trình xác minh có cơ sở kiểu CRITIC.
- `refine` — viết lại đầu ra dựa trên lịch sử.
- Điều kiện dừng — trình xác minh thông qua hoặc tối đa 4 lần lặp.

Chạy thử:

```
python3 code/main.py
```

So sánh các lần chạy Self-Refine và CRITIC. CRITIC bắt được lỗi thực tế mà Self-Refine bỏ lỡ vì trình xác minh bên ngoài có cơ sở thực tế mà trình tự phê bình không có.

## Sử dụng

Evaluator-optimizer của Anthropic chính là mô hình này bằng ngôn ngữ thân thiện với Claude. Output guardrails của OpenAI Agents SDK có hình thái của CRITIC (guardrails có thể gọi công cụ). LangGraph cung cấp một node phản chiếu (reflection node) hoạt động giống như Self-Refine. Computer Use của Gemini 2.5 (Google) bổ sung một trình đánh giá an toàn theo từng bước, đây là một biến thể của CRITIC: mọi hành động đều được xác minh trước khi thực hiện.

## Triển khai

`outputs/skill-refine-loop.md` cấu hình một vòng lặp evaluator-optimizer dựa trên hình thái tác vụ, tính khả dụng của trình xác minh và ngân sách lặp. Phát ra các prompt cho trình tạo, trình đánh giá/xác minh và trình tối ưu hóa, cùng với chính sách dừng.

## Bài tập

1. Chạy tác vụ đồ chơi với max_iterations=1. CRITIC có còn giúp ích không?
2. Thay thế trình xác minh bên ngoài bằng một trình xác minh nhiễu (ngẫu nhiên 30% dương tính giả). Vòng lặp sẽ làm gì? Đây là thực tế năm 2026 của hầu hết các ngăn xếp guardrail.
3. Triển khai biến thể "generator-critic trên các mô hình khác nhau": mô hình lớn tạo, mô hình nhỏ phê bình. Nó có đánh bại được việc dùng cùng một mô hình không?
4. Đọc CRITIC Phần 3 (arXiv:2305.11738 v4). Nêu tên ba danh mục công cụ xác minh và đưa ra ví dụ cho mỗi loại.
5. Ánh xạ `output_guardrails` của OpenAI Agents SDK vào vai trò trình xác minh của CRITIC. SDK làm sai điều gì và làm đúng điều gì?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Self-Refine | "LLM tự sửa lỗi" | Vòng lặp tạo -> phản hồi -> tinh chỉnh trong một mô hình, có lịch sử |
| CRITIC | "Xác minh dựa trên công cụ" | Thay thế phản hồi bằng trình xác minh bên ngoài (tìm kiếm, code, tính toán, kiểm thử) |
| Evaluator-Optimizer | "Mô hình quy trình Anthropic" | Hai vai trò — evaluator chấm điểm, optimizer sửa đổi — lặp lại đến khi hội tụ |
| Output guardrail | "Kiểm tra hậu kỳ" | Trình xác thực của OpenAI Agents SDK chạy sau khi agent tạo đầu ra |
| Verify step | "Giai đoạn phê bình" | Quyết định then chốt: có cơ sở thực tế hay tự đánh giá |
| Refine history | "Những gì mô hình đã thử" | Các đầu ra + phê bình trước đó được thêm vào prompt tinh chỉnh; nếu bỏ đi, chất lượng sẽ sụp đổ |
| Rubber-stamp loop | "Lỗi tự đồng thuận" | Phê bình cùng prompt trả về "trông ổn"; khắc phục bằng các prompt có cấu trúc khác nhau |
| Stop condition | "Kiểm tra hội tụ" | Trình xác minh thông qua HOẶC không có phản hồi VÀ giới hạn lặp; không bao giờ dùng điều kiện đơn lẻ |

## Đọc thêm

- [Madaan và cộng sự, Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — bài báo gốc
- [Gou và cộng sự, CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — xác minh dựa trên công cụ
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — mô hình quy trình evaluator-optimizer
- [Tài liệu OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — output guardrails dưới dạng trình xác minh kiểu CRITIC