# Red-Teaming: PAIR và các cuộc tấn công tự động

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419). PAIR — Prompt Automatic Iterative Refinement — là phương pháp jailbreak hộp đen (black-box) tự động điển hình. Một LLM tấn công với system prompt red-team sẽ liên tục đề xuất các jailbreak cho một LLM mục tiêu, tích lũy các nỗ lực và phản hồi vào lịch sử trò chuyện của chính nó như một dạng phản hồi trong ngữ cảnh (in-context feedback). PAIR thường thành công trong vòng 20 truy vấn, hiệu quả hơn nhiều bậc so với GCG (tìm kiếm gradient ở cấp độ token của Zou và cộng sự) và không yêu cầu quyền truy cập white-box. PAIR hiện là một baseline tiêu chuẩn trong JailbreakBench (arXiv:2404.01318) và HarmBench, cùng với GCG, AutoDAN, TAP và Persuasive Adversarial Prompt.

**Type:** Build
**Languages:** Python (stdlib, mock PAIR loop against a toy target)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 14 (agent engineering)
**Time:** ~75 phút

## Mục tiêu học tập

- Mô tả thuật toán PAIR: system prompt của kẻ tấn công, tinh chỉnh lặp lại, phản hồi trong ngữ cảnh.
- Giải thích lý do tại sao PAIR hiệu quả hơn hẳn GCG khi mục tiêu là hộp đen.
- Kể tên bốn baseline tấn công tự động khác (GCG, AutoDAN, TAP, PAP) và nêu một đặc điểm phân biệt của từng loại.
- Mô tả các giao thức đánh giá JailbreakBench và HarmBench cũng như ý nghĩa của "tỷ lệ tấn công thành công" (attack success rate) trong mỗi giao thức.

## Vấn đề

Red-teaming từng là một hoạt động thủ công. Một số ít người kiểm thử chuyên gia xây dựng các prompt đối nghịch và theo dõi xem cái nào hiệu quả. Cách này không thể mở rộng: tỷ lệ tấn công thành công cần một mẫu thống kê, và mục tiêu là một "mục tiêu di động" sau mỗi lần phát hành mô hình. PAIR vận hành red-teaming như một bài toán tối ưu hóa với mục tiêu hộp đen.

## Khái niệm

### Thuật toán PAIR

Đầu vào:
- LLM mục tiêu T (mô hình chúng ta đang tấn công).
- LLM giám khảo J (chấm điểm xem phản hồi có phải là jailbreak hay không).
- LLM tấn công A (bộ tối ưu hóa red-team).
- Chuỗi mục tiêu G: "respond with [harmful instruction]."
- Ngân sách K (thường là 20 truy vấn).

Vòng lặp, với k từ 1 đến K:
1. A được cung cấp mục tiêu G và lịch sử các cặp (prompt, phản hồi) cho đến hiện tại.
2. A đưa ra một prompt mới p_k.
3. Gửi p_k đến T; nhận phản hồi r_k.
4. J chấm điểm (p_k, r_k) dựa trên mục tiêu.
5. Nếu điểm >= ngưỡng, dừng lại — đã tìm thấy jailbreak.
6. Nếu không, thêm (p_k, r_k) vào lịch sử của A; tiếp tục.

Kết quả thực nghiệm (NeurIPS 2023): Tỷ lệ tấn công thành công >50% đối với GPT-3.5-turbo, Llama-2-7B-chat; số truy vấn trung bình để thành công nằm trong khoảng 10-20.

### Tại sao PAIR hiệu quả

GCG (Zou và cộng sự 2023) tìm kiếm trên các hậu tố token đối nghịch bằng gradient; nó yêu cầu quyền truy cập white-box vào mô hình và tạo ra các hậu tố không thể đọc được. PAIR là hộp đen và tạo ra các cuộc tấn công bằng ngôn ngữ tự nhiên có thể chuyển đổi giữa các mô hình. Phản hồi trong ngữ cảnh của PAIR cho phép kẻ tấn công học hỏi từ mỗi lần bị từ chối; GCG không có cơ chế tương đương (mỗi lần cập nhật token mới phải tìm lại tiến trình trước đó).

### Các cuộc tấn công tự động liên quan

- **GCG (Zou và cộng sự 2023, arXiv:2307.15043).** Tìm kiếm gradient cấp độ token cho các hậu tố đối nghịch. White-box, có khả năng chuyển đổi, tạo ra các chuỗi không thể đọc được.
- **AutoDAN (Liu và cộng sự 2023).** Tìm kiếm tiến hóa trên các prompt, được hướng dẫn bởi một mục tiêu phân cấp.
- **TAP (Mehrotra và cộng sự 2024).** Cây tấn công với cắt tỉa (pruning) — phân nhánh nhiều lần chạy theo kiểu PAIR.
- **PAP (Zeng và cộng sự 2024).** Persuasive Adversarial Prompts — mã hóa các kỹ thuật thuyết phục của con người thành các mẫu prompt.

### JailbreakBench và HarmBench

Cả hai (2024) đều chuẩn hóa việc đánh giá:

- JailbreakBench (arXiv:2404.01318). 100 hành vi độc hại trên 10 danh mục chính sách của OpenAI. Tỷ lệ tấn công thành công (ASR) là chỉ số chính. Yêu cầu một giám khảo (GPT-4-turbo, Llama Guard, hoặc StrongREJECT).
- HarmBench (Mazeika và cộng sự 2024). 510 hành vi trên 7 danh mục, với các bài kiểm tra tác hại ngữ nghĩa và chức năng. So sánh 18 cuộc tấn công chống lại 33 mô hình.

ASR thường được báo cáo ở một ngân sách truy vấn cố định. Việc so sánh các cuộc tấn công đòi hỏi phải khớp ngân sách; ASR 90% ở 200 truy vấn không thể so sánh với 85% ASR ở 20 truy vấn.

### Tại sao điều này quan trọng đối với các triển khai năm 2026

Mọi phòng thí nghiệm tiên phong hiện nay đều chạy PAIR và TAP chống lại các mô hình sản xuất trước khi phát hành. Các quỹ đạo ASR xuất hiện trong thẻ mô hình (Bài 26) và các phụ lục về an toàn (Bài 18). Cuộc tấn công này không còn là điều lạ lẫm — nó là cơ sở hạ tầng tiêu chuẩn.

### Vị trí của bài này trong Phase 18

Bài 12 là nền tảng của tấn công tự động. Bài 13 (Many-Shot Jailbreaking) là một khai thác độ dài bổ sung. Bài 14 (ASCII Art / Visual) là một cuộc tấn công mã hóa. Bài 15 (Indirect Prompt Injection) là bề mặt tấn công sản xuất năm 2026. Bài 16 bao gồm các công cụ phòng thủ tương ứng (Llama Guard, Garak, PyRIT).

```figure
al-pair-loop
```

## Sử dụng

`code/main.py` xây dựng một vòng lặp PAIR mô phỏng. Mục tiêu là một bộ phân loại giả lập từ chối các prompt độc hại "rõ ràng" (bộ lọc từ khóa). Kẻ tấn công là một bộ tinh chỉnh dựa trên quy tắc thử nghiệm diễn giải lại, đóng vai và mã hóa. Giám khảo chấm điểm phản hồi. Bạn sẽ thấy kẻ tấn công thành công sau khoảng 5-15 lần lặp đối với bộ lọc từ khóa và thất bại đối với bộ lọc ngữ nghĩa.

## Triển khai

Bài học này tạo ra `outputs/skill-attack-audit.md`. Với một báo cáo đánh giá red-team, nó kiểm tra: những cuộc tấn công nào đã được chạy (PAIR, GCG, TAP, AutoDAN, PAP), với ngân sách bao nhiêu, với giám khảo nào, trên tập hành vi độc hại nào (JailbreakBench, HarmBench, nội bộ).

## Bài tập

1. Chạy `code/main.py`. Đo số truy vấn trung bình để thành công cho ba chiến lược tấn công tích hợp. Giải thích chiến lược nào khai thác giả định phòng thủ mục tiêu nào.

2. Triển khai chiến lược tấn công thứ tư (ví dụ: dịch sang ngôn ngữ khác, mã hóa base64). Báo cáo số truy vấn trung bình để thành công mới đối với mục tiêu bộ lọc từ khóa và mục tiêu bộ lọc ngữ nghĩa.

3. Đọc Chao và cộng sự 2023 Hình 5 (so sánh PAIR vs GCG). Mô tả hai kịch bản mà GCG được ưu tiên mặc dù PAIR có lợi thế về hiệu quả.

4. JailbreakBench báo cáo ASR đối với một tập mục tiêu cố định. Thiết kế một chỉ số bổ sung đo lường sự đa dạng của cuộc tấn công (phương sai trong các prompt thành công). Giải thích tại sao sự đa dạng lại quan trọng đối với đánh giá phòng thủ.

5. TAP (Mehrotra 2024) mở rộng PAIR với phân nhánh + cắt tỉa. Phác thảo một phần mở rộng kiểu TAP cho `code/main.py` và mô tả sự đánh đổi giữa chi phí tính toán và tỷ lệ thành công.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| PAIR | "jailbreak tự động" | Prompt Automatic Iterative Refinement; vòng lặp LLM tấn công + LLM giám khảo |
| GCG | "jailbreak gradient" | Tìm kiếm gradient cấp độ token white-box cho các hậu tố đối nghịch |
| Attack success rate (ASR) | "% jailbreak tại k truy vấn" | Chỉ số chính; phải được báo cáo kèm ngân sách truy vấn và danh tính giám khảo |
| Judge LLM | "bộ chấm điểm" | LLM chấm điểm xem phản hồi có thỏa mãn mục tiêu độc hại hay không |
| JailbreakBench | "bài đánh giá" | Tập hành vi độc hại tiêu chuẩn hóa với các danh mục được gắn thẻ |
| HarmBench | "bài đánh giá rộng hơn" | 510 hành vi, các bài kiểm tra tác hại chức năng + ngữ nghĩa |
| TAP | "cây tấn công" | PAIR với phân nhánh + cắt tỉa; ASR tốt hơn ở mức tính toán cao hơn |

## Đọc thêm

- [Chao và cộng sự — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — Bài báo PAIR, NeurIPS 2023
- [Zou và cộng sự — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — Bài báo GCG
- [Chao và cộng sự — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — đánh giá tiêu chuẩn hóa
- [Mazeika và cộng sự — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — đánh giá rộng hơn