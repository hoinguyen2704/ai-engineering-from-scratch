# Lập kế hoạch với HTN và Tìm kiếm Tiến hóa (Evolutionary Search)

> Lập kế hoạch biểu tượng (symbolic planning) xử lý các trường hợp cần chứng minh tính đúng đắn của kế hoạch. Tìm kiếm mã nguồn tiến hóa (evolutionary code search) xử lý các trường hợp mà hàm thích nghi (fitness function) có thể kiểm tra được bằng máy. ChatHTN (2025) và AlphaEvolve (2025) cho thấy những gì mỗi phương pháp mở khóa khi kết hợp với LLM.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 02 (ReWOO và Plan-and-Execute)
**Time:** ~75 phút

## Mục tiêu học tập

- Giải thích Hierarchical Task Networks (HTN): tác vụ (tasks), phương pháp (methods), toán tử (operators), điều kiện tiên quyết (preconditions), và hiệu ứng (effects).
- Mô tả vòng lặp lai của ChatHTN — tìm kiếm biểu tượng với cơ chế phân rã dự phòng bằng LLM.
- Giải thích vòng lặp tiến hóa của AlphaEvolve và lý do tại sao nó chỉ hoạt động với một trình đánh giá theo chương trình (programmatic evaluator).
- Triển khai một trình lập kế hoạch HTN đơn giản và một trình tìm kiếm tiến hóa đơn giản bằng stdlib.

## Vấn đề

ReWOO (Bài 02), Plan-and-Execute, và ReAct bao phủ hầu hết các tác vụ lập kế hoạch của agent. Có hai trường hợp mà chúng không xử lý tốt:

1. **Các kế hoạch có tính đúng đắn có thể chứng minh được.** Lập lịch, định tuyến đường bay, quy trình tuân thủ — kế hoạch phải đúng ngay từ cấu trúc. Một kế hoạch LLM trôi chảy nhưng đôi khi gây ra ảo giác (hallucination) ở một bước nào đó là không thể chấp nhận được.
2. **Tối ưu hóa với hàm thích nghi có thể kiểm tra bằng máy.** Nhân ma trận, heuristic lập lịch, các bước biên dịch — mục tiêu không phải là "một kế hoạch đúng" mà là "kế hoạch tốt nhất".

Lập kế hoạch HTN và AlphaEvolve giải quyết hai vấn đề khác nhau này. Cả hai đều sử dụng LLM như bộ khuếch đại, không phải là sự thay thế.

## Khái niệm

### Hierarchical Task Networks

Một HTN bao gồm:

- **Tác vụ (Tasks)** — phức hợp (cần phân rã) và nguyên thủy (có thể thực thi trực tiếp).
- **Phương pháp (Methods)** — các cách để phân rã một tác vụ phức hợp thành các tác vụ con, kèm theo các điều kiện tiên quyết.
- **Toán tử (Operators)** — các hành động nguyên thủy với điều kiện tiên quyết và hiệu ứng.
- **Trạng thái (State)** — một tập hợp các sự kiện (facts).

Lập kế hoạch: với một tác vụ mục tiêu và trạng thái ban đầu, tìm cách phân rã thành các toán tử nguyên thủy mà các điều kiện tiên quyết của chúng được thỏa mãn theo trình tự.

HTN ra đời trước LLM và vẫn là tài liệu tham khảo cho các kế hoạch có tính đúng đắn có thể chứng minh được.

### ChatHTN (Gopalakrishnan et al., 2025)

ChatHTN (arXiv:2505.11814) đan xen HTN biểu tượng với các truy vấn LLM:

1. Cố gắng phân rã tác vụ phức hợp hiện tại bằng các phương pháp hiện có.
2. Nếu không có phương pháp nào áp dụng được, hãy hỏi LLM: "bạn sẽ phân rã `task` trong trạng thái `s` như thế nào?"
3. Chuyển đổi phản hồi của LLM thành các tác vụ con ứng viên.
4. Xác thực dựa trên lược đồ toán tử; loại bỏ các phân rã không hợp lệ.
5. Đệ quy.

Tuyên bố chính của bài báo: mọi kế hoạch được tạo ra đều có tính đúng đắn có thể chứng minh được vì các gợi ý của LLM chỉ đóng vai trò là các phân rã ứng viên, không bao giờ là chỉnh sửa kế hoạch trực tiếp. Lớp biểu tượng đảm bảo tính đúng đắn; LLM mở rộng thư viện phương pháp.

Học phương pháp trực tuyến (OpenReview `gwYEDY9j2x`, bản cập nhật 2025) bổ sung một bộ học máy giúp khái quát hóa các phân rã do LLM tạo ra bằng hồi quy — cắt giảm tần suất truy vấn LLM lên đến 75%.

### AlphaEvolve (Novikov et al., 2025)

AlphaEvolve (arXiv:2506.13131, DeepMind, tháng 6 năm 2025) là một thực thể khác biệt: tìm kiếm mã nguồn tiến hóa được điều phối bởi một tập hợp (ensemble) Gemini 2.0 Flash/Pro.

Vòng lặp:

1. Bắt đầu với một chương trình hạt giống + một trình đánh giá theo chương trình (trả về điểm thích nghi).
2. Tập hợp các LLM đề xuất các đột biến (mutations).
3. Chạy các đột biến thông qua trình đánh giá.
4. Giữ lại kết quả tốt nhất; tiếp tục đột biến.

Các thành tựu đã công bố:

- Cải tiến đầu tiên so với Strassen cho phép nhân ma trận phức hợp 4x4 trong 56 năm qua (48 phép nhân vô hướng).
- Thu hồi 0,7% tài nguyên tính toán của Google thông qua heuristic lập lịch Borg.
- Tăng tốc 32% FlashAttention trên khối lượng công việc tiên phong.

Ràng buộc cứng: hàm thích nghi phải có khả năng kiểm tra bằng máy. Tìm kiếm tiến hóa trên các câu trả lời dạng văn xuôi sẽ không hội tụ.

### Khi nào sử dụng phương pháp nào

| Lớp vấn đề | Sử dụng | Tại sao |
|---------------|-----|-----|
| Lập lịch với ràng buộc cứng | HTN + ChatHTN | Tính đúng đắn có thể chứng minh |
| Tối ưu hóa trình biên dịch | AlphaEvolve | Hàm thích nghi kiểm tra bằng máy |
| Thực thi tác vụ đa bước | ReAct / ReWOO | LLM trong vòng lặp, không có đảm bảo hình thức |
| Cải thiện mã nguồn với kiểm thử | AlphaEvolve | Kiểm thử đóng vai trò là trình đánh giá |
| Tự động hóa dựa trên chính sách | HTN | Điều kiện tiên quyết mã hóa chính sách |

### Khi nào mô hình này thất bại

- **HTN không có toán tử.** Nếu không có lược đồ điều kiện tiên quyết/hiệu ứng, tuyên bố về tính đúng đắn sẽ sụp đổ. Việc "LLM gợi ý phân rã" của ChatHTN yêu cầu lược đồ phải loại bỏ các bước đi không hợp lệ.
- **AlphaEvolve không có trình đánh giá thực tế.** "Hỏi LLM xem mã có tốt hơn không" không phải là một hàm thích nghi. Trình đánh giá phải mang tính tất định và nhanh chóng.
- **Kỹ thuật quá mức (Over-engineering).** Hầu hết các tác vụ của agent không cần đến cả hai. Hãy ưu tiên ReAct hoặc ReWOO trước.

```figure
htn-tree-expand
```

## Xây dựng

`code/main.py` triển khai hai mô hình đơn giản:

- Một trình lập kế hoạch HTN stdlib với các toán tử, phương pháp, điều kiện tiên quyết, hiệu ứng và một `LLMFallback` kích hoạt khi không có phương pháp nào khớp với tác vụ phức hợp. "LLM" ở đây là một bộ phân rã theo kịch bản để trình lập kế hoạch chạy ngoại tuyến.
- Một trình tìm kiếm tiến hóa stdlib trên các chương trình số học: phát triển các biểu thức có đầu ra tối thiểu hóa `|f(x) - target|` trên một tập kiểm thử. Trình đánh giá mang tính tất định.

Chạy nó:

```
python3 code/main.py
```

Dấu vết (trace) cho thấy trình lập kế hoạch HTN phân rã một tác vụ phức hợp (với cơ chế dự phòng LLM giữa kế hoạch) và vòng lặp tiến hóa hội tụ về biểu thức mục tiêu.

## Sử dụng

- **Trình lập kế hoạch HTN** — `pyhop`, `SHOP3`, hoặc tự xây dựng cho việc thực thi chính sách đặc thù của miền.
- **ChatHTN** — mã nguồn nghiên cứu; mô hình (biểu tượng + dự phòng LLM) có thể chuyển đổi sạch sẽ sang bất kỳ trình lập kế hoạch HTN nào.
- **AlphaEvolve** — bài báo của DeepMind; mô hình (tập hợp + trình đánh giá) có thể tái lập. OpenEvolve và các bản fork mã nguồn mở tương tự đang xuất hiện.
- **Framework cho Agent** — chưa có framework nào hỗ trợ sẵn HTN hoặc AlphaEvolve. Hãy xây dựng nó như một subagent hoặc một worker chạy ngầm.

## Triển khai

`outputs/skill-hybrid-planner.md` tạo ra một khung lập kế hoạch lai (HTN hoặc tiến hóa) với vai trò của LLM được xác định phạm vi rõ ràng.

## Bài tập

1. Mở rộng trình lập kế hoạch HTN với khả năng quay lui (backtracking): khi điều kiện hậu quyết của một toán tử thất bại trong thời gian chạy, hãy quay lại và thử phương pháp tiếp theo.
2. Thêm bộ nhớ đệm phương pháp LLM vào ChatHTN: khi LLM phân rã tác vụ `T` trong mẫu trạng thái `P`, hãy lưu kết quả. Kiểm tra thư viện phương pháp trước trong lần gọi tiếp theo.
3. Thay đổi trình đánh giá tìm kiếm tiến hóa thành một bộ kiểm thử thực tế. Phát triển một hàm sắp xếp vượt qua 20 trường hợp kiểm thử; báo cáo số thế hệ để hội tụ.
4. Đọc các ghi chú thiết kế trình đánh giá của AlphaEvolve. Thiết kế một trình đánh giá cho một miền mà bạn quan tâm (tối ưu hóa truy vấn SQL, tối thiểu hóa bộ kiểm thử, YAML triển khai).
5. Kết hợp: sử dụng HTN để phân rã một tác vụ phức hợp thành các tác vụ con, sau đó sử dụng tìm kiếm tiến hóa trên toán tử nguyên thủy của mỗi tác vụ con. Nó tỏa sáng ở đâu, và ở đâu thì nó trở nên quá mức cần thiết?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| HTN | "Trình lập kế hoạch phân cấp" | Phân rã tác vụ với toán tử, điều kiện tiên quyết, hiệu ứng |
| Method | "Quy tắc phân rã" | Cách chia nhỏ tác vụ phức hợp thành các tác vụ con |
| Operator | "Hành động nguyên thủy" | Bước cụ thể với điều kiện tiên quyết và hiệu ứng |
| ChatHTN | "LLM + HTN" | Trình lập kế hoạch biểu tượng hỏi LLM khi không có phương pháp khớp |
| AlphaEvolve | "Tìm kiếm mã nguồn tiến hóa" | Tập hợp LLM đột biến mã; trình đánh giá tất định lựa chọn |
| Fitness function | "Trình đánh giá" | Điểm số tất định, có thể kiểm tra bằng máy trên các đầu ra |
| Online method learning | "Phân rã LLM được lưu trữ" | Lưu trữ + khái quát hóa các kế hoạch LLM để cắt giảm chi phí truy vấn |

## Đọc thêm

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — trình lập kế hoạch lai biểu tượng + LLM
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — tìm kiếm mã nguồn tiến hóa với các đột biến LLM
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — khi nào nên dùng trình lập kế hoạch so với một vòng lặp đơn giản