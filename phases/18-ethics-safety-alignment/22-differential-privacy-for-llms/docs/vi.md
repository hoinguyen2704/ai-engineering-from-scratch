# Differential Privacy cho LLM

> DP-SGD vẫn là tiêu chuẩn — các cập nhật gradient được tiêm nhiễu cung cấp các đảm bảo (epsilon, delta) chính thức. Chi phí về tính toán, bộ nhớ và hiệu năng là rất đáng kể; tinh chỉnh DP hiệu quả về tham số (LoRA + DP-SGD) là cấu hình phổ biến trong năm 2025 (ACM 2025). Hai luồng bằng chứng đang mâu thuẫn: suy diễn thành viên (membership inference) dựa trên canary (Duan và cộng sự, 2024) báo cáo thành công hạn chế đối với các mô hình ngôn ngữ; trích xuất dữ liệu huấn luyện (Carlini và cộng sự, 2021; Nasr và cộng sự, 2025) khôi phục được lượng lớn dữ liệu nguyên văn. Giải quyết (arXiv:2503.06808, tháng 3 năm 2025): khoảng cách nằm ở những gì được đo lường — các canary được chèn vào so với dữ liệu "dễ trích xuất nhất". Các thiết kế canary mới cho phép thực hiện MIA dựa trên loss mà không cần mô hình bóng (shadow models) và mang lại cuộc kiểm toán DP không tầm thường đầu tiên cho một LLM được huấn luyện trên dữ liệu thực với các đảm bảo DP thực tế. Các giải pháp thay thế: PMixED (arXiv:2403.15638) — dự đoán riêng tư tại thời điểm suy luận thông qua hỗn hợp các chuyên gia (mixture of experts) trên các phân phối token tiếp theo; tạo dữ liệu tổng hợp DP (Google Research 2024). Tấn công mới nổi: Đảo ngược Differential Privacy thông qua phản hồi LLM — rò rỉ điểm tin cậy (confidence-score).

**Type:** Build
**Languages:** Python (stdlib, minh họa tiêm nhiễu DP-SGD và bộ kế toán ε-δ)
**Prerequisites:** Phase 01 · 09 (lý thuyết thông tin), Phase 10 · 01 (huấn luyện mô hình lớn)
**Time:** ~60 phút

## Mục tiêu học tập

- Định nghĩa (epsilon, delta)-differential privacy và nêu công thức DP-SGD.
- Giải thích sự mâu thuẫn giai đoạn 2024-2025: canary MIA so với trích xuất dữ liệu huấn luyện đưa ra các bức tranh khác nhau.
- Mô tả PMixED và lý do tại sao dự đoán riêng tư tại thời điểm suy luận là một giải pháp thay thế cho huấn luyện DP.
- Mô tả cuộc tấn công Đảo ngược Differential Privacy thông qua phản hồi LLM.

## Vấn đề

LLM ghi nhớ dữ liệu. Carlini và cộng sự năm 2021 đã chỉ ra rằng các mô hình ngôn ngữ thương mại tái tạo nguyên văn văn bản huấn luyện theo yêu cầu. DP là cơ chế phòng thủ chính thức: huấn luyện sao cho đầu ra có thể chứng minh là không nhạy cảm với bất kỳ ví dụ huấn luyện đơn lẻ nào. Bằng chứng giai đoạn 2024-2025 cho thấy DP-SGD là cần thiết nhưng các giá trị ε được triển khai có thể không khớp với mô hình đe dọa.

## Khái niệm

### (ε, δ)-differential privacy

Một thuật toán ngẫu nhiên M là (ε, δ)-DP nếu với bất kỳ hai tập dữ liệu nào khác nhau ở một ví dụ và bất kỳ sự kiện S nào:
P(M(D) in S) <= e^ε * P(M(D') in S) + δ.

Giải thích: phân phối đầu ra đủ gần (được tham số hóa bởi ε) để sự đóng góp của bất kỳ cá nhân nào không thể bị suy luận một cách đáng tin cậy, ngoại trừ với xác suất δ.

### DP-SGD

Abadi và cộng sự năm 2016. Công thức tiêu chuẩn:
1. Lấy mẫu một mini-batch.
2. Tính toán gradient cho từng ví dụ.
3. Cắt (clip) mỗi gradient của ví dụ theo ngưỡng C.
4. Tổng hợp các gradient đã cắt và thêm nhiễu Gaussian với độ lệch chuẩn σ * C.
5. Sử dụng tổng nhiễu để cập nhật tham số.

Chi phí quyền riêng tư được theo dõi bởi một bộ kế toán (Moments Accountant, Rényi DP accountant). Các giá trị ε được báo cáo trong tài liệu LLM thay đổi rất nhiều tùy theo mô hình đe dọa, độ nhạy của dữ liệu và mục tiêu hiệu năng; không có giá trị ε "an toàn" mặc định chung. Các ví dụ đã công bố nằm trong khoảng ε ≈ 1–10 trong một số thiết lập huấn luyện LLM, nhưng đây chỉ là minh họa — không phải là mặc định được khuyến nghị. ε thấp hơn thường đòi hỏi nhiều nhiễu hơn và có thể làm tăng mất mát hiệu năng.

### LoRA + DP-SGD

DP-SGD toàn phần cho một mô hình tiên phong là không khả thi. LoRA (Hu và cộng sự 2022) giới hạn các cập nhật gradient vào một adapter nhỏ, giảm lưu trữ gradient cho mỗi ví dụ. LoRA + DP-SGD là cấu hình phổ biến năm 2025. Các đảm bảo DP áp dụng cho adapter; mô hình cơ sở được giữ cố định.

### Sự mâu thuẫn giai đoạn 2024-2025

Hai luồng bằng chứng:

- **Canary MIA (Duan và cộng sự 2024).** Chèn các canary duy nhất vào dữ liệu huấn luyện, đo lường xem kẻ tấn công suy diễn thành viên có thể xác định chúng hay không. Báo cáo thành công hạn chế trên các mô hình ngôn ngữ. Cho thấy MIA rất khó.
- **Trích xuất dữ liệu huấn luyện (Carlini 2021, Nasr và cộng sự 2025).** Nhắc mô hình với một tiền tố; đo lường xem nó có khôi phục được văn bản nguyên văn từ quá trình huấn luyện hay không. Báo cáo sự ghi nhớ đáng kể. Cho thấy MIA dễ dàng theo nghĩa liên quan.

Giải quyết tháng 3 năm 2025 (arXiv:2503.06808): cả hai đo lường những thứ khác nhau. MIA hỏi "ví dụ e có trong D không?" trên các canary được chèn vào. Trích xuất hỏi "tôi có thể khôi phục được gì từ D?". Ví dụ "dễ trích xuất nhất" là điều quan trọng đối với quyền riêng tư; các canary báo cáo thấp hơn thực tế vì chúng không được tối ưu hóa để có thể trích xuất.

Các thiết kế canary mới. MIA dựa trên loss không cần mô hình bóng. Cuộc kiểm toán DP không tầm thường đầu tiên của một LLM trên dữ liệu thực với các đảm bảo DP thực tế.

### Các giải pháp thay thế cho huấn luyện DP

- **PMixED (arXiv:2403.15638).** Dự đoán riêng tư tại thời điểm suy luận. Hỗn hợp các chuyên gia trên các phân phối token tiếp theo; mỗi chuyên gia thấy một phần dữ liệu huấn luyện; tổng hợp thêm nhiễu cho DP. Tránh hoàn toàn việc huấn luyện DP.
- **Tạo dữ liệu tổng hợp DP (Google Research 2024).** Tinh chỉnh LoRA với DP-SGD, lấy mẫu dữ liệu tổng hợp, huấn luyện bộ phân loại hạ nguồn trên dữ liệu tổng hợp đó.

Cả hai đều tránh chi phí hiệu năng của việc huấn luyện DP toàn phần với cái giá là một mô hình đe dọa khác.

### Đảo ngược Differential Privacy thông qua phản hồi LLM

Cuộc tấn công mới nổi năm 2025. Sử dụng điểm tin cậy của mô hình được huấn luyện DP làm oracle để tái định danh các cá nhân. Ngay cả khi đầu ra không bị rò rỉ, các phân phối tin cậy vẫn có thể.

Phòng thủ: không tiết lộ điểm tin cậy, hoặc cắt bớt/lượng tử hóa chúng trước khi tiết lộ. Đây là một yêu cầu bổ sung ngoài huấn luyện (ε, δ)-DP.

### Vị trí trong Phase 18

Bài 20-21 là về thiên kiến/công bằng. Bài 22 là về quyền riêng tư. Bài 23 là về nguồn gốc dữ liệu thông qua watermarking. Bài 27 bao gồm lớp quản lý nguồn gốc dữ liệu theo quy định.

```figure
an-dp-clip-noise
```

## Sử dụng

`code/main.py` mô phỏng DP-SGD trên một tập dữ liệu phân loại nhị phân đồ chơi. Bạn có thể quét hệ số nhiễu σ và chuẩn cắt C và theo dõi ngân sách (ε, δ) cùng chi phí độ chính xác. Một "cuộc tấn công canary" chèn một ví dụ huấn luyện duy nhất và đo lường xem kiểm tra log-loss có thể phát hiện ra nó trước và sau DP hay không.

## Triển khai

Bài học này tạo ra `outputs/skill-dp-audit.md`. Với một tuyên bố DP trên một triển khai mô hình ngôn ngữ, nó kiểm toán: các giá trị (ε, δ), bộ kế toán được sử dụng, giao thức đánh giá MIA và liệu các vectơ tiết lộ điểm tin cậy đã được đánh giá hay chưa.

## Bài tập

1. Chạy `code/main.py`. Quét σ trong {0.5, 1.0, 2.0} và báo cáo sự đánh đổi giữa (ε, δ) và độ chính xác. Xác định điểm mà tại đó hiệu năng sụp đổ.

2. Triển khai việc chèn canary và kiểm tra log-loss. Đo tỷ lệ phát hiện trước và sau DP-SGD tại σ = 1.0.

3. Đọc Nasr và cộng sự 2025 về trích xuất dữ liệu huấn luyện. Tại sao thành công của việc trích xuất không sụp đổ dưới mức ε trung bình? Điều này ngụ ý gì về MIA như một phương pháp đánh giá?

4. Thiết kế một triển khai sử dụng PMixED (arXiv:2403.15638) hoạt động hoàn toàn tại thời điểm suy luận. Mô hình đe dọa mà PMixED giải quyết nhưng DP-SGD không giải quyết được là gì?

5. Phác thảo cuộc tấn công Đảo ngược DP thông qua phản hồi LLM. Thiết kế một biện pháp đối phó hạn chế rò rỉ điểm tin cậy và ước tính chi phí triển khai của nó.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| DP | "(ε, δ)-differential privacy" | Quyền riêng tư chính thức: phân phối đầu ra gần nhau khi thay đổi tập dữ liệu lân cận |
| DP-SGD | "SGD tiêm nhiễu" | Cắt gradient + thêm nhiễu Gaussian; huấn luyện DP tiêu chuẩn |
| LoRA + DP-SGD | "tinh chỉnh riêng tư hiệu quả" | DP-SGD trên các adapter hạng thấp; cấu hình tiêu chuẩn 2025 |
| MIA | "suy diễn thành viên" | Tấn công xác định xem một ví dụ có nằm trong dữ liệu huấn luyện hay không |
| Canary | "ví dụ watermark được chèn" | Ví dụ huấn luyện duy nhất được sử dụng để đo lường rò rỉ DP |
| PMixED | "hỗn hợp suy luận riêng tư" | DP tại thời điểm suy luận thông qua hỗn hợp chuyên gia trên các phân phối token tiếp theo |
| Đảo ngược DP | "tấn công rò rỉ điểm tin cậy" | Tấn công sử dụng điểm tin cậy của mô hình làm oracle để tái định danh |

## Đọc thêm

- [Abadi và cộng sự — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133) — thuật toán huấn luyện DP tiêu chuẩn
- [Carlini và cộng sự — Trích xuất dữ liệu huấn luyện (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805) — bài báo kinh điển về trích xuất
- [Duan và cộng sự — Canary MIA trên LLM (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841) — MIA với thành công hạn chế
- [Kowalczyk và cộng sự — Kiểm toán DP cho LLM (arXiv:2503.06808, tháng 3 năm 2025)](https://arxiv.org/abs/2503.06808) — giải quyết sự mâu thuẫn
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638) — dự đoán riêng tư tại thời điểm suy luận