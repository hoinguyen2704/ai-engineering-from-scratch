# Scaling Laws

> Bài báo của Kaplan năm 2020 cho biết: mô hình càng lớn, loss càng thấp. Bài báo của Hoffmann năm 2022 cho biết: bạn đã huấn luyện chưa đủ. Compute được chia thành hai nhóm — tham số (parameters) và token — và việc phân bổ không hề hiển nhiên.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 phút

## Vấn đề

Khi bạn có C FLOPs cho compute huấn luyện và muốn có mô hình tốt nhất, bạn phải đối mặt với hai biến số:

1. **Số lượng tham số (N)?** Mô hình lớn hơn, dung lượng (capacity) cao hơn.
2. **Số lượng token huấn luyện (D)?** Dữ liệu nhiều hơn, tận dụng dung lượng tốt hơn.

FLOPs tỉ lệ xấp xỉ với `6 × N × D`. Bạn có thể tăng N và giảm D, hoặc tăng D và giảm N. Cách nào tốt hơn?

Trước năm 2022, câu trả lời là "tăng mạnh N". GPT-3 (2020) có 175B tham số được huấn luyện trên ~300B token. Tỉ lệ khoảng 1,7 token trên mỗi tham số. Các định luật scaling của Kaplan đã củng cố điều này.

Hoffmann và cộng sự (2022), khi huấn luyện một nhóm mô hình nhỏ có tên là Chinchilla, đã phát hiện ra điều khác biệt: tỉ lệ tối ưu gần với **20 token trên mỗi tham số**. GPT-3 đã được huấn luyện thiếu 10 lần. Chinchilla (70B tham số, 1,4T token) đã đánh bại GPT-3 (175B, 300B token) trên mọi benchmark với chi phí inference thấp hơn 2,5 lần.

Năm 2026 là thế giới của Chinchilla — với một điểm khác biệt quan trọng. Llama 3 8B được huấn luyện trên 15 nghìn tỷ token, tỉ lệ 1.875 token trên mỗi tham số. Gấp 94 lần so với mức tối ưu của Chinchilla. Chi phí inference quan trọng hơn chi phí huấn luyện đối với các mô hình được sử dụng ở quy mô lớn, vì vậy việc huấn luyện quá mức (vượt qua Chinchilla) để có footprint triển khai nhỏ hơn là mặc định của năm 2026.

## Khái niệm

![Chinchilla curves: loss vs compute at various N/D ratios](../assets/scaling-laws.svg)

### Định luật Hoffmann

Từ bài báo Chinchilla, loss tuân theo:

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = tham số (không bao gồm embedding).
- `D` = token huấn luyện.
- `α ≈ 0.34`, `β ≈ 0.28` (xấp xỉ đối xứng).
- `E ≈ 1.69`, ngưỡng loss không thể giảm thêm.
- `A ≈ 406`, `B ≈ 411`.

Hai thành phần này bù trừ cho nhau khi bạn scale. Lấy đạo hàm theo `N` tại compute cố định (C = 6ND) và giải phương trình:

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

Compute-optimal: 20 token trên mỗi tham số.

### Tại sao vẫn huấn luyện quá mức (over-training)?

Chinchilla-optimal tối thiểu hóa loss huấn luyện trên mỗi FLOP huấn luyện. Nhưng bạn chỉ trả chi phí huấn luyện một lần; còn chi phí inference là mãi mãi.

Đối với một chatbot phục vụ hàng nghìn tỷ token mỗi tháng, inference chiếm ưu thế trong tổng chi phí. Cách tiếp cận của Llama: huấn luyện mô hình nhỏ hơn, lâu hơn. 8B với 15T token được tối ưu hóa sâu cho inference:

- Vừa với GPU người dùng.
- Độ trễ chỉ bằng một phần nhỏ so với Chinchilla-optimal 70B.
- Chất lượng đủ tốt cho hầu hết các tác vụ.

Bài báo năm 2024 của DeepMind ("Over-training is the new optimal") đã chính thức hóa điều này. Đối với các workload ưu tiên inference, tỉ lệ phù hợp gần với 100–500 token trên mỗi tham số tùy thuộc vào lưu lượng phục vụ.

### Sự xuất hiện (emergence) vs sự mượt mà (smoothness)

Tuyên bố: một số khả năng nhất định (số học, suy luận nhiều bước, tuân theo chain-of-thought) "xuất hiện" đột ngột ở một quy mô nào đó.

Schaeffer và cộng sự (2023) lập luận rằng đây là một sai số đo lường: các chỉ số "xuất hiện" sử dụng cách chấm điểm gián đoạn (khớp chính xác, độ chính xác tại ngưỡng) làm che khuất sự cải thiện mượt mà của các logit bên dưới. Các chỉ số liên tục (cross-entropy) cho thấy các đường cong mượt mà.

Năm 2026, sự đồng thuận là: các dự đoán thông qua loss liên tục là đáng tin cậy. Những bước nhảy trên benchmark thường là do cách chấm điểm. Hãy lập ngân sách dựa trên các chỉ số liên tục.

### Bức tranh năm 2026

Các định luật scaling vẫn hoạt động, nhưng:

| Yếu tố | Thay đổi như thế nào |
|--------|-------------|
| Chất lượng dữ liệu | Tuyển chọn các token "tốt" (kiểu Phi) làm dịch chuyển các đường cong >2 lần compute hiệu dụng |
| MoE | Tổng số tham số tách rời khỏi FLOPs hoạt động; định luật scaling tính theo mỗi FLOP hoạt động |
| Post-training | Một số khả năng (tuân theo hướng dẫn, code) thay đổi nhờ SFT+RLHF nhiều hơn là pretraining |
| Đa phương thức | Token hình ảnh + văn bản scale cùng nhau; các đường cong riêng biệt cho mỗi phương thức |
| Dữ liệu tổng hợp | Các mô hình tạo ra dữ liệu huấn luyện; compute hiệu dụng có thể cộng hưởng |

Optimizer Muon (Kimi Moonlight, 2024) cho thấy mức tăng compute hiệu dụng ~2 lần so với AdamW trên cùng dữ liệu. Một số đợt huấn luyện năm 2026 sử dụng Muon làm mặc định. Nó làm thay đổi hằng số tuyệt đối trong định luật scaling, chứ không phải hình dạng của nó.

```figure
scaling-laws
```

## Xây dựng

Xem `code/main.py`. Chúng ta triển khai phương trình loss Chinchilla và giải cho `(N, D)` tối ưu compute tại mỗi ngân sách compute khác nhau.

### Bước 1: Loss Chinchilla

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

Vẽ `L` dưới dạng đường đồng mức trên `(N, D)` tại `C = 6ND` cố định. Tìm giá trị tối thiểu.

### Bước 2: Biên tối ưu compute (compute-optimal frontier)

Đối với các ngân sách compute từ `1e17` đến `1e25` FLOPs, tìm `(N, D)` giúp tối thiểu hóa loss với điều kiện `6ND = C`. Xác minh tỉ lệ `D/N ≈ 20`.

### Bước 3: Chi phí huấn luyện quá mức

Tính toán phần loss tăng thêm khi bạn huấn luyện một mô hình nhỏ hơn 10 lần (1/10 của N tối ưu, 10 lần D tối ưu). Báo cáo khoản tiết kiệm FLOPs inference (tỉ lệ thuận với N) để đổi lại.

### Bước 4: So sánh với các mô hình thực tế

Nhập các cặp `(N, D)` đã biết cho GPT-3, Chinchilla, Llama 3 8B, DeepSeek-V3 (tham số hoạt động) và so sánh loss dự đoán với loss thực tế.

## Sử dụng

Bạn khó có thể tự huấn luyện một mô hình frontier. Nhưng các định luật scaling cho bạn biết:

1. **Liệu fine-tune của bạn có đủ dữ liệu không.** Nếu dữ liệu đặc thù cho tác vụ của bạn dưới 20 token trên mỗi tham số của mô hình cơ sở, hãy kỳ vọng sự bão hòa tại một ngưỡng loss nào đó.
2. **Có nên chọn mô hình cơ sở lớn hơn không.** Nếu bạn đang chi tiêu toàn bộ ngân sách cho inference, hãy ưu tiên mô hình nhỏ hơn, được huấn luyện lâu hơn.
3. **Khi nào lợi nhuận giảm dần.** Vượt quá 1000 lần Chinchilla-optimal, thay đổi log-loss trở thành nhiễu.

**Quỹ đạo nghiên cứu năm 2026:**

- **Chế độ hạn chế dữ liệu.** Web có số lượng token chất lượng cao hữu hạn (~5–10 nghìn tỷ tiếng Anh sau khi lọc). Pretraining frontier đang tiến gần đến ngưỡng này. Dữ liệu tổng hợp, đa ngôn ngữ, đa phương thức và fine-tuning scale bằng RLHF là những đòn bẩy tiếp theo.
- **Các thủ thuật nhân compute.** Optimizer Muon, MoE, tuyển chọn dữ liệu tốt hơn — mỗi thứ làm dịch chuyển các hằng số tuyệt đối, không phải tiệm cận.
- **Định luật scaling cho RL.** Câu hỏi mở. Bằng chứng ban đầu cho thấy định luật lũy thừa trong các mẫu RL nhưng với các số mũ rất khác so với pretraining.

## Triển khai

Xem `outputs/skill-training-budget-estimator.md`. Kỹ năng này chọn `(N, D, hours, GPU)` cho một đợt huấn luyện mới dựa trên ngân sách compute, các ràng buộc triển khai và loss mục tiêu.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. In ra `(N, D)` tối ưu Chinchilla cho các ngân sách compute `1e20`, `1e22`, `1e24`. So sánh với bảng mô hình thực tế.
2. **Trung bình.** Triển khai đường cong loss-theo-compute của Hoffmann. Vẽ loss so với `log10(C)` cho biên tối ưu compute. Xác định khi nào định luật dự đoán chúng ta cần `>10^28` FLOPs cho lần giảm 0,1 tiếp theo trong cross-entropy.
3. **Khó.** Tự khớp định luật scaling của riêng bạn trên 5 mô hình nhỏ (100K đến 10M tham số) được huấn luyện trên cùng một tập dữ liệu. Ước tính `α` và `E`. Các số mũ của bạn khớp với các số mũ đã công bố như thế nào?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Tham số (N) | "Kích thước mô hình" | Số lượng trọng số không bao gồm embedding; xác định dung lượng. |
| Token (D) | "Dữ liệu huấn luyện" | Số lượng token huấn luyện đã thấy; xác định mức độ tận dụng tham số. |
| Compute (C) | "FLOPs đã chi" | Xấp xỉ `6 × N × D` cho một transformer tiêu chuẩn. |
| Chinchilla-optimal | "D/N ≈ 20" | Tỉ lệ tối thiểu hóa loss trên mỗi FLOP pretraining. |
| Huấn luyện quá mức | "Vượt qua Chinchilla" | Chi thêm FLOPs huấn luyện để tiết kiệm FLOPs inference; D/N >> 20. |
| Loss không thể giảm | "Ngưỡng" | Thành phần `E` trong định luật scaling; entropy của chính dữ liệu đó. |
| Khả năng xuất hiện | "Nhảy vọt ở quy mô lớn" | Thường là sai số do cách chấm điểm; loss liên tục là mượt mà. |
| Compute hiệu dụng | "Hệ số nhân hiệu quả huấn luyện" | Dữ liệu / optimizer / kiến trúc tốt hơn làm nhân lên hiệu quả của một FLOP. |

## Đọc thêm

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — bài báo đầu tiên về định luật scaling; huấn luyện thiếu.
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla.
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — sự xuất hiện là sai số đo lường.
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — tại sao việc huấn luyện quá mức của Llama lại đúng với workload của nó.
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — hệ số nhân compute 2 lần.