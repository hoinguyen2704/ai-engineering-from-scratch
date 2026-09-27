# Jamba — Hybrid SSM-Transformer

> Các mô hình không gian trạng thái (State space models - SSM) và Transformer có những mục tiêu khác nhau. Transformer đạt được chất lượng thông qua cơ chế attention với chi phí tính toán bậc hai. SSM đạt được khả năng suy luận thời gian tuyến tính và bộ nhớ không đổi thông qua cơ chế hồi quy (recurrence) nhưng lại thua kém về chất lượng. Jamba của AI21 (tháng 3 năm 2024) và Jamba 1.5 (tháng 8 năm 2024) kết hợp chúng trong cùng một mô hình: 1 lớp Transformer cho mỗi 7 lớp Mamba, sử dụng MoE trên mỗi khối xen kẽ, và cửa sổ ngữ cảnh 256k có thể vừa vặn trên một GPU 80GB duy nhất. Mamba-3 (ICLR 2026) cải tiến phía SSM với không gian trạng thái giá trị phức và các phép chiếu MIMO. Bài học này đọc cả hai kiến trúc từ đầu đến cuối và giải thích lý do tại sao công thức lai này đã tồn tại qua ba năm mở rộng quy mô trong khi các nỗ lực SSM thuần túy và Transformer thuần túy cho ngữ cảnh dài lại không làm được.

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**Time:** ~60 minutes

## Mục tiêu học tập

- Giải thích ba thành phần nguyên thủy trong một khối Jamba — các lớp Transformer, các lớp Mamba, MoE — và công thức xen kẽ 1:7:chẵn.
- Trình bày cơ chế hồi quy của SSM ở mức độ tổng quát và lý do tại sao nó cho phép suy luận với bộ nhớ không đổi.
- Tính toán dung lượng KV cache của một mô hình Jamba ở ngữ cảnh 256k và so sánh với những gì một mô hình Transformer thuần túy cần.
- Nêu tên ba cải tiến của Mamba-3 (rời rạc hóa hình thang mũ, cập nhật trạng thái giá trị phức, MIMO) và vấn đề mà mỗi cải tiến nhắm đến.

## Vấn đề

Attention có độ phức tạp bậc hai theo độ dài chuỗi. Các mô hình không gian trạng thái có độ phức tạp tuyến tính. Sự khác biệt đó tích lũy: ở 256k token, một bản đồ attention của Transformer là 65 tỷ mục mỗi head; trạng thái hồi quy của SSM có kích thước cố định bất kể độ dài chuỗi.

Các mô hình SSM thuần túy (Mamba, Mamba-2) đạt được độ phức tạp (perplexity) tương đương Transformer ở quy mô nhỏ nhưng lại tụt hậu trong các tác vụ theo dõi trạng thái và thất bại ở một số danh mục truy xuất trong ngữ cảnh (in-context retrieval). Trực giác: SSM nén lịch sử vào một trạng thái cố định, và khi lịch sử quá dài, thông tin bị rò rỉ. Attention ghi nhớ mọi thứ một cách chính xác nhưng phải trả giá bằng chi phí bậc hai.

Giải pháp hiển nhiên: sử dụng cả hai. Đặt các lớp Transformer ở nơi cần ghi nhớ chính xác. Sử dụng các lớp SSM ở những nơi khác. Điều chỉnh tỷ lệ. Jamba là mô hình cấp sản xuất đầu tiên áp dụng công thức lai này ở quy mô lớn (tổng 52B, 12B hoạt động, ngữ cảnh 256k, một GPU 80GB). Jamba 1.5 mở rộng gia đình này lên tổng 398B / 94B hoạt động. Mamba-3 (ICLR 2026) là baseline SSM thuần túy tốt nhất hiện nay mà các mô hình lai có thể được xây dựng dựa trên đó.

Bài học này đọc cả ba bài báo và tạo ra mô hình tư duy để "chọn tỷ lệ phù hợp".

## Khái niệm

### SSM trong một trang

Một mô hình không gian trạng thái xử lý một chuỗi `x_1, ..., x_N` thông qua một trạng thái có kích thước cố định `h`:

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

Tại mỗi bước, trạng thái tiến hóa thông qua động lực học tuyến tính `A`, nhận đầu vào `B x_t`, và phát ra đầu ra `C h_t`. `A, B, C` có thể được học. Lưu ý thuộc tính quan trọng: việc tính toán `y_t` chỉ cần `h_{t-1}` và `x_t`, không cần bất kỳ `x` nào trước đó. Bộ nhớ là hằng số. Suy luận là O(1) trên mỗi token.

Mẹo để mô hình hóa chất lượng nằm ở cấu trúc của `A`. S4 (Gu 2021) đã sử dụng một ma trận có cấu trúc cao có thể được đánh giá hiệu quả như một phép tích chập dài trong quá trình huấn luyện. Mamba (Gu, Dao 2023) đã thay thế `A, B, C` cố định bằng các ma trận phụ thuộc vào dữ liệu (phần "chọn lọc"). Mamba-2 (2024) đơn giản hóa cấu trúc hơn nữa. Mamba-3 (2026) thêm lại sự phức tạp ở những vị trí cụ thể.

Thuộc tính chính: đối với một LLM giải mã, một lớp SSM là sự thay thế hoàn hảo cho một lớp attention, với trạng thái cố định theo từng lớp thay vì KV cache ngày càng tăng.

### Khối Jamba

Một khối Jamba xen kẽ các lớp theo hai con số:

- `l`: tỷ lệ attention-to-Mamba. Jamba sử dụng `l = 8`, nghĩa là 1 lớp Transformer cho mỗi 7 lớp Mamba (7 Mamba + 1 Attention = 8 lớp mỗi nhóm).
- `e`: tần suất MoE. Jamba sử dụng `e = 2`, nghĩa là cứ mỗi lớp khác nhau sẽ áp dụng MoE.

Chuỗi lớp trong một khối:

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

Mỗi khối Jamba gồm 8 lớp. Ở độ sâu 4 khối (tổng 32 lớp), bạn có 28 lớp Mamba và 4 lớp Attention. 16 trong số đó sử dụng MoE.

### Tại sao lại là tỷ lệ 1:7

AI21 đã thực hiện các thử nghiệm cắt bỏ (ablations): tỷ lệ attention-to-Mamba nào mang lại perplexity-trên-tham số tốt nhất VÀ khả năng truy xuất trong ngữ cảnh tốt nhất trên các đánh giá ngữ cảnh dài của họ?

- Quá nhiều attention (1:1): chất lượng tăng lên nhưng bộ nhớ và tốc độ suy giảm.
- Quá ít attention (1:15): bộ nhớ rất tốt nhưng truy xuất trong ngữ cảnh thất bại.
- Điểm ngọt: 1:7 hoặc 1:8.

Trực giác: các lớp Transformer xử lý việc ghi nhớ chính xác và theo dõi trạng thái. Các lớp Mamba xử lý phần lớn khối lượng công việc giá rẻ.

### Mã hóa vị trí (Positional encoding)

Các lớp Mamba tự nhận thức vị trí (thông qua cơ chế hồi quy). Các lớp Attention trong các mô hình lai dựa trên Mamba ban đầu không sử dụng RoPE — các lớp SSM đã cung cấp thông tin vị trí. Jamba 1.5 thêm RoPE vào các lớp attention để tổng quát hóa ngữ cảnh dài hơn, một sự tinh chỉnh hậu kỳ dựa trên đánh giá ngữ cảnh dài thực nghiệm.

### Ngân sách bộ nhớ

Đối với cấu hình Jamba-1 (32 lớp: 28 Mamba + 4 Attention, hidden 4096, 32 attention heads):

- KV cache (chỉ các lớp attention): `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB` ở 256k BF16. Chỉ 4 lớp attention đóng góp.
- Trạng thái SSM: `28 * hidden * state_size` trên mỗi tiền tố token, nhưng đây là kích thước cố định trên mỗi lớp, không mở rộng theo độ dài chuỗi. Trạng thái Mamba điển hình là 16 trên mỗi feature, hidden 4096: `28 * 4096 * 16 * 2 = 3.7 MB` tổng cộng.

So sánh với một Transformer thuần túy ở 32 lớp, cùng hidden, MHA đầy đủ ở 32 heads: `2 * 32 * 32 * 128 * 256k * 2 = 128 GB` ở 256k BF16. Giảm 8 lần KV cache. Ngay cả so với baseline GQA(8) mà hầu hết các mô hình năm 2024 sử dụng (`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`), mô hình lai 1:7 của Jamba ở mức 16 GB vẫn nhỏ hơn 2 lần.

Đó là ý nghĩa của AI21 khi nói về "ngữ cảnh 256k trên một GPU 80GB duy nhất". KV cache của một Transformer thuần túy MHA đầy đủ sẽ không vừa; ngay cả một baseline GQA cũng không để lại chỗ cho trọng số và kích hoạt; còn Jamba thì có.

### Mamba-3: baseline SSM thuần túy năm 2026

Mamba-3 (ICLR 2026, arXiv:2603.15569) giới thiệu ba cải tiến ở phía SSM thuần túy:

1. **Rời rạc hóa hình thang mũ (Exponential-trapezoidal discretization).** Thay thế phương pháp rời rạc hóa Euler trong Mamba-2 bằng một cơ chế hồi quy biểu cảm hơn. Phép toán giống tích chập được áp dụng trên đầu vào trạng thái trong cơ chế hồi quy cốt lõi, thay vì là một tích chập bên ngoài trên `x_t`.

2. **Cập nhật trạng thái giá trị phức.** Các phiên bản Mamba trước đã giảm ma trận trạng thái từ phức (S4) xuống đường chéo thực (Mamba) rồi đến ma trận đơn vị được chia tỷ lệ (Mamba-2). Mamba-3 thêm lại các giá trị phức — tương đương với một embedding xoay phụ thuộc vào dữ liệu trên trạng thái. Điều này khôi phục khả năng theo dõi trạng thái mà các đơn giản hóa giá trị thực trước đây đã làm mất đi.

3. **Các phép chiếu đa đầu vào đa đầu ra (MIMO).** Thay vì các phép chiếu vô hướng trên mỗi feature, hãy sử dụng các phép chiếu ma trận. Cải thiện sức mạnh mô hình hóa và hiệu suất phần cứng khi suy luận mà không làm tăng độ trễ giải mã.

Ở mức 1.5 tỷ tham số, Mamba-3 cải thiện độ chính xác trung bình hạ nguồn thêm 0.6 điểm so với Gated DeltaNet; biến thể MIMO thêm 1.2 điểm nữa cho tổng mức tăng 1.8 điểm. Ở cùng kích thước trạng thái, Mamba-3 tương đương Mamba-2 với một nửa trạng thái.

Mamba-3 hiện chưa được triển khai trong mô hình lai sản xuất ở quy mô lớn — nhưng nó là ứng cử viên rõ ràng cho phía SSM của thế hệ mô hình lớp Jamba tiếp theo.

### Khi nào nên chọn mô hình lai

Các mô hình lai chiến thắng khi:

- Ngữ cảnh đủ dài khiến KV cache của Transformer thuần túy trở nên khó chịu (64k+).
- Các tác vụ kết hợp cấu trúc phạm vi ngắn (tốt cho SSM) với khả năng ghi nhớ phạm vi dài (cần Transformer).
- Bạn muốn triển khai trên ngân sách bộ nhớ GPU đơn lẻ nơi mà chỉ riêng KV cache của Transformer cũng không vừa.

Các mô hình lai thất bại khi:

- Ngữ cảnh ngắn (dưới 16k). Chi phí SSM là lãng phí; Transformer thuần túy là ổn.
- Các tác vụ cần attention mọi nơi (lập luận sâu, tham chiếu chéo đa tài liệu). Sự thưa thớt của các lớp attention trong mô hình lai gây bất lợi.
- Bạn đang mở rộng quy mô lên các mô hình biên giới hàng nghìn tỷ tham số. Transformer thuần túy + MLA + MoE (kiểu DeepSeek-V3) hiện đang chiến thắng trong cuộc đua năng lực.

### Bối cảnh cạnh tranh

| Mô hình | Gia đình | Quy mô | Tuyên bố độc đáo |
|-------|--------|------|-------------|
| Mamba-2 | SSM thuần túy | 3B | thời gian tuyến tính, bộ nhớ hằng số |
| Jamba | lai | 52B/12B | 256k trên 80GB |
| Jamba 1.5 Large | lai | 398B/94B | ngữ cảnh dài cấp doanh nghiệp |
| Mamba-3 | SSM thuần túy | 1.5B (bài báo) | khôi phục theo dõi trạng thái |
| DeepSeek-V3 | Transformer thuần túy + MoE | 671B/37B | năng lực biên giới |

Bối cảnh năm 2026: Transformer thuần túy MoE thống trị biên giới, nhưng các mô hình lai sở hữu thị trường ngách ngữ cảnh 256k-cộng. Những chiến thắng về theo dõi trạng thái của Mamba-3 có thể đẩy tỷ lệ lai xuống thấp hơn (nhiều SSM hơn, ít attention hơn) trong thế hệ tiếp theo.

```figure
swiglu-ffn
```

## Sử dụng

`code/main.py` là một máy tính bộ nhớ cho các kiến trúc lai. Với tỷ lệ SSM-Transformer và cấu hình hidden-size / số lớp, nó tính toán:

- KV cache ở ngữ cảnh mục tiêu.
- Bộ nhớ trạng thái SSM.
- Tổng bộ nhớ ở ngữ cảnh N cho một loạt các hình dạng mô hình.

Máy tính hỗ trợ:

- Baseline Transformer thuần túy (KV cache tăng theo N).
- Mô hình lai kiểu Jamba 1:7.
- SSM thuần túy (không có KV cache).

Các con số được lấy trực tiếp từ các bài báo Jamba-1 và Jamba-1.5 cho các hình dạng đã công bố và ngoại suy cho các biến thể giả định.

Các cân nhắc tích hợp cho triển khai thực tế:

- Hầu hết các máy chủ suy luận sản xuất (vLLM, SGLang) đều hỗ trợ Jamba và Mamba. Kiểm tra phiên bản cụ thể.
- Ở ngữ cảnh 256k, lợi thế bộ nhớ của Jamba thể hiện ở thông lượng yêu cầu đồng thời. Trên cùng VRAM, bạn có thể chứa nhiều chuỗi Jamba hơn chuỗi Transformer.
- Mamba-3 như một mô hình độc lập chưa được triển khai trong sản xuất — bản xem trước nghiên cứu ở mức 1.5B.

## Triển khai

Bài học này tạo ra `outputs/skill-hybrid-picker.md`. Với đặc tả khối lượng công việc (hồ sơ độ dài ngữ cảnh, hỗn hợp tác vụ, ngân sách bộ nhớ), nó đưa ra khuyến nghị giữa Transformer thuần túy, mô hình lai kiểu Jamba và SSM thuần túy, với lập luận rõ ràng về các đánh đổi bộ nhớ và chất lượng.

## Bài tập

1. Chạy `code/main.py` để tính toán KV cache ở ngữ cảnh 256k cho một Transformer thuần túy 32 lớp (hidden 4096, 32 heads) và cho một mô hình lai Jamba-1 cùng hình dạng. Xác minh mức giảm bộ nhớ ~8 lần mà bài báo AI21 tuyên bố.

2. Sửa đổi máy tính để mô hình hóa mô hình lai 1:3 (4 Mamba : 1 Attention) và 1:15 (14 Mamba : 1 Attention). Vẽ biểu đồ KV cache so với tỷ lệ. Tại tỷ lệ nào thì KV cache bằng bộ nhớ trạng thái SSM?

3. Đọc Phần 3 của bài báo Jamba (arXiv:2403.19887). Giải thích tại sao AI21 sử dụng Mamba-1 thay vì Mamba-2 mặc dù Mamba-2 nhanh hơn. Gợi ý: phần thử nghiệm cắt bỏ mô hình lai ghi lại điều này.

4. Tính toán chi phí tham số của MoE-trên-mỗi-lớp-khác trong Jamba 1.5 Large (tổng 398B, 94B hoạt động). So sánh tỷ lệ hoạt động với DeepSeek-V3 (37B/671B) và giải thích tại sao kiến trúc của Jamba đẩy tỷ lệ hoạt động lên cao hơn.

5. Đọc Phần 3 của bài báo Mamba-3 (arXiv:2603.15569). Giải thích trong ba câu tại sao cập nhật trạng thái giá trị phức lại tương đương với một embedding xoay phụ thuộc vào dữ liệu. Liên kết câu trả lời với phần dẫn xuất RoPE của Phase 7 · Lesson 04.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói gì | Ý nghĩa thực sự |
|------|----------------|------------------------|
| State space model (SSM) | "Hồi quy với trạng thái cố định" | Một lớp với cơ chế hồi quy đã học `h_t = A h_{t-1} + B x_t`; bộ nhớ hằng số trên mỗi token |
| Selective SSM | "Mẹo của Mamba" | Các tham số A, B, C phụ thuộc vào dữ liệu mang lại cho mô hình khả năng chọn lọc giống như gating ở thời gian tuyến tính |
| Tỷ lệ Attention-to-Mamba | "Có bao nhiêu lớp attention" | Trong Jamba, `l = 8` nghĩa là 1 lớp attention trên mỗi 7 lớp Mamba |
| Khối Jamba | "Nhóm 8 lớp" | Một attention + bảy Mamba + MoE ở các vị trí xen kẽ |
| Trạng thái SSM | "Bộ đệm ẩn" | Trạng thái cố định theo từng lớp thay thế KV cache cho các lớp Mamba |
| Ngữ cảnh 256k | "Con số chủ đạo của Jamba" | Độ dài chuỗi Jamba-1 vừa vặn trên một GPU 80GB duy nhất; Transformer thuần túy không thể ở kích thước đó |
| Mamba-3 | "SSM thuần túy 2026" | Kiến trúc SSM thuần túy tốt nhất hiện nay với trạng thái phức + MIMO; baseline mà các mô hình lai xây dựng xung quanh |
| MIMO | "Đa đầu vào đa đầu ra" | Cải tiến của Mamba-3 sử dụng các phép chiếu ma trận thay vì vô hướng trên mỗi feature |
| Rời rạc hóa hình thang mũ | "Cơ chế hồi quy của Mamba-3" | Cơ chế hồi quy biểu cảm hơn bao hàm cả rời rạc hóa phương pháp Euler của Mamba-2 |
| Kiến trúc lai | "Trộn attention và SSM" | Bất kỳ mô hình nào xen kẽ các lớp Transformer và SSM; Jamba là nguyên mẫu sản xuất |

## Đọc thêm

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — bài báo Jamba gốc, các thử nghiệm cắt bỏ tỷ lệ, tuyên bố ngữ cảnh 256k
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — gia đình mở rộng quy mô, các bản phát hành công khai 398B/94B và 12B/52B
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — bài báo SSM chọn lọc mà Jamba xây dựng dựa trên đó
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — người kế nhiệm không gian trạng thái có cấu trúc đơn giản hóa
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — trạng thái giá trị phức, MIMO, biên giới SSM thuần túy năm 2026
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — bài báo S4, điểm khởi đầu của phả hệ SSM cho LLM