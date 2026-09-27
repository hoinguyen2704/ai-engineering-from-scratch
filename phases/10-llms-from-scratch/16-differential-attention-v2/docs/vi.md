# Differential Attention (V2)

> Softmax attention phân tán một lượng nhỏ xác suất lên mọi token không khớp. Với hơn 100k token, nhiễu này tích tụ và nhấn chìm tín hiệu. Differential Transformer (Ye et al., ICLR 2025) khắc phục điều này bằng cách tính toán attention dưới dạng hiệu của hai softmax, loại bỏ phần nhiễu nền chung. DIFF V2 (Microsoft, tháng 1 năm 2026) là bản viết lại cho stack sản xuất: khớp độ trễ giải mã (decode latency) với Transformer cơ sở, không cần kernel tùy chỉnh, tương thích với FlashAttention. Bài học này đi từ V1 đến V2 một cách toàn diện, với một bản cài đặt mô phỏng (toy implementation) của phép toán hiệu mà bạn có thể chạy trong Python stdlib.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**Time:** ~60 minutes

## Mục tiêu học tập

- Trình bày chính xác lý do tại sao softmax attention có nhiễu nền và tại sao nó tăng theo độ dài ngữ cảnh.
- Suy luận công thức differential attention và giải thích tại sao phép trừ triệt tiêu thành phần nhiễu chung trong khi vẫn bảo toàn tín hiệu.
- Đi qua sự khác biệt giữa V1 và V2: cái gì nhanh hơn, cái gì đơn giản hơn, cái gì ổn định hơn và tại sao mỗi thay đổi lại cần thiết cho quá trình pre-training trong sản xuất.
- Cài đặt differential attention từ đầu bằng Python thuần và kiểm chứng thực nghiệm đặc tính triệt tiêu nhiễu trên một query tín hiệu-cộng-nhiễu tổng hợp.

## Vấn đề

Standard softmax attention có một đặc tính toán học trở thành vấn đề vận hành ở quy mô lớn. Đối với một query `q`, các trọng số attention là `softmax(qK^T / sqrt(d))`. Softmax không bao giờ có thể tạo ra các số không chính xác — mọi token không khớp đều nhận được một phần khối lượng dương. Khối lượng dư thừa đó là nhiễu, và nó tỉ lệ thuận với độ dài ngữ cảnh. Ở 128k token, ngay cả khi mỗi token không khớp chỉ nhận được 0.001% xác suất, 127,999 token đó cộng lại đóng góp khoảng 12% tổng số. Mô hình phải học cách điều hướng xung quanh một nhiễu nền tăng dần theo ngữ cảnh.

Về mặt thực nghiệm, điều này xuất hiện dưới dạng nhiễu giữa các attention-head: trích dẫn ảo trong RAG ngữ cảnh dài, lỗi "lost-in-the-middle" trong các tác vụ truy xuất 100k-token, và sự suy giảm độ chính xác tinh vi trên các benchmark "needle-in-haystack" vượt quá 32k. Bài báo Differential Transformer (arXiv:2410.05258, ICLR 2025) đã đo lường khoảng cách này: DIFF Transformers đạt perplexity thấp hơn, độ chính xác ngữ cảnh dài cao hơn và ít ảo tưởng hơn so với các baseline cùng kích thước.

DIFF V1 có ba vấn đề khiến nó không thể đưa vào các pipeline pre-training tiên phong. Cache giá trị (value cache) của nó phải được tải hai lần mỗi bước giải mã, nó yêu cầu các CUDA kernel tùy chỉnh làm hỏng tính tương thích với FlashAttention, và RMSNorm trên mỗi head làm mất ổn định quá trình huấn luyện dài hạn ở quy mô 70B trở lên. DIFF V2 (blog Microsoft unilm, 20 tháng 1 năm 2026) đã khắc phục cả ba vấn đề này. Bài học này đi qua cả hai phiên bản, xây dựng toán tử hiệu và benchmark khả năng triệt tiêu nhiễu trên một query mô phỏng.

## Khái niệm

### Nhiễu nền của softmax

Đối với một query `q` và các key `K = [k_1, ..., k_N]`, các trọng số attention là:

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

Không có `w_i` nào bằng không. Nếu `k_i` hoàn toàn không liên quan đến `q`, điểm số `q . k_i` không phải là 0 — nó dao động quanh số không với phương sai `||q||^2 / d`. Sau khi chuẩn hóa softmax, mỗi token không liên quan vẫn đóng góp `O(1/N)` vào tổng có trọng số. Tổng đóng góp của các token không liên quan là `O((N-1)/N) = O(1)` — không phải là một lượng nhỏ.

Điều mô hình muốn là một cái gì đó giống như top-k cứng: trọng số cao trên các token khớp, trọng số gần bằng không ở mọi nơi khác. Softmax quá mượt để làm điều đó trực tiếp.

### Ý tưởng vi phân (Differential)

Chia các phép chiếu Q và K của mỗi head thành hai: Q = (Q_1, Q_2) và K = (K_1, K_2). Tính toán hai bản đồ attention:

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

Đầu ra:

```
DiffAttn = (A_1 - lambda * A_2) V
```

Phép trừ triệt tiêu bất kỳ phân phối nhiễu nào mà hai bản đồ chia sẻ. Nếu cả hai bản đồ đều có trọng số đồng nhất trên 127k token không liên quan (điều mà chúng sẽ có khi khởi tạo ngẫu nhiên), chúng sẽ triệt tiêu lẫn nhau. Tín hiệu — trọng số cao trên một vài token thực sự liên quan — chỉ triệt tiêu nếu nó xuất hiện trong cả hai bản đồ với cùng cường độ, điều này sẽ không xảy ra khi mô hình được huấn luyện.

`lambda` là một scalar có thể học được trên mỗi head, được tham số hóa dưới dạng `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`. Nó có thể âm. `lambda_init` mặc định là một số dương nhỏ như 0.8.

### Tại sao điều này khớp với việc triệt tiêu nhiễu theo head

Hãy nghĩ về hai micro nhiễu ghi lại cùng một giọng nói. Cả hai đều thu được người nói cộng với nhiễu nền tương quan. Trừ cái này cho cái kia và nhiễu chung sẽ bị loại bỏ. Giọng nói vẫn tồn tại vì hai tín hiệu khác nhau về pha hoặc biên độ đủ để ngăn chặn việc triệt tiêu hoàn toàn. `lambda` trên mỗi head học chính xác sự cân bằng này.

### V1 so với V2: sự khác biệt

V1 giữ số lượng tham số bằng với Transformer cơ sở. Để có hai query mỗi head, nó giảm một nửa chiều head. Điều đó làm giảm khả năng biểu đạt của head và — đau đớn hơn — giảm một nửa value cache mỗi head. Quá trình giải mã phải tải value cache hai lần mỗi bước (mỗi nhánh softmax một lần). Kết quả: giải mã chậm hơn baseline mặc dù số lượng tham số khớp nhau.

V2 tăng gấp đôi số lượng query head và giữ nguyên KV head (mượn tham số từ phép chiếu up-projection). Chiều head vẫn giữ nguyên như baseline. Sau phép trừ, chiều dư thừa được chiếu ngược xuống để khớp với phép chiếu O_W của Transformer cơ sở. Ba điều xảy ra cùng lúc:

1. Tốc độ giải mã khớp với baseline (KV cache được tải một lần).
2. FlashAttention chạy không thay đổi (không cần kernel tùy chỉnh).
3. Cường độ tính toán (arithmetic intensity) khi giải mã tăng lên (nhiều tính toán hơn trên mỗi byte được tải từ HBM).

V2 cũng loại bỏ RMSNorm trên mỗi head mà V1 đã sử dụng để ổn định phép trừ. Ở quy mô pre-training 70B, RMSNorm đó làm mất ổn định quá trình huấn luyện giai đoạn sau. V2 thay thế nó bằng một sơ đồ khởi tạo đơn giản hơn giúp giữ cho quá trình huấn luyện ổn định mà không cần thêm module phụ.

### Khi nào nên sử dụng

| Tác vụ | Lợi ích |
|----------|---------|
| Long-context RAG (64k+) | Bản đồ attention sạch hơn, ít trích dẫn ảo hơn |
| Needle-in-haystack benchmarks | Tăng độ chính xác đáng kể sau 32k |
| Multi-document QA | Ít nhiễu giữa các tài liệu hơn |
| Code completion ở 8k | Không đáng kể, không đáng để thay đổi kiến trúc |
| Short chat (< 4k) | Về cơ bản không thể phân biệt được với baseline |

Giá trị tăng theo độ dài ngữ cảnh. Ở 4k token, nhiễu nền đủ nhỏ để attention tiêu chuẩn vẫn ổn. Ở 128k, nó đang gây hại cho bạn.

### Cách nó kết hợp với các nút điều chỉnh khác năm 2026

| Tính năng | Tương thích với DIFF V2? |
|---------|------------------------|
| GQA | Có (V2 tăng Q head, không tăng KV head) |
| MLA (DeepSeek) | Có về nguyên tắc, chưa có bài báo công bố kết hợp chúng |
| MoE | Có (attention độc lập với khối MLP) |
| RoPE | Có (không thay đổi) |
| YaRN / long-context scaling | Có (chính xác là nơi DIFF giúp ích nhiều nhất) |
| FlashAttention | Có trong V2 (không trong V1) |
| Speculative decoding | Có (thay đổi attention không hiển thị với vòng lặp spec-decode) |

```figure
differential-attention
```

## Xây dựng nó

`code/main.py` cài đặt differential attention bằng Python thuần. Một query mô phỏng với cấu trúc tín hiệu-cộng-nhiễu đã biết cho phép bạn đo trực tiếp tỉ lệ triệt tiêu nhiễu.

### Bước 1: standard softmax attention

Các phép toán ma trận stdlib: danh sách các danh sách, matmul thủ công, softmax với phép trừ max để ổn định số học.

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Bước 2: chia Q, K thành hai nửa

Kiểu V1: giảm một nửa chiều head. Kiểu V2: giữ nguyên chiều head và tăng gấp đôi số lượng head. Bản cài đặt mô phỏng sử dụng V1 để làm rõ về mặt sư phạm — toán học là giống hệt nhau, chỉ khác về cách quản lý.

### Bước 3: hai nhánh softmax + phép trừ

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

Lưu ý: các trọng số đầu ra có thể âm. Điều đó hoàn toàn ổn — value cache vẫn xử lý các đóng góp có dấu. Phép chiếu V sau đó sẽ hấp thụ dấu này.

### Bước 4: đo lường triệt tiêu nhiễu

Xây dựng một chuỗi tổng hợp có độ dài 1024. Đặt token tín hiệu tại một vị trí đã biết, lấp đầy phần còn lại bằng nhiễu. Tính (a) trọng số attention softmax tiêu chuẩn tại vị trí tín hiệu và (b) trọng số differential attention. Đo tỉ lệ tín hiệu trên nhiễu trong mỗi trường hợp. DIFF attention tạo ra tỉ lệ tín hiệu trên nhiễu cao hơn đáng kể, gấp 3-10 lần tùy thuộc vào mức độ hai nhánh đã được huấn luyện để khác biệt nhau.

### Bước 5: hạch toán tham số V1 so với V2

Với một cấu hình (hidden=4096, heads=32, d_head=128), in ra:

- Baseline Transformer: Q, K, V mỗi cái kích thước `hidden * hidden`, MLP ở 4 * hidden.
- DIFF V1: Q, K mỗi cái kích thước `hidden * hidden`, V kích thước `hidden * hidden` (không đổi), chiều head giảm một nửa bên trong. Thêm các tham số `lambda` trên mỗi head (O(heads * d_head)).
- DIFF V2: Q kích thước `2 * hidden * hidden`, K kích thước `hidden * hidden`, V kích thước `hidden * hidden`. Chiều dư thừa được chiếu ngược xuống trước O_W. Thêm cùng số lượng tham số `lambda`.

Bản mô phỏng đo chi phí tham số bổ sung cho V2 (khoảng `hidden * hidden` thêm mỗi khối attention) và in nó ra.

## Sử dụng nó

DIFF V2 chưa được phát hành trong mọi máy chủ suy luận sản xuất tính đến tháng 4 năm 2026, nhưng việc tích hợp đang được tiến hành trong vLLM và SGLang. Trong khi đó, mô hình này xuất hiện trong:

- Các mô hình sản xuất ngữ cảnh dài nội bộ của Microsoft.
- Các bản sao nghiên cứu trong một số lần chạy huấn luyện mô hình mở nhắm mục tiêu ngữ cảnh 256k-plus.
- Các kiến trúc lai kết hợp DIFF attention với sliding-window attention trên các lớp xen kẽ.

Khi nào bạn nên sử dụng nó vào năm 2026:

- Huấn luyện một mô hình mới từ đầu nhắm mục tiêu ngữ cảnh hiệu dụng 64k-plus. Thêm differential attention ngay từ đầu; huấn luyện lại sau đó rất tốn kém.
- Fine-tune một mô hình ngữ cảnh dài nơi các lỗi "lost-in-the-middle" chiếm ưu thế trong đánh giá của bạn. Một LoRA trên các phép chiếu Q có thể xấp xỉ cấu trúc DIFF.

Khi nào bạn không nên:

- Bạn đang phục vụ một mô hình dày đặc (dense model) đã được pre-trained với hiệu suất ngữ cảnh dài ổn định. Chi phí huấn luyện lại hiếm khi bù đắp được trên các trọng số hiện có.
- Ngữ cảnh của bạn luôn dưới 16k. Nhiễu nền là không đáng kể.

## Triển khai nó

Bài học này tạo ra `outputs/skill-diff-attention-integrator.md`. Với một kiến trúc mô hình, độ dài ngữ cảnh mục tiêu, hồ sơ ảo tưởng và ngân sách huấn luyện, nó tạo ra một kế hoạch tích hợp để thêm differential attention vào một lần chạy pre-training mới hoặc fine-tune LoRA.

## Bài tập

1. Chạy `code/main.py`. Xác minh tỉ lệ tín hiệu trên nhiễu được báo cáo cho differential attention cao hơn so với standard softmax attention trên query tổng hợp. Thay đổi biên độ nhiễu và chỉ ra điểm giao thoa nơi attention tiêu chuẩn trở nên không thể sử dụng được.

2. Tính toán delta số lượng tham số từ baseline sang DIFF V1 và từ baseline sang DIFF V2 cho một mô hình lớp 7B (hidden=4096, heads=32, d_head=128, 32 lớp). Chỉ ra các thành phần nào tăng tham số và thành phần nào giữ nguyên.

3. Đọc Phần 3 của bài báo DIFF V1 (arXiv:2410.05258) và Phần 2 của blog Hugging Face DIFF V2. Trong hai câu, hãy giải thích tại sao RMSNorm trên mỗi head của V1 là cần thiết và tại sao V2 có thể loại bỏ nó mà không gây ra sự phân kỳ huấn luyện.

4. Thực hiện ablation: tính toán differential attention với `lambda = 0` (chỉ softmax đầu tiên) và `lambda = 1` (phép trừ đầy đủ). Trên query tổng hợp, đo lường cách tỉ lệ tín hiệu trên nhiễu thay đổi qua quá trình quét. Xác định `lambda` tối đa hóa tỉ lệ tín hiệu trên nhiễu.

5. Mở rộng bản mô phỏng sang GQA + DIFF V2. Chọn 8 KV head và 32 Q head. Chứng minh rằng kích thước KV cache khớp với một mô hình GQA baseline với cùng cấu hình (8, 32).

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Differential attention | "Hai softmax trừ cho nhau" | Chia Q, K thành hai nửa, tính hai bản đồ softmax, trừ cái thứ hai (được nhân với lambda) khỏi cái thứ nhất, sau đó nhân với V |
| Noise floor | "Phần đuôi khác không của softmax" | Trọng số O(1/N) mà softmax đặt lên mọi token không liên quan, tổng cộng thành O(1) trên các ngữ cảnh dài |
| lambda | "Tỉ lệ phép trừ" | Scalar có thể học được trên mỗi head được tham số hóa là `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`; có thể âm |
| DIFF V1 | "Phiên bản ICLR 2025" | Differential Transformer gốc; giảm một nửa chiều head để bảo toàn số lượng tham số, cần kernel tùy chỉnh, giải mã chậm hơn |
| DIFF V2 | "Bản sửa lỗi tháng 1 năm 2026" | Tăng gấp đôi Q head giữ nguyên KV head; khớp tốc độ giải mã baseline và hoạt động với FlashAttention |
| Per-head RMSNorm | "Bộ ổn định V1" | Norm bổ sung mà V1 áp dụng sau phép trừ; V2 đã loại bỏ nó để ngăn chặn sự mất ổn định khi huấn luyện giai đoạn sau |
| Signal-to-noise ratio | "Bao nhiêu attention bị lãng phí" | Tỉ lệ trọng số tại vị trí tín hiệu thực so với trọng số trung bình tại các vị trí không liên quan |
| Lost in the middle | "Chế độ lỗi ngữ cảnh dài" | Hiện tượng thực nghiệm nơi độ chính xác truy xuất giảm đối với các tài liệu ở giữa ngữ cảnh dài — DIFF attention giảm thiểu điều này |
| Arithmetic intensity | "FLOPs trên mỗi byte được tải" | Tỉ lệ mà V2 tăng lên khi giải mã bằng cách tăng gấp đôi query trên mỗi lần tải KV; quan trọng cho giải mã bị giới hạn bộ nhớ |

## Đọc thêm

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — bài báo gốc với lý thuyết triệt tiêu nhiễu và các ablation ngữ cảnh dài
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — bản viết lại cho stack sản xuất, khớp giải mã baseline, tương thích FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — phân tích lý thuyết về lý do tại sao phép trừ khôi phục cấu trúc attention đã được pre-trained
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — biến thể chia sẻ tham số
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — Transformer cơ sở mà DIFF trừ đi
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — benchmark ngữ cảnh dài mà DIFF attention nhắm tới