# Multi-Head Attention

> Một attention head học một mối quan hệ tại một thời điểm. Tám head học tám mối quan hệ. Các head này không tốn kém. Hãy sử dụng nhiều hơn.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 phút

## Vấn đề

Một self-attention head đơn lẻ tính toán một ma trận attention. Ma trận đó nắm bắt một loại mối quan hệ — thường là loại giúp giảm thiểu loss dựa trên tín hiệu huấn luyện. Nếu dữ liệu của bạn có sự hòa hợp giữa chủ ngữ và động từ, đồng tham chiếu (co-reference), diễn ngôn tầm xa và phân đoạn cú pháp đan xen vào nhau, một head đơn lẻ sẽ làm nhòe chúng thành một phân phối softmax duy nhất và làm mất đi một nửa tín hiệu.

Giải pháp từ bài báo Vaswani năm 2017: chạy song song nhiều hàm attention, mỗi hàm có các phép chiếu Q, K, V riêng và nối (concatenate) các đầu ra lại với nhau. Mỗi head hoạt động trong một không gian con nhỏ hơn với chiều `d_model / n_heads`. Tổng số tham số vẫn giữ nguyên. Khả năng biểu diễn tăng lên.

Multi-head attention là mặc định mà mọi transformer vào năm 2026 đều sử dụng. Tranh luận duy nhất là về *số lượng* head và liệu các key và value có chia sẻ phép chiếu hay không (Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention).

## Khái niệm

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Tách (Split).** Lấy `X` có hình dạng `(N, d_model)`. Chiếu sang Q, K, V, mỗi cái có hình dạng `(N, d_model)`. Định hình lại (reshape) thành `(N, n_heads, d_head)` trong đó `d_head = d_model / n_heads`. Chuyển vị (transpose) thành `(n_heads, N, d_head)`.

**Tính toán song song (Attend in parallel).** Chạy scaled dot-product attention bên trong mỗi head. Mỗi head tạo ra `(N, d_head)`. Các head hoạt động trên các không gian con khác nhau của embedding và không bao giờ tương tác trong quá trình tính toán attention.

**Nối và chiếu (Concatenate and project).** Xếp chồng các head trở lại thành `(N, d_model)` và nhân với ma trận đầu ra đã học `W_o` có hình dạng `(d_model, d_model)`. `W_o` là nơi các head được phép trộn lẫn thông tin.

**Tại sao nó hiệu quả.** Mỗi head có thể chuyên biệt hóa mà không cần cạnh tranh với các head khác về ngân sách biểu diễn. Các nghiên cứu thăm dò từ 2019–2024 cho thấy các vai trò riêng biệt của head: head vị trí, head chú ý đến token trước đó, head sao chép, head thực thể có tên, head quy nạp (induction heads - nền tảng của in-context learning).

**Các biến thể dòng dõi năm 2026:**

| Biến thể | Q heads | K/V heads | Được sử dụng bởi |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (ví dụ: N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | nén về hạng thấp | DeepSeek-V2, V3 |

GQA là mặc định hiện đại vì nó cắt giảm bộ nhớ KV-cache theo hệ số `N/G` trong khi vẫn giữ gần như toàn bộ chất lượng. MLA tiến xa hơn bằng cách nén K/V vào một không gian tiềm ẩn (latent space), sau đó chiếu ngược lại tại thời điểm tính toán — tốn FLOPs nhưng tiết kiệm bộ nhớ hơn nhiều.

```figure
multihead-split
```

## Xây dựng

### Bước 1: tách các head từ self-attention đơn lẻ mà chúng ta đã có

Lấy `SelfAttention` từ Bài 02 và bao bọc nó bằng một cặp split/concat. Xem `code/main.py` để biết cách triển khai bằng numpy; logic là:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Một lần reshape và một lần transpose. Không cần vòng lặp. Đây chính xác là những gì PyTorch thực hiện dưới `nn.MultiheadAttention`.

### Bước 2: chạy scaled-dot-product attention cho mỗi head

Mỗi head nhận một lát cắt riêng của Q, K, V. Attention trở thành một phép nhân ma trận theo lô (batched matmul):

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

Trên phần cứng thực tế, `Qh @ Kh.transpose(...)` là một `bmm`. GPU nhìn thấy một phép nhân ma trận theo lô duy nhất có hình dạng `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`. Việc thêm các head là miễn phí.

### Bước 3: Biến thể Grouped-Query Attention

Chỉ các phép chiếu key và value thay đổi. Q nhận `n_heads` nhóm; K và V nhận `n_kv_heads < n_heads` nhóm và được lặp lại để khớp:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

Khi suy luận (inference), điều này giúp tiết kiệm bộ nhớ vì chỉ có `n_kv_heads` bản sao tồn tại trong KV cache, thay vì `n_heads`. Llama 3 70B sử dụng 64 query head với 8 KV head — giảm 8 lần dung lượng cache.

### Bước 4: thăm dò những gì mỗi head đã học

Chạy MHA trên một câu ngắn với 4 head. Đối với mỗi head, in ra ma trận attention `(N, N)`. Bạn sẽ thấy các head khác nhau chọn ra các cấu trúc khác nhau ngay cả với khởi tạo ngẫu nhiên — đó một phần là tín hiệu, một phần là tính đối xứng xoay trong các không gian con.

## Sử dụng

Trong PyTorch, phiên bản một dòng:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA từ PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**Cần bao nhiêu head?** Các quy tắc ngón tay cái từ các mô hình sản xuất năm 2026:

| Kích thước mô hình | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Nhỏ (~125M) | 768 | 12 | 64 |
| Cơ sở (~350M) | 1024 | 16 | 64 |
| Lớn (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` hầu như luôn rơi vào 64 hoặc 128. Đó là đơn vị đo lường mức độ "nhìn thấy" của một head. Nếu giảm xuống dưới 32, các head bắt đầu chống lại hệ số tỉ lệ `sqrt(d_head)`; nếu vượt quá 256, bạn sẽ mất đi lợi ích của việc có "nhiều chuyên gia nhỏ".

## Triển khai

Xem `outputs/skill-mha-configurator.md`. Kỹ năng này khuyến nghị số lượng head, số lượng kv-head và chiến lược chiếu cho một transformer mới dựa trên ngân sách tham số, độ dài chuỗi và mục tiêu triển khai.

## Bài tập

1. **Dễ.** Lấy MHA từ `code/main.py` và thay đổi `n_heads` từ 1 thành 16 với `d_model=64` cố định. Vẽ biểu đồ loss của một mô hình một lớp nhỏ trên tác vụ sao chép tổng hợp. Nhiều head hơn giúp ích, bão hòa hay gây hại?
2. **Trung bình.** Triển khai MQA (một KV head được chia sẻ giữa tất cả các query head). Đo lường mức độ giảm số lượng tham số so với MHA đầy đủ. Tính toán mức độ thu nhỏ của KV-cache khi suy luận với N=2048.
3. **Khó.** Triển khai một phiên bản nhỏ của Multi-head Latent Attention: nén K,V thành một latent có hạng `r`, lưu latent đó trong KV cache, giải nén tại thời điểm attention. Tại `r` nào thì bộ nhớ cache giảm xuống dưới 1/8 so với MHA đầy đủ trong khi chất lượng vẫn nằm trong phạm vi 1 bit của validation ppl?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Head | "Một mạch attention đơn lẻ" | Một phép chiếu Q/K/V có chiều `d_head = d_model / n_heads` với ma trận attention riêng. |
| d_head | "Chiều của head" | Độ rộng ẩn mỗi head; hầu như luôn là 64 hoặc 128 trong sản xuất. |
| Split / combine | "Mẹo reshape" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose xung quanh attention. |
| W_o | "Phép chiếu đầu ra" | Ma trận `(d_model, d_model)` được áp dụng sau khi nối các head; nơi các head trộn lẫn. |
| MQA | "Một KV head" | Multi-Query Attention: phép chiếu K/V dùng chung duy nhất. KV cache nhỏ nhất, mất một chút chất lượng. |
| GQA | "Mặc định từ Llama 2" | Grouped-Query Attention với `n_kv_heads < n_heads`; lặp lại để khớp với Q. |
| MLA | "Mẹo của DeepSeek" | Multi-head Latent Attention: K,V nén thành latent hạng thấp, giải nén khi tính attention. |
| Induction head | "Mạch đằng sau in-context learning" | Một cặp head phát hiện các lần xuất hiện trước đó và sao chép những gì theo sau chúng. |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — đặc tả multi-head gốc.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — bài báo về MQA.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — cách chuyển đổi MHA sang GQA sau khi huấn luyện.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA và lý do tại sao nó vượt trội hơn MHA/GQA về bộ nhớ cache.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — cái nhìn cơ học về những gì các head thực sự làm.