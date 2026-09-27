# Các biến thể Attention — Sliding Window, Sparse, Differential

> Full attention là một vòng tròn. Mọi token đều nhìn thấy mọi token, và bộ nhớ phải trả giá cho điều đó. Bốn biến thể dưới đây uốn cong hình dạng của vòng tròn đó và thu hồi một nửa chi phí.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**Time:** ~60 phút

## Vấn đề

Full attention tiêu tốn `O(N²)` bộ nhớ và `O(N²)` tính toán theo độ dài chuỗi. Đối với Llama 3 70B với ngữ cảnh 128K, con số này là 16 tỷ mục attention mỗi layer, nhân với 80 layer. Flash Attention (Bài 12) ẩn đi bộ nhớ kích hoạt `O(N²)` nhưng không thay đổi chi phí số học — mỗi token vẫn phải attend tới mọi token khác.

Ba nhóm biến thể thay đổi cấu trúc liên kết (topology) của ma trận attention:

1. **Sliding window attention (SWA).** Mỗi token chỉ attend tới một cửa sổ cố định các token lân cận, thay vì toàn bộ tiền tố. Bộ nhớ và tính toán giảm xuống `O(N · W)`, trong đó `W` là kích thước cửa sổ. Được dùng trong Gemma 2/3, các layer đầu của Mistral 7B, Phi-3-Long.
2. **Sparse / block attention.** Chỉ các cặp `(i, j)` được chọn mới được tính điểm; các cặp còn lại bị ép về trọng số bằng 0. Ví dụ: Longformer, BigBird, OpenAI sparse transformer.
3. **Differential attention.** Tính toán hai bản đồ attention với các phép chiếu Q/K riêng biệt, sau đó trừ bản đồ này cho bản đồ kia. Loại bỏ "attention sink" (hố đen attention) vốn làm rò rỉ trọng số vào vài token đầu tiên. Microsoft's DIFF Transformer (2024).

Các biến thể này cùng tồn tại. Một mô hình tiên phong năm 2026 thường kết hợp chúng: hầu hết các layer là SWA-1024, cứ mỗi năm layer lại có một layer global full attention, và một vài head là differential để làm sạch việc truy xuất. Tỷ lệ 5:1 SWA-to-global của Gemma 3 hiện là tiêu chuẩn mặc định trong sách giáo khoa.

## Khái niệm

### Sliding Window Attention (SWA)

Mỗi query tại vị trí `i` chỉ attend tới các vị trí trong `[i - W, i]` (causal SWA) hoặc `[i - W/2, i + W/2]` (bidirectional). Các token nằm ngoài cửa sổ sẽ nhận giá trị `-inf` trong ma trận điểm.

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

Đối với `N = 8192` và `W = 1024`, ma trận điểm có 1024 × 8192 hàng khác không — giảm 8 lần.

**KV cache thu nhỏ với SWA.** Chỉ cần giữ lại `W` token cuối cùng của K và V cho mỗi layer. Với cấu hình kiểu Gemma-3 (cửa sổ 1024, ngữ cảnh 128K), KV cache giảm 128 lần.

**Chi phí về chất lượng.** Các transformer chỉ dùng SWA gặp khó khăn trong việc truy xuất tầm xa. Giải pháp: xen kẽ các layer SWA với các layer full-attention. Gemma 3 sử dụng tỷ lệ 5:1 SWA:global. Mistral 7B sử dụng một stack causal-SWA nơi thông tin "chảy về phía trước" qua các cửa sổ chồng lấp — mỗi layer mở rộng trường tiếp nhận hiệu dụng thêm `W`, và sau `L` layer, mô hình có thể attend ngược lại `L × W` token.

### Sparse / Block Attention

Chọn trước một mẫu thưa `N × N`. Ba hình dạng chính:

- **Local + strided (OpenAI sparse transformer).** Attend tới `W` token cuối cùng cộng với mỗi token thứ `stride` trước đó. Nắm bắt cả thông tin cục bộ và tầm xa với `O(N · sqrt(N))` chi phí tính toán.
- **Longformer / BigBird.** Cửa sổ cục bộ + một tập hợp nhỏ các token toàn cục (ví dụ: `[CLS]`) attend tới tất cả mọi người và được tất cả mọi người attend tới + các liên kết thưa ngẫu nhiên. Thực nghiệm cho thấy ngữ cảnh gấp 2 lần với chất lượng tương đương.
- **Native Sparse Attention (DeepSeek, 2025).** Học xem các khối `(Q, K)` nào quan trọng; bỏ qua các khối bằng 0 ở cấp độ kernel. Tương thích với FlashAttention.

Sparse attention là câu chuyện về kỹ thuật kernel. Toán học thì đơn giản (mask ma trận điểm); lợi ích đến từ việc không bao giờ nạp các mục bằng 0 vào SRAM. FlashAttention-3 và API FlexAttention 2026 biến các mẫu thưa tùy chỉnh thành tính năng hạng nhất trong PyTorch.

### Differential Attention (DIFF Transformer, 2024)

Attention thông thường gặp vấn đề "attention sink": softmax ép mỗi hàng phải có tổng bằng 1, vì vậy các token không muốn attend tới bất cứ thứ gì cụ thể sẽ đổ trọng số vào token đầu tiên (hoặc vài token đầu). Điều này chiếm dụng dung lượng đáng lẽ dành cho nội dung thực sự.

Differential attention khắc phục điều này bằng cách tính **hai** bản đồ attention và trừ chúng cho nhau:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

trong đó `λ` là một scalar đã học (thường là 0.5–0.8). A1 nắm bắt trọng số nội dung thực; A2 nắm bắt sink. Phép trừ triệt tiêu sink, tái phân bổ trọng số cho các token liên quan.

Kết quả báo cáo (Microsoft 2024): perplexity thấp hơn 5–10%, ngữ cảnh hiệu dụng dài hơn 1.5–2 lần ở cùng độ dài huấn luyện, truy xuất "kim đáy bể" sắc nét hơn.

### So sánh các biến thể

| Biến thể | Tính toán | KV cache | Chất lượng so với full | Sử dụng thực tế |
|---------|---------|----------|-----------------|----------------|
| Full attention | O(N²) | O(N) mỗi layer | baseline | layer mặc định của mọi mô hình |
| SWA (cửa sổ 1024) | O(N·W) | O(W) mỗi layer | -0.1 ppl, tốt với các layer global | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | O(N·√N) | hỗn hợp | tương tự SWA | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | O(N) xấp xỉ | hỗn hợp | bằng full ở ngữ cảnh 2× | BERT ngữ cảnh dài đời đầu |
| Native Sparse (DeepSeek-V3.2) | O(N · phần hoạt động) | O(N) | trong khoảng 0.05 ppl | DeepSeek-V3.2, 2025 |
| Differential | O(2·N²) | O(2N) | -5 đến -10% ppl | DIFF Transformer, các mô hình đầu 2026 |

```figure
gqa-kv-sharing
```

## Build It

Xem `code/main.py`. Chúng ta triển khai một bộ so sánh mask causal cho thấy full, SWA, local+strided và differential attention cạnh nhau trên một chuỗi giả lập.

### Bước 1: full causal mask (baseline)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Baseline từ Bài 07. Tam giác dưới; trọng số bằng 0 phía trên đường chéo.

### Bước 2: sliding window causal mask

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

Một tham số — `window`. Với `window >= n`, bạn khôi phục full causal attention. Với `window = 1`, mỗi token chỉ attend tới chính nó.

### Bước 3: local + strided sparse mask

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Cửa sổ cục bộ dày đặc cộng với mỗi token thứ `stride` quay ngược lại đầu chuỗi. Trường tiếp nhận tăng theo các bước log với các layer bổ sung.

### Bước 4: differential attention

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

Hai lượt attention, trừ đi với hệ số trộn đã học. Trong mã nguồn, chúng ta so sánh heatmap attention-sink của single vs differential và quan sát sink sụp đổ.

### Bước 5: Kích thước KV cache

In kích thước cache mỗi layer tại `N = 131072` cho từng biến thể. Các biến thể SWA và sparse giảm 10–100 lần. Differential tăng gấp đôi. Hãy trả hóa đơn bộ nhớ một cách có ý thức.

## Use It

Các mô hình sản xuất năm 2026:

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

FlexAttention trong PyTorch 2.5+ chấp nhận một hàm mask:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

Điều này biên dịch thành một kernel Triton tùy chỉnh. Tốc độ đạt trong khoảng 10% so với FlashAttention-3 cho các mẫu phổ biến, và hàm mask là một Python callable.

**Khi nào chọn cái nào:**

- **Pure full attention** — mọi layer lên tới ~16K ngữ cảnh, hoặc khi chất lượng truy xuất là tối quan trọng.
- **SWA + global mix** — ngữ cảnh dài (>32K), bị giới hạn bởi bộ nhớ khi huấn luyện và suy luận. Tiêu chuẩn 2026 cho trên 32K.
- **Sparse block attention** — kernel tùy chỉnh, mẫu tùy chỉnh. Dành riêng cho các khối lượng công việc chuyên biệt (truy xuất, âm thanh).
- **Differential attention** — bất kỳ khối lượng công việc nào mà sự nhiễm bẩn attention-sink gây hại (RAG ngữ cảnh dài, kim đáy bể).

## Ship It

Xem `outputs/skill-attention-variant-picker.md`. Kỹ năng này chọn cấu trúc liên kết attention cho một mô hình mới dựa trên độ dài ngữ cảnh mục tiêu, nhu cầu truy xuất và hồ sơ tính toán huấn luyện/suy luận.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Xác minh SWA tại `window=4` làm bằng 0 mọi thứ ngoài 4 token cuối mỗi hàng. Xác minh `window=n` tái tạo full causal attention giống hệt bit.
2. **Trung bình.** Triển khai causal SWA với `window=1024` trên đỉnh của capstone Bài 07. Huấn luyện 1.000 bước trên tinyshakespeare. Val loss suy giảm bao nhiêu so với full attention? Bộ nhớ đỉnh giảm bao nhiêu?
3. **Khó.** Triển khai mix layer 5:1 kiểu Gemma-3 (5 SWA, 1 global) trong mô hình capstone. So sánh loss, bộ nhớ và chất lượng tạo văn bản với các baseline pure-SWA và pure-global ở cùng tham số.
4. **Khó.** Triển khai differential attention với `λ` đã học cho mỗi head. Huấn luyện trên tác vụ truy xuất tổng hợp (một kim, 2.000 vật gây nhiễu). Đo độ chính xác truy xuất so với baseline single-attention ở cùng tham số.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Sliding window attention (SWA) | "Local attention" | Mỗi query attend tới `W` token cuối của nó; KV cache thu nhỏ còn `O(W)`. |
| Effective receptive field | "Mô hình nhìn xa bao nhiêu" | Trong stack SWA `L`-layer với cửa sổ `W`, lên tới `L × W` token. |
| Longformer / BigBird | "Local + global + random" | Các mẫu thưa với một vài token toàn cục luôn attend; cách tiếp cận ngữ cảnh dài đời đầu. |
| Native Sparse Attention | "DeepSeek's kernel trick" | Học độ thưa ở cấp độ khối; bỏ qua các khối bằng 0 ở cấp kernel trong khi vẫn giữ chất lượng. |
| Differential attention | "Hai bản đồ, một trừ" | DIFF Transformer: trừ đi `λ` lần bản đồ attention thứ hai từ bản đồ thứ nhất để triệt tiêu attention sinks. |
| Attention sink | "Trọng số rò rỉ vào token 0" | Chuẩn hóa softmax ép các hàng có tổng bằng 1; các query không thông tin đổ trọng số vào vị trí 0. |
| FlexAttention | "Mask-as-Python" | API PyTorch 2.5+ biên dịch các hàm mask tùy ý thành các kernel dạng FlashAttention. |
| Layer type mix | "5:1 SWA-to-global" | Xen kẽ các layer sparse và full attention trong một stack để giữ chất lượng ở bộ nhớ thấp hơn. |

## Đọc thêm

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — bài báo kinh điển về sliding-window + global-token.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — local + global + random.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — mẫu local+strided của OpenAI.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — mix 1:1 SWA:global.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — mix 5:1 với cửa sổ=1024 hiện là tiêu chuẩn mặc định.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — bài báo về DIFF Transformer.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — attention thưa đã học của DeepSeek-V3.2.
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — tài liệu tham khảo API cho mẫu mask-as-callable trong Use It.