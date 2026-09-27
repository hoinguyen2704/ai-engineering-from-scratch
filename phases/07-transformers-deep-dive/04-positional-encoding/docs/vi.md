# Positional Encoding — Sinusoidal, RoPE, ALiBi

> Attention có tính bất biến với hoán vị (permutation-invariant). "The cat sat on the mat" và "mat the on sat cat the" tạo ra cùng một đầu ra nếu không có tín hiệu vị trí. Ba thuật toán giải quyết vấn đề này — mỗi thuật toán có một cách tiếp cận khác nhau về ý nghĩa của "vị trí".

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 phút

## Vấn đề

Scaled dot-product attention không phân biệt được thứ tự. Ma trận attention `softmax(Q K^T / √d) V` được tính toán từ độ tương đồng theo cặp. Nếu hoán vị các hàng của `X`, các hàng của đầu ra cũng sẽ bị hoán vị tương ứng. Không có gì bên trong cơ chế attention quan tâm đến vị trí.

Điều này không phải là lỗi trong mô hình bag-of-words. Nhưng đối với ngôn ngữ, mã nguồn, âm thanh, video — bất cứ thứ gì mà thứ tự mang ý nghĩa — thì đây là một lỗi chí mạng.

Giải pháp là đưa thông tin vị trí vào các embedding bằng cách nào đó. Có ba thời kỳ của các giải pháp:

1. **Absolute sinusoidal** (Vaswani 2017). Cộng `sin/cos` của vị trí vào embedding. Đơn giản, không cần học, khả năng ngoại suy (extrapolation) kém khi vượt quá độ dài huấn luyện.
2. **RoPE — Rotary Position Embeddings** (Su 2021). Xoay các vector Q và K theo một góc tỉ lệ với vị trí. Mã hóa vị trí *tương đối* trực tiếp vào dot product. Thống trị vào năm 2026.
3. **ALiBi — Attention with Linear Biases** (Press 2022). Bỏ qua hoàn toàn embedding; thêm một hệ số phạt tuyến tính theo từng head vào điểm số attention dựa trên khoảng cách. Khả năng ngoại suy độ dài cực tốt.

Tính đến năm 2026, về cơ bản mọi mô hình mở tiên phong đều sử dụng RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. Một số ít các mô hình ngữ cảnh dài (long-context) sử dụng ALiBi hoặc các biến thể hiện đại của nó. Absolute sinusoidal hiện chỉ còn mang tính lịch sử.

## Khái niệm

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### Absolute sinusoidal

Tính toán trước một ma trận cố định `PE` có hình dạng `(max_len, d_model)`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

Sau đó `X' = X + PE[:N]` trước khi thực hiện attention. Mỗi chiều là một hàm sin ở một tần số khác nhau. Mô hình học cách đọc vị trí từ mô hình pha (phase pattern). Thất bại khi vượt quá `max_len`: không có gì báo cho mô hình biết điều gì xảy ra ở vị trí 2048 khi nó chỉ thấy các vị trí từ 0–2047.

### RoPE

Xoay các vector Q và K (không phải embedding). Đối với một cặp chiều `(2i, 2i+1)`:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

Áp dụng cùng phép xoay cho các key với vị trí `pos_k`. Dot product `q'_m · k'_n` trở thành một hàm chỉ phụ thuộc vào `(m - n)`. Đó là: **điểm số attention chỉ phụ thuộc vào khoảng cách tương đối**, mặc dù phép xoay được thực hiện dựa trên các vị trí tuyệt đối. Một thủ thuật rất đẹp.

Mở rộng RoPE: `base` có thể được điều chỉnh tỉ lệ (NTK-aware, YaRN, LongRoPE) để ngoại suy sang các ngữ cảnh dài hơn mà không cần huấn luyện lại. Llama 3 đã mở rộng từ 8K lên 128K ngữ cảnh theo cách này.

### ALiBi

Bỏ qua thủ thuật embedding. Điều chỉnh trực tiếp điểm số attention:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

Trong đó `m_h` là độ dốc riêng cho từng head (ví dụ: `1 / 2^(8·h/H)`). Các token gần được tăng cường; các token xa bị phạt. Không tốn chi phí huấn luyện. Bài báo cho thấy khả năng ngoại suy độ dài vượt trội hơn sinusoidal và tương đương với RoPE ở độ dài huấn luyện gốc.

### Chọn gì vào năm 2026

| Biến thể | Ngoại suy | Chi phí huấn luyện | Được sử dụng bởi |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | kém | miễn phí | transformer gốc, BERT đời đầu |
| Learned absolute | không có | rất nhỏ | GPT-2, GPT-3 |
| RoPE | tốt với scaling | miễn phí | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | xuất sắc | giai đoạn fine-tune | Qwen2-1M, Llama 3.1 128K |
| ALiBi | xuất sắc | miễn phí | BLOOM, MPT, Baichuan |

RoPE chiến thắng vì nó tích hợp vào attention mà không làm thay đổi kiến trúc, mã hóa vị trí tương đối, và siêu tham số `base` của nó cung cấp một nút điều chỉnh sạch sẽ cho việc fine-tune ngữ cảnh dài.

```figure
rope-explorer
```

## Xây dựng

### Bước 1: sinusoidal encoding

Xem `code/main.py`. Một phép tính 4 dòng:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

Thêm giá trị này vào ma trận embedding trước lớp attention đầu tiên.

### Bước 2: RoPE áp dụng cho Q, K

RoPE hoạt động trực tiếp (in-place) trên Q và K. Đối với mỗi cặp chiều:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Quan trọng: áp dụng cùng một hàm cho Q tại vị trí `m` và K tại vị trí `n`. Dot product của chúng sẽ nhận thêm một hệ số `cos((m-n)·θ_i)` trên mỗi cặp tọa độ. Attention học được vị trí tương đối một cách tự nhiên.

### Bước 3: ALiBi slopes và bias

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

Thêm `bias[h]` vào ma trận điểm số attention `(seq_len, seq_len)` của head `h`, sau đó thực hiện softmax.

### Bước 4: xác minh tính chất khoảng cách tương đối của RoPE

Chọn hai vector ngẫu nhiên `a, b`. Xoay theo `(pos_a, pos_b)`. Sau đó theo `(pos_a + k, pos_b + k)`. Cả hai dot product phải khớp nhau trong phạm vi sai số dấu phẩy động. Tính chất đó chính là mục đích của RoPE — nó bất biến với độ lệch tuyệt đối, chỉ khoảng cách tương đối mới quan trọng.

## Sử dụng

PyTorch 2.5+ cung cấp các tiện ích RoPE trong `torch.nn.functional`. Hầu hết mã nguồn sản xuất sử dụng `flash_attn` hoặc `xformers`, nơi RoPE được áp dụng bên trong kernel attention.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Các thủ thuật ngữ cảnh dài vào năm 2026:**

- **NTK-aware interpolation.** Thay đổi tỉ lệ `base` thành `base * (scale_factor)^(d/(d-2))` khi mở rộng từ 4K lên 16K+.
- **YaRN.** Phép nội suy thông minh hơn giúp bảo toàn entropy của attention trên các ngữ cảnh dài. Llama 3.1 128K sử dụng cách này.
- **LongRoPE.** Phương pháp năm 2024 của Microsoft sử dụng tìm kiếm tiến hóa để chọn các hệ số tỉ lệ theo từng chiều. Phi-3-Long sử dụng cách này.
- **Position interpolation + fine-tuning.** Chỉ cần thu nhỏ các vị trí theo hệ số mở rộng và fine-tune trong 1–5B token. Hiệu quả đến bất ngờ.

## Triển khai

Xem `outputs/skill-positional-encoding-picker.md`. Kỹ năng này giúp chọn chiến lược mã hóa cho một mô hình mới dựa trên độ dài ngữ cảnh mục tiêu, nhu cầu ngoại suy và ngân sách huấn luyện.

## Bài tập

1. **Dễ.** Vẽ ma trận sinusoidal `PE` dưới dạng heatmap cho `max_len=512, d=128`. Xác nhận mô hình "các sọc rộng dần khi chỉ số chiều tăng lên".
2. **Trung bình.** Triển khai NTK-aware RoPE scaling. Huấn luyện một LM nhỏ trên các chuỗi có độ dài 256, sau đó kiểm tra trên độ dài 1024 với và không có scaling. Đo lường perplexity.
3. **Khó.** Triển khai ALiBi và RoPE trong cùng một module attention. Huấn luyện một transformer 4 lớp trên tác vụ copy với chuỗi độ dài 512. Ngoại suy lên 2048 tại thời điểm kiểm tra. So sánh sự suy giảm hiệu năng.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Positional encoding | "Cho attention biết thứ tự" | Bất kỳ tín hiệu nào được thêm vào embedding hoặc attention để mã hóa vị trí. |
| Sinusoidal | "Cái gốc" | `sin/cos` ở các tần số hình học được thêm vào embedding; không ngoại suy được. |
| RoPE | "Rotary embeddings" | Xoay Q, K theo góc phụ thuộc vị trí; dot product mã hóa khoảng cách tương đối. |
| ALiBi | "Thủ thuật bias tuyến tính" | Thêm `-m·\|i-j\|` vào điểm số attention; không cần embedding, ngoại suy cực tốt. |
| base | "Nút vặn của RoPE" | Hệ số tỉ lệ tần số trong RoPE; tăng lên để mở rộng ngữ cảnh khi inference. |
| NTK-aware | "Thủ thuật scaling RoPE" | Thay đổi tỉ lệ `base` để các chiều tần số cao không bị nén khi ngữ cảnh mở rộng. |
| YaRN | "Cái xịn hơn" | Nội suy+ngoại suy theo từng chiều giúp bảo toàn entropy của attention. |
| Extrapolation | "Hoạt động ngoài độ dài huấn luyện" | Cơ chế vị trí có thể tạo ra đầu ra đúng sau `max_len` đã thấy trong huấn luyện không? |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — sinusoidal gốc.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — bài báo về RoPE.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — state of the art về RoPE scaling.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — bài báo về ngữ cảnh dài của Llama 2 từ Meta.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — phương pháp của Microsoft được sử dụng bởi Phi-3-Long và được trích dẫn trong phần Sử dụng.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — các triển khai chuẩn sản xuất của mọi sơ đồ RoPE scaling (default, linear, dynamic, YaRN, LongRoPE, Llama-3).