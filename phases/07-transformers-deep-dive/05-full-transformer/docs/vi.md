# The Full Transformer — Encoder + Decoder

> Attention là ngôi sao. Mọi thứ khác — kết nối tắt (residual), chuẩn hóa (normalization), feed-forward, cross-attention — đều là giàn giáo cho phép bạn xếp chồng nó lên thật sâu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 phút

## Vấn đề

Một lớp attention đơn lẻ chỉ là bộ trích xuất đặc trưng, không phải là một mô hình. Một phép nhân ma trận (matmul) mỗi lớp không đủ năng lực cho ngôn ngữ. Bạn cần độ sâu — và độ sâu sẽ bị phá vỡ nếu không có hệ thống hỗ trợ phù hợp.

Bài báo của Vaswani năm 2017 đã đóng gói sáu quyết định thiết kế biến một lớp attention thành một khối có thể xếp chồng. Mọi transformer kể từ đó — chỉ encoder (BERT), chỉ decoder (GPT), encoder-decoder (T5) — đều kế thừa cùng một bộ khung này. Đến năm 2026, các khối này đã được tinh chỉnh (RMSNorm, SwiGLU, pre-norm, RoPE) nhưng bộ khung vẫn giống hệt.

Bài học này nói về bộ khung đó. Các bài học tiếp theo sẽ chuyên biệt hóa nó — bài 06 cho encoder, 07 cho decoder, 08 cho encoder-decoder.

## Khái niệm

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### Sáu thành phần

1. **Embedding + tín hiệu vị trí.** Tokens → vectors. Vị trí được đưa vào thông qua RoPE (hiện đại) hoặc sinusoidal (cổ điển).
2. **Self-attention.** Mọi vị trí đều chú ý đến mọi vị trí khác. Được mask trong các decoder.
3. **Feed-forward network (FFN).** MLP hai lớp theo từng vị trí: `W_2 · activation(W_1 · x)`. Tỷ lệ mở rộng mặc định là 4×.
4. **Residual connection.** `x + sublayer(x)`. Nếu không có cái này, gradient sẽ biến mất sau khoảng 6 lớp.
5. **Layer normalization.** `LayerNorm` hoặc `RMSNorm` (hiện đại). Giúp ổn định luồng residual.
6. **Cross-attention (chỉ dành cho decoder).** Queries đến từ decoder, keys và values đến từ đầu ra của encoder.

Hãy quan sát một vector chảy qua một khối: attention trộn lẫn giữa các vị trí, residual mang nó về phía trước, FFN biến đổi nó, và norm giữ cho luồng ổn định.

```figure
transformer-block
```

### Khối Encoder (được sử dụng bởi BERT, T5 encoder)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Encoder có tính hai chiều (bidirectional). Không có masking. Tất cả các vị trí đều nhìn thấy tất cả các vị trí.

### Khối Decoder (được sử dụng bởi GPT, T5 decoder)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Decoder có ba lớp con mỗi khối. Lớp ở giữa — cross-attention — là nơi duy nhất thông tin chảy từ encoder sang decoder. Trong kiến trúc chỉ có decoder (GPT), cross-attention bị lược bỏ và bạn chỉ còn masked self-attention + FFN.

### Pre-norm vs post-norm

Bài báo gốc: `x + sublayer(LN(x))` vs `LN(x + sublayer(x))`. Post-norm đã mất dần sự ưa chuộng từ khoảng năm 2019 — nó khó huấn luyện sâu hơn nếu không có warmup cẩn thận. Pre-norm (`LN` *trước* lớp con) là mặc định của năm 2026: Llama, Qwen, GPT-3+, Mistral đều sử dụng nó.

### Khối hiện đại hóa năm 2026

Vaswani 2017 sử dụng LayerNorm + ReLU. Các stack hiện đại đã thay thế cả hai. Các khối trong thực tế trông như sau:

| Thành phần | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU sử dụng ba ma trận, tổng số tham số tương đương) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (hoặc MLA) |
| Bias terms | Có | Không |

RMSNorm loại bỏ việc trừ giá trị trung bình của LayerNorm (bớt một phép trừ), giúp tiết kiệm tính toán và ổn định tương đương về mặt thực nghiệm. SwiGLU (`Swish(W1 x) ⊙ W3 x`) liên tục vượt trội hơn FFN ReLU/GELU khoảng 0.5 điểm ppl trong các bài báo Llama, PaLM và Qwen.

### Số lượng tham số

Đối với một khối với `d_model = d` và FFN expansion `r`:

- MHA: `4 · d²` (các phép chiếu Q, K, V, O)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- Norms: không đáng kể

Tại `d = 4096, r = 2.6, layers = 32` (xấp xỉ Llama 3 8B), tổng cộng: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B` (cộng thêm embeddings và head). Khớp với các con số đã công bố.

## Xây dựng

### Bước 1: các khối xây dựng

Sử dụng lớp `Matrix` nhỏ từ Bài 03 (được sao chép vào tệp này để độc lập):

- `layer_norm(x, eps=1e-5)` — trừ trung bình, chia cho độ lệch chuẩn.
- `rms_norm(x, eps=1e-6)` — chia cho RMS. Không trừ trung bình.
- `gelu(x)` và `silu(x) * W3 x` (SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`.
- `encoder_block(x, params)` và `decoder_block(x, enc_out, params)`.

Xem `code/main.py` để biết cách kết nối đầy đủ.

### Bước 2: kết nối encoder 2 lớp và decoder 2 lớp

Xếp chồng chúng. Truyền đầu ra của encoder vào mọi cross-attention của decoder. Thêm một LN cuối cùng trước phép chiếu đầu ra.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Bước 3: chạy forward trên ví dụ đồ chơi

Đưa một nguồn 6-token và một đích 5-token qua. Xác minh hình dạng đầu ra là `(5, vocab)`. Không huấn luyện — bài học này tập trung vào kiến trúc, không phải hàm mất mát.

### Bước 4: thay thế bằng RMSNorm + SwiGLU

Thay thế LayerNorm và ReLU-FFN bằng RMSNorm và SwiGLU. Xác nhận các hình dạng vẫn khớp. Đây là quá trình hiện đại hóa năm 2026 với một lần thay thế hàm.

## Sử dụng

Các triển khai tham chiếu PyTorch/TF: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`. Nhưng hầu hết mã nguồn sản xuất năm 2026 đều tự viết khối riêng vì:

- Flash Attention được gọi bên trong attention, không thông qua `nn.MultiheadAttention`.
- GQA / MLA không có trong tham chiếu stdlib.
- RoPE, RMSNorm, SwiGLU không phải là mặc định của PyTorch.

HF `transformers` có các khối tham chiếu sạch mà bạn nên đọc: `modeling_llama.py` là khối decoder-only chuẩn của năm 2026. Nó dài khoảng 500 dòng và rất đáng để xem qua một lần.

**Encoder vs decoder vs encoder-decoder — khi nào nên chọn:**

| Nhu cầu | Chọn | Ví dụ |
|------|------|---------|
| Phân loại, embeddings, QA trên văn bản | Encoder-only | BERT, DeBERTa, ModernBERT |
| Tạo văn bản, chat, code, suy luận | Decoder-only | GPT, Llama, Claude, Qwen |
| Đầu vào cấu trúc → đầu ra cấu trúc (dịch, tóm tắt) | Encoder-decoder | T5, BART, Whisper |

Decoder-only đã chiến thắng trong lĩnh vực ngôn ngữ vì nó mở rộng quy mô sạch nhất và xử lý tốt cả việc hiểu lẫn tạo văn bản. Encoder-decoder vẫn tốt nhất khi đầu vào có danh tính "chuỗi nguồn" rõ ràng (dịch thuật, nhận dạng giọng nói, các tác vụ có cấu trúc).

## Triển khai

Xem `outputs/skill-transformer-block-reviewer.md`. Kỹ năng này đánh giá việc triển khai khối transformer mới dựa trên các mặc định của năm 2026 và gắn cờ các phần còn thiếu (pre-norm, RoPE, RMSNorm, GQA, tỷ lệ mở rộng FFN).

## Bài tập

1. **Dễ.** Đếm các tham số trong encoder_block của bạn tại `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Xác thực bằng cách triển khai khối và sử dụng `sum(p.numel() for p in block.parameters())`.
2. **Trung bình.** Chuyển từ post-norm sang pre-norm. Khởi tạo cả hai và đo lường chuẩn kích hoạt (activation norm) sau 12 lớp xếp chồng trên đầu vào ngẫu nhiên. Các kích hoạt của post-norm sẽ bùng nổ; của pre-norm sẽ bị giới hạn.
3. **Khó.** Triển khai encoder-decoder 4 lớp trên một tác vụ sao chép đồ chơi (sao chép `x` đảo ngược). Huấn luyện 100 bước. Báo cáo loss. Thay thế bằng RMSNorm + SwiGLU + RoPE — loss có giảm không?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Block | "Một lớp transformer" | Chồng norm + attention + norm + FFN, được bao bọc trong các kết nối residual. |
| Residual | "Skip connection" | `x + f(x)` đầu ra; cho phép luồng gradient chảy qua các stack sâu. |
| Pre-norm | "Chuẩn hóa trước, không phải sau" | Hiện đại: `x + sublayer(LN(x))`. Huấn luyện sâu hơn mà không cần các kỹ thuật warmup phức tạp. |
| RMSNorm | "LayerNorm không có giá trị trung bình" | Chia cho RMS; bớt một phép toán, độ ổn định thực nghiệm tương đương. |
| SwiGLU | "FFN mà mọi người đã chuyển sang" | `Swish(W1 x) ⊙ W3 x → W2`. Đánh bại ReLU/GELU về LM ppl. |
| Cross-attention | "Cách decoder nhìn thấy encoder" | MHA với Q từ decoder, K/V từ đầu ra encoder. |
| FFN expansion | "Độ rộng của MLP ở giữa" | Tỷ lệ của hidden-size so với d_model, thường là 4 (LayerNorm) hoặc 2.6 (SwiGLU). |
| Bias-free | "Bỏ các số hạng +b" | Các stack hiện đại bỏ qua bias trong các lớp tuyến tính; cải thiện nhẹ ppl, mô hình nhỏ hơn. |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — đặc tả khối gốc.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — tại sao pre-norm đánh bại post-norm ở độ sâu lớn.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — bài báo về SwiGLU.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — khối decoder-only chuẩn năm 2026.