# Xây dựng Transformer từ đầu — Đồ án cuối khóa

> Mười ba bài học. Một mô hình. Không đường tắt.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 đến 13. Không được bỏ qua.
**Time:** ~120 phút

## Vấn đề

Bạn đã đọc mọi bài báo. Bạn đã triển khai attention, multi-head splits, positional encodings, các khối encoder và decoder, BERT và GPT losses, MoE, KV cache. Bây giờ hãy làm cho chúng hoạt động cùng nhau trên một tác vụ thực tế.

Đồ án cuối khóa: huấn luyện một transformer decoder-only nhỏ từ đầu đến cuối trên tác vụ mô hình hóa ngôn ngữ ở cấp độ ký tự (character-level language modeling). Nó đọc Shakespeare. Nó tạo ra Shakespeare mới. Nó đủ nhỏ để huấn luyện trên laptop trong vòng chưa đầy 10 phút. Nó đủ chính xác để khi thay thế bằng tập dữ liệu lớn hơn và huấn luyện lâu hơn, bạn sẽ có một LM thực thụ.

Đây là "nanoGPT" của khóa học. Nó không phải là nguyên bản — hướng dẫn nanoGPT năm 2023 của Karpathy là bản triển khai tham chiếu mà mọi học viên đều viết ít nhất một lần. Chúng tôi lấy cấu trúc đó và tinh chỉnh lại dựa trên những gì chúng ta đã học.

## Khái niệm

![Transformer-from-scratch block diagram](../assets/capstone.svg)

Kiến trúc, được chú thích:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### Những gì chúng ta sẽ xây dựng

- `GPTConfig` — một nơi để cấu hình tất cả các siêu tham số.
- `MultiHeadAttention` — causal, theo batch, với tùy chọn đường dẫn kiểu Flash (`scaled_dot_product_attention` của PyTorch).
- `SwiGLUFFN` — FFN hiện đại.
- `Block` — pre-norm, attention được bao bọc bởi residual + FFN.
- `GPT` — embeddings, các khối xếp chồng, LM head, generate().
- Vòng lặp huấn luyện với AdamW, cosine LR, gradient clipping.
- Tokenizer cấp ký tự trên văn bản Shakespeare.

### Những gì chúng ta không xây dựng

- RoPE — đã triển khai về mặt khái niệm trong Bài 04. Ở đây chúng ta sử dụng learned positional embeddings cho đơn giản. Các bài tập yêu cầu bạn thay thế bằng RoPE.
- KV cache trong quá trình tạo văn bản — mỗi bước tạo sẽ tính toán lại attention trên toàn bộ prefix. Chậm hơn nhưng đơn giản hơn. Các bài tập yêu cầu bạn thêm KV cache.
- Flash Attention — PyTorch 2.0+ tự động phân phối nếu đầu vào khớp; chúng ta sử dụng `F.scaled_dot_product_attention`.
- MoE — một FFN duy nhất mỗi khối. Bạn đã thấy MoE trong Bài 11.

### Các chỉ số mục tiêu

Trên laptop Mac M2, một GPT 4 lớp, 4-head, d_model=128 được huấn luyện trong 2.000 bước trên `tinyshakespeare.txt`:

- Training loss hội tụ từ ~4.2 (ngẫu nhiên) xuống ~1.5 trong khoảng 6 phút.
- Đầu ra được lấy mẫu trông giống phong cách Shakespeare: các từ cổ, ngắt dòng, các tên riêng như "ROMEO:" xuất hiện.
- Val loss (trên 10% dữ liệu cuối cùng được giữ lại) bám sát training loss; không bị overfitting ở kích thước/ngân sách này.

```figure
n5-block-stack
```

## Xây dựng

Bài học này sử dụng PyTorch. Cài đặt `torch` (bản CPU là đủ). Xem `code/main.py`. Script xử lý:

- Tải xuống `tinyshakespeare.txt` nếu thiếu (hoặc đọc bản sao cục bộ).
- Byte-level char tokenizer.
- Chia tập train/val theo tỷ lệ 90/10.
- Vòng lặp huấn luyện với bf16 autocast trên phần cứng hỗ trợ.
- Lấy mẫu sau khi huấn luyện hoàn tất.

### Bước 1: dữ liệu

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 ký tự duy nhất. Từ vựng nhỏ. Vừa với vocab_size 4-byte. Không BPE, không rắc rối về tokenizer.

### Bước 2: mô hình

Xem `code/main.py`. Khối này là chuẩn mực từ Bài 05 — pre-norm, RMSNorm, SwiGLU, causal MHA. Số lượng tham số cho 4/4/128: ~800K.

### Bước 3: vòng lặp huấn luyện

Lấy một batch ngẫu nhiên gồm các cửa sổ token độ dài 256. Forward. Cross-entropy dịch chuyển một vị trí. Backward. Bước AdamW. Log. Lặp lại.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Bước 4: lấy mẫu

Với một prompt, liên tục forward, lấy mẫu từ top-p logits, thêm vào và tiếp tục. Dừng sau 500 token.

### Bước 5: đọc đầu ra

Sau 2.000 bước:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

Không phải Shakespeare. Nhưng mang phong cách Shakespeare. Một chiến thắng rõ ràng cho ~800K tham số và 6 phút trên laptop.

## Sử dụng

Đồ án này là một kiến trúc tham chiếu. Ba phần mở rộng để đưa nó lên mức thực tế:

1. **Thay đổi tokenizer.** Sử dụng BPE (ví dụ: `tiktoken.get_encoding("cl100k_base")`). Kích thước từ vựng tăng từ 65 lên ~50.000. Dung lượng mô hình cần tăng lên để bù đắp.
2. **Huấn luyện trên tập dữ liệu lớn hơn.** Sử dụng `OpenWebText` hoặc `fineweb-edu` (HuggingFace). 10B token trên một card A100 mất ~24 giờ cho một GPT 125M tham số.
3. **Thêm RoPE + KV cache + Flash Attention.** Các bài tập bên dưới sẽ hướng dẫn bạn từng bước.

Kết quả cuối cùng là một GPT 125M tham số tạo ra tiếng Anh trôi chảy. Không phải là mô hình tiên phong (frontier model). Nhưng cùng một lộ trình code — chỉ lớn hơn — là những gì Karpathy, EleutherAI và Allen Institute sử dụng để huấn luyện các checkpoint nghiên cứu vào năm 2026.

## Triển khai

Xem `outputs/skill-transformer-review.md`. Kỹ năng này kiểm tra tính đúng đắn của bản triển khai transformer-từ-đầu trên tất cả 13 bài học trước đó.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Xác minh val loss ở bước cuối cùng của mô hình đã huấn luyện dưới 2.0. Thay đổi `max_steps` từ 2.000 thành 5.000 — val loss có tiếp tục cải thiện không?
2. **Trung bình.** Thay thế learned positional embeddings bằng RoPE. Áp dụng phép quay cho Q và K bên trong `MultiHeadAttention`. Huấn luyện và xác minh val loss thấp ít nhất bằng mức cũ.
3. **Trung bình.** Triển khai KV cache trong vòng lặp lấy mẫu. Tạo 500 token với và không có cache. Thời gian thực tế (wall-clock) sẽ cải thiện 5–20 lần trên laptop.
4. **Khó.** Thêm một head thứ hai vào mô hình để dự đoán token tiếp theo-của-tiếp theo (MTP — Multi-Token Prediction từ DeepSeek-V3). Huấn luyện đồng thời. Nó có giúp ích không?
5. **Khó.** Thay thế FFN đơn lẻ mỗi khối bằng MoE 4 chuyên gia. Router + top-2 routing. Xem val loss thay đổi thế nào khi số lượng tham số hoạt động tương đương.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| nanoGPT | "Repo hướng dẫn của Karpathy" | Code huấn luyện transformer decoder-only tối giản, ~300 dòng; tài liệu tham khảo chuẩn. |
| tinyshakespeare | "Tập dữ liệu đồ chơi tiêu chuẩn" | ~1.1 MB văn bản; mọi hướng dẫn char-LM từ năm 2015 đều sử dụng nó. |
| Tied embeddings | "Chia sẻ ma trận đầu vào/đầu ra" | Trọng số LM head = chuyển vị của ma trận token embedding; tiết kiệm tham số, cải thiện chất lượng. |
| bf16 autocast | "Mẹo độ chính xác huấn luyện" | Chạy forward/back bằng bf16, giữ trạng thái optimizer bằng fp32; tiêu chuẩn từ năm 2021. |
| Gradient clipping | "Ngăn chặn đột biến" | Giới hạn chuẩn grad toàn cục ở mức 1.0; ngăn chặn sự bùng nổ khi huấn luyện. |
| Cosine LR schedule | "Mặc định từ 2020+" | LR tăng tuyến tính (warmup) sau đó giảm theo hình cosine xuống 10% đỉnh. |
| MFU | "Hiệu suất sử dụng FLOP của mô hình" | FLOP đạt được / đỉnh lý thuyết; 40% dense, 30% MoE là mức mạnh vào năm 2026. |
| Val loss | "Loss trên dữ liệu giữ lại" | Cross-entropy trên dữ liệu mô hình chưa từng thấy; công cụ phát hiện overfit. |

## Đọc thêm

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — bản triển khai chú thích kinh điển.