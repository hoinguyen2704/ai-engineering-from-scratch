# Cơ chế Attention — Bước ngoặt đột phá

> Decoder ngừng việc phải "nheo mắt" nhìn vào một bản tóm tắt nén và bắt đầu quan sát toàn bộ nguồn. Mọi thứ sau dấu mốc này đều là attention cộng với kỹ thuật (engineering).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 09 (Sequence-to-Sequence Models)
**Time:** ~45 minutes

## Vấn đề

Bài học 09 đã kết thúc với một thất bại có thể dự đoán được. Một mô hình encoder-decoder sử dụng GRU được huấn luyện trên một nhiệm vụ copy mô phỏng (toy copy task) đạt độ chính xác từ 89% ở độ dài 5 xuống mức gần như ngẫu nhiên ở độ dài 80. Lý do nằm ở cấu trúc, không phải lỗi huấn luyện: mọi mẩu thông tin mà encoder thu thập được đều phải nằm gọn trong một trạng thái ẩn (hidden state) có kích thước cố định, và decoder không bao giờ nhìn thấy gì khác ngoài nó.

Bahdanau, Cho, và Bengio đã công bố một giải pháp sửa lỗi chỉ với ba dòng code vào năm 2014. Thay vì chỉ đưa cho decoder trạng thái cuối cùng của encoder, hãy giữ lại mọi trạng thái của encoder. Tại mỗi bước của decoder, hãy tính toán một giá trị trung bình có trọng số của các trạng thái encoder, trong đó các trọng số cho biết "decoder cần tập trung vào vị trí encoder `i` đến mức nào tại thời điểm này?". Giá trị trung bình có trọng số đó chính là ngữ cảnh (context), và nó thay đổi theo từng bước của decoder.

Đó là toàn bộ ý tưởng. Transformers đã mở rộng nó. Self-attention áp dụng nó cho một chuỗi đơn lẻ. Multi-head attention chạy nó song song. Nhưng phiên bản năm 2014 đã phá vỡ nút thắt cổ chai, và một khi bạn đã nắm vững nó, việc chuyển sang transformers chỉ là vấn đề kỹ thuật (engineering), không phải về mặt khái niệm.

## Khái niệm

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

Tại mỗi bước của decoder `t`:

1. Sử dụng trạng thái ẩn trước đó của decoder `s_{t-1}` làm **query**.
2. Tính điểm tương đồng (score) của nó với mọi trạng thái ẩn của encoder `h_1, ..., h_T`. Mỗi vị trí encoder nhận một giá trị vô hướng (scalar).
3. Dùng hàm Softmax cho các điểm số để nhận được các trọng số attention `α_{t,1}, ..., α_{t,T}` có tổng bằng 1.
4. Vector ngữ cảnh (context vector) `c_t = Σ α_{t,i} * h_i` là trung bình có trọng số của các trạng thái encoder.
5. Decoder nhận `c_t` cùng với token đầu ra trước đó để tạo ra token tiếp theo.

Giá trị trung bình có trọng số chính là điểm mấu chốt. Khi decoder cần dịch "Je" sang "I", nó sẽ đặt trọng số cao cho trạng thái encoder tại vị trí "Je" và thấp cho các vị trí khác. Khi nó cần từ "not", nó sẽ đặt trọng số cao cho "pas". Vector ngữ cảnh tái định hình qua từng bước.

## Hình dạng/Kích thước (thứ khiến mọi người bối rối)

Đây là nơi mọi triển khai attention dễ sai sót nhất trong lần đầu tiên. Hãy đọc kỹ.

| Thành phần | Hình dạng (Shape) | Ghi chú |
|-------|-------|-------|
| Các trạng thái ẩn Encoder `H` | `(T_enc, d_h)` | Nếu là BiLSTM, `d_h = 2 * d_hidden` |
| Trạng thái ẩn Decoder `s_{t-1}` | `(d_s,)` | Một vector |
| Điểm attention (score) `e_{t,i}` | scalar | Một giá trị cho mỗi vị trí encoder |
| Trọng số attention `α_{t,i}` | scalar | Sau khi Softmax trên toàn bộ `i` |
| Vector ngữ cảnh `c_t` | `(d_h,)` | Cùng hình dạng với một trạng thái encoder |

**Điểm số Bahdanau (cộng tính).** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}` có hình dạng `(d_s,)`, `h_i` có hình dạng `(d_h,)`.
- `W_a` có hình dạng `(d_attn, d_s)`. `U_a` có hình dạng `(d_attn, d_h)`.
- Tổng của chúng bên trong hàm tanh có hình dạng `(d_attn,)`.
- `v_α` có hình dạng `(d_attn,)`. Tích vô hướng (inner product) với `v_α` thu gọn thành một đại lượng vô hướng. **Đây là những gì `v_α` thực hiện.** Nó không phải phép màu. Đó là phép chiếu (projection) biến một vector có kích thước attention-dim thành một điểm số vô hướng.

**Điểm số Luong (nhân tính).** Ba biến thể:

- `dot`: `e_{t,i} = s_t^T * h_i`. Yêu cầu `d_s == d_h`. Ràng buộc cứng. Bỏ qua nếu encoder của bạn là hai chiều (bidirectional).
- `general`: `e_{t,i} = s_t^T * W * h_i` với `W` có hình dạng `(d_s, d_h)`. Loại bỏ ràng buộc bằng nhau về kích thước.
- `concat`: về cơ bản là dạng Bahdanau. Hiếm khi được sử dụng vì hai dạng đầu tiên tiết kiệm chi phí tính toán hơn.

**Một điểm cần lưu ý giữa Bahdanau / Luong.** Bahdanau sử dụng `s_{t-1}` (trạng thái decoder *trước khi* tạo từ hiện tại). Luong sử dụng `s_t` (trạng thái *sau khi*). Việc nhầm lẫn giữa chúng tạo ra các gradient sai lệch một cách tinh vi và cực kỳ khó debug. Hãy chọn một bài báo và tuân thủ quy ước của nó.

```figure
attention-heatmap
```

## Triển khai

### Bước 1: attention cộng tính (Bahdanau)

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

Kiểm tra hình dạng (shape) của bạn với bảng trên. `encoder_states` có hình dạng `(T_enc, d_h)`. `projected_enc` có hình dạng `(T_enc, d_attn)`. `projected_dec` có hình dạng `(d_attn,)` và thực hiện broadcasting. `combined` có hình dạng `(T_enc, d_attn)`. `scores` có hình dạng `(T_enc,)`. `weights` có hình dạng `(T_enc,)`. `context` có hình dạng `(d_h,)`. Triển khai thôi.

### Bước 2: Luong dot và general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

Mỗi loại chỉ cần ba dòng code. Đây là lý do tại sao bài báo của Luong lại gây tiếng vang. Cùng độ chính xác trên hầu hết các tác vụ nhưng code ít hơn nhiều.

### Bước 3: một ví dụ số học cụ thể

Giả sử có ba trạng thái encoder (tương ứng "cat", "sat", "mat") và một trạng thái decoder khớp nhất với trạng thái đầu tiên, phân phối attention sẽ tập trung vào vị trí 0. Nếu trạng thái decoder chuyển sang khớp với trạng thái cuối cùng, attention sẽ chuyển sang vị trí 2. Vector ngữ cảnh sẽ theo sát sự thay đổi này.

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

Hàng đầu tiên thắng. Sau đó, di chuyển trạng thái decoder lại gần trạng thái encoder thứ ba và quan sát các trọng số thay đổi. Chỉ vậy thôi. Attention là sự căn chỉnh (alignment) tường minh.

### Bước 4: tại sao đây là cầu nối tới transformers

Chuyển đổi ngôn ngữ ở trên sang Q/K/V:

- **Query** = trạng thái decoder `s_{t-1}`
- **Key** = các trạng thái encoder (thứ chúng ta tính điểm tương đồng)
- **Value** = các trạng thái encoder (thứ chúng ta tính trọng số và tổng hợp)

Trong attention cổ điển, keys và values là một. Self-attention tách biệt chúng: bạn có thể query một chuỗi với chính nó, với các phép chiếu (projections) đã được học khác nhau cho K và V. Multi-head attention chạy nó song song với các phép chiếu khác nhau. Transformers xếp chồng toàn bộ giai đoạn này nhiều lần và loại bỏ RNNs.

Toán học là như nhau. Hình dạng (shapes) là như nhau. Bước nhảy vọt về mặt lý thuyết từ Bahdanau attention sang scaled dot-product attention chủ yếu nằm ở cách ký hiệu.

## Sử dụng

PyTorch và TensorFlow hỗ trợ attention trực tiếp.

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

Đó là một lớp attention của transformer. Query một batch gồm 5 vị trí, key/value một batch gồm 10 vị trí, mỗi vị trí 128 chiều, 8 heads. `output` là các query mới đã được tăng cường ngữ cảnh. `weights` là ma trận căn chỉnh 5x10 mà bạn có thể trực quan hóa.

### Khi nào attention cổ điển vẫn quan trọng

- Mục đích sư phạm. Phiên bản dựa trên RNN, đơn head, đơn lớp giúp mọi khái niệm trở nên rõ ràng.
- Các tác vụ chuỗi trên thiết bị (on-device) nơi transformers không vừa.
- Bất kỳ bài báo nào từ 2014-2017. Bạn sẽ hiểu sai nếu không biết quy ước của Bahdanau.
- Phân tích căn chỉnh chi tiết trong dịch máy (MT). Các trọng số attention thô là một công cụ giải thích được (interpretability) ngay cả trên các mô hình transformer, và việc đọc chúng đòi hỏi phải biết chúng là gì.

### Cái bẫy "trọng số attention là lời giải thích"

Các trọng số attention trông có vẻ dễ giải thích. Chúng là các trọng số có tổng bằng 1 trên các vị trí; bạn có thể vẽ biểu đồ; giá trị cao nghĩa là "đã tập trung vào đây". Những người bình duyệt (reviewers) rất thích chúng.

Tuy nhiên, chúng không dễ giải thích như vẻ ngoài. Jain và Wallace (2019) đã chỉ ra rằng các phân phối attention có thể bị hoán vị và thay thế bằng các lựa chọn thay thế tùy ý mà không làm thay đổi dự đoán của mô hình đối với một số tác vụ. Đừng bao giờ báo cáo trọng số attention như bằng chứng của lập luận mà không có kiểm tra loại bỏ (ablation) hoặc phản thực tế (counterfactual).

## Hoàn tất

Lưu thành `outputs/prompt-attention-shapes.md`:

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## Bài tập

1. **Dễ.** Triển khai `softmax` masking để các padding tokens trong encoder nhận trọng số attention bằng không. Kiểm tra trên một batch với các chuỗi có độ dài thay đổi.
2. **Trung bình.** Thêm multi-head attention vào dạng Luong `general`. Chia `d_h` thành `n_heads` nhóm, chạy attention cho mỗi head, sau đó nối lại. Xác minh rằng trường hợp single-head khớp với triển khai trước đó của bạn.
3. **Khó.** Huấn luyện một encoder-decoder GRU với Bahdanau attention trên nhiệm vụ copy mô phỏng từ bài học 09. Vẽ biểu đồ độ chính xác so với độ dài chuỗi. So sánh với mô hình cơ sở không dùng attention. Bạn sẽ thấy khoảng cách nới rộng khi độ dài tăng lên, xác nhận rằng attention đã giải tỏa được nút thắt cổ chai.

## Thuật ngữ chính

| Thuật ngữ | Mọi người thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Attention | Quan sát mọi thứ | Trung bình có trọng số của một chuỗi giá trị (value), trọng số được tính từ độ tương đồng query-key. |
| Query, Key, Value | QKV | Ba phép chiếu: Q đưa ra yêu cầu, K là thứ để khớp, V là thứ để trả về. |
| Additive attention | Bahdanau | Điểm số feed-forward: `v^T tanh(W q + U k)`. |
| Multiplicative attention | Luong dot / general | Điểm số là `q^T k` hoặc `q^T W k`. Rẻ hơn, cùng độ chính xác trên hầu hết các tác vụ. |
| Alignment matrix | Bức tranh trực quan | Các trọng số attention dưới dạng lưới `(T_dec, T_enc)`. Đọc nó để thấy mô hình đã tập trung vào điều gì. |

## Đọc thêm

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — bài báo gốc.
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — ba biến thể điểm số và sự so sánh giữa chúng.
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — những lưu ý về khả năng giải thích.
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — hướng dẫn thực hành với PyTorch.