# GPT — Causal Language Modeling

> BERT nhìn thấy cả hai phía. GPT chỉ nhìn thấy quá khứ. Mặt nạ tam giác (triangle mask) là dòng mã quan trọng nhất trong AI hiện đại.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time:** ~75 phút

## Vấn đề

Một mô hình ngôn ngữ trả lời một câu hỏi duy nhất: với `t-1` token đầu tiên, phân phối xác suất trên token `t` là gì? Huấn luyện dựa trên tín hiệu đó — dự đoán token tiếp theo — và bạn sẽ có một mô hình có thể tạo ra văn bản tùy ý từng token một.

Để huấn luyện nó theo kiểu end-to-end trên toàn bộ chuỗi một cách song song, bạn cần dự đoán của mỗi vị trí chỉ phụ thuộc vào các vị trí trước đó. Nếu không, mô hình sẽ gian lận một cách dễ dàng bằng cách nhìn vào đáp án.

Mặt nạ nhân quả (causal mask) thực hiện điều này. Đó là một ma trận tam giác trên duy nhất chứa các giá trị `-inf` được cộng vào các điểm số attention trước khi qua softmax. Sau khi qua softmax, các vị trí đó trở thành 0. Mỗi vị trí chỉ có thể chú ý đến chính nó và các vị trí trước đó. Và vì bạn áp dụng nó một lần cho toàn bộ chuỗi, bạn nhận được N dự đoán token tiếp theo song song trong một lần forward pass.

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023), GPT-5 (2025), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi — tất cả đều là các decoder-only causal transformer với cùng một vòng lặp cốt lõi. Điều phân biệt chúng là chất lượng dữ liệu, quy mô, các cải tiến kiến trúc và quá trình hậu huấn luyện (SFT, RLHF, DPO và các phương pháp kế thừa).

## Khái niệm

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### Mặt nạ (The mask)

Với một chuỗi có độ dài `N`, hãy xây dựng một ma trận `N × N`:

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

Cộng `M` vào các điểm số attention thô trước khi qua softmax. `exp(-inf) = 0`, vì vậy các vị trí bị che sẽ đóng góp trọng số bằng không. Mỗi hàng của ma trận attention là một phân phối xác suất chỉ trên các vị trí trước đó.

Chi phí triển khai: một lệnh `torch.tril()`. Thời gian tính toán: nan giây. Tác động đến lĩnh vực: thay đổi hoàn toàn.

### Nguồn gốc của hình tam giác

Mặt nạ thường được trình bày như một bản vá được gắn thêm vào attention. Hãy thực hiện phép suy luận theo hướng ngược lại và nó sẽ không còn bí ẩn nữa: attention là sự tinh chỉnh thứ ba của trung bình tiền tố (prefix average), và hình tam giác chính là giới hạn vòng lặp của trung bình đó, được viết dưới dạng ma trận.

**Giai đoạn 1 — trung bình tiền tố.** Tóm tắt nhân quả đơn giản nhất của một chuỗi: vị trí `i` trở thành trung bình của các vị trí `0…i`. Dưới dạng vòng lặp, đó là `out[i] = X[:i+1].mean(0)`. Phép tính tương tự là một phép nhân ma trận. Lấy một ma trận tam giác dưới gồm các số một, chia mỗi hàng cho số lượng phần tử của nó, rồi nhân:

```python
import numpy as np

A = np.tril(np.ones((n, n)))
A = A / A.sum(axis=1, keepdims=True)
out = A @ X
```

Hàng `i` của `A` là `[1/(i+1), …, 1/(i+1), 0, …, 0]`. Các số không phía trên đường chéo chính là tính nhân quả. Không có gì về tương lai bị che đi; tương lai chưa bao giờ nằm trong tổng.

**Giai đoạn 2 — trọng số học được.** Một trung bình đồng nhất coi mọi token quá khứ đều có mức độ liên quan như nhau. Thay thế các số một bằng một ma trận điểm số học được `S`. Bây giờ các hàng không còn tổng bằng một theo cấu trúc nữa, vì vậy hãy chuẩn hóa mỗi hàng bằng softmax thay vì chia cho số lượng. Softmax không bao giờ xuất ra chính xác bằng không, điều này phá vỡ tính nhân quả — trừ khi các điểm số tương lai được đưa vào dưới dạng `-inf`, vì `exp(-inf) = 0`:

```python
def softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = S + np.triu(np.full((n, n), -np.inf), k=1)
A = softmax(S, axis=1)
out = A @ X
```

Cùng một hình tam giác, cùng một ma trận row-stochastic, cùng một phép nhân ma trận. Mặt nạ `-inf` không phải là cơ chế mới. Nó chính là các mục nhập bằng không của giai đoạn 1, được chuyển đổi sang miền đầu vào của softmax.

**Giai đoạn 3 — trọng số phụ thuộc vào nội dung.** Trong giai đoạn 2, `S` được cố định sau khi huấn luyện: vị trí 7 luôn coi trọng vị trí 3 như nhau, bất kể token là gì. Hãy để các điểm số phụ thuộc vào chính các token: `S = Q @ K.T / sqrt(d_k)`. Không có gì khác thay đổi. Mặt nạ, softmax, nhân ma trận — giống hệt nhau.

Ba giai đoạn, một bất biến: một ma trận tam giác dưới row-stochastic nhân với chuỗi. Trung bình đồng nhất, trọng số tĩnh học được, trọng số phụ thuộc nội dung. Mặt nạ chưa bao giờ được thêm vào attention. Nó tồn tại từ phép tính trung bình.

```figure
mask-derivation
```

### Huấn luyện song song, suy luận tuần tự

Huấn luyện: thực hiện forward-pass toàn bộ chuỗi `(N, d_model)` một lần, tính N hàm mất mát cross-entropy (một cho mỗi vị trí), cộng tổng, backprop. Song song dọc theo chuỗi. Đây là lý do tại sao việc huấn luyện GPT có thể mở rộng — bạn xử lý 1 triệu token trong một batch chỉ trong một lần chạy GPU.

Suy luận: bạn tạo từng token một. Đưa vào `[t1, t2, t3]`, nhận `t4`. Đưa vào `[t1, t2, t3, t4]`, nhận `t5`. Đưa vào `[t1, t2, t3, t4, t5]`, nhận `t6`. KV cache (Bài 12) lưu các trạng thái ẩn của `t1…tn` để bạn không phải tính toán lại chúng ở mỗi bước. Nhưng độ sâu tuần tự khi suy luận = độ dài đầu ra. Đó là "thuế tự hồi quy" (autoregressive tax) và lý do tại sao giải mã là nút thắt cổ chai về độ trễ của mọi LLM.

### Hàm mất mát — dịch chuyển một đơn vị (shift-by-one)

Với các token `[t1, t2, t3, t4]`:

- Đầu vào: `[t1, t2, t3]`
- Mục tiêu: `[t2, t3, t4]`

Với mỗi vị trí `i`, tính `-log P(target_i | inputs[:i+1])`. Cộng tổng. Đây là cross-entropy cho toàn bộ chuỗi.

Mọi transformer LM mà bạn từng nghe đến đều huấn luyện trên hàm mất mát này. Pre-training, fine-tuning, SFT — cùng một hàm mất mát, dữ liệu khác nhau.

### Các chiến lược giải mã (Decoding strategies)

Sau khi huấn luyện, các lựa chọn lấy mẫu (sampling) quan trọng hơn mọi người nghĩ.

| Phương pháp | Cách hoạt động | Khi nào nên dùng |
|--------|--------------|-------------|
| Greedy | Argmax mỗi bước | Các tác vụ xác định, hoàn thiện mã nguồn |
| Temperature | Chia logit cho T, lấy mẫu | Các tác vụ sáng tạo, T cao hơn = đa dạng hơn |
| Top-k | Lấy mẫu chỉ từ k token hàng đầu | Loại bỏ các đuôi xác suất thấp |
| Top-p (nucleus) | Lấy mẫu từ tập nhỏ nhất có xác suất tích lũy ≥ p | Mặc định từ 2020+; thích ứng với hình dạng phân phối |
| Min-p | Giữ các token có `p > min_p * max_p` | 2024+; loại bỏ đuôi dài tốt hơn top-p |
| Speculative decoding | Mô hình dự thảo đề xuất N token, mô hình lớn xác minh | Giảm độ trễ 2–3 lần với cùng chất lượng |

Vào năm 2026, min-p + temperature 0.7 là mặc định hợp lý cho các mô hình open-weights. Speculative decoding là tiêu chuẩn bắt buộc cho bất kỳ stack suy luận sản xuất nào.

### Điều gì làm nên "công thức GPT" thành công

1. **Decoder-only.** Không có overhead của encoder. Một lần chạy attention + FFN mỗi lớp.
2. **Quy mô (Scaling).** 124M → 1.5B → 175B → hàng nghìn tỷ. Các định luật quy mô Chinchilla (Bài 13) cho bạn biết cách chi tiêu tài nguyên tính toán.
3. **In-context learning.** Xuất hiện ở quy mô 6B–13B. Mô hình có thể làm theo các ví dụ few-shot mà không cần fine-tuning.
4. **RLHF.** Hậu huấn luyện dựa trên sở thích con người đã chuyển đổi văn bản thô được huấn luyện trước thành các trợ lý trò chuyện.
5. **Pre-norm + RoPE + SwiGLU.** Huấn luyện ổn định ở quy mô lớn.

Kiến trúc cốt lõi không thay đổi nhiều kể từ GPT-2. Mọi thứ thú vị đều diễn ra ở dữ liệu, quy mô và hậu huấn luyện.

```figure
causal-mask
```

## Xây dựng

### Bước 1: mặt nạ nhân quả

Xem `code/main.py`. Một dòng mã:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Cộng nó vào các điểm số attention trước khi qua softmax. Đó là toàn bộ cơ chế.

### Bước 2: mô hình GPT 2 lớp

Xếp chồng hai khối decoder (masked self-attention + FFN, không có cross-attention). Thêm một token embedding, một positional encoding và một unembedding (liên kết với ma trận token embedding — một thủ thuật tiêu chuẩn từ GPT-2).

### Bước 3: dự đoán token tiếp theo, end-to-end

Trên từ vựng đồ chơi 20 token, tạo ra các logit tại mỗi vị trí. Tính hàm mất mát cross-entropy so với mục tiêu dịch chuyển một đơn vị. Không cần gradient — đây là bước kiểm tra tính đúng đắn của forward-pass.

### Bước 4: lấy mẫu (sampling)

Triển khai greedy, temperature, top-k, top-p, min-p. Chạy từng cái trên một prompt cố định và so sánh đầu ra. Một hàm lấy mẫu chỉ mất 10 dòng mã.

## Sử dụng

PyTorch, phong cách 2026:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

Bên dưới, `generate()` chạy forward pass, lấy các logit ở vị trí cuối cùng, lấy mẫu token tiếp theo, thêm nó vào và lặp lại. Mọi stack suy luận LLM sản xuất (vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX) đều triển khai cùng một vòng lặp với các tối ưu hóa mạnh mẽ — batched prefill, continuous batching, KV cache paging, speculative decoding.

**GPT vs BERT, mỗi cái một dòng:** GPT dự đoán `P(x_t | x_{<t})`. BERT dự đoán `P(x_masked | x_unmasked)`. Hàm mất mát quyết định liệu mô hình có thể tạo văn bản hay không.

## Triển khai

Xem `outputs/skill-sampling-tuner.md`. Kỹ năng này giúp chọn các tham số lấy mẫu cho một tác vụ tạo văn bản mới và gắn cờ khi nào cần giải mã xác định.

## Bài tập

1. **Dễ.** Chạy `code/main.py` và xác minh ma trận attention nhân quả là tam giác dưới sau khi qua softmax. Kiểm tra nhanh: hàng 3 chỉ nên có trọng số ở các cột 0–3.
2. **Trung bình.** Triển khai beam search với độ rộng 4. So sánh perplexity của beam-4 so với greedy trên 10 prompt ngắn. Liệu beam có luôn thắng không? (Gợi ý: thường là cho dịch thuật, không phải cho trò chuyện mở).
3. **Khó.** Triển khai speculative decoding: sử dụng một mô hình 2 lớp nhỏ làm dự thảo và một mô hình 6 lớp làm trình xác minh. Đo tốc độ thực tế trên 100 lần hoàn thành văn bản với độ dài 64. Xác nhận đầu ra khớp với greedy của trình xác minh.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Causal mask | "Hình tam giác" | Ma trận `-inf` tam giác trên được cộng vào điểm số attention để vị trí `i` chỉ thấy các vị trí `≤ i`. |
| Next-token prediction | "Hàm mất mát" | Cross-entropy của phân phối mô hình so với token tiếp theo thực tế tại mỗi vị trí. |
| Autoregressive | "Tạo từng cái một" | Đưa đầu ra ngược lại làm đầu vào; song song chỉ diễn ra trong khi huấn luyện, không phải khi tạo văn bản. |
| Logits | "Điểm số tiền softmax" | Đầu ra thô của LM head trước khi qua softmax; việc lấy mẫu diễn ra trên các giá trị này. |
| Temperature | "Núm vặn sáng tạo" | Chia logit cho T; T→0 = greedy, T→∞ = đồng nhất. |
| Top-p | "Lấy mẫu hạt nhân" | Cắt tỉa phân phối về tập nhỏ nhất có tổng xác suất ≥ p; lấy mẫu từ những gì còn lại. |
| Min-p | "Tốt hơn top-p" | Giữ các token có `p ≥ min_p × max_p`; thích ứng ngưỡng cắt theo độ sắc nét của phân phối. |
| Speculative decoding | "Dự thảo + xác minh" | Mô hình rẻ tiền đề xuất N token; mô hình lớn xác minh song song. |
| Teacher forcing | "Thủ thuật huấn luyện" | Trong khi huấn luyện, đưa vào token thực tế trước đó, không phải dự đoán của mô hình. Tiêu chuẩn cho mọi seq2seq LM. |

## Đọc thêm

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1.
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 và in-context learning.
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — bài báo về spec decoding.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — mã nguồn tham chiếu chuẩn cho causal-LM.