# Speculative Decoding — Draft, Verify, Repeat

> Giải mã tự hồi quy (autoregressive decoding) có tính tuần tự. Mỗi token phải chờ token trước đó. Speculative decoding phá vỡ chuỗi này: một mô hình giá rẻ dự thảo N token, mô hình đắt tiền xác thực tất cả N token trong một lần forward pass. Khi dự thảo đúng, bạn chỉ tốn một lần forward lớn cho N thế hệ token.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time:** ~60 phút

## Vấn đề

Một LLM 70B lấy mẫu một token mất khoảng 30 ms trên H100. Một mô hình dự thảo 3B mất khoảng 3 ms. Nếu chúng ta để mô hình 3B dự thảo trước 5 token, sau đó chạy mô hình 70B *một lần* để xác thực cả 5, tổng thời gian là `5×3 + 30 = 45 ms` cho tối đa 5 token được chấp nhận — so với `5×30 = 150 ms` cho việc tạo token theo cách thông thường. Đó là toàn bộ ý tưởng của speculative decoding: đánh đổi một lượng nhỏ bộ nhớ GPU (cho mô hình dự thảo) để giảm độ trễ giải mã xuống 2–4 lần.

Thủ thuật này phải bảo toàn được phân phối xác suất. Speculative sampling, được giới thiệu bởi Leviathan và cộng sự (2023) và Chen và cộng sự cùng thời điểm, đảm bảo rằng chuỗi đầu ra có **phân phối giống hệt** với những gì mô hình lớn sẽ tạo ra nếu chạy độc lập. Không đánh đổi chất lượng. Chỉ nhanh hơn.

Bốn nhóm cặp dự thảo-xác thực thống trị suy luận (inference) năm 2026:

1. **Vanilla speculative (Leviathan 2023).** Mô hình dự thảo riêng biệt (ví dụ: Llama 3 1B) + mô hình xác thực (ví dụ: Llama 3 70B).
2. **Medusa (Cai 2024).** Nhiều đầu giải mã (decoding heads) trên mô hình xác thực dự đoán các vị trí `t+1..t+k` song song. Không cần mô hình dự thảo riêng.
3. **Họ EAGLE (Li 2024, 2025).** Dự thảo nhẹ sử dụng lại các hidden states của mô hình xác thực; tỷ lệ chấp nhận cao hơn vanilla; thường đạt 3–4 lần.
4. **Lookahead decoding (Fu 2024).** Lặp Jacobi; không cần mô hình dự thảo. Tự suy đoán (self-speculation). Phù hợp cho các trường hợp ngách nhưng không có phụ thuộc.

Mọi stack suy luận trong sản xuất năm 2026 đều mặc định hỗ trợ speculative decoding. vLLM, TensorRT-LLM, SGLang và llama.cpp đều hỗ trợ ít nhất vanilla + EAGLE-2.

## Khái niệm

### Thuật toán cốt lõi

Cho mô hình xác thực `M_q` và mô hình dự thảo rẻ hơn `M_p`:

1. Gọi `x_1..x_k` là tiền tố đã được giải mã.
2. **Dự thảo**: sử dụng `M_p` để tự hồi quy đề xuất `d_{k+1}, d_{k+2}, ..., d_{k+N}` với xác suất dự thảo `p_1..p_N`.
3. **Xác thực song song**: chạy `M_q` một lần trên `x_1..x_k, d_{k+1}, ..., d_{k+N}`, nhận xác suất xác thực `q_1..q_{N+1}` cho các vị trí `k+1..k+N+1`.
4. **Chấp nhận/từ chối từng token dự thảo từ trái sang phải**: với mỗi `i`, chấp nhận với xác suất `min(1, q_i(d_i) / p_i(d_i))`.
5. Tại lần từ chối đầu tiên ở vị trí `j`: lấy mẫu `t_j` từ phân phối "phần dư" `(q_j - p_j)_+` đã chuẩn hóa. Tất cả các dự thảo sau `j` đều bị loại bỏ.
6. Nếu chấp nhận tất cả `N`: lấy mẫu thêm một token `t_{N+1}` từ `q_{N+1}` (token thưởng miễn phí).

Thủ thuật phân phối phần dư là hiểu biết toán học giúp giữ cho đầu ra có phân phối chính xác như thể `M_q` đã lấy mẫu từ đầu.

### Điều gì quyết định tốc độ

Gọi `α` = tỷ lệ chấp nhận kỳ vọng trên mỗi token dự thảo. Gọi `c` = tỷ lệ chi phí giữa dự thảo và xác thực. Mỗi bước:

- Tạo thông thường thực hiện 1 lần gọi mô hình lớn cho mỗi token.
- Speculative thực hiện 1 lần gọi mô hình lớn cho mỗi `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` token khi `α` cao.

Quy tắc ngón tay cái điển hình tại `α = 0.75` và `N = 5`: giảm 3 lần số lần gọi mô hình lớn. Chi phí dự thảo rẻ hơn 5 lần. Tổng thời gian thực tế giảm khoảng 2.5 lần.

**α phụ thuộc vào:**

- Mức độ mô hình dự thảo xấp xỉ mô hình xác thực. Cùng họ / cùng dữ liệu huấn luyện sẽ tăng đáng kể α.
- Chiến lược giải mã. Dự thảo greedy với xác thực greedy: α cao. Lấy mẫu nhiệt độ (temperature sampling): khó khớp hơn; tỷ lệ chấp nhận giảm.
- Loại tác vụ. Code và đầu ra có cấu trúc dễ chấp nhận hơn (dễ dự đoán); viết sáng tạo tự do khó chấp nhận hơn.

### Medusa — dự thảo không cần mô hình dự thảo

Medusa thay thế mô hình dự thảo bằng các đầu ra bổ sung trên mô hình xác thực. Tại vị trí `t`:

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

Mỗi đầu ra đưa ra logits riêng. Khi suy luận, bạn lấy mẫu từ mỗi đầu, tạo ra một chuỗi ứng viên, sau đó xác thực bằng một lần forward pass sử dụng sơ đồ tree-attention xem xét tất cả các phần tiếp theo cùng một lúc.

Ưu điểm: không cần mô hình thứ hai. Nhược điểm: thêm tham số cần huấn luyện; cần giai đoạn tinh chỉnh (SFT) (~1B token); tỷ lệ chấp nhận thấp hơn một chút so với vanilla speculative với dự thảo tốt.

### EAGLE — dự thảo tốt hơn bằng cách sử dụng lại hidden states

EAGLE-1/2/3 (Li và cộng sự, 2024–2025) biến mô hình dự thảo thành một Transformer tí hon (thường là 1 lớp) nhận đầu vào là các hidden states lớp cuối của mô hình xác thực. Vì dự thảo nhìn thấy biểu diễn đặc trưng của mô hình xác thực, các dự đoán của nó tương quan mạnh với phân phối đầu ra của mô hình xác thực. Tỷ lệ chấp nhận tăng từ ~0.6 (vanilla) lên 0.85+.

EAGLE-3 (2025) đã thêm tìm kiếm cây (tree search) trên các ứng viên tiếp theo. vLLM và SGLang cung cấp EAGLE-2/3 như đường dẫn spec mặc định cho Llama 3/4 và Qwen 3.

### Vũ điệu KV cache

Xác thực nạp `N` token dự thảo vào mô hình xác thực trong một lần forward pass. Điều này mở rộng KV cache của mô hình xác thực thêm `N` mục. Nếu một số dự thảo bị từ chối, bạn phải hoàn tác (roll back) cache về độ dài tiền tố đã được chấp nhận.

Các triển khai sản xuất (vLLM's `--speculative-model`, TensorRT-LLM's LookaheadDecoder) xử lý việc này với các bộ đệm KV tạm thời. Ghi trước, cam kết khi được chấp nhận. Về mặt khái niệm không khó, nhưng khá phức tạp khi triển khai.

```figure
draft-verify-tokens
```

## Xây dựng

Xem `code/main.py`. Chúng ta triển khai thuật toán speculative-sampling cốt lõi (bước từ chối + phân phối phần dư) với:

- Một "mô hình lớn" là deterministic-softmax trên một phân phối được mã hóa thủ công (để chúng ta có thể xác minh toán học về sự chấp nhận).
- Một "mô hình dự thảo" là một sự nhiễu loạn của mô hình lớn.
- Một vòng lặp chấp nhận/từ chối tạo ra cùng phân phối biên như lấy mẫu trực tiếp.

### Bước 1: bước từ chối

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` là một số ngẫu nhiên đồng nhất. `q_prob` là xác suất của mô hình xác thực cho token dự thảo. `p_prob` là xác suất của mô hình dự thảo. Định lý Leviathan cho rằng quyết định Bernoulli này, theo sau là lấy mẫu từ phần dư khi bị từ chối, bảo toàn chính xác phân phối của mô hình xác thực.

### Bước 2: phân phối phần dư

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

Trừ `p` từ `q` theo từng phần tử, kẹp các giá trị âm về 0, chuẩn hóa lại. Lấy mẫu từ đây khi có bất kỳ sự từ chối nào.

### Bước 3: một bước speculative

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

Năm token được chấp nhận → một token thưởng → sáu token được tạo ra trong một lần pass của mô hình xác thực.

### Bước 4: đo tỷ lệ chấp nhận

Chạy 10,000 bước speculative ở các mức chất lượng dự thảo khác nhau. Vẽ biểu đồ tỷ lệ chấp nhận so với KL divergence giữa phân phối dự thảo và xác thực. Bạn sẽ thấy một mối quan hệ đơn điệu rõ ràng.

### Bước 5: xác minh sự tương đương phân phối

Thực nghiệm: biểu đồ các token được tạo ra bởi vòng lặp speculative phải khớp với biểu đồ được tạo ra bằng cách lấy mẫu trực tiếp từ mô hình xác thực. Đây là định lý Leviathan trong thực tế. Kiểm định chi-bình phương xác nhận điều này trong sai số lấy mẫu.

## Sử dụng

Sản xuất:

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM có đường dẫn Medusa nhanh nhất tính đến giữa năm 2026. `faster-whisper` bao bọc speculative decoding cho Whisper-large với một dự thảo nhỏ.

**Chọn dự thảo:**

| Chiến lược | Khi nào nên chọn | Tăng tốc |
|----------|--------------|---------|
| Vanilla draft (họ Llama 1B/3B) | Nguyên mẫu nhanh, không cần huấn luyện | 1.8–2.3× |
| Medusa heads | Bạn có thể tinh chỉnh mô hình xác thực | 2–3× |
| EAGLE-2 / 3 | Sản xuất, tốc độ tối đa | 3–4× |
| Lookahead | Không dự thảo, không huấn luyện, không thêm tham số | 1.3–1.6× |

**Khi nào KHÔNG nên dùng spec-decode:**

- Tạo chuỗi đơn lẻ từ 1–5 token. Chi phí overhead chiếm ưu thế.
- Lấy mẫu cực kỳ sáng tạo / nhiệt độ cao (α giảm).
- Triển khai bị hạn chế bộ nhớ (mô hình dự thảo thêm VRAM).

## Triển khai

Xem `outputs/skill-spec-decode-picker.md`. Kỹ năng này chọn một chiến lược speculative decoding (vanilla / Medusa / EAGLE / lookahead) và các tham số điều chỉnh (N, nhiệt độ dự thảo) cho một workload suy luận mới.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Xác nhận phân phối token speculative khớp với phân phối lấy mẫu trực tiếp của mô hình xác thực trên 50,000 token với chi-bình phương p > 0.05.
2. **Trung bình.** Vẽ biểu đồ tăng tốc (token trên mỗi lần forward của mô hình lớn) như một hàm của `N` cho `α = 0.5, 0.7, 0.85`. Xác định `N` tối ưu cho mỗi α. (Gợi ý: số token kỳ vọng trên mỗi lần gọi xác thực = `(1 - α^{N+1}) / (1 - α)`.)
3. **Khó.** Triển khai một Medusa tí hon: lấy GPT từ Bài 14, thêm 3 đầu LM bổ sung dự đoán các vị trí t+2, t+3, t+4. Huấn luyện trên tinyshakespeare với hàm mất mát đa đầu (multi-head loss). So sánh tỷ lệ chấp nhận với một dự thảo vanilla được tạo bằng cách cắt bớt cùng mô hình đó.
4. **Khó.** Triển khai rollback: bắt đầu với KV cache tiền tố 10 token, nạp 5 token dự thảo, mô phỏng một sự từ chối tại vị trí 3. Xác minh rằng các lần đọc cache của bạn khớp chính xác với "tiền tố + 2 dự thảo được chấp nhận đầu tiên" ở lần lặp tiếp theo.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Draft model | "Mô hình rẻ" | Mô hình nhỏ hơn đề xuất các token ứng viên; thường rẻ hơn 10–50 lần so với mô hình xác thực. |
| Verifier | "Mô hình lớn" | Mô hình mục tiêu mà chúng ta bảo toàn phân phối; chạy một lần mỗi bước speculative. |
| Acceptance rate (α) | "Dự thảo đúng bao nhiêu lần" | Xác suất trên mỗi token mà mô hình xác thực chấp nhận dự thảo. Thường là 0.7–0.9. |
| Residual distribution | "Dự phòng khi từ chối" | `(q - p)_+` đã chuẩn hóa; lấy mẫu từ đây khi từ chối sẽ bảo toàn phân phối của mô hình xác thực. |
| Bonus token | "Token miễn phí" | Khi tất cả N dự thảo được chấp nhận, lấy mẫu thêm một token từ phân phối bước tiếp theo của mô hình xác thực. |
| Medusa | "Speculative không dự thảo" | Nhiều đầu LM trên mô hình xác thực dự đoán các vị trí t+1..t+k song song. |
| EAGLE | "Dự thảo dựa trên hidden-state" | Dự thảo Transformer tí hon được điều kiện hóa trên các hidden states lớp cuối của mô hình xác thực. |
| Lookahead decoding | "Lặp Jacobi" | Tự suy đoán sử dụng lặp điểm cố định; không cần mô hình dự thảo. |
| Tree attention | "Xác thực nhiều ứng viên cùng lúc" | Xác thực phân nhánh xem xét đồng thời nhiều phần tiếp theo của dự thảo. |
| KV rollback | "Hoàn tác dự thảo bị từ chối" | Bộ đệm KV tạm thời; cam kết khi chấp nhận, loại bỏ khi từ chối. |

## Đọc thêm

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — thuật toán cốt lõi và định lý tương đương.
- [Chen và cộng sự (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — giới thiệu đồng thời; chứng minh từ chối Bernoulli sạch.
- [Cai và cộng sự (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — bài báo Medusa; xác thực tree-attention.
- [Li và cộng sự (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1; dự thảo điều kiện hóa hidden-state.
- [Li và cộng sự (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2; cây dự thảo động.
- [Li và cộng sự (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3.
- [Fu và cộng sự (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — cách tiếp cận lookahead, không dự thảo.
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — tài liệu tham khảo sản xuất chính thống với cả bốn chiến lược.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — mã nguồn tham khảo cho EAGLE-1/2/3.