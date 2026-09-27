# KV Cache, Flash Attention & Tối ưu hóa Inference

> Training là quá trình song song và bị giới hạn bởi FLOP (FLOP-bound). Inference là quá trình tuần tự và bị giới hạn bởi bộ nhớ (memory-bound). Nghẽn cổ chai khác nhau, thủ thuật khác nhau.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~75 phút

## Vấn đề

Một bộ giải mã (decoder) tự hồi quy (autoregressive) ngây thơ thực hiện `O(N²)` công việc để tạo ra `N` token: tại mỗi bước, nó tính toán lại attention trên toàn bộ prefix. Đối với một phản hồi 4K-token, đó là 16 triệu phép tính attention, hầu hết là dư thừa. Mọi hidden state của một token trong prefix là xác định sau khi được tính toán — bạn chỉ cần chạy query của token mới so với các key và value đã được cache của tất cả các token trước đó.

Hơn nữa, bản thân attention di chuyển rất nhiều dữ liệu. Attention tiêu chuẩn tạo ra ma trận điểm N×N, đầu ra softmax N×d, đầu ra cuối cùng N×d — quá nhiều thao tác đọc và ghi vào HBM. Với N≥2K, attention trở nên bị giới hạn bởi bộ nhớ trước khi bị giới hạn bởi FLOP. Các kernel attention cổ điển sử dụng GPU hiện đại kém hiệu quả gấp 4–10 lần.

Hai tối ưu hóa, cả hai đều từ Dao và cộng sự, đã đẩy giới hạn inference từ "chậm" sang "nhanh":

1. **KV cache.** Lưu trữ các vector K và V của mọi token trong prefix. Attention của mỗi token mới là một truy vấn đối với các key đã được cache. Inference giảm từ `O(N²)` xuống `O(N)` cho mỗi bước tạo token.
2. **Flash Attention.** Chia nhỏ (tile) việc tính toán attention để ma trận N×N đầy đủ không bao giờ chạm vào HBM. Tất cả softmax + matmul diễn ra trong SRAM. Tăng tốc độ thực tế 2–4 lần trên A100; 5–10 lần trên H100 với FP8.

Đến năm 2026, cả hai đều là tiêu chuẩn. Mọi stack inference sản xuất (vLLM, TensorRT-LLM, SGLang, llama.cpp) đều mặc định sử dụng chúng. Mọi model tiên tiến đều được xuất xưởng với Flash Attention được bật sẵn.

## Khái niệm

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### Toán học của KV cache

Trên mỗi decoder layer, mỗi token, mỗi head:

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

Đối với một model 7B với 32 layer, 32 head, d_head=128, fp16:

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

Đối với Llama 3 70B (80 layer, d_head=128, GQA với 8 KV head):

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

10 GB đó là lý do tại sao Llama 3 70B ở ngữ cảnh 128K cần phần lớn dung lượng của một GPU A100 40 GB chỉ để chứa KV cache ở batch size 1.

**GQA là chìa khóa cho KV-cache.** MHA với 64 head sẽ tốn 32 GB. MLA nén dữ liệu còn tốt hơn nữa.

Hãy kéo các chiều dữ liệu và quan sát kích thước cache thay đổi. Tăng độ dài chuỗi hoặc batch và xem nó vượt quá khả năng của một GPU nhanh như thế nào:

```figure
kv-cache-sizer
```

### Flash Attention — thủ thuật tiling

Attention tiêu chuẩn:

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

Ba lần truy cập HBM. Trên H100, băng thông HBM là 3 TB/s; SRAM là 30 TB/s. Mỗi lần truy cập HBM là một yếu tố làm chậm gấp 10 lần so với việc giữ mọi thứ trên chip.

Flash Attention:

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

Một lần truy cập HBM cho mỗi tile. Tổng dung lượng bộ nhớ giảm từ `O(N²)` xuống `O(N)`. Backward pass tính toán lại một số giá trị từ forward pass thay vì lưu trữ chúng — một lợi ích khác về bộ nhớ.

**Thủ thuật số học.** Việc chạy softmax duy trì `(max, sum)` trên các tile để quá trình chuẩn hóa cuối cùng là chính xác. Không phải là xấp xỉ — Flash Attention tính toán đầu ra giống hệt từng bit so với attention tiêu chuẩn (ngoại trừ tính không kết hợp của fp16).

**Sự tiến hóa của các phiên bản:**

| Phiên bản | Năm | Thay đổi chính | Tăng tốc trên phần cứng tham chiếu |
|-----------|------|-----------|-------------------------------|
| Flash 1 | 2022 | Tiled SRAM kernel | 2× trên A100 |
| Flash 2 | 2023 | Song song hóa tốt hơn, thứ tự causal-first | 3× trên A100 |
| Flash 3 | 2024 | Hopper asynchrony, FP8 | 1.5–2× trên H100 (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5-stage pipeline, software exp2 | Ưu tiên Inference (chỉ forward) |

Flash 4 chỉ hỗ trợ forward-pass khi ra mắt. Training vẫn sử dụng Flash 3. Hỗ trợ GQA và varlen cho Flash 4 đang được phát triển (giữa năm 2026).

### Speculative decoding — chiến thắng khác về độ trễ

Model nhỏ, chi phí thấp đề xuất N token. Model lớn xác minh tất cả N token song song. Nếu quá trình xác minh chấp nhận k token, bạn đã trả giá cho 1 lần forward pass của model lớn để có k lần tạo token. Thông thường k=3–5 đối với code và văn bản.

Các mặc định năm 2026:
- **EAGLE 2 / Medusa.** Các draft head tích hợp chia sẻ hidden state của bộ xác minh. Tăng tốc 2–3 lần mà không làm giảm chất lượng.
- **Speculative decoding với draft model.** Tăng tốc 2–4 lần trên phần cứng người dùng cuối.
- **Lookahead decoding.** Lặp Jacobi; không cần draft model. Hẹp nhưng miễn phí.

### Continuous batching

Inference theo batch cổ điển: đợi chuỗi chậm nhất kết thúc, sau đó mới bắt đầu batch mới. Lãng phí GPU khi các phản hồi ngắn kết thúc sớm.

Continuous batching (lần đầu xuất hiện trong Orca, hiện có trong vLLM, TensorRT-LLM, SGLang): hoán đổi các yêu cầu mới vào batch ngay khi các yêu cầu cũ kết thúc. Tăng thông lượng 5–10 lần cho các tác vụ chat thông thường.

### PagedAttention — KV cache như bộ nhớ ảo

Tính năng nổi bật của vLLM. KV cache được cấp phát theo các block 16-token; một bảng trang (page table) ánh xạ các vị trí logic tới các block vật lý. Cho phép bạn chia sẻ KV giữa các mẫu song song (beam search, lấy mẫu song song), hoán đổi nóng các prefix cho prompt caching và chống phân mảnh bộ nhớ. Cải thiện thông lượng gấp 4 lần so với cấp phát liên tục ngây thơ.

```figure
flash-attention-memory
```

## Xây dựng

Xem `code/main.py`. Chúng ta triển khai:

1. Một bộ giải mã tăng dần `O(N²)` ngây thơ.
2. Một bộ giải mã `O(N)` có KV-cache.
3. Một softmax dạng tile mô phỏng thuật toán running-max của Flash Attention.

### Bước 1: KV cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

Đơn giản: tiếp tục tăng các vector K, V cho mỗi token trong các danh sách theo layer, theo head.

### Bước 2: tiled softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

Đầu ra giống hệt từng bit với `softmax(qK) V` trong một lần, nhưng tại bất kỳ thời điểm nào, tập làm việc chỉ là một block `tile × d_head`, không phải toàn bộ `N × d_head`.

### Bước 3: so sánh giải mã ngây thơ và giải mã có cache trên 100-token

Đếm các phép tính attention. Ngây thơ: `O(N²)` = 5050. Có cache: `O(N)` = 100. Code sẽ in ra cả hai.

## Sử dụng

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM trong sản xuất:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

Prefix caching giữa các yêu cầu là một chiến thắng lớn năm 2026 — cùng một system prompt, các ví dụ few-shot, hoặc tài liệu ngữ cảnh dài được tái sử dụng KV giữa các lần gọi. Đối với các tác vụ agent với các tool prompt lặp lại, prefix caching thường xuyên mang lại mức tăng thông lượng gấp 5 lần.

## Triển khai

Xem `outputs/skill-inference-optimizer.md`. Kỹ năng này chọn cách triển khai attention, chiến lược KV cache, lượng tử hóa (quantization) và speculative decoding cho một triển khai inference mới.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Xác nhận bộ giải mã ngây thơ và có cache tạo ra cùng một đầu ra; ghi chú sự khác biệt về số lượng phép tính.
2. **Trung bình.** Triển khai prefix caching: với một prompt P và một vài kết quả hoàn thành, chạy một forward pass qua P để điền vào KV cache, sau đó phân nhánh theo từng kết quả hoàn thành. Đo lường mức tăng tốc so với việc mã hóa lại P cho mỗi lần.
3. **Khó.** Triển khai một PagedAttention đơn giản: KV cache trong các block 16-token cố định với một danh sách trống (free-list). Khi một chuỗi kết thúc, trả các block của nó về pool. Mô phỏng 1.000 lần hoàn thành chat với độ dài khác nhau. So sánh sự phân mảnh bộ nhớ với cấp phát liên tục.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|-----------|-----------------|-----------------------|
| KV cache | "Thủ thuật làm cho giải mã nhanh" | Lưu trữ K và V từ mọi token prefix; các truy vấn mới chú ý đến chúng thay vì tính toán lại. |
| HBM | "Bộ nhớ chính của GPU" | High Bandwidth Memory; 80 GB trên H100, 192 GB trên B200. Băng thông ~3 TB/s. |
| SRAM | "Bộ nhớ trên chip" | Bộ nhớ nhanh trên mỗi SM, ~256 KB mỗi SM trên H100. Băng thông ~30 TB/s. |
| Flash Attention | "Kernel attention dạng tile" | Tính toán attention mà không cần tạo ma trận N×N trong HBM. |
| Continuous batching | "Batching không chờ đợi" | Hoán đổi các chuỗi đã hoàn thành ra, chuỗi mới vào mà không cần làm trống batch. |
| PagedAttention | "Tính năng nổi bật của vLLM" | KV cache được cấp phát theo các block cố định với bảng trang; loại bỏ phân mảnh. |
| Prefix caching | "Tái sử dụng prompt dài" | Cache KV cho một prefix dùng chung giữa các yêu cầu; cắt giảm chi phí lớn cho các agent. |
| Speculative decoding | "Dự thảo + xác minh" | Model dự thảo rẻ tiền đề xuất token; model lớn xác minh k token trong một lần pass. |

## Đọc thêm

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5-stage pipeline và thủ thuật software-exp2; đọc README của repo để biết các lưu ý về việc chỉ hỗ trợ forward-only khi ra mắt mà bài học này đề cập.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — bài báo vLLM.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — spec decoding.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — bài báo EAGLE-1/2 cho phương pháp integrated-draft mà bài học trích dẫn.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — phương pháp Medusa được tham chiếu cùng với EAGLE.
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — tài liệu chuyên sâu về thiết kế block 16-token và bảng trang.