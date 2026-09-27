# Speculative Decoding và EAGLE

> Việc tạo ra một token từ một LLM tiên phong đòi hỏi một lượt forward pass toàn diện qua hàng tỷ tham số. Lượt forward pass đó bị lãng phí tài nguyên một cách khủng khiếp: phần lớn thời gian, một mô hình nhỏ hơn nhiều có thể đoán đúng 3-5 token tiếp theo, và mô hình lớn chỉ cần *xác minh* dự đoán đó. Khi dự đoán đúng, bạn nhận được 5 token với chi phí của một. Speculative decoding (Leviathan và cộng sự, 2023) đã hiện thực hóa điều này một cách chính xác, và EAGLE-3 (2025) đã đẩy tỷ lệ chấp nhận lên ~4,5 token mỗi lần xác minh — tăng tốc 4-5 lần trong khi vẫn giữ nguyên phân phối đầu ra.

**Type:** Build
**Languages:** Python (với numpy)
**Prerequisites:** Phase 10 Lesson 12 (Tối ưu hóa Inference), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 phút

## Vấn đề

Thông lượng giải mã (decode throughput) cho một mô hình lớp 70B trên H100 thường đạt 40-80 token/giây. Mỗi token yêu cầu một lượt forward pass toàn diện để đọc tất cả trọng số mô hình từ HBM. Bạn không thể làm mô hình nhỏ hơn mà không làm thay đổi đầu ra của nó. Bạn không thể tăng batch size vượt quá giới hạn bộ nhớ. Bạn đang bị bế tắc — trừ khi bạn có thể để mô hình xuất ra nhiều hơn một token mỗi lượt forward pass.

Quá trình tạo autoregressive vốn dĩ mang tính tuần tự: `x_{t+1} = sample(p(· | x_{1:t}))`. Nhưng có một cơ hội cho tính đồng thời. Nếu bạn có một bộ dự đoán giá rẻ nói rằng "4 token tiếp theo có khả năng là [a, b, c, d]", bạn có thể xác minh tất cả 5 vị trí trong một **lượt forward pass duy nhất của mô hình lớn** và chấp nhận tiền tố khớp dài nhất.

Leviathan, Kalai, Matias (2023, "Fast Inference from Transformers via Speculative Decoding") đã hiện thực hóa điều này thông qua một quy tắc chấp nhận/từ chối thông minh giúp bảo toàn phân phối lấy mẫu của mô hình mục tiêu. Cùng một phân phối đầu ra, nhanh hơn 2-4 lần.

## Khái niệm

### Thiết lập hai mô hình

- **Mô hình mục tiêu (Target model)** `M_p`: mô hình lớn, chậm, chất lượng cao mà bạn thực sự muốn lấy mẫu. Phân phối: `p(x)`.
- **Mô hình dự thảo (Draft model)** `M_q`: mô hình nhỏ, nhanh, chất lượng thấp hơn. Phân phối: `q(x)`. Nhỏ hơn 5-30 lần.

Mỗi bước:

1. Mô hình dự thảo đề xuất `K` token theo kiểu autoregressive: `x_1, x_2, ..., x_K ~ q`.
2. Mô hình mục tiêu chạy MỘT lượt forward pass qua tất cả `K+1` vị trí song song, tạo ra `p(x_k)` cho mỗi token được đề xuất.
3. Chấp nhận/từ chối từng token từ trái sang phải thông qua quy tắc lấy mẫu từ chối (rejection-sampling) sửa đổi bên dưới. Chấp nhận tiền tố khớp dài nhất.
4. Nếu bất kỳ token nào bị từ chối, lấy mẫu thay thế từ phân phối đã hiệu chỉnh và dừng lại. Nếu không, lấy mẫu thêm một token thưởng từ `p(· | x_1...x_K)`.

Nếu dự thảo khớp hoàn hảo với mục tiêu, bạn nhận được K+1 token mỗi lượt forward của mục tiêu. Nếu dự thảo sai ở vị trí 1, bạn chỉ nhận được 1 token.

### Quy tắc chính xác (Exactness Rule)

Speculative decoding **có thể chứng minh là tương đương về mặt phân phối với việc lấy mẫu từ p**. Quy tắc từ chối:

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

trong đó `(p - q)+` biểu thị phần dương của sự khác biệt theo từng điểm. Khi dự thảo và mục tiêu đồng ý (`p ≈ q`), khả năng chấp nhận gần bằng 1. Khi chúng không đồng ý, phân phối phần dư (residual) được xây dựng sao cho mẫu tổng thể vẫn chính xác là `p`.

**Trường hợp Greedy.** Đối với lấy mẫu temperature=0, chỉ cần kiểm tra `argmax(p) == x_t`. Nếu có, chấp nhận; nếu không, xuất ra `argmax(p)` và dừng lại.

### Tốc độ tăng dự kiến

Nếu tỷ lệ chấp nhận ở cấp độ token của mô hình dự thảo là `α`, số lượng token dự kiến được tạo ra mỗi lượt forward của mục tiêu là:

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

Tại `α = 0.8, K = 4`: `(1 - 0.8^5)/(1 - 0.8) = 3.36` token mỗi lượt forward. Một lượt forward của mục tiêu tốn khoảng `cost_q * K + cost_p` (K bước dự thảo cộng với một lần xác minh của mục tiêu). Nếu `cost_p >> cost_q * K`, tỷ lệ tăng tốc là `3.36× / 1 = 3.36×` về thông lượng.

Tham số thực tế duy nhất là `α`, phụ thuộc hoàn toàn vào sự căn chỉnh giữa dự thảo và mục tiêu. Một bản dự thảo tốt là tất cả.

### Huấn luyện dự thảo: Chưng cất (Distillation)

Một mô hình nhỏ ngẫu nhiên tạo ra bản dự thảo kém. Công thức tiêu chuẩn là chưng cất từ mục tiêu:

1. Chọn một kiến trúc nhỏ (~1B cho mục tiêu 70B, ~500M cho mục tiêu 7B).
2. Chạy mô hình mục tiêu trên một tập dữ liệu văn bản lớn; lưu trữ các phân phối token tiếp theo của nó.
3. Huấn luyện dự thảo với KL divergence so với phân phối của mục tiêu (không phải so với các token thực tế).

Kết quả: `α` thường là 0.6-0.8 trên code, 0.7-0.85 trên chat ngôn ngữ tự nhiên. Tăng tốc 2-3 lần trong sản xuất.

### EAGLE: Tree Drafting + Tái sử dụng đặc trưng

Li, Wei, Zhang, Zhang (2024, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") đã quan sát thấy hai sự kém hiệu quả trong speculative decoding tiêu chuẩn:

1. Dự thảo thực hiện K bước tuần tự, mỗi bước đều là full-stack. Nhưng dự thảo có thể tái sử dụng các đặc trưng (hidden states) của mục tiêu từ lần xác minh gần nhất — mục tiêu đã tính toán các biểu diễn phong phú mà dự thảo đang phải tự tính lại từ đầu.
2. Dự thảo xuất ra một chuỗi tuyến tính. Nếu dự thảo có thể xuất ra một *cây* các ứng viên (mỗi nút có nhiều dự đoán), một lượt forward pass duy nhất của mục tiêu có thể xác minh nhiều đường dẫn ứng viên song song thông qua tree attention mask, và chọn nhánh khớp dài nhất.

Những thay đổi của EAGLE-1:
- Đầu vào dự thảo = hidden state cuối cùng của mục tiêu tại vị trí t, không phải token thô.
- Kiến trúc dự thảo = 1 lớp transformer decoder (không phải một mô hình nhỏ riêng biệt).
- Đầu ra = cây gồm K = 4-8 ứng viên mỗi độ sâu, độ sâu 4-6.

EAGLE-2 (2024) bổ sung cấu trúc liên kết cây động: cây mở rộng hơn ở nơi dự thảo không chắc chắn và thu hẹp ở nơi nó tự tin. Tăng `α_effective` mà không làm tăng chi phí xác minh.

EAGLE-3 (Li và cộng sự 2025, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test") loại bỏ sự phụ thuộc vào đặc trưng lớp trên cố định và huấn luyện dự thảo với một hàm mất mát "mô phỏng thời gian kiểm thử" mới — dự thảo được huấn luyện trên các đầu ra khớp với phân phối thời gian kiểm thử của mục tiêu thay vì phân phối huấn luyện teacher-forced. Tỷ lệ chấp nhận tăng từ 0.75 (EAGLE-2) lên 0.82 (EAGLE-3), và số token trung bình/xác minh từ 3.0 lên 4.5.

### Xác minh Tree Attention

Khi dự thảo xuất ra một cây, mô hình mục tiêu xác minh nó trong một lượt forward pass duy nhất bằng cách sử dụng **tree attention mask** — một mặt nạ nhân quả (causal mask) mã hóa cấu trúc liên kết cây thay vì một đường thẳng thuần túy. Mỗi token chỉ chú ý (attend) đến tổ tiên của nó trong cây. Lượt xác minh vẫn là một forward, một matmul; mặt nạ cấu trúc chỉ tốn thêm một vài mục KV.

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

Nếu `a, b` là các ứng viên token đầu tiên cạnh tranh và `c, d, e, f` là các ứng viên token thứ hai, tất cả sáu vị trí được xác minh trong một lượt forward pass. Đầu ra là tiền tố dài nhất dọc theo bất kỳ đường dẫn được chấp nhận nào.

### Khi nào nó thắng, khi nào không

**Thắng:**
- Chat / hoàn thành văn bản với văn bản có thể dự đoán (code, tiếng Anh thông dụng, đầu ra có cấu trúc). `α` cao.
- Các thiết lập có tài nguyên GPU không sử dụng trong quá trình giải mã (giai đoạn bị giới hạn bộ nhớ). Tree drafting sử dụng các FLOPs khả dụng.

**Thua / không thắng:**
- Đầu ra có tính ngẫu nhiên cao (viết sáng tạo ở nhiệt độ cao). `α` giảm về `1/|vocab|`.
- Phục vụ theo batch với độ đồng thời rất cao — batching đã lấp đầy các FLOPs, ít chỗ cho xác minh cây.
- Các mô hình mục tiêu rất nhỏ mà dự thảo không nhỏ hơn đáng kể.

Các đơn vị sản xuất thường báo cáo tốc độ tăng 2-3 lần trên chat, 3-5 lần trên tạo code, và gần như bằng không trên viết sáng tạo.

```figure
speculative-decoding
```

## Xây dựng

`code/main.py`:

- Một tham chiếu `speculative_decode(target, draft, prompt, K, temperature)` triển khai quy tắc từ chối chính xác và xác minh rằng nó bảo toàn phân phối của mục tiêu (KL thực nghiệm < 0.01 so với lấy mẫu mục tiêu thông thường).
- Một bộ tạo dự thảo cây kiểu EAGLE xây dựng cây độ sâu K với phân nhánh top-p.
- Một trình xây dựng tree attention mask tạo ra mô hình nhân quả phù hợp cho bộ xác minh.
- Một bộ đo tỷ lệ chấp nhận chạy trên cả hai mô hình LM nhỏ (chưng cất một GPT-2-small từ mục tiêu GPT-2-medium).

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## Sử dụng

- **vLLM** và **SGLang** cung cấp speculative decoding hạng nhất. Các cờ: `--speculative_model`, `--num_speculative_tokens`. Hỗ trợ EAGLE-2/3 thông qua cờ `--spec_decoding_algorithm eagle`.
- **NVIDIA TensorRT-LLM** hỗ trợ cây Medusa và EAGLE nguyên bản.
- **Các mô hình dự thảo tham chiếu**: `Qwen/Qwen3-0.6B-spec` (dự thảo cho Qwen3-32B), `meta-llama/Llama-3.2-1B-Instruct-spec` (dự thảo cho 70B).
- **Medusa heads** (Cai và cộng sự 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"): thay vì một mô hình dự thảo, hãy thêm K đầu dự đoán song song vào chính mục tiêu. Triển khai đơn giản hơn, tỷ lệ chấp nhận thấp hơn một chút so với EAGLE.

## Triển khai

Bài học này tạo ra `outputs/skill-speculative-tuning.md` — một kỹ năng lập hồ sơ khối lượng công việc của mô hình mục tiêu và lựa chọn: mô hình dự thảo, K (độ dài dự thảo), độ rộng cây, nhiệt độ, và khi nào quay lại giải mã thông thường.

## Bài tập

1. Triển khai quy tắc từ chối chính xác và xác minh thực nghiệm. Chạy 10K mẫu thông qua `speculative_decode` và thông qua lấy mẫu mục tiêu thông thường; tính khoảng cách TV giữa hai phân phối đầu ra. Nên < 0.01.

2. Tính công thức tăng tốc. Với `α` và `K` cố định, vẽ biểu đồ số token dự kiến mỗi lượt forward của mục tiêu. Tìm K tối ưu cho α ∈ {0.5, 0.7, 0.9}.

3. Huấn luyện một dự thảo nhỏ. Lấy mục tiêu GPT-2 124M và chưng cất một dự thảo GPT-2 30M trên 100M token với KL loss. Đo `α` trên văn bản giữ lại. Dự kiến: 0.6-0.7.

4. Triển khai dự thảo cây kiểu EAGLE. Thay vì một chuỗi, hãy để dự thảo xuất ra 3 nhánh hàng đầu tại mỗi độ sâu. Xây dựng tree attention mask. Xác minh mục tiêu chấp nhận nhánh đúng dài nhất.

5. Đo lường các chế độ lỗi. Chạy speculative decode ở nhiệt độ=1.5 (tính ngẫu nhiên cao). Cho thấy α sụp đổ và thuật toán chậm hơn giải mã thông thường do chi phí dự thảo.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Target model | "Mô hình lớn" | Mô hình chậm, chất lượng cao mà bạn muốn lấy mẫu (phân phối p) |
| Draft model | "Bộ dự đoán" | Bộ dự đoán nhỏ, nhanh (phân phối q); nhỏ hơn 5-30 lần |
| K / draft length | "Look-ahead" | Số lượng token được dự đoán mỗi lượt xác minh |
| α / acceptance rate | "Tỷ lệ trúng" | Xác suất trên mỗi token rằng đề xuất của dự thảo được chấp nhận |
| Exact rejection rule | "Kiểm tra chấp nhận" | So sánh r < p/q giúp bảo toàn phân phối của mục tiêu |
| Residual distribution | "p-q đã hiệu chỉnh" | (p - q)+ / ||(p - q)+||_1, phân phối để lấy mẫu khi bị từ chối |
| Tree drafting | "Dự đoán phân nhánh" | Dự thảo xuất ra một cây các ứng viên, được xác minh trong một lượt với mặt nạ chú ý cấu trúc cây |
| Tree attention mask | "Mặt nạ cấu trúc" | Mặt nạ nhân quả mã hóa cấu trúc cây để mỗi nút chỉ chú ý đến tổ tiên của nó |
| Medusa heads | "Đầu song song" | K đầu dự đoán phụ trên chính mục tiêu; không cần mô hình dự thảo riêng |
| EAGLE feature reuse | "Dự thảo hidden-state" | Đầu vào dự thảo là hidden state cuối cùng của mục tiêu, không phải token thô, giúp thu nhỏ dự thảo |
| Test-time simulation loss | "Huấn luyện EAGLE-3" | Huấn luyện dự thảo trên các đầu ra khớp với phân phối thời gian kiểm thử của mục tiêu, không phải teacher forcing |

## Đọc thêm

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — quy tắc từ chối chính xác và phân tích tốc độ tăng lý thuyết
- [Chen, Borgeaud, Irving và cộng sự, 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — bài báo về speculative-sampling đồng thời tại DeepMind
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — giải pháp thay thế đầu song song cho mô hình dự thảo
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — tái sử dụng đặc trưng và dự thảo cây
- [Li và cộng sự, 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — cấu trúc liên kết cây động
- [Li và cộng sự, 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — khớp huấn luyện với thời gian kiểm thử
- [Fu, Haotian, Peng và cộng sự, 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/lookahead decoding, một giải pháp thay thế không cần bộ dự đoán