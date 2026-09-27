# Speculative Decoding và EAGLE-3

> Giai đoạn 7 · Bài 16 đã chứng minh về mặt toán học: quy tắc bác bỏ Leviathan bảo toàn chính xác phân phối của mô hình kiểm chứng (verifier). Bài học này là góc nhìn từ hệ thống huấn luyện về speculative decoding trong môi trường sản xuất năm 2026. EAGLE-3 đã biến mô hình dự thảo (draft model) từ một phép xấp xỉ giá rẻ thành một mạng nhỏ được xây dựng chuyên biệt, huấn luyện trên chính các hidden state của mô hình kiểm chứng, sau đó bổ sung một vòng lặp kiểm thử trong quá trình huấn luyện (training-time test) giúp căn chỉnh phân phối huấn luyện và suy luận. Kết quả: tăng tốc end-to-end từ 3× đến 6,5×, tỷ lệ chấp nhận mỗi token trên 0,9 đối với chat, không đánh đổi về phân phối. Mọi hệ thống suy luận sản xuất năm 2026 đều mặc định sử dụng nó.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 7 · 16 (toán học speculative decoding), Giai đoạn 10 · 12 (tối ưu hóa suy luận)
**Time:** ~75 phút

## Mục tiêu học tập

- Phát biểu định lý Leviathan trong một câu và chứng minh rằng vòng lặp speculative tạo ra các mẫu có phân phối đồng nhất với mô hình kiểm chứng.
- Đi qua lộ trình phát triển hai năm từ speculative decoding cơ bản (Leviathan 2023) qua EAGLE, EAGLE-2 và EAGLE-3, đồng thời nêu tên chính xác hạn chế mà mỗi bước đã loại bỏ.
- Tính toán tốc độ tăng tốc kỳ vọng từ tỷ lệ chấp nhận `α` và tỷ lệ chi phí giữa mô hình dự thảo và mô hình kiểm chứng `c`, đồng thời chọn độ dài dự thảo tối ưu `N` cho mỗi chế độ.
- Triển khai toàn bộ vòng lặp speculative từ đầu: dự thảo, kiểm chứng, lấy mẫu bác bỏ từ phân phối phần dư (residual), hoàn tác (roll back) KV cache khi bị bác bỏ, phát hành token thưởng khi chấp nhận hoàn toàn.

## Vấn đề

Suy luận tự hồi quy (autoregressive) trên mô hình 70B chạy ở tốc độ khoảng 35 token mỗi giây trên H100. GPU còn lâu mới đạt ngưỡng bão hòa. Băng thông bộ nhớ là giới hạn: mỗi token tải 70B trọng số từ HBM, thực hiện một bước tính toán và tạo ra một số thực. Các đơn vị tính toán hầu như không hoạt động.

Speculative decoding biến điều đó thành một vấn đề về thông lượng mà bạn thực sự có thể giải quyết. Một mô hình dự thảo giá rẻ đề xuất `N` token trong `N` lượt forward pass nhỏ. Mô hình kiểm chứng chạy một lần trên tiền tố cộng với tất cả `N` dự thảo. Nếu phân phối của mô hình kiểm chứng tại vị trí `i` khớp với dự thảo (theo nghĩa thống kê mà chúng ta sẽ làm rõ), chúng ta chấp nhận; nếu không, chúng ta bác bỏ và lấy mẫu một sự điều chỉnh từ phân phối phần dư. Một lượt forward pass của mô hình lớn tạo ra tới `N+1` token được chấp nhận thay vì một.

Định lý quan trọng là Leviathan, Kalman, Matias (ICML 2023): phân phối đầu ra giống hệt với những gì việc lấy mẫu trực tiếp từ mô hình kiểm chứng tạo ra. Không phải xấp xỉ. Mà là giống hệt. Đây là lý do duy nhất khiến speculative decoding được chấp nhận trong sản xuất — nó là một tối ưu hóa độ trễ thuần túy mà không có sự đánh đổi về chất lượng.

Những gì Giai đoạn 7 · Bài 16 đã cung cấp cho bạn là toán học. Những gì bài học này cung cấp là hệ thống huấn luyện. Một dự thảo tốt mang lại tốc độ tăng tốc gấp 2 lần so với một dự thảo giá rẻ. EAGLE, EAGLE-2 và EAGLE-3 (Li và cộng sự, 2024–2025) đã biến "dự thảo = phiên bản nhỏ hơn của cùng một mô hình" thành một kỷ luật kỹ thuật chính xác. Các máy chủ suy luận sản xuất năm 2026 mặc định sử dụng EAGLE-3.

## Khái niệm

### Bất biến: Lấy mẫu bác bỏ Leviathan

Gọi `p(t)` là phân phối của dự thảo cho token tiếp theo với một tiền tố nhất định, và `q(t)` là của mô hình kiểm chứng. Lấy mẫu một token dự thảo `d ~ p`. Chấp nhận với xác suất `min(1, q(d) / p(d))`. Khi bác bỏ, lấy mẫu từ phân phối phần dư `(q - p)_+ / ||(q - p)_+||_1`. Các mẫu thu được tuân theo phân phối `q`. Điều này đúng bất kể `p` tệ đến mức nào — nó càng tệ, bạn càng bác bỏ thường xuyên hơn, nhưng đầu ra vẫn chính xác.

Xếp chồng `N` các lệnh gọi này liên tiếp bằng cách sử dụng một lượt forward pass của mô hình kiểm chứng trên `prefix + d_1 + ... + d_N`. Mô hình kiểm chứng trả về `q_1, q_2, ..., q_{N+1}` đồng thời. Đi từ trái sang phải. Tại lần bác bỏ đầu tiên ở vị trí `j`, lấy mẫu từ `residual(q_j, p_j)` và dừng lại. Khi chấp nhận hoàn toàn, lấy mẫu một token thưởng từ `q_{N+1}`.

### Điều gì quyết định tốc độ tăng tốc

Gọi `α` là tỷ lệ chấp nhận kỳ vọng trên mỗi token dự thảo. Gọi `c = cost(draft) / cost(verifier)` là tỷ lệ chi phí. Số lượng token được chấp nhận kỳ vọng trên mỗi lượt forward pass của mô hình kiểm chứng là:

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

Tổng thời gian thực kỳ vọng trên mỗi token được chấp nhận là `(N * c + 1) / E[accepted]`. Tối thiểu hóa giá trị đó theo `N` và bạn sẽ có điểm tối ưu. Đối với `α = 0.8, c = 0.05`: `N` tối ưu là khoảng 5–7, tốc độ tăng tốc là 3,2×. Đối với `α = 0.95, c = 0.02`: `N` tối ưu là khoảng 8–10, tốc độ tăng tốc đạt 5×.

Đòn bẩy lớn nhất là `α`. Chuyển từ `α = 0.6` (dự thảo cơ bản) sang `α = 0.9` (EAGLE-3) tại `N = 5` cố định sẽ đưa bạn từ 2,2 token được chấp nhận kỳ vọng trên mỗi lượt forward pass của mô hình kiểm chứng lên 4,1. Thông lượng tăng gần gấp 2 lần từ cùng một mô hình kiểm chứng.

### Lộ trình phát triển hai năm

**Vanilla speculative (Leviathan, 2023).** Mô hình dự thảo là một LLM nhỏ hơn được huấn luyện độc lập từ cùng một họ. Dễ dàng kết nối, `α ≈ 0.6`, tốc độ tăng tốc tốt nhất khoảng 2×.

**EAGLE-1 (Li và cộng sự, 2024).** Dự thảo là một transformer nhỏ — thường là một hoặc hai lớp — nhận hidden state lớp cuối của mô hình kiểm chứng làm đầu vào và dự đoán trực tiếp token tiếp theo. Vì dự thảo nhìn thấy biểu diễn đặc trưng của mô hình kiểm chứng, phân phối của nó gần với mô hình kiểm chứng hơn nhiều. `α` tăng lên 0,7–0,8.

**EAGLE-2 (Li và cộng sự, 2024).** Thêm cây dự thảo động: thay vì đề xuất một chuỗi duy nhất gồm `N` token, hãy đề xuất một cây nhỏ các ứng viên, chấm điểm từng ứng viên với mô hình kiểm chứng trong một lượt forward pass (tree attention), và đi theo đường có xác suất cao nhất. Độ dài dự thảo trở nên thích ứng theo từng bước. `α` trên mỗi token của đường dẫn được chấp nhận tăng lên trên 0,85.

**EAGLE-3 (Li và cộng sự, 2025, NeurIPS).** Hai thay đổi nữa. Thứ nhất, loại bỏ hoàn toàn hàm mất mát dự đoán đặc trưng — EAGLE-1/2 huấn luyện dự thảo để khớp với hidden state của mô hình kiểm chứng, điều này giới hạn mức độ hữu ích của dữ liệu. EAGLE-3 huấn luyện trực tiếp trên dự đoán token. Thứ hai, kiểm thử trong quá trình huấn luyện (TTT): trong quá trình huấn luyện dự thảo, đưa các dự đoán trước đó của chính dự thảo trở lại làm đầu vào qua nhiều bước, giống cách nó hoạt động khi suy luận. Điều này căn chỉnh phân phối huấn luyện và kiểm thử, ngăn chặn sự tích lũy lỗi. Tốc độ tăng tốc đo được: lên tới 6,5× trên chat, cải thiện thông lượng 38% ở batch 64 trong SGLang trên H100.

### Hoàn tác KV cache

Việc kiểm chứng mở rộng KV cache của mô hình kiểm chứng thêm `N` mục trong một lượt. Nếu việc bác bỏ xảy ra tại vị trí `j`, nội dung cache sau vị trí `j-1` hiện đã sai. Hai cách triển khai phổ biến: ghi vào bộ đệm tạm và commit khi chấp nhận (vLLM, TensorRT-LLM), hoặc giữ một KV cache vật lý cộng với độ dài logic và cắt bớt khi bác bỏ. Dù bằng cách nào, chi phí hoàn tác là số byte trên mỗi lớp trên mỗi head, không đáng kể so với chi phí forward pass.

Đối với tìm kiếm cây EAGLE-2, mô hình kiểm chứng chạy attention với mặt nạ không nhân quả (non-causal mask) tuân thủ cấu trúc liên kết của cây. Kỹ thuật này khá phức tạp nhưng việc tính toán là một lệnh gọi flash-attention tiêu chuẩn với mặt nạ tùy chỉnh.

### Kiến trúc dự thảo năm 2026

| Chiến lược | Loại dự thảo | `α` | Tốc độ tăng tốc | Chi phí huấn luyện |
|----------|-----------|-----|---------|---------------|
| Vanilla | LLM nhỏ riêng biệt | 0,55-0,70 | 1,8-2,3× | Không (tái sử dụng mô hình nhỏ hiện có) |
| Medusa | Các đầu LM phụ trên mô hình kiểm chứng | 0,65-0,75 | 2-3× | ~1B token SFT |
| EAGLE-1 | Transformer 1 lớp trên hidden state | 0,70-0,80 | 2,5-3× | ~60B token |
| EAGLE-2 | EAGLE-1 + cây dự thảo động | 0,80-0,88 | 3-4× | ~60B token |
| EAGLE-3 | Kết hợp đặc trưng đa lớp + TTT | 0,88-0,92 | 3,5-6,5× | ~60-200B token |
| Lookahead | Không dự thảo (lặp Jacobi) | N/A | 1,3-1,6× | Không |

Trong sản xuất năm 2026: vLLM và SGLang mặc định sử dụng EAGLE-3 khi có sẵn, nếu không thì dùng EAGLE-2. TensorRT-LLM có đường dẫn Medusa nhanh nhất cho các mô hình công khai của Meta và NVIDIA. llama.cpp cung cấp dự thảo vanilla cho các triển khai trên CPU.

```figure
l5-spec-decode-eagle
```

## Xây dựng

Xem `code/main.py`. Đây là vòng lặp speculative Leviathan đầy đủ với tất cả các thành phần: dự thảo N token, lượt kiểm chứng song song, bác bỏ theo vị trí, lấy mẫu phần dư, token thưởng, hoàn tác KV, và kiểm chứng thực nghiệm rằng phân phối đầu ra khớp với việc lấy mẫu trực tiếp từ `q`.

### Bước 1: quy tắc bác bỏ

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### Bước 2: phân phối phần dư

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### Bước 3: một bước speculative đầy đủ

Hàm `spec_step` dự thảo `N` token từ `p`, sau đó kiểm chứng tất cả chúng trong một lượt đánh giá `q` song song. Với mỗi token dự thảo, nó áp dụng quy tắc bác bỏ, và tại lần bác bỏ đầu tiên, nó lấy mẫu sự điều chỉnh từ phần dư. Nếu mọi thứ được chấp nhận, nó phát hành một token thưởng từ `q_{N+1}`.

### Bước 4: quản lý hoàn tác KV cache

Trình mô phỏng theo dõi một `kv_length` logic cho mỗi worker. Khi chấp nhận `k` dự thảo, `kv_length += k`. Khi bác bỏ tại vị trí `j`, cache đã được ghi quá `j`, nhưng độ dài logic được đặt thành `prefix_length + j + 1` — một vị trí sau token điều chỉnh. Các lần đọc tiếp theo sẽ cắt bớt theo độ dài logic.

### Bước 5: kiểm tra Leviathan

Chạy 50.000 bước speculative. Đếm phân phối thực nghiệm của các token được chấp nhận. So sánh với 50.000 mẫu trực tiếp từ `q`. Thống kê chi-square phải thấp hơn nhiều so với giá trị tới hạn. Định lý đúng trong thực tế.

### Bước 6: tốc độ tăng tốc so với α

Quét chất lượng dự thảo bằng cách làm nhiễu `p` khỏi `q` với các biên độ khác nhau. Đo `α`, sau đó vẽ biểu đồ số token kỳ vọng trên mỗi lượt gọi mô hình kiểm chứng như một hàm của `α` và `N`. Mã nguồn in ra một bảng cho thấy chất lượng dự thảo cấp EAGLE-3 (`α ≈ 0.9`) mở khóa 4–5 token trên mỗi lượt gọi mô hình kiểm chứng.

## Sử dụng

`vllm serve` cấp sản xuất với EAGLE-3:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang với EAGLE-3 ở batch 64 trên H100: thông lượng cao hơn khoảng 1,38× so với giải mã vanilla batch-64, theo bài báo EAGLE-3.

Khi nào nên dùng speculative decoding:

- Bất kỳ khối lượng công việc chat tương tác nào mà độ trễ p50 quan trọng hơn thông lượng đỉnh.
- Tạo mã và đầu ra có cấu trúc (JSON, SQL). `α` trên 0,9 vì phân phối mục tiêu rất dễ dự đoán.
- Tạo văn bản dài (hàng ngàn token). Tốc độ tăng tốc khấu hao liên tục mang lại hiệu quả.

Khi nào không nên:

- Các mô hình rất nhỏ (< 3B). Dự thảo không rẻ hơn nhiều so với mô hình kiểm chứng.
- Triển khai CPU batch-1 rất nhỏ. Chi phí bộ nhớ của mô hình dự thảo có thể không đáng giá.
- Lấy mẫu sáng tạo với nhiệt độ rất cao nơi `α` bị sụp đổ.

## Triển khai

Bài học này tạo ra `outputs/skill-eagle3-tuner.md`. Với một khối lượng công việc suy luận (mô hình, kích thước batch, độ trễ mục tiêu, hồ sơ tác vụ), nó đề xuất một chiến lược speculative-decoding và các tham số điều chỉnh (họ dự thảo, `N`, độ sâu cây, chuyển đổi theo nhiệt độ).

## Bài tập

1. Chạy `code/main.py`. Xác nhận thống kê chi-square trên kiểm tra phân phối Leviathan vẫn nằm dưới giá trị tới hạn 95% trên 50.000 mẫu.

2. Quét `N` từ 1 đến 10 với `α` giữ ở mức 0,9 và `c` giữ ở mức 0,04. Vẽ biểu đồ số token kỳ vọng trên mỗi lượt gọi mô hình kiểm chứng và thời gian thực trên mỗi token. Tìm `N` tối thiểu hóa thời gian thực. Giải thích hình dạng của đường cong.

3. Sửa đổi mã để mô phỏng tìm kiếm cây EAGLE-2: tại mỗi bước, dự thảo đề xuất một cây có hình dạng `[2, 2, 2]` (tám đường dẫn ứng viên). Mô hình kiểm chứng chạy một lần, và đường dẫn được chấp nhận có xác suất cao nhất sẽ thắng. Tính `α` trên mỗi lá và tổng số token trên mỗi lượt gọi mô hình kiểm chứng. So sánh với spec-decoding chuỗi tuyến tính ở mức tính toán tương đương.

4. Triển khai trình mô phỏng hoàn tác KV theo batch cho hai chuỗi đồng thời. Chuỗi A có tất cả các dự thảo được chấp nhận; chuỗi B bác bỏ tại vị trí 2. Cho thấy `kv_length` chính xác được cập nhật cho mỗi chuỗi và không có công việc nào bị lãng phí.

5. Đọc Phần 4 của bài báo EAGLE-3 (Kiểm thử trong quá trình huấn luyện). Giải thích trong hai câu tại sao việc huấn luyện dự thảo ngây thơ không có TTT lại bị thiên kiến phơi nhiễm (exposure bias), và tại sao việc đưa các dự đoán của chính dự thảo vào trong quá trình huấn luyện lại khắc phục được điều đó. Kết nối điều này với tài liệu về lấy mẫu theo lịch trình (scheduled sampling) trong seq2seq.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Quy tắc Leviathan | "min(1, q chia p)" | Chấp nhận/bác bỏ Bernoulli với xác suất `min(1, q(d)/p(d))`, bảo toàn chính xác phân phối mô hình kiểm chứng khi bạn lấy mẫu từ phần dư khi bác bỏ |
| Phân phối phần dư | "(q trừ p) cộng, chuẩn hóa" | `(q - p)_+` bị kẹp tại 0 và chuẩn hóa lại — phân phối chính xác để lấy mẫu khi bác bỏ |
| Tỷ lệ chấp nhận α | "dự thảo đúng bao nhiêu lần" | Xác suất thành công Bernoulli kỳ vọng trên mỗi token theo quy tắc bác bỏ; chi phối toàn bộ toán học tăng tốc |
| EAGLE-1 | "dự thảo hidden-state" | Dự thảo transformer nhỏ được điều kiện hóa trên hidden state lớp cuối của mô hình kiểm chứng (Li và cộng sự, 2024) |
| EAGLE-2 | "cây dự thảo động" | EAGLE-1 cộng với một cây các phần tiếp theo ứng viên được chấm điểm bằng tree attention trong một lượt mô hình kiểm chứng |
| EAGLE-3 | "kiểm thử trong quá trình huấn luyện" | Loại bỏ hàm mất mát dự đoán đặc trưng, huấn luyện trên dự đoán token trực tiếp với dự thảo được cung cấp đầu ra của chính nó trong quá trình huấn luyện |
| Kiểm thử trong quá trình huấn luyện (TTT) | "sửa lỗi thiên kiến phơi nhiễm" | Chạy dự thảo tự hồi quy trong quá trình huấn luyện để phân phối đầu vào huấn luyện và kiểm thử khớp nhau — tương tự trực tiếp của lấy mẫu theo lịch trình |
| Hoàn tác KV | "hoàn tác các dự thảo bị bác bỏ" | Quản lý đặt lại KV cache của mô hình kiểm chứng về độ dài tiền tố được chấp nhận sau khi bác bỏ |
| Token thưởng | "token miễn phí" | Khi tất cả `N` dự thảo được chấp nhận, lấy mẫu thêm một token từ `q_{N+1}` mà không tốn thêm chi phí mô hình kiểm chứng |
| Tree attention | "kiểm chứng nhiều ứng viên cùng lúc" | Attention với mặt nạ không nhân quả tuân thủ cấu trúc liên kết của cây dự thảo; tính toán `q_i` cho mọi nút trong cây trong một lượt forward pass |

## Đọc thêm

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — bài báo nền tảng và định lý tương đương
- [Chen và cộng sự — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — giới thiệu độc lập đồng thời với bằng chứng rõ ràng
- [Li và cộng sự — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE-1, dự thảo điều kiện hóa hidden-state
- [Li và cộng sự — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — tìm kiếm cây động
- [Li và cộng sự — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — mặc định sản xuất năm 2026
- [Cai và cộng sự — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — phương pháp thay thế không cần dự thảo
- [Tài liệu vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — tài liệu tham khảo sản xuất chính thống với tất cả các chiến lược được kết nối