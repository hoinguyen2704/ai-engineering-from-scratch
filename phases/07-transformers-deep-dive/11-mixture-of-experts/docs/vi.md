# Mixture of Experts (MoE)

> Một Transformer dày đặc (dense) 70B kích hoạt mọi tham số cho mỗi token. Một mô hình MoE 671B chỉ kích hoạt 37B tham số mỗi token nhưng vượt trội hơn ở mọi benchmark. Sparsity (độ thưa) là ý tưởng mở rộng quan trọng nhất của thập kỷ này.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 phút

## Vấn đề

Số lượng FLOPs khi inference của một Transformer dày đặc bằng với số lượng tham số của nó (nhân 2 cho forward pass). Khi mở rộng một mô hình dày đặc, mỗi token đều phải "trả phí" cho toàn bộ tham số. Đến năm 2024, các mô hình tiên phong đã chạm ngưỡng giới hạn tính toán: để thông minh hơn một cách đáng kể, bạn cần số lượng FLOPs trên mỗi token tăng theo cấp số nhân.

Mixture of Experts phá vỡ mối liên kết này. Thay thế mỗi FFN bằng `E` chuyên gia (expert) độc lập + một bộ định tuyến (router) chọn `k` chuyên gia cho mỗi token. Tổng tham số = `E × FFN_size`. Tham số hoạt động mỗi token = `k × FFN_size`. Cấu hình điển hình năm 2026: `E=256`, `k=8`. Lưu trữ mở rộng theo `E`, tính toán mở rộng theo `k`.

Các mô hình tiên phong năm 2026 gần như hoàn toàn là MoE: DeepSeek-V3 (tổng 671B / 37B hoạt động), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss. Trên bảng xếp hạng độc lập của Artificial Analysis, top 10 mô hình mã nguồn mở đều là MoE.

## Khái niệm

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### Thay thế FFN

Khối Transformer dày đặc:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

Khối MoE:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

Mỗi chuyên gia là một FFN độc lập (thường là SwiGLU). Bộ định tuyến là một lớp tuyến tính đơn lẻ. Mỗi token chọn `k` chuyên gia của riêng nó và nhận kết quả là hỗn hợp có trọng số (gated mixture) từ đầu ra của các chuyên gia đó.

### Vấn đề cân bằng tải (load-balancing)

Nếu bộ định tuyến đưa 90% token qua chuyên gia số 3, các chuyên gia khác sẽ bị "bỏ đói". Ba giải pháp đã được thử nghiệm:

1. **Auxiliary load-balancing loss** (Switch Transformer, Mixtral). Thêm một hình phạt tỉ lệ thuận với phương sai trong việc sử dụng chuyên gia. Cách này hiệu quả nhưng thêm một siêu tham số và một tín hiệu gradient thứ hai.
2. **Expert capacity + token dropping** (Switch đời đầu). Mỗi chuyên gia xử lý tối đa `C × N/E` token; các token dư thừa sẽ bỏ qua lớp này. Điều này làm giảm chất lượng.
3. **Auxiliary-loss-free balancing** (DeepSeek-V3). Thêm một bias học được cho mỗi chuyên gia để điều chỉnh lựa chọn top-k của bộ định tuyến. Bias được cập nhật bên ngoài hàm loss huấn luyện. Không gây ảnh hưởng đến mục tiêu chính. Đây là bước đột phá lớn của năm 2024.

Cách tiếp cận của DeepSeek-V3: sau mỗi bước huấn luyện, với mỗi chuyên gia, kiểm tra xem mức độ sử dụng của nó cao hay thấp hơn mục tiêu. Điều chỉnh bias bằng `±γ`. Việc lựa chọn sử dụng `scores + bias`. Xác suất chuyên gia dùng để gating là `scores` thô không thay đổi. Tách biệt việc định tuyến khỏi biểu thức tính toán.

### Chuyên gia chia sẻ (Shared experts)

DeepSeek-V2/V3 cũng chia các chuyên gia thành *shared* (chia sẻ) và *routed* (được định tuyến). Mọi token đều đi qua tất cả các chuyên gia chia sẻ. Các chuyên gia được định tuyến sẽ được chọn thông qua top-k. Chuyên gia chia sẻ nắm bắt kiến thức chung; chuyên gia được định tuyến chuyên biệt hóa. V3 chạy 1 chuyên gia chia sẻ cộng với top-8 trong số 256 chuyên gia được định tuyến.

### Chuyên gia hạt mịn (Fine-grained experts)

MoE cổ điển (GShard, Switch): mỗi chuyên gia có độ rộng bằng một FFN đầy đủ. `E` nhỏ (8–64), `k` nhỏ (1–2).

MoE hạt mịn hiện đại (DeepSeek-V3, Qwen-MoE): mỗi chuyên gia hẹp hơn (1/8 kích thước FFN). `E` lớn (256+), `k` lớn hơn (8+). Tổng tham số không đổi, nhưng các tổ hợp mở rộng nhanh hơn nhiều. `C(256, 8) = 400 trillion` "chuyên gia" có thể được chọn mỗi token. Chất lượng tăng lên, độ trễ giữ nguyên.

### Hồ sơ chi phí

Trên mỗi token, mỗi lớp:

| Cấu hình | Tham số hoạt động / token | Tổng tham số |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dày đặc) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 đánh bại Llama 3 70B (dày đặc) ở hầu hết các benchmark trong khi thực hiện **ít FLOPs hoạt động hơn trên mỗi token**. Nhiều tham số hơn = nhiều kiến thức hơn. Nhiều FLOPs hoạt động hơn = nhiều tính toán hơn trên mỗi token. MoE tách biệt hai yếu tố này.

### Điểm hạn chế: bộ nhớ

Tất cả các chuyên gia đều nằm trên GPU bất kể chuyên gia nào được kích hoạt. Một mô hình 671B cần khoảng 1.3 TB VRAM cho trọng số fp16. Việc triển khai MoE tiên phong đòi hỏi expert parallelism (song song hóa chuyên gia) — phân mảnh các chuyên gia trên các GPU, định tuyến token qua mạng. Độ trễ bị chi phối bởi giao tiếp all-to-all, không phải matmul.

```figure
expert-routing
```

## Xây dựng

Xem `code/main.py`. Một lớp MoE nhỏ gọn trong stdlib thuần túy với:

- `n_experts=8` chuyên gia kiểu SwiGLU (mỗi chuyên gia một lớp tuyến tính, để minh họa)
- Định tuyến top-k=2
- Trọng số gating chuẩn hóa bằng softmax
- Cân bằng không cần auxiliary-loss thông qua bias cho mỗi chuyên gia

### Bước 1: bộ định tuyến

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias ảnh hưởng đến việc lựa chọn, không phải trọng số gate. Đó là thủ thuật của DeepSeek-V3 — bias sửa lỗi mất cân bằng tải mà không làm chệch hướng dự đoán của mô hình.

### Bước 2: chạy 100 token qua bộ định tuyến

Theo dõi tần suất kích hoạt của các chuyên gia. Nếu không có bias, việc sử dụng sẽ bị lệch. Với vòng lặp cập nhật bias (`-γ` cho các chuyên gia bị sử dụng quá mức, `+γ` cho các chuyên gia bị sử dụng ít), mức độ sử dụng sẽ hội tụ về phân phối đồng nhất sau vài lần lặp.

### Bước 3: so sánh số lượng tham số

In ra "tương đương dày đặc" của một cấu hình MoE. Hình dạng DeepSeek-V3: 256 routed + 1 shared, 8 active, d_model=7168. Tổng số tham số rất lớn. Số lượng tham số hoạt động chỉ bằng 1/7 so với Llama 3 70B dày đặc.

## Sử dụng

Tải bằng HuggingFace:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

Inference sản xuất năm 2026: vLLM hỗ trợ định tuyến MoE nguyên bản. SGLang có đường dẫn expert-parallel nhanh nhất. Cả hai đều tự động xử lý lựa chọn top-k và expert parallelism.

**Khi nào chọn MoE:**
- Bạn muốn chất lượng tiên phong với chi phí inference thấp hơn trên mỗi token.
- Bạn có VRAM / hạ tầng expert-parallel.
- Khối lượng công việc của bạn nặng về token (chat, code) thay vì nặng về ngữ cảnh (tài liệu dài).

**Khi nào KHÔNG chọn MoE:**
- Triển khai trên thiết bị biên (edge) — bạn phải trả phí lưu trữ đầy đủ cho bất kỳ FLOPs hoạt động nào.
- Phục vụ người dùng đơn lẻ yêu cầu độ trễ thấp — định tuyến chuyên gia làm tăng overhead.
- Các mô hình nhỏ (<7B) — lợi thế chất lượng của MoE chỉ xuất hiện trên ngưỡng tính toán nhất định (~6B tham số hoạt động).

## Triển khai

Xem `outputs/skill-moe-configurator.md`. Kỹ năng này chọn E, k, và bố cục chuyên gia chia sẻ cho một MoE mới dựa trên ngân sách tham số, token huấn luyện và mục tiêu triển khai.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Quan sát cách cập nhật bias không cần auxiliary-loss làm đều mức độ sử dụng chuyên gia sau 50 lần lặp.
2. **Trung bình.** Thay thế bộ định tuyến đã học bằng bộ định tuyến dựa trên hash (xác định, không học). So sánh chất lượng và sự cân bằng. Tại sao bộ định tuyến đã học lại tốt hơn?
3. **Khó.** Triển khai "rollout-matched routing" kiểu GRPO (thủ thuật DeepSeek-V3.2): ghi lại các chuyên gia được kích hoạt trong quá trình inference, ép buộc định tuyến tương tự trong quá trình tính toán gradient. Đo lường hiệu quả trên một thiết lập policy-gradient đơn giản.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Expert | "Một FFN trong số nhiều FFN" | Một mạng feed-forward độc lập; các tham số dành riêng cho một lát cắt thưa của tính toán FFN. |
| Router | "Cổng" | Một lớp tuyến tính nhỏ chấm điểm mỗi token với mỗi chuyên gia; lựa chọn top-k. |
| Top-k routing | "k chuyên gia hoạt động mỗi token" | Tính toán FFN của mỗi token đi qua đúng k chuyên gia, được trọng số bởi gate. |
| Auxiliary loss | "Hình phạt cân bằng tải" | Thành phần loss bổ sung phạt việc sử dụng chuyên gia bị lệch. |
| Auxiliary-loss-free | "Thủ thuật của DeepSeek-V3" | Cân bằng thông qua bias cho mỗi chuyên gia chỉ trên lựa chọn của router; không có gradient bổ sung. |
| Shared expert | "Luôn bật" | Chuyên gia bổ sung mà mọi token đều đi qua; nắm bắt kiến thức chung. |
| Expert parallelism | "Phân mảnh theo chuyên gia" | Phân phối các chuyên gia khác nhau cho các GPU khác nhau; định tuyến token qua mạng. |
| Sparsity | "Tham số hoạt động < tổng tham số" | Tỉ lệ `k × expert_size / (E × expert_size)`; 37/671 ≈ 5.5% cho DeepSeek-V3. |

## Đọc thêm

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — ý tưởng gốc.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch, MoE cổ điển.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + MoE không cần auxiliary-loss + MTP.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — bài báo về cân bằng dựa trên bias.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — phân tách hạt mịn + chuyên gia chia sẻ mà bài học này sử dụng.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — bài báo gốc về chuyên gia chia sẻ.