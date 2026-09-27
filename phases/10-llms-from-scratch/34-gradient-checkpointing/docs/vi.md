# Gradient Checkpointing và Activation Recomputation

> Backprop lưu giữ mọi activation trung gian. Với 70B tham số và 128K context, con số này lên tới 3 TB activation trên mỗi rank. Checkpointing đánh đổi FLOPs lấy bộ nhớ: tính toán lại thay vì lưu trữ. Câu hỏi đặt ra là nên loại bỏ những phân đoạn nào, và câu trả lời không phải là "tất cả".

**Type:** Build
**Languages:** Python (với numpy, tùy chọn torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 phút

## Vấn đề

Việc huấn luyện một transformer yêu cầu lưu trữ, cho mỗi lớp, đầu vào của mọi toán tử được lấy đạo hàm trong quá trình backward: đầu vào của attention, các phép chiếu Q/K/V, đầu ra của softmax, đầu vào của FFN, đầu ra của norm và residual stream. Đối với một lớp có hidden size `d`, độ dài chuỗi `L`, batch `B`, con số này vào khoảng `12 * B * L * d` số thực trên mỗi lớp.

Với `d=8192, L=8192, B=1`, đó là 800 MB/lớp ở định dạng BF16. Một mô hình 64 lớp chiếm 51 GB activation — đó là chưa nhân với microbatch size, chưa cộng thêm các giá trị trung gian của attention-softmax (`L^2` trên mỗi head), và chưa tính đến các bản sao cục bộ của tensor-parallel.

Cái giá phải trả: Trọng số BF16 cộng với trạng thái optimizer có thể vừa trong 80GB, nhưng activation sẽ đẩy bạn vượt quá giới hạn. Gradient checkpointing (hay còn gọi là activation recomputation) là giải pháp tiêu chuẩn. Loại bỏ hầu hết các activation; thực hiện lại forward pass trong quá trình backward để lấy lại chúng. Chi phí: thêm FLOPs. Lợi ích: bộ nhớ giảm theo tỷ lệ giữa các phân đoạn checkpoint và tổng số lớp.

Nếu thực hiện một cách ngây thơ, checkpointing tốn thêm khoảng 33% FLOPs cho forward pass mỗi bước. Nếu thực hiện tốt — checkpointing có chọn lọc theo "smart selection" của Korthikanti và cộng sự — bạn tiết kiệm được 5x bộ nhớ với chi phí FLOPs dưới 5%. Và với FP8 matmuls, FSDP offload, và expert-parallel MoE, điều này thực sự quan trọng: bạn không thể chịu nổi cả việc thiếu bộ nhớ lẫn lãng phí tính toán.

## Khái niệm

### Backward thực sự cần gì

`output = layer(input)`. Backward cần `grad_input` và `grad_params`. Để tính toán chúng, nó cần:

- `input` (để tính `grad_params = input.T @ grad_output` cho các lớp linear)
- một số giá trị trung gian của đạo hàm activation (đạo hàm của ReLU/GELU/softmax phụ thuộc vào giá trị activation)

Forward pass lưu trữ các giá trị này tự động trong đồ thị autograd. Mọi `tensor.retain_grad()` và mọi toán tử cần đầu vào của nó đều giữ lại một tham chiếu.

### Naive Full Checkpointing

Chia mạng thành `N` phân đoạn. Trong quá trình forward, chỉ lưu *đầu vào* của mỗi phân đoạn. Khi backward cần các giá trị trung gian, chạy lại forward pass của phân đoạn đó để tái tạo chúng, sau đó mới lấy đạo hàm.

Ví dụ: Transformer 32 lớp chia thành 32 phân đoạn, mỗi phân đoạn 1 lớp.

- Bộ nhớ: 32 đầu vào lớp (nhỏ) so với 32 * (khối lượng activation mỗi lớp) (khổng lồ).
- Tính toán thêm: 1 forward pass bổ sung cho mỗi phân đoạn, tức là ~33% FLOPs forward tổng cộng (vì backward gấp 2 lần forward, bước đầy đủ trở thành 1 + 1 + 2 = 4 đơn vị thay vì 1 + 2 = 3).

Đây là công thức gốc của Chen và cộng sự năm 2016: một checkpoint sau mỗi `sqrt(L)` lớp để cân bằng bộ nhớ và tính toán. Với L=64, đó là 8 checkpoint.

### Selective Checkpointing (Korthikanti 2022)

Không phải mọi activation đều có chi phí lưu trữ như nhau. Đầu ra của attention softmax là `B*L*L*heads` và tăng *bậc hai* theo độ dài chuỗi. Activation ẩn của FFN là `B*L*4d` và tăng tuyến tính. Đối với các chuỗi dài, softmax chiếm ưu thế.

Selective checkpointing giữ lại các activation rẻ tiền (các phép chiếu tuyến tính, residual) và chỉ tính toán lại các activation đắt đỏ (attention). Bạn trả một lượng FLOPs tối thiểu để tính toán lại nhưng tiết kiệm được bộ nhớ O(L^2).

Megatron-Core triển khai điều này dưới dạng "selective" activation recomputation. Được sử dụng trong hầu hết các đợt huấn luyện quy mô lớn từ năm 2024 trở đi.

### Offload

Giải pháp thay thế cho recompute: chuyển activation sang RAM CPU giữa forward và backward. Yêu cầu băng thông PCIe; có lợi khi băng thông nhàn rỗi vượt quá chi phí tái tạo. Các chiến lược hỗn hợp rất phổ biến: checkpoint một số lớp, offload các lớp khác.

FSDP2 cung cấp offload như một tùy chọn hạng nhất. Offload tỏa sáng khi GPU bị nghẽn bộ nhớ nhưng việc truyền dữ liệu CPU-GPU vẫn còn dư địa.

### Mô hình chi phí Recompute

FLOPs mỗi bước với naive checkpointing sau mỗi `k` lớp trong tổng số `L` lớp:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

Với selective checkpointing, bạn chỉ tính toán lại attention kernel, không phải toàn bộ lớp:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Mô hình tiết kiệm bộ nhớ

Khối lượng activation mỗi lớp: `A`. Với `L` lớp, tổng bộ nhớ activation: `L * A`.

Full checkpoint (kích thước phân đoạn 1): chỉ lưu `L * input_volume` (~`L * 1/10 A` cho một transformer tiêu chuẩn). Tiết kiệm ~`9 * L * A * 1/10`.

Checkpoint mỗi `k` lớp: lưu `L/k * A` cộng với giá trị của `k-1` lớp trong phân đoạn đang hoạt động.

Tại `k = sqrt(L)`, cả bộ nhớ và chi phí tính toán lại đều tỷ lệ với `sqrt(L)` — sự đánh đổi tối ưu cho các lớp có chi phí đồng nhất.

### Khi nào không nên Checkpoint

- Các lớp trong cùng của một pipeline stage đang thực thi. Dù sao chúng cũng phải hoàn thành.
- Lớp đầu tiên và lớp cuối cùng nếu chúng chiếm ưu thế trong tính toán của stage (hiếm gặp trong transformer).
- Các attention kernel đã sử dụng FlashAttention — Flash đã tính toán lại softmax rất nhanh, nên việc checkpoint thêm ở cấp độ lớp không mang lại nhiều lợi ích.

### Các mẫu triển khai

1. **Function wrapper:** bao bọc một phân đoạn trong `torch.utils.checkpoint.checkpoint(fn, input)`. PyTorch chỉ lưu `input`, tính toán lại mọi thứ khác khi backward.

2. **Decorator-based:** gắn nhãn các lớp là có thể checkpoint; trainer quyết định tại thời điểm cấu hình xem phân đoạn nào sẽ được bao bọc.

3. **Manual explicit recompute:** tự viết backward pass, gọi một `recompute_forward` tùy chỉnh để sao chép forward pass với đầu vào đã lưu.

Cả ba đều cho kết quả chức năng giống nhau. Wrapper là cách làm tiêu chuẩn.

### Tương tác với TP / PP / FP8

- **Tensor parallel:** đầu vào checkpoint phải được thu thập (gather) hoặc phân tán lại (rescatter) khi tính toán lại; xử lý chi phí giao tiếp.
- **Pipeline parallel:** mẫu hình điển hình là checkpoint forward pass của mỗi pipeline-stage để các microbatch theo thứ tự ngược lại có thể tái sử dụng bộ nhớ activation.
- **FP8 recompute:** lịch sử amax được cập nhật trong quá trình recompute phải khớp với forward pass gốc, nếu không tỷ lệ FP8 sẽ bị lệch. Hầu hết các framework đều chụp ảnh (snapshot) tỷ lệ này.

```figure
activation-recompute
```

## Build It

### Bước 1: Mô hình đồ chơi với các phân đoạn

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Bước 2: Backward ngây thơ cần tất cả activation

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Bước 3: Bộ nhớ Checkpoint-Every-k

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Bước 4: Mô hình chi phí

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Bước 5: Công cụ ước tính bộ nhớ

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Bước 6: Kích thước phân đoạn tối ưu

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Bước 7: Quyết định Selective Checkpoint

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Sử dụng

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — wrapper tiêu chuẩn trong PyTorch. Bao bọc một hàm; chỉ lưu đầu vào, tính toán lại khi backward.
- **Megatron-Core activation recomputation**: hỗ trợ các chế độ `selective`, `full`, và `block`. Tiêu chuẩn trong huấn luyện quy mô lớn từ 2024+.
- **FSDP2 offload**: `module.to_empty(device="cpu")` với `offload_policy` trong FSDP2 chuyển activation sang CPU thay vì tính toán lại.
- **DeepSpeed ZeRO-Offload**: CPU offload cho trạng thái optimizer và activation, bổ sung cho checkpointing.

## Ship It

Bài học này tạo ra `outputs/prompt-activation-recompute-policy.md` — một prompt nhận cấu hình mô hình của bạn (số lớp, hidden, seq, batch) và bộ nhớ GPU khả dụng, sau đó đưa ra chính sách recompute cho từng lớp (none / selective / full / offload).

## Bài tập

1. Xác minh tính đúng đắn. Chạy `model_forward` + `model_backward` (full activations) so với `model_forward_checkpointed` + `model_backward_checkpointed` (các phân đoạn). Gradient tham số phải giống hệt nhau đến độ chính xác của máy tính.

2. Quét kích thước phân đoạn `k` từ 1 đến `L`. Vẽ biểu đồ chi phí FLOP và bộ nhớ. Tìm điểm uốn của đường cong.

3. Triển khai selective checkpointing: lưu đầu vào của module attention nhưng không lưu các giá trị trung gian của nó. Đo chi phí FLOP so với checkpointing toàn bộ lớp cho mô hình 32 lớp tại seq=8192.

4. Thêm offload. Lưu đầu vào phân đoạn vào một "CPU buffer" mô phỏng (một danh sách riêng). Đo "băng thông PCIe" dưới dạng byte/thời gian và tìm điểm hòa vốn giữa offload và recompute.

5. Benchmark một transformer PyTorch thực tế có và không có `torch.utils.checkpoint`. Đo bộ nhớ (thông qua `torch.cuda.max_memory_allocated`) và thời gian mỗi bước.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|----------------------|
| Gradient checkpointing | "Tiết kiệm bộ nhớ bằng cách làm lại forward" | Chỉ lưu đầu vào phân đoạn; tính toán lại các giá trị trung gian trong backward để hỗ trợ gradient |
| Activation recomputation | "Giống checkpointing" | Tên gọi theo phong cách HPC cho cùng một kỹ thuật |
| Segment size (k) | "Bao nhiêu lớp mỗi checkpoint" | Số lượng lớp mà các giá trị trung gian của chúng bị loại bỏ và tái tạo cùng nhau |
| Selective checkpointing | "Mẹo của Korthikanti" | Chỉ tính toán lại các activation đắt đỏ (attention softmax); giữ lại các activation rẻ |
| Full checkpointing | "Phiên bản ngây thơ" | Tính toán lại giá trị trung gian của mọi lớp trong mọi phân đoạn |
| Block checkpointing | "Độ chi tiết thô" | Checkpoint toàn bộ transformer block; độ chi tiết lớn nhất |
| FLOP overhead | "Thuế tính toán" | FLOPs thêm mỗi bước = (FLOPs recompute) / (FLOPs fwd + bwd); 33% cho naive, 5% cho selective |
| Activation offload | "Chuyển sang CPU" | Di chuyển activation sang RAM CPU giữa forward->backward; thay thế cho recompute |
| Quy tắc sqrt-L | "Tối ưu cổ điển" | Đối với các lớp có chi phí đồng nhất, khoảng cách checkpoint tối ưu là sqrt(L) lớp |
| Attention-softmax volume | "Vấn đề O(L^2)" | L^2 * heads * batch số thực; chiếm ưu thế trong bộ nhớ activation ở context dài |

## Đọc thêm

- [Chen và cộng sự, 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- bài báo gốc chính thức hóa gradient checkpointing
- [Korthikanti và cộng sự, 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- selective activation recomputation và phân tích chi phí chính thức
- [Pudipeddi và cộng sự, 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- phương pháp bộ nhớ hằng số thay thế thông qua reverse-mode rematerialization
- [Ren và cộng sự, 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- activation offload ở quy mô lớn
- [Tài liệu PyTorch torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html) -- API tiêu chuẩn
- [Tài liệu Megatron-Core activation recomputation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- các chế độ selective, full, và block