# GPU Setup & Cloud

> Huấn luyện trên CPU là ổn cho việc học tập. Huấn luyện cho thực tế cần có GPU.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 phút

## Mục tiêu học tập

- Xác minh khả năng sẵn sàng của GPU cục bộ bằng `nvidia-smi` và CUDA API của PyTorch
- Cấu hình Google Colab với GPU T4 cho các thử nghiệm trên cloud miễn phí
- Benchmark phép nhân ma trận trên CPU so với GPU và đo lường tốc độ tăng tốc
- Ước tính mô hình lớn nhất có thể vừa với VRAM của bạn bằng quy tắc ngón tay cái fp16

## Vấn đề

Hầu hết các bài học trong phase 1-3 đều chạy tốt trên CPU. Nhưng một khi bạn bắt đầu huấn luyện CNN, Transformer hoặc LLM (phase 4 trở đi), bạn cần tăng tốc bằng GPU. Một quá trình huấn luyện mất 8 giờ trên CPU có thể chỉ mất 10 phút trên GPU.

Bạn có ba lựa chọn: GPU cục bộ, GPU trên cloud hoặc Google Colab (miễn phí).

## Khái niệm

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

```figure
s0-gpu-dispatch
```

## Xây dựng

### Lựa chọn 1: NVIDIA GPU cục bộ

Kiểm tra xem bạn có GPU không:

```bash
nvidia-smi
```

Cài đặt PyTorch với CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Lựa chọn 2: Google Colab

1. Truy cập [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. Chạy `!nvidia-smi` để xác minh

Tải các notebook từ khóa học này trực tiếp lên Colab.

### Lựa chọn 3: Cloud GPU

Đối với Lambda Labs, RunPod hoặc Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Không có GPU? Không vấn đề gì.

Hầu hết các bài học đều hoạt động trên CPU. Những bài cần GPU sẽ được ghi chú rõ và bao gồm các liên kết Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Xây dựng: Benchmark GPU vs CPU

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Bài tập

1. Chạy benchmark ở trên và so sánh thời gian giữa CPU và GPU
2. Nếu bạn không có GPU, hãy chạy nó trên Google Colab và so sánh
3. Kiểm tra dung lượng bộ nhớ GPU bạn có và ước tính mô hình lớn nhất bạn có thể chứa (quy tắc ngón tay cái: 2 byte cho mỗi tham số đối với fp16)

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|----------------------|
| CUDA | "Lập trình GPU" | Nền tảng tính toán song song của NVIDIA cho phép bạn chạy code trên GPU |
| VRAM | "Bộ nhớ GPU" | Video RAM trên GPU, tách biệt với RAM hệ thống. Giới hạn kích thước mô hình. |
| fp16 | "Độ chính xác một nửa" | Số thực dấu phẩy động 16-bit, sử dụng một nửa bộ nhớ so với fp32 với độ chính xác giảm thiểu không đáng kể |
| Tensor Core | "Phần cứng ma trận nhanh" | Các nhân GPU chuyên dụng cho phép nhân ma trận, nhanh hơn 4-8 lần so với các nhân thông thường |