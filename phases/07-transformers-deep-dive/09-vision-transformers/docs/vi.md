# Vision Transformers (ViT)

> Một bức ảnh là một lưới các patch. Một câu là một lưới các token. Cùng một kiến trúc Transformer xử lý cả hai.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers intro)
**Time:** ~45 minutes

## The Problem

Trước năm 2020, thị giác máy tính (computer vision) đồng nghĩa với các mạng tích chập (convolutions). Mọi mô hình SOTA trên các tập dữ liệu ImageNet, COCO và các benchmark phát hiện đối tượng đều sử dụng CNN làm backbone. Transformer khi đó chỉ dành cho ngôn ngữ.

Dosovitskiy và cộng sự (2020) — "An Image is Worth 16x16 Words" — đã chứng minh rằng bạn có thể loại bỏ hoàn toàn các lớp tích chập. Cắt một bức ảnh thành các patch có kích thước cố định, chiếu tuyến tính (linearly project) mỗi patch thành một embedding, và đưa chuỗi đó vào một Transformer encoder tiêu chuẩn. Ở quy mô đủ lớn (pretraining trên ImageNet-21k hoặc lớn hơn), ViT đạt kết quả ngang bằng hoặc vượt trội so với các mô hình dựa trên ResNet.

ViT là khởi đầu cho một mô hình phổ biến hơn vào năm 2026: một kiến trúc, nhiều phương thức (modalities). Whisper token hóa âm thanh. ViT token hóa hình ảnh. Các action token cho robot. Pixel token cho video. Transformer không quan tâm — chỉ cần đưa cho nó một chuỗi và nó sẽ tự học.

Đến năm 2026, ViT và các biến thể của nó (DeiT, Swin, DINOv2, ViT-22B, SAM 3) thống trị hầu hết các tác vụ thị giác. CNN vẫn thắng thế trên các thiết bị edge và các tác vụ nhạy cảm với độ trễ. Mọi thứ khác đều có ViT nằm đâu đó trong stack.

## The Concept

![Image → patches → tokens → transformer](../assets/vit.svg)

### Step 1 — patchify

Chia một bức ảnh `H × W × C` thành một chuỗi `N × (P·P·C)` các patch phẳng. Cấu hình điển hình: ảnh `224 × 224`, `16 × 16` patch → 196 patch, mỗi patch có 768 giá trị.

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Kích thước patch là yếu tố điều chỉnh. Patch nhỏ hơn = nhiều token hơn, độ phân giải tốt hơn, chi phí attention tăng theo bình phương. Patch lớn hơn = thô hơn, chi phí thấp hơn.

### Step 2 — linear embedding

Một ma trận học được duy nhất sẽ chiếu mỗi patch phẳng thành `d_model`. Tương đương với một phép tích chập có kernel size `P` và stride `P`. Trong PyTorch, đây thực chất là `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` — một cách triển khai chỉ với 2 dòng code.

### Step 3 — thêm token `[CLS]`, thêm positional embeddings

- Thêm vào đầu một token `[CLS]` có thể học được. Trạng thái ẩn cuối cùng của nó chính là biểu diễn hình ảnh được sử dụng cho phân loại.
- Thêm positional embeddings có thể học được (ViT gốc) hoặc sinusoidal 2D (các biến thể sau này).
- Từ năm 2024 trở đi, RoPE được mở rộng sang 2D cho vị trí, đôi khi không cần các embedding tường minh.

### Step 4 — standard transformer encoder

Xếp chồng L khối `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`. Giống hệt BERT. Không có lớp nào dành riêng cho thị giác. Đây chính là điểm nhấn sư phạm của bài báo.

### Step 5 — head

Đối với phân loại: lấy trạng thái ẩn `[CLS]` → linear → softmax. Đối với DINOv2 hoặc SAM, loại bỏ `[CLS]`, sử dụng trực tiếp các patch embedding.

### Các biến thể quan trọng

| Model | Năm | Thay đổi |
|-------|------|--------|
| ViT | 2020 | Bản gốc. Kích thước patch cố định, full global attention. |
| DeiT | 2021 | Distillation; có thể huấn luyện chỉ trên ImageNet-1k. |
| Swin | 2021 | Phân cấp với các cửa sổ dịch chuyển (shifted windows). Chi phí dưới bình phương cố định. |
| DINOv2 | 2023 | Tự giám sát (không nhãn). Các đặc trưng thị giác tổng quát tốt nhất. |
| ViT-22B | 2023 | 22 tỷ tham số; áp dụng các quy luật mở rộng (scaling laws). |
| SigLIP | 2023 | ViT + cặp ngôn ngữ, hàm mất mát sigmoid contrastive. |
| SAM 3 | 2025 | Segment anything; ViT-Large + bộ giải mã mask có thể prompt. |

### Tại sao mất nhiều thời gian như vậy

ViT cần *rất nhiều* dữ liệu để bắt kịp CNN vì nó không có các inductive bias của CNN (tính bất biến tịnh tiến, tính cục bộ). Nếu không có hơn 100 triệu ảnh được gán nhãn hoặc pretraining tự giám sát mạnh mẽ, CNN vẫn thắng thế ở cùng mức tính toán. DeiT đã giải quyết vấn đề này vào năm 2021 bằng các kỹ thuật distillation; DINOv2 đã giải quyết triệt để vào năm 2023 bằng tự giám sát.

```figure
n5-patch-stream
```

## Build It

Xem `code/main.py`. Patchify bằng stdlib thuần + linear embedding + kiểm tra tính hợp lệ. Không huấn luyện — ViT ở bất kỳ quy mô thực tế nào cũng cần PyTorch và hàng giờ chạy GPU.

### Step 1: fake image

Một bức ảnh RGB 24 × 24 dưới dạng danh sách các hàng gồm các tuple `(R, G, B)`. Chúng ta sử dụng patch 6×6 → 16 patch, mỗi patch là một vector embedding 108-d.

### Step 2: patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Thứ tự raster: duyệt theo hàng trên lưới. Mọi ViT đều sử dụng thứ tự này.

### Step 3: linear embed

Nhân mỗi patch phẳng với một ma trận ngẫu nhiên `(patch_flat_size, d_model)`. Xác minh shape đầu ra là `(N_patches + 1, d_model)` sau khi thêm `[CLS]`.

### Step 4: đếm tham số cho một ViT thực tế

In số lượng tham số cho ViT-Base: 12 lớp, 12 head, d=768, patch=16. So sánh với ResNet-50 (~25M). ViT-Base đạt khoảng ~86M. ViT-Large ~307M. ViT-Huge ~632M.

## Use It

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 embeddings là mặc định cho các đặc trưng hình ảnh vào năm 2026.** Đóng băng backbone, huấn luyện một head nhỏ. Hoạt động tốt cho phân loại, truy xuất, phát hiện, chú thích ảnh. Các checkpoint DINOv2 của Meta vượt trội hơn CLIP trên mọi tác vụ thị giác không liên quan đến văn bản.

**Chọn kích thước patch.** Các mô hình nhỏ sử dụng 16×16 (ViT-B/16). Dự đoán dày đặc (segmentation) sử dụng 8×8 hoặc 14×14 (SAM, DINOv2). Các mô hình rất lớn sử dụng 14×14.

## Ship It

Xem `outputs/skill-vit-configurator.md`. Kỹ năng này giúp chọn biến thể ViT và kích thước patch cho một tác vụ thị giác mới dựa trên kích thước tập dữ liệu, độ phân giải và ngân sách tính toán.

## Exercises

1. **Dễ.** Chạy `code/main.py`. Xác minh số lượng patch bằng `(H/P) * (W/P)` và chiều của patch phẳng bằng `P*P*C`.
2. **Trung bình.** Triển khai positional embeddings hình sin 2D — hai mã hình sin độc lập cho `row` và `col` của mỗi patch, sau đó nối chúng lại. Đưa chúng vào một ViT nhỏ bằng PyTorch và so sánh độ chính xác với positional embeddings có thể học được trên CIFAR-10.
3. **Khó.** Xây dựng một ViT 3 lớp (PyTorch), huấn luyện trên 1.000 ảnh MNIST với patch 4×4. Đo độ chính xác trên tập test. Bây giờ thêm pretraining DINOv2 trên cùng 1.000 ảnh đó (đơn giản hóa: chỉ cần huấn luyện encoder để dự đoán patch embedding từ các patch bị che). Độ chính xác có cải thiện không?

## Key Terms

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Patch | "Token của vision-transformer" | Vector phẳng của các giá trị pixel cho một vùng `P × P × C` của ảnh. |
| Patchify | "Cắt + làm phẳng" | Cắt ảnh thành các patch không chồng lấp, làm phẳng mỗi patch thành một vector. |
| Token `[CLS]` | "Tóm tắt hình ảnh" | Token có thể học được được thêm vào đầu; embedding cuối cùng của nó là biểu diễn của ảnh. |
| Inductive bias | "Giả định của mô hình" | ViT có ít tiền định hơn CNN; cần nhiều dữ liệu hơn để bù đắp khoảng cách. |
| DINOv2 | "ViT tự giám sát" | Được huấn luyện không cần nhãn bằng cách tăng cường ảnh + momentum teacher. Đặc trưng ảnh tổng quát tốt nhất năm 2026. |
| SigLIP | "Người kế nhiệm CLIP" | ViT + text encoder được huấn luyện với hàm mất mát sigmoid contrastive; tốt hơn CLIP ở cùng mức tính toán. |
| Swin | "ViT cửa sổ" | ViT phân cấp với attention cục bộ + cửa sổ dịch chuyển; chi phí dưới bình phương. |
| Register tokens | "Thủ thuật 2023" | Một vài token có thể học được bổ sung giúp hấp thụ các attention sink; cải thiện đặc trưng DINOv2. |

## Further Reading

- [Dosovitskiy và cộng sự (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — bài báo gốc về ViT.
- [Touvron và cộng sự (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT.
- [Liu và cộng sự (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin.
- [Oquab và cộng sự (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2.
- [Darcet và cộng sự (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — bản sửa lỗi register-token cho DINOv2.