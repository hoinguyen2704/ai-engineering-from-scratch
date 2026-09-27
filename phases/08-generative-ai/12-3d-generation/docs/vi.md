# 3D Generation

> 3D là phương thức mà khả năng tận dụng từ 2D sang 3D đạt hiệu quả mạnh mẽ nhất. Bước đột phá năm 2023 là 3D Gaussian Splatting. Làn sóng tạo sinh giai đoạn 2024-2026 kết hợp multi-view diffusion + tái dựng 3D (3D reconstruction) để tạo ra các vật thể và cảnh quan từ một prompt hoặc một bức ảnh duy nhất.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 4 (Vision), Phase 8 · 07 (Latent Diffusion)
**Time:** ~45 phút

## Vấn đề

Nội dung 3D rất phức tạp:

- **Biểu diễn (Representation).** Meshes, point clouds, voxel grids, signed distance fields (SDFs), neural radiance fields (NeRFs), 3D Gaussians. Mỗi loại đều có những ưu nhược điểm riêng.
- **Sự khan hiếm dữ liệu.** ImageNet có 14 triệu ảnh. Tập dữ liệu 3D sạch lớn nhất (Objaverse-XL, 2023) có khoảng 10 triệu vật thể, nhưng phần lớn có chất lượng thấp.
- **Bộ nhớ.** Một lưới voxel 512³ chứa 128 triệu voxel; một NeRF cảnh quan hữu dụng cần 1 triệu mẫu/tia. Việc tạo sinh (generation) khó hơn nhiều so với tái dựng (reconstruction).
- **Giám sát (Supervision).** Với ảnh 2D, bạn có các pixel. Với 3D, bạn thường chỉ có một vài góc nhìn 2D và phải nâng cấp (lift) lên 3D.

Stack công nghệ năm 2026 tách biệt hai vấn đề này. Đầu tiên, tạo *ảnh đa góc nhìn 2D (multi-view images)* bằng mô hình diffusion. Thứ hai, khớp một *biểu diễn 3D* (thường là Gaussian splatting) vào các hình ảnh đó.

## Khái niệm

![3D generation: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### Biểu diễn: 3D Gaussian Splatting (Kerbl et al., 2023)

Biểu diễn một cảnh quan dưới dạng một đám mây gồm khoảng 1 triệu 3D Gaussians. Mỗi Gaussian có 59 tham số: vị trí (3), hiệp phương sai (6, hoặc quaternion 4 + tỷ lệ 3), độ mờ (1), màu sắc spherical-harmonics (48 ở bậc 3, 3 ở bậc 0).

Kết xuất (Rendering) = phép chiếu + alpha-compositing. Tốc độ nhanh (~100 fps ở 1080p trên card 4090). Có khả năng vi phân (differentiable). Khớp bằng gradient descent so với ảnh thực tế (ground-truth). Một cảnh quan có thể được khớp trong 5-30 phút trên GPU phổ thông.

Hai cải tiến trong giai đoạn 2023-2024:
- **Generative Gaussian splats.** Các mô hình như LGM, LRM, InstantMesh dự đoán trực tiếp đám mây Gaussian từ một hoặc vài hình ảnh.
- **4D Gaussian Splatting.** Các Gaussian với độ lệch theo từng khung hình cho các cảnh động.

### Multi-view diffusion

Fine-tune một mô hình diffusion ảnh đã được huấn luyện trước để tạo ra nhiều góc nhìn nhất quán của cùng một vật thể từ một prompt văn bản hoặc một ảnh duy nhất. Zero123 (Liu et al., 2023), MVDream (Shi et al., 2023), SV3D (Stability, 2024), CAT3D (Google, 2024). Thường xuất ra 4-16 góc nhìn xung quanh vật thể, sau đó được nâng cấp lên 3D thông qua Gaussian splatting hoặc NeRF.

### Các pipeline Text-to-3D

| Mô hình | Đầu vào | Đầu ra | Thời gian |
|-------|-------|--------|------|
| DreamFusion (2022) | văn bản | NeRF qua SDS | ~1 giờ/tài sản |
| Magic3D | văn bản | mesh + texture | ~40 phút |
| Shap-E (OpenAI, 2023) | văn bản | 3D ẩn (implicit) | ~1 phút |
| SJC / ProlificDreamer | văn bản | NeRF / mesh | ~30 phút |
| LRM (Meta, 2023) | ảnh | triplane | ~5 giây |
| InstantMesh (2024) | ảnh | mesh | ~10 giây |
| SV3D (Stability, 2024) | ảnh | góc nhìn mới | ~2 phút |
| CAT3D (Google, 2024) | 1-64 ảnh | 3D NeRF | ~1 phút |
| TripoSR (2024) | ảnh | mesh | ~1 giây |
| Meshy 4 (2025) | văn bản + ảnh | PBR mesh | ~30 giây |
| Rodin Gen-1.5 (2025) | văn bản + ảnh | PBR mesh | ~60 giây |
| Tencent Hunyuan3D 2.0 (2025) | ảnh | mesh | ~30 giây |

Hướng đi 2025-2026: các mô hình text-to-mesh trực tiếp với vật liệu PBR phù hợp cho các game engine. Bước trung gian multi-view diffusion vẫn là công thức hiệu quả nhất cho các vật thể tổng quát.

### NeRF (để tham khảo)

Neural Radiance Field (Mildenhall et al., 2020). Một MLP nhỏ nhận `(x, y, z, view direction)` và xuất ra `(color, density)`. Kết xuất bằng cách tích phân dọc theo các tia. Vượt trội hơn so với tổng hợp góc nhìn mới dựa trên mesh về chất lượng nhưng tốc độ kết xuất chậm hơn 100-1000 lần. Đã bị thay thế bởi Gaussian splatting trong hầu hết các ứng dụng thời gian thực nhưng vẫn chiếm ưu thế trong nghiên cứu.

```figure
v4-3d-multiview
```

## Xây dựng

`code/main.py` triển khai một mô hình "Gaussian splatting" 2D đơn giản: biểu diễn một ảnh mục tiêu tổng hợp (một gradient mượt mà) dưới dạng tổng của các 2D Gaussian splats. Tối ưu hóa vị trí, màu sắc và hiệp phương sai bằng gradient descent để khớp với mục tiêu. Bạn sẽ thấy hai thao tác cốt lõi: kết xuất thuận (splat + alpha-composite) và khớp bằng gradient descent.

### Bước 1: 2D Gaussian splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Bước 2: kết xuất bằng cách cộng các splat

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

3D Gaussian splatting thực tế sắp xếp các Gaussian theo độ sâu và thực hiện alpha-composite theo thứ tự. Mô hình 2D đơn giản của chúng ta chỉ thực hiện phép cộng.

### Bước 3: khớp bằng gradient descent

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Các cạm bẫy

- **Thiếu nhất quán về góc nhìn.** Nếu bạn tạo 4 góc nhìn độc lập và chúng không nhất quán về cấu trúc vật thể, kết quả 3D sẽ bị mờ. Cách khắc phục: multi-view diffusion với cơ chế attention chia sẻ.
- **Ảo giác mặt sau.** Từ một ảnh đơn lẻ → 3D, mô hình phải tự "sáng tạo" phần bị che khuất. Chất lượng thay đổi rất lớn.
- **Bùng nổ Gaussian splat.** Việc huấn luyện không kiểm soát sẽ dẫn đến 10 triệu splat và gây overfitting. Các heuristic về làm dày (densification) + cắt tỉa (pruning) (từ bài báo gốc 3D-GS) là rất cần thiết.
- **Vấn đề cấu trúc (Topology).** Các mesh từ trường ẩn (SDFs) thường có lỗ hổng hoặc tự giao cắt. Hãy chạy một công cụ remesher (ví dụ: voxel remesh của Blender) trước khi xuất bản.
- **Bản quyền dữ liệu huấn luyện.** Objaverse có các giấy phép hỗn hợp; việc sử dụng thương mại tùy thuộc vào từng mô hình.

## Sử dụng

| Tác vụ | Lựa chọn 2026 |
|------|-----------|
| Tái dựng cảnh từ ảnh | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| Text-to-3D cho game | Meshy 4 hoặc Rodin Gen-1.5 (đầu ra PBR) |
| Image-to-3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| Tổng hợp góc nhìn mới từ ít ảnh | CAT3D, SV3D |
| Tái dựng cảnh động | 4D Gaussian Splatting |
| Avatar / người mặc quần áo | Gaussian Avatar, HUGS |
| Nghiên cứu / SOTA | Bất cứ thứ gì mới ra mắt tuần trước |

Để đưa 3D vào sản xuất trong pipeline game hoặc thương mại điện tử: Meshy 4 hoặc Rodin Gen-1.5 xuất ra các mesh PBR có thể đưa thẳng vào Unity / Unreal.

## Xuất bản

Lưu `outputs/skill-3d-pipeline.md`. Kỹ năng này bao gồm việc nhận một yêu cầu 3D (đầu vào: văn bản / một ảnh / vài ảnh; đầu ra: mesh / splat / NeRF; mục đích: render / game / VR) và xuất ra: pipeline (multi-view diffusion + khớp, hoặc mô hình mesh trực tiếp), mô hình cơ sở, ngân sách lặp, xử lý hậu kỳ cấu trúc, các kênh vật liệu cần thiết.

## Bài tập

1. **Dễ.** Chạy `code/main.py` với 4, 16, 64 Gaussians. Báo cáo MSE cuối cùng so với mục tiêu.
2. **Trung bình.** Mở rộng sang các Gaussian có màu (RGB). Xác nhận việc tái dựng khớp với mẫu màu mục tiêu.
3. **Khó.** Sử dụng gsplat hoặc Nerfstudio, tái dựng một vật thể thực từ 50 bức ảnh chụp. Báo cáo thời gian khớp và SSIM cuối cùng trên các góc nhìn giữ lại (held-out views).

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | "3DGS" | Cảnh quan dưới dạng đám mây 3D Gaussians; kết xuất alpha-composite vi phân. |
| NeRF | "Neural radiance field" | MLP xuất ra màu sắc + mật độ tại một điểm 3D; kết xuất bằng tích phân tia. |
| Triplane | "Three 2-D planes" | Phân tách 3D thành ba lưới đặc trưng 2D dọc theo trục; rẻ hơn so với thể tích. |
| SDS | "Score distillation sampling" | Huấn luyện mô hình 3D bằng cách sử dụng score của 2D-diffusion làm pseudo-gradient. |
| Multi-view diffusion | "Many views at once" | Mô hình diffusion xuất ra một batch các góc nhìn camera nhất quán. |
| PBR | "Physically-based rendering" | Vật liệu với các kênh albedo, roughness, metallic, normal. |
| Densification | "Grow splats" | Heuristic huấn luyện 3DGS: tách / nhân bản các splat ở vùng có gradient cao. |

## Ghi chú sản xuất: 3D chưa có nền tảng chung

Không giống như ảnh (latent diffusion + DiT) và video (spatiotemporal DiT), 3D chưa có một runtime thống trị duy nhất vào năm 2026. Cây quyết định sản xuất phụ thuộc vào biểu diễn:

- **NeRF / triplane.** Suy luận là ray-marching + một lần forward MLP cho mỗi mẫu. Một lần render 512² yêu cầu hàng triệu lần forward MLP. Hãy batch các mẫu tia một cách quyết liệt; SDPA/xformers có thể áp dụng.
- **Multi-view diffusion + LRM reconstruction.** Pipeline hai giai đoạn. Giai đoạn 1 (multi-view DiT) là một server diffusion giống như Bài 07. Giai đoạn 2 (LRM transformer) là một lần forward one-shot qua các góc nhìn. Hồ sơ độ trễ tổng thể là "diffusion + one-shot" — hãy chọn các primitive phục vụ (serving primitives) theo từng giai đoạn.
- **SDS / DreamFusion.** Tối ưu hóa theo từng tài sản, không phải suy luận. Hãy xây dựng các job, không phải trình xử lý request.

Đối với hầu hết các sản phẩm năm 2026, câu trả lời đúng là "chạy mô hình multi-view diffusion theo yêu cầu, tái dựng sang 3DGS không đồng bộ, phục vụ 3DGS để xem thời gian thực". Điều này tách biệt khối lượng công việc giữa server suy luận GPU (nhanh) và trình tối ưu hóa ngoại tuyến (chậm).

## Đọc thêm

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — multi-view diffusion.
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D.