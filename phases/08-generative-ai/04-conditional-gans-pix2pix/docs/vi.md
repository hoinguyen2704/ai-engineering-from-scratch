# Conditional GANs & Pix2Pix

> Bước đột phá lớn đầu tiên trong giai đoạn 2014-2017 là khả năng kiểm soát những gì một GAN tạo ra. Gắn kèm một nhãn, hoặc một hình ảnh, hoặc một câu. Pix2Pix đã thực hiện phiên bản hình ảnh và nó vẫn vượt trội hơn mọi mô hình text-to-image tổng quát trong các tác vụ image-to-image hẹp.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**Time:** ~75 phút

## Vấn đề

Một GAN không điều kiện (unconditional GAN) lấy mẫu các khuôn mặt tùy ý. Hữu ích cho bản demo, nhưng vô dụng trong sản xuất. Bạn muốn: *chuyển bản phác thảo thành ảnh chụp*, *chuyển bản đồ thành ảnh chụp từ trên không*, *chuyển cảnh ban ngày thành ban đêm*, *tô màu ảnh đen trắng*. Trong tất cả các trường hợp này, bạn được cung cấp một hình ảnh đầu vào `x` và phải xuất ra `y` với sự tương ứng về ngữ nghĩa. Có rất nhiều `y` hợp lý cho mỗi `x`. Sai số bình phương trung bình (Mean-squared error) làm phẳng chúng thành một mớ hỗn độn. Hàm mất mát đối nghịch (adversarial loss) thì không, vì "trông giống thật" là một tiêu chí sắc nét.

Conditional GAN (Mirza & Osindero, 2014) thêm một điều kiện `c` làm đầu vào cho cả `G` và `D`. Pix2Pix (Isola et al., 2017) đã chuyên biệt hóa điều này: điều kiện là một hình ảnh đầu vào đầy đủ, generator là một U-Net, discriminator là một bộ phân loại *dựa trên patch* (PatchGAN), và hàm mất mát là đối nghịch + L1. Công thức đó vượt trội hơn các mô hình text-to-image xây dựng từ đầu trên các miền image-to-image hẹp ngay cả vào năm 2026 vì nó được huấn luyện trên *dữ liệu có cặp* — bạn có chính xác tín hiệu mình cần.

## Khái niệm

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**Conditional G.** `G(x, z) → y`. Trong Pix2Pix, `z` là dropout bên trong G (không có nhiễu đầu vào — Isola nhận thấy nhiễu tường minh thường bị bỏ qua).

**Conditional D.** `D(x, y) → [0, 1]`. Đầu vào là *cặp* (điều kiện, đầu ra). Đây là sự khác biệt chính: D phải đánh giá xem `y` có nhất quán với `x` hay không, chứ không chỉ là liệu `y` có trông giống thật hay không.

**U-Net generator.** Encoder-decoder với các skip connection bắc qua nút thắt cổ chai. Rất quan trọng cho các tác vụ mà đầu vào và đầu ra chia sẻ cấu trúc cấp thấp (cạnh, hình dáng). Nếu không có các skip connection, chi tiết tần số cao sẽ biến mất.

**PatchGAN discriminator.** Thay vì xuất ra một điểm số thực/giả duy nhất, D xuất ra một lưới `N×N` nơi mỗi ô đánh giá một trường tiếp nhận (receptive field) khoảng 70×70 pixel. Sau đó lấy trung bình. Đây là giả định trường ngẫu nhiên Markov: tính chân thực mang tính cục bộ. Huấn luyện nhanh hơn nhiều, ít tham số hơn, đầu ra sắc nét hơn.

**Loss.**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

Thành phần L1 ổn định quá trình huấn luyện và đẩy G về phía mục tiêu đã biết. L1 tạo ra các cạnh sắc nét hơn L2 (trung vị, không phải trung bình). `λ = 100` là mặc định của Pix2Pix.

## CycleGAN — khi bạn không có các cặp dữ liệu

Pix2Pix cần dữ liệu `(x, y)` có cặp. CycleGAN (Zhu et al., 2017) loại bỏ yêu cầu này với cái giá là một hàm mất mát bổ sung: hàm mất mát *cycle consistency*. Hai generator `G: X → Y` và `F: Y → X`. Huấn luyện chúng sao cho `F(G(x)) ≈ x` và `G(F(y)) ≈ y`. Điều này cho phép bạn chuyển đổi ngựa thành ngựa vằn, mùa hè thành mùa đông, mà không cần các ví dụ có cặp.

Vào năm 2026, image-to-image không cặp chủ yếu được thực hiện thông qua diffusion (ControlNet, IP-Adapter) thay vì CycleGAN, nhưng ý tưởng cycle-consistency vẫn tồn tại trong hầu hết các bài báo về thích nghi miền không cặp (unpaired domain adaptation).

```figure
gx-patchgan
```

## Xây dựng

`code/main.py` triển khai một conditional GAN nhỏ trên dữ liệu 1-D. Điều kiện `c` là một nhãn lớp (0 hoặc 1). Tác vụ: tạo ra một mẫu từ phân phối có điều kiện cho lớp đã cho.

### Bước 1: thêm điều kiện vào cả đầu vào của G và D

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot encoding là cách đơn giản nhất. Các mô hình lớn hơn sử dụng các embedding đã học, FiLM modulation, hoặc cross-attention.

### Bước 2: huấn luyện có điều kiện

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

Generator phải khớp với phân phối thực *cho điều kiện đã cho*, không phải phân phối biên.

### Bước 3: xác minh đầu ra theo từng lớp

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## Các cạm bẫy

- **Điều kiện bị bỏ qua.** G học cách biên hóa (marginalize), D không bao giờ phạt vì tín hiệu điều kiện yếu. Khắc phục: điều kiện hóa D mạnh mẽ hơn (lớp sớm, không chỉ lớp muộn), sử dụng projection discriminator (Miyato & Koyama 2018).
- **Trọng số L1 quá thấp.** G trôi về các đầu ra trông có vẻ thật tùy ý, không trung thực với đầu vào. Bắt đầu với λ≈100 cho các tác vụ kiểu Pix2Pix.
- **Trọng số L1 quá cao.** G tạo ra các đầu ra mờ vì L1 vẫn là một chuẩn L_p. Giảm dần khi quá trình huấn luyện ổn định.
- **Rò rỉ ground-truth trong D.** Nối `(x, y)` làm đầu vào D, không chỉ `y`. Nếu không có điều này, D không thể kiểm tra tính nhất quán.
- **Sụp đổ chế độ (Mode collapse) theo lớp.** Mỗi lớp có thể sụp đổ độc lập. Chạy các kiểm tra tính đa dạng theo điều kiện lớp.

## Sử dụng

Trạng thái của các tác vụ image-to-image năm 2026:

| Tác vụ | Cách tiếp cận tốt nhất |
|------|---------------|
| Phác thảo → ảnh, cùng miền, dữ liệu có cặp | Pix2Pix / Pix2PixHD (vẫn nhanh, vẫn sắc nét) |
| Phác thảo → ảnh, không cặp | ControlNet với mô hình điều kiện Scribble |
| Phân đoạn ngữ nghĩa → ảnh | SPADE / GauGAN2 hoặc SD + ControlNet-Seg |
| Chuyển đổi phong cách | Diffusion với IP-Adapter hoặc LoRA; các phương pháp GAN đã lỗi thời |
| Độ sâu → ảnh | ControlNet-Depth trên Stable Diffusion |
| Siêu phân giải | Real-ESRGAN (GAN), ESRGAN-Plus, hoặc SD-Upscale (diffusion) |
| Tô màu | ColTran, các bộ tô màu dựa trên diffusion, hoặc Pix2Pix-color |
| Ngày → đêm, các mùa, thời tiết | CycleGAN hoặc dựa trên ControlNet |

Pix2Pix vẫn là công cụ phù hợp khi (a) bạn có hàng ngàn ví dụ có cặp, (b) tác vụ hẹp và có thể lặp lại, và (c) bạn cần suy luận nhanh. Đối với các tác vụ mở tổng quát, diffusion thắng thế.

## Triển khai

Lưu `outputs/skill-img2img-chooser.md`. Kỹ năng này cần mô tả tác vụ, tính sẵn có của dữ liệu (có cặp vs không cặp, N mẫu), và ngân sách về độ trễ/chất lượng, sau đó xuất ra: cách tiếp cận (Pix2Pix, CycleGAN, biến thể ControlNet, SDXL + IP-Adapter), yêu cầu dữ liệu huấn luyện, chi phí suy luận, và giao thức đánh giá (LPIPS, FID, đặc thù tác vụ).

## Bài tập

1. **Dễ.** Sửa đổi `code/main.py` để thêm lớp thứ ba. Xác nhận G vẫn ánh xạ nhiễu của mỗi lớp sang chế độ (mode) chính xác.
2. **Trung bình.** Thay thế L1 bằng một hàm mất mát kiểu perceptual trong môi trường 1-D (ví dụ: một D nhỏ đã đóng băng đóng vai trò là bộ trích xuất đặc trưng). Nó có làm thay đổi độ sắc nét của phân phối có điều kiện không?
3. **Khó.** Phác thảo một CycleGAN trong môi trường 1-D: hai phân phối, hai generator, cycle loss. Chứng minh rằng nó học cách ánh xạ giữa chúng mà không cần dữ liệu có cặp.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Conditional GAN | "GAN với nhãn" | G(z, c), D(x, c). Cả hai mạng đều thấy điều kiện. |
| Pix2Pix | "GAN image-to-image" | cGAN có cặp với G là U-Net và D là PatchGAN + hàm mất mát L1. |
| U-Net | "Encoder-decoder với skip" | Mạng conv đối xứng; các skip connection bảo toàn tần số cao. |
| PatchGAN | "Bộ phân loại tính thực cục bộ" | D xuất ra điểm số theo từng patch thay vì điểm số toàn cục. |
| CycleGAN | "Dịch ảnh không cặp" | Hai G + hàm mất mát cycle-consistency; không cần dữ liệu có cặp. |
| SPADE | "GauGAN" | Chuẩn hóa các kích hoạt trung gian với bản đồ ngữ nghĩa; phân đoạn-sang-ảnh. |
| FiLM | "Feature-wise linear modulation" | Biến đổi affine theo từng đặc trưng từ điều kiện; điều kiện hóa chi phí thấp. |

## Ghi chú sản xuất: Pix2Pix như một baseline giới hạn bởi độ trễ

Khi bạn có dữ liệu có cặp và một tác vụ hẹp (phác thảo → render, bản đồ ngữ nghĩa → ảnh, ngày → đêm), suy luận một lần (one-shot) của Pix2Pix vượt trội hơn diffusion về độ trễ theo bậc độ lớn. So sánh trong sản xuất thường là:

| Đường dẫn | Các bước | Độ trễ điển hình ở 512² trên một L4 |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net forward) | 1 | ~30 ms |
| SD-Inpaint hoặc SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix thắng về thông lượng trong các batch tĩnh (mọi yêu cầu đều có cùng số FLOPs). Diffusion thắng về chất lượng và khả năng tổng quát hóa. Cách làm hiện đại thường là triển khai một mô hình chưng cất (distilled) kiểu Pix2Pix cho tác vụ hẹp và một phương án dự phòng diffusion cho các đầu vào đuôi dài (tail inputs).

## Đọc thêm

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — bài báo cGAN gốc.
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix.
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN.
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD.
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN.
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — projection D.