# StyleGAN

> Hầu hết các bộ tạo (generator) đều đưa `z` vào mọi lớp cùng một lúc. StyleGAN tách biệt điều đó: trước tiên ánh xạ `z` sang một `w` trung gian, sau đó *tiêm* `w` vào mọi cấp độ phân giải thông qua AdaIN. Thay đổi đơn lẻ đó đã gỡ rối không gian tiềm ẩn (latent space) và biến việc tạo khuôn mặt chân thực thành một bài toán đã được giải quyết trong suốt bảy năm qua.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 08 (Normalization), Phase 3 · 07 (CNNs)
**Time:** ~45 phút

## Vấn đề

DCGAN ánh xạ `z` thành một hình ảnh thông qua một chồng các lớp transposed convolution. Vấn đề là: `z` kiểm soát mọi thứ — tư thế, ánh sáng, danh tính, nền — tất cả đều bị đan xen vào nhau. Khi di chuyển dọc theo một trục của `z`, cả bốn yếu tố đều thay đổi. Bạn không thể yêu cầu mô hình "cùng một người nhưng tư thế khác" vì biểu diễn không được phân tách theo cách đó.

Karras và cộng sự (2019, NVIDIA) đã đề xuất: ngừng đưa `z` trực tiếp vào các lớp conv. Thay vào đó, hãy đưa một tensor `4×4×512` hằng số làm đầu vào mạng. Học một MLP 8 lớp để ánh xạ `z ∈ Z → w ∈ W`. Tiêm `w` vào mọi độ phân giải thông qua *adaptive instance normalization* (AdaIN): chuẩn hóa từng bản đồ đặc trưng (feature map) conv, sau đó co giãn và dịch chuyển bằng các phép chiếu affine của `w`. Thêm nhiễu theo từng lớp để tạo chi tiết ngẫu nhiên (lỗ chân lông, sợi tóc).

Kết quả: `W` có các trục gần như trực giao cho "phong cách cấp cao" (tư thế, danh tính) so với "phong cách chi tiết" (ánh sáng, màu sắc). Bạn có thể hoán đổi phong cách giữa hai hình ảnh bằng cách sử dụng `w` của ảnh A cho các cấp độ phân giải thấp và `w` của ảnh B cho các cấp độ cao. Điều này đã mở ra khả năng chỉnh sửa, cách điệu chéo miền và toàn bộ dòng nghiên cứu "StyleGAN-inversion".

## Khái niệm

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**Mạng ánh xạ (Mapping network).** `f: Z → W`, một MLP 8 lớp. `Z = N(0, I)^512`. `W` không bị ép buộc phải là phân phối Gaussian — nó học một hình dạng thích ứng với dữ liệu.

**Mạng tổng hợp (Synthesis network).** Bắt đầu từ một `4×4×512` hằng số đã học. Mỗi khối độ phân giải: `upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`. Độ phân giải tăng gấp đôi: 4, 8, 16, 32, 64, 128, 256, 512, 1024.

**AdaIN.**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

trong đó `y_scale` và `y_bias` đến từ các phép chiếu affine của `w`. Chuẩn hóa trên mỗi bản đồ đặc trưng, sau đó tạo kiểu lại. "Phong cách" ở đây là các thống kê bậc một và bậc hai của bản đồ đặc trưng.

**Nhiễu theo từng lớp (Per-layer noise).** Nhiễu Gaussian đơn kênh được thêm vào mỗi bản đồ đặc trưng, được co giãn bởi một hệ số học được cho mỗi kênh. Kiểm soát chi tiết ngẫu nhiên mà không ảnh hưởng đến cấu trúc tổng thể.

**Thủ thuật cắt bớt (Truncation trick).** Tại thời điểm suy luận (inference), lấy mẫu `z`, tính toán `w = mapping(z)`, sau đó `w' = ŵ + ψ·(w - ŵ)` trong đó `ŵ` là `w` trung bình trên nhiều mẫu. `ψ < 1` đánh đổi sự đa dạng lấy chất lượng. Hầu như mọi bản demo StyleGAN đều sử dụng `ψ ≈ 0.7`.

## StyleGAN 1 → 2 → 3

| Phiên bản | Năm | Đổi mới |
|-----------|------|----------|
| StyleGAN | 2019 | Mạng ánh xạ + AdaIN + nhiễu + tăng trưởng lũy tiến (progressive growing). |
| StyleGAN2 | 2020 | Giải điều chế trọng số (weight demodulation) thay thế AdaIN (khắc phục lỗi droplet); kiến trúc skip/residual; chính quy hóa độ dài đường dẫn. |
| StyleGAN3 | 2021 | Tích chập không răng cưa (alias-free convolution) + nhân tương đương; loại bỏ hiện tượng kết cấu dính vào lưới pixel. |
| StyleGAN-XL | 2022 | Có điều kiện theo lớp, 1024², ImageNet. |
| R3GAN | 2024 | Tái định vị với chính quy hóa mạnh hơn; thu hẹp khoảng cách với diffusion trên FFHQ-1024 với số lượng tham số ít hơn 20 lần. |

Vào năm 2026, StyleGAN3 vẫn là mặc định cho (a) tính chân thực miền hẹp ở FPS cao, (b) thích ứng miền ít dữ liệu (huấn luyện trên tập dữ liệu mới với 100 ảnh, đóng băng mạng ánh xạ), (c) chỉnh sửa dựa trên đảo ngược (tìm `w` để tái tạo ảnh thật, sau đó chỉnh sửa `w` đó). Đối với tạo ảnh từ văn bản miền mở, nó không phải là công cụ phù hợp — diffusion mới là lựa chọn đó.

```figure
gx-stylegan-mapping
```

## Xây dựng

`code/main.py` triển khai một "style-GAN lite" đồ chơi trong 1-D: một MLP ánh xạ, một hàm tổng hợp nhận một vector hằng số đã học và điều biến nó với tỷ lệ/độ lệch bắt nguồn từ `w`, và nhiễu theo từng lớp. Nó cho thấy việc tiêm `w` thông qua điều biến affine tương đương hoặc vượt trội hơn so với việc nối `z` vào đầu vào của bộ tạo.

### Bước 1: mạng ánh xạ

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Bước 2: adaptive instance normalization

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

Tỷ lệ và độ lệch trên mỗi bản đồ đặc trưng đến từ `w` thông qua phép chiếu tuyến tính.

### Bước 3: nhiễu theo từng lớp

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

Sigma trên mỗi kênh là có thể học được.

## Các cạm bẫy

- **Lỗi droplet (giọt nước).** StyleGAN 1 tạo ra các đốm giọt trong bản đồ đặc trưng vì AdaIN triệt tiêu giá trị trung bình. Giải điều chế trọng số của StyleGAN 2 khắc phục điều này bằng cách co giãn trọng số tích chập thay vì các kích hoạt.
- **Kết cấu dính (Texture sticking).** Kết cấu của StyleGAN 1 và 2 tuân theo tọa độ pixel, không phải tọa độ đối tượng (có thể thấy rõ khi nội suy). Các tích chập không răng cưa của StyleGAN 3 khắc phục điều này bằng các bộ lọc sinc có cửa sổ.
- **Độ bao phủ chế độ (Mode coverage).** Cắt bớt `ψ < 0.7` trông sạch sẽ nhưng lấy mẫu từ một hình nón hẹp; hãy sử dụng `ψ = 1.0` nếu bạn cần sự đa dạng.
- **Đảo ngược là mất mát dữ liệu.** Việc đảo ngược một bức ảnh thật thành `W` thường được thực hiện thông qua tối ưu hóa hoặc bộ mã hóa (e4e, ReStyle, HyperStyle). Kết quả bị trôi qua nhiều lần lặp.

## Sử dụng

| Trường hợp sử dụng | Cách tiếp cận |
|--------------------|---------------|
| Khuôn mặt người chân thực (anime, sản phẩm, hẹp) | StyleGAN3 FFHQ / tinh chỉnh tùy chỉnh |
| Chỉnh sửa khuôn mặt từ ảnh | Đảo ngược e4e + StyleSpace / hướng InterFaceGAN |
| Hoán đổi khuôn mặt / tái hiện | StyleGAN + bộ mã hóa + hòa trộn |
| Quy trình Avatar | StyleGAN3 với ADA để tinh chỉnh ít dữ liệu |
| Thích ứng miền từ vài hình ảnh | Đóng băng mạng ánh xạ, tinh chỉnh mạng tổng hợp |
| Tạo đa phương thức hoặc có điều kiện văn bản | Không nên — hãy sử dụng diffusion |

Đối với các bản demo cấp sản phẩm nơi câu trả lời là "ảnh khuôn mặt người", StyleGAN đánh bại diffusion về chi phí suy luận (một lần truyền xuôi, <10ms trên 4090) và độ sắc nét cho cùng một mức chất lượng.

## Triển khai

Lưu `outputs/skill-stylegan-inversion.md`. Kỹ năng lấy một bức ảnh thật và xuất ra: phương pháp đảo ngược (e4e / ReStyle / HyperStyle), mất mát tiềm ẩn dự kiến, ngân sách chỉnh sửa (bạn có thể di chuyển bao xa trong `W` trước khi xuất hiện lỗi), và danh sách các hướng chỉnh sửa tốt đã biết (tuổi, biểu cảm, tư thế).

## Bài tập

1. **Dễ.** Chạy `code/main.py` với `adain_on=True` và `adain_on=False`. So sánh sự lan tỏa của các đầu ra cho một latent cố định so với latent bị nhiễu.
2. **Trung bình.** Triển khai chính quy hóa trộn (mixing regularization): cho một batch huấn luyện, tính `w_a`, `w_b`, và áp dụng `w_a` cho nửa đầu của quá trình tổng hợp và `w_b` cho nửa sau. Bộ giải mã có học được các phong cách đã phân tách không?
3. **Khó.** Lấy một mô hình StyleGAN3 FFHQ đã được huấn luyện trước (ffhq-1024.pkl). Tìm hướng `w` kiểm soát "nụ cười" bằng cách huấn luyện SVM trên các mẫu được dán nhãn; báo cáo xem bạn có thể đẩy xa đến mức nào trước khi danh tính bị trôi.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|-----------|---------------|------------------|
| Mạng ánh xạ | "MLP" | `f: Z → W`, 8 lớp, tách hình học tiềm ẩn khỏi thống kê dữ liệu. |
| Không gian W | "Không gian phong cách" | Đầu ra của mạng ánh xạ; gần như đã được phân tách. |
| AdaIN | "Chuẩn hóa instance thích ứng" | Chuẩn hóa bản đồ đặc trưng, sau đó co giãn + dịch chuyển bằng phép chiếu `w`. |
| Thủ thuật cắt bớt | "Psi" | `w = mean + ψ·(w - mean)`, ψ<1 đánh đổi sự đa dạng lấy chất lượng. |
| Chính quy hóa độ dài đường dẫn | "PL reg" | Phạt các thay đổi lớn trong ảnh trên mỗi đơn vị thay đổi trong `w`; làm cho `W` mượt mà hơn. |
| Giải điều chế trọng số | "Bản sửa lỗi StyleGAN2" | Chuẩn hóa trọng số conv thay vì kích hoạt; loại bỏ lỗi droplet. |
| Không răng cưa | "Thủ thuật của StyleGAN3" | Bộ lọc sinc có cửa sổ; loại bỏ hiện tượng kết cấu dính vào lưới pixel. |
| Đảo ngược | "Tìm w cho ảnh thật" | Tối ưu hóa hoặc mã hóa `x → w` sao cho `G(w) ≈ x`. |

## Ghi chú sản xuất: tại sao StyleGAN vẫn được sử dụng vào năm 2026

StyleGAN3 trên 4090 tạo ra khuôn mặt FFHQ 1024² trong dưới 10 ms — `num_steps = 1`, không cần giải mã VAE, không cần truyền cross-attention. Về mặt sản xuất, đây là độ trễ sàn cho bất kỳ bộ tạo ảnh nào. Một quy trình SDXL 50 bước + giải mã VAE ở cùng độ phân giải mất khoảng 3 giây. Đó là **khoảng cách 300 lần**, và đối với các sản phẩm miền hẹp (dịch vụ avatar, quy trình tài liệu ID, tạo khuôn mặt chứng khoán), nó thắng về TCO.

Hai hệ quả vận hành:

- **Không lập lịch, không gom batch.** Batch tĩnh tại mức chiếm dụng mục tiêu là tối ưu. Gom batch liên tục (thiết yếu cho LLM và diffusion) không mang lại lợi ích gì vì mọi yêu cầu đều tốn cùng một lượng FLOP.
- **Cắt bớt `ψ` là núm an toàn.** `ψ < 0.7` lấy mẫu từ một hình nón hẹp trong phạm vi của mạng ánh xạ. Đây là đòn bẩy duy nhất mà lớp phục vụ có đối với phương sai mẫu. Giảm `ψ` khi tải cao điểm, tăng nó cho người dùng cao cấp.

## Đọc thêm

- [Karras và cộng sự (2019). Kiến trúc bộ tạo dựa trên phong cách cho GANs](https://arxiv.org/abs/1812.04948) — StyleGAN.
- [Karras và cộng sự (2020). Phân tích và cải thiện chất lượng hình ảnh của StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras và cộng sự (2021). Mạng đối nghịch tạo không răng cưa](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Tov và cộng sự (2021). Thiết kế bộ mã hóa để thao tác hình ảnh StyleGAN](https://arxiv.org/abs/2102.02766) — Đảo ngược e4e.
- [Sauer và cộng sự (2022). StyleGAN-XL: Mở rộng StyleGAN sang các tập dữ liệu đa dạng lớn](https://arxiv.org/abs/2202.00273) — StyleGAN-XL.
- [Huang và cộng sự (2024). R3GAN: GAN đã chết; GAN trường tồn!](https://arxiv.org/abs/2501.05441) — công thức GAN tối giản hiện đại.