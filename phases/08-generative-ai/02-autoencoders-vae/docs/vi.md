# Autoencoders & Variational Autoencoders (VAE)

> Một autoencoder thông thường thực hiện nén rồi tái tạo. Nó ghi nhớ. Nó không tạo ra dữ liệu mới. Chỉ cần thêm một thủ thuật — ép không gian mã (code space) phải tuân theo phân phối Gaussian — và bạn sẽ có một bộ lấy mẫu (sampler). Thủ thuật đơn lẻ đó, việc tái tham số hóa (reparameterization) của `z = μ + σ·ε`, chính là lý do tại sao mọi mô hình khuếch tán tiềm ẩn (latent-diffusion) và flow-matching mà bạn sử dụng vào năm 2026 đều có một VAE ở đầu vào.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Time:** ~75 phút

## Vấn đề

Nén một chữ số MNIST 784 pixel thành một mã gồm 16 số, sau đó tái tạo lại. Một autoencoder thông thường sẽ đạt điểm MSE tái tạo rất tốt nhưng không gian mã lại là một mớ hỗn độn. Chọn một điểm ngẫu nhiên trong không gian mã, giải mã nó, và bạn chỉ nhận được nhiễu. Nó không có bộ lấy mẫu. Nó chỉ là một mô hình nén được "trang điểm" lại.

Điều bạn thực sự muốn là: (a) không gian mã là một phân phối sạch, mượt mà mà bạn có thể lấy mẫu từ đó — ví dụ như Gaussian đẳng hướng `N(0, I)`, (b) giải mã bất kỳ mẫu nào cũng tạo ra một chữ số hợp lý, và (c) bộ mã hóa (encoder) và bộ giải mã (decoder) vẫn nén tốt. Ba mục tiêu, một kiến trúc, một hàm mất mát.

VAE của Kingma năm 2013 giải quyết vấn đề này bằng cách huấn luyện encoder xuất ra một *phân phối* `q(z|x) = N(μ(x), σ(x)²)`, kéo phân phối đó về phía phân phối tiên nghiệm (prior) `N(0, I)` thông qua hình phạt KL, và sau đó lấy mẫu `z` từ `q(z|x)` trước khi giải mã. Tại thời điểm suy luận (inference), hãy loại bỏ encoder, lấy mẫu `z ~ N(0, I)`, rồi giải mã. Hình phạt KL chính là thứ buộc không gian mã phải có cấu trúc.

Vào năm 2026, VAE hiếm khi được sử dụng độc lập — chúng đã bị vượt mặt bởi các mô hình khuếch tán (diffusion) về chất lượng hình ảnh thô — nhưng chúng là encoder được lựa chọn cho mọi mô hình latent-diffusion (SD 1/2/XL/3, Flux, AudioCraft). Học về VAE là bạn đang học về lớp đầu tiên "vô hình" của mọi pipeline hình ảnh mà bạn sử dụng.

## Khái niệm

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**Autoencoder.** `z = encoder(x)`, `x̂ = decoder(z)`, loss = `||x - x̂||²`. Không gian mã không có cấu trúc.

**VAE encoder.** Xuất ra hai vector: `μ(x)` và `log σ²(x)`. Chúng xác định `q(z|x) = N(μ, diag(σ²))`.

**Reparameterization trick.** Việc lấy mẫu từ `q(z|x)` không thể lấy đạo hàm. Hãy viết lại mẫu dưới dạng `z = μ + σ·ε` trong đó `ε ~ N(0, I)`. Bây giờ `z` là một hàm tất định của `(μ, σ)` cộng với một nhiễu không tham số — các gradient sẽ truyền qua `μ` và `σ`.

**Loss.** Evidence Lower BOund (ELBO), gồm hai thành phần:

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

Tái tạo (Reconstruction) đẩy `x̂` về phía `x`. KL đẩy `q(z|x)` về phía phân phối tiên nghiệm. Chúng cân bằng lẫn nhau. β nhỏ (<1) = mẫu sắc nét hơn, không gian mã ít mang tính Gaussian hơn. β lớn (>1) = không gian mã sạch hơn, mẫu bị mờ hơn. β-VAE (Higgins 2017) đã làm cho núm điều chỉnh này trở nên nổi tiếng và khởi đầu cho nghiên cứu về sự tách biệt (disentanglement).

**Sampling.** Tại thời điểm suy luận: rút `z ~ N(0, I)`, truyền qua decoder. Một lần truyền xuôi (forward pass) — không cần lấy mẫu lặp lại như diffusion.

```figure
vae-latent-grid
```

## Xây dựng

`code/main.py` triển khai một VAE nhỏ mà không cần numpy hay torch. Đầu vào là dữ liệu tổng hợp 8 chiều được rút ra từ hỗn hợp Gaussian 2 thành phần trong không gian 8-D. Encoder và decoder là các MLP một lớp ẩn. Chúng ta triển khai hàm kích hoạt tanh, forward pass, loss và một backward pass viết tay. Đây không phải là mã nguồn sản xuất — mà là để học tập.

### Bước 1: encoder forward

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`log σ²` thay vì `σ` để đầu ra của mạng không bị ràng buộc (softplus của σ là một cái bẫy — gradient sẽ chết tại σ ≈ 0).

### Bước 2: tái tham số hóa và giải mã

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Bước 3: ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

Sử dụng công thức KL dạng đóng (closed-form) vì cả hai phân phối đều là Gaussian. Đừng tích phân bằng số. Nhiều người vẫn gửi mã với các ước tính KL monte-carlo vào năm 2026 — nó chậm hơn gấp 3 lần mà không mang lại lợi ích gì.

### Bước 4: tạo dữ liệu

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

Đó chính là mô hình tạo sinh. Năm dòng mã.

## Các cạm bẫy

- **Posterior collapse.** Thành phần KL đẩy `q(z|x) → N(0, I)` quá mạnh khiến `z` không mang thông tin gì về `x`. Cách khắc phục: β-annealing (bắt đầu β=0, tăng dần lên 1), free bits, hoặc bỏ qua KL trên các chiều không hoạt động.
- **Mẫu bị mờ.** Hàm khả năng (likelihood) của decoder Gaussian ngụ ý tái tạo MSE, vốn là tối ưu Bayes cho L2 (giá trị trung bình) — trung bình của một tập hợp các chữ số hợp lý là một chữ số mờ. Cách khắc phục: decoder rời rạc (VQ-VAE, NVAE), hoặc chỉ sử dụng VAE làm encoder và xếp chồng diffusion lên các biến tiềm ẩn (đây là cách Stable Diffusion hoạt động).
- **β quá lớn, quá sớm.** Xem posterior collapse. Bắt đầu tại β≈0.01 và tăng dần.
- **Kích thước biến tiềm ẩn quá nhỏ.** 16-D hoạt động tốt cho MNIST, 256-D cho ImageNet 256², 2048-D cho ImageNet 1024². VAE của Stable Diffusion nén 512×512×3 → 64×64×4 (hệ số giảm mẫu 32x về diện tích không gian, 32x về kênh).

## Sử dụng

Stack VAE năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Encoder biến tiềm ẩn cho diffusion | Stable Diffusion VAE (`sd-vae-ft-ema`) hoặc Flux VAE |
| Encoder biến tiềm ẩn cho âm thanh | Encodec (Meta), SoundStream, hoặc DAC (Descript) |
| Biến tiềm ẩn cho video | Sora's spatiotemporal patches, Latte VAE, WAN VAE |
| Học biểu diễn tách biệt | β-VAE, FactorVAE, TCVAE |
| Biến tiềm ẩn rời rạc (cho mô hình transformer) | VQ-VAE, RVQ (ResidualVQ) |
| Biến tiềm ẩn liên tục cho tạo sinh | VAE thông thường, sau đó điều kiện hóa một mô hình flow/diffusion trong không gian tiềm ẩn đó |

Một mô hình latent-diffusion là một VAE với một mô hình diffusion nằm giữa encoder và decoder. VAE thực hiện nén thô, mô hình diffusion thực hiện phần việc nặng nhọc. Tương tự cho video (VAE + video-diffusion DiT) và âm thanh (Encodec + MusicGen transformer).

## Triển khai

Lưu `outputs/skill-vae-trainer.md`.

Kỹ năng cần có: hồ sơ tập dữ liệu + mục tiêu kích thước biến tiềm ẩn + mục đích sử dụng hạ nguồn (tái tạo, lấy mẫu, hoặc đầu vào cho latent-diffusion) và kết quả đầu ra: lựa chọn kiến trúc (plain/β/VQ/RVQ), lịch trình β, kích thước biến tiềm ẩn, hàm khả năng của decoder (Gaussian vs categorical), và kế hoạch đánh giá (recon MSE, KL trên mỗi chiều, khoảng cách Fréchet giữa `q(z|x)` và `N(0, I)`).

## Bài tập

1. **Dễ.** Thay đổi `β` trong `code/main.py` thành `0.01`, `0.1`, `1.0`, `5.0`. Ghi lại MSE tái tạo cuối cùng và KL. Giá trị β nào là Pareto-tốt nhất cho dữ liệu tổng hợp của bạn?
2. **Trung bình.** Thay thế hàm khả năng decoder Gaussian bằng hàm khả năng Bernoulli (loss cross-entropy). So sánh chất lượng mẫu trên phiên bản nhị phân hóa của cùng dữ liệu tổng hợp đó.
3. **Khó.** Mở rộng `code/main.py` thành một mini VQ-VAE: thay thế `z` liên tục bằng việc tra cứu láng giềng gần nhất trong một codebook gồm K=32 mục. So sánh MSE tái tạo và báo cáo có bao nhiêu mục codebook được sử dụng (hiện tượng codebook collapse là có thật).

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Autoencoder | Mạng mã hóa-giải mã | `x → z → x̂`, học MSE. Không phải mô hình tạo sinh. |
| VAE | AE với bộ lấy mẫu | Encoder xuất ra một phân phối, hình phạt KL định hình không gian mã. |
| ELBO | Evidence lower bound | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`; chặt chẽ khi `q = p(z\|x)`. |
| Reparameterization | `z = μ + σ·ε` | Viết lại nút ngẫu nhiên thành tất định + nhiễu thuần túy. Cho phép backprop qua việc lấy mẫu. |
| Prior | `p(z)` | Phân phối mục tiêu cho biến tiềm ẩn, thường là `N(0, I)`. |
| Posterior collapse | "KL term wins" | Encoder bỏ qua `x`, xuất ra prior; decoder phải tự "ảo giác". |
| β-VAE | Trọng số KL có thể điều chỉnh | `loss = recon + β·KL`. β cao hơn = tách biệt tốt hơn nhưng mờ hơn. |
| VQ-VAE | Biến tiềm ẩn rời rạc | Thay thế `z` liên tục bằng vector codebook gần nhất; cho phép mô hình hóa bằng transformer. |

## Lưu ý sản xuất: VAE là đường dẫn nóng nhất trong máy chủ diffusion

Trong pipeline Stable Diffusion / Flux / SD3, VAE được gọi hai lần mỗi yêu cầu — một lần để mã hóa (nếu thực hiện img2img / inpainting) và một lần để giải mã. Ở độ phân giải 1024², bước giải mã thường là đỉnh tiêu thụ bộ nhớ kích hoạt lớn nhất trong toàn bộ pipeline vì nó upsample các biến tiềm ẩn `128×128×16` trở lại `1024×1024×3`. Hai hệ quả thực tế:

- **Cắt hoặc lát (tile) quá trình giải mã.** `diffusers` cung cấp `pipe.vae.enable_slicing()` và `pipe.vae.enable_tiling()`. Việc lát (tiling) đánh đổi một vết nối nhỏ để lấy `O(tile²)` bộ nhớ thay vì `O(H·W)`. Rất cần thiết cho độ phân giải 1024²+ trên GPU người dùng phổ thông.
- **Decoder bf16, số học fp32 cho bước thay đổi kích thước cuối cùng.** VAE của SD 1.x được phát hành ở định dạng fp32 và *tạo ra các giá trị NaNs âm thầm* khi chuyển sang fp16 ở 1024²+. SDXL cung cấp `madebyollin/sdxl-vae-fp16-fix` — luôn ưu tiên biến thể sửa lỗi fp16 hoặc sử dụng bf16.

## Đọc thêm

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — bài báo gốc về VAE.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — β-VAE tách biệt.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — VAE hình ảnh hiện đại nhất.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion; VAE làm encoder.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec, tiêu chuẩn VAE âm thanh.