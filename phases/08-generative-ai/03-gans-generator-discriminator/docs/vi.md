# GANs — Generator vs Discriminator

> Thủ thuật của Goodfellow vào năm 2014 là bỏ qua hoàn toàn mật độ xác suất. Hai mạng lưới. Một mạng tạo ra dữ liệu giả. Một mạng bắt bài chúng. Chúng đối đầu nhau cho đến khi dữ liệu giả không thể phân biệt được với dữ liệu thật. Về lý thuyết thì không nên hoạt động. Thực tế thường là không. Nhưng khi nó hoạt động, các mẫu tạo ra vẫn là những mẫu sắc nét nhất trong tài liệu cho các miền dữ liệu hẹp.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 08 (Optimizers), Phase 8 · 02 (VAE)
**Time:** ~75 phút

## Vấn đề

VAEs tạo ra các mẫu bị mờ vì hàm mất mát MSE decoder của chúng là Bayes-optimal cho ảnh *trung bình* — và trung bình của nhiều chữ số hợp lý lại là một chữ số nhòe. Bạn cần một hàm mất mát khen thưởng tính *hợp lý*, chứ không phải sự gần gũi về pixel với bất kỳ mục tiêu cụ thể nào. Không có công thức đóng cho tính hợp lý. Bạn phải học nó.

Ý tưởng của Goodfellow: huấn luyện một bộ phân loại `D(x)` để phân biệt ảnh thật và ảnh giả. Huấn luyện một bộ tạo `G(z)` để đánh lừa `D`. Tín hiệu mất mát cho `G` chính là những gì `D` hiện đang nghĩ là làm cho một thứ trông có vẻ thật. Tín hiệu này cập nhật khi `G` cải thiện, đuổi theo một mục tiêu di động. Nếu cả hai mạng hội tụ, `G` đã học được phân phối dữ liệu mà không cần phải viết ra `log p(x)`.

Đây là huấn luyện đối nghịch (adversarial training). Toán học của nó là một trò chơi minimax:

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

Vào năm 2026, GANs không còn là bộ tạo SOTA (diffusion và flow matching đã chiếm ngôi vương đó). Nhưng StyleGAN 2/3 vẫn là các mô hình khuôn mặt sắc nét nhất từng được phát hành, các discriminator của GAN được sử dụng làm *perceptual losses* trong huấn luyện diffusion, và huấn luyện đối nghịch thúc đẩy các kỹ thuật chưng cất 1 bước nhanh (SDXL-Turbo, SD3-Turbo, LCM) cho phép bạn triển khai diffusion thời gian thực.

## Khái niệm

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**Generator `G(z)`.** Ánh xạ một vector nhiễu `z ~ N(0, I)` thành một mẫu `x̂`. Một mạng có cấu trúc decoder (dense hoặc transposed conv).

**Discriminator `D(x)`.** Ánh xạ một mẫu thành một xác suất (hoặc điểm số) vô hướng. Thật → 1, giả → 0.

**Loss.** Hai cập nhật xen kẽ:

- **Huấn luyện `D`:** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`. Binary cross-entropy trên thật=1, giả=0.
- **Huấn luyện `G`:** `loss_G = -log D(G(z))`. Đây là dạng *non-saturating* mà Goodfellow đã sử dụng (dạng `log(1 - D(G(z)))` gốc bị bão hòa và triệt tiêu gradient khi `D` quá tự tin).

**Vòng lặp huấn luyện.** Một bước của `D`, một bước của `G`. Lặp lại.

**Tại sao nó hoạt động.** Nếu `G` khớp hoàn hảo với `p_data`, thì `D` không thể làm tốt hơn việc đoán ngẫu nhiên và xuất ra 0.5 ở mọi nơi; `G` không nhận được thêm gradient nào nữa. Trạng thái cân bằng.

**Tại sao nó thất bại.** Mode collapse (`G` tìm thấy một mode mà `D` không thể phân loại và tạo ra nó mãi mãi), vanishing gradient (`D` học quá nhanh và `log D` bị bão hòa), mất ổn định huấn luyện (learning rates, batch sizes, bất cứ thứ gì).

## Các biến thể làm cho GAN hoạt động

| Năm | Cải tiến | Giải pháp |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv, batch norm, LeakyReLU — kiến trúc ổn định đầu tiên. |
| 2017 | WGAN, WGAN-GP | Thay thế BCE bằng khoảng cách Wasserstein + gradient penalty. Khắc phục vanishing gradient. |
| 2017 | Spectral normalization | Giới hạn Lipschitz cho discriminator. Vẫn được dùng trong các discriminator năm 2026. |
| 2018 | Progressive GAN | Huấn luyện độ phân giải thấp trước, thêm các lớp dần dần. Kết quả megapixel đầu tiên. |
| 2019 | StyleGAN / StyleGAN2 | Mapping network + adaptive instance norm. SOTA cho photorealism miền cố định. |
| 2021 | StyleGAN3 | Alias-free, translation-equivariant — vẫn là tiêu chuẩn vàng cho khuôn mặt năm 2026. |
| 2022 | StyleGAN-XL | Conditional, class-aware, quy mô lớn hơn. |
| 2024 | R3GAN | Tái định nghĩa với regularization mạnh hơn; hoạt động ở 1024² mà không cần thủ thuật. |

```figure
gan-minimax
```

## Xây dựng

`code/main.py` huấn luyện một GAN nhỏ trên dữ liệu 1-D: hỗn hợp của hai phân phối Gaussian. Generator và discriminator là các MLP một lớp ẩn. Chúng ta thực hiện forward, backward và vòng lặp minimax thủ công. Mục tiêu là nhìn thấy hai chế độ thất bại chính (mode collapse + vanishing gradient) khi chúng xảy ra.

### Bước 1: non-saturating loss

Hàm mất mát Goodfellow cơ bản `log(1 - D(G(z)))` tiến về 0 khi D phân loại dữ liệu giả của G là giả với độ tự tin cao. Tại thời điểm đó, gradient cho G về cơ bản bằng 0 — G không thể cải thiện. Dạng non-saturating `-log D(G(z))` có đường tiệm cận ngược lại: nó bùng nổ khi D tự tin, cung cấp cho G một tín hiệu mạnh mẽ.

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Bước 2: một bước discriminator cho mỗi bước generator

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

Dữ liệu giả mới cho G, nếu không các gradient sẽ bị cũ (stale).

### Bước 3: theo dõi mode collapse

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

Triệu chứng điển hình: một trong hai mode thật ngừng được tạo ra. Discriminator ngừng sửa lỗi vì nó không bao giờ thấy mode đó là giả.

## Các cạm bẫy

- **Discriminator quá mạnh.** Giảm learning rate của D xuống 2-5 lần, hoặc thêm instance/layer noise. Nếu D đạt độ chính xác >95%, G sẽ "chết".
- **Generator ghi nhớ một mode.** Thêm nhiễu vào đầu vào của D, sử dụng lớp minibatch-discriminator, hoặc chuyển sang WGAN-GP.
- **Batch norm làm rò rỉ thống kê.** Batch thật + batch giả chảy qua cùng một lớp BN sẽ trộn lẫn thống kê của chúng. Hãy sử dụng instance norm hoặc spectral norm thay thế.
- **Gian lận Inception-score.** FID và IS rất nhiễu ở số lượng mẫu thấp. Sử dụng ≥10k mẫu khi đánh giá.
- **One-shot sampling là lời nói dối cho các tác vụ có điều kiện.** Bạn vẫn cần CFG scales, các thủ thuật truncation và re-sampling để có đầu ra sử dụng được.

## Sử dụng

Stack GAN năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Khuôn mặt người chân thực, tư thế cố định | StyleGAN3 (sắc nét nhất, nhỏ nhất) |
| Anime / khuôn mặt cách điệu | StyleGAN-XL hoặc Stable Diffusion LoRA |
| Image-to-image translation | Pix2Pix / CycleGAN (Phase 8 · 04) hoặc ControlNet (Phase 8 · 08) |
| Fast 1-step text-to-image | Adversarial distillation của diffusion (SDXL-Turbo, SD3-Turbo) |
| Perceptual loss trong huấn luyện diffusion | GAN discriminator nhỏ trên các vùng cắt ảnh |
| Bất cứ thứ gì đa phương thức, mở | Đừng dùng — hãy dùng diffusion hoặc flow matching |

GAN sắc nét nhưng hẹp. Một khi miền dữ liệu của bạn mở rộng — ảnh chụp, prompt văn bản tùy ý, video — hãy chuyển sang diffusion. Thủ thuật đối nghịch vẫn tồn tại như một thành phần (perceptual losses, distillation), không phải là một bộ tạo độc lập.

## Triển khai

Lưu `outputs/skill-gan-debugger.md`. Skill lấy một lần chạy GAN thất bại (đường cong loss, lưới mẫu, kích thước tập dữ liệu) và xuất ra danh sách xếp hạng các nguyên nhân có khả năng xảy ra, các bản sửa lỗi một dòng và giao thức chạy lại.

## Bài tập

1. **Dễ.** Chạy `code/main.py` với các cài đặt mặc định. Sau đó đặt `D_LR = 5 * G_LR` và chạy lại. Loss của G sụp đổ về hằng số nhanh như thế nào?
2. **Trung bình.** Thay thế BCE loss của Goodfellow bằng WGAN loss: `loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]`, và cắt (clip) trọng số của D về `[-0.01, 0.01]`. Huấn luyện có ổn định hơn không? So sánh thời gian hội tụ thực tế.
3. **Khó.** Mở rộng ví dụ 1-D sang dữ liệu 2-D (hỗn hợp 8 Gaussian trên một vòng tròn). Theo dõi xem generator bắt được bao nhiêu trong số 8 mode ở các bước 1k, 5k, 10k. Triển khai minibatch discrimination và đo lường lại.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Generator | "G" | Mạng nhiễu-thành-mẫu, `G: z → x̂`. |
| Discriminator | "D" | Bộ phân loại `D: x → [0, 1]`, thật vs giả. |
| Minimax | "Trò chơi" | `min_G max_D` của một mục tiêu chung. |
| Non-saturating loss | "Bản sửa lỗi" | Sử dụng `-log D(G(z))` cho G thay vì `log(1 - D(G(z)))`. |
| Mode collapse | "G ghi nhớ một thứ" | Generator tạo ra ít đầu ra khác biệt dù dữ liệu đa dạng. |
| WGAN | "Wasserstein" | Thay BCE bằng khoảng cách Earth-Mover + gradient penalty; gradient mượt hơn. |
| Spectral norm | "Thủ thuật Lipschitz" | Ràng buộc chuẩn trọng số của D để giới hạn độ dốc; ổn định huấn luyện. |
| StyleGAN | "Cái hoạt động được" | Mapping network + AdaIN; tốt nhất trong phân khúc cho khuôn mặt, vẫn dùng năm 2026. |

## Ghi chú sản xuất: one-shot inference là lợi thế lâu dài của GAN

GAN không còn thắng về chất lượng mẫu cho tạo dữ liệu miền mở, nhưng chúng vẫn thắng về chi phí suy luận. Trong từ vựng tài liệu suy luận sản xuất, một GAN có:

- **Không có giai đoạn prefill, không có decode.** Một lần forward pass `G(z)` duy nhất. TTFT ≈ tổng độ trễ.
- **Không áp lực KV-cache.** Trạng thái duy nhất là các trọng số. Batch size bị giới hạn bởi bộ nhớ kích hoạt, không phải cache.
- **Continuous batching tầm thường.** Vì mỗi yêu cầu tốn cùng một lượng FLOPs cố định, một batch tĩnh tại mức chiếm dụng mục tiêu của máy chủ thường là tối ưu. Không cần bộ lập lịch in-flight.

Đây là lý do tại sao GAN distillation (SDXL-Turbo, SD3-Turbo, ADD, LCM) là kỹ thuật thống trị cho text-to-image nhanh vào năm 2026: nó nén một pipeline diffusion 20-50 bước thành 1-4 forward pass kiểu GAN trong khi vẫn giữ được phân phối của một base diffusion. Adversarial loss tồn tại như một núm vặn trong quá trình huấn luyện để biến các bộ tạo chậm thành bộ tạo nhanh.

## Đọc thêm

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — bài báo GAN gốc.
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — kiến trúc ổn định đầu tiên.
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN.
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo.