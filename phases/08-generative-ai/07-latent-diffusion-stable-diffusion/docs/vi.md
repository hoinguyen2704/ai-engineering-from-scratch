# Latent Diffusion & Stable Diffusion

> Khuếch tán trong không gian pixel (pixel-space diffusion) trên ảnh 512×512 là một "tội ác" về mặt tính toán. Rombach và cộng sự (2022) nhận thấy rằng bạn không cần tất cả 786k chiều để tạo ra một hình ảnh — bạn chỉ cần đủ để nắm bắt cấu trúc ngữ nghĩa, và một bộ giải mã (decoder) riêng biệt cho phần còn lại. Hãy thực hiện khuếch tán bên trong không gian tiềm ẩn (latent space) của một VAE. Ý tưởng đó chính là Stable Diffusion.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**Time:** ~75 phút

## Vấn đề

Khuếch tán trong không gian pixel ở độ phân giải 512² có nghĩa là U-Net chạy trên các tensor có hình dạng `[B, 3, 512, 512]`. Mỗi bước lấy mẫu (sampling step) tiêu tốn khoảng 100 GFLOPS cho một U-Net 500M tham số. Năm mươi bước là 5 TFLOPS cho mỗi hình ảnh. Nếu huấn luyện trên một tỷ hình ảnh, chi phí tính toán sẽ trở nên vô lý.

Hầu hết các FLOP đó bị lãng phí vào việc đẩy các chi tiết không quan trọng về mặt nhận thức qua mạng — các kết cấu tần số cao mà một VAE có tổn hao (lossy VAE) có thể nén lại. Ý tưởng của Rombach: huấn luyện một VAE một lần (*giai đoạn đầu*), đóng băng nó, và thực hiện khuếch tán hoàn toàn trong không gian tiềm ẩn 4 kênh 64×64 (*giai đoạn hai*). Vẫn là U-Net đó. Nhưng chỉ bằng 1/16 số pixel. FLOP ít hơn khoảng 64 lần với chất lượng tương đương.

Đây là công thức của Stable Diffusion. SD 1.x / 2.x sử dụng U-Net 860M trên các latent `64×64×4`, SDXL sử dụng U-Net 2.6B trên `128×128×4`, SD3 thay thế U-Net bằng Diffusion Transformer (DiT) với flow matching. Flux.1-dev (Black Forest Labs, 2024) xuất xưởng với DiT-MMDiT 12B tham số. Tất cả đều chạy trên cùng một nền tảng hai giai đoạn.

## Khái niệm

![Latent diffusion: VAE compression + diffusion in latent space](../assets/latent-diffusion.svg)

**Hai giai đoạn, được huấn luyện riêng biệt.**

1. **Giai đoạn 1 — VAE.** Encoder `E(x) → z`, decoder `D(z) → x`. Mục tiêu nén: giảm mẫu (downsample) 8× ở mỗi trục không gian + điều chỉnh các kênh sao cho tổng kích thước latent bằng khoảng 1/16 số lượng pixel. Loss = tái tạo (L1 + LPIPS perceptual) + KL (trọng số nhỏ để `z` không bị ép buộc quá mức thành Gaussian, vì chúng ta không cần lấy mẫu chính xác từ `z`). Thường được huấn luyện với adversarial loss để ảnh giải mã sắc nét hơn.

2. **Giai đoạn 2 — khuếch tán trên `z`.** Coi `z = E(x_real)` là dữ liệu. Huấn luyện một U-Net (hoặc DiT) để khử nhiễu `z_t`. Tại thời điểm suy luận (inference): lấy mẫu `z_0` thông qua khuếch tán, sau đó `x = D(z_0)`.

**Điều kiện hóa văn bản (Text conditioning).** Hai thành phần bổ sung. Một bộ mã hóa văn bản (text encoder) đã đóng băng (CLIP-L cho SD 1.x, CLIP-L+OpenCLIP-G cho SD 2/XL, T5-XXL cho SD3 và Flux). Một cơ chế cross-attention: mỗi khối U-Net nhận `[Q = image features, K = V = text tokens]` và trộn chúng vào. Các token là cách duy nhất để văn bản ảnh hưởng đến hình ảnh.

**Hàm loss giống hệt Bài 06.** Cùng là DDPM / flow matching MSE trên nhiễu. Bạn chỉ cần thay đổi miền dữ liệu.

## Các biến thể kiến trúc

| Model | Năm | Backbone | Hình dạng Latent | Text encoder | Tham số |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | same | 1-4 step sampling |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 step |

Xu hướng: thay thế U-Net bằng DiT (transformer trên các patch latent), mở rộng text encoder (T5 vượt trội hơn CLIP trong việc bám sát prompt), tăng số kênh latent (4 → 16 mang lại nhiều không gian chi tiết hơn).

```figure
noise-schedule
```

## Xây dựng

`code/main.py` xếp chồng một "VAE" 1-D đơn giản (encoder + decoder đồng nhất, để minh họa; một VAE thực tế sẽ là mạng tích chập) lên trên DDPM từ Bài 06 và thêm điều kiện hóa lớp (class conditioning) với classifier-free guidance. Nó cho thấy rằng cùng một loss khuếch tán hoạt động bất kể bạn chạy trên các giá trị 1-D thô hay trên các giá trị đã mã hóa — đó là chìa khóa quan trọng.

### Bước 1: encoder/decoder

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

Một VAE thực tế có các trọng số đã được huấn luyện. Để giảng dạy, phép ánh xạ tuyến tính này là đủ để cho thấy khuếch tán hoạt động trên `z` mà không quan tâm đến không gian dữ liệu gốc.

### Bước 2: khuếch tán trong không gian `z`

Cùng DDPM như Bài 06. Dữ liệu mà mạng nhìn thấy là `z = E(x)`. Sau khi lấy mẫu `z_0`, giải mã với `D(z_0)`.

### Bước 3: classifier-free guidance

Trong quá trình huấn luyện, loại bỏ nhãn lớp 10% thời gian (thay thế bằng một null token). Tại thời điểm suy luận, tính toán cả `ε_cond` và `ε_uncond`, sau đó:

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = không có guidance (đa dạng tối đa), `w = 3` = mặc định, `w = 7+` = bão hòa / quá sắc nét.

### Bước 4: điều kiện hóa văn bản (khái niệm, không phải mã)

Thay thế nhãn lớp bằng đầu ra của một text encoder đã đóng băng. Đưa embedding văn bản vào U-Net thông qua cross-attention:

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

Đây là sự khác biệt thực chất duy nhất giữa một mô hình khuếch tán có điều kiện lớp và Stable Diffusion.

## Các cạm bẫy

- **Sai lệch quy mô VAE.** Các VAE của SD 1.x có một hằng số tỷ lệ (`scaling_factor ≈ 0.18215`) được áp dụng sau khi mã hóa. Quên điều này sẽ khiến U-Net huấn luyện trên các latent có phương sai sai lệch nghiêm trọng. Mỗi checkpoint đều đi kèm với một hằng số này.
- **Text encoder sai lệch âm thầm.** SD3 cần T5-XXL với >=128 token, và việc quay lại chỉ dùng CLIP sẽ gây mất mát thông tin. Luôn kiểm tra `use_t5=True` nếu không độ trung thực của prompt sẽ giảm sút.
- **Trộn lẫn các không gian latent.** SDXL, SD3, Flux đều sử dụng các VAE khác nhau. Một LoRA được huấn luyện trên latent của SDXL sẽ không hoạt động trên SD3. Hugging Face diffusers 0.30+ sẽ từ chối tải các checkpoint không khớp.
- **CFG quá cao.** `w > 10` tạo ra các hình ảnh bão hòa, bóng bẩy và quá khớp (over-fit) với prompt nhưng làm giảm sự đa dạng. Điểm ngọt (sweet spot) là `w = 3-7`.
- **Rò rỉ negative prompt.** Negative prompt trống trở thành null token; một negative prompt đầy đủ trở thành `ε_uncond`. Chúng không giống nhau; một số pipeline mặc định âm thầm về null.

## Sử dụng

Các stack sản xuất năm 2026:

| Mục tiêu | Backbone khuyến nghị |
|--------|----------------------|
| Miền hẹp, dữ liệu có cặp, huấn luyện từ đầu | SDXL fine-tune (LoRA / full) — nhanh nhất để triển khai |
| Text-to-image miền mở, trọng số mở | Flux.1-dev (12B, Apache / phi thương mại) hoặc SD3.5-Large |
| Suy luận nhanh nhất, trọng số mở | Flux.1-schnell (1-4 bước, Apache) hoặc SDXL-Lightning |
| Bám sát prompt tốt nhất, được lưu trữ (hosted) | GPT-Image / DALL-E 3 (vẫn vậy), Midjourney v7, Imagen 4 |
| Quy trình chỉnh sửa | Flux.1-Kontext (Tháng 12/2024) — hỗ trợ ảnh + văn bản gốc |
| Nghiên cứu, baseline | SD 1.5 — cổ điển nhưng được nghiên cứu kỹ lưỡng |

## Triển khai

Lưu `outputs/skill-sd-prompter.md`. Kỹ năng bao gồm lấy một prompt văn bản + phong cách mục tiêu và xuất ra: mô hình + checkpoint, thang đo CFG, sampler, negative prompt, độ phân giải, tổ hợp ControlNet/IP-Adapter tùy chọn, và danh sách kiểm tra QA từng bước.

## Bài tập

1. **Dễ.** Chạy `code/main.py` với guidance `w ∈ {0, 1, 3, 7, 15}`. Ghi lại mẫu trung bình theo lớp. Tại `w` nào thì các giá trị trung bình của lớp phân kỳ vượt quá giá trị trung bình của dữ liệu thực?
2. **Trung bình.** Thay thế encoder tuyến tính đơn giản bằng một cặp encoder/decoder tanh-MLP với loss tái tạo. Huấn luyện lại khuếch tán trên các latent mới. Chất lượng mẫu có thay đổi không?
3. **Khó.** Thiết lập một suy luận Stable Diffusion thực tế với diffusers: tải `sdxl-base`, chạy 30 bước Euler với CFG=7, đo thời gian. Bây giờ chuyển sang `sdxl-turbo` với 4 bước và CFG=0. Cùng một chủ đề, chất lượng khác nhau — hãy mô tả những gì đã thay đổi và tại sao.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Giai đoạn đầu | "VAE" | Cặp encoder/decoder đã huấn luyện; nén 512² xuống 64². |
| Giai đoạn hai | "U-Net" | Mô hình khuếch tán trên không gian tiềm ẩn. |
| CFG | "Guidance scale" | `(1+w)·ε_cond - w·ε_uncond`; điều chỉnh cường độ điều kiện hóa. |
| Null token | "Empty prompt embed" | Embedding không điều kiện được sử dụng cho `ε_uncond`. |
| Cross-attention | "Cách văn bản đi vào" | Mỗi khối U-Net chú ý đến các token văn bản dưới dạng K và V. |
| DiT | "Diffusion Transformer" | Thay thế U-Net bằng transformer trên các patch latent; mở rộng tốt hơn. |
| MMDiT | "Multi-modal DiT" | Kiến trúc của SD3: các luồng văn bản và hình ảnh với sự chú ý chung. |
| VAE scaling factor | "Magic number" | Chia các latent cho ~5.4 để khuếch tán hoạt động trong không gian phương sai đơn vị. |

## Lưu ý sản xuất: chạy Flux-12B trên GPU tiêu dùng 8GB

Tích hợp Flux tham chiếu là công thức kinh điển "Tôi có GPU tiêu dùng, tôi có thể triển khai cái này không?". Bí quyết là áp dụng công thức ba nút điều chỉnh trong tài liệu suy luận sản xuất cho một Diffusion DiT:

1. **Tải so le (Staggered loading).** Flux có ba mạng không bao giờ cần tồn tại cùng lúc trong VRAM: T5-XXL text encoder (~10 GB ở fp32), CLIP-L (nhỏ), 12B MMDiT, và VAE. Mã hóa prompt trước, *xóa* các encoder, tải DiT, khử nhiễu, *xóa* DiT, tải VAE, giải mã. GPU tiêu dùng 8GB chỉ chứa được một giai đoạn tại một thời điểm.
2. **Lượng tử hóa 4-bit qua bitsandbytes.** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` trên cả T5 encoder và DiT. Giảm bộ nhớ 8 lần, chất lượng giảm không đáng kể đối với text-to-image theo các benchmark của Aritra (được liên kết trong notebook).
3. **CPU offload.** `pipe.enable_model_cpu_offload()` tự động hoán đổi các module giữa CPU và GPU khi mỗi lần truyền xuôi (forward pass) tiến triển. Thêm 10-20% độ trễ nhưng làm cho pipeline có thể chạy được.

Hạch toán bộ nhớ là: `10 GB T5 / 8 = 1.25 GB` đã lượng tử hóa, `12 B params × 0.5 bytes = ~6 GB` DiT đã lượng tử hóa, cộng với các activation. Theo thuật ngữ của stas00, đây là mức cực hạn của suy luận TP=1 — không song song hóa mô hình, lượng tử hóa tối đa. Đối với sản xuất, bạn sẽ chạy TP=2 hoặc TP=4 trên H100; đối với một laptop dev đơn lẻ, đây là công thức.

## Đọc thêm

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 family.
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — tài liệu triển khai tham chiếu cho mọi checkpoint ở trên.