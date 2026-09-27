# Generative Models — Taxonomy & History

> Mọi mô hình hình ảnh, văn bản, video và 3D đều nằm trong một trong năm nhóm. Chọn sai nhóm, bạn sẽ phải vật lộn với toán học trong nhiều tuần. Chọn đúng nhóm, mười hai năm tiến bộ của lĩnh vực này sẽ được sắp xếp gọn gàng trong tư duy của bạn.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 2 (ML Fundamentals), Phase 3 (Deep Learning Core), Phase 7 · 14 (Transformers)
**Time:** ~45 minutes

## The Problem

Một mô hình generative chỉ có một nhiệm vụ: dựa trên các mẫu huấn luyện được rút ra từ một phân phối chưa biết `p_data(x)`, hãy xuất ra các mẫu mới trông như thể chúng đến từ cùng một phân phối đó. Khuôn mặt, câu văn, tệp MIDI, cấu trúc protein — tất cả đều là cùng một vấn đề nếu bạn nhìn kỹ.

Vấn đề nằm ở chỗ `p_data` tồn tại trong một không gian với hàng triệu chiều (một hình ảnh RGB 512x512 có khoảng 786k chiều), các mẫu nằm trên một đa tạp (manifold) mỏng bên trong không gian đó, và bạn chỉ có khoảng 10 triệu ví dụ. Việc vét cạn mật độ là vô vọng. Mọi mô hình generative đều là một sự thỏa hiệp, đánh đổi một vấn đề khó lấy một vấn đề ít khó hơn.

Năm nhóm mô hình đã tồn tại qua mười hai năm qua. Biết được sự thỏa hiệp mà mỗi nhóm thực hiện sẽ cho bạn biết tại sao chúng thắng ở một số tác vụ và thất bại ở những tác vụ khác.

## The Concept

![Five families of generative models — taxonomy by what they model](../assets/taxonomy.svg)

**1. Explicit density, tractable (Mật độ tường minh, có thể tính toán được).** Viết `log p(x)` dưới dạng một tổng mà bạn thực sự có thể đánh giá. Các mô hình autoregressive (PixelCNN, WaveNet, GPT) phân tích nhân tử `p(x) = ∏ p(x_i | x_<i)`. Normalizing flows (RealNVP, Glow) xây dựng `p(x)` như một phép biến đổi khả nghịch của một cơ sở đơn giản. Ưu điểm: likelihood chính xác, loss huấn luyện sạch. Nhược điểm: inference autoregressive mang tính tuần tự (chậm đối với các chuỗi dài), flows cần các kiến trúc khả nghịch (hạn chế về kiến trúc).

**2. Explicit density, approximate (Mật độ tường minh, xấp xỉ).** Chặn dưới `log p(x)` (ELBO) và tối ưu hóa chặn đó. VAEs (Kingma 2013) sử dụng encoder-decoder với một variational posterior. Diffusion models (DDPM, Ho 2020) huấn luyện một bộ khử nhiễu (denoiser) giúp tối ưu hóa ngầm định một ELBO có trọng số. Diffusion là xương sống thống trị cho hình ảnh, video và 3D vào năm 2026.

**3. Implicit density (Mật độ ẩn).** Bỏ qua mật độ hoàn toàn; học một bộ tạo `G(z)` để tạo ra các mẫu và một bộ phân biệt `D(x)` để phân biệt thật/giả. GANs (Goodfellow 2014). Nhanh ở khâu inference (một lần forward pass) nhưng nổi tiếng là không ổn định trong quá trình huấn luyện. StyleGAN 1/2/3 vẫn là state-of-the-art cho photorealism trong miền cố định (khuôn mặt, phòng ngủ) ngay cả vào năm 2026.

**4. Score-based / continuous-time (Dựa trên điểm số / thời gian liên tục).** Học trực tiếp gradient của log-density `∇_x log p(x)` (điểm số). Song & Ermon (2019) đã chỉ ra rằng score matching tổng quát hóa diffusion thành một SDE. Flow matching (Lipman 2023) là xu hướng hot nhất giai đoạn 2024-2026: huấn luyện không cần mô phỏng, đường đi thẳng hơn, sampling nhanh hơn 4-10 lần so với DDPM. Stable Diffusion 3, Flux, AudioCraft 2 đều sử dụng flow matching.

**5. Token-based autoregressive over discrete codes (Autoregressive dựa trên token qua các mã rời rạc).** Nén dữ liệu chiều cao bằng VQ-VAE hoặc residual quantizer thành một chuỗi ngắn các token rời rạc, sau đó sử dụng Transformer để mô hình hóa chuỗi token đó. Parti, MuseNet, AudioLM, VALL-E, patch tokenizer của Sora đều sử dụng cách này. Đây là nhóm 1 cộng với một bộ tokenizer đã được học.

## A brief history

| Năm | Mô hình | Tại sao nó quan trọng |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | Mô hình generative sâu đầu tiên với loss huấn luyện khả dụng. |
| 2014 | GAN (Goodfellow) | Mật độ ẩn, không có likelihood — các mẫu sắc nét đến kinh ngạc. |
| 2015 | DRAW, PixelCNN | Tạo hình ảnh tuần tự. |
| 2017 | Glow, RealNVP | Flows khả nghịch; likelihood chính xác với độ sâu. |
| 2017 | Progressive GAN | Những khuôn mặt megapixel đầu tiên. |
| 2019 | StyleGAN / StyleGAN2 | Khuôn mặt chân thực vẫn khó bị đánh bại trong miền đó. |
| 2020 | DDPM (Ho) | Diffusion trở nên thực tế. |
| 2021 | CLIP, DALL-E 1, VQGAN | Text-to-image trở nên phổ biến. |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Latent diffusion + text conditioning = hàng hóa phổ thông. |
| 2022 | ControlNet, LoRA | Kiểm soát tinh vi trên diffusion đã được huấn luyện trước. |
| 2023 | SDXL, Midjourney v5, Flow matching | Quy mô + động lực huấn luyện tốt hơn. |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | Video diffusion; flow matching chiến thắng. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | Video cấp độ sản xuất. |
| 2026 | Consistency + Rectified Flow | Sampling một bước từ xương sống diffusion. |

## The five-question triage

Khi một bài báo về mô hình generative mới xuất hiện, hãy trả lời năm câu hỏi này trước khi đọc phần phương pháp.

1. **Cái gì đang được mô hình hóa?** Pixels, latents, discrete tokens, 3D Gaussians, meshes, waveforms?
2. **Mật độ là tường minh hay ẩn?** Họ có viết ra `log p(x)` không?
3. **Sampling: một lần hay lặp lại?** Lặp lại nghĩa là inference chậm hơn; một lần thường nghĩa là adversarial hoặc distilled.
4. **Conditioning: không điều kiện, theo lớp, văn bản, hình ảnh, tư thế?** Điều này quyết định loss và kiến trúc khung.
5. **Đánh giá: FID, CLIP score, IS, đánh giá của con người, độ chính xác tác vụ?** Mỗi loại đều có các chế độ lỗi đã biết (xem Bài 14).

Bạn sẽ trả lời lại năm câu hỏi này cho mỗi bài học trong giai đoạn này. Đến cuối cùng, chúng sẽ trở thành phản xạ.

```figure
autoencoder-bottleneck
```

## Build It

Mã cho bài học này là một công cụ trực quan hóa nhẹ: khớp một mixture-of-Gaussians 1-D từ các mẫu bằng ba phương pháp đồ chơi (kernel density, discrete histogram, và một bộ tạo "giống GAN" dựa trên mẫu gần nhất) để bạn có thể thấy sự khác biệt giữa mật độ tường minh và ẩn trên một vấn đề mà bạn có thể in ra trên một màn hình.

Chạy `code/main.py`. Nó rút 2000 mẫu từ một mixture Gaussian hai chế độ, sau đó in ra:

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

Lưu ý: hai cái đầu tiên cho phép bạn hỏi "điểm này có khả năng xảy ra bao nhiêu?". Cái thứ ba thì không. Đây là sự khác biệt giữa *tường minh và ẩn* sẽ quan trọng cho mọi bài học tương lai.

## Use It

Nhóm nào, cho tác vụ nào, vào năm 2026?

| Tác vụ | Nhóm tốt nhất | Tại sao |
|------|-------------|-----|
| Khuôn mặt chân thực, miền hẹp | StyleGAN 2/3 | Vẫn sắc nét nhất, inference nhanh nhất. |
| Text-to-image tổng quát | Latent diffusion + flow matching | SD3, Flux.1, DALL-E 3. |
| Text-to-image nhanh | Rectified flow + distillation | SDXL-Turbo, SD3-Turbo, LCM. |
| Text-to-video | Diffusion Transformer + flow matching | Sora, Veo 2, Kling. |
| Speech + music | Token-based AR hoặc flow matching | Token rời rạc mở rộng quy mô rẻ. |
| 3D scenes | Gaussian Splatting fit, diffusion prior | 3D-GS để tái tạo, diffusion cho góc nhìn mới. |
| Ước tính mật độ (không sampling) | Flows | Nhóm duy nhất có `log p(x)` chính xác. |
| Mô phỏng / vật lý | Flow matching, score SDE | Đường đi thẳng, trường vector mượt mà. |

## Ship It

Lưu dưới dạng `outputs/skill-model-chooser.md`.

Kỹ năng này lấy mô tả tác vụ và xuất ra: (1) nhóm nào nên dùng, (2) danh sách xếp hạng ba tùy chọn mã nguồn mở và ba tùy chọn thương mại, (3) chế độ lỗi tiềm ẩn bạn nên theo dõi, và (4) ngân sách tính toán/thời gian.

## Exercises

1. **Dễ.** Với mỗi sản phẩm sau, hãy xác định nhóm và xương sống: ChatGPT image, Midjourney v7, Sora, Runway Gen-3, ElevenLabs. Bằng chứng phải từ các báo cáo kỹ thuật công khai.
2. **Trung bình.** Bài báo bạn sắp đọc vào ngày mai tuyên bố sampling nhanh hơn 100 lần so với diffusion. Hãy viết ra ba câu hỏi để kiểm tra xem tốc độ đó có duy trì được khi có conditioning và độ phân giải cao hay không.
3. **Khó.** Chọn một lĩnh vực bạn quan tâm (ví dụ: cấu trúc protein, CAD, phân tử, quỹ đạo). Trả lời năm câu hỏi triage cho mô hình SOTA hiện tại trong lĩnh vực đó và phác thảo những gì một mô hình tốt hơn sẽ thay đổi.

## Key Terms

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Generative model | "Nó tạo ra thứ mới" | Học một bộ lấy mẫu cho `p_data(x)`, tùy chọn hiển thị `log p(x)`. |
| Explicit density | "Bạn có thể đánh giá nó" | Mô hình cung cấp một `log p(x)` dạng đóng hoặc có thể tính toán. |
| Implicit density | "Kiểu GAN" | Chỉ là bộ lấy mẫu — không có cách nào đánh giá `p(x)` của một điểm cho trước. |
| ELBO | "Chặn dưới bằng chứng" | Một chặn dưới có thể tính toán được của `log p(x)`; VAEs và diffusion tối ưu hóa nó. |
| Score | "Gradient của log-density" | `∇_x log p(x)`; các mô hình diffusion và SDE học trường này. |
| Manifold hypothesis | "Dữ liệu sống trên một bề mặt" | Dữ liệu chiều cao tập trung trên một đa tạp chiều thấp; lý do giảm chiều dữ liệu hoạt động. |
| Autoregressive | "Dự đoán phần tiếp theo" | Phân tích joint thành tích của các điều kiện. |
| Latent | "Mã nén" | Biểu diễn chiều thấp mà từ đó decoder có thể tái tạo đầu vào. |

## Production note: năm nhóm, năm hình thái inference

Mỗi nhóm ánh xạ tới một đường cong chi phí server-inference khác nhau. Tài liệu về production-inference coi LLM inference là prefill + decode; sự phân tách tương tự cũng áp dụng ở đây:

- **Autoregressive (nhóm 1 và 5).** Decode tuần tự chiếm ưu thế về độ trễ; KV-cache, continuous batching, và speculative decoding đều áp dụng trực tiếp.
- **VAE / diffusion / flow-matching (nhóm 2 và 4).** Không có decode theo nghĩa LLM. Chi phí = `num_steps × step_cost`, và `step_cost` là một forward pass của transformer hoặc U-Net ở độ phân giải latent đầy đủ. Các núm điều chỉnh sản xuất là số bước (DDIM / DPM-Solver / distillation), batch size, và độ chính xác (bf16 / fp8 / int4).
- **GAN (nhóm 3).** Một lần forward pass. Không có lịch trình, không có KV-cache. TTFT ≈ độ trễ tổng thể. Đây là lý do tại sao StyleGAN vẫn thắng về UX miền hẹp.

Khi bạn thấy "nhanh hơn diffusion" trong tóm tắt bài báo, hãy dịch nó thành "ít bước hơn × chi phí mỗi bước như cũ" hoặc "số bước như cũ × chi phí mỗi bước rẻ hơn". Mọi thứ khác chỉ là marketing.

## Further Reading

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — bài báo về GAN.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — bài báo về VAE.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — bài báo về DDPM.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — diffusion dưới dạng SDE.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — bài báo về flow matching.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3.