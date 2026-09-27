# Video Generation

> Một hình ảnh là một tensor 2-D. Một video là một tensor 3-D. Lý thuyết thì giống nhau; nhưng khối lượng tính toán lớn hơn gấp 10-100 lần. Sora của OpenAI (tháng 2 năm 2024) đã chứng minh điều này là khả thi. Đến năm 2026, Veo 2, Kling 1.5, Runway Gen-3, Pika 2.0 và WAN 2.2 đã sản xuất video từ văn bản ở độ phân giải 1080p — và hệ sinh thái mã nguồn mở (CogVideoX, HunyuanVideo, Mochi-1, WAN 2.2) đang chậm hơn 12 tháng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)
**Time:** ~45 phút

## Vấn đề

Một video 10 giây ở độ phân giải 1080p với 24fps bao gồm 240 khung hình, mỗi khung hình có kích thước 1920×1080×3 pixel. Đó là khoảng 1.5 GB dữ liệu thô cho mỗi clip. Diffusion trong không gian pixel là không khả thi. Bạn cần:

1. **Nén không gian-thời gian (Spatiotemporal compression).** Một VAE mã hóa video, thay vì từng khung hình đơn lẻ, thành một chuỗi các patch không gian-thời gian.
2. **Tính nhất quán theo thời gian (Temporal coherence).** Các khung hình cần chia sẻ nội dung, ánh sáng và danh tính đối tượng trong suốt nhiều giây. Mạng phải mô hình hóa được chuyển động.
3. **Ngân sách tính toán.** Huấn luyện video tốn kém hơn 10-100 lần so với hình ảnh cho cùng một kích thước mô hình.
4. **Conditioning.** Văn bản, hình ảnh (khung hình đầu tiên), âm thanh hoặc một video khác. Hầu hết các mô hình thương mại đều chấp nhận cả bốn loại này.

Kiến trúc giải quyết vấn đề này là **Diffusion Transformer (DiT)** áp dụng trên các patch không gian-thời gian, được huấn luyện trên các tập dữ liệu khổng lồ (prompt, caption, video). Hàm loss diffusion tương tự như Bài 06.

## Khái niệm

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### Patchify

Mã hóa video bằng một 3D VAE (nén không gian-thời gian đã học). Latent có hình dạng `[T_latent, H_latent, W_latent, C_latent]`. Chia thành các patch có kích thước `[t_p, h_p, w_p]`. Đối với các mô hình kiểu Sora, `t_p = 1` (patch theo từng khung hình) hoặc `t_p = 2` (mỗi hai khung hình). Một video 10 giây 1080p nén thành khoảng 20,000-100,000 patch.

### Spatiotemporal DiT

Một transformer xử lý chuỗi patch phẳng. Mỗi patch có một positional embedding 3D (thời gian + y + x). Attention thường được phân tách (factorized):

- **Spatial attention** trong các patch của mỗi khung hình.
- **Temporal attention** giữa các khung hình tại cùng một vị trí không gian.
- **Full 3D attention** đắt đỏ hơn gấp 16-100 lần; chỉ được sử dụng ở độ phân giải thấp hoặc trong nghiên cứu.

### Text conditioning

Cross-attention với một text encoder lớn (T5-XXL cho Sora, CogVideoX-5B sử dụng T5-XXL). Các prompt dài rất quan trọng — tập huấn luyện của Sora có các re-caption dày đặc do GPT tạo ra, trung bình 200 token mỗi clip.

### Huấn luyện

Hàm loss diffusion tiêu chuẩn (dự đoán ε hoặc v) trên các latent không gian-thời gian. Dữ liệu: video web + ~100 triệu clip được chọn lọc + chú thích văn bản tổng hợp. Tính toán: 10,000+ giờ GPU cho một lần chạy nghiên cứu nhỏ; quy mô Sora là 100,000+.

## Bối cảnh sản xuất năm 2026

| Mô hình | Ngày | Thời lượng tối đa | Độ phân giải tối đa | Mã nguồn mở? | Ghi chú |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | Không | Mô hình đầu tiên thể hiện khả năng mô phỏng thế giới ở quy mô lớn |
| Sora Turbo | 2024-12 | 20s | 1080p | Không | Sora thương mại với tốc độ inference nhanh gấp 5 lần |
| Veo 2 (Google) | 2024-12 | 8s | 4K | Không | Chất lượng cao nhất + vật lý trong năm 2025 |
| Veo 3 | 2025 Q3 | 15s | 4K | Không | Âm thanh gốc và kiểm soát camera mạnh mẽ hơn |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | Không | Chuyển động người tốt nhất trong Q1 2025 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | Không | Các công cụ video chuyên nghiệp đi kèm |
| Pika 2.0 | 2024-10 | 5s | 1080p | Không | Tính nhất quán nhân vật mạnh nhất |
| CogVideoX (THUDM) | 2024 | 10s | 720p | Có (2B, 5B) | Mô hình video mã nguồn mở quy mô 5B đầu tiên |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | Có (13B) | SOTA mã nguồn mở cuối 2024 |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | Có (10B) | Giấy phép sử dụng cởi mở nhất |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | Có | Mô hình mã nguồn mở mạnh nhất giữa 2025 |

Các mô hình mã nguồn mở đang thu hẹp khoảng cách nhanh hơn so với lĩnh vực hình ảnh: HunyuanVideo + WAN 2.2 LoRAs đã cung cấp sức mạnh cho hầu hết các quy trình làm việc mã nguồn mở vào giữa năm 2026.

```figure
video-diffusion-denoise
```

## Xây dựng

`code/main.py` mô phỏng ý tưởng cốt lõi của spatiotemporal DiT: patchify một video tổng hợp nhỏ, thêm positional embedding cho mỗi patch và khử nhiễu toàn bộ chuỗi bằng attention kiểu transformer trên các patch. Không dùng numpy; Python thuần. Chúng ta chứng minh rằng tính nhất quán theo thời gian xuất hiện ngay cả trong 1-D khi các patch của khung hình liền kề chia sẻ bộ khử nhiễu và positional embedding.

### Bước 1: patchify một "video" 1-D tổng hợp

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Bước 2: position embedding cho mỗi khung hình

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Bước 3: bộ khử nhiễu nhìn thấy toàn bộ chuỗi

Thay vì khử nhiễu từng khung hình độc lập, mạng nhỏ của chúng ta nối tất cả các giá trị khung hình + positional embedding của chúng và dự đoán nhiễu cho tất cả các khung hình cùng lúc.

### Bước 4: kiểm tra tính nhất quán theo thời gian

Sau khi huấn luyện, hãy lấy mẫu một video. Đo delta giữa các khung hình. Nếu mô hình đã học được cấu trúc thời gian, các delta sẽ nhỏ hơn so với việc lấy mẫu từng khung hình độc lập.

## Các cạm bẫy

- **Lấy mẫu độc lập từng khung hình = nhấp nháy (flicker).** Nếu bạn chạy diffusion hình ảnh trên mỗi khung hình riêng biệt, đầu ra sẽ bị nhấp nháy vì nhiễu của mỗi khung hình là độc lập. Video diffusion khắc phục điều này bằng cách liên kết các khung hình thông qua attention hoặc nhiễu chia sẻ.
- **Naive 3D attention = OOM.** Full 3D attention trên một latent 10 giây 1080p là hàng trăm tỷ phép tính. Hãy phân tách thành không gian + thời gian.
- **Chú thích dữ liệu quan trọng hơn kích thước.** Nâng cấp chính của Sora so với các công trình trước đó là huấn luyện trên các chú thích chi tiết hơn gấp ~10 lần (các clip được dán nhãn lại bởi GPT-4). Báo cáo kỹ thuật của OpenAI đã nêu rõ điều này.
- **Conditioning khung hình đầu tiên.** Hầu hết các mô hình thương mại cũng chấp nhận một hình ảnh làm khung hình đầu tiên. Đây là chế độ "image-to-video"; việc huấn luyện bao gồm cả biến thể này.
- **Trôi vật lý (Physics drift).** Các clip dài (>10s) tích tụ những điểm không nhất quán tinh vi. Việc tạo video theo cửa sổ trượt (sliding-window) + neo khung hình chính (keyframe anchoring) sẽ giúp ích.

## Sử dụng

| Trường hợp sử dụng | Lựa chọn 2026 |
|----------|-----------|
| Text-to-video chất lượng cao nhất, có hosting | Veo 3 hoặc Sora |
| Cinematic có kiểm soát camera | Runway Gen-3 với motion brushes |
| Tính nhất quán nhân vật giữa các clip | Pika 2.0 hoặc Kling 2.1 |
| Mã nguồn mở, fine-tune nhanh | WAN 2.2 + LoRA |
| Image-to-video | WAN 2.2-I2V, Kling 2.1 I2V, hoặc Runway |
| Audio-to-video lip sync | Veo 3 (âm thanh gốc) hoặc mô hình lip-sync chuyên dụng |
| Chỉnh sửa video | Runway Act-Two, Kling Motion Brush, Flux-Kontext (still-frame) |

Chi phí cho mỗi giây video với chất lượng tương đương đã giảm 20 lần từ năm 2024 đến 2026.

## Triển khai

Lưu `outputs/skill-video-brief.md`. Kỹ năng này lấy một bản tóm tắt video (thời lượng, tỷ lệ khung hình, phong cách, kế hoạch camera, tính nhất quán của chủ thể, âm thanh) và xuất ra: mô hình + hosting, cấu trúc prompt (ngôn ngữ camera, mô tả chủ thể, mô tả chuyển động), seed + giao thức tái lập, và danh sách kiểm tra QA cấp khung hình.

## Bài tập

1. **Dễ.** Trong `code/main.py`, so sánh delta giữa các khung hình cho (a) lấy mẫu độc lập từng khung hình, (b) lấy mẫu chuỗi chung. Báo cáo giá trị trung bình và phương sai của các delta.
2. **Trung bình.** Thêm điều kiện khung hình đầu tiên: ghim khung hình 0 vào một giá trị nhất định và lấy mẫu phần còn lại. Đo lường cách giá trị được ghim lan truyền như thế nào.
3. **Khó.** Sử dụng HuggingFace diffusers để chạy CogVideoX-2B trên GPU cục bộ. Đo thời gian 20 bước inference ở độ phân giải 720p cho một clip 6 giây. Profile spatiotemporal attention để xác định nút thắt cổ chai.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Video VAE | "3-D VAE" | Encoder nén `(T, H, W, C)` → latent không gian-thời gian. |
| Patches | "Các token" | Các khối 3-D kích thước cố định của latent; đầu vào cho DiT. |
| Factorized attention | "Không gian + thời gian" | Chạy attention trên không gian, sau đó trên thời gian; bỏ qua full 3D attention. |
| Image-to-video (I2V) | "Làm hoạt hình ảnh này" | Mô hình lấy ảnh + văn bản, xuất ra video bắt đầu từ ảnh đó. |
| Keyframe conditioning | "Khung hình neo" | Ghim các khung hình cụ thể để kiểm soát vòng cung của video. |
| Motion brush | "Gợi ý hướng" | Đầu vào UI nơi người dùng vẽ các vector chuyển động lên ảnh. |
| Re-captioning | "Chú thích dày đặc" | Sử dụng LLM để dán nhãn lại các clip huấn luyện với prompt chi tiết. |
| Flicker | "Artifact thời gian" | Sự không nhất quán giữa các khung hình; được sửa bằng coupled denoising. |

## Ghi chú sản xuất: latent video là vấn đề về băng thông bộ nhớ

Một clip 10 giây 1080p ở 24 fps là 240 khung hình × 1920 × 1080 × 3 ≈ 1.5 GB pixel thô. Sau khi nén 4× bằng video VAE (`2 × spatial × 2 × temporal`), latent là ~100 MB mỗi yêu cầu. Chạy qua spatiotemporal DiT trong 30 bước với batch 1, bạn đang di chuyển ~3 GB/bước qua HBM — băng thông bộ nhớ, không phải FLOPs, mới là nút thắt cổ chai.

Ba núm điều chỉnh sản xuất, tất cả đều từ tài liệu inference sản xuất:

- **TP trên DiT.** Các mô hình text-to-video thường có ≥10B tham số. TP=4 trên 4 H100 là tiêu chuẩn; PP=2 × TP=2 cho các mô hình lớp 405B. Độ trễ mỗi bước giảm gần như tuyến tính với TP cho đến khi chạm giới hạn all-reduce.
- **Frame batching = continuous batching.** Tại thời điểm tạo, video về mặt khái niệm là một batch các khung hình được liên kết bởi attention. Continuous batching (lập lịch in-flight) được áp dụng: bắt đầu render khung hình `t+1` trong khi khung hình `t-1` đang được trả về, nếu kiến trúc mô hình cho phép tạo theo cửa sổ trượt.
- **Clip-level prefill cache.** Đối với image-to-video, điều kiện khung hình đầu tiên tương tự như prefill prompt của LLM: tính toán một lần, tái sử dụng qua các lần truyền decoder thời gian. Đây thực chất là một KV-cache cho video.

## Đọc thêm

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Báo cáo kỹ thuật Sora.
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX.
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo.
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1.
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — SOTA mở giữa 2025.
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — bài báo nền tảng về video diffusion.
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Tổ tiên của Stable Video Diffusion.