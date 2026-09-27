# Inpainting, Outpainting & Image Editing

> Text-to-image tạo ra những thứ mới. Inpainting sửa chữa những thứ cũ. Trong thực tế sản xuất, 70% công việc xử lý hình ảnh có tính phí là chỉnh sửa — thay đổi nền, xóa logo, mở rộng khung hình, tái tạo bàn tay. Inpainting chính là nơi mà diffusion chứng minh giá trị thực tiễn của nó.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**Time:** ~75 phút

## Vấn đề

Một khách hàng gửi cho bạn một bức ảnh sản phẩm hoàn hảo nhưng lại có một biển báo gây mất tập trung ở phía sau. Bạn muốn xóa biển báo đó mà vẫn giữ nguyên mọi pixel khác. Bạn không thể chạy text-to-image từ đầu — kết quả sẽ có màu sắc, ánh sáng và góc độ sản phẩm khác biệt. Bạn muốn tái tạo *chỉ* vùng được mask, và muốn quá trình tái tạo đó tôn trọng bối cảnh xung quanh.

Đó chính là inpainting. Các biến thể:

- **Inpainting.** Tái tạo bên trong một mask, giữ nguyên các pixel bên ngoài.
- **Outpainting.** Tái tạo bên ngoài một mask (hoặc vượt ra ngoài khung hình), giữ nguyên bên trong.
- **Image editing.** Tái tạo toàn bộ hình ảnh nhưng vẫn giữ được độ trung thực về ngữ nghĩa hoặc cấu trúc so với bản gốc (SDEdit, InstructPix2Pix).

Mọi pipeline diffusion vào năm 2026 đều tích hợp chế độ inpainting. Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit. Chúng đều hoạt động dựa trên cùng một nguyên lý.

## Khái niệm

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### Cách tiếp cận ngây thơ (và tại sao nó sai)

Chạy text-to-image tiêu chuẩn với một mask. Tại mỗi bước lấy mẫu (sampling step), thay thế vùng không được mask của latent nhiễu bằng hình ảnh sạch đã được forward-diffused. Nó hoạt động... nhưng rất tệ. Các lỗi biên (boundary artifacts) sẽ xuất hiện vì mô hình không có thông tin về những gì nằm trong vùng được mask.

### Mô hình inpainting chuẩn

Huấn luyện một U-Net đã sửa đổi để nhận 9 kênh đầu vào thay vì 4:

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

Các kênh bổ sung là bản sao của hình ảnh nguồn đã được VAE-encoded cộng với một mask đơn kênh. Trong quá trình huấn luyện, bạn mask ngẫu nhiên các vùng của hình ảnh và huấn luyện mô hình khử nhiễu chỉ vùng được mask, trong khi vùng không được mask được cung cấp như một tín hiệu điều kiện (conditioning signal) sạch. Khi suy luận (inference), mô hình có thể "nhìn thấy" những gì bao quanh vùng được mask và tạo ra các phần hoàn thiện mạch lạc.

SD-Inpaint, SDXL-Inpaint, Flux-Fill đều sử dụng đầu vào 9 kênh (hoặc tương tự) này. Diffusers `StableDiffusionInpaintPipeline`, `FluxFillPipeline`.

### SDEdit (Meng et al., 2022) — chỉnh sửa tự do

Thêm nhiễu vào hình ảnh nguồn đến một mức `t` trung gian, sau đó chạy chuỗi ngược từ `t` về 0 với một prompt mới. Không cần huấn luyện lại. Việc chọn `t` bắt đầu sẽ đánh đổi giữa độ trung thực và sự tự do sáng tạo:

- `t/T = 0.3` → gần như giống hệt nguồn, thay đổi phong cách nhỏ.
- `t/T = 0.6` → chỉnh sửa vừa phải, bảo toàn cấu trúc thô.
- `t/T = 0.9` → tạo từ trạng thái gần như nhiễu, bảo toàn nguồn tối thiểu.

### InstructPix2Pix (Brooks et al., 2023)

Fine-tune một mô hình diffusion trên các bộ ba `(input_image, instruction, output_image)`. Khi suy luận, điều kiện hóa trên cả hình ảnh đầu vào và hướng dẫn bằng văn bản ("làm cho nó thành hoàng hôn", "thêm một con rồng"). Hai thang đo CFG: thang đo hình ảnh và thang đo văn bản.

### RePaint (Lugmayr et al., 2022)

Giữ một mô hình diffusion không điều kiện tiêu chuẩn. Tại mỗi bước ngược, thực hiện lấy mẫu lại (resample) — thỉnh thoảng nhảy ngược lại trạng thái nhiễu hơn và tái tạo lại. Tránh được các lỗi biên. Được sử dụng khi bạn không có mô hình inpainting đã huấn luyện.

```figure
inpaint-mask-reinject
```

## Xây dựng

`code/main.py` triển khai một lược đồ inpainting 1-D đơn giản trên dữ liệu 5 chiều. Chúng ta huấn luyện một DDPM trên dữ liệu hỗn hợp 5-D, nơi mỗi mẫu là 5 số thực từ một trong hai cụm. Khi suy luận, chúng ta "mask" 2 trong số 5 chiều, đưa phiên bản nhiễu-forward của ba chiều không được mask vào mỗi bước, và chỉ tái tạo các chiều được mask.

### Bước 1: Dữ liệu 5-D DDPM

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Bước 2: Huấn luyện bộ khử nhiễu trên cả 5 chiều

DDPM tiêu chuẩn. Mạng xuất ra dự đoán nhiễu 5-D cho đầu vào nhiễu 5-D.

### Bước 3: Suy luận với mask-aware reverse

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

Đây là cách tiếp cận ngây thơ và nó hoạt động trên dữ liệu 1-D đơn giản. Inpainting hình ảnh thực tế sử dụng đầu vào 9 kênh vì sự mạch lạc của kết cấu (texture coherence) quan trọng hơn.

### Bước 4: Outpainting

Outpainting là inpainting với mask bị đảo ngược: mask phần khung hình mới (trước đây không tồn tại), điền phần còn lại bằng bản gốc. Mục tiêu huấn luyện giống hệt nhau.

## Các cạm bẫy

- **Đường nối (Seams).** Cách tiếp cận ngây thơ để lại các ranh giới có thể nhìn thấy vì thông tin gradient không chảy qua mask. Khắc phục: làm giãn (dilate) mask thêm 8-16 pixel, hoặc sử dụng mô hình inpainting chuẩn.
- **Rò rỉ mask (Mask leakage).** Nếu vùng không được mask của hình ảnh điều kiện có chất lượng thấp hoặc nhiễu, nó sẽ làm ô nhiễm quá trình tạo bên trong mask. Hãy khử nhiễu hoặc làm mờ nhẹ.
- **CFG tương tác với kích thước mask.** CFG cao trên một mask nhỏ = vùng bị bão hòa. Giảm CFG cho các chỉnh sửa nhỏ.
- **SDEdit fidelity cliff.** Chuyển từ `t/T = 0.5` sang `t/T = 0.6` có thể làm mất danh tính của chủ thể. Hãy quét (sweep) và lưu checkpoint.
- **Prompt không khớp.** Prompt nên mô tả *toàn bộ* hình ảnh, không chỉ nội dung mới. "Một con mèo ngồi trên ghế" thay vì chỉ "một con mèo".

## Sử dụng

| Tác vụ | Pipeline |
|------|----------|
| Xóa vật thể, mask nhỏ | SD-Inpaint hoặc Flux-Fill, prompt tiêu chuẩn |
| Thay bầu trời | SD-Inpaint + "bầu trời xanh lúc hoàng hôn" |
| Mở rộng khung hình | Chế độ outpaint của SDXL (8px feather) hoặc Flux-Fill với mask outpaint |
| Tái tạo tay / mặt | SD-Inpaint với prompt mô tả lại chủ thể + ControlNet-Openpose |
| Thay đổi phong cách một vùng | SDEdit tại `t/T=0.5` trên vùng được mask |
| "Làm cho nó thành hoàng hôn" | InstructPix2Pix hoặc Flux-Kontext |
| Thay nền | SAM mask → SD-Inpaint |
| Độ trung thực siêu cao | Flux-Fill hoặc GPT-Image (hosted) cho các trường hợp khó nhất |

SAM (Segment Anything của Meta, 2023) + diffusion inpaint là pipeline xóa nền của năm 2026. SAM 2 (2024) hoạt động trên video.

## Triển khai

Lưu `outputs/skill-editing-pipeline.md`. Kỹ năng này nhận một hình ảnh gốc + mô tả chỉnh sửa + mask tùy chọn (hoặc SAM prompt) và xuất ra: phương pháp tạo mask, mô hình cơ sở, các thang đo CFG (hình ảnh + văn bản), chế độ SDEdit-t hoặc inpainting, và danh sách kiểm tra QA.

## Bài tập

1. **Dễ.** Trong `code/main.py`, thay đổi tỷ lệ các chiều được mask từ 0.2 đến 0.8. Tại tỷ lệ nào thì chất lượng inpaint (phần dư trong các chiều được mask) bằng với tạo ảnh không điều kiện?
2. **Trung bình.** Triển khai RePaint: tại mỗi bước ngược thứ 10, nhảy ngược lại 5 bước (thêm nhiễu) và khử nhiễu lại. Đo lường xem nó có làm giảm phần dư ranh giới tại cạnh mask hay không.
3. **Khó.** Sử dụng Hugging Face diffusers để so sánh: SD 1.5 Inpaint + ControlNet-Openpose so với Flux.1-Fill trên 20 tác vụ tái tạo khuôn mặt. Chấm điểm riêng biệt về độ bám sát tư thế và bảo toàn danh tính.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Inpainting | "Lấp đầy lỗ hổng" | Tái tạo bên trong mask; giữ nguyên pixel bên ngoài. |
| Outpainting | "Mở rộng khung hình" | Tái tạo bên ngoài khung hình; giữ nguyên bên trong. |
| 9-channel U-Net | "Mô hình inpainting chuẩn" | U-Net với `noisy \| encoded-source \| mask` làm đầu vào. |
| SDEdit | "Img2img với mức nhiễu" | Nhiễu đến thời điểm `t`, khử nhiễu với prompt mới. |
| InstructPix2Pix | "Chỉnh sửa chỉ bằng văn bản" | Diffusion đã fine-tune trên các bộ ba (ảnh, hướng dẫn, đầu ra). |
| RePaint | "Không cần huấn luyện lại" | Thêm nhiễu định kỳ trong quá trình ngược để giảm đường nối. |
| SAM | "Segment Anything" | Trình tạo mask bằng click hoặc hộp; kết hợp với inpaint. |
| Flux-Kontext | "Chỉnh sửa với bối cảnh" | Biến thể Flux chấp nhận ảnh tham chiếu + hướng dẫn để chỉnh sửa. |

## Lưu ý sản xuất: pipeline chỉnh sửa nhạy cảm với độ trễ

Người dùng chỉnh sửa ảnh mong đợi thời gian phản hồi dưới 5 giây. Một tác vụ SDXL-Inpaint 30 bước ở 1024² mất 3-4 giây trên L4, cộng với tạo mask SAM (~200 ms) và VAE encode/decode (~500 ms tổng cộng). Trong bối cảnh sản xuất, đây là vấn đề bị giới hạn bởi TTFT (Time To First Token) thay vì thông lượng — batch 1, độ đồng thời thấp, tối thiểu hóa mọi giai đoạn:

- **SAM-H là phần chậm nhất.** SAM-H ở 1024² mất ~200 ms; SAM-ViT-B mất ~40 ms với chất lượng giảm nhẹ. SAM 2 (video) thêm chi phí thời gian; không sử dụng cho chỉnh sửa ảnh đơn.
- **Bỏ qua encode khi có thể.** `pipe.image_processor.preprocess(img)` encode thành latents. Nếu bạn đã có latents từ lần tạo trước (thường thấy trong UI chỉnh sửa lặp lại), hãy truyền trực tiếp qua `latents=...` để bỏ qua một lần VAE encode.
- **Độ giãn mask cũng quan trọng đối với thông lượng.** Mask nhỏ nghĩa là hầu hết các bước forward của U-Net bị lãng phí (các pixel không được mask dù sao cũng bị kẹp). `diffusers`' `StableDiffusionInpaintPipeline` chạy toàn bộ U-Net bất kể mask; chỉ các biến thể inpaint chuẩn 9 kênh mới khai thác tính toán có mask.
- **Flux-Kontext là câu trả lời cho năm 2025.** Một lần forward duy nhất qua `(source_image, instruction)` — không cần mask riêng, không cần quét nhiễu SDEdit. Trên H100, nó thực hiện chỉnh sửa trong ~1.5 giây. Bài học kiến trúc: gộp các giai đoạn lại.

## Đọc thêm

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — inpainting không cần huấn luyện.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — chỉnh sửa bằng hướng dẫn văn bản.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM, nguồn tạo mask.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — video SAM.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — chỉnh sửa ở cấp độ attention.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — công cụ năm 2024.