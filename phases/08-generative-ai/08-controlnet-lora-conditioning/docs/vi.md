# ControlNet, LoRA & Conditioning

> Văn bản đơn thuần là một tín hiệu điều khiển vụng về. ControlNet cho phép bạn nhân bản một mô hình diffusion đã được huấn luyện trước và điều hướng nó bằng bản đồ chiều sâu (depth map), khung xương tư thế (pose skeleton), hình vẽ tay (scribble) hoặc ảnh cạnh (edge image). LoRA cho phép bạn tinh chỉnh (fine-tune) một mô hình 2 tỷ tham số bằng cách chỉ huấn luyện 10 triệu tham số. Kết hợp lại, chúng đã biến Stable Diffusion từ một món đồ chơi thành pipeline hình ảnh năm 2026 được sử dụng tại mọi agency.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch — cho nền tảng LoRA)
**Time:** ~75 phút

## Vấn đề

Một câu lệnh (prompt) như "một người phụ nữ mặc váy đỏ dắt chó đi dạo trên con phố đông đúc" không cung cấp cho mô hình thông tin về việc con chó đang ở *đâu*, người phụ nữ đang ở *tư thế nào*, hay *góc nhìn* của con phố ra sao. Văn bản chỉ xác định được khoảng 10% những gì bạn cần để mô tả một hình ảnh. Phần còn lại là hình ảnh và không thể mô tả hiệu quả bằng lời nói.

Việc huấn luyện một mô hình có điều kiện mới từ đầu cho mỗi tín hiệu (tư thế, chiều sâu, canny, phân đoạn) là điều không khả thi. Bạn muốn giữ nguyên backbone SDXL 2,6 tỷ tham số, gắn thêm một mạng phụ nhỏ để đọc các điều kiện và để nó điều chỉnh các đặc trưng trung gian của backbone. Đó chính là ControlNet.

Bạn cũng muốn dạy mô hình các khái niệm mới (khuôn mặt của bạn, sản phẩm của bạn, phong cách của bạn) mà không cần huấn luyện lại toàn bộ mô hình. Bạn muốn một delta nhỏ hơn 100 lần. Đó chính là LoRA — các bộ điều hợp hạng thấp (low-rank adapters) cắm vào các trọng số attention hiện có.

ControlNet + LoRA + văn bản = bộ công cụ của người làm nghề năm 2026. Hầu hết các pipeline hình ảnh trong sản xuất đều xếp chồng 2-5 LoRA, 1-3 ControlNet và một IP-Adapter lên trên nền tảng SDXL / SD3 / Flux.

## Khái niệm

![ControlNet clones the encoder; LoRA adds low-rank deltas](../assets/controlnet-lora.svg)

### ControlNet (Zhang và cộng sự, 2023)

Lấy một mô hình SD đã được huấn luyện trước. *Nhân bản* nửa encoder của U-Net. Đóng băng mô hình gốc. Huấn luyện bản sao để chấp nhận thêm một đầu vào điều kiện (cạnh, chiều sâu, tư thế). Kết nối bản sao trở lại nửa decoder của mô hình gốc bằng các kết nối tắt *zero-convolution* (các phép tích chập 1×1 được khởi tạo bằng 0 — bắt đầu như một phép toán không làm gì cả, sau đó học một delta).

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Việc khởi tạo zero-conv có nghĩa là ControlNet bắt đầu như một hàm đồng nhất (identity) — không gây hại ngay cả trước khi huấn luyện. Huấn luyện trên 1 triệu bộ ba (prompt, điều kiện, hình ảnh) với hàm mất mát diffusion tiêu chuẩn.

Các ControlNet theo từng phương thức được phát hành dưới dạng các mô hình phụ nhỏ (~360 triệu tham số cho SDXL, ~70 triệu cho SD 1.5). Bạn có thể kết hợp chúng khi suy luận (inference):

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu và cộng sự, 2021)

Đối với bất kỳ lớp tuyến tính `W ∈ R^{d×d}` nào trong mô hình, hãy đóng băng `W` và thêm một delta hạng thấp:

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

với `r << d`. Hạng (rank) 4-16 là tiêu chuẩn cho attention, hạng 64-128 cho các tinh chỉnh chuyên sâu. Số lượng tham số mới: `2 · d · r` thay vì `d²`. Đối với attention của SDXL với `d=640`, `r=16`: 20 nghìn tham số mỗi adapter thay vì 410 nghìn — giảm 20 lần. Trên toàn bộ mô hình: một LoRA thường có dung lượng 20-200MB so với 5GB của mô hình gốc.

Khi suy luận, bạn có thể điều chỉnh tỷ lệ của LoRA: `W' = W + α · B @ A`. `α = 0.5-1.5` là mức bình thường. Nhiều LoRA có thể xếp chồng lên nhau theo kiểu cộng dồn (với lưu ý thông thường là chúng tương tác theo những cách phi tuyến tính).

### IP-Adapter (Ye và cộng sự, 2023)

Một bộ điều hợp nhỏ chấp nhận một *hình ảnh* làm điều kiện (bên cạnh văn bản). Sử dụng bộ mã hóa hình ảnh CLIP để tạo ra các token hình ảnh, đưa chúng vào cross-attention cùng với các token văn bản. ~20MB mỗi mô hình cơ sở. Cho phép bạn thực hiện "tạo hình ảnh theo phong cách của hình ảnh tham chiếu này" mà không cần LoRA.

## Ma trận khả năng kết hợp

| Công cụ | Kiểm soát cái gì | Kích thước | Khi nào nên dùng |
|------|------------------|------|-------------|
| ControlNet | Cấu trúc không gian (tư thế, chiều sâu, cạnh) | 70-360MB | Bố cục chính xác, thành phần |
| LoRA | Phong cách, chủ thể, khái niệm | 20-200MB | Cá nhân hóa, phong cách |
| IP-Adapter | Phong cách hoặc chủ thể từ ảnh tham chiếu | 20MB | Không văn bản nào mô tả được vẻ ngoài |
| Textual Inversion | Khái niệm đơn lẻ dưới dạng token mới | 10KB | Cũ, phần lớn đã được thay thế bởi LoRA |
| DreamBooth | Tinh chỉnh toàn bộ trên một chủ thể | 2-5GB | Nhận diện mạnh, cần tính toán cao |
| T2I-Adapter | Giải pháp thay thế ControlNet nhẹ hơn | 70MB | Thiết bị biên, ngân sách suy luận |

ControlNet ≈ không gian. LoRA ≈ ngữ nghĩa. Hãy sử dụng cả hai.

```figure
v4-controlnet-zero
```

## Xây dựng

`code/main.py` mô phỏng hai cơ chế trên 1-D:

1. **LoRA.** Một lớp tuyến tính đã được huấn luyện trước `W`. Đóng băng nó. Huấn luyện một ma trận hạng thấp `B @ A` sao cho `W + BA` khớp với một lớp tuyến tính mục tiêu. Cho thấy rằng `r = 1` là đủ để học một hiệu chỉnh hạng 1 một cách hoàn hảo.

2. **ControlNet-lite.** Một bộ dự đoán "nền tảng đóng băng" và một "mạng phụ" đọc thêm một tín hiệu bổ sung. Đầu ra của mạng phụ được điều khiển bởi một đại lượng vô hướng có thể học được, khởi tạo bằng 0 (phiên bản zero-conv của chúng ta). Huấn luyện và quan sát cổng (gate) tăng dần lên.

### Bước 1: Toán học LoRA

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Bước 2: Mạng phụ khởi tạo bằng 0

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

Tại bước 0, đầu ra giống hệt với mô hình cơ sở. Các cập nhật huấn luyện sớm `gate` diễn ra chậm — không gây ra sự trôi dạt thảm khốc (catastrophic drift).

## Các cạm bẫy

- **Quá tải LoRA.** `α = 2` hoặc `α = 3` là một thủ thuật "làm cho nó mạnh hơn" phổ biến nhưng thường tạo ra các kết quả bị quá đà/lỗi. Hãy giữ `α ≤ 1.5`.
- **Xung đột trọng số ControlNet.** Sử dụng một Pose ControlNet ở trọng số 1.0 và một Depth ControlNet ở trọng số 1.0 thường gây ra kết quả quá mức. Tổng trọng số ≈ 1.0 là một mặc định an toàn.
- **LoRA trên nền tảng sai.** Các LoRA của SDXL sẽ âm thầm không hoạt động trên SD 1.5 vì kích thước attention không khớp. Diffusers sẽ cảnh báo trong phiên bản 0.30+.
- **Trôi dạt Textual Inversion.** Các token được huấn luyện trên một checkpoint sẽ trôi dạt rất tệ trên checkpoint khác. LoRA có tính di động cao hơn.
- **Gộp trọng số và lưu trữ LoRA.** Bạn có thể nướng (bake) một LoRA vào trọng số mô hình cơ sở để suy luận nhanh hơn (không cần cộng thêm khi chạy), nhưng bạn sẽ mất khả năng điều chỉnh tỷ lệ `α` khi đang chạy. Hãy giữ cả hai phiên bản.

## Sử dụng

| Mục tiêu | Pipeline năm 2026 |
|------|---------------|
| Tái tạo phong cách nghệ thuật của thương hiệu | LoRA huấn luyện trên ~30 ảnh được chọn lọc ở hạng 32 |
| Đưa khuôn mặt của tôi vào ảnh tạo ra | DreamBooth hoặc LoRA + IP-Adapter-FaceID |
| Tư thế cụ thể + prompt | ControlNet-Openpose + SDXL + văn bản |
| Bố cục nhận biết chiều sâu | ControlNet-Depth + SD3 |
| Tham chiếu + prompt | IP-Adapter + văn bản |
| Bố cục chính xác | ControlNet-Scribble hoặc ControlNet-Canny |
| Thay thế nền | ControlNet-Seg + Inpainting (Bài 09) |
| Phong cách 1 bước nhanh | LCM-LoRA trên SDXL-Turbo |

## Triển khai

Lưu `outputs/skill-sd-toolkit-composer.md`. Kỹ năng bao gồm việc thực hiện một tác vụ (tài sản đầu vào: prompt, ảnh tham chiếu tùy chọn, tư thế tùy chọn, chiều sâu tùy chọn, hình vẽ tùy chọn) và xuất ra bộ công cụ, trọng số và giao thức hạt giống (seed) có thể tái lập.

## Bài tập

1. **Dễ.** Trong `code/main.py`, thay đổi hạng LoRA `r` từ 1 đến 4. Tại hạng nào thì LoRA khớp chính xác với một delta mục tiêu hạng 2?
2. **Trung bình.** Huấn luyện hai LoRA riêng biệt trên hai phép biến đổi mục tiêu. Tải chúng cùng nhau và cho thấy sự tương tác cộng dồn của chúng. Khi nào sự tương tác đó phá vỡ tính tuyến tính?
3. **Khó.** Sử dụng diffusers để xếp chồng: SDXL-base + Canny-ControlNet (trọng số 0.8) + một style LoRA (α 0.8) + IP-Adapter (trọng số 0.6). Đo lường sự đánh đổi giữa FID và độ bám sát prompt khi các trọng số chồng thay đổi.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| ControlNet | "Kiểm soát không gian" | Encoder nhân bản + các kết nối tắt zero-conv; đọc ảnh điều kiện. |
| Zero convolution | "Bắt đầu như hàm đồng nhất" | Phép tích chập 1×1 khởi tạo bằng 0; ControlNet bắt đầu như một phép toán không làm gì. |
| LoRA | "Bộ điều hợp hạng thấp" | `W + B @ A`, `r << d`; ít tham số hơn 100 lần so với tinh chỉnh toàn bộ. |
| rank r | "Núm xoay" | Nén LoRA; 4-16 là điển hình, 64+ cho cá nhân hóa chuyên sâu. |
| α | "Độ mạnh LoRA" | Điều chỉnh tỷ lệ của delta LoRA khi chạy. |
| IP-Adapter | "Ảnh tham chiếu" | Bộ điều hợp điều kiện hình ảnh nhỏ thông qua các token CLIP-image. |
| DreamBooth | "Tinh chỉnh chủ thể toàn bộ" | Huấn luyện toàn bộ mô hình trên ~30 ảnh của một chủ thể. |
| Textual Inversion | "Token mới" | Chỉ học một embedding từ mới; cũ, phần lớn đã được thay thế. |

## Ghi chú sản xuất: Hoán đổi LoRA, làn đường ControlNet, phục vụ đa người thuê (multi-tenant)

Một SaaS text-to-image thực tế phục vụ hàng trăm LoRA và hàng chục ControlNet trên cùng một checkpoint cơ sở. Vấn đề phục vụ này rất giống với đa người thuê LLM (tài liệu sản xuất bao gồm trường hợp LLM dưới dạng continuous batching và LoRAX / S-LoRA):

- **Hoán đổi nóng (Hot-swap) LoRA, đừng gộp.** Gộp `W' = W + α·B·A` vào mô hình cơ sở giúp suy luận nhanh hơn ~3-5% mỗi bước nhưng làm đóng băng `α` và mô hình cơ sở. Hãy giữ các LoRA nóng trong VRAM dưới dạng các delta hạng r; diffusers cung cấp `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` để kích hoạt theo yêu cầu. Chi phí hoán đổi là các trọng số `2 · d · r · num_layers` — quy mô MB, dưới một giây.
- **ControlNet như một làn đường attention thứ hai.** Encoder nhân bản chạy song song với mô hình cơ sở. Hai ControlNet ở trọng số 1.0 mỗi cái = hai lần forward pass bổ sung mỗi bước, không phải một pass gộp. Khoảng trống batch-size giảm theo bình phương. Hãy dự trù chi phí ~1.5× mỗi bước cho mỗi ControlNet đang hoạt động.
- **Cả LoRA lượng tử hóa.** Nếu bạn đã lượng tử hóa mô hình cơ sở (xem Bài 07, Flux trên 8GB), delta LoRA cũng lượng tử hóa sạch sẽ sang 8-bit hoặc 4-bit. Việc tải theo kiểu QLoRA cho phép bạn xếp chồng 5-10 LoRA lên trên một mô hình Flux 4-bit mà không làm tràn bộ nhớ.

Đặc thù Flux: Notebook Flux-on-8GB của Niels lượng tử hóa mô hình cơ sở xuống 4-bit; việc xếp chồng một style LoRA (`pipe.load_lora_weights("user/style-lora")`) lên mô hình cơ sở đã lượng tử hóa đó ở `weight_name="pytorch_lora_weights.safetensors"` vẫn hoạt động. Đây là công thức mà hầu hết các agency SaaS sử dụng vào năm 2026.

## Đọc thêm

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu và cộng sự (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA (ban đầu cho LLM; chuyển sang diffusion).
- [Ye và cộng sự (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou và cộng sự (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — giải pháp thay thế nhẹ hơn cho ControlNet.
- [Ruiz và cộng sự (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — Tài liệu ControlNet / LoRA / IP-Adapter](https://huggingface.co/docs/diffusers/training/controlnet) — các pipeline tham khảo.