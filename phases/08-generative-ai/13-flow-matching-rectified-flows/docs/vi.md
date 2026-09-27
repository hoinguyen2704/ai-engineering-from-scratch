# Flow Matching & Rectified Flows

> Các mô hình khuếch tán (diffusion models) cần 20-50 bước lấy mẫu vì chúng di chuyển theo một đường cong từ nhiễu đến dữ liệu. Flow matching (Lipman và cộng sự, 2023) và rectified flow (Liu và cộng sự, 2022) huấn luyện các đường đi thẳng. Đường đi càng thẳng thì càng ít bước, đồng nghĩa với suy luận (inference) nhanh hơn. Stable Diffusion 3, Flux.1 và AudioCraft 2 đều đã chuyển sang flow matching trong năm 2024.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 06 (DDPM), Phase 1 · Calculus
**Time:** ~45 phút

## Vấn đề

Quá trình ngược của DDPM là một bước đi ngẫu nhiên gồm 1000 bước từ `N(0, I)` trở về phân phối dữ liệu. DDIM đã rút gọn nó xuống còn 20-50 bước tất định. Bạn muốn ít bước hơn — lý tưởng nhất là một bước. Rào cản ở đây là ODE giải quá trình ngược bị "cứng" (stiff); đường đi bị cong.

Nếu bạn có thể huấn luyện mô hình sao cho đường đi từ nhiễu đến dữ liệu là một *đường thẳng*, thì một bước Euler duy nhất từ `t=1` đến `t=0` sẽ hoạt động. Flow matching xây dựng điều này trực tiếp: xác định một phép nội suy đường thẳng từ `x_1 ∼ N(0, I)` đến `x_0 ∼ data`, huấn luyện một trường vector `v_θ(x, t)` để khớp với đạo hàm theo thời gian của nó, sau đó tích phân tại thời điểm suy luận.

Rectified flow (Liu 2022) tiến xa hơn: làm thẳng các đường đi một cách lặp đi lặp lại với quy trình reflow, tạo ra một ODE ngày càng gần với đường thẳng hơn. Sau hai lần lặp reflow, một bộ lấy mẫu 2 bước có chất lượng tương đương với DDPM 50 bước.

## Khái niệm

![Flow matching: straight-line interpolation between noise and data](../assets/flow-matching.svg)

### Straight-line flow (Dòng chảy đường thẳng)

Xác định:

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

trong đó `x_0 ~ data` và `x_1 ~ N(0, I)`. Đạo hàm theo thời gian dọc theo đường thẳng này là hằng số:

```
dx_t / dt = x_1 - x_0
```

Xác định một trường vector thần kinh `v_θ(x_t, t)` và huấn luyện nó khớp với đạo hàm này:

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

Đây là hàm mất mát **conditional flow matching** (Lipman 2023). Việc huấn luyện không cần mô phỏng (simulation-free): bạn không bao giờ cần giải ODE. Chỉ cần lấy mẫu `(x_0, x_1, t)` và thực hiện hồi quy.

### Lấy mẫu (Sampling)

Tại thời điểm suy luận, tích phân trường vector đã học *ngược* theo thời gian:

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

Bắt đầu tại `x_1 ~ N(0, I)`, thực hiện bước Euler xuống `t=0`.

### Rectified flow (Liu 2022)

Flow đường thẳng hoạt động tốt nhưng các đường đi đã học *thực tế không thẳng* — chúng bị cong vì nhiều `x_0` có thể ánh xạ tới cùng một `x_1`. Bước reflow của Rectified flow:

1. Huấn luyện mô hình flow v_1 với các cặp ngẫu nhiên.
2. Lấy mẫu N cặp `(x_1, x_0)` bằng cách tích phân v_1 từ `x_1` đến điểm kết thúc `x_0`.
3. Huấn luyện v_2 trên các cặp ví dụ đó. Vì các cặp bây giờ đã "khớp ODE", phép nội suy đường thẳng giữa chúng thực sự phẳng hơn.
4. Lặp lại.

Trong thực tế, 2 lần lặp reflow giúp bạn đạt được trạng thái gần như tuyến tính, cho phép suy luận trong 2-4 bước. SDXL-Turbo, SD3-Turbo, LCM đều là các mô hình được chưng cất (distilled) từ flow matching.

### Tại sao phương pháp này thắng thế trong lĩnh vực hình ảnh năm 2024

Ba lý do:

1. **Huấn luyện không cần mô phỏng** — không cần giải ODE trong quá trình huấn luyện, triển khai rất đơn giản.
2. **Hình học mất mát tốt hơn** — các đường thẳng có tỷ lệ tín hiệu trên nhiễu (SNR) nhất quán, trong khi hàm mất mát ε của DDPM có SNR kém ở các biên của lịch trình.
3. **Suy luận nhanh hơn** — 4-8 bước với chất lượng SDXL-Turbo; 1 bước với consistency distillation.

## Flow matching so với DDPM — mối liên hệ chính xác

Flow matching với đường dẫn điều kiện Gaussian chính là diffusion *với một lịch trình nhiễu cụ thể*. Chọn lịch trình `x_t = α(t) x_0 + σ(t) x_1` và flow matching sẽ khôi phục lại quá trình diffusion được công thức hóa theo Stratonovich với `v = α'·x_0 - σ'·x_1`. Hai phương pháp này tương đương về mặt đại số đối với các đường dẫn Gaussian.

Những gì flow matching bổ sung: sự *rõ ràng* của mục tiêu (một vận tốc đơn thuần), hàm mất mát sạch hơn và quyền tự do thử nghiệm với các phép nội suy phi Gaussian.

```figure
normalizing-flow
```

## Xây dựng

`code/main.py` triển khai flow matching 1-D trên hỗn hợp Gaussian hai chế độ. Trường vector `v_θ(x, t)` là một MLP nhỏ được huấn luyện với mục tiêu đường thẳng. Tại thời điểm suy luận, tích phân 1, 2, 4 và 20 bước Euler và so sánh chất lượng mẫu.

### Bước 1: hàm mất mát huấn luyện

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Bước 2: suy luận đa bước

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Bước 3: so sánh số bước

Dự kiến bộ lấy mẫu 4 bước đã có thể đạt chất lượng tương đương 20 bước — một yếu tố quan trọng đối với độ trễ.

## Các cạm bẫy

- **Tham số hóa thời gian.** Flow matching sử dụng `t ∈ [0, 1]` với `t=0` tại dữ liệu, `t=1` tại nhiễu. DDPM sử dụng `t ∈ [0, T]` với `t=0` tại dữ liệu, `t=T` tại nhiễu. Cùng hướng nhưng khác thang đo. Các bài báo thường xuyên nhầm lẫn điều này.
- **Lựa chọn lịch trình.** Đường thẳng của Rectified flow là lịch trình "chuẩn" của flow-matching, nhưng bạn có thể sử dụng lấy mẫu t theo cosine hoặc logit-normal (SD3 thực hiện điều này) để có độ bao phủ thang đo tốt hơn.
- **Chi phí Reflow.** Việc tạo tập dữ liệu cặp cho reflow là một lượt suy luận đầy đủ cho mỗi mẫu. Chỉ thực hiện reflow khi bạn thực sự cần suy luận 1-2 bước.
- **Classifier-free guidance vẫn áp dụng được.** Chỉ cần thay thế ε bằng v trong tổ hợp tuyến tính: `v_cfg = (1+w) v_cond - w v_uncond`.

## Sử dụng

| Trường hợp sử dụng | Stack 2026 |
|----------|-----------|
| Text-to-image, chất lượng tốt nhất | Flow matching: SD3, Flux.1-dev |
| Text-to-image, 1-4 bước | Distilled flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| Suy luận thời gian thực | Consistency distillation từ nền tảng flow-matched (LCM, PCM) |
| Tạo âm thanh | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| Tạo video | Flow matching kết hợp với diffusion (Sora, Veo, Stable Video) |
| Khoa học / vật lý (quỹ đạo hạt, phân tử) | Flow matching + trường vector bất biến (equivariant) |

Bất cứ khi nào một bài báo nói "nhanh hơn diffusion" trong giai đoạn 2025-2026, gần như luôn là flow matching + distillation.

## Triển khai

Lưu `outputs/skill-fm-tuner.md`. Kỹ năng này lấy một đặc tả mô hình kiểu diffusion và chuyển đổi nó thành cấu hình huấn luyện flow-matching: lựa chọn lịch trình, phân phối lấy mẫu thời gian (uniform / logit-normal), bộ tối ưu hóa, kế hoạch reflow, số bước mục tiêu, giao thức đánh giá.

## Bài tập

1. **Dễ.** Chạy `code/main.py` và so sánh MSE 1 bước so với 20 bước với phân phối dữ liệu thực.
2. **Trung bình.** Chuyển từ lấy mẫu `t` đồng nhất sang logit-normal (tập trung lấy mẫu tại mid-t). Chất lượng mô hình có cải thiện không?
3. **Khó.** Triển khai một lần lặp reflow: tạo cặp (x_0, x_1) bằng cách tích phân mô hình đầu tiên, huấn luyện mô hình thứ hai trên các cặp đó và so sánh chất lượng mẫu 1 bước.

## Các thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Flow matching | "Diffusion đường thẳng" | Huấn luyện `v_θ(x, t)` khớp với `x_1 - x_0` dọc theo một phép nội suy. |
| Rectified flow | "Reflow" | Quy trình lặp lại giúp làm thẳng các flow đã học. |
| Velocity field | "v_θ" | Đầu ra của mô hình — hướng di chuyển `x_t`. |
| Straight-line interpolant | "Đường đi" | `x_t = (1-t)·x_0 + t·x_1`; đạo hàm mục tiêu đơn giản. |
| Euler sampler | "Bộ giải ODE bậc 1" | Bộ tích phân đơn giản nhất; hoạt động tốt khi đường đi thẳng. |
| Logit-normal t | "Lấy mẫu SD3" | Tập trung lấy mẫu `t` về phía các giá trị trung bình nơi gradient mạnh nhất. |
| Consistency distillation | "Bộ lấy mẫu 1 bước" | Huấn luyện học viên ánh xạ bất kỳ `x_t` trực tiếp tới `x_0`. |
| CFG with velocity | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`; cùng một thủ thuật, biến số mới. |

## Ghi chú sản xuất: Flux.1-schnell là flow matching ở tốc độ nhanh nhất

Thành công trong sản xuất của flow matching chính là Flux.1-schnell — một DiT flow-matched được chưng cất xuống 1-4 bước suy luận trong khi vẫn giữ được chất lượng cấp Flux-dev. Notebook "Chạy Flux trên máy 8GB" của Niels là công thức triển khai tham chiếu: T5 + CLIP encode, quantized MMDiT denoise (trong 4 bước cho schnell so với 50 cho dev), VAE decode. Hạch toán chi phí:

| Biến thể | Bước | Độ trễ tại 1024² trên L4 | Tổng FLOPs (tương đối) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (gốc) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (nhanh hơn 12×) |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-bước | 2 | ~0.3 s | 0.03× |

Quy tắc sản xuất: **nền tảng flow-matched + distillation = mặc định cho năm 2026 đối với text-to-image nhanh.** Mọi nhà cung cấp lớn đều phát hành combo này: SD3-Turbo (SD3 + flow + distillation), Flux-schnell (Flux-dev + rectified-flow straightening), CogView-4-Flash. Các nền tảng diffusion thuần túy chỉ còn tồn tại cho các checkpoint cũ.

## Đọc thêm

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, rectified flow ở quy mô lớn.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — khung tổng quát bao gồm FM + diffusion.
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — chưng cất 1 bước của diffusion / flow.
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — biến thể turbo.
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — flow matching trong sản xuất.