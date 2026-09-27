# Diffusion Models — DDPM from Scratch

> Ho, Jain, Abbeel (2020) đã mang đến cho lĩnh vực này một công thức mà không ai có thể từ bỏ. Phá hủy dữ liệu bằng nhiễu qua hàng nghìn bước nhỏ. Huấn luyện một mạng thần kinh để dự đoán nhiễu đó. Đảo ngược quá trình này khi suy luận (inference). Ngày nay, mọi mô hình hình ảnh, video, 3D và âm nhạc phổ biến đều chạy trên vòng lặp này, có thể kết hợp thêm các kỹ thuật flow matching hoặc consistency.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 8 · 02 (VAE)
**Time:** ~75 phút

## Vấn đề

Bạn muốn có một bộ lấy mẫu (sampler) cho `p_data(x)`. Các GAN chơi một trò chơi minimax thường dẫn đến phân kỳ. Các VAE tạo ra các mẫu mờ nhạt từ bộ giải mã Gaussian. Điều bạn thực sự muốn là một mục tiêu huấn luyện: (a) một hàm mất mát ổn định duy nhất (không có điểm yên ngựa, không có minimax), (b) một cận dưới cho `log p(x)` (để bạn có các giá trị likelihood), và (c) các mẫu có chất lượng đạt chuẩn SOTA.

Sohl-Dickstein và cộng sự (2015) đã có một câu trả lời lý thuyết: định nghĩa một chuỗi Markov `q(x_t | x_{t-1})` dần dần thêm nhiễu Gaussian, và huấn luyện một chuỗi đảo ngược `p_θ(x_{t-1} | x_t)` để khử nhiễu. Ho, Jain, Abbeel (2020) đã chỉ ra rằng hàm mất mát có thể được đơn giản hóa thành một dòng duy nhất — dự đoán nhiễu — và làm gọn các công thức toán học. Năm 2020, đây chỉ là một sự tò mò. Năm 2021, nó tạo ra các mẫu đạt chuẩn SOTA. Năm 2022, nó trở thành Stable Diffusion. Năm 2026, nó là nền tảng cơ bản.

## Khái niệm

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**Quá trình thuận `q`.** Thêm nhiễu Gaussian trong `T` bước nhỏ. Dạng đóng — lý do khiến toán học trở nên khả thi — là bước tích lũy cũng là Gaussian:

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

trong đó `α̅_t = ∏_{s=1..t} (1 - β_s)` cho một lịch trình `β_t`. Chọn `β_t` từ 1e-4 đến 0.02 một cách tuyến tính qua T=1000 bước và `x_T` xấp xỉ `N(0, I)`.

**Quá trình ngược `p_θ`.** Học một mạng thần kinh `ε_θ(x_t, t)` dự đoán nhiễu đã được thêm vào. Với `x_t`, khử nhiễu bằng cách:

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

trong đó `σ_t` là `sqrt(β_t)` hoặc một phương sai đã học. Biểu thức này trông có vẻ phức tạp nhưng thực chất chỉ là đại số — giải tìm `x_{t-1}` dựa trên phân phối hậu nghiệm `q(x_{t-1} | x_t, x_0)` và thay thế `x_0` bằng ước tính dự đoán nhiễu của nó.

**Hàm mất mát huấn luyện.**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

Lấy mẫu `x_0` từ dữ liệu, chọn ngẫu nhiên `t`, lấy mẫu `ε ~ N(0, I)`, tính toán `x_t` bị nhiễu trong một lần thông qua dạng đóng, và hồi quy trên nhiễu. Một hàm mất mát, không minimax, không KL, không cần các thủ thuật tái tham số hóa (reparameterization tricks).

**Lấy mẫu.** Bắt đầu `x_T ~ N(0, I)`. Lặp lại bước ngược từ `t = T` đến `1`. Xong.

## Tại sao nó hiệu quả

Ba trực giác:

1. **Khử nhiễu thì dễ; tạo mới thì khó.** Tại `t=T`, dữ liệu là nhiễu thuần túy — mạng chỉ cần giải một bài toán tầm thường. Tại `t=0`, mạng chỉ cần làm sạch một vài pixel. Tại các `t` trung gian, bài toán khó hơn nhưng mạng có nhiều gradient chảy qua cùng các trọng số từ mọi mức nhiễu.

2. **Score matching ẩn giấu.** Vincent (2011) đã chứng minh rằng việc dự đoán nhiễu tương đương với việc ước tính `∇_x log q(x_t | x_0)`, hay còn gọi là *score*. SDE ngược sử dụng score này để đi lên theo gradient mật độ — một bước đi ngẫu nhiên có hướng về phía các vùng có xác suất cao.

3. **ELBO rút gọn thành MSE đơn giản.** Cận dưới biến phân (variational lower bound) đầy đủ có một số hạng KL cho mỗi bước thời gian. Với tham số hóa của DDPM, các số hạng KL đó đơn giản hóa thành MSE trên dự đoán nhiễu với các hệ số cụ thể; Ho đã bỏ các hệ số này (gọi nó là hàm mất mát "đơn giản") và chất lượng *đã cải thiện*.

```figure
diffusion-denoise
```

## Xây dựng

`code/main.py` triển khai một DDPM 1-D. Dữ liệu là một hỗn hợp hai chế độ (two-mode mixture). "Mạng" là một MLP nhỏ nhận `(x_t, t)` và xuất ra nhiễu dự đoán. Huấn luyện là hàm mất mát một dòng. Lấy mẫu là lặp lại chuỗi ngược.

### Bước 1: lịch trình thuận (dạng đóng)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Bước 2: lấy mẫu `x_t` trong một lần

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Bước 3: một bước huấn luyện

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Bước 4: lấy mẫu ngược

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

Đối với bài toán 1-D với 40 bước thời gian và MLP 24 đơn vị, mô hình này học được hỗn hợp hai chế độ trong khoảng 200 epoch.

## Điều kiện hóa thời gian (Time conditioning)

Mạng cần biết nó đang khử nhiễu ở bước thời gian nào. Hai tùy chọn tiêu chuẩn:

- **Sinusoidal embedding.** Giống như mã hóa vị trí (positional encoding) trong Transformer. `embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`. Truyền qua một MLP, sau đó broadcast vào mạng.
- **Film / group-norm conditioning.** Chiếu embedding thành tỷ lệ/độ lệch (scale/bias) theo kênh (FiLM) tại mỗi khối.

Mã nguồn mẫu của chúng ta sử dụng sinusoidal → concat. Các U-Net trong sản xuất sử dụng FiLM.

## Các cạm bẫy

- **Lịch trình rất quan trọng.** `β` tuyến tính là mặc định của DDPM nhưng lịch trình cosine (Nichol & Dhariwal, 2021) mang lại FID tốt hơn với cùng mức tính toán. Hãy đổi lịch trình nếu chất lượng bị chững lại.
- **Embedding bước thời gian rất nhạy cảm.** Truyền `t` thô dưới dạng float hoạt động với bài toán 1-D mẫu nhưng thất bại với hình ảnh; luôn sử dụng embedding phù hợp.
- **V-prediction so với ε-prediction.** Đối với các chế độ hẹp (t rất nhỏ hoặc rất lớn), `ε` có tỷ lệ tín hiệu trên nhiễu kém. V-prediction (`v = α·ε - σ·x`) ổn định hơn; SDXL, SD3 và Flux đều sử dụng nó.
- **Classifier-free guidance.** Khi suy luận, tính toán cả `ε` có điều kiện và không điều kiện, sau đó `ε_cfg = (1 + w) · ε_cond - w · ε_uncond` với `w ≈ 3-7`. Đã đề cập trong Bài 08.
- **1000 bước là rất nhiều.** Sản xuất sử dụng DDIM (20-50 bước), DPM-Solver (10-20 bước), hoặc chưng cất (1-4 bước). Xem Bài 12.

## Sử dụng

| Vai trò | Stack điển hình năm 2026 |
|------|-----------------------|
| Diffusion không gian pixel hình ảnh (nhỏ, mẫu) | DDPM + U-Net |
| Diffusion không gian tiềm ẩn (latent) | VAE encoder + U-Net hoặc DiT (Bài 07) |
| Diffusion video tiềm ẩn | Spatiotemporal DiT (Sora, Veo, WAN) |
| Diffusion âm thanh tiềm ẩn | Encodec + diffusion transformer |
| Khoa học (phân tử, protein, vật lý) | Equivariant diffusion (EDM, RFdiffusion, AlphaFold3) |

Diffusion là xương sống tạo sinh phổ quát. Flow matching (Bài 13) là đối thủ cạnh tranh giai đoạn 2024-2026, thường thắng về tốc độ suy luận với cùng chất lượng.

## Triển khai

Lưu `outputs/skill-diffusion-trainer.md`. Kỹ năng này cần một tập dữ liệu + ngân sách tính toán và xuất ra: lịch trình (tuyến tính/cosine/sigmoid), mục tiêu dự đoán (ε/v/x), số bước, thang đo hướng dẫn (guidance scale), họ bộ lấy mẫu và giao thức đánh giá.

## Bài tập

1. **Dễ.** Thay đổi T từ 40 thành 10 trong `code/main.py`. Chất lượng mẫu (biểu đồ trực quan của đầu ra) suy giảm như thế nào? Tại T nào thì cấu trúc hai chế độ bị sụp đổ?
2. **Trung bình.** Chuyển từ ε-prediction sang v-prediction. Suy luận lại bước ngược. So sánh chất lượng mẫu cuối cùng.
3. **Khó.** Thêm classifier-free guidance. Điều kiện hóa trên nhãn lớp `c ∈ {0, 1}`, loại bỏ nó 10% thời gian trong quá trình huấn luyện, và tại thời điểm lấy mẫu sử dụng `ε = (1+w)·ε_cond - w·ε_uncond`. Đo tỷ lệ đạt chế độ có điều kiện tại `w = 0, 1, 3, 7`.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Quá trình thuận | "Thêm nhiễu" | Chuỗi Markov cố định `q(x_t \| x_{t-1})` phá hủy dữ liệu. |
| Quá trình ngược | "Khử nhiễu" | Chuỗi đã học `p_θ(x_{t-1} \| x_t)` tái tạo dữ liệu. |
| Lịch trình β | "Thang nhiễu" | Phương sai mỗi bước; tuyến tính, cosine hoặc sigmoid. |
| α̅ | "Alpha bar" | Tích lũy `∏(1 - β)`; cho dạng đóng `x_t` từ `x_0`. |
| Hàm mất mát đơn giản | "MSE trên nhiễu" | `\|\|ε - ε_θ(x_t, t)\|\|²`; tất cả các dẫn xuất biến phân đều rút gọn về đây. |
| ε-prediction | "Dự đoán nhiễu" | Đầu ra là nhiễu đã thêm; DDPM tiêu chuẩn. |
| V-prediction | "Dự đoán vận tốc" | Đầu ra là `α·ε - σ·x`; điều kiện hóa tốt hơn qua t. |
| DDPM | "Bài báo gốc" | Ho và cộng sự 2020; β tuyến tính, 1000 bước, U-Net. |
| DDIM | "Bộ lấy mẫu tất định" | Bộ lấy mẫu phi Markov, 20-50 bước, cùng mục tiêu huấn luyện. |
| Classifier-free guidance | "CFG" | Trộn dự đoán nhiễu có điều kiện và không điều kiện để khuếch đại điều kiện hóa. |

## Lưu ý sản xuất: suy luận diffusion là bài toán đếm bước

Bài báo DDPM chạy 1000 bước ngược. Không ai đưa điều đó vào sản xuất. Mọi stack suy luận thực tế đều chọn một trong ba chiến lược — và mỗi chiến lược ánh xạ rõ ràng vào khung sản xuất về "độ trễ đến từ đâu":

1. **Bộ lấy mẫu nhanh hơn, cùng mô hình.** DDIM (20-50 bước), DPM-Solver++ (10-20), UniPC (8-16). Thay thế trực tiếp vòng lặp ngược; các trọng số `ε_θ` đã huấn luyện không bị thay đổi. Cắt giảm độ trễ 20-50 lần.
2. **Chưng cất (Distillation).** Huấn luyện một học viên (student) để khớp với giáo viên (teacher) trong ít bước hơn: Progressive Distillation (2 → 1), Consistency Models (tùy ý → 1-4), LCM, SDXL-Turbo, SD3-Turbo. Cắt giảm độ trễ thêm 5-10 lần, yêu cầu huấn luyện lại.
3. **Caching và biên dịch.** `torch.compile(unet, mode="reduce-overhead")`, các backend diffusion của TensorRT-LLM, `xformers`/SDPA attention, trọng số bf16. Cắt giảm độ trễ mỗi bước khoảng 2 lần. Có thể kết hợp với (1) và (2).

Đối với một máy chủ diffusion sản xuất, cuộc thảo luận về ngân sách cũng giống như tài liệu sản xuất mô tả cho LLM: độ trễ là `num_steps × step_cost + VAE_decode`, thông lượng là `batch_size × (num_steps × step_cost)^-1`. TTFT (thời gian đến token đầu tiên) rất nhỏ (một bước); TPOT (thời gian mỗi token) tương đương với toàn bộ thời gian phản hồi vì tạo hình ảnh là "tất cả cùng một lúc" từ góc nhìn của người dùng.

## Đọc thêm

- [Sohl-Dickstein và cộng sự (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — bài báo về diffusion, đi trước thời đại.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM.
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM, ít bước hơn.
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — lịch trình cosine, phương sai đã học.
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — hướng dẫn phân loại (classifier guidance).
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Karras và cộng sự (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — ký hiệu thống nhất, công thức sạch nhất.