# Visual Autoregressive Modeling (VAR): Next-Scale Prediction

> Các mô hình Diffusion lấy mẫu lặp lại theo thời gian (các bước khử nhiễu). VAR lấy mẫu lặp lại theo thang đo (scale) — nó dự đoán token 1x1, sau đó là 2x2, rồi 4x4, cho đến độ phân giải cuối cùng, mỗi thang đo đều được điều kiện hóa dựa trên các thang đo trước đó. Bài báo năm 2024 cho thấy VAR tuân thủ các định luật mở rộng (scaling laws) kiểu GPT cho việc tạo ảnh và vượt trội hơn DiT với cùng ngân sách tính toán. Bài học này xây dựng cơ chế cốt lõi đó.

**Type:** Build
**Languages:** Python (với PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 phút

## Vấn đề

Tạo mô hình tự hồi quy (autoregressive) đã thống trị mô hình ngôn ngữ vì nó mở rộng một cách có thể dự đoán được: nhiều tính toán hơn, nhiều tham số hơn, độ phức tạp (perplexity) thấp hơn, đầu ra tốt hơn. Tạo ảnh đã có hai nỗ lực AR chính trước năm 2024: PixelRNN/PixelCNN (từng pixel một) và DALL-E 1 / Parti / MuseGAN (từng token một dựa trên mã VQ-VAE).

Cả hai đều gặp vấn đề về thứ tự tạo. Các pixel và token được sắp xếp trong một lưới 2D, nhưng mô hình AR phải truy cập chúng theo thứ tự raster 1D. Một pixel ở góc sớm không biết hình ảnh cuối cùng sẽ trở thành gì. Chất lượng tạo ảnh mở rộng kém hơn so với GPT trên văn bản và không bao giờ đạt được chất lượng của mô hình diffusion ở cùng mức tính toán.

VAR khắc phục vấn đề thứ tự tạo bằng cách thay đổi những gì đang được tạo. Thay vì dự đoán các token ảnh từng cái một trong không gian, VAR dự đoán toàn bộ hình ảnh ở các độ phân giải tăng dần. Bước 1: dự đoán token 1x1 ("tóm tắt" toàn bộ hình ảnh). Bước 2: dự đoán lưới token 2x2 (các đặc trưng thô hơn). Bước 3: dự đoán lưới 4x4. Bước K: dự đoán lưới cuối cùng (H/8)x(W/8).

Mỗi thang đo chú ý (attend) đến tất cả các thang đo trước đó (theo thứ tự thang đo nhân quả) và song song trong chính thang đo của nó. Vấn đề thứ tự biến mất: toàn bộ hình ảnh ở thang đo k được tạo ra trong một lần truyền qua transformer.

## Khái niệm

### VQ-VAE Multi-Scale Tokenizer

VAR cần một **multi-scale discrete tokenizer**. Đối với một hình ảnh x, nó tạo ra một chuỗi các lưới token có độ phân giải tăng dần:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

Mỗi z_k sử dụng cùng một codebook (kích thước điển hình 4096-16384). Việc token hóa ở mỗi thang đo không độc lập — nó được huấn luyện sao cho việc cộng các phần dư (residuals) ở mỗi thang đo sẽ tái tạo lại f:

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

Đây là một biến thể **residual VQ**. Thang đo k nắm bắt những gì các thang đo 1..k-1 đã bỏ lỡ. Bộ giải mã (decoder) lấy tổng của tất cả các embedding thang đo và tạo ra hình ảnh.

Multi-scale VQ tokenizer được huấn luyện một lần (giống như VQGAN) và sau đó được đóng băng. Tất cả công việc tạo được thực hiện bởi mô hình tự hồi quy bên trên.

### Next-Scale Prediction

Mô hình tạo là một transformer nhìn thấy các token từ tất cả các thang đo trước đó và dự đoán các token ở thang đo tiếp theo.

Cấu trúc chuỗi đầu vào:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Các position embedding mã hóa cả chỉ số thang đo và vị trí không gian trong thang đo đó. Cơ chế Attention mang tính nhân quả theo thứ tự thang đo: token ở thang đo k, vị trí (i, j) có thể chú ý đến tất cả các token ở thang đo 1..k và các token ở chính thang đo k xuất hiện sớm hơn trong bất kỳ thứ tự nội bộ thang đo nào được sử dụng (VAR sử dụng attention vị trí cố định không có tính nhân quả nội bộ thang đo — tất cả các vị trí trong một thang đo được dự đoán song song).

Hàm mất mát huấn luyện: tại mỗi thang đo k, dự đoán các token z_k dựa trên tất cả các token thang đo trước đó. Sử dụng Cross-entropy loss trên các mã VQ rời rạc. Cấu trúc tương tự như GPT ngoại trừ "chuỗi" bây giờ được cấu trúc theo thang đo.

### Generation

Tại thời điểm suy luận (inference):
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

Với K = 10 thang đo, việc tạo ra là 10 lần truyền qua transformer. Mỗi lần truyền tạo ra toàn bộ thang đo của nó song song — không có sự tự hồi quy từng token trong một thang đo. Đối với hình ảnh 256x256, đây là khoảng 10 lần truyền so với 28-50 lần của DiT.

### Tại sao Next-Scale thắng Next-Token

Ba chiến thắng về cấu trúc:
1. **Từ thô đến tinh phù hợp với thống kê hình ảnh tự nhiên.** Nhận thức thị giác của con người và các tập dữ liệu hình ảnh đều thể hiện các quy luật phụ thuộc vào thang đo: cấu trúc tần số thấp ổn định và dễ dự đoán; chi tiết tần số cao phụ thuộc vào nội dung tần số thấp. Dự đoán theo thang đo tiếp theo khai thác điều này.
2. **Tạo song song trong thang đo.** Không giống như AR token kiểu GPT, VAR tạo ra tất cả các token ở một thang đo trong một bước. Độ dài tạo hiệu quả là log-scale thay vì tuyến tính.
3. **Không có thiên kiến về thứ tự tạo.** Các token ở thang đo k nhìn thấy toàn bộ thang đo k-1; không có thiên kiến "bên trái" hoặc "bên trên" buộc các token sớm phải cam kết trước khi có ngữ cảnh muộn.

### Scaling Law

Tian và cộng sự đã chứng minh rằng VAR tuân theo đường cong mở rộng lũy thừa cho FID trên ImageNet — giống như GPT làm cho perplexity. Việc tăng gấp đôi tham số hoặc tính toán sẽ làm giảm một nửa sai số một cách đáng tin cậy. Đây là mô hình tạo ảnh đầu tiên thể hiện hành vi mở rộng này một cách rõ ràng như các mô hình ngôn ngữ. Kết quả là các dự đoán theo thang đo của VAR trở nên có thể dự đoán được từ tính toán, không phải là các phỏng đoán thực nghiệm theo kiến trúc.

### Mối quan hệ với Diffusion

VAR và diffusion chia sẻ cùng một câu chuyện nén dữ liệu: cả hai đều chia vấn đề tạo thành một chuỗi các bài toán con dễ dàng hơn.

- Diffusion: thêm nhiễu dần dần, học cách đảo ngược một bước.
- VAR: thêm độ phân giải dần dần, học cách dự đoán thang đo tiếp theo.

Chúng là các trục khác nhau xuyên qua vấn đề. Cả hai đều mang lại các phân phối có điều kiện có thể giải quyết được. Về mặt thực nghiệm, VAR nhanh hơn khi suy luận (ít lần truyền hơn, tất cả song song trong một thang đo) và ngang bằng hoặc vượt trội hơn DiT trên ImageNet có điều kiện lớp. VAR có điều kiện văn bản (VARclip, HART) là một hướng nghiên cứu tích cực.

```figure
gx-var-next-scale
```

## Build It

Trong `code/main.py` bạn sẽ:
1. Xây dựng một **multi-scale VQ tokenizer** nhỏ trên dữ liệu "ảnh" tổng hợp (các vòng Gaussian 2D).
2. Huấn luyện một **transformer kiểu VAR** để dự đoán token theo thang đo tiếp theo.
3. Lấy mẫu bằng cách gọi transformer 4 lần (4 thang đo) và giải mã.
4. Xác minh rằng việc huấn luyện theo thứ tự thang đo giúp việc tạo song song trong một thang đo.

Đây là một bản triển khai mô hình đồ chơi. Mục đích là để thấy mặt nạ attention có cấu trúc thang đo và việc tạo song song trong thang đo thực sự hoạt động.

## Ship It

Bài học này tạo ra `outputs/skill-var-tokenizer-designer.md` — một kỹ năng thiết kế multi-scale tokenizer: số lượng thang đo, tỷ lệ thang đo, kích thước codebook, chia sẻ phần dư, kiến trúc bộ giải mã.

## Bài tập

1. **Ablation số lượng thang đo.** Huấn luyện VAR với 4, 6, 8, 10 thang đo. Đo chất lượng tái tạo so với số lần truyền tự hồi quy. Nhiều thang đo hơn = phần dư tinh hơn = chất lượng tốt hơn nhưng nhiều lần truyền hơn.

2. **Kích thước codebook.** Huấn luyện các tokenizer với kích thước codebook 512, 4096, 16384. Codebook lớn hơn cho khả năng tái tạo tốt hơn nhưng dự đoán khó hơn. Tìm điểm tối ưu.

3. **Kiểm tra song song trong thang đo.** Đối với một VAR đã huấn luyện, hãy đo lường mô hình attention một cách rõ ràng. Trong thang đo k, mô hình có chú ý đến các vị trí chéo thang đo nhưng không chú ý nội bộ thang đo không? Xác minh việc triển khai mặt nạ (mask).

4. **VAR vs DiT scaling.** Đối với cùng tác vụ ImageNet có điều kiện lớp, huấn luyện VAR và DiT ở các ngân sách tham số tương đương (ví dụ: 33M, 130M, 458M). Vẽ biểu đồ FID so với tính toán. VAR sẽ vượt lên trước DiT ở mỗi kích thước — tái tạo kết quả của bài báo ở quy mô nhỏ.

5. **Điều kiện văn bản.** Mở rộng VAR để nhận embedding văn bản (CLIP pooled) như một đầu vào điều kiện bổ sung thông qua adaLN. Đây là công thức HART. FID cải thiện bao nhiêu khi lấy mẫu theo văn bản?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | Tạo ảnh bằng dự đoán thang đo tiếp theo trên kim tự tháp các lưới token VQ |
| Next-scale prediction | "Dự đoán thô trước, tinh sau" | Mô hình dự đoán các token ở độ phân giải tăng dần, điều kiện hóa trên tất cả các thang đo trước đó |
| Multi-scale VQ tokenizer | "Residual VQ" | VQ-VAE tạo ra K lưới token có độ phân giải tăng dần, với bộ giải mã cộng tất cả các thang đo |
| Scale k | "Cấp độ kim tự tháp k" | Một trong K cấp độ phân giải, từ 1x1 tại k=1 đến (H/p)x(W/p) tại k=K |
| Parallel-within-scale | "Một lần truyền mỗi thang đo" | Tất cả các token ở thang đo k được dự đoán trong một lần truyền transformer, không phải tự hồi quy |
| Causal-across-scales | "Attention theo thứ tự thang đo" | Token ở thang đo k có thể chú ý đến tất cả các thang đo 1..k nhưng không phải thang đo k+1..K |
| Residual VQ | "Token hóa cộng dồn" | Các token của mỗi thang đo mã hóa phần dư còn lại bởi các thang đo thấp hơn; bộ giải mã cộng tất cả các embedding thang đo |
| VAR scaling law | "Scaling Image GPT" | FID tuân theo định luật lũy thừa có thể dự đoán được trong tính toán, giống như perplexity của mô hình ngôn ngữ |
| HART | "VAR lai + văn bản" | Biến thể VAR có điều kiện văn bản kết hợp giải mã lặp lại kiểu MaskGIT với cấu trúc thang đo của VAR |
| Scale position embedding | "Bộ ba (thang đo, hàng, cột)" | Mã hóa vị trí mang cả chỉ số thang đo và tọa độ không gian trong thang đo |

## Đọc thêm

- [Tian và cộng sự, 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — bài báo VAR, tài liệu tham khảo chính
- [Peebles và Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT, cơ sở so sánh cho diffusion
- [Esser và cộng sự, 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN, họ tokenizer mà multi-scale tokenizer của VAR mở rộng
- [van den Oord và cộng sự, 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE, nền tảng của token hóa ảnh rời rạc
- [Tang và cộng sự, 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — VAR có điều kiện văn bản