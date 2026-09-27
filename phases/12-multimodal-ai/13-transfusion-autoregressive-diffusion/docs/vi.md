# Transfusion: Autoregressive Text + Diffusion Image in One Transformer

> Chameleon và Emu3 đặt cược tất cả vào các discrete token. Chúng hoạt động hiệu quả, nhưng nút thắt cổ chai của quá trình lượng tử hóa (quantization) rất rõ ràng — chất lượng hình ảnh bị chững lại so với các mô hình diffusion trong không gian liên tục. Transfusion (Meta, Zhou và cộng sự, tháng 8 năm 2024) chọn hướng đi ngược lại: giữ hình ảnh ở dạng liên tục, loại bỏ hoàn toàn VQ-VAE và huấn luyện một transformer duy nhất với hai hàm mất mát (loss). Các text token thực hiện dự đoán token tiếp theo (next-token-prediction). Các image patch thực hiện hàm mất mát flow-matching / diffusion. Cả hai mục tiêu đều tối ưu hóa cùng một tập trọng số. Kiến trúc nền tảng của Stable Diffusion 3 (MMDiT) là một "người anh em" gần gũi. Bài học này sẽ đọc luận văn Transfusion, xây dựng một trình huấn luyện hai hàm mất mát đơn giản (toy trainer) và truy vết attention mask cho phép một transformer thực hiện cả hai công việc.

**Type:** Build
**Languages:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Time:** ~180 minutes

## Mục tiêu học tập

- Kết nối một transformer chạy hai hàm mất mát (NTP trên text token, diffusion MSE trên image patch) trên cùng một backbone.
- Giải thích tại sao attention hai chiều (bidirectional) trên các image patch kết hợp với attention nhân quả (causal) trên các text token lại là lựa chọn mask phù hợp.
- So sánh phong cách Transfusion (hình ảnh liên tục, diffusion loss) với phong cách Chameleon (hình ảnh rời rạc, NTP) về mặt tính toán, chất lượng và độ phức tạp của mã nguồn.
- Nêu đóng góp của MMDiT: trọng số đặc thù cho từng modality tại mỗi block, attention chung (joint attention) tại residual stream.

## Vấn đề

Cuộc tranh luận về discrete token so với continuous token cho hình ảnh đã có từ trước cả LLM. Các biểu diễn liên tục (pixel thô, VAE latents) bảo toàn chi tiết. Các token rời rạc (VQ indices) phù hợp với từ vựng gốc của transformer nhưng làm mất chi tiết ở bước lượng tử hóa.

Chameleon / Emu3 chọn hướng rời rạc: một hàm mất mát, một kiến trúc, nhưng độ trung thực của hình ảnh bị giới hạn bởi chất lượng của tokenizer.

Các mô hình diffusion chọn hướng liên tục: chất lượng hình ảnh vượt trội, nhưng lại là một mô hình tách biệt với LLM, kỹ thuật lập lịch nhiễu (noise-schedule) phức tạp và không có sự tích hợp liền mạch với việc tạo văn bản.

Transfusion đặt câu hỏi: liệu chúng ta có thể có cả hai? Giữ hình ảnh liên tục, vẫn huấn luyện một mô hình duy nhất, sử dụng hai hàm mất mát được kết hợp vào một bước gradient.

## Khái niệm

### Kiến trúc hai hàm mất mát

Một transformer decoder-only duy nhất xử lý một chuỗi bao gồm:

- Text token (rời rạc, từ từ vựng BPE).
- Image patch (liên tục, các khối pixel 16x16 được chiếu vào hidden dim thông qua linear embedding — giống như đầu vào của ViT encoder).
- Các thẻ `<image>` và `</image>` đánh dấu vị trí của các patch liên tục.

Forward pass chạy một lần. Hàm mất mát chọn một trong hai đầu ra (head) cho mỗi token:

- Đối với text token: cross-entropy tiêu chuẩn trên head dự đoán vocab-logits.
- Đối với image patch: diffusion loss trên các patch liên tục — dự đoán nhiễu đã được thêm vào mỗi patch.

Gradient truyền qua thân transformer dùng chung. Cả hai hàm mất mát đều cải thiện các trọng số dùng chung cùng một lúc.

### Attention mask: text nhân quả + hình ảnh hai chiều

Text token phải mang tính nhân quả (causal) — bạn không thể để một text token chú ý đến các text token tương lai, nếu không teacher forcing sẽ bị phá vỡ. Tuy nhiên, các image patch đại diện cho một ảnh chụp nhanh; chúng nên chú ý đến nhau theo cả hai chiều trong cùng một khối hình ảnh.

Mask:

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

Được triển khai dưới dạng mask tam giác khối (block-triangular) trong quá trình huấn luyện và suy luận (inference).

### Diffusion loss bên trong transformer

Diffusion loss là tiêu chuẩn: thêm nhiễu vào một image patch, yêu cầu mô hình dự đoán nhiễu (hoặc dự đoán patch sạch, tương đương nhau). Phiên bản của Transfusion sử dụng flow matching — dự đoán trường vận tốc (velocity field) từ trạng thái nhiễu sang trạng thái sạch.

Trong quá trình huấn luyện:
1. Với mỗi image patch x0, lấy mẫu một timestep t ngẫu nhiên.
2. Lấy mẫu nhiễu ε, tính toán xt = (1-t) * x0 + t * ε (nội suy tuyến tính cho flow matching).
3. Transformer dự đoán v_theta(xt, t); loss = MSE(v_theta(xt, t), ε - x0).
4. Backprop cùng với các loss NTP của văn bản từ cùng một chuỗi.

Tại thời điểm suy luận, quá trình tạo (generation) là:
- Text token: lấy mẫu tự hồi quy (autoregressive) tiêu chuẩn.
- Image patch: vòng lặp lấy mẫu diffusion (thường từ 10-30 bước) được điều kiện hóa (conditioned) bởi các text token trước đó.

### MMDiT: Biến thể của Stable Diffusion 3

Stable Diffusion 3 (Esser và cộng sự, tháng 3 năm 2024) đã ra mắt MMDiT (Multimodal Diffusion Transformer) cùng thời điểm với Transfusion. Các kiến trúc này là anh em của nhau.

Những khác biệt chính của MMDiT:

- Trọng số đặc thù cho từng modality tại mỗi block. Mỗi transformer block có các trọng số Q, K, V và MLP riêng biệt cho text token so với image patch. Attention là chung (liên modality); mọi thứ khác đều đặc thù cho từng modality.
- Huấn luyện Rectified flow. Một biến thể flow-matching cụ thể với cách lấy mẫu đã biết và toán học đơn giản hơn DDPM.
- Quy mô. MMDiT là backbone cho SD3 (các biến thể 2B và 8B tham số). Bài báo Transfusion mở rộng quy mô lên 7B.

Cả hai đều hội tụ về cùng một ý tưởng cốt lõi: một transformer chạy NTP trên văn bản và diffusion trên các biểu diễn hình ảnh liên tục.

### Tại sao cách này vượt trội hơn phong cách Chameleon

Khoảng cách chất lượng giữa continuous-diffusion và discrete-NTP trong việc tạo hình ảnh là có thể đo lường được. Bài báo Transfusion báo cáo:

- Ở mức 7B tham số, vượt qua mô hình phong cách Chameleon cùng kích thước từ 3-5 điểm FID.
- Không cần huấn luyện tokenizer — bộ mã hóa hình ảnh đơn giản hơn (Linear projection vào hidden, giống như lớp đầu vào của ViT).
- Suy luận có thể song song hóa việc khử nhiễu image patch, không giống như các image token tự hồi quy.

Nhược điểm: Transfusion là mô hình hai hàm mất mát, khiến động lực huấn luyện trở nên khó khăn hơn. Trọng số của hàm mất mát cần được tinh chỉnh. Sự không khớp về lịch trình giữa NTP và diffusion có thể khiến một head chiếm ưu thế.

### Những gì tiếp nối sau đó

Janus-Pro (Bài học 12.15) tinh chỉnh ý tưởng của Transfusion bằng cách tách rời vision encoder cho việc hiểu và tạo — SigLIP cho cái này, VQ cho cái kia — trong khi vẫn chia sẻ thân transformer. Show-o (Bài học 12.14) thay thế diffusion bằng discrete-diffusion (dự đoán có che giấu). Gia đình unified-generation phân nhánh nhanh chóng sau Transfusion.

Các VLM sản xuất năm 2026 có khả năng xuất hình ảnh — Gemini 3 Pro, GPT-5, đường dẫn tạo hình ảnh của Claude Opus 4.7 — gần như chắc chắn sử dụng một hậu duệ nào đó của gia đình này. Các chi tiết là độc quyền.

```figure
cfg-guidance-scale
```

## Sử dụng

`code/main.py` xây dựng một Transfusion đơn giản trên một bài toán nhỏ giống MNIST:

- Các chú thích văn bản là các chuỗi số nguyên ngắn mô tả một chữ số (0-9).
- Hình ảnh là lưới byte 4x4.
- Một cặp linear projection chia sẻ trọng số đóng vai trò là transformer thay thế; NTP loss trên văn bản, MSE loss trên các patch nhiễu.
- Vòng lặp huấn luyện luân phiên hai hàm mất mát, attention mask là tường minh.
- Quá trình tạo ra một chú thích văn bản và một hình ảnh 4x4 trong một forward pass.

Transformer này chỉ là mô hình đồ chơi. Hệ thống hai hàm mất mát, cấu trúc attention mask và vòng lặp suy luận mới là những thành phần thực sự.

## Triển khai

Bài học này tạo ra `outputs/skill-two-loss-trainer-designer.md`. Với một tác vụ huấn luyện đa phương thức mới (văn bản + hình ảnh, văn bản + âm thanh, văn bản + video), nó thiết kế lịch trình hai hàm mất mát (trọng số loss, hình dạng mask, các block chia sẻ vs đặc thù modality) và gắn cờ các rủi ro triển khai.

## Bài tập

1. Một mô hình kiểu Transfusion huấn luyện 70% text token và 30% image patch. Diffusion loss của hình ảnh lớn gấp khoảng 10 lần NTP loss của văn bản. Trọng số loss nào sẽ cân bằng chúng?

2. Triển khai mask tam giác khối cho một chuỗi: `[T, T, <image>, P, P, P, P, </image>, T]`. Đánh dấu mỗi mục là 0 hoặc 1.

3. MMDiT có trọng số QKV đặc thù cho từng modality. Điều này làm tăng bao nhiêu tham số so với transformer chia sẻ hoàn toàn của Transfusion? Ở mức 7B tham số, liệu có đáng không?

4. Quá trình tạo: với một prompt văn bản, mô hình chạy NTP cho 50 token, sau đó gặp `<image>`, rồi chạy diffusion trên 256 patch qua 20 bước khử nhiễu. Tổng cộng có bao nhiêu forward pass?

5. Đọc phần 3 của bài báo SD3. Mô tả rectified flow và tại sao nó hội tụ trong ít bước suy luận hơn DDPM.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Two-loss training | "NTP + diffusion" | Một transformer duy nhất tối ưu hóa cả cross-entropy trên text token và MSE trên image patch liên tục trong cùng một bước gradient |
| Flow matching | "Rectified flow" | Biến thể diffusion dự đoán trường vận tốc từ nhiễu sang dữ liệu sạch; toán học đơn giản hơn DDPM |
| MMDiT | "Multimodal DiT" | Kiến trúc của Stable Diffusion 3: joint attention, MLP và norm đặc thù cho từng modality |
| Block-triangular mask | "Causal text + bidirectional image" | Attention mask mang tính nhân quả trên văn bản nhưng hai chiều trong các vùng hình ảnh |
| Continuous image representation | "No VQ" | Image patch dưới dạng vector giá trị thực, không phải chỉ số codebook số nguyên |
| Velocity prediction | "v-parameterization" | Đầu ra của mạng là trường vận tốc giữa nhiễu và dữ liệu, không phải bản thân nhiễu |

## Đọc thêm

- [Zhou và cộng sự — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser và cộng sự — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao và cộng sự — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie và cộng sự — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)