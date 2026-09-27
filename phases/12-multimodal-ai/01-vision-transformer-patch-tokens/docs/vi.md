# Vision Transformers và Patch-Token Primitive

> Trước khi thực hiện bất kỳ tác vụ đa phương thức (multimodal) nào, một hình ảnh phải được chuyển đổi thành một chuỗi các token mà transformer có thể "tiêu thụ" được. Bài báo ViT năm 2020 đã giải quyết vấn đề này bằng các patch 16x16 pixel, một phép chiếu tuyến tính (linear projection) và một positional embedding. Năm năm sau, mọi mô hình tiên phong năm 2026 (Claude Opus 4.7 ở độ phân giải gốc 2576px, Gemini 3.1 Pro, Qwen3.5-Omni) vẫn bắt đầu theo cách này — bộ mã hóa (encoder) đã thay đổi từ ViT sang DINOv2 rồi đến SigLIP 2, các register token đã được thêm vào, lược đồ vị trí trở thành 2D-RoPE, nhưng primitive (nguyên thủy) vẫn được giữ nguyên. Bài học này sẽ đọc toàn bộ pipeline patch-token từ đầu đến cuối và xây dựng nó bằng Python stdlib để phần còn lại của Giai đoạn 12 có một mô hình tư duy cụ thể về "visual tokens".

**Type:** Learn
**Languages:** Python (stdlib, patch tokenizer + geometry calculator)
**Prerequisites:** Giai đoạn 7 (Transformers), Giai đoạn 4 (Computer Vision)
**Time:** ~120 phút

## Mục tiêu học tập

- Chuyển đổi một hình ảnh HxWx3 thành một chuỗi các patch token với positional encoding chính xác.
- Tính toán độ dài chuỗi, số lượng tham số và FLOPs cho một ViT với các thông số (patch size, độ phân giải, hidden dim, độ sâu) cho trước.
- Kể tên ba nâng cấp đã đưa ViT từ nghiên cứu năm 2020 lên sản xuất năm 2026: tự giám sát tiền huấn luyện (self-supervised pretraining - DINO / MAE), register tokens và đóng gói độ phân giải gốc (native-resolution packing).
- Lựa chọn giữa CLS pooling, mean pooling và register tokens cho một tác vụ hạ nguồn (downstream task).

## Vấn đề

Các Transformer hoạt động trên các chuỗi vector. Văn bản vốn đã là một chuỗi (byte hoặc token). Hình ảnh là một lưới 2D các pixel với ba kênh màu — không phải là một chuỗi. Nếu bạn làm phẳng mọi pixel, một hình ảnh RGB 224x224 sẽ trở thành 150.528 token, và self-attention ở độ dài đó là điều bất khả thi (độ phức tạp bậc hai theo độ dài chuỗi).

Các phương pháp trước năm 2020 đã gắn một bộ trích xuất đặc trưng CNN vào phía trước: ResNet tạo ra bản đồ đặc trưng 7x7 gồm các vector 2048-dim, sau đó đưa 49 token đó vào transformer. Cách này hiệu quả nhưng kế thừa các thiên kiến của CNN (translation equivariance, local receptive fields) và làm mất đi khả năng mở rộng của transformer.

Dosovitskiy và cộng sự (2020) đã đặt ra câu hỏi thẳng thắn: điều gì sẽ xảy ra nếu chúng ta bỏ qua CNN? Chia hình ảnh thành các patch có kích thước cố định (ví dụ 16x16 pixel), chiếu tuyến tính từng patch thành một vector, thêm positional embedding và đưa chuỗi đó vào một transformer thông thường. Vào thời điểm đó, đây là một sự dị giáo — thị giác máy tính không cần tích chập (convolutions). Với đủ dữ liệu (JFT-300M, sau đó là LAION), nó đã đánh bại ResNet trên ImageNet và tiếp tục cải thiện.

Đến năm 2026, ViT primitive là nền tảng không thể bàn cãi. Vision tower của mọi VLM mã nguồn mở đều là hậu duệ của nó (DINOv2, SigLIP 2, CLIP, EVA, InternViT). Câu hỏi không còn là "chúng ta có nên sử dụng patch không?" mà là "kích thước patch nào, lịch trình độ phân giải nào, mục tiêu tiền huấn luyện nào, positional encoding nào".

## Khái niệm

### Patches dưới dạng tokens

Với một hình ảnh `x` có hình dạng `(H, W, 3)` và kích thước patch `P`, bạn chia hình ảnh thành một lưới gồm `(H/P) x (W/P)` patch không chồng lấp. Mỗi patch là một khối `P x P x 3` pixel. Làm phẳng mỗi khối thành một vector `3 P^2`. Áp dụng một phép chiếu tuyến tính chia sẻ `W_E` có hình dạng `(3 P^2, D)` để ánh xạ mỗi patch vào chiều ẩn `D` của mô hình.

Đối với cấu hình chuẩn ViT-B/16:
- Độ phân giải 224, kích thước patch 16 → lưới 14x14 → 196 patch token.
- Mỗi patch là `16 x 16 x 3 = 768` giá trị pixel, được chiếu thành `D = 768`.
- Thêm một token `[CLS]` có thể học được → độ dài chuỗi 197.

Phép chiếu patch về mặt toán học giống hệt với một phép tích chập 2D với kích thước kernel `P`, stride `P` và `D` kênh đầu ra. Đó là cách mã sản xuất thực sự triển khai nó — `nn.Conv2d(3, D, kernel_size=P, stride=P)`. Cách diễn đạt "phép chiếu tuyến tính" mang tính khái niệm; cách diễn đạt kernel thì hiệu quả hơn.

### Positional embeddings

Các patch không có thứ tự vốn có — transformer coi chúng như một cái túi. Các ViT đời đầu đã thêm một positional embedding 1D có thể học được (một vector 768-dim cho mỗi vị trí, tổng cộng 197 vector). Cách này hiệu quả, nhưng gắn mô hình với độ phân giải huấn luyện: khi suy luận (inference), bạn phải nội suy bảng vị trí nếu bạn thay đổi lưới.

Các backbone thị giác hiện đại sử dụng 2D-RoPE (M-RoPE của Qwen2-VL, mặc định của SigLIP 2) hoặc các vị trí 2D phân tách. 2D-RoPE xoay các vector query và key dựa trên chỉ số (hàng, cột) của patch, vì vậy mô hình suy luận vị trí 2D tương đối từ góc xoay. Không cần bảng vị trí. Mô hình xử lý các kích thước lưới tùy ý khi suy luận.

### CLS token, pooled output và register tokens

Đâu là biểu diễn ở cấp độ hình ảnh? Ba lựa chọn cùng tồn tại:

1. Token `[CLS]`. Thêm một vector có thể học được vào đầu chuỗi patch. Sau tất cả các khối transformer, trạng thái ẩn của CLS token chính là biểu diễn hình ảnh. Kế thừa từ BERT. Được sử dụng bởi ViT gốc, CLIP.
2. Mean pool. Lấy trung bình các trạng thái ẩn đầu ra của các patch token. Được sử dụng bởi SigLIP, DINOv2, hầu hết các VLM hiện đại.
3. Register tokens. Darcet và cộng sự (2023) quan sát thấy rằng các ViT được huấn luyện mà không có token "hố đen" (sink token) rõ ràng sẽ phát triển các patch "nhiễu" có chuẩn cao, chiếm dụng self-attention. Việc thêm 4–16 register token có thể học được sẽ hấp thụ tải này và cải thiện chất lượng dự đoán dày đặc (phân đoạn, độ sâu). Cả DINOv2 và SigLIP 2 đều đi kèm với các register.

Sự lựa chọn này quan trọng đối với các tác vụ hạ nguồn. CLS phù hợp cho phân loại. Đối với các VLM đưa patch token vào LLM, bạn bỏ qua hoàn toàn việc pooling — mỗi patch trở thành một token đầu vào của LLM. Các register bị loại bỏ trước khi chuyển giao (chúng là giàn giáo, không phải nội dung).

### Tiền huấn luyện: có giám sát, tương phản, che giấu, tự chưng cất

ViT năm 2020 được tiền huấn luyện bằng phân loại có giám sát trên JFT-300M. Sau đó nhanh chóng bị thay thế bởi:

- CLIP (2021): tương phản hình ảnh-văn bản trên 400 triệu cặp. Bài học 12.02.
- MAE (2021, He và cộng sự): che giấu 75% các patch, tái tạo pixel. Tự giám sát, hoạt động trên hình ảnh thuần túy.
- DINO (2021) / DINOv2 (2023): tự chưng cất (self-distillation) với học sinh-giáo viên, không nhãn, không chú thích. DINOv2 ViT-g/14 năm 2023 là backbone thuần thị giác mạnh nhất và là mặc định cho các trường hợp sử dụng "đặc trưng dày đặc".
- SigLIP / SigLIP 2 (2023, 2025): CLIP với hàm mất mát sigmoid và NaFlex cho tỷ lệ khung hình gốc. Đây là vision tower thống trị trong các VLM mở năm 2026 (Qwen, Idefics2, LLaVA-OneVision).

Lựa chọn tiền huấn luyện của bạn quyết định backbone đó giỏi về việc gì: CLIP/SigLIP cho khớp ngữ nghĩa với văn bản, DINOv2 cho các đặc trưng thị giác dày đặc, MAE làm điểm khởi đầu cho tinh chỉnh hạ nguồn.

### Định luật mở rộng (Scaling laws)

Việc mở rộng ViT (Zhai và cộng sự 2022) đã thiết lập rằng chất lượng của ViT tuân theo các định luật có thể dự đoán được về kích thước mô hình, kích thước dữ liệu và tính toán. Ở mức tính toán cố định:
- Mô hình lớn hơn + nhiều dữ liệu hơn → chất lượng tốt hơn.
- Kích thước patch là đòn bẩy cho độ dài chuỗi so với độ trung thực. Patch 14 (điển hình cho DINOv2/SigLIP SO400m) cung cấp nhiều token hơn trên mỗi hình ảnh so với patch 16; tốt hơn cho OCR và các tác vụ dày đặc, tệ hơn về tốc độ.
- Độ phân giải là đòn bẩy lớn khác. Tăng từ 224 lên 384 lên 512 hầu như luôn giúp ích, với chi phí FLOPs tăng theo bậc hai.

ViT-g/14 (1B tham số, patch 14, độ phân giải 224 → 256 token) và SigLIP SO400m/14 (400M tham số, patch 14) là hai bộ mã hóa chủ lực cho các VLM mở năm 2026.

### Số lượng tham số cho một ViT

Phép tính đầy đủ nằm trong `code/main.py`. Đối với ViT-B/16 ở 224:

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

Hãy ước tính mọi ViT theo cách này trước khi bạn tải checkpoint. Kích thước backbone thiết lập mức sàn VRAM của bạn trong bất kỳ VLM hạ nguồn nào.

### Cấu hình sản xuất năm 2026

Bộ mã hóa mà hầu hết các VLM mở sử dụng vào năm 2026 là SigLIP 2 SO400m/14 ở độ phân giải gốc (NaFlex). Nó có:
- 400M tham số.
- Kích thước patch 14, độ phân giải mặc định 384 → 729 patch token mỗi hình ảnh.
- Mean pool cho các tác vụ cấp độ hình ảnh; tất cả 729 patch chảy vào LLM cho VQA.
- 4 register token, bị loại bỏ trước khi chuyển giao cho LLM.
- 2D-RoPE với tỷ lệ cấp độ hình ảnh cho tỷ lệ khung hình gốc.

Mọi quyết định trong cấu hình đó đều bắt nguồn từ một bài báo mà bạn có thể đọc.

```figure
image-patch-tokens
```

## Sử dụng nó

`code/main.py` là một trình tạo patch token và máy tính hình học. Nó lấy (ảnh H, W, patch P, hidden D, độ sâu L) và báo cáo:

- Hình dạng lưới và độ dài chuỗi sau khi chia patch.
- Chuỗi token cho một hình ảnh đồ chơi 8x8 pixel tổng hợp (đi qua đường dẫn làm phẳng + chiếu).
- Số lượng tham số được phân tích theo patch embed, position embed, các khối transformer và head.
- FLOPs mỗi lần forward pass ở độ phân giải mục tiêu.
- Bảng so sánh giữa ViT-B/16 @ 224, ViT-L/14 @ 336, DINOv2 ViT-g/14 @ 224, SigLIP SO400m/14 @ 384.

Hãy chạy nó. Khớp số lượng tham số với các con số đã công bố. Thử nghiệm với kích thước patch và độ phân giải để cảm nhận chi phí về số lượng token.

## Triển khai nó

Bài học này tạo ra `outputs/skill-patch-geometry-reader.md`. Với một cấu hình ViT (kích thước patch, độ phân giải, hidden dim, độ sâu), nó tạo ra số lượng token, số lượng tham số và ước tính VRAM kèm theo lý do. Hãy sử dụng kỹ năng này bất cứ khi nào bạn chọn một vision backbone cho VLM — nó ngăn chặn những bất ngờ kiểu "các token bùng nổ và ngữ cảnh LLM của tôi bị đầy".

## Bài tập

1. Tính độ dài chuỗi patch-token cho Qwen2.5-VL ở đầu vào gốc 1280x720 với kích thước patch 14. Điều đó so sánh thế nào với biểu diễn chỉ có CLS?

2. Một khung hình 1080p (1920x1080) ở patch 14 tạo ra bao nhiêu token? Ở 30 FPS trong video 5 phút, tổng cộng có bao nhiêu visual token? Chi phí nào giúp bạn tiết kiệm nhiều nhất: pooling, lấy mẫu khung hình (frame sampling) hay gộp token (token merging)?

3. Triển khai mean pooling trên các patch token bằng Python thuần. Xác minh rằng mean-pool trên 196 token của đầu ra DINOv2 khớp với những gì `forward` của mô hình trả về khi bạn yêu cầu một pooled embedding.

4. Đọc Phần 3 của "Vision Transformers Need Registers" (arXiv:2309.16588). Mô tả trong hai câu về nhiễu (artifact) mà các register hấp thụ và tại sao nó quan trọng đối với dự đoán dày đặc hạ nguồn.

5. Sửa đổi `code/main.py` để hỗ trợ patch-n'-pack: với một danh sách các hình ảnh có độ phân giải khác nhau, tạo ra một chuỗi đóng gói duy nhất và mặt nạ attention khối chéo (block-diagonal attention mask). Xác minh với Bài học 12.06 khi bạn đạt đến đó.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Patch | "Hình vuông 16x16 pixel" | Một vùng không chồng lấp có kích thước cố định của hình ảnh đầu vào; trở thành một token |
| Patch embedding | "Phép chiếu tuyến tính" | Một ma trận học được chia sẻ (hoặc Conv2d với stride=P) ánh xạ các pixel patch đã làm phẳng thành các vector D-dim |
| CLS token | "Class token" | Vector có thể học được được thêm vào đầu, trạng thái ẩn cuối cùng đại diện cho toàn bộ hình ảnh; tùy chọn vào năm 2026 |
| Register token | "Sink token" | Các token có thể học được bổ sung giúp hấp thụ các nhiễu attention có chuẩn cao mà ViT phát triển trong quá trình tiền huấn luyện |
| Position embedding | "Thông tin vị trí" | Vector hoặc phép xoay theo vị trí giúp chuỗi nhận biết thứ tự; 2D-RoPE là mặc định hiện đại |
| Grid | "Lưới patch" | Mảng 2D (H/P) x (W/P) các patch cho một độ phân giải và kích thước patch nhất định |
| NaFlex | "Độ phân giải linh hoạt gốc" | Tính năng của SigLIP 2: một mô hình duy nhất phục vụ nhiều tỷ lệ khung hình và độ phân giải mà không cần huấn luyện lại |
| Backbone | "Vision tower" | Bộ mã hóa hình ảnh đã được tiền huấn luyện có đầu ra patch-token cung cấp cho LLM trong một VLM |
| Pooling | "Tóm tắt cấp độ hình ảnh" | Chiến lược biến các patch token thành một vector: CLS, mean, attention pool hoặc dựa trên register |
| Patch 14 vs 16 | "Lưới mịn hơn vs thô hơn" | Patch 14 tạo ra nhiều token hơn trên mỗi hình ảnh, độ trung thực tốt hơn cho OCR, chậm hơn; patch 16 là mặc định cổ điển |

## Đọc thêm

- [Dosovitskiy và cộng sự — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — ViT gốc.
- [He và cộng sự — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE, tiền huấn luyện tự giám sát.
- [Oquab và cộng sự — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — tự chưng cất ở quy mô lớn, không nhãn.
- [Darcet và cộng sự — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register tokens và phân tích nhiễu.
- [Tschannen và cộng sự — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — vision tower mặc định năm 2026.
- [Zhai và cộng sự — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — các định luật mở rộng thực nghiệm.