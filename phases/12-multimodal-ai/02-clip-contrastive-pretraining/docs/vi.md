# CLIP và Contrastive Vision-Language Pretraining

> CLIP của OpenAI (2021) đã chứng minh một ý tưởng đủ lớn để thúc đẩy 5 năm tiếp theo: căn chỉnh (align) một bộ mã hóa hình ảnh (image encoder) và một bộ mã hóa văn bản (text encoder) trong cùng một không gian vector, chỉ sử dụng các cặp hình ảnh-chú thích nhiễu từ web và hàm mất mát tương phản (contrastive loss). Không cần nhãn giám sát. 400 triệu cặp dữ liệu. Không gian embedding thu được có khả năng phân loại zero-shot, truy xuất hình ảnh-văn bản và được tích hợp vào mọi VLM năm 2026 như một "vision tower". SigLIP 2 (2025) đã thay thế softmax bằng sigmoid và mở rộng quy mô vượt qua CLIP với chi phí thấp hơn. Bài học này sẽ đi qua các công thức toán học từ InfoNCE đến sigmoid pairwise loss và xây dựng bước huấn luyện bằng Python stdlib.

**Type:** Build
**Languages:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 7 (Transformers)
**Time:** ~180 phút

## Mục tiêu học tập

- Suy luận hàm mất mát InfoNCE từ thông tin tương hỗ (mutual information) và triển khai phiên bản vector hóa ổn định về mặt số học.
- Giải thích lý do tại sao sigmoid pairwise loss (SigLIP) có thể mở rộng quy mô lên batch size 32768+ mà không cần chi phí all-gather như softmax.
- Thực hiện phân loại ImageNet zero-shot bằng cách xây dựng các mẫu văn bản (`a photo of a {class}`) và lấy argmax trên độ tương đồng cosine.
- Nêu tên bốn đòn bẩy mà quá trình tiền huấn luyện CLIP / SigLIP cung cấp cho bạn: batch size, nhiệt độ (temperature), mẫu prompt, và chất lượng dữ liệu.

## Vấn đề

Thị giác máy tính trước CLIP là có giám sát. Thu thập các tập dữ liệu được gán nhãn (ImageNet: 1.2 triệu hình ảnh, 1000 lớp), huấn luyện một CNN, và triển khai. Nhãn rất đắt đỏ, nhãn bị thiên kiến bởi những gì người gán nhãn có thể đồng thuận, và nhãn không chuyển đổi sang các tác vụ mới nếu không tinh chỉnh (finetuning).

Web hình ảnh-chú thích có hơn một tỷ cặp dữ liệu được gán nhãn lỏng lẻo miễn phí. Một bức ảnh chú chó golden retriever với văn bản thay thế "chú chó Max của tôi trong công viên" mang một tín hiệu giám sát — văn bản mô tả hình ảnh. Câu hỏi đặt ra là: bạn có thể biến điều này thành dữ liệu huấn luyện hữu ích không?

Câu trả lời của CLIP: coi các cặp hình ảnh-chú thích là một tác vụ khớp (matching task). Với một batch gồm N hình ảnh và N chú thích, hãy học cách khớp mỗi hình ảnh với chú thích tương ứng của nó so với N-1 yếu tố gây nhiễu. Sự giám sát ở đây là "hai thứ này thuộc về nhau; N-1 thứ kia thì không". Không nhãn lớp. Không chú thích của con người. Chỉ là một hàm mất mát tương phản.

Không gian embedding thu được làm được nhiều việc hơn những gì CLIP được huấn luyện. Phân loại ImageNet zero-shot hoạt động vì "một bức ảnh về con mèo" được nhúng gần với các bức ảnh về mèo chưa bao giờ được gán nhãn rõ ràng là mèo. Đây là canh bạc đã tạo ra mọi VLM năm 2026.

## Khái niệm

### Bộ mã hóa kép (Dual encoder)

CLIP có hai "tòa tháp":

- Image encoder `f`: ViT hoặc ResNet, xuất ra một vector D-chiều cho mỗi hình ảnh.
- Text encoder `g`: transformer nhỏ, xuất ra một vector D-chiều cho mỗi chú thích.

Cả hai bộ mã hóa đều chuẩn hóa đầu ra của chúng về độ dài đơn vị. Độ tương đồng là `cos(f(x), g(y)) = f(x)^T g(y)` vì cả hai đều có chuẩn đơn vị.

Đối với một batch gồm N cặp (hình ảnh, chú thích), hãy xây dựng ma trận tương đồng `S` có hình dạng `(N, N)`:

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

trong đó `tau` là một nhiệt độ (temperature) đã học (CLIP khởi tạo ở mức 0.07; được học trong không gian log).

### Hàm mất mát InfoNCE

CLIP sử dụng cross-entropy đối xứng trên các hàng và cột:

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

Đây là InfoNCE. Softmax trong CE buộc mỗi hình ảnh phải khớp với chú thích của nó nhiều hơn so với mọi chú thích khác trong batch. Các "phần tử âm" (negatives) là tất cả các mục khác trong batch. Batch càng lớn = càng nhiều phần tử âm = tín hiệu càng mạnh. CLIP huấn luyện ở batch 32k; quy mô rất quan trọng.

### Nhiệt độ (Temperature)

`tau` kiểm soát độ sắc nét của softmax. Tau thấp → phân phối sắc nét, hiệu ứng khai thác phần tử âm khó (hard negative mining). Tau cao → mềm, tất cả các mẫu đều đóng góp. CLIP học log(1/tau), được cắt (clipped) để tránh sụp đổ. SigLIP 2 cố định tau ban đầu và sử dụng một bias đã học thay thế.

### Tại sao sigmoid mở rộng quy mô tốt hơn (SigLIP)

Softmax cần toàn bộ ma trận tương đồng được đồng bộ. Trong huấn luyện phân tán, bạn phải all-gather mọi embedding tới mọi bản sao, sau đó mới thực hiện softmax. Điều này có độ phức tạp bậc hai theo quy mô hệ thống đối với giao tiếp.

SigLIP thay thế softmax bằng sigmoid theo từng phần tử: đối với mỗi cặp `(i, j)`, hàm mất mát là một phân loại nhị phân "đây có phải là cặp khớp nhau không?". Các nhãn lớp dương là đường chéo, mọi thứ khác là âm. Hàm mất mát là:

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` nếu `i == j`, ngược lại là 0. Hàm mất mát của mỗi cặp là độc lập. Không cần all-gather. Mỗi GPU tính toán khối cục bộ của nó và cộng lại. SigLIP 2 mở rộng quy mô lên batch 32k-512k với chi phí thấp, trong khi CLIP sẽ cần giao tiếp nhiều hơn tương ứng.

### Phân loại Zero-shot

Với N tên lớp, đối với mỗi lớp hãy xây dựng một mẫu văn bản:

```
"a photo of a {class}"
```

Nhúng mỗi mẫu bằng text encoder. Nhúng hình ảnh của bạn bằng image encoder. Argmax độ tương đồng cosine = lớp dự đoán. Không cần huấn luyện trên các lớp mục tiêu.

Các mẫu prompt rất quan trọng. Bài báo gốc của CLIP sử dụng 80 mẫu mỗi lớp (thông thường, nghệ thuật, ảnh chụp, tranh vẽ, v.v.) và lấy trung bình các embedding. +3 điểm ImageNet. Cách sử dụng hiện đại thường chọn một hoặc hai mẫu.

### Linear probes và tinh chỉnh (Finetuning)

Zero-shot là một baseline. Một linear probe (huấn luyện một lớp tuyến tính trên các đặc trưng CLIP đã đóng băng cho các lớp mục tiêu của bạn) vượt qua zero-shot trên các tác vụ trong miền (in-domain). Tinh chỉnh toàn bộ (full finetuning) vượt qua linear probe trên in-domain nhưng có thể làm giảm khả năng chuyển đổi zero-shot. Ba chế độ với ba sự đánh đổi.

### SigLIP 2: NaFlex và các đặc trưng dày đặc (dense features)

SigLIP 2 (2025) bổ sung:
- NaFlex: mô hình đơn lẻ xử lý các tỷ lệ khung hình và độ phân giải biến đổi.
- Các đặc trưng dày đặc tốt hơn cho phân đoạn (segmentation) và ước tính độ sâu (depth estimation), nhắm mục tiêu sử dụng như một backbone đóng băng trong các VLM.
- Đa ngôn ngữ: huấn luyện trên hơn 100 ngôn ngữ trong khi CLIP chỉ có tiếng Anh.
- Quy mô 1 tỷ tham số trong khi CLIP đạt đỉnh ở 400 triệu.

Trong các VLM mở năm 2026, SigLIP 2 SO400m/14 là vision tower mặc định. CLIP vẫn là mặc định cho việc truy xuất hình ảnh-văn bản thuần túy nơi phân phối huấn luyện LAION-2B cụ thể khớp với mẫu truy vấn của bạn.

### ALIGN, BASIC, OpenCLIP, EVA-CLIP

ALIGN (Google, 2021): cùng ý tưởng với CLIP, quy mô 1.8 tỷ cặp, 90% nhiễu. Chứng minh dữ liệu nhiễu có thể mở rộng. OpenCLIP (LAION): bản tái tạo mở của CLIP trên LAION-400M / 2B, nhiều quy mô, checkpoint mở phổ biến. EVA-CLIP: khởi tạo từ masked image modeling; backbone mạnh cho VLM. BASIC: sự kết hợp CLIP+ALIGN của Google. Tất cả đều cùng một gia đình, khác nhau về dữ liệu và tinh chỉnh.

### Giới hạn Zero-shot

Các mô hình lớp CLIP đạt đỉnh khoảng 76% ImageNet zero-shot (CLIP-G, OpenCLIP-G). Vượt qua con số này đòi hỏi dữ liệu lớn hơn nhiều (SigLIP 2 đạt 80%+) hoặc thay đổi kiến trúc (các đầu ra có giám sát, nhiều tham số hơn). Benchmark đang bão hòa; giá trị thực sự nằm ở không gian embedding mà các VLM hạ nguồn tiêu thụ.

```figure
multimodal-fusion
```

## Sử dụng

`code/main.py` triển khai:

1. Một bộ mã hóa kép đồ chơi (đặc trưng hình ảnh dựa trên hash, đặc trưng ký tự văn bản) để bạn có thể thấy hình dạng InfoNCE mà không cần numpy.
2. Hàm mất mát InfoNCE bằng Python thuần (ổn định số học thông qua log-sum-exp).
3. Sigmoid pairwise loss để so sánh.
4. Một quy trình phân loại zero-shot: tính độ tương đồng cosine với một tập hợp các prompt văn bản, argmax để dự đoán.

Hãy chạy nó và quan sát đường cong mất mát. Các con số tuyệt đối là đồ chơi; hình dạng khớp với những gì một trình huấn luyện CLIP thực tế tạo ra.

## Triển khai

Bài học này tạo ra `outputs/skill-clip-zero-shot.md`. Với một tập hợp hình ảnh (thông qua đường dẫn) và một danh sách các lớp mục tiêu, nó xây dựng các prompt văn bản với mẫu CLIP, nhúng cả hai phía với một checkpoint đã nêu (ví dụ: `openai/clip-vit-large-patch14`), và trả về các dự đoán top-1 / top-5 với điểm tương đồng. Kỹ năng này từ chối đưa ra các tuyên bố về các lớp không có trong danh sách prompt.

## Bài tập

1. Triển khai InfoNCE cho một batch gồm 4 cặp bằng tay. Xây dựng ma trận tương đồng 4x4, chạy softmax, chọn đường chéo, tính cross-entropy. Xác minh triển khai Python của bạn so với tính toán tay này.

2. SigLIP sử dụng tham số bias `b` ngoài nhiệt độ: `S'[i,j] = S[i,j]/tau + b`. Vai trò của `b` là gì khi batch có sự mất cân bằng lớp lớn (nhiều phần tử âm hơn phần tử dương trên mỗi hàng)? Đọc SigLIP Phần 3 (arXiv:2303.15343).

3. Xây dựng bộ phân loại zero-shot cho mèo vs chó. Thử hai mẫu prompt: `a photo of a {class}` và `a picture of a {class}`. Đo độ chính xác trên 100 hình ảnh kiểm tra. Liệu tập hợp các mẫu (ensemble) có vượt qua mẫu đơn lẻ không?

4. Tính chi phí giao tiếp của softmax InfoNCE so với sigmoid pairwise cho một lần chạy 512-GPU tại batch 32k. Cái nào mở rộng theo O(N), cái nào theo O(N^2)? Trích dẫn SigLIP Phần 4.

5. Đọc bài báo về định luật mở rộng OpenCLIP (arXiv:2212.07143, Cherti et al.). Tái tạo kết luận của họ về việc mở rộng dữ liệu từ các hình vẽ: ở kích thước mô hình cố định, mối quan hệ log-tuyến tính giữa độ chính xác ImageNet zero-shot và kích thước dữ liệu huấn luyện là gì?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| InfoNCE | "Contrastive loss" | Cross-entropy trên ma trận tương đồng của batch; phần tử dương của mỗi mục là mục được ghép cặp với nó, các phần tử âm là mọi thứ khác |
| Sigmoid loss | "SigLIP loss" | Binary cross-entropy theo cặp; không softmax, không all-gather, mở rộng rẻ trong huấn luyện phân tán |
| Temperature | "tau" | Scalar chia tỷ lệ logit trước softmax/sigmoid; kiểm soát độ sắc nét của phân phối |
| Zero-shot | "no-finetune classification" | Sử dụng prompt văn bản để xây dựng embedding lớp và phân loại bằng độ tương đồng cosine; không huấn luyện trên các lớp mục tiêu |
| Prompt template | "a photo of a ..." | Khung văn bản xung quanh tên lớp; ảnh hưởng đến độ chính xác zero-shot từ 1-5 điểm |
| Dual encoder | "Two-tower" | Một image encoder + một text encoder, đầu ra trong không gian D-chiều chia sẻ |
| Hard negative | "Tough distractor" | Một phần tử âm đủ giống với phần tử dương khiến mô hình phải nỗ lực để tách biệt chúng |
| Linear probe | "Frozen + one layer" | Chỉ huấn luyện một bộ phân loại tuyến tính trên các đặc trưng đã đóng băng; đo lường chất lượng đặc trưng |
| NaFlex | "Native flexible resolution" | Khả năng của SigLIP 2 để tiếp nhận hình ảnh ở bất kỳ tỷ lệ khung hình và độ phân giải nào mà không cần thay đổi kích thước |
| Temperature scaling | "log-parametrized tau" | CLIP tham số hóa `log(1/tau)` để gradient hoạt động ổn định; cắt để tránh sụp đổ về tau gần bằng 0 |

## Đọc thêm

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — bài báo CLIP.
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — đa ngôn ngữ + NaFlex.
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — mở rộng với dữ liệu web nhiễu.
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — định luật mở rộng OpenCLIP.