# Vision Encoder Patches

> Một vision model đọc các pixel cần một tokenizer cho pixel. Patch embedding chính là tokenizer đó. Cắt hình ảnh thành một lưới các ô vuông, làm phẳng (flatten) mỗi ô, chiếu (project) nó qua một lớp linear, sau đó thêm tín hiệu vị trí 2D để transformer biết mỗi ô nằm ở đâu trong hình ảnh gốc.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 lessons 30-37 (Track B foundations)
**Time:** ~90 minutes

## Learning Objectives

- Tokenize một hình ảnh thành một chuỗi các patch embedding có độ dài cố định.
- Triển khai một patch projection dựa trên `Conv2d` khớp với toán học của unfold-then-linear.
- Xây dựng một 2D sinusoidal position embedding xác định (deterministic) để thứ tự token mã hóa vị trí không gian.
- Xác minh số lượng patch, shape của embedding, và sự tương đương giữa `Conv2d`/unfold trên một fixture tổng hợp.

## The Problem

Một transformer nhận vào một chuỗi các vector. Một hình ảnh là một lưới 3 kênh. Việc đọc mọi pixel như một token sẽ làm bùng nổ độ dài chuỗi: một hình ảnh RGB 224x224 là 150,528 token, điều mà một transformer 12 lớp không thể đáp ứng được trong attention. Việc đọc hình ảnh như một vector phẳng khổng lồ sẽ vứt bỏ tính cục bộ (locality), thứ mà lớp attention không thể khôi phục lại được. Nhiệm vụ của encoder front end là nén lưới pixel thành vài trăm token, mỗi token tóm tắt một vùng hình vuông.

Patch embedding giải quyết vấn đề này bằng một phép chiếu tuyến tính (linear projection). Một hình ảnh 224x224 được cắt thành các patch 16x16 sẽ tạo ra một lưới 14x14 gồm 196 patch. Mỗi patch được làm phẳng từ `(3, 16, 16) = 768` giá trị pixel thành một vector, sau đó một lớp linear sẽ ánh xạ nó vào hidden dimension của model. Transformer sẽ thấy 196 token có dimension `hidden` (thường là 768) cộng với một CLS token. Đó là một chuỗi mà phần còn lại của mạng có thể xử lý được.

## The Concept

```mermaid
flowchart LR
  Image[224x224x3 image] --> Cut[cut into 16x16 patches]
  Cut --> Grid[14x14 grid of patches]
  Grid --> Flatten[flatten each patch]
  Flatten --> Proj[linear projection]
  Proj --> Tokens[196 tokens of dim hidden]
  Tokens --> Pos[add 2D sinusoidal position]
  Pos --> Out[final token sequence]
```

### Tại sao dùng patch, không dùng pixel

Attention có độ phức tạp bậc hai theo độ dài chuỗi. Một chuỗi 196 token tiêu tốn `196 * 196 = 38,416` điểm attention cho mỗi head trên mỗi lớp; một chuỗi 150,528 token tiêu tốn `150,528 * 150,528 = 22.6 billion`. Patch giúp giảm 590,000 lần tính toán attention, và một vùng 16x16 duy nhất mang đủ tín hiệu cho các tác vụ vision cấp cao. Cái giá phải trả là mất đi chi tiết không gian mịn bên trong một patch, đó là lý do tại sao các multimodal stack hạ nguồn thường chạy một nhánh độ phân giải cao thứ hai khi việc định vị chính xác (fine localization) là quan trọng.

### Tại sao một linear projection là đủ

Mỗi patch được xử lý như một vector độc lập. Phép chiếu học một cơ sở (basis): các bộ dò cạnh (edge detectors), bộ lọc màu, kết cấu đơn giản. Một lớp linear duy nhất là rất nhỏ (`768 * 768 = 589,824` tham số cho ViT-Base) và huấn luyện nhanh. Các convolutional stem sâu hơn có tồn tại (ViT "hybrid"), nhưng một linear projection phẳng là tiêu chuẩn, và hầu hết các open-weight encoder hiện đại đều xuất xưởng với cấu trúc chính xác này.

### Mẹo `Conv2d`

Một `Conv2d(in_channels=3, out_channels=hidden, kernel_size=patch_size, stride=patch_size)` không có padding cho kết quả số học tương tự như unfold-then-linear, bởi vì mỗi vị trí đầu ra thực hiện dot-product các pixel của patch với một bộ lọc. Convolution chính là patch projection, và hầu hết các codebase production triển khai theo cách này vì nó nhanh hơn trên GPU và bớt đi một bước reshape.

### Position embeddings

Các token không mang thứ tự sau phép chiếu. 2D sinusoidal embedding cung cấp cho mỗi token một tín hiệu cố định mã hóa vị trí `(row, col)` của nó. Một nửa dimension của embedding mã hóa vị trí hàng bằng sin/cos ở nhiều tần số; nửa còn lại mã hóa vị trí cột. Việc mã hóa là xác định (deterministic) nên bạn có thể thay đổi độ phân giải mà không cần huấn luyện lại, và nó nội suy (interpolate) mượt mà cho các lưới mà model chưa từng thấy trong thời gian huấn luyện.

| Thành phần | Shape | Tham số |
|-----------|-------|------------|
| Patch projection (`Conv2d`) | `(hidden, 3, patch, patch)` | `3 * P * P * hidden + hidden` |
| Position embedding (cố định) | `(num_patches, hidden)` | 0 (được tính toán, không phải học) |
| CLS token (được học) | `(1, hidden)` | `hidden` |

Đối với ViT-Base/16 ở độ phân giải 224: 590,592 tham số trong phép chiếu, 768 trong CLS token, và không có tham số cho sinusoidal position. Bài học tiếp theo (59) sẽ xếp chồng một transformer 12 lớp lên trên front end này.

### Sự tương đương như một bước kiểm tra tính đúng đắn

Bước patch có hai cách viết: một phép chiếu `Conv2d` và một phép unfold-then-linear tường minh. Chúng phải tạo ra cùng một đầu ra cho cùng một bộ trọng số. Nếu không, toán học của unfold bị sai, và phần còn lại của encoder được xây dựng trên cát. Các bài kiểm tra trong bài học này thực hành sự tương đương đó.

```figure
ch-patch-tokenizer
```

## Build It

`code/main.py` triển khai:

- `PatchEmbed`, một `nn.Module` bao bọc `Conv2d` cho patch projection.
- `sinusoidal_2d(grid_h, grid_w, dim)`, một hàm không trạng thái (stateless) xây dựng bảng vị trí 2D.
- `VisionFrontEnd`, thứ kết hợp patch embedding, thêm CLS vào đầu, và cộng vị trí vào một lượt forward pass duy nhất.
- Một `synthesize_image(seed)` helper xây dựng một fixture 224x224x3 xác định từ `numpy.random`.
- Một bản demo chạy một hình ảnh fixture qua front end và in ra shape đầu ra, norm của CLS token, và một hàng của position embedding.

Chạy nó:

```bash
python3 code/main.py
```

Đầu ra: fixture 224x224 được tokenize thành một chuỗi có shape `(1, 197, 768)`. Token đầu tiên là CLS; 196 token tiếp theo là các patch token. Các norm của position embedding là đồng nhất trong một hàng, đó là đặc trưng của sinusoidal.

## Use It

Cùng một patch front end này xuất hiện trong mọi vision-language model hiện đại: CLIP ViT-L/14, SigLIP, DINOv2, họ Qwen-VL, và InternVL stack đều bắt đầu từ một `Conv2d` patch projection cộng với một tín hiệu vị trí. Sự khác biệt giữa các họ model nằm ở phần hạ nguồn (pooling CLS so với không dùng CLS, register tokens, kích thước patch thay đổi 14 so với 16, độ phân giải động thông qua các vị trí được nội suy). Front end trong bài học này là nền tảng mà mọi model đó đều đứng trên.

## Tests

`code/test_main.py` bao gồm:

- số lượng patch khớp với `(image_size / patch_size) ** 2`
- shape đầu ra khớp với `(batch, num_patches + 1, hidden)`
- phép chiếu `Conv2d` tương đương với unfold-then-linear thủ công trên một fixture nhỏ
- bảng vị trí sinusoidal là xác định qua các lần gọi
- CLS token broadcast qua chiều batch mà không bị rò rỉ (leakage)

Chạy chúng:

```bash
python3 -m unittest code/test_main.py
```

## Exercises

1. Thay thế sinusoidal position bằng một `nn.Parameter` được học và so sánh loss ở epoch đầu tiên trên một tác vụ phân loại tổng hợp nhỏ. Vị trí được học thắng ở độ phân giải cố định; sinusoidal thắng khi bạn thay đổi độ phân giải sau khi huấn luyện.

2. Hoán đổi `Conv2d` cho một `nn.Unfold` tường minh cộng với `nn.Linear` và khẳng định (assert) các đầu ra khớp nhau trong phạm vi sai số float. Cùng một phép toán, hai cách viết.

3. Thêm hỗ trợ cho các kích thước patch không phải hình vuông (ví dụ: 32x16 cho đầu vào có tỉ lệ rộng) và xác minh bảng vị trí xử lý được các lưới không vuông.

4. Profile bước patch tại các batch size 1, 8, 64. Patch projection hiếm khi là điểm nghẽn (bottleneck); các lớp attention hạ nguồn mới là thành phần chiếm ưu thế.

5. Huấn luyện front end như một feature extractor bị đóng băng (frozen) trên tập dữ liệu hình dạng tổng hợp 4 lớp (hình tròn, hình vuông, hình tam giác, hình ngôi sao). Đầu ra của CLS token phải có khả năng phân tách tuyến tính (linearly separate).

## Key Terms

| Thuật ngữ | Ý nghĩa |
|------|---------------|
| Patch | Một vùng con hình vuông của hình ảnh, thường là 14x14 hoặc 16x16 |
| Patch embedding | Phép chiếu tuyến tính của một patch đã được làm phẳng vào hidden dim |
| Sequence length | Số lượng token sau khi tokenize patch, thường cộng thêm CLS |
| Sinusoidal position | Tín hiệu sin/cos cố định mã hóa tọa độ lưới 2D |
| CLS token | Vector được học, được thêm vào đầu chuỗi đóng vai trò là pooling head |

## Further Reading

- An Image is Worth 16x16 Words (ViT, 2021) cho khung khái niệm patch-embed ban đầu.
- Attention Is All You Need (2017) cho công thức vị trí sinusoidal được điều chỉnh sang 2D ở đây.
- Bài báo DINOv2 về register tokens, một phần mở rộng bạn có thể thêm vào như bài tập 6.