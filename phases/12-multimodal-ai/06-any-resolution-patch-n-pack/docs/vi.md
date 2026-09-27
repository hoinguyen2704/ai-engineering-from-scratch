# Any-Resolution Vision: Patch-n'-Pack và NaFlex

> Hình ảnh thực tế không phải là các hình vuông 224x224. Một hóa đơn có tỷ lệ 9:16, một biểu đồ là 16:9, ảnh chụp y tế có thể là 4096x4096, ảnh chụp màn hình điện thoại là 9:19.5. Câu trả lời của các VLM trước năm 2024 — thay đổi kích thước mọi thứ về một hình vuông cố định — đã làm mất đi các tín hiệu quan trọng giúp OCR, hiểu tài liệu và phân tích cảnh độ phân giải cao hoạt động hiệu quả. NaViT (Google, 2023) đã chứng minh rằng bạn có thể đóng gói các patch có độ phân giải thay đổi vào một batch transformer duy nhất với mặt nạ block-diagonal. M-RoPE của Qwen2-VL (2024) đã loại bỏ hoàn toàn các bảng vị trí tuyệt đối. AnyRes của LLaVA-NeXT đã chia nhỏ hình ảnh độ phân giải cao thành một ảnh cơ sở + các ảnh con. Biến thể NaFlex của SigLIP 2 (2025) hiện là bộ mã hóa mặc định cho các VLM mã nguồn mở muốn sử dụng một checkpoint duy nhất để phục vụ mọi tỷ lệ khung hình. Bài học này triển khai patch-n'-pack từ đầu đến cuối.

**Type:** Build
**Languages:** Python (stdlib, patch packer + block-diagonal mask)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 12 · 05 (LLaVA)
**Time:** ~120 phút

## Mục tiêu học tập

- Đóng gói các patch từ một batch hình ảnh có độ phân giải thay đổi vào một chuỗi duy nhất và xây dựng mặt nạ attention block-diagonal.
- Lựa chọn giữa AnyRes tiling (LLaVA-NeXT), NaFlex (SigLIP 2) và M-RoPE (Qwen2-VL) cho một tác vụ cụ thể.
- Tính toán ngân sách token cho OCR, biểu đồ và ảnh chụp mà không cần thay đổi kích thước (resize).
- Gọi tên ba chế độ lỗi của việc resize về hình vuông: văn bản bị méo, nội dung bị cắt, lãng phí token vào phần đệm (padding).

## Vấn đề

Transformers mong đợi một chuỗi (sequence). Một batch là một tập hợp các chuỗi có cùng độ dài. Nếu hình ảnh của bạn là 224x224, bạn luôn nhận được 196 patch token, không cần padding, công việc hoàn tất. Huấn luyện trên 224, suy luận trên 224, không bao giờ phải nghĩ về độ phân giải nữa.

Thế giới không vận hành như vậy. Tài liệu thường ở dạng dọc (8.5x11 inch, tỷ lệ khoảng 2:3). Ảnh chụp màn hình biểu đồ thường ở dạng ngang (16:9). Hóa đơn thường cao và hẹp (1:3). Ảnh y tế thường có kích thước 2048x2048 hoặc lớn hơn. Ảnh chụp màn hình thiết bị di động là 1170x2532 (0.46:1).

Ba lựa chọn trước năm 2024 và lý do tại sao mỗi lựa chọn đều thất bại:

1. Resize về một hình vuông cố định (224x224 hoặc 336x336). Việc nén làm méo văn bản và khuôn mặt. Việc giảm độ phân giải làm hỏng các nhãn biểu đồ và nội dung OCR. Đây là thực hành tiêu chuẩn cho đến khi LLaVA-1.5 ra đời.
2. Cắt (crop) về một tỷ lệ khung hình cố định. Bạn vứt bỏ phần lớn hình ảnh, và việc chọn vị trí cắt là một bài toán thị giác riêng biệt.
3. Padding về cạnh dài nhất. Khắc phục được sự méo mó nhưng lãng phí hơn 50% token vào phần đệm cho các hình ảnh dọc. Chi phí attention bậc hai trên tất cả các token đệm đó là rất lớn.

Câu trả lời của giai đoạn 2024-2025: để transformer xử lý các patch ở độ phân giải gốc của hình ảnh, và tìm cách đóng gói một batch không đồng nhất vào một chuỗi duy nhất mà không lãng phí tài nguyên tính toán.

## Khái niệm

### NaViT và patch-n'-pack

NaViT (Dehghani và cộng sự, 2023) là bài báo chứng minh phương pháp này hoạt động ở quy mô lớn. Ý tưởng rất cơ học:

1. Với mỗi hình ảnh trong batch, tính toán lưới patch gốc của nó tại một kích thước patch đã chọn (ví dụ: 14).
2. Làm phẳng các patch của mỗi hình ảnh thành chuỗi có độ dài thay đổi của riêng nó.
3. Nối tất cả các patch của hình ảnh thành một chuỗi dài cho batch đó.
4. Xây dựng mặt nạ attention block-diagonal để các patch của hình ảnh A chỉ chú ý (attend) trong phạm vi hình ảnh A.
5. Mang theo thông tin vị trí cho từng patch (2D RoPE hoặc fractional position embeddings).

Một batch gồm ba hình ảnh ở kích thước 336x336 (576 token), 224x224 (256 token) và 448x336 (768 token) trở thành một chuỗi 1600 token với mặt nạ block-diagonal 1600x1600. Không padding. Không lãng phí tính toán. Transformer xử lý được các tỷ lệ khung hình tùy ý.

NaViT cũng giới thiệu việc loại bỏ patch phân đoạn (fractional patch dropping) trong quá trình huấn luyện — loại bỏ ngẫu nhiên 50% số patch trên toàn batch — giúp vừa điều chuẩn (regularize) vừa tăng tốc độ huấn luyện. SigLIP 2 đã kế thừa điều này.

### AnyRes (LLaVA-NeXT)

AnyRes của LLaVA-NeXT là giải pháp thay thế thực dụng. Với một hình ảnh độ phân giải cao và một bộ mã hóa cố định (CLIP hoặc SigLIP ở 336), hãy chia nhỏ hình ảnh:

1. Chọn một bố cục lưới từ một tập hợp định sẵn — (1x1), (1x2), (2x1), (1x3), (3x1), (2x2), v.v. — phù hợp nhất với tỷ lệ khung hình của hình ảnh.
2. Chia nhỏ toàn bộ hình ảnh vào lưới; mỗi ô (tile) trở thành một bản cắt 336x336.
3. Tạo thêm một hình thu nhỏ (thumbnail): toàn bộ hình ảnh được resize về 336x336 như một token ngữ cảnh toàn cục.
4. Mã hóa mọi ô thông qua bộ mã hóa 336 đã đóng băng. Nối các token của ô + các token hình thu nhỏ.

Đối với một hình ảnh 672x672 ở lưới 2x2 cộng với hình thu nhỏ: 4 * 576 + 576 = 2880 visual token. Đắt đỏ nhưng hiệu quả — LLM nhìn thấy cả chi tiết cục bộ và ngữ cảnh toàn cục.

AnyRes là lựa chọn khi bộ mã hóa của bạn bị đóng băng và chỉ hỗ trợ một độ phân giải. Nó làm bùng nổ số lượng token đối với các hình ảnh lớn (một hình ảnh 1344x1344 ở lưới 4x4 là 9216 + 576 ≈ 9800 token, lấp đầy hầu hết ngữ cảnh 8k của LLM).

### M-RoPE (Qwen2-VL)

Qwen2-VL giới thiệu Multimodal Rotary Position Embedding. Thay vì các vị trí phân đoạn của NaViT hay cách chia ô và hình thu nhỏ của AnyRes, mỗi patch mang một vị trí 3D (thời gian, chiều cao, chiều rộng). Các phép quay query/key xử lý được H, W và độ dài thời gian tùy ý.

M-RoPE cung cấp độ phân giải động gốc mà không cần huấn luyện lại. Khi suy luận, bạn đưa vào bất kỳ hình ảnh HxW nào, bộ nhúng patch tạo ra H/14 x W/14 token, mỗi token nhận vị trí (t=0, r=hàng, c=cột) của nó, RoPE xoay attention với tần số phù hợp, xong. Qwen2.5-VL và Qwen3-VL tiếp tục phát triển điều này. V2PE của InternVL3 cũng có ý tưởng tương tự với mã hóa thay đổi theo từng phương thức (modality).

Không giống như AnyRes, M-RoPE có số lượng token là O(H x W / P^2) ở độ phân giải gốc — không có chi phí nhân thêm do chia ô. Không giống như NaViT, nó vẫn mong đợi một hình ảnh duy nhất cho mỗi lần forward. Việc batching qua các độ phân giải vẫn cần patch-n'-pack ở phía trên.

### NaFlex (SigLIP 2)

NaFlex là chế độ native-flex của checkpoint SigLIP 2. Một mô hình duy nhất phục vụ nhiều độ dài chuỗi (256, 729, 1024 token) khi suy luận. Bên trong, nó sử dụng patch-n'-pack kiểu NaViT trong quá trình huấn luyện và các vị trí phân đoạn tuyệt đối cho mỗi patch. Điểm bán hàng: một checkpoint, chọn ngân sách token của bạn khi suy luận dựa trên tác vụ.

Đối với tác vụ ngữ nghĩa (phân loại, truy xuất), 256 token. Đối với OCR hoặc hiểu biểu đồ, 1024 token. Không cần huấn luyện lại.

### Mặt nạ đóng gói (packing mask)

Mặt nạ block-diagonal là nơi hầu hết các triển khai gặp khó khăn. Đối với một chuỗi đã đóng gói có độ dài `N_total` bao phủ các hình ảnh `i=0..B-1` với độ dài `n_i`, mặt nạ `M` có hình dạng `(N_total, N_total)` sẽ là 1 nếu cả hai chỉ số nằm trong cùng một khối của hình ảnh, ngược lại là 0. Bạn có thể xây dựng nó từ danh sách độ dài tích lũy:

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

Đây là một dòng trong PyTorch với `torch.block_diag` hoặc một lệnh gather tường minh. Đường dẫn độ dài thay đổi của FlashAttention (`cu_seqlens`) bỏ qua hoàn toàn mặt nạ và chú ý trong các chuỗi bằng cách sử dụng trực tiếp tensor độ dài tích lũy — nhanh hơn khoảng 10 lần so với mặt nạ dày (dense mask) cho các batch thông thường.

### Ngân sách token

Chọn chiến lược của bạn theo tác vụ:

- OCR / tài liệu: 1024-4096 token. SigLIP 2 NaFlex ở 1024, hoặc AnyRes 3x3 + thumbnail.
- Biểu đồ và UI: 729-1024 token ở độ phân giải gốc 384-448. Qwen2.5-VL độ phân giải động với giới hạn pixel tối đa.
- Ảnh chụp tự nhiên: 256-576 token là đủ. LLM hạ nguồn nhìn thấy đủ thông tin. Hãy trả phí token ở nơi mật độ nội dung cao.
- Video: 64-128 token mỗi khung hình sau khi gộp không gian (spatial pooling), 2-8 FPS. Bài học 12.17 đề cập đến điều này.

Quy tắc sản xuất năm 2026: chọn giới hạn pixel tối đa cho mỗi tác vụ, mã hóa ở tỷ lệ khung hình gốc lên đến giới hạn đó, đóng gói batch và bỏ qua padding. Qwen2.5-VL cung cấp `min_pixels` và `max_pixels` cho chính nút điều khiển này.

```figure
mm-patch-n-pack
```

## Sử dụng

`code/main.py` triển khai patch-n'-pack cho một batch hình ảnh không đồng nhất với tọa độ pixel số nguyên. Nó:

- Nhận danh sách các kích thước hình ảnh (H, W).
- Tính toán độ dài chuỗi patch của mỗi hình ảnh ở kích thước patch 14.
- Đóng gói chúng thành một chuỗi có tổng độ dài `sum(n_i)`.
- Xây dựng mặt nạ attention block-diagonal (dạng dày, để rõ ràng).
- So sánh chi phí đóng gói với việc resize hình vuông và chia ô AnyRes.
- In bảng ngân sách token cho một batch hỗn hợp (hóa đơn, biểu đồ, ảnh chụp màn hình, ảnh chụp).

Hãy chạy nó. Những con số thu được chính là lý do tại sao mọi VLM mã nguồn mở năm 2026 đều sử dụng patch-n'-pack.

## Triển khai

Bài học này tạo ra `outputs/skill-resolution-budget-planner.md`. Với khối lượng công việc có tỷ lệ khung hình hỗn hợp (OCR, biểu đồ, ảnh chụp, khung hình video) và tổng ngân sách token, nó chọn chiến lược phù hợp (NaFlex, AnyRes, M-RoPE hoặc hình vuông cố định) và đưa ra cấu hình cho mỗi yêu cầu. Hãy sử dụng kỹ năng này khi bạn định cỡ một VLM cho sản phẩm — nó ngăn chặn sự bùng nổ token 10 lần âm thầm làm hỏng ngân sách độ trễ.

## Bài tập

1. Một hóa đơn là 600x1500 (1:2.5). Tại kích thước patch 14, có bao nhiêu token độ phân giải gốc? Sau khi resize hình vuông về 336 thì còn bao nhiêu? Cái nào làm mất độ chính xác OCR nhiều hơn trong thực tế?

2. Xây dựng mặt nạ block-diagonal cho một batch gồm bốn hình ảnh với độ dài 256, 576, 729, 1024. Xác minh ma trận attention là 2585x2585 và có chính xác `256^2 + 576^2 + 729^2 + 1024^2` mục khác không.

3. Đối với hình ảnh 1792x896 tại patch 14, hãy so sánh: (a) resize hình vuông về 336 rồi mã hóa, (b) AnyRes 2x1 + thumbnail, (c) M-RoPE ở độ phân giải gốc. Cái nào sử dụng ít token nhất? Cái nào bảo toàn nhiều chi tiết nhất?

4. Triển khai việc loại bỏ patch phân đoạn: với một chuỗi đã đóng gói, loại bỏ 50% token ngẫu nhiên đồng nhất và cập nhật mặt nạ block-diagonal tương ứng. Đo lường sự thay đổi độ thưa (sparsity) của mặt nạ.

5. Đọc Mục 3.2 của bài báo Qwen2-VL (arXiv:2409.12191). Mô tả trong hai câu `min_pixels` và `max_pixels` kiểm soát điều gì và tại sao cả hai giới hạn đều quan trọng.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Patch-n'-pack | "Đóng gói kiểu NaViT" | Nối các chuỗi patch có độ dài thay đổi từ các hình ảnh khác nhau vào một chiều batch |
| Block-diagonal mask | "Mặt nạ đóng gói" | Mặt nạ attention giới hạn các patch của mỗi hình ảnh chỉ chú ý đến chính chúng, không chú ý đến các hình ảnh lân cận trong gói |
| AnyRes | "Chia ô LLaVA-NeXT" | Chia hình ảnh độ phân giải cao thành lưới các ô kích thước cố định cộng với hình thu nhỏ toàn cục; mã hóa mọi ô bằng bộ mã hóa cố định |
| NaFlex | "SigLIP 2 native-flex" | Checkpoint SigLIP 2 duy nhất phục vụ ngân sách 256/729/1024-token khi suy luận mà không cần huấn luyện lại |
| M-RoPE | "Multimodal RoPE" | Mã hóa vị trí xoay 3D (thời gian, hàng, cột) xử lý H, W, T tùy ý mà không cần bảng vị trí |
| cu_seqlens | "Đóng gói FlashAttention" | Tensor độ dài tích lũy mà đường dẫn varlen của FlashAttention sử dụng thay vì mặt nạ block-diagonal dày |
| min_pixels / max_pixels | "Giới hạn độ phân giải" | Các nút điều khiển theo yêu cầu của Qwen2.5-VL giới hạn số lượng token trên các đầu vào rất nhỏ hoặc rất lớn |
| Visual token budget | "Số token mỗi hình ảnh" | Số lượng token patch ước tính được phát ra cho mỗi hình ảnh; thiết lập ngân sách prompt và chi phí attention của LLM |

## Đọc thêm

- [Dehghani và cộng sự — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang và cộng sự — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon và cộng sự — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen và cộng sự — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)