# Flamingo và Gated Cross-Attention cho Few-Shot VLM

> Flamingo (2022) của DeepMind đã làm được hai điều mà chưa ai làm trước đó. Thứ nhất, nó cho thấy một mô hình duy nhất có thể xử lý các chuỗi hình ảnh, video và văn bản xen kẽ tùy ý. Thứ hai, nó chứng minh các VLM có khả năng học trong ngữ cảnh (in-context learning) — chỉ cần cung cấp một prompt few-shot với ba cặp (hình ảnh, chú thích) ví dụ, mô hình có thể chú thích cho một hình ảnh mới mà không cần bất kỳ bước gradient nào. Cơ chế cốt lõi: các lớp gated cross-attention, được chèn vào giữa các lớp hiện có của LLM đã đóng băng (frozen), với một cổng tanh được học bắt đầu từ giá trị bằng 0 để đảm bảo khả năng văn bản của LLM được bảo toàn tại thời điểm khởi tạo. Bài học này sẽ đi sâu vào kiến trúc Perceiver resampler và gated cross-attention của Flamingo — tiền thân của các đầu vào xen kẽ trong Gemini và các visual token trong Idefics2.

**Type:** Learn
**Languages:** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**Prerequisites:** Phase 12 · 03 (BLIP-2 Q-Former)
**Time:** ~120 phút

## Mục tiêu học tập

- Giải thích cách gated cross-attention bảo toàn khả năng văn bản của một LLM đã đóng băng tại thời điểm khởi tạo thông qua tanh(gate) = 0.
- Tìm hiểu về Perceiver resampler: N patch hình ảnh → K truy vấn "latent" cố định thông qua cross-attention.
- Mô tả cách Flamingo xử lý các chuỗi hình ảnh-văn bản xen kẽ với cơ chế causal masking tuân thủ vị trí hình ảnh.
- Tái tạo cấu trúc prompt đa phương thức few-shot (3 ví dụ hình ảnh-chú thích theo sau là một hình ảnh truy vấn).

## Vấn đề

BLIP-2 đưa 32 visual token vào lớp đầu vào của một LLM đã đóng băng. Cách này hiệu quả cho một hình ảnh mỗi prompt. Nhưng nếu bạn muốn đưa *nhiều* hình ảnh xen kẽ với văn bản, ví dụ như "đây là hình ảnh A, hãy chú thích nó; đây là hình ảnh B, hãy chú thích nó; bây giờ đây là hình ảnh C, hãy chú thích nó"? Cơ chế self-attention của LLM sẽ phải xử lý cả token hình ảnh và token văn bản trong một luồng duy nhất, và câu hỏi về việc vị trí nào có thể chú ý (attend) đến hình ảnh nào trở nên phức tạp.

Câu trả lời của Flamingo: không thay đổi luồng đầu vào của LLM. Chèn thêm các lớp cross-attention vào giữa các khối LLM hiện có. Các token văn bản vẫn đi qua causal self-attention của LLM như bình thường. Giữa mỗi vài khối LLM, các token văn bản cũng thực hiện cross-attend với các đặc trưng hình ảnh thông qua một lớp gated mới. Cổng (được khởi tạo bằng 0) có nghĩa là tại bước 0, các lớp mới này là no-op — mô hình hoạt động chính xác như LLM đã được huấn luyện trước đó. Khi quá trình huấn luyện tiến triển, cổng mở ra và thông tin hình ảnh bắt đầu được truyền vào.

Câu hỏi thứ hai mà Flamingo giải quyết: làm thế nào để xử lý số lượng hình ảnh biến đổi (0, 1, hoặc nhiều) trong mỗi prompt? Đó là Perceiver resampler — một mô-đun cross-attention nhỏ nhận vào bất kỳ số lượng patch nào bạn có và tạo ra một số lượng cố định các visual latent token. Lớp cross-attention của LLM sẽ thấy cùng một kích thước bất kể có bao nhiêu hình ảnh trong prompt.

## Khái niệm

### LLM đã đóng băng (Frozen LLM)

Flamingo bắt đầu với một LLM Chinchilla 70B đã đóng băng. Tất cả 70B trọng số đều không bị thay đổi. Các lớp text self-attention và FFN hiện có hoạt động bình thường.

### Perceiver resampler

Đối với mỗi hình ảnh trong prompt, ViT tạo ra N patch token. Perceiver resampler có K latent cố định có thể học được (Flamingo sử dụng K=64). Mỗi khối resampler gồm hai bước phụ:

1. Cross-attention: K latent attend trên N patch token (Q từ latent, K/V từ patch).
2. Self-attention + FFN bên trong các latent.

Sau 6 khối resampler, đầu ra là K=64 visual token với số chiều 1024, bất kể ViT tạo ra bao nhiêu patch. Một hình ảnh 224x224 (196 patch) và một hình ảnh 480x480 (900 patch) đều cho ra 64 token resampler.

Đối với video, resampler được áp dụng theo thời gian: các patch của mỗi khung hình tạo ra 64 latent, và một positional encoding theo thời gian cho phép mô hình phân biệt t=0 với t=N. Toàn bộ video trở thành T * 64 visual token.

### Gated cross-attention

Giữa mỗi M lớp của LLM đã đóng băng (Flamingo sử dụng M=4), chèn một khối gated cross-attention mới:

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` là một scalar có thể học được, khởi tạo bằng 0.
- `tanh(0) = 0`, vì vậy tại thời điểm khởi tạo, nhánh gated đóng góp bằng 0.
- Khi `alpha` dịch chuyển khỏi 0, sự đóng góp của cross-attention tăng lên một cách mượt mà.
- Kết nối residual (tắt dần) có nghĩa là ngay cả khi cổng mở hoàn toàn, nó cũng không ghi đè lên biểu diễn văn bản của LLM; nó chỉ bổ sung thông tin hình ảnh lên trên.

Đây là lựa chọn thiết kế quan trọng nhất trong Flamingo: điều kiện hóa hình ảnh mang tính cộng dồn, có cổng, và bằng 0 tại thời điểm khởi tạo. Một mô hình Flamingo ở bước 0 là một Chinchilla 70B hoàn hảo đối với các đầu vào chỉ có văn bản.

### Masked cross-attention cho đầu vào xen kẽ

Trong một prompt như "<image A> caption A <image B> caption B <image C> ?", mỗi token văn bản chỉ nên nhìn thấy những hình ảnh xuất hiện trước nó trong chuỗi. Cross-attention mask thực thi: token văn bản tại vị trí `t` chỉ attend đến các token resampler hình ảnh có chỉ số hình ảnh `i < i_t`, trong đó `i_t` là hình ảnh gần nhất trước vị trí `t`. "Chỉ nhìn thấy hình ảnh ngay trước đó" hoặc "nhìn thấy tất cả các hình ảnh trước đó" đều là những lựa chọn hợp lệ; Flamingo chọn cách thứ nhất.

### Học few-shot trong ngữ cảnh (In-context few-shot learning)

Một prompt Flamingo trông như sau:

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

Mô hình nhìn thấy mẫu hoàn thiện và đưa ra "bird" (hoặc bất cứ thứ gì hình ảnh 3 hiển thị). Không cần bước gradient. Khả năng học trong ngữ cảnh của LLM đã đóng băng được duy trì thông qua gated cross-attention — đây là điểm mấu chốt của bài báo và lý do tại sao nó quan trọng.

### Dữ liệu huấn luyện

Flamingo được huấn luyện trên ba tập dữ liệu:

1. MultiModal MassiveWeb (M3W): 43 triệu trang web với hình ảnh và văn bản xen kẽ, tái tạo thứ tự đọc.
2. Image-Text Pairs (ALIGN + LTIP): 4,4 tỷ cặp.
3. Video-Text Pairs (VTP): 27 triệu đoạn video ngắn.

OBELICS (2023) là bản tái tạo mở của tập dữ liệu web xen kẽ, được sử dụng để huấn luyện Idefics, Idefics2 và hầu hết các mô hình "kiểu Flamingo" mã nguồn mở.

### OpenFlamingo và Otter

OpenFlamingo (2023) là bản tái tạo mã nguồn mở. Kiến trúc giống hệt (Perceiver resampler + gated cross-attention trên LLaMA hoặc MPT đã đóng băng). Các checkpoint ở mức 3B, 4B, 9B. Chất lượng thấp hơn Flamingo do LLM cơ sở nhỏ hơn và ít dữ liệu hơn.

Otter (2023) xây dựng dựa trên OpenFlamingo với instruction tuning trên MIMIC-IT (tập dữ liệu các chỉ dẫn đa phương thức), cho thấy gated cross-attention cũng hiệu quả cho việc tuân thủ chỉ dẫn.

### Các thế hệ kế thừa

- Idefics / Idefics2 / Idefics3: Dòng dõi gated cross-attention của Hugging Face, ngày càng đơn giản hóa (Idefics2 đã loại bỏ resampler để thay bằng các patch token trực tiếp với adaptive pooling).
- Chuyển đổi Flamingo sang Chameleon: đến năm 2024, nhiều nhóm đã chuyển sang early-fusion (Bài 12.11); gated cross-attention kiểu Flamingo vẫn được sử dụng trong sản xuất khi yêu cầu đóng băng backbone.
- Đầu vào xen kẽ của Gemini: kế thừa về mặt khái niệm sự linh hoạt của định dạng xen kẽ từ Flamingo, mặc dù cơ chế chính xác là độc quyền.

### So sánh với BLIP-2

| | BLIP-2 | Flamingo |
|---|---|---|
| Visual bridge | Q-Former một lần tại đầu vào | Gated cross-attention tại mỗi M lớp |
| Visual tokens | 32 mỗi hình ảnh | 64 mỗi hình ảnh mỗi lớp cross-attn |
| Frozen LLM | Có | Có |
| Few-shot in-context | Yếu | Mạnh — trọng tâm của bài báo |
| Interleaved inputs | Không hỗ trợ gốc | Có, mục tiêu thiết kế |
| Dữ liệu huấn luyện | 130 triệu cặp | 1,3 tỷ cặp + 43 triệu trang xen kẽ |
| Số lượng tham số | 188 triệu được huấn luyện | ~10 tỷ được huấn luyện (các lớp cross-attn) |
| Tính toán | Vài ngày trên 8 A100 | Vài tuần trên hàng nghìn TPUv4 |

Chọn BLIP-2 cho VQA một hình ảnh với ngân sách hạn hẹp. Chọn Flamingo/Idefics2 cho suy luận xen kẽ, few-shot hoặc đa hình ảnh.

```figure
cross-attention-fusion
```

## Sử dụng

`code/main.py` minh họa:

1. Một Perceiver resampler trên 36 patch token giả với 8 latent có thể học được (cross-attention bằng Python thuần).
2. Một bước gated cross-attention với `alpha = 0` → đầu ra bằng đầu vào (LLM không đổi), sau đó `alpha = 2.0` → đóng góp hình ảnh được trộn vào.
3. Một trình xây dựng interleaved-mask tạo ra mask chú ý 2D cho chuỗi "(hình ảnh 1) (văn bản 1) (hình ảnh 2) (văn bản 2)".

## Triển khai

Bài học này tạo ra `outputs/skill-gated-bridge-diagnostic.md`. Với cấu hình của một VLM mở (có/không có resampler, tần suất cross-attn, lược đồ cổng), nó xác định các yếu tố dòng dõi Flamingo và giải thích chiến lược đóng băng. Hữu ích để gỡ lỗi tại sao việc fine-tune làm giảm hiệu suất văn bản (câu trả lời: cổng mở quá rộng quá nhanh).

## Bài tập

1. Tính số lượng tham số hình ảnh của Flamingo-9B: 9B LLM + 1,4B lớp gated cross-attention + 64M resampler. Tỷ lệ phần trăm tổng số tham số được huấn luyện là bao nhiêu?

2. Triển khai gated residual `y = tanh(alpha) * cross + x` trong PyTorch. Chứng minh bằng thực nghiệm rằng với `alpha=0`, `y==x` chính xác tại thời điểm khởi tạo.

3. Đọc OpenFlamingo Mục 3.2 (arXiv:2308.01390) về cách họ xử lý nhiều hình ảnh trong một batch khi mỗi prompt có số lượng hình ảnh khác nhau. Mô tả chiến lược padding.

4. Tại sao cross-attention mask của Flamingo cho phép một token văn bản chỉ attend đến *hình ảnh gần nhất* thay vì tất cả các hình ảnh trước đó? Đọc bài báo Flamingo Mục 2.4 và giải thích sự đánh đổi.

5. In-context few-shot: xây dựng một prompt với 4 ví dụ về "hình ảnh → màu sắc của vật thể chính" cho một biến thể Flamingo mới. Mô tả mô hình độ chính xác dự kiến khi bạn thay đổi số lượng ví dụ từ 0 đến 8.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Perceiver resampler | "Fixed-latent cross-attention" | Mô-đun tạo ra K token cố định từ số lượng patch đầu vào biến đổi |
| Gated cross-attention | "Tanh-gated bridge" | Lớp residual `y = tanh(alpha)*cross + x`, alpha có thể học, khởi tạo 0 |
| Interleaved input | "Mixed sequence" | Định dạng prompt với hình ảnh và văn bản trộn lẫn tự do theo thứ tự đọc |
| Frozen LLM | "No LLM gradients" | Trọng số của LLM văn bản không cập nhật; chỉ huấn luyện resampler + lớp cross-attn |
| Few-shot | "In-context examples" | Cung cấp vài cặp (hình ảnh, câu trả lời) trong prompt; mô hình tổng quát hóa mà không cần finetuning |
| OBELICS | "Interleaved web corpus" | Tập dữ liệu mở gồm 141 triệu trang web với hình ảnh và văn bản theo thứ tự đọc |
| Chinchilla | "70B frozen base" | LLM văn bản đã đóng băng của Flamingo, từ bài báo Chinchilla của DeepMind |
| Gate schedule | "How alpha moves" | Tốc độ cổng cross-attention mở ra trong quá trình huấn luyện |
| Cross-attn frequency | "Every M layers" | Tần suất chèn khối gated cross-attention; Flamingo sử dụng M=4 |
| OpenFlamingo | "Open reproduction" | Checkpoint mở của MosaicML/LAION ở mức 3-9B; kiến trúc giống hệt Flamingo |

## Đọc thêm

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — bài báo gốc.
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — bản tái tạo mở.
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — tập dữ liệu web xen kẽ.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — kiến trúc Perceiver tổng quát.
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — hậu duệ Flamingo đã được instruction-tuned.
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — sự đơn giản hóa hiện đại của phương pháp Flamingo.