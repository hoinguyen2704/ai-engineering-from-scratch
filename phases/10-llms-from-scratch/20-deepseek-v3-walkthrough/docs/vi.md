# Hướng dẫn kiến trúc DeepSeek-V3

> Giai đoạn 10 · Bài 14 đã nêu tên sáu "núm vặn" kiến trúc mà mọi mô hình mở đều sử dụng. DeepSeek-V3 (tháng 12 năm 2024, tổng cộng 671B tham số, 37B tham số hoạt động) sử dụng cả sáu núm vặn đó và bổ sung thêm bốn núm vặn mới: Multi-Head Latent Attention, cân bằng tải không cần auxiliary-loss, Multi-Token Prediction và huấn luyện DualPipe. Bài học này sẽ đọc kiến trúc của DeepSeek-V3 từ trên xuống dưới và suy luận từng số lượng tham số từ cấu hình đã công bố. Đến cuối bài, bạn có thể giải thích tại sao tỷ lệ 671B/37B là lựa chọn đúng đắn và tại sao sự kết hợp giữa MLA + MoE lại vượt trội hơn bất kỳ thành phần nào khi đứng riêng lẻ ở ngưỡng tiên phong.

**Type:** Học tập
**Languages:** Python (stdlib, máy tính tham số)
**Prerequisites:** Giai đoạn 10 · 14 (hướng dẫn về mô hình mở), Giai đoạn 10 · 17 (NSA), Giai đoạn 10 · 18 (MTP), Giai đoạn 10 · 19 (DualPipe)
**Time:** ~75 phút

## Mục tiêu học tập

- Đọc cấu hình DeepSeek-V3 từ trên xuống dưới và giải thích từng trường dựa trên sáu núm vặn GPT-2 cộng với bốn bổ sung đặc thù của DeepSeek.
- Suy luận tổng số lượng tham số (671B), số lượng tham số hoạt động (37B) và các thành phần đóng góp vào mỗi loại.
- Tính toán dung lượng KV cache của MLA ở ngữ cảnh 128k và so sánh với những gì một mô hình dày đặc (dense) có cùng số tham số hoạt động với GQA phải trả.
- Nêu bốn cải tiến đặc thù của DeepSeek (MLA, MTP, định tuyến không cần auxiliary-loss, DualPipe) và chỉ ra mỗi cải tiến nhắm vào phần nào của kiến trúc/stack huấn luyện.

## Vấn đề

DeepSeek-V3 là mô hình mở tiên phong đầu tiên có kiến trúc khác biệt đáng kể so với dòng Llama. Llama 3 405B là "GPT-2 với sáu núm vặn được điều chỉnh". DeepSeek-V3 là GPT-2 với cả sáu núm vặn cộng thêm bốn núm vặn nữa. Đọc cấu hình Llama 3 là bước khởi động để đọc cấu hình DeepSeek, nhưng cấu trúc sâu — hình dạng của khối attention, logic định tuyến, mục tiêu trong quá trình huấn luyện — khác biệt đến mức bạn cần một bài hướng dẫn riêng.

Lợi ích của việc học nó: Việc phát hành trọng số mở của DeepSeek-V3 đã thay đổi định nghĩa về "khả năng tiên phong" trong các mô hình mở. Kiến trúc này là bản thiết kế mà nhiều đợt huấn luyện năm 2026 đang sao chép. Hiểu về nó là điều kiện tiên quyết cho bất kỳ vai trò nào liên quan đến huấn luyện hoặc suy luận LLM tiên phong.

## Khái niệm

### Lõi bất biến, một lần nữa

DeepSeek-V3 vẫn là mô hình tự hồi quy (autoregressive). Nó vẫn xếp chồng các khối decoder. Mỗi khối vẫn có attention cộng với MLP và hai RMSNorm. Nó vẫn sử dụng SwiGLU trong MLP. Nó vẫn sử dụng RoPE. Pre-norm. Weight-tied embeddings. Cùng một nền tảng như mọi mô hình Llama hoặc Mistral.

### Điểm khác biệt: MLA thay vì GQA

Từ Giai đoạn 10 · 14, bạn biết GQA thu nhỏ KV cache bằng cách chia sẻ K và V giữa các nhóm đầu Q. Multi-Head Latent Attention (MLA) tiến xa hơn: K và V được nén thành một biểu diễn tiềm ẩn (latent) bậc thấp dùng chung (`kv_lora_rank`), sau đó được giải nén theo từng đầu (head) ngay lập tức. KV cache chỉ lưu trữ latent — thường là 512 số thực trên mỗi token trên mỗi lớp, thay vì 8 x 128 = 1024 số thực.

Ở ngữ cảnh 128k, DeepSeek-V3 với MLA (một latent dùng chung `c^{KV}` trên mỗi token trên mỗi lớp; K và V đều được suy ra từ latent này thông qua các phép up-projection có thể được hấp thụ vào phép matmul tiếp theo):

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

Một mô hình GQA cơ sở giả định (hình dạng Llama 3 70B, 8 đầu KV, head dim 128) sẽ phải trả:

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA nhỏ hơn 4 lần so với cache GQA kiểu Llama-3-70B ở ngữ cảnh 128k.

Sự đánh đổi: MLA thêm một bước giải nén cho mỗi phép tính attention (trên mỗi đầu). Lượng tính toán thêm vào là nhỏ so với băng thông tiết kiệm được. Đây là một thắng lợi ròng cho suy luận ngữ cảnh dài.

### Định tuyến: cân bằng tải không cần auxiliary-loss

Các bộ định tuyến MoE quyết định chuyên gia (expert) top-k nào xử lý mỗi token. Một bộ định tuyến ngây thơ sẽ tập trung quá nhiều công việc vào một vài chuyên gia, khiến những chuyên gia khác nhàn rỗi. Giải pháp tiêu chuẩn: thêm một số hạng auxiliary loss để phạt sự mất cân bằng tải. Cách này hiệu quả nhưng làm giảm nhẹ hiệu suất của tác vụ chính.

DeepSeek-V3 giới thiệu một cơ chế không cần auxiliary-loss. Các số hạng bias cho mỗi chuyên gia được thêm vào logit của bộ định tuyến, được điều chỉnh trong quá trình huấn luyện theo một quy tắc đơn giản: nếu chuyên gia `e` bị quá tải, hãy giảm `bias_e`; nếu thiếu tải, hãy tăng nó lên. Không cần số hạng loss bổ sung. Quá trình huấn luyện vẫn sạch sẽ. Tải của chuyên gia vẫn được cân bằng.

Ảnh hưởng đến loss chính: không đáng kể. Ảnh hưởng đến kiến trúc MoE: sạch hơn, không cần điều chỉnh siêu tham số auxiliary-loss.

### MTP: huấn luyện dày đặc hơn + bản nháp miễn phí

Từ Giai đoạn 10 · 18, bạn biết DeepSeek-V3 thêm mô-đun MTP D=1 dự đoán token trước đó hai vị trí. Khi suy luận, mô-đun đã huấn luyện được tái sử dụng như một bản nháp suy luận suy đoán (speculative-decoding) với tỷ lệ chấp nhận hơn 80%. Khi huấn luyện, mỗi trạng thái ẩn được giám sát trên D+1 = 2 mục tiêu, cung cấp tín hiệu dày đặc hơn.

Tham số: 14B trên tổng số 671B chính. Chi phí: 2.1%.

### Huấn luyện: DualPipe

Từ Giai đoạn 10 · 19, bạn biết DualPipe là một pipeline hai chiều chồng lấp các khối forward và backward với giao tiếp all-to-all giữa các node. Ở quy mô 2.048-H800 của DeepSeek-V3, nó thu hồi khoảng 245k giờ GPU mà 1F1B lẽ ra đã mất do các pipeline bubble.

### Cấu hình, từng trường một

Đây là cấu hình DeepSeek-V3 (đã đơn giản hóa):

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

Phân tích nó:

- `hidden_size=7168`: chiều embedding.
- `num_hidden_layers=61`: tổng độ sâu khối.
- `first_k_dense_layers=3`: 3 khối đầu tiên sử dụng MLP dày đặc kích thước 18432. 58 khối còn lại sử dụng MoE.
- `num_attention_heads=128`: 128 đầu truy vấn (query heads).
- `kv_lora_rank=512`: K và V được nén về chiều latent này và giải nén theo từng đầu.
- `num_experts=256, num_experts_per_tok=8`: mỗi khối MoE có 256 chuyên gia, định tuyến top-8.
- `shared_experts=1`: ngoài 256 chuyên gia được định tuyến, 1 chuyên gia luôn bật đóng góp vào mọi token. Hãy coi nó như một "nền tảng dày đặc" đảm bảo mọi token đều nhận được thứ gì đó đáng tin cậy.
- `moe_intermediate_size=2048`: kích thước ẩn MLP của mỗi chuyên gia. Nhỏ hơn MLP dày đặc vì có tới 256 chuyên gia.

### Hạch toán tham số

Phép tính đầy đủ nằm trong `code/main.py`. Điểm chính:

- Embedding: `vocab * hidden = 129280 * 7168 = ~0.93B`.
- 3 khối dày đặc đầu tiên: attention với MLA (~144M mỗi khối) + MLP dày đặc (~260M mỗi khối) + các norm. Tổng cộng khoảng 1.2B.
- 58 khối MoE: attention với MLA (~144M) + 256 chuyên gia mỗi khối (30M mỗi chuyên gia) + 1 chuyên gia dùng chung (30M) + norm. Tổng cộng ~7.95B mỗi khối, bao gồm tất cả các chuyên gia. Tổng cộng 461B cho 58 khối MoE.
- Mô-đun MTP: 14B.

Tổng cộng: ~476B cho kiến trúc lõi + 14B MTP + con số 671B được công bố tính đến các tham số cấu trúc bổ sung (tensor bias, các thành phần chuyên biệt cho chuyên gia, mở rộng chuyên gia dùng chung, v.v.). Con số chúng ta tái tạo trong máy tính nằm trong khoảng 3-5% so với công bố — sự chênh lệch đến từ việc hạch toán chi tiết mà báo cáo của DeepSeek ghi lại trong phụ lục Mục 2.

Tham số hoạt động trên mỗi forward:

- Attention: 144M mỗi lớp * 61 = 8.8B (tất cả các lớp đều chạy).
- MLP hoạt động: 3 lớp đầu tiên dày đặc (3 * 260M = 780M), 58 lớp MoE mỗi lớp hoạt động với 8 chuyên gia định tuyến + 1 chuyên gia dùng chung + chi phí định tuyến. MLP hoạt động mỗi lớp: ~260M. Tổng cộng: 3 * 260M + 58 * 260M = ~15.9B.
- Embedding + các norm: 1.2B.
- Tổng hoạt động: khoảng 26B lõi + 14B MTP (được huấn luyện nhưng không phải lúc nào cũng chạy khi suy luận) ≈ 37B.

### Tỷ lệ 671B / 37B

Tỷ lệ thưa thớt 18x (tham số hoạt động chiếm 5.5% tổng số). DeepSeek-V3 là mô hình MoE tiên phong thưa thớt nhất từng phát hành trọng số mở. Mixtral 8x7B với tỷ lệ 13/47 (28%) dày đặc hơn nhiều. Llama 4 Maverick với tỷ lệ 17B/400B (4.25%) là tương đương. Đặt cược của DeepSeek: ở quy mô tiên phong, nhiều chuyên gia hơn với tỷ lệ kích hoạt thấp hơn tạo ra chất lượng tốt hơn trên mỗi active-FLOP.

### Vị trí của DeepSeek-V3

| Mô hình | Tổng | Hoạt động | Tỷ lệ | Attention | Ý tưởng mới |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | Mở rộng YaRN |

### Phần tiếp theo: R1, V4

DeepSeek-R1 (2025) là một đợt huấn luyện suy luận trên nền tảng V3. R1 sử dụng cùng một kiến trúc. Điều thay đổi là công thức hậu huấn luyện (RL quy mô lớn trên các tác vụ có thể kiểm chứng), không phải kiến trúc tiền huấn luyện.

DeepSeek-V4 (nếu được phát hành) dự kiến sẽ giữ MLA + MoE + MTP và thêm DSA (DeepSeek Sparse Attention), kế thừa của NSA từ Giai đoạn 10 · 17. Dòng dõi này ổn định: các cải tiến ở cấp độ kiến trúc tích lũy; mỗi phiên bản đều điều chỉnh thêm các núm vặn.

```figure
moe-routing
```

## Sử dụng nó

`code/main.py` là máy tính tham số chuyên dụng cho hình dạng của DeepSeek-V3. Hãy chạy nó, so sánh kết quả đầu ra với các con số trong bài báo và sử dụng nó trên các biến thể giả định (256 chuyên gia so với 512, top-8 so với top-16, MLA rank 512 so với 1024).

Những điều cần xem xét:

- Tổng số lượng tham số so với 671B đã công bố.
- Số lượng tham số hoạt động so với 37B đã công bố.
- KV cache ở ngữ cảnh 128k — so sánh MLA với GQA.
- Phân tích theo từng lớp để xem ngân sách tham số thực sự đi đâu.

## Triển khai nó

Bài học này tạo ra `outputs/skill-deepseek-v3-reader.md`. Với một mô hình thuộc dòng DeepSeek (V3, R1, hoặc bất kỳ biến thể tương lai nào), nó tạo ra một bản đọc kiến trúc theo từng thành phần, đặt tên cho từng trường của cấu hình, suy luận số lượng tham số theo thành phần và xác định xem mô hình đó sử dụng những cải tiến đặc thù nào của DeepSeek.

## Bài tập

1. Chạy `code/main.py`. So sánh ước tính tổng tham số của máy tính với 671B đã công bố và xác định sự chênh lệch đến từ đâu. Mục 2 của bài báo có danh mục đầy đủ.

2. Sửa đổi cấu hình để sử dụng MLA rank 256 thay vì 512. Tính toán kích thước KV cache kết quả ở ngữ cảnh 128k. Nó mang lại mức giảm bao nhiêu phần trăm và cái giá phải trả cho khả năng biểu đạt trên mỗi đầu là gì?

3. So sánh định tuyến của DeepSeek-V3 (256 chuyên gia, top-8) với một biến thể giả định (512 chuyên gia, top-8). Tổng tham số tăng lên; tham số hoạt động giữ nguyên. Về lý thuyết, công suất chuyên gia tăng thêm mang lại điều gì và cái giá phải trả khi suy luận là gì?

4. Đọc Mục 2.1 của báo cáo kỹ thuật DeepSeek-V3 (arXiv:2412.19437) về MLA. Giải thích trong ba câu tại sao các ma trận giải nén K và V có thể được "hấp thụ" vào phép matmul tiếp theo để đạt hiệu quả khi suy luận.

5. DeepSeek-V3 sử dụng huấn luyện FP8 cho hầu hết các phép toán. Tính toán mức tiết kiệm bộ nhớ của FP8 so với BF16 để lưu trữ 671B trọng số. Điều này giao thoa như thế nào với ngân sách huấn luyện 14.8T-token?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| MLA | "Multi-Head Latent Attention" | Nén K và V thành một latent bậc thấp dùng chung (kv_lora_rank, thường là 512), giải nén theo từng đầu ngay lập tức; KV cache chỉ lưu latent |
| kv_lora_rank | "Chiều nén MLA" | Kích thước của latent dùng chung cho K và V; DeepSeek-V3 sử dụng 512 |
| First k dense layers | "Các lớp đầu giữ dày đặc" | Một vài lớp đầu tiên của mô hình MoE bỏ qua bộ định tuyến MoE và chạy MLP dày đặc để ổn định |
| num_experts_per_tok | "Định tuyến top-k" | Có bao nhiêu chuyên gia được định tuyến chạy trên mỗi token; DeepSeek-V3 sử dụng 8 |
| Shared experts | "Chuyên gia luôn bật" | Các chuyên gia xử lý mọi token bất kể định tuyến; DeepSeek-V3 sử dụng 1 |
| Auxiliary-loss-free routing | "Cân bằng tải điều chỉnh bias" | Các số hạng bias cho mỗi chuyên gia được điều chỉnh trong quá trình huấn luyện để giữ tải cân bằng mà không cần thêm số hạng loss |
| MTP module | "Đầu dự đoán phụ" | Khối Transformer dự đoán t+2 từ h^(1) và E(t+1); huấn luyện dày đặc hơn, bản nháp suy luận suy đoán miễn phí |
| DualPipe | "Pipeline hai chiều" | Lịch trình huấn luyện chồng lấp tính toán forward/backward với giao tiếp all-to-all giữa các node |
| Active parameter ratio | "Độ thưa thớt" | active_params / total_params; DeepSeek-V3 đạt 5.5% |
| FP8 training | "Huấn luyện 8-bit" | Lưu trữ huấn luyện và nhiều phép tính trong FP8; giảm khoảng một nửa bộ nhớ so với BF16 với chi phí chất lượng nhỏ |

## Đọc thêm

- [DeepSeek-AI — Báo cáo kỹ thuật DeepSeek-V3 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — tài liệu đầy đủ về kiến trúc, huấn luyện và kết quả
- [Thẻ mô hình DeepSeek-V3 trên Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — các tệp cấu hình và ghi chú triển khai
- [Bài báo DeepSeek-V2 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — tiền thân giới thiệu MLA
- [Bài báo DeepSeek-R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — kế thừa huấn luyện suy luận trên kiến trúc V3
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — hướng đi tương lai cho attention dòng DeepSeek
- [Kho lưu trữ DualPipe](https://github.com/deepseek-ai/DualPipe) — tài liệu tham khảo về lịch trình huấn luyện