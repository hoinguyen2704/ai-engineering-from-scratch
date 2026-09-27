# Native Sparse Attention (DeepSeek NSA)

> Ở độ dài 64k token, cơ chế attention chiếm 70-80% độ trễ khi giải mã (decode). Mọi phòng thí nghiệm mô hình mở đều có kế hoạch để khắc phục điều này. NSA của DeepSeek (bài báo xuất sắc nhất tại ACL 2025) là giải pháp thực sự hiệu quả: ba nhánh attention song song — các token được nén ở mức thô, các token được chọn lọc ở mức chi tiết, và cửa sổ trượt (sliding window) cho ngữ cảnh cục bộ — được kết hợp thông qua một cổng (gate) đã học. Nó được tối ưu hóa cho phần cứng (thân thiện với kernel), có khả năng huấn luyện tự nhiên (hoạt động ngay từ giai đoạn pre-training, không phải chắp vá khi inference), và ở độ dài 64k token, nó chạy nhanh hơn FlashAttention trong khi vẫn duy trì hoặc vượt trội hơn về chất lượng so với full attention. Bài học này xây dựng ba nhánh từ đầu đến cuối và giải thích lý do tại sao tính thưa thớt (sparsity) này có thể vi phân (differentiable) hoàn toàn.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 12 (KV cache, flash-attention), Phase 7 · 15 (attention variants), Phase 10 · 16 (differential attention)
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu được ba nhánh attention của NSA và những gì mỗi nhánh thu thập được.
- Giải thích tại sao NSA lại "có khả năng huấn luyện tự nhiên" trong khi các phương pháp sparse-attention trước đây chỉ dành cho inference.
- Tính toán mức tiết kiệm tài nguyên tính toán của NSA so với full attention ở ngữ cảnh 64k dựa trên kích thước khối nén và tham số top-k.
- Triển khai sự kết hợp ba nhánh bằng Python stdlib trên một chuỗi tổng hợp ngắn và xác minh trọng số của cổng (gating weights) hoạt động đúng.

## Vấn đề

Full attention ở độ dài chuỗi N tiêu tốn `O(N^2)` thời gian và `O(N)` KV cache cho mỗi lớp. Ở 64k token, các con số về tính toán và băng thông bộ nhớ trở nên thảm họa. Ước tính lý thuyết từ bài báo NSA: attention chiếm 70-80% tổng độ trễ giải mã ở 64k. Mọi thứ phía sau — TTFT, tokens/sec, chi phí mỗi triệu token — đều bị chi phối bởi chi phí của attention.

Sparse attention là câu trả lời hiển nhiên. Các nỗ lực trước đây rơi vào hai nhóm. Độ thưa thớt theo mẫu cố định (sliding-window, strided, block-local) làm mất thông tin và thất bại trong các tác vụ yêu cầu truy xuất tầm xa. Độ thưa thớt tại thời điểm inference (KV cache pruning, H2O, StreamingLLM) được áp dụng cho mô hình đã pre-train với dense attention và chỉ thu lại được một phần nhỏ tốc độ tiềm năng vì mô hình chưa bao giờ được yêu cầu định tuyến thông tin qua mẫu thưa thớt đó.

Native Sparse Attention (Yuan và cộng sự, DeepSeek + PKU + UW, bài báo xuất sắc nhất tại ACL 2025, arXiv:2502.11089) thực hiện cả hai: một mẫu thưa thớt mà mô hình học được trong quá trình pre-training, được triển khai như một thuật toán tương thích với kernel thực sự mang lại hiệu quả tính toán khi inference. Hai năm nữa, NSA hoặc một biến thể trực tiếp của nó sẽ là cơ chế attention mặc định trên mọi mô hình ngữ cảnh dài tiên phong.

## Khái niệm

### Ba nhánh song song

Với mỗi query, NSA chạy attention ba lần, đối chiếu với ba góc nhìn khác nhau của KV cache:

1. **Nhánh nén (Compressed branch).** Các token được nhóm thành các khối có kích thước `l` (thường là 32 hoặc 64). Mỗi khối được nén thành một token tóm tắt duy nhất thông qua một MLP nhỏ đã học. Query thực hiện attention trên các token nén này, thu được cái nhìn tổng quan (coarse-grained) về toàn bộ chuỗi.

2. **Nhánh chọn lọc (Selected branch).** Sử dụng điểm số attention từ nhánh nén, các khối top-k liên quan nhất đến query hiện tại được xác định. Các token chi tiết (chưa nén) từ các khối đó được đọc và query thực hiện attention trên tất cả chúng. Hãy coi attention của nhánh nén là tín hiệu định tuyến cho việc lựa chọn.

3. **Nhánh cửa sổ trượt (Sliding-window branch).** Query thực hiện attention với `W` token gần nhất (thường là 512) để lấy ngữ cảnh cục bộ. Nhánh này nắm bắt các mẫu ngắn hạn có cấu trúc phức tạp (cú pháp, tham chiếu cục bộ) mà hai nhánh kia có thể bỏ lỡ.

Đầu ra của ba nhánh được kết hợp thông qua một cổng (gate) học được cho mỗi vị trí:

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` là các trọng số cổng từ một MLP nhỏ trên query. Chúng không nhất thiết phải có tổng bằng 1 — chúng có thể điều chỉnh trọng số các nhánh một cách độc lập.

### Tại sao nó "có khả năng huấn luyện tự nhiên"

Bước lựa chọn (top-k blocks) là rời rạc. Các thao tác rời rạc làm gián đoạn dòng gradient. Các công trình sparse-attention trước đây hoặc là bỏ qua backprop qua bước lựa chọn (hạn chế việc huấn luyện) hoặc sử dụng các phép xấp xỉ liên tục không mang lại độ thưa thớt thực sự khi inference.

NSA tránh được điều này: attention của nhánh nén CHÍNH LÀ một attention thô có thể vi phân trên toàn bộ chuỗi. Thao tác top-k chỉ tái sử dụng các điểm số attention cao nhất từ nhánh nén để chọn khối chi tiết nào cần tải. Gradient chảy qua các điểm số của nhánh nén (vốn ảnh hưởng đến cả đầu ra nén VÀ logic lựa chọn), và đóng góp của các khối được chọn vào đầu ra cuối cùng cũng có thể vi phân. Thao tác `top_k` không thể vi phân chỉ là một no-op trên đồ thị tính toán forward — nó chỉ kiểm soát khối nào được tải từ bộ nhớ.

Đây là lý do tại sao NSA có thể được sử dụng trong pre-training từ đầu đến cuối. Mô hình học cách định tuyến thông tin qua ba nhánh một cách đồng bộ, tạo ra một mẫu thưa thớt thực sự mang lại tốc độ như mong đợi khi inference.

### Kernel tương thích với phần cứng

Kernel của NSA được thiết kế cho hệ thống phân cấp bộ nhớ GPU hiện đại. Kernel tải các query theo nhóm GQA (vòng lặp ngoài), tìm nạp các khối KV thưa thớt tương ứng cho mỗi nhóm (vòng lặp trong), và chạy attention trên SRAM. Vì mỗi nhóm query nhìn thấy cùng các khối được chọn (việc lựa chọn theo nhóm query, không phải theo đầu query), các lần tải KV được phân bổ đều trên toàn nhóm. Cường độ tính toán (arithmetic intensity) vẫn ở mức cao.

Bài báo báo cáo rằng các kernel Triton chạy nhanh hơn 9 lần so với FlashAttention ở độ dài 64k, với tỷ lệ tăng tốc tăng dần theo độ dài chuỗi. Cả kernel forward và backward đều được cung cấp.

### Ngân sách tính toán

Gọi `N` là độ dài chuỗi, `l` là kích thước khối nén, `k` là số lượng top-k, `w` là cửa sổ trượt, `b` là kích thước khối được chọn (thường bằng `l`).

- Nhánh nén: `O(N/l)` key mỗi query, tổng cộng `O(N * N / l)`.
- Nhánh chọn lọc: `O(k * b)` key mỗi query, tổng cộng `O(N * k * b)`.
- Nhánh trượt: `O(w)` key mỗi query, tổng cộng `O(N * w)`.

Tổng cộng: `O(N * (N/l + k*b + w))`.

Với `N = 64k, l = 64, k = 16, b = 64, w = 512`: chi phí mỗi query là `1000 + 1024 + 512 = 2536 keys`. Full attention là `64000 keys`. Giảm 25 lần tính toán.

Với `N = 128k, l = 64, k = 16, b = 64, w = 512`: chi phí mỗi query là `2000 + 1024 + 512 = 3536 keys`. Full attention là `128000 keys`. Giảm 36 lần. Lợi ích tăng theo độ dài chuỗi, đó chính là mục đích chính.

### So sánh

| Phương pháp | Có thể vi phân | Tăng tốc inference thực tế | Truy xuất tầm xa |
|--------|---------------|----------------------|-------------------|
| Chỉ sliding window | có | có | thất bại |
| Strided / block-sparse | có | có | một phần |
| KV pruning (H2O, StreamingLLM) | N/A (thời điểm inference) | có | một phần |
| MoBA (Moonshot) | một phần | có | tốt |
| NSA | có (tự nhiên) | có (9x tại 64k) | tương đương full attention |

MoBA (Moonshot, arXiv:2502.13189) được công bố đồng thời và có cách tiếp cận "ba tốt hơn một" tương tự, áp dụng nguyên lý MoE cho các khối attention. NSA và MoBA là hai kiến trúc cần biết cho việc pre-training ngữ cảnh dài năm 2026.

```figure
sliding-window-attention
```

## Triển khai

`code/main.py` triển khai ba nhánh trên một chuỗi tổng hợp ngắn và cho thấy:

- MLP nén (một baseline mean-pool đơn giản được sử dụng để làm rõ về mặt sư phạm; NSA thực tế sử dụng MLP đã học).
- Lựa chọn khối top-k dựa trên điểm số của nhánh nén.
- Attention cửa sổ trượt trên `w` token cuối cùng.
- Kết hợp có cổng.
- Bảng đếm tính toán so sánh với full attention.

### Bước 1: nén token thành các khối

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Bước 2: attention nhánh nén

Chạy softmax attention của query đối với các key đã nén. Điểm số của nhánh nén đóng vai trò là tín hiệu cho việc lựa chọn top-k.

### Bước 3: lựa chọn khối top-k

Chọn chỉ số của `k` khối nén có điểm số cao nhất. Tải các token gốc chưa nén từ các khối đó và chạy attention trên chúng.

### Bước 4: attention cửa sổ trượt

Lấy `w` token cuối cùng và chạy attention tiêu chuẩn trên chúng.

### Bước 5: cổng + kết hợp

Một MLP nhỏ trên query tạo ra ba trọng số cổng. Đầu ra cuối cùng là tổng có trọng số của đầu ra ba nhánh.

### Bước 6: đếm tính toán

In số lượng key được attention mỗi query cho mỗi nhánh và tổng cộng. So sánh với `N` (full attention). Trên chuỗi tổng hợp 1024 token với `l = 32, k = 4, w = 128`, NSA thấy `32 + 128 + 128 = 288` key mỗi query so với 1024 của full attention — ít hơn 3.5 lần.

## Sử dụng

NSA đang được triển khai trong pipeline pre-training ngữ cảnh dài của DeepSeek. Trạng thái tích hợp trong các stack inference công khai tính đến tháng 4 năm 2026:

- **DeepSeek nội bộ**: native, các trọng số được công bố sử dụng NSA hoặc phiên bản kế nhiệm DSA (Deepseek Sparse Attention).
- **vLLM**: hỗ trợ NSA thử nghiệm đang được phát triển cho các trọng số DeepSeek-V3.x.
- **SGLang**: các benchmark NSA đã được công bố; lộ trình sản xuất theo sau vLLM.
- **llama.cpp / CPU**: không hỗ trợ; chi phí phân tách kernel không xứng đáng với thông lượng trên CPU.

Khi nào nên dùng NSA:

- Chạy pre-training hoặc continued-training nhắm đến ngữ cảnh 64k trở lên với ngân sách tính toán lớn.
- Inference các checkpoint ngữ cảnh dài của chính DeepSeek. Các trọng số này là NSA-native.

Khi nào không nên:

- Phục vụ một mô hình đã pre-train với dense-attention. Bạn không thể trang bị thêm NSA mà không cần continued training.
- Ngữ cảnh dưới 16k. Chi phí của ba nhánh sẽ lấn át mức tiết kiệm.
- Chat tương tác batch-1. Độ trễ giải mã nhạy cảm sẽ được hưởng lợi, nhưng chỉ ở ngữ cảnh dài.

## Triển khai thực tế

Bài học này tạo ra `outputs/skill-nsa-integrator.md`. Với một đặc tả chạy pre-training ngữ cảnh dài, nó tạo ra một kế hoạch tích hợp NSA: kích thước khối nén, top-k, cửa sổ trượt, độ rộng MLP cổng, lựa chọn kernel, và các đánh giá ngữ cảnh dài cụ thể để biện minh cho việc thay đổi kiến trúc.

## Bài tập

1. Chạy `code/main.py` trên chuỗi tổng hợp 1024 token. Quét `(l, k, w)` qua ba cấu hình cài sẵn và in số lượng tính toán. Xác định cấu hình đạt được số lượng key mỗi query thấp nhất trong khi vẫn giữ được 95% khả năng truy xuất so với full attention trong bài kiểm tra needle-in-haystack.

2. Thay thế bộ nén mean-pool bằng một MLP nhỏ đã học (2 lớp, ẩn 32). Huấn luyện nó trên một tác vụ tổng hợp nơi tín hiệu là trung bình của một khối. Đo khoảng cách perplexity so với baseline mean-pool trên dữ liệu held-out.

3. Triển khai MLP cổng. Nó lấy query làm đầu vào và xuất ra ba giá trị vô hướng. Chứng minh rằng cổng hoạt động hợp lý: trọng số gần như đồng nhất trên các query ngẫu nhiên, trọng số lớn trên nhánh được chọn khi query khớp với một khối ở xa.

4. Tính toán ngân sách bộ nhớ KV cache cho mô hình 70B hỗ trợ NSA ở ngữ cảnh 128k. KV head là 8, head dim 128, BF16. So sánh với full attention và MLA (Phase 10 · 14 đã cho thấy các con số của MLA). Xác định độ dài chuỗi mà tại đó KV cache của nhánh chi tiết của NSA bằng với full attention.

5. Đọc Mục 4 của bài báo NSA (arXiv:2502.11089) và giải thích trong ba câu tại sao điểm số attention của nhánh nén được tái sử dụng cho lựa chọn top-k thay vì tính toán một điểm số định tuyến riêng biệt. Liên kết câu trả lời với dòng gradient.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Nhánh nén | "Cái nhìn thô" | Attention trên các key đã lấy trung bình khối, cung cấp ngữ cảnh toàn cục với O(N/l) key mỗi query |
| Nhánh chọn lọc | "Khối top-k" | Attention chi tiết trên `k` khối có điểm số nhánh nén cao nhất |
| Cửa sổ trượt | "Ngữ cảnh cục bộ" | Attention trên `W` token cuối cùng cho các mẫu ngắn hạn |
| Khả năng huấn luyện tự nhiên | "Pre-train với độ thưa thớt" | Mẫu thưa thớt được học trong quá trình pre-training, không phải chắp vá khi inference |
| Kích thước khối nén l | "Kích thước nhóm cho cái nhìn thô" | Bao nhiêu token được gộp thành một token tóm tắt; thường là 32-64 |
| Top-k | "Số khối cần giữ" | Số lượng khối nén mà các token chi tiết của chúng được đọc; thường là 16 |
| Cửa sổ trượt W | "Bán kính attention cục bộ" | Thường là 512; ngắn hơn làm hỏng tính nhất quán cục bộ, dài hơn gây lãng phí tính toán |
| Cổng nhánh | "Cách trộn ba nhánh" | Đầu ra MLP cho mỗi vị trí giúp điều chỉnh trọng số đóng góp của ba nhánh |
| Tương thích phần cứng | "Độ thưa thớt thân thiện với kernel" | Mẫu thưa thớt được chọn để kernel GPU thực tế đạt được tốc độ lý thuyết |
| DSA | "Người kế nhiệm NSA" | Deepseek Sparse Attention, kiến trúc theo sau NSA trong dòng dõi của DeepSeek |

## Đọc thêm

- [Yuan và cộng sự — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — bài báo gốc
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — gia đình kiến trúc mà NSA nhắm tới
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — công trình đồng thời, attention kiểu MoE trên các khối
- [Beltagy và cộng sự — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — nguồn gốc của sliding-window
- [Xiao và cộng sự — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — baseline độ thưa thớt tại thời điểm inference mà NSA cải thiện
- [Dao và cộng sự — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — baseline full-attention mà các kernel NSA đánh bại ở 64k