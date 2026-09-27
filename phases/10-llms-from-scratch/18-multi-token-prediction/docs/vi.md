# Multi-Token Prediction (MTP)

> Mọi LLM tự hồi quy (autoregressive) từ GPT-2 đến Llama 3 đều huấn luyện dựa trên một hàm mất mát (loss) cho mỗi vị trí: dự đoán token tiếp theo. DeepSeek-V3 đã bổ sung thêm hàm mất mát thứ hai cho mỗi vị trí: dự đoán token sau đó nữa. 14B tham số bổ sung (trên một mô hình 671B) đã được chưng cất (distilled) ngược lại vào mô hình chính thông qua dòng gradient, và các đầu ra MTP đã được huấn luyện được tái sử dụng tại thời điểm suy luận (inference) như các bộ dự thảo (drafters) cho speculative decoding với tỷ lệ chấp nhận trên 80%. Thông lượng tạo văn bản tăng 1.8 lần mà không tốn thêm chi phí. Bài học này xây dựng module MTP tuần tự từ báo cáo kỹ thuật của DeepSeek, tính toán hàm mất mát và bố cục tham số dùng chung, đồng thời giải thích lý do tại sao MTP duy trì được chuỗi nhân quả (causal chain) trong khi thiết kế MTP song song ban đầu của Gloeckle và cộng sự lại phá vỡ nó.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 04 (pre-training một mini GPT), Phase 10 · 15 (speculative decoding)
**Time:** ~60 phút

## Mục tiêu học tập

- Trình bày mục tiêu huấn luyện MTP và suy ra hàm mất mát kết hợp qua các độ sâu dự đoán.
- Giải thích sự khác biệt giữa các đầu ra MTP song song của Gloeckle và cộng sự (2024) và các module MTP tuần tự của DeepSeek-V3, cũng như lý do tại sao thiết kế tuần tự bảo toàn được chuỗi nhân quả.
- Tính toán chi phí tham số và bộ nhớ khi thêm các module MTP vào quá trình pre-training.
- Triển khai một module MTP từ đầu: embedding dùng chung, khối transformer theo độ sâu, phép chiếu (projection) và đầu ra (output head) dùng chung.

## Vấn đề

Dự đoán token tiếp theo là mục tiêu huấn luyện LLM tiêu chuẩn. Mỗi trạng thái ẩn (hidden state) được giám sát để dự đoán chính xác một thứ: token ngay sau đó. Đây là một tín hiệu yếu một cách đáng ngạc nhiên. Hầu hết thông tin trong một chuỗi kéo dài vượt quá một token — cấu trúc, tính mạch lạc, tính xác thực, luồng logic toán học. Mô hình phải học những điều đó bằng cách tích lũy nhiều tín hiệu đơn lẻ qua hàng nghìn tỷ token.

MTP đặt câu hỏi: điều gì sẽ xảy ra nếu mỗi trạng thái ẩn được giám sát để dự đoán nhiều token tương lai cùng một lúc? Gloeckle và cộng sự (Meta, 2024) đã chứng minh điều này có hiệu quả. Cách triển khai của họ đặt một vài đầu ra độc lập lên trên backbone, mỗi đầu ra dự đoán một độ lệch (offset) khác nhau. Cách tiếp cận này song song, đơn giản, nhưng các đầu ra nhìn thấy cùng một trạng thái ẩn mà không có sự tinh chỉnh phân cấp nào — và các dự đoán không liên kết theo chuỗi nhân quả, vì vậy chúng không thể được sử dụng cho speculative decoding.

DeepSeek-V3 (tháng 12 năm 2024) đã thiết kế lại MTP thành các module tuần tự giúp duy trì chuỗi nhân quả tại mỗi độ sâu dự đoán. Mô hình dự đoán `t+1` từ `h_i^(0)`, sau đó dự đoán `t+2` từ một trạng thái ẩn mới `h_i^(1)` kết hợp `h_i^(0)` với embedding của `E(t+1)`, và cứ tiếp tục như vậy. Mỗi độ sâu là một khối transformer nhỏ riêng biệt. Embedding dùng chung và đầu ra dùng chung giúp giữ cho chi phí tham số ở mức khiêm tốn. Ở quy mô của DeepSeek-V3, 14B tham số bổ sung trên các module MTP so với 671B trọng số mô hình chính. Mức chi phí 2% đó mang lại các tín hiệu huấn luyện dày đặc hơn VÀ một bản dự thảo speculative decoding có sẵn tại thời điểm suy luận.

Bài học này xây dựng một module MTP đơn lẻ và hàm mất mát D-độ sâu từ đầu. Toán học rất gọn gàng. Việc triển khai chỉ mất 150 dòng code.

## Khái niệm

### Công thức MTP tuần tự

DeepSeek-V3 thêm `D` module MTP lên trên mô hình chính. Mỗi module `k` (cho `k = 1..D`) dự đoán token tại độ sâu `k` — tức là, `t_{i+k}` dựa trên tiền tố (prefix) đến vị trí `i`.

Module `k` bao gồm:

- Một khối transformer `T_k` với attention và MLP riêng.
- Một ma trận chiếu `M_k` kết hợp trạng thái ẩn của độ sâu trước đó với embedding của token ground-truth ở độ sâu tiếp theo.
- Embedding dùng chung `E` (giống mô hình chính).
- Đầu ra dùng chung `Out` (giống mô hình chính).

Khi huấn luyện, đối với một tiền tố đến vị trí `i`, trạng thái ẩn theo độ sâu là:

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

Dự đoán theo độ sâu là:

```
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

Hàm mất mát theo độ sâu là cross-entropy so với ground-truth `t_{i+k}`:

```
L_k = CE(logits_{i+k}, t_{i+k})
```

Hàm mất mát kết hợp qua các độ sâu:

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` là một hệ số trọng số nhỏ — DeepSeek-V3 sử dụng 0.3 cho 10% quá trình huấn luyện đầu tiên và 0.1 sau đó. Tổng hàm mất mát huấn luyện là `L_main + L_MTP`.

### Tại sao là tuần tự, không phải song song

MTP song song ban đầu của Gloeckle có D đầu ra, mỗi đầu ra áp dụng trực tiếp lên `h_i^(0)`. Mỗi đầu ra dự đoán `t_{i+k}` từ cùng một trạng thái ẩn của backbone. Cách này huấn luyện tốt, nhưng các dự đoán không được điều kiện hóa (conditioned) lẫn nhau. Bạn không thể sử dụng đầu ra của `head_1` để hỗ trợ `head_2` — các đầu ra hoạt động song song.

Thiết kế tuần tự của DeepSeek-V3 xây dựng `h_i^(k)` từ `h_i^(k-1)` cộng với embedding của token tiếp theo thực tế `E(t_{i+k})`. Điều đó bảo toàn chuỗi nhân quả: để dự đoán `t_{i+k+1}`, module tại độ sâu `k+1` nhìn thấy những gì đã có tại `t_{i+k}`. Điều này về mặt cấu trúc giống hệt cách một bộ giải mã tự hồi quy tiêu thụ đầu ra của chính nó — làm cho các module MTP có thể sử dụng trực tiếp như các bộ dự thảo cho speculative decoding.

Tại thời điểm suy luận: đưa `h_i^(k-1)` và `t_{i+k}` đã dự thảo vào module `k+1`, nhận dự đoán cho `t_{i+k+1}`. Lặp lại. Đó chính xác là kiểu dự thảo EAGLE, sử dụng module MTP đã huấn luyện làm mạng dự thảo. DeepSeek-V3 báo cáo tỷ lệ chấp nhận trên 80% ở module MTP đầu tiên và tăng tốc độ khoảng 1.8 lần.

### Hạch toán tham số

Đối với một mô hình có hidden size `h` và từ vựng `V`:

- Mô hình chính: hàng tỷ tham số, cộng với một đầu ra kích thước `V * h`.
- Đầu ra dùng chung: tái sử dụng đầu ra của mô hình chính. Không thêm tham số.
- Embedding dùng chung: tái sử dụng embedding của mô hình chính. Không thêm tham số.
- Mỗi module MTP:
  - Phép chiếu `M_k`: `(2h) * h = 2h^2`.
  - Khối transformer `T_k`: attention (`4h^2` cho MHA) cộng với MLP (thường là `8h^2` cho SwiGLU với tỷ lệ 8/3). Khoảng `12h^2` mỗi khối.

Tổng cộng thêm mỗi module: `~14h^2`. Đối với `h = 7168` của DeepSeek-V3, D = 1 module: `~14 * 7168^2 = ~720M` tham số trên lý thuyết. DeepSeek-V3 báo cáo 14B — sự khác biệt chủ yếu là do các lớp chuyên gia (expert layers) cũng là MoE trong module MTP.

### Lợi ích của speculative-decoding

Trong quá trình pre-training, các module MTP làm chậm quá trình huấn luyện khoảng 10% (tính toán forward nhiều hơn, thêm hàm mất mát). Lợi ích mang lại là hai mặt:

1. Tín hiệu huấn luyện dày đặc hơn. Mỗi trạng thái ẩn nhìn thấy D+1 mục tiêu giám sát. Hiệu quả đo lường trên MMLU, GSM8K, MATH, HumanEval: cải thiện nhất quán vài điểm phần trăm trong các thử nghiệm của DeepSeek-V3.

2. Bản dự thảo speculative decoding miễn phí tại thời điểm suy luận. Module MTP đã được huấn luyện để dự đoán vài token tiếp theo. Được tái sử dụng như một mạng dự thảo, nó mang lại tỷ lệ chấp nhận trên 80%. Ở mức đó, spec decoding N=3 hoặc N=5 mang lại thông lượng 1.8 lần. Chi phí 10% thời gian huấn luyện được hoàn trả ngay lần đầu tiên bạn chạy suy luận.

### Mối quan hệ với EAGLE

EAGLE huấn luyện một mô hình dự thảo nhỏ RIÊNG BIỆT sau khi pre-training. MTP tích hợp bản dự thảo vào quá trình pre-training. Hai cách tiếp cận hội tụ về tỷ lệ chấp nhận tương tự nhưng thông qua các quy trình khác nhau:

| Chiều | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| Thời điểm huấn luyện | Sau pre-training | Trong khi pre-training |
| Tương thích ngược với trọng số cũ | Có | Không (cần huấn luyện lại) |
| Tham số dự thảo | 1-2 lớp transformer | 1 khối transformer + phép chiếu |
| Tỷ lệ chấp nhận | 0.88-0.92 | 0.80+ tại độ sâu 1 |
| Lợi ích ngoài tốc độ | Chỉ speculative decoding | Tín hiệu huấn luyện dày đặc hơn + tốc độ |

```figure
multi-token-predict
```

## Xây dựng

`code/main.py` xây dựng một module MTP đơn lẻ từ đầu đến cuối: embedding dùng chung, phép chiếu, khối transformer, đầu ra dùng chung. Sau đó, nó tính toán hàm mất mát cross-entropy theo độ sâu trên một chuỗi tổng hợp ngắn và in số lượng tham số theo từng thành phần. Một từ vựng đồ chơi gồm 32 token giúp các con số dễ đọc hơn.

### Bước 1: bảng embedding dùng chung

Một bảng `vocab_size x hidden` duy nhất được sử dụng bởi mô hình chính VÀ bởi mọi module MTP ở mọi độ sâu. Không phải bản sao thứ hai — mà thực sự là cùng một tensor.

### Bước 2: kết hợp theo độ sâu

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

DeepSeek-V3 thực tế nối (concatenate) hai vector đã qua RMSNorm thành `[2h]` và chiếu với ma trận `h x 2h`. Bản demo sử dụng phép cộng vector để ngắn gọn.

### Bước 3: khối transformer tại độ sâu k

Self-attention cộng với MLP. Trong bản demo, một khối linear attention một lớp và một MLP SwiGLU giữ cho cấu trúc hiển thị rõ ràng mà không cần numpy.

### Bước 4: đầu ra dùng chung

Tái sử dụng phép chiếu đầu ra của mô hình chính. Logits trên toàn bộ từ vựng.

### Bước 5: hàm mất mát theo độ sâu

Cross-entropy của softmax(logits) so với token ground-truth tại độ lệch `k`. Tổng hợp qua các độ sâu với hệ số tỷ lệ `lambda / D`.

### Bước 6: hạch toán tham số

In tổng số lượng tham số, số lượng dùng chung (embedding, đầu ra), và số lượng bổ sung mỗi module. Hiển thị tỷ lệ giữa phần bổ sung MTP và kích thước mô hình chính.

## Sử dụng

MTP được tích hợp vào DeepSeek-V3 (tháng 12 năm 2024) và dòng DeepSeek-R1. Tại thời điểm suy luận:

- Stack phục vụ (serving stack) của riêng DeepSeek tiêu thụ các module MTP như các bộ giải mã dự thảo ngay lập tức.
- vLLM và SGLang đã có các đường dẫn tích hợp cho MTP của DeepSeek-V3 kể từ tháng 4 năm 2026.
- Hướng dẫn SGLang ROCm của AMD cho thấy một cấu hình speculative-decoding MTP cụ thể với tốc độ đo được tăng 1.8 lần trên checkpoint V3.

Khi nào nên sử dụng MTP trong một đợt pre-training mới:

- Bạn kiểm soát toàn bộ quy trình pre-training và muốn tích lũy tín hiệu huấn luyện dày đặc hơn.
- Bạn biết mình sẽ phục vụ mô hình ở quy mô lớn và muốn có speculative decoding miễn phí.
- Hidden size của bạn ít nhất là 4096. Ở quy mô 1B, chi phí bổ sung gây hại nhiều hơn là lợi ích mang lại.

Khi nào không nên:

- Fine-tuning một mô hình dày (dense) đã được pre-train. Module MTP chưa được huấn luyện.
- Các mô hình nghiên cứu nơi bạn muốn một baseline sạch để so sánh. MTP làm thay đổi kiến trúc.

## Triển khai

Bài học này tạo ra `outputs/skill-mtp-planner.md`. Với một đặc tả chạy pre-training (kích thước mô hình, dữ liệu, tính toán), nó trả về một kế hoạch tích hợp MTP: số lượng độ sâu D, lịch trình `lambda`, chi phí bộ nhớ, và cấu hình speculative-decoding tại thời điểm suy luận.

## Bài tập

1. Chạy `code/main.py`. Cho thấy hàm mất mát theo độ sâu giảm đơn điệu khi tín hiệu tổng hợp mạnh lên. Sửa đổi phần tổng hợp để sử dụng một mẫu cố định và xác minh cả hàm mất mát độ sâu 1 và độ sâu 2 đều hội tụ.

2. Tính toán chi phí tham số cho một mô hình 70B dày (hidden 8192, 80 lớp) với module MTP D=1. So sánh với chi phí 14B mà DeepSeek-V3 báo cáo. Giải thích tại sao con số của DeepSeek lại cao hơn: khối transformer MTP kế thừa cùng cấu trúc MoE, làm tăng số lượng tham số mỗi module.

3. Triển khai D=2 trong bản demo: thêm module MTP thứ hai nhận h^(1) và dự đoán `t_{i+2}`. Xác minh hàm mất mát kết hợp và hạch toán tham số khớp với các phương trình 19-21 trong bài báo DeepSeek.

4. Chuyển bản demo sang MTP song song (kiểu Gloeckle): thêm D đầu ra lên trên trạng thái ẩn chính, mỗi đầu ra dự đoán một độ lệch khác nhau. Đo lường cách các hàm mất mát theo độ sâu so sánh với phiên bản tuần tự trên cùng một tín hiệu tổng hợp. Phiên bản tuần tự sẽ tạo ra hàm mất mát độ sâu k thấp hơn cho k > 1 vì nó điều kiện hóa trên các dự đoán trung gian.

5. Sử dụng module MTP đã huấn luyện như một bản dự thảo kiểu EAGLE: gọi module k để đề xuất `t_{i+k}` tại thời điểm suy luận. Đo lường tỷ lệ chấp nhận của các token dự thảo này so với dự đoán của mô hình chính trên một chuỗi giữ lại (held-out). Nếu bạn đạt 50%+ trên bản demo, bạn đã tái tạo thành công đặc tính MTP-as-draft.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Module MTP | "Khối mất mát bổ sung" | Một khối transformer nhỏ cộng với phép chiếu dự đoán một token `k` vị trí phía trước mô hình chính |
| Độ sâu dự đoán | "Độ lệch nào" | Số nguyên `k` sao cho module `k` dự đoán `t_{i+k}` từ tiền tố đến vị trí `i` |
| MTP song song | "Kiểu Gloeckle" | D đầu ra độc lập trên cùng một trạng thái ẩn của backbone, không có chuỗi điều kiện |
| MTP tuần tự | "Kiểu DeepSeek-V3" | Mỗi module điều kiện hóa trên trạng thái ẩn của độ sâu trước đó cộng với embedding của token tiếp theo; bảo toàn chuỗi nhân quả |
| Đầu ra dùng chung | "Tái sử dụng đầu ra chính" | Các module MTP gọi đầu ra LM của mô hình chính, không phải phép chiếu đầu ra riêng biệt |
| Embedding dùng chung | "Tái sử dụng bảng chính" | Bảng embedding từ vựng giống nhau được sử dụng ở mọi nơi; không trùng lặp tham số |
| Ma trận chiếu M_k | "Kết hợp ẩn + token tiếp theo" | Một lớp tuyến tính `h x 2h` gộp trạng thái ẩn trước đó và embedding token mục tiêu vào đầu vào của độ sâu tiếp theo |
| Hàm mất mát kết hợp L_MTP | "Trung bình các mất mát bổ sung" | Trung bình cộng của các hàm mất mát cross-entropy theo độ sâu, được chia tỷ lệ bởi `lambda` |
| Tỷ lệ chấp nhận tại độ sâu 1 | "MTP dự thảo đúng bao nhiêu" | Tỷ lệ mà dự đoán top-1 của module MTP D=1 bằng với dự đoán top-1 của mô hình chính; 80%+ trên DeepSeek-V3 |
| Trọng số Lambda | "Tầm quan trọng của mất mát bổ sung" | Hệ số tỷ lệ theo độ sâu; 0.3 lúc bắt đầu huấn luyện, 0.1 sau đó trên DeepSeek-V3 |

## Đọc thêm

- [DeepSeek-AI — Báo cáo kỹ thuật DeepSeek-V3 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — mô tả đầy đủ về MTP tuần tự (Mục 2.2), bao gồm các phương trình mất mát kết hợp và tốc độ tăng 1.8 lần tại thời điểm suy luận
- [Gloeckle và cộng sự — LLM tốt hơn & nhanh hơn thông qua dự đoán đa token (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) — baseline MTP song song mà thiết kế của DeepSeek cải tiến dựa trên đó
- [Thẻ mô hình DeepSeek-V3 trên Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — tổng 685B (671B chính + 14B MTP), ghi chú triển khai
- [Leviathan và cộng sự — Suy luận nhanh từ Transformer thông qua Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — khung speculative-decoding mà MTP tích hợp vào
- [Li và cộng sự — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — kiến trúc dự thảo EAGLE năm 2025, đối thủ cạnh tranh với MTP