# T5, BART — Các mô hình Encoder-Decoder

> Encoder để hiểu. Decoder để tạo. Kết hợp chúng lại và bạn có một mô hình được xây dựng cho các tác vụ input → output: dịch thuật, tóm tắt, viết lại, chuyển đổi văn bản.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT), Phase 7 · 07 (GPT)
**Time:** ~45 phút

## Vấn đề

GPT (chỉ có decoder) và BERT (chỉ có encoder) đều lược bỏ kiến trúc năm 2017 cho các mục tiêu khác nhau. Nhưng nhiều tác vụ vốn dĩ là input-output:

- Dịch thuật: Tiếng Anh → Tiếng Pháp.
- Tóm tắt: Bài báo 5.000 token → Tóm tắt 200 token.
- Nhận dạng giọng nói: token âm thanh → token văn bản.
- Trích xuất cấu trúc: văn bản xuôi → JSON.

Đối với những tác vụ này, encoder-decoder là lựa chọn phù hợp nhất. Encoder tạo ra một biểu diễn dày đặc (dense representation) của nguồn. Decoder tạo ra đầu ra, thực hiện cross-attention với biểu diễn đó tại mỗi bước. Quá trình huấn luyện là dịch chuyển một bước (shift-by-one) ở phía đầu ra. Hàm mất mát (loss) tương tự như GPT, chỉ là có điều kiện dựa trên đầu ra của encoder.

Hai bài báo đã định nghĩa lối chơi hiện đại:

1. **T5** (Raffel và cộng sự, 2019). "Text-to-Text Transfer Transformer." Mọi tác vụ NLP được định hình lại thành text-in, text-out. Một kiến trúc duy nhất, một từ vựng duy nhất, một hàm mất mát duy nhất. Được tiền huấn luyện bằng cách dự đoán đoạn bị che (masked span prediction) (làm hỏng các đoạn trong đầu vào, giải mã chúng ở đầu ra).
2. **BART** (Lewis và cộng sự, 2019). "Bidirectional and Auto-Regressive Transformer." Denoising autoencoder: làm hỏng đầu vào theo nhiều cách (xáo trộn, che, xóa, xoay), yêu cầu decoder tái tạo lại bản gốc.

Vào năm 2026, định dạng encoder-decoder vẫn tồn tại ở những nơi mà cấu trúc đầu vào quan trọng:

- Whisper (giọng nói → văn bản).
- Hệ thống dịch thuật của Google.
- Một số mô hình hoàn thiện/sửa lỗi mã nguồn có cấu trúc ngữ cảnh và chỉnh sửa riêng biệt.
- Flan-T5 và các biến thể cho các tác vụ suy luận có cấu trúc.

Mô hình chỉ có decoder đã chiếm lĩnh sự chú ý, nhưng encoder-decoder chưa bao giờ biến mất.

## Khái niệm

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### Vòng lặp forward

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

Quan trọng là, encoder chạy một lần cho mỗi đầu vào. Decoder chạy theo kiểu tự hồi quy (autoregressive) nhưng thực hiện cross-attention với *cùng một* đầu ra của encoder tại mỗi bước. Việc lưu trữ (caching) đầu ra của encoder là một cách tăng tốc miễn phí cho các đầu vào dài.

### Tiền huấn luyện T5 — span corruption

Chọn ngẫu nhiên các đoạn (span) của đầu vào (độ dài trung bình 3 token, tổng cộng 15%). Thay thế mỗi đoạn bằng một sentinel duy nhất: `<extra_id_0>`, `<extra_id_1>`, v.v. Decoder chỉ xuất ra các đoạn bị hỏng kèm theo tiền tố sentinel của chúng:

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

Tín hiệu rẻ hơn so với việc dự đoán toàn bộ chuỗi. Cạnh tranh với MLM (BERT) và prefix-LM (UniLM) trong phần phân tích (ablation) của bài báo T5.

### Tiền huấn luyện BART — multi-noise denoising

BART thử nghiệm năm hàm gây nhiễu:

1. Che token (Token masking).
2. Xóa token (Token deletion).
3. Điền văn bản (Text infilling - che một đoạn, decoder chèn độ dài chính xác).
4. Hoán vị câu (Sentence permutation).
5. Xoay tài liệu (Document rotation).

Kết hợp điền văn bản + hoán vị câu mang lại kết quả tốt nhất cho các tác vụ hạ nguồn. Decoder luôn tái tạo lại bản gốc. Đầu ra của BART là toàn bộ chuỗi, không chỉ các đoạn bị hỏng — vì vậy chi phí tính toán tiền huấn luyện cao hơn T5.

### Suy luận (Inference)

Tương tự như quá trình tạo tự hồi quy của GPT. Các phương pháp Greedy / beam / top-p sampling đều áp dụng được. Beam search (độ rộng 4–5) là tiêu chuẩn cho dịch thuật và tóm tắt vì phân phối đầu ra hẹp hơn so với chat.

### Khi nào chọn biến thể nào vào năm 2026

| Tác vụ | Encoder-decoder? | Tại sao |
|------|------------------|-----|
| Dịch thuật | Có, thường là vậy | Chuỗi nguồn rõ ràng; phân phối đầu ra cố định; beam search hoạt động tốt |
| Chuyển giọng nói thành văn bản | Có (Whisper) | Phương thức đầu vào khác với đầu ra; encoder định hình các đặc trưng âm thanh |
| Chat / suy luận | Không, chỉ decoder | Không có "đầu vào" cố định — cuộc hội thoại chính là chuỗi |
| Hoàn thiện mã nguồn | Thường là không | Decoder-only với ngữ cảnh dài chiếm ưu thế; các mô hình code như Qwen 2.5 Coder là decoder-only |
| Tóm tắt | Cả hai đều được | BART, PEGASUS đánh bại các baseline decoder-only cũ; các LLM decoder-only hiện đại đã bắt kịp |
| Trích xuất cấu trúc | Cả hai | T5 sạch sẽ vì "text → text" hấp thụ mọi định dạng đầu ra |

Xu hướng từ khoảng năm 2022: decoder-only chiếm lĩnh các tác vụ mà encoder-decoder từng sở hữu vì (a) các LLM decoder-only được tinh chỉnh hướng dẫn (instruction-tuned) có thể tổng quát hóa mọi thứ thông qua prompting, (b) một kiến trúc dễ mở rộng hơn hai, (c) RLHF giả định một decoder. Encoder-decoder vẫn giữ vị thế ở những nơi phương thức đầu vào khác biệt (giọng nói, hình ảnh) hoặc nơi chất lượng của beam search quan trọng.

```figure
encoder-decoder
```

## Xây dựng

Xem `code/main.py`. Chúng ta triển khai span corruption kiểu T5 cho một tập dữ liệu nhỏ — phần hữu ích nhất của bài học này vì nó xuất hiện trong mọi công thức tiền huấn luyện encoder-decoder kể từ đó.

### Bước 1: span corruption

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

Định dạng mục tiêu là quy ước của T5: `<sent0> span0 <sent1> span1 ...`. Đầu vào bị hỏng xen kẽ các token không thay đổi với các token sentinel tại các vị trí đoạn.

### Bước 2: xác minh round-trip

Với đầu vào bị hỏng và mục tiêu, hãy tái tạo lại câu gốc. Nếu quá trình làm hỏng của bạn có thể đảo ngược, thì forward pass được xác định rõ ràng. Đây là một bước kiểm tra tính hợp lý — quá trình huấn luyện thực tế không bao giờ làm điều này, nhưng bài kiểm tra này rẻ và giúp phát hiện các lỗi off-by-one trong việc quản lý đoạn của bạn.

### Bước 3: BART noising

Năm hàm: `token_mask`, `token_delete`, `text_infill`, `sentence_permute`, `document_rotate`. Kết hợp hai trong số đó và hiển thị kết quả.

## Sử dụng

Tham chiếu HuggingFace:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

Mẹo của T5: tên tác vụ được đưa vào văn bản đầu vào. Cùng một mô hình xử lý hàng chục tác vụ vì mỗi tác vụ đều là text-in, text-out. Vào năm 2026, mô hình này đã được tổng quát hóa bởi các mô hình decoder-only được tinh chỉnh hướng dẫn, nhưng T5 là mô hình đầu tiên hệ thống hóa nó.

## Triển khai

Xem `outputs/skill-seq2seq-picker.md`. Kỹ năng này giúp lựa chọn giữa encoder-decoder và decoder-only cho một tác vụ mới dựa trên cấu trúc input-output, độ trễ và các mục tiêu chất lượng.

## Bài tập

1. **Dễ.** Chạy `code/main.py`, áp dụng span corruption cho một câu 30 token, xác minh rằng việc nối các token nguồn không phải sentinel với các đoạn mục tiêu đã giải mã sẽ tái tạo lại câu gốc.
2. **Trung bình.** Triển khai nhiễu `text_infill` của BART: thay thế các đoạn ngẫu nhiên bằng một token `<mask>` duy nhất, và decoder phải suy luận độ dài đoạn chính xác cộng với nội dung. Hiển thị một ví dụ.
3. **Khó.** Tinh chỉnh `flan-t5-small` trên một tập dữ liệu nhỏ Anh → pig-Latin (200 cặp). Đo lường BLEU trên tập 50 cặp giữ lại. So sánh với việc tinh chỉnh `Llama-3.2-1B` trên cùng dữ liệu với cùng tài nguyên tính toán.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Encoder-decoder | "Seq2seq transformer" | Hai chồng: encoder hai chiều cho đầu vào, decoder nhân quả với cross-attention cho đầu ra. |
| Cross-attention | "Nơi nguồn giao tiếp với đích" | Q của decoder × K/V của encoder. Nơi duy nhất thông tin encoder đi vào decoder. |
| Span corruption | "Mẹo tiền huấn luyện của T5" | Thay thế các đoạn ngẫu nhiên bằng token sentinel; decoder xuất ra các đoạn đó. |
| Denoising objective | "Trò chơi của BART" | Áp dụng hàm nhiễu vào đầu vào, huấn luyện decoder tái tạo chuỗi sạch. |
| Sentinel token | "Trình giữ chỗ `<extra_id_N>`" | Các token đặc biệt gắn thẻ các đoạn bị hỏng trong nguồn và gắn thẻ lại chúng trong mục tiêu. |
| Flan | "T5 được tinh chỉnh hướng dẫn" | T5 được tinh chỉnh trên >1.800 tác vụ; giúp encoder-decoder cạnh tranh trong việc tuân thủ hướng dẫn. |
| Beam search | "Chiến lược giải mã" | Giữ lại top-k chuỗi một phần tại mỗi bước; tiêu chuẩn cho dịch thuật/tóm tắt. |
| Teacher forcing | "Đầu vào lúc huấn luyện" | Trong khi huấn luyện, đưa token đầu ra đúng thực tế vào decoder, không phải token được lấy mẫu. |

## Đọc thêm

- [Raffel và cộng sự (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5.
- [Lewis và cộng sự (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART.
- [Chung và cộng sự (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5.
- [Radford và cộng sự (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper, mô hình encoder-decoder tiêu chuẩn năm 2026.
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — triển khai tham chiếu.