# BERT — Masked Language Modeling

> GPT dự đoán từ tiếp theo. BERT dự đoán từ bị thiếu. Chỉ một câu khác biệt — và nửa thập kỷ của mọi thứ mang hình thái embedding.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**Time:** ~45 minutes

## Vấn đề

Vào năm 2018, mọi tác vụ NLP — phân tích cảm xúc, NER, QA, entailment — đều huấn luyện mô hình riêng từ đầu trên dữ liệu gán nhãn của chính nó. Không có checkpoint "hiểu tiếng Anh" được huấn luyện trước nào mà bạn có thể fine-tune. ELMo (2018) cho thấy bạn có thể pre-train các contextual embedding bằng bidirectional LSTM; nó hữu ích nhưng không có khả năng tổng quát hóa tốt.

BERT (Devlin et al. 2018) đặt câu hỏi: điều gì sẽ xảy ra nếu chúng ta lấy một transformer encoder, huấn luyện nó trên mọi câu trên internet, và ép nó dự đoán các từ bị thiếu dựa trên ngữ cảnh từ cả hai phía? Sau đó, bạn chỉ cần fine-tune một head cho tác vụ hạ nguồn (downstream task). Hiệu quả về tham số là một sự khai sáng.

Kết quả: trong vòng 18 tháng, BERT và các biến thể của nó (RoBERTa, ALBERT, ELECTRA) đã thống trị mọi bảng xếp hạng NLP hiện có. Đến năm 2020, mọi công cụ tìm kiếm, quy trình kiểm duyệt nội dung và hệ thống tìm kiếm ngữ nghĩa trên trái đất đều có BERT bên trong.

Năm 2026, các mô hình chỉ dùng encoder vẫn là công cụ phù hợp cho phân loại, truy xuất và trích xuất có cấu trúc — chúng chạy nhanh hơn 5–10 lần trên mỗi token so với các decoder và các embedding của chúng là xương sống của mọi stack truy xuất hiện đại. ModernBERT (Tháng 12/2024) đã đẩy kiến trúc này lên ngữ cảnh 8K với Flash Attention + RoPE + GeGLU.

## Khái niệm

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### Tín hiệu huấn luyện

Lấy một câu: `the quick brown fox jumps over the lazy dog`.

Mask ngẫu nhiên 15% các token:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

Huấn luyện mô hình để dự đoán các token gốc tại các vị trí bị mask. Vì encoder là bidirectional, việc dự đoán `[MASK]` tại vị trí 1 có thể sử dụng `brown fox jumps` tại các vị trí 2 trở đi. Đó là điều mà GPT không thể làm được.

### Các quy tắc mask của BERT

Trong số 15% token được chọn để dự đoán:

- 80% được thay thế bằng `[MASK]`.
- 10% được thay thế bằng một token ngẫu nhiên.
- 10% được giữ nguyên.

Tại sao không phải lúc nào cũng là `[MASK]`? Bởi vì `[MASK]` không bao giờ xuất hiện tại thời điểm inference. Việc huấn luyện mô hình mong đợi `[MASK]` tại 100% các vị trí bị mask sẽ tạo ra sự lệch phân phối (distribution shift) giữa pretraining và fine-tuning. 10% ngẫu nhiên + 10% giữ nguyên giúp mô hình trung thực hơn.

### Next Sentence Prediction (NSP) — và tại sao nó bị loại bỏ

BERT gốc cũng huấn luyện trên NSP: cho hai câu A và B, dự đoán xem B có theo sau A hay không. RoBERTa (2019) đã loại bỏ nó và cho thấy NSP gây hại chứ không giúp ích. Các encoder hiện đại đều bỏ qua nó.

### Điều gì đã thay đổi vào năm 2026: ModernBERT

Bài báo ModernBERT năm 2024 đã xây dựng lại khối này với các thành phần của năm 2026:

| Thành phần | BERT gốc (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Positional | Learned absolute | RoPE |
| Activation | GELU | GeGLU |
| Normalization | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | Alternating local (128) + global |
| Độ dài ngữ cảnh | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

Và không giống như stack năm 2018, nó hỗ trợ Flash-Attention nguyên bản. Inference nhanh hơn 2–3 lần ở độ dài chuỗi 8K so với DeBERTa-v3 với điểm số GLUE tốt hơn.

### Các trường hợp sử dụng vẫn chọn encoder vào năm 2026

| Tác vụ | Tại sao encoder tốt hơn decoder |
|------|---------------------------|
| Retrieval / semantic search embeddings | Ngữ cảnh bidirectional = chất lượng embedding tốt hơn trên mỗi token |
| Phân loại (cảm xúc, ý định, độc hại) | Một lần forward pass; không tốn chi phí tạo văn bản |
| NER / gán nhãn token | Đầu ra theo từng vị trí, bidirectional nguyên bản |
| Zero-shot entailment (NLI) | Classifier head đặt trên đỉnh encoder |
| Reranker cho RAG | Cross-encoder scoring, nhanh hơn 10 lần so với LLM rerankers |

```figure
transformer-residual
```

## Xây dựng

### Bước 1: logic masking

Xem `code/main.py`. Hàm `create_mlm_batch` nhận vào một danh sách các ID token, kích thước từ vựng và xác suất mask. Trả về input IDs (đã áp dụng mask) và labels (chỉ tại các vị trí bị mask, -100 ở những nơi khác — quy ước ignore index của PyTorch).

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Bước 2: chạy dự đoán MLM trên một corpus nhỏ

Huấn luyện một encoder 2 lớp + MLM head trên từ vựng 20 từ, 200 câu. Không cần gradient — chúng ta thực hiện kiểm tra sanity check qua forward-pass. Huấn luyện đầy đủ cần PyTorch.

### Bước 3: so sánh các loại mask

Cho thấy quy tắc ba phần giúp mô hình có thể sử dụng được mà không cần `[MASK]`. Dự đoán trên một câu không bị mask và một câu bị mask. Cả hai đều tạo ra phân phối token hợp lý vì mô hình đã thấy cả hai mẫu trong quá trình huấn luyện.

### Bước 4: fine-tune head

Thay thế MLM head bằng một classification head trên tập dữ liệu cảm xúc nhỏ. Chỉ head được huấn luyện; encoder bị đóng băng. Đây là mô hình mà mọi ứng dụng BERT đều tuân theo.

## Sử dụng

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Các mô hình embedding là BERT đã được fine-tune.** Các mô hình `sentence-transformers` như `all-MiniLM-L6-v2` là các BERT được huấn luyện với contrastive loss. Encoder vẫn giống nhau. Chỉ có loss thay đổi.

**Cross-encoder rerankers cũng là BERT đã được fine-tune.** Phân loại cặp trên `[CLS] query [SEP] doc [SEP]`. Sự chú ý bidirectional giữa truy vấn và tài liệu chính là thứ mang lại lợi thế về chất lượng cho cross-encoders so với biencoders.

**Khi nào không nên chọn BERT vào năm 2026.** Bất cứ thứ gì liên quan đến tạo văn bản (generative). Encoder không có cách nào hợp lý để tạo token theo kiểu tự hồi quy (autoregressive). Ngoài ra: bất kỳ mô hình nào dưới 1B tham số mà một decoder nhỏ có thể đạt chất lượng tương đương với sự linh hoạt cao hơn (Phi-3-Mini, Qwen2-1.5B).

## Triển khai

Xem `outputs/skill-bert-finetuner.md`. Kỹ năng này xác định phạm vi cho một quy trình fine-tune BERT (chọn backbone, đặc tả head, dữ liệu, đánh giá, dừng huấn luyện) cho một tác vụ phân loại hoặc trích xuất mới.

## Bài tập

1. **Dễ.** Chạy `code/main.py` và in phân phối mask trên 10.000 token. Xác nhận ~15% được chọn, và trong số đó ~80% trở thành `[MASK]`.
2. **Trung bình.** Triển khai whole-word masking: nếu một từ được token hóa thành các subword, hãy mask tất cả các subword cùng nhau hoặc không mask gì cả. Đo lường xem liệu điều này có cải thiện độ chính xác MLM trên corpus 500 câu hay không.
3. **Khó.** Huấn luyện một BERT siêu nhỏ (2 lớp, d=64) trên 10.000 câu từ một tập dữ liệu công khai. Fine-tune token `[CLS]` cho tác vụ cảm xúc SST-2. So sánh với baseline chỉ dùng decoder ở cùng số lượng tham số — bên nào thắng?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| MLM | "Masked language modeling" | Tín hiệu huấn luyện: thay thế ngẫu nhiên 15% token bằng `[MASK]`, dự đoán các token gốc. |
| Bidirectional | "Nhìn cả hai phía" | Encoder attention không có causal mask — mọi vị trí đều nhìn thấy mọi vị trí khác. |
| `[CLS]` | "Token pooler" | Một token đặc biệt được thêm vào đầu mỗi chuỗi; embedding cuối cùng của nó được dùng làm biểu diễn cấp câu. |
| `[SEP]` | "Dấu phân cách đoạn" | Phân tách các chuỗi cặp (ví dụ: truy vấn/tài liệu, câu A/B). |
| NSP | "Next sentence prediction" | Tác vụ pretraining thứ hai của BERT; đã được chứng minh là vô dụng trong RoBERTa, bị loại bỏ sau 2019. |
| Fine-tuning | "Thích nghi với tác vụ" | Giữ encoder gần như đóng băng; huấn luyện một head nhỏ bên trên cho tác vụ hạ nguồn. |
| Cross-encoder | "Một reranker" | Một BERT nhận cả truy vấn và tài liệu làm đầu vào, xuất ra điểm số liên quan. |
| ModernBERT | "Bản làm mới 2024" | Encoder được xây dựng lại với RoPE, RMSNorm, GeGLU, attention xen kẽ local/global, ngữ cảnh 8K. |

## Đọc thêm

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — bài báo gốc.
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — cách huấn luyện BERT đúng cách; loại bỏ NSP.
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — phát hiện token bị thay thế vượt trội hơn MLM ở cùng mức tính toán.
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — bài báo ModernBERT.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — tài liệu tham khảo chuẩn về encoder.