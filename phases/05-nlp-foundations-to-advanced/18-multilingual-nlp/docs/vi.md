# Multilingual NLP

> Một mô hình, hơn 100 ngôn ngữ, hầu hết không cần dữ liệu huấn luyện. Chuyển đổi đa ngôn ngữ (cross-lingual transfer) là phép màu thực tế của thập niên 2020.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**Time:** ~45 minutes

## Vấn đề

Tiếng Anh có hàng tỷ ví dụ được gán nhãn. Tiếng Urdu có hàng nghìn. Tiếng Maithili gần như không có gì. Bất kỳ hệ thống NLP thực tế nào phục vụ khán giả toàn cầu đều phải hoạt động trên "phần đuôi dài" (long tail) của các ngôn ngữ mà ở đó không tồn tại dữ liệu huấn luyện chuyên biệt cho tác vụ.

Các mô hình đa ngôn ngữ giải quyết vấn đề này bằng cách huấn luyện một mô hình trên nhiều ngôn ngữ cùng lúc. Các biểu diễn được chia sẻ cho phép mô hình chuyển giao các kỹ năng học được từ các ngôn ngữ tài nguyên cao (high-resource) sang các ngôn ngữ tài nguyên thấp (low-resource). Tinh chỉnh (fine-tune) mô hình trên tác vụ phân tích cảm xúc tiếng Anh, và nó sẽ tạo ra các dự đoán cảm xúc đáng kinh ngạc trên tiếng Urdu ngay lập tức. Đó là zero-shot cross-lingual transfer, và nó đã định hình lại cách NLP được triển khai trên toàn thế giới.

Bài học này nêu tên các đánh đổi, các mô hình kinh điển và một quyết định thường gây khó khăn cho các đội ngũ mới làm việc với đa ngôn ngữ: chọn ngôn ngữ nguồn để chuyển giao.

## Khái niệm

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**Từ vựng chia sẻ (Shared vocabulary).** Các mô hình đa ngôn ngữ sử dụng SentencePiece hoặc WordPiece tokenizer được huấn luyện trên văn bản từ tất cả các ngôn ngữ mục tiêu. Từ vựng được chia sẻ: cùng một đơn vị subword đại diện cho cùng một hình vị (morpheme) trên các ngôn ngữ liên quan. `anti-` trong tiếng Anh và tiếng Ý nhận được cùng một token.

**Biểu diễn chia sẻ (Shared representation).** Một Transformer được tiền huấn luyện trên masked language modeling qua nhiều ngôn ngữ sẽ học được rằng các câu tương đồng về ngữ nghĩa trong các ngôn ngữ khác nhau sẽ tạo ra các trạng thái ẩn (hidden states) tương tự nhau. mBERT, XLM-R và NLLB đều thể hiện điều này. Các embedding cho "cat" trong tiếng Anh tập trung gần "chat" trong tiếng Pháp và "gato" trong tiếng Tây Ban Nha, và các embedding toàn câu cũng vậy.

**Zero-shot transfer.** Tinh chỉnh mô hình trên dữ liệu được gán nhãn bằng một ngôn ngữ (thường là tiếng Anh). Khi suy luận (inference), chạy nó trên bất kỳ ngôn ngữ nào khác mà mô hình hỗ trợ. Không cần nhãn ngôn ngữ mục tiêu. Kết quả rất mạnh đối với các ngôn ngữ có liên quan về loại hình học (typologically related) và yếu hơn đối với các ngôn ngữ xa lạ.

**Few-shot fine-tuning.** Thêm 100-500 ví dụ được gán nhãn trong ngôn ngữ mục tiêu. Độ chính xác tăng lên 95-98% so với baseline tiếng Anh trên các tác vụ phân loại. Đây là đòn bẩy hiệu quả nhất về chi phí trong NLP đa ngôn ngữ.

## Các mô hình

| Mô hình | Năm | Độ phủ | Ghi chú |
|-------|------|----------|-------|
| mBERT | 2018 | 104 ngôn ngữ | Huấn luyện trên Wikipedia. LM đa ngôn ngữ thực tế đầu tiên. Yếu ở ngôn ngữ tài nguyên thấp. |
| XLM-R | 2019 | 100 ngôn ngữ | Huấn luyện trên CommonCrawl (lớn hơn nhiều so với Wikipedia). Thiết lập baseline đa ngôn ngữ. Base 270M, Large 550M. |
| XLM-V | 2023 | 100 ngôn ngữ | XLM-R với từ vựng 1 triệu token (so với 250k). Tốt hơn cho ngôn ngữ tài nguyên thấp. |
| mT5 | 2020 | 101 ngôn ngữ | Kiến trúc T5 cho tạo văn bản đa ngôn ngữ. |
| NLLB-200 | 2022 | 200 ngôn ngữ | Mô hình dịch thuật của Meta; bao gồm 55 ngôn ngữ tài nguyên thấp. |
| BLOOM | 2022 | 46 ngôn ngữ + 13 ngôn ngữ lập trình | LLM 176B mã nguồn mở được huấn luyện đa ngôn ngữ. |
| Aya-23 | 2024 | 23 ngôn ngữ | LLM đa ngôn ngữ của Cohere. Mạnh về tiếng Ả Rập, Hindi, Swahili. |

Chọn theo trường hợp sử dụng. Phân loại hoạt động tốt với XLM-R-base như một lựa chọn mặc định an toàn. Các tác vụ tạo văn bản cần mT5 hoặc NLLB tùy thuộc vào việc dịch thuật hay tạo văn bản mở. Công việc kiểu LLM phù hợp với Aya-23 hoặc Claude bằng cách sử dụng prompt đa ngôn ngữ rõ ràng.

## Quyết định về ngôn ngữ nguồn (nghiên cứu 2026)

Hầu hết các đội ngũ mặc định chọn tiếng Anh làm nguồn tinh chỉnh. Nghiên cứu gần đây (2026) cho thấy điều này thường sai.

Sự tương đồng ngôn ngữ dự đoán chất lượng chuyển giao tốt hơn kích thước ngữ liệu thô. Đối với các mục tiêu ngôn ngữ Slav, tiếng Đức hoặc tiếng Nga thường vượt trội hơn tiếng Anh. Đối với các mục tiêu ngôn ngữ Ấn Độ, tiếng Hindi thường vượt trội hơn tiếng Anh. Chỉ số tương đồng **qWALS** (2026, dựa trên các đặc điểm của World Atlas of Language Structures) định lượng điều này. **LANGRANK** (Lin và cộng sự, ACL 2019) là một phương pháp riêng biệt, sớm hơn, xếp hạng các ngôn ngữ nguồn ứng viên từ sự kết hợp của sự tương đồng ngôn ngữ, kích thước ngữ liệu và quan hệ di truyền.

Quy tắc thực tế: nếu ngôn ngữ mục tiêu của bạn có một ngôn ngữ liên quan tài nguyên cao về mặt loại hình học, hãy thử tinh chỉnh trên ngôn ngữ đó trước, sau đó so sánh với việc tinh chỉnh bằng tiếng Anh.

```figure
n5-crosslingual-bridge
```

## Xây dựng

### Bước 1: Phân loại đa ngôn ngữ zero-shot

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

Một mô hình, ba ngôn ngữ, cùng một API. XLM-R được huấn luyện trên dữ liệu NLI chuyển giao tốt sang phân loại thông qua thủ thuật entailment.

### Bước 2: Không gian embedding đa ngôn ngữ

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

Các bản dịch nằm gần nhau trong không gian embedding. Một câu tiếng Anh khác nằm xa hơn. Đây là điều làm cho việc truy xuất, phân cụm và đo độ tương đồng đa ngôn ngữ hoạt động.

### Bước 3: Chiến lược few-shot fine-tuning

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

Đối với 100-500 ví dụ ngôn ngữ mục tiêu, `num_train_epochs=5` và `learning_rate=2e-5` là các mặc định an toàn. Tốc độ học (learning rate) cao hơn sẽ khiến sự căn chỉnh đa ngôn ngữ bị sụp đổ và bạn sẽ nhận được một mô hình chỉ biết tiếng Anh.

## Đánh giá thực tế

- **Độ chính xác theo từng ngôn ngữ trên tập kiểm thử (held-out sets).** Không tổng hợp. Tổng hợp sẽ che giấu phần đuôi dài.
- **Benchmark so với baseline đơn ngữ.** Đối với các ngôn ngữ có đủ dữ liệu, một mô hình đơn ngữ được huấn luyện từ đầu đôi khi đánh bại mô hình đa ngôn ngữ. Hãy kiểm tra.
- **Kiểm tra cấp độ thực thể.** Các thực thể có tên (Named entities) trong ngôn ngữ mục tiêu. Các mô hình đa ngôn ngữ thường có tokenizer yếu đối với các hệ thống chữ viết khác xa với chữ Latin.
- **Tính nhất quán đa ngôn ngữ.** Cùng một ý nghĩa trong hai ngôn ngữ nên tạo ra cùng một dự đoán. Hãy đo lường khoảng cách này.

## Sử dụng

Stack công nghệ năm 2026:

| Tác vụ | Khuyến nghị |
|-----|-------------|
| Phân loại, 100 ngôn ngữ | XLM-R-base (~270M) đã tinh chỉnh |
| Phân loại văn bản zero-shot | `joeddav/xlm-roberta-large-xnli` |
| Embedding câu đa ngôn ngữ | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Dịch thuật, 200 ngôn ngữ | `facebook/nllb-200-distilled-600M` (xem bài 11) |
| Tạo văn bản đa ngôn ngữ | Claude, GPT-4, Aya-23, mT5-XXL |
| NLP ngôn ngữ tài nguyên thấp | XLM-V hoặc tinh chỉnh chuyên biệt theo miền trên ngôn ngữ tài nguyên cao liên quan |

Luôn dự trù ngân sách cho việc tinh chỉnh trong ngôn ngữ mục tiêu nếu hiệu suất là quan trọng. Zero-shot chỉ là điểm khởi đầu, không phải câu trả lời cuối cùng.

### Thuế tokenization (điều gì xảy ra với các ngôn ngữ tài nguyên thấp)

Các mô hình đa ngôn ngữ chia sẻ một tokenizer cho tất cả các ngôn ngữ của chúng. Từ vựng đó được huấn luyện trên một ngữ liệu bị thống trị bởi tiếng Anh, Pháp, Tây Ban Nha, Trung Quốc, Đức. Đối với bất kỳ ngôn ngữ nào nằm ngoài tập thống trị, ba loại "thuế" sẽ tích tụ âm thầm:

- **Thuế độ phì nhiêu (Fertility tax).** Văn bản ngôn ngữ tài nguyên thấp được token hóa thành nhiều token trên mỗi từ hơn so với tiếng Anh. Một câu tiếng Hindi có thể cần số lượng token gấp 3-5 lần một câu tiếng Anh tương đương. Điều đó làm tiêu tốn cửa sổ ngữ cảnh (context window), hiệu quả huấn luyện và độ trễ.
- **Thuế phục hồi biến thể (Variant recovery tax).** Mọi lỗi chính tả, biến thể dấu phụ, sai lệch chuẩn hóa Unicode hoặc biến thể chữ hoa/thường đều trở thành một chuỗi không liên quan trong không gian embedding. Mô hình không thể học được các tương ứng chính tả mà người bản ngữ coi là hiển nhiên.
- **Thuế tràn công suất (Capacity spillover tax).** Thuế 1 và 2 tiêu tốn các vị trí ngữ cảnh, độ sâu lớp và kích thước embedding. Những gì còn lại cho việc suy luận thực sự nhỏ hơn một cách hệ thống so với những gì một ngôn ngữ tài nguyên cao nhận được từ cùng một mô hình.

Triệu chứng thực tế: mô hình của bạn huấn luyện bình thường trên tiếng Hindi, đường cong loss trông ổn, perplexity đánh giá hợp lý, nhưng kết quả đầu ra thực tế lại sai lệch một cách tinh vi. Hình thái học bị sụp đổ giữa câu. Các biến cách hiếm gặp vẫn không thể khôi phục. **Bạn không thể mở rộng dữ liệu để thoát khỏi một tokenizer bị hỏng.**

Cách giảm thiểu: chọn một tokenizer có độ phủ tốt cho ngôn ngữ mục tiêu của bạn (từ vựng 1 triệu token của XLM-V là một bản sửa lỗi trực tiếp); xác minh độ phì nhiêu của tokenization trên văn bản mục tiêu trước khi huấn luyện; sử dụng fallback cấp byte (SentencePiece `byte_fallback=True`, BPE cấp byte kiểu GPT-2) cho các hệ thống chữ viết thực sự thuộc đuôi dài để không bao giờ gặp lỗi OOV (Out-of-Vocabulary).

## Triển khai

Lưu dưới dạng `outputs/skill-multilingual-picker.md`:

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## Bài tập

1. **Dễ.** Chạy pipeline phân loại zero-shot trên 10 câu mỗi ngôn ngữ cho tiếng Anh, Pháp, Hindi và Ả Rập. Báo cáo độ chính xác trên từng ngôn ngữ. Bạn sẽ thấy tiếng Pháp mạnh, tiếng Hindi khá, tiếng Ả Rập biến động.
2. **Trung bình.** Sử dụng `paraphrase-multilingual-MiniLM-L12-v2` để xây dựng một bộ truy xuất đa ngôn ngữ trên một ngữ liệu nhỏ đa ngôn ngữ. Truy vấn bằng tiếng Anh, truy xuất tài liệu bằng bất kỳ ngôn ngữ nào. Đo lường recall@5.
3. **Khó.** So sánh việc tinh chỉnh từ nguồn tiếng Anh và nguồn tiếng Hindi cho một tác vụ phân loại tiếng Hindi. Sử dụng 500 ví dụ ngôn ngữ mục tiêu cho few-shot fine-tuning dưới cả hai chế độ. Báo cáo nguồn nào tạo ra độ chính xác tiếng Hindi tốt hơn và tốt hơn bao nhiêu. Đây là luận điểm LANGRANK ở quy mô nhỏ.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Multilingual model | Một mô hình, nhiều ngôn ngữ | Chia sẻ từ vựng và tham số giữa các ngôn ngữ. |
| Cross-lingual transfer | Huấn luyện một ngôn ngữ, chạy ngôn ngữ khác | Tinh chỉnh trên nguồn, đánh giá trên mục tiêu mà không cần nhãn mục tiêu. |
| Zero-shot | Không có nhãn ngôn ngữ mục tiêu | Chuyển giao mà không cần tinh chỉnh trên ngôn ngữ mục tiêu. |
| Few-shot | Nhãn mục tiêu nhỏ | 100-500 ví dụ ngôn ngữ mục tiêu được sử dụng để tinh chỉnh. |
| mBERT | LM đa ngôn ngữ đầu tiên | BERT 104 ngôn ngữ được tiền huấn luyện trên Wikipedia. |
| XLM-R | Baseline đa ngôn ngữ tiêu chuẩn | RoBERTa 100 ngôn ngữ được tiền huấn luyện trên CommonCrawl. |
| NLLB | MT 200 ngôn ngữ của Meta | No Language Left Behind. Bao gồm 55 ngôn ngữ tài nguyên thấp. |

## Đọc thêm

- [Conneau và cộng sự (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — bài báo về XLM-R.
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — bài báo phân tích khởi đầu cho dòng nghiên cứu chuyển giao đa ngôn ngữ.
- [Costa-jussà và cộng sự (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — bài báo về NLLB-200.
- [Üstün và cộng sự (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya, LLM đa ngôn ngữ của Cohere.
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — bài báo về ngôn ngữ nguồn qWALS / LANGRANK.