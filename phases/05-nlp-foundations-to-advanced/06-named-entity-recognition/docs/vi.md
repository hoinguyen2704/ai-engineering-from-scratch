# Named Entity Recognition

> Trích xuất các tên riêng. Nghe có vẻ dễ dàng cho đến khi bạn phải đối mặt với các ranh giới mơ hồ, thực thể lồng nhau và thuật ngữ chuyên ngành.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 minutes

## The Problem

"Apple sued Google over its iPhone search deal in the US." Có năm thực thể: Apple (ORG), Google (ORG), iPhone (PRODUCT), search deal (có thể), US (GPE). Một hệ thống NER tốt sẽ trích xuất tất cả chúng với đúng loại. Một hệ thống tồi sẽ bỏ sót iPhone, nhầm lẫn giữa Apple là trái cây với Apple là công ty, và gán nhãn "US" là PERSON.

NER là công cụ cốt lõi bên dưới mọi pipeline trích xuất có cấu trúc. Phân tích sơ yếu lý lịch, quét nhật ký tuân thủ, ẩn danh hóa hồ sơ y tế, hiểu truy vấn tìm kiếm, tạo nền tảng cho phản hồi chatbot, trích xuất hợp đồng pháp lý. Bạn không bao giờ thực sự nhìn thấy nó; nhưng bạn luôn phụ thuộc vào nó.

Bài học này đi theo con đường cổ điển (dựa trên quy tắc, HMM, CRF) đến con đường hiện đại (BiLSTM-CRF, sau đó là transformers). Mỗi bước giải quyết một hạn chế cụ thể của bước trước đó. Mô hình này chính là bài học.

## The Concept

**BIO tagging** (hoặc BILOU) biến việc trích xuất thực thể thành một bài toán gán nhãn chuỗi. Gán nhãn mỗi token với `B-TYPE` (bắt đầu thực thể), `I-TYPE` (bên trong thực thể), hoặc `O` (bên ngoài bất kỳ thực thể nào).

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

Các thực thể nhiều token được liên kết: `New B-GPE`, `York I-GPE`, `City I-GPE`. Một mô hình hiểu được BIO có thể trích xuất các đoạn (span) tùy ý.

Sự tiến hóa của kiến trúc:

- **Dựa trên quy tắc (Rule-based).** Regex + tra cứu từ điển (gazetteer). Độ chính xác cao với các thực thể đã biết, không có khả năng bao phủ các thực thể mới.
- **HMM.** Hidden Markov Model. Xác suất phát xạ của token dựa trên nhãn, xác suất chuyển đổi từ nhãn sang nhãn. Giải mã Viterbi. Được huấn luyện trên dữ liệu đã gán nhãn.
- **CRF.** Conditional Random Field. Giống HMM nhưng mang tính phân biệt (discriminative), vì vậy bạn có thể kết hợp các đặc trưng tùy ý (hình dạng từ, viết hoa, các từ lân cận). Vẫn là công cụ sản xuất cổ điển trong năm 2026 cho các triển khai tài nguyên thấp.
- **BiLSTM-CRF.** Các đặc trưng thần kinh thay vì thủ công. LSTM đọc câu theo cả hai hướng, lớp CRF bên trên thực thi các chuỗi nhãn nhất quán.
- **Dựa trên Transformer.** Fine-tune BERT với một đầu phân loại token (token-classification head). Độ chính xác tốt nhất. Tốn nhiều tài nguyên tính toán nhất.

```figure
ner-bio-tagging
```

## Build It

### Step 1: BIO tagging helpers

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Step 2: hand-crafted features

Đối với NER cổ điển (không dùng mạng thần kinh), các đặc trưng là yếu tố quyết định. Các đặc trưng hữu ích:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` trả về `xXxxxx`. `word_shape("USA-2024")` trả về `XXX-dddd`. Các mẫu viết hoa là tín hiệu mạnh cho danh từ riêng.

### Step 3: a simple rule-based + dictionary baseline

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

Các từ điển (gazetteer) trong sản xuất có hàng triệu mục được thu thập từ Wikipedia và DBpedia. Độ bao phủ tốt. Khả năng phân biệt (`Apple` công ty vs trái cây) rất tệ. Đó là lý do tại sao các mô hình thống kê đã chiến thắng.

### Step 4: the CRF step (sketch, not full impl)

Việc triển khai CRF đầy đủ từ đầu trong 50 dòng code sẽ không mang lại nhiều kiến thức nếu không có nền tảng lý thuyết xác suất. Hãy sử dụng `sklearn-crfsuite` thay thế:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` và `c2` là các kỹ thuật điều chuẩn (regularization) L1 và L2. `all_possible_transitions=True` cho phép mô hình học rằng các chuỗi không hợp lệ (ví dụ: `I-ORG` sau `O`) là không khả thi, đây là cách CRF thực thi tính nhất quán BIO mà bạn không cần phải viết các ràng buộc thủ công.

### Step 5: what a BiLSTM-CRF adds

Các đặc trưng được học tự động. Đầu vào: token embeddings (GloVe hoặc fastText). LSTM đọc từ trái sang phải và từ phải sang trái. Các trạng thái ẩn được nối lại và đi qua lớp đầu ra CRF. CRF vẫn thực thi tính nhất quán của chuỗi nhãn; LSTM thay thế các đặc trưng thủ công bằng các đặc trưng được học.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

Đối với lớp CRF, hãy sử dụng `torchcrf.CRF` (pip install pytorch-crf). Mức tăng so với CRF thủ công là có thể đo lường được nhưng nhỏ hơn bạn mong đợi trừ khi bạn có hàng chục nghìn câu đã gán nhãn.

## Use It

spaCy cung cấp NER cấp độ sản xuất ngay khi cài đặt.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

Lưu ý `iPhone` được gán nhãn `ORG` thay vì `PRODUCT` — mô hình nhỏ của spaCy có độ bao phủ thực thể sản phẩm yếu. Mô hình lớn (`en_core_web_lg`) làm tốt hơn. Mô hình transformer (`en_core_web_trf`) làm tốt hơn nữa.

Hugging Face cho NER dựa trên BERT:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` hợp nhất các token B-X, I-X liền kề thành một đoạn (span). Nếu không có nó, bạn sẽ nhận được các nhãn ở cấp độ token và phải tự hợp nhất.

### LLM-based NER (the 2026 option)

LLM NER theo phương pháp zero-shot và few-shot hiện nay có khả năng cạnh tranh với các mô hình đã fine-tune trên nhiều lĩnh vực, và tốt hơn đáng kể khi dữ liệu gán nhãn khan hiếm.

- **Zero-shot prompting.** Cung cấp cho LLM danh sách các loại thực thể và một schema ví dụ. Yêu cầu đầu ra JSON. Hoạt động ngay lập tức; độ chính xác ở mức trung bình trên các lĩnh vực mới.
- **ZeroTuneBio-style prompting.** Phân tách nhiệm vụ thành trích xuất ứng viên → giải thích ý nghĩa → đánh giá → kiểm tra lại. Một prompt đa giai đoạn (không phải one-shot) nâng cao độ chính xác đáng kể trên NER y sinh. Mô hình tương tự hoạt động cho các lĩnh vực pháp lý, tài chính và khoa học.
- **Dynamic prompting with RAG.** Truy xuất các ví dụ đã gán nhãn tương tự nhất từ một tập hợp nhỏ các ví dụ được chú thích cho mỗi lần gọi suy luận; xây dựng prompt few-shot ngay lập tức. Trong các benchmark năm 2026, điều này nâng F1 NER y sinh của GPT-4 lên 11-12% so với prompt tĩnh.
- **Per-entity-type decomposition.** Đối với các tài liệu dài, một lần gọi trích xuất tất cả các loại thực thể cùng lúc sẽ làm giảm độ thu hồi (recall) khi độ dài tăng lên. Hãy chạy một lượt trích xuất cho mỗi loại thực thể. Chi phí suy luận cao hơn, độ chính xác cao hơn đáng kể. Đây là mô hình tiêu chuẩn cho các ghi chú lâm sàng và hợp đồng pháp lý.

Khuyến nghị sản xuất tính đến năm 2026: bắt đầu với baseline LLM zero-shot trước khi bạn thu thập dữ liệu huấn luyện. Thông thường, F1 đủ tốt đến mức bạn không bao giờ cần phải fine-tune.

### Where classical NER still wins

Ngay cả khi có LLMs, NER cổ điển vẫn thắng khi:

- Ngân sách độ trễ (latency) dưới 50ms.
- Bạn có hàng nghìn ví dụ đã gán nhãn và cần F1 trên 98%.
- Lĩnh vực đó có một ontology ổn định nơi CRF hoặc BiLSTM được huấn luyện trước có thể chuyển đổi tốt.
- Các ràng buộc pháp lý yêu cầu một mô hình on-prem, không tạo sinh (non-generative).

### Where it falls apart

- **Domain shift.** NER được huấn luyện trên CoNLL áp dụng cho hợp đồng pháp lý hoạt động kém hơn cả từ điển. Hãy fine-tune trên lĩnh vực của bạn.
- **Nested entities.** "Bank of America Tower" đồng thời là một ORG và một FACILITY. BIO tiêu chuẩn không thể biểu diễn các đoạn chồng lấp. Bạn cần NER lồng nhau (mô hình đa lượt hoặc dựa trên span).
- **Long entities.** "United States Federal Deposit Insurance Corporation." Các mô hình cấp độ token đôi khi chia nhỏ thực thể này. Hãy sử dụng `aggregation_strategy` hoặc xử lý hậu kỳ.
- **Sparse types.** Các nhãn NER y tế như DRUG_BRAND, ADVERSE_EVENT, DOSE. Các mô hình đa năng không biết về chúng. Scispacy và BioBERT là các điểm khởi đầu ở đó.

## Ship It

Lưu dưới dạng `outputs/skill-ner-picker.md`:

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## Exercises

1. **Easy.** Triển khai `bio_to_spans` (nghịch đảo của `spans_to_bio`) và xác minh tính nhất quán khứ hồi trên 10 câu.
2. **Medium.** Huấn luyện CRF sklearn-crfsuite ở trên trên tập dữ liệu CoNLL-2003 English NER. Báo cáo F1 theo từng thực thể bằng cách sử dụng `seqeval`. Kết quả điển hình: ~84 F1.
3. **Hard.** Fine-tune `distilbert-base-cased` trên một tập dữ liệu NER chuyên ngành (y tế, pháp lý hoặc tài chính). So sánh với mô hình nhỏ của spaCy. Ghi lại các kiểm tra rò rỉ dữ liệu và viết ra những điều khiến bạn ngạc nhiên.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| NER | Extract names | Gán nhãn các đoạn token với các loại (PERSON, ORG, GPE, DATE, ...). |
| BIO | Tagging scheme | `B-X` bắt đầu, `I-X` tiếp tục, `O` bên ngoài. |
| BILOU | Better BIO | Thêm `L-X` (cuối), `U-X` (đơn vị) cho ranh giới sạch hơn. |
| CRF | Structured classifier | Mô hình hóa các chuyển đổi giữa các nhãn, không chỉ các phát xạ. Thực thi các chuỗi hợp lệ. |
| Nested NER | Overlapping entities | Một đoạn là một thực thể khác với một đoạn con của nó. BIO không thể biểu diễn điều này. |
| Entity-level F1 | Proper NER metric | Đoạn dự đoán phải khớp chính xác với đoạn thực tế. F1 cấp độ token làm quá mức độ chính xác. |

## Further Reading

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — bài báo về BiLSTM-CRF. Kinh điển.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — giới thiệu mô hình phân loại token đã trở thành tiêu chuẩn.
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — tài liệu tham khảo thực tế cho mọi thuộc tính trên `Doc.ents` và `Span`.
- [seqeval](https://github.com/chakki-works/seqeval) — thư viện đo lường chính xác. Luôn luôn sử dụng nó.