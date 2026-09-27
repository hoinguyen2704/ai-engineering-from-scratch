# POS Tagging và Syntactic Parsing

> Ngữ pháp đã từng lỗi thời một thời gian. Sau đó, mọi pipeline LLM đều cần xác thực việc trích xuất có cấu trúc, và nó đã quay trở lại.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 phút

## Vấn đề

Bài 01 đã hứa rằng lemmatization cần có part-of-speech tag. Nếu không biết `running` là một động từ, bộ lemmatizer không thể đưa nó về `run`. Nếu không biết `better` là một tính từ, nó không thể đưa về `good`.

Lời hứa đó che giấu cả một lĩnh vực con. Part-of-speech tagging gán các danh mục ngữ pháp. Syntactic parsing khôi phục cấu trúc cây của câu: từ nào bổ nghĩa cho từ nào, động từ nào chi phối đối số nào. NLP cổ điển đã dành hai mươi năm để tinh chỉnh cả hai. Sau đó, deep learning đã gộp chúng thành một tác vụ phân loại token (token-classification) trên nền tảng một transformer đã được huấn luyện trước, và cộng đồng nghiên cứu đã chuyển sang các hướng khác.

Nhưng cộng đồng ứng dụng thì không. Mọi pipeline trích xuất có cấu trúc vẫn sử dụng POS và dependency tree bên dưới. JSON do LLM tạo ra được xác thực dựa trên các ràng buộc ngữ pháp. Các hệ thống hỏi đáp phân tách truy vấn bằng cách sử dụng dependency parse. Các bộ đánh giá chất lượng dịch máy kiểm tra sự liên kết của các parse tree.

Rất đáng để tìm hiểu. Bài học này giới thiệu các bộ tag (tagsets), các baseline, và thời điểm bạn nên ngừng tự triển khai từ đầu mà hãy gọi spaCy.

## Khái niệm

**POS tagging** gán nhãn cho mỗi token bằng một danh mục ngữ pháp. Bộ tag **Penn Treebank (PTB)** là mặc định cho tiếng Anh. Gồm 36 tag với những phân biệt mà người đọc thông thường thấy khá rắc rối: `NN` danh từ số ít, `NNS` danh từ số nhiều, `NNP` danh từ riêng số ít, `VBD` động từ thì quá khứ, `VBZ` động từ ngôi thứ ba số ít thì hiện tại, v.v. Bộ tag **Universal Dependencies (UD)** thì thô hơn (17 tag) và không phụ thuộc vào ngôn ngữ; nó đã trở thành mặc định cho các công việc đa ngôn ngữ.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Syntactic parsing** tạo ra một cái cây. Hai phong cách chính:

- **Constituency parsing.** Các cụm danh từ, cụm động từ, cụm giới từ lồng vào nhau. Đầu ra là một cây các danh mục không kết thúc (NP, VP, PP) với các từ là các lá.
- **Dependency parsing.** Mỗi từ có một từ gốc (head) duy nhất mà nó phụ thuộc vào, được dán nhãn bằng một quan hệ ngữ pháp. Đầu ra là một cây trong đó mỗi cạnh là một bộ ba (head, dependent, relation).

Dependency parsing đã thắng thế trong những năm 2010 vì nó tổng quát hóa một cách sạch sẽ trên các ngôn ngữ, đặc biệt là các ngôn ngữ có trật tự từ tự do.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

```figure
pos-tagger
```

```figure
dependency-arcs
```

## Build It

### Bước 1: Baseline most-frequent-tag

Bộ POS tagger đơn giản nhất mà vẫn hoạt động. Với mỗi từ, hãy dự đoán tag mà nó xuất hiện thường xuyên nhất trong tập huấn luyện.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

Trên Brown corpus, baseline này đạt độ chính xác khoảng ~85%. Không tốt, nhưng là mức sàn mà không mô hình nghiêm túc nào được phép thấp hơn.

### Bước 2: Bigram HMM tagger

Mô hình hóa xác suất đồng thời của chuỗi:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

Hai bảng: xác suất chuyển tiếp (tag dựa trên tag trước đó), xác suất phát xạ (từ dựa trên tag). Ước tính cả hai từ số đếm với Laplace smoothing. Giải mã bằng Viterbi (quy hoạch động trên lưới tag).

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Bigram HMM trên Brown đạt độ chính xác ~93%. Bước nhảy từ 85% lên 93% chủ yếu nhờ xác suất chuyển tiếp — mô hình học được `DET NOUN` là phổ biến và `NOUN DET` là hiếm.

### Bước 3: Tại sao các tagger hiện đại lại vượt trội hơn

Xác suất chuyển tiếp + phát xạ mang tính cục bộ. Chúng không thể nắm bắt được rằng `saw` là danh từ trong "I bought a saw" nhưng là động từ trong "I saw the movie." Một CRF với các đặc trưng tùy ý (hậu tố, hình dạng từ, từ trước và sau, chính từ đó) đạt ~97%. Một BiLSTM-CRF hoặc transformer đạt ~98%+.

Trần của tác vụ này được thiết lập bởi sự bất đồng giữa những người gán nhãn. Con người đồng ý với nhau khoảng 97% thời gian trên Penn Treebank. Các mô hình vượt quá 98% có lẽ đang bị overfitting trên tập kiểm tra.

### Bước 4: Phác thảo dependency parsing

Dependency parsing đầy đủ từ đầu nằm ngoài phạm vi bài này; cách tiếp cận chuẩn trong giáo trình nằm ở Jurafsky và Martin. Hai họ cổ điển cần biết:

- **Transition-based** parsers (arc-eager, arc-standard) hoạt động giống như một shift-reduce parser: chúng đọc các token, đẩy chúng vào một stack, và áp dụng các hành động reduce để tạo ra các cung (arcs). Giải mã tham lam (greedy) rất nhanh. Triển khai cổ điển là MaltParser. Phiên bản neural hiện đại: Chen and Manning's transition-based parser.
- **Graph-based** parsers (thuật toán của Eisner, biaffine của Dozat-Manning) chấm điểm mọi cạnh head-dependent có thể và chọn cây bao trùm tối đa (maximum spanning tree). Chậm hơn nhưng chính xác hơn.

Đối với hầu hết các công việc ứng dụng, hãy gọi spaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

Đọc cột `dep` từ dưới lên trên và cấu trúc ngữ pháp của câu sẽ hiện ra.

## Sử dụng

Mọi thư viện NLP sản xuất đều cung cấp POS và dependency parser như một phần của pipeline tiêu chuẩn.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). Nhanh, chính xác, tích hợp với tokenization + NER + lemmatization. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (quan hệ phụ thuộc).
- **Stanford NLP (stanza)**. Người kế nhiệm của CoreNLP từ Stanford. Đạt trạng thái tốt nhất (state-of-the-art) trên hơn 60 ngôn ngữ.
- **trankit**. Dựa trên transformer, độ chính xác UD tốt.
- **NLTK**. `pos_tag`. Có thể sử dụng, chậm, cũ. Tốt cho việc giảng dạy.

### Nơi mà nó vẫn quan trọng vào năm 2026

- **Lemmatization.** Bài 01 cần POS để lemmatize chính xác. Luôn luôn.
- **Trích xuất có cấu trúc từ đầu ra của LLM.** Xác thực rằng một câu được tạo ra tuân thủ các ràng buộc ngữ pháp (ví dụ: sự hòa hợp chủ-vị, các bổ ngữ bắt buộc).
- **Sentiment dựa trên khía cạnh (Aspect-based sentiment).** Dependency parse cho bạn biết tính từ nào bổ nghĩa cho danh từ nào.
- **Hiểu truy vấn.** "movies directed by Wes Anderson starring Bill Murray" được phân tách thành các ràng buộc có cấu trúc thông qua parse.
- **Chuyển đổi đa ngôn ngữ (Cross-lingual transfer).** Các tag UD và quan hệ phụ thuộc không phụ thuộc vào ngôn ngữ, cho phép phân tích có cấu trúc zero-shot cho các ngôn ngữ mới.
- **Pipeline yêu cầu tính toán thấp.** Nếu bạn không thể triển khai một transformer, POS + dependency parse + gazetteer sẽ giúp bạn đi được một chặng đường đáng ngạc nhiên.

## Ship It

Lưu dưới dạng `outputs/skill-grammar-pipeline.md`:

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## Bài tập

1. **Dễ.** Sử dụng baseline most-frequent-tag trên một tập dữ liệu nhỏ đã được gán nhãn (ví dụ: tập con Brown của NLTK), đo độ chính xác trên các câu giữ lại (held-out). Xác minh kết quả ~85%.
2. **Trung bình.** Huấn luyện bigram HMM ở trên và báo cáo độ chính xác/thu hồi (precision/recall) cho từng tag. HMM nhầm lẫn các tag nào nhiều nhất?
3. **Khó.** Sử dụng dependency parse của spaCy để trích xuất các bộ ba chủ-vị-tân (subject-verb-object) từ mẫu 1000 câu. Đánh giá trên 50 bộ ba được gán nhãn thủ công. Ghi lại những nơi trích xuất thất bại (thường là câu bị động, các cấu trúc phối hợp, và chủ ngữ bị lược bỏ).

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| POS tag | Loại của từ | Danh mục ngữ pháp. PTB có 36; UD có 17. |
| Penn Treebank | Bộ tag tiêu chuẩn | Đặc thù tiếng Anh. Các thì động từ và số lượng danh từ chi tiết. |
| Universal Dependencies | Bộ tag đa ngôn ngữ | Thô hơn PTB; trung lập về ngôn ngữ; mặc định cho công việc đa ngôn ngữ. |
| Dependency parse | Cây câu | Mỗi từ có một gốc, mỗi cạnh có một quan hệ ngữ pháp. |
| Viterbi | Quy hoạch động | Tìm chuỗi tag có xác suất cao nhất dựa trên phát xạ và chuyển tiếp. |

## Đọc thêm

- [Jurafsky and Martin — Speech and Language Processing, chương 8 và 18](https://web.stanford.edu/~jurafsky/slp3/) — giáo trình chuẩn về POS và parsing.
- [Dự án Universal Dependencies](https://universaldependencies.org/) — bộ sưu tập tagset và treebank đa ngôn ngữ được mọi parser đa ngôn ngữ sử dụng.
- [Hướng dẫn về các đặc trưng ngôn ngữ của spaCy](https://spacy.io/usage/linguistic-features) — tài liệu tham khảo thực tế cho mọi thuộc tính được hiển thị trên `Token`.
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — bài báo đưa các neural parser vào dòng chính.